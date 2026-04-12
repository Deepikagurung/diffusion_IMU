"""
IMU Baseline Pipeline
=====================
1. MobilePoser predicts pose/tran from real IMU input
2. Differentiable IMU synthesis converts predicted pose/tran back to simulated IMU
3. Compare real IMU vs simulated IMU (RMSE)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

from tqdm import tqdm
from config import *
from helpers import *
import articulate as art
from utils.model_utils import load_model
from data import PoseDataset
from articulate.model import ParametricModel
from config import paths, datasets

# ── constants ──
TARGET_FPS = 30
# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])


# =====================================================================
# Differentiable IMU synthesis (pure PyTorch — backprop-friendly)
# =====================================================================

def syn_acc(v, smooth_n=4, fps=30):
    """Synthesize accelerations from vertex positions. Differentiable.

    Args:
        v: (N, num_verts, 3) vertex positions over time
    Returns:
        acc: (N, num_verts, 3)
    """
    scale = fps ** 2
    mid = smooth_n // 2

    if v.shape[0] < 3:
        return torch.zeros_like(v)

    acc = torch.stack(
        [(v[i] + v[i + 2] - 2 * v[i + 1]) * scale for i in range(v.shape[0] - 2)]
    )
    acc = torch.cat([torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])])

    # Only smooth if sequence is long enough
    if mid != 0 and v.shape[0] > smooth_n * 2:
        smooth_acc = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale / smooth_n ** 2
             for i in range(v.shape[0] - smooth_n * 2)]
        )
        acc[smooth_n:-smooth_n] = smooth_acc

    return acc


def pose_to_imu(pose_rotmat, trans, body_model, fps=30):
    """Differentiable: pose/tran → MobilePoser 60-dim input tensor.

    This mirrors exactly how process.py creates training data:
        acc = _syn_acc(vert[:, vi_mask]) / acc_scale   →  (N, 5, 3)
        rot = grot[:, ji_mask][:, :5]                  →  (N, 5, 3, 3)
        imu = cat(acc.flatten(1), rot.flatten(1))      →  (N, 60)

    All operations are PyTorch — gradients flow through.

    Args:
        pose_rotmat: (N, 24, 3, 3) local joint rotations
        trans:       (N, 3)        root translation
        body_model:  ParametricModel on CPU
    Returns:
        imu_input: (N, 60) tensor on same device as input
    """
    device = pose_rotmat.device
    pose_cpu = pose_rotmat.cpu()
    trans_cpu = trans.cpu()

    # forward kinematics → global rotations + vertices (differentiable)
    grot, _, verts = body_model.forward_kinematics(
        pose_cpu, tran=trans_cpu, calc_mesh=True
    )  # grot: (N, 24, 3, 3), verts: (N, 6890, 3)

    # acceleration at sensor vertices (same as process.py)
    sensor_verts = verts[:, vi_mask]           # (N, 6, 3)
    vacc = syn_acc(sensor_verts, fps=fps)      # (N, 6, 3)

    # global rotations at sensor joints
    vrot = grot[:, ji_mask]                    # (N, 6, 3, 3)

    # select first 5 sensors (drop pelvis) + scale acceleration
    acc = vacc[:, :5] / amass.acc_scale        # (N, 5, 3)
    rot = vrot[:, :5]                          # (N, 5, 3, 3)

    # flatten and concatenate → (N, 60)
    imu_input = torch.cat([acc.flatten(1), rot.flatten(1)], dim=1)

    return imu_input.to(device)


class SimulatedIMU(nn.Module):
    """Pose → physically simulated IMU → learnable correction → corrected IMU.

    Combines two stages in one forward pass:

    Stage 1 — Physical synthesis (deterministic, no learned params):
        pose_rotmat, trans
            → forward kinematics (SMPL body model)
            → finite-difference acceleration + gravity + local-frame transform
            → (N, 60) raw physical IMU

    Stage 2 — Learnable correction (trained to close the sim-to-real gap):
        (N, 60) raw physical IMU
            → per-sensor affine on acceleration  (scale + bias)
            → per-sensor SO(3) mounting offset   (rotation frame alignment)
            → residual MLP                       (non-parametric remainder)
            → (N, 60) corrected IMU

    All three correction parameters start at identity, so before any
    training the output equals the raw physical simulation.

    Usage:
        sim = SimulatedIMU(body_model, hidden_dim=128).to(device)
        corrected_imu = sim(pose_rotmat, trans, fps=30)   # (N, 60)
    """

    GRAVITY   = torch.tensor([0.0, -9.81, 0.0])
    N_SENSORS = 5
    ACC_DIM   = N_SENSORS * 3     # 15
    ROT_DIM   = N_SENSORS * 9     # 45
    TOTAL     = ACC_DIM + ROT_DIM # 60

    def __init__(self, body_model, hidden_dim=128):
        super().__init__()
        self.body_model = body_model
        n = self.N_SENSORS

        # --- Learnable physical synthesis parameters (Stage 1)
        # learnable gravity vector (starts as physical gravity)
        self.learn_gravity = nn.Parameter(self.GRAVITY.clone())

        # per-sensor learned acceleration log-scale applied in physical synthesis
        self.phys_acc_log_scale = nn.Parameter(torch.zeros(n, 3))

        # small per-vertex offsets for the 6 sensor vertices (N_sensors=6 in verts selection)
        # this lets the synthesis slightly adjust sensor positions (starts at zero)
        self.sensor_vert_offsets = nn.Parameter(torch.zeros(6, 3))

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _6d_to_rotmat(r6d):
        """(*, 6) → (*, 3, 3) via Gram-Schmidt."""
        r6d = r6d.reshape(*r6d.shape[:-1], 3, 2)
        a1 = r6d[..., 0];  a2 = r6d[..., 1]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    def _physical_synthesis(self, pose_rotmat, trans, fps):
        """Stage 1: (N,24,3,3) + (N,3) → (N,60) raw physical IMU (no grad through body model)."""
        device = pose_rotmat.device
        grot, _, verts = self.body_model.forward_kinematics(
            pose_rotmat.cpu(), tran=trans.cpu(), calc_mesh=True
        )
        sensor_verts = verts[:, vi_mask]                          # (N, 6, 3)
        # apply small learned offsets to sensor vertex positions (keeps initial behaviour)
        sensor_verts = sensor_verts + self.sensor_vert_offsets.to(sensor_verts.device).unsqueeze(0)
        vrot         = grot[:, ji_mask]                           # (N, 6, 3, 3)

        vacc_world   = syn_acc(sensor_verts, fps=fps)             # (N, 6, 3)
        # use learnable gravity (starts as nominal gravity)
        g            = self.learn_gravity.to(vacc_world.device)
        vacc_specific = vacc_world + g.unsqueeze(0).unsqueeze(0)  # (N, 6, 3)

        # rotate into local sensor frame
        vacc_local = torch.einsum(
            'nijk,nik->nij', vrot.transpose(-1, -2), vacc_specific
        )                                                          # (N, 6, 3)

        acc = vacc_local[:, :5] / amass.acc_scale                 # (N, 5, 3)
        # apply small learned per-sensor acceleration scale (starts at 1)
        acc = acc * torch.exp(self.phys_acc_log_scale.to(acc.device)).unsqueeze(0)
        rot = vrot[:, :5]                                         # (N, 5, 3, 3)

        raw = torch.cat([acc.flatten(1), rot.flatten(1)], dim=1)  # (N, 60)
        return raw.to(device)

    # NOTE: The learnable correction stage has been removed. The module now
    # exposes only the learnable physical synthesis parameters. `_physical_synthesis`
    # returns the IMU-like raw tensor directly.

    # ── public forward ─────────────────────────────────────────────────

    def forward(self, pose_rotmat, trans, fps=TARGET_FPS):
        """
        Args:
            pose_rotmat: (N, 24, 3, 3) local joint rotations
            trans:       (N, 3)        root translation
            fps:         frame rate (default TARGET_FPS)
        Returns:
            corrected_imu: (N, 60)
        """
        raw = self._physical_synthesis(pose_rotmat, trans, fps)
        return raw


def train_imu_correction(
    sim_imu_module, mp_model, dataset, device,
    num_epochs=20, lr=1e-3, save_path="imu_correction.pth",
):
    """Train SimulatedIMU's learnable correction to minimise MSE(corrected, real_imu).

    MobilePoser is frozen. Pipeline per sequence:
        real_imu → MobilePoser (frozen) → pose_pred, tran_pred
                 → SimulatedIMU (physical synthesis + learnable correction)
                 → corrected_imu
        Loss = MSE(corrected_imu, real_imu)
    """
    mp_model.eval()
    for p in mp_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(sim_imu_module.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        sim_imu_module.train()
        total_loss = 0.0
        n_seqs = 0

        for imu, *_ in tqdm(dataset, desc=f"[SimIMU] Epoch {epoch+1}/{num_epochs}"):
            real_imu = imu.to(device)

            # frozen MobilePoser: real IMU → pose/tran
            with torch.no_grad():
                mp_model.reset()
                pose_pred, _, tran_pred, _ = mp_model.forward_offline(
                    real_imu.unsqueeze(0), [real_imu.shape[0]]
                )

            # SimulatedIMU: physical synthesis + learnable correction
            corrected = sim_imu_module(pose_pred, tran_pred)
            loss = criterion(corrected, real_imu)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sim_imu_module.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_seqs += 1

        avg = total_loss / max(n_seqs, 1)
        print(f"  [SimIMU] Epoch {epoch+1}: avg loss = {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(sim_imu_module.state_dict(), save_path)
            print(f"  → saved best SimulatedIMU ({save_path})")

    return sim_imu_module


def train_simulated_imu(
    sim_imu_module, mp_model, dataset, device,
    num_epochs=20, lr=1e-3, save_path="simulated_imu.pth",
):
    """Train the learnable simulated IMU physical parameters to minimise
    MSE(simulated_imu, real_imu).

    This mirrors the previous `train_imu_correction` behaviour but the name
    matches the new role: training the simulated (learnable physical) IMU.
    """
    return train_imu_correction(sim_imu_module, mp_model, dataset, device,
                                num_epochs=num_epochs, lr=lr, save_path=save_path)


# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MobilePoser IMU baseline + learnable correction")
    parser.add_argument("--mobileposer",   type=str, default=r"deepika\model_finetuned.pth")
    parser.add_argument("--dataset",       type=str, default="dip")
    parser.add_argument("--train-simulated", action="store_true",
                        help="Train learnable simulated-IMU physical params before evaluation")
    parser.add_argument("--sim-checkpoint", type=str, default="simulated_imu.pth",
                        help="Path to save/load the simulated IMU checkpoint")
    parser.add_argument("--sim-epochs",   type=int,   default=100)
    parser.add_argument("--sim-lr",       type=float, default=1e-3)
    parser.add_argument("--sim-hidden",   type=int,   default=128)
    args = parser.parse_args()

    device = model_config.device
    body_model = ParametricModel(paths.smpl_file)
    mp_model = load_model(args.mobileposer)
    mp_model.eval()
    mp_model.to(device)

    if args.dataset not in datasets.test_datasets:
        raise ValueError(f"Dataset '{args.dataset}' not found in test_datasets.")

    # ---- build unified SimulatedIMU module ----
    sim_imu_module = SimulatedIMU(body_model, hidden_dim=args.sim_hidden).to(device)

    if args.train_simulated:
        print("\n=== Training SimulatedIMU (learnable physical params) ===")
        train_dataset = PoseDataset(fold='train', finetune='dip')
        train_simulated_imu(
            sim_imu_module, mp_model, train_dataset, device,
            num_epochs=args.sim_epochs, lr=args.sim_lr,
            save_path=args.sim_checkpoint,
        )

    if os.path.exists(args.sim_checkpoint):
        sim_imu_module.load_state_dict(
            torch.load(args.sim_checkpoint, map_location=device)
        )
        print(f"Loaded SimulatedIMU from {args.sim_checkpoint}")
    sim_imu_module.eval()

    # ---- evaluation ----
    eval_dataset = PoseDataset(fold='test', evaluate=args.dataset)

    total_rmse_mp   = 0.0   # MobilePoser-style (world frame, no gravity)
    total_rmse_corr = 0.0   # SimulatedIMU: physical + learned correction
    n_seqs = 0

    with torch.no_grad():
        for i, (imu, pose_gt, joint_gt, tran_gt) in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            real_imu = imu.to(device)  # (N, 60)

            # Step 1: MobilePoser real IMU → pose/tran
            mp_model.reset()
            pose_pred, _, tran_pred, _ = mp_model.forward_offline(
                real_imu.unsqueeze(0), [real_imu.shape[0]]
            )

            # Step 2a: MobilePoser-style baseline (world frame, no gravity)
            sim_imu_mp = pose_to_imu(pose_pred, tran_pred, body_model, fps=TARGET_FPS)

            # Step 2b: SimulatedIMU (physical synthesis + learned correction)
            sim_imu_corr = sim_imu_module(pose_pred, tran_pred)

            # Step 3: RMSE vs real IMU
            rmse_mp   = (sim_imu_mp.cpu()   - imu).pow(2).mean().sqrt().item()
            rmse_corr = (sim_imu_corr.cpu() - imu).pow(2).mean().sqrt().item()

            print(f"Seq {i:3d}:  mp-baseline={rmse_mp:.6f}  SimulatedIMU={rmse_corr:.6f}")

            total_rmse_mp   += rmse_mp
            total_rmse_corr += rmse_corr
            n_seqs += 1

    print(f"\nMean RMSE over {n_seqs} sequences:")
    print(f"  MobilePoser-style (world frame, no gravity): {total_rmse_mp   / n_seqs:.6f}")
    print(f"  SimulatedIMU      (physical + learned):      {total_rmse_corr / n_seqs:.6f}")

