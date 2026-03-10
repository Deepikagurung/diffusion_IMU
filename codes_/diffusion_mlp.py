import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from typing import List
import random
import lightning as L
from tqdm import tqdm 

import articulate as art
from config import *
from utils import *
from helpers import *
from typing import Sequence

from utils.model_utils import reduced_pose_to_full
from articulate.math import r6d_to_rotation_matrix
from articulate.evaluator import MeanPerJointErrorEvaluator
from articulate.math import RotationRepresentation


##functions


class PoseDataset(Dataset):
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos.items())
        self.data = self._prepare_dataset()

    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            return self._get_train_files(data_folder)
        elif self.fold == 'test':
            return self._get_test_files()
        else:
            raise ValueError(f"Unknown data fold: {self.fold}.")

    def _get_train_files(self, data_folder):
        if self.finetune:
            return [datasets.finetune_datasets[self.finetune]]
        else:
            return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _get_test_files(self):
        return [datasets.test_datasets[self.evaluate]]

    def _prepare_dataset(self):
        data_folder = paths.processed_datasets / ('eval' if (self.finetune or self.evaluate) else '')
        data_files = self._get_data_files(data_folder)

        print(f"\n{'='*60}")
        print(f"Loading datasets for {self.fold.upper()} mode")
        print(f"Datasets: {data_files}")
        print(f"{'='*60}\n")

        data = {key: [] for key in [
            'imu_inputs', 'pose_outputs', 'joint_outputs',
            'tran_outputs', 'vel_outputs', 'foot_outputs'
        ]}

        for data_file in tqdm(data_files):
            try:
                file_data = torch.load(data_folder / data_file)
                self._process_file_data(file_data, data)
            except Exception as e:
                print(f"Error processing {data_file}: {e}")

        print("\nTotal pose windows stored:", len(data['pose_outputs']))
        return data

    def _process_file_data(self, file_data, data):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))

        for acc, ori, pose, tran, joint, foot in zip(accs, oris, poses, trans, joints, foots):


            acc, ori = acc[:, :5]/amass.acc_scale, ori[:, :5]

            pose_global, joint = self.bodymodel.forward_kinematics(
                pose=pose.view(-1, 216)
            )

            pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)
            joint = joint.view(-1, 24, 3)

            print("Pose after forward kinematics:", pose.shape)

            self._process_combo_data(acc, ori, pose, joint, tran, foot, data)

    def _process_combo_data(self, acc, ori, pose, joint, tran, foot, data):

        for combo_name, c in self.combos:

            print("\n" + "="*50)
            print("Processing combo:", combo_name)

            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, c] = acc[:, c]
            combo_ori[:, c] = ori[:, c]

            imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)

            data_len = len(imu_input) if self.evaluate else datasets.window_length

            for key, value in zip(
                ['imu_inputs', 'pose_outputs', 'joint_outputs', 'tran_outputs'],
                [imu_input, pose, joint, tran]
            ):
                # data[key].extend(torch.split(value, data_len))
                splits = torch.split(value, data_len)

                # remove last if smaller than full window
                if splits[-1].shape[0] < data_len:
                    splits = splits[:-1]

                data[key].extend(splits)

            if not (self.evaluate or self.finetune):
                self._process_translation_data(joint, tran, foot, data_len, data)

    def _process_translation_data(self, joint, tran, foot, data_len, data):


        root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
        vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint, dim=0)))
        vel[:, 0] = root_vel

        vel = vel * (datasets.fps / amass.vel_scale)
        vel_splits = torch.split(vel, data_len)

        data['vel_outputs'].extend(vel_splits)
        data['foot_outputs'].extend(torch.split(foot, data_len))

    def __getitem__(self, idx):

        imu = self.data['imu_inputs'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()

        num_pred_joints = len(amass.pred_joints_set)

        pose = art.math.rotation_matrix_to_r6d(
            self.data['pose_outputs'][idx]
        ).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set] \
         .reshape(-1, 6*num_pred_joints)

        if self.evaluate or self.finetune:
            return imu, pose, joint, tran

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()

        return imu, pose, joint, tran, vel, contact

    def __len__(self):
        return len(self.data['imu_inputs'])
    




def pad_seq(batch):
    """Pad sequences to same length for RNN."""
    def _pad(sequence):
        padded = nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = [seq.shape[0] for seq in sequence]
        return padded, lengths

    inputs, poses, joints, trans = zip(*[(item[0], item[1], item[2], item[3]) for item in batch])
    inputs, input_lengths = _pad(inputs)
    poses, pose_lengths = _pad(poses)
    joints, joint_lengths = _pad(joints)
    trans, tran_lengths = _pad(trans)
    
    outputs = {'poses': poses, 'joints': joints, 'trans': trans}
    output_lengths = {'poses': pose_lengths, 'joints': joint_lengths, 'trans': tran_lengths}

    if len(batch[0]) > 5: # include velocity and foot contact, if available
        vels, foots = zip(*[(item[4], item[5]) for item in batch])

        # foot contact 
        foot_contacts, foot_contact_lengths = _pad(foots)
        outputs['foot_contacts'], output_lengths['foot_contacts'] = foot_contacts, foot_contact_lengths

        # root velocities
        vels, vel_lengths = _pad(vels)
        outputs['vels'], output_lengths['vels'] = vels, vel_lengths

    return (inputs, input_lengths), (outputs, output_lengths)


class PoseDataModule(L.LightningDataModule):
    def __init__(self, finetune: str = None):
        super().__init__()
        self.finetune = finetune
        self.hypers = finetune_hypers if self.finetune else train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = PoseDataset(fold='train', finetune=self.finetune)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_dataset = PoseDataset(fold='test', finetune=self.finetune)

    def _dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.hypers.batch_size, 
            collate_fn=pad_seq, 
            num_workers=0, #self.hypers.num_workers, 
            shuffle=True, 
            drop_last=True
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

"""Diff-UNet style diffusion model with temporal transformer backbone.

Follows the same DDPM framework as Diff-UNet (ge-xing/Diff-UNet):
  - x₀-prediction (ModelMeanType.START_X)
  - Sinusoidal timestep embedding → MLP  (like temb.dense in BasicUNetDe)
  - IMU conditioning encoder              (like BasicUNetEncoder)
  - DDIM reverse sampling loop            (like ddim_sample_loop)
Backbone: temporal transformer instead of 3-D UNet (better for pose sequences).
"""

import torch.nn.functional as F_func


# ---------------- Diffusion hyperparameters (same as Diff-UNet) ---------------
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


# ---------------- Sinusoidal timestep embedding (Diff-UNet style) -------------

def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embedding — identical to Diff-UNet / DDPM.

    timesteps:     (B,)  integer diffusion timesteps
    embedding_dim: int
    Returns:       (B, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F_func.pad(emb, (0, 1, 0, 0))
    return emb


# ---------------- Sequence positional encoding --------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding over the temporal (sequence) dimension."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------------- IMU Conditioning Encoder (≈ BasicUNetEncoder) ---------------

class IMUEncoder(nn.Module):
    """Encodes IMU input into conditioning features.

    Analogous to BasicUNetEncoder in Diff-UNet which extracts
    multi-scale features from the input image for conditioning.
    """

    def __init__(self, imu_dim: int, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(imu_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        """imu: (B, T, imu_dim) → (B, T, d_model)"""
        return self.encoder(imu)


# ---------------- Transformer Denoiser (≈ BasicUNetDe) ------------------------

class TemporalTransformerDenoiser(nn.Module):
    """Predicts x₀ (clean pose) from noised x_t.

    Analogous to BasicUNetDe in Diff-UNet:
      - Conditioned on timestep t via sinusoidal embedding + MLP
      - Conditioned on IMU features (like encoder embeddings)
      - Predicts x₀ directly  (ModelMeanType.START_X)
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
    ):
        super().__init__()

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Diff-UNet style: sinusoidal → MLP  (like temb.dense)
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                conditioning: torch.Tensor = None) -> torch.Tensor:
        """Predict clean pose x₀.

        x:            (B, T, F)        noised pose x_t
        t:            (B,)             integer timesteps [0, TIMESTEPS)
        conditioning: (B, T, d_model)  IMU encoder output (optional)
        Returns:      (B, T, F)        predicted x₀
        """
        h = self.in_proj(x)                                       # (B, T, d_model)
        h = self.pos_enc(h)

        # Diff-UNet style timestep conditioning
        t_emb = get_timestep_embedding(t, self.time_embed_dim)    # (B, embed_dim)
        t_emb = self.time_mlp(t_emb)                              # (B, d_model)
        h = h + t_emb.unsqueeze(1)                                # broadcast over T

        # Inject IMU conditioning (like Diff-UNet adds encoder embeddings)
        if conditioning is not None:
            h = h + conditioning

        h = self.encoder(h)
        return self.out_proj(h)                                   # predicted x₀


# ---------------- DDIM Sampling (≈ Diff-UNet ddim_sample_loop) ----------------

@torch.no_grad()
def ddim_sample(model, shape, conditioning, num_steps=50, device="cuda"):
    """DDIM deterministic sampling loop.

    Starts from pure Gaussian noise and iteratively denoises → clean x₀.

    Args:
        model:        TemporalTransformerDenoiser (predicts x₀)
        shape:        (B, T, F)
        conditioning: (B, T, d_model) from IMU encoder
        num_steps:    DDIM steps (Diff-UNet uses 10-50)
    Returns:          (B, T, F) denoised pose
    """
    ab = alpha_bar.to(device)

    # Sub-sample timesteps evenly (high → low)
    step_indices = torch.linspace(
        TIMESTEPS - 1, 0, num_steps + 1, dtype=torch.long, device=device
    )

    x_t = torch.randn(shape, device=device)

    for i in range(num_steps):
        t_cur  = step_indices[i]
        t_next = step_indices[i + 1]

        t_batch = t_cur.expand(shape[0])

        # Model predicts clean x₀
        x0_pred = model(x_t, t_batch, conditioning=conditioning)

        if t_next > 0:
            ab_cur  = ab[t_cur]
            ab_next = ab[t_next]

            # Implied noise from x₀ prediction
            pred_noise = (
                (x_t - torch.sqrt(ab_cur) * x0_pred)
                / torch.sqrt(1.0 - ab_cur)
            )

            # DDIM deterministic step (η = 0)
            x_t = (
                torch.sqrt(ab_next) * x0_pred
                + torch.sqrt(1.0 - ab_next) * pred_noise
            )
        else:
            x_t = x0_pred

    return x_t


# ---------------- Training (Diff-UNet style: predict x₀) ---------------------


def training(train_loader, model, imu_encoder, num_epochs=1, device=None,
             patience: int = 10, min_delta: float = 1e-4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    imu_encoder = imu_encoder.to(device)

    # Optimise both encoder and denoiser jointly
    params = list(model.parameters()) + list(imu_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-4)

    criterion = nn.MSELoss(reduction="none")

    best_epoch_loss = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        model.train()
        imu_encoder.train()
        epoch_weighted_loss = 0.0
        total_frames = 0

        for batch in train_loader:

            (inputs, input_lengths), (outputs, output_lengths) = batch

            # --- Pose target x₀ ---
            x = outputs["poses"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)
            B, T, F = x.shape

            # --- IMU conditioning (like Diff-UNet encoder embeddings) ---
            imu = inputs.to(device)
            conditioning = imu_encoder(imu[:, :T])

            # --- q_sample: forward diffusion (same as Diff-UNet) ---
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            ab = alpha_bar.to(device)[t].view(B, 1, 1)

            noise = torch.randn_like(x)
            x_t = torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * noise

            # --- Denoise: predict x₀ (Diff-UNet ModelMeanType.START_X) ---
            x0_pred = model(x_t, t, conditioning=conditioning)

            # --- Loss: MSE(x₀_pred, x₀) with padding mask ---
            loss_matrix = criterion(x0_pred, x)
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()

            masked_loss = (loss_matrix * mask).sum()
            num_valid = mask.sum() * F

            if num_valid > 0:
                loss = masked_loss / num_valid
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # --- Backprop ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            print(f"Epoch {epoch} | Batch Loss: {loss.item():.6f}")

            epoch_weighted_loss += masked_loss.item()
            total_frames += num_valid.item()

        # --- Epoch Summary ---
        epoch_loss = epoch_weighted_loss / total_frames if total_frames > 0 else 0.0
        print(f"=== Epoch {epoch} | Loss: {epoch_loss:.6f} ===")

        # --- Early Stopping ---
        if best_epoch_loss is None or epoch_loss < best_epoch_loss - min_delta:
            best_epoch_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs "
                  f"(best: {best_epoch_loss:.6f})")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    # Save both encoder and denoiser
    torch.save({
        'denoiser': model.state_dict(),
        'imu_encoder': imu_encoder.state_dict(),
    }, "pose_diffusion_transformer.pth")
    print("Model saved.")


# ---------------- Validation with DDIM sampling (Diff-UNet style) -------------


def validate(val_loader, model, imu_encoder, device, mpj_eval, ddim_steps=50):
    from utils.model_utils import reduced_pose_to_full
    from articulate.math import r6d_to_rotation_matrix

    model.eval()
    imu_encoder.eval()

    val_weighted_loss = 0.0
    val_mpj_sum = 0.0
    val_total_frames = 0

    with torch.no_grad():
        for batch in val_loader:

            (inputs, input_lengths), (outputs, output_lengths) = batch

            x = outputs["poses"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)
            B, T, Fdim = x.shape

            # --- IMU conditioning ---
            imu = inputs.to(device)
            conditioning = imu_encoder(imu[:, :T])

            # --- DDIM sampling (Diff-UNet style inference) ---
            x0_hat = ddim_sample(
                model, shape=(B, T, Fdim),
                conditioning=conditioning,
                num_steps=ddim_steps, device=device,
            )

            # --- Reconstruction loss ---
            loss_matrix = F_func.mse_loss(x0_hat, x, reduction="none")
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            masked_loss = (loss_matrix * mask).sum()
            num_valid = mask.sum() * Fdim

            # --- MPJPE ---
            if num_valid > 0:
                in_rep = 6 if Fdim % 6 == 0 else 9
                num_joints = Fdim // in_rep

                pose_pred = x0_hat.view(B * T, num_joints, in_rep)
                pose_true = x.view(B * T, num_joints, in_rep)
                valid_idx = mask.squeeze(-1).reshape(-1).bool()

                if valid_idx.any():
                    if in_rep == 6:
                        pose_pred_mat = r6d_to_rotation_matrix(pose_pred)
                        pose_true_mat = r6d_to_rotation_matrix(pose_true)
                    else:
                        pose_pred_mat = pose_pred.view(-1, num_joints, 3, 3)
                        pose_true_mat = pose_true.view(-1, num_joints, 3, 3)

                    pose_pred_mat = pose_pred_mat[valid_idx]
                    pose_true_mat = pose_true_mat[valid_idx]

                    try:
                        j, _ = mpj_eval.model.get_zero_pose_joint_and_vertex(None)
                        model_num_j = j.shape[1] if j.dim() == 3 else j.shape[0]
                    except Exception:
                        model_num_j = pose_pred_mat.shape[1]

                    if pose_pred_mat.shape[1] != model_num_j:
                        N_valid = pose_pred_mat.shape[0]
                        pose_pred_mat = reduced_pose_to_full(
                            pose_pred_mat.unsqueeze(1)
                        ).view(N_valid, model_num_j, 3, 3)
                        pose_true_mat = reduced_pose_to_full(
                            pose_true_mat.unsqueeze(1)
                        ).view(N_valid, model_num_j, 3, 3)

                    mpj_tensor = mpj_eval(pose_pred_mat, pose_true_mat)
                    val_mpj_sum += mpj_tensor[0].item() * valid_idx.sum().item()

            val_weighted_loss += masked_loss.item()
            val_total_frames += num_valid.item()

    val_loss = val_weighted_loss / val_total_frames if val_total_frames > 0 else 0.0
    val_mpj  = val_mpj_sum / val_total_frames if val_total_frames > 0 else 0.0

    print(f"Validation Loss: {val_loss:.6f} | MPJPE: {val_mpj:.6f}")
    return val_loss, val_mpj


# ========================= MAIN =============================================

datamodule = PoseDataModule(finetune=None)
datamodule.setup(stage='fit')

train_loader = datamodule.train_dataloader()
val_loader   = datamodule.val_dataloader()

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- IMU Encoder (like Diff-UNet's BasicUNetEncoder) ---
imu_encoder = IMUEncoder(imu_dim=60, d_model=128).to(device)

# --- Temporal Transformer Denoiser (like Diff-UNet's BasicUNetDe) ---
model = TemporalTransformerDenoiser(
    feature_dim=144,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    time_embed_dim=128,
).to(device)

# --- Train (Diff-UNet style: predict x₀, MSE loss) ---
training(
    train_loader=train_loader, model=model, imu_encoder=imu_encoder,
    num_epochs=100, patience=10,
)

# --- Validate (Diff-UNet style: DDIM sampling) ---
mpj_eval = MeanPerJointErrorEvaluator(
    official_model_file=paths.smpl_file,
    rep=RotationRepresentation.ROTATION_MATRIX,
    device=device,
)

val_loss, val_mpj = validate(
    val_loader=val_loader, model=model, imu_encoder=imu_encoder,
    device=device, mpj_eval=mpj_eval, ddim_steps=50,
)

# ========================= VISUALIZATION =====================================

from articulate.math import r6d_to_rotation_matrix
from viewers.smpl_viewer import SMPLViewer

model.eval()
imu_encoder.eval()

k = 10  # pick the k-th batch (0-indexed)

for i, batch in enumerate(val_loader):
    if i == k:
        break

(inputs, input_lengths), (outputs, output_lengths) = batch

x = outputs["poses"].to(device)
trans = outputs["trans"].to(device)
imu = inputs.to(device)

B, T, Fdim = x.shape
lengths = output_lengths["poses"]

# --- Get IMU conditioning ---
conditioning = imu_encoder(imu[:, :T])

# --- Generate predictions via DDIM sampling (Diff-UNet style) ---
with torch.no_grad():
    pred = ddim_sample(
        model, shape=(B, T, Fdim),
        conditioning=conditioning,
        num_steps=50, device=device,
    )

# --- Visualize ---
viewer = SMPLViewer(fps=25)
num_to_show = min(10, B)

for b in range(num_to_show):
    L = int(lengths[b])

    gt_pose_r6d   = x[b, :L]       # (L, 144)
    pred_pose_r6d = pred[b, :L]    # (L, 144)
    tran_gt       = trans[b, :L]   # (L, 3)
    tran_pred     = tran_gt        # model predicts pose, not translation

    gt_pose_rot = r6d_to_rotation_matrix(
        gt_pose_r6d.view(-1, 24, 6)
    ).view(-1, 24, 3, 3)

    pred_pose_rot = r6d_to_rotation_matrix(
        pred_pose_r6d.view(-1, 24, 6)
    ).view(-1, 24, 3, 3)

    print(f"Visualizing sequence {b+1}/{num_to_show}, length = {L}")
    viewer.view(pred_pose_rot, tran_pred, gt_pose_rot, tran_gt, with_tran=True)