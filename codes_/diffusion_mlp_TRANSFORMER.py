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

"""Diffusion model and transformer definitions"""


# ---------------- Diffusion hyperparameters ----------------
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


class SinusoidalPositionalEncoding(nn.Module):
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
        """Add positional encoding to input.

        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TemporalTransformerDenoiser(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """Temporal Transformer denoiser.

        Inputs / outputs: (B, T, feature_dim)
        """
        super().__init__()

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # simple time embedding to condition on diffusion timestep
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
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
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict diffusion noise.

        x: (B, T, F)  -> noisy pose x_t
        t: (B,)       -> integer timesteps in [0, TIMESTEPS)
        """
        B = x.size(0)

        h = self.in_proj(x)          # (B, T, d_model)
        h = self.pos_enc(h)

        # normalize timestep to [0, 1] and embed
        t_norm = t.float().unsqueeze(1) / max(TIMESTEPS - 1, 1)
        t_emb = self.time_mlp(t_norm).unsqueeze(1)  # (B, 1, d_model)

        # broadcast time embedding over sequence length
        h = h + t_emb

        h = self.encoder(h)  # (B, T, d_model)
        return self.out_proj(h)  # (B, T, F), predicted noise
    

# ---------------- Diffusion training code ----------------


def training(train_loader, model, num_epochs=1, device=None, patience: int = 10, min_delta: float = 1e-4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss(reduction="none")  # elementwise for masking

    best_epoch_loss = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        model.train()
        epoch_weighted_loss = 0.0
        total_frames = 0

        for batch in train_loader:

            (inputs, input_lengths), (outputs, output_lengths) = batch

            # -------------------------------------------------
            # Get Pose Data
            # -------------------------------------------------
            x = outputs["poses"].to(device)
            lengths = torch.as_tensor(
                output_lengths["poses"],
                device=device
            )

            B, T, F = x.shape

            # -------------------------------------------------
            # Sample diffusion timesteps and add noise
            # -------------------------------------------------
            t = torch.randint(0, TIMESTEPS, (B,), device=device)

            # move schedule to correct device
            ab = alpha_bar.to(device)[t].view(B, 1, 1)  # alpha_bar_t

            noise = torch.randn_like(x)
            x_t = torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * noise

            # -------------------------------------------------
            # Predict noise with diffusion model
            # -------------------------------------------------
            noise_pred = model(x_t, t)

            # elementwise squared error
            loss_matrix = criterion(noise_pred, noise)  # (B, T, F)

            # -------------------------------------------------
            # Mask Padding Frames
            # -------------------------------------------------
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)

            masked_loss = (loss_matrix * mask).sum()
            num_valid = mask.sum() * F  # count valid elements

            if num_valid > 0:
                loss = masked_loss / num_valid
            else:
                loss = torch.tensor(
                    0.0,
                    device=device,
                    requires_grad=True
                )

            # -------------------------------------------------
            # Backprop
            # -------------------------------------------------
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            with torch.no_grad():
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

            print(
                f"Epoch {epoch} | "
                f"Batch Loss: {loss.item():.6f} | "
                f"Grad Norm: {grad_norm:.4f}"
            )

            epoch_weighted_loss += masked_loss.item()
            total_frames += num_valid.item()

        # -------------------------------------------------
        # Epoch Summary
        # -------------------------------------------------
        if total_frames > 0:
            epoch_loss = epoch_weighted_loss / total_frames
        else:
            epoch_loss = 0.0

        print("===================================")
        print(f"Epoch {epoch} Summary")
        print(f"Epoch Loss: {epoch_loss:.6f}")
        print("===================================")

        # ---------------- Early stopping ----------------
        if best_epoch_loss is None or epoch_loss < best_epoch_loss - min_delta:
            best_epoch_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s) (best {best_epoch_loss:.6f}).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Save model after training
    torch.save(model.state_dict(), "pose_diffusion_transformer.pth")
    print("Model saved.")


def validate(val_loader, model, device, mpj_eval):
    from utils.model_utils import reduced_pose_to_full
    from articulate.math import r6d_to_rotation_matrix
    import torch
    import torch.nn.functional as F

    model.eval()

    val_weighted_loss = 0.0
    val_mpj_sum = 0.0
    val_total_frames = 0

    with torch.no_grad():
        for batch in val_loader:

            (inputs, input_lengths), (outputs, output_lengths) = batch

            x = outputs["poses"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, Fdim = x.shape

            # ---------------- Diffusion forward (validation) ----------------
            # use the same training-style noising: random t, predict noise, reconstruct x0
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            ab = alpha_bar.to(device)[t].view(B, 1, 1)

            noise = torch.randn_like(x)
            x_t = torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * noise

            noise_pred = model(x_t, t)

            # reconstruction of x0 from x_t and predicted noise
            x0_hat = (x_t - torch.sqrt(1.0 - ab) * noise_pred) / torch.sqrt(ab)

            loss_matrix = F.mse_loss(noise_pred, noise, reduction="none")

            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()

            masked_loss = (loss_matrix * mask).sum()
            num_valid = mask.sum() * Fdim

            # ---------------- MPJPE Computation on reconstructed pose ----------------
            if num_valid > 0:

                if Fdim % 9 == 0:
                    in_rep = 9
                elif Fdim % 6 == 0:
                    in_rep = 6
                else:
                    raise RuntimeError(f"Unknown pose feature dim F={Fdim}")

                num_joints = Fdim // in_rep

                pose_pred = x0_hat.view(B * T, num_joints, in_rep)
                pose_true = x.view(B * T, num_joints, in_rep)

                valid_idx = mask.view(-1).bool()

                if valid_idx.any():

                    if in_rep == 6:
                        pose_pred_mat = r6d_to_rotation_matrix(pose_pred)
                        pose_true_mat = r6d_to_rotation_matrix(pose_true)
                    else:
                        pose_pred_mat = pose_pred.view(-1, num_joints, 3, 3)
                        pose_true_mat = pose_true.view(-1, num_joints, 3, 3)

                    pose_pred_mat = pose_pred_mat[valid_idx]
                    pose_true_mat = pose_true_mat[valid_idx]

                    # Get model joint count safely
                    try:
                        j, _ = mpj_eval.model.get_zero_pose_joint_and_vertex(None)
                        model_num_j = j.shape[1] if j.dim() == 3 else j.shape[0]
                    except Exception:
                        model_num_j = pose_pred_mat.shape[1]

                    # Expand reduced → full if needed
                    if pose_pred_mat.shape[1] != model_num_j:
                        N_valid = pose_pred_mat.shape[0]

                        pose_pred_full = reduced_pose_to_full(
                            pose_pred_mat.unsqueeze(1)
                        ).view(N_valid, model_num_j, 3, 3)

                        pose_true_full = reduced_pose_to_full(
                            pose_true_mat.unsqueeze(1)
                        ).view(N_valid, model_num_j, 3, 3)

                        pose_pred_mat = pose_pred_full
                        pose_true_mat = pose_true_full

                    mpj_tensor = mpj_eval(pose_pred_mat, pose_true_mat)

                    val_mpj_sum += (
                        mpj_tensor[0].item() * valid_idx.sum().item()
                    )

            val_weighted_loss += masked_loss.item()
            val_total_frames += num_valid.item()

    # ---------------- Final Metrics ----------------
    if val_total_frames > 0:
        val_loss = val_weighted_loss / val_total_frames
        val_mpj = val_mpj_sum / val_total_frames
    else:
        val_loss = 0.0
        val_mpj = 0.0

    print(f"Validation Loss: {val_loss:.6f} | MPJPE: {val_mpj:.6f}")

    return val_loss, val_mpj


datamodule = PoseDataModule(finetune=None)

# IMPORTANT: must call setup
datamodule.setup(stage='fit')

# Get train loader
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


device = "cuda" if torch.cuda.is_available() else "cpu"

# Use temporal Transformer denoiser (same input/output shape as before)
# model = TemporalTransformerDenoiser(feature_dim=144).to(device)
model = TemporalTransformerDenoiser(
    feature_dim=144,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
).to(device)
training(train_loader=train_loader, model=model, num_epochs=100, patience=5)

mpj_eval = MeanPerJointErrorEvaluator(
    official_model_file=paths.smpl_file,
    rep=RotationRepresentation.ROTATION_MATRIX,
    device=device
)

val_loss, val_mpj = validate(
    val_loader=val_loader,
    model=model,
    device=device,
    mpj_eval=mpj_eval
)


from articulate.math import r6d_to_rotation_matrix
from viewers.smpl_viewer import SMPLViewer



model.eval()

k = 0

for i, batch in enumerate(val_loader):
    if i == k:
        break

(inputs, input_lengths), (outputs, output_lengths) = batch

x = outputs["poses"].to(device)
trans = outputs["trans"].to(device)

B, T, Fdim = x.shape
lengths = output_lengths["poses"]

# ---------- DDPM reverse (ancestral) sampling ----------
_betas    = betas.to(device)
_alphas   = alphas.to(device)
_alpha_bar = alpha_bar.to(device)

x_t = torch.randn_like(x)                         # start from pure noise

with torch.no_grad():
    for t_val in reversed(range(TIMESTEPS)):
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)

        noise_pred = model(x_t, t_tensor)          # predicted noise

        beta_t     = _betas[t_val]
        alpha_t    = _alphas[t_val]
        alpha_bar_t = _alpha_bar[t_val]

        # DDPM mean: mu_theta(x_t, t)
        coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean  = (1.0 / torch.sqrt(alpha_t)) * (x_t - coeff * noise_pred)

        if t_val > 0:
            sigma = torch.sqrt(beta_t)
            x_t = mean + sigma * torch.randn_like(x_t)
        else:
            x_t = mean                              # no noise at t = 0

pred = x_t                                          # final denoised prediction

gt_batch = x
pred_batch = pred
tran_batch = trans
lengths = output_lengths["poses"]

viewer = SMPLViewer(fps=25)


batch_size = 10

for b in range(batch_size):
    # valid length for this sequence
    L = int(lengths[b])

    # Take only valid frames for this sequence
    gt_pose_r6d   = gt_batch[b, :L]      # (L, 144)
    pred_pose_r6d = pred_batch[b, :L]    # (L, 144)
    tran_gt       = tran_batch[b, :L]    # (L, 3)
    tran_pred     = tran_gt              # model predicts pose, not translation

    # Convert R6D to rotation matrices
    gt_pose_rot = r6d_to_rotation_matrix(
        gt_pose_r6d.view(-1, 24, 6)
    ).view(-1, 24, 3, 3)

    pred_pose_rot = r6d_to_rotation_matrix(
        pred_pose_r6d.view(-1, 24, 6)
    ).view(-1, 24, 3, 3)

    print(f"Visualizing sequence {b+1}/{batch_size}, length = {L}")
    viewer.view(pred_pose_rot, tran_pred, gt_pose_rot, tran_gt, with_tran=True)