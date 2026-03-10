#libraries


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


import os
from articulate.math import r6d_to_rotation_matrix
from viewers.smpl_viewer import SMPLViewer
from articulate import model as art_model
from utils.model_utils import reduced_pose_to_full
from articulate.math import r6d_to_rotation_matrix
from articulate.evaluator import MeanPerJointErrorEvaluator
from articulate.math import RotationRepresentation


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

            print("\n" + "-"*50)
            print("NEW SEQUENCE")
            print("Original pose shape from file:", pose.shape)

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

        print("\nProcessing translation module")

        root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
        vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint, dim=0)))
        vel[:, 0] = root_vel

        vel = vel * (datasets.fps / amass.vel_scale)

        vel_splits = torch.split(vel, data_len)

        print("Velocity windows created:", len(vel_splits))
        print("Velocity window shape:", vel_splits[0].shape)

        data['vel_outputs'].extend(vel_splits)
        data['foot_outputs'].extend(torch.split(foot, data_len))

    def __getitem__(self, idx):

        imu = self.data['imu_inputs'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()

        # print("\n" + "#"*60)
        # print("INSIDE __getitem__")
        # print("Index:", idx)
        # print("Raw pose window shape:",
        #       self.data['pose_outputs'][idx].shape)

        num_pred_joints = len(amass.pred_joints_set)

        pose = art.math.rotation_matrix_to_r6d(
            self.data['pose_outputs'][idx]
        ).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set] \
         .reshape(-1, 6*num_pred_joints)

        # print("Pose AFTER r6d conversion shape:", pose.shape)
        # print("#"*60)

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


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep, projected through an MLP."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps -> (B, d_model)"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Diffusion noise schedule helpers
# ---------------------------------------------------------------------------
def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008):
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


class GaussianDiffusion(nn.Module):
    """Manages the forward (noising) and reverse (denoising) diffusion process."""
    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])

        # Register all as buffers so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t]
        # broadcast to (B, 1, 1) if x0 is (B, T, D)
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_omab = sqrt_omab.unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index: int):
        """Single reverse step: p(x_{t-1} | x_t)."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_index, device=x_t.device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor)
        beta = self.betas[t_index]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t_index]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t_index]
        mean = sqrt_recip_alpha * (x_t - beta / sqrt_omab * noise_pred)
        if t_index > 0:
            sigma = torch.sqrt(self.posterior_variance[t_index])
            mean = mean + sigma * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def sample(self, model, shape, device):
        """Full reverse sampling loop: x_T -> x_0."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t)
        return x


class TemporalTransformerDiffusion(nn.Module):
    """Unconditional diffusion model (noise predictor) for pose + translation.

    Takes noisy (pose, tran) concatenated along feature dim + a diffusion
    timestep, and predicts the noise that was added.
    """
    def __init__(
        self,
        pose_dim: int,
        tran_dim: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.tran_dim = tran_dim
        combined_dim = pose_dim + tran_dim

        self.in_proj = nn.Linear(combined_dim, d_model)
        self.time_emb = SinusoidalTimestepEmbedding(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # separate output heads predicting noise for pose and translation
        self.pose_head = nn.Linear(d_model, pose_dim)
        self.tran_head = nn.Linear(d_model, tran_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """Predict noise from noisy input + timestep.

        x_t : (B, T, pose_dim + tran_dim)  noisy concatenated pose+tran
        t   : (B,) integer diffusion timesteps
        Returns: (B, T, pose_dim + tran_dim) predicted noise
        """
        h = self.in_proj(x_t)                       # (B, T, d_model)
        t_emb = self.time_emb(t).unsqueeze(1)        # (B, 1, d_model)
        h = h + t_emb                                # broadcast add timestep info
        h = self.pos_enc(h)
        h = self.encoder(h)                          # (B, T, d_model)
        pose_noise = self.pose_head(h)
        tran_noise = self.tran_head(h)
        return torch.cat([pose_noise, tran_noise], dim=-1)
    
def validate(val_loader, model, diffusion, device):
    model.eval()
    val_loss_sum = 0.0
    val_total_frames = 0

    with torch.no_grad():
        for batch in val_loader:
            (inputs, input_lengths), (outputs, output_lengths) = batch

            pose = outputs["poses"].to(device)
            tran = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, F_pose = pose.shape
            x0 = torch.cat([pose, tran], dim=-1)  # (B, T, pose_dim + tran_dim)

            # sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t, _ = diffusion.q_sample(x0, t, noise)

            noise_pred = model(x_t, t)

            loss_matrix = nn.functional.mse_loss(noise_pred, noise, reduction="none")

            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            frame_mask = mask.unsqueeze(-1).float()

            masked_loss = (loss_matrix * frame_mask).sum()
            num_valid = frame_mask.sum()

            val_loss_sum += masked_loss.item()
            val_total_frames += num_valid.item()

    val_loss = val_loss_sum / val_total_frames if val_total_frames > 0 else 0.0
    print(f"Validation Loss: {val_loss:.6f}")
    return val_loss


def training(train_loader, val_loader, model, diffusion, num_epochs=1, device=None, patience: int = 10, min_delta: float = 1e-4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    diffusion = diffusion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss(reduction="none")

    best_epoch_loss = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_weighted_loss = 0.0
        total_frames = 0

        for batch in train_loader:
            (inputs, input_lengths), (outputs, output_lengths) = batch

            pose = outputs["poses"].to(device)
            tran = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, F_pose = pose.shape

            # concatenate pose + tran as the clean signal x_0
            x0 = torch.cat([pose, tran], dim=-1)  # (B, T, pose_dim + tran_dim)

            # sample random diffusion timesteps per sample
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

            # forward diffusion: add noise according to schedule
            noise = torch.randn_like(x0)
            x_t, _ = diffusion.q_sample(x0, t, noise)

            # predict the noise
            noise_pred = model(x_t, t)

            loss_matrix = criterion(noise_pred, noise)

            # mask padding frames
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            frame_mask = mask.unsqueeze(-1).float()

            masked_loss = (loss_matrix * frame_mask).sum()
            num_valid = frame_mask.sum()

            if num_valid > 0:
                loss = masked_loss / num_valid
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5

            print(f"Batch Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.4f}")

            epoch_weighted_loss += masked_loss.item()
            total_frames += num_valid.item()

        epoch_loss = epoch_weighted_loss / total_frames if total_frames > 0 else 0.0
        print("===================================")
        print(f"Epoch {epoch}")
        print(f"Epoch Loss: {epoch_loss:.6f}")
        print("===================================")

        # early stopping
        if best_epoch_loss is None or epoch_loss < best_epoch_loss - min_delta:
            best_epoch_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "diffusion_model_best.pth")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        val_loss = validate(val_loader, model, diffusion, device)

    torch.save(model.state_dict(), "diffusion_model_final.pth")


# ---------------------------------------------------------------------------
# Sampling helper (call after training)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(model, diffusion, num_samples: int, seq_len: int, pose_dim: int, tran_dim: int, device):
    """Generate unconditional pose + translation sequences from noise."""
    model.eval()
    shape = (num_samples, seq_len, pose_dim + tran_dim)
    samples = diffusion.sample(model, shape, device)
    pose_samples = samples[..., :pose_dim]
    tran_samples = samples[..., pose_dim:]
    return pose_samples, tran_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datamodule = PoseDataModule(finetune=None)
    datamodule.setup(stage='fit')

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    POSE_DIM = 144
    TRAN_DIM = 3
    NUM_DIFFUSION_STEPS = 1000

    diffusion = GaussianDiffusion(
        num_timesteps=NUM_DIFFUSION_STEPS,
        schedule="cosine",
    ).to(device)

    model = TemporalTransformerDiffusion(
        pose_dim=POSE_DIM,
        tran_dim=TRAN_DIM,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)

    print(f"Unconditional Diffusion Model initialized — pose_dim={POSE_DIM}, tran_dim={TRAN_DIM}, "
          f"timesteps={NUM_DIFFUSION_STEPS}")

    training(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        diffusion=diffusion,
        num_epochs=100,
        patience=10,
        device=device,
    )


