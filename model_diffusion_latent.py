# Latent Diffusion Model for Pose Estimation
# Architecture: Transformer Encoder → Latent Space → Diffusion → Transformer Decoder
#
# Stage 1: Train autoencoder (pose/tran → latent → pose/tran)
# Stage 2: Freeze autoencoder, train diffusion in latent space

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
import copy
import os

import articulate as art
from config import *
from utils import *
from helpers import *

from articulate.math import r6d_to_rotation_matrix
from viewers.smpl_viewer import SMPLViewer
from articulate import model as art_model
from utils.model_utils import reduced_pose_to_full
from articulate.evaluator import MeanPerJointErrorEvaluator
from articulate.math import RotationRepresentation


# ===========================================================================
# Dataset (reused from conditional model)
# ===========================================================================

class PoseDataset(Dataset):
    def __init__(self, fold: str = 'train', evaluate: str = None, finetune: str = None):
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

        return data

    def _process_file_data(self, file_data, data):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))

        for acc, ori, pose, tran, joint, foot in zip(accs, oris, poses, trans, joints, foots):
            acc, ori = acc[:, :5] / amass.acc_scale, ori[:, :5]

            pose_global, joint = self.bodymodel.forward_kinematics(
                pose=pose.view(-1, 216)
            )

            pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)
            joint = joint.view(-1, 24, 3)

            self._process_combo_data(acc, ori, pose, joint, tran, foot, data)

    def _process_combo_data(self, acc, ori, pose, joint, tran, foot, data):
        for combo_name, c in self.combos:
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
                splits = torch.split(value, data_len)
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
         .reshape(-1, 6 * num_pred_joints)

        if self.evaluate or self.finetune:
            return imu, pose, joint, tran

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()
        return imu, pose, joint, tran, vel, contact

    def __len__(self):
        return len(self.data['imu_inputs'])


def pad_seq(batch):
    """Pad sequences to same length."""
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

    if len(batch[0]) > 5:
        vels, foots = zip(*[(item[4], item[5]) for item in batch])
        foot_contacts, foot_contact_lengths = _pad(foots)
        outputs['foot_contacts'], output_lengths['foot_contacts'] = foot_contacts, foot_contact_lengths
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
            num_workers=0,
            shuffle=True,
            drop_last=True,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)


# ===========================================================================
# Shared building blocks
# ===========================================================================

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
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


# ===========================================================================
# Stage 1: Transformer VAE (Autoencoder)
# ===========================================================================

class TemporalTransformerEncoder(nn.Module):
    """Encodes a temporal signal (B, T, input_dim) -> (B, T, latent_dim).

    Uses a Transformer encoder to capture temporal dependencies before
    projecting to mu and logvar for the VAE reparameterisation trick.
    """
    def __init__(self, input_dim: int, latent_dim: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mu_proj = nn.Linear(d_model, latent_dim)
        self.logvar_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        """Returns (mu, logvar) each of shape (B, T, latent_dim)."""
        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.mu_proj(h), self.logvar_proj(h)


class TemporalTransformerDecoder(nn.Module):
    """Decodes latent (B, T, latent_dim) -> (B, T, output_dim)."""
    def __init__(self, latent_dim: int, output_dim: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, z):
        h = self.in_proj(z)
        h = self.pos_enc(h)
        h = self.decoder(h)
        return self.out_proj(h)


class PoseTranslationVAE(nn.Module):
    """Variational Autoencoder with separate pose/translation encoders.

    Pose encoder:  (B, T, 144) -> (B, T, pose_latent_dim)
    Tran encoder:  (B, T, 3)   -> (B, T, tran_latent_dim)
    Concatenated latent: (B, T, pose_latent_dim + tran_latent_dim)
    Decoder splits the latent back into pose and translation.
    """
    def __init__(
        self,
        pose_dim: int = 144,
        tran_dim: int = 3,
        pose_latent_dim: int = 48,
        tran_latent_dim: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.tran_dim = tran_dim
        self.pose_latent_dim = pose_latent_dim
        self.tran_latent_dim = tran_latent_dim
        self.latent_dim = pose_latent_dim + tran_latent_dim  # 64

        # Separate encoders
        self.pose_encoder = TemporalTransformerEncoder(
            input_dim=pose_dim, latent_dim=pose_latent_dim,
            d_model=d_model, nhead=nhead, num_layers=enc_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.tran_encoder = TemporalTransformerEncoder(
            input_dim=tran_dim, latent_dim=tran_latent_dim,
            d_model=d_model // 2, nhead=max(1, nhead // 2), num_layers=max(2, enc_layers // 2),
            dim_feedforward=dim_feedforward // 2, dropout=dropout,
        )

        # Single decoder from combined latent
        self.decoder = TemporalTransformerDecoder(
            latent_dim=self.latent_dim,
            output_dim=pose_dim + tran_dim,
            d_model=d_model, nhead=nhead, num_layers=dec_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, pose, tran):
        """Encode pose and tran into a combined latent (B, T, latent_dim)."""
        pose_mu, pose_logvar = self.pose_encoder(pose)
        tran_mu, tran_logvar = self.tran_encoder(tran)

        mu = torch.cat([pose_mu, tran_mu], dim=-1)
        logvar = torch.cat([pose_logvar, tran_logvar], dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        """Decode latent to pose + tran."""
        out = self.decoder(z)
        pose_recon = out[..., :self.pose_dim]
        tran_recon = out[..., self.pose_dim:]
        return pose_recon, tran_recon

    def forward(self, pose, tran):
        """Full forward: encode → reparameterize → decode."""
        z, mu, logvar = self.encode(pose, tran)
        pose_recon, tran_recon = self.decode(z)
        return pose_recon, tran_recon, mu, logvar, z


# ===========================================================================
# Stage 2: Latent Diffusion
# ===========================================================================

def cosine_beta_schedule(num_timesteps: int, s: float = 0.008):
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)


class GaussianDiffusion(nn.Module):
    """Forward/reverse diffusion process — operates in latent space."""
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

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: q(z_t | z_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t]
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_omab = sqrt_omab.unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index: int, cond=None):
        """Single reverse step."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_index, device=x_t.device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor, cond=cond)
        beta = self.betas[t_index]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t_index]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t_index]
        mean = sqrt_recip_alpha * (x_t - beta / sqrt_omab * noise_pred)
        if t_index > 0:
            sigma = torch.sqrt(self.posterior_variance[t_index])
            mean = mean + sigma * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def ddim_sample(self, model, x_T, num_steps: int = 50, eta: float = 0.0, cond=None):
        """DDIM sampling for faster generation."""
        step_size = max(1, self.num_timesteps // num_steps)
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        x = x_T
        for i, t_cur in enumerate(timesteps):
            B = x.shape[0]
            t_tensor = torch.full((B,), t_cur, device=x.device, dtype=torch.long)
            noise_pred = model(x, t_tensor, cond=cond)

            alpha_bar_t = self.alpha_bar[t_cur]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_bar_prev = self.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=x.device)

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-5, 5)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0)) * noise_pred

            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            if t_cur > 0 and eta > 0:
                x = x + sigma * torch.randn_like(x)

        return x

    @torch.no_grad()
    def sample(self, model, shape, device, cond=None):
        """Full reverse sampling loop."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, cond=cond)
        return x


class LatentDenoiser(nn.Module):
    """Denoiser that operates on the latent space with IMU conditioning.

    Input:  noisy latent z_t (B, T, latent_dim)
    Cond:   IMU features    (B, T, imu_dim)
    Output: predicted noise  (B, T, latent_dim)
    """
    def __init__(
        self,
        latent_dim: int = 64,
        imu_dim: int = 60,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Noisy latent projection
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Timestep embedding
        self.time_emb = SinusoidalTimestepEmbedding(d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # IMU conditioning encoder
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        imu_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.imu_encoder = nn.TransformerEncoder(imu_enc_layer, num_layers=2)

        # Decoder: noisy latent attends to IMU features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, z_t, t, cond=None):
        """Predict noise in latent space.

        z_t  : (B, T, latent_dim)  noisy latent
        t    : (B,) diffusion timesteps
        cond : (B, T, imu_dim) IMU conditioning (optional)
        """
        h = self.in_proj(z_t)
        t_emb = self.time_emb(t).unsqueeze(1)
        h = h + t_emb
        h = self.pos_enc(h)

        if cond is not None:
            c = self.imu_proj(cond)
            c = self.pos_enc(c)
            c = self.imu_encoder(c)
            h = self.decoder(h, c)
        else:
            h = self.decoder(h, h)

        h = self.out_norm(h)
        return self.out_proj(h)


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)


# ===========================================================================
# Stage 1 Training: Autoencoder
# ===========================================================================

def train_autoencoder(
    train_loader, val_loader, vae, num_epochs=50, device=None,
    patience=10, min_delta=1e-4, kl_weight=1e-4,
):
    """Train the PoseTranslationVAE with reconstruction + KL loss."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = vae.to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)

    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        vae.train()
        epoch_loss_sum = 0.0
        total_frames = 0

        for batch in train_loader:
            (inputs, input_lengths), (outputs, output_lengths) = batch

            pose = outputs["poses"].to(device)
            tran = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, _ = pose.shape

            pose_recon, tran_recon, mu, logvar, _ = vae(pose, tran)

            # Reconstruction loss (masked for padding)
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]

            pose_recon_loss = nn.functional.mse_loss(pose_recon, pose, reduction="none")
            tran_recon_loss = nn.functional.mse_loss(tran_recon, tran, reduction="none")

            pose_mask = mask.unsqueeze(-1).expand_as(pose_recon_loss).float()
            tran_mask = mask.unsqueeze(-1).expand_as(tran_recon_loss).float()

            recon_loss = (
                (pose_recon_loss * pose_mask).sum() +
                (tran_recon_loss * tran_mask).sum()
            ) / (pose_mask.sum() + tran_mask.sum())

            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            num_valid = pose_mask.sum().item()
            epoch_loss_sum += loss.item() * num_valid
            total_frames += num_valid

        scheduler.step()
        epoch_loss = epoch_loss_sum / total_frames if total_frames > 0 else 0.0

        # Validation
        val_loss = validate_autoencoder(val_loader, vae, device, kl_weight)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[AE] Epoch {epoch:3d} | Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        if best_val_loss is None or val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(vae.state_dict(), "vae_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("AE early stopping triggered.")
                break

    print(f"\n[AE] Final Train Loss: {epoch_loss:.6f} | Final Val Loss: {val_loss:.6f}")
    with open("vae_loss.txt", "w") as f:
        f.write(f"[AE] Final Train Loss: {epoch_loss:.6f} | Final Val Loss: {val_loss:.6f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
    torch.save(vae.state_dict(), "vae_final.pth")
    print("Autoencoder training complete.")


def validate_autoencoder(val_loader, vae, device, kl_weight=1e-4):
    vae.eval()
    val_loss_sum = 0.0
    val_frames = 0

    with torch.no_grad():
        for batch in val_loader:
            (inputs, _), (outputs, output_lengths) = batch
            pose = outputs["poses"].to(device)
            tran = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, _ = pose.shape
            pose_recon, tran_recon, mu, logvar, _ = vae(pose, tran)

            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            pose_mask = mask.unsqueeze(-1).expand_as(pose_recon).float()
            tran_mask = mask.unsqueeze(-1).expand_as(tran_recon).float()

            recon_loss = (
                (nn.functional.mse_loss(pose_recon, pose, reduction="none") * pose_mask).sum() +
                (nn.functional.mse_loss(tran_recon, tran, reduction="none") * tran_mask).sum()
            ) / (pose_mask.sum() + tran_mask.sum())

            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss

            num_valid = pose_mask.sum().item()
            val_loss_sum += loss.item() * num_valid
            val_frames += num_valid

    return val_loss_sum / val_frames if val_frames > 0 else 0.0


# ===========================================================================
# Stage 2 Training: Latent Diffusion
# ===========================================================================

def train_latent_diffusion(
    train_loader, val_loader, vae, denoiser, diffusion,
    num_epochs=100, device=None, patience=10, min_delta=1e-4,
):
    """Train the latent denoiser with frozen VAE."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    denoiser = denoiser.to(device)
    diffusion = diffusion.to(device)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4, weight_decay=1e-2)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = EMA(denoiser, decay=0.9999)
    criterion = nn.MSELoss(reduction="none")

    best_val_loss = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        denoiser.train()
        epoch_loss_sum = 0.0
        total_frames = 0

        for batch in train_loader:
            (inputs, input_lengths), (outputs, output_lengths) = batch

            imu_cond = None  # unconditional
            pose = outputs["poses"].to(device)
            tran = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, _ = pose.shape

            # Encode to latent (no grad through VAE)
            with torch.no_grad():
                z, _, _ = vae.encode(pose, tran)

            # Diffusion: noise the latent
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            noise = torch.randn_like(z)
            z_t, _ = diffusion.q_sample(z, t, noise)

            # Predict noise, conditioned on IMU
            noise_pred = denoiser(z_t, t, cond=imu_cond)

            loss_matrix = criterion(noise_pred, noise)

            # Mask padding
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            frame_mask = mask.unsqueeze(-1).expand_as(loss_matrix).float()

            masked_loss = (loss_matrix * frame_mask).sum()
            num_valid = frame_mask.sum()

            loss = masked_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(denoiser)

            epoch_loss_sum += masked_loss.item()
            total_frames += num_valid.item()

            print(f"  Batch Loss: {loss.item():.6f}")

        scheduler.step()
        epoch_loss = epoch_loss_sum / total_frames if total_frames > 0 else 0.0

        # Validate with EMA model
        val_loss = validate_latent_diffusion(val_loader, vae, ema.shadow, diffusion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[LD] Epoch {epoch:3d} | Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        if best_val_loss is None or val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ema.shadow.state_dict(), "latent_denoiser_best.pth")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("Latent diffusion early stopping triggered.")
                break

    print(f"\n[LD] Final Train Loss: {epoch_loss:.6f} | Final Val Loss: {val_loss:.6f}")
    with open("latent_denoiser_loss.txt", "w") as f:
        f.write(f"[LD] Final Train Loss: {epoch_loss:.6f} | Final Val Loss: {val_loss:.6f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
    torch.save(ema.shadow.state_dict(), "latent_denoiser_final.pth")
    print("Latent diffusion training complete.")


def validate_latent_diffusion(val_loader, vae, denoiser, diffusion, device):
    denoiser.eval()
    val_loss_sum = 0.0
    val_frames = 0

    with torch.no_grad():
        for batch in val_loader:
            (inputs, _), (outputs, output_lengths) = batch

            imu_cond = None  # unconditional
            pose = outputs["poses"].to(device)
            tran = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, _ = pose.shape

            z, _, _ = vae.encode(pose, tran)

            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            noise = torch.randn_like(z)
            z_t, _ = diffusion.q_sample(z, t, noise)

            noise_pred = denoiser(z_t, t, cond=imu_cond)

            loss_matrix = nn.functional.mse_loss(noise_pred, noise, reduction="none")
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            frame_mask = mask.unsqueeze(-1).expand_as(loss_matrix).float()

            val_loss_sum += (loss_matrix * frame_mask).sum().item()
            val_frames += frame_mask.sum().item()

    return val_loss_sum / val_frames if val_frames > 0 else 0.0


# ===========================================================================
# Inference: sample from latent diffusion, decode to pose+tran
# ===========================================================================

@torch.no_grad()
def generate_samples(vae, denoiser, diffusion, num_samples, seq_len, device, cond=None):
    """Sample latents from pure noise via DDIM, then decode to pose + translation."""
    vae.eval()
    denoiser.eval()
    shape = (num_samples, seq_len, vae.latent_dim)
    x_T = torch.randn(shape, device=device)
    z_samples = diffusion.ddim_sample(denoiser, x_T, num_steps=50, eta=0.0, cond=cond)
    pose_samples, tran_samples = vae.decode(z_samples)
    return pose_samples, tran_samples


@torch.no_grad()
def refine_pose(
    vae, denoiser, diffusion, pose_r6d, tran, imu_cond=None,
    noise_level: int = 100, ddim_steps: int = 50, device=None,
):
    """Refine an input SMPL pose using encoder → diffusion → decoder.

    Pipeline:
        1. Encode input pose+tran to latent z₀ via the VAE encoder
        2. Forward-noise z₀ to z_t at the specified noise_level
        3. DDIM-denoise z_t back to z₀ (conditioned on IMU if provided)
        4. Decode refined z₀ to get cleaned pose + translation

    Args:
        vae:         Trained PoseTranslationVAE
        denoiser:    Trained LatentDenoiser
        diffusion:   GaussianDiffusion schedule
        pose_r6d:    Input pose in r6d format (B, T, 144)
        tran:        Input translation (B, T, 3)
        imu_cond:    IMU sensor data (B, T, 60), optional
        noise_level: How much noise to add before denoising (0–999).
                     Higher = more aggressive refinement.
        ddim_steps:  Number of DDIM denoising steps.
        device:      Device override.

    Returns:
        (refined_pose_r6d, refined_tran) — same shapes as inputs.
    """
    if device is None:
        device = pose_r6d.device

    vae.eval()
    denoiser.eval()

    pose_r6d = pose_r6d.to(device)
    tran = tran.to(device)
    if imu_cond is not None:
        imu_cond = imu_cond.to(device)

    # Step 1: Encode input pose+tran → latent z₀
    z0, _, _ = vae.encode(pose_r6d, tran)

    # Step 2: Forward-noise to noise_level
    B = z0.shape[0]
    t = torch.full((B,), noise_level, device=device, dtype=torch.long)
    noise = torch.randn_like(z0)
    z_noisy, _ = diffusion.q_sample(z0, t, noise)

    # Step 3: DDIM-denoise from noise_level back to 0
    step_size = max(1, noise_level // ddim_steps)
    timesteps = list(range(0, noise_level, step_size))[::-1]

    z_denoised = z_noisy
    for idx, t_cur in enumerate(timesteps):
        t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)
        noise_pred = denoiser(z_denoised, t_tensor, cond=imu_cond)

        alpha_bar_t = diffusion.alpha_bar[t_cur]
        t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
        alpha_bar_prev = diffusion.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

        z0_pred = (z_denoised - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        z0_pred = z0_pred.clamp(-5, 5)
        dir_zt = torch.sqrt(torch.clamp(1 - alpha_bar_prev, min=0)) * noise_pred
        z_denoised = torch.sqrt(alpha_bar_prev) * z0_pred + dir_zt

    # Step 4: Decode refined latent → pose + tran
    refined_pose, refined_tran = vae.decode(z_denoised)
    return refined_pose, refined_tran


# ===========================================================================
# Shared: load models + data for evaluation / visualisation
# ===========================================================================

def _load_eval_models(vae_checkpoint, denoiser_checkpoint, device=None):
    """Load trained VAE, denoiser, diffusion, data loader, and body model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    POSE_DIM = 144
    TRAN_DIM = 3
    POSE_LATENT = 48
    TRAN_LATENT = 16
    LATENT_DIM = POSE_LATENT + TRAN_LATENT
    NUM_DIFFUSION_STEPS = 1000

    vae = PoseTranslationVAE(
        pose_dim=POSE_DIM, tran_dim=TRAN_DIM,
        pose_latent_dim=POSE_LATENT, tran_latent_dim=TRAN_LATENT,
    ).to(device)
    vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    vae.eval()

    diffusion = GaussianDiffusion(num_timesteps=NUM_DIFFUSION_STEPS, schedule="cosine").to(device)
    denoiser = LatentDenoiser(latent_dim=LATENT_DIM, imu_dim=60).to(device)
    denoiser.load_state_dict(torch.load(denoiser_checkpoint, map_location=device))
    denoiser.eval()

    datamodule = PoseDataModule(finetune=None)
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()

    bodymodel = art.model.ParametricModel(paths.smpl_file, device=device)

    return vae, denoiser, diffusion, val_loader, bodymodel, device


# ===========================================================================
# Visualisation only
# ===========================================================================

def visualize_results(
    vae_checkpoint: str = "vae_best.pth",
    denoiser_checkpoint: str = "latent_denoiser_best.pth",
    noise_level: int = 100,
    num_vis_samples: int = 2,
):
    """Run the pipeline on a few samples and visualise GT vs predicted SMPL."""
    os.environ["GT"] = "1"
    vae, denoiser, diffusion, val_loader, bodymodel, device = _load_eval_models(
        vae_checkpoint, denoiser_checkpoint,
    )

    flip_rot = torch.eye(3, device=device)
    flip_rot[1, 1] = -1
    flip_rot[2, 2] = -1

    print(f"\nRunning visualisation (noise_level={noise_level}, samples={num_vis_samples})")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            (inputs, _), (outputs, output_lengths) = batch

            gt_pose_r6d = outputs["poses"].to(device)
            gt_tran = outputs["trans"].to(device)
            lengths = output_lengths["poses"]

            pred_pose_r6d, pred_tran = refine_pose(
                vae, denoiser, diffusion,
                pose_r6d=gt_pose_r6d, tran=gt_tran,
                imu_cond=None, noise_level=noise_level,
                ddim_steps=max(10, noise_level // 10), device=device,
            )

            viewer = SMPLViewer(fps=25)
            for b in range(min(num_vis_samples, gt_pose_r6d.shape[0])):
                L = int(lengths[b])
                gt_rot = r6d_to_rotation_matrix(gt_pose_r6d[b, :L].view(-1, 24, 6)).view(-1, 24, 3, 3)
                pred_rot = r6d_to_rotation_matrix(pred_pose_r6d[b, :L].view(-1, 24, 6)).view(-1, 24, 3, 3)
                gt_local = bodymodel.inverse_kinematics_R(gt_rot)
                pred_local = bodymodel.inverse_kinematics_R(pred_rot)
                gt_local[:, 0] = flip_rot @ gt_local[:, 0]
                pred_local[:, 0] = flip_rot @ pred_local[:, 0]
                viewer.view(pred_local, pred_tran[b, :L], gt_local, gt_tran[b, :L], with_tran=True)

            break  # only first batch


# ===========================================================================
# MPJPE evaluation only
# ===========================================================================

def evaluate_mpjpe(
    vae_checkpoint: str = "vae_best.pth",
    denoiser_checkpoint: str = "latent_denoiser_best.pth",
    noise_levels: list = None,
):
    """Compute MPJPE / angle errors across the validation set."""
    if noise_levels is None:
        noise_levels = [50, 80, 100]

    os.environ["GT"] = "1"
    vae, denoiser, diffusion, val_loader, bodymodel, device = _load_eval_models(
        vae_checkpoint, denoiser_checkpoint,
    )

    mpjpe_evaluator = MeanPerJointErrorEvaluator(
        official_model_file=str(paths.smpl_file),
        rep=RotationRepresentation.ROTATION_MATRIX,
        device=device,
    )

    results = []

    for NOISE_LEVEL in noise_levels:
        all_mpjpe, all_local_err, all_global_err = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Eval (noise={NOISE_LEVEL})")):
                (inputs, _), (outputs, output_lengths) = batch

                gt_pose_r6d = outputs["poses"].to(device)
                gt_tran = outputs["trans"].to(device)
                lengths = output_lengths["poses"]

                B, T, _ = gt_pose_r6d.shape

                pred_pose_r6d, pred_tran = refine_pose(
                    vae, denoiser, diffusion,
                    pose_r6d=gt_pose_r6d, tran=gt_tran,
                    imu_cond=None, noise_level=NOISE_LEVEL,
                    ddim_steps=max(10, NOISE_LEVEL // 10), device=device,
                )

                for b in range(B):
                    L = int(lengths[b])
                    gt_r6d = gt_pose_r6d[b, :L]
                    pred_r6d = pred_pose_r6d[b, :L]

                    gt_rot = r6d_to_rotation_matrix(gt_r6d.view(-1, 24, 6)).view(-1, 24, 3, 3)
                    pred_rot = r6d_to_rotation_matrix(pred_r6d.view(-1, 24, 6)).view(-1, 24, 3, 3)

                    gt_local = bodymodel.inverse_kinematics_R(gt_rot)
                    pred_local = bodymodel.inverse_kinematics_R(pred_rot)

                    error = mpjpe_evaluator(pred_local.view(L, -1), gt_local.view(L, -1))
                    all_mpjpe.append(error[0].item() * 100)
                    all_local_err.append(error[1].item())
                    all_global_err.append(error[2].item())

        mean_mpjpe = np.mean(all_mpjpe)
        mean_local = np.mean(all_local_err)
        mean_global = np.mean(all_global_err)

        results.append({
            "noise_level": NOISE_LEVEL, "samples": len(all_mpjpe),
            "mpjpe": mean_mpjpe, "local_angle": mean_local, "global_angle": mean_global,
        })

        print(f"\n{'='*60}")
        print(f"Noise level (t)   : {NOISE_LEVEL}")
        print(f"Samples evaluated : {len(all_mpjpe)}")
        print(f"Mean MPJPE        : {mean_mpjpe:.2f} cm")
        print(f"Mean Local Angle  : {mean_local:.2f}\u00b0")
        print(f"Mean Global Angle : {mean_global:.2f}\u00b0")
        print(f"{'='*60}")

    # Save results
    output_path = "latent_diffusion_results.txt"
    with open(output_path, "w") as f:
        f.write(f"VAE: {vae_checkpoint} | Denoiser: {denoiser_checkpoint}\n")
        for r in results:
            f.write(f"{'='*60}\n")
            f.write(f"Noise level (t)   : {r['noise_level']}\n")
            f.write(f"Samples evaluated : {r['samples']}\n")
            f.write(f"Mean MPJPE        : {r['mpjpe']:.2f} cm\n")
            f.write(f"Mean Local Angle  : {r['local_angle']:.2f}\u00b0\n")
            f.write(f"Mean Global Angle : {r['global_angle']:.2f}\u00b0\n")
        f.write(f"{'='*60}\n")
    print(f"\nResults saved to {output_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="train",
                        help="'train_ae', 'train_diffusion', 'train' (both), 'test', 'visualize', or 'mpjpe'")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default="vae_best.pth")
    parser.add_argument("--denoiser-checkpoint", type=str, default="latent_denoiser_best.pth")
    parser.add_argument("--noise-levels", type=str, default="50,80,100")
    parser.add_argument("--noise-level", type=int, default=100, help="Single noise level for visualisation")
    parser.add_argument("--num-vis-samples", type=int, default=2)
    parser.add_argument("--ae-epochs", type=int, default=50)
    parser.add_argument("--diff-epochs", type=int, default=100)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Dimensions
    POSE_DIM = 144
    TRAN_DIM = 3
    POSE_LATENT = 48
    TRAN_LATENT = 16
    LATENT_DIM = POSE_LATENT + TRAN_LATENT  # 64
    NUM_DIFFUSION_STEPS = 1000

    if args.mode == "visualize":
        visualize_results(
            vae_checkpoint=args.vae_checkpoint,
            denoiser_checkpoint=args.denoiser_checkpoint,
            noise_level=args.noise_level,
            num_vis_samples=args.num_vis_samples,
        )

    elif args.mode == "mpjpe":
        nlevels = [int(x) for x in args.noise_levels.split(",")]
        evaluate_mpjpe(
            vae_checkpoint=args.vae_checkpoint,
            denoiser_checkpoint=args.denoiser_checkpoint,
            noise_levels=nlevels,
        )

    elif args.mode == "test":
        # Run both: mpjpe then visualize
        nlevels = [int(x) for x in args.noise_levels.split(",")]
        evaluate_mpjpe(
            vae_checkpoint=args.vae_checkpoint,
            denoiser_checkpoint=args.denoiser_checkpoint,
            noise_levels=nlevels,
        )
        visualize_results(
            vae_checkpoint=args.vae_checkpoint,
            denoiser_checkpoint=args.denoiser_checkpoint,
            noise_level=nlevels[0],
            num_vis_samples=args.num_vis_samples,
        )

    else:
        # Load data
        datamodule = PoseDataModule(finetune=None)
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # ----- Stage 1: Autoencoder -----
        if args.mode in ("train_ae", "train"):
            print("\n" + "="*60)
            print("STAGE 1: Training Autoencoder (VAE)")
            print("="*60)

            vae = PoseTranslationVAE(
                pose_dim=POSE_DIM, tran_dim=TRAN_DIM,
                pose_latent_dim=POSE_LATENT, tran_latent_dim=TRAN_LATENT,
                d_model=256, nhead=8, enc_layers=4, dec_layers=4,
                dim_feedforward=1024, dropout=0.1,
            )
            print(f"VAE: pose({POSE_DIM})→{POSE_LATENT}, tran({TRAN_DIM})→{TRAN_LATENT}, "
                  f"combined latent={LATENT_DIM}")

            train_autoencoder(
                train_loader, val_loader, vae,
                num_epochs=args.ae_epochs, device=device,
                patience=10, kl_weight=1e-4,
            )

        # ----- Stage 2: Latent Diffusion -----
        if args.mode in ("train_diffusion", "train"):
            print("\n" + "="*60)
            print("STAGE 2: Training Latent Diffusion")
            print("="*60)

            # Load trained VAE
            vae = PoseTranslationVAE(
                pose_dim=POSE_DIM, tran_dim=TRAN_DIM,
                pose_latent_dim=POSE_LATENT, tran_latent_dim=TRAN_LATENT,
                d_model=256, nhead=8, enc_layers=4, dec_layers=4,
                dim_feedforward=1024, dropout=0.1,
            )
            vae_ckpt = args.vae_checkpoint
            print(f"Loading VAE from {vae_ckpt}")
            vae.load_state_dict(torch.load(vae_ckpt, map_location=device))

            diffusion = GaussianDiffusion(
                num_timesteps=NUM_DIFFUSION_STEPS, schedule="cosine",
            )

            denoiser = LatentDenoiser(
                latent_dim=LATENT_DIM, imu_dim=60,
                d_model=256, nhead=8, num_layers=6,
                dim_feedforward=1024, dropout=0.1,
            )
            print(f"Latent Denoiser: latent_dim={LATENT_DIM}, timesteps={NUM_DIFFUSION_STEPS}")

            train_latent_diffusion(
                train_loader, val_loader, vae, denoiser, diffusion,
                num_epochs=args.diff_epochs, device=device, patience=10,
            )
