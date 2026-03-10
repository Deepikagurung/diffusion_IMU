import os
import numpy as np
import torch
from argparse import ArgumentParser
import tqdm 

from config import *
from helpers import * 
import articulate as art
from constants import MODULES
from utils.model_utils import load_model
from data import PoseDataset
from models import MobilePoserNet

import argparse
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data import PoseDataset
import torch

ds = PoseDataset(fold='train', finetune=None)   # loads processed_datasets
pairs = []   # list of (pose_seq, tran_seq) per window/sequence
for i in range(len(ds)):
    item = ds[i]
    # training dataset returns (imu, pose, joint, tran, vel, contact)
    imu, pose, joint, tran = item[:4]
    # pose: (T, 6 * num_pred_joints) , tran: (T, 3)
    pairs.append((pose, tran))

# # example: concatenate all sequences (if lengths match) or keep as list
all_poses = torch.cat([p for p, t in pairs], dim=0)
all_trans  = torch.cat([t for p, t in pairs], dim=0)

device = "cpu"
x0 = torch.cat([all_poses.to(device), all_trans.to(device)], dim=-1)  # (B, 147)

import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiffusionMLP(nn.Module):
    def __init__(self, dim=147, cond_dim=147, hidden=1024):
        super().__init__()
        
        self.time_embed = TimeEmbedding(128)
        
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + 128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x_t, t, cond):
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(h)
    


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Example dataset
class PoseDataset(Dataset):
    def __init__(self, x, cond):
        # Ensure they are tensors
        self.x = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
        self.cond = cond if torch.is_tensor(cond) else torch.tensor(cond, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # idx could be a tensor if using batch sampler, convert to int
        if torch.is_tensor(idx):
            idx = idx.item()
        # print(self.x[idx].shape)
        return self.x[idx], self.cond[idx]

# Hyperparameters
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
batch_size = 32
learning_rate = 1e-4
num_epochs = 2
timesteps = 1000  # Number of diffusion steps

# Example: x0 and cond are your data tensors
# x0 = torch.cat([pose_t, tran_t], dim=-1)
# cond = your_condition_tensor  (same batch size)
cond = torch.zeros_like(x0)  # (batch_size, 147)
dataset = PoseDataset(x0, cond)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = DiffusionMLP(dim=147, cond_dim=147, hidden=1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Linear noise schedule (beta)
beta_start, beta_end = 1e-4, 0.02
beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


patience = 10
min_delta = 1e-5
best_loss = float("inf")
epochs_without_improvement = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for x_batch, cond_batch in dataloader:
        x_batch = x_batch.to(device)
        cond_batch = cond_batch.to(device)

        # Sample random timestep t
        t = torch.randint(0, timesteps, (x_batch.size(0),), device=device).long()

        # Sample noise
        noise = torch.randn_like(x_batch)

        # Generate noisy input x_t
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])[:, None]
        x_t = sqrt_alpha_bar * x_batch + sqrt_one_minus_alpha_bar * noise

        # Predict noise
        noise_pred = model(x_t, t.float(), cond_batch)

        # Compute loss
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.6f}")

    # ---- Early Stopping Check ----
    if epoch_loss < best_loss - min_delta:
        best_loss = epoch_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "model_best.pt")
        best_model_state = model.state_dict()  # Save best weights
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epochs")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model weights.")