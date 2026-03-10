import os
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob

from articulate.model import ParametricModel
from articulate import math
from config import paths, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from config import *
from helpers import * 
from data import PoseDataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.functional as F


device = "cuda"

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
def _dataloader(dataset):
        return DataLoader(
            dataset, 
            batch_size=32, 
            collate_fn=pad_seq, 
            num_workers=0, 
            shuffle=True, 
            drop_last=True
        )


class TemporalDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        B, T, D = x.shape
        
        # Embed time
        t_emb = self.time_mlp(t)  # [B, hidden_dim]
        t_expanded = t_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, hidden_dim]
        
        # Move t_expanded to same device as x
        t_expanded = t_expanded.to(x.device)
        
        x_in = torch.cat([x, t_expanded], dim=-1)
        x_flat = x_in.view(B * T, -1)
        
        out = self.mlp(x_flat)
        
        return out.view(B, T, D)

def add_noise_sequence(x0, t):
    """
    x0: (B, L, 147)
    t : (B,)
    """

    device = x0.device  #  get device from input

    noise = torch.randn_like(x0).to(device)

    alpha_t = alpha_bar[t].view(-1, 1, 1).to(device)

    noisy = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    return noisy, noise

# =====================================
# Training Step
# =====================================
def train_step(model, optimizer, x0):
    """
    x0: (B, L, 147)
    """

    B = x0.shape[0]

    # Sample random timesteps per sequence
    t = torch.randint(0, TIMESTEPS, (B,), device=device)

    # Normalize timestep to embedding scale
    t_emb = t.float().unsqueeze(1) / TIMESTEPS
    t_emb = t_emb.to(device)  # Move to device

    # Add noise
    noisy_x, noise = add_noise_sequence(x0, t)

    # Forward pass
    pred_noise = model(noisy_x, t_emb)

    # Noise prediction loss
    loss = F.mse_loss(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()

    # Optional: gradient clipping (IMPORTANT for long sequences)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return loss.item()





dataset_name = "dip"
# dataset = PoseDataset( evaluate=dataset_name)   # loads processed_datasets

dataset = PoseDataset(fold='test', evaluate=dataset_name)   # loads processed_datasets
# dataset = PoseDataset(fold='train', finetune=None)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_dataloader_ = _dataloader(train_dataset)
val_dataloader_ = _dataloader(val_dataset)  

all_pose_trans = []

for (inputs, input_lengths), (outputs, output_lengths) in train_dataloader_:

    poses = outputs["poses"]   # (B, S, 144)
    trans = outputs["trans"]   # (B, S, 3)

    
    pose_trans = torch.cat([poses, trans], dim=-1)  # (B, S, 147)

    all_pose_trans.append(pose_trans)



padded = []
x = all_pose_trans 
max_L = max(x[i].shape[1] for i in range(len(train_dataloader_)))
for i in range(len(train_dataloader_)):
    seq = x[i]  # (B, L_i, 147)
    L_i = seq.shape[1]

    pad_len = max_L - L_i

    padded_seq = F.pad(seq, (0,0,0,pad_len))  # pad time dimension
    padded.append(padded_seq)

x = torch.stack(padded, dim=0)  # (6, B, max_L, 147)


x = x.reshape(-1, max_L, 147)  # (6B, max_L,147)


TIMESTEPS = 1000
LR = 1e-4
input_dim = 147  # poses (144) + trans (3)
model = TemporalDenoiser(input_dim=input_dim, hidden_dim=128).to(device)
betas = torch.linspace(1e-4, 0.02, TIMESTEPS)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

alpha_bar = alpha_bar.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

x = x.to(device)  # Move data to CUDA

from tqdm import tqdm

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(range(0, x.shape[0], 32), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx in pbar:
        x_batch = x[batch_idx:batch_idx+32]
        loss = train_step(model, optimizer, x_batch)
        total_loss += loss
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss:.6f}'})
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}\n")

print("Training complete!")