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
    
# model 

# -------------------------------------------------
# Basic MLP Block (deeper)
# -------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout: float = 0.1, depth: int = 2):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(out_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Temporal Downsample (T -> T/2)
# -------------------------------------------------
class TemporalDown(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        B, T, F = x.shape

        # pad if odd length
        if T % 2 == 1:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))

        x = x.view(B, -1, 2, F)      # pair timesteps
        x = x.reshape(B, -1, 2 * F)  # concat
        x = self.proj(x)
        return x


# -------------------------------------------------
# Temporal Upsample (T -> T*2)
# -------------------------------------------------
class TemporalUp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x):
        B, T, F = x.shape
        x = self.proj(x)
        x = x.view(B, T * 2, F)
        return x


# -------------------------------------------------
# Temporal Denoising UNet (higher capacity)
# -------------------------------------------------
class TemporalDenoiseUNet(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        features: Sequence[int] = (256, 512, 512),
        dropout: float = 0.1,
    ):
        """
        feature_dim: input feature size F
        Output: reconstructed (B, T, F)
        """
        super().__init__()

        fea = list(features)

        # ------------- Encoder -------------
        self.enc0 = MLPBlock(feature_dim, fea[0], dropout=dropout, depth=2)
        self.down1 = TemporalDown(fea[0])
        self.enc1 = MLPBlock(fea[0], fea[1], dropout=dropout, depth=2)

        self.down2 = TemporalDown(fea[1])
        self.enc2 = MLPBlock(fea[1], fea[2], dropout=dropout, depth=2)

        # ------------- Decoder -------------
        self.up2 = TemporalUp(fea[2])
        self.dec2 = MLPBlock(fea[2] + fea[1], fea[1], dropout=dropout, depth=2)

        self.up1 = TemporalUp(fea[1])
        self.dec1 = MLPBlock(fea[1] + fea[0], fea[0], dropout=dropout, depth=2)

        self.final = nn.Linear(fea[0], feature_dim)

    def forward(self, x):
        """
        x: (B, T, F)
        returns reconstructed x_hat: (B, T, F)
        """
        # Encoder
        x0 = self.enc0(x)

        x1 = self.down1(x0)
        x1 = self.enc1(x1)

        x2 = self.down2(x1)
        x2 = self.enc2(x2)

        # Decoder
        u2 = self.up2(x2)

        if u2.shape[1] != x1.shape[1]:
            diff = x1.shape[1] - u2.shape[1]
            u2 = torch.nn.functional.pad(u2, (0, 0, 0, diff))

        u2 = torch.cat([u2, x1], dim=-1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)

        if u1.shape[1] != x0.shape[1]:
            diff = x0.shape[1] - u1.shape[1]
            u1 = torch.nn.functional.pad(u1, (0, 0, 0, diff))

        u1 = torch.cat([u1, x0], dim=-1)
        u1 = self.dec1(u1)

        out = self.final(u1)
        return out
    

#training code 
def training(train_loader, model, num_epochs=1, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss(reduction="none")

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
            # Optional: Add Noise
            # -------------------------------------------------
            noise = torch.randn_like(x) * 0.01
            noisy_x = x + noise

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------
            pred = model(noisy_x)

            loss_matrix = criterion(pred, x)

            # -------------------------------------------------
            # Mask Padding Frames
            # -------------------------------------------------
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()

            masked_loss = (loss_matrix * mask).sum()
            num_valid = mask.sum()

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

    # Save model after training
    torch.save(model.state_dict(), "temporal_unet_new.pth")
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

            # ---------------- Forward (no noise) ----------------
            pred = model(x)

            loss_matrix = F.mse_loss(pred, x, reduction="none")

            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()

            masked_loss = (loss_matrix * mask).sum()
            num_valid = mask.sum()

            # ---------------- MPJPE Computation ----------------
            if num_valid > 0:

                if Fdim % 9 == 0:
                    in_rep = 9
                elif Fdim % 6 == 0:
                    in_rep = 6
                else:
                    raise RuntimeError(f"Unknown pose feature dim F={Fdim}")

                num_joints = Fdim // in_rep

                pose_pred = pred.view(B * T, num_joints, in_rep)
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

model = TemporalDenoiseUNet(feature_dim=144).to(device)
training(train_loader=train_loader, model=model, num_epochs=100)

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