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

            # print("IMU input shape:", imu_input.shape)
            # print("Window length used:", data_len)
            # print("Pose shape BEFORE windowing:", pose.shape)

            # splits = torch.split(pose, data_len)
            # splits = torch.split(pose, data_len)
            # splits = splits[:-1] if splits[-1].shape[0] < data_len else splits   #added to have only similar length windows

            # print("Number of pose windows created:", len(splits))
            # print("First pose window shape:", splits[0].shape)
            # print("Last pose window shape:", splits[-1].shape)

            # if len(splits) > 1:
            #     overlap = torch.allclose(splits[0][-1], splits[1][0])
            #     print("Do windows overlap?")
            #     print("Last frame window0 == first frame window1 →", overlap)

            # print("="*50)

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


class TemporalTransformerDenoiser(nn.Module):
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
        """Temporal Transformer denoiser for pose + translation.

        Inputs:  pose (B, T, pose_dim), tran (B, T, tran_dim)
        Outputs: denoised pose (B, T, pose_dim), denoised tran (B, T, tran_dim)
        """
        super().__init__()
        self.pose_dim = pose_dim
        self.tran_dim = tran_dim
        combined_dim = pose_dim + tran_dim

        self.in_proj = nn.Linear(combined_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # separate output heads for pose and translation
        self.pose_head = nn.Linear(d_model, pose_dim)
        self.tran_head = nn.Linear(d_model, tran_dim)

    def forward(self, pose: torch.Tensor, tran: torch.Tensor):
        # pose: (B, T, pose_dim), tran: (B, T, tran_dim)
        x = torch.cat([pose, tran], dim=-1)  # (B, T, pose_dim + tran_dim)
        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (B, T, d_model)
        return self.pose_head(h), self.tran_head(h)  # pose_pred, tran_pred
    
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
            t = outputs["trans"].to(device)
            lengths = torch.as_tensor(output_lengths["poses"], device=device)

            B, T, F = x.shape

            # NO noise during validation
            pose_pred, tran_pred = model(x, t)

            pose_loss = nn.functional.mse_loss(pose_pred, x, reduction="none")
            tran_loss = nn.functional.mse_loss(tran_pred, t, reduction="none")

            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            pose_mask = mask.unsqueeze(-1).float()
            tran_mask = mask.unsqueeze(-1).float()

            masked_pose_loss = (pose_loss * pose_mask).sum()
            masked_tran_loss = (tran_loss * tran_mask).sum()
            masked_loss = masked_pose_loss + masked_tran_loss
            num_valid = pose_mask.sum()

            # compute MPJ error for valid frames
            if num_valid > 0:
                if F % 9 == 0:
                    in_rep = 9
                elif F % 6 == 0:
                    in_rep = 6
                else:
                    raise RuntimeError(f"Unknown pose feature dim F={F}")

                num_joints = F // in_rep
                pose_pred_flat = pose_pred.view(B * T, num_joints, in_rep)
                pose_true_flat = x.view(B * T, num_joints, in_rep)
                valid_idx = mask.view(-1).bool()

                if valid_idx.any():
                    if in_rep == 6:
                        pose_pred_mat = r6d_to_rotation_matrix(pose_pred_flat)
                        pose_true_mat = r6d_to_rotation_matrix(pose_true_flat)
                    else:
                        pose_pred_mat = pose_pred_flat.view(-1, num_joints, 3, 3)
                        pose_true_mat = pose_true_flat.view(-1, num_joints, 3, 3)

                    pose_pred_mat = pose_pred_mat[valid_idx]
                    pose_true_mat = pose_true_mat[valid_idx]

                    # model joint count
                    try:
                        j, _ = mpj_eval.model.get_zero_pose_joint_and_vertex(None)
                        model_num_j = j.shape[1] if j.dim() == 3 else j.shape[0]
                    except Exception:
                        model_num_j = pose_pred_mat.shape[1]

                    # expand reduced -> full (fix for shape mismatch)
                    if pose_pred_mat.shape[1] != model_num_j:
                        N_valid = pose_pred_mat.shape[0]
                        pose_pred_full = reduced_pose_to_full(pose_pred_mat.unsqueeze(1)).view(N_valid, model_num_j, 3, 3)
                        pose_true_full = reduced_pose_to_full(pose_true_mat.unsqueeze(1)).view(N_valid, model_num_j, 3, 3)
                        pose_pred_mat = pose_pred_full
                        pose_true_mat = pose_true_full

                    mpj_tensor = mpj_eval(pose_pred_mat, pose_true_mat)
                    val_mpj_sum += mpj_tensor[0].item() * valid_idx.sum().item()

            val_weighted_loss += masked_loss.item()
            val_total_frames += num_valid.item()

    if val_total_frames > 0:
        val_loss = val_weighted_loss / val_total_frames
        val_mpj = val_mpj_sum / val_total_frames
    else:
        val_loss = 0.0
        val_mpj = 0.0

    print(f"Validation Loss: {val_loss:.6f} | MPJPE: {val_mpj:.6f}")

    return val_loss, val_mpj




def training(train_loader,val_loader, model, num_epochs=1, device=None, patience: int = 10, min_delta: float = 1e-4):
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
            # Get Pose + Translation Data
            # -------------------------------------------------
            x = outputs["poses"].to(device)
            t = outputs["trans"].to(device)
            lengths = torch.as_tensor(
                output_lengths["poses"],
                device=device
            )

            B, T, F = x.shape

            # -------------------------------------------------
            # Add Noise (Denoising Training)
            # -------------------------------------------------
            pose_noise = torch.randn_like(x) * 0.01
            tran_noise = torch.randn_like(t) * 0.01
            noisy_x = x + pose_noise
            noisy_t = t + tran_noise

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------
            pose_pred, tran_pred = model(noisy_x, noisy_t)

            pose_loss_matrix = criterion(pose_pred, x)
            tran_loss_matrix = criterion(tran_pred, t)

            # -------------------------------------------------
            # Mask Padding Frames
            # -------------------------------------------------
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            pose_mask = mask.unsqueeze(-1).float()
            tran_mask = mask.unsqueeze(-1).float()

            masked_pose_loss = (pose_loss_matrix * pose_mask).sum()
            masked_tran_loss = (tran_loss_matrix * tran_mask).sum()
            masked_loss = masked_pose_loss + masked_tran_loss
            num_valid = pose_mask.sum()

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
        print(f"Epoch {epoch}")
        print(f"Epoch Loss: {epoch_loss:.6f}")
        print("===================================")

        # -------------------------------------------------
        # Early Stopping
        # -------------------------------------------------
        if best_epoch_loss is None or epoch_loss < best_epoch_loss - min_delta:
            best_epoch_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "temporal_transformer_model_best_both.pth")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # mpj_eval = MeanPerJointErrorEvaluator(
        # official_model_file=paths.smpl_file,
        # rep=RotationRepresentation.ROTATION_MATRIX,
        # device=device
        #     )

        # val_loss, val_mpj = validate(
        #     val_loader=val_loader,
        #     model=model,
        #     device=device,
        #     mpj_eval=mpj_eval
        # )
        # print(f"Validation Loss: {val_loss:.6f} | MPJPE: {val_mpj:.6f}")
    torch.save(model.state_dict(), f"temporal_transformer_model_both.pth")




# runnmning 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"


    datamodule = PoseDataModule(finetune=None)

    # IMPORTANT: must call setup
    datamodule.setup(stage='fit')

    # Get train loader
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()


    # model = TemporalDenoiseUNet(feature_dim=144).to(device)
    model = TemporalTransformerDenoiser(
        pose_dim=144,
        tran_dim=3,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)
    print('Model initialized with pose_dim=144, tran_dim=3')

    training(train_loader=train_loader,val_loader=val_loader, model=model, num_epochs=30, patience=5, device=device)



    


