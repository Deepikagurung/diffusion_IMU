import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import from your project
from data import PoseDataset, pad_seq
from config import paths, datasets


# ============================================================================
# 1. DIFFUSION CONFIGURATION
# ============================================================================
class DiffusionConfig:
    """Configuration for the diffusion process."""
    timesteps = 1000  # Number of diffusion steps
    beta_start = 0.0001  # Initial noise schedule variance
    beta_end = 0.02  # Final noise schedule variance
    
    @staticmethod
    def linear_beta_schedule(timesteps):
        """Create linear noise schedule: variance increases over time."""
        betas = torch.linspace(DiffusionConfig.beta_start, 
                               DiffusionConfig.beta_end, 
                               timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alphas_cumprod


# ============================================================================
class TimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.
    Similar to Transformer positional encoding but for time.
    
    Input: scalar timestep t (0 to 999)
    Output: vector of size embedding_dim (e.g., 128)
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Precompute sinusoidal frequencies
        # Creates frequencies: [1, 10, 100, 1000, ...] for different dimensions
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Optional: learnable projection after sinusoidal encoding
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: tensor of shape [batch_size] containing timesteps (0-999)
        
        Returns:
            embeddings of shape [batch_size, embedding_dim]
        """
        # Ensure t is on the correct device and is a 1D tensor
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Compute sinusoidal positional encoding
        # For each dimension d: sin(t / 10000^(2i/d_model)) or cos(...)
        freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq)
        
        # Interleave sin and cos: [sin(f0), sin(f1), ..., cos(f0), cos(f1), ...]
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        
        # Project through MLP for learnable transformation
        return self.mlp(emb)


# ============================================================================
# 3. CONDITIONAL DIFFUSION MODEL - Core architecture
# ============================================================================
class ConditionalDiffusionMLP(nn.Module):
    """
    Conditional Diffusion Model using MLP architecture.
    
    This model learns to reverse the diffusion process:
    - Takes noisy pose+translation (x_t) at timestep t
    - Uses IMU sensor data as conditioning
    - Predicts the noise that was added
    
    Architecture:
    [x_t (147) + cond (60) + t_emb (128)] → [hidden (512)] → [output (147)]
    """
    def __init__(self, 
                 data_dim: int = 147,        # pose (144) + translation (3)
                 cond_dim: int = 60,         # IMU inputs (15 acc + 45 ori)
                 hidden_dim: int = 512,
                 time_embedding_dim: int = 128):
        super().__init__()
        
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding layer
        self.time_embed = TimeEmbedding(time_embedding_dim)
        
        # Input projection: concatenate [x_t, cond, t_emb]
        input_dim = data_dim + cond_dim + time_embedding_dim
        
        # Main network: 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # Smoother activation than ReLU
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, data_dim)  # Predict noise
        )
        
        # Initialize weights for better training stability
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization for better convergence."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                x_t: torch.Tensor,      # Noisy data at timestep t: [batch, 147]
                t: torch.Tensor,        # Timestep indices: [batch]
                cond: torch.Tensor      # Conditioning input (IMU): [batch, 60]
                ) -> torch.Tensor:
        """
        Predict noise from noisy input.
        
        Args:
            x_t: noisy pose+translation at timestep t, shape [batch_size, 147]
            t: timestep indices, shape [batch_size], values in [0, 999]
            cond: IMU conditioning data, shape [batch_size, 60]
        
        Returns:
            predicted noise, shape [batch_size, 147]
        """
        # Embed timestep information
        t_emb = self.time_embed(t)  # [batch_size, 128]
        
        # Concatenate noisy data, conditioning, and time embedding
        # This allows the model to know: what's the current state, what should guide it, when are we
        combined = torch.cat([x_t, cond, t_emb], dim=-1)  # [batch_size, 147+60+128=335]
        
        # Predict noise
        noise_pred = self.net(combined)  # [batch_size, 147]
        
        return noise_pred
    
class DiffusionDataset(Dataset):
    """
    Wrapper around PoseDataset to prepare data for diffusion training.
    
    Converts pose+translation to targets and uses IMU data as conditioning.
    """
    def __init__(self, fold: str = 'train'):
        # Load the pose dataset (handles all preprocessing)
        # fold = 'test'
        finetune = 'dip'
        # self.pose_dataset = PoseDataset(fold=fold, finetune=finetune)
        self.pose_dataset = PoseDataset( evaluate=finetune)
        
    def __len__(self):
        return len(self.pose_dataset)
    
    def __getitem__(self, idx):
        """
        Returns tuple of (x0, condition) for diffusion training.
        """
        # batch = self.pose_dataset[idx]
        
        # # batch format from PoseDataset.__getitem__:
        # # (imu, pose, joint, tran) for eval/finetune
        # # (imu, pose, joint, tran, vel, contact) for training
        
        # imu = batch[0]          # IMU inputs: [seq_len, 60] (15 acc + 45 ori)
        # pose = batch[1]         # Pose in R6D: [seq_len, 144]
        # tran = batch[3]         # Translation: [seq_len, 3]
        
        # # Concatenate pose and translation as target for diffusion
        # # This is what the model will learn to generate
        # x0 = torch.cat([pose, tran], dim=-1)  # [seq_len, 147]
        
        # # Use IMU as condition (this guides the generation process)
        # cond = imu  # [seq_len, 60]
        
        pairs = []   # list of (pose_seq, tran_seq) per window/sequence
        for i in range(len(self.pose_dataset)):
            item = self.pose_dataset[i]
            # training dataset returns (imu, pose, joint, tran, vel, contact)
            imu, pose, joint, tran = item[:4]
            # pose: (T, 6 * num_pred_joints) , tran: (T, 3)
            pairs.append((imu,pose, tran))


        # # example: concatenate all sequences (if lengths match) or keep as list
        all_imus = torch.cat([i for i,p, t in pairs], dim=0)
        all_poses = torch.cat([p for i,p, t in pairs], dim=0)
        all_trans  = torch.cat([t for i,p, t in pairs], dim=0)

        x0 = torch.cat([all_poses, all_trans], dim=-1)  # [seq_len, 147]
        
        # # Use IMU as condition (this guides the generation process)
        cond = all_imus  # [seq_len, 60]        
        return x0, cond

# ============================================================================
# 5. TRAINING FUNCTION
# ============================================================================
def train_diffusion_model(epochs: int = 100, 
                         batch_size: int = 32,
                         learning_rate: float = 1e-3,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Main training loop for conditional diffusion model.
    
    Process:
    1. Load pose+IMU data from preprocessed AMASS
    2. For each epoch:
       - Sample random timesteps t
       - Add noise to targets (forward diffusion)
       - Train model to predict the noise (reverse diffusion)
       - Monitor validation loss and save best checkpoint
    """
    
    print(f"Training on device: {device}")
    device = torch.device(device)
    
    # ========================================================================
    # SETUP: Create diffusion config, model, and data loaders
    # ========================================================================
    config = DiffusionConfig()
    betas, alphas_cumprod = DiffusionConfig.linear_beta_schedule(config.timesteps)
    
    # Move noise schedule to device for efficient computation
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    
    # Initialize model
    model = ConditionalDiffusionMLP(
        data_dim=147,
        cond_dim=60,
        hidden_dim=512,
        time_embedding_dim=128
    ).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Load data
    print("\nLoading dataset...")
    dataset = DiffusionDataset(fold='test')
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders with custom padding for sequences
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=pad_seq,  # Pads variable-length sequences
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=pad_seq,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for (imu, lengths), (outputs, output_lengths) in pbar:
            # ============================================================
            # Extract data from batch (handle padding)
            # ============================================================
            poses = outputs['poses'].to(device)        # [batch, seq_len, 144]
            trans = outputs['trans'].to(device)        # [batch, seq_len, 3]
            imu = imu.to(device)                       # [batch, seq_len, 60]
            
            # Concatenate pose and translation
            x0 = torch.cat([poses, trans], dim=-1)    # [batch, seq_len, 147]
            
            # Flatten to treat each timestep independently
            batch_size_actual = x0.shape[0]
            seq_len = x0.shape[1]
            
            x0_flat = x0.reshape(-1, 147)              # [batch*seq_len, 147]
            imu_flat = imu.reshape(-1, 60)             # [batch*seq_len, 60]
            
            # ============================================================
            # FORWARD DIFFUSION: Add noise
            # ============================================================
            # Sample random timesteps for each sample
            t = torch.randint(0, config.timesteps, (x0_flat.shape[0],), device=device)
            
            # Sample random Gaussian noise
            noise = torch.randn_like(x0_flat)
            
            # Get noise schedule values for the selected timesteps
            # sqrt_alpha_cumprod_t: how much original signal to keep
            # sqrt_one_minus_alpha_cumprod_t: how much noise to add
            sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t]).unsqueeze(-1)
            
            # Create noisy version: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
            # This is the key diffusion equation
            x_t = sqrt_alpha_cumprod_t * x0_flat + sqrt_one_minus_alpha_cumprod_t * noise
            
            # ============================================================
            # REVERSE PROCESS: Predict noise
            # ============================================================
            noise_pred = model(x_t, t, imu_flat)
            
            # ============================================================
            # LOSS: MSE between predicted and actual noise
            # ============================================================
            # The model learns to predict what noise was added
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            # ============================================================
            # BACKWARD PASS
            # ============================================================
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion (important for diffusion models!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(train_losses[-100:])})
        
        # ====================================================================
        # VALIDATION
        # ====================================================================
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for (imu, lengths), (outputs, output_lengths) in val_loader:
                poses = outputs['poses'].to(device)
                trans = outputs['trans'].to(device)
                imu = imu.to(device)
                
                x0 = torch.cat([poses, trans], dim=-1)
                
                x0_flat = x0.reshape(-1, 147)
                imu_flat = imu.reshape(-1, 60)
                
                # Random timesteps
                t = torch.randint(0, config.timesteps, (x0_flat.shape[0],), device=device)
                noise = torch.randn_like(x0_flat)
                
                sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t]).unsqueeze(-1)
                
                x_t = sqrt_alpha_cumprod_t * x0_flat + sqrt_one_minus_alpha_cumprod_t * noise
                noise_pred = model(x_t, t, imu_flat)
                
                loss = nn.functional.mse_loss(noise_pred, noise)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # ====================================================================
        # EARLY STOPPING: Save best model
        # ====================================================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = paths.eval_dir / f"diffusion_model_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"✓ Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: validation loss did not improve for {patience} epochs")
                break
        
        # Update learning rate
        scheduler.step(avg_val_loss)
    
    print("\n" + "="*60)
    print(f"Training complete! Best val loss: {best_val_loss:.6f}")
    print("="*60)
    
    return model




if __name__ == "__main__":
    # Train the diffusion model
    model = train_diffusion_model(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        device="cuda"
    )
    
    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)