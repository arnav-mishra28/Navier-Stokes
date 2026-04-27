"""
=============================================================================
Unified Training Loop
Handles training for PINN, FNO, DeepONet, Surrogate, and Turbulence NN.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
from typing import Dict, Optional, List, Callable
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class UnifiedTrainer:
    """
    Unified training infrastructure for all NS neural network models.
    
    Features:
        - Multi-model support (PINN, FNO, DeepONet, Surrogate)
        - Mixed precision training (AMP)
        - Learning rate scheduling (cosine annealing, warmup)
        - Gradient clipping
        - Checkpointing
        - TensorBoard logging
        - Early stopping
        - Curriculum learning for PINNs
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str = "fno",  # "pinn", "fno", "deeponet", "surrogate"
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.model_type = model_type
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Mixed precision
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Paths
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        print(f"Trainer initialized: {model_type} on {self.device}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")
    
    def setup_scheduler(self, epochs: int, warmup_epochs: int = 10):
        """Setup cosine annealing with warmup."""
        # Guard against zero-division when epochs <= warmup_epochs
        warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
        
        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max(epoch / warmup_epochs, 1e-6)
            remaining = epochs - warmup_epochs
            if remaining <= 0:
                return 1.0
            progress = (epoch - warmup_epochs) / remaining
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_fno(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        loss_fn: Optional[nn.Module] = None,
        gradient_clip: float = 1.0,
        save_every: int = 50,
        physics_weight: float = 0.0,
    ):
        """
        Train FNO model with optional physics-informed loss.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        self.setup_scheduler(epochs)
        
        print(f"\n{'='*60}")
        print(f"Training FNO: {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                if len(batch) == 3:
                    x, y, params = batch
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = self.model(x)
                        loss = nn.functional.mse_loss(pred, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(x)
                    loss = nn.functional.mse_loss(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_history.append({'epoch': epoch, 'loss': avg_loss})
            
            if self.scheduler:
                self.scheduler.step()
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                self.val_history.append({'epoch': epoch, 'loss': val_loss})
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch+1:4d}/{epochs} | Train: {avg_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val: {val_loss:.6f}"
                msg += f" | LR: {lr:.2e}"
                print(msg)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            self.epoch = epoch + 1
        
        self.save_checkpoint('final_model.pt')
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")
    
    def train_pinn(
        self,
        collocation_data: Dict[str, torch.Tensor],
        epochs: int = 10000,
        loss_module: Optional[nn.Module] = None,
        resample_interval: int = 1000,
        gradient_clip: float = 1.0,
    ):
        """
        Train PINN with collocation points and physics constraints.
        """
        from training.losses import PhysicsInformedLoss
        
        if loss_module is None:
            loss_module = PhysicsInformedLoss(nu=self.model.nu).to(self.device)
        
        # Move data to device
        data = {k: v.to(self.device) for k, v in collocation_data.items()}
        
        # Add loss module params to optimizer
        all_params = list(self.model.parameters()) + list(loss_module.parameters())
        self.optimizer = optim.Adam(all_params, lr=1e-3)
        self.setup_scheduler(epochs, warmup_epochs=100)
        
        print(f"\n{'='*60}")
        print(f"Training PINN: {epochs} epochs")
        print(f"  Collocation: {len(data['x_pde'])} | BC: {len(data['x_bc'])} | IC: {len(data['x_ic'])}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Compute physics-informed loss
            losses = loss_module(
                self.model,
                data['x_pde'], data['y_pde'], data['t_pde'],
                data['x_bc'], data['y_bc'], data['t_bc'],
                data['u_bc'], data['v_bc'],
            )
            
            loss = losses['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.train_history.append({
                'epoch': epoch,
                'total': loss.item(),
                **{k: v.item() for k, v in losses.items() if k != 'total' and isinstance(v, torch.Tensor)}
            })
            
            if (epoch + 1) % 500 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1:5d}/{epochs}"
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        msg += f" | {k}: {v.item():.4e}"
                print(msg)
            
            # Resample collocation points periodically
            if resample_interval > 0 and (epoch + 1) % resample_interval == 0:
                n_col = len(data['x_pde'])
                data['x_pde'] = torch.rand(n_col, 1, device=self.device) * 2 * np.pi
                data['y_pde'] = torch.rand(n_col, 1, device=self.device) * 2 * np.pi
                data['t_pde'] = torch.rand(n_col, 1, device=self.device)
                print(f"  [Resampled {n_col} collocation points]")
            
            self.epoch = epoch + 1
        
        self.save_checkpoint('pinn_final.pt')
        print(f"\nPINN training complete.")
    
    def train_deeponet(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        gradient_clip: float = 1.0,
        save_every: int = 50,
    ):
        """
        Train DeepONet model.
        
        Data format: each batch = (branch_input, trunk_input, target)
            branch_input: (B, n_sensors) — input function values
            trunk_input:  (B, 2/3)       — query coordinates (x,y[,t])
            target:       (B, output_dim) — solution values
        """
        self.setup_scheduler(epochs)
        
        print(f"\n{'='*60}")
        print(f"Training DeepONet: {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                if len(batch) == 3:
                    branch_in, trunk_in, target = [
                        b.to(self.device) for b in batch
                    ]
                else:
                    # Fallback: (input, target) pairs
                    x, target = batch[0].to(self.device), batch[1].to(self.device)
                    branch_in = x
                    trunk_in = torch.zeros(x.shape[0], 2, device=self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(branch_in, trunk_in)
                loss = nn.functional.mse_loss(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_history.append({'epoch': epoch, 'loss': avg_loss})
            
            if self.scheduler:
                self.scheduler.step()
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_deeponet(val_loader)
                self.val_history.append({'epoch': epoch, 'loss': val_loss})
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_deeponet.pt')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch+1:4d}/{epochs} | Train: {avg_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val: {val_loss:.6f}"
                msg += f" | LR: {lr:.2e}"
                print(msg)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'deeponet_epoch_{epoch+1}.pt')
            
            self.epoch = epoch + 1
        
        self.save_checkpoint('deeponet_final.pt')
        print(f"\nDeepONet training complete. Best val loss: {self.best_val_loss:.6f}")
    
    def _validate_deeponet(self, val_loader: DataLoader) -> float:
        """Run validation pass for DeepONet."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    branch_in, trunk_in, target = [
                        b.to(self.device) for b in batch
                    ]
                else:
                    x, target = batch[0].to(self.device), batch[1].to(self.device)
                    branch_in = x
                    trunk_in = torch.zeros(x.shape[0], 2, device=self.device)
                pred = self.model(branch_in, trunk_in)
                loss = nn.functional.mse_loss(pred, target)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)
    
    def train_surrogate(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        gradient_clip: float = 1.0,
        save_every: int = 50,
    ):
        """
        Train U-Net surrogate model.
        
        Data format: each batch = (input_fields, target_fields)
            input_fields:  (B, C_in, H, W) — initial state
            target_fields: (B, C_out, H, W) — predicted next state
        """
        self.setup_scheduler(epochs)
        
        print(f"\n{'='*60}")
        print(f"Training U-Net Surrogate: {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(x)
                
                # Relative L2 loss for better scale-invariance
                diff = pred - y
                loss = torch.mean(diff**2) / (torch.mean(y**2) + 1e-8)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_history.append({'epoch': epoch, 'loss': avg_loss})
            
            if self.scheduler:
                self.scheduler.step()
            
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                self.val_history.append({'epoch': epoch, 'loss': val_loss})
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_surrogate.pt')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch+1:4d}/{epochs} | Train: {avg_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val: {val_loss:.6f}"
                msg += f" | LR: {lr:.2e}"
                print(msg)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'surrogate_epoch_{epoch+1}.pt')
            
            self.epoch = epoch + 1
        
        self.save_checkpoint('surrogate_final.pt')
        print(f"\nSurrogate training complete. Best val loss: {self.best_val_loss:.6f}")
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation pass."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                pred = self.model(x)
                loss = nn.functional.mse_loss(pred, y)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training curves with publication-quality dark styling."""
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Publication-quality dark styling
        plt.rcParams.update({
            'figure.facecolor': '#0d1117',
            'axes.facecolor': '#161b22',
            'savefig.facecolor': '#0d1117',
            'text.color': '#c9d1d9',
            'axes.labelcolor': '#c9d1d9',
            'xtick.color': '#8b949e',
            'ytick.color': '#8b949e',
            'axes.edgecolor': '#30363d',
            'legend.facecolor': '#161b22',
            'legend.edgecolor': '#30363d',
            'figure.dpi': 150,
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training loss
        train_losses = [h['loss'] if 'loss' in h else h.get('total', 0) for h in self.train_history]
        axes[0].semilogy(train_losses, color='#58a6ff', lw=2, label='Train', alpha=0.9)
        
        if self.val_history:
            val_losses = [h['loss'] for h in self.val_history]
            axes[0].semilogy(val_losses, color='#f97583', lw=2, ls='--', label='Validation', alpha=0.9)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress', color='#79c0ff')
        axes[0].legend()
        axes[0].grid(True, alpha=0.15, color='#30363d')
        
        # Component losses (for PINN)
        if self.train_history and 'pde' in self.train_history[0]:
            colors = ['#58a6ff', '#f97583', '#7ee787', '#d2a8ff', '#ffa657']
            keys = [k for k in self.train_history[0].keys() if k not in ['epoch', 'total']]
            for ci, key in enumerate(keys):
                vals = [h.get(key, 0) for h in self.train_history]
                if any(v > 0 for v in vals):
                    axes[1].semilogy(vals, color=colors[ci % len(colors)], lw=2, label=key, alpha=0.9)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Component')
            axes[1].set_title('Loss Components', color='#79c0ff')
            axes[1].legend()
            axes[1].grid(True, alpha=0.15, color='#30363d')
        else:
            axes[1].text(0.5, 0.5, 'No component data', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=14, color='#8b949e')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Training plot saved: {save_path}")
        
        # Safe show: don't crash in headless environments
        try:
            backend = matplotlib.get_backend().lower()
            if 'agg' not in backend:
                plt.show()
            else:
                plt.close(fig)
        except Exception:
            plt.close(fig)
