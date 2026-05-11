"""Turbulence Discovery Training Pipeline"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
from typing import Dict, Optional, List
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TurbulenceDiscoveryTrainer:
    """
    Orchestrates the full turbulence discovery training pipeline.
    """
    
    def __init__(
        self,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints/discovery",
        log_dir: str = "logs/discovery",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'autoencoder': [],
            'latent_ode': [],
            'blowup': [],
        }
        
        print(f"  Discovery Trainer initialized on {self.device}")
    
    # Autoencoder Training
    
    def train_autoencoder(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        in_channels: int = 4,
        latent_dim: int = 64,
        input_size: int = 64,
        variational: bool = False,
        epochs: int = 100,
        lr: float = 1e-3,
        beta: float = 0.001,
        physics_weight: float = 0.1,
    ):
        """
        Train the flow field autoencoder.
        
        Phase 1 of the discovery pipeline.
        """
        from models.autoencoder import FlowAutoencoder
        
        print("\n" + "="*60)
        print("  PHASE 1: Flow Autoencoder Training")
        print(f"  Latent dim: {latent_dim} | VAE: {variational}")
        print(f"  Input: ({in_channels}, {input_size}, {input_size})")
        print("="*60 + "\n")
        
        model = FlowAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=32,
            input_size=input_size,
            variational=variational,
        ).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            epoch_losses = {'total': 0, 'reconstruction': 0}
            n_batches = 0
            
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                optimizer.zero_grad()
                
                output = model(x)
                losses = model.compute_loss(
                    x, output,
                    beta=beta,
                    physics_weight=physics_weight,
                )
                
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                for k in epoch_losses:
                    if k in losses:
                        epoch_losses[k] += losses[k].item()
                n_batches += 1
            
            scheduler.step()
            
            avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            self.history['autoencoder'].append(avg_losses)
            
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                torch.save(model.state_dict(), 
                          self.checkpoint_dir / 'autoencoder_best.pt')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:4d}/{epochs} | "
                      f"Loss: {avg_losses['total']:.6f} | "
                      f"Recon: {avg_losses['reconstruction']:.6f} | "
                      f"LR: {lr_now:.2e}")
        
        torch.save(model.state_dict(), 
                  self.checkpoint_dir / 'autoencoder_final.pt')
        print(f"\n  Autoencoder training complete. Best loss: {best_loss:.6f}")
        
        return model
    
    # Latent ODE Training
    
    def train_latent_ode(
        self,
        autoencoder: nn.Module,
        train_loader: DataLoader,
        latent_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        """
        Train the Neural ODE in latent space.
        
        Phase 2: learns dz/dt = f_θ(z, t)
        
        Uses paired frames (x_t0, x_t1) from the data loader.
        Encodes both, then trains ODE to predict z_t1 from z_t0.
        """
        from models.autoencoder import LatentDynamicsODE
        
        print("\n" + "="*60)
        print("  PHASE 2: Latent Neural ODE Training")
        print(f"  Latent dim: {latent_dim}")
        print("="*60 + "\n")
        
        # Freeze autoencoder
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad = False
        
        ode_model = LatentDynamicsODE(
            latent_dim=latent_dim,
            hidden_dim=256,
            n_layers=4,
        ).to(self.device)
        
        optimizer = optim.AdamW(ode_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            ode_model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                x_t0, x_t1, dt_val = batch
                x_t0 = x_t0.to(self.device)
                x_t1 = x_t1.to(self.device)
                dt_val = dt_val.to(self.device)
                
                # Encode both frames
                with torch.no_grad():
                    z_t0 = autoencoder.get_latent(x_t0)
                    z_t1_true = autoencoder.get_latent(x_t1)
                
                optimizer.zero_grad()
                
                # Predict z_t1 via ODE
                dt_mean = dt_val.mean()
                t_span = torch.tensor([0.0, dt_mean.item()], device=self.device)
                z_t1_pred, _ = ode_model(z_t0, t_span, n_steps=5)
                
                # Loss: MSE in latent space
                loss = F.mse_loss(z_t1_pred, z_t1_true)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ode_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            avg_loss = epoch_loss / max(n_batches, 1)
            self.history['latent_ode'].append({'loss': avg_loss})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ode_model.state_dict(),
                          self.checkpoint_dir / 'latent_ode_best.pt')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs} | "
                      f"Latent ODE Loss: {avg_loss:.6f}")
        
        torch.save(ode_model.state_dict(),
                  self.checkpoint_dir / 'latent_ode_final.pt')
        print(f"\n  Latent ODE training complete. Best loss: {best_loss:.6f}")
        
        return ode_model
    
    # Blow-up Detection Training
    
    def train_blowup_detector(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-3,
    ):
        """
        Train the blow-up detection classifier.
        
        Phase 3: predicts flow regime and blow-up probability.
        """
        from models.regularity_analysis import BlowupDetector
        
        print("\n" + "="*60)
        print("  PHASE 3: Blow-up Detection Training")
        print("="*60 + "\n")
        
        model = BlowupDetector(
            input_channels=4,
            n_classes=5,
        ).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.BCELoss()
        
        best_acc = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                fields, labels, re_vals = batch
                fields = fields.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                output = model(fields)
                
                # Classification loss
                cls_loss = criterion_cls(output['regime_logits'], labels)
                
                # Blow-up binary loss
                blowup_target = (labels >= 3).float().unsqueeze(1)
                blowup_loss = criterion_reg(output['blowup_prob'], blowup_target)
                
                loss = cls_loss + 0.5 * blowup_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(output['regime_logits'], 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            scheduler.step()
            
            acc = correct / max(total, 1)
            avg_loss = epoch_loss / max(total // 16 + 1, 1)
            self.history['blowup'].append({'loss': avg_loss, 'accuracy': acc})
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                          self.checkpoint_dir / 'blowup_detector_best.pt')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.1%}")
        
        torch.save(model.state_dict(),
                  self.checkpoint_dir / 'blowup_detector_final.pt')
        print(f"\n  Blow-up detector training complete. Best acc: {best_acc:.1%}")
        
        return model
    
    # Symbolic Discovery
    
    def run_symbolic_discovery(
        self,
        autoencoder: nn.Module,
        data_loader: DataLoader,
        dt: float = 0.025,
        sindy_threshold: float = 0.1,
        gp_generations: int = 30,
        verbose: bool = True,
    ) -> Dict:
        """
        Run symbolic discovery on latent trajectories.
        
        Phase 4: THE BIG MOVE
        
        1. Encode flow field snapshots into latent space
        2. Construct temporal trajectories z(t)
        3. Run SINDy sparse regression → polynomial/trig equations
        4. Run Genetic Programming → free-form expressions
        5. Report discovered equations
        """
        from models.symbolic_discovery import SymbolicDiscoveryEngine
        
        print("\n" + "="*60)
        print("  PHASE 4: SYMBOLIC DISCOVERY (THE BIG MOVE)")
        print("  Discovering hidden structure in turbulence...")
        print("="*60 + "\n")
        
        # Encode all snapshots into latent space
        autoencoder.eval()
        all_latents = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                z = autoencoder.get_latent(x)
                all_latents.append(z.cpu().numpy())
        
        Z = np.concatenate(all_latents, axis=0)
        
        if verbose:
            print(f"  Encoded {Z.shape[0]} snapshots to latent dim {Z.shape[1]}")
        
        # Use only first few principal components for discovery
        n_discovery_dims = min(8, Z.shape[1])
        
        # PCA on latent space
        Z_mean = Z.mean(axis=0)
        Z_centered = Z - Z_mean
        U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        Z_pca = Z_centered @ Vt[:n_discovery_dims].T
        
        if verbose:
            explained_var = S[:n_discovery_dims]**2 / np.sum(S**2)
            print(f"  PCA: {n_discovery_dims} components explain "
                  f"{np.sum(explained_var):.1%} variance")
        
        # Run discovery engine
        engine = SymbolicDiscoveryEngine(
            n_latent_dims=n_discovery_dims,
            sindy_threshold=sindy_threshold,
            gp_population=100,
            gp_generations=gp_generations,
        )
        
        results = engine.discover_from_trajectories(
            Z_pca,
            dt=dt,
            run_gp=True,
            verbose=verbose,
        )
        
        # Save results
        results['pca_components'] = Vt[:n_discovery_dims]
        results['latent_mean'] = Z_mean
        results['explained_variance'] = (S[:n_discovery_dims]**2 / np.sum(S**2)).tolist()
        
        np.savez(
            self.checkpoint_dir / 'symbolic_results.npz',
            Z_pca=Z_pca,
            pca_components=Vt[:n_discovery_dims],
            latent_mean=Z_mean,
        )
        
        return results
    
    # Full Pipeline
    
    def run_full_pipeline(
        self,
        nx: int = 64,
        latent_dim: int = 32,
        n_ae_samples: int = 30,
        n_paired_samples: int = 20,
        n_stability_samples: int = 30,
        ae_epochs: int = 30,
        ode_epochs: int = 30,
        blowup_epochs: int = 20,
        gp_generations: int = 20,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the complete Turbulence Discovery pipeline end-to-end.
        
        This is the one-call entry point:
            Data gen → Autoencoder → Latent ODE → Blow-up → Symbolic Discovery
        """
        from training.turbulence_data import (
            TurbulenceDataGenerator, FlowSnapshotDataset,
            PairedFrameDataset, StabilityDataset
        )
        
        total_start = time.perf_counter()
        
        print("\n" + "="*72)
        print("  ╔═════════════════════════════════════════════════════════════╗")
        print("  ║     TURBULENCE DISCOVERY AI — FULL PIPELINE               ║")
        print("  ║     Compress → Learn Rules → Output Equations             ║")
        print("  ╚═════════════════════════════════════════════════════════════╝")
        print("="*72 + "\n")
        
        # ====== DATA GENERATION ======
        print("  [DATA] Generating multi-regime turbulence datasets...\n")
        gen = TurbulenceDataGenerator(nx=nx, ny=nx, dt=0.005)
        
        # Autoencoder data
        ae_data = gen.generate_autoencoder_data(
            n_samples=n_ae_samples,
            frames_per_sim=5,
            verbose=verbose,
        )
        ae_loader = DataLoader(
            FlowSnapshotDataset(ae_data),
            batch_size=8, shuffle=True
        )
        
        # Paired frames
        paired_data = gen.generate_paired_frames(
            n_trajectories=n_paired_samples,
            frames_per_traj=5,
            verbose=verbose,
        )
        paired_loader = DataLoader(
            PairedFrameDataset(paired_data),
            batch_size=8, shuffle=True
        )
        
        # Stability labels
        stability_data = gen.generate_stability_labels(
            n_samples=n_stability_samples,
            verbose=verbose,
        )
        stability_loader = DataLoader(
            StabilityDataset(stability_data),
            batch_size=8, shuffle=True
        )
        
        print(f"\n  Data generated: "
              f"{len(ae_data['snapshots'])} AE snapshots, "
              f"{len(paired_data['frames_t0'])} paired frames, "
              f"{len(stability_data['fields'])} stability samples\n")
        
        # ====== PHASE 1: AUTOENCODER ======
        autoencoder = self.train_autoencoder(
            ae_loader,
            in_channels=4,
            latent_dim=latent_dim,
            input_size=nx,
            variational=False,
            epochs=ae_epochs,
            lr=1e-3,
        )
        
        # ====== PHASE 2: LATENT ODE ======
        ode_model = self.train_latent_ode(
            autoencoder,
            paired_loader,
            latent_dim=latent_dim,
            epochs=ode_epochs,
            lr=1e-3,
        )
        
        # ====== PHASE 3: BLOW-UP DETECTOR ======
        blowup_model = self.train_blowup_detector(
            stability_loader,
            epochs=blowup_epochs,
            lr=1e-3,
        )
        
        # ====== PHASE 4: SYMBOLIC DISCOVERY ======
        discovery_results = self.run_symbolic_discovery(
            autoencoder,
            ae_loader,
            dt=0.025,
            gp_generations=gp_generations,
            verbose=verbose,
        )
        
        total_elapsed = time.perf_counter() - total_start
        
        print("\n" + "="*72)
        print("  PIPELINE COMPLETE")
        print(f"  Total time: {total_elapsed:.1f}s")
        print("="*72)
        
        return {
            'autoencoder': autoencoder,
            'ode_model': ode_model,
            'blowup_model': blowup_model,
            'discovery': discovery_results,
            'history': self.history,
        }
    
    def plot_discovery_results(
        self,
        discovery_results: Dict,
        save_path: Optional[str] = None,
    ):
        """
        Plot comprehensive discovery results.
        
        Panels:
            1. Autoencoder training loss
            2. Latent ODE training loss
            3. Blow-up detector accuracy
            4. SINDy coefficient matrix
            5. Latent space PCA
            6. Discovered equations summary
        """
        import matplotlib.pyplot as plt
        import matplotlib
        
        # Dark theme
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
        
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(
            'Turbulence Discovery AI — Results Dashboard',
            fontsize=20, fontweight='bold', color='#58a6ff', y=0.98
        )
        
        # 1. Autoencoder loss
        ax1 = fig.add_subplot(2, 3, 1)
        if self.history['autoencoder']:
            ae_losses = [h['total'] for h in self.history['autoencoder']]
            ax1.semilogy(ae_losses, color='#58a6ff', lw=2)
            ax1.fill_between(range(len(ae_losses)), ae_losses, alpha=0.1, color='#58a6ff')
        ax1.set_title('Autoencoder Loss', color='#79c0ff', fontsize=13)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.15, color='#30363d')
        
        # 2. Latent ODE loss
        ax2 = fig.add_subplot(2, 3, 2)
        if self.history['latent_ode']:
            ode_losses = [h['loss'] for h in self.history['latent_ode']]
            ax2.semilogy(ode_losses, color='#7ee787', lw=2)
            ax2.fill_between(range(len(ode_losses)), ode_losses, alpha=0.1, color='#7ee787')
        ax2.set_title('Latent ODE Loss', color='#79c0ff', fontsize=13)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.15, color='#30363d')
        
        # 3. Blow-up detector accuracy
        ax3 = fig.add_subplot(2, 3, 3)
        if self.history['blowup']:
            accs = [h.get('accuracy', 0) for h in self.history['blowup']]
            ax3.plot(accs, color='#f97583', lw=2)
            ax3.fill_between(range(len(accs)), accs, alpha=0.1, color='#f97583')
        ax3.set_title('Blow-up Detector Accuracy', color='#79c0ff', fontsize=13)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.15, color='#30363d')
        
        # 4. SINDy coefficient sparsity
        ax4 = fig.add_subplot(2, 3, 4)
        if 'sindy' in discovery_results and discovery_results['sindy'].get('Xi') is not None:
            Xi = discovery_results['sindy']['Xi']
            im = ax4.imshow(np.log10(np.abs(Xi) + 1e-10), cmap='viridis',
                           aspect='auto', interpolation='nearest')
            ax4.set_title('SINDy Coefficients (log₁₀|ξ|)', color='#79c0ff', fontsize=13)
            ax4.set_xlabel('Latent dim')
            ax4.set_ylabel('Library function')
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        else:
            ax4.text(0.5, 0.5, 'No SINDy data', ha='center', va='center',
                    transform=ax4.transAxes, color='#8b949e', fontsize=14)
        
        # 5. PCA explained variance
        ax5 = fig.add_subplot(2, 3, 5)
        if 'explained_variance' in discovery_results:
            ev = discovery_results['explained_variance']
            ax5.bar(range(len(ev)), ev, color='#d2a8ff', alpha=0.8)
            ax5.plot(range(len(ev)), np.cumsum(ev), 'o-', color='#ffa657', lw=2)
        ax5.set_title('Latent PCA Explained Variance', color='#79c0ff', fontsize=13)
        ax5.set_xlabel('Component')
        ax5.set_ylabel('Variance Ratio')
        ax5.grid(True, alpha=0.15, color='#30363d')
        
        # 6. Discovered equations text
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Discovered Equations', color='#79c0ff', fontsize=13)
        
        text_lines = ["Turbulence Discovery — Key Findings\n"]
        
        if 'sindy' in discovery_results:
            sindy = discovery_results['sindy']
            text_lines.append(f"SINDy: {sindy.get('complexity', '?')} terms, "
                            f"MSE={sindy.get('error', 0):.4e}")
            for eq in sindy.get('equations', [])[:3]:
                text_lines.append(f"  {eq[:55]}")
        
        if 'gp' in discovery_results:
            text_lines.append("\nGenetic Programming:")
            for k, v in list(discovery_results['gp'].items())[:3]:
                text_lines.append(f"  {k}: {v['equation'][:45]}")
        
        ax6.text(0.05, 0.95, '\n'.join(text_lines),
                transform=ax6.transAxes, va='top', fontsize=9,
                color='#c9d1d9', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                         edgecolor='#30363d', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Discovery results plot saved: {save_path}")
        
        try:
            backend = matplotlib.get_backend().lower()
            if 'agg' not in backend:
                plt.show()
            else:
                plt.close(fig)
        except Exception:
            plt.close(fig)
