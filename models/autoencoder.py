"""Turbulence Autoencoder — Compresses flow fields to latent representations"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class ResBlock(nn.Module):
    """Residual block with GroupNorm for stable training on small batches."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    
    def forward(self, x):
        return x + self.block(x)


class FlowAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for turbulent flow field compression.
    
    Learns: z = Encode(u, v, p, ω) ∈ ℝ^{latent_dim}
    Then:   (u', v', p', ω') = Decode(z)
    
    The latent space z captures the essential structure of the turbulence.
    """
    
    def __init__(
        self,
        in_channels: int = 4,       # (u, v, p, vorticity)
        latent_dim: int = 64,
        base_channels: int = 32,
        n_res_blocks: int = 2,
        input_size: int = 128,       # Spatial resolution (assumed square)
        variational: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.variational = variational
        self.input_size = input_size
        
        # Channel progression: 32 → 64 → 128 → 256
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        # ===== ENCODER =====
        encoder_layers = [
            nn.Conv2d(in_channels, ch[0], 3, padding=1),
            nn.GELU(),
        ]
        for i in range(len(ch) - 1):
            encoder_layers.extend([
                nn.Conv2d(ch[i], ch[i+1], 4, stride=2, padding=1),  # Downsample
                nn.GroupNorm(min(8, ch[i+1]), ch[i+1]),
                nn.GELU(),
            ])
            for _ in range(n_res_blocks):
                encoder_layers.append(ResBlock(ch[i+1]))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Compute spatial size after encoding (3 downsamples: /8)
        self.encoded_spatial = input_size // 8
        self.flat_dim = ch[-1] * self.encoded_spatial * self.encoded_spatial
        
        # Latent projection
        if variational:
            self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        else:
            self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # ===== DECODER =====
        decoder_layers = []
        for i in range(len(ch) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose2d(ch[i], ch[i-1], 4, stride=2, padding=1),  # Upsample
                nn.GroupNorm(min(8, ch[i-1]), ch[i-1]),
                nn.GELU(),
            ])
            for _ in range(n_res_blocks):
                decoder_layers.append(ResBlock(ch[i-1]))
        
        decoder_layers.append(nn.Conv2d(ch[0], in_channels, 3, padding=1))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Input normalization
        self.register_buffer('input_mean', torch.zeros(in_channels, 1, 1))
        self.register_buffer('input_std', torch.ones(in_channels, 1, 1))
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set input normalization from training statistics."""
        self.input_mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.input_std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean.to(x.device)) / (self.input_std.to(x.device) + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.input_std.to(x.device) + 1e-8) + self.input_mean.to(x.device)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode flow field to latent vector."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        if self.variational:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:
            return self.fc_encode(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to flow field."""
        ch_last = self.fc_decode.out_features // (self.encoded_spatial ** 2)
        h = self.fc_decode(z)
        h = h.view(h.size(0), ch_last, self.encoded_spatial, self.encoded_spatial)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode → (sample) → decode."""
        if self.variational:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return {
                'reconstruction': recon,
                'z': z,
                'mu': mu,
                'logvar': logvar,
            }
        else:
            z = self.encode(x)
            recon = self.decode(z)
            return {
                'reconstruction': recon,
                'z': z,
            }
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only (no decoding)."""
        if self.variational:
            mu, _ = self.encode(x)
            return mu
        else:
            return self.encode(x)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
        beta: float = 0.001,
        physics_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute autoencoder loss.
        
        Loss = MSE_reconstruction + β * KL_divergence (if VAE)
             + λ_physics * physics_loss
        
        Physics loss enforces divergence-free reconstruction:
            ∇·u_recon ≈ 0  (incompressibility)
        """
        recon = output['reconstruction']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        
        total = recon_loss
        losses = {'reconstruction': recon_loss}
        
        # KL divergence for VAE
        if self.variational and 'mu' in output:
            mu, logvar = output['mu'], output['logvar']
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total = total + beta * kl_loss
            losses['kl_divergence'] = kl_loss
        
        # Physics-informed: divergence-free constraint
        if physics_weight > 0 and recon.size(1) >= 2:
            u_recon = recon[:, 0:1]
            v_recon = recon[:, 1:2]
            
            # Central differences for divergence
            dudx = (torch.roll(u_recon, -1, 3) - torch.roll(u_recon, 1, 3)) / 2.0
            dvdy = (torch.roll(v_recon, -1, 2) - torch.roll(v_recon, 1, 2)) / 2.0
            div = dudx + dvdy
            
            div_loss = torch.mean(div ** 2)
            total = total + physics_weight * div_loss
            losses['divergence'] = div_loss
        
        losses['total'] = total
        return losses


class LatentDynamicsODE(nn.Module):
    """
    Neural ODE for latent space dynamics.
    
    Learns: dz/dt = f_θ(z, t)
    
    Given z(t₀) from the autoencoder, predicts z(t₁) by integrating
    the learned ODE in latent space. This captures the temporal evolution
    of turbulent flows in compressed form.
    
    Uses an Euler integrator for simplicity (can be upgraded to RK4).
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embedding_dim: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Dynamics network: dz/dt = f(z, t)
        layers = [nn.Linear(latent_dim + hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.05),
            ])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.dynamics_net = nn.Sequential(*layers)
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = self.time_embedding_dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    def ode_func(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = f_θ(z, t)."""
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_embed(t_emb)
        
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0).expand(z.size(0), -1)
        
        zt = torch.cat([z, t_emb], dim=-1)
        return self.dynamics_net(zt)
    
    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Integrate z forward in time using Euler method.
        
        Args:
            z0: (B, latent_dim) initial latent state
            t_span: (2,) tensor [t_start, t_end]
            n_steps: number of Euler steps
        
        Returns:
            z_final: (B, latent_dim) predicted latent state at t_end
        """
        dt = (t_span[1] - t_span[0]) / n_steps
        z = z0
        t = t_span[0]
        
        trajectory = [z0]
        
        for _ in range(n_steps):
            t_tensor = torch.full((z.size(0),), t.item(), device=z.device)
            dzdt = self.ode_func(z, t_tensor)
            z = z + dt * dzdt
            t = t + dt
            trajectory.append(z)
        
        return z, torch.stack(trajectory, dim=1)  # (B, n_steps+1, latent_dim)
    
    def forward_rk4(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """RK4 integration for higher accuracy."""
        dt = (t_span[1] - t_span[0]) / n_steps
        z = z0
        t = t_span[0]
        
        trajectory = [z0]
        
        for _ in range(n_steps):
            t_val = torch.full((z.size(0),), t.item(), device=z.device)
            t_half = torch.full((z.size(0),), (t + dt/2).item(), device=z.device)
            t_next = torch.full((z.size(0),), (t + dt).item(), device=z.device)
            
            k1 = self.ode_func(z, t_val)
            k2 = self.ode_func(z + 0.5 * dt * k1, t_half)
            k3 = self.ode_func(z + 0.5 * dt * k2, t_half)
            k4 = self.ode_func(z + dt * k3, t_next)
            
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t = t + dt
            trajectory.append(z)
        
        return z, torch.stack(trajectory, dim=1)


class TurbulenceDiscoveryAutoencoder(nn.Module):
    """
    Combined Autoencoder + Latent ODE system for turbulence discovery.
    
    Pipeline:
        1. Encode flow field → latent z
        2. Evolve z in time via Neural ODE → z(t+Δt)
        3. Decode z(t+Δt) → predicted flow field
    
    Training:
        - Reconstruction loss (autoencoder fidelity)
        - Latent dynamics loss (ODE prediction accuracy)
        - Physics constraints (divergence-free, energy conservation)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 64,
        base_channels: int = 32,
        input_size: int = 128,
        variational: bool = False,
        ode_hidden: int = 256,
        ode_layers: int = 4,
    ):
        super().__init__()
        
        self.autoencoder = FlowAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            input_size=input_size,
            variational=variational,
        )
        
        self.latent_ode = LatentDynamicsODE(
            latent_dim=latent_dim,
            hidden_dim=ode_hidden,
            n_layers=ode_layers,
        )
        
        self.latent_dim = latent_dim
    
    def forward(
        self,
        x_t0: torch.Tensor,
        x_t1: torch.Tensor,
        dt: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Full training forward pass.
        
        Args:
            x_t0: (B, C, H, W) flow field at time t
            x_t1: (B, C, H, W) flow field at time t + dt
            dt: time step
        """
        # Encode both frames
        ae_out_t0 = self.autoencoder(x_t0)
        ae_out_t1 = self.autoencoder(x_t1)
        
        z_t0 = ae_out_t0['z']
        z_t1_true = ae_out_t1['z']
        
        # Predict z at t1 from z at t0 using Neural ODE
        t_span = torch.tensor([0.0, dt], device=x_t0.device)
        z_t1_pred, trajectory = self.latent_ode(z_t0, t_span, n_steps=5)
        
        # Decode predicted latent
        x_t1_pred = self.autoencoder.decode(z_t1_pred)
        
        return {
            'recon_t0': ae_out_t0['reconstruction'],
            'recon_t1': ae_out_t1['reconstruction'],
            'x_t1_pred': x_t1_pred,
            'z_t0': z_t0,
            'z_t1_true': z_t1_true,
            'z_t1_pred': z_t1_pred,
            'trajectory': trajectory,
            **{f'ae_{k}': v for k, v in ae_out_t0.items() if k not in ['reconstruction', 'z']},
        }
    
    def predict_future(
        self,
        x: torch.Tensor,
        n_future: int = 10,
        dt: float = 0.01,
    ) -> list:
        """Autoregressively predict future flow fields."""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            z = self.autoencoder.get_latent(x)
            
            for step in range(n_future):
                t_span = torch.tensor([0.0, dt], device=x.device)
                z, _ = self.latent_ode(z, t_span, n_steps=5)
                pred = self.autoencoder.decode(z)
                predictions.append(pred.cpu())
        
        return predictions
