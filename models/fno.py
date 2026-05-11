"""Fourier Neural Operator (FNO) for Navier-Stokes"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.
    
    Performs convolution in Fourier space by:
    1. FFT of input
    2. Multiply by learnable complex weights R (truncated to `modes` frequencies)
    3. Inverse FFT
    
    This captures global, multi-scale interactions efficiently.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply (x-direction)
        self.modes2 = modes2  # Number of Fourier modes to multiply (y-direction)
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Complex weights for two sets of modes (positive and negative frequencies)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication and summation: (batch, in, x, y) × (in, out, x, y) → (batch, out, x, y)"""
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Args:
            x: (batch, channels, nx, ny) input tensor
        
        Returns:
            (batch, out_channels, nx, ny) output tensor
        """
        batchsize = x.shape[0]
        
        # Compute FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Upper left quadrant
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        
        # Lower left quadrant  
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FourierLayer(nn.Module):
    """
    Complete Fourier Layer: spectral conv + bypass + activation.
    
    v^{l+1} = σ(K(v^l) + W·v^l + b)
    """
    
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, 1)  # 1×1 convolution (local linear transform)
        self.norm = nn.InstanceNorm2d(width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fourier layer forward: spectral path + bypass path + normalization."""
        x1 = self.spectral_conv(x)
        x2 = self.bypass(x)
        out = x1 + x2
        out = self.norm(out)
        return F.gelu(out)


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for Navier-Stokes equations.
    
    Learns the mapping: (u_t, parameters) → u_{t+1}
    
    Can be used for:
        - Time-stepping (auto-regressive)
        - Super-resolution (coarse → fine)
        - Parameter-to-solution mapping
    
    Architecture:
        Input: (batch, nx, ny, channels_in)
        Lifting: channels_in → width (pointwise)
        Fourier Layers: width → width (global convolution)
        Projection: width → channels_out (pointwise)
        Output: (batch, nx, ny, channels_out)
    """
    
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 3,   # (u, v, p) or (vx, vy, params)
        out_channels: int = 3,  # (u, v, p)
        padding: int = 8,
    ):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        
        # Lifting layer: project input channels to hidden dimension
        self.lifting = nn.Conv2d(in_channels + 2, width, 1)  # +2 for grid coordinates
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes1, modes2) for _ in range(n_layers)
        ])
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )
    
    def _get_grid(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate normalized coordinate grid [0, 1] × [0, 1]."""
        batchsize, nx, ny = shape[0], shape[2], shape[3]
        
        gridx = torch.linspace(0, 1, nx, device=device).reshape(1, 1, nx, 1).repeat(batchsize, 1, 1, ny)
        gridy = torch.linspace(0, 1, ny, device=device).reshape(1, 1, 1, ny).repeat(batchsize, 1, nx, 1)
        
        return torch.cat([gridx, gridy], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, nx, ny) input tensor
        
        Returns:
            (batch, out_channels, nx, ny) output prediction
        """
        # Add grid coordinates
        grid = self._get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=1)
        
        # Lifting
        x = self.lifting(x)
        
        # Padding for non-periodic boundaries
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        # Projection
        x = self.projection(x)
        
        return x
    
    def predict_trajectory(
        self, initial_state: torch.Tensor, n_steps: int
    ) -> List[torch.Tensor]:
        """
        Auto-regressive prediction: roll out trajectory from initial state.
        
        Args:
            initial_state: (1, channels, nx, ny) initial condition
            n_steps: Number of time steps to predict
        
        Returns:
            List of predicted states
        """
        self.eval()
        predictions = [initial_state]
        state = initial_state
        
        with torch.no_grad():
            for _ in range(n_steps):
                state = self.forward(state)
                predictions.append(state)
        
        return predictions
    
    def compute_physics_loss(
        self, pred: torch.Tensor, dx: float, dy: float, nu: float
    ) -> torch.Tensor:
        """
        Optional physics-informed loss for FNO.
        Enforces continuity equation.
        """
        u = pred[:, 0:1, :, :]
        v = pred[:, 1:2, :, :]
        
        # Central differences for divergence
        dudx = (torch.roll(u, -1, dims=3) - torch.roll(u, 1, dims=3)) / (2 * dx)
        dvdy = (torch.roll(v, -1, dims=2) - torch.roll(v, 1, dims=2)) / (2 * dy)
        
        divergence = dudx + dvdy
        return torch.mean(divergence**2)


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for time-dependent Navier-Stokes.
    
    Treats time as a third spatial dimension for spatio-temporal learning.
    Input: (batch, channels, nx, ny, nt) → Output: (batch, channels, nx, ny, nt)
    """
    
    def __init__(
        self,
        modes1: int = 8, modes2: int = 8, modes3: int = 8,
        width: int = 20, n_layers: int = 4,
        in_channels: int = 4, out_channels: int = 3
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        self.lifting = nn.Conv3d(in_channels + 3, width, 1)  # +3 for coordinates
        
        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.spectral_layers.append(
                SpectralConv3d(width, width, modes1, modes2, modes3)
            )
            self.bypass_layers.append(nn.Conv3d(width, width, 1))
        
        self.projection = nn.Sequential(
            nn.Conv3d(width, 128, 1),
            nn.GELU(),
            nn.Conv3d(128, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        nx, ny, nz = x.shape[2], x.shape[3], x.shape[4]
        
        # Add coordinates
        gx = torch.linspace(0, 1, nx, device=x.device).reshape(1,1,nx,1,1).expand(batch,1,nx,ny,nz)
        gy = torch.linspace(0, 1, ny, device=x.device).reshape(1,1,1,ny,1).expand(batch,1,nx,ny,nz)
        gz = torch.linspace(0, 1, nz, device=x.device).reshape(1,1,1,1,nz).expand(batch,1,nx,ny,nz)
        x = torch.cat([x, gx, gy, gz], dim=1)
        
        x = self.lifting(x)
        
        for sp_layer, bp_layer in zip(self.spectral_layers, self.bypass_layers):
            x1 = sp_layer(x)
            x2 = bp_layer(x)
            x = F.gelu(x1 + x2)
        
        return self.projection(x)


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution."""
    
    def __init__(self, in_ch, out_ch, modes1, modes2, modes3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        scale = 1 / (in_ch * out_ch)
        
        self.weights1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
    
    def forward(self, x):
        batch = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        out_ft = torch.zeros(batch, self.weights1.shape[1], x.size(-3), x.size(-2), x.size(-1)//2+1,
                            dtype=torch.cfloat, device=x.device)
        
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz", 
            x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz",
            x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = torch.einsum("bixyz,ioxyz->boxyz",
            x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = torch.einsum("bixyz,ioxyz->boxyz",
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4)
        
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
