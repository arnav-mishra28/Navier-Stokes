"""
=============================================================================
Neural Turbulence Closure Model
Replaces traditional subgrid-scale models (Smagorinsky, etc.) with a
learned neural network that predicts subgrid stress tensor.

Learns: τ_ij^{sgs} = f(S_ij, Ω_ij, k, ε, ...)

Advantages over classic models:
    - Data-driven: captures physics that analytical models miss
    - Can learn from DNS data
    - Adapts to specific flow configurations
    - Preserves tensor invariance (optional)
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class TurbulenceClosureNN(nn.Module):
    """
    Neural network for predicting subgrid-scale (SGS) stress tensor.
    
    Input features (per grid point):
        - Strain rate tensor components: S11, S12, S22
        - Rotation rate tensor: Ω12
        - Velocity gradients: ∂u_i/∂x_j
        - Filter width: Δ
        - Local Reynolds number proxy
    
    Output:
        - SGS stress tensor: τ11, τ12, τ22 (symmetric)
        - Or equivalently: ν_t (eddy viscosity)
    
    Architecture: Invariant neural network that respects tensor symmetries.
    """
    
    def __init__(
        self,
        input_features: int = 9,
        output_features: int = 3,  # τ11, τ12, τ22
        hidden_layers: list = None,
        activation: str = "gelu",
        predict_eddy_viscosity: bool = False,
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 64]
        
        self.predict_nu_t = predict_eddy_viscosity
        
        act_map = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'silu': nn.SiLU}
        act = act_map.get(activation, nn.GELU)
        
        if predict_eddy_viscosity:
            output_features = 1  # Just ν_t
        
        layers = []
        in_dim = input_features
        for h in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                act(),
                nn.Dropout(0.05),
            ])
            in_dim = h
        
        # Output with softplus for positivity (eddy viscosity must be ≥ 0)
        layers.append(nn.Linear(in_dim, output_features))
        if predict_eddy_viscosity:
            layers.append(nn.Softplus())
        
        self.network = nn.Sequential(*layers)
        
        # Input normalization statistics
        self.register_buffer('input_mean', torch.zeros(input_features))
        self.register_buffer('input_std', torch.ones(input_features))
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set input normalization parameters from training data."""
        self.input_mean = torch.tensor(mean, dtype=torch.float32)
        self.input_std = torch.tensor(std, dtype=torch.float32)
    
    def extract_features(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        dx: float,
        dy: float,
        delta: float
    ) -> torch.Tensor:
        """
        Extract input features from velocity field.
        
        Args:
            u, v: (batch, 1, ny, nx) velocity components
            dx, dy: grid spacing
            delta: filter width
        
        Returns:
            features: (batch, n_features, ny, nx) tensor
        """
        # Velocity gradients using central differences
        dudx = (torch.roll(u, -1, 3) - torch.roll(u, 1, 3)) / (2 * dx)
        dudy = (torch.roll(u, -1, 2) - torch.roll(u, 1, 2)) / (2 * dy)
        dvdx = (torch.roll(v, -1, 3) - torch.roll(v, 1, 3)) / (2 * dx)
        dvdy = (torch.roll(v, -1, 2) - torch.roll(v, 1, 2)) / (2 * dy)
        
        # Strain rate tensor
        S11 = dudx
        S22 = dvdy
        S12 = 0.5 * (dudy + dvdx)
        
        # Rotation rate
        O12 = 0.5 * (dudy - dvdx)
        
        # Strain rate magnitude
        S_mag = torch.sqrt(2 * (S11**2 + S22**2 + 2*S12**2) + 1e-8)
        
        # Velocity magnitude
        vel_mag = torch.sqrt(u**2 + v**2 + 1e-8)
        
        # Q-criterion (vortex identification)
        Q = 0.5 * (2*O12**2 - (S11**2 + S22**2 + 2*S12**2))
        
        # Local Reynolds number proxy
        Re_local = vel_mag * delta / 1e-3  # Approximate
        
        # Normalized filter width
        delta_norm = torch.full_like(u, delta / dx)
        
        features = torch.cat([
            S11, S12, S22,          # Strain rate (3)
            O12,                    # Rotation rate (1)
            S_mag,                  # Strain magnitude (1)
            vel_mag,                # Velocity magnitude (1)
            Q,                      # Q-criterion (1)
            Re_local,               # Local Reynolds (1)
            delta_norm,             # Normalized filter width (1)
        ], dim=1)
        
        return features
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: (..., n_features) input features
        
        Returns:
            (..., n_outputs) SGS stress or eddy viscosity
        """
        # Normalize
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        normalized = (flat - self.input_mean) / (self.input_std + 1e-8)
        
        output = self.network(normalized)
        return output.reshape(*shape[:-1], -1)
    
    def predict_sgs_stress(
        self,
        u: np.ndarray,
        v: np.ndarray,
        dx: float,
        dy: float,
        delta: float,
        device: str = 'cpu'
    ) -> Dict[str, np.ndarray]:
        """
        Predict SGS stress tensor from velocity field.
        
        Returns:
            Dict with 'tau11', 'tau12', 'tau22' (or 'nu_t')
        """
        self.eval()
        
        with torch.no_grad():
            u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            features = self.extract_features(u_t, v_t, dx, dy, delta)
            
            # Reshape for network: (1, n_feat, ny, nx) → (ny*nx, n_feat)
            batch, n_feat, ny, nx = features.shape
            feat_flat = features.permute(0, 2, 3, 1).reshape(-1, n_feat)
            
            output = self.forward(feat_flat).cpu().numpy()
        
        if self.predict_nu_t:
            return {'nu_t': output[:, 0].reshape(u.shape)}
        else:
            return {
                'tau11': output[:, 0].reshape(u.shape),
                'tau12': output[:, 1].reshape(u.shape),
                'tau22': output[:, 2].reshape(u.shape),
            }


class InvariantTurbulenceNN(TurbulenceClosureNN):
    """
    Galilean-invariant turbulence model.
    
    Uses tensor basis expansion to guarantee frame invariance:
    τ = Σ_n g_n(I_1, ..., I_k) * T_n
    
    where:
        I_k = scalar invariants (traces of tensor products)
        T_n = tensor basis elements
        g_n = learned scalar functions (neural networks)
    
    This preserves physical symmetries by construction.
    """
    
    def __init__(self, hidden_layers: list = None):
        # 5 scalar invariants → 10 tensor basis coefficients
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]
        super().__init__(
            input_features=5,   # Scalar invariants
            output_features=10, # Basis coefficients
            hidden_layers=hidden_layers,
            predict_eddy_viscosity=False,
        )
    
    def compute_invariants(
        self,
        S11: torch.Tensor, S12: torch.Tensor, S22: torch.Tensor,
        O12: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scalar invariants of strain and rotation tensors.
        
        I1 = tr(S²) = S11² + S22² + 2S12²
        I2 = tr(Ω²) = -2Ω12²
        I3 = tr(S³) = S11³ + 3S11·S12² + ...
        I4 = tr(Ω²·S) = ...
        I5 = tr(Ω²·S²) = ...
        """
        I1 = S11**2 + S22**2 + 2*S12**2
        I2 = -2 * O12**2
        I3 = S11**3 + 3*S11*S12**2 + 3*S22*S12**2 + S22**3
        I4 = -2 * O12**2 * (S11 + S22)
        I5 = -2 * O12**2 * (S11**2 + S22**2 + 2*S12**2)
        
        return torch.stack([I1, I2, I3, I4, I5], dim=-1)
