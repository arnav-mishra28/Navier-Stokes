"""Physics-Informed Neural Network (PINN) for Navier-Stokes"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict


class SinActivation(nn.Module):
    """Sinusoidal activation function (SIREN-style). Captures periodic features well."""
    def forward(self, x):
        return torch.sin(x)


class AdaptiveActivation(nn.Module):
    """Trainable activation: σ(n·a·x) where a is learnable per-layer."""
    def __init__(self, n: float = 1.0):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return torch.tanh(self.n * self.a * x)


class FourierFeatureLayer(nn.Module):
    """
    Random Fourier Feature embedding for improved PINN convergence.
    Maps low-dimensional inputs to high-dimensional feature space.
    
    γ(x) = [cos(2πBx), sin(2πBx)]
    where B is a random matrix sampled from N(0, σ²)
    """
    def __init__(self, in_features: int, n_features: int = 128, sigma: float = 1.0):
        super().__init__()
        B = torch.randn(in_features, n_features) * sigma
        self.register_buffer('B', B)
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block for deep PINNs."""
    def __init__(self, width: int, activation: str = "tanh"):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        
        act_map = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'sin': SinActivation(),
        }
        self.activation = act_map.get(activation, nn.Tanh())
    
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + residual)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for incompressible Navier-Stokes equations.
    
    Architecture:
        Input: (x, y, t) → [Fourier Features] → [MLP with residual connections] → (u, v, p)
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # (x, y, t)
        output_dim: int = 3,  # (u, v, p)
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        use_fourier_features: bool = True,
        fourier_features: int = 128,
        fourier_sigma: float = 1.0,
        use_residual: bool = True,
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_fourier = use_fourier_features
        self.use_residual = use_residual
        
        # Fourier feature embedding
        if use_fourier_features:
            self.fourier = FourierFeatureLayer(input_dim, fourier_features, fourier_sigma)
            in_features = 2 * fourier_features
        else:
            self.fourier = None
            in_features = input_dim
        
        # Network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_features, hidden_layers[0]))
        
        # Hidden layers
        act_map = {
            'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU,
        }
        act_class = act_map.get(activation, nn.Tanh)
        
        if use_residual and all(h == hidden_layers[0] for h in hidden_layers):
            # Use residual blocks when widths are equal
            layers.append(act_class())
            for _ in range(len(hidden_layers) - 1):
                layers.append(ResidualBlock(hidden_layers[0], activation))
        else:
            for i in range(len(hidden_layers) - 1):
                layers.append(act_class())
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(act_class())
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Adaptive loss weights (learnable)
        self.log_w_pde = nn.Parameter(torch.tensor(0.0))
        self.log_w_bc = nn.Parameter(torch.tensor(0.0))
        self.log_w_ic = nn.Parameter(torch.tensor(0.0))
        self.log_w_data = nn.Parameter(torch.tensor(0.0))
        
        # Physical parameter
        self.nu = 0.01  # Kinematic viscosity
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 3) containing (x, y, t)
        
        Returns:
            Output tensor (batch, 3) containing (u, v, p)
        """
        if self.fourier is not None:
            x = self.fourier(x)
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict u, v, p separately."""
        out = self.forward(x)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
    
    def compute_ns_residual(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Navier-Stokes PDE residuals using automatic differentiation.
        
        Residuals:
            R_momentum_x = ∂u/∂t + u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²)
            R_momentum_y = ∂v/∂t + u·∂v/∂x + v·∂v/∂y + ∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²)
            R_continuity = ∂u/∂x + ∂v/∂y
        """
        # Enable gradient computation
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.forward(inputs)
        
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]
        
        # Residuals
        # Momentum-x: ∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
        R_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        
        # Momentum-y: ∂v/∂t + u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
        R_v = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        
        # Continuity: ∂u/∂x + ∂v/∂y = 0
        R_cont = u_x + v_y
        
        return R_u, R_v, R_cont
    
    def compute_loss(
        self,
        x_pde: torch.Tensor, y_pde: torch.Tensor, t_pde: torch.Tensor,
        x_bc: torch.Tensor, y_bc: torch.Tensor, t_bc: torch.Tensor,
        u_bc: torch.Tensor, v_bc: torch.Tensor,
        x_ic: Optional[torch.Tensor] = None,
        y_ic: Optional[torch.Tensor] = None,
        u_ic: Optional[torch.Tensor] = None,
        v_ic: Optional[torch.Tensor] = None,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None,
        t_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        v_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total PINN loss with adaptive weighting.
        
        Returns dict with individual loss components for monitoring.
        """
        losses = {}
        
        # PDE residual loss
        R_u, R_v, R_cont = self.compute_ns_residual(x_pde, y_pde, t_pde)
        loss_pde = torch.mean(R_u**2) + torch.mean(R_v**2) + torch.mean(R_cont**2)
        losses['pde'] = loss_pde
        
        # Boundary condition loss
        bc_input = torch.cat([x_bc, y_bc, t_bc], dim=1)
        bc_pred = self.forward(bc_input)
        loss_bc = (torch.mean((bc_pred[:, 0:1] - u_bc)**2) +
                   torch.mean((bc_pred[:, 1:2] - v_bc)**2))
        losses['bc'] = loss_bc
        
        # Initial condition loss
        loss_ic = torch.tensor(0.0, device=x_pde.device)
        if x_ic is not None and u_ic is not None:
            ic_input = torch.cat([x_ic, y_ic, torch.zeros_like(x_ic)], dim=1)
            ic_pred = self.forward(ic_input)
            loss_ic = (torch.mean((ic_pred[:, 0:1] - u_ic)**2) +
                       torch.mean((ic_pred[:, 1:2] - v_ic)**2))
        losses['ic'] = loss_ic
        
        # Data loss (if available)
        loss_data = torch.tensor(0.0, device=x_pde.device)
        if x_data is not None and u_data is not None:
            data_input = torch.cat([x_data, y_data, t_data], dim=1)
            data_pred = self.forward(data_input)
            loss_data = (torch.mean((data_pred[:, 0:1] - u_data)**2) +
                         torch.mean((data_pred[:, 1:2] - v_data)**2))
        losses['data'] = loss_data
        
        # Adaptive weighting
        w_pde = torch.exp(-self.log_w_pde)
        w_bc = torch.exp(-self.log_w_bc)
        w_ic = torch.exp(-self.log_w_ic)
        w_data = torch.exp(-self.log_w_data)
        
        total_loss = (w_pde * loss_pde + self.log_w_pde +
                      w_bc * loss_bc + self.log_w_bc +
                      w_ic * loss_ic + self.log_w_ic +
                      w_data * loss_data + self.log_w_data)
        
        losses['total'] = total_loss
        losses['weights'] = {
            'w_pde': w_pde.item(), 'w_bc': w_bc.item(),
            'w_ic': w_ic.item(), 'w_data': w_data.item()
        }
        
        return losses
    
    def predict_field(
        self, x_grid: np.ndarray, y_grid: np.ndarray, t: float,
        device: str = 'cpu'
    ) -> Dict[str, np.ndarray]:
        """
        Predict velocity and pressure fields on a grid.
        
        Args:
            x_grid, y_grid: 2D coordinate arrays
            t: Time value
        
        Returns:
            Dict with 'u', 'v', 'p' fields
        """
        self.eval()
        with torch.no_grad():
            x_flat = torch.tensor(x_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
            y_flat = torch.tensor(y_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
            t_flat = torch.full_like(x_flat, t)
            
            inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
            outputs = self.forward(inputs).cpu().numpy()
        
        shape = x_grid.shape
        return {
            'u': outputs[:, 0].reshape(shape),
            'v': outputs[:, 1].reshape(shape),
            'p': outputs[:, 2].reshape(shape),
        }


class PINNSteadyState(PINN):
    """
    Specialized PINN for steady-state Navier-Stokes (no time dependence).
    
    Solves: (u·∇)u = -∇p + ν∇²u, ∇·u = 0
    Input: (x, y) → Output: (u, v, p)
    """
    
    def __init__(self, **kwargs):
        kwargs['input_dim'] = 2  # (x, y) only
        super().__init__(**kwargs)
    
    def compute_ns_residual(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steady-state NS residuals (no time derivative)."""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        inputs = torch.cat([x, y], dim=1)
        outputs = self.forward(inputs)
        
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        # Derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]
        
        R_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        R_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        R_cont = u_x + v_y
        
        return R_u, R_v, R_cont
