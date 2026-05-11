"""DeepONet (Deep Operator Network) for Navier-Stokes"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict


class BranchNetwork(nn.Module):
    """
    Branch network: encodes the input function.
    
    Takes discretized input function values (e.g., boundary velocity profile,
    initial condition, or forcing function) and produces a latent representation.
    """
    
    def __init__(
        self,
        input_dim: int,  # Number of sensor locations
        hidden_layers: List[int] = None,
        latent_dim: int = 128,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 256, 256]
        
        act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'silu': nn.SiLU}
        act = act_map.get(activation, nn.ReLU)
        
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), act()])
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, latent_dim)"""
        return self.net(x)


class TrunkNetwork(nn.Module):
    """
    Trunk network: encodes query coordinates.
    
    Maps spatial-temporal coordinates (x, y, t) to latent space
    that represents the solution basis functions.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # (x, y, t)
        hidden_layers: List[int] = None,
        latent_dim: int = 128,
        activation: str = "tanh"
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 128]
        
        act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'silu': nn.SiLU}
        act = act_map.get(activation, nn.Tanh)
        
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), act()])
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, latent_dim)"""
        return self.net(x)


class DeepONet(nn.Module):
    """
    Deep Operator Network for Navier-Stokes equations.
    
    Learns the solution operator: G: input_function → solution_function
    
    Output = Σ_k branch_k(input_fn) * trunk_k(x, y, t) + bias
    
    Applications:
        - Map boundary conditions → flow field
        - Map initial condition → time evolution
        - Map forcing function → response
        - Map geometry parameters → flow solution
    
    Multi-output version: separate branch-trunk pairs for each output (u, v, p).
    """
    
    def __init__(
        self,
        branch_input_dim: int = 100,  # Discretized input function size
        trunk_input_dim: int = 3,     # (x, y, t)
        n_outputs: int = 3,           # (u, v, p)
        branch_layers: List[int] = None,
        trunk_layers: List[int] = None,
        latent_dim: int = 128,
        activation: str = "relu",
    ):
        super().__init__()
        
        if branch_layers is None:
            branch_layers = [256, 256, 256]
        if trunk_layers is None:
            trunk_layers = [128, 128, 128]
        
        self.n_outputs = n_outputs
        self.latent_dim = latent_dim
        
        # Separate branch and trunk for each output
        self.branches = nn.ModuleList([
            BranchNetwork(branch_input_dim, branch_layers, latent_dim, activation)
            for _ in range(n_outputs)
        ])
        
        self.trunks = nn.ModuleList([
            TrunkNetwork(trunk_input_dim, trunk_layers, latent_dim, "tanh")
            for _ in range(n_outputs)
        ])
        
        # Learnable bias for each output
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_outputs)
        ])
    
    def forward(
        self, input_fn: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_fn: (batch, branch_input_dim) - discretized input function
            query_points: (batch, n_points, trunk_input_dim) - query (x,y,t)
        
        Returns:
            (batch, n_points, n_outputs) - predicted values at query points
        """
        batch_size = input_fn.shape[0]
        n_points = query_points.shape[1]
        
        outputs = []
        for i in range(self.n_outputs):
            # Branch encoding: (batch, latent_dim)
            branch_out = self.branches[i](input_fn)
            
            # Trunk encoding for all query points: (batch, n_points, latent_dim)
            # Reshape query points for batch processing
            qp_flat = query_points.reshape(-1, query_points.shape[-1])
            trunk_out = self.trunks[i](qp_flat)
            trunk_out = trunk_out.reshape(batch_size, n_points, self.latent_dim)
            
            # Dot product + bias: (batch, n_points)
            result = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)
            result = result + self.biases[i]
            
            outputs.append(result)
        
        # Stack outputs: (batch, n_points, n_outputs)
        return torch.stack(outputs, dim=-1)
    
    def predict_field(
        self,
        input_fn: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        t: float = 0.0,
        device: str = 'cpu'
    ) -> Dict[str, np.ndarray]:
        """
        Predict full field from input function.
        
        Args:
            input_fn: (branch_input_dim,) input function values
            x_grid, y_grid: 2D coordinate grids
            t: time
        
        Returns:
            Dict with 'u', 'v', 'p' fields
        """
        self.eval()
        
        with torch.no_grad():
            # Prepare branch input
            branch_in = torch.tensor(input_fn, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Prepare trunk input (query points)
            x_flat = x_grid.flatten()
            y_flat = y_grid.flatten()
            t_flat = np.full_like(x_flat, t)
            
            query = torch.tensor(
                np.stack([x_flat, y_flat, t_flat], axis=-1),
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Predict
            output = self.forward(branch_in, query).cpu().numpy()[0]
        
        shape = x_grid.shape
        result = {}
        names = ['u', 'v', 'p']
        for i in range(min(self.n_outputs, 3)):
            result[names[i]] = output[:, i].reshape(shape)
        
        return result


class PhysicsInformedDeepONet(DeepONet):
    """
    Physics-Informed DeepONet: combines operator learning with PDE constraints.
    
    Loss = L_data + λ * L_physics
    
    The physics loss enforces the NS equations at collocation points
    using automatic differentiation through the trunk network.
    """
    
    def __init__(self, nu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
    
    def compute_ns_residual(
        self,
        input_fn: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute NS residuals with autodiff through trunk."""
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        query = torch.stack([x, y, t], dim=-1).unsqueeze(0)
        out = self.forward(input_fn.unsqueeze(0), query)[0]
        
        u, v, p = out[:, 0], out[:, 1], out[:, 2]
        
        # Compute gradients
        grads = lambda f, var: torch.autograd.grad(
            f, var, torch.ones_like(f), create_graph=True, retain_graph=True
        )[0]
        
        u_x, u_y, u_t = grads(u, x), grads(u, y), grads(u, t)
        v_x, v_y, v_t = grads(v, x), grads(v, y), grads(v, t)
        p_x, p_y = grads(p, x), grads(p, y)
        
        u_xx = grads(u_x, x)
        u_yy = grads(u_y, y)
        v_xx = grads(v_x, x)
        v_yy = grads(v_y, y)
        
        R_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        R_v = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        R_cont = u_x + v_y
        
        return R_u, R_v, R_cont
