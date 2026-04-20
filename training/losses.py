"""
=============================================================================
Physics-Informed Loss Functions
Custom loss functions that embed physical laws into neural network training.
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class PhysicsInformedLoss(nn.Module):
    """
    Comprehensive physics-informed loss for Navier-Stokes.
    
    L_total = w_data·L_data + w_pde·L_pde + w_bc·L_bc + w_ic·L_ic 
            + w_div·L_div + w_energy·L_energy + w_symmetry·L_symmetry
    
    Components:
        L_pde: NS residual at collocation points
        L_div: Divergence-free constraint
        L_bc: Boundary condition enforcement
        L_ic: Initial condition enforcement
        L_data: Supervised data loss (if available)
        L_energy: Energy conservation
        L_symmetry: Symmetry constraints
    """
    
    def __init__(
        self,
        nu: float = 0.01,
        weights: Optional[Dict[str, float]] = None,
        adaptive: bool = True,
    ):
        super().__init__()
        self.nu = nu
        self.adaptive = adaptive
        
        default_weights = {
            'pde': 1.0, 'div': 10.0, 'bc': 10.0,
            'ic': 10.0, 'data': 1.0, 'energy': 0.1,
        }
        if weights:
            default_weights.update(weights)
        
        if adaptive:
            self.log_weights = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(np.log(v)))
                for k, v in default_weights.items()
            })
        else:
            self.weights = default_weights
    
    def get_weight(self, key: str) -> float:
        if self.adaptive:
            return torch.exp(self.log_weights[key])
        return self.weights.get(key, 1.0)
    
    def ns_residual(
        self, model: nn.Module,
        x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute NS residuals using autograd."""
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        inp = torch.cat([x, y, t], dim=1)
        out = model(inp)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        
        # Helper for gradients
        def grad(f, var):
            return torch.autograd.grad(
                f, var, torch.ones_like(f),
                create_graph=True, retain_graph=True
            )[0]
        
        # First derivatives
        u_x, u_y, u_t = grad(u, x), grad(u, y), grad(u, t)
        v_x, v_y, v_t = grad(v, x), grad(v, y), grad(v, t)
        p_x, p_y = grad(p, x), grad(p, y)
        
        # Second derivatives
        u_xx, u_yy = grad(u_x, x), grad(u_y, y)
        v_xx, v_yy = grad(v_x, x), grad(v_y, y)
        
        # NS residuals
        R_u = u_t + u*u_x + v*u_y + p_x - self.nu*(u_xx + u_yy)
        R_v = v_t + u*v_x + v*v_y + p_y - self.nu*(v_xx + v_yy)
        R_div = u_x + v_y
        
        return R_u, R_v, R_div
    
    def pde_loss(
        self, model: nn.Module,
        x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """PDE residual loss at collocation points."""
        R_u, R_v, R_div = self.ns_residual(model, x, y, t)
        return torch.mean(R_u**2) + torch.mean(R_v**2) + torch.mean(R_div**2)
    
    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Supervised data loss."""
        diff = pred - target
        if mask is not None:
            diff = diff * mask
        return torch.mean(diff**2)
    
    def boundary_loss(
        self, model: nn.Module,
        x_bc: torch.Tensor, y_bc: torch.Tensor, t_bc: torch.Tensor,
        u_bc: torch.Tensor, v_bc: torch.Tensor
    ) -> torch.Tensor:
        """Boundary condition loss."""
        inp = torch.cat([x_bc, y_bc, t_bc], dim=1)
        pred = model(inp)
        return (torch.mean((pred[:, 0:1] - u_bc)**2) +
                torch.mean((pred[:, 1:2] - v_bc)**2))
    
    def divergence_loss(
        self, u: torch.Tensor, v: torch.Tensor,
        dx: float, dy: float
    ) -> torch.Tensor:
        """Divergence-free constraint for grid-based predictions."""
        dudx = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / (2*dx)
        dvdy = (torch.roll(v, -1, -2) - torch.roll(v, 1, -2)) / (2*dy)
        return torch.mean((dudx + dvdy)**2)
    
    def energy_conservation_loss(
        self,
        u_pred: torch.Tensor, v_pred: torch.Tensor,
        u_prev: torch.Tensor, v_prev: torch.Tensor,
        dt: float, nu: float
    ) -> torch.Tensor:
        """
        Soft energy conservation constraint.
        
        For inviscid flow: dE/dt = 0
        For viscous flow: dE/dt = -2ν * Enstrophy ≤ 0
        """
        E_pred = 0.5 * torch.mean(u_pred**2 + v_pred**2)
        E_prev = 0.5 * torch.mean(u_prev**2 + v_prev**2)
        
        dEdt = (E_pred - E_prev) / dt
        
        # Energy should decrease (dissipation)
        return torch.relu(dEdt)**2  # Penalize energy increase
    
    def forward(
        self, model: nn.Module,
        x_pde: torch.Tensor, y_pde: torch.Tensor, t_pde: torch.Tensor,
        x_bc: torch.Tensor, y_bc: torch.Tensor, t_bc: torch.Tensor,
        u_bc: torch.Tensor, v_bc: torch.Tensor,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None,
        t_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        v_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with all components."""
        losses = {}
        
        # PDE loss
        L_pde = self.pde_loss(model, x_pde, y_pde, t_pde)
        losses['pde'] = L_pde
        
        # BC loss
        L_bc = self.boundary_loss(model, x_bc, y_bc, t_bc, u_bc, v_bc)
        losses['bc'] = L_bc
        
        # Data loss
        if x_data is not None:
            inp = torch.cat([x_data, y_data, t_data], dim=1)
            pred = model(inp)
            target = torch.cat([u_data, v_data], dim=1)
            L_data = self.data_loss(pred[:, :2], target)
            losses['data'] = L_data
        
        # Total
        total = sum(self.get_weight(k) * v for k, v in losses.items())
        
        # Add log-variance terms for adaptive weighting
        if self.adaptive:
            total += sum(self.log_weights[k] for k in losses.keys() if k in self.log_weights)
        
        losses['total'] = total
        return losses


class FNOLoss(nn.Module):
    """Loss function for Fourier Neural Operator training."""
    
    def __init__(self, physics_weight: float = 0.1, dx: float = 0.1, dy: float = 0.1):
        super().__init__()
        self.physics_weight = physics_weight
        self.dx = dx
        self.dy = dy
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        include_physics: bool = True
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Relative L2 loss (standard for FNO)
        diff = pred - target
        losses['l2'] = torch.mean(diff**2) / (torch.mean(target**2) + 1e-8)
        
        # H1 Sobolev loss (penalizes gradient differences)
        dpred_dx = torch.diff(pred, dim=-1)
        dtarget_dx = torch.diff(target, dim=-1)
        dpred_dy = torch.diff(pred, dim=-2)
        dtarget_dy = torch.diff(target, dim=-2)
        
        losses['h1'] = (torch.mean((dpred_dx - dtarget_dx)**2) +
                       torch.mean((dpred_dy - dtarget_dy)**2))
        
        # Physics loss (divergence-free)
        if include_physics and pred.shape[1] >= 2:
            u = pred[:, 0:1]
            v = pred[:, 1:2]
            dudx = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / (2*self.dx)
            dvdy = (torch.roll(v, -1, -2) - torch.roll(v, 1, -2)) / (2*self.dy)
            losses['div'] = torch.mean((dudx + dvdy)**2)
        
        total = losses['l2'] + 0.1 * losses['h1']
        if 'div' in losses:
            total += self.physics_weight * losses['div']
        
        losses['total'] = total
        return losses
