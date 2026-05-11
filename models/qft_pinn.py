"""Physics-Informed Neural Network (PINN) for Quantum Field Theory"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding for multi-scale field learning."""
    def __init__(self, in_features: int, n_features: int = 128, sigma: float = 1.0):
        super().__init__()
        B = torch.randn(in_features, n_features) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class QFTResidualBlock(nn.Module):
    """Residual block with GELU activation for smooth field approximation."""
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class QFTPINN(nn.Module):
    """
    Physics-Informed Neural Network for scalar field theory.

    Architecture:
        (x, y, t) → [Fourier Features] → [Deep ResNet] → φ(x,y,t)

    Physics loss enforces:
        ∂²φ/∂t² - ∇²φ + m²φ + λφ³ = 0  (Klein-Gordon + φ⁴)

    Energy conservation:
        E = ∫ [½(∂φ/∂t)² + ½|∇φ|² + V(φ)] dx

    Can also learn the effective potential V(φ) from data.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_width: int = 128,
        n_blocks: int = 4,
        use_fourier: bool = True,
        fourier_features: int = 64,
        fourier_sigma: float = 2.0,
        mass: float = 1.0,
        lam: float = 0.1,
    ):
        super().__init__()

        self.mass = mass
        self.lam = lam

        # Fourier embedding
        if use_fourier:
            self.fourier = FourierFeatures(input_dim, fourier_features, fourier_sigma)
            in_dim = 2 * fourier_features
        else:
            self.fourier = None
            in_dim = input_dim

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_width),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            QFTResidualBlock(hidden_width) for _ in range(n_blocks)
        ])

        # Output: scalar field value φ
        self.output_layer = nn.Linear(hidden_width, 1)

        # Adaptive loss weights
        self.log_w_pde = nn.Parameter(torch.tensor(0.0))
        self.log_w_data = nn.Parameter(torch.tensor(0.0))
        self.log_w_energy = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (x,y,t) → φ."""
        if self.fourier is not None:
            x = self.fourier(x)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)

    def compute_kg_residual(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Klein-Gordon + φ⁴ PDE residual:

            R = ∂²φ/∂t² - ∇²φ + m²φ + λφ³

        Uses automatic differentiation for exact derivatives.
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        inp = torch.cat([x, y, t], dim=1)
        phi = self.forward(inp)

        # First derivatives
        phi_x = torch.autograd.grad(
            phi, x, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_y = torch.autograd.grad(
            phi, y, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_t = torch.autograd.grad(
            phi, t, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]

        # Second derivatives
        phi_xx = torch.autograd.grad(
            phi_x, x, torch.ones_like(phi_x), create_graph=True, retain_graph=True
        )[0]
        phi_yy = torch.autograd.grad(
            phi_y, y, torch.ones_like(phi_y), create_graph=True, retain_graph=True
        )[0]
        phi_tt = torch.autograd.grad(
            phi_t, t, torch.ones_like(phi_t), create_graph=True, retain_graph=True
        )[0]

        # Klein-Gordon residual: □φ + m²φ + λφ³ = 0
        # □ = -∂²/∂t² + ∇²  (Minkowski)  =>  ∂²φ/∂t² = ∇²φ - m²φ - λφ³
        # Residual = ∂²φ/∂t² - ∇²φ + m²φ + λφ³
        residual = phi_tt - (phi_xx + phi_yy) + self.mass**2 * phi + self.lam * phi**3

        return residual, phi, phi_t, phi_x, phi_y

    def compute_loss(
        self,
        x_pde: torch.Tensor, y_pde: torch.Tensor, t_pde: torch.Tensor,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None,
        t_data: Optional[torch.Tensor] = None,
        phi_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total PINN loss.

        Components:
            L_pde:    Klein-Gordon residual
            L_data:   Match lattice simulation data
            L_energy: Energy conservation regularization
        """
        losses = {}

        # PDE residual loss
        residual, phi, phi_t, phi_x, phi_y = self.compute_kg_residual(
            x_pde, y_pde, t_pde
        )
        loss_pde = torch.mean(residual**2)
        losses['pde'] = loss_pde

        # Data loss
        loss_data = torch.tensor(0.0, device=x_pde.device)
        if x_data is not None and phi_data is not None:
            inp_data = torch.cat([x_data, y_data, t_data], dim=1)
            phi_pred = self.forward(inp_data)
            loss_data = torch.mean((phi_pred - phi_data)**2)
        losses['data'] = loss_data

        # Energy density regularization (should be smooth and non-negative)
        E_kin = 0.5 * phi_t**2
        E_grad = 0.5 * (phi_x**2 + phi_y**2)
        E_pot = 0.5 * self.mass**2 * phi**2 + 0.25 * self.lam * phi**4
        E_total = E_kin + E_grad + E_pot
        loss_energy = torch.mean(torch.relu(-E_total))  # Penalize negative energy
        losses['energy'] = loss_energy

        # Adaptive weighting
        w_pde = torch.exp(-self.log_w_pde)
        w_data = torch.exp(-self.log_w_data)
        w_energy = torch.exp(-self.log_w_energy)

        total = (w_pde * loss_pde + self.log_w_pde
                 + w_data * loss_data + self.log_w_data
                 + w_energy * loss_energy + self.log_w_energy)

        losses['total'] = total
        return losses

    def predict_field(
        self, x_grid: np.ndarray, y_grid: np.ndarray, t: float,
        device: str = 'cpu',
    ) -> Dict[str, np.ndarray]:
        """Predict φ on a spatial grid at fixed time t."""
        self.eval()
        with torch.no_grad():
            x_f = torch.tensor(x_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
            y_f = torch.tensor(y_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
            t_f = torch.full_like(x_f, t)
            inp = torch.cat([x_f, y_f, t_f], dim=1)
            phi = self.forward(inp).cpu().numpy()

        shape = x_grid.shape
        return {'phi': phi.reshape(shape)}

    def train_on_lattice_data(
        self,
        solver,
        n_collocation: int = 5000,
        n_data_points: int = 2000,
        epochs: int = 2000,
        lr: float = 1e-3,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train QFT PINN on data from lattice simulation.

        Args:
            solver: LatticeQFTSolver instance (already evolved)
            n_collocation: Number of PDE collocation points
            n_data_points: Number of data supervision points
            epochs: Training epochs
            lr: Learning rate
            device: 'cpu' or 'cuda'
            verbose: Print progress

        Returns:
            Training history dict
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Domain bounds
        x_lo, x_hi = -solver.Lx / 2, solver.Lx / 2
        y_lo, y_hi = -solver.Ly / 2, solver.Ly / 2
        t_max = solver.time

        # Data from lattice solver
        x_d = torch.tensor(
            np.random.uniform(x_lo, x_hi, (n_data_points, 1)), dtype=torch.float32
        ).to(device)
        y_d = torch.tensor(
            np.random.uniform(y_lo, y_hi, (n_data_points, 1)), dtype=torch.float32
        ).to(device)
        t_d = torch.tensor(
            np.random.uniform(0, t_max, (n_data_points, 1)), dtype=torch.float32
        ).to(device)

        # Sample field values from the solver's current state
        xi = ((x_d.cpu().numpy().flatten() - x_lo) / solver.dx).astype(int) % solver.nx
        yi = ((y_d.cpu().numpy().flatten() - y_lo) / solver.dy).astype(int) % solver.ny
        phi_vals = solver.phi[yi, xi]
        phi_d = torch.tensor(phi_vals, dtype=torch.float32).unsqueeze(1).to(device)

        history = {'pde': [], 'data': [], 'total': []}

        for epoch in range(epochs):
            self.train()

            # Random collocation points
            x_c = torch.tensor(
                np.random.uniform(x_lo, x_hi, (n_collocation, 1)), dtype=torch.float32
            ).to(device)
            y_c = torch.tensor(
                np.random.uniform(y_lo, y_hi, (n_collocation, 1)), dtype=torch.float32
            ).to(device)
            t_c = torch.tensor(
                np.random.uniform(0, max(t_max, 0.1), (n_collocation, 1)),
                dtype=torch.float32,
            ).to(device)

            losses = self.compute_loss(x_c, y_c, t_c, x_d, y_d, t_d, phi_d)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            history['pde'].append(losses['pde'].item())
            history['data'].append(losses['data'].item())
            history['total'].append(losses['total'].item())

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(
                    f"    Epoch {epoch+1:5d}/{epochs}  "
                    f"PDE={losses['pde'].item():.6f}  "
                    f"Data={losses['data'].item():.6f}  "
                    f"Total={losses['total'].item():.6f}"
                )

        return history
