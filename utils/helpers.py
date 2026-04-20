"""
=============================================================================
Utility Functions & Helpers
Grid generation, diagnostics, I/O, and performance tools.
=============================================================================
"""

import numpy as np
import time
from typing import Tuple, Optional


class Timer:
    """High-resolution performance timer with context manager support."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
        self.laps = []
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        self.laps.append(self.elapsed)
    
    def lap(self) -> float:
        """Record a lap time."""
        now = time.perf_counter()
        lap_time = now - self.start_time
        self.laps.append(lap_time)
        self.start_time = now
        return lap_time
    
    @property
    def average(self) -> float:
        return np.mean(self.laps) if self.laps else 0.0
    
    def __repr__(self):
        return f"Timer({self.name}: {self.elapsed:.4f}s, avg={self.average:.4f}s)"


def create_meshgrid(
    nx: int, ny: int,
    Lx: float = 1.0, Ly: float = 1.0,
    include_endpoints: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D meshgrid for the simulation domain.
    
    Returns:
        X, Y: 2D coordinate arrays of shape (ny, nx)
    """
    if include_endpoints:
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
    else:
        dx = Lx / nx
        dy = Ly / ny
        x = np.linspace(dx/2, Lx - dx/2, nx)
        y = np.linspace(dy/2, Ly - dy/2, ny)
    
    X, Y = np.meshgrid(x, y)
    return X, Y


def create_meshgrid_3d(
    nx: int, ny: int, nz: int,
    Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 3D meshgrid."""
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z


def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute 2D vorticity: ω = ∂v/∂x - ∂u/∂y
    
    Uses second-order central differences with zero-padding at boundaries.
    """
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)
    
    # Central differences in interior
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    
    # Forward/backward at boundaries
    dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx
    dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx
    dudy[0, :] = (u[1, :] - u[0, :]) / dy
    dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy
    
    return dvdx - dudy


def compute_vorticity_3d(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    dx: float, dy: float, dz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D vorticity vector: ω = ∇ × u
    
    Returns: (omega_x, omega_y, omega_z)
    """
    # ω_x = ∂w/∂y - ∂v/∂z
    dwdy = np.gradient(w, dy, axis=1)
    dvdz = np.gradient(v, dz, axis=2)
    omega_x = dwdy - dvdz
    
    # ω_y = ∂u/∂z - ∂w/∂x
    dudz = np.gradient(u, dz, axis=2)
    dwdx = np.gradient(w, dx, axis=0)
    omega_y = dudz - dwdx
    
    # ω_z = ∂v/∂x - ∂u/∂y
    dvdx = np.gradient(v, dx, axis=0)
    dudy = np.gradient(u, dy, axis=1)
    omega_z = dvdx - dudy
    
    return omega_x, omega_y, omega_z


def compute_divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute velocity divergence: ∇·u = ∂u/∂x + ∂v/∂y
    
    For incompressible flow, this should be ~0.
    """
    dudx = np.zeros_like(u)
    dvdy = np.zeros_like(v)
    
    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    
    return dudx + dvdy


def compute_kinetic_energy(u: np.ndarray, v: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Compute total kinetic energy: E = 0.5 * ∫|u|² dV"""
    ke = 0.5 * np.mean(u**2 + v**2)
    if w is not None:
        ke += 0.5 * np.mean(w**2)
    return float(ke)


def compute_enstrophy(omega: np.ndarray) -> float:
    """Compute enstrophy: ε = 0.5 * ∫|ω|² dV"""
    return float(0.5 * np.mean(omega**2))


def compute_strain_rate(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute magnitude of strain rate tensor |S|.
    
    S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    |S| = sqrt(2 * S_ij * S_ij)
    """
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)
    
    return np.sqrt(2.0 * (S11**2 + S22**2 + 2*S12**2))


def compute_q_criterion(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute Q-criterion for vortex identification.
    
    Q = 0.5 * (|Ω|² - |S|²)
    where Ω is the rotation tensor and S is the strain tensor.
    """
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    
    # Strain rate magnitude squared
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)
    S_sq = S11**2 + S22**2 + 2*S12**2
    
    # Rotation rate magnitude squared
    O12 = 0.5 * (dudy - dvdx)
    O_sq = 2 * O12**2
    
    return 0.5 * (O_sq - S_sq)


def create_obstacle_mask(
    nx: int, ny: int,
    obstacle_type: str = "cylinder",
    center: Optional[Tuple[float, float]] = None,
    radius: float = 0.1,
    Lx: float = 1.0, Ly: float = 1.0
) -> np.ndarray:
    """
    Create a boolean mask for solid obstacles.
    
    Args:
        obstacle_type: "cylinder", "square", "airfoil", "custom"
        center: (cx, cy) normalized coordinates (0-1 of domain)
        radius: obstacle radius (fraction of domain)
    
    Returns:
        mask: boolean array (True = solid, False = fluid)
    """
    if center is None:
        center = (0.25, 0.5)
    
    X, Y = create_meshgrid(nx, ny, Lx, Ly)
    cx, cy = center[0] * Lx, center[1] * Ly
    
    if obstacle_type == "cylinder":
        r = radius * min(Lx, Ly)
        mask = (X - cx)**2 + (Y - cy)**2 < r**2
    
    elif obstacle_type == "square":
        r = radius * min(Lx, Ly)
        mask = (np.abs(X - cx) < r) & (np.abs(Y - cy) < r)
    
    elif obstacle_type == "airfoil":
        # NACA 0012 approximation
        r = radius * Lx
        chord = 2 * r
        x_local = (X - cx + r) / chord
        # Thickness distribution
        t = 0.12
        yt = 5 * t * (0.2969*np.sqrt(np.abs(x_local)) - 0.1260*x_local 
                       - 0.3516*x_local**2 + 0.2843*x_local**3 - 0.1015*x_local**4)
        yt *= chord
        mask = (x_local >= 0) & (x_local <= 1) & (np.abs(Y - cy) < yt)
    
    else:
        mask = np.zeros((ny, nx), dtype=bool)
    
    return mask


def compute_cfl(u: np.ndarray, v: np.ndarray, dx: float, dy: float, dt: float) -> float:
    """Compute CFL number."""
    max_u = np.max(np.abs(u)) + 1e-10
    max_v = np.max(np.abs(v)) + 1e-10
    return float(dt * (max_u / dx + max_v / dy))


def adaptive_timestep(
    u: np.ndarray, v: np.ndarray,
    dx: float, dy: float,
    nu: float,
    cfl_target: float = 0.5
) -> float:
    """
    Compute adaptive time step based on CFL and diffusion constraints.
    
    dt = min(CFL constraint, diffusion constraint)
    """
    max_u = np.max(np.abs(u)) + 1e-10
    max_v = np.max(np.abs(v)) + 1e-10
    
    # Convective CFL
    dt_conv = cfl_target / (max_u/dx + max_v/dy)
    
    # Diffusive stability
    dt_diff = 0.5 * min(dx, dy)**2 / (nu + 1e-10) * 0.25
    
    return min(dt_conv, dt_diff)


def pressure_drop(p: np.ndarray, axis: int = 1) -> float:
    """Compute pressure drop across the domain."""
    return float(np.mean(p[:, 0]) - np.mean(p[:, -1]))


def compute_drag_lift(
    p: np.ndarray, u: np.ndarray, v: np.ndarray,
    mask: np.ndarray, dx: float, dy: float, nu: float
) -> Tuple[float, float]:
    """
    Compute drag and lift coefficients on an obstacle.
    
    Uses surface integration of pressure and viscous forces.
    """
    # Find boundary cells (fluid cells adjacent to obstacle)
    from scipy.ndimage import binary_dilation
    boundary = binary_dilation(mask, iterations=1) & ~mask
    
    # Approximate pressure-based forces
    # Gradient of mask gives surface normals
    ny_grad = np.gradient(mask.astype(float), dy, axis=0)
    nx_grad = np.gradient(mask.astype(float), dx, axis=1)
    
    # Pressure force
    Fx_p = -np.sum(p * nx_grad) * dx * dy
    Fy_p = -np.sum(p * ny_grad) * dx * dy
    
    # Viscous force (simplified)
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    
    Fx_v = nu * np.sum((2*dudx*nx_grad + (dudy+dvdx)*ny_grad)) * dx * dy
    Fy_v = nu * np.sum(((dudy+dvdx)*nx_grad + 2*dvdy*ny_grad)) * dx * dy
    
    drag = Fx_p + Fx_v
    lift = Fy_p + Fy_v
    
    return float(drag), float(lift)
