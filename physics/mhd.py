"""
=============================================================================
Magnetohydrodynamics (MHD) Solver
Navier-Stokes + Maxwell's equations coupling.

Governing equations (incompressible MHD):
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u + (1/μ₀ρ)(B·∇)B + f
    ∂B/∂t = ∇×(u×B) + η∇²B
    ∇·u = 0
    ∇·B = 0

where:
    B = magnetic field vector
    η = magnetic diffusivity (resistivity)
    μ₀ = permeability of free space
    J = ∇×B / μ₀ = current density
    Lorentz force = J × B = (1/μ₀)(B·∇)B - ∇(B²/2μ₀)

Applications:
    - Plasma confinement (fusion reactors, tokamaks)
    - Solar physics (coronal mass ejections, sunspots)
    - Astrophysical jets and accretion disks
    - Liquid metal flows (steel casting, cooling)
    - Geodynamo (Earth's magnetic field generation)
=============================================================================
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


class MHDSolver:
    """
    2D incompressible MHD solver using projection method.
    
    Solves coupled Navier-Stokes + induction equations.
    
    The key physics: magnetic field creates Lorentz force that
    couples back to fluid motion, while fluid motion stretches
    and advects field lines.
    """
    
    def __init__(
        self,
        nx: int = 128, ny: int = 128,
        Lx: float = 2*np.pi, Ly: float = 2*np.pi,
        nu: float = 0.01,    # Kinematic viscosity
        eta: float = 0.01,   # Magnetic diffusivity
        dt: float = 0.001,
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.nu = nu
        self.eta = eta
        self.dt = dt
        
        # Velocity field
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        # Magnetic field (B = (Bx, By))
        self.Bx = np.zeros((ny, nx))
        self.By = np.zeros((ny, nx))
        
        # Current density Jz = ∂Bx/∂y - ∂By/∂x (only z-component in 2D)
        self.Jz = np.zeros((ny, nx))
        
        # Grids
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
        # FFT eigenvalues for periodic Poisson solver
        kx = np.arange(nx)
        ky = np.arange(ny)[:, None]
        self.eigenvalues = (
            2 * (np.cos(2*np.pi*kx/nx) - 1) / self.dx**2 +
            2 * (np.cos(2*np.pi*ky/ny) - 1) / self.dy**2
        )
        self.eigenvalues[0, 0] = 1.0
        
        self.time = 0.0
        self.step_count = 0
        
        # Dimensionless numbers
        self.magnetic_reynolds = 1.0 / max(eta, 1e-12)
        self.magnetic_prandtl = nu / max(eta, 1e-12)
        
        self.history: Dict[str, List[float]] = {
            'time': [], 'kinetic_energy': [], 'magnetic_energy': [],
            'total_energy': [], 'cross_helicity': [], 'current_max': []
        }
    
    def initialize_orszag_tang(self, amplitude: float = 1.0):
        """
        Orszag-Tang vortex: canonical MHD benchmark.
        
        Develops complex current sheets and magnetic reconnection.
        Tests: shock capturing, reconnection physics, energy cascade.
        
        u = -sin(y), v = sin(x)
        Bx = -sin(y), By = sin(2x)
        """
        self.u = -amplitude * np.sin(self.Y)
        self.v = amplitude * np.sin(self.X)
        self.Bx = -amplitude * np.sin(self.Y)
        self.By = amplitude * np.sin(2 * self.X)
    
    def initialize_harris_sheet(self, B0: float = 1.0, delta: float = 0.1):
        """
        Harris current sheet: magnetic reconnection setup.
        
        B = B0 * tanh(y/δ) * x̂
        
        Perturbation triggers tearing mode instability → reconnection.
        """
        y_centered = self.Y - self.Ly / 2
        self.Bx = B0 * np.tanh(y_centered / (delta * self.Ly))
        self.By = np.zeros_like(self.Bx)
        
        # Small perturbation to seed instability
        perturbation = 0.01 * B0
        self.By += perturbation * np.sin(2 * np.pi * self.X / self.Lx)
        
        self.u[:] = 0
        self.v[:] = 0
    
    def initialize_alfven_wave(self, B0: float = 1.0, amplitude: float = 0.1):
        """
        Circularly polarized Alfvén wave (exact MHD solution).
        
        Propagates at Alfvén speed: v_A = B₀ / √(μ₀ρ)
        """
        k = 2 * np.pi / self.Lx
        self.Bx = B0 * np.ones_like(self.X)
        self.By = amplitude * B0 * np.sin(k * self.X)
        self.u[:] = 0
        self.v = -amplitude * np.sin(k * self.X)  # Equipartition
    
    def _rollx(self, f, shift):
        return np.roll(f, shift, axis=1)
    
    def _rolly(self, f, shift):
        return np.roll(f, shift, axis=0)
    
    def _ddx(self, f):
        return (self._rollx(f, -1) - self._rollx(f, 1)) / (2 * self.dx)
    
    def _ddy(self, f):
        return (self._rolly(f, -1) - self._rolly(f, 1)) / (2 * self.dy)
    
    def _laplacian(self, f):
        return ((self._rollx(f, -1) - 2*f + self._rollx(f, 1)) / self.dx**2 +
                (self._rolly(f, -1) - 2*f + self._rolly(f, 1)) / self.dy**2)
    
    def _solve_poisson(self, rhs):
        rhs_hat = np.fft.fft2(rhs)
        p_hat = rhs_hat / self.eigenvalues
        p_hat[0, 0] = 0
        return np.real(np.fft.ifft2(p_hat))
    
    def _compute_lorentz_force(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Lorentz force: F_L = J × B = (∇×B) × B
        
        In 2D with Bz=0:
            Jz = ∂By/∂x - ∂Bx/∂y
            F_Lx = Jz * By
            F_Ly = -Jz * Bx
        """
        self.Jz = self._ddx(self.By) - self._ddy(self.Bx)
        
        Fx = self.Jz * self.By
        Fy = -self.Jz * self.Bx
        
        return Fx, Fy
    
    def step(self):
        """
        Advance MHD system by one time step.
        
        Uses a split approach:
        1. Compute Lorentz force
        2. Advance velocity (NS + Lorentz force)
        3. Advance magnetic field (induction equation)
        4. Project velocity (divergence-free)
        5. Clean magnetic field (divergence-free)
        """
        # Lorentz force
        FLx, FLy = self._compute_lorentz_force()
        
        # ---- Velocity prediction ----
        # Advection: -(u·∇)u
        adv_u = self.u * self._ddx(self.u) + self.v * self._ddy(self.u)
        adv_v = self.u * self._ddx(self.v) + self.v * self._ddy(self.v)
        
        # Diffusion: ν∇²u
        diff_u = self.nu * self._laplacian(self.u)
        diff_v = self.nu * self._laplacian(self.v)
        
        u_star = self.u + self.dt * (-adv_u + diff_u + FLx)
        v_star = self.v + self.dt * (-adv_v + diff_v + FLy)
        
        # ---- Pressure projection ----
        div_ustar = self._ddx(u_star) + self._ddy(v_star)
        self.p = self._solve_poisson(div_ustar / self.dt)
        
        self.u = u_star - self.dt * self._ddx(self.p)
        self.v = v_star - self.dt * self._ddy(self.p)
        
        # ---- Magnetic field update (induction equation) ----
        # ∂B/∂t = ∇×(u×B) + η∇²B
        # In 2D: ∂Bx/∂t = ∂(uBx - vBx... actually:
        # ∂Bx/∂t = -u∂Bx/∂x - v∂Bx/∂y + Bx∂u/∂x + By∂u/∂y + η∇²Bx
        # ∂By/∂t = -u∂By/∂x - v∂By/∂y + Bx∂v/∂x + By∂v/∂y + η∇²By
        
        # Advection of B
        adv_Bx = self.u * self._ddx(self.Bx) + self.v * self._ddy(self.Bx)
        adv_By = self.u * self._ddx(self.By) + self.v * self._ddy(self.By)
        
        # Stretching of B
        stretch_Bx = self.Bx * self._ddx(self.u) + self.By * self._ddy(self.u)
        stretch_By = self.Bx * self._ddx(self.v) + self.By * self._ddy(self.v)
        
        # Resistive diffusion
        diff_Bx = self.eta * self._laplacian(self.Bx)
        diff_By = self.eta * self._laplacian(self.By)
        
        self.Bx += self.dt * (-adv_Bx + stretch_Bx + diff_Bx)
        self.By += self.dt * (-adv_By + stretch_By + diff_By)
        
        # ---- Divergence cleaning for B (projection) ----
        div_B = self._ddx(self.Bx) + self._ddy(self.By)
        phi = self._solve_poisson(div_B)
        self.Bx -= self._ddx(phi)
        self.By -= self._ddy(phi)
        
        self.time += self.dt
        self.step_count += 1
    
    def advance(self, n_steps: int = 1, record: bool = True):
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()
    
    def _record_diagnostics(self):
        KE = 0.5 * np.mean(self.u**2 + self.v**2)
        ME = 0.5 * np.mean(self.Bx**2 + self.By**2)
        cross_helicity = np.mean(self.u * self.Bx + self.v * self.By)
        
        self.history['time'].append(self.time)
        self.history['kinetic_energy'].append(float(KE))
        self.history['magnetic_energy'].append(float(ME))
        self.history['total_energy'].append(float(KE + ME))
        self.history['cross_helicity'].append(float(cross_helicity))
        self.history['current_max'].append(float(np.max(np.abs(self.Jz))))
    
    def get_state(self) -> Dict[str, np.ndarray]:
        self.Jz = self._ddx(self.By) - self._ddy(self.Bx)
        B_mag = np.sqrt(self.Bx**2 + self.By**2)
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        
        return {
            'u': self.u.copy(), 'v': self.v.copy(), 'p': self.p.copy(),
            'Bx': self.Bx.copy(), 'By': self.By.copy(),
            'Jz': self.Jz.copy(), 'B_magnitude': B_mag,
            'velocity_magnitude': vel_mag,
            'time': self.time,
        }
