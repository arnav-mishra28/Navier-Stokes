"""Astrophysical Flow Solver"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class AstrophysicalFlowSolver:
    """
    Astrophysical flow solver with self-gravity and cooling.
    
    Simulates:
        - Accretion disks around compact objects
        - Gravitational collapse and star formation
        - Stellar winds and outflows
        - Rayleigh-Taylor instability in supernovae
    """
    
    # Physical constants (CGS)
    G = 6.674e-8          # Gravitational constant
    k_B = 1.381e-16       # Boltzmann constant
    m_p = 1.673e-24       # Proton mass
    sigma_SB = 5.671e-5   # Stefan-Boltzmann constant
    
    def __init__(
        self,
        nx: int = 128, ny: int = 128,
        Lx: float = 10.0, Ly: float = 10.0,
        nu: float = 0.001,
        dt: float = 0.001,
        gamma: float = 5/3,  # Adiabatic index (5/3 for ideal monoatomic gas)
        include_gravity: bool = True,
        include_cooling: bool = False,
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.nu = nu
        self.dt = dt
        self.gamma = gamma
        self.include_gravity = include_gravity
        self.include_cooling = include_cooling
        
        # Primary variables
        self.rho = np.ones((ny, nx))       # Density
        self.u = np.zeros((ny, nx))        # x-velocity
        self.v = np.zeros((ny, nx))        # y-velocity
        self.p = np.zeros((ny, nx))        # Pressure
        self.e = np.zeros((ny, nx))        # Internal energy
        self.phi = np.zeros((ny, nx))      # Gravitational potential
        self.T = np.zeros((ny, nx))        # Temperature
        
        # Grid
        x = np.linspace(-Lx/2, Lx/2, nx, endpoint=False)
        y = np.linspace(-Ly/2, Ly/2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2 + 0.01**2)  # Softened radius
        
        # FFT eigenvalues
        kx = np.arange(nx)
        ky = np.arange(ny)[:, None]
        self.eigenvalues = (
            2*(np.cos(2*np.pi*kx/nx)-1)/self.dx**2 +
            2*(np.cos(2*np.pi*ky/ny)-1)/self.dy**2
        )
        self.eigenvalues[0, 0] = 1.0
        
        self.time = 0.0
        self.step_count = 0
        self.history: Dict[str, List] = {
            'time': [], 'total_mass': [], 'kinetic_energy': [],
            'potential_energy': [], 'thermal_energy': []
        }
    
    def initialize_keplerian_disk(
        self, M_central: float = 1.0,
        rho0: float = 1.0, T0: float = 1.0,
        r_in: float = 0.5, r_out: float = 4.0
    ):
        """
        Initialize Keplerian accretion disk.
        
        Circular velocity: v_φ = sqrt(GM/r) (Keplerian rotation)
        Density profile: ρ ~ r^(-3/2) (thin disk approximation)
        Temperature: T ~ r^(-1) (viscous heating equilibrium)
        """
        # Density profile
        mask = (self.R > r_in) & (self.R < r_out)
        self.rho = np.where(mask, rho0 * (self.R / r_in)**(-1.5), 0.01 * rho0)
        
        # Keplerian velocity (circular)
        v_kep = np.sqrt(M_central / (self.R + 0.01))
        theta = np.arctan2(self.Y, self.X)
        
        self.u = np.where(mask, -v_kep * np.sin(theta), 0)
        self.v = np.where(mask, v_kep * np.cos(theta), 0)
        
        # Temperature and pressure
        self.T = np.where(mask, T0 * (self.R / r_in)**(-1), 0.01 * T0)
        self.p = self.rho * self.T  # Ideal gas (normalized)
        self.e = self.p / (self.gamma - 1)
        
        # Gravitational potential
        self.phi = -M_central / self.R
    
    def initialize_jeans_collapse(
        self, rho0: float = 1.0, perturbation: float = 0.1,
        cs: float = 1.0
    ):
        """
        Jeans gravitational instability (star formation).
        
        A uniform density cloud with small perturbation collapses
        when perturbation scale > Jeans length:
        λ_J = c_s * sqrt(π / (G * ρ₀))
        
        This creates gravitational fragmentation.
        """
        # Uniform background + sinusoidal perturbation
        k_pert = 2 * np.pi / self.Lx  # One wavelength
        self.rho = rho0 * (1 + perturbation * np.cos(k_pert * self.X))
        
        self.u[:] = 0
        self.v[:] = 0
        self.T = cs**2 * np.ones_like(self.rho)
        self.p = self.rho * self.T
        self.e = self.p / (self.gamma - 1)
    
    def initialize_rayleigh_taylor(
        self, rho_heavy: float = 2.0, rho_light: float = 1.0,
        g: float = -1.0, perturbation: float = 0.01
    ):
        """
        Rayleigh-Taylor instability: heavy fluid on top of light fluid.
        
        Occurs in supernovae, inertial confinement fusion, atmospheric dynamics.
        Growth rate: σ = sqrt(g * k * (ρ₂-ρ₁)/(ρ₂+ρ₁))  (Atwood number)
        """
        interface = self.Ly / 2 + perturbation * np.cos(4 * np.pi * self.X / self.Lx)
        
        self.rho = np.where(self.Y > interface, rho_heavy, rho_light)
        self.u[:] = 0
        self.v[:] = 0
        
        # Hydrostatic pressure
        self.p = 1.0 - self.rho * g * (self.Y - self.Ly/2)
        self.e = self.p / (self.gamma - 1)
    
    def _ddx(self, f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2*self.dx)
    
    def _ddy(self, f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2*self.dy)
    
    def _laplacian(self, f):
        return ((np.roll(f,-1,1) - 2*f + np.roll(f,1,1)) / self.dx**2 +
                (np.roll(f,-1,0) - 2*f + np.roll(f,1,0)) / self.dy**2)
    
    def _solve_gravity(self):
        """Solve Poisson equation for gravitational potential: ∇²Φ = 4πGρ"""
        rhs = 4 * np.pi * self.rho
        rhs_hat = np.fft.fft2(rhs)
        phi_hat = rhs_hat / self.eigenvalues
        phi_hat[0, 0] = 0
        self.phi = np.real(np.fft.ifft2(phi_hat))
    
    def _cooling_function(self, T: np.ndarray) -> np.ndarray:
        """
        Radiative cooling function Λ(T).
        
        Simplified piecewise power-law:
        - T < 10⁴ K: Λ ~ T² (atomic line cooling)
        - T > 10⁴ K: Λ ~ T^0.5 (bremsstrahlung)
        """
        Lambda = np.zeros_like(T)
        mask_low = T < 1.0
        mask_high = T >= 1.0
        
        Lambda[mask_low] = 1e-3 * T[mask_low]**2
        Lambda[mask_high] = 1e-3 * T[mask_high]**0.5
        
        return Lambda
    
    def step(self):
        """Advance compressible flow with gravity by one time step."""
        # Solve gravity
        if self.include_gravity:
            self._solve_gravity()
        
        # Gravitational acceleration
        gx = -self._ddx(self.phi) if self.include_gravity else 0
        gy = -self._ddy(self.phi) if self.include_gravity else 0
        
        # Advection
        adv_u = self.u * self._ddx(self.u) + self.v * self._ddy(self.u)
        adv_v = self.u * self._ddx(self.v) + self.v * self._ddy(self.v)
        
        # Pressure gradient
        dpdx = self._ddx(self.p) / (self.rho + 1e-10)
        dpdy = self._ddy(self.p) / (self.rho + 1e-10)
        
        # Viscous diffusion
        diff_u = self.nu * self._laplacian(self.u)
        diff_v = self.nu * self._laplacian(self.v)
        
        # Update velocity
        self.u += self.dt * (-adv_u - dpdx + diff_u + gx)
        self.v += self.dt * (-adv_v - dpdy + diff_v + gy)
        
        # Update density (continuity equation)
        div_rhou = self._ddx(self.rho * self.u) + self._ddy(self.rho * self.v)
        self.rho -= self.dt * div_rhou
        self.rho = np.maximum(self.rho, 1e-6)
        
        # Update energy
        if self.include_cooling:
            self.T = self.p / (self.rho + 1e-10)
            cooling = self.rho * self._cooling_function(self.T)
            self.e -= self.dt * cooling
        
        # Update pressure from energy
        self.p = (self.gamma - 1) * np.maximum(self.e, 1e-10)
        
        self.time += self.dt
        self.step_count += 1
    
    def advance(self, n_steps: int = 1, record: bool = True):
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()
    
    def _record_diagnostics(self):
        self.history['time'].append(self.time)
        self.history['total_mass'].append(float(np.sum(self.rho) * self.dx * self.dy))
        self.history['kinetic_energy'].append(float(0.5*np.sum(self.rho*(self.u**2+self.v**2))*self.dx*self.dy))
        self.history['potential_energy'].append(float(0.5*np.sum(self.rho*self.phi)*self.dx*self.dy))
        self.history['thermal_energy'].append(float(np.sum(self.e)*self.dx*self.dy))
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'rho': self.rho.copy(), 'u': self.u.copy(), 'v': self.v.copy(),
            'p': self.p.copy(), 'phi': self.phi.copy(), 'T': self.T.copy(),
            'velocity_magnitude': np.sqrt(self.u**2 + self.v**2),
            'vorticity': self._ddx(self.v) - self._ddy(self.u),
            'time': self.time,
        }
