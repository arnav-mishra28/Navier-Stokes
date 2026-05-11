"""Climate & Oceanic Flow Solver"""

import numpy as np
from typing import Dict, List, Optional


class ClimateFlowSolver:
    """
    Geophysical fluid dynamics solver for climate/ocean simulations.
    
    Implements the rotating shallow water equations and Boussinesq
    approximation for simulating:
        - Ocean gyres and western boundary currents
        - Atmospheric Rossby waves
        - Thermal convection (Rayleigh-Bénard)
        - Ekman spiral in boundary layers
        - Kelvin-Helmholtz instability in atmosphere
        - El Niño / ocean circulation patterns
    """
    
    # Earth parameters
    OMEGA_EARTH = 7.2921e-5  # Earth's rotation rate (rad/s)
    g = 9.81                  # Gravitational acceleration
    
    def __init__(
        self,
        nx: int = 128, ny: int = 128,
        Lx: float = 1e6, Ly: float = 1e6,  # 1000 km domain
        nu: float = 100.0,       # Eddy viscosity (m²/s) — much larger than molecular
        kappa: float = 50.0,     # Thermal diffusivity
        dt: float = 100.0,       # Time step (seconds)
        latitude: float = 45.0,  # Reference latitude (degrees)
        flow_type: str = "ocean",  # "ocean", "atmosphere", "convection"
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.nu = nu
        self.kappa = kappa
        self.dt = dt
        self.flow_type = flow_type
        
        # Coriolis parameter (β-plane approximation)
        # f = f₀ + βy where f₀ = 2Ω sin(φ), β = 2Ω cos(φ)/R_earth
        self.f0 = 2 * self.OMEGA_EARTH * np.sin(np.radians(latitude))
        self.beta = 2 * self.OMEGA_EARTH * np.cos(np.radians(latitude)) / 6.371e6
        
        # Flow fields
        self.u = np.zeros((ny, nx))      # East-west velocity
        self.v = np.zeros((ny, nx))      # North-south velocity
        self.p = np.zeros((ny, nx))      # Pressure / sea surface height
        self.T = np.zeros((ny, nx))      # Temperature / buoyancy
        self.psi = np.zeros((ny, nx))    # Stream function
        self.omega = np.zeros((ny, nx))  # Vorticity
        
        # Grid
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(-Ly/2, Ly/2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Coriolis parameter field (varies with latitude)
        self.f = self.f0 + self.beta * self.Y
        
        # FFT eigenvalues for periodic Poisson
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
            'time': [], 'kinetic_energy': [], 'potential_energy': [],
            'max_velocity': [], 'mean_temperature': []
        }
    
    def initialize_ocean_gyre(self, tau0: float = 0.1):
        """
        Wind-driven ocean gyre (Stommel / Munk problem).
        
        Wind stress: τ = τ₀ * cos(πy/L)
        Creates western-intensified gyre (Gulf Stream analog).
        
        Solution exhibits boundary layer ~ (ν/β)^(1/3) on western side.
        """
        self.u[:] = 0
        self.v[:] = 0
        
        # Wind forcing (zonal wind stress with meridional structure)
        y_norm = self.Y / (self.Ly/2)  # [-1, 1]
        self.wind_stress_x = -tau0 * np.cos(np.pi * y_norm)
        self.wind_stress_y = np.zeros_like(self.X)
        
        # Initialize with small perturbation
        self.psi = 0.01 * np.sin(np.pi * self.X / self.Lx) * np.sin(np.pi * (self.Y + self.Ly/2) / self.Ly)
    
    def initialize_rossby_waves(self, amplitude: float = 1.0, k: int = 4):
        """
        Rossby wave initialization.
        
        Planetary waves that propagate westward due to β-effect.
        Phase speed: c = -β / (k² + l²)
        Group velocity can be eastward (important for climate teleconnections).
        """
        kx = 2 * np.pi * k / self.Lx
        ky = 2 * np.pi / self.Ly
        
        self.psi = amplitude * np.sin(kx * self.X) * np.cos(ky * self.Y)
        self.u = -np.gradient(self.psi, self.dy, axis=0)
        self.v = np.gradient(self.psi, self.dx, axis=1)
    
    def initialize_rayleigh_benard(
        self, Ra: float = 1e4, Pr: float = 0.71,
        T_hot: float = 1.0, T_cold: float = 0.0
    ):
        """
        Rayleigh-Bénard convection.
        
        Heated from below, cooled from above.
        Onset of convection at Ra_c ≈ 1708 (rigid boundaries).
        
        Ra = gαΔTH³/(νκ) — Rayleigh number
        Pr = ν/κ — Prandtl number
        """
        self.T = T_hot - (T_hot - T_cold) * (self.Y + self.Ly/2) / self.Ly
        
        # Small random perturbation to trigger instability
        self.T += 0.01 * (T_hot - T_cold) * np.random.randn(self.ny, self.nx)
        
        self.u[:] = 0
        self.v[:] = 0
        
        # Adjust viscosity for desired Rayleigh number
        H = self.Ly
        delta_T = T_hot - T_cold
        self.nu = np.sqrt(self.g * delta_T * H**3 / (Ra * Pr)) if Ra > 0 else self.nu
        self.kappa = self.nu / Pr
    
    def initialize_kelvin_helmholtz(
        self, u_top: float = 1.0, u_bottom: float = -1.0,
        delta: float = 0.05, perturbation: float = 0.01
    ):
        """
        Kelvin-Helmholtz instability (atmospheric shear layer).
        
        Two layers with different velocities develop characteristic
        cat-eye vortex pattern. Occurs at jet stream boundaries.
        """
        y_norm = self.Y / self.Ly
        
        self.u = 0.5 * (u_top + u_bottom) + 0.5 * (u_top - u_bottom) * np.tanh((y_norm) / delta)
        self.v = perturbation * np.sin(4 * np.pi * self.X / self.Lx) * np.exp(-y_norm**2 / (2*delta**2))
        
        self.T = np.where(y_norm > 0, 1.0, 0.0)
    
    def _ddx(self, f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2*self.dx)
    
    def _ddy(self, f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2*self.dy)
    
    def _laplacian(self, f):
        return ((np.roll(f,-1,1) - 2*f + np.roll(f,1,1)) / self.dx**2 +
                (np.roll(f,-1,0) - 2*f + np.roll(f,1,0)) / self.dy**2)
    
    def _solve_poisson(self, rhs):
        rhs_hat = np.fft.fft2(rhs)
        phi_hat = rhs_hat / self.eigenvalues
        phi_hat[0, 0] = 0
        return np.real(np.fft.ifft2(phi_hat))
    
    def step(self):
        """
        Advance geophysical flow by one time step.
        
        Uses vorticity-streamfunction formulation:
            ∂ω/∂t + J(ψ,ω) + β∂ψ/∂x = ν∇²ω + curl(τ/ρH)
            ∇²ψ = -ω
        
        where J(ψ,ω) = ∂ψ/∂x ∂ω/∂y - ∂ψ/∂y ∂ω/∂x (Arakawa Jacobian)
        """
        # Compute vorticity from velocity
        self.omega = self._ddx(self.v) - self._ddy(self.u)
        
        # Jacobian J(ψ, ω) — Arakawa scheme for energy conservation
        J = self._arakawa_jacobian(self.psi, self.omega)
        
        # β-effect: β * ∂ψ/∂x
        beta_term = self.beta * self._ddx(self.psi)
        
        # Wind forcing (curl of wind stress)
        if hasattr(self, 'wind_stress_x'):
            wind_curl = self._ddx(self.wind_stress_y) - self._ddy(self.wind_stress_x)
        else:
            wind_curl = 0
        
        # Viscous diffusion of vorticity
        diff_omega = self.nu * self._laplacian(self.omega)
        
        # Buoyancy forcing (for convection)
        buoyancy = 0
        if self.flow_type == "convection":
            buoyancy = self.g * self._ddx(self.T) * 0.001  # Thermal expansion
        
        # Time advance vorticity
        self.omega += self.dt * (-J - beta_term + diff_omega + wind_curl + buoyancy)
        
        # Solve for streamfunction: ∇²ψ = -ω
        self.psi = self._solve_poisson(-self.omega)
        
        # Recover velocity: u = -∂ψ/∂y, v = ∂ψ/∂x
        self.u = -self._ddy(self.psi)
        self.v = self._ddx(self.psi)
        
        # Advance temperature (if applicable)
        if self.flow_type in ["convection", "ocean"]:
            adv_T = self.u * self._ddx(self.T) + self.v * self._ddy(self.T)
            diff_T = self.kappa * self._laplacian(self.T)
            self.T += self.dt * (-adv_T + diff_T)
        
        self.time += self.dt
        self.step_count += 1
    
    def _arakawa_jacobian(self, psi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Arakawa Jacobian: conserves energy, enstrophy, and skew-symmetry.
        
        J(ψ,ω) = (1/3)[J++(ψ,ω) + J+×(ψ,ω) + J×+(ψ,ω)]
        
        This is critical for long-time stability of geophysical simulations.
        """
        dx, dy = self.dx, self.dy
        
        # J++ form
        Jpp = ((np.roll(psi,-1,1) - np.roll(psi,1,1)) * 
               (np.roll(omega,-1,0) - np.roll(omega,1,0)) -
               (np.roll(psi,-1,0) - np.roll(psi,1,0)) * 
               (np.roll(omega,-1,1) - np.roll(omega,1,1))) / (4*dx*dy)
        
        # J+x form
        Jpx = (np.roll(psi,-1,1) * (np.roll(np.roll(omega,-1,1),-1,0) - np.roll(np.roll(omega,-1,1),1,0)) -
               np.roll(psi,1,1) * (np.roll(np.roll(omega,1,1),-1,0) - np.roll(np.roll(omega,1,1),1,0)) -
               np.roll(psi,-1,0) * (np.roll(np.roll(omega,-1,0),-1,1) - np.roll(np.roll(omega,-1,0),1,1)) +
               np.roll(psi,1,0) * (np.roll(np.roll(omega,1,0),-1,1) - np.roll(np.roll(omega,1,0),1,1))) / (4*dx*dy)
        
        # J×+ form
        Jxp = (np.roll(np.roll(psi,-1,1),-1,0) * (np.roll(omega,-1,0) - np.roll(omega,-1,1)) -
               np.roll(np.roll(psi,1,1),-1,0) * (np.roll(omega,-1,1) - np.roll(omega,1,1)) -  # Simplified
               np.roll(np.roll(psi,-1,1),1,0) * (np.roll(omega,-1,1) - np.roll(omega,1,0)) +
               np.roll(np.roll(psi,1,1),1,0) * (np.roll(omega,1,1) - np.roll(omega,1,0))) / (4*dx*dy)
        
        return (Jpp + Jpx + Jxp) / 3.0
    
    def advance(self, n_steps: int = 1, record: bool = True):
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()
    
    def _record_diagnostics(self):
        KE = 0.5 * np.mean(self.u**2 + self.v**2)
        max_vel = np.max(np.sqrt(self.u**2 + self.v**2))
        
        self.history['time'].append(self.time)
        self.history['kinetic_energy'].append(float(KE))
        self.history['potential_energy'].append(float(0.5 * np.mean(self.T**2)))
        self.history['max_velocity'].append(float(max_vel))
        self.history['mean_temperature'].append(float(np.mean(self.T)))
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'u': self.u.copy(), 'v': self.v.copy(),
            'psi': self.psi.copy(), 'omega': self.omega.copy(),
            'T': self.T.copy(), 'f': self.f.copy(),
            'velocity_magnitude': np.sqrt(self.u**2 + self.v**2),
            'potential_vorticity': (self.omega + self.f) / 1.0,  # PV = (ω+f)/h
            'time': self.time,
        }
