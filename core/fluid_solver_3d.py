"""3D Incompressible Navier-Stokes Solver"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class FluidSolver3D:
    """
    3D incompressible Navier-Stokes solver using projection method.
    
    Uses collocated grid with FFT-based pressure solver.
    Optimized for periodic boundary conditions.
    """
    
    def __init__(
        self,
        nx: int = 64, ny: int = 64, nz: int = 64,
        Lx: float = 2*np.pi, Ly: float = 2*np.pi, Lz: float = 2*np.pi,
        nu: float = 0.01, dt: float = 0.001
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz
        self.nu = nu
        self.dt = dt
        
        # Velocity fields
        self.u = np.zeros((nz, ny, nx))
        self.v = np.zeros((nz, ny, nx))
        self.w = np.zeros((nz, ny, nx))
        self.p = np.zeros((nz, ny, nx))
        
        # Coordinate grids
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        z = np.linspace(0, Lz, nz, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Pre-compute FFT eigenvalues for 3D Poisson solver
        kx = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.dy) * 2 * np.pi
        kz = np.fft.fftfreq(nz, d=self.dz) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        self.k_sq = KX**2 + KY**2 + KZ**2
        self.k_sq[0, 0, 0] = 1.0  # Avoid division by zero
        
        # Modified wavenumbers for 2nd order FD
        self.eigenvalues_3d = (
            2 * (np.cos(2*np.pi*np.arange(nx)/nx) - 1)[:, None, None] / self.dx**2 +
            2 * (np.cos(2*np.pi*np.arange(ny)/ny) - 1)[None, :, None] / self.dy**2 +
            2 * (np.cos(2*np.pi*np.arange(nz)/nz) - 1)[None, None, :] / self.dz**2
        )
        self.eigenvalues_3d[0, 0, 0] = 1.0
        
        self.time = 0.0
        self.step_count = 0
        
        self.history: Dict[str, List[float]] = {
            'time': [], 'kinetic_energy': [], 'enstrophy': [],
            'max_velocity': [], 'dissipation_rate': []
        }
    
    def initialize_taylor_green_3d(self, A: float = 1.0):
        """
        3D Taylor-Green vortex (canonical turbulence benchmark).
        
        u = A cos(x) sin(y) cos(z)
        v = -A sin(x) cos(y) cos(z)
        w = 0
        p = (A²/16)(cos(2x)+cos(2y))(cos(2z)+2)
        
        At Re ~ 1600, this develops into fully turbulent flow.
        """
        self.u = A * np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        self.v = -A * np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        self.w = np.zeros_like(self.u)
        self.p = (A**2/16) * (np.cos(2*self.X) + np.cos(2*self.Y)) * (np.cos(2*self.Z) + 2)
    
    def initialize_abc_flow(self, A: float = 1.0, B: float = 1.0, C: float = 1.0):
        """
        Arnold-Beltrami-Childress (ABC) flow.
        
        Exact stationary solution of Euler equations.
        Important for studying Lagrangian chaos.
        """
        self.u = A * np.sin(self.Z) + C * np.cos(self.Y)
        self.v = B * np.sin(self.X) + A * np.cos(self.Z)
        self.w = C * np.sin(self.Y) + B * np.cos(self.X)
    
    def _laplacian_3d(self, phi: np.ndarray) -> np.ndarray:
        """Compute 3D Laplacian using periodic central differences."""
        lap = (
            (np.roll(phi, -1, 0) - 2*phi + np.roll(phi, 1, 0)) / self.dx**2 +
            (np.roll(phi, -1, 1) - 2*phi + np.roll(phi, 1, 1)) / self.dy**2 +
            (np.roll(phi, -1, 2) - 2*phi + np.roll(phi, 1, 2)) / self.dz**2
        )
        return lap
    
    def _advection_3d(self, phi: np.ndarray) -> np.ndarray:
        """Compute (u·∇)φ using central differences with periodic BCs."""
        dpdx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2 * self.dx)
        dpdy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2 * self.dy)
        dpdz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / (2 * self.dz)
        return self.u * dpdx + self.v * dpdy + self.w * dpdz
    
    def _divergence_3d(self, u, v, w) -> np.ndarray:
        """Compute ∇·(u,v,w)."""
        dudx = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * self.dx)
        dvdy = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * self.dy)
        dwdz = (np.roll(w, -1, 2) - np.roll(w, 1, 2)) / (2 * self.dz)
        return dudx + dvdy + dwdz
    
    def _gradient_3d(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ∇φ."""
        dpdx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2 * self.dx)
        dpdy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2 * self.dy)
        dpdz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / (2 * self.dz)
        return dpdx, dpdy, dpdz
    
    def _solve_pressure_3d(self, rhs: np.ndarray) -> np.ndarray:
        """Solve 3D Poisson equation using FFT."""
        rhs_hat = np.fft.fftn(rhs)
        p_hat = rhs_hat / self.eigenvalues_3d
        p_hat[0, 0, 0] = 0.0
        return np.real(np.fft.ifftn(p_hat))
    
    def step(self):
        """Advance by one time step using 3D projection method."""
        # Predict velocity
        adv_u = self._advection_3d(self.u)
        adv_v = self._advection_3d(self.v)
        adv_w = self._advection_3d(self.w)
        
        diff_u = self.nu * self._laplacian_3d(self.u)
        diff_v = self.nu * self._laplacian_3d(self.v)
        diff_w = self.nu * self._laplacian_3d(self.w)
        
        u_star = self.u + self.dt * (-adv_u + diff_u)
        v_star = self.v + self.dt * (-adv_v + diff_v)
        w_star = self.w + self.dt * (-adv_w + diff_w)
        
        # Solve pressure
        div = self._divergence_3d(u_star, v_star, w_star)
        self.p = self._solve_pressure_3d(div / self.dt)
        
        # Correct velocity
        dpdx, dpdy, dpdz = self._gradient_3d(self.p)
        self.u = u_star - self.dt * dpdx
        self.v = v_star - self.dt * dpdy
        self.w = w_star - self.dt * dpdz
        
        self.time += self.dt
        self.step_count += 1
    
    def advance(self, n_steps: int = 1, record_history: bool = True):
        """Advance multiple steps."""
        for _ in range(n_steps):
            self.step()
            if record_history:
                self._record_diagnostics()
    
    def _record_diagnostics(self):
        """Record 3D flow diagnostics."""
        ke = 0.5 * np.mean(self.u**2 + self.v**2 + self.w**2)
        
        # Vorticity components
        omega_x = ((np.roll(self.w, -1, 1) - np.roll(self.w, 1, 1)) / (2*self.dy) -
                   (np.roll(self.v, -1, 2) - np.roll(self.v, 1, 2)) / (2*self.dz))
        omega_y = ((np.roll(self.u, -1, 2) - np.roll(self.u, 1, 2)) / (2*self.dz) -
                   (np.roll(self.w, -1, 0) - np.roll(self.w, 1, 0)) / (2*self.dx))
        omega_z = ((np.roll(self.v, -1, 0) - np.roll(self.v, 1, 0)) / (2*self.dx) -
                   (np.roll(self.u, -1, 1) - np.roll(self.u, 1, 1)) / (2*self.dy))
        
        enstrophy = 0.5 * np.mean(omega_x**2 + omega_y**2 + omega_z**2)
        
        # Dissipation rate: ε = 2ν * <|S|²>
        dissipation = 2 * self.nu * enstrophy
        
        max_vel = np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))
        
        self.history['time'].append(self.time)
        self.history['kinetic_energy'].append(float(ke))
        self.history['enstrophy'].append(float(enstrophy))
        self.history['max_velocity'].append(float(max_vel))
        self.history['dissipation_rate'].append(float(dissipation))
    
    def compute_energy_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D energy spectrum E(k).
        
        Important for verifying Kolmogorov's -5/3 law in turbulence.
        """
        u_hat = np.fft.fftn(self.u)
        v_hat = np.fft.fftn(self.v)
        w_hat = np.fft.fftn(self.w)
        
        energy_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
        energy_hat /= (self.nx * self.ny * self.nz)**2
        
        # Compute wavenumber magnitudes
        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dy) * 2 * np.pi
        kz = np.fft.fftfreq(self.nz, d=self.dz) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Bin into shells
        k_max = np.max(k_mag) / 2
        n_bins = min(self.nx, self.ny, self.nz) // 2
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        spectrum = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            spectrum[i] = np.sum(energy_hat[mask])
        
        return k_centers, spectrum
    
    def get_slice(self, axis: str = 'z', index: Optional[int] = None) -> Dict:
        """Get a 2D slice of the 3D fields for visualization."""
        if index is None:
            if axis == 'z':
                index = self.nz // 2
            elif axis == 'y':
                index = self.ny // 2
            else:
                index = self.nx // 2
        
        if axis == 'z':
            return {
                'u': self.u[index, :, :],
                'v': self.v[index, :, :],
                'w': self.w[index, :, :],
                'p': self.p[index, :, :],
                'velocity_magnitude': np.sqrt(
                    self.u[index,:,:]**2 + self.v[index,:,:]**2 + self.w[index,:,:]**2
                )
            }
        elif axis == 'y':
            return {
                'u': self.u[:, index, :],
                'v': self.v[:, index, :],
                'w': self.w[:, index, :],
                'p': self.p[:, index, :],
                'velocity_magnitude': np.sqrt(
                    self.u[:,index,:]**2 + self.v[:,index,:]**2 + self.w[:,index,:]**2
                )
            }
        else:
            return {
                'u': self.u[:, :, index],
                'v': self.v[:, :, index],
                'w': self.w[:, :, index],
                'p': self.p[:, :, index],
                'velocity_magnitude': np.sqrt(
                    self.u[:,:,index]**2 + self.v[:,:,index]**2 + self.w[:,:,index]**2
                )
            }
