"""Cosmological Fluid Modeling"""

import numpy as np
from typing import Dict, List, Optional


class CosmologicalFluidSolver:
    """
    2D Cosmological N-body + Fluid hybrid solver.

    Solves coupled Vlasov-Poisson + Euler system on expanding background:
      - Friedmann equation for scale factor a(t)
      - Poisson equation for gravitational potential in comoving coords
      - Euler equations for baryonic fluid on expanding background
      - Particle-mesh (PM) for dark matter N-body dynamics

    Units: comoving coordinates, H0 = 1.
    """

    def __init__(
        self,
        nx: int = 256, ny: int = 256,
        Lbox: float = 100.0,
        dt: float = 0.005,
        Omega_m: float = 0.3,
        Omega_Lambda: float = 0.7,
        Omega_b: float = 0.05,
        H0: float = 1.0,
        n_particles: int = 8000,
        seed: int = 42,
    ):
        self.nx, self.ny = nx, ny
        self.Lbox = Lbox
        self.dx = Lbox / nx
        self.dy = Lbox / ny
        self.dt = dt
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_b = Omega_b
        self.Omega_dm = Omega_m - Omega_b
        self.H0 = H0
        self.n_particles = n_particles

        # Scale factor (start at a=0.01, i.e. z=99)
        self.a = 0.01
        self.a_dot = self.H0 * self.a * np.sqrt(
            self.Omega_m / self.a**3 + self.Omega_Lambda
        )

        # Baryonic fluid fields (comoving)
        self.rho_b = Omega_b * np.ones((ny, nx))
        self.vx_b = np.zeros((ny, nx))
        self.vy_b = np.zeros((ny, nx))
        self.temp = 1e-4 * np.ones((ny, nx))  # Temperature proxy

        # Dark matter particle arrays
        np.random.seed(seed)
        self.x_dm = np.random.uniform(0, Lbox, n_particles)
        self.y_dm = np.random.uniform(0, Lbox, n_particles)
        self.vx_dm = np.zeros(n_particles)
        self.vy_dm = np.zeros(n_particles)
        self.m_dm = (Omega_m - Omega_b) * Lbox**2 / n_particles

        # Gravitational potential
        self.Phi = np.zeros((ny, nx))
        self.rho_dm_grid = np.zeros((ny, nx))
        self.rho_total = np.zeros((ny, nx))

        # Grid
        x = np.linspace(0, Lbox, nx, endpoint=False)
        y = np.linspace(0, Lbox, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)

        # FFT Poisson eigenvalues
        kx = np.arange(nx)
        ky = np.arange(ny)[:, None]
        self.poisson_eig = (
            2 * (np.cos(2 * np.pi * kx / nx) - 1) / self.dx**2
            + 2 * (np.cos(2 * np.pi * ky / ny) - 1) / self.dy**2
        )
        self.poisson_eig[0, 0] = 1.0

        self.time = 0.0
        self.step_count = 0
        self.history: Dict[str, List] = {
            'time': [], 'scale_factor': [], 'redshift': [],
            'hubble': [], 'total_mass': [],
            'max_density': [], 'rms_density_contrast': [],
            'kinetic_energy': [], 'potential_energy': [],
            'dm_clumping': [], 'baryon_clumping': [],
        }

    # Finite differences

    def _ddx(self, f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * self.dx)

    def _ddy(self, f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * self.dy)

    def _laplacian(self, f):
        return (
            (np.roll(f, -1, 1) - 2*f + np.roll(f, 1, 1)) / self.dx**2
            + (np.roll(f, -1, 0) - 2*f + np.roll(f, 1, 0)) / self.dy**2
        )

    # Friedmann equation

    def _friedmann_rhs(self, a, a_dot):
        """d(a_dot)/dt from Friedmann + acceleration equation."""
        H2 = self.H0**2 * (self.Omega_m / a**3 + self.Omega_Lambda)
        a_ddot = -0.5 * self.H0**2 * self.Omega_m / a**2 + self.Omega_Lambda * self.H0**2 * a
        return a_ddot

    def _advance_scale_factor(self):
        """RK2 integration of scale factor."""
        a_ddot1 = self._friedmann_rhs(self.a, self.a_dot)
        a_mid = self.a + 0.5 * self.dt * self.a_dot
        ad_mid = self.a_dot + 0.5 * self.dt * a_ddot1
        a_ddot2 = self._friedmann_rhs(a_mid, ad_mid)

        self.a_dot += self.dt * a_ddot2
        self.a += self.dt * self.a_dot
        self.a = max(self.a, 1e-6)

    def hubble(self):
        """Hubble parameter H(a)."""
        return self.a_dot / (self.a + 1e-12)

    def redshift(self):
        return 1.0 / self.a - 1.0

    # CIC deposit dark matter onto grid

    def _deposit_dm(self):
        """Cloud-in-Cell (CIC) deposit of DM particles onto grid."""
        self.rho_dm_grid[:] = 0.0
        # Wrap positions
        xp = self.x_dm % self.Lbox
        yp = self.y_dm % self.Lbox

        # Grid indices
        ix = (xp / self.dx).astype(int) % self.nx
        iy = (yp / self.dy).astype(int) % self.ny

        fx = xp / self.dx - ix
        fy = yp / self.dy - iy

        ix1 = (ix + 1) % self.nx
        iy1 = (iy + 1) % self.ny

        # CIC weights
        for i in range(self.n_particles):
            w00 = (1 - fx[i]) * (1 - fy[i])
            w10 = fx[i] * (1 - fy[i])
            w01 = (1 - fx[i]) * fy[i]
            w11 = fx[i] * fy[i]

            self.rho_dm_grid[iy[i], ix[i]] += self.m_dm * w00
            self.rho_dm_grid[iy[i], ix1[i]] += self.m_dm * w10
            self.rho_dm_grid[iy1[i], ix[i]] += self.m_dm * w01
            self.rho_dm_grid[iy1[i], ix1[i]] += self.m_dm * w11

        self.rho_dm_grid /= (self.dx * self.dy)

    # Poisson solver

    def _solve_poisson(self):
        """Solve comoving Poisson: nabla^2 Phi = 4*pi*G * a * (rho - rho_bar)."""
        rho_bar = np.mean(self.rho_total)
        delta_rho = self.rho_total - rho_bar
        rhs = 4.0 * np.pi * self.a * delta_rho
        rhs_hat = np.fft.fft2(rhs)
        phi_hat = rhs_hat / self.poisson_eig
        phi_hat[0, 0] = 0.0
        self.Phi = np.real(np.fft.ifft2(phi_hat))

    # DM particle kick-drift

    def _interpolate_force(self, x, y):
        """Interpolate gravitational force at particle positions (CIC)."""
        gx = -self._ddx(self.Phi)
        gy = -self._ddy(self.Phi)

        xp = x % self.Lbox
        yp = y % self.Lbox
        ix = (xp / self.dx).astype(int) % self.nx
        iy = (yp / self.dy).astype(int) % self.ny
        fx = xp / self.dx - ix
        fy = yp / self.dy - iy
        ix1 = (ix + 1) % self.nx
        iy1 = (iy + 1) % self.ny

        ax_p = (
            gx[iy, ix] * (1-fx)*(1-fy) + gx[iy, ix1] * fx*(1-fy)
            + gx[iy1, ix] * (1-fx)*fy + gx[iy1, ix1] * fx*fy
        )
        ay_p = (
            gy[iy, ix] * (1-fx)*(1-fy) + gy[iy, ix1] * fx*(1-fy)
            + gy[iy1, ix] * (1-fx)*fy + gy[iy1, ix1] * fx*fy
        )
        return ax_p, ay_p

    def _advance_dm_particles(self):
        """Leapfrog kick-drift for DM particles in comoving coords."""
        H = self.hubble()
        ax_p, ay_p = self._interpolate_force(self.x_dm, self.y_dm)

        # Kick (comoving peculiar velocity)
        self.vx_dm += self.dt * (ax_p / self.a - H * self.vx_dm)
        self.vy_dm += self.dt * (ay_p / self.a - H * self.vy_dm)

        # Clamp
        v_max = 5.0 * self.Lbox
        self.vx_dm = np.clip(self.vx_dm, -v_max, v_max)
        self.vy_dm = np.clip(self.vy_dm, -v_max, v_max)

        # Drift
        self.x_dm += self.dt * self.vx_dm / self.a
        self.y_dm += self.dt * self.vy_dm / self.a

        # Periodic wrap
        self.x_dm %= self.Lbox
        self.y_dm %= self.Lbox

    # Baryon fluid step

    def _advance_baryons(self):
        """Euler equations on expanding background."""
        H = self.hubble()

        # Gravitational + pressure acceleration
        gx = -self._ddx(self.Phi) / self.a
        gy = -self._ddy(self.Phi) / self.a

        cs2 = np.maximum(self.temp, 1e-8)
        dpdx = cs2 * self._ddx(self.rho_b) / (self.rho_b + 1e-12)
        dpdy = cs2 * self._ddy(self.rho_b) / (self.rho_b + 1e-12)

        # Advection
        adv_vx = self.vx_b * self._ddx(self.vx_b) + self.vy_b * self._ddy(self.vx_b)
        adv_vy = self.vx_b * self._ddx(self.vy_b) + self.vy_b * self._ddy(self.vy_b)

        # Viscous diffusion
        nu = 0.01
        diff_vx = nu * self._laplacian(self.vx_b)
        diff_vy = nu * self._laplacian(self.vy_b)

        # Update velocity (Hubble drag included)
        self.vx_b += self.dt * (-adv_vx - dpdx + gx - H * self.vx_b + diff_vx)
        self.vy_b += self.dt * (-adv_vy - dpdy + gy - H * self.vy_b + diff_vy)

        v_max = 5.0 * self.Lbox
        self.vx_b = np.clip(self.vx_b, -v_max, v_max)
        self.vy_b = np.clip(self.vy_b, -v_max, v_max)

        # Continuity
        div_rv = self._ddx(self.rho_b * self.vx_b) + self._ddy(self.rho_b * self.vy_b)
        self.rho_b -= self.dt * (div_rv / self.a + 2 * H * self.rho_b)
        self.rho_b = np.clip(self.rho_b, 1e-10, 1e6)

        # Density diffusion for stability
        self.rho_b += 0.002 * self.dt * self._laplacian(self.rho_b)
        self.rho_b = np.clip(self.rho_b, 1e-10, 1e6)

        # Sanitize
        self.vx_b = np.nan_to_num(self.vx_b, nan=0.0, posinf=v_max, neginf=-v_max)
        self.vy_b = np.nan_to_num(self.vy_b, nan=0.0, posinf=v_max, neginf=-v_max)
        self.rho_b = np.nan_to_num(self.rho_b, nan=1e-10, posinf=1e6, neginf=1e-10)

    # Initial conditions

    def initialize_cosmic_web(self, P_k_slope: float = -1.0, amplitude: float = 0.05):
        """
        Harrison-Zeldovich-like primordial power spectrum.

        P(k) ~ k^n_s with n_s ~ 1 (scale-invariant).
        Seeds cosmic web formation: filaments, voids, halos.
        """
        np.random.seed(42)
        kx = np.fft.fftfreq(self.nx, d=self.dx)
        ky = np.fft.fftfreq(self.ny, d=self.dy)[:, None]
        k_mag = np.sqrt(kx**2 + ky**2 + 1e-10)

        # Power spectrum P(k) ~ k^n
        P_k = k_mag**P_k_slope
        P_k[0, 0] = 0.0

        # Random Gaussian field with this spectrum
        phases = np.random.uniform(0, 2*np.pi, (self.ny, self.nx))
        delta_hat = np.sqrt(P_k) * np.exp(1j * phases)
        delta = np.real(np.fft.ifft2(delta_hat))
        delta *= amplitude / (np.std(delta) + 1e-12)

        # Baryonic density
        self.rho_b = self.Omega_b * (1.0 + delta)
        self.rho_b = np.maximum(self.rho_b, 1e-10)

        # DM particles displaced by Zeldovich approximation
        psi_hat = -1j * delta_hat / (k_mag**2 + 1e-10)
        psi_hat[0, 0] = 0.0

        psi_x = np.real(np.fft.ifft2(1j * kx * psi_hat))
        psi_y = np.real(np.fft.ifft2(1j * ky * psi_hat))

        # Uniform grid of DM particles + Zeldovich displacement
        n_side = int(np.sqrt(self.n_particles))
        xg = np.linspace(0, self.Lbox, n_side, endpoint=False)
        yg = np.linspace(0, self.Lbox, n_side, endpoint=False)
        XG, YG = np.meshgrid(xg, yg)

        actual_n = n_side * n_side
        self.n_particles = actual_n
        self.m_dm = self.Omega_dm * self.Lbox**2 / self.n_particles

        # Interpolate displacement to particle positions
        ix = (XG.ravel() / self.dx).astype(int) % self.nx
        iy = (YG.ravel() / self.dy).astype(int) % self.ny

        self.x_dm = (XG.ravel() + amplitude * 10 * psi_x[iy, ix]) % self.Lbox
        self.y_dm = (YG.ravel() + amplitude * 10 * psi_y[iy, ix]) % self.Lbox
        self.vx_dm = amplitude * 10 * psi_x[iy, ix] * self.a_dot
        self.vy_dm = amplitude * 10 * psi_y[iy, ix] * self.a_dot

    def initialize_inflation(self, phi0: float = 3.0, dphi0: float = -0.1):
        """Inflation-like scenario: scalar field drives exponential expansion."""
        self.rho_b = self.Omega_b * np.ones((self.ny, self.nx))
        # Tiny quantum fluctuations
        delta = 0.001 * np.random.randn(self.ny, self.nx)
        self.rho_b *= (1.0 + delta)
        self.rho_b = np.maximum(self.rho_b, 1e-10)

        # Scalar field energy density as extra cosmological term
        self.inflaton_phi = phi0
        self.inflaton_dphi = dphi0
        self.inflaton_V0 = 0.5  # potential energy scale

        n_side = int(np.sqrt(self.n_particles))
        xg = np.linspace(0, self.Lbox, n_side, endpoint=False)
        yg = np.linspace(0, self.Lbox, n_side, endpoint=False)
        XG, YG = np.meshgrid(xg, yg)
        self.n_particles = n_side * n_side
        self.m_dm = self.Omega_dm * self.Lbox**2 / self.n_particles
        self.x_dm = XG.ravel() + 0.001 * np.random.randn(self.n_particles) * self.Lbox
        self.y_dm = YG.ravel() + 0.001 * np.random.randn(self.n_particles) * self.Lbox
        self.x_dm %= self.Lbox
        self.y_dm %= self.Lbox
        self.vx_dm = np.zeros(self.n_particles)
        self.vy_dm = np.zeros(self.n_particles)

    # Time stepping

    def step(self):
        """Advance coupled cosmological system by one time step."""
        # 1. Advance scale factor (Friedmann)
        self._advance_scale_factor()

        # 2. Deposit DM particles onto grid
        self._deposit_dm()

        # 3. Total density
        self.rho_total = self.rho_dm_grid + self.rho_b

        # 4. Solve Poisson for gravitational potential
        self._solve_poisson()

        # 5. Advance DM particles
        self._advance_dm_particles()

        # 6. Advance baryonic fluid
        self._advance_baryons()

        self.time += self.dt
        self.step_count += 1

    def advance(self, n_steps: int = 1, record: bool = True):
        for _ in range(n_steps):
            self.step()
            if record and self.step_count % max(1, n_steps // 200) == 0:
                self._record_diagnostics()

    # Power spectrum

    def compute_power_spectrum(self, field: Optional[np.ndarray] = None):
        """Compute 1D power spectrum P(k) of density contrast."""
        if field is None:
            field = self.rho_total
        mean_f = np.mean(field)
        delta = (field - mean_f) / (mean_f + 1e-12)
        delta_hat = np.fft.fft2(delta)
        P2d = np.abs(delta_hat)**2 / (self.nx * self.ny)

        kx = np.fft.fftfreq(self.nx, d=self.dx)
        ky = np.fft.fftfreq(self.ny, d=self.dy)[:, None]
        k_mag = np.sqrt(kx**2 + ky**2)

        k_bins = np.linspace(0, np.max(k_mag) * 0.5, 40)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        P_k = np.zeros(len(k_centers))

        for i in range(len(k_centers)):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                P_k[i] = np.mean(P2d[mask])

        valid = P_k > 0
        return k_centers[valid], P_k[valid]

    # Diagnostics

    def _record_diagnostics(self):
        dV = self.dx * self.dy

        def safe(x):
            return x if np.isfinite(x) else 0.0

        rho_bar = np.mean(self.rho_total)
        delta = (self.rho_total - rho_bar) / (rho_bar + 1e-12)
        rms_delta = float(np.sqrt(np.mean(delta**2)))

        v2_b = self.vx_b**2 + self.vy_b**2
        KE = float(0.5 * np.nansum(self.rho_b * v2_b) * dV)
        PE = float(0.5 * np.nansum(self.rho_total * self.Phi) * dV)

        dm_rho_bar = np.mean(self.rho_dm_grid) + 1e-12
        dm_clump = float(np.mean(self.rho_dm_grid**2) / dm_rho_bar**2)

        b_rho_bar = np.mean(self.rho_b) + 1e-12
        b_clump = float(np.mean(self.rho_b**2) / b_rho_bar**2)

        self.history['time'].append(self.time)
        self.history['scale_factor'].append(self.a)
        self.history['redshift'].append(safe(self.redshift()))
        self.history['hubble'].append(safe(self.hubble()))
        self.history['total_mass'].append(safe(float(np.nansum(self.rho_total) * dV)))
        self.history['max_density'].append(safe(float(np.max(self.rho_total))))
        self.history['rms_density_contrast'].append(safe(rms_delta))
        self.history['kinetic_energy'].append(safe(KE))
        self.history['potential_energy'].append(safe(PE))
        self.history['dm_clumping'].append(safe(dm_clump))
        self.history['baryon_clumping'].append(safe(b_clump))

    def get_state(self) -> Dict[str, np.ndarray]:
        self._deposit_dm()
        self.rho_total = self.rho_dm_grid + self.rho_b

        rho_bar = np.mean(self.rho_total) + 1e-12
        delta = (self.rho_total - rho_bar) / rho_bar

        return {
            'rho_total': self.rho_total.copy(),
            'rho_dm': self.rho_dm_grid.copy(),
            'rho_baryon': self.rho_b.copy(),
            'density_contrast': delta,
            'Phi': self.Phi.copy(),
            'vx_b': self.vx_b.copy(), 'vy_b': self.vy_b.copy(),
            'temp': self.temp.copy(),
            'velocity_magnitude': np.sqrt(self.vx_b**2 + self.vy_b**2),
            'x_dm': self.x_dm.copy(), 'y_dm': self.y_dm.copy(),
            'scale_factor': self.a,
            'redshift': self.redshift(),
            'hubble': self.hubble(),
            'time': self.time,
        }
