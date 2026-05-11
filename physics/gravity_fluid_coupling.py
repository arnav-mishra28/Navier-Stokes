"""Gravity-Fluid Coupling Engine"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class GravityFluidSolver:
    """
    2+1D Gravity-Fluid Coupling Engine.

    Solves the coupled Einstein-Euler system at three approximation levels:
      - newtonian:       ∇²Φ = 4πGρ, fluid feels -∇Φ
      - post_newtonian:  1PN corrections (v²/c², Φ/c² terms)
      - numerical_gr:    Conformally-flat GR (BSSN-lite with lapse+shift)

    Natural units: G = c = 1 (for GR modes).
    """

    def __init__(
        self,
        nx: int = 128, ny: int = 128,
        Lx: float = 20.0, Ly: float = 20.0,
        dt: float = 0.005,
        gravity_level: str = "newtonian",
        gamma_eos: float = 5.0 / 3.0,
        G_const: float = 1.0,
        c_light: float = 1.0,
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.gravity_level = gravity_level
        self.gamma_eos = gamma_eos
        self.G = G_const
        self.c = c_light

        # Fluid primitives
        self.rho = np.ones((ny, nx)) * 0.01
        self.vx = np.zeros((ny, nx))
        self.vy = np.zeros((ny, nx))
        self.pressure = np.zeros((ny, nx))
        self.epsilon = np.zeros((ny, nx))  # specific internal energy

        # Gravitational fields
        self.Phi = np.zeros((ny, nx))          # Newtonian potential
        self.Phi_1PN = np.zeros((ny, nx))      # 1PN correction potential
        self.lapse = np.ones((ny, nx))         # α (GR lapse function)
        self.shift_x = np.zeros((ny, nx))      # β^x (GR shift vector)
        self.shift_y = np.zeros((ny, nx))      # β^y
        self.conformal_factor = np.ones((ny, nx))  # ψ (conformal factor)

        # Energy-momentum tensor components
        self.T00 = np.zeros((ny, nx))
        self.T0x = np.zeros((ny, nx))
        self.T0y = np.zeros((ny, nx))
        self.Txx = np.zeros((ny, nx))
        self.Txy = np.zeros((ny, nx))
        self.Tyy = np.zeros((ny, nx))

        # Coordinate grids
        x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
        y = np.linspace(-Ly / 2, Ly / 2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2 + 1e-6)

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
            'time': [], 'total_mass': [], 'kinetic_energy': [],
            'gravitational_energy': [], 'thermal_energy': [],
            'max_density': [], 'max_velocity': [],
            'virial_ratio': [], 'gw_strain': [],
        }

    # Finite Differences

    def _ddx(self, f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * self.dx)

    def _ddy(self, f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * self.dy)

    def _laplacian(self, f):
        return (
            (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / self.dx**2
            + (np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / self.dy**2
        )

    # Equation of State

    def _apply_eos(self):
        """Ideal gas EOS: p = (γ-1) ρ ε."""
        self.pressure = (self.gamma_eos - 1.0) * self.rho * self.epsilon
        self.pressure = np.maximum(self.pressure, 1e-12)

    def _sound_speed(self):
        """Sound speed: c_s² = γ p / ρ."""
        return np.sqrt(self.gamma_eos * self.pressure / (self.rho + 1e-12))

    # Energy-Momentum Tensor

    def compute_Tmunu(self):
        """
        Compute fluid energy-momentum tensor:
            T_μν = (ρ + ρε + p) u_μ u_ν + p g_μν

        In the Newtonian limit:
            T^00 ≈ ρ (rest mass energy density)
            T^0i ≈ ρ v^i (momentum density)
            T^ij ≈ ρ v^i v^j + p δ^ij (stress tensor)
        """
        rho_h = self.rho * (1.0 + self.epsilon) + self.pressure  # enthalpy
        rho_h = np.clip(rho_h, 1e-12, 1e6)
        v2 = np.clip(self.vx**2 + self.vy**2, 0, 1e6)

        if self.gravity_level == "newtonian":
            W = 1.0  # No Lorentz factor
        else:
            W2 = 1.0 / (1.0 - np.minimum(v2 / self.c**2, 0.99))
            W = np.sqrt(np.clip(W2, 1.0, 100.0))

        self.T00 = rho_h * W**2 - self.pressure
        self.T0x = rho_h * W**2 * self.vx
        self.T0y = rho_h * W**2 * self.vy
        self.Txx = rho_h * W**2 * self.vx**2 + self.pressure
        self.Txy = rho_h * W**2 * self.vx * self.vy
        self.Tyy = rho_h * W**2 * self.vy**2 + self.pressure

        # Sanitize
        for attr in ['T00', 'T0x', 'T0y', 'Txx', 'Txy', 'Tyy']:
            arr = getattr(self, attr)
            arr[:] = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    # Gravity Solvers

    def _solve_newtonian_gravity(self):
        """Solve ∇²Φ = 4πG ρ via FFT."""
        rhs = 4.0 * np.pi * self.G * self.rho
        rhs_hat = np.fft.fft2(rhs)
        phi_hat = rhs_hat / self.poisson_eig
        phi_hat[0, 0] = 0.0
        self.Phi = np.real(np.fft.ifft2(phi_hat))

    def _solve_post_newtonian_gravity(self):
        """
        1st Post-Newtonian correction.

        Φ_PN = Φ_N + (1/c²) [2Φ² + Ψ]
        where Ψ satisfies ∇²Ψ = 4πG (2v² ρ + 3p + ρΦ)
        """
        self._solve_newtonian_gravity()

        v2 = np.clip(self.vx**2 + self.vy**2, 0, 1e4)
        rhs_pn = 4.0 * np.pi * self.G * (
            2.0 * v2 * self.rho + 3.0 * self.pressure + self.rho * self.Phi
        )
        rhs_pn = np.nan_to_num(rhs_pn, nan=0.0, posinf=1e6, neginf=-1e6)
        rhs_hat = np.fft.fft2(rhs_pn)
        psi_hat = rhs_hat / self.poisson_eig
        psi_hat[0, 0] = 0.0
        Psi = np.real(np.fft.ifft2(psi_hat))

        self.Phi_1PN = np.clip((2.0 * self.Phi**2 + Psi) / self.c**2, -1e4, 1e4)
        self.Phi += self.Phi_1PN
        self.Phi = np.nan_to_num(self.Phi, nan=0.0, posinf=1e4, neginf=-1e4)

    def _solve_numerical_gr(self):
        """
        Conformally-flat GR approximation (CFC / BSSN-lite).

        Metric: ds² = -α² dt² + ψ⁴ (dx² + dy²)

        Hamiltonian constraint: ∇²ψ = -2πG ψ⁵ ρ_ADM
        Lapse equation:         ∇²(αψ) = 2πG αψ⁵ (ρ_ADM + 2S)
        Momentum constraint:    ∇²β^i = 16πG α ψ⁴ S^i
        """
        self.compute_Tmunu()

        rho_ADM = self.T00
        S = self.Txx + self.Tyy  # trace of spatial stress
        Sx = self.T0x
        Sy = self.T0y

        # Solve Hamiltonian constraint for conformal factor ψ
        # Linearized: ∇²ψ ≈ -2πG ρ_ADM (weak field)
        rho_ADM = np.clip(rho_ADM, 0, 1e6)
        S = np.clip(S, -1e6, 1e6)
        Sx = np.clip(Sx, -1e6, 1e6)
        Sy = np.clip(Sy, -1e6, 1e6)

        for _ in range(3):
            cf5 = np.clip(self.conformal_factor**5, 0, 1e4)
            rhs_psi = -2.0 * np.pi * self.G * cf5 * rho_ADM
            rhs_psi = np.nan_to_num(rhs_psi, nan=0.0, posinf=1e6, neginf=-1e6)
            rhs_hat = np.fft.fft2(rhs_psi)
            dpsi_hat = rhs_hat / self.poisson_eig
            dpsi_hat[0, 0] = 0.0
            dpsi = np.real(np.fft.ifft2(dpsi_hat))
            self.conformal_factor = np.clip(1.0 + 0.3 * dpsi, 0.5, 5.0)

        # Solve lapse equation
        cf5 = np.clip(self.conformal_factor**5, 0, 1e4)
        rhs_alpha = 2.0 * np.pi * self.G * self.lapse * cf5 * (rho_ADM + 2 * S)
        rhs_alpha = np.nan_to_num(rhs_alpha, nan=0.0, posinf=1e6, neginf=-1e6)
        rhs_hat = np.fft.fft2(rhs_alpha)
        dalpha_hat = rhs_hat / self.poisson_eig
        dalpha_hat[0, 0] = 0.0
        alpha_psi = np.real(np.fft.ifft2(dalpha_hat))
        self.lapse = np.clip(1.0 - 0.3 * np.abs(alpha_psi), 0.1, 2.0)

        # Solve momentum constraint for shift
        cf4 = np.clip(self.conformal_factor**4, 0, 1e3)
        for i, (Si, beta) in enumerate([(Sx, 'shift_x'), (Sy, 'shift_y')]):
            rhs_beta = 16.0 * np.pi * self.G * self.lapse * cf4 * Si
            rhs_beta = np.nan_to_num(rhs_beta, nan=0.0, posinf=1e6, neginf=-1e6)
            rhs_hat = np.fft.fft2(rhs_beta)
            dbeta_hat = rhs_hat / self.poisson_eig
            dbeta_hat[0, 0] = 0.0
            val = 0.1 * np.real(np.fft.ifft2(dbeta_hat))
            setattr(self, beta, np.clip(val, -1.0, 1.0))

        # Effective potential from metric
        self.Phi = np.clip((1.0 - self.lapse) * self.c**2, -1e4, 1e4)

    def _solve_gravity(self):
        """Dispatch to appropriate gravity solver."""
        if self.gravity_level == "newtonian":
            self._solve_newtonian_gravity()
        elif self.gravity_level == "post_newtonian":
            self._solve_post_newtonian_gravity()
        elif self.gravity_level == "numerical_gr":
            self._solve_numerical_gr()

    # Gravitational Wave Extraction

    def compute_gw_strain(self) -> float:
        """
        Estimate gravitational wave strain via quadrupole formula:
            h ~ (2G/c⁴) d²I_ij/dt²
        where I_ij = ∫ ρ x_i x_j dV is the mass quadrupole moment.
        """
        rho_safe = np.clip(self.rho, 0, 1e6)
        vx_safe = np.clip(self.vx, -1e3, 1e3)
        vy_safe = np.clip(self.vy, -1e3, 1e3)

        # Approximate d²I/dt² from momentum flux
        dIxx_dt2 = 2.0 * np.nansum(rho_safe * vx_safe**2) * self.dx * self.dy
        dIyy_dt2 = 2.0 * np.nansum(rho_safe * vy_safe**2) * self.dx * self.dy
        dIxy_dt2 = 2.0 * np.nansum(rho_safe * vx_safe * vy_safe) * self.dx * self.dy

        # Plus and cross polarizations
        h_plus = (dIxx_dt2 - dIyy_dt2) * self.G / max(self.c**4, 1e-12)
        h_cross = 2.0 * dIxy_dt2 * self.G / max(self.c**4, 1e-12)

        result = np.sqrt(h_plus**2 + h_cross**2)
        return float(result) if np.isfinite(result) else 0.0

    # Time Step

    def step(self):
        """
        Advance coupled gravity-fluid system by one time step.

        1. Solve gravity (Φ, lapse, shift, conformal factor)
        2. Compute T_μν from fluid state
        3. Compute gravitational + pressure accelerations
        4. Update fluid velocities and density
        5. Apply EOS
        """
        self._apply_eos()
        self._solve_gravity()
        self.compute_Tmunu()

        # Gravitational acceleration
        gx = -self._ddx(self.Phi)
        gy = -self._ddy(self.Phi)

        # GR corrections to acceleration
        if self.gravity_level == "numerical_gr":
            # Geodesic acceleration includes lapse gradient
            gx += -self.c**2 * self._ddx(np.log(self.lapse + 1e-12))
            gy += -self.c**2 * self._ddy(np.log(self.lapse + 1e-12))
            # Shift advection
            gx -= self.shift_x * self._ddx(self.vx) + self.shift_y * self._ddy(self.vx)
            gy -= self.shift_x * self._ddx(self.vy) + self.shift_y * self._ddy(self.vy)
        elif self.gravity_level == "post_newtonian":
            v2 = self.vx**2 + self.vy**2
            pn_factor = 1.0 + v2 / self.c**2 + 4.0 * np.abs(self.Phi) / self.c**2
            gx *= pn_factor
            gy *= pn_factor

        # Pressure gradient
        dpdx = self._ddx(self.pressure) / (self.rho + 1e-12)
        dpdy = self._ddy(self.pressure) / (self.rho + 1e-12)

        # Advection
        adv_vx = self.vx * self._ddx(self.vx) + self.vy * self._ddy(self.vx)
        adv_vy = self.vx * self._ddx(self.vy) + self.vy * self._ddy(self.vy)

        # Viscous diffusion (numerical stability)
        nu_eff = 0.005
        diff_vx = nu_eff * self._laplacian(self.vx)
        diff_vy = nu_eff * self._laplacian(self.vy)

        # Update velocities
        self.vx += self.dt * (-adv_vx - dpdx + gx + diff_vx)
        self.vy += self.dt * (-adv_vy - dpdy + gy + diff_vy)

        # Clamp velocities to prevent blowup
        v_max = 10.0 * self.c
        self.vx = np.clip(self.vx, -v_max, v_max)
        self.vy = np.clip(self.vy, -v_max, v_max)

        # Continuity equation: ∂ρ/∂t + ∇·(ρv) = 0
        div_rho_v = self._ddx(self.rho * self.vx) + self._ddy(self.rho * self.vy)
        self.rho -= self.dt * div_rho_v
        self.rho = np.clip(self.rho, 1e-10, 1e6)

        # Energy equation: ∂(ρε)/∂t = -p ∇·v + viscous heating
        div_v = self._ddx(self.vx) + self._ddy(self.vy)
        self.epsilon -= self.dt * self.pressure * div_v / (self.rho + 1e-12)
        self.epsilon = np.clip(self.epsilon, 1e-10, 1e6)

        # Numerical diffusion for stability
        self.rho += 0.002 * self.dt * self._laplacian(self.rho)
        self.rho = np.clip(self.rho, 1e-10, 1e6)

        # Sanitize all fields
        self.vx = np.nan_to_num(self.vx, nan=0.0, posinf=v_max, neginf=-v_max)
        self.vy = np.nan_to_num(self.vy, nan=0.0, posinf=v_max, neginf=-v_max)
        self.rho = np.nan_to_num(self.rho, nan=1e-10, posinf=1e6, neginf=1e-10)
        self.epsilon = np.nan_to_num(self.epsilon, nan=1e-10, posinf=1e6, neginf=1e-10)

        self._apply_eos()
        self.time += self.dt
        self.step_count += 1

    def advance(self, n_steps: int = 1, record: bool = True):
        """Advance simulation by n_steps."""
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()

    # Initial Conditions

    def initialize_accretion_disk(
        self, M_bh: float = 10.0, rho0: float = 1.0,
        r_in: float = 1.0, r_out: float = 8.0, T0: float = 1.0,
    ):
        """
        Black hole accretion disk.

        Keplerian rotation: v_φ = sqrt(GM/r)
        Density: ρ ~ r^{-3/2} (thin disk)
        Includes pseudo-Newtonian Paczyński-Wiita potential:
            Φ_PW = -GM / (r - r_s)  where r_s = 2GM/c²
        """
        r_s = 2.0 * self.G * M_bh / self.c**2  # Schwarzschild radius
        r_safe = np.maximum(self.R, r_s + 0.5)

        # Paczyński-Wiita potential
        self.Phi = -self.G * M_bh / (r_safe - r_s)

        # Disk mask
        mask = (self.R > r_in) & (self.R < r_out)

        # Density profile
        self.rho = np.where(mask, rho0 * (self.R / r_in) ** (-1.5), 0.01)

        # Keplerian velocity
        v_kep = np.sqrt(self.G * M_bh * r_safe / (r_safe - r_s) ** 2)
        v_kep = np.minimum(v_kep, 0.8 * self.c)
        theta = np.arctan2(self.Y, self.X)

        self.vx = np.where(mask, -v_kep * np.sin(theta), 0)
        self.vy = np.where(mask, v_kep * np.cos(theta), 0)

        # Temperature & internal energy
        self.epsilon = np.where(mask, T0 * (self.R / r_in) ** (-1), 0.01)
        self._apply_eos()

    def initialize_neutron_star_merger(
        self, M_star: float = 5.0, separation: float = 4.0,
        v_orbit: float = 0.15, sigma: float = 1.0,
    ):
        """
        Binary neutron star pre-merger.

        Two compact stars in quasi-circular orbit.
        Produces tidal tails, shear heating, and gravitational waves.
        """
        R1 = np.sqrt((self.X - separation / 2) ** 2 + self.Y ** 2)
        R2 = np.sqrt((self.X + separation / 2) ** 2 + self.Y ** 2)

        rho1 = M_star * np.exp(-R1**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        rho2 = M_star * np.exp(-R2**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        self.rho = rho1 + rho2 + 0.001

        # Orbital velocities (tangential)
        theta1 = np.arctan2(self.Y, self.X - separation / 2)
        theta2 = np.arctan2(self.Y, self.X + separation / 2)

        w1 = rho1 / (self.rho + 1e-12)
        w2 = rho2 / (self.rho + 1e-12)

        self.vx = w1 * (-v_orbit * np.sin(theta1)) + w2 * (v_orbit * np.sin(theta2))
        self.vy = w1 * (v_orbit * np.cos(theta1)) + w2 * (-v_orbit * np.cos(theta2))

        self.epsilon = 0.1 * self.rho
        self._apply_eos()

    def initialize_galaxy_formation(
        self, n_clumps: int = 8, rho_bg: float = 0.05,
        perturbation: float = 0.3, seed: int = 42,
    ):
        """
        Proto-galactic cloud collapse and fragmentation.

        Jeans-unstable cloud with random density perturbations.
        Produces filaments, clumps, and hierarchical merging.
        """
        np.random.seed(seed)

        # Background + large-scale mode
        self.rho = rho_bg * np.ones((self.ny, self.nx))

        # Add random Gaussian clumps (proto-galactic seeds)
        for _ in range(n_clumps):
            cx = np.random.uniform(-self.Lx / 3, self.Lx / 3)
            cy = np.random.uniform(-self.Ly / 3, self.Ly / 3)
            mass = np.random.uniform(1.0, 5.0)
            sig = np.random.uniform(0.5, 1.5)
            Ri = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            self.rho += mass * np.exp(-Ri**2 / (2 * sig**2)) / (2 * np.pi * sig**2)

        # Small-scale turbulent perturbations
        for kk in range(1, 6):
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            amp = perturbation / kk
            self.rho += amp * rho_bg * np.cos(
                2 * np.pi * kk * self.X / self.Lx + phase_x
            ) * np.cos(2 * np.pi * kk * self.Y / self.Ly + phase_y)

        self.rho = np.maximum(self.rho, 1e-6)

        # Small random velocities (turbulence)
        self.vx = 0.02 * np.random.randn(self.ny, self.nx)
        self.vy = 0.02 * np.random.randn(self.ny, self.nx)

        # Add bulk rotation
        self.vx += -0.01 * self.Y / self.Ly
        self.vy += 0.01 * self.X / self.Lx

        self.epsilon = 0.05 * np.ones((self.ny, self.nx))
        self._apply_eos()

    def initialize_cosmological_expansion(
        self, H0: float = 0.1, rho_bg: float = 1.0, delta: float = 0.01,
    ):
        """
        Cosmological perturbation on expanding background.

        Hubble flow: v = H₀ r (expansion)
        Small density perturbation δρ/ρ → gravitational collapse.
        """
        self.rho = rho_bg * (1.0 + delta * np.cos(2 * np.pi * self.X / self.Lx)
                             * np.cos(2 * np.pi * self.Y / self.Ly))
        self.vx = H0 * self.X
        self.vy = H0 * self.Y
        self.epsilon = 0.01 * np.ones((self.ny, self.nx))
        self._apply_eos()

    # Diagnostics

    def _record_diagnostics(self):
        dV = self.dx * self.dy
        v2 = np.clip(self.vx**2 + self.vy**2, 0, 1e8)
        rho_safe = np.clip(self.rho, 0, 1e6)
        mass = float(np.nansum(rho_safe) * dV)
        KE = float(0.5 * np.nansum(rho_safe * v2) * dV)
        GE = float(0.5 * np.nansum(rho_safe * np.clip(self.Phi, -1e6, 1e6)) * dV)
        TE = float(np.nansum(rho_safe * np.clip(self.epsilon, 0, 1e6)) * dV)
        virial = abs(2 * KE / (abs(GE) + 1e-12))
        gw = self.compute_gw_strain()

        # Ensure all values are finite
        def safe(x):
            return x if np.isfinite(x) else 0.0

        self.history['time'].append(self.time)
        self.history['total_mass'].append(safe(mass))
        self.history['kinetic_energy'].append(safe(KE))
        self.history['gravitational_energy'].append(safe(GE))
        self.history['thermal_energy'].append(safe(TE))
        self.history['max_density'].append(safe(float(np.max(rho_safe))))
        self.history['max_velocity'].append(safe(float(np.max(np.sqrt(v2)))))
        self.history['virial_ratio'].append(safe(virial))
        self.history['gw_strain'].append(safe(gw))

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return full state for visualization."""
        self.compute_Tmunu()
        speed = np.sqrt(self.vx**2 + self.vy**2)
        omega = self._ddx(self.vy) - self._ddy(self.vx)

        state = {
            'rho': self.rho.copy(),
            'u': self.vx.copy(), 'v': self.vy.copy(),
            'pressure': self.pressure.copy(),
            'epsilon': self.epsilon.copy(),
            'velocity_magnitude': speed,
            'vorticity': omega,
            'Phi': self.Phi.copy(),
            'T00': self.T00.copy(),
            'T0x': self.T0x.copy(),
            'T0y': self.T0y.copy(),
            'lapse': self.lapse.copy(),
            'conformal_factor': self.conformal_factor.copy(),
            'time': self.time,
        }
        if self.gravity_level == "numerical_gr":
            state['shift_x'] = self.shift_x.copy()
            state['shift_y'] = self.shift_y.copy()
        return state
