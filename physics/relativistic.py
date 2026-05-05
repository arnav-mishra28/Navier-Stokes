"""
=============================================================================
Relativistic Navier-Stokes Solver (Israel-Stewart Formulation)

Naive relativistic Navier-Stokes is acausal and unstable.
This module implements the Israel-Stewart (IS) second-order theory,
which restores causality by promoting dissipative fluxes to
dynamical variables with finite relaxation times.

Governing Equations:
    d_mu T^{mu nu} = 0          (energy-momentum conservation)

    T^{mu nu} = (e + p) u^mu u^nu + p g^{mu nu} + pi^{mu nu}

    tau_pi * D(pi^{mu nu}) + pi^{mu nu} = 2 eta sigma^{mu nu}
        - tau_pi * pi^{mu nu} * theta / 3  (IS relaxation)

Where:
    u^mu    = Lorentz factor * (1, v)   (4-velocity)
    e       = energy density
    p       = pressure (EOS: p = e/3 for ultrarelativistic)
    pi^{mu nu} = viscous stress tensor (dissipative correction)
    eta     = shear viscosity
    tau_pi  = relaxation time (causality parameter)
    sigma^{mu nu} = velocity shear tensor
    theta   = expansion scalar (d_mu u^mu)
    D       = u^mu d_mu  (comoving derivative)

Applications:
    - Quark-gluon plasma (RHIC/LHC heavy-ion collisions)
    - Neutron star mergers
    - Relativistic astrophysical jets
    - Gamma-ray burst afterglows

References:
    Israel & Stewart, Ann. Phys. 118, 341 (1979)
    Romatschke, Int. J. Mod. Phys. E19, 1 (2010)
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class RelativisticNSSolver:
    """
    2+1D Relativistic Navier-Stokes solver with Israel-Stewart
    causal dissipation.

    Uses Minkowski metric g^{mu nu} = diag(-1, +1, +1) in 2+1D.
    Natural units: c = 1.

    The solver evolves conserved quantities:
        T^{00} = energy density in lab frame
        T^{0i} = momentum density
        pi^{ij} = viscous stress (IS dynamical variable)
    """

    # Speed of light (natural units)
    c = 1.0

    def __init__(
        self,
        nx: int = 128, ny: int = 128,
        Lx: float = 10.0, Ly: float = 10.0,
        eta_s: float = 0.2,       # eta/s (shear viscosity to entropy ratio)
        tau_pi: float = 0.5,      # Relaxation time (fm/c or code units)
        dt: float = 0.005,
        eos: str = "ultrarelativistic",  # "ultrarelativistic" or "ideal_gas"
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / nx, Ly / ny
        self.eta_s = eta_s
        self.tau_pi = max(tau_pi, dt * 2)  # Ensure causality
        self.dt = dt
        self.eos = eos

        # Primitive variables
        self.energy_density = np.ones((ny, nx)) * 1.0       # e (proper frame)
        self.vx = np.zeros((ny, nx))                        # v^x (3-velocity)
        self.vy = np.zeros((ny, nx))                        # v^y
        self.pressure = np.zeros((ny, nx))                  # p

        # Israel-Stewart viscous stress tensor components (symmetric traceless)
        # In 2+1D: pi^{xx}, pi^{xy}, pi^{yy} with pi^{xx} + pi^{yy} = 0 (traceless)
        self.pi_xx = np.zeros((ny, nx))
        self.pi_xy = np.zeros((ny, nx))
        self.pi_yy = np.zeros((ny, nx))

        # Bulk viscous pressure (simplified)
        self.Pi_bulk = np.zeros((ny, nx))

        # Entropy density
        self.entropy = np.ones((ny, nx))

        # Coordinate grids
        x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
        y = np.linspace(-Ly / 2, Ly / 2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)

        self.time = 0.0
        self.step_count = 0

        # Diagnostics
        self.history: Dict[str, List] = {
            'time': [], 'total_energy': [], 'max_lorentz': [],
            'max_speed': [], 'entropy_total': [], 'max_pi': [],
        }

        # Apply EOS to set initial pressure
        self._apply_eos()

    # ─── Equation of State ───────────────────────────────────────────

    def _apply_eos(self):
        """Apply equation of state: p = p(e)."""
        if self.eos == "ultrarelativistic":
            # Conformal EOS: p = e/3 (massless quarks/gluons, photon gas)
            self.pressure = self.energy_density / 3.0
        else:
            # Ideal gas with gamma = 4/3 (relativistic)
            gamma = 4.0 / 3.0
            self.pressure = (gamma - 1.0) * self.energy_density
        # Entropy density: s ~ (e + p) / T, with T ~ e^{1/4} for conformal
        e_safe = np.maximum(self.energy_density, 1e-12)
        T_eff = e_safe ** 0.25
        self.entropy = (self.energy_density + self.pressure) / (T_eff + 1e-12)

    def _speed_of_sound(self) -> np.ndarray:
        """Speed of sound: c_s^2 = dp/de."""
        if self.eos == "ultrarelativistic":
            return np.full_like(self.energy_density, 1.0 / 3.0)
        else:
            return np.full_like(self.energy_density, 1.0 / 3.0)

    # ─── Lorentz Factor ──────────────────────────────────────────────

    def lorentz_factor(self) -> np.ndarray:
        """Lorentz factor: gamma = 1 / sqrt(1 - v^2/c^2)."""
        v2 = self.vx**2 + self.vy**2
        v2 = np.minimum(v2, 0.9999 * self.c**2)  # Cap to prevent singularity
        return 1.0 / np.sqrt(1.0 - v2 / self.c**2)

    def four_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4-velocity: u^mu = gamma * (1, v^x, v^y)."""
        gamma = self.lorentz_factor()
        return gamma, gamma * self.vx, gamma * self.vy

    # ─── Finite Difference Operators ─────────────────────────────────

    def _ddx(self, f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * self.dx)

    def _ddy(self, f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * self.dy)

    def _laplacian(self, f):
        return (
            (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / self.dx**2
            + (np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / self.dy**2
        )

    # ─── Energy-Momentum Tensor ──────────────────────────────────────

    def compute_Tmunu(self) -> Dict[str, np.ndarray]:
        """
        Compute T^{mu nu} = (e+p) u^mu u^nu + p g^{mu nu} + pi^{mu nu}.

        Returns dict with components T00, T0x, T0y, Txx, Txy, Tyy.
        """
        gamma = self.lorentz_factor()
        u0, ux, uy = self.four_velocity()
        e = self.energy_density
        p = self.pressure

        w = e + p  # enthalpy density

        # Ideal part
        T00 = w * u0 * u0 - p  # T^{00} = w*gamma^2 - p
        T0x = w * u0 * ux      # T^{0x} = w*gamma^2*vx
        T0y = w * u0 * uy
        Txx = w * ux * ux + p + self.pi_xx
        Txy = w * ux * uy + self.pi_xy
        Tyy = w * uy * uy + p + self.pi_yy

        # Add viscous corrections to T^{0i} components
        T0x += self.pi_xx * self.vx + self.pi_xy * self.vy
        T0y += self.pi_xy * self.vx + self.pi_yy * self.vy

        return {
            'T00': T00, 'T0x': T0x, 'T0y': T0y,
            'Txx': Txx, 'Txy': Txy, 'Tyy': Tyy,
        }

    # ─── Velocity Shear Tensor ───────────────────────────────────────

    def _compute_shear_expansion(self):
        """
        Compute velocity shear tensor sigma^{ij} and expansion scalar theta.

        sigma^{ij} = 1/2 (d^i u^j + d^j u^i) - 1/2 Delta^{ij} theta
        theta = d_mu u^mu (expansion scalar)
        """
        gamma = self.lorentz_factor()
        dvx_dx = self._ddx(self.vx)
        dvx_dy = self._ddy(self.vx)
        dvy_dx = self._ddx(self.vy)
        dvy_dy = self._ddy(self.vy)

        dgamma_dt_approx = gamma**3 * (
            self.vx * (self.vx * dvx_dx + self.vy * dvx_dy)
            + self.vy * (self.vx * dvy_dx + self.vy * dvy_dy)
        )

        # Expansion scalar: theta = d_mu u^mu
        theta = dgamma_dt_approx + self._ddx(gamma * self.vx) + self._ddy(gamma * self.vy)

        # Shear tensor (spatial part, simplified for flat spacetime)
        sigma_xx = gamma * dvx_dx - theta / 2.0
        sigma_yy = gamma * dvy_dy - theta / 2.0
        sigma_xy = 0.5 * gamma * (dvx_dy + dvy_dx)

        return sigma_xx, sigma_xy, sigma_yy, theta

    # ─── Israel-Stewart Relaxation Step ──────────────────────────────

    def _israel_stewart_step(self):
        """
        Evolve viscous stress via Israel-Stewart relaxation equation:

        tau_pi * D(pi^{ij}) + pi^{ij} = 2*eta*sigma^{ij}
                                         - tau_pi * pi^{ij} * theta / 3

        This ensures:
        1. Causality (signal speed < c)
        2. Stability (no runaway growth)
        3. Correct Navier-Stokes limit when tau_pi -> 0
        """
        sigma_xx, sigma_xy, sigma_yy, theta = self._compute_shear_expansion()

        # Shear viscosity: eta = (eta/s) * s
        eta = self.eta_s * self.entropy

        # Navier-Stokes target values
        pi_NS_xx = 2.0 * eta * sigma_xx
        pi_NS_xy = 2.0 * eta * sigma_xy
        pi_NS_yy = 2.0 * eta * sigma_yy

        # Relaxation: tau_pi * d(pi)/dt + pi = pi_NS - tau_pi * pi * theta/3
        # => d(pi)/dt = (pi_NS - pi) / tau_pi - pi * theta / 3
        # Plus advection of pi by the flow
        adv_xx = self.vx * self._ddx(self.pi_xx) + self.vy * self._ddy(self.pi_xx)
        adv_xy = self.vx * self._ddx(self.pi_xy) + self.vy * self._ddy(self.pi_xy)
        adv_yy = self.vx * self._ddx(self.pi_yy) + self.vy * self._ddy(self.pi_yy)

        relax = 1.0 / self.tau_pi
        expansion_corr = theta / 3.0

        self.pi_xx += self.dt * (
            (pi_NS_xx - self.pi_xx) * relax - self.pi_xx * expansion_corr - adv_xx
        )
        self.pi_xy += self.dt * (
            (pi_NS_xy - self.pi_xy) * relax - self.pi_xy * expansion_corr - adv_xy
        )
        self.pi_yy += self.dt * (
            (pi_NS_yy - self.pi_yy) * relax - self.pi_yy * expansion_corr - adv_yy
        )

        # Enforce tracelessness: pi^{xx} + pi^{yy} = 0
        trace = 0.5 * (self.pi_xx + self.pi_yy)
        self.pi_xx -= trace
        self.pi_yy -= trace

        # Regulate viscous stress (prevent exceeding energy density)
        pi_mag = np.sqrt(self.pi_xx**2 + 2 * self.pi_xy**2 + self.pi_yy**2)
        e_safe = np.maximum(self.energy_density, 1e-10)
        excess = pi_mag / (e_safe + self.pressure + 1e-10)
        mask = excess > 1.0
        if np.any(mask):
            scale = 1.0 / (excess + 1e-10)
            self.pi_xx = np.where(mask, self.pi_xx * scale, self.pi_xx)
            self.pi_xy = np.where(mask, self.pi_xy * scale, self.pi_xy)
            self.pi_yy = np.where(mask, self.pi_yy * scale, self.pi_yy)

    # ─── Conservation Law Step ───────────────────────────────────────

    def step(self):
        """
        Advance by one time step using operator splitting:
            1. Compute T^{mu nu}
            2. Solve d_mu T^{mu nu} = 0 for conserved variables
            3. Israel-Stewart relaxation for pi^{mu nu}
            4. Recover primitive variables
        """
        Tmn = self.compute_Tmunu()

        # d_mu T^{mu 0} = 0 => dT00/dt = -dT0x/dx - dT0y/dy  (energy conservation)
        dE = -(self._ddx(Tmn['T0x']) + self._ddy(Tmn['T0y']))

        # d_mu T^{mu x} = 0 => dT0x/dt = -dTxx/dx - dTxy/dy  (x-momentum)
        dMx = -(self._ddx(Tmn['Txx']) + self._ddy(Tmn['Txy']))

        # d_mu T^{mu y} = 0 => dT0y/dt = -dTxy/dx - dTyy/dy  (y-momentum)
        dMy = -(self._ddx(Tmn['Txy']) + self._ddy(Tmn['Tyy']))

        # Update conserved quantities
        gamma = self.lorentz_factor()
        w = self.energy_density + self.pressure

        # Lab-frame energy density and momenta
        E_lab = Tmn['T00'] + self.dt * dE
        Mx = Tmn['T0x'] + self.dt * dMx
        My = Tmn['T0y'] + self.dt * dMy

        # Recover primitive variables (simplified inversion)
        E_lab = np.maximum(E_lab, 1e-10)
        M2 = Mx**2 + My**2

        # Iterative recovery (Newton-Raphson for e from conserved vars)
        e_guess = self.energy_density.copy()
        for _ in range(3):
            p_guess = e_guess / 3.0
            w_guess = e_guess + p_guess
            v2 = M2 / (w_guess * gamma + 1e-10)**2
            v2 = np.minimum(v2, 0.9999)
            gamma_new = 1.0 / np.sqrt(1.0 - v2)
            e_guess = (E_lab + p_guess) / gamma_new**2 - p_guess
            e_guess = np.maximum(e_guess, 1e-10)

        self.energy_density = e_guess
        self._apply_eos()

        w_new = self.energy_density + self.pressure
        gamma_rec = self.lorentz_factor()
        denom = w_new * gamma_rec**2 + 1e-10
        self.vx = Mx / denom
        self.vy = My / denom

        # Zero out velocities in vacuum regions (prevents spurious artifacts)
        e_threshold = 0.05 * np.max(self.energy_density)
        vacuum = self.energy_density < e_threshold
        self.vx[vacuum] *= 0.1
        self.vy[vacuum] *= 0.1

        # Cap velocities
        speed = np.sqrt(self.vx**2 + self.vy**2)
        max_speed = 0.9999 * self.c
        over = speed > max_speed
        if np.any(over):
            scale = max_speed / (speed + 1e-20)
            self.vx = np.where(over, self.vx * scale, self.vx)
            self.vy = np.where(over, self.vy * scale, self.vy)

        # Israel-Stewart viscous evolution
        self._israel_stewart_step()

        # Numerical diffusion for stability
        diff_coeff = 0.002
        self.energy_density += diff_coeff * self.dt * self._laplacian(self.energy_density)
        self.vx += diff_coeff * self.dt * self._laplacian(self.vx)
        self.vy += diff_coeff * self.dt * self._laplacian(self.vy)

        self.energy_density = np.maximum(self.energy_density, 1e-10)
        self._apply_eos()

        self.time += self.dt
        self.step_count += 1

    def advance(self, n_steps: int = 1, record: bool = True):
        """Advance simulation by n_steps."""
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()

    # ─── Initial Conditions ──────────────────────────────────────────

    def initialize_bjorken_flow(self, e0: float = 10.0, sigma: float = 1.0):
        """
        Bjorken flow analog in 2+1D — central fireball.

        Models quark-gluon plasma created in heavy-ion collisions:
        Gaussian energy deposition, zero initial velocity.
        """
        R2 = self.X**2 + self.Y**2
        self.energy_density = e0 * np.exp(-R2 / (2 * sigma**2)) + 0.01
        self.vx[:] = 0.0
        self.vy[:] = 0.0
        self.pi_xx[:] = 0.0
        self.pi_xy[:] = 0.0
        self.pi_yy[:] = 0.0
        self._apply_eos()

    def initialize_relativistic_jet(
        self, v_jet: float = 0.99, e_jet: float = 5.0,
        jet_width: float = 0.5, e_ambient: float = 0.1,
    ):
        """
        Relativistic jet injection.

        A collimated beam of ultrarelativistic fluid (v ~ c)
        propagating into a low-density ambient medium.
        Produces shocks, cocoons, and Kelvin-Helmholtz instabilities.
        """
        self.energy_density = np.full((self.ny, self.nx), e_ambient)
        self.vx[:] = 0.0
        self.vy[:] = 0.0

        # Jet: high-energy beam along x-axis
        jet_mask = (np.abs(self.Y) < jet_width) & (self.X < -self.Lx / 4)
        self.energy_density[jet_mask] = e_jet
        self.vx[jet_mask] = v_jet

        # Smooth edges
        edge = np.exp(-self.Y**2 / (2 * jet_width**2)) * (self.X < -self.Lx / 4).astype(float)
        blend = 0.3 * edge
        self.energy_density += blend * e_jet
        self.vx += blend * v_jet * 0.5

        self.pi_xx[:] = 0.0
        self.pi_xy[:] = 0.0
        self.pi_yy[:] = 0.0
        self._apply_eos()

    def initialize_neutron_star_merger(
        self, e_max: float = 20.0, separation: float = 2.0,
        v_orbit: float = 0.3,
    ):
        """
        Simplified neutron star merger configuration.

        Two dense cores in quasi-circular orbit, about to merge.
        Produces shear layers, tidal tails, and shock heating.
        """
        sigma = 0.8
        R1 = np.sqrt((self.X - separation / 2)**2 + self.Y**2)
        R2 = np.sqrt((self.X + separation / 2)**2 + self.Y**2)

        e1 = e_max * np.exp(-R1**2 / (2 * sigma**2))
        e2 = e_max * np.exp(-R2**2 / (2 * sigma**2))
        self.energy_density = e1 + e2 + 0.01

        # Orbital velocity (tangential)
        theta1 = np.arctan2(self.Y, self.X - separation / 2)
        theta2 = np.arctan2(self.Y, self.X + separation / 2)

        w1 = e1 / (e1 + e2 + 1e-10)
        w2 = e2 / (e1 + e2 + 1e-10)

        self.vx = w1 * (-v_orbit * np.sin(theta1)) + w2 * (v_orbit * np.sin(theta2))
        self.vy = w1 * (v_orbit * np.cos(theta1)) + w2 * (-v_orbit * np.cos(theta2))

        self.pi_xx[:] = 0.0
        self.pi_xy[:] = 0.0
        self.pi_yy[:] = 0.0
        self._apply_eos()

    def initialize_shock_tube(
        self, e_left: float = 10.0, e_right: float = 1.0,
    ):
        """
        Relativistic Riemann problem (shock tube).

        Sod-like problem extended to relativistic regime.
        Tests shock speed, contact discontinuity, rarefaction wave.
        """
        self.energy_density = np.where(self.X < 0, e_left, e_right)
        self.vx[:] = 0.0
        self.vy[:] = 0.0
        self.pi_xx[:] = 0.0
        self.pi_xy[:] = 0.0
        self.pi_yy[:] = 0.0
        self._apply_eos()

    # ─── Diagnostics ─────────────────────────────────────────────────

    def _record_diagnostics(self):
        gamma = self.lorentz_factor()
        speed = np.sqrt(self.vx**2 + self.vy**2)
        pi_mag = np.sqrt(self.pi_xx**2 + 2 * self.pi_xy**2 + self.pi_yy**2)

        self.history['time'].append(self.time)
        self.history['total_energy'].append(
            float(np.sum(self.energy_density * gamma**2) * self.dx * self.dy)
        )
        self.history['max_lorentz'].append(float(np.max(gamma)))
        self.history['max_speed'].append(float(np.max(speed)))
        self.history['entropy_total'].append(
            float(np.sum(self.entropy) * self.dx * self.dy)
        )
        self.history['max_pi'].append(float(np.max(pi_mag)))

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return full state for visualization."""
        gamma = self.lorentz_factor()
        speed = np.sqrt(self.vx**2 + self.vy**2)
        pi_mag = np.sqrt(self.pi_xx**2 + 2 * self.pi_xy**2 + self.pi_yy**2)
        Tmn = self.compute_Tmunu()

        return {
            'energy_density': self.energy_density.copy(),
            'u': self.vx.copy(), 'v': self.vy.copy(),
            'pressure': self.pressure.copy(),
            'lorentz_factor': gamma,
            'velocity_magnitude': speed,
            'entropy': self.entropy.copy(),
            'pi_xx': self.pi_xx.copy(),
            'pi_xy': self.pi_xy.copy(),
            'pi_yy': self.pi_yy.copy(),
            'pi_magnitude': pi_mag,
            'T00': Tmn['T00'], 'T0x': Tmn['T0x'], 'T0y': Tmn['T0y'],
            'time': self.time,
        }
