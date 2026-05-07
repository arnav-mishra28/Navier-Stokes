"""
=============================================================================
Lattice Quantum Field Theory (QFT) Simulator

Simulates scalar field dynamics on a discretized spacetime lattice.

Governing Equation (Klein-Gordon + phi^4 interaction):

    □φ + m²φ + λφ³ = 0

    where □ = d'Alembertian = -∂²/∂t² + ∇²  (Minkowski signature -,+,+)

Expanded in 2+1D:

    ∂²φ/∂t² = ∇²φ - m²φ - λφ³

This models:
    - Vacuum fluctuations (quantum zero-point energy)
    - Scalar field interactions (φ⁴ theory)
    - Early universe inflation (slow-roll scalar field)
    - Higgs-like symmetry breaking (Mexican hat potential)
    - Kink/anti-kink solitons
    - Bubble nucleation (first-order phase transitions)

Numerical Method:
    - Leapfrog (Störmer-Verlet) time integration
    - 2nd-order finite differences on spatial lattice
    - Periodic boundary conditions
    - Symplectic integrator preserves phase-space volume

References:
    Peskin & Schroeder, "An Introduction to Quantum Field Theory"
    Montvay & Münster, "Quantum Fields on a Lattice"
    Rajaraman, "Solitons and Instantons"
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class LatticeQFTSolver:
    """
    2+1D Lattice Quantum Field Theory solver for real scalar field.

    Evolves the Klein-Gordon equation with φ⁴ self-interaction:

        ∂²φ/∂t² = ∇²φ - dV/dφ

    where V(φ) = ½m²φ² + ¼λφ⁴  (or Mexican hat variant).

    Uses leapfrog (Störmer-Verlet) integrator:
        φ(t+dt) = 2φ(t) - φ(t-dt) + dt² * [∇²φ - dV/dφ]

    This is a symplectic integrator that conserves energy to O(dt²).

    Applications:
        - Lattice field theory studies
        - Cosmological scalar field dynamics
        - Phase transitions and symmetry breaking
        - Topological defect formation (kinks, domain walls)
        - Vacuum decay and bubble nucleation
    """

    def __init__(
        self,
        nx: int = 128, ny: int = 128,
        Lx: float = 20.0, Ly: float = 20.0,
        mass: float = 1.0,
        lam: float = 0.1,
        dt: float = 0.01,
        potential_type: str = "standard",
    ):
        """
        Args:
            nx, ny: Lattice sites in x, y directions
            Lx, Ly: Physical domain size
            mass: Field mass parameter m
            lam: Quartic coupling constant λ
            dt: Time step
            potential_type: "standard" (V = ½m²φ² + ¼λφ⁴)
                          or "mexican_hat" (V = λ(φ²-v²)²/4, Higgs-like)
        """
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.mass = mass
        self.lam = lam
        self.dt = dt
        self.potential_type = potential_type

        # Vacuum expectation value for Mexican hat
        if potential_type == "mexican_hat":
            self.v_vev = mass / np.sqrt(lam) if lam > 0 else 1.0
        else:
            self.v_vev = 0.0

        # Field and conjugate momentum (velocity)
        self.phi = np.zeros((ny, nx))        # φ(x, t)
        self.phi_prev = np.zeros((ny, nx))   # φ(x, t-dt)  for leapfrog
        self.pi_field = np.zeros((ny, nx))   # π = ∂φ/∂t   (conjugate momentum)

        # Coordinate grids
        x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
        y = np.linspace(-Ly / 2, Ly / 2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)

        # Momentum-space grids (for spectrum analysis)
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2

        self.time = 0.0
        self.step_count = 0

        # Diagnostics history
        self.history: Dict[str, List] = {
            'time': [], 'total_energy': [], 'kinetic_energy': [],
            'gradient_energy': [], 'potential_energy': [],
            'field_mean': [], 'field_rms': [], 'field_max': [],
        }

    # ─── Potential and Force ─────────────────────────────────────────

    def potential(self, phi: np.ndarray) -> np.ndarray:
        """Compute V(φ) at each lattice site."""
        if self.potential_type == "mexican_hat":
            # V = λ/4 * (φ² - v²)²
            return 0.25 * self.lam * (phi**2 - self.v_vev**2)**2
        else:
            # V = ½m²φ² + ¼λφ⁴
            return 0.5 * self.mass**2 * phi**2 + 0.25 * self.lam * phi**4

    def dVdphi(self, phi: np.ndarray) -> np.ndarray:
        """Compute dV/dφ (force term in EOM)."""
        if self.potential_type == "mexican_hat":
            # dV/dφ = λφ(φ² - v²)
            return self.lam * phi * (phi**2 - self.v_vev**2)
        else:
            # dV/dφ = m²φ + λφ³
            return self.mass**2 * phi + self.lam * phi**3

    # ─── Spatial Operators ───────────────────────────────────────────

    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        """Discrete Laplacian with periodic BC."""
        return (
            (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / self.dx**2
            + (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / self.dy**2
        )

    def _gradient_sq(self, f: np.ndarray) -> np.ndarray:
        """Compute |∇φ|² for energy computation."""
        dfdx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * self.dx)
        dfdy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * self.dy)
        return dfdx**2 + dfdy**2

    # ─── Initial Conditions ──────────────────────────────────────────

    def initialize_vacuum_fluctuations(self, amplitude: float = 0.01, seed: int = 42):
        """
        Initialize with quantum vacuum fluctuations.

        In QFT, even the vacuum has zero-point energy.
        Each mode k has energy ½ω_k where ω_k = √(k² + m²).

        We sample φ_k ~ N(0, 1/(2ω_k)) for each mode,
        producing Gaussian random field with correct power spectrum.
        """
        rng = np.random.RandomState(seed)

        # Dispersion relation: ω_k = √(k² + m²)
        omega_k = np.sqrt(self.K2 + self.mass**2)
        omega_k[0, 0] = 1.0  # Avoid division by zero

        # Amplitude per mode: <|φ_k|²> = 1/(2ω_k)
        amp_k = amplitude / np.sqrt(2 * omega_k)

        # Random phases and amplitudes
        phi_k = amp_k * (rng.randn(self.ny, self.nx) + 1j * rng.randn(self.ny, self.nx))
        phi_k[0, 0] = 0  # Zero mean

        self.phi = np.real(np.fft.ifft2(phi_k * self.nx * self.ny))
        self.phi_prev = self.phi.copy()
        self.pi_field = np.zeros_like(self.phi)

    def initialize_kink(self, x0: float = 0.0, width: float = None):
        """
        Initialize a kink soliton (topological defect).

        For Mexican hat potential, the kink connects two vacua:
            φ(x) = v * tanh((x - x0) / (√2 * width))

        For standard potential with m²<0 (tachyonic), similar profile.
        """
        if width is None:
            width = 2.0 / max(self.mass, 0.1)

        if self.potential_type == "mexican_hat":
            self.phi = self.v_vev * np.tanh((self.X - x0) / (np.sqrt(2) * width))
        else:
            self.phi = np.tanh((self.X - x0) / width)

        self.phi_prev = self.phi.copy()
        self.pi_field = np.zeros_like(self.phi)

    def initialize_bubble_nucleation(
        self, R: float = 3.0, phi_true: float = None, phi_false: float = None
    ):
        """
        Initialize bubble nucleation (first-order phase transition).

        A spherical bubble of true vacuum expands into false vacuum.
        Models: early universe phase transitions, Higgs decay.

        φ = φ_true inside bubble, φ_false outside, smooth wall.
        """
        if self.potential_type == "mexican_hat":
            phi_true = phi_true or self.v_vev
            phi_false = phi_false or -self.v_vev
        else:
            phi_true = phi_true or 0.0
            phi_false = phi_false or 1.0

        r = np.sqrt(self.X**2 + self.Y**2)
        wall_width = max(1.0 / self.mass, self.dx * 2)

        # Smooth bubble wall
        profile = 0.5 * (1 + np.tanh((r - R) / wall_width))
        self.phi = phi_true * (1 - profile) + phi_false * profile

        self.phi_prev = self.phi.copy()
        self.pi_field = np.zeros_like(self.phi)

    def initialize_inflation(self, phi0: float = 3.0, noise: float = 0.01, seed: int = 42):
        """
        Initialize slow-roll inflation scenario.

        Start with large field value (slow-roll region).
        Field rolls down potential, loses energy to expansion.
        Small perturbations seed structure formation.
        """
        rng = np.random.RandomState(seed)
        self.phi = phi0 + noise * rng.randn(self.ny, self.nx)
        self.phi_prev = self.phi.copy()
        self.pi_field = np.zeros_like(self.phi)

    def initialize_higgs_quench(self, noise: float = 0.1, seed: int = 42):
        """
        Higgs-like symmetry breaking via thermal quench.

        Start near φ=0 (symmetric phase) with small fluctuations.
        Mexican hat potential drives spontaneous symmetry breaking.
        Domains form with φ → ±v, separated by domain walls.
        """
        rng = np.random.RandomState(seed)
        self.phi = noise * rng.randn(self.ny, self.nx)
        self.phi_prev = self.phi.copy()
        self.pi_field = np.zeros_like(self.phi)

    def initialize_colliding_waves(self, k: float = 2.0, amplitude: float = 0.5):
        """
        Two counter-propagating plane waves.
        Tests nonlinear wave interactions in φ⁴ theory.
        """
        wave1 = amplitude * np.cos(k * self.X)
        wave2 = amplitude * np.cos(k * self.Y)
        self.phi = wave1 + wave2

        # Give initial momenta for propagation
        omega = np.sqrt(k**2 + self.mass**2)
        self.pi_field = -amplitude * omega * np.sin(k * self.X)
        self.phi_prev = self.phi - self.dt * self.pi_field

    # ─── Time Evolution ──────────────────────────────────────────────

    def step(self):
        """
        Advance field by one time step using leapfrog (Störmer-Verlet).

        Equation of motion:
            ∂²φ/∂t² = ∇²φ - dV/dφ

        Leapfrog update:
            φ(t+dt) = 2φ(t) - φ(t-dt) + dt² * acceleration

        This is a symplectic integrator:
            - Time-reversible
            - Conserves phase-space volume (Liouville theorem)
            - Energy conservation to O(dt²)
        """
        # Acceleration: ∂²φ/∂t² = ∇²φ - dV/dφ
        accel = self._laplacian(self.phi) - self.dVdphi(self.phi)

        # Leapfrog update
        phi_new = 2 * self.phi - self.phi_prev + self.dt**2 * accel

        # Update conjugate momentum (for diagnostics)
        self.pi_field = (phi_new - self.phi_prev) / (2 * self.dt)

        # Shift time levels
        self.phi_prev = self.phi.copy()
        self.phi = phi_new

        self.time += self.dt
        self.step_count += 1

    def advance(self, n_steps: int = 1, record: bool = True):
        """Advance simulation by n_steps."""
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()

    # ─── Energy Computation ──────────────────────────────────────────

    def compute_energy_density(self) -> Dict[str, np.ndarray]:
        """
        Compute energy density components:

            T^{00} = ½(∂φ/∂t)² + ½|∇φ|² + V(φ)

        Returns dict with kinetic, gradient, potential, and total.
        """
        # Kinetic: ½π² = ½(∂φ/∂t)²
        E_kin = 0.5 * self.pi_field**2

        # Gradient: ½|∇φ|²
        E_grad = 0.5 * self._gradient_sq(self.phi)

        # Potential: V(φ)
        E_pot = self.potential(self.phi)

        return {
            'kinetic': E_kin,
            'gradient': E_grad,
            'potential': E_pot,
            'total': E_kin + E_grad + E_pot,
        }

    def compute_total_energy(self) -> Dict[str, float]:
        """Integrate energy over the lattice."""
        E = self.compute_energy_density()
        dA = self.dx * self.dy
        return {
            'kinetic': float(np.sum(E['kinetic']) * dA),
            'gradient': float(np.sum(E['gradient']) * dA),
            'potential': float(np.sum(E['potential']) * dA),
            'total': float(np.sum(E['total']) * dA),
        }

    # ─── Field Spectrum ──────────────────────────────────────────────

    def compute_field_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum |φ_k|² as function of |k|.

        In QFT:
            - Free field: P(k) ~ 1/(2ω_k) (vacuum fluctuations)
            - Interactions modify the spectrum
            - UV divergence regularized by the lattice cutoff
        """
        phi_k = np.fft.fft2(self.phi) / (self.nx * self.ny)
        power = np.abs(phi_k)**2

        # Radial binning
        k_mag = np.sqrt(self.K2)
        k_max = np.max(k_mag) / 2
        n_bins = min(self.nx, self.ny) // 4
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        spectrum = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            if np.any(mask):
                spectrum[i] = np.sum(power[mask])

        return k_centers, spectrum

    def compute_correlation_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute two-point correlation function ⟨φ(0)φ(r)⟩.

        For a free massive scalar field:
            G(r) ~ exp(-m*r) / r^{(d-2)/2}   (Yukawa potential in 2D)

        The correlation length ξ = 1/m characterizes the range
        of field correlations (inverse mass gap).
        """
        phi_k = np.fft.fft2(self.phi)
        power = np.abs(phi_k)**2
        corr_2d = np.real(np.fft.ifft2(power)) / (self.nx * self.ny)

        # Radial average
        r_grid = np.sqrt(self.X**2 + self.Y**2)
        r_max = min(self.Lx, self.Ly) / 3
        n_bins = min(self.nx, self.ny) // 4
        r_bins = np.linspace(0, r_max, n_bins + 1)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

        # Shift correlation to be centered
        corr_shifted = np.fft.fftshift(corr_2d)
        cx, cy = self.nx // 2, self.ny // 2
        x_c = np.arange(self.nx) - cx
        y_c = np.arange(self.ny) - cy
        XC, YC = np.meshgrid(x_c * self.dx, y_c * self.dy)
        RC = np.sqrt(XC**2 + YC**2)

        corr_r = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (RC >= r_bins[i]) & (RC < r_bins[i + 1])
            if np.any(mask):
                corr_r[i] = np.mean(corr_shifted[mask])

        # Normalize
        if corr_r[0] > 0:
            corr_r /= corr_r[0]

        return r_centers, corr_r

    # ─── Topological Charge ──────────────────────────────────────────

    def compute_domain_walls(self) -> np.ndarray:
        """
        Detect domain walls (for Mexican hat potential).

        Domain walls form at boundaries between φ ≈ +v and φ ≈ -v regions.
        Identified by |∇φ| being large.
        """
        grad_sq = self._gradient_sq(self.phi)
        return np.sqrt(grad_sq)

    # ─── Diagnostics ─────────────────────────────────────────────────

    def _record_diagnostics(self):
        E = self.compute_total_energy()
        self.history['time'].append(self.time)
        self.history['total_energy'].append(E['total'])
        self.history['kinetic_energy'].append(E['kinetic'])
        self.history['gradient_energy'].append(E['gradient'])
        self.history['potential_energy'].append(E['potential'])
        self.history['field_mean'].append(float(np.mean(self.phi)))
        self.history['field_rms'].append(float(np.sqrt(np.mean(self.phi**2))))
        self.history['field_max'].append(float(np.max(np.abs(self.phi))))

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return full state for visualization."""
        E = self.compute_energy_density()
        grad_mag = np.sqrt(self._gradient_sq(self.phi))

        return {
            'phi': self.phi.copy(),
            'pi': self.pi_field.copy(),
            'u': self.pi_field.copy(),   # Alias for unified interface
            'v': grad_mag,               # Alias for unified interface
            'velocity_magnitude': np.abs(self.pi_field),
            'energy_density': E['total'],
            'kinetic_density': E['kinetic'],
            'gradient_density': E['gradient'],
            'potential_density': E['potential'],
            'gradient_magnitude': grad_mag,
            'time': self.time,
        }
