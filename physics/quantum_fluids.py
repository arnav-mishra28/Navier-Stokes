"""
=============================================================================
Quantum Fluid Solver
Simulates superfluid dynamics using the Gross-Pitaevskii equation (GPE).

The GPE describes:
    - Bose-Einstein condensates (BEC)
    - Superfluid helium-4 (He-II)
    - Quantized vortices

Governing equation:
    iℏ ∂ψ/∂t = [-ℏ²/(2m)∇² + V(r) + g|ψ|²]ψ

The wavefunction ψ encodes both density and velocity via Madelung transform:
    ρ = |ψ|²             (superfluid density)
    u = (ℏ/m)∇(arg(ψ))   (superfluid velocity = gradient of phase)

Key features:
    - Irrotational flow EXCEPT at quantized vortices where ψ = 0
    - Circulation is quantized: ∮ u·dl = n × h/m
    - Quantum pressure (Bohm potential) prevents classical singularities
    - Vortex reconnection events
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class QuantumFluidSolver:
    """
    Gross-Pitaevskii equation solver for quantum fluids.
    
    Uses split-step Fourier method (SSFM):
        1. Half-step potential evolution: ψ → exp(-iV*dt/2ℏ) * ψ
        2. Full-step kinetic (in k-space): ψ̂ → exp(-iℏk²dt/2m) * ψ̂
        3. Half-step potential: ψ → exp(-iV*dt/2ℏ) * ψ
    
    This is exactly the quantum analog of classical operator splitting.
    
    Applications:
        - Cold atom BEC experiments
        - Superfluid helium vortex dynamics
        - Dark solitons and vortex rings
        - Quantum turbulence (Kolmogorov-like cascade)
    """
    
    def __init__(
        self,
        nx: int = 256, ny: int = 256,
        Lx: float = 20.0, Ly: float = 20.0,
        hbar: float = 1.0,      # Reduced Planck constant
        m: float = 1.0,          # Particle mass
        g_int: float = 100.0,    # Interaction strength (nonlinearity)
        dt: float = 0.001,
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.hbar = hbar
        self.m = m
        self.g_int = g_int
        self.dt = dt
        
        # Complex wavefunction
        self.psi = np.ones((ny, nx), dtype=complex)
        
        # External potential
        self.V_ext = np.zeros((ny, nx))
        
        # Coordinate grids
        x = np.linspace(-Lx/2, Lx/2, nx, endpoint=False)
        y = np.linspace(-Ly/2, Ly/2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Momentum space grids
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        
        # Pre-compute kinetic energy propagator
        self.kinetic_propagator = np.exp(-1j * self.hbar * self.K2 * dt / (2 * m))
        
        self.time = 0.0
        self.step_count = 0
        
        # Healing length: ξ = ℏ / √(2 * m * g * n₀) — sets vortex core size
        self.healing_length = None
        
        self.history: Dict[str, List] = {
            'time': [], 'total_particles': [], 'energy': [],
            'n_vortices': [], 'angular_momentum': []
        }
    
    def initialize_ground_state(self, n0: float = 1.0, trap_freq: float = 0.0):
        """
        Initialize condensate ground state (Thomas-Fermi approximation).
        
        For harmonic trap: V = 0.5 * m * ω² * r²
        TF profile: n(r) = max(0, (μ - V(r)) / g)
        
        For uniform: n(r) = n₀
        """
        if trap_freq > 0:
            # Harmonic trap
            self.V_ext = 0.5 * self.m * trap_freq**2 * (self.X**2 + self.Y**2)
            
            # Thomas-Fermi chemical potential
            R_TF = np.sqrt(2 * self.g_int * n0 / (self.m * trap_freq**2))
            mu = 0.5 * self.m * trap_freq**2 * R_TF**2
            
            # TF density profile
            n_TF = np.maximum(0, (mu - self.V_ext) / self.g_int)
            self.psi = np.sqrt(n_TF + 1e-10).astype(complex)
        else:
            # Uniform condensate
            self.psi = np.sqrt(n0) * np.ones((self.ny, self.nx), dtype=complex)
        
        # Healing length
        self.healing_length = self.hbar / np.sqrt(2 * self.m * self.g_int * n0)
    
    def imprint_vortex(self, x0: float, y0: float, charge: int = 1):
        """
        Imprint a quantized vortex at position (x0, y0).
        
        Phase winding: φ = charge * arctan2(y-y0, x-x0)
        Density: goes to zero at vortex core (size ~ healing length)
        
        charge = ±1 (singly quantized), ±2 (doubly, usually unstable)
        """
        theta = np.arctan2(self.Y - y0, self.X - x0)
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        
        # Vortex phase
        phase = charge * theta
        
        # Density profile: tanh(r/ξ) approximation for vortex core
        xi = self.healing_length if self.healing_length else 0.5
        density_mod = np.tanh(r / (np.sqrt(2) * xi))
        
        self.psi *= density_mod * np.exp(1j * phase)
    
    def imprint_vortex_pair(
        self, separation: float = 2.0,
        same_sign: bool = False
    ):
        """
        Create a vortex-antivortex pair (or co-rotating pair).
        
        Opposite-sign pair: translates linearly
        Same-sign pair: orbits around center of vorticity
        """
        x1 = -separation / 2
        x2 = separation / 2
        
        charge2 = 1 if same_sign else -1
        
        self.imprint_vortex(x1, 0, charge=1)
        self.imprint_vortex(x2, 0, charge=charge2)
    
    def imprint_dark_soliton(self, x0: float = 0.0, velocity: float = 0.0):
        """
        Create a dark soliton (1D density notch in 2D condensate).
        
        Exact solution: ψ(x) = √n₀ * [i*v/c + √(1-v²/c²) * tanh(x/ξ_s)]
        where c = √(g*n₀/m) is the speed of sound.
        """
        n0 = np.mean(np.abs(self.psi)**2)
        c = np.sqrt(self.g_int * n0 / self.m)
        v_ratio = velocity / c if c > 0 else 0
        
        xi_s = self.healing_length / np.sqrt(1 - v_ratio**2 + 1e-10)
        
        soliton_profile = (1j * v_ratio + 
                          np.sqrt(1 - v_ratio**2) * np.tanh((self.X - x0) / xi_s))
        
        self.psi *= soliton_profile
    
    def initialize_quantum_turbulence(self, n_vortices: int = 20):
        """
        Initialize quantum turbulence with random vortex tangle.
        
        Quantum turbulence differs from classical:
        - Vorticity is concentrated in filaments
        - Circulation is quantized
        - Shows Kolmogorov -5/3 at large scales
        - But different spectrum at small scales (Kelvin waves)
        """
        n0 = 1.0
        self.initialize_ground_state(n0=n0)
        
        for _ in range(n_vortices):
            x = np.random.uniform(-self.Lx/3, self.Lx/3)
            y = np.random.uniform(-self.Ly/3, self.Ly/3)
            charge = np.random.choice([-1, 1])
            self.imprint_vortex(x, y, charge)
    
    def step(self):
        """
        Advance GPE by one time step using split-step Fourier method.
        
        This is a symplectic integrator (preserves unitarity/norm).
        """
        # Potential energy (external + nonlinear)
        V_total = self.V_ext + self.g_int * np.abs(self.psi)**2
        
        # Half-step potential evolution
        self.psi *= np.exp(-1j * V_total * self.dt / (2 * self.hbar))
        
        # Full-step kinetic evolution (in Fourier space)
        psi_k = np.fft.fft2(self.psi)
        psi_k *= self.kinetic_propagator
        self.psi = np.fft.ifft2(psi_k)
        
        # Half-step potential evolution (with updated |ψ|²)
        V_total = self.V_ext + self.g_int * np.abs(self.psi)**2
        self.psi *= np.exp(-1j * V_total * self.dt / (2 * self.hbar))
        
        self.time += self.dt
        self.step_count += 1
    
    def advance(self, n_steps: int = 1, record: bool = True):
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()
    
    def get_density(self) -> np.ndarray:
        """Superfluid density: n = |ψ|²"""
        return np.abs(self.psi)**2
    
    def get_phase(self) -> np.ndarray:
        """Phase of the wavefunction."""
        return np.angle(self.psi)
    
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Superfluid velocity from phase gradient:
        u = (ℏ/m) * ∇(arg(ψ))
        
        Uses phase unwrapping for smooth gradients.
        """
        phase = self.get_phase()
        
        # Phase gradient with periodic differences
        dphase_dx = np.angle(np.roll(self.psi, -1, 1) * np.conj(self.psi)) / self.dx
        dphase_dy = np.angle(np.roll(self.psi, -1, 0) * np.conj(self.psi)) / self.dy
        
        ux = (self.hbar / self.m) * dphase_dx
        uy = (self.hbar / self.m) * dphase_dy
        
        return ux, uy
    
    def detect_vortices(self) -> List[Tuple[float, float, int]]:
        """
        Detect quantized vortices by computing phase winding number.
        
        For each plaquette (grid cell), sum phase differences around it.
        If sum ≈ ±2π → vortex with charge ±1
        
        Returns: List of (x, y, charge) tuples
        """
        vortices = []
        
        phase = self.get_phase()
        
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                # Phase differences around plaquette
                dp1 = np.angle(np.exp(1j * (phase[j, (i+1)%self.nx] - phase[j, i])))
                dp2 = np.angle(np.exp(1j * (phase[(j+1)%self.ny, (i+1)%self.nx] - phase[j, (i+1)%self.nx])))
                dp3 = np.angle(np.exp(1j * (phase[(j+1)%self.ny, i] - phase[(j+1)%self.ny, (i+1)%self.nx])))
                dp4 = np.angle(np.exp(1j * (phase[j, i] - phase[(j+1)%self.ny, i])))
                
                winding = (dp1 + dp2 + dp3 + dp4) / (2 * np.pi)
                
                if np.abs(winding) > 0.5:
                    charge = int(np.round(winding))
                    x = (i + 0.5) * self.dx - self.Lx/2
                    y = (j + 0.5) * self.dy - self.Ly/2
                    vortices.append((x, y, charge))
        
        return vortices
    
    def compute_energy(self) -> Dict[str, float]:
        """
        Compute energy components:
            E_kin = (ℏ²/2m) ∫ |∇ψ|² dx
            E_pot = ∫ V|ψ|² dx  
            E_int = (g/2) ∫ |ψ|⁴ dx
        """
        psi_k = np.fft.fft2(self.psi)
        
        # Kinetic energy (computed in k-space)
        E_kin = (self.hbar**2 / (2*self.m)) * np.sum(self.K2 * np.abs(psi_k)**2) * self.dx * self.dy / (self.nx * self.ny)
        
        # Potential energy
        E_pot = np.sum(self.V_ext * np.abs(self.psi)**2) * self.dx * self.dy
        
        # Interaction energy
        E_int = 0.5 * self.g_int * np.sum(np.abs(self.psi)**4) * self.dx * self.dy
        
        return {
            'kinetic': float(E_kin),
            'potential': float(E_pot),
            'interaction': float(E_int),
            'total': float(E_kin + E_pot + E_int)
        }
    
    def compute_incompressible_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute incompressible kinetic energy spectrum.
        
        For quantum turbulence, expect:
            - E(k) ~ k^(-5/3) at large scales (Kolmogorov)
            - E(k) ~ k^(-1) at intermediate scales (Kelvin waves)
            - E(k) ~ k^(-3) at small scales
        """
        ux, uy = self.get_velocity()
        density = self.get_density()
        
        # Weighted velocity
        wx = np.sqrt(density + 1e-10) * ux
        wy = np.sqrt(density + 1e-10) * uy
        
        wx_k = np.fft.fft2(wx)
        wy_k = np.fft.fft2(wy)
        
        E_k = 0.5 * (np.abs(wx_k)**2 + np.abs(wy_k)**2) / (self.nx * self.ny)**2
        
        # Radial binning
        k_mag = np.sqrt(self.KX**2 + self.KY**2)
        k_max = np.max(k_mag) / 2
        n_bins = min(self.nx, self.ny) // 4
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        spectrum = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            spectrum[i] = np.sum(E_k[mask])
        
        return k_centers, spectrum
    
    def _record_diagnostics(self):
        density = self.get_density()
        N = np.sum(density) * self.dx * self.dy  # Total particle number
        energy = self.compute_energy()
        vortices = self.detect_vortices()
        
        # Angular momentum
        ux, uy = self.get_velocity()
        Lz = np.sum(density * (self.X * uy - self.Y * ux)) * self.dx * self.dy
        
        self.history['time'].append(self.time)
        self.history['total_particles'].append(float(N))
        self.history['energy'].append(energy['total'])
        self.history['n_vortices'].append(len(vortices))
        self.history['angular_momentum'].append(float(Lz))
    
    def get_state(self) -> Dict[str, np.ndarray]:
        density = self.get_density()
        phase = self.get_phase()
        ux, uy = self.get_velocity()
        
        return {
            'density': density, 'phase': phase,
            'u': ux, 'v': uy,
            'velocity_magnitude': np.sqrt(ux**2 + uy**2),
            'psi_real': np.real(self.psi),
            'psi_imag': np.imag(self.psi),
            'V_ext': self.V_ext.copy(),
            'time': self.time,
        }
