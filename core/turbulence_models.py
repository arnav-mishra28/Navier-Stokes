"""Turbulence Models"""

import numpy as np
from typing import Tuple, Optional


class TurbulenceModelFactory:
    """Factory for creating turbulence model instances."""
    
    @staticmethod
    def create(model_type: str, dx: float, dy: float, **kwargs):
        models = {
            'none': NoTurbulenceModel,
            'smagorinsky': SmagorinskyModel,
            'dynamic_smagorinsky': DynamicSmagorinskyModel,
            'k_epsilon': KEpsilonModel,
            'k_omega': KOmegaModel,
            'dns': NoTurbulenceModel,  # DNS = no model, resolve everything
        }
        if model_type not in models:
            raise ValueError(f"Unknown turbulence model: {model_type}")
        return models[model_type](dx, dy, **kwargs)


class NoTurbulenceModel:
    """No turbulence model (laminar flow or DNS)."""
    
    def __init__(self, dx: float, dy: float, **kwargs):
        self.dx = dx
        self.dy = dy
        self.nu_t = None
    
    def compute_eddy_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        """Return zero eddy viscosity."""
        return np.zeros_like(u)
    
    def get_total_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> float:
        """Return molecular viscosity only (scalar for constant-viscosity path)."""
        return nu


class SmagorinskyModel:
    """
    Smagorinsky LES subgrid model.
    
    ν_t = (C_s * Δ)² * |S̄|
    
    where:
        C_s ≈ 0.1-0.2 (Smagorinsky constant)
        Δ = (dx * dy)^(1/2) (filter width)
        |S̄| = sqrt(2 * S̄_ij * S̄_ij) (resolved strain rate magnitude)
    """
    
    def __init__(self, dx: float, dy: float, Cs: float = 0.17, **kwargs):
        self.dx = dx
        self.dy = dy
        self.Cs = Cs
        self.delta = np.sqrt(dx * dy)  # Filter width
        self.nu_t = None
    
    def compute_strain_rate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute resolved strain rate magnitude |S̄|."""
        dudx = np.gradient(u, self.dx, axis=1)
        dudy = np.gradient(u, self.dy, axis=0)
        dvdx = np.gradient(v, self.dx, axis=1)
        dvdy = np.gradient(v, self.dy, axis=0)
        
        S11 = dudx
        S22 = dvdy
        S12 = 0.5 * (dudy + dvdx)
        
        return np.sqrt(2.0 * (S11**2 + S22**2 + 2*S12**2) + 1e-10)
    
    def compute_eddy_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        """
        Compute Smagorinsky eddy viscosity.
        
        ν_t = (C_s * Δ)² * |S̄|
        """
        S_mag = self.compute_strain_rate(u, v)
        self.nu_t = (self.Cs * self.delta)**2 * S_mag
        return self.nu_t
    
    def get_total_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        """Return total viscosity: ν_total = ν + ν_t."""
        nu_t = self.compute_eddy_viscosity(u, v, nu)
        return nu + nu_t


class DynamicSmagorinskyModel:
    """
    Germano Dynamic Smagorinsky Model.
    
    Automatically computes the optimal C_s using a test filter.
    Uses the Germano identity with Lilly's least-squares formulation.
    """
    
    def __init__(self, dx: float, dy: float, test_filter_ratio: float = 2.0, **kwargs):
        self.dx = dx
        self.dy = dy
        self.delta = np.sqrt(dx * dy)
        self.test_delta = test_filter_ratio * self.delta
        self.test_filter_ratio = test_filter_ratio
        self.Cs_field = None
        self.nu_t = None
    
    def _test_filter(self, phi: np.ndarray) -> np.ndarray:
        """Apply test filter (box filter with width = 2Δ)."""
        from scipy.ndimage import uniform_filter
        size = max(int(self.test_filter_ratio), 2)
        return uniform_filter(phi, size=size, mode='wrap')
    
    def compute_eddy_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        """
        Compute dynamic Smagorinsky eddy viscosity.
        
        Uses Germano identity: L_ij = T_ij - τ̂_ij
        where L_ij = <u_i u_j> - <u_i><u_j> (resolved turbulent stress)
        """
        # Grid-level strain rate
        dudx = np.gradient(u, self.dx, axis=1)
        dudy = np.gradient(u, self.dy, axis=0)
        dvdx = np.gradient(v, self.dx, axis=1)
        dvdy = np.gradient(v, self.dy, axis=0)
        
        S11 = dudx
        S22 = dvdy
        S12 = 0.5 * (dudy + dvdx)
        S_mag = np.sqrt(2.0 * (S11**2 + S22**2 + 2*S12**2) + 1e-10)
        
        # Test-filtered velocities
        u_hat = self._test_filter(u)
        v_hat = self._test_filter(v)
        
        # Test-level strain rate
        dudx_h = np.gradient(u_hat, self.dx, axis=1)
        dudy_h = np.gradient(u_hat, self.dy, axis=0)
        dvdx_h = np.gradient(v_hat, self.dx, axis=1)
        dvdy_h = np.gradient(v_hat, self.dy, axis=0)
        
        S11_h = dudx_h
        S22_h = dvdy_h
        S12_h = 0.5 * (dudy_h + dvdx_h)
        S_mag_h = np.sqrt(2.0 * (S11_h**2 + S22_h**2 + 2*S12_h**2) + 1e-10)
        
        # Leonard stress tensor
        L11 = self._test_filter(u * u) - u_hat * u_hat
        L22 = self._test_filter(v * v) - v_hat * v_hat
        L12 = self._test_filter(u * v) - u_hat * v_hat
        
        # Germano identity components
        M11 = 2 * (self.test_delta**2 * S_mag_h * S11_h - self._test_filter(self.delta**2 * S_mag * S11))
        M22 = 2 * (self.test_delta**2 * S_mag_h * S22_h - self._test_filter(self.delta**2 * S_mag * S22))
        M12 = 2 * (self.test_delta**2 * S_mag_h * S12_h - self._test_filter(self.delta**2 * S_mag * S12))
        
        # Lilly's least-squares: C_s² = <L_ij M_ij> / <M_ij M_ij>
        LM = L11*M11 + L22*M22 + 2*L12*M12
        MM = M11**2 + M22**2 + 2*M12**2
        
        Cs_sq = np.clip(LM / (MM + 1e-10), 0, 0.5)
        
        # Averaging for stability
        from scipy.ndimage import uniform_filter
        Cs_sq = uniform_filter(Cs_sq, size=3, mode='wrap')
        
        self.Cs_field = np.sqrt(np.abs(Cs_sq))
        self.nu_t = Cs_sq * self.delta**2 * S_mag
        
        return self.nu_t
    
    def get_total_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        nu_t = self.compute_eddy_viscosity(u, v, nu)
        return nu + nu_t


class KEpsilonModel:
    """
    Standard k-ε RANS turbulence model.
    
    Transport equations:
        ∂k/∂t + u·∇k = ∇·[(ν + ν_t/σ_k)∇k] + P_k - ε
        ∂ε/∂t + u·∇ε = ∇·[(ν + ν_t/σ_ε)∇ε] + C₁ε(ε/k)P_k - C₂ε(ε²/k)
    
    where: ν_t = C_μ * k² / ε
    
    Model constants:
        C_μ = 0.09, C₁ε = 1.44, C₂ε = 1.92, σ_k = 1.0, σ_ε = 1.3
    """
    
    def __init__(self, dx: float, dy: float, **kwargs):
        self.dx = dx
        self.dy = dy
        
        # Model constants (standard values)
        self.C_mu = 0.09
        self.C1e = 1.44
        self.C2e = 1.92
        self.sigma_k = 1.0
        self.sigma_e = 1.3
        
        # State variables
        self.k = None  # Turbulent kinetic energy
        self.epsilon = None  # Dissipation rate
        self.nu_t = None
    
    def initialize(self, shape: Tuple[int, int], k0: float = 0.01, eps0: float = 0.001):
        """Initialize k and ε fields."""
        self.k = np.full(shape, k0)
        self.epsilon = np.full(shape, eps0)
    
    def _production_term(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute turbulence production: P_k = ν_t * |S|²"""
        dudx = np.gradient(u, self.dx, axis=1)
        dudy = np.gradient(u, self.dy, axis=0)
        dvdx = np.gradient(v, self.dx, axis=1)
        dvdy = np.gradient(v, self.dy, axis=0)
        
        S_sq = 2 * (dudx**2 + dvdy**2 + 0.5*(dudy + dvdx)**2)
        return self.nu_t * S_sq
    
    def compute_eddy_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        """Compute eddy viscosity: ν_t = C_μ * k² / ε"""
        if self.k is None:
            self.initialize(u.shape)
        
        self.nu_t = self.C_mu * self.k**2 / (self.epsilon + 1e-10)
        self.nu_t = np.clip(self.nu_t, 0, 1000 * nu)  # Limit for stability
        return self.nu_t
    
    def update(self, u: np.ndarray, v: np.ndarray, nu: float, dt: float):
        """
        Advance k-ε transport equations by one time step.
        Uses explicit Euler for simplicity.
        """
        if self.k is None:
            self.initialize(u.shape)
        
        self.compute_eddy_viscosity(u, v, nu)
        
        # Production term
        P_k = self._production_term(u, v)
        
        # Diffusion of k
        nu_eff_k = nu + self.nu_t / self.sigma_k
        lap_k = (
            np.gradient(nu_eff_k * np.gradient(self.k, self.dx, axis=1), self.dx, axis=1) +
            np.gradient(nu_eff_k * np.gradient(self.k, self.dy, axis=0), self.dy, axis=0)
        )
        
        # Diffusion of ε
        nu_eff_e = nu + self.nu_t / self.sigma_e
        lap_e = (
            np.gradient(nu_eff_e * np.gradient(self.epsilon, self.dx, axis=1), self.dx, axis=1) +
            np.gradient(nu_eff_e * np.gradient(self.epsilon, self.dy, axis=0), self.dy, axis=0)
        )
        
        # Advection
        advect_k = u * np.gradient(self.k, self.dx, axis=1) + v * np.gradient(self.k, self.dy, axis=0)
        advect_e = u * np.gradient(self.epsilon, self.dx, axis=1) + v * np.gradient(self.epsilon, self.dy, axis=0)
        
        # Time advancement
        self.k += dt * (-advect_k + lap_k + P_k - self.epsilon)
        self.epsilon += dt * (-advect_e + lap_e + 
                             self.C1e * self.epsilon / (self.k + 1e-10) * P_k -
                             self.C2e * self.epsilon**2 / (self.k + 1e-10))
        
        # Ensure positivity
        self.k = np.maximum(self.k, 1e-10)
        self.epsilon = np.maximum(self.epsilon, 1e-10)
    
    def get_total_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        nu_t = self.compute_eddy_viscosity(u, v, nu)
        return nu + nu_t


class KOmegaModel:
    """
    Wilcox k-ω turbulence model.
    
    Transport equations:
        ∂k/∂t + u·∇k = P_k - β*k*ω + ∇·[(ν + σ*ν_t)∇k]
        ∂ω/∂t + u·∇ω = α(ω/k)P_k - βω² + ∇·[(ν + σ*ν_t)∇ω]
    
    where: ν_t = k / ω
    
    Advantages over k-ε:
        - Better near-wall behavior
        - No wall functions needed
        - Better for adverse pressure gradients
    """
    
    def __init__(self, dx: float, dy: float, **kwargs):
        self.dx = dx
        self.dy = dy
        
        # Model constants (Wilcox 2006)
        self.alpha = 5.0 / 9.0
        self.beta = 3.0 / 40.0
        self.beta_star = 0.09
        self.sigma = 0.5
        self.sigma_d = 0.125
        
        self.k = None
        self.omega = None
        self.nu_t = None
    
    def initialize(self, shape: Tuple[int, int], k0: float = 0.01, omega0: float = 1.0):
        self.k = np.full(shape, k0)
        self.omega = np.full(shape, omega0)
    
    def compute_eddy_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        if self.k is None:
            self.initialize(u.shape)
        
        self.nu_t = self.k / (self.omega + 1e-10)
        self.nu_t = np.clip(self.nu_t, 0, 1000 * nu)
        return self.nu_t
    
    def update(self, u: np.ndarray, v: np.ndarray, nu: float, dt: float):
        if self.k is None:
            self.initialize(u.shape)
        
        self.compute_eddy_viscosity(u, v, nu)
        
        # Production
        dudx = np.gradient(u, self.dx, axis=1)
        dudy = np.gradient(u, self.dy, axis=0)
        dvdx = np.gradient(v, self.dx, axis=1)
        dvdy = np.gradient(v, self.dy, axis=0)
        S_sq = 2 * (dudx**2 + dvdy**2 + 0.5*(dudy + dvdx)**2)
        P_k = self.nu_t * S_sq
        
        # Diffusion
        nu_eff = nu + self.sigma * self.nu_t
        lap_k = (
            np.gradient(nu_eff * np.gradient(self.k, self.dx, axis=1), self.dx, axis=1) +
            np.gradient(nu_eff * np.gradient(self.k, self.dy, axis=0), self.dy, axis=0)
        )
        lap_w = (
            np.gradient(nu_eff * np.gradient(self.omega, self.dx, axis=1), self.dx, axis=1) +
            np.gradient(nu_eff * np.gradient(self.omega, self.dy, axis=0), self.dy, axis=0)
        )
        
        # Advection
        advect_k = u * np.gradient(self.k, self.dx, axis=1) + v * np.gradient(self.k, self.dy, axis=0)
        advect_w = u * np.gradient(self.omega, self.dx, axis=1) + v * np.gradient(self.omega, self.dy, axis=0)
        
        # Cross-diffusion term
        dkdx = np.gradient(self.k, self.dx, axis=1)
        dkdy = np.gradient(self.k, self.dy, axis=0)
        dwdx = np.gradient(self.omega, self.dx, axis=1)
        dwdy = np.gradient(self.omega, self.dy, axis=0)
        cross_diff = self.sigma_d / (self.omega + 1e-10) * np.maximum(
            dkdx * dwdx + dkdy * dwdy, 0
        )
        
        # Update
        self.k += dt * (-advect_k + lap_k + P_k - self.beta_star * self.k * self.omega)
        self.omega += dt * (-advect_w + lap_w + 
                           self.alpha * self.omega / (self.k + 1e-10) * P_k -
                           self.beta * self.omega**2 + cross_diff)
        
        self.k = np.maximum(self.k, 1e-10)
        self.omega = np.maximum(self.omega, 1e-10)
    
    def get_total_viscosity(
        self, u: np.ndarray, v: np.ndarray, nu: float
    ) -> np.ndarray:
        nu_t = self.compute_eddy_viscosity(u, v, nu)
        return nu + nu_t
