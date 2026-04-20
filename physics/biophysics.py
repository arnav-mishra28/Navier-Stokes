"""
=============================================================================
Biophysics Flow Solver
Fluid dynamics in biological systems.

Applications:
    1. Blood flow in arteries (hemodynamics)
       - Pulsatile flow with Womersley profiles
       - Non-Newtonian rheology (shear-thinning blood)
       - Vessel compliance and elasticity
    
    2. Airflow in lungs
       - Branching airways (Weibel model)
       - Particle deposition
    
    3. Cerebrospinal fluid dynamics
    4. Cell swimming (low Reynolds number)

Key models:
    - Carreau-Yasuda for blood viscosity: η(γ̇) = η∞ + (η₀-η∞)[1+(λγ̇)²]^((n-1)/2)
    - Womersley number: α = R√(ω/ν) (pulsatile flow parameter)
    - Murray's law: r³_parent = r³_child1 + r³_child2 (optimal branching)
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class BiophysicsFlowSolver:
    """
    Solver for biological fluid dynamics.
    
    Specializes in hemodynamics (blood flow) with:
        - Pulsatile (cardiac-cycle) inflow
        - Non-Newtonian blood rheology
        - Vessel wall compliance
        - Wall shear stress computation (atherosclerosis indicator)
    """
    
    # Blood properties (typical values)
    BLOOD_DENSITY = 1060.0         # kg/m³
    PLASMA_VISCOSITY = 0.0035      # Pa·s
    BLOOD_VISCOSITY_INF = 0.0035   # Pa·s (high shear rate)
    BLOOD_VISCOSITY_0 = 0.056      # Pa·s (zero shear rate)
    HEART_RATE = 72                # bpm
    SYSTOLIC_VELOCITY = 0.5        # m/s
    
    def __init__(
        self,
        nx: int = 256, ny: int = 64,
        Lx: float = 0.1, Ly: float = 0.01,  # 10cm × 1cm artery
        dt: float = 0.0001,
        flow_type: str = "arterial",  # "arterial", "venous", "capillary", "airway"
        non_newtonian: bool = True,
        pulsatile: bool = True,
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.dt = dt
        self.flow_type = flow_type
        self.non_newtonian = non_newtonian
        self.pulsatile = pulsatile
        
        # Flow fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        # Viscosity field (non-Newtonian → spatially varying)
        self.viscosity = np.full((ny, nx), self.PLASMA_VISCOSITY)
        
        # Wall shear stress
        self.wss = np.zeros(nx)  # Wall shear stress along vessel
        
        # Vessel wall (can be compliant)
        self.wall_top = np.full(nx, Ly)
        self.wall_bottom = np.zeros(nx)
        self.wall_displacement = np.zeros(nx)
        
        # Grid
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Set cardiac cycle parameters
        self.omega = 2 * np.pi * self.HEART_RATE / 60  # Angular frequency
        self.womersley_number = (Ly/2) * np.sqrt(self.omega * self.BLOOD_DENSITY / self.PLASMA_VISCOSITY)
        
        # Carreau-Yasuda model parameters
        self.eta_0 = self.BLOOD_VISCOSITY_0
        self.eta_inf = self.BLOOD_VISCOSITY_INF
        self.lambda_cy = 3.313    # Relaxation time (s)
        self.n_cy = 0.3568        # Power-law index
        self.a_cy = 2.0           # Yasuda exponent
        
        self.time = 0.0
        self.step_count = 0
        
        self.history: Dict[str, List] = {
            'time': [], 'flow_rate': [], 'max_wss': [],
            'mean_viscosity': [], 'pressure_drop': []
        }
    
    def _carreau_yasuda(self, shear_rate: np.ndarray) -> np.ndarray:
        """
        Carreau-Yasuda model for blood viscosity.
        
        η(γ̇) = η∞ + (η₀ - η∞) * [1 + (λ·γ̇)^a]^((n-1)/a)
        
        At high shear rates (arteries): η → η∞ (Newtonian-like)
        At low shear rates (veins): η → η₀ (highly viscous, non-Newtonian)
        """
        return (self.eta_inf + 
                (self.eta_0 - self.eta_inf) * 
                (1 + (self.lambda_cy * shear_rate)**self.a_cy)**((self.n_cy - 1)/self.a_cy))
    
    def _compute_shear_rate(self) -> np.ndarray:
        """
        Compute strain rate magnitude: γ̇ = |∂u/∂y| (dominant in pipe flow).
        """
        dudy = np.gradient(self.u, self.dy, axis=0)
        dvdx = np.gradient(self.v, self.dx, axis=1)
        return np.sqrt(dudy**2 + dvdx**2 + 1e-10)
    
    def _womersley_profile(self, t: float) -> np.ndarray:
        """
        Womersley velocity profile for pulsatile flow.
        
        Combines steady Poiseuille + oscillatory (Bessel function) component.
        Simplified version using Fourier modes.
        
        u(y,t) = U_steady(y) + Σ_n U_n(y) * sin(nωt + φ_n)
        """
        y = np.linspace(0, 1, self.ny)  # Normalized
        R = self.Ly / 2
        y_centered = (y - 0.5) * 2  # [-1, 1]
        
        # Steady Poiseuille component
        u_steady = self.SYSTOLIC_VELOCITY * (1 - y_centered**2)
        
        # Pulsatile component (cardiac waveform approximation)
        # Typical cardiac output: systole + diastole
        phase = self.omega * t
        
        # Multi-harmonic cardiac waveform
        pulsatile = (
            0.6 * np.sin(phase) +                    # Fundamental
            0.2 * np.sin(2*phase - 0.5) +            # 2nd harmonic
            0.1 * np.sin(3*phase - 1.0) +            # 3rd harmonic
            0.05 * np.sin(4*phase - 1.5)             # 4th harmonic
        )
        
        # Womersley modification: blunter profile at high Womersley number
        alpha = self.womersley_number
        if alpha > 5:
            # High Womersley: nearly flat profile (plug flow)
            profile_mod = np.ones_like(y_centered)
        elif alpha < 1:
            # Low Womersley: quasi-steady (parabolic)
            profile_mod = 1 - y_centered**2
        else:
            # Intermediate: blunted parabola
            profile_mod = 1 - np.abs(y_centered)**(2 + alpha/5)
        
        return u_steady + pulsatile * self.SYSTOLIC_VELOCITY * 0.5 * profile_mod
    
    def initialize_straight_vessel(self, stenosis: float = 0.0, stenosis_pos: float = 0.5):
        """
        Initialize flow in a straight vessel, optionally with stenosis.
        
        Args:
            stenosis: Fraction of vessel diameter blocked (0-0.9)
            stenosis_pos: Axial position of stenosis (0-1 normalized)
        """
        # Parabolic initial velocity
        y_norm = np.linspace(0, 1, self.ny)
        self.u[:, :] = self.SYSTOLIC_VELOCITY * 4 * y_norm[:, None] * (1 - y_norm[:, None])
        self.v[:] = 0
        
        # Add stenosis (constriction)
        if stenosis > 0:
            x_s = stenosis_pos * self.Lx
            sigma = 0.02 * self.Lx
            constriction = stenosis * np.exp(-(self.X[0, :] - x_s)**2 / (2*sigma**2))
            
            for j in range(self.ny):
                y_frac = j / self.ny
                # Narrow the vessel
                if y_frac < 0.5:
                    if y_frac < constriction[0] / 2:
                        self.u[j, :] = 0
                elif y_frac > 1 - constriction[0] / 2:
                    self.u[j, :] = 0
    
    def initialize_bifurcation(self):
        """
        Initialize arterial bifurcation (Y-junction).
        
        Models blood flow splitting at arterial branches.
        Critical for studying atherosclerosis at branch points.
        """
        # Parent vessel in first third, then split
        y_norm = self.Y / self.Ly
        x_norm = self.X / self.Lx
        
        # Simple bifurcation mask
        in_parent = x_norm < 0.3
        
        angle = np.pi / 6  # 30° bifurcation
        center = self.Ly / 2
        
        branch_top = (x_norm >= 0.3) & (self.Y > center + (self.X - 0.3*self.Lx) * np.tan(angle) * 0.3)
        branch_bottom = (x_norm >= 0.3) & (self.Y < center - (self.X - 0.3*self.Lx) * np.tan(angle) * 0.3)
        
        # Set velocity in parent vessel
        self.u = np.where(in_parent, 
                         self.SYSTOLIC_VELOCITY * 4 * y_norm * (1-y_norm), 
                         0.0)
    
    def _compute_wss(self):
        """
        Compute Wall Shear Stress (WSS).
        
        WSS = μ * ∂u/∂y|_{wall}
        
        Low WSS → atherosclerosis risk
        High WSS → endothelial activation
        Oscillatory WSS → plaque development
        """
        # Bottom wall
        dudy_bottom = (self.u[1, :] - self.u[0, :]) / self.dy
        wss_bottom = np.mean(self.viscosity[0, :]) * np.abs(dudy_bottom)
        
        # Top wall
        dudy_top = (self.u[-1, :] - self.u[-2, :]) / self.dy
        wss_top = np.mean(self.viscosity[-1, :]) * np.abs(dudy_top)
        
        self.wss = 0.5 * (wss_bottom + wss_top)
    
    def step(self):
        """Advance hemodynamic simulation by one time step."""
        # Update non-Newtonian viscosity
        if self.non_newtonian:
            shear_rate = self._compute_shear_rate()
            self.viscosity = self._carreau_yasuda(shear_rate) / self.BLOOD_DENSITY
        else:
            self.viscosity[:] = self.PLASMA_VISCOSITY / self.BLOOD_DENSITY
        
        # Advection
        dudx = np.gradient(self.u, self.dx, axis=1)
        dudy = np.gradient(self.u, self.dy, axis=0)
        dvdx = np.gradient(self.v, self.dx, axis=1)
        dvdy = np.gradient(self.v, self.dy, axis=0)
        
        adv_u = self.u * dudx + self.v * dudy
        adv_v = self.u * dvdx + self.v * dvdy
        
        # Diffusion with variable viscosity
        lap_u = np.gradient(self.viscosity * dudx, self.dx, axis=1) + \
                np.gradient(self.viscosity * dudy, self.dy, axis=0)
        lap_v = np.gradient(self.viscosity * dvdx, self.dx, axis=1) + \
                np.gradient(self.viscosity * dvdy, self.dy, axis=0)
        
        # Predict velocity
        u_star = self.u + self.dt * (-adv_u + lap_u)
        v_star = self.v + self.dt * (-adv_v + lap_v)
        
        # Apply pulsatile inlet BC
        if self.pulsatile:
            u_star[:, 0] = self._womersley_profile(self.time)
            v_star[:, 0] = 0
        
        # No-slip walls
        u_star[0, :] = 0
        u_star[-1, :] = 0
        v_star[0, :] = 0
        v_star[-1, :] = 0
        
        # Outflow BC
        u_star[:, -1] = u_star[:, -2]
        v_star[:, -1] = v_star[:, -2]
        
        # Solve pressure (simplified for channel)
        div = np.gradient(u_star, self.dx, axis=1) + np.gradient(v_star, self.dy, axis=0)
        # Iterative pressure solve
        for _ in range(50):
            self.p[1:-1, 1:-1] = 0.25 * (
                self.p[1:-1, 2:] + self.p[1:-1, :-2] +
                self.p[2:, 1:-1] + self.p[:-2, 1:-1] -
                self.dx**2 * div[1:-1, 1:-1] / self.dt
            )
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = 0  # Reference pressure at outlet
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
        
        # Correct velocity
        self.u = u_star - self.dt * np.gradient(self.p, self.dx, axis=1)
        self.v = v_star - self.dt * np.gradient(self.p, self.dy, axis=0)
        
        # Re-apply BCs
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0
        
        # Compute WSS
        self._compute_wss()
        
        self.time += self.dt
        self.step_count += 1
    
    def advance(self, n_steps: int = 1, record: bool = True):
        for _ in range(n_steps):
            self.step()
            if record:
                self._record_diagnostics()
    
    def _record_diagnostics(self):
        flow_rate = np.sum(self.u[:, self.nx//2]) * self.dy
        pressure_drop = np.mean(self.p[:, 0]) - np.mean(self.p[:, -1])
        
        self.history['time'].append(self.time)
        self.history['flow_rate'].append(float(flow_rate))
        self.history['max_wss'].append(float(np.max(self.wss)))
        self.history['mean_viscosity'].append(float(np.mean(self.viscosity)))
        self.history['pressure_drop'].append(float(pressure_drop))
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'u': self.u.copy(), 'v': self.v.copy(), 'p': self.p.copy(),
            'viscosity': self.viscosity.copy(), 'wss': self.wss.copy(),
            'velocity_magnitude': np.sqrt(self.u**2 + self.v**2),
            'shear_rate': self._compute_shear_rate(),
            'time': self.time,
        }
