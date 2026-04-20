"""
=============================================================================
2D Incompressible Navier-Stokes Solver
Chorin's Projection Method (Fractional Step)

Governing equations:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f    (Momentum)
    ∇·u = 0                                 (Continuity)

Algorithm:
    1. Predict velocity (ignore pressure): u* = u^n + Δt[-（u·∇)u + ν∇²u + f]
    2. Solve pressure Poisson: ∇²p = (ρ/Δt)∇·u*
    3. Correct velocity: u^{n+1} = u* - (Δt/ρ)∇p
    4. Apply boundary conditions
=============================================================================
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from .pressure_solver import PressureSolver, VectorizedPressureSolver
from .boundary_conditions import BoundaryConditionManager, BCType
from .turbulence_models import TurbulenceModelFactory
from .discretization import AdvectionSchemes, DiffusionSchemes, GradientOperators


class FluidSolver2D:
    """
    2D incompressible Navier-Stokes solver using Chorin's projection method.
    
    Features:
        - Multiple advection schemes (upwind, central, WENO-5)
        - Multiple pressure solvers (FFT, Jacobi, SOR, CG, multigrid)
        - Turbulence models (Smagorinsky LES, k-ε, k-ω)
        - Obstacle support (immersed boundary)
        - Adaptive time stepping
        - External force injection
        - Diagnostics (KE, enstrophy, divergence, CFL)
    """
    
    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        Lx: float = 1.0,
        Ly: float = 1.0,
        nu: float = 0.01,
        dt: float = 0.001,
        density: float = 1.0,
        pressure_solver: str = "fft",
        advection_scheme: str = "central",
        turbulence_model: str = "none"
    ):
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        
        # Physical parameters
        self.nu = nu  # Kinematic viscosity
        self.dt = dt
        self.density = density
        
        # Flow fields (collocated grid)
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # pressure
        
        # Intermediate velocity (prediction step)
        self.u_star = np.zeros((ny, nx))
        self.v_star = np.zeros((ny, nx))
        
        # External forces
        self.fx = np.zeros((ny, nx))
        self.fy = np.zeros((ny, nx))
        
        # Obstacle mask
        self.obstacle = np.zeros((ny, nx), dtype=bool)
        
        # Meshgrid
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Solvers and schemes
        self.advection_scheme = advection_scheme
        
        # Pressure solver
        self.pressure_solver_type = pressure_solver
        if pressure_solver == "fft":
            self.psolver = VectorizedPressureSolver(nx, ny, self.dx, self.dy)
        else:
            self.psolver = PressureSolver(nx, ny, self.dx, self.dy, method=pressure_solver)
        
        # Boundary condition manager
        self.bc_manager = BoundaryConditionManager(nx, ny, self.dx, self.dy)
        
        # Turbulence model
        self.turb_model = TurbulenceModelFactory.create(
            turbulence_model, self.dx, self.dy
        )
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        
        # Diagnostics history
        self.history: Dict[str, List[float]] = {
            'time': [],
            'kinetic_energy': [],
            'enstrophy': [],
            'max_divergence': [],
            'cfl': [],
            'max_velocity': [],
        }
    
    @property
    def is_periodic(self) -> bool:
        """Check if all boundaries are periodic."""
        return all(v == BCType.PERIODIC for v in self.bc_manager.bc_types.values())
    
    def initialize_taylor_green(self, amplitude: float = 1.0):
        """
        Initialize with Taylor-Green vortex (analytical benchmark).
        
        u(x,y,0) = A * cos(x) * sin(y)
        v(x,y,0) = -A * sin(x) * cos(y)
        p(x,y,0) = -A²/4 * (cos(2x) + cos(2y))
        
        Analytical solution: decays as exp(-2νt)
        """
        self.u = amplitude * np.cos(self.X) * np.sin(self.Y)
        self.v = -amplitude * np.sin(self.X) * np.cos(self.Y)
        self.p = -amplitude**2 / 4.0 * (np.cos(2*self.X) + np.cos(2*self.Y))
        
        # Set periodic BCs for this case
        self.bc_manager.set_periodic()
    
    def initialize_lid_driven_cavity(self, u_lid: float = 1.0):
        """Initialize lid-driven cavity flow."""
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.bc_manager.set_lid_driven_cavity(u_lid)
    
    def initialize_channel_flow(self, u_inlet: float = 1.0):
        """Initialize channel flow with parabolic inlet."""
        y_norm = np.linspace(0, 1, self.ny)
        self.u[:, :] = 4 * u_inlet * y_norm[:, None] * (1 - y_norm[:, None])
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.bc_manager.set_channel_flow(u_inlet)
    
    def initialize_double_shear_layer(self, amplitude: float = 0.05, delta: float = 0.05):
        """
        Double shear layer instability (Kelvin-Helmholtz).
        
        Tests ability to resolve vortex roll-up and pairing.
        """
        y = self.Y / self.Ly
        
        self.u = np.where(y < 0.5,
                         np.tanh((y - 0.25) / delta),
                         np.tanh((0.75 - y) / delta))
        
        self.v = amplitude * np.sin(2 * np.pi * self.X / self.Lx)
        self.p[:] = 0.0
        
        self.bc_manager.set_periodic()
    
    def initialize_vortex_pair(self, strength: float = 1.0, separation: float = 0.3):
        """Initialize counter-rotating vortex pair."""
        cx1, cy1 = self.Lx/2, self.Ly/2 + separation * self.Ly / 2
        cx2, cy2 = self.Lx/2, self.Ly/2 - separation * self.Ly / 2
        
        r1_sq = (self.X - cx1)**2 + (self.Y - cy1)**2
        r2_sq = (self.X - cx2)**2 + (self.Y - cy2)**2
        
        sigma = 0.05 * self.Lx
        
        self.u = (strength * -(self.Y - cy1) * np.exp(-r1_sq/(2*sigma**2)) +
                  strength * (self.Y - cy2) * np.exp(-r2_sq/(2*sigma**2)))
        self.v = (strength * (self.X - cx1) * np.exp(-r1_sq/(2*sigma**2)) +
                  strength * -(self.X - cx2) * np.exp(-r2_sq/(2*sigma**2)))
        self.p[:] = 0.0
        
        self.bc_manager.set_periodic()
    
    def set_obstacle(self, mask: np.ndarray):
        """Set obstacle mask (True = solid body)."""
        self.obstacle = mask.astype(bool)
        self.bc_manager.set_obstacle(mask)
    
    def add_circular_obstacle(self, cx: float, cy: float, radius: float):
        """Add a circular obstacle at (cx, cy) with given radius."""
        r_sq = (self.X - cx)**2 + (self.Y - cy)**2
        self.obstacle |= (r_sq < radius**2)
        self.bc_manager.set_obstacle(self.obstacle)
    
    def add_rectangular_obstacle(self, x0: float, y0: float, width: float, height: float):
        """Add a rectangular obstacle."""
        mask = ((self.X >= x0) & (self.X <= x0 + width) &
                (self.Y >= y0) & (self.Y <= y0 + height))
        self.obstacle |= mask
        self.bc_manager.set_obstacle(self.obstacle)
    
    def set_force(self, fx: np.ndarray, fy: np.ndarray):
        """Set external force fields."""
        self.fx = fx
        self.fy = fy
    
    def add_point_force(self, x: float, y: float, fx: float, fy: float, sigma: float = 0.05):
        """Add a Gaussian-distributed point force."""
        r_sq = (self.X - x)**2 + (self.Y - y)**2
        gaussian = np.exp(-r_sq / (2 * sigma**2))
        self.fx += fx * gaussian
        self.fy += fy * gaussian
    
    def _compute_advection(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advection terms: (u·∇)u and (u·∇)v."""
        periodic = self.is_periodic
        adv_u = AdvectionSchemes.advect(u, u, v, self.dx, self.dy, self.advection_scheme, periodic=periodic)
        adv_v = AdvectionSchemes.advect(v, u, v, self.dx, self.dy, self.advection_scheme, periodic=periodic)
        return adv_u, adv_v
    
    def _compute_diffusion(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute diffusion terms: ν_total * ∇²u."""
        # Get total viscosity (molecular + turbulent)
        nu_total = self.turb_model.get_total_viscosity(u, v, self.nu)
        periodic = self.is_periodic
        
        if isinstance(nu_total, np.ndarray):
            # Variable viscosity: ∇·(ν∇u)
            dudx = np.gradient(u, self.dx, axis=1)
            dudy = np.gradient(u, self.dy, axis=0)
            dvdx = np.gradient(v, self.dx, axis=1)
            dvdy = np.gradient(v, self.dy, axis=0)
            
            diff_u = (np.gradient(nu_total * dudx, self.dx, axis=1) +
                      np.gradient(nu_total * dudy, self.dy, axis=0))
            diff_v = (np.gradient(nu_total * dvdx, self.dx, axis=1) +
                      np.gradient(nu_total * dvdy, self.dy, axis=0))
        else:
            # Constant viscosity — use periodic Laplacian when appropriate
            if periodic:
                diff_u = nu_total * DiffusionSchemes.laplacian_periodic(u, self.dx, self.dy)
                diff_v = nu_total * DiffusionSchemes.laplacian_periodic(v, self.dx, self.dy)
            else:
                diff_u = nu_total * DiffusionSchemes.laplacian_2nd(u, self.dx, self.dy)
                diff_v = nu_total * DiffusionSchemes.laplacian_2nd(v, self.dx, self.dy)
        
        return diff_u, diff_v
    
    def _predict_velocity(self):
        """
        Step 1: Velocity prediction (explicit Euler).
        
        u* = u^n + Δt * [-(u·∇)u + ν∇²u + f]
        """
        adv_u, adv_v = self._compute_advection(self.u, self.v)
        diff_u, diff_v = self._compute_diffusion(self.u, self.v)
        
        self.u_star = self.u + self.dt * (-adv_u + diff_u + self.fx)
        self.v_star = self.v + self.dt * (-adv_v + diff_v + self.fy)
    
    def _solve_pressure(self):
        """
        Step 2: Solve pressure Poisson equation.
        
        ∇²p = (ρ/Δt) * ∇·u*
        """
        periodic = self.is_periodic
        div_ustar = GradientOperators.divergence(
            self.u_star, self.v_star, self.dx, self.dy, periodic=periodic
        )
        rhs = (self.density / self.dt) * div_ustar
        
        if self.pressure_solver_type == "fft":
            self.p = self.psolver.solve(rhs)
        else:
            bc_type = "periodic" if periodic else "neumann"
            self.p = self.psolver.solve(rhs, p_init=self.p, bc_type=bc_type)
    
    def _correct_velocity(self):
        """
        Step 3: Velocity correction (pressure gradient).
        
        u^{n+1} = u* - (Δt/ρ) * ∇p
        """
        periodic = self.is_periodic
        dpdx, dpdy = GradientOperators.gradient(self.p, self.dx, self.dy, periodic=periodic)
        
        self.u = self.u_star - (self.dt / self.density) * dpdx
        self.v = self.v_star - (self.dt / self.density) * dpdy
    
    def _apply_boundary_conditions(self):
        """Step 4: Apply all boundary conditions."""
        self.u, self.v = self.bc_manager.apply_velocity_bc(
            self.u, self.v, self.time
        )
        self.p = self.bc_manager.apply_pressure_bc(self.p)
        
        # Enforce zero velocity inside obstacles
        if np.any(self.obstacle):
            self.u[self.obstacle] = 0.0
            self.v[self.obstacle] = 0.0
    
    def _adapt_timestep(self, cfl_target: float = 0.5):
        """Compute adaptive time step based on CFL condition."""
        max_u = np.max(np.abs(self.u)) + 1e-10
        max_v = np.max(np.abs(self.v)) + 1e-10
        
        # Convective CFL
        dt_conv = cfl_target / (max_u/self.dx + max_v/self.dy)
        
        # Diffusive stability
        nu_eff = self.nu
        if hasattr(self.turb_model, 'nu_t') and self.turb_model.nu_t is not None:
            nu_eff += np.max(self.turb_model.nu_t)
        dt_diff = 0.25 * min(self.dx, self.dy)**2 / (nu_eff + 1e-10)
        
        self.dt = min(dt_conv, dt_diff)
    
    def step(self, adaptive_dt: bool = False):
        """
        Advance the solution by one time step.
        
        Chorin's projection method:
            1. Predict velocity
            2. Solve pressure
            3. Correct velocity
            4. Apply BCs
        """
        if adaptive_dt:
            self._adapt_timestep()
        
        # Update turbulence model if needed
        if hasattr(self.turb_model, 'update'):
            self.turb_model.update(self.u, self.v, self.nu, self.dt)
        
        # Projection method
        self._predict_velocity()
        self._solve_pressure()
        self._correct_velocity()
        self._apply_boundary_conditions()
        
        # Update state
        self.time += self.dt
        self.step_count += 1
    
    def advance(self, n_steps: int = 1, adaptive_dt: bool = False,
                record_history: bool = True):
        """Advance simulation by multiple time steps."""
        for _ in range(n_steps):
            self.step(adaptive_dt=adaptive_dt)
            
            if record_history:
                self._record_diagnostics()
    
    def _record_diagnostics(self):
        """Record diagnostic quantities."""
        from utils.helpers import compute_vorticity, compute_kinetic_energy, compute_enstrophy
        
        periodic = self.is_periodic
        omega = compute_vorticity(self.u, self.v, self.dx, self.dy)
        ke = compute_kinetic_energy(self.u, self.v)
        ens = compute_enstrophy(omega)
        div = np.max(np.abs(GradientOperators.divergence(self.u, self.v, self.dx, self.dy, periodic=periodic)))
        
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))
        cfl = self.dt * (max_u / self.dx + max_v / self.dy)
        
        self.history['time'].append(self.time)
        self.history['kinetic_energy'].append(ke)
        self.history['enstrophy'].append(ens)
        self.history['max_divergence'].append(div)
        self.history['cfl'].append(cfl)
        self.history['max_velocity'].append(max(max_u, max_v))
    
    def get_vorticity(self) -> np.ndarray:
        """Compute and return current vorticity field."""
        from utils.helpers import compute_vorticity
        return compute_vorticity(self.u, self.v, self.dx, self.dy)
    
    def get_velocity_magnitude(self) -> np.ndarray:
        """Compute velocity magnitude."""
        return np.sqrt(self.u**2 + self.v**2)
    
    def get_streamfunction(self) -> np.ndarray:
        """
        Compute stream function ψ from velocity field.
        
        u = ∂ψ/∂y, v = -∂ψ/∂x → ∇²ψ = -ω
        """
        omega = self.get_vorticity()
        
        # Solve Poisson equation for stream function
        if self.pressure_solver_type == "fft":
            psi_solver = VectorizedPressureSolver(self.nx, self.ny, self.dx, self.dy)
            return psi_solver.solve(-omega)
        else:
            psi_solver = PressureSolver(self.nx, self.ny, self.dx, self.dy, method="cg")
            return psi_solver.solve(-omega)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current simulation state as a dictionary."""
        return {
            'u': self.u.copy(),
            'v': self.v.copy(),
            'p': self.p.copy(),
            'vorticity': self.get_vorticity(),
            'velocity_magnitude': self.get_velocity_magnitude(),
            'obstacle': self.obstacle.copy(),
            'time': self.time,
            'step': self.step_count,
        }
    
    def reset(self):
        """Reset all fields to zero."""
        self.u[:] = 0
        self.v[:] = 0
        self.p[:] = 0
        self.fx[:] = 0
        self.fy[:] = 0
        self.obstacle[:] = False
        self.time = 0.0
        self.step_count = 0
        self.history = {k: [] for k in self.history}


class SIMPLESolver2D:
    """
    SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) solver.
    
    Alternative to Chorin's projection for steady-state problems.
    Iterative pressure-velocity coupling.
    
    Algorithm:
        1. Solve momentum with guessed pressure p*
        2. Compute pressure correction p'
        3. Correct velocity: u = u* + u'
        4. Correct pressure: p = p* + α_p * p'
        5. Iterate until convergence
    """
    
    def __init__(
        self,
        nx: int, ny: int,
        Lx: float, Ly: float,
        nu: float, density: float = 1.0
    ):
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.nu = nu
        self.density = density
        
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        # Under-relaxation factors
        self.alpha_u = 0.7  # Velocity
        self.alpha_p = 0.3  # Pressure
        
        self.psolver = PressureSolver(nx, ny, self.dx, self.dy, method="sor")
        self.converged = False
        self.iteration = 0
    
    def iterate(self, max_iter: int = 1000, tol: float = 1e-5):
        """Run SIMPLE iterations until convergence."""
        for self.iteration in range(max_iter):
            u_old = self.u.copy()
            v_old = self.v.copy()
            
            # Step 1: Solve momentum equations with current pressure
            lap_u = DiffusionSchemes.laplacian_2nd(self.u, self.dx, self.dy)
            lap_v = DiffusionSchemes.laplacian_2nd(self.v, self.dx, self.dy)
            
            dpdx, dpdy = GradientOperators.gradient(self.p, self.dx, self.dy)
            
            u_star = self.u + self.alpha_u * (
                self.nu * lap_u - 
                self.u * np.gradient(self.u, self.dx, axis=1) -
                self.v * np.gradient(self.u, self.dy, axis=0) -
                dpdx / self.density
            )
            v_star = self.v + self.alpha_u * (
                self.nu * lap_v -
                self.u * np.gradient(self.v, self.dx, axis=1) -
                self.v * np.gradient(self.v, self.dy, axis=0) -
                dpdy / self.density
            )
            
            # Step 2: Pressure correction
            div = GradientOperators.divergence(u_star, v_star, self.dx, self.dy)
            p_prime = self.psolver.solve(-self.density * div, p_init=np.zeros_like(self.p))
            
            # Step 3: Correct velocity
            dp_dx, dp_dy = GradientOperators.gradient(p_prime, self.dx, self.dy)
            self.u = u_star - dp_dx / self.density
            self.v = v_star - dp_dy / self.density
            
            # Step 4: Correct pressure
            self.p += self.alpha_p * p_prime
            
            # Check convergence
            residual = np.max(np.abs(self.u - u_old)) + np.max(np.abs(self.v - v_old))
            
            if residual < tol:
                self.converged = True
                print(f"SIMPLE converged in {self.iteration + 1} iterations (residual={residual:.2e})")
                return
        
        print(f"SIMPLE did not converge after {max_iter} iterations (residual={residual:.2e})")
