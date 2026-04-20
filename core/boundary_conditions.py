"""
=============================================================================
Boundary Condition Manager
Handles all types of boundary conditions for the Navier-Stokes solver.
=============================================================================
"""

import numpy as np
from typing import Optional, Callable, Dict, Tuple
from enum import Enum


class BCType(Enum):
    NO_SLIP = "no_slip"
    FREE_SLIP = "free_slip"
    PERIODIC = "periodic"
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PRESSURE_OUTLET = "pressure_outlet"
    SYMMETRY = "symmetry"


class BoundaryConditionManager:
    """
    Manages and applies boundary conditions for velocity and pressure fields.
    
    Supports:
        - No-slip (u = 0 at wall)
        - Free-slip (u_n = 0, ∂u_t/∂n = 0)
        - Periodic
        - Inflow (specified velocity profile)
        - Outflow (zero gradient)
        - Pressure outlet
        - Symmetry
        - Custom (user-defined function)
    """
    
    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Boundary types for each wall
        self.bc_types: Dict[str, BCType] = {
            'top': BCType.NO_SLIP,
            'bottom': BCType.NO_SLIP,
            'left': BCType.NO_SLIP,
            'right': BCType.NO_SLIP,
        }
        
        # Boundary values for Dirichlet / Inflow conditions
        self.bc_values: Dict[str, Dict[str, float]] = {
            'top': {'u': 0.0, 'v': 0.0, 'p': 0.0},
            'bottom': {'u': 0.0, 'v': 0.0, 'p': 0.0},
            'left': {'u': 0.0, 'v': 0.0, 'p': 0.0},
            'right': {'u': 0.0, 'v': 0.0, 'p': 0.0},
        }
        
        # Custom BC functions
        self.custom_bc: Dict[str, Optional[Callable]] = {
            'top': None, 'bottom': None, 'left': None, 'right': None
        }
        
        # Obstacle mask
        self.obstacle_mask: Optional[np.ndarray] = None
    
    def set_bc(self, wall: str, bc_type: BCType, values: Optional[Dict] = None,
               custom_func: Optional[Callable] = None):
        """
        Set boundary condition for a specific wall.
        
        Args:
            wall: 'top', 'bottom', 'left', 'right'
            bc_type: Type of boundary condition
            values: Dict with 'u', 'v', 'p' values for Dirichlet/Inflow
            custom_func: Custom function(u, v, p, t) -> (u, v, p) for custom BCs
        """
        self.bc_types[wall] = bc_type
        if values:
            self.bc_values[wall].update(values)
        if custom_func:
            self.custom_bc[wall] = custom_func
    
    def set_lid_driven_cavity(self, u_lid: float = 1.0):
        """Configure standard lid-driven cavity benchmark."""
        self.set_bc('top', BCType.DIRICHLET, {'u': u_lid, 'v': 0.0})
        self.set_bc('bottom', BCType.NO_SLIP)
        self.set_bc('left', BCType.NO_SLIP)
        self.set_bc('right', BCType.NO_SLIP)
    
    def set_channel_flow(self, u_inlet: float = 1.0):
        """Configure channel flow with inlet/outlet."""
        self.set_bc('top', BCType.NO_SLIP)
        self.set_bc('bottom', BCType.NO_SLIP)
        self.set_bc('left', BCType.INFLOW, {'u': u_inlet, 'v': 0.0})
        self.set_bc('right', BCType.OUTFLOW)
    
    def set_periodic(self):
        """Set all boundaries to periodic."""
        for wall in ['top', 'bottom', 'left', 'right']:
            self.set_bc(wall, BCType.PERIODIC)
    
    def set_obstacle(self, mask: np.ndarray):
        """Set solid obstacle mask (True = solid)."""
        self.obstacle_mask = mask.astype(bool)
    
    def apply_velocity_bc(
        self, u: np.ndarray, v: np.ndarray,
        t: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply velocity boundary conditions.
        
        Args:
            u, v: Velocity components (ny, nx)
            t: Current simulation time
        
        Returns:
            u, v: Modified velocity fields
        """
        # ---- TOP boundary (j = ny-1) ----
        bc = self.bc_types['top']
        if bc == BCType.NO_SLIP:
            u[-1, :] = 0.0
            v[-1, :] = 0.0
        elif bc == BCType.DIRICHLET:
            u[-1, :] = self.bc_values['top']['u']
            v[-1, :] = self.bc_values['top']['v']
        elif bc == BCType.FREE_SLIP:
            u[-1, :] = u[-2, :]  # ∂u/∂n = 0
            v[-1, :] = 0.0       # u_n = 0
        elif bc == BCType.SYMMETRY:
            u[-1, :] = u[-2, :]
            v[-1, :] = -v[-2, :]
        elif bc == BCType.PERIODIC:
            u[-1, :] = u[1, :]
            v[-1, :] = v[1, :]
        
        # ---- BOTTOM boundary (j = 0) ----
        bc = self.bc_types['bottom']
        if bc == BCType.NO_SLIP:
            u[0, :] = 0.0
            v[0, :] = 0.0
        elif bc == BCType.DIRICHLET:
            u[0, :] = self.bc_values['bottom']['u']
            v[0, :] = self.bc_values['bottom']['v']
        elif bc == BCType.FREE_SLIP:
            u[0, :] = u[1, :]
            v[0, :] = 0.0
        elif bc == BCType.SYMMETRY:
            u[0, :] = u[1, :]
            v[0, :] = -v[1, :]
        elif bc == BCType.PERIODIC:
            u[0, :] = u[-2, :]
            v[0, :] = v[-2, :]
        
        # ---- LEFT boundary (i = 0) ----
        bc = self.bc_types['left']
        if bc == BCType.NO_SLIP:
            u[:, 0] = 0.0
            v[:, 0] = 0.0
        elif bc == BCType.INFLOW:
            u_in = self.bc_values['left']['u']
            v_in = self.bc_values['left']['v']
            # Parabolic profile for channel flow
            y = np.linspace(0, 1, self.ny)
            u[:, 0] = 4 * u_in * y * (1 - y)  # Parabolic
            v[:, 0] = v_in
        elif bc == BCType.DIRICHLET:
            u[:, 0] = self.bc_values['left']['u']
            v[:, 0] = self.bc_values['left']['v']
        elif bc == BCType.PERIODIC:
            u[:, 0] = u[:, -2]
            v[:, 0] = v[:, -2]
        
        # ---- RIGHT boundary (i = nx-1) ----
        bc = self.bc_types['right']
        if bc == BCType.NO_SLIP:
            u[:, -1] = 0.0
            v[:, -1] = 0.0
        elif bc == BCType.OUTFLOW:
            u[:, -1] = u[:, -2]  # Zero gradient
            v[:, -1] = v[:, -2]
        elif bc == BCType.PRESSURE_OUTLET:
            u[:, -1] = u[:, -2]
            v[:, -1] = v[:, -2]
        elif bc == BCType.PERIODIC:
            u[:, -1] = u[:, 1]
            v[:, -1] = v[:, 1]
        
        # ---- Custom BCs ----
        for wall, func in self.custom_bc.items():
            if func is not None:
                u, v = func(u, v, t, wall)
        
        # ---- Obstacle enforcement ----
        if self.obstacle_mask is not None:
            u[self.obstacle_mask] = 0.0
            v[self.obstacle_mask] = 0.0
        
        return u, v
    
    def apply_pressure_bc(self, p: np.ndarray) -> np.ndarray:
        """Apply pressure boundary conditions."""
        # Top
        bc = self.bc_types['top']
        if bc in [BCType.NO_SLIP, BCType.FREE_SLIP, BCType.DIRICHLET, BCType.SYMMETRY]:
            p[-1, :] = p[-2, :]  # Neumann
        elif bc == BCType.PERIODIC:
            p[-1, :] = p[1, :]
        
        # Bottom
        bc = self.bc_types['bottom']
        if bc in [BCType.NO_SLIP, BCType.FREE_SLIP, BCType.DIRICHLET, BCType.SYMMETRY]:
            p[0, :] = p[1, :]
        elif bc == BCType.PERIODIC:
            p[0, :] = p[-2, :]
        
        # Left
        bc = self.bc_types['left']
        if bc in [BCType.NO_SLIP, BCType.INFLOW, BCType.DIRICHLET]:
            p[:, 0] = p[:, 1]
        elif bc == BCType.PERIODIC:
            p[:, 0] = p[:, -2]
        
        # Right
        bc = self.bc_types['right']
        if bc in [BCType.NO_SLIP, BCType.OUTFLOW]:
            p[:, -1] = p[:, -2]
        elif bc == BCType.PRESSURE_OUTLET:
            p[:, -1] = 0.0  # Reference pressure
        elif bc == BCType.PERIODIC:
            p[:, -1] = p[:, 1]
        
        return p
    
    def get_inflow_profile(
        self, profile_type: str = "parabolic",
        u_max: float = 1.0,
        t: float = 0.0
    ) -> np.ndarray:
        """
        Generate inflow velocity profile.
        
        Args:
            profile_type: "uniform", "parabolic", "womersley", "turbulent"
            u_max: Maximum velocity
            t: Time (for pulsatile flows)
        """
        y = np.linspace(0, 1, self.ny)
        
        if profile_type == "uniform":
            return np.full(self.ny, u_max)
        
        elif profile_type == "parabolic":
            # Poiseuille profile: u(y) = 4*u_max*y*(1-y)
            return 4 * u_max * y * (1 - y)
        
        elif profile_type == "womersley":
            # Pulsatile flow: Womersley profile (biophysics application)
            omega = 2 * np.pi  # Heart rate ~1 Hz
            steady = 4 * u_max * y * (1 - y)
            pulsatile = 0.3 * u_max * np.sin(omega * t) * (1 - (2*y - 1)**2)
            return steady + pulsatile
        
        elif profile_type == "turbulent":
            # 1/7th power law
            y_norm = y.copy()
            y_norm[y_norm < 0.5] = y_norm[y_norm < 0.5] / 0.5
            y_norm[y_norm >= 0.5] = (1 - y_norm[y_norm >= 0.5]) / 0.5
            return u_max * y_norm ** (1/7)
        
        else:
            return np.full(self.ny, u_max)
