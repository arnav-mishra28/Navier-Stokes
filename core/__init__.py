"""Core CFD Engine - Classical numerical solvers for Navier-Stokes equations."""
from .fluid_solver_2d import FluidSolver2D
from .fluid_solver_3d import FluidSolver3D
from .pressure_solver import PressureSolver
from .boundary_conditions import BoundaryConditionManager
from .turbulence_models import TurbulenceModelFactory
from .discretization import AdvectionSchemes, DiffusionSchemes

__all__ = [
    'FluidSolver2D', 'FluidSolver3D', 'PressureSolver',
    'BoundaryConditionManager', 'TurbulenceModelFactory',
    'AdvectionSchemes', 'DiffusionSchemes'
]
