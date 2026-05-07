"""Cross-Domain Physics Extensions for Navier-Stokes."""
from .mhd import MHDSolver
from .astrophysics import AstrophysicalFlowSolver
from .biophysics import BiophysicsFlowSolver
from .climate import ClimateFlowSolver
from .quantum_fluids import QuantumFluidSolver
from .relativistic import RelativisticNSSolver
from .qft_lattice import LatticeQFTSolver
from .gravity_fluid_coupling import GravityFluidSolver
from .cosmology import CosmologicalFluidSolver

__all__ = [
    'MHDSolver', 'AstrophysicalFlowSolver', 'BiophysicsFlowSolver',
    'ClimateFlowSolver', 'QuantumFluidSolver', 'RelativisticNSSolver',
    'LatticeQFTSolver', 'GravityFluidSolver', 'CosmologicalFluidSolver',
]

