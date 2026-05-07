"""Deep Learning Models for Navier-Stokes."""
from .pinn import PINN
from .fno import FNO2d
from .deeponet import DeepONet
from .surrogate import UNetSurrogate
from .turbulence_nn import TurbulenceClosureNN
from .autoencoder import FlowAutoencoder, LatentDynamicsODE, TurbulenceDiscoveryAutoencoder
from .symbolic_discovery import SINDy, GeneticProgramming, SymbolicDiscoveryEngine
from .regularity_analysis import BlowupDetector, StabilityAnalyzer, FlowDiagnostics, TurbulenceMetrics
from .physics_discovery import PhysicsAwareSINDy, PhysicsAwareLibrary, ConservationValidator, CorrectionTermDiscovery
from .hypothesis_engine import HypothesisGenerator, SymbolicReasoner, ExperimentValidator, KnowledgeBase

__all__ = [
    'PINN', 'FNO2d', 'DeepONet', 'UNetSurrogate', 'TurbulenceClosureNN',
    'FlowAutoencoder', 'LatentDynamicsODE', 'TurbulenceDiscoveryAutoencoder',
    'SINDy', 'GeneticProgramming', 'SymbolicDiscoveryEngine',
    'BlowupDetector', 'StabilityAnalyzer', 'FlowDiagnostics', 'TurbulenceMetrics',
    'PhysicsAwareSINDy', 'PhysicsAwareLibrary', 'ConservationValidator', 'CorrectionTermDiscovery',
    'HypothesisGenerator', 'SymbolicReasoner', 'ExperimentValidator', 'KnowledgeBase',
]
