"""Real-Time Visualization for Navier-Stokes simulations."""
from .realtime_2d import RealtimeVisualizer2D
from .realtime_3d import RealtimeVisualizer3D
from .renderer import FlowRenderer

__all__ = ['RealtimeVisualizer2D', 'RealtimeVisualizer3D', 'FlowRenderer']
