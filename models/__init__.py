"""Deep Learning Models for Navier-Stokes."""
from .pinn import PINN
from .fno import FNO2d
from .deeponet import DeepONet
from .surrogate import UNetSurrogate
from .turbulence_nn import TurbulenceClosureNN

__all__ = ['PINN', 'FNO2d', 'DeepONet', 'UNetSurrogate', 'TurbulenceClosureNN']
