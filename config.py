"""Global Configuration & Hyperparameters"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class SolverType(Enum):
    """Available solver types."""
    PROJECTION = "projection"
    SIMPLE = "simple"
    FRACTIONAL_STEP = "fractional_step"


class TurbulenceModel(Enum):
    """Available turbulence models."""
    NONE = "none"
    SMAGORINSKY = "smagorinsky"
    DYNAMIC_SMAG = "dynamic_smagorinsky"
    K_EPSILON = "k_epsilon"
    K_OMEGA = "k_omega"
    DNS = "dns"


class MLModel(Enum):
    """Available ML model types."""
    PINN = "pinn"
    FNO = "fno"
    DEEPONET = "deeponet"
    SURROGATE = "surrogate"
    TURBULENCE_NN = "turbulence_nn"
    AUTOENCODER = "autoencoder"
    LATENT_ODE = "latent_ode"
    BLOWUP_DETECTOR = "blowup_detector"


class DiscoveryMode(Enum):
    """Turbulence discovery pipeline modes."""
    FULL_PIPELINE = "full"
    AUTOENCODER_ONLY = "autoencoder"
    SYMBOLIC_ONLY = "symbolic"
    STABILITY_ANALYSIS = "stability"
    REGULARITY_MAP = "regularity"


class PhysicsDomain(Enum):
    """Cross-physics domain types."""
    FLUID = "fluid"
    MHD = "magnetohydrodynamics"
    ASTROPHYSICS = "astrophysics"
    BIOPHYSICS = "biophysics"
    CLIMATE = "climate"
    QUANTUM_FLUID = "quantum_fluid"
    RELATIVISTIC = "relativistic"


class BoundaryType(Enum):
    """Boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    NO_SLIP = "no_slip"
    FREE_SLIP = "free_slip"
    INFLOW = "inflow"
    OUTFLOW = "outflow"


@dataclass
class GridConfig:
    """Grid configuration for the simulation domain."""
    nx: int = 128
    ny: int = 128
    nz: int = 1  # 1 = 2D mode
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi
    Lz: float = 2.0 * np.pi
    
    @property
    def dx(self) -> float:
        return self.Lx / self.nx
    
    @property
    def dy(self) -> float:
        return self.Ly / self.ny
    
    @property
    def dz(self) -> float:
        return self.Lz / max(self.nz, 1)
    
    @property
    def is_3d(self) -> bool:
        return self.nz > 1
    
    @property
    def shape_2d(self) -> Tuple[int, int]:
        return (self.ny, self.nx)
    
    @property
    def shape_3d(self) -> Tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)


@dataclass
class FluidConfig:
    """Physical parameters for the fluid."""
    density: float = 1.0
    viscosity: float = 0.01  # kinematic viscosity (nu)
    dt: float = 0.001
    t_end: float = 10.0
    
    # External forces
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_amplitude: float = 0.0
    force_frequency: float = 1.0
    
    @property
    def reynolds_number(self) -> float:
        """Estimate Reynolds number based on domain and viscosity."""
        return 1.0 / max(self.viscosity, 1e-12)


@dataclass
class SolverConfig:
    """Solver configuration."""
    solver_type: SolverType = SolverType.PROJECTION
    turbulence_model: TurbulenceModel = TurbulenceModel.NONE
    
    # Pressure solver
    pressure_solver: str = "fft"  # "fft", "jacobi", "sor", "cg"
    pressure_max_iter: int = 500
    pressure_tol: float = 1e-6
    sor_omega: float = 1.7
    
    # Time stepping
    cfl_target: float = 0.5
    adaptive_dt: bool = True
    
    # Advection scheme
    advection_scheme: str = "central"  # "upwind", "central", "weno5"
    
    # Boundary conditions (top, bottom, left, right)
    bc_top: BoundaryType = BoundaryType.NO_SLIP
    bc_bottom: BoundaryType = BoundaryType.NO_SLIP
    bc_left: BoundaryType = BoundaryType.NO_SLIP
    bc_right: BoundaryType = BoundaryType.NO_SLIP


@dataclass
class PINNConfig:
    """PINN model configuration."""
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 128, 128, 128])
    activation: str = "tanh"
    learning_rate: float = 1e-3
    epochs: int = 10000
    batch_size: int = 4096
    
    # Loss weights
    w_data: float = 1.0
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 10.0
    
    # Collocation points
    n_collocation: int = 10000
    n_boundary: int = 2000
    n_initial: int = 2000
    
    # Adaptive weighting
    adaptive_weights: bool = True
    weight_method: str = "grad_norm"  # "grad_norm", "ntk", "softattention"


@dataclass
class FNOConfig:
    """Fourier Neural Operator configuration."""
    modes: int = 16
    width: int = 64
    n_layers: int = 4
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    
    # Architecture
    lifting_channels: int = 128
    projection_channels: int = 128
    padding: int = 8


@dataclass
class DeepONetConfig:
    """DeepONet configuration."""
    branch_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    trunk_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    latent_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    activation: str = "relu"


@dataclass
class SurrogateConfig:
    """U-Net surrogate model configuration."""
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 3
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 16
    
    # Skip connections
    skip_connections: bool = True
    residual_blocks: int = 2


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    # 2D settings
    screen_width: int = 1200
    screen_height: int = 800
    fps: int = 60
    colormap: str = "inferno"
    
    # 3D settings
    window_size_3d: Tuple[int, int] = (1400, 900)
    opacity: float = 0.6
    show_streamlines: bool = True
    show_vectors: bool = True
    show_pressure: bool = True
    
    # Rendering
    vector_scale: float = 0.05
    vector_density: int = 8
    contour_levels: int = 20


@dataclass
class TrainingConfig:
    """Training infrastructure configuration."""
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 50
    
    # Data generation
    n_train_samples: int = 1000
    n_val_samples: int = 200
    n_test_samples: int = 100
    
    # Mixed precision
    use_amp: bool = True
    gradient_clip: float = 1.0


@dataclass
class TurbulenceDiscoveryConfig:
    """Turbulence Discovery AI pipeline configuration."""
    # Autoencoder
    latent_dim: int = 64
    base_channels: int = 32
    variational: bool = False
    ae_epochs: int = 100
    ae_lr: float = 1e-3
    beta_vae: float = 0.001
    physics_weight: float = 0.1
    
    # Latent ODE
    ode_hidden_dim: int = 256
    ode_layers: int = 4
    ode_epochs: int = 100
    ode_lr: float = 1e-3
    
    # Blow-up Detection
    blowup_epochs: int = 50
    blowup_lr: float = 1e-3
    
    # Symbolic Discovery
    sindy_poly_order: int = 3
    sindy_threshold: float = 0.1
    sindy_include_trig: bool = True
    gp_population: int = 200
    gp_generations: int = 50
    gp_max_depth: int = 5
    gp_parsimony: float = 0.001
    
    # Data generation
    n_ae_samples: int = 200
    n_paired_samples: int = 100
    n_stability_samples: int = 100
    re_range_low: float = 10.0
    re_range_high: float = 10000.0
    
    # Discovery mode
    mode: str = "full"  # full, autoencoder, symbolic, stability


@dataclass
class MasterConfig:
    """Master configuration combining all sub-configs."""
    grid: GridConfig = field(default_factory=GridConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)
    deeponet: DeepONetConfig = field(default_factory=DeepONetConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    discovery: TurbulenceDiscoveryConfig = field(default_factory=TurbulenceDiscoveryConfig)
    
    # Active physics domain
    physics_domain: PhysicsDomain = PhysicsDomain.FLUID


# Preset Configurations

def lid_driven_cavity(re: float = 100) -> MasterConfig:
    """Lid-driven cavity flow benchmark."""
    cfg = MasterConfig()
    cfg.grid = GridConfig(nx=64, ny=64, Lx=1.0, Ly=1.0)
    cfg.fluid = FluidConfig(viscosity=1.0/re, dt=0.001, t_end=20.0)
    cfg.solver.bc_top = BoundaryType.DIRICHLET
    cfg.solver.bc_bottom = BoundaryType.NO_SLIP
    cfg.solver.bc_left = BoundaryType.NO_SLIP
    cfg.solver.bc_right = BoundaryType.NO_SLIP
    return cfg


def taylor_green_vortex(re: float = 100) -> MasterConfig:
    """Taylor-Green vortex decay benchmark."""
    cfg = MasterConfig()
    cfg.grid = GridConfig(nx=128, ny=128, Lx=2*np.pi, Ly=2*np.pi)
    cfg.fluid = FluidConfig(viscosity=1.0/re, dt=0.005, t_end=10.0)
    cfg.solver.bc_top = BoundaryType.PERIODIC
    cfg.solver.bc_bottom = BoundaryType.PERIODIC
    cfg.solver.bc_left = BoundaryType.PERIODIC
    cfg.solver.bc_right = BoundaryType.PERIODIC
    return cfg


def channel_flow(re: float = 1000) -> MasterConfig:
    """Turbulent channel flow."""
    cfg = MasterConfig()
    cfg.grid = GridConfig(nx=256, ny=64, Lx=4*np.pi, Ly=2.0)
    cfg.fluid = FluidConfig(viscosity=1.0/re, dt=0.001, t_end=50.0)
    cfg.solver.turbulence_model = TurbulenceModel.SMAGORINSKY
    cfg.solver.bc_top = BoundaryType.NO_SLIP
    cfg.solver.bc_bottom = BoundaryType.NO_SLIP
    cfg.solver.bc_left = BoundaryType.PERIODIC
    cfg.solver.bc_right = BoundaryType.PERIODIC
    return cfg


def flow_around_cylinder() -> MasterConfig:
    """Flow around a cylinder (von Kármán vortex street)."""
    cfg = MasterConfig()
    cfg.grid = GridConfig(nx=256, ny=128, Lx=10.0, Ly=5.0)
    cfg.fluid = FluidConfig(viscosity=0.005, dt=0.002, t_end=30.0)
    cfg.solver.bc_top = BoundaryType.FREE_SLIP
    cfg.solver.bc_bottom = BoundaryType.FREE_SLIP
    cfg.solver.bc_left = BoundaryType.INFLOW
    cfg.solver.bc_right = BoundaryType.OUTFLOW
    return cfg


def blood_flow_artery() -> MasterConfig:
    """Pulsatile blood flow in an artery."""
    cfg = MasterConfig()
    cfg.grid = GridConfig(nx=128, ny=64, Lx=0.1, Ly=0.01)  # 10cm x 1cm
    cfg.fluid = FluidConfig(
        density=1060.0, viscosity=3.5e-6,  # blood
        dt=0.0001, t_end=2.0
    )
    cfg.physics_domain = PhysicsDomain.BIOPHYSICS
    return cfg


def mhd_reconnection() -> MasterConfig:
    """Magnetic reconnection scenario."""
    cfg = MasterConfig()
    cfg.grid = GridConfig(nx=128, ny=128, Lx=4*np.pi, Ly=4*np.pi)
    cfg.fluid = FluidConfig(viscosity=0.01, dt=0.001, t_end=10.0)
    cfg.physics_domain = PhysicsDomain.MHD
    cfg.solver.bc_top = BoundaryType.PERIODIC
    cfg.solver.bc_bottom = BoundaryType.PERIODIC
    cfg.solver.bc_left = BoundaryType.PERIODIC
    cfg.solver.bc_right = BoundaryType.PERIODIC
    return cfg


# Dictionary of all presets
PRESETS = {
    "lid_driven_cavity": lid_driven_cavity,
    "taylor_green_vortex": taylor_green_vortex,
    "channel_flow": channel_flow,
    "flow_around_cylinder": flow_around_cylinder,
    "blood_flow_artery": blood_flow_artery,
    "mhd_reconnection": mhd_reconnection,
}
