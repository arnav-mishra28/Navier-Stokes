"""Smoke Test Suite for Navier-Stokes ML/DL Hybrid System"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force headless matplotlib
import matplotlib
matplotlib.use('Agg')


# Core Solver Tests

class TestCoreSolver2D:
    """Test 2D incompressible NS solver."""
    
    def test_taylor_green_initialization(self):
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(nx=32, ny=32, Lx=2*np.pi, Ly=2*np.pi, nu=0.01, dt=0.01)
        solver.initialize_taylor_green(amplitude=1.0)
        solver.bc_manager.set_periodic()
        assert np.max(np.abs(solver.u)) > 0, "Velocity should be nonzero after init"
        assert solver.u.shape == (32, 32)
    
    def test_taylor_green_step(self):
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(nx=32, ny=32, Lx=2*np.pi, Ly=2*np.pi, nu=0.01, dt=0.01)
        solver.initialize_taylor_green()
        solver.bc_manager.set_periodic()
        for _ in range(10):
            solver.step()
        assert solver.time == pytest.approx(0.1, abs=1e-10)
        assert solver.step_count == 10
    
    def test_taylor_green_energy_decay(self):
        """Verify KE decays exponentially (analytical benchmark)."""
        from core.fluid_solver_2d import FluidSolver2D
        from utils.helpers import compute_kinetic_energy
        nu = 0.01
        solver = FluidSolver2D(nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi, nu=nu, dt=0.01)
        solver.initialize_taylor_green(1.0)
        solver.bc_manager.set_periodic()
        
        for _ in range(100):
            solver.step()
        ke = compute_kinetic_energy(solver.u, solver.v)
        ke_exact = 0.25 * np.exp(-4 * nu * solver.time)
        rel_error = abs(ke - ke_exact) / ke_exact
        assert rel_error < 0.06, f"KE error {rel_error:.4f} too large (should be <6%)"
    
    def test_double_shear_layer(self):
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi, nu=0.005, dt=0.01)
        solver.initialize_double_shear_layer()
        solver.bc_manager.set_periodic()
        for _ in range(10):
            solver.step()
        assert np.isfinite(solver.u).all(), "Solution blew up"
    
    def test_pressure_solvers(self):
        from core.fluid_solver_2d import FluidSolver2D
        for method in ["fft", "jacobi", "sor", "cg"]:
            solver = FluidSolver2D(
                nx=32, ny=32, Lx=2*np.pi, Ly=2*np.pi,
                nu=0.01, dt=0.01, pressure_solver=method
            )
            solver.initialize_taylor_green()
            solver.bc_manager.set_periodic()
            solver.step()
            assert np.isfinite(solver.p).all(), f"Pressure solver '{method}' produced NaN"
    
    def test_obstacle(self):
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(nx=64, ny=64, Lx=1.0, Ly=1.0, nu=0.01, dt=0.005)
        solver.initialize_taylor_green()
        solver.bc_manager.set_periodic()
        solver.add_circular_obstacle(0.5, 0.5, 0.1)
        solver.step()
        assert np.all(solver.u[solver.obstacle] == 0.0)
    
    def test_get_state(self):
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(nx=32, ny=32, nu=0.01, dt=0.01)
        solver.initialize_taylor_green()
        state = solver.get_state()
        assert 'u' in state and 'v' in state and 'p' in state


class TestCoreSolver3D:
    """Test 3D incompressible NS solver."""
    
    def test_taylor_green_3d(self):
        from core.fluid_solver_3d import FluidSolver3D
        solver = FluidSolver3D(nx=16, ny=16, nz=16, nu=0.01, dt=0.01)
        solver.initialize_taylor_green_3d()
        for _ in range(5):
            solver.step()
        assert solver.step_count == 5
        assert np.isfinite(solver.u).all()
    
    def test_abc_flow(self):
        from core.fluid_solver_3d import FluidSolver3D
        solver = FluidSolver3D(nx=16, ny=16, nz=16, nu=0.01, dt=0.01)
        solver.initialize_abc_flow()
        solver.step()
        assert np.isfinite(solver.u).all()
    
    def test_energy_spectrum(self):
        from core.fluid_solver_3d import FluidSolver3D
        solver = FluidSolver3D(nx=16, ny=16, nz=16, nu=0.01, dt=0.01)
        solver.initialize_taylor_green_3d()
        solver.advance(5)
        k, E = solver.compute_energy_spectrum()
        assert len(k) > 0 and len(E) > 0


# Physics Domain Tests

class TestPhysicsDomains:
    """Test all cross-physics solvers."""
    
    def test_mhd(self):
        from physics.mhd import MHDSolver
        solver = MHDSolver(nx=32, ny=32, nu=0.005, eta=0.005, dt=0.005)
        solver.initialize_orszag_tang()
        solver.advance(5)
        assert np.isfinite(solver.u).all()
    
    def test_astrophysics(self):
        from physics.astrophysics import AstrophysicalFlowSolver
        solver = AstrophysicalFlowSolver(nx=32, ny=32, nu=0.01, dt=0.005)
        solver.initialize_rayleigh_taylor()
        solver.advance(5)
        state = solver.get_state()
        assert 'rho' in state
        assert np.isfinite(state['rho']).all()
    
    def test_biophysics(self):
        from physics.biophysics import BiophysicsFlowSolver
        solver = BiophysicsFlowSolver(nx=50, ny=20, dt=0.0005)
        solver.initialize_straight_vessel(stenosis=0.5)
        solver.advance(5)
        assert np.isfinite(solver.u).all()
    
    def test_climate(self):
        from physics.climate import ClimateFlowSolver
        solver = ClimateFlowSolver(nx=32, ny=32, nu=500, dt=500)
        solver.initialize_kelvin_helmholtz()
        solver.advance(5)
        state = solver.get_state()
        assert 'T' in state
    
    def test_quantum(self):
        from physics.quantum_fluids import QuantumFluidSolver
        solver = QuantumFluidSolver(nx=32, ny=32, g_int=500, dt=0.0005)
        solver.initialize_quantum_turbulence(n_vortices=3)
        solver.advance(5)
        assert np.isfinite(solver.get_density()).all()


# ML Model Tests

class TestMLModels:
    """Test all neural network model architectures."""
    
    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch")
    
    def test_pinn_forward(self):
        import torch
        from models.pinn import PINN
        model = PINN(hidden_layers=[32, 32], activation='tanh')
        inp = torch.randn(10, 3)
        out = model(inp)
        assert out.shape == (10, 3)
    
    def test_fno_forward(self):
        import torch
        from models.fno import FNO2d
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2,
                      in_channels=3, out_channels=3)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 3, 32, 32)
    
    def test_deeponet_forward(self):
        import torch
        from models.deeponet import DeepONet
        model = DeepONet(
            branch_input_dim=64, trunk_input_dim=2,
            latent_dim=32, n_outputs=3,
        )
        branch = torch.randn(8, 64)
        trunk = torch.randn(8, 5, 2)  # (batch, n_points, trunk_dim)
        out = model(branch, trunk)
        assert out.shape == (8, 5, 3)
    
    def test_surrogate_forward(self):
        import torch
        from models.surrogate import UNetSurrogate
        model = UNetSurrogate(in_channels=3, out_channels=3)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)
        assert y.shape == (2, 3, 64, 64)
    
    def test_turbulence_nn_forward(self):
        import torch
        from models.turbulence_nn import TurbulenceClosureNN
        model = TurbulenceClosureNN()
        x = torch.randn(4, 9)  # 9 inputs (velocity gradients)
        y = model(x)
        assert y.shape[0] == 4


# Training Pipeline Tests

class TestTrainingPipeline:
    """Test training infrastructure."""
    
    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch")
    
    def test_data_generator_pinn(self):
        from training.data_generator import NSDataGenerator
        gen = NSDataGenerator(nx=16, ny=16)
        data = gen.generate_pinn_collocation_points(
            n_collocation=50, n_boundary=20, n_initial=20
        )
        assert 'x_pde' in data and 'y_pde' in data
        assert len(data['x_pde']) == 50
    
    def test_data_generator_fno(self):
        from training.data_generator import NSDataGenerator
        gen = NSDataGenerator(nx=16, ny=16, n_timesteps=5, dt=0.01)
        data = gen.generate_taylor_green_dataset(n_samples=2)
        assert 'inputs' in data and 'outputs' in data
        assert data['inputs'].shape[1] == 3  # (u, v, p) channels
    
    def test_trainer_fno_short(self):
        """Verify FNO trainer runs without crashing for very short epochs."""
        import torch
        from models.fno import FNO2d
        from training.data_generator import NSDataGenerator, create_dataloaders
        from training.trainer import UnifiedTrainer
        
        gen = NSDataGenerator(nx=16, ny=16, n_timesteps=3, dt=0.01)
        data = gen.generate_taylor_green_dataset(n_samples=2)
        train_ldr, val_ldr, _ = create_dataloaders(data, batch_size=8)
        
        model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2,
                      in_channels=3, out_channels=3)
        trainer = UnifiedTrainer(model, model_type='fno', device='cpu', learning_rate=1e-3)
        trainer.train_fno(train_ldr, val_ldr, epochs=3)
        assert trainer.epoch == 3
    
    def test_physics_informed_loss(self):
        import torch
        from training.losses import PhysicsInformedLoss
        loss = PhysicsInformedLoss(nu=0.01, adaptive=False)
        assert loss is not None
    
    def test_fno_loss(self):
        import torch
        from training.losses import FNOLoss
        loss_fn = FNOLoss()
        pred = torch.randn(2, 3, 16, 16)
        tgt = torch.randn(2, 3, 16, 16)
        result = loss_fn(pred, tgt)
        assert 'total' in result
        assert result['total'].item() > 0


# Utility Tests

class TestUtilities:
    """Test helper functions."""
    
    def test_compute_vorticity(self):
        from utils.helpers import compute_vorticity
        u = np.random.randn(32, 32)
        v = np.random.randn(32, 32)
        omega = compute_vorticity(u, v, 0.1, 0.1)
        assert omega.shape == (32, 32)
    
    def test_compute_kinetic_energy(self):
        from utils.helpers import compute_kinetic_energy
        u = np.ones((32, 32))
        v = np.zeros((32, 32))
        ke = compute_kinetic_energy(u, v)
        assert ke == pytest.approx(0.5, abs=1e-6)
    
    def test_compute_enstrophy(self):
        from utils.helpers import compute_enstrophy
        omega = np.ones((32, 32))
        ens = compute_enstrophy(omega)
        assert ens == pytest.approx(0.5, abs=1e-6)
    
    def test_create_obstacle_mask(self):
        from utils.helpers import create_obstacle_mask
        mask = create_obstacle_mask(64, 64, "cylinder", center=(0.5, 0.5), radius=0.1)
        assert mask.shape == (64, 64)
        assert np.any(mask)  # Some cells should be solid
    
    def test_timer(self):
        from utils.helpers import Timer
        import time as _time
        with Timer("test") as t:
            _time.sleep(0.01)
        assert t.elapsed > 0


# Visualization Tests

class TestVisualization:
    """Test rendering utilities."""
    
    def test_scalar_to_rgb(self):
        from visualization.renderer import FlowRenderer
        field = np.random.randn(32, 32)
        rgb = FlowRenderer.scalar_to_rgb(field, cmap='inferno')
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
    
    def test_velocity_to_rgb(self):
        from visualization.renderer import FlowRenderer
        u = np.random.randn(32, 32)
        v = np.random.randn(32, 32)
        rgb = FlowRenderer.velocity_to_rgb(u, v)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
    
    def test_streamlines(self):
        from visualization.renderer import FlowRenderer
        u = np.ones((32, 32)) * 0.5
        v = np.ones((32, 32)) * 0.3
        lines = FlowRenderer.draw_streamlines(u, v, 32, 32, n_lines=5, n_steps=20)
        assert isinstance(lines, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
