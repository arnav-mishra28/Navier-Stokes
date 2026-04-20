"""
=============================================================================
Training Data Generator
Creates training datasets from CFD simulations for ML models.
=============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NSDataGenerator:
    """
    Generate training data from CFD simulations.
    
    Creates paired (input, output) datasets for:
        - FNO: (u_t, params) → u_{t+1}
        - PINN: collocation points + BC/IC data
        - DeepONet: (input_function, query_points) → solution
        - Surrogate: (conditions) → flow fields
    """
    
    def __init__(
        self,
        nx: int = 64, ny: int = 64,
        Lx: float = 2*np.pi, Ly: float = 2*np.pi,
        nu_range: Tuple[float, float] = (0.001, 0.1),
        n_timesteps: int = 50,
        dt: float = 0.01,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu_range = nu_range
        self.n_timesteps = n_timesteps
        self.dt = dt
    
    def generate_taylor_green_dataset(
        self, n_samples: int = 100, save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate Taylor-Green vortex dataset at various Reynolds numbers.
        
        Returns:
            Dict with 'inputs', 'outputs', 'params', 'times'
        """
        from core.fluid_solver_2d import FluidSolver2D
        
        inputs_list = []
        outputs_list = []
        params_list = []
        
        for i in range(n_samples):
            nu = np.random.uniform(*self.nu_range)
            amplitude = np.random.uniform(0.5, 2.0)
            
            solver = FluidSolver2D(
                nx=self.nx, ny=self.ny,
                Lx=self.Lx, Ly=self.Ly,
                nu=nu, dt=self.dt,
                pressure_solver="fft"
            )
            solver.initialize_taylor_green(amplitude=amplitude)
            solver.bc_manager.set_periodic()
            
            # Record trajectory
            for t in range(self.n_timesteps):
                state_before = np.stack([solver.u.copy(), solver.v.copy(), solver.p.copy()])
                solver.step()
                state_after = np.stack([solver.u.copy(), solver.v.copy(), solver.p.copy()])
                
                inputs_list.append(state_before)
                outputs_list.append(state_after)
                params_list.append([nu, amplitude, solver.time])
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{n_samples} Taylor-Green samples")
        
        data = {
            'inputs': np.array(inputs_list),    # (N, 3, ny, nx)
            'outputs': np.array(outputs_list),   # (N, 3, ny, nx)
            'params': np.array(params_list),     # (N, 3)
        }
        
        if save_path:
            np.savez_compressed(save_path, **data)
            print(f"Saved dataset to {save_path} ({data['inputs'].shape[0]} samples)")
        
        return data
    
    def generate_varied_flow_dataset(
        self, n_samples: int = 200, save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate diverse flow configurations:
        - Taylor-Green at various Re
        - Shear layers
        - Vortex pairs
        - Random initial conditions
        """
        from core.fluid_solver_2d import FluidSolver2D
        
        inputs_list = []
        outputs_list = []
        params_list = []
        
        configs = ['taylor_green', 'shear_layer', 'vortex_pair', 'random']
        
        per_config = n_samples // len(configs)
        
        for config_type in configs:
            for i in range(per_config):
                nu = np.random.uniform(*self.nu_range)
                
                solver = FluidSolver2D(
                    nx=self.nx, ny=self.ny,
                    Lx=self.Lx, Ly=self.Ly,
                    nu=nu, dt=self.dt,
                    pressure_solver="fft"
                )
                solver.bc_manager.set_periodic()
                
                if config_type == 'taylor_green':
                    solver.initialize_taylor_green(np.random.uniform(0.5, 2.0))
                elif config_type == 'shear_layer':
                    solver.initialize_double_shear_layer(
                        amplitude=np.random.uniform(0.01, 0.1),
                        delta=np.random.uniform(0.02, 0.1)
                    )
                elif config_type == 'vortex_pair':
                    solver.initialize_vortex_pair(
                        strength=np.random.uniform(0.5, 2.0),
                        separation=np.random.uniform(0.2, 0.5)
                    )
                else:
                    # Random smooth IC
                    k_modes = np.random.randint(2, 6)
                    solver.u = np.random.randn(self.ny, self.nx) * 0.1
                    solver.v = np.random.randn(self.ny, self.nx) * 0.1
                    # Smooth
                    from scipy.ndimage import gaussian_filter
                    solver.u = gaussian_filter(solver.u, sigma=3)
                    solver.v = gaussian_filter(solver.v, sigma=3)
                
                # Warm up
                for _ in range(5):
                    solver.step()
                
                # Record n_timesteps
                for t in range(min(self.n_timesteps, 20)):
                    state_in = np.stack([solver.u.copy(), solver.v.copy(), solver.p.copy()])
                    solver.step()
                    state_out = np.stack([solver.u.copy(), solver.v.copy(), solver.p.copy()])
                    
                    inputs_list.append(state_in)
                    outputs_list.append(state_out)
                    params_list.append([nu, solver.time, configs.index(config_type)])
            
            print(f"  Completed {config_type} configurations")
        
        data = {
            'inputs': np.array(inputs_list),
            'outputs': np.array(outputs_list),
            'params': np.array(params_list),
        }
        
        if save_path:
            np.savez_compressed(save_path, **data)
            print(f"Saved dataset: {data['inputs'].shape[0]} samples")
        
        return data
    
    def generate_pinn_collocation_points(
        self, n_collocation: int = 10000,
        n_boundary: int = 2000,
        n_initial: int = 2000,
        t_range: Tuple[float, float] = (0, 1.0),
    ) -> Dict[str, torch.Tensor]:
        """
        Generate collocation points for PINN training.
        
        Returns tensors for:
            - Interior collocation points
            - Boundary points with BC values
            - Initial condition points
        """
        # Interior collocation (Latin Hypercube Sampling)
        x_col = torch.rand(n_collocation, 1) * self.Lx
        y_col = torch.rand(n_collocation, 1) * self.Ly
        t_col = torch.rand(n_collocation, 1) * (t_range[1] - t_range[0]) + t_range[0]
        
        # Boundary points
        x_bc_list, y_bc_list, t_bc_list = [], [], []
        u_bc_list, v_bc_list = [], []
        
        n_per_wall = n_boundary // 4
        
        for wall in ['top', 'bottom', 'left', 'right']:
            t_wall = torch.rand(n_per_wall, 1) * (t_range[1] - t_range[0]) + t_range[0]
            
            if wall == 'top':
                x_w = torch.rand(n_per_wall, 1) * self.Lx
                y_w = torch.full((n_per_wall, 1), self.Ly)
            elif wall == 'bottom':
                x_w = torch.rand(n_per_wall, 1) * self.Lx
                y_w = torch.zeros(n_per_wall, 1)
            elif wall == 'left':
                x_w = torch.zeros(n_per_wall, 1)
                y_w = torch.rand(n_per_wall, 1) * self.Ly
            else:
                x_w = torch.full((n_per_wall, 1), self.Lx)
                y_w = torch.rand(n_per_wall, 1) * self.Ly
            
            x_bc_list.append(x_w)
            y_bc_list.append(y_w)
            t_bc_list.append(t_wall)
            u_bc_list.append(torch.zeros(n_per_wall, 1))
            v_bc_list.append(torch.zeros(n_per_wall, 1))
        
        # Initial condition points
        x_ic = torch.rand(n_initial, 1) * self.Lx
        y_ic = torch.rand(n_initial, 1) * self.Ly
        
        # Taylor-Green IC
        u_ic = torch.cos(x_ic) * torch.sin(y_ic)
        v_ic = -torch.sin(x_ic) * torch.cos(y_ic)
        
        return {
            'x_pde': x_col, 'y_pde': y_col, 't_pde': t_col,
            'x_bc': torch.cat(x_bc_list), 'y_bc': torch.cat(y_bc_list),
            't_bc': torch.cat(t_bc_list),
            'u_bc': torch.cat(u_bc_list), 'v_bc': torch.cat(v_bc_list),
            'x_ic': x_ic, 'y_ic': y_ic,
            'u_ic': u_ic, 'v_ic': v_ic,
        }


class FlowDataset(Dataset):
    """PyTorch Dataset for flow field data."""
    
    def __init__(self, data: Dict[str, np.ndarray], transform=None):
        self.inputs = torch.tensor(data['inputs'], dtype=torch.float32)
        self.outputs = torch.tensor(data['outputs'], dtype=torch.float32)
        self.params = torch.tensor(data['params'], dtype=torch.float32) if 'params' in data else None
        self.transform = transform
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.params is not None:
            return x, y, self.params[idx]
        return x, y


def create_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from generated data."""
    n = len(data['inputs'])
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val
    
    indices = np.random.permutation(n)
    
    train_data = {k: v[indices[:n_train]] for k, v in data.items()}
    val_data = {k: v[indices[n_train:n_train+n_val]] for k, v in data.items()}
    test_data = {k: v[indices[n_train+n_val:]] for k, v in data.items()}
    
    train_loader = DataLoader(FlowDataset(train_data), batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(FlowDataset(val_data), batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(FlowDataset(test_data), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
