"""
=============================================================================
  Turbulence Discovery Data Generator
  
  Generates training data for:
      1. Autoencoder:       flow field snapshots at various Re
      2. Latent ODE:        consecutive frame pairs (z_t, z_{t+dt})
      3. Blow-up detector:  ICs labeled by long-term stability
      4. Symbolic discovery: latent trajectories from autoencoder
  
  Data includes multiple flow regimes:
      - Laminar (Re < 100)
      - Transitional (Re ~ 100-1000)
      - Turbulent (Re > 1000)
      - Near-singular (very high Re, coarse grid)
=============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TurbulenceDataGenerator:
    """
    Generate multi-regime flow data for turbulence discovery AI.
    """
    
    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2*np.pi,
        Ly: float = 2*np.pi,
        dt: float = 0.005,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
    
    def _create_solver(self, nu: float, ic_type: str = 'taylor_green'):
        """Create a solver with given parameters and initial condition."""
        from core.fluid_solver_2d import FluidSolver2D
        
        solver = FluidSolver2D(
            nx=self.nx, ny=self.ny,
            Lx=self.Lx, Ly=self.Ly,
            nu=nu, dt=self.dt,
            pressure_solver="fft"
        )
        solver.bc_manager.set_periodic()
        
        if ic_type == 'taylor_green':
            solver.initialize_taylor_green(amplitude=np.random.uniform(0.5, 2.0))
        elif ic_type == 'shear_layer':
            solver.initialize_double_shear_layer(
                amplitude=np.random.uniform(0.02, 0.1),
                delta=np.random.uniform(0.02, 0.08)
            )
        elif ic_type == 'vortex_pair':
            solver.initialize_vortex_pair(
                strength=np.random.uniform(0.5, 2.0),
                separation=np.random.uniform(0.2, 0.5)
            )
        elif ic_type == 'random':
            k = np.random.randint(2, 5)
            solver.u = np.random.randn(self.ny, self.nx) * 0.3
            solver.v = np.random.randn(self.ny, self.nx) * 0.3
            # Smooth with spectral filter
            u_hat = np.fft.fft2(solver.u)
            v_hat = np.fft.fft2(solver.v)
            kx = np.fft.fftfreq(self.nx) * self.nx
            ky = np.fft.fftfreq(self.ny) * self.ny
            KX, KY = np.meshgrid(kx, ky)
            filt = np.exp(-(KX**2 + KY**2) / (2 * k**2))
            solver.u = np.real(np.fft.ifft2(u_hat * filt))
            solver.v = np.real(np.fft.ifft2(v_hat * filt))
        
        return solver
    
    def generate_autoencoder_data(
        self,
        n_samples: int = 200,
        re_range: Tuple[float, float] = (10, 5000),
        warmup_steps: int = 20,
        record_interval: int = 5,
        frames_per_sim: int = 20,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate flow field snapshots for autoencoder training.
        
        Each sample: (u, v, p, ω) at shape (4, ny, nx)
        """
        from utils.helpers import compute_vorticity
        
        snapshots = []
        re_labels = []
        ic_labels = []
        
        ic_types = ['taylor_green', 'shear_layer', 'vortex_pair', 'random']
        per_type = n_samples // len(ic_types)
        
        for ic_type in ic_types:
            for i in range(per_type):
                re = np.exp(np.random.uniform(
                    np.log(re_range[0]), np.log(re_range[1])
                ))
                nu = 1.0 / re
                
                try:
                    solver = self._create_solver(nu, ic_type)
                    
                    # Warmup
                    for _ in range(warmup_steps):
                        solver.step()
                        if not np.all(np.isfinite(solver.u)):
                            raise ValueError("Diverged during warmup")
                    
                    # Record frames
                    for f in range(frames_per_sim):
                        for _ in range(record_interval):
                            solver.step()
                            if not np.all(np.isfinite(solver.u)):
                                raise ValueError("Diverged")
                        
                        omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                        snapshot = np.stack([
                            solver.u, solver.v, solver.p, omega
                        ])  # (4, ny, nx)
                        
                        snapshots.append(snapshot)
                        re_labels.append(re)
                        ic_labels.append(ic_types.index(ic_type))
                    
                except (ValueError, RuntimeError):
                    continue
            
            if verbose:
                print(f"  Generated {ic_type} data ({len(snapshots)} total snapshots)")
        
        data = {
            'snapshots': np.array(snapshots, dtype=np.float32),
            'reynolds': np.array(re_labels, dtype=np.float32),
            'ic_type': np.array(ic_labels, dtype=np.int32),
        }
        
        if verbose:
            print(f"  Total: {len(snapshots)} snapshots, shape {data['snapshots'].shape}")
        
        return data
    
    def generate_paired_frames(
        self,
        n_trajectories: int = 50,
        frames_per_traj: int = 20,
        re_range: Tuple[float, float] = (50, 2000),
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate consecutive frame pairs for latent ODE training.
        
        Returns:
            frames_t0: (N, 4, ny, nx) — flow at time t
            frames_t1: (N, 4, ny, nx) — flow at time t + dt
            dt_values:  (N,) — time step sizes
        """
        from utils.helpers import compute_vorticity
        
        frames_t0 = []
        frames_t1 = []
        dt_values = []
        
        ic_types = ['taylor_green', 'shear_layer', 'vortex_pair']
        
        for traj in range(n_trajectories):
            re = np.exp(np.random.uniform(np.log(re_range[0]), np.log(re_range[1])))
            nu = 1.0 / re
            ic_type = ic_types[traj % len(ic_types)]
            
            try:
                solver = self._create_solver(nu, ic_type)
                
                # Warmup
                for _ in range(10):
                    solver.step()
                
                for f in range(frames_per_traj):
                    # Record t0
                    omega0 = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                    snap0 = np.stack([solver.u.copy(), solver.v.copy(), 
                                      solver.p.copy(), omega0])
                    
                    # Advance one macro-step (5 micro-steps)
                    for _ in range(5):
                        solver.step()
                        if not np.all(np.isfinite(solver.u)):
                            raise ValueError("Diverged")
                    
                    # Record t1
                    omega1 = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                    snap1 = np.stack([solver.u.copy(), solver.v.copy(), 
                                      solver.p.copy(), omega1])
                    
                    frames_t0.append(snap0)
                    frames_t1.append(snap1)
                    dt_values.append(5 * self.dt)
                    
            except (ValueError, RuntimeError):
                continue
            
            if verbose and (traj + 1) % 10 == 0:
                print(f"  Generated {traj+1}/{n_trajectories} trajectories "
                      f"({len(frames_t0)} pairs)")
        
        return {
            'frames_t0': np.array(frames_t0, dtype=np.float32),
            'frames_t1': np.array(frames_t1, dtype=np.float32),
            'dt': np.array(dt_values, dtype=np.float32),
        }
    
    def generate_stability_labels(
        self,
        n_samples: int = 100,
        re_range: Tuple[float, float] = (10, 50000),
        test_steps: int = 200,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate labeled data for blow-up detection.
        
        Labels:
            0 = SMOOTH (stayed smooth throughout)
            1 = TRANSITIONAL (moderate vorticity growth)
            2 = TURBULENT (high vorticity, but stable)
            3 = UNSTABLE (very high vorticity, near blow-up)
            4 = SINGULAR_RISK (diverged numerically)
        """
        from utils.helpers import compute_vorticity
        
        initial_fields = []
        labels = []
        re_values = []
        
        ic_types = ['taylor_green', 'shear_layer', 'vortex_pair', 'random']
        
        for i in range(n_samples):
            re = np.exp(np.random.uniform(np.log(re_range[0]), np.log(re_range[1])))
            nu = 1.0 / re
            ic_type = ic_types[i % len(ic_types)]
            
            try:
                solver = self._create_solver(nu, ic_type)
                
                # Record initial state
                omega0 = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                from utils.helpers import compute_strain_rate
                strain0 = compute_strain_rate(solver.u, solver.v, solver.dx, solver.dy)
                
                initial_field = np.stack([
                    solver.u.copy(), solver.v.copy(), omega0, strain0
                ])
                
                # Simulate forward
                max_omega = 0
                diverged = False
                
                for step in range(test_steps):
                    solver.step()
                    if not np.all(np.isfinite(solver.u)):
                        diverged = True
                        break
                    
                    if step % 10 == 0:
                        omega = compute_vorticity(solver.u, solver.v, 
                                                   solver.dx, solver.dy)
                        max_omega = max(max_omega, np.max(np.abs(omega)))
                
                # Classify
                if diverged:
                    label = 4  # SINGULAR_RISK
                elif max_omega > 100:
                    label = 3  # UNSTABLE
                elif max_omega > 10:
                    label = 2  # TURBULENT
                elif max_omega > 1:
                    label = 1  # TRANSITIONAL
                else:
                    label = 0  # SMOOTH
                
                initial_fields.append(initial_field)
                labels.append(label)
                re_values.append(re)
                
            except Exception:
                continue
            
            if verbose and (i + 1) % 20 == 0:
                print(f"  Labeled {i+1}/{n_samples} samples")
        
        return {
            'fields': np.array(initial_fields, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'reynolds': np.array(re_values, dtype=np.float32),
        }


class FlowSnapshotDataset(Dataset):
    """PyTorch Dataset for flow field snapshots."""
    
    def __init__(self, data: Dict[str, np.ndarray], key: str = 'snapshots'):
        self.data = torch.tensor(data[key], dtype=torch.float32)
        self.re = torch.tensor(data.get('reynolds', np.zeros(len(self.data))), 
                               dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.re[idx]


class PairedFrameDataset(Dataset):
    """PyTorch Dataset for consecutive frame pairs."""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.t0 = torch.tensor(data['frames_t0'], dtype=torch.float32)
        self.t1 = torch.tensor(data['frames_t1'], dtype=torch.float32)
        self.dt = torch.tensor(data['dt'], dtype=torch.float32)
    
    def __len__(self):
        return len(self.t0)
    
    def __getitem__(self, idx):
        return self.t0[idx], self.t1[idx], self.dt[idx]


class StabilityDataset(Dataset):
    """PyTorch Dataset for blow-up detection."""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.fields = torch.tensor(data['fields'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.re = torch.tensor(data['reynolds'], dtype=torch.float32)
    
    def __len__(self):
        return len(self.fields)
    
    def __getitem__(self, idx):
        return self.fields[idx], self.labels[idx], self.re[idx]
