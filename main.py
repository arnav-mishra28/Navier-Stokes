"""
=============================================================================
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║     NAVIER-STOKES ML/DL HYBRID SIMULATION SYSTEM                    ║
  ║     Research-Grade CFD + Deep Learning Platform                      ║
  ║                                                                     ║
  ║     Core: Incompressible Navier-Stokes (Projection Method)          ║
  ║     ML:   PINN / FNO / DeepONet / U-Net Surrogate                  ║
  ║     Physics: Fluid · MHD · Astro · Bio · Climate · Quantum          ║
  ║     Viz:  Real-time 2D (Pygame) + 3D (PyVista/Matplotlib)          ║
  ╚═══════════════════════════════════════════════════════════════════════╝

  Master entry point — orchestrates the entire system.
  
  Usage:
      python main.py                     → Interactive mode selector
      python main.py --demo              → Run Taylor-Green vortex demo
      python main.py --viz2d             → Launch 2D real-time visualizer
      python main.py --viz3d             → Launch 3D visualizer
      python main.py --dashboard         → Launch Streamlit dashboard
      python main.py --train pinn        → Train PINN model
      python main.py --train fno         → Train FNO model
      python main.py --train deeponet    → Train DeepONet model
      python main.py --train surrogate   → Train U-Net surrogate model
      python main.py --benchmark         → Run CFD benchmarks
      python main.py --physics mhd       → Run MHD simulation
      python main.py --physics astro     → Run astrophysics simulation
      python main.py --physics bio       → Run biophysics simulation
      python main.py --physics climate   → Run climate simulation
      python main.py --physics quantum   → Run quantum fluid simulation
      python main.py --no-gui            → Headless mode (no plt.show())
=============================================================================
"""

import sys
import os
import argparse
import numpy as np
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Global flag for headless mode
HEADLESS = False


# =============================================================================
# Publication-quality plot styling
# =============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality dark-themed plots."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.rcParams.update({
        # Dark background
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'savefig.facecolor': '#0d1117',
        # Text
        'text.color': '#c9d1d9',
        'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        # Grid
        'axes.grid': False,
        'grid.color': '#21262d',
        'grid.alpha': 0.5,
        # Axes
        'axes.edgecolor': '#30363d',
        'axes.linewidth': 0.8,
        # Legend
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',
        'legend.fontsize': 10,
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
    })


def show_or_close(fig):
    """Show plot if GUI available, otherwise close to free memory."""
    import matplotlib.pyplot as plt
    if not HEADLESS:
        try:
            plt.show()
        except Exception:
            plt.close(fig)
    else:
        plt.close(fig)


# =============================================================================
# Banner
# =============================================================================

def print_banner():
    """Print the system banner."""
    print("\n" + "="*72)
    print(r"""
     _   _    _   __     __ ___  ___  ___        
    | \ | |  / \  \ \   / /|_ _|| __|| _ \       
    |  \| | / _ \  \ \ / /  | | | _| |   /       
    |_|\__||_/ \_\  \_V_/  |___||___||_|_\       
                                                  
     ___  _____  ___   _  __ ___  ___             
    / __||_   _|/ _ \ | |/ /| __|/ __|            
    \__ \  | | | (_) ||   < | _| \__ \            
    |___/  |_|  \___/ |_|\_\|___||___/            
    """)
    print("  ML/DL Hybrid Navier-Stokes Simulation System")
    print("  du/dt + (u.grad)u = -grad(p) + nu*laplacian(u) + f")
    print("="*72 + "\n")


def check_dependencies():
    """Check and report available dependencies."""
    deps = {
        'numpy': True, 'scipy': False, 'torch': False,
        'matplotlib': False, 'pygame': False,
        'pyvista': False, 'streamlit': False, 'plotly': False,
    }
    
    for pkg in deps:
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    
    print("  Dependencies:")
    for pkg, available in deps.items():
        status = "✓" if available else "✗"
        color = "" if available else " (install: pip install " + pkg + ")"
        print(f"    [{status}] {pkg}{color}")
    
    # Check CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            print(f"    [✓] CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"    [·] CUDA (not available, using CPU)")
    except ImportError:
        print(f"    [✗] CUDA (PyTorch not installed)")
    
    print()
    return deps


# =============================================================================
# Taylor-Green Vortex Demo
# =============================================================================

def run_demo():
    """Run the Taylor-Green vortex decay demo with publication-quality plots."""
    print("\n" + "="*60)
    print("  DEMO: Taylor-Green Vortex Decay")
    print("  Analytical benchmark for NS solver validation")
    print("="*60 + "\n")
    
    from core.fluid_solver_2d import FluidSolver2D
    from utils.helpers import compute_vorticity, compute_kinetic_energy, compute_enstrophy
    
    # Setup solver
    nx, ny = 128, 128
    Re = 100
    nu = 1.0 / Re
    
    solver = FluidSolver2D(
        nx=nx, ny=ny,
        Lx=2*np.pi, Ly=2*np.pi,
        nu=nu, dt=0.01,
        pressure_solver="fft",
        advection_scheme="central"
    )
    solver.initialize_taylor_green(amplitude=1.0)
    solver.bc_manager.set_periodic()
    
    print(f"  Grid: {nx}×{ny}")
    print(f"  Reynolds: {Re}")
    print(f"  Viscosity: {nu:.4f}")
    print(f"  dt: {solver.dt}")
    print()
    
    # Simulate
    n_steps = 500
    record_interval = 10
    times, ke_numerical, ke_analytical = [], [], []
    
    t_start = time.perf_counter()
    
    for step in range(n_steps):
        solver.step()
        
        if step % record_interval == 0:
            omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
            ke = compute_kinetic_energy(solver.u, solver.v)
            ke_exact = 0.25 * np.exp(-4 * nu * solver.time)
            
            times.append(solver.time)
            ke_numerical.append(ke)
            ke_analytical.append(ke_exact)
            
            if step % 100 == 0:
                error = abs(ke - ke_exact) / max(ke_exact, 1e-10)
                print(f"  Step {step:4d} | t={solver.time:.3f} | KE={ke:.6f} | "
                      f"KE_exact={ke_exact:.6f} | Error={error:.2e}")
    
    elapsed = time.perf_counter() - t_start
    
    print(f"\n  Simulation complete: {n_steps} steps in {elapsed:.2f}s")
    print(f"  Performance: {n_steps/elapsed:.0f} steps/sec")
    
    # ---- Publication-quality visualization ----
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        setup_plot_style()
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(
            f'Taylor-Green Vortex Decay   ·   Re = {Re}   ·   {nx}×{ny}',
            fontsize=18, fontweight='bold', color='#58a6ff', y=0.97
        )
        
        # — Vorticity field —
        ax1 = fig.add_subplot(2, 3, 1)
        omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
        vmax = np.max(np.abs(omega))
        im1 = ax1.imshow(
            omega, cmap='RdBu_r', origin='lower',
            extent=[0, 2*np.pi, 0, 2*np.pi],
            norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
            interpolation='bicubic'
        )
        ax1.set_title('Vorticity  ω', color='#79c0ff')
        ax1.set_xlabel('x'); ax1.set_ylabel('y')
        cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.yaxis.set_tick_params(color='#8b949e')
        cb1.outline.set_edgecolor('#30363d')
        
        # — Velocity magnitude —
        ax2 = fig.add_subplot(2, 3, 2)
        speed = solver.get_velocity_magnitude()
        im2 = ax2.imshow(
            speed, cmap='magma', origin='lower',
            extent=[0, 2*np.pi, 0, 2*np.pi],
            interpolation='bicubic'
        )
        ax2.set_title('Velocity Magnitude  |u|', color='#79c0ff')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.yaxis.set_tick_params(color='#8b949e')
        cb2.outline.set_edgecolor('#30363d')
        
        # — Pressure —
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(
            solver.p, cmap='cividis', origin='lower',
            extent=[0, 2*np.pi, 0, 2*np.pi],
            interpolation='bicubic'
        )
        ax3.set_title('Pressure  p', color='#79c0ff')
        ax3.set_xlabel('x'); ax3.set_ylabel('y')
        cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.ax.yaxis.set_tick_params(color='#8b949e')
        cb3.outline.set_edgecolor('#30363d')
        
        # — Streamlines —
        ax4 = fig.add_subplot(2, 3, 4)
        X, Y = np.meshgrid(
            np.linspace(0, 2*np.pi, nx),
            np.linspace(0, 2*np.pi, ny)
        )
        ax4.imshow(
            speed, cmap='magma', origin='lower', alpha=0.35,
            extent=[0, 2*np.pi, 0, 2*np.pi], interpolation='bicubic'
        )
        strm = ax4.streamplot(
            X, Y, solver.u, solver.v,
            color=speed, cmap='cool', density=2.0, linewidth=0.8,
            arrowsize=0.8, arrowstyle='->'
        )
        ax4.set_title('Streamlines', color='#79c0ff')
        ax4.set_xlabel('x'); ax4.set_ylabel('y')
        ax4.set_xlim(0, 2*np.pi); ax4.set_ylim(0, 2*np.pi)
        
        # — KE Decay —
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(times, ke_numerical, color='#58a6ff', lw=2.2, label='Numerical', zorder=3)
        ax5.plot(times, ke_analytical, color='#f97583', lw=2.2, ls='--', label='Analytical', zorder=2)
        ax5.fill_between(times, ke_numerical, ke_analytical, alpha=0.12, color='#58a6ff')
        ax5.set_xlabel('Time  t')
        ax5.set_ylabel('Kinetic Energy  E')
        ax5.set_title('KE Decay  (Validation)', color='#79c0ff')
        ax5.legend(framealpha=0.8)
        ax5.grid(True, alpha=0.15, color='#30363d')
        
        # — Error —
        ax6 = fig.add_subplot(2, 3, 6)
        errors = [abs(n - a) / max(a, 1e-10) for n, a in zip(ke_numerical, ke_analytical)]
        ax6.semilogy(times, errors, color='#7ee787', lw=2.2)
        ax6.fill_between(times, errors, alpha=0.10, color='#7ee787')
        ax6.set_xlabel('Time  t')
        ax6.set_ylabel('Relative Error')
        ax6.set_title('KE Relative Error', color='#79c0ff')
        ax6.grid(True, alpha=0.15, color='#30363d')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filepath = os.path.join(PROJECT_ROOT, 'demo_taylor_green.png')
        plt.savefig(filepath, bbox_inches='tight')
        show_or_close(fig)
        print(f"\n  Plot saved to {filepath}")
        
    except ImportError:
        print("  (matplotlib not available, skipping plots)")


# =============================================================================
# Benchmarks
# =============================================================================

def run_benchmark():
    """Run CFD solver benchmarks."""
    print("\n" + "="*60)
    print("  BENCHMARK: CFD Solver Performance")
    print("="*60 + "\n")
    
    from core.fluid_solver_2d import FluidSolver2D
    
    results = []
    resolutions = [32, 64, 128, 256]
    
    for nx in resolutions:
        solver = FluidSolver2D(
            nx=nx, ny=nx,
            Lx=2*np.pi, Ly=2*np.pi,
            nu=0.01, dt=0.005,
            pressure_solver="fft"
        )
        solver.initialize_taylor_green()
        solver.bc_manager.set_periodic()
        
        n_steps = 100
        t0 = time.perf_counter()
        for _ in range(n_steps):
            solver.step()
        elapsed = time.perf_counter() - t0
        
        rate = n_steps / elapsed
        gridpoints = nx * nx * n_steps
        
        result = {
            'resolution': f"{nx}×{nx}",
            'steps': n_steps,
            'time': elapsed,
            'rate': rate,
            'mlups': gridpoints / elapsed / 1e6,
        }
        results.append(result)
        
        print(f"  {result['resolution']:>8s} | {result['rate']:8.1f} steps/s | "
              f"{result['mlups']:6.2f} MLUPS | {result['time']:.2f}s")
    
    print(f"\n  MLUPS = Million Lattice-point Updates Per Second")
    
    # Test pressure solvers
    print(f"\n  Pressure Solver Comparison (128×128, 100 steps):")
    for method in ["fft", "jacobi", "sor", "cg"]:
        try:
            solver = FluidSolver2D(
                nx=128, ny=128,
                Lx=2*np.pi, Ly=2*np.pi,
                nu=0.01, dt=0.005,
                pressure_solver=method
            )
            solver.initialize_taylor_green()
            solver.bc_manager.set_periodic()
            
            t0 = time.perf_counter()
            for _ in range(100):
                solver.step()
            elapsed = time.perf_counter() - t0
            
            print(f"    {method:>10s}: {100/elapsed:8.1f} steps/s ({elapsed:.2f}s)")
        except Exception as e:
            print(f"    {method:>10s}: Failed ({e})")


# =============================================================================
# Physics Domain Demos
# =============================================================================

def run_physics_demo(domain: str):
    """Run a physics domain-specific demo with publication-quality visuals."""
    print(f"\n{'='*60}")
    print(f"  PHYSICS DEMO: {domain.upper()}")
    print(f"{'='*60}\n")
    
    if domain == "mhd":
        from physics.mhd import MHDSolver
        solver = MHDSolver(nx=128, ny=128, nu=0.005, eta=0.005, dt=0.005)
        solver.initialize_orszag_tang()
        title = "Orszag-Tang MHD Vortex"
        n_steps = 400
        
    elif domain == "astro":
        from physics.astrophysics import AstrophysicalFlowSolver
        solver = AstrophysicalFlowSolver(nx=128, ny=128, nu=0.01, dt=0.005)
        solver.initialize_rayleigh_taylor()
        title = "Rayleigh-Taylor Instability"
        n_steps = 300
        
    elif domain == "bio":
        from physics.biophysics import BiophysicsFlowSolver
        solver = BiophysicsFlowSolver(nx=200, ny=50, dt=0.0005)
        solver.initialize_straight_vessel(stenosis=0.5)
        title = "Pulsatile Blood Flow with Stenosis"
        n_steps = 500
        
    elif domain == "climate":
        from physics.climate import ClimateFlowSolver
        solver = ClimateFlowSolver(nx=128, ny=128, nu=500, dt=500)
        solver.initialize_kelvin_helmholtz()
        title = "Kelvin-Helmholtz Atmospheric Instability"
        n_steps = 200
        
    elif domain == "quantum":
        from physics.quantum_fluids import QuantumFluidSolver
        solver = QuantumFluidSolver(nx=256, ny=256, g_int=500, dt=0.0005)
        solver.initialize_quantum_turbulence(n_vortices=15)
        title = "Quantum Turbulence (Bose-Einstein Condensate)"
        n_steps = 500
    else:
        print(f"  Unknown domain: {domain}")
        return
    
    print(f"  Running: {title}")
    print(f"  Steps: {n_steps}")
    
    t0 = time.perf_counter()
    solver.advance(n_steps, record=True)
    elapsed = time.perf_counter() - t0
    
    print(f"  Completed in {elapsed:.2f}s ({n_steps/elapsed:.1f} steps/s)")
    
    # ---- Publication-quality visualization ----
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        setup_plot_style()
        
        state = solver.get_state()
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(
            f'{title}   ·   t = {state["time"]:.3f}',
            fontsize=16, fontweight='bold', color='#58a6ff', y=0.98
        )
        
        # — Field 1: Velocity magnitude —
        vel = state.get('velocity_magnitude', np.sqrt(state['u']**2 + state['v']**2))
        im1 = axes[0].imshow(vel, cmap='magma', origin='lower', interpolation='bicubic')
        axes[0].set_title('Velocity Magnitude', color='#79c0ff')
        cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cb1.outline.set_edgecolor('#30363d')
        
        # — Field 2: Domain-specific —
        if 'Jz' in state:
            field2, cmap2, t2 = state['Jz'], 'RdBu_r', 'Current Density  Jz'
            vabs = np.max(np.abs(field2))
            norm2 = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
        elif 'rho' in state:
            field2, cmap2, t2 = state['rho'], 'inferno', 'Density  ρ'
            norm2 = None
        elif 'density' in state:
            field2, cmap2, t2 = state['density'], 'viridis', 'Superfluid Density  |ψ|²'
            norm2 = None
        elif 'omega' in state:
            field2, cmap2, t2 = state['omega'], 'RdBu_r', 'Vorticity  ω'
            vabs = np.max(np.abs(field2))
            norm2 = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs) if vabs > 0 else None
        else:
            vort = np.gradient(state['v'], axis=1) - np.gradient(state['u'], axis=0)
            field2, cmap2, t2 = vort, 'RdBu_r', 'Vorticity  ω'
            vabs = np.max(np.abs(vort))
            norm2 = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs) if vabs > 0 else None
        
        im2 = axes[1].imshow(field2, cmap=cmap2, origin='lower', norm=norm2,
                             interpolation='bicubic')
        axes[1].set_title(t2, color='#79c0ff')
        cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb2.outline.set_edgecolor('#30363d')
        
        # — Field 3: Pressure / phase / temperature —
        if 'phase' in state:
            im3 = axes[2].imshow(state['phase'], cmap='twilight_shifted', origin='lower',
                                 interpolation='bicubic')
            axes[2].set_title('Phase  arg(ψ)', color='#79c0ff')
        elif 'T' in state:
            im3 = axes[2].imshow(state['T'], cmap='inferno', origin='lower',
                                 interpolation='bicubic')
            axes[2].set_title('Temperature  T', color='#79c0ff')
        elif 'p' in state:
            im3 = axes[2].imshow(state['p'], cmap='cividis', origin='lower',
                                 interpolation='bicubic')
            axes[2].set_title('Pressure  p', color='#79c0ff')
        else:
            # Compute vorticity as fallback
            vort = np.gradient(state['v'], axis=1) - np.gradient(state['u'], axis=0)
            im3 = axes[2].imshow(vort, cmap='RdBu_r', origin='lower',
                                 interpolation='bicubic')
            axes[2].set_title('Vorticity  ω', color='#79c0ff')
        
        cb3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cb3.outline.set_edgecolor('#30363d')
        
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.tick_params(axis='both', colors='#8b949e')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        filename = f'demo_{domain}.png'
        filepath = os.path.join(PROJECT_ROOT, filename)
        plt.savefig(filepath, bbox_inches='tight')
        show_or_close(fig)
        print(f"  Saved: {filepath}")
        
    except ImportError:
        print("  (matplotlib not available)")


# =============================================================================
# ML Training
# =============================================================================

def run_training(model_type: str):
    """Run ML model training."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_type.upper()} Model")
    print(f"{'='*60}\n")
    
    try:
        import torch
    except ImportError:
        print("  ERROR: PyTorch is required for training. Install with: pip install torch")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    if model_type == "pinn":
        from models.pinn import PINN
        from training.data_generator import NSDataGenerator
        from training.trainer import UnifiedTrainer
        
        # Create PINN
        model = PINN(
            hidden_layers=[128, 128, 128, 128],
            activation="tanh",
            use_fourier_features=True,
            fourier_features=64,
        )
        model.nu = 0.01
        
        # Generate collocation points
        gen = NSDataGenerator(nx=64, ny=64)
        data = gen.generate_pinn_collocation_points(
            n_collocation=5000,
            n_boundary=1000,
            n_initial=1000,
        )
        
        # Train
        trainer = UnifiedTrainer(model, model_type="pinn", device=device, learning_rate=1e-3)
        trainer.train_pinn(data, epochs=5000)
        trainer.plot_training_history(save_path=os.path.join(PROJECT_ROOT, 'pinn_training.png'))
        
    elif model_type == "fno":
        from models.fno import FNO2d
        from training.data_generator import NSDataGenerator, create_dataloaders
        from training.trainer import UnifiedTrainer
        
        # Generate training data
        print("  Generating training data...")
        gen = NSDataGenerator(nx=64, ny=64, dt=0.01)
        data = gen.generate_taylor_green_dataset(
            n_samples=50,
            save_path=os.path.join(PROJECT_ROOT, 'data', 'fno_training_data.npz')
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data, batch_size=16, val_split=0.1, test_split=0.1
        )
        
        # Create FNO
        model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4,
                      in_channels=3, out_channels=3)
        
        # Train
        trainer = UnifiedTrainer(model, model_type="fno", device=device, learning_rate=1e-3)
        trainer.train_fno(train_loader, val_loader, epochs=100)
        trainer.plot_training_history(save_path=os.path.join(PROJECT_ROOT, 'fno_training.png'))
    
    elif model_type == "deeponet":
        from models.deeponet import DeepONet
        from training.data_generator import NSDataGenerator, create_dataloaders
        from training.trainer import UnifiedTrainer
        
        # Generate training data (reuse FNO data format for simplicity)
        print("  Generating training data...")
        gen = NSDataGenerator(nx=64, ny=64, dt=0.01)
        data = gen.generate_taylor_green_dataset(n_samples=50)
        
        train_loader, val_loader, _ = create_dataloaders(
            data, batch_size=16, val_split=0.1, test_split=0.1
        )
        
        # Create DeepONet
        n_sensors = 3 * 64 * 64  # flattened (u,v,p) field
        model = DeepONet(
            branch_input_dim=n_sensors,
            trunk_input_dim=2,
            latent_dim=128,
            n_outputs=3,
        )
        
        trainer = UnifiedTrainer(model, model_type="deeponet", device=device, learning_rate=1e-3)
        # Use FNO-style training as data has same shape
        trainer.train_fno(train_loader, val_loader, epochs=100)
        trainer.plot_training_history(save_path=os.path.join(PROJECT_ROOT, 'deeponet_training.png'))
    
    elif model_type == "surrogate":
        from models.surrogate import UNetSurrogate
        from training.data_generator import NSDataGenerator, create_dataloaders
        from training.trainer import UnifiedTrainer
        
        # Generate training data
        print("  Generating training data...")
        gen = NSDataGenerator(nx=64, ny=64, dt=0.01)
        data = gen.generate_taylor_green_dataset(n_samples=50)
        
        train_loader, val_loader, _ = create_dataloaders(
            data, batch_size=16, val_split=0.1, test_split=0.1
        )
        
        # Create U-Net Surrogate
        model = UNetSurrogate(in_channels=3, out_channels=3)
        
        trainer = UnifiedTrainer(model, model_type="surrogate", device=device, learning_rate=1e-3)
        trainer.train_surrogate(train_loader, val_loader, epochs=100)
        trainer.plot_training_history(save_path=os.path.join(PROJECT_ROOT, 'surrogate_training.png'))
    
    else:
        print(f"  Training for '{model_type}' not recognized.")
        print(f"  Available: pinn, fno, deeponet, surrogate")


# =============================================================================
# Interactive Menu
# =============================================================================

def interactive_menu():
    """Interactive mode selector."""
    print_banner()
    deps = check_dependencies()
    
    print("  Select Mode:")
    print("  ─────────────────────────────────────────")
    print("  [1]  🎬 Demo: Taylor-Green Vortex Decay")
    print("  [2]  🎮 Real-Time 2D Visualizer (Pygame)")
    print("  [3]  🌐 3D Visualizer (Matplotlib/PyVista)")
    print("  [4]  📊 Streamlit Dashboard")
    print("  [5]  🏋 Train PINN Model")
    print("  [6]  🏋 Train FNO Model")
    print("  [7]  🏋 Train DeepONet Model")
    print("  [8]  🏋 Train U-Net Surrogate")
    print("  [9]  ⚡ MHD Simulation")
    print("  [10] 🌟 Astrophysics Simulation")
    print("  [11] ❤ Biophysics (Blood Flow)")
    print("  [12] 🌍 Climate Simulation")
    print("  [13] ⚛ Quantum Fluid Simulation")
    print("  [14] 📏 CFD Benchmarks")
    print("  [0]  ❌ Exit")
    print()
    
    try:
        choice = input("  Enter choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Goodbye!")
        return
    
    dispatch = {
        "1": lambda: run_demo(),
        "2": lambda: _launch_viz2d(),
        "3": lambda: _launch_viz3d(),
        "4": lambda: os.system(f'streamlit run "{os.path.join(PROJECT_ROOT, "dashboard", "app.py")}"'),
        "5": lambda: run_training("pinn"),
        "6": lambda: run_training("fno"),
        "7": lambda: run_training("deeponet"),
        "8": lambda: run_training("surrogate"),
        "9": lambda: run_physics_demo("mhd"),
        "10": lambda: run_physics_demo("astro"),
        "11": lambda: run_physics_demo("bio"),
        "12": lambda: run_physics_demo("climate"),
        "13": lambda: run_physics_demo("quantum"),
        "14": lambda: run_benchmark(),
        "0": lambda: print("  Goodbye!"),
    }
    
    action = dispatch.get(choice)
    if action:
        action()
    else:
        print(f"  Unknown choice: {choice}")


def _launch_viz2d():
    from visualization.realtime_2d import RealtimeVisualizer2D
    viz = RealtimeVisualizer2D()
    viz.run()


def _launch_viz3d():
    from visualization.realtime_3d import RealtimeVisualizer3D
    viz = RealtimeVisualizer3D()
    viz.run()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    global HEADLESS
    
    parser = argparse.ArgumentParser(
        description="Navier-Stokes ML/DL Hybrid Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      Interactive menu
  python main.py --demo               Taylor-Green vortex demo
  python main.py --demo --no-gui      Demo (headless, save plots only)
  python main.py --viz2d              Real-time 2D pygame visualizer
  python main.py --viz3d              3D matplotlib/PyVista visualizer
  python main.py --dashboard          Streamlit web dashboard
  python main.py --train pinn         Train PINN model
  python main.py --train fno          Train FNO model
  python main.py --train deeponet     Train DeepONet model
  python main.py --train surrogate    Train U-Net surrogate model
  python main.py --physics mhd        MHD simulation
  python main.py --physics quantum    Quantum fluid simulation
  python main.py --benchmark          CFD performance benchmarks
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run Taylor-Green demo')
    parser.add_argument('--viz2d', action='store_true', help='Launch 2D real-time visualizer')
    parser.add_argument('--viz3d', action='store_true', help='Launch 3D visualizer')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    parser.add_argument('--train', type=str, choices=['pinn', 'fno', 'deeponet', 'surrogate'],
                       help='Train ML model')
    parser.add_argument('--physics', type=str,
                       choices=['mhd', 'astro', 'bio', 'climate', 'quantum'],
                       help='Run physics domain simulation')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--no-gui', action='store_true', dest='no_gui',
                       help='Headless mode: save plots to disk, skip plt.show()')
    
    args = parser.parse_args()
    
    # Handle headless mode
    if args.no_gui:
        HEADLESS = True
        import matplotlib
        matplotlib.use('Agg')
    
    print_banner()
    
    if args.demo:
        run_demo()
    elif args.viz2d:
        _launch_viz2d()
    elif args.viz3d:
        _launch_viz3d()
    elif args.dashboard:
        os.system(f'streamlit run "{os.path.join(PROJECT_ROOT, "dashboard", "app.py")}"')
    elif args.train:
        run_training(args.train)
    elif args.physics:
        run_physics_demo(args.physics)
    elif args.benchmark:
        run_benchmark()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
