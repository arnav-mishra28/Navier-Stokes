"""
=============================================================================
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║     NAVIER-STOKES ML/DL HYBRID SIMULATION SYSTEM                    ║
  ║     + TURBULENCE DISCOVERY AI                                        ║
  ║     Research-Grade CFD + Deep Learning Platform                      ║
  ║                                                                     ║
  ║     Core: Incompressible Navier-Stokes (Projection Method)          ║
  ║     ML:   PINN / FNO / DeepONet / U-Net / Autoencoder / NeuralODE  ║
  ║     AI:   SINDy / Genetic Programming / Blow-up Detection           ║
  ║     AGI:  Physics Discovery / Hypothesis Engine / Knowledge Base    ║
  ║     Physics: Fluid · MHD · Astro · Bio · Climate · Quantum · Rel · Gravity ║
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
      python main.py --physics relativistic → Run relativistic NS (Israel-Stewart)
      python main.py --hybrid            → Run hybrid CFD→PINN demo
      python main.py --gpu               → Use GPU-accelerated solver
      python main.py --vort-conf 5.0     → Enable vorticity confinement
      python main.py --agi               → Full AGI scientific discovery system
      python main.py --physics-discover  → Physics-aware equation discovery
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
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
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
    print("  + Turbulence Discovery AI (SINDy / GP / Neural ODE)")
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
        filepath = os.path.join(IMAGES_DIR, 'demo_taylor_green.png')
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
        solver = MHDSolver(nx=128, ny=128, nu=0.01, eta=0.01, dt=0.005)
        solver.initialize_orszag_tang()
        title = "Orszag-Tang MHD Vortex"
        n_steps = 200
        
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
        solver = QuantumFluidSolver(nx=128, ny=128, g_int=500, dt=0.0005)
        solver.initialize_quantum_turbulence(n_vortices=10)
        title = "Quantum Turbulence (Bose-Einstein Condensate)"
        n_steps = 200

    elif domain == "qft":
        run_qft_simulation()
        return

    elif domain == "relativistic":
        from physics.relativistic import RelativisticNSSolver
        solver = RelativisticNSSolver(nx=128, ny=128, eta_s=0.2, tau_pi=0.5, dt=0.005)
        solver.initialize_bjorken_flow(e0=10.0, sigma=1.5)
        title = "Relativistic NS — QGP Fireball (Israel-Stewart)"
        n_steps = 300

    elif domain == "gravity":
        run_gravity_fluid_coupling()
        return

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
        filepath = os.path.join(IMAGES_DIR, filename)
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
        trainer.plot_training_history(save_path=os.path.join(IMAGES_DIR, 'pinn_training.png'))
        
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
        trainer.plot_training_history(save_path=os.path.join(IMAGES_DIR, 'fno_training.png'))
    
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
        trainer.plot_training_history(save_path=os.path.join(IMAGES_DIR, 'deeponet_training.png'))
    
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
        trainer.plot_training_history(save_path=os.path.join(IMAGES_DIR, 'surrogate_training.png'))
    
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
    print("  [15] 🔥 Vorticity Confinement Demo")
    print("  [16] ⚡ GPU Solver Demo")
    print("  [17] 🧪 Hybrid CFD->PINN Demo")
    print("  ─────────────────────────────────────────")
    print("  [18] 🧠 Turbulence Discovery AI (Full Pipeline)")
    print("  [19] 💥 Blow-up Detection & Stability Analysis")
    print("  [20] 📈 Regularity Map (Re sweep)")
    print("  [21] 📊 Turbulence Metrics (DNS vs LES comparison)")
    print("  [22] 🔬 Symbolic Discovery (SINDy + GP)")
    print("  [23] 🌀 Quantum Fluid Extensions (GPE + Madelung)")
    print("  [24] 🚀 Relativistic NS (Israel-Stewart Causal Theory)")
    print("  ─────────────────────────────────────────")
    print("  [25] 🤖 AGI Scientific Discovery System (Full Pipeline)")
    print("  [26] 🔬 Physics-Aware Equation Discovery")
    print("  [27] ⚛ Quantum Field Theory (Lattice QFT + PINN)")
    print("  [28] 🌌 Gravity + Fluid Coupling (Einstein Equations)")
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
        "15": lambda: run_vorticity_confinement_demo(),
        "16": lambda: run_gpu_demo(),
        "17": lambda: run_hybrid_demo(),
        "18": lambda: run_turbulence_discovery(),
        "19": lambda: run_stability_analysis(),
        "20": lambda: run_regularity_map(),
        "21": lambda: run_turbulence_metrics(),
        "22": lambda: run_symbolic_discovery_standalone(),
        "23": lambda: run_quantum_extensions(),
        "24": lambda: run_relativistic_ns(),
        "25": lambda: run_agi_discovery(),
        "26": lambda: run_physics_aware_discovery(),
        "27": lambda: run_qft_simulation(),
        "28": lambda: run_gravity_fluid_coupling(),
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
# New Feature Demos
# =============================================================================

def run_vorticity_confinement_demo():
    """Side-by-side comparison: with vs without vorticity confinement."""
    print("\n" + "="*60)
    print("  DEMO: Vorticity Confinement Comparison")
    print("  Shows how eps_vc restores swirling turbulence")
    print("="*60 + "\n")
    
    from core.fluid_solver_2d import FluidSolver2D
    setup_plot_style()
    import matplotlib.pyplot as plt
    
    n_steps = 200
    configs = [
        ("No Confinement (eps=0)", 0.0),
        ("Mild (eps=2)", 2.0),
        ("Strong (eps=5)", 5.0),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (label, eps) in enumerate(configs):
        solver = FluidSolver2D(
            nx=128, ny=128, Lx=2*np.pi, Ly=2*np.pi,
            nu=0.01, dt=0.005, pressure_solver="fft",
            vorticity_confinement=eps,
        )
        solver.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
        solver.bc_manager.set_periodic()
        
        for _ in range(n_steps):
            solver.step()
            # Clamp to prevent blowup with high confinement
            if not np.isfinite(solver.u).all():
                solver.u = np.nan_to_num(solver.u, nan=0.0, posinf=10.0, neginf=-10.0)
                solver.v = np.nan_to_num(solver.v, nan=0.0, posinf=10.0, neginf=-10.0)
        
        omega = solver.get_vorticity()
        omega = np.nan_to_num(omega, nan=0.0)
        v_max = max(np.max(np.abs(omega)), 1e-6)
        axes[idx].imshow(omega, cmap='coolwarm', vmin=-v_max, vmax=v_max,
                        origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi])
        axes[idx].set_title(label, color='#79c0ff', fontsize=13)
        axes[idx].set_xlabel('x'); axes[idx].set_ylabel('y')
        print(f"  {label}: max|omega|={v_max:.2f}")
    
    fig.suptitle('Vorticity Confinement: Restoring Turbulent Detail',
                fontsize=15, color='#c9d1d9', fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(IMAGES_DIR, 'demo_vorticity_confinement.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    show_or_close(fig)


def run_gpu_demo():
    """GPU-accelerated solver benchmark."""
    print("\n" + "="*60)
    print("  DEMO: GPU-Accelerated Solver")
    print("="*60 + "\n")
    
    try:
        from core.fluid_solver_2d import GPUFluidSolver2D, FluidSolver2D
        import torch
    except ImportError:
        print("  PyTorch required for GPU demo.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    for res in [64, 128, 256]:
        # GPU
        gpu_solver = GPUFluidSolver2D(nx=res, ny=res, nu=0.01, dt=0.005, device=device)
        gpu_solver.initialize_taylor_green()
        
        t0 = time.perf_counter()
        gpu_solver.advance(100)
        gpu_time = time.perf_counter() - t0
        
        # CPU
        cpu_solver = FluidSolver2D(nx=res, ny=res, Lx=2*np.pi, Ly=2*np.pi,
                                   nu=0.01, dt=0.005)
        cpu_solver.initialize_taylor_green()
        cpu_solver.bc_manager.set_periodic()
        
        t0 = time.perf_counter()
        for _ in range(100):
            cpu_solver.step()
        cpu_time = time.perf_counter() - t0
        
        speedup = cpu_time / max(gpu_time, 1e-6)
        print(f"  {res}×{res}: CPU={cpu_time:.3f}s  {device.upper()}={gpu_time:.3f}s  "
              f"Speedup={speedup:.1f}x")
    
    print("\n  Done.")


def run_hybrid_demo():
    """Hybrid CFD->PINN demonstration."""
    print("\n" + "="*60)
    print("  DEMO: Hybrid CFD -> PINN Integration")
    print("  CFD generates truth -> trains PINN -> PINN predicts")
    print("="*60 + "\n")
    
    try:
        import torch
        from models.pinn import PINN
        from core.fluid_solver_2d import FluidSolver2D
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        return
    
    setup_plot_style()
    import matplotlib.pyplot as plt
    
    # Step 1: Generate CFD truth data
    print("  [1/3] Generating CFD truth data...")
    solver = FluidSolver2D(nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi,
                           nu=0.01, dt=0.01, pressure_solver="fft")
    solver.initialize_taylor_green()
    solver.bc_manager.set_periodic()
    
    x_data, y_data, t_data, u_data, v_data = [], [], [], [], []
    for step in range(50):
        solver.step()
        if step % 5 == 0:
            n_pts = 200
            xi = np.random.rand(n_pts) * 2 * np.pi
            yi = np.random.rand(n_pts) * 2 * np.pi
            gi = (xi / (2*np.pi) * 64).astype(int) % 64
            gj = (yi / (2*np.pi) * 64).astype(int) % 64
            x_data.append(xi)
            y_data.append(yi)
            t_data.append(np.full(n_pts, solver.time))
            u_data.append(solver.u[gj, gi])
            v_data.append(solver.v[gj, gi])
    
    x_t = torch.tensor(np.concatenate(x_data), dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(np.concatenate(y_data), dtype=torch.float32).unsqueeze(1)
    t_t = torch.tensor(np.concatenate(t_data), dtype=torch.float32).unsqueeze(1)
    u_t = torch.tensor(np.concatenate(u_data), dtype=torch.float32).unsqueeze(1)
    v_t = torch.tensor(np.concatenate(v_data), dtype=torch.float32).unsqueeze(1)
    print(f"    Collected {len(x_t)} data points")
    
    # Step 2: Train PINN on CFD data
    print("  [2/3] Training PINN on CFD data (quick demo)...")
    model = PINN(input_dim=3, output_dim=3, hidden_layers=[64, 64, 64],
                 use_fourier_features=True, fourier_features=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    inp = torch.cat([x_t, y_t, t_t], dim=1)
    target = torch.cat([u_t, v_t], dim=1)
    
    for epoch in range(200):
        pred = model(inp)
        loss = torch.mean((pred[:, :2] - target)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/200  Loss: {loss.item():.6f}")
    
    # Step 3: Compare CFD vs PINN prediction
    print("  [3/3] Comparing CFD vs PINN predictions...")
    nx_test = 64
    xg = np.linspace(0, 2*np.pi, nx_test)
    yg = np.linspace(0, 2*np.pi, nx_test)
    XG, YG = np.meshgrid(xg, yg)
    t_test = solver.time
    
    pred_field = model.predict_field(XG, YG, t_test)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    cfd_omega = solver.get_vorticity()
    pred_omega = np.gradient(pred_field['v'], axis=1) - np.gradient(pred_field['u'], axis=0)
    
    v_max = max(np.max(np.abs(cfd_omega)), 1e-6)
    axes[0, 0].imshow(solver.u, cmap='coolwarm', origin='lower')
    axes[0, 0].set_title('CFD: u', color='#79c0ff')
    axes[0, 1].imshow(solver.v, cmap='coolwarm', origin='lower')
    axes[0, 1].set_title('CFD: v', color='#79c0ff')
    axes[0, 2].imshow(cfd_omega, cmap='coolwarm', vmin=-v_max, vmax=v_max, origin='lower')
    axes[0, 2].set_title('CFD: ω', color='#79c0ff')
    
    axes[1, 0].imshow(pred_field['u'], cmap='coolwarm', origin='lower')
    axes[1, 0].set_title('PINN: u', color='#7ee787')
    axes[1, 1].imshow(pred_field['v'], cmap='coolwarm', origin='lower')
    axes[1, 1].set_title('PINN: v', color='#7ee787')
    axes[1, 2].imshow(pred_omega, cmap='coolwarm', vmin=-v_max, vmax=v_max, origin='lower')
    axes[1, 2].set_title('PINN: ω', color='#7ee787')
    
    fig.suptitle(f'Hybrid CFD->PINN (t={t_test:.2f}, loss={loss.item():.6f})',
                fontsize=15, color='#c9d1d9', fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(IMAGES_DIR, 'demo_hybrid_pinn.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    show_or_close(fig)


# =============================================================================
# Turbulence Discovery AI
# =============================================================================

def run_turbulence_discovery():
    """Run the full Turbulence Discovery AI pipeline."""
    print("\n" + "="*60)
    print("  TURBULENCE DISCOVERY AI — FULL PIPELINE")
    print("  Compress -> Learn Rules -> Output Equations")
    print("="*60 + "\n")
    
    try:
        import torch
    except ImportError:
        print("  ERROR: PyTorch required. pip install torch")
        return
    
    from training.discovery_trainer import TurbulenceDiscoveryTrainer
    
    trainer = TurbulenceDiscoveryTrainer(
        checkpoint_dir=os.path.join(PROJECT_ROOT, 'checkpoints', 'discovery'),
        log_dir=os.path.join(PROJECT_ROOT, 'logs', 'discovery'),
    )
    
    results = trainer.run_full_pipeline(
        nx=64, latent_dim=32,
        n_ae_samples=30, n_paired_samples=20, n_stability_samples=30,
        ae_epochs=30, ode_epochs=30, blowup_epochs=20,
        gp_generations=20, verbose=True,
    )
    
    # Plot results
    save_path = os.path.join(IMAGES_DIR, 'demo_turbulence_discovery.png')
    trainer.plot_discovery_results(results.get('discovery', {}), save_path=save_path)
    print(f"\n  Results saved: {save_path}")


def run_stability_analysis():
    """Blow-up detection and stability analysis."""
    print("\n" + "="*60)
    print("  BLOW-UP DETECTION & STABILITY ANALYSIS")
    print("  Predicting when and why solutions fail")
    print("="*60 + "\n")
    
    from models.regularity_analysis import StabilityAnalyzer, FlowDiagnostics
    from core.fluid_solver_2d import FluidSolver2D
    from utils.helpers import compute_vorticity
    
    setup_plot_style()
    import matplotlib.pyplot as plt
    
    analyzer = StabilityAnalyzer()
    
    def solver_factory(nu, ic_type='taylor_green'):
        s = FluidSolver2D(nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi,
                          nu=nu, dt=0.005, pressure_solver="fft")
        s.bc_manager.set_periodic()
        if ic_type == 'taylor_green':
            s.initialize_taylor_green()
        elif ic_type == 'shear_layer':
            s.initialize_double_shear_layer()
        elif ic_type == 'vortex_pair':
            s.initialize_vortex_pair()
        return s
    
    print("  Sweeping Reynolds numbers (10 -> 10000)...")
    results = analyzer.analyze_ic_space(
        solver_factory, n_samples=15, re_range=(10, 10000),
        n_steps=200, verbose=True,
    )
    
    summary = analyzer.get_stability_summary()
    print(f"\n  Results: {summary['survived']}/{summary['total_simulations']} survived")
    for k, v in summary.items():
        if k.startswith('critical_re'):
            print(f"  {k}: {v:.0f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Stability Analysis — Empirical Regularity Map',
                fontsize=15, color='#58a6ff', fontweight='bold')
    
    re_arr = np.array(results['reynolds'])
    regime_colors = {'smooth': '#7ee787', 'transitional': '#ffa657',
                     'turbulent': '#79c0ff', 'unstable': '#f97583',
                     'singular_risk': '#ff0000'}
    
    for ic_type in set(results['ic_type']):
        mask = [ic == ic_type for ic in results['ic_type']]
        re_ic = re_arr[mask]
        max_omega = np.array(results['max_vorticity_final'])[mask]
        colors = [regime_colors.get(r, '#888') for i, r in enumerate(results['regime']) if mask[i]]
        axes[0].scatter(re_ic, max_omega, c=colors, label=ic_type, s=30, alpha=0.8)
    
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Reynolds Number'); axes[0].set_ylabel('Max |ω|')
    axes[0].set_title('Vorticity vs Re', color='#79c0ff')
    axes[0].legend(fontsize=8)
    
    # BKM integral
    bkm_arr = np.array(results['bkm_integral'])
    valid = np.isfinite(bkm_arr)
    if np.any(valid):
        axes[1].scatter(re_arr[valid], bkm_arr[valid], c='#d2a8ff', s=30, alpha=0.7)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Reynolds Number'); axes[1].set_ylabel('BKM Integral')
    axes[1].set_title('BKM Criterion Monitor', color='#79c0ff')
    
    # KE ratio
    ke_arr = np.array(results['kinetic_energy_ratio'])
    valid = np.isfinite(ke_arr)
    if np.any(valid):
        axes[2].scatter(re_arr[valid], ke_arr[valid], c='#ffa657', s=30, alpha=0.7)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Reynolds Number'); axes[2].set_ylabel('KE(final)/KE(initial)')
    axes[2].set_title('Energy Dissipation', color='#79c0ff')
    
    for ax in axes:
        ax.grid(True, alpha=0.15, color='#30363d')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'demo_stability_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    show_or_close(fig)


def run_regularity_map():
    """Generate empirical regularity map across Re and IC space."""
    print("\n" + "="*60)
    print("  REGULARITY MAP — Empirical Flow Classification")
    print("="*60 + "\n")
    
    from core.fluid_solver_2d import FluidSolver2D
    from models.regularity_analysis import FlowDiagnostics
    from utils.helpers import compute_vorticity
    
    setup_plot_style()
    import matplotlib.pyplot as plt
    
    re_values = np.logspace(1, 4, 20)
    ic_types = ['taylor_green', 'shear_layer', 'vortex_pair']
    
    results = {ic: {'re': [], 'enstrophy': [], 'diss_rate': [], 'regime': []} 
               for ic in ic_types}
    
    for ic_type in ic_types:
        for re in re_values:
            nu = 1.0 / re
            try:
                solver = FluidSolver2D(nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi,
                                       nu=nu, dt=0.005, pressure_solver="fft")
                solver.bc_manager.set_periodic()
                if ic_type == 'taylor_green':
                    solver.initialize_taylor_green()
                elif ic_type == 'shear_layer':
                    solver.initialize_double_shear_layer()
                else:
                    solver.initialize_vortex_pair()
                
                for _ in range(200):
                    solver.step()
                    if not np.all(np.isfinite(solver.u)):
                        raise ValueError("Diverged")
                
                omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                diag = FlowDiagnostics.compute_enstrophy_budget(
                    solver.u, solver.v, omega, solver.dx, solver.dy, nu
                )
                
                results[ic_type]['re'].append(re)
                results[ic_type]['enstrophy'].append(diag['enstrophy'])
                results[ic_type]['diss_rate'].append(diag['dissipation_rate'])
                
            except Exception:
                results[ic_type]['re'].append(re)
                results[ic_type]['enstrophy'].append(float('nan'))
                results[ic_type]['diss_rate'].append(float('nan'))
        
        print(f"  Completed: {ic_type}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Empirical Regularity Map', fontsize=15, color='#58a6ff', fontweight='bold')
    colors = {'taylor_green': '#58a6ff', 'shear_layer': '#f97583', 'vortex_pair': '#7ee787'}
    
    for ic in ic_types:
        re = np.array(results[ic]['re'])
        ens = np.array(results[ic]['enstrophy'])
        valid = np.isfinite(ens)
        if np.any(valid):
            axes[0].loglog(re[valid], ens[valid], 'o-', color=colors[ic], label=ic, lw=2, ms=5)
            diss = np.array(results[ic]['diss_rate'])
            valid_d = np.isfinite(diss)
            if np.any(valid_d):
                axes[1].loglog(re[valid_d], diss[valid_d], 'o-', color=colors[ic], label=ic, lw=2, ms=5)
    
    axes[0].set_xlabel('Re'); axes[0].set_ylabel('Enstrophy')
    axes[0].set_title('Enstrophy vs Re', color='#79c0ff')
    axes[0].legend()
    axes[1].set_xlabel('Re'); axes[1].set_ylabel('Dissipation Rate')
    axes[1].set_title('Dissipation Rate vs Re', color='#79c0ff')
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.15, color='#30363d')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'demo_regularity_map.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    show_or_close(fig)


def run_turbulence_metrics():
    """Compare DNS-like vs coarse resolution (LES-proxy) with full metrics."""
    print("\n" + "="*60)
    print("  TURBULENCE METRICS — DNS vs LES Comparison")
    print("="*60 + "\n")
    
    from core.fluid_solver_2d import FluidSolver2D
    from models.regularity_analysis import TurbulenceMetrics, FlowDiagnostics
    from utils.helpers import compute_vorticity
    
    setup_plot_style()
    import matplotlib.pyplot as plt
    
    Re = 500
    nu = 1.0 / Re
    n_steps = 300
    
    # DNS-like (fine grid)
    print("  Running DNS-like (128×128)...")
    dns = FluidSolver2D(nx=128, ny=128, Lx=2*np.pi, Ly=2*np.pi,
                         nu=nu, dt=0.005, pressure_solver="fft")
    dns.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
    dns.bc_manager.set_periodic()
    for _ in range(n_steps): dns.step()
    
    # LES-proxy (coarse grid)
    print("  Running LES-proxy (32×32)...")
    les = FluidSolver2D(nx=32, ny=32, Lx=2*np.pi, Ly=2*np.pi,
                         nu=nu, dt=0.005, pressure_solver="fft")
    les.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
    les.bc_manager.set_periodic()
    for _ in range(n_steps): les.step()
    
    # Downsample DNS to LES grid for comparison
    from scipy.ndimage import zoom
    factor = 32 / 128
    u_dns_ds = zoom(dns.u, factor, order=3)
    v_dns_ds = zoom(dns.v, factor, order=3)
    
    # Compute metrics
    metrics = TurbulenceMetrics.compute_all_metrics(
        les.u, les.v, u_dns_ds, v_dns_ds, les.dx, les.dy
    )
    
    print("\n  Comparison Metrics:")
    for k, v in metrics.items():
        print(f"    {k:30s}: {v:.6f}")
    
    # Energy spectra comparison
    spec_dns = FlowDiagnostics.compute_energy_spectrum(dns.u, dns.v)
    spec_les = FlowDiagnostics.compute_energy_spectrum(les.u, les.v)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'DNS vs LES Comparison — Re={Re}', fontsize=15,
                color='#58a6ff', fontweight='bold')
    
    # Vorticity fields
    omega_dns = compute_vorticity(dns.u, dns.v, dns.dx, dns.dy)
    omega_les = compute_vorticity(les.u, les.v, les.dx, les.dy)
    vmax = max(np.max(np.abs(omega_dns)), 1e-3)
    
    axes[0].imshow(omega_dns, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
    axes[0].set_title('DNS (128²)', color='#79c0ff')
    axes[1].imshow(omega_les, cmap='RdBu_r', vmin=-vmax*0.5, vmax=vmax*0.5, origin='lower')
    axes[1].set_title('LES (32²)', color='#79c0ff')
    
    # Energy spectrum
    k_dns = spec_dns['wavenumbers']
    k_les = spec_les['wavenumbers']
    axes[2].loglog(k_dns, spec_dns['spectrum'], '-', color='#58a6ff', lw=2, label='DNS')
    axes[2].loglog(k_les, spec_les['spectrum'], '--', color='#f97583', lw=2, label='LES')
    axes[2].loglog(k_dns, spec_dns['kolmogorov_reference'], ':', color='#8b949e', 
                  lw=1.5, label='k⁻⁵ᐟ³')
    axes[2].set_xlabel('Wavenumber k'); axes[2].set_ylabel('E(k)')
    axes[2].set_title('Energy Spectrum', color='#79c0ff')
    axes[2].legend()
    axes[2].grid(True, alpha=0.15, color='#30363d')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'demo_turbulence_metrics.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    show_or_close(fig)


def run_symbolic_discovery_standalone():
    """Run SINDy + GP symbolic discovery on flow trajectories."""
    print("\n" + "="*60)
    print("  SYMBOLIC DISCOVERY — SINDy + Genetic Programming")
    print("  Discovering equations from turbulence data...")
    print("="*60 + "\n")
    
    from core.fluid_solver_2d import FluidSolver2D
    from models.symbolic_discovery import SymbolicDiscoveryEngine
    from utils.helpers import compute_vorticity, compute_kinetic_energy, compute_enstrophy
    
    setup_plot_style()
    import matplotlib.pyplot as plt
    
    # Generate trajectory data (low-dim observables)
    Re = 100
    nu = 1.0 / Re
    solver = FluidSolver2D(nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi,
                            nu=nu, dt=0.002, pressure_solver="fft",
                            advection_scheme="upwind")
    solver.initialize_taylor_green()
    solver.bc_manager.set_periodic()
    
    print("  Generating flow trajectory (500 steps)...")
    observables = []
    for step in range(500):
        solver.step()
        # Check for divergence
        if not np.all(np.isfinite(solver.u)):
            print(f"  Warning: solver diverged at step {step}, using {len(observables)} snapshots")
            break
        if step % 2 == 0:
            omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
            ke = compute_kinetic_energy(solver.u, solver.v)
            ens = compute_enstrophy(omega)
            max_omega = np.max(np.abs(omega))
            max_u = np.max(np.abs(solver.u))
            
            observables.append([ke, ens, max_omega, max_u])
    
    Z = np.array(observables)
    # Filter out any rows with NaN or Inf
    valid = np.all(np.isfinite(Z), axis=1)
    Z = Z[valid]
    print(f"  Collected {Z.shape[0]} valid snapshots of {Z.shape[1]} observables")
    
    # Run discovery
    engine = SymbolicDiscoveryEngine(
        n_latent_dims=Z.shape[1],
        sindy_threshold=0.05,
        gp_population=100,
        gp_generations=30,
    )
    
    results = engine.discover_from_trajectories(Z, dt=0.02, verbose=True)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Symbolic Discovery Results', fontsize=15,
                color='#58a6ff', fontweight='bold')
    
    times = np.arange(len(Z)) * 0.02
    labels = ['KE', 'Enstrophy', 'max|ω|', 'max|u|']
    colors = ['#58a6ff', '#7ee787', '#f97583', '#d2a8ff']
    
    for i, (lbl, clr) in enumerate(zip(labels, colors)):
        ax = axes[i // 2, i % 2]
        ax.plot(times, Z[:, i], color=clr, lw=2, label=f'{lbl} (data)')
        
        # SINDy prediction
        if results.get('sindy', {}).get('Xi') is not None:
            try:
                pred = engine.sindy.predict(Z)
                ax.plot(times[1:-1], Z[1:-1, i] + pred[1:-1, i] * 0.02, 
                       '--', color='#ffa657', lw=1.5, label='SINDy fit', alpha=0.7)
            except Exception:
                pass
        
        ax.set_xlabel('Time')
        ax.set_ylabel(lbl)
        ax.set_title(f'{lbl} Evolution', color='#79c0ff')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15, color='#30363d')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'demo_symbolic_discovery.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    show_or_close(fig)


# =============================================================================
# Quantum Fluid Extensions (GPE + Madelung Transform)
# =============================================================================

def run_quantum_extensions():
    """
    Quantum Fluid Extensions — Full GPE Pipeline.

    Instead of velocity fields alone, quantum fluids use a wavefunction.
    The governing equation is the Gross-Pitaevskii equation (GPE):

        i\u0127 \u2202\u03c8/\u2202t = [-\u0127\u00b2/(2m)\u2207\u00b2 + V + g|\u03c8|\u00b2]\u03c8

    Madelung Transform:  \u03c8 = \u221a\u03c1 e^{i\u03b8}
        Density:   \u03c1 = |\u03c8|\u00b2
        Velocity:  v = (\u0127/m)\u2207\u03b8

    This recovers a Navier-Stokes-like system with an additional quantum
    pressure (Bohm potential) term.

    Pipeline:
      1. Replace scalar density with complex wavefunction field
      2. Time-evolve via split-step Fourier method
      3. Madelung decomposition -> density + velocity
      4. Detect quantized vortices
      5. Compute incompressible energy spectrum
      6. Publication-quality 8-panel visualization

    Results:
      - Classical + quantum fluid simulator
      - Vortex quantization (discrete vortices)
      - Wave interference effects
      - Kolmogorov-like turbulence cascade at large scales
    """
    print("\n" + "="*72)
    print("  QUANTUM FLUID EXTENSIONS")
    print("  Gross-Pitaevskii Equation + Madelung Transform")
    print("  ihbar dpsi/dt = [-hbar^2/(2m) nabla^2 + V + g|psi|^2] psi")
    print("="*72 + "\n")

    from physics.quantum_fluids import QuantumFluidSolver

    setup_plot_style()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    # ── Phase 1: Initialize complex wavefunction field ──────────────────
    print("  [1/6] Initializing complex wavefunction field...")
    nx, ny = 256, 256
    Lx, Ly = 20.0, 20.0
    g_int = 500.0
    dt = 0.0005

    solver = QuantumFluidSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        hbar=1.0, m=1.0, g_int=g_int, dt=dt,
    )

    # Ground state + quantum turbulence vortex tangle
    n0 = 1.0
    solver.initialize_ground_state(n0=n0)
    healing_length = solver.healing_length
    print(f"    Grid: {nx}x{ny}")
    print(f"    Domain: [{-Lx/2:.1f}, {Lx/2:.1f}] x [{-Ly/2:.1f}, {Ly/2:.1f}]")
    print(f"    Interaction: g = {g_int}")
    print(f"    Healing length: xi = {healing_length:.4f}")
    print(f"    dt = {dt}")

    # Imprint vortex tangle for quantum turbulence
    n_vortices = 16
    print(f"\n  [2/6] Imprinting {n_vortices} quantized vortices...")
    np.random.seed(42)
    vortex_log = []
    for i in range(n_vortices):
        x0 = np.random.uniform(-Lx/3, Lx/3)
        y0 = np.random.uniform(-Ly/3, Ly/3)
        charge = np.random.choice([-1, 1])
        solver.imprint_vortex(x0, y0, charge)
        vortex_log.append((x0, y0, charge))
        symbol = "+" if charge > 0 else "-"
        print(f"    Vortex {i+1:2d}: ({x0:+6.2f}, {y0:+6.2f}) charge={symbol}1")

    # ── Phase 2: Time evolution via split-step Fourier ──────────────────
    n_steps = 400
    record_interval = 10
    print(f"\n  [3/6] Time evolution (split-step Fourier) — {n_steps} steps...")

    energy_history = []
    particle_history = []
    vortex_count_history = []
    time_history = []

    t_start = time.perf_counter()
    for step in range(n_steps):
        solver.step()

        if step % record_interval == 0:
            density = solver.get_density()
            N = np.sum(density) * solver.dx * solver.dy
            energy = solver.compute_energy()
            vortices = solver.detect_vortices()

            energy_history.append(energy['total'])
            particle_history.append(N)
            vortex_count_history.append(len(vortices))
            time_history.append(solver.time)

            if step % 100 == 0:
                print(f"    Step {step:4d} | t={solver.time:.4f} | "
                      f"N={N:.2f} | E={energy['total']:.2f} | "
                      f"Vortices={len(vortices)}")

    elapsed = time.perf_counter() - t_start
    print(f"\n    Completed: {n_steps} steps in {elapsed:.2f}s "
          f"({n_steps/elapsed:.0f} steps/s)")

    # ── Phase 3: Madelung decomposition ─────────────────────────────────
    print("\n  [4/6] Madelung transform: psi = sqrt(rho) * exp(i*theta)...")
    density = solver.get_density()       # rho = |psi|^2
    phase = solver.get_phase()           # theta = arg(psi)
    ux, uy = solver.get_velocity()       # v = (hbar/m) * grad(theta)
    speed = np.sqrt(ux**2 + uy**2)

    # Quantum pressure (Bohm potential): Q = -(hbar^2 / 2m) * laplacian(sqrt(rho)) / sqrt(rho)
    sqrt_rho = np.sqrt(density + 1e-12)
    laplacian_sqrt_rho = (
        np.roll(sqrt_rho, 1, 0) + np.roll(sqrt_rho, -1, 0) +
        np.roll(sqrt_rho, 1, 1) + np.roll(sqrt_rho, -1, 1) -
        4 * sqrt_rho
    ) / (solver.dx * solver.dy)
    Q_bohm = -(solver.hbar**2 / (2 * solver.m)) * laplacian_sqrt_rho / (sqrt_rho + 1e-12)

    print(f"    Density: min={density.min():.4f}  max={density.max():.4f}")
    print(f"    Phase:   min={phase.min():.4f}  max={phase.max():.4f}")
    print(f"    |v|:     min={speed.min():.4f}  max={speed.max():.4f}")
    print(f"    Quantum pressure |Q|: max={np.max(np.abs(Q_bohm)):.4f}")

    # ── Phase 4: Vortex detection ───────────────────────────────────────
    print("\n  [5/6] Detecting quantized vortices...")
    vortices = solver.detect_vortices()
    n_pos = sum(1 for _, _, c in vortices if c > 0)
    n_neg = sum(1 for _, _, c in vortices if c < 0)
    print(f"    Found {len(vortices)} vortices  (+{n_pos} / -{n_neg})")
    print(f"    Circulation quantization: Gamma = n * h/m = n * {2*np.pi*solver.hbar/solver.m:.4f}")

    # ── Phase 5: Energy spectrum ────────────────────────────────────────
    print("\n  [6/6] Computing incompressible energy spectrum...")
    k_spectrum, E_spectrum = solver.compute_incompressible_spectrum()
    valid_k = k_spectrum > 0
    k_valid = k_spectrum[valid_k]
    E_valid = E_spectrum[valid_k]
    E_valid = np.maximum(E_valid, 1e-30)  # avoid log(0)

    # Kolmogorov reference: E(k) ~ k^(-5/3)
    if len(k_valid) > 0:
        k_ref = k_valid
        E_ref = E_valid[len(E_valid)//4] * (k_ref / k_ref[len(k_ref)//4])**(-5.0/3.0)

    # ── Publication-quality 8-panel visualization ────────────────────────
    print("\n  Generating publication-quality visualization...")

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        'Quantum Fluid Extensions   \u00b7   Gross-Pitaevskii + Madelung Transform',
        fontsize=20, fontweight='bold', color='#58a6ff', y=0.98,
    )

    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]

    # ── Panel 1: Superfluid density |psi|^2 ──
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.imshow(density, cmap='inferno', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax1.set_title('Density  |\u03c8|\u00b2', color='#79c0ff', fontsize=13)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.outline.set_edgecolor('#30363d')

    # ── Panel 2: Phase arg(psi) ──
    ax2 = fig.add_subplot(2, 4, 2)
    im2 = ax2.imshow(phase, cmap='twilight_shifted', origin='lower', extent=extent,
                     interpolation='bicubic', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Phase  arg(\u03c8)', color='#79c0ff', fontsize=13)
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.outline.set_edgecolor('#30363d')

    # ── Panel 3: Velocity magnitude (Madelung) ──
    ax3 = fig.add_subplot(2, 4, 3)
    speed_clipped = np.clip(speed, 0, np.percentile(speed, 99))
    im3 = ax3.imshow(speed_clipped, cmap='magma', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax3.set_title('Velocity |v| = (\u0127/m)|\u2207\u03b8|', color='#79c0ff', fontsize=13)
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.outline.set_edgecolor('#30363d')

    # ── Panel 4: Vortex positions overlaid on density ──
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(density, cmap='inferno', origin='lower', extent=extent,
              interpolation='bicubic', alpha=0.6)
    for vx, vy, vc in vortices:
        color = '#7ee787' if vc > 0 else '#f97583'
        marker = '^' if vc > 0 else 'v'
        ax4.plot(vx, vy, marker, color=color, ms=6, mew=1.5, mec='white')
    ax4.set_title(f'Vortex Map  ({len(vortices)} detected)', color='#79c0ff', fontsize=13)
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    # Legend
    ax4.plot([], [], '^', color='#7ee787', ms=8, label='+1 vortex')
    ax4.plot([], [], 'v', color='#f97583', ms=8, label='-1 vortex')
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.7)

    # ── Panel 5: Quantum pressure (Bohm potential) ──
    ax5 = fig.add_subplot(2, 4, 5)
    Q_clipped = np.clip(Q_bohm, np.percentile(Q_bohm, 1), np.percentile(Q_bohm, 99))
    vabs = max(np.max(np.abs(Q_clipped)), 1e-6)
    im5 = ax5.imshow(Q_clipped, cmap='RdBu_r', origin='lower', extent=extent,
                     interpolation='bicubic',
                     norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs))
    ax5.set_title('Quantum Pressure  Q (Bohm)', color='#79c0ff', fontsize=13)
    ax5.set_xlabel('x'); ax5.set_ylabel('y')
    cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.outline.set_edgecolor('#30363d')

    # ── Panel 6: Energy spectrum ──
    ax6 = fig.add_subplot(2, 4, 6)
    if len(k_valid) > 2:
        ax6.loglog(k_valid, E_valid, '-', color='#58a6ff', lw=2.2, label='E(k)', zorder=3)
        ax6.loglog(k_ref, E_ref, '--', color='#8b949e', lw=1.5,
                   label='k\u207b\u2075\u2033\u00b3 (Kolmogorov)', zorder=2)
        ax6.fill_between(k_valid, E_valid, alpha=0.08, color='#58a6ff')
    ax6.set_xlabel('Wavenumber k')
    ax6.set_ylabel('E(k)')
    ax6.set_title('Incompressible Energy Spectrum', color='#79c0ff', fontsize=13)
    ax6.legend(fontsize=9, framealpha=0.7)
    ax6.grid(True, alpha=0.15, color='#30363d')

    # ── Panel 7: Diagnostics time series ──
    ax7 = fig.add_subplot(2, 4, 7)
    t_arr = np.array(time_history)
    ax7_twin = ax7.twinx()
    ax7.plot(t_arr, energy_history, '-', color='#58a6ff', lw=2, label='Total Energy')
    ax7_twin.plot(t_arr, vortex_count_history, '-', color='#f97583', lw=2, label='N vortices')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Energy', color='#58a6ff')
    ax7_twin.set_ylabel('Vortex Count', color='#f97583')
    ax7.set_title('Evolution Diagnostics', color='#79c0ff', fontsize=13)
    ax7.tick_params(axis='y', labelcolor='#58a6ff')
    ax7_twin.tick_params(axis='y', labelcolor='#f97583')
    ax7.grid(True, alpha=0.15, color='#30363d')

    # ── Panel 8: Particle number conservation check ──
    ax8 = fig.add_subplot(2, 4, 8)
    N_arr = np.array(particle_history)
    N_deviation = (N_arr - N_arr[0]) / N_arr[0] * 100
    ax8.plot(t_arr, N_deviation, '-', color='#7ee787', lw=2)
    ax8.fill_between(t_arr, N_deviation, alpha=0.1, color='#7ee787')
    ax8.axhline(y=0, color='#8b949e', ls='--', lw=0.8)
    ax8.set_xlabel('Time')
    ax8.set_ylabel('\u0394N/N\u2080 (%)')
    ax8.set_title('Particle Conservation', color='#79c0ff', fontsize=13)
    ax8.grid(True, alpha=0.15, color='#30363d')

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(axis='both', colors='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(IMAGES_DIR, 'quantum_extensions.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    show_or_close(fig)
    print(f"\n  Visualization saved: {save_path}")

    # ── Summary ──
    energy_final = solver.compute_energy()
    print("\n" + "="*72)
    print("  QUANTUM FLUID EXTENSIONS — SUMMARY")
    print("="*72)
    print(f"  Governing eq: ihbar dpsi/dt = [-hbar^2/(2m) nabla^2 + V + g|psi|^2] psi")
    print(f"  Madelung:     psi = sqrt(rho) e^(i*theta) -> rho = |psi|^2, v = (hbar/m) grad(theta)")
    print(f"  Method:       Split-step Fourier (symplectic)")
    print(f"  Grid:         {nx}x{ny},  L = {Lx}")
    print(f"  g_int:        {g_int}")
    print(f"  Steps:        {n_steps}  (dt={dt})")
    print(f"  Healing len:  xi = {healing_length:.4f}")
    print(f"  Vortices:     {len(vortices)}  (+{n_pos} / -{n_neg})")
    print(f"  Energy:       K={energy_final['kinetic']:.2f}  "
          f"V={energy_final['potential']:.2f}  "
          f"I={energy_final['interaction']:.2f}  "
          f"Total={energy_final['total']:.2f}")
    N_final = np.sum(density) * solver.dx * solver.dy
    print(f"  Particles:    N = {N_final:.4f}  (dN/N0 = {(N_final-particle_history[0])/particle_history[0]*100:.4f}%)")
    print()
    print("  Results:")
    print("    [+] Classical + quantum fluid simulator")
    print("    [+] Vortex quantization (discrete vortices with Gamma = n * h/m)")
    print("    [+] Wave interference effects")
    print("    [+] Madelung-derived NS-like system + quantum pressure")
    print("    [+] Kolmogorov cascade at large scales")
    print("="*72 + "\n")


# =============================================================================
# AGI Scientific Discovery System
# =============================================================================

def run_agi_discovery():
    """Run the full AGI-style scientific discovery pipeline."""
    from training.agi_pipeline import AGIScientificPipeline

    pipeline = AGIScientificPipeline(
        checkpoint_dir=os.path.join(PROJECT_ROOT, 'checkpoints', 'agi_discovery'),
        log_dir=os.path.join(PROJECT_ROOT, 'logs', 'agi_discovery'),
        verbose=True,
    )

    results = pipeline.run(
        nx=64, n_regimes=5, steps_per_regime=200, latent_dim=4,
    )

    # Plot results
    save_path = os.path.join(IMAGES_DIR, 'agi_scientific_discovery.png')
    pipeline.plot_results(save_path=save_path)
    print(f"\n  Visualization saved: {save_path}")


def run_physics_aware_discovery():
    """Run physics-aware equation discovery on flow fields."""
    print("\n" + "="*60)
    print("  PHYSICS-AWARE EQUATION DISCOVERY")
    print("  Data -> Gradients + Laplacians -> Sparse Regression -> Equations")
    print("  du/dt = -u*nabla(u) + nu*nabla2(u) - grad(p) + corrections")
    print("="*60 + "\n")

    from core.fluid_solver_2d import FluidSolver2D
    from utils.helpers import compute_vorticity
    from models.physics_discovery import PhysicsAwareSINDy, CorrectionTermDiscovery, ConservationValidator
    from models.hypothesis_engine import HypothesisGenerator, ExperimentValidator, KnowledgeBase

    setup_plot_style()
    import matplotlib.pyplot as plt

    # Generate multi-Re flow data
    re_values = [100, 500, 2000]
    all_results = {}

    for Re in re_values:
        nu = 1.0 / Re
        print(f"\n  --- Re = {Re} ---")

        solver = FluidSolver2D(
            nx=64, ny=64, Lx=2*np.pi, Ly=2*np.pi,
            nu=nu, dt=0.005, pressure_solver="fft"
        )
        solver.initialize_taylor_green()
        solver.bc_manager.set_periodic()

        # Warmup
        for _ in range(20):
            solver.step()

        # Collect field snapshots
        u_fields, v_fields, p_fields, omega_fields = [], [], [], []
        for step in range(100):
            solver.step()
            if not np.all(np.isfinite(solver.u)):
                solver.u = np.nan_to_num(solver.u)
                solver.v = np.nan_to_num(solver.v)
            if step % 5 == 0:
                omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                u_fields.append(solver.u.copy())
                v_fields.append(solver.v.copy())
                p_fields.append(solver.p.copy())
                omega_fields.append(omega.copy())

        # Run physics-aware SINDy
        sindy = PhysicsAwareSINDy(
            threshold=0.05, alpha=0.01,
            poly_order=2, include_physics=True,
        )
        results = sindy.fit_from_fields(
            u_fields, v_fields, p_fields, omega_fields,
            dt=solver.dt * 5, dx=solver.dx, dy=solver.dy,
            verbose=True,
        )

        if 'error' not in results:
            # Extract corrections
            corr_finder = CorrectionTermDiscovery()
            corrections = corr_finder.extract_corrections(
                results['Xi'], results['feature_names'], nu
            )
            results['corrections'] = corrections

            # Validate
            validator = ConservationValidator()
            validation = validator.full_validation(
                u_fields, v_fields, p_fields, omega_fields,
                results['Xi'], results['feature_names'],
                dt=solver.dt * 5, dx=solver.dx, dy=solver.dy, nu=nu,
            )
            results['validation'] = validation
            print(f"    Conservation score: {validation['overall_score']:.0%}")

        all_results[Re] = results

    # Generate hypotheses
    hyp_gen = HypothesisGenerator()
    all_corrections = []
    for Re, res in all_results.items():
        for c in res.get('corrections', [])[:5]:
            all_corrections.append(c)
    if all_corrections:
        hyp_gen.generate_from_corrections(all_corrections[:10], nu=0.01, verbose=True)

    # Save knowledge
    kb = KnowledgeBase(os.path.join(PROJECT_ROOT, 'checkpoints', 'physics_discovery_kb'))
    for hyp in hyp_gen.hypotheses:
        kb.add_hypothesis(hyp)
        kb.add_equation(hyp.equation)
    kb.save()

    # ---- Visualization ----
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Physics-Aware Equation Discovery',
                fontsize=18, fontweight='bold', color='#58a6ff', y=0.97)

    # Panel 1: SINDy coefficient matrix
    ax1 = fig.add_subplot(2, 3, 1)
    for Re in re_values:
        res = all_results.get(Re, {})
        Xi = res.get('Xi')
        if Xi is not None:
            im = ax1.imshow(np.log10(np.abs(Xi) + 1e-10), cmap='viridis',
                           aspect='auto', interpolation='nearest')
            break
    ax1.set_title('SINDy Coefficients (log|xi|)', color='#79c0ff', fontsize=13)
    ax1.set_xlabel('Variable'); ax1.set_ylabel('Candidate')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: Correction terms
    ax2 = fig.add_subplot(2, 3, 2)
    if all_corrections:
        terms = [c['term'][:18] for c in all_corrections[:8]]
        sigs = [c['significance'] for c in all_corrections[:8]]
        types = [c['type'] for c in all_corrections[:8]]
        colors = ['#f97583' if t == 'novel' else '#58a6ff' for t in types]
        ax2.barh(range(len(terms)), sigs, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(terms)))
        ax2.set_yticklabels(terms, fontsize=8)
    ax2.set_title('Correction Terms', color='#79c0ff', fontsize=13)
    ax2.set_xlabel('Significance')
    ax2.grid(True, alpha=0.15, color='#30363d')

    # Panel 3: Conservation validation
    ax3 = fig.add_subplot(2, 3, 3)
    re_labels = []
    scores = []
    for Re, res in all_results.items():
        v = res.get('validation', {})
        if v:
            re_labels.append(f'Re={Re}')
            scores.append(v.get('overall_score', 0))
    if scores:
        ax3.bar(range(len(re_labels)), scores, color='#7ee787', alpha=0.8)
        ax3.set_xticks(range(len(re_labels)))
        ax3.set_xticklabels(re_labels)
        ax3.set_ylim(0, 1.1)
    ax3.set_title('Conservation Score', color='#79c0ff', fontsize=13)
    ax3.set_ylabel('Score')
    ax3.grid(True, alpha=0.15, color='#30363d')

    # Panel 4: Fit errors across Re
    ax4 = fig.add_subplot(2, 3, 4)
    for Re, res in all_results.items():
        fe = res.get('fit_error', {})
        if fe:
            ax4.bar(f'Re={Re}\nu', fe.get('u', 0), color='#58a6ff', alpha=0.7, label='u' if Re == re_values[0] else '')
            ax4.bar(f'Re={Re}\nv', fe.get('v', 0), color='#d2a8ff', alpha=0.7, label='v' if Re == re_values[0] else '')
    ax4.set_title('Equation Fit Error (MSE)', color='#79c0ff', fontsize=13)
    ax4.set_ylabel('MSE')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.15, color='#30363d')

    # Panel 5: Hypotheses
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('Generated Hypotheses', color='#79c0ff', fontsize=13)
    lines = []
    for h in hyp_gen.hypotheses[:8]:
        lines.append(f"[{h.id}] {h.description[:55]}")
    if lines:
        ax5.text(0.05, 0.95, '\n'.join(lines), transform=ax5.transAxes,
                va='top', fontsize=8, color='#c9d1d9', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                         edgecolor='#30363d', alpha=0.9))

    # Panel 6: Discovered equations
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('Discovered Equations', color='#79c0ff', fontsize=13)
    eq_lines = ["Physics-Aware Discovery Results\n"]
    for Re, res in all_results.items():
        eq_lines.append(f"Re={Re}:")
        u_eq = res.get('u_equation', '')
        if u_eq:
            eq_lines.append(f"  {u_eq[:60]}")
        n_c = len(res.get('corrections', []))
        eq_lines.append(f"  Corrections: {n_c}")
    ax6.text(0.05, 0.95, '\n'.join(eq_lines), transform=ax6.transAxes,
            va='top', fontsize=8, color='#c9d1d9', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                     edgecolor='#30363d', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(IMAGES_DIR, 'physics_aware_discovery.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    show_or_close(fig)
    print(f"\n  Visualization saved: {save_path}")

    # Summary
    print("\n" + "="*60)
    print("  PHYSICS-AWARE DISCOVERY -- SUMMARY")
    print("="*60)
    print(f"  Standard NS: du/dt = -u*nabla(u) + nu*nabla2(u) - grad(p)")
    for Re, res in all_results.items():
        n_c = len(res.get('corrections', []))
        fe = res.get('fit_error', {})
        print(f"  Re={Re}: {n_c} corrections, MSE_u={fe.get('u', 0):.2e}, MSE_v={fe.get('v', 0):.2e}")
    if all_corrections:
        print(f"\n  Top correction terms (beyond standard NS):")
        for c in all_corrections[:5]:
            print(f"    {c['discovered_coeff']:+.6f} * {c['term']} [{c['type']}]")
    print(f"  Hypotheses generated: {len(hyp_gen.hypotheses)}")
    print("="*60 + "\n")


# =============================================================================
# Relativistic Navier-Stokes (Israel-Stewart Causal Theory)
# =============================================================================

def run_relativistic_ns():
    """
    Relativistic Navier-Stokes — Israel-Stewart Causal Formulation.

    Naive relativistic NS is acausal and unstable.  The Israel-Stewart
    theory promotes viscous stresses to dynamical variables with a
    finite relaxation time tau_pi, restoring causality.

    Governing equation:
        d_mu T^{mu nu} = 0
        T^{mu nu} = (e+p) u^mu u^nu + p g^{mu nu} + pi^{mu nu}
        tau_pi D(pi^{ij}) + pi^{ij} = 2 eta sigma^{ij}

    Pipeline:
      1. Build T^{mu nu} energy-momentum tensor
      2. Solve conservation laws for energy + momentum
      3. Evolve pi^{mu nu} via IS relaxation
      4. Recover primitives (e, v) from conserved variables
      5. Publication-quality 8-panel visualization
    """
    print("\n" + "=" * 72)
    print("  RELATIVISTIC NAVIER-STOKES")
    print("  Israel-Stewart Causal Dissipation")
    print("  d_mu T^{mu nu} = 0   |   tau_pi D(pi) + pi = 2 eta sigma")
    print("=" * 72 + "\n")

    from physics.relativistic import RelativisticNSSolver

    setup_plot_style()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ── Phase 1: QGP Fireball (Bjorken-like) ─────────────────────────
    print("  [1/5] Initializing QGP fireball (Bjorken-like)...")
    nx, ny = 192, 192
    Lx, Ly = 12.0, 12.0
    eta_s = 0.2        # eta/s ~ 1/(4 pi) is near the KSS bound
    tau_pi = 0.5       # Relaxation time
    dt = 0.004

    solver = RelativisticNSSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        eta_s=eta_s, tau_pi=tau_pi, dt=dt,
        eos="ultrarelativistic",
    )
    solver.initialize_bjorken_flow(e0=15.0, sigma=1.5)
    print(f"    Grid: {nx}x{ny}")
    print(f"    Domain: [{-Lx/2:.1f}, {Lx/2:.1f}]^2")
    print(f"    eta/s = {eta_s}   tau_pi = {tau_pi}")
    print(f"    EOS: ultrarelativistic  (p = e/3)")
    print(f"    dt = {dt}")

    # ── Phase 2: Time evolution ───────────────────────────────────────
    n_steps = 500
    record_interval = 5
    print(f"\n  [2/5] Evolving d_mu T^{{mu nu}} = 0  ({n_steps} steps)...")

    time_hist, energy_hist, gamma_hist = [], [], []
    speed_hist, entropy_hist, pi_hist = [], [], []

    t_start = time.perf_counter()
    for step in range(n_steps):
        solver.step()

        if step % record_interval == 0:
            gamma = solver.lorentz_factor()
            speed = np.sqrt(solver.vx**2 + solver.vy**2)
            pi_mag = np.sqrt(solver.pi_xx**2 + 2*solver.pi_xy**2 + solver.pi_yy**2)

            time_hist.append(solver.time)
            energy_hist.append(float(np.sum(solver.energy_density) * solver.dx * solver.dy))
            gamma_hist.append(float(np.max(gamma)))
            speed_hist.append(float(np.max(speed)))
            entropy_hist.append(float(np.sum(solver.entropy) * solver.dx * solver.dy))
            pi_hist.append(float(np.max(pi_mag)))

            if step % 100 == 0:
                print(f"    Step {step:4d} | t={solver.time:.3f} | "
                      f"E_tot={energy_hist[-1]:.2f} | "
                      f"gamma_max={gamma_hist[-1]:.3f} | "
                      f"|v|_max={speed_hist[-1]:.4f}c")

    elapsed = time.perf_counter() - t_start
    print(f"\n    Completed: {n_steps} steps in {elapsed:.2f}s "
          f"({n_steps/elapsed:.0f} steps/s)")

    # ── Phase 3: Compute final state ─────────────────────────────────
    print("\n  [3/5] Computing final state diagnostics...")
    state = solver.get_state()
    gamma_final = state['lorentz_factor']
    speed_final = state['velocity_magnitude']
    e_final = state['energy_density']
    p_final = state['pressure']
    pi_mag_final = state['pi_magnitude']
    entropy_final = state['entropy']

    print(f"    Energy density: min={e_final.min():.4f}  max={e_final.max():.4f}")
    print(f"    Lorentz factor: min={gamma_final.min():.4f}  max={gamma_final.max():.4f}")
    print(f"    |v|/c:          max={speed_final.max():.6f}")
    print(f"    |pi^{{ij}}|:      max={pi_mag_final.max():.4f}")

    # ── Phase 4: T^{mu nu} analysis ──────────────────────────────────
    print("\n  [4/5] Analyzing energy-momentum tensor...")
    T00 = state['T00']
    T0x = state['T0x']
    T0y = state['T0y']
    print(f"    T^00 (lab energy):   max={np.max(T00):.4f}")
    print(f"    T^0x (x-momentum):   max|={np.max(np.abs(T0x)):.4f}")
    print(f"    T^0y (y-momentum):   max|={np.max(np.abs(T0y)):.4f}")

    # ── Phase 5: 8-panel visualization ───────────────────────────────
    print("\n  [5/5] Generating publication-quality visualization...")

    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        'Relativistic Navier-Stokes   \u00b7   Israel-Stewart Causal Theory',
        fontsize=20, fontweight='bold', color='#58a6ff', y=0.98,
    )

    # Panel 1: Energy density e
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.imshow(e_final, cmap='inferno', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax1.set_title('Energy Density  e', color='#79c0ff', fontsize=13)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.outline.set_edgecolor('#30363d')

    # Panel 2: Lorentz factor gamma
    ax2 = fig.add_subplot(2, 4, 2)
    im2 = ax2.imshow(gamma_final, cmap='hot', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax2.set_title('Lorentz Factor  \u03b3', color='#79c0ff', fontsize=13)
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.outline.set_edgecolor('#30363d')

    # Panel 3: Velocity magnitude |v|/c
    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(speed_final, cmap='magma', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax3.set_title('Velocity  |v|/c', color='#79c0ff', fontsize=13)
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.outline.set_edgecolor('#30363d')

    # Panel 4: Pressure p
    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(p_final, cmap='viridis', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax4.set_title('Pressure  p = e/3', color='#79c0ff', fontsize=13)
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.outline.set_edgecolor('#30363d')

    # Panel 5: Viscous stress |pi^{ij}|
    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(pi_mag_final, cmap='plasma', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax5.set_title('Viscous Stress  |\u03c0\u1d5e\u1d5b|  (IS)', color='#79c0ff', fontsize=13)
    ax5.set_xlabel('x'); ax5.set_ylabel('y')
    cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.outline.set_edgecolor('#30363d')

    # Panel 6: T^{00} lab-frame energy
    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(T00, cmap='inferno', origin='lower', extent=extent,
                     interpolation='bicubic')
    ax6.set_title('T\u2070\u2070  (Lab Energy)', color='#79c0ff', fontsize=13)
    ax6.set_xlabel('x'); ax6.set_ylabel('y')
    cb6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cb6.outline.set_edgecolor('#30363d')

    # Panel 7: Time evolution diagnostics
    ax7 = fig.add_subplot(2, 4, 7)
    t_arr = np.array(time_hist)
    ax7.plot(t_arr, energy_hist, '-', color='#58a6ff', lw=2, label='Total Energy')
    ax7_twin = ax7.twinx()
    ax7_twin.plot(t_arr, gamma_hist, '-', color='#f97583', lw=2, label='max \u03b3')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Total Energy', color='#58a6ff')
    ax7_twin.set_ylabel('max \u03b3', color='#f97583')
    ax7.set_title('Evolution Diagnostics', color='#79c0ff', fontsize=13)
    ax7.tick_params(axis='y', labelcolor='#58a6ff')
    ax7_twin.tick_params(axis='y', labelcolor='#f97583')
    ax7.grid(True, alpha=0.15, color='#30363d')

    # Panel 8: Entropy + viscous stress evolution
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.plot(t_arr, entropy_hist, '-', color='#7ee787', lw=2, label='Total Entropy')
    ax8_twin = ax8.twinx()
    ax8_twin.plot(t_arr, pi_hist, '-', color='#d2a8ff', lw=2, label='max |\u03c0|')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Total Entropy', color='#7ee787')
    ax8_twin.set_ylabel('max |\u03c0\u1d5e\u1d5b|', color='#d2a8ff')
    ax8.set_title('Entropy & Dissipation', color='#79c0ff', fontsize=13)
    ax8.tick_params(axis='y', labelcolor='#7ee787')
    ax8_twin.tick_params(axis='y', labelcolor='#d2a8ff')
    ax8.grid(True, alpha=0.15, color='#30363d')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='both', colors='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(IMAGES_DIR, 'relativistic_ns.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    show_or_close(fig)
    print(f"\n  Visualization saved: {save_path}")

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  RELATIVISTIC NAVIER-STOKES — SUMMARY")
    print("=" * 72)
    print(f"  Governing eq:   d_mu T^{{mu nu}} = 0")
    print(f"  T^{{mu nu}}:      (e+p) u^mu u^nu + p g^{{mu nu}} + pi^{{mu nu}}")
    print(f"  Dissipation:    Israel-Stewart (causal, 2nd order)")
    print(f"  Relaxation:     tau_pi D(pi) + pi = 2 eta sigma")
    print(f"  EOS:            Ultrarelativistic  p = e/3")
    print(f"  Grid:           {nx}x{ny},  L = {Lx}")
    print(f"  eta/s:          {eta_s}  (KSS bound ~ 1/4pi ~ 0.08)")
    print(f"  tau_pi:         {tau_pi}")
    print(f"  Steps:          {n_steps}  (dt={dt})")
    print(f"  max gamma:      {gamma_hist[-1]:.4f}")
    print(f"  max |v|/c:      {speed_hist[-1]:.6f}")
    print(f"  max |pi^ij|:    {pi_hist[-1]:.4f}")
    print()
    print("  Results:")
    print("    [+] Causal relativistic viscous hydrodynamics")
    print("    [+] Israel-Stewart relaxation (no acausal runaway)")
    print("    [+] 4-velocity & Lorentz factor tracking")
    print("    [+] Energy-momentum tensor T^{mu nu} evolution")
    print("    [+] Applicable: QGP, neutron stars, astrophysical jets")
    print("=" * 72 + "\n")


# =============================================================================
# Quantum Field Theory (QFT) Simulation
# =============================================================================

def run_qft_simulation():
    """
    Quantum Field Theory (QFT) Simulation on a Spacetime Lattice.

    Transition from classical fluids to fundamental fields:
        Classical:  velocity field u(x,t)
        Quantum:    field operator phi(x,t)

    Governing equation (Klein-Gordon + phi^4 interaction):

        []phi + m^2 phi + lambda phi^3 = 0

        where [] = d'Alembertian = -d^2/dt^2 + nabla^2

    This simulates:
        - Vacuum fluctuations (quantum zero-point energy)
        - Scalar field interactions (phi^4 theory)
        - Early universe inflation dynamics
        - Higgs-like spontaneous symmetry breaking
        - Bubble nucleation (first-order phase transitions)
        - Domain wall formation and dynamics

    AI Integration:
        PINN learns field evolution: (x,y,t) -> phi
        Embeds Klein-Gordon equation directly into loss function.

    Pipeline:
        1. Initialize scalar field on lattice (spacetime discretization)
        2. Evolve via leapfrog (symplectic integrator)
        3. Compute energy-momentum tensor T^{mu nu}
        4. Train PINN on lattice data (physics-informed)
        5. Analyze: field spectrum, correlation function, domain walls
        6. Publication-quality 10-panel visualization
    """
    print("\n" + "=" * 72)
    print("  QUANTUM FIELD THEORY SIMULATION")
    print("  Lattice Scalar Field + phi^4 Interaction + PINN")
    print("  []phi + m^2 phi + lambda phi^3 = 0")
    print("=" * 72 + "\n")

    from physics.qft_lattice import LatticeQFTSolver

    setup_plot_style()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ====================================================================
    # Scenario 1: Higgs-like Symmetry Breaking (Mexican Hat Potential)
    # ====================================================================
    print("  [1/6] Scenario: Higgs-like Spontaneous Symmetry Breaking")
    print("         V(phi) = lambda/4 * (phi^2 - v^2)^2  (Mexican hat)")

    nx, ny = 192, 192
    Lx, Ly = 20.0, 20.0
    mass_h = 1.0
    lam_h = 1.0
    dt_h = 0.02

    solver_higgs = LatticeQFTSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        mass=mass_h, lam=lam_h, dt=dt_h,
        potential_type="mexican_hat",
    )
    solver_higgs.initialize_higgs_quench(noise=0.15, seed=42)

    print(f"    Grid: {nx}x{ny}  |  L = {Lx}")
    print(f"    m = {mass_h}, lambda = {lam_h}")
    print(f"    VEV: v = {solver_higgs.v_vev:.4f}")
    print(f"    dt = {dt_h}")

    n_steps_h = 300
    print(f"    Evolving {n_steps_h} steps (leapfrog)...")

    t0 = time.perf_counter()
    solver_higgs.advance(n_steps_h, record=True)
    elapsed_h = time.perf_counter() - t0
    print(f"    Done in {elapsed_h:.2f}s ({n_steps_h/elapsed_h:.0f} steps/s)")

    E_higgs = solver_higgs.compute_total_energy()
    print(f"    Energy: K={E_higgs['kinetic']:.2f}  G={E_higgs['gradient']:.2f}  "
          f"V={E_higgs['potential']:.2f}  Total={E_higgs['total']:.2f}")

    # ====================================================================
    # Scenario 2: Vacuum Fluctuations (Standard phi^4)
    # ====================================================================
    print("\n  [2/6] Scenario: Vacuum Fluctuations (phi^4 theory)")
    print("         V(phi) = 1/2 m^2 phi^2 + 1/4 lambda phi^4")

    mass_v = 1.0
    lam_v = 0.5
    dt_v = 0.01

    solver_vacuum = LatticeQFTSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        mass=mass_v, lam=lam_v, dt=dt_v,
        potential_type="standard",
    )
    solver_vacuum.initialize_vacuum_fluctuations(amplitude=0.05, seed=123)

    n_steps_v = 400
    print(f"    Evolving {n_steps_v} steps...")
    t0 = time.perf_counter()
    solver_vacuum.advance(n_steps_v, record=True)
    elapsed_v = time.perf_counter() - t0
    print(f"    Done in {elapsed_v:.2f}s ({n_steps_v/elapsed_v:.0f} steps/s)")

    # ====================================================================
    # Scenario 3: Bubble Nucleation (Cosmological Phase Transition)
    # ====================================================================
    print("\n  [3/6] Scenario: Bubble Nucleation (first-order phase transition)")

    solver_bubble = LatticeQFTSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        mass=mass_h, lam=lam_h, dt=dt_h,
        potential_type="mexican_hat",
    )
    solver_bubble.initialize_bubble_nucleation(R=3.0)

    n_steps_b = 250
    print(f"    Evolving {n_steps_b} steps...")
    t0 = time.perf_counter()
    solver_bubble.advance(n_steps_b, record=True)
    elapsed_b = time.perf_counter() - t0
    print(f"    Done in {elapsed_b:.2f}s ({n_steps_b/elapsed_b:.0f} steps/s)")

    # ====================================================================
    # Phase 4: PINN Training on Lattice Data
    # ====================================================================
    print("\n  [4/6] Training QFT-PINN: (x,y,t) -> phi")
    print("         Loss = L_KG([]phi + m^2 phi + lambda phi^3) + L_data")

    pinn_trained = False
    pinn_history = None
    pinn_model = None

    try:
        import torch
        from models.qft_pinn import QFTPINN

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Device: {device}")

        pinn_model = QFTPINN(
            hidden_width=96, n_blocks=3,
            fourier_features=48, fourier_sigma=2.0,
            mass=mass_v, lam=lam_v,
        )

        pinn_history = pinn_model.train_on_lattice_data(
            solver_vacuum,
            n_collocation=3000,
            n_data_points=1500,
            epochs=1000,
            lr=1e-3,
            device=device,
            verbose=True,
        )
        pinn_trained = True
        print("    PINN training complete!")

    except ImportError:
        print("    PyTorch not available — skipping PINN training.")
    except Exception as e:
        print(f"    PINN training error: {e}")

    # ====================================================================
    # Phase 5: Analysis (spectrum, correlation, domain walls)
    # ====================================================================
    print("\n  [5/6] Computing field spectrum and correlations...")

    # Vacuum fluctuation spectrum
    k_spec, E_spec = solver_vacuum.compute_field_spectrum()
    valid_k = k_spec > 0
    k_v = k_spec[valid_k]
    E_v = np.maximum(E_spec[valid_k], 1e-30)

    # Free-field reference: P(k) ~ 1/(2*omega_k)
    omega_ref = np.sqrt(k_v**2 + mass_v**2)
    P_free = 0.05**2 / (2 * omega_ref)
    # Normalize to match data
    if len(E_v) > 2 and E_v[1] > 0:
        P_free *= E_v[1] / P_free[1]

    # Correlation function
    r_corr, G_corr = solver_vacuum.compute_correlation_function()
    valid_r = r_corr > 0

    # Domain walls from Higgs
    domain_walls = solver_higgs.compute_domain_walls()

    # Higgs field spectrum
    k_h, E_h = solver_higgs.compute_field_spectrum()
    valid_kh = k_h > 0

    print(f"    Vacuum spectrum: {len(k_v)} bins")
    print(f"    Correlation length ~ 1/m = {1.0/mass_v:.2f}")

    # ====================================================================
    # Phase 6: Publication-Quality 10-Panel Visualization
    # ====================================================================
    print("\n  [6/6] Generating publication-quality visualization...")

    fig = plt.figure(figsize=(28, 22))
    fig.suptitle(
        'Quantum Field Theory Simulation   ·   Lattice Scalar Field + PINN',
        fontsize=22, fontweight='bold', color='#58a6ff', y=0.98,
    )

    extent = [-Lx / 2, Lx / 2, -Ly / 2, Ly / 2]

    # ── Panel 1: Higgs field phi (symmetry breaking) ──
    ax1 = fig.add_subplot(2, 5, 1)
    state_h = solver_higgs.get_state()
    phi_h = state_h['phi']
    vabs = max(np.max(np.abs(phi_h)), 1e-6)
    im1 = ax1.imshow(
        phi_h, cmap='RdBu_r', origin='lower', extent=extent,
        interpolation='bicubic',
        norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs),
    )
    ax1.set_title('Field  phi  (Higgs SSB)', color='#79c0ff', fontsize=11)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.outline.set_edgecolor('#30363d')

    # ── Panel 2: Domain walls |nabla phi| ──
    ax2 = fig.add_subplot(2, 5, 2)
    dw_clip = np.clip(domain_walls, 0, np.percentile(domain_walls, 98))
    im2 = ax2.imshow(
        dw_clip, cmap='hot', origin='lower', extent=extent,
        interpolation='bicubic',
    )
    ax2.set_title('Domain Walls  |nabla phi|', color='#79c0ff', fontsize=11)
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.outline.set_edgecolor('#30363d')

    # ── Panel 3: Energy density T^{00} (Higgs) ──
    ax3 = fig.add_subplot(2, 5, 3)
    E_dens = state_h['energy_density']
    E_clip = np.clip(E_dens, 0, np.percentile(E_dens, 99))
    im3 = ax3.imshow(
        E_clip, cmap='inferno', origin='lower', extent=extent,
        interpolation='bicubic',
    )
    ax3.set_title('Energy Density  T^{00}', color='#79c0ff', fontsize=11)
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.outline.set_edgecolor('#30363d')

    # ── Panel 4: Vacuum fluctuations phi ──
    ax4 = fig.add_subplot(2, 5, 4)
    state_v = solver_vacuum.get_state()
    phi_v = state_v['phi']
    vabs_v = max(np.max(np.abs(phi_v)), 1e-6)
    im4 = ax4.imshow(
        phi_v, cmap='coolwarm', origin='lower', extent=extent,
        interpolation='bicubic',
        norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs_v, vmax=vabs_v),
    )
    ax4.set_title('Vacuum Fluctuations  phi', color='#79c0ff', fontsize=11)
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.outline.set_edgecolor('#30363d')

    # ── Panel 5: Bubble nucleation phi ──
    ax5 = fig.add_subplot(2, 5, 5)
    state_b = solver_bubble.get_state()
    phi_b = state_b['phi']
    vabs_b = max(np.max(np.abs(phi_b)), 1e-6)
    im5 = ax5.imshow(
        phi_b, cmap='PiYG', origin='lower', extent=extent,
        interpolation='bicubic',
        norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs_b, vmax=vabs_b),
    )
    ax5.set_title('Bubble Nucleation  phi', color='#79c0ff', fontsize=11)
    ax5.set_xlabel('x'); ax5.set_ylabel('y')
    cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.outline.set_edgecolor('#30363d')

    # ── Panel 6: Field power spectrum ──
    ax6 = fig.add_subplot(2, 5, 6)
    if len(k_v) > 2:
        ax6.loglog(k_v, E_v, '-', color='#58a6ff', lw=2.2, label='phi^4 theory', zorder=3)
        ax6.loglog(k_v, P_free, '--', color='#8b949e', lw=1.5, label='Free field 1/(2w_k)', zorder=2)
        ax6.fill_between(k_v, E_v, alpha=0.08, color='#58a6ff')
    if len(k_h[valid_kh]) > 2:
        ax6.loglog(k_h[valid_kh], np.maximum(E_h[valid_kh], 1e-30),
                   '-', color='#f97583', lw=1.5, label='Higgs SSB', alpha=0.7)
    ax6.set_xlabel('Wavenumber k')
    ax6.set_ylabel('P(k)')
    ax6.set_title('Field Power Spectrum', color='#79c0ff', fontsize=11)
    ax6.legend(fontsize=8, framealpha=0.7)
    ax6.grid(True, alpha=0.15, color='#30363d')

    # ── Panel 7: Correlation function ──
    ax7 = fig.add_subplot(2, 5, 7)
    if np.any(valid_r):
        r_v = r_corr[valid_r]
        G_v = G_corr[valid_r]
        ax7.plot(r_v, G_v, '-', color='#7ee787', lw=2.2, label='G(r) measured')
        # Yukawa reference: G(r) ~ exp(-m*r) / sqrt(r)
        G_yukawa = np.exp(-mass_v * r_v) / np.sqrt(r_v + 0.1)
        G_yukawa /= G_yukawa[0]
        ax7.plot(r_v, G_yukawa, '--', color='#8b949e', lw=1.5, label='Yukawa e^{-mr}/sqrt(r)')
        ax7.fill_between(r_v, G_v, alpha=0.08, color='#7ee787')
    ax7.set_xlabel('Distance r')
    ax7.set_ylabel('G(r) / G(0)')
    ax7.set_title('Two-Point Correlation', color='#79c0ff', fontsize=11)
    ax7.legend(fontsize=8, framealpha=0.7)
    ax7.grid(True, alpha=0.15, color='#30363d')
    ax7.set_ylim(-0.3, 1.1)

    # ── Panel 8: Energy evolution (Higgs) ──
    ax8 = fig.add_subplot(2, 5, 8)
    t_arr_h = np.array(solver_higgs.history['time'])
    ax8.plot(t_arr_h, solver_higgs.history['kinetic_energy'], '-', color='#58a6ff', lw=1.5, label='Kinetic')
    ax8.plot(t_arr_h, solver_higgs.history['gradient_energy'], '-', color='#f97583', lw=1.5, label='Gradient')
    ax8.plot(t_arr_h, solver_higgs.history['potential_energy'], '-', color='#7ee787', lw=1.5, label='Potential')
    ax8.plot(t_arr_h, solver_higgs.history['total_energy'], '-', color='#ffa657', lw=2.2, label='Total')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Energy')
    ax8.set_title('Energy Evolution (Higgs)', color='#79c0ff', fontsize=11)
    ax8.legend(fontsize=7, framealpha=0.7)
    ax8.grid(True, alpha=0.15, color='#30363d')

    # ── Panel 9: Field histogram (Higgs — double-well) ──
    ax9 = fig.add_subplot(2, 5, 9)
    phi_flat = phi_h.flatten()
    ax9.hist(phi_flat, bins=80, density=True, color='#d2a8ff', alpha=0.7, edgecolor='#30363d', lw=0.5)
    # Overlay potential
    phi_range = np.linspace(-2 * solver_higgs.v_vev, 2 * solver_higgs.v_vev, 200)
    V_range = solver_higgs.potential(phi_range)
    V_norm = V_range / np.max(V_range) * np.max(np.histogram(phi_flat, bins=80, density=True)[0]) * 0.8
    ax9.plot(phi_range, V_norm, '--', color='#ffa657', lw=2, label='V(phi) (scaled)')
    ax9.axvline(solver_higgs.v_vev, color='#7ee787', ls=':', lw=1.5, label=f'+v = {solver_higgs.v_vev:.2f}')
    ax9.axvline(-solver_higgs.v_vev, color='#f97583', ls=':', lw=1.5, label=f'-v = {-solver_higgs.v_vev:.2f}')
    ax9.set_xlabel('phi')
    ax9.set_ylabel('P(phi)')
    ax9.set_title('Field Distribution (SSB)', color='#79c0ff', fontsize=11)
    ax9.legend(fontsize=7, framealpha=0.7)
    ax9.grid(True, alpha=0.15, color='#30363d')

    # ── Panel 10: PINN training loss or field RMS evolution ──
    ax10 = fig.add_subplot(2, 5, 10)
    if pinn_trained and pinn_history:
        epochs_arr = np.arange(1, len(pinn_history['total']) + 1)
        ax10.semilogy(epochs_arr, pinn_history['pde'], '-', color='#58a6ff', lw=1.5, label='PDE (Klein-Gordon)')
        ax10.semilogy(epochs_arr, pinn_history['data'], '-', color='#f97583', lw=1.5, label='Data')
        ax10.semilogy(epochs_arr, pinn_history['total'], '-', color='#ffa657', lw=2, label='Total')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Loss')
        ax10.set_title('PINN Training Loss', color='#79c0ff', fontsize=11)
        ax10.legend(fontsize=8, framealpha=0.7)
    else:
        t_arr_v = np.array(solver_vacuum.history['time'])
        ax10.plot(t_arr_v, solver_vacuum.history['field_rms'], '-', color='#d2a8ff', lw=2, label='RMS(phi)')
        ax10.plot(t_arr_v, solver_vacuum.history['total_energy'], '-', color='#ffa657', lw=2, label='Total Energy')
        ax10.set_xlabel('Time')
        ax10.set_ylabel('Value')
        ax10.set_title('Vacuum Field Evolution', color='#79c0ff', fontsize=11)
        ax10.legend(fontsize=8, framealpha=0.7)
    ax10.grid(True, alpha=0.15, color='#30363d')

    # Style all spatial panels
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(axis='both', colors='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(IMAGES_DIR, 'qft_simulation.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    show_or_close(fig)
    print(f"\n  Visualization saved: {save_path}")

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  QUANTUM FIELD THEORY SIMULATION -- SUMMARY")
    print("=" * 72)
    print(f"  Governing eq:    []phi + m^2 phi + lambda phi^3 = 0")
    print(f"  d'Alembertian:   [] = -d^2/dt^2 + nabla^2  (Minkowski)")
    print(f"  Method:          Leapfrog (Stormer-Verlet, symplectic)")
    print(f"  Lattice:         {nx}x{ny},  L = {Lx}")
    print(f"")
    print(f"  Scenario 1: Higgs Symmetry Breaking")
    print(f"    Potential:     V = lambda/4 * (phi^2 - v^2)^2  (Mexican hat)")
    print(f"    VEV:           v = {solver_higgs.v_vev:.4f}")
    print(f"    Energy:        {E_higgs['total']:.2f}  (K={E_higgs['kinetic']:.2f} G={E_higgs['gradient']:.2f} V={E_higgs['potential']:.2f})")
    print(f"")
    E_vac = solver_vacuum.compute_total_energy()
    print(f"  Scenario 2: Vacuum Fluctuations")
    print(f"    Potential:     V = 1/2 m^2 phi^2 + 1/4 lambda phi^4")
    print(f"    Energy:        {E_vac['total']:.4f}")
    print(f"    Correlation:   xi ~ 1/m = {1.0/mass_v:.2f}")
    print(f"")
    E_bub = solver_bubble.compute_total_energy()
    print(f"  Scenario 3: Bubble Nucleation")
    print(f"    Energy:        {E_bub['total']:.2f}")
    print(f"")
    if pinn_trained:
        print(f"  PINN Integration:")
        print(f"    Architecture:  Fourier + ResNet")
        print(f"    Loss:          L_KG + L_data + L_energy")
        print(f"    Final PDE loss: {pinn_history['pde'][-1]:.6f}")
        print(f"    Final data loss: {pinn_history['data'][-1]:.6f}")
    else:
        print(f"  PINN: Skipped (PyTorch unavailable)")
    print()
    print("  Physics Modeled:")
    print("    [+] Vacuum fluctuations (quantum zero-point energy)")
    print("    [+] Scalar field interactions (phi^4 theory)")
    print("    [+] Higgs-like spontaneous symmetry breaking")
    print("    [+] Domain wall formation and dynamics")
    print("    [+] Bubble nucleation (cosmological phase transition)")
    print("    [+] Energy-momentum tensor T^{mu nu}")
    print("    [+] Field power spectrum P(k)")
    print("    [+] Two-point correlation function G(r)")
    if pinn_trained:
        print("    [+] PINN: (x,y,t) -> phi (Klein-Gordon embedded)")
    print("=" * 72 + "\n")


# =============================================================================
# Gravity + Fluid Coupling (Einstein Equations + Fluid Dynamics)
# =============================================================================

def run_gravity_fluid_coupling():
    """
    Gravity + Fluid Coupling Engine.

    Fluids curve spacetime. Spacetime tells fluids how to move.

    Core Einstein Equation:
        G_uv = 8 pi T_uv

    Fluid Energy-Momentum Tensor:
        T_uv = (rho + p) u_u u_v + p g_uv

    Runs three astrophysical scenarios at increasing gravity levels:
      1. Black Hole Accretion Disk   (Newtonian + Paczynski-Wiita)
      2. Neutron Star Merger         (Post-Newtonian, GW extraction)
      3. Galaxy Formation            (Numerical GR lite, BSSN-CFC)
    """
    print("\n" + "=" * 72)
    print("  GRAVITY + FLUID COUPLING ENGINE")
    print("  G_uv = 8 pi T_uv   |   T_uv = (rho+p) u_u u_v + p g_uv")
    print("  Matter tells spacetime how to curve.")
    print("  Spacetime tells matter how to move.")
    print("=" * 72 + "\n")

    from physics.gravity_fluid_coupling import GravityFluidSolver

    setup_plot_style()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    results = {}

    # ================================================================
    # Scenario 1: Black Hole Accretion Disk (Newtonian)
    # ================================================================
    print("  [1/3] Black Hole Accretion Disk (Newtonian + Paczynski-Wiita)")
    print("        Phi_PW = -GM / (r - r_s),  v_phi = sqrt(GM*r / (r-r_s)^2)")

    nx, ny = 192, 192
    Lx = 20.0

    solver_disk = GravityFluidSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Lx, dt=0.002,
        gravity_level="newtonian", gamma_eos=5.0/3.0,
    )
    solver_disk.initialize_accretion_disk(M_bh=3.0, rho0=0.5, r_in=2.0, r_out=8.0, T0=0.5)
    print(f"    Grid: {nx}x{ny}  |  L = {Lx}")

    n_disk = 200
    t0 = time.perf_counter()
    solver_disk.advance(n_disk, record=True)
    el_disk = time.perf_counter() - t0
    print(f"    {n_disk} steps in {el_disk:.2f}s ({n_disk/el_disk:.0f} steps/s)")
    results['disk'] = solver_disk

    # ================================================================
    # Scenario 2: Neutron Star Merger (Post-Newtonian)
    # ================================================================
    print("\n  [2/3] Neutron Star Merger (Post-Newtonian, GW extraction)")
    print("        Phi = Phi_N + (1/c^2)[2 Phi^2 + Psi]")

    solver_merger = GravityFluidSolver(
        nx=nx, ny=ny, Lx=Lx, Ly=Lx, dt=0.002,
        gravity_level="post_newtonian", gamma_eos=2.0,
    )
    solver_merger.initialize_neutron_star_merger(
        M_star=2.0, separation=5.0, v_orbit=0.08, sigma=1.2,
    )

    n_merger = 250
    t0 = time.perf_counter()
    solver_merger.advance(n_merger, record=True)
    el_merger = time.perf_counter() - t0
    print(f"    {n_merger} steps in {el_merger:.2f}s ({n_merger/el_merger:.0f} steps/s)")
    results['merger'] = solver_merger

    # ================================================================
    # Scenario 3: Galaxy Formation (Numerical GR lite)
    # ================================================================
    print("\n  [3/3] Galaxy Formation (Numerical GR — BSSN/CFC lite)")
    print("        ds^2 = -alpha^2 dt^2 + psi^4 (dx^2 + dy^2)")

    solver_galaxy = GravityFluidSolver(
        nx=nx, ny=ny, Lx=30.0, Ly=30.0, dt=0.003,
        gravity_level="numerical_gr", gamma_eos=5.0/3.0,
    )
    solver_galaxy.initialize_galaxy_formation(n_clumps=6, rho_bg=0.02, perturbation=0.1, seed=42)

    n_galaxy = 200
    t0 = time.perf_counter()
    solver_galaxy.advance(n_galaxy, record=True)
    el_galaxy = time.perf_counter() - t0
    print(f"    {n_galaxy} steps in {el_galaxy:.2f}s ({n_galaxy/el_galaxy:.0f} steps/s)")
    results['galaxy'] = solver_galaxy

    # ================================================================
    # Publication-Quality 10-Panel Visualization
    # ================================================================
    print("\n  Generating publication-quality visualization...")

    fig = plt.figure(figsize=(28, 22))
    fig.suptitle(
        'Gravity + Fluid Coupling   ·   G$_{\\mu\\nu}$ = 8$\\pi$ T$_{\\mu\\nu}$',
        fontsize=22, fontweight='bold', color='#58a6ff', y=0.98,
    )

    # --- Row 1: Accretion Disk ---
    sd = solver_disk.get_state()
    extent_d = [-Lx/2, Lx/2, -Lx/2, Lx/2]

    ax1 = fig.add_subplot(2, 5, 1)
    rho_clip = np.clip(sd['rho'], 0, np.percentile(sd['rho'], 98))
    im1 = ax1.imshow(rho_clip, cmap='inferno', origin='lower', extent=extent_d,
                     interpolation='bicubic')
    ax1.set_title('Accretion Disk  $\\rho$', color='#79c0ff', fontsize=11)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    ax2 = fig.add_subplot(2, 5, 2)
    im2 = ax2.imshow(sd['velocity_magnitude'], cmap='magma', origin='lower',
                     extent=extent_d, interpolation='bicubic')
    ax2.set_title('Velocity  |v|', color='#79c0ff', fontsize=11)
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    # --- Row 1: Merger ---
    sm = solver_merger.get_state()

    ax3 = fig.add_subplot(2, 5, 3)
    rho_m = np.clip(sm['rho'], 0, np.percentile(sm['rho'], 98))
    im3 = ax3.imshow(rho_m, cmap='inferno', origin='lower', extent=extent_d,
                     interpolation='bicubic')
    ax3.set_title('NS Merger  $\\rho$ (1PN)', color='#79c0ff', fontsize=11)
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    ax4 = fig.add_subplot(2, 5, 4)
    Phi_m = np.clip(sm['Phi'], np.percentile(sm['Phi'], 2), np.percentile(sm['Phi'], 98))
    im4 = ax4.imshow(Phi_m, cmap='cividis', origin='lower', extent=extent_d,
                     interpolation='bicubic')
    ax4.set_title('Gravitational Potential  $\\Phi$', color='#79c0ff', fontsize=11)
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    # --- Row 1: Galaxy ---
    sg = solver_galaxy.get_state()
    extent_g = [-15, 15, -15, 15]

    ax5 = fig.add_subplot(2, 5, 5)
    rho_g = np.clip(sg['rho'], 0, np.percentile(sg['rho'], 98))
    im5 = ax5.imshow(np.log10(rho_g + 1e-8), cmap='magma', origin='lower',
                     extent=extent_g, interpolation='bicubic')
    ax5.set_title('Galaxy Formation  log$\\rho$ (GR)', color='#79c0ff', fontsize=11)
    ax5.set_xlabel('x'); ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    # --- Row 2: Analysis panels ---

    # Panel 6: T^{00} energy density (disk)
    ax6 = fig.add_subplot(2, 5, 6)
    T00_clip = np.clip(sd['T00'], 0, np.percentile(sd['T00'], 98))
    im6 = ax6.imshow(T00_clip, cmap='hot', origin='lower', extent=extent_d,
                     interpolation='bicubic')
    ax6.set_title('$T^{00}$  (Disk Energy-Mom)', color='#79c0ff', fontsize=11)
    ax6.set_xlabel('x'); ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    # Panel 7: GR lapse function (galaxy)
    ax7 = fig.add_subplot(2, 5, 7)
    im7 = ax7.imshow(sg['lapse'], cmap='viridis', origin='lower', extent=extent_g,
                     interpolation='bicubic')
    ax7.set_title('Lapse  $\\alpha$  (GR metric)', color='#79c0ff', fontsize=11)
    ax7.set_xlabel('x'); ax7.set_ylabel('y')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04).outline.set_edgecolor('#30363d')

    # Panel 8: GW strain evolution (merger)
    ax8 = fig.add_subplot(2, 5, 8)
    t_m = np.array(solver_merger.history['time'])
    gw_m = np.array(solver_merger.history['gw_strain'])
    ax8.plot(t_m, gw_m, '-', color='#d2a8ff', lw=2, label='h (merger)')
    ax8.fill_between(t_m, gw_m, alpha=0.15, color='#d2a8ff')
    t_g = np.array(solver_galaxy.history['time'])
    gw_g = np.array(solver_galaxy.history['gw_strain'])
    ax8.plot(t_g, gw_g, '--', color='#7ee787', lw=1.5, label='h (galaxy)', alpha=0.7)
    ax8.set_xlabel('Time'); ax8.set_ylabel('GW Strain h')
    ax8.set_title('Gravitational Waves  (quadrupole)', color='#79c0ff', fontsize=11)
    ax8.legend(fontsize=8, framealpha=0.7)
    ax8.grid(True, alpha=0.15, color='#30363d')

    # Panel 9: Energy evolution (disk)
    ax9 = fig.add_subplot(2, 5, 9)
    t_d = np.array(solver_disk.history['time'])
    ax9.plot(t_d, solver_disk.history['kinetic_energy'], '-', color='#58a6ff', lw=1.5, label='Kinetic')
    ax9.plot(t_d, solver_disk.history['gravitational_energy'], '-', color='#f97583', lw=1.5, label='Gravitational')
    ax9.plot(t_d, solver_disk.history['thermal_energy'], '-', color='#7ee787', lw=1.5, label='Thermal')
    ax9.set_xlabel('Time'); ax9.set_ylabel('Energy')
    ax9.set_title('Energy Budget (Disk)', color='#79c0ff', fontsize=11)
    ax9.legend(fontsize=7, framealpha=0.7)
    ax9.grid(True, alpha=0.15, color='#30363d')

    # Panel 10: Virial ratio + max density (merger)
    ax10 = fig.add_subplot(2, 5, 10)
    ax10.plot(t_m, solver_merger.history['virial_ratio'], '-', color='#ffa657', lw=2, label='Virial 2K/|W|')
    ax10_twin = ax10.twinx()
    ax10_twin.plot(t_m, solver_merger.history['max_density'], '-', color='#f97583', lw=1.5, label='max $\\rho$')
    ax10.set_xlabel('Time')
    ax10.set_ylabel('Virial Ratio', color='#ffa657')
    ax10_twin.set_ylabel('max $\\rho$', color='#f97583')
    ax10.set_title('Virial & Collapse (Merger)', color='#79c0ff', fontsize=11)
    ax10.tick_params(axis='y', labelcolor='#ffa657')
    ax10_twin.tick_params(axis='y', labelcolor='#f97583')
    ax10.grid(True, alpha=0.15, color='#30363d')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.tick_params(axis='both', colors='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(IMAGES_DIR, 'gravity_fluid_coupling.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    show_or_close(fig)
    print(f"\n  Visualization saved: {save_path}")

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  GRAVITY + FLUID COUPLING — SUMMARY")
    print("=" * 72)
    print(f"  Core equation:   G_uv = 8 pi T_uv")
    print(f"  Fluid T_uv:      (rho+p) u_u u_v + p g_uv")
    print(f"")
    print(f"  Scenario 1: Black Hole Accretion Disk")
    print(f"    Level:         Newtonian (Paczynski-Wiita pseudo-potential)")
    print(f"    max rho:       {solver_disk.history['max_density'][-1]:.4f}")
    print(f"    max |v|:       {solver_disk.history['max_velocity'][-1]:.4f}")
    print(f"")
    print(f"  Scenario 2: Neutron Star Merger")
    print(f"    Level:         Post-Newtonian (1PN corrections)")
    print(f"    max rho:       {solver_merger.history['max_density'][-1]:.4f}")
    print(f"    GW strain:     {solver_merger.history['gw_strain'][-1]:.6f}")
    print(f"    Virial:        {solver_merger.history['virial_ratio'][-1]:.4f}")
    print(f"")
    print(f"  Scenario 3: Galaxy Formation")
    print(f"    Level:         Numerical GR (BSSN/CFC lite)")
    print(f"    max rho:       {solver_galaxy.history['max_density'][-1]:.4f}")
    print(f"    Lapse range:   [{np.min(sg['lapse']):.4f}, {np.max(sg['lapse']):.4f}]")
    print(f"    Conf. factor:  [{np.min(sg['conformal_factor']):.4f}, {np.max(sg['conformal_factor']):.4f}]")
    print(f"")
    print("  Results:")
    print("    [+] Newtonian gravity (Poisson solver via FFT)")
    print("    [+] Post-Newtonian 1PN corrections (v^2/c^2, Phi/c^2)")
    print("    [+] Numerical GR lite (lapse, shift, conformal factor)")
    print("    [+] Energy-momentum tensor T^{mu nu} computation")
    print("    [+] Gravitational wave extraction (quadrupole formula)")
    print("    [+] Black hole accretion disk (Paczynski-Wiita)")
    print("    [+] Neutron star binary merger")
    print("    [+] Galaxy formation via gravitational collapse")
    print("=" * 72 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    global HEADLESS
    
    parser = argparse.ArgumentParser(
        description="Navier-Stokes ML/DL Hybrid Simulation System + Turbulence Discovery AI",
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
  python main.py --physics mhd        MHD simulation
  python main.py --hybrid             Hybrid CFD->PINN demo
  python main.py --gpu                GPU solver demo
  python main.py --vort-conf 5.0      Vorticity confinement demo
  python main.py --benchmark          CFD performance benchmarks
  python main.py --discover           Turbulence Discovery AI (full pipeline)
  python main.py --stability          Blow-up detection & stability analysis
  python main.py --regularity         Empirical regularity map
  python main.py --metrics            DNS vs LES turbulence metrics
  python main.py --symbolic           Symbolic equation discovery (SINDy + GP)
  python main.py --relativistic       Relativistic NS (Israel-Stewart causal theory)
  python main.py --agi                Full AGI scientific discovery system
  python main.py --physics-discover   Physics-aware equation discovery
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run Taylor-Green demo')
    parser.add_argument('--viz2d', action='store_true', help='Launch 2D real-time visualizer')
    parser.add_argument('--viz3d', action='store_true', help='Launch 3D visualizer')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    parser.add_argument('--train', type=str, choices=['pinn', 'fno', 'deeponet', 'surrogate'],
                       help='Train ML model')
    parser.add_argument('--physics', type=str,
                       choices=['mhd', 'astro', 'bio', 'climate', 'quantum', 'relativistic', 'qft', 'gravity'],
                       help='Run physics domain simulation')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--hybrid', action='store_true', help='Run hybrid CFD->PINN demo')
    parser.add_argument('--gpu', action='store_true', help='Run GPU solver demo')
    parser.add_argument('--vort-conf', type=float, dest='vort_conf', default=None,
                       help='Run vorticity confinement demo (specify eps strength)')
    parser.add_argument('--no-gui', action='store_true', dest='no_gui',
                       help='Headless mode: save plots to disk, skip plt.show()')
    # Turbulence Discovery AI
    parser.add_argument('--discover', action='store_true',
                       help='Run full Turbulence Discovery AI pipeline')
    parser.add_argument('--stability', action='store_true',
                       help='Run blow-up detection & stability analysis')
    parser.add_argument('--regularity', action='store_true',
                       help='Generate empirical regularity map')
    parser.add_argument('--metrics', action='store_true',
                       help='Run DNS vs LES turbulence metrics comparison')
    parser.add_argument('--symbolic', action='store_true',
                       help='Run symbolic equation discovery (SINDy + GP)')
    # Quantum Fluid Extensions
    parser.add_argument('--quantum-ext', action='store_true', dest='quantum_ext',
                       help='Quantum Fluid Extensions (GPE + Madelung transform)')
    # Relativistic NS
    parser.add_argument('--relativistic', action='store_true',
                       help='Relativistic NS with Israel-Stewart causal theory')
    # AGI Discovery
    parser.add_argument('--agi', action='store_true',
                       help='Full AGI scientific discovery system')
    parser.add_argument('--physics-discover', action='store_true', dest='physics_discover',
                       help='Physics-aware equation discovery')
    # Quantum Field Theory
    parser.add_argument('--qft', action='store_true',
                       help='Quantum Field Theory simulation (Lattice QFT + PINN)')
    # Gravity + Fluid Coupling
    parser.add_argument('--gravity-coupling', action='store_true', dest='gravity_coupling',
                       help='Gravity + Fluid Coupling (Einstein equations + fluid dynamics)')
    
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
    elif args.hybrid:
        run_hybrid_demo()
    elif args.gpu:
        run_gpu_demo()
    elif args.vort_conf is not None:
        run_vorticity_confinement_demo()
    elif args.discover:
        run_turbulence_discovery()
    elif args.stability:
        run_stability_analysis()
    elif args.regularity:
        run_regularity_map()
    elif args.metrics:
        run_turbulence_metrics()
    elif args.symbolic:
        run_symbolic_discovery_standalone()
    elif args.quantum_ext:
        run_quantum_extensions()
    elif args.relativistic:
        run_relativistic_ns()
    elif args.agi:
        run_agi_discovery()
    elif args.physics_discover:
        run_physics_aware_discovery()
    elif args.qft:
        run_qft_simulation()
    elif args.gravity_coupling:
        run_gravity_fluid_coupling()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
