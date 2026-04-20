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
      python main.py --benchmark         → Run CFD benchmarks
      python main.py --physics mhd       → Run MHD simulation
      python main.py --physics astro     → Run astrophysics simulation
      python main.py --physics bio       → Run biophysics simulation
      python main.py --physics climate   → Run climate simulation
      python main.py --physics quantum   → Run quantum fluid simulation
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


def run_demo():
    """Run the Taylor-Green vortex decay demo with live plotting."""
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
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f'Taylor-Green Vortex Decay (Re={Re}, {nx}×{ny})',
                    fontsize=16, fontweight='bold')
        
        # Vorticity
        omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
        im1 = axes[0, 0].imshow(omega, cmap='RdBu_r', origin='lower',
                                extent=[0, 2*np.pi, 0, 2*np.pi])
        axes[0, 0].set_title('Vorticity ω')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Velocity magnitude
        speed = solver.get_velocity_magnitude()
        im2 = axes[0, 1].imshow(speed, cmap='inferno', origin='lower',
                                extent=[0, 2*np.pi, 0, 2*np.pi])
        axes[0, 1].set_title('Velocity Magnitude |u|')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Pressure
        im3 = axes[0, 2].imshow(solver.p, cmap='viridis', origin='lower',
                                extent=[0, 2*np.pi, 0, 2*np.pi])
        axes[0, 2].set_title('Pressure p')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Velocity vectors
        skip = 8
        X, Y = np.meshgrid(
            np.linspace(0, 2*np.pi, nx),
            np.linspace(0, 2*np.pi, ny)
        )
        axes[1, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                         solver.u[::skip, ::skip], solver.v[::skip, ::skip],
                         speed[::skip, ::skip], cmap='coolwarm', scale=15)
        axes[1, 0].set_title('Velocity Vectors')
        axes[1, 0].set_aspect('equal')
        
        # KE decay comparison
        axes[1, 1].plot(times, ke_numerical, 'b-', lw=2, label='Numerical')
        axes[1, 1].plot(times, ke_analytical, 'r--', lw=2, label='Analytical')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Kinetic Energy')
        axes[1, 1].set_title('KE Decay (Validation)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Error
        errors = [abs(n - a) / max(a, 1e-10) for n, a in zip(ke_numerical, ke_analytical)]
        axes[1, 2].semilogy(times, errors, 'g-', lw=2)
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Relative Error')
        axes[1, 2].set_title('KE Relative Error')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, 'demo_taylor_green.png'), dpi=150)
        plt.show()
        print(f"\n  Plot saved to demo_taylor_green.png")
        
    except ImportError:
        print("  (matplotlib not available, skipping plots)")


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


def run_physics_demo(domain: str):
    """Run a physics domain-specific demo."""
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
    
    # Visualize final state
    try:
        import matplotlib.pyplot as plt
        
        state = solver.get_state()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{title} (t = {state["time"]:.3f})', fontsize=14, fontweight='bold')
        
        # Velocity magnitude
        vel = state.get('velocity_magnitude', np.sqrt(state['u']**2 + state['v']**2))
        im1 = axes[0].imshow(vel, cmap='inferno', origin='lower')
        axes[0].set_title('Velocity Magnitude')
        plt.colorbar(im1, ax=axes[0])
        
        # Vorticity or domain-specific field
        if 'Jz' in state:
            field2 = state['Jz']
            t2 = 'Current Density Jz'
        elif 'rho' in state:
            field2 = state['rho']
            t2 = 'Density ρ'
        elif 'density' in state:
            field2 = state['density']
            t2 = 'Superfluid Density |ψ|²'
        elif 'wss' in state:
            field2 = state['viscosity']
            t2 = 'Viscosity Field'
        elif 'omega' in state:
            field2 = state['omega']
            t2 = 'Vorticity ω'
        else:
            vort = np.gradient(state['v'], axis=1) - np.gradient(state['u'], axis=0)
            field2 = vort
            t2 = 'Vorticity ω'
        
        im2 = axes[1].imshow(field2, cmap='RdBu_r', origin='lower')
        axes[1].set_title(t2)
        plt.colorbar(im2, ax=axes[1])
        
        # Pressure or extra field
        if 'p' in state:
            im3 = axes[2].imshow(state['p'], cmap='viridis', origin='lower')
            axes[2].set_title('Pressure p')
        elif 'phase' in state:
            im3 = axes[2].imshow(state['phase'], cmap='hsv', origin='lower')
            axes[2].set_title('Phase arg(ψ)')
        elif 'T' in state:
            im3 = axes[2].imshow(state['T'], cmap='hot', origin='lower')
            axes[2].set_title('Temperature T')
        else:
            im3 = axes[2].imshow(state['p'], cmap='viridis', origin='lower')
            axes[2].set_title('Pressure')
        plt.colorbar(im3, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        filename = f'demo_{domain}.png'
        plt.savefig(os.path.join(PROJECT_ROOT, filename), dpi=150)
        plt.show()
        print(f"  Saved: {filename}")
        
    except ImportError:
        print("  (matplotlib not available)")


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
    
    else:
        print(f"  Training for {model_type} not yet configured.")
        print(f"  Available: pinn, fno")


def interactive_menu():
    """Interactive mode selector."""
    print_banner()
    deps = check_dependencies()
    
    print("  Select Mode:")
    print("  ─────────────────────────────────────────")
    print("  [1] 🎬 Demo: Taylor-Green Vortex Decay")
    print("  [2] 🎮 Real-Time 2D Visualizer (Pygame)")
    print("  [3] 🌐 3D Visualizer (Matplotlib/PyVista)")
    print("  [4] 📊 Streamlit Dashboard")
    print("  [5] 🏋 Train PINN Model")
    print("  [6] 🏋 Train FNO Model")
    print("  [7] ⚡ MHD Simulation")
    print("  [8] 🌟 Astrophysics Simulation")
    print("  [9] ❤ Biophysics (Blood Flow)")
    print("  [10] 🌍 Climate Simulation")
    print("  [11] ⚛ Quantum Fluid Simulation")
    print("  [12] 📏 CFD Benchmarks")
    print("  [0] ❌ Exit")
    print()
    
    try:
        choice = input("  Enter choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Goodbye!")
        return
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        from visualization.realtime_2d import RealtimeVisualizer2D
        viz = RealtimeVisualizer2D()
        viz.run()
    elif choice == "3":
        from visualization.realtime_3d import RealtimeVisualizer3D
        viz = RealtimeVisualizer3D()
        viz.run()
    elif choice == "4":
        print("\n  Launching Streamlit dashboard...")
        os.system(f'streamlit run "{os.path.join(PROJECT_ROOT, "dashboard", "app.py")}"')
    elif choice == "5":
        run_training("pinn")
    elif choice == "6":
        run_training("fno")
    elif choice == "7":
        run_physics_demo("mhd")
    elif choice == "8":
        run_physics_demo("astro")
    elif choice == "9":
        run_physics_demo("bio")
    elif choice == "10":
        run_physics_demo("climate")
    elif choice == "11":
        run_physics_demo("quantum")
    elif choice == "12":
        run_benchmark()
    elif choice == "0":
        print("  Goodbye!")
    else:
        print(f"  Unknown choice: {choice}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Navier-Stokes ML/DL Hybrid Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      Interactive menu
  python main.py --demo               Taylor-Green vortex demo
  python main.py --viz2d              Real-time 2D pygame visualizer
  python main.py --viz3d              3D matplotlib/PyVista visualizer
  python main.py --dashboard          Streamlit web dashboard
  python main.py --train pinn         Train PINN model
  python main.py --train fno          Train FNO model
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
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.demo:
        run_demo()
    elif args.viz2d:
        from visualization.realtime_2d import RealtimeVisualizer2D
        viz = RealtimeVisualizer2D()
        viz.run()
    elif args.viz3d:
        from visualization.realtime_3d import RealtimeVisualizer3D
        viz = RealtimeVisualizer3D()
        viz.run()
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
