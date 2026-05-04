<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/CFD-Research_Grade-blueviolet?style=for-the-badge" alt="CFD">
</p>

<h1 align="center">
  🌊 Navier-Stokes ML/DL Hybrid Simulation System<br>
  + 🧠 Turbulence Discovery AI
</h1>

<p align="center">
  <b>Research-Grade Computational Fluid Dynamics + Deep Learning Platform</b><br>
  <code>∂u/∂t + (u·∇)u = −∇p + ν∇²u + f</code>
</p>

---

A production-ready, hybrid Navier-Stokes simulation system that integrates **classical CFD solvers** with **deep learning surrogates** (PINN, FNO, DeepONet, U-Net) and a **Turbulence Discovery AI** pipeline (Autoencoder + Neural ODE + SINDy + Genetic Programming). Supports 6 physics domains, blow-up detection & regularity analysis, real-time 2D/3D visualization, a Streamlit web dashboard, and a unified CLI for simulation, training, discovery, and benchmarking.

## ✨ Features

| Category | Details |
|----------|---------|
| **CFD Solvers** | 2D/3D incompressible NS with projection method, FFT/Jacobi/SOR/CG pressure solvers, central/upwind/WENO-5 advection |
| **ML Models** | PINN (physics-informed), FNO (Fourier operator), DeepONet (operator network), U-Net Surrogate (instant prediction) |
| **Discovery AI** | Convolutional Autoencoder (flow compression), Neural ODE (latent dynamics), SINDy (sparse regression), Genetic Programming (symbolic equations) |
| **Regularity** | Blow-up detection CNN, stability analysis (Re sweeps), BKM criterion monitoring, empirical regularity maps |
| **Physics** | Classical Fluid · MHD · Astrophysics · Biophysics · Climate · Quantum Fluids |
| **Turbulence** | DNS, Smagorinsky LES, Dynamic Smagorinsky, k-ε RANS, vorticity confinement |
| **Metrics** | L2 error, energy spectrum error, vorticity accuracy, structure functions, DNS vs LES comparison |
| **Visualization** | Real-time 2D (Pygame), 3D (PyVista/Matplotlib), Streamlit dashboard (Plotly) |
| **Training** | AMP, cosine annealing + warmup LR, gradient clipping, checkpointing, curriculum learning |

---

## 📂 Project Structure

```
Navier-Stokes/
├── main.py                 # Master entry point — CLI + interactive menu
├── config.py               # Global configuration & hyperparameters
├── requirements.txt        # Python dependencies
│
├── core/                   # Classical CFD engine
│   ├── fluid_solver_2d.py  # 2D incompressible NS (projection method)
│   ├── fluid_solver_3d.py  # 3D incompressible NS
│   ├── pressure_solver.py  # FFT, Jacobi, SOR, CG, Multigrid
│   ├── discretization.py   # Finite difference operators (central, upwind, WENO)
│   ├── boundary_conditions.py  # Dirichlet, Neumann, periodic, no-slip, inflow/outflow
│   └── turbulence_models.py    # Smagorinsky, Dynamic Smag, k-ε, k-ω
│
├── models/                 # Deep learning architectures
│   ├── pinn.py             # Physics-Informed Neural Network
│   ├── fno.py              # Fourier Neural Operator (2D)
│   ├── deeponet.py         # Deep Operator Network
│   ├── surrogate.py        # U-Net Surrogate with attention gates
│   ├── turbulence_nn.py    # Neural turbulence closure model
│   ├── autoencoder.py      # Flow Autoencoder + Latent ODE (discovery)
│   ├── symbolic_discovery.py  # SINDy + Genetic Programming
│   └── regularity_analysis.py # Blow-up detection + stability analysis
│
├── physics/                # Cross-physics domain solvers
│   ├── mhd.py              # Magnetohydrodynamics (Orszag-Tang vortex)
│   ├── astrophysics.py     # Stellar/galactic flows (Rayleigh-Taylor)
│   ├── biophysics.py       # Blood flow (pulsatile, non-Newtonian)
│   ├── climate.py          # Geophysical flows (Kelvin-Helmholtz)
│   └── quantum_fluids.py   # BEC / superfluids (Gross-Pitaevskii)
│
├── training/               # ML training infrastructure
│   ├── trainer.py          # Unified training loop (PINN, FNO, DeepONet, Surrogate)
│   ├── discovery_trainer.py # 4-phase Turbulence Discovery pipeline
│   ├── turbulence_data.py  # Multi-regime flow data generator
│   ├── data_generator.py   # CFD-to-ML dataset generation
│   └── losses.py           # Physics-informed loss functions
│
├── visualization/          # Rendering & real-time viz
│   ├── renderer.py         # Flow field → RGB conversion, streamlines
│   ├── realtime_2d.py      # Pygame-based interactive 2D visualizer
│   └── realtime_3d.py      # PyVista/Matplotlib 3D visualizer
│
├── dashboard/              # Web interface
│   └── app.py              # Streamlit dashboard (simulation + ML)
│
├── utils/                  # Shared utilities
│   └── helpers.py          # Vorticity, KE, enstrophy, CFL, drag/lift
│
├── tests/                  # Test suite
│   └── test_smoke.py       # Comprehensive smoke tests
│
├── checkpoints/            # Saved model weights
├── logs/                   # Training logs
└── data/                   # Generated datasets
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Minimum:** `numpy`, `scipy`, `matplotlib`. **Full:** add `torch`, `pygame`, `pyvista`, `streamlit`, `plotly`.

### 2. Run the Interactive Menu

```bash
python main.py
```

This launches a menu where you can select from 22 modes: demos, training, visualization, physics simulations, discovery AI, regularity analysis, and benchmarks.

### 3. CLI Usage

```bash
# Taylor-Green vortex demo (saves publication-quality plot)
python main.py --demo

# Headless mode (no GUI, save plots only)
python main.py --demo --no-gui

# Real-time 2D visualizer (Pygame)
python main.py --viz2d

# 3D visualizer
python main.py --viz3d

# Streamlit web dashboard
python main.py --dashboard

# Train ML models
python main.py --train pinn
python main.py --train fno
python main.py --train deeponet
python main.py --train surrogate

# Physics domain simulations
python main.py --physics mhd
python main.py --physics astro
python main.py --physics bio
python main.py --physics climate
python main.py --physics quantum

# Turbulence Discovery AI
python main.py --discover           # Full pipeline (AE -> ODE -> Blow-up -> Symbolic)
python main.py --stability           # Blow-up detection & stability analysis
python main.py --regularity          # Empirical regularity map (Re sweep)
python main.py --metrics             # DNS vs LES turbulence metrics comparison
python main.py --symbolic            # SINDy + GP equation discovery

# CFD performance benchmarks
python main.py --benchmark
```

---

## 🧪 Physics Domains

### Classical Fluid Dynamics
Standard incompressible Navier-Stokes with the projection method. Supports Taylor-Green vortex, double shear layer, vortex pair, lid-driven cavity, and channel flow initial conditions.

### Magnetohydrodynamics (MHD)
Coupled NS + Maxwell equations with Lorentz force and magnetic induction. Benchmark: Orszag-Tang vortex system.

### Astrophysics
Self-gravitating flows with radiative cooling. Benchmark: Rayleigh-Taylor instability.

### Biophysics (Blood Flow)
Pulsatile hemodynamics with non-Newtonian Carreau-Yasuda viscosity model, stenosis geometry, and wall shear stress computation.

### Climate & Ocean
Geophysical flows with Coriolis force, thermal stratification, and Ekman layers. Benchmark: Kelvin-Helmholtz instability.

### Quantum Fluids
Bose-Einstein condensate dynamics via the Gross-Pitaevskii equation. Quantized vortex nucleation and quantum turbulence.

---

## 🧠 ML Models

### PINN (Physics-Informed Neural Network)
- Embeds NS equations directly into the loss function via automatic differentiation
- Random Fourier feature embedding for multi-scale learning
- Adaptive loss weighting (grad-norm based)
- No labeled data required — learns from physics alone

### FNO (Fourier Neural Operator)
- Learns solution operators in Fourier space
- Resolution-invariant: train on 64² → infer on 256²
- Spectral convolution for global information mixing
- ~100x faster than CFD at inference

### DeepONet (Deep Operator Network)
- Maps input functions (BCs, ICs, forces) to solution functions
- Branch-trunk architecture with multi-output support
- Physics-informed variant available
- Generalizes to unseen input functions without retraining

### U-Net Surrogate
- Encoder-decoder with attention-gated skip connections
- Conditional variant with FiLM (Feature-wise Linear Modulation)
- ~1000x faster than CFD (~10ms per prediction)
- Input: conditions (obstacle mask, Re, BCs) → Output: flow fields (u, v, p)

---

## 🧠 Turbulence Discovery AI

> **Goal:** Not just simulate turbulence — *discover hidden structure inside chaos.*

The Turbulence Discovery AI is a 4-phase pipeline that compresses turbulent flow fields, learns their temporal dynamics, and discovers interpretable governing equations:

### Phase 1: Flow Autoencoder
- **Convolutional autoencoder** with residual blocks and GroupNorm
- Compresses 4-channel flow fields `(u, v, p, ω)` → latent vector `z ∈ ℝ^d`
- Physics-informed loss enforces divergence-free reconstructions (`∇·u ≈ 0`)
- Optional **variational** (VAE) mode for generative modeling

### Phase 2: Neural ODE (Latent Dynamics)
- Learns `dz/dt = f_θ(z, t)` in latent space
- Sinusoidal time embedding for temporal context
- Euler and **RK4** integrators for forward prediction
- Autoregressively predicts future flow fields

### Phase 3: Blow-up Detection
- CNN classifier predicts flow regime: `smooth / transitional / turbulent / unstable / singular_risk`
- Binary blow-up probability head: `P(blow-up) ∈ [0, 1]`
- **BKM criterion** monitoring: `∫‖ω‖_∞ dt`
- Enstrophy budget analysis, strain-vorticity alignment

### Phase 4: Symbolic Discovery (THE BIG MOVE)
- **SINDy** (Sparse Identification of Nonlinear Dynamics):
  - Builds library of candidate functions (polynomials, trig)
  - STRidge sparse regression discovers parsimonious equations
  - Output: `dz/dt = Θ(z) · ξ` where `ξ` is sparse
- **Genetic Programming**:
  - Evolves mathematical expression trees via tournament selection
  - Subtree crossover + point mutation
  - Parsimony pressure for interpretable equations

### Validation & Metrics
- DNS vs LES comparison with full turbulence metrics
- Energy spectrum analysis (Kolmogorov `k^{-5/3}` reference)
- Structure functions `S_n(r)` up to 4th order
- Empirical regularity maps across Reynolds number space

```bash
# Run the full pipeline end-to-end
python main.py --discover --no-gui

# Individual components
python main.py --symbolic     # SINDy + GP equation discovery
python main.py --stability    # Blow-up detection & Re sweeps
python main.py --regularity   # Enstrophy & dissipation vs Re
python main.py --metrics      # DNS vs LES comparison
```

> ⚠️ **Disclaimer:** The Navier-Stokes existence and smoothness problem is **unsolved** (Millennium Prize). We do NOT claim to solve it. These are empirical tools for generating regularity maps, failure predictions, and possibly new conjectures.

---

## 📊 Benchmarks

Run with `python main.py --benchmark`:

| Resolution | Steps/s | MLUPS | Method |
|------------|---------|-------|--------|
| 32×32      | ~5000+  | ~0.1  | FFT    |
| 64×64      | ~1200+  | ~0.5  | FFT    |
| 128×128    | ~300+   | ~1.6  | FFT    |
| 256×256    | ~80+    | ~5.2  | FFT    |

*MLUPS = Million Lattice-point Updates Per Second. Performance varies by hardware.*

---

## 🧪 Testing

```bash
# Run all smoke tests
python -m pytest tests/test_smoke.py -v

# Quick validation
python -m pytest tests/ -x --tb=short
```

The smoke test suite validates:
- Core 2D/3D solvers (Taylor-Green, shear layer, obstacles, pressure solvers)
- All 5 physics domains (MHD, astro, bio, climate, quantum)
- All 5 ML model architectures (PINN, FNO, DeepONet, U-Net, Turbulence NN)
- Training pipeline (data generation, FNO trainer, loss functions)
- Utility functions (vorticity, KE, enstrophy, CFL, obstacle masks)
- Visualization (scalar→RGB, velocity→RGB, streamlines)

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.py` with preset configurations:

```python
from config import taylor_green_vortex, lid_driven_cavity, PRESETS

# Use a preset
cfg = taylor_green_vortex(re=100)

# Or build custom
from config import MasterConfig, GridConfig, FluidConfig
cfg = MasterConfig(
    grid=GridConfig(nx=256, ny=256),
    fluid=FluidConfig(viscosity=0.005, dt=0.001),
)
```

Available presets: `lid_driven_cavity`, `taylor_green_vortex`, `channel_flow`, `flow_around_cylinder`, `blood_flow_artery`, `mhd_reconnection`.

---

## 🖥️ Dashboard

Launch the Streamlit web dashboard for interactive simulation control:

```bash
streamlit run dashboard/app.py
```

Features:
- Real-time simulation with parameter sliders (grid, viscosity, dt, BCs)
- Physics domain switching (all 6 domains)
- Flow visualization (vorticity, velocity, pressure, streamlines)
- Diagnostics (KE, enstrophy, divergence tracking)
- ML model status and hybrid architecture overview

---

## 📜 Governing Equations

**Incompressible Navier-Stokes:**

```
Momentum:    ∂u/∂t + (u·∇)u = −(1/ρ)∇p + ν∇²u + f
Continuity:  ∇·u = 0
```

**Projection Method (Chorin's splitting):**

1. **Advection-Diffusion:** Compute intermediate velocity u* ignoring pressure
2. **Pressure Poisson:** Solve ∇²p = (ρ/Δt)∇·u* for pressure
3. **Projection:** Correct velocity u^{n+1} = u* − (Δt/ρ)∇p

---

## 🤝 Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | Core numerics | ✅ |
| scipy | Sparse solvers, filters | ✅ |
| torch | Deep learning models | For ML features |
| matplotlib | Plotting | For visualization |
| pygame | Real-time 2D viz | Optional |
| pyvista | 3D visualization | Optional |
| streamlit | Web dashboard | Optional |
| plotly | Interactive plots | Optional |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
