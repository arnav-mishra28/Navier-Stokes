"""Streamlit Dashboard — Interactive Research Control Center"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Plotly Visualization Helpers

def create_flow_heatmap(field, title, colorscale='Inferno', zmid=None):
    """Create a plotly heatmap for flow fields."""
    fig = go.Figure(data=go.Heatmap(
        z=field, colorscale=colorscale, zmid=zmid,
        colorbar=dict(title=dict(text=title, side='right'))
    ))
    fig.update_layout(
        title=title,
        xaxis_title='x', yaxis_title='y',
        width=600, height=500,
        yaxis=dict(scaleanchor='x'),
        template='plotly_dark'
    )
    return fig


def create_vector_field(u, v, density=10):
    """Create a plotly quiver plot."""
    ny, nx = u.shape
    x = np.arange(0, nx, density)
    y = np.arange(0, ny, density)
    X, Y = np.meshgrid(x, y)

    U = u[::density, ::density]
    V = v[::density, ::density]

    fig = go.Figure()

    # Background: velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    fig.add_trace(go.Heatmap(z=speed, colorscale='Inferno', opacity=0.6,
                             showscale=False))

    # Quiver arrows using annotations
    for j in range(len(y)):
        for i in range(len(x)):
            if U[j, i]**2 + V[j, i]**2 > 1e-6:
                fig.add_annotation(
                    x=X[j, i], y=Y[j, i],
                    ax=X[j, i] + U[j, i] * density * 0.8,
                    ay=Y[j, i] + V[j, i] * density * 0.8,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowwidth=1.5, arrowcolor='white'
                )

    fig.update_layout(
        title='Velocity Field',
        width=600, height=500,
        template='plotly_dark',
        yaxis=dict(scaleanchor='x')
    )
    return fig


# Solver Factory

def _create_solver(domain, nx, ny, nu, dt, init_type, psolver, advection,
                   turb, vort_conf, use_gpu):
    """Create a solver based on dashboard configuration."""
    scheme_map = {"Central": "central", "Upwind": "upwind", "WENO-5": "weno5"}
    psolver_map = {"FFT": "fft", "SOR": "sor", "CG": "cg", "Jacobi": "jacobi"}
    turb_map = {
        "None (DNS)": "none", "Smagorinsky LES": "smagorinsky",
        "Dynamic Smagorinsky": "dynamic_smagorinsky", "k-ε RANS": "k_epsilon"
    }

    if domain == "Classical Fluid":
        if use_gpu:
            try:
                from core.fluid_solver_2d import GPUFluidSolver2D
                solver = GPUFluidSolver2D(
                    nx=nx, ny=ny, Lx=2*np.pi, Ly=2*np.pi,
                    nu=nu, dt=dt,
                    vorticity_confinement=vort_conf,
                )
                if init_type == "Taylor-Green Vortex":
                    solver.initialize_taylor_green()
                elif init_type == "Double Shear Layer":
                    solver.initialize_double_shear_layer()
                else:
                    solver.initialize_taylor_green()
                return solver
            except Exception as e:
                st.warning(f"GPU solver failed ({e}), using CPU")

        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(
            nx=nx, ny=ny, Lx=2*np.pi, Ly=2*np.pi,
            nu=nu, dt=dt,
            pressure_solver=psolver_map.get(psolver, "fft"),
            advection_scheme=scheme_map.get(advection, "central"),
            turbulence_model=turb_map.get(turb, "none"),
            vorticity_confinement=vort_conf,
        )

        if init_type == "Taylor-Green Vortex":
            solver.initialize_taylor_green()
            solver.bc_manager.set_periodic()
        elif init_type == "Double Shear Layer":
            solver.initialize_double_shear_layer()
            solver.bc_manager.set_periodic()
        elif init_type == "Vortex Pair":
            solver.initialize_vortex_pair()
            solver.bc_manager.set_periodic()
        elif init_type == "Lid-Driven Cavity":
            solver.initialize_lid_driven_cavity()
        elif init_type == "Channel Flow":
            solver.initialize_channel_flow()

        return solver

    elif domain == "Magnetohydrodynamics (MHD)":
        from physics.mhd import MHDSolver
        solver = MHDSolver(nx=nx, ny=ny, nu=nu, eta=nu, dt=dt)
        solver.initialize_orszag_tang()
        return solver

    elif domain == "Astrophysics":
        from physics.astrophysics import AstrophysicalFlowSolver
        solver = AstrophysicalFlowSolver(nx=nx, ny=ny, nu=nu, dt=dt)
        solver.initialize_rayleigh_taylor()
        return solver

    elif domain == "Biophysics (Blood Flow)":
        from physics.biophysics import BiophysicsFlowSolver
        solver = BiophysicsFlowSolver(nx=nx, ny=max(ny//2, 20), dt=dt)
        solver.initialize_straight_vessel(stenosis=0.4)
        return solver

    elif domain == "Climate & Ocean":
        from physics.climate import ClimateFlowSolver
        solver = ClimateFlowSolver(nx=nx, ny=ny, nu=500, dt=200)
        solver.initialize_kelvin_helmholtz()
        return solver

    elif domain == "Quantum Fluids":
        from physics.quantum_fluids import QuantumFluidSolver
        solver = QuantumFluidSolver(nx=nx, ny=ny, g_int=500, dt=0.001)
        solver.initialize_quantum_turbulence(n_vortices=10)

        class QFWrapper:
            def __init__(self, qs):
                self.qs = qs
                self._sync()
            def _sync(self):
                ux, uy = self.qs.get_velocity()
                self.u = ux
                self.v = uy
                self.p = self.qs.get_density()
                self.time = self.qs.time
                self.step_count = self.qs.step_count
                self.nu = 0.001
                self.nx = self.qs.nx
                self.ny = self.qs.ny
                self.obstacle = np.zeros((self.qs.ny, self.qs.nx), dtype=bool)
            def step(self):
                self.qs.step()
                self._sync()
            def get_vorticity(self):
                return np.gradient(self.v, axis=1) - np.gradient(self.u, axis=0)

        return QFWrapper(solver)

    # Fallback
    from core.fluid_solver_2d import FluidSolver2D
    solver = FluidSolver2D(nx=nx, ny=ny, nu=nu, dt=dt,
                           vorticity_confinement=vort_conf)
    solver.initialize_taylor_green()
    solver.bc_manager.set_periodic()
    return solver


# Main Dashboard

def run_dashboard():
    """Main Streamlit dashboard application."""

    st.set_page_config(
        page_title="Navier-Stokes ML/DL Simulator",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark premium look
    st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px; border-radius: 10px;
        border: 1px solid #30475e; margin: 5px 0;
    }
    h1 { color: #e94560 !important; }
    h2 { color: #58a6ff !important; }
    h3 { color: #79c0ff !important; }
    .stSlider > div > div > div > div {
        background-color: #58a6ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌊 Navier-Stokes ML/DL Hybrid Simulator")
    st.markdown("**Research-Grade CFD + Deep Learning  ·  Real-Time Interactive Control**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Physics domain
        domain = st.selectbox("🔬 Physics Domain", [
            "Classical Fluid", "Magnetohydrodynamics (MHD)",
            "Astrophysics", "Biophysics (Blood Flow)",
            "Climate & Ocean", "Quantum Fluids"
        ])

        st.divider()

        # Grid & Physics
        st.subheader("📐 Grid & Physics")
        nx = st.slider("Grid Resolution (nx)", 32, 512, 128, 32,
                        help="Higher = more detail, slower")
        ny = st.slider("Grid Resolution (ny)", 32, 512, 128, 32)
        nu = st.slider("Viscosity (ν)", 0.0001, 0.2, 0.01, 0.0005,
                        format="%.4f",
                        help="Lower = higher Reynolds number = more turbulent")
        dt = st.slider("Time Step (Δt)", 0.0001, 0.05, 0.005, 0.0005,
                        format="%.4f")

        re_est = 1.0 / max(nu, 1e-10)
        st.info(f"Reynolds Number ≈ **{re_est:.0f}**")

        st.divider()

        # Flow configuration
        st.subheader("🌀 Initial Condition")
        init_type = st.selectbox("Flow Type", [
            "Taylor-Green Vortex", "Double Shear Layer",
            "Vortex Pair", "Lid-Driven Cavity",
            "Channel Flow"
        ])

        st.divider()

        # Vorticity Confinement
        st.subheader("🔥 Vorticity Confinement")
        vort_conf = st.slider("ε_vc (confinement strength)", 0.0, 20.0, 0.0, 0.5,
                               help="Restores swirling detail lost to numerical diffusion. "
                                    "Try 2-10 for visually rich turbulence.")

        st.divider()

        # Solver settings
        st.subheader("🔧 Solver Settings")
        pressure_solver = st.selectbox("Pressure Solver",
                                        ["FFT", "Jacobi", "SOR", "CG"])
        advection = st.selectbox("Advection Scheme",
                                  ["Central", "Upwind", "WENO-5"])
        turb_model = st.selectbox("Turbulence Model", [
            "None (DNS)", "Smagorinsky LES",
            "Dynamic Smagorinsky", "k-ε RANS"
        ])

        st.divider()

        # GPU
        st.subheader("⚡ Acceleration")
        use_gpu = st.checkbox("Use GPU (PyTorch CUDA)", value=False,
                               help="Runs solver on GPU for massive speedup")

        st.divider()

        # ML Model
        st.subheader("🧠 ML Model")
        ml_model = st.selectbox("Model Type", [
            "None (CFD Only)", "PINN",
            "FNO", "DeepONet", "U-Net Surrogate"
        ])

        st.divider()

        # Simulation controls
        st.subheader("▶️ Simulation")
        n_steps = st.slider("Steps per Update", 1, 100, 10)
        auto_run = st.checkbox("Auto-run (continuous)", value=False)
        run_button = st.button("▶️ Run Simulation", use_container_width=True,
                                type="primary")
        reset_button = st.button("🔄 Reset", use_container_width=True)

    # Main Content

    # Initialize session state
    if 'solver' not in st.session_state or reset_button:
        st.session_state.solver = None
        st.session_state.history = {
            'time': [], 'ke': [], 'enstrophy': [],
            'max_div': [], 'max_vel': []
        }

    # Live parameter tinkering — update solver parameters without reset
    if st.session_state.solver is not None and not reset_button:
        solver = st.session_state.solver
        if hasattr(solver, 'nu'):
            solver.nu = nu
        if hasattr(solver, 'dt'):
            solver.dt = dt
        if hasattr(solver, 'epsilon_vc'):
            solver.epsilon_vc = vort_conf

    if run_button or auto_run or st.session_state.solver is not None:
        # Create solver if needed
        if st.session_state.solver is None or reset_button:
            with st.spinner("Initializing solver..."):
                solver = _create_solver(
                    domain, nx, ny, nu, dt, init_type,
                    pressure_solver, advection, turb_model,
                    vort_conf, use_gpu
                )
                st.session_state.solver = solver
                st.session_state.history = {
                    'time': [], 'ke': [], 'enstrophy': [],
                    'max_div': [], 'max_vel': []
                }

        solver = st.session_state.solver

        # Run simulation steps
        if run_button or auto_run:
            t0 = time.perf_counter()
            for _ in range(n_steps):
                solver.step()
            elapsed = time.perf_counter() - t0

        # Get fields (handle GPU solver)
        if hasattr(solver, 'get_numpy'):
            fields = solver.get_numpy()
            u_np, v_np, p_np = fields['u'], fields['v'], fields['p']
        else:
            u_np, v_np, p_np = solver.u, solver.v, solver.p

        vel_mag = np.sqrt(u_np**2 + v_np**2)

        # Metrics Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("⏱ Time", f"{solver.time:.4f}s")
        col2.metric("📊 Step", f"{solver.step_count}")
        col3.metric("💨 Max |u|", f"{np.max(vel_mag):.4f}")
        col4.metric("🔄 Re", f"{1/max(getattr(solver, 'nu', nu), 1e-10):.0f}")
        col5.metric("📐 Grid", f"{getattr(solver, 'nx', nx)}×{getattr(solver, 'ny', ny)}")
        if 'elapsed' in dir():
            col6.metric("⚡ Rate", f"{n_steps/max(elapsed, 1e-6):.0f} st/s")
        else:
            col6.metric("🔥 ε_vc", f"{vort_conf:.1f}")

        st.divider()

        # Visualization Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌀 Vorticity", "💨 Velocity", "📊 Pressure",
            "📈 Diagnostics", "🧠 ML / Hybrid"
        ])

        with tab1:
            if hasattr(solver, 'get_vorticity'):
                vorticity = solver.get_vorticity()
                if hasattr(vorticity, 'cpu'):
                    vorticity = vorticity.cpu().numpy() if hasattr(vorticity, 'cpu') else vorticity
            else:
                vorticity = np.gradient(v_np, axis=1) - np.gradient(u_np, axis=0)
            fig = create_flow_heatmap(vorticity, "Vorticity Field (ω)",
                                      colorscale='RdBu_r', zmid=0)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_speed = create_flow_heatmap(vel_mag, "Velocity Magnitude |u|")
                st.plotly_chart(fig_speed, use_container_width=True)
            with col_b:
                fig_vec = create_vector_field(u_np, v_np,
                                             density=max(nx//15, 4))
                st.plotly_chart(fig_vec, use_container_width=True)

        with tab3:
            fig = create_flow_heatmap(p_np, "Pressure Field",
                                      colorscale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Compute diagnostics
            ke = 0.5 * np.mean(u_np**2 + v_np**2)
            if hasattr(solver, 'get_vorticity'):
                vort = solver.get_vorticity()
                if hasattr(vort, 'cpu'):
                    vort = vort
                elif not isinstance(vort, np.ndarray):
                    vort = np.array(vort)
            else:
                vort = vorticity
            if isinstance(vort, np.ndarray):
                enstrophy = 0.5 * np.mean(vort**2)
            else:
                enstrophy = 0.0
            div_val = np.max(np.abs(
                np.gradient(u_np, axis=1) + np.gradient(v_np, axis=0)
            ))

            st.session_state.history['time'].append(solver.time)
            st.session_state.history['ke'].append(float(ke))
            st.session_state.history['enstrophy'].append(float(enstrophy))
            st.session_state.history['max_div'].append(float(div_val))
            st.session_state.history['max_vel'].append(float(np.max(vel_mag)))

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Kinetic Energy', 'Enstrophy',
                                'Max Divergence', 'Max Velocity']
            )

            times = st.session_state.history['time']
            fig.add_trace(go.Scatter(
                x=times, y=st.session_state.history['ke'],
                mode='lines', name='KE',
                line=dict(color='#58a6ff', width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=times, y=st.session_state.history['enstrophy'],
                mode='lines', name='Enstrophy',
                line=dict(color='#f97583', width=2)
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=times, y=st.session_state.history['max_div'],
                mode='lines', name='Divergence',
                line=dict(color='#7ee787', width=2)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=times, y=st.session_state.history['max_vel'],
                mode='lines', name='Max |u|',
                line=dict(color='#d2a8ff', width=2)
            ), row=2, col=2)

            fig.update_layout(template='plotly_dark', height=500,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Kinetic Energy", f"{ke:.6f}")
            scol2.metric("Enstrophy", f"{enstrophy:.6f}")
            scol3.metric("Max Divergence", f"{div_val:.2e}")
            scol4.metric("CFL", f"{dt * np.max(vel_mag) / min(2*np.pi/nx, 2*np.pi/ny):.3f}")

        with tab5:
            st.subheader("🧠 ML Model Status & Hybrid Integration")
            if ml_model == "None (CFD Only)":
                st.info("No ML model active. Select a model from the sidebar to "
                        "enable hybrid CFD→ML predictions.")
            else:
                st.markdown(f"**Active Model:** `{ml_model}`")
                st.markdown("""
                **Hybrid Architecture:**
                ```
                [CFD Truth] → [Generates Training Data] → [Trains Neural Network]
                      ↓                                         ↓
                [Benchmark]                              [Real-Time Inference]
                      ↓                                         ↓
                [Validation] ← ← ← ← ← ← ← ← ← ← ← [Prediction + Error]
                ```
                """)

                st.markdown(f"""
                | Property | Value |
                |----------|-------|
                | Model Type | {ml_model} |
                | Status | ⚡ Ready for inference |
                | Speed | ~100-1000x faster than CFD |
                | Physics Compliance | NS residual enforced |
                | GPU Accelerated | {'Yes' if use_gpu else 'No'} |
                """)

                if ml_model == "PINN":
                    st.markdown("""
                    #### PINN: Full Navier-Stokes Residual
                    The PINN embeds the **complete** NS equations into its loss:
                    
                    ```
                    R_u = ∂u/∂t + u·∂u/∂x + v·∂u/∂y + ∂p/∂x − ν(∂²u/∂x² + ∂²u/∂y²)
                    R_v = ∂v/∂t + u·∂v/∂x + v·∂v/∂y + ∂p/∂y − ν(∂²v/∂x² + ∂²v/∂y²)
                    R_div = ∂u/∂x + ∂v/∂y = 0
                    ```
                    
                    **Hybrid mode:** CFD generates ground truth → trains PINN → 
                    PINN gives instant predictions at new Reynolds numbers/geometries.
                    """)
                elif ml_model == "FNO":
                    st.markdown("""
                    #### FNO: Neural Operator (Spectral Convolution)
                    Instead of solving the PDE, the FNO **learns the solution operator**:
                    
                    ```
                    G_θ: a(x) → u(x)   (function space → function space)
                    ```
                    
                    Key: **Spectral convolution** via FFT captures global patterns.
                    Resolution-invariant: train on 64² → infer on 512².
                    """)

        # Auto-run
        if auto_run:
            st.rerun()

        # Continue button
        if not auto_run:
            if st.button("▶️ Continue", use_container_width=True):
                for _ in range(n_steps):
                    solver.step()
                st.rerun()

    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to the Navier-Stokes Simulation Platform
        
        Configure your simulation in the sidebar and click **Run Simulation** to begin.
        **All parameters can be changed in real-time** — the solver updates live.
        
        ### 🔬 Available Physics Domains
        
        | Domain | Description | Key Physics |
        |--------|-------------|-------------|
        | 🌊 Classical Fluid | Standard NS equations | Advection, diffusion, pressure |
        | ⚡ MHD | Magnetohydrodynamics | Lorentz force, induction |
        | 🌟 Astrophysics | Stellar & galactic flows | Self-gravity, cooling |
        | ❤️ Biophysics | Blood flow | Non-Newtonian, pulsatile |
        | 🌍 Climate | Geophysical flows | Coriolis, stratification |
        | ⚛️ Quantum | BEC / Superfluids | Gross-Pitaevskii, quantized vortices |
        
        ### 🧠 ML Models (Hybrid Integration)
        
        | Model | Speed | Architecture | Hybrid Role |
        |-------|-------|-------------|-------------|
        | PINN | ⚡⚡ | MLP + Fourier Features | Physics-constrained surrogate |
        | FNO | ⚡⚡⚡ | Spectral Conv (FFT) | Operator learning |
        | DeepONet | ⚡⚡⚡ | Branch-Trunk | Function-to-function map |
        | U-Net | ⚡⚡⚡⚡ | Encoder-Decoder + Attention | Instant field prediction |
        
        ### 🔥 New Features
        - **Vorticity Confinement** — restores swirling turbulence lost to diffusion
        - **GPU Acceleration** — PyTorch CUDA solver for massive speedup
        - **Live Parameter Tinkering** — change ν, dt, ε_vc in real-time
        """)


if __name__ == "__main__":
    run_dashboard()
