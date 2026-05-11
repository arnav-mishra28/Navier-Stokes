"""Real-Time 2D Visualization using Pygame"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealtimeVisualizer2D:
    """
    Real-time 2D fluid simulation viewer using Pygame.
    
    Renders flow fields (velocity, vorticity, pressure) with
    interactive user controls for obstacle placement, force injection,
    and REAL-TIME parameter modification.
    """
    
    def __init__(
        self,
        solver=None,
        width: int = 1280,
        height: int = 800,
        fps: int = 60,
        title: str = "Navier-Stokes Real-Time Simulator",
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.title = title
        self.solver = solver
        
        # Display mode
        self.display_mode = "vorticity"
        self.show_arrows = True
        self.show_streamlines = False
        self.show_obstacles = True
        self.show_info = True
        self.show_help = False
        
        # Colormap
        self.colormap = "inferno"
        self.colormap_idx = 0
        self.cmaps = ["inferno", "viridis", "plasma", "coolwarm", "turbo", "magma"]
        
        # Interaction state
        self.paused = False
        self.mouse_down_left = False
        self.mouse_down_right = False
        self.mouse_prev = None
        self.brush_radius = 5
        
        # Physics domain
        self.physics_domain = "fluid"
        
        # Solver params (tinkerable)
        self.vort_conf = 0.0
        self.use_gpu = False
        self.turb_model_idx = 0
        self.turb_models = ["none", "smagorinsky", "dynamic_smagorinsky"]
        
        # Steps per frame
        self.steps_per_frame = 5
        
        # Performance tracking
        self.frame_times = []
        self.sim_times = []
    
    def _create_default_solver(self, use_gpu=False):
        """Create a default 2D solver."""
        if use_gpu:
            try:
                from core.fluid_solver_2d import GPUFluidSolver2D
                solver = GPUFluidSolver2D(
                    nx=200, ny=120,
                    Lx=10.0, Ly=6.0,
                    nu=0.005, dt=0.01,
                    vorticity_confinement=self.vort_conf,
                )
                solver.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
                return solver
            except Exception:
                pass
        
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(
            nx=200, ny=120,
            Lx=10.0, Ly=6.0,
            nu=0.005, dt=0.01,
            pressure_solver="fft",
            advection_scheme="central",
            vorticity_confinement=self.vort_conf,
        )
        solver.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
        solver.bc_manager.set_periodic()
        return solver
    
    def run(self):
        """Main visualization loop."""
        import pygame
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()
        
        if self.solver is None:
            self.solver = self._create_default_solver()
        
        # Fonts
        font = pygame.font.SysFont('Consolas', 14)
        font_large = pygame.font.SysFont('Consolas', 18)
        font_title = pygame.font.SysFont('Consolas', 22, bold=True)
        
        # Colors
        WHITE = (255, 255, 255)
        YELLOW = (255, 255, 100)
        CYAN = (100, 255, 255)
        GREEN = (100, 255, 100)
        RED = (255, 100, 100)
        ORANGE = (255, 180, 80)
        DIM = (150, 150, 160)
        
        running = True
        
        print(f"\n{'='*65}")
        print(f"  🌊 Real-Time 2D Navier-Stokes Simulator")
        print(f"  Resolution: {self.solver.nx}×{getattr(self.solver, 'ny', '?')}")
        print(f"  Press H for full controls help")
        print(f"{'='*65}\n")
        
        while running:
            frame_start = time.perf_counter()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event.key, pygame)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mouse_down_left = True
                        self.mouse_prev = event.pos
                    elif event.button == 3:
                        self.mouse_down_right = True
                        self.mouse_prev = event.pos
                    elif event.button == 4:  # Scroll up
                        self.solver.nu *= 1.1
                    elif event.button == 5:  # Scroll down
                        self.solver.nu = max(1e-5, self.solver.nu * 0.9)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_down_left = False
                    elif event.button == 3:
                        self.mouse_down_right = False
                    self.mouse_prev = None
                
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)
            
            # Simulation step
            if not self.paused:
                sim_start = time.perf_counter()
                for _ in range(self.steps_per_frame):
                    self.solver.step()
                sim_elapsed = time.perf_counter() - sim_start
                self.sim_times.append(sim_elapsed)
                if len(self.sim_times) > 100:
                    self.sim_times = self.sim_times[-50:]
            
            # Rendering
            self._render_frame(screen)
            
            # Info overlay
            if self.show_info:
                self._render_info(screen, font, font_large, font_title,
                                  WHITE, YELLOW, CYAN, GREEN, RED, ORANGE, DIM)
            
            # Help overlay
            if self.show_help:
                self._render_help(screen, font, WHITE, YELLOW, CYAN)
            
            pygame.display.flip()
            clock.tick(self.fps)
            
            frame_time = time.perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 100:
                self.frame_times = self.frame_times[-50:]
        
        pygame.quit()
    
    def _handle_keydown(self, key, pygame) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == pygame.K_ESCAPE:
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self._reset_simulation()
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_i:
            self.show_info = not self.show_info
        elif key == pygame.K_v:
            self.show_arrows = not self.show_arrows
        elif key == pygame.K_s:
            self.show_streamlines = not self.show_streamlines
        elif key == pygame.K_p:
            self.display_mode = "pressure"
        elif key == pygame.K_w:
            self.display_mode = "vorticity"
        elif key == pygame.K_m:
            self.display_mode = "speed"
        elif key == pygame.K_d:
            self.display_mode = "direction"
        elif key == pygame.K_c:
            self.colormap_idx = (self.colormap_idx + 1) % len(self.cmaps)
            self.colormap = self.cmaps[self.colormap_idx]
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.steps_per_frame = min(30, self.steps_per_frame + 1)
        elif key == pygame.K_MINUS:
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        # Vorticity confinement
        elif key == pygame.K_f:
            if self.vort_conf == 0:
                self.vort_conf = 5.0
            else:
                self.vort_conf = 0.0
            if hasattr(self.solver, 'epsilon_vc'):
                self.solver.epsilon_vc = self.vort_conf
        elif key == pygame.K_UP:
            self.vort_conf = min(50.0, self.vort_conf + 1.0)
            if hasattr(self.solver, 'epsilon_vc'):
                self.solver.epsilon_vc = self.vort_conf
        elif key == pygame.K_DOWN:
            self.vort_conf = max(0.0, self.vort_conf - 1.0)
            if hasattr(self.solver, 'epsilon_vc'):
                self.solver.epsilon_vc = self.vort_conf
        # Time step
        elif key == pygame.K_RIGHT:
            self.solver.dt = min(0.1, self.solver.dt * 1.2)
        elif key == pygame.K_LEFT:
            self.solver.dt = max(1e-4, self.solver.dt * 0.8)
        # Clear obstacles
        elif key == pygame.K_o:
            if hasattr(self.solver, 'obstacle'):
                self.solver.obstacle[:] = False
                if hasattr(self.solver, 'bc_manager'):
                    self.solver.bc_manager.set_obstacle(self.solver.obstacle)
        # Turbulence model
        elif key == pygame.K_t:
            self.turb_model_idx = (self.turb_model_idx + 1) % len(self.turb_models)
            self._rebuild_with_turb()
        # GPU toggle
        elif key == pygame.K_g:
            self.use_gpu = not self.use_gpu
            self._reset_simulation()
        # Domain switching
        elif key == pygame.K_1:
            self._switch_domain("fluid")
        elif key == pygame.K_2:
            self._switch_domain("mhd")
        elif key == pygame.K_3:
            self._switch_domain("astrophysics")
        elif key == pygame.K_4:
            self._switch_domain("biophysics")
        elif key == pygame.K_5:
            self._switch_domain("climate")
        elif key == pygame.K_6:
            self._switch_domain("quantum")
        return True
    
    def _rebuild_with_turb(self):
        """Rebuild solver with new turbulence model."""
        if self.physics_domain != "fluid":
            return
        turb = self.turb_models[self.turb_model_idx]
        from core.fluid_solver_2d import FluidSolver2D
        solver = FluidSolver2D(
            nx=200, ny=120, Lx=10, Ly=6,
            nu=self.solver.nu, dt=self.solver.dt,
            pressure_solver="fft", advection_scheme="central",
            turbulence_model=turb,
            vorticity_confinement=self.vort_conf,
        )
        solver.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
        solver.bc_manager.set_periodic()
        self.solver = solver
    
    def _handle_mouse_motion(self, pos):
        """Handle mouse motion for interaction."""
        if self.solver is None:
            return
        
        mx, my = pos
        nx = getattr(self.solver, 'nx', 200)
        ny = getattr(self.solver, 'ny', 120)
        
        gx = int(mx * nx / self.width)
        gy = int(my * ny / self.height)
        gx = np.clip(gx, 0, nx - 1)
        gy = np.clip(gy, 0, ny - 1)
        
        # Get u/v arrays (handle GPU solver)
        if hasattr(self.solver, 'get_numpy'):
            # GPU solver — skip direct mutation (too slow for mouse interaction)
            return
        
        if self.mouse_down_left:
            r = self.brush_radius
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        ix = (gx + dx) % nx
                        iy = (gy + dy) % ny
                        if hasattr(self.solver, 'obstacle'):
                            self.solver.obstacle[iy, ix] = True
                        self.solver.u[iy, ix] = 0
                        self.solver.v[iy, ix] = 0
            if hasattr(self.solver, 'bc_manager'):
                self.solver.bc_manager.set_obstacle(self.solver.obstacle)
        
        elif self.mouse_down_right and self.mouse_prev is not None:
            pmx, pmy = self.mouse_prev
            fx = (mx - pmx) * 0.5
            fy = (my - pmy) * 0.5
            
            r = self.brush_radius * 2
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        ix = (gx + dx) % nx
                        iy = (gy + dy) % ny
                        weight = np.exp(-(dx*dx + dy*dy) / (r*r/2))
                        self.solver.u[iy, ix] += fx * weight * 0.1
                        self.solver.v[iy, ix] += fy * weight * 0.1
            
            self.mouse_prev = pos
    
    def _render_frame(self, screen):
        """Render current field to screen."""
        import pygame
        from visualization.renderer import FlowRenderer
        
        # Get numpy arrays
        if hasattr(self.solver, 'get_numpy'):
            fields = self.solver.get_numpy()
            u_np, v_np, p_np = fields['u'], fields['v'], fields['p']
        else:
            u_np, v_np, p_np = self.solver.u, self.solver.v, self.solver.p
        
        # Get field to display
        if self.display_mode == "vorticity":
            if hasattr(self.solver, 'get_vorticity'):
                field = self.solver.get_vorticity()
                if hasattr(field, 'cpu'):
                    field = field
                if not isinstance(field, np.ndarray):
                    field = np.array(field)
            else:
                field = np.gradient(v_np, axis=1) - np.gradient(u_np, axis=0)
            symmetric = True
        elif self.display_mode == "pressure":
            field = p_np
            symmetric = True
        elif self.display_mode == "speed":
            field = np.sqrt(u_np**2 + v_np**2)
            symmetric = False
        elif self.display_mode == "direction":
            field = np.arctan2(v_np, u_np)
            symmetric = False
        else:
            field = np.sqrt(u_np**2 + v_np**2)
            symmetric = False
        
        # Convert to RGB
        cmap = "coolwarm" if symmetric else self.colormap
        rgb = FlowRenderer.scalar_to_rgb(field, cmap=cmap, symmetric=symmetric)
        
        # Mark obstacles
        if self.show_obstacles and hasattr(self.solver, 'obstacle'):
            obstacle_mask = self.solver.obstacle
            if isinstance(obstacle_mask, np.ndarray) and np.any(obstacle_mask):
                rgb[obstacle_mask] = [40, 40, 40]
        
        # Scale to screen size
        surface = pygame.surfarray.make_surface(
            np.transpose(rgb, (1, 0, 2))
        )
        surface = pygame.transform.scale(surface, (self.width, self.height))
        screen.blit(surface, (0, 0))
        
        # Draw velocity arrows
        if self.show_arrows:
            FlowRenderer.draw_velocity_arrows(
                screen, u_np, v_np,
                self.width, self.height,
                density=12, scale=15.0,
                color=(255, 255, 255)
            )
    
    def _render_info(self, screen, font, font_large, font_title,
                     WHITE, YELLOW, CYAN, GREEN, RED, ORANGE, DIM):
        """Render info overlay with real-time parameter values."""
        import pygame
        
        panel_w, panel_h = 320, 310
        info_surface = pygame.Surface((panel_w, panel_h))
        info_surface.set_alpha(200)
        info_surface.fill((12, 17, 23))
        # Border
        pygame.draw.rect(info_surface, (48, 54, 61), (0, 0, panel_w, panel_h), 1)
        screen.blit(info_surface, (5, 5))
        
        nu_val = getattr(self.solver, 'nu', 0.01)
        dt_val = getattr(self.solver, 'dt', 0.01)
        re_val = 1.0 / max(nu_val, 1e-10)
        
        vc_color = GREEN if self.vort_conf > 0 else DIM
        gpu_color = GREEN if self.use_gpu else DIM
        
        lines = [
            (f"NAVIER-STOKES [{self.physics_domain.upper()}]", font_title, CYAN),
            ("", font, WHITE),
            (f"t = {self.solver.time:.3f}s    step = {self.solver.step_count}", font, WHITE),
            (f"Grid: {getattr(self.solver, 'nx', '?')}×{getattr(self.solver, 'ny', '?')}    cmap: {self.colormap}", font, DIM),
            ("", font, WHITE),
            (f"ν = {nu_val:.5f}  (scroll to adjust)", font, YELLOW),
            (f"dt = {dt_val:.5f}  (←/→ to adjust)", font, YELLOW),
            (f"Re ≈ {re_val:.0f}", font, ORANGE),
            ("", font, WHITE),
            (f"Vort. Confinement: ε={self.vort_conf:.1f}  (F/↑/↓)", font, vc_color),
            (f"GPU: {'ON' if self.use_gpu else 'OFF'}  (G)", font, gpu_color),
            (f"Turb: {self.turb_models[self.turb_model_idx]}  (T)", font, DIM),
            (f"Display: {self.display_mode}    Steps/f: {self.steps_per_frame}", font, DIM),
        ]
        
        if self.frame_times:
            fps_actual = 1.0 / max(np.mean(self.frame_times), 1e-6)
            sim_rate = self.steps_per_frame / max(np.mean(self.sim_times or [1]), 1e-6) if self.sim_times else 0
            lines.append(("", font, WHITE))
            lines.append((f"FPS: {fps_actual:.1f}   Sim: {sim_rate:.0f} st/s", font, GREEN))
        
        if self.paused:
            lines.append(("", font, WHITE))
            lines.append(("▌▌ PAUSED  (SPACE to resume)", font_large, RED))
        
        y_offset = 10
        for text, f, color in lines:
            if text:
                rendered = f.render(text, True, color)
                screen.blit(rendered, (15, y_offset))
            y_offset += 18
    
    def _render_help(self, screen, font, WHITE, YELLOW, CYAN):
        """Render full help overlay."""
        import pygame
        
        panel_w, panel_h = 400, 460
        help_surface = pygame.Surface((panel_w, panel_h))
        help_surface.set_alpha(230)
        help_surface.fill((12, 17, 23))
        pygame.draw.rect(help_surface, (88, 166, 255), (0, 0, panel_w, panel_h), 2)
        
        x_pos = self.width - panel_w - 10
        screen.blit(help_surface, (x_pos, 5))
        
        controls = [
            ("CONTROLS", CYAN),
            ("", WHITE),
            ("Left-click     Draw obstacles", YELLOW),
            ("Right-drag     Apply force", YELLOW),
            ("Scroll         Adjust viscosity (ν)", YELLOW),
            ("←/→            Adjust time step (dt)", YELLOW),
            ("↑/↓            Adjust vort. confinement (ε)", YELLOW),
            ("", WHITE),
            ("F              Toggle vort. confinement", WHITE),
            ("G              Toggle GPU solver", WHITE),
            ("T              Cycle turbulence model", WHITE),
            ("O              Clear obstacles", WHITE),
            ("C              Cycle colormap", WHITE),
            ("+/-            Sim speed (steps/frame)", WHITE),
            ("", WHITE),
            ("W / P / M / D  Vorticity/Pressure/Speed/Dir", WHITE),
            ("V              Toggle velocity arrows", WHITE),
            ("S              Toggle streamlines", WHITE),
            ("I              Toggle info panel", WHITE),
            ("", WHITE),
            ("1-6            Switch physics domain", WHITE),
            ("R              Reset simulation", WHITE),
            ("SPACE          Pause/Resume", WHITE),
            ("ESC            Quit", WHITE),
            ("H              Toggle this help", WHITE),
        ]
        
        y = 10
        for text, color in controls:
            if text:
                rendered = font.render(text, True, color)
                screen.blit(rendered, (x_pos + 15, y))
            y += 17
    
    def _switch_domain(self, domain: str):
        """Switch to a different physics domain."""
        self.physics_domain = domain
        print(f"\n  Switching to: {domain}")
        
        if domain == "fluid":
            self.solver = self._create_default_solver(self.use_gpu)
        
        elif domain == "mhd":
            from physics.mhd import MHDSolver
            self.solver = MHDSolver(nx=128, ny=128, nu=0.005, eta=0.005, dt=0.005)
            self.solver.initialize_orszag_tang()
        
        elif domain == "astrophysics":
            from physics.astrophysics import AstrophysicalFlowSolver
            self.solver = AstrophysicalFlowSolver(nx=128, ny=128, nu=0.01, dt=0.005)
            self.solver.initialize_rayleigh_taylor()
        
        elif domain == "biophysics":
            from physics.biophysics import BiophysicsFlowSolver
            self.solver = BiophysicsFlowSolver(nx=200, ny=50, dt=0.0005)
            self.solver.initialize_straight_vessel(stenosis=0.5)
        
        elif domain == "climate":
            from physics.climate import ClimateFlowSolver
            self.solver = ClimateFlowSolver(nx=128, ny=128, nu=500, dt=500)
            self.solver.initialize_kelvin_helmholtz()
        
        elif domain == "quantum":
            from physics.quantum_fluids import QuantumFluidSolver
            self.solver = QuantumFluidSolver(nx=256, ny=256, g_int=500, dt=0.0005)
            self.solver.initialize_quantum_turbulence(n_vortices=15)
            self._wrap_quantum_solver()
    
    def _wrap_quantum_solver(self):
        """Wrap quantum solver to expose u,v interface."""
        original_solver = self.solver
        
        class QuantumWrapper:
            def __init__(self, qs):
                self.qs = qs
                self.time = qs.time
                self.step_count = qs.step_count
                self.nx = qs.nx
                self.ny = qs.ny
                self.nu = 0.001
                self.dt = qs.dt
                self.epsilon_vc = 0.0
                ux, uy = qs.get_velocity()
                self.u = ux
                self.v = uy
                self.p = qs.get_density()
                self.obstacle = np.zeros((qs.ny, qs.nx), dtype=bool)
            
            def step(self):
                self.qs.step()
                ux, uy = self.qs.get_velocity()
                self.u = ux
                self.v = uy
                self.p = self.qs.get_density()
                self.time = self.qs.time
                self.step_count = self.qs.step_count
            
            def get_vorticity(self):
                return np.gradient(self.v, axis=1) - np.gradient(self.u, axis=0)
        
        self.solver = QuantumWrapper(original_solver)
    
    def _reset_simulation(self):
        """Reset current simulation."""
        self._switch_domain(self.physics_domain)
        print("  Simulation reset.")
