"""
=============================================================================
Real-Time 2D Visualization using Pygame
Interactive fluid simulation with user controls.

Controls:
    LEFT MOUSE:  Draw obstacles / inject dye
    RIGHT MOUSE: Apply force in drag direction
    SCROLL:      Change viscosity
    1-6:         Switch physics domain
    V:           Toggle velocity arrows
    S:           Toggle streamlines
    P:           Toggle pressure view
    W:           Toggle vorticity view
    SPACE:       Pause/Resume
    R:           Reset simulation
    +/-:         Increase/decrease resolution
=============================================================================
"""

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
    and parameter modification.
    """
    
    def __init__(
        self,
        solver=None,
        width: int = 1200,
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
        self.display_mode = "vorticity"  # "velocity", "vorticity", "pressure", "speed", "direction"
        self.show_arrows = True
        self.show_streamlines = False
        self.show_obstacles = True
        self.show_info = True
        
        # Colormap
        self.colormap = "inferno"
        self.symmetric_colormap = True
        
        # Interaction state
        self.paused = False
        self.mouse_down_left = False
        self.mouse_down_right = False
        self.mouse_prev = None
        self.draw_mode = "obstacle"  # "obstacle", "force", "dye"
        self.brush_radius = 5
        
        # Physics domain
        self.physics_domain = "fluid"
        
        # Steps per frame (for speed control)
        self.steps_per_frame = 5
        
        # Performance tracking
        self.frame_times = []
    
    def _create_default_solver(self):
        """Create a default 2D solver if none provided."""
        from core.fluid_solver_2d import FluidSolver2D
        
        solver = FluidSolver2D(
            nx=200, ny=120,
            Lx=10.0, Ly=6.0,
            nu=0.005, dt=0.01,
            pressure_solver="fft",
            advection_scheme="central"
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
        
        # Font
        font = pygame.font.SysFont('Consolas', 14)
        font_large = pygame.font.SysFont('Consolas', 18)
        
        # Color constants
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        YELLOW = (255, 255, 100)
        CYAN = (100, 255, 255)
        
        running = True
        frame_count = 0
        
        print(f"\n{'='*60}")
        print(f"  Real-Time 2D Navier-Stokes Simulator")
        print(f"  Resolution: {self.solver.nx}×{self.solver.ny}")
        print(f"  Viscosity: {self.solver.nu:.4f}")
        print(f"{'='*60}")
        print(f"  Controls:")
        print(f"    Left-click:  Draw obstacles")
        print(f"    Right-drag:  Apply force")
        print(f"    Scroll:      Change viscosity")
        print(f"    1-6:         Physics domain")
        print(f"    V/S/P/W:     Toggle views")
        print(f"    SPACE:       Pause")
        print(f"    R:           Reset")
        print(f"{'='*60}\n")
        
        while running:
            frame_start = time.perf_counter()
            
            # ---- Event handling ----
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
            
            # ---- Simulation step ----
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.solver.step()
            
            # ---- Rendering ----
            self._render_frame(screen)
            
            # ---- Info overlay ----
            if self.show_info:
                self._render_info(screen, font, font_large, WHITE, YELLOW, CYAN)
            
            pygame.display.flip()
            clock.tick(self.fps)
            
            frame_time = time.perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 100:
                self.frame_times = self.frame_times[-50:]
            
            frame_count += 1
        
        pygame.quit()
    
    def _handle_keydown(self, key, pygame) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == pygame.K_ESCAPE:
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self._reset_simulation()
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
            # Cycle colormap
            cmaps = ["inferno", "viridis", "plasma", "coolwarm", "turbo"]
            idx = cmaps.index(self.colormap) if self.colormap in cmaps else 0
            self.colormap = cmaps[(idx + 1) % len(cmaps)]
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.steps_per_frame = min(20, self.steps_per_frame + 1)
        elif key == pygame.K_MINUS:
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
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
    
    def _handle_mouse_motion(self, pos):
        """Handle mouse motion for interaction."""
        if self.solver is None:
            return
        
        mx, my = pos
        nx, ny = self.solver.nx, self.solver.ny
        
        # Map screen coords to grid coords
        gx = int(mx * nx / self.width)
        gy = int(my * ny / self.height)
        
        gx = np.clip(gx, 0, nx - 1)
        gy = np.clip(gy, 0, ny - 1)
        
        if self.mouse_down_left:
            # Draw obstacle
            r = self.brush_radius
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        ix = (gx + dx) % nx
                        iy = (gy + dy) % ny
                        self.solver.obstacle[iy, ix] = True
                        self.solver.u[iy, ix] = 0
                        self.solver.v[iy, ix] = 0
            if hasattr(self.solver, 'bc_manager'):
                self.solver.bc_manager.set_obstacle(self.solver.obstacle)
        
        elif self.mouse_down_right and self.mouse_prev is not None:
            # Apply force in drag direction
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
        
        # Get field to display
        if self.display_mode == "vorticity":
            field = self.solver.get_vorticity() if hasattr(self.solver, 'get_vorticity') else \
                    np.gradient(self.solver.v, axis=1) - np.gradient(self.solver.u, axis=0)
            symmetric = True
        elif self.display_mode == "pressure":
            field = self.solver.p
            symmetric = True
        elif self.display_mode == "speed":
            field = np.sqrt(self.solver.u**2 + self.solver.v**2)
            symmetric = False
        elif self.display_mode == "direction":
            field = np.arctan2(self.solver.v, self.solver.u)
            symmetric = False
        else:
            field = np.sqrt(self.solver.u**2 + self.solver.v**2)
            symmetric = False
        
        # Convert to RGB
        cmap = "coolwarm" if symmetric else self.colormap
        rgb = FlowRenderer.scalar_to_rgb(field, cmap=cmap, symmetric=symmetric)
        
        # Mark obstacles
        if self.show_obstacles and hasattr(self.solver, 'obstacle'):
            obstacle_mask = self.solver.obstacle
            if np.any(obstacle_mask):
                rgb[obstacle_mask] = [40, 40, 40]  # Dark gray
        
        # Scale to screen size
        surface = pygame.surfarray.make_surface(
            np.transpose(rgb, (1, 0, 2))
        )
        surface = pygame.transform.scale(surface, (self.width, self.height))
        screen.blit(surface, (0, 0))
        
        # Draw velocity arrows
        if self.show_arrows:
            FlowRenderer.draw_velocity_arrows(
                screen, self.solver.u, self.solver.v,
                self.width, self.height,
                density=12, scale=15.0,
                color=(255, 255, 255)
            )
    
    def _render_info(self, screen, font, font_large, WHITE, YELLOW, CYAN):
        """Render info overlay."""
        import pygame
        
        # Semi-transparent background
        info_surface = pygame.Surface((280, 220))
        info_surface.set_alpha(180)
        info_surface.fill((20, 20, 40))
        screen.blit(info_surface, (5, 5))
        
        lines = [
            (f"Navier-Stokes Solver [{self.physics_domain.upper()}]", font_large, CYAN),
            (f"", font, WHITE),
            (f"Time: {self.solver.time:.3f}s  Step: {self.solver.step_count}", font, WHITE),
            (f"Grid: {self.solver.nx}×{self.solver.ny}  ν={self.solver.nu:.5f}", font, WHITE),
            (f"Re ≈ {1.0/max(self.solver.nu, 1e-10):.0f}", font, YELLOW),
            (f"Max|u|: {np.max(np.abs(self.solver.u)):.3f}", font, WHITE),
            (f"Display: {self.display_mode}", font, WHITE),
            (f"Steps/frame: {self.steps_per_frame}", font, WHITE),
        ]
        
        if self.frame_times:
            fps_actual = 1.0 / max(np.mean(self.frame_times), 1e-6)
            lines.append((f"FPS: {fps_actual:.1f}", font, WHITE))
        
        if self.paused:
            lines.append(("** PAUSED **", font_large, YELLOW))
        
        y_offset = 10
        for text, f, color in lines:
            if text:
                rendered = f.render(text, True, color)
                screen.blit(rendered, (15, y_offset))
            y_offset += 18
    
    def _switch_domain(self, domain: str):
        """Switch to a different physics domain."""
        self.physics_domain = domain
        print(f"\nSwitching to physics domain: {domain}")
        
        if domain == "fluid":
            from core.fluid_solver_2d import FluidSolver2D
            self.solver = FluidSolver2D(nx=200, ny=120, Lx=10, Ly=6, nu=0.005, dt=0.01)
            self.solver.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
            self.solver.bc_manager.set_periodic()
        
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
            # Wrap to provide .u, .v interface
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
        print("Simulation reset.")
