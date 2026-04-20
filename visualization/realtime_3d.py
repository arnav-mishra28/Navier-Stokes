"""
=============================================================================
Real-Time 3D Visualization using Matplotlib + PyVista
Renders volumetric flow fields with isosurfaces, streamlines,
and interactive controls.
=============================================================================
"""

import numpy as np
import sys
import os
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealtimeVisualizer3D:
    """
    3D flow field visualization using Matplotlib (fallback) or PyVista.
    
    Displays:
        - Velocity magnitude isosurfaces
        - Vorticity isosurfaces (Q-criterion)
        - Streamlines
        - Vector fields on slices
        - Pressure contours
    """
    
    def __init__(self, solver=None, backend: str = "matplotlib"):
        """
        Args:
            solver: 3D flow solver instance
            backend: "matplotlib" or "pyvista"
        """
        self.solver = solver
        self.backend = backend
        self.fig = None
        self.axes = None
    
    def _create_default_solver(self):
        """Create default 3D solver."""
        from core.fluid_solver_3d import FluidSolver3D
        solver = FluidSolver3D(nx=32, ny=32, nz=32, nu=0.01, dt=0.01)
        solver.initialize_taylor_green_3d()
        return solver
    
    def run_matplotlib(self, n_frames: int = 200, interval: int = 50):
        """
        Run 3D visualization using matplotlib animation with 2D slices.
        
        Shows a 2×2 grid:
            - XY slice of velocity magnitude
            - XZ slice of velocity magnitude
            - Vorticity on XY slice
            - Energy spectrum
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.colors import Normalize
        
        if self.solver is None:
            self.solver = self._create_default_solver()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('3D Navier-Stokes: Taylor-Green Vortex', fontsize=16, fontweight='bold')
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Initialize plots
        mid_z = self.solver.nz // 2
        mid_y = self.solver.ny // 2
        
        vel_mag = np.sqrt(self.solver.u**2 + self.solver.v**2 + self.solver.w**2)
        
        im1 = axes[0, 0].imshow(vel_mag[mid_z, :, :], cmap='inferno', animated=True)
        axes[0, 0].set_title('XY Slice - Velocity Magnitude')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        im2 = axes[0, 1].imshow(vel_mag[:, mid_y, :], cmap='inferno', animated=True)
        axes[0, 1].set_title('XZ Slice - Velocity Magnitude')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # Vorticity
        omega_z = ((np.roll(self.solver.v, -1, 2) - np.roll(self.solver.v, 1, 2)) /
                   (2*self.solver.dx) -
                   (np.roll(self.solver.u, -1, 1) - np.roll(self.solver.u, 1, 1)) /
                   (2*self.solver.dy))
        
        im3 = axes[1, 0].imshow(omega_z[mid_z, :, :], cmap='coolwarm', animated=True)
        axes[1, 0].set_title('XY Slice - Vorticity (ωz)')
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        
        # Energy history
        line_ke, = axes[1, 1].plot([], [], 'r-', lw=2, label='KE')
        line_ens, = axes[1, 1].plot([], [], 'b-', lw=2, label='Enstrophy')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].set_title('Energy Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        time_text = fig.text(0.02, 0.98, '', fontsize=12, va='top',
                           fontfamily='monospace', color='white',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        def update(frame):
            # Step simulation
            self.solver.advance(5, record_history=True)
            
            vel_mag = np.sqrt(self.solver.u**2 + self.solver.v**2 + self.solver.w**2)
            
            # Update XY slice
            im1.set_array(vel_mag[mid_z, :, :])
            im1.set_clim(0, np.max(vel_mag) + 1e-6)
            
            # Update XZ slice
            im2.set_array(vel_mag[:, mid_y, :])
            im2.set_clim(0, np.max(vel_mag) + 1e-6)
            
            # Update vorticity
            omega_z = ((np.roll(self.solver.v, -1, 2) - np.roll(self.solver.v, 1, 2)) /
                       (2*self.solver.dx) -
                       (np.roll(self.solver.u, -1, 1) - np.roll(self.solver.u, 1, 1)) /
                       (2*self.solver.dy))
            im3.set_array(omega_z[mid_z, :, :])
            v_max = max(np.max(np.abs(omega_z)), 1e-6)
            im3.set_clim(-v_max, v_max)
            
            # Update energy plot
            if self.solver.history['time']:
                times = self.solver.history['time']
                ke = self.solver.history['kinetic_energy']
                ens = self.solver.history['enstrophy']
                line_ke.set_data(times, ke)
                line_ens.set_data(times, ens)
                axes[1, 1].relim()
                axes[1, 1].autoscale_view()
            
            time_text.set_text(
                f"t = {self.solver.time:.3f}  |  step = {self.solver.step_count}\n"
                f"max|u| = {np.max(vel_mag):.4f}  |  ν = {self.solver.nu:.4f}"
            )
            
            return [im1, im2, im3, line_ke, line_ens, time_text]
        
        ani = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=interval, blit=False
        )
        
        plt.show()
        return ani
    
    def run_pyvista(self):
        """
        Run 3D visualization using PyVista (GPU-accelerated).
        
        Shows:
            - Volume rendering of velocity magnitude
            - Isosurfaces of Q-criterion (vortex cores)
            - Streamlines through the domain
        """
        try:
            import pyvista as pv
        except ImportError:
            print("PyVista not available, falling back to matplotlib.")
            return self.run_matplotlib()
        
        if self.solver is None:
            self.solver = self._create_default_solver()
        
        # Run a few steps first
        self.solver.advance(20, record_history=False)
        
        # Create structured grid
        nx, ny, nz = self.solver.nx, self.solver.ny, self.solver.nz
        
        x = np.linspace(0, self.solver.Lx, nx)
        y = np.linspace(0, self.solver.Ly, ny)
        z = np.linspace(0, self.solver.Lz, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add flow data
        vel_mag = np.sqrt(self.solver.u**2 + self.solver.v**2 + self.solver.w**2)
        grid['velocity_magnitude'] = vel_mag.flatten(order='F')
        grid['velocity'] = np.column_stack([
            self.solver.u.flatten(order='F'),
            self.solver.v.flatten(order='F'),
            self.solver.w.flatten(order='F')
        ])
        grid['pressure'] = self.solver.p.flatten(order='F')
        
        # Compute Q-criterion
        omega_x = ((np.roll(self.solver.w, -1, 1) - np.roll(self.solver.w, 1, 1)) / (2*self.solver.dy) -
                   (np.roll(self.solver.v, -1, 2) - np.roll(self.solver.v, 1, 2)) / (2*self.solver.dz))
        omega_y = ((np.roll(self.solver.u, -1, 2) - np.roll(self.solver.u, 1, 2)) / (2*self.solver.dz) -
                   (np.roll(self.solver.w, -1, 0) - np.roll(self.solver.w, 1, 0)) / (2*self.solver.dx))
        omega_z = ((np.roll(self.solver.v, -1, 0) - np.roll(self.solver.v, 1, 0)) / (2*self.solver.dx) -
                   (np.roll(self.solver.u, -1, 1) - np.roll(self.solver.u, 1, 1)) / (2*self.solver.dy))
        
        vorticity_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        grid['vorticity'] = vorticity_mag.flatten(order='F')
        
        # Setup plotter
        plotter = pv.Plotter(window_size=(1400, 900))
        plotter.set_background('black')
        
        # Volume rendering
        plotter.add_volume(
            grid, scalars='velocity_magnitude',
            cmap='inferno', opacity='sigmoid',
            shade=True
        )
        
        # Add streamlines
        try:
            seed = pv.Plane(
                center=(self.solver.Lx/2, self.solver.Ly/2, self.solver.Lz/2),
                direction=(0, 0, 1),
                i_size=self.solver.Lx*0.5, j_size=self.solver.Ly*0.5,
                i_resolution=5, j_resolution=5
            )
            streamlines = grid.streamlines_from_source(
                seed, vectors='velocity',
                max_time=10.0, integration_direction='both'
            )
            if streamlines.n_points > 0:
                plotter.add_mesh(streamlines, color='cyan', line_width=2, opacity=0.7)
        except Exception:
            pass  # Streamlines may fail for some configurations
        
        # Add annotations
        plotter.add_text(
            f"3D Navier-Stokes: {nx}×{ny}×{nz}\nν={self.solver.nu:.4f}",
            position='upper_left', font_size=12, color='white'
        )
        
        plotter.add_axes()
        plotter.show()
    
    def run(self, backend: Optional[str] = None):
        """Run visualization with specified backend."""
        backend = backend or self.backend
        
        if backend == "pyvista":
            return self.run_pyvista()
        else:
            return self.run_matplotlib()
    
    def create_snapshot(self, filename: str = "flow_3d.png"):
        """Create a static snapshot of the 3D flow field."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if self.solver is None:
            self.solver = self._create_default_solver()
            self.solver.advance(50)
        
        fig = plt.figure(figsize=(16, 12))
        
        vel_mag = np.sqrt(self.solver.u**2 + self.solver.v**2 + self.solver.w**2)
        mid = self.solver.nz // 2
        
        # XY slice
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(vel_mag[mid, :, :], cmap='inferno', origin='lower')
        ax1.set_title('Z-midplane velocity')
        plt.colorbar(im1, ax=ax1)
        
        # XZ slice
        ax2 = fig.add_subplot(222)
        mid_y = self.solver.ny // 2
        im2 = ax2.imshow(vel_mag[:, mid_y, :], cmap='inferno', origin='lower')
        ax2.set_title('Y-midplane velocity')
        plt.colorbar(im2, ax=ax2)
        
        # Energy spectrum
        ax3 = fig.add_subplot(223)
        try:
            k, E = self.solver.compute_energy_spectrum()
            ax3.loglog(k[1:], E[1:], 'b-', lw=2, label='E(k)')
            # -5/3 reference line
            k_ref = k[1:len(k)//4]
            E_ref = E[1] * (k_ref / k_ref[0])**(-5/3) * 0.5
            ax3.loglog(k_ref, E_ref, 'r--', label='k^(-5/3)', alpha=0.7)
            ax3.set_xlabel('Wavenumber k')
            ax3.set_ylabel('E(k)')
            ax3.set_title('Energy Spectrum')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except Exception:
            ax3.text(0.5, 0.5, 'Spectrum unavailable', ha='center', va='center',
                    transform=ax3.transAxes)
        
        # Energy evolution
        ax4 = fig.add_subplot(224)
        if self.solver.history['time']:
            ax4.plot(self.solver.history['time'], self.solver.history['kinetic_energy'],
                    'r-', lw=2, label='KE')
            ax4.plot(self.solver.history['time'], self.solver.history['enstrophy'],
                    'b-', lw=2, label='Enstrophy')
            ax4.set_xlabel('Time')
            ax4.set_title('Energy Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Snapshot saved to {filename}")
