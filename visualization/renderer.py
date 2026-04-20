"""
=============================================================================
Flow Renderer Utilities
Colormaps, arrow drawing, field conversion for visualization.
=============================================================================
"""

import numpy as np
from typing import Tuple, Optional


class FlowRenderer:
    """Shared rendering utilities for 2D and 3D visualization."""
    
    # Scientific colormaps as RGB arrays
    COLORMAPS = {
        'inferno': None,
        'viridis': None,
        'plasma': None,
        'coolwarm': None,
        'turbo': None,
    }
    
    @staticmethod
    def scalar_to_rgb(
        field: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "inferno",
        symmetric: bool = False,
    ) -> np.ndarray:
        """
        Convert scalar field to RGB image.
        
        Args:
            field: (H, W) scalar field
            vmin, vmax: Data range (auto if None)
            cmap: Colormap name
            symmetric: If True, center colormap at zero
        
        Returns:
            (H, W, 3) uint8 RGB array
        """
        if vmin is None:
            vmin = np.min(field)
        if vmax is None:
            vmax = np.max(field)
        
        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        
        # Normalize to [0, 1]
        if vmax - vmin < 1e-12:
            normalized = np.zeros_like(field)
        else:
            normalized = np.clip((field - vmin) / (vmax - vmin), 0, 1)
        
        # Apply colormap
        rgb = FlowRenderer._apply_colormap(normalized, cmap)
        return (rgb * 255).astype(np.uint8)
    
    @staticmethod
    def _apply_colormap(normalized: np.ndarray, cmap: str) -> np.ndarray:
        """Apply a colormap to normalized [0,1] data."""
        try:
            import matplotlib.cm as cm
            mapper = cm.get_cmap(cmap)
            rgba = mapper(normalized)
            return rgba[..., :3]
        except ImportError:
            # Fallback: manual inferno-like colormap
            return FlowRenderer._manual_colormap(normalized, cmap)
    
    @staticmethod
    def _manual_colormap(normalized: np.ndarray, cmap: str) -> np.ndarray:
        """Manual colormap implementation (no matplotlib dependency)."""
        h, w = normalized.shape
        rgb = np.zeros((h, w, 3))
        
        if cmap in ['inferno', 'plasma']:
            # Dark-to-bright warm colormap
            rgb[..., 0] = np.clip(normalized * 2, 0, 1)  # R
            rgb[..., 1] = np.clip(normalized * 1.5 - 0.3, 0, 1)  # G
            rgb[..., 2] = np.clip(1 - normalized * 2, 0, 0.5) + normalized * 0.3  # B
        elif cmap == 'coolwarm':
            # Diverging blue-white-red
            rgb[..., 0] = np.clip(normalized * 2, 0, 1)
            rgb[..., 1] = 1 - np.abs(normalized - 0.5) * 2
            rgb[..., 2] = np.clip(2 - normalized * 2, 0, 1)
        else:
            # Default grayscale-ish
            rgb[..., 0] = normalized * 0.27 + 0.0
            rgb[..., 1] = normalized * 0.67 + 0.1
            rgb[..., 2] = normalized * 0.33 + 0.3
        
        return np.clip(rgb, 0, 1)
    
    @staticmethod
    def velocity_to_rgb(
        u: np.ndarray, v: np.ndarray,
        max_speed: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert velocity field to HSV-based direction+magnitude coloring.
        
        Hue = flow direction (0-360°)
        Saturation = 1.0
        Value = flow speed (normalized)
        """
        speed = np.sqrt(u**2 + v**2)
        direction = np.arctan2(v, u)  # [-π, π]
        
        if max_speed is None:
            max_speed = np.max(speed) + 1e-10
        
        # Normalize
        hue = (direction + np.pi) / (2 * np.pi)  # [0, 1]
        value = np.clip(speed / max_speed, 0, 1)
        saturation = np.ones_like(hue) * 0.9
        
        # HSV to RGB
        return FlowRenderer._hsv_to_rgb(hue, saturation, value)
    
    @staticmethod
    def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Convert HSV arrays to RGB uint8 array."""
        h6 = h * 6.0
        i = np.floor(h6).astype(int) % 6
        f = h6 - np.floor(h6)
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        rgb = np.zeros((*h.shape, 3))
        
        mask = i == 0; rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)
        mask = i == 1; rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)
        mask = i == 2; rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)
        mask = i == 3; rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)
        mask = i == 4; rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)
        mask = i == 5; rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)
        
        return (rgb * 255).astype(np.uint8)
    
    @staticmethod
    def draw_velocity_arrows(
        surface,  # pygame.Surface
        u: np.ndarray, v: np.ndarray,
        screen_w: int, screen_h: int,
        density: int = 8,
        color: Tuple[int, int, int] = (255, 255, 255),
        scale: float = 20.0,
        alpha: int = 180,
    ):
        """Draw velocity arrows on a pygame surface."""
        import pygame
        
        ny, nx = u.shape
        step_x = max(1, nx // density)
        step_y = max(1, ny // density)
        
        sx = screen_w / nx
        sy = screen_h / ny
        
        for j in range(0, ny, step_y):
            for i in range(0, nx, step_x):
                x0 = int(i * sx + sx/2)
                y0 = int(j * sy + sy/2)
                
                vel_u = u[j, i]
                vel_v = v[j, i]
                speed = np.sqrt(vel_u**2 + vel_v**2)
                
                if speed < 1e-6:
                    continue
                
                # Arrow endpoint
                x1 = int(x0 + vel_u * scale)
                y1 = int(y0 + vel_v * scale)
                
                # Draw line
                pygame.draw.line(surface, color, (x0, y0), (x1, y1), 1)
                
                # Arrowhead
                if speed > 0.01:
                    angle = np.arctan2(vel_v, vel_u)
                    head_len = min(5, speed * scale * 0.3)
                    for da in [2.5, -2.5]:
                        hx = x1 - int(head_len * np.cos(angle + da))
                        hy = y1 - int(head_len * np.sin(angle + da))
                        pygame.draw.line(surface, color, (x1, y1), (hx, hy), 1)
    
    @staticmethod
    def draw_streamlines(
        u: np.ndarray, v: np.ndarray,
        nx: int, ny: int,
        n_lines: int = 20,
        n_steps: int = 100,
        dt: float = 0.5,
    ) -> list:
        """
        Compute streamlines using Euler integration.
        
        Returns list of (x_coords, y_coords) arrays.
        """
        streamlines = []
        
        # Random seed points
        for _ in range(n_lines):
            x = np.random.uniform(0, nx-1)
            y = np.random.uniform(0, ny-1)
            
            path_x = [x]
            path_y = [y]
            
            for step in range(n_steps):
                ix = int(np.clip(x, 0, nx-1))
                iy = int(np.clip(y, 0, ny-1))
                
                ux = u[iy, ix]
                vy = v[iy, ix]
                
                speed = np.sqrt(ux**2 + vy**2)
                if speed < 1e-6:
                    break
                
                x += ux * dt
                y += vy * dt
                
                if x < 0 or x >= nx or y < 0 or y >= ny:
                    break
                
                path_x.append(x)
                path_y.append(y)
            
            if len(path_x) > 2:
                streamlines.append((np.array(path_x), np.array(path_y)))
        
        return streamlines
