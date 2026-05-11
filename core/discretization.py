"""Discretization Schemes"""

import numpy as np
from typing import Optional


class AdvectionSchemes:
    """
    Advection (convective) discretization schemes.
    
    Computes: (u·∇)φ for a scalar or vector field φ
    """
    
    @staticmethod
    def central_2nd(phi: np.ndarray, dx: float, axis: int = 1, periodic: bool = False) -> np.ndarray:
        """
        2nd-order central difference: ∂φ/∂x ≈ (φ_{i+1} - φ_{i-1}) / (2Δx)
        
        Properties: 2nd-order accurate, non-dissipative, can cause oscillations.
        """
        if periodic:
            return (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2 * dx)
        result = np.zeros_like(phi)
        if axis == 1:  # x-direction
            result[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
        elif axis == 0:  # y-direction
            result[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
        return result
    
    @staticmethod
    def central_4th(phi: np.ndarray, dx: float, axis: int = 1) -> np.ndarray:
        """
        4th-order central difference: 
        ∂φ/∂x ≈ (-φ_{i+2} + 8φ_{i+1} - 8φ_{i-1} + φ_{i-2}) / (12Δx)
        """
        result = np.zeros_like(phi)
        if axis == 1:
            result[:, 2:-2] = (-phi[:, 4:] + 8*phi[:, 3:-1] - 8*phi[:, 1:-3] + phi[:, :-4]) / (12 * dx)
        elif axis == 0:
            result[2:-2, :] = (-phi[4:, :] + 8*phi[3:-1, :] - 8*phi[1:-3, :] + phi[:-4, :]) / (12 * dx)
        return result
    
    @staticmethod
    def upwind_1st(phi: np.ndarray, vel: np.ndarray, dx: float, axis: int = 1) -> np.ndarray:
        """
        1st-order upwind scheme.
        
        Uses backward difference where vel > 0, forward where vel < 0.
        Properties: 1st-order, highly dissipative but stable.
        """
        result = np.zeros_like(phi)
        
        if axis == 1:
            # Forward difference
            fwd = np.zeros_like(phi)
            fwd[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx
            # Backward difference
            bwd = np.zeros_like(phi)
            bwd[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx
            # Choose based on velocity sign
            result = np.where(vel > 0, bwd, fwd)
        elif axis == 0:
            fwd = np.zeros_like(phi)
            fwd[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dx
            bwd = np.zeros_like(phi)
            bwd[1:, :] = (phi[1:, :] - phi[:-1, :]) / dx
            result = np.where(vel > 0, bwd, fwd)
        
        return result
    
    @staticmethod
    def upwind_3rd(phi: np.ndarray, vel: np.ndarray, dx: float, axis: int = 1) -> np.ndarray:
        """
        3rd-order upwind-biased scheme (QUICK-like).
        
        ∂φ/∂x ≈ (-2φ_{i-1} + 3φ_i - 6φ_{i+1} + φ_{i+2}) / (6Δx)  for vel > 0
        """
        result = np.zeros_like(phi)
        
        if axis == 1:
            # Interior points only
            pos = (2*phi[:, :-3] + 3*phi[:, 1:-2] - 6*phi[:, 2:-1] + phi[:, 3:]) / (6 * dx)
            neg = (-phi[:, :-3] + 6*phi[:, 1:-2] - 3*phi[:, 2:-1] - 2*phi[:, 3:]) / (6 * dx)
            vel_int = vel[:, 2:-1] if vel.shape == phi.shape else vel[:, 2:-1]
            result[:, 2:-1] = np.where(vel_int > 0, -pos, -neg)
        elif axis == 0:
            pos = (2*phi[:-3, :] + 3*phi[1:-2, :] - 6*phi[2:-1, :] + phi[3:, :]) / (6 * dx)
            neg = (-phi[:-3, :] + 6*phi[1:-2, :] - 3*phi[2:-1, :] - 2*phi[3:, :]) / (6 * dx)
            vel_int = vel[2:-1, :] if vel.shape == phi.shape else vel[2:-1, :]
            result[2:-1, :] = np.where(vel_int > 0, -pos, -neg)
        
        return result
    
    @staticmethod
    def weno5(phi: np.ndarray, vel: np.ndarray, dx: float, axis: int = 1) -> np.ndarray:
        """
        5th-order WENO (Weighted Essentially Non-Oscillatory) scheme.
        
        Used for capturing sharp gradients and shocks without spurious oscillations.
        Essential for high Reynolds number flows.
        """
        epsilon = 1e-6  # prevent division by zero
        
        def _weno5_reconstruct(v: np.ndarray, direction: int) -> np.ndarray:
            """Reconstruct interface values using WENO-5."""
            # Optimal weights
            d = np.array([1/10, 6/10, 3/10])
            
            n = len(v) - 4
            if n < 1:
                return np.zeros(max(n, 0))
            
            # Candidate stencils
            q0 = (2*v[:n] - 7*v[1:n+1] + 11*v[2:n+2]) / 6
            q1 = (-v[1:n+1] + 5*v[2:n+2] + 2*v[3:n+3]) / 6
            q2 = (2*v[2:n+2] + 5*v[3:n+3] - v[4:n+4]) / 6
            
            # Smoothness indicators
            beta0 = (13/12)*(v[:n] - 2*v[1:n+1] + v[2:n+2])**2 + \
                     (1/4)*(v[:n] - 4*v[1:n+1] + 3*v[2:n+2])**2
            beta1 = (13/12)*(v[1:n+1] - 2*v[2:n+2] + v[3:n+3])**2 + \
                     (1/4)*(v[1:n+1] - v[3:n+3])**2
            beta2 = (13/12)*(v[2:n+2] - 2*v[3:n+3] + v[4:n+4])**2 + \
                     (1/4)*(3*v[2:n+2] - 4*v[3:n+3] + v[4:n+4])**2
            
            # Non-linear weights
            alpha0 = d[0] / (epsilon + beta0)**2
            alpha1 = d[1] / (epsilon + beta1)**2
            alpha2 = d[2] / (epsilon + beta2)**2
            
            alpha_sum = alpha0 + alpha1 + alpha2
            w0 = alpha0 / alpha_sum
            w1 = alpha1 / alpha_sum
            w2 = alpha2 / alpha_sum
            
            return w0*q0 + w1*q1 + w2*q2
        
        result = np.zeros_like(phi)
        
        if axis == 1:
            for j in range(phi.shape[0]):
                row = phi[j, :]
                vel_row = vel[j, :] if vel.ndim > 1 else vel
                
                if len(row) >= 5:
                    # Positive velocity: left-biased
                    fplus = _weno5_reconstruct(row, 1)
                    # Negative velocity: right-biased
                    fminus = _weno5_reconstruct(row[::-1], -1)[::-1]
                    
                    n = min(len(fplus), len(fminus), phi.shape[1] - 4)
                    if n > 0:
                        vel_int = vel_row[2:2+n]
                        flux = np.where(vel_int > 0, fplus[:n], fminus[:n])
                        result[j, 2:2+n] = -(flux - np.roll(flux, 1)) / dx
        
        elif axis == 0:
            for i in range(phi.shape[1]):
                col = phi[:, i]
                vel_col = vel[:, i] if vel.ndim > 1 else vel
                
                if len(col) >= 5:
                    fplus = _weno5_reconstruct(col, 1)
                    fminus = _weno5_reconstruct(col[::-1], -1)[::-1]
                    
                    n = min(len(fplus), len(fminus), phi.shape[0] - 4)
                    if n > 0:
                        vel_int = vel_col[2:2+n]
                        flux = np.where(vel_int > 0, fplus[:n], fminus[:n])
                        result[2:2+n, i] = -(flux - np.roll(flux, 1)) / dx
        
        return result
    
    @staticmethod
    def advect(
        phi: np.ndarray,
        u: np.ndarray, v: np.ndarray,
        dx: float, dy: float,
        scheme: str = "central",
        **kwargs
    ) -> np.ndarray:
        """
        Compute advection term: (u·∇)φ
        
        Args:
            phi: scalar field to advect
            u, v: velocity components
            dx, dy: grid spacing
            scheme: "upwind", "central", "central4", "weno5"
        
        Returns:
            Advection term: u * ∂φ/∂x + v * ∂φ/∂y
        """
        periodic = kwargs.get('periodic', False)
        if scheme == "upwind":
            dpdx = AdvectionSchemes.upwind_1st(phi, u, dx, axis=1)
            dpdy = AdvectionSchemes.upwind_1st(phi, v, dy, axis=0)
        elif scheme == "central":
            dpdx = AdvectionSchemes.central_2nd(phi, dx, axis=1, periodic=periodic)
            dpdy = AdvectionSchemes.central_2nd(phi, dy, axis=0, periodic=periodic)
        elif scheme == "central4":
            dpdx = AdvectionSchemes.central_4th(phi, dx, axis=1)
            dpdy = AdvectionSchemes.central_4th(phi, dy, axis=0)
        elif scheme == "weno5":
            dpdx = AdvectionSchemes.weno5(phi, u, dx, axis=1)
            dpdy = AdvectionSchemes.weno5(phi, v, dy, axis=0)
        elif scheme == "upwind3":
            dpdx = AdvectionSchemes.upwind_3rd(phi, u, dx, axis=1)
            dpdy = AdvectionSchemes.upwind_3rd(phi, v, dy, axis=0)
        else:
            raise ValueError(f"Unknown advection scheme: {scheme}")
        
        return u * dpdx + v * dpdy


class DiffusionSchemes:
    """
    Diffusion discretization schemes.
    
    Computes: ν∇²φ (Laplacian operator)
    """
    
    @staticmethod
    def laplacian_2nd(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        2nd-order 5-point Laplacian stencil:
        
        ∇²φ ≈ (φ_{i+1,j} + φ_{i-1,j} - 2φ_{i,j}) / Δx² 
             + (φ_{i,j+1} + φ_{i,j-1} - 2φ_{i,j}) / Δy²
        """
        lap = np.zeros_like(phi)
        
        # x-direction (columns)
        lap[:, 1:-1] += (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / dx**2
        
        # y-direction (rows)  
        lap[1:-1, :] += (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / dy**2
        
        return lap
    
    @staticmethod
    def laplacian_4th(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        4th-order compact Laplacian stencil:
        
        ∇²φ ≈ (-φ_{i+2} + 16φ_{i+1} - 30φ_i + 16φ_{i-1} - φ_{i-2}) / (12Δx²)
        """
        lap = np.zeros_like(phi)
        
        # x-direction
        lap[:, 2:-2] += (-phi[:, 4:] + 16*phi[:, 3:-1] - 30*phi[:, 2:-2] 
                         + 16*phi[:, 1:-3] - phi[:, :-4]) / (12 * dx**2)
        
        # y-direction
        lap[2:-2, :] += (-phi[4:, :] + 16*phi[3:-1, :] - 30*phi[2:-2, :] 
                         + 16*phi[1:-3, :] - phi[:-4, :]) / (12 * dy**2)
        
        return lap
    
    @staticmethod  
    def laplacian_periodic(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Laplacian with periodic boundary conditions using np.roll."""
        lap_x = (np.roll(phi, -1, axis=1) - 2*phi + np.roll(phi, 1, axis=1)) / dx**2
        lap_y = (np.roll(phi, -1, axis=0) - 2*phi + np.roll(phi, 1, axis=0)) / dy**2
        return lap_x + lap_y
    
    @staticmethod
    def biharmonic(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        Biharmonic operator ∇⁴φ = ∇²(∇²φ) for hyperviscosity.
        Used in LES and spectral methods.
        """
        lap1 = DiffusionSchemes.laplacian_2nd(phi, dx, dy)
        return DiffusionSchemes.laplacian_2nd(lap1, dx, dy)


class GradientOperators:
    """Gradient and divergence operators."""
    
    @staticmethod
    def gradient(phi: np.ndarray, dx: float, dy: float, periodic: bool = False) -> tuple:
        """
        Compute ∇φ = (∂φ/∂x, ∂φ/∂y) using central differences.
        """
        if periodic:
            dpdx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
            dpdy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dy)
            return dpdx, dpdy
        
        dpdx = np.zeros_like(phi)
        dpdy = np.zeros_like(phi)
        
        dpdx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
        dpdy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dy)
        
        return dpdx, dpdy
    
    @staticmethod
    def divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float, periodic: bool = False) -> np.ndarray:
        """Compute ∇·(u, v) = ∂u/∂x + ∂v/∂y."""
        if periodic:
            dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
            dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
            return dudx + dvdy
        
        dudx = np.zeros_like(u)
        dvdy = np.zeros_like(v)
        
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
        
        return dudx + dvdy
    
    @staticmethod
    def curl_2d(u: np.ndarray, v: np.ndarray, dx: float, dy: float, periodic: bool = False) -> np.ndarray:
        """Compute 2D curl (vorticity): ω = ∂v/∂x - ∂u/∂y."""
        if periodic:
            dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
            dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
            return dvdx - dudy
        
        dvdx = np.zeros_like(v)
        dudy = np.zeros_like(u)
        
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        
        return dvdx - dudy
