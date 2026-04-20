"""
=============================================================================
Pressure Poisson Solver
Multiple methods: FFT (fast), Jacobi, SOR, Conjugate Gradient
Solves: ∇²p = f (Pressure Poisson Equation)
=============================================================================
"""

import numpy as np
from scipy import fft as scipy_fft
from typing import Optional, Tuple


class PressureSolver:
    """
    Multi-method pressure Poisson solver.
    
    Solves ∇²p = rhs with appropriate boundary conditions.
    Available methods:
        - FFT (fastest, periodic BCs only)
        - Jacobi (simple iterative)
        - SOR (accelerated iterative)
        - Conjugate Gradient (general purpose)
        - Multigrid V-cycle (optimal complexity)
    """
    
    def __init__(self, nx: int, ny: int, dx: float, dy: float, method: str = "fft"):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.method = method
        
        # Pre-compute FFT eigenvalues for spectral solver
        if method == "fft":
            self._setup_fft()
        
        # Multigrid levels
        if method == "multigrid":
            self._setup_multigrid()
        
        # Statistics
        self.iterations = 0
        self.residual = 0.0
    
    def _setup_fft(self):
        """Pre-compute modified wavenumbers for FFT-based solver."""
        # Modified wavenumbers for 2nd-order central differences
        kx = np.arange(self.nx)
        ky = np.arange(self.ny)
        
        # Eigenvalues of the discrete Laplacian
        self.eigenvalues = (
            2 * (np.cos(2 * np.pi * kx / self.nx) - 1) / self.dx**2
            + 2 * (np.cos(2 * np.pi * ky[:, np.newaxis] / self.ny) - 1) / self.dy**2
        )
        # Avoid division by zero at (0,0) mode
        self.eigenvalues[0, 0] = 1.0
    
    def _setup_multigrid(self):
        """Setup multigrid hierarchy."""
        self.mg_levels = []
        nx, ny = self.nx, self.ny
        while nx >= 4 and ny >= 4:
            self.mg_levels.append((nx, ny))
            nx //= 2
            ny //= 2
    
    def solve(
        self, 
        rhs: np.ndarray, 
        p_init: Optional[np.ndarray] = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        omega: float = 1.7,
        bc_type: str = "neumann"
    ) -> np.ndarray:
        """
        Solve ∇²p = rhs.
        
        Args:
            rhs: Right-hand side array (ny, nx)
            p_init: Initial guess for iterative methods
            max_iter: Maximum iterations for iterative methods
            tol: Convergence tolerance
            omega: Over-relaxation parameter for SOR
            bc_type: "periodic", "neumann", "dirichlet"
        
        Returns:
            p: Pressure field (ny, nx)
        """
        if self.method == "fft":
            return self._solve_fft(rhs)
        elif self.method == "jacobi":
            return self._solve_jacobi(rhs, p_init, max_iter, tol, bc_type)
        elif self.method == "sor":
            return self._solve_sor(rhs, p_init, max_iter, tol, omega, bc_type)
        elif self.method == "cg":
            return self._solve_conjugate_gradient(rhs, p_init, max_iter, tol, bc_type)
        elif self.method == "multigrid":
            return self._solve_multigrid(rhs, max_iter, tol, bc_type)
        else:
            raise ValueError(f"Unknown pressure solver method: {self.method}")
    
    def _solve_fft(self, rhs: np.ndarray) -> np.ndarray:
        """
        FFT-based solver (spectral method).
        
        Fastest method, O(N log N) complexity.
        Requires periodic boundary conditions.
        """
        # Forward FFT
        rhs_hat = np.fft.fft2(rhs)
        
        # Solve in spectral space: p_hat = rhs_hat / eigenvalues
        p_hat = rhs_hat / self.eigenvalues
        
        # Zero mean pressure
        p_hat[0, 0] = 0.0
        
        # Inverse FFT
        p = np.real(np.fft.ifft2(p_hat))
        
        self.iterations = 1
        self.residual = 0.0
        
        return p
    
    def _solve_jacobi(
        self, rhs: np.ndarray, p_init: Optional[np.ndarray],
        max_iter: int, tol: float, bc_type: str
    ) -> np.ndarray:
        """
        Jacobi iterative solver.
        
        Simple but slow convergence. Good for parallelization.
        p_{i,j}^{n+1} = [rhs_{i,j}*h² - p_{i+1,j} - p_{i-1,j} - p_{i,j+1} - p_{i,j-1}] / (-4)
        """
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        coeff = 2.0 * (1.0/dx2 + 1.0/dy2)
        
        p = p_init.copy() if p_init is not None else np.zeros_like(rhs)
        
        for iteration in range(max_iter):
            p_old = p.copy()
            
            # Interior points
            p[1:-1, 1:-1] = (
                (p_old[1:-1, 2:] + p_old[1:-1, :-2]) / dx2 +
                (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / dy2 -
                rhs[1:-1, 1:-1]
            ) / coeff
            
            # Apply boundary conditions
            self._apply_bc(p, bc_type)
            
            # Check convergence
            residual = np.max(np.abs(p - p_old))
            if residual < tol:
                self.iterations = iteration + 1
                self.residual = residual
                return p
        
        self.iterations = max_iter
        self.residual = residual
        return p
    
    def _solve_sor(
        self, rhs: np.ndarray, p_init: Optional[np.ndarray],
        max_iter: int, tol: float, omega: float, bc_type: str
    ) -> np.ndarray:
        """
        Successive Over-Relaxation (SOR) solver.
        
        Accelerated Gauss-Seidel with over-relaxation parameter ω.
        Optimal ω ≈ 2 / (1 + sin(π/N)) for Poisson equation.
        """
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        coeff = 2.0 * (1.0/dx2 + 1.0/dy2)
        
        p = p_init.copy() if p_init is not None else np.zeros_like(rhs)
        
        for iteration in range(max_iter):
            max_change = 0.0
            
            # Red-Black ordering for better convergence
            for color in [0, 1]:
                for j in range(1, self.ny - 1):
                    for i in range(1 + (j + color) % 2, self.nx - 1, 2):
                        p_gs = (
                            (p[j, i+1] + p[j, i-1]) / dx2 +
                            (p[j+1, i] + p[j-1, i]) / dy2 -
                            rhs[j, i]
                        ) / coeff
                        
                        change = omega * (p_gs - p[j, i])
                        p[j, i] += change
                        max_change = max(max_change, abs(change))
            
            # Apply boundary conditions
            self._apply_bc(p, bc_type)
            
            if max_change < tol:
                self.iterations = iteration + 1
                self.residual = max_change
                return p
        
        self.iterations = max_iter
        self.residual = max_change
        return p
    
    def _solve_conjugate_gradient(
        self, rhs: np.ndarray, p_init: Optional[np.ndarray],
        max_iter: int, tol: float, bc_type: str
    ) -> np.ndarray:
        """
        Conjugate Gradient solver.
        
        Optimal for symmetric positive definite systems.
        Convergence in at most N iterations (in exact arithmetic).
        """
        def apply_laplacian(x: np.ndarray) -> np.ndarray:
            """Apply discrete Laplacian operator."""
            Ax = np.zeros_like(x)
            Ax[1:-1, 1:-1] = (
                (x[1:-1, 2:] + x[1:-1, :-2] - 2*x[1:-1, 1:-1]) / self.dx**2 +
                (x[2:, 1:-1] + x[:-2, 1:-1] - 2*x[1:-1, 1:-1]) / self.dy**2
            )
            return Ax
        
        p = p_init.copy() if p_init is not None else np.zeros_like(rhs)
        
        r = rhs - apply_laplacian(p)
        d = r.copy()
        r_norm_sq = np.sum(r * r)
        
        for iteration in range(max_iter):
            Ad = apply_laplacian(d)
            dAd = np.sum(d * Ad)
            
            if abs(dAd) < 1e-30:
                break
            
            alpha = r_norm_sq / dAd
            p += alpha * d
            r -= alpha * Ad
            
            self._apply_bc(p, bc_type)
            
            r_norm_sq_new = np.sum(r * r)
            residual = np.sqrt(r_norm_sq_new / max(r.size, 1))
            
            if residual < tol:
                self.iterations = iteration + 1
                self.residual = residual
                return p
            
            beta = r_norm_sq_new / max(r_norm_sq, 1e-30)
            d = r + beta * d
            r_norm_sq = r_norm_sq_new
        
        self.iterations = max_iter
        self.residual = residual
        return p
    
    def _solve_multigrid(
        self, rhs: np.ndarray, max_iter: int, tol: float, bc_type: str
    ) -> np.ndarray:
        """
        Geometric Multigrid V-cycle solver.
        
        Optimal O(N) complexity for elliptic PDEs.
        Uses restriction (fine→coarse) and prolongation (coarse→fine).
        """
        def smooth(p, rhs, nx, ny, dx, dy, n_smooth=3):
            """Weighted Jacobi smoother."""
            dx2 = dx**2
            dy2 = dy**2
            coeff = 2.0 * (1.0/dx2 + 1.0/dy2)
            omega = 2.0/3.0  # Optimal Jacobi weight for multigrid
            
            for _ in range(n_smooth):
                p_new = np.zeros_like(p)
                p_new[1:-1, 1:-1] = (
                    (p[1:-1, 2:] + p[1:-1, :-2]) / dx2 +
                    (p[2:, 1:-1] + p[:-2, 1:-1]) / dy2 -
                    rhs[1:-1, 1:-1]
                ) / coeff
                p = (1 - omega) * p + omega * p_new
                self._apply_bc(p, bc_type)
            return p
        
        def restrict(fine):
            """Full weighting restriction operator."""
            coarse = fine[::2, ::2].copy()
            return coarse
        
        def prolongate(coarse, fine_shape):
            """Bilinear interpolation prolongation."""
            from scipy.ndimage import zoom
            factors = (fine_shape[0] / coarse.shape[0], fine_shape[1] / coarse.shape[1])
            return zoom(coarse, factors, order=1)
        
        def residual(p, rhs, dx, dy):
            """Compute residual r = rhs - ∇²p."""
            Lp = np.zeros_like(p)
            Lp[1:-1, 1:-1] = (
                (p[1:-1, 2:] + p[1:-1, :-2] - 2*p[1:-1, 1:-1]) / dx**2 +
                (p[2:, 1:-1] + p[:-2, 1:-1] - 2*p[1:-1, 1:-1]) / dy**2
            )
            return rhs - Lp
        
        def v_cycle(p, rhs, level=0):
            """Recursive V-cycle."""
            if level >= len(self.mg_levels) - 1:
                # Coarsest level: solve directly
                return smooth(p, rhs, *self.mg_levels[level], 
                            self.dx * 2**level, self.dy * 2**level, n_smooth=50)
            
            nx, ny = self.mg_levels[level]
            dx = self.dx * 2**level
            dy = self.dy * 2**level
            
            # Pre-smooth
            p = smooth(p, rhs, nx, ny, dx, dy, n_smooth=3)
            
            # Compute residual
            r = residual(p, rhs, dx, dy)
            
            # Restrict residual to coarse grid
            r_coarse = restrict(r)
            
            # Solve on coarse grid
            e_coarse = np.zeros_like(r_coarse)
            e_coarse = v_cycle(e_coarse, r_coarse, level + 1)
            
            # Prolongate error and correct
            e_fine = prolongate(e_coarse, p.shape)
            p += e_fine
            
            # Post-smooth
            p = smooth(p, rhs, nx, ny, dx, dy, n_smooth=3)
            
            return p
        
        p = np.zeros_like(rhs)
        
        for iteration in range(max_iter):
            p = v_cycle(p, rhs)
            
            r = residual(p, rhs, self.dx, self.dy)
            res_norm = np.max(np.abs(r))
            
            if res_norm < tol:
                self.iterations = iteration + 1
                self.residual = res_norm
                return p
        
        self.iterations = max_iter
        self.residual = res_norm
        return p
    
    def _apply_bc(self, p: np.ndarray, bc_type: str):
        """Apply boundary conditions to pressure field."""
        if bc_type == "neumann":
            # Zero normal gradient (∂p/∂n = 0)
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
        elif bc_type == "dirichlet":
            # Zero pressure at boundaries
            p[0, :] = 0
            p[-1, :] = 0
            p[:, 0] = 0
            p[:, -1] = 0
        elif bc_type == "periodic":
            p[0, :] = p[-2, :]
            p[-1, :] = p[1, :]
            p[:, 0] = p[:, -2]
            p[:, -1] = p[:, 1]
        
        # Ensure zero mean for uniqueness
        p -= np.mean(p)


class VectorizedPressureSolver:
    """
    Optimized pressure solver using fully vectorized NumPy operations.
    For maximum performance on CPU.
    """
    
    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Pre-compute spectral eigenvalues
        i = np.arange(nx)
        j = np.arange(ny)[:, None]
        
        self.eigenvalues = (
            2 * (np.cos(2 * np.pi * i / nx) - 1) / dx**2 +
            2 * (np.cos(2 * np.pi * j / ny) - 1) / dy**2
        )
        self.eigenvalues[0, 0] = 1.0  # Avoid /0
    
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Ultra-fast FFT solve for periodic domains."""
        p_hat = np.fft.fft2(rhs) / self.eigenvalues
        p_hat[0, 0] = 0.0
        return np.real(np.fft.ifft2(p_hat))
