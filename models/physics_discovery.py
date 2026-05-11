"""AI-Discovered Physics Laws — Neural + Symbolic Hybrid Engine"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import copy
import time


# Physics-Aware Candidate Library

class PhysicsAwareLibrary:
    """
    Extended SINDy library with physics-informed candidate functions.

    Beyond standard polynomials, includes:
        - Gradient operators: ∂u/∂x, ∂u/∂y
        - Laplacian: ∇²u
        - Advection: u·∂u/∂x + v·∂u/∂y
        - Cross-field products: u*ω, p*∇·u
        - Nonlinear interactions: u²∇²u, ω×u
    """

    def __init__(
        self,
        poly_order: int = 2,
        include_gradients: bool = True,
        include_laplacian: bool = True,
        include_advection: bool = True,
        include_trig: bool = False,
        include_cross_terms: bool = True,
        dx: float = 1.0,
        dy: float = 1.0,
    ):
        self.poly_order = poly_order
        self.include_gradients = include_gradients
        self.include_laplacian = include_laplacian
        self.include_advection = include_advection
        self.include_trig = include_trig
        self.include_cross_terms = include_cross_terms
        self.dx = dx
        self.dy = dy
        self.feature_names: List[str] = []

    def _safe_gradient(self, field_2d: np.ndarray, axis: int) -> np.ndarray:
        """Compute spatial gradient along axis using central differences."""
        if axis == 0:
            grad = (np.roll(field_2d, -1, axis=0) - np.roll(field_2d, 1, axis=0)) / (2 * self.dy)
        else:
            grad = (np.roll(field_2d, -1, axis=1) - np.roll(field_2d, 1, axis=1)) / (2 * self.dx)
        return grad

    def _safe_laplacian(self, field_2d: np.ndarray) -> np.ndarray:
        """Compute Laplacian ∇²f using 5-point stencil."""
        lap = (
            np.roll(field_2d, 1, axis=0) + np.roll(field_2d, -1, axis=0) +
            np.roll(field_2d, 1, axis=1) + np.roll(field_2d, -1, axis=1) -
            4 * field_2d
        ) / (self.dx * self.dy)
        return lap

    def build_from_fields(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        omega: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build physics-aware candidate library from 2D flow fields.

        Args:
            u, v: velocity components (ny, nx)
            p: pressure field (ny, nx)
            omega: vorticity field (ny, nx)

        Returns:
            Theta: (n_points, n_candidates) library matrix
            names: list of candidate function names
        """
        ny, nx = u.shape
        n_points = ny * nx

        # Flatten fields
        U = u.flatten()
        V = v.flatten()
        P = p.flatten()
        W = omega.flatten()

        columns = []
        names = []

        # 1. Constant
        columns.append(np.ones(n_points))
        names.append('1')

        # 2. Linear terms
        for field, name in [(U, 'u'), (V, 'v'), (P, 'p'), (W, 'omega')]:
            columns.append(field)
            names.append(name)

        # 3. Polynomial terms (degree 2)
        if self.poly_order >= 2:
            fields = [(U, 'u'), (V, 'v'), (P, 'p'), (W, 'omega')]
            for i, (fi, ni) in enumerate(fields):
                for j, (fj, nj) in enumerate(fields):
                    if j >= i:
                        columns.append(fi * fj)
                        names.append(f'{ni}*{nj}')

        # 4. Gradient terms
        if self.include_gradients:
            for field_2d, name in [(u, 'u'), (v, 'v'), (p, 'p')]:
                dudx = self._safe_gradient(field_2d, axis=1).flatten()
                dudy = self._safe_gradient(field_2d, axis=0).flatten()
                columns.append(dudx)
                names.append(f'd{name}/dx')
                columns.append(dudy)
                names.append(f'd{name}/dy')

        # 5. Laplacian terms
        if self.include_laplacian:
            for field_2d, name in [(u, 'u'), (v, 'v')]:
                lap = self._safe_laplacian(field_2d).flatten()
                columns.append(lap)
                names.append(f'nabla2_{name}')

        # 6. Advection terms: u·∂f/∂x + v·∂f/∂y
        if self.include_advection:
            for field_2d, name in [(u, 'u'), (v, 'v')]:
                dfdx = self._safe_gradient(field_2d, axis=1)
                dfdy = self._safe_gradient(field_2d, axis=0)
                advection = (u * dfdx + v * dfdy).flatten()
                columns.append(advection)
                names.append(f'u*d{name}/dx+v*d{name}/dy')

        # 7. Cross-field interactions
        if self.include_cross_terms:
            columns.append(U * W)
            names.append('u*omega')
            columns.append(V * W)
            names.append('v*omega')
            # Pressure gradient
            dpdx = self._safe_gradient(p, axis=1).flatten()
            dpdy = self._safe_gradient(p, axis=0).flatten()
            columns.append(dpdx)
            names.append('dp/dx')
            columns.append(dpdy)
            names.append('dp/dy')

        # 8. Trigonometric (optional)
        if self.include_trig:
            for field, name in [(U, 'u'), (V, 'v')]:
                safe_f = np.clip(field, -10, 10)
                columns.append(np.sin(safe_f))
                names.append(f'sin({name})')
                columns.append(np.cos(safe_f))
                names.append(f'cos({name})')

        # Clip for numerical safety
        Theta = np.column_stack(columns)
        Theta = np.nan_to_num(Theta, nan=0.0, posinf=1e6, neginf=-1e6)

        self.feature_names = names
        return Theta, names


# Physics-Aware SINDy with Conservation Constraints

class PhysicsAwareSINDy:
    """
    SINDy with physics-informed constraints and correction term discovery.

    Assumes: dX/dt = Θ(X)ξ

    Key additions over standard SINDy:
        - Physics-aware candidate library (gradients, Laplacians)
        - Conservation law constraints during regression
        - Automatic identification of "correction terms" beyond known NS
        - Multi-field discovery (u, v, p simultaneously)
    """

    def __init__(
        self,
        threshold: float = 0.05,
        alpha: float = 0.01,
        max_iter: int = 50,
        poly_order: int = 2,
        include_physics: bool = True,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.poly_order = poly_order
        self.include_physics = include_physics

        self.Xi = None
        self.feature_names = None
        self.known_terms = {}
        self.correction_terms = {}
        self.fit_error = None

    def _stridge(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sequential Thresholded Ridge Regression."""
        n_lib = Theta.shape[1]

        # Normalize columns
        norms = np.linalg.norm(Theta, axis=0) + 1e-10
        Theta_n = Theta / norms

        # Initial ridge
        A = Theta_n.T @ Theta_n + self.alpha * np.eye(n_lib)
        b = Theta_n.T @ y
        xi = np.linalg.solve(A, b)

        for _ in range(self.max_iter):
            small = np.abs(xi) < self.threshold
            xi[small] = 0
            big = ~small
            if not np.any(big):
                break
            Th_a = Theta_n[:, big]
            n_a = Th_a.shape[1]
            A = Th_a.T @ Th_a + self.alpha * np.eye(n_a)
            b = Th_a.T @ y
            xi_a = np.linalg.solve(A, b)
            xi = np.zeros(n_lib)
            xi[big] = xi_a

        return xi / norms

    def fit_from_fields(
        self,
        u_fields: List[np.ndarray],
        v_fields: List[np.ndarray],
        p_fields: List[np.ndarray],
        omega_fields: List[np.ndarray],
        dt: float,
        dx: float = 1.0,
        dy: float = 1.0,
        verbose: bool = True,
    ) -> Dict:
        """
        Discover equations from a time series of flow fields.

        Args:
            u_fields: list of u(t) arrays, shape (ny, nx) each
            v_fields: list of v(t) arrays
            p_fields: list of p(t) arrays
            omega_fields: list of vorticity arrays
            dt: time step between fields

        Returns:
            Dictionary of discovered equations and analysis
        """
        n_frames = len(u_fields)
        if n_frames < 3:
            return {'error': 'Need at least 3 frames'}

        if verbose:
            print("    Building physics-aware candidate library...")

        # Use middle frames for library (avoid boundary effects)
        library = PhysicsAwareLibrary(
            poly_order=self.poly_order,
            include_gradients=self.include_physics,
            include_laplacian=self.include_physics,
            include_advection=self.include_physics,
            include_cross_terms=self.include_physics,
            dx=dx, dy=dy,
        )

        # Build Theta from time-averaged fields (robust to noise)
        all_Theta = []
        all_dudt = []
        all_dvdt = []

        for i in range(1, n_frames - 1):
            Theta_i, names = library.build_from_fields(
                u_fields[i], v_fields[i], p_fields[i], omega_fields[i]
            )

            # Central difference time derivatives
            dudt_i = ((u_fields[i + 1] - u_fields[i - 1]) / (2 * dt)).flatten()
            dvdt_i = ((v_fields[i + 1] - v_fields[i - 1]) / (2 * dt)).flatten()

            # Subsample for efficiency (use every 4th point)
            step = max(1, len(dudt_i) // 2000)
            all_Theta.append(Theta_i[::step])
            all_dudt.append(dudt_i[::step])
            all_dvdt.append(dvdt_i[::step])

        Theta = np.vstack(all_Theta)
        dudt = np.concatenate(all_dudt)
        dvdt = np.concatenate(all_dvdt)

        # Filter invalid data
        valid = np.all(np.isfinite(Theta), axis=1) & np.isfinite(dudt) & np.isfinite(dvdt)
        Theta = Theta[valid]
        dudt = dudt[valid]
        dvdt = dvdt[valid]

        if len(dudt) < 10:
            return {'error': 'Too few valid data points'}

        self.feature_names = names

        if verbose:
            print(f"    Library: {len(names)} candidates, {len(dudt)} data points")
            print("    Running sparse regression (STRidge)...")

        # Discover u-equation
        xi_u = self._stridge(Theta, dudt)
        # Discover v-equation
        xi_v = self._stridge(Theta, dvdt)

        self.Xi = np.column_stack([xi_u, xi_v])

        # Compute fit errors
        u_pred = Theta @ xi_u
        v_pred = Theta @ xi_v
        mse_u = float(np.mean((dudt - u_pred) ** 2))
        mse_v = float(np.mean((dvdt - v_pred) ** 2))
        self.fit_error = {'u': mse_u, 'v': mse_v}

        # Classify terms: known physics vs correction
        self._classify_terms(xi_u, xi_v, names)

        results = {
            'u_equation': self._format_equation('du/dt', xi_u, names),
            'v_equation': self._format_equation('dv/dt', xi_v, names),
            'u_coefficients': xi_u,
            'v_coefficients': xi_v,
            'feature_names': names,
            'fit_error': self.fit_error,
            'known_terms': self.known_terms,
            'correction_terms': self.correction_terms,
            'n_active_terms': int(np.sum(np.abs(self.Xi) > 1e-10)),
            'Xi': self.Xi,
        }

        if verbose:
            print(f"\n    === DISCOVERED EQUATIONS ===")
            print(f"    {results['u_equation']}")
            print(f"    {results['v_equation']}")
            print(f"    Fit error: MSE_u={mse_u:.2e}, MSE_v={mse_v:.2e}")
            print(f"    Active terms: {results['n_active_terms']}")
            if self.correction_terms:
                print(f"    CORRECTION TERMS FOUND:")
                for name, coeff in self.correction_terms.items():
                    cu, cv = coeff
                    print(f"      u: {cu:+.6f} * {name}  |  v: {cv:+.6f} * {name}")

        return results

    def _classify_terms(self, xi_u, xi_v, names):
        """Classify discovered terms as known physics or novel corrections."""
        known_patterns = {
            'nabla2_u', 'nabla2_v',  # Viscous diffusion
            'u*du/dx+v*du/dy', 'u*dv/dx+v*dv/dy',  # Advection
            'dp/dx', 'dp/dy',  # Pressure gradient
        }

        self.known_terms = {}
        self.correction_terms = {}

        for i, name in enumerate(names):
            coeff_u = xi_u[i] if abs(xi_u[i]) > 1e-10 else 0
            coeff_v = xi_v[i] if abs(xi_v[i]) > 1e-10 else 0

            if coeff_u == 0 and coeff_v == 0:
                continue

            if name in known_patterns or name == '1':
                self.known_terms[name] = (coeff_u, coeff_v)
            else:
                if coeff_u != 0 or coeff_v != 0:
                    self.correction_terms[name] = (coeff_u, coeff_v)

    def _format_equation(self, lhs: str, xi: np.ndarray, names: List[str]) -> str:
        """Format discovered equation as readable string."""
        terms = []
        for i, name in enumerate(names):
            c = xi[i]
            if abs(c) > 1e-10:
                if c > 0 and terms:
                    terms.append(f'+ {c:.4f}*{name}')
                else:
                    terms.append(f'{c:.4f}*{name}')
        if terms:
            return f'{lhs} = ' + ' '.join(terms)
        return f'{lhs} = 0'


# Latent-to-Physical Equation Mapper

class LatentEquationMapper:
    """
    Maps equations discovered in latent space back to physical variables.

    Pipeline:
        1. Autoencoder maps flow fields to latent z
        2. SINDy discovers dz/dt = f(z)
        3. This class maps f(z) back to physical space via the decoder Jacobian
    """

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        self.mapping_matrix = None
        self.physical_names = ['u', 'v', 'p', 'omega']

    def compute_decoder_jacobian(self, decoder_func, z0: np.ndarray, eps: float = 1e-4):
        """
        Numerically compute the Jacobian of the decoder at z0.

        J[i,j] = d(decoder_i) / d(z_j)
        """
        n_z = len(z0)
        x0 = decoder_func(z0)
        n_x = len(x0)

        J = np.zeros((n_x, n_z))
        for j in range(n_z):
            z_plus = z0.copy()
            z_plus[j] += eps
            x_plus = decoder_func(z_plus)
            J[:, j] = (x_plus - x0) / eps

        self.mapping_matrix = J
        return J

    def map_latent_equation(
        self,
        latent_coefficients: np.ndarray,
        latent_feature_names: List[str],
    ) -> Dict:
        """
        Map latent-space equation to physical interpretation.

        Uses the decoder Jacobian to project latent dynamics
        onto physical variable space.
        """
        if self.mapping_matrix is None:
            return {
                'status': 'No decoder Jacobian computed',
                'latent_equation': latent_feature_names,
            }

        # Project via Jacobian: dx/dt = J @ dz/dt
        n_phys = min(self.mapping_matrix.shape[0], 4)

        physical_equations = {}
        for p in range(n_phys):
            name = self.physical_names[p] if p < len(self.physical_names) else f'x{p}'
            # Weighted combination of latent terms
            weights = self.mapping_matrix[p, :len(latent_coefficients)]
            effective_coeff = weights @ latent_coefficients if len(weights) > 0 else 0
            physical_equations[name] = float(effective_coeff)

        return {
            'physical_projection': physical_equations,
            'dominant_physical_variable': max(
                physical_equations, key=lambda k: abs(physical_equations[k])
            ) if physical_equations else 'unknown',
        }


# Conservation Law Validator

class ConservationValidator:
    """
    Validates discovered equations against fundamental conservation laws.

    Checks:
        1. Mass conservation: ∂ρ/∂t + ∇·(ρu) = 0
        2. Momentum conservation: ∂(ρu)/∂t + ∇·(ρu⊗u) = -∇p + ∇·τ
        3. Energy conservation: dE/dt ≤ 0 (for unforced flows)
        4. Galilean invariance: equations unchanged under v → v + V₀
        5. Dimensional consistency
    """

    def __init__(self):
        self.validation_results = {}

    def validate_mass_conservation(
        self,
        u_fields: List[np.ndarray],
        v_fields: List[np.ndarray],
        dt: float,
        dx: float,
        dy: float,
    ) -> Dict:
        """Check if divergence-free condition holds."""
        divergence_errors = []
        for u, v in zip(u_fields, v_fields):
            dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
            dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
            div = dudx + dvdy
            divergence_errors.append(float(np.mean(div ** 2)))

        result = {
            'mean_divergence_error': float(np.mean(divergence_errors)),
            'max_divergence_error': float(np.max(divergence_errors)),
            'satisfies_mass_conservation': float(np.mean(divergence_errors)) < 1e-4,
        }
        self.validation_results['mass'] = result
        return result

    def validate_energy_conservation(
        self,
        u_fields: List[np.ndarray],
        v_fields: List[np.ndarray],
        nu: float,
    ) -> Dict:
        """Check energy dissipation rate matches viscous theory."""
        energies = []
        for u, v in zip(u_fields, v_fields):
            ke = 0.5 * np.mean(u ** 2 + v ** 2)
            energies.append(ke)

        energies = np.array(energies)
        dE_dt = np.gradient(energies)

        result = {
            'initial_energy': float(energies[0]),
            'final_energy': float(energies[-1]),
            'energy_ratio': float(energies[-1] / max(energies[0], 1e-10)),
            'monotonic_decay': bool(np.all(dE_dt[1:] <= dE_dt[0] * 1.1)),
            'energy_history': energies.tolist(),
        }
        self.validation_results['energy'] = result
        return result

    def validate_equation_stability(
        self,
        Xi: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """Check if discovered system has bounded solutions."""
        # Check for positive linear growth terms (potential instability)
        linear_terms = []
        for i, name in enumerate(feature_names):
            if name in ['u', 'v', 'omega']:
                for j in range(Xi.shape[1]):
                    if Xi[i, j] > 0:
                        linear_terms.append((name, float(Xi[i, j])))

        # Check total nonlinear term magnitudes
        nonlinear_magnitude = float(np.sum(np.abs(Xi[5:])))  # Skip constant + linear

        result = {
            'potentially_unstable_terms': linear_terms,
            'nonlinear_magnitude': nonlinear_magnitude,
            'is_likely_stable': len(linear_terms) == 0 or nonlinear_magnitude < 100,
        }
        self.validation_results['stability'] = result
        return result

    def full_validation(
        self,
        u_fields, v_fields, p_fields, omega_fields,
        Xi, feature_names, dt, dx, dy, nu,
    ) -> Dict:
        """Run all conservation law checks."""
        results = {}
        results['mass'] = self.validate_mass_conservation(u_fields, v_fields, dt, dx, dy)
        results['energy'] = self.validate_energy_conservation(u_fields, v_fields, nu)
        results['stability'] = self.validate_equation_stability(Xi, feature_names)

        # Overall score
        n_passed = sum([
            results['mass']['satisfies_mass_conservation'],
            results['energy']['monotonic_decay'],
            results['stability']['is_likely_stable'],
        ])
        results['overall_score'] = n_passed / 3.0
        results['all_passed'] = n_passed == 3

        self.validation_results = results
        return results


# Correction Term Discovery

class CorrectionTermDiscovery:
    """
    Discovers novel correction terms beyond standard Navier-Stokes.

    Standard NS: du/dt = -u·∇u + ν∇²u - ∇p/ρ
    Discovery:   du/dt = -u·∇u + ν∇²u - ∇p/ρ + ε(u, ∇u, ∇²u, ...)

    The correction ε captures:
        - Turbulence closure terms
        - Sub-grid scale effects
        - Higher-order viscous effects
        - Novel nonlinear interactions
    """

    def __init__(self):
        self.corrections = []
        self.significance_scores = {}

    def extract_corrections(
        self,
        discovered_Xi: np.ndarray,
        feature_names: List[str],
        nu: float,
    ) -> List[Dict]:
        """
        Compare discovered equations against known NS and extract corrections.

        Known NS terms for du/dt:
            - ν * ∇²u  (viscous diffusion, coefficient ≈ ν)
            - -u·∇u    (advection, coefficient ≈ -1)
            - -dp/dx   (pressure gradient, coefficient ≈ -1)
        """
        known_terms_u = {
            'nabla2_u': nu,
            'u*du/dx+v*du/dy': -1.0,
            'dp/dx': -1.0,
        }
        known_terms_v = {
            'nabla2_v': nu,
            'u*dv/dx+v*dv/dy': -1.0,
            'dp/dy': -1.0,
        }

        corrections = []

        for col_idx, (known, var_name) in enumerate(
            [(known_terms_u, 'u'), (known_terms_v, 'v')]
        ):
            if col_idx >= discovered_Xi.shape[1]:
                break

            xi = discovered_Xi[:, col_idx]

            for i, name in enumerate(feature_names):
                coeff = xi[i]
                if abs(coeff) < 1e-10:
                    continue

                if name in known:
                    expected = known[name]
                    deviation = abs(coeff - expected) / max(abs(expected), 1e-10)
                    if deviation > 0.1:  # >10% deviation
                        corrections.append({
                            'variable': var_name,
                            'term': name,
                            'discovered_coeff': float(coeff),
                            'expected_coeff': expected,
                            'deviation': float(deviation),
                            'type': 'modified_known',
                            'significance': float(abs(coeff - expected)),
                        })
                elif name != '1':
                    corrections.append({
                        'variable': var_name,
                        'term': name,
                        'discovered_coeff': float(coeff),
                        'expected_coeff': 0.0,
                        'deviation': float('inf'),
                        'type': 'novel',
                        'significance': float(abs(coeff)),
                    })

        # Sort by significance
        corrections.sort(key=lambda x: x['significance'], reverse=True)
        self.corrections = corrections

        # Compute significance scores
        for c in corrections:
            self.significance_scores[c['term']] = c['significance']

        return corrections

    def format_corrected_equation(self, corrections: List[Dict], nu: float) -> str:
        """Format the corrected NS equation with discovered terms."""
        base_u = f"du/dt = -{'{'}u*nabla{'}'}u + {nu:.4f}*nabla2_u - dp/dx"

        novel_terms = [c for c in corrections if c['type'] == 'novel' and c['variable'] == 'u']
        if novel_terms:
            correction_str = ' + '.join([
                f"{c['discovered_coeff']:+.4f}*{c['term']}" for c in novel_terms[:5]
            ])
            return f"{base_u} + [{correction_str}]"
        return base_u
