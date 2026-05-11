"""Symbolic Discovery Engine — SINDy-style + Genetic Programming"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import itertools
import copy
import random


# SINDy — Sparse Identification of Nonlinear Dynamics

class SINDyLibrary:
    """
    Builds the library matrix Θ(Z) from candidate nonlinear functions.
    
    Given state Z = [z₁, z₂, ..., z_d], constructs:
        Θ = [1, z₁, z₂, ..., z₁², z₁z₂, ..., sin(z₁), cos(z₁), ...]
    
    Then solves: dZ/dt = Θ(Z) · Ξ  (sparse regression for Ξ)
    """
    
    def __init__(
        self,
        poly_order: int = 3,
        include_trig: bool = True,
        include_exp: bool = False,
        custom_functions: Optional[List[Tuple[str, Callable]]] = None,
    ):
        self.poly_order = poly_order
        self.include_trig = include_trig
        self.include_exp = include_exp
        self.custom_functions = custom_functions or []
        self.feature_names: List[str] = []
    
    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Build library matrix Θ(Z).
        
        Args:
            Z: (n_samples, n_features) state matrix
        
        Returns:
            Theta: (n_samples, n_library) library matrix
        """
        n_samples, n_features = Z.shape
        self.feature_names = []
        columns = []
        
        # Constant term
        columns.append(np.ones(n_samples))
        self.feature_names.append('1')
        
        # Linear terms
        for i in range(n_features):
            columns.append(Z[:, i])
            self.feature_names.append(f'z{i+1}')
        
        # Polynomial terms (degree 2 to poly_order)
        for degree in range(2, self.poly_order + 1):
            for combo in itertools.combinations_with_replacement(range(n_features), degree):
                term = np.ones(n_samples)
                name_parts = []
                for idx in combo:
                    term *= Z[:, idx]
                    name_parts.append(f'z{idx+1}')
                columns.append(term)
                self.feature_names.append('·'.join(name_parts))
        
        # Trigonometric terms
        if self.include_trig:
            for i in range(n_features):
                columns.append(np.sin(Z[:, i]))
                self.feature_names.append(f'sin(z{i+1})')
                columns.append(np.cos(Z[:, i]))
                self.feature_names.append(f'cos(z{i+1})')
        
        # Exponential terms (with clipping for stability)
        if self.include_exp:
            for i in range(n_features):
                columns.append(np.exp(np.clip(Z[:, i], -5, 5)))
                self.feature_names.append(f'exp(z{i+1})')
        
        # Custom functions
        for name, func in self.custom_functions:
            for i in range(n_features):
                try:
                    columns.append(func(Z[:, i]))
                    self.feature_names.append(f'{name}(z{i+1})')
                except Exception:
                    pass
        
        return np.column_stack(columns)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class STRidge:
    """
    Sequential Thresholded Ridge Regression (STRidge).
    
    Iteratively performs ridge regression then thresholds small coefficients
    to zero, promoting sparsity. This is the default SINDy solver.
    
    Algorithm:
        1. Solve Ξ = (Θᵀ Θ + λI)⁻¹ Θᵀ dZ/dt
        2. Set Ξᵢⱼ = 0 if |Ξᵢⱼ| < threshold
        3. Re-solve with remaining features
        4. Repeat until convergence
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        alpha: float = 0.05,
        max_iter: int = 30,
        normalize_columns: bool = True,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize_columns = normalize_columns
    
    def fit(
        self,
        Theta: np.ndarray,
        dZdt: np.ndarray,
    ) -> np.ndarray:
        """
        Solve for sparse coefficient matrix Ξ.
        
        dZ/dt = Θ(Z) · Ξ
        
        Args:
            Theta: (n_samples, n_library) library matrix
            dZdt: (n_samples, n_features) time derivatives
        
        Returns:
            Xi: (n_library, n_features) sparse coefficient matrix
        """
        n_lib = Theta.shape[1]
        n_feat = dZdt.shape[1] if dZdt.ndim > 1 else 1
        
        if dZdt.ndim == 1:
            dZdt = dZdt.reshape(-1, 1)
        
        # Column normalization
        if self.normalize_columns:
            norms = np.linalg.norm(Theta, axis=0) + 1e-10
            Theta_norm = Theta / norms
        else:
            Theta_norm = Theta
            norms = np.ones(n_lib)
        
        Xi = np.zeros((n_lib, n_feat))
        
        for j in range(n_feat):
            y = dZdt[:, j]
            xi = self._stridge_single(Theta_norm, y)
            Xi[:, j] = xi / norms  # Unnormalize
        
        return Xi
    
    def _stridge_single(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """STRidge for a single target variable."""
        n_lib = Theta.shape[1]
        
        # Initial ridge regression
        A = Theta.T @ Theta + self.alpha * np.eye(n_lib)
        b = Theta.T @ y
        xi = np.linalg.solve(A, b)
        
        for iteration in range(self.max_iter):
            # Threshold small coefficients
            small_inds = np.abs(xi) < self.threshold
            xi[small_inds] = 0
            
            # Get active set
            big_inds = ~small_inds
            if not np.any(big_inds):
                break
            
            # Re-solve with active features only
            Theta_active = Theta[:, big_inds]
            n_active = Theta_active.shape[1]
            A = Theta_active.T @ Theta_active + self.alpha * np.eye(n_active)
            b = Theta_active.T @ y
            xi_active = np.linalg.solve(A, b)
            
            xi = np.zeros(n_lib)
            xi[big_inds] = xi_active
        
        return xi


class SINDy:
    """
    Complete SINDy pipeline for equation discovery from latent trajectories.
    
    Usage:
        sindy = SINDy(poly_order=3)
        equations = sindy.fit(Z_trajectory, dt=0.01)
        print(sindy.print_equations())
    """
    
    def __init__(
        self,
        poly_order: int = 3,
        threshold: float = 0.1,
        alpha: float = 0.05,
        include_trig: bool = True,
        include_exp: bool = False,
    ):
        self.library = SINDyLibrary(
            poly_order=poly_order,
            include_trig=include_trig,
            include_exp=include_exp,
        )
        self.solver = STRidge(
            threshold=threshold,
            alpha=alpha,
        )
        self.Xi = None
        self.feature_names = None
    
    def fit(
        self,
        Z: np.ndarray,
        dt: float = 0.01,
        dZdt: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Discover governing equations from trajectory data.
        
        Args:
            Z: (n_timesteps, n_features) state trajectory
            dt: time step (used to compute numerical derivatives)
            dZdt: optional pre-computed derivatives
        
        Returns:
            Xi: sparse coefficient matrix
        """
        # Compute time derivatives if not provided
        if dZdt is None:
            # Central differences (interior), forward/backward at boundaries
            dZdt = np.zeros_like(Z)
            dZdt[1:-1] = (Z[2:] - Z[:-2]) / (2 * dt)
            dZdt[0] = (Z[1] - Z[0]) / dt
            dZdt[-1] = (Z[-1] - Z[-2]) / dt
        
        # Build library
        Theta = self.library.fit_transform(Z)
        self.feature_names = self.library.get_feature_names()
        
        # Sparse regression
        self.Xi = self.solver.fit(Theta, dZdt)
        
        return self.Xi
    
    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict dZ/dt using discovered equations."""
        Theta = self.library.fit_transform(Z)
        return Theta @ self.Xi
    
    def simulate(
        self,
        z0: np.ndarray,
        t_span: np.ndarray,
        integrator: str = 'rk4',
    ) -> np.ndarray:
        """
        Simulate the discovered system forward in time.
        
        Args:
            z0: (n_features,) initial condition
            t_span: (n_timesteps,) time points
        """
        n_steps = len(t_span)
        n_feat = len(z0)
        Z = np.zeros((n_steps, n_feat))
        Z[0] = z0
        
        for i in range(n_steps - 1):
            dt = t_span[i+1] - t_span[i]
            
            if integrator == 'euler':
                dz = self.predict(Z[i:i+1])[0]
                Z[i+1] = Z[i] + dt * dz
            elif integrator == 'rk4':
                k1 = self.predict(Z[i:i+1])[0]
                k2 = self.predict((Z[i] + 0.5*dt*k1).reshape(1, -1))[0]
                k3 = self.predict((Z[i] + 0.5*dt*k2).reshape(1, -1))[0]
                k4 = self.predict((Z[i] + dt*k3).reshape(1, -1))[0]
                Z[i+1] = Z[i] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return Z
    
    def get_equations(self) -> List[str]:
        """Return discovered equations as human-readable strings."""
        if self.Xi is None:
            return ["No equations discovered yet. Call fit() first."]
        
        equations = []
        n_feat = self.Xi.shape[1]
        
        for j in range(n_feat):
            terms = []
            for i, name in enumerate(self.feature_names):
                coeff = self.Xi[i, j]
                if abs(coeff) > 1e-10:
                    if coeff > 0 and terms:
                        terms.append(f'+ {coeff:.4f}·{name}')
                    else:
                        terms.append(f'{coeff:.4f}·{name}')
            
            if terms:
                eq = f'dz{j+1}/dt = ' + ' '.join(terms)
            else:
                eq = f'dz{j+1}/dt = 0'
            equations.append(eq)
        
        return equations
    
    def get_complexity(self) -> int:
        """Count number of non-zero terms (model complexity)."""
        if self.Xi is None:
            return 0
        return int(np.sum(np.abs(self.Xi) > 1e-10))
    
    def get_equation_error(
        self,
        Z: np.ndarray,
        dZdt: np.ndarray,
    ) -> float:
        """Compute equation residual error."""
        dZdt_pred = self.predict(Z)
        return float(np.mean((dZdt - dZdt_pred) ** 2))


# Genetic Programming — Symbolic Regression

class ExprNode:
    """Node in a symbolic expression tree."""
    
    OPERATORS = {
        '+': (lambda a, b: a + b, 2),
        '-': (lambda a, b: a - b, 2),
        '*': (lambda a, b: a * b, 2),
        '/': (lambda a, b: np.where(np.abs(b) > 1e-10, a / b, 0.0), 2),
        'sin': (lambda a: np.sin(a), 1),
        'cos': (lambda a: np.cos(a), 1),
        'exp': (lambda a: np.exp(np.clip(a, -5, 5)), 1),
        'sqrt': (lambda a: np.sqrt(np.abs(a)), 1),
        'sq': (lambda a: a ** 2, 1),
        'neg': (lambda a: -a, 1),
    }
    
    def __init__(self, op=None, value=None, var_idx=None, children=None):
        self.op = op          # Operator name
        self.value = value    # Constant value (float)
        self.var_idx = var_idx  # Variable index (int)
        self.children = children or []
    
    @property
    def is_leaf(self):
        return self.op is None
    
    @property
    def is_constant(self):
        return self.value is not None
    
    @property
    def is_variable(self):
        return self.var_idx is not None
    
    def evaluate(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate expression tree on data Z (n_samples, n_vars)."""
        if self.is_constant:
            return np.full(Z.shape[0], self.value)
        elif self.is_variable:
            return Z[:, self.var_idx]
        else:
            func, arity = self.OPERATORS[self.op]
            child_vals = [c.evaluate(Z) for c in self.children]
            try:
                result = func(*child_vals)
                return np.where(np.isfinite(result), result, 0.0)
            except Exception:
                return np.zeros(Z.shape[0])
    
    def complexity(self) -> int:
        """Count number of nodes in the tree."""
        if self.is_leaf:
            return 1
        return 1 + sum(c.complexity() for c in self.children)
    
    def __str__(self) -> str:
        if self.is_constant:
            return f'{self.value:.3f}'
        elif self.is_variable:
            return f'z{self.var_idx + 1}'
        elif len(self.children) == 1:
            return f'{self.op}({self.children[0]})'
        else:
            return f'({self.children[0]} {self.op} {self.children[1]})'
    
    def copy(self):
        return copy.deepcopy(self)


class GeneticProgramming:
    """
    Genetic Programming for symbolic regression of turbulence dynamics.
    
    Evolves mathematical expressions to fit dz/dt data.
    Uses tournament selection, subtree crossover, and point mutation.
    
    Fitness = -MSE - λ * complexity  (parsimony pressure)
    """
    
    def __init__(
        self,
        n_vars: int = 4,
        population_size: int = 200,
        max_depth: int = 5,
        parsimony_coefficient: float = 0.001,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 5,
    ):
        self.n_vars = n_vars
        self.population_size = population_size
        self.max_depth = max_depth
        self.parsimony = parsimony_coefficient
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
        self.binary_ops = ['+', '-', '*']
        self.unary_ops = ['sin', 'cos', 'sq', 'neg']
        self.constants = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, np.pi]
        
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.history = []
    
    def _random_tree(self, depth: int = 0) -> ExprNode:
        """Generate a random expression tree."""
        if depth >= self.max_depth or (depth > 1 and random.random() < 0.4):
            # Leaf node
            if random.random() < 0.7:
                return ExprNode(var_idx=random.randint(0, self.n_vars - 1))
            else:
                return ExprNode(value=random.choice(self.constants))
        
        # Internal node
        if random.random() < 0.6:
            # Binary
            op = random.choice(self.binary_ops)
            left = self._random_tree(depth + 1)
            right = self._random_tree(depth + 1)
            return ExprNode(op=op, children=[left, right])
        else:
            # Unary
            op = random.choice(self.unary_ops)
            child = self._random_tree(depth + 1)
            return ExprNode(op=op, children=[child])
    
    def _fitness(self, individual: ExprNode, Z: np.ndarray, target: np.ndarray) -> float:
        """Compute fitness = -MSE - parsimony * complexity."""
        try:
            pred = individual.evaluate(Z)
            if not np.all(np.isfinite(pred)):
                return -1e10
            mse = np.mean((pred - target) ** 2)
            complexity = individual.complexity()
            return -mse - self.parsimony * complexity
        except Exception:
            return -1e10
    
    def _tournament_select(self, fitnesses: np.ndarray) -> int:
        """Tournament selection."""
        indices = random.sample(range(len(self.population)), self.tournament_size)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return best_idx
    
    def _get_all_nodes(self, node: ExprNode) -> List[ExprNode]:
        """Get all nodes in a tree (for crossover/mutation)."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _crossover(self, p1: ExprNode, p2: ExprNode) -> ExprNode:
        """Subtree crossover."""
        child = p1.copy()
        
        nodes1 = self._get_all_nodes(child)
        nodes2 = self._get_all_nodes(p2)
        
        if len(nodes1) <= 1 or len(nodes2) <= 1:
            return child
        
        # Select random non-root node from child to replace
        replace_node = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
        donor_node = random.choice(nodes2).copy()
        
        # Find parent and replace
        for node in nodes1:
            for i, c in enumerate(node.children):
                if c is replace_node:
                    node.children[i] = donor_node
                    return child
        
        return child
    
    def _mutate(self, individual: ExprNode) -> ExprNode:
        """Point mutation."""
        mutant = individual.copy()
        nodes = self._get_all_nodes(mutant)
        
        if not nodes:
            return mutant
        
        target = random.choice(nodes)
        
        if target.is_constant:
            target.value = random.choice(self.constants)
        elif target.is_variable:
            target.var_idx = random.randint(0, self.n_vars - 1)
        elif target.op in self.binary_ops:
            target.op = random.choice(self.binary_ops)
        elif target.op in self.unary_ops:
            target.op = random.choice(self.unary_ops)
        
        return mutant
    
    def fit(
        self,
        Z: np.ndarray,
        target: np.ndarray,
        n_generations: int = 50,
        verbose: bool = True,
    ) -> ExprNode:
        """
        Evolve symbolic expressions to fit target data.
        
        Args:
            Z: (n_samples, n_vars) input features
            target: (n_samples,) target values (e.g., dz₁/dt)
            n_generations: number of evolution generations
        
        Returns:
            Best evolved expression tree
        """
        # Initialize population
        self.population = [self._random_tree() for _ in range(self.population_size)]
        
        for gen in range(n_generations):
            # Evaluate fitness
            fitnesses = np.array([
                self._fitness(ind, Z, target) for ind in self.population
            ])
            
            # Track best
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            # Guard: if no individual ever had finite fitness, create a trivial one
            if self.best_individual is None:
                self.best_individual = ExprNode(value=0.0)
                self.best_fitness = self._fitness(self.best_individual, Z, target)
            
            self.history.append({
                'generation': gen,
                'best_fitness': float(self.best_fitness),
                'mean_fitness': float(np.nanmean(fitnesses)),
                'best_complexity': self.best_individual.complexity(),
                'best_equation': str(self.best_individual),
            })
            
            if verbose and (gen + 1) % 10 == 0:
                mse = -self.best_fitness - self.parsimony * self.best_individual.complexity()
                print(f"  Gen {gen+1:3d} | MSE: {abs(mse):.6f} | "
                      f"Complexity: {self.best_individual.complexity()} | "
                      f"Eq: {str(self.best_individual)[:60]}")
            
            # Create next generation
            new_pop = [self.best_individual.copy()]  # Elitism
            
            while len(new_pop) < self.population_size:
                r = random.random()
                if r < self.crossover_prob:
                    p1 = self.population[self._tournament_select(fitnesses)]
                    p2 = self.population[self._tournament_select(fitnesses)]
                    child = self._crossover(p1, p2)
                elif r < self.crossover_prob + self.mutation_prob:
                    parent = self.population[self._tournament_select(fitnesses)]
                    child = self._mutate(parent)
                else:
                    child = self._random_tree()
                
                # Depth limit
                if child.complexity() <= 2 ** (self.max_depth + 1):
                    new_pop.append(child)
            
            self.population = new_pop[:self.population_size]
        
        return self.best_individual
    
    def get_best_equation(self) -> str:
        if self.best_individual is None:
            return "No equation discovered yet."
        return str(self.best_individual)


class SymbolicDiscoveryEngine:
    """
    High-level symbolic discovery pipeline combining SINDy and GP.
    
    Pipeline:
        1. Receive latent trajectories from autoencoder
        2. Run SINDy for sparse polynomial/trig models
        3. Run GP for free-form symbolic regression
        4. Compare and rank discovered equations
        5. Cross-validate on held-out trajectories
    """
    
    def __init__(
        self,
        n_latent_dims: int = 8,
        sindy_threshold: float = 0.1,
        sindy_poly_order: int = 3,
        gp_population: int = 200,
        gp_generations: int = 50,
    ):
        self.n_dims = n_latent_dims
        
        self.sindy = SINDy(
            poly_order=sindy_poly_order,
            threshold=sindy_threshold,
            include_trig=True,
        )
        
        self.gp_engines = {}
        self.gp_config = {
            'population_size': gp_population,
            'n_generations': gp_generations,
        }
        
        self.results = {}
    
    def discover_from_trajectories(
        self,
        Z: np.ndarray,
        dt: float = 0.01,
        run_gp: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Run full discovery pipeline on latent trajectories.
        
        Args:
            Z: (n_timesteps, n_latent) latent trajectory
            dt: time step between frames
        
        Returns:
            Dictionary of discovered equations and metrics
        """
        if verbose:
            print("\n" + "="*60)
            print("  SYMBOLIC DISCOVERY ENGINE")
            print("="*60)
        
        # Normalize data to prevent overflow in polynomial library
        Z_mean = Z.mean(axis=0)
        Z_std = Z.std(axis=0) + 1e-10
        Z_norm = (Z - Z_mean) / Z_std
        
        if verbose:
            print(f"  Data normalized: mean={np.mean(np.abs(Z_mean)):.4f}, "
                  f"std={np.mean(Z_std):.4f}")
        
        # 1. SINDy Discovery
        if verbose:
            print("\n  [1/2] SINDy — Sparse Regression Discovery")
            print("  " + "-"*50)
        
        self.sindy.fit(Z_norm, dt=dt)
        sindy_eqs = self.sindy.get_equations()
        sindy_complexity = self.sindy.get_complexity()
        
        # Compute error
        dZdt = np.zeros_like(Z_norm)
        dZdt[1:-1] = (Z_norm[2:] - Z_norm[:-2]) / (2 * dt)
        dZdt[0] = (Z_norm[1] - Z_norm[0]) / dt
        dZdt[-1] = (Z_norm[-1] - Z_norm[-2]) / dt
        
        sindy_error = self.sindy.get_equation_error(Z_norm, dZdt)
        
        if verbose:
            print(f"\n  SINDy Results (complexity={sindy_complexity}, MSE={sindy_error:.6f}):")
            for eq in sindy_eqs[:min(8, len(sindy_eqs))]:
                print(f"    {eq[:80]}")
        
        self.results['sindy'] = {
            'equations': sindy_eqs,
            'complexity': sindy_complexity,
            'error': sindy_error,
            'Xi': self.sindy.Xi,
        }
        
        # 2. Genetic Programming Discovery
        if run_gp:
            if verbose:
                print("\n  [2/2] Genetic Programming — Symbolic Regression")
                print("  " + "-"*50)
            
            gp_equations = {}
            n_dims_to_fit = min(Z_norm.shape[1], 4)  # Limit GP to first few dims
            
            for dim in range(n_dims_to_fit):
                if verbose:
                    print(f"\n  Evolving equation for dz{dim+1}/dt...")
                
                gp = GeneticProgramming(
                    n_vars=Z_norm.shape[1],
                    population_size=self.gp_config['population_size'],
                    max_depth=5,
                    parsimony_coefficient=0.001,
                )
                
                best = gp.fit(
                    Z_norm, dZdt[:, dim],
                    n_generations=self.gp_config['n_generations'],
                    verbose=verbose,
                )
                
                gp_equations[f'dz{dim+1}/dt'] = {
                    'equation': str(best),
                    'complexity': best.complexity(),
                    'fitness': gp.best_fitness,
                }
                self.gp_engines[dim] = gp
            
            self.results['gp'] = gp_equations
        
        # Summary
        if verbose:
            print("\n" + "="*60)
            print("  DISCOVERY SUMMARY")
            print("="*60)
            print(f"  SINDy:  {sindy_complexity} terms, MSE = {sindy_error:.6f}")
            if run_gp and 'gp' in self.results:
                for k, v in self.results['gp'].items():
                    print(f"  GP {k}: {v['equation'][:50]}... "
                          f"(complexity={v['complexity']})")
        
        return self.results
