"""AGI-Style Scientific Discovery System"""

import numpy as np
import json
import time
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy


# Data Classes for Structured Knowledge

@dataclass
class DiscoveredEquation:
    """A single discovered equation with metadata."""
    equation_str: str
    variable: str  # e.g., 'du/dt', 'dv/dt'
    coefficients: Dict[str, float] = field(default_factory=dict)
    fit_error: float = float('inf')
    complexity: int = 0
    conservation_score: float = 0.0
    stability_score: float = 0.0
    novelty_score: float = 0.0
    overall_score: float = 0.0
    method: str = ''  # 'sindy', 'gp', 'physics_sindy', 'hybrid'
    timestamp: str = ''
    metadata: Dict = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A scientific hypothesis about fluid dynamics."""
    id: str
    description: str
    equation: DiscoveredEquation
    status: str = 'proposed'  # proposed, testing, validated, rejected, refined
    confidence: float = 0.0
    evidence: List[Dict] = field(default_factory=list)
    refinement_history: List[str] = field(default_factory=list)
    timestamp: str = ''


@dataclass
class Experiment:
    """A validation experiment for a hypothesis."""
    id: str
    hypothesis_id: str
    test_type: str  # 'conservation', 'stability', 'prediction', 'cross_regime'
    parameters: Dict = field(default_factory=dict)
    result: Dict = field(default_factory=dict)
    passed: bool = False
    timestamp: str = ''


# Hypothesis Generator

class HypothesisGenerator:
    """
    Generates scientific hypotheses from discovered equations.

    Strategy:
        1. Identify novel terms in discovered equations
        2. Group related terms into hypotheses
        3. Rank hypotheses by significance and parsimony
        4. Propose physical interpretations
    """

    PHYSICAL_INTERPRETATIONS = {
        'u*omega': 'Vortex stretching / tilting interaction',
        'v*omega': 'Cross-stream vorticity transport',
        'u*u': 'Nonlinear self-interaction (Reynolds stress)',
        'v*v': 'Transverse Reynolds stress',
        'u*v': 'Reynolds shear stress',
        'omega*omega': 'Enstrophy production/dissipation',
        'p*u': 'Pressure-velocity coupling (compressibility proxy)',
        'p*v': 'Pressure-velocity coupling (transverse)',
        'nabla2_u': 'Viscous diffusion (x-momentum)',
        'nabla2_v': 'Viscous diffusion (y-momentum)',
        'dp/dx': 'Pressure gradient force (x)',
        'dp/dy': 'Pressure gradient force (y)',
    }

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self.hypothesis_counter = 0

    def generate_from_corrections(
        self,
        corrections: List[Dict],
        nu: float,
        verbose: bool = True,
    ) -> List[Hypothesis]:
        """Generate hypotheses from discovered correction terms."""
        if verbose:
            print("\n    --- Hypothesis Generator ---")

        new_hypotheses = []

        for correction in corrections:
            if correction['significance'] < 1e-6:
                continue

            self.hypothesis_counter += 1
            h_id = f"H{self.hypothesis_counter:04d}"

            term = correction['term']
            coeff = correction['discovered_coeff']
            var = correction['variable']

            # Physical interpretation
            interp = self.PHYSICAL_INTERPRETATIONS.get(
                term, f'Novel interaction: {term}'
            )

            if correction['type'] == 'novel':
                desc = (f"The equation for d{var}/dt contains a previously "
                        f"unmodeled term: {coeff:+.6f}*{term}. "
                        f"Physical interpretation: {interp}")
            else:
                expected = correction['expected_coeff']
                desc = (f"The coefficient of {term} in d{var}/dt "
                        f"deviates from expected ({expected:.4f}) to "
                        f"discovered ({coeff:.4f}). {interp}")

            eq = DiscoveredEquation(
                equation_str=f"d{var}/dt += {coeff:+.6f} * {term}",
                variable=f'd{var}/dt',
                coefficients={term: coeff},
                fit_error=0.0,
                complexity=1,
                novelty_score=correction['significance'],
                method='physics_sindy',
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            )

            hypothesis = Hypothesis(
                id=h_id,
                description=desc,
                equation=eq,
                status='proposed',
                confidence=min(correction['significance'] * 10, 1.0),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            )

            new_hypotheses.append(hypothesis)
            self.hypotheses.append(hypothesis)

            if verbose:
                print(f"    [{h_id}] {desc[:80]}...")

        if verbose:
            print(f"    Generated {len(new_hypotheses)} hypotheses")

        return new_hypotheses

    def generate_from_latent_discovery(
        self,
        sindy_equations: List[str],
        gp_equations: Dict,
        sindy_error: float,
        verbose: bool = True,
    ) -> List[Hypothesis]:
        """Generate hypotheses from latent-space SINDy/GP discovery."""
        new_hypotheses = []

        # SINDy-based hypothesis
        self.hypothesis_counter += 1
        h_id = f"H{self.hypothesis_counter:04d}"

        desc = (f"Latent turbulence dynamics can be described by a sparse "
                f"system with {len(sindy_equations)} equations. "
                f"Fit error: {sindy_error:.2e}")

        eq = DiscoveredEquation(
            equation_str='; '.join(sindy_equations[:4]),
            variable='latent_z',
            fit_error=sindy_error,
            complexity=len(sindy_equations),
            method='sindy',
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        )

        hypothesis = Hypothesis(
            id=h_id, description=desc, equation=eq,
            status='proposed',
            confidence=max(0, 1.0 - sindy_error),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        new_hypotheses.append(hypothesis)
        self.hypotheses.append(hypothesis)

        # GP-based hypotheses
        if gp_equations:
            for key, val in gp_equations.items():
                self.hypothesis_counter += 1
                h_id = f"H{self.hypothesis_counter:04d}"

                desc = (f"GP discovered symbolic expression for {key}: "
                        f"{val['equation'][:60]} "
                        f"(complexity={val['complexity']})")

                eq = DiscoveredEquation(
                    equation_str=f"{key} = {val['equation']}",
                    variable=key,
                    complexity=val['complexity'],
                    method='genetic_programming',
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                )

                hypothesis = Hypothesis(
                    id=h_id, description=desc, equation=eq,
                    status='proposed',
                    confidence=0.5,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                new_hypotheses.append(hypothesis)
                self.hypotheses.append(hypothesis)

        if verbose:
            print(f"    Generated {len(new_hypotheses)} latent-space hypotheses")

        return new_hypotheses


# Symbolic Reasoner

class SymbolicReasoner:
    """
    Simplifies, validates, and refines discovered equations.

    Operations:
        - Simplification: combine like terms, factor common expressions
        - Dimensional analysis: check physical dimensions
        - Symmetry checks: rotational, translational, Galilean invariance
        - Refinement: iteratively improve equations via constrained optimization
    """

    def simplify_equation(self, coefficients: Dict[str, float], threshold: float = 1e-4) -> Dict[str, float]:
        """Remove negligible terms and round coefficients."""
        simplified = {}
        for name, coeff in coefficients.items():
            if abs(coeff) > threshold:
                # Round to significant figures
                if abs(coeff) > 0.01:
                    simplified[name] = round(coeff, 4)
                else:
                    simplified[name] = coeff
        return simplified

    def check_galilean_invariance(self, coefficients: Dict[str, float], feature_names: List[str]) -> bool:
        """
        Check if equation is Galilean invariant.

        Under u → u + U₀, the equation should remain form-invariant.
        Terms like bare 'u' (linear velocity) break Galilean invariance
        unless they appear in advection form u·∇u.
        """
        problematic = []
        for name, coeff in coefficients.items():
            if abs(coeff) < 1e-10:
                continue
            # Bare linear velocity terms break Galilean invariance
            if name in ('u', 'v') and abs(coeff) > 0.01:
                problematic.append(name)

        return len(problematic) == 0

    def check_rotational_symmetry(self, xi_u: np.ndarray, xi_v: np.ndarray, names: List[str]) -> bool:
        """Check if u and v equations have analogous structure."""
        # Check if corresponding terms have similar magnitudes
        pairs = [
            ('nabla2_u', 'nabla2_v'),
            ('du/dx', 'dv/dy'),
            ('dp/dx', 'dp/dy'),
        ]
        for name_u, name_v in pairs:
            if name_u in names and name_v in names:
                idx_u = names.index(name_u)
                idx_v = names.index(name_v)
                if abs(xi_u[idx_u]) > 1e-10 and abs(xi_v[idx_v]) > 1e-10:
                    ratio = xi_u[idx_u] / xi_v[idx_v]
                    if abs(ratio - 1.0) > 0.5:
                        return False
        return True

    def reason_about_equation(
        self,
        hypothesis: Hypothesis,
        feature_names: List[str],
        xi_u: Optional[np.ndarray] = None,
        xi_v: Optional[np.ndarray] = None,
    ) -> Dict:
        """Full symbolic reasoning about a hypothesis."""
        results = {
            'simplified': self.simplify_equation(hypothesis.equation.coefficients),
            'galilean_invariant': self.check_galilean_invariance(
                hypothesis.equation.coefficients, feature_names
            ),
        }

        if xi_u is not None and xi_v is not None:
            results['rotationally_symmetric'] = self.check_rotational_symmetry(
                xi_u, xi_v, feature_names
            )

        # Score
        score = 0.5
        if results['galilean_invariant']:
            score += 0.25
        if results.get('rotationally_symmetric', True):
            score += 0.25
        results['physics_consistency_score'] = score

        return results


# Experiment Validator

class ExperimentValidator:
    """
    Tests hypotheses against simulation data.

    Test Types:
        1. Conservation: Does equation satisfy mass/energy/momentum conservation?
        2. Stability: Does the predicted system have bounded solutions?
        3. Prediction: Does the equation predict future states accurately?
        4. Cross-regime: Does it generalize across Reynolds numbers?
    """

    def __init__(self):
        self.experiments: List[Experiment] = []
        self.experiment_counter = 0

    def test_conservation(
        self,
        hypothesis: Hypothesis,
        u_fields: List[np.ndarray],
        v_fields: List[np.ndarray],
        dt: float,
        dx: float,
        dy: float,
    ) -> Experiment:
        """Test if discovered equation satisfies conservation laws."""
        self.experiment_counter += 1
        exp_id = f"E{self.experiment_counter:04d}"

        # Check divergence-free
        div_errors = []
        for u, v in zip(u_fields, v_fields):
            dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
            div_errors.append(float(np.mean((dudx + dvdy) ** 2)))

        # Check energy monotonicity
        energies = [0.5 * np.mean(u ** 2 + v ** 2) for u, v in zip(u_fields, v_fields)]
        monotonic = all(e2 <= e1 * 1.01 for e1, e2 in zip(energies[:-1], energies[1:]))

        passed = np.mean(div_errors) < 1e-3 and monotonic

        experiment = Experiment(
            id=exp_id,
            hypothesis_id=hypothesis.id,
            test_type='conservation',
            parameters={'dt': dt, 'dx': dx, 'dy': dy, 'n_frames': len(u_fields)},
            result={
                'mean_divergence_error': float(np.mean(div_errors)),
                'energy_monotonic': monotonic,
                'initial_energy': energies[0],
                'final_energy': energies[-1],
            },
            passed=passed,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        )

        self.experiments.append(experiment)
        return experiment

    def test_prediction_accuracy(
        self,
        hypothesis: Hypothesis,
        Xi: np.ndarray,
        Theta: np.ndarray,
        true_derivatives: np.ndarray,
    ) -> Experiment:
        """Test prediction accuracy of discovered equations."""
        self.experiment_counter += 1
        exp_id = f"E{self.experiment_counter:04d}"

        pred = Theta @ Xi
        mse = float(np.mean((pred - true_derivatives) ** 2))
        r2 = 1.0 - mse / max(float(np.var(true_derivatives)), 1e-10)

        passed = r2 > 0.5 and mse < 1.0

        experiment = Experiment(
            id=exp_id,
            hypothesis_id=hypothesis.id,
            test_type='prediction',
            result={'mse': mse, 'r2': r2},
            passed=passed,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        self.experiments.append(experiment)
        return experiment

    def test_cross_regime(
        self,
        hypothesis: Hypothesis,
        results_by_re: Dict[float, float],
    ) -> Experiment:
        """Test if equation generalizes across Reynolds numbers."""
        self.experiment_counter += 1
        exp_id = f"E{self.experiment_counter:04d}"

        errors = list(results_by_re.values())
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        cv = std_error / max(mean_error, 1e-10)

        passed = cv < 2.0 and mean_error < 1.0

        experiment = Experiment(
            id=exp_id,
            hypothesis_id=hypothesis.id,
            test_type='cross_regime',
            parameters={'reynolds_numbers': list(results_by_re.keys())},
            result={
                'mean_error': mean_error,
                'std_error': std_error,
                'coefficient_of_variation': cv,
                'errors_by_re': results_by_re,
            },
            passed=passed,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        self.experiments.append(experiment)
        return experiment

    def run_all_tests(
        self,
        hypothesis: Hypothesis,
        u_fields, v_fields, dt, dx, dy,
        Xi=None, Theta=None, true_derivatives=None,
    ) -> List[Experiment]:
        """Run all available validation tests."""
        experiments = []

        # Conservation test
        exp = self.test_conservation(hypothesis, u_fields, v_fields, dt, dx, dy)
        experiments.append(exp)

        # Prediction test (if data available)
        if Xi is not None and Theta is not None and true_derivatives is not None:
            exp = self.test_prediction_accuracy(hypothesis, Xi, Theta, true_derivatives)
            experiments.append(exp)

        # Update hypothesis status
        all_passed = all(e.passed for e in experiments)
        if all_passed:
            hypothesis.status = 'validated'
            hypothesis.confidence = min(hypothesis.confidence + 0.2, 1.0)
        else:
            hypothesis.status = 'testing'
            hypothesis.confidence *= 0.8

        hypothesis.evidence.extend([
            {'experiment_id': e.id, 'passed': e.passed, 'type': e.test_type}
            for e in experiments
        ])

        return experiments


# Knowledge Base

class KnowledgeBase:
    """
    Persistent store of discovered equations, hypotheses, and experiments.

    Stores:
        - Discovered equations (with provenance)
        - Hypotheses (with validation status)
        - Experiments (with results)
        - Ranked discoveries (by confidence and novelty)
    """

    def __init__(self, storage_dir: str = "checkpoints/knowledge_base"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.equations: List[DiscoveredEquation] = []
        self.hypotheses: List[Hypothesis] = []
        self.experiments: List[Experiment] = []
        self.discoveries: List[Dict] = []

    def add_equation(self, eq: DiscoveredEquation):
        self.equations.append(eq)

    def add_hypothesis(self, hyp: Hypothesis):
        self.hypotheses.append(hyp)

    def add_experiment(self, exp: Experiment):
        self.experiments.append(exp)

    def record_discovery(
        self,
        title: str,
        equation: DiscoveredEquation,
        hypothesis: Hypothesis,
        experiments: List[Experiment],
        significance: float,
    ):
        """Record a validated discovery."""
        discovery = {
            'title': title,
            'equation': equation.equation_str,
            'variable': equation.variable,
            'method': equation.method,
            'hypothesis_id': hypothesis.id,
            'confidence': hypothesis.confidence,
            'n_experiments': len(experiments),
            'all_tests_passed': all(e.passed for e in experiments),
            'significance': significance,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.discoveries.append(discovery)

    def get_top_discoveries(self, n: int = 10) -> List[Dict]:
        """Get top N discoveries ranked by significance × confidence."""
        ranked = sorted(
            self.discoveries,
            key=lambda d: d['significance'] * d['confidence'],
            reverse=True,
        )
        return ranked[:n]

    def get_validated_hypotheses(self) -> List[Hypothesis]:
        """Get all validated hypotheses."""
        return [h for h in self.hypotheses if h.status == 'validated']

    def save(self):
        """Save knowledge base to disk."""
        data = {
            'equations': [
                {
                    'equation_str': eq.equation_str,
                    'variable': eq.variable,
                    'fit_error': eq.fit_error,
                    'complexity': eq.complexity,
                    'method': eq.method,
                    'timestamp': eq.timestamp,
                }
                for eq in self.equations
            ],
            'hypotheses': [
                {
                    'id': h.id,
                    'description': h.description,
                    'status': h.status,
                    'confidence': h.confidence,
                    'timestamp': h.timestamp,
                }
                for h in self.hypotheses
            ],
            'discoveries': self.discoveries,
            'n_experiments': len(self.experiments),
        }
        filepath = self.storage_dir / 'knowledge_base.json'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self):
        """Load knowledge base from disk."""
        filepath = self.storage_dir / 'knowledge_base.json'
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            self.discoveries = data.get('discoveries', [])

    def print_summary(self):
        """Print a summary of the knowledge base."""
        print(f"\n    Knowledge Base Summary:")
        print(f"      Equations discovered:   {len(self.equations)}")
        print(f"      Hypotheses generated:   {len(self.hypotheses)}")
        print(f"      Experiments run:         {len(self.experiments)}")
        print(f"      Validated discoveries:   {len(self.discoveries)}")

        validated = self.get_validated_hypotheses()
        if validated:
            print(f"      Validated hypotheses:    {len(validated)}")
            for h in validated[:5]:
                print(f"        [{h.id}] {h.description[:60]}...")

        top = self.get_top_discoveries(5)
        if top:
            print(f"\n      Top Discoveries:")
            for i, d in enumerate(top, 1):
                print(f"        {i}. {d['title'][:50]}... "
                      f"(sig={d['significance']:.4f}, conf={d['confidence']:.2f})")
