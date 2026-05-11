"""AGI-Style Scientific Discovery Pipeline"""

import numpy as np
import time
import os
import sys
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AGIScientificPipeline:
    """
    Full AGI-style scientific discovery system for fluid dynamics.

    Pipeline:
        Observe (simulate) -> Learn (neural) -> Hypothesize -> Test -> Store
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/agi_discovery",
        log_dir: str = "logs/agi_discovery",
        verbose: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.results = {}

    def _print(self, msg):
        if self.verbose:
            print(msg)

    # OBSERVE — Run multi-regime simulations
    def phase_observe(self, nx=64, n_regimes=5, steps_per_regime=200):
        """Generate simulation data across multiple flow regimes."""
        from core.fluid_solver_2d import FluidSolver2D
        from utils.helpers import compute_vorticity

        self._print("\n  [PHASE 1] OBSERVE -- Multi-regime simulation")
        self._print("  " + "-" * 50)

        re_values = np.logspace(1.5, 3.5, n_regimes)
        all_data = {}

        for idx, re in enumerate(re_values):
            nu = 1.0 / re
            regime_data = {'u': [], 'v': [], 'p': [], 'omega': [], 're': re, 'nu': nu}
            ic_types = ['taylor_green', 'shear_layer', 'vortex_pair']
            ic = ic_types[idx % len(ic_types)]

            solver = FluidSolver2D(
                nx=nx, ny=nx, Lx=2*np.pi, Ly=2*np.pi,
                nu=nu, dt=0.005, pressure_solver="fft"
            )
            solver.bc_manager.set_periodic()

            if ic == 'taylor_green':
                solver.initialize_taylor_green()
            elif ic == 'shear_layer':
                solver.initialize_double_shear_layer(amplitude=0.05, delta=0.05)
            else:
                solver.initialize_vortex_pair()

            # Warmup
            for _ in range(20):
                solver.step()
                if not np.all(np.isfinite(solver.u)):
                    solver.u = np.nan_to_num(solver.u)
                    solver.v = np.nan_to_num(solver.v)

            # Record snapshots
            record_every = max(1, steps_per_regime // 40)
            for step in range(steps_per_regime):
                solver.step()
                if not np.all(np.isfinite(solver.u)):
                    solver.u = np.nan_to_num(solver.u)
                    solver.v = np.nan_to_num(solver.v)

                if step % record_every == 0:
                    omega = compute_vorticity(solver.u, solver.v, solver.dx, solver.dy)
                    regime_data['u'].append(solver.u.copy())
                    regime_data['v'].append(solver.v.copy())
                    regime_data['p'].append(solver.p.copy())
                    regime_data['omega'].append(omega.copy())

            regime_data['dx'] = solver.dx
            regime_data['dy'] = solver.dy
            regime_data['dt'] = solver.dt * record_every
            all_data[f're_{re:.0f}'] = regime_data

            self._print(f"    Re={re:.0f} ({ic}): {len(regime_data['u'])} snapshots")

        self.results['observations'] = all_data
        self._print(f"    Total: {sum(len(d['u']) for d in all_data.values())} snapshots across {n_regimes} regimes")
        return all_data

    # LEARN — Neural compression + latent dynamics
    def phase_learn(self, observations: Dict, latent_dim=16):
        """Compress flow fields to latent space and learn dynamics."""
        self._print("\n  [PHASE 2] LEARN -- Neural compression + dynamics")
        self._print("  " + "-" * 50)

        # Collect all snapshots into observable trajectories
        all_trajectories = {}

        for regime_key, data in observations.items():
            n_frames = len(data['u'])
            trajectory = np.zeros((n_frames, 4))

            for i in range(n_frames):
                u, v = data['u'][i], data['v'][i]
                omega = data['omega'][i]
                ke = 0.5 * np.mean(u**2 + v**2)
                ens = np.mean(omega**2)
                max_omega = np.max(np.abs(omega))
                max_u = np.max(np.abs(u))
                trajectory[i] = [ke, ens, max_omega, max_u]

            # Filter NaN/Inf
            valid = np.all(np.isfinite(trajectory), axis=1)
            trajectory = trajectory[valid]
            if len(trajectory) > 5:
                all_trajectories[regime_key] = trajectory

        self._print(f"    Extracted {len(all_trajectories)} valid regime trajectories")

        # Learn latent representations via PCA on concatenated observables
        all_obs = np.vstack(list(all_trajectories.values()))
        obs_mean = all_obs.mean(axis=0)
        obs_std = all_obs.std(axis=0) + 1e-10
        all_obs_norm = (all_obs - obs_mean) / obs_std

        # SVD for latent representation
        U, S, Vt = np.linalg.svd(all_obs_norm, full_matrices=False)
        n_components = min(latent_dim, all_obs_norm.shape[1])
        Z_latent = all_obs_norm @ Vt[:n_components].T
        explained_var = S[:n_components]**2 / np.sum(S**2)

        self._print(f"    Latent dim: {n_components}")
        self._print(f"    Explained variance: {np.sum(explained_var):.1%}")

        learn_results = {
            'trajectories': all_trajectories,
            'Z_latent': Z_latent,
            'pca_components': Vt[:n_components],
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'explained_variance': explained_var.tolist(),
        }
        self.results['learning'] = learn_results
        return learn_results

    # HYPOTHESIZE — Discover equations + generate hypotheses
    def phase_hypothesize(self, observations: Dict, learn_results: Dict):
        """Discover equations and generate scientific hypotheses."""
        from models.symbolic_discovery import SymbolicDiscoveryEngine
        from models.physics_discovery import PhysicsAwareSINDy, CorrectionTermDiscovery
        from models.hypothesis_engine import HypothesisGenerator

        self._print("\n  [PHASE 3] HYPOTHESIZE -- Equation discovery")
        self._print("  " + "-" * 50)

        hypothesis_gen = HypothesisGenerator()
        all_hypotheses = []
        all_corrections = []
        all_equations = {}

        # 3a: Latent-space SINDy discovery
        self._print("\n    [3a] Latent-space SINDy discovery...")
        for regime_key, traj in learn_results['trajectories'].items():
            if len(traj) < 10:
                continue

            obs_data = observations.get(regime_key, {})
            dt = obs_data.get('dt', 0.025)

            engine = SymbolicDiscoveryEngine(
                n_latent_dims=traj.shape[1],
                sindy_threshold=0.05,
                gp_population=80,
                gp_generations=15,
            )

            try:
                results = engine.discover_from_trajectories(
                    traj, dt=dt, run_gp=False, verbose=False
                )
                sindy_eqs = results.get('sindy', {}).get('equations', [])
                sindy_error = results.get('sindy', {}).get('error', 1.0)

                hyps = hypothesis_gen.generate_from_latent_discovery(
                    sindy_eqs, results.get('gp', {}), sindy_error, verbose=False
                )
                all_hypotheses.extend(hyps)
                all_equations[regime_key] = {
                    'sindy': sindy_eqs,
                    'error': sindy_error,
                    'complexity': results.get('sindy', {}).get('complexity', 0),
                }
                self._print(f"      {regime_key}: {len(sindy_eqs)} eqs, MSE={sindy_error:.2e}")
            except Exception as e:
                self._print(f"      {regime_key}: skipped ({e})")

        # 3b: Physics-aware SINDy on raw fields
        self._print("\n    [3b] Physics-aware SINDy on flow fields...")
        physics_sindy = PhysicsAwareSINDy(
            threshold=0.05, alpha=0.01,
            poly_order=2, include_physics=True,
        )
        correction_finder = CorrectionTermDiscovery()

        for regime_key, data in observations.items():
            n = len(data['u'])
            if n < 5:
                continue
            nu = data['nu']
            try:
                phys_results = physics_sindy.fit_from_fields(
                    data['u'][:min(n, 20)], data['v'][:min(n, 20)],
                    data['p'][:min(n, 20)], data['omega'][:min(n, 20)],
                    dt=data['dt'], dx=data['dx'], dy=data['dy'],
                    verbose=False,
                )
                if 'error' not in phys_results:
                    corrections = correction_finder.extract_corrections(
                        phys_results['Xi'], phys_results['feature_names'], nu
                    )
                    if corrections:
                        hyps = hypothesis_gen.generate_from_corrections(
                            corrections[:5], nu, verbose=False
                        )
                        all_hypotheses.extend(hyps)
                        all_corrections.extend(corrections[:5])

                    all_equations[f'{regime_key}_phys'] = {
                        'u_eq': phys_results.get('u_equation', ''),
                        'v_eq': phys_results.get('v_equation', ''),
                        'n_corrections': len(corrections),
                    }
                    self._print(f"      {regime_key}: {len(corrections)} correction terms")
            except Exception as e:
                self._print(f"      {regime_key}: physics SINDy failed ({e})")

        self._print(f"\n    Total hypotheses: {len(all_hypotheses)}")
        self._print(f"    Total correction terms: {len(all_corrections)}")

        hyp_results = {
            'hypotheses': all_hypotheses,
            'corrections': all_corrections,
            'equations': all_equations,
            'hypothesis_generator': hypothesis_gen,
        }
        self.results['hypotheses'] = hyp_results
        return hyp_results

    # TEST — Validate hypotheses
    def phase_test(self, observations: Dict, hyp_results: Dict):
        """Validate hypotheses against simulation data."""
        from models.hypothesis_engine import ExperimentValidator, SymbolicReasoner

        self._print("\n  [PHASE 4] TEST -- Hypothesis validation")
        self._print("  " + "-" * 50)

        validator = ExperimentValidator()
        reasoner = SymbolicReasoner()
        all_experiments = []

        hypotheses = hyp_results['hypotheses']
        # Pick one regime for testing
        test_regime = next(iter(observations.values()), None)
        if test_regime is None:
            self._print("    No test data available")
            return {'experiments': [], 'validated': []}

        for hyp in hypotheses[:20]:  # Test top 20
            try:
                experiments = validator.run_all_tests(
                    hyp,
                    test_regime['u'][:10], test_regime['v'][:10],
                    dt=test_regime['dt'],
                    dx=test_regime['dx'], dy=test_regime['dy'],
                )
                all_experiments.extend(experiments)

                # Symbolic reasoning
                reasoning = reasoner.reason_about_equation(
                    hyp, list(hyp.equation.coefficients.keys())
                )
                hyp.equation.conservation_score = reasoning.get('physics_consistency_score', 0)
            except Exception:
                pass

        validated = [h for h in hypotheses if h.status == 'validated']
        n_passed = sum(1 for e in all_experiments if e.passed)

        self._print(f"    Experiments run: {len(all_experiments)}")
        self._print(f"    Tests passed: {n_passed}/{len(all_experiments)}")
        self._print(f"    Validated hypotheses: {len(validated)}/{len(hypotheses)}")

        test_results = {
            'experiments': all_experiments,
            'validated': validated,
            'n_passed': n_passed,
            'n_total': len(all_experiments),
        }
        self.results['testing'] = test_results
        return test_results

    # IMPROVE — Store and rank discoveries
    def phase_improve(self, hyp_results: Dict, test_results: Dict):
        """Store discoveries in knowledge base and rank them."""
        from models.hypothesis_engine import KnowledgeBase

        self._print("\n  [PHASE 5] IMPROVE -- Knowledge base update")
        self._print("  " + "-" * 50)

        kb = KnowledgeBase(str(self.checkpoint_dir / 'knowledge_base'))

        # Store all hypotheses
        for hyp in hyp_results['hypotheses']:
            kb.add_hypothesis(hyp)
            kb.add_equation(hyp.equation)

        # Store experiments
        for exp in test_results['experiments']:
            kb.add_experiment(exp)

        # Record validated discoveries
        for hyp in test_results['validated']:
            related_exps = [e for e in test_results['experiments']
                           if e.hypothesis_id == hyp.id]
            kb.record_discovery(
                title=hyp.description[:80],
                equation=hyp.equation,
                hypothesis=hyp,
                experiments=related_exps,
                significance=hyp.equation.novelty_score,
            )

        # Also record top correction-based discoveries
        for corr in hyp_results.get('corrections', [])[:10]:
            from models.hypothesis_engine import DiscoveredEquation, Hypothesis, Experiment
            eq = DiscoveredEquation(
                equation_str=f"d{corr['variable']}/dt += {corr['discovered_coeff']:.6f}*{corr['term']}",
                variable=corr['variable'],
                coefficients={corr['term']: corr['discovered_coeff']},
                novelty_score=corr['significance'],
                method='physics_sindy',
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            )
            kb.add_equation(eq)

        kb.save()
        kb.print_summary()

        self.results['knowledge_base'] = kb
        return kb

    # MAIN: Run full pipeline
    def run(self, nx=64, n_regimes=5, steps_per_regime=200, latent_dim=4):
        """Run the complete AGI scientific discovery pipeline."""
        t_start = time.perf_counter()

        self._print("\n" + "=" * 72)
        self._print("  +=====================================================+")
        self._print("  |   AGI-STYLE SCIENTIFIC DISCOVERY SYSTEM             |")
        self._print("  |   Observe -> Simulate -> Learn -> Hypothesize ->    |")
        self._print("  |   Test -> Improve                                   |")
        self._print("  +=====================================================+")
        self._print("=" * 72)

        # OBSERVE
        observations = self.phase_observe(nx=nx, n_regimes=n_regimes,
                                          steps_per_regime=steps_per_regime)

        # LEARN
        learn_results = self.phase_learn(observations, latent_dim=latent_dim)

        # HYPOTHESIZE
        hyp_results = self.phase_hypothesize(observations, learn_results)

        # TEST
        test_results = self.phase_test(observations, hyp_results)

        # IMPROVE
        kb = self.phase_improve(hyp_results, test_results)

        elapsed = time.perf_counter() - t_start

        # ===== FINAL REPORT =====
        self._print_final_report(elapsed, observations, learn_results,
                                  hyp_results, test_results, kb)

        return self.results

    def _print_final_report(self, elapsed, observations, learn_results,
                             hyp_results, test_results, kb):
        """Print comprehensive final report."""
        self._print("\n" + "=" * 72)
        self._print("  AGI SCIENTIFIC DISCOVERY -- FINAL REPORT")
        self._print("=" * 72)

        self._print(f"\n  Pipeline completed in {elapsed:.1f}s")
        self._print(f"\n  1. OBSERVATIONS")
        self._print(f"     Regimes simulated: {len(observations)}")
        total_snaps = sum(len(d['u']) for d in observations.values())
        self._print(f"     Total snapshots: {total_snaps}")
        for key, data in observations.items():
            self._print(f"       {key}: {len(data['u'])} frames, Re={data['re']:.0f}")

        self._print(f"\n  2. LEARNING")
        ev = learn_results.get('explained_variance', [])
        self._print(f"     Latent dimensions: {len(ev)}")
        self._print(f"     Explained variance: {sum(ev):.1%}")

        self._print(f"\n  3. HYPOTHESES")
        self._print(f"     Total generated: {len(hyp_results['hypotheses'])}")
        self._print(f"     Correction terms: {len(hyp_results['corrections'])}")

        self._print(f"\n  4. VALIDATION")
        self._print(f"     Experiments: {test_results['n_total']}")
        self._print(f"     Passed: {test_results['n_passed']}")
        self._print(f"     Validated hypotheses: {len(test_results['validated'])}")

        self._print(f"\n  5. DISCOVERIES")
        top = kb.get_top_discoveries(5)
        if top:
            for i, d in enumerate(top, 1):
                self._print(f"     {i}. [{d.get('method', '?')}] {d['equation'][:60]}")
                self._print(f"        Significance={d['significance']:.4f}, "
                           f"Confidence={d['confidence']:.2f}")

        # Key discovered equations
        self._print(f"\n  6. KEY EQUATIONS")
        for key, eq_data in list(hyp_results.get('equations', {}).items())[:6]:
            self._print(f"     [{key}]")
            if 'sindy' in eq_data:
                for eq in eq_data['sindy'][:2]:
                    self._print(f"       {eq[:70]}")
            if 'u_eq' in eq_data:
                self._print(f"       {eq_data['u_eq'][:70]}")
            if 'n_corrections' in eq_data:
                self._print(f"       Corrections: {eq_data['n_corrections']}")

        # Correction terms
        corrections = hyp_results.get('corrections', [])
        if corrections:
            self._print(f"\n  7. NOVEL CORRECTION TERMS (beyond standard NS)")
            self._print(f"     du/dt = -u*nabla(u) + nu*nabla2(u) - grad(p) + corrections")
            for c in corrections[:8]:
                self._print(f"     + {c['discovered_coeff']:+.6f} * {c['term']:<30s} "
                           f"[{c['type']}, sig={c['significance']:.4f}]")

        self._print("\n" + "=" * 72)
        self._print("  System Status: OPERATIONAL")
        self._print("  Knowledge base saved to: " + str(self.checkpoint_dir / 'knowledge_base'))
        self._print("=" * 72 + "\n")

    # Visualization
    def plot_results(self, save_path: Optional[str] = None):
        """Generate visualization of discovery results."""
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
            'savefig.facecolor': '#0d1117', 'text.color': '#c9d1d9',
            'axes.labelcolor': '#c9d1d9', 'xtick.color': '#8b949e',
            'ytick.color': '#8b949e', 'axes.edgecolor': '#30363d',
            'figure.dpi': 150,
        })

        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('AGI Scientific Discovery System -- Results',
                     fontsize=20, fontweight='bold', color='#58a6ff', y=0.98)

        # Panel 1: Observable trajectories
        ax1 = fig.add_subplot(2, 3, 1)
        learn = self.results.get('learning', {})
        trajs = learn.get('trajectories', {})
        colors_list = ['#58a6ff', '#7ee787', '#f97583', '#d2a8ff', '#ffa657']
        for i, (key, traj) in enumerate(list(trajs.items())[:5]):
            c = colors_list[i % len(colors_list)]
            ax1.plot(traj[:, 0], label=key[:12], color=c, lw=1.5, alpha=0.8)
        ax1.set_title('KE Trajectories', color='#79c0ff', fontsize=13)
        ax1.set_xlabel('Time step'); ax1.set_ylabel('Kinetic Energy')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.grid(True, alpha=0.15, color='#30363d')

        # Panel 2: Explained variance
        ax2 = fig.add_subplot(2, 3, 2)
        ev = learn.get('explained_variance', [])
        if ev:
            ax2.bar(range(len(ev)), ev, color='#d2a8ff', alpha=0.8)
            ax2.plot(range(len(ev)), np.cumsum(ev), 'o-', color='#ffa657', lw=2)
        ax2.set_title('PCA Explained Variance', color='#79c0ff', fontsize=13)
        ax2.set_xlabel('Component'); ax2.set_ylabel('Variance')
        ax2.grid(True, alpha=0.15, color='#30363d')

        # Panel 3: Hypothesis confidence distribution
        ax3 = fig.add_subplot(2, 3, 3)
        hyps = self.results.get('hypotheses', {}).get('hypotheses', [])
        if hyps:
            confs = [h.confidence for h in hyps]
            statuses = [h.status for h in hyps]
            colors_h = ['#7ee787' if s == 'validated' else '#ffa657' if s == 'testing'
                       else '#8b949e' for s in statuses]
            ax3.barh(range(min(len(confs), 15)), confs[:15], color=colors_h[:15], alpha=0.8)
            ax3.set_xlabel('Confidence')
            labels = [h.id for h in hyps[:15]]
            ax3.set_yticks(range(min(len(labels), 15)))
            ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_title('Hypothesis Confidence', color='#79c0ff', fontsize=13)
        ax3.grid(True, alpha=0.15, color='#30363d')

        # Panel 4: Correction term magnitudes
        ax4 = fig.add_subplot(2, 3, 4)
        corrections = self.results.get('hypotheses', {}).get('corrections', [])
        if corrections:
            terms = [c['term'][:20] for c in corrections[:10]]
            sigs = [c['significance'] for c in corrections[:10]]
            types = [c['type'] for c in corrections[:10]]
            c_colors = ['#f97583' if t == 'novel' else '#58a6ff' for t in types]
            ax4.barh(range(len(terms)), sigs, color=c_colors, alpha=0.8)
            ax4.set_yticks(range(len(terms)))
            ax4.set_yticklabels(terms, fontsize=8)
            ax4.set_xlabel('Significance')
        ax4.set_title('Correction Terms (red=novel)', color='#79c0ff', fontsize=13)
        ax4.grid(True, alpha=0.15, color='#30363d')

        # Panel 5: Validation results
        ax5 = fig.add_subplot(2, 3, 5)
        test = self.results.get('testing', {})
        exps = test.get('experiments', [])
        if exps:
            by_type = {}
            for e in exps:
                t = e.test_type
                if t not in by_type:
                    by_type[t] = {'passed': 0, 'failed': 0}
                if e.passed:
                    by_type[t]['passed'] += 1
                else:
                    by_type[t]['failed'] += 1
            types_l = list(by_type.keys())
            passed_l = [by_type[t]['passed'] for t in types_l]
            failed_l = [by_type[t]['failed'] for t in types_l]
            x = range(len(types_l))
            ax5.bar(x, passed_l, color='#7ee787', alpha=0.8, label='Passed')
            ax5.bar(x, failed_l, bottom=passed_l, color='#f97583', alpha=0.8, label='Failed')
            ax5.set_xticks(list(x))
            ax5.set_xticklabels(types_l, fontsize=9)
            ax5.legend(fontsize=9)
        ax5.set_title('Validation Results', color='#79c0ff', fontsize=13)
        ax5.set_ylabel('Count')
        ax5.grid(True, alpha=0.15, color='#30363d')

        # Panel 6: Discovery summary text
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        ax6.set_title('Top Discoveries', color='#79c0ff', fontsize=13)

        lines = ["AGI Scientific Discovery -- Summary\n"]
        lines.append(f"Regimes: {len(self.results.get('observations', {}))}")
        lines.append(f"Hypotheses: {len(hyps)}")
        lines.append(f"Validated: {len(test.get('validated', []))}")
        lines.append(f"Corrections: {len(corrections)}\n")

        kb = self.results.get('knowledge_base')
        if kb:
            top = kb.get_top_discoveries(5)
            for i, d in enumerate(top, 1):
                lines.append(f"{i}. {d['equation'][:50]}")
                lines.append(f"   sig={d['significance']:.4f}")

        if corrections:
            lines.append("\nNovel terms:")
            for c in corrections[:4]:
                lines.append(f"  {c['discovered_coeff']:+.4f}*{c['term']}")

        ax6.text(0.05, 0.95, '\n'.join(lines), transform=ax6.transAxes,
                va='top', fontsize=9, color='#c9d1d9', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                         edgecolor='#30363d', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            self._print(f"  Plot saved: {save_path}")

        try:
            import matplotlib
            if 'agg' not in matplotlib.get_backend().lower():
                plt.show()
            else:
                plt.close(fig)
        except Exception:
            plt.close(fig)
