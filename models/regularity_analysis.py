"""Blow-up Detection & Regularity Analysis AI"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from enum import Enum


class FlowRegime(Enum):
    """Classification of flow regularity state."""
    SMOOTH = "smooth"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"
    UNSTABLE = "unstable"
    SINGULAR_RISK = "singular_risk"


class FlowDiagnostics:
    """
    Comprehensive flow diagnostics for regularity analysis.
    
    Implements multiple criteria for detecting blow-up precursors:
    
    1. BKM Criterion: ∫₀ᵀ ‖ω(·,t)‖_∞ dt < ∞  ⟹  smooth
       (Beale-Kato-Majda, 1984)
    
    2. Enstrophy production: dE/dt = -2ν ∫|∇ω|² dx + 2 ∫ω·S·ω dx
       If production > dissipation ⟹ potential instability
    
    3. Energy spectrum: E(k) ~ k^{-5/3} (Kolmogorov)
       Deviation from Kolmogorov scaling indicates anomalies
    
    4. Strain-vorticity alignment:
       cos(θ) between ω and eigenvectors of S
    """
    
    @staticmethod
    def compute_bkm_criterion(
        omega_history: List[np.ndarray],
        dt: float,
    ) -> Dict[str, float]:
        """
        Beale-Kato-Majda blow-up criterion monitoring.
        
        If ∫₀ᵀ ‖ω‖_∞ dt → ∞, solution may blow up.
        
        Returns:
            dict with bkm_integral, max_vorticity, growth_rate
        """
        max_omegas = [np.max(np.abs(omega)) for omega in omega_history]
        
        # Numerical integration (trapezoidal)
        bkm_integral = np.trapz(max_omegas, dx=dt)
        
        # Growth rate (finite difference of log)
        if len(max_omegas) > 2:
            log_omega = np.log(np.array(max_omegas) + 1e-30)
            growth_rates = np.diff(log_omega) / dt
            max_growth = float(np.max(growth_rates))
        else:
            max_growth = 0.0
        
        return {
            'bkm_integral': float(bkm_integral),
            'max_vorticity': float(max(max_omegas)),
            'max_vorticity_growth_rate': max_growth,
            'final_max_omega': float(max_omegas[-1]) if max_omegas else 0.0,
            'is_growing': max_growth > 0,
        }
    
    @staticmethod
    def compute_enstrophy_budget(
        u: np.ndarray, v: np.ndarray,
        omega: np.ndarray,
        dx: float, dy: float,
        nu: float,
    ) -> Dict[str, float]:
        """
        Enstrophy production and dissipation rates.
        
        dΩ/dt = Production - Dissipation
        
        Production = 2 ∫ ωᵢ Sᵢⱼ ωⱼ dx  (vortex stretching in 3D, ~0 in 2D)
        Dissipation = 2ν ∫ |∇ω|² dx     (always positive)
        
        In 2D, enstrophy is conserved in inviscid case, so:
        dΩ/dt = -2ν ∫ |∇ω|² dx  (pure dissipation)
        """
        # Enstrophy
        enstrophy = 0.5 * np.mean(omega ** 2)
        
        # Palinstrophy (measures enstrophy dissipation in 2D)
        domega_dx = np.gradient(omega, dx, axis=1)
        domega_dy = np.gradient(omega, dy, axis=0)
        palinstrophy = 0.5 * np.mean(domega_dx**2 + domega_dy**2)
        
        # Enstrophy dissipation rate
        dissipation = 2 * nu * palinstrophy
        
        # Strain rate tensor
        dudx = np.gradient(u, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        dvdy = np.gradient(v, dy, axis=0)
        
        S11 = dudx
        S22 = dvdy
        S12 = 0.5 * (dudy + dvdx)
        
        # Strain rate magnitude
        strain_mag = np.sqrt(2 * (S11**2 + S22**2 + 2*S12**2) + 1e-10)
        
        # Vortex stretching proxy (3D-like measure for 2D)
        # In 2D this is actually the vorticity-strain alignment
        alignment = omega * (S11 + S22)
        stretch_proxy = np.mean(np.abs(alignment))
        
        return {
            'enstrophy': float(enstrophy),
            'palinstrophy': float(palinstrophy),
            'dissipation_rate': float(dissipation),
            'strain_magnitude': float(np.mean(strain_mag)),
            'vortex_stretch_proxy': float(stretch_proxy),
            'dissipation_ratio': float(dissipation / max(enstrophy, 1e-10)),
        }
    
    @staticmethod
    def compute_energy_spectrum(
        u: np.ndarray, v: np.ndarray,
        Lx: float = 2*np.pi, Ly: float = 2*np.pi,
    ) -> Dict[str, np.ndarray]:
        """
        Compute 1D kinetic energy spectrum E(k).
        
        E(k) = ½ ∫ |û(k)|² dk_shell
        
        For Kolmogorov turbulence: E(k) ~ k^{-5/3}
        Deviation indicates non-equilibrium or anomalous dynamics.
        """
        ny, nx = u.shape
        
        # FFT of velocity components
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        # Energy density in Fourier space
        energy_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
        
        # Wavenumber magnitudes
        kx = np.fft.fftfreq(nx, d=Lx/nx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=Ly/ny) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K_mag = np.sqrt(KX**2 + KY**2)
        
        # Bin into shells
        k_max = int(np.sqrt(nx**2 + ny**2) / 2)
        dk = 1.0
        k_bins = np.arange(1, k_max + 1)
        spectrum = np.zeros(k_max)
        
        for i, k in enumerate(k_bins):
            shell_mask = (K_mag >= k - dk/2) & (K_mag < k + dk/2)
            spectrum[i] = np.sum(energy_density[shell_mask])
        
        # Kolmogorov reference: E(k) = C * k^(-5/3)
        valid = spectrum > 0
        if np.any(valid):
            C_k = np.median(spectrum[valid] * k_bins[valid]**(5/3))
            kolmogorov_ref = C_k * k_bins**(-5/3)
        else:
            kolmogorov_ref = np.zeros_like(k_bins, dtype=float)
        
        return {
            'wavenumbers': k_bins,
            'spectrum': spectrum,
            'kolmogorov_reference': kolmogorov_ref,
            'total_energy': float(np.sum(spectrum)),
        }
    
    @staticmethod
    def classify_regime(
        diagnostics: Dict[str, float],
        spectrum: Dict[str, np.ndarray],
    ) -> FlowRegime:
        """
        Classify flow regime based on diagnostics.
        
        Rules:
            - If BKM integral growing super-linearly → SINGULAR_RISK
            - If max vorticity growth > 10/s → UNSTABLE
            - If enstrophy ratio high → TURBULENT
            - If spectrum follows Kolmogorov → TURBULENT
            - If low Re-like behavior → SMOOTH
        """
        max_growth = diagnostics.get('max_vorticity_growth_rate', 0)
        dissipation_ratio = diagnostics.get('dissipation_ratio', 0)
        enstrophy = diagnostics.get('enstrophy', 0)
        
        if max_growth > 50:
            return FlowRegime.SINGULAR_RISK
        elif max_growth > 10:
            return FlowRegime.UNSTABLE
        elif dissipation_ratio > 0.5 or enstrophy > 100:
            return FlowRegime.TURBULENT
        elif enstrophy > 10:
            return FlowRegime.TRANSITIONAL
        else:
            return FlowRegime.SMOOTH


class BlowupDetector(nn.Module):
    """
    Neural network for predicting solution blow-up probability.
    
    Input: Velocity field (u, v) at time t₀
    Output: P(blow-up within T) ∈ [0, 1]
    
    Architecture: CNN feature extractor → MLP classifier
    
    Training:
        - Positive examples: ICs that lead to numerical instability
        - Negative examples: ICs that remain smooth
        - Uses CFD solver to generate labels
    """
    
    def __init__(
        self,
        input_channels: int = 4,   # (u, v, ω, |S|)
        input_size: int = 128,
        n_classes: int = 5,        # FlowRegime enum
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
        
        # Blow-up probability head (binary)
        self.blowup_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) flow field channels
        
        Returns:
            dict with 'regime_logits', 'blowup_prob', 'features'
        """
        feat = self.features(x)
        
        regime_logits = self.classifier(feat)
        blowup_prob = self.blowup_head(feat)
        
        return {
            'regime_logits': regime_logits,
            'blowup_prob': blowup_prob,
            'features': feat,
        }
    
    def predict_regime(self, x: torch.Tensor) -> Tuple[str, float]:
        """Predict flow regime and blow-up probability."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            regime_idx = torch.argmax(output['regime_logits'], dim=-1).item()
            regimes = list(FlowRegime)
            regime = regimes[regime_idx % len(regimes)]
            
            blowup_p = output['blowup_prob'].item()
        
        return regime.value, blowup_p


class StabilityAnalyzer:
    """
    Maps initial conditions → long-term behavior.
    
    Instead of proving "all solutions are smooth",
    we aim to "predict when and why they fail".
    
    Produces:
        - Empirical regularity maps
        - Failure region prediction
        - Reynolds number dependence curves
        - Critical exponent estimates
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_ic_space(
        self,
        solver_factory,
        n_samples: int = 50,
        re_range: Tuple[float, float] = (10, 10000),
        n_steps: int = 500,
        verbose: bool = True,
    ) -> Dict:
        """
        Sweep initial conditions and Reynolds numbers to build stability map.
        
        Args:
            solver_factory: callable that creates a solver given (Re, ic_type)
            n_samples: number of IC samples
            re_range: Reynolds number range to sweep
        
        Returns:
            Stability map data
        """
        re_values = np.logspace(np.log10(re_range[0]), np.log10(re_range[1]), n_samples)
        ic_types = ['taylor_green', 'shear_layer', 'vortex_pair']
        
        results = {
            'reynolds': [],
            'ic_type': [],
            'max_vorticity_final': [],
            'kinetic_energy_ratio': [],
            'regime': [],
            'survived': [],
            'bkm_integral': [],
        }
        
        for ic_type in ic_types:
            for re in re_values:
                try:
                    nu = 1.0 / re
                    solver = solver_factory(nu=nu, ic_type=ic_type)
                    
                    # Record initial state
                    ke_initial = 0.5 * np.mean(solver.u**2 + solver.v**2)
                    omega_history = []
                    
                    # Run simulation
                    survived = True
                    for step in range(n_steps):
                        solver.step()
                        
                        if step % 10 == 0:
                            omega = solver.get_vorticity()
                            omega_history.append(omega)
                        
                        # Check for blow-up
                        if not np.all(np.isfinite(solver.u)):
                            survived = False
                            break
                    
                    # Compute diagnostics
                    ke_final = 0.5 * np.mean(solver.u**2 + solver.v**2)
                    
                    if omega_history:
                        bkm = FlowDiagnostics.compute_bkm_criterion(
                            omega_history, dt=solver.dt * 10
                        )
                        max_omega = bkm['max_vorticity']
                        bkm_int = bkm['bkm_integral']
                    else:
                        max_omega = 0.0
                        bkm_int = 0.0
                    
                    # Classify
                    if not survived:
                        regime = FlowRegime.SINGULAR_RISK.value
                    elif max_omega > 100:
                        regime = FlowRegime.UNSTABLE.value
                    elif max_omega > 10:
                        regime = FlowRegime.TURBULENT.value
                    elif max_omega > 1:
                        regime = FlowRegime.TRANSITIONAL.value
                    else:
                        regime = FlowRegime.SMOOTH.value
                    
                    results['reynolds'].append(re)
                    results['ic_type'].append(ic_type)
                    results['max_vorticity_final'].append(max_omega)
                    results['kinetic_energy_ratio'].append(
                        ke_final / max(ke_initial, 1e-10)
                    )
                    results['regime'].append(regime)
                    results['survived'].append(survived)
                    results['bkm_integral'].append(bkm_int)
                    
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Re={re:.0f}, IC={ic_type} failed: {e}")
                    results['reynolds'].append(re)
                    results['ic_type'].append(ic_type)
                    results['max_vorticity_final'].append(float('nan'))
                    results['kinetic_energy_ratio'].append(float('nan'))
                    results['regime'].append(FlowRegime.SINGULAR_RISK.value)
                    results['survived'].append(False)
                    results['bkm_integral'].append(float('nan'))
            
            if verbose:
                survived_count = sum(1 for i, ic in enumerate(results['ic_type']) 
                                     if ic == ic_type and results['survived'][i])
                total = sum(1 for ic in results['ic_type'] if ic == ic_type)
                print(f"  {ic_type}: {survived_count}/{total} survived")
        
        self.results = results
        return results
    
    def get_stability_summary(self) -> Dict:
        """Summarize stability analysis results."""
        if not self.results:
            return {}
        
        summary = {
            'total_simulations': len(self.results['reynolds']),
            'survived': sum(self.results['survived']),
            'blew_up': sum(not s for s in self.results['survived']),
        }
        
        # Critical Reynolds number estimate (where instability starts)
        for ic_type in set(self.results['ic_type']):
            mask = [ic == ic_type for ic in self.results['ic_type']]
            re_vals = [self.results['reynolds'][i] for i, m in enumerate(mask) if m]
            survived = [self.results['survived'][i] for i, m in enumerate(mask) if m]
            
            # Find approximate critical Re
            failed_re = [re for re, s in zip(re_vals, survived) if not s]
            if failed_re:
                summary[f'critical_re_{ic_type}'] = min(failed_re)
            else:
                summary[f'critical_re_{ic_type}'] = float('inf')
        
        # Regime distribution
        regime_counts = {}
        for r in self.results['regime']:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        summary['regime_distribution'] = regime_counts
        
        return summary


class TurbulenceMetrics:
    """
    Comprehensive metrics for comparing turbulence models.
    
    Required experiments:
        1. Compare with DNS
        2. Compare with LES
        3. Generalization tests
    
    Metrics:
        - L2 error (velocity)
        - Energy spectrum error
        - Vorticity accuracy
        - Structure function error
        - Correlation time
    """
    
    @staticmethod
    def l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
        """Relative L2 error: ‖pred - truth‖₂ / ‖truth‖₂"""
        return float(
            np.sqrt(np.sum((pred - truth)**2)) / 
            (np.sqrt(np.sum(truth**2)) + 1e-10)
        )
    
    @staticmethod
    def energy_spectrum_error(
        u_pred: np.ndarray, v_pred: np.ndarray,
        u_truth: np.ndarray, v_truth: np.ndarray,
    ) -> float:
        """
        Error in energy spectrum E(k).
        
        Measures how well the model reproduces the energy distribution
        across scales (Kolmogorov cascade).
        """
        spec_pred = FlowDiagnostics.compute_energy_spectrum(u_pred, v_pred)
        spec_truth = FlowDiagnostics.compute_energy_spectrum(u_truth, v_truth)
        
        E_pred = spec_pred['spectrum']
        E_truth = spec_truth['spectrum']
        
        # Match lengths
        n = min(len(E_pred), len(E_truth))
        E_pred = E_pred[:n]
        E_truth = E_truth[:n]
        
        # Relative error in log-space (better for power-law spectra)
        mask = (E_truth > 1e-20) & (E_pred > 1e-20)
        if np.any(mask):
            log_error = np.mean(
                (np.log10(E_pred[mask]) - np.log10(E_truth[mask]))**2
            )
            return float(np.sqrt(log_error))
        return float('inf')
    
    @staticmethod
    def vorticity_accuracy(
        omega_pred: np.ndarray,
        omega_truth: np.ndarray,
    ) -> Dict[str, float]:
        """
        Vorticity-based accuracy metrics.
        
        Returns:
            - l2_vort: relative L2 error of vorticity
            - peak_vort_error: error in max vorticity
            - correlation: spatial correlation coefficient
        """
        l2 = float(
            np.sqrt(np.sum((omega_pred - omega_truth)**2)) / 
            (np.sqrt(np.sum(omega_truth**2)) + 1e-10)
        )
        
        peak_error = float(
            abs(np.max(np.abs(omega_pred)) - np.max(np.abs(omega_truth))) /
            (np.max(np.abs(omega_truth)) + 1e-10)
        )
        
        # Spatial correlation
        op_flat = omega_pred.flatten()
        ot_flat = omega_truth.flatten()
        correlation = float(
            np.corrcoef(op_flat, ot_flat)[0, 1]
        ) if np.std(op_flat) > 1e-10 and np.std(ot_flat) > 1e-10 else 0.0
        
        return {
            'l2_vorticity': l2,
            'peak_vorticity_error': peak_error,
            'spatial_correlation': correlation,
        }
    
    @staticmethod
    def structure_functions(
        u: np.ndarray, v: np.ndarray,
        orders: List[int] = [2, 3, 4],
        n_separations: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Compute velocity structure functions S_n(r).
        
        S_n(r) = <|δu(r)|^n>  where δu = u(x+r) - u(x)
        
        Kolmogorov theory predicts:
            S_2(r) ~ r^(2/3)  (inertial range)
            S_3(r) ~ -4/5 εr  (Kolmogorov 4/5 law)
        """
        ny, nx = u.shape
        max_sep = min(nx, ny) // 4
        separations = np.linspace(1, max_sep, n_separations, dtype=int)
        
        results = {'separations': separations}
        
        for n in orders:
            S_n = np.zeros(len(separations))
            for i, r in enumerate(separations):
                # Longitudinal structure function (x-direction)
                du = np.roll(u, -r, axis=1) - u
                S_n[i] = np.mean(np.abs(du)**n)
            results[f'S{n}'] = S_n
        
        return results
    
    @staticmethod
    def compute_all_metrics(
        u_pred: np.ndarray, v_pred: np.ndarray,
        u_truth: np.ndarray, v_truth: np.ndarray,
        dx: float, dy: float,
    ) -> Dict[str, float]:
        """Compute comprehensive comparison metrics."""
        from utils.helpers import compute_vorticity
        
        omega_pred = compute_vorticity(u_pred, v_pred, dx, dy)
        omega_truth = compute_vorticity(u_truth, v_truth, dx, dy)
        
        metrics = {
            'l2_velocity': TurbulenceMetrics.l2_error(
                np.stack([u_pred, v_pred]),
                np.stack([u_truth, v_truth])
            ),
            'energy_spectrum_error': TurbulenceMetrics.energy_spectrum_error(
                u_pred, v_pred, u_truth, v_truth
            ),
        }
        
        vort_metrics = TurbulenceMetrics.vorticity_accuracy(omega_pred, omega_truth)
        metrics.update(vort_metrics)
        
        # Energy metrics
        ke_pred = 0.5 * np.mean(u_pred**2 + v_pred**2)
        ke_truth = 0.5 * np.mean(u_truth**2 + v_truth**2)
        metrics['kinetic_energy_error'] = abs(ke_pred - ke_truth) / (ke_truth + 1e-10)
        
        return metrics
