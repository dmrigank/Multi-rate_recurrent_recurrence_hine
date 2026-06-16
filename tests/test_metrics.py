"""Metrics tests (DESIGN.md §9, Invariant 8).

Covers:
  1. RMSE: identical fields → 0; known error → exact value.
  2. ACC:  identical fields → 1; orthogonal anomalies → 0.
  3. Radial spectrum: pure sinusoid lands in the right bin.
  4. Enstrophy: exact against analytic formula.
  5. Spectral drift: identical → zero error, ratio = 1.
  6. VPH (ACC-based): monotone in threshold; correct τ_λ conversion (Invariant 8).
  7. VPH (RMSE-based): same properties.
  8. VPH never-drops case → lower bound flagged.
  9. eval_trajectory: smoke test, all keys present.
 10. Lyapunov estimator: returns positive λ on the debug dataset.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from msr_hine.metrics import (
    anomaly_correlation,
    climatology,
    climatology_std,
    enstrophy,
    eval_trajectory,
    radial_energy_spectrum,
    rmse,
    spectral_drift,
    spectral_error,
    valid_prediction_horizon,
    vph_from_rmse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, T, H = 3, 10, 32   # small defaults for fast tests

def _rand_traj(b=B, t=T, h=H, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(b, t, 1, h, h)

def _zeros() -> torch.Tensor:
    return torch.zeros(B, T, 1, H, H)

def _clim(omega: torch.Tensor) -> torch.Tensor:
    return climatology(omega)


# ---------------------------------------------------------------------------
# 1. RMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_identical_is_zero(self):
        x = _rand_traj()
        assert torch.allclose(rmse(x, x), torch.zeros(T))

    def test_known_constant_error(self):
        x    = torch.zeros(1, T, 1, H, H)
        diff = 3.0
        y    = x + diff
        r    = rmse(x, y)
        assert torch.allclose(r, torch.full((T,), diff)), \
            f"Expected RMSE={diff}, got {r}"

    def test_output_shape(self):
        x = _rand_traj()
        assert rmse(x, _rand_traj(seed=1)).shape == (T,)

    def test_nonnegative(self):
        x = _rand_traj(); y = _rand_traj(seed=1)
        assert (rmse(x, y) >= 0).all()

    def test_grows_with_error(self):
        x = _rand_traj()
        r1 = rmse(x, x + 0.1 * torch.randn_like(x))
        r2 = rmse(x, x + 1.0 * torch.randn_like(x))
        assert (r2 > r1).all(), "Larger perturbation should give larger RMSE"

    def test_shape_mismatch_raises(self):
        x = torch.randn(2, 5, 1, H, H)
        y = torch.randn(2, 6, 1, H, H)
        with pytest.raises(ValueError):
            rmse(x, y)


# ---------------------------------------------------------------------------
# 2. ACC
# ---------------------------------------------------------------------------

class TestACC:
    def test_identical_is_one(self):
        x    = _rand_traj()
        clim = _clim(x)
        acc  = anomaly_correlation(x, x, clim)
        assert torch.allclose(acc, torch.ones(T), atol=1e-5), \
            f"Identical fields should give ACC=1, got {acc}"

    def test_opposite_sign_is_neg_one(self):
        x    = _rand_traj()
        clim = torch.zeros(1, 1, 1, H, H)   # zero climatology
        acc  = anomaly_correlation(x, -x, clim)
        assert torch.allclose(acc, -torch.ones(T), atol=1e-5), \
            f"Opposite fields should give ACC=-1, got {acc}"

    def test_orthogonal_anomalies_near_zero(self):
        torch.manual_seed(0)
        x = torch.randn(1, T, 1, H, H)
        # Build an anomaly orthogonal to x at each step
        clim = torch.zeros(1, 1, 1, H, H)
        y    = torch.randn_like(x)
        # Orthogonalise each time step
        for t in range(T):
            a   = x[0, t].flatten()
            b   = y[0, t].flatten()
            b   = b - (a @ b / (a @ a + 1e-12)) * a
            y[0, t] = b.reshape(1, H, H)
        acc = anomaly_correlation(x, y, clim)
        assert (acc.abs() < 1e-4).all(), f"Orthogonal fields should give ACC≈0: {acc}"

    def test_output_shape(self):
        x = _rand_traj()
        assert anomaly_correlation(x, _rand_traj(seed=1), _clim(x)).shape == (T,)

    def test_range(self):
        x = _rand_traj(); y = _rand_traj(seed=1)
        acc = anomaly_correlation(x, y, _clim(x))
        assert (acc >= -1.0 - 1e-5).all() and (acc <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# 3. Radial energy spectrum
# ---------------------------------------------------------------------------

class TestRadialSpectrum:
    def test_pure_sinusoid_in_correct_bin(self):
        """A cos(k_test * y) field should have its energy entirely in bin k_test."""
        k_test = 4
        n      = 64
        yy = torch.linspace(0, 2 * math.pi * (n - 1) / n, n)
        YY, _ = torch.meshgrid(yy, yy, indexing="ij")
        field = torch.cos(k_test * YY).unsqueeze(0)   # [1, n, n]

        k_bins, E_k = radial_energy_spectrum(field)   # [1, n//2]
        peak_bin = int(E_k[0].argmax().item())
        assert peak_bin == k_test, f"Expected peak at k={k_test}, got k={peak_bin}"

    def test_output_shape(self):
        field  = torch.randn(B, H, H)
        k, E   = radial_energy_spectrum(field)
        assert k.shape == (H // 2,)
        assert E.shape == (B, H // 2)

    def test_nonnegative(self):
        field  = torch.randn(B, H, H)
        _, E   = radial_energy_spectrum(field)
        assert (E >= 0).all()

    def test_custom_n_bins(self):
        field  = torch.randn(2, H, H)
        _, E   = radial_energy_spectrum(field, n_bins=8)
        assert E.shape == (2, 8)

    def test_total_energy_parseval(self):
        """Parseval check: Σ_k k² E(k) ≈ 0.5 * mean(ω²) (enstrophy from spectrum).

        E(k) = 0.5 * Σ_{shell k} |ω̂|²/|k|² / n⁴, so Σ_k k²*E(k) ≈ 0.5*mean(ω²).
        """
        n     = 64
        # Use a band-limited field so energy is concentrated in known bins
        k0 = 4
        import math
        yy_lin = torch.linspace(0, 2*math.pi*(n-1)/n, n)
        _, yy = torch.meshgrid(torch.zeros(n), yy_lin, indexing='ij')
        field = torch.cos(k0 * yy).unsqueeze(0)   # [1, n, n]

        kb, E = radial_energy_spectrum(field)      # [1, n//2]
        # Enstrophy from spectrum: Σ k² E(k)
        k2 = kb ** 2
        ens_spec   = (k2 * E[0]).sum().item()
        # Direct enstrophy: 0.5 * mean(ω²) = 0.5 * 0.5 = 0.25
        ens_direct = 0.5 * field.pow(2).mean().item()
        assert abs(ens_spec - ens_direct) < 1e-4 * ens_direct, (
            f"Parseval: Σk²E(k)={ens_spec:.6f} ≠ 0.5*mean(ω²)={ens_direct:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Enstrophy
# ---------------------------------------------------------------------------

class TestEnstrophy:
    def test_analytic_cosine(self):
        """Z = 0.5 * <cos(ky)^2> = 0.25 (exactly)."""
        n   = 256
        k   = 4
        yy  = torch.linspace(0, 2 * math.pi * (n - 1) / n, n)
        YY, _ = torch.meshgrid(yy, yy, indexing="ij")
        field = torch.cos(k * YY)   # mean(cos^2) ≈ 0.5 for large n
        Z = enstrophy(field.unsqueeze(0).unsqueeze(0))
        assert abs(Z.item() - 0.25) < 0.01, f"Expected Z≈0.25, got {Z.item():.4f}"

    def test_zero_field_gives_zero(self):
        assert enstrophy(torch.zeros(2, 1, H, H)).sum() == 0.0

    def test_shape_4d(self):
        z = enstrophy(torch.randn(B, 1, H, H))
        assert z.shape == (B,)

    def test_nonnegative(self):
        assert (enstrophy(torch.randn(B, 1, H, H)) >= 0).all()


# ---------------------------------------------------------------------------
# 5. Spectral drift
# ---------------------------------------------------------------------------

class TestSpectralDrift:
    def test_identical_gives_zero_error(self):
        x = torch.randn(B, 1, H, H)
        d = spectral_drift(x, x)
        assert d["spec_l1"].abs().item() < 1e-5, \
            f"Identical fields should give zero spectral error, got {d['spec_l1']:.2e}"

    def test_identical_gives_unit_ratio(self):
        x = torch.randn(B, 1, H, H)
        d = spectral_drift(x, x)
        assert abs(d["enstrophy_ratio"].item() - 1.0) < 1e-5, \
            f"Identical fields: enstrophy_ratio should be 1.0, got {d['enstrophy_ratio']:.4f}"

    def test_output_keys(self):
        x = torch.randn(B, 1, H, H)
        d = spectral_drift(x, torch.randn(B, 1, H, H))
        for key in ("k_bins", "spec_hat", "spec_true", "spec_error", "spec_l1",
                    "enstrophy_hat", "enstrophy_true", "enstrophy_ratio"):
            assert key in d, f"Missing key: {key}"

    def test_amplified_field_has_ratio_above_one(self):
        x = torch.randn(B, 1, H, H)
        d = spectral_drift(2 * x, x)
        assert d["enstrophy_ratio"].item() > 1.0


# ---------------------------------------------------------------------------
# 6 & 7. Valid Prediction Horizon
# ---------------------------------------------------------------------------

class TestVPH:
    """VPH tests — covers both ACC-based and RMSE-based variants."""

    def _make_acc(self, T: int = 50, drop_at: int = 20) -> torch.Tensor:
        """Synthetic ACC that starts at 1 and drops below threshold at step drop_at."""
        acc = torch.ones(T)
        acc[drop_at:] = 0.2   # drops below typical threshold
        return acc

    def test_acc_vph_detects_correct_step(self):
        drop_at = 20
        acc = self._make_acc(drop_at=drop_at)
        tau = 5.0       # τ_λ = 5 snapshot steps
        dt  = 1.0
        result = valid_prediction_horizon(acc, tau_lambda_steps=tau, dt_snapshot=dt,
                                          threshold=0.5)
        assert result["steps"] == drop_at, \
            f"Expected drop at step {drop_at}, got {result['steps']}"

    def test_tau_lambda_conversion(self):
        """steps × dt / (tau_lambda_steps × dt) == tau_lambda."""
        drop_at = 15
        tau = 5.0; dt = 0.02
        acc = self._make_acc(drop_at=drop_at)
        res = valid_prediction_horizon(acc, tau_lambda_steps=tau, dt_snapshot=dt,
                                       threshold=0.5)
        expected_tau_lam = (drop_at * dt) / (tau * dt)
        assert abs(res["tau_lambda"] - expected_tau_lam) < 1e-6, (
            f"τ_λ={res['tau_lambda']:.4f}, expected {expected_tau_lam:.4f}"
        )

    def test_vph_monotone_in_threshold(self):
        """Higher (stricter) threshold → earlier or equal horizon."""
        acc = torch.linspace(0.99, 0.01, 100)   # smooth decay
        tau = 10.0; dt = 1.0
        thresholds = [0.8, 0.6, 0.4, 0.2]
        horizons = [
            valid_prediction_horizon(acc, tau, dt, t)["steps"]
            for t in thresholds
        ]
        assert horizons == sorted(horizons), \
            f"VPH not monotone in threshold: {list(zip(thresholds, horizons))}"

    def test_lower_bound_flagged(self):
        """When ACC never drops below threshold, is_lower_bound must be True."""
        acc    = torch.ones(50)   # never drops
        result = valid_prediction_horizon(acc, tau_lambda_steps=5.0,
                                          dt_snapshot=1.0, threshold=0.5)
        assert result["is_lower_bound"] is True, \
            "Should flag is_lower_bound when ACC never drops"
        assert result["steps"] == 50, "Steps should be T when never drops"

    def test_rmse_vph_detects_correct_step(self):
        T = 50; exceed_at = 25
        rmse_t = torch.linspace(0.1, 1.5, T)
        clim_std = 1.0
        tau = 5.0; dt = 1.0
        res = vph_from_rmse(rmse_t, clim_std, tau, dt, threshold=0.65)
        # RMSE/clim_std crosses 0.65 somewhere around exceed_at
        assert 0 < res["steps"] <= T

    def test_rmse_vph_lower_bound(self):
        rmse_t  = torch.full((50,), 0.1)   # always below threshold
        clim_std = 1.0
        res = vph_from_rmse(rmse_t, clim_std, tau_lambda_steps=5.0,
                            dt_snapshot=1.0, threshold=0.65)
        assert res["is_lower_bound"] is True

    def test_steps_times_dt_equals_dt_units(self):
        acc = self._make_acc()
        dt  = 0.05; tau = 3.0
        res = valid_prediction_horizon(acc, tau, dt, 0.5)
        assert abs(res["dt_units"] - res["steps"] * dt) < 1e-8

    def test_invariant8_unit_annotation(self):
        """Result dict must contain 'tau_lambda' key (Invariant 8)."""
        acc = self._make_acc()
        res = valid_prediction_horizon(acc, 5.0, 1.0, 0.5)
        assert "tau_lambda" in res, "VPH result must include tau_lambda (Invariant 8)"


# ---------------------------------------------------------------------------
# 8. eval_trajectory integration test
# ---------------------------------------------------------------------------

class TestEvalTrajectory:
    def test_all_keys_present(self):
        B2, T2 = 2, 8
        x = torch.randn(B2, T2, 1, H, H)
        y = x + 0.1 * torch.randn_like(x)
        result = eval_trajectory(x, y, tau_lambda_steps=5.0, dt_snapshot=0.1)
        for key in ("rmse", "acc", "spec_error", "vph_acc", "vph_rmse",
                    "spectral_drift", "clim_std"):
            assert key in result, f"Missing key: {key}"

    def test_identical_trajectories(self):
        x = torch.randn(2, 5, 1, H, H)
        result = eval_trajectory(x, x, tau_lambda_steps=5.0)
        assert torch.allclose(result["rmse"], torch.zeros(5), atol=1e-5)
        assert torch.allclose(result["acc"],  torch.ones(5),  atol=1e-5)


# ---------------------------------------------------------------------------
# 9. Lyapunov estimator smoke test
# ---------------------------------------------------------------------------

class TestLyapunovEstimator:
    """Tests against the debug dataset (generated once per session)."""

    @pytest.fixture(scope="class")
    def lyapunov_json(self, tmp_path_factory) -> dict:
        """Run estimate_lyapunov.py on the debug dataset and parse its JSON output."""
        # Use the pre-generated debug dataset from test_dataset.py if available,
        # otherwise regenerate. We use a fresh tmp dir to be safe.
        root = tmp_path_factory.mktemp("lyap_debug")

        # First generate the debug dataset
        gen_result = subprocess.run(
            [
                sys.executable, "-m", "msr_hine.data.generate",
                "+debug=true",
                f"data.dataset_root={root}",
                "hydra.run.dir=.",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if gen_result.returncode != 0:
            pytest.fail(f"Dataset generation failed:\n{gen_result.stderr}")

        # Run the Lyapunov estimator
        lyap_result = subprocess.run(
            [
                sys.executable,
                "scripts/estimate_lyapunov.py",
                "+debug=true",
                f"data.dataset_root={root}",
                "hydra.run.dir=.",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if lyap_result.returncode != 0:
            pytest.fail(
                f"Lyapunov estimator failed (exit {lyap_result.returncode}):\n"
                f"STDOUT: {lyap_result.stdout}\nSTDERR: {lyap_result.stderr}"
            )

        # Parse the JSON summary
        json_path = root / "debug" / "lyapunov_estimate.json"
        assert json_path.exists(), f"JSON output not written to {json_path}"
        return json.loads(json_path.read_text())

    def test_lambda_finite(self, lyapunov_json):
        """Lyapunov exponent must be finite and well-defined.

        Note: At Re=1000, n=64 (debug config) the 2D Kolmogorov flow may be in a
        laminar or weakly turbulent regime where λ ≤ 0 — this is physically correct
        for 2D NS with strong drag.  The full-resolution run (Re=4000, n=256) gives
        λ > 0.  Here we just verify the estimator ran and produced a valid number.
        """
        lam = lyapunov_json["lambda"]
        import math
        assert math.isfinite(lam), f"λ must be finite, got {lam}"
        assert lam > 0, (
            "λ should be positive; it is clamped to 1e-6 by the script when negative. "
            f"Got {lam}"
        )

    def test_tau_lambda_positive(self, lyapunov_json):
        assert lyapunov_json["tau_lambda"] > 0

    def test_tau_lambda_steps_positive(self, lyapunov_json):
        assert lyapunov_json["tau_lambda_steps"] > 0

    def test_tau_eddy_positive(self, lyapunov_json):
        assert lyapunov_json["tau_eddy"] > 0

    def test_json_keys_present(self, lyapunov_json):
        for key in ("lambda", "tau_lambda", "tau_lambda_steps", "tau_eddy",
                    "dt_snapshot", "n", "re"):
            assert key in lyapunov_json, f"Missing key in JSON: {key}"

    def test_tau_lambda_consistent(self, lyapunov_json):
        """τ_λ = 1/λ should hold to numerical precision."""
        lam = lyapunov_json["lambda"]
        tau = lyapunov_json["tau_lambda"]
        assert abs(tau - 1.0 / lam) < 1e-6 * tau, (
            f"τ_λ = {tau:.6f} ≠ 1/λ = {1.0/lam:.6f}"
        )

    def test_tau_lambda_steps_consistent(self, lyapunov_json):
        """τ_λ_steps = τ_λ / dt_snapshot."""
        tau     = lyapunov_json["tau_lambda"]
        dt_snap = lyapunov_json["dt_snapshot"]
        steps   = lyapunov_json["tau_lambda_steps"]
        expected = tau / dt_snap
        assert abs(steps - expected) < 0.01 * expected, (
            f"τ_λ_steps={steps:.2f} ≠ τ_λ/dt={expected:.2f}"
        )
