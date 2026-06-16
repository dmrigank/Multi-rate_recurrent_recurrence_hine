"""Estimate the largest Lyapunov exponent λ and Lyapunov time τ_λ for
2D forced Kolmogorov flow (DESIGN.md §9, Invariant 8).

Method: Benettin finite-perturbation algorithm with periodic renormalisation.

    1. Load one trajectory from the dataset (or run the solver fresh) as the
       reference orbit.
    2. Add a small random perturbation δω₀ (‖δω₀‖ = ε₀).
    3. Advance both reference and perturbed trajectories for T_renorm steps.
    4. Measure growth: r = ‖δω_T‖ / ε₀.
    5. Accumulate log(r); renormalise δω back to ε₀.
    6. Repeat for N_iters windows.
    7. λ = mean(log(r_i)) / (T_renorm × dt_snapshot)   [time⁻¹]
       τ_λ = 1 / λ   [time units]
       τ_λ_steps = τ_λ / dt_snapshot  [snapshot steps]

Also reports the large-eddy turnover time τ = 1 / (k_f × U_rms) from the
reference trajectory's time-averaged energy.

Usage:
    # Debug dataset (small grid, fast)
    python scripts/estimate_lyapunov.py +debug=true data.dataset_root=data/kolmogorov

    # Full dataset
    python scripts/estimate_lyapunov.py data.dataset_root=data/kolmogorov

Output: prints λ, τ_λ, τ_λ_steps, and τ; optionally writes a JSON summary.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msr_hine.data.solver import KolmogorovSolver, wavenumbers
from msr_hine.utils import get_device, seed_everything, setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Benettin estimator
# ---------------------------------------------------------------------------

def _rms_norm(omega: torch.Tensor) -> float:
    """RMS norm of a vorticity field."""
    return float(omega.pow(2).mean().sqrt().item())


def benettin_lyapunov(
    solver:           KolmogorovSolver,
    omega0_hat:       torch.Tensor,
    eps0:             float = 1e-4,
    t_renorm_steps:   int   = 10,
    n_iters:          int   = 200,
    seed:             int   = 42,
) -> dict[str, float]:
    """Estimate the largest Lyapunov exponent by Benettin's method.

    Args:
        solver:          Configured KolmogorovSolver.
        omega0_hat:      Initial *spectral* vorticity [1, n, n//2+1] complex128
                         on the solver's device.  Used as the reference starting point.
        eps0:            Initial perturbation amplitude (RMS).
        t_renorm_steps:  Steps between renormalisations (in solver substeps).
        n_iters:         Number of renormalisation cycles.
        seed:            RNG seed for the initial perturbation.

    Returns:
        {
          'lambda':           float  — largest Lyapunov exponent [time⁻¹],
          'tau_lambda':       float  — Lyapunov time in physical time units,
          'tau_lambda_steps': float  — τ_λ in snapshot steps,
          'dt_snapshot':      float  — snapshot dt = solver.dt × t_renorm_steps,
          'log_growths':      list   — per-iteration log stretch factors,
          'n_iters':          int,
        }
    """
    device = solver.device
    n      = solver.n
    dt     = solver.dt

    # Random initial perturbation in physical space, normalised to eps0
    torch.manual_seed(seed)
    domega_phys = torch.randn(1, n, n, dtype=torch.float64, device=device)
    domega_phys = domega_phys * (eps0 / _rms_norm(domega_phys))
    dhat = torch.fft.rfft2(domega_phys) * solver.dealias

    ref_hat  = omega0_hat.clone()
    pert_hat = ref_hat + dhat

    log_growths: list[float] = []
    energy_sum  = 0.0

    for it in range(n_iters):
        # Advance both trajectories for t_renorm_steps substeps
        for _ in range(t_renorm_steps):
            ref_hat  = solver.step(ref_hat)
            pert_hat = solver.step(pert_hat)

        # Measure growth
        delta_hat  = pert_hat - ref_hat
        delta_phys = torch.fft.irfft2(delta_hat, s=(n, n))
        rms_new    = _rms_norm(delta_phys)

        if rms_new < 1e-30:
            log.warning("Perturbation collapsed to zero at iter %d; skipping", it)
            continue

        lg = math.log(rms_new / eps0)
        log_growths.append(lg)

        # Renormalise perturbation back to eps0
        scale     = eps0 / rms_new
        dhat      = delta_hat * scale
        pert_hat  = ref_hat + dhat

        # Accumulate energy for turnover time estimate
        from msr_hine.data.solver import energy, wavenumbers as _wn
        _, _, ksq = _wn(n, device)
        energy_sum += solver.get_energy(ref_hat).item()

    # Lyapunov exponent: mean log growth per unit time
    mean_log = float(np.mean(log_growths)) if log_growths else 0.0
    dt_window = dt * t_renorm_steps          # physical time per renorm window
    lam = mean_log / dt_window               # [time⁻¹]

    if lam <= 0:
        log.warning("Estimated λ ≤ 0 (%.4f); system may not be chaotic on this trajectory", lam)
        lam = max(lam, 1e-6)

    tau_lambda = 1.0 / lam

    # Energy-based turnover time: τ = 1 / (k_f * sqrt(2E))
    E_mean   = energy_sum / max(len(log_growths), 1)
    U_rms    = math.sqrt(max(2.0 * E_mean, 1e-10))
    tau_eddy = 1.0 / (solver.k_f * U_rms)

    return {
        "lambda":      lam,
        "tau_lambda":  tau_lambda,
        "dt_window":   dt_window,
        # tau_lambda_steps is NOT set here — it depends on dt_snapshot which is
        # the caller's choice (dataset snapshot spacing, not renorm window).
        "log_growths": log_growths,
        "n_iters":     len(log_growths),
        "tau_eddy":    tau_eddy,
        "E_mean":           E_mean,
    }


# ---------------------------------------------------------------------------
# Load a reference trajectory from HDF5
# ---------------------------------------------------------------------------

def _load_reference(h5_path: Path, traj_idx: int = 0) -> tuple[np.ndarray, dict]:
    """Load one vorticity trajectory from an HDF5 split file."""
    with h5py.File(h5_path, "r") as f:
        traj = f["vorticity"][traj_idx][:]   # [T, n, n] float32
        meta = dict(f.attrs)
    return traj, meta


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Estimate τ_λ and report results."""
    setup_logging(cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    debug: bool = cfg.get("debug", False)
    dcfg = cfg.data

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # --- Select dataset root and parameters ---
    if debug:
        dataset_root = Path(dcfg.get("dataset_root", "data/kolmogorov")) / "debug"
        n     = dcfg.get("debug_n",  64)
        re    = dcfg.get("debug_re", 1000.0)
        dt    = dcfg.get("debug_dt", 5e-3)
        sps   = dcfg.get("debug_substeps_per_snapshot", 5)
        # Use longer renorm windows — perturbation growth needs at least ~0.1 time units.
        # At dt=5e-3, 20 substeps = 0.1 time units ≈ several e-folding windows for Re=1000.
        t_renorm = max(20, sps * 4)
        n_iters  = 100
    else:
        dataset_root = Path(dcfg.get("dataset_root", "data/kolmogorov"))
        n     = dcfg.n
        re    = dcfg.re
        dt    = dcfg.get("dt_substep", 2.5e-4)
        sps   = dcfg.substeps_per_snapshot
        t_renorm = sps * 4      # 4 snapshots per renorm window
        n_iters  = 500

    train_h5 = dataset_root / "train.h5"
    if not train_h5.exists():
        log.error("Dataset not found at %s. Run generate.py first.", train_h5)
        raise FileNotFoundError(str(train_h5))

    # --- Build solver ---
    solver = KolmogorovSolver(
        n=n, re=re, k_f=dcfg.k_f, mu=dcfg.mu, dt=dt, device=device
    )
    log.info("Solver: n=%d, Re=%.0f, dt=%.2e", n, re, dt)

    # --- Load reference trajectory and spin up to its end state ---
    log.info("Loading reference trajectory from %s", train_h5)
    traj, meta = _load_reference(train_h5, traj_idx=0)
    dt_snapshot = float(meta.get("dt_snapshot", dt * sps))
    log.info("Reference traj shape: %s, dt_snapshot=%.4f", traj.shape, dt_snapshot)

    # Spin up from the last stored frame to ensure we're on the attractor.
    # The stored trajectory may have been short (debug mode); add extra warmup.
    omega_last = torch.from_numpy(traj[-1].astype(np.float64)).unsqueeze(0).to(device)
    ref_hat = torch.fft.rfft2(omega_last) * solver.dealias
    extra_spinup = t_renorm * 20   # burn-in before measuring
    log.info("Extra spinup: %d substeps (%.3f time units) to ensure attractor.",
             extra_spinup, dt * extra_spinup)
    for _ in range(extra_spinup):
        ref_hat = solver.step(ref_hat)
    log.info("Starting Benettin iteration.")

    # --- Run Benettin ---
    log.info(
        "Benettin: n_iters=%d, t_renorm=%d substeps (= %.4f time units), eps0=1e-4",
        n_iters, t_renorm, dt * t_renorm,
    )
    results = benettin_lyapunov(
        solver=solver,
        omega0_hat=ref_hat,
        eps0=1e-4,
        t_renorm_steps=t_renorm,
        n_iters=n_iters,
        seed=cfg.get("seed", 42),
    )

    lam            = results["lambda"]
    tau_lambda     = results["tau_lambda"]
    # τ_λ in snapshot steps = τ_λ (time units) / dt_snapshot (time per snapshot)
    tau_lam_steps  = tau_lambda / max(dt_snapshot, 1e-10)
    tau_eddy       = results["tau_eddy"]

    log.info("=" * 60)
    log.info("Largest Lyapunov exponent:  λ    = %.6f  [time⁻¹]", lam)
    log.info("Lyapunov time:              τ_λ  = %.4f  [time units]", tau_lambda)
    log.info("Lyapunov time (steps):      τ_λ  = %.2f  [snapshot steps]", tau_lam_steps)
    log.info("Large-eddy turnover time:   τ    = %.4f  [time units]", tau_eddy)
    log.info("τ_λ / τ (ratio):                 = %.3f", tau_lambda / max(tau_eddy, 1e-10))
    log.info("dt_snapshot / τ_λ:               = %.4f  (snapshot spacing in Lyap-units)",
             dt_snapshot / max(tau_lambda, 1e-10))
    log.info("=" * 60)

    # --- Save JSON summary ---
    out_path = dataset_root / "lyapunov_estimate.json"
    summary = {
        "lambda":             lam,
        "tau_lambda":         tau_lambda,
        "tau_lambda_steps":   tau_lam_steps,   # τ_λ / dt_snapshot (in snapshot steps)
        "tau_eddy":           tau_eddy,
        "dt_snapshot":        dt_snapshot,     # dataset snapshot spacing (time units)
        "n":                  n,
        "re":                 re,
        "n_iters":            results["n_iters"],
        "t_renorm_substeps":  t_renorm,
        "dt_renorm_window":   results["dt_window"],
    }
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved Lyapunov summary to %s", out_path)

    print(f"\n  λ = {lam:.6f}  |  τ_λ = {tau_lambda:.4f}  |  τ_λ_steps = {tau_lam_steps:.1f}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
