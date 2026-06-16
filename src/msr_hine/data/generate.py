"""Trajectory generation for 2D forced Kolmogorov flow.

Hydra CLI entry point:
    python -m msr_hine.data.generate            # full dataset (50 train / 5 val / 10 test)
    python -m msr_hine.data.generate --debug    # 2 short trajectories for smoke-testing

Output layout (one HDF5 file per split):
    <root>/train.h5
    <root>/val.h5
    <root>/test.h5

Each file contains:
    vorticity   [N_traj, T, n, n]  float32   # saved snapshots
    seeds       [N_traj]           int64      # IC seeds
    phases      [N_traj]           float64    # forcing phase φ per trajectory

And scalar attributes: re, k_f, mu, nu, n, dt_substep, dt_snapshot,
    substeps_per_snapshot, tau_estimate, spinup_steps.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from msr_hine.data.solver import (
    KolmogorovSolver,
    energy,
    make_kolmogorov_forcing,
    wavenumbers,
)
from msr_hine.utils import get_device, seed_everything, setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turnover-time estimation
# ---------------------------------------------------------------------------

def estimate_turnover_time(
    solver: KolmogorovSolver,
    spinup_substeps: int = 2000,
    n_samples: int = 50,
    seed: int = 0,
) -> float:
    """Estimate the large-eddy turnover time τ = 1 / (k_f * U_rms).

    Runs a short trajectory, measures steady-state U_rms = sqrt(2*E), then
    returns τ = 1 / (k_f * U_rms).

    Args:
        solver: Configured KolmogorovSolver.
        spinup_substeps: Number of substeps before measuring.
        n_samples: Number of snapshots to average for the energy estimate.
        seed: IC seed for the estimation run.

    Returns:
        Estimated turnover time τ (in non-dimensional time units).
    """
    omega0 = solver.random_ic(batch=1, seed=seed)
    ohat = solver.spinup(omega0, n_substeps=spinup_substeps)

    E_sum = 0.0
    for _ in range(n_samples):
        for _ in range(10):
            ohat = solver.step(ohat)
        E_sum += solver.get_energy(ohat).item()

    E_mean = E_sum / n_samples
    U_rms = math.sqrt(max(2.0 * E_mean, 1e-10))
    tau = 1.0 / (solver.k_f * U_rms)
    log.info("Estimated τ = %.4f  (E_mean=%.4f, U_rms=%.4f)", tau, E_mean, U_rms)
    return tau


# ---------------------------------------------------------------------------
# Per-trajectory generation
# ---------------------------------------------------------------------------

def make_phased_forcing(
    n: int,
    k_f: int,
    phase: float,
    device: torch.device,
) -> torch.Tensor:
    """Return forcing spectrum with a phase shift φ applied.

    Physical force: f = -k_f * cos(k_f * y + φ)
    Spectral coefficient at (kx=0, ky=k_f): -k_f * exp(i φ) / 1
    (rfft2 convention: real field, coefficient at positive ky absorbs the full amplitude).

    Args:
        n: Grid size.
        k_f: Forcing wavenumber.
        phase: Phase shift φ ∈ [0, 2π).
        device: Target device.

    Returns:
        forcing_hat [n, n//2+1] complex128.
    """
    fhat = torch.zeros(n, n // 2 + 1, dtype=torch.complex128, device=device)
    if k_f <= n // 2:
        # Same normalisation as make_kolmogorov_forcing: coefficient = -k_f * n²/2 * exp(iφ).
        amp = -k_f * (n * n) / 2
        fhat[0, k_f] = complex(amp * math.cos(phase), amp * math.sin(phase))
    return fhat


def generate_one_trajectory(
    solver: KolmogorovSolver,
    seed: int,
    phase: float,
    spinup_substeps: int,
    n_snapshots: int,
    substeps_per_snapshot: int,
    device: torch.device,
) -> np.ndarray:
    """Generate a single vorticity trajectory.

    Args:
        solver: Configured solver (will have forcing temporarily replaced).
        seed: IC seed for this trajectory.
        phase: Forcing phase φ for this trajectory.
        spinup_substeps: Substeps to discard before saving.
        n_snapshots: Number of snapshots to save.
        substeps_per_snapshot: Substeps between saved snapshots.
        device: Compute device.

    Returns:
        vorticity [n_snapshots, n, n] float32 as a numpy array.
    """
    # Override forcing with phase-shifted version
    solver.forcing_hat = make_phased_forcing(
        solver.n, solver.k_f, phase, device
    )
    # Recompute ETDRK4 coeffs are unchanged (phase only affects forcing, not L)

    omega0 = solver.random_ic(batch=1, seed=seed)  # [1, n, n]

    # Spin up
    ohat = solver.spinup(omega0, n_substeps=spinup_substeps)

    # Collect snapshots
    snaps = []
    for _ in range(n_snapshots):
        for _ in range(substeps_per_snapshot):
            ohat = solver.step(ohat)
        snap = torch.fft.irfft2(ohat, s=(solver.n, solver.n))  # [1, n, n]
        snaps.append(snap[0].to(torch.float32).cpu())

    traj = torch.stack(snaps, dim=0)  # [n_snapshots, n, n]
    return traj.numpy()


# ---------------------------------------------------------------------------
# Dataset-level generation and HDF5 writing
# ---------------------------------------------------------------------------

def generate_split(
    solver: KolmogorovSolver,
    out_path: Path,
    n_traj: int,
    n_snapshots: int,
    spinup_substeps: int,
    substeps_per_snapshot: int,
    seed_offset: int,
    tau_estimate: float,
    device: torch.device,
) -> None:
    """Generate all trajectories for one split and write to a single HDF5 file.

    Seeds: seed_offset, seed_offset+1, ..., seed_offset+n_traj-1.
    Phases: uniformly spaced in [0, 2π), one per trajectory.

    Args:
        solver: Configured solver.
        out_path: Output HDF5 file path.
        n_traj: Number of trajectories.
        n_snapshots: Snapshots per trajectory.
        spinup_substeps: Substeps to discard.
        substeps_per_snapshot: Substeps between saved snapshots.
        seed_offset: First IC seed.
        tau_estimate: Estimated turnover time (stored as metadata).
        device: Compute device.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = solver.n

    phases = np.linspace(0, 2 * math.pi, n_traj, endpoint=False)
    seeds = np.arange(seed_offset, seed_offset + n_traj, dtype=np.int64)

    dt_snapshot = solver.dt * substeps_per_snapshot

    with h5py.File(out_path, "w") as f:
        vort_ds = f.create_dataset(
            "vorticity",
            shape=(n_traj, n_snapshots, n, n),
            dtype=np.float32,
            chunks=(1, min(n_snapshots, 50), n, n),
            compression="lzf",
        )
        f.create_dataset("seeds", data=seeds)
        f.create_dataset("phases", data=phases)

        # Scalar metadata
        f.attrs["re"]                    = solver.re
        f.attrs["k_f"]                   = solver.k_f
        f.attrs["mu"]                    = solver.mu
        f.attrs["nu"]                    = solver.nu
        f.attrs["n"]                     = n
        f.attrs["dt_substep"]            = solver.dt
        f.attrs["dt_snapshot"]           = dt_snapshot
        f.attrs["substeps_per_snapshot"] = substeps_per_snapshot
        f.attrs["tau_estimate"]          = tau_estimate
        f.attrs["spinup_substeps"]       = spinup_substeps
        f.attrs["n_traj"]                = n_traj
        f.attrs["n_snapshots"]           = n_snapshots

        for i in range(n_traj):
            t0 = time.time()
            traj = generate_one_trajectory(
                solver=solver,
                seed=int(seeds[i]),
                phase=float(phases[i]),
                spinup_substeps=spinup_substeps,
                n_snapshots=n_snapshots,
                substeps_per_snapshot=substeps_per_snapshot,
                device=device,
            )
            vort_ds[i] = traj
            elapsed = time.time() - t0
            log.info(
                "  traj %3d/%d  seed=%d  phase=%.3f  "
                "shape=%s  %.1fs",
                i + 1, n_traj, seeds[i], phases[i], traj.shape, elapsed,
            )

    log.info("Wrote %s  (%d trajectories × %d steps)", out_path, n_traj, n_snapshots)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate the Kolmogorov flow dataset.

    Pass --debug (or +debug=true) to generate a tiny dataset quickly.
    """
    setup_logging(cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))

    debug: bool = cfg.get("debug", False)
    dcfg = cfg.data  # data sub-config

    device = get_device(cfg.get("device", "auto"))
    log.info("Using device: %s", device)
    log.info("Full config:\n%s", OmegaConf.to_yaml(cfg))

    # --- Debug overrides ---
    if debug:
        log.info("DEBUG MODE: generating tiny dataset")
        n           = dcfg.get("debug_n", 64)
        re          = dcfg.get("debug_re", 1000.0)
        n_train     = 2
        n_val       = 1
        n_test      = 2
        train_steps = 20
        val_steps   = 20
        test_steps  = 30
        spinup_sub  = 200
        dt          = dcfg.get("debug_dt", 5e-3)
        sps         = dcfg.get("debug_substeps_per_snapshot", 5)
        root        = Path(dcfg.get("dataset_root", "data/kolmogorov")) / "debug"
    else:
        n           = dcfg.n
        re          = dcfg.re
        n_train     = dcfg.n_train
        n_val       = dcfg.n_val
        n_test      = dcfg.n_test
        train_steps = dcfg.train_steps
        val_steps   = dcfg.val_steps
        test_steps  = dcfg.test_steps
        spinup_sub  = dcfg.spinup_steps
        dt          = dcfg.get("dt_substep", 2.5e-4)
        sps         = dcfg.substeps_per_snapshot
        root        = Path(dcfg.dataset_root)

    root.mkdir(parents=True, exist_ok=True)
    log.info("Output root: %s", root)

    # --- Build solver ---
    solver = KolmogorovSolver(
        n=n,
        re=re,
        k_f=dcfg.k_f,
        mu=dcfg.mu,
        dt=dt,
        device=device,
    )
    log.info("Solver: n=%d  Re=%.0f  k_f=%d  mu=%.3f  dt=%.2e",
             n, re, dcfg.k_f, dcfg.mu, dt)

    # --- Estimate turnover time ---
    log.info("Estimating turnover time τ ...")
    tau = estimate_turnover_time(
        solver,
        spinup_substeps=min(spinup_sub, 2000),
        n_samples=30,
        seed=9999,
    )
    dt_snapshot = dt * sps
    log.info("τ = %.4f  |  dt_snapshot = %.4f  |  dt_snapshot/τ = %.3f",
             tau, dt_snapshot, dt_snapshot / tau)

    # Seed offsets: train=0, val=10000, test=20000 — guaranteed disjoint.
    splits = [
        ("train", n_train, train_steps, 0),
        ("val",   n_val,   val_steps,   10_000),
        ("test",  n_test,  test_steps,  20_000),
    ]

    for split_name, n_traj, n_steps, seed_offset in splits:
        log.info("--- Generating %s split: %d trajectories × %d steps ---",
                 split_name, n_traj, n_steps)
        generate_split(
            solver=solver,
            out_path=root / f"{split_name}.h5",
            n_traj=n_traj,
            n_snapshots=n_steps,
            spinup_substeps=spinup_sub,
            substeps_per_snapshot=sps,
            seed_offset=seed_offset,
            tau_estimate=tau,
            device=device,
        )

    log.info("Dataset generation complete. Root: %s", root)


if __name__ == "__main__":
    main()
