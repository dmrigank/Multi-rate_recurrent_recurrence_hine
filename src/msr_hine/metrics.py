"""Evaluation metrics for MSR-HINE-2D (DESIGN.md §9, Invariant 8).

All quantities are computed from *physical-space* vorticity fields.
Horizons are always in Lyapunov-time units τ_λ (Invariant 8).

Public API
──────────
Field-level (from physical vorticity [B, T, 1, H, W]):
    rmse(omega_hat, omega_true)                     → [T]
    anomaly_correlation(omega_hat, omega_true, clim) → [T]
    climatology(omega_traj)                         → [1,1,1,H,W]  (spatial mean over T)
    climatology_std(omega_traj)                     → scalar

Spectrum / enstrophy (from [B, 1, H, W] or [..., H, W]):
    radial_energy_spectrum(omega, n_bins)            → (k_bins [K], E_k [B, K])
    spectral_error(omega_hat, omega_true)            → [T]  (per-step spectrum L1)
    enstrophy(omega)                                 → [...] scalar per sample
    spectral_drift(omega_hat_long, omega_true_long)  → dict[str, Tensor]

Valid prediction horizon (Invariant 8):
    valid_prediction_horizon(acc_t, tau_lambda_steps, dt_snapshot, threshold)
        → {'steps': int, 'tau_lambda': float, 'dt_units': float}
    vph_from_rmse(rmse_t, clim_std, tau_lambda_steps, dt_snapshot, threshold)
        → same dict

Summary helper:
    eval_trajectory(omega_hat, omega_true, tau_lambda_steps, dt_snapshot, clim)
        → dict with all metrics

Notes
─────
• All tensor inputs are float32 on any device; computations stay float32.
• B = batch (trajectories), T = lead-time steps, H = W = grid size.
• omega_hat / omega_true shape convention: [B, T, 1, H, W].
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

def climatology(omega_traj: Tensor) -> Tensor:
    """Time-mean climatology from a trajectory or batch of trajectories.

    Args:
        omega_traj: [B, T, 1, H, W]  or  [T, 1, H, W].

    Returns:
        Time-mean field broadcastable to the input shape:
        [1, 1, 1, H, W] if input was 5-D, [1, 1, H, W] if 4-D.
    """
    if omega_traj.dim() == 5:
        # mean over B and T
        return omega_traj.mean(dim=(0, 1), keepdim=True).mean(dim=0, keepdim=True)
    else:
        return omega_traj.mean(dim=0, keepdim=True)


def climatology_std(omega_traj: Tensor) -> float:
    """RMS standard deviation of the vorticity over the trajectory (for VPH normalisation).

    Args:
        omega_traj: [B, T, 1, H, W]  or  [T, 1, H, W].

    Returns:
        Scalar standard deviation (float).
    """
    clim = climatology(omega_traj)
    return float((omega_traj - clim).pow(2).mean().sqrt().item())


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse(
    omega_hat:  Tensor,
    omega_true: Tensor,
) -> Tensor:
    """Root mean squared error vs. lead time, averaged over batch and space.

    Args:
        omega_hat:  Predictions  [B, T, 1, H, W].
        omega_true: Ground truth [B, T, 1, H, W].

    Returns:
        RMSE [T], float32.
    """
    if omega_hat.shape != omega_true.shape:
        raise ValueError(
            f"Shape mismatch: omega_hat {omega_hat.shape} vs omega_true {omega_true.shape}"
        )
    # MSE averaged over B, C, H, W → [T]
    mse = (omega_hat - omega_true).pow(2).mean(dim=(0, 2, 3, 4))   # [T]
    return mse.sqrt()


# ---------------------------------------------------------------------------
# Anomaly Correlation Coefficient
# ---------------------------------------------------------------------------

def anomaly_correlation(
    omega_hat:   Tensor,
    omega_true:  Tensor,
    clim:        Tensor,
) -> Tensor:
    """Anomaly correlation coefficient vs. lead time.

    ACC_t = Σ_b Σ_x (ω̂_t - clim)(ω_t - clim) /
            sqrt( Σ_b Σ_x (ω̂_t - clim)² · Σ_b Σ_x (ω_t - clim)² )

    Pooled over B and spatial dims at each lead time t.

    Args:
        omega_hat:  [B, T, 1, H, W].
        omega_true: [B, T, 1, H, W].
        clim:       Climatology, broadcastable to the above (e.g. [1,1,1,H,W]).

    Returns:
        ACC [T] ∈ [-1, 1].
    """
    anom_hat  = omega_hat  - clim   # [B, T, 1, H, W]
    anom_true = omega_true - clim

    # Sum over B, C, H, W at each T
    num   = (anom_hat * anom_true).sum(dim=(0, 2, 3, 4))        # [T]
    denom = (
        anom_hat .pow(2).sum(dim=(0, 2, 3, 4)).sqrt() *
        anom_true.pow(2).sum(dim=(0, 2, 3, 4)).sqrt()
    )   # [T]
    # Avoid division by zero at t=0 when pred == truth exactly
    return num / denom.clamp(min=1e-10)


# ---------------------------------------------------------------------------
# Radial energy spectrum (physical-space entry point)
# ---------------------------------------------------------------------------

def _rfft2_ksq(n: int, device: torch.device) -> Tensor:
    """Return |k|² for the rfft2 half-spectrum [n, n//2+1], float32."""
    kx = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    ky = torch.arange(n // 2 + 1, device=device, dtype=kx.dtype)
    return kx.unsqueeze(1).expand(n, n // 2 + 1) ** 2 + \
           ky.unsqueeze(0).expand(n, n // 2 + 1) ** 2


def radial_energy_spectrum(
    omega:  Tensor,
    n_bins: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Radial kinetic energy spectrum E(k) from physical vorticity.

    E(k) = 0.5 * Σ_{|k'|∈shell_k} |ω̂(k')|² / |k'|² / n⁴
    summed over wavenumber shells with Parseval-correct rfft2 weighting.

    The n⁴ factor arises because torch.fft.rfft2 is unnormalised:
    mean(ω²) = Σ'|ω̂|² / n⁴ (see solver._parseval_factor).

    Args:
        omega:  Real vorticity [..., H, W].  H == W assumed.
        n_bins: Number of radial bins (default H//2).

    Returns:
        (k_bins [K], E_k [..., K])  both float32.
    """
    *batch, H, W = omega.shape
    assert H == W, f"Expected square field, got {H}×{W}"
    n = H
    n_bins = n_bins or n // 2

    omega_hat = torch.fft.rfft2(omega.float())          # [..., n, n//2+1] complex

    ksq = _rfft2_ksq(n, omega.device).to(omega_hat.real.dtype)
    ksq_safe = ksq.clone(); ksq_safe[0, 0] = 1.0

    # Energy density: 0.5 |ω̂|²/|k|²
    e_density = 0.5 * omega_hat.abs().pow(2) / ksq_safe
    e_density[..., 0, 0] = 0.0   # zero-mode carries no kinetic energy

    # Parseval correction for rfft2 half-spectrum (double interior ky modes)
    w = torch.ones(n, n // 2 + 1, device=omega.device, dtype=omega_hat.real.dtype)
    w[:, 1 : n // 2] = 2.0
    e_density = e_density * w / (n * n) ** 2

    # Integer radial shell index
    k_mag = ksq.sqrt().round().long()   # [n, n//2+1]
    k_bins = torch.arange(n_bins, device=omega.device, dtype=torch.float32)

    # Scatter into bins
    flat_e = e_density.reshape(*batch, -1)          # [..., n*(n//2+1)]
    flat_k = k_mag.reshape(-1)                       # [n*(n//2+1)]
    valid  = (flat_k >= 0) & (flat_k < n_bins)

    E_k = torch.zeros(*batch, n_bins, device=omega.device, dtype=e_density.dtype)
    E_k.scatter_add_(-1,
                     flat_k[valid].expand(*batch, valid.sum()),
                     flat_e[..., valid])

    return k_bins, E_k


# ---------------------------------------------------------------------------
# Spectral error vs. lead time
# ---------------------------------------------------------------------------

def spectral_error(
    omega_hat:  Tensor,
    omega_true: Tensor,
    n_bins:     Optional[int] = None,
) -> Tensor:
    """L1 radial spectrum error per lead time, averaged over batch.

    Args:
        omega_hat:  [B, T, 1, H, W].
        omega_true: [B, T, 1, H, W].
        n_bins:     Radial bins (default H//2).

    Returns:
        Spectrum L1 error [T].
    """
    B, T, C, H, W = omega_hat.shape
    # Compute spectrum per (B, T) sample
    _, E_hat  = radial_energy_spectrum(omega_hat .reshape(B * T, H, W), n_bins)
    _, E_true = radial_energy_spectrum(omega_true.reshape(B * T, H, W), n_bins)

    E_hat  = E_hat .reshape(B, T, -1)   # [B, T, K]
    E_true = E_true.reshape(B, T, -1)

    # L1 over wavenumber bins, mean over batch
    return (E_hat - E_true).abs().sum(dim=-1).mean(dim=0)   # [T]


# ---------------------------------------------------------------------------
# Enstrophy
# ---------------------------------------------------------------------------

def enstrophy(omega: Tensor) -> Tensor:
    """Mean enstrophy Z = 0.5 ⟨ω²⟩ per sample.

    Args:
        omega: [..., 1, H, W]  or  [..., H, W].

    Returns:
        Scalar enstrophy per sample [...].
    """
    return 0.5 * omega.pow(2).mean(dim=(-2, -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Spectral drift at long lead times
# ---------------------------------------------------------------------------

def spectral_drift(
    omega_hat_long:  Tensor,
    omega_true_long: Tensor,
    n_bins:          Optional[int] = None,
) -> dict[str, Tensor]:
    """Radial spectrum L1 and enstrophy ratio at long lead times.

    Args:
        omega_hat_long:  Predicted vorticity at long lead [B, 1, H, W].
        omega_true_long: Reference (ground truth) fields  [B, 1, H, W].
        n_bins:          Radial bins (default H//2).

    Returns:
        {
          'k_bins':          [K]   — wavenumber bin centres,
          'spec_hat':        [K]   — mean predicted spectrum,
          'spec_true':       [K]   — mean reference spectrum,
          'spec_error':      [K]   — absolute spectral error per bin,
          'spec_l1':         scalar — total L1 spectral error,
          'enstrophy_hat':   scalar — mean predicted enstrophy,
          'enstrophy_true':  scalar — mean reference enstrophy,
          'enstrophy_ratio': scalar — hat / true (1.0 = no drift),
        }
    """
    k_bins, E_hat  = radial_energy_spectrum(
        omega_hat_long .squeeze(1), n_bins
    )   # [B, K]
    _,      E_true = radial_energy_spectrum(
        omega_true_long.squeeze(1), n_bins
    )

    E_hat_mean  = E_hat .mean(0)   # [K]
    E_true_mean = E_true.mean(0)

    spec_err = (E_hat_mean - E_true_mean).abs()   # [K]

    Z_hat  = enstrophy(omega_hat_long ).mean()
    Z_true = enstrophy(omega_true_long).mean()

    return {
        "k_bins":          k_bins,
        "spec_hat":        E_hat_mean,
        "spec_true":       E_true_mean,
        "spec_error":      spec_err,
        "spec_l1":         spec_err.sum(),
        "enstrophy_hat":   Z_hat,
        "enstrophy_true":  Z_true,
        "enstrophy_ratio": Z_hat / Z_true.clamp(min=1e-10),
    }


# ---------------------------------------------------------------------------
# Valid Prediction Horizon (Invariant 8)
# ---------------------------------------------------------------------------

def valid_prediction_horizon(
    acc_t:             Tensor,
    tau_lambda_steps:  float,
    dt_snapshot:       float = 1.0,
    threshold:         float = 0.5,
) -> dict[str, float]:
    """Compute valid prediction horizon from ACC drop-off.

    VPH is defined as the first lead-time step where ACC < threshold.
    Returns the result both in snapshot steps, physical time units, and
    Lyapunov-time units (Invariant 8).

    If ACC never drops below the threshold the horizon is set to T (the
    maximum lead time available), reported as a lower bound.

    Args:
        acc_t:             Anomaly correlation [T], typically 1→0.
        tau_lambda_steps:  Lyapunov time τ_λ expressed in snapshot steps.
        dt_snapshot:       Physical time per snapshot step.
        threshold:         ACC drop threshold (default 0.5).

    Returns:
        {
          'steps':      int   — first step where ACC < threshold,
          'dt_units':   float — horizon in physical time (steps × dt_snapshot),
          'tau_lambda': float — horizon in τ_λ units (Invariant 8),
          'is_lower_bound': bool — True if ACC never dropped below threshold,
        }
    """
    below = (acc_t < threshold).nonzero(as_tuple=False)
    if len(below) == 0:
        steps = int(len(acc_t))
        is_lb = True
    else:
        steps = int(below[0].item())
        is_lb = False

    dt_units   = steps * dt_snapshot
    tau_lambda = dt_units / max(tau_lambda_steps * dt_snapshot, 1e-10)

    return {
        "steps":          steps,
        "dt_units":       dt_units,
        "tau_lambda":     tau_lambda,
        "is_lower_bound": is_lb,
    }


def vph_from_rmse(
    rmse_t:            Tensor,
    clim_std:          float,
    tau_lambda_steps:  float,
    dt_snapshot:       float = 1.0,
    threshold:         float = 0.65,
) -> dict[str, float]:
    """Compute VPH from normalised RMSE exceeding a threshold.

    VPH is the first step where RMSE_t / clim_std > threshold.

    Args:
        rmse_t:            RMSE per lead time [T].
        clim_std:          Climatological standard deviation (normaliser).
        tau_lambda_steps:  Lyapunov time in snapshot steps.
        dt_snapshot:       Physical time per snapshot step.
        threshold:         Normalised RMSE threshold (default 0.65).

    Returns:
        Same dict structure as valid_prediction_horizon.
    """
    norm_rmse = rmse_t / max(clim_std, 1e-10)
    above = (norm_rmse > threshold).nonzero(as_tuple=False)
    if len(above) == 0:
        steps = int(len(rmse_t))
        is_lb = True
    else:
        steps = int(above[0].item())
        is_lb = False

    dt_units   = steps * dt_snapshot
    tau_lambda = dt_units / max(tau_lambda_steps * dt_snapshot, 1e-10)

    return {
        "steps":          steps,
        "dt_units":       dt_units,
        "tau_lambda":     tau_lambda,
        "is_lower_bound": is_lb,
    }


# ---------------------------------------------------------------------------
# Summary evaluator
# ---------------------------------------------------------------------------

def eval_trajectory(
    omega_hat:         Tensor,
    omega_true:        Tensor,
    tau_lambda_steps:  float,
    dt_snapshot:       float = 1.0,
    clim:              Optional[Tensor] = None,
    acc_threshold:     float = 0.5,
    rmse_threshold:    float = 0.65,
    n_bins:            Optional[int] = None,
) -> dict:
    """Compute all metrics for a rollout trajectory.

    Args:
        omega_hat:        Predictions  [B, T, 1, H, W].
        omega_true:       Ground truth [B, T, 1, H, W].
        tau_lambda_steps: Lyapunov time τ_λ in snapshot steps.
        dt_snapshot:      Physical time per snapshot step.
        clim:             Climatology [1,1,1,H,W]; computed from omega_true if None.
        acc_threshold:    ACC threshold for VPH.
        rmse_threshold:   Normalised RMSE threshold for RMSE-based VPH.
        n_bins:           Radial spectrum bins.

    Returns:
        Dict with keys: rmse [T], acc [T], spec_error [T],
        vph_acc, vph_rmse, spectral_drift (at last lead time).
    """
    if clim is None:
        clim = climatology(omega_true)
    clim_std = climatology_std(omega_true)

    rmse_t = rmse(omega_hat, omega_true)
    acc_t  = anomaly_correlation(omega_hat, omega_true, clim)
    spec_t = spectral_error(omega_hat, omega_true, n_bins)

    vph_acc  = valid_prediction_horizon(
        acc_t, tau_lambda_steps, dt_snapshot, acc_threshold
    )
    vph_rmse = vph_from_rmse(
        rmse_t, clim_std, tau_lambda_steps, dt_snapshot, rmse_threshold
    )

    # Spectral drift at the final lead time
    drift = spectral_drift(
        omega_hat [:, -1],   # [B, 1, H, W]
        omega_true[:, -1],
        n_bins,
    )

    return {
        "rmse":           rmse_t,
        "acc":            acc_t,
        "spec_error":     spec_t,
        "vph_acc":        vph_acc,
        "vph_rmse":       vph_rmse,
        "spectral_drift": drift,
        "clim_std":       clim_std,
    }
