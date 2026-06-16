"""Autoregressive rollout with warmup (DESIGN.md §5, Invariants 1, 7, 8).

Public API
──────────
rollout(model, omega_seed, n_steps, warmup_frames, ...)
    Teacher-forced warmup → free autoregressive rollout.
    Returns only the FREE-rollout predictions (warmup excluded — Invariant 7).
    The inference path contains NO posterior re-encode/fusion (Invariant 1).

evaluate_trajectory(model, omega_traj, warmup_len, tau_lambda_steps, dt_snapshot, ...)
    Slice warmup / target from a stored trajectory, call rollout, compute
    all metrics via metrics.eval_trajectory.

ModelInterface
    Both FNOBaseline and (later) MSRHINE must satisfy this duck-typed interface:
        model(omega [B,1,H,W]) → omega_hat [B,1,H,W]   (one-step forward)
    Stateful models additionally expose:
        model.init_state(B, device) → state
        model.warmup(omega_history [B,W,1,H,W], state) → state
        model.step(omega, state, step_n) → (omega_hat, next_state)
    If those methods are absent the fallback is the stateless one-step forward.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_stateful(model: nn.Module) -> bool:
    """Return True if the model exposes the stateful (warmup/step) interface."""
    return hasattr(model, "warmup") and hasattr(model, "step") and hasattr(model, "init_state")


def _one_step(model: nn.Module, omega: Tensor, state, step_n: int):
    """Advance one step, handling both stateless and stateful models.

    For stateful models (MSRHINE), step_n is intentionally NOT passed —
    the model carries its own clock inside state.step_n which _advance_state
    increments.  Passing an external step_n would reset the clock and break
    stride continuity after warmup.

    Returns (omega_hat, new_state).  For stateless models state is always None.
    """
    if _is_stateful(model):
        return model.step(omega, state)
    else:
        # Stateless (fno_1step, hine): no persistent state, no re-encode.
        # Invariant 1 is trivially satisfied — there is no latent to fuse.
        return model(omega), None


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(
    model:          nn.Module,
    omega_seed:     Tensor,
    n_steps:        int,
    warmup_frames:  Optional[Tensor] = None,
    return_all:     bool = True,
) -> Tensor:
    """Free autoregressive rollout with optional teacher-forced warmup.

    Invariant 1:  The inference path performs NO posterior re-encode or fusion.
                  Predictions feed back only through the standard forward path.
    Invariant 7:  Warmup frames are excluded from the returned trajectory;
                  the caller must measure the prediction horizon from step 0
                  of the returned tensor (= end of warmup).

    Args:
        model:          Trained model.  Must implement __call__(omega) → omega_hat
                        and optionally the stateful interface (warmup/step/init_state).
        omega_seed:     Starting vorticity [B, 1, H, W] — the first frame
                        AFTER warmup from which free rollout begins.
        n_steps:        Number of free-rollout steps.
        warmup_frames:  Teacher-forced warmup history [B, W, 1, H, W].
                        If None: stateful models are zero-initialised (no_warmup
                        ablation); stateless models ignore this.
        return_all:     If True return all n_steps predictions [B, n_steps, 1, H, W].
                        If False return only the final step [B, 1, H, W].

    Returns:
        Predictions [B, n_steps, 1, H, W] or [B, 1, H, W].
    """
    model.eval()
    device = omega_seed.device
    B      = omega_seed.shape[0]

    # -- warmup phase (no gradient, no loss) --
    state = None
    if _is_stateful(model):
        state = model.init_state(B, device)
        if warmup_frames is not None:
            state = model.warmup(warmup_frames, state)
        # else: zero-init state (no_warmup ablation)

    # -- free rollout (Invariant 1: purely forward, no re-encode) --
    omega    = omega_seed
    preds: list[Tensor] = []

    for step_n in range(n_steps):
        omega, state = _one_step(model, omega, state, step_n)
        if return_all:
            preds.append(omega)

    if return_all:
        return torch.stack(preds, dim=1)   # [B, n_steps, 1, H, W]
    else:
        return omega                        # [B, 1, H, W]


# ---------------------------------------------------------------------------
# evaluate_trajectory
# ---------------------------------------------------------------------------

def evaluate_trajectory(
    model:              nn.Module,
    omega_traj:         Tensor,
    warmup_len:         int,
    tau_lambda_steps:   float,
    dt_snapshot:        float = 1.0,
    device:             Optional[torch.device] = None,
    acc_threshold:      float = 0.5,
    rmse_threshold:     float = 0.65,
) -> dict:
    """Evaluate a single stored trajectory: rollout then full metrics.

    Slices the trajectory into warmup / seed / target regions:
        omega_traj[:warmup_len]          → warmup_frames (teacher-forced)
        omega_traj[warmup_len]           → omega_seed    (first free step)
        omega_traj[warmup_len+1:]        → omega_target  (ground truth)

    Horizon is measured from end of warmup (Invariant 7).
    Returned VPH is in τ_λ units (Invariant 8).

    Args:
        model:             Trained model.
        omega_traj:        Full trajectory [T, 1, H, W] float32.
        warmup_len:        Number of warmup frames W (excluded from metrics).
        tau_lambda_steps:  Lyapunov time in snapshot steps (for VPH).
        dt_snapshot:       Physical time per snapshot step.
        device:            Compute device (defaults to model's first parameter device).
        acc_threshold:     ACC threshold for VPH.
        rmse_threshold:    Normalised RMSE threshold for RMSE-based VPH.

    Returns:
        Dict from metrics.eval_trajectory plus 'n_steps' and 'warmup_len'.
    """
    from msr_hine.metrics import eval_trajectory as _eval

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    T = omega_traj.shape[0]
    # Warmup consumes frames 0..W-1. Seed = frame[W] (first free frame, step_n=W).
    # Targets start at frame[W+1]. n_steps = T - W - 1.
    n_steps = T - warmup_len - 1
    if n_steps <= 0:
        raise ValueError(
            f"Trajectory too short: T={T}, warmup_len={warmup_len}, "
            f"leaving {n_steps} prediction steps."
        )

    # Move to device, add batch dim
    traj_d = omega_traj.to(device).unsqueeze(0)   # [1, T, 1, H, W]

    warmup_frames = traj_d[:, :warmup_len]          # [1, W, 1, H, W]
    # Seed = frame[W], first free frame — matches training (omega_window[:, warmup])
    omega_seed    = traj_d[:, warmup_len]            # [1, 1, H, W]
    omega_target  = traj_d[:, warmup_len + 1:]       # [1, n_steps, 1, H, W]

    # Run free rollout (warmup excluded from returned tensor — Invariant 7)
    preds = rollout(model, omega_seed, n_steps,
                    warmup_frames=warmup_frames)                   # [1, n_steps, 1, H, W]

    result = _eval(
        omega_hat        = preds,
        omega_true       = omega_target,
        tau_lambda_steps = tau_lambda_steps,
        dt_snapshot      = dt_snapshot,
        acc_threshold    = acc_threshold,
        rmse_threshold   = rmse_threshold,
    )
    result["n_steps"]   = n_steps
    result["warmup_len"] = warmup_len
    return result
