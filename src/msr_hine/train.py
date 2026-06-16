"""Training loop for MSR-HINE-2D (DESIGN.md §7, Invariants 1, 7).

Hydra entry point:
    python -m msr_hine.train model=model_fno data=data_kolmogorov
    python -m msr_hine.train --config-name ablations/fno_1step

Design choices:
  • TBPTT: each batch window covers warmup_steps + rollout_steps frames.
    Warmup is teacher-forced with no loss (Invariant 7).
    Free-rollout K steps accumulate L_state + L_spec + L_highk.
  • Scheduled sampling on the FIELD only (DESIGN.md §7).  At each free-rollout
    step the previous input is drawn from {ground truth, previous prediction}
    with probability {tf_prob, 1-tf_prob}.  No posterior channel exists.
  • Gradient checkpointing per unrolled step bounds GPU memory at 256².
  • AdamW + cosine LR schedule with linear warmup.
  • Optional W&B logging; all hparams logged as config.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
try:
    # PyTorch ≥ 2.4: device-aware autocast/GradScaler in torch.amp
    from torch.amp import GradScaler, autocast as _autocast
    def autocast(enabled: bool = True):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return _autocast(dev, enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]

from msr_hine.data.dataset import build_dataloaders
from msr_hine.losses import l_state, total_loss
from msr_hine.metrics import anomaly_correlation, valid_prediction_horizon
from msr_hine.rollout import rollout
from msr_hine.utils import get_device, seed_everything, setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate the model specified in cfg.model."""
    name = cfg.model.get("name", "fno_1step")
    if name == "fno_1step":
        from msr_hine.models.fno_baseline import FNOBaseline
        return FNOBaseline(
            width   = cfg.model.fno.get("width",    64),
            modes   = cfg.model.fno.get("modes",    32),
            n_layers= cfg.model.fno.get("n_layers", 4),
        )
    elif name == "hine":
        from msr_hine.models.hine import HINE
        mcfg = cfg.model
        return HINE(
            medium_dim         = mcfg.get("medium_dim",          128),
            coarse_dim         = mcfg.get("coarse_dim",           64),
            unet_base_channels = mcfg.unet.get("base_channels",   64),
            unet_channel_mults = tuple(mcfg.unet.get("channel_mults", [1,2,2,4,4])),
            attn_resolutions   = tuple(mcfg.unet.get("attn_resolutions", [16])),
            input_size         = cfg.data.get("n", 256),
            high_k_damping     = mcfg.unet.get("high_k_damping", False),
            enc_hidden_ch      = mcfg.get("enc_hidden_ch", 32),
        )
    elif name == "msr_hine":
        from msr_hine.models.msr_hine import MSRHINE
        mcfg = cfg.model
        return MSRHINE(
            medium_dim         = mcfg.recurrence.get("medium_dim",     128),
            coarse_dim         = mcfg.recurrence.get("coarse_dim",      64),
            medium_stride      = mcfg.recurrence.get("medium_stride",    2),
            coarse_stride      = mcfg.recurrence.get("coarse_stride",    4),
            warmup_steps       = mcfg.get("warmup_steps",               12),
            alpha_max          = mcfg.recurrence.get("alpha_max",       0.2),
            unet_base_channels = mcfg.unet.get("base_channels",         64),
            unet_channel_mults = tuple(mcfg.unet.get("channel_mults", [1,2,2,4,4])),
            attn_resolutions   = tuple(mcfg.unet.get("attn_resolutions", [16])),
            input_size         = cfg.data.get("n", 256),
            high_k_damping     = mcfg.unet.get("high_k_damping",     False),
            use_contraction    = mcfg.recurrence.get("use_contraction", True),
            enc_hidden_ch      = mcfg.get("enc_hidden_ch",              32),
            film_gamma_mode    = mcfg.get("film", {}).get(
                "gamma_mode", "direct"),
            film_gamma_scale   = mcfg.get("film", {}).get(
                "gamma_scale", 0.5),
            # Ablation toggles
            single_scale       = mcfg.get("single_scale",    False),
            use_topdown        = mcfg.get("use_topdown",      True),
            use_warmup         = mcfg.get("use_warmup",       True),
            _inference_fusion_CONTROL_ONLY = mcfg.get(
                "_inference_fusion_CONTROL_ONLY", False),
        )
    elif name == "unet_1step":
        from msr_hine.models.unet_baseline import UNetBaseline
        mcfg = cfg.model
        return UNetBaseline(
            base_channels    = mcfg.unet.get("base_channels",    64),
            channel_mults    = tuple(mcfg.unet.get("channel_mults", [1,2,2,4,4])),
            n_res_blocks     = mcfg.unet.get("n_res_blocks",      2),
            groups           = mcfg.unet.get("groups",            8),
            attn_resolutions = tuple(mcfg.unet.get("attn_resolutions", [16])),
            input_size       = cfg.data.get("n", 256),
            high_k_damping   = mcfg.unet.get("high_k_damping", False),
        )
    else:
        raise ValueError(f"Unknown model name: {name!r}")


# ---------------------------------------------------------------------------
# Scheduled sampling
# ---------------------------------------------------------------------------

def scheduled_sampling_prob(
    epoch:        int,
    total_epochs: int,
    start_prob:   float = 1.0,
    end_prob:     float = 0.2,
    phase1_prob:  float = 0.8,
    phase1_frac:  float = 0.1,
    decay_end_frac: float = 0.5,
) -> float:
    """Three-stage teacher-forcing schedule.

    For the default 100-epoch control run:
      epochs 0..10:  1.0 → 0.8
      epochs 10..50: 0.8 → 0.2
      epochs 50..99: hold at 0.2

    Args:
        epoch:        Current epoch (0-based).
        total_epochs: Total training epochs.
        start_prob:   Teacher-forcing probability at epoch 0 (default 1.0).
        end_prob:     Final teacher-forcing probability.
        phase1_prob:  Probability at the end of the initial transition.
        phase1_frac:  Fraction of training assigned to the initial transition.
        decay_end_frac: Fraction by which end_prob must be reached.

    Returns:
        Scalar probability in [end_prob, start_prob].
    """
    if total_epochs <= 1:
        return start_prob
    if not 0.0 <= phase1_frac <= decay_end_frac <= 1.0:
        raise ValueError(
            "scheduled sampling fractions must satisfy "
            "0 <= phase1_frac <= decay_end_frac <= 1"
        )

    last_epoch = total_epochs - 1
    phase1_end = int(round(phase1_frac * last_epoch))
    decay_end = int(round(decay_end_frac * last_epoch))

    if epoch <= phase1_end:
        frac = epoch / max(phase1_end, 1)
        value = start_prob + frac * (phase1_prob - start_prob)
        return max(phase1_prob, value) if start_prob >= phase1_prob else min(phase1_prob, value)
    if epoch <= decay_end:
        frac = (epoch - phase1_end) / max(decay_end - phase1_end, 1)
        value = phase1_prob + frac * (end_prob - phase1_prob)
        return max(end_prob, value) if phase1_prob >= end_prob else min(end_prob, value)
    return end_prob


# ---------------------------------------------------------------------------
# TBPTT step
# ---------------------------------------------------------------------------

def _unet_step(model: nn.Module, omega_in: Tensor, state) -> tuple[Tensor, object]:
    """One model step, used as the gradient-checkpointing unit.

    For gradient checkpointing to work with the stateful MSRHINE, we checkpoint
    the U-Net call only (the most memory-intensive part) while letting the
    state-update path (GRU, encoders) run normally.  For stateless models we
    checkpoint the entire forward call.
    """
    from msr_hine.rollout import _is_stateful
    if _is_stateful(model):
        # step() carries its own clock in state.step_n — no external step_n needed
        omega_hat, new_state = model.step(omega_in, state)
        return omega_hat, new_state
    else:
        return model(omega_in), None


def _checkpointed_forward(model: nn.Module, omega_in: Tensor) -> Tensor:
    """Stateless forward for gradient checkpointing (fno_1step, etc.)."""
    return model(omega_in)


def _build_down_fn(model: nn.Module):
    """Build the Down latent operator closure for L_cons.

    Down^(medium→coarse) in latent space: z_medium → enc_coarse(dec_medium(z_m))
    This implements the nested spectral truncation on latents:
      Down(E^medium(P^medium ω̂)) ≈ E^coarse(P^coarse(P^medium ω̂))
                                  = E^coarse(P^coarse ω̂)  (by nesting)
    Both sides of L_cons are computed as direct encodings of ω̂; the down_fn
    maps medium latent → coarse-resolution approximation via dec_medium → enc_coarse.

    Returns None if the model doesn't have the needed encoder/decoder attributes.
    """
    if not (hasattr(model, 'enc_coarse') and model.enc_coarse is not None and
            hasattr(model, 'dec_medium') and model.dec_medium is not None):
        return None
    def _down(z_medium: Tensor) -> Tensor:
        field = model.dec_medium(z_medium)     # latent → band-limited field
        return model.enc_coarse(field)         # re-encode at coarse resolution
    return _down


def tbptt_step(
    model:                nn.Module,
    omega_window:         Tensor,
    optimizer:            torch.optim.Optimizer,
    cfg:                  DictConfig,
    teacher_forcing_prob: float,
    scaler:               Optional[GradScaler],
    window_start_n:       int = 0,
) -> dict[str, float]:
    """One TBPTT update over a window of frames.

    Window layout:
        omega_window[:, :warmup_steps]     — teacher-forced warmup (no loss, Invariant 7)
        omega_window[:, warmup_steps:]     — K free-rollout steps with loss

    Losses accumulated per step:
        L_state, L_spec, L_highk   — every step
        L_prior (MSRHINE only)     — only on update steps (Invariant 3)
        L_cons  (MSRHINE only)     — only on update steps (Invariant 3)

    Gradient checkpointing: when cfg.train.grad_checkpoint=True, the forward
    pass at each step is wrapped in torch.utils.checkpoint.  For MSRHINE,
    only the U-Net call is checkpointed (state updates must flow normally).

    Invariant 1: the model is NEVER asked to re-encode ω̂_{t+1} and fuse it back.
    Invariant 2: ground truth enters only as loss targets, never as model inputs.
    Invariant 7: warmup produces no loss contribution.

    Args:
        model:          Trained model (fno_1step, hine, or msr_hine).
        omega_window:   [B, warmup+K, 1, H, W].
        optimizer:      AdamW.
        cfg:            Hydra config (cfg.train.*).
        teacher_forcing_prob: Probability of using GT as next step input.
        scaler:         Optional AMP GradScaler.
        window_start_n: Global step index of the first free-rollout step.
                        Used for stride-respecting loss masking (Invariant 3).

    Returns:
        Dict of scalar loss values for logging.
    """
    from msr_hine.rollout import _is_stateful, _one_step

    tcfg       = cfg.train
    warmup     = tcfg.warmup_steps
    K          = tcfg.rollout_steps
    gamma      = tcfg.get("gamma",         0.99)
    lam_sp     = tcfg.get("lambda_spec",   0.01)
    lam_hk     = tcfg.get("lambda_highk",  0.0)
    lam_prior  = tcfg.get("lambda_prior",  0.0)
    lam_cons   = tcfg.get("lambda_cons",   0.0)
    k_c        = tcfg.get("k_c",           64)
    use_ckpt   = tcfg.get("grad_checkpoint", False)
    use_amp    = tcfg.get("amp", False) and torch.cuda.is_available()
    m_stride   = tcfg.get("medium_stride", 2)
    c_stride   = tcfg.get("coarse_stride", 4)

    B      = omega_window.shape[0]
    device = omega_window.device
    stateful = _is_stateful(model)

    # ── Warmup: teacher-forced, NO loss, NO grad (Invariant 7) ───────────────
    state = None
    if stateful:
        warmup_frames = omega_window[:, :warmup]        # [B, W, 1, H, W]
        with torch.no_grad():
            state = model.init_state(B, device)
            state = model.warmup(warmup_frames, state)

    omega_in = omega_window[:, warmup]                   # seed: first free frame (frame W)

    # ── Free rollout with gradient accumulation ───────────────────────────────
    preds:             list[Tensor] = []
    z_medium_prior_seq: list[Tensor] = []
    z_coarse_prior_seq: list[Tensor] = []
    z_medium_target_seq: list[Tensor] = []
    z_coarse_target_seq: list[Tensor] = []
    z_medium_seq:      list[Tensor] = []   # encodings of ω̂ for L_cons
    z_coarse_seq:      list[Tensor] = []

    model.train()

    for k in range(K):
        # ── Forward step (with optional gradient checkpointing) ─────────────
        with autocast(enabled=use_amp):
            if use_ckpt and not stateful:
                # Stateless models: checkpoint the full forward call
                omega_out = checkpoint.checkpoint(
                    _checkpointed_forward, model, omega_in, use_reentrant=False)
                new_state = None
            else:
                # Stateful models: run normally (state is non-pickleable)
                omega_out, new_state = _one_step(model, omega_in, state, k)

        preds.append(omega_out)
        if stateful:
            state = new_state

        # ── Collect latent sequences for L_prior and L_cons (MSRHINE only) ──
        # Invariant 2: GT frames are ONLY used as loss targets below, never
        # fed as model inputs.
        if stateful and lam_prior > 0 and state is not None:
            # z_prior comes from the model state (GRU output)
            if hasattr(state, 'z_medium_prior'):
                z_medium_prior_seq.append(state.z_medium_prior)
                z_coarse_prior_seq.append(state.z_coarse_prior)
            # z_target: encode the GT frame as target — Invariant 2 satisfied
            # Target for rollout step k is frame[warmup + 1 + k] (seed is frame[warmup])
            gt_frame = omega_window[:, warmup + 1 + k]
            _has_enc = (hasattr(model, 'enc_medium') and model.enc_medium is not None)
            if _has_enc:
                with torch.no_grad():
                    z_medium_target_seq.append(model.enc_medium(gt_frame).detach())
                    _has_coarse_enc = (hasattr(model, 'enc_coarse')
                                       and model.enc_coarse is not None)
                    if _has_coarse_enc:
                        z_coarse_target_seq.append(model.enc_coarse(gt_frame).detach())

        if stateful and lam_cons > 0 and state is not None:
            _has_enc = (hasattr(model, 'enc_medium') and model.enc_medium is not None)
            _has_coarse_enc = (hasattr(model, 'enc_coarse')
                               and model.enc_coarse is not None)
            if _has_enc and _has_coarse_enc:
                z_medium_seq.append(model.enc_medium(omega_out))
                z_coarse_seq.append(model.enc_coarse(omega_out))

        # ── Scheduled sampling on FIELD only (DESIGN.md §7, Invariant 1) ────
        # GT for step k is frame[warmup + 1 + k] (seed is frame[warmup])
        ground_truth = omega_window[:, warmup + 1 + k]
        if teacher_forcing_prob >= 1.0:
            omega_in = ground_truth
        elif teacher_forcing_prob <= 0.0:
            omega_in = omega_out.detach()
        else:
            use_gt = (torch.rand(B, device=device) < teacher_forcing_prob
                      ).view(B, 1, 1, 1)
            omega_in = torch.where(use_gt.expand_as(omega_out),
                                   ground_truth, omega_out.detach())

    preds_t = torch.stack(preds, dim=1)                        # [B, K, 1, H, W]
    target  = omega_window[:, warmup + 1 : warmup + K + 1]    # [B, K, 1, H, W]

    # ── Build down_fn for L_cons ─────────────────────────────────────────────
    down_fn = _build_down_fn(model) if lam_cons > 0 else None

    # ── Compute total loss ────────────────────────────────────────────────────
    with autocast(enabled=use_amp):
        losses = total_loss(
            omega_hat            = preds_t,
            omega_target         = target,
            z_medium_prior_seq   = z_medium_prior_seq  or None,
            z_coarse_prior_seq   = z_coarse_prior_seq  or None,
            z_medium_target_seq  = z_medium_target_seq or None,
            z_coarse_target_seq  = z_coarse_target_seq or None,
            z_medium_seq         = z_medium_seq        or None,
            z_coarse_seq         = z_coarse_seq        or None,
            down_fn              = down_fn,
            medium_stride        = m_stride,
            coarse_stride        = c_stride,
            # seq_start_n: global step index of the first loss target frame.
            # Target k=0 is at frame[warmup+1+0], so offset = warmup+1.
            # Valid when window_stride is divisible by all recurrent strides
            # (stride 2 and 4), which the current config (window_stride=16) satisfies.
            seq_start_n          = warmup + 1,
            gamma                = gamma,
            lambda_prior         = lam_prior,
            lambda_cons          = lam_cons,
            lambda_spec          = lam_sp,
            lambda_hk            = lam_hk,
            k_c                  = k_c,
        )

    # ── Backward + optimiser step ─────────────────────────────────────────────
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       tcfg.get("clip_grad_norm", 1.0))
        scaler.step(optimizer)
        scaler.update()
    else:
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       tcfg.get("clip_grad_norm", 1.0))
        optimizer.step()

    return {key: val.item() for key, val in losses.items()}


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg:       DictConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Cosine annealing with linear warmup (warmup_epochs epochs)."""
    tcfg = cfg.train
    warmup_e = tcfg.get("lr_warmup_epochs", 5)
    total_e  = tcfg.epochs
    warmup_s = warmup_e * steps_per_epoch
    total_s  = total_e  * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_s:
            return step / max(warmup_s, 1)
        progress = (step - warmup_s) / max(total_s - warmup_s, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model:   nn.Module,
    opt:     torch.optim.Optimizer,
    epoch:   int,
    metrics: dict,
    path:    Path,
    cfg:     Optional[DictConfig] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": opt.state_dict(), "metrics": metrics,
    }
    # Save the full config so aggregate_results.py can reconstruct the model
    if cfg is not None:
        payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    torch.save(payload, path)
    log.info("Checkpoint saved → %s", path)


def _load_checkpoint(
    path:  Path,
    model: nn.Module,
    opt:   Optional[torch.optim.Optimizer] = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    log.info("Loaded checkpoint from %s (epoch %d)", path, epoch)
    return epoch


def _checkpoint_metric(path: Path, key: str, default: float) -> float:
    """Read one saved metric without rebuilding the model."""
    if not path.exists():
        return default
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    return float(checkpoint_data.get("metrics", {}).get(key, default))


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:      nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg:        DictConfig,
    device:     torch.device,
) -> float:
    """Run a deterministic trajectory-balanced short validation pass."""
    model.eval()
    tcfg   = cfg.train
    warmup = tcfg.warmup_steps
    K      = tcfg.rollout_steps
    gamma  = tcfg.get("gamma", 0.99)

    total_loss_val = 0.0
    n_batches = 0
    max_batches = int(cfg.get("eval", {}).get("short_val_max_batches", 10))

    loader = val_loader
    dataset = val_loader.dataset
    if max_batches > 0 and hasattr(dataset, "balanced_indices"):
        max_samples = max_batches * int(val_loader.batch_size or 1)
        indices = dataset.balanced_indices(max_samples)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    from msr_hine.rollout import _is_stateful, _one_step
    for batch in loader:
        omega_window = batch.unsqueeze(2).to(device)   # [B,T,1,H,W]
        B            = omega_window.shape[0]
        state        = model.init_state(B, device) if _is_stateful(model) else None
        # Warmup: spin up hidden state over frames 0..W-1 (no grad)
        if _is_stateful(model):
            state = model.warmup(omega_window[:, :warmup], state)
        omega_in = omega_window[:, warmup]             # seed: frame[W]
        preds    = []
        for k in range(K):
            omega_in, state = _one_step(model, omega_in, state, k)
            preds.append(omega_in)
        preds_t = torch.stack(preds, dim=1)
        target  = omega_window[:, warmup + 1 : warmup + K + 1]  # frames[W+1..W+K]
        total_loss_val += l_state(preds_t, target, gamma).item()
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return total_loss_val / max(n_batches, 1)


@torch.no_grad()
def validate_long_rollout(
    model:      nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg:        DictConfig,
    device:     torch.device,
) -> dict[str, float]:
    """Evaluate fixed-length fully autoregressive rollouts on validation trajectories."""
    model.eval()
    ecfg = cfg.get("eval", {})
    tcfg = cfg.train
    warmup = int(tcfg.warmup_steps)
    n_steps = int(ecfg.get("long_rollout_steps", 64))
    max_trajs = int(ecfg.get("long_rollout_max_trajs", 5))
    tau_steps = float(ecfg.get("tau_lambda_steps", 83.9))
    gamma = float(tcfg.get("gamma", 0.99))

    losses: list[float] = []
    rmses: list[Tensor] = []
    accs: list[Tensor] = []

    for traj_idx, batch in enumerate(val_loader):
        if max_trajs > 0 and traj_idx >= max_trajs:
            break
        traj = batch[0].unsqueeze(1).to(device)  # [T,1,H,W]
        available = traj.shape[0] - warmup - 1
        steps = min(n_steps, available)
        if steps <= 0:
            continue

        warmup_frames = traj[:warmup].unsqueeze(0)
        omega_seed = traj[warmup].unsqueeze(0)
        target = traj[warmup + 1 : warmup + 1 + steps].unsqueeze(0)
        pred = rollout(
            model,
            omega_seed,
            steps,
            warmup_frames=warmup_frames,
        )

        weight_sum = sum(gamma ** k for k in range(steps))
        losses.append(float(l_state(pred, target, gamma).item() / max(weight_sum, 1e-12)))
        rmses.append((pred - target).pow(2).mean(dim=(0, 2, 3, 4)).sqrt().cpu())
        clim = target.mean(dim=(0, 1), keepdim=True)
        accs.append(anomaly_correlation(pred, target, clim).cpu())

    if not losses:
        return {
            "val_long_loss": float("inf"),
            "val_long_rmse": float("inf"),
            "val_long_acc_final": float("-inf"),
            "val_long_vph_steps": 0.0,
        }

    common_steps = min(len(curve) for curve in accs)
    rmse_mean = torch.stack([curve[:common_steps] for curve in rmses]).mean(0)
    acc_mean = torch.stack([curve[:common_steps] for curve in accs]).mean(0)
    vph = valid_prediction_horizon(acc_mean, tau_lambda_steps=tau_steps)
    return {
        "val_long_loss": float(sum(losses) / len(losses)),
        "val_long_rmse": float(rmse_mean.mean().item()),
        "val_long_acc_final": float(acc_mean[-1].item()),
        "val_long_vph_steps": float(vph["steps"]),
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase-0: autoencoder pretraining
# ---------------------------------------------------------------------------

def pretrain_autoencoders(
    model:        nn.Module,
    train_loader: torch.utils.data.DataLoader,
    cfg:          DictConfig,
    device:       torch.device,
) -> None:
    """Phase 0: pretrain E^l / D^l as band-limited autoencoders (DESIGN.md §7).

    Skipped when cfg.train.pretrain_autoencoders is False or model has no encoders.
    Optimises reconstruction loss: ‖D^l(E^l(P^l ω)) − P^l ω‖² for each level.
    Only updates encoder/decoder parameters; the U-Net and GRU are frozen.

    Args:
        model:        Model with enc_medium/dec_medium/enc_coarse/dec_coarse attributes.
        train_loader: DataLoader yielding [B, T, H, W] windows.
        cfg:          Full Hydra config.
        device:       Compute device.
    """
    if not (hasattr(model, 'enc_medium') and hasattr(model, 'dec_medium')):
        log.info("Phase-0 skipped: model has no band-limited encoders.")
        return

    tcfg       = cfg.train
    n_epochs   = tcfg.get("pretrain_epochs", 20)
    lr         = tcfg.get("pretrain_lr", 1e-3)
    use_amp    = tcfg.get("amp", False) and torch.cuda.is_available()

    from msr_hine.spectral.truncation import project
    K_MEDIUM, K_COARSE = 16, 8

    ae_params = list(model.enc_medium.parameters()) + \
                list(model.dec_medium.parameters()) + \
                list(model.enc_coarse.parameters()) + \
                list(model.dec_coarse.parameters())
    ae_opt = torch.optim.Adam(ae_params, lr=lr)

    log.info("Phase-0 autoencoder pretraining: %d epochs", n_epochs)
    for ep in range(n_epochs):
        ep_loss = 0.0; n = 0
        for batch in train_loader:
            omega_window = batch.unsqueeze(2).to(device)   # [B, T, 1, H, W]
            # Use one random frame per window
            t_idx  = torch.randint(0, omega_window.shape[1], (1,)).item()
            omega  = omega_window[:, t_idx]                # [B, 1, H, W]

            with autocast(enabled=use_amp):
                # Medium level
                z_m    = model.enc_medium(omega)
                recon_m = model.dec_medium(z_m)
                target_m = project(omega, K_MEDIUM)
                loss_m   = (recon_m - target_m).pow(2).mean()

                # Coarse level
                z_c    = model.enc_coarse(omega)
                recon_c = model.dec_coarse(z_c)
                target_c = project(omega, K_COARSE)
                loss_c   = (recon_c - target_c).pow(2).mean()

                loss = loss_m + loss_c

            ae_opt.zero_grad(); loss.backward(); ae_opt.step()
            ep_loss += loss.item(); n += 1

        log.info("  Phase-0 epoch %3d/%d  recon=%.5f", ep + 1, n_epochs, ep_loss / max(n, 1))
    log.info("Phase-0 complete.")


# ---------------------------------------------------------------------------
# Re/viscosity curriculum
# ---------------------------------------------------------------------------

def _curriculum_nu(epoch: int, cfg: DictConfig) -> Optional[float]:
    """Return the viscosity for this epoch under the Re curriculum, or None.

    The curriculum anneals ν from high (low Re) to ν_target (Re=4000).
    Returns None when curriculum is disabled or completed.
    """
    tcfg = cfg.train
    curr = tcfg.get("curriculum", {})
    if not curr.get("enabled", False):
        return None

    re_schedule    = list(curr.get("re_schedule",    [1000, 2000, 4000]))
    epoch_schedule = list(curr.get("epoch_schedule", [0,    50,   100]))

    if epoch >= epoch_schedule[-1]:
        return None   # curriculum done; use base Re

    for i in range(len(epoch_schedule) - 1):
        if epoch_schedule[i] <= epoch < epoch_schedule[i + 1]:
            frac = (epoch - epoch_schedule[i]) / max(
                epoch_schedule[i + 1] - epoch_schedule[i], 1)
            re = re_schedule[i] + frac * (re_schedule[i + 1] - re_schedule[i])
            return 1.0 / max(re, 1.0)   # ν = 1/Re
    return None


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Hydra-driven training entry point.

    Usage:
        python -m msr_hine.train model=model_fno data=data_kolmogorov
        python -m msr_hine.train --config-name ablations/fno_1step
    """
    setup_logging(cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    tcfg    = cfg.train
    warmup  = tcfg.warmup_steps
    K       = tcfg.rollout_steps
    window  = warmup + K + 1   # +1: seed=frame[W], targets=frames[W+1..W+K]

    # ── data ──────────────────────────────────────────────────────────────
    from msr_hine.data.dataset import TrajectoryDataset, build_dataloaders
    root = Path(cfg.data.dataset_root)
    train_loader, val_loader, test_loader, norm_stats = build_dataloaders(
        root       = root,
        window     = window,
        batch_size = tcfg.batch_size,
        num_workers= tcfg.get("num_workers", 0),
        stride     = tcfg.get("window_stride", 1),
        normalize  = False,   # work in physical units throughout
    )
    log.info("Train batches/epoch: %d", len(train_loader))

    long_eval_every = int(cfg.get("eval", {}).get(
        "long_rollout_every_n_epochs", 0))
    val_trajectory_loader = None
    if long_eval_every > 0:
        val_trajectory_loader = torch.utils.data.DataLoader(
            TrajectoryDataset(root / "val.h5", norm_stats=None),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    # ── model ──────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s  (%d params)", cfg.model.name, n_params)

    # ── optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = tcfg.get("learning_rate", 1e-4),
        weight_decay = tcfg.get("weight_decay",  1e-5),
    )
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler("cuda") if tcfg.get("amp", False) and torch.cuda.is_available() else None

    # ── optional Phase-0 autoencoder pretraining ───────────────────────────
    if tcfg.get("pretrain_autoencoders", False):
        pretrain_autoencoders(model, train_loader, cfg, device)

    # ── optional resume ────────────────────────────────────────────────────
    out_dir  = Path(cfg.get("output_dir", "outputs"))
    ckpt_dir = out_dir / "checkpoints"
    start_epoch = 0
    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        start_epoch = _load_checkpoint(resume_path, model, optimizer) + 1

    # ── optional W&B ───────────────────────────────────────────────────────
    wbcfg = cfg.get("wandb", None)
    use_wb = wbcfg is not None and wbcfg.get("enabled", False)
    if use_wb:
        try:
            import wandb
            wandb.init(
                project = wbcfg.get("project", "msr_hine_2d"),
                entity  = wbcfg.get("entity",  None),
                config  = OmegaConf.to_container(cfg, resolve=True),
                tags    = list(wbcfg.get("tags", [])),
            )
        except ImportError:
            log.warning("wandb not installed; disabling W&B logging.")
            use_wb = False

    # ── early stopping ─────────────────────────────────────────────────────
    es_patience = tcfg.get("early_stopping_patience", 0)   # 0 = disabled
    es_counter  = 0

    # ── training loop ──────────────────────────────────────────────────────
    best_short_path = ckpt_dir / "best_short.pt"
    best_long_path = ckpt_dir / "best_long.pt"
    best_val_loss = _checkpoint_metric(
        best_short_path, "val_loss", float("inf"))
    best_long_loss = _checkpoint_metric(
        best_long_path, "val_long_loss", float("inf"))
    last_long_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        tf_prob = scheduled_sampling_prob(
            epoch, tcfg.epochs,
            tcfg.scheduled_sampling.get("start_prob", 1.0),
            tcfg.scheduled_sampling.get("end_prob", 0.2),
            tcfg.scheduled_sampling.get("phase1_prob", 0.8),
            tcfg.scheduled_sampling.get("phase1_fraction", 0.1),
            tcfg.scheduled_sampling.get("decay_end_fraction", 0.5),
        )

        epoch_losses: dict[str, float] = {}
        n_batches    = 0
        global_step  = epoch * len(train_loader)

        for batch in train_loader:
            # batch: [B, window, H, W]  (no channel dim from WindowDataset)
            omega_window = batch.unsqueeze(2).to(device)  # [B, window, 1, H, W]

            # window_start_n: step index of the first free-rollout frame
            # Used by stride-respecting losses (Invariant 3).
            win_start = (global_step * tcfg.rollout_steps) % max(
                tcfg.get("traj_len", 500), 1)

            step_losses = tbptt_step(
                model                = model,
                omega_window         = omega_window,
                optimizer            = optimizer,
                cfg                  = cfg,
                teacher_forcing_prob = tf_prob,
                scaler               = scaler,
                window_start_n       = win_start,
            )
            scheduler.step()
            n_batches   += 1
            global_step += 1

            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        # Average over batches
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation
        val_loss = validate(model, val_loader, cfg, device)

        log.info(
            "Epoch %4d/%d | train L=%.4f spec=%.4f highk=%.3e "
            "(weighted=%.3e) | val L=%.4f | tf=%.2f | lr=%.2e",
            epoch + 1, tcfg.epochs,
            epoch_losses.get("state", 0),
            epoch_losses.get("spec",  0),
            epoch_losses.get("highk", 0),
            tcfg.get("lambda_highk", 0.0) * epoch_losses.get("highk", 0),
            val_loss, tf_prob,
            optimizer.param_groups[0]["lr"],
        )

        run_long_eval = (
            val_trajectory_loader is not None
            and ((epoch + 1) % long_eval_every == 0 or epoch + 1 == tcfg.epochs)
        )
        if run_long_eval:
            last_long_metrics = validate_long_rollout(
                model, val_trajectory_loader, cfg, device)
            log.info(
                "Long val %3d-step | loss=%.4f mean_rmse=%.4f "
                "final_acc=%.4f vph_steps=%.0f",
                int(cfg.eval.get("long_rollout_steps", 64)),
                last_long_metrics["val_long_loss"],
                last_long_metrics["val_long_rmse"],
                last_long_metrics["val_long_acc_final"],
                last_long_metrics["val_long_vph_steps"],
            )

        if use_wb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in epoch_losses.items()},
                **{f"val/{k.removeprefix('val_')}": v
                   for k, v in last_long_metrics.items()},
            })

        # Checkpoint — save cfg so aggregate_results.py can rebuild the model
        checkpoint_metrics = {"val_loss": val_loss, **last_long_metrics}
        _save_checkpoint(model, optimizer, epoch,
                         checkpoint_metrics, ckpt_dir / "latest.pt", cfg=cfg)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            es_counter = 0
            _save_checkpoint(model, optimizer, epoch,
                             checkpoint_metrics, ckpt_dir / "best_short.pt", cfg=cfg)
            if val_trajectory_loader is None:
                _save_checkpoint(model, optimizer, epoch,
                                 checkpoint_metrics, ckpt_dir / "best.pt", cfg=cfg)
        else:
            es_counter += 1

        if run_long_eval and last_long_metrics["val_long_loss"] < best_long_loss:
            best_long_loss = last_long_metrics["val_long_loss"]
            _save_checkpoint(model, optimizer, epoch,
                             checkpoint_metrics, ckpt_dir / "best_long.pt", cfg=cfg)
            _save_checkpoint(model, optimizer, epoch,
                             checkpoint_metrics, ckpt_dir / "best.pt", cfg=cfg)

        # Early stopping
        if es_patience > 0 and es_counter >= es_patience:
            log.info(
                "Early stopping: val loss has not improved for %d epochs "
                "(best=%.4f). Stopping at epoch %d/%d.",
                es_patience, best_val_loss, epoch + 1, tcfg.epochs,
            )
            break

    if val_trajectory_loader is None:
        log.info("Training complete. Best short val loss: %.4f", best_val_loss)
    else:
        log.info(
            "Training complete. Best short val loss: %.4f | "
            "best long val loss: %.4f",
            best_val_loss, best_long_loss,
        )
    if use_wb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
