"""Loss functions for MSR-HINE-2D training (DESIGN.md §6).

Implemented here (Prompt 8 scope):
    l_state  — discounted multi-step rollout MSE
    l_spec   — radial energy spectrum L1
    l_highk  — high-k energy penalty
    total_loss — weighted combination (state + spec + highk)

Hooks present but raise NotImplementedError (added in Prompt 9):
    l_prior  — stride-respecting latent prior loss (Invariant 3)
    l_cons   — training-only consistency loss via nested Down (Invariants 1, 3)

The spectrum functions reuse the physical-space implementation from metrics.py
so that training and evaluation spectra are computed identically.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from msr_hine.metrics import radial_energy_spectrum as _radial_energy_spectrum


# ---------------------------------------------------------------------------
# L_state: discounted multi-step rollout MSE
# ---------------------------------------------------------------------------

def l_state(
    omega_hat:    Tensor,
    omega_target: Tensor,
    gamma:        float = 0.99,
) -> Tensor:
    """Discounted multi-step rollout MSE (DESIGN.md §6).

    L_state = Σ_{k=1}^{K} γ^{k-1} ‖ω̂_{n+k} − ω_{n+k}‖²_mean

    Args:
        omega_hat:    Predicted vorticity [B, K, 1, H, W].
        omega_target: Ground-truth vorticity [B, K, 1, H, W].
        gamma:        Discount factor per step (default 0.99).

    Returns:
        Scalar loss tensor.
    """
    K = omega_hat.shape[1]
    # Discount weights γ^0, γ^1, ..., γ^{K-1}
    weights = torch.tensor(
        [gamma ** k for k in range(K)],
        dtype=omega_hat.dtype,
        device=omega_hat.device,
    )  # [K]

    # MSE per step averaged over B, C, H, W
    mse_per_step = (omega_hat - omega_target).pow(2).mean(dim=(0, 2, 3, 4))  # [K]
    return (weights * mse_per_step).sum()


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def _spectrum(omega: Tensor) -> Tensor:
    """Return radial energy spectrum [B, K] from vorticity [B, 1, H, W]."""
    B, C, H, W = omega.shape
    _, E_k = _radial_energy_spectrum(omega.squeeze(1))  # [B, H//2]
    return E_k  # [B, H//2]


def _spectrum_sequence(omega_seq: Tensor) -> Tensor:
    """Return spectrum for each step in [B, K, 1, H, W] → [B, K, H//2]."""
    B, K, C, H, W = omega_seq.shape
    flat = omega_seq.reshape(B * K, 1, H, W)
    _, E_k = _radial_energy_spectrum(flat.squeeze(1))  # [B*K, H//2]
    return E_k.reshape(B, K, -1)  # [B, K, H//2]


# ---------------------------------------------------------------------------
# L_spec: radial energy spectrum L1
# ---------------------------------------------------------------------------

def l_spec(
    omega_hat:    Tensor,
    omega_target: Tensor,
) -> Tensor:
    """Radial energy spectrum L1 loss (DESIGN.md §6).

    L_spec = Σ_k |Ê(k) − E(k)|  averaged over steps and batch.

    Accepts [B, 1, H, W] (single step) or [B, K, 1, H, W] (sequence).

    Args:
        omega_hat:    Predicted vorticity.
        omega_target: Ground-truth vorticity (same shape).

    Returns:
        Scalar loss tensor.
    """
    if omega_hat.dim() == 4:
        # Single-step: [B, 1, H, W]
        E_hat  = _spectrum(omega_hat)
        E_true = _spectrum(omega_target)
        return (E_hat - E_true).abs().sum(dim=-1).mean()
    else:
        # Sequence: [B, K, 1, H, W]
        E_hat  = _spectrum_sequence(omega_hat)   # [B, K, H//2]
        E_true = _spectrum_sequence(omega_target)
        return (E_hat - E_true).abs().sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# L_highk: high-k energy penalty
# ---------------------------------------------------------------------------

def l_highk(omega_hat: Tensor, k_c: int) -> Tensor:
    """Penalise spectral energy in modes |k| > k_c (DESIGN.md §6).

    L_highk = Σ_{|k|>k_c} |ω̂(k)|²  averaged over B (and K if sequence).

    Args:
        omega_hat: Predicted vorticity [B, 1, H, W] or [B, K, 1, H, W].
        k_c:       Cut-off wavenumber; modes with radial |k| > k_c are penalised.

    Returns:
        Scalar loss tensor.
    """
    if omega_hat.dim() == 5:
        B, K, C, H, W = omega_hat.shape
        omega_hat = omega_hat.reshape(B * K, C, H, W)

    B, C, H, W = omega_hat.shape
    ohat = torch.fft.rfft2(omega_hat if omega_hat.dtype != torch.float16 else omega_hat.float())

    # Radial wavenumber mask: True for modes |k| > k_c
    kx = torch.fft.fftfreq(H, d=1.0 / H, device=omega_hat.device)
    ky = torch.fft.rfftfreq(W, d=1.0 / W, device=omega_hat.device)
    k_rad = (kx.unsqueeze(1) ** 2 + ky.unsqueeze(0) ** 2).sqrt()
    highk_mask = k_rad > k_c  # True = penalise

    # Parseval-correct rfft2 weights: interior frequencies represent both
    # positive and negative ky modes, while DC and Nyquist are unique.
    rfft_weight = torch.ones(W // 2 + 1, device=omega_hat.device,
                             dtype=ohat.real.dtype)
    if W % 2 == 0:
        rfft_weight[1:-1] = 2.0
    else:
        rfft_weight[1:] = 2.0

    # Sum physical high-k energy per sample, then average samples/channels.
    # torch.fft.rfft2 is unnormalised, so Parseval contributes (H*W)^-2.
    weighted_power = ohat.abs().pow(2) * rfft_weight.view(1, 1, 1, -1)
    energy_per_sample = (
        weighted_power * highk_mask.view(1, 1, H, W // 2 + 1)
    ).sum(dim=(-2, -1)) / (H * W) ** 2
    return energy_per_sample.mean()


# ---------------------------------------------------------------------------
# L_prior — stride-respecting latent prior loss (Invariant 3)
# ---------------------------------------------------------------------------

def l_prior(
    z_prior_seq:  list[Tensor],
    z_target_seq: list[Tensor],
    stride:       int,
    seq_start_n:  int = 0,
) -> Tensor:
    """Stride-respecting latent prior loss (DESIGN.md §6, Invariant 3).

    L_prior^l = Σ_{n ∈ U_l} ‖z^l_prior_n − E^l(P^l ω_n)‖²

    where U_l = {n : n mod s_l = 0} is the set of update steps for level l.
    Only those steps contribute; off-stride steps are ignored.

    Invariant 3: losses for level l are computed ONLY on its update steps.
    Ground truth enters here only as a target (z_target), never as a model
    input (Invariant 2).

    Args:
        z_prior_seq:  Length-K list of predicted priors  z^l_prior [B, d] per step.
        z_target_seq: Length-K list of GT-encoded targets E^l(P^l ω) [B, d] per step.
        stride:       Update stride s_l for this level.
        seq_start_n:  Global step index of the first element (default 0).
                      Used to compute which steps are in U_l.

    Returns:
        Scalar loss (zero tensor if no update steps in the window).
    """
    assert len(z_prior_seq) == len(z_target_seq), \
        "z_prior_seq and z_target_seq must have the same length"

    device = z_prior_seq[0].device
    acc    = torch.zeros((), device=device, dtype=z_prior_seq[0].dtype)
    n_updates = 0

    for k, (z_prior, z_target) in enumerate(zip(z_prior_seq, z_target_seq)):
        global_n = seq_start_n + k
        if global_n % stride == 0:                  # Invariant 3: update step only
            acc = acc + (z_prior - z_target).pow(2).mean()
            n_updates += 1

    if n_updates == 0:
        return acc   # zero, but on the right device/graph
    return acc / n_updates


# ---------------------------------------------------------------------------
# L_cons — training-only consistency loss (Invariants 1, 3)
# ---------------------------------------------------------------------------

def l_cons(
    z_medium_seq:  list[Tensor],
    z_coarse_seq:  list[Tensor],
    down_fn:       Callable[[Tensor], Tensor],
    medium_stride: int,
    seq_start_n:   int = 0,
) -> Tensor:
    """Training-only inter-level consistency loss (DESIGN.md §6, Invariants 1, 3).

    L_cons = Σ_{l≥1} Σ_{n ∈ U_l} ‖E^l(P^l ω̂_n) − Down^l(E^{l-1}(P^{l-1}ω̂_n))‖²

    Both z_medium and z_coarse are encodings of the PREDICTED field ω̂ (not GT).
    This is a pure training-time regulariser — it never modifies the latent
    state at inference (Invariant 1).

    The nested Down operator acts on the latent of the coarser level:
        Down(z_medium) ≈ enc_coarse(field reconstructed from z_medium)
    In practice we pass a `down_fn` closure that the training loop constructs
    from the model's encoders/decoders, keeping this function model-agnostic.

    Invariant 3: computed only on medium-level update steps (U_{medium}).
    Invariant 1: z_medium and z_coarse are from predicted fields; GT never
                 enters this loss as an input (only as a target elsewhere).

    Args:
        z_medium_seq:  Length-K list of E^medium(P^medium ω̂) [B, d_m] per step.
        z_coarse_seq:  Length-K list of E^coarse(P^coarse ω̂) [B, d_c] per step.
        down_fn:       Callable mapping z_medium [B, d_m] → z_coarse_approx [B, d_c].
                       Implements Down^(medium→coarse) in latent space.
        medium_stride: Update stride for the medium level.
        seq_start_n:   Global step index of the first element.

    Returns:
        Scalar loss (zero if no update steps in window).
    """
    assert len(z_medium_seq) == len(z_coarse_seq)

    device    = z_medium_seq[0].device
    acc       = torch.zeros((), device=device, dtype=z_medium_seq[0].dtype)
    n_updates = 0

    for k, (z_m, z_c) in enumerate(zip(z_medium_seq, z_coarse_seq)):
        global_n = seq_start_n + k
        if global_n % medium_stride == 0:           # Invariant 3
            z_c_approx = down_fn(z_m)               # Down^l applied in latent space
            acc = acc + (z_c - z_c_approx).pow(2).mean()
            n_updates += 1

    if n_updates == 0:
        return acc
    return acc / n_updates


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------

def total_loss(
    omega_hat:           Tensor,
    omega_target:        Tensor,
    # Latent sequences — None for fno_1step and any non-hierarchical model
    z_medium_prior_seq:  Optional[list[Tensor]] = None,
    z_coarse_prior_seq:  Optional[list[Tensor]] = None,
    z_medium_target_seq: Optional[list[Tensor]] = None,
    z_coarse_target_seq: Optional[list[Tensor]] = None,
    z_medium_seq:        Optional[list[Tensor]] = None,
    z_coarse_seq:        Optional[list[Tensor]] = None,
    down_fn:             Optional[Callable] = None,
    medium_stride:       int   = 2,
    coarse_stride:       int   = 4,
    seq_start_n:         int   = 0,
    gamma:               float = 0.99,
    lambda_prior:        float = 0.0,
    lambda_cons:         float = 0.0,
    lambda_spec:         float = 0.01,
    lambda_hk:           float = 0.0,
    k_c:                 int   = 64,
) -> dict[str, Tensor]:
    """Compute the total weighted loss and return a component dict.

    For models without latents (fno_1step, hine) pass lambda_prior=0
    and lambda_cons=0; the latent arguments are ignored.

    Args:
        seq_start_n: Global step index of the first element in the sequences.
                     Needed so stride masking is correct inside a TBPTT window
                     that doesn't start at step 0.

    Returns:
        Dict with keys: total, state, spec, highk, prior, cons.
    """
    device = omega_hat.device
    zero   = torch.zeros((), device=device, dtype=omega_hat.dtype)

    ls  = l_state(omega_hat, omega_target, gamma)
    lsp = l_spec(omega_hat, omega_target) if lambda_spec > 0 else zero
    lhk = l_highk(omega_hat, k_c)        if lambda_hk   > 0 else zero

    # Latent losses — only called when lambdas > 0 and sequences are provided
    lpr = zero
    if lambda_prior > 0:
        if z_medium_prior_seq is not None and z_medium_target_seq is not None:
            lpr = lpr + l_prior(z_medium_prior_seq, z_medium_target_seq,
                                 medium_stride, seq_start_n)
        if z_coarse_prior_seq is not None and z_coarse_target_seq is not None:
            lpr = lpr + l_prior(z_coarse_prior_seq, z_coarse_target_seq,
                                 coarse_stride, seq_start_n)

    lco = zero
    if lambda_cons > 0 and z_medium_seq is not None and z_coarse_seq is not None:
        assert down_fn is not None, "down_fn required for l_cons"
        lco = l_cons(z_medium_seq, z_coarse_seq, down_fn, medium_stride, seq_start_n)

    total = ls + lambda_spec * lsp + lambda_hk * lhk + lambda_prior * lpr + lambda_cons * lco

    return {
        "total":  total,
        "state":  ls,
        "spec":   lsp,
        "highk":  lhk,
        "prior":  lpr,
        "cons":   lco,
    }
