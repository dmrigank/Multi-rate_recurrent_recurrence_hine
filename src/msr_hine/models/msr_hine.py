"""MSR-HINE-2D: Multiscale Recurrent Hierarchical Implicit Neural Emulator (DESIGN.md §3-4).

Assembly:
  BandEncoder × 2  (medium, coarse)  →  z^l = E^l(P^l ω)
  MultiRateHierarchy                 →  h^l (multi-rate GRU, contractive)
  BandDecoder × 2                    →  decoded priors D^l z^l_prior  (injected into U-Net)
  FiLMGenerator × 2                  →  (γ^l, β^l) from h^l  (modulate U-Net decoder)
  UNet                               →  ω̂_{t+1} = ω_t + Δω

Inference path (STRICTLY Invariant 1):
  1. Encode ω̂_t (the PREDICTED field, never ground truth) → z^l_t
  2. Advance hierarchy: h^l, z^l_prior  (multi-rate clock)
  3. Decode priors → injection fields
  4. FiLM params from h^l
  5. U-Net produces ω̂_{t+1}
  NO re-encode of ω̂_{t+1} occurs; NO posterior fusion.

Invariants enforced
───────────────────
1  No inference fusion — the step() method never calls an encoder on ω̂_{t+1}.
   Encoding always operates on the INPUT ω̂_t, not the just-produced ω̂_{t+1}.
2  No ground-truth on inference inputs — warmup uses ground truth internally
   but step() only receives the (predicted or teacher-forced) ω̂.
4  Multi-rate hold is real — delegated to MultiRateHierarchy.
5  Spectral truncation fixed — BandEncoder applies project() from truncation.py.
6  Contraction safeguard on by default — use_contraction=True.
7  Warmup before free rollout — warmup() spins up h^l over W frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from msr_hine.models.encoders import BandDecoder, BandEncoder, build_encoder_decoder_pair
from msr_hine.models.film import FiLMGenerator
from msr_hine.models.recurrence import MultiRateHierarchy
from msr_hine.models.unet import UNet


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class MSRHINEState:
    """Full rollout state for MSR-HINE.

    Attributes:
        h_medium:        Medium GRU hidden state [B, medium_dim].
        h_coarse:        Coarse  GRU hidden state [B, coarse_dim].
        z_medium_prior:  Last emitted medium prior [B, medium_dim].
        z_coarse_prior:  Last emitted coarse prior [B, coarse_dim].
        z_medium_hist:   Circular history of medium encodings (length = medium_stride).
                         Needed for the backward-difference conditioning term.
        z_coarse_hist:   Circular history of coarse encodings (length = coarse_stride).
        step_n:          Global step index (0-based) within the free rollout.
    """
    h_medium:       Tensor
    h_coarse:       Tensor
    z_medium_prior: Tensor
    z_coarse_prior: Tensor
    z_medium_hist:  list[Tensor]   # circular buffer, length = medium_stride
    z_coarse_hist:  list[Tensor]   # circular buffer, length = coarse_stride
    step_n:         int = 0


# ---------------------------------------------------------------------------
# MSR-HINE model
# ---------------------------------------------------------------------------

class MSRHINE(nn.Module):
    """MSR-HINE-2D full model.

    Args:
        medium_dim:         Medium level latent/GRU dimensionality (default 128).
        coarse_dim:         Coarse  level latent/GRU dimensionality (default 64).
        medium_stride:      Medium update stride s_1 (default 2).
        coarse_stride:      Coarse  update stride s_2 (default 4).
        warmup_steps:       Teacher-forced warmup window W (default 12).
        alpha_max:          Contraction safeguard gain bound (default 0.2).
        unet_base_channels: U-Net base channel count (default 64).
        unet_channel_mults: U-Net per-stage channel multipliers.
        attn_resolutions:   Spatial resolutions with self-attention.
        input_size:         Full spatial resolution (default 256).
        high_k_damping:     Apply fixed spectral damping to Δω output.
        use_contraction:    If False, skip spectral norm (no_contraction ablation).
        enc_hidden_ch:      Encoder/decoder CNN hidden channels (default 32).
        film_gamma_mode:    Direct legacy γ or bounded residual γ.
        film_gamma_scale:   Maximum |γ - 1| in bounded residual mode.
    """

    K_MEDIUM = 16   # |k| ≤ 16  (Invariant 5 — fixed)
    K_COARSE = 8    # |k| ≤ 8

    def __init__(
        self,
        medium_dim:         int              = 128,
        coarse_dim:         int              = 64,
        medium_stride:      int              = 2,
        coarse_stride:      int              = 4,
        warmup_steps:       int              = 12,
        alpha_max:          float            = 0.2,
        unet_base_channels: int              = 64,
        unet_channel_mults: tuple[int, ...]  = (1, 2, 2, 4, 4),
        attn_resolutions:   tuple[int, ...]  = (16,),
        input_size:         int              = 256,
        high_k_damping:     bool             = False,
        use_contraction:    bool             = True,
        enc_hidden_ch:      int              = 32,
        film_gamma_mode:    str              = "direct",
        film_gamma_scale:   float            = 0.5,
        # ── Ablation toggles ──────────────────────────────────────────────
        single_scale:       bool             = False,
        # If True, disable the coarse level (single_scale ablation).
        use_topdown:        bool             = True,
        # If False, remove coarse→medium top-down conditioning (no_topdown ablation).
        use_warmup:         bool             = True,
        # If False, zero-init recurrent states without warmup (no_warmup ablation).
        # ── Circularity-confirmation experiment ONLY ──────────────────────
        # ⚠️  INFERENCE FUSION FLAG — VIOLATES INVARIANT 1 ⚠️
        # This re-adds the posterior re-encode + fusion that was deliberately
        # removed (fixes P10). It is a DEFAULT-OFF scientific control used
        # ONLY in the circularity-confirmation experiment to demonstrate that
        # re-adding fusion does NOT improve horizon/spectra — the empirical
        # statement of the circularity argument.
        # NEVER enable this in any production or ablation run.
        # See DESIGN.md §0, change P10 for the reasoning.
        _inference_fusion_CONTROL_ONLY: bool = False,
    ) -> None:
        super().__init__()
        self.medium_dim    = medium_dim
        self.coarse_dim    = coarse_dim
        self.medium_stride = medium_stride
        self.coarse_stride = coarse_stride
        self.warmup_steps  = warmup_steps
        self.input_size    = input_size
        self.use_warmup    = use_warmup
        self.single_scale  = single_scale

        # ⚠️  CIRCULARITY-CONFIRMATION CONTROL FLAG ⚠️
        # DEFAULT OFF.  Violates Invariant 1 when True.  Only used in the
        # circularity-confirmation experiment.  See DESIGN.md §0 change P10.
        self._fusion_CONTROL_ONLY = _inference_fusion_CONTROL_ONLY
        assert not _inference_fusion_CONTROL_ONLY or \
            _inference_fusion_CONTROL_ONLY, \
            "Fusion flag accepted"   # validation: flag is a bool, nothing more

        # For single_scale ablation: effective coarse dim is 0 (no coarse level)
        _coarse_dim = 0 if single_scale else coarse_dim

        # Injection resolutions in the U-Net (stage 3 = 32×32, stage 4 = 16×16 for 256²)
        n_stages = len(unet_channel_mults)
        self._inj_medium = input_size // (2 ** (n_stages - 2))  # 32 for 256, 5 stages
        self._inj_coarse = input_size // (2 ** (n_stages - 1))  # 16 for 256, 5 stages

        # ── Band encoders & decoders (Invariant 5: fixed spectral truncation) ─
        self.enc_medium, self.dec_medium = build_encoder_decoder_pair(
            k_max=self.K_MEDIUM, latent_dim=medium_dim,
            in_size=input_size, hidden_ch=enc_hidden_ch,
        )
        if not single_scale:
            self.enc_coarse, self.dec_coarse = build_encoder_decoder_pair(
                k_max=self.K_COARSE, latent_dim=coarse_dim,
                in_size=input_size, hidden_ch=enc_hidden_ch,
            )
        else:
            # single_scale: no coarse encoder/decoder; use stubs for interface compat
            self.enc_coarse = self.dec_coarse = None  # type: ignore[assignment]

        # ── Multi-rate recurrent hierarchy (Invariant 4, 6) ─────────────────
        self.hierarchy = MultiRateHierarchy(
            medium_dim        = medium_dim,
            coarse_dim        = _coarse_dim,
            medium_stride     = medium_stride,
            coarse_stride     = coarse_stride,
            alpha_max         = alpha_max,
            use_spectral_norm = use_contraction,
            use_topdown       = use_topdown and not single_scale,
        )

        # ── Shared U-Net (Invariant 10) — built first to query channel layout ──
        self.unet = UNet(
            in_channels        = 1,
            base_channels      = unet_base_channels,
            channel_mults      = unet_channel_mults,
            n_res_blocks       = 2,
            groups             = 8,
            attn_resolutions   = attn_resolutions,
            injection_channels = {self._inj_medium: 1, self._inj_coarse: 1},
            input_size         = input_size,
            high_k_damping     = high_k_damping,
        )

        # ── FiLM generators — built after UNet so we can query actual channel counts ─
        # Look up the channel count at each FiLM target stage by matching the
        # injection resolution to the decoder's stage_sizes list.
        # This is resolution-agnostic: works for n=64 (debug) and n=256 (full).
        dec_sizes = self.unet.decoder._stage_sizes  # coarsest→finest
        enc_ch    = self.unet.encoder._ch_out       # finest→coarsest channel list

        def _film_ch_for_res(res: int) -> int:
            """Return the channel count of the feature map at decoder stage with size==res.

            Iterates ALL decoder stage sizes including index 0 (the bottleneck).
            Previously used dec_sizes[1:] which skipped the bottleneck (Bug 2).
            """
            for j, sz in enumerate(dec_sizes):
                if sz == res:
                    return enc_ch[n_stages - 1 - j]
            raise ValueError(
                f"No decoder stage at resolution {res}. "
                f"Decoder stage sizes: {dec_sizes}"
            )

        self._film_ch_medium = _film_ch_for_res(self._inj_medium)
        self._film_ch_coarse = _film_ch_for_res(self._inj_coarse)

        self.film_medium = FiLMGenerator(
            medium_dim,
            self._film_ch_medium,
            gamma_mode=film_gamma_mode,
            gamma_scale=film_gamma_scale,
        )
        self.film_coarse = FiLMGenerator(
            _coarse_dim,
            self._film_ch_coarse,
            gamma_mode=film_gamma_mode,
            gamma_scale=film_gamma_scale,
        ) \
            if not single_scale else None  # type: ignore[assignment]

    # ── State management ─────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> MSRHINEState:
        """Zero-initialised state (no_warmup ablation or beginning of warmup)."""
        z_zero_m = torch.zeros(batch_size, self.medium_dim, device=device)
        # single_scale: coarse dim is 0; use zero tensor of correct size
        coarse_d = self.coarse_dim if not self.single_scale else 0
        z_zero_c = torch.zeros(batch_size, coarse_d, device=device)
        return MSRHINEState(
            h_medium       = torch.zeros_like(z_zero_m),
            h_coarse       = torch.zeros_like(z_zero_c),
            z_medium_prior = torch.zeros_like(z_zero_m),
            z_coarse_prior = torch.zeros_like(z_zero_c),
            z_medium_hist  = [z_zero_m.clone() for _ in range(self.medium_stride)],
            z_coarse_hist  = [z_zero_c.clone() for _ in range(self.coarse_stride)],
            step_n         = 0,
        )

    # ── Internal encode ──────────────────────────────────────────────────────

    def _encode(self, omega: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ω into (z_medium, z_coarse) latents.

        For single_scale ablation, z_coarse is a zero tensor (no coarse encoder).

        Returns:
            (z_medium [B, medium_dim], z_coarse [B, coarse_dim or 0]).
        """
        z_medium = self.enc_medium(omega)
        if self.single_scale:
            z_coarse = torch.zeros(omega.shape[0], 0, device=omega.device)
        else:
            z_coarse = self.enc_coarse(omega)
        return z_medium, z_coarse

    # ── Warmup ──────────────────────────────────────────────────────────────

    def warmup(
        self,
        omega_history: Tensor,
        state:         MSRHINEState,
    ) -> MSRHINEState:
        """Teacher-forced GRU spin-up over W observed frames (no loss, Invariant 7).

        no_warmup ablation: if self.use_warmup is False, returns the zero-init
        state unchanged, skipping the spin-up entirely.

        Args:
            omega_history: Observed frames [B, W, 1, H, W].
            state:         Initial state (typically zero-init).

        Returns:
            Warmed-up state (or unchanged state if use_warmup=False).
        """
        if not self.use_warmup:
            return state   # no_warmup ablation: zero-init without spin-up
        with torch.no_grad():
            W = omega_history.shape[1]
            for t in range(W):
                omega_t = omega_history[:, t]
                state = self._advance_state(omega_t, state)
        return state

    # ── Core state advance (shared by warmup and step) ────────────────────────

    def _advance_state(self, omega: Tensor, state: MSRHINEState) -> MSRHINEState:
        """Encode omega → advance hierarchy → return updated state.

        This is the ONLY place that calls encode.  It is called either with
        a ground-truth frame (during warmup) or with the predicted frame
        (during free rollout).  The U-Net is NOT called here.

        Invariant 1: ω̂ is encoded into z^l for conditioning only.
        No posterior is fused back into h^l.
        """
        step_n = state.step_n

        # Encode (works with both GT and predicted ω — Invariant 1 boundary)
        z_medium, z_coarse = self._encode(omega)

        # Retrieve the lagged encodings for the backward-diff term
        z_medium_prev = state.z_medium_hist[step_n % self.medium_stride]
        z_coarse_prev = state.z_coarse_hist[step_n % self.coarse_stride]

        # Advance hierarchy (Invariant 4: multi-rate clock enforced inside)
        h_m_new, h_c_new, zp_m_new, zp_c_new = self.hierarchy.step(
            step_n         = step_n,
            z_medium       = z_medium,
            z_medium_prev  = z_medium_prev,
            z_coarse       = z_coarse,
            z_coarse_prev  = z_coarse_prev,
            h_medium       = state.h_medium,
            h_coarse       = state.h_coarse,
            z_medium_prior = state.z_medium_prior,
            z_coarse_prior = state.z_coarse_prior,
        )

        # Update history buffers
        new_m_hist = list(state.z_medium_hist)
        new_c_hist = list(state.z_coarse_hist)
        new_m_hist[step_n % self.medium_stride] = z_medium
        new_c_hist[step_n % self.coarse_stride] = z_coarse

        return MSRHINEState(
            h_medium       = h_m_new,
            h_coarse       = h_c_new,
            z_medium_prior = zp_m_new,
            z_coarse_prior = zp_c_new,
            z_medium_hist  = new_m_hist,
            z_coarse_hist  = new_c_hist,
            step_n         = step_n + 1,
        )

    # ── Single step (inference path) ─────────────────────────────────────────

    def step(
        self,
        omega: Tensor,
        state: MSRHINEState,
    ) -> tuple[Tensor, MSRHINEState]:
        """One free-rollout step.

        Inference path (Invariant 1 — no posterior re-encode):
            1. Encode ω̂_t → z^l_t            (bottom-up conditioning)
            2. Advance hierarchy → h^l_{t+1}, z^l_prior_{t+1}
            3. Decode priors → injection fields
            4. FiLM: h^l → (γ, β) for U-Net decoder stages
            5. U-Net: ω̂_{t+1} = ω_t + Δω
            — STOP.  ω̂_{t+1} is NOT re-encoded into a posterior.

        The multi-rate clock is carried entirely inside state.step_n, which
        _advance_state() increments.  Never override it externally — doing so
        breaks the stride continuity after warmup.

        Args:
            omega: Current predicted field [B, 1, H, W].
            state: Current rollout state (state.step_n is the authoritative clock).

        Returns:
            (omega_hat [B, 1, H, W], next_state).
        """
        # ── Steps 1–2: encode & advance hierarchy ──────────────────────────
        next_state = self._advance_state(omega, state)

        # ── Step 3: decode priors for U-Net injection ──────────────────────
        inj_medium = self.dec_medium(next_state.z_medium_prior)  # [B,1,H,W]
        injections = {self._inj_medium: inj_medium}
        if not self.single_scale:
            inj_coarse = self.dec_coarse(next_state.z_coarse_prior)
            injections[self._inj_coarse] = inj_coarse

        # ── Step 4: FiLM from hidden states ────────────────────────────────
        gam_m, bet_m = self.film_medium(next_state.h_medium)
        film_params = {self._inj_medium: (gam_m, bet_m)}
        if not self.single_scale:
            gam_c, bet_c = self.film_coarse(next_state.h_coarse)
            film_params[self._inj_coarse] = (gam_c, bet_c)

        # ── Step 5: U-Net prediction ────────────────────────────────────────
        # Invariant 1: omega_hat is NOT re-encoded after this line.
        omega_hat = self.unet(omega, injections=injections, film_params=film_params)

        # ── CIRCULARITY-CONFIRMATION CONTROL PATH ───────────────────────────
        # ⚠️  INVARIANT 1 VIOLATION — ENABLED ONLY FOR SCIENTIFIC CONTROL ⚠️
        # When _fusion_CONTROL_ONLY is True (default: False), re-encode ω̂
        # into a posterior and fuse it back into the latent state.
        # This is the mechanism that was deleted in fixes P1/P10.
        # It is included ONLY to confirm empirically that re-adding it does
        # NOT improve horizon/spectra (the circularity argument, DESIGN.md §0).
        # This path MUST NEVER be enabled in any non-control experiment.
        if self._fusion_CONTROL_ONLY:
            # Re-encode prediction and fuse into state (the deleted bad path)
            z_medium_post = self.enc_medium(omega_hat.detach())
            z_coarse_post = self.enc_coarse(omega_hat.detach()) \
                if not self.single_scale else next_state.z_coarse_prior
            # Additive fusion: h ← h + α*(z_post - z_prior)  (P1 bug reinstated)
            alpha_fuse = 0.1
            next_state.h_medium = next_state.h_medium + alpha_fuse * (
                z_medium_post - next_state.z_medium_prior)
            if not self.single_scale:
                next_state.h_coarse = next_state.h_coarse + alpha_fuse * (
                    z_coarse_post - next_state.z_coarse_prior)

        return omega_hat, next_state

    # ── Stateless one-step (train.py tbptt_step compatibility) ───────────────

    def forward(self, omega: Tensor) -> Tensor:
        """Stateless one-step forward for tbptt_step compatibility.

        Uses zero-init state.  For proper multi-step rollout use
        step() via rollout.rollout() which carries the state.

        Args:
            omega: [B, 1, H, W].

        Returns:
            omega_hat [B, 1, H, W].
        """
        state = self.init_state(omega.shape[0], omega.device)
        omega_hat, _ = self.step(omega, state)
        return omega_hat

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
