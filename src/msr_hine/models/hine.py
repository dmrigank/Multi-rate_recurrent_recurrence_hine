"""HINE baseline: hierarchical latent-future injection, no recurrence.

Implements DESIGN.md §4.1 and CLAUDE.md §3 (HINE baseline).

The HINE mechanism — preserved from the original HINE paper:
  1. Consume CURRENT latent-future estimates (z_medium_t, z_coarse_t) as
     U-Net encoder-stage injections (medium → 32×32, coarse → 16×16).
  2. Run the U-Net to produce Δω and ADVANCED latent-future estimates at
     the matching decoder stages:
       z_medium_{t+1_pred}  (medium level, looks 1 step ahead)
       z_coarse_{t+2_pred}  (coarse level, looks 2 steps ahead)
  3. Next call: use z_medium_{t+1_pred} and z_coarse_{t+2_pred} as injections.

The ladder is REGENERATED from scratch each step — no persistent GRU, no
hidden state between forward() calls.  This is the no-recurrence ablation.

Invariant 10 — SHARED U-Net backbone with MSR-HINE:
    HINE instantiates UNet with the same architecture (base_channels,
    channel_mults, attn_resolutions) so the recurrence ablation is
    backbone-controlled.  FNO is NOT used here.

Staggered horizons (CLAUDE.md §3):
    Medium level l=1: looks ahead 1 step  (stride s_1 = 2 → latent horizon 1)
    Coarse  level l=2: looks ahead 2 steps (stride s_2 = 4 → latent horizon 2)

Parameter count matching:
    The encoder/decoder pairs add ~2-4M parameters (for 256² at latent_dim=128/64).
    UNet(base=64, mults=[1,2,2,4,4]) is ~13.6M, giving ~16-18M total — comparable
    to MSR-HINE which has the same UNet plus GRU blocks.

The step() method follows the train.py model interface:
    model(omega [B,1,H,W]) → omega_hat [B,1,H,W]   (one-step, stateless)
This is used by tbptt_step and rollout.rollout().
The latent state (z_medium, z_coarse) is carried inside the HINEState dataclass
and passed explicitly — matching the pattern the stateful MSR-HINE will use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from msr_hine.models.encoders import BandDecoder, BandEncoder, build_encoder_decoder_pair
from msr_hine.models.unet import UNet


# ---------------------------------------------------------------------------
# Latent state container
# ---------------------------------------------------------------------------

@dataclass
class HINEState:
    """Carrier for the HINE latent-future ladder across autoregressive steps.

    Attributes:
        z_medium: Current medium-level latent-future estimate [B, medium_dim].
        z_coarse: Current coarse-level latent-future estimate [B, coarse_dim].
    """
    z_medium: Tensor
    z_coarse: Tensor


# ---------------------------------------------------------------------------
# HINE model
# ---------------------------------------------------------------------------

class HINE(nn.Module):
    """HINE: hierarchical latent-future injection, no recurrence.

    Shared U-Net backbone (Invariant 10).  No GRU, no persistent hidden state.

    Args:
        medium_dim:         Medium level latent dimensionality (default 128).
        coarse_dim:         Coarse level latent dimensionality (default 64).
        unet_base_channels: U-Net base channel count (default 64).
        unet_channel_mults: Per-stage channel multipliers.
        attn_resolutions:   Spatial sizes with self-attention.
        input_size:         Full spatial input resolution (default 256).
        high_k_damping:     Apply fixed spectral damping to Δω output.
        enc_hidden_ch:      CNN hidden channels in encoder/decoder (default 32).

    Injection and emission stages:
        Medium (|k|≤16): encoder injection at 32×32, decoder emission at 32×32
        Coarse  (|k|≤8):  encoder injection at 16×16, decoder emission at 16×16
    """

    # Spectral bands (fixed, DESIGN.md §2)
    K_MEDIUM = 16
    K_COARSE = 8

    def __init__(
        self,
        medium_dim:         int              = 128,
        coarse_dim:         int              = 64,
        unet_base_channels: int              = 64,
        unet_channel_mults: tuple[int, ...]  = (1, 2, 2, 4, 4),
        attn_resolutions:   tuple[int, ...]  = (16,),
        input_size:         int              = 256,
        high_k_damping:     bool             = False,
        enc_hidden_ch:      int              = 32,
    ) -> None:
        super().__init__()
        self.medium_dim  = medium_dim
        self.coarse_dim  = coarse_dim
        self.input_size  = input_size

        # Compute injection resolutions from the U-Net stage hierarchy
        # Stages: input_size → /2 → /4 → /8 → /16
        # Stage 3 (0-indexed): input_size // 8 → medium injection
        # Stage 4             : input_size // 16 → coarse injection (bottleneck)
        self._inj_medium = input_size // 8   # 32 for 256×256
        self._inj_coarse = input_size // 16  # 16 for 256×256

        # ── Band encoders/decoders (shared with MSR-HINE when added) ──────
        self.enc_medium, self.dec_medium = build_encoder_decoder_pair(
            k_max=self.K_MEDIUM, latent_dim=medium_dim,
            in_size=input_size, hidden_ch=enc_hidden_ch,
        )
        self.enc_coarse, self.dec_coarse = build_encoder_decoder_pair(
            k_max=self.K_COARSE, latent_dim=coarse_dim,
            in_size=input_size, hidden_ch=enc_hidden_ch,
        )

        # ── Decoder emission heads: project decoder stage features → latent ─
        # These heads sit on the U-Net decoder output at the matching resolution
        # and produce the ADVANCED latent future.
        # Medium: receives dec_ch at the 32×32 decoder stage
        # Coarse:  receives dec_ch at the 16×16 decoder stage
        # We infer channel counts from the U-Net encoder's _ch_out list.
        n_stages   = len(unet_channel_mults)
        ch_per_stage = [unet_base_channels * m for m in unet_channel_mults]

        # Decoder stage i (coarsest→finest) outputs ch_per_stage[n-1-i].
        # Stage 0 of decoder = bottleneck (no upsampling yet);
        # Stage 1 → 16×16 output (coarse injection res);
        # Stage 2 → 32×32 output (medium injection res).
        self._dec_ch_medium = ch_per_stage[n_stages - 1 - 2]  # channels at 32×32 decode
        self._dec_ch_coarse = ch_per_stage[n_stages - 1 - 1]  # channels at 16×16 decode

        # Global-average-pool the decoder feature map → linear → latent
        self.emit_medium = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._dec_ch_medium, medium_dim),
        )
        self.emit_coarse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._dec_ch_coarse, coarse_dim),
        )

        # ── Shared U-Net (Invariant 10) ──────────────────────────────────────
        # injection_channels: {resolution: extra_channels}
        # The decoded prior is 1-channel (vorticity-like band-limited field).
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

        # Hook: intercept decoder feature maps for emission.
        # We attach a forward hook to the decoder to capture the feature maps
        # at the medium and coarse decoder stages.
        self._dec_feat_medium: Optional[Tensor] = None
        self._dec_feat_coarse: Optional[Tensor] = None
        self._register_decoder_hooks()

    # ── Decoder hook registration ────────────────────────────────────────────

    def _register_decoder_hooks(self) -> None:
        """Register forward hooks on the U-Net decoder to capture intermediate features.

        The decoder processes stages coarsest→finest. Stage 0 is the bottleneck
        (no hook needed), stage 1 outputs at the coarse resolution, stage 2 at
        the medium resolution.
        """
        decoder = self.unet.decoder

        def _make_hook(level: str):
            def _hook(module, input, output):
                if level == "coarse":
                    self._dec_feat_coarse = output
                else:
                    self._dec_feat_medium = output
            return _hook

        # stage_blocks[0] → 16×16 (coarse), stage_blocks[1] → 32×32 (medium)
        # Each element is an nn.ModuleList of ResBlocks (and optional attention).
        # Hook the last block in each stage to get the post-residual features.
        coarse_blocks = decoder.stage_blocks[0]
        medium_blocks = decoder.stage_blocks[1]

        # Register on the last module in each block list
        list(coarse_blocks.children())[-1].register_forward_hook(_make_hook("coarse"))
        list(medium_blocks.children())[-1].register_forward_hook(_make_hook("medium"))

    # ── State initialisation ─────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> HINEState:
        """Return zero-initialised latent-future state (used at rollout start)."""
        return HINEState(
            z_medium=torch.zeros(batch_size, self.medium_dim, device=device),
            z_coarse=torch.zeros(batch_size, self.coarse_dim, device=device),
        )

    # ── Core step ────────────────────────────────────────────────────────────

    def forward_with_state(
        self,
        omega: Tensor,
        state: HINEState,
    ) -> tuple[Tensor, HINEState]:
        """One HINE step: consume latent futures → predict Δω → emit advanced latents.

        The latent-future ladder is REGENERATED every call — no persistent GRU.

        Args:
            omega: Current vorticity [B, 1, H, W].
            state: Latent futures from the PREVIOUS step (or zeros for step 0).

        Returns:
            (omega_hat [B, 1, H, W], next_state with advanced latent futures)
        """
        B, _, H, W = omega.shape

        # ── Decode current latent futures to injection fields ─────────────
        # z_medium → 1-ch band-limited field at full res; U-Net injection harness
        # bilinearly resamples to the stage resolution.
        inj_medium = self.dec_medium(state.z_medium)   # [B, 1, H, W]
        inj_coarse = self.dec_coarse(state.z_coarse)   # [B, 1, H, W]

        injections = {
            self._inj_medium: inj_medium,
            self._inj_coarse: inj_coarse,
        }

        # ── U-Net forward (decoder hooks capture intermediate features) ───
        self._dec_feat_medium = None
        self._dec_feat_coarse = None

        omega_hat = self.unet(omega, injections=injections)  # [B, 1, H, W]

        # ── Emit advanced latent futures from decoder features ────────────
        # Staggered horizons: medium looks 1 step ahead, coarse looks 2 ahead.
        # The emission heads convert the decoder feature map into a latent that
        # will be used as the injection at the NEXT step.
        assert self._dec_feat_medium is not None, "Decoder hook (medium) did not fire"
        assert self._dec_feat_coarse is not None, "Decoder hook (coarse) did not fire"

        z_medium_adv = self.emit_medium(self._dec_feat_medium)  # [B, medium_dim]
        z_coarse_adv = self.emit_coarse(self._dec_feat_coarse)  # [B, coarse_dim]

        next_state = HINEState(z_medium=z_medium_adv, z_coarse=z_coarse_adv)
        return omega_hat, next_state

    def forward(self, omega: Tensor) -> Tensor:
        """Stateless one-step forward — compatible with the train.py model interface.

        Uses zero latent futures (step 0 approximation).  For proper multi-step
        rollout use forward_with_state() via rollout.rollout() which carries state.

        This method satisfies the interface: model(omega) → omega_hat.

        Args:
            omega: Current vorticity [B, 1, H, W].

        Returns:
            Predicted next vorticity [B, 1, H, W].
        """
        state = self.init_state(omega.shape[0], omega.device)
        omega_hat, _ = self.forward_with_state(omega, state)
        return omega_hat

    # ── Stateful interface (used by rollout.rollout) ─────────────────────────

    def warmup(
        self,
        omega_history: Tensor,
        state: HINEState,
    ) -> HINEState:
        """Teacher-forced warmup: build up latent-future estimates over W frames.

        HINE has no GRU to spin up, but warmup still builds useful latent-future
        estimates by running the full forward_with_state chain over the history.
        No loss is computed; the state is discarded after warmup.

        Args:
            omega_history: Warmup frames [B, W, 1, H, W].
            state:         Initial state (typically zeros).

        Returns:
            State after running through the warmup window.
        """
        W = omega_history.shape[1]
        with torch.no_grad():
            for t in range(W):
                _, state = self.forward_with_state(omega_history[:, t], state)
        return state

    def step(
        self,
        omega:  Tensor,
        state:  HINEState,
    ) -> tuple[Tensor, HINEState]:
        """One stateful step — interface used by rollout.rollout().

        Args:
            omega:  Current predicted vorticity [B, 1, H, W].
            state:  Latent-future state from the previous step.

        Returns:
            (omega_hat, next_state)
        """
        return self.forward_with_state(omega, state)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
