"""U-Net one-step explicit autoregressive baseline (unet_1step config).

Mirrors FNOBaseline exactly but uses the shared U-Net backbone instead of FNO.
Intended as a direct apples-to-apples comparison with fno_1step using the same
backbone architecture as HINE and MSR-HINE.

unet_1step semantics:  ω̂_{t+1} = ω_t + UNet(ω_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from msr_hine.models.unet import UNet


class UNetBaseline(nn.Module):
    """One-step U-Net autoregressive baseline.

    Forward: ω̂_{t+1} = ω_t + UNet(ω_t)

    Args:
        base_channels:   U-Net base channel count (default 64).
        channel_mults:   Per-stage channel multipliers.
        n_res_blocks:    Residual blocks per stage.
        groups:          GroupNorm group count.
        attn_resolutions: Spatial sizes with self-attention.
        input_size:      Spatial grid size H=W (default 256).
        high_k_damping:  Apply fixed spectral high-k damping to Δω.
    """

    def __init__(
        self,
        base_channels:    int             = 64,
        channel_mults:    tuple[int, ...] = (1, 2, 2, 4, 4),
        n_res_blocks:     int             = 2,
        groups:           int             = 8,
        attn_resolutions: tuple[int, ...] = (16,),
        input_size:       int             = 256,
        high_k_damping:   bool            = False,
    ) -> None:
        super().__init__()
        self.unet = UNet(
            in_channels      = 1,
            base_channels    = base_channels,
            channel_mults    = channel_mults,
            n_res_blocks     = n_res_blocks,
            groups           = groups,
            attn_resolutions = attn_resolutions,
            injection_channels = None,
            input_size       = input_size,
            high_k_damping   = high_k_damping,
        )

    def forward(self, omega: Tensor) -> Tensor:
        """Predict ω̂_{t+1} = ω_t + Δω.

        Args:
            omega: Current vorticity [B, 1, H, W].

        Returns:
            omega_hat: [B, 1, H, W].
        """
        return omega + self.unet(omega)

    def rollout(self, omega_init: Tensor, n_steps: int) -> Tensor:
        """Autoregressive rollout for n_steps.

        Args:
            omega_init: Starting vorticity [B, 1, H, W].
            n_steps:    Number of steps.

        Returns:
            Predictions [B, n_steps, 1, H, W].
        """
        preds: list[Tensor] = []
        omega = omega_init
        for _ in range(n_steps):
            omega = self.forward(omega)
            preds.append(omega)
        return torch.stack(preds, dim=1)   # [B, n_steps, 1, H, W]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
