"""FNO one-step explicit autoregressive baseline (fno_1step config).

Wraps FNO2d as the cross-dataset reference baseline.  The backbone is
intentionally FNO, not U-Net — do NOT unify with hine/msr_hine (Invariant 10).

fno_1step semantics:  ω̂_{t+1} = ω_t + FNO2d(ω_t)

The FNO2d predicts the increment Δω and this module adds the input field.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from msr_hine.models.fno import FNO2d


class FNOBaseline(nn.Module):
    """One-step FNO autoregressive baseline.

    Forward: ω̂_{t+1} = ω_t + FNO2d(ω_t)

    Args:
        width:    FNO hidden channel width (default 64).
        modes:    Retained Fourier modes for both kx and ky (default 32).
        n_layers: Number of FNO blocks (default 4).
    """

    def __init__(self, width: int = 64, modes: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.fno = FNO2d(
            in_channels=1,
            out_channels=1,
            width=width,
            modes1=modes,
            modes2=modes,
            n_layers=n_layers,
        )

    def forward(self, omega: Tensor) -> Tensor:
        """Predict ω̂_{t+1} = ω_t + Δω.

        Args:
            omega: Current vorticity [B, 1, H, W].

        Returns:
            omega_hat: [B, 1, H, W].
        """
        return omega + self.fno(omega)

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
