"""Feature-wise Linear Modulation (FiLM) for injecting vector RNN states into the U-Net.

Each recurrent hidden state h^l produces per-channel (γ, β) that modulate the
feature maps at its matching U-Net decoder stage (DESIGN.md §4, fixes P7).

Invariant: NEVER concatenate a raw hidden vector to a spatial feature map.
           h → (γ, β) via a learned MLP → applied as γ*x + β.

FiLM is applied after the residual blocks in each decoder stage.  The U-Net
decoder already accepts `film_params: dict[resolution, (γ, β)]`; the MSRHINE
model assembles that dict from the FiLMGenerators defined here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FiLMGenerator(nn.Module):
    """Map a GRU hidden state h^l [B, hidden_dim] to FiLM parameters (γ, β) [B, n_channels].

    Architecture: linear → SiLU → linear × 2 (one head each for γ and β).
    γ is initialised near 1 and β near 0 so that FiLM starts as near-identity.

    Args:
        hidden_dim: Dimensionality of h^l.
        n_channels: Number of feature-map channels to modulate at this stage.
        gamma_mode: ``"direct"`` for unconstrained legacy FiLM or
            ``"bounded_residual"`` for γ = 1 + scale*tanh(raw_γ).
        gamma_scale: Maximum absolute deviation from one in bounded mode.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_channels: int,
        gamma_mode: str = "direct",
        gamma_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if gamma_mode not in {"direct", "bounded_residual"}:
            raise ValueError(
                "gamma_mode must be 'direct' or 'bounded_residual', "
                f"got {gamma_mode!r}"
            )
        if gamma_scale <= 0:
            raise ValueError(f"gamma_scale must be positive, got {gamma_scale}")

        self.gamma_mode = gamma_mode
        self.gamma_scale = float(gamma_scale)
        mid = max(hidden_dim, n_channels)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.SiLU(),
        )
        self.gamma_head = nn.Linear(mid, n_channels)
        self.beta_head  = nn.Linear(mid, n_channels)

        # Both parameterisations initialise to exact identity modulation.
        nn.init.zeros_(self.gamma_head.weight)
        if gamma_mode == "direct":
            nn.init.ones_(self.gamma_head.bias)
        else:
            nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Produce per-channel (γ, β) from the hidden state.

        Args:
            h: Hidden state [B, hidden_dim].

        Returns:
            (gamma [B, n_channels], beta [B, n_channels]).
        """
        shared = self.shared(h)
        raw_gamma = self.gamma_head(shared)
        if self.gamma_mode == "bounded_residual":
            gamma = 1.0 + self.gamma_scale * torch.tanh(raw_gamma)
        else:
            gamma = raw_gamma
        return gamma, self.beta_head(shared)


def apply_film(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """Apply FiLM modulation: out = γ * x + β, broadcast over spatial dims.

    Args:
        x:     Feature map [B, C, H, W].
        gamma: Scale  [B, C] or [B, C, 1, 1].
        beta:  Shift  [B, C] or [B, C, 1, 1].

    Returns:
        Modulated feature map [B, C, H, W].
    """
    if gamma.dim() == 2:
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
    if beta.dim() == 2:
        beta = beta.unsqueeze(-1).unsqueeze(-1)
    return gamma * x + beta
