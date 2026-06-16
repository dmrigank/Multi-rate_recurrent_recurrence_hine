"""Self-contained 2D Fourier Neural Operator (FNO) backbone.

Used ONLY for the fno_1step explicit autoregressive baseline.
Do NOT use this backbone for HINE or MSR-HINE (Invariant 10).

Spec (DESIGN.md §10):
  Layers:         4 FNO blocks
  Width:          64
  Retained modes: 32–48 (configurable)

Architecture:
  1. Lifting:     Conv1×1  in_ch → width
  2. N × FNOBlock: SpectralConv2d (Fourier integral) + Conv1×1 (residual), GeLU
  3. Projection:  Conv1×1  width → 128 → out_ch  (two-layer MLP in channel dim)

SpectralConv2d learns complex weights for the retained low-frequency modes in
the rfft2 half-spectrum.  Both the top-left (positive kx, positive ky) and
bottom-left (negative kx, positive ky) corners are retained so that the
operator is not restricted to positive-kx modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Spectral convolution
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """Global spectral convolution in Fourier space (2D rfft2 formulation).

    Keeps the lowest ``modes1 × modes2`` Fourier modes in both the positive-kx
    and negative-kx corners of the rfft2 half-spectrum, applies a learned
    complex linear mix, then IFFT back.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        modes1:       Retained modes along kx (rows of rfft2 output).
        modes2:       Retained modes along ky (cols of rfft2 output).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        modes1:       int,
        modes2:       int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, modes1, modes2)
        # Store real and imaginary parts as separate float32 parameters so that
        # GradScaler can unscale them (it does not support cfloat parameters).
        self.w_pos_r = nn.Parameter(scale * torch.randn(*shape))
        self.w_pos_i = nn.Parameter(scale * torch.randn(*shape))
        self.w_neg_r = nn.Parameter(scale * torch.randn(*shape))
        self.w_neg_i = nn.Parameter(scale * torch.randn(*shape))

    def _weight(self, r: Tensor, i: Tensor) -> Tensor:
        """Reconstruct complex weight from real/imaginary parts."""
        return torch.complex(r, i)

    def _mix(self, x_ft: Tensor, w: Tensor) -> Tensor:
        """Einsum: (batch, in_ch, m1, m2) × (in_ch, out_ch, m1, m2) → (batch, out_ch, m1, m2)."""
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Args:
            x: [B, C_in, H, W] real.

        Returns:
            [B, C_out, H, W] real.
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x if x.dtype != torch.float16 else x.float())

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        w_pos = self._weight(self.w_pos_r, self.w_pos_i)
        w_neg = self._weight(self.w_neg_r, self.w_neg_i)

        out_ft[:, :, : self.modes1, : self.modes2] = self._mix(
            x_ft[:, :, : self.modes1, : self.modes2], w_pos
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._mix(
            x_ft[:, :, -self.modes1 :, : self.modes2], w_neg
        )

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# FNO block
# ---------------------------------------------------------------------------

class FNOBlock(nn.Module):
    """Single FNO layer: SpectralConv2d + pointwise Conv1×1 + GeLU.

    y = GeLU( SpectralConv(x) + W(x) )

    where W is a bias-free 1×1 convolution (the "local" skip path).

    Args:
        width:  Channel width (in == out).
        modes1: Retained Fourier modes along kx.
        modes2: Retained Fourier modes along ky.
    """

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spec_conv = SpectralConv2d(width, width, modes1, modes2)
        self.w         = nn.Conv2d(width, width, 1)
        self.act       = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.spec_conv(x) + self.w(x))


# ---------------------------------------------------------------------------
# Full FNO
# ---------------------------------------------------------------------------

class FNO2d(nn.Module):
    """2D FNO: lifting → N FNO blocks → two-layer projection → increment Δω.

    Forward returns Δω (increment, NOT ω̂); the caller adds the input field.

    Args:
        in_channels:  Input channels (1 for vorticity).
        out_channels: Output channels (1 for Δω).
        width:        Hidden channel width (default 64).
        modes1:       Retained Fourier modes along kx (default 32).
        modes2:       Retained Fourier modes along ky (default 32).
        n_layers:     Number of FNO blocks (default 4).
    """

    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        width:        int = 64,
        modes1:       int = 32,
        modes2:       int = 32,
        n_layers:     int = 4,
    ) -> None:
        super().__init__()
        self.lift    = nn.Conv2d(in_channels, width, 1)
        self.blocks  = nn.ModuleList(
            [FNOBlock(width, modes1, modes2) for _ in range(n_layers)]
        )
        # Two-layer projection head (width → width//2 → out_channels)
        mid = max(width // 2, out_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(width, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return Δω (not ω̂).

        Args:
            x: [B, in_channels, H, W] real.

        Returns:
            delta_omega: [B, out_channels, H, W] real.
        """
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
