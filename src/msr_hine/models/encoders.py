"""Band-limited latent encoders and decoders (DESIGN.md §3).

Each level l has:
  E^l : P^l(ω) → z^l   compact latent vector  [B, latent_dim]
  D^l : z^l   → band-limited field at full resolution  [B, 1, H, W]

The field is spectrally projected before encoding (P^l applied inside BandEncoder),
so the encoder only ever sees frequencies within its band.  This is the mechanism
that fixes P4 ("coarse = lower-frequency" by construction).

Latent dimensions (DESIGN.md §10):
    Medium (|k|≤16): latent_dim = 128
    Coarse  (|k|≤8): latent_dim = 64

Architecture choices (kept small for the reduced-data regime):
  BandEncoder:
    spectral project → small conv stack (2 × Conv3×3/BN/ReLU, stride-2)
    → adaptive avg pool → linear   →  z [B, d]
  BandDecoder:
    linear → reshape → small transposed-conv stack (2 × ConvTranspose/BN/ReLU)
    → Conv1×1 → full-resolution band-limited field  [B, 1, H, W]
    The decoder output is spectrally projected (P^l) to guarantee it is
    band-limited regardless of decoder drift.

Both modules are shared by HINE and MSR-HINE.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from msr_hine.spectral.truncation import project


# ---------------------------------------------------------------------------
# BandEncoder  E^l
# ---------------------------------------------------------------------------

class BandEncoder(nn.Module):
    """Encode a vorticity field to a compact band-limited latent vector.

    Spectral truncation P^l is applied inside the encoder so the network
    only processes the relevant frequency band.

    Args:
        k_max:      Maximum radial wavenumber for this level (16 or 8).
        latent_dim: Output latent dimensionality (128 or 64).
        in_size:    Full spatial resolution of the input field (default 256).
        hidden_ch:  Intermediate CNN channel count.
    """

    def __init__(
        self,
        k_max:      int,
        latent_dim: int,
        in_size:    int = 256,
        hidden_ch:  int = 32,
    ) -> None:
        super().__init__()
        self.k_max      = k_max
        self.latent_dim = latent_dim
        self.in_size    = in_size

        # Two stride-2 conv blocks: in_size → in_size/4 spatially
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        # Global average pool → hidden_ch features
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(hidden_ch, latent_dim)

    def forward(self, omega: Tensor) -> Tensor:
        """Encode ω to z^l.

        Args:
            omega: Vorticity field [B, 1, H, W] (full resolution).

        Returns:
            Latent vector [B, latent_dim].
        """
        # Project to band first — encoder only sees |k| ≤ k_max
        x = project(omega, self.k_max)         # [B, 1, H, W], band-limited
        x = self.conv(x)                        # [B, hidden_ch, H/4, W/4]
        x = self.pool(x).flatten(1)             # [B, hidden_ch]
        return self.linear(x)                   # [B, latent_dim]


# ---------------------------------------------------------------------------
# BandDecoder  D^l
# ---------------------------------------------------------------------------

class BandDecoder(nn.Module):
    """Decode a compact latent vector to a band-limited spatial field.

    The output is projected through P^l to guarantee band-limitedness.
    The U-Net injection harness bilinearly resamples this to the target
    stage resolution before concatenation.

    Args:
        latent_dim:  Input latent dimensionality (128 or 64).
        k_max:       Maximum radial wavenumber; used for spectral projection.
        out_channels: Output channels (1 for scalar vorticity).
        out_size:    Full spatial output resolution (default 256).
        hidden_ch:   Intermediate CNN channel count.
    """

    def __init__(
        self,
        latent_dim:   int,
        k_max:        int,
        out_channels: int = 1,
        out_size:     int = 256,
        hidden_ch:    int = 32,
    ) -> None:
        super().__init__()
        self.k_max    = k_max
        self.out_size = out_size

        # Start from a 4×4 spatial map and upsample
        self.stem_size = 4
        self.linear = nn.Linear(latent_dim, hidden_ch * self.stem_size ** 2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch, hidden_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        # 1×1 head — upsample to out_size separately
        self.head = nn.Conv2d(hidden_ch, out_channels, 1)
        self._hidden_ch = hidden_ch

    def forward(self, z: Tensor) -> Tensor:
        """Decode z^l to a band-limited field.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Band-limited field [B, out_channels, out_size, out_size].
        """
        B = z.shape[0]
        x = self.linear(z)                               # [B, hidden_ch * 4 * 4]
        x = x.view(B, self._hidden_ch, self.stem_size, self.stem_size)
        x = self.deconv(x)                               # [B, hidden_ch, 16, 16]
        x = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        x = self.head(x)                                 # [B, out_ch, out_size, out_size]
        # Guarantee band-limitedness (band defined by k_max)
        return project(x, self.k_max)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_encoder_decoder_pair(
    k_max:      int,
    latent_dim: int,
    in_size:    int = 256,
    hidden_ch:  int = 32,
) -> tuple[BandEncoder, BandDecoder]:
    """Build a matched (encoder, decoder) pair for one spectral level.

    Args:
        k_max:      Radial wavenumber cutoff.
        latent_dim: Latent dimensionality.
        in_size:    Full spatial resolution.
        hidden_ch:  CNN hidden channels.

    Returns:
        (BandEncoder, BandDecoder)
    """
    enc = BandEncoder(k_max=k_max, latent_dim=latent_dim,
                      in_size=in_size, hidden_ch=hidden_ch)
    dec = BandDecoder(latent_dim=latent_dim, k_max=k_max,
                      out_size=in_size, hidden_ch=hidden_ch)
    return enc, dec
