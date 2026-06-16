"""Fixed nested radial low-pass spectral truncation projectors.

Implements the three-level hierarchy from DESIGN.md §2:

    P^0 : |k| ≤ 85   (full resolved field, 256×256)
    P^1 : |k| ≤ 16   (medium band, stride s₁ = 2)
    P^2 : |k| ≤ 8    (coarse band, stride s₂ = 4)

Nesting: {k : |k|≤8} ⊂ {k : |k|≤16} ⊂ {k : |k|≤85}, so P^2 = P^2∘P^1 exactly.

All operators are:
  • Fixed (never learned).  Masks registered as nn.Module buffers.
  • Differentiable via torch.autograd (rfft2/irfft2 + float multiply).
  • Batch-safe: any number of leading dimensions are preserved.

Down / Up convention
--------------------
``down(x, k_max)``
    Spectrally truncate to |k| ≤ k_max then produce the compact spatial
    representation at size  n_out = 2*(k_max+1).  This is one larger than the
    strict Nyquist minimum (2*k_max) so that both kx=+k_max and kx=-k_max
    occupy distinct rows in the half-spectrum, avoiding the Nyquist collision.

``up(x_low, n)``
    Zero-pad the half-spectrum of a coarse field back to n×n with the
    (n/n_low)² amplitude correction so that every retained mode has the same
    physical amplitude before and after a down→up round-trip.

Public functional API (stateless):
    radial_mask_rfft2(n, k_max, device) → bool [n, n//2+1]
    project(x, k_max)                   → [..., H, W]
    down(x, k_max)                      → [..., 2*(k_max+1), 2*(k_max+1)]
    up(x_low, n)                        → [..., n, n]
    extract_band_coeffs(x, k_max)       → complex [..., M]
    check_nesting(n, device)            → True or raises AssertionError

Public nn.Module:
    SpectralPool(n, k_max)              — wraps the above with registered mask
    build_hierarchy(n)                  → {'fine', 'medium', 'coarse'} dict
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Wavenumber grid
# ---------------------------------------------------------------------------

def _k_radial_rfft2(n: int, device: torch.device) -> Tensor:
    """Return |k| = sqrt(kx²+ky²) for the rfft2 half-spectrum [n, n//2+1].

    kx ∈ {0,1,...,n//2,-(n//2-1),...,-1}  (rows, torch.fft.fftfreq convention)
    ky ∈ {0, 1, ..., n//2}                 (cols, non-negative half only)
    """
    kx = torch.fft.fftfreq(n, d=1.0 / n, device=device)      # [n]
    ky = torch.arange(n // 2 + 1, dtype=kx.dtype, device=device)
    kx2d = kx.unsqueeze(1).expand(n, n // 2 + 1)
    ky2d = ky.unsqueeze(0).expand(n, n // 2 + 1)
    return (kx2d ** 2 + ky2d ** 2).sqrt()                     # [n, n//2+1]


# ---------------------------------------------------------------------------
# Radial mask
# ---------------------------------------------------------------------------

def radial_mask_rfft2(n: int, k_max: int, device: torch.device) -> Tensor:
    """Boolean mask [n, n//2+1]: True for modes |k| ≤ k_max.

    The +0.5 tolerance ensures that modes exactly at the cutoff radius are
    always retained regardless of rounding in the radial distance.

    Invariant 5 — this mask is fixed and must never appear in an optimiser.

    Args:
        n: Spatial grid size (field is n×n).
        k_max: Maximum retained radial wavenumber.
        device: Target device.

    Returns:
        bool tensor [n, n//2+1].
    """
    return _k_radial_rfft2(n, device) <= float(k_max) + 0.5


def _fft_dtype(x: "Tensor") -> "Tensor":
    """Promote float16 → float32 for rfft2 (AMP-safe); leave float32/float64 unchanged."""
    if x.dtype == torch.float16:
        return x.float()
    return x


# ---------------------------------------------------------------------------
# project  P^l(x)
# ---------------------------------------------------------------------------

def project(x: Tensor, k_max: int) -> Tensor:
    """Radial low-pass projection P^l: zero all modes |k| > k_max.

    Fully differentiable: rfft2 and irfft2 have autograd support, and
    the mask enters as a fixed float multiplier (not boolean indexing).

    Args:
        x:     Real field [..., H, W] with H == W.
        k_max: Radial cutoff wavenumber.

    Returns:
        Band-limited field [..., H, W] — same shape, dtype, device as x.
    """
    *batch, H, W = x.shape
    if H != W:
        raise ValueError(f"project requires a square field, got {H}×{W}")
    n = H
    orig_dtype = x.dtype
    xhat = torch.fft.rfft2(_fft_dtype(x))   # rfft2 requires float32 (AMP-safe)
    mask = radial_mask_rfft2(n, k_max, x.device).to(xhat.dtype)
    return torch.fft.irfft2(xhat * mask, s=(n, n)).to(orig_dtype)


# ---------------------------------------------------------------------------
# down  (spectral crop → compact coarse grid)
# ---------------------------------------------------------------------------

def _coarse_size(k_max: int) -> int:
    """Return the coarse spatial size for a given cutoff: n_out = 2*(k_max+1).

    Using n_out = 2*(k_max+1) rather than 2*k_max avoids the Nyquist
    collision: both kx=+k_max and kx=-k_max are distinct rows in the
    half-spectrum of the n_out-point grid.
    """
    return 2 * (k_max + 1)


def down(x: Tensor, k_max: int) -> Tensor:
    """Spectrally truncate to |k| ≤ k_max and crop to a compact coarse grid.

    The coarse grid size is n_out = 2*(k_max+1).  The amplitude correction
    factor (n_out/n_in)² is applied so that every retained mode has the same
    physical amplitude on the coarse grid as on the original grid.

    This operation is lossless for fields already band-limited to k_max.
    For non-band-limited fields it is equivalent to project(x, k_max) followed
    by a lossless change of grid size.

    Args:
        x:     Real field [..., H, W] with H == W.
        k_max: Spectral cutoff; output size = 2*(k_max+1).

    Returns:
        Coarsened field [..., n_out, n_out].
    """
    *batch, H, W = x.shape
    if H != W:
        raise ValueError(f"down requires a square field, got {H}×{W}")
    n_in  = H
    n_out = _coarse_size(k_max)
    if n_out > n_in:
        raise ValueError(f"down: k_max={k_max} → n_out={n_out} > n_in={n_in}")

    # Zero out-of-band modes
    orig_dtype = x.dtype
    xhat = torch.fft.rfft2(_fft_dtype(x))   # rfft2 requires float32
    mask = radial_mask_rfft2(n_in, k_max, x.device).to(xhat.dtype)
    xhat = xhat * mask                                          # [..., n_in, n_in//2+1]

    # Crop to the n_out-point half-spectrum.
    # Positive-kx rows: 0 .. k_max  (k_pos rows)
    # Negative-kx rows: n_in-k_max .. n_in-1  (k_max rows)
    # Both groups map cleanly into the n_out-point grid because n_out//2 = k_max+1 > k_max.
    k_pos = k_max + 1
    k_neg = k_max
    C_out = n_out // 2 + 1

    xhat_small = torch.zeros(
        *batch, n_out, C_out, dtype=xhat.dtype, device=xhat.device
    )
    xhat_small[..., :k_pos, :]          = xhat[..., :k_pos, :C_out]
    xhat_small[..., n_out - k_neg :, :] = xhat[..., n_in - k_neg :, :C_out]

    # Amplitude correction: rfft2 coefficients of the n_in-grid are scaled by n_in²,
    # but irfft2 at n_out expects coefficients scaled by n_out².
    scale = (n_out / n_in) ** 2
    return torch.fft.irfft2(xhat_small * scale, s=(n_out, n_out)).to(orig_dtype)


# ---------------------------------------------------------------------------
# up  (zero-pad back to full resolution)
# ---------------------------------------------------------------------------

def up(x_low: Tensor, n: int) -> Tensor:
    """Upsample a coarse band-limited field to n×n by spectral zero-padding.

    Expects x_low to have been produced by ``down(·, k_max)`` for some k_max,
    so its spatial size is n_low = 2*(k_max+1).

    The inverse (n/n_low)² amplitude correction restores physical amplitudes.

    Args:
        x_low: Coarse real field [..., H', W'] with H' == W'.
        n:     Target spatial size (n × n); must satisfy n ≥ H'.

    Returns:
        Upsampled field [..., n, n].
    """
    *batch, H, W = x_low.shape
    if H != W:
        raise ValueError(f"up requires a square coarse field, got {H}×{W}")
    n_low = H
    if n < n_low:
        raise ValueError(f"up: target size {n} < input size {n_low}")

    # Infer k_max from the coarse grid size: n_low = 2*(k_max+1) → k_max = n_low//2 - 1
    k_max = n_low // 2 - 1
    k_pos = k_max + 1
    k_neg = k_max
    C_low = n_low // 2 + 1

    orig_dtype = x_low.dtype
    xhat_low = torch.fft.rfft2(_fft_dtype(x_low))   # rfft2 requires float32

    # Embed in the larger n-point half-spectrum, keeping the same kx/ky positions
    xhat_big = torch.zeros(
        *batch, n, n // 2 + 1, dtype=xhat_low.dtype, device=xhat_low.device
    )
    xhat_big[..., :k_pos, :C_low]       = xhat_low[..., :k_pos, :]
    xhat_big[..., n - k_neg :, :C_low]  = xhat_low[..., n_low - k_neg :, :]

    # Inverse amplitude correction
    scale = (n / n_low) ** 2
    return torch.fft.irfft2(xhat_big * scale, s=(n, n)).to(orig_dtype)


# ---------------------------------------------------------------------------
# extract_band_coeffs
# ---------------------------------------------------------------------------

def extract_band_coeffs(x: Tensor, k_max: int) -> Tensor:
    """Return the non-redundant complex rfft2 coefficients for |k| ≤ k_max.

    Produces a flat complex vector of the M retained modes.  Useful for
    feeding band content into a latent encoder without explicit spatial convs.

    Note: boolean masking is not differentiable in general.  Use this for
    inference / encoder inputs, not directly in a loss that needs gradients
    through the indexing operation (use ``project`` + a linear layer instead).

    Args:
        x:     Real field [..., H, W] with H == W.
        k_max: Radial cutoff.

    Returns:
        Complex tensor [..., M] of M retained rfft2 coefficients.
    """
    *batch, H, W = x.shape
    if H != W:
        raise ValueError(f"extract_band_coeffs requires square field, got {H}×{W}")
    n = H
    xhat = torch.fft.rfft2(_fft_dtype(x))   # rfft2 requires float32
    mask = radial_mask_rfft2(n, k_max, x.device)
    return xhat[..., mask]


# ---------------------------------------------------------------------------
# check_nesting  (functional verification)
# ---------------------------------------------------------------------------

def check_nesting(
    n: int = 256,
    device: torch.device | None = None,
    k_fine:   int   = 16,
    k_coarse: int   = 8,
    atol:     float = 1e-5,
) -> bool:
    """Verify P^coarse == P^coarse ∘ P^fine on a random batch.

    Since {|k|≤k_coarse} ⊂ {|k|≤k_fine}, applying P^fine first is a no-op
    for P^coarse.  The identity must hold to machine precision.

    Args:
        n:        Grid size.
        device:   Defaults to CPU.
        k_fine:   Cutoff for the finer level (P^1, default 16).
        k_coarse: Cutoff for the coarser level (P^2, default 8).
        atol:     Absolute tolerance.

    Returns:
        True if the identity holds.

    Raises:
        AssertionError: if the maximum absolute difference exceeds atol.
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(0)
    x = torch.randn(2, n, n, device=device)

    lhs = project(x, k_coarse)
    rhs = project(project(x, k_fine), k_coarse)

    diff = (lhs - rhs).abs().max().item()
    assert diff <= atol, (
        f"Nesting P^{k_coarse} == P^{k_coarse}∘P^{k_fine} violated: "
        f"max |diff| = {diff:.2e} > atol={atol}"
    )
    return True


# ---------------------------------------------------------------------------
# SpectralPool  nn.Module
# ---------------------------------------------------------------------------

class SpectralPool(nn.Module):
    """Fixed radial low-pass projector as an nn.Module.

    The boolean mask is a registered buffer:
      • Moves with .to(device) / .cuda()
      • Persists in state_dict (non-parameter buffer)
      • Never appears in optimizer.parameters()  (Invariant 5)

    Args:
        n:     Full spatial grid size expected as input.
        k_max: Radial cutoff wavenumber.
    """

    def __init__(self, n: int, k_max: int) -> None:
        super().__init__()
        self.n     = n
        self.k_max = k_max
        self.n_out = _coarse_size(k_max)

        # bool [n, n//2+1] — registered buffer, never a parameter
        self.register_buffer(
            "mask",
            radial_mask_rfft2(n, k_max, torch.device("cpu")),
        )

    # --- core operations ---

    def project(self, x: Tensor) -> Tensor:
        """P^l(x): band-limited field at full resolution [..., n, n]."""
        xhat = torch.fft.rfft2(_fft_dtype(x))   # rfft2 requires float32
        xhat = xhat * self.mask.to(xhat.dtype)
        return torch.fft.irfft2(xhat, s=(self.n, self.n)).to(x.dtype)

    def down(self, x: Tensor) -> Tensor:
        """Down^l(x): coarsened field [..., n_out, n_out]."""
        return down(x, self.k_max)

    def up(self, x_low: Tensor) -> Tensor:
        """Up^l(x_low): upsample back to [..., n, n]."""
        return up(x_low, self.n)

    def forward(self, x: Tensor) -> Tensor:
        """Alias for project(x)."""
        return self.project(x)

    def extra_repr(self) -> str:
        return f"n={self.n}, k_max={self.k_max}, n_out={self.n_out}"


# ---------------------------------------------------------------------------
# Canonical hierarchy for 256×256
# ---------------------------------------------------------------------------

K_FINE   = 85   # full resolved band (2/3 dealiasing of 128)
K_MEDIUM = 16   # medium recurrent level  (stride 2)
K_COARSE = 8    # coarse recurrent level  (stride 4)


def build_hierarchy(n: int = 256) -> dict[str, SpectralPool]:
    """Construct the three canonical SpectralPool projectors.

    Returns:
        {'fine': P^0(k≤85), 'medium': P^1(k≤16), 'coarse': P^2(k≤8)}
    """
    return {
        "fine":   SpectralPool(n, K_FINE),
        "medium": SpectralPool(n, K_MEDIUM),
        "coarse": SpectralPool(n, K_COARSE),
    }
