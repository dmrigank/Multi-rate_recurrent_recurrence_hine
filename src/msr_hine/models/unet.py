"""Modern 2D U-Net backbone shared by HINE and MSR-HINE (Invariant 10).

Spec (DESIGN.md §4, §10, CLAUDE.md §3):
  Stages:         256→128→64→32→16  (5 stages, 4 downsampling steps)
  Channels:       base=64, mults=[1,2,2,4,4] → [64,128,128,256,256]
  Per stage:      2 residual blocks (GroupNorm+SiLU), optional self-attention
  Attention:      always at 16×16; optionally also at 32×32 (config-gated)
  Skip conns:     encoder feature map concatenated at each decoder stage
  Latent injection (HINE mechanism):
      medium prior (|k|≤16) → 32×32 encoder stage
      coarse  prior (|k|≤8)  → 16×16 encoder stage
  FiLM conditioning:
      per-stage (γ,β) applied after residual blocks in the decoder
  Output:         Δω increment; caller computes ω̂ = ω + Δω

Internal geometry for H=256 input:
  stage 0:  256×256,  64ch   (no injection)
  stage 1:  128×128, 128ch   (no injection)
  stage 2:   64×64,  128ch   (no injection)
  stage 3:   32×32,  256ch   ← medium injection
  stage 4:   16×16,  256ch   ← coarse injection + bottleneck attention

The encoder produces feature maps at the END of each stage (after residual
blocks, before downsampling), stored as skip connections coarsest-first.
Decoder uses those skips in reverse order (finest first from the skip list).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-norm residual block: GN → SiLU → Conv → GN → SiLU → Conv + skip.

    If in_channels != out_channels a 1×1 conv is used for the residual path.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        groups:       GroupNorm group count (must divide both channel counts).
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip  = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention over spatial positions.

    Flattens (H,W) → tokens, runs nn.MultiheadAttention, reshapes back.
    Used at the 16×16 (and optionally 32×32) stage.

    Args:
        channels:  Feature channel count (must be divisible by num_heads).
        num_heads: Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        # Flatten spatial dims → token sequence [B, H*W, C]
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h   # residual


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _make_stage_blocks(
    in_ch: int,
    out_ch: int,
    n_res: int,
    groups: int,
    use_attn: bool,
    attn_heads: int = 4,
) -> nn.ModuleList:
    """Build [ResBlock, ..., optional SelfAttention2D] for one stage."""
    blocks: list[nn.Module] = []
    for i in range(n_res):
        blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, groups))
    if use_attn:
        blocks.append(SelfAttention2D(out_ch, attn_heads))
    return nn.ModuleList(blocks)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class UNetEncoder(nn.Module):
    """Downsampling arm: stem conv → N stages, each followed by strided-conv.

    Injection interface: at stages whose spatial resolution is in
    ``injection_resolutions``, the feature map is concatenated with an
    externally supplied tensor before processing, via a learnable 1×1 conv
    that merges channels back to the stage channel count.

    Args:
        in_channels:           Input field channels (1 for scalar vorticity).
        base_channels:         Channel count at the first stage.
        channel_mults:         Per-stage channel multipliers (length = n_stages).
        n_res_blocks:          Residual blocks per stage.
        groups:                GroupNorm group count.
        attn_resolutions:      Set of spatial sizes at which to use attention.
        injection_resolutions: Set of spatial sizes that accept extra injection.
        injection_channels:    Dict {resolution: extra_channels} for 1×1 merge conv.
        input_size:            Spatial size of the input field (H = W).
    """

    def __init__(
        self,
        in_channels:           int = 1,
        base_channels:         int = 64,
        channel_mults:         tuple[int, ...] = (1, 2, 2, 4, 4),
        n_res_blocks:          int = 2,
        groups:                int = 8,
        attn_resolutions:      tuple[int, ...] = (16,),
        injection_resolutions: tuple[int, ...] = (32, 16),
        injection_channels:    dict[int, int] | None = None,
        input_size:            int = 256,
    ) -> None:
        super().__init__()
        self.n_stages = len(channel_mults)
        self.attn_resolutions = set(attn_resolutions)
        self.injection_resolutions = set(injection_resolutions)
        injection_channels = injection_channels or {}

        # Stem: project to base_channels
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Per-stage channel counts
        ch_in_list  = [base_channels * channel_mults[0]]
        ch_out_list = [base_channels * m for m in channel_mults]
        # stage 0 input is base_channels (from stem)
        ch_in_list = [base_channels] + [base_channels * channel_mults[i] for i in range(self.n_stages - 1)]

        # Compute the spatial size at the START of each stage
        # Stage 0 starts at input_size; subsequent stages are halved each time.
        sizes = [input_size // (2 ** i) for i in range(self.n_stages)]
        self._stage_sizes = sizes  # for injection look-up

        # Optional injection merge convs  {resolution: Conv2d}
        # The merge conv sits BEFORE the stage's first ResBlock, so it sees
        # ch_in_list[si] + extra_ch channels and outputs ch_in_list[si] so
        # the subsequent ResBlocks receive the channel count they were built for.
        self.inject_convs = nn.ModuleDict()
        for res in injection_resolutions:
            if res in sizes:
                si = sizes.index(res)
                pre_ch  = ch_in_list[si]    # channels entering this stage
                extra_ch = injection_channels.get(res, 1)
                self.inject_convs[str(res)] = nn.Conv2d(
                    pre_ch + extra_ch, pre_ch, 1
                )

        # Stage blocks and downsampling convs
        self.stage_blocks = nn.ModuleList()
        self.downsample   = nn.ModuleList()
        for i in range(self.n_stages):
            use_attn = (sizes[i] in self.attn_resolutions)
            self.stage_blocks.append(
                _make_stage_blocks(ch_in_list[i], ch_out_list[i], n_res_blocks, groups, use_attn)
            )
            # Downsample with stride-2 conv except at the last stage
            if i < self.n_stages - 1:
                self.downsample.append(
                    nn.Conv2d(ch_out_list[i], ch_out_list[i], 4, stride=2, padding=1)
                )
            else:
                self.downsample.append(nn.Identity())

        self._ch_out = ch_out_list   # expose for decoder

    def forward(
        self,
        x: Tensor,
        injections: Optional[dict[int, Tensor]] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode x with optional stage-level injection.

        Args:
            x:          [B, in_channels, H, W]
            injections: {spatial_resolution: tensor [B, extra_ch, res, res]}
                        Concatenated at the matching stage before residual blocks.

        Returns:
            (h [B, C_last, H/16, W/16],
             skips [coarsest→finest]:  list of length n_stages)
        """
        injections = injections or {}
        h = self.stem(x)
        skips: list[Tensor] = []

        for i, (blocks, down) in enumerate(zip(self.stage_blocks, self.downsample)):
            res = self._stage_sizes[i]
            # Optional injection at this resolution
            if res in injections and str(res) in self.inject_convs:
                inj = injections[res]
                # Resize injection to current spatial size if needed
                if inj.shape[-1] != h.shape[-1]:
                    inj = F.interpolate(inj, size=(h.shape[-2], h.shape[-1]), mode="bilinear", align_corners=False)
                h = torch.cat([h, inj], dim=1)
                h = self.inject_convs[str(res)](h)

            for block in blocks:
                h = block(h)
            skips.append(h)          # save skip before downsampling
            h = down(h)

        # skips[0] = largest stage feature, skips[-1] = bottleneck stage feature
        # Reverse so index 0 is coarsest (for decoder to consume first)
        return h, list(reversed(skips))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """Upsampling arm: upsample + concat skip → residual blocks → optional FiLM.

    FiLM conditioning: the caller supplies per-stage (γ, β) tensors keyed by
    the stage's output spatial resolution.  Applied after the residual blocks.

    Args:
        ch_out_list:      Per-stage channel counts from the encoder
                          (length = n_stages, index 0 = first/largest stage).
        n_res_blocks:     Residual blocks per stage.
        groups:           GroupNorm group count.
        attn_resolutions: Spatial sizes with self-attention.
        input_size:       Full spatial size (H = W) of the final output.
    """

    def __init__(
        self,
        ch_out_list:      list[int],
        n_res_blocks:     int = 2,
        groups:           int = 8,
        attn_resolutions: tuple[int, ...] = (16,),
        input_size:       int = 256,
    ) -> None:
        super().__init__()
        n_stages = len(ch_out_list)
        self.attn_resolutions = set(attn_resolutions)

        # Decoder processes stages from coarsest→finest (reverse of encoder).
        # After the bottleneck (index 0 of reversed ch_out_list), each stage:
        #   upsample → concat skip (doubles channels) → ResBlocks → optional attn
        # Stage i of the decoder merges the bottleneck/previous with skip i.
        # ch_in for stage i = ch_out_list[i] (from upsample) + ch_out_list[i] (skip)
        #   = 2 * ch_out_list[i] ... except that the *skip* comes from the matching
        #   encoder stage which has the same channel count.

        # Spatial sizes: coarsest first
        # input_size / 2^(n_stages-1), ..., input_size/2, input_size
        sizes_coarse_first = [input_size // (2 ** (n_stages - 1 - i)) for i in range(n_stages)]
        self._stage_sizes = sizes_coarse_first  # coarsest→finest

        self.upsample   = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()

        for i in range(n_stages - 1):
            # i=0: bottleneck (no upsample needed, handled by merge below)
            # We upsample from stage i to stage i+1
            ch_cur  = ch_out_list[n_stages - 1 - i]   # coarsest ch first
            ch_next = ch_out_list[n_stages - 1 - i - 1]

            self.upsample.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch_cur, ch_next, 3, padding=1),
                )
            )
            # After concat with skip: in_ch = ch_next + ch_next = 2*ch_next
            # (skip has same channel count as the target stage's output)
            use_attn = (sizes_coarse_first[i + 1] in self.attn_resolutions)
            self.stage_blocks.append(
                _make_stage_blocks(ch_next * 2, ch_next, n_res_blocks, groups, use_attn)
            )

        self._out_ch = ch_out_list[0]   # finest stage output channels

    def forward(
        self,
        h: Tensor,
        skips: list[Tensor],
        film_params: Optional[dict[int, tuple[Tensor, Tensor]]] = None,
    ) -> Tensor:
        """Decode h using skip connections and optional FiLM.

        Args:
            h:           Bottleneck tensor [B, C_coarse, H', W'].
            skips:       Encoder skips, coarsest first (skip[0] matches h spatially,
                         skip[-1] is the finest/full-resolution skip).
            film_params: {spatial_resolution: (gamma [B,C], beta [B,C])}

        Returns:
            Feature map [B, out_ch, H_in, W_in] at the full input resolution.
        """
        film_params = film_params or {}
        n_up = len(self.upsample)

        # ── FiLM at the bottleneck BEFORE any upsampling ──────────────────────
        # _stage_sizes[0] is the coarsest resolution (e.g. 16 for 256-input 5-stage).
        # The decoder loop only checks _stage_sizes[i+1] (i=0..n-2), so resolution
        # _stage_sizes[0] would never be reached otherwise — this was Bug 1.
        res_btl = self._stage_sizes[0]
        if res_btl in film_params:
            gamma, beta = film_params[res_btl]
            h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)

        for i, (up, blocks) in enumerate(zip(self.upsample, self.stage_blocks)):
            h = up(h)
            skip = skips[i + 1]   # skip[0] was the bottleneck stage itself; skip[1..] are finer
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h)

            # FiLM: apply at the output resolution of this decoder stage
            res = self._stage_sizes[i + 1]
            if res in film_params:
                gamma, beta = film_params[res]
                # gamma, beta: [B, C]; broadcast over spatial
                h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)

        return h


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Full U-Net: encoder + decoder + head → Δω increment.

    Shared backbone for HINE and MSR-HINE (Invariant 10).
    Accepts optional latent injection and FiLM conditioning.

    The caller adds the increment to the input field:
        ω̂_{t+1} = ω_t + UNet(ω_t, injections, film_params)

    Args:
        in_channels:           Input channels (1 for scalar vorticity).
        base_channels:         Channel count at the first stage.
        channel_mults:         Per-stage channel multipliers.
        n_res_blocks:          Residual blocks per stage.
        groups:                GroupNorm group count.
        attn_resolutions:      Spatial sizes with self-attention (default: 16 only).
        injection_channels:    {resolution: extra_channels} for injection merge.
        input_size:            Full spatial size of the input (H = W).
        high_k_damping:        If True apply a fixed spectral high-k damping to Δω.
        damping_k_frac:        Fraction of Nyquist above which damping is applied.
    """

    def __init__(
        self,
        in_channels:        int = 1,
        base_channels:      int = 64,
        channel_mults:      tuple[int, ...] = (1, 2, 2, 4, 4),
        n_res_blocks:       int = 2,
        groups:             int = 8,
        attn_resolutions:   tuple[int, ...] = (16,),
        injection_channels: dict[int, int] | None = None,
        input_size:         int = 256,
        high_k_damping:     bool = False,
        damping_k_frac:     float = 0.65,
    ) -> None:
        super().__init__()
        self.high_k_damping = high_k_damping
        self.damping_k_frac = damping_k_frac

        injection_resolutions = tuple(injection_channels.keys()) if injection_channels else (32, 16)

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            n_res_blocks=n_res_blocks,
            groups=groups,
            attn_resolutions=attn_resolutions,
            injection_resolutions=injection_resolutions,
            injection_channels=injection_channels or {},
            input_size=input_size,
        )

        ch_out_list = self.encoder._ch_out   # per-stage channels, finest-first

        self.decoder = UNetDecoder(
            ch_out_list=ch_out_list,
            n_res_blocks=n_res_blocks,
            groups=groups,
            attn_resolutions=attn_resolutions,
            input_size=input_size,
        )

        # Head: map finest-stage channels → 1-channel increment
        self.head = nn.Sequential(
            nn.GroupNorm(groups, ch_out_list[0]),
            nn.SiLU(),
            nn.Conv2d(ch_out_list[0], in_channels, 1),
        )

    def forward(
        self,
        omega: Tensor,
        injections:  Optional[dict[int, Tensor]] = None,
        film_params: Optional[dict[int, tuple[Tensor, Tensor]]] = None,
    ) -> Tensor:
        """Predict ω_{t+1} = ω_t + Δω.

        Args:
            omega:       Current vorticity [B, 1, H, W].
            injections:  {resolution: tensor} decoded band-limited priors,
                         injected at the 32×32 and 16×16 encoder stages.
            film_params: {resolution: (gamma [B,C], beta [B,C])} FiLM per decoder stage.

        Returns:
            omega_hat: [B, 1, H, W]  (= omega + delta_omega)
        """
        h, skips = self.encoder(omega, injections=injections)
        h = self.decoder(h, skips, film_params=film_params)
        delta = self.head(h)

        if self.high_k_damping:
            delta = self._damp_high_k(delta)

        return omega + delta

    def _damp_high_k(self, delta: Tensor) -> Tensor:
        """Apply a fixed smooth spectral taper to high-k modes of Δω."""
        B, C, H, W = delta.shape
        dhat = torch.fft.rfft2(delta if delta.dtype != torch.float16 else delta.float())
        k_ny = H // 2
        k_cut = int(self.damping_k_frac * k_ny)

        # Build a 1D raised-cosine taper
        ky = torch.arange(H // 2 + 1, device=delta.device, dtype=delta.dtype)
        kx = torch.fft.fftfreq(H, d=1.0 / H, device=delta.device).abs()
        k_rad = (kx.unsqueeze(1) ** 2 + ky.unsqueeze(0) ** 2).sqrt()
        # Taper: 1 for k <= k_cut, smooth rolloff to 0 at k_ny
        t = torch.ones_like(k_rad)
        band = (k_rad > k_cut) & (k_rad < k_ny)
        t[band] = 0.5 * (1 + torch.cos(torch.pi * (k_rad[band] - k_cut) / (k_ny - k_cut)))
        t[k_rad >= k_ny] = 0.0

        dhat = dhat * t.unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft2(dhat, s=(H, W)).to(delta.dtype)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
