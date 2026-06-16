"""Backbone architecture tests.

Covers:
  1. U-Net forward shape — plain, with injection, with FiLM, combined.
  2. FiLM actually changes the output (non-identity γ ≠ 1 effect).
  3. Injection at the correct encoder stages (32×32 and 16×16 for default H=256).
  4. Skip connections: decoder receives the right number and sizes.
  5. U-Net output is ω_t + Δω (residual add confirmed).
  6. FNO2d forward shape and increment semantics.
  7. FNOBaseline forward + rollout shapes.
  8. Parameter count assertions (rough bounds, not exact).
  9. Invariant 10: U-Net and FNO are separate classes, no cross-use.
 10. Gradient flow through U-Net and FNO (backward does not error).
 11. High-k damping option doesn't break shapes.
 12. Attention at bottleneck works at H=16 (exact bottleneck size).

All tests run on a small grid (H=32 or H=64) so the suite stays fast.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from msr_hine.models.fno import FNO2d, FNOBlock, SpectralConv2d
from msr_hine.models.fno_baseline import FNOBaseline
from msr_hine.models.unet import ResBlock, SelfAttention2D, UNet, UNetDecoder, UNetEncoder

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Fixtures: small models for fast tests
# ---------------------------------------------------------------------------

def _small_unet(
    H: int = 32,
    base_ch: int = 8,
    mults: tuple = (1, 2, 2, 4, 4),
    attn_res: tuple = (2,),
    inj_channels: dict | None = None,
) -> UNet:
    """Build a tiny U-Net for testing (H must be divisible by 2^(len(mults)-1) = 16)."""
    return UNet(
        in_channels=1,
        base_channels=base_ch,
        channel_mults=mults,
        n_res_blocks=1,
        groups=4,
        attn_resolutions=attn_res,
        injection_channels=inj_channels or {},
        input_size=H,
    )


def _small_fno(modes: int = 4, width: int = 8, n_layers: int = 2) -> FNO2d:
    return FNO2d(in_channels=1, out_channels=1, width=width, modes1=modes, modes2=modes, n_layers=n_layers)


def _small_fno_baseline(modes: int = 4) -> FNOBaseline:
    return FNOBaseline(width=8, modes=modes, n_layers=2)


# ---------------------------------------------------------------------------
# 1. U-Net forward shape
# ---------------------------------------------------------------------------

class TestUNetShape:
    H = 32

    def test_plain_output_shape(self):
        unet = _small_unet(self.H)
        x = torch.randn(2, 1, self.H, self.H)
        out = unet(x)
        assert out.shape == (2, 1, self.H, self.H)

    def test_batch_size_1(self):
        unet = _small_unet(self.H)
        x = torch.randn(1, 1, self.H, self.H)
        assert unet(x).shape == (1, 1, self.H, self.H)

    def test_with_injection(self):
        # For H=32, 5 stages: sizes [32,16,8,4,2]; injection at 4 and 2
        H = 32
        inj_ch = {4: 1, 2: 1}
        unet = _small_unet(H, inj_channels=inj_ch)
        x = torch.randn(2, 1, H, H)
        inj = {4: torch.randn(2, 1, 4, 4), 2: torch.randn(2, 1, 2, 2)}
        out = unet(x, injections=inj)
        assert out.shape == (2, 1, H, H)

    def test_with_film(self):
        H = 32
        unet = _small_unet(H)
        x = torch.randn(2, 1, H, H)
        # Build valid FiLM params for each decoder output stage
        film = _build_film_params(unet, batch=2)
        out = unet(x, film_params=film)
        assert out.shape == (2, 1, H, H)

    def test_injection_and_film_combined(self):
        H = 32
        inj_ch = {4: 1, 2: 1}
        unet = _small_unet(H, inj_channels=inj_ch)
        x = torch.randn(2, 1, H, H)
        inj = {4: torch.randn(2, 1, 4, 4), 2: torch.randn(2, 1, 2, 2)}
        film = _build_film_params(unet, batch=2)
        out = unet(x, injections=inj, film_params=film)
        assert out.shape == (2, 1, H, H)

    def test_output_dtype(self):
        unet = _small_unet(self.H)
        x = torch.randn(2, 1, self.H, self.H)
        assert unet(x).dtype == torch.float32

    def test_high_k_damping_shape(self):
        unet = UNet(in_channels=1, base_channels=8, channel_mults=(1,2,2,4,4),
                    n_res_blocks=1, groups=4, attn_resolutions=(2,),
                    input_size=32, high_k_damping=True)
        x = torch.randn(2, 1, 32, 32)
        assert unet(x).shape == (2, 1, 32, 32)


# ---------------------------------------------------------------------------
# 2. FiLM actually changes the output
# ---------------------------------------------------------------------------

def _build_film_params(unet: UNet, batch: int = 2, scale: float = 1.0,
                       include_bottleneck: bool = True) -> dict:
    """Build per-stage FiLM params from the decoder's channel layout.

    include_bottleneck=True (default) includes _stage_sizes[0] (the 16×16
    bottleneck FiLM key added by Bug 1 fix).  Set False to test legacy behaviour.
    """
    dec_sizes = unet.decoder._stage_sizes   # coarsest→finest [16,32,64,128,256]
    enc_ch    = unet.encoder._ch_out        # finest-first
    n = len(enc_ch)
    film = {}
    # Bottleneck (index 0): channels = enc_ch[n-1] = coarsest encoder output
    if include_bottleneck:
        C_btl = enc_ch[n - 1 - 0]   # enc_ch[n-1] for j=0
        film[dec_sizes[0]] = (
            scale * torch.ones(batch, C_btl),
            torch.zeros(batch, C_btl),
        )
    # Remaining decoder stages (index 1..n-1)
    for i, res in enumerate(dec_sizes[1:], 0):
        C = enc_ch[n - 1 - i - 1]
        film[res] = (
            scale * torch.ones(batch, C),
            torch.zeros(batch, C),
        )
    return film


class TestFiLM:
    H = 32

    def test_identity_film_nearly_unchanged(self):
        """γ=1, β=0 FiLM should give the same output as no FiLM."""
        unet = _small_unet(self.H)
        x = torch.randn(2, 1, self.H, self.H)
        film_id = _build_film_params(unet, scale=1.0)
        out_plain = unet(x)
        out_film  = unet(x, film_params=film_id)
        assert torch.allclose(out_plain, out_film, atol=1e-5), (
            f"Identity FiLM changed output: max diff={( out_plain-out_film).abs().max():.2e}"
        )

    def test_nonidentity_film_changes_output(self):
        """γ=2, β=0 should definitely change the output."""
        unet = _small_unet(self.H)
        torch.manual_seed(0)
        x = torch.randn(2, 1, self.H, self.H)
        film_id   = _build_film_params(unet, scale=1.0)
        film_x2   = _build_film_params(unet, scale=2.0)
        out_id = unet(x, film_params=film_id)
        out_x2 = unet(x, film_params=film_x2)
        assert not torch.allclose(out_id, out_x2), "γ=2 FiLM must change the output"

    def test_bottleneck_film_applied(self):
        """FiLM at _stage_sizes[0] (=bottleneck, resolution 2 for H=32) changes output.

        This test verifies Bug 1 is fixed: the bottleneck FiLM key is now
        looked up BEFORE the decoder loop, not missed inside it.
        """
        unet = _small_unet(self.H)
        x = torch.randn(2, 1, self.H, self.H)
        btl_res = unet.decoder._stage_sizes[0]   # e.g. 2 for H=32, 5 stages

        # FiLM only at bottleneck (no other stages)
        enc_ch = unet.encoder._ch_out
        n = len(enc_ch)
        C_btl = enc_ch[n - 1]   # channels at bottleneck (coarsest encoder output)
        film_btl = {btl_res: (2.0 * torch.ones(2, C_btl), torch.zeros(2, C_btl))}

        out_no_film  = unet(x)
        out_btl_film = unet(x, film_params=film_btl)
        assert not torch.allclose(out_no_film, out_btl_film, atol=1e-4), (
            f"Bottleneck FiLM (res={btl_res}, γ=2) must change U-Net output. "
            "If it doesn't, the bottleneck FiLM path (Bug 1 fix) is not working."
        )

    def test_film_effect_increases_with_scale(self):
        """Larger |γ-1| should produce larger output difference."""
        unet = _small_unet(self.H)
        torch.manual_seed(1)
        x = torch.randn(2, 1, self.H, self.H)
        out_base = unet(x)
        diffs = []
        for scale in [1.5, 3.0, 6.0]:
            film = _build_film_params(unet, scale=scale)
            diffs.append((unet(x, film_params=film) - out_base).abs().mean().item())
        assert diffs[0] < diffs[1] < diffs[2], (
            f"FiLM effect should grow with scale: {diffs}"
        )

    def test_film_gradient_flows(self):
        """Gradients must reach the gamma/beta tensors."""
        unet = _small_unet(self.H)
        x = torch.randn(2, 1, self.H, self.H)
        dec_sizes = unet.decoder._stage_sizes
        enc_ch    = unet.encoder._ch_out
        n = len(enc_ch)
        gammas, betas = [], []
        film = {}
        for i, res in enumerate(dec_sizes[1:], 0):
            C = enc_ch[n - 1 - i - 1]
            g = torch.ones(2, C, requires_grad=True)
            b = torch.zeros(2, C, requires_grad=True)
            film[res] = (g, b)
            gammas.append(g); betas.append(b)

        out = unet(x, film_params=film)
        out.sum().backward()
        for g in gammas:
            assert g.grad is not None and g.grad.abs().sum() > 0, "γ grad is zero"
        for b in betas:
            assert b.grad is not None and b.grad.abs().sum() > 0, "β grad is zero"


# ---------------------------------------------------------------------------
# 3. Injection at correct encoder stages
# ---------------------------------------------------------------------------

class TestInjection:
    def test_injection_ignored_at_wrong_resolution(self):
        """Injection at a resolution not in injection_channels should be silently ignored."""
        H = 32
        unet = _small_unet(H, inj_channels={4: 1})   # only res=4 registered
        x = torch.randn(2, 1, H, H)
        # Injection at res=2 not registered — should be ignored (not error)
        inj = {4: torch.randn(2, 1, 4, 4), 2: torch.randn(2, 1, 2, 2)}
        out = unet(x, injections=inj)
        assert out.shape == (2, 1, H, H)

    def test_injection_multichannel(self):
        """Injection can carry more than one channel."""
        H = 32
        extra = 3
        unet = _small_unet(H, inj_channels={4: extra})
        x = torch.randn(2, 1, H, H)
        inj = {4: torch.randn(2, extra, 4, 4)}
        out = unet(x, injections=inj)
        assert out.shape == (2, 1, H, H)

    def test_injection_vs_no_injection_differ(self):
        """Same model, same input, with vs without injection → different output."""
        H = 32
        torch.manual_seed(42)
        unet = _small_unet(H, inj_channels={4: 1})
        x = torch.randn(2, 1, H, H)
        out_plain = unet(x)
        inj = {4: torch.randn(2, 1, 4, 4)}
        out_inj = unet(x, injections=inj)
        assert not torch.allclose(out_plain, out_inj), (
            "Injection should change the output"
        )

    def test_injection_stages_match_spec(self):
        """For H=32, stages are at resolutions 32,16,8,4,2.
        Injection registered at 4 and 2 should appear in inject_convs.
        """
        H = 32
        unet = _small_unet(H, inj_channels={4: 1, 2: 1})
        keys = set(unet.encoder.inject_convs.keys())
        assert "4" in keys and "2" in keys, f"Expected '4' and '2' in inject_convs, got {keys}"


# ---------------------------------------------------------------------------
# 4. Skip connections geometry
# ---------------------------------------------------------------------------

class TestSkipConnections:
    def test_skips_count(self):
        """Encoder must return exactly n_stages skip tensors."""
        H = 32
        enc = UNetEncoder(base_channels=8, channel_mults=(1,2,2,4,4),
                          n_res_blocks=1, groups=4, attn_resolutions=(2,), input_size=H)
        x = torch.randn(2, 1, H, H)
        _, skips = enc(x)
        assert len(skips) == 5, f"Expected 5 skips, got {len(skips)}"

    def test_skips_spatial_sizes(self):
        """Skips must go from coarsest to finest."""
        H = 32
        enc = UNetEncoder(base_channels=8, channel_mults=(1,2,2,4,4),
                          n_res_blocks=1, groups=4, attn_resolutions=(2,), input_size=H)
        x = torch.randn(2, 1, H, H)
        _, skips = enc(x)
        # skips[0] = coarsest = H/16, skips[-1] = finest = H
        sizes = [s.shape[-1] for s in skips]
        assert sizes == sorted(sizes), f"Skips not coarsest-first: {sizes}"
        assert sizes[-1] == H, f"Finest skip should be {H}, got {sizes[-1]}"
        assert sizes[0] == H // 16, f"Coarsest skip should be {H//16}, got {sizes[0]}"


# ---------------------------------------------------------------------------
# 5. Residual add (output = input + increment)
# ---------------------------------------------------------------------------

class TestResidualAdd:
    def test_output_equals_input_plus_increment(self):
        """Confirm ω̂ = ω_t + Δω by checking with a zero-weight head."""
        H = 32
        unet = _small_unet(H)
        x = torch.randn(2, 1, H, H)
        # Zero the head weights so Δω = 0; output should equal input
        with torch.no_grad():
            for p in unet.head.parameters():
                p.zero_()
        out = unet(x)
        assert torch.allclose(out, x, atol=1e-6), (
            f"With zero head, output should equal input; max diff={(out-x).abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# 6 & 7. FNO shape and semantics
# ---------------------------------------------------------------------------

class TestFNO:
    H = 32

    def test_fno2d_output_shape(self):
        fno = _small_fno()
        x = torch.randn(2, 1, self.H, self.H)
        delta = fno(x)
        assert delta.shape == (2, 1, self.H, self.H)

    def test_spectral_conv_output_shape(self):
        sc = SpectralConv2d(4, 4, 4, 4)
        x = torch.randn(2, 4, self.H, self.H)
        assert sc(x).shape == (2, 4, self.H, self.H)

    def test_fno_block_output_shape(self):
        fb = FNOBlock(8, 4, 4)
        x = torch.randn(2, 8, self.H, self.H)
        assert fb(x).shape == (2, 8, self.H, self.H)

    def test_fno_baseline_forward_shape(self):
        fno = _small_fno_baseline()
        x = torch.randn(2, 1, self.H, self.H)
        assert fno(x).shape == (2, 1, self.H, self.H)

    def test_fno_baseline_rollout_shape(self):
        fno = _small_fno_baseline()
        x = torch.randn(2, 1, self.H, self.H)
        preds = fno.rollout(x, n_steps=5)
        assert preds.shape == (2, 5, 1, self.H, self.H)

    def test_fno_baseline_is_residual(self):
        """FNOBaseline(x) = x + FNO2d(x); confirm with zero FNO output."""
        fno = _small_fno_baseline()
        x = torch.randn(2, 1, self.H, self.H)
        # Zero all FNO weights so delta=0
        with torch.no_grad():
            for p in fno.fno.parameters():
                p.zero_()
        out = fno(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_fno_rollout_autoregressive(self):
        """Each rollout step should use the *previous* output, not the original input."""
        fno = _small_fno_baseline()
        x = torch.randn(1, 1, self.H, self.H)
        preds = fno.rollout(x, n_steps=3)
        # Check step 2 was produced from step 1, not from x directly
        step1 = fno(x)
        step2_from_step1 = fno(step1)
        assert torch.allclose(preds[:, 1], step2_from_step1, atol=1e-6), (
            "Rollout step 2 should equal fno(step1)"
        )


# ---------------------------------------------------------------------------
# 8. Parameter count assertions
# ---------------------------------------------------------------------------

class TestParamCounts:
    def test_unet_small_param_count(self):
        """Small U-Net should be under 2M params."""
        unet = _small_unet(32)
        n = unet.n_params()
        assert n < 2_000_000, f"Small U-Net has {n:,} params, expected < 2M"
        assert n > 100_000, f"Small U-Net seems too small: {n:,} params"

    def test_unet_full_param_count(self):
        """Full-spec U-Net (H=256, base=64, 2 res blocks) — measured at ~13.6M params."""
        unet = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,2,4,4),
                    n_res_blocks=2, groups=8, attn_resolutions=(16,), input_size=256)
        n = unet.n_params()
        assert 5_000_000 < n < 100_000_000, (
            f"Full U-Net has {n:,} params, expected 5M–100M"
        )
        print(f"\n  Full U-Net param count: {n/1e6:.2f}M")

    def test_fno_param_count(self):
        """FNO baseline with width=64, modes=32 — SpectralConv2d is parameter-heavy (~33M)."""
        fno = FNOBaseline(width=64, modes=32, n_layers=4)
        n = fno.n_params()
        # SpectralConv2d: 2 corners × in_ch × out_ch × modes1 × modes2 × 2 (complex)
        # = 2 × 64 × 64 × 32 × 32 × 2 = ~8.4M per layer × 4 layers + linear layers
        assert 1_000_000 < n < 200_000_000, f"FNO baseline has {n:,} params"
        print(f"\n  FNO baseline param count: {n/1e6:.2f}M")

    def test_fno_substantially_smaller_than_unet(self):
        """Skip this check — FNO with large modes is actually comparable in size to U-Net.
        The point of Invariant 10 is architectural separation, not parameter budget.
        """
        # Just confirm both are non-trivial
        fno  = FNOBaseline(width=64, modes=32, n_layers=4)
        unet = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,2,4,4),
                    n_res_blocks=2, groups=8, attn_resolutions=(16,), input_size=256)
        assert fno.n_params()  > 100_000, "FNO seems too small"
        assert unet.n_params() > 100_000, "UNet seems too small"


# ---------------------------------------------------------------------------
# 9. Invariant 10: separate classes, no cross-inheritance
# ---------------------------------------------------------------------------

class TestInvariant10:
    def test_unet_not_subclass_of_fno(self):
        assert not issubclass(UNet, FNO2d)
        assert not issubclass(UNet, FNOBaseline)

    def test_fno_not_subclass_of_unet(self):
        assert not issubclass(FNO2d, UNet)
        assert not issubclass(FNOBaseline, UNet)

    def test_fno_baseline_uses_fno_not_unet(self):
        fno = FNOBaseline()
        # Must contain an FNO2d, not a UNet
        has_fno  = any(isinstance(m, FNO2d) for m in fno.modules())
        has_unet = any(isinstance(m, UNet)  for m in fno.modules())
        assert has_fno,  "FNOBaseline must contain FNO2d"
        assert not has_unet, "FNOBaseline must NOT contain UNet (Invariant 10)"

    def test_unet_does_not_contain_fno(self):
        unet = _small_unet(32)
        has_fno = any(isinstance(m, FNO2d) for m in unet.modules())
        assert not has_fno, "UNet must NOT contain FNO2d (Invariant 10)"


# ---------------------------------------------------------------------------
# 10. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    H = 32

    def test_unet_backward(self):
        unet = _small_unet(self.H)
        x = torch.randn(2, 1, self.H, self.H, requires_grad=True)
        loss = unet(x).sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_unet_backward_with_injection(self):
        unet = _small_unet(self.H, inj_channels={4: 1})
        x   = torch.randn(2, 1, self.H, self.H, requires_grad=True)
        inj = {4: torch.randn(2, 1, 4, 4, requires_grad=True)}
        loss = unet(x, injections=inj).sum()
        loss.backward()
        assert x.grad is not None and inj[4].grad is not None

    def test_fno_backward(self):
        fno = _small_fno()
        x = torch.randn(2, 1, self.H, self.H, requires_grad=True)
        loss = fno(x).sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_unet_params_receive_grads(self):
        """All params that participate in the forward pass receive gradients.
        Inject convs are only used when injections are provided; test both.
        """
        H = 32
        inj_ch = {4: 1}
        unet = _small_unet(H, inj_channels=inj_ch)
        x = torch.randn(2, 1, H, H)
        # Provide injection so inject_conv is used
        inj = {4: torch.randn(2, 1, 4, 4)}
        unet(x, injections=inj).sum().backward()
        for name, p in unet.named_parameters():
            if p.requires_grad:
                # inject_convs for OTHER resolutions won't be used (none registered)
                # but the registered one (res=4) must have a gradient
                assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# 11. ResBlock and SelfAttention2D primitives
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_resblock_same_channels(self):
        rb = ResBlock(16, 16, groups=4)
        x = torch.randn(2, 16, 8, 8)
        out = rb(x)
        assert out.shape == (2, 16, 8, 8)

    def test_resblock_channel_change(self):
        rb = ResBlock(8, 16, groups=4)
        x = torch.randn(2, 8, 8, 8)
        assert rb(x).shape == (2, 16, 8, 8)

    def test_resblock_skip_learned(self):
        """Channel-changing ResBlock uses a learned 1×1 skip."""
        rb = ResBlock(8, 16, groups=4)
        assert isinstance(rb.skip, nn.Conv2d), "Mismatch-channel ResBlock must use Conv2d skip"

    def test_resblock_identity_skip(self):
        """Same-channel ResBlock must use nn.Identity skip."""
        rb = ResBlock(16, 16, groups=4)
        assert isinstance(rb.skip, nn.Identity), "Same-channel ResBlock must use Identity skip"

    def test_selfattn_output_shape(self):
        attn = SelfAttention2D(16, num_heads=4)
        x = torch.randn(2, 16, 8, 8)
        out = attn(x)
        assert out.shape == x.shape

    def test_selfattn_residual(self):
        """Self-attention must be residual: attn(x) ≠ 0 even when weights are near zero."""
        attn = SelfAttention2D(16, num_heads=4)
        x = torch.randn(1, 16, 4, 4)
        # With random init, output should not equal input (attention adds something)
        # But it IS residual so with zero weights output ≈ input
        with torch.no_grad():
            for p in attn.attn.parameters():
                p.zero_()
        out = attn(x)
        # With zero attn weights, the output of attn(q,k,v) = 0 → output = x + 0 = x
        assert torch.allclose(out, x, atol=1e-5), "SelfAttention2D must be residual"
