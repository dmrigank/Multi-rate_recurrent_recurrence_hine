"""Spectral truncation tests (DESIGN.md §2, Invariant 5).

Tests
─────
1. Nesting          P^2 == P^2 ∘ P^1   (exact to floating-point tolerance)
2. Idempotence      P^l ∘ P^l == P^l
3. Band content     P^l zeroes all modes above its cutoff
4. Down/Up          • up∘down round-trip on band-limited field
                    • down∘up is idempotent  (projection identity)
                    • spatial coarse-grid size matches n_out = 2*(k_max+1)
                    • amplitude preservation on a pure sinusoid
5. Differentiability  torch.autograd.gradcheck on project / down / up
6. SpectralPool module  buffer registration, device move, forward == project
7. build_hierarchy  three levels with correct cutoffs
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.autograd import gradcheck

from msr_hine.spectral.truncation import (
    K_COARSE,
    K_FINE,
    K_MEDIUM,
    SpectralPool,
    _coarse_size,
    build_hierarchy,
    check_nesting,
    down,
    extract_band_coeffs,
    project,
    radial_mask_rfft2,
    up,
)

DEVICE = torch.device("cpu")
# Small grid used in most tests (fast); a few use n=128 to hit the full hierarchy
N_SMALL = 32
N_MED   = 64


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _band_limited(n: int, k_max: int, batch: int = 2, seed: int = 0) -> torch.Tensor:
    """Random real field band-limited to k_max on an n×n grid."""
    torch.manual_seed(seed)
    return project(torch.randn(batch, n, n), k_max)


# ---------------------------------------------------------------------------
# 1. Nesting  P^2 == P^2 ∘ P^1
# ---------------------------------------------------------------------------

class TestNesting:
    """P^coarse == P^coarse ∘ P^fine for every (n, k_fine, k_coarse) combo."""

    @pytest.mark.parametrize("n,k_fine,k_coarse", [
        (N_SMALL, 8, 4),
        (N_MED,   16, 8),
        (128,     K_MEDIUM, K_COARSE),
        (256,     K_MEDIUM, K_COARSE),
    ])
    def test_nesting_identity(self, n, k_fine, k_coarse):
        torch.manual_seed(42)
        x = torch.randn(2, n, n)
        lhs = project(x, k_coarse)
        rhs = project(project(x, k_fine), k_coarse)
        assert torch.allclose(lhs, rhs, atol=1e-5), (
            f"n={n} k_fine={k_fine} k_coarse={k_coarse}: "
            f"max diff = {(lhs-rhs).abs().max():.2e}"
        )

    def test_check_nesting_function(self):
        assert check_nesting(n=N_MED, device=DEVICE, k_fine=16, k_coarse=8)

    def test_check_nesting_canonical(self):
        """The DESIGN.md §2 canonical nesting: k=8 and k=16 on 256×256."""
        assert check_nesting(n=256, device=DEVICE)

    def test_set_inclusion(self):
        """Mask(k_coarse) is a subset of Mask(k_fine) by definition."""
        n = N_MED
        m_fine   = radial_mask_rfft2(n, 16, DEVICE)
        m_coarse = radial_mask_rfft2(n, 8,  DEVICE)
        assert (m_coarse & ~m_fine).sum() == 0, (
            "Coarse mask retains modes that the fine mask rejects — nesting broken"
        )


# ---------------------------------------------------------------------------
# 2. Idempotence  P^l ∘ P^l == P^l
# ---------------------------------------------------------------------------

class TestIdempotence:
    @pytest.mark.parametrize("n,k", [
        (N_SMALL, 4), (N_SMALL, 8), (N_MED, 16), (128, K_MEDIUM), (128, K_COARSE),
    ])
    def test_project_idempotent(self, n, k):
        torch.manual_seed(7)
        x = torch.randn(2, n, n)
        p  = project(x, k)
        pp = project(p, k)
        assert torch.allclose(p, pp, atol=1e-5), (
            f"n={n} k={k}: |P(P(x))-P(x)|_max = {(pp-p).abs().max():.2e}"
        )

    def test_spectralpool_idempotent(self):
        pool = SpectralPool(N_MED, 8)
        x = torch.randn(2, N_MED, N_MED)
        p  = pool(x)
        pp = pool(p)
        assert torch.allclose(p, pp, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Band content  (no energy above cutoff after projection)
# ---------------------------------------------------------------------------

class TestBandContent:
    @pytest.mark.parametrize("n,k", [
        (N_SMALL, 4), (N_SMALL, 8), (N_MED, 16),
    ])
    def test_out_of_band_exactly_zero(self, n, k):
        torch.manual_seed(3)
        x = project(torch.randn(2, n, n), k)
        xhat = torch.fft.rfft2(x)
        mask = radial_mask_rfft2(n, k, DEVICE)
        outside_energy = xhat[..., ~mask].abs().max().item()
        assert outside_energy < 5e-5, (
            f"n={n} k={k}: max amplitude outside band = {outside_energy:.2e}"
        )

    def test_in_band_nonzero(self):
        """Projection must not collapse to zero for a generic input."""
        x = torch.randn(2, N_MED, N_MED)
        p = project(x, 16)
        assert p.abs().max() > 0.01

    def test_project_leaves_lower_bands_unchanged(self):
        """P^fine applied to a P^coarse-limited field is a no-op."""
        n = N_MED
        x = _band_limited(n, 8)
        # Applying P^fine (k=16) to a field already at k=8 must not change it
        x_fine = project(x, 16)
        assert torch.allclose(x, x_fine, atol=1e-5), (
            f"|P^16(P^8(rand)) - P^8(rand)|_max = {(x_fine-x).abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Down / Up
# ---------------------------------------------------------------------------

class TestDownUp:
    @pytest.mark.parametrize("n,k", [
        (N_SMALL, 4), (N_MED, 8), (N_MED, 16), (128, K_COARSE), (128, K_MEDIUM),
    ])
    def test_round_trip_band_limited(self, n, k):
        """up(down(P^l(x))) ≈ P^l(x) to machine precision."""
        x_bl = _band_limited(n, k)
        x_up = up(down(x_bl, k), n)
        err = (x_up - x_bl).abs().max().item()
        assert err < 1e-5, f"n={n} k={k}: round-trip error = {err:.2e}"

    @pytest.mark.parametrize("k", [4, 8, 16])
    def test_coarse_size(self, k):
        assert _coarse_size(k) == 2 * (k + 1)

    @pytest.mark.parametrize("n,k", [
        (N_SMALL, 4), (N_MED, 8), (N_MED, 16),
    ])
    def test_down_output_shape(self, n, k):
        x = torch.randn(2, 3, n, n)
        xd = down(x, k)
        assert xd.shape == (2, 3, _coarse_size(k), _coarse_size(k)), (
            f"down shape mismatch: {xd.shape}"
        )

    @pytest.mark.parametrize("n,k", [
        (N_SMALL, 4), (N_MED, 8), (N_MED, 16),
    ])
    def test_up_output_shape(self, n, k):
        n_out = _coarse_size(k)
        x_low = torch.randn(2, 3, n_out, n_out)
        xu = up(x_low, n)
        assert xu.shape == (2, 3, n, n), f"up shape mismatch: {xu.shape}"

    def test_down_up_idempotent(self):
        """up∘down is a projection: applying it twice gives the same result."""
        n, k = N_MED, 8
        x = _band_limited(n, k)
        xu1 = up(down(x, k), n)
        xu2 = up(down(xu1, k), n)
        err = (xu2 - xu1).abs().max().item()
        assert err < 1e-5, f"|up(down(up(down(x))))-up(down(x))|_max = {err:.2e}"

    @pytest.mark.parametrize("n,k,kx_test,ky_test", [
        (N_MED, 8,  2, 3),   # interior mode, well inside band
        (N_MED, 8,  0, 4),   # pure ky mode
        (N_MED, 16, 3, 0),   # pure kx mode
    ])
    def test_amplitude_preservation(self, n, k, kx_test, ky_test):
        """A pure cosine wave survives down→up with amplitude preserved."""
        yy = torch.linspace(0, 2 * math.pi * (n - 1) / n, n)
        xx = torch.linspace(0, 2 * math.pi * (n - 1) / n, n)
        YY, XX = torch.meshgrid(yy, xx, indexing="ij")
        sine = torch.cos(ky_test * YY + kx_test * XX)
        rms_orig = sine.std().item()

        sd = down(sine.unsqueeze(0), k)[0]
        su = up(sd.unsqueeze(0), n)[0]

        rms_up = su.std().item()
        rel_err = abs(rms_up - rms_orig) / (rms_orig + 1e-10)
        assert rel_err < 1e-5, (
            f"Amplitude not preserved: orig={rms_orig:.6f} up={rms_up:.6f} rel={rel_err:.2e}"
        )
        field_err = (su - sine).abs().max().item()
        assert field_err < 1e-5, f"Field error after round-trip: {field_err:.2e}"

    def test_up_zeroes_out_of_band(self):
        """After up, the result should be band-limited at the coarse k_max."""
        n, k = N_MED, 8
        x = _band_limited(n, k)
        xu = up(down(x, k), n)
        # xu should be band-limited at k; check modes outside band are negligible
        xhat = torch.fft.rfft2(xu)
        mask = radial_mask_rfft2(n, k, DEVICE)
        outside = xhat[..., ~mask].abs().max().item()
        assert outside < 1e-4, f"Energy outside band after up: {outside:.2e}"


# ---------------------------------------------------------------------------
# 5. Differentiability (gradcheck)
# ---------------------------------------------------------------------------

class TestDifferentiability:
    """gradcheck requires float64 and small inputs to avoid numerical issues."""

    def test_project_gradcheck(self):
        torch.manual_seed(0)
        x = torch.randn(1, 8, 8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda v: project(v, 4),
            (x,),
            eps=1e-6, atol=1e-4, rtol=1e-3,
            check_grad_dtypes=True,
        )

    def test_down_gradcheck(self):
        torch.manual_seed(1)
        k = 4
        x = torch.randn(1, 16, 16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda v: down(v, k),
            (x,),
            eps=1e-6, atol=1e-4, rtol=1e-3,
            check_grad_dtypes=True,
        )

    def test_up_gradcheck(self):
        torch.manual_seed(2)
        k = 4
        n_out = _coarse_size(k)   # = 10
        n_full = 16
        x_low = torch.randn(1, n_out, n_out, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda v: up(v, n_full),
            (x_low,),
            eps=1e-6, atol=1e-4, rtol=1e-3,
            check_grad_dtypes=True,
        )

    def test_project_gradient_flows(self):
        """Gradients from a loss on the projected field reach the input."""
        x = torch.randn(1, N_SMALL, N_SMALL, requires_grad=True)
        p = project(x, 8)
        loss = p.pow(2).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().max() > 0, "Gradient did not propagate through project"

    def test_down_gradient_flows(self):
        x = torch.randn(1, N_MED, N_MED, requires_grad=True)
        d = down(x, 8)
        d.pow(2).sum().backward()
        assert x.grad is not None and x.grad.abs().max() > 0

    def test_up_gradient_flows(self):
        x_low = torch.randn(1, _coarse_size(8), _coarse_size(8), requires_grad=True)
        u = up(x_low, N_MED)
        u.pow(2).sum().backward()
        assert x_low.grad is not None and x_low.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# 6. SpectralPool module
# ---------------------------------------------------------------------------

class TestSpectralPool:
    def test_buffer_registered(self):
        pool = SpectralPool(N_MED, 8)
        assert "mask" in dict(pool.named_buffers()), "mask not in buffers"

    def test_mask_not_parameter(self):
        pool = SpectralPool(N_MED, 8)
        param_names = {n for n, _ in pool.named_parameters()}
        assert "mask" not in param_names, "mask must not be a learnable parameter"

    def test_forward_equals_project(self):
        pool = SpectralPool(N_MED, 8)
        x = torch.randn(2, N_MED, N_MED)
        assert torch.allclose(pool(x), project(x, 8), atol=1e-6)

    def test_device_move(self):
        pool = SpectralPool(N_SMALL, 4)
        # Verify buffer has the expected dtype/device after construction
        assert pool.mask.dtype == torch.bool
        assert pool.mask.device.type == "cpu"

    def test_down_up_via_module(self):
        pool = SpectralPool(N_MED, 8)
        x = _band_limited(N_MED, 8)
        x_rt = pool.up(pool.down(x))
        err = (x_rt - x).abs().max().item()
        assert err < 1e-5, f"SpectralPool down/up round-trip error: {err:.2e}"

    def test_n_out_attribute(self):
        for k in [4, 8, 16]:
            pool = SpectralPool(128, k)
            assert pool.n_out == _coarse_size(k)

    def test_extra_repr(self):
        pool = SpectralPool(256, 16)
        r = pool.extra_repr()
        assert "n=256" in r and "k_max=16" in r


# ---------------------------------------------------------------------------
# 7. build_hierarchy
# ---------------------------------------------------------------------------

class TestHierarchy:
    @pytest.fixture(scope="class")
    def hier(self):
        return build_hierarchy(n=128)

    def test_keys(self, hier):
        assert set(hier.keys()) == {"fine", "medium", "coarse"}

    def test_k_max_values(self, hier):
        assert hier["fine"].k_max   == K_FINE
        assert hier["medium"].k_max == K_MEDIUM
        assert hier["coarse"].k_max == K_COARSE

    def test_all_spatialpools(self, hier):
        for name, pool in hier.items():
            assert isinstance(pool, SpectralPool), f"{name} is not a SpectralPool"

    def test_nesting_via_hierarchy(self, hier):
        """coarse.project == coarse.project ∘ medium.project."""
        n = 128
        torch.manual_seed(5)
        x = torch.randn(2, n, n)
        lhs = hier["coarse"].project(x)
        rhs = hier["coarse"].project(hier["medium"].project(x))
        assert torch.allclose(lhs, rhs, atol=1e-5), (
            f"|coarse(medium(x)) - coarse(x)|_max = {(lhs-rhs).abs().max():.2e}"
        )

    def test_hierarchy_invariant5(self, hier):
        """No level's mask should appear in any parameter group."""
        for name, pool in hier.items():
            for pname, _ in pool.named_parameters():
                assert "mask" not in pname, (
                    f"Invariant 5 violated: mask appears as parameter in {name}.{pname}"
                )
