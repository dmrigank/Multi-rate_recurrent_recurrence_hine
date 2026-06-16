"""MSR-HINE model tests (DESIGN.md §3-4, Invariants 1, 4, 5, 6).

Tests
─────
1. FiLM: generator output shapes; near-identity init; modulation changes output.
2. ContractiveGRUCell: spectral norm present on recurrent weights; output shape;
   gradient flow; with vs without spectral norm.
3. RecurrentLevel: conditioning assembly shape; bounded gains ≤ α_max; GRU step.
4. Multi-rate clock (Invariant 4): over a 4-step rollout, coarse hidden state
   changes only at step 4 (0-indexed: step_n=3), medium changes at 2 and 4.
   Off-stride holds must be exact (same tensor values).
5. Warmup: after warmup over non-zero history, h_medium and h_coarse are nonzero.
6. MSRHINE forward/step shapes.
7. Invariant 1 structural check: step() does NOT encode ω̂ after producing it.
8. Stateful rollout: short rollout via rollout.rollout() returns correct shape;
   state is carried across steps (different states → different outputs).
9. Spectral norm on recurrent weights (Invariant 6): σ_max ≤ 1 + ε after a step.
10. no_contraction ablation: spectral norm absent when use_contraction=False.
11. FiLM changes U-Net output when non-identity (γ ≠ 1 or β ≠ 0).
12. Short rollout on debug data: forward produces finite outputs.

All tests run on a tiny model (base_ch=8, latent_dim=4/2, H=32) to stay fast.
"""

from __future__ import annotations

import math
import inspect

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from msr_hine.models.film import FiLMGenerator, apply_film
from msr_hine.models.recurrence import (
    ContractiveGRUCell,
    MultiRateHierarchy,
    RecurrentLevel,
)
from msr_hine.models.msr_hine import MSRHINE, MSRHINEState
from msr_hine.rollout import rollout
from msr_hine.train import build_model

DEVICE = torch.device("cpu")
H = 32   # small spatial grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model(H=H, use_contraction=True) -> MSRHINE:
    return MSRHINE(
        medium_dim         = 4,
        coarse_dim         = 2,
        medium_stride      = 2,
        coarse_stride      = 4,
        warmup_steps       = 3,
        alpha_max          = 0.2,
        unet_base_channels = 8,
        unet_channel_mults = (1, 1, 1, 1, 1),
        attn_resolutions   = (H // 16,),
        input_size         = H,
        use_contraction    = use_contraction,
        enc_hidden_ch      = 4,
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# 1. FiLM
# ---------------------------------------------------------------------------

class TestFiLM:
    def test_generator_output_shapes(self):
        gen = FiLMGenerator(hidden_dim=8, n_channels=16)
        h = torch.randn(2, 8)
        gam, bet = gen(h)
        assert gam.shape == (2, 16)
        assert bet.shape == (2, 16)

    def test_near_identity_init(self):
        """γ ≈ 1 and β ≈ 0 at initialisation so FiLM starts as near-identity."""
        gen = FiLMGenerator(hidden_dim=8, n_channels=16)
        h = torch.zeros(1, 8)
        gam, bet = gen(h)
        assert gam.abs().mean().item() == pytest.approx(1.0, abs=0.01)
        assert bet.abs().mean().item() == pytest.approx(0.0, abs=0.01)

    def test_bounded_residual_init_is_identity(self):
        with torch.random.fork_rng():
            gen = FiLMGenerator(
                hidden_dim=8,
                n_channels=16,
                gamma_mode="bounded_residual",
                gamma_scale=0.5,
            )
            gam, bet = gen(torch.randn(3, 8))
        assert torch.equal(gam, torch.ones_like(gam))
        assert torch.equal(bet, torch.zeros_like(bet))

    def test_bounded_residual_gamma_stays_in_range(self):
        with torch.random.fork_rng():
            gen = FiLMGenerator(
                hidden_dim=8,
                n_channels=16,
                gamma_mode="bounded_residual",
                gamma_scale=0.5,
            )
            with torch.no_grad():
                gen.gamma_head.weight.normal_(mean=0.0, std=10.0)
                gen.gamma_head.bias.normal_(mean=0.0, std=10.0)
            gam, _ = gen(torch.randn(32, 8))
        assert torch.all(gam >= 0.5)
        assert torch.all(gam <= 1.5)

    def test_bounded_residual_gamma_has_gradients(self):
        with torch.random.fork_rng():
            gen = FiLMGenerator(
                hidden_dim=8,
                n_channels=16,
                gamma_mode="bounded_residual",
            )
            gam, _ = gen(torch.randn(2, 8))
            gam.sum().backward()
            assert gen.gamma_head.weight.grad is not None
            assert gen.gamma_head.weight.grad.abs().sum() > 0

    def test_invalid_gamma_mode_raises(self):
        with pytest.raises(ValueError, match="gamma_mode"):
            FiLMGenerator(8, 16, gamma_mode="unknown")

    def test_build_model_propagates_bounded_gamma_config(self):
        cfg = OmegaConf.create({
            "data": {"n": H},
            "model": {
                "name": "msr_hine",
                "warmup_steps": 3,
                "enc_hidden_ch": 4,
                "unet": {
                    "base_channels": 8,
                    "channel_mults": [1, 1, 1, 1, 1],
                    "attn_resolutions": [H // 16],
                    "high_k_damping": False,
                },
                "recurrence": {
                    "medium_dim": 4,
                    "coarse_dim": 2,
                    "medium_stride": 2,
                    "coarse_stride": 4,
                    "alpha_max": 0.2,
                    "use_contraction": True,
                },
                "film": {
                    "gamma_mode": "bounded_residual",
                    "gamma_scale": 0.5,
                },
            },
        })
        with torch.random.fork_rng():
            model = build_model(cfg)
            assert model.film_medium.gamma_mode == "bounded_residual"
            assert model.film_medium.gamma_scale == pytest.approx(0.5)
            assert model.film_coarse.gamma_mode == "bounded_residual"
            assert model.film_coarse.gamma_scale == pytest.approx(0.5)

    def test_apply_film_shape(self):
        x = torch.randn(2, 16, 8, 8)
        gam = torch.ones(2, 16)
        bet = torch.zeros(2, 16)
        out = apply_film(x, gam, bet)
        assert out.shape == x.shape

    def test_identity_film_unchanged(self):
        x = torch.randn(2, 4, 8, 8)
        out = apply_film(x, torch.ones(2, 4), torch.zeros(2, 4))
        assert torch.allclose(out, x)

    def test_nonidentity_film_changes_output(self):
        x = torch.randn(2, 4, 8, 8)
        out = apply_film(x, 2.0 * torch.ones(2, 4), torch.ones(2, 4))
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# 2. ContractiveGRUCell
# ---------------------------------------------------------------------------

class TestContractiveGRU:
    def test_output_shape(self):
        cell = ContractiveGRUCell(input_dim=8, hidden_dim=4)
        c = torch.randn(2, 8)
        h = torch.randn(2, 4)
        h_new = cell(c, h)
        assert h_new.shape == (2, 4)

    def test_spectral_norm_applied(self):
        """Spectral norm wrappers must be present on recurrent weights."""
        cell = ContractiveGRUCell(input_dim=4, hidden_dim=4, use_spectral_norm=True)
        # nn.utils.spectral_norm wraps a module and adds 'weight_v' attribute
        assert hasattr(cell.Wh_r, "weight_orig"), "Wh_r lacks spectral_norm wrapper"
        assert hasattr(cell.Wh_z, "weight_orig"), "Wh_z lacks spectral_norm wrapper"
        assert hasattr(cell.Wh_n, "weight_orig"), "Wh_n lacks spectral_norm wrapper"

    def test_no_spectral_norm_when_disabled(self):
        cell = ContractiveGRUCell(input_dim=4, hidden_dim=4, use_spectral_norm=False)
        assert not hasattr(cell.Wh_r, "weight_orig"), \
            "Spectral norm should be absent when use_spectral_norm=False"

    def test_spectral_norm_sigma_bounded(self):
        """After power-iteration convergence, σ_max of the normalised weight ≤ 1 + ε.

        PyTorch's spectral_norm uses a one-step power iteration per forward pass.
        After enough forward passes the estimate converges and W_normalised
        (stored as .weight) has σ_max ≈ 1.  We run 30 passes to converge.
        """
        cell = ContractiveGRUCell(input_dim=4, hidden_dim=4, use_spectral_norm=True)
        c = torch.randn(2, 4); h = torch.randn(2, 4)
        # Run until power iteration has converged (~20-50 steps suffice)
        for _ in range(30):
            cell(c, h)
        for attr in ("Wh_r", "Wh_z", "Wh_n"):
            W = getattr(cell, attr).weight   # normalised weight
            sigma_max = torch.linalg.svdvals(W).max().item()
            assert sigma_max <= 1.0 + 1e-3, \
                f"{attr} σ_max = {sigma_max:.4f} > 1 after convergence"

    def test_gradient_flows(self):
        cell = ContractiveGRUCell(input_dim=4, hidden_dim=4)
        c = torch.randn(1, 4, requires_grad=True)
        h = torch.randn(1, 4)
        cell(c, h).sum().backward()
        assert c.grad is not None and c.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. RecurrentLevel
# ---------------------------------------------------------------------------

class TestRecurrentLevel:
    def test_conditioning_shape_medium(self):
        """Medium level: c = [z_cur(4) | α_td*z_coarse(2) | α_bd*(z_cur-z_prev)(4)] = 10"""
        lvl = RecurrentLevel(latent_dim=4, stride=2, has_topdown=True, coarse_dim=2)
        z_cur   = torch.randn(2, 4)
        z_prev  = torch.randn(2, 4)
        z_coarse = torch.randn(2, 2)
        c = lvl.build_conditioning(z_cur, z_prev, z_coarse)
        assert c.shape == (2, 4 + 2 + 4), f"Unexpected shape {c.shape}"

    def test_conditioning_shape_coarse(self):
        """Coarse level: c = [z_cur(2) | α_bd*(z_cur-z_prev)(2)] = 4"""
        lvl = RecurrentLevel(latent_dim=2, stride=4, has_topdown=False)
        z_cur  = torch.randn(2, 2)
        z_prev = torch.randn(2, 2)
        c = lvl.build_conditioning(z_cur, z_prev, None)
        assert c.shape == (2, 2 + 2), f"Unexpected shape {c.shape}"

    def test_gains_bounded_by_alpha_max(self):
        """Learned gains α = α_max * sigmoid(raw) must be ≤ α_max."""
        alpha_max = 0.2
        lvl = RecurrentLevel(latent_dim=4, stride=2, has_topdown=True,
                              coarse_dim=2, alpha_max=alpha_max)
        with torch.no_grad():
            # Push raw parameters to extremes
            lvl.raw_alpha_bd.fill_(10.0)
            lvl.raw_alpha_td.fill_(10.0)
        assert lvl._alpha_bd().item() <= alpha_max + 1e-6
        assert lvl._alpha_td().item() <= alpha_max + 1e-6

    def test_prior_emission_shape(self):
        lvl = RecurrentLevel(latent_dim=4, stride=2, has_topdown=False)
        h = torch.randn(2, 4)
        z_prior = lvl.emit_prior(h)
        assert z_prior.shape == (2, 4)


# ---------------------------------------------------------------------------
# 4. Multi-rate clock (Invariant 4)
# ---------------------------------------------------------------------------

class TestMultiRateClock:
    """Verify the hold-and-skip clock over a 4-step rollout.

    medium_stride=2: updates at step_n=1 (next_step=2) and step_n=3 (next_step=4).
    coarse_stride=4: updates only at step_n=3 (next_step=4).

    Steps 0 and 2 must hold both levels.
    Step 1 updates medium only.
    Step 3 updates both.
    """

    @pytest.fixture
    def hierarchy(self):
        return MultiRateHierarchy(
            medium_dim=4, coarse_dim=2,
            medium_stride=2, coarse_stride=4,
        )

    def _zero_state(self, B=1):
        h_m = torch.zeros(B, 4)
        h_c = torch.zeros(B, 2)
        zp_m = torch.zeros(B, 4)
        zp_c = torch.zeros(B, 2)
        return h_m, h_c, zp_m, zp_c

    def _run_step(self, hier, step_n, h_m, h_c, zp_m, zp_c):
        z_m = torch.randn(1, 4); z_mp = torch.randn(1, 4)
        z_c = torch.randn(1, 2); z_cp = torch.randn(1, 2)
        return hier.step(step_n, z_m, z_mp, z_c, z_cp, h_m, h_c, zp_m, zp_c)

    def test_step0_holds_both(self, hierarchy):
        h_m, h_c, zp_m, zp_c = self._zero_state()
        # Give non-zero inputs so any update would be detectable
        h_m = torch.ones(1, 4); h_c = torch.ones(1, 2)
        h_m_new, h_c_new, zp_m_new, zp_c_new = self._run_step(
            hierarchy, step_n=0, h_m=h_m, h_c=h_c, zp_m=zp_m, zp_c=zp_c)
        assert torch.equal(h_m_new, h_m), "step_n=0: medium should be held"
        assert torch.equal(h_c_new, h_c), "step_n=0: coarse should be held"

    def test_step1_updates_medium_holds_coarse(self, hierarchy):
        h_m = torch.ones(1, 4); h_c = torch.ones(1, 2)
        zp_m = torch.zeros(1, 4); zp_c = torch.zeros(1, 2)
        h_m_new, h_c_new, _, _ = self._run_step(
            hierarchy, step_n=1, h_m=h_m, h_c=h_c, zp_m=zp_m, zp_c=zp_c)
        assert not torch.equal(h_m_new, h_m), "step_n=1: medium should update"
        assert torch.equal(h_c_new, h_c),     "step_n=1: coarse should be held"

    def test_step2_holds_both(self, hierarchy):
        h_m = torch.ones(1, 4) * 2.0; h_c = torch.ones(1, 2) * 3.0
        zp_m = torch.zeros(1, 4); zp_c = torch.zeros(1, 2)
        h_m_new, h_c_new, _, _ = self._run_step(
            hierarchy, step_n=2, h_m=h_m, h_c=h_c, zp_m=zp_m, zp_c=zp_c)
        assert torch.equal(h_m_new, h_m), "step_n=2: medium should be held"
        assert torch.equal(h_c_new, h_c), "step_n=2: coarse should be held"

    def test_step3_updates_both(self, hierarchy):
        h_m = torch.ones(1, 4); h_c = torch.ones(1, 2)
        zp_m = torch.zeros(1, 4); zp_c = torch.zeros(1, 2)
        h_m_new, h_c_new, _, _ = self._run_step(
            hierarchy, step_n=3, h_m=h_m, h_c=h_c, zp_m=zp_m, zp_c=zp_c)
        assert not torch.equal(h_m_new, h_m), "step_n=3: medium should update"
        assert not torch.equal(h_c_new, h_c), "step_n=3: coarse should update"

    def test_prior_held_on_off_stride(self, hierarchy):
        """z_prior must be returned unchanged on off-stride steps."""
        zp_m = torch.randn(1, 4); zp_c = torch.randn(1, 2)
        h_m, h_c = torch.zeros(1, 4), torch.zeros(1, 2)
        _, _, zp_m_new, zp_c_new = self._run_step(
            hierarchy, step_n=0, h_m=h_m, h_c=h_c, zp_m=zp_m, zp_c=zp_c)
        assert torch.equal(zp_m_new, zp_m), "Medium prior must be held at step_n=0"
        assert torch.equal(zp_c_new, zp_c), "Coarse prior must be held at step_n=0"


# ---------------------------------------------------------------------------
# 5. Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_produces_nonzero_hidden_states(self):
        model = _tiny_model()
        state = model.init_state(2, DEVICE)
        assert state.h_medium.abs().sum() == 0.0, "Init state must be zero"
        omega_history = torch.randn(2, 4, 1, H, H)
        state_warm = model.warmup(omega_history, state)
        assert state_warm.h_medium.abs().sum() > 0, "Warmup must produce nonzero h_medium"
        assert state_warm.h_coarse.abs().sum() > 0, "Warmup must produce nonzero h_coarse"

    def test_warmup_with_zero_history_changes_state(self):
        """Even a zero history causes state to evolve (bias terms in GRU)."""
        model = _tiny_model()
        state0 = model.init_state(1, DEVICE)
        omega_history = torch.zeros(1, 4, 1, H, H)
        state_warm = model.warmup(omega_history, state0)
        # History of zeros still passes through GRU; state need not stay zero
        # (depends on bias init, but at minimum step_n should advance)
        assert state_warm.step_n == 4


# ---------------------------------------------------------------------------
# 6. MSRHINE forward/step shapes
# ---------------------------------------------------------------------------

class TestMSRHINEShapes:
    def test_forward_shape(self):
        model = _tiny_model()
        omega = torch.randn(2, 1, H, H)
        out = model(omega)
        assert out.shape == (2, 1, H, H)

    def test_step_shape(self):
        model = _tiny_model()
        omega = torch.randn(2, 1, H, H)
        state = model.init_state(2, DEVICE)
        omega_hat, next_state = model.step(omega, state)
        assert omega_hat.shape == (2, 1, H, H)
        assert next_state.h_medium.shape == (2, 4)
        assert next_state.h_coarse.shape == (2, 2)

    def test_rollout_shape(self):
        model = _tiny_model()
        omega_seed = torch.randn(2, 1, H, H)
        preds = rollout(model, omega_seed, n_steps=6)
        assert preds.shape == (2, 6, 1, H, H)

    def test_rollout_warmup_excluded(self):
        """Invariant 7: warmup frames excluded from returned predictions."""
        model = _tiny_model()
        B, W, K = 2, 3, 5
        omega_seed    = torch.randn(B, 1, H, H)
        warmup_frames = torch.randn(B, W, 1, H, H)
        preds = rollout(model, omega_seed, n_steps=K, warmup_frames=warmup_frames)
        assert preds.shape == (B, K, 1, H, H)

    def test_output_finite(self):
        model = _tiny_model()
        omega = torch.randn(2, 1, H, H)
        assert torch.isfinite(model(omega)).all()

    def test_state_carries_across_steps(self):
        """Different states → different predictions (state is actually used)."""
        model = _tiny_model()
        model.eval()
        omega = torch.randn(1, 1, H, H)
        state_zero = model.init_state(1, DEVICE)
        # Build a non-trivial state via warmup
        history = torch.randn(1, 4, 1, H, H)
        state_warm = model.warmup(history, model.init_state(1, DEVICE))
        with torch.no_grad():
            out_zero, _ = model.step(omega, state_zero)
            out_warm, _ = model.step(omega, state_warm)
        assert not torch.allclose(out_zero, out_warm, atol=1e-4), \
            "Different states must produce different outputs (state is used)"


# ---------------------------------------------------------------------------
# 6b. Multi-rate clock continuity through warmup
# ---------------------------------------------------------------------------

class TestClockContinuity:
    """Verify that step_n is NOT reset to 0 at the start of free rollout.

    After W warmup steps, state.step_n = W.  The multi-rate clock must
    continue from W, not restart from 0.  This test checks this by verifying
    that the coarse GRU fires at the correct global step, not at loop counter 3.
    """

    def test_step_n_monotonically_increases(self):
        """state.step_n increments by 1 on every step() call."""
        model = _tiny_model()
        state = model.init_state(1, DEVICE)
        assert state.step_n == 0
        omega = torch.randn(1, 1, H, H)
        for expected_n in range(5):
            assert state.step_n == expected_n
            _, state = model.step(omega, state)
        assert state.step_n == 5

    def test_warmup_advances_step_n(self):
        """Warmup over W frames leaves state.step_n == W."""
        W = 4
        model = _tiny_model()
        state = model.init_state(1, DEVICE)
        history = torch.randn(1, W, 1, H, H)
        state = model.warmup(history, state)
        assert state.step_n == W, (
            f"Expected step_n={W} after warmup, got {state.step_n}"
        )

    def test_free_rollout_continues_from_warmup_step_n(self):
        """After warmup, free-rollout step_n continues from W (not reset to 0)."""
        W = 4
        model = _tiny_model()
        state = model.init_state(1, DEVICE)
        history = torch.randn(1, W, 1, H, H)
        state = model.warmup(history, state)
        assert state.step_n == W

        # Run 3 free-rollout steps; step_n must be W+1, W+2, W+3
        omega = history[:, -1]   # last warmup frame as seed
        for i in range(3):
            _, state = model.step(omega, state)
            assert state.step_n == W + i + 1, (
                f"step_n should be {W+i+1} after {i+1} free steps, got {state.step_n}"
            )

    def test_coarse_update_uses_global_step_n(self):
        """Coarse GRU fires based on state.step_n, not a loop counter reset to 0.

        With medium_stride=2, coarse_stride=4: after warmup of 4 steps
        (state.step_n=4), the first coarse update in free rollout happens when
        state.step_n goes from 7 to 8 (since (7+1)%4=0), i.e. at free step 3.
        If the clock were reset to 0, it would fire at free step 3 (same in
        this case — but step_n=7 gives (7+1)%4=0 ✓, step_n=3 gives (3+1)%4=0 ✓).
        We instead check with warmup=6 where the two diverge:
          Correct: first coarse fires when (6+k)%4==3 → k=1 (global step 7)
          Wrong (reset): first coarse fires at k=3 (global step 3).
        """
        model = _tiny_model()   # medium_stride=2, coarse_stride=4
        W = 6   # NOT divisible by coarse_stride=4 → clocks diverge if reset
        state = model.init_state(1, DEVICE)
        history = torch.randn(1, W, 1, H, H)
        state = model.warmup(history, state)
        assert state.step_n == W

        # Track when h_coarse changes (= coarse GRU fired)
        omega = torch.randn(1, 1, H, H)
        first_coarse_update = None
        prev_h_coarse = state.h_coarse.clone()

        for free_step in range(8):
            _, new_state = model.step(omega, state)
            if not torch.equal(new_state.h_coarse, prev_h_coarse):
                first_coarse_update = free_step
                break
            prev_h_coarse = new_state.h_coarse.clone()
            state = new_state

        # With warmup=6 and coarse_stride=4:
        # Global step_n after warmup = 6.
        # Coarse fires when (step_n + 1) % 4 == 0 → step_n = 3, 7, 11, ...
        # First free-rollout coarse fire: global step_n=7 → free_step=1
        # If clock were RESET to 0: would fire at free_step=3 (global 3)
        assert first_coarse_update is not None, "Coarse GRU never updated in 8 steps"
        assert first_coarse_update == 1, (
            f"Coarse GRU should fire at free_step=1 (global step 7) with warmup=6, "
            f"but fired at free_step={first_coarse_update}. "
            "This indicates the stride clock was reset to 0 instead of continuing from warmup."
        )


# ---------------------------------------------------------------------------
# 7. Invariant 1 structural check
# ---------------------------------------------------------------------------

class TestInvariant1:
    """Verify that step() does not encode omega_hat after producing it.

    The only call to _encode inside step() is on the INPUT omega, not on
    the OUTPUT omega_hat.  We verify this by inspecting the source of step()
    and by checking that no encoder call on the output path is reachable.
    """

    def test_step_source_has_no_reencode(self):
        """step() must not re-encode omega_hat outside the explicitly-flagged control block.

        The CIRCULARITY-CONFIRMATION control block is the one documented exception.
        It is guarded by `self._fusion_CONTROL_ONLY` and loudly flagged.
        No other encode call should appear after the U-Net prediction.
        """
        src = inspect.getsource(MSRHINE.step)
        unet_pos = src.find("omega_hat = self.unet(")
        src_after_unet = src[unet_pos:]

        # Strip the explicitly-flagged control block (between the ⚠️ markers)
        control_start = src_after_unet.find("_fusion_CONTROL_ONLY")
        if control_start != -1:
            src_after_unet_no_control = src_after_unet[:control_start]
        else:
            src_after_unet_no_control = src_after_unet

        encode_after = (src_after_unet_no_control.count("_encode") +
                        src_after_unet_no_control.count("enc_medium") +
                        src_after_unet_no_control.count("enc_coarse"))
        assert encode_after == 0, (
            "Invariant 1 violated: step() calls an encoder AFTER the U-Net prediction "
            "outside the explicitly-flagged circularity-control block."
        )

    def test_step_does_not_fuse_posterior(self):
        """step() must not modify h after omega_hat is produced."""
        src = inspect.getsource(MSRHINE.step)
        unet_pos     = src.find("omega_hat = self.unet")
        hierarchy_after = src[unet_pos:].count("self.hierarchy") + \
                          src[unet_pos:].count("h_medium") + \
                          src[unet_pos:].count("h_coarse")
        # Only reading from next_state is fine; writing to hidden state is not
        # The return statement reads from next_state, which is acceptable.
        # We check there's no EXTRA hierarchy.step call after unet.
        extra_steps = src[unet_pos:].count("hierarchy.step")
        assert extra_steps == 0, (
            "Invariant 1 violated: hierarchy.step() called after U-Net prediction "
            "— this would fuse a posterior into the latent state."
        )


# ---------------------------------------------------------------------------
# 8. Spectral norm bounds (Invariant 6)
# ---------------------------------------------------------------------------

class TestSpectralNorm:
    def test_recurrent_weights_bounded(self):
        """σ_max of normalised recurrent weights ≤ 1 + ε after convergence (Invariant 6)."""
        model = _tiny_model(use_contraction=True)
        omega = torch.randn(1, 1, H, H)
        # Run many steps so spectral norm power iteration converges.
        # Coarse GRU updates every 4 steps; need ~60 steps for ~15 coarse updates.
        state = model.init_state(1, DEVICE)
        for _ in range(60):
            omega, state = model.step(omega, state)

        for level_name, level in [("medium", model.hierarchy.medium_level),
                                   ("coarse", model.hierarchy.coarse_level)]:
            for attr in ("Wh_r", "Wh_z", "Wh_n"):
                W = getattr(level.gru, attr).weight   # normalised weight
                sigma = torch.linalg.svdvals(W).max().item()
                assert sigma <= 1.0 + 1e-3, (
                    f"{level_name}.{attr} σ_max={sigma:.4f} > 1 (Invariant 6)"
                )

    def test_no_contraction_skips_spectral_norm(self):
        """With use_contraction=False, recurrent weights have no spectral norm wrapper."""
        model = _tiny_model(use_contraction=False)
        for level in (model.hierarchy.medium_level, model.hierarchy.coarse_level):
            for attr in ("Wh_r", "Wh_z", "Wh_n"):
                cell_layer = getattr(level.gru, attr)
                assert not hasattr(cell_layer, "weight_orig"), (
                    f"Spectral norm should be absent on {attr} when use_contraction=False"
                )


# ---------------------------------------------------------------------------
# 9. FiLM actually modulates the U-Net output
# ---------------------------------------------------------------------------

class TestFiLMEffect:
    def test_film_changes_output_when_nonidentity(self):
        """Setting γ=2 (non-identity) via the film generators should change output."""
        model = _tiny_model()
        model.eval()
        omega = torch.randn(1, 1, H, H)

        # Get output with default (near-identity) FiLM
        with torch.no_grad():
            out_default = model(omega).clone()

        # Force the FiLM generator to produce γ=2, β=0
        with torch.no_grad():
            for gen in (model.film_medium, model.film_coarse):
                nn.init.zeros_(gen.gamma_head.weight)
                gen.gamma_head.bias.fill_(2.0)    # γ=2
                nn.init.zeros_(gen.beta_head.weight)
                nn.init.zeros_(gen.beta_head.bias) # β=0

        with torch.no_grad():
            out_film2 = model(omega).clone()

        assert not torch.allclose(out_default, out_film2, atol=1e-4), (
            "γ=2 FiLM should change the U-Net output"
        )


# ---------------------------------------------------------------------------
# 10. Debug-data smoke test
# ---------------------------------------------------------------------------

class TestDebugSmoke:
    @pytest.fixture(scope="class")
    def debug_root(self, tmp_path_factory):
        import subprocess, sys
        root = tmp_path_factory.mktemp("msr_debug")
        r = subprocess.run(
            [sys.executable, "-m", "msr_hine.data.generate",
             "+debug=true", f"data.dataset_root={root}",
             "hydra.run.dir=.", "hydra/job_logging=disabled",
             "hydra/hydra_logging=disabled"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            pytest.fail(f"Dataset gen failed:\n{r.stderr}")
        return root / "debug"

    def test_forward_on_debug_data(self, debug_root):
        """Load a frame from debug data, run forward pass, check finite output."""
        import h5py, numpy as np
        with h5py.File(debug_root / "train.h5", "r") as f:
            frame = f["vorticity"][0, 0]   # [64, 64]
        omega = torch.from_numpy(frame.astype("float32")).unsqueeze(0).unsqueeze(0)
        model = _tiny_model(H=64)
        out = model(omega)
        assert out.shape == omega.shape
        assert torch.isfinite(out).all(), "Forward pass on debug data produced non-finite output"

    def test_rollout_on_debug_data(self, debug_root):
        """Short rollout from debug data returns correct shape and finite values."""
        import h5py, numpy as np
        with h5py.File(debug_root / "train.h5", "r") as f:
            traj = torch.from_numpy(f["vorticity"][0].astype("float32"))  # [T, 64, 64]
        traj = traj.unsqueeze(1)   # [T, 1, 64, 64]
        model = _tiny_model(H=64)
        preds = rollout(model, traj[3].unsqueeze(0), n_steps=4,
                        warmup_frames=traj[:3].unsqueeze(0))
        assert preds.shape == (1, 4, 1, 64, 64)
        assert torch.isfinite(preds).all()
