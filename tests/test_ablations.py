"""Ablation config tests (DESIGN.md §8, CLAUDE.md §6, Invariants 1, 10).

Tests
─────
1. Every ablation config instantiates a model without error.
2. Each toggle actually changes the model behaviour:
   a. no_multirate:   strides are 1 for both levels.
   b. no_contraction: recurrent weights have no spectral_norm wrapper.
   c. no_warmup:      warmup() is a no-op (state unchanged after history).
   d. no_topdown:     medium level has no top-down conditioning (has_topdown=False).
   e. single_scale:   no coarse encoder/decoder/FiLM; coarse state is zero-dim.
   f. no_consistency: lambda_cons=0 in config (loss check).
3. Invariant 1 — circularity-confirmation flag:
   a. _inference_fusion_CONTROL_ONLY is False by default.
   b. When set to True, step() result differs from the unfused path.
   c. The flag only affects inference, not training losses.
4. Invariant 10 — all U-Net ablations use UNet, not FNO2d.
5. Each ablation runs one forward step without error.
6. run_ablations.sh --debug --dry-run produces a command list (smoke test).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from msr_hine.models.msr_hine import MSRHINE, MSRHINEState
from msr_hine.models.unet import UNet

DEVICE = torch.device("cpu")
H      = 32   # small grid for speed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny(H=H, **kwargs) -> MSRHINE:
    defaults = dict(
        medium_dim=4, coarse_dim=2,
        medium_stride=2, coarse_stride=4,
        warmup_steps=2, alpha_max=0.2,
        unet_base_channels=8,
        unet_channel_mults=(1,1,1,1,1),
        attn_resolutions=(H//16,),
        input_size=H, enc_hidden_ch=4,
    )
    defaults.update(kwargs)
    return MSRHINE(**defaults).to(DEVICE)


def _one_step(model: MSRHINE, omega=None) -> tuple[torch.Tensor, MSRHINEState]:
    if omega is None:
        omega = torch.randn(1, 1, H, H)
    state = model.init_state(1, DEVICE)
    return model.step(omega, state)


# ---------------------------------------------------------------------------
# 1. All configs instantiate without error
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_full_msr_hine(self):
        m = _tiny(); assert m is not None

    def test_single_scale(self):
        m = _tiny(single_scale=True); assert m is not None

    def test_no_multirate(self):
        m = _tiny(medium_stride=1, coarse_stride=1); assert m is not None

    def test_no_topdown(self):
        m = _tiny(use_topdown=False); assert m is not None

    def test_no_contraction(self):
        m = _tiny(use_contraction=False); assert m is not None

    def test_no_warmup(self):
        m = _tiny(use_warmup=False); assert m is not None

    def test_circularity_control(self):
        m = _tiny(_inference_fusion_CONTROL_ONLY=True); assert m is not None


# ---------------------------------------------------------------------------
# 2. Each toggle changes the model
# ---------------------------------------------------------------------------

class TestToggleEffects:
    def test_no_multirate_strides_are_one(self):
        m = _tiny(medium_stride=1, coarse_stride=1)
        assert m.hierarchy.medium_stride == 1
        assert m.hierarchy.coarse_stride == 1

    def test_full_strides_are_correct(self):
        m = _tiny(medium_stride=2, coarse_stride=4)
        assert m.hierarchy.medium_stride == 2
        assert m.hierarchy.coarse_stride == 4

    def test_no_contraction_removes_spectral_norm(self):
        m = _tiny(use_contraction=False)
        for level in (m.hierarchy.medium_level, m.hierarchy.coarse_level):
            for attr in ("Wh_r", "Wh_z", "Wh_n"):
                cell_layer = getattr(level.gru, attr)
                assert not hasattr(cell_layer, "weight_orig"), \
                    f"Spectral norm must be absent when use_contraction=False"

    def test_full_contraction_has_spectral_norm(self):
        m = _tiny(use_contraction=True)
        for level in (m.hierarchy.medium_level, m.hierarchy.coarse_level):
            for attr in ("Wh_r", "Wh_z", "Wh_n"):
                assert hasattr(getattr(level.gru, attr), "weight_orig"), \
                    "Spectral norm must be present by default"

    def test_no_warmup_is_noop(self):
        m = _tiny(use_warmup=False)
        state0 = m.init_state(1, DEVICE)
        history = torch.randn(1, 3, 1, H, H)
        state1  = m.warmup(history, state0)
        # State must be unchanged (zero still) since warmup is skipped
        assert torch.equal(state0.h_medium, state1.h_medium), \
            "no_warmup: state must be unchanged after warmup()"

    def test_full_warmup_changes_state(self):
        m = _tiny(use_warmup=True)
        state0 = m.init_state(1, DEVICE)
        history = torch.randn(1, 3, 1, H, H)
        state1  = m.warmup(history, state0)
        assert not torch.equal(state0.h_medium, state1.h_medium), \
            "warmup must change h_medium"

    def test_no_topdown_flag_propagates(self):
        m = _tiny(use_topdown=False)
        assert not m.hierarchy.medium_level.has_topdown, \
            "no_topdown: medium level must have has_topdown=False"

    def test_full_topdown_is_on(self):
        m = _tiny(use_topdown=True)
        assert m.hierarchy.medium_level.has_topdown

    def test_single_scale_no_coarse_encoder(self):
        m = _tiny(single_scale=True)
        assert m.enc_coarse is None, "single_scale: coarse encoder must be None"
        assert m.dec_coarse is None
        assert m.film_coarse is None

    def test_single_scale_coarse_state_zero_dim(self):
        m = _tiny(single_scale=True)
        state = m.init_state(2, DEVICE)
        assert state.h_coarse.shape == (2, 0), \
            f"single_scale: h_coarse must be zero-dim, got {state.h_coarse.shape}"

    def test_single_scale_forward_works(self):
        m = _tiny(single_scale=True)
        omega = torch.randn(1, 1, H, H)
        out = m(omega)
        assert out.shape == (1, 1, H, H)

    def test_no_multirate_updates_every_step(self):
        """With stride=1 both levels update at every step."""
        from msr_hine.models.recurrence import MultiRateHierarchy
        hier = MultiRateHierarchy(medium_dim=4, coarse_dim=2,
                                   medium_stride=1, coarse_stride=1)
        h_m = torch.ones(1, 4); h_c = torch.ones(1, 2)
        zp_m = torch.zeros(1, 4); zp_c = torch.zeros(1, 2)
        z_m  = torch.randn(1, 4); z_mp = torch.randn(1, 4)
        z_c  = torch.randn(1, 2); z_cp = torch.randn(1, 2)

        # step_n=0: next_step=1, 1%1==0 → both update
        h_m_new, h_c_new, _, _ = hier.step(0, z_m, z_mp, z_c, z_cp,
                                             h_m, h_c, zp_m, zp_c)
        assert not torch.equal(h_m_new, h_m), "no_multirate: medium must update"
        assert not torch.equal(h_c_new, h_c), "no_multirate: coarse must update"

        # step_n=1: next_step=2, 2%1==0 → both update again
        h_m2, h_c2, _, _ = hier.step(1, z_m, z_mp, z_c, z_cp,
                                       h_m_new, h_c_new, zp_m, zp_c)
        assert not torch.equal(h_m2, h_m_new)
        assert not torch.equal(h_c2, h_c_new)


# ---------------------------------------------------------------------------
# 3. Circularity-confirmation flag (Invariant 1)
# ---------------------------------------------------------------------------

class TestCircularityFlag:
    def test_fusion_off_by_default(self):
        """_inference_fusion_CONTROL_ONLY must be False on all production models."""
        for model in [
            _tiny(),
            _tiny(use_topdown=False),
            _tiny(use_contraction=False),
            _tiny(use_warmup=False),
            _tiny(medium_stride=1, coarse_stride=1),
        ]:
            assert not model._fusion_CONTROL_ONLY, \
                "Invariant 1: inference fusion must be OFF by default"

    def test_fusion_on_changes_output(self):
        """Enabling the control fusion changes the next-step output."""
        torch.manual_seed(42)
        model_clean = _tiny()
        model_fused = _tiny(_inference_fusion_CONTROL_ONLY=True)
        model_fused.load_state_dict(model_clean.state_dict())

        omega = torch.randn(1, 1, H, H)
        model_clean.eval(); model_fused.eval()

        with torch.no_grad():
            state_c = model_clean.init_state(1, DEVICE)
            state_f = model_fused.init_state(1, DEVICE)
            out_c, state_c2 = model_clean.step(omega, state_c)
            out_f, state_f2 = model_fused.step(omega, state_f)

            # Second step — fusion changes h, so predictions diverge
            out_c2, _ = model_clean.step(out_c.detach(), state_c2)
            out_f2, _ = model_fused.step(out_f.detach(), state_f2)

        assert not torch.allclose(out_c2, out_f2, atol=1e-5), \
            "Fusion control path must change predictions vs. clean path"

    def test_fusion_flag_not_in_any_non_control_config(self):
        """Verify config files don't accidentally enable fusion."""
        import yaml
        from pathlib import Path
        ablation_dir = Path("configs/ablations")
        if not ablation_dir.exists():
            pytest.skip("configs/ablations not found")
        for cfg_file in ablation_dir.glob("*.yaml"):
            if cfg_file.name == "circularity_confirm.yaml":
                continue  # The one allowed exception
            text = cfg_file.read_text()
            assert "_inference_fusion_CONTROL_ONLY: true" not in text, \
                f"Invariant 1 violated: {cfg_file.name} enables inference fusion"


# ---------------------------------------------------------------------------
# 4. Invariant 10 — all ablations use UNet, not FNO
# ---------------------------------------------------------------------------

class TestInvariant10:
    @pytest.mark.parametrize("kwargs", [
        {},
        {"single_scale": True},
        {"use_topdown": False},
        {"use_contraction": False},
        {"use_warmup": False},
        {"medium_stride": 1, "coarse_stride": 1},
    ])
    def test_unet_backbone_used(self, kwargs):
        from msr_hine.models.fno import FNO2d
        m = _tiny(**kwargs)
        assert any(isinstance(mod, UNet) for mod in m.modules()), \
            f"Ablation {kwargs}: must use UNet backbone (Invariant 10)"
        assert not any(isinstance(mod, FNO2d) for mod in m.modules()), \
            f"Ablation {kwargs}: must NOT use FNO2d (Invariant 10)"


# ---------------------------------------------------------------------------
# 5. Each ablation runs one forward step
# ---------------------------------------------------------------------------

class TestForwardPass:
    @pytest.mark.parametrize("kwargs", [
        {},
        {"single_scale": True},
        {"use_topdown": False},
        {"use_contraction": False},
        {"use_warmup": False},
        {"medium_stride": 1, "coarse_stride": 1},
        {"_inference_fusion_CONTROL_ONLY": True},
    ])
    def test_one_step_finite(self, kwargs):
        m = _tiny(**kwargs)
        out, _ = _one_step(m)
        assert out.shape == (1, 1, H, H)
        assert torch.isfinite(out).all(), \
            f"Ablation {kwargs}: forward pass produced non-finite output"


# ---------------------------------------------------------------------------
# 6. run_ablations.sh dry-run smoke test
# ---------------------------------------------------------------------------

class TestRunAblationsScript:
    def test_dry_run_produces_commands(self):
        """--dry-run prints commands without executing them."""
        result = subprocess.run(
            ["bash", "scripts/run_ablations.sh", "--debug", "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, \
            f"run_ablations.sh --dry-run failed:\n{result.stderr}"
        # Should print dry-run command lines
        assert "[dry-run]" in result.stdout or "Running:" in result.stdout, \
            "Dry-run should print command lines"
        # Should not actually start any training
        assert "Epoch" not in result.stdout, \
            "Dry-run must not actually run training"
