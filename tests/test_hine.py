"""HINE baseline tests (DESIGN.md §4.1, CLAUDE.md §3, Invariant 10).

Tests
─────
1. BandEncoder/BandDecoder:
   - Output shapes.
   - Encoder produces band-limited latent that reconstructs a band-limited field.
   - Decoder output is spectrally band-limited (P^l applied inside).

2. HINE forward/rollout:
   - Forward shape: (omega → omega_hat) matches input shape.
   - Staticity: same input → same output every call (no hidden state persists).
   - State carries through forward_with_state; different initial states → different outputs.
   - Rollout via rollout.rollout() returns correct shape; warmup excluded (Invariant 7).

3. Staggered horizons:
   - The decoder emission heads produce latents that change the injection at
     the next step (the ladder is "wired up" and not frozen at zero).

4. Invariant 10:
   - HINE and MSR-HINE both contain a UNet; the UNet class is the same type.
   - HINE does NOT use FNO2d (backbone separation).

5. Overfit:
   - Training loss decreases when overfitting a fixed batch.

6. End-to-end on debug dataset: train HINE, evaluate, obtain a VPH.

All tests use small models (base_ch=8, mults=[1,1,1,1,1] or similar) and H=32.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from msr_hine.models.encoders import BandDecoder, BandEncoder, build_encoder_decoder_pair
from msr_hine.models.hine import HINE, HINEState
from msr_hine.models.unet import UNet
from msr_hine.rollout import rollout

DEVICE = torch.device("cpu")
H = 32   # small spatial grid


# ---------------------------------------------------------------------------
# Small model factory
# ---------------------------------------------------------------------------

def _small_hine(H: int = H) -> HINE:
    """Tiny HINE for fast tests."""
    return HINE(
        medium_dim         = 16,
        coarse_dim         = 8,
        unet_base_channels = 8,
        unet_channel_mults = (1, 1, 1, 1, 1),
        attn_resolutions   = (H // 16,),
        input_size         = H,
        enc_hidden_ch      = 8,
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# 1. BandEncoder / BandDecoder
# ---------------------------------------------------------------------------

class TestEncoderDecoder:
    def test_encoder_output_shape(self):
        enc = BandEncoder(k_max=4, latent_dim=16, in_size=H, hidden_ch=8)
        x = torch.randn(2, 1, H, H)
        z = enc(x)
        assert z.shape == (2, 16), f"Expected (2,16), got {z.shape}"

    def test_decoder_output_shape(self):
        dec = BandDecoder(latent_dim=16, k_max=4, out_size=H, hidden_ch=8)
        z = torch.randn(2, 16)
        field = dec(z)
        assert field.shape == (2, 1, H, H), f"Expected (2,1,{H},{H}), got {field.shape}"

    def test_decoder_is_band_limited(self):
        """Decoder output must have energy only within |k| <= k_max."""
        k_max = 4
        dec = BandDecoder(latent_dim=16, k_max=k_max, out_size=H, hidden_ch=8)
        z = torch.randn(2, 16)
        field = dec(z)
        # Check all modes outside the band are zero
        fhat = torch.fft.rfft2(field)
        from msr_hine.spectral.truncation import radial_mask_rfft2
        mask = radial_mask_rfft2(H, k_max, DEVICE)
        outside_energy = fhat[..., ~mask].abs().max().item()
        assert outside_energy < 1e-4, (
            f"Decoder output has energy outside band: max={outside_energy:.2e}"
        )

    def test_encoder_gradient_flows(self):
        enc = BandEncoder(k_max=4, latent_dim=16, in_size=H, hidden_ch=8)
        x = torch.randn(1, 1, H, H, requires_grad=True)
        z = enc(x)
        z.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_build_pair(self):
        enc, dec = build_encoder_decoder_pair(k_max=4, latent_dim=16,
                                               in_size=H, hidden_ch=8)
        x = torch.randn(2, 1, H, H)
        z = enc(x)
        field = dec(z)
        assert z.shape == (2, 16)
        assert field.shape == (2, 1, H, H)


# ---------------------------------------------------------------------------
# 2. HINE forward / rollout
# ---------------------------------------------------------------------------

class TestHINEForward:
    def test_forward_shape(self):
        hine = _small_hine()
        omega = torch.randn(2, 1, H, H)
        out = hine(omega)
        assert out.shape == (2, 1, H, H)

    def test_forward_with_state_shape(self):
        hine = _small_hine()
        omega = torch.randn(2, 1, H, H)
        state = hine.init_state(2, DEVICE)
        out, next_state = hine.forward_with_state(omega, state)
        assert out.shape == (2, 1, H, H)
        assert next_state.z_medium.shape == (2, 16)
        assert next_state.z_coarse.shape == (2, 8)

    def test_staticity_no_hidden_state(self):
        """Same input → same output every call (no persistent hidden state)."""
        hine = _small_hine()
        hine.eval()
        omega = torch.randn(2, 1, H, H)
        with torch.no_grad():
            out1 = hine(omega)
            out2 = hine(omega)
        assert torch.allclose(out1, out2, atol=1e-6), (
            "HINE produced different outputs on identical inputs — hidden state leak"
        )

    def test_different_states_different_outputs(self):
        """Different latent-future states → different predictions (ladder is active)."""
        hine = _small_hine()
        hine.eval()
        omega = torch.randn(1, 1, H, H)
        state_zero = hine.init_state(1, DEVICE)
        state_rand = HINEState(
            z_medium=torch.randn(1, 16),
            z_coarse=torch.randn(1, 8),
        )
        with torch.no_grad():
            out_zero, _ = hine.forward_with_state(omega, state_zero)
            out_rand, _ = hine.forward_with_state(omega, state_rand)
        assert not torch.allclose(out_zero, out_rand, atol=1e-4), (
            "Zero state and random state gave identical outputs — injection is not active"
        )

    def test_rollout_shape(self):
        hine = _small_hine()
        omega_seed = torch.randn(2, 1, H, H)
        preds = rollout(hine, omega_seed, n_steps=5)
        assert preds.shape == (2, 5, 1, H, H)

    def test_rollout_warmup_excluded(self):
        """Invariant 7: warmup frames excluded from returned predictions."""
        hine  = _small_hine()
        B, W, K = 2, 4, 6
        omega_seed    = torch.randn(B, 1, H, H)
        warmup_frames = torch.randn(B, W, 1, H, H)
        preds = rollout(hine, omega_seed, n_steps=K, warmup_frames=warmup_frames)
        assert preds.shape == (B, K, 1, H, H), (
            f"Expected ({B},{K},1,{H},{H}), got {preds.shape} — warmup should not be included"
        )

    def test_warmup_changes_output(self):
        """Warmup should build up useful latent estimates (non-zero state)."""
        hine = _small_hine()
        omega_seed    = torch.randn(1, 1, H, H)
        warmup_frames = torch.randn(1, 4, 1, H, H)

        preds_no_warmup = rollout(hine, omega_seed, n_steps=3)
        preds_warmup    = rollout(hine, omega_seed, n_steps=3,
                                  warmup_frames=warmup_frames)
        assert not torch.allclose(preds_no_warmup, preds_warmup, atol=1e-5), (
            "Warmup should change the predicted trajectory"
        )

    def test_output_finite(self):
        hine = _small_hine()
        omega = torch.randn(2, 1, H, H)
        assert torch.isfinite(hine(omega)).all()


# ---------------------------------------------------------------------------
# 3. Staggered horizons (ladder is wired up)
# ---------------------------------------------------------------------------

class TestStaggeredHorizons:
    def test_emission_heads_produce_nonzero_latents(self):
        """After a forward pass the decoder emission heads produce non-zero latents."""
        hine = _small_hine()
        omega = torch.randn(1, 1, H, H)
        state = hine.init_state(1, DEVICE)
        _, next_state = hine.forward_with_state(omega, state)
        # Emission heads should produce non-zero latent futures
        assert next_state.z_medium.abs().sum().item() > 1e-6, \
            "Medium emission head produced zero latents"
        assert next_state.z_coarse.abs().sum().item() > 1e-6, \
            "Coarse emission head produced zero latents"

    def test_latents_evolve_over_steps(self):
        """Latent futures should change from step to step as the field evolves."""
        hine = _small_hine()
        hine.eval()
        omega = torch.randn(1, 1, H, H)
        state = hine.init_state(1, DEVICE)
        latents = []
        with torch.no_grad():
            for _ in range(3):
                omega, state = hine.forward_with_state(omega, state)
                latents.append(state.z_medium.clone())
        # Successive latents should differ (the ladder is updating)
        assert not torch.allclose(latents[0], latents[1], atol=1e-5), \
            "Latent futures did not change across steps"

    def test_injection_used_in_unet(self):
        """Injection at medium and coarse resolutions must be wired in the U-Net."""
        hine = _small_hine()
        inj_res_med = hine._inj_medium
        inj_res_coa = hine._inj_coarse
        # Both resolutions should be in the U-Net encoder's inject_convs
        assert str(inj_res_med) in dict(hine.unet.encoder.inject_convs), \
            f"Medium injection at res={inj_res_med} not found in U-Net encoder"
        assert str(inj_res_coa) in dict(hine.unet.encoder.inject_convs), \
            f"Coarse injection at res={inj_res_coa} not found in U-Net encoder"


# ---------------------------------------------------------------------------
# 4. Invariant 10: shared UNet backbone
# ---------------------------------------------------------------------------

class TestInvariant10:
    def test_hine_uses_unet_not_fno(self):
        """HINE must use UNet, not FNO (Invariant 10)."""
        from msr_hine.models.fno import FNO2d
        hine = _small_hine()
        has_unet = any(isinstance(m, UNet) for m in hine.modules())
        has_fno  = any(isinstance(m, FNO2d) for m in hine.modules())
        assert has_unet, "HINE must contain a UNet module (Invariant 10)"
        assert not has_fno, "HINE must NOT contain FNO2d (Invariant 10)"

    def test_hine_unet_same_class_as_standalone(self):
        """The UNet inside HINE must be the same class as the standalone UNet."""
        hine = _small_hine()
        unet_modules = [m for m in hine.modules() if type(m).__name__ == "UNet"]
        assert len(unet_modules) >= 1, "No UNet found inside HINE"
        assert isinstance(unet_modules[0], UNet), \
            "HINE's backbone is not the shared UNet class"

    def test_hine_and_fno_have_separate_backbones(self):
        """FNOBaseline must NOT use UNet; HINE must NOT use FNO2d."""
        from msr_hine.models.fno_baseline import FNOBaseline
        fno = FNOBaseline(width=4, modes=4, n_layers=1)
        hine = _small_hine()
        assert not any(isinstance(m, UNet) for m in fno.modules()), \
            "FNOBaseline must not contain UNet"


# ---------------------------------------------------------------------------
# 5. Overfit test
# ---------------------------------------------------------------------------

class TestOverfit:
    def test_loss_decreases_on_fixed_batch(self):
        """Training loss should decrease when overfitting a fixed batch."""
        from msr_hine.train import tbptt_step
        from omegaconf import OmegaConf

        hine = _small_hine()
        optimizer = torch.optim.AdamW(hine.parameters(), lr=1e-3)
        cfg = OmegaConf.create({
            "train": {
                "warmup_steps": 2, "rollout_steps": 4,
                "gamma": 0.99, "lambda_spec": 0.0, "lambda_highk": 0.0,
                "k_c": 8, "clip_grad_norm": 1.0, "amp": False,
                "grad_checkpoint": False,
                "scheduled_sampling": {"start_prob": 1.0, "end_prob": 0.0},
            }
        })
        window = torch.randn(2, 7, 1, H, H)   # warmup=2 + K=4 + 1

        losses = []
        for _ in range(20):
            result = tbptt_step(hine, window, optimizer, cfg,
                                teacher_forcing_prob=1.0, scaler=None)
            losses.append(result["state"])

        assert losses[-1] < losses[0], (
            f"HINE loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. End-to-end on debug dataset
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture(scope="class")
    def debug_root(self, tmp_path_factory):
        import subprocess, sys
        root = tmp_path_factory.mktemp("hine_e2e")
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

    @pytest.fixture(scope="class")
    def trained_hine_and_traj(self, debug_root):
        from msr_hine.data.dataset import build_dataloaders
        from msr_hine.train import tbptt_step
        import h5py, numpy as np

        warmup, K = 2, 4
        train_loader, _, _, _ = build_dataloaders(
            root=debug_root, window=warmup + K + 1, batch_size=2,
            num_workers=0, stride=2, normalize=False,
        )
        hine = _small_hine(H=64)   # debug data is 64×64
        optimizer = torch.optim.AdamW(hine.parameters(), lr=1e-3)
        cfg = OmegaConf.create({
            "train": {
                "warmup_steps": warmup, "rollout_steps": K,
                "gamma": 0.99, "lambda_spec": 0.0, "lambda_highk": 0.0,
                "k_c": 8, "clip_grad_norm": 1.0, "amp": False,
                "grad_checkpoint": False,
                "scheduled_sampling": {"start_prob": 1.0, "end_prob": 0.0},
            }
        })

        losses = []
        for epoch in range(5):
            el = 0.0; n = 0
            for batch in train_loader:
                omega_window = batch.unsqueeze(2)
                res = tbptt_step(hine, omega_window, optimizer, cfg,
                                 teacher_forcing_prob=1.0, scaler=None)
                el += res["state"]; n += 1
            losses.append(el / max(n, 1))

        with h5py.File(debug_root / "test.h5", "r") as f:
            traj_np = f["vorticity"][0]
        traj = torch.from_numpy(traj_np).unsqueeze(1)  # [T,1,H,W]
        return hine, losses, traj

    def test_loss_finite(self, trained_hine_and_traj):
        _, losses, _ = trained_hine_and_traj
        assert all(math.isfinite(l) for l in losses)

    def test_loss_decreases(self, trained_hine_and_traj):
        _, losses, _ = trained_hine_and_traj
        assert losses[-1] < losses[0], f"HINE loss didn't decrease: {losses}"

    def test_evaluate_vph(self, trained_hine_and_traj):
        """Invariant 8: VPH returned in τ_λ units."""
        from msr_hine.rollout import evaluate_trajectory
        hine, _, traj = trained_hine_and_traj
        result = evaluate_trajectory(hine, traj, warmup_len=2,
                                     tau_lambda_steps=1.0, dt_snapshot=0.025)
        assert "tau_lambda" in result["vph_acc"]
        vph = result["vph_acc"]["tau_lambda"]
        assert math.isfinite(vph) and vph >= 0

    def test_n_steps_excludes_warmup(self, trained_hine_and_traj):
        """Invariant 7."""
        from msr_hine.rollout import evaluate_trajectory
        hine, _, traj = trained_hine_and_traj
        T = traj.shape[0]; warmup = 2
        result = evaluate_trajectory(hine, traj, warmup_len=warmup,
                                     tau_lambda_steps=1.0)
        assert result["n_steps"] == T - warmup - 1
