"""Train → rollout → metrics end-to-end tests.

Tests
─────
1. Losses: l_state/l_spec/l_highk shapes and values; total_loss dict keys.
2. Scheduled sampling: linear annealing, boundary values.
3. Rollout: correct shape; warmup steps excluded from returned tensor (Invariant 7).
4. Rollout with stateless model: gradients do not pass through (inference path).
5. Overfit tiny dataset: training loss decreases over a few steps.
6. End-to-end: train fno_1step on debug data, evaluate, obtain a VPH in τ_λ (Invariant 8).

All tests use small models (width=4, modes=4, n_layers=1) and small grids (H=32)
to stay fast on CPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from msr_hine.losses import l_highk, l_spec, l_state, total_loss
from msr_hine.models.fno_baseline import FNOBaseline
from msr_hine.rollout import evaluate_trajectory, rollout
from msr_hine.train import scheduled_sampling_prob, tbptt_step, validate_long_rollout

DEVICE = torch.device("cpu")
H = 32     # small spatial grid for fast tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_fno():
    return FNOBaseline(width=4, modes=4, n_layers=1).to(DEVICE)


@pytest.fixture(scope="module")
def tiny_cfg():
    """Minimal Hydra-like DictConfig for training tests."""
    return OmegaConf.create({
        "train": {
            "warmup_steps":    2,
            "rollout_steps":   4,
            "gamma":           0.99,
            "lambda_spec":     0.01,
            "lambda_highk":    0.0,
            "k_c":             8,
            "clip_grad_norm":  1.0,
            "amp":             False,
            "grad_checkpoint": False,
            "learning_rate":   1e-3,
            "weight_decay":    1e-5,
            "scheduled_sampling": {"start_prob": 1.0, "end_prob": 0.0},
        }
    })


# ---------------------------------------------------------------------------
# 1. Losses
# ---------------------------------------------------------------------------

class TestLosses:
    B, K = 2, 4

    def _batch(self, K=None):
        K = K or self.K
        return torch.randn(self.B, K, 1, H, H)

    def test_l_state_zero_on_identical(self):
        x = self._batch()
        assert l_state(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_l_state_positive(self):
        x, y = self._batch(), self._batch()
        assert l_state(x, y).item() > 0

    def test_l_state_discounting(self):
        """Later steps should be down-weighted: loss with γ<1 < loss with γ=1."""
        x, y = self._batch(), self._batch()
        loss_disc = l_state(x, y, gamma=0.5)
        loss_full = l_state(x, y, gamma=1.0)
        assert loss_disc.item() < loss_full.item()

    def test_l_state_output_is_scalar(self):
        x, y = self._batch(), self._batch()
        assert l_state(x, y).shape == ()

    def test_l_spec_zero_on_identical(self):
        x = torch.randn(self.B, 1, H, H)
        assert l_spec(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_l_spec_sequence_accepted(self):
        x, y = self._batch(), self._batch()
        loss = l_spec(x, y)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_l_highk_zero_for_low_k_field(self):
        """A field with only k<=2 energy should have zero high-k loss at k_c=4."""
        k_test = 2
        yy = torch.linspace(0, 2 * math.pi * (H - 1) / H, H)
        _, yy_grid = torch.meshgrid(torch.zeros(H), yy, indexing="ij")
        field = torch.cos(k_test * yy_grid).unsqueeze(0).unsqueeze(0)  # [1,1,H,H]
        loss = l_highk(field, k_c=4)
        assert loss.item() < 1e-10

    def test_l_highk_positive_for_high_k_field(self):
        """Parseval-correct high-k energy of unit cosine is mean(cos²)=0.5."""
        k_test = 10   # above k_c=4
        yy = torch.linspace(0, 2 * math.pi * (H - 1) / H, H)
        _, yy_grid = torch.meshgrid(torch.zeros(H), yy, indexing="ij")
        field = torch.cos(k_test * yy_grid).unsqueeze(0).unsqueeze(0)
        loss = l_highk(field, k_c=4)
        assert loss.item() == pytest.approx(0.5, abs=1e-5)

    def test_total_loss_keys(self):
        x, y = self._batch(), self._batch()
        d = total_loss(x, y)
        for key in ("total", "state", "spec", "highk", "prior", "cons"):
            assert key in d, f"Missing key: {key}"

    def test_total_loss_total_equals_sum(self):
        x, y = self._batch(), self._batch()
        d = total_loss(x, y, lambda_spec=0.01, lambda_hk=0.0)
        expected = d["state"] + 0.01 * d["spec"]
        assert abs(d["total"].item() - expected.item()) < 1e-5

    def test_total_loss_gradients_flow(self):
        model = FNOBaseline(width=4, modes=4, n_layers=1)
        x = torch.randn(1, 1, H, H)
        omega = torch.randn(1, self.K, 1, H, H, requires_grad=True)
        target = torch.randn_like(omega)
        d = total_loss(omega, target)
        d["total"].backward()
        assert omega.grad is not None and omega.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 2. Scheduled sampling
# ---------------------------------------------------------------------------

class TestScheduledSampling:
    def test_start_is_start_prob(self):
        assert scheduled_sampling_prob(0, 100, 1.0, 0.2) == pytest.approx(1.0)

    def test_end_is_end_prob(self):
        assert scheduled_sampling_prob(99, 100, 1.0, 0.2) == pytest.approx(0.2)

    def test_three_stage_boundaries(self):
        assert scheduled_sampling_prob(10, 100, 1.0, 0.2) == pytest.approx(0.8)
        assert scheduled_sampling_prob(30, 100, 1.0, 0.2) == pytest.approx(0.5)
        assert scheduled_sampling_prob(50, 100, 1.0, 0.2) == pytest.approx(0.2)
        assert scheduled_sampling_prob(75, 100, 1.0, 0.2) == pytest.approx(0.2)

    def test_monotone_decreasing(self):
        probs = [scheduled_sampling_prob(e, 100, 1.0, 0.2) for e in range(100)]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_single_epoch(self):
        assert scheduled_sampling_prob(0, 1, 0.8, 0.2) == pytest.approx(0.8)

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            scheduled_sampling_prob(
                0, 100, phase1_frac=0.7, decay_end_frac=0.5)


# ---------------------------------------------------------------------------
# 3. Rollout shapes and Invariant 7
# ---------------------------------------------------------------------------

class TestRollout:
    def test_output_shape(self, tiny_fno):
        omega_seed = torch.randn(2, 1, H, H)
        preds = rollout(tiny_fno, omega_seed, n_steps=5)
        assert preds.shape == (2, 5, 1, H, H)

    def test_warmup_excluded_from_output(self, tiny_fno):
        """Returned tensor must have exactly n_steps frames — warmup is excluded (Invariant 7)."""
        B, W, n_steps = 2, 4, 8
        omega_seed    = torch.randn(B, 1, H, H)
        warmup_frames = torch.randn(B, W, 1, H, H)
        preds = rollout(tiny_fno, omega_seed, n_steps=n_steps,
                        warmup_frames=warmup_frames)
        assert preds.shape == (B, n_steps, 1, H, H), (
            f"Expected ({B}, {n_steps}, 1, {H}, {H}), got {preds.shape}. "
            "Warmup frames must not be included in the returned predictions (Invariant 7)."
        )

    def test_return_all_false(self, tiny_fno):
        omega_seed = torch.randn(1, 1, H, H)
        last = rollout(tiny_fno, omega_seed, n_steps=10, return_all=False)
        assert last.shape == (1, 1, H, H)

    def test_rollout_autoregressive(self, tiny_fno):
        """Step k uses step k-1 output as input (autoregressive property)."""
        omega_seed = torch.randn(1, 1, H, H)
        preds = rollout(tiny_fno, omega_seed, n_steps=3)
        # Manual step-by-step
        manual = []
        omega = omega_seed
        with torch.no_grad():
            for _ in range(3):
                omega = tiny_fno(omega)
                manual.append(omega)
        for k in range(3):
            assert torch.allclose(preds[:, k], manual[k], atol=1e-5), \
                f"Step {k} mismatch: rollout is not autoregressive"

    def test_no_grad_in_rollout(self, tiny_fno):
        """rollout() runs under no_grad — no gradients accumulate."""
        omega_seed = torch.randn(1, 1, H, H)
        preds = rollout(tiny_fno, omega_seed, n_steps=4)
        assert not preds.requires_grad

    def test_finite_output(self, tiny_fno):
        omega_seed = torch.randn(2, 1, H, H)
        preds = rollout(tiny_fno, omega_seed, n_steps=20)
        assert torch.isfinite(preds).all(), "Rollout produced non-finite values"


# ---------------------------------------------------------------------------
# 4. tbptt_step
# ---------------------------------------------------------------------------

class TestTBPTTStep:
    def test_loss_decreases_on_overfit(self, tiny_cfg):
        """Training on a fixed batch should decrease L_state over a few steps."""
        model = FNOBaseline(width=8, modes=4, n_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        warmup, K = tiny_cfg.train.warmup_steps, tiny_cfg.train.rollout_steps
        window = torch.randn(2, warmup + K + 1, H, H)   # [B, T, H, W] — channel added inside

        losses = []
        for _ in range(20):
            result = tbptt_step(model, window.unsqueeze(2), optimizer, tiny_cfg,
                                teacher_forcing_prob=1.0, scaler=None)
            losses.append(result["state"])

        assert losses[-1] < losses[0], (
            f"Loss should decrease when overfitting a fixed batch: "
            f"initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_tbptt_returns_dict(self, tiny_fno, tiny_cfg):
        optimizer = torch.optim.AdamW(tiny_fno.parameters(), lr=1e-4)
        warmup, K = tiny_cfg.train.warmup_steps, tiny_cfg.train.rollout_steps
        window = torch.randn(2, warmup + K + 1, 1, H, H)
        result = tbptt_step(tiny_fno, window, optimizer, tiny_cfg,
                            teacher_forcing_prob=1.0, scaler=None)
        for key in ("total", "state", "spec"):
            assert key in result
        assert all(math.isfinite(v) for v in result.values())

    def test_scheduled_sampling_self_fed(self, tiny_cfg):
        """With tf_prob=0 (self-fed), gradient still flows and loss is finite."""
        model = FNOBaseline(width=4, modes=4, n_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        warmup, K = tiny_cfg.train.warmup_steps, tiny_cfg.train.rollout_steps
        window = torch.randn(2, warmup + K + 1, 1, H, H)
        result = tbptt_step(model, window, optimizer, tiny_cfg,
                            teacher_forcing_prob=0.0, scaler=None)
        assert math.isfinite(result["total"])


# ---------------------------------------------------------------------------
# 5. evaluate_trajectory
# ---------------------------------------------------------------------------

class TestEvaluateTrajectory:
    def test_output_keys(self, tiny_fno):
        T = 20; warmup = 4
        traj = torch.randn(T, 1, H, H)
        result = evaluate_trajectory(tiny_fno, traj, warmup_len=warmup,
                                     tau_lambda_steps=5.0, dt_snapshot=0.1)
        for key in ("rmse", "acc", "vph_acc", "vph_rmse", "n_steps", "warmup_len"):
            assert key in result, f"Missing key: {key}"

    def test_n_steps_excludes_warmup(self, tiny_fno):
        """n_steps = T - warmup_len - 1: seed=frame[W], targets=frames[W+1..T-1]."""
        T = 20; warmup = 4
        traj = torch.randn(T, 1, H, H)
        result = evaluate_trajectory(tiny_fno, traj, warmup_len=warmup,
                                     tau_lambda_steps=5.0)
        assert result["n_steps"] == T - warmup - 1

    def test_rmse_shape(self, tiny_fno):
        T = 15; warmup = 3
        traj = torch.randn(T, 1, H, H)
        result = evaluate_trajectory(tiny_fno, traj, warmup_len=warmup,
                                     tau_lambda_steps=5.0)
        n_steps = T - warmup - 1
        assert result["rmse"].shape == (n_steps,)

    def test_vph_in_tau_lambda_units(self, tiny_fno):
        """VPH dict must contain tau_lambda key (Invariant 8)."""
        traj = torch.randn(15, 1, H, H)
        result = evaluate_trajectory(tiny_fno, traj, warmup_len=3,
                                     tau_lambda_steps=5.0, dt_snapshot=0.1)
        assert "tau_lambda" in result["vph_acc"], "VPH must include tau_lambda (Invariant 8)"
        assert "tau_lambda" in result["vph_rmse"]

    def test_too_short_trajectory_raises(self, tiny_fno):
        """Trajectory with T <= warmup_len should raise ValueError."""
        traj = torch.randn(5, 1, H, H)
        with pytest.raises(ValueError):
            evaluate_trajectory(tiny_fno, traj, warmup_len=5, tau_lambda_steps=1.0)

    def test_long_validation_returns_rollout_metrics(self, tiny_fno):
        trajectories = torch.randn(2, 12, H, H)
        loader = torch.utils.data.DataLoader(
            trajectories, batch_size=1, shuffle=False)
        cfg = OmegaConf.create({
            "train": {
                "warmup_steps": 2,
                "gamma": 0.99,
            },
            "eval": {
                "long_rollout_steps": 4,
                "long_rollout_max_trajs": 2,
                "tau_lambda_steps": 4.0,
            },
        })
        metrics = validate_long_rollout(tiny_fno, loader, cfg, DEVICE)
        for key in (
            "val_long_loss",
            "val_long_rmse",
            "val_long_acc_final",
            "val_long_vph_steps",
        ):
            assert key in metrics
            assert math.isfinite(metrics[key])


# ---------------------------------------------------------------------------
# 6. End-to-end: train → rollout → VPH on debug dataset
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Train an FNO on the debug dataset for a few epochs and get a VPH."""

    @pytest.fixture(scope="class")
    def debug_root(self, tmp_path_factory):
        import subprocess, sys
        root = tmp_path_factory.mktemp("e2e_debug")
        r = subprocess.run(
            [sys.executable, "-m", "msr_hine.data.generate",
             "+debug=true", f"data.dataset_root={root}",
             "hydra.run.dir=.", "hydra/job_logging=disabled",
             "hydra/hydra_logging=disabled"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            pytest.fail(f"Debug dataset gen failed:\n{r.stderr}")
        return root / "debug"

    @pytest.fixture(scope="class")
    def trained_model_and_meta(self, debug_root):
        """Train a tiny FNO for 5 epochs on the debug dataset."""
        from msr_hine.data.dataset import build_dataloaders

        warmup, K = 2, 4
        window = warmup + K + 1

        train_loader, val_loader, test_loader, _ = build_dataloaders(
            root=debug_root, window=window, batch_size=2,
            num_workers=0, stride=2, normalize=False,
        )

        model = FNOBaseline(width=8, modes=4, n_layers=1).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = OmegaConf.create({
            "train": {
                "warmup_steps": warmup, "rollout_steps": K,
                "gamma": 0.99, "lambda_spec": 0.0, "lambda_highk": 0.0,
                "k_c": 8, "clip_grad_norm": 1.0, "amp": False,
                "grad_checkpoint": False,
                "scheduled_sampling": {"start_prob": 1.0, "end_prob": 0.0},
            }
        })

        losses_per_epoch = []
        for epoch in range(5):
            epoch_loss = 0.0; n = 0
            for batch in train_loader:
                omega_window = batch.unsqueeze(2).to(DEVICE)
                result = tbptt_step(model, omega_window, optimizer, cfg,
                                    teacher_forcing_prob=1.0, scaler=None)
                epoch_loss += result["state"]; n += 1
            losses_per_epoch.append(epoch_loss / max(n, 1))

        # Load a test trajectory for evaluation
        import h5py, numpy as np
        with h5py.File(debug_root / "test.h5", "r") as f:
            traj_np = f["vorticity"][0]   # [T, H, W] float32
        traj = torch.from_numpy(traj_np).unsqueeze(1)  # [T, 1, H, W]

        return model, losses_per_epoch, traj

    def test_loss_is_finite(self, trained_model_and_meta):
        _, losses, _ = trained_model_and_meta
        assert all(math.isfinite(l) for l in losses), \
            f"Training produced non-finite loss: {losses}"

    def test_loss_decreases(self, trained_model_and_meta):
        _, losses, _ = trained_model_and_meta
        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 5 epochs: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_rollout_shape(self, trained_model_and_meta):
        model, _, traj = trained_model_and_meta
        T = traj.shape[0]; warmup = 2
        omega_seed = traj[warmup].unsqueeze(0)
        preds = rollout(model, omega_seed, n_steps=5)
        assert preds.shape == (1, 5, 1, traj.shape[2], traj.shape[3])

    def test_evaluate_returns_vph(self, trained_model_and_meta):
        model, _, traj = trained_model_and_meta
        # Use tau_lambda_steps=1 (single snapshot step = 1 τ_λ for debug)
        result = evaluate_trajectory(model, traj, warmup_len=2,
                                     tau_lambda_steps=1.0, dt_snapshot=0.025)
        assert "vph_acc"  in result
        assert "tau_lambda" in result["vph_acc"]
        # VPH must be a positive finite number
        vph = result["vph_acc"]["tau_lambda"]
        assert math.isfinite(vph) and vph >= 0, f"VPH not valid: {vph}"

    def test_warmup_excluded_from_n_steps(self, trained_model_and_meta):
        """Invariant 7: n_steps reported by evaluate_trajectory excludes warmup."""
        model, _, traj = trained_model_and_meta
        T = traj.shape[0]; warmup = 2
        result = evaluate_trajectory(model, traj, warmup_len=warmup,
                                     tau_lambda_steps=1.0)
        assert result["n_steps"] == T - warmup - 1
        assert result["warmup_len"] == warmup
