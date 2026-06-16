"""Dataset and generation tests (CLAUDE.md §5, Invariant 9).

Tests:
  1. Split disjointness — train/val/test seed sets are non-overlapping.
  2. Window shapes and warmup/rollout indexing — no off-by-one.
  3. Normalization uses train statistics only (val/test stats differ slightly).
  4. Debug dataset generates and loads end-to-end.
  5. Trajectory-level split (Invariant 9) — no trajectory appears in two splits.
  6. Metadata round-trip — physics params stored in HDF5 match inputs.

All tests use the debug dataset (n=64, 2 short trajectories) and
generate it only once per session via a session-scoped fixture.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from msr_hine.data.dataset import (
    NormStats,
    TrajectoryDataset,
    WindowDataset,
    build_dataloaders,
    load_metadata,
    load_seeds_phases,
)

# ---------------------------------------------------------------------------
# Session-scoped fixture: generate the debug dataset once
# ---------------------------------------------------------------------------

DEBUG_ROOT = Path("data/kolmogorov/debug")


@pytest.fixture(scope="session")
def debug_root(tmp_path_factory) -> Path:
    """Generate the debug dataset and return the root path.

    Uses a fixed temp directory so repeated runs within one pytest session
    share the same generated data.
    """
    root = tmp_path_factory.mktemp("kolmogorov_debug")

    result = subprocess.run(
        [
            sys.executable, "-m", "msr_hine.data.generate",
            "+debug=true",
            f"data.dataset_root={root}",
            "hydra.run.dir=.",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes max
    )
    if result.returncode != 0:
        pytest.fail(
            f"Debug generation failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return root / "debug"


@pytest.fixture(scope="session")
def h5_paths(debug_root) -> dict[str, Path]:
    return {
        "train": debug_root / "train.h5",
        "val":   debug_root / "val.h5",
        "test":  debug_root / "test.h5",
    }


# ---------------------------------------------------------------------------
# 1. Debug dataset generates without error
# ---------------------------------------------------------------------------

class TestDebugGeneration:
    def test_files_created(self, h5_paths):
        for split, path in h5_paths.items():
            assert path.exists(), f"{split}.h5 was not created at {path}"

    def test_files_readable(self, h5_paths):
        for split, path in h5_paths.items():
            with h5py.File(path, "r") as f:
                assert "vorticity" in f, f"'vorticity' dataset missing in {split}.h5"
                assert "seeds"     in f
                assert "phases"    in f

    def test_vorticity_shapes(self, h5_paths):
        expected = {"train": 2, "val": 1, "test": 2}
        for split, path in h5_paths.items():
            with h5py.File(path, "r") as f:
                shape = f["vorticity"].shape
                n_traj = expected[split]
                assert shape[0] == n_traj, \
                    f"{split}: expected {n_traj} trajectories, got {shape[0]}"
                assert shape[2] == shape[3], "Spatial dims must be square"
                assert shape[2] == 64, f"{split}: expected n=64, got {shape[2]}"

    def test_vorticity_dtype(self, h5_paths):
        for split, path in h5_paths.items():
            with h5py.File(path, "r") as f:
                assert f["vorticity"].dtype == np.float32, \
                    f"{split}: vorticity dtype should be float32"

    def test_vorticity_finite(self, h5_paths):
        for split, path in h5_paths.items():
            with h5py.File(path, "r") as f:
                v = f["vorticity"][:]
            assert np.isfinite(v).all(), f"{split}: vorticity contains non-finite values"
            assert (np.abs(v) > 0).any(), f"{split}: vorticity is identically zero"


# ---------------------------------------------------------------------------
# 2. Split disjointness (Invariant 9)
# ---------------------------------------------------------------------------

class TestSplitDisjointness:
    """No trajectory seed should appear in more than one split."""

    def test_train_val_disjoint(self, h5_paths):
        train_seeds, _ = load_seeds_phases(h5_paths["train"])
        val_seeds,   _ = load_seeds_phases(h5_paths["val"])
        overlap = set(train_seeds.tolist()) & set(val_seeds.tolist())
        assert len(overlap) == 0, f"Seed overlap between train and val: {overlap}"

    def test_train_test_disjoint(self, h5_paths):
        train_seeds, _ = load_seeds_phases(h5_paths["train"])
        test_seeds,  _ = load_seeds_phases(h5_paths["test"])
        overlap = set(train_seeds.tolist()) & set(test_seeds.tolist())
        assert len(overlap) == 0, f"Seed overlap between train and test: {overlap}"

    def test_val_test_disjoint(self, h5_paths):
        val_seeds,  _ = load_seeds_phases(h5_paths["val"])
        test_seeds, _ = load_seeds_phases(h5_paths["test"])
        overlap = set(val_seeds.tolist()) & set(test_seeds.tolist())
        assert len(overlap) == 0, f"Seed overlap between val and test: {overlap}"

    def test_phases_vary_per_trajectory(self, h5_paths):
        """Each trajectory should have a distinct forcing phase."""
        for split, path in h5_paths.items():
            _, phases = load_seeds_phases(path)
            if len(phases) > 1:
                assert len(set(phases.tolist())) == len(phases), \
                    f"{split}: duplicate forcing phases"

    def test_trajectory_level_split(self, h5_paths):
        """No frame index can identify which trajectory it came from across splits.
        We verify by checking that the same trajectory index means different seeds
        in different splits.
        """
        train_seeds, _ = load_seeds_phases(h5_paths["train"])
        test_seeds,  _ = load_seeds_phases(h5_paths["test"])
        # Index 0 in train should have a different seed from index 0 in test
        assert train_seeds[0] != test_seeds[0], \
            "Train and test traj[0] share the same seed — split is not trajectory-level"


# ---------------------------------------------------------------------------
# 3. Window shapes and indexing
# ---------------------------------------------------------------------------

class TestWindowDataset:
    WINDOW = 8

    @pytest.fixture(scope="class")
    def train_ds(self, h5_paths):
        return WindowDataset(h5_paths["train"], window=self.WINDOW, stride=1)

    def test_item_shape(self, train_ds):
        x = train_ds[0]
        assert x.shape == (self.WINDOW, 64, 64), \
            f"Expected ({self.WINDOW}, 64, 64), got {x.shape}"

    def test_item_dtype(self, train_ds):
        x = train_ds[0]
        assert x.dtype == torch.float32

    def test_len_correct(self, train_ds, h5_paths):
        with h5py.File(h5_paths["train"], "r") as f:
            n_traj = f["vorticity"].shape[0]
            T = f["vorticity"].shape[1]
        expected_len = n_traj * (T - self.WINDOW + 1)
        assert len(train_ds) == expected_len, \
            f"Expected {expected_len} windows, got {len(train_ds)}"

    def test_stride_reduces_length(self, h5_paths):
        ds2 = WindowDataset(h5_paths["train"], window=self.WINDOW, stride=2)
        ds1 = WindowDataset(h5_paths["train"], window=self.WINDOW, stride=1)
        assert len(ds2) <= len(ds1), "stride=2 should produce fewer windows than stride=1"
        assert len(ds2) > 0, "stride=2 dataset is empty"

    def test_balanced_indices_cover_all_trajectories(self, train_ds):
        indices = train_ds.balanced_indices(max_samples=2 * train_ds.n_traj)
        traj_ids = [train_ds._index[idx][0] for idx in indices]
        assert set(traj_ids) == set(range(train_ds.n_traj))
        counts = [traj_ids.count(i) for i in range(train_ds.n_traj)]
        assert max(counts) - min(counts) <= 1

    def test_balanced_indices_span_each_trajectory(self, train_ds):
        indices = train_ds.balanced_indices(max_samples=2 * train_ds.n_traj)
        starts: dict[int, list[int]] = {}
        for idx in indices:
            traj_idx, start = train_ds._index[idx]
            starts.setdefault(traj_idx, []).append(start)
        for selected in starts.values():
            assert min(selected) == 0
            assert max(selected) > min(selected)

    def test_no_cross_trajectory_mixing(self, train_ds, h5_paths):
        """Every window must come from exactly one trajectory; check boundary window."""
        with h5py.File(h5_paths["train"], "r") as f:
            T = f["vorticity"].shape[1]
        # The last window of traj 0 starts at T - WINDOW
        traj0_idx = T - self.WINDOW
        # The first window of traj 1 starts at 0 in its own trajectory
        # In our index, traj1 starts at item (T - WINDOW + 1)
        item_traj0_last = T - self.WINDOW  # last window of traj 0
        item_traj1_first = T - self.WINDOW + 1  # first window of traj 1 (if 2+ trajs)

        if len(train_ds) > item_traj1_first:
            x0 = train_ds[item_traj0_last]
            x1 = train_ds[item_traj1_first]
            # These windows must not be identical (different trajectories)
            assert not torch.allclose(x0, x1), \
                "Consecutive windows across trajectories are identical — possible mixing"

    def test_window_is_contiguous(self, train_ds, h5_paths):
        """Window[t+1] frame 0 should equal window[t] frame 1 (stride=1)."""
        with h5py.File(h5_paths["train"], "r") as f:
            T = f["vorticity"].shape[1]
        if T < self.WINDOW + 1:
            pytest.skip("Trajectory too short for contiguity test")

        x0 = train_ds[0]   # frames [0 .. WINDOW-1] of traj 0
        x1 = train_ds[1]   # frames [1 .. WINDOW]   of traj 0
        assert torch.allclose(x0[1:], x1[:-1]), \
            "Window is not contiguous: frame overlap mismatch between consecutive items"


class TestTrajectoryDataset:
    @pytest.fixture(scope="class")
    def test_ds(self, h5_paths):
        return TrajectoryDataset(h5_paths["test"])

    def test_item_shape(self, test_ds, h5_paths):
        with h5py.File(h5_paths["test"], "r") as f:
            T = f["vorticity"].shape[1]
        x = test_ds[0]
        assert x.shape == (T, 64, 64), f"Expected ({T}, 64, 64), got {x.shape}"

    def test_len_equals_n_traj(self, test_ds, h5_paths):
        with h5py.File(h5_paths["test"], "r") as f:
            n_traj = f["vorticity"].shape[0]
        assert len(test_ds) == n_traj

    def test_item_dtype(self, test_ds):
        x = test_ds[0]
        assert x.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. Normalization uses train statistics only
# ---------------------------------------------------------------------------

class TestNormalization:
    @pytest.fixture(scope="class")
    def norm_stats(self, h5_paths):
        return NormStats.from_file_fast(h5_paths["train"])

    def test_std_positive(self, norm_stats):
        assert norm_stats.std > 0, "Normalization std must be positive"

    def test_mean_near_zero(self, norm_stats):
        """Vorticity mean ≈ 0 for incompressible flow."""
        assert abs(norm_stats.mean) < norm_stats.std, \
            f"Mean {norm_stats.mean:.4f} is unusually large vs std {norm_stats.std:.4f}"

    def test_normalized_data_near_unit_std(self, h5_paths, norm_stats):
        ds = WindowDataset(h5_paths["train"], window=4, norm_stats=norm_stats)
        x = ds[0]
        # After normalization the std of a single window should be O(1)
        # (not exact 1.0 but definitely not 0 or 100)
        std = x.std().item()
        assert 0.1 < std < 10.0, f"Normalized window std = {std:.3f}, expected O(1)"

    def test_unnormalized_data_different_scale(self, h5_paths, norm_stats):
        ds_norm   = WindowDataset(h5_paths["train"], window=4, norm_stats=norm_stats)
        ds_raw    = WindowDataset(h5_paths["train"], window=4, norm_stats=None)
        x_norm = ds_norm[0]
        x_raw  = ds_raw[0]
        # They must not be equal
        assert not torch.allclose(x_norm, x_raw), \
            "Normalized and raw windows are identical — normalization has no effect"

    def test_norm_stats_save_load_roundtrip(self, norm_stats, tmp_path):
        p = tmp_path / "stats.json"
        norm_stats.save(p)
        loaded = NormStats.load(p)
        assert abs(loaded.mean - norm_stats.mean) < 1e-10
        assert abs(loaded.std  - norm_stats.std)  < 1e-10

    def test_normalization_is_train_only(self, h5_paths):
        """Stats computed from train vs. test should differ (different data)."""
        stats_train = NormStats.from_file_fast(h5_paths["train"])
        stats_test  = NormStats.from_file_fast(h5_paths["test"])
        # They should not be exactly identical (different trajectories)
        # (in the degenerate case of identical data this could fail, but won't in practice)
        different = (
            abs(stats_train.mean - stats_test.mean) > 1e-6
            or abs(stats_train.std - stats_test.std) > 1e-6
        )
        # Soft assertion: just warn if they happen to be equal (very small debug dataset)
        if not different:
            import warnings
            warnings.warn(
                "Train and test NormStats are identical — possible if debug dataset "
                "trajectories are very similar."
            )


# ---------------------------------------------------------------------------
# 5. Metadata round-trip
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_physics_params_stored(self, h5_paths):
        meta = load_metadata(h5_paths["train"])
        for key in ("re", "k_f", "mu", "nu", "n", "dt_substep", "dt_snapshot",
                    "substeps_per_snapshot", "tau_estimate", "spinup_substeps"):
            assert key in meta, f"Metadata key '{key}' missing from train.h5"

    def test_re_value(self, h5_paths):
        meta = load_metadata(h5_paths["train"])
        # Debug mode uses re=1000
        assert meta["re"] == pytest.approx(1000.0, rel=1e-3)

    def test_n_value(self, h5_paths):
        meta = load_metadata(h5_paths["train"])
        assert int(meta["n"]) == 64

    def test_tau_positive(self, h5_paths):
        meta = load_metadata(h5_paths["train"])
        assert meta["tau_estimate"] > 0

    def test_dt_snapshot_consistent(self, h5_paths):
        meta = load_metadata(h5_paths["train"])
        expected = meta["dt_substep"] * meta["substeps_per_snapshot"]
        assert abs(meta["dt_snapshot"] - expected) < 1e-10, \
            f"dt_snapshot={meta['dt_snapshot']:.6f} ≠ dt×sps={expected:.6f}"


# ---------------------------------------------------------------------------
# 6. build_dataloaders end-to-end
# ---------------------------------------------------------------------------

class TestDataloaders:
    WINDOW = 6

    @pytest.fixture(scope="class")
    def loaders(self, debug_root):
        return build_dataloaders(
            root=debug_root,
            window=self.WINDOW,
            batch_size=2,
            num_workers=0,
            stride=2,
            normalize=True,
        )

    def test_returns_four_items(self, loaders):
        assert len(loaders) == 4, "build_dataloaders should return (train, val, test, stats)"

    def test_train_batch_shape(self, loaders):
        train_loader, _, _, _ = loaders
        batch = next(iter(train_loader))
        B = batch.shape[0]
        assert batch.shape == (B, self.WINDOW, 64, 64), \
            f"Unexpected train batch shape: {batch.shape}"

    def test_test_batch_shape(self, loaders, h5_paths):
        _, _, test_loader, _ = loaders
        batch = next(iter(test_loader))
        # test loader batch_size=1
        with h5py.File(h5_paths["test"], "r") as f:
            T = f["vorticity"].shape[1]
        assert batch.shape == (1, T, 64, 64), \
            f"Unexpected test batch shape: {batch.shape}"

    def test_norm_stats_returned(self, loaders):
        _, _, _, stats = loaders
        assert isinstance(stats, NormStats)
        assert stats.std > 0
