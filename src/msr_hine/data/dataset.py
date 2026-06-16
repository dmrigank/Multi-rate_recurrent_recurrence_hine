"""HDF5-backed dataset for vorticity trajectories.

Trajectory-level train/val/test splits only (Invariant 9 — never frame-level).

Two Dataset classes:
    WindowDataset:     yields (warmup_window + rollout_window) frames for TBPTT training.
    TrajectoryDataset: yields full trajectories for rollout evaluation.

Normalization (mean=0 by incompressibility; std from train split only):
    NormStats.from_file(train_h5_path) — computes running std over all train frames.
    NormStats can be saved/loaded as a small JSON so it is computed once.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Vorticity normalization: mean and std computed from the training split only.

    Vorticity has zero spatial mean (incompressible, periodic), so mean≈0.
    std is used to normalize inputs to O(1) range.
    """
    mean: float
    std: float

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = json.loads(path.read_text())
        return cls(**d)

    @classmethod
    def from_file(cls, h5_path: Path, sample_frac: float = 1.0) -> "NormStats":
        """Compute mean and std from all vorticity frames in an HDF5 file.

        Uses Welford's online algorithm to avoid loading everything at once.

        Args:
            h5_path: Path to a split HDF5 file (train split only).
            sample_frac: Fraction of frames to sample (1.0 = all).

        Returns:
            NormStats with mean and std.
        """
        with h5py.File(h5_path, "r") as f:
            vort = f["vorticity"]  # [N_traj, T, n, n]
            n_traj, T, n, _ = vort.shape

            running_mean = 0.0
            running_m2 = 0.0
            count = 0

            for i in range(n_traj):
                # Load one trajectory at a time to bound memory
                traj = vort[i]  # [T, n, n] float32
                if sample_frac < 1.0:
                    n_keep = max(1, int(T * sample_frac))
                    idx = np.linspace(0, T - 1, n_keep, dtype=int)
                    traj = traj[idx]

                arr = traj.ravel().astype(np.float64)
                for val in arr:
                    count += 1
                    delta = val - running_mean
                    running_mean += delta / count
                    running_m2 += delta * (val - running_mean)

        std = math.sqrt(running_m2 / max(count - 1, 1))
        log.info(
            "NormStats from %s: mean=%.6f  std=%.6f  (n_frames=%d)",
            h5_path.name, running_mean, std, count // (n * n),
        )
        return cls(mean=float(running_mean), std=float(std))

    @classmethod
    def from_file_fast(cls, h5_path: Path) -> "NormStats":
        """Faster version: loads all data at once (requires enough RAM)."""
        with h5py.File(h5_path, "r") as f:
            vort = f["vorticity"][:]  # [N_traj, T, n, n] float32
        mean = float(vort.mean())
        std = float(vort.std())
        log.info("NormStats (fast): mean=%.6f  std=%.6f", mean, std)
        return cls(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Window Dataset (for TBPTT training)
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Sliding-window dataset yielding [window, n, n] vorticity tensors.

    A window covers warmup_len + rollout_len consecutive frames from a single
    trajectory. Windows are slid with a configurable stride.

    Trajectories are never mixed across splits (Invariant 9).

    Args:
        h5_path: HDF5 file for this split (train or val).
        window: Total frames per item = warmup_len + rollout_len.
        stride: Step between window start indices (default 1).
        norm_stats: If provided, normalize vorticity to zero-mean unit-std.
    """

    def __init__(
        self,
        h5_path: Path,
        window: int,
        stride: int = 1,
        norm_stats: Optional[NormStats] = None,
    ) -> None:
        self.h5_path = h5_path
        self.window = window
        self.stride = stride
        self.norm_stats = norm_stats

        # Build index: list of (traj_idx, start_frame)
        with h5py.File(h5_path, "r") as f:
            self._n_traj: int = int(f["vorticity"].shape[0])
            self._T: int      = int(f["vorticity"].shape[1])
            self._n: int      = int(f["vorticity"].shape[2])

        self._index: list[tuple[int, int]] = []
        for t in range(self._n_traj):
            for start in range(0, self._T - window + 1, stride):
                self._index.append((t, start))

        # HDF5 file handle opened lazily (one per worker in DataLoader)
        self._file: Optional[h5py.File] = None

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tensor:
        """Return vorticity window [window, n, n] float32."""
        traj_idx, start = self._index[idx]
        f = self._open()
        # Read a contiguous slice from one trajectory — no cross-trajectory mixing
        frames = f["vorticity"][traj_idx, start : start + self.window]  # [window, n, n]
        x = torch.from_numpy(frames.astype(np.float32))
        if self.norm_stats is not None:
            x = self.norm_stats.normalize(x)
        return x

    def balanced_indices(self, max_samples: int) -> list[int]:
        """Return deterministic window indices distributed across trajectories.

        Within each trajectory, selected windows are spread from beginning to
        end. Results are interleaved by trajectory so even a partial final
        batch remains representative.
        """
        if max_samples <= 0 or len(self._index) == 0:
            return []
        if max_samples >= len(self._index):
            return list(range(len(self._index)))

        by_traj: list[list[int]] = [[] for _ in range(self._n_traj)]
        for dataset_idx, (traj_idx, _) in enumerate(self._index):
            by_traj[traj_idx].append(dataset_idx)

        base, remainder = divmod(max_samples, self._n_traj)
        selected_by_traj: list[list[int]] = []
        for traj_idx, candidates in enumerate(by_traj):
            quota = base + (1 if traj_idx < remainder else 0)
            quota = min(quota, len(candidates))
            if quota == 0:
                selected_by_traj.append([])
                continue
            positions = np.linspace(0, len(candidates) - 1, quota, dtype=int)
            selected_by_traj.append([candidates[pos] for pos in positions])

        return [
            idx
            for group in zip_longest(*selected_by_traj)
            for idx in group
            if idx is not None
        ][:max_samples]

    @property
    def n_traj(self) -> int:
        return self._n_traj

    @property
    def traj_len(self) -> int:
        return self._T

    @property
    def spatial_size(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Trajectory Dataset (for full-rollout evaluation)
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Dataset yielding entire trajectories [T, n, n] for rollout evaluation.

    One item = one complete trajectory.  Used for test-time evaluation only.

    Args:
        h5_path: HDF5 file for the test (or val) split.
        norm_stats: If provided, normalize vorticity.
    """

    def __init__(
        self,
        h5_path: Path,
        norm_stats: Optional[NormStats] = None,
    ) -> None:
        self.h5_path = h5_path
        self.norm_stats = norm_stats

        with h5py.File(h5_path, "r") as f:
            self._n_traj: int = int(f["vorticity"].shape[0])
            self._T: int      = int(f["vorticity"].shape[1])
            self._n: int      = int(f["vorticity"].shape[2])

        self._file: Optional[h5py.File] = None

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self._n_traj

    def __getitem__(self, idx: int) -> Tensor:
        """Return full trajectory [T, n, n] float32."""
        f = self._open()
        traj = f["vorticity"][idx]  # [T, n, n]
        x = torch.from_numpy(traj.astype(np.float32))
        if self.norm_stats is not None:
            x = self.norm_stats.normalize(x)
        return x

    @property
    def n_traj(self) -> int:
        return self._n_traj

    @property
    def traj_len(self) -> int:
        return self._T

    @property
    def spatial_size(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# HDF5 metadata helpers
# ---------------------------------------------------------------------------

def load_metadata(h5_path: Path) -> dict:
    """Return a dict of scalar attributes stored in the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        return dict(f.attrs)


def load_seeds_phases(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (seeds, phases) arrays for all trajectories in the file."""
    with h5py.File(h5_path, "r") as f:
        return f["seeds"][:], f["phases"][:]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_norm_stats(
    train_h5: Path,
    cache_path: Optional[Path] = None,
    fast: bool = True,
) -> NormStats:
    """Compute (or load cached) normalization statistics from the training split.

    Args:
        train_h5: Path to train.h5.
        cache_path: If provided, load from this JSON if it exists, else compute and save.
        fast: Use fast (load-all) method; set False for large datasets.

    Returns:
        NormStats (from train split only).
    """
    if cache_path is not None and cache_path.exists():
        log.info("Loading cached NormStats from %s", cache_path)
        return NormStats.load(cache_path)

    stats = NormStats.from_file_fast(train_h5) if fast else NormStats.from_file(train_h5)

    if cache_path is not None:
        stats.save(cache_path)
        log.info("Saved NormStats to %s", cache_path)

    return stats


def build_dataloaders(
    root: Path,
    window: int,
    batch_size: int,
    num_workers: int = 0,
    stride: int = 1,
    normalize: bool = True,
    fast_norm: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, NormStats]:
    """Construct train, val, and test DataLoaders.

    Train/val use WindowDataset (sliding windows for TBPTT).
    Test uses TrajectoryDataset (full trajectories for evaluation).

    Normalization statistics are computed from the training split only.

    Args:
        root: Dataset root containing train.h5, val.h5, test.h5.
        window: Frames per window sample (warmup_len + rollout_len).
        batch_size: Training batch size.
        num_workers: DataLoader worker processes.
        stride: Window stride within a trajectory (train only).
        normalize: If True, apply NormStats normalization.
        fast_norm: Use fast (load-all) NormStats computation.

    Returns:
        (train_loader, val_loader, test_loader, norm_stats).
    """
    train_h5 = root / "train.h5"
    val_h5   = root / "val.h5"
    test_h5  = root / "test.h5"

    for p in (train_h5, val_h5, test_h5):
        if not p.exists():
            raise FileNotFoundError(
                f"HDF5 split file not found: {p}. "
                "Run `python -m msr_hine.data.generate` first."
            )

    # Norm stats from train only
    norm_cache = root / "norm_stats.json"
    stats = build_norm_stats(train_h5, cache_path=norm_cache, fast=fast_norm)
    ns = stats if normalize else None

    train_ds = WindowDataset(train_h5, window=window, stride=stride, norm_stats=ns)
    val_ds   = WindowDataset(val_h5,   window=window, stride=1,      norm_stats=ns)
    test_ds  = TrajectoryDataset(test_h5, norm_stats=ns)

    # num_workers=0 avoids HDF5 forking issues on some platforms
    _kw = dict(num_workers=num_workers, pin_memory=(num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **_kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, **_kw)

    log.info(
        "DataLoaders ready — train: %d windows / val: %d windows / test: %d trajectories",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_loader, val_loader, test_loader, stats
