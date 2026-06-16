"""Sanity plots for a generated Kolmogorov flow dataset.

Produces two figures per split (train/val/test):
  1. Vorticity fields at 5 equally-spaced snapshots from trajectory 0.
  2. Radial energy spectra at those same snapshots.

Usage:
    python scripts/plot_dataset_sanity.py [--root data/kolmogorov] [--split train]
    python scripts/plot_dataset_sanity.py --root data/kolmogorov/debug  # debug dataset
    python scripts/plot_dataset_sanity.py --root data/kolmogorov         # full dataset

Output is written to <root>/plots/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sure the package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msr_hine.data.solver import radial_energy_spectrum, wavenumbers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_traj(h5_path: Path, traj_idx: int = 0) -> tuple[np.ndarray, dict]:
    with h5py.File(h5_path, "r") as f:
        traj = f["vorticity"][traj_idx][:]   # [T, n, n] float32
        meta = dict(f.attrs)
        seeds = f["seeds"][:]
        phases = f["phases"][:]
    meta["seed"] = int(seeds[traj_idx])
    meta["phase"] = float(phases[traj_idx])
    return traj, meta


def pick_time_indices(T: int, n_frames: int = 5) -> np.ndarray:
    return np.linspace(0, T - 1, n_frames, dtype=int)


def plot_vorticity(
    traj: np.ndarray,
    meta: dict,
    t_idx: np.ndarray,
    split: str,
    traj_idx: int,
    out_path: Path,
) -> None:
    T, n, _ = traj.shape
    times = t_idx * float(meta["dt_snapshot"])

    fig, axes = plt.subplots(1, len(t_idx), figsize=(4 * len(t_idx), 3.8))
    if len(t_idx) == 1:
        axes = [axes]

    for j, (ti, t) in enumerate(zip(t_idx, times)):
        frame = traj[ti]
        vmax = max(float(np.abs(frame).max()), 1e-6)
        im = axes[j].imshow(frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[j].set_title(f"t = {t:.3f}\n(step {ti})", fontsize=10)
        axes[j].axis("off")
        plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Vorticity ω — {split} split  traj {traj_idx}  "
        f"(Re={meta['re']:.0f}, n={n}, seed={meta['seed']}, φ={meta['phase']:.3f})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_spectra(
    traj: np.ndarray,
    meta: dict,
    t_idx: np.ndarray,
    split: str,
    traj_idx: int,
    out_path: Path,
) -> None:
    T, n, _ = traj.shape
    times = t_idx * float(meta["dt_snapshot"])
    device = torch.device("cpu")
    _, _, ksq = wavenumbers(n, device)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_idx)))

    for j, (ti, t) in enumerate(zip(t_idx, times)):
        ft = torch.from_numpy(traj[ti].astype(np.float64))
        ohat = torch.fft.rfft2(ft).unsqueeze(0)          # [1, n, n//2+1]
        kb, Ek = radial_energy_spectrum(ohat, ksq, n)    # [n//2], [1, n//2]
        k = kb.numpy()
        E = Ek[0].numpy()
        ax.semilogy(k[1:], E[1:], color=colors[j], label=f"t = {t:.3f}", linewidth=1.8)

    # Reference slopes
    k_ref = np.arange(2, n // 3)
    ax.semilogy(k_ref, 3e-2 * k_ref ** (-3.0), "k--", lw=0.8, alpha=0.5, label="k⁻³")
    ax.semilogy(k_ref, 1e-2 * k_ref ** (-5.0 / 3), "k:",  lw=0.8, alpha=0.5, label="k⁻⁵/³")

    k_f = int(meta["k_f"])
    k_cut = n // 3
    ax.axvline(k_f,   color="steelblue", ls="--", lw=1.2, label=f"k_f = {k_f}")
    ax.axvline(k_cut, color="gray",      ls=":",  lw=1.2, label=f"2/3 cut (k={k_cut})")

    ax.set_xlabel("Wavenumber k", fontsize=12)
    ax.set_ylabel("E(k)", fontsize=12)
    ax.set_xlim([1, n // 2])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        f"Radial energy spectrum — {split} traj {traj_idx}  "
        f"(Re={meta['re']:.0f}, n={n})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_diagnostics(
    h5_path: Path,
    split: str,
    traj_idx: int,
    out_dir: Path,
    n_frames: int = 5,
) -> None:
    """Generate vorticity + spectrum plots for one trajectory in one split."""
    print(f"\n[{split}] traj {traj_idx}  from {h5_path.name}")
    traj, meta = load_traj(h5_path, traj_idx=traj_idx)
    T, n, _ = traj.shape
    print(f"  shape={traj.shape}  Re={meta['re']:.0f}  τ={meta['tau_estimate']:.4f}"
          f"  dt_snap={meta['dt_snapshot']:.5f}")

    t_idx = pick_time_indices(T, n_frames)

    # Vorticity RMS / max summary
    for ti in t_idx:
        fr = traj[ti]
        t = ti * meta["dt_snapshot"]
        print(f"  t={t:.3f}: rms={np.sqrt((fr**2).mean()):.4f}  max={np.abs(fr).max():.4f}")

    stem = f"{split}_traj{traj_idx}"
    plot_vorticity(traj, meta, t_idx, split, traj_idx,
                   out_dir / f"{stem}_vorticity.png")
    plot_spectra(traj, meta, t_idx, split, traj_idx,
                 out_dir / f"{stem}_spectrum.png")


# ---------------------------------------------------------------------------
# Summary figure: one trajectory from each split
# ---------------------------------------------------------------------------

def plot_split_summary(root: Path, out_dir: Path, n_frames: int = 5) -> None:
    """Grid figure: one row per split, columns = time snapshots."""
    splits = ["train", "val", "test"]
    h5s = {s: root / f"{s}.h5" for s in splits}
    missing = [s for s, p in h5s.items() if not p.exists()]
    if missing:
        print(f"Skipping summary: missing splits {missing}")
        return

    fig, axes = plt.subplots(
        len(splits), n_frames,
        figsize=(3.5 * n_frames, 3.5 * len(splits)),
    )

    for row, split in enumerate(splits):
        traj, meta = load_traj(h5s[split], traj_idx=0)
        T, n, _ = traj.shape
        t_idx = pick_time_indices(T, n_frames)
        times = t_idx * float(meta["dt_snapshot"])

        for col, (ti, t) in enumerate(zip(t_idx, times)):
            ax = axes[row, col]
            frame = traj[ti]
            vmax = max(float(np.abs(frame).max()), 1e-6)
            ax.imshow(frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            if row == 0:
                ax.set_title(f"t = {t:.3f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{split}\n(Re={meta['re']:.0f})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Vorticity overview — all splits (n={n}, traj 0 each)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = out_dir / "summary_all_splits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary: {out.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="data/kolmogorov",
        help="Dataset root directory (contains train.h5, val.h5, test.h5)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Which splits to plot",
    )
    parser.add_argument(
        "--traj", type=int, default=0,
        help="Trajectory index to plot within each split",
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="Number of equally-spaced time frames to show",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    for split in args.splits:
        h5_path = root / f"{split}.h5"
        if not h5_path.exists():
            print(f"  [SKIP] {h5_path} not found")
            continue
        plot_diagnostics(
            h5_path=h5_path,
            split=split,
            traj_idx=args.traj,
            out_dir=out_dir,
            n_frames=args.frames,
        )

    plot_split_summary(root, out_dir, n_frames=args.frames)
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
