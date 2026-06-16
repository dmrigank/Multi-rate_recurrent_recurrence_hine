"""Plot 3 random ground-truth test trajectories at 5 equally-spaced timesteps.

Output: outputs/paper_misc_plots/gt_trajectories.pdf / .png
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   9,
    "axes.titlesize": 9,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

N_TRAJS  = 3
N_STEPS  = 5
TAU_LAM  = 83.856   # snapshot steps
SEED     = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/kolmogorov"))
    parser.add_argument("--out-dir",   type=Path, default=Path("outputs/paper_misc_plots"))
    parser.add_argument("--n-trajs",   type=int,  default=N_TRAJS)
    parser.add_argument("--n-steps",   type=int,  default=N_STEPS)
    parser.add_argument("--seed",      type=int,  default=SEED)
    parser.add_argument("--t-max",     type=int,  default=None,
                        help="Last timestep index to include (default: last frame)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data_root / "test.h5", "r") as f:
        trajs = f["vorticity"][:]          # [N, T, H, W]

    N_total, T, H, W = trajs.shape
    t_max = args.t_max if args.t_max is not None else T - 1

    # Pick equally-spaced timesteps from 0 to t_max
    t_indices = [int(round(i * t_max / (args.n_steps - 1))) for i in range(args.n_steps)]
    t_labels  = [f"$t={t}$\n({t/TAU_LAM:.2f}$\\tau_\\lambda$)" for t in t_indices]

    # Random trajectory selection
    rng = random.Random(args.seed)
    traj_ids = rng.sample(range(N_total), args.n_trajs)
    print(f"Selected trajectories: {traj_ids}")
    print(f"Timesteps: {t_indices}")

    # Shared colour scale: use global vorticity range across selected frames
    selected = trajs[np.array(traj_ids)][:, t_indices]   # [n_trajs, n_steps, H, W]
    vmax = np.percentile(np.abs(selected), 99)
    vmin = -vmax

    # Layout: n_trajs rows × n_steps cols
    fig, axes = plt.subplots(
        args.n_trajs, args.n_steps,
        figsize=(args.n_steps * 2.2, args.n_trajs * 2.1),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    for row, traj_id in enumerate(traj_ids):
        for col, t in enumerate(t_indices):
            ax = axes[row, col]
            field = trajs[traj_id, t]
            im = ax.imshow(
                field, origin="lower", cmap="RdBu_r",
                vmin=vmin, vmax=vmax, interpolation="bilinear",
            )
            ax.set_xticks([]); ax.set_yticks([])

            # Column headers (top row only)
            if row == 0:
                ax.set_title(t_labels[col], fontsize=8, pad=4)

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(
                    f"Traj {traj_id}",
                    fontsize=8, rotation=0,
                    ha="right", va="center", labelpad=32,
                )

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Vorticity $\\omega$", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Ground-truth vorticity — 3 test trajectories",
        fontsize=10, y=1.01,
    )

    for ext in (".pdf", ".png"):
        out = args.out_dir / ("gt_trajectories" + ext)
        fig.savefig(str(out))
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
