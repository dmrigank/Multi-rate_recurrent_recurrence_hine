"""Plot 3 random 1D trajectories (KS and L96) as Hovmöller (space × time) diagrams.

Saves to outputs/paper_misc_plots/
"""
from __future__ import annotations
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT_DIR = Path("outputs/paper_misc_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS_PATH  = Path("/mnt/c/Users/mdhingra/Documents/code_implementations/Hierarchical_implicit/1d_chaotic_case/ks_bdf2_dataset_single.npz")
L96_PATH = Path("/mnt/c/Users/mdhingra/Documents/code_implementations/Hierarchical_implicit/1d_L96_case/l96_N40_F8_T1000_traj100.npz")

N_TRAJS   = 3
KS_T      = 400    # number of timesteps to show
L96_T     = 100
SEED      = 42


def hovmoller(ax, field, t_vals, x_vals, vmax=None, cmap="RdBu_r"):
    """Plot field [T, X] as imshow with time on x-axis, space on y-axis."""
    if vmax is None:
        vmax = np.percentile(np.abs(field), 99)
    im = ax.imshow(
        field.T,                       # [X, T] so x-axis = time, y-axis = space
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-vmax, vmax=vmax,
        extent=[t_vals[0], t_vals[-1], x_vals[0], x_vals[-1]],
        interpolation="bilinear",
    )
    return im


# ── KS ────────────────────────────────────────────────────────────────────────
print("Loading KS data...")
ks   = np.load(KS_PATH)
U_ks = ks["U"]          # [200, 1501, 128]
t_ks = ks["t"]          # [1501]  t=50..200
N_ks = U_ks.shape[-1]   # 128
L_ks = float(ks["L"])   # domain length
x_ks = np.linspace(0, L_ks, N_ks, endpoint=False)

# Use first KS_T timesteps
t_ks_plot = t_ks[:KS_T]

rng = random.Random(SEED)
ks_ids = rng.sample(range(U_ks.shape[0]), N_TRAJS)
print(f"  KS trajectories: {ks_ids}")

fig_ks, axes_ks = plt.subplots(1, N_TRAJS, figsize=(N_TRAJS * 3.6, 3.2),
                                gridspec_kw={"wspace": 0.08})

vmax_ks = np.percentile(np.abs(U_ks[ks_ids, :KS_T, :]), 99)
for col, tid in enumerate(ks_ids):
    ax  = axes_ks[col]
    field = U_ks[tid, :KS_T, :]    # [T, X]
    im = hovmoller(ax, field, t_ks_plot, x_ks, vmax=vmax_ks)
    ax.set_xlabel("Time $t$", fontsize=9)
    ax.set_title(f"Traj {tid}", fontsize=9)
    if col == 0:
        ax.set_ylabel("Space $x$", fontsize=9)
    else:
        ax.set_yticks([])

# Shared colorbar
cbar_ax = fig_ks.add_axes([0.92, 0.15, 0.015, 0.70])
cb = fig_ks.colorbar(im, cax=cbar_ax)
cb.set_label("$u(x,t)$", fontsize=9)
cb.ax.tick_params(labelsize=7)

fig_ks.suptitle("Kuramoto–Sivashinsky (KS) — sample trajectories",
                fontsize=10, y=1.02)

for ext in (".pdf", ".png"):
    out = OUT_DIR / ("ks_trajectories" + ext)
    fig_ks.savefig(str(out))
    print(f"  Saved {out}")
plt.close(fig_ks)


# ── L96 ───────────────────────────────────────────────────────────────────────
print("Loading L96 data...")
l96   = np.load(L96_PATH)
X_l96 = l96["X"]           # [100, 1000, 40]
t_l96 = l96["t"]           # [1000]  t=0..50
N_l96 = X_l96.shape[-1]    # 40
x_l96 = np.arange(N_l96)   # node indices 0..39

t_l96_plot = t_l96[:L96_T]

l96_ids = rng.sample(range(X_l96.shape[0]), N_TRAJS)
print(f"  L96 trajectories: {l96_ids}")

fig_l96, axes_l96 = plt.subplots(1, N_TRAJS, figsize=(N_TRAJS * 3.6, 3.2),
                                  gridspec_kw={"wspace": 0.08})

vmax_l96 = np.percentile(np.abs(X_l96[l96_ids, :L96_T, :]), 99)
for col, tid in enumerate(l96_ids):
    ax    = axes_l96[col]
    field = X_l96[tid, :L96_T, :]   # [T, N]
    im = hovmoller(ax, field, t_l96_plot, x_l96, vmax=vmax_l96)
    ax.set_xlabel("Time $t$", fontsize=9)
    ax.set_title(f"Traj {tid}", fontsize=9)
    if col == 0:
        ax.set_ylabel("Node index $i$", fontsize=9)
    else:
        ax.set_yticks([])

cbar_ax = fig_l96.add_axes([0.92, 0.15, 0.015, 0.70])
cb = fig_l96.colorbar(im, cax=cbar_ax)
cb.set_label("$X_i(t)$", fontsize=9)
cb.ax.tick_params(labelsize=7)

fig_l96.suptitle("Lorenz-96 (L96, $N=40$, $F=8$) — sample trajectories",
                 fontsize=10, y=1.02)

for ext in (".pdf", ".png"):
    out = OUT_DIR / ("l96_trajectories" + ext)
    fig_l96.savefig(str(out))
    print(f"  Saved {out}")
plt.close(fig_l96)

print(f"\nAll saved to {OUT_DIR}/")
