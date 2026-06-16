"""Visualise the MSR-HINE spectral hierarchy on a real vorticity snapshot.

Produces a figure with 4 columns:
  Col 0 — Full field (fine, |k| ≤ 85, 256×256)
  Col 1 — Medium band (|k| ≤ 16), shown at full resolution (project only)
  Col 2 — Coarse band (|k| ≤ 8),  shown at full resolution (project only)
  Col 3 — High-k residual (fine − medium), the small-scale content

Row 0: vorticity field (spatial)
Row 1: log10 radial power spectrum E(k) with shaded retained bands

Saved to outputs/paper_misc_plots/multiscale_hierarchy.pdf/.png
"""
from __future__ import annotations
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msr_hine.spectral.truncation import project
from msr_hine.metrics import radial_energy_spectrum

OUT_DIR  = Path("outputs/paper_misc_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_H5  = Path("data/kolmogorov/test.h5")
TRAJ_IDX = 0
FRAME_IDX = 100          # pick a frame well into the attractor

K_FINE   = 85            # full resolved band (2/3 dealiasing of 256-grid)
K_MEDIUM = 16
K_COARSE = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load frame ────────────────────────────────────────────────────────────────
print(f"Loading frame [{TRAJ_IDX}, {FRAME_IDX}] from {DATA_H5}...")
with h5py.File(DATA_H5, "r") as f:
    omega_np = f["vorticity"][TRAJ_IDX, FRAME_IDX]   # [256, 256]

omega = torch.from_numpy(omega_np).float().unsqueeze(0)   # [1, 256, 256]

# ── Spectral projections (all at full 256×256 resolution for visual comparison)
omega_fine   = omega                                    # full field
omega_medium = project(omega, K_MEDIUM)                # |k| ≤ 16
omega_coarse = project(omega, K_COARSE)                # |k| ≤ 8
omega_highk  = omega_fine - omega_medium               # high-k residual

fields = [
    (omega_fine  [0].numpy(), f"Fine  ($|k| \\leq {K_FINE}$)\n256×256",   "fine"),
    (omega_medium[0].numpy(), f"Medium  ($|k| \\leq {K_MEDIUM}$)\nstride $s_1=2$", "medium"),
    (omega_coarse[0].numpy(), f"Coarse  ($|k| \\leq {K_COARSE}$)\nstride $s_2=4$",  "coarse"),
    (omega_highk [0].numpy(), f"High-$k$ residual\n(fine − medium)",       "residual"),
]

BAND_COLORS = {
    "fine":     "#1f77b4",
    "medium":   "#ff7f0e",
    "coarse":   "#2ca02c",
    "residual": "#9467bd",
}

# ── Radial spectra ────────────────────────────────────────────────────────────
# radial_energy_spectrum takes [..., H, W] → (k_bins [K], E_k [..., K])
kb, Ek_fine   = radial_energy_spectrum(omega_fine)    # [1,H,W] → E [1,K]
_,  Ek_medium = radial_energy_spectrum(omega_medium)
_,  Ek_coarse = radial_energy_spectrum(omega_coarse)
_,  Ek_highk  = radial_energy_spectrum(omega_highk)

kb_np = kb.numpy()
spectra = {
    "fine":     Ek_fine  [0].numpy(),   # [K]
    "medium":   Ek_medium[0].numpy(),
    "coarse":   Ek_coarse[0].numpy(),
    "residual": Ek_highk [0].numpy(),
}

# ── Layout: 2 rows × 4 cols  ─────────────────────────────────────────────────
n_cols = 4
fig = plt.figure(figsize=(n_cols * 3.4, 6.8))

gs = matplotlib.gridspec.GridSpec(
    2, n_cols + 1,
    figure=fig,
    height_ratios=[1.0, 0.85],
    hspace=0.32, wspace=0.06,
    left=0.06, right=0.93,
    width_ratios=[1, 1, 1, 1, 0.04],
)

# Shared vorticity colour scale anchored to fine field
vmax = float(np.percentile(np.abs(omega_np), 99))

axes_field  = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
ax_spec     = fig.add_subplot(gs[1, :n_cols])   # spectrum spans field columns only
ax_cbar     = fig.add_subplot(gs[0, n_cols])

# ── Row 0: vorticity fields ───────────────────────────────────────────────────
for col, (field, title, key) in enumerate(fields):
    ax = axes_field[col]
    # residual has smaller amplitude — use its own scale for clarity
    if key == "residual":
        vm = float(np.percentile(np.abs(field), 99))
    else:
        vm = vmax
    im = ax.imshow(field, origin="lower", cmap="RdBu_r",
                   vmin=-vm, vmax=vm, interpolation="bilinear")
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

    # Colour border matching band colour
    for spine in ax.spines.values():
        spine.set_edgecolor(BAND_COLORS[key])
        spine.set_linewidth(2.0)

# Shared colorbar for fine/medium/coarse (not residual)
plt.colorbar(im, cax=ax_cbar, label="$\\omega$")
ax_cbar.tick_params(labelsize=7)

# ── Row 1: radial energy spectra ──────────────────────────────────────────────
k_cut  = 256 // 3   # 2/3 dealiasing cutoff ≈ 85
k_show = k_cut + 5  # show a little beyond to show the cutoff wall

ax = ax_spec

# Shaded band regions
ax.axvspan(1,          K_COARSE,  alpha=0.08, color=BAND_COLORS["coarse"],  zorder=0)
ax.axvspan(K_COARSE+1, K_MEDIUM,  alpha=0.08, color=BAND_COLORS["medium"],  zorder=0)
ax.axvspan(K_MEDIUM+1, k_cut,     alpha=0.06, color=BAND_COLORS["residual"], zorder=0)

# Vertical cutoff lines
for k_cut_line, color, ls in [
    (K_COARSE, BAND_COLORS["coarse"],   "--"),
    (K_MEDIUM, BAND_COLORS["medium"],   "--"),
    (k_cut,    "gray",                  ":"),
]:
    ax.axvline(k_cut_line, color=color, ls=ls, lw=1.2, alpha=0.8, zorder=1)

# Plot spectra
labels = {
    "fine":     f"Fine ($|k|\\leq{K_FINE}$)",
    "medium":   f"Medium ($|k|\\leq{K_MEDIUM}$)",
    "coarse":   f"Coarse ($|k|\\leq{K_COARSE}$)",
    "residual": "High-$k$ residual",
}
lws   = {"fine": 2.2, "medium": 1.8, "coarse": 1.8, "residual": 1.4}
lss   = {"fine": "-",  "medium": "-",  "coarse": "-",  "residual": "--"}

for key, Ek in spectra.items():
    Ek_clip = np.clip(Ek[1:k_show], 1e-12, None)
    ax.loglog(kb_np[1:k_show], Ek_clip,
              color=BAND_COLORS[key], lw=lws[key], ls=lss[key],
              label=labels[key], zorder=3)

# k^{-3} reference slope
k_ref = kb_np[5:k_show]
E_ref = spectra["fine"][5]
ax.loglog(k_ref, E_ref * (k_ref / k_ref[0])**(-3),
          "k:", lw=0.9, alpha=0.5, label="$k^{-3}$", zorder=2)

# Cutoff annotations — use axes-fraction y coords so they stay visible regardless of ylim
trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
for k_ann, label_ann, color_ann in [
    (K_COARSE, f"$k_c^{{\\rm crs}}={K_COARSE}$", BAND_COLORS["coarse"]),
    (K_MEDIUM, f"$k_c^{{\\rm med}}={K_MEDIUM}$", BAND_COLORS["medium"]),
]:
    ax.text(k_ann * 1.05, 0.04, label_ann,
            color=color_ann, fontsize=7, va="bottom", transform=trans)

ax.set_xlabel("Wavenumber $|k|$", fontsize=9)
ax.set_ylabel("$E(k)$", fontsize=9)
ax.set_title("Radial energy spectrum — spectral band decomposition", fontsize=9)
ax.set_xlim([1, k_show])
ax.legend(fontsize=7.5, loc="lower left", ncol=2)
ax.grid(True, which="both", alpha=0.2)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "MSR-HINE spectral hierarchy  —  2D Kolmogorov flow ($256\\times256$, Re=4000)",
    fontsize=10, y=1.01,
)

for ext in (".pdf", ".png"):
    out = OUT_DIR / ("multiscale_hierarchy" + ext)
    fig.savefig(str(out))
    print(f"Saved {out}")
plt.close(fig)
