"""Journal-quality results plotting for MSR-HINE-2D.

Produces four figures:
  Fig 1 — RMSE vs. lead time (timesteps) with ±1 std band
  Fig 2 — ACC  vs. lead time (timesteps) with ±1 std band
  Fig 3 — Vorticity rollout comparison (GT / model / |error|) for best and worst
           MSR-HINE trajectory (6 rows × 5 columns each)
  Fig 4 — Radial energy spectrum E(k) at 5 lead times: all models + GT

Usage:
    python scripts/plot_results.py [--n-rollout 200] [--warmup 12]
                                   [--out-dir outputs/paper_figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msr_hine.rollout import rollout
from msr_hine.metrics import (
    rmse as _rmse, anomaly_correlation, climatology, radial_energy_spectrum,
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_COLORS = {
    "fno_1step":  "#9467bd",   # purple
    "unet_1step": "#2ca02c",   # green
    "hine":       "#ff7f0e",   # orange
    "msr_hine":   "#1f77b4",   # blue
}
MODEL_LABELS = {
    "fno_1step":  "FNO-AR",
    "unet_1step": "UNet-AR",
    "hine":       "HINE-L2",
    "msr_hine":   "MSR-HINE",
}
MODEL_ORDER = ["fno_1step", "unet_1step", "hine", "msr_hine"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    from omegaconf import OmegaConf
    from msr_hine.train import build_model
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = OmegaConf.create(ckpt["cfg"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def run_rollout(model, traj_tensor: torch.Tensor, warmup_len: int,
                n_steps: int, device: torch.device) -> torch.Tensor:
    """Return [n_steps, 1, H, W] predictions on CPU."""
    traj = traj_tensor.to(device).unsqueeze(0)       # [1, T, 1, H, W]
    warmup_frames = traj[:, :warmup_len]              # [1, W, 1, H, W]
    omega_seed    = traj[:,  warmup_len]              # [1, 1, H, W]
    with torch.no_grad():
        preds = rollout(model, omega_seed, n_steps,
                        warmup_frames=warmup_frames)  # [1, n_steps, 1, H, W]
    return preds[0].cpu()   # [n_steps, 1, H, W]


def compute_metrics_per_traj(
    models: dict,
    trajs:  np.ndarray,
    warmup_len: int,
    n_steps: int,
    device: torch.device,
) -> dict[str, dict]:
    """Return per-trajectory RMSE and ACC arrays for each model.

    Returns:
        {model_name: {'rmse': [N_traj, n_steps], 'acc': [N_traj, n_steps]}}
    """
    N, T, H, W = trajs.shape
    clim = trajs.mean(axis=(0, 1), keepdims=True)     # spatial climatology [1,1,H,W]
    clim_t = torch.from_numpy(clim.astype("float32")).unsqueeze(0)  # [1,1,1,H,W]

    results = {name: {"rmse": [], "acc": [], "preds": []} for name in models}

    for ti in range(N):
        traj  = torch.from_numpy(trajs[ti].astype("float32")).unsqueeze(1)
        avail = T - warmup_len - 1
        k     = min(n_steps, avail)
        gt    = traj[warmup_len + 1 : warmup_len + 1 + k]
        gt_b  = gt.unsqueeze(0)

        for name, model in models.items():
            preds = run_rollout(model, traj, warmup_len, k, device)
            preds_b = preds.unsqueeze(0)                 # [1, k, 1, H, W]

            rmse_t = _rmse(preds_b, gt_b)               # [k]
            acc_t  = anomaly_correlation(preds_b, gt_b, clim_t)  # [k]

            results[name]["rmse"].append(rmse_t.numpy())
            results[name]["acc" ].append(acc_t .numpy())
            results[name]["preds"].append(preds.numpy())  # [k, 1, H, W]

        if (ti + 1) % 2 == 0:
            print(f"  [{ti+1}/{N}] done", flush=True)

    # Stack: [N, k]
    for name in models:
        results[name]["rmse"]  = np.stack(results[name]["rmse"],  axis=0)
        results[name]["acc"]   = np.stack(results[name]["acc"],   axis=0)
        results[name]["preds"] = np.stack(results[name]["preds"], axis=0)  # [N,k,1,H,W]
    return results


# ── Figure 1 & 2: RMSE and ACC with std bands ─────────────────────────────────

def plot_rmse_acc(results: dict, tau_lam: float, dt_snap: float, out_dir: Path):
    ref_name = next(n for n in MODEL_ORDER if n in results)
    n_plot   = results[ref_name]["rmse"].shape[1]
    steps    = np.arange(n_plot)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, (metric, ylabel, logy) in enumerate([
        ("rmse", "RMSE",  True),
        ("acc",  "ACC",   False),
    ]):
        ax = axes[ax_idx]

        # Collect all means to set a sensible y-range before drawing bands
        all_means = {name: results[name][metric].mean(axis=0)
                     for name in MODEL_ORDER if name in results}

        # 1-step models diverge so fast their std bands are uninformative;
        # only draw bands for the recurrent models
        BAND_MODELS = {"hine", "msr_hine"}
        for name in MODEL_ORDER:
            if name not in results:
                continue
            arr  = results[name][metric]           # [N, k]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            c    = MODEL_COLORS[name]
            lbl  = MODEL_LABELS[name]
            ax.plot(steps, mean, color=c, label=lbl, zorder=3)
            if name in BAND_MODELS:
                lo = np.maximum(mean - std, 1e-6 if logy else 0)
                ax.fill_between(steps, lo, mean + std, color=c, alpha=0.15, zorder=2)

        # τ_λ marker
        ax.axvline(tau_lam, color="k", ls="--", lw=0.9, alpha=0.7,
                   label=r"$\tau_\lambda$")

        if metric == "acc":
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)

        if logy:
            ax.set_yscale("log")
            hine_mean = all_means.get("hine", list(all_means.values())[0])
            msr_mean  = all_means.get("msr_hine", hine_mean)
            ymin = min(hine_mean.min(), msr_mean.min()) * 0.5
            ymax = max(all_means[n].max() for n in ["fno_1step", "unet_1step"]
                       if n in all_means) * 2.0
            ax.set_ylim([ymin, ymax])

        ax.set_xlabel("Lead time (snapshot steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+ax_idx)}) {ylabel} vs. lead time")
        ax.legend(loc="upper right" if metric == "rmse" else "lower left")
        ax.grid(True, which="both" if logy else "major", alpha=0.25)

    plt.tight_layout()
    out = out_dir / "fig1_rmse_acc.pdf"
    fig.savefig(str(out)); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 3: Vorticity rollout panels ───────────────────────────────────────

def plot_vorticity_rollout(
    results:    dict,
    trajs:      np.ndarray,
    warmup_len: int,
    n_steps:    int,
    tau_lam:    float,
    out_dir:    Path,
    ablation_results: dict | None = None,
):
    msr_rmse  = results["msr_hine"]["rmse"]
    best_idx  = int(np.argmin(msr_rmse.mean(axis=1)))
    worst_idx = int(np.argmax(msr_rmse.mean(axis=1)))
    print(f"  Best  MSR-HINE traj: {best_idx}  (mean RMSE {msr_rmse.mean(axis=1)[best_idx]:.4f})")
    print(f"  Worst MSR-HINE traj: {worst_idx} (mean RMSE {msr_rmse.mean(axis=1)[worst_idx]:.4f})")

    # (a) Full comparison — all core models
    for traj_label, traj_idx in [("best", best_idx), ("worst", worst_idx)]:
        _plot_one_rollout(
            results=results, trajs=trajs, traj_idx=traj_idx,
            traj_label=traj_label, warmup_len=warmup_len, n_steps=n_steps,
            tau_lam=tau_lam, out_dir=out_dir,
            model_names=MODEL_ORDER,
            filename=f"fig3_vorticity_{traj_label}",
        )

    # (b) Clean HINE vs MSR-HINE comparison only
    core_two = [n for n in ["hine", "msr_hine"] if n in results]
    if len(core_two) == 2:
        for traj_label, traj_idx in [("best", best_idx), ("worst", worst_idx)]:
            _plot_one_rollout(
                results=results, trajs=trajs, traj_idx=traj_idx,
                traj_label=traj_label, warmup_len=warmup_len, n_steps=n_steps,
                tau_lam=tau_lam, out_dir=out_dir,
                model_names=core_two,
                filename=f"fig3_hine_vs_msr_{traj_label}",
                anchor_errors_to_hine=True,
            )
        print("  Saved fig3_hine_vs_msr_best/worst")

    # (c) Ablation comparison: GT + MSR-HINE + 3 ablations
    if ablation_results is not None:
        abl_names   = list(ablation_results.keys())
        abl_labels_map = {
            "no_multirate": "No multirate",
            "no_topdown":   "No top-down",
            "single_scale": "Single scale",
        }
        # Merge ablation preds into a combined results dict for _plot_one_rollout
        combined = {"msr_hine": results["msr_hine"]}
        for abl in abl_names:
            combined[abl] = ablation_results[abl]
        # Temporarily register ablation labels
        orig_labels = MODEL_LABELS.copy()
        for abl in abl_names:
            MODEL_LABELS[abl] = abl_labels_map.get(abl, abl)
            MODEL_COLORS[abl] = {
                "no_multirate": "#d62728",
                "no_topdown":   "#e377c2",
                "single_scale": "#8c564b",
            }.get(abl, "#888888")

        _plot_one_rollout(
            results=combined, trajs=trajs, traj_idx=best_idx,
            traj_label="best", warmup_len=warmup_len, n_steps=n_steps,
            tau_lam=tau_lam, out_dir=out_dir,
            model_names=["msr_hine"] + abl_names,
            filename="fig3_ablation_comparison",
            title_suffix=" (ablation study)",
            anchor_errors_to_hine=False,
        )
        print("  Saved fig3_ablation_comparison")

        # Restore
        for k in list(MODEL_LABELS.keys()):
            if k in orig_labels:
                MODEL_LABELS[k] = orig_labels[k]
            elif k not in ["fno_1step", "unet_1step", "hine", "msr_hine"]:
                del MODEL_LABELS[k]


def _make_rows(results, model_names, traj_idx, avail, t_idx, gt_show, vmax_omega,
               hine_errors=None):
    """Build row-data list and compute shared per-column error scale.

    If hine_errors is provided (list of per-column error arrays from the HINE
    model), the shared error scale is anchored to HINE's errors so that all
    models are judged on the same scale regardless of how bad they get.
    """
    rows = []
    pred_data = {}
    for name in model_names:
        preds_traj = results[name]["preds"][traj_idx, :avail, 0]
        err = np.abs(preds_traj[t_idx] - gt_show)
        pred_data[name] = (preds_traj, err)

    # Anchor error scale to HINE if provided, else max across all shown models
    if hine_errors is not None:
        shared_vmax_err = np.array([e.max() for e in hine_errors])
    else:
        all_errors = [pred_data[n][1] for n in model_names]
        shared_vmax_err = np.array([
            max(all_errors[mi][ci].max() for mi in range(len(model_names)))
            for ci in range(len(t_idx))
        ])

    rows.append(("Ground\nTruth", None, gt_show, False, np.full(len(t_idx), vmax_omega)))
    for name in model_names:
        preds_traj, err = pred_data[name]
        pred_show = np.clip(preds_traj[t_idx], -3 * vmax_omega, 3 * vmax_omega)
        rows.append((MODEL_LABELS.get(name, name), name, pred_show, False,
                     np.full(len(t_idx), vmax_omega)))
        rows.append((f"|Error|\n{MODEL_LABELS.get(name, name)}", name, err, True,
                     shared_vmax_err))
    return rows


def _plot_one_rollout(results, trajs, traj_idx, traj_label,
                      warmup_len, n_steps, tau_lam, out_dir,
                      model_names=None, filename="fig3_vorticity", title_suffix="",
                      anchor_errors_to_hine=True, t_step_override=None):
    """Plot panel: GT row then (prediction / |error|) rows for each model.

    Error colourscale is anchored to HINE's errors by default so all models
    are judged on the same scale. Pass anchor_errors_to_hine=False to use the
    max error across all shown models instead.
    t_step_override: optional list of integer step indices to use as columns.
    """
    if model_names is None:
        model_names = MODEL_ORDER

    avail    = min(n_steps, trajs.shape[1] - warmup_len - 1)
    tau2_idx = min(int(round(2 * tau_lam)) - 1, avail - 1)  # 2 τ_λ cap
    if t_step_override is not None:
        t_idx = np.array([min(t, avail - 1) for t in t_step_override], dtype=int)
    else:
        t_idx = np.linspace(0, tau2_idx, 5, dtype=int)

    gt_frames  = trajs[traj_idx][warmup_len + 1 : warmup_len + 1 + avail]
    gt_show    = gt_frames[t_idx]
    vmax_omega = float(np.percentile(np.abs(gt_show), 99))

    # Pre-compute HINE errors for scale anchoring
    hine_errors = None
    if anchor_errors_to_hine and "hine" in results:
        hine_preds = results["hine"]["preds"][traj_idx, :avail, 0]
        hine_errors = [np.abs(hine_preds[t] - gt_show[ci])
                       for ci, t in enumerate(t_idx)]

    rows_data = _make_rows(results, model_names, traj_idx, avail,
                           t_idx, gt_show, vmax_omega, hine_errors=hine_errors)

    n_rows = len(rows_data)
    n_cols = len(t_idx)
    col_w, row_h = 2.3, 2.1

    fig = plt.figure(figsize=(n_cols * col_w + 2.0, n_rows * row_h + 0.6))
    gs  = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig,
                            hspace=0.06, wspace=0.04,
                            left=0.09, right=0.97,
                            width_ratios=[1] * n_cols + [0.06])

    for row_i, (row_label, model_name, frames, is_error, vmax_per_col) in enumerate(rows_data):
        cmap  = "hot_r" if is_error else "RdBu_r"
        color = MODEL_COLORS.get(model_name, "k") if model_name else "k"
        lw    = 1.4 if not is_error else 0.7

        axes_row = []
        for col_i, (frame, t) in enumerate(zip(frames, t_idx)):
            ax  = fig.add_subplot(gs[row_i, col_i])
            vmx = float(vmax_per_col[col_i])
            vmn = 0.0 if is_error else -vmx
            im  = ax.imshow(frame, cmap=cmap, vmin=vmn, vmax=vmx,
                            origin="lower", interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            axes_row.append((ax, im))

            if row_i == 0:
                t_tau_str = f"{t / tau_lam:.2f}" + r"$\tau_\lambda$"
                ax.set_title(f"$t={t}$\n({t_tau_str})", fontsize=8, pad=3)
            if col_i == 0:
                # Reduced labelpad so labels sit closer to the panels
                ax.set_ylabel(row_label, fontsize=8.5, color=color,
                              rotation=0, ha="right", va="center", labelpad=28,
                              multialignment="center")
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(lw)

        # Single colorbar per row
        cbar_ax = fig.add_subplot(gs[row_i, n_cols])
        cb = fig.colorbar(axes_row[-1][1], cax=cbar_ax)
        cbar_ax.tick_params(labelsize=7)
        if row_i == 0:
            cb.set_label("$\\omega$", fontsize=8)
        elif is_error:
            cb.set_label(r"$|\Delta\omega|$", fontsize=8)

    model_str = " / ".join(MODEL_LABELS.get(n, n) for n in model_names)
    err_note  = ("HINE-anchored" if (anchor_errors_to_hine and "hine" in results)
                 else "shared") + " error scale"
    fig.suptitle(
        f"Rollout: GT vs. {model_str}{title_suffix}  |  "
        f"traj {traj_idx} ({traj_label})\n"
        f"$\\tau_\\lambda \\approx {tau_lam:.0f}$ steps.  {err_note}.",
        fontsize=9, y=1.005,
    )
    out = out_dir / f"{filename}.pdf"
    fig.savefig(str(out)); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Ablation vorticity panels ─────────────────────────────────────────────────

ABLATION_NAMES = [
    "single_scale", "no_multirate", "no_topdown",
    "no_consistency", "no_contraction", "no_warmup",
    "circularity_confirm",
]
ABLATION_LABELS = {
    "single_scale":       "Single scale",
    "no_multirate":       "No multirate",
    "no_topdown":         "No top-down",
    "no_consistency":     "No consistency",
    "no_contraction":     "No contraction",
    "no_warmup":          "No warmup",
    "circularity_confirm":"Circ. confirm",
}

def plot_ablation_rollouts(
    results:    dict,
    trajs:      np.ndarray,
    warmup_len: int,
    n_steps:    int,
    tau_lam:    float,
    out_dir:    Path,
):
    """One plot per ablation: GT / HINE / MSR-HINE (ablation) with shared error scale."""
    # Use same best trajectory as the main comparison
    msr_rmse = results["msr_hine"]["rmse"]
    best_idx = int(np.argmin(msr_rmse.mean(axis=1)))

    ablation_dir = out_dir / "ablations"
    ablation_dir.mkdir(exist_ok=True)

    for abl_name in ABLATION_NAMES:
        if abl_name not in results:
            print(f"  Skipping {abl_name} (not in results)")
            continue

        # Temporarily add the ablation predictions under its own label
        # so _make_rows can access them via results dict
        abl_results = {
            "hine":    results["hine"],
            "msr_hine": results["msr_hine"],   # kept for scale reference
            abl_name:  results[abl_name],
        }
        # MODEL_LABELS lookup will use ABLATION_LABELS for the ablation name
        orig_labels = MODEL_LABELS.copy()
        MODEL_LABELS[abl_name] = ABLATION_LABELS.get(abl_name, abl_name)

        # Rows: GT, HINE, ablation
        model_names = ["hine", abl_name]
        title = f"  (vs. HINE & MSR-HINE ablation: {ABLATION_LABELS.get(abl_name, abl_name)})"

        _plot_one_rollout(
            results     = abl_results,
            trajs       = trajs,
            traj_idx    = best_idx,
            traj_label  = "best",
            warmup_len  = warmup_len,
            n_steps     = n_steps,
            tau_lam     = tau_lam,
            out_dir     = ablation_dir,
            model_names = model_names,
            filename    = f"ablation_{abl_name}",
            title_suffix= title,
        )

        # Restore
        for k in list(MODEL_LABELS.keys()):
            if k not in orig_labels:
                del MODEL_LABELS[k]
            else:
                MODEL_LABELS[k] = orig_labels[k]


# ── Figure 5: Long-run vorticity degeneration ────────────────────────────────

def plot_longrun_degeneration(
    results:    dict,
    trajs:      np.ndarray,
    warmup_len: int,
    n_steps:    int,
    tau_lam:    float,
    out_dir:    Path,
):
    """Single row of snapshots per model at very long horizons.

    Shows vorticity field (no error) so the viewer can judge whether each model
    maintains coherent turbulent structures or collapses to artifacts.
    Models: fno_1step, unet_1step, hine, msr_hine.
    Columns: 5 timesteps spread across the full rollout horizon.
    """
    long_models = [n for n in ["fno_1step", "unet_1step", "hine", "msr_hine"]
                   if n in results]
    if len(long_models) < 2:
        print("  Skipping fig5: need at least 2 of fno/unet/hine/msr_hine")
        return

    # Use the worst MSR-HINE trajectory — most dramatic degeneration visible
    msr_rmse  = results["msr_hine"]["rmse"]
    traj_idx  = int(np.argmax(msr_rmse.mean(axis=1)))

    avail = min(n_steps, trajs.shape[1] - warmup_len - 1)
    # Columns: [0.25τ, 0.5τ, 1τ, 1.5τ, t=167]
    candidates = [
        int(tau_lam * 0.25),
        int(tau_lam * 0.5),
        int(tau_lam * 1.0),
        int(tau_lam * 1.5),
        167,
    ]
    t_idx = np.array([min(t, avail - 1) for t in candidates], dtype=int)

    gt_frames  = trajs[traj_idx][warmup_len + 1 : warmup_len + 1 + avail]
    gt_show    = gt_frames[t_idx]
    vmax_omega = float(np.percentile(np.abs(gt_show), 99))

    n_rows = 1 + len(long_models)   # GT + one row per model
    n_cols = len(t_idx)
    col_w, row_h = 2.3, 2.0

    fig = plt.figure(figsize=(n_cols * col_w + 2.0, n_rows * row_h + 0.6))
    gs  = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig,
                            hspace=0.06, wspace=0.04,
                            left=0.09, right=0.97,
                            width_ratios=[1] * n_cols + [0.06])

    all_rows = [("Ground\nTruth", None, gt_show)] + [
        (MODEL_LABELS.get(n, n), n,
         np.clip(results[n]["preds"][traj_idx, :avail, 0][t_idx],
                 -3 * vmax_omega, 3 * vmax_omega))
        for n in long_models
    ]

    for row_i, (row_label, model_name, frames) in enumerate(all_rows):
        color = MODEL_COLORS.get(model_name, "k") if model_name else "k"
        axes_row = []
        for col_i, (frame, t) in enumerate(zip(frames, t_idx)):
            ax  = fig.add_subplot(gs[row_i, col_i])
            im  = ax.imshow(frame, cmap="RdBu_r", vmin=-vmax_omega, vmax=vmax_omega,
                            origin="lower", interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            axes_row.append((ax, im))
            if row_i == 0:
                t_tau_str = f"{t / tau_lam:.1f}" + r"$\tau_\lambda$"
                ax.set_title(f"$t={t}$\n({t_tau_str})", fontsize=8, pad=3)
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=8.5, color=color,
                              rotation=0, ha="right", va="center", labelpad=28,
                              multialignment="center")
            lw = 1.6 if model_name == "msr_hine" else 0.8
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(lw)

        cbar_ax = fig.add_subplot(gs[row_i, n_cols])
        cb = fig.colorbar(axes_row[-1][1], cax=cbar_ax)
        cbar_ax.tick_params(labelsize=7)
        if row_i == 0:
            cb.set_label("$\\omega$", fontsize=8)

    fig.suptitle(
        f"Long-run vorticity degeneration  |  traj {traj_idx} (worst MSR-HINE)\n"
        f"MSR-HINE maintains coherent structures; baselines develop artifacts.",
        fontsize=9, y=1.005,
    )
    out = out_dir / "fig5_longrun_degeneration.pdf"
    fig.savefig(str(out)); fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")

    # ── Zoom-in companion plot at t=167 ───────────────────────────────────────
    _plot_longrun_zoomin(
        results=results, trajs=trajs, warmup_len=warmup_len,
        traj_idx=traj_idx, t_final=167, avail=avail,
        vmax_omega=vmax_omega, tau_lam=tau_lam, out_dir=out_dir,
    )


def _plot_longrun_zoomin(results, trajs, warmup_len, traj_idx, t_final,
                         avail, vmax_omega, tau_lam, out_dir):
    """Zoom-in on the region of strongest HINE artifacting at t=167.

    The box (rows 128:224, cols 112:176) is where HINE error peaks while
    MSR-HINE remains comparatively accurate — identified by scanning the
    HINE-error minus MSR-error gap across 32x32 patches.
    """
    t = min(t_final, avail - 1)

    gt_frames = trajs[traj_idx][warmup_len + 1 : warmup_len + 1 + avail]
    gt_full   = gt_frames[t]                                  # [H, W]
    H, W      = gt_full.shape

    if "msr_hine" not in results or "hine" not in results:
        print("  Skipping zoom-in: need both hine and msr_hine in results")
        return

    # Fixed box covering the centre-left strip where HINE shows worst artifacts
    # (rows 128-224, cols 112-176 — top-4 HINE-error patches all fall here)
    r0, c0 = 128, 112
    r1, c1 = 224, 176

    hine_full = results["hine"]["preds"][traj_idx, t, 0]
    msr_full  = results["msr_hine"]["preds"][traj_idx, t, 0]
    hine_mae  = float(np.abs(hine_full[r0:r1, c0:c1] - gt_full[r0:r1, c0:c1]).mean())
    msr_mae   = float(np.abs(msr_full[r0:r1, c0:c1]  - gt_full[r0:r1, c0:c1]).mean())
    print(f"  Zoom-in box: rows [{r0}:{r1}], cols [{c0}:{c1}]  "
          f"HINE MAE={hine_mae:.3f}  MSR-HINE MAE={msr_mae:.3f}")

    # ── Figure A: full fields with zoom-in rectangle (GT / HINE / MSR-HINE) ─
    zoom_row_models = ["hine", "msr_hine"]
    zoom_row_labels = ["Ground\nTruth"] + [MODEL_LABELS.get(n, n) for n in zoom_row_models]
    zoom_row_data   = [gt_full] + [
        np.clip(results[n]["preds"][traj_idx, t, 0], -3*vmax_omega, 3*vmax_omega)
        for n in zoom_row_models
    ]
    zoom_row_names  = [None] + zoom_row_models

    n_rows_z = len(zoom_row_labels)
    fig_a, axes_a = plt.subplots(1, n_rows_z, figsize=(n_rows_z * 3.5, 3.8))
    if n_rows_z == 1:
        axes_a = [axes_a]

    from matplotlib.patches import Rectangle
    for ax, label, frame, mname in zip(axes_a, zoom_row_labels, zoom_row_data, zoom_row_names):
        color = MODEL_COLORS.get(mname, "k") if mname else "k"
        ax.imshow(frame, cmap="RdBu_r", vmin=-vmax_omega, vmax=vmax_omega,
                  origin="lower", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=9, color=color, pad=4)
        # Draw zoom-in rectangle (origin='lower' flips rows)
        rect = Rectangle((c0, H - r1), c1 - c0, r1 - r0,
                          linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(1.2)

    t_tau_str = f"{t / tau_lam:.2f}"
    fig_a.suptitle(
        f"Full field at $t={t}$ ({t_tau_str}" + r"$\tau_\lambda$" + ")"
        f"  |  yellow box = best-match region (min MSR-HINE MAE)",
        fontsize=9,
    )
    plt.tight_layout()
    out_a = out_dir / "fig5b_zoomin_fullfield.pdf"
    fig_a.savefig(str(out_a)); fig_a.savefig(str(out_a).replace(".pdf", ".png"))
    plt.close(fig_a)

    # ── Figure B: magnified crops ─────────────────────────────────────────────
    crops = {}
    crops["gt"]   = gt_full[r0:r1, c0:c1]
    for n in zoom_row_models:
        pred = results[n]["preds"][traj_idx, t, 0]
        crops[n] = np.clip(pred[r0:r1, c0:c1], -3*vmax_omega, 3*vmax_omega)

    crop_labels = {"gt": "Ground\nTruth", "hine": MODEL_LABELS["hine"],
                   "msr_hine": MODEL_LABELS["msr_hine"]}
    crop_order  = ["gt"] + zoom_row_models
    vmax_crop   = float(np.percentile(np.abs(crops["gt"]), 99))

    fig_b, axes_b = plt.subplots(1, len(crop_order), figsize=(len(crop_order) * 3.2, 3.4))
    if len(crop_order) == 1:
        axes_b = [axes_b]

    for ax, key in zip(axes_b, crop_order):
        mname = key if key != "gt" else None
        color = MODEL_COLORS.get(mname, "k") if mname else "k"
        im = ax.imshow(crops[key], cmap="RdBu_r",
                       vmin=-vmax_crop, vmax=vmax_crop,
                       origin="lower", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(crop_labels[key], fontsize=9, color=color, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0 if key == "msr_hine" else 1.0)

    # Shared colorbar
    fig_b.subplots_adjust(right=0.88, wspace=0.06)
    cbar_ax = fig_b.add_axes([0.90, 0.12, 0.02, 0.76])
    fig_b.colorbar(im, cax=cbar_ax).set_label("$\\omega$", fontsize=9)

    fig_b.suptitle(
        f"Zoom-in crop ({r1-r0}×{c1-c0} px) at $t={t}$ "
        f"({t_tau_str}" + r"$\tau_\lambda$" + ")\n"
        f"Region of minimum MSR-HINE error  |  "
        f"HINE MAE={hine_mae:.3f}  MSR-HINE MAE={msr_mae:.3f}",
        fontsize=9,
    )
    out_b = out_dir / "fig5b_zoomin_crop.pdf"
    fig_b.savefig(str(out_b)); fig_b.savefig(str(out_b).replace(".pdf", ".png"))
    plt.close(fig_b)
    print(f"  Saved {out_a.name} + {out_b.name}")


# ── Figure 6: Saturation energy spectrum ─────────────────────────────────────

def plot_saturation_spectrum(
    results:    dict,
    trajs:      np.ndarray,
    warmup_len: int,
    n_steps:    int,
    out_dir:    Path,
):
    """Time-averaged E(k) over the last 20% of rollout steps vs GT climatology.

    Answers: does MSR-HINE maintain the correct inertial range at saturation,
    while 1-step baselines develop spurious energy at high-k?
    """
    avail       = min(n_steps, trajs.shape[1] - warmup_len - 1)
    sat_start   = int(avail * 0.8)           # last 20% of rollout
    sat_steps   = list(range(sat_start, avail))
    k_cut       = trajs.shape[-1] // 3      # 2/3 dealiasing cutoff

    def mean_spectrum(frames):
        """frames: [N_step, H, W] → mean radial spectrum [H//2]"""
        specs = []
        for f in frames:
            kb, Ek = radial_energy_spectrum(
                torch.from_numpy(f).float().unsqueeze(0))
            specs.append(Ek[0].numpy())
        return np.stack(specs).mean(axis=0)

    # GT climatological spectrum: average over all trajectories × saturation steps
    gt_specs = []
    for ti in range(trajs.shape[0]):
        gt_frames = trajs[ti][warmup_len + 1 : warmup_len + 1 + avail]
        gt_sat    = gt_frames[sat_steps]            # [n_sat, H, W]
        gt_specs.append(mean_spectrum(gt_sat))
    gt_mean = np.stack(gt_specs).mean(axis=0)       # [H//2]

    # Only HINE-L2 and MSR-HINE
    SAT_MODELS = [n for n in ["hine", "msr_hine"] if n in results]
    k_show = min(40, k_cut)   # truncate high-k for clarity

    model_sat = {}
    for name in SAT_MODELS:
        preds_all = results[name]["preds"]          # [N_traj, avail, 1, H, W]
        specs = []
        for ti in range(preds_all.shape[0]):
            sat_frames = preds_all[ti, sat_steps, 0]
            specs.append(mean_spectrum(sat_frames))
        model_sat[name] = np.stack(specs)           # [N_traj, H//2]

    # Compute kb from one spectrum call
    _s = torch.from_numpy(gt_frames[0]).float().unsqueeze(0)
    kb, _ = radial_energy_spectrum(_s)
    kb_np = kb.numpy()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # GT — thick dashed black
    ax.loglog(kb_np[1:k_show], gt_mean[1:k_show],
              color="k", lw=2.2, ls="--", label="GT (climatology)", zorder=5)

    for name in SAT_MODELS:
        arr  = model_sat[name]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        c    = MODEL_COLORS[name]
        lbl  = MODEL_LABELS[name]
        mean_clipped = np.clip(mean, 1e-30, None)
        ax.loglog(kb_np[1:k_show], mean_clipped[1:k_show],
                  color=c, lw=1.8, label=lbl, zorder=3)
        lo = np.maximum(mean - std, 1e-30)
        ax.fill_between(kb_np[1:k_show], lo[1:k_show], (mean+std)[1:k_show],
                        color=c, alpha=0.18, zorder=2)

    # k^{-3} reference slope
    k_ref = kb_np[3:k_show]
    E_ref = gt_mean[3]
    ax.loglog(k_ref, E_ref * (k_ref / k_ref[0]) ** (-3),
              "k:", lw=0.9, alpha=0.6, label=r"$k^{-3}$")

    ymin = gt_mean[1:k_show].min() * 0.05
    ymax = gt_mean[1:k_show].max() * 5.0
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([1, k_show])
    ax.set_xlabel("Wavenumber $k$", fontsize=11)
    ax.set_ylabel("$E(k)$", fontsize=11)
    ax.set_title(
        f"Time-averaged energy spectrum at saturation\n"
        f"(steps {sat_start}–{avail-1}, last 20% of rollout)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)

    # ── Inset zoom at very low-k (k=1–6) to show HINE-L2 mismatch at curve onset
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    k_zoom_lo, k_zoom_hi = 1, 6
    ki_lo = np.searchsorted(kb_np, k_zoom_lo)
    ki_hi = np.searchsorted(kb_np, k_zoom_hi) + 1

    # Upper-right: legend is lower-left, x-axis is at bottom — safe zone is top-right.
    # fig coords for a 5.5×4.5 figure with ~0.13 left margin and ~0.10 bottom margin:
    #   axes span roughly [0.13, 0.12] to [0.95, 0.88] in figure fraction.
    # Place inset in upper-right corner of that axes area, small (26%×24% of fig).
    axins = fig.add_axes([0.66, 0.60, 0.26, 0.24])   # [left, bottom, width, height]

    axins.loglog(kb_np[ki_lo:ki_hi], gt_mean[ki_lo:ki_hi],
                 color="k", lw=1.8, ls="--", zorder=5)
    for name in SAT_MODELS:
        arr  = model_sat[name]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        c    = MODEL_COLORS[name]
        mean_clipped = np.clip(mean, 1e-30, None)
        axins.loglog(kb_np[ki_lo:ki_hi], mean_clipped[ki_lo:ki_hi],
                     color=c, lw=1.4, zorder=3)
        lo = np.maximum(mean - std, 1e-30)
        axins.fill_between(kb_np[ki_lo:ki_hi], lo[ki_lo:ki_hi],
                           (mean+std)[ki_lo:ki_hi], color=c, alpha=0.20)

    axins.set_xlim([k_zoom_lo, k_zoom_hi])
    y_zoom_min = gt_mean[ki_lo:ki_hi].min() * 0.3
    y_zoom_max = gt_mean[ki_lo:ki_hi].max() * 3.0
    axins.set_ylim([y_zoom_min, y_zoom_max])
    # x-axis: place integer ticks at k=1,2,3,4,5,6 with small labels, no overlap
    axins.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 3, 4, 5, 6]))
    axins.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["1","2","3","4","5","6"]))
    axins.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # y-axis: 3 ticks max, no minor clutter
    axins.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
    axins.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axins.tick_params(axis="both", labelsize=5, pad=1)
    axins.set_xlabel("$k$", fontsize=5, labelpad=1)
    axins.grid(True, which="major", alpha=0.2)
    axins.set_title("Low-$k$ ($k\\!=\\!1$–$6$)", fontsize=6, pad=2)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.7)

    out = out_dir / "fig6_saturation_spectrum.pdf"
    fig.savefig(str(out))
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 4: E(k) log-log at 5 lead times ────────────────────────────────────

def plot_spectra(
    results:    dict,
    trajs:      np.ndarray,
    warmup_len: int,
    n_steps:    int,
    out_dir:    Path,
    tau_lam:    float = 83.856,
):
    cap    = int(round(2 * tau_lam))                        # 2 τ_λ in steps
    avail  = min(n_steps, cap, trajs.shape[1] - warmup_len - 1)
    t_idx  = np.linspace(0, avail - 1, 5, dtype=int)

    # Average spectra over all test trajectories
    def get_spectra_all_trajs(frames_all):
        """frames_all: [N, avail, H, W], return [5, H//2]"""
        N = frames_all.shape[0]
        specs = []
        for ti in range(N):
            traj_specs = []
            for t in t_idx:
                omega = torch.from_numpy(frames_all[ti, t]).float()
                kb, Ek = radial_energy_spectrum(omega.unsqueeze(0))
                traj_specs.append(Ek[0].numpy())
            specs.append(np.stack(traj_specs, axis=0))   # [5, H//2]
        return np.stack(specs, axis=0)   # [N, 5, H//2]

    # GT spectra
    gt_frames_all = np.stack([
        trajs[ti][warmup_len + 1 : warmup_len + 1 + avail]
        for ti in range(trajs.shape[0])
    ], axis=0)   # [N, avail, H, W]
    gt_specs = get_spectra_all_trajs(gt_frames_all)   # [N, 5, H//2]

    # Model spectra
    model_specs = {}
    for name in MODEL_ORDER:
        preds_all = results[name]["preds"][:, :avail, 0]   # [N, avail, H, W]
        model_specs[name] = get_spectra_all_trajs(preds_all)

    # Compute kb once from a single spectrum call
    _sample = torch.from_numpy(gt_frames_all[0, 0]).float()
    kb, _   = radial_energy_spectrum(_sample.unsqueeze(0))
    kb_np   = kb.numpy()

    SPEC_MODELS = [n for n in ["hine", "msr_hine"] if n in results]
    k_cut  = trajs.shape[-1] // 3   # 2/3 dealiasing cutoff
    k_show = min(40, k_cut)          # truncate high-k for clarity

    n_cols = 5
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.2, 3.5), sharey=True)

    for col_i, (t, ax) in enumerate(zip(t_idx, axes)):
        gt_mean = gt_specs[:, col_i].mean(axis=0)

        ymin_col = gt_mean[1:k_show].min() * 0.05
        ymax_col = gt_mean[1:k_show].max() * 5.0

        ax.loglog(kb_np[1:k_show], gt_mean[1:k_show], color="k",
                  lw=2.0, ls="--", label="GT" if col_i == 0 else None, zorder=5)

        for name in SPEC_MODELS:
            ms         = model_specs[name][:, col_i].mean(axis=0)
            c          = MODEL_COLORS[name]
            ms_clipped = np.clip(ms, 1e-30, ymax_col * 100)
            ax.loglog(kb_np[1:k_show], ms_clipped[1:k_show], color=c, lw=1.8,
                      label=MODEL_LABELS[name] if col_i == 0 else None, zorder=3)

        ax.set_ylim([ymin_col, ymax_col])
        ax.set_title(f"$t = {t}$", fontsize=9)
        ax.set_xlabel("Wavenumber $k$", fontsize=9)
        if col_i == 0:
            ax.set_ylabel("$E(k)$", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlim([1, k_show])

    axes[0].legend(fontsize=8, loc="lower left")

    k_ref = kb_np[5:k_show]
    E5    = gt_specs[:, -1, 5].mean()
    axes[-1].loglog(k_ref, E5 * (k_ref / k_ref[0]) ** (-3),
                    "k:", lw=0.8, alpha=0.6, label=r"$k^{-3}$")
    axes[-1].legend(fontsize=8, loc="lower left")

    fig.suptitle("Radial energy spectrum E(k) — model vs. ground truth",
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / "fig4_spectra.pdf"
    fig.savefig(str(out)); fig.savefig(str(out).replace(".pdf",".png"))
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations-dir", type=Path,
                        default=Path("outputs/ablations"))
    parser.add_argument("--data-root",     type=Path,
                        default=Path("data/kolmogorov"))
    parser.add_argument("--out-dir",       type=Path,
                        default=Path("outputs/paper_figures"))
    parser.add_argument("--n-rollout",     type=int,   default=200)
    parser.add_argument("--warmup",        type=int,   default=12)
    parser.add_argument("--tau-lam",       type=float, default=83.856)
    parser.add_argument("--dt-snapshot",   type=float, default=0.02)
    parser.add_argument("--n-trajs",       type=int,   default=10,
                        help="Number of test trajectories to evaluate (all=10)")
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument("--msr-hine-ckpt", type=Path,  default=None,
                        help="Override path to msr_hine best.pt (e.g. outputs/msr_hine_fixed_v3/checkpoints/best.pt)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load test trajectories
    test_h5 = args.data_root / "test.h5"
    print(f"Loading test data from {test_h5}...")
    with h5py.File(test_h5, "r") as f:
        trajs = f["vorticity"][:args.n_trajs].astype("float32")  # [N, T, H, W]
    print(f"  Loaded {trajs.shape[0]} trajectories, shape {trajs.shape}")

    # Load models — core three + all ablations
    # For msr_hine: prefer msr_hine_fixed if it exists (bug-fixed retraining),
    # fall back to msr_hine. Always labelled "MSR-HINE" in plots.
    all_model_names = MODEL_ORDER + ABLATION_NAMES
    print("Loading models...")
    models = {}
    for name in all_model_names:
        if name == "msr_hine":
            if args.msr_hine_ckpt is not None:
                ckpt_path = args.msr_hine_ckpt
                print(f"  msr_hine: using --msr-hine-ckpt override ({ckpt_path})")
            else:
                fixed_path = args.ablations_dir / "msr_hine_fixed" / "checkpoints" / "best.pt"
                orig_path  = args.ablations_dir / "msr_hine"       / "checkpoints" / "best.pt"
                if fixed_path.exists():
                    ckpt_path = fixed_path
                    print(f"  msr_hine: using fixed checkpoint ({fixed_path})")
                else:
                    ckpt_path = orig_path
        else:
            ckpt_path = args.ablations_dir / name / "checkpoints" / "best.pt"

        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found for {name}, skipping.")
            continue
        try:
            models[name] = load_model(ckpt_path, device)
        except RuntimeError as e:
            print(f"  WARNING: skipping {name} — {e}")
            continue
        n_p = sum(p.numel() for p in models[name].parameters())
        print(f"  {name}: {n_p:,} params")

    if not any(n in models for n in MODEL_ORDER):
        print("No core models found. Exiting.")
        return

    # ── Load ablation models for vorticity comparison panel ──────────────────
    ABLATION_CKPTS = {
        "no_multirate": Path("outputs/msr_hine_v4_no_multirate/checkpoints/best.pt"),
        "no_topdown":   Path("outputs/msr_hine_v4_no_topdown_v2/checkpoints/best.pt"),
        "single_scale": Path("outputs/ablations/single_scale/checkpoints/best.pt"),
    }
    ablation_models = {}
    for abl_name, ckpt_path in ABLATION_CKPTS.items():
        if not ckpt_path.exists():
            print(f"  ablation {abl_name}: checkpoint not found, skipping.")
            continue
        try:
            ablation_models[abl_name] = load_model(ckpt_path, device)
            print(f"  ablation {abl_name}: loaded")
        except RuntimeError as e:
            print(f"  ablation {abl_name}: skipping — {e}")

    # ── Compute rollout metrics for all trajectories ──────────────────────────
    print(f"\nRunning rollouts ({args.n_rollout} steps, {len(trajs)} trajectories)...")
    results = compute_metrics_per_traj(
        models, trajs, args.warmup, args.n_rollout, device
    )

    # Also compute metrics for ablation models (needed for vorticity panel preds)
    ablation_results = {}
    if ablation_models:
        print(f"Running rollouts for {len(ablation_models)} ablation models...")
        ablation_results = compute_metrics_per_traj(
            ablation_models, trajs, args.warmup, args.n_rollout, device
        )

    # ── Figure 1 & 2: RMSE and ACC ────────────────────────────────────────────
    print("\nPlotting Fig 1 & 2: RMSE / ACC curves...")
    plot_rmse_acc(results, args.tau_lam, args.dt_snapshot, args.out_dir)

    # ── Figure 3: Vorticity panels ────────────────────────────────────────────
    print("\nPlotting Fig 3: Vorticity rollout panels...")
    plot_vorticity_rollout(
        results, trajs, args.warmup, args.n_rollout, args.tau_lam, args.out_dir,
        ablation_results=ablation_results if ablation_results else None,
    )

    # ── Figure 3 ablations: GT / HINE / ablation-MSR-HINE per ablation ────────
    print("\nPlotting Fig 3 ablation panels...")
    plot_ablation_rollouts(
        results, trajs, args.warmup, args.n_rollout, args.tau_lam, args.out_dir
    )

    # ── Figure 4: Spectra ─────────────────────────────────────────────────────
    print("\nPlotting Fig 4: Energy spectra...")
    plot_spectra(results, trajs, args.warmup, args.n_rollout, args.out_dir, tau_lam=args.tau_lam)

    # ── Figure 5: Long-run degeneration ──────────────────────────────────────
    print("\nPlotting Fig 5: Long-run vorticity degeneration...")
    plot_longrun_degeneration(
        results, trajs, args.warmup, args.n_rollout, args.tau_lam, args.out_dir
    )

    # ── Figure 6: Saturation spectrum ────────────────────────────────────────
    print("\nPlotting Fig 6: Saturation energy spectrum...")
    plot_saturation_spectrum(
        results, trajs, args.warmup, args.n_rollout, args.out_dir
    )

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
