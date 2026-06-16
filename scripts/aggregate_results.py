"""Aggregate ablation results: load per-run metrics, produce comparison table and plots.

Usage (called by run_ablations.sh after all runs complete):
    python scripts/aggregate_results.py \
        --ablations-dir outputs/ablations \
        --data-root data/kolmogorov \
        --n 256 \
        --output-dir outputs/ablations

Or, to evaluate a single checkpoint:
    python scripts/aggregate_results.py \
        --ckpt outputs/ablations/msr_hine/checkpoints/best.pt \
        --config-name msr_hine \
        --data-root data/kolmogorov \
        --output outputs/ablations/msr_hine/metrics.json \
        --n 256

Outputs:
    results_table.csv   — VPH (τ_λ), RMSE@t=τ_λ, spectral L1 per ablation
    plots/rmse_acc.png  — RMSE and ACC vs lead time for all models
    plots/spectra.png   — Long-horizon energy spectra
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def _rebuild_model(ckpt: dict) -> "torch.nn.Module | None":
    """Reconstruct the model from the config saved inside the checkpoint."""
    cfg_dict = ckpt.get("cfg")
    if cfg_dict is None:
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(cfg_dict)
        from msr_hine.train import build_model
        return build_model(cfg)
    except Exception as e:
        print(f"[aggregate] Could not rebuild model: {e}")
        return None


def evaluate_checkpoint(
    ckpt_path:   Path,
    model_name:  str,
    data_root:   Path,
    output_path: Path,
    n:           int,
    warmup_len:  Optional[int] = None,
    n_steps:     int   = 200,
    tau_lam:     float = 83.9,
    dt_snapshot: float = 0.025,
    device:      str   = "auto",
    max_trajs:   int   = 10,
) -> dict:
    """Load a checkpoint, rebuild model, run rollout on the test set, compute metrics.

    Rebuilds the model using the Hydra config stored inside the checkpoint
    (saved by train.py since the last ablation run).

    Args:
        ckpt_path:   Path to best.pt.
        model_name:  Name for labelling.
        data_root:   Root containing test.h5.
        output_path: Where to write metrics.json.
        n:           Spatial grid size (64 for debug, 256 for full).
        warmup_len:  Warmup frames (excluded from VPH, Invariant 7).
        n_steps:     Free-rollout steps.
        tau_lam:     Lyapunov time in snapshot steps (for VPH, Invariant 8).
        dt_snapshot: Physical time per snapshot.

    Returns:
        Metrics dict.
    """
    import h5py
    from msr_hine.rollout import evaluate_trajectory

    if not ckpt_path.exists():
        print(f"[aggregate] Checkpoint not found: {ckpt_path}")
        return {}

    test_h5 = data_root / "test.h5"
    if not test_h5.exists():
        print(f"[aggregate] Test split not found: {test_h5}")
        return {}

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    val_loss = float(ckpt.get("metrics", {}).get("val_loss", float("nan")))
    epoch    = ckpt.get("epoch", -1)
    if warmup_len is None:
        cfg_dict = ckpt.get("cfg", {})
        warmup_len = int(cfg_dict.get("train", {}).get("warmup_steps", 4))
    print(f"[aggregate] {model_name}: epoch={epoch+1 if isinstance(epoch,int) else epoch}, "
          f"val_loss={val_loss:.4f}, warmup={warmup_len}")

    # Rebuild model from saved config
    model = _rebuild_model(ckpt)
    if model is None:
        print(f"[aggregate] No config in checkpoint; skipping rollout for {model_name}.")
        metrics = {
            "name": model_name, "epoch": epoch,
            "val_loss": val_loss,
            "vph_tau_lambda": float("nan"),
            "vph_acc_steps": float("nan"),
            "rmse_at_1tau": float("nan"),
            "spec_l1": float("nan"),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        return metrics

    # Resolve device
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(_device)
    print(f"[aggregate] {model_name}: running on {_device}")

    # Load test trajectories — limit to max_trajs for speed
    with h5py.File(test_h5, "r") as f:
        trajs = f["vorticity"][:max_trajs]   # [N, T, n, n]
        meta  = dict(f.attrs)

    dt_snap = float(meta.get("dt_snapshot", dt_snapshot))
    N_traj  = trajs.shape[0]

    all_rmse_t = []
    all_acc_t  = []
    all_spec_t = []

    for ti in range(N_traj):
        traj = torch.from_numpy(trajs[ti].astype("float32")).unsqueeze(1).to(_device)  # [T,1,n,n]
        T    = traj.shape[0]
        avail = T - warmup_len - 1
        if avail <= 0:
            continue
        k_eval = min(n_steps, avail)

        try:
            traj_eval = traj[:warmup_len + 1 + k_eval]
            result = evaluate_trajectory(
                model            = model,
                omega_traj       = traj_eval,
                warmup_len       = warmup_len,
                tau_lambda_steps = tau_lam,
                dt_snapshot      = dt_snap,
            )
            # Trim to k_eval steps; move to CPU for numpy conversion
            all_rmse_t.append(result["rmse"][:k_eval].cpu().numpy())
            all_acc_t .append(result["acc"] [:k_eval].cpu().numpy())
            if "spec_error" in result:
                all_spec_t.append(result["spec_error"][:k_eval].cpu().numpy())
        except Exception as e:
            print(f"[aggregate] Eval failed for traj {ti}: {e}")

    if not all_rmse_t:
        print(f"[aggregate] No valid trajectories for {model_name}")
        metrics = {"name": model_name, "epoch": epoch, "val_loss": val_loss,
                   "vph_tau_lambda": float("nan"), "vph_acc_steps": float("nan"),
                   "rmse_at_1tau": float("nan"), "spec_l1": float("nan")}
    else:
        rmse_mean = np.stack(all_rmse_t, axis=0).mean(axis=0)  # [k_eval]
        acc_mean  = np.stack(all_acc_t,  axis=0).mean(axis=0)

        import torch as _torch
        acc_t  = _torch.from_numpy(acc_mean)
        rmse_t = _torch.from_numpy(rmse_mean)

        from msr_hine.metrics import (
            valid_prediction_horizon, vph_from_rmse, climatology_std
        )

        # VPH from ACC (primary metric, Invariant 8)
        vph_acc = valid_prediction_horizon(
            acc_t, tau_lambda_steps=tau_lam, dt_snapshot=dt_snap, threshold=0.5)

        # Also compute clim_std from the test trajectories for RMSE-based VPH
        clim = float(
            trajs[:, warmup_len + 1 : warmup_len + 1 + k_eval].std()
        )
        vph_rmse = vph_from_rmse(
            rmse_t, clim_std=clim,
            tau_lambda_steps=tau_lam, dt_snapshot=dt_snap, threshold=0.65)

        # RMSE at 1 τ_λ
        tau_step = max(1, int(round(tau_lam)))
        rmse_1tau = float(rmse_mean[min(tau_step - 1, len(rmse_mean) - 1)])

        spec_l1 = float(np.stack(all_spec_t).mean()) if all_spec_t else float("nan")

        metrics = {
            "name":           model_name,
            "epoch":          epoch + 1 if isinstance(epoch, int) else epoch,
            "val_loss":       val_loss,
            "vph_tau_lambda": float(vph_acc["tau_lambda"]),
            "vph_acc_steps":  float(vph_acc["steps"]),
            "vph_rmse_tau":   float(vph_rmse["tau_lambda"]),
            "rmse_at_1tau":   rmse_1tau,
            "spec_l1":        spec_l1,
            "rmse_curve":     rmse_mean.tolist(),
            "acc_curve":      acc_mean.tolist(),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"[aggregate] {model_name}: VPH(ACC)={metrics.get('vph_tau_lambda','?'):.2f} τ_λ  "
          f"RMSE@1τ={metrics.get('rmse_at_1tau','?'):.4f}  → {output_path.name}")
    return metrics


# ---------------------------------------------------------------------------
# Aggregation from multiple runs
# ---------------------------------------------------------------------------

def aggregate_all(
    ablations_dir: Path,
    data_root:     Path,
    n:             int,
    output_dir:    Path,
    warmup_len:    Optional[int] = None,
    n_steps:       int   = 200,
    tau_lam:       float = 83.9,
    dt_snapshot:   float = 0.025,
    device:        str   = "auto",
    max_trajs:     int   = 10,
) -> None:
    """Run full evaluation (rollout + metrics) for each ablation, then produce plots."""
    ablation_names = [
        "fno_1step", "hine", "msr_hine",
        "single_scale", "no_multirate", "no_topdown",
        "no_consistency", "no_contraction", "no_warmup",
        "circularity_confirm",
    ]

    rows = []
    for name in ablation_names:
        ckpt_path  = ablations_dir / name / "checkpoints" / "best.pt"
        output_path = ablations_dir / name / "metrics.json"

        if not ckpt_path.exists():
            print(f"[aggregate] No checkpoint for {name}, skipping.")
            continue

        m = evaluate_checkpoint(
            ckpt_path   = ckpt_path,
            model_name  = name,
            data_root   = data_root,
            output_path = output_path,
            n           = n,
            warmup_len  = warmup_len,
            n_steps     = n_steps,
            tau_lam     = tau_lam,
            dt_snapshot = dt_snapshot,
            device      = device,
            max_trajs   = max_trajs,
        )
        if m:
            rows.append(m)

    if not rows:
        print("[aggregate] No results found.")
        return

    if not rows:
        print("[aggregate] No results found.")
        return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_table.csv"
    header = "model,epoch,val_loss,vph_tau_lambda,rmse_at_tau,spec_l1\n"
    lines  = [header]
    for r in rows:
        lines.append(
            f"{r.get('name','?')},"
            f"{r.get('epoch','?')},"
            f"{r.get('val_loss',       float('nan')):.4f},"
            f"{r.get('vph_tau_lambda', float('nan')):.3f},"
            f"{r.get('rmse_at_1tau',   float('nan')):.4f},"
            f"{r.get('spec_l1',        float('nan')):.4f}\n"
        )
    csv_path.write_text("".join(lines))
    print(f"[aggregate] Results table → {csv_path}")

    # ── Print table ────────────────────────────────────────────────────────────
    print()
    print(f"{'Model':<22} {'Epoch':>6} {'Val L_state':>12} "
          f"{'VPH (τ_λ)':>10} {'RMSE@1τ':>9}")
    print("-" * 65)
    for r in rows:
        print(f"{r.get('name','?'):<22} "
              f"{str(r.get('epoch','?')):>6} "
              f"{r.get('val_loss',       float('nan')):>12.4f} "
              f"{r.get('vph_tau_lambda', float('nan')):>10.3f} "
              f"{r.get('rmse_at_1tau',   float('nan')):>9.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_val_losses(rows, output_dir)
    _plot_vph_bars(rows, plots_dir)
    _plot_rmse_acc_curves(rows, plots_dir)


def _plot_vph_bars(rows: list[dict], plots_dir: Path) -> None:
    """Horizontal bar chart of VPH in τ_λ units."""
    names = [r.get("name", "?") for r in rows]
    vphs  = [r.get("vph_tau_lambda", float("nan")) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.5))
    y   = np.arange(len(names))
    colors = ["steelblue" if n == "msr_hine" else
              "firebrick" if n == "circularity_confirm" else "gray"
              for n in names]
    bars = ax.barh(y, vphs, color=colors, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("VPH (τ_λ units)", fontsize=11)
    ax.set_title("Valid prediction horizon — ablation study", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vphs):
        if np.isfinite(val):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=7)
    plt.tight_layout()
    out = plots_dir / "ablation_vph.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[aggregate] Plot → {out.name}")


def _plot_rmse_acc_curves(rows: list[dict], plots_dir: Path) -> None:
    """RMSE and ACC vs. lead time for all ablations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    for r, c in zip(rows, colors):
        name       = r.get("name", "?")
        rmse_curve = r.get("rmse_curve")
        acc_curve  = r.get("acc_curve")
        lw  = 2.0  if name in ("msr_hine", "fno_1step", "hine") else 1.2
        ls  = "--" if name == "circularity_confirm" else "-"
        if rmse_curve:
            t = np.arange(len(rmse_curve))
            axes[0].semilogy(t, rmse_curve, label=name, color=c, lw=lw, ls=ls)
        if acc_curve:
            t = np.arange(len(acc_curve))
            axes[1].plot(t, acc_curve, label=name, color=c, lw=lw, ls=ls)

    for ax, title, ylabel in zip(axes,
                                  ["RMSE vs. lead time", "ACC vs. lead time"],
                                  ["RMSE", "ACC"]):
        ax.set_xlabel("Lead-time step"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    axes[1].axhline(0.5, color="k", ls=":", lw=0.8, label="ACC=0.5 threshold")
    plt.tight_layout()
    out = plots_dir / "rmse_acc_curves.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[aggregate] Plot → {out.name}")


def _parse_checkpoint(ckpt_path: Path) -> tuple[float, object]:
    """Read val_loss and epoch from a checkpoint file."""
    if not ckpt_path.exists():
        return float("nan"), "?"
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        val_loss = ckpt.get("metrics", {}).get("val_loss", float("nan"))
        epoch    = ckpt.get("epoch", "?")
        if isinstance(epoch, int):
            epoch += 1   # stored as 0-based, display as 1-based
        # Reject inf (training resumed from checkpoint and skipped all epochs)
        if not np.isfinite(val_loss):
            val_loss = float("nan")
        return float(val_loss), epoch
    except Exception:
        return float("nan"), "?"


def _parse_train_log(log_path: Path) -> tuple[float, object]:
    """Parse the last valid val loss and epoch from a train.log file."""
    last_val   = float("nan")
    last_epoch = "?"
    try:
        for line in log_path.read_text().splitlines():
            # Match lines like: "Epoch    5/5 | train L=... | val L=0.0615 | ..."
            if "val L=" in line and "Epoch" in line:
                try:
                    tok = line.split("val L=")[1].split()[0]
                    val = float(tok)
                    if np.isfinite(val):
                        last_val = val
                    # Parse epoch
                    ep_part = line.split("Epoch")[1].strip().split()[0]
                    last_epoch = ep_part.split("/")[0].strip()
                except Exception:
                    pass
            # Also accept "Best val loss: X.XXXX" as a last resort
            elif "Best val loss:" in line and np.isnan(last_val):
                try:
                    val = float(line.split("Best val loss:")[1].strip())
                    if np.isfinite(val):
                        last_val = val
                except Exception:
                    pass
    except Exception:
        pass
    return last_val, last_epoch


def _plot_val_losses(rows: list[dict], output_dir: Path) -> None:
    """Bar chart of final validation losses across ablations."""
    names  = [r.get("name", "?") for r in rows]
    losses = [r.get("val_loss", float("nan")) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    colors = ["steelblue" if n == "msr_hine" else
              "firebrick" if n == "circularity_confirm" else
              "gray" for n in names]
    bars = ax.bar(x, losses, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Final val L_state", fontsize=11)
    ax.set_title("Ablation study: final validation loss", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, losses):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(str(plots_dir / "ablation_val_losses.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] Plot → {plots_dir}/ablation_val_losses.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablations-dir", type=Path, default=None,
                   help="Directory containing per-ablation subdirs (aggregate mode).")
    p.add_argument("--ckpt",          type=Path, default=None,
                   help="Single checkpoint path (single-eval mode).")
    p.add_argument("--config-name",   default="unknown",
                   help="Ablation name for single-eval labelling.")
    p.add_argument("--data-root",     type=Path, required=True)
    p.add_argument("--output",        type=Path, default=None,
                   help="Output metrics.json path (single-eval mode).")
    p.add_argument("--output-dir",    type=Path, default=Path("outputs/ablations"),
                   help="Output directory (aggregate mode).")
    p.add_argument("--n",             type=int,   default=256)
    p.add_argument("--warmup",        type=int,   default=None,
                   help="Warmup frames; defaults to train.warmup_steps from checkpoint.")
    p.add_argument("--n-steps",       type=int,   default=200)
    p.add_argument("--tau-lam",       type=float, default=83.9)
    p.add_argument("--dt-snapshot",   type=float, default=0.025)
    p.add_argument("--device",        type=str,   default="auto",
                   help="Compute device: auto, cuda, cpu, cuda:N")
    p.add_argument("--max-trajs",     type=int,   default=10,
                   help="Max test trajectories to average over")
    args = p.parse_args()

    if args.ablations_dir is not None:
        # Aggregate mode — run full evaluation + plots
        aggregate_all(
            ablations_dir = args.ablations_dir,
            data_root     = args.data_root,
            n             = args.n,
            output_dir    = args.output_dir,
            warmup_len    = args.warmup,
            n_steps       = args.n_steps,
            tau_lam       = args.tau_lam,
            dt_snapshot   = args.dt_snapshot,
            device        = args.device,
            max_trajs     = args.max_trajs,
        )
    elif args.ckpt is not None:
        # Single-eval mode
        out = args.output or (args.ckpt.parent.parent / "metrics.json")
        evaluate_checkpoint(
            ckpt_path   = args.ckpt,
            model_name  = args.config_name,
            data_root   = args.data_root,
            output_path = out,
            n           = args.n,
            warmup_len  = args.warmup,
            n_steps     = args.n_steps,
            tau_lam     = args.tau_lam,
            dt_snapshot = args.dt_snapshot,
            device      = args.device,
            max_trajs   = args.max_trajs,
        )
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
