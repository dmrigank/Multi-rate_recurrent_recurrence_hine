"""Plot ablation comparison: MSR-HINE full vs no_multirate, no_topdown, single_scale."""

from __future__ import annotations
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf
from msr_hine.train import build_model
from msr_hine.rollout import evaluate_trajectory

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RUNS = {
    "MSR-HINE (full)": ("outputs/msr_hine_v5_bounded_film/checkpoints/best.pt", "#1f77b4", "-",  2.0),
    "No multirate":    ("outputs/msr_hine_v4_no_multirate/checkpoints/best.pt",  "#d62728", "--", 1.4),
    "No top-down":     ("outputs/msr_hine_v4_no_topdown_v2/checkpoints/best.pt",  "#e377c2", "--", 1.4),
    "Single scale":    ("outputs/ablations/single_scale/checkpoints/best.pt",    "#8c564b", "--", 1.4),
}

WARMUP    = 12
N_ROLLOUT = 200
TAU_LAM   = 83.856
DT_SNAP   = 0.02
DATA_ROOT = Path("data/kolmogorov")
OUT_DIR   = Path("outputs/paper_figures_v5/ablations")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with h5py.File(DATA_ROOT / "test.h5", "r") as f:
        trajs = f["vorticity"][:].astype("float32")  # [N, T, H, W]
    print(f"Loaded {trajs.shape[0]} test trajectories")

    results = {}
    for label, (ckpt_path, color, ls, lw) in RUNS.items():
        ck    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(OmegaConf.create(ck["cfg"]))
        model.load_state_dict(ck["model"])
        model.eval().to(device)

        rmse_list, acc_list = [], []
        for ti in range(trajs.shape[0]):
            traj = torch.from_numpy(trajs[ti]).unsqueeze(1).to(device)
            res  = evaluate_trajectory(
                model, traj,
                warmup_len       = WARMUP,
                tau_lambda_steps = TAU_LAM,
                dt_snapshot      = DT_SNAP,
            )
            rmse_list.append(res["rmse"][:N_ROLLOUT].cpu().numpy())
            acc_list .append(res["acc"] [:N_ROLLOUT].cpu().numpy())

        results[label] = {
            "rmse":  np.stack(rmse_list),
            "acc":   np.stack(acc_list),
            "color": color, "ls": ls, "lw": lw,
        }
        print(f"  {label}: done")

    steps = np.arange(N_ROLLOUT)

    # ── RMSE / ACC figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, ylabel, logy) in zip(axes, [
        ("rmse", "RMSE", True),
        ("acc",  "ACC",  False),
    ]):
        for label, d in results.items():
            mean = d[metric].mean(axis=0)
            std  = d[metric].std(axis=0)
            ax.plot(steps, mean, color=d["color"], label=label,
                    ls=d["ls"], lw=d["lw"], zorder=3)
            lo = np.maximum(mean - std, 1e-6 if logy else 0)
            ax.fill_between(steps, lo, mean + std,
                            color=d["color"], alpha=0.12, zorder=2)

        ax.axvline(TAU_LAM, color="k", ls="--", lw=0.9, alpha=0.7,
                   label=r"$\tau_\lambda$")
        if metric == "acc":
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Lead time (snapshot steps)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right" if metric == "rmse" else "lower left")
        ax.grid(True, which="both" if logy else "major", alpha=0.25)

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(OUT_DIR / ("ablation_rmse_acc" + ext)))
    plt.close(fig)
    print("Saved ablation_rmse_acc.pdf/.png")

    # ── CSV table ─────────────────────────────────────────────────────────────
    tau_step = int(round(TAU_LAM))
    rows = []
    for label, d in results.items():
        rmse_mean = d["rmse"].mean(axis=0)
        acc_mean  = d["acc"].mean(axis=0)
        vph_steps = next((i for i, v in enumerate(acc_mean) if v < 0.5), N_ROLLOUT)
        vph_tau   = vph_steps / TAU_LAM
        rmse_1tau = float(rmse_mean[min(tau_step - 1, len(rmse_mean) - 1)])
        rows.append((label, vph_tau, vph_steps, rmse_1tau))

    csv_path = OUT_DIR / "ablation_table.csv"
    with open(csv_path, "w") as f:
        f.write("model,vph_tau_lambda,vph_steps,rmse_at_1tau\n")
        for label, vph_tau, vph_steps, rmse_1tau in rows:
            f.write(f"{label},{vph_tau:.3f},{vph_steps},{rmse_1tau:.4f}\n")

    print()
    print(f"{'Model':<22}  {'VPH (tau_lam)':>13}  {'VPH steps':>9}  {'RMSE@1tau':>9}")
    print("-" * 60)
    for label, vph_tau, vph_steps, rmse_1tau in rows:
        print(f"{label:<22}  {vph_tau:>13.3f}  {vph_steps:>9}  {rmse_1tau:>9.4f}")
    print()
    print(f"CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
