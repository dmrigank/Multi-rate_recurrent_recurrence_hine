"""Generate LaTeX and CSV tables for the MSR-HINE-2D paper.

Table 1 — Rollout RMSE at 5 equally-spaced timesteps for all 4 models.
Table 2 — Long-horizon summary metrics + relative gains over HINE at t=200.
Table 3 — Ablation study comparison.

Usage:
    python scripts/generate_tables.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/paper_figures_v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_LAM = 83.856   # Lyapunov time in snapshot steps

# ── Metric files ──────────────────────────────────────────────────────────────
CORE = {
    "UNet-AR":  Path("outputs/ablations/unet_1step/metrics.json"),
    "FNO-AR":   Path("outputs/ablations/fno_1step/metrics.json"),
    "HINE-L2":  Path("outputs/ablations/hine/metrics.json"),
    "MSR-HINE": Path("outputs/msr_hine_v5_bounded_film/metrics.json"),
}
# First mean-ACC index below 0.5 using the fixed full-test climatology
# convention of Fig. 1.
CORE_VPH_CROSSING_STEPS = {
    "UNet-AR":  13,
    "FNO-AR":   56,
    "HINE-L2":  106,
    "MSR-HINE": 156,
}
# Mean and population standard deviation over the 10 test trajectories from
# the corrected 200-step Fig. 1 rollout. ACC uses the fixed full-test
# climatology. VPH means remain the crossing of the mean ACC curve, while
# VPH std is the spread of the per-trajectory crossing horizons.
RECURRENT_TEST_STATS = {
    "HINE-L2": {
        "rmse_table1_mean": [0.069612, 2.189005, 3.764862, 4.952164, 6.124406],
        "rmse_table1_std":  [0.013323, 0.261087, 0.407625, 0.319613, 0.657779],
        "vph_std": 0.172335,
        "acc_final_mean": 0.171741,
        "acc_final_std": 0.054614,
        "spec_mean": 0.244063,
        "spec_std": 0.093865,
    },
    "MSR-HINE": {
        "rmse_table1_mean": [0.065585, 1.599916, 2.770633, 3.747958, 4.834300],
        "rmse_table1_std":  [0.013042, 0.178987, 0.305310, 0.441492, 0.563554],
        "vph_std": 0.138707,
        "acc_final_mean": 0.438711,
        "acc_final_std": 0.056192,
        "spec_mean": 0.110964,
        "spec_std": 0.029860,
    },
}
ABLATIONS = {
    "MSR-HINE (full)": Path("outputs/msr_hine_v5_bounded_film/metrics.json"),
    "No multirate":    Path("outputs/msr_hine_v4_no_multirate/metrics.json"),
    "No top-down":     Path("outputs/msr_hine_v4_no_topdown_v2/metrics.json"),
    "Single scale":    Path("outputs/ablations/single_scale/metrics.json"),
}
# First mean-ACC index below 0.5 using the Fig. 1 convention: a single
# climatology over all test trajectories and timesteps, followed by averaging
# ACC over the 10 test trajectories. The per-run metrics.json files use a
# different, trajectory-local climatology and therefore must not supply Table 3
# VPH values.
ABLATION_VPH_CROSSING_STEPS = {
    "MSR-HINE (full)": 156,
    "No multirate":    152,
    "No top-down":     154,
    "Single scale":    137,
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def safe(val, fmt=".4f", max_val=9999.0):
    """Format val; return '—' if nan/inf/None or exceeds max_val (diverged)."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "—"
    if max_val is not None and abs(val) > max_val:
        return "—"
    return format(val, fmt)


def _get(arr, idx):
    """Return float(arr[idx]) or nan if out of bounds or non-finite."""
    if idx >= len(arr):
        return float("nan")
    v = float(arr[idx])
    return v if np.isfinite(v) else float("nan")


def pct_gain(baseline, model_val, higher_better=False):
    """% gain of model_val over baseline (positive = improvement)."""
    if not np.isfinite(baseline) or not np.isfinite(model_val):
        return float("nan")
    raw = (baseline - model_val) / abs(baseline) * 100
    return raw if not higher_better else -raw


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Rollout RMSE at 5 equally-spaced timesteps
# t indices are five equally spaced points through approximately 2 tau_lambda.
# ─────────────────────────────────────────────────────────────────────────────
# 5 equally-spaced steps up to 2 τ_λ (≈167 steps, 0-based index 167)
T_MAX    = int(round(2 * TAU_LAM)) - 1   # = 167
T_STEPS  = [int(round(i * T_MAX / 4)) for i in range(5)]   # 0, 42, 84, 126, 167
T_LABELS = [f"$t={t+1}$" for t in T_STEPS]

rows1 = []
for label, path in CORE.items():
    m   = load(path)
    rc  = m.get("rmse_curve", [])
    stats = RECURRENT_TEST_STATS.get(label)
    means = (
        stats["rmse_table1_mean"]
        if stats is not None
        else [_get(rc, t) for t in T_STEPS]
    )
    stds = stats["rmse_table1_std"] if stats is not None else [None] * len(T_STEPS)
    rows1.append(dict(label=label, means=means, stds=stds))

# CSV
csv1 = OUT_DIR / "table1_rollout_rmse.csv"
with open(csv1, "w") as f:
    mean_cols = [f"t={t+1}" for t in T_STEPS]
    std_cols = [f"t={t+1}_std" for t in T_STEPS]
    f.write("Model," + ",".join(mean_cols + std_cols) + "\n")
    for row in rows1:
        means = [safe(v, ".4f") for v in row["means"]]
        stds = ["" if v is None else safe(v, ".4f") for v in row["stds"]]
        f.write(",".join([row["label"]] + means + stds) + "\n")

# LaTeX
header1 = " & ".join(["Model"] + T_LABELS)
tex1_rows = []
for row in rows1:
    cells = []
    for mean, std in zip(row["means"], row["stds"]):
        cell = safe(mean, ".4f")
        if std is not None:
            cell += r" $\pm$ " + safe(std, ".4f")
        cells.append(cell)
    tex1_rows.append(" & ".join([row["label"]] + cells) + r" \\")
tex1 = (
    r"\begin{table}[t]" + "\n"
    r"\centering" + "\n"
    r"\caption{Rollout RMSE at five equally-spaced lead times up to $2\,\tau_\lambda$ "
    r"($\approx 168$ steps). Each step $\approx 0.02$ time units. "
    r"For HINE-L2 and MSR-HINE, values are mean $\pm$ one standard deviation "
    r"over 10 test trajectories. "
    r"`—' = diverged (non-finite).}" + "\n"
    r"\label{tab:rollout_rmse}" + "\n"
    r"\resizebox{\columnwidth}{!}{%" + "\n"
    r"\begin{tabular}{lrrrrr}" + "\n"
    r"\toprule" + "\n"
    + header1 + r" \\" + "\n"
    r"\midrule" + "\n"
    + "\n".join(tex1_rows) + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}}" + "\n"
    r"\end{table}"
)
(OUT_DIR / "table1_rollout_rmse.tex").write_text(tex1)

# Console
print("=" * 68)
print("TABLE 1 — Rollout RMSE at 5 timesteps")
print("=" * 68)
W = 11
print(f"{'Model':<18}" + "".join(f"{'t='+str(t+1):>{W}}" for t in T_STEPS))
print("-" * (18 + W * len(T_STEPS)))
for row in rows1:
    values = [safe(v, ".4f") for v in row["means"]]
    print(f"{row['label']:<18}" + "".join(f"{v:>{W}}" for v in values))
print()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — Long-horizon summary + relative gains over HINE at 2 tau_lambda
# Note: UNet-AR and FNO diverge before 2 tau_lambda; deltas use HINE baseline.
# ─────────────────────────────────────────────────────────────────────────────
IDX_FINAL = int(round(2 * TAU_LAM)) - 1   # 0-based → t ≈ 2 τ_λ (step 167)

hine_stats  = RECURRENT_TEST_STATS["HINE-L2"]
hine_rmse_f = hine_stats["rmse_table1_mean"][-1]
hine_acc_f  = hine_stats["acc_final_mean"]
hine_spec   = hine_stats["spec_mean"]

rows2 = []
for label, path in CORE.items():
    m      = load(path)
    rc     = m.get("rmse_curve", [])
    ac     = m.get("acc_curve",  [])
    stats  = RECURRENT_TEST_STATS.get(label)
    rmse_f = (
        stats["rmse_table1_mean"][-1]
        if stats is not None else _get(rc, IDX_FINAL)
    )
    acc_f  = (
        stats["acc_final_mean"]
        if stats is not None else _get(ac, IDX_FINAL)
    )
    spec   = (
        stats["spec_mean"]
        if stats is not None else float(m.get("spec_l1", float("nan")))
    )
    vph    = CORE_VPH_CROSSING_STEPS[label] / TAU_LAM
    vph_std = stats["vph_std"] if stats is not None else None
    rmse_std = stats["rmse_table1_std"][-1] if stats is not None else None
    acc_std = stats["acc_final_std"] if stats is not None else None
    spec_std = stats["spec_std"] if stats is not None else None

    # Suppress deltas for diverged models (RMSE > 9999 or non-finite)
    diverged = (not np.isfinite(rmse_f)) or (rmse_f > 9999.0)
    d_rmse = float("nan") if diverged else pct_gain(hine_rmse_f, rmse_f, higher_better=False)
    d_acc  = float("nan") if diverged else pct_gain(hine_acc_f,  acc_f,  higher_better=True)
    d_spec = float("nan") if diverged else pct_gain(hine_spec,   spec,   higher_better=False)

    rows2.append(dict(label=label, vph=vph, vph_std=vph_std,
                      rmse=rmse_f, acc=acc_f, spec=spec,
                      rmse_std=rmse_std, acc_std=acc_std, spec_std=spec_std,
                      d_rmse=d_rmse, d_acc=d_acc, d_spec=d_spec))

# CSV
csv2 = OUT_DIR / "table2_summary_metrics.csv"
with open(csv2, "w") as f:
    f.write("Model,VPH(tau_lam),VPH_std,RMSE@t168,RMSE_std,ACC@t168,ACC_std,"
            "SpecErr,SpecErr_std,"
            "delta_RMSE%_vs_HINE-L2,delta_ACC%_vs_HINE-L2,delta_SpecErr%_vs_HINE-L2\n")
    for r in rows2:
        f.write(f"{r['label']},"
                f"{safe(r['vph'],'.3f')},"
                f"{'' if r['vph_std'] is None else safe(r['vph_std'],'.3f')},"
                f"{safe(r['rmse'],'.4f')},"
                f"{'' if r['rmse_std'] is None else safe(r['rmse_std'],'.4f')},"
                f"{safe(r['acc'],'.4f')},"
                f"{'' if r['acc_std'] is None else safe(r['acc_std'],'.4f')},"
                f"{safe(r['spec'],'.4f')},"
                f"{'' if r['spec_std'] is None else safe(r['spec_std'],'.4f')},"
                f"{safe(r['d_rmse'],'+.1f')},"
                f"{safe(r['d_acc'],'+.1f')},"
                f"{safe(r['d_spec'],'+.1f')}\n")

# LaTeX
def mean_std_tex(mean, std, fmt):
    text = safe(mean, fmt)
    return text if std is None else text + r" $\pm$ " + safe(std, fmt)


tex2_rows = "\n".join(
    f"{r['label']} & "
    f"{mean_std_tex(r['vph'], r['vph_std'], '.3f')} & "
    f"{mean_std_tex(r['rmse'], r['rmse_std'], '.4f')} & "
    f"{mean_std_tex(r['acc'], r['acc_std'], '.4f')} & "
    f"{mean_std_tex(r['spec'], r['spec_std'], '.4f')} & "
    f"{safe(r['d_rmse'],'+.1f')} & "
    f"{safe(r['d_acc'],'+.1f')} & "
    f"{safe(r['d_spec'],'+.1f')}" + r" \\"
    for r in rows2
)
tex2 = (
    r"\begin{table}[t]" + "\n"
    r"\centering" + "\n"
    r"\caption{Long-horizon summary metrics at $2\,\tau_\lambda$ ($\approx 168$ steps) "
    r"and valid prediction horizon (VPH). VPH uses the fixed full-test climatology "
    r"convention of Fig.~1. HINE-L2 and MSR-HINE values are mean $\pm$ one "
    r"standard deviation over 10 test trajectories; the VPH center is the crossing "
    r"of the mean ACC curve and its uncertainty is the per-trajectory VPH spread. "
    r"$\Delta$\% columns show relative improvement over HINE "
    r"(positive\,=\,better). UNet-AR and FNO-AR diverge before $2\,\tau_\lambda$ (—).}" + "\n"
    r"\label{tab:summary_metrics}" + "\n"
    r"\resizebox{\columnwidth}{!}{%" + "\n"
    r"\begin{tabular}{lrrrrrrrr}" + "\n"
    r"\toprule" + "\n"
    r"Model & VPH\,($\tau_\lambda$) & RMSE & ACC & SpecErr"
    r" & $\Delta$RMSE\,\% & $\Delta$ACC\,\% & $\Delta$SpecErr\,\% \\" + "\n"
    r"\midrule" + "\n"
    + tex2_rows + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}}" + "\n"
    r"\end{table}"
)
(OUT_DIR / "table2_summary_metrics.tex").write_text(tex2)

# Console
t_final_label = IDX_FINAL + 1
print("=" * 92)
print(f"TABLE 2 — Long-horizon summary metrics + relative gains over HINE at t={t_final_label} (≈2τ_λ)")
print("=" * 92)
hdr = (f"{'Model':<18} {'VPH(τ)':>8} {f'RMSE@{t_final_label}':>10} {f'ACC@{t_final_label}':>9} "
       f"{'SpecErr':>10} {'ΔRMSE%':>8} {'ΔACC%':>7} {'ΔSpec%':>8}")
print(hdr)
print("-" * len(hdr))
for r in rows2:
    print(f"{r['label']:<18} "
          f"{safe(r['vph'],'.3f'):>8} "
          f"{safe(r['rmse'],'.4f'):>10} "
          f"{safe(r['acc'],'.4f'):>9} "
          f"{safe(r['spec'],'.4f'):>10} "
          f"{safe(r['d_rmse'],'+.1f'):>8} "
          f"{safe(r['d_acc'],'+.1f'):>7} "
          f"{safe(r['d_spec'],'+.1f'):>8}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — Ablation study
# Metrics: VPH(τ_λ), RMSE@1τ, SpecErr, ΔVPH%, ΔRMSE%, ΔSpecErr% vs full model
# ─────────────────────────────────────────────────────────────────────────────
full_m    = load(ABLATIONS["MSR-HINE (full)"])
full_vph  = ABLATION_VPH_CROSSING_STEPS["MSR-HINE (full)"] / TAU_LAM
full_rmse = float(full_m.get("rmse_at_1tau",   float("nan")))
full_spec = float(full_m.get("spec_l1",        float("nan")))

rows3 = []
for label, path in ABLATIONS.items():
    m    = load(path)
    vph  = ABLATION_VPH_CROSSING_STEPS[label] / TAU_LAM
    rmse = float(m.get("rmse_at_1tau",   float("nan")))
    spec = float(m.get("spec_l1",        float("nan")))

    d_vph  = 0.0 if label == "MSR-HINE (full)" else pct_gain(
        full_vph, vph, higher_better=True
    )
    d_rmse = pct_gain(full_rmse, rmse, higher_better=False)
    d_spec = pct_gain(full_spec, spec, higher_better=False)

    rows3.append(dict(label=label, vph=vph, rmse=rmse, spec=spec,
                      d_vph=d_vph, d_rmse=d_rmse, d_spec=d_spec))

# CSV
csv3 = OUT_DIR / "table3_ablations.csv"
with open(csv3, "w") as f:
    f.write("Model,VPH(tau_lam),RMSE@1tau,SpecErr,delta_VPH%,delta_RMSE%,delta_SpecErr%\n")
    for r in rows3:
        f.write(f"{r['label']},"
                f"{safe(r['vph'],'.3f')},"
                f"{safe(r['rmse'],'.4f')},"
                f"{safe(r['spec'],'.4f')},"
                f"{safe(r['d_vph'],'+.1f')},"
                f"{safe(r['d_rmse'],'+.1f')},"
                f"{safe(r['d_spec'],'+.1f')}\n")

# LaTeX
tex3_rows = "\n".join(
    f"{r['label']} & "
    f"{safe(r['vph'],'.3f')} & "
    f"{safe(r['rmse'],'.4f')} & "
    f"{safe(r['spec'],'.4f')} & "
    f"{safe(r['d_vph'],'+.1f')} & "
    f"{safe(r['d_rmse'],'+.1f')} & "
    f"{safe(r['d_spec'],'+.1f')}" + r" \\"
    for r in rows3
)
tex3 = (
    r"\begin{table}[t]" + "\n"
    r"\centering" + "\n"
    r"\caption{Ablation study. Each variant removes one mechanism from "
    r"MSR-HINE. $\Delta$\% is the relative change vs.\ the full model "
    r"(negative\,=\,degradation). VPH uses the fixed full-test climatology "
    r"convention of Fig.~1; RMSE and SpecErr are evaluated at $1\,\tau_\lambda$.}" + "\n"
    r"\label{tab:ablation}" + "\n"
    r"\begin{tabular}{lrrrrrr}" + "\n"
    r"\toprule" + "\n"
    r"Model & VPH\,($\tau_\lambda$) & RMSE@$1\tau$ & SpecErr"
    r" & $\Delta$VPH\,\% & $\Delta$RMSE\,\% & $\Delta$SpecErr\,\% \\" + "\n"
    r"\midrule" + "\n"
    + tex3_rows + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}" + "\n"
    r"\end{table}"
)
(OUT_DIR / "table3_ablations.tex").write_text(tex3)

# Console
print("=" * 75)
print("TABLE 3 — Ablation study")
print("=" * 75)
hdr = (f"{'Model':<22} {'VPH(τ)':>8} {'RMSE@1τ':>9} "
       f"{'SpecErr':>9} {'ΔVPH%':>7} {'ΔRMSE%':>8} {'ΔSpec%':>8}")
print(hdr)
print("-" * len(hdr))
for r in rows3:
    print(f"{r['label']:<22} "
          f"{safe(r['vph'],'.3f'):>8} "
          f"{safe(r['rmse'],'.4f'):>9} "
          f"{safe(r['spec'],'.4f'):>9} "
          f"{safe(r['d_vph'],'+.1f'):>7} "
          f"{safe(r['d_rmse'],'+.1f'):>8} "
          f"{safe(r['d_spec'],'+.1f'):>8}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4 — Model parameter counts across datasets
# 2D Kolmogorov: read from checkpoints. 1D KS / L96: analytical from tables.
# ─────────────────────────────────────────────────────────────────────────────

def _unet_1d_params(C_base=64, mults=(1,2,2,4,4), n_res=2, k=5):
    def gn(ch): return 2 * ch
    def res(ic, oc):
        return (ic*oc*k+oc) + (oc*oc*k+oc) + gn(ic) + gn(oc) + ((ic*oc+oc) if ic!=oc else 0)
    def attn(ch): return 4*(ch*ch+ch)
    chs = [C_base*m for m in mults]; n = len(chs); tot = 1*chs[0]*k+chs[0]
    enc = [chs[0]]
    for i, ch in enumerate(chs):
        ic = enc[-1]
        for b in range(n_res): tot += res(ic if b==0 else ch, ch)
        if i == n-1: tot += attn(ch)
        if i < n-1:  tot += ch*ch*k+ch
        enc.append(ch)
    for _ in range(n_res): tot += res(chs[-1], chs[-1])
    prev = chs[-1]
    for i, ch in enumerate(reversed(chs)):
        ic0 = prev + ch
        for b in range(n_res): tot += res(ic0 if b==0 else ch, ch)
        if i == 0: tot += attn(ch)
        if i < n-1: tot += ch*ch*k+ch
        prev = ch
    tot += chs[0]*1*1+1
    return tot

def _gru(inp, hid):
    return sum(hid*inp + hid*hid + 2*hid for _ in range(3))

def _film(h, ch):   return h*(2*ch) + 2*ch
def _enc(i, l):     return i*l + l
def _dec(l, o):     return l*o + o
def _prior(h, l):   return h*l + l

_U1D = _unet_1d_params()

def _ks_params():
    d_in, d_lat, d_hid = (64,48), (32,16), (120,80)
    film_ch = (128, 128)
    hine_extra = sum(_enc(i,l)+_dec(l,i) for i,l in zip(d_in,d_lat))
    msr_extra  = sum(_enc(i,l)+_gru(i,h)+_film(h,fc)+_dec(l,i)+_prior(h,l)
                     for i,l,h,fc in zip(d_in,d_lat,d_hid,film_ch))
    return _U1D, _U1D+hine_extra, _U1D+msr_extra

def _l96_params():
    d_in, d_lat, d_hid = (48,32), (20,8), (64,48)
    film_ch = (64, 128)
    hine_extra = sum(_enc(i,l)+_dec(l,i) for i,l in zip(d_in,d_lat))
    msr_extra  = sum(_enc(i,l)+_gru(i,h)+_film(h,fc)+_dec(l,i)+_prior(h,l)
                     for i,l,h,fc in zip(d_in,d_lat,d_hid,film_ch))
    return _U1D, _U1D+hine_extra, _U1D+msr_extra

# 2D actuals from checkpoints
import torch
from omegaconf import OmegaConf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msr_hine.train import build_model as _build

def _count(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m  = _build(OmegaConf.create(ck["cfg"]))
        return sum(p.numel() for p in m.parameters())
    except Exception:
        return None

def _ckpt(metrics_path: Path, fname="best.pt") -> str:
    return str(metrics_path.parent / "checkpoints" / fname)

_2d = {
    "FNO-AR":   _count(_ckpt(CORE["FNO-AR"])),
    "UNet-AR":  _count(_ckpt(CORE["UNet-AR"])),
    "HINE-L2":  _count(_ckpt(CORE["HINE-L2"])),
    "MSR-HINE": _count(_ckpt(CORE["MSR-HINE"], fname="best_long.pt")),
}

_ks_unet, _ks_hine, _ks_msr   = _ks_params()
_l96_unet, _l96_hine, _l96_msr = _l96_params()

PARAM_ROWS = [
    # (display_name, 2D, KS, L96)   — None = not applicable
    ("FNO-AR",   _2d["FNO-AR"],   None,      None),
    ("UNet-AR",  _2d["UNet-AR"],  _ks_unet,  _l96_unet),
    ("HINE-L2",  _2d["HINE-L2"], _ks_hine,  _l96_hine),
    ("MSR-HINE", _2d["MSR-HINE"],_ks_msr,   _l96_msr),
]

def _fmt_M(v):
    if v is None: return "—"
    return f"{v/1e6:.2f}"

# CSV
csv4 = OUT_DIR / "table4_param_counts.csv"
with open(csv4, "w") as fh:
    fh.write("Model,2D_Kflow_M,1D_KS_M,1D_L96_M\n")
    for name, d2, ks, l96 in PARAM_ROWS:
        fh.write(f"{name},{_fmt_M(d2)},{_fmt_M(ks)},{_fmt_M(l96)}\n")

# LaTeX
tex4_rows = "\n".join(
    f"{name} & {_fmt_M(d2)} & {_fmt_M(ks)} & {_fmt_M(l96)}" + r" \\"
    for name, d2, ks, l96 in PARAM_ROWS
)
tex4 = (
    r"\begin{table}[t]" + "\n"
    r"\centering" + "\n"
    r"\caption{Model parameter counts (millions) across datasets. "
    r"2D Kolmogorov Flow counts are measured from trained checkpoints. "
    r"1D KS and L96 counts are derived analytically from the hyperparameter "
    r"tables using the same U-Net backbone ($C_\text{base}=64$, kernel$=5$). "
    r"FNO-AR is only used as a 2D cross-dataset baseline.}" + "\n"
    r"\label{tab:param_counts}" + "\n"
    r"\begin{tabular}{lrrr}" + "\n"
    r"\toprule" + "\n"
    r"Model & 2D Kflow (M) & 1D KS (M) & 1D L96 (M) \\" + "\n"
    r"\midrule" + "\n"
    + tex4_rows + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}" + "\n"
    r"\end{table}"
)
(OUT_DIR / "table4_param_counts.tex").write_text(tex4)

# Console
print("=" * 55)
print("TABLE 4 — Model parameter counts (M)")
print("=" * 55)
print(f"{'Model':<12} {'2D Kflow':>10} {'1D KS':>10} {'1D L96':>10}")
print("-" * 46)
for name, d2, ks, l96 in PARAM_ROWS:
    print(f"{name:<12} {_fmt_M(d2):>10} {_fmt_M(ks):>10} {_fmt_M(l96):>10}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5 — Training and inference timing (2D Kolmogorov, RTX 4090)
# Numbers come from scripts/measure_timing.py — run that first.
# ─────────────────────────────────────────────────────────────────────────────
TIMING_JSON = OUT_DIR / "timing.json"

if TIMING_JSON.exists():
    import json as _json
    timing = _json.loads(TIMING_JSON.read_text())

    TIMING_ORDER = ["FNO-AR", "UNet-AR", "HINE-L2", "MSR-HINE"]

    def _ft(v, fmt): return format(v, fmt) if v is not None else "—"

    rows5 = []
    for label in TIMING_ORDER:
        r = timing.get(label, {})
        rows5.append(dict(
            label    = label,
            inf_ms   = r.get("inf_ms"),
            train_s  = r.get("train_s"),
        ))

    # CSV
    csv5 = OUT_DIR / "table5_timing.csv"
    with open(csv5, "w") as fh:
        fh.write("Model,Inference_ms_per_step,Training_s_per_iter\n")
        for r in rows5:
            fh.write(f"{r['label']},"
                     f"{_ft(r['inf_ms'],'.2f')},"
                     f"{_ft(r['train_s'],'.3f')}\n")

    # LaTeX
    tex5_rows = "\n".join(
        f"{r['label']} & {_ft(r['inf_ms'],'.2f')} & {_ft(r['train_s'],'.3f')}" + r" \\"
        for r in rows5
    )
    tex5 = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Inference and training wall-clock times on the 2D Kolmogorov "
        r"flow task ($256\times256$, batch size 1, NVIDIA RTX 4090). "
        r"Inference reports the mean time per autoregressive step (100 steps). "
        r"Training reports the mean time per iteration including warmup ($W=12$) "
        r"and rollout ($K=16$) with backpropagation.}" + "\n"
        r"\label{tab:timing}" + "\n"
        r"\begin{tabular}{lrr}" + "\n"
        r"\toprule" + "\n"
        r"Model & Inference (ms/step) & Training (s/iter, $K=16$) \\" + "\n"
        r"\midrule" + "\n"
        + tex5_rows + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    (OUT_DIR / "table5_timing.tex").write_text(tex5)

    print("=" * 52)
    print("TABLE 5 — Training & inference timing (RTX 4090)")
    print("=" * 52)
    print(f"{'Model':<12} {'Inf (ms/step)':>15} {'Train (s/iter)':>16}")
    print("-" * 45)
    for r in rows5:
        print(f"{r['label']:<12} {_ft(r['inf_ms'],'.2f'):>15} {_ft(r['train_s'],'.3f'):>16}")
    print()
else:
    print("WARNING: timing.json not found — skipping Table 5. "
          "Run scripts/measure_timing.py first.")
    csv5 = None

print(f"Files written to {OUT_DIR}/")
files = [csv1, csv2, csv3, csv4,
         OUT_DIR / "table1_rollout_rmse.tex",
         OUT_DIR / "table2_summary_metrics.tex",
         OUT_DIR / "table3_ablations.tex",
         OUT_DIR / "table4_param_counts.tex"]
if csv5:
    files += [csv5, OUT_DIR / "table5_timing.tex"]
for f in files:
    print(f"  {f.name}")
