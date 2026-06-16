"""Measure per-step inference time and per-iteration training time for all 2D models."""
from __future__ import annotations
import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from omegaconf import OmegaConf
from msr_hine.train import build_model
from msr_hine.models.hine import HINE
from msr_hine.models.msr_hine import MSRHINE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CKPTS = {
    "FNO-AR":   "outputs/ablations/fno_1step/checkpoints/best.pt",
    "UNet-AR":  "outputs/ablations/unet_1step/checkpoints/best.pt",
    "HINE-L2":  "outputs/ablations/hine/checkpoints/best.pt",
    "MSR-HINE": "outputs/msr_hine_v5_bounded_film/checkpoints/best_long.pt",
}

B        = 1
H, W     = 256, 256
WARMUP   = 12
K        = 16          # rollout steps for training timing
N_INF    = 100         # timed inference steps
N_TRAIN  = 10          # timed training iters (after 3 warmup iters)

def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()

def _is_stateful(m):
    return isinstance(m, (HINE, MSRHINE))

def _step(model, omega, state):
    """Single forward step; returns (omega_next, state_next)."""
    if isinstance(model, MSRHINE):
        omega_next, state_next = model.step(omega, state)
        return omega_next, state_next
    elif isinstance(model, HINE):
        omega_next, state_next = model.forward_with_state(omega, state)
        return omega_next, state_next
    else:
        return model(omega), None

def _train_iter(model, opt, omega_seq):
    """One full TBPTT iter: warmup + K-step rollout + backward."""
    warmup_seq = omega_seq[:, :WARMUP]          # [B, W, 1, H, W]
    seed       = omega_seq[:, WARMUP]           # [B, 1, H, W]
    targets    = omega_seq[:, WARMUP+1:WARMUP+1+K]  # [B, K, 1, H, W]

    opt.zero_grad()
    if _is_stateful(model):
        state = model.init_state(B, device)
        state = model.warmup(warmup_seq, state)
    else:
        state = None

    cur  = seed.clone()
    loss = torch.tensor(0.0, device=device)
    for k in range(K):
        cur, state = _step(model, cur, state)
        loss = loss + ((cur - targets[:, k]) ** 2).mean()
    loss.backward()
    opt.step()

results = {}

for name, ckpt_path in CKPTS.items():
    print(f"\n{'─'*50}")
    print(f"  {name}")
    ck    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(OmegaConf.create(ck["cfg"])).to(device)

    # ── Inference timing ────────────────────────────────────────
    model.eval()
    omega = torch.randn(B, 1, H, W, device=device)
    state = model.init_state(B, device) if _is_stateful(model) else None
    if _is_stateful(model):
        dummy_warmup = torch.randn(B, WARMUP, 1, H, W, device=device)
        state = model.warmup(dummy_warmup, state)

    with torch.no_grad():
        for _ in range(20):           # JIT / cache warmup
            omega, state = _step(model, omega, state)
        sync()
        t0 = time.perf_counter()
        for _ in range(N_INF):
            omega, state = _step(model, omega, state)
        sync()
        t1 = time.perf_counter()
    inf_ms = (t1 - t0) / N_INF * 1000
    print(f"  Inference: {inf_ms:.2f} ms/step")

    # ── Training timing ─────────────────────────────────────────
    model.train()
    opt       = torch.optim.Adam(model.parameters(), lr=1e-4)
    omega_seq = torch.randn(B, WARMUP + K + 1, 1, H, W, device=device)

    for _ in range(3):                # warmup iters
        _train_iter(model, opt, omega_seq)
    sync()
    t0 = time.perf_counter()
    for _ in range(N_TRAIN):
        _train_iter(model, opt, omega_seq)
    sync()
    t1 = time.perf_counter()
    train_s = (t1 - t0) / N_TRAIN
    print(f"  Training:  {train_s:.3f} s/iter  (K={K}, warmup={WARMUP})")

    results[name] = {"inf_ms": round(inf_ms, 2), "train_s": round(train_s, 3)}

# ── Save JSON ───────────────────────────────────────────────────
out = Path("outputs/paper_figures_v5/timing.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")

print("\n" + "="*55)
print(f"{'Model':<12} {'Inf (ms/step)':>15} {'Train (s/iter)':>16}")
print("-" * 45)
for name, r in results.items():
    print(f"{name:<12} {r['inf_ms']:>15.2f} {r['train_s']:>16.3f}")
