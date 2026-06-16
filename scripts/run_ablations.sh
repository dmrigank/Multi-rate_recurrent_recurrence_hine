#!/usr/bin/env bash
# run_ablations.sh — train and evaluate all MSR-HINE ablations.
#
# Usage:
#   bash scripts/run_ablations.sh                   # full dataset, default epochs
#   bash scripts/run_ablations.sh --debug           # debug dataset, 5 epochs
#   bash scripts/run_ablations.sh --debug --dry-run # print commands only
#
# Output:
#   outputs/ablations/<name>/   — checkpoints + metrics JSON per run
#   outputs/ablations/results_table.csv
#   outputs/ablations/plots/    — comparison figures
#
# Environment:
#   conda activate 2d_hine   (must be active)
#   DATA_ROOT (default: data/kolmogorov)
#   DEVICE    (default: auto)
#   EPOCHS    (default: 100 for full, 5 for debug)

set -euo pipefail

# ── Parse args ──────────────────────────────────────────────────────────────
DEBUG=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --debug)   DEBUG=true  ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-data/kolmogorov}"
DEVICE="${DEVICE:-auto}"
TAU_LAM="${TAU_LAM:-83.9}"
EVAL_STEPS="${EVAL_STEPS:-200}"
EVAL_TRAJS="${EVAL_TRAJS:-10}"
OUT_DIR="outputs/ablations"
# Resolve python: prefer the active conda env, fall back to explicit path
if command -v python &>/dev/null && python -c "import msr_hine" 2>/dev/null; then
    PYTHON="$(command -v python)"
elif [ -f "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="$(conda run -n 2d_hine which python 2>/dev/null || echo python)"
fi

if [ "$DEBUG" = true ]; then
    DATA_ROOT="${DATA_ROOT}/debug"
    EPOCHS="${EPOCHS:-5}"
    N_DATA=64
    # Warmup / rollout for stateful models
    WARMUP=2; ROLLOUT=4
    # FNO-specific: shorter rollout, bigger batch (no state)
    FNO_WARMUP=1; FNO_ROLLOUT=1; FNO_BATCH=4; FNO_STRIDE=4
    MSR_WARMUP=2; MSR_ROLLOUT=4; MSR_BATCH=2; MSR_STRIDE=4
    EARLY_STOP=3    # fast feedback in debug mode
    AMP_FLAG="train.amp=false"
    echo "[run_ablations] DEBUG mode: data=${DATA_ROOT}, epochs=${EPOCHS}"
else
    EPOCHS="${EPOCHS:-100}"
    N_DATA=256
    # FNO: K=1 (one-step model — no multi-step rollout needed), B=16, stride=16
    # → ~97 steps/epoch → ~0.2 min/epoch → ~0.3 h for 100 ep (+ early stop ≈ faster)
    FNO_WARMUP=1; FNO_ROLLOUT=1; FNO_BATCH=16; FNO_STRIDE=16
    # MSR-HINE/HINE: B=2, K=16 (coarse gets 4 updates/window), stride=24
    # → ~520 steps/epoch → ~8 min/epoch; ~13 h for 100 ep
    MSR_WARMUP=4; MSR_ROLLOUT=16; MSR_BATCH=2; MSR_STRIDE=24
    WARMUP=$MSR_WARMUP; ROLLOUT=$MSR_ROLLOUT
    # Early stopping: stop if val loss doesn't improve for N epochs
    EARLY_STOP=15
    AMP_FLAG="train.amp=false"
    echo "[run_ablations] FULL mode: data=${DATA_ROOT}, epochs=${EPOCHS}"
fi

mkdir -p "${OUT_DIR}"

# ── Common Hydra overrides (apply to all runs; per-model overrides below) ─────
COMMON="hydra.run.dir=. device=${DEVICE} \
data.dataset_root=${DATA_ROOT} data.n=${N_DATA} \
train.epochs=${EPOCHS} train.num_workers=0 ${AMP_FLAG} \
+train.early_stopping_patience=${EARLY_STOP}"

# ── Helper: run one experiment ───────────────────────────────────────────────
run_exp() {
    local name="$1"
    local extra_args="${2:-}"
    local out="${OUT_DIR}/${name}"

    # Remove checkpoints only when --fresh flag is set (default: keep existing)
    if [ "${FRESH:-false}" = "true" ]; then
        rm -rf "${out}/checkpoints"
    fi
    mkdir -p "${out}"

    local cmd="${PYTHON} -m msr_hine.train \
        ${COMMON} \
        output_dir=${out} \
        ${extra_args}"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Running: ${name}"
    echo "  Extra:   ${extra_args}"
    echo "══════════════════════════════════════════════════════"

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $cmd"
        return
    fi

    eval "$cmd" 2>&1 | tee "${out}/train.log"
    echo "[run_ablations] Finished: ${name}"
}

# ── Evaluate helper ──────────────────────────────────────────────────────────
eval_exp() {
    local name="$1"
    local extra_args="${2:-}"
    local out="${OUT_DIR}/${name}"

    local cmd="${PYTHON} scripts/aggregate_results.py \
        --ckpt ${out}/checkpoints/best.pt \
        --config-name ${name} \
        --data-root ${DATA_ROOT} \
        --output ${out}/metrics.json \
        --n ${N_DATA} \
        --n-steps ${EVAL_STEPS} \
        --tau-lam ${TAU_LAM} \
        --max-trajs ${EVAL_TRAJS}"

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run eval] $cmd"
        return
    fi

    if [ -f "${out}/checkpoints/best.pt" ]; then
        echo "[run_ablations] Evaluating: ${name}"
        eval "$cmd" 2>&1 || echo "[WARNING] Evaluation failed for ${name}"
    else
        echo "[WARNING] No checkpoint found for ${name}, skipping eval."
    fi
}

# ── Per-model training hyperparameters ────────────────────────────────────────
# FNO: short rollout (no state to propagate), large batch
FNO_ARGS="train.warmup_steps=${FNO_WARMUP} train.rollout_steps=${FNO_ROLLOUT} \
  train.batch_size=${FNO_BATCH} ++train.window_stride=${FNO_STRIDE} \
  train.lambda_prior=0.0 train.lambda_cons=0.0 train.lambda_spec=0.01 \
  train.scheduled_sampling.start_prob=1.0 train.scheduled_sampling.end_prob=0.0"

# MSR/HINE: longer rollout for recurrence, smaller batch due to memory
MSR_TRAIN="train.warmup_steps=${MSR_WARMUP} train.rollout_steps=${MSR_ROLLOUT} \
  train.batch_size=${MSR_BATCH} ++train.window_stride=${MSR_STRIDE} \
  train.lambda_spec=0.01 train.lambda_highk=1.0 \
  train.scheduled_sampling.start_prob=1.0 train.scheduled_sampling.end_prob=0.2"

# MSR-HINE base: model + training
MSR_BASE="model=model_msr_hine model.name=msr_hine \
  train.lambda_prior=0.1 train.lambda_cons=0.1 ${MSR_TRAIN}"

# ── Experiment list ───────────────────────────────────────────────────────────
# 1. FNO baseline — short rollout, big batch
run_exp "fno_1step" "model=model_fno model.name=fno_1step ${FNO_ARGS}"

# 2. HINE — same rollout budget as MSR-HINE for fair comparison
run_exp "hine" "model=model_msr_hine model.name=hine \
  +model.medium_dim=128 +model.coarse_dim=64 +model.enc_hidden_ch=32 \
  train.lambda_prior=0.0 train.lambda_cons=0.0 ${MSR_TRAIN}"

# 3. Full MSR-HINE
run_exp "msr_hine" "${MSR_BASE}"

# 4. Internal ablations (each toggles exactly one mechanism)
run_exp "single_scale"   "${MSR_BASE} +model.single_scale=true train.lambda_cons=0.0"
run_exp "no_multirate"   "${MSR_BASE} model.recurrence.medium_stride=1 \
                          model.recurrence.coarse_stride=1 \
                          +train.medium_stride=1 +train.coarse_stride=1"
run_exp "no_topdown"     "${MSR_BASE} +model.use_topdown=false"
run_exp "no_consistency" "${MSR_BASE} train.lambda_cons=0.0"
run_exp "no_contraction" "${MSR_BASE} model.recurrence.use_contraction=false"
run_exp "no_warmup"      "${MSR_BASE} +model.use_warmup=false"

# 5. Circularity-confirmation (Invariant 1 intentionally violated as control)
run_exp "circularity_confirm" "${MSR_BASE} +model._inference_fusion_CONTROL_ONLY=true"

# ── Aggregate results ─────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Aggregating results"
echo "══════════════════════════════════════════════════════"

if [ "$DRY_RUN" = false ]; then
    ${PYTHON} scripts/aggregate_results.py \
        --ablations-dir "${OUT_DIR}" \
        --data-root "${DATA_ROOT}" \
        --n "${N_DATA}" \
        --output-dir "${OUT_DIR}" \
        --warmup "${WARMUP}" \
        --n-steps "${EVAL_STEPS}" \
        --tau-lam "${TAU_LAM}" \
        --max-trajs "${EVAL_TRAJS}" \
        --dt-snapshot 0.025 \
        2>&1 | tee "${OUT_DIR}/aggregate.log"
    echo "[run_ablations] Results table → ${OUT_DIR}/results_table.csv"
    echo "[run_ablations] Plots        → ${OUT_DIR}/plots/"
fi

echo ""
echo "[run_ablations] Done."
