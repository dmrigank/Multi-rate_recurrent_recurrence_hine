# Changes from Initial Design — MSR-HINE-2D

This document records every deviation between the initial design spec (`DESIGN.md`) and the
final implemented and trained model (`msr_hine_v5_bounded_film`). Changes fall into three
categories: **structural corrections** (intentional architectural fixes to the original
design), **implementation bugs** found and fixed during development, and **hyperparameter
deviations** (where the trained config differs from DESIGN.md recommendations).

---

## 1. Structural corrections (intentional changes from initial design)

These are the ten pitfalls (P1–P10) documented in `DESIGN.md §0`. Each was a flaw in the
original 1D HINE formulation that was corrected for the 2D extension. They are listed here
for completeness; full rationale is in `DESIGN.md`.

| ID | Original mechanism | Correction in implementation |
|---|---|---|
| P1 | In-place hidden-state corrector `h ← h + αU(z − z_prior)` at inference | Removed entirely. Contraction enforced via spectral norm on GRU recurrent weights + bounded gains `α = α_max·σ(·)`, `α_max = 0.2` |
| P2 | Per-step predictor/corrector ignoring strides | Explicit multi-rate hold clock: GRU only evolves on update steps (`n mod s_l = 0`); prior held otherwise |
| P3 | Per-step `L_prior`; short TBPTT leaving coarse level gradient-starved | Stride-respecting losses computed only on update steps; TBPTT window ≥ 32 with gradient checkpointing |
| P4 | Learned encoders assumed to separate scales | Fixed nested spectral truncation `P^(l)` (radial low-pass); scale separation by construction |
| P5 | Redundant fusion gate overlapping with GRU update gate | Single mechanism: GRU update gate only; separate fusion gate removed |
| P6 | `h₀` unspecified (zero-init) | W=12 frame history warmup before free rollout; horizon reported from end of warmup |
| P7 | Raw hidden vector `h` concatenated directly into field backbone | Decoded priors injected at matching U-Net resolution stages; `h` injected via FiLM (per-channel γ, β) |
| P8 | "EnKF-style / Bayesian correction" framing | Reframed as self-consistent latent refinement (training regularizer only); no observation, no innovation |
| P9 | `L_cons` with undefined inter-level `Down` operator | `Down` = nested spectral truncation (exact, unambiguous); level-0 term dropped |
| P10 | Posterior re-encode + fusion at inference | **Posterior demoted to training-only consistency loss `L_cons`.** No inference-time re-encode, no fusion. Empirically confirmed: re-adding inference fusion (`msr_hine_v4_posterior_fusion`, VPH=1.28 τ_λ) performs worse than the full corrected model (VPH=1.60 τ_λ) |

---

## 2. Implementation bugs found and fixed during development

Four bugs were identified after the initial implementation pass and corrected before the
final training run (`v5_bounded_film`). All four affected the coarse recurrent level.

### Bug 1 — Coarse FiLM silently dropped at resolution 16

**Location:** `src/msr_hine/models/unet.py`, `UNetDecoder.forward()`

**Problem:** The decoder loop iterated over output resolutions (32, 64, 128, 256), looking
up `film_params` at each. The bottleneck resolution (16×16) is the loop *input*, never an
output, so `film_params[16]` was never applied. The coarse GRU's hidden state `h_coarse` had
zero influence through the FiLM pathway despite being computed correctly.

**Fix:** Added an explicit FiLM application at the bottleneck resolution before the decoder
loop begins:
```python
res_btl = self._stage_sizes[0]   # = 16
if res_btl in film_params:
    gamma, beta = film_params[res_btl]
    h = gamma * h + beta
```

---

### Bug 2 — `_film_ch_for_res(16)` returned wrong channel count

**Location:** `src/msr_hine/models/msr_hine.py`, `_film_ch_for_res()`

**Problem:** The helper that looks up how many channels the FiLM generator should produce
iterated `dec_sizes[1:]`, skipping index 0 (= resolution 16). It never matched `res=16` and
fell back to a default of 128 channels. The bottleneck actually has 256 channels (base 64 ×
mult 4). This meant `film_coarse = FiLMGenerator(coarse_dim, 128)` — half the required width
— so even if Bug 1 had been absent, the FiLM projection would have been wrong.

**Fix:** Changed the iteration to `dec_sizes` (all indices), raising a `ValueError` if no
match is found. For the 256-grid U-Net this correctly returns 256 for resolution 16.

---

### Bug 3 — Free-rollout seed off by one frame (backward diff wrong stride)

**Location:** `src/msr_hine/train.py` (`tbptt_step`, `train`), `src/msr_hine/rollout.py`
(`evaluate_trajectory`)

**Problem:** The first free-rollout step used `omega_seed = frame[W-1]` while the recurrent
state counter was at `step_n = W`. At that point the backward diff is:

```
z_current = E(P(frame[W-1]))
z_prev    = hist[W % stride] = E(P(frame[W - stride]))   ← from warmup
diff      = z[W-1] - z[W-stride]                         ← (stride-1)-step diff, not stride-step
```

For warmup=12, stride=2: diff = z[11] − z[10] = 1-step diff instead of the required 2-step.
For stride=4: diff = z[11] − z[8] = 3-step diff instead of 4-step.

**Fix:** Advance seed by one frame to `frame[W]`, making the TBPTT window size `warmup + K + 1`
(was `warmup + K`). At the corrected seed frame:
```
z_current = E(P(frame[W]))
z_prev    = hist[W % stride] = E(P(frame[W - stride]))   ← exactly stride steps ✓
```

Affected lines: `omega_in`, `gt_frame`, `ground_truth`, `target` in `tbptt_step`;
`window` size in `train`; `n_steps`, `omega_seed`, `omega_target` in `evaluate_trajectory`.

---

### Bug 4 — `validate()` skipped warmup (zero-initialized latent state)

**Location:** `src/msr_hine/train.py`, `validate()`

**Problem:** `validate()` called `model.init_state(B, device)` but never called
`model.warmup()`. The validation rollout ran from zero latent state, while training and
test evaluation both used warmed-up states. This made validation loss artificially low
(best-case lower bound) and therefore an unreliable early-stopping signal.

**Fix:** Added `state = model.warmup(omega_window[:, :warmup], state)` before the free
rollout in `validate()`, mirroring the training and evaluation paths exactly.

---

### Bug 5 — `seq_start_n` offset wrong for stride masking in `L_prior` / `L_cons`

**Location:** `src/msr_hine/train.py`, `tbptt_step()`

**Problem:** The stride mask `(seq_start_n + k) % stride == 0` used a synthetic
`seq_start_n = window_start` counter. After the Bug 3 fix, the first target frame is at
physical index `window_start + warmup + 1`, not `window_start`, so the offset was off by
`warmup + 1` steps.

**Fix:** Set `seq_start_n = warmup + 1` directly. This is exact when `window_stride` is
divisible by all recurrent strides (2 and 4), which the training config guarantees
(`window_stride = 16`).

---

## 3. Hyperparameter deviations from DESIGN.md recommendations

The following values in the final trained model differ from the recommendations in
`DESIGN.md §10`.

| Parameter | DESIGN.md recommendation | Actual trained value | Reason |
|---|---|---|---|
| TBPTT window `K` | ≥ 32 ("annealed up") | **16** | GPU memory constraint at 256² with batch size 2; TBPTT window=32 with K=32 exceeded VRAM. Gradient checkpointing partially compensates |
| Latent dim (medium) | 128 | **128** | Match |
| Latent dim (coarse) | 64 | **64** | Match |
| `λ_highk` | 0.0 (default) | **0.25** | Added in v5 to suppress high-k energy drift observed in v3/v4 |
| FiLM `gamma_mode` | "direct" (unspecified in DESIGN.md) | **"bounded_residual"** (`γ = 1 + 0.5·tanh(·)`) | Introduced in v5 to bound FiLM scale and prevent coarse FiLM from saturating after Bug 1/2 fix |
| FiLM `gamma_scale` | — | **0.5** | Paired with `bounded_residual` mode |
| Batch size | 4 | **2** | VRAM constraint on 256² with TBPTT=32 |
| AMP (mixed precision) | on | **off** | Disabled for stability on this hardware |
| Scheduled sampling `end_prob` | — | **0.2** (not fully self-fed) | Full self-feeding (`end_prob=0.0`) caused instability; held at 20% teacher-forcing through training |