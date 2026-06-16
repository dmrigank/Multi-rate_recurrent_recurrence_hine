# MSR-HINE-2D: Design Specification

Multiscale Recurrent Hierarchical Implicit Neural Emulator for stabilizing long-horizon
autoregressive rollouts of chaotic PDEs, on 2D forced Kolmogorov flow. This is the canonical
method specification (source of truth). It extends the 1D study (KS, L96) to 2D and corrects
the structural flaws found in the original formulation. The hierarchy, per-scale recurrent
memory, multi-rate clock, conditioning inputs, U-Net backbone, and loss family are preserved
from the original HINE line; the changes are surgical and each is tied to the pitfall it
removes.

---

## 0. Change log (what moved, and why)

| # | Original mechanism | Problem | Correction |
|---|---|---|---|
| P1 | In-place hidden-state corrector `h ← h + αU(z − z_prior)` | Autograd hazard + non-contractive feedback that can amplify drift | Removed as an inference-time state update. Field→latent coupling now via RNN conditioning only; contraction enforced via spectral norm + bounded gain |
| P2 | Per-step predictor/corrector vs. multi-rate strides | Strides say "hold," predictor/corrector say "update every step" | Explicit hold-and-skip clock on RNN evolution, prior emission, and per-level losses |
| P3 | Per-step `L_prior`, short TBPTT with stride-4 coarse | Coarse level gets almost no gradient; prior loss fights the slow-manifold goal | Stride-respecting losses; TBPTT ≥ 32 with gradient checkpointing |
| P4 | Learned encoders assumed to separate scales | No guarantee coarse = low-frequency; stride-hold then aliases fast content | Latents built on fixed nested spectral truncation `P^(l)`; scale separation by construction |
| P5 | Gate (3.4) + corrector (3.5) | Two overlapping reconciliation mechanisms → flat loss directions | Single mechanism (RNN update gate). Separate fusion gate removed |
| P6 | `h₀` unspecified | Zero-init memory exactly when chaotic error compounds worst | History **warmup** spins up `h^(l)`; horizon reported excluding warmup |
| P7 | Raw vector `h` concatenated into predictor | Ill-defined for a field backbone | Decoded priors injected at matching U-Net resolution stages; vector `h` injected via **FiLM** |
| P8 | "EnKF-style / Bayesian correction" framing | No observation → no innovation; overclaim | Reframed as **self-consistent latent refinement** (training-time regularizer) |
| P9 | `L_cons` with undefined `Down` and level-0 term | No canonical inter-level operator | `Down` = spectral truncation (nested, exact); level-0 term dropped |
| P10 | Posterior re-encode + fuse at inference | **Circular**: posterior is a deterministic transform of the prior → zero new information; gate trained on truth-derived posteriors is miscalibrated at inference and can self-reinforce drift | Posterior demoted to a **training-only** consistency loss; no inference-time re-encode, no fusion |

The single most important change is P10/P5: the inference-time posterior-fusion path is
deleted. Its legitimate job (keeping latents consistent with the decoded state) returns as a
training loss; its illusory job (correcting without observations) is dropped.

---

## 1. System and state

Forced 2D Kolmogorov flow, vorticity formulation, doubly-periodic `[0, 2π]²`:

\[
\partial_t \omega + (u\cdot\nabla)\omega = \nu \nabla^2 \omega + f_\omega,
\qquad f_\omega = -k_f \cos(k_f y) - \mu\,\omega,
\]

with `k_f = 4`, drag `μ = 0.1`, `Re = 4000`, on a `256×256` grid (2/3 dealiasing,
effective max mode ≈ 85). The emulator state is the scalar vorticity `ω_n ∈ R^{256×256}`;
velocity is recoverable via the streamfunction. The emulator timestep `Δt` is the snapshot
spacing (≈ `τ/15`, `τ` = large-eddy turnover time), not the solver substep.

---

## 2. Latent hierarchy by nested spectral truncation (fixes P4, P9)

Fixed radial low-pass spectral projectors:

\[
P^{(0)}: |k|\le 85 \ (\text{full field}), \qquad
P^{(1)}: |k|\le 16, \qquad
P^{(2)}: |k|\le 8 .
\]

These are **nested** (`P^(2) = P^(2)∘P^(1)`), so the inter-level down-operator `Down` is the
spectral truncation itself — exact and unambiguous (fixes P9). With `k_f = 4`, the coarse
band sits at and below forcing (large-scale / inverse-cascade dynamics), the fine band
carries the enstrophy range. "Coarse = lower-frequency" is true by construction, so the
multi-rate hold cannot alias fast content (fixes P4).

The three spectral bands map directly onto the U-Net's resolution stages (Sec 4): fine =
full-resolution input/output, medium band → the U-Net's intermediate stage, coarse band →
the deepest stage. The multiscale latent hierarchy and the U-Net resolution hierarchy are
the same hierarchy.

---

## 3. Recurrent latent dynamics (multi-rate, contractive)

**Resolved scope (was DESIGN §11):**
- The **fine scale is the raw vorticity field** — no recurrence on it. The U-Net handles the
  full field directly.
- Persistent recurrent state exists at **two** levels only: **medium** (`|k| ≤ 16`,
  stride `s₁ = 2`) and **coarse** (`|k| ≤ 8`, stride `s₂ = 4`).
- Recurrence is a **vector GRU on a compact latent** at each level (the low-`k` bands are
  genuinely low-dimensional; no ConvGRU needed).
- **Top-down coarse→medium is kept** (it carries the coarse level's forecast, not redundant
  band content).

Compact latent and decoder per recurrent level:
\[
z^{(l)}_n = E^{(l)}\!\big(P^{(l)}\omega_n\big), \qquad D^{(l)}: z^{(l)} \mapsto \text{band-limited field}.
\]

### 3.1 Multi-rate clock (fixes P2)

\[
h^{(l)}_{n+1} =
\begin{cases}
\mathrm{GRU}^{(l)}\!\big(h^{(l)}_n,\, c^{(l)}_n\big) & (n+1)\bmod s_l = 0 \\
h^{(l)}_n & \text{otherwise (held)}
\end{cases}
\qquad
z^{(l)}_{\text{prior},\,n+1} =
\begin{cases}
W^{(l)} h^{(l)}_{n+1} & \text{update step} \\
z^{(l)}_{\text{prior},\,n} & \text{held}
\end{cases}
\]

The predictor always receives a defined prior for every level — freshly emitted on update
steps, held otherwise.

### 3.2 Conditioning inputs

\[
c^{(l)}_n = \Big[\; E^{(l)}(P^{(l)}\omega_n)\ \text{(bottom-up)},\;
z^{(l+1)}_{\text{prior}}\ \text{(top-down, medium only)},\;
z^{(l)}_n - z^{(l)}_{n-s_l}\ \text{(stride-}l\text{ backward diff)} \;\Big]
\]

The bottom-up term is the only field→latent coupling: it conditions the recurrence on the
current (predicted, at inference) field. This is ordinary recurrent autoregressive coupling,
not the deleted circular re-encode-and-fuse.

### 3.3 Contraction safeguard (fixes P1)

Spectral normalization on each GRU recurrent weight; top-down/backward-diff gains bounded as
`α^(l) = α_max·σ(·)`, `α_max = 0.2`. This is the principled replacement for the deleted
additive corrector — it keeps the recurrence non-amplifying over long rollouts.

---

## 4. Predictor: modern hierarchical U-Net (fixes P7)

The backbone is a **modern 2D U-Net** (residual blocks, GroupNorm, SiLU, self-attention at
the bottleneck, skip connections), matching the original HINE architecture so that the
recurrence ablation is backbone-controlled.

\[
\hat{\omega}_{n+1} = \mathrm{UNet}\Big(\omega_n;\ \{\text{injected priors } D^{(l)} z^{(l)}_{\text{prior}}\},\ \mathrm{FiLM}(\{h^{(l)}\})\Big)
\]

**Latent injection (HINE mechanism, preserved).** Each decoded prior is resampled to its
stage resolution and injected into the matching encoder stage (concatenate or 1×1-conv add):
medium band → intermediate stage (≈ 32×32), coarse band → deepest stage (≈ 16×16). This is
the original HINE design of feeding latent context at intermediate encoder layers.

**FiLM conditioning.** Each recurrent hidden state `h^(l)` produces per-channel `(γ, β)` that
modulate the feature maps at its matching stage. This is the well-defined way to inject a
vector RNN state into a field backbone.

**Output.** Predict the increment `Δω` and add the input (`ω̂_{n+1} = ω_n + Δω`); this is
more stable for autoregressive rollout. An optional fixed high-`k` damping on the output
spectrum enforces dissipation (carried over from the 1D high-`k` penalty).

**No inference-time posterior re-encode and no fusion.** The latent state for the next step
updates only via Sec 3.

### 4.1 Relationship to the HINE baseline

The original HINE (no recurrence) uses the *same* U-Net but **regenerates** the hierarchical
latent-future ladder each step: it consumes current latent-future estimates at the encoder
stages and emits advanced latent futures at the matching decoder stages, at staggered
horizons (level `l` looks `l` steps ahead). MSR-HINE replaces that regenerate-each-step
behavior with **persistent multi-rate GRU evolution** of the medium/coarse latents, injected
at the same U-Net stages. The difference between the baseline and the full model is therefore
*exactly* "regenerated latents vs. persistent multi-rate recurrent latents," with everything
else held fixed.

### 4.2 Predictor–corrector reading (reframed, fixes P8)

The recurrent prior *predicts* the band-limited future; conditioning the next recurrence on
the realized field *corrects* the latent toward the trajectory actually taken. This is a
self-consistency loop, not a Bayesian/DA update — no observation, no innovation. A
DA-integrated variant (real observations entering `c^(l)`) is a future extension where the
"correction" language would become accurate.

---

## 5. Warmup and hidden-state initialization (fixes P6)

Before any free rollout (train and eval), spin up the hierarchy on a short observed history
window of `W = 12` frames: run encoders + GRUs on ground-truth `ω_n` (teacher-forced, no
loss), evolving each `h^(l)` on its stride clock. Free rollout begins after the window with
the spun-up states. Report the valid-prediction-horizon **from the end of warmup**.

---

## 6. Loss functions (stride-respecting) (fixes P3, P9, P10)

Let `U_l = {n : n mod s_l = 0}` be level `l`'s update steps.

**State / rollout:** `L_state = Σ_k γ^{k-1} ‖ω̂_{n+k} − ω_{n+k}‖²` over a `K`-step unroll.

**Spectral:** `L_spec = Σ_k |Ê(k) − E(k)|` (radial energy spectrum); `L_highk = Σ_{|k|>k_c} |ω̂(k)|²`.

**Latent prior (stride-respecting, fixes P3):**
\[
\mathcal{L}_{\text{prior}} = \sum_{l}\sum_{n\in U_l} \big\| z^{(l)}_{\text{prior},n} - E^{(l)}(P^{(l)}\omega_n) \big\|^2 .
\]

**Consistency (training only, fixes P10; nested `Down`, fixes P9):**
\[
\mathcal{L}_{\text{cons}} = \sum_{l\ge 1}\sum_{n\in U_l} \big\| E^{(l)}(P^{(l)}\hat\omega_n) - \mathrm{Down}^{(l)}\!\big(E^{(l-1)}(P^{(l-1)}\hat\omega_n)\big) \big\|^2 .
\]

**Total:** `L = L_state + λ_prior L_prior + λ_cons L_cons + λ_spec L_spec + λ_hk L_highk`.

No loss is computed against a ground-truth-derived posterior fed into the latent state;
ground truth enters only as targets.

---

## 7. Training

- TBPTT window **≥ 32** (so the stride-4 coarse level gets ≥ 8 updates); **gradient
  checkpointing** per unrolled step (memory-dominant cost at 256²).
- **Scheduled sampling on the field only** (teacher-forcing → self-feeding). No posterior
  channel exists to schedule, so there is no hidden leak.
- Optional Phase 0: pretrain `E^(l)`/`D^(l)` as band-limited autoencoders before coupling.
- Optional curriculum: higher viscosity / lower Re first, anneal to Re=4000.
- **Reduced data regime (50 train / 10 test trajectories):** lengthen per-trajectory rollouts
  (train ≈ 500 steps, test ≈ 1500 steps). Watch attractor coverage and overfitting: small
  encoders, lean on spectral/consistency regularizers, trajectory-level splits.

---

## 8. Models and ablations

Two roles, kept distinct:

**Explicit autoregressive reference baseline (cross-dataset):**
- `fno_1step` — one-step FNO `ω_{t+1} = F(ω_t)`. This is the explicit operator baseline used
  consistently across all three datasets (KS, L96, Kolmogorov). It is a *reference point*,
  not the recurrence ablation, and intentionally uses a different backbone.

**Backbone-controlled recurrence ablation (all U-Net):**
- `hine` — original HINE, U-Net backbone, latent-future ladder regenerated each step, no
  persistent recurrence (Sec 4.1). This is the no-recurrence ablation.
- `msr_hine` — full model: U-Net + multi-rate recurrent hierarchy.

**Internal ablations of `msr_hine`** (all U-Net):
- `single_scale` — one recurrent scale only.
- `no_multirate` — all strides = 1.
- `no_topdown` — remove coarse→medium top-down.
- `no_consistency` — remove `L_cons`.
- `no_contraction` — spectral norm off, gains unbounded.
- `no_warmup` — zero-init recurrent states.

Plus a confirmation that the **removed** inference-time fusion is not missed: `msr_hine`
horizon/spectra should be no worse than a variant that re-adds inference fusion (the
empirical statement of the circularity argument).

---

## 9. Metrics

- Rollout RMSE and anomaly correlation vs. lead time.
- **Valid prediction horizon in Lyapunov-time units** `τ_λ` (estimate `τ_λ` via a
  tangent-linear / finite-difference perturbation run), measured from end of warmup.
- Radial energy spectrum `E(k)` at long lead times; spectral / enstrophy drift.

---

## 10. Recommended hyperparameters (2D Kolmogorov, 256², Re = 4000)

**U-Net backbone**

| Parameter | Value |
|---|---|
| Stages (resolutions) | 256 → 128 → 64 → 32 → 16 |
| Base channels / mults | 64 / [1, 2, 2, 4, 4] |
| Residual blocks per stage | 2 |
| Norm / activation | GroupNorm / SiLU |
| Self-attention | at 16×16 (bottleneck), optionally 32×32 |
| Output | increment `Δω`, residual add |
| Latent injection | medium → 32×32 stage, coarse → 16×16 stage |

**Recurrent hierarchy**

| Parameter | Medium | Coarse |
|---|---|---|
| Spectral band `\|k\|` | ≤ 16 | ≤ 8 |
| Stride `s_l` | 2 | 4 |
| Latent dim | 128 | 64 |
| GRU hidden dim | 128 | 64 |

**Training / rollout**

| Parameter | Value |
|---|---|
| TBPTT window | 32 |
| Warmup `W` | 12 |
| Rollout loss `K` | 16, annealed up |
| `α_max` (gain bound) | 0.2 |
| Spectral norm | on all recurrent weights |
| Snapshot `Δt` | ≈ τ/15 |

**FNO reference baseline**

| Parameter | Value |
|---|---|
| Layers | 4 |
| Retained modes | 32–48 |
| Width | 64 |

---

## 11. Resolved design decisions (formerly open)

1. **Fine level = raw field; recurrence on medium + coarse only.** Compressing the fine band
   would bottleneck the small-scale structure that drives spectral fidelity, and there is no
   slow memory to exploit at stride 1.
2. **Vector GRU on compact latents** for both recurrent levels; the low-`k` bands are
   low-dimensional.
3. **Top-down coarse→medium kept** — it carries forecast information, not redundant band
   content.
4. **Backbone = modern U-Net**, shared across `hine` and `msr_hine` so the recurrence
   ablation is backbone-controlled. The FNO is retained separately as the cross-dataset
   explicit autoregressive reference baseline.
