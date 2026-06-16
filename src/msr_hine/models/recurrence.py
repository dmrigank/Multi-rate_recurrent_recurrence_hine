"""Multi-rate contractive recurrent latent dynamics (DESIGN.md §3).

Implements the two-level GRU hierarchy for MSR-HINE:

  Medium level: stride s_1 = 2, latent/GRU dim = 128
  Coarse  level: stride s_2 = 4, latent/GRU dim = 64

Invariants enforced here
────────────────────────
Invariant 4  (multi-rate hold is REAL): off-stride steps copy h and z_prior
             unchanged; the GRU cell is never called; no gradient flows
             through the held level on that step.
Invariant 6  (contraction safeguard on by default): spectral norm on every
             GRU recurrent weight; top-down and backward-diff gains bounded
             α = α_max·σ(raw_α), α_max = 0.2.

Conditioning vector (DESIGN.md §3.2):
  c^l_n = [ E^l(P^l ω_n)              (bottom-up, plain concat)
           ; α_td  · z^{l+1}_prior    (top-down, medium only, bounded gain)
           ; α_bd  · (z^l_n − z^l_{n−s_l})  (backward diff, bounded gain) ]

The bottom-up encoding is computed OUTSIDE this module (in MSRHINE.step) and
passed in as z_current.  This keeps encoder/recurrence concerns separate and
makes the Invariant-1 boundary explicit.

Multi-rate clock (DESIGN.md §3.1):
  (n+1) mod s_l == 0  →  evolve GRU, emit new z_prior
  otherwise           →  hold h, hold z_prior
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Spectral-norm GRU cell
# ---------------------------------------------------------------------------

class ContractiveGRUCell(nn.Module):
    """GRU cell with spectral normalisation on the three recurrent weight matrices.

    Implements the standard GRU equations manually so we can apply
    spectral_norm selectively to the recurrent (h→h) weights only.

    r  = σ( W_ir x + b_ir + spectral_norm(W_hr) h + b_hr )
    z  = σ( W_iz x + b_iz + spectral_norm(W_hz) h + b_hz )
    n  = tanh( W_in x + b_in + r ⊙ (spectral_norm(W_hn) h + b_hn) )
    h' = (1−z) ⊙ n + z ⊙ h

    Spectral norm bounds σ_max(W_hh) ≤ 1, which (together with the sigmoid
    gate) keeps the recurrent Jacobian bounded.  This replaces the deleted
    additive corrector (fixes P1, Invariant 6).

    Args:
        input_dim:  Dimensionality of the conditioning input c^l.
        hidden_dim: Dimensionality of the hidden state h^l.
        use_spectral_norm: If False, skip spectral norm (no_contraction ablation).
    """

    def __init__(
        self,
        input_dim:          int,
        hidden_dim:         int,
        use_spectral_norm:  bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input-to-hidden projections (plain)
        self.W_i = nn.Linear(input_dim,  3 * hidden_dim, bias=True)

        # Recurrent projections — spectral-normed or plain
        def _make_Wh(d):
            layer = nn.Linear(d, d, bias=True)
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        self.Wh_r = _make_Wh(hidden_dim)   # reset gate recurrent
        self.Wh_z = _make_Wh(hidden_dim)   # update gate recurrent
        self.Wh_n = _make_Wh(hidden_dim)   # candidate recurrent

    def forward(self, c: Tensor, h: Tensor) -> Tensor:
        """One GRU step.

        Args:
            c: Conditioning vector [B, input_dim].
            h: Previous hidden state [B, hidden_dim].

        Returns:
            h_new [B, hidden_dim].
        """
        # Split the input projection
        gates_i = self.W_i(c)                                     # [B, 3*d]
        ir, iz, _in = gates_i.chunk(3, dim=-1)

        r = torch.sigmoid(ir + self.Wh_r(h))
        z = torch.sigmoid(iz + self.Wh_z(h))
        n = torch.tanh(_in + r * self.Wh_n(h))
        return (1.0 - z) * n + z * h


# ---------------------------------------------------------------------------
# Single recurrent level
# ---------------------------------------------------------------------------

class RecurrentLevel(nn.Module):
    """One level of the multi-rate recurrent hierarchy.

    Holds:
      • ContractiveGRUCell for computing h^l
      • Linear prior-emission head  W^l: h^l → z^l_prior
      • Learned bounded-gain scalars for top-down and backward-diff terms

    Conditioning assembly (DESIGN.md §3.2):
      c^l = [ z_current                          (bottom-up, plain)
            | α_td·z_coarse_prior                (top-down, medium only)
            | α_bd·(z_current − z_current_prev)  (stride-l backward diff) ]

    Args:
        latent_dim:  Latent dimensionality for this level (= GRU hidden dim).
        stride:      Update stride s_l (2 = medium, 4 = coarse).
        has_topdown: True for the medium level; False for coarse.
        coarse_dim:  Dimensionality of the coarse prior (only used if has_topdown).
        alpha_max:   Maximum gain for top-down and backward-diff terms (0.2).
        use_spectral_norm: Passed through to ContractiveGRUCell.
    """

    def __init__(
        self,
        latent_dim:         int,
        stride:             int,
        has_topdown:        bool  = False,
        coarse_dim:         int   = 0,
        alpha_max:          float = 0.2,
        use_spectral_norm:  bool  = True,
    ) -> None:
        super().__init__()
        self.latent_dim   = latent_dim
        self.stride       = stride
        self.has_topdown  = has_topdown
        self.alpha_max    = alpha_max

        # Conditioning input dimensionality
        # bottom-up: latent_dim
        # top-down:  coarse_dim  (only medium)
        # backward:  latent_dim
        td_dim = coarse_dim if has_topdown else 0
        self._input_dim = latent_dim + td_dim + latent_dim

        self.gru = ContractiveGRUCell(
            input_dim  = self._input_dim,
            hidden_dim = latent_dim,
            use_spectral_norm = use_spectral_norm,
        )

        # Prior emission head: h → z_prior
        self.prior_head = nn.Linear(latent_dim, latent_dim)

        # Learnable unconstrained scalars for bounded gains
        # α = alpha_max * sigmoid(raw_α)
        self.raw_alpha_bd = nn.Parameter(torch.zeros(1))   # backward-diff gain
        if has_topdown:
            self.raw_alpha_td = nn.Parameter(torch.zeros(1))  # top-down gain

    # ── Bounded gain accessors ─────────────────────────────────────────────

    def _alpha_bd(self) -> Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha_bd)

    def _alpha_td(self) -> Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha_td)

    # ── Conditioning assembly ──────────────────────────────────────────────

    def build_conditioning(
        self,
        z_current:       Tensor,
        z_prev:          Tensor,
        z_coarse_prior:  Tensor | None,
    ) -> Tensor:
        """Assemble c^l (DESIGN.md §3.2).

        Args:
            z_current:      Bottom-up encoding of the current (predicted) field
                            [B, latent_dim].  Invariant 1: this is E^l(P^l ω̂_n),
                            never E^l(P^l ω_n_ground_truth).
            z_prev:         Bottom-up encoding from s_l steps ago [B, latent_dim].
            z_coarse_prior: Coarse-level emitted prior [B, coarse_dim], or None.

        Returns:
            Conditioning vector [B, input_dim].
        """
        parts = [z_current]

        if self.has_topdown:
            assert z_coarse_prior is not None, "Medium level requires coarse prior"
            parts.append(self._alpha_td() * z_coarse_prior)

        # Stride-l backward difference (bounded gain)
        backward_diff = self._alpha_bd() * (z_current - z_prev)
        parts.append(backward_diff)

        return torch.cat(parts, dim=-1)

    # ── GRU step ───────────────────────────────────────────────────────────

    def gru_step(self, c: Tensor, h: Tensor) -> Tensor:
        """Advance the GRU by one update step (called only on update steps)."""
        return self.gru(c, h)

    def emit_prior(self, h: Tensor) -> Tensor:
        """Map hidden state to emitted prior z^l_prior = W^l h^l."""
        return self.prior_head(h)


# ---------------------------------------------------------------------------
# Multi-rate hierarchy
# ---------------------------------------------------------------------------

class MultiRateHierarchy(nn.Module):
    """Two-level multi-rate recurrent hierarchy (medium + coarse).

    Enforces Invariant 4: off-stride levels are held exactly — the GRU is
    never called for them and their prior is returned unchanged.

    Args:
        medium_dim:        Latent/GRU dim for medium level.
        coarse_dim:        Latent/GRU dim for coarse level.
        medium_stride:     Update stride s_1 (default 2).
        coarse_stride:     Update stride s_2 (default 4).
        alpha_max:         Gain bound for contraction safeguard.
        use_spectral_norm: If False, disables spectral norm (no_contraction ablation).
    """

    def __init__(
        self,
        medium_dim:         int   = 128,
        coarse_dim:         int   = 64,
        medium_stride:      int   = 2,
        coarse_stride:      int   = 4,
        alpha_max:          float = 0.2,
        use_spectral_norm:  bool  = True,
        use_topdown:        bool  = True,   # no_topdown ablation: set False
    ) -> None:
        super().__init__()
        self.medium_stride = medium_stride
        self.coarse_stride = coarse_stride
        self.use_topdown   = use_topdown

        # single_scale: no top-down possible (no coarse level exists)
        _effective_topdown = use_topdown and (coarse_dim > 0)

        self.medium_level = RecurrentLevel(
            latent_dim        = medium_dim,
            stride            = medium_stride,
            has_topdown       = _effective_topdown,
            coarse_dim        = coarse_dim if _effective_topdown else 0,
            alpha_max         = alpha_max,
            use_spectral_norm = use_spectral_norm,
        )
        # single_scale: coarse_dim=0 means no real coarse GRU
        self._has_coarse = coarse_dim > 0
        if self._has_coarse:
            self.coarse_level = RecurrentLevel(
                latent_dim        = coarse_dim,
                stride            = coarse_stride,
                has_topdown       = False,
                coarse_dim        = 0,
                alpha_max         = alpha_max,
                use_spectral_norm = use_spectral_norm,
            )
        else:
            self.coarse_level = None  # type: ignore[assignment]

    def step(
        self,
        step_n:          int,
        z_medium:        Tensor,
        z_medium_prev:   Tensor,
        z_coarse:        Tensor,
        z_coarse_prev:   Tensor,
        h_medium:        Tensor,
        h_coarse:        Tensor,
        z_medium_prior:  Tensor,
        z_coarse_prior:  Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance the hierarchy by one step, obeying the multi-rate clock.

        Invariant 4: levels whose stride does NOT divide (step_n + 1) are
        held — h and z_prior returned unchanged, GRU NOT called.

        Args:
            step_n:          0-based step index within the rollout.
            z_medium:        Current medium encoding of ω̂ [B, medium_dim].
            z_medium_prev:   Medium encoding from s_1 steps ago [B, medium_dim].
            z_coarse:        Current coarse encoding of ω̂ [B, coarse_dim].
            z_coarse_prev:   Coarse encoding from s_2 steps ago [B, coarse_dim].
            h_medium:        Current medium hidden state [B, medium_dim].
            h_coarse:        Current coarse hidden state [B, coarse_dim].
            z_medium_prior:  Held medium prior [B, medium_dim].
            z_coarse_prior:  Held coarse prior [B, coarse_dim].

        Returns:
            (h_medium_new, h_coarse_new, z_medium_prior_new, z_coarse_prior_new).
        """
        next_step = step_n + 1   # 1-based step number for modulo check

        # ── Coarse level (skipped for single_scale ablation) ────────────────
        if self._has_coarse and next_step % self.coarse_stride == 0:
            c_coarse = self.coarse_level.build_conditioning(
                z_current      = z_coarse,
                z_prev         = z_coarse_prev,
                z_coarse_prior = None,
            )
            h_coarse_new      = self.coarse_level.gru_step(c_coarse, h_coarse)
            z_coarse_prior_new = self.coarse_level.emit_prior(h_coarse_new)
        else:
            # Invariant 4: HOLD — no GRU call, no gradient through the level
            # (also reached when _has_coarse=False for single_scale ablation)
            h_coarse_new      = h_coarse
            z_coarse_prior_new = z_coarse_prior

        # ── Medium level (top-down uses the JUST-UPDATED coarse prior) ──────
        # no_topdown ablation: pass None when use_topdown=False.
        if next_step % self.medium_stride == 0:
            c_medium = self.medium_level.build_conditioning(
                z_current      = z_medium,
                z_prev         = z_medium_prev,
                z_coarse_prior = z_coarse_prior_new if self.use_topdown else None,
            )
            h_medium_new      = self.medium_level.gru_step(c_medium, h_medium)
            z_medium_prior_new = self.medium_level.emit_prior(h_medium_new)
        else:
            # Invariant 4: HOLD
            h_medium_new      = h_medium
            z_medium_prior_new = z_medium_prior

        return h_medium_new, h_coarse_new, z_medium_prior_new, z_coarse_prior_new
