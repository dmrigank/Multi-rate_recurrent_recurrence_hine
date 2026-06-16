"""Pseudo-spectral solver for 2D forced Kolmogorov flow.

Vorticity formulation on doubly-periodic [0, 2π]², float64 internally.
Spatial discretisation: rfft2 / irfft2 on an n×n grid.
Time integration: ETDRK4 (Cox–Matthews) with exact integrating factor for the
linear stiff part  L(k) = −ν|k|² − μ.
Dealiasing: 2/3 rule – modes with max(|kx|,|ky|) > n//3 are zeroed.

Public API (all tensors on the caller-supplied device, float64 unless noted):
    wavenumbers(n, device)
    dealias_mask(n, device)
    make_kolmogorov_forcing(n, k_f, mu, device)
    vorticity_to_velocity(omega_hat, kx2d, ky2d)
    energy(omega_hat, n)
    enstrophy(omega_hat, n)
    radial_energy_spectrum(omega_hat, n, n_bins)
    nonlinear_rhs(omega_hat, kx2d, ky2d, ksq, forcing_hat, dealias)
    etdrk4_step(omega_hat, dt, E, E2, f1, f2, f3, kx2d, ky2d, ksq, forcing_hat, dealias)
    make_etdrk4_coeffs(L, dt)
    integrate(omega0, n_steps, dt, substeps_per_snapshot, kx2d, ky2d, ksq,
              forcing_hat, dealias, nu, mu)

Convenience entry point:
    KolmogorovSolver — wraps all of the above with sensible defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Wavenumber grids and dealiasing
# ---------------------------------------------------------------------------

def wavenumbers(n: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Return (kx2d, ky2d, ksq) for an n×n grid with rfft2 layout.

    rfft2 output shape: [n, n//2+1].
    kx indexes rows (the "x" direction, full-range after fftshift convention).
    ky indexes columns (the half-spectrum 0..n//2).

    Returns:
        kx2d: [n, n//2+1] integer wavenumbers along x (rows).
        ky2d: [n, n//2+1] integer wavenumbers along y (cols).
        ksq:  [n, n//2+1] |k|² = kx²+ky².
    """
    # ky: 0, 1, ..., n//2  (rfft half-spectrum)
    ky = torch.arange(n // 2 + 1, dtype=torch.float64, device=device)
    # kx: 0, 1, ..., n//2, -(n//2-1), ..., -1  (full fftfreq convention)
    kx_pos = torch.arange(n // 2 + 1, dtype=torch.float64, device=device)
    kx_neg = torch.arange(-(n // 2 - 1), 0, dtype=torch.float64, device=device)
    kx = torch.cat([kx_pos, kx_neg])  # length n
    kx2d = kx.unsqueeze(1).expand(n, n // 2 + 1)
    ky2d = ky.unsqueeze(0).expand(n, n // 2 + 1)
    ksq = kx2d ** 2 + ky2d ** 2
    return kx2d, ky2d, ksq


def dealias_mask(n: int, device: torch.device) -> Tensor:
    """Boolean 2/3-dealiasing mask, shape [n, n//2+1].

    Implements the radial cutoff  |k| ≤ n//3  (DESIGN.md §2: P^0 : |k| ≤ 85 at n=256).
    The +0.5 tolerance ensures the boundary wavenumber is always retained.

    Using radial (L2) rather than L∞ (square) criterion is important so that
    the solver's fine-band definition is consistent with the spectral truncation
    projectors in spectral/truncation.py, which also use radial masks.
    """
    _, _, ksq = wavenumbers(n, device)
    k_cut = n // 3
    return ksq.sqrt() <= k_cut + 0.5


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------

def make_kolmogorov_forcing(
    n: int,
    k_f: int,
    device: torch.device,
) -> Tensor:
    """Return the spectral representation of the deterministic Kolmogorov body force.

    Physical forcing (y-direction only):  f_body(x,y) = −k_f cos(k_f y)
    The drag term  −μω  is linear in ω and handled inside the RHS / ETDRK4
    linear operator; it is NOT included here.

    The forcing is purely real and its only non-zero spectral components are at
    ky = k_f, kx = 0  with amplitude  −k_f / 2  (and the conjugate).

    Returns:
        forcing_hat: Complex64, shape [n, n//2+1].
    """
    forcing_hat = torch.zeros(n, n // 2 + 1, dtype=torch.complex128, device=device)
    if k_f <= n // 2:
        # f(y) = −k_f cos(k_f y).
        # torch.fft.rfft2 is unnormalised: rfft2(cos(k_f y))[0, k_f] = n²/2.
        # So to inject f = −k_f cos(k_f y) into the RHS (which lives in rfft2 space),
        # the coefficient must be −k_f * n²/2.
        forcing_hat[0, k_f] = complex(-k_f * (n * n) / 2, 0.0)
    return forcing_hat


# ---------------------------------------------------------------------------
# Velocity from vorticity via streamfunction
# ---------------------------------------------------------------------------

def vorticity_to_velocity(
    omega_hat: Tensor,
    kx2d: Tensor,
    ky2d: Tensor,
    ksq: Tensor,
) -> tuple[Tensor, Tensor]:
    """Recover (u_hat, v_hat) from the vorticity spectrum.

    Vorticity: ω = ∂v/∂x − ∂u/∂y.
    Streamfunction u = ∂ψ/∂y, v = −∂ψ/∂x  →  ω = −∇²ψ.
    In spectral space (integer wavenumbers on [0,2π]²):
        ∇² ↔ −|k|²   →   −|k|² ψ̂ = −ω̂   →   ψ̂ = ω̂ / |k|²
        û  =  i ky ψ̂  =  i ky ω̂ / |k|²
        v̂  = −i kx ψ̂  = −i kx ω̂ / |k|²

    Accepts a leading batch dimension: omega_hat [..., n, n//2+1].
    """
    ksq_safe = ksq.clone()
    ksq_safe[0, 0] = 1.0

    psi_hat = omega_hat / ksq_safe      # ψ̂ = +ω̂/|k|²  (not negative)
    psi_hat[..., 0, 0] = 0.0           # zero mean mode

    u_hat =  1j * ky2d * psi_hat
    v_hat = -1j * kx2d * psi_hat
    return u_hat, v_hat


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _parseval_factor(n: int) -> float:
    """Parseval normalisation for unnormalised rfft2.

    torch.fft.rfft2 is unnormalised: û[k] = Σ_j u[j] exp(-2πikj/n).
    By Parseval:  Σ_j |u[j]|² = (1/n²) Σ_k |û[k]|²
    Equivalently: mean |u|² = (1/n²) Σ_k |û[k]|² / n² = Σ_k |û[k]|² / n^4
    """
    return 1.0 / (n * n) ** 2


def energy(omega_hat: Tensor, ksq: Tensor, n: int) -> Tensor:
    """Total kinetic energy E = 0.5 * <|u|²+|v|²> = 0.5 * Σ' |ω̂|²/|k|² / n⁴.

    Accepts batch: omega_hat [..., n, n//2+1].
    Returns scalar per batch item [...].
    """
    ksq_safe = ksq.clone()
    ksq_safe[0, 0] = 1.0
    # |ω̂|²/|k|²
    ratio = omega_hat.abs() ** 2 / ksq_safe
    ratio[..., 0, 0] = 0.0  # zero-mode carries no energy

    # rfft2 half-spectrum: double-count interior ky modes (1..n//2-1)
    w = torch.ones_like(ratio, dtype=torch.float64)
    w[..., 1 : n // 2] *= 2.0  # ky interior

    e = 0.5 * (ratio * w).sum(dim=(-2, -1)) * _parseval_factor(n)
    return e.real


def enstrophy(omega_hat: Tensor, n: int) -> Tensor:
    """Total enstrophy Z = 0.5 * <|ω|²> = 0.5 * sum |ω̂|² / n².

    Accepts batch: omega_hat [..., n, n//2+1].
    Returns scalar per batch item [...].
    """
    w = torch.ones(n, n // 2 + 1, dtype=torch.float64, device=omega_hat.device)
    w[:, 1 : n // 2] *= 2.0

    z = 0.5 * (omega_hat.abs() ** 2 * w).sum(dim=(-2, -1)) * _parseval_factor(n)
    return z.real


def radial_energy_spectrum(
    omega_hat: Tensor,
    ksq: Tensor,
    n: int,
    n_bins: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Radial (shell-averaged) kinetic energy spectrum E(k).

    Bins |k| = round(sqrt(kx²+ky²)) into integer wavenumber shells.

    Args:
        omega_hat: [..., n, n//2+1] complex, batch supported.
        ksq:       [n, n//2+1] float64 wavenumber-squared grid.
        n:         grid size.
        n_bins:    number of bins (default n//2).

    Returns:
        (k_bins [n_bins], E_k [..., n_bins]) both float64.
    """
    if n_bins is None:
        n_bins = n // 2

    k_mag = ksq.sqrt().round().long()  # integer shell index [n, n//2+1]
    k_bins = torch.arange(n_bins, dtype=torch.float64, device=omega_hat.device)

    ksq_safe = ksq.clone()
    ksq_safe[0, 0] = 1.0
    e_density = (omega_hat.abs() ** 2 / ksq_safe)
    e_density[..., 0, 0] = 0.0

    # rfft half-spectrum weight
    w = torch.ones(n, n // 2 + 1, dtype=torch.float64, device=omega_hat.device)
    w[:, 1 : n // 2] *= 2.0
    e_density = e_density * w * _parseval_factor(n) * 0.5

    # flatten spatial dims for scatter_add
    batch_shape = omega_hat.shape[:-2]
    e_flat = e_density.reshape(*batch_shape, -1)
    k_flat = k_mag.reshape(-1)

    E_k = torch.zeros(*batch_shape, n_bins, dtype=torch.float64, device=omega_hat.device)
    # mask out-of-range bins
    valid = (k_flat >= 0) & (k_flat < n_bins)
    E_k.scatter_add_(-1, k_flat[valid].expand(*batch_shape, valid.sum()),
                     e_flat[..., valid])
    return k_bins, E_k


# ---------------------------------------------------------------------------
# Nonlinear RHS (advection + deterministic forcing, NO viscosity/drag)
# ---------------------------------------------------------------------------

def nonlinear_rhs(
    omega_hat: Tensor,
    kx2d: Tensor,
    ky2d: Tensor,
    ksq: Tensor,
    forcing_hat: Tensor,
    dealias: Tensor,
) -> Tensor:
    """Nonlinear part of dω/dt: advection + body forcing, dealiased.

    N(ω) = −(u·∇)ω + f_body
         = −u ∂ω/∂x − v ∂ω/∂y + f_body

    Computed pseudo-spectrally: multiply in physical space, dealiase result.
    Viscosity and drag are NOT included here (handled by the integrating factor).

    Args:
        omega_hat: [..., n, n//2+1] complex128.
        kx2d, ky2d, ksq: wavenumber grids [n, n//2+1].
        forcing_hat: [n, n//2+1] complex128.
        dealias: [n, n//2+1] bool mask.

    Returns:
        N_hat: [..., n, n//2+1] complex128.
    """
    n = omega_hat.shape[-2]

    u_hat, v_hat = vorticity_to_velocity(omega_hat, kx2d, ky2d, ksq)

    # spectral derivatives of omega
    domega_dx_hat = 1j * kx2d * omega_hat
    domega_dy_hat = 1j * ky2d * omega_hat

    # to physical space (dealiased inputs)
    def to_phys(fhat: Tensor) -> Tensor:
        return torch.fft.irfft2(fhat * dealias, s=(n, n))

    u = to_phys(u_hat)
    v = to_phys(v_hat)
    domega_dx = to_phys(domega_dx_hat)
    domega_dy = to_phys(domega_dy_hat)

    # nonlinear product in physical space, back to spectral
    adv = u * domega_dx + v * domega_dy
    adv_hat = torch.fft.rfft2(adv)

    N_hat = -adv_hat + forcing_hat
    # dealias result
    N_hat = N_hat * dealias
    return N_hat


# ---------------------------------------------------------------------------
# ETDRK4 coefficient precomputation
# ---------------------------------------------------------------------------

def make_etdrk4_coeffs(
    L: Tensor,
    dt: float,
    n_contour: int = 32,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Precompute phi-function ETDRK4 scalar coefficients.

    Uses the Kassam–Trefethen (2005) contour-integral method to evaluate
    the phi functions stably at all magnitudes of L·dt (including near zero).

    The phi functions are:
        phi1(c) = (e^c - 1) / c
        phi2(c) = (e^c - 1 - c) / c²
        phi3(c) = (e^c - 1 - c - c²/2) / c³

    The ETDRK4 scheme (Cox–Matthews / Hochbruck–Ostermann):
        a   = E2·ω  + f1·N(ω)
        b   = E2·ω  + f1·N(a)
        c   = E2·a  + f1·(2·N(b) − N(ω))
        ω'  = E·ω   + fa·N(ω) + fb·(N(a)+N(b)) + fc·N(c)

    where:
        f1 = (dt/2) · phi1(Ldt/2)
        fa = dt · (phi1(Ldt) − 3·phi2(Ldt) + 4·phi3(Ldt))
        fb = dt · 2·(phi2(Ldt) − 2·phi3(Ldt))
        fc = dt · (−phi2(Ldt) + 4·phi3(Ldt))

    L:  [n, n//2+1] float64 — linear operator eigenvalue per mode.
    dt: time step (scalar float).

    Returns (E, E2, f1, fa, fb, fc) each [n, n//2+1] complex128,
    with fa, fb, fc already absorbing the dt factor.
    """
    Ldt  = (L * dt).to(torch.complex128)
    Ldt2 = Ldt / 2.0

    E  = torch.exp(Ldt)
    E2 = torch.exp(Ldt2)

    # Kassam–Trefethen contour: radius r centred at the evaluation point.
    # The mean of a function over a circle = function at centre (mean-value property
    # for analytic functions), so this evaluates phi_k(Ldt) stably even when Ldt ≈ 0
    # (the removable singularities of phi_k at 0 do not cause cancellation on the contour).
    r = 1.0
    theta = torch.linspace(0, 2 * math.pi, n_contour + 1, dtype=torch.float64,
                           device=L.device)[:-1]
    zc = r * torch.exp(1j * theta)   # [M] contour points

    # Contours centred at Ldt and Ldt/2
    z  = Ldt .unsqueeze(-1) + zc    # [n, n//2+1, M]
    z2 = Ldt2.unsqueeze(-1) + zc

    ez  = torch.exp(z)
    ez2 = torch.exp(z2)

    # phi1(z) = (e^z - 1)/z
    # phi2(z) = (e^z - 1 - z)/z^2
    # phi3(z) = (e^z - 1 - z - z^2/2)/z^3
    phi1_full  = ((ez  - 1.0) / z ).mean(dim=-1)   # phi1 at Ldt
    phi1_half  = ((ez2 - 1.0) / z2).mean(dim=-1)   # phi1 at Ldt/2
    phi2_full  = ((ez  - 1.0 - z ) / z **2).mean(dim=-1)
    phi3_full  = ((ez  - 1.0 - z  - z **2 / 2.0) / z **3).mean(dim=-1)

    # Half-step: E2·ω + f1·N advances by dt/2
    f1 = (dt / 2.0) * phi1_half

    # Full-step weights (absorbing dt):
    fa = dt * (phi1_full - 3.0 * phi2_full + 4.0 * phi3_full)
    fb = dt * 2.0 * (phi2_full - 2.0 * phi3_full)
    fc = dt * (-phi2_full + 4.0 * phi3_full)

    return E, E2, f1, fa, fb, fc


# ---------------------------------------------------------------------------
# ETDRK4 step
# ---------------------------------------------------------------------------

def etdrk4_step(
    omega_hat: Tensor,
    E: Tensor,
    E2: Tensor,
    f1: Tensor,
    fa: Tensor,
    fb: Tensor,
    fc: Tensor,
    kx2d: Tensor,
    ky2d: Tensor,
    ksq: Tensor,
    forcing_hat: Tensor,
    dealias: Tensor,
) -> Tensor:
    """Advance omega_hat by one step using ETDRK4 (Cox–Matthews / Kassam–Trefethen).

    Linear part L = −ν|k|² − μ is handled exactly via precomputed exponentials.
    Nonlinear part N = advection + body forcing, evaluated pseudo-spectrally.

    Scheme:
        N1 = N(ω̂)
        a  = E2·ω̂ + f1·N1
        N2 = N(a)
        b  = E2·ω̂ + f1·N2
        N3 = N(b)
        c  = E2·a  + f1·(2·N3 − N1)
        N4 = N(c)
        ω̂' = E·ω̂ + fa·N1 + fb·(N2+N3) + fc·N4

    fa, fb, fc already absorb the dt factor (from make_etdrk4_coeffs).

    Args:
        omega_hat: [..., n, n//2+1] complex128.
        E, E2: exp(L dt) and exp(L dt/2) [n, n//2+1].
        f1: half-step phi1 weight [n, n//2+1].
        fa, fb, fc: full-step weights [n, n//2+1].
        kx2d, ky2d, ksq: wavenumber arrays.
        forcing_hat: body forcing spectrum.
        dealias: dealiasing mask.

    Returns:
        omega_hat_new: [..., n, n//2+1] complex128.
    """
    N1 = nonlinear_rhs(omega_hat, kx2d, ky2d, ksq, forcing_hat, dealias)

    a  = E2 * omega_hat + f1 * N1
    N2 = nonlinear_rhs(a, kx2d, ky2d, ksq, forcing_hat, dealias)

    b  = E2 * omega_hat + f1 * N2
    N3 = nonlinear_rhs(b, kx2d, ky2d, ksq, forcing_hat, dealias)

    c  = E2 * a + f1 * (2.0 * N3 - N1)
    N4 = nonlinear_rhs(c, kx2d, ky2d, ksq, forcing_hat, dealias)

    omega_new = E * omega_hat + fa * N1 + fb * (N2 + N3) + fc * N4
    omega_new = omega_new * dealias
    return omega_new


# ---------------------------------------------------------------------------
# Main integration loop
# ---------------------------------------------------------------------------

def integrate(
    omega0: Tensor,
    n_steps: int,
    dt: float,
    substeps_per_snapshot: int,
    kx2d: Tensor,
    ky2d: Tensor,
    ksq: Tensor,
    forcing_hat: Tensor,
    dealias: Tensor,
    nu: float,
    mu: float,
) -> Tensor:
    """Integrate and return saved vorticity snapshots.

    Args:
        omega0: Initial vorticity [B, n, n] real float64 (batch).
        n_steps: Number of snapshots to save.
        dt: Substep size.
        substeps_per_snapshot: How many substeps between saved frames.
        kx2d, ky2d, ksq: Wavenumber grids [n, n//2+1].
        forcing_hat: Body forcing [n, n//2+1] complex128.
        dealias: Dealiasing mask [n, n//2+1] bool.
        nu: Kinematic viscosity.
        mu: Linear drag.

    Returns:
        snapshots: [B, n_steps, n, n] float32 (cast before return to save memory).
    """
    n = omega0.shape[-1]
    device = omega0.device

    # linear operator eigenvalue per mode: L(k) = −ν|k|² − μ
    L = (-nu * ksq - mu).to(torch.float64)

    E, E2, f1, fa, fb, fc = make_etdrk4_coeffs(L, dt)

    # to spectral
    omega_hat = torch.fft.rfft2(omega0.to(torch.float64))
    omega_hat = omega_hat * dealias  # ensure initial condition is dealiased

    snapshots = []
    for _ in range(n_steps):
        for _ in range(substeps_per_snapshot):
            omega_hat = etdrk4_step(
                omega_hat, E, E2, f1, fa, fb, fc,
                kx2d, ky2d, ksq, forcing_hat, dealias,
            )
        snap = torch.fft.irfft2(omega_hat, s=(n, n)).to(torch.float32)
        snapshots.append(snap)

    return torch.stack(snapshots, dim=1)  # [B, n_steps, n, n]


# ---------------------------------------------------------------------------
# Convenience solver class
# ---------------------------------------------------------------------------

@dataclass
class KolmogorovSolver:
    """Stateful wrapper around the functional solver API.

    All internal computation is float64. Snapshots returned as float32.

    Args:
        n: Grid size (n × n).
        re: Reynolds number.
        k_f: Forcing wavenumber.
        mu: Linear drag coefficient.
        dt: Substep size (CFL-limited; default 1e-3 is safe for n=64,
            use ~2.5e-4 for n=256).
        device: Torch device.
    """
    n: int = 64
    re: float = 4000.0
    k_f: int = 4
    mu: float = 0.1
    dt: float = 1e-3
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # derived, filled by __post_init__
    nu: float = field(init=False)
    kx2d: Tensor = field(init=False, repr=False)
    ky2d: Tensor = field(init=False, repr=False)
    ksq: Tensor = field(init=False, repr=False)
    forcing_hat: Tensor = field(init=False, repr=False)
    dealias: Tensor = field(init=False, repr=False)
    _E: Tensor = field(init=False, repr=False)
    _E2: Tensor = field(init=False, repr=False)
    _f1: Tensor = field(init=False, repr=False)
    _fa: Tensor = field(init=False, repr=False)
    _fb: Tensor = field(init=False, repr=False)
    _fc: Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nu = 1.0 / self.re
        self.kx2d, self.ky2d, self.ksq = wavenumbers(self.n, self.device)
        self.forcing_hat = make_kolmogorov_forcing(self.n, self.k_f, self.device)
        self.dealias = dealias_mask(self.n, self.device)
        L = (-self.nu * self.ksq - self.mu).to(torch.float64)
        self._E, self._E2, self._f1, self._fa, self._fb, self._fc = make_etdrk4_coeffs(L, self.dt)

    def random_ic(self, batch: int = 1, seed: Optional[int] = None) -> Tensor:
        """Return a random vorticity field [B, n, n] float64 on self.device.

        Energy concentrated in low wavenumbers (|k| <= 4) for fast spin-up.
        """
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)
        n, device = self.n, self.device
        omega_hat = torch.zeros(batch, n, n // 2 + 1, dtype=torch.complex128, device=device)
        k_init = 4
        kx2d, ky2d, _ = wavenumbers(n, device)
        mask = (kx2d.abs() <= k_init) & (ky2d.abs() <= k_init) & (self.ksq > 0)
        n_modes = mask.sum().item()
        amp = torch.randn(batch, n_modes, 2, dtype=torch.float64, device=device, generator=rng)
        omega_hat[:, mask] = torch.view_as_complex(amp.contiguous())
        omega = torch.fft.irfft2(omega_hat * self.dealias, s=(n, n))
        # normalise to unit rms
        rms = omega.std(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
        return omega / rms

    def step(self, omega_hat: Tensor) -> Tensor:
        """Advance one substep; returns updated omega_hat."""
        return etdrk4_step(
            omega_hat, self._E, self._E2, self._f1, self._fa, self._fb, self._fc,
            self.kx2d, self.ky2d, self.ksq, self.forcing_hat, self.dealias,
        )

    def spinup(self, omega0: Tensor, n_substeps: int) -> Tensor:
        """Run n_substeps without saving; returns final omega_hat [B, n, n//2+1]."""
        omega_hat = torch.fft.rfft2(omega0.to(torch.float64)) * self.dealias
        for _ in range(n_substeps):
            omega_hat = self.step(omega_hat)
        return omega_hat

    def run(
        self,
        omega0: Tensor,
        n_snapshots: int,
        substeps_per_snapshot: int,
    ) -> Tensor:
        """Integrate from omega0 and return [B, n_snapshots, n, n] float32."""
        return integrate(
            omega0, n_snapshots, self.dt, substeps_per_snapshot,
            self.kx2d, self.ky2d, self.ksq, self.forcing_hat, self.dealias,
            self.nu, self.mu,
        )

    # --- diagnostics (accept physical vorticity [B, n, n] float64) ---

    def omega_to_hat(self, omega: Tensor) -> Tensor:
        return torch.fft.rfft2(omega.to(torch.float64)) * self.dealias

    def get_energy(self, omega_hat: Tensor) -> Tensor:
        return energy(omega_hat, self.ksq, self.n)

    def get_enstrophy(self, omega_hat: Tensor) -> Tensor:
        return enstrophy(omega_hat, self.n)

    def get_spectrum(self, omega_hat: Tensor, n_bins: Optional[int] = None) -> tuple[Tensor, Tensor]:
        return radial_energy_spectrum(omega_hat, self.ksq, self.n, n_bins)
