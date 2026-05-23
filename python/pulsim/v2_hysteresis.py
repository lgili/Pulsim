"""Pulsim v2 — magnetic hysteresis (Jiles-Atherton B-H loop).

Closes Phase-2 gap #5 from the v1→v2 retirement punch list.

What this module is
-------------------
A **post-processing** Jiles-Atherton (J-A) magnetic-hysteresis model.
Given the current waveform through an inductor (from any v2
simulation), the model computes:

* the B(t) and H(t) waveforms (the B-H loop),
* the energy lost per cycle (= ∮ H dB area = J·m⁻³),
* the average core-loss power (W).

This is the engineering deliverable for hysteresis: most users want
loss accounting + B-H loop visualisation, not the kernel-level
coupling. For most circuits the inductance is set by air gap +
turns, NOT by saturating core material, so the hysteresis voltage
drop is a tiny correction compared to ωL — modelling it inside the
kernel loop adds complexity but rarely changes the operating point
materially. By contrast, the **core loss** matters a lot for
thermal design — and that's what this module quantifies.

What this module is NOT
-----------------------
* Not a kernel-coupled nonlinear inductor — the J-A B(H) is NOT
  fed back into KCL. For that, use :func:`add_saturable_inductor`
  (smooth saturation curve, no memory) inside the kernel and this
  module for the loss accounting on top.
* Not Preisach / Bouc-Wen — those are alternative models; J-A is
  the textbook standard.

Theory recap (Jiles-Atherton, 1986)
-----------------------------------
Five parameters: ``Ms`` (saturation magnetisation, A/m), ``a``
(shape, A/m), ``alpha`` (mean-field coupling, dimensionless), ``c``
(reversibility, dimensionless), ``k`` (pinning coefficient, A/m).

Anhysteretic magnetisation (Langevin)::

    He = H + alpha · M
    M_an = Ms · ( coth(He/a) − a/He )      (Ms · L(He/a))

Differential::

    dM/dH = (1 − c) · (M_an − M) /
            (k · sign(dH/dt) − alpha · (M_an − M))
          + c · dM_an/dH

Resulting flux density: ``B = μ₀ · (H + M)``.

Energy density per cycle = ∮ H dB (J/m³).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


__all__ = [
    "JilesAthertonParams",
    "JilesAthertonModel",
    "compute_bh_loop",
    "core_loss_jiles_atherton",
]


# 4π × 10⁻⁷ — vacuum permeability.
_MU0 = 4.0 * np.pi * 1e-7


# =============================================================================
# Parameter struct + reference values
# =============================================================================

@dataclass
class JilesAthertonParams:
    """Five Jiles-Atherton parameters.

    Reference values for common soft magnetic materials (from
    published J-A fits — useful starting points for tuning):

    +----------------+--------+--------+-------+-------+--------+
    | Material       | Ms     | a      | alpha | c     | k      |
    +----------------+--------+--------+-------+-------+--------+
    | Annealed iron  | 1.7e6  | 1.0e3  | 1e-3  | 0.10  | 200    |
    | Si-steel (M19) | 1.6e6  | 1.5e3  | 5e-4  | 0.05  | 350    |
    | Ferrite N87    | 4.0e5  | 50     | 5e-5  | 0.20  | 30     |
    | Permalloy      | 8.0e5  | 30     | 1e-4  | 0.30  | 10     |
    +----------------+--------+--------+-------+-------+--------+

    Ms is in A/m; a + k are in A/m; alpha + c are dimensionless.

    Attributes
    ----------
    Ms
        Saturation magnetisation (A/m).
    a
        Anhysteretic-curve shape parameter (A/m).
    alpha
        Mean-field interdomain coupling (dimensionless, ~1e-5–1e-2).
    c
        Reversibility coefficient (dimensionless, ∈ [0, 1]). c=0 →
        purely irreversible hysteresis (max loss); c=1 → purely
        anhysteretic (zero loss).
    k
        Pinning coefficient (A/m) — controls the loop width.
        Larger k = wider loop = more loss.
    """
    Ms: float
    a: float
    alpha: float
    c: float
    k: float


# Hand-tuned demo parameter sets for tests / smoke runs.
_REFERENCE_MATERIALS: Dict[str, JilesAthertonParams] = {
    "annealed_iron":  JilesAthertonParams(Ms=1.7e6, a=1.0e3,
                                            alpha=1e-3, c=0.10, k=200),
    "si_steel_m19":   JilesAthertonParams(Ms=1.6e6, a=1.5e3,
                                            alpha=5e-4, c=0.05, k=350),
    "ferrite_n87":    JilesAthertonParams(Ms=4.0e5, a=50.0,
                                            alpha=5e-5, c=0.20, k=30),
    "permalloy":      JilesAthertonParams(Ms=8.0e5, a=30.0,
                                            alpha=1e-4, c=0.30, k=10),
}


def reference_material(name: str) -> JilesAthertonParams:
    """Look up a pre-tuned J-A parameter set by material name.
    Available: ``annealed_iron``, ``si_steel_m19``, ``ferrite_n87``,
    ``permalloy``."""
    try:
        return _REFERENCE_MATERIALS[name]
    except KeyError:
        avail = ", ".join(sorted(_REFERENCE_MATERIALS))
        raise KeyError(
            f"unknown reference material {name!r}; "
            f"available: {avail}")


# =============================================================================
# Per-step model (stateful)
# =============================================================================

def _langevin(x: float) -> float:
    """Langevin function L(x) = coth(x) − 1/x. Numerically robust
    near zero via Taylor expansion."""
    ax = abs(x)
    if ax < 1e-4:
        # L(x) ≈ x/3 − x³/45 + …
        return x * (1.0 / 3.0 - x * x / 45.0)
    return float(np.cosh(x) / np.sinh(x) - 1.0 / x)


def _langevin_deriv(x: float) -> float:
    """dL/dx. Robust near zero (Taylor)."""
    ax = abs(x)
    if ax < 1e-4:
        # L'(x) ≈ 1/3 − x²/15 + …
        return 1.0 / 3.0 - x * x / 15.0
    sinh_x = np.sinh(x)
    return float(1.0 / (x * x) - 1.0 / (sinh_x * sinh_x))


class JilesAthertonModel:
    """Stateful J-A integrator. Feed ``update(H_new, dt)`` once per
    simulation step; query ``M``, ``B`` between calls.

    The integrator is forward-Euler on dM/dH with explicit ``sign(dH/dt)``.
    For typical soft-magnetic materials this is stable and accurate
    when ``dt · dH/dt << k``. Bumps in sign(dH/dt) (zero-crossings of
    H) are handled by clamping the denominator away from zero.

    Parameters
    ----------
    params
        :class:`JilesAthertonParams`.
    M0
        Initial magnetisation (A/m). Default 0.

    Attributes
    ----------
    params
        The parameter set.
    M
        Current magnetisation (A/m).
    B
        Current flux density (T), computed lazily from current H.
    H
        Current applied field (A/m), from last ``update`` call.
    """

    def __init__(self, params: JilesAthertonParams, M0: float = 0.0):
        self.params = params
        self.M = float(M0)
        self.H = 0.0
        self.B = _MU0 * (self.H + self.M)
        self._H_prev = 0.0
        self._initialized = False

    def reset(self, M0: float = 0.0) -> None:
        """Clear the magnetisation memory."""
        self.M = float(M0)
        self.H = 0.0
        self.B = _MU0 * (self.H + self.M)
        self._H_prev = 0.0
        self._initialized = False

    def update(self, H_new: float, dt: float) -> float:
        """Advance one step. Returns the new ``B`` (T).

        Caller supplies the applied field ``H_new`` (A/m) and the
        timestep. For a coil of ``N`` turns with current ``i``
        on a magnetic path of length ``l_m``: ``H = N · i / l_m``.
        """
        p = self.params
        H = float(H_new)
        if not self._initialized:
            self._H_prev = H
            self._initialized = True
            self.H = H
            self.B = _MU0 * (H + self.M)
            return self.B

        dH = H - self._H_prev
        sign_dH = 1.0 if dH >= 0 else -1.0

        # Anhysteretic magnetisation at He = H + αM.
        He = H + p.alpha * self.M
        x = He / p.a if p.a > 0 else 0.0
        L = _langevin(x)
        L_prime = _langevin_deriv(x)
        M_an = p.Ms * L
        dM_an_dHe = (p.Ms / p.a) * L_prime if p.a > 0 else 0.0
        # dM_an/dH via chain rule: dM_an/dHe · dHe/dH = dM_an/dHe · (1 + α dM/dH).
        # For the explicit step we approximate dM/dH ≈ slope from
        # previous step (treated implicitly enough for forward
        # Euler).

        # Irreversible-magnetisation derivative. Robust against the
        # degenerate k→0 / dH→0 boundary: when |denom| underflows we
        # smoothly degenerate dM_irr_dH to zero (purely anhysteretic
        # behaviour — no irreversible flow when the pinning vanishes).
        denom = p.k * sign_dH - p.alpha * (M_an - self.M)
        denom_floor = max(1e-9, 1e-6 * p.k)
        if abs(denom) < denom_floor:
            dM_irr_dH = 0.0
        else:
            dM_irr_dH = (M_an - self.M) / denom

        # Reversible contribution.
        rev_denom = 1.0 - p.c * p.alpha * dM_an_dHe
        if abs(rev_denom) > 1e-12:
            dM_rev_dH = p.c * dM_an_dHe / rev_denom
        else:
            dM_rev_dH = p.c * dM_an_dHe

        dM_dH = (1.0 - p.c) * dM_irr_dH + dM_rev_dH

        # Forward-Euler integration.
        self.M = self.M + dM_dH * dH

        self._H_prev = H
        self.H = H
        self.B = _MU0 * (H + self.M)
        return self.B


# =============================================================================
# Post-processing: B-H loop from a current history
# =============================================================================

@dataclass
class BHLoopResult:
    """Output of :func:`compute_bh_loop`.

    Attributes
    ----------
    H
        Applied field history (A/m), one entry per input current
        sample.
    B
        Flux density history (T), parallel to H.
    M
        Magnetisation history (A/m), parallel to H.
    energy_per_cycle_per_m3
        Last-cycle area ∮ H dB (J/m³).
    avg_power_per_m3
        ``energy_per_cycle_per_m3 · frequency`` (W/m³).
    avg_power_W
        ``avg_power_per_m3 · core_volume_m3`` (W). Only present when
        the caller supplied ``A_core`` × ``l_m`` (volume).
    """
    H: np.ndarray
    B: np.ndarray
    M: np.ndarray
    energy_per_cycle_per_m3: float
    avg_power_per_m3: float
    avg_power_W: float


def compute_bh_loop(current,
                       times,
                       *,
                       params: JilesAthertonParams,
                       N_turns: int,
                       l_m: float,
                       A_core: float,
                       period: float | None = None,
                       M0: float = 0.0,
                       ) -> BHLoopResult:
    """Compute the B-H loop from an inductor's current history.

    Parameters
    ----------
    current
        Array of inductor current samples (A).
    times
        Array of timestamps (s) parallel to ``current``. Need not be
        uniform.
    params
        :class:`JilesAthertonParams`.
    N_turns
        Number of turns in the coil.
    l_m
        Mean magnetic path length (m).
    A_core
        Effective core cross-sectional area (m²).
    period
        Optional fundamental period (s). When given, the energy
        integral is taken over the LAST period (assumes steady
        state). When ``None``, integrates over the entire input.
    M0
        Initial magnetisation (A/m). Default 0.

    Returns
    -------
    BHLoopResult
    """
    i_arr = np.asarray(current, dtype=np.float64)
    t_arr = np.asarray(times, dtype=np.float64)
    if i_arr.shape != t_arr.shape:
        raise ValueError("current and times must have the same shape")
    if i_arr.size < 2:
        raise ValueError("need at least 2 samples")

    H_arr = (N_turns * i_arr) / l_m
    model = JilesAthertonModel(params, M0=M0)
    B_arr = np.empty_like(H_arr)
    M_arr = np.empty_like(H_arr)

    for k in range(H_arr.size):
        dt = (t_arr[k] - t_arr[k - 1]) if k > 0 else 0.0
        B = model.update(float(H_arr[k]), dt)
        B_arr[k] = B
        M_arr[k] = model.M

    # Energy-per-cycle area integral ∮ H dB. Use trapezoidal on the
    # LAST period (or the whole input).
    if period is None:
        t_lo = t_arr[0]
        t_hi = t_arr[-1]
    else:
        t_hi = t_arr[-1]
        t_lo = t_hi - period
        if t_lo < t_arr[0]:
            raise ValueError(
                "period longer than the supplied current history")

    mask = t_arr >= t_lo
    H_p = H_arr[mask]
    B_p = B_arr[mask]
    # ∮ H dB ≈ Σ H_k · (B_{k+1} − B_{k}); use 0.5(H_k + H_{k+1}) for
    # trapezoidal accuracy.
    H_mid = 0.5 * (H_p[:-1] + H_p[1:])
    dB = np.diff(B_p)
    energy_per_m3 = float(np.sum(H_mid * dB))

    f = 1.0 / period if period else 1.0 / (t_arr[-1] - t_arr[0])
    avg_pow_per_m3 = energy_per_m3 * f
    volume = float(A_core) * float(l_m)
    avg_pow_W = avg_pow_per_m3 * volume

    return BHLoopResult(
        H=H_arr, B=B_arr, M=M_arr,
        energy_per_cycle_per_m3=energy_per_m3,
        avg_power_per_m3=avg_pow_per_m3,
        avg_power_W=avg_pow_W,
    )


# =============================================================================
# Convenience: average J-A core loss from a v2 SimulationResult
# =============================================================================

def core_loss_jiles_atherton(result,
                                  *,
                                  inductor_branch_id: int,
                                  builder,
                                  params: JilesAthertonParams,
                                  N_turns: int,
                                  l_m: float,
                                  A_core: float,
                                  period: float | None = None,
                                  M0: float = 0.0,
                                  ) -> BHLoopResult:
    """Convenience: pull the inductor current straight from a
    :class:`SimulationResult` and pass to :func:`compute_bh_loop`.

    Parameters
    ----------
    result
        A :class:`SimulationResult`.
    inductor_branch_id
        Builder-relative branch ID returned by ``add_inductor``
        (track it manually — ``builder.graph.num_branches`` before
        the ``add_inductor`` call).
    builder
        :class:`CircuitBuilder` used to run the sim — needed to
        resolve the inductor's state-vector index via
        ``pool.branch_var_id_for_inductor``.
    params, N_turns, l_m, A_core, period, M0
        Forwarded to :func:`compute_bh_loop`.

    Returns
    -------
    BHLoopResult
    """
    times_raw = result.times
    states_raw = result.states
    if hasattr(states_raw, "shape"):
        states = np.asarray(states_raw, dtype=np.float64)
    else:
        states = np.asarray(
            [list(v) for v in states_raw], dtype=np.float64)
    times = np.asarray(times_raw, dtype=np.float64)
    i_idx = builder.pool.branch_var_id_for_inductor(
        inductor_branch_id, builder.graph)
    i_L = states[:, i_idx]
    return compute_bh_loop(
        i_L, times,
        params=params,
        N_turns=N_turns,
        l_m=l_m,
        A_core=A_core,
        period=period,
        M0=M0)
