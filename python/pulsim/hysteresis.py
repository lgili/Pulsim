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

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


__all__ = [
    "JilesAthertonParams",
    "JilesAthertonModel",
    "reference_material",
    "list_reference_materials",
    "compute_bh_loop",
    "core_loss_jiles_atherton",
    "fit_ja_from_bh_curve",
    "BHLoopResult",
    # In-loop hysteretic-inductor (Python device — reuses the
    # linear-inductor MNA path).
    "HystereticInductor",
    "add_hysteretic_inductor",
    "make_hysteretic_inductor_observer",
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


def list_reference_materials() -> list:
    """Names of materials with pre-fitted Jiles-Atherton parameter
    sets — useful for ``pulsim.catalog()``-style discovery."""
    return sorted(_REFERENCE_MATERIALS)


# =============================================================================
# Per-step model (stateful)
# =============================================================================

def _langevin(x: float) -> float:
    """Langevin function L(x) = coth(x) − 1/x. Numerically robust
    near zero (Taylor) AND at large |x| (coth saturates to ±1).
    Without the large-|x| guard, sinh/cosh overflow to inf for
    |x| ≳ 710 and the ratio gives NaN — typical in heavily-driven
    saturable cores."""
    ax = abs(x)
    if ax < 1e-4:
        # L(x) ≈ x/3 − x³/45 + …
        return x * (1.0 / 3.0 - x * x / 45.0)
    if ax > 30.0:
        # coth(x) ≈ 1 + 2·e^(-2|x|), so L(x) ≈ sign(x) - 1/x.
        return (1.0 if x > 0 else -1.0) - 1.0 / x
    return float(np.cosh(x) / np.sinh(x) - 1.0 / x)


def _langevin_deriv(x: float) -> float:
    """dL/dx. Robust near zero (Taylor) and at large |x| where
    sinh²(x) overflows."""
    ax = abs(x)
    if ax < 1e-4:
        # L'(x) ≈ 1/3 − x²/15 + …
        return 1.0 / 3.0 - x * x / 15.0
    if ax > 30.0:
        # 1/sinh²(x) → 0 for large |x|; L'(x) ≈ 1/x².
        return 1.0 / (x * x)
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

        # NOTE: ``dM_dH`` is RE-evaluated inside the sub-step loop
        # below at the sub-step's intermediate H value (that's what
        # makes the sub-stepping numerically stable). The outer-scope
        # quantities ``dM_irr_dH`` / ``dM_rev_dH`` are still useful
        # diagnostics for the JA model fitter but are not used by
        # the integration step itself.
        del dM_irr_dH, dM_rev_dH

        # Forward-Euler integration with two safeguards:
        #   (1) Sub-step on |dH| if the jump per step is too large
        #       (forward Euler diverges when |dH| ≳ a, the Langevin
        #       shape parameter).
        #   (2) Hard clamp M to ±Ms (saturation magnetization) so
        #       any residual numerical overshoot doesn't propagate
        #       across steps.
        max_dH = max(0.5 * p.a, 1.0)
        n_sub = max(1, int(math.ceil(abs(dH) / max_dH)))
        dH_sub = dH / n_sub
        # Re-evaluate dM/dH at each sub-step for stability when n_sub > 1.
        for _ in range(n_sub):
            H_sub = self._H_prev + dH_sub
            He_s = H_sub + p.alpha * self.M
            x_s = He_s / p.a if p.a > 0 else 0.0
            M_an_s = p.Ms * _langevin(x_s)
            dM_an_dHe_s = ((p.Ms / p.a) * _langevin_deriv(x_s)
                                if p.a > 0 else 0.0)
            denom_s = p.k * sign_dH - p.alpha * (M_an_s - self.M)
            if abs(denom_s) < denom_floor:
                dM_irr_s = 0.0
            else:
                dM_irr_s = (M_an_s - self.M) / denom_s
            rev_d_s = 1.0 - p.c * p.alpha * dM_an_dHe_s
            dM_rev_s = (p.c * dM_an_dHe_s / rev_d_s
                              if abs(rev_d_s) > 1e-12
                              else p.c * dM_an_dHe_s)
            dM_dH_s = (1.0 - p.c) * dM_irr_s + dM_rev_s
            self.M += dM_dH_s * dH_sub
            # Saturation clamp.
            if self.M > p.Ms:
                self.M = p.Ms
            elif self.M < -p.Ms:
                self.M = -p.Ms
            self._H_prev = H_sub

        self._H_prev = H   # restore final position after sub-steps
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


# =============================================================================
# Curve fitting — derive J-A parameters from a measured B-H trace
# =============================================================================

def fit_ja_from_bh_curve(B_array,
                              H_array,
                              *,
                              p0: JilesAthertonParams | None = None,
                              max_iter: int = 400,
                              ) -> JilesAthertonParams:
    """Least-squares fit of the five Jiles-Atherton parameters to a
    measured B-H loop.

    Parameters
    ----------
    B_array, H_array
        Sample pairs along a measured B-H loop, one full period.
        For best fit quality supply 200+ samples covering both
        ascending and descending branches.
    p0
        Initial-guess parameters. Default uses the ``ferrite_n87``
        catalog entry. Pass an entry from :func:`reference_material`
        when fitting to a known material family.
    max_iter
        Maximum number of optimizer iterations (``scipy.optimize.
        least_squares``). Increase if the residual is still
        decreasing at exit; decrease for a faster, rougher fit.

    Returns
    -------
    JilesAthertonParams
        Best-fit parameters minimising the RMS B-error along the
        provided trace.

    Notes
    -----
    Requires :mod:`scipy.optimize`. Falls back to a simple
    coordinate-descent if scipy isn't available — much slower and
    less accurate, but still produces sane numbers for the catalog
    materials.

    The fit replays the supplied ``H_array`` through a fresh
    :class:`JilesAthertonModel` and minimises ``Σ (B_pred − B_meas)²``.
    Wall-clock cost ≈ ``n_samples · max_iter / 5000`` seconds on a
    modern laptop (200 samples × 200 iter ≈ 10 s).
    """
    B_arr = np.asarray(B_array, dtype=np.float64)
    H_arr = np.asarray(H_array, dtype=np.float64)
    if B_arr.shape != H_arr.shape or B_arr.size < 8:
        raise ValueError(
            "B_array and H_array must be the same shape and "
            "contain at least 8 sample pairs.")
    if p0 is None:
        p0 = _REFERENCE_MATERIALS["ferrite_n87"]

    def _residuals(theta):
        Ms, a, alpha, c, k = theta
        if Ms <= 0 or a <= 0 or k <= 0 or not (0.0 <= c <= 1.0):
            return np.full_like(B_arr, 1e6)
        params = JilesAthertonParams(
            Ms=Ms, a=a, alpha=alpha, c=c, k=k)
        model = JilesAthertonModel(params)
        B_pred = np.empty_like(B_arr)
        for i, H in enumerate(H_arr):
            B_pred[i] = model.update(float(H), 1.0)
        return B_pred - B_arr

    try:
        from scipy.optimize import least_squares
        theta0 = np.array([p0.Ms, p0.a, p0.alpha, p0.c, p0.k],
                              dtype=np.float64)
        bounds = (np.array([1e3, 1.0, 1e-6, 0.0, 1.0]),
                       np.array([5e6, 1e4, 1.0, 1.0, 1e4]))
        result = least_squares(_residuals, theta0, bounds=bounds,
                                       max_nfev=int(max_iter))
        Ms_f, a_f, alpha_f, c_f, k_f = result.x
        return JilesAthertonParams(
            Ms=float(Ms_f), a=float(a_f),
            alpha=float(alpha_f), c=float(c_f), k=float(k_f))
    except ImportError:
        # Coordinate-descent fallback (works without scipy).
        params = JilesAthertonParams(
            Ms=p0.Ms, a=p0.a, alpha=p0.alpha, c=p0.c, k=p0.k)
        names_and_steps = [("Ms", 0.1), ("a", 0.05), ("alpha", 0.05),
                                ("c", 0.02), ("k", 0.05)]
        for _ in range(max(20, max_iter // 25)):
            for field_name, frac in names_and_steps:
                value = getattr(params, field_name)
                trials = [value * (1 + frac), value, value * (1 - frac)]
                best_err = float("inf")
                best_v = value
                for v in trials:
                    if field_name == "c" and not (0.0 <= v <= 1.0):
                        continue
                    if v <= 0 and field_name != "alpha":
                        continue
                    theta = np.array([
                        v if field_name == "Ms" else params.Ms,
                        v if field_name == "a" else params.a,
                        v if field_name == "alpha" else params.alpha,
                        v if field_name == "c" else params.c,
                        v if field_name == "k" else params.k,
                    ])
                    res = _residuals(theta)
                    err = float(np.sqrt(np.mean(res ** 2)))
                    if err < best_err:
                        best_err = err
                        best_v = v
                params = JilesAthertonParams(
                    Ms=best_v if field_name == "Ms" else params.Ms,
                    a=best_v if field_name == "a" else params.a,
                    alpha=best_v if field_name == "alpha"
                              else params.alpha,
                    c=best_v if field_name == "c" else params.c,
                    k=best_v if field_name == "k" else params.k,
                )
        return params


# =============================================================================
# In-loop hysteretic inductor — Python device that PARTICIPATES in MNA
# =============================================================================
#
# Wires the JilesAthertonModel into a real circuit using the same
# "linear inductor + dummy voltage source modulated by observer"
# pattern that already powers ``add_pmsm`` / ``add_bldc`` /
# ``add_induction_motor``. No C++ kernel changes needed.
#
# Decomposition (B-H constitutive split):
#
#   B = μ_0 · (H + M)
#   λ = N · A · B = N·A·μ_0·H + N·A·μ_0·M
#       = L_0 · i + ψ_M(M)             (because H = N·i/l_m)
#
# where
#     L_0 = N² · A · μ_0 / l_m         (air-core inductance — linear)
#     ψ_M = N · A · μ_0 · M            (magnetization-contribution flux)
#
# Differentiating wrt time:
#     v_L = dλ/dt = L_0·di/dt  +  dψ_M/dt
#         = L_0·di/dt  +  N·A·μ_0·dM/dt
#
# The first term is just a linear inductor (handled by the kernel
# trivially). The second term is a per-step voltage that the JA
# observer computes from the rotor-flux-style ODE — it appears as
# a "back-EMF" injected into ``b_extra`` at the dummy voltage
# source's branch row. The observer ALSO reads back the current
# at every step to feed the JA model with the actual H.
#
# Trade-offs vs a fully kernel-coupled implicit C++ device:
#   * Pro: zero C++ changes, ships on every wheel that has motors.
#   * Pro: stays within the proven observer pattern.
#   * Con: forward-Euler integration of M (the JA ODE is integrated
#     at the simulation dt). For typical SMPS / mains-transformer
#     work this is plenty — the JA dynamics live at the mains /
#     switching scale, well above the simulator's dt.

@dataclass
class HystereticInductor:
    """A hysteretic-core inductor backed by a Jiles-Atherton model.

    Returned by :func:`add_hysteretic_inductor`. Stash the
    handle to (a) read the live ``M`` / ``B`` / ``H`` per-step
    via the observer, and (b) pass it back to
    :func:`make_hysteretic_inductor_observer` for the
    ``step_observer`` + ``b_extra_fn`` pair the simulator
    needs.
    """
    name: str
    params: "JilesAthertonParams"
    N_turns: int
    """Number of turns on the coil."""
    l_m: float
    """Mean magnetic path length [m]."""
    A_core: float
    """Effective core cross-sectional area [m²]."""
    L_0: float
    """Air-core inductance L_0 = N²·A·μ_0/l_m [H] — the LINEAR
    component the kernel sees on the MNA branch."""
    inductor_branch_id: int = -1
    bemf_source_id: int = -1
    # Live observer state (snapshotted between steps for diagnostics).
    M: float = 0.0
    H: float = 0.0
    B: float = 0.0


def add_hysteretic_inductor(builder,
                                *,
                                name: str,
                                from_node: str,
                                to_node: str,
                                params: "JilesAthertonParams",
                                N_turns: int,
                                l_m: float,
                                A_core: float,
                                ) -> "HystereticInductor":
    """Add a Jiles-Atherton hysteretic inductor between two nodes.

    Topology:

        from_node ──[ L_0 ]── (mid) ──[ V_M ]── to_node

    where ``L_0 = N²·A·μ_0/l_m`` is the linear air-core inductance
    and ``V_M`` is a dummy voltage source that the JA observer
    modulates to encode ``N·A·μ_0·dM/dt`` — the hysteresis
    contribution to ``v_L = dλ/dt``.

    Parameters
    ----------
    name
        Prefix for the added components (``{name}_L0`` and
        ``{name}_V_M``).
    from_node, to_node
        Terminal node names.
    params
        :class:`JilesAthertonParams` — the 5-parameter JA set.
        See :func:`reference_material` for catalog lookup or
        :func:`fit_ja_from_bh_curve` for fitting from measurements.
    N_turns
        Number of turns on the coil.
    l_m
        Mean magnetic path length [m].
    A_core
        Effective core cross-sectional area [m²].

    Returns
    -------
    HystereticInductor
        Handle to pass to :func:`make_hysteretic_inductor_observer`.
    """
    if N_turns <= 0 or l_m <= 0.0 or A_core <= 0.0:
        raise ValueError(
            f"Non-physical geometry: N={N_turns}, l_m={l_m}, "
            f"A_core={A_core}. All must be positive.")
    L_0 = (N_turns ** 2) * A_core * _MU0 / l_m
    mid = f"{name}_mid"
    ind_id = builder.graph.num_branches
    builder.add_inductor(f"{name}_L0", from_node, mid, float(L_0))
    src_id = builder.graph.num_branches
    builder.add_voltage_source(f"{name}_V_M", mid, to_node, 0.0)
    return HystereticInductor(
        name=name,
        params=params,
        N_turns=int(N_turns),
        l_m=float(l_m),
        A_core=float(A_core),
        L_0=float(L_0),
        inductor_branch_id=ind_id,
        bemf_source_id=src_id,
    )


def make_hysteretic_inductor_observer(builder,
                                            hyst: "HystereticInductor",
                                            *,
                                            dt: float):
    """Build the ``(step_observer, b_extra_fn)`` pair to attach a
    :class:`HystereticInductor` to a running simulation.

    The observer:
      1. Reads the inductor current ``i_L`` from the state vector.
      2. Computes ``H = N·i_L / l_m``.
      3. Advances the JA model state ``M`` by one ``dt``.
      4. Writes back ``V_M = N·A·μ_0·dM/dt`` into ``b_extra``
         at the dummy voltage source's row (negated to match the
         kernel's source-row sign convention).

    The ``hyst`` handle is updated in place each step so the
    caller can read ``hyst.M``, ``hyst.B``, ``hyst.H`` between
    sim steps for live plotting / channel logging.
    """
    state_size = builder.pool.state_size(builder.graph)
    ind_idx = builder.pool.branch_var_id_for_inductor(
        hyst.inductor_branch_id, builder.graph)
    src_idx = builder.pool.branch_var_id_for_source(
        hyst.bemf_source_id, builder.graph)

    model = JilesAthertonModel(hyst.params)
    # Coupling constant ψ_M = N · A · μ_0 · M → dψ_M/dt = const · dM/dt.
    psi_coupling = hyst.N_turns * hyst.A_core * _MU0

    # Snapshot the previous M to compute dM/dt → V_M each step.
    state = {"M_prev": 0.0, "v_M": 0.0}

    def step_observer(t, x):
        i_L = float(x[ind_idx])
        H = (hyst.N_turns / hyst.l_m) * i_L
        model.update(H, dt)
        dM = model.M - state["M_prev"]
        state["M_prev"] = model.M
        # V_M = N·A·μ_0·dM/dt.
        state["v_M"] = psi_coupling * (dM / max(dt, 1e-18))
        # Live diagnostics on the handle.
        hyst.M = model.M
        hyst.H = H
        hyst.B = model.B

    def b_extra_fn(t):
        out = [0.0] * state_size
        # Total flux linkage of the hysteretic coil:
        #   λ = L_0·i + N·A·μ_0·M
        # → dλ/dt = L_0·di/dt + N·A·μ_0·dM/dt = L_0·di/dt + v_M
        # The branch is:  from_node ── L_0 ── mid ── V_dummy ── to_node
        # Loop KVL: V_applied = V_L0 + V_dummy_setpoint
        #         = L_0·di/dt + V_dummy_setpoint
        # For this to equal dλ/dt we need V_dummy_setpoint = +v_M.
        # (Earlier sign was inherited from the motor back-EMF pattern by
        # analogy, but the topology here is single-branch passive — the
        # hysteresis voltage adds in the SAME direction as the L_0 drop,
        # not opposite to a separately-driven excitation source.)
        out[src_idx] = +state["v_M"]
        return out

    return step_observer, b_extra_fn
