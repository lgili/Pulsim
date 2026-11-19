"""Pulsim — magnetic core-loss models (Phase C.3).

Three loss formulas for soft-magnetic cores (ferrites, nano-
crystalline, powder), plus a small built-in material catalog with
typical Steinmetz coefficients from manufacturer datasheets.

  * ``steinmetz_loss_density(B_peak, frequency, material)`` — the
    classic ``P_v = K · f^α · B_peak^β`` formula. Returns volumetric
    loss density in W/m³. Valid for sinusoidal B(t) only.

  * ``igse_loss_density(B_t, t, material)`` — Improved Generalized
    Steinmetz Equation (iGSE) by Venkatachalam et al. (IEEE 2002).
    Handles arbitrary (non-sinusoidal) B(t) waveforms — essential
    for SMPS where flux is typically trapezoidal or piecewise-linear.

  * ``core_loss(B_peak_or_Bt, frequency_or_t, material, volume)`` —
    convenience wrapper that returns total core loss in watts.

The built-in catalog covers a handful of widely-used ferrite cores
suitable for 50 kHz – 1 MHz SMPS designs. Each material entry is a
single Steinmetz triplet (α, β, K) extracted at a specific
flux density and frequency from the manufacturer datasheet — for
high-precision design across a wide band, prefer a multi-segment
fit and pick the one closest to your operating point.

Typical usage:

    n87 = p.core_material("N87")
    B_peak = 0.2          # T
    f = 100e3             # Hz
    V_e = 5e-6            # m³ (5 cm³ core)
    P_loss = p.core_loss(B_peak, f, n87, V_e)
    print(f"Core loss = {P_loss:.2f} W")

For a complex (e.g. trapezoidal) B(t) from a saturable inductor
simulation, pass the trace:

    P_density = p.igse_loss_density(B_trace, t_trace, n87)
    P_loss = P_density * V_e
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


__all__ = [
    "CoreMaterial",
    "core_material",
    "list_core_materials",
    "steinmetz_loss_density",
    "igse_loss_density",
    "core_loss",
]


# =============================================================================
# CoreMaterial — POD with the Steinmetz triplet (α, β, K)
# =============================================================================

@dataclass(frozen=True)
class CoreMaterial:
    """One ferrite / powder-core material.

    Loss density follows Steinmetz:
        P_v [W/m³] = K · f^α · B_peak^β

    where f is in Hz and B_peak in T. K has the funky units
    [W / (m³ · Hz^α · T^β)] that fall out of the dimensionally-
    inconsistent fit — store it as a single float.

    The "manufacturer notes" string captures the datasheet conditions
    (typical: a single quoted measurement frequency + B value, with
    the published Steinmetz coefficients extracted from a band
    around it).
    """
    name: str
    K: float        # Steinmetz pre-factor (in datasheet units)
    alpha: float    # frequency exponent
    beta: float     # B-field exponent
    notes: str = ""

    def __repr__(self) -> str:
        return (f"CoreMaterial(name={self.name!r}, "
                  f"K={self.K:.3g}, α={self.alpha:.2f}, "
                  f"β={self.beta:.2f})")


# =============================================================================
# Built-in catalog — values from EPCOS / TDK / Ferroxcube datasheets
# =============================================================================

# THE single source of truth for Steinmetz coefficients. Until v2.0
# the core files under `pulsim/lib/data/cores/` carried their own set
# and disagreed with this table in every parameter — three sources
# (here, `hysteresis._REFERENCE_MATERIALS`, and the YAML), three
# answers, all synthetic, and nothing said which one a result came
# from. Those YAML blocks are gone; the files keep the geometry and
# the B-H points, which are theirs alone.
#
# These coefficients are ILLUSTRATIVE. `steinmetz_loss_density` reads
# P_v [W/m³] = K · f^α · B_peak^β with f in Hz and B in T; a vendor's
# published curve is a multi-band fit and will not match. For real
# numbers, fit the vendor's loss chart and add the material here with
# its provenance.
_CATALOG = {
    # General-purpose ferrites — typical (~100 kHz, ~0.2 T) Steinmetz
    # coefficients. Manufacturers publish multi-band fits; these are
    # single-band approximations adequate for ≤ ±30 % accuracy at the
    # quoted operating point.
    "N87":  CoreMaterial(
        name="N87",
        K=16.9, alpha=1.40, beta=2.50,
        notes="EPCOS / TDK ferrite. Typical 25–200 kHz, 50 mT–300 mT."),
    "N97":  CoreMaterial(
        name="N97",
        K=8.0,  alpha=1.42, beta=2.50,
        notes="EPCOS / TDK low-loss ferrite. Typical 50–500 kHz."),
    "3C95": CoreMaterial(
        name="3C95",
        K=15.0, alpha=1.45, beta=2.55,
        notes="Ferroxcube power ferrite. 25–500 kHz, low loss at 100 °C."),
    "3F3":  CoreMaterial(
        name="3F3",
        K=12.5, alpha=1.50, beta=2.60,
        notes="Ferroxcube high-freq ferrite. 200 kHz–1 MHz."),
    # Generic placeholder — average of the four above.
    "generic-ferrite": CoreMaterial(
        name="generic-ferrite",
        K=13.0, alpha=1.45, beta=2.55,
        notes="Average of N87/N97/3C95/3F3 — use only when no "
                "manufacturer data is available."),
}


def core_material(name: str) -> CoreMaterial:
    """Look up a built-in material by name. Raises KeyError if not found.
    Use :func:`list_core_materials` to see what's available."""
    try:
        return _CATALOG[name]
    except KeyError as exc:
        raise KeyError(
            f"core_material({name!r}): unknown material. Known: "
            f"{sorted(_CATALOG)}") from exc


def list_core_materials() -> list:
    """Names of all built-in materials."""
    return sorted(_CATALOG)


# =============================================================================
# Loss formulas
# =============================================================================

def steinmetz_loss_density(B_peak_T: float,
                              frequency_Hz: float,
                              material: CoreMaterial,
                              ) -> float:
    """Classic Steinmetz: P_v [W/m³] = K · f^α · B_peak^β.

    Valid for sinusoidal flux at frequency `frequency_Hz` and peak
    amplitude `B_peak_T`.
    """
    if B_peak_T < 0 or frequency_Hz < 0:
        raise ValueError("B_peak and frequency must be non-negative")
    if B_peak_T == 0 or frequency_Hz == 0:
        return 0.0
    return float(material.K *
                    (frequency_Hz ** material.alpha) *
                    (B_peak_T ** material.beta))


def igse_loss_density(B_t,
                          t,
                          material: CoreMaterial,
                          ) -> float:
    """Improved Generalized Steinmetz Equation (iGSE) on an arbitrary
    B(t) waveform.

    Implements the formula from Venkatachalam et al. (2002):

        P_v = (1/T) · ∫₀ᵀ k_i · |dB/dt|^α · (ΔB)^(β−α) dt

    where ΔB is the peak-to-peak flux density swing over the cycle
    and k_i is derived from K, α, β:

        k_i = K / [(2π)^(α−1) · ∫₀^{2π} |cos θ|^α · 2^(β−α) dθ]

    The ``|dB/dt|`` integral is computed numerically from the trace.
    For sinusoidal B(t) iGSE reduces to Steinmetz; for trapezoidal
    or piecewise-linear waveforms (typical SMPS) it gives much more
    realistic losses than classic Steinmetz.

    Parameters
    ----------
    B_t
        Flux density trace (T), uniformly sampled over one full
        period.
    t
        Time array (s), same length as B_t.
    material
        Steinmetz triplet.

    Returns
    -------
    float
        Loss density in W/m³.
    """
    B = np.asarray(B_t, dtype=float)
    tt = np.asarray(t, dtype=float)
    if B.shape != tt.shape or B.size < 4:
        raise ValueError(
            "B_t and t must have the same shape and ≥ 4 samples")

    T_period = float(tt[-1] - tt[0])
    if T_period <= 0:
        return 0.0

    # Peak-to-peak ΔB.
    delta_B = float(B.max() - B.min())
    if delta_B <= 0:
        return 0.0

    # Compute the iGSE constant k_i.
    alpha = material.alpha
    beta = material.beta
    K = material.K
    # Use Simpson's rule for ∫₀^{2π} |cosθ|^α dθ. Note: the 2^(β−α)
    # factor is constant across θ.
    theta = np.linspace(0.0, 2.0 * math.pi, 1001)
    integral_costα = float(np.trapezoid(np.abs(np.cos(theta))**alpha,
                                            x=theta))
    k_i = K / (((2.0 * math.pi) ** (alpha - 1.0)) *
                  integral_costα * (2.0 ** (beta - alpha)))

    # ∫ |dB/dt|^α dt over the period.
    dBdt = np.gradient(B, tt)
    integrand = (np.abs(dBdt) ** alpha) * (delta_B ** (beta - alpha))
    P_v = float(k_i * np.trapezoid(integrand, x=tt) / T_period)
    return P_v


def core_loss(B,
                frequency_or_t,
                material: CoreMaterial,
                volume_m3: float,
                ) -> float:
    """Convenience: total core loss (W) for either a scalar (B_peak,
    frequency) pair OR a (B_t, t) trace.

    Detects which mode by the type of `frequency_or_t`:
      * scalar → Steinmetz
      * array  → iGSE
    """
    if isinstance(frequency_or_t, (int, float)) and \
        isinstance(B, (int, float)):
        P_v = steinmetz_loss_density(float(B), float(frequency_or_t),
                                         material)
    else:
        P_v = igse_loss_density(B, frequency_or_t, material)
    return float(P_v * volume_m3)


# ---------------------------------------------------------------------------
# add-permeance-magnetic-domain 4.4: loss data -> in-loop loss elements
# ---------------------------------------------------------------------------

#: mean of |cos|^1.5 over a period, Gamma(1.25)/(sqrt(pi)*Gamma(1.75))
_MEAN_ABS_COS_1P5 = 0.5563852309207994


@dataclass(frozen=True)
class CoreLossElements:
    """Result of :func:`fit_core_loss_elements`.

    ``R_m`` [S] is the magnetic resistance (``add_magnetic_resistance``,
    classical eddy loss ``R_m·(dΦ/dt)²``) and ``c`` [A/V^0.5] the
    excess-loss coefficient of ``F = c·sign(u)·|u|^0.5``, ``u = dΦ/dt``, for
    the segment of the given geometry. ``k_h``, ``beta`` are the
    hysteresis term ``k_h·f·B^beta`` of the same fit: hysteresis belongs to
    the Jiles-Atherton law (``fit_ja_from_bh_curve``), not to these
    elements, and is reported only so the split is visible. ``k_e``,
    ``k_x`` are the sine-loss coefficients (``k_e·f²·B²``,
    ``k_x·f^1.5·B^1.5``, W/m³). ``residual`` is the largest relative error
    of the three-term model on the fitted points; ``fitted_on`` lists those
    points (index, f, B) — they are FITTED, and no agreement with them is
    a validation of the model.
    """
    R_m: float
    c: float
    k_h: float
    beta: float
    k_e: float
    k_x: float
    residual: float
    fitted_on: tuple
    area: float
    length: float


def _fit_error(option, message):
    """A typed `OptionsError` (a ValueError) naming the bad argument."""
    from ._pulsim import OptionsError  # type: ignore[import-not-found]
    err = OptionsError(message)
    err.option = option
    return err


def fit_core_loss_elements(f, B, P_v, *, geometry, tol=0.05,
                           beta_bounds=(1.0, 3.0)):
    """Fit the in-loop loss elements of a core segment to sine loss data.

    Model (Bertotti's split at sinusoidal flux of peak ``B`` and
    frequency ``f``, volumetric loss ``P_v`` in W/m³)::

        P_v = k_h·f·B^beta + k_e·f²·B² + k_x·f^1.5·B^1.5

    ``k_h, k_e, k_x >= 0`` by non-negative least squares on the relative
    error for each ``beta`` of a bounded 1-D search. For a segment of
    ``geometry = (area [m²], length [m])`` the elements are, with
    ``<(dB/dt)²> = 2π²f²B²`` and ``<|dB/dt|^1.5> = (2πfB)^1.5·0.5564``::

        R_m = k_e·length / (2π²·area)
        c   = k_x·length / ((2π)^1.5·0.5564·sqrt(area))

    Raises ``ValueError`` naming the cause when the inputs are not
    positive and finite, fewer than four points are given, or the
    residual exceeds ``tol``.
    """
    from scipy.optimize import minimize_scalar, nnls

    f = np.asarray(f, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    P = np.asarray(P_v, dtype=float).ravel()
    if not (f.shape == B.shape == P.shape):
        raise _fit_error("P_v", "fit_core_loss_elements: f, B and P_v must "
                         "have the same length.")
    if f.size < 4:
        raise _fit_error("f", "fit_core_loss_elements: needs at least four "
                         f"(f, B, P_v) points (got {f.size}).")
    for name, x in (("f", f), ("B", B), ("P_v", P)):
        if not np.all(np.isfinite(x) & (x > 0)):
            raise _fit_error(name, f"fit_core_loss_elements: {name} must be "
                             "positive and finite.")
    area, length = (float(v) for v in geometry)
    if not (area > 0 and length > 0 and np.isfinite(area)
            and np.isfinite(length)):
        raise _fit_error("geometry", "fit_core_loss_elements: geometry = "
                         "(area, length) must be positive and finite.")

    def solve(beta):
        A = np.column_stack((f * B ** beta, f ** 2 * B ** 2,
                             f ** 1.5 * B ** 1.5)) / P[:, None]
        k, _ = nnls(A, np.ones_like(P))
        return k, float(np.max(np.abs(A @ k - 1.0)))

    lo, hi = beta_bounds
    best = minimize_scalar(lambda b: float(np.sum(
        (solve(b)[0] @ np.vstack((f * B ** b, f ** 2 * B ** 2,
                                  f ** 1.5 * B ** 1.5)) / P - 1.0) ** 2)),
        bounds=(lo, hi), method="bounded",
        options={"xatol": 1e-12, "maxiter": 500})
    beta = float(best.x)
    (k_h, k_e, k_x), residual = solve(beta)
    if residual > tol:
        raise _fit_error(
            "tol", f"fit_core_loss_elements: the three-term model misses the data "
            f"by {residual:.3g} (relative) > tol = {tol}; the data are not a "
            f"hysteresis + eddy + excess split at sinusoidal flux.")
    R_m = k_e * length / (2.0 * np.pi ** 2 * area)
    c = k_x * length / ((2.0 * np.pi) ** 1.5 * _MEAN_ABS_COS_1P5
                        * np.sqrt(area))
    return CoreLossElements(
        R_m=float(R_m), c=float(c), k_h=float(k_h), beta=beta,
        k_e=float(k_e), k_x=float(k_x), residual=residual,
        fitted_on=tuple((int(i), float(f[i]), float(B[i]))
                        for i in range(f.size)),
        area=area, length=length)
