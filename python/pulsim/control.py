"""Control-loop helpers for Pulsim notebooks and scripts.

Provides an IMC-style auto-tuner that computes Kp / Ki for a
buck-converter plant from the operating-point parameters.

The standalone PI controller is the C++-backed `pulsim.PIController`
class (see `pulsim/bindings.cpp`). It has the same API as a textbook
PI (Kp, Ki, output_min, output_max, anti-windup) but runs at native
speed and works without the state-propagation quirks of the
`add_virtual_component("pi_controller", ...)` mixed-domain block.

Why prefer `ps.PIController` over the virtual block
----------------------------------------------------

The runtime `virtual_component("pi_controller", ...)` reads node
voltages from the state vector passed to `execute_mixed_domain_step`.
When that state vector comes back from a transient solve that wasn't
seeded with a DC operating point — the typical case for per-period
closed-loop notebooks — node voltages enforced by `voltage_source`
components are NOT propagated into the returned state. The virtual PI
then sees `signal = 0 − v_out` (inverted sign) and saturates at
`output_min`.

The standalone `ps.PIController` doesn't read circuit state at all —
you compute the error in Python, hand it to `update()`, and apply
the result. No hidden coupling, no surprises.

Quick usage
-----------

>>> import pulsim as ps
>>> pi = ps.PIController(Kp=0.02, Ki=10.0, output_min=0.10, output_max=0.50)
>>> # inside per-period loop:
>>> out = pi.update(error=v_ref - v_out, t=current_time)

For a buck-converter starting point with PM ≈ 60°:

>>> gains = ps.auto_tune_pi_buck(Vin=24, L=330e-6, C=220e-6, R=6.0,
...                              fsw=25e3, target_pm_deg=60.0)
>>> pi = ps.PIController(Kp=gains.kp, Ki=gains.ki,
...                      output_min=0.1, output_max=0.9)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = [
    "PIGains",
    "auto_tune_pi_buck",
]


@dataclass(frozen=True)
class PIGains:
    """Result of `auto_tune_pi_buck()` — the tuned PI parameters plus
    diagnostic info about how they were derived."""

    kp: float
    """Proportional gain (pass as Kp to `pulsim.PIController`)."""

    ki: float
    """Integral gain, per-second (pass as Ki to `pulsim.PIController`)."""

    fc_hz: float
    """Crossover frequency (Hz) the design was placed at."""

    f0_hz: float
    """LC resonance of the buck output filter (Hz)."""

    Q: float
    """Quality factor of the LC filter, `R·sqrt(C/L)`."""

    target_pm_deg: float
    """Phase margin target used to pick the crossover."""

    estimated_pm_deg: float
    """Estimated phase margin at fc, given the chosen Kp/Ki. Closed-form
    for a damped 2nd-order plant + PI compensator."""

    fz_pi_hz: float
    """Frequency of the PI's zero (Hz). Below fc for clean integrator
    action at the crossover."""


def auto_tune_pi_buck(
    *,
    Vin: float,
    L: float,
    C: float,
    R: float,
    fsw: float,
    target_pm_deg: float = 60.0,
    target_bandwidth_hz: float | None = None,
) -> PIGains:
    """Compute Kp / Ki for a voltage-mode CCM buck via IMC + Bode placement.

    Uses the closed-form small-signal control-to-output transfer
    function for an ideal buck:

        G_vd(s) = Vin / (1 + s·L/R + s²·L·C)

    Crossover placement:

    - If `target_bandwidth_hz` is supplied, fc = target_bandwidth_hz
      (clamped to fsw/5 maximum for Shannon sanity).
    - Otherwise fc is picked sub-resonance (fc = f0 / N) such that
      the PI compensator alone hits the requested phase margin.
      Conservative default — stable but slow response. For faster
      response use a Type-II / Type-III compensator (out of scope
      for this helper).

    Parameters
    ----------
    Vin : float
        Input voltage (V). Must be > 0.
    L : float
        Output inductor (H). Must be > 0.
    C : float
        Output capacitor (F). Must be > 0.
    R : float
        Load resistance (Ω). Must be > 0.
    fsw : float
        Switching frequency (Hz). Used as the upper sanity bound —
        fc is clamped to ≤ fsw/5 even if the caller asks for more.
    target_pm_deg : float
        Desired phase margin (degrees), in (0, 180). Default 60°.
    target_bandwidth_hz : float, optional
        Desired closed-loop bandwidth (Hz). When omitted, picked
        automatically from `target_pm_deg`.

    Returns
    -------
    PIGains
        `kp`, `ki`, `fc_hz`, `f0_hz`, `Q`, `target_pm_deg`,
        `estimated_pm_deg`, `fz_pi_hz`.

    Raises
    ------
    ValueError
        If any of Vin, L, C, R, fsw is non-positive, or if
        target_pm_deg is outside (0, 180).
    """
    if Vin <= 0 or L <= 0 or C <= 0 or R <= 0 or fsw <= 0:
        raise ValueError(
            f"auto_tune_pi_buck: all of Vin, L, C, R, fsw must be > 0 "
            f"(got Vin={Vin}, L={L}, C={C}, R={R}, fsw={fsw})"
        )
    if not (0.0 < target_pm_deg < 180.0):
        raise ValueError(
            f"auto_tune_pi_buck: target_pm_deg must be in (0, 180), "
            f"got {target_pm_deg}"
        )

    # LC filter natural frequency + Q
    omega_0 = 1.0 / math.sqrt(L * C)
    f0 = omega_0 / (2.0 * math.pi)
    Q = R * math.sqrt(C / L)

    # Pick fc. The PI alone gives:
    #   PI phase ≈ atan(fc/fz_pi) − 90°
    # For PM = target_pm_deg with a sub-resonance plant (phase ≈ 0°),
    # we need PI phase ≈ target_pm_deg − 180°. With fz_pi 1 decade
    # below fc the PI's phase boost is ~84° → effective PI phase ≈ -6°.
    # That gives PM dominated by plant phase + (-6°). For the buck
    # plant the phase rolls from 0° (DC) to -180° (above f0), so fc
    # must be sub-resonance.
    #
    # Empirical mapping (PM → fc/f0):
    #   PM 30° → fc ≈ f0       (aggressive)
    #   PM 45° → fc ≈ f0 / 2
    #   PM 60° → fc ≈ f0 / 3   (good default — stable, no oscillation)
    #   PM 75° → fc ≈ f0 / 5
    #   PM 90° → fc ≈ f0 / 10  (over-damped)
    if target_bandwidth_hz is not None:
        fc = float(target_bandwidth_hz)
    else:
        pm_to_ratio = {
            30: 1.0,
            45: 0.5,
            60: 1.0 / 3.0,
            75: 0.2,
            90: 0.1,
        }
        nearest = min(pm_to_ratio.keys(), key=lambda k: abs(k - target_pm_deg))
        fc = f0 * pm_to_ratio[nearest]

    # Sanity bounds: fc must be sub-Shannon and positive
    fc = min(fc, fsw / 5.0)
    fc = max(fc, f0 / 100.0)
    wc = 2.0 * math.pi * fc

    # Zero placement: 1 decade below fc → PI ≈ pure integrator at fc
    fz_pi = fc / 10.0
    wz_pi = 2.0 * math.pi * fz_pi

    # Plant magnitude at fc for an ideal buck:
    #   |Gvd(jωc)| = Vin / |1 + jωc·L/R - ωc²·L·C|
    L_R = L / R
    LC = L * C
    denom_re = 1.0 - wc * wc * LC
    denom_im = wc * L_R
    denom_mag = math.sqrt(denom_re * denom_re + denom_im * denom_im)
    G_mag_fc = Vin / denom_mag

    # Compensator magnitude at fc:
    #   |Gc(jωc)| = Kp · sqrt(1 + (fc/fz_pi)²) / (fc/fz_pi)
    # Solve Kp so |Gc · G| = 1:
    ratio = fc / fz_pi
    Gc_norm = math.sqrt(1.0 + ratio * ratio) / ratio
    Kp = 1.0 / (G_mag_fc * Gc_norm)

    # Ki from Kp and zero placement: Ki = Kp · ω_z
    Ki = Kp * wz_pi

    # PM estimate
    plant_phase_rad = math.atan2(-denom_im, denom_re)
    pi_phase_rad = math.atan2(wc, wz_pi) - math.pi / 2.0
    total_phase_deg = math.degrees(plant_phase_rad + pi_phase_rad)
    estimated_pm_deg = 180.0 + total_phase_deg

    return PIGains(
        kp=Kp,
        ki=Ki,
        fc_hz=fc,
        f0_hz=f0,
        Q=Q,
        target_pm_deg=target_pm_deg,
        estimated_pm_deg=estimated_pm_deg,
        fz_pi_hz=fz_pi,
    )
