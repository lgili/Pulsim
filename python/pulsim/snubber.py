"""Snubber-sizing advisor for hard-switched topologies in PWL Ideal mode.

Why this module exists
======================
When a MOSFET / IGBT runs in `SwitchingMode::Ideal` (PSIM-style PWL), the
device toggles between `g_on` and `g_off` discretely. With an inductor in
series with the switch (boost, buck-boost, flyback, half-bridge with motor
load, …), the *only* path for `I_L` during the brief MOSFET-OFF / diode-not-
yet-clamped interval is the parasitic output capacitance `C_oss`. The
backend solves the resulting L-C-switch network with Tustin (trapezoidal),
which has zero numerical damping — so the first solver step after the gate
edge captures the genuine LC half-cycle peak:

    V_overshoot ≈ I_L · √(L / C_oss)

For a 100 µH boost inductor with 5 A peak and the default 10 nF C_oss:

    V_overshoot ≈ 5 · √(1e-4 / 1e-8) = 500 V

…on top of the 400 V output bus — a 125 % spike. The fix is to size
`C_oss` (and optionally an R-C snubber) so the overshoot stays bounded
*and* the cap can charge to V_bus within the duty-cycle OFF time.

This module gives users the closed-form sizing rules + a one-shot diagnostic
that prints the predicted overshoot, switching loss, and recommended
parameter values for a given (L, I_peak, V_bus, dt) operating point.

It does *not* modify the circuit. The user reads the recommendation and
either:

    mp = ps.MOSFETParams()
    mp.C_oss = snubber.recommend_C_oss(L=100e-6, I_peak=5, V_bus=400)["C_oss"]

…or wires it via `Circuit.add_rc_snubber(...)` for damped variants.

Trade-off
=========
There is no single C_oss that wins on both axes for PWL Ideal:

  - Small C  → fast V_sw rise (cap charges to V_bus within OFF time),
              but large overshoot V = I·√(L/C).
  - Large C  → bounded overshoot, but slow V_sw rise can leave the
              cap below V_bus when the next gate edge arrives, breaking
              the boost transfer.

The right answer for a hard-switched topology with this solver is
**Behavioral mode** (smooth Shichman-Hodges), at the cost of a few more
Newton iterations per step. PWL Ideal stays valid for soft-switched /
resonant topologies where I_L is small at the commutation instant.

API
===
recommend_C_oss(L, I_peak, V_bus, *, max_overshoot_frac=0.2, f_sw=None)
    Closed-form C_oss sizing. Returns dict with predicted V_overshoot,
    rise-time, snubber loss estimate, and a feasibility verdict.

recommend_rc_snubber(L, C, *, V_bus=None, f_sw=None)
    R/C pair for critical damping of the L-C tank.

predict_overshoot(L, C, I_peak)
    Predicted V_overshoot for current operating point (no sizing).

advise(L, I_peak, V_bus, *, duty_off=0.75, f_sw=100e3, dt=100e-9,
       max_overshoot_frac=0.2)
    Print a human-readable report covering the trade-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "SnubberRecommendation",
    "RCSnubberRecommendation",
    "recommend_C_oss",
    "recommend_rc_snubber",
    "predict_overshoot",
    "advise",
]


@dataclass(frozen=True)
class SnubberRecommendation:
    """Output of `recommend_C_oss`. All quantities SI."""
    C_oss: float                  # Recommended parasitic cap (F)
    V_overshoot_predicted: float  # Peak V over V_bus at commutation (V)
    t_rise_to_V_bus: float        # Time for C to charge from 0 to V_bus (s)
    P_loss_estimate: float        # C·V²·f_sw, if f_sw provided (W); else nan
    feasible: bool                # True if t_rise < off-time AND overshoot OK
    notes: str                    # Human-readable explanation


@dataclass(frozen=True)
class RCSnubberRecommendation:
    """Output of `recommend_rc_snubber`. R-C series snubber across the switch."""
    R: float                      # Series resistance (Ω)
    C: float                      # Series capacitance (F)
    zeta: float                   # Damping ratio (1.0 = critical)
    P_dissipation_estimate: float # Per-cycle dissipation in R (W); nan if no f_sw


def predict_overshoot(L: float, C: float, I_peak: float) -> float:
    """Predicted V_sw overshoot above the clamp rail for a PWL Ideal switch
    fed by an inductor of inductance `L` and peak current `I_peak`, with
    parasitic output capacitance `C` across the switch terminals.

    The Tustin integrator is symplectic for ideal LC → no numerical damping.
    The first sample after the gate edge captures the LC half-cycle peak::

        V_overshoot = I_peak · √(L / C)

    This is the *worst case* for a single commutation. In practice the
    diode catches V_sw at V_bus + V_F when V_sw crosses that level, so
    the actual observed peak is `min(V_overshoot, V_bus + V_F + headroom)`.
    """
    if L <= 0 or C <= 0:
        raise ValueError(f"L and C must be > 0 (got L={L}, C={C})")
    return abs(I_peak) * math.sqrt(L / C)


def recommend_C_oss(L: float,
                    I_peak: float,
                    V_bus: float,
                    *,
                    max_overshoot_frac: float = 0.2,
                    f_sw: Optional[float] = None,
                    duty_off: float = 0.75) -> SnubberRecommendation:
    """Closed-form C_oss sizing for a hard-switched PWL Ideal converter.

    Picks the smallest C_oss that keeps the Tustin LC-overshoot below
    `max_overshoot_frac · V_bus`::

        V_overshoot_max = max_overshoot_frac · V_bus
        C_oss          = (I_peak / V_overshoot_max)² · L

    Then sanity-checks that the cap can charge from 0 to V_bus during the
    OFF interval (linear approximation, constant I_peak)::

        t_rise = C_oss · V_bus / I_peak

    For an idealised boost the OFF interval is `duty_off / f_sw`. If
    `t_rise > duty_off / f_sw`, the cap can't fully charge between
    edges — the PWL solver will saturate V_sw below V_bus and the boost
    transfer breaks. The recommendation is then flagged `feasible=False`
    and the caller should drop to Behavioral mode (or switch to a smooth
    body-diode + clamp model).

    Parameters
    ----------
    L : float
        Inductance in series with the switch (H).
    I_peak : float
        Peak inductor current at commutation (A). Use the design-point
        peak — for a boost in CCM, this is `I_avg + ΔI/2`.
    V_bus : float
        Voltage the switch must block in the OFF state (V).
    max_overshoot_frac : float, default 0.2
        Tolerated V_sw overshoot as fraction of V_bus. 0.2 = 20 % over the
        bus is typical for a power-stage with proper clamping.
    f_sw : float, optional
        Switching frequency (Hz). When provided, the report includes
        a per-cycle switching-loss estimate `P = C·V²·f_sw`.
    duty_off : float, default 0.75
        Fraction of the switching period the MOSFET is OFF. Boost in CCM
        with `D_on = 0.25` gives `duty_off = 0.75`.

    Returns
    -------
    SnubberRecommendation
        Dataclass with the recommended `C_oss`, predicted overshoot,
        rise time, optional loss estimate, and a feasibility flag.
    """
    if L <= 0 or I_peak <= 0 or V_bus <= 0:
        raise ValueError("L, I_peak, V_bus must all be > 0")
    if max_overshoot_frac <= 0:
        raise ValueError("max_overshoot_frac must be > 0")

    V_overshoot_max = max_overshoot_frac * V_bus
    C_oss = (I_peak / V_overshoot_max) ** 2 * L

    V_pred = predict_overshoot(L, C_oss, I_peak)
    t_rise = C_oss * V_bus / I_peak

    P_loss = float("nan")
    feasible = True
    if f_sw is not None and f_sw > 0:
        # Standard "C_oss switching loss" for a hard-switched converter:
        # the cap charges to V_bus during OFF-time (energy stored = ½·C·V²)
        # and dissipates that stored energy through the MOSFET when it
        # turns back ON. Net loss per switching cycle = ½·C·V².
        P_loss = 0.5 * C_oss * V_bus ** 2 * f_sw
        t_off = duty_off / f_sw
        if t_rise > t_off:
            feasible = False

    notes_parts = []
    notes_parts.append(
        f"V_overshoot predicted = I·√(L/C) = "
        f"{I_peak:.2f}·√({L*1e6:.0f}µH/{C_oss*1e9:.0f}nF) = {V_pred:.0f} V "
        f"({V_pred/V_bus*100:.0f} % of V_bus)"
    )
    notes_parts.append(
        f"t_rise (linear, constant I_peak) = C·V_bus/I = {t_rise*1e9:.0f} ns"
    )
    if f_sw is not None:
        notes_parts.append(
            f"OFF interval at D={1-duty_off:.0%}, f_sw={f_sw/1e3:.0f} kHz = "
            f"{duty_off/f_sw*1e9:.0f} ns"
        )
        notes_parts.append(
            f"P_loss (½·C·V²·f_sw, hard-switched) = "
            f"½·{C_oss*1e9:.0f}nF·{V_bus}²·{f_sw/1e3:.0f}kHz = {P_loss:.2f} W"
        )
        if not feasible:
            notes_parts.append(
                "⚠ t_rise > OFF interval — V_sw won't reach V_bus before "
                "next ON edge. Boost transfer will fail. Recommend dropping "
                "to Behavioral mode (set Eon_25 = Eoff_25 = 0)."
            )

    return SnubberRecommendation(
        C_oss=C_oss,
        V_overshoot_predicted=V_pred,
        t_rise_to_V_bus=t_rise,
        P_loss_estimate=P_loss,
        feasible=feasible,
        notes="\n  ".join(notes_parts),
    )


def recommend_rc_snubber(L: float,
                          C: float,
                          *,
                          V_bus: Optional[float] = None,
                          f_sw: Optional[float] = None,
                          zeta: float = 1.0) -> RCSnubberRecommendation:
    """R/C pair for a series R-C snubber across the switch, sized to give
    damping ratio `zeta` (1.0 = critical) of the L-C tank.

    For a series-R, series-C snubber across the switch terminals, with
    the inductor providing the loop inductance::

        ω_n = 1 / √(L·C)
        R   = 2·ζ·√(L/C)

    Per-cycle dissipation in R (cap charges to V_bus once and discharges
    once)::

        P_R = C · V_bus² · f_sw

    Note: this snubber is most effective when the LC period exceeds a few
    `dt` (so Tustin can resolve the ring). For dt << LC period, the snubber
    barely affects the *first-step* overshoot — that's bounded by C_oss
    sizing, not by R.

    Parameters
    ----------
    L : float
        Inductance in series with the switch (H).
    C : float
        Snubber capacitance (F). Typically same order as C_oss (1 – 100 nF).
    V_bus : float, optional
        Bus voltage for loss estimation.
    f_sw : float, optional
        Switching frequency for loss estimation.
    zeta : float, default 1.0
        Target damping ratio. 1.0 = critical, 0.7 = lightly underdamped.
    """
    if L <= 0 or C <= 0:
        raise ValueError("L and C must be > 0")
    if zeta <= 0:
        raise ValueError("zeta must be > 0")

    R = 2.0 * zeta * math.sqrt(L / C)
    P_R = float("nan")
    if V_bus is not None and f_sw is not None and V_bus > 0 and f_sw > 0:
        # ½·C·V² stored each commutation, dissipated through R every
        # discharge → P_R = ½·C·V²·f_sw for a hard-switched converter.
        P_R = 0.5 * C * V_bus ** 2 * f_sw

    return RCSnubberRecommendation(R=R, C=C, zeta=zeta,
                                    P_dissipation_estimate=P_R)


def advise(L: float,
           I_peak: float,
           V_bus: float,
           *,
           duty_off: float = 0.75,
           f_sw: float = 100e3,
           dt: float = 100e-9,
           max_overshoot_frac: float = 0.2,
           file=None) -> None:
    """Print a human-readable trade-off report for a PWL Ideal switching
    cell with `L` in series with the switch. Use this *before* picking
    parameters to understand whether PWL Ideal is even appropriate for
    your topology, or whether you should drop to Behavioral mode.

    Example
    -------
        >>> from pulsim import snubber
        >>> snubber.advise(L=100e-6, I_peak=5.0, V_bus=400.0)
        Boost-class PWL snubber-sizing report
        -------------------------------------
        L = 100 µH   I_peak = 5.0 A   V_bus = 400 V
        f_sw = 100 kHz   dt = 100 ns   D_on = 25 %
        ...
    """
    rec = recommend_C_oss(L, I_peak, V_bus,
                           max_overshoot_frac=max_overshoot_frac,
                           f_sw=f_sw, duty_off=duty_off)
    snub = recommend_rc_snubber(L, rec.C_oss, V_bus=V_bus, f_sw=f_sw)
    T_LC = 2 * math.pi * math.sqrt(L * rec.C_oss)

    out = file if file is not None else __import__("sys").stdout

    def p(*args, **kwargs):
        print(*args, file=out, **kwargs)

    p("Boost-class PWL snubber-sizing report")
    p("-------------------------------------")
    p(f"L = {L*1e6:.1f} µH   I_peak = {I_peak:.2f} A   V_bus = {V_bus:.0f} V")
    p(f"f_sw = {f_sw/1e3:.0f} kHz   dt = {dt*1e9:.0f} ns   "
      f"D_on = {(1-duty_off)*100:.0f} %")
    p(f"target overshoot ≤ {max_overshoot_frac*100:.0f} % of V_bus = "
      f"{max_overshoot_frac*V_bus:.0f} V")
    p()
    # Reality-check the C_oss = 10 nF default the runtime auto-installs
    # when Eon_25 > 0. Even when our recommendation flags this regime as
    # infeasible, users should *see* what the default does so they can
    # interpret the simulation waveforms they get.
    V_default = predict_overshoot(L, 10e-9, I_peak)
    P_default = 0.5 * 10e-9 * V_bus * V_bus * f_sw
    p("For reference, the runtime's default C_oss = 10 nF predicts:")
    p(f"  V_overshoot = {V_default:.0f} V ({V_default/V_bus*100:.0f} % of V_bus)")
    p(f"  P_loss      = {P_default:.2f} W (½·C·V²·f_sw, hard-switched)")
    p()
    p("Recommendation:")
    p(f"  MOSFETParams.C_oss = {rec.C_oss*1e9:.1f} nF")
    p(f"  {rec.notes}")
    p()
    p(f"LC period for that C_oss = 2π·√(L·C) = {T_LC*1e6:.2f} µs "
      f"(= {T_LC/dt:.0f} · dt)")
    if T_LC < 4 * dt:
        p("  → LC period < 4·dt: Tustin can't resolve the ring; expect "
          "first-step overshoot at the I·√(L/C) bound.")
    else:
        p("  → LC period > 4·dt: Tustin captures the ring; a damped "
          "R-C snubber can help.")
    p()
    p("Optional R-C snubber across the switch (critical damping):")
    p(f"  R = 2·√(L/C) = {snub.R:.1f} Ω")
    p(f"  C = {snub.C*1e9:.1f} nF (same as C_oss)")
    if not math.isnan(snub.P_dissipation_estimate):
        p(f"  P_R (avg, ½·C·V²·f_sw) = "
          f"{snub.P_dissipation_estimate:.2f} W")
    p()
    if not rec.feasible:
        p("VERDICT: PWL Ideal NOT recommended for this operating point.")
        p("         t_rise > OFF interval — V_sw can't reach V_bus before")
        p("         the next gate edge. Drop to Behavioral mode:")
        p("             mp.Eon_25 = 0.0")
        p("             mp.Eoff_25 = 0.0")
        p("         and let smooth Shichman-Hodges handle the LC dynamics.")
    else:
        p("VERDICT: PWL Ideal feasible at this operating point.")
