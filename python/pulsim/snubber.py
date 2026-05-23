"""Pulsim — snubber advisor for hard-switched topologies.

Closed-form sizing rules for the parasitic-resonance snubber that
sits across a switching device (MOSFET, IGBT, diode) in topologies
with loop inductance — boost, buck-boost, flyback, half-bridge with
inductive load, etc.

The fundamental problem: when the switch opens and the diode/body-
diode hasn't yet clamped the current, the loop inductance L together
with the parasitic device output capacitance C_oss form an LC tank.
The resulting overshoot is:

    V_overshoot = I_L · √(L / C_oss)

For a 100 µH inductor with 5 A peak and 10 nF C_oss, that's 500 V on
top of the bus — easily exceeding the device V_DS rating.

A damped RC snubber (R_snub in series with C_snub, both across the
switch) tames the ring. The sizing rules:

  * C_snub ≈ 5 · C_oss   — large enough to dominate the parasitic.
  * R_snub = ζ · 2 · √(L / C_snub)  — damping ratio ζ ∈ [0.4, 0.7]
    gives a critically-damped to slightly-underdamped response.
  * P_diss = 0.5 · C_snub · V_DC² · f_sw  — switching loss in R_snub.

This module exposes the formulas as one function and a higher-level
advisor that returns a dataclass with the recommended values plus
the predicted overshoot, switching loss, and a sanity warning if the
snubber's RC time constant exceeds half the switching period (it
won't fully charge/discharge each cycle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = [
    "SnubberRecommendation",
    "predict_overshoot",
    "recommend_snubber",
]


@dataclass
class SnubberRecommendation:
    """Sized RC snubber + predicted operating-point metrics."""
    C_snub_F: float
    R_snub_ohm: float
    V_overshoot_no_snubber_V: float
    V_overshoot_with_snubber_V: float
    P_diss_W: float
    damping_ratio: float
    tau_snub_s: float
    fits_in_period: bool
    warnings: list

    def summary(self) -> str:
        lines = []
        lines.append("  Recommended RC snubber:")
        lines.append(f"    C_snub        = {self.C_snub_F*1e9:.1f} nF")
        lines.append(f"    R_snub        = {self.R_snub_ohm:.1f} Ω")
        lines.append(f"    ζ (damping)   = {self.damping_ratio:.2f}")
        lines.append(f"    τ = R·C       = {self.tau_snub_s*1e9:.0f} ns")
        lines.append("")
        lines.append("  Predicted overshoot:")
        lines.append(f"    no snubber    = {self.V_overshoot_no_snubber_V:.1f} V")
        lines.append(f"    with snubber  ≈ {self.V_overshoot_with_snubber_V:.1f} V")
        lines.append("")
        lines.append("  Snubber dissipation:")
        lines.append(f"    P_diss = ½·C·V²·f_sw = {self.P_diss_W:.3f} W")
        if self.warnings:
            lines.append("")
            lines.append("  ⚠ Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def predict_overshoot(L_loop_H: float,
                         C_F: float,
                         I_peak_A: float) -> float:
    """Predict the LC-tank overshoot voltage:
       V = I · √(L / C)."""
    if C_F <= 0:
        return float("inf")
    return float(I_peak_A) * math.sqrt(L_loop_H / C_F)


def recommend_snubber(*,
                        L_loop_H: float,
                        C_oss_F: float,
                        V_bus_V: float,
                        I_peak_A: float,
                        f_sw_Hz: float,
                        damping_ratio: float = 0.5,
                        c_ratio: float = 5.0,
                        ) -> SnubberRecommendation:
    """Size an RC snubber for a given hard-switched topology.

    Parameters
    ----------
    L_loop_H
        Loop inductance seen by the commutating current (boost
        inductor + PCB traces). The dominant value.
    C_oss_F
        Parasitic output capacitance of the switching device
        (typically 10 pF – 10 nF for a SiC / Si MOSFET).
    V_bus_V
        DC bus voltage across the switch when it's off.
    I_peak_A
        Peak commutating current (typically the inductor's pp ripple
        amplitude, or the full inductor current for hard turn-off).
    f_sw_Hz
        Switching frequency.
    damping_ratio
        Target damping ratio for the LC ring (0.4–0.7 is typical;
        0.5 = light underdamping, 0.707 = critically damped).
    c_ratio
        How much larger C_snub should be relative to C_oss. Default
        5× — large enough to dominate the parasitic.

    Returns
    -------
    SnubberRecommendation
        Sized R_snub, C_snub, predicted overshoot, snubber losses,
        and any warnings about the operating regime.
    """
    if L_loop_H <= 0 or C_oss_F <= 0 or V_bus_V <= 0 or \
        I_peak_A <= 0 or f_sw_Hz <= 0:
        raise ValueError(
            "recommend_snubber: all inputs must be positive")

    C_snub = c_ratio * C_oss_F
    R_snub = float(damping_ratio) * 2.0 * math.sqrt(L_loop_H / C_snub)

    v_no_snub  = predict_overshoot(L_loop_H, C_oss_F, I_peak_A)
    v_with_snub = predict_overshoot(L_loop_H,
                                       C_oss_F + C_snub,
                                       I_peak_A)
    # With damping ζ, the overshoot factor is exp(-π·ζ/√(1-ζ²)) of
    # the undamped LC peak — apply that on top of the new LC value.
    if damping_ratio < 1.0:
        decay = math.exp(-math.pi * damping_ratio /
                            math.sqrt(1 - damping_ratio**2))
        v_with_snub *= decay

    p_diss = 0.5 * C_snub * (V_bus_V ** 2) * f_sw_Hz
    tau_snub = R_snub * C_snub
    T_sw = 1.0 / f_sw_Hz
    fits = tau_snub < T_sw / 2.0

    warnings = []
    if not fits:
        warnings.append(
            f"τ_snub = {tau_snub*1e9:.0f} ns > T_sw/2 = "
            f"{T_sw/2*1e9:.0f} ns — snubber may not fully "
            f"charge/discharge each switching cycle.")
    if p_diss > 5.0:
        warnings.append(
            f"P_diss = {p_diss:.1f} W is high — consider a "
            f"smaller C_snub or a different topology (e.g. "
            f"resonant ZVS).")
    if v_with_snub > V_bus_V * 1.5:
        warnings.append(
            f"Predicted overshoot ({v_with_snub:.0f} V) still "
            f"exceeds 1.5×V_bus — consider increasing c_ratio or "
            f"using an active clamp.")

    return SnubberRecommendation(
        C_snub_F=C_snub, R_snub_ohm=R_snub,
        V_overshoot_no_snubber_V=v_no_snub,
        V_overshoot_with_snubber_V=v_with_snub,
        P_diss_W=p_diss,
        damping_ratio=damping_ratio,
        tau_snub_s=tau_snub,
        fits_in_period=fits,
        warnings=warnings,
    )
