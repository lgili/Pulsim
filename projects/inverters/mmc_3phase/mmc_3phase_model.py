"""3-φ MMC DC/AC inverter — analytical helpers + plant builders.

This module is imported by the notebook(s) in this folder:

  * ``01_mmc_validation_gean.ipynb`` — model the MMC, design open-loop
    modulation, simulate with pulsim, and validate against the
    experimental run from Section 4.1 of Sousa's thesis
    (UFSC PhD, 2022).

The thesis prototype runs at:

  * ``S = 15 kVA``, ``V_cc = 640 V`` (DC bus), ``V̂ = 272 V`` (AC peak)
  * ``N = 5`` SMs per arm, ``V_CSM = 128 V`` (each cap target)
  * ``C_SM = 470 µF`` per submodule
  * ``L_b ≈ 1 mH`` (typical, used as a tuning knob below)
  * RL load (Y-connected): ``R_load = 9.75 Ω``, ``L_load = 2.8 mH``
    (R nominal 9.2 Ω + parasitic; L nominal 1.9 mH + 0.9 mH parasitic)
  * ``T_d = T_m = 5 µs`` (dead-time and minimum pulse width)
  * ``f_grid = 60 Hz``
  * ``M = 0.85`` (modulation depth ⇒ V̂ = M·V_cc/2 = 272 V)
  * Modulation: In-Phase Disposition (IPD)

The thesis paper reports the following key metrics (Tabela 4.2,
open-loop run, 1 µs sim step):

  * ``THD(i_a) = 0.706 %`` (with dead-time, sim 1)
  * ``RMS(i_ca) = 4.55 A``  (circulating-current, with dead-time)
  * ``RMS(CA(i_cc)) = 1.14 A``  (DC-bus current AC component)

Pulsim's L1 PS-PWM modulator is NOT IPD, so the *instantaneous*
switching waveforms differ from the thesis figures. However the
*averaged* quantities — cap-voltage means and ripples, AC port-current
RMS, DC-bus mean — should match within a few percent. The notebook
walks through this comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt
from typing import Callable

import numpy as np

import pulsim as p


# =============================================================================
# Operating point — values from Tabela 4.1 / Section 4.1 of the thesis
# =============================================================================


@dataclass(frozen=True)
class GeanThesisParams:
    """Open-loop operating point from Section 4.1 of Sousa (2022)."""

    # DC + AC ratings
    V_dc: float = 640.0          # DC bus voltage [V]
    V_ac_peak: float = 272.0     # target AC phase-voltage peak [V]
    f_grid: float = 60.0         # AC fundamental [Hz]

    # MMC topology
    n_sm: int = 5                # SMs per arm
    c_sm: float = 470e-6         # per-SM capacitance [F]
    l_b: float = 1.0e-3          # arm inductance [H] (tuning knob —
                                 #  thesis prototype value not pinned
                                 #  in the open-loop section)
    r_b: float = 0.675           # arm-side parasitic resistance [Ω]
                                 #  (from Section 4.1 — used by sim 1
                                 #  to match the experimental damping)

    # AC-side RL load (Y-connected, per phase)
    r_load: float = 9.75         # [Ω]  (9.2 Ω nominal + parasitic)
    l_load: float = 2.8e-3       # [H]  (1.9 mH nominal + 0.9 mH parasitic)

    # IGBT non-idealities
    t_dead: float = 5e-6         # dead-time [s]
    t_min: float = 5e-6          # minimum pulse width [s]

    # Modulation
    m_depth: float = 0.85        # M = V̂ / (V_dc / 2)
    f_carrier: float = 1800.0    # carrier per SM [Hz]
    modulation_scheme: str = "ipd"  # "ipd" (thesis) or "ps_pwm"

    # Initial conditions
    v_c0: float | None = None    # capacitor-sum IC; default = V_dc.

    @property
    def v_c_init(self) -> float:
        return self.V_dc if self.v_c0 is None else float(self.v_c0)

    @property
    def omega_grid(self) -> float:
        return 2.0 * pi * self.f_grid


# =============================================================================
# Reference signals — IPD-equivalent sinusoidal modulation for HB MMC
# =============================================================================


def make_phase_mref_fns(
    params: GeanThesisParams,
) -> "tuple[Callable[[float], float], Callable[[float], float], Callable[[float], float]]":
    """Return three modulation references ``(m_a_p_ref, m_b_p_ref, m_c_p_ref)``
    for the *upper* arms of an MMC inverter in open-loop operation.

    Convention (matches our 3-φ MMC integration tests):

      * Upper-arm ``m_X_p = 0.5 − v_X_ac / V_dc``
      * Lower-arm ``m_X_n = 0.5 + v_X_ac / V_dc``  (1 − m_X_p for HB)

    where ``v_X_ac`` is the desired AC phase voltage relative to the
    bus midpoint. For a balanced 3-φ sinusoidal output:

      ``v_X_ac(t) = (M · V_dc / 2) · cos(ω·t − φ_X)``

    with ``φ_a = 0``, ``φ_b = 2π/3``, ``φ_c = 4π/3``.
    """
    omega = params.omega_grid
    v_peak = 0.5 * params.m_depth * params.V_dc  # = M · V_dc / 2

    def m_a_p(t: float) -> float:
        v_a = v_peak * cos(omega * t)
        return 0.5 - v_a / params.V_dc

    def m_b_p(t: float) -> float:
        v_b = v_peak * cos(omega * t - 2.0 * pi / 3.0)
        return 0.5 - v_b / params.V_dc

    def m_c_p(t: float) -> float:
        v_c = v_peak * cos(omega * t + 2.0 * pi / 3.0)
        return 0.5 - v_c / params.V_dc

    return m_a_p, m_b_p, m_c_p


# =============================================================================
# Plant builders — three layers of fidelity
# =============================================================================


@dataclass
class MmcPlant:
    """Bundle of (builder, six-arm list, filter-inductor indices)
    returned by the plant builders below."""

    builder: object
    arms: list[object] = field(default_factory=list)
    iL_indices: tuple[int, int, int] = (0, 0, 0)


def build_l1_plant(params: GeanThesisParams) -> MmcPlant:
    """3-φ MMC inverter using L1 PS-PWM multilevel arms (no dead-time)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", params.V_dc)

    m_a, m_b_, m_c = make_phase_mref_fns(params)

    arm_params = p.MmcArmMultilevelParams(
        n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
        f_carrier=params.f_carrier,
        modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
    )

    arms: list[object] = []
    # Upper arms + upper arm inductors.
    upper_refs = (m_a, m_b_, m_c)
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params, m_ref=upper_refs[k],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        b.add_resistor(f"Rb_{ph}_p", f"rb_{ph}_p", f"ac_{ph}", params.r_b)

    # Lower arm inductors + lower arms (complement modulation).
    def _complement(f):
        return lambda t, _f=f: 1.0 - float(_f(t))

    lower_refs = tuple(_complement(f) for f in upper_refs)
    for k, ph in enumerate("abc"):
        b.add_resistor(f"Rb_{ph}_n", f"ac_{ph}", f"rb_{ph}_n", params.r_b)
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="gnd",
            params=arm_params, m_ref=lower_refs[k],
        )
        arms.append(arm_n)

    # Y-connected RL load.
    iL_indices: list[int] = []
    for ph in "abc":
        l_id = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_id, b.graph),
        )
    # Weak star tie for MNA conditioning.
    b.add_resistor("R_star", "star", "gnd", 1e6)

    return MmcPlant(builder=b, arms=arms,
                     iL_indices=(iL_indices[0], iL_indices[1], iL_indices[2]))


def build_l2_plant(params: GeanThesisParams) -> MmcPlant:
    """3-φ MMC inverter using L2 SM-equivalent arms (dead-time aware)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", params.V_dc)

    m_a, m_b_, m_c = make_phase_mref_fns(params)

    arm_params = p.MmcArmEquivalentParams(
        n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
        f_carrier=params.f_carrier,
        t_dead=params.t_dead, t_min=params.t_min,
        modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
    )

    arms: list[object] = []
    upper_refs = (m_a, m_b_, m_c)
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_equivalent(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params, m_ref=upper_refs[k],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        b.add_resistor(f"Rb_{ph}_p", f"rb_{ph}_p", f"ac_{ph}", params.r_b)

    def _complement(f):
        return lambda t, _f=f: 1.0 - float(_f(t))

    lower_refs = tuple(_complement(f) for f in upper_refs)
    for k, ph in enumerate("abc"):
        b.add_resistor(f"Rb_{ph}_n", f"ac_{ph}", f"rb_{ph}_n", params.r_b)
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = p.add_mmc_arm_equivalent(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="gnd",
            params=arm_params, m_ref=lower_refs[k],
        )
        arms.append(arm_n)

    iL_indices: list[int] = []
    for ph in "abc":
        l_id = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_id, b.graph),
        )
    b.add_resistor("R_star", "star", "gnd", 1e6)

    return MmcPlant(builder=b, arms=arms,
                     iL_indices=(iL_indices[0], iL_indices[1], iL_indices[2]))


# =============================================================================
# IGBT-aware plant builders (Phase 20.18) — anti-parallel SwitchedDiode pair
# =============================================================================
#
# Goal: model the **non-linear conduction loss** of an MMC arm where N
# IGBT/diode pairs sit in series. At any instant exactly N switches
# conduct in the same direction; aggregating per arm:
#
#   V_F0_aggr = N · V_CE_sat   (knee voltage of N IGBTs in series)
#   R_d_aggr  = N · R_CE_sat   (on-state resistance, summed)
#
# Implementation: replace the lumped ``R_b`` resistor with an
# **anti-parallel pair of SwitchedDiodes** (``b.add_diode``, the
# *linear* binary-state model with ``V_th``, ``g_on`` and ``g_off``).
# SwitchedDiode is event-detected (Layer 4 V3 / Phase 20.14), not
# smooth-blend, so:
#
#   * the Jacobian stays well-conditioned (it's piecewise linear);
#   * the Newton solver doesn't need to iterate inside a step — each
#     state is just an LTI region;
#   * commutation events (zero-crossing of i_arm) are detected by
#     the kernel and the substep-state-correction path is used to
#     resolve them at sub-dt resolution.
#
# We *also* tried ``b.add_nonlinear_diode`` (smooth-blend IdealDiode),
# but ``solve_with_newton`` reports a numerically singular combined
# matrix at the first transient step — the smooth-blend Jacobian
# collapses when both diodes of the pair are in their off-state
# simultaneously (the natural condition at DC OP). The linear
# SwitchedDiode sidesteps this entirely because there's no Newton
# iteration on its branch — only event detection.


def igbt_equivalent_r_b(
    *, n_sm: int, V_CE_sat: float, R_CE_sat: float, I_op: float,
) -> float:
    """Linear-equivalent arm resistance for an N-SM half-bridge whose
    IGBTs have ``V_CE_sat`` knee and ``R_CE_sat`` on-state slope per
    switch, sized to match the peak voltage drop at operating peak
    current ``I_op``.

    Returns:
        ``R_b_eq = N · R_CE_sat + N · V_CE_sat / I_op``  [Ω]

    Use the returned value in :class:`GeanThesisParams` and pass to
    :func:`build_l1_plant` / :func:`build_l2_plant` to model an MMC
    with IGBT-level-1 conduction physics, without needing pulsim's
    nonlinear-diode infrastructure.

    See :func:`build_l1_plant_igbt` / :func:`build_l2_plant_igbt`
    for the *non-linear* alternative that captures the V_F0 step at
    each zero-crossing of i_arm.
    """
    if n_sm <= 0:
        raise ValueError(f"n_sm must be positive (got {n_sm})")
    if I_op <= 0:
        raise ValueError(f"I_op must be positive (got {I_op})")
    return n_sm * R_CE_sat + n_sm * V_CE_sat / I_op


@dataclass(frozen=True)
class GeanThesisIgbtParams(GeanThesisParams):
    """Extension of :class:`GeanThesisParams` adding physical IGBT
    level-1 conduction-loss parameters per SM.

    The per-arm conduction is modeled as a pair of anti-parallel
    SwitchedDiodes between the arm inductor and the AC node:

      * V_F0_aggr = ``n_sm · V_CE_sat_per_sm`` — the knee voltage
        (``V_th`` of the SwitchedDiode model);
      * R_d_aggr  = ``n_sm · R_CE_sat_per_sm`` — the on-state
        slope (``g_on = 1 / R_d_aggr``);
      * g_off (off-state conductance) = ``g_on · 1e-4`` by
        default, so the blocking diode looks like a ~10 kΩ leak.

    Typical IGBT level-1 parameters (FF150R12KE3 class — used in
    Sousa's 15-kVA prototype):

      * V_CE_sat ≈ 1.5–2.5 V
      * R_CE_sat ≈ 30–80 mΩ

    The base ``r_b`` is set to 0 by default — the diode pair *is*
    the arm parasitic. Set ``r_b > 0`` to add a small residual
    linear series resistance for extra damping.
    """

    V_CE_sat_per_sm: float = 1.5      # IGBT saturation voltage [V]
    R_CE_sat_per_sm: float = 0.05     # IGBT on-state slope [Ω]
    g_off_ratio: float = 1e-4         # off/on conductance ratio
    r_b: float = 0.0                  # override base default (no residual)


def _add_arm_diode_pair_switched(
    b: object,
    *,
    name_prefix: str,
    node_in: str,
    node_out: str,
    V_F0_aggr: float,
    g_on: float,
    g_off: float,
) -> None:
    """Insert an anti-parallel pair of SwitchedDiodes between
    ``node_in`` and ``node_out`` modeling bidirectional IGBT-arm
    conduction.

    Forward diode: ``node_in → node_out`` (knee at V_F0_aggr).
    Reverse diode: ``node_out → node_in`` (same knee on the other
    side).
    """
    b.add_diode(  # type: ignore[attr-defined]
        f"D_F_{name_prefix}", node_in, node_out, g_on, g_off, V_F0_aggr,
    )
    b.add_diode(  # type: ignore[attr-defined]
        f"D_R_{name_prefix}", node_out, node_in, g_on, g_off, V_F0_aggr,
    )


def build_l1_plant_igbt(params: GeanThesisIgbtParams) -> MmcPlant:
    """L1 MMC plant with **anti-parallel SwitchedDiode pair** in each
    arm (modeling N IGBTs in series with V_CE_sat + R_CE_sat), replacing
    the lumped ``R_b`` resistor of :func:`build_l1_plant`.

    Optionally also inserts a residual ``r_b`` resistor in series if
    ``params.r_b > 0`` (useful to capture parasitic interconnect that
    isn't part of the semiconductor itself).
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", params.V_dc)

    m_a, m_b_, m_c = make_phase_mref_fns(params)

    arm_params = p.MmcArmMultilevelParams(
        n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
        f_carrier=params.f_carrier,
        modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
    )

    V_F0_aggr = params.n_sm * params.V_CE_sat_per_sm
    R_d_aggr  = params.n_sm * params.R_CE_sat_per_sm
    g_on  = 1.0 / R_d_aggr if R_d_aggr > 0 else 1.0
    g_off = g_on * params.g_off_ratio

    arms: list[object] = []
    upper_refs = (m_a, m_b_, m_c)
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params, m_ref=upper_refs[k],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        if params.r_b > 0:
            b.add_resistor(
                f"Rb_{ph}_p", f"rb_{ph}_p", f"dpre_{ph}_p", params.r_b,
            )
            diode_in = f"dpre_{ph}_p"
        else:
            diode_in = f"rb_{ph}_p"
        _add_arm_diode_pair_switched(
            b, name_prefix=f"{ph}_p",
            node_in=diode_in, node_out=f"ac_{ph}",
            V_F0_aggr=V_F0_aggr, g_on=g_on, g_off=g_off,
        )

    def _complement(f):
        return lambda t, _f=f: 1.0 - float(_f(t))

    lower_refs = tuple(_complement(f) for f in upper_refs)
    for k, ph in enumerate("abc"):
        if params.r_b > 0:
            _add_arm_diode_pair_switched(
                b, name_prefix=f"{ph}_n",
                node_in=f"ac_{ph}", node_out=f"dpre_{ph}_n",
                V_F0_aggr=V_F0_aggr, g_on=g_on, g_off=g_off,
            )
            b.add_resistor(
                f"Rb_{ph}_n", f"dpre_{ph}_n", f"rb_{ph}_n", params.r_b,
            )
        else:
            _add_arm_diode_pair_switched(
                b, name_prefix=f"{ph}_n",
                node_in=f"ac_{ph}", node_out=f"rb_{ph}_n",
                V_F0_aggr=V_F0_aggr, g_on=g_on, g_off=g_off,
            )
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="gnd",
            params=arm_params, m_ref=lower_refs[k],
        )
        arms.append(arm_n)

    iL_indices: list[int] = []
    for ph in "abc":
        l_id = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_id, b.graph),
        )
    b.add_resistor("R_star", "star", "gnd", 1e6)

    return MmcPlant(builder=b, arms=arms,
                     iL_indices=(iL_indices[0], iL_indices[1], iL_indices[2]))


def build_l2_plant_igbt(params: GeanThesisIgbtParams) -> MmcPlant:
    """L2 (dead-time aware) MMC plant with anti-parallel SwitchedDiode
    pair per arm (V_CE_sat + R_CE_sat physics)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", params.V_dc)

    m_a, m_b_, m_c = make_phase_mref_fns(params)

    arm_params = p.MmcArmEquivalentParams(
        n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
        f_carrier=params.f_carrier,
        t_dead=params.t_dead, t_min=params.t_min,
        modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
    )

    V_F0_aggr = params.n_sm * params.V_CE_sat_per_sm
    R_d_aggr  = params.n_sm * params.R_CE_sat_per_sm
    g_on  = 1.0 / R_d_aggr if R_d_aggr > 0 else 1.0
    g_off = g_on * params.g_off_ratio

    arms: list[object] = []
    upper_refs = (m_a, m_b_, m_c)
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_equivalent(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params, m_ref=upper_refs[k],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        if params.r_b > 0:
            b.add_resistor(
                f"Rb_{ph}_p", f"rb_{ph}_p", f"dpre_{ph}_p", params.r_b,
            )
            diode_in = f"dpre_{ph}_p"
        else:
            diode_in = f"rb_{ph}_p"
        _add_arm_diode_pair_switched(
            b, name_prefix=f"{ph}_p",
            node_in=diode_in, node_out=f"ac_{ph}",
            V_F0_aggr=V_F0_aggr, g_on=g_on, g_off=g_off,
        )

    def _complement(f):
        return lambda t, _f=f: 1.0 - float(_f(t))

    lower_refs = tuple(_complement(f) for f in upper_refs)
    for k, ph in enumerate("abc"):
        if params.r_b > 0:
            _add_arm_diode_pair_switched(
                b, name_prefix=f"{ph}_n",
                node_in=f"ac_{ph}", node_out=f"dpre_{ph}_n",
                V_F0_aggr=V_F0_aggr, g_on=g_on, g_off=g_off,
            )
            b.add_resistor(
                f"Rb_{ph}_n", f"dpre_{ph}_n", f"rb_{ph}_n", params.r_b,
            )
        else:
            _add_arm_diode_pair_switched(
                b, name_prefix=f"{ph}_n",
                node_in=f"ac_{ph}", node_out=f"rb_{ph}_n",
                V_F0_aggr=V_F0_aggr, g_on=g_on, g_off=g_off,
            )
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = p.add_mmc_arm_equivalent(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="gnd",
            params=arm_params, m_ref=lower_refs[k],
        )
        arms.append(arm_n)

    iL_indices: list[int] = []
    for ph in "abc":
        l_id = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_id, b.graph),
        )
    b.add_resistor("R_star", "star", "gnd", 1e6)

    return MmcPlant(builder=b, arms=arms,
                     iL_indices=(iL_indices[0], iL_indices[1], iL_indices[2]))


# =============================================================================
# Lumped-R_b equivalent helper (kept as an alternative, simpler approach)
# =============================================================================
#
# A physically motivated alternative to a manually-tuned ``R_b`` is to
# derive it from the per-SM IGBT level-1 conduction parameters
# (``V_CE_sat``, ``R_CE_sat``). For an arm with N SMs in series, the
# instantaneous conduction drop is approximately:
#
#   v_drop(i) ≈ N · V_CE_sat · sign(i) + N · R_CE_sat · i
#
# A linear-equivalent ``R_b`` that delivers the SAME peak voltage drop
# at a chosen operating peak current ``I_op``:
#
#   R_b_eq = N · R_CE_sat + N · V_CE_sat / I_op
#
# This formula understates the V_F0 step-at-zero-crossings (no jump at
# i = 0), so it can't predict the small zero-crossing distortion that
# real IGBTs introduce. But it correctly captures the *peak* voltage
# drop and *RMS power dissipation* — the dominant effect that bumps
# the AC current peak down. Sousa's thesis Sec 4.1 ``R_b = 0.675 Ω``
# is exactly this kind of one-knob lump.
#
# Notebook 04 ("MMC com IGBT level-1") uses this formula to sweep
# realistic IGBT-equivalent ``R_b`` values and compare the impact on
# THD, peak/RMS current and v_C ripple against Sousa's Tabela 4.2.
# It is also kept as a simpler alternative to the SwitchedDiode-pair
# physical model (above) for users who don't want to incur the
# extra branch count from 12 added switched diodes.


# =============================================================================
# Run drivers + metrics
# =============================================================================


@dataclass
class MmcRunResult:
    """Output of :func:`run_mmc_open_loop`."""

    t: np.ndarray
    i_a: np.ndarray
    i_b: np.ndarray
    i_c: np.ndarray
    v_b_a_p: np.ndarray              # arm-generated voltage, phase a upper
    v_C: np.ndarray                  # shape (6, n_samples) — per-arm cap sums
    arm_names: tuple[str, ...] = (
        "a_p", "b_p", "c_p", "a_n", "b_n", "c_n",
    )


def run_mmc_open_loop(
    plant: MmcPlant,
    *,
    t_end: float = 50e-3,
    dt: float = 5e-6,
    layer: str = "l1",
) -> MmcRunResult:
    """Run a plant produced by :func:`build_l1_plant` or
    :func:`build_l2_plant` for ``t_end`` seconds at ``dt`` step.

    Args:
        plant: Output of one of the ``build_*_plant`` helpers.
        t_end: Simulation horizon [s].
        dt: Fixed time step [s].
        layer: ``"l1"`` or ``"l2"`` — which observer factory to use.

    Returns:
        :class:`MmcRunResult` with the logged time series.
    """
    if layer == "l1":
        obs, bex = p.make_mmc_arm_multilevel_observers(
            plant.builder, plant.arms, dt=dt,  # type: ignore[arg-type]
        )
    elif layer == "l2":
        obs, bex = p.make_mmc_arm_equivalent_observers(
            plant.builder, plant.arms, dt=dt,  # type: ignore[arg-type]
        )
    else:
        raise ValueError(f"layer must be 'l1' or 'l2' (got {layer!r})")

    iLa, iLb, iLc = plant.iL_indices
    n_samples = int(round(t_end / dt)) + 1
    log_t   = np.zeros(n_samples)
    log_ia  = np.zeros(n_samples)
    log_ib  = np.zeros(n_samples)
    log_ic  = np.zeros(n_samples)
    log_vba = np.zeros(n_samples)
    log_vC  = np.zeros((6, n_samples))
    counter = [0]

    def log_obs(t, x):
        obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i]   = t
            log_ia[i]  = x[iLa]
            log_ib[i]  = x[iLb]
            log_ic[i]  = x[iLc]
            arms_list = plant.arms  # type: ignore[assignment]
            log_vba[i] = arms_list[0].v_b  # type: ignore[attr-defined]
            for k in range(6):
                log_vC[k, i] = arms_list[k].v_C  # type: ignore[attr-defined]
        counter[0] += 1

    p.simulate(
        plant.builder, t_end=t_end, dt=dt,  # type: ignore[arg-type]
        step_observer=log_obs, b_extra_fn=bex,
        start_from_dc_op=True,
    )

    n = counter[0]
    return MmcRunResult(
        t=log_t[:n], i_a=log_ia[:n], i_b=log_ib[:n], i_c=log_ic[:n],
        v_b_a_p=log_vba[:n], v_C=log_vC[:, :n],
    )


def thd(signal: np.ndarray, fs: float, f0: float, n_harm: int = 50) -> float:
    """Total harmonic distortion of ``signal`` at fundamental ``f0`` [%].

    Computes ``THD = sqrt(sum H_k²) / H_1 × 100 %`` over ``2..n_harm``.
    Uses a Hann window + rfft.
    """
    sig = np.asarray(signal, dtype=np.float64)
    sig = sig - sig.mean()
    n = len(sig)
    win = np.hanning(n)
    spec = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    # Find the bin closest to the fundamental.
    k1 = int(round(f0 / (fs / n)))
    if k1 < 1:
        return float("nan")
    fund = abs(spec[k1])
    harmonics_sq = 0.0
    for k in range(2, n_harm + 1):
        ki = k * k1
        if ki < len(spec):
            harmonics_sq += abs(spec[ki]) ** 2
    return float(100.0 * sqrt(harmonics_sq) / fund) if fund > 0 else float("nan")


def circulating_current(arm_p_branch_currents: np.ndarray,
                        arm_n_branch_currents: np.ndarray) -> np.ndarray:
    """Circulating current = (i_arm_p + i_arm_n) / 2 (Sousa eq 2.22).

    Both inputs are per-phase arm currents; the DC component of the
    sum equals the DC-port contribution; the AC component is the
    circulating part. Average of the two arms removes the AC port
    current entirely.
    """
    return 0.5 * (arm_p_branch_currents + arm_n_branch_currents)


def rms(signal: np.ndarray) -> float:
    sig = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(sig**2)))


def rms_ac(signal: np.ndarray) -> float:
    """RMS of the AC component (signal minus its DC mean)."""
    sig = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean((sig - sig.mean()) ** 2)))


# =============================================================================
# Closed-loop control helpers — dq current control on the MMC (Section 4.3)
# =============================================================================
#
# The thesis Section 4.3 derives a discrete-time RST controller for the
# MMC port currents (and a separate one for the circulating currents).
# In pulsim we use the simpler PIController class — same architectural
# pattern (decoupled per-axis PI) but in continuous time, suitable for
# a demonstration notebook. Tuning targets:
#
#   * Crossover ≈ 200 Hz (well below f_switch = 9 kHz/arm)
#   * Phase margin > 60° to leave room for the multilevel staircase
#     and dead-time non-idealities


def clarke_2_3(a: float, b: float, c: float) -> "tuple[float, float]":
    """Power-invariant Clarke (2/3) transform: abc → αβ."""
    alpha = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c)
    beta  = (2.0 / 3.0) * ((np.sqrt(3) / 2.0) * b
                              - (np.sqrt(3) / 2.0) * c)
    return alpha, beta


def park(alpha: float, beta: float, theta: float) -> "tuple[float, float]":
    """Park transform: αβ → dq (rotating frame at angle θ)."""
    c, s = np.cos(theta), np.sin(theta)
    return c * alpha + s * beta, -s * alpha + c * beta


def inv_park(d: float, q: float,
                theta: float) -> "tuple[float, float]":
    """Inverse Park: dq → αβ."""
    c, s = np.cos(theta), np.sin(theta)
    return c * d - s * q, s * d + c * q


def inv_clarke_2_3(alpha: float,
                       beta: float) -> "tuple[float, float, float]":
    """Inverse Clarke (matches the 2/3 forward): αβ → abc."""
    a = alpha
    b = -0.5 * alpha + (np.sqrt(3) / 2.0) * beta
    c = -0.5 * alpha - (np.sqrt(3) / 2.0) * beta
    return a, b, c


@dataclass
class ClosedLoopResult:
    """Output of :func:`run_mmc_closed_loop`."""

    t: np.ndarray
    # AC-port currents (load inductor branches).
    i_a: np.ndarray
    i_b: np.ndarray
    i_c: np.ndarray
    # dq frame currents reconstructed post-hoc from the abc samples.
    i_d: np.ndarray
    i_q: np.ndarray
    # Setpoint trajectories.
    i_d_ref: np.ndarray
    i_q_ref: np.ndarray
    # Modulation indices written into the upper arms (for plot).
    m_a_p: np.ndarray
    m_b_p: np.ndarray
    m_c_p: np.ndarray
    # Per-arm capacitor sums.
    v_C: np.ndarray  # shape (6, n)
    # Circulating currents per phase (Phase 20.16): (i_arm_p + i_arm_n)/2.
    # When ``with_circulating=False`` these are still logged for analysis.
    i_circ_a: np.ndarray = field(default_factory=lambda: np.empty(0))
    i_circ_b: np.ndarray = field(default_factory=lambda: np.empty(0))
    i_circ_c: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Mean per-arm capacitor voltage (for energy-loop diagnostics).
    v_C_mean: np.ndarray = field(default_factory=lambda: np.empty(0))


def run_mmc_closed_loop(
    params: GeanThesisParams,
    *,
    i_d_ref_fn: "Callable[[float], float]",
    i_q_ref_fn: "Callable[[float], float]",
    kp: float,
    ki: float,
    layer: str = "l1",
    t_end: float = 200e-3,
    dt: float = 5e-6,
    with_decoupling: bool = False,
    with_circulating: bool = False,
    kp_circ: float = 0.002,
    ki_circ: float = 5.0,
    with_energy_loop: bool = False,
    kp_energy: float = 0.005,    # Gentle — outer loop ~6 Hz bandwidth.
    ki_energy: float = 0.2,
    v_C_target: "float | None" = None,
) -> ClosedLoopResult:
    """Run an MMC inverter under closed-loop dq current control.

    Architecture mirrors the thesis Section 4.3 plus the optional
    Section 5.3 enhancements:

      * abc → αβ (Clarke 2/3) → dq (Park at θ = ω·t)
      * Two decoupled PI loops, one per axis
      * (optional) ω·L cross-coupling feedforward: cancels the
        natural coupling between d- and q-axis dynamics
      * (optional) per-phase circulating-current control: PI on the
        AC residual of ``(i_arm_p + i_arm_n)/2`` drives a common-mode
        offset ``δ_X`` to suppress the 2ω circulating ripple without
        disturbing the AC port voltage
      * (optional) outer energy loop: PI on ``mean(v_C) − V_dc``
        offsets ``i_d_ref`` so the converter draws / sources active
        power to balance the cap energy storage
      * Inverse Park / Clarke → per-phase voltage references
      * Per-arm modulation: ``m_X_p = (0.5 − v_X / V_dc) − δ_X``,
        ``m_X_n = (0.5 + v_X / V_dc) − δ_X``

    Args:
        params: Plant parameters (uses ``params.modulation_scheme``).
        i_d_ref_fn, i_q_ref_fn: Setpoint generators for the d- and q-axis
            currents (callables of ``t``).
        kp, ki: Current PI gains (same for both axes).
        layer: ``"l1"`` (PS-PWM / IPD multilevel) or ``"l2"`` (with
            dead-time + min-pulse-width).
        t_end, dt: Simulation horizon and step.
        with_decoupling: If True, add ω·L_eff feedforward terms to
            decouple the d/q axes. ``L_eff = l_load + l_b/2``.
        with_circulating: If True, enable per-phase PI on the AC
            residual of the circulating currents.
        kp_circ, ki_circ: Gains for the circulating-current PI
            (units: 1/(A) and 1/(A·s) — output is a modulation-index
            offset δ).
        with_energy_loop: If True, enable the outer slow PI on
            mean(v_C) that offsets i_d_ref.
        kp_energy, ki_energy: Outer-loop PI gains (units A/V and
            A/(V·s)).
        v_C_target: Energy-loop setpoint. Defaults to ``V_dc``.

    Returns:
        :class:`ClosedLoopResult` with the full logged trajectories.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", params.V_dc)

    m_a_holder = [0.5]
    m_b_holder = [0.5]
    m_c_holder = [0.5]

    if layer == "l1":
        arm_params = p.MmcArmMultilevelParams(
            n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
            f_carrier=params.f_carrier,
            modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
        )
        add_arm = p.add_mmc_arm_multilevel
        make_obs = p.make_mmc_arm_multilevel_observers
    elif layer == "l2":
        arm_params = p.MmcArmEquivalentParams(  # type: ignore[assignment]
            n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
            f_carrier=params.f_carrier,
            t_dead=params.t_dead, t_min=params.t_min,
            modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
        )
        add_arm = p.add_mmc_arm_equivalent
        make_obs = p.make_mmc_arm_equivalent_observers
    else:
        raise ValueError(f"layer must be 'l1' or 'l2' (got {layer!r})")

    arms: list[object] = []

    # Upper arms.
    for k, ph in enumerate("abc"):
        holder = (m_a_holder, m_b_holder, m_c_holder)[k]
        arm_p = add_arm(  # type: ignore[arg-type]
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params,  # type: ignore[arg-type]
            m_ref=lambda _t, _h=holder: _h[0],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        b.add_resistor(f"Rb_{ph}_p", f"rb_{ph}_p", f"ac_{ph}", params.r_b)

    # Lower arms with complement modulation.
    for k, ph in enumerate("abc"):
        holder = (m_a_holder, m_b_holder, m_c_holder)[k]
        b.add_resistor(f"Rb_{ph}_n", f"ac_{ph}", f"rb_{ph}_n", params.r_b)
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = add_arm(  # type: ignore[arg-type]
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="gnd",
            params=arm_params,  # type: ignore[arg-type]
            m_ref=lambda _t, _h=holder: 1.0 - _h[0],
        )
        arms.append(arm_n)

    iL_idx: list[int] = []
    for ph in "abc":
        lid = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_idx.append(b.pool.branch_var_id_for_inductor(lid, b.graph))
    b.add_resistor("R_star", "star", "gnd", 1e6)

    obs_arms, bex = make_obs(b, arms, dt=dt)  # type: ignore[arg-type]

    # Arm-current branch indices (for circulating + load decomposition).
    # The pulsim sign convention: x[src_idx] is positive when the
    # internal current flows ``from → to``. For our upper arm
    # (from=dc_p, to=mid_X_p) that's the natural arm-current direction.
    # The matching make_*_observers helpers also call this with the
    # same sign — see ``pulsim.mmc.make_mmc_arms_observer`` and the
    # commit ``feat(mmc): correct i_b sign``.
    arm_src_idx = [
        b.pool.branch_var_id_for_source(
            arm.source_branch_id, b.graph,  # type: ignore[attr-defined]
        )
        for arm in arms
    ]

    pi_d = p.PIController(
        Kp=kp, Ki=ki,
        output_min=-params.V_dc / 2.0,
        output_max=+params.V_dc / 2.0,
    )
    pi_q = p.PIController(
        Kp=kp, Ki=ki,
        output_min=-params.V_dc / 2.0,
        output_max=+params.V_dc / 2.0,
    )

    # Per-phase circulating-current PIs. Output ``δ`` is a modulation
    # index offset — keep it small (±0.1) so we don't lose load
    # control margin.
    pi_circ = [
        p.PIController(
            Kp=kp_circ, Ki=ki_circ,
            output_min=-0.10, output_max=+0.10,
        )
        for _ in range(3)
    ]

    # Outer energy-loop PI on mean(v_C). Output is a small offset
    # added to ``i_d_ref`` (clamped to ±5 A to stay well within the
    # main current loop's authority).
    energy_pi = p.PIController(
        Kp=kp_energy, Ki=ki_energy,
        output_min=-5.0, output_max=+5.0,
    )

    # Low-pass filters that extract the DC component of each
    # circulating current; the PI then drives the *AC residual* to
    # zero (we don't want to fight the natural DC bus current sharing).
    lpf_circ = [p.FirstOrderLowPass(tau=20e-3) for _ in range(3)]

    # Low-pass for the v_C mean — smooths out the 2ω ripple so the
    # outer loop sees the slow energy trend.
    lpf_vC = p.FirstOrderLowPass(tau=20e-3)

    v_C_set = params.V_dc if v_C_target is None else float(v_C_target)

    # Effective phase inductance for the decoupling feedforward.
    # AC current sees ``L_load`` plus ``L_b/2`` from the parallel arm
    # inductors at the AC port.
    L_eff = params.l_load + params.l_b / 2.0

    n = int(round(t_end / dt)) + 1
    log = {
        "t":      np.zeros(n),
        "i_a":    np.zeros(n),
        "i_b":    np.zeros(n),
        "i_c":    np.zeros(n),
        "i_d":    np.zeros(n),
        "i_q":    np.zeros(n),
        "i_d_ref": np.zeros(n),
        "i_q_ref": np.zeros(n),
        "m_a_p":  np.zeros(n),
        "m_b_p":  np.zeros(n),
        "m_c_p":  np.zeros(n),
        "v_C":    np.zeros((6, n)),
        "i_circ_a": np.zeros(n),
        "i_circ_b": np.zeros(n),
        "i_circ_c": np.zeros(n),
        "v_C_mean": np.zeros(n),
    }
    counter = [0]
    omega = params.omega_grid
    m_max = 0.95     # arm-modulation depth limit

    def control_and_observe(t, x):
        i_a = float(x[iL_idx[0]])
        i_b_ = float(x[iL_idx[1]])
        i_c_ = float(x[iL_idx[2]])
        theta = omega * t

        # ---- Arm currents + per-phase circulating currents ----
        # arms list order: a_p, b_p, c_p, a_n, b_n, c_n.
        i_arm = [float(x[arm_src_idx[k]]) for k in range(6)]
        i_circ_a = 0.5 * (i_arm[0] + i_arm[3])
        i_circ_b = 0.5 * (i_arm[1] + i_arm[4])
        i_circ_c = 0.5 * (i_arm[2] + i_arm[5])

        # ---- Outer energy loop (slow) ----
        # Sign convention for an inverter (V_dc → AC):
        #   * positive ``i_d`` drains power from the DC bus → caps
        #     discharge → ``v_C`` drops.
        #   * If ``v_C < v_C_target`` we need to drain LESS, so
        #     ``i_d_offset`` should be NEGATIVE.
        # We achieve that by swapping setpoint/measured in the PI:
        # error = (v_C_mean − v_C_target). When v_C is low (negative
        # error) the integral output goes negative → i_d shrinks.
        v_C_mean = sum(arm.v_C for arm in arms) / 6.0  # type: ignore[attr-defined]
        v_C_mean_filt = lpf_vC.update(input_value=v_C_mean, dt=dt)
        i_d_ref = float(i_d_ref_fn(t))
        if with_energy_loop:
            i_d_offset = energy_pi.update(
                setpoint=v_C_mean_filt, measured=v_C_set, dt=dt,
            )
            i_d_ref = i_d_ref + i_d_offset

        i_q_ref = float(i_q_ref_fn(t))

        # ---- Inner dq current PI ----
        i_alpha, i_beta = clarke_2_3(i_a, i_b_, i_c_)
        i_d, i_q = park(i_alpha, i_beta, theta)
        v_d = pi_d.update(setpoint=i_d_ref, measured=i_d, dt=dt)
        v_q = pi_q.update(setpoint=i_q_ref, measured=i_q, dt=dt)

        # ---- ω·L decoupling feedforward ----
        # Continuous-time plant in dq frame:
        #   v_d = R·i_d + L·di_d/dt − ω·L·i_q
        #   v_q = R·i_q + L·di_q/dt + ω·L·i_d
        # The PI controllers regulate (R·i + L·di/dt). Adding the
        # cross-coupling term cancels the natural coupling so each
        # axis becomes independent.
        if with_decoupling:
            v_d = v_d - omega * L_eff * i_q
            v_q = v_q + omega * L_eff * i_d

        v_alpha, v_beta = inv_park(v_d, v_q, theta)
        v_a, v_b, v_c = inv_clarke_2_3(v_alpha, v_beta)

        # Clip the half-swing voltage so the modulation index stays
        # in [0.5 - m_max/2, 0.5 + m_max/2].
        half_swing = m_max * params.V_dc / 2.0
        v_a = max(-half_swing, min(half_swing, v_a))
        v_b = max(-half_swing, min(half_swing, v_b))
        v_c = max(-half_swing, min(half_swing, v_c))

        # ---- Per-phase circulating-current damping ----
        # Proper MMC circulating-current control uses a PI in the
        # 2ω-synchronous (negative-sequence) frame so that the 2ω
        # ripple appears as DC to the integrator. Implementing that
        # full structure properly is its own exercise; for this
        # notebook we use a lighter-weight scheme — pure proportional
        # damping on the AC residual ``(i_circ − lpf(i_circ))``. It
        # *damps* the natural circulating ripple without trying to
        # zero it (which would need accurate phase tracking).
        if with_circulating:
            i_circ_dc_a = lpf_circ[0].update(input_value=i_circ_a, dt=dt)
            i_circ_dc_b = lpf_circ[1].update(input_value=i_circ_b, dt=dt)
            i_circ_dc_c = lpf_circ[2].update(input_value=i_circ_c, dt=dt)
            delta_a = max(-0.05, min(0.05,
                kp_circ * (i_circ_a - i_circ_dc_a)))
            delta_b = max(-0.05, min(0.05,
                kp_circ * (i_circ_b - i_circ_dc_b)))
            delta_c = max(-0.05, min(0.05,
                kp_circ * (i_circ_c - i_circ_dc_c)))
        else:
            delta_a = delta_b = delta_c = 0.0

        # Base modulation from the load-side voltage commands.
        m_a_base = 0.5 - v_a / params.V_dc
        m_b_base = 0.5 - v_b / params.V_dc
        m_c_base = 0.5 - v_c / params.V_dc
        # Apply circulating-current correction.
        m_a_holder[0] = m_a_base - delta_a
        m_b_holder[0] = m_b_base - delta_b
        m_c_holder[0] = m_c_base - delta_c

        obs_arms(t, x)

        i = counter[0]
        if i < n:
            log["t"][i] = t
            log["i_a"][i] = i_a
            log["i_b"][i] = i_b_
            log["i_c"][i] = i_c_
            log["i_d"][i] = i_d
            log["i_q"][i] = i_q
            log["i_d_ref"][i] = i_d_ref
            log["i_q_ref"][i] = i_q_ref
            log["m_a_p"][i] = m_a_holder[0]
            log["m_b_p"][i] = m_b_holder[0]
            log["m_c_p"][i] = m_c_holder[0]
            log["i_circ_a"][i] = i_circ_a
            log["i_circ_b"][i] = i_circ_b
            log["i_circ_c"][i] = i_circ_c
            log["v_C_mean"][i] = v_C_mean
            for k in range(6):
                log["v_C"][k, i] = arms[k].v_C  # type: ignore[attr-defined]
        counter[0] += 1

    p.simulate(
        b, t_end=t_end, dt=dt,
        step_observer=control_and_observe, b_extra_fn=bex,
        start_from_dc_op=True,
    )

    nn = counter[0]
    return ClosedLoopResult(
        t=log["t"][:nn],
        i_a=log["i_a"][:nn], i_b=log["i_b"][:nn], i_c=log["i_c"][:nn],
        i_d=log["i_d"][:nn], i_q=log["i_q"][:nn],
        i_d_ref=log["i_d_ref"][:nn], i_q_ref=log["i_q_ref"][:nn],
        m_a_p=log["m_a_p"][:nn], m_b_p=log["m_b_p"][:nn], m_c_p=log["m_c_p"][:nn],
        v_C=log["v_C"][:, :nn],
        i_circ_a=log["i_circ_a"][:nn],
        i_circ_b=log["i_circ_b"][:nn],
        i_circ_c=log["i_circ_c"][:nn],
        v_C_mean=log["v_C_mean"][:nn],
    )
