"""Modular Multilevel Converter (MMC) — analytical model (single-phase).

Third entry under ``projects/inverters/``. The MMC is the
state-of-the-art multilevel converter and the de-facto standard for
HVDC transmission. It scales seamlessly to hundreds of voltage levels
just by adding more series sub-modules — every cap of every SM is a
floating energy buffer that the controller must actively regulate.

Single-phase topology (this module models one phase; 3-phase = three
identical phase legs sharing the DC bus)::

                 +V_dc/2
                    │
                  L_arm
                    │
        arm_up_in ──┼── SM_u1 ── SM_u2 ── ··· ── SM_uN ──┐
                                                          │
                                                       ac_out
                                                          │
        arm_lo_in ──┼── SM_l1 ── SM_l2 ── ··· ── SM_lN ──┘
                    │
                  L_arm
                    │
                 -V_dc/2

Half-bridge SM topology::

       Terminal A (midpoint of S1, S2)
            │
       ┌────┤
       │    │
       │   S1
       │    │
     C_SM   midpoint
       │    │
       │   S2
       │    │
       └────┤
            │
       Terminal B

State INSERT (S1 ON, S2 OFF):  V_SM = +V_C  (cap in series with arm current)
State BYPASS (S1 OFF, S2 ON):  V_SM = 0     (cap isolated, arm current passes around)

References
----------

* Lesnicar, A. & Marquardt, R. *An innovative modular multilevel
  converter topology suitable for a wide power range*. IEEE Bologna
  PowerTech (2003). The original MMC paper.
* Sharifabadi, K., Harnefors, L., Nee, H.-P., Norrga, S.,
  Teodorescu, R. *Design, Control, and Application of Modular
  Multilevel Converters for HVDC Transmission Systems*. Wiley/IEEE
  Press, 2016 — the comprehensive textbook reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SubModuleState = Literal["INSERT", "BYPASS"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMCParams:
    """Single-phase MMC parameters (SI units).

    Defaults: 400 V DC bus → 110 V_rms AC at 60 Hz with **N = 3**
    sub-modules per arm. SM caps sized for ~10 % voltage ripple at
    rated load. Arm inductors sized so the LC corner formed with the
    paralleled SM caps falls between $f_o$ and the carrier — provides
    natural damping of circulating-current harmonics.

    Operating-point note: the single-phase MMC has
    :math:`v_{ac,pk} \\le V_{dc}/2` by topology. We pick a comfortable
    margin: :math:`V_{o,pk} = 155.6` V with :math:`V_{dc}/2 = 200` V,
    so :math:`m_a \\approx 0.78`. (For a higher-voltage application
    like HVDC, increase :math:`V_{dc}` proportionally — the
    topology and control machinery are unchanged.)
    """

    # DC bus
    V_dc: float = 400.0          # total DC bus voltage (V)

    # AC output spec
    V_o_rms: float = 110.0       # single-phase output rms (V) — typical US residential
    f_o: float = 60.0            # output frequency (Hz)
    P_o: float = 500.0           # output power (W)

    # MMC sub-modules
    N_sm: int = 3                # sub-modules per arm
    C_sm: float = 470e-6         # sub-module cap (F)
    L_arm: float = 1e-3          # arm inductor (H)

    # AC load (RL, single-phase)
    R_load: float = 24.0         # load resistance (Ω) — gives ~500 W at 110 V_rms
    L_load: float = 5e-3         # load inductance (H)

    # PSC-PWM
    f_carrier: float = 1.0e3     # per-carrier frequency (Hz). The
                                 # effective load-side ripple appears
                                 # at N_sm × f_carrier.

    # ---- Derived quantities ----

    @property
    def V_dc_half(self) -> float:
        """Nominal voltage of each DC rail relative to ground (V)."""
        return self.V_dc / 2.0

    @property
    def V_C_nominal(self) -> float:
        """Nominal steady-state SM cap voltage (V).

        Sum of all SM voltages in an arm must equal V_dc in DC
        equilibrium (when N_sm SMs of value V_C are inserted on one
        arm and 0 on the other), so V_C = V_dc / N_sm.
        """
        return self.V_dc / self.N_sm

    @property
    def V_o_pk(self) -> float:
        """Peak AC output voltage (V)."""
        return self.V_o_rms * np.sqrt(2.0)

    @property
    def m_a(self) -> float:
        """Modulation index.

        For an MMC with N sub-modules per arm,
        :math:`v_{ac}(t) \\le V_{dc}/2`, so
        :math:`m_a = 2 V_{o,pk} / V_{dc} \\in [0, 1]`.
        """
        return 2.0 * self.V_o_pk / self.V_dc

    @property
    def omega_o(self) -> float:
        return 2.0 * np.pi * self.f_o

    @property
    def T_carrier(self) -> float:
        return 1.0 / self.f_carrier

    @property
    def f_ripple_effective(self) -> float:
        """Effective load-side ripple frequency for PSC-PWM (Hz).

        With N phase-shifted carriers per arm, the lowest significant
        harmonic in the output appears at :math:`N \\cdot f_{carrier}`
        — the MMC's "N-frequency multiplication" effect.
        """
        return self.N_sm * self.f_carrier

    @property
    def R_eq(self) -> float:
        """Equivalent AC load resistance for power balance (Ω).

        For a purely resistive load: P_o = V_o_rms² / R.
        """
        return self.V_o_rms ** 2 / self.P_o

    @property
    def I_o_pk(self) -> float:
        """Peak load current (A) at the rated operating point."""
        return self.V_o_pk / self.R_load


# ---------------------------------------------------------------------------
# Sub-module voltage as a function of state and cap voltage
# ---------------------------------------------------------------------------


def sm_voltage(state: SubModuleState, v_cap: float) -> float:
    """Voltage across a half-bridge SM as a function of its state.

    Parameters
    ----------
    state
        ``'INSERT'`` (cap in series with arm) or ``'BYPASS'`` (cap
        isolated).
    v_cap
        Current SM capacitor voltage (V).
    """
    if state == "INSERT":
        return v_cap
    if state == "BYPASS":
        return 0.0
    raise ValueError(f"unknown SM state: {state!r}")


def arm_voltage(states: "list[SubModuleState]",
                  v_caps: "list[float]") -> float:
    """Total arm voltage = sum of per-SM voltages."""
    if len(states) != len(v_caps):
        raise ValueError("states and v_caps length mismatch")
    return sum(sm_voltage(s, v) for s, v in zip(states, v_caps))


def n_insert(states: "list[SubModuleState]") -> int:
    """Count of inserted (active) SMs in an arm."""
    return sum(1 for s in states if s == "INSERT")


# ---------------------------------------------------------------------------
# Phase-Shifted Carrier (PSC) PWM
# ---------------------------------------------------------------------------


def psc_pwm_insertion_count(v_ref_norm: float, t: float, f_carrier: float,
                             N: int, carrier_phase_offset: float = 0.0
                             ) -> int:
    """Decide how many SMs in an arm should be **inserted** at time
    ``t``, given a normalised reference signal ``v_ref_norm ∈ [0, 1]``.

    PSC-PWM uses ``N`` triangular carriers all at ``f_carrier`` but
    phase-shifted by ``2π/N`` from each other. The number of inserted
    SMs equals the count of carriers below ``v_ref_norm`` at time
    ``t``.

    Parameters
    ----------
    v_ref_norm
        Reference signal, normalised to ``[0, 1]`` (= duty-cycle for
        the arm, 0 = all SMs bypassed, 1 = all inserted).
    t
        Current time (s).
    f_carrier
        Frequency of each carrier (Hz).
    N
        Number of SMs (= number of carriers).
    carrier_phase_offset
        Optional bulk phase offset (rad) applied to all carriers.
        Useful to phase-shift upper vs lower arm carriers by π/N for
        extra ripple suppression.
    """
    v_ref_norm = float(np.clip(v_ref_norm, 0.0, 1.0))
    n_below = 0
    for i in range(N):
        # Carrier i phase: i · 2π/N + bulk offset
        phi_c = (2.0 * np.pi * f_carrier * t
                  + carrier_phase_offset
                  + i * 2.0 * np.pi / N) % (2.0 * np.pi)
        # Triangle [0, 1].
        tri = 0.5 + 0.5 * (2.0 / np.pi) * np.arcsin(np.sin(phi_c))
        if v_ref_norm > tri:
            n_below += 1
    return n_below


def psc_pwm_insertion_count_vectorised(v_ref_norm: np.ndarray, t: np.ndarray,
                                          f_carrier: float, N: int,
                                          carrier_phase_offset: float = 0.0
                                          ) -> np.ndarray:
    """Vectorised version of :func:`psc_pwm_insertion_count`."""
    v_ref_norm = np.clip(v_ref_norm, 0.0, 1.0)
    n_below = np.zeros(t.shape, dtype=int)
    for i in range(N):
        phi_c = (2.0 * np.pi * f_carrier * t
                  + carrier_phase_offset
                  + i * 2.0 * np.pi / N) % (2.0 * np.pi)
        tri = 0.5 + 0.5 * (2.0 / np.pi) * np.arcsin(np.sin(phi_c))
        n_below += (v_ref_norm > tri).astype(int)
    return n_below


# ---------------------------------------------------------------------------
# Arm references (open-loop, before any cap-balancing override)
# ---------------------------------------------------------------------------


def arm_references(params: MMCParams, t: np.ndarray | float,
                    m_a: float | None = None,
                    ) -> "tuple[np.ndarray | float, np.ndarray | float]":
    """Compute the upper- and lower-arm normalised duty references.

    For an open-loop MMC producing
    :math:`v_{ac}(t) = (V_{dc}/2) \\, m_a \\sin(\\omega_o t)`,
    we need:

        v_{arm,upper}(t) = V_dc/2 - v_{ac}(t)
                          = V_dc/2 (1 - m_a sin(ω t))
        v_{arm,lower}(t) = V_dc/2 + v_{ac}(t)
                          = V_dc/2 (1 + m_a sin(ω t))

    Normalised to ``[0, 1]`` (i.e. fraction of N_sm SMs to insert):

        d_{up}(t) = (1 - m_a sin(ω t)) / 2
        d_{lo}(t) = (1 + m_a sin(ω t)) / 2

    They sum to 1 at every instant — the **invariant** that keeps
    v_{arm,upper} + v_{arm,lower} = V_dc on average.
    """
    if m_a is None:
        m_a = float(params.m_a)
    s = np.sin(params.omega_o * np.asarray(t))
    d_up = 0.5 * (1.0 - m_a * s)
    d_lo = 0.5 * (1.0 + m_a * s)
    return d_up, d_lo


# ---------------------------------------------------------------------------
# Circulating current model
# ---------------------------------------------------------------------------


def decompose_arm_currents(i_arm_up: np.ndarray, i_arm_lo: np.ndarray
                              ) -> "tuple[np.ndarray, np.ndarray]":
    """Split the two arm currents into output and circulating components.

    By KCL at the AC output node:
        i_ac(t) = i_arm_up(t) - i_arm_lo(t)
    By definition of the circulating current:
        i_circ(t) = (i_arm_up(t) + i_arm_lo(t)) / 2

    The circulating current has two important components:
    * **DC component** = average input current from the bus
      (= P_o / V_dc in steady state, for a lossless converter).
    * **2·f_o component** = the parasitic ripple that pumps the cap
      voltages and must be suppressed by the resonant controller in
      ``02_mmc_control.ipynb``.
    """
    i_ac = i_arm_up - i_arm_lo
    i_circ = 0.5 * (i_arm_up + i_arm_lo)
    return i_ac, i_circ


# ---------------------------------------------------------------------------
# Sort-and-select capacitor balancing
# ---------------------------------------------------------------------------


def sort_and_select(n_to_insert: int, v_caps: np.ndarray,
                      i_arm_sign: int) -> np.ndarray:
    """Sort-and-select algorithm: choose **which** SMs to insert.

    Given:
    * ``n_to_insert``: how many SMs the modulator wants to insert
      (output of PSC-PWM).
    * ``v_caps``: array of N cap voltages.
    * ``i_arm_sign``: +1 if arm current is *charging* the inserted
      SMs (we want to insert the LOWEST cap voltages to even out);
      -1 if *discharging* (insert the HIGHEST to bring them down).

    Returns a boolean array of length N where ``True`` means INSERT
    that SM.
    """
    N = len(v_caps)
    if n_to_insert <= 0:
        return np.zeros(N, dtype=bool)
    if n_to_insert >= N:
        return np.ones(N, dtype=bool)
    # Sort indices by cap voltage.
    order = np.argsort(v_caps)
    if i_arm_sign > 0:
        # Charging: insert the lowest-voltage caps so they catch up.
        selected = order[:n_to_insert]
    else:
        # Discharging: insert the highest-voltage caps so they bleed down.
        selected = order[N - n_to_insert:]
    out = np.zeros(N, dtype=bool)
    out[selected] = True
    return out


# ---------------------------------------------------------------------------
# Operating-point report
# ---------------------------------------------------------------------------


def operating_point_report(params: MMCParams) -> str:
    """Human-readable summary of the analytical operating point."""
    p = params
    lines = [
        "MMC single-phase operating point",
        f"  V_dc        = {p.V_dc:8.2f} V       (each rail = {p.V_dc_half:.1f} V)",
        f"  V_o,rms     = {p.V_o_rms:8.2f} V_rms ({p.V_o_pk:.1f} V_pk)",
        f"  f_o         = {p.f_o:8.1f} Hz",
        f"  P_o         = {p.P_o:8.1f} W       (R_load = {p.R_load:.2f} Ω, "
        f"L_load = {p.L_load*1e3:.2f} mH)",
        f"  I_o_pk      = {p.I_o_pk:8.3f} A",
        "",
        "Sub-modules",
        f"  N_sm        = {p.N_sm:8d}        (per arm)",
        f"  V_C nominal = {p.V_C_nominal:8.2f} V       (= V_dc / N_sm)",
        f"  C_sm        = {p.C_sm * 1e6:8.0f} µF",
        f"  L_arm       = {p.L_arm * 1e3:8.2f} mH",
        "",
        "Modulation (PSC-PWM)",
        f"  m_a         = {p.m_a:8.4f}",
        f"  f_carrier   = {p.f_carrier/1e3:8.2f} kHz  (per carrier)",
        f"  f_ripple    = {p.f_ripple_effective/1e3:8.2f} kHz  "
        f"(= N · f_carrier, load-side)",
        "",
        "Multilevel signature",
        f"  Phase pole voltage: {p.N_sm + 1} discrete levels in [-V_dc/2, +V_dc/2]",
        f"  Arm voltage: {p.N_sm + 1} levels in [0, N·V_C]",
    ]
    return "\n".join(lines)


__all__ = [
    "MMCParams",
    "sm_voltage",
    "arm_voltage",
    "n_insert",
    "psc_pwm_insertion_count",
    "psc_pwm_insertion_count_vectorised",
    "arm_references",
    "decompose_arm_currents",
    "sort_and_select",
    "operating_point_report",
]
