"""Three-phase Neutral-Point-Clamped (NPC) 3-level inverter — analytical model.

Second entry under ``projects/inverters/``. Mirrors ``vsi_3phase_model.py``
in structure but for a 3-level NPC topology instead of a 2-level VSI.

Topology (per leg)::

         Vdc_pos (+V_dc/2)
            │
           S1
            │
            n1 ──── D_clamp_top  (anode at NP, cathode at n1)
            │
           S2
            │
           mid_X    ← phase output
            │
           S3
            │
            n2 ──── D_clamp_bot  (anode at n2, cathode at NP)
            │
           S4
            │
         Vdc_neg (-V_dc/2)

The DC bus is split into two halves by capacitors :math:`C_{dc}` each,
with the **neutral point (NP)** at the midpoint (0 V w.r.t. system
ground).

References
----------

* Nabae, A., Takahashi, I. & Akagi, H. *A New Neutral-Point-Clamped
  PWM Inverter*. IEEE Trans. Ind. Appl. IA-17, 518–523 (1981).
* Holmes, D. G. & Lipo, T. A. *Pulse Width Modulation for Power
  Converters*. IEEE Press 2003 — Chapter 11 (multicarrier PWM).
* Wu, B. *High-Power Converters and AC Drives*. IEEE Press 2006 —
  Chapter 8 (NPC inverters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


SwitchingState = Literal["P", "O", "N"]


@dataclass(frozen=True)
class NPC3PhaseParams:
    """Three-phase NPC 3-level inverter parameters (SI units).

    Defaults match the 2-level :class:`VSI3PhaseParams` for direct
    comparison: 400 V DC bus → 230 V line-to-line rms at 60 Hz,
    500 W, 1 mH + 10 µF LC output filter per phase. Switching at
    **5 kHz** (vs the 2-level's 20 kHz) — the NPC's three-level
    output already cuts the fundamental ripple amplitude in half,
    so a lower :math:`f_{sw}` produces comparable output quality with
    less switching loss.
    """

    # DC bus
    V_dc: float = 400.0          # total DC bus voltage (V), so V_dc/2 on each half
    C_dc: float = 470e-6         # each split bus cap (F), 2 of them

    # AC output spec
    V_o_LL_rms: float = 230.0    # line-to-line rms output (V)
    f_o: float = 60.0            # output frequency (Hz)
    P_o: float = 500.0           # nominal output power (W)

    # Output LC filter (per phase, between leg pole and load terminal)
    L_f: float = 1e-3            # filter inductor (H)
    C_f: float = 10e-6           # filter cap, Y-connected (F)
    R_L: float = 1.0             # filter inductor ESR (Ω) — primary damping

    # Switching
    f_sw: float = 5e3            # carrier frequency (Hz) — see header note

    # Modulation
    m_a_max: float = 1.0         # SVPWM-style limit (with min-max injection);
                                  #   linear sinusoidal limit is 0.866.

    # ---- Derived quantities ----

    @property
    def V_dc_half(self) -> float:
        """Half-bus voltage = nominal neutral-point voltage (V)."""
        return self.V_dc / 2.0

    @property
    def V_o_LN_pk(self) -> float:
        """Peak line-to-neutral output voltage (V).

        For a balanced 3-phase set with line-to-line rms = V_LL_rms:
            V_LN_rms = V_LL_rms / sqrt(3)
            V_LN_pk  = sqrt(2) * V_LN_rms = V_LL_rms * sqrt(2/3)
        """
        return self.V_o_LL_rms * np.sqrt(2.0) / np.sqrt(3.0)

    @property
    def V_o_LL_pk(self) -> float:
        """Peak line-to-line output voltage (V)."""
        return self.V_o_LL_rms * np.sqrt(2.0)

    @property
    def m_a(self) -> float:
        """Modulation index :math:`m_a = 2 V_{o,LN,pk} / V_{dc}`.

        With SVPWM-style min-max injection the maximum is 1.0 (so the
        line-to-line peak equals :math:`V_{dc}`). With plain sinusoidal
        PWM (no injection) the maximum is :math:`\\sqrt{3}/2 = 0.866`.
        """
        return 2.0 * self.V_o_LN_pk / self.V_dc

    @property
    def omega_o(self) -> float:
        return 2.0 * np.pi * self.f_o

    @property
    def T_sw(self) -> float:
        return 1.0 / self.f_sw

    @property
    def R_load(self) -> float:
        """Equivalent per-phase load resistance for the rated P_o (Ω).

        :math:`P_o = 3 V_{o,LN,rms}^2 / R_{load}`
        """
        V_LN_rms = self.V_o_LN_pk / np.sqrt(2.0)
        return 3.0 * V_LN_rms ** 2 / self.P_o

    @property
    def I_o_pk(self) -> float:
        """Peak per-phase load current at rated power (A)."""
        return self.V_o_LN_pk / self.R_load

    @property
    def f_filter_corner(self) -> float:
        """Output LC corner frequency (Hz)."""
        return 1.0 / (2.0 * np.pi * np.sqrt(self.L_f * self.C_f))

    @property
    def f_ripple_effective(self) -> float:
        """Dominant load-side ripple frequency for PD-PWM (Hz).

        With phase-disposition multicarrier PWM the lowest significant
        harmonic in the output appears at :math:`2 f_{sw}`. That's the
        "frequency doubling" effect of multilevel — exactly why we can
        run at lower carrier frequency than a 2-level VSI for the
        same output quality.
        """
        return 2.0 * self.f_sw


# ---------------------------------------------------------------------------
# Switching state table
# ---------------------------------------------------------------------------


def switching_state_to_pole_voltage(state: SwitchingState, V_dc: float
                                       ) -> float:
    """Return the leg's pole voltage for a given switching state.

    Voltage is measured between the phase output ``mid_X`` and the
    neutral point (NP), so ranges over ``{+V_dc/2, 0, -V_dc/2}``.

    Parameters
    ----------
    state
        ``'P'`` (positive, V_dc/2), ``'O'`` (zero, NP), or ``'N'``
        (negative, -V_dc/2).
    V_dc
        Total DC bus voltage. Half-bus is V_dc/2.
    """
    if state == "P":
        return 0.5 * V_dc
    if state == "O":
        return 0.0
    if state == "N":
        return -0.5 * V_dc
    raise ValueError(f"unknown switching state: {state!r} (expected P/O/N)")


SWITCH_TABLE = {
    # state: (S1, S2, S3, S4)  (1 = ON, 0 = OFF)
    "P": (1, 1, 0, 0),
    "O": (0, 1, 1, 0),
    "N": (0, 0, 1, 1),
}
"""Per-leg switch states for the three valid NPC modes.

Any switch combination not listed here either short-circuits the bus
or violates the diode-clamp constraint — the modulator must never
produce such combinations.
"""


# ---------------------------------------------------------------------------
# Multicarrier PD-PWM
# ---------------------------------------------------------------------------


def pd_pwm_state(v_ref: float, t: float, f_sw: float,
                  carrier_phase: float = 0.0) -> SwitchingState:
    """Decide the NPC switching state for one phase by comparing the
    reference signal against two phase-disposition (PD) triangular
    carriers stacked vertically.

    The two carriers occupy ``[0, +1]`` (upper) and ``[-1, 0]`` (lower).
    Both have the same frequency and phase; the reference ``v_ref ∈
    [-1, +1]`` is compared against each:

    * ``v_ref > tri_upper`` → state ``P``
    * ``v_ref < tri_lower`` → state ``N``
    * otherwise → state ``O``

    Parameters
    ----------
    v_ref
        Reference signal (normalised to ``[-1, +1]``). For a balanced
        3-phase set, ``v_ref(t) = m_a * sin(omega_o*t - theta_phi)``
        where ``m_a ∈ [0, 1]`` is the modulation index.
    t
        Current time (s).
    f_sw
        Carrier frequency (Hz).
    carrier_phase
        Carrier phase offset (rad). Default 0.
    """
    phase = (2.0 * np.pi * f_sw * t + carrier_phase) % (2.0 * np.pi)
    # Triangle from -1 to +1 over one period.
    tri = 2.0 / np.pi * np.arcsin(np.sin(phase))
    # Upper carrier: [0, +1].
    tri_upper = 0.5 + 0.5 * tri
    # Lower carrier: [-1, 0].
    tri_lower = -0.5 + 0.5 * tri

    if v_ref > tri_upper:
        return "P"
    if v_ref < tri_lower:
        return "N"
    return "O"


def pd_pwm_state_vectorised(v_ref: np.ndarray, t: np.ndarray, f_sw: float,
                              carrier_phase: float = 0.0,
                              ) -> np.ndarray:
    """Vectorised PD-PWM state lookup. Returns a string-like array."""
    phase = (2.0 * np.pi * f_sw * t + carrier_phase) % (2.0 * np.pi)
    tri = 2.0 / np.pi * np.arcsin(np.sin(phase))
    tri_upper = 0.5 + 0.5 * tri
    tri_lower = -0.5 + 0.5 * tri
    out = np.full(v_ref.shape, "O", dtype="<U1")
    out[v_ref > tri_upper] = "P"
    out[v_ref < tri_lower] = "N"
    return out


# ---------------------------------------------------------------------------
# Fundamental output amplitude (analytical)
# ---------------------------------------------------------------------------


def fundamental_pole_voltage(params: NPC3PhaseParams, m_a: float | None = None
                              ) -> float:
    """Peak fundamental amplitude of the phase-to-NP voltage (V).

    Under linear (sub-overmodulation) operation the average phase
    pole voltage over one carrier period is

    .. math::
        \\langle v_{pole}(t) \\rangle = \\frac{m_a V_{dc}}{2}
                                      \\sin(\\omega_o t - \\theta_\\phi)

    Same formula as the 2-level VSI; multilevel changes the *shape*
    of the instantaneous waveform (3 levels instead of 2) but not the
    fundamental amplitude.
    """
    if m_a is None:
        m_a = params.m_a
    return m_a * params.V_dc_half


def fundamental_line_to_line(params: NPC3PhaseParams,
                                m_a: float | None = None) -> float:
    """Peak fundamental amplitude of the line-to-line voltage (V).

    Two pole voltages 120° apart subtract:
        v_ab(t) = v_a - v_b
                = (m_a V_dc/2) [sin(ωt) - sin(ωt - 2π/3)]
                = sqrt(3) * (m_a V_dc/2) * sin(ωt + π/6)
    """
    return np.sqrt(3.0) * fundamental_pole_voltage(params, m_a)


# ---------------------------------------------------------------------------
# THD predictions (closed-form approximations)
# ---------------------------------------------------------------------------


def thd_voltage_unfiltered(time: np.ndarray, signal_pwm: np.ndarray,
                              f_fundamental: float,
                              n_harm_max: int = 50) -> float:
    """Measure THD of a PWM waveform numerically from its FFT.

    Compute the FFT of ``signal_pwm`` sampled uniformly on ``time``,
    identify the fundamental component at ``f_fundamental``, and
    return the THD up to the ``n_harm_max``-th harmonic:

    .. math::
        \\mathrm{THD} = \\frac{\\sqrt{\\sum_{n=2}^{N} V_n^2}}{V_1}

    Used by the validation notebook to compare 2-level vs 3-level PWM
    THD on identical signals.
    """
    dt = float(time[1] - time[0])
    n = len(signal_pwm)
    # Use only integer number of fundamental periods to limit leakage.
    T_fund = 1.0 / f_fundamental
    n_per_period = max(1, int(round(T_fund / dt)))
    n_periods = n // n_per_period
    n_clip = n_periods * n_per_period
    if n_clip < n_per_period:
        return float("nan")
    sig = np.asarray(signal_pwm[:n_clip], dtype=float)
    spectrum = np.fft.rfft(sig) / (n_clip / 2.0)
    amplitudes = np.abs(spectrum)
    # Fundamental bin is at index n_periods (== n_clip / n_per_period).
    k_fund = n_periods
    if k_fund >= len(amplitudes):
        return float("nan")
    v1 = amplitudes[k_fund]
    if v1 < 1e-12:
        return float("nan")
    n_top = min(n_harm_max + 1, len(amplitudes) // k_fund + 1)
    harmonic_powers = sum(
        amplitudes[k * k_fund] ** 2 for k in range(2, n_top)
        if k * k_fund < len(amplitudes)
    )
    return float(np.sqrt(harmonic_powers) / v1)


# ---------------------------------------------------------------------------
# Neutral-point balancing dynamics
# ---------------------------------------------------------------------------


def neutral_point_current(state_a: SwitchingState,
                            state_b: SwitchingState,
                            state_c: SwitchingState,
                            i_a: float, i_b: float, i_c: float) -> float:
    """Instantaneous current flowing out of the neutral point (A).

    A phase contributes to the NP current **only** when its switching
    state is ``O`` — in that case the load current of that phase
    passes through the clamping diodes into (or out of) the NP. P
    and N states route the load current around the NP entirely.

    Sign convention: positive ``i_NP`` flows OUT of the NP into the
    load. By KCL, the top split cap is *charged* and the bottom is
    *discharged* (or vice versa) at rate :math:`dV_{NP}/dt = -i_{NP}
    / C_{dc}`.

    For a balanced 3-phase load, the time average of ``i_NP`` is
    zero only if the three phases' contributions cancel — which
    happens at certain symmetric ``m_a`` and load angles but **not**
    in general. That's why the NP voltage drifts open-loop.
    """
    contrib = 0.0
    if state_a == "O":
        contrib += i_a
    if state_b == "O":
        contrib += i_b
    if state_c == "O":
        contrib += i_c
    return contrib


# ---------------------------------------------------------------------------
# Reference-frame transforms (re-exported from vsi_3phase for convenience)
# ---------------------------------------------------------------------------


def clarke_transform(v_a: np.ndarray, v_b: np.ndarray, v_c: np.ndarray
                      ) -> "tuple[np.ndarray, np.ndarray]":
    """abc → αβ, amplitude-invariant (2/3 factor)."""
    v_alpha = (2.0 / 3.0) * (v_a - 0.5 * v_b - 0.5 * v_c)
    v_beta = (2.0 / 3.0) * (np.sqrt(3.0) / 2.0) * (v_b - v_c)
    return v_alpha, v_beta


# ---------------------------------------------------------------------------
# Operating-point report
# ---------------------------------------------------------------------------


def operating_point_report(params: NPC3PhaseParams) -> str:
    """Human-readable summary of the analytical operating point."""
    p = params
    lines = [
        "NPC 3-level inverter operating point",
        f"  V_dc       = {p.V_dc:8.2f} V       (half = {p.V_dc_half:.1f} V)",
        f"  V_o_LL,rms = {p.V_o_LL_rms:8.2f} V_rms ({p.V_o_LL_pk:.1f} V_pk)",
        f"  V_o_LN,pk  = {p.V_o_LN_pk:8.2f} V",
        f"  f_o        = {p.f_o:8.1f} Hz",
        f"  P_o        = {p.P_o:8.1f} W       (R_load Y = {p.R_load:.2f} Ω/φ)",
        f"  I_o_pk     = {p.I_o_pk:8.3f} A",
        "",
        "Modulation",
        f"  m_a        = {p.m_a:8.4f}        (cap with min-max injection: 1.0)",
        f"  fundamental pole peak = m_a · V_dc/2 = {fundamental_pole_voltage(p):.1f} V",
        f"  fundamental L-L peak  = √3 · m_a · V_dc/2 = "
        f"{fundamental_line_to_line(p):.1f} V",
        "",
        "Switching",
        f"  f_sw       = {p.f_sw / 1e3:8.1f} kHz    (carrier)",
        f"  f_ripple   = {p.f_ripple_effective / 1e3:8.1f} kHz    "
        f"(= 2 f_sw on the load — PD doubling)",
        f"  f_LC corner = {p.f_filter_corner:8.1f} Hz",
        f"  f_ripple / f_LC = {p.f_ripple_effective / p.f_filter_corner:7.1f} ×",
        "",
        "DC bus / NP",
        f"  C_dc (each split cap) = {p.C_dc * 1e6:.0f} µF",
        f"  Nominal NP voltage    = 0 V (= V_dc/2 above V_dc_neg)",
    ]
    return "\n".join(lines)


__all__ = [
    "NPC3PhaseParams",
    "SWITCH_TABLE",
    "switching_state_to_pole_voltage",
    "pd_pwm_state",
    "pd_pwm_state_vectorised",
    "fundamental_pole_voltage",
    "fundamental_line_to_line",
    "thd_voltage_unfiltered",
    "neutral_point_current",
    "clarke_transform",
    "operating_point_report",
]
