"""Buck converter — analytical model (CCM, voltage-mode).

This module is imported by both notebooks in this folder:

  * ``01_buck_modeling.ipynb``  — derives and validates the model.
  * ``02_buck_controller.ipynb`` — designs a compensator on top of it.

Keeping the functions here (instead of duplicating them inline in
notebooks) lets us unit-test them and lets the controller notebook
reuse the modeling primitives without copy-paste.

Conventions
-----------

State vector
    x = [ î_L ; v̂_o ]  (inductor current first, capacitor voltage second)

Input vector
    u = [ d̂ ; v̂_g ]   (control input first: duty perturbation,
                         disturbance second: line voltage perturbation)

Output
    y = v̂_o          (small-signal output-voltage perturbation)

Sign convention
    All hatted quantities are perturbations around an operating point.
    Capital letters denote the operating-point values:
    ``V_g`` (input bus), ``D`` (steady duty), ``V_o = D * V_g``,
    ``I_L = V_o / R`` (load current in CCM).

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
Chapters 7 (state-space averaging) and 8 (transfer functions). The
matrices below are equation (7.108) in that text, restricted to the
ideal-buck case (no parasitics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BuckParams:
    """Buck-converter design parameters.

    All values in SI units. ``D`` is the steady-state duty cycle the
    converter is designed for — average model is linearized around
    this operating point.

    Defaults match a representative 12 V / 5 A digital-power point of
    load with a 100 µH filter inductor and 100 µF output cap, ~1 MHz
    natural frequency cut. Easy to relate to common datasheets.
    """

    V_g: float = 24.0       # input bus voltage (V)
    V_o: float = 12.0       # target output voltage (V)
    R: float = 2.4          # load resistance (Ω); R = V_o^2 / P_out
    L: float = 100e-6       # filter inductor (H)
    C: float = 100e-6       # output capacitor (F)
    f_sw: float = 100e3     # switching frequency (Hz)

    @property
    def D(self) -> float:
        """Steady-state duty cycle (assumes ideal-buck CCM: V_o = D*V_g)."""
        return self.V_o / self.V_g

    @property
    def I_L(self) -> float:
        """Steady-state inductor (= load) current (A)."""
        return self.V_o / self.R

    @property
    def omega_n(self) -> float:
        """Natural angular frequency of the LC filter (rad/s)."""
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def f_n(self) -> float:
        """Natural frequency of the LC filter (Hz)."""
        return self.omega_n / (2.0 * np.pi)

    @property
    def zeta(self) -> float:
        """Damping ratio of the LC filter (loaded by R)."""
        return (1.0 / (2.0 * self.R)) * np.sqrt(self.L / self.C)

    @property
    def Q(self) -> float:
        """Q-factor of the LC filter (1 / (2·ζ))."""
        return 1.0 / (2.0 * self.zeta)


# ----------------------------------------------------------------------
# State-space model
# ----------------------------------------------------------------------


def buck_state_space(params: BuckParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the small-signal state-space (A, B, C, D) of an ideal CCM buck.

    Derivation (one-line summary — the notebook walks through every
    step):

        L · di_L/dt = d·v_g − v_o
        C · dv_o/dt = i_L − v_o / R

    Linearize around (V_g, V_o = D·V_g, I_L = V_o/R):

        L · dî_L/dt = D·v̂_g + V_g·d̂ − v̂_o
        C · dv̂_o/dt = î_L − v̂_o / R

    Rewrite as dx/dt = A·x + B·u with

        x = [ î_L ; v̂_o ]   u = [ d̂ ; v̂_g ]

    Returns
    -------
    A : (2, 2) np.ndarray
    B : (2, 2) np.ndarray   # column 0: d̂ input, column 1: v̂_g input
    C : (1, 2) np.ndarray   # output = v̂_o
    D : (1, 2) np.ndarray   # no direct feedthrough
    """
    L, C, R = params.L, params.C, params.R
    V_g, D = params.V_g, params.D

    A = np.array([
        [0.0,     -1.0 / L],
        [1.0 / C, -1.0 / (R * C)],
    ])
    B = np.array([
        [V_g / L, D / L],   # row 1: dî_L/dt contributions from (d̂, v̂_g)
        [0.0,     0.0  ],   # row 2: dv̂_o/dt has no direct input
    ])
    C_mat = np.array([[0.0, 1.0]])  # y = v̂_o
    D_mat = np.zeros((1, 2))
    return A, B, C_mat, D_mat


# ----------------------------------------------------------------------
# Transfer functions
# ----------------------------------------------------------------------


def control_to_output_tf(params: BuckParams) -> signal.TransferFunction:
    """Return Gvd(s) = v̂_o(s) / d̂(s) as a scipy TransferFunction.

    Closed-form for the ideal CCM buck:

        Gvd(s) = (V_g / (L·C)) / (s² + s/(R·C) + 1 / (L·C))

    The DC gain is V_g (a 1% duty change moves the output by 1% · V_g
    at DC, which matches V_o = D · V_g). The double pole sits at
    ω_n = 1/√(L·C); the Q factor is R · √(C / L).
    """
    L, C, R = params.L, params.C, params.R
    V_g = params.V_g

    num = [V_g / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def line_to_output_tf(params: BuckParams) -> signal.TransferFunction:
    """Return Gvg(s) = v̂_o(s) / v̂_g(s).

        Gvg(s) = (D / (L·C)) / (s² + s/(R·C) + 1 / (L·C))

    Same denominator as Gvd but DC gain = D.
    """
    L, C, R = params.L, params.C, params.R
    D = params.D

    num = [D / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def output_impedance_tf(params: BuckParams) -> signal.TransferFunction:
    """Return Zout(s) = v̂_o(s) / î_load(s) — open-loop output impedance.

    Treating an external load current perturbation î_load as an input
    that subtracts from the capacitor current node:

        Zout(s) = (s · L) / (L·C · s² + L/R · s + 1)

    Two zeros at the origin (only the s·L term in the numerator → 1 zero
    at origin). DC value = 0 (ideal source rejects DC load step at DC).
    Peak impedance at ω_n, magnitude ≈ L · ω_n · Q = √(L/C) · Q.
    """
    L, C, R = params.L, params.C, params.R

    num = [L, 0.0]
    den = [L * C, L / R, 1.0]
    return signal.TransferFunction(num, den)


# ----------------------------------------------------------------------
# Numerical helpers used by the validation cells
# ----------------------------------------------------------------------


def step_response(
    tf: signal.TransferFunction,
    t: np.ndarray,
    step_magnitude: float = 1.0,
) -> np.ndarray:
    """Return the response of ``tf`` to a step of given magnitude on its input.

    Thin wrapper around :func:`scipy.signal.step` that scales by the
    step magnitude so callers can pass e.g. a 1% duty step
    (``step_magnitude=0.01``) without remembering scipy's unit-step
    convention.
    """
    _, y = signal.step(tf, T=t)
    return step_magnitude * y


def bode(
    tf: signal.TransferFunction,
    f_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude (dB) and phase (deg) of ``tf`` at the given Hz grid."""
    w = 2.0 * np.pi * f_hz
    _, mag_db, phase_deg = signal.bode(tf, w=w)
    return mag_db, phase_deg


def linf_error(y_ref: np.ndarray, y_test: np.ndarray) -> float:
    """L-infinity (max-abs) error between two equally-sampled signals."""
    return float(np.max(np.abs(np.asarray(y_ref) - np.asarray(y_test))))


def relative_rms_error(y_ref: np.ndarray, y_test: np.ndarray) -> float:
    """RMS error normalized by the RMS of the reference signal."""
    err = np.asarray(y_ref) - np.asarray(y_test)
    ref = np.asarray(y_ref)
    rms_err = float(np.sqrt(np.mean(err**2)))
    rms_ref = float(np.sqrt(np.mean(ref**2)))
    if rms_ref < 1e-12:
        return rms_err
    return rms_err / rms_ref


# ----------------------------------------------------------------------
# Operating-point sanity checks
# ----------------------------------------------------------------------


def operating_point_report(params: BuckParams) -> str:
    """Return a human-readable summary of the analytical operating point.

    Used at the top of the modeling notebook so students can sanity-check
    their parameter set before diving into the small-signal math.
    """
    p = params
    lines = [
        f"Buck operating point",
        f"  V_g    = {p.V_g:8.3f} V",
        f"  V_o    = {p.V_o:8.3f} V       (target)",
        f"  D      = {p.D:8.4f}         (= V_o / V_g)",
        f"  R      = {p.R:8.3f} Ω",
        f"  I_L    = {p.I_L:8.3f} A       (= V_o / R)",
        f"  P_out  = {p.V_o * p.I_L:8.3f} W",
        f"",
        f"Filter dynamics",
        f"  L      = {p.L * 1e6:8.3f} µH",
        f"  C      = {p.C * 1e6:8.3f} µF",
        f"  f_n    = {p.f_n:8.1f} Hz      (LC natural)",
        f"  ω_n    = {p.omega_n:8.1f} rad/s",
        f"  ζ      = {p.zeta:8.3f}         (damping ratio)",
        f"  Q      = {p.Q:8.3f}         (1 / 2ζ)",
        f"",
        f"Switching",
        f"  f_sw   = {p.f_sw / 1e3:8.1f} kHz",
        f"  f_sw / f_n = {p.f_sw / p.f_n:8.1f}× (must be >> 1 for averaging to hold)",
    ]
    return "\n".join(lines)


__all__ = [
    "BuckParams",
    "buck_state_space",
    "control_to_output_tf",
    "line_to_output_tf",
    "output_impedance_tf",
    "step_response",
    "bode",
    "linf_error",
    "relative_rms_error",
    "operating_point_report",
]
