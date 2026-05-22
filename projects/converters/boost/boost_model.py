"""Boost converter — analytical model (CCM, voltage-mode).

Shared model functions imported by `01_boost_modeling.ipynb` and
`02_boost_controller.ipynb`. Same conventions as `buck_model.py` so
students can compare the two derivations side by side.

Conventions
-----------

State vector
    x = [ î_L ; v̂_o ]

Input vector
    u = [ d̂ ; v̂_g ]    (control input first, line disturbance second)

Output
    y = v̂_o

Operating point
    V_o = V_g / (1 - D)       (boost ratio — step-up)
    I_L = V_o / (R · (1 - D)) (input current must support output power)

Why the boost is *fundamentally* harder than the buck
-----------------------------------------------------

Because of where energy is stored. During the ON interval the
switch shorts the inductor to ground, so the output capacitor
**dis-charges** through the load until the OFF interval delivers a
fresh packet of energy. When the control loop commands MORE duty (more
ON time), the immediate effect is *less* time delivering energy to the
output → the output **dips** before it eventually rises. That is the
classic non-minimum-phase / right-half-plane (RHP) zero of the boost.

For control design the RHP zero is the dominant headache: it caps the
maximum achievable closed-loop bandwidth at roughly $f_z / 5$ to
$f_z / 10$. Trying to push the loop faster causes the controller to
add duty when it should remove it, destabilizing the loop.

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
Section 7.6 (boost averaged model) and 8.2 (RHP zero, control
implications).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BoostParams:
    """Boost-converter design parameters (SI units throughout).

    Defaults: 12 V → 24 V step-up at 50 W output, 100 kHz switching,
    100 µH input inductor, 100 µF output cap. This sits firmly in CCM
    at full load (D = 0.5) and gives an RHP zero around 4 kHz — well
    below the LC corner, so the controller is RHP-zero-limited.
    """

    V_g: float = 12.0       # input voltage (V)
    V_o: float = 24.0       # target output voltage (V)
    R: float = 11.52        # load resistance (Ω); R = V_o²/P_out at 50 W
    L: float = 100e-6       # input/boost inductor (H)
    C: float = 100e-6       # output capacitor (F)
    f_sw: float = 100e3     # switching frequency (Hz)

    @property
    def D(self) -> float:
        """Steady-state duty (ideal-CCM: V_o = V_g/(1-D), so D = 1 - V_g/V_o)."""
        return 1.0 - self.V_g / self.V_o

    @property
    def I_L(self) -> float:
        """Steady-state inductor current.

        Power balance: V_g · I_L = V_o² / R  (ideal converter).
        Equivalently: I_L = V_o / (R · (1 - D)) = I_out / (1 - D).
        """
        return self.V_o / (self.R * (1.0 - self.D))

    @property
    def omega_n(self) -> float:
        """Natural angular frequency of the LC double pole (rad/s).

        ω_n = (1 - D) / √(LC). Note the (1-D) factor — the LC corner
        DROPS as duty increases, which is why high-D boosts are harder
        to compensate (resonance moves down to the bandwidth region).
        """
        return (1.0 - self.D) / np.sqrt(self.L * self.C)

    @property
    def f_n(self) -> float:
        """Natural frequency of the LC double pole (Hz)."""
        return self.omega_n / (2.0 * np.pi)

    @property
    def omega_z_rhp(self) -> float:
        """Right-half-plane zero (rad/s).

        ω_z = R · (1 - D)² / L. This is THE design constraint for boost
        voltage-mode control: closed-loop bandwidth must be well below
        this frequency.
        """
        return self.R * (1.0 - self.D) ** 2 / self.L

    @property
    def f_z_rhp(self) -> float:
        """Right-half-plane zero frequency (Hz)."""
        return self.omega_z_rhp / (2.0 * np.pi)

    @property
    def Q(self) -> float:
        """Quality factor of the LC double pole."""
        return (1.0 - self.D) * self.R * np.sqrt(self.C / self.L)

    @property
    def zeta(self) -> float:
        """Damping ratio of the LC double pole (= 1 / (2 Q))."""
        return 1.0 / (2.0 * self.Q)


# ----------------------------------------------------------------------
# State-space model
# ----------------------------------------------------------------------


def boost_state_space(
    params: BoostParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the small-signal (A, B, C, D) of an ideal CCM boost.

    Derivation (one-line summary — the notebook walks every step):

        L · di_L/dt = (1 - q(t)) · (v_g - v_o) + q(t) · v_g  =  v_g - (1-q) v_o
        C · dv_o/dt = (1 - q(t)) · i_L - v_o / R

    Average q(t) → d, perturb around the operating point
    (I_L, V_o, D, V_g), drop products, get:

        L · dî_L/dt = -(1-D) · v̂_o + V_o · d̂ + v̂_g
        C · dv̂_o/dt =  (1-D) · î_L - v̂_o / R - I_L · d̂

    The **non-minimum-phase term** is `-I_L · d̂` in the v̂_o equation:
    a positive duty step momentarily *reduces* dv̂_o/dt before î_L has
    time to build up. That's what produces the RHP zero in G_vd(s).

    Returns
    -------
    A : (2, 2) np.ndarray
    B : (2, 2) np.ndarray  # col 0: d̂, col 1: v̂_g
    C : (1, 2) np.ndarray
    D : (1, 2) np.ndarray  (no direct feedthrough)
    """
    L, C, R = params.L, params.C, params.R
    Dprime = 1.0 - params.D    # 1 - D appears a LOT in boost math
    V_o = params.V_o
    I_L = params.I_L

    A = np.array([
        [0.0,        -Dprime / L],
        [Dprime / C, -1.0 / (R * C)],
    ])
    B = np.array([
        [ V_o / L,  1.0 / L],   # dî_L/dt contributions from (d̂, v̂_g)
        [-I_L / C,  0.0    ],   # dv̂_o/dt has the NON-MIN-PHASE term -I_L·d̂
    ])
    C_mat = np.array([[0.0, 1.0]])
    D_mat = np.zeros((1, 2))
    return A, B, C_mat, D_mat


# ----------------------------------------------------------------------
# Transfer functions
# ----------------------------------------------------------------------


def control_to_output_tf(params: BoostParams) -> signal.TransferFunction:
    """Return G_vd(s) = v̂_o / d̂.

    Closed form (Erickson Eq 8.49 for the ideal CCM boost):

        G_vd(s) = G_vd0 · (1 - s/ω_z_rhp) / (1 + s/(Q ω_n) + (s/ω_n)²)

    where G_vd0 = V_o / (1 - D), ω_z_rhp = R(1-D)²/L, ω_n = (1-D)/√(LC),
    Q = (1-D)·R·√(C/L).
    """
    omega_n = params.omega_n
    omega_z = params.omega_z_rhp
    Q = params.Q
    G_dc = params.V_o / (1.0 - params.D)

    # Numerator: G_dc · (1 - s/ω_z)  →  G_dc - G_dc·s/ω_z  →  in numpy
    # polynomial form (descending powers of s): [-G_dc/ω_z, G_dc]
    num = [-G_dc / omega_z, G_dc]

    # Denominator: 1 + s/(Q ω_n) + (s/ω_n)²  →  in numpy form:
    # [1/ω_n², 1/(Q ω_n), 1]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


def line_to_output_tf(params: BoostParams) -> signal.TransferFunction:
    """Return G_vg(s) = v̂_o / v̂_g.

    Closed form: G_vg(s) = (1/(1-D)) / (1 + s/(Q ω_n) + (s/ω_n)²).

    Same denominator as G_vd, DC gain 1/(1-D). NO RHP zero — line
    disturbances propagate as a clean second-order low-pass.
    """
    omega_n = params.omega_n
    Q = params.Q
    G_dc = 1.0 / (1.0 - params.D)
    num = [G_dc]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


def output_impedance_tf(params: BoostParams) -> signal.TransferFunction:
    """Return Z_out(s) — open-loop output impedance under a load-current
    perturbation. For the ideal boost:

        Z_out(s) = (s · L / (1-D)²) / (1 + s/(Q ω_n) + (s/ω_n)²)

    Single zero at the origin; peaks at ω_n.
    """
    L = params.L
    Dprime2 = (1.0 - params.D) ** 2
    omega_n = params.omega_n
    Q = params.Q
    num = [L / Dprime2, 0.0]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


# ----------------------------------------------------------------------
# Helpers used by validation cells
# ----------------------------------------------------------------------


def step_response(
    tf: signal.TransferFunction, t: np.ndarray, step_magnitude: float = 1.0
) -> np.ndarray:
    _, y = signal.step(tf, T=t)
    return step_magnitude * y


def linf_error(y_ref: np.ndarray, y_test: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(y_ref) - np.asarray(y_test))))


def relative_rms_error(y_ref: np.ndarray, y_test: np.ndarray) -> float:
    err = np.asarray(y_ref) - np.asarray(y_test)
    ref = np.asarray(y_ref)
    rms_err = float(np.sqrt(np.mean(err ** 2)))
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    if rms_ref < 1e-12:
        return rms_err
    return rms_err / rms_ref


def operating_point_report(params: BoostParams) -> str:
    """Human-readable summary of the analytical operating point."""
    p = params
    P_out = p.V_o ** 2 / p.R
    lines = [
        "Boost operating point",
        f"  V_g    = {p.V_g:8.3f} V",
        f"  V_o    = {p.V_o:8.3f} V       (target)",
        f"  D      = {p.D:8.4f}         (= 1 - V_g/V_o)",
        f"  R      = {p.R:8.3f} Ω",
        f"  I_out  = {p.V_o / p.R:8.3f} A       (= V_o / R)",
        f"  I_L    = {p.I_L:8.3f} A       (= V_o / (R·(1-D)))",
        f"  P_out  = {P_out:8.3f} W",
        "",
        "Filter dynamics",
        f"  L      = {p.L * 1e6:8.3f} µH",
        f"  C      = {p.C * 1e6:8.3f} µF",
        f"  f_n    = {p.f_n:8.1f} Hz      (LC pole pair — drops as D rises)",
        f"  Q      = {p.Q:8.3f}         (LC quality factor)",
        f"  ζ      = {p.zeta:8.3f}",
        "",
        "RHP zero (the boost's nemesis)",
        f"  f_z_rhp = {p.f_z_rhp:8.1f} Hz",
        f"  Bandwidth must be ≪ f_z_rhp — typical f_c ≤ f_z_rhp/5 = "
        f"{p.f_z_rhp / 5:.0f} Hz",
        "",
        "Switching",
        f"  f_sw   = {p.f_sw / 1e3:8.1f} kHz",
        f"  f_sw / f_n   = {p.f_sw / p.f_n:8.1f}×",
        f"  f_sw / f_z_rhp = {p.f_sw / p.f_z_rhp:7.1f}×",
    ]
    return "\n".join(lines)


__all__ = [
    "BoostParams",
    "boost_state_space",
    "control_to_output_tf",
    "line_to_output_tf",
    "output_impedance_tf",
    "step_response",
    "linf_error",
    "relative_rms_error",
    "operating_point_report",
]
