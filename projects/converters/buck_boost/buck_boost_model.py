"""Buck-boost converter — analytical model (CCM, voltage-mode, inverting).

Third entry in `projects/converters/`. Builds on the buck and boost
references; the student should work through those first.

What's new vs the buck and boost
--------------------------------

The inverting buck-boost is the only single-switch DC-DC topology where:

1. **Output polarity is inverted** — $V_o < 0$ relative to ground.
2. **Both step-down AND step-up are possible** — $D < 0.5$ behaves like
   a buck, $D > 0.5$ behaves like a boost.
3. **The RHP zero is even more punishing than the boost's** — its
   location scales as $1/D$, so high duty cycles (which are needed for
   high step-up ratios) push the zero to LOWER frequencies, capping the
   loop bandwidth tighter.

Conventions
-----------

We model $v_o$ as a **positive scalar** equal to the *magnitude* of the
output voltage; the inversion sign is a topology detail handled by the
physical layout. This keeps the math clean and lets us reuse the
boost's derivation almost verbatim.

State vector:
    x = [ î_L ; v̂_o ]    (same as boost)

Input vector:
    u = [ d̂ ; v̂_g ]

DC operating point:
    V_o = (D / (1 - D)) · V_g
    I_L = V_o / (R · (1 - D))     (same I_L formula as boost)

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
Section 7.5 (buck-boost average model) and Eq. 8.55 (RHP zero formula).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BuckBoostParams:
    """Inverting buck-boost converter parameters (SI units).

    Defaults: 12 V → ±12 V (magnitude) at ~12 W (D = 0.5 is the
    buck-boost crossover, where |V_o| = V_g). At this duty the LC pole
    sits at ~800 Hz, Q ≈ 6, and the RHP zero is at ~9.5 kHz — a comfort­
    able starting point for control design that still demonstrates all
    the buck-boost's pedagogically important quirks.
    """

    V_g: float = 12.0       # input voltage (V)
    V_o: float = 12.0       # output voltage magnitude (V); actual polarity is negative
    R: float = 12.0         # load resistance (Ω); R = V_o²/P_out for 12 W
    L: float = 100e-6       # inductor (H)
    C: float = 100e-6       # output capacitor (F)
    f_sw: float = 100e3     # switching frequency (Hz)

    @property
    def D(self) -> float:
        """Steady-state duty (V_o = D/(1-D) · V_g → D = V_o/(V_o + V_g))."""
        return self.V_o / (self.V_o + self.V_g)

    @property
    def I_L(self) -> float:
        """Steady-state inductor current (A)."""
        return self.V_o / (self.R * (1.0 - self.D))

    @property
    def omega_n(self) -> float:
        """LC double-pole natural angular frequency (rad/s)."""
        return (1.0 - self.D) / np.sqrt(self.L * self.C)

    @property
    def f_n(self) -> float:
        return self.omega_n / (2.0 * np.pi)

    @property
    def omega_z_rhp(self) -> float:
        """RHP zero (rad/s) — `R · (1-D)² / (L · D)`. INVERSE in D,
        so high duty makes this zero LOWER → tighter bandwidth cap."""
        return self.R * (1.0 - self.D) ** 2 / (self.L * self.D)

    @property
    def f_z_rhp(self) -> float:
        return self.omega_z_rhp / (2.0 * np.pi)

    @property
    def Q(self) -> float:
        return (1.0 - self.D) * self.R * np.sqrt(self.C / self.L)

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.Q)


def buck_boost_state_space(
    p: BuckBoostParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small-signal (A, B, C, D) of the inverting buck-boost in CCM.

    Average ODEs (after q→d substitution):

        L · di_L/dt = d · v_g - (1-d) · v_o
        C · dv_o/dt = (1-d) · i_L - v_o / R

    Linearize around (I_L, V_o, D, V_g). The same `-I_L · d̂` term that
    gave the boost its RHP zero shows up here too — the buck-boost
    inherits the boost's non-minimum-phase headache.

        L · dî_L/dt = D · v̂_g + (V_g + V_o) · d̂ - (1-D) · v̂_o
        C · dv̂_o/dt = (1-D) · î_L - I_L · d̂ - v̂_o / R

    Note `V_g + V_o = V_g / (1-D)` at the operating point; this appears
    in the B matrix's d̂-column for the inductor row.

    Returns (A, B, C, D) with:
        x = [ î_L ; v̂_o ],  u = [ d̂ ; v̂_g ]
    """
    L, C, R = p.L, p.C, p.R
    Dprime = 1.0 - p.D
    V_g = p.V_g
    V_o = p.V_o
    I_L = p.I_L

    A = np.array([
        [0.0,         -Dprime / L],
        [Dprime / C,  -1.0 / (R * C)],
    ])
    B = np.array([
        [(V_g + V_o) / L,  p.D / L],
        [-I_L / C,         0.0    ],
    ])
    C_mat = np.array([[0.0, 1.0]])
    D_mat = np.zeros((1, 2))
    return A, B, C_mat, D_mat


def control_to_output_tf(p: BuckBoostParams) -> signal.TransferFunction:
    """Return $G_{vd}(s) = \\hat v_o(s) / \\hat d(s)$ as a positive-magnitude
    transfer function.

    Closed form (Erickson Eq 8.55, taking magnitudes):

        G_vd(s) = G_d0 · (1 - s / ω_z_rhp) / (1 + s/(Q ω_n) + (s/ω_n)²)

    with $G_{d0} = V_g / (1-D)^2$.

    The actual converter inverts polarity → real $\\hat v_o = -G_{vd}(s) \\hat d$.
    Treating $v_o$ as positive magnitude keeps the math identical to the
    boost's and is the convention used throughout the notebooks.
    """
    omega_n = p.omega_n
    omega_z = p.omega_z_rhp
    Q = p.Q
    G_dc = p.V_g / (1.0 - p.D) ** 2
    num = [-G_dc / omega_z, G_dc]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


def line_to_output_tf(p: BuckBoostParams) -> signal.TransferFunction:
    """Return G_vg(s) = v̂_o / v̂_g.

    Closed form:
        G_vg(s) = (D/(1-D)) / (1 + s/(Q ω_n) + (s/ω_n)²)

    Same denominator as G_vd, DC gain = D/(1-D) = V_o/V_g. No RHP zero.
    """
    omega_n = p.omega_n
    Q = p.Q
    G_dc = p.D / (1.0 - p.D)
    num = [G_dc]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


def output_impedance_tf(p: BuckBoostParams) -> signal.TransferFunction:
    """Return Z_out(s) = v̂_o / î_load.

    Closed form:
        Z_out(s) = (sL/(1-D)²) / (1 + s/(Q ω_n) + (s/ω_n)²)

    Single zero at the origin; peaks at ω_n.
    """
    L = p.L
    Dprime2 = (1.0 - p.D) ** 2
    omega_n = p.omega_n
    Q = p.Q
    num = [L / Dprime2, 0.0]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


# Convenience helpers — same shapes as buck_model / boost_model
def step_response(tf, t, step_magnitude: float = 1.0) -> np.ndarray:
    _, y = signal.step(tf, T=t)
    return step_magnitude * y


def linf_error(y_ref, y_test) -> float:
    return float(np.max(np.abs(np.asarray(y_ref) - np.asarray(y_test))))


def relative_rms_error(y_ref, y_test) -> float:
    err = np.asarray(y_ref) - np.asarray(y_test)
    ref = np.asarray(y_ref)
    rms_err = float(np.sqrt(np.mean(err ** 2)))
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    if rms_ref < 1e-12:
        return rms_err
    return rms_err / rms_ref


def operating_point_report(p: BuckBoostParams) -> str:
    P_out = p.V_o * p.V_o / p.R
    lines = [
        "Inverting buck-boost operating point",
        f"  V_g     = {p.V_g:8.3f} V",
        f"  V_o     = {p.V_o:8.3f} V       (magnitude; actual polarity is negative)",
        f"  D       = {p.D:8.4f}         (= V_o / (V_o + V_g))",
        f"  R       = {p.R:8.3f} Ω",
        f"  I_L     = {p.I_L:8.3f} A",
        f"  P_out   = {P_out:8.3f} W",
        f"  Mode    = {'BUCK-like (D < 0.5)' if p.D < 0.5 else 'BOOST-like (D ≥ 0.5)'}",
        "",
        "Filter dynamics",
        f"  L       = {p.L * 1e6:8.3f} µH",
        f"  C       = {p.C * 1e6:8.3f} µF",
        f"  f_n     = {p.f_n:8.1f} Hz      (LC double pole — drops as D rises)",
        f"  Q       = {p.Q:8.3f}",
        "",
        "RHP zero — the buck-boost's nemesis",
        f"  f_z_rhp = {p.f_z_rhp:8.1f} Hz     (∝ 1/D — gets WORSE at high duty)",
        f"  Practical cap: f_c ≤ f_z_rhp / 5 = {p.f_z_rhp / 5:.0f} Hz",
        "",
        "Switching",
        f"  f_sw    = {p.f_sw / 1e3:8.1f} kHz",
        f"  f_sw / f_n      = {p.f_sw / p.f_n:8.1f}×",
        f"  f_sw / f_z_rhp  = {p.f_sw / p.f_z_rhp:7.1f}×",
    ]
    return "\n".join(lines)


__all__ = [
    "BuckBoostParams",
    "buck_boost_state_space",
    "control_to_output_tf",
    "line_to_output_tf",
    "output_impedance_tf",
    "step_response",
    "linf_error",
    "relative_rms_error",
    "operating_point_report",
]
