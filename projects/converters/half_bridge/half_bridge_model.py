"""Half-bridge converter — analytical model (CCM, voltage-mode).

Sixth entry in `projects/converters/`. First **multi-switch** topology.

What's new vs the forward
-------------------------

The half-bridge has the same small-signal model as the forward
(buck-derived, isolated, no RHP zero), but the IMPLEMENTATION is
substantially different:

  1. **Two switches alternating** (S1 high-side, S2 low-side). One
     conducts during the first half of the period, the other during
     the second half. A short **dead-time** between them prevents
     shoot-through (both on simultaneously → short across V_g).

  2. **Rail-splitter input**: two series caps (C1, C2) across V_g
     produce a midpoint at V_g/2. The transformer primary connects
     between the switching node and this midpoint, so the primary
     sees ±V_g/2 (not ±V_g like a full-bridge).

  3. **Center-tapped secondary** with two rectifier diodes — each
     diode handles one half-cycle. Output current is **continuous** at
     2·f_sw frequency (vs forward's discontinuous current at f_sw).

  4. **No reset winding needed**: each half-cycle resets the
     transformer's flux symmetrically. Duty cycle per switch must
     stay below 0.5 to keep the dead-time intact.

The control-design payoff: the average model is the forward's, so
voltage-mode is buck-easy. The output ripple frequency doubles (2·f_sw),
so the LC filter can be smaller for the same ripple spec.

Conventions
-----------

We define $D$ as the **duty cycle of ONE switch** (per full period —
not per half-period). So $D \\in [0, 0.5]$. The effective conduction
fraction at the output is $2D$ (both half-cycles contribute equally
after rectification).

State vector:
    x = [ î_L ; v̂_o ]   (filter inductor current; output cap voltage)

DC operating point:
    V_o = n · V_g · D    (same form as forward — V_g/2 from rail-split
                          times 2D effective duty gives V_g · D)
    I_L = V_o / R        (load current — buck-like)

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
Section 6.3 (half-bridge converter average model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class HalfBridgeParams:
    """Half-bridge converter parameters (SI units).

    Defaults: 48 V → 5 V at 2 A (10 W output), 1:0.25 transformer
    (n = N_s/N_p = 0.25), 50 µH filter inductor, 200 µF output cap,
    100 kHz switching. With $D = 0.417$ (well below the 0.45 safety
    cap), the output ripple appears at **200 kHz** at the inductor —
    twice the switching frequency.

    Typical use case: 48 V telecom rail → 5 V CPU/logic supply, medium
    power (10 W to a few hundred W).
    """

    V_g: float = 48.0       # input voltage (V)
    V_o: float = 5.0        # output voltage (V)
    R: float = 2.5          # load resistance (Ω) — 2 A load
    L: float = 50e-6        # output filter inductor (H)
    C: float = 200e-6       # output cap (F)
    f_sw: float = 100e3     # per-switch switching frequency (Hz)
    n: float = 0.25         # turns ratio N_s / N_p (per half on secondary)
    D_max: float = 0.45     # per-switch duty cap (must leave dead-time)
    t_dead: float = 100e-9  # dead-time between S1 off and S2 on (s)

    @property
    def D(self) -> float:
        """Steady-state per-switch duty.

        From $V_o = n \\cdot V_g \\cdot D$:
            $D = V_o / (n \\cdot V_g)$.
        """
        return self.V_o / (self.n * self.V_g)

    @property
    def I_L(self) -> float:
        """Steady-state filter inductor current = load current (A)."""
        return self.V_o / self.R

    @property
    def f_ripple(self) -> float:
        """Output ripple frequency (= 2·f_sw because both half-cycles
        contribute to the rectified output)."""
        return 2.0 * self.f_sw

    @property
    def omega_n(self) -> float:
        """LC natural frequency (rad/s) — same as forward/buck."""
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def f_n(self) -> float:
        return self.omega_n / (2.0 * np.pi)

    @property
    def Q(self) -> float:
        return self.R * np.sqrt(self.C / self.L)

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.Q)

    def assert_dead_time_ok(self) -> None:
        """Raise if D and t_dead together would cause shoot-through."""
        T_sw = 1.0 / self.f_sw
        D = self.D
        # Each switch on-time: D · T_sw. Dead-time happens after.
        on_time = D * T_sw
        period_half = T_sw / 2.0
        # On-time + dead-time must fit in a half-period (next switch
        # starts in the second half)
        if on_time + self.t_dead > period_half:
            raise ValueError(
                f"D={D:.3f} + t_dead={self.t_dead*1e9:.0f}ns would force "
                f"shoot-through: D·T_sw + t_dead = "
                f"{(on_time + self.t_dead)*1e6:.2f}µs > T_sw/2 = "
                f"{period_half*1e6:.2f}µs"
            )


def half_bridge_state_space(
    p: HalfBridgeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small-signal (A, B, C, D_feed) for the half-bridge.

    The average model is identical to the forward converter:

        L · di_L/dt = d · n · v_g - v_o
        C · dv_o/dt = i_L - v_o / R

    where d is the duty per switch and v_g is the input bus (NOT
    V_g/2). The 1/2 voltage division from the rail split is exactly
    cancelled by the 2× effective conduction (both half-cycles).

    Linearize around (I_L, V_o, D, V_g):

        L · dî_L/dt = D · n · v̂_g + n · V_g · d̂ - v̂_o
        C · dv̂_o/dt = î_L - v̂_o / R

    No RHP zero — buck-derived.
    """
    L, C, R = p.L, p.C, p.R
    n_t = p.n
    V_g = p.V_g

    A = np.array([
        [0.0,     -1.0 / L],
        [1.0 / C, -1.0 / (R * C)],
    ])
    B = np.array([
        [n_t * V_g / L,  n_t * p.D / L],
        [0.0,            0.0          ],
    ])
    C_mat = np.array([[0.0, 1.0]])
    D_mat = np.zeros((1, 2))
    return A, B, C_mat, D_mat


def control_to_output_tf(p: HalfBridgeParams) -> signal.TransferFunction:
    """$G_{vd}(s) = \\hat v_o / \\hat d$.

        G_vd(s) = (n V_g / (L C)) / (s² + s/(R C) + 1/(L C))

    Second-order low-pass, no zeros. DC gain = $n V_g$, natural
    frequency $\\omega_n = 1/\\sqrt{LC}$.
    """
    L, C, R = p.L, p.C, p.R
    num = [p.n * p.V_g / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def line_to_output_tf(p: HalfBridgeParams) -> signal.TransferFunction:
    L, C, R = p.L, p.C, p.R
    num = [p.n * p.D / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def output_impedance_tf(p: HalfBridgeParams) -> signal.TransferFunction:
    L, C, R = p.L, p.C, p.R
    num = [L, 0.0]
    den = [L * C, L / R, 1.0]
    return signal.TransferFunction(num, den)


# ---------- helpers ----------

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


def operating_point_report(p: HalfBridgeParams) -> str:
    P_out = p.V_o * p.V_o / p.R
    lines = [
        "Half-bridge operating point",
        f"  V_g         = {p.V_g:8.3f} V",
        f"  V_o         = {p.V_o:8.3f} V",
        f"  n = N_s/N_p = {p.n:8.4f}    (per half on secondary)",
        f"  D           = {p.D:8.4f}     (= V_o / (n·V_g))",
        f"  D_max       = {p.D_max:8.4f}     (must leave dead-time)",
        f"  R           = {p.R:8.3f} Ω",
        f"  I_L         = {p.I_L:8.3f} A   (= load current)",
        f"  P_out       = {P_out:8.3f} W",
        "",
        "Two-switch implementation",
        f"  f_sw  per switch = {p.f_sw / 1e3:8.1f} kHz",
        f"  f_ripple output  = {p.f_ripple / 1e3:8.1f} kHz  (= 2·f_sw)",
        f"  Dead-time        = {p.t_dead * 1e9:8.0f} ns",
        "",
        "Rail-split input",
        "  Each cap on the input bus holds V_g/2.",
        f"  Switch voltage stress (per switch off): {p.V_g:.0f} V",
        f"  (vs forward: 2·V_g during reset, vs full-bridge: V_g)",
        "",
        "Filter dynamics (same as forward / buck)",
        f"  L           = {p.L * 1e6:8.3f} µH",
        f"  C           = {p.C * 1e6:8.3f} µF",
        f"  f_n         = {p.f_n:8.1f} Hz",
        f"  Q           = {p.Q:8.3f}",
        "",
        "Control headline",
        "  Same as forward: NO RHP zero, voltage-mode is buck-easy.",
    ]
    if p.D > p.D_max:
        lines.append("")
        lines.append(f"  ⚠️  D = {p.D:.3f} > D_max = {p.D_max:.3f} — would cause shoot-through!")
    return "\n".join(lines)


__all__ = [
    "HalfBridgeParams",
    "half_bridge_state_space",
    "control_to_output_tf",
    "line_to_output_tf",
    "output_impedance_tf",
    "step_response",
    "linf_error",
    "relative_rms_error",
    "operating_point_report",
]
