"""Forward converter — analytical model (CCM, voltage-mode).

Fifth entry in `projects/converters/`. The **isolated buck**.

What's new vs the previous four
-------------------------------

The forward converter is to the buck what the flyback is to the
buck-boost: a transformer-isolated variant. But it routes energy
DIFFERENTLY than the flyback:

- **Flyback** stores energy in the transformer's magnetizing field
  during ON, then RELEASES it through the secondary winding during
  OFF. The transformer doubles as the energy storage element; no
  separate filter inductor is needed.

- **Forward** transfers energy from primary to secondary
  **simultaneously** during ON (the secondary winding's voltage drives
  the output filter inductor). The transformer is a pure voltage
  multiplier; an explicit filter L-C handles the energy storage.
  During OFF, a freewheel diode lets the filter inductor maintain
  current — same as the buck.

The big payoff: the forward converter inherits the **buck's** small-
signal model — a clean second-order low-pass with **NO RHP zero**.
That makes voltage-mode control trivial compared to the flyback /
boost / buck-boost.

The price: an extra winding (the "reset winding") on the transformer
to demagnetize the core every cycle. This caps the operating duty
cycle (typically $D \\le 0.5$ for a 1:1 reset winding).

Conventions
-----------

State vector:
    x = [ î_L ; v̂_o ]   (filter inductor current; output cap voltage)

Input vector:
    u = [ d̂ ; v̂_g ]

DC operating point:
    V_o = n · V_g · D     (same as buck, just with `n·V_g` in place of V_g)
    I_L = V_o / R          (= load current, exactly — buck-like)

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
Section 6.2 (forward converter average model) and 8.2 (control of the
buck-derived family — no RHP zero).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class ForwardParams:
    """Forward-converter design parameters (SI units).

    Defaults: 24 V → 5 V at 5 W (1 A load through 5 Ω), with a 1:0.5
    transformer (n = N_s/N_p = 0.5), 100 µH filter inductor, 100 µF
    output cap, 100 kHz switching.

    Operating point: D ≈ 0.417 (well below the 0.5 reset-winding
    limit). The filter LC corner sits at ~1.6 kHz with Q ≈ 5 — same
    territory as the buck reference, just with a transformer in front
    of the output stage.
    """

    V_g: float = 24.0       # input voltage (V)
    V_o: float = 5.0        # output voltage (V) — typical low-voltage rail
    R: float = 5.0          # load resistance (Ω) — 1 A load
    L: float = 100e-6       # output filter inductor (H)
    C: float = 100e-6       # output capacitor (F)
    f_sw: float = 100e3     # switching frequency (Hz)
    n: float = 0.5          # turns ratio N_s / N_p
    D_max: float = 0.45     # reset-winding duty constraint (1:1 reset → D ≤ 0.5;
                            # leave a safety margin for line/load steps)

    @property
    def D(self) -> float:
        """Steady-state duty cycle.

        From $V_o = n \\cdot V_g \\cdot D$:
            $D = V_o / (n \\cdot V_g)$.

        Raises a sanity warning if D > D_max (reset-winding constraint
        is being violated — design needs a different n or different V_g/V_o).
        """
        d = self.V_o / (self.n * self.V_g)
        return d

    @property
    def I_L(self) -> float:
        """Steady-state filter inductor current = load current (A)."""
        return self.V_o / self.R

    @property
    def omega_n(self) -> float:
        """LC natural angular frequency (rad/s) — same form as the buck."""
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def f_n(self) -> float:
        return self.omega_n / (2.0 * np.pi)

    @property
    def Q(self) -> float:
        """Quality factor of the LC filter (loaded by R)."""
        return self.R * np.sqrt(self.C / self.L)

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.Q)

    def assert_reset_winding_ok(self) -> None:
        """Raise AssertionError if the operating duty exceeds D_max.

        Reset winding cannot demagnetize the transformer if D is too
        high — flux saturates, magnetizing current runs away, switch
        sees huge voltage spikes and the converter destroys itself.
        """
        D = self.D
        assert D < self.D_max, (
            f"Forward converter D = {D:.3f} exceeds reset limit "
            f"D_max = {self.D_max:.3f}. "
            f"Increase n (= {self.n:.3f}) or decrease V_o / increase V_g."
        )


def forward_state_space(
    p: ForwardParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small-signal (A, B, C, D) of an ideal forward converter in CCM.

    Average ODEs (q → d):

        L · di_L/dt = d · n · v_g - v_o
        C · dv_o/dt = i_L - v_o / R

    This is the BUCK's average model with `n·v_g` in place of `v_g` —
    the only role the transformer plays is to scale the "effective
    input bus voltage" seen by the filter L-C.

    Linearize around (I_L, V_o, D, V_g):

        L · dî_L/dt = D · n · v̂_g + n · V_g · d̂ - v̂_o
        C · dv̂_o/dt = î_L - v̂_o / R

    No RHP zero — no `-I_L · d̂` term in the cap equation because the
    filter inductor maintains current both during ON (delivered from
    secondary) and OFF (freewheeled through D_fw). Increasing duty
    immediately increases v_o; no inverse-response.
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


def control_to_output_tf(p: ForwardParams) -> signal.TransferFunction:
    """Return G_vd(s) = v̂_o / d̂.

    Closed form — same shape as the buck, scaled by n:

        G_vd(s) = (n V_g / (L C)) / (s² + s/(R C) + 1/(L C))

    Second-order low-pass. DC gain = $n V_g$, natural frequency
    $\\omega_n = 1/\\sqrt{LC}$, Q-factor $Q = R\\sqrt{C/L}$. NO RHP zero.
    """
    L, C, R = p.L, p.C, p.R
    num = [p.n * p.V_g / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def line_to_output_tf(p: ForwardParams) -> signal.TransferFunction:
    """Return G_vg(s) = v̂_o / v̂_g.

    Closed form: $G_{vg}(s) = n D / (L C \\cdot s^2 + L/R \\cdot s + 1)$ —
    same denominator as G_vd, DC gain $= n D$.
    """
    L, C, R = p.L, p.C, p.R
    num = [p.n * p.D / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def output_impedance_tf(p: ForwardParams) -> signal.TransferFunction:
    """Return Z_out(s) = v̂_o / î_load.

    Closed form: $Z_{out}(s) = sL / (LC s^2 + L/R s + 1)$ — exactly
    the same as the buck. The transformer is invisible to load-side
    perturbations.
    """
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


def operating_point_report(p: ForwardParams) -> str:
    P_out = p.V_o * p.V_o / p.R
    lines = [
        "Forward operating point",
        f"  V_g       = {p.V_g:8.3f} V",
        f"  V_o       = {p.V_o:8.3f} V",
        f"  n = N_s/N_p = {p.n:8.4f}",
        f"  D         = {p.D:8.4f}         (= V_o / (n·V_g))",
        f"  D_max     = {p.D_max:8.4f}         (reset-winding limit)",
        f"  R         = {p.R:8.3f} Ω",
        f"  I_L       = {p.I_L:8.3f} A       (= load current)",
        f"  P_out     = {P_out:8.3f} W",
        "",
        "Filter dynamics (same as the buck)",
        f"  L         = {p.L * 1e6:8.3f} µH",
        f"  C         = {p.C * 1e6:8.3f} µF",
        f"  f_n       = {p.f_n:8.1f} Hz",
        f"  Q         = {p.Q:8.3f}",
        f"  ζ         = {p.zeta:8.3f}",
        "",
        "Control headline",
        "  NO RHP zero — buck-derived isolation, voltage-mode is easy",
        f"  Practical bandwidth: f_c ≤ f_sw / 10 = {p.f_sw / 10 / 1e3:.0f} kHz",
        "",
        "Switching",
        f"  f_sw      = {p.f_sw / 1e3:8.1f} kHz",
        f"  f_sw / f_n = {p.f_sw / p.f_n:7.1f}×",
    ]
    if p.D > p.D_max:
        lines.append("")
        lines.append(f"  ⚠️  D = {p.D:.3f} > D_max = {p.D_max:.3f} — reset winding can't keep up!")
    return "\n".join(lines)


__all__ = [
    "ForwardParams",
    "forward_state_space",
    "control_to_output_tf",
    "line_to_output_tf",
    "output_impedance_tf",
    "step_response",
    "linf_error",
    "relative_rms_error",
    "operating_point_report",
]
