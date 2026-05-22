"""Flyback converter — analytical model (CCM, voltage-mode).

Fourth entry in `projects/converters/`. First **isolated** topology.

What's new vs the previous three references
--------------------------------------------

The flyback is a buck-boost where the inductor is split into a coupled
primary + secondary winding (a transformer with intentional
magnetizing inductance). This gives the converter two new powers:

1. **Galvanic isolation** between input and output — no shared
   ground, so you can stack flybacks, drive earth-referenced or
   floating loads, and meet safety isolation requirements.

2. **A "duty-cycle multiplier"** via the turns ratio. The
   steady-state ratio is

   $$ V_o = n \\cdot V_g \\cdot \\frac{D}{1-D}, \\quad n = \\frac{N_s}{N_p} $$

   The user picks $n$ at design time so the operating $D$ falls in a
   sensible range (typically 0.3–0.6) for any combination of $V_g$
   and $V_o$.

Everything else — the polarity inversion (when modeled in the
canonical way), the RHP zero in $G_{vd}(s)$, the lightly-damped LC
double pole — carries straight over from the buck-boost. The flyback
is the buck-boost's natural production-grade descendant.

Conventions
-----------

We work with primary-side magnetizing current $i_{L_m}$ and
secondary-side cap voltage $v_o$ (in positive magnitude, with the
polarity inversion being a topology detail). The turns ratio is
$n = N_s/N_p$; typical step-down designs have $n < 1$.

The output transformer is **idealized** as a pure-magnetizing-inductance
+ ideal turns ratio (no leakage, no series resistance). Real flybacks
have a snubber + clamp circuit for leakage energy; that's a separate
follow-up that doesn't affect the small-signal control model.

References
----------

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
  Section 7.5 (buck-boost) and 6.3 (flyback derivation).
- The buck-boost reference notebook (`projects/converters/buck_boost/`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FlybackParams:
    """Flyback-converter design parameters (SI units).

    Defaults: 24 V → 12 V step-down at 1 A through a 1:0.5 transformer
    (n = N_s/N_p = 0.5), 300 µH magnetizing inductance, 100 µF output
    cap, 100 kHz switching. With D = 0.5 (the "even-split" design point)
    the converter has $f_n \\approx 920$ Hz, $f_{z,RHP} \\approx 13$ kHz —
    similar in shape to the buck-boost, but pushed UP in frequency by
    the transformer ratio.
    """

    V_g: float = 24.0       # primary-side input bus voltage (V)
    V_o: float = 12.0       # secondary-side output voltage magnitude (V)
    R: float = 12.0         # load on secondary (Ω); R = V_o²/P_out at 12 W
    L_m: float = 300e-6     # primary-side magnetizing inductance (H)
    C: float = 100e-6       # secondary output capacitor (F)
    f_sw: float = 100e3     # switching frequency (Hz)
    n: float = 0.5          # turns ratio N_s / N_p

    @property
    def D(self) -> float:
        """Steady-state duty cycle.

        From $V_o = n \\cdot V_g \\cdot D/(1-D)$:
            $D = V_o / (V_o + n \\cdot V_g)$.
        """
        return self.V_o / (self.V_o + self.n * self.V_g)

    @property
    def I_Lm(self) -> float:
        """Primary-side magnetizing current (A).

        Power balance: $V_g \\cdot I_{Lm,avg,on} = P_{out}$, with the
        average primary current = $D \\cdot I_{Lm}$ (only flows during
        ON). Working through gives $I_{Lm} = V_o \\cdot n / (R \\cdot (1-D))$.
        """
        return self.V_o * self.n / (self.R * (1.0 - self.D))

    @property
    def omega_n(self) -> float:
        """Natural angular frequency of the LC double pole (rad/s).

        Same form as buck-boost but with the transformer reflection:
            $\\omega_n = (1-D) / (n \\cdot \\sqrt{L_m \\cdot C})$
        """
        return (1.0 - self.D) / (self.n * np.sqrt(self.L_m * self.C))

    @property
    def f_n(self) -> float:
        return self.omega_n / (2.0 * np.pi)

    @property
    def omega_z_rhp(self) -> float:
        """RHP zero (rad/s).

        Buck-boost RHP zero reflected to primary side:
            $\\omega_z = R \\cdot (1-D)^2 / (n^2 \\cdot L_m \\cdot D)$
        """
        return self.R * (1.0 - self.D) ** 2 / (self.n ** 2 * self.L_m * self.D)

    @property
    def f_z_rhp(self) -> float:
        return self.omega_z_rhp / (2.0 * np.pi)

    @property
    def Q(self) -> float:
        """Quality factor of the LC double pole.

        $Q = (1-D) \\cdot R \\cdot \\sqrt{C/L_m} / n$
        """
        return (1.0 - self.D) * self.R * np.sqrt(self.C / self.L_m) / self.n

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.Q)


def flyback_state_space(
    p: FlybackParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small-signal (A, B, C, D) of an ideal flyback in CCM.

    Average ODEs (after $q \\to d$, with $v_o$ on the secondary side and
    $i_{L_m}$ on the primary):

        L_m · di_Lm/dt = d · v_g - (1-d) · v_o / n
        C   · dv_o/dt  = (1-d) · i_Lm / n - v_o / R

    Notice the symmetric appearance of $n$: it scales the secondary
    reflection (primary sees $v_o/n$, secondary sees $i_{Lm}/n$). The
    transformer is just a turns ratio in the average equations.

    Linearize around $(I_{Lm}, V_o, D, V_g)$, drop products:

        L_m · dî_Lm/dt = D·v̂_g + (V_g + V_o/n)·d̂ - (1-D)/n · v̂_o
        C   · dv̂_o/dt  = (1-D)/n · î_Lm - I_Lm/n · d̂ - v̂_o/R

    The state vector is $x = [\\hat i_{Lm}; \\hat v_o]$ (primary current,
    secondary voltage). The non-min-phase $-I_{Lm}/n \\cdot \\hat d$ term
    in the cap equation is what produces the RHP zero — same as boost
    and buck-boost.
    """
    L, C, R, n = p.L_m, p.C, p.R, p.n
    Dprime = 1.0 - p.D
    V_g, V_o = p.V_g, p.V_o
    I_Lm = p.I_Lm

    A = np.array([
        [0.0,             -Dprime / (n * L)],
        [Dprime / (n * C), -1.0 / (R * C)  ],
    ])
    B = np.array([
        [(V_g + V_o / n) / L,  p.D / L],
        [-I_Lm / (n * C),      0.0    ],
    ])
    C_mat = np.array([[0.0, 1.0]])    # output = v̂_o (secondary)
    D_mat = np.zeros((1, 2))
    return A, B, C_mat, D_mat


def control_to_output_tf(p: FlybackParams) -> signal.TransferFunction:
    """Return $G_{vd}(s) = \\hat v_o(s) / \\hat d(s)$ on the secondary side.

    Closed form (Erickson Eq 8.55 reflected through n):

        G_vd(s) = G_d0 · (1 - s/ω_{z,RHP}) /
                  (1 + s/(Q ω_n) + (s/ω_n)²)

    with $G_{d0} = n \\cdot V_g / (1-D)^2$.
    """
    omega_n = p.omega_n
    omega_z = p.omega_z_rhp
    Q = p.Q
    G_dc = p.n * p.V_g / (1.0 - p.D) ** 2
    num = [-G_dc / omega_z, G_dc]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


def line_to_output_tf(p: FlybackParams) -> signal.TransferFunction:
    """Return G_vg(s) = v̂_o / v̂_g on the secondary side.

    Closed form: G_vg(s) = (n · D / (1-D)) / (1 + s/(Q ω_n) + (s/ω_n)²).
    """
    omega_n = p.omega_n
    Q = p.Q
    G_dc = p.n * p.D / (1.0 - p.D)
    num = [G_dc]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


def output_impedance_tf(p: FlybackParams) -> signal.TransferFunction:
    """Return Z_out(s) = v̂_o / î_load on the secondary side.

    Closed form: Z_out(s) = (s·L_m/(n²·(1-D)²)) / (1 + s/(Q ω_n) + (s/ω_n)²)
    """
    L = p.L_m
    n = p.n
    Dprime2 = (1.0 - p.D) ** 2
    omega_n = p.omega_n
    Q = p.Q
    num = [L / (n ** 2 * Dprime2), 0.0]
    den = [1.0 / omega_n ** 2, 1.0 / (Q * omega_n), 1.0]
    return signal.TransferFunction(num, den)


# ---------- helpers (same as other converters) ----------

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


def operating_point_report(p: FlybackParams) -> str:
    P_out = p.V_o * p.V_o / p.R
    lines = [
        "Flyback operating point",
        f"  V_g      = {p.V_g:8.3f} V",
        f"  V_o      = {p.V_o:8.3f} V       (magnitude on secondary)",
        f"  n = N_s/N_p = {p.n:8.4f}",
        f"  D        = {p.D:8.4f}         (= V_o / (V_o + n·V_g))",
        f"  R        = {p.R:8.3f} Ω       (secondary load)",
        f"  I_Lm     = {p.I_Lm:8.3f} A       (primary magnetizing avg)",
        f"  P_out    = {P_out:8.3f} W",
        "",
        "Transformer reflection",
        f"  R reflected to primary = R/n² = {p.R / p.n ** 2:8.3f} Ω",
        f"  C reflected to primary = C·n² = {p.C * p.n ** 2 * 1e6:.2f} µF",
        "",
        "Filter dynamics",
        f"  L_m      = {p.L_m * 1e6:8.3f} µH",
        f"  C        = {p.C * 1e6:8.3f} µF",
        f"  f_n      = {p.f_n:8.1f} Hz      (LC pole — drops as D rises)",
        f"  Q        = {p.Q:8.3f}",
        "",
        "RHP zero",
        f"  f_z_rhp  = {p.f_z_rhp:8.1f} Hz     "
        f"(higher than buck-boost by 1/n² factor)",
        f"  Practical cap: f_c ≤ f_z_rhp / 5 = {p.f_z_rhp / 5:.0f} Hz",
        "",
        "Switching",
        f"  f_sw     = {p.f_sw / 1e3:8.1f} kHz",
        f"  f_sw / f_n       = {p.f_sw / p.f_n:7.1f}×",
        f"  f_sw / f_z_rhp   = {p.f_sw / p.f_z_rhp:7.1f}×",
    ]
    return "\n".join(lines)


__all__ = [
    "FlybackParams",
    "flyback_state_space",
    "control_to_output_tf",
    "line_to_output_tf",
    "output_impedance_tf",
    "step_response",
    "linf_error",
    "relative_rms_error",
    "operating_point_report",
]
