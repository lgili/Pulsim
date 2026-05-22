"""Boost PFC — analytical model (CCM **and** DCM).

Seventh entry in `projects/converters/`. First **AC-input** converter
and first one with **two control objectives** running simultaneously
(regulate $V_o$ AND shape $i_{in}$ to track $|v_{ac}|$).

Why PFC matters
---------------

A rectifier-with-bulk-cap front-end (the cheap default) draws current
only at the peaks of the line — sharp narrow pulses with **PF ≈ 0.6**
and **THD > 100%**. Standards (IEC 61000-3-2, ENERGY STAR) require
PF > 0.9 and bounded harmonics for any equipment above ~75 W.

The boost PFC fixes that by inserting a boost stage **after** the
diode bridge and forcing the inductor current to track the rectified
sinusoid:

```
  v_ac(t) ──┤
            │ diode  +┌────┐   ┌─── D ───┬───── V_o (400 V DC)
  v_ac(t) ──┤ bridge  │  L │   │         │
            │        └────┘   │         C  R_load
            │                 S          │
            └─────────────────┴──────────┴───── gnd
                |v_ac(t)|       boost stage
                = v_g(t)         (CCM or DCM)
```

Two ways to make $i_L$ follow $|v_{ac}|$
----------------------------------------

**DCM (Discontinuous Conduction Mode)**: keep $D$ approximately
constant over a line half-cycle. The inductor current is a triangle
each $T_{sw}$ with peak $i_{pk}(t) = v_g(t) \\cdot D \\cdot T_{sw}/L$.
Averaged over $T_{sw}$, the **input** current is proportional to
$v_g(t)$ (with a small correction near the peak). **Natural PFC**:
no current loop, just a voltage loop. The price is high peak current
stress and EMI.

**CCM (Continuous Conduction Mode)**: explicitly close a current loop
that forces $i_L(t) = i_{ref}(t) = k \\cdot |v_{ac}(t)|$. The voltage
loop (slow) sets the amplitude $k$. **Two-loop architecture** with a
**multiplier** — the textbook PFC. Lower peak currents, lower EMI,
more complex control.

Both share:
- The voltage loop **must be slow** — bandwidth $\\ll 2 f_{line}$ —
  otherwise it tries to "correct" the 100/120 Hz output ripple by
  distorting the input current.
- A bulk output cap large enough to hold $V_o$ within tolerance
  across one line cycle.
- Same boost RHP zero issue (CCM only; DCM PFC has no RHP zero —
  that's another reason DCM is "easier" at low power).

Conventions
-----------

- $V_g(t) = |v_{ac}(t)| = V_{g,pk} |\\sin(\\omega_{line} t)|$ — the
  rectified line voltage seen by the boost stage.
- $V_{g,rms}$ refers to the **AC line** rms (e.g. 230 V), so
  $V_{g,pk} = \\sqrt 2 \\cdot V_{g,rms}$.
- $D$ in DCM is the on-time duty PER SWITCHING CYCLE, held roughly
  constant across a line half-cycle by the slow voltage loop.
- $D(t)$ in CCM varies within a line cycle to track the current
  reference; the small-signal analysis is at a **specific operating
  point** in the line cycle (typically $V_g = V_{g,rms}$ for "average"
  dynamics).

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
**Chapter 18** (low-harmonic rectifiers): Sec. 18.1 (PFC overview),
Sec. 18.2 (DCM averaged model + equivalent resistance), Sec. 18.4
(CCM average current-mode control, multiplier loop). Mohan, Undeland
& Robbins, *Power Electronics*, 3rd ed., Chapter 18 has a parallel
treatment with worked examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal


Mode = Literal["DCM", "CCM"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoostPFCParams:
    """Boost PFC design parameters (SI units).

    Defaults: universal mains (90-265 V_ac rms) → 400 V DC at 100 W,
    50 Hz line, 100 kHz switching. The inductor value $L$ is the main
    knob that selects DCM vs CCM operation — use the `dcm_design()`
    and `ccm_design()` class methods for sensible defaults.
    """

    # AC line
    V_ac_min: float = 90.0       # universal mains low (V rms)
    V_ac_max: float = 265.0      # universal mains high (V rms)
    V_ac_nom: float = 230.0      # nominal operating point for control design (V rms)
    f_line: float = 50.0         # line frequency (Hz)

    # DC bus
    V_o: float = 400.0           # output bus voltage (V)
    P_o: float = 100.0           # nominal output power (W)
    V_o_min: float = 340.0       # hold-up minimum (V) — for cap sizing
    V_o_ripple_pp_max: float = 20.0  # peak-to-peak ripple budget at 2·f_line (V)

    # Power stage
    L: float = 1e-3              # boost inductor (H) — overridden by ccm/dcm design
    C: float = 220e-6            # bulk output cap (F)
    f_sw: float = 100e3          # switching frequency (Hz)
    D_max: float = 0.95          # duty cap (boost switch reliability)

    # Series losses (used in switched simulator only; analytical models assume ideal)
    R_L: float = 0.0             # inductor ESR (Ω)
    R_C: float = 0.0             # cap ESR (Ω) — useful for ripple analysis

    @classmethod
    def dcm_design(cls, **overrides) -> "BoostPFCParams":
        """Defaults tuned for **DCM operation at full load over universal mains**.

        Small inductor (~100 µH) forces DCM at all line conditions for
        the 100 W operating point. Bulk cap unchanged (driven by ripple
        + hold-up).
        """
        kwargs = dict(L=100e-6, C=220e-6)
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def ccm_design(cls, **overrides) -> "BoostPFCParams":
        """Defaults tuned for **CCM operation at full load over universal mains**.

        Larger inductor (~1 mH) forces CCM at all line conditions
        for the 100 W operating point.
        """
        kwargs = dict(L=1e-3, C=220e-6)
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---- Basic derived quantities ----

    @property
    def V_g_pk_min(self) -> float:
        return np.sqrt(2.0) * self.V_ac_min

    @property
    def V_g_pk_max(self) -> float:
        return np.sqrt(2.0) * self.V_ac_max

    @property
    def V_g_pk_nom(self) -> float:
        return np.sqrt(2.0) * self.V_ac_nom

    @property
    def omega_line(self) -> float:
        return 2.0 * np.pi * self.f_line

    @property
    def T_line(self) -> float:
        return 1.0 / self.f_line

    @property
    def T_line_half(self) -> float:
        return 0.5 / self.f_line

    @property
    def T_sw(self) -> float:
        return 1.0 / self.f_sw

    @property
    def R_load(self) -> float:
        """Equivalent load resistance at the operating point."""
        return self.V_o * self.V_o / self.P_o

    @property
    def I_o_avg(self) -> float:
        return self.P_o / self.V_o

    def v_g(self, t: np.ndarray | float) -> np.ndarray | float:
        """Instantaneous rectified line voltage $|v_{ac}(t)|$ at NOMINAL line."""
        return self.V_g_pk_nom * np.abs(np.sin(self.omega_line * np.asarray(t)))

    # ---- Output cap sizing constraints ----

    @property
    def C_for_holdup_min(self) -> float:
        """Minimum $C$ for one line cycle of hold-up at full load (F)."""
        # P · T_line = (1/2) · C · (V_o² - V_o_min²)
        return 2.0 * self.P_o * self.T_line / (self.V_o ** 2 - self.V_o_min ** 2)

    @property
    def C_for_ripple_min(self) -> float:
        """Minimum $C$ to keep 2·f_line ripple within budget (F).

        For a PFC, $\\hat V_{o,pp} = I_o / (\\pi f_{line} C)$.
        """
        return self.I_o_avg / (np.pi * self.f_line * self.V_o_ripple_pp_max)

    @property
    def V_o_ripple_pp_predicted(self) -> float:
        """Predicted output 2·f_line ripple (peak-to-peak, V) at full load."""
        return self.I_o_avg / (np.pi * self.f_line * self.C)


# ---------------------------------------------------------------------------
# DCM analysis  --  Erickson Sec. 18.2
# ---------------------------------------------------------------------------


def dcm_critical_inductance(p: BoostPFCParams, *, V_ac: float | None = None) -> float:
    """Maximum $L$ that still guarantees DCM at full load.

    Operation stays in DCM only if $L < L_{crit}$. Beyond that the
    converter enters CCM near the peak of the line cycle (mixed mode).

    From Erickson eq. 18.13:
        $L_{crit} = \\frac{V_o R_e (D')^2_{min}}{2 f_{sw}}$
    where $R_e = 2 L f_{sw} / D^2$ is the equivalent input resistance.

    Closed form: $L_{crit} = (V_o - V_{g,pk}) V_{g,rms}^2 / (2 f_{sw} P_o V_o)$
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_min
    V_g_pk = np.sqrt(2.0) * V_ac
    return (p.V_o - V_g_pk) * V_ac ** 2 / (2.0 * p.f_sw * p.P_o * p.V_o)


def dcm_equivalent_resistance(p: BoostPFCParams, D: float) -> float:
    """DCM input equivalent resistance $R_e = 2 L f_{sw} / D^2$ (Ω).

    The DCM boost looks like a **resistor** $R_e$ to the rectified line
    if $D$ is held constant — that's the natural-PFC mechanism. Output
    averages to a power source $V_o^2/R_{load}$.
    """
    return 2.0 * p.L * p.f_sw / (D ** 2)


def dcm_steady_state_duty(p: BoostPFCParams, *, V_ac: float | None = None) -> float:
    """Steady-state $D$ in DCM PFC at a given line voltage (full load).

    **Linearized formula** (Erickson eq. 18.18, low-distortion limit):

        $D = \\sqrt{2 L f_{sw} P_o} / V_{ac,rms}$

    From $V_{ac}^2 / R_e = P_o$ and $R_e = 2 L f_{sw}/D^2$ — treats
    the converter as a pure resistor. **Valid only when $V_{g,pk} \\ll V_o$**
    (the cusp-distortion factor $F(m_{pk}) \\to 1$). For universal mains
    the actual operating duty is smaller (less power needed to overcome
    the cusp boost); use `dcm_steady_state_duty_exact()` for the exact value.
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_nom
    return float(np.sqrt(2.0 * p.L * p.f_sw * p.P_o) / V_ac)


def dcm_power_correction_factor(m_pk: float) -> float:
    """DCM PFC cusp-distortion correction factor.

    Defined as the ratio of *actual* DCM input power to the linearized
    "resistive-emulator" estimate:

        $F(m_{pk}) = \\dfrac{2}{\\pi} \\int_0^\\pi
            \\dfrac{\\sin^2 \\theta}{1 - m_{pk}\\sin\\theta} d\\theta
            \\;/\\; \\tfrac{1}{2}$

    where $m_{pk} = V_{g,pk}/V_o$. $F \\to 1$ as $m_{pk} \\to 0$ and
    diverges as $m_{pk} \\to 1$ (the converter can't boost from a
    voltage equal to $V_o$).

    For universal mains: F ≈ 1.4 at 90 V, F ≈ 7.5 at 230 V — that's
    why a fixed-D DCM PFC over the universal range is non-trivial.
    """
    from scipy.integrate import quad
    if m_pk <= 0.0:
        return 1.0
    if m_pk >= 0.9999:
        return float("inf")
    val, _ = quad(lambda th: np.sin(th) ** 2 / (1.0 - m_pk * np.sin(th)),
                  0.0, np.pi)
    # Divide by π·<sin²>·F_resistive = π · 1/2
    return float(val / (np.pi * 0.5))


def dcm_steady_state_duty_exact(
    p: BoostPFCParams, *, V_ac: float | None = None, V_o: float | None = None
) -> float:
    """Exact steady-state $D$ at a given $V_{ac}$ and $V_o$ (default $V_o = p.V_o$).

    Solves the implicit equation:

        $P_o = (V_{ac}^2 D^2 / (2 L f_{sw})) \\cdot F(V_{g,pk}/V_o)$

    where $F$ is the cusp-distortion correction. The linearized
    formula `dcm_steady_state_duty` ignores $F$.
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_nom
    V_o = V_o if V_o is not None else p.V_o
    m_pk = np.sqrt(2.0) * V_ac / V_o
    F = dcm_power_correction_factor(m_pk)
    return float(np.sqrt(2.0 * p.L * p.f_sw * p.P_o / F) / V_ac)


def dcm_input_current_instantaneous(
    p: BoostPFCParams, t: np.ndarray, D: float, *, V_ac: float | None = None
) -> np.ndarray:
    """DCM input current averaged over one $T_{sw}$, as a function of time
    over a line cycle.

    From Erickson eq. 18.16:
        $\\langle i_g(t) \\rangle_{T_{sw}} = \\frac{v_g(t)}{R_e}
            \\cdot \\frac{1}{1 - v_g(t)/V_o}$
    The pre-factor is the resistive part (PF = 1 if alone); the
    denominator is the cusp distortion that grows as $v_g \\to V_o$.
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_nom
    V_g_pk = np.sqrt(2.0) * V_ac
    v_g = V_g_pk * np.abs(np.sin(p.omega_line * t))
    R_e = dcm_equivalent_resistance(p, D)
    safe_v_g = np.minimum(v_g, 0.999 * p.V_o)  # avoid divide-by-zero at cusps
    return (safe_v_g / R_e) / (1.0 - safe_v_g / p.V_o)


def dcm_voltage_loop_plant(p: BoostPFCParams) -> signal.TransferFunction:
    """Control-to-output TF $G_{vd}(s) = \\hat v_o / \\hat d$ for DCM PFC.

    Derived from the line-cycle-averaged model (LCAM): treat the
    output cap voltage and the duty as slow variables, average power
    flow over one line cycle.

    Power input: $P_{in} = V_{ac,rms}^2 / R_e = V_{ac,rms}^2 D^2 / (2 L f_{sw})$
    Power output: $P_o = V_o^2 / R_{load}$
    Cap balance: $C \\cdot dV_o/dt \\cdot V_o = P_{in} - V_o^2/R_{load}$

    Linearize around $(V_o, D)$:
        $C V_o \\, d\\hat v_o/dt = (V_{ac,rms}^2 D / (L f_{sw})) \\hat d
                                  - (2 V_o / R_{load}) \\hat v_o$

    Rearrange:
        $\\hat v_o / \\hat d = K_{dc} / (1 + s \\tau)$
        $K_{dc} = (V_{ac,rms}^2 D / (L f_{sw})) \\cdot R_{load} / (2 V_o)$
        $\\tau = R_{load} C / 2$

    **Single first-order pole at $1/(R_{load} C/2)$ — much easier to
    compensate than CCM.** This is the headline result of DCM PFC
    control.

    Evaluated at $V_{ac} = V_{ac,nom}$.
    """
    D = dcm_steady_state_duty(p, V_ac=p.V_ac_nom)
    V_ac_rms = p.V_ac_nom
    R = p.R_load

    K_dc = (V_ac_rms ** 2 * D / (p.L * p.f_sw)) * R / (2.0 * p.V_o)
    tau = R * p.C / 2.0
    num = [K_dc]
    den = [tau, 1.0]
    return signal.TransferFunction(num, den)


# ---------------------------------------------------------------------------
# CCM analysis  --  Erickson Sec. 18.4
# ---------------------------------------------------------------------------


def ccm_current_loop_plant(p: BoostPFCParams) -> signal.TransferFunction:
    """Control-to-inductor-current TF $G_{id}(s) = \\hat i_L / \\hat d$
    for CCM, at a specific point in the line cycle.

    Within the current-loop bandwidth (tens of kHz) we can ignore the
    cap dynamics — the output is "stiff" on that timescale. The
    inductor sees:
        $L \\cdot di_L/dt = v_g - (1-d) V_o$
    Linearize: $L \\cdot d\\hat i_L/dt = V_o \\hat d$ →
        $G_{id}(s) = V_o / (s L)$
    A clean integrator. The current loop is **easy**.

    Evaluated at the line peak (worst case for slew). Strictly the
    linearization is valid for small perturbations around any
    operating point in the line cycle.
    """
    V_o = p.V_o
    num = [V_o]
    den = [p.L, 0.0]
    return signal.TransferFunction(num, den)


def ccm_voltage_loop_plant(p: BoostPFCParams) -> signal.TransferFunction:
    """Control-to-output TF $G_{vc}(s) = \\hat v_o / \\hat i_{ref,amp}$
    for the **outer** voltage loop, with the inner current loop assumed
    ideal (closed and fast).

    Outer loop variable: $i_{ref,amp}$ — the amplitude of the current
    reference that the multiplier scales by $|v_{ac}(t)|$ to form
    $i_{ref}(t)$.

    Power balance (averaged over one line cycle):
        $\\bar P_{in} = (1/2) V_{ac,pk} \\cdot i_{ref,amp}$
                     = $V_{ac,rms} \\cdot I_{ref,amp,rms}$
                     = $V_{ac,rms} \\cdot i_{ref,amp}/\\sqrt 2$

    Cap balance: $C \\cdot d(V_o^2/2)/dt = \\bar P_{in} - V_o^2/R_{load}$.

    Linearize around the operating point:
        $C V_o \\cdot d\\hat v_o/dt = (V_{ac,rms}/\\sqrt 2) \\hat i_{ref,amp}
                                       - (2 V_o / R_{load}) \\hat v_o$

    Result:
        $G_{vc}(s) = \\frac{V_{ac,rms}/\\sqrt 2}{2 V_o/R_{load}} \\cdot
                     \\frac{1}{1 + s R_{load} C / 2}$

    **Same single-pole shape as DCM** — at this level of abstraction
    DCM and CCM PFC look identical at the voltage loop. The
    difference is hidden inside the inner current loop (DCM doesn't
    have one).
    """
    V_ac_rms = p.V_ac_nom
    R = p.R_load
    K_dc = (V_ac_rms / np.sqrt(2.0)) * R / (2.0 * p.V_o)
    tau = R * p.C / 2.0
    return signal.TransferFunction([K_dc], [tau, 1.0])


def ccm_voltage_loop_disturbance_120hz(p: BoostPFCParams) -> float:
    """Magnitude of the $2 f_{line}$ output ripple seen by the voltage loop (V).

    This is the fundamental disturbance that constrains the outer-loop
    bandwidth: pick $f_{c,v} < 2 f_{line} / 10$ to **attenuate** this
    ripple. If you make the voltage loop faster, it tries to "correct"
    the ripple by chopping the current reference at $2 f_{line}$,
    which distorts the input current.
    """
    return p.I_o_avg / (np.pi * p.f_line * p.C)


# ---------------------------------------------------------------------------
# Power-quality metrics
# ---------------------------------------------------------------------------


def line_band_filter(
    x: np.ndarray, f_s: float, f_line: float, n_harmonics: int = 20,
) -> np.ndarray:
    """Brick-wall low-pass at $n_{harmonics} \\cdot f_{line}$ via FFT.

    Removes switching-frequency content so the result reflects only
    the "line-band" current that a real PQ meter or downstream EMI
    filter would see. Applied before PF / THD computation to mirror
    standard practice.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/f_s)
    f_cut = (n_harmonics + 0.5) * f_line
    X[freqs > f_cut] = 0.0
    return np.fft.irfft(X, n=N)


def power_factor(
    v_ac: np.ndarray, i_in: np.ndarray, *,
    f_s: float | None = None, f_line: float | None = None,
    n_harmonics: int = 20,
) -> float:
    """Compute PF = $\\bar P / (V_{rms} I_{rms})$ from time-series.

    $\\bar P = \\langle v_{ac}(t) \\cdot i_{in}(t) \\rangle$ — true average
    power (after the diode bridge it's the line-current pre-bridge,
    so v_ac must be SIGNED while i_in is the line-current).

    For a boost PFC, the line current is $i_{in}(t) = i_g(t) \\cdot \\text{sgn}(v_{ac}(t))$.

    If both `f_s` and `f_line` are provided, the current is **band-
    limited** to the first `n_harmonics` of the line frequency before
    computing the metric — the standard PQ-meter convention that
    rejects switching-frequency content above the line band.
    """
    v_ac = np.asarray(v_ac, dtype=float)
    i_in = np.asarray(i_in, dtype=float)
    if f_s is not None and f_line is not None:
        i_in = line_band_filter(i_in, f_s, f_line, n_harmonics)
    p_avg = float(np.mean(v_ac * i_in))
    v_rms = float(np.sqrt(np.mean(v_ac ** 2)))
    i_rms = float(np.sqrt(np.mean(i_in ** 2)))
    if v_rms < 1e-12 or i_rms < 1e-12:
        return 0.0
    return p_avg / (v_rms * i_rms)


def thd_current(i_in: np.ndarray, f_s: float, f_line: float,
                n_harmonics: int = 20) -> float:
    """Total Harmonic Distortion of the line current, as a fraction.

    THD = $\\sqrt{\\sum_{n \\ge 2} I_n^2} / I_1$ where $I_n$ is the
    rms magnitude of the $n$-th line harmonic.

    Requires an integer number of line cycles in the input for clean
    binning. Uses an FFT and bins by harmonic index.
    """
    i_in = np.asarray(i_in, dtype=float)
    N = len(i_in)
    # FFT magnitudes (one-sided)
    X = np.fft.rfft(i_in)
    mag = np.abs(X) * 2.0 / N
    freqs = np.fft.rfftfreq(N, d=1.0/f_s)
    # Find harmonic bins
    harmonic_freqs = f_line * np.arange(1, n_harmonics + 1)
    harmonic_mags = np.array([
        mag[np.argmin(np.abs(freqs - hf))] for hf in harmonic_freqs
    ])
    if harmonic_mags[0] < 1e-12:
        return 0.0
    rest_sq = float(np.sum(harmonic_mags[1:] ** 2))
    return float(np.sqrt(rest_sq) / harmonic_mags[0])


# ---------------------------------------------------------------------------
# Switched simulation engines  --  pure Python forward-Euler
# ---------------------------------------------------------------------------


def simulate_open_loop_dcm(
    p: BoostPFCParams,
    *,
    D: float | None = None,
    V_ac: float | None = None,
    n_line_cycles: int = 4,
    samples_per_period: int = 50,
) -> dict[str, np.ndarray]:
    """Open-loop DCM PFC simulation — fixed $D$ over $n$ line cycles.

    Demonstrates the **natural PFC** property: a constant duty produces
    an input current that's already proportional to $|v_{ac}(t)|$.
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_nom
    if D is None:
        # Use the exact duty so v_o settles at the target — the linearized
        # formula's ~30-50% error would let v_o drift far from V_o nominal.
        D = dcm_steady_state_duty_exact(p, V_ac=V_ac)
    V_g_pk = np.sqrt(2.0) * V_ac
    T_sw = p.T_sw
    dt = T_sw / samples_per_period
    T_total = n_line_cycles / p.f_line
    n_steps = int(T_total / dt) + 1

    v_o = p.V_o
    i_L = 0.0

    record_every = max(1, samples_per_period // 10)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_ac_hist = np.zeros(n_rec)
    v_g_hist = np.zeros(n_rec)
    i_L_hist = np.zeros(n_rec)
    i_in_hist = np.zeros(n_rec)
    v_o_hist = np.zeros(n_rec)
    rec_idx = 0

    for i in range(n_steps):
        t = i * dt
        v_ac = V_g_pk * np.sin(p.omega_line * t)
        v_g = abs(v_ac)
        cycle_pos = (i % samples_per_period) / samples_per_period
        switch_on = cycle_pos < D

        # Boost switched ODE on the rectified-bus side.
        # Inductor: connected between v_g and either gnd (S on) or v_o (D on).
        # Diode is ideal; if i_L tries to go negative we clamp at 0 (DCM).
        if switch_on:
            v_L = v_g
        else:
            # Diode on if i_L > 0; otherwise both off (DCM third phase)
            if i_L > 0:
                v_L = v_g - v_o
            else:
                v_L = 0.0

        # Trapezoidal step: average i_L over the step so the diode-shutoff
        # transition doesn't overcount charge into the cap (forward-Euler
        # bug that causes V_o to drift upward).
        i_L_new = i_L + (v_L / p.L) * dt
        i_L_new_clamped = max(i_L_new, 0.0)
        i_L_avg = 0.5 * (i_L + i_L_new_clamped)
        # Resistive load: i_load = v_o / R_load
        i_C = -v_o / p.R_load
        if not switch_on and i_L > 0:
            i_C += i_L_avg  # diode charge to cap (trapezoid-averaged)
        v_o = v_o + (i_C / p.C) * dt
        i_L = i_L_new_clamped

        # Line-side input current = i_g · sign(v_ac) (with sgn applied
        # after the bridge — i_g is post-bridge so always >= 0)
        sign_v_ac = 1.0 if v_ac >= 0 else -1.0

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_ac_hist[rec_idx] = v_ac
            v_g_hist[rec_idx] = v_g
            i_L_hist[rec_idx] = i_L
            i_in_hist[rec_idx] = i_L * sign_v_ac
            v_o_hist[rec_idx] = v_o
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx], "v_ac": v_ac_hist[:rec_idx],
        "v_g": v_g_hist[:rec_idx], "i_L": i_L_hist[:rec_idx],
        "i_in": i_in_hist[:rec_idx], "v_o": v_o_hist[:rec_idx],
    }


def simulate_closed_loop_dcm(
    p: BoostPFCParams,
    b_v: np.ndarray, a_v: np.ndarray,
    *,
    V_ac: float | None = None,
    n_line_cycles: int = 6,
    samples_per_period: int = 50,
    v_ref: float = 400.0,
    V_ramp: float = 5.0,
    warm_start: bool = True,
) -> dict[str, np.ndarray]:
    """Closed-loop DCM PFC simulation with a digital voltage loop only.

    The voltage compensator runs once per switching period
    (sample-and-hold) and outputs a duty in $[0, D_{max}]$.
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_nom
    V_g_pk = np.sqrt(2.0) * V_ac
    T_sw = p.T_sw
    dt = T_sw / samples_per_period
    T_total = n_line_cycles / p.f_line
    n_steps = int(T_total / dt) + 1

    n_state = len(a_v) - 1
    state = np.zeros(n_state)

    if warm_start:
        # Exact duty so the loop opens at the right operating point
        D_init = dcm_steady_state_duty_exact(p, V_ac=V_ac, V_o=v_ref)
        duty = D_init
        v_o = v_ref
        i_L = 0.0
        v_c_ss = duty * V_ramp
        for k in range(n_state):
            state[k] = -np.sum(a_v[k+1:]) * v_c_ss
    else:
        duty = 0.1
        v_o = 0.0
        i_L = 0.0

    record_every = max(1, samples_per_period // 10)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_ac_hist = np.zeros(n_rec)
    v_g_hist = np.zeros(n_rec)
    i_L_hist = np.zeros(n_rec)
    i_in_hist = np.zeros(n_rec)
    v_o_hist = np.zeros(n_rec)
    duty_hist = np.zeros(n_rec)
    rec_idx = 0

    for i in range(n_steps):
        t = i * dt
        v_ac = V_g_pk * np.sin(p.omega_line * t)
        v_g = abs(v_ac)
        cycle_pos_int = i % samples_per_period

        if cycle_pos_int == 0:
            err = v_ref - v_o
            v_c = b_v[0] * err + state[0]
            new_state = np.zeros_like(state)
            for j in range(n_state - 1):
                new_state[j] = b_v[j+1] * err - a_v[j+1] * v_c + state[j+1]
            new_state[n_state - 1] = b_v[n_state] * err - a_v[n_state] * v_c
            state = new_state
            duty = float(np.clip(v_c / V_ramp, 0.0, p.D_max))

        switch_on = (cycle_pos_int / samples_per_period) < duty
        if switch_on:
            v_L = v_g
        else:
            v_L = v_g - v_o if i_L > 0 else 0.0

        # Trapezoidal cap update (see open-loop comment)
        i_L_new = i_L + (v_L / p.L) * dt
        i_L_new_clamped = max(i_L_new, 0.0)
        i_L_avg = 0.5 * (i_L + i_L_new_clamped)
        i_C = -v_o / p.R_load
        if not switch_on and i_L > 0:
            i_C += i_L_avg
        v_o = v_o + (i_C / p.C) * dt
        i_L = i_L_new_clamped

        sign_v_ac = 1.0 if v_ac >= 0 else -1.0

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_ac_hist[rec_idx] = v_ac
            v_g_hist[rec_idx] = v_g
            i_L_hist[rec_idx] = i_L
            i_in_hist[rec_idx] = i_L * sign_v_ac
            v_o_hist[rec_idx] = v_o
            duty_hist[rec_idx] = duty
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx], "v_ac": v_ac_hist[:rec_idx],
        "v_g": v_g_hist[:rec_idx], "i_L": i_L_hist[:rec_idx],
        "i_in": i_in_hist[:rec_idx], "v_o": v_o_hist[:rec_idx],
        "duty": duty_hist[:rec_idx],
    }


def simulate_closed_loop_ccm(
    p: BoostPFCParams,
    b_i: np.ndarray, a_i: np.ndarray,
    b_v: np.ndarray, a_v: np.ndarray,
    *,
    V_ac: float | None = None,
    n_line_cycles: int = 6,
    samples_per_period: int = 50,
    v_ref: float = 400.0,
    V_ramp: float = 5.0,
    warm_start: bool = True,
    K_mult: float = 0.01,
) -> dict[str, np.ndarray]:
    """Closed-loop CCM PFC simulation with **two** digital loops.

    Architecture:
        outer voltage loop (slow):  err_v = v_ref - v_o → i_ref_amp (DC level)
        multiplier:                  i_ref(t) = K_mult · i_ref_amp · |v_ac(t)|
        inner current loop (fast):   err_i = i_ref(t) - i_L → duty

    Both loops run once per switching period; that's enough because
    the line is much slower than f_sw.
    """
    V_ac = V_ac if V_ac is not None else p.V_ac_nom
    V_g_pk = np.sqrt(2.0) * V_ac
    T_sw = p.T_sw
    dt = T_sw / samples_per_period
    T_total = n_line_cycles / p.f_line
    n_steps = int(T_total / dt) + 1

    n_state_i = len(a_i) - 1
    n_state_v = len(a_v) - 1
    state_i = np.zeros(n_state_i)
    state_v = np.zeros(n_state_v)

    if warm_start:
        D_nom = 1.0 - V_ac * np.sqrt(2.0) / np.pi / p.V_o  # rough average duty
        # Steady-state i_ref_amp such that <P_in> = P_o
        i_ref_amp = 2.0 * p.P_o / (V_g_pk * K_mult)
        v_o = v_ref
        i_L = 0.0
        duty = D_nom
        # Warm-start state arrays so v_c outputs settle at i_ref_amp / duty
        v_c_v = i_ref_amp  # outer loop output
        v_c_i = duty * V_ramp  # inner loop output
        for k in range(n_state_v):
            state_v[k] = -np.sum(a_v[k+1:]) * v_c_v
        for k in range(n_state_i):
            state_i[k] = -np.sum(a_i[k+1:]) * v_c_i
    else:
        i_ref_amp = 0.0
        v_o = 0.0
        i_L = 0.0
        duty = 0.5

    record_every = max(1, samples_per_period // 10)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_ac_hist = np.zeros(n_rec)
    v_g_hist = np.zeros(n_rec)
    i_L_hist = np.zeros(n_rec)
    i_in_hist = np.zeros(n_rec)
    i_ref_hist = np.zeros(n_rec)
    v_o_hist = np.zeros(n_rec)
    duty_hist = np.zeros(n_rec)
    rec_idx = 0

    for i in range(n_steps):
        t = i * dt
        v_ac = V_g_pk * np.sin(p.omega_line * t)
        v_g = abs(v_ac)
        cycle_pos_int = i % samples_per_period

        if cycle_pos_int == 0:
            # Outer voltage loop
            err_v = v_ref - v_o
            v_c_v = b_v[0] * err_v + state_v[0]
            new_state_v = np.zeros_like(state_v)
            for j in range(n_state_v - 1):
                new_state_v[j] = b_v[j+1] * err_v - a_v[j+1] * v_c_v + state_v[j+1]
            new_state_v[n_state_v - 1] = b_v[n_state_v] * err_v - a_v[n_state_v] * v_c_v
            state_v = new_state_v
            i_ref_amp = max(v_c_v, 0.0)

            # Multiplier
            i_ref = K_mult * i_ref_amp * v_g

            # Inner current loop
            err_i = i_ref - i_L
            v_c_i = b_i[0] * err_i + state_i[0]
            new_state_i = np.zeros_like(state_i)
            for j in range(n_state_i - 1):
                new_state_i[j] = b_i[j+1] * err_i - a_i[j+1] * v_c_i + state_i[j+1]
            new_state_i[n_state_i - 1] = b_i[n_state_i] * err_i - a_i[n_state_i] * v_c_i
            state_i = new_state_i
            duty = float(np.clip(v_c_i / V_ramp, 0.0, p.D_max))
        else:
            i_ref = K_mult * i_ref_amp * v_g

        switch_on = (cycle_pos_int / samples_per_period) < duty
        if switch_on:
            v_L = v_g
        else:
            v_L = v_g - v_o if i_L > 0 else 0.0

        # Trapezoidal cap update
        i_L_new = i_L + (v_L / p.L) * dt
        i_L_new_clamped = max(i_L_new, 0.0)
        i_L_avg = 0.5 * (i_L + i_L_new_clamped)
        i_C = -v_o / p.R_load
        if not switch_on and i_L > 0:
            i_C += i_L_avg
        v_o = v_o + (i_C / p.C) * dt
        i_L = i_L_new_clamped

        sign_v_ac = 1.0 if v_ac >= 0 else -1.0

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_ac_hist[rec_idx] = v_ac
            v_g_hist[rec_idx] = v_g
            i_L_hist[rec_idx] = i_L
            i_in_hist[rec_idx] = i_L * sign_v_ac
            i_ref_hist[rec_idx] = i_ref
            v_o_hist[rec_idx] = v_o
            duty_hist[rec_idx] = duty
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx], "v_ac": v_ac_hist[:rec_idx],
        "v_g": v_g_hist[:rec_idx], "i_L": i_L_hist[:rec_idx],
        "i_in": i_in_hist[:rec_idx], "i_ref": i_ref_hist[:rec_idx],
        "v_o": v_o_hist[:rec_idx], "duty": duty_hist[:rec_idx],
    }


# ---------------------------------------------------------------------------
# Operating-point report
# ---------------------------------------------------------------------------


def operating_point_report(p: BoostPFCParams, mode: Mode = "CCM") -> str:
    lines = [
        f"Boost PFC operating point ({mode})",
        f"  AC line       = {p.V_ac_min:5.1f} - {p.V_ac_max:5.1f} V rms "
        f"(nom {p.V_ac_nom:5.1f})",
        f"  V_g peak nom  = {p.V_g_pk_nom:8.2f} V",
        f"  V_o (DC bus)  = {p.V_o:8.2f} V",
        f"  P_o           = {p.P_o:8.2f} W",
        f"  R_load (eq.)  = {p.R_load:8.1f} Ω",
        f"  I_o (avg)     = {p.I_o_avg:8.4f} A",
        "",
        f"  f_line        = {p.f_line:8.1f} Hz",
        f"  f_sw          = {p.f_sw / 1e3:8.1f} kHz",
        f"  L             = {p.L * 1e6:8.1f} µH",
        f"  C             = {p.C * 1e6:8.1f} µF",
        "",
        "Bulk-cap sizing",
        f"  C_for_holdup_min  = {p.C_for_holdup_min * 1e6:8.1f} µF   "
        f"(one-cycle hold-up to {p.V_o_min:.0f} V)",
        f"  C_for_ripple_min  = {p.C_for_ripple_min * 1e6:8.1f} µF   "
        f"(2·f_line ripple ≤ {p.V_o_ripple_pp_max:.1f} V pk-pk)",
        f"  C present         = {p.C * 1e6:8.1f} µF",
        f"  V_o ripple predicted: {p.V_o_ripple_pp_predicted:.2f} V pk-pk",
    ]
    if mode == "DCM":
        L_crit_lo = dcm_critical_inductance(p, V_ac=p.V_ac_min)
        L_crit_hi = dcm_critical_inductance(p, V_ac=p.V_ac_max)
        D_lo = dcm_steady_state_duty(p, V_ac=p.V_ac_min)
        D_hi = dcm_steady_state_duty(p, V_ac=p.V_ac_max)
        D_nom = dcm_steady_state_duty(p, V_ac=p.V_ac_nom)
        lines += [
            "",
            "DCM design check",
            f"  L_crit @ V_ac_min   = {L_crit_lo * 1e6:8.1f} µH "
            f"(L must be < this for DCM at low line)",
            f"  L_crit @ V_ac_max   = {L_crit_hi * 1e6:8.1f} µH "
            f"(easier — DCM at high line is automatic)",
            f"  L present           = {p.L * 1e6:8.1f} µH "
            f"({'OK ✅' if p.L < L_crit_lo else '⚠️  FORCES CCM at low line'})",
            f"  D @ V_ac_min        = {D_lo:.4f}",
            f"  D @ V_ac_nom        = {D_nom:.4f}",
            f"  D @ V_ac_max        = {D_hi:.4f}",
            f"  Re @ V_ac_nom       = {dcm_equivalent_resistance(p, D_nom):8.1f} Ω "
            f"(= V_ac²/P_o = {p.V_ac_nom**2/p.P_o:.1f})",
        ]
    else:  # CCM
        I_L_avg_peak = p.P_o * np.sqrt(2.0) / p.V_ac_min  # current peak at low line
        delta_i_pp = p.V_g_pk_min * 0.5 / (p.L * p.f_sw)  # crude max ripple estimate
        lines += [
            "",
            "CCM design check",
            f"  Peak line current   = {I_L_avg_peak:6.2f} A  (at V_ac_min, "
            f"full load)",
            f"  Inductor ripple est = {delta_i_pp * 1e3:6.0f} mA  "
            f"(crude — ignore for design intuition)",
            f"  D @ V_ac_pk (nom)   = {1.0 - p.V_g_pk_nom/p.V_o:.4f}",
            f"  D @ valley (cusp)   ≈ {1.0 - 0.0/p.V_o:.4f}  (D → 1 near cusps)",
        ]
    lines += [
        "",
        "Control headline",
        f"  Voltage-loop BW max ≈ 2·f_line/10 = {2*p.f_line/10:.1f} Hz",
        f"  2·f_line ripple at V_o = {p.V_o_ripple_pp_predicted:.2f} V pk-pk",
    ]
    return "\n".join(lines)


__all__ = [
    "BoostPFCParams",
    "Mode",
    "dcm_critical_inductance",
    "dcm_equivalent_resistance",
    "dcm_steady_state_duty",
    "dcm_steady_state_duty_exact",
    "dcm_power_correction_factor",
    "dcm_input_current_instantaneous",
    "dcm_voltage_loop_plant",
    "ccm_current_loop_plant",
    "ccm_voltage_loop_plant",
    "ccm_voltage_loop_disturbance_120hz",
    "line_band_filter",
    "power_factor",
    "thd_current",
    "simulate_open_loop_dcm",
    "simulate_closed_loop_dcm",
    "simulate_closed_loop_ccm",
    "operating_point_report",
]
