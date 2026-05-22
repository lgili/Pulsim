"""Three-phase VSI (Voltage Source Inverter) — analytical model.

First entry under `projects/inverters/` — the canonical **DC → AC**
converter. A 6-switch H-bridge per leg (3 legs total) creates a
three-phase AC output from a DC bus.

What's new vs the 7 DC-DC / PFC converters
-------------------------------------------

1. **Direction is reversed**: power flows DC → AC (or bidirectionally
   for grid-tie). The bulk cap on the DC bus is now the *source*, not
   the *sink*.

2. **Three phases**: the output is a balanced three-phase set of
   sinusoids 120° apart, $v_{a,b,c}(t)$. Six switches in three legs,
   each leg with a complementary pair (top + bottom). One switching
   period produces a snapshot of a rotating space vector.

3. **Two reference frames**:
   - **abc frame**: physical phase quantities ($v_a$, $v_b$, $v_c$).
   - **αβ frame** (Clarke): rotates with the line, sees the three-
     phase as a 2D rotating vector. Three sinusoids → one vector at
     constant magnitude rotating at $\\omega$.
   - **dq frame** (Park): rotates *with* the vector → AC quantities
     become DC. Compensator design happens here.

4. **Space Vector PWM (SVPWM)** instead of three independent SPWMs.
   SVPWM produces ~15% more output for the same DC bus and lower
   harmonic content. Implementation: **min-max injection**
   (mathematically equivalent to vector-based SVPWM but simpler).

5. **Two operating modes** in this same module:
   - **Stand-alone**: VSI feeds an isolated RL load through an LC
     filter. Voltage-control loop in dq frame regulates output
     magnitude and frequency.
   - **Grid-tie**: VSI is connected to a stiff AC grid. A **PLL**
     (Phase-Locked Loop) extracts the grid angle, and a current-
     control loop in dq injects active power P and reactive power Q.

Conventions
-----------

- Positive sequence: $v_a$ leads $v_b$ leads $v_c$.
- Park angle $\\theta = \\omega t$ aligns the d-axis with phase-a at
  $t = 0$. With this convention a balanced three-phase positive
  sequence maps to constant $V_d > 0$, $V_q = 0$ in steady state.
- Clarke transform is **amplitude-invariant** (factor 2/3): the
  αβ vector magnitude equals the *peak* of the phase voltage.

References
----------

Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
Chapter 15 (multi-pulse rectifiers, includes 3-phase basics). Mohan,
Undeland & Robbins, *Power Electronics*, 3rd ed., Chapter 8 (VSI +
SVPWM). Yazdani & Iravani, *Voltage-Sourced Converters in Power
Systems*, Wiley 2010 — the comprehensive grid-tie reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal


Mode = Literal["standalone", "gridtie"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VSI3PhaseParams:
    """Three-phase VSI design parameters (SI units).

    Defaults: 400 V DC bus → 230 V line-to-line rms at 60 Hz, 500 W
    output through an LC filter (1 mH + 10 µF). Switching at 20 kHz
    (audible threshold; LC filter corner at ~1.6 kHz cleanly
    separates the modulating signal from the carrier).

    Brazil-style 60 Hz operating point. With $V_{dc} = 400$ V the
    SVPWM modulation index for 230 V_rms line-to-line is $m_a
    \\approx 0.815$ — comfortable margin below the SVPWM limit of
    $m_a = 1.0$.
    """

    # DC bus
    V_dc: float = 400.0          # DC bus voltage (V)

    # AC output spec
    V_o_LL_rms: float = 230.0    # line-to-line rms output (V)
    f_o: float = 60.0            # output frequency (Hz)
    P_o: float = 500.0           # nominal output power (W)
    # Equivalent line-to-neutral peak: V_o_LN_pk = V_o_LL_rms · √2/√3

    # Output LC filter (per phase, between leg pole and load terminal)
    L_f: float = 1e-3            # filter inductor (H)
    C_f: float = 10e-6           # filter cap, Y-connected (F)
    R_L: float = 0.0             # filter inductor ESR (Ω) — set non-zero for damping

    # Switching
    f_sw: float = 20e3           # switching frequency (Hz)

    # Grid-tie only
    V_grid_LL_rms: float = 230.0  # grid line-to-line rms (V)
    f_grid: float = 60.0          # grid frequency (Hz)
    L_grid: float = 2e-3          # grid-side coupling inductor (H, for grid-tie)
    R_grid: float = 0.05          # grid-side coupling inductor ESR (Ω)

    # Modulation
    m_a_max: float = 1.0          # SVPWM upper limit; 1.0 = SVPWM, 0.866 = SPWM only

    # ---- Derived quantities ----

    @property
    def V_o_LN_pk(self) -> float:
        """Peak line-to-neutral output voltage (V)."""
        return self.V_o_LL_rms * np.sqrt(2.0) / np.sqrt(3.0)

    @property
    def V_o_LL_pk(self) -> float:
        """Peak line-to-line output voltage (V)."""
        return self.V_o_LL_rms * np.sqrt(2.0)

    @property
    def m_a(self) -> float:
        """SVPWM modulation index: $m_a = 2 V_{o,LN,pk}/V_{dc}$.

        With SVPWM the maximum is $m_a = 1.0$ (corresponds to line-
        to-line peak = $V_{dc}$). With SPWM the maximum is $m_a =
        0.866 = \\sqrt 3/2$.
        """
        return 2.0 * self.V_o_LN_pk / self.V_dc

    @property
    def omega_o(self) -> float:
        """Output angular frequency (rad/s)."""
        return 2.0 * np.pi * self.f_o

    @property
    def omega_grid(self) -> float:
        """Grid angular frequency (rad/s)."""
        return 2.0 * np.pi * self.f_grid

    @property
    def T_sw(self) -> float:
        return 1.0 / self.f_sw

    @property
    def R_load(self) -> float:
        """Equivalent per-phase load resistance for the rated $P_o$ (Ω).

        $P_o = 3 V_{o,LN,rms}^2 / R_{load}$ → $R_{load} = 3 V_{o,LN,rms}^2 / P_o$
        Equivalent Y-connected resistive load.
        """
        V_LN_rms = self.V_o_LN_pk / np.sqrt(2.0)
        return 3.0 * V_LN_rms ** 2 / self.P_o

    @property
    def I_o_pk(self) -> float:
        """Peak per-phase load current at $P_o$ into the rated load (A).

        For purely resistive load: $I_{pk} = V_{o,LN,pk}/R_{load}$.
        """
        return self.V_o_LN_pk / self.R_load

    @property
    def f_filter_corner(self) -> float:
        """Output LC filter corner frequency (Hz).

        $f_c = 1/(2\\pi \\sqrt{L_f C_f})$
        """
        return 1.0 / (2.0 * np.pi * np.sqrt(self.L_f * self.C_f))

    @property
    def Q_filter(self) -> float:
        """Output LC filter Q at rated load.

        $Q = R_{load}\\sqrt{C_f/L_f}$ — buck-style.
        """
        return self.R_load * np.sqrt(self.C_f / self.L_f)


# ---------------------------------------------------------------------------
# Reference-frame transformations
# ---------------------------------------------------------------------------


def clarke_transform(v_a: np.ndarray, v_b: np.ndarray, v_c: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """abc → αβ. Amplitude-invariant (2/3 factor).

    $v_\\alpha = \\frac{2}{3}(v_a - \\frac{v_b}{2} - \\frac{v_c}{2})$,
    $v_\\beta = \\frac{2}{3} \\cdot \\frac{\\sqrt 3}{2}(v_b - v_c)$

    With the 2/3 factor, the αβ vector magnitude equals the *peak* of
    the phase voltage. (Power-invariant variant uses √(2/3) instead.)
    """
    v_alpha = (2.0/3.0) * (v_a - 0.5*v_b - 0.5*v_c)
    v_beta = (2.0/3.0) * (np.sqrt(3.0)/2.0) * (v_b - v_c)
    return v_alpha, v_beta


def inverse_clarke_transform(v_alpha: np.ndarray, v_beta: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """αβ → abc. Inverse of `clarke_transform`."""
    v_a = v_alpha
    v_b = -0.5 * v_alpha + (np.sqrt(3.0)/2.0) * v_beta
    v_c = -0.5 * v_alpha - (np.sqrt(3.0)/2.0) * v_beta
    return v_a, v_b, v_c


def park_transform(v_alpha: np.ndarray, v_beta: np.ndarray, theta: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """αβ → dq. Rotates by angle $\\theta$ aligning d with the rotating vector.

    $v_d = v_\\alpha \\cos\\theta + v_\\beta \\sin\\theta$,
    $v_q = -v_\\alpha \\sin\\theta + v_\\beta \\cos\\theta$
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    v_d = v_alpha * cos_t + v_beta * sin_t
    v_q = -v_alpha * sin_t + v_beta * cos_t
    return v_d, v_q


def inverse_park_transform(v_d: np.ndarray, v_q: np.ndarray, theta: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """dq → αβ. Inverse of `park_transform`."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    v_alpha = v_d * cos_t - v_q * sin_t
    v_beta = v_d * sin_t + v_q * cos_t
    return v_alpha, v_beta


# ---------------------------------------------------------------------------
# SVPWM via min-max injection
# ---------------------------------------------------------------------------


def svpwm_duties(v_ref_a: float | np.ndarray,
                 v_ref_b: float | np.ndarray,
                 v_ref_c: float | np.ndarray,
                 V_dc: float) -> tuple:
    """Compute the three leg duties for SVPWM via **min-max injection**.

    The standard SVPWM via vector decomposition (find sector, compute
    T0, T1, T2 dwell times, assemble switching pattern) is equivalent
    to *adding a common-mode injection* of $-(\\max(v_{abc}) +
    \\min(v_{abc}))/2$ to each phase reference before applying
    standard SPWM. This is far simpler to implement and produces
    identical line-to-line voltages.

    The extra output: SVPWM achieves a peak line-to-line equal to
    $V_{dc}$ (so peak phase = $V_{dc}/\\sqrt 3$ vs SPWM's $V_{dc}/2$
    — that's the ~15% bonus).

    Returns (duty_a, duty_b, duty_c), each in [0, 1] representing the
    on-fraction of the **top** switch in the corresponding leg.
    """
    v_ref_a = np.asarray(v_ref_a)
    v_ref_b = np.asarray(v_ref_b)
    v_ref_c = np.asarray(v_ref_c)
    v_max = np.maximum(np.maximum(v_ref_a, v_ref_b), v_ref_c)
    v_min = np.minimum(np.minimum(v_ref_a, v_ref_b), v_ref_c)
    v_offset = -(v_max + v_min) * 0.5
    v_a_pwm = v_ref_a + v_offset
    v_b_pwm = v_ref_b + v_offset
    v_c_pwm = v_ref_c + v_offset
    # Map to duty: 0.5 corresponds to mid-rail (0 V output relative to
    # bus midpoint), 1.0 to top-switch always on (+V_dc/2 output).
    duty_a = 0.5 + v_a_pwm / V_dc
    duty_b = 0.5 + v_b_pwm / V_dc
    duty_c = 0.5 + v_c_pwm / V_dc
    return duty_a, duty_b, duty_c


def spwm_duties(v_ref_a: float | np.ndarray,
                v_ref_b: float | np.ndarray,
                v_ref_c: float | np.ndarray,
                V_dc: float) -> tuple:
    """Standard SPWM duties (no common-mode injection).

    Each phase's duty is computed independently:
        $d_x = 0.5 + v_{ref,x} / V_{dc}$
    Linear range limited to $|v_{ref}| \\le V_{dc}/2$ → peak
    line-to-line $\\le V_{dc} \\sqrt 3 / 2 = 0.866 V_{dc}$.
    """
    v_ref_a = np.asarray(v_ref_a)
    v_ref_b = np.asarray(v_ref_b)
    v_ref_c = np.asarray(v_ref_c)
    duty_a = 0.5 + v_ref_a / V_dc
    duty_b = 0.5 + v_ref_b / V_dc
    duty_c = 0.5 + v_ref_c / V_dc
    return duty_a, duty_b, duty_c


# ---------------------------------------------------------------------------
# Open-loop switched simulator — 3-phase VSI with LC filter + RL load
# ---------------------------------------------------------------------------


def simulate_open_loop_svpwm(
    p: VSI3PhaseParams,
    *,
    m_a: float | None = None,
    n_cycles: int = 3,
    samples_per_period: int = 50,
    load_RL: tuple[float, float] | None = None,  # (R, L) of load if non-default
    use_svpwm: bool = True,
) -> dict[str, np.ndarray]:
    """Open-loop switched simulation: SVPWM (or SPWM) into LC filter + RL load.

    Produces the canonical "modulating sinusoid → switched output →
    filtered sinusoid" demonstration.

    State: filter inductor currents (i_La, i_Lb, i_Lc) and filter cap
    voltages (v_Ca, v_Cb, v_Cc). Three-phase Y-connected load.

    The 6 switches are represented implicitly: each leg's "pole
    voltage" is $0$ (bottom on) or $V_{dc}$ (top on). The phase voltage
    seen by the load (line-to-neutral) is computed assuming a floating
    neutral.
    """
    if m_a is None:
        m_a = p.m_a
    if load_RL is None:
        load_RL = (p.R_load, 0.0)  # resistive default
    R_l, L_l = load_RL

    T_o = 1.0 / p.f_o
    T_sw = p.T_sw
    dt = T_sw / samples_per_period
    t_end = n_cycles * T_o
    n_steps = int(t_end / dt) + 1

    # State vector: [i_La, i_Lb, i_Lc, v_Ca, v_Cb, v_Cc, i_loada, i_loadb, i_loadc]
    i_La = i_Lb = i_Lc = 0.0
    v_Ca = v_Cb = v_Cc = 0.0
    i_load_a = i_load_b = i_load_c = 0.0

    V_ref_pk = m_a * p.V_dc / 2.0  # peak per-phase reference

    record_every = max(1, samples_per_period // 4)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_pole_a_hist = np.zeros(n_rec)
    v_pole_b_hist = np.zeros(n_rec)
    v_pole_c_hist = np.zeros(n_rec)
    v_a_hist = np.zeros(n_rec)
    v_b_hist = np.zeros(n_rec)
    v_c_hist = np.zeros(n_rec)
    v_LL_ab_hist = np.zeros(n_rec)
    v_Ca_hist = np.zeros(n_rec)
    v_Cb_hist = np.zeros(n_rec)
    v_Cc_hist = np.zeros(n_rec)
    i_La_hist = np.zeros(n_rec)
    i_Lb_hist = np.zeros(n_rec)
    i_Lc_hist = np.zeros(n_rec)
    rec_idx = 0

    # Compensator runs once per Tsw; sample duties at the start of each cycle
    duty_a = duty_b = duty_c = 0.5

    for i in range(n_steps):
        t = i * dt
        # Update modulation at start of each Tsw
        cycle_pos_int = i % samples_per_period
        if cycle_pos_int == 0:
            theta = p.omega_o * t
            v_ref_a = V_ref_pk * np.cos(theta)
            v_ref_b = V_ref_pk * np.cos(theta - 2*np.pi/3)
            v_ref_c = V_ref_pk * np.cos(theta + 2*np.pi/3)
            if use_svpwm:
                duty_a, duty_b, duty_c = svpwm_duties(v_ref_a, v_ref_b, v_ref_c, p.V_dc)
            else:
                duty_a, duty_b, duty_c = spwm_duties(v_ref_a, v_ref_b, v_ref_c, p.V_dc)
            duty_a = float(np.clip(duty_a, 0.0, 1.0))
            duty_b = float(np.clip(duty_b, 0.0, 1.0))
            duty_c = float(np.clip(duty_c, 0.0, 1.0))

        cycle_pos = cycle_pos_int / samples_per_period
        # Pole voltage = V_dc if top switch on, else 0 (referenced to negative rail)
        v_pole_a = p.V_dc if cycle_pos < duty_a else 0.0
        v_pole_b = p.V_dc if cycle_pos < duty_b else 0.0
        v_pole_c = p.V_dc if cycle_pos < duty_c else 0.0

        # Floating-neutral phase voltage at the LEG output (before filter)
        v_n = (v_pole_a + v_pole_b + v_pole_c) / 3.0  # bus-mid neutral offset
        v_leg_a = v_pole_a - v_n
        v_leg_b = v_pole_b - v_n
        v_leg_c = v_pole_c - v_n

        # Filter inductor: L_f · di_L/dt = v_leg - v_C - R_L · i_L
        di_La = (v_leg_a - v_Ca - p.R_L * i_La) / p.L_f
        di_Lb = (v_leg_b - v_Cb - p.R_L * i_Lb) / p.L_f
        di_Lc = (v_leg_c - v_Cc - p.R_L * i_Lc) / p.L_f

        # Filter cap: C_f · dv_C/dt = i_L - i_load
        # i_load through (R, L) load: L_l · di_load/dt = v_C - R_l · i_load
        # If L_l = 0 (pure resistive): i_load = v_C / R_l (instantaneous)
        if L_l > 0:
            di_load_a = (v_Ca - R_l * i_load_a) / L_l
            di_load_b = (v_Cb - R_l * i_load_b) / L_l
            di_load_c = (v_Cc - R_l * i_load_c) / L_l
            i_load_a += di_load_a * dt
            i_load_b += di_load_b * dt
            i_load_c += di_load_c * dt
        else:
            i_load_a = v_Ca / R_l
            i_load_b = v_Cb / R_l
            i_load_c = v_Cc / R_l

        dv_Ca = (i_La - i_load_a) / p.C_f
        dv_Cb = (i_Lb - i_load_b) / p.C_f
        dv_Cc = (i_Lc - i_load_c) / p.C_f

        i_La += di_La * dt; i_Lb += di_Lb * dt; i_Lc += di_Lc * dt
        v_Ca += dv_Ca * dt; v_Cb += dv_Cb * dt; v_Cc += dv_Cc * dt

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_pole_a_hist[rec_idx] = v_pole_a; v_pole_b_hist[rec_idx] = v_pole_b
            v_pole_c_hist[rec_idx] = v_pole_c
            v_a_hist[rec_idx] = v_leg_a; v_b_hist[rec_idx] = v_leg_b
            v_c_hist[rec_idx] = v_leg_c
            v_LL_ab_hist[rec_idx] = v_pole_a - v_pole_b
            v_Ca_hist[rec_idx] = v_Ca; v_Cb_hist[rec_idx] = v_Cb
            v_Cc_hist[rec_idx] = v_Cc
            i_La_hist[rec_idx] = i_La; i_Lb_hist[rec_idx] = i_Lb
            i_Lc_hist[rec_idx] = i_Lc
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx],
        "v_pole_a": v_pole_a_hist[:rec_idx], "v_pole_b": v_pole_b_hist[:rec_idx],
        "v_pole_c": v_pole_c_hist[:rec_idx],
        "v_leg_a": v_a_hist[:rec_idx], "v_leg_b": v_b_hist[:rec_idx],
        "v_leg_c": v_c_hist[:rec_idx],
        "v_LL_ab": v_LL_ab_hist[:rec_idx],
        "v_Ca": v_Ca_hist[:rec_idx], "v_Cb": v_Cb_hist[:rec_idx],
        "v_Cc": v_Cc_hist[:rec_idx],
        "i_La": i_La_hist[:rec_idx], "i_Lb": i_Lb_hist[:rec_idx],
        "i_Lc": i_Lc_hist[:rec_idx],
    }


# ---------------------------------------------------------------------------
# Closed-loop stand-alone: dq voltage control with RL load
# ---------------------------------------------------------------------------


def standalone_voltage_plant(p: VSI3PhaseParams) -> signal.TransferFunction:
    """Plant for the stand-alone voltage loop in the dq frame.

    Within the dq frame, the LC filter loaded by $R_{load}$ looks
    like a second-order low-pass:

        $G(s) = \\frac{V_d}{d_d \\cdot V_{dc}/2}
              = \\frac{1/(L_f C_f)}{s^2 + s/(R_{load} C_f) + 1/(L_f C_f)}$

    DC gain = 1 (the dq-frame modulating signal directly sets the dq
    voltage at the filter output). $\\omega_n = 1/\\sqrt{L_f C_f}$.

    This is exactly the buck's plant — once you're in dq, a 3-phase
    inverter looks like a buck. That's the magic of dq.
    """
    L, C, R = p.L_f, p.C_f, p.R_load
    num = [1.0 / (L * C)]
    den = [1.0, 1.0 / (R * C), 1.0 / (L * C)]
    return signal.TransferFunction(num, den)


def simulate_closed_loop_standalone(
    p: VSI3PhaseParams,
    b_d: np.ndarray, a_d: np.ndarray,
    *,
    V_d_ref: float | None = None,
    V_q_ref: float = 0.0,
    n_cycles: int = 6,
    samples_per_period: int = 50,
    load_RL: tuple[float, float] | None = None,
    use_svpwm: bool = True,
) -> dict[str, np.ndarray]:
    """Closed-loop stand-alone VSI: dq voltage control + LC filter + RL load.

    Two compensators in parallel — one for $v_d$, one for $v_q$. The
    Park angle $\\theta = \\omega_o t$ rotates at the desired output
    frequency (we *generate* the angle internally, not measure it —
    that's what makes this "stand-alone").

    Compensator runs once per $T_{sw}$, sample-and-hold.
    """
    if V_d_ref is None:
        V_d_ref = p.V_o_LN_pk  # set d-axis to peak phase voltage
    if load_RL is None:
        load_RL = (p.R_load, 0.0)
    R_l, L_l = load_RL

    T_o = 1.0 / p.f_o
    T_sw = p.T_sw
    dt = T_sw / samples_per_period
    t_end = n_cycles * T_o
    n_steps = int(t_end / dt) + 1

    i_La = i_Lb = i_Lc = 0.0
    v_Ca = v_Cb = v_Cc = 0.0
    i_load_a = i_load_b = i_load_c = 0.0
    n_state = len(a_d) - 1
    state_d = np.zeros(n_state)
    state_q = np.zeros(n_state)
    duty_a = duty_b = duty_c = 0.5

    record_every = max(1, samples_per_period // 4)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_Ca_hist = np.zeros(n_rec); v_Cb_hist = np.zeros(n_rec); v_Cc_hist = np.zeros(n_rec)
    v_d_hist = np.zeros(n_rec); v_q_hist = np.zeros(n_rec)
    v_d_ref_hist = np.zeros(n_rec); v_q_ref_hist = np.zeros(n_rec)
    i_La_hist = np.zeros(n_rec)
    rec_idx = 0

    for i in range(n_steps):
        t = i * dt
        theta = p.omega_o * t
        # Measure v_C in dq
        v_alpha_meas, v_beta_meas = clarke_transform(np.array([v_Ca]),
                                                      np.array([v_Cb]),
                                                      np.array([v_Cc]))
        v_d_meas, v_q_meas = park_transform(v_alpha_meas, v_beta_meas,
                                             np.array([theta]))
        v_d_meas = float(v_d_meas[0]); v_q_meas = float(v_q_meas[0])

        cycle_pos_int = i % samples_per_period
        if cycle_pos_int == 0:
            # Compensator: two parallel PIs
            err_d = V_d_ref - v_d_meas
            v_c_d = b_d[0] * err_d + state_d[0]
            ns_d = np.zeros_like(state_d)
            for j in range(n_state - 1):
                ns_d[j] = b_d[j+1] * err_d - a_d[j+1] * v_c_d + state_d[j+1]
            ns_d[n_state - 1] = b_d[n_state] * err_d - a_d[n_state] * v_c_d
            state_d = ns_d

            err_q = V_q_ref - v_q_meas
            v_c_q = b_d[0] * err_q + state_q[0]
            ns_q = np.zeros_like(state_q)
            for j in range(n_state - 1):
                ns_q[j] = b_d[j+1] * err_q - a_d[j+1] * v_c_q + state_q[j+1]
            ns_q[n_state - 1] = b_d[n_state] * err_q - a_d[n_state] * v_c_q
            state_q = ns_q

            # dq → αβ → abc (reference voltages for SVPWM)
            v_alpha_ref, v_beta_ref = inverse_park_transform(
                np.array([v_c_d]), np.array([v_c_q]), np.array([theta])
            )
            v_a_ref, v_b_ref, v_c_ref = inverse_clarke_transform(
                v_alpha_ref, v_beta_ref
            )
            v_a_ref = float(v_a_ref[0]); v_b_ref = float(v_b_ref[0])
            v_c_ref = float(v_c_ref[0])
            if use_svpwm:
                duty_a, duty_b, duty_c = svpwm_duties(v_a_ref, v_b_ref, v_c_ref, p.V_dc)
            else:
                duty_a, duty_b, duty_c = spwm_duties(v_a_ref, v_b_ref, v_c_ref, p.V_dc)
            duty_a = float(np.clip(duty_a, 0.0, 1.0))
            duty_b = float(np.clip(duty_b, 0.0, 1.0))
            duty_c = float(np.clip(duty_c, 0.0, 1.0))

        cycle_pos = cycle_pos_int / samples_per_period
        v_pole_a = p.V_dc if cycle_pos < duty_a else 0.0
        v_pole_b = p.V_dc if cycle_pos < duty_b else 0.0
        v_pole_c = p.V_dc if cycle_pos < duty_c else 0.0
        v_n = (v_pole_a + v_pole_b + v_pole_c) / 3.0
        v_leg_a = v_pole_a - v_n
        v_leg_b = v_pole_b - v_n
        v_leg_c = v_pole_c - v_n

        di_La = (v_leg_a - v_Ca - p.R_L * i_La) / p.L_f
        di_Lb = (v_leg_b - v_Cb - p.R_L * i_Lb) / p.L_f
        di_Lc = (v_leg_c - v_Cc - p.R_L * i_Lc) / p.L_f

        if L_l > 0:
            i_load_a += (v_Ca - R_l * i_load_a) / L_l * dt
            i_load_b += (v_Cb - R_l * i_load_b) / L_l * dt
            i_load_c += (v_Cc - R_l * i_load_c) / L_l * dt
        else:
            i_load_a = v_Ca / R_l
            i_load_b = v_Cb / R_l
            i_load_c = v_Cc / R_l

        dv_Ca = (i_La - i_load_a) / p.C_f
        dv_Cb = (i_Lb - i_load_b) / p.C_f
        dv_Cc = (i_Lc - i_load_c) / p.C_f

        i_La += di_La * dt; i_Lb += di_Lb * dt; i_Lc += di_Lc * dt
        v_Ca += dv_Ca * dt; v_Cb += dv_Cb * dt; v_Cc += dv_Cc * dt

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_Ca_hist[rec_idx] = v_Ca; v_Cb_hist[rec_idx] = v_Cb
            v_Cc_hist[rec_idx] = v_Cc
            v_d_hist[rec_idx] = v_d_meas
            v_q_hist[rec_idx] = v_q_meas
            v_d_ref_hist[rec_idx] = V_d_ref
            v_q_ref_hist[rec_idx] = V_q_ref
            i_La_hist[rec_idx] = i_La
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx],
        "v_Ca": v_Ca_hist[:rec_idx], "v_Cb": v_Cb_hist[:rec_idx],
        "v_Cc": v_Cc_hist[:rec_idx],
        "v_d": v_d_hist[:rec_idx], "v_q": v_q_hist[:rec_idx],
        "v_d_ref": v_d_ref_hist[:rec_idx], "v_q_ref": v_q_ref_hist[:rec_idx],
        "i_La": i_La_hist[:rec_idx],
    }


# ---------------------------------------------------------------------------
# Grid-tie: PLL + dq current control
# ---------------------------------------------------------------------------


def simulate_closed_loop_gridtie(
    p: VSI3PhaseParams,
    b_i: np.ndarray, a_i: np.ndarray,
    b_pll: np.ndarray, a_pll: np.ndarray,
    *,
    I_d_ref: float = 0.0,   # active current reference (sets P injection)
    I_q_ref: float = 0.0,   # reactive current reference (sets Q)
    n_cycles: int = 8,
    samples_per_period: int = 50,
    use_svpwm: bool = True,
) -> dict[str, np.ndarray]:
    """Grid-tie VSI: SRF-PLL + dq current control + grid coupling.

    Architecture:
        1. Grid voltages $v_{ga,b,c}$ measured.
        2. SRF-PLL extracts grid angle $\\theta_{pll}$: project to dq
           via the *PLL angle*, drive $v_{q,grid}$ to zero with a PI →
           frequency correction → integrate → PLL angle.
        3. Measure grid current $i_{a,b,c}$, transform to dq.
        4. Current loop: PI on (I_d_ref - i_d) and (I_q_ref - i_q) →
           v_d_cmd, v_q_cmd.
        5. Inverse-transform to abc, SVPWM, gates.

    Power injection: P = (3/2) V_d I_d, Q = -(3/2) V_d I_q (with PLL
    locked so that V_q = 0).

    Grid voltage modelled as an ideal source (zero impedance); the
    grid coupling inductor $L_{grid}$ is the only thing between the
    inverter output and the grid.
    """
    T_grid = 1.0 / p.f_grid
    T_sw = p.T_sw
    dt = T_sw / samples_per_period
    t_end = n_cycles * T_grid
    n_steps = int(t_end / dt) + 1

    V_grid_pk = p.V_grid_LL_rms * np.sqrt(2.0) / np.sqrt(3.0)
    omega_grid_true = p.omega_grid

    # State: grid-side inductor currents (filtered through L_grid)
    i_a = i_b = i_c = 0.0

    # PLL state
    theta_pll = 0.0
    omega_pll = omega_grid_true  # warm-start at grid frequency
    n_state_pll = len(a_pll) - 1
    state_pll = np.zeros(n_state_pll)

    # Current-loop state (d and q parallel)
    n_state_i = len(a_i) - 1
    state_id = np.zeros(n_state_i)
    state_iq = np.zeros(n_state_i)
    v_d_cmd = 0.0
    v_q_cmd = 0.0
    duty_a = duty_b = duty_c = 0.5

    record_every = max(1, samples_per_period // 4)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_ga_hist = np.zeros(n_rec)
    i_a_hist = np.zeros(n_rec)
    i_d_hist = np.zeros(n_rec); i_q_hist = np.zeros(n_rec)
    theta_pll_hist = np.zeros(n_rec)
    theta_true_hist = np.zeros(n_rec)
    omega_pll_hist = np.zeros(n_rec)
    rec_idx = 0

    for i in range(n_steps):
        t = i * dt
        theta_true = omega_grid_true * t
        v_ga = V_grid_pk * np.cos(theta_true)
        v_gb = V_grid_pk * np.cos(theta_true - 2*np.pi/3)
        v_gc = V_grid_pk * np.cos(theta_true + 2*np.pi/3)

        cycle_pos_int = i % samples_per_period
        if cycle_pos_int == 0:
            # PLL: project grid voltage to dq using PLL angle, drive Vq → 0
            v_alpha_g, v_beta_g = clarke_transform(
                np.array([v_ga]), np.array([v_gb]), np.array([v_gc])
            )
            _, v_qg_arr = park_transform(v_alpha_g, v_beta_g,
                                          np.array([theta_pll]))
            v_qg = float(v_qg_arr[0])

            # PI on v_qg → omega correction
            err_pll = -v_qg  # want v_qg = 0
            domega = b_pll[0] * err_pll + state_pll[0]
            ns_pll = np.zeros_like(state_pll)
            for j in range(n_state_pll - 1):
                ns_pll[j] = b_pll[j+1] * err_pll - a_pll[j+1] * domega + state_pll[j+1]
            ns_pll[n_state_pll - 1] = b_pll[n_state_pll] * err_pll - a_pll[n_state_pll] * domega
            state_pll = ns_pll
            omega_pll = omega_grid_true + domega

            # Current loop: measure i in dq via PLL angle
            i_alpha, i_beta = clarke_transform(np.array([i_a]),
                                                np.array([i_b]),
                                                np.array([i_c]))
            i_d, i_q = park_transform(i_alpha, i_beta, np.array([theta_pll]))
            i_d = float(i_d[0]); i_q = float(i_q[0])

            err_id = I_d_ref - i_d
            v_c_id = b_i[0] * err_id + state_id[0]
            ns_id = np.zeros_like(state_id)
            for j in range(n_state_i - 1):
                ns_id[j] = b_i[j+1] * err_id - a_i[j+1] * v_c_id + state_id[j+1]
            ns_id[n_state_i - 1] = b_i[n_state_i] * err_id - a_i[n_state_i] * v_c_id
            state_id = ns_id

            err_iq = I_q_ref - i_q
            v_c_iq = b_i[0] * err_iq + state_iq[0]
            ns_iq = np.zeros_like(state_iq)
            for j in range(n_state_i - 1):
                ns_iq[j] = b_i[j+1] * err_iq - a_i[j+1] * v_c_iq + state_iq[j+1]
            ns_iq[n_state_i - 1] = b_i[n_state_i] * err_iq - a_i[n_state_i] * v_c_iq
            state_iq = ns_iq

            # Add feed-forward of grid voltage (canonical for grid-tie)
            v_dg_meas = float(park_transform(v_alpha_g, v_beta_g,
                                              np.array([theta_pll]))[0][0])
            v_d_cmd = v_c_id + v_dg_meas - omega_pll * p.L_grid * i_q
            v_q_cmd = v_c_iq + 0.0 + omega_pll * p.L_grid * i_d

            v_alpha_cmd, v_beta_cmd = inverse_park_transform(
                np.array([v_d_cmd]), np.array([v_q_cmd]), np.array([theta_pll])
            )
            v_a_cmd, v_b_cmd, v_c_cmd = inverse_clarke_transform(
                v_alpha_cmd, v_beta_cmd
            )
            v_a_cmd = float(v_a_cmd[0]); v_b_cmd = float(v_b_cmd[0])
            v_c_cmd = float(v_c_cmd[0])
            if use_svpwm:
                duty_a, duty_b, duty_c = svpwm_duties(v_a_cmd, v_b_cmd, v_c_cmd, p.V_dc)
            else:
                duty_a, duty_b, duty_c = spwm_duties(v_a_cmd, v_b_cmd, v_c_cmd, p.V_dc)
            duty_a = float(np.clip(duty_a, 0.0, 1.0))
            duty_b = float(np.clip(duty_b, 0.0, 1.0))
            duty_c = float(np.clip(duty_c, 0.0, 1.0))

        # Integrate PLL angle
        theta_pll = (theta_pll + omega_pll * dt) % (2*np.pi)

        cycle_pos = cycle_pos_int / samples_per_period
        v_pole_a = p.V_dc if cycle_pos < duty_a else 0.0
        v_pole_b = p.V_dc if cycle_pos < duty_b else 0.0
        v_pole_c = p.V_dc if cycle_pos < duty_c else 0.0
        v_n = (v_pole_a + v_pole_b + v_pole_c) / 3.0
        v_leg_a = v_pole_a - v_n
        v_leg_b = v_pole_b - v_n
        v_leg_c = v_pole_c - v_n

        # Grid-side inductor: L_grid · di/dt = v_leg - v_grid - R_grid · i
        di_a = (v_leg_a - v_ga - p.R_grid * i_a) / p.L_grid
        di_b = (v_leg_b - v_gb - p.R_grid * i_b) / p.L_grid
        di_c = (v_leg_c - v_gc - p.R_grid * i_c) / p.L_grid
        i_a += di_a * dt; i_b += di_b * dt; i_c += di_c * dt

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_ga_hist[rec_idx] = v_ga
            i_a_hist[rec_idx] = i_a
            # Recompute dq for plotting
            ia_arr, ib_arr = clarke_transform(np.array([i_a]),
                                                np.array([i_b]),
                                                np.array([i_c]))
            id_, iq_ = park_transform(ia_arr, ib_arr, np.array([theta_pll]))
            i_d_hist[rec_idx] = float(id_[0])
            i_q_hist[rec_idx] = float(iq_[0])
            theta_pll_hist[rec_idx] = theta_pll
            theta_true_hist[rec_idx] = theta_true % (2*np.pi)
            omega_pll_hist[rec_idx] = omega_pll
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx],
        "v_ga": v_ga_hist[:rec_idx],
        "i_a": i_a_hist[:rec_idx],
        "i_d": i_d_hist[:rec_idx], "i_q": i_q_hist[:rec_idx],
        "theta_pll": theta_pll_hist[:rec_idx],
        "theta_true": theta_true_hist[:rec_idx],
        "omega_pll": omega_pll_hist[:rec_idx],
    }


# ---------------------------------------------------------------------------
# Power-quality / harmonic analysis
# ---------------------------------------------------------------------------


def fundamental_rms(x: np.ndarray, f_s: float, f_fund: float) -> float:
    """Extract the rms value of the fundamental component at `f_fund`."""
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/f_s)
    idx = np.argmin(np.abs(freqs - f_fund))
    return float(np.abs(X[idx]) * 2 / N / np.sqrt(2.0))


def thd(x: np.ndarray, f_s: float, f_fund: float, n_harmonics: int = 40) -> float:
    """THD of `x` referenced to its fundamental at `f_fund`."""
    N = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X) * 2.0 / N
    freqs = np.fft.rfftfreq(N, d=1.0/f_s)
    harm_freqs = f_fund * np.arange(1, n_harmonics + 1)
    harm_mags = np.array([mag[np.argmin(np.abs(freqs - hf))] for hf in harm_freqs])
    if harm_mags[0] < 1e-12:
        return 0.0
    rest_sq = float(np.sum(harm_mags[1:] ** 2))
    return float(np.sqrt(rest_sq) / harm_mags[0])


# ---------------------------------------------------------------------------
# Operating-point report
# ---------------------------------------------------------------------------


def operating_point_report(p: VSI3PhaseParams, mode: Mode = "standalone") -> str:
    lines = [
        f"Three-phase VSI operating point ({mode})",
        f"  V_dc           = {p.V_dc:8.2f} V",
        f"  Output spec",
        f"    V_o, LL rms  = {p.V_o_LL_rms:8.2f} V",
        f"    V_o, LN pk   = {p.V_o_LN_pk:8.2f} V",
        f"    f_o          = {p.f_o:8.2f} Hz",
        f"    P_o          = {p.P_o:8.2f} W",
        f"    R_load (Y)   = {p.R_load:8.2f} Ω",
        f"    I_o peak     = {p.I_o_pk:8.3f} A",
        "",
        f"  Modulation",
        f"    m_a (SVPWM)  = {p.m_a:8.4f}    (max 1.0; SPWM limit 0.866)",
        f"    Bonus vs SPWM: {(1.0/0.866 - 1)*100:.1f}% more output",
        "",
        f"  Output LC filter (per phase)",
        f"    L_f          = {p.L_f * 1e3:8.3f} mH",
        f"    C_f          = {p.C_f * 1e6:8.1f} µF",
        f"    f_c          = {p.f_filter_corner:8.1f} Hz",
        f"    Q            = {p.Q_filter:8.3f}",
        "",
        f"  Switching",
        f"    f_sw         = {p.f_sw / 1e3:8.2f} kHz",
        f"    f_sw / f_o   = {p.f_sw / p.f_o:8.0f}",
        f"    f_sw / f_c   = {p.f_sw / p.f_filter_corner:8.1f}× (attenuation margin)",
    ]
    if mode == "gridtie":
        lines += [
            "",
            f"  Grid parameters",
            f"    V_grid LL    = {p.V_grid_LL_rms:8.2f} V rms",
            f"    f_grid       = {p.f_grid:8.2f} Hz",
            f"    L_grid       = {p.L_grid * 1e3:8.3f} mH",
            f"    R_grid       = {p.R_grid:8.4f} Ω",
        ]
    return "\n".join(lines)


__all__ = [
    "VSI3PhaseParams",
    "Mode",
    "clarke_transform",
    "inverse_clarke_transform",
    "park_transform",
    "inverse_park_transform",
    "svpwm_duties",
    "spwm_duties",
    "standalone_voltage_plant",
    "simulate_open_loop_svpwm",
    "simulate_closed_loop_standalone",
    "simulate_closed_loop_gridtie",
    "fundamental_rms",
    "thd",
    "operating_point_report",
]
