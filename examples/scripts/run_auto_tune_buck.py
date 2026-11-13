#!/usr/bin/env python3
"""Auto-tune a buck PI from its measured plant Bode, then measure it.

The workflow a control engineer does by hand:

  1. Sweep the open-loop plant: measure G_vd(jω) (duty → V_out).
  2. Form the plant the PI really sees: G_vd · F_lpf · e^(−jωT/2),
     i.e. the measured power stage, the feedback low-pass the closed
     loop runs (τ = 320 µs) and the half-period hold of a PI that
     updates once per PWM period.
  3. `tune_pi_from_bode(...)` for f_c = 400 Hz, PM = 60°, and read the
     predicted margins off L = C·P.
  4. MEASURE the loop gain of the switching closed loop by injecting
     a small sine into the duty (L = −d_PI/d, one DFT bin per point)
     and compare its crossover / phase margin with the design.
  5. Simulate a cold start and a 12 → 14 V setpoint step.

Why f_c = 400 Hz and not the 2 kHz an earlier version used: the LC
resonance (f_LC = 2.32 kHz, Q = 3.4) puts the plant phase in the
(−120°, −30°) window a PI needs for 60° only around f_LC, where |G|
peaks. A PI tuned "for 2 kHz" there has |L| > 1 again past the peak,
and the last 0 dB crossing (the one that sets stability) had 37.9°,
while the tuner printed 60° "by construction". Below f_LC the feedback
low-pass supplies the phase lag and keeps a single crossover.
An earlier version also tuned without the low-pass it then simulated,
and judged regulation 1.0–1.5 ms into a cold start (11.12 V, target
12 V): the loop needs ~6 ms to settle.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# Plant
V_IN     = 24.0
V_REF    = 12.0
L_VAL    = 100e-6
C_VAL    = 47e-6
R_LOAD   = 5.0
F_PWM    = 100e3
T_PWM    = 1.0 / F_PWM
D_OP     = V_REF / V_IN

# Bode sweep settings. The duty sine is quantised to ε·T_PWM/dt levels
# per period: dt = ε·T_PWM/16 keeps its fundamental within 0.6 % (see
# run_ac_sweep_buck.py). Seeded ICs start each point at the averaged
# operating point, so 3 ms (6·2RC) of settling is enough.
EPS_DUTY    = 0.05
N_BODE_PTS  = 16
F_MIN_BODE  = 100.0
F_MAX_BODE  = 8000.0
DT_BODE     = T_PWM * EPS_DUTY / 16.0     # 31.25 ns

# Controller as simulated: feedback low-pass every dt, PI once per period.
TAU_LPF     = 320e-6
F_CROSSOVER = 400.0
PM_TARGET   = 60.0
DUTY_MIN, DUTY_MAX = 0.05, 0.95

# Closed-loop runs. dt = T_PWM/100 → duty steps of 0.01 (0.24 V).
DT_CL       = 1e-7
T_END_CL    = 20e-3
T_SETPOINT_STEP = 10e-3

# Loop-gain injection: f = F_PWM/n (a whole number of PI ticks per
# injection period), ε small enough to stay linear.
INJ_N       = (500, 354, 250, 177, 125)   # 200 … 800 Hz
INJ_EPS     = 0.02
DT_INJ      = 5e-8
INJ_SETTLE  = 4e-3
INJ_CYCLES  = 4


def build_buck(seeded: bool = False) -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source        ("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1",  "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode                 ("D_FW","gnd", "sw", 1e3, 1e-9, V_th=0.7)
    # Seeded: start at the averaged operating point (CCM valley current
    # at the start of the ON time), so runs skip the cold-start ring.
    i_valley = V_REF / R_LOAD - (V_IN - V_REF) * D_OP * T_PWM / (2 * L_VAL)
    b.add_inductor("L1", "sw", "vout", L_VAL,
                   i0=i_valley if seeded else None)
    b.add_capacitor("Cout", "vout", "gnd", C_VAL,
                    c0=V_REF if seeded else None)
    b.add_resistor              ("R_L", "vout","gnd", R_LOAD)
    return b


def sweep_plant_bode():
    """Step 1: sweep the open-loop plant at constant D = 0.5."""
    builder = build_buck(seeded=True)
    vout_idx = builder.node_id_of("vout")
    q1 = builder.switch_index_of("Q1")
    n_sw = builder.graph.num_switches
    state_n = builder.pool.state_size(builder.graph)

    def excite(t, eps, freq):
        return [0.0] * state_n

    def make_switch_fn(eps, freq):
        omega = 2 * math.pi * freq
        def sw(t):
            duty = D_OP + eps * math.sin(omega * t)
            m = p.SwitchStateMask(n_sw)
            if p.control.pwm_on(t, T_PWM, duty):
                m.set(q1, True)
            return m
        return sw

    freqs = np.logspace(math.log10(F_MIN_BODE),
                          math.log10(F_MAX_BODE), N_BODE_PTS)
    print(f"  Sweeping buck open-loop plant (D={D_OP}) at "
          f"{len(freqs)} freqs from {F_MIN_BODE:.0f} Hz to "
          f"{F_MAX_BODE/1e3:.1f} kHz...")
    H = []
    for f in freqs:   # per point: settling is a time (3 ms), not cycles
        r = p.run_ac_sweep(
            builder, freqs=[f], excite_fn=excite,
            switch_fn_factory=make_switch_fn, output_idx=vout_idx,
            dt_fn=lambda f: min(DT_BODE, (1.0 / f) / 200.0),
            perturbation_amplitude=EPS_DUTY, cycles_per_point=2,
            settling_cycles=max(1, math.ceil(3e-3 * f)),
            start_from_dc_op=False, verbose=False)
        H.append(r.H[0])
    return freqs, np.asarray(H)


def controller_plant(freqs, H):
    """Step 2: the plant the PI sees — measured G_vd · LPF · T/2 hold."""
    s = 2j * np.pi * np.asarray(freqs)
    return H / (1 + s * TAU_LPF) * np.exp(-s * T_PWM / 2)


def run_closed_loop(Kp, Ki, *, t_end, dt, setpoint, inj=None, seeded=False):
    """Switching closed loop: LPF every dt, PI + optional duty injection
    on the t = k·T_PWM schedule. Returns (result, ticks[t, d_pi, d])."""
    b = build_buck(seeded=seeded)
    vi, q1, n_sw = b.node_id_of("vout"), b.switch_index_of("Q1"), b.graph.num_switches
    pi = p.PIController(Kp=Kp, Ki=Ki, output_min=DUTY_MIN, output_max=DUTY_MAX,
                        integrator_state=D_OP if seeded else 0.0)
    lpf = p.FirstOrderLowPass(tau=TAU_LPF, state=V_REF if seeded else 0.0)
    duty = [D_OP]
    ticks = []

    def tick(t, x):
        d_pi = pi.update(setpoint=setpoint(t), measured=lpf.state, dt=T_PWM)
        d = d_pi + (inj[0] * math.sin(2 * math.pi * inj[1] * t) if inj else 0.0)
        duty[0] = min(DUTY_MAX, max(DUTY_MIN, d))
        ticks.append((t, d_pi, d))

    next_k = [0]

    def observer(t, x):           # called every step
        lpf.update(input_value=float(x[vi]), dt=dt)
        if t >= (next_k[0] - 1e-9) * T_PWM:   # PI on the k·T_PWM schedule
            tick(t, x)
            next_k[0] = math.floor(t / T_PWM + 1e-9) + 1

    def switch_fn(t):
        m = p.SwitchStateMask(n_sw)
        if p.control.pwm_on(t, T_PWM, duty[0]):
            m.set(q1, True)
        return m

    res = p.simulate(b, t_end=t_end, dt=dt, switch_fn=switch_fn,
                     step_observer=observer)
    return res, np.asarray(ticks)


def measure_loop_gain(Kp, Ki):
    """Step 4: L(jω) = −D_PI/D from a duty injection, one DFT bin."""
    f_meas, L_meas = [], []
    for n in INJ_N:
        f = F_PWM / n
        t_end = INJ_SETTLE + INJ_CYCLES / f
        _, tk = run_closed_loop(Kp, Ki, t_end=t_end, dt=DT_INJ,
                                setpoint=lambda t: V_REF,
                                inj=(INJ_EPS, f), seeded=True)
        w = tk[:, 0] >= t_end - INJ_CYCLES / f - 0.5 * T_PWM
        tk = tk[w][: INJ_CYCLES * n]
        e = np.exp(-2j * np.pi * f * tk[:, 0])
        f_meas.append(f)
        L_meas.append(-(tk[:, 1] @ e) / (tk[:, 2] @ e))
    return np.asarray(f_meas), np.asarray(L_meas)


def crossovers_hz(f, L):
    """Frequencies where |L| falls through 1, log-interpolated."""
    lf, lm = np.log10(f), np.log10(np.abs(L))
    return [10 ** (lf[i] - lm[i] * (lf[i + 1] - lf[i]) / (lm[i + 1] - lm[i]))
            for i in np.where((lm[:-1] >= 0) & (lm[1:] < 0))[0]]


def main() -> None:
    print("=" * 60)
    print("STEP 1 — sweep open-loop plant Bode")
    print("=" * 60)
    freqs, H = sweep_plant_bode()
    s = 2j * np.pi * freqs
    H_avg = V_IN / (L_VAL * C_VAL * s**2 + L_VAL / R_LOAD * s + 1)
    dB_err = 20 * np.log10(np.abs(H) / np.abs(H_avg))
    print(f"  vs averaged model V_in/(LCs²+(L/R)s+1): |dB error| "
          f"max {np.abs(dB_err).max():.2f} dB (largest at the LC peak, "
          f"where ε = {EPS_DUTY} is large-signal)")

    print(f"\n{'=' * 60}")
    print(f"STEP 2/3 — tune PI for f_c={F_CROSSOVER:.0f} Hz, PM={PM_TARGET}°")
    print(f"{'=' * 60}")
    P = controller_plant(freqs, H)
    tune = p.tune_pi_from_bode(freqs, P, f_crossover=F_CROSSOVER,
                               phase_margin_deg=PM_TARGET,
                               output_min=DUTY_MIN, output_max=DUTY_MAX)
    for w in tune["warnings"]:
        print(f"  WARNING: {w}")
    Kp, Ki = tune["Kp"], tune["Ki"]
    L = p.loop_gain(freqs, P, Kp, Ki)
    pm = p.phase_margin_from_loop(freqs, L)
    gm = p.gain_margin_from_loop(freqs, L)
    xo = tune["loop_crossovers_hz"]
    print(f"  plant seen by the PI @ {F_CROSSOVER:.0f} Hz: "
          f"|P| = {tune['plant_mag_at_crossover']:.3f}, "
          f"∠P = {tune['plant_phase_at_crossover_deg']:+.2f}°")
    print(f"  Kp = {Kp:.5f}   Ki = {Ki:.2f}")
    print(f"  predicted: crossover {[round(x) for x in xo]} Hz, "
          f"PM = {pm:.1f}°, GM = {gm:.1f} dB")

    print(f"\n{'=' * 60}")
    print("STEP 4 — measure L(jω) by duty injection in the switching loop")
    print(f"{'=' * 60}")
    f_m, L_m = measure_loop_gain(Kp, Ki)
    for f, Lv in zip(f_m, L_m):
        print(f"  {f:6.1f} Hz: |L| = {abs(Lv):.3f}  ∠L = "
              f"{np.degrees(np.angle(Lv)):+7.1f}°")
    pm_m = p.phase_margin_from_loop(f_m, L_m)
    xo_m = crossovers_hz(f_m, L_m)
    print(f"  measured: crossover {[round(x) for x in xo_m]} Hz, "
          f"PM = {pm_m:.1f}° (designed {PM_TARGET:.0f}°, "
          f"log-interpolated between injection points)")

    print(f"\n{'=' * 60}")
    print("STEP 5 — closed-loop cold start + setpoint step")
    print(f"{'=' * 60}")
    setpoint_fn = lambda t: V_REF if t < T_SETPOINT_STEP else V_REF + 2.0  # noqa: E731
    res, _ = run_closed_loop(Kp, Ki, t_end=T_END_CL, dt=DT_CL,
                             setpoint=setpoint_fn)
    times = np.asarray(res.times) * 1e3
    v_out = res.v("vout")
    v_pre = p.time_average(res.times, v_out, T_SETPOINT_STEP - 2e-3,
                           T_SETPOINT_STEP)
    v_post = p.time_average(res.times, v_out, t_from=T_END_CL - 2e-3)
    print(f"  V_out pre-step  (8-10 ms):  mean={v_pre:.3f} V "
          f"(target {V_REF:.1f} ± 0.05)")
    print(f"  V_out post-step (18-20 ms): mean={v_post:.3f} V "
          f"(target {V_REF + 2.0:.1f} ± 0.05)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ((ax_m, ax_t), (ax_p, ax_x)) = plt.subplots(2, 2, figsize=(13, 8))
    ax_m.semilogx(freqs, 20 * np.log10(np.abs(L)), label="predicted L = C·P")
    ax_m.semilogx(f_m, 20 * np.log10(np.abs(L_m)), "o", label="measured (injection)")
    ax_m.axhline(0, color="k", lw=0.5)
    ax_m.set_ylabel("|L| [dB]"); ax_m.grid(True, which="both", alpha=0.3)
    ax_m.legend(); ax_m.set_title(f"Loop gain — PM {pm:.1f}° predicted, "
                                  f"{pm_m:.1f}° measured")
    ax_p.semilogx(freqs, np.degrees(np.unwrap(np.angle(L))))
    ax_p.semilogx(f_m, np.degrees(np.unwrap(np.angle(L_m))), "o")
    ax_p.axhline(-180, color="k", ls="--", lw=0.5)
    ax_p.set_xlabel("freq [Hz]"); ax_p.set_ylabel("∠L [deg]")
    ax_p.grid(True, which="both", alpha=0.3)
    ax_t.plot(times, v_out, lw=0.6, label="V_out")
    ax_t.plot(times, [setpoint_fn(t) for t in res.times], "r--", lw=0.8,
              label="setpoint")
    ax_t.set_xlabel("time [ms]"); ax_t.set_ylabel("V_out [V]")
    ax_t.legend(); ax_t.grid(alpha=0.3)
    ax_t.set_title(f"Kp={Kp:.4f}, Ki={Ki:.1f}")
    ax_x.semilogx(freqs, 20 * np.log10(np.abs(H)), label="measured G_vd")
    ax_x.semilogx(freqs, 20 * np.log10(np.abs(H_avg)), "--", label="averaged model")
    ax_x.set_xlabel("freq [Hz]"); ax_x.set_ylabel("|G_vd| [dB]")
    ax_x.legend(); ax_x.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "auto_tune_buck.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
