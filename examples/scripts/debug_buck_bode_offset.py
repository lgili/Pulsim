#!/usr/bin/env python3
"""Debug the +10 dB magnitude offset on the buck control-to-output Bode.

Strategy: run the buck at ONE frequency (100 Hz, well below the LC
resonance so DC gain dominates), plot V_out vs time, manually
measure the AC amplitude, and compare:

  1. Direct peak-to-peak measurement
  2. RMS-based amplitude
  3. extract_phasor() result
  4. Analytical prediction (V_in·G_LC at 100 Hz ≈ V_in = 24)

If (1)==(2)==(3) but ≠ (4), the plant response really is bigger
than analytical predicts → averaged-model error.
If (1)==(2) but (3) ≠ them → DFT bug.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_IN     = 24.0
L_VAL    = 100e-6
C_VAL    = 47e-6
R_LOAD   = 5.0
F_PWM    = 100e3
T_PWM    = 1.0 / F_PWM
D_OP     = 0.5
F_TEST   = 100.0             # well below LC resonance
T_TEST   = 1.0 / F_TEST
EPS_DUTY = 0.01
DT       = T_PWM / 400        # 25 ns — finer PWM duty resolution (1/400=0.0025) than EPS=0.01
T_SETTLE = 30.0e-3
T_END    = T_SETTLE + 5 * T_TEST


def build_buck() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source        ("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1",  "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode                 ("D_FW","gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor              ("L1",  "sw", "vout", L_VAL)
    b.add_capacitor             ("Cout","vout","gnd", C_VAL)
    b.add_resistor              ("R_L", "vout","gnd", R_LOAD)
    return b


def main() -> None:
    builder = build_buck()
    vout_idx = builder.node_id_of("vout")
    n_sw = builder.graph.num_switches

    # SAMPLE-AND-HOLD PWM: duty is latched at the start of each
    # PWM cycle, NOT recomputed every dt. This is what a real
    # digital modulator does.
    latched_duty = [D_OP]
    cycle_start_t = [-1.0]

    def switch_fn(t: float):
        # Detect a new PWM cycle.
        cycle_idx = int(t / T_PWM)
        new_cycle_start = cycle_idx * T_PWM
        if abs(new_cycle_start - cycle_start_t[0]) > 1e-12:
            # Latch the duty using the sine sampled AT the start
            # of the new cycle (not at t).
            latched_duty[0] = (D_OP +
                EPS_DUTY * math.sin(2 * math.pi * F_TEST
                                     * new_cycle_start))
            cycle_start_t[0] = new_cycle_start
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(n_sw)
        if phase < latched_duty[0]:
            m.set(0, True)
        return m

    print(f"Buck Bode debug:")
    print(f"  V_in = {V_IN}V, D_op = {D_OP}, ε = {EPS_DUTY}")
    print(f"  f_test = {F_TEST} Hz  (T = {T_TEST*1e3:.1f} ms)")
    print(f"  t_end = {T_END*1e3:.1f} ms, dt = {DT*1e9:.0f} ns")

    res = p.simulate(builder, t_end=T_END, dt=DT,
                       switch_fn=switch_fn,
                       start_from_dc_op=False,
                       max_event_iterations=8)
    print(f"  samples: {res.num_steps()}")

    times = np.asarray(res.times)
    v_out = np.array([s[vout_idx] for s in res.states])

    # Filter PWM ripple before measurement: average over each
    # PWM cycle to recover the slow-varying envelope.
    samples_per_pwm = int(T_PWM / DT)
    # Cycle-aligned moving average — chop into PWM blocks and
    # average each, then expand back.
    n_blocks = len(v_out) // samples_per_pwm
    v_out_avg = np.zeros(n_blocks * samples_per_pwm)
    t_avg = np.zeros(n_blocks * samples_per_pwm)
    for k in range(n_blocks):
        i0 = k * samples_per_pwm
        i1 = i0 + samples_per_pwm
        v_out_avg[i0:i1] = v_out[i0:i1].mean()
        t_avg[i0:i1] = times[i0:i1]

    # Drop settling region — trim everything to the same length
    # as the cycle-averaged signal (which is a multiple of
    # samples_per_pwm).
    n_aligned = len(v_out_avg)
    k_settle = int(T_SETTLE / DT)
    t_w = times[k_settle:n_aligned]
    v_w = v_out[k_settle:n_aligned]
    v_w_avg = v_out_avg[k_settle:n_aligned]

    # Method 1: peak-to-peak / 2 on the cycle-averaged signal.
    v_pp = float(np.ptp(v_w_avg))
    amp_pp = v_pp / 2.0

    # Method 2: RMS-based amplitude. For a pure sine, A = √2 · RMS.
    v_w_avg_ac = v_w_avg - v_w_avg.mean()
    rms = float(np.sqrt(np.mean(v_w_avg_ac ** 2)))
    amp_rms = rms * math.sqrt(2)

    # Method 3: extract_phasor on the cycle-averaged AC signal.
    phasor = p.extract_phasor(t_w, v_w_avg_ac, F_TEST)
    amp_phasor = abs(phasor)
    phase_phasor = math.degrees(np.angle(phasor))

    # Method 4: same but on the RAW (un-averaged) signal.
    v_raw_ac = v_w - v_w.mean()
    phasor_raw = p.extract_phasor(t_w, v_raw_ac, F_TEST)
    amp_phasor_raw = abs(phasor_raw)

    # Analytical: H_LC at 100 Hz with V_in DC gain.
    omega = 2 * math.pi * F_TEST
    H_LC = 1.0 / (L_VAL * C_VAL * (1j*omega)**2
                    + (L_VAL / R_LOAD) * (1j*omega) + 1.0)
    amp_analytical = EPS_DUTY * V_IN * abs(H_LC)

    print(f"\nMeasured V_out AC amplitude at f={F_TEST} Hz:")
    print(f"  peak-to-peak/2 (cycle-avg):    {amp_pp:.5f} V")
    print(f"  RMS √2 (cycle-avg):            {amp_rms:.5f} V")
    print(f"  DFT phasor (cycle-avg):        {amp_phasor:.5f} V  "
          f"(phase {phase_phasor:+.2f}°)")
    print(f"  DFT phasor (raw with PWM):     {amp_phasor_raw:.5f} V")
    print(f"  ANALYTICAL  ε·V_in·|H_LC|:     {amp_analytical:.5f} V")
    print(f"  amp_phasor / amp_analytical:   {amp_phasor/amp_analytical:.3f}x")
    print(f"  dB(phasor/analytical):         "
          f"{20*math.log10(amp_phasor/amp_analytical):+.2f} dB")
    print(f"  V_out DC:                      {v_w_avg.mean():.3f} V "
          f"(target D·V_in = {D_OP*V_IN:.1f})")

    # Plot.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax_full, ax_avg) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_full.plot(times * 1e3, v_out, color="tab:blue", lw=0.3, alpha=0.7,
                  label="V_out raw (PWM ripple)")
    ax_full.plot(t_avg * 1e3, v_out_avg, color="tab:red", lw=1.0,
                  label="V_out cycle-averaged")
    ax_full.axvline(T_SETTLE * 1e3, color="g", ls=":", lw=0.6,
                     label="end of settling")
    ax_full.set_ylabel("V_out [V]"); ax_full.grid(alpha=0.3)
    ax_full.legend(loc="lower right")
    ax_full.set_title(f"Buck Bode debug — duty(t) = {D_OP} + "
                       f"{EPS_DUTY}·sin(2π·{F_TEST:.0f} Hz·t)")

    # Zoom on the measurement window.
    ax_avg.plot(t_w * 1e3, v_w, color="tab:blue", lw=0.3, alpha=0.5,
                 label="raw")
    ax_avg.plot(t_w * 1e3, v_w_avg, color="tab:red", lw=1.0,
                 label="cycle-avg")
    ax_avg.set_xlabel("time [ms]"); ax_avg.set_ylabel("V_out [V]")
    ax_avg.grid(alpha=0.3); ax_avg.legend(loc="best")
    ax_avg.set_xlim(T_SETTLE * 1e3, T_END * 1e3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "debug_buck_bode_offset.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
