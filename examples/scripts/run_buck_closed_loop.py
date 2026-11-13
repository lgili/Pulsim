#!/usr/bin/env python3
"""Closed-loop buck — PI voltage regulation and a setpoint step.

Pedagogical AND stress test:

   Vin (24 V) ── Q1 (HS) ── sw ──┬── L ── vout ──┬── R_load (→ load_step!)
                                  │              │
                                  D_FW           Cout
                                  │              │
                                  gnd ───────────┴── gnd

The plant: 24 V → 12 V buck (D ≈ 0.5), 5 Ω load. A first-order
low-pass on the feedback runs every dt; the PI runs once per PWM
period (`bind_pi_to_switch`, ticks at k·T_PWM) and its output is the
duty of the period that starts at that tick.

Stress points for the kernel:
  - **Duty changes every period** → switch_fn(t) visits different
    cached PWL segments. Exercises cache lookup.
  - **Cold start from 0 V** → the duty saturates at 0.95 and V_out
    overshoots to ~19 V before the loop pulls it back (anti-windup).
  - **Setpoint step 12 → 14 V at t = 10 ms.**

What to look for (checked by the KPI printout, time-weighted):
  - V_out settles to 12.00 V before the step and 14.00 V after it,
    within ±0.05 V. The loop needs ~5 ms to settle (crossover
    391 Hz), so both windows sit at the end of a 10 ms interval:
    an earlier version measured 0.5–1 ms into a cold start and
    printed 16.47 V against its 12 V target.
  - The ~0.2 V pp slow wander is the duty quantisation of a
    fixed-step run: dt = T_PWM/100 gives duty steps of 0.01, i.e.
    0.24 V, and the integrator hunts between two of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


# =============================================================================
# Plant + controller config
# =============================================================================
V_IN     = 24.0
V_REF    = 12.0
F_PWM    = 100e3              # 100 kHz switching
T_PWM    = 1.0 / F_PWM
DT       = 1.0e-7             # 100 ns step (1000 samples/PWM cycle)
T_END    = 20.0e-3            # 20 ms total
T_STEP   = 10.0e-3            # setpoint step at t = 10 ms (12 V → 14 V)
R_LOAD   = 5.0

# Tuning analysis (averaged model, p.loop_gain-style, LPF included):
#   LC resonance:  f_LC = 1/(2π√(100µ·47µ)) ≈ 2.32 kHz, Q = R·√(C/L)
#   ≈ 3.4 at R_load = 5 Ω (+10.7 dB peak). Plant DC gain = V_in = 24.
#   Strategy: low-pass the feedback (τ = 500 µs, 318 Hz) so the PI
#   sees the LC peak attenuated, and cross over well below f_LC.
#   L(s) = (Kp + Ki/s)·G_vd(s)/(1 + sτ): crossover 391 Hz, PM 54°
#   (53° with a 10 µs sampling delay), GM 11.5 dB; PI zero at
#   Ki/(2π·Kp) = 1.19 kHz.
KP       = 0.020
KI       = 150.0
TAU_LPF  = 500.0e-6           # 318 Hz cutoff
DUTY_MIN = 0.05
DUTY_MAX = 0.95


def build_plant() -> p.CircuitBuilder:
    """Buck topology — the open-loop run_buck.py plant."""
    b = p.CircuitBuilder()
    b.add_voltage_source        ("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1",  "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode                 ("D_FW","gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor              ("L1",  "sw",  "vout", 100e-6)
    b.add_capacitor             ("Cout","vout","gnd",  47e-6)
    # Two parallel load resistors with an extra switch in series with
    # the second one — start with switch OFF (only heavy load), turn
    # ON at t_load_step to add the lighter load.
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)
    return b


def make_setpoint_with_step(t_step: float,
                              v_pre: float,
                              v_post: float):
    """Setpoint generator with a step at t_step."""
    def setpoint(t: float) -> float:
        return v_pre if t < t_step else v_post
    return setpoint


def main() -> None:
    builder = build_plant()
    vout_idx = builder.node_id_of("vout")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")
    print(f"  has_nonlinear:  {builder.pool.has_nonlinear_devices()}")

    # --- Controller ---------------------------------------------------------
    pi = p.PIController(
        Kp=KP, Ki=KI,
        output_min=DUTY_MIN, output_max=DUTY_MAX,
    )

    # Low-pass filter on feedback attenuates the LC resonance peak.
    # Runs every `dt` (composed observer below), so its bandwidth stays
    # at ~500 Hz independent of the PI's PWM-rate sample throttle.
    lpf = p.FirstOrderLowPass(tau=TAU_LPF)

    setpoint_fn = make_setpoint_with_step(
        T_STEP, v_pre=V_REF, v_post=V_REF + 2.0,
    )

    # `bind_pi_to_switch` packages the throttled PI update + PWM
    # switch_fn into one handle. We read the already-filtered LPF
    # state via `measured=` (the LPF itself runs every `dt` in the
    # composed observer below).
    loop = p.bind_pi_to_switch(
        builder,
        pi=pi,
        measured=lambda x: lpf.state,
        setpoint=setpoint_fn,
        switch="Q1",
        freq=F_PWM,
    )

    # Compose: LPF still ticks every dt (for ~500 Hz bandwidth),
    # then forward to the PI loop's throttled observer.
    def combined_observer(t: float, x) -> None:
        lpf.update(input_value=float(x[vout_idx]), dt=DT)
        loop.step_observer(t, x)

    # --- Run ---------------------------------------------------------------
    print("\n  Closed-loop buck:")
    print(f"    Kp={KP}, Ki={KI},  duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    setpoint = {V_REF} V until t={T_STEP*1e3:.1f} ms, "
          f"then {V_REF + 2.0} V")
    res = p.simulate(
        builder,
        t_end=T_END, dt=DT, t_start=0.0,
        switch_fn=loop.switch_fn,
        step_observer=combined_observer,
        start_from_dc_op=False,
        max_event_iterations=8,
    )
    print(f"  samples: {res.num_steps()}")

    # Duty / error history captured live by the loop helper. Resample
    # to the per-sample grid for the cosmetic plot.
    times = np.asarray(res.times) * 1e3   # ms
    v_out_arr = res.v("vout")

    if loop.duty_history:
        d_times = np.asarray([t for t, _ in loop.duty_history])
        d_vals  = np.asarray([d for _, d in loop.duty_history])
        e_vals  = np.asarray([e for _, e in loop.error_history])
        duty_history  = np.interp(res.times, d_times, d_vals, left=d_vals[0])
        error_history = np.interp(res.times, d_times, e_vals, left=e_vals[0])
    else:
        duty_history  = np.zeros(res.num_steps())
        error_history = np.zeros(res.num_steps())

    # --- Quick KPI ---------------------------------------------------------
    # Time-weighted window means (p.time_average): correct on any grid,
    # unlike a plain mean over the samples of a variable-step run.
    pre_step  = p.time_average(res.times, v_out_arr, T_STEP - 2e-3, T_STEP)
    post_step = p.time_average(res.times, v_out_arr, t_from=T_END - 2e-3)
    print("\n  KPI (settled windows, time-weighted):")
    print(f"    V_out @ pre-step  (8–10 ms):   mean={pre_step:.3f} V (target 12.0 ± 0.05)")
    print(f"    V_out @ post-step (18–20 ms):  mean={post_step:.3f} V (target 14.0 ± 0.05)")
    n_ticks = len(loop.duty_history)
    print(f"    PI updates: {n_ticks} (schedule t = k·T_PWM: "
          f"{int(round(T_END * F_PWM)) + 1})")
    print(f"    max V_out during transient:    {v_out_arr.max():.3f} V")
    print(f"    min V_out during transient:    {v_out_arr.min():.3f} V")

    # --- Plot --------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the closed-loop waveforms)")
        return

    fig, (ax_vout, ax_duty, ax_err) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True)

    setpoint_history = np.array([setpoint_fn(t) for t in res.times])
    ax_vout.plot(times, v_out_arr, color="tab:blue", lw=0.9, label="V_out")
    ax_vout.plot(times, setpoint_history, color="r", ls="--", lw=0.8,
                  label="setpoint")
    ax_vout.set_ylabel("V_out [V]"); ax_vout.grid(alpha=0.3)
    ax_vout.legend(loc="lower right")
    ax_vout.set_title(f"Buck closed-loop — PI(Kp={KP}, Ki={KI}), "
                       f"setpoint step at t={T_STEP*1e3:.1f} ms")

    ax_duty.plot(times, duty_history, color="tab:purple", lw=0.7)
    ax_duty.axhline(DUTY_MIN, color="k", ls=":", lw=0.5)
    ax_duty.axhline(DUTY_MAX, color="k", ls=":", lw=0.5)
    ax_duty.set_ylabel("duty [-]"); ax_duty.grid(alpha=0.3)
    ax_duty.set_ylim(0, 1)

    ax_err.plot(times, error_history, color="tab:orange", lw=0.7)
    ax_err.axhline(0, color="k", lw=0.5)
    ax_err.set_xlabel("time [ms]"); ax_err.set_ylabel("error [V]")
    ax_err.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "buck_closed_loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
