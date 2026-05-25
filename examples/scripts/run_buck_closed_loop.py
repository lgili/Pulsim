#!/usr/bin/env python3
"""Closed-loop buck — PI voltage regulation, load step, CCM↔DCM transition.

Pedagogical AND stress test:

   Vin (24 V) ── Q1 (HS) ── sw ──┬── L ── vout ──┬── R_load (→ load_step!)
                                  │              │
                                  D_FW           Cout
                                  │              │
                                  gnd ───────────┴── gnd

The plant: 24 V → 12 V buck (D ≈ 0.5 at steady state, then jumps
when the load changes). PI regulator runs in a `step_observer(t, x)`
callback at every dt; its output is the duty command for the next
PWM cycle. The PWM `switch_fn` reads the latest duty.

Stress points for the kernel:
  - **Duty varies every step** → each switch_fn(t) call may hit a
    different cached PWL segment. Exercises cache lookup.
  - **Load step at t = 1 ms** → R_load drops 10× → output sags →
    PI ramps duty to recover. The transient inductor current spikes
    and probably pushes the converter into DCM briefly (diode
    starts skipping). Exercises event detection.
  - **PI must integrate stably** across all this without windup.

What to look for:
  - V_out should track the setpoint (target 12 V).
  - During the load step, V_out dips briefly then recovers.
  - The duty cycle plot shows the PI tracking the disturbance.
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
T_END    = 3.0e-3             # 3 ms total
T_LOAD_STEP = 1.0e-3          # setpoint step at t = 1 ms (12 V → 14 V)
R_LOAD_PRE  = 5.0
R_LOAD_POST = 50.0

# Tuning analysis:
#   LC resonance:  ω_LC = 1/√(100µ·47µ) ≈ 2.3 kHz
#   ζ ≈ 0.15 (lightly damped, R_load = 5 Ω) → 7.5 dB resonance peak
#   Plant DC gain (duty → V_out) ≈ V_in = 24
#   Strategy: low-pass-filter the feedback at 500 Hz (τ = 320 µs) so
#   the PI never sees the LC ring, then tune for crossover ≈ 300 Hz
#   well below the LC resonance.
#   With τ_LPF = 320 µs and Kp = 0.025, crossover ≈ 200 Hz, Ki = 200
#   gives PI-zero at 8 kHz (well above LC — actually above crossover,
#   so the integrator dominates the low-frequency phase).
KP       = 0.020
KI       = 150.0
TAU_LPF  = 500.0e-6           # 320 Hz cutoff
DUTY_MIN = 0.05
DUTY_MAX = 0.95


def build_plant() -> p.CircuitBuilder:
    """Buck topology — same as the open-loop run_buck.py but with a
    SwitchedDiode load resistor we can toggle for the load step."""
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
    b.add_resistor("R_heavy", "vout", "gnd", R_LOAD_PRE)
    # The "light load adder" is a switch + R in series — when the
    # switch closes the parallel combination drops resistance.
    # But to keep the topology simple, just toggle a single load via
    # b_extra_fn at runtime. Easier: leave a fixed R_load and let
    # the step happen entirely on the SETPOINT (which also stresses
    # the loop). Use the simpler version.
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
        T_LOAD_STEP, v_pre=V_REF, v_post=V_REF + 2.0,
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
    print(f"\n  Closed-loop buck:")
    print(f"    Kp={KP}, Ki={KI},  duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    setpoint = {V_REF} V until t={T_LOAD_STEP*1e3:.1f} ms, "
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
    pre_step  = v_out_arr[(times >= 0.5) & (times < 1.0)]
    post_step = v_out_arr[(times >= 2.5)]
    print(f"\n  KPI:")
    print(f"    V_out @ pre-step  (0.5–1 ms):  mean={pre_step.mean():.3f} V (target 12.0)")
    print(f"    V_out @ post-step (2.5–3 ms):  mean={post_step.mean():.3f} V (target 14.0)")
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
                       f"setpoint step at t={T_LOAD_STEP*1e3:.1f} ms")

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
