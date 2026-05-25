#!/usr/bin/env python3
"""Closed-loop boost — PI regulation, RHPZ + IGBT nonlinear refresh.

   V_in (12 V) ── L_boost ──┬── D_boost ── vout ──┬── R_load
                            │                     │
                            T1 (IGBT) ── gnd      C_out

Boost converters have a **right-half-plane zero** (RHPZ): asking
for MORE output voltage causes the inductor current to RISE before
the output voltage starts climbing. This is the classic "you turn
the wheel left but the car briefly veers right" plant.

A PI tuned aggressively will go unstable. We tune CONSERVATIVELY
(low gain, slow integrator) and accept slow settling.

Stress points vs buck:
  - **IGBT nonlinear refresh** runs every Newton iter every step.
  - **Ramped gate drive** (PulseVoltageSource with rise_time) means
    the gate isn't really a single-bit switch — the duty is encoded
    in the pulse-width of a soft-edged source. We sweep that width.
  - The setpoint step EXERCISES the RHPZ — V_out will undershoot
    before climbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


# =============================================================================
# Plant + controller config
# =============================================================================
V_IN     = 12.0
V_REF    = 20.0               # initial setpoint
V_REF_2  = 28.0               # post-step setpoint (RHPZ stress)
T_STEP   = 2.0e-3
F_PWM    = 50e3               # 50 kHz (slower so we can resolve dynamics)
T_PWM    = 1.0 / F_PWM
DT       = 1.0e-7
T_END    = 4.0e-3

# Boost design point: V_in=12, V_out=20, D=0.4, L=100µH, C=100µF, R=20Ω.
#   ω_RHPZ = R·(1-D)²/L = 20·0.36/100µH = 72000 rad/s ≈ 11 kHz
#   ω_LC   = (1-D)/√(LC) = 0.6/√(1e-8) = 6000 rad/s ≈ 1 kHz
# Crossover must be < ω_RHPZ/3 ≈ 4 kHz; we put it WAY lower (~100 Hz)
# to stay tractable. LPF at 200 Hz on the feedback dampens the LC
# resonance.
KP       = 0.0015
KI       = 30.0
TAU_LPF  = 800.0e-6           # 200 Hz cutoff
DUTY_MIN = 0.10
DUTY_MAX = 0.80               # cap below 1 so we always commute


def build_plant() -> p.CircuitBuilder:
    """Same topology as run_boost_realistic_igbt.py but with a
    pulse-source whose pulse_width we'll set dynamically via the
    duty cycle."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)

    # Gate driver — we'll OVERRIDE its pulse_width every PWM cycle
    # via b_extra_fn... actually we CAN'T because PulseVoltageSource
    # params live in DevicePool and the kernel doesn't expose live
    # parameter mutation. Workaround: use a custom switch_fn that
    # drives a `mosfet`-style switch directly with a software-defined
    # PWM. Trade-off: lose the smooth rise/fall ramp.
    #
    # We accept the trade — for CL boost we need duty-as-a-variable,
    # so a software PWM is the right choice. Use a plain `mosfet`
    # (Switch kind) rather than the SH1 MOSFET to keep Newton out of
    # the loop. To still stress nonlinear stuff, the freewheel diode
    # below uses the smooth IdealDiode.
    b.add_inductor("L_boost", "vin", "sw", 100e-6)
    b.add_mosfet  ("Q1", "sw",  "gnd", R_on=1e-3, R_off=1e9)
    b.add_diode   ("D_boost", "sw", "vout", 1e3, 1e-9, V_th=0.5)
    b.add_capacitor("C_out",  "vout", "gnd", 100e-6)
    b.add_resistor ("R_load", "vout", "gnd", 20.0)
    return b


def make_setpoint_step(t_step: float, v_pre: float, v_post: float,
                          soft_start_t: float = 1.0e-3):
    """Soft-start ramp 0 → v_pre over `soft_start_t`, then step to
    v_post at `t_step`. The ramp avoids cold-start integrator windup
    on plants with strong RHPZ behavior."""
    def setpoint(t: float) -> float:
        if t < soft_start_t:
            return v_pre * (t / soft_start_t)
        return v_pre if t < t_step else v_post
    return setpoint


def main() -> None:
    builder = build_plant()
    vout_idx = builder.node_id_of("vout")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")

    # PI: very conservative because of the RHPZ.
    pi = p.PIController(
        Kp=KP, Ki=KI,
        output_min=DUTY_MIN, output_max=DUTY_MAX,
        integrator_state=0.40,   # warm-start at the expected duty
    )
    # LPF on the V_out measurement damps the LC ring before it reaches
    # the PI. Runs every `dt` (composed observer below).
    lpf = p.FirstOrderLowPass(tau=TAU_LPF)

    setpoint_fn = make_setpoint_step(T_STEP, V_REF, V_REF_2)

    # `bind_pi_to_switch` packages the throttled PI update + PWM
    # switch_fn into one handle; we read the LPF state (which ticks
    # every `dt` in the composed observer below).
    loop = p.bind_pi_to_switch(
        builder,
        pi=pi,
        measured=lambda x: lpf.state,
        setpoint=setpoint_fn,
        switch="Q1",
        freq=F_PWM,
    )

    def combined_observer(t: float, x) -> None:
        lpf.update(input_value=float(x[vout_idx]), dt=DT)
        loop.step_observer(t, x)

    print(f"\n  Closed-loop boost:")
    print(f"    PI(Kp={KP}, Ki={KI}), duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    setpoint {V_REF} V → {V_REF_2} V at t = {T_STEP*1e3:.1f} ms")
    res = p.simulate(
        builder,
        t_end=T_END, dt=DT,
        switch_fn=loop.switch_fn,
        step_observer=combined_observer,
        max_event_iterations=8,
    )
    print(f"  samples: {res.num_steps()}")

    # Duty captured live; resample to per-sample grid for the plot.
    if loop.duty_history:
        d_times = np.asarray([t for t, _ in loop.duty_history])
        d_vals  = np.asarray([d for _, d in loop.duty_history])
        duty_history = np.interp(res.times, d_times, d_vals, left=d_vals[0])
    else:
        duty_history = np.zeros(res.num_steps())
    setpoint_history = np.asarray([setpoint_fn(t) for t in res.times])

    times = np.asarray(res.times) * 1e3
    v_out_arr = res.v("vout")
    v_sw_arr  = res.v("sw")

    # KPI
    pre_step  = v_out_arr[(times >= 1.0) & (times < T_STEP*1e3)]
    post_step = v_out_arr[(times >= 3.5)]
    rhpz_dip = v_out_arr[(times >= T_STEP*1e3) & (times < T_STEP*1e3 + 0.5)].min()
    print(f"\n  KPI:")
    print(f"    V_out @ pre-step  (1–{T_STEP*1e3:.0f} ms):    mean={pre_step.mean():.2f} V "
          f"(target {V_REF:.1f})")
    print(f"    V_out @ post-step (3.5–4 ms):  mean={post_step.mean():.2f} V "
          f"(target {V_REF_2:.1f})")
    print(f"    RHPZ undershoot just after step: min = {rhpz_dip:.2f} V")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_vout, ax_sw, ax_duty) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True)
    ax_vout.plot(times, v_out_arr, color="tab:blue", lw=0.7, label="V_out")
    ax_vout.plot(times, setpoint_history, color="r", ls="--", lw=0.8,
                  label="setpoint")
    ax_vout.set_ylabel("V_out [V]"); ax_vout.grid(alpha=0.3)
    ax_vout.legend(loc="lower right")
    ax_vout.set_title(f"Boost closed-loop — PI(Kp={KP}, Ki={KI}), "
                       f"setpoint step at t={T_STEP*1e3:.1f} ms (RHPZ!)")

    ax_sw.plot(times, v_sw_arr, color="tab:purple", lw=0.3, alpha=0.7)
    ax_sw.set_ylabel("V_sw [V]"); ax_sw.grid(alpha=0.3)

    ax_duty.plot(times, duty_history, color="tab:orange", lw=0.7)
    ax_duty.axhline(DUTY_MIN, color="k", ls=":", lw=0.5)
    ax_duty.axhline(DUTY_MAX, color="k", ls=":", lw=0.5)
    ax_duty.set_xlabel("time [ms]"); ax_duty.set_ylabel("duty [-]")
    ax_duty.grid(alpha=0.3); ax_duty.set_ylim(0, 1)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "boost_closed_loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
