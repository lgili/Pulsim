#!/usr/bin/env python3
"""Closed-loop flyback — PI voltage regulation with isolated topology.

Topology: 48 V → transformer (4:1) → diode → output cap → load.
Primary-side MOSFET driven by software-defined PWM whose duty is
set by a PI controller comparing V_out to a setpoint.

   V_in (48 V) → T1.primary → Q1 (LS MOSFET) → gnd
   T1.secondary → D1 → vout
   vout || Cout || R_L → secondary_gnd ── R_iso → gnd

Why flyback is the hardest stress test of the three:

  - **Transformer** adds magnetizing-inductance dynamics (the trap
    companion model is exact only for linear inductance, so any
    interaction with the smooth-blend diode could drift).
  - **Hard commutation** every PWM cycle: diode flips ON then OFF
    as the secondary voltage polarity inverts. Event detection runs
    twice per cycle.
  - **Software PWM with duty change every step** → cache lookups
    span many mask combinations (Q1 ON + diode ON/OFF + body diodes
    of Q1, etc.).
  - **Galvanic isolation** numerical artifact: the secondary ground
    is tied to primary ground via a 1 µΩ link — pathological matrix
    conditioning that stresses KLU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


V_IN     = 48.0
V_REF    = 24.0
V_REF_2  = 32.0
T_STEP   = 1.5e-3
F_PWM    = 50e3
T_PWM    = 1.0 / F_PWM
DT       = 1.0e-7
T_END    = 3.0e-3

KP       = 0.002
KI       = 50.0
DUTY_MIN = 0.10
DUTY_MAX = 0.75


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_mosfet        ("Q1",  "sw",  "gnd", R_on=1e-2, R_off=1e9)
    b.add_transformer(
        "T1",
        p_from="vin", p_to="sw",
        s_from="sec_anode", s_to="sec_neg",
        L_p=200e-6, L_s=50e-6, k=0.97,
    )
    b.add_diode    ("D1",   "sec_anode", "vout",    1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout",      "sec_neg", 220e-6)
    b.add_resistor ("R_L",  "vout",      "sec_neg", 10.0)
    b.add_resistor ("Rgnd", "sec_neg",   "gnd",     1.0e-6)
    return b


def make_setpoint(t_step: float, v1: float, v2: float):
    def sp(t: float) -> float:
        return v1 if t < t_step else v2
    return sp


def main() -> None:
    builder = build_plant()
    vout_idx = builder.node_id_of("vout")
    sw_idx   = builder.node_id_of("sw")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")
    print(f"  has_nonlinear:  {builder.pool.has_nonlinear_devices()}")

    pi = p.PIController(
        Kp=KP, Ki=KI,
        output_min=DUTY_MIN, output_max=DUTY_MAX,
        integrator_state=0.40,
    )
    duty = [0.40]
    last_pi_t = [-1.0]

    setpoint_fn = make_setpoint(T_STEP, V_REF, V_REF_2)

    def observe(t: float, x) -> None:
        v_out = float(x[vout_idx])
        sp = setpoint_fn(t)
        if t - last_pi_t[0] >= T_PWM:
            dt_pi = t - last_pi_t[0] if last_pi_t[0] >= 0 else T_PWM
            duty[0] = pi.update(setpoint=sp, measured=v_out, dt=dt_pi)
            last_pi_t[0] = t

    num_switches = builder.graph.num_switches

    def switch_fn(t: float):
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(num_switches)
        if phase < duty[0]:
            m.set(0, True)
        return m

    print(f"\n  Closed-loop flyback:")
    print(f"    PI(Kp={KP}, Ki={KI}), duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    setpoint {V_REF} V → {V_REF_2} V at t = {T_STEP*1e3:.1f} ms")
    res = p.simulate(
        builder, t_end=T_END, dt=DT,
        switch_fn=switch_fn,
        step_observer=observe,
        max_event_iterations=12,
    )
    print(f"  samples: {res.num_steps()}")

    # Replay for plot.
    pi2 = p.PIController(
        Kp=KP, Ki=KI, output_min=DUTY_MIN, output_max=DUTY_MAX,
        integrator_state=0.40,
    )
    duty_history = np.zeros(res.num_steps())
    setpoint_history = np.zeros(res.num_steps())
    last_t = [-1.0]
    d_curr = [0.40]
    for k, (t, st) in enumerate(zip(res.times, res.states)):
        v_out = float(st[vout_idx])
        sp = setpoint_fn(t)
        if t - last_t[0] >= T_PWM:
            dt_pi = t - last_t[0] if last_t[0] >= 0 else T_PWM
            d_curr[0] = pi2.update(setpoint=sp, measured=v_out, dt=dt_pi)
            last_t[0] = t
        duty_history[k] = d_curr[0]
        setpoint_history[k] = sp

    times = np.asarray(res.times) * 1e3
    v_out_arr = np.array([s[vout_idx] for s in res.states])
    v_sw_arr  = np.array([s[sw_idx]   for s in res.states])

    pre  = v_out_arr[(times > 1.0) & (times < T_STEP*1e3)]
    post = v_out_arr[(times > T_STEP*1e3 + 0.5)]
    print(f"\n  KPI:")
    print(f"    V_out pre-step  ({1.0}–{T_STEP*1e3:.1f} ms): "
          f"mean = {pre.mean():.2f} V (target {V_REF:.1f})")
    print(f"    V_out post-step (>{T_STEP*1e3 + 0.5} ms): "
          f"mean = {post.mean():.2f} V (target {V_REF_2:.1f})")

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
    ax_vout.set_title(f"Flyback closed-loop — PI(Kp={KP}, Ki={KI}), "
                       f"setpoint step at t={T_STEP*1e3:.1f} ms")

    ax_sw.plot(times, v_sw_arr, color="tab:purple", lw=0.3, alpha=0.7)
    ax_sw.set_ylabel("V_sw [V]"); ax_sw.grid(alpha=0.3)

    ax_duty.plot(times, duty_history, color="tab:orange", lw=0.7)
    ax_duty.axhline(DUTY_MIN, color="k", ls=":", lw=0.5)
    ax_duty.axhline(DUTY_MAX, color="k", ls=":", lw=0.5)
    ax_duty.set_xlabel("time [ms]"); ax_duty.set_ylabel("duty [-]")
    ax_duty.grid(alpha=0.3); ax_duty.set_ylim(0, 1)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "flyback_closed_loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
