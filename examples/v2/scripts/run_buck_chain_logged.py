#!/usr/bin/env python3
"""Closed-loop buck with FULL chain-channel logging.

Demonstrates `chain.record_channel(...)` — register any channel
(controller error, integrator state, PWM gate, filter output, …)
for per-step logging. After simulate() returns, plot any signal
directly from chain history vectors with `chain.get_channel_history`.

No more rolling your own step_observer with shared lists to capture
internal signals — the chain does it for you, in C++ when active.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim.v2 as p


V_IN     = 24.0
V_REF    = 12.0
F_PWM    = 100e3
DT       = 1e-7
T_END    = 3e-3


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                    R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 100e-6)
    b.add_capacitor("Cout", "vout", "gnd", 47e-6)
    b.add_resistor("R_load", "vout", "gnd", 5.0)
    return b


def main() -> None:
    b = build_plant()
    chain = p.MixedDomainBlockChain()

    chain.add("lpf", p.FirstOrderLowPass(tau=500e-6),
                inputs=dict(input_value="vout", dt="dt"),
                output="v_filt")
    chain.add("err", p.Subtract(),
                inputs=dict(a=V_REF, b="channel:v_filt"),
                output="error")
    chain.add("pi", p.PIController(Kp=0.020, Ki=150.0,
                                       output_min=0.05,
                                       output_max=0.95),
                inputs=dict(setpoint=V_REF,
                              measured="channel:v_filt",
                              dt="dt"),
                output="duty")
    chain.add("pwm", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:duty", t="time"),
                output="gate")

    observe = chain.make_step_observer(b, dt=DT, use_kernel=True)
    # Register the chain channels we want to plot — does NOT add any
    # step observer overhead, just allocates a buffer + flips a flag.
    n_steps = int(T_END / DT) + 1
    for name in ("v_filt", "error", "duty"):
        chain.record_channel(name, reserve_n=n_steps)

    sw_fn = chain.make_pwm_switch_fn("gate",
                                          num_switches=b.graph.num_switches,
                                          switch_idx=0)
    print(f"  Running {T_END*1e3:.0f} ms buck CL with chain "
          f"logging…")
    res = p.simulate(b, t_end=T_END, dt=DT, switch_fn=sw_fn,
                       step_observer=observe,
                       max_event_iterations=8)
    times = np.asarray(res.times)
    v_out = np.asarray(res.states)[:, b.node_id_of("vout")]

    # Pull logged signals.
    t_log = chain.get_recording_times()
    v_filt = chain.get_channel_history("v_filt")
    err = chain.get_channel_history("error")
    duty = chain.get_channel_history("duty")
    print(f"    samples: {len(t_log)}")
    print(f"    final v_out = {v_out[-1]:.3f} V")
    print(f"    final v_filt = {v_filt[-1]:.3f} V")
    print(f"    final error  = {err[-1]:+.3f} V")
    print(f"    final duty   = {duty[-1]:.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    t_ms = times * 1e3
    tlog_ms = np.asarray(t_log) * 1e3
    axes[0].plot(t_ms, v_out, "C0-", lw=0.5, label="V_out (state)")
    axes[0].plot(tlog_ms, v_filt, "C1-", lw=1.0, label="v_filt (chain)")
    axes[0].axhline(V_REF, color="r", ls="--", lw=0.7,
                       label=f"setpoint {V_REF} V")
    axes[0].set_ylabel("V [V]"); axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Buck CL — chain channels logged at every step")

    axes[1].plot(tlog_ms, err, "C2-", lw=0.7)
    axes[1].axhline(0, color="k", lw=0.4)
    axes[1].set_ylabel("error [V]"); axes[1].grid(alpha=0.3)

    axes[2].plot(tlog_ms, duty, "C3-", lw=0.7)
    axes[2].set_ylabel("duty [-]"); axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1)

    # Show how the PI integrator handles the windup — derive
    # i_pi = duty - Kp·error (the integrator-only contribution).
    Kp = 0.020
    integ = duty - Kp * err
    axes[3].plot(tlog_ms, integ, "C4-", lw=0.7,
                    label="PI integrator state (derived)")
    axes[3].set_xlabel("time [ms]"); axes[3].set_ylabel("∫ [-]")
    axes[3].grid(alpha=0.3); axes[3].legend(loc="lower right")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "buck_chain_logged.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
