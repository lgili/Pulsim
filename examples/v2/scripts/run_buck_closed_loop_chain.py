#!/usr/bin/env python3
"""Closed-loop buck **using the MixedDomainBlockChain**.

Same plant as `run_buck_closed_loop.py`, but the entire control law is
expressed as a chain of mixed-domain blocks:

    V_out --[FirstOrderLowPass]--> v_filt
                                     │
                       12 V       v_filt
                         └──[Sub]───┘
                              │ error
                              ▼
                           [PI]──► duty
                              ▼
                          [PWM(f=100k)]─► gate ─► switch_fn → bit 0

The chain replaces ~40 lines of hand-written observer/state code with
a 4-block declarative spec, while reaching the same KPI (V_out tracks
setpoint, recovers from setpoint step at t = 1 ms).

This validates:
  * inputs from a real circuit node (`from_node="vout"`),
  * inputs from another channel (`from_channel="error"`),
  * inputs from `time` and `dt`,
  * chain.make_pwm_switch_fn() wiring,
  * chain.make_step_observer() wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pulsim.v2 as p


# =============================================================================
# Plant + controller config (same as run_buck_closed_loop.py)
# =============================================================================
V_IN     = 24.0
V_REF    = 12.0
F_PWM    = 100e3
T_PWM    = 1.0 / F_PWM
DT       = 1.0e-7
T_END    = 3.0e-3
T_STEP   = 1.0e-3
R_LOAD   = 5.0

KP       = 0.020
KI       = 150.0
TAU_LPF  = 500.0e-6
DUTY_MIN = 0.05
DUTY_MAX = 0.95


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 100e-6)
    b.add_capacitor("Cout", "vout", "gnd", 47e-6)
    b.add_resistor("R_heavy", "vout", "gnd", R_LOAD)
    return b


class SetpointBlock:
    """Tiny step-setpoint custom block — illustrates that the chain
    accepts any class with `update(...) -> float` (not just stdlib
    blocks)."""

    def __init__(self, v_pre: float, v_post: float, t_step: float):
        self.v_pre = v_pre
        self.v_post = v_post
        self.t_step = t_step

    def reset(self) -> None:
        pass

    def update(self, t: float) -> float:
        return self.v_pre if t < self.t_step else self.v_post


def main() -> None:
    builder = build_plant()
    num_switches = builder.graph.num_switches

    # -------------------------------------------------------------------------
    # Build the chain.
    # -------------------------------------------------------------------------
    chain = p.MixedDomainBlockChain()
    chain.add("setpoint", SetpointBlock(V_REF, V_REF + 2.0, T_STEP),
                inputs=dict(t="time"),
                output="sp")
    chain.add("lpf", p.FirstOrderLowPass(tau=TAU_LPF),
                inputs=dict(input_value="vout", dt="dt"),
                output="v_filt")
    chain.add("err", p.Subtract(),
                inputs=dict(a="channel:sp", b="channel:v_filt"),
                output="error")
    chain.add("pi", p.PIController(Kp=KP, Ki=KI,
                                     output_min=DUTY_MIN,
                                     output_max=DUTY_MAX),
                inputs=dict(setpoint="channel:sp",
                              measured="channel:v_filt",
                              dt="dt"),
                output="duty")
    chain.add("pwm", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:duty", t="time"),
                output="gate")

    # -------------------------------------------------------------------------
    # Wire to simulate().
    # -------------------------------------------------------------------------
    observe   = chain.make_step_observer(builder, dt=DT)
    switch_fn = chain.make_pwm_switch_fn("gate", num_switches=num_switches,
                                            switch_idx=0)

    print(f"  Closed-loop buck via chain:")
    print(f"    Kp={KP}, Ki={KI},  duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    setpoint = {V_REF} V until t={T_STEP*1e3:.1f} ms, "
          f"then {V_REF + 2.0} V")
    print(f"    blocks in chain: {[b.name for b in chain.blocks]}")

    res = p.simulate(
        builder, t_end=T_END, dt=DT, t_start=0.0,
        switch_fn=switch_fn,
        step_observer=observe,
        max_event_iterations=8,
    )

    vout_idx = builder.node_id_of("vout")
    times = np.asarray(res.times) * 1e3
    v_out_arr = np.array([s[vout_idx] for s in res.states])

    pre_step  = v_out_arr[(times >= 0.5) & (times < 1.0)]
    post_step = v_out_arr[(times >= 2.5)]
    print(f"\n  KPI:")
    print(f"    V_out pre-step  (0.5–1 ms):  mean={pre_step.mean():.3f} V "
          f"(target {V_REF:.1f})")
    print(f"    V_out post-step (2.5–3 ms):  mean={post_step.mean():.3f} V "
          f"(target {V_REF + 2.0:.1f})")
    print(f"    max during transient:        {v_out_arr.max():.3f} V")
    print(f"    min during transient:        {v_out_arr.min():.3f} V")
    print(f"    samples:                     {res.num_steps()}")

    # -------------------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the closed-loop waveforms)")
        return

    setpoint_arr = np.where(np.asarray(res.times) < T_STEP, V_REF, V_REF + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(times, v_out_arr, color="tab:blue", lw=0.8, label="V_out")
    ax.plot(times, setpoint_arr, color="r", ls="--", lw=0.8,
              label="setpoint")
    ax.set_xlabel("time [ms]"); ax.set_ylabel("V_out [V]")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(f"Buck closed-loop via MixedDomainBlockChain "
                  f"(setpoint step at {T_STEP*1e3:.1f} ms)")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "buck_closed_loop_chain.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
