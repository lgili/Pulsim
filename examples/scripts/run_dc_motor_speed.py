#!/usr/bin/env python3
"""DC motor — closed-loop speed control with PWM-driven H-bridge.

   V_DC ─┬─ Q1(HS) ─ leg ── R_a ── L_a ── E_bemf ── gnd
         │                                       (≡ motor armature)
         └─ D_FW ─ leg (freewheel during off-time)

Speed setpoint → PI(ω) → duty → PWM → switch on/off the HS leg.

The motor (`CircuitBuilder.add_dc_machine`) is solved inside the
matrix: armature R_a + L_a + back-EMF K·ω, and the shaft
J·dω/dt = K·i − B·ω − T_load as an analogue node whose voltage is ω
[rad/s]. (The retired `add_dc_motor` observer integrated ω by forward
Euler outside the matrix and is deprecated.)

Plant parameters (small geared DC motor):
  R_a = 1 Ω         L_a = 5 mH
  Kt = Ke = 0.05 V·s/rad ≈ 0.05 Nm/A
  J = 1e-4 kg·m²    B = 1e-5 Nm·s/rad
  Load: constant 0.1 Nm

What it does (measured): the duty saturates at 0.95 and ω reaches the
100 rad/s setpoint in ~15 ms, overshoots ~20 % (peak 120 rad/s at
~21 ms), and is within ±2 % by ~40 ms; steady state ω = 100.00 rad/s,
i_a = (T_load + B·ω)/K = 2.02 A. (The old text promised a ~0.5 s ramp
with < 5 % overshoot; these gains do not do that.)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_DC      = 24.0
F_PWM     = 20e3
T_PWM     = 1.0 / F_PWM
DT        = 5e-6
T_END     = 1.5

R_A       = 1.0
L_A       = 5e-3
KE        = 0.05
J_MOT     = 1.0e-4
B_MOT     = 1.0e-5
T_LOAD    = 0.1
OMEGA_REF = 100.0          # rad/s setpoint

# Speed-loop PI tuning:
#   electrical pole: ω_pe = R_a / L_a = 200 rad/s
#   mechanical pole: ω_pm = B / J = 0.1 rad/s (very slow)
#   For a tight speed loop, target ω_c ≈ 50 rad/s (8 Hz) — well
#   below the electrical pole.
KP_SPEED  = 0.3            # V / (rad/s)
KI_SPEED  = 5.0            # V / (rad·s·s)


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", V_DC)
    # HS switch (PWM) — connects V_DC to leg.
    b.add_switch("HS", "vdc", "leg", g_on=1e3, g_off=1e-9)
    # Freewheel diode for the inductive armature.
    b.add_diode("D_FW", "gnd", "leg",
                  g_on=1e3, g_off=1e-9, V_th=0.7)
    # Motor armature between `leg` and `gnd`; speed on node "omega".
    b.add_dc_machine("M", "leg", "gnd", "omega",
                     R_a=R_A, L_a=L_A, K=KE,
                     J=J_MOT, B=B_MOT, T_load=T_LOAD)
    return b


def main() -> None:
    b = build_plant()
    omega_idx = b.node_id_of("omega")
    num_switches = b.graph.num_switches
    print(f"  state_size: {b.pool.state_size(b.graph)}, "
          f"switches: {num_switches}")
    print(f"  Motor: R_a={R_A}Ω, L_a={L_A*1e3}mH, Ke=Kt={KE} V·s/rad")
    print(f"  Mechanical: J={J_MOT*1e4:.1f}e-4 kg·m², "
          f"T_load={T_LOAD} Nm")

    # --- Speed-loop chain --------------------------------------------------
    chain = p.MixedDomainBlockChain()
    chain.add(
        "pi_speed",
        p.PIController(Kp=KP_SPEED, Ki=KI_SPEED,
                          output_min=0.0, output_max=0.95),
        # Setpoint is constant OMEGA_REF; measured is the motor's
        # mechanical ω — exposed via a tiny lambda block below.
        inputs=dict(setpoint=OMEGA_REF, measured="channel:omega",
                      dt="dt"),
        output="duty",
    )
    chain.add(
        "pwm",
        p.PwmGenerator(frequency=F_PWM),
        inputs=dict(duty="channel:duty", t="time"),
        output="gate",
    )

    def combined_observer(t, x):
        # Expose the solved ω (node voltage) to the chain via a channel.
        chain.channels["omega"] = float(x[omega_idx])
        chain.update(t, x)

    chain.bind(b); chain._dt = DT

    # Combined switch_fn: chain's gate drives HS only.
    switch_fn = chain.make_pwm_switch_fn(
        "gate", num_switches=num_switches, switch_idx=0)

    print(f"  Running {T_END*1000:.0f} ms simulation ({int(T_END/DT)} steps)...")
    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=switch_fn,
        step_observer=combined_observer,
        max_event_iterations=8,
        progress=10,
    )

    times = np.asarray(res.times)
    i_arm = res.i("M.la")
    omega_trace = res.v("omega")
    v_leg = res.v("leg")

    # KPI
    # Window means are time-weighted (p.time_average): correct on any
    # grid, unlike a plain mean over the samples of a variable-step run.
    t_ss = T_END - 0.2
    print("\n  KPI (last 200 ms):")
    print(f"    ω mean        = {p.time_average(times, omega_trace, t_from=t_ss):.2f} rad/s  "
          f"(target {OMEGA_REF})")
    print(f"    ω peak (overshoot) = {omega_trace.max():.2f} rad/s "
          f"({100*(omega_trace.max()/OMEGA_REF-1):+.1f} %)")
    print(f"    i_armature mean = {p.time_average(times, i_arm, t_from=t_ss):.3f} A")
    theta = float(np.trapezoid(omega_trace, times))
    print(f"    Final ω = {omega_trace[-1]:.2f} rad/s")
    print(f"    Final θ = {theta:.2f} rad ({theta/(2*math.pi):.2f} revolutions)")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t_ms = times * 1000.0
    fig, (ax_w, ax_i, ax_v) = plt.subplots(3, 1, figsize=(11, 8),
                                                sharex=True)
    ax_w.plot(t_ms, omega_trace, "C0-", lw=0.9)
    ax_w.axhline(OMEGA_REF, color="r", ls="--", lw=0.7,
                   label=f"setpoint {OMEGA_REF} rad/s")
    ax_w.set_ylabel("ω [rad/s]"); ax_w.grid(alpha=0.3)
    ax_w.legend(loc="lower right")
    ax_w.set_title("DC motor speed control — PI(ω) → PWM")

    ax_i.plot(t_ms, i_arm, "C1-", lw=0.5)
    ax_i.set_ylabel("i_armature [A]"); ax_i.grid(alpha=0.3)

    ax_v.plot(t_ms, v_leg, "C2-", lw=0.5)
    ax_v.set_xlabel("time [ms]"); ax_v.set_ylabel("V_leg [V]")
    ax_v.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "dc_motor_speed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
