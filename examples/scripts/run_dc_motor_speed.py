#!/usr/bin/env python3
"""DC motor — closed-loop speed control with PWM-driven H-bridge.

   V_DC ─┬─ Q1(HS) ─ leg ── R_a ── L_a ── E_bemf ── gnd
         │                                       (≡ motor armature)
         └─ D_FW ─ leg (freewheel during off-time)

Speed setpoint → PI(ω) → duty → PWM → switch on/off the HS leg.

The motor (`p.add_dc_motor`) carries the mechanical state (J, B,
T_load, ω, θ). At each sim step the observer:
  1. Reads i_armature from the kernel state vector.
  2. Computes T_em = Kt · i_armature.
  3. Integrates ω forward by dt with J·dω/dt = T_em − T_load − B·ω.
  4. Sets the back-EMF source value to Ke · ω for the next step.

Plant parameters (small geared DC motor):
  R_a = 1 Ω         L_a = 5 mH
  Kt = Ke = 0.05 V·s/rad ≈ 0.05 Nm/A
  J = 1e-4 kg·m²    B = 1e-5 Nm·s/rad
  Load: constant 0.1 Nm

Target: ramp speed from 0 → 100 rad/s in ~0.5 s with overshoot < 5 %.
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
    # Motor armature between `leg` and `gnd`.
    motor = p.add_dc_motor(
        b, name="M",
        armature_pos="leg", armature_neg="gnd",
        R_a=R_A, L_a=L_A, Ke=KE,
        J=J_MOT, B=B_MOT, T_load=T_LOAD,
    )
    return b, motor


def main() -> None:
    b, motor = build_plant()
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

    # Motor observer: integrates mechanical + sets back-EMF + also
    # writes the live ω onto the chain so the PI can read it.
    motor_obs, motor_b_extra = p.make_dc_motor_observer(
        b, motor, dt=DT)

    def combined_observer(t, x):
        # Run motor mechanical update first.
        motor_obs(t, x)
        # Expose ω to the chain via a channel.
        chain.channels["omega"] = motor.mech.omega_rad_s
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
        b_extra_fn=motor_b_extra,
        max_event_iterations=8,
        progress=10,
    )

    times = np.asarray(res.times)
    states = np.asarray(res.states)
    i_arm = states[:, b.pool.branch_var_id_for_inductor(
        motor.inductor_branch_id, b.graph)]
    v_leg = states[:, b.node_id_of("leg")]

    # Reconstruct ω(t) by re-running motor.mech in a clean pass.
    # (The original `motor` object holds only the FINAL state.)
    mech_replay = p.Mechanical(
        J_kgm2=J_MOT, B_Nms_per_rad=B_MOT, T_load_Nm=T_LOAD)
    omega_trace = np.zeros_like(times)
    for k in range(len(times)):
        omega_trace[k] = mech_replay.omega_rad_s
        T_em = KE * i_arm[k]   # Kt = Ke
        mech_replay.integrate(times[k], T_em, DT)

    # KPI
    mask = times >= (T_END - 0.2)
    print(f"\n  KPI (last 200 ms):")
    print(f"    ω mean        = {omega_trace[mask].mean():.2f} rad/s  "
          f"(target {OMEGA_REF})")
    print(f"    ω peak (overshoot) = {omega_trace.max():.2f} rad/s "
          f"({100*(omega_trace.max()/OMEGA_REF-1):+.1f} %)")
    print(f"    i_armature mean = {i_arm[mask].mean():.3f} A")
    print(f"    Final mechanical ω = "
          f"{motor.mech.omega_rad_s:.2f} rad/s")
    print(f"    Final θ = {motor.mech.theta_rad:.2f} rad "
          f"({motor.mech.theta_rad/(2*math.pi):.2f} revolutions)")

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
