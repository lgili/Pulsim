#!/usr/bin/env python3
"""PMSM motor — closed-loop FOC current control with rotor back-EMF.

Plant (uses `p.add_pmsm`):
   * 3-phase RL stator (R_s = 0.5 Ω, L_s = 200 µH each phase)
   * Sinusoidal back-EMF proportional to ω_m and θ_e
   * pp = 4 (4 pole pairs, common for hobby outrunners)
   * Y-connected windings to floating neutral
   * J = 1e-5 kg·m², T_load = 0.05 Nm (small fan-style load)

Controller (uses the existing `MixedDomainBlockChain`):
   abc → Clarke → Park(θ_e) → PI(i_d, i_q) → iPark → SVM → 3 PWMs

The chain is IDENTICAL to `run_foc_motor_drive.py` — only the plant
changed (added back-EMF + mechanical state). This proves the FOC
chain is plant-agnostic: replace the passive RL load with a real
motor and the same control still works.

The key difference vs the RL-only test: now the PMSM has a SPEED
state. Setting i_q > 0 produces positive torque → motor accelerates
→ back-EMF rises → eventually i_q would shrink unless the controller
keeps applying voltage. Steady state is reached when
``T_em = T_load + B·ω``, i.e. when ω stabilises.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_DC      = 48.0
F_PWM     = 20e3
DT        = 5e-7
# 50 ms is long enough to see the dq loops lock (i_d→0, i_q→i_q_ref)
# and the motor accelerate. Past ~30 ms the back-EMF reaches V_DC/2
# and i_q can no longer track — that's voltage-saturated operation
# (the "base speed" of the motor) and a real physical limit.
T_END     = 50e-3

R_S       = 0.5
L_S       = 200e-6
PSI_PM    = 0.01           # PM flux linkage [Wb]
PP        = 4              # pole pairs
J_MOT     = 1e-5
B_MOT     = 1e-4
T_LOAD    = 0.05
I_Q_REF   = 1.5            # q-axis current setpoint — chosen so
                            # back-EMF stays below V_DC/2 over the
                            # 50 ms window (motor base-speed limit).

# Current-loop PI (per-axis): same tuning as the RL-only test, since
# the electrical plant is unchanged (R_s, L_s same magnitudes).
KP_I = 0.2
KI_I = 500.0


def build_plant():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", V_DC)
    # 3-phase 2-level VSI (HS + LS per phase + body diodes).
    for phase in ("a", "b", "c"):
        leg = f"leg_{phase}"
        b.add_switch(f"HS_{phase}", "vdc", leg, g_on=1e3, g_off=1e-9)
        b.add_diode(f"D_HS_{phase}", leg, "vdc",
                      g_on=1e3, g_off=1e-9, V_th=0.7)
        b.add_switch(f"LS_{phase}", leg, "gnd", g_on=1e3, g_off=1e-9)
        b.add_diode(f"D_LS_{phase}", "gnd", leg,
                      g_on=1e3, g_off=1e-9, V_th=0.7)

    motor = p.add_pmsm(
        b, name="M",
        phase_nodes=("leg_a", "leg_b", "leg_c"),
        neutral_node="neutral",
        R_s=R_S, L_s=L_S, psi_pm=PSI_PM, pole_pairs=PP,
        J=J_MOT, B=B_MOT, T_load=T_LOAD,
    )
    # Leakage so the MNA stays non-singular with the floating neutral.
    b.add_resistor("R_leak", "neutral", "gnd", 1.0e6)
    return b, motor


def main():
    b, motor = build_plant()
    num_switches = b.graph.num_switches
    print(f"  state_size: {b.pool.state_size(b.graph)}, "
          f"switches: {num_switches}")

    # --- FOC chain (identical to run_foc_motor_drive.py) ----------------
    chain = p.MixedDomainBlockChain()

    # Rotor-angle source — pulls θ_e from the motor's mechanical state.
    class RotorAngleFromMotor:
        def __init__(self, motor, pole_pairs):
            self.motor = motor; self.pp = pole_pairs
        def reset(self): pass
        def update(self, t):
            return self.motor.mech.theta_rad * self.pp

    chain.add("theta", p.PythonBlock(RotorAngleFromMotor(motor, PP)),
                inputs=dict(t="time"), output="theta_elec")

    # i_a, i_b, i_c → derived from inductor branch currents (already
    # in the motor's `phase_branch_ids`).
    # We'll wire the chain to read them via direct probes (Gain block
    # on the corresponding *channel*, after we manually inject the
    # branch currents at each step via a wrapper observer).
    chain.add("clarke", p.ClarkeTransform(),
                inputs=dict(a="channel:i_a", b="channel:i_b",
                              c="channel:i_c"),
                output=("i_alpha", "i_beta", "i_zero"))
    chain.add("park", p.ParkTransform(),
                inputs=dict(alpha="channel:i_alpha",
                              beta="channel:i_beta",
                              theta="channel:theta_elec"),
                output=("i_d", "i_q"))
    chain.add("pi_d", p.PIController(Kp=KP_I, Ki=KI_I,
                                       output_min=-V_DC/2,
                                       output_max=V_DC/2),
                inputs=dict(setpoint=0.0, measured="channel:i_d",
                              dt="dt"),
                output="v_d")
    chain.add("pi_q", p.PIController(Kp=KP_I, Ki=KI_I,
                                       output_min=-V_DC/2,
                                       output_max=V_DC/2),
                inputs=dict(setpoint=I_Q_REF, measured="channel:i_q",
                              dt="dt"),
                output="v_q")
    chain.add("ipark", p.InverseParkTransform(),
                inputs=dict(d="channel:v_d", q="channel:v_q",
                              theta="channel:theta_elec"),
                output=("v_alpha", "v_beta"))
    chain.add("svm", p.SpaceVectorModulator(v_dc=V_DC),
                inputs=dict(v_alpha="channel:v_alpha",
                              v_beta="channel:v_beta"),
                output=("d_a", "d_b", "d_c"))
    chain.add("pwm_a", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:d_a", t="time"),
                output="ga_hs")
    chain.add("pwm_b", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:d_b", t="time"),
                output="gb_hs")
    chain.add("pwm_c", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:d_c", t="time"),
                output="gc_hs")
    # Low side = 1 - high side (1.0 + (-1)*x is 1.0 - x exactly).
    for ph in "abc":
        chain.add(f"ls_{ph}", p.Sum(weights=(1.0, -1.0)),
                  inputs=dict(inputs=(1.0, f"channel:g{ph}_hs")),
                  output=f"g{ph}_ls")

    chain_obs = chain.make_step_observer(b, dt=DT)
    set_channel = chain._cxx_chain.set_channel

    # --- PMSM observer (mechanical update + back-EMF injection) ---------
    motor_obs, motor_b_extra = p.make_pmsm_observer(b, motor, dt=DT)

    # Probe phase currents from the motor's stored branch IDs.
    i_a_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[0], b.graph)
    i_b_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[1], b.graph)
    i_c_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[2], b.graph)

    def combined_observer(t, x):
        # Read phase currents and stash on chain channels.
        set_channel("i_a", float(x[i_a_idx]))
        set_channel("i_b", float(x[i_b_idx]))
        set_channel("i_c", float(x[i_c_idx]))
        # Run the motor's mechanical/back-EMF update.
        motor_obs(t, x)
        # Then evaluate the chain (uses fresh i_a/b/c + new θ_elec).
        chain_obs(t, x)

    # Switch order in builder: each phase adds HS, D_HS, LS, D_LS
    # → 4 switches per phase × 3 phases = 12 switches.
    switch_fn = chain.make_multi_pwm_switch_fn(
        ["ga_hs", "ga_ls", "gb_hs", "gb_ls", "gc_hs", "gc_ls"],
        num_switches=num_switches,
        switch_indices=[0, 2, 4, 6, 8, 10],
    )

    print(f"  PMSM: R_s={R_S}Ω, L_s={L_S*1e6}µH, "
          f"ψ_pm={PSI_PM*1e3:.1f}mWb, pp={PP}")
    print(f"  Mechanical: J={J_MOT*1e5:.1f}e-5 kg·m², "
          f"T_load={T_LOAD} Nm")
    print(f"  Controller: PI(i_d=0), PI(i_q={I_Q_REF}A), "
          f"Kp={KP_I}, Ki={KI_I}")
    print(f"  Running {T_END*1e3:.0f} ms ({int(T_END/DT)} steps)...")

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
    i_a = states[:, i_a_idx]
    i_b = states[:, i_b_idx]
    i_c = states[:, i_c_idx]

    # Reconstruct ω(t) by replay (similar to the DC-motor example).
    # Here we need both ω AND θ_e, so let's reconstruct via the
    # motor mechanical equations.
    mech_replay = p.Mechanical(
        J_kgm2=J_MOT, B_Nms_per_rad=B_MOT, T_load_Nm=T_LOAD)
    omega_trace = np.zeros_like(times)
    theta_m_trace = np.zeros_like(times)
    for k in range(len(times)):
        omega_trace[k] = mech_replay.omega_rad_s
        theta_m_trace[k] = mech_replay.theta_rad
        # Compute T_em from phase currents + position-aware shape.
        # Same sign convention as the live observer (Phase D.2 fix):
        # e_a ∝ −sin(θ_e + offset).
        theta_e = mech_replay.theta_rad * PP
        e_a = -math.sin(theta_e)
        e_b = -math.sin(theta_e - 2*math.pi/3)
        e_c = -math.sin(theta_e + 2*math.pi/3)
        T_em = PP * PSI_PM * (e_a*i_a[k] + e_b*i_b[k] + e_c*i_c[k])
        mech_replay.integrate(times[k], T_em, DT)

    # Reconstruct dq currents for the plot.
    theta_e = theta_m_trace * PP
    alpha = (2*i_a - i_b - i_c) / 3
    beta  = (i_b - i_c) / math.sqrt(3)
    i_d_trace =  np.cos(theta_e)*alpha + np.sin(theta_e)*beta
    i_q_trace = -np.sin(theta_e)*alpha + np.cos(theta_e)*beta

    # Look at the 15-25 ms window: post-startup, pre-saturation.
    # Window means are time-weighted (p.time_average): correct on any
    # grid, unlike a plain mean over the samples of a variable-step run.
    mask = (times >= 15e-3) & (times <= 25e-3)
    win = (15e-3, 25e-3)
    print("\n  KPI (15-25 ms — post-startup, pre-voltage-saturation):")
    print(f"    ω        = {p.time_average(times, omega_trace, *win):.2f} rad/s")
    print(f"    θ_m turns= {theta_m_trace[-1]/(2*math.pi):.2f}")
    print(f"    i_d mean = {p.time_average(times, i_d_trace, *win):+.3f} A (target 0)")
    print(f"    i_q mean = {p.time_average(times, i_q_trace, *win):+.3f} A "
          f"(setpoint {I_Q_REF})")
    # The motor is still accelerating, so the q-axis back-EMF
    # E_q = pp·ψ·ω is a ramp; a PI (type-1 loop) tracks a ramp
    # disturbance with a steady error ≈ (dE_q/dt)/Ki. That accounts for
    # ~0.18 A of the ~0.23 A shortfall at these gains (Ki = 2000 cuts
    # the shortfall to 0.07 A); the rest is not yet explained.
    k0, k1 = np.searchsorted(times, win)
    alpha_m = (omega_trace[k1] - omega_trace[k0]) / (times[k1] - times[k0])
    print(f"    ramp-tracking lag ≈ pp·ψ·(dω/dt)/Ki = "
          f"{PP * PSI_PM * alpha_m / KI_I:.3f} A "
          f"(dω/dt = {alpha_m:.0f} rad/s²)")
    print(f"    |i_a|_peak = {np.abs(i_a[mask]).max():.2f} A")
    i_q_win = p.time_average(times, i_q_trace, *win)
    T_em_avg = PP * PSI_PM * 1.5 * i_q_win  # 1.5·pp·ψ·i_q (measured i_q)
    print(f"    T_em = (3/2)·pp·ψ_pm·i_q "
          f"= {T_em_avg:.3f} Nm  (load = {T_LOAD}; J·dω/dt = "
          f"{J_MOT * alpha_m:.3f} Nm)")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t_ms = times * 1e3
    fig, (ax_w, ax_dq, ax_abc) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True)
    ax_w.plot(t_ms, omega_trace, "C0-", lw=0.9)
    ax_w.set_ylabel("ω [rad/s]"); ax_w.grid(alpha=0.3)
    ax_w.set_title(f"PMSM FOC — i_q_ref = {I_Q_REF} A, T_load = "
                       f"{T_LOAD} Nm, pp = {PP}")

    ax_dq.plot(t_ms, i_d_trace, "C3-", lw=0.6,
                 label="i_d (target 0)")
    ax_dq.plot(t_ms, i_q_trace, "C4-", lw=0.6,
                 label=f"i_q (target {I_Q_REF})")
    ax_dq.set_ylabel("dq current [A]"); ax_dq.grid(alpha=0.3)
    ax_dq.legend(loc="lower right")
    ax_dq.axhline(0, color="k", ls=":", lw=0.4)
    ax_dq.axhline(I_Q_REF, color="g", ls=":", lw=0.4)

    ax_abc.plot(t_ms, i_a, "C0-", lw=0.5, label="i_a")
    ax_abc.plot(t_ms, i_b, "C1-", lw=0.5, label="i_b")
    ax_abc.plot(t_ms, i_c, "C2-", lw=0.5, label="i_c")
    ax_abc.set_xlabel("time [ms]"); ax_abc.set_ylabel("phase i [A]")
    ax_abc.grid(alpha=0.3); ax_abc.legend(loc="upper right")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "pmsm_foc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
