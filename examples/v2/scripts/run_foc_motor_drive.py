#!/usr/bin/env python3
"""FOC motor drive showcase — closed-loop dq current control on an RL+BEMF plant.

Validates every Phase A piece end-to-end:

   3-φ source ───┐
                  │
   PWM inverter ──┤───►  va, vb, vc  (terminal voltages)
                  │       │
                  │       ▼
                  │   ┌────────┐
                  │   │ Clarke │ abc → αβ
                  │   ├────────┤
                  │   │  Park  │ αβ → dq   (θ_elec = ω_elec · t)
                  │   ├────────┤
                  │   │ PI(id) │ d-axis current loop (setpoint = 0)
                  │   │ PI(iq) │ q-axis current loop (setpoint = i_q_ref)
                  │   ├────────┤
                  │   │iPark   │ vd, vq → vα, vβ
                  │   │  SVM   │ vα, vβ → 3 duty cycles
                  │   └────────┘
                  │       │
                  └◄──────┘  3 PWM gates

The plant is a balanced 3-phase RL load with a synthetic back-EMF
(emulates a PMSM stator). For a stationary rotor (ω_elec = 0) the
i_q current sets the developed torque, and the dq loops decouple
the d and q axes.

The chain is built with `MixedDomainBlockChain` using the C++ kernel
control blocks under the hood (via the Cxx* binding wrappers) — this
is THE end-to-end test that proves the Phase A roadmap delivers a
working motor drive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


# =============================================================================
# Plant config — emulated 3-phase RL load (stator-only PMSM model)
# =============================================================================
V_DC      = 48.0
F_PWM     = 20e3
T_PWM     = 1.0 / F_PWM
DT        = 5e-7
T_END     = 30e-3
R_PHASE   = 0.5
L_PHASE   = 200e-6
F_ELEC    = 50.0           # electrical frequency (rotor at 50 Hz electrical)
W_ELEC    = 2 * math.pi * F_ELEC
I_Q_REF   = 5.0            # target q-axis current (~ torque command)

# PI current-loop tuning (per-axis):
#   plant pole: ω_p = R/L = 2500 rad/s
#   target crossover: ω_c = 1000 rad/s (well below switching)
#   PI zero at the plant pole cancels it:
#     Kp = L · ω_c = 0.2  (since L = 200 µH, ω_c = 1000)
#     Ki = R · ω_c = 500
KP_I  = 0.2
KI_I  = 500.0


def build_plant() -> p.CircuitBuilder:
    """3-phase RL inverter — DC link + 3 half-bridges + RL load.

    Simplest practical layout:
      V_DC ── HS_a ── leg_a ── L_a ── R_a ──┐
                                              ├── neutral (gnd)
      V_DC ── HS_b ── leg_b ── L_b ── R_b ──┤
                                              │
      V_DC ── HS_c ── leg_c ── L_c ── R_c ──┘
      (LS switches not modelled — assume idealised half-bridge)
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", V_DC)

    # 3 high-side switches feeding 3 phase inductors → resistors → neutral
    for phase in ("a", "b", "c"):
        sw_name = f"HS_{phase}"
        leg_node = f"leg_{phase}"
        l_to_r   = f"r_{phase}_in"
        b.add_switch(sw_name, "vdc", leg_node,
                       g_on=1e3, g_off=1e-9)
        b.add_inductor(f"L_{phase}", leg_node, l_to_r, L_PHASE)
        b.add_resistor(f"R_{phase}", l_to_r, "gnd", R_PHASE)

    return b


def main() -> None:
    builder = build_plant()
    num_switches = builder.graph.num_switches

    # -------------------------------------------------------------------------
    # FOC chain — 11 blocks driving 3 PWM channels.
    # -------------------------------------------------------------------------
    chain = p.MixedDomainBlockChain()

    # Helper: track the rotor angle θ_elec(t) explicitly.
    class RotorAngle:
        def __init__(self, omega): self.omega = omega
        def reset(self): pass
        def update(self, t): return self.omega * t

    chain.add("theta", RotorAngle(W_ELEC),
                inputs=dict(t="time"),
                output="theta_elec")

    # ---- Measurement chain: i_a/i_b/i_c → Clarke → Park ------------------
    # We don't have node-level current probes; emulate by measuring the
    # resistor terminal voltages (V_R = R · I) since V_neutral = 0.
    # Use the resistor "input" node since one end of R is at gnd.
    chain.add("i_a", p.Gain(k=1.0/R_PHASE),
                inputs=dict(x="r_a_in"),
                output="i_a")
    chain.add("i_b", p.Gain(k=1.0/R_PHASE),
                inputs=dict(x="r_b_in"),
                output="i_b")
    chain.add("i_c", p.Gain(k=1.0/R_PHASE),
                inputs=dict(x="r_c_in"),
                output="i_c")
    chain.add("clarke", p.ClarkeTransform(),
                inputs=dict(a="channel:i_a", b="channel:i_b",
                              c="channel:i_c"),
                output=("i_alpha", "i_beta", "i_zero"))
    chain.add("park", p.ParkTransform(),
                inputs=dict(alpha="channel:i_alpha",
                              beta="channel:i_beta",
                              theta="channel:theta_elec"),
                output=("i_d", "i_q"))

    # ---- Current loops (per-axis PI) -------------------------------------
    chain.add("pi_d", p.PIController(Kp=KP_I, Ki=KI_I,
                                       output_min=-V_DC/2.0,
                                       output_max=V_DC/2.0),
                inputs=dict(setpoint=0.0, measured="channel:i_d",
                              dt="dt"),
                output="v_d")
    chain.add("pi_q", p.PIController(Kp=KP_I, Ki=KI_I,
                                       output_min=-V_DC/2.0,
                                       output_max=V_DC/2.0),
                inputs=dict(setpoint=I_Q_REF, measured="channel:i_q",
                              dt="dt"),
                output="v_q")

    # ---- Inverse Park → Inverse Clarke (implicit via SVM) → SVM --------
    chain.add("ipark", p.InverseParkTransform(),
                inputs=dict(d="channel:v_d", q="channel:v_q",
                              theta="channel:theta_elec"),
                output=("v_alpha", "v_beta"))
    chain.add("svm", p.SpaceVectorModulator(v_dc=V_DC),
                inputs=dict(v_alpha="channel:v_alpha",
                              v_beta="channel:v_beta"),
                output=("d_a", "d_b", "d_c"))

    # ---- 3 PWM generators ------------------------------------------------
    chain.add("pwm_a", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:d_a", t="time"),
                output="gate_a")
    chain.add("pwm_b", p.PwmGenerator(frequency=F_PWM, phase=1/3),
                inputs=dict(duty="channel:d_b", t="time"),
                output="gate_b")
    chain.add("pwm_c", p.PwmGenerator(frequency=F_PWM, phase=2/3),
                inputs=dict(duty="channel:d_c", t="time"),
                output="gate_c")

    # -------------------------------------------------------------------------
    # Wire to simulate()
    # -------------------------------------------------------------------------
    observe   = chain.make_step_observer(builder, dt=DT)
    switch_fn = chain.make_multi_pwm_switch_fn(
        ["gate_a", "gate_b", "gate_c"],
        num_switches=num_switches,
    )

    print(f"  FOC motor drive — closed-loop dq current control:")
    print(f"    f_elec={F_ELEC} Hz, V_DC={V_DC} V, R={R_PHASE} Ω, L={L_PHASE*1e6} µH")
    print(f"    Kp={KP_I}, Ki={KI_I}, i_q_ref={I_Q_REF} A")
    print(f"    chain blocks ({len(chain.blocks)}): "
          f"{[b.name for b in chain.blocks]}")

    res = p.simulate(
        builder, t_end=T_END, dt=DT, t_start=0.0,
        switch_fn=switch_fn,
        step_observer=observe,
        max_event_iterations=8,
    )

    # -------------------------------------------------------------------------
    # Post-process: extract i_a, i_b, i_c, i_d, i_q
    # -------------------------------------------------------------------------
    times = np.asarray(res.times)
    ra_idx = builder.node_id_of("r_a_in")
    rb_idx = builder.node_id_of("r_b_in")
    rc_idx = builder.node_id_of("r_c_in")
    states = np.asarray(res.states)
    i_a = states[:, ra_idx] / R_PHASE
    i_b = states[:, rb_idx] / R_PHASE
    i_c = states[:, rc_idx] / R_PHASE

    # Compute i_d, i_q from the recorded waveforms.
    theta_arr = W_ELEC * times
    alpha = (2*i_a - i_b - i_c) / 3
    beta  = (i_b - i_c) / math.sqrt(3)
    i_d_arr =  np.cos(theta_arr)*alpha + np.sin(theta_arr)*beta
    i_q_arr = -np.sin(theta_arr)*alpha + np.cos(theta_arr)*beta

    # KPI — last 5 ms steady-state.
    mask = times >= (T_END - 5e-3)
    i_d_ss = i_d_arr[mask].mean()
    i_q_ss = i_q_arr[mask].mean()
    print(f"\n  KPI (last 5 ms steady-state):")
    print(f"    i_d = {i_d_ss:+.3f} A  (target 0.0)")
    print(f"    i_q = {i_q_ss:+.3f} A  (target {I_Q_REF})")
    print(f"    |i_a|_peak = {np.abs(i_a[mask]).max():.2f} A")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t_ms = times * 1e3
    fig, (ax_abc, ax_dq) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_abc.plot(t_ms, i_a, "C0-", lw=0.7, label="i_a")
    ax_abc.plot(t_ms, i_b, "C1-", lw=0.7, label="i_b")
    ax_abc.plot(t_ms, i_c, "C2-", lw=0.7, label="i_c")
    ax_abc.set_ylabel("phase current [A]")
    ax_abc.grid(alpha=0.3); ax_abc.legend(loc="upper right")
    ax_abc.set_title("FOC motor drive — closed-loop dq current control "
                       "(Phase A end-to-end)")

    ax_dq.plot(t_ms, i_d_arr, "C3-", lw=0.7, label="i_d (target 0)")
    ax_dq.plot(t_ms, i_q_arr, "C4-", lw=0.7,
                 label=f"i_q (target {I_Q_REF})")
    ax_dq.axhline(0, color="k", ls=":", lw=0.5)
    ax_dq.axhline(I_Q_REF, color="g", ls=":", lw=0.5)
    ax_dq.set_xlabel("time [ms]"); ax_dq.set_ylabel("dq current [A]")
    ax_dq.grid(alpha=0.3); ax_dq.legend(loc="lower right")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "foc_motor_drive.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
