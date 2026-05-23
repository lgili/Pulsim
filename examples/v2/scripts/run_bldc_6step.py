#!/usr/bin/env python3
"""BLDC motor — 6-step (120°-block) commutation with PWM speed control.

Plant (`p.add_bldc`):
  * 3-phase R_s + L_s + trapezoidal back-EMF
  * pp = 4 pole pairs
  * Y-connected to floating neutral

Drive: 6-step block commutation driven by the ROTOR ELECTRICAL
ANGLE θ_e (idealised Hall sensors). The commutation table:

   Sector    θ_e (rad)         HS ON      LS ON      Current path
   ──────    ─────────────     ──────     ──────     ──────────────
      1     [-π/6, +π/6)        A          B         A → load → B
      2     [+π/6, +π/2)        A          C         A → load → C
      3     [+π/2, +5π/6)       B          C         B → load → C
      4     [+5π/6, +7π/6)      B          A         B → load → A
      5     [+7π/6, +3π/2)      C          A         C → load → A
      6     [+3π/2, +11π/6)     C          B         C → load → B

A single PWM modulates the active HS gate (chopper drive) to control
the average voltage. LS gates conduct full-on. This is the simplest
BLDC drive — no current control loop, just a constant duty.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


V_DC      = 24.0
F_PWM     = 20e3
T_PWM     = 1.0 / F_PWM
DT        = 1e-6
T_END     = 100e-3

R_S       = 0.3
L_S       = 100e-6
PSI_PM    = 0.005          # 5 mWb — small hobby BLDC
PP        = 4
J_MOT     = 5e-5
B_MOT     = 5e-5
T_LOAD    = 0.02
DUTY      = 0.5            # constant 50 % duty on the chopped HS gate


def build_plant():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", V_DC)
    for ph in ("a", "b", "c"):
        leg = f"leg_{ph}"
        b.add_switch(f"HS_{ph}", "vdc", leg, g_on=1e3, g_off=1e-9)
        b.add_diode(f"D_HS_{ph}", leg, "vdc",
                      g_on=1e3, g_off=1e-9, V_th=0.7)
        b.add_switch(f"LS_{ph}", leg, "gnd", g_on=1e3, g_off=1e-9)
        b.add_diode(f"D_LS_{ph}", "gnd", leg,
                      g_on=1e3, g_off=1e-9, V_th=0.7)

    motor = p.add_bldc(
        b, name="M",
        phase_nodes=("leg_a", "leg_b", "leg_c"),
        neutral_node="neutral",
        R_s=R_S, L_s=L_S, psi_pm=PSI_PM, pole_pairs=PP,
        J=J_MOT, B=B_MOT, T_load=T_LOAD,
    )
    b.add_resistor("R_leak", "neutral", "gnd", 1.0e6)
    return b, motor


def sector_from_angle(theta_e: float) -> int:
    """Return sector index 0..5 from electrical angle.

    Sector 0: θ_e ∈ [−π/6, π/6),  Sector 1: [π/6, π/2),  …
    Output range 0-5.
    """
    th = math.fmod(theta_e + math.pi / 6.0, 2.0 * math.pi)
    if th < 0:
        th += 2.0 * math.pi
    return int(th / (math.pi / 3.0))


# (HS index, LS index) per sector — kernel switch ordering:
#   0 HS_a  1 D_HS_a  2 LS_a  3 D_LS_a
#   4 HS_b  5 D_HS_b  6 LS_b  7 D_LS_b
#   8 HS_c  9 D_HS_c 10 LS_c 11 D_LS_c
# HS_x at indices [0, 4, 8], LS_x at [2, 6, 10].
#
# Sector boundaries at θ_e ∈ {30°, 90°, 150°, 210°, 270°, 330°}.
# Sector midpoints (where BEMF is at its trapezoidal peak):
#   midpoint 60°:  shape_b=+1 (peak),  shape_a=−1, shape_c=0 (silent)
#   midpoint 120°: shape_c=+1,         shape_a=−1, shape_b=0
#   midpoint 180°: shape_c=+1,         shape_b=−1, shape_a=0
#   midpoint 240°: shape_a=+1,         shape_b=−1, shape_c=0
#   midpoint 300°: shape_a=+1,         shape_c=−1, shape_b=0
#   midpoint  0°/360°: shape_b=+1,     shape_c=−1, shape_a=0
# Rule: HS the phase with peak +BEMF, LS the phase with peak −BEMF.
COMMUTATION_TABLE = [
    (4, 2),    # sector 0 (30°-90°):   HS_b + LS_a
    (8, 2),    # sector 1 (90°-150°):  HS_c + LS_a
    (8, 6),    # sector 2 (150°-210°): HS_c + LS_b
    (0, 6),    # sector 3 (210°-270°): HS_a + LS_b
    (0, 10),   # sector 4 (270°-330°): HS_a + LS_c
    (4, 10),   # sector 5 (330°-30°):  HS_b + LS_c
]


def main():
    b, motor = build_plant()
    num_switches = b.graph.num_switches

    motor_obs, motor_b_extra = p.make_bldc_observer(b, motor, dt=DT)

    def switch_fn(t):
        m = p.SwitchStateMask(num_switches)
        theta_e = motor.mech.theta_rad * PP
        sector = sector_from_angle(theta_e)
        hs_idx, ls_idx = COMMUTATION_TABLE[sector]
        # PWM chops the HS gate at DUTY.
        phase = math.fmod(t, T_PWM) / T_PWM
        if phase < DUTY:
            m.set(hs_idx, True)
        m.set(ls_idx, True)
        return m

    print(f"  BLDC: R_s={R_S}Ω, L_s={L_S*1e6}µH, "
          f"ψ_pm={PSI_PM*1e3}mWb, pp={PP}")
    print(f"  6-step commutation, duty={DUTY*100:.0f}%, "
          f"f_PWM={F_PWM/1e3:.0f}kHz")
    print(f"  Running {T_END*1e3:.0f} ms "
          f"({int(T_END/DT)} steps)...")

    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=switch_fn,
        step_observer=motor_obs,
        b_extra_fn=motor_b_extra,
        max_event_iterations=8,
        progress=10,
    )

    times = np.asarray(res.times)
    states = np.asarray(res.states)
    i_a_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[0], b.graph)
    i_b_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[1], b.graph)
    i_c_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[2], b.graph)
    i_a = states[:, i_a_idx]
    i_b = states[:, i_b_idx]
    i_c = states[:, i_c_idx]

    # Replay mechanical state for ω(t) and the back-EMF traces.
    mech_replay = p.Mechanical(
        J_kgm2=J_MOT, B_Nms_per_rad=B_MOT, T_load_Nm=T_LOAD)
    omega = np.zeros_like(times)
    theta_m = np.zeros_like(times)
    e_a_trace = np.zeros_like(times)
    for k in range(len(times)):
        omega[k] = mech_replay.omega_rad_s
        theta_m[k] = mech_replay.theta_rad
        theta_e = theta_m[k] * PP
        # Trapezoidal BEMF shape — same convention as the observer:
        # e_a = E_peak · clip(−2·sin(θ_e), −1, +1)
        E_peak = PP * PSI_PM * omega[k]
        s_a = -math.sin(theta_e)
        shape_a = max(-1.0, min(1.0, 2.0 * s_a))
        e_a_trace[k] = E_peak * shape_a
        # T_em from power balance.
        s_b = -math.sin(theta_e - 2*math.pi/3)
        s_c = -math.sin(theta_e + 2*math.pi/3)
        shape_b = max(-1.0, min(1.0, 2.0 * s_b))
        shape_c = max(-1.0, min(1.0, 2.0 * s_c))
        e_b = E_peak * shape_b
        e_c = E_peak * shape_c
        if abs(omega[k]) > 1e-6:
            T_em = (e_a_trace[k]*i_a[k] + e_b*i_b[k] +
                      e_c*i_c[k]) / omega[k]
        else:
            T_em = PP * PSI_PM * (shape_a*i_a[k] + shape_b*i_b[k] +
                                      shape_c*i_c[k])
        mech_replay.integrate(times[k], T_em, DT)

    mask = times >= (T_END - 20e-3)
    print(f"\n  KPI (last 20 ms):")
    print(f"    ω mean        = {omega[mask].mean():.2f} rad/s")
    print(f"    θ_m turns     = {theta_m[-1]/(2*math.pi):.2f}")
    print(f"    |i_a|_peak    = {np.abs(i_a[mask]).max():.2f} A")
    print(f"    final ω       = {motor.mech.omega_rad_s:.2f} rad/s")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t_ms = times * 1e3
    fig, (ax_w, ax_e, ax_i) = plt.subplots(3, 1, figsize=(11, 9),
                                                sharex=True)
    ax_w.plot(t_ms, omega, "C0-", lw=0.9)
    ax_w.set_ylabel("ω [rad/s]"); ax_w.grid(alpha=0.3)
    ax_w.set_title(f"BLDC — 6-step block commutation, duty = {DUTY*100:.0f}%")

    ax_e.plot(t_ms, e_a_trace, "C1-", lw=0.5)
    ax_e.set_ylabel("e_a (back-EMF) [V]"); ax_e.grid(alpha=0.3)

    ax_i.plot(t_ms, i_a, "C0-", lw=0.6, label="i_a")
    ax_i.plot(t_ms, i_b, "C1-", lw=0.6, label="i_b")
    ax_i.plot(t_ms, i_c, "C2-", lw=0.6, label="i_c")
    ax_i.set_xlabel("time [ms]"); ax_i.set_ylabel("phase i [A]")
    ax_i.grid(alpha=0.3); ax_i.legend(loc="upper right")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "bldc_6step.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
