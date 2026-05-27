#!/usr/bin/env python3
"""Electro-thermal co-simulation — buck with MOSFET conduction loss.

   V_DC ── HS(R_on=10mΩ) ── L ── R_load ── gnd
                 │
                 ▼
       P_loss = R_on · i² (when HS is ON)
                 │
                 ▼
       Foster Z_th network → T_j(t)

The Foster network is a 3-stage thermal model representative of a
TO-247 SiC MOSFET on a heatsink:

   stage  R_th [K/W]  τ [s]
   ─────  ──────────  ───────
       0  0.05        0.0001    (chip-to-case fast pole)
       1  0.30        0.005     (case-to-heatsink medium)
       2  0.80        0.5       (heatsink-to-ambient slow)

Total R_th(∞) = 1.15 K/W. For ~5 W average loss the steady-state
junction-temperature rise is ≈ 5.75 K above 25 °C ambient.

This example does NOT close the feedback loop (R_DS_ON does not
update with T_j) — that requires kernel-side support for live
conductance updates and is part of Phase C.1's stretch goals.
For most SiC operating points the open-loop thermal trace is the
useful design metric anyway (datasheet R_DS_ON_max governs T_j_max).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_DC      = 48.0
F_PWM     = 100e3
T_PWM     = 1.0 / F_PWM
DUTY      = 0.5
R_LOAD    = 1.0          # heavy load → significant conduction loss
R_ON_MOS  = 50e-3        # 50 mΩ on-resistance (typical for cheap Si MOSFET)
L_BUCK    = 100e-6
C_OUT     = 47e-6
T_AMB_C   = 25.0
T_END     = 1.5          # 1.5 s — capture full settling on the slow Foster pole
DT        = 5e-7         # 5 % of T_PWM — plenty for thermal-dominated dynamics

# Foster stages (SiC MOSFET in TO-247 on a small heatsink).
THERMAL_STAGES = [
    p.FosterStage(R_th_K_per_W=0.05, tau_s=1e-4),   # fast (chip)
    p.FosterStage(R_th_K_per_W=0.30, tau_s=5e-3),   # medium (case)
    p.FosterStage(R_th_K_per_W=0.80, tau_s=0.5),    # slow (heatsink)
]


def build_plant() -> p.CircuitBuilder:
    """Buck + Foster thermal network in one builder."""
    b = p.CircuitBuilder()
    # --- Electrical side ---
    b.add_voltage_source("Vin", "vin", "gnd", V_DC)
    b.add_switch("Q1", "vin", "sw", g_on=1.0/R_ON_MOS, g_off=1e-9)
    b.add_diode("D_FW", "gnd", "sw", g_on=1e3, g_off=1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", L_BUCK)
    b.add_capacitor("Cout", "vout", "gnd", C_OUT)
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)

    # --- Thermal side ---
    p.add_foster_network(
        b, THERMAL_STAGES,
        junction_node="T_j", ambient_node="T_amb",
        T_amb_C=T_AMB_C, name_prefix="Th",
    )
    return b


def main() -> None:
    b = build_plant()
    num_switches = b.graph.num_switches
    print(f"  state_size: {b.pool.state_size(b.graph)}")
    print(f"  switches:   {num_switches}")

    # Index look-ups for the post-processing step.
    sw_idx = b.node_id_of("sw")
    vin_idx = b.node_id_of("vin")
    L_branch_var = b.pool.branch_var_id_for_inductor(3, b.graph)

    # Power dissipation in the high-side switch:
    #   when HS is ON   : P = R_on · i_L²       (conduction)
    #   when HS is OFF  : P = 0                  (LS diode conducts)
    # Detect ON state by checking the leg voltage (V_sw ≈ V_in when ON).
    def power_fn(t, x):
        # Use the branch current squared.
        i_L = x[L_branch_var]
        v_sw = x[sw_idx]
        v_in = x[vin_idx]
        # HS conducts when v_sw is near v_in.
        is_on = abs(v_sw - v_in) < (V_DC * 0.5)
        if is_on:
            return R_ON_MOS * (i_L ** 2)
        return 0.0

    # PWM switch_fn for the buck.
    def switch_fn(t):
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(num_switches)
        if phase < DUTY:
            m.set(0, True)
        return m

    # Thermal observer + b_extra injection.
    thermal_obs, thermal_b_extra = p.make_thermal_observer(
        b, junction_node="T_j", power_fn=power_fn,
        ambient_node="T_amb", T_amb_C=T_AMB_C,
    )

    print(f"  Foster network: {len(THERMAL_STAGES)} stages, "
          f"R_th(∞) = {sum(s.R_th_K_per_W for s in THERMAL_STAGES):.3f} K/W")
    print(f"  Running {T_END*1000:.0f} ms electro-thermal co-sim "
          f"(dt={DT*1e9:.0f} ns) — {int(T_END/DT)} steps...")

    res = p.simulate(
        b, t_end=T_END, dt=DT, t_start=0.0,
        switch_fn=switch_fn,
        step_observer=thermal_obs,
        b_extra_fn=thermal_b_extra,
        max_event_iterations=8,
        progress=10,
    )

    times = np.asarray(res.times)
    T_j = res.v("T_j")
    v_out = res.v("vout")
    i_L = res.i("L1")

    # Re-derive instantaneous power for the plot.
    P_inst = np.where(
        np.abs(res.v("sw") - res.v("vin")) < V_DC*0.5,
        R_ON_MOS * i_L**2, 0.0,
    )
    # Average power per PWM cycle (smooth out switching).
    window = max(1, int(T_PWM / DT))
    if len(P_inst) >= window:
        kernel = np.ones(window) / window
        P_avg = np.convolve(P_inst, kernel, mode="same")
    else:
        P_avg = P_inst

    print(f"\n  KPI (last 50 ms):")
    mask = times >= (T_END - 50e-3)
    print(f"    V_out mean:  {v_out[mask].mean():.3f} V "
          f"(target {V_DC*DUTY:.1f})")
    print(f"    I_L mean:    {i_L[mask].mean():.3f} A")
    print(f"    P_loss avg:  {P_avg[mask].mean():.3f} W")
    print(f"    T_j final:   {T_j[-1]:.2f} °C")
    print(f"    T_j rise:    {T_j[-1] - T_AMB_C:.2f} K")
    R_th_total = sum(s.R_th_K_per_W for s in THERMAL_STAGES)
    print(f"    expected rise = P_avg · R_th(∞) = "
          f"{P_avg[mask].mean() * R_th_total:.2f} K")

    # Plot.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t_ms = times * 1e3
    fig, (ax_e, ax_p, ax_t) = plt.subplots(3, 1, figsize=(11, 9),
                                                sharex=True)

    ax_e.plot(t_ms, v_out, "C0-", lw=0.7)
    ax_e.set_ylabel("V_out [V]"); ax_e.grid(alpha=0.3)
    ax_e.set_title("Electro-thermal buck co-simulation")

    ax_p.plot(t_ms, P_avg, "C1-", lw=0.7, label="P_avg (cycle-mean)")
    ax_p.set_ylabel("P_loss [W]"); ax_p.grid(alpha=0.3)
    ax_p.legend(loc="best")

    ax_t.plot(t_ms, T_j, "C3-", lw=1.0)
    ax_t.axhline(T_AMB_C, color="k", ls=":", lw=0.6,
                   label=f"T_amb = {T_AMB_C}°C")
    ax_t.set_xlabel("time [ms]"); ax_t.set_ylabel("T_j [°C]")
    ax_t.grid(alpha=0.3); ax_t.legend(loc="best")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "thermal_buck.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
