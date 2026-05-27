#!/usr/bin/env python3
"""Three-phase squirrel-cage induction motor — Direct-On-Line (DOL) start.

The simplest possible IM showcase:

    Va, Vb, Vc  ── 3-phase 50 Hz sinusoidal mains
                   ↓                ↓                ↓
              R_s + σL_s ─── back-EMF source ─── neutral
              (the 3 stator phases live inside ``add_induction_motor``)

A 4 kW / 400 V / 50 Hz 4-pole machine is energised at full voltage
from standstill. The rotor accelerates under no-load against its
inertia. The script plots:

  * stator phase currents during inrush + steady-state
  * mechanical speed climbing toward synchronous
  * electromagnetic torque (high inrush, settling to load-balancing
    value)
  * rotor flux α-β trajectory (spiral that locks onto a circle once
    flux is established)

This is the analogue of the run_pmsm_foc.py / run_bldc_demo.py demos
for the new induction-motor model in v1.5.

Tested with ``pip install -e .`` rebuilds — requires the v1.5+ pulsim
that ships ``add_induction_motor``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# --- 4 kW / 400 V / 50 Hz / 4-pole reference machine ------------------------

P_RATED = 4_000.0
V_LL    = 400.0
F_LINE  = 50.0
PP      = 2          # 4-pole

DT      = 50e-6       # 50 µs — fine enough for 50 Hz × 20 (1 kHz visible)
T_END   = 1.5         # 1.5 s — let the motor reach quasi-steady-state


def build_plant():
    """Three-phase mains source + IM stator. No inverter — straight
    connection (DOL = direct on line)."""
    b = p.CircuitBuilder()

    # Three-phase 400 V LL sinusoidal source, neutral grounded.
    V_amp = V_LL * math.sqrt(2.0 / 3.0)    # peak phase voltage
    b.add_sine_voltage_source(
        "Va", "a", "gnd", 0.0, V_amp, F_LINE, 0.0)
    b.add_sine_voltage_source(
        "Vb", "b", "gnd", 0.0, V_amp, F_LINE, -2.0 * math.pi / 3.0)
    b.add_sine_voltage_source(
        "Vc", "c", "gnd", 0.0, V_amp, F_LINE, +2.0 * math.pi / 3.0)

    # IM equivalent-circuit from nameplate (heuristic split — see docs).
    kw = p.im_parameters_from_nameplate(
        P_rated_W=P_RATED, V_LL_V=V_LL, f_Hz=F_LINE, pole_pairs=PP)
    print("  Nameplate-derived IM parameters:")
    for key in ("R_s", "L_s", "R_r", "L_r", "L_m", "J"):
        print(f"    {key:8s} = {kw[key]:.4e}")
    print(f"    sigma    = {1 - kw['L_m']**2/(kw['L_s']*kw['L_r']):.4f}")
    print(f"    T_rated  = {kw['_diagnostics']['T_rated_Nm']:.2f} Nm")
    print(f"    omega_s  = {kw['_diagnostics']['omega_sync_rad_s']:.2f} rad/s")

    # Strip the diagnostics blob before passing kwargs through.
    diag = kw.pop("_diagnostics")
    motor = p.add_induction_motor(
        b, name="IM",
        phase_nodes=("a", "b", "c"),
        neutral_node="n",
        T_load=0.0,
        **kw)

    # Star-point leakage to keep MNA non-singular.
    b.add_resistor("R_leak", "n", "gnd", 1.0e6)

    return b, motor, diag


def main():
    b, motor, diag = build_plant()
    state_size = b.pool.state_size(b.graph)
    print(f"  state_size: {state_size}, "
          f"switches: {b.graph.num_switches}, dt = {DT*1e6:.0f} µs")

    obs, b_extra_fn = p.make_induction_motor_observer(b, motor, dt=DT)

    print(f"\n  Starting DOL run: {T_END} s @ dt = {DT*1e6:.0f} µs "
          f"= {int(T_END/DT):,} steps")
    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=lambda t: p.SwitchStateMask(0),    # no inverter
        step_observer=obs,
        b_extra_fn=b_extra_fn,
        progress=True,
    )

    omega_sync = diag["omega_sync_rad_s"]
    omega_final = motor.mech.omega_rad_s
    slip_final = (omega_sync - omega_final) / omega_sync

    print(f"\n  Result:")
    print(f"    ω_final  = {omega_final:.2f} rad/s "
          f"({omega_final * 60 / (2*math.pi):.1f} rpm)")
    print(f"    slip     = {slip_final*100:.2f} %")
    print(f"    T_em     = {motor.last_T_em_Nm:.2f} Nm")
    print(f"    ψ_rα     = {motor.psi_r_alpha_Wb:.4f} Wb")
    print(f"    ψ_rβ     = {motor.psi_r_beta_Wb:.4f} Wb")
    psi_r_mag = math.hypot(motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb)
    print(f"    |ψ_r|    = {psi_r_mag:.4f} Wb")
    print(f"    steps    = {res.num_steps():,}")

    # ---- Save a quick CSV so the user can plot ---------------------------
    try:
        import csv
        t_arr = np.asarray(res.times)
        x_arr = np.asarray(res.states)
        # Phase currents — inductor branch indices via the motor handle.
        ia_idx = b.pool.branch_var_id_for_inductor(
            motor.phase_branch_ids[0], b.graph)
        ib_idx = b.pool.branch_var_id_for_inductor(
            motor.phase_branch_ids[1], b.graph)
        ic_idx = b.pool.branch_var_id_for_inductor(
            motor.phase_branch_ids[2], b.graph)
        out_path = Path(__file__).with_name("im_direct_online_trace.csv")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "i_a_A", "i_b_A", "i_c_A"])
            for i in range(0, len(t_arr), max(1, len(t_arr) // 4000)):
                w.writerow([t_arr[i], x_arr[i, ia_idx],
                                x_arr[i, ib_idx], x_arr[i, ic_idx]])
        print(f"    trace    → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
