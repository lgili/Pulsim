#!/usr/bin/env python3
"""PSFB transformer core loss by Jiles-Atherton post-processing, checked.

Runs the phase-shift full bridge (``run_phase_shift_full_bridge.py``
topology), takes the transformer's MAGNETISING current, and feeds it to
``compute_bh_loop`` for three J-A material sets.

Post-processing is only meaningful when the J-A core is the core the
circuit simulated. The circuit's transformer is linear (L_p = 40 µH),
so its flux density is B = λ_p / (N·A). The script prints that B next
to the J-A B for each material and gives a loss figure only when the
two agree within a factor 1.5 and the loop area is positive.

* Magnetising current: i_m = λ_p / L_p = i_p + (M/L_p)·i_s. The primary
  (leakage-inductor) current also carries the reflected load current,
  which does not magnetise the core.
* Core: ETD49 (A_e = 211 mm², l_e = 114 mm). N = 3 turns, because
  L_p = 40 µH on this core needs μ_r = L_p·l_e/(N²·μ0·A_e) ≈ 1900 — an
  ungapped ferrite. So only the ferrite set can agree with the circuit.
* The J-A and Steinmetz sets in Pulsim are illustrative (see
  ``hysteresis._REFERENCE_MATERIALS`` and ``magnetic._CATALOG``); the
  iGSE figure from the circuit's B(t) is printed beside the J-A one as
  an independent estimate, not as a datasheet value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


# ---------- PSFB plant -----------------------------------------------------

V_BUS    = 100.0
F_PWM    = 100e3
PHASE    = 1.0      # rad of phase-shift between legs
DT       = 5e-8     # 50 ns — 200 samples per PWM cycle
T_END    = 2e-3     # 2 ms = 200 PWM cycles (settles)
L_P, L_S, K_T = 40e-6, 10e-6, 0.99


# ---------- Core geometry (ETD49) ------------------------------------------

N_PRIMARY = 3        # primary turns (see docstring)
L_M_PATH  = 0.114    # effective magnetic path length [m]
A_CORE    = 2.11e-4  # effective area [m²]
V_E       = A_CORE * L_M_PATH   # effective volume [m³]


def build_plant():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", V_BUS)
    for name, frm, to in [
        ("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
        ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd"),
    ]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)
    for name, anode, cathode in [
        ("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
        ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b"),
    ]:
        b.add_diode(name, anode, cathode, 1e3, 1e-9)
    b.add_inductor("L_leak", "mid_a", "pri_pos", 1e-6)
    b.add_transformer(
        "T1",
        p_from="pri_pos", p_to="mid_b",
        s_from="sec_pos", s_to="sec_neg",
        L_p=L_P, L_s=L_S, k=K_T,
    )
    # Bridge rectifier.
    b.add_diode("D1", "sec_pos",  "rect_pos", 1e3, 1e-9)
    b.add_diode("D2", "sec_neg",  "rect_pos", 1e3, 1e-9)
    b.add_diode("D3", "rect_neg", "sec_pos",  1e3, 1e-9)
    b.add_diode("D4", "rect_neg", "sec_neg",  1e3, 1e-9)
    # LC filter + load.
    b.add_inductor ("L_out", "rect_pos", "vout",     100e-6)
    b.add_capacitor("C_out", "vout",     "rect_neg", 47e-6)
    b.add_resistor ("R_L",   "vout",     "rect_neg", 10.0)
    return b


def main():
    b = build_plant()
    sw_fn = p.make_phase_shift_full_bridge_fn(
        switching_frequency=F_PWM,
        phase_shift=PHASE,
        leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3,
        num_switches=b.graph.num_switches,
        dead_time=100e-9,
    )
    print(f"  PSFB: V_bus={V_BUS}V, f_pwm={F_PWM/1e3:.0f}kHz, "
          f"phase_shift={PHASE:.2f} rad")
    res = p.simulate(b, t_end=T_END, dt=DT, switch_fn=sw_fn)

    times = np.asarray(res.times)
    states = np.asarray(res.states)

    def i_of(name):
        k = b.pool.branch_var_id_for_inductor(b.branch_id_of(name), b.graph)
        return states[:, k]

    M = K_T * np.sqrt(L_P * L_S)
    lam = L_P * i_of("T1.p") + M * i_of("T1.s")   # primary flux linkage
    i_m = lam / L_P
    period = 1.0 / F_PWM
    last = times >= times[-1] - period
    v_out = states[:, b.node_id_of("vout")] - states[:, b.node_id_of("rect_neg")]
    B_circ = lam / (N_PRIMARY * A_CORE)
    B_circ_pk = 0.5 * float(np.ptp(B_circ[last]))

    print(f"  P_out (last period)      = {np.mean(v_out[last] ** 2) / 10.0:.1f} W")
    print(f"  i_m peak (last period)   = {0.5 * np.ptp(i_m[last]):.3f} A")
    print(f"  circuit B peak, N = {N_PRIMARY}   = {B_circ_pk:.3f} T")
    P_igse = p.core_loss(B_circ[last], times[last], p.core_material("N87"), V_E)
    print(f"  iGSE on circuit B(t), N87 Steinmetz set: {P_igse:.2f} W")

    print(f"\n  Jiles-Atherton on i_m (last PWM period, V_e = {V_E*1e6:.1f} cm³):")
    print(f"    {'material':<14} {'B_JA peak':>10} {'B_JA/B_circ':>12}  P_loss")
    for mat in ("ferrite_n87", "si_steel_m19", "permalloy"):
        r = p.compute_bh_loop(i_m, times, params=p.reference_material(mat),
                              N_turns=N_PRIMARY, l_m=L_M_PATH, A_core=A_CORE,
                              period=period)
        B_ja_pk = 0.5 * float(np.ptp(np.asarray(r.B)[last]))
        ratio = B_ja_pk / B_circ_pk
        if r.energy_per_cycle_per_m3 <= 0:
            verdict = "none: negative loop area (J-A set outside its valid range)"
        elif not 1 / 1.5 <= ratio <= 1.5:
            verdict = "none: not the core the circuit's L_p describes"
        else:
            verdict = f"{r.avg_power_W:.2f} W"
        print(f"    {mat:<14} {B_ja_pk:>8.3f} T {ratio:>12.2f}  {verdict}")

    # Save the magnetising-current trace + the ferrite B-H for plotting.
    out_path = Path(__file__).with_name("psfb_ja_loss_trace.csv")
    r = p.compute_bh_loop(i_m, times, params=p.reference_material("ferrite_n87"),
                          N_turns=N_PRIMARY, l_m=L_M_PATH, A_core=A_CORE,
                          period=period)
    stride = max(1, len(times) // 4000)
    np.savetxt(out_path,
               np.column_stack([times, i_m, r.H, r.B, B_circ])[::stride],
               delimiter=",", header="t_s,i_m_A,H_A_per_m,B_JA_T,B_circuit_T",
               comments="")
    print(f"\n    trace + B-H → {out_path}")


if __name__ == "__main__":
    main()
