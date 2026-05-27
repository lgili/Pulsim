#!/usr/bin/env python3
"""PSFB transformer core-loss analysis with Jiles-Atherton.

Combines the existing phase-shift full-bridge converter
(``run_phase_shift_full_bridge.py`` topology) with the
post-processing JA pipeline ``core_loss_jiles_atherton`` to
quantify the transformer's core loss.

This is the **canonical magnetics workflow**:
  1. Run the switched-mode converter at fixed dt.
  2. Capture the inductor current history (the magnetising-
     branch current for the transformer's primary).
  3. Feed it to ``compute_bh_loop`` / ``core_loss_jiles_atherton``
     to extract:
        * B(t) and H(t) waveforms
        * energy per cycle = ∮ H dB (J/m³)
        * average core-loss power (W) given a core volume

The same captured waveform can be re-analysed across multiple
materials (ferrite N87 vs N97 vs 3C95) without re-simulating
the converter — the JA model is invariant once the (B, H)
history is in hand.
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


# ---------- Core geometry --------------------------------------------------

N_PRIMARY = 12       # primary turns
L_M_PATH  = 0.10     # 10 cm magnetic path length
A_CORE    = 3.5e-4   # 3.5 cm² ETD-49-class core
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
    # Leakage L + transformer. Track the leakage branch id so we can
    # pull its current straight from the result.
    leak_branch = b.graph.num_branches
    b.add_inductor("L_leak", "mid_a", "pri_pos", 1e-6)
    b.add_transformer(
        "T1",
        p_from="pri_pos", p_to="mid_b",
        s_from="sec_pos", s_to="sec_neg",
        L_p=40e-6, L_s=10e-6, k=0.99,
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
    return b, leak_branch


def main():
    b, leak_branch = build_plant()
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
    print(f"  Simulating {T_END*1e3:.1f} ms @ dt = {DT*1e9:.0f} ns "
          f"= {int(T_END/DT):,} steps")
    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=sw_fn,
        progress=True,
    )

    # Extract leakage inductor current (proxy for the magnetising
    # branch current; for a high-k transformer they are the same
    # within the leakage-vs-mutual split).
    times = np.asarray(res.times)
    states = np.asarray(res.states)
    i_idx = b.pool.branch_var_id_for_inductor(leak_branch, b.graph)
    i_L = states[:, i_idx]

    print(f"\n  Captured current trace:")
    print(f"    samples   = {len(times):,}")
    print(f"    i_peak    = {float(np.max(np.abs(i_L))):.3f} A")

    # Run JA core-loss analysis for THREE different materials —
    # the same trace works for all of them (the point of the
    # post-processing approach: one sim, many materials).
    unique_mats = ["ferrite_n87", "si_steel_m19", "permalloy"]

    period = 1.0 / F_PWM
    print(f"\n  Core-loss analysis (last full PWM period, "
          f"core volume = {V_E*1e6:.1f} cm³):")
    print(f"    {'material':<18} {'∮ H dB':>12} {'P_loss':>10}")
    print(f"    {'-'*18} {'-'*12} {'-'*10}")
    for mat in unique_mats:
        params = p.reference_material(mat)
        result = p.compute_bh_loop(
            i_L, times,
            params=params,
            N_turns=N_PRIMARY,
            l_m=L_M_PATH,
            A_core=A_CORE,
            period=period,
        )
        print(f"    {mat:<18}  "
              f"{result.energy_per_cycle_per_m3:>10.3g} J/m³  "
              f"{result.avg_power_W:>8.4f} W")

    # Save the trace + B/H series for the user to plot.
    try:
        import csv
        out_path = Path(__file__).with_name(
            "psfb_ja_loss_trace.csv")
        # Recompute B-H with default material for the CSV.
        params = p.reference_material("ferrite_n87")
        r = p.compute_bh_loop(
            i_L, times, params=params,
            N_turns=N_PRIMARY, l_m=L_M_PATH, A_core=A_CORE,
            period=period)
        stride = max(1, len(times) // 4000)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "i_L_A", "H_A_per_m", "B_T", "M_A_per_m"])
            for k in range(0, len(times), stride):
                w.writerow([times[k], i_L[k], r.H[k], r.B[k], r.M[k]])
        print(f"\n    trace + B-H → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
