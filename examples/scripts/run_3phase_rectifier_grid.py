#!/usr/bin/env python3
"""Three-phase grid + 6-pulse diode rectifier — sequence-component analysis.

   3-φ grid (230 V_rms, 50 Hz)
        │
        ▼
   per-phase line impedance (R + L)
        │
        ▼
   6-diode bridge → V_dc bus → R_load
        │
        ▼
   (instantaneous power, harmonic, sequence analysis)

Demonstrates the Phase C.2 grid helpers:
  * `add_three_phase_grid` — one call → 3 sine sources + neutral.
  * `add_three_phase_line_impedance` — per-phase RL on each line.
  * `sequence_components(...)` — DFT-based phasor extraction →
    positive/negative/zero sequence at the grid frequency.
  * `instantaneous_power_3phase(...)` — Σ v·i across phases.

Expected: a balanced 6-pulse rectifier draws a slightly distorted
trapezoidal phase-current (THD ~30 %), but its FUNDAMENTAL is
balanced positive-sequence → ``|V_p| ≈ V_amp`` and ``|V_n| ≈ 0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


V_RMS    = 230.0
F_GRID   = 50.0
R_LINE   = 0.05     # 50 mΩ line resistance
L_LINE   = 50e-6    # 50 µH line inductance
R_LOAD   = 50.0     # DC load resistance
C_DC     = 470e-6   # DC bus capacitor
T_END    = 80e-3
DT       = 1e-5


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    # 1) Grid: 3 balanced sine sources, neutral floating at "n".
    p.add_three_phase_grid(
        b, V_rms=V_RMS, f_Hz=F_GRID,
        phase_nodes=("a", "b", "c"),
        neutral_node="n",
    )
    # 2) Line impedance from grid (a/b/c) to rectifier input (a1/b1/c1).
    p.add_three_phase_line_impedance(
        b, R=R_LINE, L=L_LINE,
        src=("a", "b", "c"),
        dst=("a1", "b1", "c1"),
    )
    # 3) 6-diode bridge: upper diodes anode = phase, cathode = vdc+.
    #    Lower diodes anode = vdc−, cathode = phase.
    for k, ph in enumerate(("a1", "b1", "c1")):
        b.add_diode(f"D_HS_{k}", ph, "vdc_p",
                       g_on=1e3, g_off=1e-9, V_th=0.7)
        b.add_diode(f"D_LS_{k}", "vdc_n", ph,
                       g_on=1e3, g_off=1e-9, V_th=0.7)
    # 4) DC link cap + load between vdc_p and vdc_n.
    b.add_capacitor("Cdc", "vdc_p", "vdc_n", C_DC)
    b.add_resistor("Rload", "vdc_p", "vdc_n", R_LOAD)
    # 5) Neutral grounded to fix the reference.
    b.add_resistor("R_n", "n", "gnd", 1.0e6)
    # 6) DC bus midpoint grounded (so vdc_p ≈ +V_dc/2, vdc_n ≈ −V_dc/2).
    b.add_resistor("R_dcg", "vdc_n", "gnd", 1.0e6)
    return b


def main() -> None:
    b = build_plant()
    print(f"  state_size: {b.pool.state_size(b.graph)}, "
          f"switches: {b.graph.num_switches}")
    print(f"  Running 80 ms 3-φ rectifier sim "
          f"({int(T_END/DT)} steps)…")

    res = p.simulate(b, t_end=T_END, dt=DT,
                       max_event_iterations=8)
    times = np.asarray(res.times)
    states = np.asarray(res.states)

    # Per-phase voltages (line-to-neutral) and currents (into the rectifier).
    n_idx = b.node_id_of("n")
    va = states[:, b.node_id_of("a")] - states[:, n_idx]
    vb = states[:, b.node_id_of("b")] - states[:, n_idx]
    vc = states[:, b.node_id_of("c")] - states[:, n_idx]
    # Line currents — derive from line-impedance voltage drop:
    # i_x = (V(x) − V(x1)) / R_line + ... but the inductor branch
    # current is exactly what we want.
    # Inductor branches (line impedance L on each phase). Order:
    # branches 0/1/2 = grid sources, 3/4 = Zline_R_a / Zline_L_a,
    # 5/6 = Zline_R_b / Zline_L_b, 7/8 = Zline_R_c / Zline_L_c.
    iL_a = states[:, b.pool.branch_var_id_for_inductor(4, b.graph)]
    iL_b = states[:, b.pool.branch_var_id_for_inductor(6, b.graph)]
    iL_c = states[:, b.pool.branch_var_id_for_inductor(8, b.graph)]

    # Skip the first cycle (transient).
    mask = times > 20e-3
    t_ss = times[mask]
    V0, Vp, Vn = p.sequence_components(
        va[mask], vb[mask], vc[mask],
        t=t_ss, frequency=F_GRID)
    I0, Ip, In = p.sequence_components(
        iL_a[mask], iL_b[mask], iL_c[mask],
        t=t_ss, frequency=F_GRID)
    vuf = p.voltage_unbalance_factor(
        va[mask], vb[mask], vc[mask],
        t=t_ss, frequency=F_GRID)
    cuf = p.voltage_unbalance_factor(
        iL_a[mask], iL_b[mask], iL_c[mask],
        t=t_ss, frequency=F_GRID)

    # DC bus + power.
    v_dc = states[:, b.node_id_of("vdc_p")] - \
              states[:, b.node_id_of("vdc_n")]
    p_inst = p.instantaneous_power_3phase(va, vb, vc,
                                              iL_a, iL_b, iL_c)

    print(f"\n  Sequence analysis (steady state, 50 Hz fundamental):")
    print(f"    |V_p|   = {abs(Vp):>7.1f} V  (line-to-neutral peak; "
          f"target {V_RMS*np.sqrt(2):.1f})")
    print(f"    |V_n|   = {abs(Vn):>7.2f} V")
    print(f"    VUF     = {vuf*100:>6.3f} %")
    print(f"    |I_p|   = {abs(Ip):>7.2f} A")
    print(f"    |I_n|   = {abs(In):>7.3f} A")
    print(f"    CUF     = {cuf*100:>6.2f} %  (current "
          f"unbalance — symmetric rectifier ≈ 0)")
    print(f"\n  DC bus:")
    print(f"    V_dc mean (after 40 ms): "
          f"{v_dc[times > 40e-3].mean():.1f} V  "
          f"(target ≈ V_LL_peak·1.35 ≈ "
          f"{V_RMS*np.sqrt(2)*np.sqrt(3)*1.35/np.sqrt(2):.0f})")
    print(f"    V_dc ripple (40-80 ms): "
          f"{v_dc[times > 40e-3].max() - v_dc[times > 40e-3].min():.1f} V")
    print(f"    P_grid mean (40-80 ms): {p_inst[times > 40e-3].mean():.1f} W")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t_ms = times * 1e3
    fig, (ax_v, ax_i, ax_dc) = plt.subplots(3, 1, figsize=(11, 8),
                                                  sharex=True)
    ax_v.plot(t_ms, va, "C0-", lw=0.6, label="v_a")
    ax_v.plot(t_ms, vb, "C1-", lw=0.6, label="v_b")
    ax_v.plot(t_ms, vc, "C2-", lw=0.6, label="v_c")
    ax_v.set_ylabel("phase V [V]"); ax_v.grid(alpha=0.3)
    ax_v.legend(loc="upper right")
    ax_v.set_title("3-φ grid → 6-pulse rectifier → DC load")

    ax_i.plot(t_ms, iL_a, "C0-", lw=0.6, label="i_a")
    ax_i.plot(t_ms, iL_b, "C1-", lw=0.6, label="i_b")
    ax_i.plot(t_ms, iL_c, "C2-", lw=0.6, label="i_c")
    ax_i.set_ylabel("line current [A]"); ax_i.grid(alpha=0.3)
    ax_i.legend(loc="upper right")

    ax_dc.plot(t_ms, v_dc, "C3-", lw=0.9)
    ax_dc.set_xlabel("time [ms]"); ax_dc.set_ylabel("V_dc [V]")
    ax_dc.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "3phase_rectifier_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
