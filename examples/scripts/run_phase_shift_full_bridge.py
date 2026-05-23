#!/usr/bin/env python3
"""ZVS phase-shift full-bridge isolated DC-DC converter.

100 V DC bus → 4-switch full bridge → 1 µH leakage → 2:1 transformer
→ 4-diode bridge rectifier → LC output filter → 10 Ω load.

The driver is `make_phase_shift_full_bridge_fn` (Layer 2 V9): two
diagonals (Q1+Q4 and Q2+Q3) each run at 50 % duty with a phase
offset that controls power transfer.

This script just smoke-tests the topology — true ZVS analysis needs
the leakage current waveform overlaid with the gate signals, which
is beyond the scope of a one-page runner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True
F_PWM = 100e3
PHASE_SHIFT = 1.0   # radians (will shift bridge B 1 rad relative to A)


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "phase_shift_full_bridge.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", 100.0)
    # 4 controlled switches.
    for name, frm, to in [
        ("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
        ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd"),
    ]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)
    # Body diodes.
    for name, anode, cathode in [
        ("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
        ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b"),
    ]:
        b.add_diode(name, anode, cathode, 1e3, 1e-9)
    # Leakage L + transformer.
    b.add_inductor   ("L_leak", "mid_a", "pri_pos", 1e-6)
    b.add_transformer(
        "T1",
        p_from="pri_pos", p_to="mid_b",
        s_from="sec_pos", s_to="sec_neg",
        L_p=40e-6, L_s=10e-6, k=0.99,
    )
    # Bridge rectifier (8..11).
    b.add_diode("D1", "sec_pos",  "rect_pos", 1e3, 1e-9)
    b.add_diode("D2", "sec_neg",  "rect_pos", 1e3, 1e-9)
    b.add_diode("D3", "rect_neg", "sec_pos",  1e3, 1e-9)
    b.add_diode("D4", "rect_neg", "sec_neg",  1e3, 1e-9)
    # LC output filter + load.
    b.add_inductor ("L_out", "rect_pos", "vout",     100e-6)
    b.add_capacitor("C_out", "vout",     "rect_neg", 47e-6)
    b.add_resistor ("R_L",   "vout",     "rect_neg", 10.0)
    # Secondary reference tie.
    b.add_resistor ("R_ref", "rect_neg", "gnd",      1e-6)
    return b, 0.0, 1.0e-3, 1.0e-7


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")

    # Phase-shift FB driver: two diagonals (Q1+Q4 and Q2+Q3).
    # Switch indices follow YAML order: 0=HS_A, 1=LS_A, 2=HS_B, 3=LS_B.
    ps = p.make_phase_shift_full_bridge_fn(
        switching_frequency=F_PWM,
        phase_shift=PHASE_SHIFT,
        leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3,
        num_switches=builder.graph.num_switches,
        dead_time=100e-9,
    )

    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                      switch_fn=ps)
    print(f"  samples: {res.num_steps()}")

    vout_idx = builder.node_id_of("vout")
    a_idx    = builder.node_id_of("mid_a")
    b_idx    = builder.node_id_of("mid_b")
    times = np.asarray(res.times) * 1e3
    v_out = np.array([s[vout_idx] for s in res.states])
    v_ab  = np.array([s[a_idx] - s[b_idx] for s in res.states])

    k_skip = int(0.7 * res.num_steps())
    v_mean = v_out[k_skip:].mean()
    print(f"  V_out DC ≈ {v_mean:.2f} V")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_pri, ax_out) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_pri.plot(times, v_ab, color="tab:purple", lw=0.4, alpha=0.7)
    ax_pri.set_ylabel("V_pri (mid_a − mid_b) [V]")
    ax_pri.grid(alpha=0.3)
    ax_pri.set_title(f"Phase-shift full bridge — {F_PWM/1e3:.0f} kHz, "
                      f"{PHASE_SHIFT:.2f} rad shift")

    ax_out.plot(times, v_out, color="tab:blue", lw=0.8)
    ax_out.set_xlabel("time [ms]"); ax_out.set_ylabel("V_out [V]")
    ax_out.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "phase_shift_full_bridge.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
