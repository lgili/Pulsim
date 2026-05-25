#!/usr/bin/env python3
"""3-phase 2-level voltage-source inverter — 300 V DC bus → 3-φ AC load.

   vbus ─┬─ HS_A ─┬─ HS_B ─┬─ HS_C ─
         │  mid_a │  mid_b │  mid_c
   gnd ──┴─ LS_A ─┴─ LS_B ─┴─ LS_C ─

   mid_X → R_X → L_X → n → gnd (Y-connected RL load: 5 Ω + 5 mH/phase)

Driven by `make_three_phase_spwm_fn` (Layer 2 V8): 20 kHz carrier,
50 Hz modulating sine, modulation index = 0.85, with the 6 controlled
switches at indices 0..5 (per the YAML insertion order). Body diodes
(idx 6..11) are auto-commutated by the diode-event system.

Expected: phase currents are sinusoidal at 50 Hz with 120° between
phases, ~30 A peak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "three_phase_inverter.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", 300.0)

    # 6 controlled switches (insertion order = bit order).
    for name, frm, to in [
        ("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
        ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd"),
        ("HS_C", "vbus", "mid_c"), ("LS_C", "mid_c", "gnd"),
    ]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)

    # 6 body diodes (auto-commutated by event detection).
    for name, anode, cathode in [
        ("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
        ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b"),
        ("D_HS_C", "mid_c", "vbus"), ("D_LS_C", "gnd", "mid_c"),
    ]:
        b.add_diode(name, anode, cathode, 1e3, 1e-9)

    # Y-connected RL load per phase.
    for phase in ("a", "b", "c"):
        b.add_resistor(f"R_{phase}", f"mid_{phase}", f"r{phase}_lend", 5.0)
        b.add_inductor(f"L_{phase}", f"r{phase}_lend", "n", 5.0e-3)
    b.add_resistor("R_neutral", "n", "gnd", 1.0e-3)
    return b, 0.0, 3.0e-2, 2.0e-7


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")

    # 3-φ SPWM driver: 20 kHz carrier, 50 Hz mod, m = 0.85.
    legs = p.ThreePhaseLegIndices(
        hs_a=0, ls_a=1,
        hs_b=2, ls_b=3,
        hs_c=4, ls_c=5,
    )
    spwm = p.make_three_phase_spwm_fn(
        carrier_frequency=20e3,
        modulation_frequency=50.0,
        modulation_index=0.85,
        legs=legs,
        num_switches=builder.graph.num_switches,
        dead_time=0.0,
        modulation_phase=0.0,
        carrier_phase=0.0,
    )

    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                      switch_fn=spwm)
    print(f"  samples: {res.num_steps()}")

    # Phase-leg midpoints.
    times = np.asarray(res.times) * 1e3   # ms
    v_a = res.v("mid_a")
    v_b = res.v("mid_b")

    # Line-to-line.
    v_ab = v_a - v_b

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_mid, ax_ll) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_mid.plot(times, v_a, color="tab:blue", lw=0.3, alpha=0.6, label="V_mid_a")
    ax_mid.plot(times, v_b, color="tab:orange", lw=0.3, alpha=0.6, label="V_mid_b")
    ax_mid.set_ylabel("V_mid [V]"); ax_mid.grid(alpha=0.3)
    ax_mid.set_title("3-phase VSI — 20 kHz SPWM, m = 0.85, 50 Hz output")
    ax_mid.legend(loc="upper right")

    ax_ll.plot(times, v_ab, color="tab:red", lw=0.4)
    ax_ll.set_xlabel("time [ms]"); ax_ll.set_ylabel("V_ab [V]")
    ax_ll.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "three_phase_inverter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
