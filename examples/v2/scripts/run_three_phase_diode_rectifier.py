#!/usr/bin/env python3
"""3-phase 6-pulse diode bridge rectifier — 100 V phase peak → ~232 V DC.

Three V11 SineVoltageSource (120° apart at 50 Hz) feed a 6-diode
full-bridge that produces a 300 Hz-ripple DC output through a
C-R filter.

Analytical V_dc = (3·√6/π) · V_peak ≈ 2.34 · 100 ≈ 234 V (minus
~1.4 V for two diode drops at any time, so ~232 V real).
"""

from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np

import pulsim.v2 as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "three_phase_diode_rectifier.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    # 3-φ sine sources
    for name, node, phase in [
        ("V_a", "a", 0.0),
        ("V_b", "b", -2.0 * pi / 3.0),
        ("V_c", "c", -4.0 * pi / 3.0),
    ]:
        b.add_sine_voltage_source(name, node, "gnd",
                                    v_dc=0.0, v_amplitude=100.0,
                                    frequency=50.0, phase=phase)
    # Upper bridge: phase → vdc_p
    for name, anode in [("D1", "a"), ("D3", "b"), ("D5", "c")]:
        b.add_diode(name, anode, "vdc_p", 1e3, 1e-9)
    # Lower bridge: vdc_n → phase
    for name, cathode in [("D4", "a"), ("D6", "b"), ("D2", "c")]:
        b.add_diode(name, "vdc_n", cathode, 1e3, 1e-9)
    # DC bus filter + load
    b.add_capacitor("C_out",  "vdc_p", "vdc_n", 1e-3)
    b.add_resistor ("R_load", "vdc_p", "vdc_n", 100.0)
    b.add_resistor ("R_ref",  "vdc_n", "gnd",   1e6)
    return b, 0.0, 0.80, 2.0e-5


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")

    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start)
    print(f"  samples: {res.num_steps()}")

    p_idx = builder.node_id_of("vdc_p")
    n_idx = builder.node_id_of("vdc_n")
    a_idx = builder.node_id_of("a")
    b_idx = builder.node_id_of("b")
    c_idx = builder.node_id_of("c")
    times = np.asarray(res.times) * 1e3   # ms
    v_dc  = np.array([s[p_idx] - s[n_idx] for s in res.states])
    v_a   = np.array([s[a_idx] for s in res.states])
    v_b   = np.array([s[b_idx] for s in res.states])
    v_c   = np.array([s[c_idx] for s in res.states])

    k_skip = int(0.7 * res.num_steps())
    v_dc_mean = v_dc[k_skip:].mean()
    print(f"  V_dc mean ≈ {v_dc_mean:.1f} V (target ≈ 232 V)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_ph, ax_dc) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_ph.plot(times, v_a, label="V_a", lw=0.9)
    ax_ph.plot(times, v_b, label="V_b", lw=0.9)
    ax_ph.plot(times, v_c, label="V_c", lw=0.9)
    ax_ph.set_ylabel("phase V [V]"); ax_ph.grid(alpha=0.3)
    ax_ph.set_title("3-phase 6-pulse diode bridge rectifier")
    ax_ph.legend(loc="upper right", ncol=3)

    ax_dc.plot(times, v_dc, color="tab:red", lw=0.8)
    ax_dc.axhline(v_dc_mean, color="k", ls="--", lw=0.6,
                   label=f"mean ({v_dc_mean:.0f} V)")
    ax_dc.set_xlabel("time [ms]"); ax_dc.set_ylabel("V_dc [V]")
    ax_dc.legend(loc="lower right"); ax_dc.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "three_phase_diode_rectifier.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
