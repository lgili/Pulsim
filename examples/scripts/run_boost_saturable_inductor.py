#!/usr/bin/env python3
"""Boost converter with V17 SATURABLE inductor.

Same topology as the IGBT boost, but `L_boost` is a saturable
inductor (V17). As `i_L` approaches `I_sat = 1.5 A`, the effective
inductance collapses toward `L_residual = 1 µH`, throttling further
current growth. This is the protection mechanism real designers
must engineer around or risk catastrophic core saturation.

Compare the plot vs `run_boost_realistic_igbt.py`: the steady-state
V_out is lower because the saturating L can't deliver as much
energy per cycle, and the I_L waveform shows the characteristic
"clamped peak" shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "boost_saturable_inductor.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_pulse_voltage_source(
        "V_g_drive", "V_g", "gnd",
        v_initial=0.0, v_pulsed=10.0,
        t_start=5e-6, pulse_width=2e-6, period=1e-5,
        rise_time=2e-6, fall_time=2e-6,
    )
    b.add_saturable_inductor("L_boost", "vin", "sw",
                              L_0=100e-6, I_sat=1.5, L_residual=1e-6)
    b.add_diode      ("D_boost", "sw",  "vout", 1e3, 1e-9, V_th=0.5)
    b.add_igbt_level1("T1", "sw", "gnd", "V_g",
                       V_CE_sat=1.0, R_CE_sat=0.02,
                       V_T=5.0, kappa=15.0)
    b.add_capacitor  ("C_out",   "vout", "gnd", 100e-6)
    b.add_resistor   ("R_load",  "vout", "gnd", 20.0)
    return b, 0.0, 5e-3, 1e-7


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  has nonlinear:  {builder.pool.has_nonlinear_devices()}")

    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                      max_newton_iterations=50)
    print(f"  samples: {res.num_steps()}")

    vout_idx = builder.node_id_of("vout")
    sw_idx   = builder.node_id_of("sw")
    times = np.asarray(res.times) * 1e3
    v_out = np.array([s[vout_idx] for s in res.states])
    v_sw  = np.array([s[sw_idx]   for s in res.states])

    k_skip = int(0.7 * res.num_steps())
    v_mean = v_out[k_skip:].mean()
    print(f"  V_out DC ≈ {v_mean:.2f} V  (lower than the linear-L boost)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_sw, ax_out) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_sw.plot(times, v_sw, color="tab:purple", lw=0.4, alpha=0.7)
    ax_sw.set_ylabel("V_sw [V]"); ax_sw.grid(alpha=0.3)
    ax_sw.set_title("Boost converter — V17 saturable inductor (I_sat = 1.5 A)")

    ax_out.plot(times, v_out, color="tab:blue", lw=0.8)
    ax_out.set_xlabel("time [ms]"); ax_out.set_ylabel("V_out [V]")
    ax_out.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "boost_saturable_inductor.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
