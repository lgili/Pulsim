#!/usr/bin/env python3
"""PWM chopper with V13 SH1 MOSFET driven by a V12 pulse source.

   V_in (12 V) ── R_in (50 Ω) ── drain ── M1 (SH1) ── gnd
                                    │
                                    R_load (100 Ω) ── gnd

The gate V_g is driven by a pulse source (4 µs ON / 6 µs OFF →
40 % duty at 100 kHz). When the MOSFET is OFF, the voltage divider
gives V_drain = V_in·R_load/(R_in+R_load) = 8 V. When ON, the
triode law settles at V_drain ≈ 1.45 V. Time-averaged V_drain ≈
5.4 V.

Demonstrates a "no switch_fn needed" pattern: the MOSFET is driven
intrinsically by a `PulseVoltageSource` on the gate; there are no
controlled-switch branches to schedule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "pwm_chopper_realistic_mosfet.yaml"
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
        t_start=0.0, pulse_width=4.0e-6, period=1.0e-5,
    )
    b.add_resistor     ("R_in",  "vin", "drain", 50.0)
    b.add_mosfet_level1("M1",    "drain", "gnd", "V_g",
                          K=0.01, V_T=2.0, lambda_=0.02, kappa=10.0)
    b.add_resistor     ("R_load","drain", "gnd",   100.0)
    return b, 0.0, 3.0e-5, 1.0e-7


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  has nonlinear:  {builder.pool.has_nonlinear_devices()}")

    # No switch_fn needed — gate is driven by the pulse source.
    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                      start_from_dc_op=True, max_newton_iterations=50)
    print(f"  samples: {res.num_steps()}")

    drain_idx = builder.node_id_of("drain")
    gate_idx  = builder.node_id_of("V_g")
    times = np.asarray(res.times) * 1e6   # µs
    v_drain = np.array([s[drain_idx] for s in res.states])
    v_gate  = np.array([s[gate_idx]  for s in res.states])

    print(f"  V_drain range: [{v_drain.min():.2f}, {v_drain.max():.2f}] V "
          f"(target ~[1.45, 8])")
    print(f"  V_drain mean:  {v_drain.mean():.2f} V (target ~5.4)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_g, ax_d) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_g.plot(times, v_gate, color="tab:purple", lw=1.0)
    ax_g.set_ylabel("V_gate [V]"); ax_g.grid(alpha=0.3)
    ax_g.set_title("PWM chopper — V13 SH1 MOSFET, 40 % duty @ 100 kHz")

    ax_d.plot(times, v_drain, color="tab:red", lw=1.0)
    ax_d.set_xlabel("time [µs]"); ax_d.set_ylabel("V_drain [V]")
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "pwm_chopper_realistic_mosfet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
