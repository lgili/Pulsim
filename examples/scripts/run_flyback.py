#!/usr/bin/env python3
"""Isolated flyback converter — 48 V → ~24 V, 4:1 turns, 100 kHz / 50 % duty.

Topology:
   Vin (48 V) → T1.primary → Q1 (LS MOSFET) → gnd
   T1.secondary → D1 → vout
   vout || Cout || R_L → secondary ground (tied to gnd via 1 µΩ)

Turns ratio is L_p/L_s = 100/25 = 4, so the basic flyback equation
gives V_out = (D / (1−D)) · V_in / n = (0.5/0.5) · 48 / 2 = 24 V
(continuous conduction, ideal). Real-world losses bring it down a
bit but the topology should settle around 20–25 V.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True
F_PWM = 100e3
DUTY  = 0.50


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "flyback.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet        ("Q1", "sw", "gnd", R_on=1e-2, R_off=1e9)
    b.add_transformer(
        "T1",
        p_from="vin", p_to="sw",
        s_from="sec_anode", s_to="sec_neg",
        L_p=100e-6, L_s=25e-6, k=0.95,
    )
    b.add_diode    ("D1",   "sec_anode", "vout",    1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout",      "sec_neg", 100e-6)
    b.add_resistor ("R_L",  "vout",      "sec_neg", 5.0)
    b.add_resistor ("Rgnd", "sec_neg",   "gnd",     1.0e-6)
    return b, 0.0, 2.0e-4, 1.0e-8


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")
    print(f"  t_end/dt:       {t_end*1e6:.1f} µs / {dt*1e9:.0f} ns")

    pwm = p.make_pwm_switch_fn(
        frequency=F_PWM, duty=DUTY,
        switch_idx=0,
        num_switches=builder.graph.num_switches,
        phase=0.0,
    )

    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                      switch_fn=pwm, progress="bar")
    print(f"  samples: {res.num_steps()}")

    vout_idx = builder.node_id_of("vout")
    v_out = np.array([s[vout_idx] for s in res.states])
    k_skip = int(0.7 * res.num_steps())
    v_mean   = v_out[k_skip:].mean()
    v_ripple = float(np.ptp(v_out[k_skip:]))
    print(f"  V_out DC ≈ {v_mean:.2f} V (target ~20–25, "
          f"ripple {v_ripple:.2f} V pp)")

    # One-line plot — sw + vout stacked, tight_lw for dense PWM waveform.
    out = Path(__file__).resolve().parent / "output" / "flyback.png"
    p.scope(
        builder, res,
        signals=["sw", "vout"],
        title=f"Flyback converter — 48 V in, {F_PWM/1e3:.0f} kHz, "
              f"{DUTY*100:.0f}% duty, 4:1 turns",
        tight_lw=True,
        save=out,
    )


if __name__ == "__main__":
    main()
