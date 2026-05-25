#!/usr/bin/env python3
"""Buck converter — 24 V → ~12 V at 5 Ω load, 100 kHz / 50 % PWM.

   Vin ── Q1 (HS MOSFET + body diode) ── sw ──┬── L ── vout
                                              │
                                              D_FW (sw → gnd)

Steady-state V_out ≈ V_in · D = 24 · 0.5 = 12 V (minus small
R_on + diode-drop losses).

Demonstrates the canonical SMPS pattern: switch + free-wheel
diode + LC filter + load. The PWM driver uses
`make_pwm_switch_fn` (Layer 2 V5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True
F_PWM = 100e3
DUTY  = 0.50


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "buck.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode    ("D_FW", "gnd",  "sw",   1e3, 1e-9, V_th=0.7)
    b.add_inductor ("L1",   "sw",   "vout", 100e-6)
    b.add_capacitor("Cout", "vout", "gnd",  47e-6)
    b.add_resistor ("R_L",  "vout", "gnd",  5.0)
    return b, 0.0, 5.0e-3, 1.0e-7


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")
    print(f"  t_end/dt:       {t_end*1e3:.2f} ms / {dt*1e9:.0f} ns")

    # PWM driver — Q1 (MOSFET switch) is the FIRST inserted switch (bit 0).
    # The body diode + D_FW occupy the remaining bits; they are auto-
    # commutated by the diode-event system inside run_transient.
    pwm = p.make_pwm_switch_fn(
        frequency=F_PWM, duty=DUTY,
        switch_idx=0,
        num_switches=builder.graph.num_switches,
        phase=0.0,
    )

    # `progress="bar"` shows an animated ASCII bar during the run.
    # For terse output, use `progress=True` (prints every 10%) or
    # `progress=False` (silent — default).
    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                      switch_fn=pwm, progress="bar")
    print(f"  samples: {res.num_steps()}")

    times = np.asarray(res.times) * 1e3   # ms
    v_out = res.v("vout")
    v_sw  = res.v("sw")

    # Steady-state stats over the last 10 % of samples.
    k_skip = int(0.9 * res.num_steps())
    v_mean   = v_out[k_skip:].mean()
    v_ripple = float(np.ptp(v_out[k_skip:]))
    v_target = 24.0 * DUTY
    print(f"  V_out DC ≈ {v_mean:.3f} V (target {v_target:.1f}, "
          f"ripple {v_ripple*1000:.1f} mV pp)")

    # One-liner plot via `p.scope()` — auto-detects node indices,
    # picks sensible time units, saves to PNG.
    out = Path(__file__).resolve().parent / "output" / "buck.png"
    p.scope(
        builder, res,
        signals=["sw", "vout"],
        title=f"Buck converter — {F_PWM/1e3:.0f} kHz, "
              f"{DUTY*100:.0f}% duty  (target V_out ≈ {v_target:.1f} V)",
        tight_lw=True,
        save=out,
    )


if __name__ == "__main__":
    main()
