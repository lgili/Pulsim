#!/usr/bin/env python3
"""Isolated flyback converter — 48 V → ~20 V, 2:1 turns, 100 kHz / 50 % duty.

Topology:
   Vin (48 V) → T1.primary → Q1 (LS MOSFET) → gnd
   T1.secondary (wound sec_neg → sec_anode) → D1 → vout
   vout || Cout || R_L → secondary ground (tied to gnd via 1 µΩ)
   sw → D_cl → (C_cl || R_cl) → vin   (RCD leakage clamp)

Inductance ratio L_p/L_s = 100/25 = 4 is a 2:1 turns ratio, so the
ideal flyback equation gives V_out = (D / (1−D)) · V_in / 2 = 24 V
(CCM, k = 1). With k = 0.95 the ~10 µH primary leakage dumps ~17 W
into the clamp and V_out settles at ~19.8 V; the energy balance
P_in = P_out + P_clamp printed below is the check.
"""

from __future__ import annotations

from pathlib import Path

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
        s_from="sec_neg", s_to="sec_anode",
        L_p=100e-6, L_s=25e-6, k=0.95,
    )
    b.add_diode    ("D_cl", "sw", "clamp", 1e3, 1e-9)
    b.add_capacitor("C_cl", "clamp", "vin", 1e-6)
    b.add_resistor ("R_cl", "clamp", "vin", 2e3)
    b.add_diode    ("D1",   "sec_anode", "vout",    1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout",      "sec_neg", 100e-6)
    b.add_resistor ("R_L",  "vout",      "sec_neg", 5.0)
    b.add_resistor ("Rgnd", "sec_neg",   "gnd",     1.0e-6)
    return b, 0.0, 5.0e-3, 1.0e-8


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

    # Time-weighted (p.steady_state_stats): correct on any grid, including
    # the variable-step one simulate() uses when no dt is given -- a plain
    # mean over the samples is not.
    ss = p.steady_state_stats(res, "vout", last=0.3 * (t_end - t_start))
    v_mean   = ss.mean
    v_ripple = ss.ripple_pp
    print(f"  V_out DC ≈ {v_mean:.2f} V (ideal k=1 CCM: 24 V; "
          f"ripple {v_ripple:.2f} V pp)")
    # Energy balance over the same window (secondary ground ≈ gnd).
    t0 = t_end - 0.3 * (t_end - t_start)
    v_cl = res.v("clamp") - res.v("vin")
    p_in = p.time_average(res.times, -48.0 * res.i("Vin"), t_from=t0)
    p_out = p.time_average(res.times, res.v("vout") ** 2 / 5.0, t_from=t0)
    p_cl = p.time_average(res.times, v_cl ** 2 / 2e3, t_from=t0)
    print(f"  P_in = {p_in:.1f} W, P_out = {p_out:.1f} W, "
          f"P_clamp = {p_cl:.1f} W, max V_sw = {res.v('sw').max():.0f} V")

    # One-line plot — sw + vout stacked, tight_lw for dense PWM waveform.
    out = Path(__file__).resolve().parent / "output" / "flyback.png"
    p.scope(
        builder, res,
        signals=["sw", "vout"],
        title=f"Flyback converter — 48 V in, {F_PWM/1e3:.0f} kHz, "
              f"{DUTY*100:.0f}% duty, 2:1 turns",
        tight_lw=True,
        save=out,
    )


if __name__ == "__main__":
    main()
