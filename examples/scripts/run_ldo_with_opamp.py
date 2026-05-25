#!/usr/bin/env python3
"""LDO with op-amp feedback — 12 V → 7.5 V at 50 Ω load (150 mA).

Closed-loop feedback:
   V_in ── M1 (drain) ── V_out ─┬── R1 ── fb ── R2 ── gnd
                                │
        op-amp + = V_ref         R_load (50 Ω)
        op-amp − = fb            │
        op-amp out = M1.gate     gnd

Design: V_ref = 2.5 V, R1 = 10 kΩ, R2 = 5 kΩ
        → V_out = V_ref · (R1+R2)/R2 = 7.5 V

Demonstrates: V15 op-amp ideal (VCVS with high gain) + V13 SH1 MOSFET
+ negative-feedback virtual-short. Newton refresh auto-enabled by
`simulate()` since the MOSFET is nonlinear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "ldo_with_opamp.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source("V_in",  "vin",  "gnd", 12.0)
    b.add_voltage_source("V_ref", "vref", "gnd", 2.5)
    # Op-amp: + = vref, − = fb, out = vgate. Modest gain (100) to keep
    # the cold-start Newton step bounded.
    b.add_op_amp_ideal(
        "U1",
        in_pos="vref", in_neg="fb",
        out="vgate",
        gain=100.0,
    )
    # Source-follower MOSFET: drain = vin, source = vout.
    b.add_mosfet_level1(
        "M1", "vin", "vout", "vgate",
        K=0.1, V_T=2.0, lambda_=0.02, kappa=15.0,
    )
    # Feedback divider + load.
    b.add_resistor("R1",          "vout",  "fb",    10000.0)
    b.add_resistor("R2",          "fb",    "gnd",    5000.0)
    b.add_resistor("R_load",      "vout",  "gnd",       50.0)
    # G_min bleed on vgate so the op-amp output isn't floating.
    b.add_resistor("R_gmin_gate", "vgate", "gnd",    1.0e6)
    return b, 0.0, 2.0e-3, 1.0e-5


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  has nonlinear:  {builder.pool.has_nonlinear_devices()}")

    res = p.simulate(
        builder, t_end=t_end, dt=dt, t_start=t_start,
        start_from_dc_op=True,
        max_newton_iterations=200,
        tol_newton_dx=1e-5, tol_newton_res=1e-5,
        enable_newton_line_search=True,
    )
    print(f"  samples: {res.num_steps()}")

    times = np.asarray(res.times) * 1e3   # ms
    v_out  = res.v("vout")
    v_gate = res.v("vgate")

    k_skip = int(0.7 * res.num_steps())
    v_mean = v_out[k_skip:].mean()
    print(f"  V_out DC ≈ {v_mean:.3f} V (target 7.5)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_g, ax_out) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_g.plot(times, v_gate, color="tab:purple", lw=1.0)
    ax_g.set_ylabel("V_gate [V]"); ax_g.grid(alpha=0.3)
    ax_g.set_title("LDO with op-amp feedback — 12 V → 7.5 V at 50 Ω")

    ax_out.plot(times, v_out, color="tab:blue", lw=1.0)
    ax_out.axhline(7.5, color="r", ls="--", lw=0.8, label="target (7.5 V)")
    ax_out.set_xlabel("time [ms]"); ax_out.set_ylabel("V_out [V]")
    ax_out.legend(loc="lower right"); ax_out.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "ldo_with_opamp.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
