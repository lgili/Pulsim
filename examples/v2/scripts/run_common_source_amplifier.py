#!/usr/bin/env python3
"""Common-source MOSFET amplifier with SH1 (V13) device + V11 sine.

   V_DD ── R_D ── drain ── M1 (SH1) ── gnd
                            │
                  V_in = 3 V + 0.1·sin(2π·1 kHz·t)
                            │
                          gate

DC bias: V_drain ≈ 4.55 V, I_D ≈ 1.09 mA.
Small-signal gain Av = −g_m·R_D ≈ −10 V/V → 0.1 V_pp at gate
inverts and amplifies to ~1 V_pp at drain.

Demonstrates: V13 SH1 MOSFET + V11 SineVoltageSource + auto
Newton-refresh detection (the MOSFET makes
`pool.has_nonlinear_devices()` return True).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim.v2 as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "common_source_amplifier.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_voltage_source     ("VDD", "vdd", "gnd",   10.0)
    b.add_resistor           ("R_D", "vdd", "drain", 5000.0)
    b.add_sine_voltage_source("V_in", "gate", "gnd",
                                v_dc=3.0, v_amplitude=0.1,
                                frequency=1000.0, phase=0.0)
    b.add_mosfet_level1(
        "M1", "drain", "gnd", "gate",
        K=1.0e-3, V_T=2.0, lambda_=0.02, kappa=20.0,
    )
    return b, 0.0, 3.0e-3, 5.0e-6


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  has nonlinear:  {builder.pool.has_nonlinear_devices()}")

    # The MOSFET auto-enables Newton refresh via simulate().
    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                     start_from_dc_op=True,
                     max_newton_iterations=50)
    print(f"  samples: {res.num_steps()}")

    gate_idx  = builder.node_id_of("gate")
    drain_idx = builder.node_id_of("drain")
    times = np.asarray(res.times) * 1e3   # ms
    v_gate  = np.array([s[gate_idx]  for s in res.states])
    v_drain = np.array([s[drain_idx] for s in res.states])

    # Steady-state stats over the last 50% of samples.
    k_skip = len(v_drain) // 2
    v_drain_dc  = v_drain[k_skip:].mean()
    v_drain_pp  = np.ptp(v_drain[k_skip:])
    v_gate_pp   = np.ptp(v_gate[k_skip:])
    av = -v_drain_pp / v_gate_pp if v_gate_pp > 1e-9 else float("nan")

    print(f"  V_drain DC ≈ {v_drain_dc:.3f} V (target 4.55)")
    print(f"  V_gate  pp ≈ {v_gate_pp:.3f} V (target 0.20)")
    print(f"  V_drain pp ≈ {v_drain_pp:.3f} V (target ~2.0)")
    print(f"  Av (sim)   ≈ {av:+.2f}  V/V (target -10)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_g, ax_d) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_g.plot(times, v_gate, color="tab:blue", lw=1.0)
    ax_g.set_ylabel("V_gate [V]")
    ax_g.set_title("Common-source amplifier — small-signal AC")
    ax_g.grid(alpha=0.3)

    ax_d.plot(times, v_drain, color="tab:red", lw=1.0)
    ax_d.axhline(v_drain_dc, color="k", ls="--", lw=0.8,
                  label=f"DC bias ({v_drain_dc:.2f} V)")
    ax_d.set_xlabel("time [ms]")
    ax_d.set_ylabel("V_drain [V]")
    ax_d.legend(loc="lower right"); ax_d.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "common_source_amplifier.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
