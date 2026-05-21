#!/usr/bin/env python3
"""Half-wave rectifier — 60 Hz sine into one diode + load resistor.

   V_sine(60 Hz, 10 V) ── D1 ── R_load ── gnd

Expected output: V_out = max(V_in − V_F, 0). Negative half-cycles
are blocked; positive half-cycles pass minus the diode drop.

Demonstrates: V11 SineVoltageSource + automatic diode-event detection.
Toggle USE_YAML to switch between authoring styles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim.v2 as p


USE_YAML = True


def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "half_wave_rectifier.yaml"
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(
        "Vin", "n0", "gnd",
        v_dc=0.0, v_amplitude=10.0, frequency=60.0, phase=0.0,
    )
    b.add_diode   ("D1",  "n0", "n1", 1.0e3, 1.0e-9, V_th=0.0)
    b.add_resistor("R_L", "n1", "gnd", 10.0)
    return b, 0.0, 0.0333, 1.0e-4


def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  t_end/dt:       {t_end*1e3:.2f} ms / {dt*1e6:.1f} µs")

    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start)
    print(f"  samples: {res.num_steps()}")

    n0_idx = builder.node_id_of("n0")
    n1_idx = builder.node_id_of("n1")
    times = np.asarray(res.times) * 1e3   # ms
    v_in  = np.array([s[n0_idx] for s in res.states])
    v_out = np.array([s[n1_idx] for s in res.states])

    print(f"  v_in  range: [{v_in.min():+.2f}, {v_in.max():+.2f}] V "
          f"(expected ±10)")
    print(f"  v_out range: [{v_out.min():+.2f}, {v_out.max():+.2f}] V "
          f"(expected [0, +10])")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_in.plot(times, v_in, color="tab:blue", lw=1.0, label="V_in (60 Hz sine)")
    ax_in.axhline(0, color="k", lw=0.5)
    ax_in.set_ylabel("V_in [V]"); ax_in.set_title("Half-wave rectifier")
    ax_in.legend(loc="upper right"); ax_in.grid(alpha=0.3)

    ax_out.plot(times, v_out, color="tab:green", lw=1.0, label="V_out (after diode)")
    ax_out.axhline(0, color="k", lw=0.5)
    ax_out.set_xlabel("time [ms]"); ax_out.set_ylabel("V_out [V]")
    ax_out.legend(loc="upper right"); ax_out.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "half_wave_rectifier.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
