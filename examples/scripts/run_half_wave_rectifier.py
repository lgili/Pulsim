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

import pulsim as p


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

    # `progress=True` prints the percentage every 10% — useful so the
    # user knows the sim is alive on long runs.
    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start,
                       progress=True)
    print(f"  samples: {res.num_steps()}")

    v_in  = res.v("n0")
    v_out = res.v("n1")
    print(f"  v_in  range: [{v_in.min():+.2f}, {v_in.max():+.2f}] V "
          f"(expected ±10)")
    print(f"  v_out range: [{v_out.min():+.2f}, {v_out.max():+.2f}] V "
          f"(expected [0, +10])")

    # One-line scope plot for the time domain.
    out_dir = Path(__file__).resolve().parent / "output"
    p.scope(
        builder, res,
        signals=["n0", "n1"],
        title="Half-wave rectifier — V_in (60 Hz sine) → V_out (after diode)",
        save=out_dir / "half_wave_rectifier.png",
    )

    # Bonus: harmonic content of V_out via `p.scope_fft()`. The
    # half-wave rectified output has rich harmonic content — DC + 60Hz
    # + 120Hz + 180Hz + ... — useful for showing the FFT helper.
    p.scope_fft(
        builder, res,
        signal="n1",
        f_max=600.0,
        f_fundamental=60.0,
        log_y=False,
        title="Half-wave rectifier — V_out spectrum",
        save=out_dir / "half_wave_rectifier_fft.png",
    )


if __name__ == "__main__":
    main()
