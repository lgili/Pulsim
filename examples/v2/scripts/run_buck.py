#!/usr/bin/env python3
"""End-to-end buck-converter runner — the v2 SMPS showcase.

Loads `examples/v2/buck.yaml`, drives Q1 with a 100 kHz / 50 %
PWM signal, runs `run_transient`, and prints steady-state
output-voltage statistics.

The buck topology:

    Vin (24V)
       │
       ├── Q1 (high-side MOSFET, body diode included)
       │     │
       │     └── sw ────────[D_FW]──── gnd
       │     │
       │     └── L(100µH) ── vout
       │                       │
       │                       ├── Cout(47µF) ── gnd
       │                       │
       │                       └── R_load(5Ω) ── gnd

Analytical steady-state V_out = V_in · D = 24 V · 0.5 = 12 V
(minus small R_on + IR losses).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import pulsim.v2 as p


def main() -> None:
    # Resolve YAML path relative to this script.
    script_dir = Path(__file__).resolve().parent
    yaml_path = script_dir.parent / "buck.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"missing YAML: {yaml_path}")

    print(f"Loading {yaml_path} ...")
    loaded = p.load_yaml_file(str(yaml_path))
    print(f"  num_branches = {loaded.builder.num_branches}")
    print(f"  dt = {loaded.options.dt} s")
    print(f"  t_end = {loaded.options.t_end} s")

    # Build the PWL state-space cache.
    cache = p.PwlStateSpaceCache(
        loaded.builder.graph, loaded.builder.pool)
    cache.build(loaded.options.dt)
    print("  cache built ✓")

    # PWM parameters: 100 kHz, 50 % duty cycle.
    f_sw = 100e3
    T_sw = 1.0 / f_sw
    duty = 0.5

    # How many switches does the cache enumerate? We need
    # this to construct correctly-sized SwitchStateMask
    # objects (the cache only has segments at the right
    # bit width).
    num_switches = loaded.builder.graph.num_switches

    def switch_fn(t: float) -> p.SwitchStateMask:
        phase = math.fmod(t, T_sw) / T_sw
        # Build a mask with `num_switches` bits.
        # Bit 0 = Q1 (the MOSFET); the auto-commutating
        # diodes (Q1 body diode + D_FW) are at higher
        # bits and managed by DiodeEventState at runtime.
        m = p.SwitchStateMask(num_switches)
        if phase < duty:
            m.set(0, True)   # Q1 ON
        return m

    # Run the transient.
    print("Running transient ...")
    result = p.run_transient(
        cache, loaded.builder.graph, loaded.builder.pool,
        loaded.options, switch_fn=switch_fn)
    print(f"  {result.num_steps()} samples")

    # Locate vout.
    vout_idx = loaded.builder.node_id_of("vout")

    # Steady-state stats over the last 10 % of samples.
    k_start = int(0.9 * result.num_steps())
    v_out_samples = np.array([
        result.states[k][vout_idx]
        for k in range(k_start, result.num_steps())
    ])

    v_in = 24.0   # from buck.yaml
    v_target = v_in * duty

    print()
    print("===== Steady-state V_out =====")
    print(f"  Mean:     {v_out_samples.mean():.3f} V")
    # numpy 2.0 removed `ndarray.ptp()`; use the
    # functional form.
    ripple = np.ptp(v_out_samples)
    print(f"  Ripple:   {ripple:.3f} V (p-p)")
    print(f"  Target:   {v_target:.3f} V  (= V_in · D)")
    print(f"  Error:    {abs(v_out_samples.mean() - v_target):.3f} V")

    # Optional: plot V_out if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print()
        print("(install matplotlib to see a V_out vs time "
              "plot)")
        return

    times = np.array(result.times)
    vouts = np.array([result.states[k][vout_idx]
                      for k in range(result.num_steps())])
    plt.figure(figsize=(10, 4))
    plt.plot(times * 1e3, vouts, lw=0.8)
    plt.axhline(v_target, color="r", ls="--",
                label=f"target ({v_target:.1f} V)")
    plt.xlabel("time [ms]")
    plt.ylabel("V_out [V]")
    plt.title("Buck converter — open-loop 100 kHz @ 50% duty")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = script_dir / "run_buck_output.png"
    plt.savefig(out, dpi=120)
    print(f"Plot written to: {out}")


if __name__ == "__main__":
    main()
