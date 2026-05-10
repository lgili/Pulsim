"""AC sweep — RC low-pass Bode plot.

Demonstrates the simplest small-signal analysis: linearize an RC filter at
its DC operating point, sweep frequency from 1 Hz to 1 MHz, and plot the
Bode magnitude / phase. The analytical corner sits at
    f_corner = 1 / (2π·R·C) = 159.155 Hz
where the gain crosses -3 dB and the phase crosses -45°. The script prints
the measured values at the corner so you can sanity-check the run.

Run::

    python 01_ac_sweep_rc.py
    # or, if matplotlib is missing:
    PULSIM_EXAMPLE_NOPLOT=1 python 01_ac_sweep_rc.py

See also: docs/ac-analysis.md
"""

from __future__ import annotations

import math
import os

import pulsim


def build_rc(R: float = 1e3, C: float = 1e-6) -> tuple[pulsim.Circuit, pulsim.SimulationOptions]:
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
    ckt.add_resistor("R1", in_, out, R)
    ckt.add_capacitor("C1", out, ckt.ground(), C, 0.0)

    opts = pulsim.SimulationOptions()
    opts.tstop = 1e-6
    opts.dt = 1e-7
    opts.adaptive_timestep = False
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()
    return ckt, opts


def main() -> None:
    R, C = 1e3, 1e-6
    ckt, opts = build_rc(R, C)
    sim = pulsim.Simulator(ckt, opts)

    ac = pulsim.AcSweepOptions()
    ac.f_start = 1.0
    ac.f_stop = 1e6
    ac.points_per_decade = 30
    ac.scale = pulsim.AcSweepScale.Logarithmic
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]

    result = sim.run_ac_sweep(ac)
    if not result.success:
        raise SystemExit(f"AC sweep failed: {result.failure_reason}")

    f_corner = 1.0 / (2.0 * math.pi * R * C)
    m = result.measurements[0]
    i_corner = min(
        range(len(result.frequencies)),
        key=lambda i: abs(math.log10(result.frequencies[i]) - math.log10(f_corner)),
    )
    print(f"RC corner frequency: {f_corner:.3f} Hz")
    print(f"  Closest sweep point: {result.frequencies[i_corner]:.3f} Hz")
    print(f"  Magnitude: {m.magnitude_db[i_corner]:.3f} dB   (analytical: -3.010)")
    print(f"  Phase:     {m.phase_deg[i_corner]:.3f} deg     (analytical: -45.000)")
    print(f"Sweep cost: {result.total_factorizations} factorizations + "
          f"{result.total_solves} solves in {result.wall_seconds*1e3:.2f} ms")

    if os.environ.get("PULSIM_EXAMPLE_NOPLOT"):
        return
    try:
        fig, _ = pulsim.bode_plot(result, title="RC low-pass — Pulsim AC sweep")
        out_path = os.path.join(os.path.dirname(__file__), "01_ac_sweep_rc.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved Bode plot: {out_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot. "
              "Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
