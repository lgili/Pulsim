"""Frequency Response Analysis (FRA) vs analytical AC sweep.

Both run the same RC low-pass, but FRA injects a small-signal sinusoid at
each frequency and measures the response by transient + DFT — exactly how
a benchtop network analyzer would. The two methods agree within 1 dB / 5°
on linearizable circuits (the contract from
``add-frequency-domain-analysis``).

Use FRA when nonlinear behaviour or PWM modulation matters; use the
analytical AC sweep when you want a fast Bode of a linearizable circuit.

Run::

    python 02_fra_vs_ac_sweep.py

See also: docs/fra.md
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

    f_corner = 1.0 / (2.0 * math.pi * R * C)
    f_start = f_corner / 10.0
    f_stop = f_corner * 10.0

    # ---------------- analytical AC sweep ----------------
    ac = pulsim.AcSweepOptions()
    ac.f_start = f_start
    ac.f_stop = f_stop
    ac.points_per_decade = 4
    ac.scale = pulsim.AcSweepScale.Logarithmic
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]
    ac_result = sim.run_ac_sweep(ac)
    if not ac_result.success:
        raise SystemExit(f"AC sweep failed: {ac_result.failure_reason}")

    # ---------------- empirical FRA ----------------
    fra = pulsim.FraOptions()
    fra.f_start = f_start
    fra.f_stop = f_stop
    fra.points_per_decade = ac.points_per_decade
    fra.scale = ac.scale
    fra.perturbation_source = "V1"
    fra.perturbation_amplitude = 1e-2          # 10 mV small-signal sine
    fra.measurement_nodes = ["out"]
    fra.n_cycles = 6                          # injection cycles per point
    fra.discard_cycles = 2                    # transients dropped before DFT
    fra.samples_per_cycle = 64
    fra_result = sim.run_fra(fra)
    if not fra_result.success:
        raise SystemExit(f"FRA failed: {fra_result.failure_reason}")

    # ---------------- compare ----------------
    print(f"  freq (Hz) |  AC mag dB  |  FRA mag dB  |  ΔdB  |  AC phase  |  FRA phase  |  Δdeg")
    print("-" * 95)
    ac_m = ac_result.measurements[0]
    fra_m = fra_result.measurements[0]
    max_db_err = 0.0
    max_deg_err = 0.0
    for i, f in enumerate(ac_result.frequencies):
        d_db = abs(ac_m.magnitude_db[i] - fra_m.magnitude_db[i])
        d_deg = abs(ac_m.phase_deg[i] - fra_m.phase_deg[i])
        max_db_err = max(max_db_err, d_db)
        max_deg_err = max(max_deg_err, d_deg)
        print(f"  {f:8.2f}  | {ac_m.magnitude_db[i]:>10.3f}  "
              f"| {fra_m.magnitude_db[i]:>11.3f}  | {d_db:>4.2f}  "
              f"| {ac_m.phase_deg[i]:>9.2f}  | {fra_m.phase_deg[i]:>10.2f}  | {d_deg:>4.2f}")
    print()
    print(f"max |ΔdB|  = {max_db_err:.3f}  (contract: ≤ 1.0)")
    print(f"max |Δdeg| = {max_deg_err:.3f}  (contract: ≤ 5.0)")

    if os.environ.get("PULSIM_EXAMPLE_NOPLOT"):
        return
    try:
        fig, _ = pulsim.fra_overlay(ac_result, fra_result, title="RC: AC sweep vs FRA")
        out_path = os.path.join(os.path.dirname(__file__), "02_fra_vs_ac_sweep.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved overlay: {out_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
