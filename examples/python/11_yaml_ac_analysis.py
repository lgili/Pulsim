"""Run an AC sweep on a circuit loaded from a YAML netlist.

The YAML schema supports an `analysis:` top-level block (validated by
`add-frequency-domain-analysis` Phase 7 — see ``docs/ac-analysis.md``),
but the Python bindings don't currently expose ``options.ac_sweeps`` /
``options.fra_sweeps``. The pragmatic workaround until the binding lands
is:

  1. Load the circuit from YAML via ``pulsim.YamlParser``.
  2. Build the ``AcSweepOptions`` by hand in Python (this script).
  3. Dispatch through ``Simulator.run_ac_sweep``.

This script demonstrates that pipeline against a small RC low-pass — the
analytical -3 dB / -45° corner at 159.155 Hz validates the round-trip.

Run::

    python 11_yaml_ac_analysis.py

See also: docs/ac-analysis.md, docs/netlist-format.md
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pulsim


YAML_NETLIST = """\
schema: pulsim-v1
version: 1

simulation:
  tstop: 1.0e-3
  dt:    1.0e-7

components:
  - {type: voltage_source, name: V1, nodes: [in, gnd], value: 1.0}
  - {type: resistor,       name: R1, nodes: [in, out], value: 1.0e3}
  - {type: capacitor,      name: C1, nodes: [out, gnd], value: 1.0e-6, init: 0.0}
"""


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="pulsim_yaml_ac_"))
    netlist_path = work / "rc.yaml"
    netlist_path.write_text(YAML_NETLIST)
    print(f"Wrote netlist: {netlist_path}")

    parser = pulsim.YamlParser()
    circuit, options = parser.load(str(netlist_path))
    print(f"Circuit:   {circuit.num_nodes()} nodes, "
          f"{circuit.num_branches()} branches\n")

    # The C++ parser supports a top-level `analysis:` block that maps
    # one-to-one onto AcSweepOptions / FraOptions. The Python bindings
    # don't expose `options.ac_sweeps` yet (deferred follow-up of
    # `add-frequency-domain-analysis`). For now, build AcSweepOptions
    # directly in Python.
    if hasattr(options, "ac_sweeps") and len(options.ac_sweeps) > 0:
        ac = options.ac_sweeps[0]
        source = "YAML analysis: block"
    else:
        ac = pulsim.AcSweepOptions()
        ac.f_start = 1.0
        ac.f_stop = 1e6
        ac.points_per_decade = 30
        ac.scale = pulsim.AcSweepScale.Logarithmic
        ac.perturbation_source = "V1"
        ac.measurement_nodes = ["out"]
        source = "hand-built (binding for ac_sweeps not yet exposed)"

    print(f"Sweep config source: {source}")

    sim = pulsim.Simulator(circuit, options)
    result = sim.run_ac_sweep(ac)
    if not result.success:
        raise SystemExit(f"AC sweep failed: {result.failure_reason}")

    f_corner = 1.0 / (2.0 * math.pi * 1e3 * 1e-6)
    m = result.measurements[0]
    i_corner = min(
        range(len(result.frequencies)),
        key=lambda i: abs(math.log10(result.frequencies[i])
                          - math.log10(f_corner)),
    )
    print(f"\nResults:")
    print(f"  points:      {len(result.frequencies)}")
    print(f"  wallclock:   {result.wall_seconds*1e3:.2f} ms")
    print(f"  factorize:   {result.total_factorizations}    "
          f"solve: {result.total_solves}")
    print(f"  RC corner check (f ≈ {f_corner:.2f} Hz):")
    print(f"    mag   = {m.magnitude_db[i_corner]:+.3f} dB    "
          f"(analytical -3.010)")
    print(f"    phase = {m.phase_deg[i_corner]:+.3f} deg    "
          f"(analytical -45.000)")


if __name__ == "__main__":
    main()
