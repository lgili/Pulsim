# User Guide

PulsimCore exposes a stable backend workflow centered on the Python package `pulsim`.

## Supported Product Surface

Supported:

- Python runtime (`import pulsim`)
- YAML netlists with `schema: pulsim-v1`
- Benchmark/parity/stress tooling under `benchmarks/`

Not part of the supported product surface:

- Legacy CLI-first workflows
- Legacy gRPC surface as primary integration path
- JSON netlist loading as a canonical input format

## Canonical Backend Workflow

1. Build or install the Python package.
2. Load a YAML netlist with `YamlParser`.
3. Validate/update `SimulationOptions`.
4. Run `Simulator(...).run_transient(...)`.
5. Consume waveforms and telemetry.

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/rc_step.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())
options.step_mode = ps.StepMode.Variable

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())
```

## Core Concepts

### 1. Circuit and Topology

- `Circuit` holds nodes, devices, virtual channels, and runtime state.
- Parser-generated circuits are deterministic and suitable for CI reproducibility.

### 2. Simulation Options

Use `SimulationOptions` to control:

- time window and timestep strategy (`Fixed` vs `Variable`)
- nonlinear convergence (`NewtonOptions`)
- linear solver stack and fallback behavior
- event handling, periodic methods, and thermal coupling

### 3. Telemetry and Diagnostics

The backend reports structured diagnostics such as:

- solver/fallback traces
- linear solver telemetry
- event metadata
- thermal summary / loss metrics

This data should be consumed in CI gates instead of relying on visual-only inspection.

## Recommended Configuration Baseline

For switched converters in production-like runs:

- start with variable timestep
- keep robust fallback policy enabled
- monitor rejection ratio and fallback causes
- add KPI regression thresholds before merging solver changes

## Integration Tips

- Always set `num_nodes` and `num_branches` when manually composing options.
- Keep YAML netlists as source-of-truth artifacts for reproducibility.
- Prefer explicit benchmark manifests instead of ad-hoc scripts in CI.

## Where To Go Next

- [Netlist YAML Format](netlist-format.md)
- [Examples and Results](examples-and-results.md)
- [Benchmarks and Parity](benchmarks-and-parity.md)
- [Troubleshooting](troubleshooting.md)
