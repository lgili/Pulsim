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

For frequency-domain workflows, configure `options.frequency_analysis` and call:

```python
ac_result = sim.run_frequency_analysis(options.frequency_analysis)
```

For averaged plant workflows, configure `options.averaged_converter.enabled = True`
and run the same transient entrypoint:

```python
avg_result = sim.run_transient(circuit.initial_state())
```

## Core Concepts

### 1. Circuit and Topology

- `Circuit` holds nodes, devices, virtual channels, and runtime state.
- Parser-generated circuits are deterministic and suitable for CI reproducibility.

### 2. Simulation Options

Use `SimulationOptions` to control:

- time window and timestep strategy (`Fixed` vs `Variable`)
- averaged-converter mode (`simulation.averaged_converter`)
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

## Closed-Loop + Thermal Workflow

For converter workflows with control and electrothermal coupling:

1. Enable `simulation.enable_losses: true`.
2. Configure `simulation.thermal.enabled: true`.
3. Configure control update policy with `simulation.control.mode`.
4. Add `component.thermal` blocks to thermal-capable devices.
5. Validate `result.component_electrothermal` together with waveforms.

Recommended defaults:

- `simulation.control.mode: auto` for PWM-driven loops.
- PI/PID with `output_min/output_max` and `anti_windup: 1.0`.
- Thermal per component with explicit `rth`/`cth` in strict parser mode.

Primary result fields:

- `result.loss_summary`
- `result.thermal_summary`
- `result.component_electrothermal`

## Recommended Configuration Baseline

For switched converters in production-like runs:

- start with variable timestep
- keep robust fallback policy enabled
- monitor rejection ratio and fallback causes
- add KPI regression thresholds before merging solver changes

For fast control-loop plant iteration in supported envelope:

- use averaged mode (`simulation.averaged_converter.enabled: true`)
- keep explicit duty bounds and envelope policy
- validate parity against a paired switching case before broad usage
- keep switched/electrothermal as final sign-off runs

## Integration Tips

- Always set `num_nodes` and `num_branches` when manually composing options.
- Keep YAML netlists as source-of-truth artifacts for reproducibility.
- Prefer explicit benchmark manifests instead of ad-hoc scripts in CI.

## Where To Go Next

- [Netlist YAML Format](netlist-format.md)
- [Frontend Control and Signals Contract](frontend-control-signals.md)
- [Frequency Analysis (AC Sweep)](frequency-analysis-ac-sweep.md)
- [Averaged Converter Modeling](averaged-converter-modeling.md)
- [Electrothermal Workflow](electrothermal-workflow.md)
- [Electrothermal Migration (Scalar to Datasheet)](electrothermal-migration-scalar-to-datasheet.md)
- [Configuration](configuration.md)
- [Examples and Results](examples-and-results.md)
- [Benchmarks and Parity](benchmarks-and-parity.md)
- [Troubleshooting](troubleshooting.md)
