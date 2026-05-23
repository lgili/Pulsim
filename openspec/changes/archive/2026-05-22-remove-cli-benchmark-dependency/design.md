## Context

The v1 kernel already supports advanced solver stacks, adaptive/stiff integration, and periodic steady-state methods. However, benchmark tooling is split between a CLI path and a Python fallback path with intentionally reduced capability. This split causes maintenance overhead and inconsistent validation outcomes.

The target architecture is a single Python-first execution path that still uses the same C++ kernel and parsing semantics.

## Goals / Non-Goals

### Goals

- Use one runtime path for benchmark execution and validation.
- Expose full simulation control surface to Python bindings.
- Keep YAML parsing semantics consistent between benchmark execution and core parser behavior.
- Eliminate backend-driven `skipped` statuses in benchmark matrix runs.

### Non-Goals

- Rewriting kernel numerics or changing solver algorithms.
- Providing full SPICE directive compatibility in this change.
- Adding a new external benchmark service.

## Architecture Decisions

### 1) Python-First Runtime Path

Benchmark runners will execute simulations through Python APIs backed by the v1 kernel, not by spawning a CLI command. This removes process boundary drift and ensures shared runtime semantics.

### 2) Full Runtime Exposure in Bindings

Bindings will expose simulation primitives currently needed only by internal C++ paths:

- `SimulationOptions` with solver/integrator/timestep/periodic configuration
- `Simulator` class methods:
  - `dc_operating_point`
  - `run_transient`
  - `run_periodic_shooting`
  - `run_harmonic_balance`
- YAML parser access (`YamlParser`) that returns `(Circuit, SimulationOptions)` with strict diagnostics support

### 3) Unified YAML Semantics

Benchmark execution must use the same v1 YAML parser semantics used elsewhere in kernel-based flows. Local Python-only emulation of YAML simulation semantics is removed for matrix-critical paths.

### 4) Structured Telemetry

Benchmark telemetry moves from CLI stdout regex parsing to structured fields sourced from simulation result objects (iterations, steps, rejections, runtime, solver telemetry).

### 5) Validation Status Discipline

Status outcomes are limited to explicit execution semantics:

- `passed`
- `failed`
- `baseline` (where applicable)
- `skipped` only for explicit manifest/data reasons (e.g., missing ngspice netlist), not backend type.

## Data Flow

1. Load benchmark YAML + scenario override.
2. Parse with exposed C++ `YamlParser`.
3. Build `Simulator` with parsed/overridden `SimulationOptions`.
4. Execute transient or periodic method based on options.
5. Export waveform output and structured telemetry.
6. Run analytical/reference/ngspice comparison and emit artifacts.

## Compatibility and Migration

- Benchmark invocation remains Python-script based.
- Legacy `--pulsim-cli` behavior is removed or converted to no-op deprecation shim during migration window.
- Documentation is updated to point to Python-first benchmark commands.

## Risks and Mitigations

- Risk: Binding surface expansion can increase maintenance load.
  - Mitigation: expose thin wrappers around existing kernel types, avoid duplicated logic in Python.
- Risk: Behavior differences appear after removing fallback.
  - Mitigation: add parity tests between previous benchmark outputs and new structured path on representative circuits.
- Risk: Periodic flows expose edge-case instability in automation.
  - Mitigation: explicit error reporting and dedicated periodic benchmark fixtures in CI.

## Test Strategy

- Unit tests for new bindings (`SimulationOptions`, `Simulator`, parser, periodic methods).
- Integration tests for benchmark runners with matrix scenarios.
- Regression checks for artifact schema (`results.csv`, `results.json`, `summary.json`).
- ngspice comparator smoke tests using Python runtime execution path.
