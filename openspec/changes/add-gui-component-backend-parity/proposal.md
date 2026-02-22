## Why

PulsimGui currently exposes 47 draggable components, but only 13 are fully converted to backend runtime devices. As of **2026-02-22**, the GUI-to-backend gap is:

- 27 components present in GUI library but rejected by backend conversion (`Backend converter does not yet support component ...`)
- 7 GUI instrumentation/routing components currently skipped as GUI-only blocks

This gap blocks end-to-end usability for advanced converter and control workflows and causes user-visible failures when building realistic schematics.

## What Changes

- Add a backend parity program for all missing GUI component families:
  - Power semiconductors and switching: `BJT_NPN`, `BJT_PNP`, `THYRISTOR`, `TRIAC`, `SWITCH`
  - Protection: `FUSE`, `CIRCUIT_BREAKER`, `RELAY`
  - Analog/control: `OP_AMP`, `COMPARATOR`, `PI_CONTROLLER`, `PID_CONTROLLER`, `MATH_BLOCK`, `PWM_GENERATOR`, `INTEGRATOR`, `DIFFERENTIATOR`, `LIMITER`, `RATE_LIMITER`, `HYSTERESIS`, `LOOKUP_TABLE`, `TRANSFER_FUNCTION`, `DELAY_BLOCK`, `SAMPLE_HOLD`, `STATE_MACHINE`
  - Magnetic/networks: `SATURABLE_INDUCTOR`, `COUPLED_INDUCTOR`, `SNUBBER_RC`
  - Instrumentation/routing: `VOLTAGE_PROBE`, `CURRENT_PROBE`, `POWER_PROBE`, `ELECTRICAL_SCOPE`, `THERMAL_SCOPE`, `SIGNAL_MUX`, `SIGNAL_DEMUX`
- Define a mixed-domain runtime contract (electrical + control + instrumentation graph) for deterministic simulation and waveform extraction.
- Extend YAML schema and Python bindings so all missing GUI components can be represented, validated, and executed.
- Add validation/benchmark coverage to prevent regression to “unsupported component” behavior.

## Impact

- Affected specs:
  - `device-models`
  - `kernel-v1-core`
  - `netlist-yaml`
  - `python-bindings`
  - `benchmark-suite`
- Affected code (expected):
  - `core/include/pulsim/v1/device_base.hpp`
  - `core/include/pulsim/v1/runtime_circuit.hpp`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/__init__.pyi`
  - `python/tests/**` and `benchmarks/**`

## Breaking / Behavioral Notes

- Existing APIs remain supported, but runtime behavior changes from “reject/skip unsupported components” to explicit model execution (or strict validation diagnostics where modeling is intentionally deferred).
- YAML strict-validation rules will be expanded with new component schemas and parameter constraints.

## Success Criteria

1. No component in the current GUI catalog is silently skipped or rejected without a structured diagnostic policy.
2. All 34 currently missing GUI components are covered by backend representation (physical model or explicit virtual instrumentation/routing model).
3. YAML + Python binding paths can construct and run circuits containing the newly supported components.
4. Benchmark/validation suite includes dedicated parity coverage for each previously missing component family.
