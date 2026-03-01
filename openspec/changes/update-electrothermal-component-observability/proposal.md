## Why
Pulsim already computes electrothermal quantities (`loss_summary` and `thermal_summary`), but the current contract is not strict enough for production validation and external reporting. Today it is possible to finish a run without a deterministic per-component view that combines losses and temperature for each component, and thermal-port parameter behavior is not explicit enough when users enable thermal on YAML components.

To support trustworthy converter analysis and CI quality gates, we need a clear electrothermal contract: temperature and losses per component, deterministic validation behavior, and explicit parameter rules when thermal ports are enabled.

## What Changes
- Add a deterministic per-component electrothermal reporting contract in the v1 kernel result surface.
- Define validation rules for thermal-port enablement in YAML, including strict-mode behavior for missing required thermal constants.
- Enforce capability checks so thermal port enablement on unsupported components fails with typed diagnostics.
- Expose the unified per-component electrothermal view in Python bindings while preserving existing `loss_summary` and `thermal_summary` compatibility.
- Add benchmark/validation requirements that verify component-level electrothermal outputs (not only aggregate KPIs).

## Impact
- Affected specs: `kernel-v1-core`, `netlist-yaml`, `python-bindings`, `device-models`, `benchmark-suite`
- Affected code:
  - `core/src/v1/transient_services.cpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/_pulsim.pyi`
  - electrothermal benchmark and parser-validation test suites
