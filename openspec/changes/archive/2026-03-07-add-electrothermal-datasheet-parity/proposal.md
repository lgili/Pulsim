## Why
Pulsim already has a solid electrothermal baseline, but it is still below the practical modeling workflow users expect from professional power-electronics tools for semiconductor loss and thermal analysis. The current runtime does not yet provide full datasheet-grade switching/conduction characterization, multi-stage thermal networks as first-class runtime behavior, or a fully backend-owned contract that lets GUI stay thin and avoid physics heuristics.

This capability is strategic: electrothermal fidelity in closed-loop converters must become a differentiator for Pulsim.

## What Changes
- Add a datasheet-grade semiconductor loss engine in the backend for MOSFET/IGBT/diode, including multidimensional switching-energy interpolation and temperature-dependent conduction.
- Generalize switching-event accounting so native and externally forced switch-like devices (including PWM-driven semiconductor targets) produce correct switching-loss telemetry.
- Add multi-stage thermal network support (single RC, Foster, Cauer) and optional shared thermal coupling constructs in runtime.
- Extend canonical transient channel export with per-component loss time-series (`Pcond`, `Psw_on`, `Psw_off`, `Prr`, `Ploss`) alongside canonical temperature channels.
- Keep backward compatibility for existing scalar loss fields and existing summary surfaces while adding richer contracts.
- Define backend-complete YAML and Python contracts so GUI only handles data-entry UX and visualization orchestration.
- Add benchmark and validation gates for parity-quality electrothermal behavior, including closed-loop buck and analytic component-level checks.
- Add explicit documentation of backend vs GUI responsibilities for this capability.

## Impact
- Affected specs: `kernel-v1-core`, `device-models`, `netlist-yaml`, `python-bindings`, `benchmark-suite`
- Affected code:
  - `core/src/v1/transient_services.cpp`
  - `core/src/v1/simulation.cpp`
  - `core/src/v1/simulation_step.cpp`
  - `core/include/pulsim/v1/runtime_circuit.hpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/_pulsim.pyi`
  - electrothermal tests/benchmarks and docs under `python/tests`, `benchmarks`, `docs`
