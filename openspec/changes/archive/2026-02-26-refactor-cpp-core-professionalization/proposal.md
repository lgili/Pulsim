## Why
The C++ core has grown quickly and now concentrates multiple responsibilities in a few very large files (`runtime_circuit.hpp`, `high_performance.hpp`, `integration.hpp`, `simulation.cpp`, `yaml_parser.cpp`). This increases coupling, slows safe evolution, and raises regression risk when adding new features.

To make Pulsim professionally maintainable and consistently fast, we need a deliberate refactor program that hardens architecture boundaries, codifies safety/performance contracts, and enforces KPI-based regression gates in CI.

## What Changes
- Define and enforce a layered C++ core architecture with one-way dependencies and explicit module ownership.
- Standardize modern C++20/23 engineering rules across core modules (RAII ownership, constrained interfaces, deterministic error modeling, and strict warning policy).
- Introduce stable extension contracts for new devices, solvers, integrators, and control blocks so features can be added without touching orchestrator internals.
- Establish safety hardening requirements (failure containment, finite-value guards, sanitizer/static-analysis gates, deterministic error semantics).
- Define hot-path performance discipline (allocation-free steady-state stepping, cache reuse policies, deterministic invalidation rules).
- Strengthen YAML and Python compatibility contracts to keep feature growth safe and migration-friendly.
- Add benchmark governance requirements with frozen baselines and strict non-regression KPI gates.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `linear-solver`
  - `benchmark-suite`
  - `python-bindings`
  - `netlist-yaml`
- Affected code (planned):
  - `core/include/pulsim/v1/runtime_circuit.hpp`
  - `core/include/pulsim/v1/high_performance.hpp`
  - `core/include/pulsim/v1/integration.hpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/simulation.cpp`
  - `core/src/v1/transient_services.cpp`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - benchmark and CI workflow files
- Relationship with active changes:
  - Complements `refactor-unified-native-solver-core` with architecture governance, extensibility contracts, and KPI enforcement.
  - Must not regress acceptance criteria already introduced by `refactor-unified-native-solver-core` and `add-sundials-direct-dae-runtime`.
