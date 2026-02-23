## Why
The current transient runtime has accumulated multiple overlapping paths (native robust path, external-backend escalation paths, projected-wrapper compatibility logic, and procedural retry wrappers). This overlap increases maintenance cost, duplicates mathematical logic, and makes it harder to guarantee that switched-converter accuracy and runtime improve together.

For Pulsim to differentiate in power-electronics simulation, we need one solver architecture optimized specifically for switched converters, with deterministic event handling, high convergence resilience, and strict phase-by-phase regression gates. The user experience should stay simple: select only `fixed` or `variable` timestep mode.

## What Changes
- Introduce a **Unified Native Solver Core** for transient simulation with a single mathematical pipeline shared by fixed-step and variable-step execution modes.
- Remove supported runtime dependence on alternate transient backends and duplicate projected-wrapper solver paths.
- Define a **dual-mode user surface** (`fixed` or `variable`) with internal solver policy selection, while keeping advanced controls as expert overrides.
- Implement a deterministic convergence-recovery ladder (dt backoff, damping/trust-region escalation, transient regularization, controlled retry exhaustion).
- Implement event-segmented integration for switched converters (PWM boundaries, threshold crossings, deterministic earliest-event targeting).
- Consolidate linear/nonlinear solver services to avoid duplicated assembly/solve code between DC/transient and between timestep modes.
- Add strict benchmark/parity/stress phase gates so each implementation phase must demonstrate non-regression in precision and efficiency before continuing.
- Add migration guidance and explicit diagnostics for removed legacy transient-backend configuration keys.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `transient-timestep`
  - `linear-solver`
  - `netlist-yaml`
  - `python-bindings`
  - `benchmark-suite`
- Affected code:
  - `core/src/v1/simulation.cpp`
  - `core/src/v1/sundials_backend.cpp` (decommissioned from supported runtime path)
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/include/pulsim/v1/solver.hpp`
  - `core/include/pulsim/v1/integration.hpp`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/__init__.py`
  - `python/pulsim/__init__.pyi`
  - `core/tests/*` and `benchmarks/*`
- Breaking changes:
  - Runtime support for legacy transient-backend selection keys is removed from the supported path.
  - Canonical user-facing transient mode selection becomes `fixed` or `variable`.
- Relationship with active changes:
  - This change supersedes architecture goals of `add-sundials-direct-dae-runtime` for the supported runtime path.
