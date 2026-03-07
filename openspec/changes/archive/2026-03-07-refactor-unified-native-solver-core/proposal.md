## Why
The current transient runtime has accumulated multiple overlapping paths (native robust path, external-backend escalation paths, projected-wrapper compatibility logic, and procedural retry wrappers). This overlap increases maintenance cost, duplicates mathematical logic, and makes it harder to guarantee that switched-converter accuracy and runtime improve together.

For Pulsim to differentiate in power-electronics simulation, we need one solver architecture optimized specifically for switched converters: an event-driven hybrid core that uses state-space segment solving as primary path and deterministic nonlinear DAE fallback only when required. The user experience should stay simple: select only `fixed` or `variable` timestep mode.

## What Changes
- Introduce a **Unified Native Solver Core** for transient simulation with a single mathematical pipeline shared by fixed-step and variable-step execution modes.
- Introduce a **Hybrid Segment Engine** for switched converters:
  - primary path: state-space segment solve (`E x_dot = A x + B u + c`) between events
  - fallback path: shared nonlinear DAE solve for nonlinearity/stiffness edge cases
- Remove supported runtime dependence on alternate transient backends and duplicate projected-wrapper solver paths.
- Define a **dual-mode user surface** (`fixed` or `variable`) with internal solver policy selection, while keeping advanced controls as expert overrides.
- Implement a deterministic convergence-recovery ladder (dt backoff, damping/trust-region escalation, transient regularization, controlled retry exhaustion).
- Implement event-segmented integration for switched converters (PWM boundaries, threshold crossings, deterministic earliest-event targeting).
- Consolidate linear/nonlinear solver services to avoid duplicated assembly/solve code between DC/transient and between timestep modes.
- Integrate **losses + electrothermal coupling** into the same segmented runtime:
  - switching losses on event commits
  - conduction losses on accepted segments
  - thermal RC update with optional temperature-to-electrical feedback
- Add strict benchmark/parity/stress phase gates so each implementation phase must demonstrate non-regression in precision and efficiency before continuing.
- Extend KPI gating to include loss/thermal regression checks for converter-focused suites.
- Add migration guidance and explicit diagnostics for removed legacy transient-backend configuration keys.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `transient-timestep`
  - `linear-solver`
  - `device-models`
  - `netlist-yaml`
  - `python-bindings`
  - `benchmark-suite`
- Affected code:
  - `core/src/v1/simulation.cpp`
  - `core/src/v1/sundials_backend.cpp` (legacy/transitional path; removed from supported runtime path)
  - `core/include/pulsim/v1/transient_services.hpp`
  - `core/src/v1/transient_services.cpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/include/pulsim/v1/solver.hpp`
  - `core/include/pulsim/v1/integration.hpp`
  - `core/include/pulsim/v1/losses.hpp`
  - `core/include/pulsim/v1/thermal.hpp`
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
