## Why
Pulsim already has strong converter and electrothermal fundamentals, but large portions of runtime orchestration are still concentrated in broad integration units. This increases coupling between events, control, losses, thermal, and channel emission, which slows independent improvements and raises regression risk for community contributions.

Benchmarking against PSIM/PLECS-level workflows also shows that future growth (analysis modules, codegen/HIL adapters, richer domain packages) requires stricter modular contracts inside the kernel.

## What Changes
- Introduce a runtime module architecture for the v1 kernel with deterministic lifecycle hooks.
- Refactor transient orchestration to execute a module pipeline where each module owns:
  - state,
  - channel/telemetry outputs,
  - diagnostics namespace,
  - declared dependencies/capabilities.
- Split existing cross-cutting concerns into isolated modules:
  - events/topology,
  - control/mixed-domain execution,
  - losses,
  - thermal,
  - channel/telemetry emission.
- Add deterministic module dependency resolution and compatibility diagnostics.
- Add module-scoped performance/quality gates (allocation, deterministic output, CPU overhead ceilings).
- Preserve all current external behavior for YAML/Python contracts during refactor (non-breaking internal architecture change).

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `device-models`
- Affected code:
  - `core/src/v1/simulation.cpp`
  - `core/src/v1/simulation_step.cpp`
  - `core/src/v1/transient_services.cpp`
  - `core/include/pulsim/v1/transient_services.hpp`
  - `core/include/pulsim/v1/runtime_circuit.hpp`
  - new modular runtime contract headers/sources under `core/include/pulsim/v1/` and `core/src/v1/`
  - module-scoped tests in `core/tests/` and Python runtime contract tests
- Breaking changes:
  - none intended for user-facing YAML/Python APIs in this change.
