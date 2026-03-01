## Why
The current runtime only exposes in-house implicit integrators and Newton loop behavior. For very stiff converter DAEs, convergence and robustness are still inconsistent across platforms and large models. We already gate SUNDIALS at build level, but there is no full runtime backend integration.

## What Changes
- Add a full SUNDIALS transient backend in v1 runtime with selectable solver families:
  - IDA for DAE-first MNA systems
  - CVODE BDF/Newton path for ODE-compatible reduced systems
  - ARKODE implicit path for stiff transient integration
- Add runtime fallback policy that can escalate from native solver stack to SUNDIALS backend when repeated transient failures occur.
- Extend `SimulationOptions` and YAML schema to configure SUNDIALS backend, tolerances, Jacobian/reinit policies, and fallback behavior.
- Expose SUNDIALS backend configuration/telemetry in Python bindings and keep procedural API backward-compatible.
- Add regression/stress tests for stiff switching converters validating convergence behavior and deterministic fallback traces with/without SUNDIALS.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `python-bindings`
- Affected code:
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/simulation.cpp`
  - new `core/src/v1/sundials_backend.cpp` (+ headers)
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/__init__.py` and stubs
  - CMake dependency wiring and CI/publish validation for optional SUNDIALS builds
