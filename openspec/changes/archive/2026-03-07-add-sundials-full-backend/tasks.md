## 1. Core SUNDIALS backend
- [x] 1.1 Add v1 SUNDIALS backend abstraction and runtime implementation for IDA/CVODE/ARKODE integration.
- [x] 1.2 Wire backend selection in `Simulator::run_transient` with deterministic escalation path from native backend to SUNDIALS.
- [x] 1.3 Add structured backend telemetry (selected backend, solver family, reinit/recovery counters, failure reason mapping).

## 2. Configuration surface
- [x] 2.1 Extend `SimulationOptions` with SUNDIALS config and fallback policy fields.
- [x] 2.2 Extend YAML parser schema/validation and map new options to runtime options.
- [x] 2.3 Keep non-SUNDIALS builds behavior-compatible (feature off, no behavior regression).

## 3. Python surface
- [x] 3.1 Expose SUNDIALS configuration and telemetry through pybind bindings.
- [x] 3.2 Expose equivalent ergonomic helpers in `python/pulsim/__init__.py` and stubs.
- [x] 3.3 Preserve backward compatibility for current procedural API defaults.

## 4. Validation and hardening
- [x] 4.1 Add C++ regression tests for stiff converter transients using SUNDIALS-enabled and disabled paths.
- [x] 4.2 Add Python runtime tests for config/telemetry/fallback behavior.
- [x] 4.3 Add/adjust CI and release checks so SUNDIALS integration is verified (optional path + at least one enabled build).
