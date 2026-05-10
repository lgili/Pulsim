## Why

Two files are pathologically large and hurt iteration speed:

- `python/bindings.cpp` — 139 KB, ~3500 lines, single translation unit binding the entire pybind11 surface. A one-line edit triggers a full recompilation of the binding (~30 s on a fast workstation, longer in CI). Splitting modules can shrink incremental rebuilds by 10×.
- `core/include/pulsim/v1/runtime_circuit.hpp` — 3110 lines, mostly inline template implementations. Every translation unit including this header re-instantiates the full `Circuit` class and all device variants. Net effect: every C++ source file pays a heavy compile cost. Moving inline implementations to a `.cpp` cuts compile time substantially.

Other bloated headers (`high_performance.hpp` 2544 lines, `integration.hpp` 1862 lines) exhibit the same pattern but to a lesser degree.

This change is pure refactoring — no behavior change, no API change. The objective is build-time and iteration-time recovery.

## What Changes

### `python/bindings.cpp` Modular Split
- Split into per-domain modules:
  - `bindings/devices.cpp` — device class bindings (Resistor, Capacitor, MOSFET, IGBT, ...)
  - `bindings/control.cpp` — PI/PID/PWM/comparator/etc.
  - `bindings/simulation.cpp` — Simulator, SimulationOptions, SimulationResult
  - `bindings/parser.cpp` — YamlParser
  - `bindings/thermal.cpp` — Foster/Cauer, ThermalSimulator
  - `bindings/loss.cpp` — loss accumulators
  - `bindings/solver.cpp` — NewtonOptions, LinearSolverStackConfig, ...
  - `bindings/analysis.cpp` — periodic, harmonic balance (post-Phase-1, AC sweep too)
  - `bindings/main.cpp` — `PYBIND11_MODULE(_pulsim, m)` orchestrator calling per-domain `register_*(m)` functions.
- Each per-domain file ≤500 lines.
- Single shared header `bindings/common.hpp` with helpers (`unwrap_result`, `apply_robust_*_defaults`).

### `runtime_circuit.hpp` Implementation Split
- Header keeps declarations, public templates, type aliases.
- Move method bodies to `core/src/v1/runtime_circuit.cpp`:
  - `add_resistor`, `add_capacitor`, etc.
  - `assemble_jacobian`, `assemble_residual`, `assemble_state_space`.
  - `apply_numerical_regularization`.
- Templated methods that need to be instantiated per-type stay in header but in a separate `runtime_circuit_impl.hpp` included only where needed.
- Explicit instantiation declarations (`extern template`) in header; definitions in `.cpp`.

### Other Bloated Headers (Audit and Trim)
- `high_performance.hpp` (2544 lines): identify which functions are actually used; move unused or rarely-used to `.cpp`. Many SIMD utilities in this file are likely speculative.
- `integration.hpp` (1862 lines): keep coefficient structs in header, move `BDFOrderController` and `AdvancedTimestepController` impl to `.cpp`.

### Build-Time Measurement
- Add `make build-bench` target that compiles a clean tree and reports total wallclock.
- CI tracks build time per platform; regression alerts if >10% growth without justification.

### Compile Definition Hygiene
- Audit `add_compile_definitions(...)` for unused macros.
- Consolidate platform-specific guards.

## Impact

- **Affected specs**: `kernel-v1-core` (modularization), `python-bindings` (modular build).
- **Affected code**: `python/bindings.cpp` → `python/bindings/*.cpp`; `core/include/pulsim/v1/runtime_circuit.hpp` → `core/src/v1/runtime_circuit.cpp` + slim header; analogous for other bloated headers.
- **No API change**, **no behavior change**. Pure refactor.
- **Performance target**: clean build time -25%, incremental edit-rebuild -50%.

## Success Criteria

1. **Clean build**: `pulsim_core` + `_pulsim` clean build wallclock ≤75% of baseline.
2. **Incremental rebuild**: editing one device header triggers only that device + linker, not full rebuild.
3. **Test parity**: every existing test passes unmodified.
4. **No ABI/API change**: existing user code (Python and C++) works without recompilation.
5. **CI tracking**: build-time metric tracked across platforms; regression alerts in place.
