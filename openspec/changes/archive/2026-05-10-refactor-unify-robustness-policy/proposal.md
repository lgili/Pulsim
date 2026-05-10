## Why

The "robust defaults" logic for switching simulations is currently duplicated in **three** places:

1. `core/src/v1/simulation.cpp:62-130` — `apply_robust_linear_solver_defaults` and `apply_robust_newton_defaults` plus `apply_auto_transient_profile`.
2. `python/bindings.cpp:50-150` — `apply_robust_linear_solver_defaults` and `apply_robust_newton_defaults` (independently maintained C++ duplication, plus `build_robust_transient_options`).
3. `python/pulsim/__init__.py:314-373` — `_tune_linear_solver_for_robust` and `_tune_newton_for_robust` (Python duplication of the same magic numbers).

These three implementations drift over time. Every change to the robustness profile risks getting applied in only one or two places, producing unpredictable behavior. The Python wrapper retry layer (`run_transient` with auto-bleeders) compounds the problem by adding **another** recovery axis on top.

This change consolidates all robustness policy into a single owner inside the kernel, exposes it as a typed `RobustnessProfile`, and removes the duplicates.

## What Changes

### Single Source of Truth: `RobustnessProfile`
- New header `core/include/pulsim/v1/robustness_profile.hpp` with:
  - `enum class RobustnessTier { Strict, Standard, Aggressive }` for declared user intent.
  - `struct RobustnessProfile` collecting all knobs (Newton, linear solver, integrator, recovery, fallback) in one struct.
  - `RobustnessProfile::for_circuit(const Circuit&, RobustnessTier)` factory that derives the profile from circuit content (switching count, nonlinear count, control blocks).
- `SimulationOptions::apply_robustness(profile)` mutates options in place once, deterministically.

### Remove Duplicates
- Delete `apply_robust_*_defaults` in `python/bindings.cpp`.
- Delete `_tune_*_for_robust` in `python/pulsim/__init__.py`.
- Both call sites use `RobustnessProfile::for_circuit()` exposed via pybind11.

### Deprecate Python Retry Layer (with migration path)
- Once Phase-0 PWL engine is the resolved default for switching circuits, the Python wrapper retry/auto-bleeder layer in `__init__.py:run_transient` is unnecessary.
- Behind feature flag `PULSIM_LEGACY_RETRY_FALLBACK`, deprecate. Warn on use.
- Removed in a separate change `remove-legacy-retry-fallback` after one release window.

### Configuration Surface
- Top-level YAML: `simulation.robustness: strict | standard | aggressive` (default `standard`).
- Per-component override: `<component>.robustness: ...` for fine-grained control of voltage limiting, damping at device level.
- Python: `SimulationOptions.robustness = RobustnessTier.Aggressive`.

### Telemetry
- `BackendTelemetry.robustness_profile` records the resolved tier and key knob values for reproducibility.
- Diff vs default profile reported in result message when verbosity enabled.

### Documentation
- Single page `docs/robustness-profile.md` documenting tiers, knobs, when to use each.
- Replaces scattered notes across multiple docs files.

## Impact

- **Affected specs**: `kernel-v1-core` (single robustness owner), `python-bindings` (Python surface, retry layer deprecation).
- **Affected code**: new `core/include/pulsim/v1/robustness_profile.hpp` + `.cpp`; removals in `python/bindings.cpp` and `python/pulsim/__init__.py`; updates to `Simulator` ctor; YAML parser for new field.
- **No behavior change** when defaults stay the same; existing tests pass without modification.
- **Risk**: behavioral drift between old and new for users relying on undocumented Python wrapper retries; mitigated by deprecation flag.

## Success Criteria

1. **Single owner**: only `core/include/pulsim/v1/robustness_profile.hpp` declares robust defaults; grep for `_tune_robust` / `apply_robust_*` shows just one definition + uses.
2. **Behavior preserved**: existing test suite passes unchanged; existing benchmark KPIs unchanged.
3. **Python retry deprecated**: wrapper retry layer logs deprecation when used; default path uses kernel-resolved profile + PWL engine.
4. **YAML surface**: `simulation.robustness: aggressive` produces the same telemetry-recorded knob set as the C++/Python equivalent.
5. **Documentation**: single robustness page replaces scattered notes; reviewed for clarity.
