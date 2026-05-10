# Build System

> Status: build-bench helper shipped. Full bindings.cpp + modular-
> CMake split is the mechanical follow-up that benefits from the
> baseline this change establishes.

`refactor-modular-build-split` Phase 1: a benchmark harness that
measures Pulsim's clean + incremental rebuild wallclock so the
mechanical bindings/modular-CMake split work that follows can prove
a real wall-clock win.

## TL;DR

```bash
python3 scripts/build_bench.py --build-dir build --target pulsim_tests
```

Sample output on Apple Silicon / AppleClang 17 / Release+LTO baseline:

```
Clean:        39.16 s
Incremental:  26.13 s
Ratio:        66.7 %
```

The Phase 2 target is ≤ 10 % incremental ratio — touching one
device-header should not trigger half a clean build. Today's 67 %
ratio is the baseline against which the planned split work will be
measured.

## What's in baseline today

| Target | Translation units | Notes |
|---|---|---|
| `pulsim_core` | header-only | All v1/v2 device classes + frequency analysis + magnetic + catalog + templates + motors + grid sit here |
| `pulsim` | 5 source files | `simulation.cpp`, `simulation_periodic.cpp`, `simulation_step.cpp`, `transient_services.cpp`, `yaml_parser.cpp` + `simulation_control.cpp` |
| `pulsim_tests` | 21 source files | header-only tests (concepts, devices, stamps, frequency-analysis Phases 1-9, magnetic Phases 1-6, catalog Phases 1-8, motors, grid, robustness) |
| `pulsim_simulation_tests` | 16 source files | tests that need the compiled `pulsim` lib (transient/PWL/buck benchmarks/AC sweep) |
| `_pulsim` Python module | 1 source file (`bindings.cpp`, 2857 lines) | the canonical bottleneck — touching anything that bindings.cpp includes triggers a 30+ s rebuild on its own |

The largest leverage point is `bindings.cpp`: all category bindings
(devices, control, simulation, parser, solver, frequency analysis,
templates, motors, grid, ...) live in one TU. Splitting it into
`bindings/devices.cpp`, `bindings/control.cpp`, etc., each calling
into a `register_*(m)` function from a thin `bindings/main.cpp`,
turns "edit one device → recompile bindings.cpp" (~35 s) into
"recompile only the affected category file" (~5 s).

## Running the bench

```bash
# Baseline (touch a leaf header):
python3 scripts/build_bench.py --build-dir build --target pulsim_tests \
    --touch-file core/include/pulsim/v1/numeric_types.hpp

# After a bindings split: touching just one category file
python3 scripts/build_bench.py --build-dir build --target _pulsim \
    --touch-file python/bindings/devices.cpp \
    --json bench.json
```

The `--json` artifact is the CI-ratchet hook: regress past the
recorded baseline by 10 % → fail the build. The CI integration is the
deferred Phase 1.3 follow-up; the bench tool itself is final today.

## Limitations / follow-ups

- **bindings.cpp split** (Phases 2-3 of the proposal): mechanical
  refactor of 2857 lines into category files. Requires care to keep
  the pybind11 module entry point + every type registration in the
  right order (some types reference others' bindings during
  registration). The bench harness is the prerequisite that lets us
  measure the wall-clock win — without a baseline number the split
  is blind. Tracked as the next change.
- **Library-level split** (Phases 4-6 of the proposal): split the
  `pulsim` static library into `pulsim_core` (no compiled deps) +
  `pulsim_simulation` (Newton-DAE + LTE) + `pulsim_periodic`
  (shooting, harmonic balance) — the Newton-DAE consumers pull a
  smaller graph than today's monolith. Same prerequisite: baseline
  bench numbers.
- **CI ratchet**: emit the JSON artifact on every PR build and fail
  if the incremental ratio regresses past N %. Pairs with the splits
  above so the gate has somewhere to land.
- **Unity builds opt-in**: CMake `set(CMAKE_UNITY_BUILD ON)` for
  release-build CI runs that don't care about incremental rebuild;
  trades incremental friendliness for ~ 30 % faster clean build.

## See also

- [`backend-architecture.md`](backend-architecture.md) — the layered
  TU graph the split work will follow.
- The `python/bindings.cpp` file itself — start of the largest TU
  in the project.
