## Gates & Definition of Done

- [ ] G.1 Clean build ≤ 75 % of baseline — gated on the deferred bindings split (Phases 2-3 below). The bench tool establishes the baseline (39.16 s on Apple Silicon / Release+LTO / `pulsim_tests`); the split work targets 75 % of that, ≈ 29 s.
- [ ] G.2 Incremental rebuild ≤ 10 % of clean — gated on the bindings split. Baseline measured at 66.7 % today (one leaf-header touch triggers ~ 26 s rebuild because `bindings.cpp` and several large TUs all transitively include the touched header). Splitting `bindings.cpp` plus library-level partitioning brings this under 10 %.
- [x] G.3 All existing tests pass unmodified — 4001 + 1090 C++ assertions green, 41 Python tests green. The `RobustnessProfile` change shipped alongside this one is purely additive.
- [x] G.4 No public API/ABI change — only build-system additions (a benchmark script, a new `RobustnessProfile` header). Existing imports / links unchanged.
- [x] G.5 Build-time metric tracked — [`scripts/build_bench.py`](../../../scripts/build_bench.py) emits both wallclock numbers + JSON artifact. CI integration (auto-fail on regression) is the deferred Phase 1.3 follow-up; the script itself is final.

## Phase 1: Build-time baseline measurement
- [x] 1.1 [`scripts/build_bench.py`](../../../scripts/build_bench.py) measures clean + incremental rebuild times. Touches a single header between runs and reports the ratio (Phase 2 target ≤ 10 %).
- [x] 1.2 Baseline captured on Apple Silicon / AppleClang 17 / Release+LTO: 39.16 s clean, 26.13 s incremental, 66.7 % ratio for `pulsim_tests`. Linux + Windows numbers are CI-environment specific; the script runs on any platform with cmake + python3.
- [ ] 1.3 CI artifact JSON + regression gate — script emits JSON on `--json` flag; CI integration that fails the build on regression is the deferred follow-up.

## Phase 2: bindings.cpp split — devices and control
- [ ] 2.1 / 2.2 / 2.3 / 2.4 / 2.5 Split `python/bindings.cpp` (2857 lines) into `python/bindings/devices.cpp`, `bindings/control.cpp`, etc. — deferred. Mechanical refactor that benefits from the bench baseline above for objective wall-clock proof. The split work is its own change because:
    - Pybind11 type ordering matters (some classes register before others depend on them); refactoring requires careful audit of every `py::class_<>` registration.
    - The `_pulsim` CMake target needs reorganization to compile multiple TUs.
    - Risk-bearing — a bad split silently breaks Python imports.
  Split tracked as next-change priority once the user signs off on the approach.

## Phase 3: bindings.cpp split — solver/parser/sim
- [ ] 3.1 / 3.2 / 3.3 / 3.4 — deferred (paired with Phase 2; same change).

## Phase 4-6: Library-level partition
- [ ] 4.1 / 5.1 / 6.1 Split `pulsim` static library into `pulsim_core` (no compiled deps) + `pulsim_simulation` (Newton-DAE + LTE + transient_services) + `pulsim_periodic` (shooting, harmonic balance) — deferred. The library-level split is the second-largest leverage point after the bindings split; same change cycle.

## Phase 7-8: Targeted header refactors
- [ ] 7.1 / 8.1 Move large device-class implementations from monolithic headers into per-device translation units to break the "touch one header → rebuild everything" cascade — deferred. Pairs with the library-level split.

## Phase 9: CI integration
- [ ] 9.1 GitHub Actions / Azure Pipelines step that runs `scripts/build_bench.py --json out.json` and fails on regression — deferred. The script is final today; the CI hook is downstream.

## Phase 10: Docs
- [x] 10.1 [`docs/build-system.md`](../../../docs/build-system.md) — bench tool reference + baseline numbers + split-work follow-up plan. Linked from `mkdocs.yml`.
