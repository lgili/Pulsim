## Gates & Definition of Done

- [ ] G.1 Benchmark runners execute without `pulsim` executable dependency.
- [ ] G.2 Matrix runs do not produce backend-type `skipped` results.
- [ ] G.3 Periodic scenarios (shooting/HB) execute via Python runtime path.
- [ ] G.4 Telemetry is sourced from structured simulation results.
- [ ] G.5 Benchmark docs no longer depend on `build/cli/pulsim` instructions.
- [ ] G.6 CLI path is removed only after all Python validation gates are green.

## Phase 1: Binding Surface

- [ ] 1.1 Expose `SimulationOptions` in Python with solver/integrator/timestep/periodic fields.
- [ ] 1.2 Expose `Simulator` class with transient, shooting, and harmonic balance methods.
- [ ] 1.3 Expose v1 YAML parser bindings (`YamlParser`, options, errors/warnings).
- [ ] 1.4 Add/update Python API tests for new classes and methods.

## Phase 2: Benchmark Runtime Refactor

- [ ] 2.1 Refactor `benchmark_runner.py` to execute through Python runtime API only.
- [ ] 2.2 Remove CLI process execution dependency from matrix-critical execution path.
- [ ] 2.3 Replace stdout regex telemetry extraction with structured telemetry mapping.
- [ ] 2.4 Keep artifact schema (`results.csv`, `results.json`, `summary.json`) stable.

## Phase 3: Validation and Comparator Behavior

- [ ] 3.1 Remove backend-driven validation skips from core benchmark matrix flow.
- [ ] 3.2 Ensure unsupported scenario features fail with explicit actionable diagnostics.
- [ ] 3.3 Refactor `benchmark_ngspice.py` to use the same Python runtime execution path.
- [ ] 3.4 Add tests for ngspice comparison flow with Python runtime output.

## Phase 4: Documentation and Migration

- [ ] 4.1 Update benchmark docs to Python-first workflow.
- [ ] 4.2 Remove or deprecate `--pulsim-cli` in benchmark scripts with migration notes.
- [ ] 4.3 Update root/user docs to remove stale benchmark CLI instructions.
- [ ] 4.4 Document known non-goals (e.g., no full SPICE directive execution in this change).

## Phase 5: Verification

- [ ] 5.1 Run benchmark suite smoke set and store sample artifacts.
- [ ] 5.2 Run validation matrix and verify no backend-type skips.
- [ ] 5.3 Run ngspice comparator smoke tests on mapped circuits.
- [ ] 5.4 Validate OpenSpec change with `openspec validate remove-cli-benchmark-dependency --strict`.

## Phase 6: Final CLI Removal (Post-Python Validation)

- [ ] 6.1 Confirm Phases 1-5 completed and Python validation gates are all green.
- [ ] 6.2 Remove `--pulsim-cli` argument and CLI discovery/dispatch code from benchmark runners.
- [ ] 6.3 Remove benchmark documentation references to CLI-based execution fallback.
- [ ] 6.4 Run full benchmark + matrix + ngspice smoke regression after CLI path removal.
