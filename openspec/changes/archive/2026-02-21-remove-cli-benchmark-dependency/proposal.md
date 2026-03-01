## Why

The benchmark toolchain still depends on a CLI execution path that is not reliably available in this repository layout, and the current Python fallback is intentionally incomplete for solver/integrator parity. This creates three recurring problems:

- Validation matrix runs produce backend-based `skipped` results instead of strict pass/fail outcomes.
- Periodic scenarios (shooting/harmonic balance) cannot be executed in Python fallback mode.
- Maintenance complexity increases because behavior diverges between CLI and Python paths.

To simplify maintenance and improve reproducibility, benchmark execution should become Python-first and rely on a single runtime path backed by the same v1 kernel capabilities.

## What Changes

- Make benchmark runners (`benchmark_runner.py`, `validation_matrix.py`, `benchmark_ngspice.py`) execute via Python runtime APIs as the primary and required path.
- Remove hard dependency on external `pulsim` executable from benchmark execution and validation workflows.
- Expand Python bindings to expose runtime-complete simulation APIs:
  - `SimulationOptions`
  - `Simulator`
  - periodic steady-state methods (shooting/harmonic balance)
  - YAML parser entrypoints that reuse the v1 C++ parser semantics
- Replace stdout/regex telemetry extraction with structured telemetry from simulation results.
- Ensure scenario matrix validations produce deterministic pass/fail/baseline outcomes, not backend-driven skips.
- Update benchmark and user docs to remove outdated CLI build paths for benchmark execution.

## Impact

- Affected specs:
  - `benchmark-suite`
  - `python-bindings` (new capability)
  - `netlist-yaml`
- Affected code:
  - `python/bindings.cpp`
  - `python/pulsim/__init__.py` and type stubs
  - `benchmarks/*.py`
  - benchmark docs and usage docs

### Breaking/Behavioral Changes

- Benchmark scripts no longer depend on `--pulsim-cli` to execute the core suite.
- Backend-specific `skipped` statuses caused by missing CLI parity are removed; unsupported features become explicit failures with actionable diagnostics.

## Success Criteria

1. `benchmark_runner.py` and `validation_matrix.py` execute without requiring a `pulsim` executable path.
2. Validation matrix no longer marks scenarios as skipped due to backend type.
3. Periodic benchmark scenarios run through Python runtime APIs and produce standard artifacts.
4. ngspice comparator keeps working with the Python runtime execution path.
5. Python API exposes full simulation configuration/runtime entrypoints needed for benchmark parity.
