## Why

The current benchmark folder is based on legacy JSON netlists and a single ngspice comparison script. It does not exercise the new kernel capabilities (solver stack, JFNK, stiff integrators, periodic steady‑state) and cannot validate future expansions. We need a modern, extensible benchmark + validation harness aligned with the v1 YAML netlist format and the new solver matrix.

## What Changes

- Migrate benchmark circuits from JSON to YAML (pulsim-v1 schema).
- Expand the benchmark corpus to cover linear, nonlinear, switching, stiff, and periodic steady‑state scenarios.
- Replace the single-purpose ngspice benchmark with a modular benchmark runner that can:
  - Execute multiple scenarios and solver configurations
  - Produce reproducible result artifacts (CSV/JSON)
  - Track performance, accuracy, and solver telemetry
- Add a kernel validation matrix runner to validate all solver/integrator combinations (current + future).
- Keep ngspice comparison optional and separated from core validation.

## Impact

- Affected specs: `benchmark-suite` (new capability).
- Affected code: `benchmarks/` tooling, `benchmarks/circuits/` data, docs.

## Success Criteria

1. Benchmark circuits are YAML‑only and aligned with `pulsim-v1` schema.
2. Validation runner can execute a matrix of solver/integrator options and report pass/fail.
3. New benchmark harness produces consistent outputs and telemetry for performance tracking.
4. Existing ngspice comparisons still work (optional) or are clearly separated.
