# Tasks

## Phase 1 — Spec & Data Migration

- [x] Define `benchmark-suite` capability spec and data model.
- [x] Migrate existing JSON circuits to YAML (pulsim-v1 schema).
- [x] Add YAML fixtures for nonlinear, switching, stiff, and periodic steady‑state cases.
- [x] Deprecate JSON benchmarks and update docs/reports.

## Phase 2 — Benchmark Runner

- [x] Implement a modular benchmark runner (scenarios + solver config matrix).
- [x] Standardize output artifacts (CSV/JSON + metadata).
- [x] Emit telemetry (iterations, steps, residuals, solver info).
- [x] Optional ngspice comparator kept separate.

## Phase 3 — Validation Matrix

- [x] Implement validation matrix runner for solver/integrator combinations.
- [x] Define pass/fail criteria per circuit and capability.
- [x] Add summary reports and baseline tracking.

## Phase 4 — Documentation

- [x] Update `benchmarks/` README/reporting.
- [x] Document how to add new benchmark circuits and validation cases.
