# ADR: Advanced Solver Adoption (Gate ADV)

- Status: Accepted
- Date: 2026-03-11
- Change: `add-hardened-convergence-platform`
- Scope: `SUNDIALS/PETSc/KINSOL/IDA` advanced-backend evaluation track

## Context

The convergence hardening program introduced an isolated Gate ADV track to evaluate
advanced solver backends without destabilizing the native solver path.

Implemented artifacts:

- Decision contract: `benchmarks/advanced_solver_decision_matrix.yaml`
- Contract validator: `benchmarks/validate_advanced_solver_decision_matrix.py`
- Isolated prototype runner: `benchmarks/advanced_solver_prototype_runner.py`

The prototype runner compares native baseline vs candidate profile in a controlled,
opt-in flow and writes reproducible JSON/CSV artifacts.

## Decision

Do **not** adopt a new external advanced backend in the main runtime path at this stage.

Adopt the following phased decision instead:

1. Keep advanced solver work isolated in Gate ADV tooling.
2. Use `sundials_ida_direct` as the primary candidate for deeper evaluation.
3. Defer production integration until hard constraints are met with reproducible evidence.

This keeps mainline behavior stable while preserving a concrete path to future adoption.

## Objective Criteria

Decision constraints from the benchmark contract:

- `min_success_rate_gain_abs >= 0.02`
- `max_runtime_regression_rel <= 0.15`
- `max_memory_regression_rel <= 0.20`
- `min_portability_score >= 0.70`
- `max_maintenance_cost_score <= 6.0`

Current portability/maintenance estimates used by the prototype track:

- `sundials_ida_direct`: portability `0.75`, maintenance `5.0` (passes static constraints)
- `sundials_cvode`: portability `0.75`, maintenance `7.5` (fails maintenance threshold)
- `sundials_arkode`: portability `0.75`, maintenance `7.5` (fails maintenance threshold)
- `sundials_kinsol`: portability `0.75`, maintenance `7.5` (fails maintenance threshold)
- `petsc_snes_ksp`: portability `0.65`, maintenance `7.5` (fails portability and maintenance thresholds)

Therefore, `sundials_ida_direct` is the only candidate that currently clears
non-performance constraints and is the correct next target for Gate ADV evidence.

## Rationale

- Mainline robustness work must avoid cross-class regressions.
- External solver integration has high portability and maintenance risk.
- Isolated prototype execution gives measurable evidence without impacting default users.
- The new contract and runner are sufficient to run decision-focused experiments
  before committing to invasive runtime integration.

## Consequences

Positive:

- No regression risk to default native backend behavior.
- Reproducible evaluation path is now codified and test-covered.
- Candidate selection is objective and auditable.

Trade-offs:

- No immediate robustness gain from external backends in production runs.
- Additional evaluation cycle is required before integration.

## Reproducible Procedure

```bash
python3 benchmarks/validate_advanced_solver_decision_matrix.py \
  --matrix benchmarks/advanced_solver_decision_matrix.yaml

python3 benchmarks/advanced_solver_prototype_runner.py \
  --candidate sundials_ida_direct \
  --output-dir benchmarks/out_advanced_solver \
  --enforce-hard-constraints
```

Generated artifacts:

- `benchmarks/out_advanced_solver/advanced_solver_prototype_report.json`
- `benchmarks/out_advanced_solver/advanced_solver_prototype_results.csv`

## Exit Criteria To Revisit This ADR

Promote to adoption consideration only when:

1. Gate ADV hard constraints pass on the objective matrix.
2. Results are reproducible under equivalent environment fingerprints.
3. Build/distribution portability impact is documented for Linux/macOS/Windows.
4. Maintenance ownership for the integrated backend path is explicitly assigned.
