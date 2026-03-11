# Convergence Playbook

This playbook standardizes triage and verification when transient convergence becomes unstable.

## Scope

The workflow covers these canonical classes:

- `event_burst_zero_cross`
- `switch_chattering`
- `nonlinear_magnetic_stiffness`
- `control_discrete_stiffness`
- `linear_breakdown`
- `newton_globalization_failure`

The executable reference corpus is tracked in:

- `benchmarks/convergence_reference_examples.yaml`

## Triage Matrix

| Failure class | Primary symptom | First checks | Policy focus |
| --- | --- | --- | --- |
| `event_burst_zero_cross` | many retries near edges/zero-cross | `class_zero_cross_*`, `classified_fallback_events_p95` | bounded `DtBackoff` / event splitting |
| `switch_chattering` | repeated switch retries without progress | `class_switch_heavy_*`, `policy_target_mismatch_rate` | damping dt and anti-ping-pong guards |
| `nonlinear_magnetic_stiffness` | Newton difficulty with magnetic fixtures | `class_magnetic_nonlinear_*`, `magnetic_*` KPIs | regularization bounded by model constraints |
| `control_discrete_stiffness` | mixed-domain loop instability | `class_closed_loop_control_*`, `runtime_p95` | discrete-time stiffness backoff |
| `linear_breakdown` | singular/numerical linear solve failures | `linear_fallbacks`, fallback trace reason mix | solver-path recovery with strict guard |
| `newton_globalization_failure` | repeated Newton globalization fallback | `newton_iterations`, `timestep_rejections` | trust-region/damping escalation bounded |

## Standard Flow

1. Reproduce with canonical suite:

```bash
python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out
python3 benchmarks/stress_suite.py \
  --catalog benchmarks/convergence_stress_catalog.yaml \
  --output-dir benchmarks/stress_out
```

2. Validate reference corpus consistency:

```bash
python3 benchmarks/validate_reference_examples.py \
  --manifest benchmarks/benchmarks.yaml \
  --examples benchmarks/convergence_reference_examples.yaml
```

3. Execute class-focused references:

```bash
python3 benchmarks/run_reference_examples.py \
  --class event_burst_zero_cross \
  --output-dir benchmarks/out_reference_examples
```

4. Run Gate B (policy passive validation):

```bash
python3 benchmarks/kpi_gate.py \
  --baseline benchmarks/kpi_baselines/convergence_platform_phase16_2026-03-11/kpi_baseline.json \
  --thresholds benchmarks/kpi_thresholds_convergence_platform.yaml \
  --bench-results benchmarks/out/results.json \
  --class-matrix benchmarks/convergence_class_matrix.yaml \
  --phase-budget benchmarks/convergence_phase_budgets.yaml \
  --phase-key gate_b \
  --report-out benchmarks/out/kpi_gate_convergence_platform_report.json \
  --print-report
```

5. Inspect required Gate B metrics:

- `policy_target_pass_rate`
- `policy_target_mismatch_rate`
- `policy_stable_pass_rate`
- `policy_stable_mismatch_rate`
- `policy_stable_anti_overfit_violation_rate`

## Escalation Rules

- If target-class metrics regress and stable-class guards stay healthy: focus contextual policy tuning for target class.
- If stable-class guards regress: revert policy expansion and restore bounded behavior before any further tuning.
- If schema coverage drops (`typed_convergence_schema_coverage_rate`): block progression and fix instrumentation first.
