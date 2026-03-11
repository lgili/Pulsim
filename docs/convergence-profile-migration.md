# Convergence Profile Migration Guide

This guide explains how to move safely between `strict`, `balanced`, and `robust` convergence profiles.

## Profile Contract

| Profile | Goal | Typical use | Risk posture |
| --- | --- | --- | --- |
| `strict` | deterministic diagnostics, bounded recovery | CI reproducibility, bug triage, regression isolation | lowest tolerance for implicit recovery |
| `balanced` | broad robustness with bounded cost | default program gate (Gate B/C) | moderate adaptive behavior |
| `robust` | maximize convergence on hard cases | deep stress debugging and extreme scenarios | highest adaptive tolerance |

## Migration Path

1. Start in `strict` and ensure baseline reproducibility and typed telemetry coverage.
2. Move to `balanced` only after Gate A is stable.
3. Enable Gate B budgets (`--phase-key gate_b`) and require stable-class non-regression.
4. Move to `robust` only for targeted stress classes and keep artifacts isolated.

## Python API Snippet

```python
import pulsim as ps

opts = ps.SimulationOptions()
opts.fallback_policy.convergence_profile = ps.ConvergenceProfile.Balanced
opts.fallback_policy.policy_dry_run = True
opts.fallback_policy.anti_overfit_check = True
opts.fallback_policy.anti_overfit_stable_budget = 0
```

## Gate Expectations by Profile

- `strict`
  - prioritize deterministic failures over aggressive recovery
  - require complete typed diagnostics for each fallback event
- `balanced`
  - require Gate B pass with stable-class guard metrics
  - allow bounded policy recommendations and dry-run comparison
- `robust`
  - allowed for targeted stress tracks
  - must still satisfy cross-class non-regression budgets before promotion

## Rollback Guidance

Rollback immediately when any of these occur:

- `policy_stable_mismatch_rate` regresses over budget
- `policy_stable_anti_overfit_violation_rate` regresses over budget
- `typed_convergence_schema_coverage_rate` drops below baseline budget

Recommended rollback sequence:

1. revert profile to `strict`
2. capture fresh artifacts (`benchmark_runner`, `stress_suite`, `kpi_gate`)
3. inspect fallback trace deltas before re-enabling `balanced`
