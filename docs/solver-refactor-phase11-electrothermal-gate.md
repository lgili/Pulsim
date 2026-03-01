# Solver Refactor Phase 11 Electrothermal Gate

This document records the archived artifacts used to validate Phase 11 (loss + electrothermal integration).

## Baseline ID

- `electrothermal_phase11_2026-02-25`
- Baseline file: `benchmarks/kpi_baselines/electrothermal_phase11_2026-02-25/kpi_baseline.json`
- Artifact manifest: `benchmarks/kpi_baselines/electrothermal_phase11_2026-02-25/artifact_manifest.json`

## Archived Artifacts

- Baseline source artifacts:
  - `benchmarks/phase11_electrothermal_artifacts/benchmarks/results.json`
  - `benchmarks/phase11_electrothermal_artifacts/benchmarks/summary.json`
  - `benchmarks/phase11_electrothermal_artifacts/stress/stress_results.json`
  - `benchmarks/phase11_electrothermal_artifacts/stress/stress_summary.json`
- Gate reports:
  - `benchmarks/phase11_artifacts/reports/kpi_gate_full.json`
  - `benchmarks/phase11_artifacts/reports/kpi_gate_electrothermal.json`

## KPI Thresholds

- General gate: `benchmarks/kpi_thresholds.yaml`
- Electrothermal required gate: `benchmarks/kpi_thresholds_electrothermal.yaml`

## Required Electrothermal Metrics

- `loss_energy_balance_error`
- `thermal_peak_temperature_delta`

Both required electrothermal metrics passed without regression in the recorded Phase 11 gate run.
