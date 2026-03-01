# Solver Refactor Phase 0 Baseline

This document freezes the Phase 0 baseline used by KPI regression gates for the unified native solver refactor.

## Baseline ID

- `phase0_2026-02-23`
- Baseline file: `benchmarks/kpi_baselines/phase0_2026-02-23/kpi_baseline.json`
- Artifact manifest: `benchmarks/kpi_baselines/phase0_2026-02-23/artifact_manifest.json`

## Source Artifacts

- `benchmarks/phase8_artifacts/benchmarks/results.json`
- `benchmarks/phase8_artifacts/benchmarks/summary.json`
- `benchmarks/phase8_artifacts/parity_ltspice/parity_results.json`
- `benchmarks/phase8_artifacts/parity_ltspice/parity_summary.json`
- `benchmarks/phase8_artifacts/parity_ngspice/parity_results.json`
- `benchmarks/phase8_artifacts/parity_ngspice/parity_summary.json`
- `benchmarks/phase8_artifacts/stress/stress_results.json`
- `benchmarks/phase8_artifacts/stress/stress_summary.json`

## Frozen KPI Snapshot

- `convergence_success_rate`: `1.0`
- `runtime_p50`: `0.020824584004003555`
- `runtime_p95`: `0.21907356639276224`
- `parity_rms_error_ltspice`: `0.0022381726201627684`
- `parity_rms_error_ngspice`: `0.002220472285308209`
- `stress_tier_failure_count`: `0`

## Environment Fingerprint (baseline capture)

- Captured at UTC: `2026-02-23T10:27:20Z`
- OS: `Darwin 25.4.0 (arm64)`
- Python: `3.13.5`
- CMake: `4.2.3`
- Clang: `21.1.2` (Homebrew)
- CC: `Apple clang 17.0.0`

## Gate Configuration

- Thresholds file: `benchmarks/kpi_thresholds.yaml`
- Gate script: `benchmarks/kpi_gate.py`

Run gate locally:

```bash
python3 benchmarks/kpi_gate.py \
  --bench-results benchmarks/out/results.json \
  --stress-summary benchmarks/stress_out/stress_summary.json \
  --report-out benchmarks/out/kpi_gate_report.json \
  --print-report
```

Optional parity inputs:

```bash
python3 benchmarks/kpi_gate.py \
  --bench-results benchmarks/out/results.json \
  --parity-ltspice-results benchmarks/ltspice_out/parity_results.json \
  --parity-ngspice-results benchmarks/ngspice_out/parity_results.json \
  --stress-summary benchmarks/stress_out/stress_summary.json
```
