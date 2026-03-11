# Benchmarks

This folder contains the YAML benchmark suite and validation runners.

## Structure

- `circuits/` — YAML netlists with embedded `benchmark` metadata.
- `benchmarks.yaml` — scenario matrix and benchmark list.
- `benchmark_runner.py` — executes benchmarks and produces results artifacts.
- `validation_matrix.py` — runs all solver/integrator combinations.
- `benchmark_ngspice.py` — Pulsim vs external SPICE parity runner (`ngspice` or `ltspice` backends).
- `stress_suite.py` — tiered stress validation runner (tiers A/B/C + pass criteria).
- `stress_catalog.yaml` — stress tier definitions, mapped cases, and acceptance criteria.
- `convergence_stress_catalog.yaml` — reproducible stress catalog with fixed simulation contract and fingerprint contract.
- `convergence_class_matrix.yaml` — canonical convergence-class case matrix (`diode_heavy`, `switch_heavy`, `zero_cross`, `magnetic_nonlinear`, `closed_loop_control`).
- `electrothermal_benchmarks.yaml` — focused matrix for electrothermal converter validation.
- `electrothermal_stress_catalog.yaml` — stress criteria for electrothermal KPI coverage.
- `kpi_gate.py` — regression gate that compares current KPIs against frozen baseline
  and evaluates runtime quantiles (`p50/p95`) on the case intersection with baseline artifacts.
- `freeze_kpi_baseline.py` — creates `kpi_baseline.json` + `artifact_manifest.json`
  with environment fingerprint and artifact hashes for provenance-safe gating.
- `kpi_thresholds.yaml` — threshold policy for required/optional KPI regressions.
- `kpi_thresholds_convergence_platform.yaml` — optional policy-gate thresholds for convergence dry-run KPIs.
- `convergence_phase_budgets.yaml` — versioned per-phase (Gate A..F/ADV) functional/performance budget contract.
- `kpi_thresholds_electrothermal.yaml` — required KPI thresholds for electrothermal gates.
- `kpi_thresholds_averaged.yaml` — required KPI thresholds for averaged-mode paired gate.
- `kpi_baselines/` — frozen baseline snapshots and artifact manifests.

## Running

```bash
# Use local build bindings (repository workflow)
export PYTHONPATH=build/python

python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out
python3 benchmarks/validation_matrix.py --output-dir benchmarks/matrix
python3 benchmarks/variable_mode_matrix.py --output-dir benchmarks/out_variable_matrix
python3 benchmarks/stress_suite.py \
  --catalog benchmarks/convergence_stress_catalog.yaml \
  --output-dir benchmarks/stress_out
python3 benchmarks/kpi_gate.py \
  --bench-results benchmarks/out/results.json \
  --stress-summary benchmarks/stress_out/stress_summary.json \
  --report-out benchmarks/out/kpi_gate_report.json \
  --print-report

# AC-focused KPI gate
python3 benchmarks/kpi_gate.py \
  --baseline benchmarks/kpi_baselines/<ac-baseline>/kpi_baseline.json \
  --bench-results benchmarks/out_ac/results.json \
  --thresholds benchmarks/kpi_thresholds_ac.yaml \
  --report-out benchmarks/out_ac/kpi_gate_report.json \
  --print-report

# Convergence-platform KPI gate (M1 policy dry-run)
python3 benchmarks/kpi_gate.py \
  --baseline benchmarks/kpi_baselines/convergence_platform_phase16_2026-03-11/kpi_baseline.json \
  --bench-results benchmarks/out/results.json \
  --class-matrix benchmarks/convergence_class_matrix.yaml \
  --phase-budget benchmarks/convergence_phase_budgets.yaml \
  --phase-key gate_b \
  --thresholds benchmarks/kpi_thresholds_convergence_platform.yaml \
  --report-out benchmarks/out/kpi_gate_convergence_platform_report.json \
  --print-report

# Freeze a new baseline snapshot from a validated run
python3 benchmarks/freeze_kpi_baseline.py \
  --baseline-id convergence_platform_phase16_2026-03-11 \
  --bench-results benchmarks/phase16_artifacts/benchmarks/results.json \
  --stress-summary benchmarks/phase16_artifacts/stress/stress_summary.json \
  --class-matrix benchmarks/convergence_class_matrix.yaml \
  --source-artifacts-root benchmarks/phase16_artifacts

# Electrothermal focused matrix + stress
python3 benchmarks/benchmark_runner.py \
  --benchmarks benchmarks/electrothermal_benchmarks.yaml \
  --output-dir benchmarks/out_electrothermal

# AC sweep focused benchmarks
python3 benchmarks/benchmark_runner.py \
  --only ac_rc_lowpass ac_control_workflow_expected_failure \
  --output-dir benchmarks/out_ac

# Averaged converter focused paired gate
python3 benchmarks/benchmark_runner.py \
  --only buck_switching_paired buck_averaged_mvp buck_averaged_expected_failure \
  --output-dir benchmarks/phase14_averaged_artifacts/benchmarks
python3 benchmarks/kpi_gate.py \
  --baseline benchmarks/kpi_baselines/averaged_converter_phase14_2026-03-07/kpi_baseline.json \
  --bench-results benchmarks/phase14_averaged_artifacts/benchmarks/results.json \
  --thresholds benchmarks/kpi_thresholds_averaged.yaml \
  --report-out benchmarks/phase14_averaged_artifacts/reports/kpi_gate_averaged.json \
  --print-report
python3 benchmarks/stress_suite.py \
  --benchmarks benchmarks/electrothermal_benchmarks.yaml \
  --catalog benchmarks/electrothermal_stress_catalog.yaml \
  --output-dir benchmarks/stress_out_electrothermal

# Compare Pulsim vs ngspice (manifest mode)
python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --output-dir benchmarks/ngspice_out

# Compare Pulsim vs LTspice (explicit executable path is required)
python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out

# Compare one YAML vs one .cir directly
python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --pulsim-netlist benchmarks/circuits/rc_step.yaml \
  --spice-netlist benchmarks/ngspice/rc_step.cir \
  --output-dir benchmarks/ngspice_single

# Local limit suite (10 circuits, fixed + variable)
python3 benchmarks/local_limit_suite.py \
  --manifest benchmarks/local_limit/benchmarks_local_limit.yaml \
  --output-dir benchmarks/out_local_limit \
  --mode both \
  --duration-scale 1.0

# Same flow via Makefile
make benchmark-local-limit BUILD_DIR=build
```

Benchmark runners are Python-first and execute through `pulsim` runtime bindings
(`YamlParser` + `Simulator`).
When a benchmark netlist omits `simulation.adaptive_timestep`, runners default to
fixed-step mode (`adaptive_timestep: false`) for deterministic comparisons.
Use `variable_mode_matrix.py` when you need the adaptive-variable benchmark gate.
Current default scope is the stiff-variable set (`stiff_rlc` scenarios) and can be expanded with `--only`.

`kpi_gate.py` validates baseline/manifest provenance in strict mode by default
and fails early when metadata or artifact hashes are inconsistent.
Use `--no-strict-provenance` only for local debugging.
Runtime latency KPIs (`runtime_p50`, `runtime_p95`) are auto-skipped when
`environment.machine_class` differs between baseline and current runner.
When `--class-matrix` is provided, `kpi_gate.py` also emits per-class KPIs:
coverage, pass rate, runtime p95, timestep rejections p95, Newton iterations p95,
and typed convergence schema coverage per class.
When `--phase-budget` and `--phase-key` are provided, the gate merges
the selected phase budget with the threshold policy before evaluating regressions.

`local_limit_suite.py` is intended for PC-local stress discovery and reports
exact failure reasons per circuit/scenario. It always supports:
- `--mode fixed|variable|both`
- `--duration-scale <factor>` to run longer than the base `tstop`
- `--only <benchmark_id ...>` to focus specific circuits
- `--max-runtime-s <seconds>` to fail runs above a runtime budget

Generate missing reference baselines:

```bash
python3 benchmarks/benchmark_runner.py --generate-baselines
```

## Output Artifacts

Each run produces:

- `results.csv` — per-scenario metrics.
- `results.json` — full structured results and metadata.
- `summary.json` — pass/fail summary.

Telemetry fields are sourced from structured simulation result objects and included in `results.json`.
Analytical `max_error` thresholds in `circuits/*.yaml` are calibrated for the current
Python-first runtime defaults (fixed-step unless explicitly overridden).
AC analytical workflows can use `benchmark.validation.type: ac_analytical`
with model `rc_lowpass` to validate `magnitude_db`/`phase_deg` against theory.
Hybrid/electrothermal KPI fields are emitted per scenario when available:
`state_space_primary_ratio`, `dae_fallback_ratio`, `loss_energy_balance_error`,
`thermal_peak_temperature_delta`, `component_coverage_rate`, `component_coverage_gap`,
`component_loss_summary_consistency_error`, `component_thermal_summary_consistency_error`,
`runtime_module_order_crc32`, `runtime_module_count_match`, `output_reallocation_total`,
`ac_sweep_mag_error`, `ac_sweep_phase_error`, `averaged_pair_fidelity_error`,
`averaged_pair_runtime_speedup_min`, `classified_fallback_events`,
`policy_dry_run_events`, `policy_recommendation_matches`,
`policy_recommendation_mismatches`, `anti_overfit_violations`,
`anti_overfit_budget_exceeded`, `typed_convergence_schema_coverage_rate`,
`policy_target_pass_rate`, `policy_target_match_rate`,
`policy_target_mismatch_rate`, `policy_stable_pass_rate`,
`policy_stable_mismatch_rate`, and `policy_stable_anti_overfit_violation_rate`.

`benchmark_ngspice.py` also emits:

- `parity_results.csv` — per benchmark/scenario parity results.
- `parity_results.json` — machine-readable parity payload (`schema_version`, backend metadata, per-observable metrics).
- `parity_summary.json` — pass/fail totals and grouped failure reasons.

For `--backend ngspice`, legacy filenames (`ngspice_results.*`, `ngspice_summary.json`) are also written for compatibility.

`stress_suite.py` emits:

- `stress_results.csv` — per scenario stress execution rows with telemetry columns.
- `stress_results.json` — tier criteria + per-tier evaluation + scenario records.
- `stress_summary.json` — overall pass/fail and per-tier status.
  - Includes reproducibility metadata with catalog hash, manifest path,
    runtime environment fingerprint, and declared reproducibility contract.

`local_limit_suite.py` emits:

- `results.csv` — one row per circuit/scenario with `pass/fail`, runtime, steps, completion ratio and failure message.
- `results.json` — full machine-readable payload with telemetry and run settings.
- `summary.json` — totals, per-scenario/per-difficulty split, and grouped failure reasons.

## Adding Benchmarks

1. Create a YAML netlist in `circuits/` with a `benchmark` block.
2. Add it to `benchmarks.yaml` and assign scenarios.
3. If you want parity for this benchmark:
   - add `ngspice_netlist: ngspice/<file>.cir` and/or `ltspice_netlist: ltspice/<file>.cir` in `benchmarks.yaml`.
4. Optional backend-specific mapping:
   - `ngspice_observables`: `{ column: "V(out)", ngspice_vector: "v(out)" }`
   - `ltspice_observables`: `{ column: "V(out)", ltspice_vector: "V(out)" }`
5. Metric thresholds can be configured under `benchmark.expectations.metrics`:
   - `max_error`, `rms_error`, `phase_error_deg`, `steady_state_max_error`, `steady_state_rms_error`
6. To model expected deterministic failures (for unsupported workflows), use:
   - `benchmark.expectations.expected_failure.diagnostic`
   - `benchmark.expectations.expected_failure.mode`
   - `benchmark.expectations.expected_failure.message_contains`
7. If using `reference` validation, add a baseline CSV under `baselines/`.
8. Optional validation window controls:
   - `benchmark.validation.ignore_initial_samples`: ignore N leading samples.
   - `benchmark.validation.start_time`: compare only from a minimum time.
9. For paired switching-vs-averaged fidelity checks in one run, use:
   - `benchmark.validation.type: paired_reference`
   - `benchmark.validation.pair_benchmark_id`
   - Optional `benchmark.validation.pair_scenario` and `benchmark.validation.pair_observable`

## Converter stress cases included

The suite now includes larger converter-focused regression cases:

- `buck_switching` (surrogate switch + freewheel diode + LC output stage)
- `buck_switching_paired` (switching reference used by averaged-pair fidelity checks)
- `buck_averaged_mvp` (paired averaged-mode buck reference against switching baseline)
- `buck_averaged_expected_failure` (typed deterministic failure for invalid averaged mapping)
- `boost_switching_complex` (boost stage with switched inductor and output filter)
- `interleaved_buck_3ph` (3-phase interleaved buck)
- `buck_mosfet_nonlinear` (nonlinear MOSFET-based buck)

Run only these converter cases:

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching buck_switching_paired buck_averaged_mvp buck_averaged_expected_failure boost_switching_complex interleaved_buck_3ph buck_mosfet_nonlinear \
  --output-dir benchmarks/out_converters
```

Or use the Make targets (includes terminal summary table):

```bash
make benchmark-converters BUILD_DIR=build
make benchmark BUILD_DIR=build LTSPICE_EXE=/Applications/LTspice.app/Contents/MacOS/LTspice
```
