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
- `electrothermal_benchmarks.yaml` — focused matrix for electrothermal converter validation.
- `electrothermal_stress_catalog.yaml` — stress criteria for electrothermal KPI coverage.
- `kpi_gate.py` — regression gate that compares current KPIs against frozen baseline
  and evaluates runtime quantiles (`p50/p95`) on the case intersection with baseline artifacts.
- `kpi_thresholds.yaml` — threshold policy for required/optional KPI regressions.
- `kpi_thresholds_electrothermal.yaml` — required KPI thresholds for electrothermal gates.
- `kpi_baselines/` — frozen baseline snapshots and artifact manifests.

## Running

```bash
# Use local build bindings (repository workflow)
export PYTHONPATH=build/python

python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out
python3 benchmarks/validation_matrix.py --output-dir benchmarks/matrix
python3 benchmarks/variable_mode_matrix.py --output-dir benchmarks/out_variable_matrix
python3 benchmarks/stress_suite.py --output-dir benchmarks/stress_out
python3 benchmarks/kpi_gate.py \
  --bench-results benchmarks/out/results.json \
  --stress-summary benchmarks/stress_out/stress_summary.json \
  --report-out benchmarks/out/kpi_gate_report.json \
  --print-report

# Electrothermal focused matrix + stress
python3 benchmarks/benchmark_runner.py \
  --benchmarks benchmarks/electrothermal_benchmarks.yaml \
  --output-dir benchmarks/out_electrothermal
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
```

Benchmark runners are Python-first and execute through `pulsim` runtime bindings
(`YamlParser` + `Simulator`).
When a benchmark netlist omits `simulation.adaptive_timestep`, runners default to
fixed-step mode (`adaptive_timestep: false`) for deterministic comparisons.
Use `variable_mode_matrix.py` when you need the adaptive-variable benchmark gate.
Current default scope is the stiff-variable set (`stiff_rlc` scenarios) and can be expanded with `--only`.

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
Hybrid/electrothermal KPI fields are emitted per scenario when available:
`state_space_primary_ratio`, `dae_fallback_ratio`, `loss_energy_balance_error`,
and `thermal_peak_temperature_delta`.

`benchmark_ngspice.py` also emits:

- `parity_results.csv` — per benchmark/scenario parity results.
- `parity_results.json` — machine-readable parity payload (`schema_version`, backend metadata, per-observable metrics).
- `parity_summary.json` — pass/fail totals and grouped failure reasons.

For `--backend ngspice`, legacy filenames (`ngspice_results.*`, `ngspice_summary.json`) are also written for compatibility.

`stress_suite.py` emits:

- `stress_results.csv` — per scenario stress execution rows with telemetry columns.
- `stress_results.json` — tier criteria + per-tier evaluation + scenario records.
- `stress_summary.json` — overall pass/fail and per-tier status.

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
6. If using `reference` validation, add a baseline CSV under `baselines/`.
7. Optional validation window controls:
   - `benchmark.validation.ignore_initial_samples`: ignore N leading samples.
   - `benchmark.validation.start_time`: compare only from a minimum time.

## Converter stress cases included

The suite now includes larger converter-focused regression cases:

- `buck_switching` (surrogate switch + freewheel diode + LC output stage)
- `boost_switching_complex` (boost stage with switched inductor and output filter)
- `interleaved_buck_3ph` (3-phase interleaved buck)
- `buck_mosfet_nonlinear` (nonlinear MOSFET-based buck)

Run only these converter cases:

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching boost_switching_complex interleaved_buck_3ph buck_mosfet_nonlinear \
  --output-dir benchmarks/out_converters
```

Or use the Make targets (includes terminal summary table):

```bash
make benchmark-converters BUILD_DIR=build
make benchmark BUILD_DIR=build LTSPICE_EXE=/Applications/LTspice.app/Contents/MacOS/LTspice
```
