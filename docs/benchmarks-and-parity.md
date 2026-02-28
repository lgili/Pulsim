# Benchmarks and Parity

Use these workflows to guard runtime, numerical stability, and external parity.

## 1) Standard Benchmark Suite

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --output-dir benchmarks/out
```

For switching-heavy converter cases:

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching boost_switching_complex interleaved_buck_3ph buck_mosfet_nonlinear \
  --output-dir benchmarks/out_converters
```

## 2) Solver/Integrator Validation Matrix

```bash
PYTHONPATH=build/python python3 benchmarks/validation_matrix.py \
  --output-dir benchmarks/matrix
```

This matrix is useful to detect solver regressions across fixed/variable timestep modes.

## 3) External Parity

### ngspice

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --output-dir benchmarks/ngspice_out
```

### LTspice

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out
```

## 4) Tiered Stress Suite

```bash
PYTHONPATH=build/python python3 benchmarks/stress_suite.py \
  --output-dir benchmarks/stress_out
```

Electro-thermal stress variant:

```bash
PYTHONPATH=build/python python3 benchmarks/stress_suite.py \
  --benchmarks benchmarks/electrothermal_benchmarks.yaml \
  --catalog benchmarks/electrothermal_stress_catalog.yaml \
  --output-dir benchmarks/stress_out_electrothermal
```

## 5) GUI-to-Backend Parity Gate

```bash
PYTHONPATH=build/python pytest -q python/tests/test_gui_component_parity.py
PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py
./build-test/core/pulsim_simulation_tests "[v1][yaml][gui-parity]"
```

## 6) KPI Regression Gate

```bash
python3 benchmarks/kpi_gate.py \
  --bench-results benchmarks/out/results.json \
  --stress-summary benchmarks/stress_out/stress_summary.json \
  --report-out benchmarks/out/kpi_gate_report.json \
  --print-report
```

Baseline and threshold files:

- `benchmarks/kpi_baselines/phase0_2026-02-23/kpi_baseline.json`
- `benchmarks/kpi_thresholds.yaml`

## Artifact Contract

These files are the recommended CI contract:

- benchmark: `results.csv`, `results.json`, `summary.json`
- parity: `parity_results.csv`, `parity_results.json`, `parity_summary.json`
- stress: `stress_results.csv`, `stress_results.json`, `stress_summary.json`

Primary hybrid/electro-thermal KPI keys emitted in benchmark outputs:

- `state_space_primary_ratio`
- `dae_fallback_ratio`
- `loss_energy_balance_error`
- `thermal_peak_temperature_delta`
