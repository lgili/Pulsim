## Acceptance Evidence

Date: 2026-03-07

### Validation Commands

1. OpenSpec validation:

```bash
openspec validate add-frequency-domain-ac-sweep-analysis --strict
```

Result: `Change 'add-frequency-domain-ac-sweep-analysis' is valid`

2. Targeted Python runtime tests (frequency subset):

```bash
PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py -k 'frequency_analysis'
```

Result: `12 passed`

3. Benchmark/runner/KPI tests:

```bash
PYTHONPATH=build/python pytest -q \
  python/tests/test_benchmark_python_first.py \
  python/tests/test_kpi_gate.py
```

Result: `24 passed` (11 benchmark-python-first + 13 kpi_gate)

4. Core C++ test suite:

```bash
ctest --test-dir build --output-on-failure
```

Result: `257/257 passed`

5. AC benchmark run (analytical + converter/control expected failure):

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only ac_rc_lowpass ac_control_workflow_expected_failure \
  --output-dir /tmp/pulsim_ac_out
```

Result summary:

- `ac_rc_lowpass/direct_trap`: passed, frequency mode, analytical AC errors emitted
- `ac_control_workflow_expected_failure/direct_trap`: passed via expected typed failure contract

6. AC KPI gate run:

```bash
python3 benchmarks/kpi_gate.py \
  --baseline /tmp/pulsim_ac_baseline/kpi_baseline.json \
  --bench-results /tmp/pulsim_ac_out/results.json \
  --thresholds benchmarks/kpi_thresholds_ac.yaml \
  --report-out /tmp/pulsim_ac_out/kpi_gate_report_ac.json \
  --print-report
```

Result: `overall_status: passed`, `failed_required_metrics: 0`

### Thresholds and KPI Keys

AC KPI keys now produced by benchmark tooling and evaluated by gates:

- `ac_sweep_mag_error`
- `ac_sweep_phase_error`
- `ac_runtime_p95`

`benchmarks/kpi_thresholds_ac.yaml` required gate values:

- `ac_sweep_mag_error.max_regression_abs = 0.01`
- `ac_sweep_phase_error.max_regression_abs = 0.50`
- `ac_runtime_p95.max_regression_rel = 0.10`

### Known Limitations (Current Phase)

- AC solver path currently supports linear/passive electrical devices and independent sources.
- AC path currently rejects control/event virtual components in sweep execution with deterministic typed diagnostics (`FrequencyUnsupportedConfiguration`).
- Mixed-domain converter/control small-signal linearization remains roadmap work and is intentionally not synthesized by frontend.
