## Acceptance Evidence

### Implemented Scope
- Added canonical `simulation.averaged_converter` contract parsing and strict diagnostics.
- Added kernel averaged transient path (MVP: buck topology) with deterministic mapping checks.
- Added typed Python bindings for averaged enums/options/diagnostics.
- Added paired benchmark coverage:
  - `buck_switching_paired`
  - `buck_averaged_mvp`
  - `buck_averaged_expected_failure`
- Added benchmark runner support for `validation.type: paired_reference`.
- Added KPI metrics for averaged-pair fidelity and runtime speedup.

### Validation Commands Executed
- `openspec validate add-averaged-converter-modeling --strict`
- `PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py -k 'averaged_converter or averaged_buck'`
- `PYTHONPATH=build/python pytest -q python/tests/test_benchmark_python_first.py -k 'averaged or expected_failure or periodic_benchmark_shooting_default or frequency_analysis'`
- `PYTHONPATH=build/python pytest -q python/tests/test_kpi_gate.py -k 'kpi_gate or ac_sweep or averaged_pair'`
- `PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py --only buck_switching_paired buck_averaged_mvp buck_averaged_expected_failure --output-dir /tmp/pulsim_avg_bench_out`

### KPI/Telemetry Keys Added
- `averaged_pair_case`
- `averaged_pair_group_crc32`
- `averaged_pair_role_switching`
- `averaged_pair_role_averaged`
- `paired_reference_case`
- `paired_reference_group_crc32`
- `averaged_pair_fidelity_error`
- `averaged_pair_runtime_speedup_min`
- `averaged_pair_runtime_speedup_mean`

### Known Limitations (MVP)
- Runtime topology support is currently limited to buck.
- Envelope check is CCM-oriented and uses inductor-current threshold.
- `paired_reference` validation requires the paired switching scenario to run in the same benchmark execution.
- KPI thresholds for averaged metrics are currently optional in `benchmarks/kpi_thresholds.yaml` pending frozen baseline adoption in CI.

### CI/Artifacts
- Local benchmark artifact example:
  - `/tmp/pulsim_avg_bench_out/results.json`
