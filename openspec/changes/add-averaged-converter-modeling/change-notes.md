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
- Added dedicated averaged KPI baseline + threshold policy:
  - `benchmarks/kpi_baselines/averaged_converter_phase14_2026-03-07`
  - `benchmarks/kpi_thresholds_averaged.yaml`
- Added averaged contract and frontend-boundary docs:
  - `docs/averaged-converter-modeling.md`
  - updates in `docs/netlist-format.md`, `docs/frontend-control-signals.md`, `docs/user-guide.md`
- Added runnable YAML + Python examples:
  - `examples/10_buck_averaged_mvp_backend.yaml`
  - `examples/run_buck_averaged_mvp.py`

### Validation Commands Executed
- `openspec validate add-averaged-converter-modeling --strict`
- `PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py -k 'averaged_converter or averaged_buck'`
- `PYTHONPATH=build/python pytest -q python/tests/test_benchmark_python_first.py -k 'averaged or expected_failure or periodic_benchmark_shooting_default or frequency_analysis'`
- `PYTHONPATH=build/python pytest -q python/tests/test_kpi_gate.py -k 'kpi_gate or ac_sweep or averaged_pair'`
- `PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py --only buck_switching_paired buck_averaged_mvp buck_averaged_expected_failure --output-dir benchmarks/phase14_averaged_artifacts/benchmarks`
- `python3 benchmarks/kpi_gate.py --baseline benchmarks/kpi_baselines/averaged_converter_phase14_2026-03-07/kpi_baseline.json --bench-results benchmarks/phase14_averaged_artifacts/benchmarks/results.json --thresholds benchmarks/kpi_thresholds_averaged.yaml --report-out benchmarks/phase14_averaged_artifacts/reports/kpi_gate_averaged.json --print-report`

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
- Averaged KPI gating is enforced in dedicated thresholds/baseline files. General `kpi_thresholds.yaml` keeps averaged metrics optional.

### CI/Artifacts
- `benchmarks/phase14_averaged_artifacts/benchmarks/results.json`
- `benchmarks/phase14_averaged_artifacts/benchmarks/summary.json`
- `benchmarks/phase14_averaged_artifacts/reports/kpi_gate_averaged.json`
