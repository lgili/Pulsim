# Change Notes: add-magnetic-core-nonlinear-models

## Status
In progress (stage-2 runtime + Python/benchmark extension slices completed).

## Evidence Checklist
- [x] Contract validation (`openspec validate --strict`)
- [x] Parser tests
- [x] Kernel tests
- [x] Python bindings tests
- [x] Benchmark KPI gate reports
- [x] Determinism repeat-run reports
- [x] Known limitations documented

## Implemented Slice (2026-03-09)
- Canonical `component.magnetic_core` support accepted in parser for:
  - `saturable_inductor`
  - `coupled_inductor`
  - `transformer`
- Strict typed validation remains active (`PULSIM_YAML_E_MAGNETIC_CONFIG_INVALID`) for:
  - unsupported component families
  - unsupported magnetic model family (`model != saturation`)
  - invalid numeric ranges for magnetic-core parameters.
- Runtime exports deterministic magnetic loss channels with metadata:
  - `Lsat.core_loss` (`saturable_inductor`)
  - `Kmag.core_loss` (`coupled_inductor`)
  - `Tmag.core_loss` (`transformer`)
  - metadata contract: `domain="loss"`, `unit="W"`, `source_component=<component_name>`.
- Added optional frequency-sensitive loss coefficient:
  - `magnetic_core.core_loss_freq_coeff` (`>= 0`)
  - runtime multiplier uses `|di/dt|` proxy to increase loss under faster current dynamics.
- Added canonical policy/initialization fields:
  - `magnetic_core.loss_policy` (`telemetry_only` | `loss_summary`)
  - `magnetic_core.i_equiv_init` (`>= 0`) for deterministic initialization of frequency-sensitive loss dynamics.
- Added hysteresis model family support (`magnetic_core.model: hysteresis`) with deterministic state semantics:
  - explicit init parameter `magnetic_core.hysteresis_state_init`
  - optional parameters `hysteresis_band`, `hysteresis_strength`, and `hysteresis_loss_coeff`
  - state updates commit only when accepted-time advances (no same-timestamp re-commit).
- Implemented loss-summary coupling path:
  - when `simulation.enable_losses: true` and `loss_policy: loss_summary`,
    backend appends deterministic summary row `<component>.core` sourced from `<component>.core_loss` integration.
- Implemented thermal coupling path for magnetic core-loss rows:
  - when losses + thermal are enabled, backend exports `T(<component>.core)` channels
  - updates `thermal_summary.device_temperatures` and `component_electrothermal` for `<component>.core`
  - enforces deterministic consistency against thermal channels.
- Implemented runtime fail-fast guard for invalid magnetic loss policy in runtime metadata path:
  - marks run as failed with machine-readable reason `magnetic_core_runtime_invalid`.
- Added/updated regression tests:
  - `python/tests/test_netlist_parser.py`
    - `test_yaml_transformer_accepts_magnetic_core_loss_block`
    - `test_yaml_coupled_inductor_accepts_magnetic_core_loss_block`
    - `test_yaml_rejects_negative_magnetic_core_freq_coeff`
  - `python/tests/test_runtime_bindings.py`
    - `test_saturable_inductor_magnetic_core_exports_core_loss_channel`
    - `test_coupled_inductor_magnetic_core_exports_core_loss_channel`
    - `test_transformer_magnetic_core_exports_core_loss_channel`
    - `test_saturable_inductor_core_loss_freq_coefficient_increases_loss_with_frequency`
    - `test_magnetic_core_loss_channel_is_deterministic_across_repeated_runs`
    - `test_magnetic_core_loss_policy_loss_summary_exports_summary_row`
    - `test_magnetic_core_loss_summary_row_is_deterministic_across_repeated_runs`
  - `core/tests/test_v1_kernel.cpp`
    - magnetic telemetry test extended with:
      - coupled-inductor `core_loss` power-law assertion
      - transformer `core_loss` power-law assertion
      - frequency-coefficient trend assertion (`high f` > `low f` average loss)
- Added docs/examples/tutorial assets:
  - `docs/magnetic-core-backend-frontend-contract.md`
  - `examples/magnetic_core_saturation_freq_loss.yaml`
  - `examples/run_magnetic_core_saturation_freq_loss.py`
  - `examples/notebooks/35_magnetic_core_mvp_tutorial.ipynb`
  - migration note updates in `docs/migration-guide.md`
- Added typed Python magnetic-core configuration surface for programmatic virtual-component setup:
  - `python/pulsim/magnetic_core.py`
  - API export/stubs updates in `python/pulsim/__init__.py` and `python/pulsim/__init__.pyi`
  - packaging/install integration in `python/CMakeLists.txt`
- Added benchmark fixture catalog and validation models for magnetic-core regression/KPI workflows:
  - fixture netlists in `benchmarks/circuits/magnetic_core_*`
  - fixture manifest `benchmarks/magnetic_core_benchmarks.yaml`
  - magnetic KPI threshold policy `benchmarks/kpi_thresholds_magnetic_core.yaml`
  - benchmark runner magnetic validations/telemetry:
    - `magnetic_saturation`
    - `magnetic_hysteresis`
    - trend/determinism tags and telemetry extraction
  - KPI aggregation in `benchmarks/kpi_gate.py` for:
    - `magnetic_sat_error`
    - `magnetic_hysteresis_cycle_energy_error`
    - `magnetic_core_loss_trend_error`
    - `magnetic_determinism_drift`
    - `magnetic_runtime_p95`
    - `magnetic_allocation_regression`
- Extended benchmark/regression tests:
  - `python/tests/test_magnetic_core_python_config.py`
  - `python/tests/test_magnetic_core_theory.py`
  - `python/tests/test_benchmark_python_first.py`
  - `python/tests/test_kpi_gate.py`
- Added magnetic benchmark acceptance artifacts + KPI baseline/gate evidence:
  - artifacts: `benchmarks/phase15_magnetic_core_artifacts/benchmarks/results.json`
  - gate report: `benchmarks/phase15_magnetic_core_artifacts/reports/kpi_gate_magnetic_core.json`
  - baseline: `benchmarks/kpi_baselines/magnetic_core_phase15_2026-03-09/*`
  - CI wiring: `.github/workflows/ci.yml` now runs a dedicated magnetic-core KPI gate step.

## KPI Snapshot
From `benchmarks/phase15_magnetic_core_artifacts/reports/kpi_gate_magnetic_core.json`:
- `magnetic_sat_error`: `2.3789e-13`
- `magnetic_hysteresis_cycle_energy_error`: `1.3401e-11`
- `magnetic_core_loss_trend_error`: `0.0`
- `magnetic_determinism_drift`: `1.0658e-14`
- `magnetic_runtime_p95`: `9.9858e-02 s`
- `magnetic_allocation_regression`: `0.0`

## Known Limitations (Initial)
- Core thermal coupling currently uses simulation-level default thermal RC (`default_rth/default_cth`) for magnetic virtual rows.
- Magnetic-core state telemetry is currently limited to scalar memory-state channel (`<component>.h_state`) and loss channel (`<component>.core_loss`).
- Advanced hysteresis families (e.g., Preisach/Jiles-Atherton parameter sets) remain out of scope for this slice.
