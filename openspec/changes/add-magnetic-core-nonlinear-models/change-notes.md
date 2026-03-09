# Change Notes: add-magnetic-core-nonlinear-models

## Status
In progress (stage-2 runtime slice completed: saturation + hysteresis + loss/thermal coupling).

## Evidence Checklist
- [x] Contract validation (`openspec validate --strict`)
- [x] Parser tests
- [x] Kernel tests
- [x] Python bindings tests
- [ ] Benchmark KPI gate reports
- [ ] Determinism repeat-run reports
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

## KPI Snapshot
Populate after implementation:
- `magnetic_sat_error`
- `magnetic_hysteresis_cycle_energy_error`
- `magnetic_core_loss_trend_error`
- `magnetic_determinism_drift`
- `magnetic_runtime_p95`
- `magnetic_allocation_regression`

## Known Limitations (Initial)
- Core thermal coupling currently uses simulation-level default thermal RC (`default_rth/default_cth`) for magnetic virtual rows.
- Magnetic-core state telemetry is currently limited to scalar memory-state channel (`<component>.h_state`) and loss channel (`<component>.core_loss`).
- Advanced hysteresis families (e.g., Preisach/Jiles-Atherton parameter sets) remain out of scope for this slice.
