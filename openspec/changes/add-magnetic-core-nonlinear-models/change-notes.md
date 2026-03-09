# Change Notes: add-magnetic-core-nonlinear-models

## Status
In progress (MVP slice implemented for saturation + core-loss telemetry).

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
- Added/updated regression tests:
  - `python/tests/test_netlist_parser.py`
    - `test_yaml_transformer_accepts_magnetic_core_loss_block`
    - `test_yaml_coupled_inductor_accepts_magnetic_core_loss_block`
  - `python/tests/test_runtime_bindings.py`
    - `test_saturable_inductor_magnetic_core_exports_core_loss_channel`
    - `test_coupled_inductor_magnetic_core_exports_core_loss_channel`
    - `test_transformer_magnetic_core_exports_core_loss_channel`

## KPI Snapshot
Populate after implementation:
- `magnetic_sat_error`
- `magnetic_hysteresis_cycle_energy_error`
- `magnetic_core_loss_trend_error`
- `magnetic_determinism_drift`
- `magnetic_runtime_p95`
- `magnetic_allocation_regression`

## Known Limitations (Initial)
- Only `model: saturation` is implemented in MVP. Hysteresis model families are not implemented yet.
- Core-loss currently exports as canonical virtual channels/metadata; it is not yet coupled into global `loss_summary` and electrothermal accumulation.
- Magnetic-core state telemetry beyond scalar `core_loss` (e.g., loop state/flux memory channels) is not yet implemented.
