# PR Package - Electrothermal Datasheet Parity

## Suggested PR Title

`feat(core): datasheet-grade electrothermal modeling with shared sink coupling and deterministic runtime contract`

## Scope Summary

This PR package consolidates the electrothermal parity work for backend-first physics, frontend-safe contracts, and deterministic runtime behavior.

It includes:

- Datasheet switching-loss surfaces (`Eon/Eoff/Err`) with strict parser validation.
- Thermal networks (`single_rc`, `foster`, `cauer`) with stable integration.
- Shared sink thermal coupling for multiple components.
- Canonical per-sample thermal/loss channels and metadata guarantees.
- Closed-loop buck regression (PI + PWM + losses + thermal).
- Determinism and allocation gates in runtime tests.
- Frontend integration contract and migration docs.

## Included Commits

- `a223985` - `fix: stabilize cauer thermal integration for stiff ladders`
- `77178a4` - `feat: add shared-sink electrothermal coupling and validation`
- `7034a50` - `test: add electrothermal allocation gate for closed-loop buck`
- `d7d5217` - `docs/test: complete electrothermal parity contract and gates`

## Changelog

### Added

- `ThermalDeviceConfig` shared sink fields:
  - `shared_sink_id`
  - `shared_sink_rth`
  - `shared_sink_cth`
- Runtime shared sink coupling model with deterministic sink aggregation.
- Parser/schema support for shared sink thermal descriptors.
- New migration guide from scalar to datasheet electrothermal setup.
- OpenSpec acceptance notes with executed evidence and residual risks.

### Changed

- Cauer thermal integration moved to an implicit tridiagonal (Backward Euler) solve for stiff stability.
- Thermal service now supports local device rise plus optional common sink rise composition.
- Runtime thermal and loss tests expanded to include coupling, determinism, and allocation gates.

### Fixed

- Numerical instability in stiff Cauer ladders with large timesteps.
- Parser/runtime consistency checks for mixed shared sink parameters across devices.

### Documentation

- Backend electrothermal contract clarified with explicit channel/metadata guarantees.
- GUI responsibility boundaries and forbidden behaviors documented.
- User guide linked to the new migration path.

## Backend Contract Snapshot (Frontend-Critical)

### Canonical channels

- Thermal: `T(<component_name>)`
- Losses: `Pcond(<component>)`, `Psw_on(<component>)`, `Psw_off(<component>)`, `Prr(<component>)`, `Ploss(<component>)`

### Emission conditions for thermal channels

`T(...)` is exported only when all are true:

- `simulation.enable_losses = true`
- `simulation.thermal.enabled = true`
- component thermal is enabled

### Required consistency

For each thermal-enabled component `X`:

- `len(T(X)) == len(result.time)`
- `component_electrothermal[X].final_temperature == last(T(X))`
- `component_electrothermal[X].peak_temperature == max(T(X))`
- `component_electrothermal[X].average_temperature == mean(T(X))`

### Metadata expectation

`result.virtual_channel_metadata[name]` is authoritative for routing and units.
Frontend must not infer domain by channel-name regex when metadata is available.

## Frontend Integration Checklist

### Must do

- Route channels by metadata (`domain`, `source_component`, `unit`, `component_type`).
- Treat backend channels as source of truth for thermal and loss plots.
- Validate channel-length alignment against `result.time` before plotting.
- Show thermal traces only when backend actually exports `T(...)`.

### Must not do

- Generate synthetic thermal traces when backend does not export `T(...)`.
- Reconstruct switching losses heuristically from electrical traces.
- Overwrite backend physics with UI heuristics/smoothing without explicit user-labeled post-processing.

## Validation Evidence

Executed on 2026-03-07:

- `make python -j4`
- `PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py`
  - result: `83 passed`
- `openspec validate add-electrothermal-datasheet-parity --strict`
  - result: valid

Targeted gates added/validated:

- Stiff Cauer thermal stability.
- Shared sink parser + runtime coupling behavior.
- Closed-loop buck output-allocation gate (no output reallocations in validated scenario).
- Deterministic loss-channel ordering across repeated runs.

## Key Files for Review

- `core/src/v1/transient_services.cpp`
- `core/src/v1/yaml_parser.cpp`
- `core/src/v1/simulation.cpp`
- `core/include/pulsim/v1/simulation.hpp`
- `python/tests/test_runtime_bindings.py`
- `docs/electrothermal-workflow.md`
- `docs/frontend-control-signals.md`
- `docs/electrothermal-migration-scalar-to-datasheet.md`
- `openspec/changes/add-electrothermal-datasheet-parity/tasks.md`
- `openspec/changes/add-electrothermal-datasheet-parity/notes.md`

## Residual Risks

- Shared sink is modeled as one deterministic common RC sink rise (not a full arbitrary thermal matrix network).
- Datasheet quality still depends on provided tables; parser validates shape/range but cannot infer missing physics.

## Ready-to-Copy PR Body

```markdown
## Summary
This PR delivers backend-first electrothermal parity for closed-loop power converter simulation with deterministic contracts for frontend consumption.

Main outcomes:
- datasheet switching-loss surfaces (`Eon/Eoff/Err`) in runtime;
- thermal networks (`single_rc`, `foster`, `cauer`) with stable integration;
- shared sink coupling (`shared_sink_id`) across components;
- canonical sampled channels for thermal/loss with metadata and strict consistency checks;
- expanded regression, determinism, and allocation gates;
- migration and frontend-boundary documentation.

## Validation
- `make python -j4`
- `PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py` -> `83 passed`
- `openspec validate add-electrothermal-datasheet-parity --strict` -> valid

## Frontend Contract
- thermal channel: `T(<component>)`
- loss channels: `Pcond/Psw_on/Psw_off/Prr/Ploss(<component>)`
- consume metadata from `result.virtual_channel_metadata`
- do not synthesize thermal or loss curves in GUI

## Risks
- shared sink currently uses common RC aggregation, not a full thermal matrix.
- datasheet fidelity depends on user-provided table quality.
```
