# Release Notes - Electrothermal Parity (2026-03-07)

## Summary

This release finalizes the backend electrothermal contract for closed-loop power converter simulations with professional-grade loss/thermal observability.

Main outcome: frontend no longer needs to synthesize thermal or loss curves. Backend exports sampled canonical channels with metadata and consistency guarantees.

## Highlights

- Datasheet switching-loss surfaces are supported in runtime:
  - `loss.model: datasheet`
  - axes: `current`, `voltage`, `temperature`
  - tables: `eon`, `eoff`, `err`
- Thermal network support expanded and stabilized:
  - `single_rc`, `foster`, `cauer`
  - stiff Cauer stability improved with implicit tridiagonal integration
- Shared sink coupling added:
  - `shared_sink_id`, `shared_sink_rth`, `shared_sink_cth`
  - multiple devices can share one sink thermal rise
- Canonical sampled channels are enforced:
  - thermal: `T(<component>)`
  - loss: `Pcond`, `Psw_on`, `Psw_off`, `Prr`, `Ploss`
- Channel metadata is first-class and required for frontend routing.
- Determinism/performance gates added in tests:
  - deterministic loss-channel ordering
  - no unplanned output-series reallocations in validated closed-loop scenario

## Frontend Impact

### Required behavior

- Consume channel metadata from `result.virtual_channel_metadata`.
- Route charts by metadata domain (`control`, `events`, `instrumentation`, `thermal`, `loss`).
- Plot backend thermal/loss channels directly.

### Forbidden behavior

- Do not generate synthetic thermal curves.
- Do not reconstruct switching losses heuristically in GUI.
- Do not “fix” backend summary inconsistencies in frontend.

## Compatibility

- Backward compatible for scalar `loss` and single-RC thermal netlists.
- Richer datasheet and staged thermal blocks are additive.

## Migration Notes

For teams moving from scalar-only loss setup:

- follow: [`docs/electrothermal-migration-scalar-to-datasheet.md`](electrothermal-migration-scalar-to-datasheet.md)
- keep this contract reference handy:
  - [`docs/electrothermal-workflow.md`](electrothermal-workflow.md)
  - [`docs/frontend-control-signals.md`](frontend-control-signals.md)

## Validation Snapshot

Validated on March 7, 2026:

- `PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py` -> `83 passed`
- `openspec validate add-electrothermal-datasheet-parity --strict` -> valid

## Known Limits

- Shared sink currently uses deterministic common RC aggregation (not a full arbitrary thermal matrix).
- Datasheet fidelity remains dependent on input table quality.
