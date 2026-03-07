# Acceptance Notes - add-electrothermal-datasheet-parity

## Evidence

Executed on 2026-03-07:

- `make python -j4`
- `PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py`
- `openspec validate add-electrothermal-datasheet-parity --strict`

Key runtime outcomes validated:

- Canonical thermal traces `T(<component>)` exported per sample with strict consistency vs `thermal_summary` and `component_electrothermal`.
- Datasheet switching-loss surfaces (`Eon/Eoff/Err`) active in runtime channels.
- Thermal networks `single_rc`, `foster`, and `cauer` validated, including stiff Cauer stability scenario.
- Shared sink coupling validated (`shared_sink_id`, `shared_sink_rth`, `shared_sink_cth`) with parser/runtime consistency checks.
- Closed-loop buck electrothermal regression remains green with PI + PWM + losses + thermal.
- Allocation gate for closed-loop buck confirms no output-series reallocations in the validated scenario.
- Loss-channel emission order is deterministic across repeated closed-loop runs.

## Residual Risks / Follow-ups

- Shared sink currently models a single common RC sink rise coupled to member devices; this is deterministic and efficient, but not a full arbitrary thermal matrix network.
- Datasheet workflow depends on user-provided table quality; parser catches shape/range errors but cannot infer missing physics.
