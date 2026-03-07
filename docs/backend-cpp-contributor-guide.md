# Backend C++ Contributor Guide

This guide is for contributors working in the runtime kernel under `core/`.

## 1. What is performance-critical

The most critical path is transient simulation:

- `core/src/v1/simulation.cpp`
- `core/src/v1/simulation_step.cpp`
- `core/src/v1/transient_services.cpp`

Execution cadence is dominated by the accepted-step loop in `Simulator::run_transient_native_impl(...)`.

If you add overhead inside that loop, total runtime cost scales with step count immediately.

## 2. Runtime architecture in practice

Core flow:

1. Parser builds `Circuit` + `SimulationOptions`.
2. `Simulator` wires a `TransientServiceRegistry`.
3. `run_transient_native_impl(...)` executes adaptive/fixed stepping.
4. Services handle assembly, nonlinear solve, segment model, recovery, losses, thermal.
5. Outputs are written to `SimulationResult` (states, events, channels, telemetry).

Key boundaries:

- `TransientService` interfaces in `core/include/pulsim/v1/transient_services.hpp`
- Runtime module contracts in `core/include/pulsim/v1/runtime_modules.hpp`
- Circuit mixed-domain execution in `RuntimeCircuit`
- Python ABI surface in `python/bindings.cpp`

## 2.1 Runtime module contracts (new)

The runtime now exposes deterministic module dependency resolution for core concerns (`events_topology`, `control_mixed_domain`, `loss_accounting`, `thermal_coupling`, `telemetry_channels`).

Rules:
- add a module by declaring `module_id`, `provides_capabilities`, `requires_capabilities`, and lifecycle hooks;
- dependency graph must resolve deterministically (no missing providers, no cycles, no duplicate module IDs);
- keep module ownership explicit for channel/telemetry outputs.
- check runtime resolution through `result.backend_telemetry.runtime_module_count` and
  `result.backend_telemetry.runtime_module_order`.

Current non-breaking extraction status:
- `events_topology` adapter (forced-switch sampled transitions): `core/src/v1/runtime_module_adapters.hpp/.cpp`
- `control_mixed_domain` adapter: `core/src/v1/runtime_module_adapters.hpp/.cpp`
- `telemetry_channels` adapter: `core/src/v1/runtime_module_adapters.hpp/.cpp`
- orchestrator delegates mixed-domain block execution/channel append via module hook (`on_sample_emit`)
- orchestrator now delegates accepted-step electrothermal service updates, canonical `P*`/`T(*)` traces, and electrothermal finalization/consistency checks through module hooks (`on_step_accepted`, `on_sample_emit`, `on_finalize`)
- channel names, metadata, and Python/YAML-facing behavior remain unchanged

## 2.2 Dependency rules and anti-patterns

Dependency rules:
- depend on capabilities, never on module IDs for runtime data flow;
- one capability must have exactly one provider;
- modules must be acyclic and deterministic (lexical tie-break in resolver);
- module state must be self-owned; shared mutation must go through typed contracts.

Anti-patterns to avoid:
- reading internal state from another module implementation;
- emitting channels in multiple modules for the same canonical signal;
- adding hidden ordering assumptions not declared in `requires_capabilities`;
- allocating large buffers inside per-step/per-sample hooks when size is predictable.

## 3. Hot-loop coding rules

Inside per-step and per-sample paths:

- Avoid repeated dynamic allocations.
- Avoid repeated string formatting (`std::string`, `std::ostringstream`) on success path.
- Avoid rebuilding summaries that can be queried incrementally.
- Pre-reserve vectors/maps whenever capacity is estimable.

Recent optimization example:

- Thermal virtual channels `T(...)` are now pre-registered once and sampled by indexed temperature access.
- Avoided calling thermal `finalize()` on every sample.
- Kept full compatibility of `thermal_summary` and `component_electrothermal` at finalize time.

## 4. Thermal and loss invariants

When thermal is active, preserve these invariants:

- Thermal channels use canonical name `T(<component_name>)`.
- `len(T(component)) == len(result.time)`.
- `final_temperature == last(T(component))`.
- `peak_temperature == max(T(component))`.
- `average_temperature == mean(T(component))`.

Generation conditions for `T(...)`:

- `simulation.enable_losses = true`
- `simulation.thermal.enabled = true`
- thermal enabled for the component

## 5. Control and signal invariants

Frontend consumers should rely on metadata, not string heuristics.

Keep `virtual_channel_metadata` complete:

- thermal channels: `domain="thermal"`, `unit="degC"`, `source_component=<name>`
- control/event/instrumentation channels: stable `domain` classification

Reference: `docs/frontend-control-signals.md`.

## 6. Quality gates for backend changes

Minimum local checks for runtime changes:

```bash
cmake --build build -j
PYTHONPATH=build/python pytest python/tests/test_runtime_bindings.py -q
ctest --test-dir build --output-on-failure -R "^v1 "
```

If you modify parser/runtime contract fields, also validate docs and Python stubs.

Module-scoped checks (recommended before full suite):

```bash
# runtime-module contracts and ordering
ctest --test-dir build --output-on-failure -R "runtime module"

# switching/loss/thermal channel path (telemetry_channels + loss/thermal adapters)
ctest --test-dir build --output-on-failure -R "v1 switching loss accumulation|v1 fixed-step resolves multiple switching events inside one macro interval|v1 backend telemetry reports topology-driven linear cache invalidation"

# Python contract and thermal/loss consistency
PYTHONPATH=build/python pytest -q \
  python/tests/test_runtime_bindings.py \
  python/tests/test_benchmark_python_first.py \
  python/tests/test_thermal_component_theoretical.py \
  python/tests/test_docs_release_config.py
```

## 7. What reviewers will prioritize

- Numerical correctness and deterministic behavior.
- Regressions in fallback/recovery telemetry.
- Contract stability for Python and frontend consumers.
- Runtime overhead added in hot path.
- Test coverage for new/changed behavior.

## 8. Doxygen standard for backend `.cpp`

Keep `.cpp` files self-documented for community contributors:

- Add `@file` header to each backend `.cpp`.
- Document non-trivial helper functions and all `Simulator` runtime entrypoints.
- Keep parameter/return semantics explicit in comments for stateful flows.
- Update docs whenever behavior contracts change.

Doxygen input includes `core/src`, so these comments appear in generated reference docs.

## 9. Safe optimization strategy

1. Confirm bottleneck with telemetry or benchmark.
2. Optimize smallest high-impact region first.
3. Keep behavior unchanged unless explicitly intended.
4. Add or update tests that lock the expected contract.
5. Document new invariants or integration expectations.
