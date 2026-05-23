## Why

Pulsim's current segment engine is effectively unused for switching converters. In `core/src/v1/transient_services.cpp`, `DefaultSegmentModelService::build_model` marks any segment containing `IdealDiode | MOSFET | IGBT` as `not_admissible` and falls back to DAE/Newton solving. Since real power converters always contain these devices, the "primary state-space path" is never taken — the kernel runs Newton iteration on every step even when the topology is piecewise-linear and stable.

Worse, the smooth nonlinear models (`tanh`-smoothed diode, Shichman-Hodges MOSFET) force Newton to iterate inside each topology, even when no event is happening. Commercial simulators like PLECS achieve 10–100× speedup on switching converters by treating ideal switching devices as **piecewise-linear** components: each topology yields a closed linear system that is integrated without iteration; events are detected via state crossings and trigger a deterministic topology rebuild.

Without this architectural change, Pulsim cannot compete with PSIM/PLECS on the dominant power-electronics workload (switching converters).

## What Changes

### Piecewise-Linear Switching Mode for Devices
- **BREAKING (opt-in initially)**: Introduce `SwitchingMode::Ideal` for `IdealDiode`, `IdealSwitch`, `VoltageControlledSwitch`, `MOSFET`, `IGBT`. In this mode the device behaves as a two-state piecewise-linear element (closed = `Ron`, open = `Roff`) with no smooth transition.
- Existing smooth (tanh / Shichman-Hodges) models retained as `SwitchingMode::Behavioral` for backward compatibility.
- New top-level option `simulation.switching_mode: ideal | behavioral | auto` (default `auto`: `ideal` if all switching devices support it, else `behavioral`).

### State-Space Segment Engine
- Replace the stub `linear_model->E = jacobian; linear_model->A = jacobian;` with a real piecewise-linear state-space model. For each topology, derive `M ẋ + N x = b(t)` from MNA, where `M` collects reactive (capacitor/inductor) contributions and `N` collects resistive contributions.
- Topology signature = bitmask over switch states (no numeric value hashing). Cache up to 2^k state-space matrices for k switching devices, with LRU eviction beyond a configurable bound.
- Per accepted step:
  - Topology unchanged → reuse cached factorization, take a single linear solve. **No Newton iteration.**
  - Event detected (state crossing) → bisect to event time, snap step boundary, rebuild matrices for new topology.

### Event Detection Tightening
- Switch-state crossing detection becomes the only path to topology change in `Ideal` mode. Diode conduction is determined by sign of `i` when on / sign of `v` when off (PLECS-style).
- Event tolerance default tightened to `1e-12` for time bisection.

### Telemetry & Diagnostics
- `BackendTelemetry.state_space_primary_steps` and `dae_fallback_steps` actually populated and reported per run.
- New telemetry fields: `pwl_topology_cache_size`, `pwl_topology_transitions`, `pwl_event_bisections`.
- New diagnostic `SimulationDiagnosticCode::PwlTopologyExplosion` when topology bitmask exceeds configurable bound (e.g. >2^16 distinct topologies seen).

### YAML Schema
- `simulation.switching_mode` field accepted under strict validation.
- Per-device override: `components[].switching_mode` for finer control.

### Removal of Auto-Bleeders / Retry-Halving Workarounds (Cleanup)
- The Python wrapper retry layer in `python/pulsim/__init__.py` (auto-bleeders, dt-halving) was a band-aid masking the architectural problem. Behind a feature flag `PULSIM_LEGACY_RETRY_FALLBACK`, deprecate this layer once PWL engine is the default for switching circuits.

## Impact

- **Affected specs**: `kernel-v1-core` (segment engine semantics), `device-models` (switching modes), `transient-timestep` (event detection, no-Newton stable segments).
- **Affected code**: `core/include/pulsim/v1/components/{ideal_diode,ideal_switch,voltage_controlled_switch,mosfet,igbt}.hpp`, `core/include/pulsim/v1/transient_services.hpp`, `core/src/v1/transient_services.cpp`, `core/src/v1/simulation.cpp`, `core/src/v1/yaml_parser.cpp`, `python/bindings.cpp`, `python/pulsim/__init__.py`.
- **Performance target**: ≥10× speedup on `buck_switching.yaml`, `boost_switching_complex.yaml`, `interleaved_buck_3ph.yaml` benchmarks (vs current behavioral path).
- **Backward compat**: existing YAML netlists default to `auto` mode; behavior preserved unless user opts into `ideal`.

## Success Criteria

1. **Speedup**: ≥10× wall-clock improvement on switching benchmarks (buck/boost/interleaved-3φ) at equal or better accuracy vs LTspice/NgSpice parity.
2. **Convergence**: zero Newton iterations recorded in telemetry across stable topology windows (only stamp + linear solve).
3. **Accuracy**: parity with LTspice/NgSpice within 0.5% on all converter benchmarks; analytical RC/RL/RLC parity unchanged.
4. **Determinism**: identical topology trace across reruns of the same netlist on the same hardware.
5. **Cache hit rate**: >95% segment cache hit on steady-state cycles after warmup.
