## Gates & Definition of Done

- [ ] G.1 Speedup: ≥10× on buck/boost/interleaved-3φ benchmarks vs current behavioral path (PULSIM_CI_BUILD=0, Release+LTO)
- [ ] G.2 Accuracy: ≤0.5% parity vs LTspice on the same benchmarks; analytical RC/RL/RLC unchanged
- [ ] G.3 Determinism: identical topology bitmask trace and identical step trace across 3 reruns on same hardware
- [ ] G.4 Convergence: zero Newton iterations in stable-topology windows (assert via telemetry)
- [ ] G.5 Cache: ≥95% topology cache hit rate after warmup on periodic switching benchmarks
- [ ] G.6 Compatibility: existing YAML netlists with `auto` mode produce results within numerical noise of pre-change baseline

## Phase 1: Foundation — PWL device contract
- [x] 1.1 Define `SwitchingMode` enum (`Ideal | Behavioral | Auto`) in `core/include/pulsim/v1/components/base.hpp`
- [x] 1.2 Add `SwitchingMode` field to `IdealDiode`, `IdealSwitch`, `VoltageControlledSwitch`, `MOSFET`, `IGBT`
- [x] 1.3 Define PWL two-state contract: `pwl_state()`, `commit_pwl_state(bool)`, `should_commute(PwlEventContext) const` (per-device predicate; `IdealSwitch` is externally commanded and never auto-commutes)
- [x] 1.4 Move tanh smoothing in `IdealDiode` and `VoltageControlledSwitch` behind `SwitchingMode::Behavioral` only (sharp PWL when `Ideal`); split `MOSFET` Shichman-Hodges from `Ideal` two-state stamp; same for `IGBT`
- [x] 1.5 Unit tests in `core/tests/test_switching_mode.cpp` (18 cases / 68 assertions) covering enum, resolve helper, `supports_pwl_v` trait, per-device Ideal vs Behavioral stamping, and `should_commute` semantics; full suite (258 tests / 3989 assertions) passes with no regression
- [x] 1.6 Document switching mode contract in component header comments
- [x] 1.7 (Cleanup) Replace `mutable bool is_on_` with explicit `bool pwl_state_` in `IdealDiode`, `MOSFET`, `IGBT`; Ideal stamp methods now `const` (no state mutation in stamping path)

## Phase 2: State-space segment model
- [x] 2.1 Implemented `Circuit::assemble_state_space(M, N, b, t)` separating reactive (M) from resistive (N) contributions and time-varying sources (b). Inline in `runtime_circuit.hpp`; mirrors `assemble_residual_hb()` conventions for branch-row, KCL, and source signs.
- [x] 2.2 Replaced `DefaultSegmentModelService::build_model` stub with Tustin discretization `E = M + dt/2·N`, `A = M − dt/2·N`, `c = dt/2·(b_now + b_next)`. Admissibility now gated by `Circuit::all_switching_devices_in_ideal_mode()` (default Behavioral) AND `pwl_state_space_supports_all_devices()` (transformers excluded for now).
- [x] 2.3 Topology signature = `Circuit::pwl_topology_bitmask()` mixed with structural metadata (node count, branch count, switch count). The legacy O(nnz) numeric-value hash is dropped from the build path; the stepper still uses a similar fingerprint as a within-step check until Phase 0.3 (`refactor-linear-solver-cache`) overhauls the cache key.
- [x] 2.4a New helper `Circuit::set_pwl_state(name, on)` writes directly to `pwl_state_` for any switching device — the path the segment engine will use to commit event transitions.
- [x] 2.4b New helper `Circuit::set_switching_mode_for_all(mode)` to opt-in to Ideal mode without per-device boilerplate.
- [x] 2.5 Dropped the segment-stepper `residual_not_improved` check (legacy contract assumed `c = −residual_at_x_now`; incompatible with the Tustin form). Replaced with `‖E·x_next − rhs‖` consistency check that validates the linear solve directly.
- [x] 2.6 Tests in `core/tests/test_pwl_state_space.cpp` (12 cases / 56 assertions) covering RC, RL, capacitor, switch, diode, PWM time-varying source, Tustin-recipe identities, and an end-to-end Tustin RC step matching the analytical first-order response within 1% from a consistent initial condition.
- [x] 2.7 Updated legacy v1 kernel tests in `core/tests/test_v1_kernel.cpp` that asserted the broken stub contract (`E = jacobian`, `c = −residual`, `segment_solution == DAE_solution`) to assert the new Tustin contract.

> **Deferred to follow-up PR**: LRU bound on topology cache + `PwlTopologyExplosion` diagnostic (originally Phase 2.5/2.6) belong with the linear-solver cache rework in `refactor-linear-solver-cache`. That change owns the cache eviction policy and key design end-to-end. The segment-model topology cache is unbounded for now.

## Phase 3: Step engine — no-Newton stable path
- [x] 3.1 The Tustin step is implemented inside `DefaultSegmentStepperService::try_advance` (single linear solve of `E·x_next = A·x_now + c`), populated by Phase 2's `build_model`. The `Simulator::solve_step` lambda `solve_segment_primary` returns the segment outcome as the step's `NewtonResult` with `iterations = 1` (representing the linear solve, not a Newton iteration).
- [x] 3.2 The Simulator step path already calls `segment_model->build_model` and `segment_stepper->try_advance` from `solve_step` (`core/src/v1/simulation_step.cpp:139-159`); when admissible the step is committed via the segment-primary path and `last_step_solve_path_ = StepSolvePath::SegmentPrimary`. Telemetry counters `state_space_primary_steps` and `dae_fallback_steps` are incremented per accepted step in `collect_step_solve_telemetry` (`core/src/v1/simulation.cpp:759`).
- [x] 3.3 Contract validated by `core/tests/test_pwl_segment_primary.cpp`:
  - Passive RC: `state_space_primary_steps == total_steps`, `dae_fallback_steps == 0`, `newton_iterations_total ≤ total_steps` (each segment step contributes `iterations = 1`, the linear solve).
  - Switching device pinned to `SwitchingMode::Ideal` with closed PWL state: same property.
  - Switching device left at default `Auto/Behavioral`: `state_space_primary_steps == 0`, `dae_fallback_steps == total_steps`, `segment_non_admissible_steps == total_steps` — backward-compatible regression guard.
- [x] 3.4 LTE controller / adaptive timestep preserved: a fourth case in the contract suite enables `adaptive_timestep = true` and asserts segment-primary still serves every accepted step. Cache hit rate degrades because the matrix hash includes `dt` (will be fixed by `refactor-linear-solver-cache`).
- [x] 3.5 Side note for users: `apply_auto_transient_profile` rewrites `Integrator::Trapezoidal` → `Integrator::TRBDF2` for switching circuits, and TRBDF2's multi-stage Newton path bypasses `solve_segment_primary`. To exercise the segment engine on switching circuits today, set `options.integrator = Integrator::BDF1` (or any non-Trapezoidal/non-TRBDF2 integrator) explicitly. The integration with TRBDF2 is left to a follow-up change once the segment engine learns multistage support.

## Phase 4: Event detection (PWL semantics)
- [x] 4.1 Diode `should_commute` already implemented in Phase 1 ([`ideal_diode.hpp`](../../../core/include/pulsim/v1/components/ideal_diode.hpp)): on-state commutes when `i < −hysteresis`, off-state commutes when `v > +hysteresis`. Phase 4 wires the predicate into the Simulator step loop via `Circuit::scan_pwl_commutations(x_next)`.
- [x] 4.2 MOSFET / IGBT use a gate-threshold predicate (`Vgs > vth ± hysteresis`) for the *channel* state. Body-diode-as-embedded-PWL-diode behaviour is deferred to `add-catalog-device-models` (catalog tier) — that change introduces the body-diode sub-component with its own state and embedded `should_commute`. Phase 4's MOSFET/IGBT predicate covers the dominant gate-driven switching transition.
- [x] 4.3 `VoltageControlledSwitch::should_commute` checks the control-node voltage against `v_threshold` with hysteresis. Wired through the same `scan_pwl_commutations` path.
- [ ] 4.4 Bisection-to-event time deferred to a follow-up change. Phase 4 lands a *first-order* event scheduler: events are recognized at the end of each accepted step and committed for the next step. The Tustin step's accuracy in the event-adjacent step is reduced (the device assumed the old state across the whole `dt` even though the crossing happened mid-step), but the post-event topology is correct and subsequent steps converge as expected. Bisection lifts this to second-order accuracy at events; the existing `find_switch_event_time` infrastructure is the natural place to extend.
- [x] 4.5 At each accepted step, `Simulator::process_accepted_step_events` calls `circuit_.scan_pwl_commutations(x_next)`, applies the events via `circuit_.commit_pwl_commutations(...)`, records `SimulationEvent` entries, and fires the user `event_callback`. The next step's `build_model` rebuilds the segment under the new topology bitmask automatically — `pwl_topology_bitmask()` reflects the post-commit state.

### Tests delivered for Phase 4
- 4 header-only Catch2 tests in `test_pwl_state_space.cpp` (14 assertions) for `scan_pwl_commutations` and `commit_pwl_commutations`:
  - Diode flips state on voltage reversal / current reversal.
  - Devices in Auto/Behavioral mode are filtered out of the scan.
  - VCSwitch follows control-voltage threshold.
  - Idempotent commit.
- 1 integration-level test in `test_pwl_segment_primary.cpp` (10 assertions): pulse source drives a VCSwitch from off to closed during a 4 µs transient; verifies that the topology bitmask flips, an `SimulationEvent::SwitchOn` is recorded near the pulse edge, and the segment-primary path serves every accepted step.

## Phase 5: YAML & Python surface
- [x] 5.1 YAML parser maps `simulation.switching_mode: auto | ideal | behavioral` (plus aliases `pwl` for ideal and `smooth` for behavioral) into `SimulationOptions::switching_mode`. Implemented in [yaml_parser.cpp](../../../core/src/v1/yaml_parser.cpp).
- [ ] 5.2 Per-device `components[].switching_mode` override deferred to follow-up. Today the user opts in either at the simulation level (Phase 5.1) or via the C++/Python API on individual devices (`Circuit::set_switching_mode_for_all` or per-device `set_switching_mode`). YAML per-component override requires extending the component-parser dispatch table for every device type.
- [x] 5.3 pybind11 exposes:
  - `pulsim.SwitchingMode` enum (Auto / Ideal / Behavioral) — registered before Circuit/SimulationOptions so default arguments referencing it resolve at module init time.
  - `SimulationOptions.switching_mode` read/write field.
  - `Circuit.set_pwl_state(name, on)`, `set_switching_mode_for_all(mode)`, `set_default_switching_mode(mode)`, `default_switching_mode()`, `pwl_topology_bitmask()`, `pwl_switching_device_count()`, `all_switching_devices_in_ideal_mode(circuit_default=Behavioral)`.
  - Re-exported from [`python/pulsim/__init__.py`](../../../python/pulsim/__init__.py).
- [x] 5.4 Strict validation: unknown values for `simulation.switching_mode` produce a deterministic diagnostic (`Invalid simulation.switching_mode: 'X'. Expected one of: auto, ideal, behavioral.`). Also added `switching_mode` to the validate_keys allow-list for the simulation block.

### Plumbing added in Phase 5
- `SimulationOptions::switching_mode` field (default `SwitchingMode::Auto`) in [simulation.hpp](../../../core/include/pulsim/v1/simulation.hpp).
- `Circuit::default_switching_mode_` member + getter/setter in [runtime_circuit.hpp](../../../core/include/pulsim/v1/runtime_circuit.hpp). The Simulator's constructor pushes `options_.switching_mode` into this slot via `circuit_.set_default_switching_mode(...)` before constructing the service registry.
- `DefaultSegmentModelService::build_model` and `Simulator::process_accepted_step_events` consume `circuit_.default_switching_mode()` as the resolution argument to `all_switching_devices_in_ideal_mode(...)` and `scan_pwl_commutations(...)`. Auto-mode devices now resolve up to Ideal whenever the user sets `options.switching_mode = Ideal` — no per-device opt-in required.

### Tests delivered for Phase 5
- 4 C++ test cases in [test_pwl_segment_primary.cpp](../../../core/tests/test_pwl_segment_primary.cpp) (14 assertions):
  - `SimulationOptions.switching_mode` propagates through `Simulator(...)` to `Circuit::default_switching_mode()` for Auto / Ideal / Behavioral.
  - With `options.switching_mode = Ideal`, an `Auto`-mode device resolves up and `all_switching_devices_in_ideal_mode(SwitchingMode::Ideal)` returns true.
  - YAML parser maps `auto`, `ideal`, `pwl`, `behavioral`, `smooth` correctly.
  - Strict-mode YAML parser emits a deterministic diagnostic for `switching_mode: bogus`.
- Manual Python smoke (verified locally) — `pulsim.SwitchingMode`, `SimulationOptions.switching_mode`, all new `Circuit` methods, and YAML round-trip via `pulsim.YamlParser` all work end-to-end through the Python wrapper.

## Phase 6: Telemetry & observability
- [x] 6.1 `BackendTelemetry.state_space_primary_steps` and `dae_fallback_steps` populated correctly (Phase 3 wiring; gate validated by `test_pwl_segment_primary.cpp`).
- [x] 6.2 Added `pwl_topology_transitions` (count of accepted-step boundaries where the PWL switch bitmask changed) and `pwl_event_commutations` (count of individual device commutations committed) to `BackendTelemetry`. Wired into `Simulator::process_accepted_step_events`. Exposed via pybind11. Asserted in the Phase 4 integration test (pulse-driven VCSwitch transient produces exactly 1 transition / 1 commutation).
  - Deferred: `pwl_topology_cache_size` (depends on the LRU bound work in `refactor-linear-solver-cache`), `pwl_event_bisections` (counter for Phase 4.4 bisection-to-event work; currently always 0 under first-order scheduler).
- [ ] 6.3 `SimulationResult.message` summary of topology trace under verbose mode — deferred to a follow-up small change. Today the topology trace can be reconstructed from `result.events` (each PWL commutation appears as a `SimulationEvent` with `description == "pwl_commutation"`) and from `pwl_topology_transitions` / `pwl_event_commutations` counters.

## Phase 7: Benchmarks & deprecation
- [x] 7.1 First-pass `pwl_buck` benchmark landed in [test_pwl_speedup_benchmark.cpp](../../../core/tests/test_pwl_speedup_benchmark.cpp): same buck topology (Vin → VCSwitch → L1 → C1+Rload, with free-wheeling diode) instantiated twice, once with `switching_mode = Behavioral` and once with `Ideal`, both pinned to `Integrator::BDF1` so they traverse the same `solve_step → solve_segment_primary` entrypoint and the cost difference is purely Newton-DAE vs PWL state-space. Boost/interleaved-3φ variants deferred to a follow-up benchmark expansion.
- [x] 7.2 Speedup gate validated empirically: on the buck reference scenario (5 PWM cycles @ 100 kHz, dt = 100 ns), Ideal-mode wall-clock is **~240× faster** than Behavioral (4.6 ms vs 1.09 s on the development machine), comfortably above the proposal's ≥10× target. The CI assertion is loose (`ideal ≤ 3 × behavioral`) to absorb runner noise; the headline metrics are reported via Catch2 INFO so regression triage has full numbers without strict thresholds. Output-voltage equivalence asserted within ±20% (loose because the run is short — only 5 cycles — and Phase 4's first-order event scheduler smears commutation timing by up to one dt vs Behavioral's bisected step splitting).
- [ ] 7.3 Parity vs LTspice/NgSpice — deferred. The benchmark suite under `benchmarks/ltspice` / `benchmarks/ngspice` already runs through the Behavioral path; extending it to a side-by-side PWL run requires the linear-solver-cache fix (`refactor-linear-solver-cache`) and a longer `tstop` to reach steady state.
- [ ] 7.4 Wrapper retry layer deprecation — pending. Today `python/pulsim/__init__.py:run_transient` still calls `_apply_auto_bleeders` and `_tune_*_for_robust` profiles. Will be removed in `refactor-unify-robustness-policy` once `Auto` mode flips to `Ideal` by default.
- [ ] 7.5 `docs/pwl-switching-migration.md` — deferred to a docs PR.

## Phase 8: Validation suite expansion
- [x] 8.1 KCL invariant test landed in [test_pwl_segment_primary.cpp](../../../core/tests/test_pwl_segment_primary.cpp): runs an RC transient under `switching_mode = Ideal`, then assembles the DAE residual `f = N·x_final + (M/dt)·(x_final − x_prev) − (b_now + b_next)/2` from the final two states and asserts every node-and-branch row is ≤ `max(1e-9, ‖x‖_∞ · 1e-9)`. Full Hypothesis-style fuzzing across random circuits is the scope of the separate `add-property-based-testing` change; this targeted check guarantees the PWL kernel does not violate Kirchhoff at the assembly level.
- [x] 8.2 Energy invariant test landed in the same file: charges a 1 µF cap to 5 V on an LC ringing circuit (1 µH, ~159 kHz) with a 1 mΩ damping resistor, runs ~1.6 cycles in PWL Ideal mode, asserts `e_max ≤ e₀ · 1.02` (passivity guard — no energy gain) and `e_min ≥ e₀ · 0.85` (over-damping guard). Measured: `e₀ = 12.5 µJ`, `e_max = 12.5 µJ`, `e_min = 12.38 µJ` — passes well inside the band.
- [ ] 8.3 Forced-PWL regression on the existing benchmark suite — deferred. Today the YAML netlists in `benchmarks/circuits/` rely on the legacy auto_transient_profile path. A follow-up sweep needs to (a) opt each netlist into `simulation.switching_mode: ideal`, (b) add `set_pwl_state` initial commits, (c) extend the KPI gates for PWL-mode parity. This is a small-but-broad change best done after `refactor-linear-solver-cache` lands the cache fixes that unlock longer transients.
