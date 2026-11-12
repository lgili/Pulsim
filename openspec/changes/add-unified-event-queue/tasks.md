Every numbered section is one or more PRs and ends with the
verification set: `ruff check`, `pytest python/tests`, `ctest`,
`openspec validate --all --strict`, the stub check, and
`core/tests/perf/test_zero_alloc.cpp`. "Goldens" = the fixed-engine
checkpoint hashes and the differential runner of
`restore-reference-benchmark-harness` (when it has landed; otherwise the
full suite plus the per-step hash test added in 0.2); "matrix" = the
conformance test of `add-engine-conformance-matrix`.

## 0. Preconditions and baseline
- [ ] 0.1 `add-engine-conformance-matrix` merged; record its schedule × engine and feature × engine cells
- [ ] 0.2 Record for the buck, the synchronous buck with dead time, the three-phase VSI and one closed-loop buck: fixed-engine state hash, TR-BDF2 `n_accept` / `n_reject` / `n_gate_events` / `n_ctrl_ticks` / `n_diode_events`, DSED event count, ns/step (audit rows 1, 2, 16–18)
- [ ] 0.3 Analytic edge counts for those circuits (edges per period × periods) written into the test as constants

## 1. One detection in the binding
- [ ] 1.1 `python/gate_source.hpp`: `to_gate_source(obj, num_switches, default_resolution)` covering `SwitchSchedule`, `NativePwm2Switch`, `NativeMultiMaskPwm`, any callable; built on the shared adapter of `add-engine-conformance-matrix`
- [ ] 1.2 The fixed, TR-BDF2 and DSED bindings call it and convert the result to today's `switch_fn` + `next_edge_fn`; delete any of the three detections Phase 0 left (`bindings.cpp:2598-2636`, `:4016-4047`, `dsed_bindings.cpp:749-815` on this branch)
- [ ] 1.3 Oracle: goldens bit-identical; TR-BDF2 counters identical; `test_pwm_grid_edges.py`; matrix unchanged or improved (`NativeMultiMaskPwm` exact on every engine)

## 2. The queue, used by TR-BDF2
- [ ] 2.1 `events/time_event.hpp`, `events/event_queue.hpp`: `Kind`, `TimeEvent`, `TimeSource`, `EventQueue` (fixed-size heap, `next_landing`, `take_at`, `invalidate`, counters)
- [ ] 2.2 `events/gate_source.hpp`: `GateSource`, `ScheduleGate`, index composition with the overlap check
- [ ] 2.3 Unit tests: k-way merge order, simultaneity order, integer-counter times (no drift over 10⁷ periods), no allocation after `start`
- [ ] 2.4 TR-BDF2 uses the queue for gate, tick and end-of-run landing; `next_edge_fn` / `observer_period` become adapters that register sources
- [ ] 2.5 Oracle: for every harness circuit with an analytic schedule, the accepted-step sequence and the states are identical to 0.2; zero-alloc tests green

## 3. The opaque-callable adapter
- [ ] 3.1 `events/sampled_gate.hpp`: move the sampling, pulse-width learning and bisection of `trbdf2_transient.hpp:960-1190` into `SampledGate` without changing the algorithm; `resolution()`, `n_gate_undersampled`, `gate_pulse_min`
- [ ] 3.2 TR-BDF2 calls `SampledGate`; oracle: results and counters identical for the opaque-callable tests
- [ ] 3.3 DSED calls `SampledGate`; delete `dsed_bindings.cpp:174-258` (**BREAKING** for opaque callables on dsed); oracle: synchronous buck with a plain Python `switch_fn` reads 11.49 V within the existing tolerance; an edge is within the declared resolution of the analytic one
- [ ] 3.4 Diagnostic when a seen pulse is narrower than the resolution, on every engine
- [ ] 3.5 Docs: `docs/dsed.md`, `AI_GUIDE.md` (switch_fn section), CHANGELOG migration note

## 4. Per-rate control ticks
- [ ] 4.1 `Rate`, `add_tick`, `TickInfo{scheduled_t, k}`; exact count `ceil((t_end − t0 − offset)/T)` (ticks earlier than `t_end`)
- [ ] 4.2 The one composition function of `fix-control-tick-semantics` registers one rate per closed loop and per observer on the queue; the per-source period list of the TR-BDF2 driver and the fixed engine's tick counters become queue sources; `result.control_tick_counts` is filled from the queue
- [ ] 4.3 Oracle: the tests of `fix-control-tick-semantics` (2 kHz PI + 20 kHz monitor: 40 and 400 ticks; two loops without `dt`); `test_engine_auto_closed_loop.py`, `test_closed_loop_helper.py`

## 5. The fixed engine on the queue
- [ ] 5.1 The fixed engine builds the mask of step `k` from the gate sources and reuses it until `next_landing`; ticks come from the queue
- [ ] 5.2 Delete the second composition (`__init__.py:2420-2455`) and the default-mask lambdas in favour of index composition (the undriven-switch policy itself is `add-kernel-simulation-object`; until then the current per-engine behaviour is kept)
- [ ] 5.3 `n_gate_events` and `n_gate_events_unresolved` on the fixed engine and on DSED
- [ ] 5.4 Oracle: fixed goldens bit-identical (no ticks); with ticks identical to the Phase 0 behaviour; `n_gate_events` equals the analytic count of 0.3 on pwl, trbdf2 and dsed; ns/step on audit rows 1–2 not above 0.2
- [ ] 5.5 Test: two edges of one bit inside one fixed step are counted and reported, not dropped silently

## 6. Modulators and duty registers
- [ ] 6.1 `events/modulator.hpp`: `DutyRegister`, `LatchPolicy`, `CarrierPwm` (single switch and complementary leg with dead time), `EdgeList`; Python bindings
- [ ] 6.2 `bind_pi_to_switch` builds a `CarrierPwm` with `Immediate` latching; the tick handler writes the register; delete the throttle in `control.py:1350-1355`
- [ ] 6.3 C blocks and `BlockChain` register rates; delete the throttle in `c_block.py:387-390` and `BlockChain::make_step_observer`
- [ ] 6.4 Oracle: closed-loop tests identical on pwl, within tolerance on trbdf2; analytic edges of `CarrierPwm` equal those of `make_pwm_switch_fn` for constant duty to 1 ulp
- [ ] 6.5 Measure the closed-loop buck ns/step before and after (estimate to verify: 1.3–2.5 µs → about 0.4 µs)
- [ ] 6.6 Docs: closed-loop guide, `AI_GUIDE.md`, stubs

## 7. Source breakpoints
- [ ] 7.1 `next_breakpoint(t)` for the pulse and PWM voltage sources; registered as time sources (moving the table of `update-trbdf2-error-control` if present)
- [ ] 7.2 Oracle: pulse-driven RC on TR-BDF2: error below 1.8e-4 V and decreasing with `rtol`; fixed goldens bit-identical

## 8. State events
- [ ] 8.1 `events/locate.hpp`: Illinois over a trial evaluator; unit tests on analytic crossings
- [ ] 8.2 `events/mode_set.hpp`: `redecide`, `crossing_signal`, `settle` with cycle report
- [ ] 8.3 TR-BDF2 localisation, the DSED predictor, the fixed-step event iteration and the DC operating point call 8.1 / 8.2
- [ ] 8.4 Oracle: `commutation_events`, `event_iteration_count`, `event_iteration_breaches`, `n_diode_events` identical on the harness; the resistive half-wave conduction instants (223 µs, 9.777 ms) unchanged on every engine

## 9. Remove the legacy inputs (one release later)
- [ ] 9.1 Delete `next_edge_fn`, `observer_period`, the callable-only driver parameters and their adapters
- [ ] 9.2 Oracle: matrix; full suite; stub check
- [ ] 9.3 Docs: CHANGELOG, `docs/mental-model.md`, `openspec/CORE_ROADMAP.md` status
