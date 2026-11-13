Every numbered section is one or more PRs and ends with the
verification set: `ruff check`, `pytest python/tests`, `ctest`,
`openspec validate --all --strict`, the stub check, and
`core/tests/perf/test_zero_alloc.cpp`. "Goldens" = the fixed-engine
checkpoint hashes; "runner" = the differential runner; "counter gate" =
the counter-based perf gate, all from
`restore-reference-benchmark-harness`. "Matrix" = the conformance test
of `add-engine-conformance-matrix`. "Step-identical" = equal accepted
times, states, mode record and `n_accept` / `n_reject` / `n_solves` /
`n_gate_events` / `n_diode_events`. Comparisons across grids use states
at equal times or time-weighted measures, never sample means.

## 0. Preconditions and baseline
- [ ] 0.1 Merged: `restore-reference-benchmark-harness`, `add-engine-conformance-matrix`, `add-unified-device-interface` sections 1–3, `add-unified-event-queue` sections 2, 5 and 8
- [ ] 0.2 Record on `main`: goldens; TR-BDF2 step sequences of the harness; counter gate values; ns/step for audit rows 1–3 and 5–9; extension build time
- [ ] 0.3 Owner decision on `add-adaptive-runge-kutta-solvers` (design Open Question 1)

## 1. Extract the source accumulation
- [ ] 1.1 `solver/step/source_accumulator.hpp`: one implementation of `run_transient.hpp:777-793`, `trbdf2_transient.hpp:480-500` and the DSED adapter's overlay
- [ ] 1.2 Both MNA drivers and the DSED adapter call it
- [ ] 1.3 Oracle: goldens bit-identical; TR-BDF2 step-identical; DSED tests green

## 2. Extract the single solve point
- [ ] 2.1 `solver/step/stepper_core.hpp`: `StageSpec`, `StepperCore::solve_stage` (step context, history right-hand side, sources, linear or Newton dispatch)
- [ ] 2.2 TR-BDF2: `trap_solve`, `probe_solve`, the stage solves and the settle call `solve_stage`; fixed engine: `run_event_iteration`, the retry sub-steps and the sub-step correction call it
- [ ] 2.3 Test: a grep-based check that neither driver calls `cache.solve`, `cache.solve_at` or the Newton solver except through `solve_stage`
- [ ] 2.4 Oracle: goldens bit-identical; step-identical; `n_solves` identical; `test_trbdf2_stateful_with_diodes.py`, `test_stateful_device_stamps_compose.py`

## 3. Extract commit, rollback and seeding
- [ ] 3.1 `StepperCore::commit`, `snapshot_into`, `restore`, `seed(InitialCondition)`; both drivers call them
- [ ] 3.2 Oracle: goldens; step-identical; `test_snapshot_resume.py`, `test_snapshot_resume_stateful.py`, `test_initial_conditions_across_engines.py`, `test_first_step_order.py` (its xfails keep their status)

## 4. Integrators as stage tables
- [ ] 4.1 `solver/step/integrators.hpp`: `Trapezoidal`, `BackwardEuler`, `TrBdf2` (`stage`, `lte` moved from `trbdf2_transient.hpp:1286-1375`)
- [ ] 4.2 Both drivers obtain their `StageSpec`s from the tables
- [ ] 4.3 Unit tests: both TR-BDF2 stages return the same `factor_dt`; `BackwardEuler::stage(h/2)` returns `factor_dt == h`; every stage's `h_elapsed` sums to `h`
- [ ] 4.4 Oracle: goldens; step-identical; counter gate equal

## 5. Controllers and policies as types
- [ ] 5.1 `solver/step/controllers.hpp`: `Fixed` (integer counter), `Lte` (moved from `trbdf2_transient.hpp:1710-1810` with the step memory), `ControllerState`
- [ ] 5.2 `solver/step/policies.hpp`: `IterateAtStepEnd`, `Localize`, `HalvingLadder`, `ShrinkAndRepeat`, `GuardPolicy`, `RestartPolicy` (`None` plus whatever Phase 2 has landed), each extracted in place and called by its driver
- [ ] 5.3 Oracle: goldens; step-identical; `dt_retries`, `event_iteration_breaches`, `inductor_guard_actions` records identical

## 6. The variable-step stepper
- [ ] 6.1 `solver/step/stepper.hpp`: `Stepper<I, C>`; instantiate `Stepper<TrBdf2, Lte>`; hidden engine name `trbdf2-next`
- [ ] 6.2 Equality proof: `trbdf2-next` step-identical to `trbdf2` on every harness circuit and every suite circuit that runs on `trbdf2`; runner job added
- [ ] 6.3 Zero-alloc test for the variable-step stepper (accepted, rejected and Newton-retried steps)
- [ ] 6.4 Switch `engine='trbdf2'` and `engine='auto'` without `dt` to the stepper; keep `engine='trbdf2-legacy'`
- [ ] 6.5 Oracle: matrix equal for `trbdf2` and `trbdf2-legacy`; counter gate equal; ns/step rows 5–9 not above 0.2
- [ ] 6.6 Docs: `docs/solvers.md`, CHANGELOG (the legacy name and its removal date)

## 7. The fixed-step stepper, feature by feature
- [ ] 7.1 `Stepper<Trapezoidal, Fixed>` as `pwl-next` for plain runs (linear, switches, PWL diodes, Newton devices, `should_continue`, `b_extra_fn`, `initial_state`, `resume_from`)
- [ ] 7.2 `start_from_dc_op`
- [ ] 7.3 Recorder hooks: `store_every`, `progress`, `live_stream`
- [ ] 7.4 `strict_event_iterations`
- [ ] 7.5 `HalvingLadder` (`max_dt_halvings`), including the saturable-inductor refusal
- [ ] 7.6 Sub-step state correction
- [ ] 7.7 Inductor guards (`inductor_freeze_di_max`, `inductor_abs_clamp`)
- [ ] 7.8 Observers, closed loops, `BlockChain` (`run_transient_with_chain`)
- [ ] 7.9 C blocks
- [ ] 7.10 `on_cache`
- [ ] 7.11 `mmc_arms`
- [ ] 7.12 Oracle for each of 7.1–7.11: the feature's test files give bit-identical results, records and warnings on `pwl-next` and `pwl`; the `pwl-next` cell of the matrix flips in that PR only

## 8. Switch the fixed engine
- [ ] 8.1 Equality proof over the whole harness and suite: trajectory hashes, mode record, `commutation_events`, `event_iteration_count`, `dt_retries`, `inductor_guard_actions`
- [ ] 8.2 Port the five cases of `test_zero_alloc.cpp` to the stepper
- [ ] 8.3 Switch `engine='pwl'` and `engine='auto'` with `dt`; keep `engine='pwl-legacy'` and the raw `run_transient` binding on the old function
- [ ] 8.4 Oracle: goldens bit-identical; counter gate equal; ns/step rows 1–3 not above 0.2; build time within +10 %
- [ ] 8.5 Docs: `docs/mental-model.md`, `docs/solvers.md`, `AI_GUIDE.md` engine section, CHANGELOG

## 9. Features on the LTE controller
- [ ] 9.1 One PR each: `start_from_dc_op`, `strict_event_iterations`, inductor guards, `on_cache`, C blocks; each with a cross-engine agreement test at equal times
- [ ] 9.2 Structural refusals (`max_dt_halvings`, sub-step correction, observer without a cadence) raise `UnsupportedError` with the reason stated in the design
- [ ] 9.3 Recorder features on the LTE controller when `add-probes-and-recorders` has defined them
- [ ] 9.4 Oracle: matrix updated cell by cell; no cell silent
- [ ] 9.5 Docs: `docs/known-limitations.md`, the feature table in `AI_GUIDE.md`

## 10. Replace bdf1
- [ ] 10.1 `Stepper<BackwardEuler, Fixed>` behind `run_transient_bdf1` (deprecated alias)
- [ ] 10.2 Delete `core/include/pulsim/solver/bdf1.hpp`
- [ ] 10.3 Oracle: `test_dc_operating_point.py`, `core/tests/layer4/test_gmin.cpp`, `core/tests/dsed/test_switch_state_record.cpp`; first-order convergence on an RC; the nonlinear-capacitor probe (`arch/probe5.py`) reads the pwl value; bdf1 cells of the matrix supported for Newton kinds and PWL diodes

## 11. Retire the dead integrators
- [ ] 11.1 Move the tests of the V0 overload to the stepper; delete `run_transient.hpp:179-265`
- [ ] 11.2 Deprecate `core/include/pulsim/integrators/`, `dopri5_solve`, `radau_iia3_solve`, `python/pulsim/integrators.py`, `python/pulsim/adaptive.py` (warning for one release), then delete (**BREAKING**)
- [ ] 11.3 Remove the four recorded-only fields' consumers (`__init__.py:2391-2415`)
- [ ] 11.4 Oracle: suite; stub check; docs build
- [ ] 11.5 Docs: `docs/solvers.md` (remove the Python RK section), CHANGELOG migration note

## 12. Controller state in the snapshot
- [ ] 12.1 Optional `ControllerState` in the snapshot; pickle version bump with a reader for the previous version
- [ ] 12.2 Tests: a split variable-step run is bit-identical to the continuous one; a snapshot without controller state resumes to tolerance; a fixed-step snapshot resumes on the LTE controller and the reverse
- [ ] 12.3 Docs: snapshot page

## 13. Delete the legacy drivers (one release after 8.3)
- [ ] 13.1 Delete `pwl-legacy`, `trbdf2-legacy`, the hidden names, the bodies of `run_transient` and `run_transient_trbdf2`
- [ ] 13.2 Oracle: matrix; harness; suite; counter gate
- [ ] 13.3 Final measurement table: lines of transient driver code before and after, ns/step rows 1–3 and 5–9, build time; update `openspec/CORE_ROADMAP.md`

<!-- Re-centred scope, 2026-11-13: groups 14–17 are appended; groups 1–13 are unchanged. -->

## 14. Event-driven outer loop (re-centred scope, D8)
- [ ] 14.1 Outer loop over queue events with `seg.advance(x, t, t_next, watch)`; the D4 step loop becomes the implicit segment's inner loop; `process_instant` moves to the outer loop
- [ ] 14.2 Oracle: with the legacy policies selected, the fixed engine stays bit-identical and the variable-step engine step-identical on the harness (the D7 proofs re-run)

## 15. Shared services (D10; needs `add-complementarity-mode-selection` group 6 and `add-branch-form-ideal-switch` group 5)
- [ ] 15.1 `ModeSelector` and `ReinitService` called once per event instant by every strategy; `modes.settle` kept as the legacy policy
- [ ] 15.2 `RestartPolicy::Sdirk22x3` (three consecutive Alexander SDIRK(2,2) steps, `factor_dt = 2γh`); `Cda` kept only as a legacy policy
- [ ] 15.3 Tests: the DCM buck, boost and bridges give the same mode sequence on the three strategies; the interrupted-inductor and capacitor-loop records are identical on the three strategies within 1e-9 relative

## 16. Strategy selectors (D11)
- [ ] 16.1 `engine='auto'|'exact'|'pwl'|'trbdf2'`, `'dsed'` alias; route reason in the result
- [ ] 16.2 Tests: each name runs the strategy of the D11 table (asserted from the route record); `'dsed'` and `'exact'` bit-identical

## 17. Default flip (in the single number-changing release)
- [ ] 17.1 Defaults: `auto` route, selection service, SDIRK×3 restart; legacy names and policies warn
- [ ] 17.2 Re-baseline through the reference harness with the other number-changing defaults; CHANGELOG
