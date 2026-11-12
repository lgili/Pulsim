Every numbered section is one or more PRs and ends with the
verification set: `ruff check`, `pytest python/tests`, `ctest`,
`openspec validate --all --strict`, the stub check, and
`core/tests/perf/test_zero_alloc.cpp`. "Goldens" = the fixed-engine
checkpoint hashes and the differential runner of
`restore-reference-benchmark-harness` (the full suite stands in until it
lands); "matrix" = the conformance test of
`add-engine-conformance-matrix`. Stage A PRs additionally assert that
the set of warnings (category, message) emitted by the suite is
unchanged.

## 0. Preconditions and baseline
- [ ] 0.1 `add-engine-conformance-matrix` merged
- [ ] 0.2 Record: `simulate()` line count (1435) and parameter count (40); `raise ValueError` count (406), broad-handler count (68; 13 in `__init__.py`), `warnings.warn(` count (44; 14 in `__init__.py`); fixed overhead of `simulate()` (audit row 21); the suite's warning set

## 1. Stage A — tests first
- [ ] 1.1 `test_plan_resolves_every_kwarg`: `set(plan.resolved_from) == set(signature(simulate).parameters) − {"builder"}` (xfail until 2.x lands)
- [ ] 1.2 `test_engine_branches_read_only_the_plan`: `ast` check on `_run_pwl`, `_run_trbdf2`, `_run_dsed` (xfail until 7)
- [ ] 1.3 Ratchets: `raise ValueError` and broad-handler counts in `python/pulsim` may not rise; no broad handler without re-raise around a kernel call on the `simulate()` call path (one level deep)
- [ ] 1.4 `test_cross_cutting_on_every_engine`: builder ICs, `resume_from` validation, preflight ties, undriven-switch behaviour and voltage sanity asserted per engine against today's behaviour

## 2. Stage A — plan the options
- [ ] 2.1 `python/pulsim/_plan.py`: `RunPlan` (frozen), `NewtonOptions`, `EventOptions`, `FixedStep` / `VariableStep` / `ExactStep`, `Diagnostic`
- [ ] 2.2 Move `LoadedCircuit` resolution, the `solver=` merge and keyword validation (`__init__.py:1815-1900`) into `_plan_run`; fill `resolved_from`
- [ ] 2.3 Oracle: goldens bit-identical; warning set unchanged; 1.1 passes

## 3. Stage A — plan the state
- [ ] 3.1 Preflight in `_plan_run` returning `plan.ties`; `_apply_ties(plan)` is the single mutation site
- [ ] 3.2 `resume_from` validation and the initial condition (`__init__.py:1956-1988`) → `plan.initial`
- [ ] 3.3 Oracle: `test_initial_conditions_across_engines.py`, `test_snapshot_resume*.py`; goldens

## 4. Stage A — plan the route
- [ ] 4.1 `_trbdf2_blockers`, `_dsed_unsupported` and the `auto` rule → one `_route(plan_inputs)` that consults the conformance table; `plan.engine`, `plan.route_reasons`, `plan.features`
- [ ] 4.2 Oracle: `test_engine_auto.py`, `test_engine_default_routing.py`, `test_engine_auto_closed_loop.py`; matrix unchanged

## 5. Stage A — plan the drive
- [ ] 5.1 One `_compose(plan_inputs)` for closed loops, observers, user `switch_fn`, `b_extra_fn` (the single function of `fix-control-tick-semantics`, or `__init__.py:2215-2283` and `:2420-2455` if it has not landed); gate drives composed by switch index
- [ ] 5.2 One `_undriven(plan_inputs)` keeping today's per-engine behaviour (`:2304`, `:2664-2673`, `_dsed_dispatch.py:617-630`)
- [ ] 5.3 Oracle: `test_closed_loops_compose_switch_fn.py`, `test_closed_loop_helper.py`, the tests of `fix-control-tick-semantics`; goldens

## 6. Stage A — finalize once
- [ ] 6.1 `_finalize(result, plan)`: handles, `engine_used`, `engine_route_reason`, stats warnings, voltage sanity, for every engine
- [ ] 6.2 Separate PR: voltage sanity on `dsed` (**BREAKING**: a new warning); CHANGELOG note
- [ ] 6.3 Oracle: goldens; warning set unchanged in 6.1, plus the documented new warning in 6.2

## 7. Stage A — dispatch table
- [ ] 7.1 `_run_dsed(plan)`, `_run_trbdf2(plan)`, `_run_pwl(plan)`; `simulate()` = plan → table → finalize
- [ ] 7.2 Oracle: 1.2 passes; goldens bit-identical; fixed overhead not above 0.2
- [ ] 7.3 Docs: `docs/internals` page "how a run is planned"; update the owner note's rule in `AI_GUIDE.md` ("put it in the plan")

## 8. Stage B — typed errors and diagnostics in Python
- [ ] 8.1 `python/pulsim/errors.py`: `PulsimError`, `NetlistError`, `OptionsError`, `ConvergenceError`, `SingularError`; the Phase 0 `UnsupportedError`, `SimulationAborted`, `Cancelled` and `NameNotFoundError` re-based under `PulsimError`; exported from `pulsim`
- [ ] 8.2 Convert every raise reachable from `simulate()`; replace the 13 broad handlers in `__init__.py` by typed catches or remove them
- [ ] 8.3 `result.diagnostics`; `_emit_diagnostics`; route the 14 `warnings.warn` sites in `__init__.py` through it
- [ ] 8.4 Oracle: suite green without editing any `pytest.raises` / `pytest.warns`; ratchets hold; a test per class asserts `feature`, `engine`, `device` on `UnsupportedError`
- [ ] 8.5 Docs: error reference page; `AI_GUIDE.md` error section; stubs; CHANGELOG; note for PulsimGUI (`BackendRefusal` can catch `PulsimError`)

## 9. Stage B — options and errors in the kernel
- [ ] 9.1 `core/include/pulsim/sim/options.hpp`: `RunOptions` and its parts; one defaults table; `to_simulation_options`, `to_trbdf2_options`
- [ ] 9.2 `core/include/pulsim/sim/error.hpp` (the `PulsimError` mix-in; each class keeps its `std` base; `solver/unsupported.hpp` of Phase 0 gains the mix-in), `diagnostic.hpp`; pybind translators to the Python classes of 8.1
- [ ] 9.3 Remove the sentinels (`max_newton_iterations=0` warns for one release) and the four recorded-only fields of `SimulationOptions` (**BREAKING**)
- [ ] 9.4 Raw TR-BDF2 binding defaults follow the table (**BREAKING**); CHANGELOG migration note
- [ ] 9.5 Oracle: goldens bit-identical through `simulate()`; a test asserts every default in the Python signature, the YAML loader and the kernel comes from the one table

## 10. Stage B — the Simulation facade
- [ ] 10.1 `core/include/pulsim/sim/simulation.hpp`: `Circuit` view, `RunPlan`, `Simulation` (constructor plans; `run`, `run_until`, `snapshot`, `restore`, `diagnostics`, attachments)
- [ ] 10.2 The facade builds the device stamps, the cache and the default mask policy and calls the existing drivers
- [ ] 10.3 C++ tests (no Python): buck with a Shockley diode matches the Python result bit for bit; an unsupported device on `ExactStep` throws `UnsupportedError` naming device and engine; `run_until` + `snapshot` + `restore` reproduces a continuous fixed-step run exactly
- [ ] 10.4 Binding `pulsim.Simulation`, `pulsim.RunOptions`; stubs
- [ ] 10.5 Oracle: the conformance matrix run through `Simulation` equals the matrix run through `simulate()`; zero-alloc tests green

## 11. Stage B — simulate() on the facade
- [ ] 11.1 `simulate()` builds `RunOptions` and calls `Simulation` for runs with no Python callback; then for every run through `add_gate` / `add_tick` / `add_source_term` adapters
- [ ] 11.2 Oracle: goldens bit-identical; warning set unchanged; fixed overhead not above 0.2; a native-schedule run holds no GIL (thread sweep scales as in audit row 23)
- [ ] 11.3 One undriven-switch policy on every engine (**BREAKING**), per the owner's answer; migration note; tests for `Refuse`, `Open`, `Closed`

## 12. Stage B — remaining Python-only semantics
- [ ] 12.1 C-block zero-order hold as a kernel tick handler
- [ ] 12.2 `steady_state.py`, `sweep.py`, `adaptive.py`, `fra.py`, `mmc.py` call through `RunOptions`
- [ ] 12.3 Oracle: their test files unchanged

## 13. Close
- [ ] 13.1 Deprecate raw `run_transient`, `run_transient_trbdf2`, `SimulationOptions`, `TrBdf2Options`; `simulate()` body ≤ 150 lines (test)
- [ ] 13.2 One release later: remove them and the emptied Stage A helpers
- [ ] 13.3 Docs: `docs/mental-model.md`, C++ quick start using `Simulation`, `openspec/CORE_ROADMAP.md` status
