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
- [x] 0.2 Record: `simulate()` line count (1435) and parameter count (40); `raise ValueError` count (406), broad-handler count (68; 13 in `__init__.py`), `warnings.warn(` count (44; 14 in `__init__.py`); fixed overhead of `simulate()` (audit row 21); the suite's warning set
  - Done (2026-11-19, merged f220a1da, before Stage A): `simulate()` 1846 lines (306 of them the docstring), 50 parameters; `raise ValueError` 560 in `python/pulsim` (ratchet 562), broad handlers 90 (13 in `__init__.py`; ratchet 90), `warnings.warn(` 65 (21 in `__init__.py`). Fixed overhead, information only (never asserted): bytecodes per call (`.scratch/opcount.py`, 2-switch buck) pwl 2480, trbdf2 3317; best of 300 wall, RC, 2 steps: pwl 34.4 us, trbdf2 194 us. The suite's warning set: 414 warnings recorded per test as (test, category, message, file; line when outside pulsim) by a pytest plugin (`pytest_warning_recorded`), compared after every Stage A group.

## 1. Stage A — tests first
- [x] 1.1 `test_plan_resolves_every_kwarg`: `set(plan.resolved_from) == set(signature(simulate).parameters) − {"builder"}` (xfail until 2.x lands)
  - Stands in: `test_the_table_covers_every_simulate_parameter` (every keyword is a row of the kernel's option table; the options are resolved by the kernel, DECISIONS 6). Open: a Python `plan.resolved_from`, if Stage A is built.
  - Done (2026-11-19): `python/tests/test_run_plan.py`; passes since group 2 (also the `solver=` / YAML marks).
- [x] 1.2 `test_engine_branches_read_only_the_plan`: `ast` check on `_run_pwl`, `_run_trbdf2`, `_run_dsed` (xfail until 7)
  - Done: `test_run_plan.py` (one argument `plan`; free names are module globals or builtins; never `_KERNEL_RUN` / `_PWL_RECIPE`); passes since group 7, with `test_simulate_is_plan_table_finalize`.
- [x] 1.3 Ratchets: `raise ValueError` and broad-handler counts in `python/pulsim` may not rise; no broad handler without re-raise around a kernel call on the `simulate()` call path (one level deep)
  - Done: the two count ratchets (`test_simulation_object.py`; broad handlers 89 -> 86); `test_simulate_swallows_no_kernel_error` (no broad handler in `simulate()`'s own body swallows, except the post-run voltage-sanity check; three narrowed to `AttributeError`). One level down: `test_helpers_one_level_down_swallow_no_kernel_error` (`test_run_plan.py`): the helpers that `simulate()`'s body calls may not add a swallowing broad handler; today's five are listed by name (census of test doubles, a bundle's `attach_to_result`, the post-run voltage check), shrink-only.
- [x] 1.4 `test_cross_cutting_on_every_engine`: builder ICs, `resume_from` validation, preflight ties, undriven-switch behaviour and voltage sanity asserted per engine against today's behaviour
  - Done: `python/tests/test_cross_cutting_on_every_engine.py` (15 cases, pwl / trbdf2 / exact; also green on 656b894f).

## 2. Stage A — plan the options
- [x] 2.1 `python/pulsim/_plan.py`: `RunPlan` (frozen), `NewtonOptions`, `EventOptions`, `FixedStep` / `VariableStep` / `ExactStep`, `Diagnostic`
  - Done: named `SimPlan` (`pulsim.RunPlan` is the public circuit record; DECISIONS 23), with `Initial`, `Drive`, `RunOutcome`.
- [x] 2.2 Move `LoadedCircuit` resolution, the `solver=` merge and keyword validation (`__init__.py:2575-2768` at f220a1da; now the head of `_plan_run`, `__init__.py:3038-3249`) into `_plan_run`; fill `resolved_from`
- [x] 2.3 Oracle: goldens bit-identical; warning set unchanged; 1.1 passes

## 3. Stage A — plan the state
- [x] 3.1 Preflight in `_plan_run` returning `plan.ties`; `_apply_ties(plan)` is the single mutation site
  - Done: `_apply_ties(builder, run_circuit, kernel_run, ties)` (`__init__.py:2187`); the ties are on `plan.circuit_plan` (`pulsim.RunPlan`).
- [x] 3.2 `resume_from` validation and the initial condition (`__init__.py:2769-2871` at f220a1da; now `:3287-3344`) → `plan.initial`
- [x] 3.3 Oracle: `test_initial_conditions_across_engines.py`, `test_snapshot_resume*.py`; goldens

## 4. Stage A — plan the route
- [x] 4.1 `_trbdf2_blockers`, `_dsed_unsupported` and the `auto` rule → one `_route(plan_inputs)` that consults the conformance table; `plan.engine`, `plan.route_reasons`, `plan.features`
  - Done: `_route` (`__init__.py:2792`, called at `:3386`).
- [x] 4.2 Oracle: `test_engine_auto.py`, `test_engine_default_routing.py`, `test_engine_auto_closed_loop.py`; matrix unchanged

## 5. Stage A — plan the drive
- [x] 5.1 One `_compose(plan_inputs)` for closed loops, observers, user `switch_fn`, `b_extra_fn` (the single function of `fix-control-tick-semantics`, or `__init__.py:2215-2283` and `:2420-2455` if it has not landed); gate drives composed by switch index
  - Done: `_compose` (`__init__.py:2693`) and `_compose_for(engine, ...)` (`:2716`).
- [x] 5.2 One `_undriven(plan_inputs)` keeping today's per-engine behaviour (now `__init__.py:2631`, called at `:3467` (trbdf2), `:3892` (pwl), `_dsed_dispatch.py:698`)
  - Done: positions still differ per engine (DECISIONS 23, item 1).
- [x] 5.3 Oracle: `test_closed_loops_compose_switch_fn.py`, `test_closed_loop_helper.py`, the tests of `fix-control-tick-semantics`; goldens

## 6. Stage A — finalize once
- [x] 6.1 `_finalize(result, plan)`: handles, `engine_used`, `engine_route_reason`, stats warnings, voltage sanity, for every engine
- [ ] 6.2 Separate PR: voltage sanity on `dsed` (**BREAKING**: a new warning); CHANGELOG note
- [ ] 6.3 Oracle: goldens; warning set unchanged in 6.1, plus the documented new warning in 6.2
  - Done for 6.1 (`_finalize(out, plan)`, `__init__.py:4090`): goldens identical, warning set 414 = 414. Open: 6.2 (for the release; DECISIONS 23).

## 7. Stage A — dispatch table
- [x] 7.1 `_run_dsed(plan)`, `_run_trbdf2(plan)`, `_run_pwl(plan)`; `simulate()` = plan → table → finalize
- [ ] 7.2 Oracle: 1.2 passes; goldens bit-identical; fixed overhead not above 0.2
  - Done: 1.2 passes; goldens bit-identical. Open: the overhead clause (pwl 2480 -> 3538 bytecodes per call, 34.4 -> 49.1 us; DECISIONS 23).
  - Overhead (DECISIONS 25): same counter and `.so`, pre-Stage A / Stage A / now: pwl 2551 / 3609 / 2834, trbdf2 3380 / 4413 / 3683, dsed 3993 / 5051 / 4348 bytecodes per call; RC wall pwl 41 / 57 / 45 us. NamedTuple records built with `tuple.__new__`, positional plan calls (no `locals()`), lazy diagnostics, hoisted imports. Still open: +283 (pwl) is the layout's own cost (five calls, the plan built and read back). Local-only ratchet `test_simulate_overhead.py`.
- [x] 7.3 Docs: `docs/internals` page "how a run is planned"; update the owner note's rule in `AI_GUIDE.md` ("put it in the plan")

## 8. Stage B — typed errors and diagnostics in Python
- [ ] 8.1 `python/pulsim/errors.py`: `PulsimError`, `NetlistError`, `OptionsError`, `ConvergenceError`, `SingularError`; the Phase 0 `UnsupportedError`, `SimulationAborted`, `Cancelled` and `NameNotFoundError` re-based under `PulsimError`; exported from `pulsim`
  - Done: `PulsimError`, `OptionsError`; `UnsupportedError`, `ConvergenceError` / `SimulationAborted`, `Cancelled`, `NameNotFoundError` re-based; exported. Open: `NetlistError`, `SingularError`, `errors.py` (DECISIONS 10).
- [ ] 8.2 Convert every raise reachable from `simulate()`; replace the 13 broad handlers in `__init__.py` by typed catches or remove them
- [ ] 8.3 `result.diagnostics`; `_emit_diagnostics`; route the 14 `warnings.warn` sites in `__init__.py` through it
- [ ] 8.4 Oracle: suite green without editing any `pytest.raises` / `pytest.warns`; ratchets hold; a test per class asserts `feature`, `engine`, `device` on `UnsupportedError`
- [ ] 8.5 Docs: error reference page; `AI_GUIDE.md` error section; stubs; CHANGELOG; note for PulsimGUI (`BackendRefusal` can catch `PulsimError`)

## 9. Stage B — options and errors in the kernel
- [x] 9.1 `core/include/pulsim/sim/options.hpp`: `RunOptions` and its parts; one defaults table; `to_simulation_options`, `to_trbdf2_options`
- [ ] 9.2 `core/include/pulsim/sim/error.hpp` (the `PulsimError` mix-in; each class keeps its `std` base; `solver/unsupported.hpp` of Phase 0 gains the mix-in), `diagnostic.hpp`; pybind translators to the Python classes of 8.1
  - Done: `sim/error.hpp` `PulsimError` mix-in on `UnsupportedError`, `sim::OptionsError`, `SimulationAborted`, `ConvergenceFailure`, `Cancelled`; translators to `pulsim.PulsimError` subclasses; `sim/diagnostic.hpp`. Open: `NetlistError` / `SingularError` (DECISIONS 10).
- [ ] 9.3 Remove the sentinels (`max_newton_iterations=0` warns for one release) and the four recorded-only fields of `SimulationOptions` (**BREAKING**)
  - Done: no sentinel in `RunOptions`, `simulate()` (`max_newton_iterations=0` warns) or the raw TR-BDF2 binding. Open: the four recorded-only `SimulationOptions` fields (the YAML loader stores them there; they need a home on `LoadedCircuit` first).
- [x] 9.4 Raw TR-BDF2 binding defaults follow the table (**BREAKING**); CHANGELOG migration note
- [x] 9.5 Oracle: goldens bit-identical through `simulate()`; a test asserts every default in the Python signature, the YAML loader and the kernel comes from the one table

## 10. Stage B — the Simulation facade
- [x] 10.1 `core/include/pulsim/sim/simulation.hpp`: `Circuit` view, `RunPlan`, `Simulation` (constructor plans; `run`, `run_until`, `snapshot`, `restore`, `diagnostics`, attachments)
- [x] 10.2 The facade builds the device stamps, the cache and the default mask policy and calls the existing drivers
- [ ] 10.3 C++ tests (no Python): buck with a Shockley diode matches the Python result bit for bit; an unsupported device on `ExactStep` throws `UnsupportedError` naming device and engine; `run_until` + `snapshot` + `restore` reproduces a continuous fixed-step run exactly
  - Done: the Shockley buck bit-for-bit test and the split-run test. Open: the `ExactStep` device refusal (the facade refuses the exact engine as a whole, DECISIONS 7; no device kind is refused by pwl or trbdf2).
- [x] 10.4 Binding `pulsim.Simulation`, `pulsim.RunOptions`; stubs
- [ ] 10.5 Oracle: the conformance matrix run through `Simulation` equals the matrix run through `simulate()`; zero-alloc tests green

## 11. Stage B — simulate() on the facade
- [ ] 11.1 `simulate()` builds `RunOptions` and calls `Simulation` for runs with no Python callback; then for every run through `add_gate` / `add_tick` / `add_source_term` adapters
  - Done: `'pwl'` and `'trbdf2'` runs go through `Simulation` (callbacks included: `set_switch_fn`, `add_tick` (monitors too), `set_source_term`, `set_step_observer`, `set_tick_before`, live ring, resume with the v5 tick grid), goldens identical (DECISIONS 14, 15). `mmc_arms` / `on_cache` and the C++ chain fast path run on the object's builder and cache (DECISIONS 20). Open: `sweep()`'s replay and the exact engine (DECISIONS 16).
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
