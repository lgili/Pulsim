## 1. Failing tests first
`python/tests/test_control_tick_semantics.py`. Circuit A: two bucks on one 24 V source (`probe6.py`); circuit B: one buck, 2 kHz PI, optional 20 kHz monitor (`probe7.py`); circuit C: RLC with a counting observer (`probe2.py`).
- [x] 1.1 A, `closed_loops=[la (20 kHz), lb (2 kHz)]`, 10 ms, no `dt`: completes; `len(la.duty_history) == 200`, `len(lb.duty_history) == 20`. Fails today (`TypeError`)
- [x] 1.2 A with one loop plus a user `switch_fn` driving the other switch, no `dt`: completes, loop ticks 200
- [x] 1.3 B on `engine='auto'` (routes to `trbdf2`), 20 ms, with the monitor at `controller_period = 50 µs`: PI ticks 40, monitor calls 400, time-weighted V_out over the last 5 ms within 0.01 V of the run without the monitor (13.033 V). Fails today (400 ticks, 12.091 V)
- [x] 1.4 C, `controller_period = 100 µs`, 2 ms: observer calls 20 on `pwl` (`dt = 1 µs`) and on `trbdf2`. Fails today (2001 on `pwl`)
- [x] 1.5 A on `pwl`, `dt = 1 µs`, 10 ms: ticks 200 and 20. Fails today (198 and 20)
- [x] 1.6 Tick instants on `trbdf2`: every tick time recorded in `duty_history` is within `1e-12·T` of `k·T`; the first is `t_start`
- [x] 1.7 `pwl`, `T = 2.5·dt`: runs, one warning naming the jitter, count `ceil(span/T)`, `control_tick_jitter_max < dt`; `T = 0.5·dt`: raises
- [x] 1.8 Coincident ticks fire in registration order (a list appended to by three sources at 1, 2 and 4 kHz reads `[0, 1, 2]` at t = 0 and at every 1 ms)
- [x] 1.9 Identity: one `ClosedLoop` on `trbdf2` gives states equal (`np.array_equal`) to the pre-change run (reference captured before task 2). Checked by the scratch gate `one_loop_gate.py` (1628 x 5 states, 20 ticks): identical; not a committed test, the reference is a binary capture

## 2. Composition
- [x] 2.1 `python/pulsim/_control_plan.py`: `ControlPlan`, `_compose_control` (named `compose_control`); the three lists are exercised through `simulate()` by tests 1.1-1.8 and `test_controller_period_without_observer_is_refused`
- [x] 2.2 `simulate()`: call it once above the dispatch; delete `__init__.py:2215-2283` composition and `2420-2455`; keep `_inner_observers` for the motor-trace walk
- [x] 2.3 `controller_period` without an observer: refusal (Open Question 3)

## 3. Variable-step kernel
- [x] 3.1 `TrBdf2Options::observer_periods`, per-source counters, indexed callback, per-source tick counts in `TrBdf2Stats`; correct the `floor(span/T) + 1` comment
- [x] 3.2 Binding: `observer_periods` / `observers` lists; the single-source keywords remain and map to one-element lists
- [x] 3.3 C++ test: three sources at 1, 2 and 4 kHz over 10 ms fire 10, 20 and 40 times at exact multiples

## 4. Fixed-step scheduler
- [x] 4.1 Tick scheduler (integer `k`, tolerance `1e-9·dt`), tick 0 at `t_start`, refusal for `T < dt`, jitter warning and `control_tick_jitter_max`
- [x] 4.2 `closed_loops` on `pwl` use `.tick` through the scheduler; `step_observer` + `controller_period` likewise
- [x] 4.3 `result.control_tick_counts` on both engines
- [x] 4.4 Update the closed-loop tests whose expected values move; each new value explained in the test (tick count, not a re-recorded number)

## 5. Docs
- [x] 5.1 `CHANGELOG.md` [Unreleased]: the three BREAKING notes, migration for `controller_period` on `pwl`
- [x] 5.2 `python/pulsim/AI_GUIDE.md`: tick rule, multi-rate example, removal of the "198 of 200" caveat
- [x] 5.3 `KNOWN_LIMITATIONS.md` / `docs/known-limitations.md`: fixed-engine tick jitter; controllers still refused on `dsed`
- [x] 5.4 Docstrings: `simulate` (`controller_period`, `closed_loops`), `bind_pi_to_switch`, `ClosedLoop.period`; stubs via `scripts/generate_stubs.py`
- [x] 5.5 Remove the strict-xfail markers for these cells in `test_engine_conformance.py` if that change has landed

## 6. Verification
- [x] 6.1 `ruff check`
- [x] 6.2 `pytest python/tests`
- [x] 6.3 `ctest -j2`
- [x] 6.4 `openspec validate --all --strict`
- [x] 6.5 Stub check
- [x] 6.6 Re-run `probe2.py`, `probe6.py`, `probe7.py`; record the lines in the PR
