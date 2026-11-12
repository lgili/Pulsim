## 1. Failing tests first
`python/tests/test_control_tick_semantics.py`. Circuit A: two bucks on one 24 V source (`probe6.py`); circuit B: one buck, 2 kHz PI, optional 20 kHz monitor (`probe7.py`); circuit C: RLC with a counting observer (`probe2.py`).
- [ ] 1.1 A, `closed_loops=[la (20 kHz), lb (2 kHz)]`, 10 ms, no `dt`: completes; `len(la.duty_history) == 200`, `len(lb.duty_history) == 20`. Fails today (`TypeError`)
- [ ] 1.2 A with one loop plus a user `switch_fn` driving the other switch, no `dt`: completes, loop ticks 200
- [ ] 1.3 B on `engine='auto'` (routes to `trbdf2`), 20 ms, with the monitor at `controller_period = 50 µs`: PI ticks 40, monitor calls 400, time-weighted V_out over the last 5 ms within 0.01 V of the run without the monitor (13.033 V). Fails today (400 ticks, 12.091 V)
- [ ] 1.4 C, `controller_period = 100 µs`, 2 ms: observer calls 20 on `pwl` (`dt = 1 µs`) and on `trbdf2`. Fails today (2001 on `pwl`)
- [ ] 1.5 A on `pwl`, `dt = 1 µs`, 10 ms: ticks 200 and 20. Fails today (198 and 20)
- [ ] 1.6 Tick instants on `trbdf2`: every tick time recorded in `duty_history` is within `1e-12·T` of `k·T`; the first is `t_start`
- [ ] 1.7 `pwl`, `T = 2.5·dt`: runs, one warning naming the jitter, count `ceil(span/T)`, `control_tick_jitter_max < dt`; `T = 0.5·dt`: raises
- [ ] 1.8 Coincident ticks fire in registration order (a list appended to by three sources at 1, 2 and 4 kHz reads `[0, 1, 2]` at t = 0 and at every 1 ms)
- [ ] 1.9 Identity: one `ClosedLoop` on `trbdf2` gives states equal (`np.array_equal`) to the pre-change run (reference captured before task 2)

## 2. Composition
- [ ] 2.1 `python/pulsim/_control_plan.py`: `ControlPlan`, `_compose_control`; unit tests for the three lists
- [ ] 2.2 `simulate()`: call it once above the dispatch; delete `__init__.py:2215-2283` composition and `2420-2455`; keep `_inner_observers` for the motor-trace walk
- [ ] 2.3 `controller_period` without an observer: refusal (Open Question 3)

## 3. Variable-step kernel
- [ ] 3.1 `TrBdf2Options::observer_periods`, per-source counters, indexed callback, per-source tick counts in `TrBdf2Stats`; correct the `floor(span/T) + 1` comment
- [ ] 3.2 Binding: `observer_periods` / `observers` lists; the single-source keywords remain and map to one-element lists
- [ ] 3.3 C++ test: three sources at 1, 2 and 4 kHz over 10 ms fire 10, 20 and 40 times at exact multiples

## 4. Fixed-step scheduler
- [ ] 4.1 Tick scheduler (integer `k`, tolerance `1e-9·dt`), tick 0 at `t_start`, refusal for `T < dt`, jitter warning and `control_tick_jitter_max`
- [ ] 4.2 `closed_loops` on `pwl` use `.tick` through the scheduler; `step_observer` + `controller_period` likewise
- [ ] 4.3 `result.control_tick_counts` on both engines
- [ ] 4.4 Update the closed-loop tests whose expected values move; each new value explained in the test (tick count, not a re-recorded number)

## 5. Docs
- [ ] 5.1 `CHANGELOG.md` [Unreleased]: the three BREAKING notes, migration for `controller_period` on `pwl`
- [ ] 5.2 `python/pulsim/AI_GUIDE.md`: tick rule, multi-rate example, removal of the "198 of 200" caveat
- [ ] 5.3 `KNOWN_LIMITATIONS.md` / `docs/known-limitations.md`: fixed-engine tick jitter; controllers still refused on `dsed`
- [ ] 5.4 Docstrings: `simulate` (`controller_period`, `closed_loops`), `bind_pi_to_switch`, `ClosedLoop.period`; stubs via `scripts/generate_stubs.py`
- [ ] 5.5 Remove the strict-xfail markers for these cells in `test_engine_conformance.py` if that change has landed

## 6. Verification
- [ ] 6.1 `ruff check`
- [ ] 6.2 `pytest python/tests`
- [ ] 6.3 `ctest -j2`
- [ ] 6.4 `openspec validate --all --strict`
- [ ] 6.5 Stub check
- [ ] 6.6 Re-run `probe2.py`, `probe6.py`, `probe7.py`; record the lines in the PR
