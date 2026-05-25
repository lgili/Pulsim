## 1. ClosedLoop dataclass + bind_pi_to_switch

- [ ] 1.1 Add `pulsim/control.py:ClosedLoop` frozen dataclass with
      fields: `switch_fn`, `step_observer`,
      `duty_history: list[tuple[float, float]]`,
      `error_history: list[tuple[float, float]]`.
- [ ] 1.2 Add `pulsim/control.py:bind_pi_to_switch(builder, *, pi,
      measured, setpoint, switch, freq, t_start=0.0) -> ClosedLoop`.
      Internals:
      - `T_PWM = 1.0 / freq`
      - `last_t = [t_start - T_PWM]` (forces immediate first update)
      - `duty = [0.5]` (initial guess; PI.update will overwrite)
      - `n = builder.num_switches`
      - `idx = builder.switch_index_of(switch) if isinstance(switch, str)
        else int(switch)`
      - Build `switch_fn(t)` that returns `SwitchStateMask(n)` with
        bit `idx` set when `(t % T_PWM) / T_PWM < duty[0]`.
      - Build `step_observer(t, x)` that throttles to `T_PWM`,
        calls `pi.update(setpoint=setpoint, measured=measured(x),
        dt=T_PWM)`, updates `duty[0]`, appends to both histories.
      - Return `ClosedLoop(switch_fn=…, step_observer=…,
        duty_history=…, error_history=…)`.
- [ ] 1.3 Add `bind_pi_to_duty_callable(...)` — same factory but
      returns `(duty_callable, step_observer, history)` tuple
      instead of a switch_fn-bound loop. The duty_callable is a
      0-ary function returning the current duty, suitable for
      callers driving multiple switches from one controller.

## 2. simulate(closed_loops=…) composition

- [ ] 2.1 Extend `pulsim/__init__.py:simulate` to accept
      `closed_loops: Sequence[ClosedLoop] | None = None`.
- [ ] 2.2 When `closed_loops` is non-empty:
      - Verify `switch_fn is None and step_observer is None`,
        else raise
        `ValueError("pass closed_loops OR switch_fn/step_observer,
        not both")`.
      - Compose `switch_fn` via
        `make_combined_switch_fn(num_switches,
                                  [l.switch_fn for l in closed_loops])`.
      - Compose `step_observer` via a fan-out wrapper:
        `def _composed(t, x): [l.step_observer(t, x) for l in closed_loops]`.
      - Pass the composed callbacks to the inner `simulate` call.
- [ ] 2.3 Document the kwarg in the function's docstring with
      a buck + boost dual-output example.

## 3. Tests + benchmark

- [ ] 3.1 `python/tests/test_closed_loop_helper.py` — test cases:
      - Buck closed-loop (V_in=12, V_ref=5) using
        `bind_pi_to_switch` converges to within ±5 % steady-state
        in the last 2 ms of a 20 ms run.
      - Same circuit reaches the same steady-state as the
        hand-rolled version in `scripts/test_cl_buck.py`
        (compare `vout` mean over last 2 ms; ≤ 1 % delta).
      - `simulate(closed_loops=[l1, l2])` with two independent
        loops on a dual-rail buck regulates both rails to their
        respective setpoints.
      - `bind_pi_to_duty_callable` lets a caller drive a
        half-bridge complementary pair (Q_high uses the duty
        directly; Q_low uses 1 - duty).
- [ ] 3.2 Negative-path tests:
      - `simulate(closed_loops=[l], switch_fn=custom)` raises
        `ValueError` with a message mentioning both kwarg names.
      - `bind_pi_to_switch(..., switch="Q_does_not_exist")`
        raises `KeyError` (propagated from `switch_index_of`).
- [ ] 3.3 Benchmark: `python/benchmarks/bench_closed_loop_helper.py`
      — measure runtime of the helper-driven buck vs the
      hand-rolled version over 10 ms / 5 µs dt; assert
      ≤ 5 % regression on median wall time over 10 reps.

## 4. Examples + docs

- [ ] 4.1 Rewrite `python/scripts/test_cl_buck.py` to use
      `bind_pi_to_switch` (target: 20 lines of actual logic,
      excluding imports and the build_plant helper).
- [ ] 4.2 Add or extend `docs/tutorials/closed-loop.md` with two
      walkthroughs: (a) single-loop buck via the helper, (b)
      dual-rail buck via `closed_loops=[l1, l2]`.
- [ ] 4.3 Add a section to `docs/v2/helpers.md` listing
      `pulsim.control.bind_pi_to_switch` as a UX helper alongside
      `scope`, `bode`, `currents`.

## 5. Validation

- [ ] 5.1 `openspec validate add-python-closed-loop-helper --strict`
      passes.
- [ ] 5.2 Existing `python/tests/` pytest suite green (no
      regressions in `PIController` API or the bare
      `simulate(switch_fn=, step_observer=)` path).
- [ ] 5.3 PulsimGUI sanity check: rebuild
      `feat/ux-cleanup-and-scripts`, point
      `sim_buck_closed_loop.py` at `bind_pi_to_switch`, confirm
      `V_out` reaches 6 V (removing the limitation that script
      currently documents).
