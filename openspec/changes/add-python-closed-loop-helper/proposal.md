## Why

Pulsim 1.4's retirement of the in-kernel `pi_controller` /
`pwm_generator` virtual-component blocks means every closed-loop
script today reinvents the same ~30 lines of Python boilerplate:

1. Carry the controller's output in a mutable single-element list
   so `switch_fn` and `step_observer` share state without
   `nonlocal` tricks (`current_duty = [0.50]`).
2. Throttle the PI update to one tick per PWM period
   (`if t - last_pi_t[0] < T_PWM: return`).
3. Hand-roll the PWM phase test inside `switch_fn`
   (`if (t % T_PWM) / T_PWM < current_duty[0]: mask.set(idx, True)`).
4. Wire the step observer to call `pi.update(setpoint, measured,
   dt)` and update the shared mutable.

`scripts/test_cl_buck.py` (in `python/scripts/`) is the canonical
template — every new closed-loop example copies it. PulsimGUI's
`scripts/sim_buck_closed_loop.py` documents this exact gap as a
"known limitation pending PR #9" because the GUI's closed-loop
topologies all relied on the retired in-kernel blocks.

The boilerplate is verbose, error-prone (forgetting the throttle
yields a loop that fights ripple), and prevents the GUI from
offering "drop a PI on this node" as a built-in interaction.

## What Changes

- **NEW**: `pulsim.control.bind_pi_to_switch(builder, *, pi,
  measured, setpoint, switch, freq, t_start=0.0) -> ClosedLoop` —
  factory that packages a PI controller + PWM-driven switch into a
  `ClosedLoop` dataclass exposing the `switch_fn`, `step_observer`,
  and per-tick history lists.
- **NEW**: `pulsim.control.ClosedLoop` — frozen dataclass with
  fields:
  - `switch_fn: Callable[[float], SwitchStateMask]`
  - `step_observer: Callable[[float, np.ndarray], None]`
  - `duty_history: list[tuple[float, float]]` — `(t, duty)`
  - `error_history: list[tuple[float, float]]` — `(t, e)`
- **NEW**: `pulsim.simulate(..., closed_loops=[loop1, loop2, …])`
  — optional kwarg accepting one or more `ClosedLoop` instances.
  The helper composes their `switch_fn` / `step_observer`
  automatically (via `make_combined_switch_fn` plus a step-observer
  fan-out wrapper). Mutually exclusive with explicit
  `switch_fn=` / `step_observer=`.
- **NEW**: `pulsim.control.bind_pi_to_duty_callable(...)` — same
  primitive but returns a `(duty_callable, step_observer, history)`
  triple instead of a switch-bound loop. Useful when one controller
  drives multiple switches (e.g., a half-bridge pair with
  complementary PWM).

## Impact

- **Affected specs**: `python-bindings`
- **Affected code**:
  - `python/pulsim/control.py` — add `bind_pi_to_switch`,
    `bind_pi_to_duty_callable`, `ClosedLoop` dataclass.
  - `python/pulsim/__init__.py:simulate` — accept `closed_loops=`
    kwarg; reject the combination with `switch_fn=` /
    `step_observer=`.
- **Downstream**: PulsimGUI can wire the GUI's PI + PWM-generator
  virtual components straight into `bind_pi_to_switch` and drop
  the "closed-loop not converging" limitation in
  `scripts/sim_buck_closed_loop.py`. `scripts/test_cl_buck.py`
  collapses from ~70 lines to ~20.
- **No breaking changes** — purely additive. Existing scripts
  that hand-roll the closure pattern keep working.

## Dependencies

This proposal benefits from `add-python-named-lookups` for the
`builder.switch_index_of(name)` lookup so callers can pass
`switch="Q1"` instead of `switch_idx=0`. If named lookups land
first, the factory uses `switch_index_of` internally; if both ship
together, the implementation order in `tasks.md` reflects that.
When neither is available, callers fall back to the
`switch_idx=` + `num_switches=` pair (still supported).
