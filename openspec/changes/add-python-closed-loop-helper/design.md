## Context

Closed-loop control in pulsim 1.4 is implemented entirely on the
Python side via `step_observer`. The kernel exposes:

- A `step_observer(t, x)` callback after each accepted step.
- A `switch_fn(t) -> SwitchStateMask` that decides switch state.

The pattern works but every script duplicates the same plumbing,
and the closure-capture-via-mutable-list idiom (`current_duty =
[0.50]`) is unidiomatic Python. This proposal packages the
plumbing into a reusable primitive that:

1. Owns the controller's mutable state internally (no shared lists).
2. Throttles the controller's `update()` to the PWM period
   automatically.
3. Generates the PWM phase comparison without the caller writing
   `(t % T) / T < duty`.
4. Composes cleanly when a circuit has multiple independent loops.

## Goals / Non-Goals

### Goals

- A single function call replaces the 30-line boilerplate.
- Multiple loops on one builder compose without name clashes.
- Loop's internal state is observable post-run (duty/error
  history) for plotting and tuning.
- Zero performance regression vs. the hand-rolled pattern — the
  helper is just a closure factory; the inner loop is the same
  `pi.update()` + bit-set.

### Non-Goals

- Generic controller framework (state machines, feedforward,
  cascaded loops). This is a narrow helper for "PI → PWM duty →
  switch bit"; cascaded / feedforward stays manual until a real
  use-case demands it.
- Multi-rate control (sampling at a multiple of the PWM rate).
  Future extension.
- Adaptive / gain-scheduled PI. Out of scope.

## Decisions

### Decision: Return a frozen dataclass, not a class with `__call__`

Bundling `switch_fn` + `step_observer` + history lists into a
dataclass keeps the interface flat: callers pass `loop.switch_fn`
and `loop.step_observer` directly to `simulate()`. The mutable
history attributes are append-only lists; the dataclass itself is
`frozen=True` so the references can't be reassigned. The
histories are populated by the captured closure inside the
factory.

**Alternative considered**: a `ClosedLoop` class with `__call__`
that multiplexes `(t)` vs `(t, x)` based on arity. Rejected
because the dispatch is brittle
(`signature.parameters` inspection) and the two callbacks have
different invocation cadence (one per step, one per accepted
step).

### Decision: Throttle the PI to one update per PWM period

The factory captures `T_PWM = 1.0 / freq` and keeps a `last_pi_t`
in the closure. The observer compares `t - last_pi_t` and skips
the update otherwise. This matches the canonical pattern in
`scripts/test_cl_buck.py:104` (the throttle exists exactly to
prevent the loop from chasing PWM ripple).

**Alternative**: trigger the update on a derived sample-and-hold
edge inside `switch_fn`. Rejected — `switch_fn` is supposed to be
side-effect-free, and the cleanest split is "switch_fn reads
state, step_observer mutates state".

### Decision: `closed_loops=` kwarg on `simulate()` for composition

When a circuit has two independent loops (e.g., a dual-output
flyback with two regulated rails), the caller passes
`closed_loops=[loop_a, loop_b]`. `simulate()` then:

1. Combines `switch_fn`s with
   `make_combined_switch_fn(n_switches, fns)`.
2. Composes `step_observer`s into a single fan-out wrapper that
   calls each in sequence.
3. Rejects the combination with explicit `switch_fn=` or
   `step_observer=` to avoid silent override.

**Alternative**: have callers compose manually. Rejected because
the combinator is the same 5 lines every time, and getting it
wrong silently drops a controller.

### Decision: Accept `switch=` (name) when named lookups are available, `switch_idx=` (int) otherwise

The factory has two parameter spellings:

- `bind_pi_to_switch(..., switch="Q1", ...)` — preferred, resolved
  via `builder.switch_index_of("Q1")`.
- `bind_pi_to_switch(..., switch_idx=0, num_switches=N)` —
  fallback for callers without the named-lookup capability.

When `add-python-named-lookups` lands, `switch=` resolves through
the builder. Until then, the function accepts both spellings; the
helper raises if neither is provided.

### Decision: Histories are `list[tuple[float, float]]` not numpy arrays

The history lists are appended to in `step_observer`, which fires
hundreds of times per simulation. Allocating numpy arrays on each
append is wasteful; growing Python lists and converting at the
end (when the caller wants) is the standard idiom for online
collection. The factory documents how to convert:

```python
import numpy as np
duty_t, duty_v = np.asarray(loop.duty_history).T
```

## Risks / Trade-offs

- **Latency of the Python step_observer.** The throttled update
  costs ~10 µs per PWM cycle (PI math + closure overhead). At
  100 kHz PWM that's ~1 % of a 10 µs solver step. Acceptable. We
  benchmark in `tasks.md §3`.
- **Implicit time-base coupling.** The helper assumes the switch
  frequency is constant. Variable-frequency control (e.g.,
  boundary-mode flyback) needs a different primitive.
- **Multiple loops on the same switch.** Rejected at the API
  level — `bind_pi_to_switch(..., switch="Q1")` twice raises
  `ValueError`. Callers that need this pattern compose duty
  callables themselves via `bind_pi_to_duty_callable`.
- **History growth.** A 100 ms run at 100 kHz PWM produces 10 000
  history entries (~ 160 kB at 16 bytes/tuple). Fine for
  interactive use; long stress runs should pass
  `record_history=False` (future extension if it ever bites).

## Migration Plan

1. Land the helper as additive API in `pulsim.control`.
2. Rewrite `python/scripts/test_cl_buck.py` to demonstrate the
   helper (the existing script becomes 20 lines + the helper).
3. Update `docs/tutorials/closed-loop.md` (or add it if missing)
   to point at the helper first, the manual closure pattern
   second.
4. Coordinate with PulsimGUI: GUI's "PI + PWM-generator virtual
   components" wiring becomes a single `bind_pi_to_switch` call,
   removing the `sim_buck_closed_loop.py` known-limitation.

No deprecation needed — the manual closure pattern keeps working.
