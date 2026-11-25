## Context

A digital controller samples at `k·T`. Pulsim offers it through two
objects:

- `ClosedLoop` (`bind_pi_to_switch`): a gate function, a throttled
  `step_observer`, its unthrottled `.tick`, and `period`.
- A user `step_observer(t, x)`, optionally with `controller_period`.

`simulate()` resolves them in two places:

| | Fixed (`pwl`), `__init__.py:2420-2455` | Variable (`trbdf2`), `__init__.py:2215-2283` |
|---|---|---|
| Gate composition | `make_combined_switch_fn` | hand loop with `m | f(t)` (crashes) |
| Loop observer | throttled `step_observer`, every step | `.tick`, at scheduled instants |
| Cadence | each loop's own throttle | one period: `min(periods)` for all |
| `controller_period` | not read | read |

The kernel side of the variable path is sound for one source:
`trbdf2_transient.hpp:953-956` keeps an integer `tick_k`, lands the step
on `t_start + tick_k·T` (`:1115-1118`) and fires before the step's mask
is sampled (`:1065-1078`). It fires for every `k` whose instant is
before `t_end`, so the count is `ceil(span/T)`: 40 for 20 ms at 2 kHz
(measured), not the `floor(span/T) + 1` its comment states.

## Goals / Non-Goals

- Goals: one definition of when a tick fires; the same count on both
  engines; several sources at several rates; `controller_period`
  honoured or refused, never ignored; one composition function.
- Non-Goals:
  - Tick offsets, block ordering by rate, algebraic-loop detection
    (`add-compiled-control-graph`).
  - Controllers on `dsed` (still refused).
  - Sub-step tick landing on the fixed engine
    (`add-fixed-step-event-splitting`, `add-unified-event-queue`).
  - The C-block and block-chain observers, which already carry their own
    step counter and stay every-step callbacks.

## Decisions

### 1. Tick instants

Source `j` with period `T_j > 0` ticks at `τ_{j,k} = t_start + k·T_j`,
`k = 0, 1, …`, while `τ_{j,k} < t_end`. `τ` is computed from the integer
`k` each time, never accumulated. Count: `ceil((t_end − t_start)/T_j)`.
The tick at `t_start` sees the initial state; a duty it sets applies
from the first step.

- Alternative *"first tick after one period"*: rejected, the kernel
  already ticks at `t_start` and a controller that waits a period starts
  open loop.
- Alternative *"include the tick at `t_end`"*: rejected, nothing is
  simulated after it.

### 2. Variable-step engine

`TrBdf2Options::observer_periods` (vector) replaces `observer_period`;
the observer callback receives the source index. Per source: `k_j`,
`t_next_j`. Each step: fire every source with `t ≥ t_next_j − h_min`,
in index order, then invalidate the edge cache once (as today); cap `h`
at `min_j t_next_j − t`. With one source the arithmetic is the current
one, so single-loop runs are bit-identical.

- Alternative *"fire everything at `min(T_j)` and let Python skip the
  slow ones"*: exact only when the periods are integer multiples, and it
  is the defect being fixed.
- Alternative *"a GCD base period in Python"*: floating-point GCD of
  unrelated periods is not robust, and it adds step boundaries nobody
  asked for.

### 3. Fixed-step engine

The engine has a grid `t_n = t_start + n·dt`. Tick `τ_{j,k}` is
delivered at the first `t_n ≥ τ_{j,k} − 1e-9·dt`. `k` advances by one
per delivery, so the count is exact and nothing drifts. Tick 0 is
delivered at `t_start`, before the first step.

- `T_j < dt` → refused (two ticks would fall in one step).
- `T_j / dt` not an integer (relative 1e-9) → allowed, one warning per
  run: ticks are late by up to `dt`; `result.control_tick_jitter_max`
  records the largest lateness.
- Alternative *"refuse non-multiples"*: stricter, but many existing
  closed-loop scripts choose `dt` and `f_sw` independently and run
  today. Left to the owner (Open Question 1).
- Alternative *"fix the throttle with a tolerance"*: keeps a
  last-tick-based rule, which is what accumulates the loss.

### 4. Composition

`_compose_control(builder, closed_loops, step_observer,
controller_period, switch_fn) -> ControlPlan` in a new private module,
called once above the engine dispatch:

- `switch_fn`: `make_combined_switch_fn(n_switches, [loop gates…, user
  switch_fn])`, or the single function unchanged when there is one.
- `ticks`: `[(loop.step_observer.tick, loop.period)…]`, then
  `(step_observer, controller_period)` if a period was given (or the
  observer carries `.period`).
- `every_step`: `step_observer` when it has no period.

Engines consume the plan: `pwl` wraps `ticks` in the scheduler of
decision 3 and calls `every_step` each step; `trbdf2` passes `ticks` to
the kernel and refuses a non-empty `every_step` (as today). Order at a
shared instant: closed loops in list order, then the user observer —
the order the fixed path already uses.

### 5. Reporting

`result.control_tick_counts` (list of integers, registration order) on
both engines; `TrBdf2Stats::n_ctrl_ticks` stays as their sum.

## Risks / Trade-offs

- Every `pwl` closed-loop golden moves (two extra ticks per 200 and a
  tick at `t_start`). Mitigation: land before
  `restore-reference-benchmark-harness` freezes the fixed-engine
  checkpoints, or re-freeze in the same PR.
- A user observer that relied on being called every step while passing
  `controller_period` on `pwl` changes behaviour → CHANGELOG entry and a
  one-line migration (drop the keyword).
- Extra step boundaries on `trbdf2` for the slow sources are few (one
  per tick) and were already there for the fast one.
- `make_combined_switch_fn` returns a `SwitchSchedule`; when a member
  has no `next_edge_after` the combined schedule has no edge oracle and
  the engine samples it, which is today's behaviour for a Python gate.

## Migration Plan

1. `_compose_control` with both branches calling it (fixes the crash).
2. Kernel period list and binding (fixes multi-rate).
3. Fixed-engine scheduler and `controller_period` (the BREAKING step).
4. Docs, CHANGELOG, re-run of the three probes.

Steps 1–2 change no result that was correct. Step 3 can be released
separately if the owner wants the number change in a later version.

## Open Questions

1. Fixed engine, `T_j` not a multiple of `dt`: run with a jitter warning
   (proposed) or refuse and ask for `dt = T_j/n`?
2. Fixed engine, tick 0 at `t_start` with the initial state (proposed,
   equal to `trbdf2`) or after the first step (today)?
3. `controller_period` given without a `step_observer`: refuse
   (proposed) or ignore as today?
4. Order of coincident ticks: registration order (proposed). Is
   "slowest first" (outer loop before inner loop) wanted instead?
5. Should the fixed-engine number change (decision 3) ship with this
   change or wait for the release that carries
   `add-discontinuity-restart-rule`, so users see one migration?
