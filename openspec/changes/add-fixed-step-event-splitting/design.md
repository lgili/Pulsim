## Context

`run_transient` already has the pieces of a split: per-step snapshots
of every history, `cache.solve_at(mask, dt, …)` for an off-nominal
step, and a correction path (`run_transient.hpp:1315-1400`) that
rolls back and re-takes a step as `dt1` under `mask_pre` plus `dt2`
under the post-event mask. It never runs for a turn-off because the
time it is given is wrong, and it does not know about gate edges at
all.

After `add-discontinuity-restart-rule`, a mask change is handled
correctly when it happens at a step start. This change moves the mask
change to where it actually happens.

Evidence: `raw/numerics.md` item 3; `p15_substep.py`, `p16_events.py`,
`p12.py`, `p1_edge.py`; and the numpy prototype `p19_split_proto.py`
(gate edges only), built on the step of `p18_cda_proto.py`.

## Goals / Non-Goals

- Goals: second-order accuracy of the fixed engine for gate edges and
  source jumps anywhere in a step; diode commutation times located to
  a small fraction of dt; `commutation_events` that mean what they say;
  the uniform output grid kept.
- Non-Goals: a general event queue; control ticks and comparator
  events; changing the diode law or its hysteresis; the Newton-loop
  mask chatter; recording samples at event times.

## Decisions

### D1. Where the crossing is read

The event iteration's first solve is taken under the bits the step
started with. If a diode's signal changes sign between `x_prev` and
that solve, the crossing is inside the step and

`t_e = t_prev + dt · s_0 / (s_0 − s_1)`,

with `s = crossing_signal` (`v − V_f` while ON, `v − V_on` while OFF,
the one law every engine uses). For an ON diode `s` is proportional to
its current, so this is the current's zero crossing, the PSIM rule.
The trapezoidal solution is piecewise linear in the states to O(dt²),
so the located time is within O(dt²) of the crossing of the discrete
trajectory.

- Alternatives considered: the post-flip state (today: the estimate
  collapses to `t_prev`); a second-order (Hermite) interpolant using
  the derivative slots (more accurate, and wrong whenever the slots are
  stale, which is exactly after an event); Illinois iteration with
  trial solves as TR-BDF2 does (several off-nominal factors per event).

### D2. Gate edges and source jumps

- A schedule with `next_edge_after(t)` (the C++ schedules:
  `NativePwm2Switch`, `NativeMultiMaskPwm`, the dead-time pair, the
  phase-shift full bridge, three-phase SPWM, combined schedules) is
  asked once per step for the next edge after `t_prev`.
- An opaque `switch_fn` is bisected on `(t_prev, t]` only when
  `switch_fn(t_prev) ≠ switch_fn(t)`, to `1e-3·dt` (10 calls). An even
  number of edges inside one step is not seen; that is today's
  behaviour and is stated in the guide.
- Source jumps come from the table `add-discontinuity-restart-rule`
  builds.

### D3. The split

For the earliest event `t_e` in `(t_prev, t)`:

1. Roll back to the step's snapshot.
2. Sub-step 1, `dt1 = t_e − t_prev`, pre-event mask, trapezoidal
   (`solve_at(mask_pre, dt1)`; on the Newton path the segment is
   assembled at `dt1` exactly as the dt-retry does). If the step was
   itself a restart step, sub-step 1 is two backward-Euler half-steps
   of `dt1/2`. Commit every history with the step and stage used.
3. Apply the event: gate bits from the schedule, diode bits from the
   located crossing.
4. Sub-step 2, `dt2 = t − t_e`: the restart rule, two backward-Euler
   half-steps of `dt2/2` on the matrix at `dt2`
   (`solve_at(mask_post, dt2)`), each with its event iteration.
5. If another event lies in `(t_e, t)`, repeat from 1 on the remaining
   interval, at most `max_events_per_step` (8) times; beyond that the
   remainder is taken as one restart sub-step and an
   `EventIterationBreach` is recorded.

The final commit, the freeze guard and the record happen once, at `t`.

- Alternatives considered:
  - *B: interpolate the state to `t_e` instead of solving sub-step 1.*
    One off-nominal factor instead of two. Prototype: same error as A
    on the RL edge; on the buck 1.1–1.5× A's error (order 2 as well).
    Kept as the fallback if the cost gate of task 6.2 fails.
  - *C: interpolate to `t_e`, take the two half-steps at the nominal
    `dt/2`, interpolate back to the grid, restart again.* No
    off-nominal factor at all (the scheme EMTP-type programs use, from
    memory). Not prototyped; four backward-Euler half-steps per event
    and two interpolations of every device state, including the
    Jiles–Atherton path state, for which interpolation is not defined.
  - *A variable output grid with a sample at `t_e`.* Rejected here:
    the uniform grid is a documented property (`times[k] = t_start +
    k·dt`).

### D4. Snap threshold

An event within 1 % of dt of either end of its interval is not split
(the existing `substep_min_dt`): it is taken at the step boundary and
the restart rule applies. The timing error of a snapped event is at
most 0.01 dt.

### D5. Stateful devices and the single choke point

Both sub-steps go through `run_event_iteration`, which sets
`refresh_dt` and `refresh_stage` for every solve, so the five stateful
devices stamp `dt1` (trapezoidal) and `dt2/2` (backward Euler). The
present correction path calls `cache.solve_at` directly and therefore
solves a circuit with Newton devices linearly; that call is removed.

### D6. One option

`enable_substep_state_correction` already means "split a step at a
commutation". It keeps its name and gains gate edges and source jumps.
It requires the restart rule; set alone, the engine raises an error
that says so.

### D7. Cost

A split step costs sub-step 1 plus two half-steps, each with its event
iteration, against one solve. On the linear path it also needs the
factors `(mask_pre, dt1)` and `(mask_post, dt2)`. With fixed off-grid
edges they repeat every period and the event cache serves them; with a
duty that changes every period they are refactored at every edge. On
the Newton path there is no extra factorisation.

## Risks / Trade-offs

- **What the split is worth after the restart rule.** In the prototype
  the DCM buck is already second order with the restart rule alone
  (diode turn-off resolved to half a step, charge error O(dt²)); the
  split's measured value there is the off-grid gate edge. Whether the
  restart rule alone removes the T/101 outlier of `p15_substep.py` is
  not known; task 1.3 measures it so the split's own contribution is
  on record.
- **The −1.9 mV bias.** `p15_substep.py` shows a DCM error of about
  −0.022 % that does not shrink from T/100 to T/400; its cause is
  unknown (`raw/numerics.md` §4). The acceptance of this change is
  agreement across N, not agreement with the exact value; task 5.3
  investigates and does not promise.
- Two refactorisations per split step on the linear path with a moving
  duty. Mitigation: the measurement gate and alternative B.
- An opaque `switch_fn` is called twice per step and ten more times on
  an edge step.
- `t_estimated` stops being a grid index: `steady_state.py:258` rounds
  it to one and `_diode_state.py:279` reads it; both are reviewed.

## Migration Plan

1. Land behind `enable_substep_state_correction` (default false);
   harness hashes unchanged.
2. Harness run with the restart rule on, with and without the split;
   no reference error may grow.
3. Flip the default in the same release that flips the restart rule,
   or the next; CHANGELOG **BREAKING**; regenerate the goldens in one
   commit.
- Rollback: `enable_substep_state_correction=False`.

## Open Questions

1. Flip together with the restart rule, or one release later?
2. Is alternative B (one refactor) preferred from the start?
3. `max_events_per_step` = 8 and the 1 % snap: acceptable defaults?
4. Should an opaque `switch_fn` be bisected at all, or should the
   split be limited to schedules with `next_edge_after` and the guide
   say so?
