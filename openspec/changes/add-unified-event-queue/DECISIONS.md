# Decisions taken while implementing add-unified-event-queue

## 1. The queue (task 2.1) — `core/include/pulsim/events/event_queue.hpp`

- **Linear scan, not a binary heap.** A run registers a handful of sources
  (one gate source, one periodic source per control rate, the end of the
  run). A scan of that many slots is cheaper than heap maintenance and
  needs no reordering on `invalidate`. Marked in the header as a
  `shortcut:` with the upgrade condition (more than ~32 sources).
- **Three source shapes instead of one `TimeSource` concept.**
  `add_periodic(kind, Rate)` (integer counter, the k-th event is
  `(t0 + offset) + k·period` computed directly), `add_function(kind,
  next)` (an oracle "first event strictly after t": the analytic gate
  edges of `SwitchSchedule`, `NativePwm2Switch`, `NativeMultiMaskPwm`),
  `add_located(kind)` (events a locator finds: the bisection of an opaque
  gate function) and `add_once(kind, t)` (the end of the run). They cover
  every source the engines have today; a concept with type erasure would
  add code and no caller.
- **Half-open run.** A periodic source fires at every `k ≥ 0` with
  `t_k < t_end` (the `fix-control-tick-semantics` rule, open question 6):
  about `ceil((t_end − t0 − offset)/period)` events; when that quotient is
  within rounding of an integer the `t_k < t_end` test decides (T = 1e-3/3
  over 50 ms gives 151, not 150). An exhausted source reports `+inf`.
- **Simultaneity is the caller's tolerance, applied as today.**
  `take_due(t, tol, kinds)` takes a source's event when `t >= t_k − tol`,
  written exactly that way: the TR-BDF2 engine passes its `h_min`, the
  comparison it used before, so the tick instants and counts did not move.
  `dsed/time_eps.hpp`'s rule is the natural default for new callers; it is
  not imposed because it would have changed which TR-BDF2 steps fire a
  tick.
- **Order at one instant (D2)** is `Kind` order (sample, tick, gate edge,
  breakpoint, end), then registration order inside a kind. Tested with
  sources registered in the reverse order.
- **Lazy oracle refresh** (`next_after(id, t)`): a function source asks its
  oracle only when its pending edge is invalid (a tick invalidated it) or
  no longer after `t`; a located source drops an edge it has passed. This
  is the TR-BDF2 edge cache's rule moved into the queue, unchanged.
- **Allocation**: `start` sizes the take buffer (the only allocation);
  `take_due`, `next_after`, `next_of`, `invalidate`, `locate` allocate
  nothing (`[event_queue]` case of `core/tests/perf/test_zero_alloc.cpp`:
  10 000 steps with two tick rates, a six-edge gate oracle invalidated
  every 97 steps, zero `operator new`). The engine builds its queue per
  run (setup); the test covers the queue, not the engine's loop.
- **Bit-equality across compilers** is not claimed: `(t0 + offset) +
  k·period` may be contracted to an FMA by one compiler and not another
  (one ulp). Within one binary every engine computes the same expression;
  the unit tests compare against the queue's own `periodic_time` exactly
  and against an independent product to 1e-12 of a period.
- `invalidate(id)` leaves periodic and one-shot sources alone (their
  events depend on no register), like `invalidate(Kind)`. An oracle that
  answers a time not after its own event is asked again by the next
  `next_after` instead of being re-taken (an endless loop otherwise).

## 2. TR-BDF2 on the queue (task 2.4)

- Tick sources (one per controller rate, ids 0..n−1 = the source index
  `j` of `tick_fn(j, t, x)`), the gate-edge source (the analytic oracle
  when the binding found one, else a located source the sampling and
  bisection write with `locate`) and the end of the run are registered on
  one queue. The tick block is `take_due(t, h_min, ControlTick)`; the tick
  landing, the post-event commit guard and the post-event sample guard
  read `next_of(ControlTick)`; the edge landing reads
  `next_after(edge_src, t)`; a tick invalidates the gate source; the step
  clamp reads the end-of-run event (`next_of(EndOfRun)`, equal to
  `t_end`); the loop condition still reads `t_end`.
- **The three landing clamps keep their own comparisons.** The tick clamp
  is `t_tick − t < h`, the breakpoint and edge clamps `t_x <= t + h`, in
  that order, each followed by `max(·, h_min)`, and the engine needs to
  know which one landed (`landed_on_edge`, `landed_on_bp`). Replacing them
  by one `h = min(h, next_landing() − t)` changes which comparison decides
  a tie at the last bit, and so the accepted-step sequence; the migration
  invariant is an identical sequence. The single landing belongs to the
  unified stepper (`refactor-unified-transient-stepper`), which may change
  sequences.
- **Source breakpoints stay in the engine** (`next_breakpoint_after`):
  they are recomputed from `t + h_min/2` at every step, which a cached
  pending event would not reproduce at the last bit. Moving them is task
  7.1.
- Oracle (task 2.5): `.scratch/probe/golden_baseline.py` (the golden
  compare of earlier batches plus 13 cases for this change: one and two
  closed loops, a loop plus a 20 kHz monitor, a periodic observer, an
  opaque Python gate, a dead-time pair and a pulse-driven RC on trbdf2;
  closed loops and a dead-time pair on pwl; an opaque gate and a closed
  loop on dsed). All 34 byte-identical (times, states, switch states,
  every TR-BDF2 counter, kernel counters) against the tree before the
  change (a50a2c77).

## 3. Rate offsets on trbdf2 (part of task 4.1)

- `TrBdf2Options::observer_offsets` (parallel to `observer_periods`,
  empty = 0) registers `Rate{T_j, offset_j}` on the queue;
  `ControlPlan.for_variable_step` passes the control graph's group
  offsets (zeros for closed loops and observers), and the
  `control_rate_offset` refusal is removed. A `compute_delay` time is
  still refused (it needs a second event per tick).
- The graph's tick handler learned `t0` from its first call; with an
  offset that call is at `t0 + min offset`, so `observe_tick` now sets
  `t0 = t − min_group_offset()` (exact at t_start = 0; at other starts it
  can differ by an ulp, inside the graph's 1e-9·period tick tolerance;
  tested at t_start = 3e-4 s).
- Not done in 4.1: `add_tick` / `TickInfo{scheduled_t, k}` (the handlers
  still receive the step time, which on trbdf2 is the tick time); the
  fixed engine's ticks are still Python (`FixedStepTicks`).

## 4. The control graph's PWM gate is a modulator with edges (towards group 6)

- `ChainPwmGate::next_edge_after(t)` is the `pwm_phase` crossing of each
  leg for the duty its sink holds now (the arithmetic of
  `NativePwm2Switch`); valid until the graph's next tick. The engines
  already re-ask an oracle after a tick (trbdf2: the queue's
  `invalidate`; pwl: the observer call resets its edge cache), which is
  the duty-register invalidation of D3 without a separate register type.
  `CarrierPwm` / `DutyRegister` / `LatchPolicy` are not added: the gate
  with its chain sink is that pair, with `Immediate` latching.
- Results that change (documented, not bit-identical): closed loops on
  trbdf2 (edges landed instead of bisected; the step is no longer capped
  at twice the narrowest sampled pulse) — cl1/cl2/clmon of the golden
  compare move by at most 6.4e-2 V on a switch node at one sample next to
  an edge (sample instants differ by 7e-8 s), output 1e-9 relative;
  pwl with `enable_substep_state_correction=True` now splits at the
  loop's edges. Reference status: `cl_boost_pi` error -> agree on trbdf2
  and pwl+split, disagree (0.058) on pwl / pwl+restart (it had no
  reference before); `ll15_boost` trbdf2 3.9e-4 -> 1.24e-3 (tol 1e-3).
  Freezes updated in `benchmarks/manifests/v2_triage.yaml`.
