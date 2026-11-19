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
- **Bit-equality across compilers**: `(t0 + offset) + k*period` is two
  roundings; the tree is built with `-ffp-contract=off` (CMakeLists.txt)
  and MSVC's /fp:precise does not contract, so no compiler fuses it into an
  FMA and every engine, platform and the old Python scheduler compute the
  same double (see 5). The unit tests compare against the queue's own
  `periodic_time` exactly and against an independent product to 1e-12 of a
  period.
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

Superseded at the merge of integration batch 11 (a0ae6bb8). Slice 1
implemented the gate's edge declaration (db687962) at the same time as the
control graph's G23, which batch 11 merged. The merge keeps ONE
implementation, G23's: `ChainPwmGate::next_edge_after`, the adapter branch
in `switch_fn_adapter.hpp`, and `make_combined_switch_fn` building a
`SwitchSchedule` with `follows_controller = true`. What this change adds
on top is only the consumer: trbdf2 reads that oracle through the queue's
gate source, and a tick invalidates it (the duty-register invalidation of
D3 without a separate register type; `CarrierPwm` / `DutyRegister` /
`LatchPolicy` are group 6). Slice 1's test file
`python/tests/test_control_graph_gate_edges.py` is kept: it checks the
edges equal `make_pwm_switch_fn`'s exactly, that the mask is constant
between edges, the trbdf2 counters of a landed closed loop, and the
combination.

The closed-loop reference rows are those of a0ae6bb8 (cl_boost_pi agree on
trbdf2 and pwl+split; ll14 / ll15 disagree as frozen there). Slice 1 called
the ll14/ll15 trbdf2 movement a cost of losing the gate-sampling step cap;
the control-graph agent's diagnosis replaces that: trbdf2 records no
post-edge sample when a rising edge falls on a control tick (the reference
and the cell then differ on V(sw) between samples), plus a sticky
narrowest-pulse cap. Both are fixed in `trbdf2_transient.hpp` by the
TR-BDF2 track, not here; no step cap from the modulator's period is added.

## 5. The fixed engine's ticks on the queue (group 5), step hooks, breakpoints

- **Ticks.** `run_transient(..., const events::TickPlan* ticks, events::StepHooks* hooks)`:
  the kernel registers one periodic `ControlTick` source per Python tick source
  and runs `take_due(t, 1e-9*dt)` at every observer call (the priming call at
  `t_start` included), BEFORE the observer. The queue is started with
  `t_end - 1e-9*dt` as its end, which is the old Python rule
  (`due < t_end - tol`) written once. `_control_plan.FixedStepTicks` is
  deleted; `ControlPlan.fixed_tick_kwargs()` passes `tick_periods` /
  `tick_fns`; counts and lateness come back as
  `SimulationResult.kernel_tick_counts` / `kernel_tick_jitter_max` and are
  copied into `control_tick_counts` / `control_tick_jitter_max` as before.
- **Tick instants are the same doubles.** The Python scheduler computed
  `t0 + k*T` (two roundings, no FMA); the queue computes
  `(t0 + 0.0) + k*T` and the whole tree is built with `-ffp-contract=off`
  (CMakeLists.txt; MSVC /fp:precise does not contract), so the instant is
  the same double, not merely a correctly rounded neighbour. (This also
  corrects DECISIONS 1's "may be contracted to an FMA": not with this
  build.) Drift: event k is computed from k directly, so the error is one
  rounding of `k*T` plus one of the sum at any k; over 1e9 periods the
  relative error stays ~2 ulp of t_k, with no accumulation.
- **Order kept.** Before, the fixed engine's observer ran, per step: the
  internal drivers simulate() puts in front (MMC arms, `on_cache` hooks),
  the closed-loop graphs (label time), the Python tick sources, the
  every-step observers, the C blocks. The kernel keeps that order: what ran
  before the ticks is `TickPlan::before` (`run_transient(tick_before=)`),
  what ran after is the observer. (A first version ran the ticks before
  the whole observer; the review found it moved MMC and `on_cache` runs.)
  Graph-plus-Python-tick runs now compose the graphs into one C++ observer
  instead of a Python loop over them; covered by the golden cases
  clmon_pwl / clmon_split.
- **Speed.** Python ticks used to put a Python object (`FixedStepTicks`) in
  the observer, so every step crossed into Python even when no tick was due;
  now Python is entered only at a tick, and a run whose other observers are
  all C++ graphs takes the composed C++ observer.
- **Mask reuse between oracle edges (5.1, second half) is not done.** Reusing
  the mask of `switch_fn(t_last)` until the oracle's next edge is bit-identical
  only if the oracle and the mask function agree to the last bit at an edge
  that falls on a grid point (a 100 kHz PWM at dt = 1e-7 puts every edge on
  the grid). `NativePwm2Switch` shares `pwm_phase` between both, a Python
  `next_edge_after` need not; the gate memo already makes one call per
  distinct time. Left for the unified stepper, which may change sequences.
- **Event splitting reads the same oracle** (`next_edge_fn` with its
  passed/invalidated cache); moving that cache into a queue source is a
  rename with identical semantics and was not needed for the gate.
- **Step hooks** (`events/step_hooks.hpp`, the commutation callback the
  in-kernel event losses hook on, its task 3.1): `on_accept(t_prev, t, x,
  mask)` once per accepted step and `on_commutation({t, before, after,
  x_before, x_after})` when the solved mask changes between two accepted
  steps. Wired in pwl (dynamic and static paths) and trbdf2 (the three
  accept sites: normal step, post-edge commit, diode landing; the
  post-event probe sample, which trbdf2 may drop later, is not an accepted
  step). dsed is not wired. No listener: one branch per step, results
  unchanged. Zero allocation after `start` (tested on pwl).
- **Source breakpoints and the run end on trbdf2's queue (7.1, trbdf2 part).**
  The corner table is a `SourceBreakpoint` function source asked at
  `t + h_min/2`, refreshed only when passed; the loop end reads the
  `EndOfRun` event. Every corner table holds offset 0, so a cached corner
  P > q implies q sits in the same period as when P was found and a fresh
  query returns P (review). "No corner" (+inf) is not cached: a single-
  offset table (PWM source at duty 0 or 1) with a user h_min below
  ulp(t) can answer +inf once from the floor index, and the old per-step
  query recovered on the next step.

## 6. A short pulse at a period boundary (the exact engine's dropped pair)

- The exact engine lost both edges of a PWM at D = 1 − 1e-7 (a 1e-12 s
  pulse at the period end) and gave 0 W. Its scheduler asks
  `next_edge_after(t)` and skips an answer that is `near_time` of the edge
  it just fired (`dsed/scheduler.hpp` `next_time_event_` / the `past`
  lambda): two distinct edges closer than its time tolerance are taken for
  one.
- The queue consumes events by identity, not by time: a function source
  holds ONE pending edge; `take_due` takes it and refills from
  `next(ev.t)`, strictly after the edge just taken, however close. An
  engine that calls `take_due(t, tol)` until it returns nothing gets both
  edges of the pulse, in order, each with its own `ev.t`; a tolerance wider
  than the pulse still yields two events. Tested in
  `core/tests/modes/test_event_queue.cpp` with the real `NativePwm2Switch`
  at D = 1 − 1e-7 (tolerances 0, 1e-15, 1e-11) and with a one-ulp pulse at
  1000 period boundaries; both exact (integer / nextafter arithmetic).
- `scheduler.hpp` is owned by the exact-engine agent and is being changed
  there; it is NOT edited here. What its consumption site needs: register
  the switch function's oracle (`next_edge_after`) and the source corners
  (`next_source_corner_after`) as `add_function(Kind::GateEdge, …)` /
  `add_function(Kind::SourceBreakpoint, …)` on one `EventQueue`; replace
  `next_gate_edge_(t)` by `queue.next_landing()` after
  `queue.next_after(id, t)` on each source; at a landed edge call
  `take_due(t_gate, eps)` in a loop and fire each event at `ev.t` in the
  order returned (Kind, then registration); drop the `near_time(r,
  last_time_event_t_)` skip, which identity consumption makes unnecessary;
  `invalidate(GateEdge)` after a control tick. Opaque Python gates keep
  their poll until the shared `SampledGate` adapter (group 3) exists.

## 7. Gate sources and the carrier modulator (tasks 2.2, part of 6.1)

`core/include/pulsim/events/gate_source.hpp` holds both, one header
instead of the `gate_source.hpp` + `modulator.hpp` pair the tasks name:
the modulator is a gate source and nothing else uses it yet.

- A source states `owned_bits`, `resolution` (0 = exact edges), `apply`
  (writes only its bits) and `next_edge`. `GateSet` composes BY INDEX: a
  bit outside the circuit, or one already owned, is refused naming the bit
  and both sources; an unowned bit stays OFF. `to_schedule()` gives the
  engines a `SwitchSchedule`, with edges only when every source is exact,
  so the engines need no new entry point.
- `CarrierPwm` reuses `pwm_phase` and `make_*_fn`'s thresholds, so for a
  constant duty its masks and edges equal `make_pwm_switch_fn` /
  `make_dead_time_pwm_pair_fn` bit for bit (tested on grid points, edges
  and their one-ulp neighbours).
- `DutyRegister::AtCarrierStart` latches by carrier PERIOD INDEX (with the
  carrier's own snap), not by comparing t with the period-start time, so
  the register and the modulator never disagree about which period an
  instant within the snap tolerance belongs to. The period start is always
  one of `next_constant_crossing`'s candidates, so a latch is never missed.
- Not done: `EdgeList`, the Python bindings, and wiring `bind_pi_to_switch`
  (6.2) — the control graph's PWM gate (section 4) already gives the
  closed-loop rows their edges, and moving it onto `CarrierPwm` must keep
  a0ae6bb8's rows byte-identical; that is the next step, not this one.

## 8. Published signatures (for the control-graph and event-losses agents)

Stable as of this commit; all in `namespace pulsim::events`.

| need | signature | header |
|---|---|---|
| tick registration (control graph 6.1) | `std::uint32_t EventQueue::add_periodic(Kind::ControlTick, Rate{period, offset, origin = NaN}, bool lands = true)`; engine entry: `run_transient(..., const TickPlan* ticks)` with `TickPlan{std::vector<Rate> rates; fn(std::uint32_t j, const TickInfo& {scheduled_t, k}, Real t, const Vector& x); before(Real t, const Vector& x)}`; trbdf2: `TrBdf2Options::observer_periods / observer_offsets / observer_origin` | `event_queue.hpp`, `run_transient.hpp`, `trbdf2_transient.hpp` |
| modulator latch and dead time as queue events (control graph 6.2) | `CarrierPwm{T, phase, const DutyRegister* duty, Index hs, Index ls = -1, Real dead_time = 0}`, `DutyRegister{value, LatchPolicy::{Immediate, AtCarrierStart}, T, phase}::write(Real v, Real scheduled_t)`; `GateSet::add(gate, name)` then `GateSet::to_schedule()` gives the engines the edges (dead-time edges and the carrier start that latches a held duty are among them); after a write, `EventQueue::invalidate(edge_src)` (trbdf2) | `gate_source.hpp` |
| commutation callback (event losses 3.1) | `StepHooks::on_commutation(std::function<void(const Commutation&)>)`, `Commutation{Real t; const SwitchStateMask& before, after; const Vector& x_before, x_after}`; `StepHooks::on_accept(fn(Real t_prev, Real t, const Vector& x, const SwitchStateMask& mask))`; engines: `run_transient(..., TickPlan*, StepHooks* hooks)`, `run_transient_trbdf2(..., StepHooks* hooks)` | `step_hooks.hpp` |

**Tick phase on resume.** `Rate::origin` (NaN = the run's t0) puts a
periodic source on the grid `(origin + offset) + k*period` and starts it at
the first instant `>= t0 - t0_tol` (`EventQueue::start(t0, t_end,
t0_tol)`; the fixed engine passes its take tolerance `1e-9*dt`, the same
tolerance that left that tick out of the previous run's end). `TickInfo::k`
is then the index on the original grid. Python: `run_transient(...,
tick_origin=)` and `run_transient_trbdf2(..., observer_origin=)`. A run
resumed at 1.53 ms with origin 0 ticks at 1.6, 1.7, 1.8 ms. On trbdf2 these
are bit-equal to the uninterrupted run (tested). On the fixed engine the
scheduled instants are bit-equal, and the step time is the first grid point
of the resumed grid at or after each instant. NOT done: carrying the origin
in `resume_from` snapshots and lifting `simulate()`'s trbdf2 resume
refusal. That is the caller's wiring: pass the original `t_start` as the
origin.

**`control_tick="state"`** (control-graph branch, not yet on
feat/core-integration). The kernel tick sources call `fn(j, info, t, x)`
with the step's (t, x), which is the "label" mode. "state" mode samples at
the state time; when that branch merges, it maps onto `TickPlan::fn`
reading `info.scheduled_t` against `t`. Both modes must stay bit-identical
to `FixedStepTicks`, so this needs a golden row each. Pending merge.

## 9. The exact engine on the queue (groups 3 + 7, consumption site)

`dsed/scheduler.hpp`, `PEDSimulator` only (the BDF2 and auto schedulers
are untouched and refuse ticks: `simulate()` routes a run with controllers
to the event-driven integrator, an explicit `integrator='bdf2'` is
refused by name). Functions touched: `simulate()` (queue start, ticks at
t = 0, the loop-top tick block, the two landing sites), `next_gate_edge_`
(now reads the queue), `next_time_event_` (the probe horizon also stops
at the next tick), `next_switch_edge_` (deleted, its poll inlined);
added `start_time_events_`, `tick_due_`, `run_due_ticks_`, `set_ticks`,
`tick_counts`.

- **One queue per run**: tick sources (ids 0..n-1 = the plan's j), the
  gate oracle (`GateEdge`), the source corners (`SourceBreakpoint`).
- **The oracles are asked at every step, as before.** A cached edge is
  NOT bit-identical: the three-phase SPWM oracle answers the same edge
  one ulp apart depending on the query point (vsi_dsed moved by 8.7e-19 s
  in times, 2e-12 in states with the lazy refresh). So both sources are
  invalidated before each `next_after`; caching belongs to the unified
  stepper, which may change sequences. The poll of a gate with no oracle
  is unchanged.
- **Ticks land like gate edges**: the step target is
  `min(t_gate, t_end, next tick)`. At a landing that is an edge, a tick or
  both, the ticks of that instant run FIRST (D2), then the existing gate
  event samples the switch function at the right limit (dropped when the
  mask did not change), so a duty written at a period start already
  decides that period's rising edge (tested: duty 0 written at the 5th
  period start keeps the switch OFF from that instant). A tick
  invalidates the gate source and the RK45 FSAL slope (a driven source
  may have jumped). Take tolerance: `max(1e-12, 8 eps |t|)`, the
  engine's `near_time` rule. Ticks run with the full MNA state rebuilt
  through the current mode (`recover_full`); the step time equals the
  scheduled instant to that tolerance.
- **Count rule** as everywhere: instants `t_k < t_end`.
- **A tick and a diode event at one instant** are ordered by path: on the
  exact path a diode root located at the very end of the step fires
  first and the tick runs at the next loop top on the projected state;
  on the RK45 path a root at the step end is not taken and the tick runs
  on the pre-event state. Both are one instant; D2 does not order state
  events against ticks (that is group 11's single selection).
- A native-adapter rejection with controllers raises the named
  `step_observer` refusal; `sensitivities=` with controllers is refused
  (the sensitivity run has no ticks).
- **t_start != 0 with controllers is refused**: the engine runs its gate
  in local time from 0, the controller in `simulate()`'s time.
- Result: the six closed-loop reference circuits were refused on dsed
  for want of ticks. Now cl_boost_pi 1.3e-6, cl_buck_aggressive_pi
  3.2e-5, cl_buck_pi 1.8e-5 agree; ll14 / ll15 / ll16 are refused for a
  Shockley diode (a Newton device the exact engine does not run), which
  their earlier refusal had hidden. Open-loop dsed byte-identical (buck,
  buck at dt, VSI, pulse RC, sync buck, opaque gate).
- Not allocation-free: the exact engine's RK45 step returns vectors per
  step, so the zero-alloc test is not extended to it; the queue calls
  themselves allocate nothing after `start`.

## 10. A declared monitor (`Kind::Monitor`)

The TR-BDF2 branch had added no monitor kind (checked:
`git log origin/feat/core-p2-trbdf2 -- core/include/pulsim/events`), so it
is added here, appended to the enum so no existing value moves:
`Kind::Monitor = 5` (`kAllKinds` 0x3F), last at an instant.

- A monitor source is registered with `lands = false`. Engines call it at
  the first accepted step at or after each instant (`t >= t_k - tol`),
  with that step's (t, x), and once more after the loop for instants
  inside the last step, so the count is the controllers' rule,
  ceil(span / T). It never ends a step, never invalidates the gate,
  never resets the RK45 slope and is not read by any landing or cap
  (those read `next_of(Kind::ControlTick)`), so the trajectory is
  bit-identical to the run without it.
- `TickPlan::monitor` (per rate, empty = none), `TrBdf2Options::
  observer_monitors` (trbdf2 consumption sites: the registration and the
  loop-top tick call), dsed `tick_monitors`. The fixed engine has no
  landings: a monitor ticks there like any source (unchanged).
- Python: `pulsim.control.Monitor(fn, period)` passed as `step_observer`;
  `_control_plan.monitor_flags` tells the engines. Tested bit-identical
  on pwl, trbdf2 and dsed (times and states), with a control that the
  same observer undeclared does move the run.
- A monitor's `t` is the accepted step's time, not its scheduled
  instant (dense output at the instant is not done); on dsed it runs at
  the loop top, so at an event instant it sees the post-event state.
  A step that spans several instants calls each one (`take_due` is
  repeated until none is due; a 1 us monitor under 20 kHz PWM: 2000
  calls in 2 ms on trbdf2 and dsed, runs still bit-identical).
- dsed: ticks force the event-driven (rk45) integrator. For a monitor
  alone on a diode-free circuit that would change the integrator the run
  uses, so it is refused unless `integrator='rk45'` is passed.

## 11. Step hooks on the exact engine (event losses task 3.1)

`PEDSimulator::set_hooks(StepHooks*)`: `start` after the t = 0 settle and
ticks, `accepted(t0, t, x, step_mask)` at both accepted-step sites, with
the pre-event state (`x_pre_event`) when an event ended the step. So
`on_commutation` fires at every change of the integrated mask with the
state before the event, as on pwl and trbdf2 (a zero-time cascade is one
commutation). Tested against the result's own record (`step_masks`):
same count and same instants on a PWM buck with a freewheel diode.

## 12. Modulators: the control graph's PWM gate on `CarrierPwm` (group 6)

- `events/modulator.hpp` holds `LatchPolicy`, `DutyRegister`,
  `CarrierPwm` (moved out of `gate_source.hpp`) and the new `EdgeList`
  (sorted (t, bit, on) edges; a bit holds its last edge at or before t,
  OFF before its first; the last edge of an instant wins). It includes
  only `pwm_phase`, `switch_state` and `sources/pwm_leg.hpp` (the leg
  thresholds, moved verbatim out of `switch_schedule.hpp`), so the
  control graph can include it without pulling in the engines.
- `ChainPwmGate` is a list of `CarrierPwm` legs. Each leg reads a
  `DutyRegister` the chain owns (`BlockChain::duty_register(slot)`,
  `std::deque` for stable addresses, Immediate latch). The chain writes
  every register after each call that can change what a sink sees:
  `compile`, `run_pass` (both exits), `step`, `set_output`,
  `set_channel`, `reset`, `restore_state`. So the tick handler writes
  the register, and the gate reads only the register.
- Why byte-identical: `CarrierPwm::apply` sets the bit to
  `!reached(p, d)`, which is `p.u < d - tol` (the old test) for every
  non-NaN duty; `next_edge` is the same `next_constant_crossing` call on
  the same duty. Two legs on one bit used to OR; that is now refused
  ("driven by two legs"), as `GateSet` refuses it. No caller did it.
- `bind_pi_to_switch` builds its gate through `make_pwm_gate`, so it is
  on `CarrierPwm` with Immediate latching. The throttle task 6.2 names
  (`control.py:1350-1355`) no longer exists: the loop is a control graph
  whose rate group ticks once per period.
- Python: `pulsim.DutyRegister(value, latch='immediate'|'carrier_start',
  period, phase)`, `make_carrier_pwm_fn(period, duty, switch_idx,
  num_switches, low_side=-1, dead_time=0, phase=0)` and
  `make_edge_list_fn(edges, num_switches)`. Both return a
  `SwitchSchedule` with edges, so every engine lands them; the carrier
  one holds its register (shared_ptr) and is `follows_controller`.

## 13. Tick phase in snapshots (resume)

- `SolverSnapshot::tick_origin` / `tick_next_k`: one entry per tick
  source of the run, in registration order. The origin is the grid's
  (`Rate::origin`, or the run's t0); next_k is the index of the next
  instant (`EventQueue::grid_origin`, `next_k`). Filled at the end of a
  pwl and a trbdf2 run (the final-snapshot sites only).
- Pickle version 4 = version 3's 13 fields + these two (15 fields).
  Version 3 (12 or 13 fields) still loads, with no tick phase.
- `simulate(resume_from=)` passes the origins back when the snapshot has
  one per tick source of the resumed run: `observer_origins` (trbdf2)
  and `tick_origins` (pwl Python tick sources). The queue then starts
  each source at its first instant >= t_start on the old grid, with the
  old k. trbdf2 no longer refuses a graph resumed between two of its
  ticks; it still refuses when the snapshot has no usable origins
  (pickled before version 4, built by hand, or a different number of
  tick sources, as after a pwl run whose graphs tick themselves).
- Number change: a plain Python tick source resumed mid-period on pwl
  used to restart its grid at t_start; it now keeps the run's grid.
