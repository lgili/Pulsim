## Why

The fixed engine cannot place an event inside a step. Three
measurements (core review 2026-11, `raw/numerics.md` item 3; probes in
`/Users/lgili/pulsim_wt/core_review_2026-11/numerics/`, re-run on this
branch):

- **The diode watcher is dead code for turn-off.**
  `run_transient.hpp:1264-1292` interpolates the diode signal between
  the state before the step and the state after the event iteration.
  For a turn-off the first is ON (about 1e-5 V) and the second is
  already OFF (volts), so the estimate lands on `t_prev`.
  `p16_events.py`: every `commutation_events.t_estimated` is an integer
  grid index, while the current's zero crossing is 0.32–0.33 dt later.
- **`enable_substep_state_correction` does nothing.** The split is
  skipped when `dt1 < 1 % dt` (`run_transient.hpp:1308`), which is
  always. `p15_substep.py`: the DCM buck reads −0.023 % at T/100,
  −0.497 % at T/101 and −0.022 % at T/400, bit-identical with the
  option on and off.
- **A gate edge off the grid is first order.** `p1_edge.py`: an edge at
  +0.3 dt or +0.7 dt is placed ∓0.200 dt away, order 1.00. After
  `add-discontinuity-restart-rule` an edge on the grid is exact, but
  one inside a step is snapped to a step boundary, which is a duty
  error. Prototype `p19_split_proto.py`, buck CCM with D = 0.4567: the
  output is 8.0e-2 V off at T/50 … T/200 and 1.9e-2 V at T/400 and
  T/800 with snapping, and 7.3e-6 … 2.9e-8 V (order 2.0) with the
  split. Closed-loop PWM never has its edges on the grid.

## What Changes

- **Locate.** A diode crossing is located by linear interpolation of
  `SwitchedDiode::crossing_signal` between the state at `t_prev` and
  the FIRST solve of the event iteration (taken under the pre-event
  bits), not the post-flip one. A gate edge is located by
  `next_edge_after` when the schedule offers it, and by bisection of an
  opaque `switch_fn` on the steps where the mask at the two ends
  differs. A source jump inside a step is located from the jump table.
- **Split (PSIM-style).** The step `(t_prev, t]` with an event at `t_e`
  becomes a trapezoidal sub-step of `dt1 = t_e − t_prev` under the
  pre-event mask and then the restart rule from `t_e`: two
  backward-Euler half-steps of `dt2/2`, `dt2 = t − t_e`, each with its
  own event iteration. The run lands on the grid point `t`; the output
  grid and the integer step counter are unchanged.
- **Several events in one step** are taken in time order, up to a fixed
  count; an event closer than 1 % of dt to a step boundary is snapped
  to it and handled by the restart rule.
- `commutation_events.t_estimated` becomes the located time. A new
  `SimulationResult::n_split_steps` counts the splits.
- **BREAKING.** `enable_substep_state_correction` is the switch for
  this behaviour and its default becomes true together with the
  restart rule's. Fixed-step results with an off-grid edge or a diode
  crossing change; the sub-step solves use off-nominal `(mask, dt)`
  factors through `cache.solve_at`.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `solver/run_transient.hpp` (watcher, split,
  multi-event loop, records), `solver/options.hpp`,
  `solver/result.hpp`, `pwl/cache.hpp` (event-entry accounting only),
  `python/bindings.cpp` (edge oracle reaches the fixed engine),
  `python/pulsim/{__init__,_diode_state,steady_state}.py`
  (`t_estimated` is no longer a grid index), stubs, `AI_GUIDE.md`,
  docs, `CHANGELOG.md`.
- Depends on: `add-discontinuity-restart-rule` (the post-event restart
  and the start-sampled mask), `restore-reference-benchmark-harness`.
  The off-nominal factors are served by the event cache that
  `update-trbdf2-factor-reuse` budgets by bytes.
- Golden baselines: every fixed-step baseline with a diode crossing or
  an off-grid edge moves when the default flips; baselines whose edges
  are all on the grid and that have no diode crossing are unchanged
  with respect to the restart rule's. Users are told through a
  CHANGELOG **BREAKING** entry, and guide trap 10 ("gate edges land on
  the dt grid") becomes obsolete.
- Out of scope: the unified event queue (`add-unified-event-queue`),
  control-tick events, comparator-driven edges, a variable output grid.
