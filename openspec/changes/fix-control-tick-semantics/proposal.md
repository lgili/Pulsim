## Why

When a discrete controller runs — a `ClosedLoop` from
`bind_pi_to_switch`, or a `step_observer` with `controller_period` —
depends on the engine, and three of the combinations are wrong. Measured
on `feat/diode-forward-drop`, probes in
`/Users/lgili/pulsim_wt/core_review_2026-11/arch/`:

- **Two closed loops without `dt` crash.**
  `simulate(b, 10e-3, closed_loops=[a, b])` raises
  `TypeError: unsupported operand type(s) for |: 'SwitchStateMask' and
  'SwitchStateMask'` at `python/pulsim/__init__.py:2281`: the
  variable-step branch composes the loops' masks with an operator that
  does not exist (`probe6.py`). The fixed-step branch composes the same
  thing with `make_combined_switch_fn` (`__init__.py:2439`) and runs. By
  reading, one loop plus a user `switch_fn` reaches the same line.
- **Multi-rate control is wrong on `trbdf2`.** A 2 kHz PI loop plus a
  passive 20 kHz monitor (`step_observer`, `controller_period = 50 µs`)
  over 20 ms: the PI updates **400 times instead of 40** and V_out moves
  from 13.033 V to **12.091 V** (`probe7.py`). Every observer's
  unthrottled `.tick` is fired at `min(periods)`
  (`__init__.py:2224-2266`), while the loop still integrates with
  `dt = T_PWM` (`python/pulsim/control.py:1329-1345`).
- **`controller_period` is ignored on `pwl`.** The keyword is read only
  on the variable-step branch. An observer with
  `controller_period = 100 µs` on a 2 ms run is called **2001 times on
  `pwl`** (`dt = 1 µs`) and 20 times on `trbdf2` (`probe2.py`); the
  20 kHz monitor above is called 20 001 times instead of 400.
- **The fixed engine loses ticks.** A `ClosedLoop` there is throttled by
  `t − last_tick < T` (`control.py:1350-1356`), which drifts: a 20 kHz
  loop gets 198 ticks in 10 ms instead of 200 (`probe6.py`).

The common cause: the composition of loops, observers and gate functions
is written twice in `simulate()` (`__init__.py:2215-2283` and
`2420-2455`) and the two copies have diverged.

## What Changes

- **One tick rule for every engine.** Each tick source (a `ClosedLoop`,
  or `step_observer` with `controller_period`) has its own period `T_j`
  and fires at `t_start + k·T_j` for every integer `k ≥ 0` with
  `t_start + k·T_j < t_end`: exactly `ceil((t_end − t_start)/T_j)`
  times, in registration order when several coincide.
- **Variable-step engine: per-source periods.** `run_transient_trbdf2`
  takes a list of periods and lands a step boundary on every tick of
  every source (integer counters, as today for one source); only the
  sources due at that instant fire.
- **Fixed-step engine: scheduled, not throttled.** Tick `k` of source
  `j` fires at the first grid point at or after `t_start + k·T_j`,
  counted with an integer `k`, so no tick is lost. `controller_period`
  is honoured for `step_observer`. `T_j < dt` is refused; `T_j` not a
  multiple of `dt` runs with a warning stating the jitter (< `dt`).
- **One composition function** used above the engine dispatch by both
  paths: gate masks are merged with `make_combined_switch_fn`, tick
  sources and every-step observers are collected once. The `|` line is
  deleted.
- `result.control_tick_counts` reports the ticks fired per source.
- **BREAKING (numbers):**
  - `pwl` closed-loop runs get the ticks they lost (198 → 200 in the
    case above) and the first tick at `t_start`.
  - `pwl` with `step_observer` + `controller_period`: the observer is
    called at the period, not every step (2001 → 20 calls).
  - `trbdf2` with sources at different rates: each runs at its own rate
    (400 → 40 PI updates; V_out back to the single-loop value).
  - Single-source `trbdf2` runs are unchanged.

## Impact

- Affected specs: `transient-timestep`, `python-bindings`.
- Affected code: `python/pulsim/__init__.py` (both composition blocks
  replaced by one helper, `_trbdf2_blockers` wording),
  `python/pulsim/control.py` (the throttle stays for callers that drive
  a loop by hand; `simulate()` uses `.tick`),
  `core/include/pulsim/solver/trbdf2_transient.hpp`
  (`TrBdf2Options::observer_period` → list, tick loop at 953-956 and
  1065-1078 and 1115-1118, `TrBdf2Stats::n_ctrl_ticks` per source, its
  comment), `python/bindings.cpp` (trbdf2 binding), stubs, tests, docs.
- Dependencies: none. `add-engine-conformance-matrix` carries the three
  cells fixed here as strict expected failures; whichever lands second
  removes the markers. Rates with offsets, a sorted schedule and
  in-kernel blocks belong to `add-compiled-control-graph`; one queue for
  ticks, edges and breakpoints belongs to `add-unified-event-queue`.
- Users affected: closed-loop scripts on `pwl` (small numerical shift,
  correct tick count), scripts that passed `controller_period` on `pwl`
  expecting an every-step observer (drop the keyword to keep that), and
  multi-rate scripts on the variable-step engine (results become those
  of the rates the user declared). PulsimGUI's closed-loop builders pass
  through `simulate()` and inherit the change.
