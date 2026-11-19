# add-probes-and-recorders — decisions (track feat/core-p3-probes-recorders)

Base: 656b894f (draft PR #147 head). Loop freeze in force: no edit to
`solver/run_transient.hpp`, `solver/trbdf2_transient.hpp`,
`dsed/scheduler.hpp`.

## 1. Scope of this slice

Kernel probes and recorders with exact windowed statistics, fed from what
exists today (tasks.md section 0b). The result-type unification (sections
1-3, 5-7, 9-10: one constructible `SimulationResult`, `GridKind`,
`RunStats`, event rows on every engine, the `_result_i` ladder replaced by
`ProbeSet`) is not started; nothing in the default path changed.

## 2. One expression parser

`record::Probe::compile` cuts the readings out lexically — `v(`, `i(`,
`p(` at an identifier boundary up to the matching `)` — replaces each by a
placeholder `__a<k>` and compiles the rest with `ir::Expr::compile`. So a
probe has every operator and function of the IR's parameter expressions
and no grammar of its own. Readings take names, not expressions
(`v(abs(x))` is refused). Atom text is normalised (blanks removed) and
de-duplicated. Dotted channels (`M1.q_gate`, `S1.on`) are not in this
slice.

## 3. Signal model and exactness

Between two accepted samples a probe is the straight line from the step's
start (the right limit where the engine stored one, else the previous
sample) to its end (the left limit). Integral h(a+b)/2, square
h(a²+ab+b²)/3 — exact for that line; no squared samples, no trapezoid of
y². This is the arithmetic of `pulsim.time_average` / `time_rms` (the
same `_lerp`, operation for operation), so the two agree to round-off on
the same samples (measured ≤ 3.1e-15 relative on the bridge runs). For a linear probe of the
engines' linear dense output this is the exact integral of the dense
output. A nonlinear probe (`v(out)*i(L1)`) is linear between ITS samples,
like `time_average` of the product trace; the exact product of the two
lines (`_result_views.pl_integral_product`) differs by O(h²). Chosen for
agreement with `time_average` (brief requirement 4); an exact bilinear
mode is a possible follow-up.

## 4. Window clipping and the two limits

- A window [t0, t1] is clipped by interpolating on the step that straddles
  each edge (`stats._lerp`).
- A window that starts exactly at an event instant starts from the right
  limit; one that ends exactly at an event ends on the left limit: the
  piece that ends at t0 has zero overlap and is skipped, the piece that
  starts at t1 likewise.
- min / max see both limits of an event strictly inside the window (the
  left limit ends one piece, the right limit starts the next); `t_min` /
  `t_max` are the first occurrence.
- Declared windows must lie in [t_start, t_end] (refused by name before
  the run) and be covered by the record (refused after it otherwise).
  Period windows [offset + kT, offset + (k+1)T] have edges multiplied,
  not accumulated; only complete periods are reported in Python (C++
  keeps the `complete()` flag).
- Zero-length steps (repeated times) contribute nothing; the later row
  becomes the start of the next step.

## 5. Feeding modes (which one is active)

- **Python (active on every engine): stored-array pass.** After the run,
  `_recorders.record_result` evaluates each reading over the stored
  result (`res.v`, `res.i` — the existing per-kind laws under the
  recorded stamped mask) and calls `record::feed_table`. The exact
  engine's right limits come from `right_limit_index` /
  `right_limit_states` (through `_result_views.with_right_limits`).
- **C++ (available, pwl and trbdf2): in-loop through the existing
  `events::StepHooks::on_accept`** (`record::attach`), state columns
  only. The hook carries neither the step's start state nor a right
  limit: the record starts at the end of the first accepted step unless
  the caller seeds `t_start`; no event rows. The exact scheduler has
  `set_hooks` but no test feeds a recorder from it here.

## 6. Right limits per engine

Only the exact engine stores right limits; pwl and trbdf2 store one row
per accepted step and their record is continuous, so no event rows are
reported there (a switched device's current under the next step's mask
at the same state is not stored). The D4 re-initialisation jumps of
pwl / trbdf2 (`reinit_records`) are not modelled by the recorder, as
`time_average` does not model them.

## 7. Exact engine closed forms

The design fixes linear dense output (Non-Goals: no higher-order dense
output), so the recorder does not compute segment integrals through the
augmented matrix exponential. Recorded as a request (R4). Measured: the
exact engine's bridge average through the recorder is −4.5e-10 V off the
mpmath truth already.

## 8. Snapshot and resume

Recorder state is not in the snapshot (pickle stays version 5 / 17
fields). `simulate(recorders=..., resume_from=...)` is refused by name
(tasks 0b.7).

## 9. store_every

A stored-array pass on a decimated record would alias (the comment on
`switch_state_bits` measures +14 %), so `store_every > 1` with recorders is
refused by name; `Recorder(keep='decimate', every=n)` keeps fewer rows
with statistics from every step.

## 10. Option table

`recorders` is honoured by every engine. `sim::kRunOptionTable` is not
at 656b894f (it is on feat/core-p3-simulation-object); when that branch
integrates, the row `{"recorders", kEvery, Unset::Falsy, "", ""}` is
needed — its test asserts the table's names equal simulate()'s
parameters (tasks 0b.5).

## 11. Allocation

`Recorder::reserve(n_samples, t_last)` sizes kept rows and period
windows (two spare: a piece looks one window past t_last). Measured: a
statistics-only and a decimating recorder fed in-loop on the fixed engine
make the same number of `operator new` calls for 2000 and 4000 steps
(11 / 11), and their buffers keep their capacity over 2·10⁵ samples. An
event row allocates (events are not per step).

## 12. Accuracy bounds used in tests

The RC charge on pwl (dt = 1 µs, τ = 1 ms) is off by −1.33e-3 V in a
window average: the fixed engine's first-order start from the
inconsistent t = 0 state (owner note fixed-engine-first-order-start),
bound V·dt/τ. trbdf2 (rtol 1e-8) −9.7e-7 V, exact −2.2e-7 V, bound
2e-5·V.

## 13. Warning attribution

With `recorders=` the engines' warnings are attributed to
`_recorders.py` (one frame deeper); without it nothing changed.

## 14. Fresh-context review (opus, read-only), fixed

Clipping vs `time_window`, events at t0 / t1 / inside / on row 0 / on the
last row, ties, the period loop and the Python row split were checked
correct. Fixed:
- a window edge within 4 ulps of a sample snaps to it (`3 * 0.1` above a
  run ending at 0.3 dropped the last period in 8527 of 24000 typed
  `t_end = n*T` cases; an edge one ulp before an event started on the
  left limit);
- `offset` must be finite (NaN reached an undefined float-to-size cast);
  more than 1e8 period windows are refused;
- `__a` in probe text is refused (it aliased the placeholders);
- a blank inside a name (`v(a b)`) is refused instead of merging with
  `v(ab)`;
- `right_limit_index` without `right_limit_states` raises a named error.
Not changed: `attach()` with `Keep::Every` regrows if `reserve` under-
counts (the caller sizes it; the in-loop test covers None and Decimate).

## 15. One result type (sections 1, 2 — slice 2)

- `SimulationResult` is default-constructible; the exact engine's adapter
  (`_dsed_dispatch._exact_result`) fills an empty one through
  `_adapter_fill(times, states, switch_states, reinit_records)` and
  `_set_run_info(engine, grid, stats)`. `_PEDSimulationResult` is deleted;
  `isinstance(res, pulsim.SimulationResult)` on every engine.
- Byte compare against 656b894f (all 109 harness circuits x 3 engines,
  every public attribute, `res.v/i` of every observable, the warnings,
  `device_loss_summary`): see the hand-off for the numbers. Type changes
  on the exact engine, values bit-identical: `times` is the read-only
  float64 view (was a list of floats, as pwl's was before
  update-fixed-step-loop-bookkeeping); `switch_states` /
  `switch_states_unsolved` are fresh writable arrays per access (were
  read-only); `reinit_records` are `pulsim.ReinitRecord` (were
  SimpleNamespace; same fields, plus `cause = ""`); the result gains the
  pwl attributes (`final_snapshot` with `valid = False`, `newton`,
  `counters`, `dt_retries = []`, ...), so code that tested
  `getattr(res, "final_snapshot", None) is None` must test `.valid`
  (every in-tree caller already did).
- `n_accept`, `n_reject`, `n_events`, `n_rk45_steps`, `n_bdf2_steps`,
  `n_exact_steps`, `cpu_time_seconds` are read-only properties over
  `res.stats`, present on exact-engine results only: on pwl / trbdf2 they
  raise AttributeError as before, because `design.py` and
  `periodic_steady_state.py` use `getattr(res, "n_accept", None)` to tell
  the engines apart (a trbdf2 `n_accept` there would have changed the
  `engine_steps` count design studies report).
- The exact-engine extras stay instance attributes (`states_reduced`,
  `has_full_states`, `right_limit_*`, `exact`, `event_times`, ...).
- `_result_views.with_right_limits` returned `copy.copy(result)` with
  patched arrays; a kernel result's arrays are read-only properties, so it
  now returns a `_ResultView` (attributes it sets, the rest read from the
  result; the result type's Python methods run against the view).
- shortcut: `res.grid` / `res.stats` are set by the Python finalisation
  of `simulate()` (pwl and trbdf2 tails, the exact adapter), not by the
  C++ drivers (loop freeze). A bare `run_transient` result carries
  `grid = Uniform` and `stats.engine = ""`. pwl reports only
  `n_solves`, `n_factorizations`, `n_newton_iterations` (from
  `res.newton`); `n_steps` is 0 there (`store_every` hides it). trbdf2
  maps its dict (`n_diode_events` -> `n_state_events`); exact: its
  integrator counts, `n_events` -> `n_mode_events`.
- `_trbdf2_stats` stays, the same dict, for one release (2.2);
  `_trbdf2_warn_stats` keeps reading its `stats` argument (in simulate()
  the numbers of `res.stats`): tests drive its confessions with edited
  dicts on a real result.

## 16. Measures and t_eval (sections 4, 0b.8)

- `res.avg / rms / integral / minmax(probe, t0=None, t1=None)` and
  `res.at(probe, t)` are Python methods over the kernel recorder pass
  (`record::feed_table`), not C++ members of `SimulationResult`: the
  readings come from `res.v` / `res.i` (the per-kind laws live in Python
  until section 5's `ProbeSet`). Same arithmetic as `time_average` /
  `time_rms` (tests: rel 1e-12 on three engines).
- `at(t)` at an exact-engine event returns the right limit; pwl / trbdf2
  store no right limit (section 3), so there `at` and the windows see a
  jump as a ramp over the following step, as `time_average` does.
- `Recorder(keep='times', times=[...])` (t_eval, `Times{land=false}`):
  one row per requested time, linear on the step that holds it, the right
  limit at an event instant, `interpolated` flags the rows that are not
  samples. Landing (`land=true`) needs `OutputSample` events in the loop:
  not done (loop freeze, event-queue integration).
- `RingSink`: `Recorder::stream_to(LiveRing*)` pushes every kept row (C++;
  in-loop use through `attach`). `live_stream` from Python on every engine
  waits for in-loop Python feeding (0b.6).

## 17. Device readings in the stored-array pass (item 3)

- `p(dev)` = `v(from, to) * i(dev)` per sample (both limits on the exact
  engine) for kinds whose `i()` is exact on a stored sample and runs from
  the first terminal to the second through the device: resistor,
  inductor, PWL diode, switch, Shockley diode, and the voltage sources
  (absorbed power; a source that delivers reads negative — tested:
  source energy = resistor energy + capacitor energy).
- Refused by name before the run (reading and reason):
  `i()` / `p()` of `capacitor` (derivative of the stored voltage, not the
  companion law: section 6, owner decision pending),
  `nonlinear_capacitor`, `mosfet_level1`, `igbt_level1`,
  `nonlinear_diode`, `lauritzen_diode`, `vcvs`, `saturable_inductor` (no
  per-sample current law on a stored result); `p()` of any other kind
  (multi-terminal, or a source whose sign convention is "out of n_pos").
  `i()` of the remaining kinds is `res.i` (the existing ladder).

## 18. Sections stopped, and why

- 3 (event rows on pwl / trbdf2): the drivers must store the right limit
  at every event: an edit to the loop bodies (freeze); requests R1/R2.
- 5, 10 (`ProbeSet` in C++, delete the `_result_i` ladder): precondition
  0.1 (device-interface section 9: generic pin currents) is not merged on
  this base; the per-kind laws stay in Python until it is.
- 6: owner decision pending (BREAKING capacitor current).
- 7 (in-kernel recording at the push sites): loop freeze; waits for the
  stepper's single hook.
- 8.1 (decimation on the accepted grid, lifting the `store_every`
  refusal) waits for in-loop feeding (0b.6); 8.2 landing: see 16.
- 9 (`Signal` ndarray subclass, `GridWarning`): not started. It changes
  what `res.v` returns on trbdf2 / exact (a subclass propagates through
  arithmetic) and every in-tree `.mean()` over those results would warn;
  it needs its own PR with the suite run under `-W error::GridWarning`.

## Requests to the stepper track

- **R1.** One per-accepted-step call for all engines, after commit:
  `on_step(t_prev, x_start, mask_start, t, x_end, mask_stamped)` where
  `x_start` is the state the step started from (the right limit at
  `t_prev`, after any settle / re-initialisation) — today's
  `StepHooks::accepted` lacks it. With it the recorder needs no seeding
  and pwl / trbdf2 gain right limits (and reinit jumps) for free.
- **R2.** `on_event(t, x_left, mask_left, x_right, mask_right)` after the
  settle at an event instant, on every engine (the exact engine stores
  these rows today; pwl / trbdf2 do not).
- **R3.** At commit, the device interface's pin currents under the
  stamped mode (so `i(dev)` can be a reading in-loop, not only post hoc).
- **R4.** Exact engine: an optional segment-integral callback
  (∫x dt, ∫x xᵀ dt over the LTI segment through
  `dsed/period_integrals.hpp`) so recorders can be exact rather than
  linear-between-samples there.
- All four: no allocation per step, zero cost with no listener (as
  `StepHooks::empty()`).

## Hand-off

Slice 2 (this branch, after 2b18f11a):
- Done: one result type (1.1-1.3, CHANGELOG), `GridKind` / `RunStats`
  (2.1-2.3), result measures `avg / rms / integral / minmax / at` (4.1,
  part of 4.2-4.3), 0b.8 (`keep='times'`, `RingSink`, `at`), `p()`
  readings and pre-run refusals of inexact `i()` / `p()` (17), in-loop
  feeding through the exact scheduler's `set_hooks` tested
  (`test_exact_queue_ticks.cpp`: equals the stored-array pass of the
  left-limit table to 1e-12), `attach()` keep-every with an undercounting
  `reserve` tested, engine warnings of `simulate(recorders=...)` at the
  caller's line (re-emitted; under `-W error` the run's own exception
  wins).
- Byte compare against 656b894f: 109 circuits x 3 engines (327 cells;
  every public attribute, `v/i` of every observable, warning lists,
  `device_loss_summary`): identical except, on the 100 exact-engine
  cells, `times` list -> array (same bytes) and the new pwl attributes
  (`dt_retries`, `heat_injected`, `n_restart_steps` were absent).
  Goldens 22/22 IDENTICAL (syncbuck_dsed: counters absent -> 0).
  Reference: `pytest -m reference` 368 passed, REFERENCE_STATUS.md
  unchanged. pytest 5214 passed / 108 skipped / 6 xfailed. C++: record +
  dsed targets on Eigen 3.4.0 (13 + 126 cases) and locally (`ctest -R
  'record|dsed'` 32/32). Full ctest NOT run (orchestrator rule: the
  integration tree runs it).
- Open: 0b.5 (option-table row at the Simulation-object integration),
  0b.6 (stepper R1-R3), 0b.7 (snapshot index 17, see tasks), 1.4 result
  reference page, 3, 4.2 delegation of `stats.time_average`, 4.3 PR #140
  check, 5-10 (see 18).
