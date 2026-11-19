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

(Superseded for pwl / trbdf2 by 19, slice 3.)

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

(Superseded for pwl / trbdf2 by 19, slice 3.)

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

(Superseded for pwl / trbdf2 by 19, slice 3.)

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

## 19. In-loop feeding (0b.6) and event semantics (slice 3, 2026-11-20)

- **Feeding mode per engine** (`Recording.feeding`, `res.recorder_feed`):
  pwl and trbdf2 through the kernel `Simulation` object: **in-loop**
  (`record/loop_feed.hpp`, `LoopFeed` on `StepHooks::step` / `on_event`;
  `Simulation::set_hooks`; Python `_recorders.attach_loop_feed`, called by
  `_run_on_simulation` and the trbdf2 object route). Stored-array pass
  (the fallback, unchanged numbers): the exact engine (its scheduler's
  hook carries `accepted` only, left limits; the stored right limits are
  exact, so the pass is the better feed there), the routes that bypass
  the object (`_kernel_run`, sweep recipes, `control_tick='state'` queue
  groups, the static and legacy loops — the hook does not fire), and any
  reading that is not a function of the state vector alone. In-loop
  readings: `v(n)`, `v(a,b)`, `i()` of a state column (inductor, voltage
  sources, branch-form switch / diode) or a resistor, `p()` of those
  (product of two linear forms) — the same columns / laws `res.v` /
  `res.i` read from a stored sample.
- **Agreement, same run, both feeds** (`test_in_loop_feeding_equals_the_
  stored_array_pass`): pwl bit-identical on RC, rectifier, bridge (kept
  rows, windows, periods). trbdf2 bit-identical on RC and bridge; the
  rectifier differs by 5.5e-10 relative in the window statistics, a
  legitimate difference: trbdf2 STORES its settled post-event probe as an
  extra sample (4 ns after each zero crossing: 1007 stored vs 1004
  accepted steps), while the in-loop record starts the next line from the
  same settled state as an event row.
- **store_every lifted** when the feed is in-loop: the recorders see every
  accepted step, so store_every = 50 gives bit-identical windows / periods
  / event rows to the full run (pwl rectifier; C++ `LoopFeed` test with
  store_every = 10, statistics-only keeps capacity 0, event rows reserved,
  no per-step allocation). When the run cannot be fed in-loop,
  `store_every > 1` is refused AFTER the run (the route is decided inside
  simulate()), naming why. trbdf2 refuses store_every itself.
- **Event rows** (`event_t`, `event_t_right`, `event_resolution`):
  `exact` (the exact engine's stored right limit), `settled` (trbdf2's
  `on_event`: the settled post-event state at the instant; the next line
  starts from it), `within_dt` (a bracket: left limit at `event_t`, the end
  of the first step under the new mode at `event_t_right`; the line is not
  moved, no right limit is fabricated) — pwl always (its `on_event`);
  trbdf2 where its settle did not report (a gate edge with
  refresh_after_event off, a landing without a net flip, an entry into a
  mode with dependent states: the step was solved under a mask other than
  the previous step's, with no settled report; bracket [t0, t] of that
  step). A restart from a state other than the committed one is a
  `settled` row (`n_jumps`; none seen on the validation circuits).
- **Bridge truth table** (mpmath, 11 commutations; `test_event_rows_
  against_the_bridge_truth`): exact 11/11 rows `exact`, |dt| <= 1.1e-11 s,
  v_C both limits <= 7.4e-8 V; pwl dt = 1e-6: 787 rows (diode chatter),
  every truth instant inside a bracket, v_C limits <= 7.6e-3 V; trbdf2
  rtol 1e-7 (after the review fixes): 17 rows, 3 `settled` + 14
  `within_dt`; every truth instant has a row whose left limit is within
  2.6e-8 s of it (v_C left <= 1.6e-3 V); the `within_dt` rights are the
  end of the first step under the new mode (v_C up to 0.25 V off where that
  step is 40 us long).
- **Pure observer**: the feed reads its arguments only; `Simulation`
  passes the hooks pointer, the loops' hook tests are the existing
  run-invariant ones. Verified by the step-sequence compare (see the
  hand-off) and the goldens with no recorder.

## 20. Device readings in the loop (item 1, slice 4, 2026-11-19)

- **Definition (orchestrator):** a device's `i()` is read under the mode the
  engine COMMITTED, at the committed state, in the step hook after commit
  (`StepHooks::step`'s end mask is the stamp the step was solved under).
  A capacitor's `i()` is the KCL-CONSISTENT current: the companion's
  committed branch current (`HistoryState` entry `i_prev` after the
  commit: trapezoidal `2C/h·Δv − i_n`, the restart's `C/h_e·Δv`, the
  TR-BDF2 stage-2 derivative) — so Σ i = 0 at every node of a recorded
  sample and the trapezoid of i_C over a trapezoidal step is C·Δv. Not
  C·dv/dt from samples (the stored pass still refuses it).
- **How a listener reads it:** `StepHooks::start(x0, mask0, RunView)`; the
  `RunView` (`history`, `stateful`, `probe_dt`) is set once per run by the
  two loops (the only loop hunks: `run_transient.hpp` and
  `trbdf2_transient.hpp`, the `hooks->start` line); `on_start` listeners
  seed the start row from the histories at t0.
- **Laws (record/loop_feed.hpp `DeviceCurrent`, resolved from the pool's
  kind by `LoopFeed::set_circuit`):** PWL switch / diode
  `pwl::PwlBranchLaw` (the loss tracker's `v_of_` / `i_of_`, moved into
  one header and used by both); capacitor (above); Shockley / smooth
  diode, MOSFET, IGBT level 1 `pwl::static_reading` (one branch, the law
  both device paths stamp); saturable inductor, nonlinear capacitor,
  Lauritzen diode `for_each_stateful_reading` summed per branch (IGBT
  tail included); current sources (constant, sine / pulse / table at t);
  inductor, voltage sources, VCVS, branch-form switch / diode: the state
  column (as before). `p()` = v(from, to)·i, or Σ v_pin·i_pin for a
  Newton law (MOSFET / IGBT: drain-source and gate pins).
- **Event right sides (TR-BDF2):** the right side is a settled probe; the
  histories at the report are the left side's, so a capacitor's right
  current is the probe's own companion current
  `g_eq(probe_dt)(v_r − v_prev) − i_prev` (what `refresh_derivatives`
  stores). A report at a step's end that arrives before the step hook
  (the slots may be replaced by the probe's before the hook) keeps the
  history readings taken at the report for the sample. Limitations: a
  stateful device's right side uses the left committed rate (Coss's
  current is its rate: its right value is the left one); at a diode
  landing the right side is the h_zero cascade probe's companion current,
  which the engine does not commit (no derivative refresh at diode
  events): the next step starts from the landing's own committed current,
  so the first piece after such a row is not exactly C·Δv and the right
  value carries ~2C·eps·|v|/h_zero of noise.
- **Refused by name (nothing else):** in the loop, kinds with no current
  law here — VCCS, CCCS, CCVS, gyrator, ideal transformer, PMSM / IM,
  external devices (`set_circuit` names the reading; the run falls back to
  the stored-array pass, which reads what `res.i` reads); `p()` of a
  current source (sign convention "out of n_pos": write it). On the
  stored-array pass (exact engine, routes that bypass the Simulation
  object): `i()` / `p()` of capacitor, nonlinear capacitor, MOSFET, IGBT,
  nonlinear / Lauritzen diode, saturable inductor, VCVS — before the run
  on `engine='exact'`, after it otherwise (the route is decided inside
  simulate), naming why the run was not fed in the loop.
- **Measured** (`test_recorder_device_readings.py`): KCL at every node of
  every sample, RC / buck / capacitor-filtered rectifier, pwl and trbdf2:
  ≤ 5.6e-11 of the node's current scale except where the solver's own
  floor C·eps·|v|/h dominates (femtosecond trbdf2 steps near events:
  8e-7 A at h = 1e-13 s); event right sides likewise. ∫ i_C dt vs C·Δv
  over windows that start and end mid-step: pwl ≤ 2.3e-5 relative (RC
  1e-9: the residue is the two partial steps), trbdf2 ≤ 4.5e-4 (the BDF2
  stage). Energy balance from recorder outputs only (buck, window mid-step,
  stored energy from `keep='times'` rows at the window ends): pwl
  1.4e-5, trbdf2 3.8e-7, exact (stored-array pass, dt = 1e-6 samples)
  1.1e-5 relative; Tellegen (Σ ∫p = 0 over the six devices) ≤ 1e-9.
  Newton kinds against a series resistor (KCL at the terminal) on both
  engines; VCVS and a sine current source.

## 21. Recorder state in the snapshot (0b.7), cost, leftovers (slice 4)

- **Pickle field 18** (version 5, 19 fields): `(recorder_spec, recorder_state)`
  or None; `SolverSnapshot.recorder_state` / `recorder_spec`. Every older
  layout still loads (2.0, v3, the v4 variants, 17- and 18-field v5). The
  state is `LoopFeed::save()`: the last sample's state, mask, a settled
  event at it, the counters, and each `Recorder::save()` — the line's
  start (a partial window ends there), kept rows, event rows, every
  window's and period's statistics. So a resumed run's `recorded` IS the
  uninterrupted run's (not only from the seam on). Saved before
  `finish()` (no `keep='times'` flush is counted twice). The spec
  signature must match on resume (refused by name otherwise).
- **Resume without state** (an older pickle, or a run without in-loop
  recorders): the recorders start at the resume instant and
  `res.recorder_feed["resume"]` says so; only complete periods are
  reported; a declared window that began before the instant is refused
  ("not covered"), never reported complete. A snapshot WITH state on a
  run that cannot be fed in the loop is refused. A run that ends with an
  event report pending writes no state and says so.
- **Windows past t_end** are allowed now (`t0 < t_end`): reported open
  (NaN statistics), continued by a resume.
- Measured: pwl buck split at 0.43 ms (mid-window, mid-period), through
  the pickle: kept rows, event rows, windows, periods bit-identical to
  the uninterrupted run; trbdf2: the resumed controller takes its own
  steps (2 rows fewer), statistics within 6.0e-6 relative.
- **Cost (instructions per accepted step, in-process, min of 6; buck,
  one statistics-only recorder, period 1e-4):**

  | probes | pwl before | pwl now | trbdf2 before | trbdf2 now |
  |---|---|---|---|---|
  | 0 | 3091.9 | 3094.5–3102.8 (≤ +0.35 %) | 10983.6 | 10981.3–10990.3 (≤ +0.06 %) |
  | 1 | 4033.2 (+30 %) | 3954.7–3957.1 (+28 %) | 12393.1 (+13 %) | 12800.7–12840.4 (+17 %) |
  | 4 | — | 4504.7 | — | 14215.3 |
  | 16 | — | 6555.8 | — | 19676.5 |

  Per-step C++ cost of one probe (pwl, run length doubled): ~820
  instructions; a microbenchmark of the feed alone: ~665 per step
  (Recorder::sample ~200 with no window, +~190 with a period: two
  floor() + three window visits with 4-ulp snaps; LoopFeed::on_step ~280:
  std::function dispatch, the jump check, the state copy). Marginal per
  extra probe: ~175 (pwl), ~470 (trbdf2, fewer steps: fixed Python setup
  weighs more). Done: plain `v(n)` probes bypass `ir::Expr`; windows
  wholly outside a piece are skipped before the snaps; the jump check
  compares bitwise first. The +5 % / +3 % targets are NOT met: they need
  a current-window fast path (cached window index and edges; snaps only
  near an edge) and a leaner feed step — next slice.
- **Exact engine in the loop (item 4): requested, not wired.** The
  scheduler's hook calls `accepted` only (left limits;
  `dsed/scheduler.hpp` `hook_accepted_`); no per-segment callback carries
  both limits. Request R5 below; the stored-array pass stays.
- **Leftovers done:** a Simulation's feed is detached after every run
  (`clear_loop_feed`, in a `finally`: a run that threw cannot feed a later
  run on that object); every column a reading reads is bounds-checked at
  the run's start (the per-step reads stay unchecked). NOT done:
  `stats.time_average` delegation, `live_stream` over `RingSink`, the
  result reference page.

## 22. Cost of a recorder fed in the loop (0b.10, slice 5, 2026-11-19)

- **Method.** Instructions retired (`proc_pid_rusage`), in-process around
  `simulate()`, min of 4-5; buck of the golden set (pwl dt = 1e-7, 20000
  steps; trbdf2 3299 stored steps, 2471 hook steps, 829 event rows). The
  per-step and fixed parts were separated by running 1x / 2x / 4x the
  length; the C++ feed was isolated by RECORDING the hook calls of a real
  run and replaying them into a `LoopFeed` in a standalone binary
  (instructions per hook step), then ablating pieces of it. Tools (not
  committed): `.scratch/instr2.py`, `.scratch/bench/replay.cpp`.
- **Where the instructions went, before** (pwl, 1 probe, per step):
  loop side + hook dispatch ~170 (an ablation whose feed returns at once:
  the `device_state` / `accepted` / mask bookkeeping in the loop and the
  `std::function` hop — outside `record/`); `LoopFeed::on_step` ~280;
  `Recorder::sample` ~200 (no window) / +185 with a period (two `floor`,
  three window visits with 4-ulp snaps per piece) / +85 keep every;
  and, the largest item with `keep='every'`, ~770 per KEPT ROW in Python:
  `snap.recorder_state = feed.save()` built a Python list of every saved
  double (kept rows, event rows) and converted it back to a vector — on
  every run, not only on resume. On trbdf2 also ~1.2k instructions per
  EVENT ROW in `Recording.__init__` (a Python loop over the resolution
  codes) and ~176 per step of fixed Python setup at this run length.
- **Changes** (all in `record/`, `record_bindings.cpp`, `_recorders.py`;
  no loop-file hunk):
  1. `LoopFeed.save_to(snapshot, spec)` writes the state in C++; the
     resolution labels are one numpy take.
  2. Flat reading tables: a reading that is one state entry (v(n), a state
     column) is `{column, destination}` (`0 + x[c]`, the general sum bit
     for bit, -0 included); linear forms (v(a,b), a resistor's current)
     one CSR table with the general path's operation order; products and
     device readings keep the general path.
  3. `LoopFeed::on_step` fast step: no report pending, no event row at the
     start, the same mode, the start state bit for bit the last sample's
     (one vectorised pass) — exactly what the general path does there,
     without its branches; the general path out of line.
  4. `Recorder::sample` forced inline with its helpers; a per-probe source
     table for plain probes; cold throws out of line; the start copied in
     place (no `vector::operator=`); `piece()` only when there are
     statistics.
  5. Period windows: a fast path for a piece inside the current window's
     interior (64 ulps from its edges, beyond the 4-ulp snap of it and of
     its neighbours' shared edges; only once the windows a piece there
     could request exist): the same operations as `accumulate` on that
     window alone. Oracle test: period windows equal the same windows
     declared (which visit every window on every piece) on a jittered grid
     with samples within ±6 ulps of the edges and events — 0 mismatching
     windows bit for bit over 117 grids with both the old and the new
     header (`.scratch/bench/oracle.cpp`); the committed test uses 1e-12.
- **Identity.** Recorder outputs on 12 cases (buck pwl / trbdf2 with ten
  probes incl. i(D1), i(C), p(C), products, period + declared windows,
  store_every = 10 with decimate + times, vsi, Shockley buck, MOSFET,
  saturable inductor, 16-node RLC) — kept rows, event rows (both limits,
  resolution, t_right), windows, periods, counters AND the snapshot's
  recorder state: bit-identical to 2b61202d; recorder on vs off: the same
  step sequence (times and states hashes) on all 12. Goldens 22/22
  identical (no recorder: the default path untouched).
- **Cost after** (instructions per accepted step; base = no recorder):

  | buck, 1 probe | pwl before | pwl after | trbdf2 before | trbdf2 after |
  |---|---|---|---|---|
  | none (base) | 3095 | 3080 | 10986 | 10989 |
  | keep='every' | 4626 (+49 %) | 3569 (+15.9 %) | 13401 (+22 %) | 11700 (+6.5 %) |
  | keep='none', period 1e-4 | 3950 (+28 %) | 3573 (+16.0 %) | 12802 (+17 %) | 11743 (+6.9 %) |
  | keep='every', 4x length | — | +450 (+14.7 %) | — | +559 (+5.6 %) |

  Each extra plain probe (v(n) / state column), 4x length: pwl 26
  (keep='none') / 49 (keep='every'), trbdf2 50 / 67; in the replayed feed
  alone 36 per probe on pwl (16 probes, a third of them v(a,b)).
- **Targets NOT met** (+5 % pwl, +3 % trbdf2). What remains, pwl, one
  probe, keep='every' (~450 per step at 4x): ~170 outside `record/` (loop
  bookkeeping with a listener + the `std::function` hop: measured by the
  return-at-once ablation; changing it is a loop-file edit); in the feed
  (replay 335 per hook step): dispatch ~45, the jump test + state copy
  ~75 (5 entries; it decides whether a jump row is due and cannot be
  dropped without changing event rows), the recorder ~70-90, the
  readings ~15, the 447 event steps through the general path and the
  branches the rest; Python ~25 per kept row (snapshot copy, arrays).
  trbdf2 (replay 566 per hook step): one event row per three hook steps
  (each: two readings of both sides, the event row, the general path of
  the next step) and ~170 per step of fixed Python setup at 3299 steps
  (recorder construction, reading resolution, result arrays), ~45 at 4x.

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
- **R5 (exact engine, slice 4).** In `dsed/scheduler.hpp`, beside
  `hook_accepted_`: `hooks_->step(t0, x_start, mask_start, t, x_left,
  mask)` per accepted segment and `hooks_->event(t, x_left, mask_left,
  x_right, mask_right)` where the scheduler stores a right limit — one
  call each, so the exact engine can be fed in the loop (memory with
  `store_every`).
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

## Hand-off (slice 3, 2026-11-20)

- Done: 0b.6 in-loop feeding (19), event resolution on every engine (19),
  `near` -> `close_to` in `recorder.hpp`, api-reference feeding table.
- Cost per accepted step (instructions retired, in-process, min of 6):
  no recorder — buck pwl 3096.1 (baseline 3093.7, +0.08 %), buck trbdf2
  10959.9 (baseline 10963.1, -0.03 %); one statistics-only recorder (one
  probe, period 1e-4) — pwl 4031.6 (+935, +30 %), trbdf2 12375.5 (+1416,
  +13 %). Not optimised: the probe is evaluated through `ir::Expr`, every
  piece walks the period windows k0-1..k1, and the jump check compares
  the whole state vector.
- Open (not started in this slice):
  - item 3 / section 6: KCL-consistent `i(C)` / `p(C)` and the nonlinear
    kinds' readings at commit through `device_state`
    (`pwl/device_readings.hpp`: `for_each_static_reading`,
    `for_each_stateful_reading`), then the energy-balance test on three
    circuits; until then those readings are refused / stored-array.
  - 0b.7 snapshot: a new pickle index after the 17 fields with a reader
    branch (`python/bindings.cpp` ~4589-4752, SolverSnapshot pickle;
    field-count tests: test_device_state_snapshot, test_loss_port_seams,
    test_snapshot_controller_block, test_switch_state_record,
    test_tick_phase_resume); needs `Recorder` state export / import
    (partial windows, period index, start row, LoopFeed's last state and
    mask); `resume_from` stays refused.
  - 4.2 `stats.time_average` delegation, 8.3 `live_stream` from Python
    over `RingSink`, 1.4 result reference page.
  - The exact engine stays on the stored-array pass: its scheduler hook
    (`set_hooks`) calls `accepted` only (left limits); feeding it in-loop
    needs its `step` / `event` calls in `dsed/scheduler.hpp` (owned by the
    exact-engine agent).

## Hand-off (slice 4, 2026-11-19)

- Done: item 1 device readings in the loop (20), 0b.7 recorder state in
  the snapshot (21, pickle field 18), first cost steps (21), leftovers
  (feed detached after every run; column bounds at the run's start).
- Loop hunks (each the existing `hooks->start` line, now passing the run
  view): `solver/run_transient.hpp:1632` and `:3504` (static path),
  `solver/trbdf2_transient.hpp:1553`. No per-step code in the loops.
- Fresh-context review (opus, read-only): A committed-mode readings and
  C pure observer — no defect; fixed: restore rejected counters above the
  state length (a long `keep='none'` resume threw; test added), a resumed
  run with no hook firing reported "in-loop" (now requires steps from THIS
  run), the pwl bracket read a source at the report's t (now the step's
  end row), the static path's `start` dropped the view, reserve before
  restore (now after), the constant current source read from the pool.
  Recorded, not fixed: the diode-landing right side (20). Checked after
  the review: a split ON a gate edge (buck, 0.42 / 0.45 ms) gives the
  uninterrupted run's 41 event rows on trbdf2 and pwl, one at the seam
  on trbdf2 (no duplicate).
- Open: cost targets (0b.10: a current-window fast path, a leaner feed
  step); R5 (exact engine in the loop); 4.2 delegation, 8.3 `live_stream`
  over `RingSink`, 1.4 result reference page. Eigen 3.4.0: record / perf / step / thermal targets 29/29 at 98419f48.

