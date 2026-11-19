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

- tasks 0b.5 (option-table row), 0b.6 (feeding on R1-R3, then lift the
  `store_every` refusal), 0b.7 (snapshot index), 0b.8 (`Times`, `RingSink`,
  `at`).
- Sections 1-3, 5-10 of tasks.md untouched.
- The exact scheduler's `set_hooks` path is not exercised with a recorder.
