# Decisions taken while applying update-fixed-step-loop-bookkeeping

Rule: the open question's proposed option, else the most conservative
one. Format: question → option → why → how to reverse.

## Open questions of design.md

1. **"`switch_fn` is a function of `t` within a step; state belongs in
   `step_observer`" as the documented contract?** → Yes (proposed). →
   Every helper in the package already follows it; the closed-loop and
   retry probes are bit-identical. → Reverse: replace `gate_at(...)` by
   `switch_fn(...)` in `run_transient.hpp`.
2. **`res.times` as an array now, or a list with a deprecation for one
   release?** → Array now (proposed), marked BREAKING in the CHANGELOG
   with a migration-guide line. → The package already wrapped it in
   `np.asarray` everywhere; the sweep (and the full suite) found one
   in-place write through such an `asarray`
   (`_result_views.TimeWindow`, now a copy; 199 tests failed without
   it). → Reverse:
   `.def_readonly("times", &SimulationResult::times)` in `bindings.cpp`.
3. **`res.counters` as the common counter surface?** →
   `restore-reference-benchmark-harness` does not define one in this
   branch; `res.counters` is a plain dict of this change's three
   counters, which later changes may extend. → Reverse: drop the
   property.

## Choices the design left open

- **Baseline (task 1.1).** `restore-reference-benchmark-harness` is not
  on this branch; the baseline is the change's own probe
  (`probes/bookkeeping_baseline.py`, 69 runs: restart rule off / on /
  split, `trbdf2`, `dsed`, no `switch_fn`, `store_every`, closed loop,
  dt-retry, static path) saved from 662c137c, compared by value and by
  bytes (signed zeros).
- **Gate memo.** Up to 16 `(t, mask)` pairs per outer step, cleared
  after the observer; beyond 16 distinct times the last slot is reused
  (more calls, same masks).
- **`segment_searches` bound (task 4.3).** The design's "≤ mask changes
  + 1" counted recorded masks; a step whose diode flips solves under
  two masks, so the bound is "mask changes between consecutive solves
  + 1". Pinned exactly: 81 for 2000 buck steps (40 with the rule).
- **Memo scope.** Nominal-dt solves of the main cache (linear and
  Newton, dynamic and static paths). The restart cache and the
  off-nominal `solve_at` LRU keep their own lookups.
- **Diode bits** are already packed (`std::vector<bool>`); unchanged.
- **Rollback snapshots** are skipped when `dt_retry_can_help` is false
  and neither `discontinuity_restart` nor
  `enable_substep_state_correction` is set (every reader of the
  snapshots is one of those three paths).
- **Cached companion coefficients (task 7.2): not done.** Two divisions
  per dynamic branch per step; the A/B gain did not need it, and the
  four history kinds (capacitor, inductor, BE stage, coupling) would
  each need a dt-keyed cache that a retry or split invalidates. →
  Reverse/do: store `2C/dt`, `1/g_eq_inv` per entry keyed on `dt` in
  `HistoryState`.
- **Samples through a cursor (task 8.1): not done.** `times` stays a
  `std::vector` with `push_back`; the view is zero-copy over it.
- **Default schedule on `trbdf2` (task 6.3).** The constant schedule's
  edge oracle (`+inf`) gave byte-identical results on the diode-only
  case `trbdf2` accepts without `switch_fn` (controlled switches are
  refused there) and on the probe's RLC; no fallback needed.

## Measurements (engine, `probes/perf_ab.py`, interleaved, best of 5)

| case | before (662c137c) | after | B/A | target |
|---|---|---|---|---|
| buck, `make_pwm_switch_fn`, ns/step | 173.1 | 131.3 | 0.76 | ≤ 0.75 — **missed by 0.01** |
| buck, restart rule | 209.8 | 140.4 | 0.67 | |
| buck, split | 195.9 | 139.8 | 0.71 | |
| VSI | 358.2 | 260.7 | 0.73 | ≤ 0.75 ✓ |
| RLC n=5, no `switch_fn` | 350.5 | 125.3 | 0.36 | within 10 % of constant (127.8) ✓ |
| RLC n=5, constant schedule | 178.7 | 127.8 | 0.72 | |
| RLC n=1502 | 36 015 | 34 082 | 0.95 | not slower ✓ |
| `res.times`, 100 k samples, ms | 2.85 | 0.01 | | ≤ 0.1 ms ✓ |

The buck misses the 0.75 target by 0.01 (reported, not waived); task
7.2 is the remaining item aimed at it.

Counters, buck 2000 steps, Python `switch_fn`: Python calls 4041 →
2001 (restart rule 4160 → 2001); segment lookups one per solve
(≥ 2000) → 81; rollback snapshots 2000 (0 with `max_dt_halvings=0`).
