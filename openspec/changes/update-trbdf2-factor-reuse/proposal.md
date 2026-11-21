## Why

The variable-step engine (`trbdf2`) is bound by factorisation, not by
the solve: it refactorises on nearly every step, and on the VSI it
rebuilds whole cache entries thousands of times. Core review 2026-11,
`raw/perf.md`; the counters were re-run on this branch on 2026-11-12
with the counter part of `perf/p5_events.py` and came out identical
(they are deterministic):

| Case | Accepted steps | `event_refactors` | `event_hits` | `event_builds` | Cost |
|---|---|---|---|---|---|
| Buck, 10 ms, n = 5 | 20,413 | 25,115 | 20,885 | 5 | 1.6–1.9 µs per accepted step |
| Three-phase VSI, 20 ms, n = 12 | 16,396 | 16,337 | 16,548 | **3,198** | 5.4 µs per accepted step |
| RLC ladder, 2 ms, n = 1502 | 2,143 | 1,442 | 2,847 | 1 | 156–166 µs per accepted step |

Time split (`perf/samp_vsi_tr.txt`, `perf/samp_ladtr.txt`): VSI — 19 %
numeric factorise, 7 % analyse, 26 % malloc/free + memset, 15 % sparse
copy, 6 % triangular solve. RLC n = 1502 — 41 % factorise, 32 % solve,
15–24 % sparse copy.

Three causes:

- **The step size is continuous.** The controller sets `h = h * fac`
  (`solver/trbdf2_transient.hpp:1806`, reject path `:1370`), so almost
  every step has a new `γh` and `solve_at` takes its refactor branch
  (`pwl/cache.hpp:1136-1157`). About one hit per accepted step is what
  the two-stage method gives by itself: stage 2 reuses the factor of
  stage 1.
- **The event table holds 8 entries** (`kMaxEventEntries`,
  `pwl/cache.hpp:1172`). The VSI visits 87 masks (measured on the fixed
  engine), so entries are evicted and rebuilt — assembly, symbolic
  analysis and factorisation — 3,198 times.
- **A refactor allocates.** `combine_event_J_` builds a new sparse
  matrix for every `dt` (`pwl/cache.hpp:2030`) and `factorize` allocates
  six n-sized work vectors per call
  (`sparse/pulsim_lu_solver.hpp:237-245`).

## What Changes

- **Part 1 — byte-budgeted event table (bit-identical).** The fixed
  8-entry limit is replaced by the byte budget the lazy segment table
  already has (same default, 1 GiB, applied to each table). New counter
  `event_evictions`. `kMaxEventEntries` is removed.
- **Part 2 — allocation-free refactor (bit-identical).** Each event
  entry stores `G` and `C` scattered on the union pattern of `J`; a new
  `dt` writes `J[k] = G[k] + (1/dt)·C[k]` in place. The in-house LU
  keeps its factorisation work vectors and its L/U capacity between
  calls.
- **Part 3 — step size on a geometric grid. BREAKING (step
  sequence).** A step size chosen by the controller is rounded down to
  the grid `{c_j · 2^e}`, `c_j = 2^(j/m)`, `m = 4` per octave (ratio
  about 1.19). Steps shortened to land on a gate edge, a controller
  tick, a diode crossing or `t_end` are exempt, and so is `h_max`. Each
  event entry keeps up to four factors keyed by `dt`, plus one scratch
  factor for off-grid solves, so a step size that was used before on
  that mask is a hit.
  - Sample times, `n_accept` and the low-order digits of the states
    change. Every accepted step still passes the same LTE test, and a
    rounded step is never larger than the controller's proposal.
  - `step_grid_per_octave = 0` restores the continuous controller and
    with it the previous step sequence bit for bit.
- **Counters:** `event_evictions`, `event_analyses`,
  `event_slot_evictions` in `CacheMetrics`; `n_offgrid_steps` in the
  TR-BDF2 stats.

Not in this change: the error estimator, the stiff filter and the step
bounds (`update-trbdf2-error-control`); the post-event restart
(`update-trbdf2-post-event-restart`); one symbolic analysis shared by
all masks (`update-linear-time-assembly`).

## Impact

- Affected specs: `transient-timestep`, `linear-solver`.
- Affected code: `pwl/cache.hpp` (`solve_at`, `EventEntry`,
  `combine_event_J_`, metrics), `sparse/pulsim_lu_solver.hpp`
  (`factorize` workspace), `solver/trbdf2_transient.hpp` (step
  selection, stats, option), `python/bindings.cpp`,
  `python/pulsim/__init__.py`, `core/tests/perf/test_zero_alloc.cpp`,
  tests that pin `n_accept`, sample counts or `num_event_entries() ≤ 8`.
- Bit-identity: parts 1 and 2 leave every `trbdf2` result identical
  (a factor is a pure function of mask and `dt`). Part 3 does not; with
  the grid disabled the result is identical to parts 1 + 2.
- Depends on `restore-reference-benchmark-harness`: goldens for parts 1
  and 2, the analytic reference set for the accuracy check of part 3.
- Expected speed-up, **estimates from the time split, not prototyped**:
  2.5–3× on small switched converters, 1.6–2× at n ≥ 1500.
- The review asked for the accuracy lens to sign off the quantisation;
  that sign-off is a task gate.
