## Context

v2.0 Phase 1 removed the per-step heap allocations of the fixed engine
(`core/tests/perf/test_zero_alloc.cpp` locks that in). What is left on
small circuits is work that is done every step and is not arithmetic:
a 3.5 KB zero-fill, a duplicated callback, a table search, snapshot
copies, dense passes over sparse data. The 2026-11 review measured the
buck at 380–460 ns/step with the solve at 13 %.

Constraints: the owner wants this phase bit-identical; the thread-local
solve workspaces, the GIL release, the lazy mask cache and
`-ffp-contract=off` stay as they are.

## Goals / Non-Goals

- Goals: remove the listed per-step costs; keep every state value, the
  switch-state record and the event counts identical; give the step loop
  counters that a test can assert on.
- Non-Goals: fewer events or a different event model
  (`add-unified-event-queue`); a dense path for tiny n
  (`add-exact-propagator-strategy`); any change inside `cache.solve`,
  the factorisation or assembly (`update-linear-time-assembly`).

## Decisions

- **`masks_seen` is hoisted, not shrunk.** The array keeps its 64
  entries; only its construction moves above the loop and the per-step
  reset is `n_masks_seen = 0`, which the code already does at
  `run_transient.hpp:949`. Alternative: a smaller array or a hash set.
  Rejected: both change when a cycle is declared.
- **Gate mask read once per solve time.** `gate_mask = switch_fn(t)` is
  evaluated once; `mask_pre` and each event iteration combine it with
  the current diode mask. A solve at another time (`at_t != t`) reads
  the schedule again, once. Alternative: cache `(t, mask)` inside the
  `std::function` wrapper. Rejected: it hides the rule in the binding
  layer and does not help C++ callers.
  - Consequence: the result is identical whenever `switch_fn` is a
    function of `t` and of state written by `step_observer`. Every
    helper in the package follows that pattern (`switchgear.py`,
    `control.py`, `blockchain.py` update state in the observer and read
    it in `switch_fn`). A callable that mutates on each call already
    sees a data-dependent number of calls (one per event iteration).
- **Memo lives in the loop, validated by a cache generation.** The loop
  holds `{mask, const PwlSegment*, generation}`. `PwlStateSpaceCache`
  gains `generation()` (relaxed atomic), incremented by lazy insert,
  eviction, `refactor_parametric` and `clear`. A memo hit still writes
  the LRU tick through the pointer. Alternative: a "last hit" slot inside
  the cache. Rejected: caches are shared between GIL-released runs, so a
  mutable slot would be a data race (review finding ZA-1). Alternative:
  skip the LRU touch on a memo hit. Rejected: eviction order would change
  under a tight byte budget.
- **Snapshots copy what a rollback restores.** `HistoryEntry` has six
  constant fields and two mutable ones; the snapshot becomes a flat
  `(v_prev, i_prev)` array, as `to_flat()` already produces. Diode bits
  are copied word-wise. With the default `max_dt_halvings = 6` a rollback
  is possible on every step, so the snapshot cannot be dropped by
  default; it is dropped only when no rollback path is enabled.
- **Default schedule: the same mask, in C++.** `simulate()` builds the
  default mask exactly as today (all closed on `pwl`, with the existing
  warning; all open on `trbdf2`) and wraps it with
  `make_constant_switch_fn`. On `trbdf2` a constant schedule carries an
  edge oracle (`next_edge = +inf`) that the lambda did not, which routes
  the stepper around its gate-learning samples. If the goldens show any
  difference there, the binding passes the mask function without the
  oracle; that rule is fixed by task 6.3, not left open.
- **Sparse writes are bitwise equal to the dense add.** Today a source
  pass fills a zeroed vector and adds it to `b_extra`. For the rows a
  source owns the arithmetic is the same `b + (−v)`. For every other row
  the dense add computes `b + (+0.0)`, which is `b` bit for bit unless
  `b` is `−0.0`; `b_extra` starts from a `+0.0` fill and is only ever
  updated with `+=` / `−=`, which cannot produce `−0.0` from `+0.0`. The
  golden byte hashes are the check.
- **Cached companion coefficients use the same expression.** `g_eq(dt,
  p)`, `g_eq_inv(dt, p)` and `1 / g_eq_inv(dt, p)` are evaluated by the
  same model functions and stored per entry with the `dt` they were
  computed for; the history term is then formed by the same multiply and
  add.
- **`res.times` follows `res.states`.** A read-only array view whose
  base is the result object. Alternative: keep the list and add
  `res.t`. Rejected: two names for one thing; the package already wraps
  `.times` in `np.asarray` / `np.array` in 23 places, which keep working
  on an array.
- **Counters are plain members written by the loop** (`Size`, no
  atomics), copied into the result at the end of the run and exposed as
  `res.counters`.

## Risks / Trade-offs

- A stale memo pointer would be a use-after-free → the generation check
  is on every memo use; a test evicts between two steps with a tight
  byte budget.
- `res.times` as an array breaks list-only idioms (`if res.times:`,
  `res.times + [...]`, `.index`) → migration note; the package itself is
  swept in task 8.2.
- The `-0.0` argument for sparse writes rests on how `b_extra` is built
  today → the byte-hash goldens catch a violation; if one appears, the
  sparse pass is dropped for that source kind.
- The measured gain is an estimate → the baseline and the acceptance
  measurement use the same probe on the same machine.

## Migration Plan

1. Counters first (no behaviour change), baseline recorded.
2. One item per commit, each followed by the golden comparison.
3. `res.times`: CHANGELOG entry and `docs/migration-guide.md` line
   (`list(res.times)` restores the old type).
4. Rollback: every item is independent; reverting one commit restores
   the previous loop.

## Open Questions

- Is "`switch_fn` is a function of `t` within a step; state belongs in
  `step_observer`" acceptable as the documented contract?
- `res.times` as a read-only array: accept the type change now, or keep
  the list for one release with a deprecation warning?
- Should `res.counters` be the common counter surface for every Phase 1
  change, or does `restore-reference-benchmark-harness` define one?
