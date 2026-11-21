## Why

Building the matrix of a switch state is quadratic in the number of
unknowns, every switch state repeats a symbolic analysis that is the
same for all of them, and the large-n triangular solve carries avoidable
passes. Core review 2026-11, `raw/perf.md` (throttled machine; trust the
ratios):

| Cost today | Measurement | Probe |
|---|---|---|
| Segment build (assemble + analyse + factorise), RLC ladder | n = 5 → 5.1 µs; 17 → 11.7 µs; 152 → 109 µs; 1502 → 3,234 µs; **15,002 → 295,346 µs** (91× for 10× the size) | `perf/p6_build.py` |
| Same, with switches (n = 5 / 12 / 310 / 3008) | 9.7 µs / 16.1 µs / 332 µs / **13.6 ms** per mask | `perf/p6_build.py` |
| Share of the build at n = 15,002 | 96 % in Eigen insert and `memmove`; numeric factorisation 1.2 % | `perf/samp_build5000.txt` |
| A first visit to a mask | about 25 steps at n = 12, about 200 steps at n = 3008 | rows 2, 15 |
| Triangular solve at n = 1502 | 58 % of the step, about 13 ns per unknown | `perf/p2_scale.py`, `perf/samp_lad500.txt` |

Where:

- **Assembly.** `G` and `C` are created without reserve
  (`pwl/assemble.hpp:337`) and stamped through `coeffRef(...) +=`
  (40 sites in `stamping/*.hpp`, 5 in `assemble.hpp`); `J` is then a
  third matrix, `G + (1/dt)·C` (`:363`). Each insert into a column
  shifts everything after it.
- **Symbolic analysis per mask.** Every segment creates a solver and
  runs COLAMD and the elimination tree again
  (`pwl/cache.hpp:1664-1676`), although the pattern does not depend on
  the mask (the code says so at `:1349-1350`). `analyze` also fills an
  L/U pattern (`sparse/pulsim_lu_solver.hpp:1134`, called at `:175`)
  that `factorize` clears at once (`:228-233`).
- **Unused low-rank path.** `solve_rank1` and `partial_refactor` exist
  and no engine calls them (callers of the path code are only
  `refactor_parametric` users: electrothermal, MMC, sweeps).
  `solve_rank1` also reassembles `J` on every call
  (`pwl/cache.hpp:1248-1251`).
- **Solve.** `y[k] /= ukk` sits in the back-substitution dependency
  chain (`sparse/pulsim_lu_solver.hpp:477`); the right-hand side is
  formed, row-permuted and column-permuted in three separate passes
  (`pwl/cache.hpp:352`, `pulsim_lu_solver.hpp:453-455`, `:490-492`); the
  work vector is found through a thread-local lookup (`:450`).

## What Changes

Three tiers. Tier 1 is bit-identical; tiers 2 and 3 change rounding and
are gated separately.

- **Tier 1 — linear-time assembly and shared analysis
  (bit-identical).**
  - Per-column reserve from the node degrees, so the first assembly of
    a circuit is linear.
  - A per-circuit stamp-slot map: the patterns of `G`, `C` and `J` are
    built once per assembly regime (dynamic `dt > 0`, static `dt = 0`);
    later masks zero the values and stamp into existing entries in the
    same stamp order, then write `J[k] = G[k] + (1/dt)·C[k]` through
    slot arrays.
  - One symbolic object per circuit and regime (column ordering and
    elimination tree), shared read-only by every segment, event entry
    and the rank-1 solver. A new mask costs a numeric factorisation.
  - `analyze` stops filling the L/U pattern that `factorize` discards.
  - `solve_rank1` uses the slot-map assembly and stops rebuilding
    structure.
  - The solve fuses right-hand-side formation with the row permutation,
    stores `b_constant` pre-permuted per segment, and takes the work
    vector from the caller.
- **Tier 2 — neighbouring masks through `partial_refactor`. BREAKING
  (rounding):** the first visit to a mask that differs from a resident
  one in few bits clones that factor and re-eliminates only the
  elimination paths of the changed columns; pivot faults and long paths
  fall back to a numeric factorisation. The factor is not bit-equal to
  a fresh one when the pivot order differs.
- **Tier 3 — reciprocal pivots. BREAKING (rounding):** the factor
  stores `1/U[k,k]`; back substitution multiplies instead of dividing.
  Last-bit differences in every solution of the in-house LU. Kept only
  if a standalone measurement confirms the gain.
- **Counters:** `structure_builds`, `symbolic_analyses`,
  `symbolic_fallbacks`, `flip_partial_refactors`, `flip_fallbacks` in
  `CacheMetrics`.

Left alone, as the review asked: LU fill and the numeric factorisation,
the lazy mask cache, per-step scaling, thread-local workspaces as the
default, `-ffp-contract=off`. Supernodes, a dense path and Φ/Γ updates
are not part of this change.

## Impact

- Affected specs: `linear-solver`, `pwl-rank1-update`.
- Affected code: `pwl/assemble.hpp`, `pwl/dc_assemble.hpp` (shared
  stamp helper only), `pwl/cache.hpp` (`try_make_segment`, `solve`,
  `solve_at`, `solve_rank1`, metrics), `pwl/segment.hpp`,
  `sparse/pulsim_lu_solver.hpp`, `sparse/solver.hpp`,
  `python/bindings.cpp` (metrics), tests under `core/tests/layer0`,
  `layer4`, `perf`.
- Bit-identity: tier 1 — `J` identical entry by entry, same column
  ordering, same factors, same states on every golden. Tiers 2 and 3 —
  states within a stated tolerance; goldens of the fixed engine are
  regenerated once per tier, with the owner's sign-off.
- Depends on `restore-reference-benchmark-harness`. Feeds
  `update-newton-persistent-solver` (slot map) and
  `update-trbdf2-factor-reuse` (shared analysis for the factor slots);
  neither needs it to start.
- Expected speed-up, **estimates from the time split, not prototyped**:
  first visit of a mask 20–100× faster at n ≥ 1500 (assembly); 2–3× on
  first visits after that (shared analysis and flips); 1.3–1.5× on the
  58 % solve share at large n (the review attributes the 13 ns per
  unknown to the division chain by inference, not by measurement).
- Pending changes `replace-klu-with-pulsim-sparse-lu` and
  `add-generalised-path-refactor` modify requirements of
  `pwl-rank1-update`; this change only adds requirements there.
