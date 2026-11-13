# Decisions taken while implementing tier 1

Each entry: question → option taken → why → how to reverse.

1. **Tiers 2 and 3 in Phase 1?** (design.md Open Question 1) → No.
   Only tier 1 is implemented; tasks 7.x and 8.x stay open. → They
   change rounding and the owner has not signed off on regenerating the
   fixed-engine goldens. → Implement them in the release that carries
   the other numerically breaking changes.

2. **Tolerance and stop rule for tiers 2–3** (Open Questions 2 and 3)
   → not decided; not needed for tier 1.

3. **Reserve step (task 2.1) and slot map (task 3.1) landed together.**
   → The first assembly of a regime reserves per column from the node
   incidence (an estimate; Eigen grows a column that overflows); every
   later assembly stamps into a copy of the regime's template. → The
   reserve only matters for the one structural assembly per regime, so
   a separate measurement of "reserve alone" would describe a state
   the code never ships in. → Revert the slot-map hunk in
   `pwl/assemble.hpp` / `pwl/cache.hpp`; the reserve stays valid alone.

4. **Template invalidation** → no explicit invalidation. Each template
   assembly checks the number of stored entries after stamping; a
   stamp that lands outside the template (a device added to the pool
   after the cache was created) inserts, the count changes, and the
   assembler rebuilds the template from a fresh assembly
   (`structure_builds` + 1). A change of `state_size` also rebuilds. →
   Stamp positions depend only on topology (no stamp is conditional on
   the mask or a value), and the pool is append-only, so a stale
   template can only be smaller than the real pattern. → Clear the
   templates in `build()` / `build_lazy()` if a pool ever becomes
   able to remove devices.

5. **Dead symbolic L/U fill (task 4.1)** → kept in `analyze()`, not
   run by `adopt_symbolic()`. → `l_nnz()` / `u_nnz()` are read after
   `analyze()` alone by two LU tests (fill comparisons against Eigen's
   COLAMD), so the fill has readers; with the shared analysis
   `analyze()` runs at most twice per cache, so the fill is no longer
   on any per-mask path ("lazy" in effect). → Delete
   `compute_symbolic_pattern_` and change those tests to factorise
   first.

6. **Symbolic object holds copies** → `adopt_symbolic` copies the
   column permutation, its inverse and the elimination tree into the
   solver (O(n)) instead of pointing at the shared object. → Keeps every
   existing member access in the LU unchanged; O(n) is below the O(nnz)
   pattern check and the factorisation it precedes. → Hold a
   `shared_ptr<const SymbolicAnalysis>` in the solver and read through
   it.

7. **Fallback does not replace the shared analysis** → after a pattern
   mismatch the cache keeps the first analysis of the regime. → The
   mismatch case is a pool changed under a live cache, which is
   unsupported; the counter makes it visible. → Replace the regime's
   analysis on fallback.

8. **Pre-permuted `b_constant` (task 5.1, first half)** → not done. The
   fused pass is `y[i] = −(b_constant[P[i]] + b_extra[P[i]])` through a
   new `DirectSolver::solve_neg_sum`, which removes the separate
   right-hand-side vector and pass. A per-segment pre-permuted copy
   would save one indexed load per row and needs the row order outside
   the solver. → Smallest change with the same arithmetic. → Add a
   `b_perm` to `PwlSegment` filled after `factorize`.

9. **Run-local solve work vector passed by the engine loops (task
   5.2)** → not done; the thread-local workspace stays the only one. →
   It needs new parameters through `solver/run_transient.hpp` and
   `solver/trbdf2_transient.hpp`, which another track owns, for a TLS
   lookup that is not visible in the profile. → Add an optional
   `Vector* work` to `solve` / `solve_at` and pass it from the loops.

10. **`structure_builds` counts template builds of the cache's
    assembler**, i.e. both `G`/`C` (dynamic) and `J` (static) count as
    one build per regime. The spec bound "≤ 2 per cache" is that count.

11. **Acceptance 6.1 is three-quarters met; the VSI first-visit target
    is missed.** Measured with the probe's `perf` mode, base and tier-1
    builds interleaved three times on the shared, loaded Mac (best of
    all runs per case; run-to-run spread ±25 %, one whole run 2.4×
    slower):

    | Measure | Before | After | Target |
    |---|---|---|---|
    | build(15,002) ÷ build(1502) | 85 | 10.6 | ≤ 12 ✓ |
    | per-mask build, switched ladder n = 3008 | 5737 µs | 243 µs (0.042×) | ≤ 0.1× ✓ |
    | per-mask build, switched ladder n = 310 | 143 µs | 24 µs (0.17×) | — |
    | per-mask eager build, VSI (n = 12, 4096 masks) | 7.1 µs | 4.4 µs (0.62×) | ≤ 0.5× ✗ |
    | first build of a single segment, RLC n = 5 / 17 / 152 | 2.5 / 5.5 / 48 µs | 3.3 / 7.9 / 78 µs | — |
    | ns/step, 19 golden cases | — | 0.80–1.24×, inside the noise | no regression |

    At n = 12 there is no quadratic insert cost to remove; what remains
    per mask is the numeric factorisation and the cache bookkeeping, so
    the VSI cannot reach 0.5× from tier 1 alone. The first segment of a
    cache now also builds the template and its slot arrays (one extra
    `G + C` and two merge walks), a one-time cost of a few µs to ~30 µs
    at n ≤ 152. → Keep tier 1 as is, leave 6.1 unticked, and report the
    numbers. → Tier 2 (partial refactor on single-switch flips) is the
    proposal's lever for first visits on small circuits.
