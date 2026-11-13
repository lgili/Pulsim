## Context

A cached segment is built by `PwlStateSpaceCache::try_make_segment`
(`pwl/cache.hpp:1648`): assemble `J` and `b`, create an in-house LU,
`analyze`, `factorize`. Assembly stamps `G` and `C` with `coeffRef +=`
and adds them into `J`. The in-house LU (Gilbert–Peierls, COLAMD,
partial pivoting) keeps the column ordering and the elimination tree
from `analyze`; row pivots and the L/U pattern are produced by
`factorize`.

Facts the design rests on, read from the code:

- The stamp positions do not depend on the mask: a switch stamps the
  same four entries with `g_on` or `g_off`
  (`stamping/stamp_switch.hpp:65-75`); the diode drop goes to `b` only.
- Static assembly (`dt = 0`) skips capacitors and inductors, so it has
  another pattern (`pwl/cache.hpp:1978-1989`).
- `partial_refactor` re-eliminates along elimination paths with the
  existing row order and reports a pivot fault instead of re-pivoting
  (`sparse/pulsim_lu_solver.hpp:520-545`).
- The solver is not copyable.

Constraints: LU fill and the numeric factorisation are not the problem
(1.2 % of the build at n = 15,002) and stay; the thread-local solve
workspace stays as the default; `-ffp-contract=off` stays.

## Goals / Non-Goals

- Goals: the first assembly of a circuit linear in its size; a later
  mask costs a value pass and a numeric factorisation; one symbolic
  analysis per circuit and regime; fewer passes in the solve; tier 1
  bit-identical.
- Non-Goals: supernodes, BTF, SIMD, threshold pivoting; a dense path
  or Φ/Γ (`add-exact-propagator-strategy`); the Newton path
  (`update-newton-persistent-solver`); the TR-BDF2 event table
  (`update-trbdf2-factor-reuse`); templating the stampers.

## Decisions

- **Reserve first, slot map second.** Step S: `G` and `C` reserve per
  column from the node degrees (branches incident to the node plus
  constraint rows). The stamps still run `coeffRef +=` in the same
  order, so pattern and values are unchanged; only the shifting goes.
  Step M: the first assembled `G`, `C`, `J` of a regime become the
  circuit's pattern template. Later assemblies copy the index arrays,
  zero the values and stamp; `coeffRef` then finds an existing entry.
  `J[k] = G[k] + inv_dt · C[k]` is written through two slot arrays that
  map the entries of `G` and of `C` to the union pattern, absent
  entries taken as `+0.0`.
  - Alternative: triplets and `setFromTriplets`. Rejected for tier 1:
    Eigen does not document the order in which duplicate triplets are
    summed, and an entry stamped by three devices could round
    differently.
  - Alternative: template every stamper on a sink that writes through
    a cursor. Rejected for now: 45 stamp sites, and the per-stamp
    search in a column of a handful of entries is not on the profile
    once inserts are gone. Revisit if task 9.2 shows the search.
- **Why this is bit-identical.** A fresh entry is created as `0` and
  then `+=`; a zeroed existing entry gives the same sum in the same
  order. The slot write is the per-entry arithmetic of
  `G + (Real{1} / dt) * C`. The check is entry-by-entry equality of `J`
  on every mask of every golden circuit.
- **The symbolic object lives on the cache.** `PwlStateSpaceCache`
  owns one immutable `Symbolic` per regime (column permutation, its
  inverse, the elimination tree), created by the first segment of that
  regime and held by `shared_ptr<const>`. Segment solvers, event
  solvers and the rank-1 solver adopt it instead of analysing. The
  per-solver path cache of `partial_refactor` stays per solver.
  - Invalidation: the object is tied to the cache's graph and pool.
    `refactor_parametric` changes values only and keeps it. A solver
    asked to adopt a symbolic object checks the matrix's index arrays
    against the template; on a mismatch it analyses privately and
    `symbolic_fallbacks` is incremented. A regime change uses the other
    object.
  - Alternative: keep `Eigen::SparseLU` style per-solver analysis and
    cache only the permutation. Same thing with more copies.
  - Bit-identity: COLAMD and the tree depend only on the pattern, and
    the pattern is the same array; tested by comparing the shared
    ordering with a private analysis on every golden mask.
- **Dead symbolic fill removed.** `compute_symbolic_pattern_` fills
  L/U index arrays that `factorize` clears before use. Task 4.1 first
  confirms nothing reads them between `analyze` and `factorize`
  (introspection accessors, tests); if something does, the fill
  becomes lazy.
- **Fused solve passes are bit-identical; the reciprocal is not.**
  `y[i] = −(b_constant[Prow[i]] + b_extra[Prow[i]])` is the same
  arithmetic as forming `rhs` and permuting it. `b_constant` is stored
  already permuted per segment (the row order is per factor). The
  engine loops pass a run-local work vector; the thread-local one stays
  as the default for every other caller. Multiplying by a stored
  `1/U[k,k]` rounds differently from dividing, so it is tier 3.
- **Tier 2 routes first visits, not hits.** On a lazy miss the cache
  looks for the resident segment nearest in Hamming distance (ties:
  most recent). If the changed columns' path is short (the existing
  `MAX_PATH_LENGTH_RATIO = 0.6` gate), it clones that factor (`clone()` on the in-house
  solver, O(nnz)) and calls `partial_refactor`. Otherwise, or on a
  pivot fault, it factorises numerically on the shared analysis.
  - Not bit-identical: without re-pivoting the factor can differ from a
    fresh one. Gate: states within 1e-9 relative on the goldens, solve
    backward error not above baseline.
  - Alternative: use `solve_rank1`'s single sliding factor in the main
    loop. Rejected: it keeps one factor, so every revisit of a mask is
    a refactor; the per-mask cache makes revisits free.
- **Tier 3 needs its own evidence.** The review did not benchmark the
  solve in isolation. Task 8.1 builds a standalone microbenchmark; tier
  3 proceeds only if the reciprocal gives at least 1.15× on the solve
  at n = 1502.

## Risks / Trade-offs

- A mask whose pattern differs from the template would silently use a
  wrong ordering → index-array check on adoption, `symbolic_fallbacks`
  counter, test with a hand-built mismatching matrix.
- Shared symbolic object and concurrent runs: it is immutable after
  creation and created on the miss path, which already mutates the
  cache → same discipline as a segment insert.
- Tier 2 on stiff conductance ratios (`g_on/g_off = 1e12`): pivot
  faults may be frequent → `flip_fallbacks` reported; tier 2 is dropped
  if the fallback ratio exceeds 50 % on the multilevel probe.
- Regenerating fixed-engine goldens twice (tier 2, tier 3) → do both in
  one release, or drop tier 3.
- Estimates from a time split → baseline first, acceptance on the same
  probe.

## Migration Plan

1. Counters and baseline.
2. Tier 1 in four commits (reserve, slot map, shared symbolic, solve
   passes), each against the goldens. Rollback: revert the commit.
3. Tier 2 behind a cache flag, off by default; default on after the
   tolerance gate and the owner's sign-off.
4. Tier 3 only after task 8.1; same gate.

## Open Questions

- Tier 2 and tier 3 both change fixed-engine rounding. Does the owner
  accept regenerating the fixed-engine goldens in this phase, or should
  Phase 1 stop at tier 1 and leave tiers 2–3 for the release that
  already carries **BREAKING** numerical changes (Phase 2)?
- Tolerance for tiers 2–3: 1e-9 relative on states, or a multiple of
  the solve's backward error?
- Is a 50 % fallback ratio the right stop rule for tier 2?
