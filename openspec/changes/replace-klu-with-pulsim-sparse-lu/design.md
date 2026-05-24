## Context

V0 (v1.2.0, merged 2026-05-24, PR #33) shipped a KLU-backed
`partial_refactor` with MVP delegation to `klu_refactor` (full
numeric refactor with cached symbolic). The shipped capability spec
`pwl-rank1-update` documents KLU as a required backend and references
KLU-specific functions in its scenarios.

V8.1 was originally proposed (commit 93353bf, since deleted) as a
vendor-the-DPsim-fork upgrade: pull in the third-party
dpsim-simulator/SuiteSparse fork which adds path-based partial
refactor primitives on top of upstream KLU. The project owner
rejected that direction on 2026-05-24 with the explicit reasoning:
"vamos fazer a nossa do zero ... pode se basear na deles mas vamos
implementar a nossa nem que demore."

This change implements the algorithm from scratch in C++23, on top
of Pulsim's existing `Eigen::SparseMatrix<Real, ColMajor, int32>`
container, with **no SuiteSparse dependency of any kind** (neither
upstream Davis KLU nor the DPsim fork). Eigen is used only as a
matrix container — none of its own LU machinery (SparseLU,
COLAMDOrdering, etc.) is invoked.

## Goals / Non-Goals

**Goals**

- Drop every direct and indirect SuiteSparse dependency from Pulsim.
- Implement a complete sparse LU stack in C++23: ordering, symbolic
  analysis, numeric factorization with partial pivoting, triangular
  solve, elimination tree, path-based partial refactor.
- Preserve the public `DirectSolver` interface and
  `PwlStateSpaceCache::solve_rank1` contract — no caller code change.
- The path-based partial refactor produces L and U values
  bit-identical (within 1e-14) to a fresh full factorize when pivots
  are unchanged.
- Honest performance baseline: PulsimSparseLuSolver's analyze + factor
  costs are allowed to be 1.5-3× slower than `Eigen::SparseLU` (we are
  not Davis; our COLAMD will be RCM, etc.). The win comes from the
  partial_refactor path, where Eigen can't compete.

**Non-goals**

- Out-performing KLU's analyze + factor wall-time. KLU has 20+ years
  of refinement we are not replicating in 6-10 weeks.
- Implementing BTF (block triangular form) decomposition. BTF is
  KLU's secret sauce for circuit MNA, but it's a separate optimization
  layer; we ship without it in V8.1 MVP and add later if benchmarks
  warrant.
- Implementing COLAMD or AMD fill-reducing ordering in the MVP.
  Reverse Cuthill-McKee is simpler (~80 lines), well-understood, and
  produces orderings within 1.5-2× of COLAMD's fill on typical circuit
  matrices. Replace with COLAMD/AMD as a follow-up only if benchmark
  data shows it's binding.
- Multi-bit partial refactor. Gray-code enumeration in `solve_rank1`
  guarantees single-bit transitions are the common case; multi-bit
  flips go to full factorize via the existing `solve_rank1` fallback.
- GPU / multi-threaded factorization. Single-threaded baseline only.

## Decisions

### D1. RCM ordering as MVP (vs COLAMD or no ordering)

Reverse Cuthill-McKee (RCM) is a bandwidth-reducing ordering — not
fill-reducing per se, but bandwidth reduction correlates with fill
reduction on most circuit matrices. Implementation: ~80 lines of
BFS-based reordering, very well-tested algorithm with no numerical
subtleties.

Alternatives considered:
- **NATURAL ordering** (no reorder): simplest possible, but fill-in
  explodes on circuits with hub nodes. Unacceptable.
- **AMD** (Approximate Minimum Degree, Amestoy/Davis/Duff 1996): the
  go-to fill-reducing ordering for symmetric matrices. ~400-600 lines
  to implement; complex graph data structures.
- **COLAMD** (Column Approximate Minimum Degree, Davis 2004): the
  asymmetric variant used by both KLU and Eigen::SparseLU. ~1000+
  lines.

RCM is the MVP. AMD or COLAMD is a follow-up — easy to swap in once
the rest of the stack is validated, because the ordering layer is
isolated behind `ColumnOrdering`.

### D2. Numeric factorization: Gilbert-Peierls left-looking with partial pivoting

This is the canonical algorithm (Gilbert & Peierls, *SIAM J. Sci.
Stat. Comput.* 9, 1988, 862-874). KLU uses it. Eigen::SparseLU uses
it. Davis 2006 §3-4 has reference C code. Well-tested across decades
of use; numerical stability properties are documented.

Alternative — right-looking factorization: also a classical choice,
slightly faster per-step but harder to combine with partial pivoting
(needs full row updates per column). Left-looking is the better fit
for path-based partial refactor too, because the path-traversal
visits columns in increasing order — the same order left-looking
processes them.

### D3. Partial pivoting: column-by-column threshold

For each column k during factorization, scan the lower-triangular
entries in column k (post-update), find the row with the largest
magnitude, swap that row with row k. Standard partial pivoting.

Threshold variant ("threshold pivoting" or "thresh-pivot") — used
by KLU/Davis — allows a pivot below the column max provided it's
above `thresh * column_max` for some `thresh ∈ (0,1]` (default 0.1).
This produces sparser factors than strict partial pivoting at small
numerical-stability cost. We use thresh = 1.0 (= strict partial
pivoting) for the MVP; tune later if benchmarks warrant.

### D4. Storage: L and U in separate Eigen::SparseMatrix members

After factorization, L (unit-lower triangular) and U (upper
triangular) live in two separate `Eigen::SparseMatrix<Real, ColMajor,
int32>` member variables. Triangular solve reads them directly via
Eigen's CSC accessors.

Alternative — combined LU matrix (one CSC structure holding L below
diagonal, U on/above diagonal): more compact, slightly faster solve.
KLU uses this. The downside: column iteration during partial refactor
is fiddly (have to skip the U part when re-eliminating L's
contributions to a column). Separate L+U is clearer; combined LU is
a possible V8.2 optimization.

### D5. Path-based partial refactor: lazy union design (V8.1 carry-over)

Same design as the previously-deleted V8.1 proposal:
`PulsimSparseLuSolver` maintains a lazy union of all columns ever
passed as `changed_cols`. On each `partial_refactor` call:
- If a new column joined the union → recompute path
- Else → reuse the cached path
- Either way, re-eliminate path columns

This handles the Gray-code enumeration case (after N flips, the
union covers all N switches' stamps, then never recomputes). Cache
invalidation: clear union on `analyze()` call or on pivot-fault.

### D6. Pivot-fault threshold: |pivot| < 0.1 × column_max → fault

Per Dinkelbach §3.2. When a perturbation pushes the would-be pivot
below 0.1× the column's largest magnitude, the precomputed path is
no longer correct (row swaps would be needed). Conservatively:
return false, invalidate the cached path, let the caller fall back
to a fresh `factorize`.

The 0.1 threshold is the conventional choice (KLU's default too).
Lower threshold = fewer faults (more partial refactor wins) but
greater numerical risk. Tune via benchmark data.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Sparse LU with partial pivoting has many edge cases (near-singular pivots, fill-in explosions on pathological matrices). | Comprehensive unit tests in Section 3.5: SPD reference, asymmetric MNA, deliberately-singular, deliberately-zero-pivot. Plus integration tests in Section 5.7 covering pivot fault recovery. |
| Our RCM ordering may produce significantly more fill than Eigen's COLAMD, slowing factorize substantially on real circuits. | Acceptance test 2.8.2 allows ±50% slack vs Eigen's fill count. If benchmark on the 10 reference projects shows >2× slowdown, prioritize AMD or COLAMD migration as a v1.4.0 follow-up. |
| 6-10 week schedule slip cascades to TPEL paper. | Honest expectation set in `proposal.md`. TPEL submit target shifts from Oct 2026 to Q1 2027. The JOSS paper (paper #1) is unaffected — it's already in JOSS review queue. |
| Eigen::SparseLU's analyze() uses int32 indices; our equivalent must match. | All Pulsim sparse storage is already `Index = int32`. No mismatch. |
| The new code path is genuinely novel — bugs we ship aren't covered by KLU's decades of in-the-wild testing. | Validation test 6.5 re-runs the NPC + MMC projects (the most complex circuits in `projects/`) and asserts bit-identical output vs the V0 baseline. Any drift signals a kernel bug. Plus the layer4_v1 stress tests already exercise hundreds of switching configurations. |
| Eigen-only build was the "fallback path" in V0 — now it's the only path. Any latent issue with `Eigen::SparseLU` for solve_rank1's full-refactor fallback is now load-bearing. | Section 1.8 verifies layer0 + layer4 tests pass post-KLU-removal (with only Eigen::SparseLU). This is the existing V0 fallback code; it has been working since V0 shipped. |

## Migration Plan

Pure additive at the **Python API + reference projects** level — the
public `pp.simulate(...)` keeps working, all 8 reference projects in
`projects/` produce identical output.

**Breaking change at the C++ kernel-builder level only**: any
out-of-tree code that referenced `KluSolver` or `Backend::KLU` must
switch to `PulsimSparseLuSolver` / `Backend::Pulsim`. The
`make_default_solver()` factory entry point continues to work.

Rollback: revert this change's commits. V0 (KLU + Eigen fallback)
re-engages. No data format changes.

## Open Questions

- Q: Should we ship RCM as the only ordering, or include AMD now (3-4
  extra weeks of work)?
  Tentative: ship RCM in V8.1; AMD in V8.2 if benchmark warrants.
- Q: Should `Backend::Auto` ever fall through to `Eigen::SparseLU`?
  The Eigen fallback was useful in V0 when KLU was optional. Now there's
  no scenario where we'd prefer Eigen over Pulsim's LU.
  Tentative: drop Eigen from `Backend::Auto`; only `Backend::Eigen`
  (explicit) returns SparseLuSolver.
- Q: Pivot threshold default 0.1 vs 1.0 (strict)?
  Tentative: 1.0 for MVP (strict pivoting, maximum stability). Tune later.
- Q: Do we expose a `pivot_threshold` setter on PulsimSparseLuSolver?
  Useful for research benchmarks. Tentative: yes, with default 1.0.
