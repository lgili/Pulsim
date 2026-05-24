# 6. PulsimSparseLuSolver — Our In-House Implementation

!!! info "Status: outline / next iteration"
    Source material:
    `core/include/pulsim/sparse/pulsim_lu_solver.hpp` (~900 lines,
    header-only), the v1.3.0 release commit history, and the
    OpenSpec proposal `replace-klu-with-pulsim-sparse-lu`.

The v1.3.0 release replaced SuiteSparse KLU with a from-scratch
C++23 implementation. This chapter walks the implementation
line-by-line: the `analyze` → `factorize` → `solve` lifecycle,
the threshold-pivoting rule, why we ditched the symbolic-pattern-
prediction approach that Section 2 of the proposal originally
called for, and the lessons learned from the 3-bug debugging
arc captured in the OpenSpec tasks.

## Planned sections

1. **Why in-house?**: the project owner's 2026-05-24 decision —
   "the TPEL paper's algorithmic novelty must be ours" — and the
   week-by-week tradeoffs vs vendoring DPsim's KLU fork.

2. **`analyze(M)` — symbolic phase**: build symmetric adjacency
   → RCM column order → elimination tree (Liu 1986 disjoint-set
   ancestor compression) → symbolic L+U pattern. ~370 lines of
   header-only C++23. Test coverage: 60 assertions.

3. **`factorize(M)` — Gilbert-Peierls + partial pivoting**: dense
   workspace `x[n]`, left-looking column elimination, the L-update
   inner loop. Why the symbolic pattern from §2 was *abandoned*
   in favour of dynamic pattern discovery from $x$'s runtime
   nonzeros (because partial pivoting changes $P_{\mathrm{row}}$
   in ways the pre-pivot pattern doesn't anticipate). The KLU-
   style threshold pivot rule (`PIVOT_THRESH = 1e-3`).

4. **`solve(b, x)` — forward + back substitution**: 4 steps —
   apply row permutation, forward subs on $L$ (unit diagonal),
   back subs on $U$ (diagonal at LAST slot), apply inverse
   column permutation. Why the "diagonal at last slot" convention
   matters for the back-subs inner loop.

5. **The 3 bugs we hit**: (a) zero pivot at column 2 of the
   buck-like 8×8 (fix: implement partial pivoting), (b) `err =
   1.0` on solve identity (fix: dynamic pattern discovery), (c)
   `l_col_ptr_[k+1]` set at the wrong time (fix: move the
   update to the END of column $k$'s storage). Each bug is a
   useful teaching moment about sparse-LU implementation.

6. **Backend factory**: `make_default_solver(n, hint)` returns
   `PulsimSparseLuSolver` for `Backend::Auto` (since v1.3.0).
   `Backend::Eigen` returns the reference `SparseLuSolver`
   (kept intentionally as the benchmark baseline).

## Planned figures

- **Fig 6.1** — Lifecycle state diagram (`uninitialised` →
  `analyzed` → `factorised` → `solved`). Mermaid.
- **Fig 6.2** — Dynamic pattern discovery animation: show how
  the L and U patterns emerge from $x$'s runtime nonzeros, vs
  what the pre-pivot symbolic phase predicted. Highlight the
  rows that the symbolic phase missed.
- **Fig 6.3** — Pivot-row swap visualisation: matrix elements
  rearranged before vs after a row-2-↔-row-7 swap during the
  factorisation of the buck-like 8×8 fixture.
- **Fig 6.4** — Layer breakdown of the ~900-line header: pie
  chart by file region (analyze / factorize / solve /
  partial_refactor / state).

## Cross-references

- [Chapter 5 — Sparse LU Foundations](05-sparse-lu-foundations.md)
  is the prerequisite reading; this chapter assumes you've seen
  Gilbert-Peierls + RCM + etree in the abstract.
- [Chapter 7 — Path-Based Partial Refactor](07-rank1-partial-refactor.md)
  extends this class with the algorithmic contribution of v1.3.0.
- [Layer 0 internals doc](../internals/layer0-numeric-and-sparse.md)
  has the type alias reference and the original `DirectSolver`
  base-class definition.
- OpenSpec proposal `replace-klu-with-pulsim-sparse-lu` (archived
  2026-05-24) has the full task-by-task history of the rewrite.
