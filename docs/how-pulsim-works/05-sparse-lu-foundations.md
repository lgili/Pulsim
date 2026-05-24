# 5. Sparse Direct LU Foundations

!!! info "Status: outline / next iteration"
    Source material: `docs/internals/layer0-numeric-and-sparse.md`,
    Davis 2006 *Direct Methods for Sparse Linear Systems*,
    Gilbert & Peierls 1988, George 1971.

The numerical foundations a reader needs before
[chapter 6](06-pulsim-sparse-lu.md) walks through Pulsim's
in-house implementation. Why direct (not iterative), why sparse
(not dense), and the three algorithms — Reverse Cuthill-McKee
ordering, elimination-tree construction, and Gilbert-Peierls
left-looking factorisation — that make sparse LU practical.

## Planned sections

1. **Why direct, not iterative**: SMPS MNA matrices have $n
   \approx 5\text{–}50$, factorise in microseconds, and need to
   be solved exactly per step. GMRES/BiCGSTAB are great for
   $n > 10^5$; below that, direct wins by 1-2 orders of magnitude.

2. **Why sparse, not dense**: a $50 \times 50$ dense LU is
   $\sim 25\text{ µs}$ on modern hardware. With $\mathrm{nnz}
   \approx 200$ (typical SMPS), sparse LU is $\sim 3\text{ µs}$.
   At the per-step cadence Pulsim targets ($10^7\text{–}10^9$
   steps), that 10× matters.

3. **Fill-reducing orderings**: George 1971 RCM, Davis & Hu
   COLAMD, AMD. What fill is, why minimum-fill orderings matter,
   why Pulsim picks RCM (banded structure for SMPS topologies
   recovers $O(n\sqrt{n})$ fill instead of $O(n^2)$).

4. **Elimination trees**: Liu 1986 / Davis 2006 §4.10. The
   parent[k] data structure that captures "to eliminate column
   k, you may add fill into columns parent[k], grandparent[k],
   …". This is the backbone of both symbolic factorisation and
   chapter 7's path-based partial refactor.

5. **Gilbert-Peierls left-looking factorisation**: the
   workspace-driven algorithm that materialises one L+U column
   at a time, using the etree to predict the column's nonzero
   pattern. Pseudo-code + complexity analysis.

6. **Partial pivoting under sparse direct**: how the row-swap
   for numerical stability interacts with the column ordering;
   why pivots that violate the fill-reducing column permutation
   are usually safe to accept (threshold pivoting per KLU).

## Planned figures

- **Fig 5.1** — Fill comparison: natural ordering vs RCM vs
  COLAMD on the MMC-arm matrix. Side-by-side spy plots showing
  fill explosion under natural, banded structure under RCM.
- **Fig 5.2** — Elimination tree for the buck-like 8×8 fixture.
  Hierarchical tree diagram with parent → child arrows.
- **Fig 5.3** — Gilbert-Peierls column trajectory: animation-
  style sequence of frames showing $L$ and $U$ accumulating
  column-by-column.
- **Fig 5.4** — Per-call cost vs $n$: dense Gaussian
  elimination $O(n^3)$, dense LU $O(n^3/3)$, sparse LU (RCM
  + GP) $O(\mathrm{nnz} \log n)$, on the same log-log axes.

## Cross-references

- [Chapter 6 — PulsimSparseLuSolver](06-pulsim-sparse-lu.md) is
  Pulsim's implementation of every algorithm covered here.
- [Layer 0 internals doc](../internals/layer0-numeric-and-sparse.md)
  has the type-aliases reference (`Real`, `Index`, `Matrix`,
  `Vector`, `DirectSolver`).
- **Davis 2006** is THE textbook reference for this whole
  chapter; chapter 5's bibliography points to specific sections.
