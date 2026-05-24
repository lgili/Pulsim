# 7. Path-Based Partial Refactorisation

!!! info "Status: outline / next iteration"
    Source material:
    `core/include/pulsim/sparse/pulsim_lu_solver.hpp`
    (`partial_refactor` method, ~200 lines), the v1.3.0 release
    benchmark, the OpenSpec proposal
    `add-pwl-rank1-partial-refactor`, and the IEEE TPEL paper
    draft under `artigos/02_tpel_methods/`.

**The algorithmic contribution that backs the planned IEEE TPEL
methods paper.** This chapter explains the path-based partial
refactor in full: what it does, why it's correct, when it
applies, when it falls back, and how it earns the captured
2.7-2.9× speedup over the baseline cache.

## Planned sections

1. **The problem**: even with the PWL cache (chapter 4),
   first-encounter of a new mask costs a full LU. SMPS PWM
   often generates Gray-coded single-bit transitions between
   masks; the cache lookup misses on every new permutation.
   We want first-encounter cost to scale with the *change* in
   the matrix, not with its size.

2. **The intuition**: changing one column $c$ of $M$ from
   $M_{\mathrm{old}}$ to $M_{\mathrm{new}}$ only affects $L$ and
   $U$ entries along the **elimination-tree path from $c$ to the
   root**. Updating just that path is $O(|\text{path}|) = O(\sqrt{n})$
   on average, vs $O(\mathrm{nnz} \log n)$ for fresh factorise.

3. **The math**: Chan, Brandwajn & Tinney 1986 (PICA-86) first
   described path-based refactorisation for power-system fault
   analysis. Dinkelbach, Liegmann & Riedel, *Energies* 14:7989,
   2021 generalised it to circuit MNA and gave the modern
   etree-walk derivation. Pulsim implements the Dinkelbach
   formulation directly.

4. **The Pulsim implementation**:
   - `compute_path_()` — for each changed col $c$, map to
     permuted index via $P_{\mathrm{col}}^{-1}[c]$, walk
     `etree_parent_` to root, mark via in-path bitmap.
   - Path re-elimination — iterate path ascending, for each
     column $k$ load $x$ from $M_{\mathrm{new}}$, apply L-updates
     from $j < k$ (some path-updated, some not — both stable),
     update $L$ and $U$ values in-place.
   - **Lazy union**: `varying_set_` accumulates every changed
     col ever seen across the cache's lifetime. Re-computing the
     path only when `varying_set_` actually grows. Cache hits on
     repeat patterns.

5. **Pivot fault recovery**: KLU-style threshold check after
   each path column. $|x[i]| > (1/\mathrm{PIVOT\_THRESH}) \cdot
   |x[k]|$ for $i > k$ means we'd need a row swap the existing
   $P_{\mathrm{row}}$ doesn't permit → invalidate the path cache,
   return false, caller falls back to full factorise. No silent
   corruption.

6. **Pattern-change detection**: if the new column has nonzeros
   at positions the symbolic phase didn't predict (rare under
   PWL switching — switch toggles don't change sparsity pattern
   per chapter 2 §2.6), pattern check rejects + fallback.

## Planned figures

- **Fig 7.1** — Etree path walk: starting from a changed column
  (say col 4), trace parent → grandparent → root and highlight
  the affected $L$ and $U$ entries. Mermaid + matplotlib
  overlay.
- **Fig 7.2** — Single-bit flip example: matrix before and after
  toggling switch S2 of the buck-like 8×8 fixture. Diff overlay
  showing which columns/rows changed. Then the path computed
  for that change.
- **Fig 7.3** — Fault recovery flow: pivot-threshold check
  fails → invalidate path cache → caller re-factorises → next
  call rebuilds varying-set. State diagram.
- **Fig 7.4** — Per-call cost breakdown (path-compute vs path-
  re-eliminate vs pattern-check) as $n$ scales from 6 to 26.
  Bar chart with stacked segments.

## Cross-references

- [Chapter 6 — PulsimSparseLuSolver](06-pulsim-sparse-lu.md)
  explains the data structures (`Prow_`, `Pcol_`,
  `etree_parent_`, `l_col_ptr_`, `u_col_ptr_`) this algorithm
  reads and writes.
- [Chapter 8 — Benchmarks](08-benchmarks.md) shows the captured
  2.7-2.9× speedup decomposition.
- OpenSpec `add-pwl-rank1-partial-refactor` (archived) has the
  original requirements + scenarios this algorithm satisfies.
- **Chan, Brandwajn & Tinney 1986** and **Dinkelbach et al.
  2021** are the canonical literature references.
