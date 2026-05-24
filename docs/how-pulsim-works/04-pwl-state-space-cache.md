# 4. PWL State-Space Cache

!!! info "Status: outline / next iteration"
    This chapter is scaffolded with the full section structure +
    figure plan but the body prose is being written in the next
    iteration of the "How Pulsim Works" doc effort. Source
    material is `docs/internals/layer4-pwl-state-space-cache.md`
    plus `core/include/pulsim/pwl/cache.hpp`.

The architectural pivot. Chapter 3 left us with one linear solve
per step. This chapter explains how Pulsim makes that solve
practically free by pre-building one factorisation per switch
mask and looking it up at runtime.

## Planned sections

1. **The cache lifecycle**: `build_lazy(dt)` → `solve(mask, b_extra, x)`.
   How `PwlStateSpaceCache` lazily fills its `segments_` map on
   first encounter of each mask. Why eager build was rejected as
   the default.

2. **The `PwlSegment` data structure**: per-mask record holding
   the discretised companion matrix `J`, the constant RHS
   contribution `b_constant`, and an analyzed+factorised
   `DirectSolver` handle.

3. **Per-step `solve(mask, b_extra, x)` walkthrough**: 4 lines
   of C++ that replace SPICE's 200-line Newton loop. The
   triangular-solve cost vs the per-mask build cost; amortisation
   curve.

4. **What happens when a brand-new mask shows up**: `build_lazy`
   adds an entry to `segments_`, runs `analyze + factorize` once,
   the next `solve(...)` is fast. The "first hit is slow"
   property and why it's almost never visible in practice.

5. **The 3-backend story** (preview of chapter 8): same cache
   exercised against three `DirectSolver` backends — baseline
   per-mask cache, sliding-solver amortisation,
   path-based partial refactor.

## Planned figures

- **Fig 4.1** — Cache state at $t=0$ (empty) vs after 1000
  switching cycles (filled with the 2-3 visited masks). Side-by-
  side memory-diagram.
- **Fig 4.2** — Lifecycle flowchart: mask → cache hit? → solve;
  mask → cache miss? → build + factorise + solve. Mermaid.
- **Fig 4.3** — Build-cost-vs-step-count amortisation curve.
  Pulsim baseline solve vs SPICE-style fresh-factor on the
  buck fixture over 1000 cycles.

## Cross-references

- [Chapter 3 — Trapezoidal Companion](03-trapezoidal-companion.md)
  defines the $J\mathbf{x}_{n+1} = \mathbf{b}_n$ system this cache stores.
- [Chapter 7 — Path-Based Partial Refactor](07-rank1-partial-refactor.md)
  extends this cache with a hot-path optimisation for single-bit
  mask transitions.
- [Layer 4 internals doc](../internals/layer4-pwl-state-space-cache.md)
  has the per-method walkthrough.
