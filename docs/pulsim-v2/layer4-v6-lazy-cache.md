# Layer 4 V6 — Lazy cache build

Layer 4 V0-V5's `cache.build(dt)` factorises ALL `2^N` segments
upfront. For circuits where only a few switch states are
actually visited (typical of PWM converters in fixed-duty
operation), this wastes time + memory on never-used factors.

**Layer 4 V6 adds `build_lazy(dt)`**: stores the dt but defers
factorisation until each unique mask is first requested via
`solve(mask, ...)`. The on-demand factor build is amortised
over later solves with the same mask.

## API

```cpp
class PwlStateSpaceCache {
public:
    // Eager build (V0-V5 default): factorise all 2^N upfront.
    void build(Real dt);

    // NEW: lazy build (V6). Stores dt, defers factor.
    void build_lazy(Real dt);

    // Number of segments currently in the cache.
    // Eager build → 2^N immediately.
    // Lazy build  → grows as solve() populates it.
    [[nodiscard]] Size num_built_segments() const noexcept;

    // solve() in lazy mode builds the requested segment if
    // not already cached. Eager mode throws on missing mask.
    void solve(const SwitchStateMask& mask,
                const Vector& b_extra, Vector& x) const;
};
```

## Why not Sherman-Morrison

The original V0 target was Sherman-Morrison rank-1 updates
between Gray-code-adjacent switch states. The math:
`(A + uvᵀ)⁻¹ = A⁻¹ − (A⁻¹uvᵀA⁻¹) / (1 + vᵀA⁻¹u)`.

**Blocker**: SM updates the inverse, but our cache uses
**sparse LU factors** via KLU/SparseLU. KLU doesn't expose
factor-update primitives, and implementing SM at the inverse
level would destroy sparsity.

Specialised libraries exist (Cholmod for SPD, custom non-
symmetric updaters), but integrating one is a significant
backend change. Future research OpenSpec.

Lazy build hits the SAME "wasted factor" target from a
different angle: skip factors that are never needed. For
typical PE workloads where only ~3-12 of 2^N states are
visited, lazy build provides 5-20× speedup on cache
construction.

## Verified

- **Lazy build starts empty**: `num_built_segments() == 0`
  after `build_lazy(dt)`.
- **Populates on demand**: each unique mask seen by `solve()`
  bumps the count by 1.
- **Caches**: repeat solves on the same mask do NOT rebuild.
- **Same results as eager**: lazy + eager produce identical
  `x` for the same `(mask, b_extra)` inputs.
- **Mode switches**: `build()` after `build_lazy()` clears +
  populates all. `build_lazy()` after `build()` clears + waits.

## Status

| Layer | Cases | Assertions |
|---|---|---|
| 0 | 19 | 80 |
| 1 | 36 | 126 |
| 2 | 36 | 93 |
| 3 | 16 | 61 |
| 4 V0 | 24 | 58 |
| 5 V0 | 21 | 2069 |
| **4 V1 + V6** | **37** | **90** ← +5 / +14 lazy tests |
| 5 V1 | 17 | 59 |
| 5 V2.2 | 20 | 46 |
| 4 V2 | 9 | 520 |
| 4 V3 | 5 | 13 |
| 5 V4 + V5 | 7 | 66 |
| **Total** | **247** | **3281** |
