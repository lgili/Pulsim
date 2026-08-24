# Layer 4 V7 — Multi-dt cache (`solve_at`)

Layer 5 V2.2 added sub-step commutation TIMING via linear
interpolation. To take a true sub-step STATE correction, we
need the ability to solve at dt values other than the primary
build dt. The trap-companion's `g_eq = 2C/dt` makes the matrix
dt-specific, so a partial step at `dt_partial < dt_main` needs
a separately factored matrix.

**Layer 4 V7 adds `solve_at(mask, dt, b_extra, x)`**: a
primitive that lazily builds + caches segments at arbitrary
auxiliary dt values, separate from the primary `dt_` set by
`build()` / `build_lazy()`.

This is the **enabling primitive** for true sub-step state
correction (a follow-up OpenSpec for the run_transient
integration).

## API

```cpp
class PwlStateSpaceCache {
public:
    // Primary single-dt path (V0-V6, unchanged).
    void build(Real dt);
    void build_lazy(Real dt);
    void solve(const SwitchStateMask& mask,
                const Vector& b_extra, Vector& x) const;

    // V7: multi-dt path.
    void solve_at(const SwitchStateMask& mask, Real dt,
                   const Vector& b_extra, Vector& x) const;

    // Diagnostics for the multi-dt cache.
    [[nodiscard]] Size num_alt_dt_values() const noexcept;
    [[nodiscard]] Size num_alt_segments_at(Real dt) const noexcept;
};
```

When `dt == this->dt()` (the primary dt), `solve_at` delegates
to `solve` — same fast path.

**v2.0 Phase 1 rework** (audit finding
`alt-dt-cache-unbounded-factorization`): the original V7 kept an
auxiliary map keyed by *exact Real dt* of per-mask segments — one
fully analyzed + factorized segment per `(mask, dt)` pair,
forever. Sub-step event correction produces an essentially unique
dt per commutation, so that map grew without bound. Now each
recently-evented mask owns ONE **event solver** (small LRU,
`kMaxEventEntries = 8`) storing the mask's `(G, C, b)` split; a dt
change recombines `J = G + (1/dt)·C` and runs a numeric
`factorize()` on the entry's single symbolic analysis (the
sparsity pattern is dt-invariant). No dt-keyed storage exists.
`num_alt_dt_values()` / `num_alt_segments_at(dt)` keep their
signatures but now report over the ≤ 8 resident entries' *current*
dt values. `refactor_parametric` drops the entries (their split is
a snapshot of the pool's parameters).

## Why not Sherman-Morrison

The original V7-as-Sherman-Morrison idea would build factors
for ADJACENT Gray-code switch states via rank-1 updates of an
existing factor. That requires sparse LU factor-update support
that KLU doesn't expose. We pivoted to **lazy + multi-dt**
caching, which achieves the same "factor reuse" goal from a
different angle:
- Lazy build: skip factors for unvisited masks.
- Multi-dt: cache factors at additional dt values so sub-step
  partial steps don't repeat builds.

## Verified

- `solve_at(mask, primary_dt, ...)` matches `solve(mask, ...)`
  bit-identical.
- A dt change on a known mask is an in-place refactor
  (`metrics().event_refactors`), not a new entry; same `(mask,
  dt)` is pure factor reuse (`event_hits`).
- The commutation pattern (pre/post masks at fresh dts every
  event) holds steady at 2 entries with zero growth.
- Entries are LRU-bounded at `kMaxEventEntries`; evicted masks
  rebuild transparently and match a fresh cache at 1e-12.
- The refactor path matches a from-scratch build at the same dt
  on a dynamic RLC circuit (1e-12).
- A numerically singular refactorize ERASES the entry (a stale
  entry would poison later solves at the previously working dt).
- `refactor_parametric` invalidates entries; post-update
  `solve_at` agrees with a fresh cache built from the updated
  pool.

## Sub-step state correction (future)

Once `solve_at` exists, `run_transient` can:

1. Detect a commutation between t_n and t_n+1 via the V2.2
   interpolation.
2. Find t* exactly.
3. Solve at `dt_partial = t* - t_n` with the OLD mask:
   `cache.solve_at(old_mask, dt_partial, b_extra_partial,
   x_at_tstar)`.
4. Switch to new_mask at t*.
5. Solve at `dt_remaining = t_n+1 - t*` with the new mask:
   `cache.solve_at(new_mask, dt_remaining, b_extra_remaining,
   x_at_t_n+1)`.

Implementing this in `run_transient` (with proper history-term
plumbing for the partial steps) is a sibling OpenSpec to V7.
This OpenSpec ships just the cache-level primitive.

## Memory considerations

For a PWM converter where partial-dt values recur each cycle,
the auxiliary cache stays bounded:
- M visited masks × K distinct partial-dt values × ~10 KB per
  segment ≈ M·K·10 KB.
- For typical M = 4, K = 10: 400 KB. Acceptable.

v2.0 Phase 1: the pathological case is gone by construction —
memory is `min(visited_masks, 8) × (split + one factor)`,
independent of how many distinct dt values the run produces. The
primary lazy segment cache is separately bounded by a byte budget
(`set_segment_budget_bytes`, default 1 GiB).

## Status

| Layer | Cases | Assertions |
|---|---|---|
| 0 | 19 | 80 |
| 1 | 36 | 126 |
| 2 | 36 | 93 |
| 3 | 16 | 61 |
| 4 V0 | 24 | 58 |
| 5 V0 | 21 | 2069 |
| **4 V1 + V6 + V7** | **40** | **103** ← +3 / +13 multi-dt |
| 5 V1 | 17 | 59 |
| 5 V2.2 | 20 | 46 |
| 4 V2 | 9 | 520 |
| 4 V3 | 5 | 13 |
| 5 V4 + V5 | 7 | 66 |
| **Total** | **250** | **3294** |
