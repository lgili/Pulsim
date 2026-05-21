## Why

Layer 5 V2.2 added sub-step COMMUTATION TIMING via linear
interpolation, but the recorded state vectors are still at the
dt grid because the cache stores factors only at ONE dt value.
True sub-step state correction would need partial-step solves
at dt_partial < opts.dt — and that requires the cache to
factor matrices at multiple dt values (trap-companion's
`g_eq = 2C/dt` is dt-specific).

This OpenSpec ships **multi-dt cache support** as a PRIMITIVE:
`solve_at(mask, dt, b_extra, x)` builds-and-caches a factor at
the requested `(mask, dt)` pair on demand. The single-dt API
(`solve(mask, b_extra, x)`) stays as the default fast path.

True sub-step STATE correction in `run_transient` is a follow-
up — this OpenSpec just ships the primitive that enables it.

## What Changes

**Scope decision — Layer 4 V7** (multi-dt cache):

- Add `solve_at(mask, dt, b_extra, x)` method on
  `PwlStateSpaceCache`:
  - If `dt == this->dt()` (the build dt): use the primary
    `segments_` map (build if lazy + missing).
  - Else: use a secondary `alt_segments_[dt]` map, building
    the missing segment on demand. Each unique `(mask, dt)`
    pair is built once and cached.

- Add `num_built_dt_values()` and
  `num_built_segments_at(dt)` accessors for diagnostics.

- The single-dt `solve(mask, b_extra, x)` is unchanged
  (uses primary `segments_` only).

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-pwl-cache`.
- **Affected code** (~80 LOC):
  - MODIFIED `pwl/cache.hpp`: add `solve_at`, `alt_segments_`,
    diagnostics.
  - NEW `tests/v2/layer4_v1/test_multi_dt_cache.cpp` (~150 LOC).
- **Migration**: zero. All existing single-dt callers are
  bit-identical.
- **Risk**: low. The multi-dt path is purely additive; primary
  segments unaffected.
