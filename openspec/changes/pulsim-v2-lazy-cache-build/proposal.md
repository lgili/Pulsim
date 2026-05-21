## Why

`PwlStateSpaceCache::build(dt)` currently factorises ALL `2^N`
segments upfront. For circuits where the simulation visits only
a few switch states (typical of PWM converters in fixed-duty
operation), this wastes time and memory.

Concrete example: the boost converter test has 2 switches → 4
segments → 4 factorisations. With PWM, only 2 of those states
are EVER visited (the two complementary "Q on, D off" and
"Q off, D on" configurations). The "both on" and "both off"
states are built but never used.

For larger circuits — say a 3-phase inverter with 6 switches —
that's 64 segments, but in normal operation only ~12 are
ever used. Building all 64 is 5× the actual need.

**This OpenSpec adds lazy build-on-first-lookup**: factors
are built only when `cache.solve(mask, ...)` first asks for
them. Cached after that. The opt-in flag enables it; the
default behaviour (`build_eager`) is unchanged for
backwards compatibility.

Sherman-Morrison rank-1 updates (the original ask) require
sparse LU factor-update support that KLU doesn't expose.
Lazy building hits the same "wasted work" concern from a
different angle: instead of speeding up the factor, skip it
when not needed.

## What Changes

**Scope decision — Layer 4 V6** (lazy cache build):

- Add `build_lazy(Real dt)` method to `PwlStateSpaceCache`.
  - Stores dt; does NOT factorise any segment.
  - On `solve(mask, ...)`, checks if `segments_[mask]` exists;
    if not, builds it on-demand.
  - The first call for a new mask pays the build cost; later
    calls hit the cached factor.

- Keep `build(dt)` (eager build) as the default for
  predictable behaviour and bench compatibility.

- Add `num_built_segments()` const accessor (vs the existing
  `num_segments()` which would return the number of cached
  factors — same answer for eager, different for lazy until
  fully populated).

- New test: build_lazy on the boost converter, simulate the
  PWM cycle, verify only 2 segments end up built (out of 4
  possible).

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-pwl-cache`
  for `build_lazy` + on-demand factor build.
- **Affected code** (~80 LOC):
  - MODIFIED `pwl/cache.hpp`: add `build_lazy()` and modify
    `solve()` to handle missing segments by building on demand.
  - NEW `tests/v2/layer4_v1/test_lazy_cache.cpp` (~150 LOC).
- **Migration**: zero. Default `build(dt)` unchanged.
- **Risk**: low. The lazy path is purely additive; existing
  build/solve sequences are bit-identical.
