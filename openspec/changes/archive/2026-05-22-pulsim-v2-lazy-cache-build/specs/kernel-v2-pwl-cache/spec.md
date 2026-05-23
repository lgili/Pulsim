## ADDED Requirements

### Requirement: PwlStateSpaceCache — build_lazy() method

`PwlStateSpaceCache` SHALL expose a `build_lazy(Real dt)`
method that stores the dt but does NOT factorise any segment
upfront. Subsequent calls to `solve(mask, ...)` MUST build the
factor for `mask` on demand (and cache it) the first time each
unique mask is requested.

`build_lazy(dt)` SHALL clear any segments built by a previous
`build(dt)` (eager) call, and `build(dt)` SHALL likewise clear
any lazy-built segments.

`num_built_segments() const` SHALL return the number of
segments currently in the cache:
- After `build(dt)` (eager): `2^N` immediately.
- After `build_lazy(dt)`: 0, then grows as masks are solved.

#### Scenario: build_lazy followed by solve builds on demand

- **GIVEN** a 1-switch graph and a `PwlStateSpaceCache`
- **WHEN** the user calls `cache.build_lazy(dt=1e-7)`
- **THEN** `num_built_segments()` SHALL be `0`
- **WHEN** the user calls `cache.solve(mask_off, ..., x)`
- **THEN** `num_built_segments()` SHALL become `1`
- **WHEN** the user calls `cache.solve(mask_on, ..., x)`
- **THEN** `num_built_segments()` SHALL become `2`
- **WHEN** the user calls `cache.solve(mask_off, ..., x)`
  AGAIN
- **THEN** `num_built_segments()` SHALL stay at `2` (cached).

#### Scenario: build() after build_lazy() clears + rebuilds

- **GIVEN** a `PwlStateSpaceCache` with some lazy-built
  segments
- **WHEN** the user calls `cache.build(dt)` (eager)
- **THEN** `num_built_segments()` SHALL equal `2^N` (all
  segments built fresh).

#### Scenario: build_lazy() after build() clears + makes lazy

- **GIVEN** a `PwlStateSpaceCache` after `cache.build(dt)`
- **WHEN** the user calls `cache.build_lazy(dt)`
- **THEN** `num_built_segments()` SHALL be `0`.
