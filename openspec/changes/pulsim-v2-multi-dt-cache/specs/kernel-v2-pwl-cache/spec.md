## ADDED Requirements

### Requirement: PwlStateSpaceCache — multi-dt support

`PwlStateSpaceCache` SHALL expose `solve_at(mask, dt, b_extra,
x) const` that uses a SEPARATE auxiliary cache keyed on the
requested dt. The auxiliary cache MUST be lazily populated: the
first call for each `(mask, dt)` pair builds the factor; later
calls reuse it.

When `dt == this->dt()` (the primary cache's dt),
`solve_at(mask, dt, b_extra, x)` MUST delegate to
`solve(mask, b_extra, x)` (use the primary cache).

`num_alt_dt_values() const` SHALL return the number of
DISTINCT auxiliary dt values that have at least one cached
segment. `num_alt_segments_at(dt) const` SHALL return the
number of segments cached at the given auxiliary dt (0 if
that dt has never been requested).

#### Scenario: solve_at with primary dt matches solve

- **GIVEN** a cache built at dt = 1e-6 (eager or lazy)
- **WHEN** the user calls `solve_at(mask, 1e-6, b_extra, x)`
- **THEN** the result SHALL be bit-identical to
  `solve(mask, b_extra, x)`.

#### Scenario: solve_at with new dt builds in auxiliary cache

- **GIVEN** a cache built at dt = 1e-6 with a 1-switch graph
- **AND** `num_alt_dt_values() == 0`
- **WHEN** the user calls `solve_at(mask, 5e-7, b_extra, x)`
- **THEN** the call SHALL build the (mask, 5e-7) segment in
  the auxiliary cache
- **AND** `num_alt_dt_values()` SHALL become `1`
- **AND** `num_alt_segments_at(5e-7)` SHALL be `1`.

#### Scenario: Auxiliary cache reuses on repeat calls

- **GIVEN** an auxiliary cache populated with one segment at
  `(mask_a, dt=5e-7)`
- **WHEN** the user calls `solve_at(mask_a, 5e-7, ..., x)`
  again
- **THEN** `num_alt_segments_at(5e-7)` SHALL stay at `1`
  (the segment was reused, not rebuilt).

#### Scenario: Different dt values produce different solutions

- **GIVEN** a V-R-C circuit (capacitor with `g_eq = 2C/dt`
  in the matrix)
- **WHEN** the user calls `solve_at(mask, 1e-6, b, x_1)` and
  `solve_at(mask, 5e-7, b, x_2)`
- **THEN** `x_1 != x_2` (the trap companion's effective
  impedance differs between the two dt values).
