## ADDED Requirements

### Requirement: First Visit Of A Neighbouring Mask Through Partial Refactor

A lazy cache SHALL, when the option is enabled, build the factor of a
mask it has not visited from the factor of the nearest resident mask:
it SHALL copy that factor and apply `partial_refactor` to the columns
of the switches that differ. It SHALL fall back to a numeric
factorisation on the shared symbolic analysis when no resident mask
exists, when the elimination path exceeds the path-length gate, or
when `partial_refactor` reports a pivot fault. The resident mask's own
factor SHALL NOT be modified. `CacheMetrics` SHALL report
`flip_partial_refactors` and `flip_fallbacks`. With the option disabled
every first visit SHALL be a numeric factorisation and results SHALL be
bit-equal to those of a cache without this feature.

#### Scenario: Single-switch flip
- **GIVEN** a lazy cache with the option enabled and mask `A` resident
- **WHEN** mask `B`, which differs from `A` in one switch, is solved
  for the first time
- **THEN** `flip_partial_refactors` increases by 1
- **AND** the solution matches a fresh factorisation of `B` within
  1e-12
- **AND** a later solve of `A` returns the same bits as before

#### Scenario: Pivot fault falls back
- **GIVEN** a flip for which `partial_refactor` returns `false`
- **WHEN** the mask is solved
- **THEN** it is factorised numerically, `flip_fallbacks` increases by
  1 and no exception reaches the caller

#### Scenario: Revisit is a plain hit
- **GIVEN** masks `A` and `B` both resident
- **WHEN** the run alternates between them 1000 times
- **THEN** `flip_partial_refactors` and `flip_fallbacks` do not change

#### Scenario: Option off is bit-identical
- **WHEN** the reference golden circuits are run with the option
  disabled
- **THEN** states are `np.array_equal` to the goldens

### Requirement: Rank-1 Solve Without Structural Reassembly

`PwlStateSpaceCache::solve_rank1` SHALL obtain the matrix of the
requested mask through the cache's assembly structure and SHALL NOT
build sparse structure on any call after the cache's first assembly.
Its output SHALL be unchanged.

#### Scenario: No structure built per call
- **GIVEN** a cache whose assembly structure exists
- **WHEN** `solve_rank1` is called 1000 times over a Gray-code sequence
  of masks
- **THEN** `structure_builds` is unchanged
- **AND** each output equals the output of the previous implementation
  for the same call sequence
