## ADDED Requirements

### Requirement: Byte-Budgeted Event Solver Table

The table of event solvers used by `PwlStateSpaceCache::solve_at` SHALL
be bounded by a byte budget, not by a fixed number of entries. The
budget SHALL be the one set by `set_segment_budget_bytes`, applied to
the event table on its own account. When an insert would exceed the
budget, least-recently-used entries SHALL be evicted, after every step
of the insert that can throw; the incoming entry SHALL always be
inserted. `CacheMetrics` SHALL report `event_evictions`. Which entries
are resident SHALL NOT change any solution.

#### Scenario: Every mask is built once
- **GIVEN** the three-phase VSI reference run on `trbdf2` at the default
  budget
- **WHEN** the run completes
- **THEN** `event_builds` equals the number of distinct masks solved
  and `event_evictions` is 0

#### Scenario: Tight budget evicts without changing the result
- **GIVEN** the same run with a budget that holds two entries
- **WHEN** the run completes
- **THEN** `event_evictions` is greater than 0
- **AND** states and sample times are `np.array_equal` to the
  default-budget run

### Requirement: Allocation-Free Event Refactorisation

A change of `dt` on a resident event entry SHALL recombine
`J = G + (1/dt)·C` in place and refactorise without heap allocation,
once that entry has been refactorised before. The recombined values
SHALL be byte-equal to the sum formed as a new matrix. The in-house LU
SHALL keep its factorisation work vectors between calls.

#### Scenario: Steady-state refactor allocates nothing
- **GIVEN** an event entry that has been refactorised at two `dt` values
- **WHEN** it is refactorised at a third `dt` under the allocation
  counters of `test_zero_alloc`
- **THEN** 0 heap allocations are observed

#### Scenario: Same matrix as before
- **WHEN** an entry is recombined in place at 100 different `dt` values
- **THEN** each result has the index arrays and the bytes of
  `G + (1/dt)·C`

### Requirement: Per-Mask Step-Size Factor Slots

An event entry SHALL keep several factors of the same mask, each keyed
by the exact `dt` it was factorised at, and one scratch factor. A solve
marked on-grid whose `dt` equals the key of a resident factor SHALL use
that factor without factorising. A solve marked off-grid SHALL use the
scratch factor only and SHALL NOT evict an on-grid factor. `CacheMetrics`
SHALL report `event_slot_evictions` and `event_analyses`.

#### Scenario: A step size used before is a hit
- **GIVEN** a mask solved on-grid at `dt_a`, then at `dt_b`
- **WHEN** it is solved on-grid at `dt_a` again
- **THEN** `event_hits` increases by 1 and `event_refactors` does not
  change

#### Scenario: A landing step does not evict
- **GIVEN** a mask whose four on-grid factors are resident
- **WHEN** it is solved off-grid at ten different `dt` values
- **THEN** the four on-grid factors are still resident and
  `event_slot_evictions` is unchanged

#### Scenario: Structural bound on factorisations
- **WHEN** any `trbdf2` run completes
- **THEN** `event_builds + event_refactors` is at most the number of
  off-grid solves plus the number of distinct on-grid `(mask, dt)` pairs
  plus `event_slot_evictions`
