# pwl-rank1-update Specification

## Purpose
TBD - created by archiving change add-pwl-rank1-update. Update Purpose after archive.
## Requirements
### Requirement: Single-Bit Gray-Code Partial Refactorization

The `PwlStateSpaceCache::solve_rank1` method SHALL invoke
`DirectSolver::partial_refactor` to update the cached LU factor when
(a) the new `mask` differs from the previously-solved mask by
exactly one bit AND (b) the underlying solver returns `true` from
`supports_partial_refactor()`. The default backend
(`PulsimSparseLuSolver`) implements `partial_refactor` via
path-based re-elimination (Chan/Brandwajn/Tinney, *IEEE Trans.
Power Syst.* 1, 1986; Dinkelbach et al., *Energies* 14:7989, 2021).
The output `x` MUST match the result of the existing full `solve`
path within tolerance 1e-12, and the resulting L and U factor values
MUST match a fresh full `factorize` within tolerance 1e-14 when the
pivot order is preserved.

#### Scenario: Single-bit flip uses path-based partial refactor

- **GIVEN** the cache was built with a `PulsimSparseLuSolver` backend
- **AND** the most recent `solve_rank1` was for `mask_prev`
- **WHEN** `solve_rank1(mask_curr, b_extra, x)` is called with
  `popcount(mask_prev XOR mask_curr) == 1`
- **THEN** the cache invokes `partial_refactor` on the changed
  columns only
- **AND** the underlying operation is the path-based re-elimination
  (NOT a fresh full factorize)
- **AND** the `metrics().rank1_hits` counter increments by 1
- **AND** the output `x` matches the full-refactor result within
  tolerance 1e-12
- **AND** the resulting L and U values match a fresh full
  `factorize` within tolerance 1e-14 (pivot order preserved)

#### Scenario: Multi-bit flip falls back to full refactor

- **GIVEN** the most recent `solve_rank1` was for `mask_prev`
- **WHEN** `solve_rank1(mask_curr, b_extra, x)` is called with
  `popcount(mask_prev XOR mask_curr) >= 2`
- **THEN** the cache invokes the existing full `make_segment` /
  `solve` path
- **AND** the `metrics().full_refactor_hits` counter increments by 1

#### Scenario: First-encounter mask (no previous)

- **GIVEN** a freshly-constructed `PwlStateSpaceCache`
- **WHEN** `solve_rank1(mask, b_extra, x)` is called for the first
  time
- **THEN** the cache invokes the existing full path (no previous
  factor to update from)
- **AND** the `metrics().full_refactor_hits` counter increments by 1

### Requirement: Transparent Fallback on Backend Without Partial Refactor

The `PwlStateSpaceCache::solve_rank1` method SHALL fall back to the
existing full `solve` path when the underlying solver does not
support partial refactorisation (i.e.
`DirectSolver::supports_partial_refactor()` returns `false`). The
visible output of `solve_rank1` MUST match exactly what
`solve(mask, b_extra, x)` would produce in the same situation.

#### Scenario: SparseLuSolver backend always falls back

- **GIVEN** the cache was built with a `SparseLuSolver` backend
  (`Backend::Eigen` explicitly forced via factory)
- **WHEN** `solve_rank1(mask, b_extra, x)` is called for any mask
- **THEN** the cache delegates to the existing full `solve` path
- **AND** the `metrics().fallbacks` counter increments by 1
- **AND** the output is bit-identical to what `solve(mask, b_extra,
  x)` would produce

#### Scenario: Partial refactor reports pivot fault

- **GIVEN** the cache was built with a `PulsimSparseLuSolver` backend
- **AND** `partial_refactor` for a specific column change reports a
  pivot-tolerance violation (the precomputed path's pivot order is
  no longer valid)
- **WHEN** the cache encounters that case during `solve_rank1`
- **THEN** the cache silently falls back to the full `solve` path
- **AND** the `metrics().fallbacks` counter increments by 1
- **AND** no exception is propagated to the caller

### Requirement: Per-Cache Telemetry Counters

The `PwlStateSpaceCache::metrics()` accessor SHALL return a
`CacheMetrics` struct exposing three monotonic `uint64_t` counters
incremented on every `solve_rank1` call: `rank1_hits`,
`full_refactor_hits`, and `fallbacks`. The counters SHALL be
read-only from outside the cache (no public mutation API); the
cache MAY use `std::atomic<uint64_t>` internally for thread-safe
observation.

#### Scenario: Counters accumulate across solves

- **GIVEN** a fresh `PwlStateSpaceCache` built with a `KluSolver`
  backend
- **WHEN** the simulator issues 100 `solve_rank1` calls — 70 with
  single-bit Gray-code flips, 25 with multi-bit flips, 5
  first-encounters
- **THEN** `metrics().rank1_hits == 70`
- **AND** `metrics().full_refactor_hits == 30` (25 multi-bit + 5
  first-encounter)
- **AND** `metrics().fallbacks == 0` (KLU backend, no partial-refactor
  singularities)

#### Scenario: Fallback counter accumulates separately

- **GIVEN** a fresh `PwlStateSpaceCache` built with a `SparseLuSolver`
  backend
- **WHEN** the simulator issues 50 `solve_rank1` calls
- **THEN** `metrics().rank1_hits == 0`
- **AND** `metrics().full_refactor_hits == 0`
- **AND** `metrics().fallbacks == 50`

