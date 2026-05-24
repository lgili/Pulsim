# pwl-rank1-update Specification

## Purpose
TBD - created by archiving change add-pwl-rank1-update. Update Purpose after archive.
## Requirements
### Requirement: KLU Backend for Sparse Direct Solvers

Pulsim SHALL provide a `KluSolver` implementation of the
`pulsim::sparse::DirectSolver` interface, wrapping SuiteSparse KLU,
available behind a CMake feature detection flag `PULSIM_HAVE_KLU`.
When KLU is detected, `pulsim::sparse::make_default_solver(n, hint)`
SHALL return a `KluSolver` for `Backend::KLU` or for `Backend::Auto`
when `n >= PULSIM_KLU_AUTO_THRESHOLD` (default 100). When KLU is not
detected, the factory SHALL return a `SparseLuSolver` for any hint
except `Backend::KLU`, which MUST raise `std::runtime_error`.

#### Scenario: KLU available, used for large circuits

- **GIVEN** SuiteSparse KLU is detected at build time
- **AND** `make_default_solver(200, Backend::Auto)` is called
- **THEN** the returned solver is a `KluSolver` instance
- **AND** `analyze + factorize + solve` produces results matching
  `SparseLuSolver` within tolerance 1e-12

#### Scenario: KLU unavailable, transparent Eigen fallback

- **GIVEN** SuiteSparse KLU was NOT detected at build time
- **WHEN** `make_default_solver(200, Backend::Auto)` is called
- **THEN** the returned solver is a `SparseLuSolver` instance
- **AND** no exception is raised

#### Scenario: KLU explicitly requested but unavailable

- **GIVEN** SuiteSparse KLU was NOT detected at build time
- **WHEN** `make_default_solver(200, Backend::KLU)` is called with an
  explicit hint
- **THEN** `std::runtime_error` is raised with a message that names
  the missing dependency

### Requirement: Single-Bit Gray-Code Partial Refactorization

The `PwlStateSpaceCache::solve_rank1` method SHALL use path-based
partial refactorization (Chen et al., *IEEE Trans. Power Electron.*,
2024) to update the cached LU factor when (a) the new `mask` differs
from the previously-solved mask by exactly one bit AND (b) the
underlying solver returns `true` from `supports_partial_refactor()`.
The output `x` MUST match the result of the existing full `solve`
path within tolerance 1e-12.

#### Scenario: Single-bit flip with KLU backend

- **GIVEN** the cache was built with a `KluSolver` backend
- **AND** the most recent `solve_rank1` was for `mask_prev`
- **WHEN** `solve_rank1(mask_curr, b_extra, x)` is called with
  `popcount(mask_prev XOR mask_curr) == 1`
- **THEN** the cache invokes `partial_refactor` on the changed
  column only
- **AND** the `metrics().rank1_hits` counter increments by 1
- **AND** the output `x` matches the full-refactor result within
  tolerance 1e-12

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

The `pulsim::pwl::PwlStateSpaceCache::solve_rank1` method SHALL fall
back to the existing full `solve` path when the underlying solver
does not support partial refactorisation (i.e.
`DirectSolver::supports_partial_refactor()` returns `false`). The
visible output of `solve_rank1` MUST match exactly what
`solve(mask, b_extra, x)` would produce in the same situation.

#### Scenario: SparseLuSolver backend always falls back

- **GIVEN** the cache was built with a `SparseLuSolver` backend
  (KLU disabled at build OR explicitly forced via Backend::Eigen)
- **WHEN** `solve_rank1(mask, b_extra, x)` is called for any mask
- **THEN** the cache delegates to the existing full `solve` path
- **AND** the `metrics().fallbacks` counter increments by 1
- **AND** the output is bit-identical to what `solve(mask, b_extra,
  x)` would produce

#### Scenario: Partial refactor reports numerical singularity

- **GIVEN** the cache was built with a `KluSolver` backend
- **AND** `partial_refactor` for a specific column change returns
  `false` (numerical singularity detected mid-elimination)
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

