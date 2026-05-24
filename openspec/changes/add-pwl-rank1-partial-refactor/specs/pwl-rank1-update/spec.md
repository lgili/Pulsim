## MODIFIED Requirements

### Requirement: Single-Bit Gray-Code Partial Refactorization

The `PwlStateSpaceCache::solve_rank1` method SHALL invoke
`DirectSolver::partial_refactor` to update the cached LU factor
when (a) the new `mask` differs from the previously-solved mask by
exactly one bit AND (b) the underlying solver returns `true` from
`supports_partial_refactor()`. For `KluSolver`, `partial_refactor`
SHALL use path-based partial refactorisation (Chan, Brandwajn &
Tinney, *IEEE Trans. PAS* PAS-104, 1985; Dinkelbach et al.,
*Energies* 14:7989, 2021) — re-eliminating only the columns of the
LU factor that depend on the changed entries via the elimination
tree — using the vendored DPsim KLU fork's
`klu_compute_path` + `klu_partial_factorization_path` primitives.
The output `x` MUST match the result of the existing full `solve`
path within tolerance 1e-12; the L and U factor values MUST match
a fresh full `factorize` within tolerance 1e-14 when the pivot
order is preserved.

#### Scenario: Single-bit flip with KLU backend uses path-based refactor

- **GIVEN** the cache was built with a `KluSolver` backend
- **AND** the most recent `solve_rank1` was for `mask_prev`
- **WHEN** `solve_rank1(mask_curr, b_extra, x)` is called with
  `popcount(mask_prev XOR mask_curr) == 1`
- **THEN** the cache invokes `partial_refactor` on the changed
  columns only
- **AND** the underlying KLU operation is `klu_partial_factorization_path`
  (NOT `klu_refactor` — the V0 MVP behaviour)
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

## ADDED Requirements

### Requirement: Pivot-Fault Fallback in Path-Based Partial Refactor

`KluSolver::partial_refactor` SHALL handle the `KLU_PIVOT_FAULT`
status emitted by `klu_partial_factorization_path` (the
pivot-tolerance violation signal raised when the post-perturbation
pivot magnitude falls below `klu_common.pivot_tol_fail`) by
returning `false` to the caller AND invalidating the internal
path-cache. The caller's existing full-refactor fallback path then
re-establishes fresh pivots via `klu_factor`, and subsequent
`partial_refactor` calls re-compute paths against the fresh
factor. The `PwlStateSpaceCache::metrics().fallbacks` counter SHALL
increment in this case.

#### Scenario: Pivot fault triggers transparent fallback

- **GIVEN** a `KluSolver`-backed cache that has successfully
  `partial_refactor`ed at least once
- **WHEN** the next `solve_rank1` call's perturbation forces a pivot
  below `klu_common.pivot_tol_fail`
- **THEN** `KluSolver::partial_refactor` returns `false`
- **AND** `KluSolver` clears its internal path cache
- **AND** `PwlStateSpaceCache::solve_rank1` falls back to a full
  `factorize` call
- **AND** the `metrics().fallbacks` counter increments by 1
- **AND** the subsequent solve produces correct output (within 1e-12
  of an independent fresh-solver reference)

#### Scenario: Genuinely singular post-flip matrix surfaces as Singular

- **GIVEN** a `KluSolver`-backed cache
- **WHEN** a `solve_rank1` perturbation produces a
  numerically-singular matrix (zero pivot, not just below threshold)
- **THEN** `KluSolver::partial_refactor` returns `false` with
  `common_.status == KLU_SINGULAR`
- **AND** the subsequent full `factorize` ALSO returns false
  (genuinely degenerate input)
- **AND** `PwlStateSpaceCache::solve_rank1` propagates the failure
  as `std::runtime_error` per its existing contract

### Requirement: Path-Cache Reuse on Repeated Change-Sets

`KluSolver` SHALL maintain an internal cache mapping
`hash(sorted(changed_cols))` to the precomputed elimination path
arrays produced by `klu_compute_path`. Repeated `partial_refactor`
calls with the SAME `changed_cols` set MUST reuse the cached path
(no re-call of `klu_compute_path`). The cache is invalidated when
either (a) `analyze` is called (symbolic factor changes), or (b) a
`KLU_PIVOT_FAULT` is reported by `klu_partial_factorization_path`
(pivot order may have shifted).

#### Scenario: Repeated identical change-set hits the path cache

- **GIVEN** a `KluSolver` that has analyzed and factorized matrix M
- **WHEN** `partial_refactor(M', changed_cols)` is called 3 times in
  sequence with identical `changed_cols`
- **THEN** the first call calls `klu_compute_path` (cache miss)
- **AND** the second and third calls reuse the cached path (no
  `klu_compute_path` invocation)

#### Scenario: `analyze` invalidates the path cache

- **GIVEN** a `KluSolver` with a non-empty path cache (≥ 1
  successful `partial_refactor`)
- **WHEN** `analyze(M_new)` is called (e.g. due to topology change)
- **THEN** the path cache is cleared
- **AND** the next `partial_refactor` re-computes the path
