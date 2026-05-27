## MODIFIED Requirements

### Requirement: Single-Bit Gray-Code Partial Refactorization

The `PwlStateSpaceCache::solve_rank1` method SHALL use path-based
partial refactorisation (Chan/Brandwajn/Tinney 1986; Dinkelbach
et al., *Energies* 14:7989, 2021; Chen et al., *IEEE Trans. Power
Electron.*, 2024) to update the cached LU factor when:

1. The new `mask` differs from the previously-solved mask by
   exactly one bit (Hamming distance 1), AND
2. The underlying solver returns `true` from
   `supports_partial_refactor()`.

The output `x` MUST match the result of the existing full
`solve` path within tolerance $10^{-12}$. On success, the
`metrics().single_bit_rank1_hits` counter (renamed from
`rank1_hits` per Part A.3 of `add-generalised-path-refactor`,
with backward-compat alias) SHALL increment by 1.

#### Scenario: Single-bit flip with PulsimSparseLuSolver backend
- **GIVEN** the cache was built with a `PulsimSparseLuSolver` backend
- **AND** the most recent `solve_rank1` was for `mask_prev`
- **WHEN** `solve_rank1(mask_curr, b_extra, x)` is called with
  `popcount(mask_prev XOR mask_curr) == 1`
- **THEN** the cache invokes `partial_refactor` on the changed
  column only
- **AND** the `metrics().single_bit_rank1_hits` counter
  increments by 1
- **AND** the output `x` matches the full-refactor result within
  tolerance $10^{-12}$

#### Scenario: First-encounter mask (no previous)
- **GIVEN** a freshly-constructed `PwlStateSpaceCache`
- **WHEN** `solve_rank1(mask, b_extra, x)` is called for the first
  time
- **THEN** the cache invokes the existing full path (no previous
  factor to update from)
- **AND** the `metrics().full_refactor_hits` counter increments by 1

## ADDED Requirements

### Requirement: Multi-Bit Transition via Path Union

The `PwlStateSpaceCache::solve_rank1` method SHALL handle
multi-bit switch transitions (Hamming distance ≥ 2 between
`mask_prev` and `mask_curr`) via the **union of the etree paths
of the affected columns**, NOT by unconditional fallback to
full `factorize()` as in v1.3.0.

The routing rule SHALL be:

1. Compute `delta_bits = popcount(mask_prev XOR mask_curr)`.
2. Identify `changed_cols` via
   `DevicePool::columns_affected_by_switch(sw_id)` for each
   toggled switch bit.
3. Query `solver.partial_refactor_count_path(changed_cols)` to
   get the union-path length $L$.
4. If $L / n \le \mathrm{MAX\_PATH\_LENGTH\_RATIO}$
   (default $0.6$), call
   `solver.partial_refactor(new_J, changed_cols)`. On success,
   `metrics().multi_bit_rank1_hits` SHALL increment by 1. On
   failure, fall back to full `factorize()` and
   `metrics().fallbacks` SHALL increment by 1.
5. If $L / n > \mathrm{MAX\_PATH\_LENGTH\_RATIO}$, call
   `solver.factorize(new_J)` directly (no point trying path-
   based when the union path covers most of the matrix), and
   `metrics().full_refactor_hits` SHALL increment by 1.

The numerical-correctness contract is the same as the single-bit
case: solve output MUST match fresh-factorise within $10^{-10}$.

#### Scenario: 2-bit transition routed via path union
- **GIVEN** a cache built with `PulsimSparseLuSolver`
- **AND** the most recent solve was for `mask_prev = 0b00`
- **WHEN** `solve_rank1(mask_curr = 0b11, ...)` is called
  (Hamming distance 2)
- **AND** the union path for the 2 affected columns has length
  $\le 0.6 \cdot n$
- **THEN** the cache invokes `partial_refactor(new_J, {c_1,
  c_2})`
- **AND** `metrics().multi_bit_rank1_hits` increments by 1
- **AND** output `x` matches fresh-factorise within $10^{-10}$

#### Scenario: 4-bit transition with wide path falls back
- **GIVEN** a cache built with `PulsimSparseLuSolver` on an
  $n = 20$ matrix
- **AND** the most recent solve was for `mask_prev`
- **WHEN** `solve_rank1(mask_curr, ...)` is called with
  `delta_bits = 4` and union-path length $L = 14 > 0.6 \cdot 20 = 12$
- **THEN** the cache invokes `solver.factorize(new_J)` directly
  (no path-based call)
- **AND** `metrics().full_refactor_hits` increments by 1
- **AND** output `x` matches the legacy v1.3.0 behaviour
  exactly

#### Scenario: Telemetry invariant holds
- **GIVEN** a cache after $N$ `solve_rank1` calls
- **THEN** `metrics().single_bit_rank1_hits +
  metrics().multi_bit_rank1_hits +
  metrics().full_refactor_hits +
  metrics().fallbacks == N`
- **AND** all four counters are monotonic across calls

### Requirement: Parametric Refactor for Sweep / Monte Carlo Workloads

The cache SHALL expose `refactor_parametric(param_names,
new_values, mode)` that updates every active mask's cached
factor when the user changes one or more physical parameters
(resistor value, inductor value, capacitor value, etc.) without
changing the graph topology.

The method SHALL:

1. Update the parameter(s) via `pool.update_param(name, value)`
   for each `(name, value)` pair.
2. Build `affected_cols` as the union of
   `pool.columns_affected_by_param(name)` for each `name`.
3. For each `(mask, segment)` in `segments_` (per `mode`):
   - Re-stamp $J$ at `affected_cols` only.
   - If `|affected_cols| / n > MAX_PATH_LENGTH_RATIO`, call
     `segment.solver.factorize(new_J)` and increment
     `result.fallback_hits`.
   - Otherwise call `segment.solver.partial_refactor(new_J,
     affected_cols)` and increment `result.path_refactor_hits`
     (or `fallback_hits` on `false` return).
4. Return a `ParametricRefactorResult` with the per-mask
   success/fallback counts and wall-time spent.

`Mode::AllActive` (default) processes every cached segment.
`Mode::CurrentOnly` processes just the most-recent segment.

If `pool.columns_affected_by_param(name)` returns empty for any
name (parameter not in the pool, or topology change rather
than value change), the method SHALL fail-soft: that parameter
is silently routed to a full re-stamp + factorize for every
active mask, and `result.fallback_hits` SHALL include those.

#### Scenario: Single-parameter sweep gives correct results
- **GIVEN** a buck cache built with `dt = 100\,\mathrm{ns}` and
  visited masks `{0, 1}`
- **AND** a parameter `L_out` registered in the pool with
  current value $100\,\mu\mathrm{H}$
- **WHEN** `cache.refactor_parametric(["L_out"], [101e-6])` is
  called
- **THEN** the returned `ParametricRefactorResult` has
  `masks_processed == 2`
- **AND** `path_refactor_hits + fallback_hits == 2`
- **AND** subsequent `cache.solve(mask=0, ...)` and
  `cache.solve(mask=1, ...)` match the result of a fresh
  `PwlStateSpaceCache` built with the updated $L_{\mathrm{out}}$
  within tolerance $10^{-10}$ on every step

#### Scenario: Mode::CurrentOnly processes one segment
- **GIVEN** a cache with 3 active masks
- **AND** the most recent `solve(...)` call was for mask 2
- **WHEN** `refactor_parametric(["R"], [1.5], Mode::CurrentOnly)`
  is called
- **THEN** only the segment for mask 2 is updated
- **AND** `result.masks_processed == 1`
- **AND** `cache.solve(mask = 0, ...)` would still reflect the
  OLD value of `R` until that segment is also refactored

#### Scenario: Unknown parameter triggers per-mask fallback
- **GIVEN** a cache with 2 active masks
- **AND** a parameter name `"unknown_param"` that the pool does
  not recognise
- **WHEN** `refactor_parametric(["unknown_param"], [42.0])` is
  called
- **THEN** the returned `ParametricRefactorResult` has
  `fallback_hits == 2`
- **AND** `path_refactor_hits == 0`
- **AND** subsequent solves produce numerically valid output
  (full re-factorise was performed under the hood)
