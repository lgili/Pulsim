## ADDED Requirements

### Requirement: Multi-Column Path Union for partial_refactor

The solver SHALL accept arbitrary-length `changed_cols` in
`partial_refactor(new_M, changed_cols)` and SHALL compute the
union of the elimination-tree paths of all changed columns
before re-eliminating. The implementation MUST
deduplicate path columns (via the existing `in_path` bitmap
mechanism), MUST process path columns in ascending permuted-
column order, and MUST satisfy the same numerical-correctness
contract as the single-column case (solve parity vs fresh-
factorise within $10^{-10}$).

The per-column pivot-threshold check from v1.3.0
(`PIVOT_THRESH = 10^{-3}`) SHALL continue to apply unchanged on
each path column. If the check fails for any path column,
`partial_refactor` SHALL return false and the caller SHALL fall
back to full `factorize()`.

#### Scenario: Two-column changed_cols produces correct solve
- **GIVEN** a factorised matrix $M_1$ and a new matrix $M_2$
  differing from $M_1$ at exactly columns $c_1$ and $c_2$
- **WHEN** `partial_refactor(M_2, {c_1, c_2})` is called and
  returns `true`
- **AND** `solve(b, x_partial)` is called against the updated
  factor
- **AND** a fresh `factorize(M_2)` + `solve(b, x_fresh)` provides
  a reference
- **THEN** $\|x_{\mathrm{partial}} - x_{\mathrm{fresh}}\|_\infty
  \le 10^{-10}$
- **AND** the number of L+U entries re-computed equals the
  union of the etree paths of $c_1$ and $c_2$

#### Scenario: Three-column changed_cols path is monotone
- **GIVEN** the same matrix factorised
- **WHEN** `partial_refactor_count_path()` is called with
  `changed_cols = {c}` returning $L_1$
- **AND** then with `changed_cols = {c, c_2}` returning $L_2$
- **THEN** $L_2 \ge L_1$ (the union path is at least as long as
  the single-column path)
- **AND** $L_2 \le L_1 + L_{\mathrm{path}(c_2)}$ (path-union
  doesn't exceed sum of individual paths)

### Requirement: Fallback Gate via Path-Length Ratio

`PulsimSparseLuSolver` SHALL expose a query method
`partial_refactor_count_path(changed_cols)` returning the length
of the union path that `partial_refactor(new_M, changed_cols)`
would walk, **without executing the refactor**. Callers SHALL
use this to decide between path-based update and full
factorise via the `MAX_PATH_LENGTH_RATIO` constant (default
$0.6$, compile-time tunable).

When `path_length / n > MAX_PATH_LENGTH_RATIO`, the path-based
update would cost approximately the same as a fresh factorise,
so the caller SHALL prefer the fresh factorise to keep the
solver's numerical state simpler (no risk of accumulated
floating-point drift from multiple updates).

#### Scenario: Short path takes the fast path
- **GIVEN** a factorised matrix and `changed_cols = {c}` whose
  union-path length is $L < 0.6 \cdot n$
- **WHEN** `partial_refactor_count_path({c})` is called
- **THEN** the returned length is $L$
- **AND** the caller is signalled (by the ratio comparison) to
  proceed with `partial_refactor(new_M, {c})`

#### Scenario: Long path triggers fallback
- **GIVEN** a factorised matrix and `changed_cols = {c_1, c_2,
  c_3, c_4}` whose union-path length is $L > 0.6 \cdot n$
- **WHEN** `partial_refactor_count_path({c_1, c_2, c_3, c_4})`
  is called
- **THEN** the returned length is $L$
- **AND** the caller is signalled to call `factorize(new_M)`
  instead

## MODIFIED Requirements

### Requirement: Path-Based Partial Refactorisation

The solver SHALL implement path-based partial refactorisation in
`partial_refactor(new_M, changed_cols)` following
Chan/Brandwajn/Tinney 1986 + Dinkelbach 2021. For each column
$c$ in `changed_cols`, the method SHALL walk the elimination-tree
parent chain from $c$ to its root, deduplicate across all
changed-column paths via an `in_path` bitmap, and re-eliminate
the resulting union of columns in ascending permuted order.

The L+U values along the path SHALL be updated in-place at the
existing CSC storage slots (no re-allocation; pattern is
assumed unchanged — sparsity-pattern invariance under value
changes per the SMPS MNA stamping rules).

A KLU-style threshold pivot check (`PIVOT_THRESH = 10^{-3}`)
SHALL apply to each path column. On pivot fault for any path
column, the method SHALL invalidate the path cache (via
`invalidate_path_cache_()`) and SHALL return `false`. Callers
that receive `false` SHALL fall back to full `factorize()`.

A lazy `varying_set_` SHALL accumulate every `changed_cols`
column ever seen across the lifetime of the solver. The
`compute_path_()` step SHALL only re-run when `varying_set_`
grows or when `path_valid_` is `false`. Repeated calls with the
same `changed_cols` SHALL reuse the cached path.

The method SHALL handle `changed_cols.empty()` as a no-op
returning `true` (no path computation, no L+U update).

#### Scenario: Single-column flip preserves v1.3.0 behaviour
- **GIVEN** a factorised matrix and `changed_cols = {c}`
  representing a single switch-bit flip
- **WHEN** `partial_refactor(new_M, {c})` is called
- **THEN** the returned status is `true` (assuming the
  pivot-threshold check passes — empirically true with zero
  fallbacks across the captured rank1 microbench)
- **AND** subsequent `solve(b, x)` matches fresh-factorise
  within $10^{-12}$
- **AND** `rank1_hits` increments by 1

#### Scenario: Multi-column changed_cols handled via path union
- **GIVEN** a factorised matrix and `changed_cols = {c_1, c_2}`
  where $|\mathrm{path}(c_1) \cup \mathrm{path}(c_2)| / n \le 0.6$
- **WHEN** `partial_refactor(new_M, {c_1, c_2})` is called
- **THEN** the returned status is `true`
- **AND** the union of `path(c_1)` and `path(c_2)` columns is
  re-eliminated exactly once each (no double-work via the
  `in_path` bitmap)
- **AND** subsequent `solve(b, x)` matches fresh-factorise
  within $10^{-10}$
- **AND** the multi-bit hit counter (per Part A of the
  proposal) increments by 1

#### Scenario: Pivot fault propagates fallback signal
- **GIVEN** a factorised matrix and `changed_cols = {c}` where
  the resulting new column produces $|x[k]| <
  \mathrm{PIVOT\_THRESH} \cdot \max_{i \ge k}|x[i]|$ for some
  path column $k$
- **WHEN** `partial_refactor(new_M, {c})` is called
- **THEN** the returned status is `false`
- **AND** `path_valid_` becomes `false`
- **AND** `varying_set_` is cleared
- **AND** the caller is expected to call `factorize(new_M)` to
  rebuild
