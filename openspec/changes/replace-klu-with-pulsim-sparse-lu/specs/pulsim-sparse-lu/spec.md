## ADDED Requirements

### Requirement: In-House Sparse LU Solver

Pulsim SHALL provide a `pulsim::sparse::PulsimSparseLuSolver` class
implementing the `pulsim::sparse::DirectSolver` interface, with NO
external sparse-LU dependency (neither SuiteSparse KLU nor
Eigen::SparseLU). The implementation SHALL use `Eigen::SparseMatrix`
only as a passive matrix container; ordering, symbolic analysis,
numeric factorization, triangular solve, and partial refactorization
SHALL all be implemented in C++23 from scratch in Pulsim's source.

#### Scenario: PulsimSparseLuSolver is the default for Backend::Auto

- **WHEN** `make_default_solver(n, Backend::Auto)` is called for any
  `n >= 1`
- **THEN** the returned solver is a `PulsimSparseLuSolver` instance
- **AND** `analyze + factorize + solve` produces results matching
  `Eigen::SparseLU` (via the existing `SparseLuSolver` reference)
  within tolerance 1e-12 on the SPD 3x3 + buck-like 8x8 unit-test
  fixtures

#### Scenario: Backend::Pulsim explicit hint

- **WHEN** `make_default_solver(n, Backend::Pulsim)` is called for
  any `n >= 1`
- **THEN** the returned solver is a `PulsimSparseLuSolver` instance

#### Scenario: Backend::Eigen explicit hint returns the reference

- **WHEN** `make_default_solver(n, Backend::Eigen)` is called
- **THEN** the returned solver is a `SparseLuSolver` instance
  (the Eigen-based reference, retained for parity testing)

### Requirement: Fill-Reducing Column Ordering

`PulsimSparseLuSolver::analyze(M)` SHALL apply a fill-reducing column
permutation to `M` before symbolic factorization. The MVP
implementation SHALL use Reverse Cuthill-McKee (RCM, George 1971).
Production-quality fill-reducing orderings (COLAMD, AMD) are
explicitly OUT OF SCOPE of this change and tracked as follow-ups.

#### Scenario: RCM ordering produces a non-trivial permutation

- **GIVEN** a buck-like 8x8 asymmetric MNA matrix
- **WHEN** `analyze(M)` is called
- **THEN** the internal column permutation differs from the identity
  permutation (`Pcol[k] != k` for at least one k)
- **AND** the fill-in count is within ±50% of `Eigen::SparseLU`'s
  COLAMD-based fill on the same matrix

### Requirement: Gilbert-Peierls Left-Looking Factorization with Partial Pivoting

`PulsimSparseLuSolver::factorize(M)` SHALL implement the
left-looking Gilbert-Peierls algorithm (Gilbert & Peierls, *SIAM J.
Sci. Stat. Comput.* 9, 1988) with column-by-column partial pivoting.
The factor SHALL satisfy `L * U == Prow · M · Pcol^T` within
tolerance 1e-12. The factorization SHALL detect numerical
singularity (zero pivot after partial pivoting) and return `false`
without throwing.

#### Scenario: Factorization identity holds

- **GIVEN** an SPD 3x3 or buck-like 8x8 reference matrix M
- **WHEN** `analyze(M)` then `factorize(M)` are called
- **THEN** assembling `L * U` and comparing against
  `Prow · M · Pcol^T` shows max element-wise error ≤ 1e-12

#### Scenario: Numerical singularity returns false

- **WHEN** `factorize(M)` is called on a matrix with a structurally
  singular column (e.g. all-zero column)
- **THEN** the method returns `false`
- **AND** does NOT throw

#### Scenario: Partial pivoting rescues a zero diagonal

- **GIVEN** a matrix M whose first diagonal entry is 0 but whose
  first column has a non-zero entry below the diagonal
- **WHEN** `factorize(M)` is called
- **THEN** partial pivoting swaps in the non-zero row
- **AND** factorization succeeds

### Requirement: Triangular Solve

`PulsimSparseLuSolver::solve(b, x)` SHALL perform forward
substitution on L then back substitution on U, applying the row and
column permutations from `factorize`. The result x MUST match
`Eigen::SparseLU`'s solution (via `SparseLuSolver`) within tolerance
1e-12 on the SPD 3x3 + buck-like 8x8 fixtures.

#### Scenario: Solve matches Eigen reference

- **GIVEN** a matrix M and RHS vector b
- **WHEN** `analyze(M) + factorize(M) + solve(b, x)` is called on
  `PulsimSparseLuSolver`
- **AND** the same sequence is run on a reference `SparseLuSolver`
- **THEN** both `x` vectors match element-wise within tolerance 1e-12

#### Scenario: Solve before factorize throws

- **WHEN** `solve(b, x)` is called on a freshly-constructed
  `PulsimSparseLuSolver` (before `factorize`)
- **THEN** `std::logic_error` is raised with a message naming the
  missing prerequisite

### Requirement: Path-Based Partial Refactor with Pivot-Fault Detection

`PulsimSparseLuSolver::partial_refactor(new_M, changed_cols)` SHALL
implement path-based partial refactorization on top of the cached
elimination tree (Dinkelbach et al., *Energies* 14:7989, 2021, §3).
The method SHALL maintain a lazy union of all changed column indices
seen since the last `analyze()` and re-run path computation only
when the union grows. When a perturbation forces a would-be pivot
below `pivot_tol_fail` (default 0.1 × column max magnitude), the
method SHALL invalidate the path cache and return `false`. The
resulting L and U values MUST match a fresh full `factorize(new_M)`
within tolerance 1e-14 when no pivot fault occurs.

#### Scenario: Single-bit perturbation produces bit-identical L+U vs full factor

- **GIVEN** a factored matrix M1 (analyze + factorize succeeded)
- **WHEN** `partial_refactor(M2, changed_cols)` is called with M2
  differing from M1 only in the values of `changed_cols`'s columns
- **AND** `factorize(M2)` is called on an independent reference
  solver
- **THEN** the two L matrices match element-wise within tolerance 1e-14
- **AND** the two U matrices match element-wise within tolerance 1e-14

#### Scenario: Repeated identical change-set reuses the cached path

- **GIVEN** a `PulsimSparseLuSolver` that has analyzed + factorized
- **WHEN** `partial_refactor(M', changed_cols)` is called 3 times in
  sequence with identical `changed_cols`
- **THEN** the first call computes the path (cache miss)
- **AND** the second and third calls reuse the cached path
  (verifiable via the `path_compute_count()` diagnostic accessor)

#### Scenario: Pivot fault returns false and invalidates cache

- **GIVEN** a `PulsimSparseLuSolver` with a non-empty path cache
- **WHEN** `partial_refactor(M', changed_cols)` is called with a
  perturbation that forces a pivot below `pivot_tol_fail`
- **THEN** the method returns `false`
- **AND** the internal lazy-union of varying columns is cleared
- **AND** a subsequent `factorize(M')` succeeds normally

#### Scenario: `analyze` invalidates the path cache

- **GIVEN** a `PulsimSparseLuSolver` with a non-empty path cache
  (≥ 1 successful `partial_refactor`)
- **WHEN** `analyze(M_new)` is called (e.g. due to topology change)
- **THEN** the path cache is cleared
- **AND** the next `partial_refactor` re-computes the path
