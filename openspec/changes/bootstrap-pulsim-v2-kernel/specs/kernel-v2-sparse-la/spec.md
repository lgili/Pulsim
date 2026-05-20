## ADDED Requirements

### Requirement: Sparse Matrix Wrapper with Cache-Optimal Layout

`pulsim::v2::sparse::Matrix` SHALL be an alias for
`Eigen::SparseMatrix<Real, Eigen::ColMajor, Index>` — column-major
storage, `int32` indices. This layout MUST match what direct sparse
solvers (Eigen SparseLU, KLU, UMFPACK, MKL Pardiso) expect natively,
so the solver path never copies or transposes the matrix.

The wrapper SHALL provide:
- `pulsim::v2::sparse::Triplet` = `Eigen::Triplet<Real, Index>` for
  triplet-style assembly.
- `pulsim::v2::sparse::stamp_dense(Matrix& M, Index row, Index col,
  const DenseMatrix& block)` — adds a small dense block at
  `(row, col)`. Returns the count of entries added. Used by Layer 4
  to stamp state-space (A, B, C, D) blocks.
- `pulsim::v2::sparse::reserve_capacity(Matrix& M, Size nnz_estimate)`
  — wraps `Eigen::SparseMatrix::reserve` with a clearer name.
- `pulsim::v2::sparse::compress_in_place(Matrix& M)` — ensures the
  matrix is in compressed form before solver use.

No operator overloads beyond what Eigen already provides. The wrapper
exists for type clarity and for the small set of stamping utilities
listed above; it does not re-implement linear algebra.

#### Scenario: Triplet-assembled matrix preserves all entries

- **GIVEN** a 4×4 `pulsim::v2::sparse::Matrix` constructed from
  triplets `[(0,0,1), (1,1,2), (2,2,3), (3,3,4)]`
- **WHEN** the consumer queries `M.nonZeros()`
- **THEN** the result SHALL be `4`
- **AND** `M.coeff(i, i)` for `i ∈ {0,1,2,3}` SHALL return `i+1`.

#### Scenario: stamp_dense adds a small block at the requested offset

- **GIVEN** an empty 4×4 sparse matrix M
- **WHEN** `stamp_dense(M, 1, 1, B)` is called with B = 2×2 identity
- **THEN** `M.coeff(1, 1) == 1.0` AND `M.coeff(2, 2) == 1.0`
- **AND** `M.coeff(1, 2) == 0.0` AND `M.coeff(2, 1) == 0.0`
- **AND** the call SHALL return `4` (the number of entries the
  2×2 block could add — even if some are exactly zero, they count
  as added for triplet bookkeeping).

### Requirement: Direct Solver Lifecycle Contract

`pulsim::v2::sparse::DirectSolver` SHALL be an abstract base class
that separates the three phases of a direct sparse solve:

1. `bool analyze(const Matrix& M)` — symbolic factorization.
   Determines the sparsity pattern of L + U and the column
   permutation. Result depends ONLY on the matrix's sparsity
   pattern, NOT on its numeric values. Returns `false` if M is
   structurally singular (e.g. a row of all-zeros).
2. `bool factorize(const Matrix& M)` — numeric factorization.
   Computes L + U using the symbolic structure cached by
   `analyze`. Returns `false` if M is numerically singular
   (zero pivot encountered).
3. `void solve(const Vector& b, Vector& x) const` — triangular
   solve `L · U · x = b` using the cached factor.

The lifecycle is strictly ordered: `analyze → factorize → solve`.
Calls out of order MUST throw `std::logic_error` with a clear
message naming the missing prerequisite.

`analyze` runs ONCE per topology change. `factorize` runs ONCE per
matrix-value change. `solve` runs every step. This separation is
the foundation of the Layer 4 PWL state-space cache: for a stable
switch combination, the sparsity pattern AND the matrix values are
constant across many steps, so `analyze + factorize` run ONCE and
many `solve` calls reuse the cached factor.

#### Scenario: Out-of-order calls throw with a clear diagnostic

- **GIVEN** a freshly-constructed `pulsim::v2::sparse::SparseLuSolver`
- **WHEN** the consumer calls `solve(b, x)` before any `factorize`
- **THEN** the call SHALL throw `std::logic_error`
- **AND** the exception's `what()` SHALL contain "factorize" (so
  the user knows what step they missed).

#### Scenario: Multiple factorize calls reuse the analyze cache

- **GIVEN** a SparseLuSolver and a matrix M1 with sparsity pattern P
- **WHEN** the consumer calls `analyze(M1)` followed by
  `factorize(M1)`, then later `factorize(M2)` where M2 has the
  SAME sparsity pattern P but different numeric values
- **THEN** both `factorize` calls SHALL succeed
- **AND** `solve` after the second `factorize` SHALL produce the
  correct solution for M2 (not M1).

#### Scenario: Structurally singular matrix rejected at analyze stage

- **GIVEN** a 3×3 sparse matrix M with one row of all zeros
- **WHEN** the consumer calls `analyze(M)`
- **THEN** the call SHALL return `false`
- **AND** subsequent `factorize` SHALL NOT be called on the
  consumer side (the contract is "if analyze returns false, the
  caller stops; factorize on a structurally-singular matrix has
  undefined behaviour").

#### Scenario: Numerically singular matrix rejected at factorize stage

- **GIVEN** a 3×3 sparse matrix M with full structural rank but
  zero numeric determinant (e.g. two identical rows after
  symbolic permutation)
- **WHEN** the consumer calls `analyze(M)` (returns true), then
  `factorize(M)`
- **THEN** `factorize` SHALL return `false`
- **AND** subsequent `solve` calls SHALL NOT be made (analogous
  contract: caller stops on factorize failure).

### Requirement: SparseLuSolver Reference Implementation

`pulsim::v2::sparse::SparseLuSolver` SHALL implement
`pulsim::v2::sparse::DirectSolver` using
`Eigen::SparseLU<Eigen::SparseMatrix<Real, Eigen::ColMajor, Index>>`
as the underlying factorization. This is the reference
implementation that Layer 0 tests run against and that the default
factory returns.

Future direct-solver backends (KLU, UMFPACK, MKL Pardiso) MAY be
added through the same `DirectSolver` interface without modifying
any consumer. They are out of scope for this OpenSpec.

#### Scenario: SPD system solves to high precision

- **GIVEN** the SPD 3×3 system M = `[[4,-1,0],[-1,4,-1],[0,-1,4]]`
  and right-hand side b = `[2, 4, 2]`
- **WHEN** the consumer calls `analyze(M)`, `factorize(M)`, and
  `solve(b, x)` on a `SparseLuSolver`
- **THEN** the analytic solution `x = [0.7857, 1.1428, 0.7857]`
  (up to numerical precision) SHALL be recovered within absolute
  tolerance `1e-12`.

### Requirement: Default Solver Factory

`pulsim::v2::sparse::make_default_solver()` SHALL return a
`std::unique_ptr<DirectSolver>` to a default-constructed solver.
For Layer 0 the default backend is `SparseLuSolver`. The factory
exists so that consumers depend only on the abstract `DirectSolver`
interface; future layers can override the default through a runtime
hint without touching consumer code.

#### Scenario: Factory returns a non-null solver

- **WHEN** the consumer calls `pulsim::v2::sparse::make_default_solver()`
- **THEN** the returned `unique_ptr` SHALL be non-null
- **AND** the contained object SHALL be a `SparseLuSolver`
  (verifiable via `dynamic_cast`).
