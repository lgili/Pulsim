## Phase 0 — Scaffolding (~0.25 days)

### 0.1 Directory layout + namespace skeleton
- [x] 0.1.1 Create `core/include/pulsim/v2/` root directory.
- [x] 0.1.2 Sub-directories per layer (only `numeric/` and `sparse/`
      land in this proposal; later layers stub-only):
      - `core/include/pulsim/v2/numeric/`
      - `core/include/pulsim/v2/sparse/`
- [x] 0.1.3 Add `core/include/pulsim/v2/README.md` linking to the
      layered-architecture design doc.
- [x] 0.1.4 Confirm `pulsim::v2` namespace does NOT collide with
      `pulsim::v1` symbols (compiler will catch this at link time).

### 0.2 CMake target + test wiring
- [x] 0.2.1 In `core/CMakeLists.txt`, add `pulsim_v2` interface
      library (header-only, mirrors `pulsim_core`). Alias as
      `pulsim::v2`.
- [x] 0.2.2 Add `pulsim_v2_layer0_tests` test executable in
      `core/CMakeLists.txt`. Sources: `tests/v2/layer0/test_main.cpp`
      + the Layer 0 unit test files below. Links: `pulsim::v2 +
      Catch2::Catch2WithMain`.
- [x] 0.2.3 Register via `catch_discover_tests(pulsim_v2_layer0_tests)`.

## Phase 1 — Numeric primitives (~0.5 days)

### 1.1 `numeric/types.hpp` — scalar + index types
- [x] 1.1.1 `pulsim::v2::Real` — fixed at `double` for now.
      Pre-declared as a compile-time-overridable alias (`#ifndef
      PULSIM_V2_REAL_TYPE` block) so a future `-DPULSIM_V2_REAL_TYPE=
      float` build can compile in single precision.
- [x] 1.1.2 `pulsim::v2::Index` — `std::int32_t`. Used for node IDs,
      branch IDs, matrix indices. Signed (negative sentinels for
      ground / "no such device"). 4-byte alignment is cache-friendly
      vs v1's `int` (typically 4 B but implementation-defined).
- [x] 1.1.3 `pulsim::v2::Size` — `std::size_t`. For container sizes,
      iteration counts.
- [x] 1.1.4 Compile-time sentinels: `pulsim::v2::kInvalidIndex = -1`,
      `pulsim::v2::kGround = -1` (mirrors v1's convention).

### 1.2 `numeric/dense.hpp` — dense vector / matrix wrappers
- [x] 1.2.1 `pulsim::v2::Vector` = `Eigen::VectorX<Real>`. Wrap as
      a type alias for header-only friendliness; Eigen is already a
      Pulsim dependency.
- [x] 1.2.2 `pulsim::v2::DenseMatrix` = `Eigen::MatrixX<Real>`. Used
      for small block-dense operations (state-space (A, B, C, D)
      matrices later).
- [x] 1.2.3 No new operations on top of Eigen — pure aliases. The
      goal is type-level clarity: when Layer 4 says it operates on a
      `pulsim::v2::Vector`, the reader knows precisely what it means.

### 1.3 `numeric/concepts.hpp` — generic numeric concepts
- [x] 1.3.1 `pulsim::v2::numeric::FloatingPoint` concept = `Real` or
      anything `std::floating_point`. Used to constrain templates
      that should accept both `double` and `ad::ADReal` (AD scalar
      from v1's `ad/ad_scalar.hpp` will be ported / re-implemented in
      Layer 2; Layer 0 just defines the concept).
- [x] 1.3.2 `pulsim::v2::numeric::IndexLike` concept = any signed
      integer ≥ 32 bit. Used to constrain templates that index into
      matrices.

## Phase 2 — Sparse linear algebra (~1 day)

### 2.1 `sparse/matrix.hpp` — sparse matrix wrapper
- [x] 2.1.1 `pulsim::v2::sparse::Matrix` = `Eigen::SparseMatrix<Real,
      Eigen::ColMajor, Index>`. **ColMajor** — matches the major
      direct sparse solvers' (SparseLU, KLU, UMFPACK) expectations.
      Index = `pulsim::v2::Index` (int32) keeps the matrix's index
      arrays packed and cache-friendly.
- [x] 2.1.2 `pulsim::v2::sparse::Triplet` = `Eigen::Triplet<Real,
      Index>`. Standard triplet-assembly pattern: collect (row, col,
      value) triplets, then `setFromTriplets` to build the
      compressed form. Stamping path in Layer 3 will use this.
- [x] 2.1.3 Convenience free functions:
      `pulsim::v2::sparse::stamp_dense(Matrix& M, Index r, Index c,
      const DenseMatrix& block)` — adds a small dense block at
      (r, c). Returns the count of triplets added. Used by Layer 4
      for stamping state-space blocks.
- [x] 2.1.4 NO operator overloads on `Matrix` other than what Eigen
      provides. We don't wrap Eigen in our own ops — too much
      maintenance for too little gain.

### 2.2 `sparse/solver.hpp` — solver abstraction
- [x] 2.2.1 `class pulsim::v2::sparse::DirectSolver` — abstract base
      class with three methods:
      - `bool analyze(const Matrix& M)` — symbolic factorization.
        Called once per topology change. Returns false if M is
        structurally singular.
      - `bool factorize(const Matrix& M)` — numeric factorization.
        Called once per numeric change. Returns false if M is
        numerically singular.
      - `void solve(const Vector& b, Vector& x) const` — triangular
        solve using the cached factor.
- [x] 2.2.2 `class pulsim::v2::sparse::SparseLuSolver : public
      DirectSolver` — implementation via
      `Eigen::SparseLU<SparseMatrix<Real>>`. Concrete reference
      implementation for Layer 0 tests.
- [x] 2.2.3 `std::unique_ptr<DirectSolver>
      pulsim::v2::sparse::make_default_solver()` — factory returning
      a `SparseLuSolver`. Future KLU / UMFPACK implementations
      register through the same factory.
- [x] 2.2.4 Solver lifecycle contract: `analyze` MUST be called
      before `factorize`; `factorize` MUST be called before `solve`.
      Calls out of order MUST throw `std::logic_error` with a clear
      message. Test the contract explicitly.

### 2.3 `sparse/sparsity.hpp` — sparsity-pattern utilities
- [x] 2.3.1 `pulsim::v2::sparse::reserve_capacity(Matrix& M, Size
      nnz_estimate)` — reserve capacity for nnz_estimate non-zeros
      without changing structural sparsity. Wraps
      `Eigen::SparseMatrix::reserve` with a clearer name.
- [x] 2.3.2 `pulsim::v2::sparse::compress_in_place(Matrix& M)` —
      ensures the matrix is in compressed form before passing to a
      solver. Wraps `makeCompressed`.

## Phase 3 — Layer 0 tests (~0.5 days)

### 3.1 `tests/v2/layer0/test_main.cpp`
- [x] 3.1.1 Standard Catch2 main, includes
      `<catch2/catch_session.hpp>` + `CATCH_CONFIG_MAIN`.

### 3.2 `tests/v2/layer0/test_numeric_types.cpp`
- [x] 3.2.1 Static asserts: `sizeof(Index) == 4`,
      `std::is_signed_v<Index>`, `std::is_floating_point_v<Real>`.
- [x] 3.2.2 `kInvalidIndex == -1` and `kGround == -1`.
- [x] 3.2.3 `Vector::Zero(N)` creates an N-vector of zeros.
- [x] 3.2.4 Concept-instantiation test: `static_assert(numeric::
      FloatingPoint<Real>)`.

### 3.3 `tests/v2/layer0/test_sparse_matrix.cpp`
- [x] 3.3.1 Build a 4×4 sparse matrix from triplets, check `nnz()`
      and individual entries via `coeff()`.
- [x] 3.3.2 `stamp_dense` adds a 2×2 dense block at (1, 1) — verify
      4 entries appear in M.
- [x] 3.3.3 `compress_in_place` makes a previously-uncompressed
      matrix compressed; the round-trip preserves all entries.

### 3.4 `tests/v2/layer0/test_sparse_solver.cpp`
- [x] 3.4.1 Solve an SPD 3×3 system (e.g. `[[4,-1,0],[-1,4,-1],
      [0,-1,4]] x = [2,4,2]`) with `SparseLuSolver`. Compare
      against `Eigen::FullPivLU` on the dense form to within 1e-12.
- [x] 3.4.2 Singular-matrix test: feed a structurally-singular
      matrix → `analyze` returns false, `factorize` not called.
- [x] 3.4.3 Numerically-singular matrix: structurally OK but
      determinant = 0 → `analyze` returns true, `factorize` returns
      false.
- [x] 3.4.4 Out-of-order call test: `solve` before `factorize`
      throws `std::logic_error`.
- [x] 3.4.5 Re-factor with same sparsity: `analyze` once, `factorize`
      twice (different values, same pattern) — both solves succeed,
      asymmetric assertion that `analyze` was indeed called once.

### 3.5 `tests/v2/layer0/test_factory.cpp`
- [x] 3.5.1 `make_default_solver()` returns a non-null
      `unique_ptr<DirectSolver>`.
- [x] 3.5.2 The returned solver is a `SparseLuSolver` (dynamic_cast
      check).

## Phase 4 — Documentation (~0.25 days)

### 4.1 Architecture review doc
- [x] 4.1.1 NEW `docs/architecture-review-v1.md` — distillation of
      the May 2026 architecture review that motivated v2: the seven
      structural problems, the four winning patterns, the language
      analysis, the phased plan.
- [x] 4.1.2 NEW `docs/pulsim-v2/README.md` — explains the layered
      v2 architecture, why each layer exists, and the order layers
      will land in OpenSpec follow-ups.

### 4.2 Per-layer design notes
- [x] 4.2.1 NEW `docs/pulsim-v2/layer0-numeric-and-sparse.md` —
      documents the Layer 0 surface (types, sparse matrix, solver
      interface) and the design decisions behind each (why
      ColMajor, why int32 index, why factor-and-cache contract).

## Phase 5 — Validation gate

- [x] 5.1 `pulsim_v2_layer0_tests` MUST run and pass with zero
      failures. Initial target: ≥ 15 assertions / ≥ 6 test cases.
- [x] 5.2 The existing v1 test suites (`pulsim_tests`,
      `pulsim_simulation_tests`) MUST stay green. v2 lands in
      parallel; zero v1 regressions allowed.
- [x] 5.3 `openspec validate bootstrap-pulsim-v2-kernel --strict`
      MUST pass.
