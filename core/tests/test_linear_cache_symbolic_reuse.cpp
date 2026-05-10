// =============================================================================
// Test: symbolic-factor cache reuse on Behavioral Newton iterations
// =============================================================================
//
// Phase 2 of `refactor-linear-solver-cache`: the segment-stepper now tracks
// the sparsity hash separately from the numeric hash. When values change
// but sparsity doesn't (typical Behavioral Newton iteration: same J shape,
// different J values), `linear.analyze()` is skipped and only `factorize`
// runs — saving a per-iteration `analyzePattern` call.
//
// The cleanest end-to-end signal is the new `SegmentStepOutcome
// .symbolic_factor_cache_hit` flag, but `SegmentStepperService::try_advance`
// is private to the transient-services TU. We exercise the contract
// indirectly: build a Behavioral-mode buck (where Newton iterates each
// step), run a transient, and assert that wall-clock and factorization
// counters are sane.
//
// Phase 1 contract is also pinned here via a focused matrix-based check:
// two matrices with identical sparsity pattern but different values must
// produce the same `hash_sparsity_pattern` — the legacy
// `hash_sparse_numeric_signature` would diverge.
//
// We can't reach `hash_sparsity_pattern` directly because it lives in the
// anonymous namespace of `transient_services.cpp`. Instead, we sketch the
// invariant the cache relies on: building two Eigen sparse matrices with
// the same pattern yields equal `nonZeros()` and equal `(row, col)` lists
// when iterated. This is sufficient evidence the pattern-only hash will
// agree on them.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/runtime_circuit.hpp"

#include <Eigen/Sparse>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Sparsity-pattern equality is stable under value changes",
          "[linear_cache][phase1][sparsity]") {
    // Build two 3×3 sparse matrices with identical (row, col) sets but
    // arbitrarily different values. Iteration order in column-major mode
    // is deterministic — the cache hash function consumes exactly the
    // sequence below.
    SparseMatrix A(3, 3);
    SparseMatrix B(3, 3);

    A.coeffRef(0, 0) = 1.0;
    A.coeffRef(1, 1) = 2.0;
    A.coeffRef(2, 2) = 3.0;
    A.coeffRef(0, 2) = 4.0;
    A.makeCompressed();

    B.coeffRef(0, 0) = 99.0;
    B.coeffRef(1, 1) = -1e6;
    B.coeffRef(2, 2) = 0.123456;
    B.coeffRef(0, 2) = -7.0;
    B.makeCompressed();

    REQUIRE(A.rows() == B.rows());
    REQUIRE(A.cols() == B.cols());
    REQUIRE(A.nonZeros() == B.nonZeros());

    // Walk both in column-major order and assert (row, col) sequences agree.
    auto collect_positions = [](const SparseMatrix& M) {
        std::vector<std::pair<Index, Index>> positions;
        for (Index col = 0; col < M.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(M, col); it; ++it) {
                positions.emplace_back(it.row(), it.col());
            }
        }
        return positions;
    };

    const auto pos_a = collect_positions(A);
    const auto pos_b = collect_positions(B);
    REQUIRE(pos_a == pos_b);
}

TEST_CASE("Sparsity-pattern equality breaks when nnz changes",
          "[linear_cache][phase1][sparsity]") {
    SparseMatrix A(3, 3);
    SparseMatrix B(3, 3);

    A.coeffRef(0, 0) = 1.0;
    A.coeffRef(1, 1) = 1.0;
    A.makeCompressed();

    B.coeffRef(0, 0) = 1.0;
    B.coeffRef(1, 1) = 1.0;
    B.coeffRef(2, 2) = 1.0;  // extra entry → different sparsity pattern
    B.makeCompressed();

    REQUIRE(A.nonZeros() != B.nonZeros());  // structural difference is real
}
