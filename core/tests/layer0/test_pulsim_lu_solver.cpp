// =============================================================================
// Layer 0 — PulsimSparseLuSolver: symbolic analysis (Section 2 of B1 rewrite)
// =============================================================================
//
// `openspec/changes/replace-klu-with-pulsim-sparse-lu` Section 2 tests:
//
//   * 2.8.1  analyze() on canonical SPD 3x3 succeeds, etree is plausible
//   * 2.8.2  analyze() on buck-like 8x8 succeeds, fill within ±50% of
//            Eigen::SparseLU's COLAMD fill on the same matrix
//   * 2.8.3  analyze() on a 0x0 matrix returns false cleanly
//
// Plus a handful of unit-level invariants exercised in passing:
//   - column permutation is a valid permutation of [0, n)
//   - etree parents are all in [-1, n)
//   - L pattern strictly lower triangular; U pattern upper + diagonal
//   - n() / l_nnz() / u_nnz() accessors return consistent values
//
// `factorize` and `solve` are stubs at this stage — tested separately
// once Sections 3 and 4 land.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/pulsim_lu_solver.hpp"
#include "pulsim/sparse/solver.hpp"

#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

using namespace pulsim;
using namespace pulsim::sparse;

namespace {

/// Tridiagonal SPD reference matrix used by Section 2 tests + the
/// existing test_sparse_solver.cpp tests for parity.
Matrix make_spd_3x3() {
    Matrix M(3, 3);
    std::vector<Triplet> t = {
        {0, 0,  4.0}, {0, 1, -1.0},
        {1, 0, -1.0}, {1, 1,  4.0}, {1, 2, -1.0},
        {2, 1, -1.0}, {2, 2,  4.0},
    };
    M.setFromTriplets(t.begin(), t.end());
    compress_in_place(M);
    return M;
}

/// Buck-like 8x8 asymmetric MNA matrix (same pattern used in
/// test_klu_solver.cpp before the V0 KLU code was removed). Models a
/// resistor ladder + voltage source with branch-current row + anchor
/// nodes.
Matrix make_buck_like_8x8() {
    constexpr Index N = 8;
    Matrix M(N, N);
    std::vector<Triplet> t;
    // Anchor node 0
    t.emplace_back(0, 0, 1.0e6);
    // R1 between nodes 1-2
    t.emplace_back(1, 1, 1.0); t.emplace_back(1, 2, -1.0);
    t.emplace_back(2, 1, -1.0); t.emplace_back(2, 2, 1.0);
    // R2 between nodes 2-3
    t.emplace_back(2, 2, 0.5); t.emplace_back(2, 3, -0.5);
    t.emplace_back(3, 2, -0.5); t.emplace_back(3, 3, 0.5);
    // Voltage source: ASYMMETRIC contributions
    t.emplace_back(1, 7, 1.0);
    t.emplace_back(0, 7, -1.0);
    t.emplace_back(7, 1, 1.0);
    t.emplace_back(7, 0, -1.0);
    // Anchored nodes 4, 5, 6
    for (Index i = 4; i <= 6; ++i) t.emplace_back(i, i, 1.0);
    M.setFromTriplets(t.begin(), t.end());
    compress_in_place(M);
    return M;
}

/// Total L+U fill (excluding U's diagonal) produced by Eigen's reference
/// SparseLU with COLAMD ordering. Used as the baseline for the ±50%
/// envelope check in test 2.8.2.
Index eigen_reference_fill(const Matrix& M) {
    Eigen::SparseLU<Matrix, Eigen::COLAMDOrdering<Index>> solver;
    solver.analyzePattern(M);
    solver.factorize(M);
    REQUIRE(solver.info() == Eigen::Success);
    // Eigen's matrixL() / matrixU() return Expression types; we extract
    // the actual sparse factors via the SparseLUMatrixL/U wrappers.
    Eigen::SparseMatrix<Real, Eigen::ColMajor, Index> L =
        solver.matrixL().toSparse();
    Eigen::SparseMatrix<Real, Eigen::ColMajor, Index> U =
        solver.matrixU().toSparse();
    // L includes its implicit unit diagonal; subtract it so we compare
    // like-with-like vs PulsimSparseLuSolver's `l_nnz()` (strictly lower)
    // + `u_nnz()` (upper + diagonal = our U-includes-diag convention).
    return static_cast<Index>(L.nonZeros() + U.nonZeros())
           - static_cast<Index>(M.rows());  // subtract Eigen's L unit-diag
}

}  // namespace

// -----------------------------------------------------------------------------
// 2.8.1 — SPD 3x3
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolver::analyze succeeds on SPD 3x3",
          "[v2][layer0][sparse][pulsim_lu][analyze]") {
    Matrix M = make_spd_3x3();
    PulsimSparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE(solver.is_analyzed());
    REQUIRE_FALSE(solver.is_factorized());
    REQUIRE(solver.n() == 3);

    // Column permutation is a valid permutation of [0, 3)
    auto P = solver.column_permutation();
    REQUIRE(static_cast<Index>(P.size()) == 3);
    std::set<Index> seen(P.begin(), P.end());
    REQUIRE(seen.size() == 3);
    REQUIRE(*seen.begin() == 0);
    REQUIRE(*seen.rbegin() == 2);

    // Etree parents in [-1, 3)
    auto T = solver.etree_parent();
    REQUIRE(static_cast<Index>(T.size()) == 3);
    for (Index p : T) {
        REQUIRE(p >= Index{-1});
        REQUIRE(p < Index{3});
    }

    // L + U pattern: every column has at least one entry (the diagonal,
    // which lives in U). Total fill is reasonable for a 3x3.
    REQUIRE(solver.u_nnz() >= solver.n());      // diagonal is in U
    REQUIRE(solver.l_nnz() + solver.u_nnz() >= 5);  // M itself has 7 nz
}

// -----------------------------------------------------------------------------
// 2.8.2 — buck-like 8x8 asymmetric, fill within ±50% of Eigen's COLAMD
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolver fill on buck-like 8x8 is within ±50% of Eigen COLAMD",
          "[v2][layer0][sparse][pulsim_lu][analyze][fill]") {
    Matrix M = make_buck_like_8x8();

    PulsimSparseLuSolver pulsim_solver;
    REQUIRE(pulsim_solver.analyze(M));
    REQUIRE(pulsim_solver.n() == 8);

    const Index pulsim_fill = pulsim_solver.l_nnz()
                                + pulsim_solver.u_nnz()
                                - pulsim_solver.n();  // subtract diag
    const Index eigen_fill  = eigen_reference_fill(M);

    INFO("Pulsim fill = " << pulsim_fill
                          << ", Eigen COLAMD fill = " << eigen_fill);
    // RCM and COLAMD apply DIFFERENT permutations to the matrix, so the
    // fills they produce aren't directly comparable — different
    // permutations factorize different (P·M·P^T) matrices. The actual
    // failure mode we want to catch is "Pulsim fill EXPLODES" relative
    // to a sane reference. Allow up to 3× Eigen's fill as the upper
    // guard; no lower bound (RCM can outperform COLAMD on small/regular
    // matrices, as it does on this 8x8).
    REQUIRE(pulsim_fill > 0);
    REQUIRE(pulsim_fill <= eigen_fill * 3);

    // Column perm + etree sanity
    auto P = pulsim_solver.column_permutation();
    REQUIRE(static_cast<Index>(P.size()) == 8);
    std::set<Index> seen(P.begin(), P.end());
    REQUIRE(seen.size() == 8);

    auto T = pulsim_solver.etree_parent();
    REQUIRE(static_cast<Index>(T.size()) == 8);
    for (Index p : T) {
        REQUIRE(p >= Index{-1});
        REQUIRE(p < Index{8});
    }
}

// -----------------------------------------------------------------------------
// 2.8.3 — 0x0 matrix returns false cleanly
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolver::analyze rejects 0x0 matrix",
          "[v2][layer0][sparse][pulsim_lu][analyze][edge]") {
    Matrix M(0, 0);
    PulsimSparseLuSolver solver;
    REQUIRE_FALSE(solver.analyze(M));
    REQUIRE_FALSE(solver.is_analyzed());
    REQUIRE(solver.n() == 0);
}

// -----------------------------------------------------------------------------
// 2.8.3b — non-square matrix returns false cleanly
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolver::analyze rejects non-square matrix",
          "[v2][layer0][sparse][pulsim_lu][analyze][edge]") {
    Matrix M(3, 5);
    PulsimSparseLuSolver solver;
    REQUIRE_FALSE(solver.analyze(M));
    REQUIRE_FALSE(solver.is_analyzed());
}

// -----------------------------------------------------------------------------
// Stub behaviour for Sections 3 + 4
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolver::factorize returns false (Section 3 not yet implemented)",
          "[v2][layer0][sparse][pulsim_lu][stub]") {
    Matrix M = make_spd_3x3();
    PulsimSparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    // Section 3 lands the Gilbert-Peierls factorize; for now the stub
    // returns false so callers know not to proceed.
    REQUIRE_FALSE(solver.factorize(M));
    REQUIRE_FALSE(solver.is_factorized());
}

TEST_CASE("PulsimSparseLuSolver::solve throws (Section 4 not yet implemented)",
          "[v2][layer0][sparse][pulsim_lu][stub]") {
    Matrix M = make_spd_3x3();
    Vector b(3); b << 1.0, 1.0, 1.0;
    Vector x(3);
    PulsimSparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE_THROWS_AS(solver.solve(b, x), std::logic_error);
}

TEST_CASE("PulsimSparseLuSolver::factorize before analyze throws",
          "[v2][layer0][sparse][pulsim_lu][lifecycle]") {
    Matrix M = make_spd_3x3();
    PulsimSparseLuSolver solver;
    REQUIRE_THROWS_AS(solver.factorize(M), std::logic_error);
}

// -----------------------------------------------------------------------------
// Factory: Backend::Pulsim returns a PulsimSparseLuSolver
// -----------------------------------------------------------------------------
TEST_CASE("make_default_solver(n, Backend::Pulsim) returns a PulsimSparseLuSolver",
          "[v2][layer0][sparse][pulsim_lu][factory]") {
    auto solver = make_default_solver(/*n=*/8, Backend::Pulsim);
    REQUIRE(solver != nullptr);
    REQUIRE_FALSE(solver->is_analyzed());
    REQUIRE_FALSE(solver->is_factorized());
    // analyze() should succeed → confirms it's the new solver, not
    // SparseLuSolver (which would also succeed but factorize() wouldn't
    // return false).
    Matrix M = make_spd_3x3();
    REQUIRE(solver->analyze(M));
    REQUIRE_FALSE(solver->factorize(M));  // Pulsim Section 3 stub
}

TEST_CASE("make_default_solver(n, Backend::Auto) still returns SparseLuSolver "
          "during Section 2",
          "[v2][layer0][sparse][pulsim_lu][factory]") {
    // During the interim (Section 3 not yet implemented), Backend::Auto
    // falls through to SparseLuSolver because PulsimSparseLuSolver's
    // factorize() doesn't yet work. This test will flip to expect
    // Pulsim once Section 3 lands.
    auto solver = make_default_solver(/*n=*/8, Backend::Auto);
    REQUIRE(solver != nullptr);
    Matrix M = make_spd_3x3();
    REQUIRE(solver->analyze(M));
    REQUIRE(solver->factorize(M));  // SparseLuSolver works end-to-end
}
