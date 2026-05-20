// =============================================================================
// Layer 0 — sparse matrix wrapper
// =============================================================================
//
// Triplet assembly, dense block stamping, compression round-trip.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/sparse/matrix.hpp"

#include <vector>

using namespace pulsim::v2;
using namespace pulsim::v2::sparse;
using Catch::Approx;

TEST_CASE("Sparse matrix is ColMajor int32 by default",
          "[v2][layer0][sparse]") {
    // Locks in the layout decision documented in design.md. ColMajor
    // matches every direct sparse solver's expected input. int32 keeps
    // indices cache-dense.
    STATIC_REQUIRE(Matrix::IsRowMajor == 0);  // 0 == ColMajor in Eigen
    STATIC_REQUIRE(sizeof(Matrix::StorageIndex) == 4);
}

TEST_CASE("Triplet assembly preserves all entries",
          "[v2][layer0][sparse]") {
    Matrix M(4, 4);
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {1, 1, 2.0}, {2, 2, 3.0}, {3, 3, 4.0}
    };
    M.setFromTriplets(triplets.begin(), triplets.end());

    REQUIRE(M.nonZeros() == 4);
    for (Index i = 0; i < 4; ++i) {
        REQUIRE(M.coeff(i, i) == Approx(static_cast<Real>(i + 1)));
    }
    // Off-diagonal stays zero.
    REQUIRE(M.coeff(0, 1) == Approx(Real{0}));
    REQUIRE(M.coeff(2, 0) == Approx(Real{0}));
}

TEST_CASE("stamp_dense adds a small block at the requested offset",
          "[v2][layer0][sparse]") {
    Matrix M(4, 4);
    DenseMatrix block = DenseMatrix::Identity(2, 2);  // [[1, 0], [0, 1]]

    const Size added = stamp_dense(M, /*row=*/1, /*col=*/1, block);

    REQUIRE(added == 4);                        // 2×2 = 4 entries
    REQUIRE(M.coeff(1, 1) == Approx(Real{1}));
    REQUIRE(M.coeff(2, 2) == Approx(Real{1}));
    REQUIRE(M.coeff(1, 2) == Approx(Real{0}));
    REQUIRE(M.coeff(2, 1) == Approx(Real{0}));
    REQUIRE(M.coeff(0, 0) == Approx(Real{0}));  // untouched
    REQUIRE(M.coeff(3, 3) == Approx(Real{0}));
}

TEST_CASE("stamp_dense accumulates on repeated calls",
          "[v2][layer0][sparse]") {
    // Stamping the same block twice should double the entry values —
    // matches the MNA pattern where multiple devices contribute to
    // the same matrix entry (e.g. two parallel resistors).
    Matrix M(2, 2);
    DenseMatrix block(2, 2);
    block << 1.0, 2.0, 3.0, 4.0;

    stamp_dense(M, 0, 0, block);
    stamp_dense(M, 0, 0, block);

    REQUIRE(M.coeff(0, 0) == Approx(Real{2}));
    REQUIRE(M.coeff(0, 1) == Approx(Real{4}));
    REQUIRE(M.coeff(1, 0) == Approx(Real{6}));
    REQUIRE(M.coeff(1, 1) == Approx(Real{8}));
}

TEST_CASE("compress_in_place puts the matrix in compressed form",
          "[v2][layer0][sparse]") {
    Matrix M(3, 3);
    M.coeffRef(0, 0) = 1.0;
    M.coeffRef(1, 1) = 2.0;
    M.coeffRef(2, 2) = 3.0;
    // After random coeffRef inserts, the matrix MAY be uncompressed.
    compress_in_place(M);
    REQUIRE(M.isCompressed());
    // Round-trip preserves all entries.
    REQUIRE(M.coeff(0, 0) == Approx(Real{1}));
    REQUIRE(M.coeff(1, 1) == Approx(Real{2}));
    REQUIRE(M.coeff(2, 2) == Approx(Real{3}));
}

TEST_CASE("reserve_capacity pre-allocates so subsequent inserts avoid "
          "quadratic re-allocation",
          "[v2][layer0][sparse]") {
    // Eigen's SparseMatrix::reserve must be called on a freshly-
    // constructed matrix BEFORE coeffRef inserts. Calling it after
    // inserts (when the matrix is in uncompressed transitional state)
    // would abort — that's an Eigen API constraint, not a Pulsim one,
    // and reserve_capacity inherits it.
    Matrix M(3, 3);
    reserve_capacity(M, 10);
    M.coeffRef(0, 0) = 1.0;
    M.coeffRef(1, 1) = 2.0;
    M.coeffRef(2, 2) = 3.0;
    REQUIRE(M.nonZeros() == 3);
    REQUIRE(M.coeff(0, 0) == Approx(Real{1}));
    REQUIRE(M.coeff(1, 1) == Approx(Real{2}));
    REQUIRE(M.coeff(2, 2) == Approx(Real{3}));
}
