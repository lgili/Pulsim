// =============================================================================
// Layer 0 — numeric primitive types
// =============================================================================
//
// Locks in the size / signedness contract for Real, Index, Size. If a
// future change widens Index to 8 B or flips Real to a non-floating
// type, these tests fail loudly rather than silently degrading every
// consumer.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <cstdint>
#include <type_traits>

using namespace pulsim;

TEST_CASE("Index is exactly 4 bytes and signed", "[v2][layer0][numeric]") {
    STATIC_REQUIRE(sizeof(Index) == 4);
    STATIC_REQUIRE(std::is_same_v<Index, std::int32_t>);
    STATIC_REQUIRE(std::is_signed_v<Index>);
}

TEST_CASE("Real is a floating-point type", "[v2][layer0][numeric]") {
    STATIC_REQUIRE(std::is_floating_point_v<Real>);
    // The default build is double precision; a single-precision build
    // (`-DPULSIM_V2_REAL_TYPE=float`) flips this to 4. Both are valid.
    STATIC_REQUIRE(sizeof(Real) == 8 || sizeof(Real) == 4);
}

TEST_CASE("Sentinel values are well-defined", "[v2][layer0][numeric]") {
    STATIC_REQUIRE(kInvalidIndex == Index{-1});
    STATIC_REQUIRE(kGround == Index{-1});
    STATIC_REQUIRE(kInvalidIndex == kGround);  // intentional aliases
}

TEST_CASE("Vector::Zero creates a zero-initialised column vector",
          "[v2][layer0][numeric]") {
    const auto v = Vector::Zero(5);
    REQUIRE(v.size() == 5);
    for (Index i = 0; i < v.size(); ++i) {
        REQUIRE(v[i] == Real{0});
    }
}

TEST_CASE("Fixed-size dense aliases compile and match Eigen layout",
          "[v2][layer0][numeric]") {
    STATIC_REQUIRE(Vector3::RowsAtCompileTime == 3);
    STATIC_REQUIRE(Vector3::ColsAtCompileTime == 1);
    STATIC_REQUIRE(Matrix3::RowsAtCompileTime == 3);
    STATIC_REQUIRE(Matrix3::ColsAtCompileTime == 3);
    // Sanity: a Vector3 fits on the stack with NO heap allocation.
    // Eigen's small-fixed-size aliases use Eigen::DontAlign by default
    // for vectors of size ≤ 4; checking sizeof here protects against
    // an accidental "use std::vector underneath" regression.
    STATIC_REQUIRE(sizeof(Vector3) == 3 * sizeof(Real));
}

TEST_CASE("FloatingPoint concept accepts Real and standard floats, "
          "rejects int", "[v2][layer0][numeric]") {
    STATIC_REQUIRE(numeric::FloatingPoint<Real>);
    STATIC_REQUIRE(numeric::FloatingPoint<double>);
    STATIC_REQUIRE(numeric::FloatingPoint<float>);
    STATIC_REQUIRE_FALSE(numeric::FloatingPoint<int>);
    STATIC_REQUIRE_FALSE(numeric::FloatingPoint<long>);
}

TEST_CASE("IndexLike concept accepts Index and rejects unsigned",
          "[v2][layer0][numeric]") {
    STATIC_REQUIRE(numeric::IndexLike<Index>);
    STATIC_REQUIRE(numeric::IndexLike<std::int32_t>);
    STATIC_REQUIRE(numeric::IndexLike<std::int64_t>);
    STATIC_REQUIRE_FALSE(numeric::IndexLike<std::uint32_t>);
    STATIC_REQUIRE_FALSE(numeric::IndexLike<std::int16_t>);  // too small
}
