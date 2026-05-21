// =============================================================================
// Layer 3 — voltage-source constraint stamper
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/stamp_voltage_source.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::stamping;
using Catch::Approx;

TEST_CASE("Voltage source between active nodes — residual + Jacobian",
          "[v2][layer3][stamp_vsource]") {
    // 2 nodes + 1 branch-current unknown = state size 3.
    sparse::Matrix J(3, 3);
    Vector f = Vector::Zero(3);
    Vector x(3);
    x << Real{3}, Real{0}, Real{2};   // [v0, v1, i_branch]

    BranchCoord coord{Index{0}, Index{1}, Index{0}};
    stamp_voltage_source(J, f, x, coord,
                          /*branch_var_id=*/Index{2}, /*V=*/Real{5});

    // Constraint row: f[2] = (3 - 0) - 5 = -2
    REQUIRE(f[2] == Approx(Real{-2}));
    // KCL contributions: f[0] += i_branch (= 2), f[1] -= 2.
    REQUIRE(f[0] == Approx(Real{2}));
    REQUIRE(f[1] == Approx(Real{-2}));

    // Jacobian: 4 entries on the source's cross.
    REQUIRE(J.coeff(0, 2) == Approx(Real{ 1}));  // KCL: +i_branch at node 0
    REQUIRE(J.coeff(1, 2) == Approx(Real{-1}));  // KCL: -i_branch at node 1
    REQUIRE(J.coeff(2, 0) == Approx(Real{ 1}));  // Constraint: +v[0]
    REQUIRE(J.coeff(2, 1) == Approx(Real{-1}));  // Constraint: -v[1]
}

TEST_CASE("Voltage source to ground — only active-node entries stamped",
          "[v2][layer3][stamp_vsource]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    x << Real{0}, Real{0};

    BranchCoord coord{Index{0}, kGround, Index{0}};
    stamp_voltage_source(J, f, x, coord,
                          /*branch_var_id=*/Index{1}, /*V=*/Real{12});

    // Constraint: f[1] = (0 - 0) - 12 = -12.
    REQUIRE(f[1] == Approx(Real{-12}));
    // KCL: f[0] += i_branch (= 0 at the start).
    REQUIRE(f[0] == Approx(Real{0}));

    // Jacobian: only the (0, 1) and (1, 0) entries.
    REQUIRE(J.coeff(0, 1) == Approx(Real{ 1}));
    REQUIRE(J.coeff(1, 0) == Approx(Real{ 1}));
    // Ground entries skipped.
    REQUIRE(J.nonZeros() == 2);
}

TEST_CASE("Voltage source residual is zero at convergence",
          "[v2][layer3][stamp_vsource]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    // At convergence: v[0] = V and i_branch = 0 (open-circuit source).
    x << Real{12}, Real{0};

    BranchCoord coord{Index{0}, kGround, Index{0}};
    stamp_voltage_source(J, f, x, coord, Index{1}, Real{12});

    REQUIRE(f[0] == Approx(Real{0}));            // KCL at node 0
    REQUIRE(f[1] == Approx(Real{0}));            // constraint
}
