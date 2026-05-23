// =============================================================================
// Layer 4 V1 — stamp_companion helpers
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/stamping/stamp_companion.hpp"

using namespace pulsim;
using namespace pulsim::stamping;
using Catch::Approx;

namespace {

// Helper to compress + read out a single coefficient.
Real coeff(sparse::Matrix& J, Index r, Index c) {
    sparse::compress_in_place(J);
    return J.coeff(r, c);
}

}  // namespace

TEST_CASE("stamp_capacitor_companion: between node0 and ground",
          "[v2][layer4_v1][stamp_companion]") {
    sparse::Matrix J(1, 1);
    Vector b = Vector::Zero(1);
    BranchCoord coord{.from = 0, .to = -1, .branch_id = 0};

    stamp_capacitor_companion(J, b, coord, /*g_eq=*/2.0);

    REQUIRE(coeff(J, 0, 0) == Approx(2.0).margin(1e-12));
    // b is NOT touched by the companion stamp.
    REQUIRE(b[0] == Approx(0.0).margin(1e-15));
}

TEST_CASE("stamp_capacitor_companion: between two non-ground nodes",
          "[v2][layer4_v1][stamp_companion]") {
    sparse::Matrix J(2, 2);
    Vector b = Vector::Zero(2);
    BranchCoord coord{.from = 0, .to = 1, .branch_id = 0};

    stamp_capacitor_companion(J, b, coord, /*g_eq=*/3.0);

    REQUIRE(coeff(J, 0, 0) == Approx(+3.0).margin(1e-12));
    REQUIRE(coeff(J, 0, 1) == Approx(-3.0).margin(1e-12));
    REQUIRE(coeff(J, 1, 0) == Approx(-3.0).margin(1e-12));
    REQUIRE(coeff(J, 1, 1) == Approx(+3.0).margin(1e-12));
}

TEST_CASE("stamp_capacitor_companion: skip rows for ground endpoints",
          "[v2][layer4_v1][stamp_companion]") {
    sparse::Matrix J(1, 1);
    Vector b = Vector::Zero(1);
    // from=ground, to=node0: only the (0,0) entry should appear.
    BranchCoord coord{.from = -1, .to = 0, .branch_id = 0};

    stamp_capacitor_companion(J, b, coord, /*g_eq=*/5.0);

    REQUIRE(coeff(J, 0, 0) == Approx(5.0).margin(1e-12));
}

TEST_CASE("stamp_inductor_companion: node0 → ground, branch_var_id=1",
          "[v2][layer4_v1][stamp_companion]") {
    sparse::Matrix J(2, 2);
    Vector b = Vector::Zero(2);
    BranchCoord coord{.from = 0, .to = -1, .branch_id = 0};
    const Index branch_var = 1;
    // g_eq_inv = 5e-4 → 2L/dt = 2000.
    stamp_inductor_companion(J, b, coord, branch_var,
                              /*g_eq_inv=*/5e-4);

    // KCL of i_L on node 0:
    REQUIRE(coeff(J, 0, branch_var) == Approx(+1.0).margin(1e-12));
    // Constraint row at branch_var (= row 1):
    REQUIRE(coeff(J, branch_var, 0) == Approx(+1.0).margin(1e-12));
    REQUIRE(coeff(J, branch_var, branch_var) ==
            Approx(-2000.0).margin(1e-9));
}

TEST_CASE("stamp_inductor_companion: between two non-ground nodes",
          "[v2][layer4_v1][stamp_companion]") {
    sparse::Matrix J(3, 3);
    Vector b = Vector::Zero(3);
    BranchCoord coord{.from = 0, .to = 1, .branch_id = 0};
    const Index branch_var = 2;
    // g_eq_inv = 1 → 2L/dt = 1 (toy values to make checks easy)
    stamp_inductor_companion(J, b, coord, branch_var,
                              /*g_eq_inv=*/1.0);

    // KCL of i_L on each terminal:
    REQUIRE(coeff(J, 0, branch_var) == Approx(+1.0).margin(1e-12));
    REQUIRE(coeff(J, 1, branch_var) == Approx(-1.0).margin(1e-12));
    // Constraint row:
    REQUIRE(coeff(J, branch_var, 0) == Approx(+1.0).margin(1e-12));
    REQUIRE(coeff(J, branch_var, 1) == Approx(-1.0).margin(1e-12));
    REQUIRE(coeff(J, branch_var, branch_var) ==
            Approx(-1.0).margin(1e-12));
}
