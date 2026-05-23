// =============================================================================
// Layer 3 — fixed-state switch stamper
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/stamp_switch.hpp"

using namespace pulsim;
using namespace pulsim::stamping;
using Catch::Approx;

TEST_CASE("Closed switch stamps g_on between its two terminals",
          "[v2][layer3][stamp_switch]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    x << Real{1}, Real{0};

    BranchCoord coord{Index{0}, Index{1}, Index{0}};
    stamp_switch_fixed(J, f, x, coord,
                        /*closed=*/true,
                        /*g_on=*/Real{1e3},
                        /*g_off=*/Real{1e-9});

    REQUIRE(J.coeff(0, 0) == Approx(Real{1e3}));
    REQUIRE(J.coeff(0, 1) == Approx(Real{-1e3}));
    REQUIRE(J.coeff(1, 0) == Approx(Real{-1e3}));
    REQUIRE(J.coeff(1, 1) == Approx(Real{1e3}));
    REQUIRE(f[0] == Approx(Real{1e3}));        // G·(1 - 0)
    REQUIRE(f[1] == Approx(Real{-1e3}));
}

TEST_CASE("Open switch stamps g_off (small but non-singular)",
          "[v2][layer3][stamp_switch]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    x << Real{10}, Real{0};

    BranchCoord coord{Index{0}, Index{1}, Index{0}};
    stamp_switch_fixed(J, f, x, coord,
                        /*closed=*/false,
                        /*g_on=*/Real{1e3},
                        /*g_off=*/Real{1e-9});

    REQUIRE(J.coeff(0, 0) == Approx(Real{1e-9}));
    REQUIRE(f[0] == Approx(Real{1e-8}));       // 1e-9 · 10
}

TEST_CASE("Switch touching ground stamps only the active-row entry",
          "[v2][layer3][stamp_switch]") {
    sparse::Matrix J(1, 1);
    Vector f = Vector::Zero(1);
    Vector x(1);
    x << Real{5};

    BranchCoord coord{Index{0}, kGround, Index{0}};
    stamp_switch_fixed(J, f, x, coord, /*closed=*/true,
                        Real{1e3}, Real{1e-9});

    REQUIRE(J.coeff(0, 0) == Approx(Real{1e3}));
    REQUIRE(f[0] == Approx(Real{5e3}));
    REQUIRE(J.nonZeros() == 1);
}
