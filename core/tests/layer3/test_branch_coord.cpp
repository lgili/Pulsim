// =============================================================================
// Layer 3 — BranchCoord + ground-aware helpers
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/stamping/mna_convention.hpp"

using namespace pulsim;
using namespace pulsim::stamping;
using Catch::Approx;

TEST_CASE("read_node_voltage returns x[i] for active nodes",
          "[v2][layer3][branch_coord]") {
    Vector x(3);
    x << Real{5}, Real{7}, Real{3};
    REQUIRE(read_node_voltage(x, Index{0}) == Approx(Real{5}));
    REQUIRE(read_node_voltage(x, Index{1}) == Approx(Real{7}));
    REQUIRE(read_node_voltage(x, Index{2}) == Approx(Real{3}));
}

TEST_CASE("read_node_voltage returns 0 for ground",
          "[v2][layer3][branch_coord]") {
    Vector x(3);
    x << Real{5}, Real{7}, Real{3};
    REQUIRE(read_node_voltage(x, kGround) == Approx(Real{0}));
}

TEST_CASE("node_is_active distinguishes real nodes from ground",
          "[v2][layer3][branch_coord]") {
    REQUIRE(node_is_active(Index{0}));
    REQUIRE(node_is_active(Index{42}));
    REQUIRE_FALSE(node_is_active(kGround));
    REQUIRE_FALSE(node_is_active(Index{-1}));
}
