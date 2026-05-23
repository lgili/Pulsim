// =============================================================================
// Layer 4 — PwlSegment (move-only contract)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/segment.hpp"

#include <type_traits>

using namespace pulsim::pwl;

TEST_CASE("PwlSegment is move-only",
          "[v2][layer4][segment]") {
    STATIC_REQUIRE(std::is_move_constructible_v<PwlSegment>);
    STATIC_REQUIRE(std::is_move_assignable_v<PwlSegment>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<PwlSegment>);
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<PwlSegment>);
}

TEST_CASE("Default-constructed PwlSegment has zero state size",
          "[v2][layer4][segment]") {
    PwlSegment seg;
    REQUIRE(seg.state_size == 0);
    REQUIRE(seg.solver == nullptr);
}
