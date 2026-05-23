// =============================================================================
// Layer 5 V2 — SwitchedDiode state-decision unit tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/models/switched_diode.hpp"

using namespace pulsim;
using namespace pulsim::models;

TEST_CASE("SwitchedDiode: OFF + forward bias above V_th → ON",
          "[v2][layer5_v2][switched_diode]") {
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.0};
    REQUIRE(SwitchedDiode::decide_next_state(
        /*currently_on=*/false, /*v=*/0.5, /*i=*/0.0, p));
}

TEST_CASE("SwitchedDiode: OFF + forward bias below V_th stays OFF",
          "[v2][layer5_v2][switched_diode]") {
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.7};
    REQUIRE_FALSE(SwitchedDiode::decide_next_state(
        /*currently_on=*/false, /*v=*/0.5, /*i=*/0.0, p));
}

TEST_CASE("SwitchedDiode: ON + reverse current → OFF",
          "[v2][layer5_v2][switched_diode]") {
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.0};
    REQUIRE_FALSE(SwitchedDiode::decide_next_state(
        /*currently_on=*/true, /*v=*/1.0, /*i=*/-0.1, p));
}

TEST_CASE("SwitchedDiode: ON + zero current → OFF (boundary)",
          "[v2][layer5_v2][switched_diode]") {
    // i_diode == 0 is the natural commutation point. Rule says
    // OFF → ON iff v ≥ V_th, ON → OFF iff i ≤ 0.
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.0};
    REQUIRE_FALSE(SwitchedDiode::decide_next_state(
        /*currently_on=*/true, /*v=*/1.0, /*i=*/0.0, p));
}

TEST_CASE("SwitchedDiode: ON + forward current stays ON",
          "[v2][layer5_v2][switched_diode]") {
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.0};
    REQUIRE(SwitchedDiode::decide_next_state(
        /*currently_on=*/true, /*v=*/1.0, /*i=*/+0.5, p));
}

TEST_CASE("SwitchedDiode: OFF + reverse bias stays OFF",
          "[v2][layer5_v2][switched_diode]") {
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.0};
    REQUIRE_FALSE(SwitchedDiode::decide_next_state(
        /*currently_on=*/false, /*v=*/-1.0, /*i=*/0.0, p));
}

TEST_CASE("SwitchedDiode: Si V_th = 0.7 threshold check",
          "[v2][layer5_v2][switched_diode]") {
    SwitchedDiode::Params p{.g_on = 1e3, .g_off = 1e-9, .V_th = 0.7};
    SECTION("0.5 V forward → still OFF") {
        REQUIRE_FALSE(SwitchedDiode::decide_next_state(
            false, 0.5, 0.0, p));
    }
    SECTION("0.7 V forward → just barely ON") {
        REQUIRE(SwitchedDiode::decide_next_state(
            false, 0.7, 0.0, p));
    }
    SECTION("0.8 V forward → ON") {
        REQUIRE(SwitchedDiode::decide_next_state(
            false, 0.8, 0.0, p));
    }
}
