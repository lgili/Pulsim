// =============================================================================
// Layer 5 V2 — SwitchedDiode state-decision unit tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

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

// -----------------------------------------------------------------------------
// PSIM-style forward drop V_f — the single law every engine decides with.
// -----------------------------------------------------------------------------

TEST_CASE("SwitchedDiode: turn_on_voltage is max(V_th, V_f), V_th at V_f = 0",
          "[v2][layer5_v2][switched_diode][forward_drop]") {
    using P = SwitchedDiode::Params;
    REQUIRE(SwitchedDiode::turn_on_voltage(P{20.0, 1e-9, 0.3, 0.7}) == 0.7);
    REQUIRE(SwitchedDiode::turn_on_voltage(P{20.0, 1e-9, 0.7, 0.7}) == 0.7);
    REQUIRE(SwitchedDiode::turn_on_voltage(P{20.0, 1e-9, 0.75, 0.7}) == 0.75);
    REQUIRE(SwitchedDiode::turn_on_voltage(P{20.0, 1e-9, 0.7}) == 0.7);
    // Negative V_th with no drop is still honoured (no clamp to 0).
    REQUIRE(SwitchedDiode::turn_on_voltage(P{20.0, 1e-9, -0.2}) == -0.2);
}

TEST_CASE("SwitchedDiode: branch_current carries the drop only while ON",
          "[v2][layer5_v2][switched_diode][forward_drop]") {
    const SwitchedDiode::Params p{.g_on = 20.0, .g_off = 1e-9,
                                  .V_th = 0.0, .V_f = 0.7};
    REQUIRE(SwitchedDiode::branch_current(true, 1.2, p) ==
            Catch::Approx(10.0).epsilon(1e-14));
    REQUIRE(SwitchedDiode::branch_current(true, 0.5, p) < 0.0);
    REQUIRE(SwitchedDiode::branch_current(false, 1.2, p) == 1e-9 * 1.2);
    // V_f = 0: exactly g·v, bit for bit.
    const SwitchedDiode::Params q{.g_on = 20.0, .g_off = 1e-9, .V_th = 0.0};
    REQUIRE(SwitchedDiode::branch_current(true, 0.3, q) == 20.0 * 0.3);
}

TEST_CASE("SwitchedDiode: decide and crossing_signal across V_th vs V_f",
          "[v2][layer5_v2][switched_diode][forward_drop]") {
    constexpr Real eps = 1e-9;
    struct Row { Real V_th, V_f, V_on; };
    for (const Row r : {Row{0.3, 0.7, 0.7}, Row{0.7, 0.7, 0.7},
                        Row{0.75, 0.7, 0.75}, Row{-0.2, 0.0, -0.2},
                        Row{0.0, 0.0, 0.0}}) {
        const SwitchedDiode::Params p{.g_on = 20.0, .g_off = 1e-9,
                                      .V_th = r.V_th, .V_f = r.V_f};
        INFO("V_th=" << r.V_th << " V_f=" << r.V_f);
        // OFF → ON at V_on (non-strict), not before.
        REQUIRE(SwitchedDiode::decide(false, r.V_on, p));
        REQUIRE_FALSE(SwitchedDiode::decide(false, r.V_on - eps, p));
        // ON holds strictly above V_f, drops at and below it.
        REQUIRE(SwitchedDiode::decide(true, r.V_f + eps, p));
        REQUIRE_FALSE(SwitchedDiode::decide(true, r.V_f, p));
        REQUIRE_FALSE(SwitchedDiode::decide(true, r.V_f - eps, p));
        // The crossing signal's sign is the decision.
        for (const Real v : {r.V_f - eps, r.V_f + eps, r.V_on - eps,
                             r.V_on + eps}) {
            REQUIRE((SwitchedDiode::crossing_signal(true, v, p) > 0) ==
                    SwitchedDiode::decide(true, v, p));
            REQUIRE((SwitchedDiode::crossing_signal(false, v, p) >= 0) ==
                    SwitchedDiode::decide(false, v, p));
        }
        // The 4-argument form agrees when fed branch_current.
        for (const bool on : {false, true}) {
            for (const Real v : {r.V_f - eps, r.V_f + eps, r.V_on}) {
                REQUIRE(SwitchedDiode::decide_next_state(
                            on, v, SwitchedDiode::branch_current(on, v, p),
                            p) == SwitchedDiode::decide(on, v, p));
            }
        }
    }
}
