// =============================================================================
// Layer 2 V10 — `make_combined_switch_fn` tests
// =============================================================================
//
// Validates:
//   * Empty list → all bits OFF.
//   * Single fn → identical to that fn.
//   * Two fns with DISJOINT bits → both contribute.
//   * Two fns with OVERLAPPING bits → OR semantics.
//   * Sub-masks with size > num_switches → extra bits
//     ignored (no overflow).
//   * Sub-masks with size < num_switches → missing bits
//     stay OFF (no underflow).
//   * Null `std::function` entries skipped gracefully.
//   * Realistic composition: PSFB primary (V9, bits 0..3)
//     + sync rectifier pair (V6, bits 4..5) — all 6 bits
//     toggle from the same combined SwitchScheduleFn,
//     each independently driven, and never collide.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/sources/combined_switch_fn.hpp"
#include "pulsim/v2/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/v2/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v2;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::sources;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("make_combined_switch_fn — empty list → all OFF",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    auto sw = make_combined_switch_fn(4, {});
    REQUIRE(static_cast<bool>(sw));
    auto m = sw(0.5);
    REQUIRE(m.size() == 4);
    for (Size b = 0; b < 4; ++b) {
        REQUIRE(m.get(b) == false);
    }
}

TEST_CASE("make_combined_switch_fn — single fn ≡ that fn",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    auto pair = make_dead_time_pwm_pair_fn(
        1000.0, 0.4, /*hs=*/0, /*ls=*/1, 4, /*dt=*/1e-5);
    auto combo = make_combined_switch_fn(4, {pair});

    for (int k = 0; k < 200; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 200.0;
        auto a = pair(t);
        auto b = combo(t);
        INFO("t=" << t);
        for (Size i = 0; i < 4; ++i) {
            REQUIRE(a.get(i) == b.get(i));
        }
    }
}

TEST_CASE("make_combined_switch_fn — disjoint bit indices both contribute",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    // pair A drives bits 0,1; pair B drives bits 2,3.
    auto a = make_dead_time_pwm_pair_fn(
        1000.0, 0.5, 0, 1, /*n_sw=*/4, 0.0);
    auto b = make_dead_time_pwm_pair_fn(
        1000.0, 0.5, 2, 3, /*n_sw=*/4, 0.0,
        /*phase=*/5e-4);   // half cycle offset
    auto combo = make_combined_switch_fn(4, {a, b});

    for (int k = 0; k < 100; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 100.0;
        auto m = combo(t);
        // Reference: union of two pair outputs.
        auto ma = a(t);
        auto mb = b(t);
        INFO("t=" << t);
        for (Size i = 0; i < 4; ++i) {
            const bool expected = ma.get(i) || mb.get(i);
            REQUIRE(m.get(i) == expected);
        }
    }
}

TEST_CASE("make_combined_switch_fn — overlapping bits OR'd together",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    // Both pairs drive bit 0 as HS. Different duties, so
    // their HS-on intervals overlap part of the time.
    // Result: bit 0 ON iff either pair has it ON.
    // Use f=1 Hz so the period is T=1 s and our sample
    // times map directly to phase.
    auto a = make_dead_time_pwm_pair_fn(
        /*f=*/1.0, /*duty=*/0.3, /*hs=*/0, /*ls=*/1, 2, 0.0);
    auto b = make_dead_time_pwm_pair_fn(
        /*f=*/1.0, /*duty=*/0.7, /*hs=*/0, /*ls=*/1, 2, 0.0);
    auto combo = make_combined_switch_fn(2, {a, b});

    // a's HS region: [0, 0.3); b's HS region: [0, 0.7).
    // At t=0.2: BOTH pairs have HS ON → bit 0 = ON.
    auto m02 = combo(0.2);
    REQUIRE(m02.get(0) == true);
    REQUIRE(m02.get(1) == false);

    // At t=0.6: a wants LS (in [0.3, 1)); b still wants HS
    // (in [0, 0.7)). Union: bit 0 = ON (from b), bit 1 =
    // ON (from a). Shoot-through ALLOWED in the combined
    // case — it's the caller's job to avoid by assigning
    // disjoint bit indices.
    auto m06 = combo(0.6);
    REQUIRE(m06.get(0) == true);   // from b's HS
    REQUIRE(m06.get(1) == true);   // from a's LS
}

TEST_CASE("make_combined_switch_fn — sub-mask larger than num_switches → ignored bits",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    // Sub-fn returns a mask of size 4. We request 2 in the
    // combined mask. Bits 2,3 from the sub-fn must NOT
    // overflow into bits 0,1.
    auto big = make_dead_time_pwm_pair_fn(
        1000.0, 0.5, /*hs=*/2, /*ls=*/3, /*n_sw=*/4, 0.0);
    auto combo = make_combined_switch_fn(2, {big});
    for (int k = 0; k < 50; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 50.0;
        auto m = combo(t);
        REQUIRE(m.size() == 2);
        // The bits we asked for (0,1) shouldn't toggle —
        // the sub-fn only drove bits 2,3 which exceed the
        // combined num_switches.
        REQUIRE(m.get(0) == false);
        REQUIRE(m.get(1) == false);
    }
}

TEST_CASE("make_combined_switch_fn — sub-mask smaller than num_switches",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    // Sub-fn returns size-2 mask; combined asks for size 4.
    // Bits 2,3 in the combined mask stay OFF (sub didn't
    // know about them).
    auto small_pair = make_dead_time_pwm_pair_fn(
        1000.0, 0.5, 0, 1, /*n_sw=*/2, 0.0);
    auto combo = make_combined_switch_fn(4, {small_pair});

    // Sample several times.
    for (int k = 0; k < 50; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 50.0;
        auto m = combo(t);
        REQUIRE(m.size() == 4);
        // Bits 2,3 always OFF (sub-fn was smaller).
        REQUIRE(m.get(2) == false);
        REQUIRE(m.get(3) == false);
    }
}

TEST_CASE("make_combined_switch_fn — null function entries skipped",
          "[v2][layer2_v10][combined_switch_fn][unit]") {
    SwitchScheduleFn null_fn;   // default-constructed: empty
    auto real_fn = make_dead_time_pwm_pair_fn(
        1000.0, 0.5, 0, 1, 2, 0.0);
    auto combo = make_combined_switch_fn(
        2, {null_fn, real_fn, null_fn});

    // Should behave as if only real_fn were in the list.
    for (int k = 0; k < 50; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 50.0;
        auto a = real_fn(t);
        auto b = combo(t);
        INFO("t=" << t);
        REQUIRE(a.get(0) == b.get(0));
        REQUIRE(a.get(1) == b.get(1));
    }
}

TEST_CASE("make_combined_switch_fn — realistic: PSFB primary + sync rectifier",
          "[v2][layer2_v10][combined_switch_fn][integration]") {
    // Primary: 4-switch full bridge driven by V9 helper on
    // bits 0..3. Synchronous rectifier secondary: HS/LS pair
    // driven by V6 helper on bits 4..5 (at 2× carrier so
    // the SR follows the rectified secondary square wave).
    constexpr Size n_sw = 6;
    auto primary = make_phase_shift_full_bridge_fn(
        100e3, std::numbers::pi_v<Real> * Real{0.5},
        0, 1, 2, 3, n_sw, /*dt=*/100e-9);
    auto sync_rect = make_dead_time_pwm_pair_fn(
        200e3, 0.5, /*hs=*/4, /*ls=*/5, n_sw,
        /*dt=*/100e-9);

    auto combo = make_combined_switch_fn(
        n_sw, {primary, sync_rect});
    REQUIRE(static_cast<bool>(combo));

    // Sample across one primary period. Verify:
    //   * Bits 0..3 follow primary(t).
    //   * Bits 4..5 follow sync_rect(t).
    //   * No interaction between the two groups.
    constexpr Real T_primary = 1.0 / 100e3;
    for (int k = 0; k < 500; ++k) {
        const Real t = (k + 0.317) * 5.0 * T_primary / 500.0;
        auto m  = combo(t);
        auto mp = primary(t);
        auto ms = sync_rect(t);
        INFO("t=" << t);
        REQUIRE(m.size() == n_sw);
        REQUIRE(m.get(0) == mp.get(0));
        REQUIRE(m.get(1) == mp.get(1));
        REQUIRE(m.get(2) == mp.get(2));
        REQUIRE(m.get(3) == mp.get(3));
        REQUIRE(m.get(4) == ms.get(4));
        REQUIRE(m.get(5) == ms.get(5));
    }
}
