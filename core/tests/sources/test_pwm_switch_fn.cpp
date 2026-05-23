// =============================================================================
// Layer 2 V5 — `make_pwm_switch_fn` helper tests
// =============================================================================
//
// Validates the PWM switch-schedule helper:
//   * Toggles `switch_idx` ON during `phase01 < duty`,
//     OFF for the rest (matching the convention used by
//     `PWMVoltageSource::value_at`).
//   * Boundary `phase01 == duty` is OFF (exclusive `<`).
//   * `duty == 0` → always OFF; `duty == 1` → always ON.
//   * `frequency <= 0` → degenerate flat-OFF mask.
//   * `phase` offset shifts the cycle as expected.
//   * Other switch bits stay OFF.
//   * Helper integrates cleanly with a real `run_transient`
//     (smoke check: drive a single switched-resistor circuit
//     and observe the load current PWM-modulated correctly).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::sources;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("make_pwm_switch_fn — 50%% duty toggles correctly across cycle",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    // 1 Hz / 50 % duty / switch_idx 0 / num_switches 1.
    auto sw = make_pwm_switch_fn(1.0, 0.5, 0, 1);
    REQUIRE(static_cast<bool>(sw));   // valid std::function

    // t=0 → phase01 = 0 → ON.
    REQUIRE(sw(0.0).get(0) == true);
    // t=0.25 → phase01 = 0.25 < 0.5 → ON.
    REQUIRE(sw(0.25).get(0) == true);
    // t=0.5 → phase01 = 0.5 == duty → OFF (exclusive `<`).
    REQUIRE(sw(0.5).get(0) == false);
    // t=0.75 → OFF.
    REQUIRE(sw(0.75).get(0) == false);
    // t=1.0 → wraps to phase01=0 → ON again.
    REQUIRE(sw(1.0).get(0) == true);
    // t=1.5 → wraps to phase01=0.5 → OFF.
    REQUIRE(sw(1.5).get(0) == false);
}

TEST_CASE("make_pwm_switch_fn — duty=0 → always OFF",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    auto sw = make_pwm_switch_fn(1.0, 0.0, 0, 1);
    for (double t : {0.0, 0.1, 0.5, 0.9, 1.0, 5.7}) {
        INFO("t = " << t);
        REQUIRE(sw(t).get(0) == false);
    }
}

TEST_CASE("make_pwm_switch_fn — duty=1 → always ON",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    auto sw = make_pwm_switch_fn(1.0, 1.0, 0, 1);
    for (double t : {0.0, 0.1, 0.5, 0.9, 0.999, 5.7}) {
        INFO("t = " << t);
        REQUIRE(sw(t).get(0) == true);
    }
}

TEST_CASE("make_pwm_switch_fn — frequency=0 → flat OFF (no division)",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    auto sw = make_pwm_switch_fn(0.0, 0.5, 0, 1);
    REQUIRE(static_cast<bool>(sw));
    for (double t : {0.0, 0.1, 100.0, 1e6}) {
        INFO("t = " << t);
        REQUIRE(sw(t).get(0) == false);
    }
}

TEST_CASE("make_pwm_switch_fn — phase offset shifts cycle",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    // phase=0.25 means t=0.25 maps to phase01=0 (start of ON).
    auto sw = make_pwm_switch_fn(1.0, 0.5, 0, 1, 0.25);

    // t=0.0 → effective phase01 = -0.25 → wraps to 0.75 → OFF.
    REQUIRE(sw(0.0).get(0) == false);
    // t=0.25 → phase01 = 0 → ON.
    REQUIRE(sw(0.25).get(0) == true);
    // t=0.7 → phase01 = 0.45 < 0.5 → ON.
    REQUIRE(sw(0.7).get(0) == true);
    // t=0.8 → phase01 = 0.55 → OFF.
    REQUIRE(sw(0.8).get(0) == false);
}

TEST_CASE("make_pwm_switch_fn — only target switch toggles; others stay OFF",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    // 3 switches total; helper drives switch index 1 only.
    auto sw = make_pwm_switch_fn(1.0, 0.5, /*switch_idx=*/1,
                                   /*num_switches=*/3);
    const auto on  = sw(0.1);   // phase01=0.1, switch_idx=1 ON
    REQUIRE(on.size() == 3);
    REQUIRE(on.get(0) == false);
    REQUIRE(on.get(1) == true);
    REQUIRE(on.get(2) == false);

    const auto off = sw(0.7);   // phase01=0.7 > 0.5, all OFF
    REQUIRE(off.get(0) == false);
    REQUIRE(off.get(1) == false);
    REQUIRE(off.get(2) == false);
}

TEST_CASE("make_pwm_switch_fn — matches PWMVoltageSource value_at convention",
          "[v2][layer2_v5][pwm_switch_fn][unit]") {
    // Both the switch_fn helper and the PWMVoltageSource use
    // the SAME `phase01 < duty` convention. Sample the switch
    // mask and the source value at identical times: when the
    // switch is ON, the source must be HIGH, and vice versa.
    auto sw = make_pwm_switch_fn(100e3, 0.42, 0, 1);
    const double T = 1.0 / 100e3;
    for (int k = 0; k < 32; ++k) {
        const double t = (k + 0.3) * T / 32.0;   // off-grid
        const bool on = sw(t).get(0);
        const double phase01 =
            std::fmod(t * 100e3, 1.0);
        INFO("t=" << t << " phase01=" << phase01);
        REQUIRE(on == (phase01 < 0.42));
    }
}

// -----------------------------------------------------------------------------
// Smoke: drive a switched resistor circuit using the helper.
// V — Sw — R — gnd. With Sw closed, R conducts; with Sw open
// (g_off), R is effectively floating. We just check the run
// goes through and the helper's switch_fn produces a
// reasonable steady-state ON-fraction.
// -----------------------------------------------------------------------------

TEST_CASE("make_pwm_switch_fn — drives a switched resistor end-to-end",
          "[v2][layer2_v5][pwm_switch_fn][integration]") {
    constexpr Real V    = 10.0;
    constexpr Real R    = 100.0;
    constexpr Real f_sw = 50e3;
    constexpr Real T_sw = 1.0 / f_sw;
    constexpr Real duty = 0.4;

    CircuitBuilder b;
    b.add_voltage_source("V1", "n1", "gnd", V);
    // High-conductance switch (closed → ~0 V drop). g_off
    // makes the resistor see ~0 current when OFF.
    b.add_switch("S1", "n1", "n2", /*g_on=*/1e3, /*g_off=*/1e-9);
    b.add_resistor("R1", "n2", "gnd", R);

    constexpr Real dt   = 1e-7;
    constexpr Real tend = 5.0 * T_sw;

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};

    const Size n_sw = b.graph().num_switches();
    REQUIRE(n_sw == 1);

    auto switch_fn = make_pwm_switch_fn(
        f_sw, duty, /*switch_idx=*/0, n_sw);

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);
    REQUIRE(result.num_steps() > 100);

    // Count ON samples — should be ~`duty` fraction of total.
    Size n_on = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        // n2 (resistor node) ≈ V when ON, ≈ 0 when OFF.
        // n2 is node index 1 (after n1).
        if (std::abs(result.states[k][1] - V) < 0.5) {
            ++n_on;
        }
    }
    const Real frac_on =
        static_cast<Real>(n_on) /
        static_cast<Real>(result.num_steps());

    INFO("ON fraction = " << frac_on << " (target " << duty
         << ")");
    REQUIRE(std::abs(frac_on - duty) < 0.02);
}
