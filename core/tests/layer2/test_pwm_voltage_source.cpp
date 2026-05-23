// =============================================================================
// Layer 2 V4 — PWMVoltageSource device model tests
// =============================================================================
//
// Validates the new PWMVoltageSource:
//   * Model `value_at(p, t)` returns v_high / v_low at the
//     right phases.
//   * DevicePool stores and retrieves PWM params.
//   * `add_pwm_voltage_source` adds a Source-kind branch
//     with a branch-current unknown (same as VoltageSource).
//   * `compute_pwm_b_extra` produces the correct overlay.
//   * Integration: PWM source driving a resistor through
//     `run_transient` reproduces the expected square wave
//     at the load node (built-in PWM pass works).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::models;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::sources;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("PWMVoltageSource::value_at returns v_high in ON portion",
          "[v2][layer2_v4][pwm_voltage_source][unit]") {
    PWMVoltageSource::Params p{
        .v_high = 24.0,
        .v_low  = 0.0,
        .frequency = 1.0,    // T = 1 s
        .duty   = 0.5,
        .phase  = 0.0,
    };

    // At t=0: phase01 = 0 → ON.
    REQUIRE(PWMVoltageSource::value_at(p, 0.0) ==
              Approx(24.0));
    // At t=0.25 (25 % of period): still ON.
    REQUIRE(PWMVoltageSource::value_at(p, 0.25) ==
              Approx(24.0));
    // At t=0.5: phase01 = 0.5 = duty → boundary → OFF
    // (`< duty` is exclusive).
    REQUIRE(PWMVoltageSource::value_at(p, 0.5) ==
              Approx(0.0));
    // At t=0.75: OFF.
    REQUIRE(PWMVoltageSource::value_at(p, 0.75) ==
              Approx(0.0));
    // At t=1.0: new cycle → ON again.
    REQUIRE(PWMVoltageSource::value_at(p, 1.0) ==
              Approx(24.0));
}

TEST_CASE("PWMVoltageSource handles phase offset correctly",
          "[v2][layer2_v4][pwm_voltage_source][unit]") {
    PWMVoltageSource::Params p{
        .v_high = 5.0,
        .v_low  = 0.0,
        .frequency = 1.0,
        .duty   = 0.5,
        .phase  = 0.25,   // shift cycle by 1/4 period
    };
    // At t=0.0 (= phase -0.25 → wraps to 0.75): OFF.
    REQUIRE(PWMVoltageSource::value_at(p, 0.0) ==
              Approx(0.0));
    // At t=0.25 (= phase 0): ON.
    REQUIRE(PWMVoltageSource::value_at(p, 0.25) ==
              Approx(5.0));
    // At t=0.7 (= phase 0.45): ON.
    REQUIRE(PWMVoltageSource::value_at(p, 0.7) ==
              Approx(5.0));
    // At t=0.8 (= phase 0.55 > duty=0.5): OFF.
    REQUIRE(PWMVoltageSource::value_at(p, 0.8) ==
              Approx(0.0));
}

TEST_CASE("PWMVoltageSource handles duty=0 and duty=1 edge cases",
          "[v2][layer2_v4][pwm_voltage_source][unit]") {
    // duty=0: always OFF.
    PWMVoltageSource::Params zero{.v_high=10, .v_low=0,
                                    .frequency=1.0,
                                    .duty=0.0, .phase=0};
    REQUIRE(PWMVoltageSource::value_at(zero, 0.0) ==
              Approx(0.0));
    REQUIRE(PWMVoltageSource::value_at(zero, 0.5) ==
              Approx(0.0));

    // duty=1: always ON.
    PWMVoltageSource::Params one{.v_high=10, .v_low=0,
                                   .frequency=1.0,
                                   .duty=1.0, .phase=0};
    REQUIRE(PWMVoltageSource::value_at(one, 0.0) ==
              Approx(10.0));
    REQUIRE(PWMVoltageSource::value_at(one, 0.99) ==
              Approx(10.0));
}

TEST_CASE("PWMVoltageSource handles frequency=0 safely",
          "[v2][layer2_v4][pwm_voltage_source][unit]") {
    PWMVoltageSource::Params p{.v_high=10, .v_low=2,
                                 .frequency=0.0,
                                 .duty=0.5, .phase=0};
    // Degenerate case → returns v_low (no division by zero).
    REQUIRE(PWMVoltageSource::value_at(p, 1.0) ==
              Approx(2.0));
    REQUIRE(PWMVoltageSource::value_at(p, 100.0) ==
              Approx(2.0));
}

TEST_CASE("DevicePool stores PWMVoltageSource params + branch-var",
          "[v2][layer2_v4][pwm_voltage_source][unit]") {
    CircuitBuilder b;
    b.add_pwm_voltage_source(
        "VPWM", "n0", "gnd",
        /*v_high=*/24.0, /*v_low=*/0.0,
        /*frequency=*/100e3, /*duty=*/0.5);

    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::PWMVoltageSource);
    REQUIRE(b.pool().pwm_voltage_source_params(0).v_high ==
              Approx(24.0));
    REQUIRE(b.pool().pwm_voltage_source_params(0).duty ==
              Approx(0.5));

    // PWM sources ADD a branch-current unknown (same as
    // VoltageSource). state_size should include it.
    // 1 node + 1 source = 2.
    REQUIRE(b.pool().state_size(b.graph()) == 2);
}

TEST_CASE("compute_pwm_b_extra produces correct overlay",
          "[v2][layer2_v4][pwm_voltage_source][unit]") {
    CircuitBuilder b;
    b.add_pwm_voltage_source(
        "VPWM", "n0", "gnd",
        /*v_high=*/24.0, /*v_low=*/0.0,
        /*frequency=*/1.0, /*duty=*/0.5);

    // At t=0.0 (ON portion): b_extra[src_var] = -v_high.
    Vector b_extra = compute_pwm_b_extra(
        b.pool(), b.graph(), 0.0);
    // src_var = num_nodes (1) + relative_offset (0) = 1.
    REQUIRE(b_extra[1] == Approx(-24.0));

    // At t=0.75 (OFF portion): b_extra[src_var] = 0.
    b_extra = compute_pwm_b_extra(b.pool(), b.graph(), 0.75);
    REQUIRE(b_extra[1] == Approx(0.0));
}

// -----------------------------------------------------------------------------
// Integration: PWM source driving a resistor through
// run_transient. The output should track the PWM waveform.
// -----------------------------------------------------------------------------

TEST_CASE("PWM source drives a resistor — output tracks square wave",
          "[v2][layer2_v4][pwm_voltage_source][integration]") {
    constexpr Real V_high = 24.0;
    constexpr Real V_low  = 0.0;
    constexpr Real f_sw   = 100e3;       // 100 kHz
    constexpr Real T_sw   = 1.0 / f_sw;
    constexpr Real duty   = 0.5;

    CircuitBuilder b;
    b.add_pwm_voltage_source(
        "VPWM", "n0", "gnd",
        V_high, V_low, f_sw, duty);
    b.add_resistor("R_L", "n0", "gnd", 10.0);

    constexpr Real dt   = 1e-7;
    constexpr Real tend = 3.0 * T_sw;

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);

    REQUIRE(result.num_steps() > 100);

    // Sample some "should be ON" times — v_n0 ≈ V_high.
    // Sample some "should be OFF" times — v_n0 ≈ V_low.
    Size n_on = 0, n_on_correct = 0;
    Size n_off = 0, n_off_correct = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        const Real t = result.times[k];
        // Use the same model to determine expected V at t.
        const PWMVoltageSource::Params pp{
            .v_high = V_high, .v_low = V_low,
            .frequency = f_sw, .duty = duty,
            .phase = 0.0};
        const Real v_expected =
            PWMVoltageSource::value_at(pp, t);
        const Real v_n0 = result.states[k][0];

        if (v_expected == V_high) {
            ++n_on;
            if (std::abs(v_n0 - V_high) < 0.1) {
                ++n_on_correct;
            }
        } else {
            ++n_off;
            if (std::abs(v_n0 - V_low) < 0.1) {
                ++n_off_correct;
            }
        }
    }

    INFO("PWM ON: " << n_on_correct << "/" << n_on
         << " match, OFF: " << n_off_correct << "/" << n_off);
    REQUIRE(n_on > 0);
    REQUIRE(n_off > 0);
    // > 99 % match each side (boundary samples might fall
    // exactly on duty transition — small slack).
    REQUIRE(n_on_correct * 100 >= n_on * 99);
    REQUIRE(n_off_correct * 100 >= n_off * 99);
}

TEST_CASE("PWM duty=0.25 → mean output = V_high · 0.25",
          "[v2][layer2_v4][pwm_voltage_source][integration]") {
    constexpr Real V_high = 24.0;
    constexpr Real V_low  = 0.0;
    constexpr Real f_sw   = 100e3;
    constexpr Real T_sw   = 1.0 / f_sw;
    constexpr Real duty   = 0.25;

    CircuitBuilder b;
    b.add_pwm_voltage_source(
        "VPWM", "n0", "gnd",
        V_high, V_low, f_sw, duty);
    b.add_resistor("R_L", "n0", "gnd", 10.0);

    constexpr Real dt   = 1e-8;            // fine dt
    constexpr Real tend = 5.0 * T_sw;

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);

    // Time-average v_n0 over the last cycle.
    const Size k_start = result.num_steps() -
        static_cast<Size>(T_sw / dt);
    Real sum = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        sum += result.states[k][0];
    }
    const Real mean = sum /
        static_cast<Real>(result.num_steps() - k_start);
    const Real expected = V_high * duty;

    INFO("Mean v_n0 = " << mean << " V (expected "
         << expected << " V)");
    // Mean should match V_high · duty within ~1 %.
    REQUIRE(mean == Approx(expected).margin(0.5));
}
