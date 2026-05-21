// =============================================================================
// Layer 2 V12 — PulseVoltageSource device model tests
// =============================================================================
//
// Validates:
//   * `value_at` returns v_initial before t_start, v_pulsed
//     during the pulse window, v_initial after (single-shot).
//   * Periodic mode wraps correctly.
//   * DevicePool stores params + branch-current row id.
//   * `compute_pulse_b_extra` produces the correct overlay.
//   * Integration: a step source driving an RC circuit
//     produces the classic exponential charge curve
//     V_C(t) = V·(1 − e^(−t/τ)).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/models/pulse_voltage_source.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/sources/pulse_b_extra.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>

using namespace pulsim::v2;
using namespace pulsim::v2::builder;
using namespace pulsim::v2::models;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::sources;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("PulseVoltageSource — single-shot pulse semantics",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    PulseVoltageSource::Params p{
        .v_initial = 0.0, .v_pulsed = 5.0,
        .t_start = 1.0, .pulse_width = 2.0,
        .period = 0.0};
    // Before t_start
    REQUIRE(PulseVoltageSource::value_at(p, 0.0)  == 0.0);
    REQUIRE(PulseVoltageSource::value_at(p, 0.99) == 0.0);
    // At t_start the pulse fires
    REQUIRE(PulseVoltageSource::value_at(p, 1.0)  == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 2.0)  == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 2.99) == 5.0);
    // After the pulse: back to baseline (single-shot)
    REQUIRE(PulseVoltageSource::value_at(p, 3.0)  == 0.0);
    REQUIRE(PulseVoltageSource::value_at(p, 100.0) == 0.0);
}

TEST_CASE("PulseVoltageSource — periodic mode wraps correctly",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    // 5 V high for 1 s every 4 s; first pulse at t=0.
    PulseVoltageSource::Params p{
        .v_initial = 0.0, .v_pulsed = 5.0,
        .t_start = 0.0, .pulse_width = 1.0,
        .period = 4.0};
    // First cycle
    REQUIRE(PulseVoltageSource::value_at(p, 0.0) == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 0.5) == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 1.0) == 0.0);
    REQUIRE(PulseVoltageSource::value_at(p, 3.5) == 0.0);
    // Second cycle (4 s later)
    REQUIRE(PulseVoltageSource::value_at(p, 4.0) == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 4.5) == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 5.0) == 0.0);
    // Third cycle
    REQUIRE(PulseVoltageSource::value_at(p, 8.0) == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 8.99) == 5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 9.0)  == 0.0);
}

TEST_CASE("PulseVoltageSource — rise/fall ramps interpolate linearly",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    // V_initial=0, V_pulsed=10, t_start=0, pulse_width=1s,
    // rise_time=0.5s, fall_time=0.5s. Total active = 2s.
    PulseVoltageSource::Params p{
        .v_initial = 0.0, .v_pulsed = 10.0,
        .t_start = 0.0, .pulse_width = 1.0,
        .period = 0.0,
        .rise_time = 0.5, .fall_time = 0.5};
    // During rise: linearly interpolate.
    REQUIRE(PulseVoltageSource::value_at(p, 0.0)  == Approx(0.0));
    REQUIRE(PulseVoltageSource::value_at(p, 0.25) == Approx(5.0));
    REQUIRE(PulseVoltageSource::value_at(p, 0.5)  == Approx(10.0));
    // During hold.
    REQUIRE(PulseVoltageSource::value_at(p, 1.0)  == Approx(10.0));
    REQUIRE(PulseVoltageSource::value_at(p, 1.49) == Approx(10.0));
    // During fall.
    REQUIRE(PulseVoltageSource::value_at(p, 1.75) == Approx(5.0));
    // After fall.
    REQUIRE(PulseVoltageSource::value_at(p, 2.5)  == Approx(0.0));
}

TEST_CASE("PulseVoltageSource — rise_time=0 preserves V12 behavior",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    // Backwards-compat: no ramp → instantaneous step.
    PulseVoltageSource::Params p{
        .v_initial = 0.0, .v_pulsed = 5.0,
        .t_start = 1.0, .pulse_width = 2.0};
    REQUIRE(PulseVoltageSource::value_at(p, 0.5) == Approx(0.0));
    REQUIRE(PulseVoltageSource::value_at(p, 1.0) == Approx(5.0));
    REQUIRE(PulseVoltageSource::value_at(p, 2.5) == Approx(5.0));
    REQUIRE(PulseVoltageSource::value_at(p, 3.5) == Approx(0.0));
}

TEST_CASE("PulseVoltageSource — negative-going pulse",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    PulseVoltageSource::Params p{
        .v_initial = 12.0, .v_pulsed = -5.0,
        .t_start = 0.0, .pulse_width = 1.0,
        .period = 0.0};
    REQUIRE(PulseVoltageSource::value_at(p, -1.0) == 12.0);
    REQUIRE(PulseVoltageSource::value_at(p, 0.0)  == -5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 0.5)  == -5.0);
    REQUIRE(PulseVoltageSource::value_at(p, 1.5)  == 12.0);
}

TEST_CASE("DevicePool stores PulseVoltageSource params + branch-var",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    CircuitBuilder b;
    b.add_pulse_voltage_source(
        "Vpulse", "n0", "gnd",
        /*v_initial=*/0.0, /*v_pulsed=*/5.0,
        /*t_start=*/1e-3, /*pulse_width=*/2e-3);

    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
            DevicePool::StoredKind::PulseVoltageSource);
    REQUIRE(b.pool().pulse_voltage_source_params(0)
                .v_pulsed == Approx(5.0));
    REQUIRE(b.pool().pulse_voltage_source_params(0)
                .t_start == Approx(1e-3));
    // state_size = 1 node + 1 source branch-current = 2.
    REQUIRE(b.pool().state_size(b.graph()) == 2);
}

TEST_CASE("compute_pulse_b_extra produces correct overlay",
          "[v2][layer2_v12][pulse_voltage_source][unit]") {
    CircuitBuilder b;
    b.add_pulse_voltage_source(
        "Vpulse", "n0", "gnd",
        0.0, 10.0, 1.0, 2.0);

    // Before pulse: b_extra[src_var] = -v_initial = 0.
    Vector b_extra =
        compute_pulse_b_extra(b.pool(), b.graph(), 0.5);
    // src_var = num_nodes (1) + relative_offset (0) = 1.
    REQUIRE(b_extra[1] == Approx(0.0));

    // During pulse: b_extra[src_var] = -v_pulsed = -10.
    b_extra = compute_pulse_b_extra(b.pool(), b.graph(), 2.0);
    REQUIRE(b_extra[1] == Approx(-10.0));

    // After pulse: back to 0.
    b_extra = compute_pulse_b_extra(b.pool(), b.graph(), 5.0);
    REQUIRE(b_extra[1] == Approx(0.0));
}

// -----------------------------------------------------------------------------
// Integration: a step source charging an RC circuit produces
// the textbook exponential V_C(t) = V_step · (1 − e^(−t/τ)).
// -----------------------------------------------------------------------------

TEST_CASE("Pulse source step charges an RC circuit exponentially",
          "[v2][layer2_v12][pulse_voltage_source][integration]") {
    constexpr Real V_step = 10.0;
    constexpr Real R      = 1000.0;
    constexpr Real C      = 1e-6;
    constexpr Real tau    = R * C;        // 1 ms

    CircuitBuilder b;
    // Single-shot step: 0 → 10 V at t=0, pulse_width very
    // large so it stays high for the whole simulation.
    b.add_pulse_voltage_source(
        "Vstep", "n0", "gnd",
        0.0, V_step, /*t_start=*/0.0,
        /*pulse_width=*/1.0);
    b.add_resistor("R", "n0", "vc", R);
    b.add_capacitor("C", "vc", "gnd", C);

    constexpr Real dt   = 1e-6;
    constexpr Real tend = 5.0 * tau;     // 5 τ → ~99% settled

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };
    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);

    // Sample V_C at several time points and compare with
    // analytical V_C(t) = V·(1 - exp(-t/τ)).
    const Index vc_idx = b.node_id_of("vc");
    REQUIRE(vc_idx >= 0);

    Real max_err = 0;
    for (Real t_check : {tau, 2.0 * tau, 3.0 * tau,
                          4.0 * tau}) {
        const Size k = static_cast<Size>(t_check / dt);
        const Real v_sim = result.states[k][vc_idx];
        const Real v_exp =
            V_step * (Real{1} - std::exp(-t_check / tau));
        const Real err = std::abs(v_sim - v_exp);
        INFO("t=" << t_check << " v_sim=" << v_sim
             << " v_exp=" << v_exp << " err=" << err);
        max_err = std::max(max_err, err);
    }
    // Trapezoidal integration with dt = τ/1000 → accuracy
    // far better than 0.05 V.
    REQUIRE(max_err < 0.05);
}
