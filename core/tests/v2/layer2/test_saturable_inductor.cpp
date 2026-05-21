// =============================================================================
// Layer 2 V16 — SaturableInductor model unit tests
// =============================================================================
//
// V0 scope: validate the L(i) function (used by future
// transient-integration work). Tests the smooth saturation
// curve: L → L_0 at low |i|, L → L_residual at high |i|,
// monotonically decreasing in |i|, C¹ smooth.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/models/saturable_inductor.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/solver/run_transient.hpp"

#include <algorithm>
#include <cmath>

using namespace pulsim::v2;
using namespace pulsim::v2::models;
using Catch::Approx;

namespace {

Real L_at(const SaturableInductor::Params& p, Real i) {
    const Real iv[1] = {i};
    return SaturableInductor::current<Real>(iv, p);
}

}  // namespace

TEST_CASE("SaturableInductor — at i=0 returns L_0",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    SaturableInductor::Params p{
        .L_0 = 1e-3, .I_sat = 5.0,
        .L_residual = 0.0};
    REQUIRE(L_at(p, 0.0) == Approx(1e-3));
}

TEST_CASE("SaturableInductor — at |i| = I_sat returns L_0 / 2",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    SaturableInductor::Params p{
        .L_0 = 1e-3, .I_sat = 5.0,
        .L_residual = 0.0};
    // L(I_sat) = L_0 / (1 + 1) = L_0 / 2.
    REQUIRE(L_at(p, 5.0) == Approx(0.5e-3));
    REQUIRE(L_at(p, -5.0) == Approx(0.5e-3));   // symmetric in |i|
}

TEST_CASE("SaturableInductor — high |i| approaches L_residual",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    SaturableInductor::Params p{
        .L_0 = 1e-3, .I_sat = 5.0,
        .L_residual = 1e-5};
    // At i = 50 A (10x I_sat): L = L_residual + (L_0 - L_residual)/(1+100)
    //                         ≈ 1e-5 + 0.99e-3/101 ≈ 1e-5 + 9.8e-6 ≈ 1.98e-5
    const Real L_high = L_at(p, 50.0);
    REQUIRE(L_high < 0.1e-3);
    REQUIRE(L_high > p.L_residual);
}

TEST_CASE("SaturableInductor — monotonically decreasing in |i|",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    SaturableInductor::Params p{
        .L_0 = 1e-3, .I_sat = 5.0,
        .L_residual = 0.0};
    Real prev = L_at(p, 0.0);
    for (Real i = 0.1; i <= 20.0; i += 0.5) {
        const Real L = L_at(p, i);
        REQUIRE(L < prev);
        prev = L;
    }
}

// (V17 simplifies: fixed n=2 Atan saturation. Removed the
// "sharper knee with higher n" test that depended on tunable n.)

// -----------------------------------------------------------------------------
// V17 Newton-integration tests.
// -----------------------------------------------------------------------------
//
// Circuit: V_step (10V) ── L_sat ── R ── gnd
// With L_0 = 1 mH, I_sat = 0.5 A, R = 10 Ω:
//   Below I_sat: τ ≈ L_0/R = 100 µs, steady-state I = V/R = 1 A.
//   But I tries to settle at 1 A, which exceeds I_sat — L saturates,
//   the inductor "shrinks" and the dynamics speed up. Final I = 1 A
//   (resistive limit).
// -----------------------------------------------------------------------------

TEST_CASE("SaturableInductor — transient: current settles at resistive limit",
          "[v2][layer2_v17][saturable_inductor][integration]") {
    using namespace pulsim::v2::builder;
    using namespace pulsim::v2::pwl;
    using namespace pulsim::v2::solver;
    using namespace pulsim::v2::topology;

    constexpr Real V_step = 10.0;
    constexpr Real R = 10.0;
    constexpr Real L_0 = 1e-3;
    constexpr Real I_sat = 0.5;

    CircuitBuilder b;
    b.add_voltage_source("V1", "vin", "gnd", V_step);
    b.add_saturable_inductor(
        "L_sat", "vin", "mid", L_0, I_sat);
    b.add_resistor("R1", "mid", "gnd", R);

    constexpr Real dt   = 1e-6;
    constexpr Real tend = 5e-3;   // ~50·τ_unsaturated

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt,
        .max_newton_iterations = 100};
    opts.enable_newton_line_search = true;
    auto sw_fn = [](Real) { return SwitchStateMask(0); };

    // Pass an EMPTY nl_refresh — the run_transient wrapper
    // will auto-detect saturable inductors and add the
    // refresh internally.
    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, sw_fn,
        /*b_extra_fn=*/{}, /*start_from_dc_op=*/false,
        /*nl_refresh=*/{});
    REQUIRE(result.num_steps() > 100);

    // Final current = V / R = 1 A (above I_sat, but resistor
    // limits it). Saturation just speeds up the transient.
    // The saturable inductor has branch_id = 1 (added after
    // the voltage source); look up its actual state-vector row.
    const Index l_branch_var =
        b.pool().branch_var_id_for_inductor(1, b.graph());
    const Real i_L_final =
        result.states[result.num_steps() - 1][l_branch_var];
    INFO("Final I_L = " << i_L_final << " A (expected 1 A)");
    REQUIRE(i_L_final == Approx(1.0).margin(0.05));

    // At an intermediate time, the saturable inductor should
    // reach steady state FASTER than the unsaturated version
    // would (τ shrinks when L decreases). Specifically, with
    // L_0 = 1 mH unsaturated → τ = 100 µs. With saturation
    // kicking in around 0.5 A, τ_effective drops. Check that
    // by 1 ms (10·τ_unsaturated), I is essentially at final.
    const Size k_1ms = static_cast<Size>(1e-3 / dt);
    const Real i_at_1ms = result.states[k_1ms][l_branch_var];
    REQUIRE(i_at_1ms == Approx(1.0).margin(0.1));
}
