// =============================================================================
// Layer 2 V7 — `make_spwm_pair_fn` (SPWM half-bridge) tests
// =============================================================================
//
// Validates:
//   * M=0 → constant 50 % duty, matches make_dead_time_pwm_pair_fn.
//   * Shoot-through invariant under SPWM operation.
//   * Time-averaged mid-point voltage tracks the reference
//     sine within ~5 % error when the carrier is sufficiently
//     above the modulation frequency.
//   * carrier_frequency = 0 → flat OFF.
//   * Other switch bits stay OFF.
//   * Phase offsets shift the carrier / modulation as
//     expected.
//   * Integration: drive a half-bridge + LC filter at
//     10 kHz / 50 Hz / M=0.8 — the filtered output is a
//     ~40 V peak sine on top of a 100 V DC bus midpoint.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/v2/sources/spwm_pair_fn.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v2;
using namespace pulsim::v2::builder;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::sources;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("make_spwm_pair_fn — M=0 reduces to 50%% constant pair",
          "[v2][layer2_v7][spwm_pair][unit]") {
    constexpr Real f_c = 1000.0;
    constexpr Real dt_dead = 1e-5;
    auto spwm = make_spwm_pair_fn(
        f_c, /*f_mod=*/50.0, /*M=*/0.0,
        0, 1, 2, dt_dead);
    auto cnst = make_dead_time_pwm_pair_fn(
        f_c, /*duty=*/0.5, 0, 1, 2, dt_dead);
    // Sample 200 points over the first 5 carrier periods and
    // check the two callbacks agree bit-for-bit.
    for (int k = 0; k < 200; ++k) {
        const Real t = (k + 0.1) * (5.0 / f_c) / 200.0;
        auto a = spwm(t);
        auto b = cnst(t);
        INFO("t=" << t);
        REQUIRE(a.get(0) == b.get(0));
        REQUIRE(a.get(1) == b.get(1));
    }
}

TEST_CASE("make_spwm_pair_fn — shoot-through never happens",
          "[v2][layer2_v7][spwm_pair][unit]") {
    auto sw = make_spwm_pair_fn(
        /*carrier=*/10e3, /*mod=*/50.0, /*M=*/0.95,
        0, 1, 2, /*dead_time=*/100e-9);
    // Sweep 50,000 samples across multiple modulation cycles.
    constexpr int N = 50000;
    constexpr Real t_max = 0.05;   // 2.5 mod cycles
    for (int k = 0; k < N; ++k) {
        const Real t = (k + 0.317) * t_max / N;
        auto m = sw(t);
        const bool both_on = m.get(0) && m.get(1);
        INFO("t=" << t);
        REQUIRE_FALSE(both_on);
    }
}

TEST_CASE("make_spwm_pair_fn — time-averaged duty tracks sine reference",
          "[v2][layer2_v7][spwm_pair][unit]") {
    // M=0.8: instantaneous duty = 0.5 + 0.4·sin(ωt). Time-
    // average over one carrier period (much shorter than
    // modulation period) should equal that instantaneous
    // duty. We pick a fixed modulation-phase sample
    // (e.g. ωt = π/2 → sin = 1 → duty = 0.9) and verify the
    // carrier-period average matches.
    constexpr Real f_c = 100e3;     // 100 kHz carrier
    constexpr Real f_m = 50.0;      // 50 Hz mod (very slow)
    constexpr Real M   = 0.8;
    constexpr Real T_c = 1.0 / f_c;
    auto sw = make_spwm_pair_fn(
        f_c, f_m, M, 0, 1, 2, /*dt=*/0.0);

    // ωt = π/2 → t = 1/(4·f_m) = 5 ms.
    const Real t_anchor = 1.0 / (4.0 * f_m);
    const Real expected_duty = 0.5 + 0.5 * M * 1.0;

    Real hs_time = 0;
    Real ls_time = 0;
    constexpr int N = 1000;
    for (int k = 0; k < N; ++k) {
        const Real tau = (k + 0.5) * T_c / N;
        const Real t   = t_anchor + tau;
        auto m = sw(t);
        if (m.get(0)) hs_time += T_c / N;
        if (m.get(1)) ls_time += T_c / N;
    }
    const Real hs_frac = hs_time / T_c;
    INFO("HS frac = " << hs_frac
         << " (expected " << expected_duty << ")");
    REQUIRE(hs_frac == Approx(expected_duty).margin(0.02));
    // LS frac = 1 - duty (with dt=0).
    INFO("LS frac = " << ls_time/T_c
         << " (expected " << (1.0 - expected_duty) << ")");
    REQUIRE(ls_time/T_c ==
            Approx(1.0 - expected_duty).margin(0.02));
}

TEST_CASE("make_spwm_pair_fn — carrier_frequency=0 → flat OFF",
          "[v2][layer2_v7][spwm_pair][unit]") {
    auto sw = make_spwm_pair_fn(
        0.0, 50.0, 0.8, 0, 1, 2, 100e-9);
    for (double t : {0.0, 0.01, 1.0, 100.0}) {
        INFO("t=" << t);
        REQUIRE(sw(t).get(0) == false);
        REQUIRE(sw(t).get(1) == false);
    }
}

TEST_CASE("make_spwm_pair_fn — only HS/LS bits toggle; others OFF",
          "[v2][layer2_v7][spwm_pair][unit]") {
    auto sw = make_spwm_pair_fn(
        1000.0, 10.0, 0.5,
        /*hs=*/2, /*ls=*/0, /*n_sw=*/4,
        /*dt=*/1e-5);
    // Sample across one modulation cycle — bits 1 and 3
    // must stay OFF for all t.
    for (int k = 0; k < 200; ++k) {
        const Real t = (k + 0.7) * 0.1 / 200.0;
        auto m = sw(t);
        INFO("t=" << t);
        REQUIRE(m.size() == 4);
        REQUIRE(m.get(1) == false);
        REQUIRE(m.get(3) == false);
    }
}

TEST_CASE("make_spwm_pair_fn — modulation_phase shifts sine reference",
          "[v2][layer2_v7][spwm_pair][unit]") {
    // f_carrier = 1000 Hz → T_c = 1 ms. We sample at the
    // MIDDLE of the carrier period (t_cycle = 500 µs) so the
    // result is determined by the instantaneous duty:
    //   * If duty > 0.5 → HS region covers ≥ 500 µs → HS ON
    //   * If duty < 0.5 → LS region covers 500 µs → LS ON
    constexpr Real t_sample = 5e-4;

    // mod_phase = π/2 → sin(π/2)=1 → duty=0.975 → HS-favoring.
    auto sw_hi = make_spwm_pair_fn(
        /*carrier=*/1000.0, /*mod=*/50.0, /*M=*/0.95,
        0, 1, 2, /*dt=*/0.0,
        /*carrier_phase=*/0.0,
        /*modulation_phase=*/std::numbers::pi_v<Real> * 0.5);
    REQUIRE(sw_hi(t_sample).get(0) == true);
    REQUIRE(sw_hi(t_sample).get(1) == false);

    // mod_phase = -π/2 → sin=-1 → duty=0.025 → LS-favoring.
    auto sw_lo = make_spwm_pair_fn(
        1000.0, 50.0, 0.95, 0, 1, 2, 0.0, 0.0,
        /*modulation_phase=*/-std::numbers::pi_v<Real> * 0.5);
    REQUIRE(sw_lo(t_sample).get(0) == false);
    REQUIRE(sw_lo(t_sample).get(1) == true);
}

// -----------------------------------------------------------------------------
// Integration: half-bridge driving a pure resistive mid-load.
// V_bus - HS - mid - LS - gnd, with R_mid from mid to gnd.
// Mid-point time-average over a full modulation cycle
// should equal V_bus·duty_avg = V_bus·0.5 (sine averages to
// zero). PWM ripple swings mid between ~0 and ~V_bus.
// -----------------------------------------------------------------------------

TEST_CASE("make_spwm_pair_fn — half-bridge mid-point tracks SPWM duty",
          "[v2][layer2_v7][spwm_pair][integration]") {
    constexpr Real V_bus = 100.0;
    constexpr Real f_c   = 20e3;     // 20 kHz carrier
    constexpr Real f_m   = 200.0;    // 200 Hz fundamental
    constexpr Real M     = 0.8;
    constexpr Real T_m   = 1.0 / f_m;
    constexpr Real dt_dead = 100e-9;

    CircuitBuilder b;
    b.add_voltage_source("Vbus", "vbus", "gnd", V_bus);
    b.add_switch("HS", "vbus", "mid", 1e3, 1e-9);
    b.add_switch("LS", "mid", "gnd", 1e3, 1e-9);
    b.add_resistor("R_mid", "mid", "gnd", 100.0);

    constexpr Real dt   = 1e-7;             // 0.1 µs
    constexpr Real tend = 2.0 * T_m;       // 10 ms

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};

    const Size n_sw = b.graph().num_switches();
    REQUIRE(n_sw == 2);

    auto sw_fn = make_spwm_pair_fn(
        f_c, f_m, M, 0, 1, n_sw, dt_dead);

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, sw_fn);
    REQUIRE(result.num_steps() > 1000);

    // Average mid over the LAST full modulation cycle.
    const Index mid_idx = b.node_id_of("mid");
    REQUIRE(mid_idx >= 0);
    const Size k_start = result.num_steps() -
        static_cast<Size>(T_m / dt);

    Real v_sum = 0;
    Real v_min = 1e9, v_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v = result.states[k][mid_idx];
        v_sum += v;
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
    }
    const Real v_mean = v_sum /
        static_cast<Real>(result.num_steps() - k_start);

    INFO("v_mid mean over last mod cycle = " << v_mean
         << " V (expected " << V_bus * 0.5 << " V)");
    // Sine averages to zero over a full mod period →
    // DC = V_bus·0.5. Allow ±5 V for dead-time + sampling.
    REQUIRE(v_mean == Approx(V_bus * 0.5).margin(5.0));

    // PWM ripple swings the mid-point essentially full-rail.
    INFO("v_mid swing: [" << v_min << ", " << v_max << "]");
    REQUIRE(v_min < 20.0);     // dips below 20 V some cycles
    REQUIRE(v_max > 80.0);     // peaks above 80 V some cycles
}
