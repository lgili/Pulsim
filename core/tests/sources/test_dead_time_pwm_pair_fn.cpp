// =============================================================================
// Layer 2 V6 — `make_dead_time_pwm_pair_fn` helper tests
// =============================================================================
//
// Validates the complementary HS/LS PWM helper with
// symmetric dead-time:
//   * dt=0: HS and LS are exact complements (no overlap, no
//     gap) — should match `make_pwm_switch_fn` driving HS
//     plus its inverse on LS.
//   * dt>0: HS turns OFF dt seconds BEFORE the nominal duty
//     boundary; LS turns ON at the nominal duty boundary
//     and turns OFF dt seconds before period end.
//   * Shoot-through invariant: HS and LS are NEVER both ON.
//   * Edge cases: duty=0/1, frequency=0, oversized dt.
//   * Phase offset shifts the entire cycle.
//   * Other switch bits stay OFF.
//   * Integration: drives a synchronous-buck-style RC load
//     through run_transient — output tracks expected duty.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/dead_time_pwm_pair_fn.hpp"
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

TEST_CASE("make_dead_time_pwm_pair_fn — dt=0 → exact complement",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    // 1 Hz / 50 % / dt=0: HS = first half, LS = second half,
    // no overlap, no gap.
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 0.5, /*hs=*/0, /*ls=*/1, /*n_sw=*/2,
        /*dt=*/0.0);
    REQUIRE(static_cast<bool>(sw));

    // t=0: HS ON, LS OFF.
    auto m0 = sw(0.0);
    REQUIRE(m0.get(0) == true);
    REQUIRE(m0.get(1) == false);

    // t=0.25: HS ON, LS OFF.
    auto m1 = sw(0.25);
    REQUIRE(m1.get(0) == true);
    REQUIRE(m1.get(1) == false);

    // t=0.5: nominal commutation. With dt=0 → HS OFF, LS ON.
    // (boundary `t_cycle < t_hs_end (=0.5)` is FALSE; LS
    // band `[0.5, 1.0)` is TRUE.)
    auto m2 = sw(0.5);
    REQUIRE(m2.get(0) == false);
    REQUIRE(m2.get(1) == true);

    // t=0.75: HS OFF, LS ON.
    auto m3 = sw(0.75);
    REQUIRE(m3.get(0) == false);
    REQUIRE(m3.get(1) == true);

    // t=1.0: wraps to start of cycle → HS ON, LS OFF.
    auto m4 = sw(1.0);
    REQUIRE(m4.get(0) == true);
    REQUIRE(m4.get(1) == false);
}

TEST_CASE("make_dead_time_pwm_pair_fn — dt>0 inserts gap before each commutation",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    // 1 Hz / 50 % / dt=0.05. Boundaries:
    //   HS region:    [0,    0.45)
    //   dead band:    [0.45, 0.5)
    //   LS region:    [0.5,  0.95)
    //   dead band:    [0.95, 1.0)
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 0.5, 0, 1, 2, /*dt=*/0.05);

    // Middle of HS: HS ON.
    REQUIRE(sw(0.2).get(0) == true);
    REQUIRE(sw(0.2).get(1) == false);

    // Inside falling dead-band: both OFF.
    REQUIRE(sw(0.47).get(0) == false);
    REQUIRE(sw(0.47).get(1) == false);

    // Right at LS start: LS ON.
    REQUIRE(sw(0.5).get(0) == false);
    REQUIRE(sw(0.5).get(1) == true);

    // Middle of LS: LS ON.
    REQUIRE(sw(0.7).get(0) == false);
    REQUIRE(sw(0.7).get(1) == true);

    // Inside rising dead-band: both OFF.
    REQUIRE(sw(0.97).get(0) == false);
    REQUIRE(sw(0.97).get(1) == false);

    // Just after wrap: HS ON again.
    REQUIRE(sw(0.01).get(0) == true);
    REQUIRE(sw(0.01).get(1) == false);
}

TEST_CASE("make_dead_time_pwm_pair_fn — shoot-through never happens",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    // Sweep dense samples across multiple cycles for several
    // duty values + dt. HS and LS must never be both ON.
    for (Real duty : {Real{0.1}, Real{0.25}, Real{0.5},
                       Real{0.75}, Real{0.9}}) {
        for (Real dt : {Real{0.0}, Real{1e-3}, Real{1e-2},
                         Real{0.05}}) {
            auto sw = make_dead_time_pwm_pair_fn(
                1.0, duty, 0, 1, 2, dt);
            for (int k = 0; k < 1000; ++k) {
                const Real t = (k + 0.123) / 100.0;  // off-grid
                auto m = sw(t);
                INFO("duty=" << duty << " dt=" << dt
                     << " t=" << t);
                // Strict invariant: not both ON. Compute
                // outside the macro because Catch2 forbids
                // `&&` inside REQUIRE_*.
                const bool both_on = m.get(0) && m.get(1);
                REQUIRE_FALSE(both_on);
            }
        }
    }
}

TEST_CASE("make_dead_time_pwm_pair_fn — duty=0 → HS never ON, LS most of cycle",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 0.0, 0, 1, 2, /*dt=*/0.05);
    // duty=0 → HS region [0, -0.05) which is empty.
    // LS region: [0, 0.95). Dead-band: [0.95, 1.0).
    for (double t : {0.0, 0.1, 0.5, 0.9}) {
        INFO("t=" << t);
        REQUIRE(sw(t).get(0) == false);
        REQUIRE(sw(t).get(1) == true);
    }
    // Inside trailing dead-band.
    REQUIRE(sw(0.97).get(0) == false);
    REQUIRE(sw(0.97).get(1) == false);
}

TEST_CASE("make_dead_time_pwm_pair_fn — duty=1 → HS most of cycle, LS never",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 1.0, 0, 1, 2, /*dt=*/0.05);
    // duty=1 → HS region [0, 0.95). LS region [1, 0.95) is
    // empty (start ≥ end). Trailing dead: [0.95, 1.0).
    for (double t : {0.0, 0.1, 0.5, 0.94}) {
        INFO("t=" << t);
        REQUIRE(sw(t).get(0) == true);
        REQUIRE(sw(t).get(1) == false);
    }
    REQUIRE(sw(0.97).get(0) == false);
    REQUIRE(sw(0.97).get(1) == false);
}

TEST_CASE("make_dead_time_pwm_pair_fn — frequency=0 → flat OFF",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    auto sw = make_dead_time_pwm_pair_fn(
        0.0, 0.5, 0, 1, 2, 0.05);
    for (double t : {0.0, 0.1, 100.0, 1e6}) {
        INFO("t=" << t);
        REQUIRE(sw(t).get(0) == false);
        REQUIRE(sw(t).get(1) == false);
    }
}

TEST_CASE("make_dead_time_pwm_pair_fn — oversized dt saturates to T/2",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    // dt = 2·T → clamped to T/2 = 0.5. HS region [0, 0.0) =
    // empty. LS region [0.5, 0.5) = empty. Both switches
    // always OFF — degenerate, but no crash / wrap bug.
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 0.5, 0, 1, 2, /*dt=*/2.0);
    for (double t : {0.0, 0.25, 0.5, 0.75, 0.99}) {
        INFO("t=" << t);
        REQUIRE(sw(t).get(0) == false);
        REQUIRE(sw(t).get(1) == false);
    }
}

TEST_CASE("make_dead_time_pwm_pair_fn — phase offset shifts entire cycle",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 0.5, 0, 1, 2, /*dt=*/0.05,
        /*phase=*/0.25);
    // Effective t=0 maps to t_cycle = -0.25 → wraps to 0.75
    // → LS region. Expect LS ON, HS OFF.
    REQUIRE(sw(0.0).get(0) == false);
    REQUIRE(sw(0.0).get(1) == true);

    // Effective t=0.25 maps to t_cycle = 0 → HS region.
    REQUIRE(sw(0.25).get(0) == true);
    REQUIRE(sw(0.25).get(1) == false);
}

TEST_CASE("make_dead_time_pwm_pair_fn — only HS/LS bits toggle; others OFF",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    // 4 switches; helper drives bits 1 and 3 only.
    auto sw = make_dead_time_pwm_pair_fn(
        1.0, 0.5, /*hs=*/1, /*ls=*/3,
        /*n_sw=*/4, /*dt=*/0.05);
    auto m_hs = sw(0.2);   // HS on, LS off.
    REQUIRE(m_hs.size() == 4);
    REQUIRE(m_hs.get(0) == false);
    REQUIRE(m_hs.get(1) == true);
    REQUIRE(m_hs.get(2) == false);
    REQUIRE(m_hs.get(3) == false);

    auto m_ls = sw(0.7);   // HS off, LS on.
    REQUIRE(m_ls.get(0) == false);
    REQUIRE(m_ls.get(1) == false);
    REQUIRE(m_ls.get(2) == false);
    REQUIRE(m_ls.get(3) == true);
}

TEST_CASE("make_dead_time_pwm_pair_fn — average HS time = duty - dt/T",
          "[v2][layer2_v6][dead_time_pwm][unit]") {
    // Numerical check over one period: with dt=0.05, T=1,
    // duty=0.6 → HS ON for 0.55s, LS ON for 0.35s, dead 0.1s.
    constexpr Real T = 1.0;
    constexpr Real duty = 0.6;
    constexpr Real dt = 0.05;
    auto sw = make_dead_time_pwm_pair_fn(
        1.0/T, duty, 0, 1, 2, dt);

    Real hs_on = 0, ls_on = 0, dead = 0;
    constexpr int N = 100000;
    for (int k = 0; k < N; ++k) {
        const Real t = (k + 0.5) * T / N;
        auto m = sw(t);
        if      (m.get(0)) hs_on += T/N;
        else if (m.get(1)) ls_on += T/N;
        else               dead  += T/N;
    }
    INFO("HS on=" << hs_on << " LS on=" << ls_on
         << " dead=" << dead);
    REQUIRE(hs_on == Approx(duty - dt).margin(2e-4));
    REQUIRE(ls_on == Approx((1.0 - duty) - dt).margin(2e-4));
    REQUIRE(dead  == Approx(2.0 * dt).margin(2e-4));
}

// -----------------------------------------------------------------------------
// Smoke: drive a half-bridge inverter leg with the helper.
// Vbus — HS — mid — LS — gnd, plus an RC load on `mid`.
// Time-averaged `mid` voltage should approach V_bus · duty
// (small loss for dead-time + R_on).
// -----------------------------------------------------------------------------

TEST_CASE("make_dead_time_pwm_pair_fn — half-bridge leg drives RC load to V·duty",
          "[v2][layer2_v6][dead_time_pwm][integration]") {
    constexpr Real V_bus = 24.0;
    constexpr Real f_sw  = 100e3;
    constexpr Real T_sw  = 1.0 / f_sw;
    constexpr Real duty  = 0.4;
    constexpr Real dt_dead = 100e-9;

    CircuitBuilder b;
    // Bus voltage.
    b.add_voltage_source("Vbus", "vbus", "gnd", V_bus);
    // HS = Q1 (drain=vbus, source=mid).
    b.add_switch("HS", "vbus", "mid",
                 /*g_on=*/1e3, /*g_off=*/1e-9);
    // LS = Q2 (drain=mid, source=gnd).
    b.add_switch("LS", "mid", "gnd",
                 /*g_on=*/1e3, /*g_off=*/1e-9);
    // RC averaging load.
    b.add_resistor("R", "mid", "vout", 1.0);
    b.add_capacitor("C", "vout", "gnd", 1e-6);
    b.add_resistor("R_L", "vout", "gnd", 100.0);

    constexpr Real dt   = 1e-8;
    constexpr Real tend = 20.0 * T_sw;   // ~20 cycles

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};

    const Size n_sw = b.graph().num_switches();
    REQUIRE(n_sw == 2);

    auto switch_fn = make_dead_time_pwm_pair_fn(
        f_sw, duty,
        /*hs=*/0, /*ls=*/1, n_sw,
        dt_dead);

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);
    REQUIRE(result.num_steps() > 100);

    // Last 5 cycles mean of v_out (node index = 2, after
    // vbus + mid).
    const Index vout_idx = b.node_id_of("vout");
    REQUIRE(vout_idx >= 0);
    const Size k_start = result.num_steps() -
        static_cast<Size>(5.0 * T_sw / dt);
    Real v_sum = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        v_sum += result.states[k][vout_idx];
    }
    const Real v_mean = v_sum /
        static_cast<Real>(result.num_steps() - k_start);
    const Real v_target = V_bus * duty;

    INFO("Half-bridge V_out mean = " << v_mean
         << " V (target " << v_target << " V)");
    // Allow small slack: dead-time chops ~2 % of the duty
    // power, and we haven't reached deep steady state yet.
    REQUIRE(v_mean == Approx(v_target).margin(1.0));
}
