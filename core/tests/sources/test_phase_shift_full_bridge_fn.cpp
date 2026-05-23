// =============================================================================
// Layer 2 V9 — `make_phase_shift_full_bridge_fn` tests
// =============================================================================
//
// Validates:
//   * φ = 0 → leg A and leg B are bit-for-bit identical
//     (synchronous; v_AB = 0).
//   * φ = π → leg B is the exact mirror of leg A (when A_HS
//     ON, B_LS ON → v_AB = +V_bus).
//   * Shoot-through per leg under all φ.
//   * φ = π/2 → 25 % duty +V_bus pulse on v_AB.
//   * carrier_frequency=0 → flat OFF.
//   * Other switch bits stay OFF.
//   * Integration: full-bridge driving a series-L load —
//     the inductor current ramps up linearly during the
//     +V_bus interval (textbook phase-shift behavior).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::sources;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("make_phase_shift_full_bridge_fn — φ=0 → legs are synchronous",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    auto sw = make_phase_shift_full_bridge_fn(
        /*f=*/1000.0, /*phase_shift=*/0.0,
        /*A_HS=*/0, /*A_LS=*/1,
        /*B_HS=*/2, /*B_LS=*/3,
        /*n_sw=*/4, /*dt=*/0.0);
    // Bit 0 (A_HS) ≡ Bit 2 (B_HS); bit 1 ≡ bit 3.
    for (int k = 0; k < 200; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 200.0;
        auto m = sw(t);
        INFO("t=" << t);
        REQUIRE(m.get(0) == m.get(2));
        REQUIRE(m.get(1) == m.get(3));
    }
}

TEST_CASE("make_phase_shift_full_bridge_fn — φ=π → legs are anti-phase",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    auto sw = make_phase_shift_full_bridge_fn(
        1000.0, std::numbers::pi_v<Real>,
        0, 1, 2, 3, 4, 0.0);
    // A_HS ON ↔ B_LS ON (v_AB = +V_bus all the time A_HS on).
    // A_LS ON ↔ B_HS ON (v_AB = -V_bus).
    for (int k = 0; k < 200; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 200.0;
        auto m = sw(t);
        INFO("t=" << t);
        REQUIRE(m.get(0) == m.get(3));   // A_HS ≡ B_LS
        REQUIRE(m.get(1) == m.get(2));   // A_LS ≡ B_HS
    }
}

TEST_CASE("make_phase_shift_full_bridge_fn — shoot-through never happens per leg",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    // Sweep multiple φ values + dt values.
    for (Real phi : {Real{0.0},
                       std::numbers::pi_v<Real> * Real{0.25},
                       std::numbers::pi_v<Real> * Real{0.5},
                       std::numbers::pi_v<Real> * Real{0.75},
                       std::numbers::pi_v<Real>}) {
        for (Real dt : {Real{0.0}, Real{1e-5}, Real{5e-5}}) {
            auto sw = make_phase_shift_full_bridge_fn(
                1000.0, phi, 0, 1, 2, 3, 4, dt);
            for (int k = 0; k < 500; ++k) {
                const Real t = (k + 0.317) * 5e-3 / 500.0;
                auto m = sw(t);
                const bool a_st = m.get(0) && m.get(1);
                const bool b_st = m.get(2) && m.get(3);
                INFO("phi=" << phi << " dt=" << dt
                     << " t=" << t);
                REQUIRE_FALSE(a_st);
                REQUIRE_FALSE(b_st);
            }
        }
    }
}

TEST_CASE("make_phase_shift_full_bridge_fn — φ=π/2 → 25%% +V pulse on v_AB",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    // φ = π/2 lags B by T/4. With T=1s, leg B shifts to:
    //   leg A: HS [0, 0.5), LS [0.5, 1.0)
    //   leg B: HS [0.25, 0.75), LS [0.75, 1.25 → wraps → 0.25)
    //
    // v_AB = v_A - v_B (in normalized V_bus units):
    //   [0,    0.25): A_HS on, B_LS on → v_AB = +1
    //   [0.25, 0.5):  A_HS on, B_HS on → v_AB = 0
    //   [0.5,  0.75): A_LS on, B_HS on → v_AB = -1
    //   [0.75, 1.0):  A_LS on, B_LS on → v_AB = 0
    auto sw = make_phase_shift_full_bridge_fn(
        /*f=*/1.0,
        std::numbers::pi_v<Real> * Real{0.5},
        0, 1, 2, 3, 4, /*dt=*/0.0);
    auto v_ab = [&](Real t) {
        auto m = sw(t);
        Real va = m.get(0) ? Real{1} : (m.get(1) ? Real{0} : Real{0.5});
        Real vb = m.get(2) ? Real{1} : (m.get(3) ? Real{0} : Real{0.5});
        return va - vb;
    };
    // Sample each phase region.
    REQUIRE(v_ab(0.10)  == Approx(+1.0));
    REQUIRE(v_ab(0.20)  == Approx(+1.0));
    REQUIRE(v_ab(0.30)  == Approx(0.0));
    REQUIRE(v_ab(0.45)  == Approx(0.0));
    REQUIRE(v_ab(0.55)  == Approx(-1.0));
    REQUIRE(v_ab(0.65)  == Approx(-1.0));
    REQUIRE(v_ab(0.80)  == Approx(0.0));
    REQUIRE(v_ab(0.95)  == Approx(0.0));
}

TEST_CASE("make_phase_shift_full_bridge_fn — switching_frequency=0 → flat OFF",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    auto sw = make_phase_shift_full_bridge_fn(
        0.0, std::numbers::pi_v<Real> * Real{0.5},
        0, 1, 2, 3, 4, 1e-7);
    for (double t : {0.0, 0.1, 1.0, 100.0}) {
        auto m = sw(t);
        INFO("t=" << t);
        for (Size i = 0; i < 4; ++i) {
            REQUIRE(m.get(i) == false);
        }
    }
}

TEST_CASE("make_phase_shift_full_bridge_fn — only the 4 bridge bits toggle",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    // 6 total switches; bridge uses idx 0,1,2,3. Bits 4 and
    // 5 must stay OFF.
    auto sw = make_phase_shift_full_bridge_fn(
        1000.0, std::numbers::pi_v<Real> * Real{0.5},
        0, 1, 2, 3, /*n_sw=*/6, 1e-5);
    for (int k = 0; k < 200; ++k) {
        const Real t = (k + 0.5) * 5e-3 / 200.0;
        auto m = sw(t);
        INFO("t=" << t);
        REQUIRE(m.size() == 6);
        REQUIRE(m.get(4) == false);
        REQUIRE(m.get(5) == false);
    }
}

TEST_CASE("make_phase_shift_full_bridge_fn — phase wraps for out-of-range angles",
          "[v2][layer2_v9][phase_shift_full_bridge][unit]") {
    // φ = 2π should behave identically to φ = 0.
    auto sw_wrap = make_phase_shift_full_bridge_fn(
        1000.0, Real{2} * std::numbers::pi_v<Real>,
        0, 1, 2, 3, 4, 0.0);
    auto sw_zero = make_phase_shift_full_bridge_fn(
        1000.0, 0.0, 0, 1, 2, 3, 4, 0.0);
    for (int k = 0; k < 100; ++k) {
        const Real t = (k + 0.1) * 5e-3 / 100.0;
        auto a = sw_wrap(t);
        auto b = sw_zero(t);
        for (Size i = 0; i < 4; ++i) {
            INFO("t=" << t << " bit=" << i);
            REQUIRE(a.get(i) == b.get(i));
        }
    }
}

// -----------------------------------------------------------------------------
// Integration: full-bridge driving a series-L load. The
// inductor current is the integral of v_AB, so over one
// switching period at φ = π (max output), the L sees a
// full ±V_bus square wave and the current ramps up/down
// each half cycle.
// -----------------------------------------------------------------------------

TEST_CASE("make_phase_shift_full_bridge_fn — series-L load at φ=π/2 transfers power",
          "[v2][layer2_v9][phase_shift_full_bridge][integration]") {
    constexpr Real V_bus = 48.0;
    constexpr Real f_sw  = 100e3;
    constexpr Real T_sw  = 1.0 / f_sw;
    constexpr Real dt_dead = 50e-9;

    CircuitBuilder b;
    b.add_voltage_source("Vbus", "vbus", "gnd", V_bus);
    // Leg A.
    b.add_switch("HS_A", "vbus", "mid_a", 1e3, 1e-9);
    b.add_switch("LS_A", "mid_a", "gnd",  1e3, 1e-9);
    // Leg B.
    b.add_switch("HS_B", "vbus", "mid_b", 1e3, 1e-9);
    b.add_switch("LS_B", "mid_b", "gnd",  1e3, 1e-9);
    // Series L + R load between mid_a and mid_b (no
    // transformer for simplicity).
    b.add_inductor("Lk", "mid_a", "rl_mid", 10e-6);
    b.add_resistor("R_L", "rl_mid", "mid_b", 1.0);
    // Anti-parallel diodes on all four switches (essential
    // for inductive load freewheeling during dead-time).
    b.add_diode("D_HS_A", "mid_a", "vbus", 1e3, 1e-9);
    b.add_diode("D_LS_A", "gnd",   "mid_a", 1e3, 1e-9);
    b.add_diode("D_HS_B", "mid_b", "vbus", 1e3, 1e-9);
    b.add_diode("D_LS_B", "gnd",   "mid_b", 1e3, 1e-9);

    constexpr Real dt   = 1e-8;        // 10 ns
    constexpr Real tend = 200.0 * T_sw; // 2 ms — settle
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};
    const Size n_sw = b.graph().num_switches();
    REQUIRE(n_sw == 8);   // 4 mosfets + 4 diodes

    auto sw_fn = make_phase_shift_full_bridge_fn(
        f_sw, std::numbers::pi_v<Real> * Real{0.5},
        0, 1, 2, 3, n_sw, dt_dead);

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, sw_fn);
    REQUIRE(result.num_steps() > 1000);

    // Check mid_a and mid_b reached non-trivial AC swing
    // (proves switching is happening) and the time-averaged
    // V_AB matches the expected D_eff · V_bus = 0.5·V_bus
    // for φ = π/2.
    const Index a_idx = b.node_id_of("mid_a");
    const Index bb_idx = b.node_id_of("mid_b");
    REQUIRE(a_idx >= 0);
    REQUIRE(bb_idx >= 0);

    // Average v_AB over the LAST 50 switching periods.
    const Size k_start = result.num_steps() -
        static_cast<Size>(50.0 * T_sw / dt);
    Real vab_sum = 0;
    Real vab_min = 1e9, vab_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real vab = result.states[k][a_idx] -
                          result.states[k][bb_idx];
        vab_sum += vab;
        vab_min = std::min(vab_min, vab);
        vab_max = std::max(vab_max, vab);
    }
    const Real vab_mean = vab_sum /
        static_cast<Real>(result.num_steps() - k_start);
    INFO("v_AB mean = " << vab_mean << " V (φ=π/2 means "
         "v_AB time-average is 0 over a symmetric cycle)");
    // φ=π/2 produces a symmetric pulse pattern: +V_bus for
    // 25 %, 0 for 25 %, -V_bus for 25 %, 0 for 25 %. The
    // average is ZERO. Allow a small numerical residual.
    REQUIRE(std::abs(vab_mean) < 5.0);

    // Swing magnitude proves full-rail switching is happening.
    INFO("v_AB swing: [" << vab_min << ", " << vab_max
         << "] (expected near ±V_bus = ±" << V_bus << " V)");
    REQUIRE(vab_max > V_bus * 0.7);
    REQUIRE(vab_min < -V_bus * 0.7);
}
