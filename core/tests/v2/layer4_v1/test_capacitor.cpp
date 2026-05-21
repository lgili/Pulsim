// =============================================================================
// Layer 4 V1 — Capacitor companion math
// =============================================================================
//
// Unit tests for the static helpers on `models::Capacitor`. No
// stamping, no cache — just the trap-companion formulas in
// isolation.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/models/capacitor.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::models;
using Catch::Approx;

TEST_CASE("Capacitor: g_eq for 1 µF at 1 µs dt equals 2 S",
          "[v2][layer4_v1][capacitor]") {
    Capacitor::Params p{.C = 1e-6};
    REQUIRE(Capacitor::g_eq(1e-6, p) == Approx(2.0).margin(1e-12));
}

TEST_CASE("Capacitor: g_eq scales inversely with dt",
          "[v2][layer4_v1][capacitor]") {
    Capacitor::Params p{.C = 1e-6};
    REQUIRE(Capacitor::g_eq(10e-6, p) == Approx(0.2).margin(1e-12));
    REQUIRE(Capacitor::g_eq(0.5e-6, p) == Approx(4.0).margin(1e-12));
}

TEST_CASE("Capacitor: history_term = g_eq · v_prev + i_prev",
          "[v2][layer4_v1][capacitor]") {
    Capacitor::Params p{.C = 1e-6};
    const Real dt = 1e-6;
    const Real v_prev = 10.0;
    const Real i_prev = 0.5;
    const Real i_hist =
        Capacitor::history_term(v_prev, i_prev, dt, p);
    // g_eq = 2, so i_hist = 2 · 10 + 0.5 = 20.5
    REQUIRE(i_hist == Approx(20.5).margin(1e-12));
}

TEST_CASE("Capacitor: zero previous state gives zero history",
          "[v2][layer4_v1][capacitor]") {
    Capacitor::Params p{.C = 5e-9};
    REQUIRE(Capacitor::history_term(0, 0, 1e-6, p) ==
            Approx(0).margin(1e-15));
}

TEST_CASE("Capacitor: companion reproduces back-substitution",
          "[v2][layer4_v1][capacitor]") {
    // For step n+1: i_{n+1} = g_eq · v_{n+1} − I_hist.
    // Pick known values and verify.
    Capacitor::Params p{.C = 2e-6};
    const Real dt = 1e-6;
    const Real g_eq = Capacitor::g_eq(dt, p);
    REQUIRE(g_eq == Approx(4.0).margin(1e-12));

    const Real v_prev = 3.0;
    const Real i_prev = 1.0;
    const Real i_hist =
        Capacitor::history_term(v_prev, i_prev, dt, p);
    REQUIRE(i_hist == Approx(13.0).margin(1e-12));

    // Now suppose at step n+1 the voltage settled at v=5.
    // Companion says: i_{n+1} = 4 · 5 − 13 = 7.
    REQUIRE(g_eq * 5.0 - i_hist == Approx(7.0).margin(1e-12));
}

TEST_CASE("Capacitor: is_dynamic flag is true",
          "[v2][layer4_v1][capacitor]") {
    STATIC_REQUIRE(Capacitor::is_dynamic);
    STATIC_REQUIRE(Capacitor::num_terminals == 2);
    STATIC_REQUIRE(Capacitor::kind ==
                    topology::BranchKind::PassiveLinear);
}

TEST_CASE("Capacitor: static current contract returns 0",
          "[v2][layer4_v1][capacitor]") {
    Capacitor::Params p{.C = 1e-6};
    Real v[2] = {3.0, 1.0};
    REQUIRE(Capacitor::current(v, p) == Approx(0).margin(1e-15));
}
