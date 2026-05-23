// =============================================================================
// Layer 4 V1 — Inductor companion math
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/models/inductor.hpp"

using namespace pulsim;
using namespace pulsim::models;
using Catch::Approx;

TEST_CASE("Inductor: g_eq_inv for 1 mH at 1 µs dt equals 5e-4",
          "[v2][layer4_v1][inductor]") {
    Inductor::Params p{.L = 1e-3};
    REQUIRE(Inductor::g_eq_inv(1e-6, p) ==
            Approx(5e-4).margin(1e-15));
}

TEST_CASE("Inductor: g_eq_inv scales linearly with dt",
          "[v2][layer4_v1][inductor]") {
    Inductor::Params p{.L = 10e-6};   // 10 µH
    REQUIRE(Inductor::g_eq_inv(1e-6, p) ==
            Approx(0.05).margin(1e-12));
    REQUIRE(Inductor::g_eq_inv(2e-6, p) ==
            Approx(0.10).margin(1e-12));
}

TEST_CASE("Inductor: history_term = i_prev + g_eq_inv · v_prev",
          "[v2][layer4_v1][inductor]") {
    Inductor::Params p{.L = 1e-3};
    const Real dt = 1e-6;
    // g_eq_inv = 5e-4. history = i + g_eq_inv · v = 2 + 5e-4 · 12.
    const Real h =
        Inductor::history_term(12.0, 2.0, dt, p);
    REQUIRE(h == Approx(2.006).margin(1e-12));
}

TEST_CASE("Inductor: zero previous state gives zero history",
          "[v2][layer4_v1][inductor]") {
    Inductor::Params p{.L = 100e-6};
    REQUIRE(Inductor::history_term(0, 0, 1e-6, p) ==
            Approx(0).margin(1e-15));
}

TEST_CASE("Inductor: companion reproduces algebra",
          "[v2][layer4_v1][inductor]") {
    // i_{n+1} = G_L,eq · v_{n+1} + I_hist,L,
    // where G_L,eq = g_eq_inv = dt/(2L), I_hist,L = i_n +
    // g_eq_inv · v_n.
    Inductor::Params p{.L = 1e-6};
    const Real dt = 1e-7;
    const Real g_eq_inv = Inductor::g_eq_inv(dt, p);   // 0.05

    const Real v_prev = 4.0, i_prev = 1.0;
    const Real h = Inductor::history_term(v_prev, i_prev, dt, p);
    // h = 1 + 0.05 · 4 = 1.2
    REQUIRE(h == Approx(1.2).margin(1e-12));

    // Now suppose v_{n+1} = 8: i_{n+1} = 0.05 · 8 + 1.2 = 1.6.
    REQUIRE(g_eq_inv * 8.0 + h == Approx(1.6).margin(1e-12));
}

TEST_CASE("Inductor: flags",
          "[v2][layer4_v1][inductor]") {
    STATIC_REQUIRE(Inductor::is_dynamic);
    STATIC_REQUIRE(Inductor::needs_branch_unknown);
    STATIC_REQUIRE(Inductor::num_terminals == 2);
    STATIC_REQUIRE(Inductor::kind ==
                    topology::BranchKind::PassiveLinear);
}
