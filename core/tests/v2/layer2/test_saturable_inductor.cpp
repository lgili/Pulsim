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

#include "pulsim/v2/models/saturable_inductor.hpp"

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
        .L_0 = 1e-3, .I_sat = 5.0, .n_exp = 2.0,
        .L_residual = 0.0};
    REQUIRE(L_at(p, 0.0) == Approx(1e-3));
}

TEST_CASE("SaturableInductor — at |i| = I_sat returns L_0 / 2",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    SaturableInductor::Params p{
        .L_0 = 1e-3, .I_sat = 5.0, .n_exp = 2.0,
        .L_residual = 0.0};
    // L(I_sat) = L_0 / (1 + 1) = L_0 / 2.
    REQUIRE(L_at(p, 5.0) == Approx(0.5e-3));
    REQUIRE(L_at(p, -5.0) == Approx(0.5e-3));   // symmetric in |i|
}

TEST_CASE("SaturableInductor — high |i| approaches L_residual",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    SaturableInductor::Params p{
        .L_0 = 1e-3, .I_sat = 5.0, .n_exp = 2.0,
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
        .L_0 = 1e-3, .I_sat = 5.0, .n_exp = 2.0,
        .L_residual = 0.0};
    Real prev = L_at(p, 0.0);
    for (Real i = 0.1; i <= 20.0; i += 0.5) {
        const Real L = L_at(p, i);
        REQUIRE(L < prev);
        prev = L;
    }
}

TEST_CASE("SaturableInductor — sharper knee with higher n",
          "[v2][layer2_v16][saturable_inductor][unit]") {
    // At i = 2·I_sat, higher n → L closer to L_residual.
    SaturableInductor::Params p_soft{
        .L_0 = 1e-3, .I_sat = 5.0, .n_exp = 1.0,
        .L_residual = 0.0};
    SaturableInductor::Params p_sharp{
        .L_0 = 1e-3, .I_sat = 5.0, .n_exp = 4.0,
        .L_residual = 0.0};
    const Real L_soft  = L_at(p_soft,  10.0);
    const Real L_sharp = L_at(p_sharp, 10.0);
    INFO("L_soft = " << L_soft << ", L_sharp = " << L_sharp);
    REQUIRE(L_sharp < L_soft);   // sharp knee saturates faster
}
