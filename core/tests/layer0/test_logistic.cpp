// =============================================================================
// Layer 0 — the overflow-free logistic
// =============================================================================
//
// v2.0 Phase 2. Every smooth device model blends its regions with
// `1/(1 + exp(-kappa*u))`. Written that way the VALUE survives a
// large negative u — 1/(1+inf) is 0 — but forward-mode AD propagates
// `d = exp(x)*dx`, so the derivative is inf, and the reciprocal's is
// inf/inf = NaN. One NaN in the Jacobian defeats Levenberg-Marquardt
// at every lambda.
//
// The threshold is exactly kappa*|u| > 709. At the default
// kappa = 20 that is 35 V of reverse bias, which a mains rectifier
// passes in its first half-cycle.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/logistic.hpp"

#include <cmath>
#include <limits>

using namespace pulsim;
using Catch::Approx;

TEST_CASE("logistic matches the textbook formula where that works",
          "[v2][layer0][logistic]") {
    for (const Real z : {-30.0, -5.0, -1.0, -0.25, 0.0,
                          0.25, 1.0, 5.0, 30.0}) {
        const Real want = Real{1} / (Real{1} + std::exp(-z));
        INFO("z = " << z);
        REQUIRE(numeric::logistic(z) == Approx(want).epsilon(1e-14));
    }
    REQUIRE(numeric::logistic(Real{0}) == Approx(0.5));
}

TEST_CASE("logistic saturates instead of overflowing",
          "[v2][layer0][logistic]") {
    // Past |z| = 709 the textbook form's exp() is inf. These must
    // still be finite, monotone and correctly clamped.
    for (const Real z : {-1e3, -1e6, -1e300}) {
        INFO("z = " << z);
        REQUIRE(std::isfinite(numeric::logistic(z)));
        REQUIRE(numeric::logistic(z) == Approx(0.0).margin(1e-300));
    }
    for (const Real z : {1e3, 1e6, 1e300}) {
        INFO("z = " << z);
        REQUIRE(numeric::logistic(z) == Approx(1.0));
    }
}

TEST_CASE("the AD derivative stays finite where the naive form NaNs",
          "[v2][layer0][logistic][ad]") {
    using AD = ad::ADRealN<1>;

    // Prove the premise first: written the textbook way, the
    // derivative really is NaN out here. If this ever stops being
    // true the fix has become unnecessary and this file should say
    // so rather than quietly passing.
    {
        const AD z{-1000.0, {Real{1}}};
        const AD naive = AD{Real{1}} / (AD{Real{1}} + exp(-z));
        REQUIRE(std::isnan(naive.deriv(0)));
    }

    // The fixed form: value and derivative both finite, and the
    // derivative is the true alpha*(1 - alpha).
    for (const Real z0 : {-1e4, -1000.0, -50.0, -1.0, 0.0,
                           1.0, 50.0, 1000.0, 1e4}) {
        const AD z{z0, {Real{1}}};
        const AD a = numeric::logistic(z);
        INFO("z = " << z0);
        REQUIRE(std::isfinite(a.value()));
        REQUIRE(std::isfinite(a.deriv(0)));
        const Real want = a.value() * (Real{1} - a.value());
        REQUIRE(a.deriv(0) == Approx(want).margin(1e-18));
    }
}

TEST_CASE("the two branches agree across the seam",
          "[v2][layer0][logistic][ad]") {
    // The implementation switches form at z = 0. Both expressions
    // are the same function, so neither the value nor the slope may
    // jump there.
    using AD = ad::ADRealN<1>;
    constexpr Real eps = 1e-9;
    const AD lo{-eps, {Real{1}}};
    const AD hi{ eps, {Real{1}}};
    // The two points are 2*eps apart and the slope is 1/4 there, so
    // the values must differ by 2*eps/4 and no more — a JUMP would
    // show up as a difference far larger than that bound.
    const Real dv = numeric::logistic(hi).value() -
                     numeric::logistic(lo).value();
    REQUIRE(dv == Approx(0.5 * eps).margin(1e-15));
    REQUIRE(numeric::logistic(lo).deriv(0) ==
             Approx(numeric::logistic(hi).deriv(0)).epsilon(1e-9));
    REQUIRE(numeric::logistic(AD{Real{0}, {Real{1}}}).deriv(0) ==
             Approx(0.25));
}
