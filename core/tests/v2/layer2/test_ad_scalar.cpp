// =============================================================================
// Layer 2 — forward-mode AD scalar (ADRealN<N>)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/ad/ad_scalar.hpp"

#include <cmath>

using namespace pulsim::v2;
using namespace pulsim::v2::ad;
using Catch::Approx;

TEST_CASE("Default-constructed ADRealN<3> is zero with zero derivs",
          "[v2][layer2][ad]") {
    ADRealN<3> x;
    REQUIRE(x.value() == Real{0});
    REQUIRE(x.deriv(0) == Real{0});
    REQUIRE(x.deriv(1) == Real{0});
    REQUIRE(x.deriv(2) == Real{0});
}

TEST_CASE("Construct from Real → value set, derivs zero (constant)",
          "[v2][layer2][ad]") {
    ADRealN<2> x{Real{2.5}};
    REQUIRE(x.value() == Real{2.5});
    REQUIRE(x.deriv(0) == Real{0});
    REQUIRE(x.deriv(1) == Real{0});
}

TEST_CASE("seed2 returns inputs with identity-row derivative slots",
          "[v2][layer2][ad]") {
    auto inputs = seed2(Real{3}, Real{4});
    REQUIRE(inputs[0].value() == Real{3});
    REQUIRE(inputs[0].deriv(0) == Real{1});
    REQUIRE(inputs[0].deriv(1) == Real{0});
    REQUIRE(inputs[1].value() == Real{4});
    REQUIRE(inputs[1].deriv(0) == Real{0});
    REQUIRE(inputs[1].deriv(1) == Real{1});
}

TEST_CASE("Addition propagates value + derivs additively",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{3}, Real{4});
    const auto z = v[0] + v[1];
    REQUIRE(z.value() == Approx(Real{7}));
    REQUIRE(z.deriv(0) == Approx(Real{1}));
    REQUIRE(z.deriv(1) == Approx(Real{1}));
}

TEST_CASE("Multiplication uses the product rule",
          "[v2][layer2][ad]") {
    // d(x·y)/dx = y, d(x·y)/dy = x.
    auto v = seed2(Real{3}, Real{4});
    const auto w = v[0] * v[1];
    REQUIRE(w.value() == Approx(Real{12}));
    REQUIRE(w.deriv(0) == Approx(Real{4}));
    REQUIRE(w.deriv(1) == Approx(Real{3}));
}

TEST_CASE("Division uses the quotient rule",
          "[v2][layer2][ad]") {
    // d(x/y)/dx = 1/y; d(x/y)/dy = -x/y².
    auto v = seed2(Real{6}, Real{2});
    const auto q = v[0] / v[1];
    REQUIRE(q.value() == Approx(Real{3}));
    REQUIRE(q.deriv(0) == Approx(Real{0.5}));
    REQUIRE(q.deriv(1) == Approx(Real{-1.5}));    // -6/4 = -1.5
}

TEST_CASE("Unary minus negates value and all derivs",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{3}, Real{4});
    const auto z = -v[0];
    REQUIRE(z.value() == Approx(Real{-3}));
    REQUIRE(z.deriv(0) == Approx(Real{-1}));
    REQUIRE(z.deriv(1) == Approx(Real{0}));
}

TEST_CASE("Mixed ad-vs-real arithmetic preserves derivs",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{3}, Real{4});
    const auto z = Real{2} * v[0] + v[1] - Real{1};
    REQUIRE(z.value() == Approx(Real{9}));     // 2·3 + 4 - 1
    REQUIRE(z.deriv(0) == Approx(Real{2}));
    REQUIRE(z.deriv(1) == Approx(Real{1}));
}

TEST_CASE("exp propagates exp(x) as its own derivative",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{2}, Real{0});
    const auto y = exp(v[0]);
    REQUIRE(y.value() == Approx(std::exp(Real{2})).margin(1e-12));
    REQUIRE(y.deriv(0) == Approx(std::exp(Real{2})).margin(1e-12));
    REQUIRE(y.deriv(1) == Approx(Real{0}));
}

TEST_CASE("tanh derivative is 1 - tanh²",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{0.5}, Real{0});
    const auto y = tanh(v[0]);
    const Real tv = std::tanh(Real{0.5});
    REQUIRE(y.value() == Approx(tv).margin(1e-12));
    REQUIRE(y.deriv(0) == Approx(Real{1} - tv * tv).margin(1e-12));
}

TEST_CASE("sqrt derivative is 1 / (2 sqrt(x))",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{4}, Real{0});
    const auto y = sqrt(v[0]);
    REQUIRE(y.value() == Approx(Real{2}).margin(1e-12));
    REQUIRE(y.deriv(0) == Approx(Real{0.25}).margin(1e-12));  // 1/(2·2)
}

TEST_CASE("Composite function exp(x) + y² with chain rule",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{1}, Real{2});
    const auto f = exp(v[0]) + v[1] * v[1];
    REQUIRE(f.value() == Approx(std::exp(Real{1}) + Real{4}).margin(1e-12));
    REQUIRE(f.deriv(0) == Approx(std::exp(Real{1})).margin(1e-12));
    REQUIRE(f.deriv(1) == Approx(Real{4}).margin(1e-12));     // d(y²)/dy = 2y = 4
}

TEST_CASE("Comparison operators compare values only",
          "[v2][layer2][ad]") {
    // Two ADRealN<2> with same value but different derivs compare equal.
    ADRealN<2> a{Real{3}, {Real{1}, Real{0}}};
    ADRealN<2> b{Real{3}, {Real{0}, Real{1}}};
    REQUIRE(a == b);          // values match
    REQUIRE_FALSE(a != b);
    REQUIRE_FALSE(a < b);
    REQUIRE(a <= b);

    ADRealN<2> c{Real{4}, {Real{0}, Real{0}}};
    REQUIRE(a < c);
    REQUIRE(c > a);
    REQUIRE(a != c);
}

TEST_CASE("Compound assignment += updates value and derivs",
          "[v2][layer2][ad]") {
    auto v = seed2(Real{3}, Real{4});
    ADRealN<2> z = v[0];
    z += v[1];
    REQUIRE(z.value() == Approx(Real{7}));
    REQUIRE(z.deriv(0) == Approx(Real{1}));
    REQUIRE(z.deriv(1) == Approx(Real{1}));
}
