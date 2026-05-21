// =============================================================================
// Layer 2 — Resistor device model
// =============================================================================
//
// Ohm's law forward eval + AD partial derivatives = ±G analytically.
// This is the simplest possible test of the AD-only stamping
// pattern — if the Resistor case fails, nothing else can work.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/resistor.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::models;
using Catch::Approx;

TEST_CASE("Resistor: forward eval i = G·(v[0] - v[1])",
          "[v2][layer2][resistor]") {
    Resistor::Params p{Real{2.0}};
    ModelInputs<Resistor> v = {Real{3}, Real{1}};
    REQUIRE(evaluate_current<Resistor>(v, p) == Approx(Real{4}));
}

TEST_CASE("Resistor: AD partials are +G and -G",
          "[v2][layer2][resistor]") {
    Resistor::Params p{Real{2.0}};
    ModelInputs<Resistor> v = {Real{3}, Real{1}};
    const auto [i, J] = evaluate_current_and_jacobian<Resistor>(v, p);
    REQUIRE(i == Approx(Real{4}));
    REQUIRE(J[0] == Approx(Real{2}));
    REQUIRE(J[1] == Approx(Real{-2}));
}

TEST_CASE("Resistor: v[0] == v[1] → i = 0, partials still ±G",
          "[v2][layer2][resistor]") {
    Resistor::Params p{Real{5.0}};
    ModelInputs<Resistor> v = {Real{2.5}, Real{2.5}};
    const auto [i, J] = evaluate_current_and_jacobian<Resistor>(v, p);
    REQUIRE(i == Approx(Real{0}));
    REQUIRE(J[0] == Approx(Real{5}));
    REQUIRE(J[1] == Approx(Real{-5}));
}

TEST_CASE("Resistor: partials sum to zero (current depends on Δv only)",
          "[v2][layer2][resistor]") {
    Resistor::Params p{Real{1.5}};
    ModelInputs<Resistor> v = {Real{7}, Real{3}};
    const auto [i, J] = evaluate_current_and_jacobian<Resistor>(v, p);
    REQUIRE(J[0] + J[1] == Approx(Real{0}).margin(1e-15));
}

TEST_CASE("Resistor: large G stamps proportionally",
          "[v2][layer2][resistor]") {
    Resistor::Params p{Real{1e6}};   // 1 MΩ⁻¹
    ModelInputs<Resistor> v = {Real{1.0}, Real{0}};
    REQUIRE(evaluate_current<Resistor>(v, p) == Approx(Real{1e6}));
}
