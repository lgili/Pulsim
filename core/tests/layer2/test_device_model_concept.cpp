// =============================================================================
// Layer 2 — DeviceModel concept + evaluate_current_and_jacobian
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/models/device_model.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/models/resistor.hpp"
#include "pulsim/models/voltage_source.hpp"

using namespace pulsim;
using namespace pulsim::models;
using Catch::Approx;

// Reference models satisfy the concept.
static_assert(DeviceModel<Resistor>);
static_assert(DeviceModel<VoltageSource>);
static_assert(DeviceModel<IdealDiode>);

// Negative test: a stub missing `current<S>` fails the concept.
struct BrokenStub {
    struct Params {};
    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = true;
    // Intentionally NO current<S>(...) template.
};
static_assert(!DeviceModel<BrokenStub>);

TEST_CASE("Resistor satisfies DeviceModel (runtime confirmation)",
          "[v2][layer2][concept]") {
    REQUIRE(Resistor::kind == topology::BranchKind::PassiveLinear);
    REQUIRE(Resistor::num_terminals == 2);
    REQUIRE(Resistor::is_linear);
}

TEST_CASE("VoltageSource satisfies DeviceModel with Source kind",
          "[v2][layer2][concept]") {
    REQUIRE(VoltageSource::kind == topology::BranchKind::Source);
    REQUIRE(VoltageSource::is_linear);
}

TEST_CASE("IdealDiode satisfies DeviceModel with Nonlinear kind",
          "[v2][layer2][concept]") {
    REQUIRE(IdealDiode::kind == topology::BranchKind::Nonlinear);
    REQUIRE_FALSE(IdealDiode::is_linear);
}

TEST_CASE("evaluate_current_and_jacobian returns current value + AD partials",
          "[v2][layer2][concept]") {
    Resistor::Params p{Real{2.0}};   // G = 2
    ModelInputs<Resistor> v = {Real{3}, Real{1}};

    const auto [i, J] = evaluate_current_and_jacobian<Resistor>(v, p);

    REQUIRE(i == Approx(Real{4}).margin(1e-12));    // G·(3-1) = 4
    REQUIRE(J[0] == Approx(Real{2}).margin(1e-12)); // ∂i/∂v[0] = +G
    REQUIRE(J[1] == Approx(Real{-2}).margin(1e-12));// ∂i/∂v[1] = -G
}

TEST_CASE("evaluate_current returns the value-only forward eval",
          "[v2][layer2][concept]") {
    Resistor::Params p{Real{0.5}};   // G = 0.5
    ModelInputs<Resistor> v = {Real{10}, Real{2}};
    const Real i = evaluate_current<Resistor>(v, p);
    REQUIRE(i == Approx(Real{4}));                  // 0.5·(10-2) = 4
}

TEST_CASE("ModelInputs<T> is an std::array of the right size",
          "[v2][layer2][concept]") {
    STATIC_REQUIRE(std::tuple_size_v<ModelInputs<Resistor>> == 2);
    STATIC_REQUIRE(std::tuple_size_v<ModelInputs<IdealDiode>> == 2);
    STATIC_REQUIRE(std::tuple_size_v<ModelInputs<VoltageSource>> == 2);
}
