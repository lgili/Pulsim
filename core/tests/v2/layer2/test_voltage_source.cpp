// =============================================================================
// Layer 2 — VoltageSource device model
// =============================================================================
//
// VoltageSource is a CONSTRAINT, not a current contributor. The
// `current<S>` function returns zero (the source's contribution is
// added by Layer 3 via the constraint-row path); `static_voltage`
// exposes the configured V for that path.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/voltage_source.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::models;
using Catch::Approx;

TEST_CASE("VoltageSource::current returns 0 regardless of terminal voltages",
          "[v2][layer2][vsource]") {
    VoltageSource::Params p{Real{12}};
    ModelInputs<VoltageSource> v = {Real{10}, Real{0}};
    REQUIRE(evaluate_current<VoltageSource>(v, p) == Approx(Real{0}));
}

TEST_CASE("VoltageSource::current returns 0 even with high voltages",
          "[v2][layer2][vsource]") {
    VoltageSource::Params p{Real{12}};
    ModelInputs<VoltageSource> v = {Real{1e6}, Real{-1e6}};
    REQUIRE(evaluate_current<VoltageSource>(v, p) == Approx(Real{0}));
}

TEST_CASE("VoltageSource AD partials are all zero",
          "[v2][layer2][vsource]") {
    // Since current<S> returns S{0}, its partials w.r.t. every
    // terminal voltage must be zero. Layer 3 will not stamp anything
    // off-diagonal for this branch — it adds the constraint row
    // instead.
    VoltageSource::Params p{Real{5}};
    ModelInputs<VoltageSource> v = {Real{3}, Real{1}};
    const auto [i, J] = evaluate_current_and_jacobian<VoltageSource>(v, p);
    REQUIRE(i == Approx(Real{0}));
    REQUIRE(J[0] == Approx(Real{0}));
    REQUIRE(J[1] == Approx(Real{0}));
}

TEST_CASE("VoltageSource::static_voltage returns the configured V",
          "[v2][layer2][vsource]") {
    VoltageSource::Params p{Real{12}};
    REQUIRE(VoltageSource::static_voltage(p) == Approx(Real{12}));

    VoltageSource::Params p2{Real{-3.3}};
    REQUIRE(VoltageSource::static_voltage(p2) == Approx(Real{-3.3}));
}
