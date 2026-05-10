// =============================================================================
// Test: linear devices opt out of the AD stamp path (Phase 3 of
// add-automatic-differentiation)
// =============================================================================
//
// Static / compile-time guard that linear devices (`Resistor`, `Capacitor`,
// `Inductor`, `VoltageSource`, `CurrentSource`) do NOT expose
// `stamp_jacobian_via_ad`. Their Jacobian is structurally constant per
// topology; running an AD pass for them buys nothing while paying ADReal
// arithmetic cost. The opt-out is enforced by simply not providing the
// member; this test pins that contract via SFINAE.
//
// Conversely, the four nonlinear devices (`IdealDiode`, `MOSFET`, `IGBT`,
// `VoltageControlledSwitch`) MUST expose `stamp_jacobian_via_ad` — that is
// the path the build flag `PULSIM_USE_AD_STAMP=ON` retargets the simulator
// onto.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/components/resistor.hpp"
#include "pulsim/v1/components/capacitor.hpp"
#include "pulsim/v1/components/inductor.hpp"
#include "pulsim/v1/components/voltage_source.hpp"
#include "pulsim/v1/components/current_source.hpp"
#include "pulsim/v1/components/ideal_diode.hpp"
#include "pulsim/v1/components/mosfet.hpp"
#include "pulsim/v1/components/igbt.hpp"
#include "pulsim/v1/components/voltage_controlled_switch.hpp"

#include <Eigen/Sparse>
#include <array>
#include <span>
#include <type_traits>

using namespace pulsim::v1;

namespace {

// Detects whether `Device::stamp_jacobian_via_ad` is callable with the
// canonical signature used by every nonlinear device that opts in.
template <typename Device, typename = void>
struct has_stamp_via_ad : std::false_type {};

template <typename Device>
struct has_stamp_via_ad<
    Device,
    std::void_t<decltype(std::declval<Device&>().stamp_jacobian_via_ad(
        std::declval<Eigen::SparseMatrix<Real>&>(),
        std::declval<Eigen::VectorXd&>(),
        std::declval<const Eigen::VectorXd&>(),
        std::declval<std::span<const NodeIndex>>()))>>
    : std::true_type {};

template <typename Device>
inline constexpr bool has_stamp_via_ad_v = has_stamp_via_ad<Device>::value;

}  // namespace

TEST_CASE("Linear devices do NOT expose stamp_jacobian_via_ad",
          "[ad][opt_out][linear]") {
    STATIC_REQUIRE_FALSE(has_stamp_via_ad_v<Resistor>);
    STATIC_REQUIRE_FALSE(has_stamp_via_ad_v<Capacitor>);
    STATIC_REQUIRE_FALSE(has_stamp_via_ad_v<Inductor>);
    STATIC_REQUIRE_FALSE(has_stamp_via_ad_v<VoltageSource>);
    STATIC_REQUIRE_FALSE(has_stamp_via_ad_v<CurrentSource>);
}

TEST_CASE("Nonlinear devices DO expose stamp_jacobian_via_ad",
          "[ad][opt_in][nonlinear]") {
    STATIC_REQUIRE(has_stamp_via_ad_v<IdealDiode>);
    STATIC_REQUIRE(has_stamp_via_ad_v<MOSFET>);
    STATIC_REQUIRE(has_stamp_via_ad_v<IGBT>);
    STATIC_REQUIRE(has_stamp_via_ad_v<VoltageControlledSwitch>);
}
