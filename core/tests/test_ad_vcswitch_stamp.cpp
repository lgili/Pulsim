// =============================================================================
// Test: AD-derived VCSwitch stamp matches manual stamp (Phase 2 of
// add-automatic-differentiation)
// =============================================================================
//
// Cross-validates `VoltageControlledSwitch::stamp_jacobian_via_ad` against
// the legacy manual `stamp_jacobian_impl` (Behavioral mode) — VCSwitch uses
// standard form (`f[t1] += i_sw`, `J[t1,*] = ∂i/∂x`), unlike MOSFET/IGBT
// which use Norton companion form. The transition zone around v_threshold
// is the most demanding region for cross-validation since the tanh
// derivative is at its peak there.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/voltage_controlled_switch.hpp"

#include <Eigen/Sparse>
#include <array>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

struct StampResult {
    Eigen::SparseMatrix<Real> J;
    Eigen::VectorXd f;
};

[[nodiscard]] StampResult stamp_manual(VoltageControlledSwitch& s,
                                       Real v_ctrl, Real v_t1, Real v_t2) {
    StampResult r;
    r.J.resize(3, 3);
    r.f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << v_ctrl, v_t1, v_t2;
    std::array<Index, 3> nodes{0, 1, 2};
    s.stamp_jacobian(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

[[nodiscard]] StampResult stamp_ad(VoltageControlledSwitch& s,
                                   Real v_ctrl, Real v_t1, Real v_t2) {
    StampResult r;
    r.J.resize(3, 3);
    r.f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << v_ctrl, v_t1, v_t2;
    std::array<Index, 3> nodes{0, 1, 2};
    s.stamp_jacobian_via_ad(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

void cross_validate(VoltageControlledSwitch& sw,
                    Real v_ctrl, Real v_t1, Real v_t2, const char* label) {
    const auto manual = stamp_manual(sw, v_ctrl, v_t1, v_t2);
    const auto ad     = stamp_ad(sw, v_ctrl, v_t1, v_t2);

    INFO("Op-point: " << label
         << " v_ctrl=" << v_ctrl << " v_t1=" << v_t1 << " v_t2=" << v_t2);

    REQUIRE(manual.J.rows() == ad.J.rows());
    REQUIRE(manual.J.cols() == ad.J.cols());

    for (Index r = 0; r < manual.J.rows(); ++r) {
        for (Index c = 0; c < manual.J.cols(); ++c) {
            INFO("J(" << r << ", " << c << ")");
            CHECK(manual.J.coeff(r, c) ==
                  Approx(ad.J.coeff(r, c)).margin(1e-12));
        }
    }
    REQUIRE(manual.f.size() == ad.f.size());
    for (Index i = 0; i < manual.f.size(); ++i) {
        INFO("f[" << i << "]");
        CHECK(manual.f[i] == Approx(ad.f[i]).margin(1e-12));
    }
}

}  // namespace

TEST_CASE("AD VCSwitch stamp matches manual across the transition curve",
          "[ad][vcswitch][stamp][cross_validation]") {
    VoltageControlledSwitch sw(/*v_th=*/2.5, /*g_on=*/1e3, /*g_off=*/1e-9,
                               "S_test");

    SECTION("control well above threshold (saturated on)") {
        cross_validate(sw, /*v_ctrl=*/5.0, /*v_t1=*/1.0, /*v_t2=*/0.0,
                       "saturated_on");
    }
    SECTION("control well below threshold (saturated off)") {
        cross_validate(sw, /*v_ctrl=*/0.0, /*v_t1=*/1.0, /*v_t2=*/0.0,
                       "saturated_off");
    }
    SECTION("control exactly at threshold (peak dg/dv_ctrl)") {
        cross_validate(sw, /*v_ctrl=*/2.5, /*v_t1=*/1.0, /*v_t2=*/0.0,
                       "at_threshold");
    }
    SECTION("control mid-transition above threshold") {
        cross_validate(sw, /*v_ctrl=*/2.7, /*v_t1=*/2.0, /*v_t2=*/0.0,
                       "mid_transition_above");
    }
    SECTION("control mid-transition below threshold") {
        cross_validate(sw, /*v_ctrl=*/2.3, /*v_t1=*/2.0, /*v_t2=*/0.0,
                       "mid_transition_below");
    }
    SECTION("non-zero t2 reference") {
        cross_validate(sw, /*v_ctrl=*/3.0, /*v_t1=*/5.0, /*v_t2=*/3.0,
                       "asymmetric");
    }
    SECTION("zero v_sw (terminals at same potential)") {
        cross_validate(sw, /*v_ctrl=*/3.0, /*v_t1=*/2.0, /*v_t2=*/2.0,
                       "zero_v_sw");
    }
}

TEST_CASE("AD VCSwitch stamp matches manual with custom hysteresis",
          "[ad][vcswitch][stamp][hysteresis]") {
    VoltageControlledSwitch::Params p;
    p.v_threshold = 1.0;
    p.g_on = 500.0;
    p.g_off = 1e-7;
    p.hysteresis = 0.05;  // tighter transition than default
    VoltageControlledSwitch sw(p, "S_tight");

    SECTION("tight transition mid-band") {
        cross_validate(sw, /*v_ctrl=*/1.02, /*v_t1=*/3.0, /*v_t2=*/0.0,
                       "tight_mid_band");
    }
}
