// =============================================================================
// Test: AD-derived diode stamp matches manual stamp (Phase 2 of
// add-automatic-differentiation)
// =============================================================================
//
// Validates the migration of `IdealDiode` to use `forward_current_behavioral`
// + `stamp_jacobian_via_ad`. The contract: at every operating point we
// exercise, the AD-derived J entries and residual contribution must agree
// with the legacy `stamp_jacobian_impl` (Behavioral mode) within 1e-12
// relative tolerance.
//
// We sweep operating points across the diode's full operating range:
//   * deep reverse bias (v_diode = −5 V, dominated by g_off)
//   * mid reverse (v_diode = −0.05 V, tanh transition zone)
//   * exactly at threshold (v_diode = 0)
//   * mid forward (v_diode = +0.05 V, transition zone)
//   * strong forward (v_diode = +5 V, saturated to g_on)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/ideal_diode.hpp"

#include <Eigen/Sparse>
#include <array>
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

struct StampResult {
    Eigen::SparseMatrix<Real> J;
    Eigen::VectorXd f;
};

[[nodiscard]] StampResult stamp_manual(IdealDiode& d, Real v_anode, Real v_cathode) {
    StampResult r;
    r.J.resize(2, 2);
    r.f = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd x(2);
    x << v_anode, v_cathode;
    std::array<Index, 2> nodes{0, 1};
    d.stamp_jacobian(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

[[nodiscard]] StampResult stamp_ad(IdealDiode& d, Real v_anode, Real v_cathode) {
    StampResult r;
    r.J.resize(2, 2);
    r.f = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd x(2);
    x << v_anode, v_cathode;
    std::array<Index, 2> nodes{0, 1};
    d.stamp_jacobian_via_ad(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

void cross_validate(IdealDiode& diode, Real v_anode, Real v_cathode) {
    const auto manual = stamp_manual(diode, v_anode, v_cathode);
    const auto ad     = stamp_ad(diode, v_anode, v_cathode);

    INFO("Operating point: v_anode = " << v_anode << ", v_cathode = " << v_cathode);
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

TEST_CASE("AD diode stamp matches manual stamp across the operating range",
          "[ad][diode][stamp][cross_validation]") {
    IdealDiode d(/*g_on=*/1e3, /*g_off=*/1e-9);
    d.set_smoothing(0.05);  // tanh-smoothed Behavioral mode

    SECTION("deep reverse bias") {
        cross_validate(d, /*v_a=*/0.0, /*v_c=*/5.0);
    }
    SECTION("mid reverse (transition zone)") {
        cross_validate(d, 0.0, 0.05);
    }
    SECTION("at threshold (v_diode = 0)") {
        cross_validate(d, 0.0, 0.0);
    }
    SECTION("mid forward (transition zone)") {
        cross_validate(d, 0.05, 0.0);
    }
    SECTION("strong forward bias") {
        cross_validate(d, 5.0, 0.0);
    }
    SECTION("non-zero cathode reference") {
        cross_validate(d, 1.5, 1.0);
    }
}

TEST_CASE("AD diode stamp matches manual stamp with sharp v_smooth = 0",
          "[ad][diode][stamp][sharp]") {
    IdealDiode d(/*g_on=*/1e3, /*g_off=*/1e-9);
    d.set_smoothing(0.0);  // sharp Behavioral fallback

    SECTION("forward bias → g_on") {
        cross_validate(d, 5.0, 0.0);
    }
    SECTION("reverse bias → g_off") {
        cross_validate(d, 0.0, 5.0);
    }
}
