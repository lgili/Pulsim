// =============================================================================
// Test: AD-derived IGBT stamp matches manual stamp (Phase 2 of
// add-automatic-differentiation)
// =============================================================================
//
// Cross-validates `IGBT::stamp_jacobian_via_ad` against the legacy manual
// `stamp_jacobian_impl` (Behavioral mode) across cutoff, conducting, and
// "saturated" (Vce > Vce_sat) regions plus an asymmetric emitter reference.
//
// Both stamps must agree on every J entry and `f` row to within 1e-12
// absolute tolerance — Gate G.1 of `add-automatic-differentiation` for
// the IGBT.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/igbt.hpp"

#include <Eigen/Sparse>
#include <array>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

struct StampResult {
    Eigen::SparseMatrix<Real> J;
    Eigen::VectorXd f;
};

[[nodiscard]] StampResult stamp_manual(IGBT& q, Real v_g, Real v_c, Real v_e) {
    StampResult r;
    r.J.resize(3, 3);
    r.f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << v_g, v_c, v_e;
    std::array<Index, 3> nodes{0, 1, 2};
    q.stamp_jacobian(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

[[nodiscard]] StampResult stamp_ad(IGBT& q, Real v_g, Real v_c, Real v_e) {
    StampResult r;
    r.J.resize(3, 3);
    r.f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << v_g, v_c, v_e;
    std::array<Index, 3> nodes{0, 1, 2};
    q.stamp_jacobian_via_ad(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

void cross_validate(IGBT& igbt, Real v_g, Real v_c, Real v_e, const char* label) {
    const auto manual = stamp_manual(igbt, v_g, v_c, v_e);
    const auto ad     = stamp_ad(igbt, v_g, v_c, v_e);

    INFO("Op-point: " << label
         << " v_g=" << v_g << " v_c=" << v_c << " v_e=" << v_e);

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

TEST_CASE("AD IGBT stamp matches manual across operating regions",
          "[ad][igbt][stamp][cross_validation]") {
    IGBT::Params params{
        .vth = 5.0,
        .g_on = 1e4,
        .g_off = 1e-12,
        .v_ce_sat = 1.5,
    };
    IGBT q(params, "Q1");

    SECTION("cutoff (Vge < Vth)") {
        cross_validate(q, /*v_g=*/0.0, /*v_c=*/100.0, /*v_e=*/0.0, "cutoff");
    }
    SECTION("on-state, Vce below Vce_sat") {
        cross_validate(q, /*v_g=*/15.0, /*v_c=*/1.0, /*v_e=*/0.0, "on_below_sat");
    }
    SECTION("on-state, Vce well above Vce_sat") {
        cross_validate(q, /*v_g=*/15.0, /*v_c=*/10.0, /*v_e=*/0.0, "on_above_sat");
    }
    SECTION("Vce ≤ 0 → off (one of the AND conditions for conducting fails)") {
        cross_validate(q, /*v_g=*/15.0, /*v_c=*/0.0, /*v_e=*/5.0, "vce_negative");
    }
    SECTION("asymmetric emitter reference") {
        cross_validate(q, /*v_g=*/20.0, /*v_c=*/15.0, /*v_e=*/5.0, "asymmetric");
    }
}
