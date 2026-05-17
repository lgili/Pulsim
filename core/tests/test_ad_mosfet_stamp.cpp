// =============================================================================
// Test: AD-derived MOSFET stamp matches manual stamp (Phase 2 of
// add-automatic-differentiation)
// =============================================================================
//
// Cross-validates `MOSFET::stamp_jacobian_via_ad` against the legacy manual
// `stamp_jacobian_impl` (Behavioral mode) on all three Shichman-Hodges
// regions (cutoff, triode, saturation), both NMOS and PMOS, plus an
// asymmetric source-reference operating point.
//
// Both stamps must agree on every J entry and `f` row to within 1e-12
// absolute tolerance — this is Gate G.1 of the
// `add-automatic-differentiation` change.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/mosfet.hpp"

#include <Eigen/Sparse>
#include <array>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

struct StampResult {
    Eigen::SparseMatrix<Real> J;
    Eigen::VectorXd f;
};

[[nodiscard]] StampResult stamp_manual(MOSFET& m, Real v_g, Real v_d, Real v_s) {
    StampResult r;
    r.J.resize(3, 3);
    r.f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << v_g, v_d, v_s;
    std::array<Index, 3> nodes{0, 1, 2};
    m.stamp_jacobian(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

[[nodiscard]] StampResult stamp_ad(MOSFET& m, Real v_g, Real v_d, Real v_s) {
    StampResult r;
    r.J.resize(3, 3);
    r.f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << v_g, v_d, v_s;
    std::array<Index, 3> nodes{0, 1, 2};
    m.stamp_jacobian_via_ad(r.J, r.f, x, nodes);
    r.J.makeCompressed();
    return r;
}

void cross_validate(MOSFET& mosfet, Real v_g, Real v_d, Real v_s,
                    const char* label) {
    const auto manual = stamp_manual(mosfet, v_g, v_d, v_s);
    const auto ad     = stamp_ad(mosfet, v_g, v_d, v_s);

    INFO("Op-point: " << label);
    INFO("  v_g = " << v_g << ", v_d = " << v_d << ", v_s = " << v_s);

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

TEST_CASE("AD MOSFET stamp matches manual across Shichman-Hodges regions",
          "[ad][mosfet][stamp][cross_validation]") {
    MOSFET::Params params{
        .vth = 2.0,
        .kp = 0.1,
        .lambda = 0.01,
        .g_off = 1e-12,
        .is_nmos = true,
        .g_on = 1e3,
    };
    MOSFET m(params, "M_nmos");

    SECTION("cutoff (Vgs ≤ Vth)") {
        cross_validate(m, /*v_g=*/1.0, /*v_d=*/5.0, /*v_s=*/0.0, "cutoff");
    }
    SECTION("triode / linear region (Vds < Vgs − Vth)") {
        cross_validate(m, /*v_g=*/5.0, /*v_d=*/1.0, /*v_s=*/0.0, "triode");
    }
    SECTION("saturation (Vds ≥ Vgs − Vth)") {
        cross_validate(m, /*v_g=*/5.0, /*v_d=*/10.0, /*v_s=*/0.0, "saturation");
    }
    SECTION("asymmetric source reference") {
        cross_validate(m, /*v_g=*/8.0, /*v_d=*/12.0, /*v_s=*/3.0, "asymmetric");
    }
    SECTION("near saturation boundary (Vds = Vgs − Vth)") {
        // Vov = 3, Vds = 3 → exactly on the boundary; both branches should
        // give the same id and partials at this point. Test ε-perturbations.
        cross_validate(m, 5.0, 3.0001, 0.0, "saturation_boundary_above");
        cross_validate(m, 5.0, 2.9999, 0.0, "saturation_boundary_below");
    }
}

TEST_CASE("AD MOSFET stamp matches manual across Shichman-Hodges regions (PMOS)",
          "[ad][mosfet][stamp][cross_validation][pmos]") {
    // PMOS cross-validation manual vs AD. The legacy concern documented in
    // the prior PMOS section was a sign mismatch in the `i_eq` linearization
    // terms; that has been resolved by the manual-stamp Phase-8 rework which
    // applies the chain rule in actual coords (matching the AD path's
    // Norton companion form). This test pins the agreement so the
    // regression cannot return.
    MOSFET::Params params{
        .vth = 2.0,
        .kp = 0.1,
        .lambda = 0.01,
        .g_off = 1e-12,
        .is_nmos = false,
        .g_on = 1e3,
    };
    MOSFET m(params, "M_pmos_xv");

    SECTION("PMOS cutoff") {
        cross_validate(m, /*v_g=*/-1.0, /*v_d=*/-5.0, /*v_s=*/0.0, "pmos_cutoff");
    }
    SECTION("PMOS saturation") {
        cross_validate(m, /*v_g=*/-5.0, /*v_d=*/-10.0, /*v_s=*/0.0, "pmos_saturation");
    }
    SECTION("PMOS triode") {
        cross_validate(m, /*v_g=*/-5.0, /*v_d=*/-1.0, /*v_s=*/0.0, "pmos_triode");
    }
    SECTION("PMOS asymmetric source") {
        cross_validate(m, /*v_g=*/-8.0, /*v_d=*/-12.0, /*v_s=*/-3.0, "pmos_asymmetric");
    }
}

TEST_CASE("AD MOSFET stamp matches manual for PMOS",
          "[ad][mosfet][stamp][pmos]") {
    MOSFET::Params params{
        .vth = 2.0,
        .kp = 0.1,
        .lambda = 0.01,
        .g_off = 1e-12,
        .is_nmos = false,   // PMOS
        .g_on = 1e3,
    };
    MOSFET m(params, "M_pmos");

    // Independent ground-truth check: cross-validates the AD-derived
    // partials against centered finite differences regardless of manual
    // stamp behavior. A complementary case above
    // (`AD MOSFET stamp matches manual across Shichman-Hodges regions
    // (PMOS)`) pins the manual ↔ AD equivalence at 1e-12 — the legacy
    // PMOS sign issue called out in earlier revisions of this comment
    // was resolved by the Phase-8 smooth-region rework which applies
    // the chain rule in actual coords, matching the AD Norton companion
    // form. This test keeps the FD-based safety net so any future
    // change to the residual template is checked against ground truth.

    auto i_at = [&](Real v_g, Real v_d, Real v_s) -> Real {
        return m.drain_current_behavioral<Real>(v_g, v_d, v_s);
    };

    auto check_partials = [&](Real v_g, Real v_d, Real v_s, const char* label) {
        INFO("PMOS op-point: " << label
             << " v_g=" << v_g << " v_d=" << v_d << " v_s=" << v_s);

        const auto ad = stamp_ad(m, v_g, v_d, v_s);

        // AD-derived ∂id/∂(v_g, v_d, v_s) live in J(drain, ·).
        const Real ad_di_dvg = ad.J.coeff(1, 0);
        const Real ad_di_dvd = ad.J.coeff(1, 1);
        const Real ad_di_dvs = ad.J.coeff(1, 2);

        const Real h = 1e-6;
        const Real fd_di_dvg = (i_at(v_g + h, v_d, v_s) - i_at(v_g - h, v_d, v_s)) / (2 * h);
        const Real fd_di_dvd = (i_at(v_g, v_d + h, v_s) - i_at(v_g, v_d - h, v_s)) / (2 * h);
        const Real fd_di_dvs = (i_at(v_g, v_d, v_s + h) - i_at(v_g, v_d, v_s - h)) / (2 * h);

        CHECK(ad_di_dvg == Approx(fd_di_dvg).margin(1e-6));
        CHECK(ad_di_dvd == Approx(fd_di_dvd).margin(1e-6));
        CHECK(ad_di_dvs == Approx(fd_di_dvs).margin(1e-6));
    };

    SECTION("PMOS cutoff") {
        check_partials(-1.0, -5.0, 0.0, "pmos_cutoff");
    }
    SECTION("PMOS saturation") {
        check_partials(-5.0, -10.0, 0.0, "pmos_saturation");
    }
    SECTION("PMOS triode") {
        check_partials(-5.0, -1.0, 0.0, "pmos_triode");
    }
}
