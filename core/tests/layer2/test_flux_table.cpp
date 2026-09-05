// =============================================================================
// Phase 4 C.4 — monotone λ(i) flux table
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/models/flux_table.hpp"
#include "pulsim/models/saturable_inductor.hpp"

#include <cmath>
#include <vector>

using namespace pulsim;
using Catch::Approx;

namespace {
models::SaturableInductor::Params atan_law() {
    models::SaturableInductor::Params p;
    p.L_0 = 1e-3; p.I_sat = 5.0; p.L_residual = 5e-5;
    return p;
}
/// Tabulate the analytic law with sqrt-spaced knots on [0, i_max].
models::FluxTable tabulate(Size n, Real i_max) {
    const auto p = atan_law();
    std::vector<Real> i, lam;
    for (Size k = 0; k < n; ++k) {
        const Real u = static_cast<Real>(k) / static_cast<Real>(n - 1);
        const Real ik = i_max * u * u;
        i.push_back(ik);
        lam.push_back(models::SaturableInductor::flux(ik, p));
    }
    return models::FluxTable(i, lam, "test");
}
}  // namespace

TEST_CASE("FluxTable: reproduces the analytic law, value and derivative",
          "[v2][c4][flux_table][unit]") {
    const auto p = atan_law();
    const auto tab = tabulate(64, 10.0 * p.I_sat);
    Real e_lam = 0, e_L = 0, e_L_knee = 0;
    for (int k = 1; k <= 20000; ++k) {
        const Real i = 10.0 * p.I_sat * k / 20000.0;
        const Real lam_a = models::SaturableInductor::flux(i, p);
        const Real L_a = models::SaturableInductor::current<Real>(&i, p);
        e_lam = std::max(e_lam, std::abs(tab.flux(i) - lam_a) / lam_a);
        const Real eL = std::abs(tab.inductance(i) - L_a) / L_a;
        e_L = std::max(e_L, eL);
        if (i > 0.3 * p.I_sat && i < 3.0 * p.I_sat) e_L_knee = std::max(e_L_knee, eL);
    }
    INFO("max rel err: lambda " << e_lam << "  L " << e_L << "  L(knee) " << e_L_knee);
    CHECK(e_lam < 5e-5);
    CHECK(e_L < 3e-3);
    CHECK(e_L_knee < 3e-3);
    CHECK(tab.L_0() == Approx(p.L_0).epsilon(1e-4));
}

TEST_CASE("FluxTable: exact knot slopes beat estimated ones by two orders",
          "[v2][c4][flux_table][unit]") {
    const auto p = atan_law();
    const Size n = 64;
    const Real i_max = 10.0 * p.I_sat;
    std::vector<Real> i, lam, L;
    for (Size k = 0; k < n; ++k) {
        const Real u = static_cast<Real>(k) / static_cast<Real>(n - 1);
        const Real ik = i_max * u * u;
        i.push_back(ik);
        lam.push_back(models::SaturableInductor::flux(ik, p));
        L.push_back(models::SaturableInductor::current<Real>(&ik, p));
    }
    const Real L_tail = models::SaturableInductor::current<Real>(&i_max, p);
    const models::FluxTable exact(i, lam, L, L_tail, "exact");
    const models::FluxTable est(i, lam, "estimated");
    Real e_exact = 0, e_est = 0;
    for (int k = 1; k <= 20000; ++k) {
        const Real x = i_max * k / 20000.0;
        const Real La = models::SaturableInductor::current<Real>(&x, p);
        e_exact = std::max(e_exact, std::abs(exact.inductance(x) - La) / La);
        e_est   = std::max(e_est,   std::abs(est.inductance(x)   - La) / La);
    }
    INFO("max rel err in L: exact slopes " << e_exact << ", estimated " << e_est);
    CHECK(e_exact < 1e-4);
    CHECK(e_est > 10 * e_exact);
    CHECK(exact.clamped_tangents() == 0);
    CHECK(exact.max_fritsch_carlson_radius2() <= 9.0);
}

TEST_CASE("FluxTable: a tangent outside the monotone region is clamped, "
          "not trusted", "[v2][c4][flux_table][unit]") {
    // Wildly wrong user slopes: 10× the secant on a straight line.
    std::vector<Real> i{0, 1, 2, 3}, lam{0, 1e-3, 2e-3, 3e-3}, L{1e-2, 1e-2, 1e-2, 1e-2};
    const models::FluxTable tab(i, lam, L, 1e-3, "kinked");
    CHECK(tab.clamped_tangents() > 0);
    Real prev = -1;
    for (int k = 0; k <= 3000; ++k) {
        const Real x = 3.0 * k / 3000.0;
        CHECK(tab.inductance(x) >= 0.0);
        CHECK(tab.flux(x) >= prev);
        prev = tab.flux(x);
    }
}

TEST_CASE("FluxTable: odd in i, even L, continuous at the origin and at "
          "the last knot", "[v2][c4][flux_table][unit]") {
    const auto tab = tabulate(48, 30.0);
    for (const Real i : {0.01, 0.7, 4.0, 12.0, 29.9}) {
        CHECK(tab.flux(-i) == Approx(-tab.flux(i)).epsilon(1e-14));
        CHECK(tab.inductance(-i) == Approx(tab.inductance(i)).epsilon(1e-14));
        CHECK(tab.inductance(i) > 0.0);
    }
    CHECK(tab.flux(0.0) == 0.0);
    // Origin: λ(±ε) ≈ ±L_0 ε, no kink.
    const Real eps = 1e-7;
    CHECK((tab.flux(eps) - tab.flux(-eps)) / (2 * eps) == Approx(tab.L_0()).epsilon(1e-6));
    // Past the last knot: linear at L_residual, continuous in value
    // and slope with the last segment.
    const Real im = tab.i_max();
    CHECK(tab.flux(im * 1.5) == Approx(tab.flux(im) + tab.L_residual() * 0.5 * im).epsilon(1e-12));
    CHECK(tab.inductance(im - 1e-9) == Approx(tab.inductance(im + 1e-9)).epsilon(1e-6));
}

TEST_CASE("FluxTable: monotone even where a spline would overshoot",
          "[v2][c4][flux_table][unit]") {
    // A hard knee: L drops 1000× between two knots. A cubic spline
    // through these overshoots and yields a negative slope; PCHIP
    // must not.
    std::vector<Real> i{0, 1, 2, 2.05, 10, 20};
    std::vector<Real> lam{0, 1e-3, 2e-3, 2.05e-3, 2.06e-3, 2.07e-3};
    const models::FluxTable tab(i, lam);
    Real L_min = 1e9;
    for (int k = 0; k <= 4000; ++k) {
        const Real x = 20.0 * k / 4000.0;
        L_min = std::min(L_min, tab.inductance(x));
    }
    CHECK(L_min >= 0.0);
    // And λ itself never decreases.
    Real prev = -1;
    for (int k = 0; k <= 4000; ++k) {
        const Real x = 20.0 * k / 4000.0;
        const Real f = tab.flux(x);
        CHECK(f >= prev);
        prev = f;
    }
}

TEST_CASE("FluxTable: refuses bad tables by name",
          "[v2][c4][flux_table][unit]") {
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(models::FluxTable({0, 1}, {0, 1e-3}, "Lx"),
                      ContainsSubstring("at least 3 knots"));
    CHECK_THROWS_WITH(models::FluxTable({0.5, 1, 2}, {0, 1e-3, 2e-3}, "Lx"),
                      ContainsSubstring("origin"));
    CHECK_THROWS_WITH(models::FluxTable({0, 1, 2}, {0, 2e-3, 1.5e-3}, "Lx"),
                      ContainsSubstring("NEGATIVE"));
    CHECK_THROWS_WITH(models::FluxTable({0, 2, 1}, {0, 1e-3, 2e-3}, "Lx"),
                      ContainsSubstring("strictly increasing"));
    CHECK_THROWS_WITH(models::FluxTable({0, 1, 2}, {0, 1e-3}, "Lx"),
                      ContainsSubstring("current knots but"));
}
