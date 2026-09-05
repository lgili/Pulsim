// =============================================================================
// Phase 4 C.4 — gapped core → λ(i)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/models/gapped_core.hpp"

#include <cmath>

using namespace pulsim;
using Catch::Approx;

namespace {
models::GappedCore::Params etd29() {
    models::GappedCore::Params c;
    c.N = 25; c.Ae = 76e-6; c.le = 72e-3; c.lg = 0.5e-3;
    c.mu_r0 = 2000; c.B_sat = 0.35; c.p = 4; c.knots = 96;
    return c;
}
}  // namespace

TEST_CASE("GappedCore: reluctances and the unsaturated inductance",
          "[v2][c4][gapped_core][unit]") {
    const auto c = etd29();
    // Independent reference (python, first principles): R_core = 3.769e5,
    // R_gap = 5.235e6 A/Wb, L_unsat = 111.4 µH, L_air = 0.82 µH,
    // i(B_sat) = 6.37 A, λ(B_sat) = 0.665 mWb·t.
    CHECK(models::GappedCore::reluctance_core(c) == Approx(3.769e5).epsilon(1e-3));
    CHECK(models::GappedCore::reluctance_gap(c)  == Approx(5.235e6).epsilon(1e-3));
    CHECK(models::GappedCore::reluctance_gap(c) / models::GappedCore::reluctance_core(c)
          == Approx(13.9).epsilon(1e-2));
    CHECK(models::GappedCore::L_unsat(c) == Approx(111.4e-6).epsilon(1e-3));
    CHECK(models::GappedCore::L_air(c)   == Approx(0.82e-6).epsilon(2e-2));
    const Real lam_sat = c.N * c.Ae * c.B_sat;
    CHECK(lam_sat == Approx(0.665e-3).epsilon(1e-3));
    CHECK(models::GappedCore::current_at_flux(c, lam_sat) == Approx(6.37).epsilon(2e-3));
}

TEST_CASE("GappedCore: the table's L_0 is the reluctance L_unsat and its "
          "tail heads for the air value", "[v2][c4][gapped_core][unit]") {
    const auto c = etd29();
    const auto tab = models::GappedCore::make_table(c);
    CHECK(tab.L_0() == Approx(models::GappedCore::L_unsat(c)).epsilon(2e-3));
    // Below the knee the incremental inductance is flat (the gap
    // dominates); well past it, it has collapsed by an order of
    // magnitude and is still falling toward L_air.
    const Real L_low  = tab.inductance(1.0);
    const Real L_knee = tab.inductance(6.37);
    const Real L_high = tab.inductance(20.0);
    INFO("L(1 A) = " << L_low << "  L(6.37 A) = " << L_knee << "  L(20 A) = " << L_high);
    CHECK(L_low  == Approx(tab.L_0()).epsilon(5e-2));
    CHECK(L_knee < 0.85 * tab.L_0());
    CHECK(L_high < 0.25 * tab.L_0());
    CHECK(L_high > models::GappedCore::L_air(c));
    // The table is the explicit law, inverted for free: λ(i(λ)) = λ.
    for (const Real lam : {1e-4, 3e-4, 6.65e-4, 1.2e-3}) {
        const Real i = models::GappedCore::current_at_flux(c, lam);
        CHECK(tab.flux(i) == Approx(lam).epsilon(2e-3));
    }
}

TEST_CASE("GappedCore: no gap is a high, fragile inductance; more gap is "
          "lower and stiffer", "[v2][c4][gapped_core][unit]") {
    auto c = etd29();
    c.lg = 0;
    const Real L_nogap = models::GappedCore::L_unsat(c);
    c.lg = 0.5e-3;
    const Real L_half = models::GappedCore::L_unsat(c);
    c.lg = 1.0e-3;
    const Real L_one = models::GappedCore::L_unsat(c);
    CHECK(L_nogap > 10 * L_half);
    CHECK(L_half > L_one);
    // With no gap, the knee arrives at a far smaller current.
    c.lg = 0;
    const Real i_sat_nogap = models::GappedCore::current_at_flux(c, c.N * c.Ae * c.B_sat);
    c.lg = 0.5e-3;
    const Real i_sat_gap = models::GappedCore::current_at_flux(c, c.N * c.Ae * c.B_sat);
    // Un-gapped: H(B_sat) with μ_r halved to ~1000 gives ~0.8 A;
    // gapped: 6.4 A. The gap carries 14× the core's reluctance, so the
    // knee current scales by about that — a factor of 8 here, since
    // μ_r has already halved at B_sat.
    CHECK(i_sat_nogap < 0.2 * i_sat_gap);
}

TEST_CASE("GappedCore: refuses unit mistakes by name",
          "[v2][c4][gapped_core][unit]") {
    using Catch::Matchers::ContainsSubstring;
    auto c = etd29();
    c.Ae = 76.0;   // mm² typed as m²
    CHECK_THROWS_WITH(models::GappedCore::make_table(c, "T1"), ContainsSubstring("mm²"));
    c = etd29(); c.le = 72.0;
    CHECK_THROWS_WITH(models::GappedCore::make_table(c, "T1"), ContainsSubstring("mm typed"));
    c = etd29(); c.lg = 0.5;
    CHECK_THROWS_WITH(models::GappedCore::make_table(c, "T1"), ContainsSubstring("gap longer"));
    c = etd29(); c.N = 2.5;
    CHECK_THROWS_WITH(models::GappedCore::make_table(c, "T1"), ContainsSubstring("integer"));
    c = etd29(); c.B_sat = 0;
    CHECK_THROWS_WITH(models::GappedCore::make_table(c, "T1"), ContainsSubstring("B_sat"));
}
