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
    c.mu_r0 = 2000; c.B_sat = 0.35; c.knots = 128;
    return c;
}
}  // namespace

TEST_CASE("GappedCore: reluctances and the unsaturated inductance",
          "[v2][c4][gapped_core][unit]") {
    const auto c = etd29();
    // Independent reference (python, first principles): R_core = 3.769e5,
    // R_gap = 5.235e6 A/Wb, L_unsat = 111.4 µH, L_air = 0.82 µH.
    CHECK(models::GappedCore::reluctance_core(c) == Approx(3.769e5).epsilon(1e-3));
    CHECK(models::GappedCore::reluctance_gap(c)  == Approx(5.235e6).epsilon(1e-3));
    CHECK(models::GappedCore::reluctance_gap(c) / models::GappedCore::reluctance_core(c)
          == Approx(13.9).epsilon(1e-2));
    CHECK(models::GappedCore::L_unsat(c) == Approx(111.4e-6).epsilon(1e-3));
    CHECK(models::GappedCore::L_air(c)   == Approx(0.82e-6).epsilon(2e-2));
    // The law's initial slope IS μ_r0, and the differential inductance
    // at rest IS the reluctance value — two routes, one number.
    CHECK(models::GappedCore::inductance_of_H(c, 0.0)
          == Approx(models::GappedCore::L_unsat(c)).epsilon(1e-9));
    CHECK(models::GappedCore::H_0(c) == Approx(139.3).epsilon(1e-2));
    // Knee: M = 0.96 M_s at H = 2 H₀, a few amps on this core.
    const Real i_knee = models::GappedCore::knee_current(c);
    INFO("knee current = " << i_knee << " A");
    CHECK(i_knee > 4.0);
    CHECK(i_knee < 8.0);
}

TEST_CASE("GappedCore: the material law is bounded, monotone, and ends "
          "in air", "[v2][c4][gapped_core][unit]") {
    const auto c = etd29();
    const Real Ms = models::GappedCore::M_s(c);
    Real prev_B = -1, prev_i = -1;
    for (int k = 0; k <= 400; ++k) {
        const Real H = 60.0 * models::GappedCore::H_0(c) * k / 400.0;
        const Real M = models::GappedCore::magnetisation(c, H);
        CHECK(M >= 0.0);
        CHECK(M <= Ms * (1 + 1e-12));                         // C2: bounded
        const Real B = models::GappedCore::B_of_H(c, H);
        const Real i = models::GappedCore::current_of_H(c, H);
        CHECK(B > prev_B);                                     // C1: monotone
        CHECK(i > prev_i);
        prev_B = B; prev_i = i;
        // L never below the air floor.
        CHECK(models::GappedCore::inductance_of_H(c, H)
              >= models::GappedCore::L_air(c) * (1 - 1e-12));
    }
    // Deep in saturation the differential inductance IS air.
    CHECK(models::GappedCore::inductance_of_H(c, 50 * models::GappedCore::H_0(c))
          == Approx(models::GappedCore::L_air(c)).epsilon(1e-6));
}

TEST_CASE("GappedCore: the table's L_0 is L_unsat, its tail is L_air, and "
          "it inverts the explicit law exactly", "[v2][c4][gapped_core][unit]") {
    const auto c = etd29();
    const auto tab = models::GappedCore::make_table(c);
    CHECK(tab.L_0() == Approx(models::GappedCore::L_unsat(c)).epsilon(1e-9));
    CHECK(tab.L_residual() == Approx(models::GappedCore::L_air(c)).epsilon(1e-12));
    // No jump where the table hands over to the tail.
    const Real im = tab.i_max();
    CHECK(tab.inductance(im - 1e-9) == Approx(tab.inductance(im + 1e-9)).epsilon(2e-2));
    // Below the knee flat (the gap dominates); past it, collapsed.
    const Real i_knee = models::GappedCore::knee_current(c);
    const Real L_low  = tab.inductance(0.3 * i_knee);
    const Real L_knee = tab.inductance(i_knee);
    const Real L_4H0  = tab.inductance(
        models::GappedCore::current_of_H(c, 4 * models::GappedCore::H_0(c)));
    const Real L_high = tab.inductance(3.0 * i_knee);
    INFO("L(0.3 knee) = " << L_low << "  L(knee) = " << L_knee
         << "  L(4H0) = " << L_4H0 << "  L(3 knee) = " << L_high);
    CHECK(L_low  == Approx(tab.L_0()).epsilon(2e-2));
    // The differential L halves where the core's reluctance has grown
    // to equal the gap's: 1 + dM/dH = le/(lg + le/μ_r0) = 134 on this
    // core, i.e. H = 2.03 H₀ — the knee current, by construction.
    CHECK(L_knee == Approx(0.53 * tab.L_0()).epsilon(0.1));
    // Two H₀ further on it is a few percent of L_0 …
    CHECK(L_4H0 < 0.05 * tab.L_0());
    // … and past three knee currents it is air.
    CHECK(L_high < 1.05 * models::GappedCore::L_air(c));
    // λ(i(H)) = λ(H) — the table is the explicit law, inverted for
    // free, with the EXACT slope: 1e-4 rather than PCHIP's 1e-2.
    Real e_lam = 0, e_L = 0;
    for (int k = 1; k <= 2000; ++k) {
        const Real H = 6.0 * models::GappedCore::H_0(c) * k / 2000.0;
        const Real i = models::GappedCore::current_of_H(c, H);
        const Real lam = models::GappedCore::flux_of_H(c, H);
        const Real L = models::GappedCore::inductance_of_H(c, H);
        e_lam = std::max(e_lam, std::abs(tab.flux(i) - lam) / lam);
        e_L = std::max(e_L, std::abs(tab.inductance(i) - L) / L);
    }
    INFO("max rel err: λ " << e_lam << "  L " << e_L);
    CHECK(e_lam < 1e-5);
    CHECK(e_L < 2e-3);
}

TEST_CASE("GappedCore: no gap is a high, fragile inductance; more gap is "
          "lower and stiffer", "[v2][c4][gapped_core][unit]") {
    auto c = etd29();
    c.lg = 0;
    const Real L_nogap = models::GappedCore::L_unsat(c);
    const Real i_knee_nogap = models::GappedCore::knee_current(c);
    c.lg = 0.5e-3;
    const Real L_half = models::GappedCore::L_unsat(c);
    const Real i_knee_gap = models::GappedCore::knee_current(c);
    c.lg = 1.0e-3;
    const Real L_one = models::GappedCore::L_unsat(c);
    CHECK(L_nogap > 10 * L_half);
    CHECK(L_half > L_one);
    // Same knee (same H, same B): the gapped core needs the gap's MMF
    // on top of the core's — (B/μ₀)·lg against H·le, 5.4 A against
    // 0.8 A here, a factor of 7.7. (The reluctance ratio of 13.9 is
    // the small-signal figure at H → 0; at the knee the core's own
    // H·le share has grown.)
    CHECK(i_knee_gap > 5 * i_knee_nogap);
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
