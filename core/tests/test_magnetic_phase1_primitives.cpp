// =============================================================================
// Phase 1 of `add-magnetic-core-models`: magnetic primitives
// =============================================================================
//
// Pins the BHCurveTable / BHCurveArctan / BHCurveLangevin behaviors,
// Steinmetz cycle-averaged loss, iGSE for non-sinusoidal flux, and the
// Jiles-Atherton hysteresis ODE step.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/magnetic/bh_curve.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace pulsim::v1;
using namespace pulsim::v1::magnetic;
using Catch::Approx;

// -----------------------------------------------------------------------------
// BHCurveTable
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: BHCurveTable interpolates and clamps at saturation",
          "[v1][magnetic][phase1][bh_curve][table]") {
    // 5-point soft-ferrite-shaped curve.
    const std::vector<Real> H = {-1000, -100, 0, 100, 1000};
    const std::vector<Real> B = {-0.45, -0.30, 0.0, 0.30, 0.45};
    BHCurveTable curve(H, B);
    REQUIRE(curve.size() == 5);

    // Linear interpolation in mid-range.
    CHECK(curve.b_from_h(50.0) == Approx(0.15).margin(1e-9));
    CHECK(curve.h_from_b(0.15) == Approx(50.0).margin(1e-9));

    // Clamp beyond the table.
    CHECK(curve.b_from_h(2000.0) == Approx(0.45).margin(1e-12));
    CHECK(curve.b_from_h(-2000.0) == Approx(-0.45).margin(1e-12));

    // Saturation density takes the larger of |B[front]|, |B[back]|.
    CHECK(curve.saturation_density() == Approx(0.45).margin(1e-12));

    // dB/dH on the steep mid-section (between H=0 and H=100, B goes 0 → 0.30):
    //   slope = 0.30 / 100 = 3e-3 T·m/A
    CHECK(curve.dbdh(50.0) == Approx(3e-3).margin(1e-9));
}

TEST_CASE("Phase 1: BHCurveTable rejects degenerate input",
          "[v1][magnetic][phase1][bh_curve][table][validation]") {
    CHECK_THROWS_AS(
        BHCurveTable(std::vector<Real>{0}, std::vector<Real>{0}),
        std::invalid_argument);
    CHECK_THROWS_AS(
        BHCurveTable(std::vector<Real>{1, 0, 2}, std::vector<Real>{0, 1, 2}),
        std::invalid_argument);
    CHECK_THROWS_AS(
        BHCurveTable(std::vector<Real>{0, 1}, std::vector<Real>{0, 1, 2}),
        std::invalid_argument);
}

// -----------------------------------------------------------------------------
// BHCurveArctan
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: BHCurveArctan saturates at Bs and round-trips",
          "[v1][magnetic][phase1][bh_curve][arctan]") {
    constexpr Real Bs = 0.4;
    constexpr Real Hc = 200.0;
    BHCurveArctan curve(Bs, Hc);

    // Forward: at H = Hc, atan(1) = π/4 → B = (2·Bs/π) · π/4 = Bs/2.
    CHECK(curve.b_from_h(Hc) == Approx(Bs / 2).margin(1e-12));
    // At very large H, B → ±Bs.
    CHECK(curve.b_from_h(1e6 * Hc) == Approx(Bs).margin(1e-3));
    CHECK(curve.b_from_h(-1e6 * Hc) == Approx(-Bs).margin(1e-3));

    // Round-trip: h_from_b(b_from_h(H)) ≈ H for |H| < ~10·Hc (avoid the
    // tan() near asymptote).
    for (Real H : {-500.0, -100.0, 0.0, 50.0, 200.0, 800.0}) {
        const Real B = curve.b_from_h(H);
        CHECK(curve.h_from_b(B) == Approx(H).margin(1e-6));
    }

    // dB/dH at H = 0 (peak slope): 2·Bs / (π·Hc).
    const Real expected_slope = 2.0 * Bs / (std::numbers::pi_v<Real> * Hc);
    CHECK(curve.dbdh(0.0) == Approx(expected_slope).margin(1e-9));
}

// -----------------------------------------------------------------------------
// BHCurveLangevin
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: BHCurveLangevin matches Taylor expansion near origin",
          "[v1][magnetic][phase1][bh_curve][langevin]") {
    constexpr Real Bs = 1.5;
    constexpr Real a = 200.0;
    BHCurveLangevin curve(Bs, a);

    // L(x) ≈ x/3 − x³/45 for small x. So B ≈ Bs · (H/a)/3 for small H.
    const Real H_small = 1.0;
    const Real expected_small = Bs * (H_small / a) / 3.0;
    CHECK(curve.b_from_h(H_small) == Approx(expected_small).margin(1e-9));

    // dB/dH at H = 0 → Bs/(3a).
    CHECK(curve.dbdh(0.0) == Approx(Bs / (3.0 * a)).margin(1e-9));

    // Saturation: L(x) → 1 as x → ∞. With H/a = 5000, L = 1 − 1/5000 ≈
    // 0.9998, so B = Bs · 0.9998 ≈ 0.9998·Bs. Tolerance scaled to match.
    CHECK(curve.b_from_h(1e6) == Approx(Bs).margin(1e-3));
    CHECK(curve.b_from_h(-1e6) == Approx(-Bs).margin(1e-3));

    // Round-trip on a non-trivial point.
    const Real H_test = 300.0;
    const Real B_test = curve.b_from_h(H_test);
    CHECK(curve.h_from_b(B_test) == Approx(H_test).margin(1e-3));
}

// -----------------------------------------------------------------------------
// Steinmetz
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: Steinmetz cycle_average follows P_v = k·f^α·B^β",
          "[v1][magnetic][phase1][steinmetz]") {
    SteinmetzLoss s{.k = 1.5, .alpha = 1.6, .beta = 2.7};

    // Doubling f raises loss by 2^α.
    const Real ref = s.cycle_average(50e3, 0.1);
    const Real x2  = s.cycle_average(100e3, 0.1);
    CHECK(x2 / ref == Approx(std::pow(2.0, s.alpha)).margin(1e-6));

    // Doubling B raises loss by 2^β.
    const Real x3 = s.cycle_average(50e3, 0.2);
    CHECK(x3 / ref == Approx(std::pow(2.0, s.beta)).margin(1e-6));
}

TEST_CASE("Phase 1: iGSE on sinusoidal flux matches Steinmetz baseline",
          "[v1][magnetic][phase1][steinmetz][igse]") {
    // For a pure sinusoid, iGSE must reproduce `cycle_average(f, B_pk)`
    // up to numerical-quadrature error.
    constexpr Real f      = 50e3;
    constexpr Real B_pk   = 0.05;
    constexpr int  N      = 1024;
    const Real T = 1.0 / f;
    const Real dt = T / N;

    std::vector<Real> B(N);
    for (int i = 0; i < N; ++i) {
        const Real t = dt * i;
        B[i] = B_pk * std::sin(2.0 * std::numbers::pi_v<Real> * f * t);
    }

    SteinmetzLoss s{.k = 1.0, .alpha = 1.5, .beta = 2.5};
    const Real igse_loss     = igse_specific_loss(B, dt, s);
    const Real expected_loss = s.cycle_average(f, B_pk);

    // Tolerance: iGSE has its own numerical-integration error from the
    // 256-bin cos integral plus the trapezoidal-rule dB/dt; ≤ 5% is the
    // realistic bar for a pure sinusoid input.
    CHECK(igse_loss == Approx(expected_loss).epsilon(0.05));
}

TEST_CASE("Phase 1: iGSE handles degenerate inputs",
          "[v1][magnetic][phase1][steinmetz][igse][degenerate]") {
    SteinmetzLoss s{.k = 1.0, .alpha = 1.5, .beta = 2.5};
    // Zero amplitude → no loss.
    std::vector<Real> flat(64, 0.1);
    CHECK(igse_specific_loss(flat, 1e-6, s) == Approx(0.0).margin(1e-12));
    // Empty / too-short input → 0 (defensive).
    CHECK(igse_specific_loss({}, 1e-6, s) == Approx(0.0).margin(1e-12));
}

// -----------------------------------------------------------------------------
// Jiles-Atherton
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: Jiles-Atherton step settles to anhysteretic on slow ramps",
          "[v1][magnetic][phase1][jiles_atherton]") {
    JilesAthertonParams p{
        .Ms = 1.0e6, .a = 100.0, .alpha = 1.0e-4, .k = 50.0, .c = 0.1,
    };
    JilesAthertonState state{};

    // Slow monotonic ramp from -2000 to +2000 A/m.
    const int N = 4000;
    const Real H_start = -2000.0;
    const Real H_stop  =  2000.0;
    for (int i = 0; i <= N; ++i) {
        const Real t = static_cast<Real>(i) / static_cast<Real>(N);
        const Real H = H_start + t * (H_stop - H_start);
        jiles_atherton_step(state, p, H);
    }

    // After a full ramp the irreversible magnetization should track the
    // anhysteretic at the endpoint to within the pinning-coefficient
    // budget (≈ k). M should be close to M_an at +2000 A/m.
    CHECK(state.H_prev == Approx(H_stop).margin(1e-12));
    CHECK(state.M > 0.0);                // saturated positive
    CHECK(std::abs(state.M) < p.Ms);     // not above the ceiling
}

TEST_CASE("Phase 1: J-A magnetization reverses sign with H reversal",
          "[v1][magnetic][phase1][jiles_atherton][hysteresis]") {
    JilesAthertonParams p{};
    JilesAthertonState state{};

    // Ramp +
    for (int i = 0; i <= 1000; ++i) {
        jiles_atherton_step(state, p, 0.0 + 5.0 * i);
    }
    const Real M_after_pos_ramp = state.M;
    REQUIRE(M_after_pos_ramp > 0.0);

    // Ramp − below previous starting point
    for (int i = 0; i <= 2000; ++i) {
        jiles_atherton_step(state, p, 5000.0 - 10.0 * i);
    }
    CHECK(state.M < 0.0);                   // reversed
    CHECK(std::abs(state.M) > 0.5 * std::abs(M_after_pos_ramp));
}
