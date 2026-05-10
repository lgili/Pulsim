// =============================================================================
// Phase 2 of `add-magnetic-core-models`: SaturableInductor
// =============================================================================
//
// Pins the device-mathematics: linear regime matches the analytical
// linear inductor, saturation produces the expected current rise,
// differential inductance collapses past the saturation knee.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/magnetic/saturable_inductor.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using namespace pulsim::v1::magnetic;
using Catch::Approx;

namespace {

constexpr Real Bs = 0.4;       // saturation flux density (T)
constexpr Real Hc = 200.0;     // characteristic field (A/m)
constexpr Real N  = 50.0;      // turns
constexpr Real Ae = 1e-4;      // 1 cm² core cross section
constexpr Real le = 5e-2;      // 5 cm magnetic path

// Approximation of the "linear" inductance in the small-signal regime.
// In the arctan B-H model, dB/dH at H = 0 is 2·Bs / (π · Hc).
// L_d(0) = N²·Ae·(dB/dH) / le.
constexpr Real L_linear_expected =
    (N * N * Ae * 2.0 * Bs) / (le * std::numbers::pi_v<Real> * Hc);

}  // namespace

TEST_CASE("Phase 2: SaturableInductor matches linear inductance at low flux",
          "[v1][magnetic][phase2][saturable_inductor][linear]") {
    SaturableInductor<BHCurveArctan> ind(
        {.turns = N, .area = Ae, .path_length = le},
        BHCurveArctan{Bs, Hc});

    // At λ = 0: i = 0 (origin of the curve).
    CHECK(ind.current_from_flux(0.0) == Approx(0.0).margin(1e-12));

    // Differential inductance at λ = 0 equals the small-signal
    // inductance computed from the curve's slope at H = 0.
    CHECK(ind.differential_inductance(0.0) ==
          Approx(L_linear_expected).margin(1e-3));

    // Round trip: flux_from_current → current_from_flux ≈ identity in
    // the linear region.
    const Real i_test = 0.05;     // small current, well below saturation
    const Real lambda = ind.flux_from_current(i_test);
    CHECK(ind.current_from_flux(lambda) == Approx(i_test).margin(1e-9));
}

TEST_CASE("Phase 2: SaturableInductor's differential inductance collapses at saturation",
          "[v1][magnetic][phase2][saturable_inductor][saturation]") {
    SaturableInductor<BHCurveArctan> ind(
        {.turns = N, .area = Ae, .path_length = le},
        BHCurveArctan{Bs, Hc});

    // At λ corresponding to B = 0.95·Bs (deep saturation):
    const Real B_sat = 0.95 * Bs;
    const Real lambda_sat = B_sat * N * Ae;

    const Real L_sat = ind.differential_inductance(lambda_sat);
    const Real L_linear = ind.differential_inductance(0.0);

    INFO("L_d at λ=0    = " << L_linear);
    INFO("L_d at λ_sat  = " << L_sat);
    INFO("ratio         = " << L_sat / L_linear);

    // At 95% of Bs the arctan-curve slope has dropped by roughly
    // (1 + (H_at_0.95Bs / Hc)²) — for the chosen Bs/Hc this is ≈ 60×
    // smaller. The exact ratio depends on the curve shape; the
    // contract is "much smaller" — we use 10× as the regression floor.
    CHECK(L_sat < L_linear / 10.0);

    // Current at saturation should be much larger than the current at
    // the same fraction of linear extrapolation: i_actual / i_linear
    // = (l_e/N · H_actual) / (l_e/N · B_sat / (slope·μ-equivalent))
    const Real i_sat = ind.current_from_flux(lambda_sat);
    const Real i_linear_extrap = lambda_sat / L_linear;
    INFO("i_actual at λ_sat       = " << i_sat);
    INFO("i_linear extrap at λ_sat = " << i_linear_extrap);
    CHECK(i_sat > i_linear_extrap * 5.0);   // saturation kicks in
}

TEST_CASE("Phase 2: trapezoidal advance integrates v=dλ/dt correctly",
          "[v1][magnetic][phase2][saturable_inductor][advance]") {
    SaturableInductor<BHCurveArctan> ind(
        {.turns = N, .area = Ae, .path_length = le},
        BHCurveArctan{Bs, Hc});

    // Apply a constant 1 V across the inductor for 1 µs. λ should grow
    // by exactly 1 µV·s = 1e-6 V·s.
    const Real v = 1.0;
    const Real dt = 1e-6;
    ind.advance_trapezoidal(v, dt);

    CHECK(ind.flux() == Approx(1e-6).margin(1e-15));

    // Reverse the sign: λ returns to ≈ 0.
    ind.advance_trapezoidal(-v, dt);
    CHECK(ind.flux() == Approx(0.0).margin(1e-15));
}

TEST_CASE("Phase 2: tabulated B-H curve drives a SaturableInductor too",
          "[v1][magnetic][phase2][saturable_inductor][table]") {
    // Soft-ferrite-style 5-point curve.
    BHCurveTable curve(
        {-1500, -200, 0, 200, 1500},
        {-0.40, -0.30, 0.0, 0.30, 0.40});

    SaturableInductor<BHCurveTable> ind(
        {.turns = N, .area = Ae, .path_length = le},
        std::move(curve));

    // Origin behavior: i(0) = 0.
    CHECK(ind.current_from_flux(0.0) == Approx(0.0).margin(1e-12));

    // Pick a flux corresponding to B = 0.20 T (mid-range, linear).
    const Real B_test = 0.20;
    const Real lambda_test = B_test * N * Ae;
    // From the table: between H=0 (B=0) and H=200 (B=0.30) — slope is
    //   dB/dH = 0.30 / 200 = 1.5e-3.
    // So at B=0.20: H = 0 + (0.20 / 1.5e-3) ≈ 133.3 A/m.
    // i = H · l_e / N = 133.3 · 0.05 / 50 ≈ 0.1333 A.
    CHECK(ind.current_from_flux(lambda_test) ==
          Approx(0.1333).epsilon(0.01));
}
