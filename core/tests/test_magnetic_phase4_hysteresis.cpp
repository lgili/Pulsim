// =============================================================================
// Phase 4 of `add-magnetic-core-models`: hysteresis (Jiles-Atherton)
// =============================================================================
//
// `HysteresisInductor` wraps the J-A model around the geometry/turns
// machinery used by `SaturableInductor`. The contract: tracing a closed
// flux loop produces non-zero loop area in (B, H) — i.e. the device
// actually exhibits hysteresis, with M_irr changing sign across
// reversals.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/magnetic/hysteresis_inductor.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using namespace pulsim::v1::magnetic;
using Catch::Approx;

TEST_CASE("Phase 4: HysteresisInductor traces a non-trivial M_irr loop",
          "[v1][magnetic][phase4][hysteresis]") {
    HysteresisInductor ind(
        {.turns = 50.0, .area = 1e-4, .path_length = 5e-2},
        JilesAthertonParams{.Ms = 1.0e6, .a = 100.0,
                            .alpha = 1.0e-4, .k = 50.0, .c = 0.1});

    // Forward sweep: ramp λ from 0 to a peak.
    constexpr Real lambda_peak = 1e-4;
    const int N = 200;
    Real M_irr_max = 0.0;
    for (int i = 0; i <= N; ++i) {
        const Real lambda = lambda_peak * static_cast<Real>(i) / static_cast<Real>(N);
        ind.apply_flux_step(lambda);
        if (std::abs(ind.magnetization_irreversible()) >
            std::abs(M_irr_max)) {
            M_irr_max = ind.magnetization_irreversible();
        }
    }
    REQUIRE(M_irr_max > 0.0);

    // Reverse sweep: ramp back through zero to a negative peak — the
    // J-A state should swing M_irr through sign.
    Real M_irr_min = 0.0;
    for (int i = 0; i <= 2 * N; ++i) {
        const Real t = static_cast<Real>(i) / static_cast<Real>(N);
        const Real lambda = lambda_peak * (1.0 - t);
        ind.apply_flux_step(lambda);
        if (ind.magnetization_irreversible() < M_irr_min) {
            M_irr_min = ind.magnetization_irreversible();
        }
    }
    CHECK(M_irr_min < 0.0);
    CHECK(std::abs(M_irr_min) > 0.1 * std::abs(M_irr_max));
}

TEST_CASE("Phase 4: HysteresisInductor reset restores unmagnetized state",
          "[v1][magnetic][phase4][hysteresis][reset]") {
    HysteresisInductor ind(
        {.turns = 50.0, .area = 1e-4, .path_length = 5e-2},
        JilesAthertonParams{});

    // Drive to some non-zero state.
    for (int i = 1; i <= 50; ++i) {
        ind.apply_flux_step(1e-5 * i);
    }
    REQUIRE(ind.flux() != 0.0);

    ind.reset();
    CHECK(ind.flux() == Approx(0.0).margin(1e-15));
    CHECK(ind.magnetization() == Approx(0.0).margin(1e-15));
    CHECK(ind.magnetization_irreversible() == Approx(0.0).margin(1e-15));
}

TEST_CASE("Phase 4: current_from_flux is hysteresis-free (linear in λ)",
          "[v1][magnetic][phase4][hysteresis][stateless]") {
    // The stateless `current_from_flux` is the linear "no-hysteresis"
    // viewpoint — used by the linearization / AD layer where path-
    // dependent behavior would break Newton convergence. It must be
    // pure: doubling λ doubles i, regardless of past history.
    HysteresisInductor ind(
        {.turns = 50.0, .area = 1e-4, .path_length = 5e-2},
        JilesAthertonParams{});

    const Real i_at_1 = ind.current_from_flux(1e-5);
    const Real i_at_2 = ind.current_from_flux(2e-5);
    CHECK(i_at_2 == Approx(2.0 * i_at_1).margin(1e-9));

    // Driving the device through some history doesn't change the
    // stateless lookup.
    for (int i = 0; i < 20; ++i) {
        ind.apply_flux_step(1e-4 * std::sin(i * 0.5));
    }
    CHECK(ind.current_from_flux(1e-5) == Approx(i_at_1).margin(1e-9));
}
