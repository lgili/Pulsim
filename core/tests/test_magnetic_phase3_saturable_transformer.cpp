// =============================================================================
// Phase 3 of `add-magnetic-core-models`: SaturableTransformer
// =============================================================================
//
// Pins multi-winding turns-ratio behavior, magnetizing-branch saturation,
// trapezoidal advance of the core flux, and per-winding leakage state.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/magnetic/saturable_transformer.hpp"

using namespace pulsim::v1;
using namespace pulsim::v1::magnetic;
using Catch::Approx;

TEST_CASE("Phase 3: SaturableTransformer enforces ≥ 1 winding",
          "[v1][magnetic][phase3][transformer][validation]") {
    using Trafo = SaturableTransformer<BHCurveArctan>;

    CHECK_THROWS_AS(
        Trafo({.area = 1e-4, .path_length = 5e-2},
              {},
              BHCurveArctan{0.4, 200.0}),
        std::invalid_argument);
}

TEST_CASE("Phase 3: 1:1 transformer reproduces SaturableInductor on the primary",
          "[v1][magnetic][phase3][transformer][parity]") {
    constexpr Real Bs = 0.4;
    constexpr Real Hc = 200.0;
    constexpr Real N  = 50.0;
    constexpr Real Ae = 1e-4;
    constexpr Real le = 5e-2;

    SaturableInductor<BHCurveArctan> ind(
        {.turns = N, .area = Ae, .path_length = le},
        BHCurveArctan{Bs, Hc});

    SaturableTransformer<BHCurveArctan> trafo(
        {.area = Ae, .path_length = le},
        {{.turns = N, .leakage = 0.0},
         {.turns = N, .leakage = 0.0}},   // 1:1, no leakage
        BHCurveArctan{Bs, Hc});

    // At λ_m = some test flux, the magnetizing current of the trafo
    // (looking into the primary) must equal the saturable inductor's
    // current at the same λ.
    const Real lambda_test = 0.3 * Bs * N * Ae;   // mid-saturation
    ind.set_flux(lambda_test);
    trafo.set_core_flux(lambda_test);

    CHECK(trafo.magnetizing_current() ==
          Approx(ind.current()).margin(1e-12));
    CHECK(trafo.magnetizing_inductance() ==
          Approx(ind.differential_inductance()).margin(1e-9));
}

TEST_CASE("Phase 3: turns ratio scales the secondary voltage",
          "[v1][magnetic][phase3][transformer][turns_ratio]") {
    SaturableTransformer<BHCurveArctan> trafo(
        {.area = 1e-4, .path_length = 5e-2},
        {{.turns = 10.0, .leakage = 0.0},
         {.turns = 20.0, .leakage = 0.0}},   // 1:2 step-up
        BHCurveArctan{0.4, 200.0});

    CHECK(trafo.turns_ratio(0) == Approx(1.0).margin(1e-12));
    CHECK(trafo.turns_ratio(1) == Approx(2.0).margin(1e-12));

    // For a given common dλ_m/dt, secondary winding voltage should be
    // 2× the primary's (no leakage, dt=0).
    const Real dlambda_dt = 0.5;
    const Real v_pri = trafo.winding_voltage(0, dlambda_dt, 0.0);
    const Real v_sec = trafo.winding_voltage(1, dlambda_dt, 0.0);
    CHECK(v_sec == Approx(2.0 * v_pri).margin(1e-12));
}

TEST_CASE("Phase 3: leakage inductance contributes its own di/dt term",
          "[v1][magnetic][phase3][transformer][leakage]") {
    SaturableTransformer<BHCurveArctan> trafo(
        {.area = 1e-4, .path_length = 5e-2},
        {{.turns = 50.0, .leakage = 1e-6},
         {.turns = 50.0, .leakage = 5e-6}},
        BHCurveArctan{0.4, 200.0});

    // With no core dλ_m/dt and a pure di_leak/dt on each winding, the
    // voltage equals L_leak[k] · di/dt (independent of turns ratio).
    const Real di_dt = 1000.0;
    CHECK(trafo.winding_voltage(0, 0.0, di_dt) ==
          Approx(1e-6 * di_dt).margin(1e-12));
    CHECK(trafo.winding_voltage(1, 0.0, di_dt) ==
          Approx(5e-6 * di_dt).margin(1e-12));
}

TEST_CASE("Phase 3: trapezoidal advance integrates voltage into core flux",
          "[v1][magnetic][phase3][transformer][advance]") {
    SaturableTransformer<BHCurveArctan> trafo(
        {.area = 1e-4, .path_length = 5e-2},
        {{.turns = 50.0, .leakage = 0.0}},
        BHCurveArctan{0.4, 200.0});

    // Apply 1V (referred to the magnetizing branch) for 1 µs.
    trafo.advance_core_flux_trapezoidal(1.0, 1e-6);
    CHECK(trafo.core_flux() == Approx(1e-6).margin(1e-15));

    // Reverse: returns to ≈ 0.
    trafo.advance_core_flux_trapezoidal(-1.0, 1e-6);
    CHECK(trafo.core_flux() == Approx(0.0).margin(1e-15));
}

TEST_CASE("Phase 3: per-winding leakage state is independently tracked",
          "[v1][magnetic][phase3][transformer][leakage_state]") {
    SaturableTransformer<BHCurveArctan> trafo(
        {.area = 1e-4, .path_length = 5e-2},
        {{.turns = 50.0, .leakage = 1e-6},
         {.turns = 50.0, .leakage = 1e-6},
         {.turns = 50.0, .leakage = 1e-6}},
        BHCurveArctan{0.4, 200.0});

    REQUIRE(trafo.num_windings() == 3);
    trafo.set_leakage_current(0, 1.0);
    trafo.set_leakage_current(1, 2.0);
    trafo.set_leakage_current(2, 3.0);
    CHECK(trafo.leakage_current(0) == 1.0);
    CHECK(trafo.leakage_current(1) == 2.0);
    CHECK(trafo.leakage_current(2) == 3.0);
}
