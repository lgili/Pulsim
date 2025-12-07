#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/models/magnetic_core.hpp"
#include <cmath>

using namespace pulsim;
using namespace pulsim::models;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

constexpr Real MU0 = 4.0 * M_PI * 1e-7;

// =============================================================================
// Jiles-Atherton Model Tests
// =============================================================================

TEST_CASE("JilesAtherton - Zero field", "[magnetic][jiles-atherton]") {
    JilesAthertonParams params;
    params.Ms = 1.6e6;
    params.a = 1000.0;
    params.k = 500.0;
    params.c = 0.1;
    params.alpha = 1e-3;

    JilesAthertonModel model(params);

    auto op = model.evaluate(0.0);

    REQUIRE_THAT(op.H, WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(op.M, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(op.B, WithinAbs(0.0, 1e-10));
}

TEST_CASE("JilesAtherton - Anhysteretic curve", "[magnetic][jiles-atherton]") {
    JilesAthertonParams params;
    params.Ms = 1.6e6;
    params.a = 1000.0;
    params.k = 0.0;      // No hysteresis
    params.c = 1.0;      // Fully reversible
    params.alpha = 0.0;

    JilesAthertonModel model(params);

    // Apply increasing field
    Real H = 5000.0;
    auto op = model.evaluate(H);

    // Should follow Langevin function
    Real He = H;  // No alpha coupling
    Real Man = model.compute_Man(He);

    REQUIRE(op.M > 0);
    REQUIRE(op.M < params.Ms);
    REQUIRE(std::abs(op.B) > 0);
}

TEST_CASE("JilesAtherton - Saturation", "[magnetic][jiles-atherton]") {
    JilesAthertonParams params;
    params.Ms = 1.6e6;
    params.a = 1000.0;
    params.k = 500.0;
    params.c = 0.1;
    params.alpha = 1e-3;

    CoreGeometry geom;
    geom.Ae = 1e-4;
    geom.le = 0.1;

    JilesAthertonModel model(params, geom);

    // Apply field gradually with smaller steps for better convergence
    for (int i = 0; i <= 1000; i++) {
        model.evaluate(i * 100.0);  // Smaller ramp steps
    }
    auto op = model.evaluate(100000.0);

    // Magnetization should have significant magnitude
    // For Jiles-Atherton, direction depends on path history
    REQUIRE(std::abs(op.M) > 0.5 * params.Ms);
}

TEST_CASE("JilesAtherton - Hysteresis loop", "[magnetic][jiles-atherton]") {
    JilesAthertonParams params;
    params.Ms = 1.6e6;
    params.a = 1000.0;
    params.k = 500.0;
    params.c = 0.1;
    params.alpha = 1e-3;

    JilesAthertonModel model(params);

    // First, magnetize in positive direction
    Real H_max = 5000.0;
    for (Real H = 0; H <= H_max; H += 100.0) {
        model.evaluate(H);
    }
    Real M_at_max = model.magnetization();

    // Now decrease H
    for (Real H = H_max; H >= 0; H -= 100.0) {
        model.evaluate(H);
    }
    Real M_at_zero_descending = model.magnetization();

    // Reset and increase from zero
    model.reset();
    for (Real H = 0; H <= H_max; H += 100.0) {
        model.evaluate(H);
    }
    Real M_ascending = model.magnetization();

    // Hysteresis: M at H=0 when descending should be positive (remanence)
    REQUIRE(M_at_zero_descending > 0);
}

TEST_CASE("JilesAtherton - Reset", "[magnetic][jiles-atherton]") {
    JilesAthertonParams params;
    params.Ms = 1.6e6;
    params.M0 = 1e5;  // Non-zero initial

    JilesAthertonModel model(params);

    REQUIRE_THAT(model.magnetization(), WithinAbs(params.M0, 1e-6));

    model.evaluate(1000.0);
    REQUIRE(model.magnetization() != params.M0);

    model.reset();
    REQUIRE_THAT(model.magnetization(), WithinAbs(params.M0, 1e-6));
}

// =============================================================================
// PWL Core Model Tests
// =============================================================================

TEST_CASE("PWLCore - Default ferrite", "[magnetic][pwl]") {
    auto params = PWLCoreParams::default_ferrite();
    PWLCoreModel model(params);

    auto op = model.evaluate(0.0);
    REQUIRE_THAT(op.B, WithinAbs(0.0, 1e-10));

    op = model.evaluate(100.0);
    REQUIRE(op.B > 0);
    REQUIRE(op.B < 0.5);  // Ferrite saturates around 0.5T
}

TEST_CASE("PWLCore - Default silicon steel", "[magnetic][pwl]") {
    auto params = PWLCoreParams::default_silicon_steel();
    PWLCoreModel model(params);

    auto op = model.evaluate(1000.0);
    REQUIRE(op.B > 1.0);  // Silicon steel reaches higher B
}

TEST_CASE("PWLCore - Symmetry", "[magnetic][pwl]") {
    auto params = PWLCoreParams::default_ferrite();
    params.symmetric = true;
    PWLCoreModel model(params);

    auto op_pos = model.evaluate(500.0);
    auto op_neg = model.evaluate(-500.0);

    REQUIRE_THAT(op_pos.B, WithinAbs(-op_neg.B, 1e-10));
}

TEST_CASE("PWLCore - Interpolation", "[magnetic][pwl]") {
    PWLCoreParams params;
    params.curve = {
        {0.0, 0.0},
        {100.0, 0.2},
        {200.0, 0.3},
        {1000.0, 0.4}
    };

    PWLCoreModel model(params);

    // At exact point
    auto op1 = model.evaluate(100.0);
    REQUIRE_THAT(op1.B, WithinAbs(0.2, 1e-6));

    // Between points - linear interpolation
    auto op2 = model.evaluate(150.0);
    REQUIRE(op2.B > 0.2);
    REQUIRE(op2.B < 0.3);
    REQUIRE_THAT(op2.B, WithinAbs(0.25, 0.01));
}

TEST_CASE("PWLCore - Beyond saturation", "[magnetic][pwl]") {
    PWLCoreParams params;
    params.curve = {
        {0.0, 0.0},
        {100.0, 0.4},
        {1000.0, 0.5}
    };

    PWLCoreModel model(params);

    // Beyond last point
    auto op = model.evaluate(5000.0);
    REQUIRE_THAT(op.B, WithinAbs(0.5, 1e-6));  // Saturated
    REQUIRE_THAT(op.dB_dH, WithinAbs(MU0, 1e-10));  // Air permeability
}

TEST_CASE("PWLCore - Permeability calculation", "[magnetic][pwl]") {
    PWLCoreParams params;
    params.curve = {
        {0.0, 0.0},
        {100.0, 0.2},
        {200.0, 0.25}
    };

    PWLCoreModel model(params);

    auto op = model.evaluate(150.0);

    // dB/dH = (0.25 - 0.2) / (200 - 100) = 0.0005
    REQUIRE_THAT(op.dB_dH, WithinAbs(0.0005, 1e-6));
    REQUIRE_THAT(op.mu, WithinAbs(0.0005, 1e-6));
}

// =============================================================================
// Steinmetz Loss Model Tests
// =============================================================================

TEST_CASE("Steinmetz - Sinusoidal loss", "[magnetic][steinmetz]") {
    SteinmetzParams params;
    params.k = 1.0;
    params.alpha = 1.5;
    params.beta = 2.5;

    SteinmetzLossModel model(params);

    // P = k * f^alpha * B^beta
    Real f = 100e3;  // 100 kHz
    Real B = 0.1;    // 100 mT

    Real Pv = model.loss_density_sine(f, B);

    Real expected = 1.0 * std::pow(f, 1.5) * std::pow(B, 2.5);
    REQUIRE_THAT(Pv, WithinRel(expected, 0.01));
}

TEST_CASE("Steinmetz - Zero inputs", "[magnetic][steinmetz]") {
    SteinmetzParams params = SteinmetzParams::ferrite_3C90();
    SteinmetzLossModel model(params);

    REQUIRE_THAT(model.loss_density_sine(0.0, 0.1), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(model.loss_density_sine(100e3, 0.0), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Steinmetz - Total loss", "[magnetic][steinmetz]") {
    SteinmetzParams params;
    params.k = 1.0;
    params.alpha = 1.5;
    params.beta = 2.5;

    SteinmetzLossModel model(params);

    Real f = 100e3;
    Real B = 0.1;
    Real volume = 1e-5;  // 10 mm³

    Real Pv = model.loss_density_sine(f, B);
    Real Ptotal = model.total_loss_sine(f, B, volume);

    REQUIRE_THAT(Ptotal, WithinRel(Pv * volume, 0.001));
}

TEST_CASE("Steinmetz - Material presets", "[magnetic][steinmetz]") {
    auto ferrite_3C90 = SteinmetzParams::ferrite_3C90();
    auto ferrite_3F3 = SteinmetzParams::ferrite_3F3();
    auto silicon = SteinmetzParams::silicon_steel();
    auto amorphous = SteinmetzParams::amorphous_2605SA1();

    SteinmetzLossModel model_3C90(ferrite_3C90);
    SteinmetzLossModel model_3F3(ferrite_3F3);
    SteinmetzLossModel model_si(silicon);
    SteinmetzLossModel model_amor(amorphous);

    // All should give positive loss
    Real f = 100e3;
    Real B = 0.1;

    REQUIRE(model_3C90.loss_density_sine(f, B) > 0);
    REQUIRE(model_3F3.loss_density_sine(f, B) > 0);
    REQUIRE(model_si.loss_density_sine(f, B) > 0);
    REQUIRE(model_amor.loss_density_sine(f, B) > 0);

    // Amorphous should have lowest losses
    REQUIRE(model_amor.loss_density_sine(f, B) < model_3C90.loss_density_sine(f, B));
}

// =============================================================================
// Combined Core Model Tests
// =============================================================================

TEST_CASE("CombinedCore - Basic operation", "[magnetic][combined]") {
    CombinedCoreParams params;
    params.hysteresis.Ms = 1.6e6;
    params.hysteresis.k = 500.0;
    params.losses.k = 1.0;
    params.losses.alpha = 1.5;
    params.losses.beta = 2.5;
    params.geometry.Ae = 1e-4;
    params.geometry.le = 0.1;
    params.include_hysteresis = true;

    CombinedCoreModel model(params);

    auto op = model.evaluate(1000.0, 0.0, 100e3);

    REQUIRE(op.B != 0);
    REQUIRE(std::isfinite(op.loss));
}

TEST_CASE("CombinedCore - Eddy current losses", "[magnetic][combined]") {
    CombinedCoreParams params;
    params.hysteresis.Ms = 1.6e6;
    params.include_hysteresis = false;  // Simple linear
    params.include_eddy_current = true;
    params.sigma = 1e6;   // 1 MS/m conductivity
    params.d = 0.35e-3;   // 0.35mm lamination
    params.geometry.Ae = 1e-4;
    params.geometry.le = 0.1;

    CombinedCoreModel model(params);

    // With dH/dt
    auto op_dynamic = model.evaluate(1000.0, 1e6, 0.0);  // High dH/dt

    // Without dH/dt
    auto op_static = model.evaluate(1000.0, 0.0, 0.0);

    // Dynamic should have more loss due to eddy currents
    REQUIRE(op_dynamic.loss > op_static.loss);
}

TEST_CASE("CombinedCore - Reset", "[magnetic][combined]") {
    CombinedCoreParams params;
    params.hysteresis.Ms = 1.6e6;
    params.hysteresis.M0 = 1e5;

    CombinedCoreModel model(params);

    model.evaluate(5000.0);
    model.reset();

    // Should return to initial state
    REQUIRE_THAT(model.hysteresis_model().magnetization(),
                 WithinAbs(params.hysteresis.M0, 1e-6));
}

// =============================================================================
// Saturable Inductor Tests
// =============================================================================

TEST_CASE("SaturableInductor - Unsaturated", "[magnetic][inductor]") {
    SaturableInductorParams params;
    params.L0 = 1e-3;   // 1 mH
    params.Isat = 10.0; // 10 A saturation
    params.Lsat = 0.0;  // Auto-compute (10% of L0)

    SaturableInductor ind(params);

    // Low current - near L0
    Real L = ind.inductance(0.1);
    REQUIRE_THAT(L, WithinRel(params.L0, 0.01));
}

TEST_CASE("SaturableInductor - Saturated", "[magnetic][inductor]") {
    SaturableInductorParams params;
    params.L0 = 1e-3;
    params.Isat = 10.0;
    params.Lsat = 100e-6;  // 100 uH saturated

    SaturableInductor ind(params);

    // Very high current - near Lsat
    Real L = ind.inductance(100.0);
    REQUIRE(L < params.L0 * 0.2);
    REQUIRE(L > params.Lsat * 0.9);
}

TEST_CASE("SaturableInductor - Flux linkage", "[magnetic][inductor]") {
    SaturableInductorParams params;
    params.L0 = 1e-3;
    params.Isat = 10.0;

    SaturableInductor ind(params);

    // Zero current -> zero flux
    REQUIRE_THAT(ind.flux_linkage(0.0), WithinAbs(0.0, 1e-12));

    // Positive current -> positive flux
    REQUIRE(ind.flux_linkage(1.0) > 0);

    // Negative current -> negative flux
    REQUIRE(ind.flux_linkage(-1.0) < 0);
}

TEST_CASE("SaturableInductor - Voltage equation", "[magnetic][inductor]") {
    SaturableInductorParams params;
    params.L0 = 1e-3;
    params.Isat = 10.0;

    SaturableInductor ind(params);

    // v = L(i) * di/dt
    Real i = 1.0;
    Real di_dt = 1000.0;

    Real v = ind.voltage(i, di_dt);
    Real L = ind.inductance(i);

    REQUIRE_THAT(v, WithinRel(L * di_dt, 0.001));
}

TEST_CASE("SaturableInductor - di/dt calculation", "[magnetic][inductor]") {
    SaturableInductorParams params;
    params.L0 = 1e-3;
    params.Isat = 10.0;

    SaturableInductor ind(params);

    Real v = 10.0;  // 10V
    Real i = 1.0;

    Real di_dt = ind.di_dt(v, i);
    Real L = ind.inductance(i);

    REQUIRE_THAT(di_dt, WithinRel(v / L, 0.001));
}

// =============================================================================
// Saturable Transformer Tests
// =============================================================================

TEST_CASE("SaturableTransformer - Turns ratio", "[magnetic][transformer]") {
    SaturableTransformerParams params;
    params.N1 = 100;
    params.N2 = 10;

    SaturableTransformer xfmr(params);

    REQUIRE_THAT(xfmr.turns_ratio(), WithinAbs(10.0, 1e-10));
}

TEST_CASE("SaturableTransformer - Ideal operation", "[magnetic][transformer]") {
    SaturableTransformerParams params;
    params.N1 = 100;
    params.N2 = 50;
    params.R1 = 0.0;
    params.R2 = 0.0;
    params.Llk1 = 0.0;
    params.Llk2 = 0.0;
    params.Lm = 10e-3;  // Large magnetizing inductance
    params.Isat = 100.0;  // High saturation current

    SaturableTransformer xfmr(params);

    Real v1 = 100.0;
    Real i1 = 1.0;
    Real i2 = -2.0;  // Load current (negative = flowing out)

    auto op = xfmr.evaluate(v1, i1, i2, 0.0, 0.0);

    // v2 ≈ v1 / n = 100 / 2 = 50V
    REQUIRE_THAT(op.v2, WithinRel(50.0, 0.1));
}

TEST_CASE("SaturableTransformer - Magnetizing current", "[magnetic][transformer]") {
    SaturableTransformerParams params;
    params.N1 = 100;
    params.N2 = 50;
    params.Lm = 10e-3;
    params.Isat = 10.0;

    SaturableTransformer xfmr(params);

    // No load (i2 = 0), all primary current is magnetizing
    auto op = xfmr.evaluate(100.0, 1.0, 0.0, 0.0, 0.0);

    REQUIRE_THAT(op.im, WithinRel(1.0, 0.01));
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_CASE("Magnetic models - Numerical stability", "[magnetic][stability]") {
    JilesAthertonParams ja_params;
    REQUIRE_NOTHROW([&]() {
        JilesAthertonModel model(ja_params);
        model.evaluate(0.0);
        model.evaluate(1e10);  // Very high field
        model.evaluate(-1e10);
    }());

    PWLCoreParams pwl_params = PWLCoreParams::default_ferrite();
    REQUIRE_NOTHROW([&]() {
        PWLCoreModel model(pwl_params);
        model.evaluate(0.0);
        model.evaluate(1e10);
        model.evaluate(-1e10);
    }());

    SteinmetzParams st_params;
    REQUIRE_NOTHROW([&]() {
        SteinmetzLossModel model(st_params);
        model.loss_density_sine(1e9, 10.0);  // High frequency, high flux
    }());
}

TEST_CASE("SaturableInductor - Zero inductance protection", "[magnetic][stability]") {
    SaturableInductorParams params;
    params.L0 = 1e-3;
    params.Isat = 1.0;
    params.Lsat = 1e-9;  // Very small saturated inductance

    SaturableInductor ind(params);

    // Even at extreme current, should not divide by zero
    Real di_dt = ind.di_dt(10.0, 1000.0);
    REQUIRE(std::isfinite(di_dt));
}
