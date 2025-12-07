#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/models/mosfet_models.hpp"
#include <cmath>

using namespace pulsim;
using namespace pulsim::models;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// =============================================================================
// Level 1 MOSFET (Shichman-Hodges) Tests
// =============================================================================

TEST_CASE("MOSFETLevel1 - Cutoff region", "[mosfet][level1]") {
    MOSFETModelParams params;
    params.vth0 = 0.7;
    params.kp = 110e-6;
    params.lambda = 0.0;

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Vgs < Vth -> cutoff
    auto op = mos.evaluate(0.3, 1.0, 0.0);

    REQUIRE(op.region == MOSRegion::CUTOFF);
    REQUIRE_THAT(op.ids, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(op.gm, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(op.gds, WithinAbs(0.0, 1e-15));
}

TEST_CASE("MOSFETLevel1 - Linear region", "[mosfet][level1]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.lambda = 0.0;

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Vgs = 1.5V, Vds = 0.2V, Vth = 0.5V
    // Vov = 1.0V, Vds < Vov -> linear
    auto op = mos.evaluate(1.5, 0.2, 0.0);

    REQUIRE(op.region == MOSRegion::LINEAR);

    // Ids = Kp * W/L * ((Vgs - Vth) * Vds - Vds²/2)
    // Ids = 100e-6 * 10 * (1.0 * 0.2 - 0.02) = 1e-3 * 0.18 = 180uA
    Real expected_ids = 100e-6 * 10.0 * (1.0 * 0.2 - 0.2 * 0.2 / 2.0);
    REQUIRE_THAT(op.ids, WithinRel(expected_ids, 0.01));
    REQUIRE(op.gm > 0);
    REQUIRE(op.gds > 0);
}

TEST_CASE("MOSFETLevel1 - Saturation region", "[mosfet][level1]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.lambda = 0.02;  // Channel length modulation

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Vgs = 1.5V, Vds = 2.0V, Vth = 0.5V
    // Vov = 1.0V, Vds > Vov -> saturation
    auto op = mos.evaluate(1.5, 2.0, 0.0);

    REQUIRE(op.region == MOSRegion::SATURATION);

    // Ids = (Kp/2) * W/L * (Vgs - Vth)² * (1 + lambda * Vds)
    Real beta = 100e-6 * 10.0;
    Real vov = 1.0;
    Real expected_ids = (beta / 2.0) * vov * vov * (1.0 + 0.02 * 2.0);
    REQUIRE_THAT(op.ids, WithinRel(expected_ids, 0.01));

    // gm = Kp * W/L * (Vgs - Vth) * (1 + lambda * Vds)
    Real expected_gm = beta * vov * (1.0 + 0.02 * 2.0);
    REQUIRE_THAT(op.gm, WithinRel(expected_gm, 0.05));

    // gds = lambda * Ids_sat
    REQUIRE(op.gds > 0);
}

TEST_CASE("MOSFETLevel1 - Body effect", "[mosfet][level1]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.gamma = 0.4;  // Body effect coefficient
    params.phi = 0.6;    // Surface potential

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Without body effect (Vbs = 0)
    auto op1 = mos.evaluate(1.5, 2.0, 0.0);

    // With body effect (Vbs = -1V, reverse bias)
    auto op2 = mos.evaluate(1.5, 2.0, -1.0);

    // Vth should increase with reverse body bias
    REQUIRE(op2.vth > op1.vth);
    // Current should decrease
    REQUIRE(op2.ids < op1.ids);
}

TEST_CASE("MOSFETLevel1 - PMOS", "[mosfet][level1]") {
    MOSFETModelParams params;
    params.is_pmos = true;
    params.vth0 = -0.5;
    params.kp = 50e-6;  // Lower mobility for PMOS

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 20e-6;  // Wider for same current
    inst.l = 1e-6;
    mos.set_instance(inst);

    // PMOS: Vgs = -1.5V, Vds = -2.0V
    auto op = mos.evaluate(-1.5, -2.0, 0.0);

    REQUIRE(op.region == MOSRegion::SATURATION);
    REQUIRE(op.ids < 0);  // Current flows from source to drain
}

TEST_CASE("MOSFETLevel1 - Temperature dependence", "[mosfet][level1]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // At room temperature
    auto op_300K = mos.evaluate(1.5, 2.0, 0.0, 300.0);

    // At higher temperature
    auto op_400K = mos.evaluate(1.5, 2.0, 0.0, 400.0);

    // Vth typically decreases with temperature (~-2mV/K)
    REQUIRE(op_400K.vth < op_300K.vth);
    // Mobility decreases with temperature -> current decreases
    REQUIRE(op_400K.ids < op_300K.ids);
}

// =============================================================================
// Level 2 MOSFET (Grove-Frohman) Tests
// =============================================================================

TEST_CASE("MOSFETLevel2 - Velocity saturation", "[mosfet][level2]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.vmax = 1e5;  // Enable velocity saturation

    MOSFETLevel2 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 0.5e-6;  // Short channel
    mos.set_instance(inst);

    auto op = mos.evaluate(2.0, 3.0, 0.0);

    REQUIRE(op.region == MOSRegion::SATURATION);
    // With velocity saturation, current is limited
    REQUIRE(op.ids > 0);
    REQUIRE(op.vdsat > 0);
    REQUIRE(op.vdsat < (2.0 - 0.5));  // Vdsat < Vov due to velocity sat
}

TEST_CASE("MOSFETLevel2 - Mobility degradation", "[mosfet][level2]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.ucrit = 1e4;  // Critical field for mobility degradation

    MOSFETLevel2 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Low Vgs - less degradation
    auto op_low = mos.evaluate(1.0, 1.0, 0.0);

    // High Vgs - more degradation
    auto op_high = mos.evaluate(3.0, 1.0, 0.0);

    // Current should not scale quadratically due to mobility degradation
    Real ratio = op_high.ids / op_low.ids;
    Real ideal_ratio = std::pow((3.0 - 0.5) / (1.0 - 0.5), 2);
    REQUIRE(ratio < ideal_ratio);
}

// =============================================================================
// Level 3 MOSFET Tests
// =============================================================================

TEST_CASE("MOSFETLevel3 - Short channel effects", "[mosfet][level3]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.eta = 0.1;  // DIBL coefficient

    MOSFETLevel3 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 0.25e-6;  // Very short channel
    mos.set_instance(inst);

    // Low Vds
    auto op_low_vds = mos.evaluate(1.0, 0.5, 0.0);

    // High Vds - DIBL effect
    auto op_high_vds = mos.evaluate(1.0, 2.0, 0.0);

    // DIBL: Vth decreases with higher Vds
    REQUIRE(op_high_vds.vth < op_low_vds.vth);
}

// =============================================================================
// BSIM3v3 Tests
// =============================================================================

TEST_CASE("BSIM3 - Basic operation", "[mosfet][bsim3]") {
    MOSFETModelParams params;
    params.vth0 = 0.4;
    params.k1 = 0.5;
    params.k2 = 0.0;
    params.vsat = 1.5e5;
    params.u0 = 400.0;      // Mobility for 180nm
    params.tox = 4e-9;      // Gate oxide thickness
    params.phi = 0.8;       // Surface potential

    MOSFETBSIM3 mos(params);
    MOSFETInstance inst;
    inst.w = 1e-6;
    inst.l = 0.18e-6;  // 180nm technology
    mos.set_instance(inst);

    auto op = mos.evaluate(1.0, 1.0, 0.0);

    REQUIRE(op.ids > 0);
    REQUIRE(op.gm > 0);
    REQUIRE(op.gds > 0);
}

TEST_CASE("BSIM3 - Subthreshold conduction", "[mosfet][bsim3]") {
    MOSFETModelParams params;
    params.vth0 = 0.7;      // Threshold voltage
    params.voff = -0.1;
    params.nfactor = 1.0;   // Subthreshold swing factor
    params.k1 = 0.0;        // Disable body effect for simpler calculation
    params.u0 = 400.0;      // Mobility
    params.tox = 4e-9;      // Gate oxide thickness
    params.phi = 0.8;       // Surface potential
    params.pdiblc1 = 0.0;   // Disable DIBL for predictable threshold
    params.pdiblc2 = 0.0;   // Disable DIBL
    params.dvt0 = 0.0;      // Disable short channel effect
    params.nlx = 0.0;       // Disable lateral non-uniform doping

    MOSFETBSIM3 mos(params);
    MOSFETInstance inst;
    inst.w = 1e-6;
    inst.l = 0.18e-6;
    mos.set_instance(inst);

    // Well below threshold (vgs = 0.3, vth ~ 0.7)
    auto op = mos.evaluate(0.3, 1.0, 0.0);

    // Should be in subthreshold region
    REQUIRE(op.region == MOSRegion::SUBTHRESHOLD);
    // Should have some subthreshold current
    REQUIRE(op.ids > 0);
    REQUIRE(op.ids < 1e-2);  // Much smaller than strong inversion
}

// =============================================================================
// EKV Model Tests
// =============================================================================

TEST_CASE("MOSFETEKV - Weak inversion", "[mosfet][ekv]") {
    EKVParams params;
    params.vto = 0.5;
    params.kp = 50e-6;
    params.gamma = 0.5;

    MOSFETEKV mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Well below threshold - weak inversion
    auto op = mos.evaluate(0.2, 1.0, 0.0);

    REQUIRE(op.ids > 0);
    REQUIRE(op.ids < 1e-9);  // Very small current
}

TEST_CASE("MOSFETEKV - Strong inversion", "[mosfet][ekv]") {
    EKVParams params;
    params.vto = 0.5;
    params.kp = 50e-6;

    MOSFETEKV mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Well above threshold - strong inversion
    auto op = mos.evaluate(2.0, 2.0, 0.0);

    REQUIRE(op.ids > 1e-6);  // Significant current
    REQUIRE(op.gm > 0);      // Transconductance must be positive
}

TEST_CASE("MOSFETEKV - Continuous transition", "[mosfet][ekv]") {
    EKVParams params;
    params.vto = 0.5;
    params.kp = 50e-6;

    MOSFETEKV mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Sweep Vgs and check continuity
    Real prev_ids = 0;
    bool first = true;

    for (Real vgs = 0.0; vgs <= 2.0; vgs += 0.05) {
        auto op = mos.evaluate(vgs, 1.0, 0.0);

        if (!first) {
            // Current should increase monotonically
            REQUIRE(op.ids >= prev_ids);
            // Check for reasonable continuity (no jumps)
            Real delta_ids = op.ids - prev_ids;
            Real max_delta = (prev_ids + 1e-12) * 1.0;  // Allow 100% change per step
            REQUIRE(delta_ids < max_delta + 1e-7);  // Relaxed tolerance for numerical precision
        }

        prev_ids = op.ids;
        first = false;
    }
}

// =============================================================================
// Model Factory Tests
// =============================================================================

TEST_CASE("MOSFET Model Factory", "[mosfet][factory]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;

    auto level1 = create_mosfet_model(MOSFETModelLevel::LEVEL1, params);
    REQUIRE(level1 != nullptr);
    REQUIRE(level1->level() == MOSFETModelLevel::LEVEL1);

    auto level2 = create_mosfet_model(MOSFETModelLevel::LEVEL2, params);
    REQUIRE(level2 != nullptr);
    REQUIRE(level2->level() == MOSFETModelLevel::LEVEL2);

    auto level3 = create_mosfet_model(MOSFETModelLevel::LEVEL3, params);
    REQUIRE(level3 != nullptr);
    REQUIRE(level3->level() == MOSFETModelLevel::LEVEL3);

    auto bsim3 = create_mosfet_model(MOSFETModelLevel::BSIM3, params);
    REQUIRE(bsim3 != nullptr);
    REQUIRE(bsim3->level() == MOSFETModelLevel::BSIM3);

    auto ekv = create_mosfet_model(MOSFETModelLevel::EKV, params);
    REQUIRE(ekv != nullptr);
    REQUIRE(ekv->level() == MOSFETModelLevel::EKV);
}

// =============================================================================
// Capacitance Tests
// =============================================================================

TEST_CASE("MOSFET Gate Capacitances", "[mosfet][capacitance]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;
    params.tox = 5e-9;  // 5nm oxide
    params.cgso = 1e-10;  // Overlap capacitance
    params.cgdo = 1e-10;

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // In cutoff
    auto op_cutoff = mos.evaluate(0.2, 1.0, 0.0);
    REQUIRE(op_cutoff.cgs > 0);
    REQUIRE(op_cutoff.cgd > 0);
    REQUIRE(op_cutoff.cgb > 0);  // Gate-bulk in cutoff

    // In saturation
    auto op_sat = mos.evaluate(1.5, 2.0, 0.0);
    REQUIRE(op_sat.cgs > op_cutoff.cgs);  // Channel formed
    REQUIRE(op_sat.cgb < op_cutoff.cgb);  // Channel shields gate from bulk
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_CASE("MOSFET Numerical stability - extreme voltages", "[mosfet][stability]") {
    MOSFETModelParams params;
    params.vth0 = 0.5;
    params.kp = 100e-6;

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 10e-6;
    inst.l = 1e-6;
    mos.set_instance(inst);

    // Very high Vgs
    REQUIRE_NOTHROW(mos.evaluate(100.0, 100.0, 0.0));

    // Very negative Vgs
    REQUIRE_NOTHROW(mos.evaluate(-100.0, 1.0, 0.0));

    // Zero voltages
    auto op_zero = mos.evaluate(0.0, 0.0, 0.0);
    REQUIRE(std::isfinite(op_zero.ids));
    REQUIRE(std::isfinite(op_zero.gm));
    REQUIRE(std::isfinite(op_zero.gds));
}

TEST_CASE("MOSFET Numerical stability - small geometry", "[mosfet][stability]") {
    MOSFETModelParams params;
    params.vth0 = 0.3;
    params.kp = 100e-6;

    MOSFETLevel1 mos(params);
    MOSFETInstance inst;
    inst.w = 100e-9;   // 100nm width
    inst.l = 20e-9;    // 20nm length
    mos.set_instance(inst);

    auto op = mos.evaluate(1.0, 0.5, 0.0);

    REQUIRE(std::isfinite(op.ids));
    REQUIRE(op.ids > 0);
}
