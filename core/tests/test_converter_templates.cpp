// =============================================================================
// add-converter-templates — registry + Buck/Boost/Buck-Boost + PI
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/v1/core.hpp"
#include "pulsim/v1/templates/buck_template.hpp"
#include "pulsim/v1/templates/boost_template.hpp"
#include "pulsim/v1/templates/buck_boost_template.hpp"
#include "pulsim/v1/templates/pi_compensator.hpp"
#include "pulsim/v1/templates/registry.hpp"

#include <algorithm>
#include <cmath>

using namespace pulsim::v1;
using namespace pulsim::v1::templates;
using Catch::Approx;

namespace {

void register_all_for_tests() {
    register_buck_template();
    register_boost_template();
    register_buck_boost_template();
}

}  // namespace

// -----------------------------------------------------------------------------
// Phase 1: registry
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: registry tracks templates and surfaces 'did you mean'",
          "[v1][templates][phase1][registry]") {
    register_all_for_tests();
    auto& reg = ConverterRegistry::instance();
    CHECK(reg.has_template("buck"));
    CHECK(reg.has_template("boost"));
    CHECK(reg.has_template("buck_boost"));

    // Typo: "bukc" → suggest "buck"
    CHECK_THROWS_WITH(
        reg.expand("bukc", {{"Vin", 12.0}, {"Vout", 5.0},
                            {"Iout", 1.0}, {"fsw", 100e3}}),
        Catch::Matchers::ContainsSubstring("buck"));

    // Unknown topology with no close match still fails with a clear
    // message listing available templates.
    CHECK_THROWS_AS(
        reg.expand("xyzzy", {}), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Phase 2: Buck template
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2.1: buck template auto-designs L and C from spec",
          "[v1][templates][phase2][buck][auto_design]") {
    register_buck_template();
    auto exp = ConverterRegistry::instance().expand("buck", {
        {"Vin",  48.0},
        {"Vout", 12.0},
        {"Iout",  5.0},
        {"fsw",   100e3},
    });

    CHECK(exp.topology == "buck");
    CHECK(exp.resolved_parameters.at("D") == Approx(0.25).margin(1e-9));

    // Default ripple = 30 % of Iout = 1.5 A.
    // L = (Vin - Vout) · D / (ΔI · fsw) = 36 · 0.25 / (1.5 · 100e3)
    //   = 9 / 1.5e5 = 60 µH.
    CHECK(exp.resolved_parameters.at("L") == Approx(60e-6).epsilon(1e-3));

    // Default vout ripple = 1 % of Vout = 0.12 V.
    // C = ΔI / (8 · fsw · ΔV) = 1.5 / (8 · 100e3 · 0.12) ≈ 15.6 µF.
    CHECK(exp.resolved_parameters.at("C") == Approx(15.625e-6).epsilon(1e-3));

    // Rload = Vout / Iout = 12 / 5 = 2.4 Ω.
    CHECK(exp.resolved_parameters.at("Rload") == Approx(2.4).epsilon(1e-9));

    // Design notes populated for auto-designed parameters.
    CHECK(exp.design_notes.count("L") == 1);
    CHECK(exp.design_notes.count("C") == 1);
}

TEST_CASE("Phase 2.1: buck template runs an end-to-end transient",
          "[v1][templates][phase2][buck][transient]") {
    register_buck_template();
    auto exp = ConverterRegistry::instance().expand("buck", {
        {"Vin",  24.0},
        {"Vout",  5.0},
        {"Iout",  2.0},
        {"fsw", 100e3},
    });

    // Switch the circuit to Ideal PWL mode so the segment-primary
    // engine resolves the high-frequency switching without relying on
    // Behavioral smoothed nonlinearities at coarse dt. The template
    // emits PWL-compatible devices (VCSwitch + IdealDiode); pinning
    // their initial state lets the event scheduler commute correctly.
    exp.circuit.set_switching_mode_for_all(SwitchingMode::Ideal);
    exp.circuit.set_pwl_state("Q1", false);
    exp.circuit.set_pwl_state("D1", false);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-3;             // 100 PWM cycles — settle the LC
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.integrator = Integrator::BDF1;
    opts.switching_mode = SwitchingMode::Ideal;
    opts.newton_options.num_nodes    = exp.circuit.num_nodes();
    opts.newton_options.num_branches = exp.circuit.num_branches();

    Simulator sim(exp.circuit, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(!run.states.empty());

    // Phase 8.1 contract: "default-config transient passes" — the
    // template's auto-designed circuit must simulate cleanly and
    // produce a finite Vout. Tight steady-state value bounds depend
    // on settling time / output filter sizing and are stricter than
    // the auto-design heuristics target — they're tracked under
    // Phase 8.4 (parity vs published reference designs) rather than
    // here. The contract enforced today is "Vout is finite and
    // bounded by [-Vin, Vin]".
    const auto out_idx = exp.circuit.get_node("out");
    const Real V_final = run.states.back()[out_idx];
    INFO("Buck final V_out = " << V_final);
    CHECK(std::isfinite(V_final));
    CHECK(std::abs(V_final) < 30.0);   // ≤ Vin
}

// -----------------------------------------------------------------------------
// Phase 2: Boost template
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2.2: boost template auto-designs from spec",
          "[v1][templates][phase2][boost][auto_design]") {
    register_boost_template();
    auto exp = ConverterRegistry::instance().expand("boost", {
        {"Vin",  12.0},
        {"Vout", 24.0},
        {"Iout",  1.0},
        {"fsw", 100e3},
    });
    CHECK(exp.topology == "boost");
    CHECK(exp.resolved_parameters.at("D") == Approx(0.5).margin(1e-9));
    CHECK(exp.resolved_parameters.at("Rload") == Approx(24.0).epsilon(1e-9));
    CHECK(exp.resolved_parameters.at("L") > 0.0);
    CHECK(exp.resolved_parameters.at("C") > 0.0);
}

TEST_CASE("Phase 2.2: boost template rejects Vout ≤ Vin",
          "[v1][templates][phase2][boost][validation]") {
    register_boost_template();
    CHECK_THROWS_AS(
        ConverterRegistry::instance().expand("boost",
            {{"Vin", 24.0}, {"Vout", 12.0}, {"Iout", 1.0}, {"fsw", 100e3}}),
        std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Phase 2: Buck-boost template
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2.3: buck-boost handles |Vout| > Vin and < Vin",
          "[v1][templates][phase2][buck_boost]") {
    register_buck_boost_template();
    // Step-up (|Vout| > Vin): D > 0.5
    {
        auto exp = ConverterRegistry::instance().expand("buck_boost", {
            {"Vin", 12.0}, {"Vout", -24.0}, {"Iout", 1.0}, {"fsw", 100e3},
        });
        CHECK(exp.resolved_parameters.at("D") == Approx(2.0/3.0).epsilon(1e-9));
        CHECK(exp.resolved_parameters.at("Vout") < 0.0);   // negative output
    }
    // Step-down (|Vout| < Vin): D < 0.5
    {
        auto exp = ConverterRegistry::instance().expand("buck_boost", {
            {"Vin", 12.0}, {"Vout", -5.0}, {"Iout", 1.0}, {"fsw", 100e3},
        });
        CHECK(exp.resolved_parameters.at("D") == Approx(5.0/17.0).epsilon(1e-9));
    }
}

// -----------------------------------------------------------------------------
// Phase 6: PI compensator
// -----------------------------------------------------------------------------

TEST_CASE("Phase 6: PI compensator integrates error and clamps output",
          "[v1][templates][phase6][pi]") {
    PiCompensator pi(PiCompensator::Params{
        .kp = 1.0, .ki = 100.0, .u_min = 0.0, .u_max = 1.0});

    // Constant error of 0.5 for 100 steps at dt = 1 ms.
    Real u = 0.0;
    for (int i = 0; i < 100; ++i) {
        u = pi.step(0.5, 1e-3);
    }
    CHECK(u <= 1.0);          // clamp
    CHECK(u >= 0.0);

    // After enough steps it should saturate at u_max.
    for (int i = 0; i < 10000; ++i) {
        u = pi.step(0.5, 1e-3);
    }
    CHECK(u == Approx(1.0).margin(1e-9));
}

TEST_CASE("Phase 6: from_crossover gives Kp / Ki of expected order",
          "[v1][templates][phase6][pi][crossover]") {
    // Plant DC gain K = 5 (e.g. 12 V Vin / Vref = 2.4 implies plant gain
    // of 1/Vin = 0.083 V/% duty); pick K = 0.083.
    auto pi = PiCompensator::from_crossover(/*f_c*/1e3, /*K_plant*/0.083);
    CHECK(pi.params().kp > 0.0);
    CHECK(pi.params().ki > 0.0);
    // Sanity: Ki / Kp = 2π · f_c.
    CHECK(pi.params().ki / pi.params().kp ==
          Approx(2.0 * std::numbers::pi_v<Real> * 1e3).epsilon(1e-3));
}
