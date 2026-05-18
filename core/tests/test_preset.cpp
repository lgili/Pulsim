// simplify-and-harden-numerical-surface — Phase 2 unit tests.
//
// Verifies the `Preset` enum + `SimulationOptions::from_preset(...)`
// factory produce the documented profiles. These tests pin the
// per-preset tuning so future drift gets caught by CI rather than
// surprising users in production.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/numerical/preset.hpp"
#include "pulsim/v1/simulation.hpp"

#include <string_view>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Preset: parse_preset_or_auto handles canonical + variant spellings",
          "[preset][parse]") {
    CHECK(parse_preset_or_auto("auto")          == Preset::Auto);
    CHECK(parse_preset_or_auto("Auto")          == Preset::Auto);
    CHECK(parse_preset_or_auto("AUTO")          == Preset::Auto);

    CHECK(parse_preset_or_auto("fast")          == Preset::Fast);
    CHECK(parse_preset_or_auto("Fast")          == Preset::Fast);

    CHECK(parse_preset_or_auto("robust")        == Preset::Robust);
    CHECK(parse_preset_or_auto("ROBUST")        == Preset::Robust);

    CHECK(parse_preset_or_auto("high_fidelity") == Preset::HighFidelity);
    CHECK(parse_preset_or_auto("high-fidelity") == Preset::HighFidelity);
    CHECK(parse_preset_or_auto("highfidelity")  == Preset::HighFidelity);
    CHECK(parse_preset_or_auto("HighFidelity")  == Preset::HighFidelity);

    // Unknown falls back to Auto (per the contract — strict callers
    // should compare against the canonical name themselves).
    CHECK(parse_preset_or_auto("ultra-mega-turbo") == Preset::Auto);
    CHECK(parse_preset_or_auto("")                  == Preset::Auto);
}

TEST_CASE("Preset: to_string round-trip",
          "[preset][parse]") {
    CHECK(to_string(Preset::Auto)         == std::string_view{"Auto"});
    CHECK(to_string(Preset::Fast)         == std::string_view{"Fast"});
    CHECK(to_string(Preset::Robust)       == std::string_view{"Robust"});
    CHECK(to_string(Preset::HighFidelity) == std::string_view{"HighFidelity"});
}

TEST_CASE("Preset::Fast targets pure-switching topologies",
          "[preset][fast]") {
    const auto opts = SimulationOptions::from_preset(Preset::Fast,
                                                      /*dt=*/1e-6,
                                                      /*tstop=*/1e-3);

    CHECK(opts.tstart == Approx(0.0));
    CHECK(opts.tstop  == Approx(1e-3));
    CHECK(opts.dt     == Approx(1e-6));

    CHECK(opts.switching_mode == SwitchingMode::Ideal);
    CHECK(opts.integrator     == Integrator::Trapezoidal);
    CHECK(opts.step_mode      == TransientStepMode::Fixed);
    CHECK(opts.step_mode_explicit);
    CHECK_FALSE(opts.adaptive_timestep);
    CHECK_FALSE(opts.enable_bdf_order_control);
    CHECK_FALSE(opts.stiffness_config.enable);
    CHECK(opts.max_step_retries == 2);
    CHECK(opts.dt_max == Approx(1e-6));  // capped to user dt for fixed-step.
}

TEST_CASE("Preset::Robust targets mixed-domain / motor / nonlinear circuits",
          "[preset][robust]") {
    const auto opts = SimulationOptions::from_preset(Preset::Robust,
                                                      /*dt=*/1e-6,
                                                      /*tstop=*/1e-3);

    CHECK(opts.integrator           == Integrator::TRBDF2);
    CHECK(opts.step_mode            == TransientStepMode::Variable);
    CHECK(opts.step_mode_explicit);
    CHECK(opts.adaptive_timestep);

    CHECK(opts.enable_bdf_order_control);
    CHECK(opts.bdf_config.min_order == 1);
    CHECK(opts.bdf_config.max_order == 2);

    CHECK(opts.stiffness_config.enable);
    CHECK(opts.stiffness_config.switch_integrator);
    CHECK(opts.stiffness_config.stiff_integrator == Integrator::BDF1);

    CHECK(opts.max_step_retries == 12);

    CHECK(opts.fallback_policy.trace_retries);
    CHECK(opts.fallback_policy.enable_transient_gmin);
}

TEST_CASE("Preset::Auto currently aliases Robust",
          "[preset][auto]") {
    const auto a = SimulationOptions::from_preset(Preset::Auto,   1e-6, 1e-3);
    const auto r = SimulationOptions::from_preset(Preset::Robust, 1e-6, 1e-3);
    CHECK(a.integrator           == r.integrator);
    CHECK(a.step_mode            == r.step_mode);
    CHECK(a.stiffness_config.enable == r.stiffness_config.enable);
    CHECK(a.max_step_retries     == r.max_step_retries);
}

TEST_CASE("Preset::HighFidelity tightens tolerances and dt_max",
          "[preset][highfidelity]") {
    const auto robust = SimulationOptions::from_preset(Preset::Robust,
                                                        1e-6, 1e-3);
    const auto hi     = SimulationOptions::from_preset(Preset::HighFidelity,
                                                        1e-6, 1e-3);

    CHECK(hi.integrator == Integrator::TRBDF2);

    // 10× tighter LTE tolerance vs Robust.
    CHECK(hi.timestep_config.error_tolerance ==
          Approx(robust.timestep_config.error_tolerance / 10.0));
    CHECK(hi.lte_config.voltage_tolerance ==
          Approx(robust.lte_config.voltage_tolerance / 10.0));
    CHECK(hi.lte_config.current_tolerance ==
          Approx(robust.lte_config.current_tolerance / 10.0));

    // Stricter step ceiling.
    CHECK(hi.timestep_config.dt_max < robust.timestep_config.dt_max);

    // More retries before declaring failure.
    CHECK(hi.max_step_retries > robust.max_step_retries);
}

TEST_CASE("Preset: explicit override wins over preset defaults",
          "[preset][override]") {
    auto opts = SimulationOptions::from_preset(Preset::Robust, 1e-6, 1e-3);
    // Robust ships TRBDF2; user wants BDF1.
    opts.integrator = Integrator::BDF1;
    CHECK(opts.integrator == Integrator::BDF1);
    // Sanity: the rest of the Robust profile still applies.
    CHECK(opts.stiffness_config.enable);
    CHECK(opts.max_step_retries == 12);
}

TEST_CASE("Preset: raw SimulationOptions{} is unchanged (back-compat)",
          "[preset][backcompat]") {
    SimulationOptions opts;
    // Raw default still has Trapezoidal + Variable (the historical default).
    CHECK(opts.integrator == Integrator::Trapezoidal);
    CHECK(opts.step_mode  == TransientStepMode::Variable);
    CHECK(opts.adaptive_timestep);
    // Raw defaults differ from Robust on the structural fields:
    //   integrator:        Trapezoidal vs TRBDF2
    //   max_step_retries:  6 vs 12
    //   bdf_order_control: false vs true
    // Stiffness happens to default to enabled in both paths.
    CHECK(opts.max_step_retries == 6);
    CHECK_FALSE(opts.enable_bdf_order_control);
}
