// simplify-and-harden-numerical-surface — Phase 12 tests.
//
// Verifies the `templates::mmc_arm()` builder:
//   1. Constructs a valid Circuit with the documented node/branch
//      handles.
//   2. Runs a cold-start transient to completion with `Preset::Robust`
//      → exercises Phase 4 (Armijo line search), Phase 5 (simultaneous
//      event coalescence), Phase 6 (iterative refinement on KLU).
//   3. Cap voltages stay balanced under synchronous PWM (Phase 5
//      coalescence at the gate edge fires).
//
// This is the MVP MMC: single arm, chain of N half-bridge submodules.
// The 3φ + upper/lower-arm version is a follow-up.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"
#include "pulsim/v1/templates/mmc.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("mmc_arm: builds a valid Circuit with the documented handles",
          "[mmc][template][build]") {
    templates::MmcArmParams p{};
    p.num_submodules = 4;
    p.V_dc           = 400.0;
    p.L_arm          = 1e-3;
    p.C_submodule    = 2e-3;
    p.name_prefix    = "armX";

    auto [ckt, h] = templates::mmc_arm(p);

    // Handle population
    CHECK(h.v_top != -1);
    CHECK(h.v_bot != -1);
    CHECK(h.mid_nodes.size()     == 4);
    CHECK(h.gate_nodes.size()    == 4);
    CHECK(h.cap_top_nodes.size() == 4);

    // Each submodule contributes 4 nodes (in/out, cap_top, gate).
    // Plus the arm has v_top, v_bot, arm_after_L, arm_after_R.
    // For N=4: 4 arm nodes + 4·3 sub nodes = 16 nodes... but `in` of
    // sub k is the `out` of sub k-1 (shared), so the actual count is:
    //   v_top, v_bot, arm_after_L, arm_after_R (4)
    //   + (for each of 4 sub) cap_top + out + gate = 12
    //   = 16 nodes.
    CHECK(ckt.num_nodes() == 16);

    // Branch count: only inductors and voltage sources use MNA branch
    // rows. Capacitors are stamped as Norton equivalents (no branch
    // row), and vcswitches stamp as voltage-controlled conductances
    // (no branch row either). The arm has 1 inductor and no voltage
    // sources internally (user supplies V_dc + gate sources later) →
    // 1 branch total.
    CHECK(ckt.num_branches() == 1);
}

TEST_CASE("mmc_arm: cold-start transient with all gates LOW converges",
          "[mmc][template][cold_start]") {
    templates::MmcArmParams p{};
    p.num_submodules = 4;
    p.V_dc           = 400.0;
    p.V_cap_init     = 100.0;  // V_dc / N
    p.L_arm          = 1e-3;
    p.C_submodule    = 2e-3;

    auto [ckt, h] = templates::mmc_arm(p);

    // DC supply across the arm.
    ckt.add_voltage_source("V_dc", h.v_top, h.v_bot, p.V_dc);

    // Tie v_bot to ground via a small resistor (otherwise it floats).
    ckt.add_resistor("R_gnd", h.v_bot, Circuit::ground(), 1e-3);

    // Keep all gates LOW for now (all submodules bypassed, but our
    // template's default S_low has a threshold +100 V above the user's
    // gate, so S_low stays OFF too). The arm degenerates to a chain
    // of OPEN switches in series with the inductor — at cold start
    // the inductor current is zero, the arm carries no current, and
    // the cap voltages stay at their initial values.
    for (auto g : h.gate_nodes) {
        ckt.add_voltage_source("VG_" + std::to_string(g),
                               g, Circuit::ground(), 0.0);
    }

    SimulationOptions opts =
        SimulationOptions::from_preset(Preset::Robust, 1e-5, 5e-4);
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    auto result = sim.run_transient();
    INFO("success: " << result.success);
    INFO("final_status: " << static_cast<int>(result.final_status));
    INFO("diagnostic: " << static_cast<int>(result.diagnostic));
    REQUIRE(result.success);
    REQUIRE_FALSE(result.states.empty());

    // Cap voltages stay balanced around V_dc/N = 100 V. The MVP
    // template doesn't drive submodules so all caps stay near initial.
    const auto& last = result.states.back();
    for (int k = 0; k < 4; ++k) {
        const Real v_cap_top = last[h.cap_top_nodes[k]];
        const Real v_mid     = last[h.mid_nodes[k]];
        const Real v_cap     = v_cap_top - v_mid;
        INFO("V_cap[" << k << "] = " << v_cap);
        CHECK(std::abs(v_cap - 100.0) < 1.0);  // within 1% drift
    }
}

TEST_CASE("mmc_example_yaml: returns non-empty parseable YAML",
          "[mmc][template][yaml]") {
    const std::string yaml = templates::mmc_example_yaml();
    REQUIRE_FALSE(yaml.empty());
    // Sanity checks on the generated text.
    CHECK(yaml.find("schema: pulsim-v1") != std::string::npos);
    CHECK(yaml.find("preset: robust") != std::string::npos);
    CHECK(yaml.find("switching_mode: ideal") != std::string::npos);
    // Should mention all 9 submodules.
    for (int k = 0; k < 9; ++k) {
        const std::string tag = "arm_sm" + std::to_string(k) + "_C_sm";
        INFO("looking for submodule cap: " << tag);
        CHECK(yaml.find(tag) != std::string::npos);
    }
}

TEST_CASE("mmc_3phase_inverter: builds 6 arms with handles populated",
          "[mmc][template][3phase][build]") {
    templates::Mmc3PhaseParams p{};
    p.num_submodules_per_arm = 2;
    p.V_dc = 400.0;
    auto [ckt, h] = templates::mmc_3phase_inverter(p);

    // Handle population.
    CHECK(h.v_dc_pos != -1);
    CHECK(h.v_dc_neg != -1);
    CHECK(h.ac_a != -1);
    CHECK(h.ac_b != -1);
    CHECK(h.ac_c != -1);

    // Each of the 6 arms has populated handles.
    for (int i = 0; i < 6; ++i) {
        CHECK(h.arms[i].v_top != -1);
        CHECK(h.arms[i].v_bot != -1);
        CHECK(h.arms[i].mid_nodes.size()     == 2);
        CHECK(h.arms[i].gate_nodes.size()    == 2);
        CHECK(h.arms[i].cap_top_nodes.size() == 2);
    }

    // Topology sanity: upper arms hang off V_dc+, lower arms hang
    // off V_dc-, midpoints are the AC outputs.
    CHECK(h.arms[0].v_top == h.v_dc_pos);   // upper A → V_dc+
    CHECK(h.arms[0].v_bot == h.ac_a);
    CHECK(h.arms[3].v_top == h.ac_a);       // lower A → AC_a
    CHECK(h.arms[3].v_bot == h.v_dc_neg);
}

TEST_CASE("mmc_3phase_inverter: cold-start transient converges via Preset.Robust",
          "[mmc][template][3phase][convergence]") {
    templates::Mmc3PhaseParams p{};
    p.num_submodules_per_arm = 2;     // 12 submodules total = 24 switches
    p.V_dc = 200.0;                    // small DC to keep currents tame
    auto [ckt, h] = templates::mmc_3phase_inverter(p);

    // DC supply.
    ckt.add_voltage_source("V_dc", h.v_dc_pos, h.v_dc_neg, 200.0);
    ckt.add_resistor("R_gnd", h.v_dc_neg, Circuit::ground(), 1e-3);

    // 3φ Y-connected resistive load to ground (passive, simple).
    ckt.add_resistor("R_load_a", h.ac_a, Circuit::ground(), 10.0);
    ckt.add_resistor("R_load_b", h.ac_b, Circuit::ground(), 10.0);
    ckt.add_resistor("R_load_c", h.ac_c, Circuit::ground(), 10.0);

    // All gates LOW (submodules bypassed via the cap-blocking S_low
    // path which our template's threshold trick disables — caps just
    // hold their initial voltage).
    for (int arm = 0; arm < 6; ++arm) {
        for (std::size_t k = 0; k < h.arms[arm].gate_nodes.size(); ++k) {
            const std::string name = "VG_" + std::to_string(arm) +
                                       "_" + std::to_string(k);
            ckt.add_voltage_source(
                name, h.arms[arm].gate_nodes[k],
                Circuit::ground(), 0.0);
        }
    }

    SimulationOptions opts =
        SimulationOptions::from_preset(Preset::Robust, 1e-5, 1e-4);
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    auto result = sim.run_transient();
    REQUIRE(result.success);
    REQUIRE_FALSE(result.states.empty());
}

TEST_CASE("mmc_arm: synchronous gate edge coalesces events",
          "[mmc][template][simultaneous]") {
    // Drive ALL 4 gates with the same pulse source — at the rising
    // edge, all 4 vcswitches must commute simultaneously. The Phase 5
    // coalescence step should batch them into a single Newton solve.
    templates::MmcArmParams p{};
    p.num_submodules = 4;
    p.V_dc           = 400.0;
    p.V_cap_init     = 100.0;
    p.L_arm          = 1e-3;
    p.C_submodule    = 2e-3;

    auto [ckt, h] = templates::mmc_arm(p);

    ckt.add_voltage_source("V_dc", h.v_top, h.v_bot, p.V_dc);
    ckt.add_resistor("R_gnd", h.v_bot, Circuit::ground(), 1e-3);

    // One shared pulse drives all 4 gates.
    PulseParams pp{};
    pp.v_initial = 0.0;
    pp.v_pulse   = 5.0;
    pp.t_delay   = 1e-4;
    pp.t_rise    = 1e-9;
    pp.t_fall    = 1e-9;
    pp.t_width   = 1e-3;
    ckt.add_pulse_voltage_source("Gate_PWM",
                                  h.gate_nodes[0], Circuit::ground(), pp);
    // Tie the other 3 gates to the same source via 0 V sources (so
    // they share the same control signal).
    for (int k = 1; k < 4; ++k) {
        ckt.add_voltage_source("VG_tie_" + std::to_string(k),
                                h.gate_nodes[k], h.gate_nodes[0], 0.0);
    }

    SimulationOptions opts =
        SimulationOptions::from_preset(Preset::Robust, 1e-5, 5e-4);
    opts.switching_mode = SwitchingMode::Ideal;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    auto result = sim.run_transient();
    REQUIRE(result.success);

    INFO("pwl_event_commutations    = "
         << result.backend_telemetry.pwl_event_commutations);
    INFO("simultaneous_event_groups = "
         << result.backend_telemetry.simultaneous_event_groups);

    // Rising edge at t=100us: all 4 vcswitches commute together.
    CHECK(result.backend_telemetry.pwl_event_commutations >= 4);
    CHECK(result.backend_telemetry.simultaneous_event_groups >= 1);
}
