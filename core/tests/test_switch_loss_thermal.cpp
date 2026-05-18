// MOSFET + IGBT loss / thermal — API surface tests (Phase 2 of
// inverter-bridge-losses).
//
// Verifies that the loss-accumulator + thermal infrastructure is wired
// into the runtime. Numerical-accuracy validation against analytical
// I²·R_on for the MOSFET/IGBT conduction-loss formula is deferred — the
// runtime's simplified stamp (`stamp_mosfet_jacobian` /
// `stamp_igbt_jacobian` in runtime_circuit.hpp) needs a Norton-shifted
// upgrade (analogous to what was done for the diode in Phase 1) before
// the loss numbers match analytical expectations.
//
// What's tested here:
//   * Legacy mode (R_th_ja == 0): the accumulator stays at 0, ensuring
//     pre-Phase-2 behaviour is preserved bit-for-bit.
//   * Thermal mode (R_th_ja > 0): the accumulator runs, exposes
//     finite values, and the derived T_j formula matches
//     T_amb + P_avg · R_th_ja.
//   * set_mosfet_T_j / reset_mosfet_loss work as advertised.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

SimulationOptions make_opts(Real tstop, Real dt) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = tstop;
    opts.dt = dt;
    opts.dt_min = 1e-9;
    opts.dt_max = dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    return opts;
}

}  // namespace

TEST_CASE("MOSFET loss accumulator runs without thermal feedback",
          "[switch_loss][regression]") {
    // Unified pipeline: losses are always tracked from device's own
    // accumulator; `R_th_ja == 0` (default) only disables T_j feedback,
    // so the device dissipates but its junction temperature stays at
    // T_amb. The previous "loss model off by default" semantic is gone —
    // the loss number is now consistent whether or not the user wired
    // a thermal model.
    Circuit circuit;
    const Index gate = circuit.add_node("gate");
    const Index drain = circuit.add_node("drain");
    const Index source = Circuit::ground();

    circuit.add_voltage_source("Vgate", gate, source, 5.0);
    circuit.add_voltage_source("Vdrain", drain, source, 12.0);
    MOSFET::Params mp{};   // default — R_th_ja=0 (no thermal feedback)
    circuit.add_mosfet("M1", gate, drain, source, mp);

    auto opts = make_opts(1e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    Simulator sim(circuit, opts);
    REQUIRE(sim.run_transient().success);

    // V_gs > vth → channel is on, with V_drain = 12 V across `g_on`.
    // Conduction loss is always tracked.
    CHECK(circuit.mosfet_total_energy("M1") > 0.0);
    CHECK(circuit.mosfet_average_power("M1") > 0.0);
    // No thermal feedback (R_th_ja = 0) so T_j stays at T_amb.
    CHECK(circuit.mosfet_junction_temperature("M1") == Approx(25.0));
}

TEST_CASE("IGBT loss accumulator runs without thermal feedback",
          "[switch_loss][regression]") {
    Circuit circuit;
    const Index gate = circuit.add_node("gate");
    const Index coll = circuit.add_node("coll");
    const Index emit = Circuit::ground();

    circuit.add_voltage_source("Vgate", gate, emit, 15.0);
    circuit.add_voltage_source("Vc", coll, emit, 10.0);
    IGBT::Params ip{};         // default — R_th_ja=0 (no thermal feedback)
    circuit.add_igbt("G1", gate, coll, emit, ip);

    auto opts = make_opts(1e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    Simulator sim(circuit, opts);
    REQUIRE(sim.run_transient().success);

    CHECK(circuit.igbt_total_energy("G1") > 0.0);
    CHECK(circuit.igbt_average_power("G1") > 0.0);
    CHECK(circuit.igbt_junction_temperature("G1") == Approx(25.0));
}

TEST_CASE("MOSFET loss accessors return NaN for unknown device name",
          "[switch_loss][api]") {
    Circuit circuit;
    CHECK(std::isnan(circuit.mosfet_average_power("doesnt_exist")));
    CHECK(std::isnan(circuit.mosfet_peak_power("doesnt_exist")));
    CHECK(std::isnan(circuit.mosfet_total_energy("doesnt_exist")));
    CHECK(std::isnan(circuit.mosfet_junction_temperature("doesnt_exist")));
    CHECK(std::isnan(circuit.igbt_average_power("doesnt_exist")));
    CHECK(std::isnan(circuit.igbt_peak_power("doesnt_exist")));
}

TEST_CASE("MOSFET set_T_j / reset_loss are no-ops for unknown name",
          "[switch_loss][api]") {
    // No throw — these accessors silently skip unknown devices.
    Circuit circuit;
    circuit.set_mosfet_T_j("nope", 80.0);
    circuit.reset_mosfet_loss("nope");
    circuit.set_igbt_T_j("nope", 80.0);
    circuit.reset_igbt_loss("nope");
    CHECK(true);   // didn't crash
}

TEST_CASE("MOSFET conduction loss: I²·Rds_on matches analytical",
          "[switch_loss][mosfet][regression]") {
    // With the Phase 2 stamp fix (forced-state path now uses the same
    // Newton convention as the smooth path), the steady-state V_ds
    // should match the analytical KCL solution V_ds = V_src ·
    // Rds_on/(R_load + Rds_on), and the accumulated P_avg should match
    // I²·Rds_on within 10 %.
    Circuit circuit;
    const Index gate = circuit.add_node("gate");
    const Index drain = circuit.add_node("drain");
    const Index n_pos = circuit.add_node("pos");
    const Index source = Circuit::ground();

    MOSFET::Params mp{};
    mp.vth = 2.0;
    mp.kp = 10.0;
    mp.g_on = 100.0;            // R_ds(on) = 10 mΩ
    mp.g_off = 1e-12;
    mp.is_nmos = true;
    mp.Rds_on_tc = 0.0;
    mp.R_th_ja = 1.0;
    mp.T_amb = 25.0;

    circuit.add_voltage_source("Vgate", gate, source, 10.0);
    circuit.add_voltage_source("Vdrain", n_pos, source, 5.0);
    const Real R_load = 0.5;
    circuit.add_resistor("Rload", n_pos, drain, R_load);
    circuit.add_mosfet("M1", gate, drain, source, mp);
    circuit.set_switch_state("M1", true);    // force ON

    auto opts = make_opts(10e-3, 100e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    Simulator sim(circuit, opts);
    REQUIRE(sim.run_transient().success);

    const Real Rds_on = 1.0 / mp.g_on;
    const Real I_expected = 5.0 / (R_load + Rds_on);    // ≈ 9.80 A
    const Real P_expected = I_expected * I_expected * Rds_on;  // ≈ 0.961 W

    const Real p_avg = circuit.mosfet_average_power("M1");
    const Real i_last = circuit.mosfet_last_current("M1");
    const Real v_last = circuit.mosfet_last_voltage("M1");
    INFO("Expected: V_ds=" << I_expected * Rds_on << " V, I=" << I_expected
         << " A, P=" << P_expected << " W. "
         "Got: V_ds=" << v_last << " V, I=" << i_last << " A, P=" << p_avg
         << " W");
    CHECK(p_avg == Approx(P_expected).epsilon(0.10));
    CHECK(p_avg > 0.0);
    CHECK(i_last == Approx(I_expected).epsilon(0.10));

    // T_j derived = T_amb + P_avg · R_th_ja
    const Real t_j_derived = circuit.mosfet_steady_state_junction_temperature("M1");
    CHECK(t_j_derived == Approx(mp.T_amb + p_avg * mp.R_th_ja).epsilon(0.001));
    CHECK(t_j_derived > mp.T_amb);    // self-heating
}

TEST_CASE("MOSFET T_j feedback raises Rds_on (electrothermal coupling)",
          "[switch_loss][mosfet][regression]") {
    // With Rds_on_tc > 0, push T_j higher should increase R_ds(on)
    // → less current → MORE power (P = I²·R(T) where I_drop is small).
    Circuit circuit;
    const Index gate = circuit.add_node("gate");
    const Index drain = circuit.add_node("drain");
    const Index n_pos = circuit.add_node("pos");
    const Index source = Circuit::ground();

    MOSFET::Params mp{};
    mp.vth = 2.0; mp.kp = 10.0; mp.g_on = 100.0; mp.g_off = 1e-12;
    mp.is_nmos = true;
    mp.Rds_on_tc = 5e-3;     // 0.5 %/K
    mp.T_ref = 25.0;
    mp.R_th_ja = 1.0;
    mp.T_amb = 25.0;

    circuit.add_voltage_source("Vgate", gate, source, 10.0);
    circuit.add_voltage_source("Vdrain", n_pos, source, 5.0);
    circuit.add_resistor("Rload", n_pos, drain, 0.5);
    circuit.add_mosfet("M1", gate, drain, source, mp);
    circuit.set_switch_state("M1", true);

    auto opts = make_opts(10e-3, 100e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    // Pass 1: T_j = T_amb = 25 °C
    {
        Simulator sim(circuit, opts);
        REQUIRE(sim.run_transient().success);
    }
    const Real p_25 = circuit.mosfet_average_power("M1");
    REQUIRE(p_25 > 0.0);

    // Pass 2: T_j = 125 °C (R_ds doubles via the 0.5%/K coefficient)
    circuit.set_mosfet_T_j("M1", 125.0);
    circuit.reset_mosfet_loss("M1");
    {
        Simulator sim2(circuit, opts);
        REQUIRE(sim2.run_transient().success);
    }
    const Real p_125 = circuit.mosfet_average_power("M1");
    REQUIRE(p_125 > 0.0);
    INFO("P @ T_j=25 °C: " << p_25 << " W; P @ T_j=125 °C: " << p_125 << " W");
    CHECK(p_125 > p_25);                           // R goes up → P goes up
    CHECK((p_125 - p_25) / p_25 > 0.05);          // > 5 % rise
}

TEST_CASE("MOSFET thermal mode: T_j follows P_avg via R_th_ja",
          "[switch_loss][thermal]") {
    // Pure API/integration test: a MOSFET with R_th_ja > 0 enables the
    // accumulator. Even if V_ds is small (because the runtime stamp's
    // forced-state path may sign-mismatch), the accumulator still
    // integrates V·I·dt and produces a well-defined T_j_derived value.
    Circuit circuit;
    const Index gate = circuit.add_node("gate");
    const Index drain = circuit.add_node("drain");
    const Index source = Circuit::ground();

    MOSFET::Params mp{};
    mp.vth = 2.0; mp.kp = 1.0; mp.g_on = 1000.0; mp.g_off = 1e-12;
    mp.is_nmos = true;
    mp.Rds_on_tc = 0.0;
    mp.R_th_ja = 5.0;
    mp.T_amb = 25.0;

    circuit.add_voltage_source("Vgate", gate, source, 10.0);
    circuit.add_voltage_source("Vdrain", drain, source, 5.0);
    circuit.add_mosfet("M1", gate, drain, source, mp);

    auto opts = make_opts(1e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    Simulator sim(circuit, opts);
    REQUIRE(sim.run_transient().success);

    // T_j_derived = T_amb + P_avg · R_th_ja. Whatever P_avg the
    // accumulator settled on, the derived T_j formula must hold.
    const Real p_avg = circuit.mosfet_average_power("M1");
    const Real t_j_derived =
        circuit.mosfet_steady_state_junction_temperature("M1");
    CHECK(std::isfinite(p_avg));
    CHECK(std::isfinite(t_j_derived));
    CHECK(t_j_derived == Approx(mp.T_amb + p_avg * mp.R_th_ja).epsilon(0.001));
    // Set T_j and reset loss — accessors must respond.
    circuit.set_mosfet_T_j("M1", 100.0);
    CHECK(circuit.mosfet_junction_temperature("M1") == Approx(100.0));
    circuit.reset_mosfet_loss("M1");
    CHECK(circuit.mosfet_average_power("M1") == Approx(0.0));
}
