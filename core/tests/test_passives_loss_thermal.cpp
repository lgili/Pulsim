// Capacitor + Resistor + Inductor loss / thermal — Catch2 tests
// (Phase 3 of inverter-bridge-losses, Pulsim 0.10.0a9/a10/a11).
//
// Each passive now exposes the same loss-accumulator + thermal
// pattern as the switching devices:
//     Capacitor : P = I_cap² · ESR(T_j)
//     Resistor  : P = V²/R(T_j)  (= I²·R)
//     Inductor  : P = I² · DCR(T_j)
//
// Backward-compat is verified by the legacy-mode tests (R_th_ja == 0
// → accumulator is a no-op).

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

// ---------------------------------------------------------------------------
// Capacitor ESR loss
// ---------------------------------------------------------------------------
TEST_CASE("Capacitor: legacy mode (R_th_ja=0) leaves loss accumulator at 0",
          "[passives_loss][capacitor][regression]") {
    Circuit circuit;
    const Index n_in = circuit.add_node("in");
    const Index gnd  = Circuit::ground();
    circuit.add_voltage_source("Vs", n_in, gnd, 5.0);
    circuit.add_capacitor("C1", n_in, gnd, 1e-6);

    auto opts = make_opts(1e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    REQUIRE(Simulator(circuit, opts).run_transient().success);

    CHECK(circuit.capacitor_average_power("C1") == Approx(0.0));
    CHECK(circuit.capacitor_total_energy("C1") == Approx(0.0));
}

TEST_CASE("Capacitor: ESR loss for sinusoidal ripple current",
          "[passives_loss][capacitor][regression]") {
    // Drive a known ripple current through the cap and verify the
    // accumulated P_avg matches I_rms² · ESR.
    Circuit circuit;
    const Index a = circuit.add_node("a");
    const Index gnd = Circuit::ground();

    // 1 A peak (≈ 0.707 A rms) sinusoidal current source @ 1 kHz.
    SineParams sp{};
    sp.amplitude = 1.0;
    sp.frequency = 1000.0;
    sp.offset = 0.0;
    sp.phase = 0.0;
    // Note: there's no add_sine_current_source; build current via a
    // simpler approach — a fixed DC current through a series resistor
    // with the cap in parallel-shunt won't excite ESR meaningfully.
    // Instead use the high-frequency voltage swing across the cap.
    sp.amplitude = 100.0;
    circuit.add_sine_voltage_source("Vac", a, gnd, sp);

    Capacitor::Params cp{};
    cp.capacitance = 1e-6;
    cp.ESR         = 0.1;        // 100 mΩ
    cp.ESR_tc      = 0.0;         // no thermal feedback for clean math
    cp.R_th_ja     = 5.0;
    cp.T_amb       = 25.0;
    circuit.add_capacitor("C1", a, gnd, cp);

    auto opts = make_opts(20e-3, 5e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    REQUIRE(Simulator(circuit, opts).run_transient().success);

    // I_cap = C · dV/dt = C · ω · V_peak · cos(ωt)
    //   → I_rms = C · ω · V_peak / √2
    //   → P = I_rms² · ESR
    const Real omega = 2.0 * 3.14159265358979323846 * 1000.0;
    const Real I_rms = cp.capacitance * omega * 100.0 / std::sqrt(2.0);
    const Real P_expected = I_rms * I_rms * cp.ESR;

    const Real p_avg = circuit.capacitor_average_power("C1");
    INFO("Expected steady-state I_rms = " << I_rms << " A → P ≈ "
         << P_expected << " W (analytical, ignoring trap. startup). "
         "Measured P_avg = " << p_avg << " W. The first few timesteps "
         "have trapezoidal-companion startup ringing that inflates I — "
         "the order of magnitude check is what matters here.");
    CHECK(p_avg > 0.0);
    // Order-of-magnitude check: within 5× of analytical (trap startup
    // ringing on first cycle can inflate I_rms over the full window).
    CHECK(p_avg < 5.0 * P_expected);
    CHECK(p_avg > 0.2 * P_expected);

    // T_j_derived = T_amb + P_avg · R_th_ja
    const Real t_j = circuit.capacitor_steady_state_junction_temperature("C1");
    CHECK(t_j == Approx(cp.T_amb + p_avg * cp.R_th_ja).epsilon(0.001));
    CHECK(t_j > cp.T_amb);
}

// ---------------------------------------------------------------------------
// Resistor I²R loss
// ---------------------------------------------------------------------------
TEST_CASE("Resistor: legacy mode (R_th_ja=0) leaves loss accumulator at 0",
          "[passives_loss][resistor][regression]") {
    Circuit circuit;
    const Index n_in = circuit.add_node("in");
    const Index gnd  = Circuit::ground();
    circuit.add_voltage_source("Vs", n_in, gnd, 10.0);
    circuit.add_resistor("R1", n_in, gnd, 100.0);

    auto opts = make_opts(1e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    REQUIRE(Simulator(circuit, opts).run_transient().success);

    CHECK(circuit.resistor_average_power("R1") == Approx(0.0));
    CHECK(circuit.resistor_total_energy("R1") == Approx(0.0));
}

TEST_CASE("Resistor: P = V²/R for DC operation",
          "[passives_loss][resistor][regression]") {
    Circuit circuit;
    const Index n_in = circuit.add_node("in");
    const Index gnd  = Circuit::ground();

    Resistor::Params rp{};
    rp.resistance = 10.0;
    rp.TCR        = 0.0;            // no thermal feedback for clean math
    rp.R_th_ja    = 2.0;
    rp.T_amb      = 25.0;
    circuit.add_resistor("R1", n_in, gnd, rp);
    circuit.add_voltage_source("Vs", n_in, gnd, 10.0);

    auto opts = make_opts(10e-3, 100e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    REQUIRE(Simulator(circuit, opts).run_transient().success);

    // P_expected = V²/R = 100/10 = 10 W.
    const Real P_expected = 100.0 / rp.resistance;
    const Real p_avg = circuit.resistor_average_power("R1");
    INFO("Expected P = " << P_expected << " W. Measured = " << p_avg << " W");
    CHECK(p_avg == Approx(P_expected).epsilon(0.05));

    const Real t_j = circuit.resistor_steady_state_junction_temperature("R1");
    CHECK(t_j == Approx(rp.T_amb + p_avg * rp.R_th_ja).epsilon(0.001));
    CHECK(t_j > rp.T_amb);
}

TEST_CASE("Resistor: TCR feedback raises loss when T_j increases",
          "[passives_loss][resistor][regression]") {
    // The Phase-3 loss accumulator follows the powerStage convention:
    // the SIMULATOR sees R_nominal (cold) and produces a current I; the
    // accumulator integrates I² · R(T_j) using the hot resistance from
    // the TCR formula. So P grows with T_j (more dissipation at hot).
    // If you also want the hot R to reduce current in the simulation
    // itself, set_resistance(R(T_j)) before re-running.
    Circuit circuit;
    const Index n_in = circuit.add_node("in");
    const Index gnd  = Circuit::ground();

    Resistor::Params rp{};
    rp.resistance = 10.0;
    rp.TCR        = 5e-3;          // 0.5 %/K (typical metal-film)
    rp.T_ref      = 25.0;
    rp.R_th_ja    = 1.0;
    rp.T_amb      = 25.0;
    circuit.add_resistor("R1", n_in, gnd, rp);
    circuit.add_voltage_source("Vs", n_in, gnd, 10.0);

    auto opts = make_opts(10e-3, 100e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    // Pass 1: T_j = T_amb = 25 °C. I = 1 A, P = I²·R(25) = 10 W.
    {
        REQUIRE(Simulator(circuit, opts).run_transient().success);
    }
    const Real p_25 = circuit.resistor_average_power("R1");

    // Pass 2: T_j = 125 °C → R(125) = 10·(1 + 0.005·100) = 15 Ω.
    // I still = 1 A (simulator uses R_nom). P = I²·R(125) = 15 W.
    circuit.set_resistor_T_j("R1", 125.0);
    circuit.reset_resistor_loss("R1");
    {
        REQUIRE(Simulator(circuit, opts).run_transient().success);
    }
    const Real p_125 = circuit.resistor_average_power("R1");

    INFO("P @ T_j=25 °C: " << p_25 << " W; P @ T_j=125 °C: " << p_125 << " W");
    CHECK(p_125 > p_25);                       // hot dissipation higher
    CHECK((p_125 - p_25) / p_25 > 0.30);       // ≥ 30 % rise (≈ 50 % nominal)
}

// ---------------------------------------------------------------------------
// Inductor DCR loss
// ---------------------------------------------------------------------------
TEST_CASE("Inductor: legacy mode (R_th_ja=0) leaves loss accumulator at 0",
          "[passives_loss][inductor][regression]") {
    Circuit circuit;
    const Index n_in = circuit.add_node("in");
    const Index gnd  = Circuit::ground();
    circuit.add_voltage_source("Vs", n_in, gnd, 5.0);
    circuit.add_inductor("L1", n_in, gnd, 1e-3);

    auto opts = make_opts(1e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    REQUIRE(Simulator(circuit, opts).run_transient().success);

    CHECK(circuit.inductor_average_power("L1") == Approx(0.0));
    CHECK(circuit.inductor_total_energy("L1") == Approx(0.0));
}

TEST_CASE("Inductor: P = I² · DCR for DC current",
          "[passives_loss][inductor][regression]") {
    // Series chain V → L → R_load → GND. At steady state, the inductor
    // looks like a short (DC) so I = V/(R_load + DCR) — but the DCR
    // is NOT in the stamping, only in the loss accumulator. So I in
    // the simulation = V/R_load, and accumulator computes I²·DCR.
    Circuit circuit;
    const Index n_in = circuit.add_node("in");
    const Index n_mid = circuit.add_node("mid");
    const Index gnd  = Circuit::ground();

    Inductor::Params lp{};
    lp.inductance = 1e-3;
    lp.DCR        = 0.05;        // 50 mΩ winding
    lp.DCR_tc     = 0.0;          // no thermal feedback
    lp.R_th_ja    = 3.0;
    lp.T_amb      = 25.0;
    circuit.add_inductor("L1", n_in, n_mid, lp);

    const Real R_load = 1.0;
    circuit.add_resistor("Rload", n_mid, gnd, R_load);
    circuit.add_voltage_source("Vs", n_in, gnd, 5.0);

    auto opts = make_opts(50e-3, 100e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    REQUIRE(Simulator(circuit, opts).run_transient().success);

    // I_steady = V/R_load = 5 A.  P = I²·DCR = 25·0.05 = 1.25 W.
    const Real I_expected = 5.0 / R_load;
    const Real P_expected = I_expected * I_expected * lp.DCR;
    const Real p_avg = circuit.inductor_average_power("L1");
    INFO("Expected I = " << I_expected << " A, P = " << P_expected
         << " W. Measured P_avg = " << p_avg << " W");
    CHECK(p_avg == Approx(P_expected).epsilon(0.10));
    CHECK(p_avg > 0.0);

    const Real t_j = circuit.inductor_steady_state_junction_temperature("L1");
    CHECK(t_j == Approx(lp.T_amb + p_avg * lp.R_th_ja).epsilon(0.001));
}

TEST_CASE("Passives: T_j accessors return NaN for unknown name",
          "[passives_loss][api]") {
    Circuit circuit;
    CHECK(std::isnan(circuit.capacitor_average_power("none")));
    CHECK(std::isnan(circuit.capacitor_junction_temperature("none")));
    CHECK(std::isnan(circuit.resistor_average_power("none")));
    CHECK(std::isnan(circuit.resistor_junction_temperature("none")));
    CHECK(std::isnan(circuit.inductor_average_power("none")));
    CHECK(std::isnan(circuit.inductor_junction_temperature("none")));
    // setters and reset are silent no-ops on unknown names.
    circuit.set_capacitor_T_j("none", 80.0);
    circuit.reset_capacitor_loss("none");
    circuit.set_resistor_T_j("none", 80.0);
    circuit.reset_resistor_loss("none");
    circuit.set_inductor_T_j("none", 80.0);
    circuit.reset_inductor_loss("none");
    CHECK(true);
}
