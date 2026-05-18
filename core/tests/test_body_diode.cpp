// =============================================================================
// Phase B1 + B2 of harden-component-models-vs-psim-plecs:
//   * MOSFET body diode (anode = source, cathode = drain)
//   * IGBT antiparallel "freewheel" diode (anode = emitter, cathode = collector)
// =============================================================================
//
// Both features are OFF by default (`body_diode_enable = false` /
// `antiparallel_diode_enable = false`) so existing tests stay green. When
// the user opts in, the device stamp adds a Norton-shifted smooth-blend
// diode current to the channel current, which clamps the drain/collector
// node when reverse-biased by the load — exactly the synchronous-
// rectification dead-time / inverter freewheel behaviour PSIM and PLECS
// produce out of the box.
//
// Topology under test:
//
//      V_high ──[ Vsource ]── n_top ──[ R_pull = 1 Ω ]── n_drain ───┬─── GND
//                                                                    │
//                                                                  device DS/CE
//                                                                    │
//                                                                  GND (forced)
//
// Without the body diode, the MOSFET is in cutoff (V_gs = 0), so the only
// path between n_top and GND is the resistor + g_off ~ 1e-12 — V_drain
// settles at ≈ V_high.
//
// With the body diode enabled, n_drain (= cathode) is held BELOW
// n_source (= GND), i.e. V_sd = -V_drain > 0, which forward-biases the
// diode. Newton clamps V_drain at GND + V_F0 = V_F0 above ground.
//
// Wait — re-reading: anode = source = GND (0 V); cathode = drain. For
// forward bias we need V_anode > V_cathode + V_F0, i.e., 0 > V_drain + V_F0,
// i.e., V_drain < −V_F0. So we drive n_drain negative via a current sink
// (or via a pull-DOWN to a negative rail). Simpler: swap the diode
// direction by reversing the device pins, OR test with a negative rail.
//
// Easier setup: build a synchronous-buck dead-time vignette where the
// MOSFET source sits ABOVE the drain. We just connect:
//     V_source pin → connected to V_high voltage source
//     V_drain pin  → connected through R_pull to GND
// With gate floating (V_gs ≈ 0), the channel is OFF. With body diode
// disabled, V_drain settles around 0 (current path: V_high → MOSFET DS
// (with g_off ≈ 1e-12) → R_pull → GND, giving V_drain ≈ V_high · R_pull / (1/g_off + R_pull) ≈ 1 µV).
// With body diode enabled, n_source > n_drain → forward bias → V_drain
// clamps at V_high − V_F0.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

SimulationOptions make_dc_opts(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-6;
    opts.dt = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("MOSFET body diode: anode-side bias clamps V_drain at V_source - V_F0",
          "[v1][mosfet][body_diode][b1][regression]") {
    Circuit circuit;
    const Index n_high  = circuit.add_node("high");
    const Index n_drain = circuit.add_node("drain");
    const Index n_gate  = circuit.add_node("gate");
    const Index gnd     = Circuit::ground();

    constexpr Real V_high = 10.0;
    circuit.add_voltage_source("Vhigh", n_high, gnd, V_high);
    // Pull-down on drain so the body diode is the only current path
    // when the channel is OFF.
    circuit.add_resistor("Rload", n_drain, gnd, 100.0);
    // Pull gate to GND so V_gs = 0, channel firmly OFF.
    circuit.add_resistor("Rg", n_gate, gnd, 1.0e3);

    MOSFET::Params mp{};
    mp.vth = 2.0;
    mp.kp = 0.02;
    mp.lambda = 0.0;
    mp.g_off = 1e-12;
    mp.is_nmos = true;
    mp.body_diode_enable = true;
    mp.body_diode_V_F0 = 0.8;
    mp.body_diode_R_d = 25e-3;
    mp.body_diode_g_off = 1e-9;
    // Source pin is at V_high (anode), drain pin is at 0 V (cathode-side
    // pull-down via Rload). V_sd = V_high - V_drain ≈ V_high → forward.
    circuit.add_mosfet("M", n_gate, n_drain, n_high, mp);

    auto opts = make_dc_opts(circuit);
    Simulator sim(circuit, opts);
    const auto dc = sim.dc_operating_point();

    INFO("DC OP message: " << dc.message);
    REQUIRE(dc.success);

    const auto& x = dc.newton_result.solution;
    INFO("V(high)  = " << x[n_high]  << " V");
    INFO("V(drain) = " << x[n_drain] << " V");
    INFO("V(gate)  = " << x[n_gate]  << " V");

    // Forward-conducting body diode pins V_drain at V_source - V_F0
    // (with small drop across the 25 mΩ slope at the steady-state
    // current). Newton tolerance allows ~0.5 V slack.
    CHECK(x[n_drain] == Approx(V_high - mp.body_diode_V_F0).margin(0.5));
}

TEST_CASE("MOSFET body diode default state is OFF (backward-compat)",
          "[v1][mosfet][body_diode][b1]") {
    // Pure API-level invariant: the body diode is opt-in, default OFF,
    // so existing MOSFET tests that don't model the diode keep their
    // legacy behaviour. This test catches accidental default flips.
    MOSFET::Params mp{};
    CHECK_FALSE(mp.body_diode_enable);
    CHECK(mp.body_diode_V_F0   == Approx(0.8));
    CHECK(mp.body_diode_R_d    == Approx(25e-3));
    CHECK(mp.body_diode_g_off  == Approx(1e-9));
}

TEST_CASE("IGBT antiparallel diode: emitter-side bias clamps V_collector "
          "at V_emitter - V_F0",
          "[v1][igbt][antiparallel_diode][b2][regression]") {
    Circuit circuit;
    const Index n_high = circuit.add_node("high");
    const Index n_coll = circuit.add_node("coll");
    const Index n_gate = circuit.add_node("gate");
    const Index gnd    = Circuit::ground();

    constexpr Real V_high = 50.0;
    circuit.add_voltage_source("Vhigh", n_high, gnd, V_high);
    circuit.add_resistor("Rload", n_coll, gnd, 100.0);
    circuit.add_resistor("Rg", n_gate, gnd, 1.0e3);

    IGBT::Params ip{};
    ip.vth = 5.0;
    ip.g_on = 1e4;
    ip.g_off = 1e-12;
    ip.v_ce_sat = 1.5;
    ip.antiparallel_diode_enable = true;
    ip.antiparallel_diode_V_F0 = 1.0;
    ip.antiparallel_diode_R_d = 20e-3;
    ip.antiparallel_diode_g_off = 1e-9;
    // Emitter pin (anode) at V_high; collector pin (cathode) → pulled
    // toward 0 by Rload. V_ec = V_high − V_coll > 0 → forward-bias.
    circuit.add_igbt("Q", n_gate, n_coll, n_high, ip);

    auto opts = make_dc_opts(circuit);
    Simulator sim(circuit, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);

    const auto& x = dc.newton_result.solution;
    INFO("V(coll) = " << x[n_coll] << " V");

    // Forward-conducting freewheel diode pins V_coll at V_high - V_F0.
    CHECK(x[n_coll] == Approx(V_high - ip.antiparallel_diode_V_F0).margin(0.5));
}

TEST_CASE("IGBT antiparallel diode disabled: legacy behaviour preserved",
          "[v1][igbt][antiparallel_diode][b2][regression]") {
    Circuit circuit;
    const Index n_high = circuit.add_node("high");
    const Index n_coll = circuit.add_node("coll");
    const Index n_gate = circuit.add_node("gate");
    const Index gnd    = Circuit::ground();

    constexpr Real V_high = 50.0;
    circuit.add_voltage_source("Vhigh", n_high, gnd, V_high);
    circuit.add_resistor("Rload", n_coll, gnd, 100.0);
    circuit.add_resistor("Rg", n_gate, gnd, 1.0e3);

    IGBT::Params ip{};
    ip.vth = 5.0;
    ip.g_on = 1e4;
    ip.g_off = 1e-12;
    REQUIRE_FALSE(ip.antiparallel_diode_enable);
    circuit.add_igbt("Q", n_gate, n_coll, n_high, ip);

    auto opts = make_dc_opts(circuit);
    Simulator sim(circuit, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);

    const auto& x = dc.newton_result.solution;
    INFO("V(coll) = " << x[n_coll] << " V");

    // With OFF antiparallel diode and OFF channel, V_coll pulled to 0
    // through Rload via g_off.
    CHECK(std::abs(x[n_coll]) < 0.1);
}
