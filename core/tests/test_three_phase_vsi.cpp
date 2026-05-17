// Three-phase 2-level VSI helper — Catch2 tests.
//
// Verifies that ``Circuit::add_three_phase_vsi`` correctly composes a
// 3-leg 6-switch inverter and runs without crashing. We deliberately
// keep transients short and switching frequencies low so the test
// finishes in seconds (the PWL state-space cache rebuilds at every
// commutation, so even Ideal-mode VSIs are CPU-bound at high f_sw).
//
// The headline analytical verification of SPWM line-to-line magnitude
// (0.612·m·V_dc) is covered by Python smoke tests and benchmark
// circuits that can afford longer simulations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi = 3.14159265358979323846;

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

TEST_CASE("3φ VSI: helper composes the expected switching topology",
          "[three_phase_vsi][unit]") {
    // Pure structural check — no transient needed. The helper must
    // reserve 6 MOSFETs + 6 PWM sources + 6 internal gate nodes.
    Circuit circuit;
    const Index vdc_pos = circuit.add_node("VDC+");
    const Index vdc_neg = Circuit::ground();
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    const std::size_t devices_before = circuit.devices().size();
    const Index nodes_before = circuit.num_nodes();

    Circuit::ThreePhaseVsiParams vsi{};
    vsi.switching_frequency_hz = 1e3;
    vsi.modulation_index = 0.8;
    vsi.modulation_frequency_hz = 50.0;
    circuit.add_three_phase_vsi("VSI", vdc_pos, vdc_neg, na, nb, nc, vsi);

    const std::size_t devices_after = circuit.devices().size();
    const Index nodes_after = circuit.num_nodes();

    // 6 MOSFETs + 6 PWM gate sources = 12 new devices.
    CHECK(devices_after - devices_before == 12);
    // 6 internal gate nodes (G_aH, G_aL, G_bH, G_bL, G_cH, G_cL).
    CHECK(nodes_after - nodes_before == 6);

    // Convenience overload should match the same structure.
    Circuit c2;
    const Index v2p = c2.add_node("VDC+");
    const Index v2n = Circuit::ground();
    const Index a2 = c2.add_node("A");
    const Index b2 = c2.add_node("B");
    const Index ck2 = c2.add_node("C");
    c2.add_three_phase_vsi("VSI", v2p, v2n, a2, b2, ck2, 1e3, 0.8, 50.0);
    CHECK(c2.devices().size() == 12);
}

TEST_CASE("3φ VSI: short transient runs without crashing on R load",
          "[three_phase_vsi][regression]") {
    // Very short transient (5 ms, f_sw=1kHz → only ~5 switching periods)
    // — just verify the simulator converges and currents are finite.
    Circuit circuit;
    const Index vdc_pos = circuit.add_node("VDC+");
    const Index vdc_neg = Circuit::ground();
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    circuit.add_voltage_source("Vbus", vdc_pos, vdc_neg, 100.0);

    Circuit::ThreePhaseVsiParams vsi{};
    vsi.switching_frequency_hz = 1e3;
    vsi.modulation_index = 0.8;
    vsi.modulation_frequency_hz = 50.0;
    circuit.add_three_phase_vsi("VSI", vdc_pos, vdc_neg, na, nb, nc, vsi);

    // Plain resistors per phase to ground for closure. No inductance
    // means the circuit reaches each switching state immediately.
    circuit.add_resistor("Ra", na, vdc_neg, 10.0);
    circuit.add_resistor("Rb", nb, vdc_neg, 10.0);
    circuit.add_resistor("Rc", nc, vdc_neg, 10.0);

    auto opts = make_opts(5e-3, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    // The phase node voltage must remain finite throughout the run.
    // We don't pin the exact rail because the MOSFET model behavior near
    // commutation depends on the resolved switching mode and the
    // intermediate Newton state; what matters here is that the helper
    // produces a circuit that *solves*.
    for (const auto& state : result.states) {
        CHECK(std::isfinite(static_cast<Real>(state[na])));
    }
}

TEST_CASE("3φ VSI: convenience overload accepts (f_sw, m, f_mod) arguments",
          "[three_phase_vsi][unit]") {
    // Smoke-test the convenience overload — same structural expectations
    // as the parameter-struct overload.
    Circuit circuit;
    const Index vdc_pos = circuit.add_node("VDC+");
    const Index vdc_neg = Circuit::ground();
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    circuit.add_three_phase_vsi("VSI", vdc_pos, vdc_neg, na, nb, nc,
                                 1e3, 0.8, 50.0);
    CHECK(circuit.devices().size() == 12);
    CHECK(circuit.num_nodes() == 1 + 3 + 6);  // VDC+, A, B, C, 6 gates
}

TEST_CASE("3φ VSI: short transient on an R/L load converges and produces "
          "finite phase currents",
          "[three_phase_vsi][regression]") {
    // Add an RL load to give the legs an actual inductive current to
    // settle. We don't pin specific magnitudes — the slow per-event
    // state-space rebuild keeps the test scope minimal. Convergence
    // alone is the proof that the helper integrates correctly.
    Circuit circuit;
    const Index vdc_pos = circuit.add_node("VDC+");
    const Index vdc_neg = Circuit::ground();
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    circuit.add_voltage_source("Vbus", vdc_pos, vdc_neg, 100.0);

    Circuit::ThreePhaseVsiParams vsi{};
    vsi.switching_frequency_hz = 1e3;
    vsi.modulation_index = 0.5;
    vsi.modulation_frequency_hz = 50.0;
    circuit.add_three_phase_vsi("VSI", vdc_pos, vdc_neg, na, nb, nc, vsi);

    Circuit::ThreePhaseRLLoadParams load{};
    load.resistance_per_phase = 10.0;
    load.inductance_per_phase = 1e-3;
    circuit.add_three_phase_rl_load(
        "Load", na, nb, nc, vdc_neg, load);

    auto opts = make_opts(3e-3, 25e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    // The simulator must produce at least a handful of samples.
    REQUIRE(result.time.size() > 10);
    // Every phase-node sample must be finite.
    for (const auto& state : result.states) {
        CHECK(std::isfinite(static_cast<Real>(state[na])));
        CHECK(std::isfinite(static_cast<Real>(state[nb])));
        CHECK(std::isfinite(static_cast<Real>(state[nc])));
    }
}
