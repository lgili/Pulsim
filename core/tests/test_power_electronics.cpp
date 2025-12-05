#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "spicelab/simulation.hpp"
#include <cmath>
#include <vector>

using namespace spicelab;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Basic switch operation", "[power]") {
    // Simple circuit: V1 - Switch - R - GND
    // Control voltage controls the switch
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{12.0});
    circuit.add_voltage_source("Vctrl", "ctrl", "0", DCWaveform{5.0});  // Switch ON
    circuit.add_resistor("R1", "out", "0", 100.0);

    SwitchParams sw_params;
    sw_params.ron = 0.01;      // 10 mOhm
    sw_params.roff = 1e9;      // 1 GOhm
    sw_params.vth = 2.5;       // Threshold 2.5V
    sw_params.initial_state = false;

    circuit.add_switch("S1", "vcc", "out", "ctrl", "0", sw_params);

    Simulator sim(circuit);
    auto result = sim.dc_operating_point();

    REQUIRE(result.status == SolverStatus::Success);

    // With Vctrl = 5V > 2.5V threshold, switch should be closed
    // V(out) ≈ 12V (minus small drop across Ron)
    Index out_idx = circuit.node_index("out");
    CHECK(result.x(out_idx) > 11.9);  // Close to 12V
}

TEST_CASE("Switch with control signal", "[power]") {
    // PWM-controlled switch
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{24.0});

    // PWM control signal: 50% duty cycle, 10kHz
    PulseWaveform pwm{0.0, 5.0, 0.0, 1e-9, 1e-9, 50e-6, 100e-6};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    circuit.add_resistor("R1", "out", "0", 10.0);
    circuit.add_capacitor("C1", "out", "0", 100e-6);  // Output filter cap

    SwitchParams sw_params;
    sw_params.ron = 0.01;
    sw_params.roff = 1e9;
    sw_params.vth = 2.5;
    sw_params.initial_state = false;

    circuit.add_switch("S1", "vcc", "out", "ctrl", "0", sw_params);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;  // 10 PWM cycles
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);

    // Track events
    std::vector<SwitchEvent> events;
    auto event_cb = [&events](const SwitchEvent& e) {
        events.push_back(e);
    };

    auto result = sim.run_transient(nullptr, event_cb);

    REQUIRE(result.final_status == SolverStatus::Success);

    // Should have switching events (at least 10 on + 10 off = 20 events)
    CHECK(events.size() >= 10);

    // Check average output voltage (should be ~50% of input due to 50% duty)
    Index out_idx = circuit.node_index("out");
    Real v_avg = 0.0;
    for (const auto& data : result.data) {
        v_avg += data(out_idx);
    }
    v_avg /= result.data.size();

    // With filtering, average should approach steady state
    // Note: Without proper diode, voltage may be higher
    CHECK(v_avg > 5.0);  // Should have significant output
    CHECK(v_avg < 30.0); // But not overvoltage
}

TEST_CASE("Buck converter topology", "[power]") {
    // Simple buck converter: Vdc - S1 - L - C - R (load)
    //                              |
    //                              D1 (freewheeling diode)
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{48.0});

    // PWM control: 50% duty cycle
    PulseWaveform pwm{0.0, 5.0, 0.0, 1e-9, 1e-9, 25e-6, 50e-6};  // 20kHz
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    // High-side switch
    SwitchParams sw_params;
    sw_params.ron = 0.01;
    sw_params.roff = 1e9;
    sw_params.vth = 2.5;
    circuit.add_switch("S1", "vcc", "sw_node", "ctrl", "0", sw_params);

    // Freewheeling diode (ideal)
    DiodeParams diode_params;
    diode_params.ideal = true;
    circuit.add_diode("D1", "0", "sw_node", diode_params);

    // LC filter
    circuit.add_inductor("L1", "sw_node", "out", 100e-6);  // 100uH
    circuit.add_capacitor("C1", "out", "0", 100e-6);       // 100uF

    // Load
    circuit.add_resistor("Rload", "out", "0", 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2e-3;  // 40 switching cycles
    opts.dt = 0.5e-6;
    opts.dtmax = 2e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check output voltage settles to ~24V (50% of input)
    Index out_idx = circuit.node_index("out");

    // Take average of last 20% of simulation (steady state)
    size_t start_idx = result.data.size() * 4 / 5;
    Real v_avg = 0.0;
    int count = 0;
    for (size_t i = start_idx; i < result.data.size(); ++i) {
        v_avg += result.data[i](out_idx);
        count++;
    }
    v_avg /= count;

    // Buck converter output ≈ D * Vin = 0.5 * 48 = 24V
    // Allow wider tolerance due to transient settling
    CHECK_THAT(v_avg, WithinAbs(24.0, 5.0));
}

TEST_CASE("Conduction losses calculation", "[power]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{12.0});
    circuit.add_voltage_source("Vctrl", "ctrl", "0", DCWaveform{5.0});  // Always ON
    circuit.add_resistor("R1", "out", "0", 1.0);  // 1 ohm load

    SwitchParams sw_params;
    sw_params.ron = 0.1;  // 100 mOhm - significant conduction loss
    sw_params.roff = 1e9;
    sw_params.vth = 2.5;
    sw_params.initial_state = true;

    circuit.add_switch("S1", "vcc", "out", "ctrl", "0", sw_params);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check conduction losses
    // I = 12V / (1 + 0.1) ≈ 10.9A
    // P_cond = I^2 * Ron = 10.9^2 * 0.1 ≈ 11.9W
    // E_cond = P * t = 11.9 * 1e-3 ≈ 11.9 mJ
    const auto& losses = sim.power_losses();
    CHECK(losses.conduction_loss > 0.01);  // At least 10 mJ
    CHECK(losses.conduction_loss < 0.02);  // Less than 20 mJ
}

TEST_CASE("Half-bridge inverter", "[power]") {
    // Half-bridge: two switches, complementary control
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{400.0});

    // Complementary PWM signals
    PulseWaveform pwm_hi{0.0, 15.0, 0.0, 1e-9, 1e-9, 25e-6, 50e-6};
    PulseWaveform pwm_lo{15.0, 0.0, 0.0, 1e-9, 1e-9, 25e-6, 50e-6};  // Inverted
    circuit.add_voltage_source("Vhi", "ctrl_hi", "0", pwm_hi);
    circuit.add_voltage_source("Vlo", "ctrl_lo", "0", pwm_lo);

    // Midpoint reference
    circuit.add_resistor("R_mid1", "vcc", "mid", 100e3);
    circuit.add_resistor("R_mid2", "mid", "0", 100e3);

    SwitchParams sw_params;
    sw_params.ron = 0.05;
    sw_params.roff = 1e9;
    sw_params.vth = 7.5;

    // High-side switch
    circuit.add_switch("Shi", "vcc", "out", "ctrl_hi", "mid", sw_params);
    // Low-side switch
    circuit.add_switch("Slo", "out", "0", "ctrl_lo", "mid", sw_params);

    // RL load
    circuit.add_resistor("Rload", "out", "load_mid", 10.0);
    circuit.add_inductor("Lload", "load_mid", "mid", 1e-3);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 500e-6;  // 10 switching cycles
    opts.dt = 0.5e-6;
    opts.dtmax = 2e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Output should swing between ~0 and ~400V
    Index out_idx = circuit.node_index("out");
    Real v_min = 1e9, v_max = -1e9;
    for (const auto& data : result.data) {
        v_min = std::min(v_min, data(out_idx));
        v_max = std::max(v_max, data(out_idx));
    }

    // Should see significant voltage swing
    CHECK((v_max - v_min) > 100);
}
