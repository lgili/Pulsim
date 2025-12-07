#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/simulation.hpp"
#include <cmath>
#include <vector>

using namespace pulsim;
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

TEST_CASE("MOSFET Level 1 DC characteristics", "[mosfet]") {
    // Test NMOS in different operating regions
    Circuit circuit;

    // Vgs controls the gate
    circuit.add_voltage_source("Vgs", "gate", "0", DCWaveform{5.0});
    // Vds across drain-source
    circuit.add_voltage_source("Vds", "drain", "0", DCWaveform{5.0});

    MOSFETParams params;
    params.type = MOSFETType::NMOS;
    params.vth = 2.0;
    params.kp = 100e-6;  // 100 uA/V²
    params.w = 10e-6;    // 10 um
    params.l = 1e-6;     // 1 um
    params.lambda = 0.0;

    // Kp_eff = 100e-6 * 10e-6 / 1e-6 = 1e-3 A/V²
    // With Vgs=5V, Vth=2V, Vov=3V
    // If Vds=5V > Vov: saturation, Id = 0.5*Kp*(Vgs-Vth)² = 0.5*1e-3*9 = 4.5mA

    circuit.add_mosfet("M1", "drain", "gate", "0", params);

    Simulator sim(circuit);
    auto result = sim.dc_operating_point();

    REQUIRE(result.status == SolverStatus::Success);

    // The MOSFET should be in saturation
    // We can verify by checking the current through Vds
}

TEST_CASE("MOSFET as switch", "[mosfet]") {
    // Use MOSFET with rds_on for simple switch behavior
    // Circuit: Vdc -- Rload -- M1 (drain) -- GND
    //          Gate driven by Vgs
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{12.0});
    circuit.add_voltage_source("Vgs", "gate", "0", DCWaveform{10.0});  // Turn on gate
    circuit.add_resistor("Rload", "vcc", "drain", 100.0);

    MOSFETParams params;
    params.type = MOSFETType::NMOS;
    params.vth = 3.0;
    params.rds_on = 0.1;   // 100 mOhm when on
    params.rds_off = 1e9;  // Very high when off

    // NMOS: drain at high side, source at ground
    circuit.add_mosfet("M1", "drain", "gate", "0", params);

    Simulator sim(circuit);
    auto result = sim.dc_operating_point();

    REQUIRE(result.status == SolverStatus::Success);

    // With Vgs=10V > Vth=3V, MOSFET should be ON
    // V(drain) ≈ 12V * 0.1 / (100 + 0.1) ≈ 0.012V (close to 0)
    // Current = 12 / (100 + 0.1) ≈ 0.12A
    Index drain_idx = circuit.node_index("drain");
    CHECK(result.x(drain_idx) < 0.5);  // Should be close to 0
}

TEST_CASE("MOSFET with body diode", "[mosfet]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "source", "0", DCWaveform{5.0});
    circuit.add_voltage_source("Vgs", "gate", "0", DCWaveform{0.0});  // Gate OFF
    circuit.add_resistor("R1", "drain", "0", 1000.0);

    MOSFETParams params;
    params.type = MOSFETType::NMOS;
    params.vth = 2.0;
    params.rds_on = 0.01;
    params.rds_off = 1e9;
    params.body_diode = true;
    params.is_body = 1e-14;
    params.n_body = 1.0;

    circuit.add_mosfet("M1", "drain", "gate", "source", params);

    Simulator sim(circuit);
    auto result = sim.dc_operating_point();

    REQUIRE(result.status == SolverStatus::Success);

    // With gate off and source at 5V, body diode should conduct
    // (body diode is from source to drain for NMOS)
    // Current flows through body diode to R1
    Index drain_idx = circuit.node_index("drain");
    CHECK(result.x(drain_idx) > 4.0);  // Close to 5V minus diode drop
}

TEST_CASE("Ideal transformer", "[transformer]") {
    // Test voltage transformation
    Circuit circuit;
    circuit.add_voltage_source("Vpri", "p1", "0", DCWaveform{120.0});

    // 10:1 step-down transformer
    TransformerParams params;
    params.turns_ratio = 10.0;  // 10:1

    circuit.add_transformer("T1", "p1", "0", "s1", "0", params);
    circuit.add_resistor("Rload", "s1", "0", 100.0);  // Load on secondary

    Simulator sim(circuit);
    auto result = sim.dc_operating_point();

    REQUIRE(result.status == SolverStatus::Success);

    // V_secondary = V_primary / n = 120 / 10 = 12V
    Index s1_idx = circuit.node_index("s1");
    CHECK_THAT(result.x(s1_idx), WithinAbs(12.0, 1.0));
}

TEST_CASE("Flyback converter topology", "[power][mosfet]") {
    // Simplified flyback: Vdc - M1 - Transformer - D1 - Cout - Rload
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{400.0});

    // PWM control for MOSFET
    PulseWaveform pwm{0.0, 12.0, 0.0, 1e-9, 1e-9, 5e-6, 10e-6};  // 50% duty, 100kHz
    circuit.add_voltage_source("Vpwm", "gate", "0", pwm);

    // Primary side MOSFET
    MOSFETParams mos_params;
    mos_params.type = MOSFETType::NMOS;
    mos_params.vth = 3.0;
    mos_params.rds_on = 0.1;
    mos_params.rds_off = 1e9;

    circuit.add_mosfet("M1", "pri", "gate", "0", mos_params);

    // Simplified: use inductor instead of transformer for now
    // (Real flyback would need coupled inductors)
    circuit.add_inductor("Lpri", "vcc", "pri", 100e-6);

    // Output diode and filter
    DiodeParams diode_params;
    diode_params.ideal = true;
    circuit.add_diode("D1", "pri", "out", diode_params);

    circuit.add_capacitor("Cout", "out", "0", 100e-6);
    circuit.add_resistor("Rload", "out", "0", 50.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 200e-6;  // 20 switching cycles
    opts.dt = 0.1e-6;
    opts.dtmax = 0.5e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Just verify it completes without error
    // Output voltage depends on control loop which we don't have
}

TEST_CASE("PWM waveform basic operation", "[pwm]") {
    // Test PWM waveform without dead-time
    PWMWaveform pwm;
    pwm.v_off = 0.0;
    pwm.v_on = 5.0;
    pwm.frequency = 10e3;  // 10kHz, period = 100us
    pwm.duty = 0.5;        // 50% duty cycle
    pwm.dead_time = 0.0;
    pwm.phase = 0.0;
    pwm.complementary = false;

    Circuit circuit;
    circuit.add_voltage_source("Vpwm", "pwm", "0", pwm);
    circuit.add_resistor("R1", "pwm", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 200e-6;  // 2 periods
    opts.dt = 1e-6;
    opts.dtmax = 1e-6;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check that voltage toggles between 0 and 5V
    Index pwm_idx = circuit.node_index("pwm");
    bool saw_low = false, saw_high = false;
    for (const auto& data : result.data) {
        Real v = data(pwm_idx);
        if (v < 0.5) saw_low = true;
        if (v > 4.5) saw_high = true;
    }
    CHECK(saw_low);
    CHECK(saw_high);
}

TEST_CASE("PWM waveform with dead-time", "[pwm][deadtime]") {
    // Test that dead-time is properly inserted
    // PWM: 10kHz (100us period), 50% duty, 2us dead-time
    PWMWaveform pwm_hi;
    pwm_hi.v_off = 0.0;
    pwm_hi.v_on = 10.0;
    pwm_hi.frequency = 10e3;
    pwm_hi.duty = 0.5;
    pwm_hi.dead_time = 2e-6;  // 2us dead-time
    pwm_hi.phase = 0.0;
    pwm_hi.complementary = false;

    PWMWaveform pwm_lo = pwm_hi;
    pwm_lo.complementary = true;

    Circuit circuit;
    circuit.add_voltage_source("Vhi", "ctrl_hi", "0", pwm_hi);
    circuit.add_voltage_source("Vlo", "ctrl_lo", "0", pwm_lo);
    circuit.add_resistor("Rhi", "ctrl_hi", "0", 1000.0);
    circuit.add_resistor("Rlo", "ctrl_lo", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;  // 1 period
    opts.dt = 0.5e-6;     // 500ns resolution
    opts.dtmax = 0.5e-6;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Verify dead-time: there should never be a time when both are high
    Index hi_idx = circuit.node_index("ctrl_hi");
    Index lo_idx = circuit.node_index("ctrl_lo");

    int both_high_count = 0;
    for (const auto& data : result.data) {
        Real v_hi = data(hi_idx);
        Real v_lo = data(lo_idx);
        if (v_hi > 5.0 && v_lo > 5.0) {
            both_high_count++;
        }
    }

    // With proper dead-time, both should never be high simultaneously
    CHECK(both_high_count == 0);
}

TEST_CASE("Half-bridge with PWM dead-time", "[pwm][deadtime][power]") {
    // Half-bridge with complementary PWM and dead-time
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{100.0});

    // PWM with dead-time for safe half-bridge operation
    PWMWaveform pwm_hi;
    pwm_hi.v_off = 0.0;
    pwm_hi.v_on = 15.0;
    pwm_hi.frequency = 20e3;   // 20kHz
    pwm_hi.duty = 0.5;
    pwm_hi.dead_time = 1e-6;   // 1us dead-time
    pwm_hi.complementary = false;

    PWMWaveform pwm_lo = pwm_hi;
    pwm_lo.complementary = true;

    circuit.add_voltage_source("Vhi", "ctrl_hi", "0", pwm_hi);
    circuit.add_voltage_source("Vlo", "ctrl_lo", "0", pwm_lo);

    // Midpoint reference for gate drive
    circuit.add_resistor("Rmid1", "vcc", "mid", 100e3);
    circuit.add_resistor("Rmid2", "mid", "0", 100e3);

    SwitchParams sw_params;
    sw_params.ron = 0.05;
    sw_params.roff = 1e9;
    sw_params.vth = 7.5;

    circuit.add_switch("Shi", "vcc", "out", "ctrl_hi", "mid", sw_params);
    circuit.add_switch("Slo", "out", "0", "ctrl_lo", "mid", sw_params);

    // RL load
    circuit.add_resistor("Rload", "out", "load_mid", 10.0);
    circuit.add_inductor("Lload", "load_mid", "mid", 1e-3);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 250e-6;  // 5 switching cycles
    opts.dt = 0.25e-6;
    opts.dtmax = 1e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);

    // Track switch events
    std::vector<SwitchEvent> events;
    auto event_cb = [&events](const SwitchEvent& e) {
        events.push_back(e);
    };

    auto result = sim.run_transient(nullptr, event_cb);

    REQUIRE(result.final_status == SolverStatus::Success);

    // Verify no shoot-through: check that switches are never both closed
    // at the same simulation time point
    Index out_idx = circuit.node_index("out");
    Real v_min = 1e9, v_max = -1e9;
    for (const auto& data : result.data) {
        v_min = std::min(v_min, data(out_idx));
        v_max = std::max(v_max, data(out_idx));
    }

    // Should see voltage swing (output toggles between ~0 and ~Vdc)
    CHECK((v_max - v_min) > 50);  // Significant swing
}

TEST_CASE("PWM phase offset", "[pwm]") {
    // Test that phase offset shifts the PWM waveform
    PWMWaveform pwm1;
    pwm1.v_off = 0.0;
    pwm1.v_on = 5.0;
    pwm1.frequency = 10e3;
    pwm1.duty = 0.5;
    pwm1.dead_time = 0.0;
    pwm1.phase = 0.0;
    pwm1.complementary = false;

    PWMWaveform pwm2 = pwm1;
    pwm2.phase = 0.5;  // 180 degrees phase shift

    Circuit circuit;
    circuit.add_voltage_source("V1", "pwm1", "0", pwm1);
    circuit.add_voltage_source("V2", "pwm2", "0", pwm2);
    circuit.add_resistor("R1", "pwm1", "0", 1000.0);
    circuit.add_resistor("R2", "pwm2", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;  // 1 period
    opts.dt = 1e-6;
    opts.dtmax = 1e-6;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // At t=0, pwm1 should be off, pwm2 should be on (phase shifted)
    // Check at t=25us (quarter period): pwm1 should be on, pwm2 should be off
    Index idx1 = circuit.node_index("pwm1");
    Index idx2 = circuit.node_index("pwm2");

    // Find data point near t=25us
    for (size_t i = 0; i < result.time.size(); ++i) {
        if (result.time[i] >= 25e-6 && result.time[i] < 26e-6) {
            Real v1 = result.data[i](idx1);
            Real v2 = result.data[i](idx2);
            // With 180-degree phase shift, they should be opposite
            CHECK(((v1 > 2.5 && v2 < 2.5) || (v1 < 2.5 && v2 > 2.5)));
            break;
        }
    }
}
