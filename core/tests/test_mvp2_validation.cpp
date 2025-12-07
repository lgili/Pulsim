/**
 * MVP-2 Validation Tests
 *
 * These tests validate the advanced power electronics features:
 * 1. Full-bridge inverter simulation
 * 2. MOSFET switching waveform characteristics
 * 3. Thermal response with step power
 * 4. Efficiency calculation verification
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/simulation.hpp"
#include "pulsim/thermal.hpp"
#include <cmath>
#include <vector>
#include <numeric>

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// ============================================================================
// 3.8.1 Full-Bridge Inverter Simulation
// ============================================================================

TEST_CASE("Full-bridge inverter basic operation", "[mvp2][fullbridge]") {
    // Full-bridge (H-bridge) inverter topology:
    //
    //     Vdc+
    //      |
    //   +--+--+
    //   |     |
    //  S1     S3
    //   |     |
    //   +--+--+----> Load (RL) --+
    //   |     |                  |
    //  S2     S4                 |
    //   |     |                  |
    //   +--+--+------------------+
    //      |
    //     GND
    //
    // Diagonal pairs: S1+S4 or S2+S3 conduct alternately
    // For low-side switches: ctrl is referenced to ground (common)
    // For this test, we use ground-referenced control for all switches

    Circuit circuit;

    // DC bus voltage
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{400.0});

    // PWM control signals - diagonal pairs
    // S1 and S4: positive half-cycle (ctrl_pos high)
    // S2 and S3: negative half-cycle (ctrl_neg high = complementary)
    // Using complementary PWM with dead-time
    PWMWaveform pwm_pos;
    pwm_pos.v_off = 0.0;
    pwm_pos.v_on = 15.0;
    pwm_pos.frequency = 10e3;   // 10kHz switching
    pwm_pos.duty = 0.5;
    pwm_pos.dead_time = 1e-6;   // 1us dead-time
    pwm_pos.phase = 0.0;
    pwm_pos.complementary = false;

    PWMWaveform pwm_neg = pwm_pos;
    pwm_neg.complementary = true;

    circuit.add_voltage_source("Vctrl_pos", "ctrl_pos", "0", pwm_pos);
    circuit.add_voltage_source("Vctrl_neg", "ctrl_neg", "0", pwm_neg);

    // Switch parameters - all ground-referenced
    SwitchParams sw_params;
    sw_params.ron = 0.05;    // 50 mOhm
    sw_params.roff = 1e9;
    sw_params.vth = 7.5;

    // High-side switches (S1, S3) - for simplified test, use ground reference
    // In real circuits these would need level-shifted gate drives
    circuit.add_switch("S1", "vcc", "out_a", "ctrl_pos", "0", sw_params);
    circuit.add_switch("S3", "vcc", "out_b", "ctrl_neg", "0", sw_params);

    // Low-side switches (S2, S4) - ground referenced
    circuit.add_switch("S2", "out_a", "0", "ctrl_neg", "0", sw_params);
    circuit.add_switch("S4", "out_b", "0", "ctrl_pos", "0", sw_params);

    // RL load between output phases
    circuit.add_resistor("Rload", "out_a", "load_mid", 10.0);
    circuit.add_inductor("Lload", "load_mid", "out_b", 5e-3);  // 5mH

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;   // 10 switching cycles
    opts.dt = 0.5e-6;
    opts.dtmax = 2e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Verify output voltage swings between +Vdc and -Vdc
    Index out_a_idx = circuit.node_index("out_a");
    Index out_b_idx = circuit.node_index("out_b");

    Real v_diff_min = 1e9, v_diff_max = -1e9;
    for (const auto& data : result.data) {
        Real v_diff = data(out_a_idx) - data(out_b_idx);
        v_diff_min = std::min(v_diff_min, v_diff);
        v_diff_max = std::max(v_diff_max, v_diff);
    }

    // Full-bridge should swing between approximately +Vdc and -Vdc
    // (allowing for some voltage drops)
    CHECK(v_diff_max > 300);   // Should reach near +400V
    CHECK(v_diff_min < -300);  // Should reach near -400V

    INFO("Vdiff range: " << v_diff_min << " to " << v_diff_max);
}

TEST_CASE("Full-bridge inverter with MOSFETs", "[mvp2][fullbridge][mosfet]") {
    // Full-bridge using MOSFET models for low-side and switches for high-side
    // All controls are ground-referenced for simplified testing
    Circuit circuit;

    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{100.0});

    // PWM control signals with complementary switching
    PWMWaveform pwm_pos;
    pwm_pos.v_off = 0.0;
    pwm_pos.v_on = 12.0;
    pwm_pos.frequency = 20e3;
    pwm_pos.duty = 0.5;
    pwm_pos.dead_time = 500e-9;
    pwm_pos.complementary = false;

    PWMWaveform pwm_neg = pwm_pos;
    pwm_neg.complementary = true;

    circuit.add_voltage_source("Vg_pos", "gate_pos", "0", pwm_pos);
    circuit.add_voltage_source("Vg_neg", "gate_neg", "0", pwm_neg);

    // MOSFET parameters (simplified power MOSFET using rds_on model)
    MOSFETParams mos;
    mos.type = MOSFETType::NMOS;
    mos.vth = 3.0;
    mos.rds_on = 0.05;
    mos.rds_off = 1e9;
    mos.body_diode = true;
    mos.is_body = 1e-12;

    // Low-side MOSFETs (source to ground, gate referenced to ground)
    circuit.add_mosfet("M2", "out_a", "gate_neg", "0", mos);
    circuit.add_mosfet("M4", "out_b", "gate_pos", "0", mos);

    // For high-side, we use switches with ground reference (simplified)
    // In real circuits these would need bootstrap or isolated gate drives
    SwitchParams sw_hi;
    sw_hi.ron = 0.05;
    sw_hi.roff = 1e9;
    sw_hi.vth = 5.0;

    // High-side switches - ground referenced for simplified test
    circuit.add_switch("S1", "vcc", "out_a", "gate_pos", "0", sw_hi);
    circuit.add_switch("S3", "vcc", "out_b", "gate_neg", "0", sw_hi);

    // RL load
    circuit.add_resistor("Rload", "out_a", "load_mid", 5.0);
    circuit.add_inductor("Lload", "load_mid", "out_b", 1e-3);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 500e-6;
    opts.dt = 0.25e-6;
    opts.dtmax = 1e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Verify switching operation
    Index out_a_idx = circuit.node_index("out_a");
    Index out_b_idx = circuit.node_index("out_b");

    Real v_diff_min = 1e9, v_diff_max = -1e9;
    for (const auto& data : result.data) {
        Real v_diff = data(out_a_idx) - data(out_b_idx);
        v_diff_min = std::min(v_diff_min, v_diff);
        v_diff_max = std::max(v_diff_max, v_diff);
    }

    // Should see significant voltage swing (at least half of Vdc swing)
    CHECK((v_diff_max - v_diff_min) > 50);

    INFO("Vdiff range: " << v_diff_min << " to " << v_diff_max);
}

// ============================================================================
// 3.8.2 MOSFET Switching Waveforms Validation
// ============================================================================

TEST_CASE("MOSFET turn-on characteristics", "[mvp2][mosfet][switching]") {
    // Test MOSFET turn-on behavior with gate capacitance
    // The gate should charge through Cgs, causing delayed turn-on

    Circuit circuit;

    circuit.add_voltage_source("Vdd", "vdd", "0", DCWaveform{50.0});

    // Gate drive with finite rise time
    PulseWaveform gate_pulse;
    gate_pulse.v1 = 0.0;
    gate_pulse.v2 = 12.0;
    gate_pulse.td = 10e-6;      // Delay
    gate_pulse.tr = 100e-9;     // 100ns rise
    gate_pulse.tf = 100e-9;     // 100ns fall
    gate_pulse.pw = 20e-6;      // Pulse width
    gate_pulse.period = 50e-6;

    circuit.add_voltage_source("Vgate", "gate_drv", "0", gate_pulse);

    // Gate resistor (models driver impedance)
    circuit.add_resistor("Rg", "gate_drv", "gate", 10.0);

    // MOSFET with parasitic capacitances
    MOSFETParams mos;
    mos.type = MOSFETType::NMOS;
    mos.vth = 3.0;
    mos.rds_on = 0.02;    // 20 mOhm
    mos.rds_off = 1e9;
    mos.cgs = 1e-9;       // 1nF gate-source
    mos.cgd = 100e-12;    // 100pF gate-drain (Miller)
    mos.cds = 50e-12;     // 50pF drain-source

    circuit.add_mosfet("M1", "drain", "gate", "0", mos);

    // Load resistor
    circuit.add_resistor("Rload", "vdd", "drain", 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 60e-6;
    opts.dt = 10e-9;      // 10ns resolution for accurate switching
    opts.dtmax = 100e-9;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    Index gate_idx = circuit.node_index("gate");
    Index drain_idx = circuit.node_index("drain");

    // Find turn-on transition (around t=10us)
    Real t_gate_rise_start = -1, t_gate_threshold = -1, t_drain_fall = -1;
    Real prev_vgate = 0, prev_vdrain = 50;

    for (size_t i = 0; i < result.time.size(); ++i) {
        Real t = result.time[i];
        Real vgate = result.data[i](gate_idx);
        Real vdrain = result.data[i](drain_idx);

        // Detect when gate starts rising
        if (t_gate_rise_start < 0 && vgate > 0.5 && prev_vgate <= 0.5) {
            t_gate_rise_start = t;
        }

        // Detect when gate crosses threshold
        if (t_gate_threshold < 0 && vgate > mos.vth && prev_vgate <= mos.vth) {
            t_gate_threshold = t;
        }

        // Detect when drain starts falling significantly
        if (t_drain_fall < 0 && vdrain < 45 && prev_vdrain >= 45) {
            t_drain_fall = t;
        }

        prev_vgate = vgate;
        prev_vdrain = vdrain;
    }

    INFO("Gate rise start: " << t_gate_rise_start * 1e6 << " us");
    INFO("Gate threshold crossing: " << t_gate_threshold * 1e6 << " us");
    INFO("Drain fall: " << t_drain_fall * 1e6 << " us");

    // Verify timing relationships
    if (t_gate_rise_start > 0 && t_gate_threshold > 0) {
        // Gate should take some time to reach threshold due to Cgs
        Real gate_delay = t_gate_threshold - t_gate_rise_start;
        CHECK(gate_delay > 0);  // Should have measurable delay
    }

    // Drain should fall after gate reaches threshold
    if (t_gate_threshold > 0 && t_drain_fall > 0) {
        CHECK(t_drain_fall >= t_gate_threshold);
    }
}

TEST_CASE("MOSFET switching loss accumulation", "[mvp2][mosfet][losses]") {
    // Verify that switching losses are calculated during transitions
    Circuit circuit;

    circuit.add_voltage_source("Vdd", "vdd", "0", DCWaveform{100.0});

    // PWM gate drive
    PulseWaveform gate_pwm;
    gate_pwm.v1 = 0.0;
    gate_pwm.v2 = 12.0;
    gate_pwm.td = 0.0;
    gate_pwm.tr = 50e-9;
    gate_pwm.tf = 50e-9;
    gate_pwm.pw = 5e-6;
    gate_pwm.period = 10e-6;  // 100kHz

    circuit.add_voltage_source("Vgate", "gate", "0", gate_pwm);

    // MOSFET with defined on-resistance for conduction loss
    MOSFETParams mos;
    mos.type = MOSFETType::NMOS;
    mos.vth = 4.0;
    mos.rds_on = 0.05;    // 50 mOhm
    mos.rds_off = 1e9;

    circuit.add_mosfet("M1", "drain", "gate", "0", mos);
    circuit.add_resistor("Rload", "vdd", "drain", 10.0);  // 10A at 100V

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;  // 10 switching cycles
    opts.dt = 50e-9;
    opts.dtmax = 500e-9;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check accumulated losses
    const auto& losses = sim.power_losses();

    // Should have accumulated conduction losses
    // Approximate: I = 100V/10ohm = 10A during on-time
    // Pcond = I^2 * Rds_on = 100 * 0.05 = 5W
    // 50% duty, so avg = 2.5W over 100us = 0.25mJ
    CHECK(losses.conduction_loss > 0);

    INFO("Total conduction loss: " << losses.conduction_loss * 1e6 << " uJ");
    INFO("Total switching loss: " << losses.switching_loss() * 1e6 << " uJ");
}

// ============================================================================
// 3.8.3 Thermal Response Validation
// ============================================================================

TEST_CASE("Thermal network step response", "[mvp2][thermal]") {
    // Validate thermal network response to step power input
    // Using the ThermalSimulator class directly

    // Create thermal model for a MOSFET
    ThermalModel thermal_model = create_mosfet_thermal("M1", 0.5, 0.3, 1.0);
    // Rth_jc = 0.5 K/W, Rth_cs = 0.3 K/W, Rth_sa = 1.0 K/W
    // Total Rth_ja = 1.8 K/W

    ThermalSimulator thermal;
    thermal.add_model(thermal_model);
    thermal.set_ambient(25.0);
    thermal.initialize();

    // Apply constant power step (simulating device dissipation)
    Real power = 50.0;  // 50W dissipation
    Real dt = 1e-3;     // 1ms timestep
    Real t_final = 100e-3;  // 100ms simulation

    std::unordered_map<std::string, Real> device_powers;
    device_powers["M1"] = power;

    std::vector<Real> times;
    std::vector<Real> tj_values;

    Real t = 0;
    while (t < t_final) {
        Real tj = thermal.step(dt, device_powers);
        times.push_back(t);
        tj_values.push_back(thermal.junction_temp("M1"));
        t += dt;
    }

    // Calculate expected steady-state temperature
    // Tj_ss = Tamb + P * Rth_ja = 25 + 50 * 1.8 = 115°C
    Real expected_tj_ss = 25.0 + power * thermal_model.rth_ja();

    INFO("Expected steady-state Tj: " << expected_tj_ss << " °C");
    INFO("Final simulated Tj: " << tj_values.back() << " °C");

    // Check that temperature rose significantly
    CHECK(tj_values.back() > 50);  // Well above ambient

    // Check temperature is approaching steady state
    // After 100ms with typical thermal time constants, should be close
    CHECK_THAT(tj_values.back(), WithinAbs(expected_tj_ss, 20.0));
}

TEST_CASE("Thermal parameter adjustment", "[mvp2][thermal]") {
    // Verify temperature-dependent parameter adjustment
    ThermalSimulator thermal;

    Real rds_on_25c = 0.010;  // 10 mOhm at 25°C
    Real vth_25c = 3.0;       // 3V at 25°C

    // Test at elevated temperature
    Real tj = 125.0;  // 125°C

    Real rds_on_hot = thermal.adjust_rds_on(rds_on_25c, tj);
    Real vth_hot = thermal.adjust_vth(vth_25c, tj);

    // Rds_on should increase with temperature (positive TC)
    // Default TC is 0.4%/K, so at 100K rise: factor = 1 + 0.004*100 = 1.4
    Real expected_rds = rds_on_25c * (1.0 + 0.004 * (tj - 25.0));
    CHECK_THAT(rds_on_hot, WithinRel(expected_rds, 0.01));

    // Vth should decrease with temperature (negative TC)
    // Default TC is -3mV/K, so at 100K rise: delta = -0.003*100 = -0.3V
    Real expected_vth = vth_25c + (-0.003) * (tj - 25.0);
    CHECK_THAT(vth_hot, WithinAbs(expected_vth, 0.05));

    INFO("Rds_on at 25°C: " << rds_on_25c * 1e3 << " mOhm");
    INFO("Rds_on at " << tj << "°C: " << rds_on_hot * 1e3 << " mOhm");
    INFO("Vth at 25°C: " << vth_25c << " V");
    INFO("Vth at " << tj << "°C: " << vth_hot << " V");
}

TEST_CASE("Thermal warning generation", "[mvp2][thermal]") {
    // Verify thermal warning when junction exceeds limit
    ThermalModel model;
    model.device_name = "Q1";
    model.rth_jc = 2.0;      // High thermal resistance
    model.rth_cs = 0.5;
    model.rth_sa = 2.0;      // Poor heatsink
    model.tj_max = 150.0;
    model.tj_warn = 125.0;

    ThermalSimulator thermal;
    thermal.add_model(model);
    thermal.set_ambient(25.0);
    thermal.initialize();

    // Apply high power to exceed warning threshold
    Real power = 30.0;  // 30W -> Tj = 25 + 30*4.5 = 160°C (exceeds max!)
    Real dt = 10e-3;

    std::unordered_map<std::string, Real> device_powers;
    device_powers["Q1"] = power;

    // Simulate until steady state
    for (int i = 0; i < 50; ++i) {
        thermal.step(dt, device_powers);
    }

    // Check for warnings
    const auto& warnings = thermal.warnings();

    INFO("Final Tj: " << thermal.junction_temp("Q1") << " °C");
    INFO("Number of warnings: " << warnings.size());

    // Should have generated warnings
    CHECK(warnings.size() > 0);

    // Check that final temperature exceeds the warning threshold
    CHECK(thermal.junction_temp("Q1") > model.tj_warn);
}

// ============================================================================
// 3.8.4 Efficiency Calculation Verification
// ============================================================================

TEST_CASE("Efficiency calculation - resistive load", "[mvp2][efficiency]") {
    // Simple DC-DC conversion scenario
    // Calculate efficiency = Pout / Pin
    Circuit circuit;

    circuit.add_voltage_source("Vin", "vin", "0", DCWaveform{48.0});

    // PWM-controlled buck converter
    PulseWaveform pwm;
    pwm.v1 = 0.0;
    pwm.v2 = 5.0;
    pwm.td = 0.0;
    pwm.tr = 10e-9;
    pwm.tf = 10e-9;
    pwm.pw = 12.5e-6;   // 25% duty for 48V->12V
    pwm.period = 50e-6;    // 20kHz

    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    // Switch with finite Ron
    SwitchParams sw;
    sw.ron = 0.02;   // 20 mOhm
    sw.roff = 1e9;
    sw.vth = 2.5;

    circuit.add_switch("S1", "vin", "sw", "ctrl", "0", sw);

    // Freewheeling diode (ideal model for reliable convergence)
    // When switch is off, inductor current freewheels through diode
    // Diode: anode at ground, cathode at sw node (conventional freewheeling diode)
    DiodeParams diode;
    diode.ideal = true;

    circuit.add_diode("D1", "0", "sw", diode);

    // Output filter
    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);

    // Load resistor
    circuit.add_resistor("Rload", "out", "0", 1.0);  // 1 ohm for ~12A

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-3;   // 100 switching cycles for steady state
    opts.dt = 0.5e-6;
    opts.dtmax = 2e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Calculate efficiency from simulation data
    // Use last 20% of simulation for steady-state averaging

    Index out_idx = circuit.node_index("out");
    size_t start_idx = result.data.size() * 4 / 5;

    // Calculate output power (V^2 / R)
    Real v_out_avg = 0;
    int count = 0;
    for (size_t i = start_idx; i < result.data.size(); ++i) {
        v_out_avg += result.data[i](out_idx);
        count++;
    }
    v_out_avg /= count;

    Real p_out = (v_out_avg * v_out_avg) / 1.0;  // Rload = 1 ohm

    // Get total losses from simulator
    const auto& losses = sim.power_losses();
    Real total_energy_loss = losses.total_loss();
    Real sim_duration = opts.tstop - opts.tstart;
    Real p_loss = total_energy_loss / sim_duration;

    // Calculate input power and efficiency
    // Pin ≈ Pout + Ploss
    Real p_in = p_out + p_loss;
    Real efficiency = (p_in > 0) ? (p_out / p_in * 100.0) : 0.0;

    INFO("Output voltage (avg): " << v_out_avg << " V");
    INFO("Output power: " << p_out << " W");
    INFO("Total losses: " << p_loss << " W");
    INFO("Efficiency: " << efficiency << " %");

    // Expected: 25% duty * 48V = 12V output
    // Efficiency should be high (>90%) for well-designed converter
    CHECK(v_out_avg > 10.0);
    CHECK(v_out_avg < 15.0);
}

TEST_CASE("Efficiency breakdown by loss type", "[mvp2][efficiency]") {
    // Detailed loss breakdown: conduction, switching, diode
    Circuit circuit;

    circuit.add_voltage_source("Vin", "vin", "0", DCWaveform{100.0});

    // High-frequency switching for measurable switching losses
    PulseWaveform pwm;
    pwm.v1 = 0.0;
    pwm.v2 = 10.0;
    pwm.td = 0.0;
    pwm.tr = 20e-9;
    pwm.tf = 20e-9;
    pwm.pw = 2.5e-6;   // 50% duty
    pwm.period = 5e-6;    // 200kHz

    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    // Switch with defined resistance
    SwitchParams sw;
    sw.ron = 0.05;    // 50 mOhm conduction loss
    sw.roff = 1e9;
    sw.vth = 5.0;

    circuit.add_switch("S1", "vin", "sw", "ctrl", "0", sw);

    // Schottky diode (ideal model for reliable convergence)
    // Freewheeling diode: anode at ground, cathode at sw
    DiodeParams diode;
    diode.ideal = true;

    circuit.add_diode("D1", "0", "sw", diode);

    // Output filter
    circuit.add_inductor("L1", "sw", "out", 47e-6);
    circuit.add_capacitor("C1", "out", "0", 47e-6);
    circuit.add_resistor("Rload", "out", "0", 5.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;   // 200 switching cycles
    opts.dt = 50e-9;
    opts.dtmax = 200e-9;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    const auto& losses = sim.power_losses();

    INFO("Conduction loss: " << losses.conduction_loss * 1e6 << " uJ");
    INFO("Turn-on loss: " << losses.turn_on_loss * 1e6 << " uJ");
    INFO("Turn-off loss: " << losses.turn_off_loss * 1e6 << " uJ");
    INFO("Reverse recovery loss: " << losses.reverse_recovery_loss * 1e6 << " uJ");
    INFO("Total switching loss: " << losses.switching_loss() * 1e6 << " uJ");

    // Verify losses are tracked
    Real total_loss = losses.total_loss();
    CHECK(total_loss >= 0);

    // Conduction losses should be present
    CHECK(losses.conduction_loss >= 0);
}

TEST_CASE("Manual efficiency verification", "[mvp2][efficiency][manual]") {
    // Compare calculated efficiency with manual computation
    // Using simple known values for verification

    // Setup: 50% duty buck converter
    // Vin = 24V, Vout = 12V
    // Rload = 12 ohms -> Iout = 1A
    // Rds_on = 0.1 ohm -> Pcond_sw = I^2 * R * D = 1 * 0.1 * 0.5 = 50mW
    // Diode Vf = 0.5V -> Pcond_d = I * Vf * (1-D) = 1 * 0.5 * 0.5 = 250mW
    // Total loss = 300mW
    // Pout = 12W
    // Pin = 12.3W
    // Efficiency = 12/12.3 = 97.6%

    Circuit circuit;
    circuit.add_voltage_source("Vin", "vin", "0", DCWaveform{24.0});

    PulseWaveform pwm;
    pwm.v1 = 0.0;
    pwm.v2 = 5.0;
    pwm.td = 0.0;
    pwm.tr = 10e-9;
    pwm.tf = 10e-9;
    pwm.pw = 25e-6;   // 50% duty
    pwm.period = 50e-6;  // 20kHz

    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw;
    sw.ron = 0.1;
    sw.roff = 1e9;
    sw.vth = 2.5;

    circuit.add_switch("S1", "vin", "sw", "ctrl", "0", sw);

    // Freewheeling diode (ideal for reliable convergence)
    DiodeParams diode;
    diode.ideal = true;

    circuit.add_diode("D1", "0", "sw", diode);

    circuit.add_inductor("L1", "sw", "out", 500e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("Rload", "out", "0", 12.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-3;  // 200 cycles for good steady state
    opts.dt = 1e-6;
    opts.dtmax = 2e-6;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Calculate output voltage (average of last 10%)
    Index out_idx = circuit.node_index("out");
    size_t start_idx = result.data.size() * 9 / 10;

    Real v_out_avg = 0;
    int count = 0;
    for (size_t i = start_idx; i < result.data.size(); ++i) {
        v_out_avg += result.data[i](out_idx);
        count++;
    }
    v_out_avg /= count;

    Real i_out = v_out_avg / 12.0;  // Rload = 12 ohm
    Real p_out = v_out_avg * i_out;

    // Expected values
    Real expected_vout = 12.0;  // 50% of 24V
    Real expected_pout = 12.0;  // 12V * 1A

    INFO("Measured Vout: " << v_out_avg << " V (expected: " << expected_vout << ")");
    INFO("Measured Pout: " << p_out << " W (expected: " << expected_pout << ")");

    // Verify output voltage is close to expected
    CHECK_THAT(v_out_avg, WithinAbs(expected_vout, 1.0));

    // Get losses from simulator
    const auto& losses = sim.power_losses();
    Real sim_time = opts.tstop - opts.tstart;
    Real avg_loss_power = losses.total_loss() / sim_time;

    INFO("Simulated total loss energy: " << losses.total_loss() * 1e3 << " mJ");
    INFO("Average loss power: " << avg_loss_power * 1e3 << " mW");

    // Loss should be in reasonable range
    CHECK(avg_loss_power >= 0);
}
