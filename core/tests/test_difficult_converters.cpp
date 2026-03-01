/**
 * @file test_difficult_converters.cpp
 * @brief Fast convergence tests for power electronics circuits
 *
 * These tests verify basic operation of converter topologies with
 * minimal simulation time. They check:
 * - Simulation converges without errors
 * - Output voltage is in reasonable range
 * - Switch events are detected
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/simulation.hpp"
#include <cmath>
#include <vector>
#include <chrono>

// Detect if running with sanitizers (ASan, UBSan, etc.)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    #define PULSIM_SANITIZERS_ENABLED 1
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
        #define PULSIM_SANITIZERS_ENABLED 1
    #else
        #define PULSIM_SANITIZERS_ENABLED 0
    #endif
#else
    #define PULSIM_SANITIZERS_ENABLED 0
#endif

// Skip timing checks in CI environments or with sanitizers
#if PULSIM_SANITIZERS_ENABLED || defined(PULSIM_CI_BUILD)
    #define PULSIM_SKIP_TIMING_CHECKS 1
#else
    #define PULSIM_SKIP_TIMING_CHECKS 0
#endif

using namespace pulsim;
using Catch::Matchers::WithinAbs;

// =============================================================================
// Helper: Get average voltage from last portion of simulation
// =============================================================================
Real get_average(const SimulationResult& result, Index node_idx, double fraction = 0.5) {
    size_t start = static_cast<size_t>(result.data.size() * fraction);
    Real sum = 0.0;
    int count = 0;
    for (size_t i = start; i < result.data.size(); ++i) {
        sum += result.data[i](node_idx);
        count++;
    }
    return count > 0 ? sum / count : 0.0;
}

// =============================================================================
// Quick Buck Converter Tests (10 cycles, ~100µs)
// =============================================================================

TEST_CASE("Buck converter - 50% duty quick test", "[converter][buck][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{48.0});

    // 100kHz PWM, 50% duty
    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.5 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "vcc", "sw", "ctrl", "0", sw);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "0", "sw", dp);

    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 10.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;  // 10 cycles only
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();
    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    INFO("Duration: " << ms << " ms, Steps: " << result.data.size());
    REQUIRE(result.final_status == SolverStatus::Success);

    // Just check output is building up (not steady state yet)
    Index out = circuit.node_index("out");
    Real v_out = get_average(result, out);
    INFO("Average Vout: " << v_out);
    CHECK(v_out > 5.0);   // Some output voltage
    CHECK(v_out < 30.0);  // Not overvoltage
}

TEST_CASE("Buck converter - 90% duty quick test", "[converter][buck][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{48.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.9 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "vcc", "sw", "ctrl", "0", sw);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "0", "sw", dp);

    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 10.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
}

TEST_CASE("Buck converter - 10% duty quick test", "[converter][buck][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{48.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.1 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "vcc", "sw", "ctrl", "0", sw);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "0", "sw", dp);

    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 10.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
}

// =============================================================================
// Quick Boost Converter Tests
// =============================================================================

TEST_CASE("Boost converter - 50% duty quick test", "[converter][boost][quick]") {
    // Boost: Vin -> L -> switch node
    //                    |-- D --> Cout -> Rload
    //                    |-- SW -> GND
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vin", "0", DCWaveform{24.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.5 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    circuit.add_inductor("L1", "vin", "sw", 100e-6);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "sw", "0", "ctrl", "0", sw);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "sw", "out", dp);

    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 50.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
}

TEST_CASE("Boost converter - 80% duty quick test", "[converter][boost][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vin", "0", DCWaveform{12.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.8 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    circuit.add_inductor("L1", "vin", "sw", 100e-6);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "sw", "0", "ctrl", "0", sw);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "sw", "out", dp);

    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 100.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
}

// =============================================================================
// Buck-Boost Converter (Inverting topology)
// =============================================================================

TEST_CASE("Buck-Boost converter quick test", "[converter][buckboost][quick]") {
    // Inverting buck-boost: Vin -> SW -> L -> GND
    //                                 D |
    //                                   V
    //                        GND <- C <- out
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vin", "0", DCWaveform{24.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.5 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "vin", "sw", "ctrl", "0", sw);

    circuit.add_inductor("L1", "sw", "0", 100e-6);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "out", "sw", dp);  // Note: reversed for inverting

    circuit.add_capacitor("C1", "0", "out", 100e-6);  // Output referenced to GND
    circuit.add_resistor("R1", "0", "out", 10.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
}

// =============================================================================
// Synchronous Buck (two switches, no diode)
// =============================================================================

TEST_CASE("Synchronous buck quick test", "[converter][syncbuck][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{48.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    Real duty = 0.5;
    Real dead_time = 100e-9;

    // High-side PWM (on when ctrl > threshold)
    PulseWaveform pwm_hi{0.0, 10.0, dead_time, 100e-9, 100e-9, duty * period - dead_time, period};
    circuit.add_voltage_source("Vpwm_hi", "ctrl_hi", "0", pwm_hi);

    // Low-side PWM (complementary, offset by duty)
    PulseWaveform pwm_lo{0.0, 10.0, duty * period + dead_time, 100e-9, 100e-9,
                         (1 - duty) * period - 2 * dead_time, period};
    circuit.add_voltage_source("Vpwm_lo", "ctrl_lo", "0", pwm_lo);

    SwitchParams sw_hi{0.01, 1e9, 5.0, false};
    circuit.add_switch("S_hi", "vcc", "sw", "ctrl_hi", "0", sw_hi);

    SwitchParams sw_lo{0.01, 1e9, 5.0, false};
    circuit.add_switch("S_lo", "sw", "0", "ctrl_lo", "0", sw_lo);

    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 10.0);

    SimulationOptions opts;
    opts.tstop = 100e-6;
    opts.dt = 500e-9;  // Smaller timestep for dead-time
    opts.dtmax = 2e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
}

// =============================================================================
// Minimal PWM Test (no switches)
// =============================================================================

TEST_CASE("DC source RC baseline", "[converter][baseline][quick]") {
    // Baseline test - DC source, R, C only - should be very fast
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstop = 50e-6;
    opts.dt = 1e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();
    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    INFO("Duration: " << ms << " ms, Steps: " << result.total_steps);
    REQUIRE(result.final_status == SolverStatus::Success);
    // Timing logged for informational purposes only (no assertion - CI/sanitizer timing varies)
}

TEST_CASE("Pulse source RC test", "[converter][pulse][quick]") {
    // Test with Pulse source - should still be reasonably fast
    Circuit circuit;
    PulseWaveform pwm{0.0, 5.0, 0.0, 1e-6, 1e-6, 5e-6, 10e-6};  // 100kHz, 50%
    circuit.add_voltage_source("Vpwm", "in", "0", pwm);
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstop = 50e-6;  // 5 cycles
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();
    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    INFO("Duration: " << ms << " ms, Steps: " << result.total_steps);
    REQUIRE(result.final_status == SolverStatus::Success);
    // Timing logged for informational purposes only (no assertion - CI/sanitizer timing varies)
}

// =============================================================================
// Switch Event Detection Test
// =============================================================================

TEST_CASE("Switch event detection", "[converter][events][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{12.0});

    // 50kHz PWM for more events in short time
    Real fsw = 50e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.5 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "vcc", "out", "ctrl", "0", sw);

    circuit.add_resistor("R1", "out", "0", 100.0);
    circuit.add_capacitor("C1", "out", "0", 10e-6);

    SimulationOptions opts;
    opts.tstop = 100e-6;  // 5 PWM cycles
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    std::vector<SwitchEvent> events;
    auto event_cb = [&events](const SwitchEvent& e) {
        events.push_back(e);
    };

    Simulator sim(circuit, opts);
    auto result = sim.run_transient(nullptr, event_cb);

    REQUIRE(result.final_status == SolverStatus::Success);
    INFO("Events detected: " << events.size());
    CHECK(events.size() >= 5);  // At least 5 transitions
}

// =============================================================================
// Convergence Statistics Test
// =============================================================================

TEST_CASE("Simulation convergence statistics", "[converter][stats][quick]") {
    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{48.0});

    Real fsw = 100e3;
    Real period = 1.0 / fsw;
    PulseWaveform pwm{0.0, 10.0, 0.0, 1e-6, 1e-6, 0.5 * period, period};
    circuit.add_voltage_source("Vpwm", "ctrl", "0", pwm);

    SwitchParams sw{0.01, 1e9, 5.0, false};
    circuit.add_switch("S1", "vcc", "sw", "ctrl", "0", sw);

    DiodeParams dp;
    dp.ideal = true;
    circuit.add_diode("D1", "0", "sw", dp);

    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);
    circuit.add_resistor("R1", "out", "0", 10.0);

    SimulationOptions opts;
    opts.tstop = 50e-6;  // 5 cycles
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();
    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    REQUIRE(result.final_status == SolverStatus::Success);

    INFO("Steps: " << result.total_steps);
    INFO("Newton iterations: " << result.newton_iterations_total);
    INFO("Convergence failures: " << result.convergence_failures);
    INFO("Timestep reductions: " << result.timestep_reductions);
    INFO("Duration: " << ms << " ms");

    CHECK(result.total_steps > 10);
    CHECK(result.convergence_failures == 0);
}

// =============================================================================
// H-Bridge Inverter (4 switches)
// AC converter test - generates AC output from DC input
// =============================================================================

TEST_CASE("H-bridge inverter 4 switches", "[converter][hbridge][quick]") {
    // H-bridge topology:
    //     Vdc+
    //      |
    //   S1---S3
    //      |
    //     Load
    //      |
    //   S2---S4
    //      |
    //     GND
    //
    // S1+S4 on = positive output, S2+S3 on = negative output
    // Dead-time between transitions prevents shoot-through

    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{100.0});

    // 10kHz switching frequency for AC generation
    Real fsw = 10e3;
    Real period = 1.0 / fsw;
    Real dead_time = 1e-6;

    // PWM for S1 and S4 (positive half)
    PulseWaveform pwm_pos{0.0, 10.0, dead_time, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    circuit.add_voltage_source("Vpwm_pos", "ctrl_pos", "0", pwm_pos);

    // PWM for S2 and S3 (negative half) - inverted timing
    PulseWaveform pwm_neg{0.0, 10.0, 0.5 * period + dead_time, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    circuit.add_voltage_source("Vpwm_neg", "ctrl_neg", "0", pwm_neg);

    SwitchParams sw{0.01, 1e9, 5.0, false};

    // High-side switches
    circuit.add_switch("S1", "vcc", "out_pos", "ctrl_pos", "0", sw);
    circuit.add_switch("S3", "vcc", "out_neg", "ctrl_neg", "0", sw);

    // Low-side switches
    circuit.add_switch("S2", "out_pos", "0", "ctrl_neg", "0", sw);
    circuit.add_switch("S4", "out_neg", "0", "ctrl_pos", "0", sw);

    // Load between output nodes (RL load for inductive motor-like behavior)
    circuit.add_resistor("Rload", "out_pos", "out_neg", 10.0);
    circuit.add_inductor("Lload", "out_neg", "mid", 1e-3);
    circuit.add_resistor("Rmid", "mid", "out_pos", 1.0);  // Small series resistance

    // Snubber capacitors for each switch leg
    circuit.add_capacitor("C1", "out_pos", "0", 100e-9);
    circuit.add_capacitor("C2", "out_neg", "0", 100e-9);

    SimulationOptions opts;
    opts.tstop = 500e-6;  // 5 switching cycles
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();
    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Get output voltage across load
    Index n_pos = circuit.node_index("out_pos");
    Index n_neg = circuit.node_index("out_neg");

    // Check voltage swings (should see both positive and negative values)
    Real v_max = -1e9, v_min = 1e9;
    for (size_t i = result.data.size() / 2; i < result.data.size(); ++i) {
        Real v_out = result.data[i](n_pos) - result.data[i](n_neg);
        v_max = std::max(v_max, v_out);
        v_min = std::min(v_min, v_out);
    }

    INFO("Duration: " << ms << " ms, Steps: " << result.total_steps);
    INFO("V_max: " << v_max << "V, V_min: " << v_min << "V");

    // Output should swing significantly in both directions
    CHECK(v_max > 20.0);
    CHECK(v_min < -20.0);
    // Timing logged for informational purposes only (no assertion - CI/sanitizer timing varies)
}

// =============================================================================
// 3-Phase Inverter (6 switches)
// AC converter for 3-phase motor drives
// =============================================================================

TEST_CASE("3-phase inverter 6 switches", "[converter][3phase][quick]") {
#if PULSIM_SKIP_TIMING_CHECKS
    SKIP("Heavy simulation skipped in CI/sanitizer builds (timeout risk)");
#endif
    // 3-phase inverter topology (each phase has 2 switches):
    //     Vdc+
    //      |
    //   S1--S3--S5 (high-side)
    //      |   |   |
    //     PhA PhB PhC
    //      |   |   |
    //   S2--S4--S6 (low-side)
    //      |
    //     GND
    //
    // 120° phase shift between phases for balanced 3-phase output

    Circuit circuit;
    circuit.add_voltage_source("Vdc", "vcc", "0", DCWaveform{300.0});

    // 10kHz switching frequency
    Real fsw = 10e3;
    Real period = 1.0 / fsw;
    Real dead_time = 1e-6;

    // Phase A: 0° phase shift
    PulseWaveform pwm_a_hi{0.0, 10.0, dead_time, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    PulseWaveform pwm_a_lo{0.0, 10.0, 0.5 * period + dead_time, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    circuit.add_voltage_source("VpwmA_hi", "ctrl_a_hi", "0", pwm_a_hi);
    circuit.add_voltage_source("VpwmA_lo", "ctrl_a_lo", "0", pwm_a_lo);

    // Phase B: 120° phase shift (1/3 period delay)
    Real delay_b = period / 3.0;
    PulseWaveform pwm_b_hi{0.0, 10.0, dead_time + delay_b, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    PulseWaveform pwm_b_lo{0.0, 10.0, 0.5 * period + dead_time + delay_b, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    circuit.add_voltage_source("VpwmB_hi", "ctrl_b_hi", "0", pwm_b_hi);
    circuit.add_voltage_source("VpwmB_lo", "ctrl_b_lo", "0", pwm_b_lo);

    // Phase C: 240° phase shift (2/3 period delay)
    Real delay_c = 2.0 * period / 3.0;
    PulseWaveform pwm_c_hi{0.0, 10.0, dead_time + delay_c, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    PulseWaveform pwm_c_lo{0.0, 10.0, 0.5 * period + dead_time + delay_c, 1e-6, 1e-6, 0.5 * period - dead_time, period};
    circuit.add_voltage_source("VpwmC_hi", "ctrl_c_hi", "0", pwm_c_hi);
    circuit.add_voltage_source("VpwmC_lo", "ctrl_c_lo", "0", pwm_c_lo);

    SwitchParams sw{0.01, 1e9, 5.0, false};

    // Phase A switches
    circuit.add_switch("S1", "vcc", "phase_a", "ctrl_a_hi", "0", sw);
    circuit.add_switch("S2", "phase_a", "0", "ctrl_a_lo", "0", sw);

    // Phase B switches
    circuit.add_switch("S3", "vcc", "phase_b", "ctrl_b_hi", "0", sw);
    circuit.add_switch("S4", "phase_b", "0", "ctrl_b_lo", "0", sw);

    // Phase C switches
    circuit.add_switch("S5", "vcc", "phase_c", "ctrl_c_hi", "0", sw);
    circuit.add_switch("S6", "phase_c", "0", "ctrl_c_lo", "0", sw);

    // Y-connected RL load (motor equivalent)
    circuit.add_resistor("Ra", "phase_a", "neutral", 10.0);
    circuit.add_inductor("La", "neutral", "la_out", 5e-3);
    circuit.add_resistor("Ra2", "la_out", "phase_a", 0.1);

    circuit.add_resistor("Rb", "phase_b", "neutral", 10.0);
    circuit.add_inductor("Lb", "neutral", "lb_out", 5e-3);
    circuit.add_resistor("Rb2", "lb_out", "phase_b", 0.1);

    circuit.add_resistor("Rc", "phase_c", "neutral", 10.0);
    circuit.add_inductor("Lc", "neutral", "lc_out", 5e-3);
    circuit.add_resistor("Rc2", "lc_out", "phase_c", 0.1);

    // Output filter capacitors
    circuit.add_capacitor("Ca", "phase_a", "0", 100e-9);
    circuit.add_capacitor("Cb", "phase_b", "0", 100e-9);
    circuit.add_capacitor("Cc", "phase_c", "0", 100e-9);

    SimulationOptions opts;
    opts.tstop = 500e-6;  // 5 switching cycles
    opts.dt = 1e-6;
    opts.dtmax = 5e-6;
    opts.use_ic = true;
    opts.abstol = 1e-4;
    opts.reltol = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();
    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Get phase voltages
    Index n_a = circuit.node_index("phase_a");
    Index n_b = circuit.node_index("phase_b");
    Index n_c = circuit.node_index("phase_c");

    // Check all phases are active
    Real va_max = -1e9, va_min = 1e9;
    Real vb_max = -1e9, vb_min = 1e9;
    Real vc_max = -1e9, vc_min = 1e9;

    for (size_t i = result.data.size() / 2; i < result.data.size(); ++i) {
        va_max = std::max(va_max, result.data[i](n_a));
        va_min = std::min(va_min, result.data[i](n_a));
        vb_max = std::max(vb_max, result.data[i](n_b));
        vb_min = std::min(vb_min, result.data[i](n_b));
        vc_max = std::max(vc_max, result.data[i](n_c));
        vc_min = std::min(vc_min, result.data[i](n_c));
    }

    INFO("Duration: " << ms << " ms, Steps: " << result.total_steps);
    INFO("Phase A: " << va_min << "V to " << va_max << "V");
    INFO("Phase B: " << vb_min << "V to " << vb_max << "V");
    INFO("Phase C: " << vc_min << "V to " << vc_max << "V");

    // All phases should have significant voltage swings
    CHECK(va_max > 50.0);
    CHECK(vb_max > 50.0);
    CHECK(vc_max > 50.0);
    // Timing logged for informational purposes only (no assertion - CI/sanitizer timing varies)
}
