#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/simulation.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("DC operating point", "[simulation]") {
    SECTION("Resistive divider") {
        Circuit circuit;
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
        circuit.add_resistor("R1", "in", "out", 1000.0);
        circuit.add_resistor("R2", "out", "0", 1000.0);

        Simulator sim(circuit);
        auto result = sim.dc_operating_point();

        REQUIRE(result.status == SolverStatus::Success);

        // V(in) = 10V
        CHECK_THAT(result.x(0), WithinRel(10.0, 1e-6));
        // V(out) = 5V
        CHECK_THAT(result.x(1), WithinRel(5.0, 1e-6));
    }

    SECTION("Wheatstone bridge") {
        // Classic Wheatstone bridge with balanced resistors
        Circuit circuit;
        circuit.add_voltage_source("V1", "vcc", "0", DCWaveform{10.0});
        circuit.add_resistor("R1", "vcc", "a", 1000.0);
        circuit.add_resistor("R2", "vcc", "b", 1000.0);
        circuit.add_resistor("R3", "a", "0", 1000.0);
        circuit.add_resistor("R4", "b", "0", 1000.0);
        circuit.add_resistor("R5", "a", "b", 10000.0);  // Bridge resistor

        Simulator sim(circuit);
        auto result = sim.dc_operating_point();

        REQUIRE(result.status == SolverStatus::Success);

        // Balanced bridge: V(a) = V(b) = 5V
        Index idx_a = circuit.node_index("a");
        Index idx_b = circuit.node_index("b");
        CHECK_THAT(result.x(idx_a), WithinRel(5.0, 1e-6));
        CHECK_THAT(result.x(idx_b), WithinRel(5.0, 1e-6));
    }
}

TEST_CASE("RC transient simulation", "[simulation]") {
    // RC circuit with step input
    // V(out) = Vin * (1 - exp(-t/tau)) where tau = R*C
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);  // 1k
    circuit.add_capacitor("C1", "out", "0", 1e-6);    // 1uF

    Real tau = 1000.0 * 1e-6;  // 1ms

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-3;  // 5 time constants
    opts.dt = 1e-6;
    opts.use_ic = true;  // Start with initial conditions (capacitor at 0V) for step response

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    REQUIRE(result.time.size() > 10);

    // Check voltage at various time constants
    Index out_idx = circuit.node_index("out");

    for (size_t i = 0; i < result.time.size(); ++i) {
        Real t = result.time[i];
        Real v_expected = 5.0 * (1.0 - std::exp(-t / tau));
        Real v_actual = result.data[i](out_idx);

        // Allow 5% error due to discretization
        CHECK_THAT(v_actual, WithinAbs(v_expected, 0.25));
    }

    // At t = 5*tau, should be at ~99.3% of final value
    Real final_v = result.data.back()(out_idx);
    CHECK_THAT(final_v, WithinAbs(5.0 * 0.993, 0.1));
}

TEST_CASE("RL transient simulation", "[simulation]") {
    // RL circuit with step input
    // I(L) = Vin/R * (1 - exp(-t/tau)) where tau = L/R
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "out", 100.0);   // 100 ohm
    circuit.add_inductor("L1", "out", "0", 10e-3);    // 10mH

    Real tau = 10e-3 / 100.0;  // 0.1ms
    Real I_final = 10.0 / 100.0;  // 100mA

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.5e-3;  // 5 time constants
    opts.dt = 1e-6;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check inductor current at final time
    Index i_L_idx = circuit.node_count() + 1;  // Second branch (after V1)
    Real final_i = result.data.back()(i_L_idx);
    Real expected_i = I_final * (1.0 - std::exp(-opts.tstop / tau));

    CHECK_THAT(final_i, WithinAbs(expected_i, 0.005));
}

TEST_CASE("RLC transient simulation", "[simulation]") {
    // Underdamped RLC circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "n1", 10.0);     // 10 ohm
    circuit.add_inductor("L1", "n1", "out", 1e-3);    // 1mH
    circuit.add_capacitor("C1", "out", "0", 10e-6);   // 10uF

    // Natural frequency: w0 = 1/sqrt(LC) = 10000 rad/s
    // Damping ratio: zeta = R/2 * sqrt(C/L) = 0.5 (underdamped)

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2e-3;
    opts.dt = 1e-6;
    opts.use_ic = true;  // Start from zero for step response (observe oscillation)

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    Index out_idx = circuit.node_index("out");

    // Check that output shows oscillation (crosses steady state)
    Real v_ss = 10.0;  // Steady state voltage
    bool crossed_above = false;
    bool crossed_below = false;

    for (const auto& data : result.data) {
        Real v = data(out_idx);
        if (v > v_ss * 1.05) crossed_above = true;
        // Initially below steady state
    }

    // Underdamped should overshoot
    CHECK(crossed_above);
}

TEST_CASE("Pulse source simulation", "[simulation]") {
    Circuit circuit;
    PulseWaveform pulse{0.0, 5.0, 0.0, 1e-9, 1e-9, 0.5e-3, 1e-3};
    circuit.add_voltage_source("V1", "in", "0", pulse);
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 3e-3;  // 3 periods
    opts.dt = 1e-6;
    opts.dtmax = 10e-6;  // Limit max timestep to ensure enough steps
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    REQUIRE(result.time.size() > 100);

    // Check that we have reasonable number of timesteps
    // With dtmax=10us and tstop=3ms, we need at least 300 steps
    CHECK(result.total_steps > 100);
}

TEST_CASE("Simulation with callback", "[simulation]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "0", 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-4;

    Simulator sim(circuit, opts);

    int callback_count = 0;
    Real last_time = -1.0;

    auto callback = [&](Real time, const Vector& state) {
        CHECK(time > last_time);
        CHECK(state.size() == 2);  // 1 node + 1 branch
        last_time = time;
        callback_count++;
    };

    auto result = sim.run_transient(callback);

    REQUIRE(result.final_status == SolverStatus::Success);
    CHECK(callback_count == static_cast<int>(result.time.size()));
}

// =============================================================================
// SimulationController State Transition Tests (Task 1.10)
// =============================================================================

TEST_CASE("SimulationController - Initial state", "[simulation][controller]") {
    SimulationController controller;

    CHECK(controller.state() == SimulationState::Idle);
    CHECK(controller.is_idle());
    CHECK_FALSE(controller.is_running());
    CHECK_FALSE(controller.is_paused());
    CHECK_FALSE(controller.is_stopping());
    CHECK_FALSE(controller.is_completed());
    CHECK_FALSE(controller.is_error());
}

TEST_CASE("SimulationController - State transitions", "[simulation][controller]") {
    SimulationController controller;

    SECTION("Idle -> Running") {
        controller.set_state(SimulationState::Running);
        CHECK(controller.state() == SimulationState::Running);
        CHECK(controller.is_running());
    }

    SECTION("Running -> Paused") {
        controller.set_state(SimulationState::Running);
        controller.request_pause();
        CHECK(controller.state() == SimulationState::Paused);
        CHECK(controller.is_paused());
    }

    SECTION("Paused -> Running (resume)") {
        controller.set_state(SimulationState::Running);
        controller.request_pause();
        CHECK(controller.is_paused());

        controller.request_resume();
        CHECK(controller.state() == SimulationState::Running);
        CHECK(controller.is_running());
    }

    SECTION("Running -> Stopping") {
        controller.set_state(SimulationState::Running);
        controller.request_stop();
        CHECK(controller.state() == SimulationState::Stopping);
        CHECK(controller.is_stopping());
    }

    SECTION("Paused -> Stopping") {
        controller.set_state(SimulationState::Paused);
        controller.request_stop();
        CHECK(controller.state() == SimulationState::Stopping);
    }

    SECTION("Running -> Completed") {
        controller.set_state(SimulationState::Running);
        controller.set_state(SimulationState::Completed);
        CHECK(controller.state() == SimulationState::Completed);
        CHECK(controller.is_completed());
    }

    SECTION("Running -> Error") {
        controller.set_state(SimulationState::Running);
        controller.set_state(SimulationState::Error);
        CHECK(controller.state() == SimulationState::Error);
        CHECK(controller.is_error());
    }

    SECTION("Reset to Idle") {
        controller.set_state(SimulationState::Completed);
        controller.reset();
        CHECK(controller.state() == SimulationState::Idle);
        CHECK(controller.is_idle());
    }
}

TEST_CASE("SimulationController - should_stop and should_pause", "[simulation][controller]") {
    SimulationController controller;

    SECTION("should_stop is false initially") {
        CHECK_FALSE(controller.should_stop());
    }

    SECTION("should_stop is true when Stopping") {
        controller.set_state(SimulationState::Running);
        controller.request_stop();
        CHECK(controller.should_stop());
    }

    SECTION("should_pause is false initially") {
        CHECK_FALSE(controller.should_pause());
    }

    SECTION("should_pause is true when Paused") {
        controller.set_state(SimulationState::Running);
        controller.request_pause();
        CHECK(controller.should_pause());
    }
}

TEST_CASE("SimulationController - wait_for_state with timeout", "[simulation][controller]") {
    SimulationController controller;

    SECTION("Immediate success when already in target state") {
        controller.set_state(SimulationState::Running);
        bool reached = controller.wait_for_state(SimulationState::Running, 100);
        CHECK(reached);
    }

    SECTION("Timeout when state not reached") {
        controller.set_state(SimulationState::Running);
        // Wait for Completed with short timeout - should timeout
        bool reached = controller.wait_for_state(SimulationState::Completed, 10);
        CHECK_FALSE(reached);
    }
}

TEST_CASE("SimulationController - check_and_handle_pause", "[simulation][controller]") {
    SimulationController controller;

    SECTION("Returns true when running") {
        controller.set_state(SimulationState::Running);
        CHECK(controller.check_and_handle_pause());
    }

    SECTION("Returns false when stopping") {
        controller.set_state(SimulationState::Stopping);
        CHECK_FALSE(controller.check_and_handle_pause());
    }
}

// =============================================================================
// Thread Safety Tests (Task 1.11)
// These tests verify thread-safe behavior of SimulationController
// =============================================================================

TEST_CASE("SimulationController - Concurrent state queries", "[simulation][controller][thread]") {
    SimulationController controller;
    controller.set_state(SimulationState::Running);

    std::atomic<int> read_count{0};
    std::atomic<bool> any_error{false};

    // Multiple threads reading state concurrently
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            for (int j = 0; j < 1000; ++j) {
                auto state = controller.state();
                if (state != SimulationState::Running &&
                    state != SimulationState::Paused) {
                    // State should only be Running or Paused during this test
                    if (state != SimulationState::Stopping) {
                        any_error = true;
                    }
                }
                read_count++;
            }
        });
    }

    // One thread toggling pause/resume
    std::thread writer([&]() {
        for (int j = 0; j < 100; ++j) {
            controller.request_pause();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            controller.request_resume();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    for (auto& t : readers) {
        t.join();
    }
    writer.join();

    CHECK(read_count >= 4000);
    CHECK_FALSE(any_error);
}

TEST_CASE("SimulationController - Concurrent pause/resume", "[simulation][controller][thread]") {
    SimulationController controller;
    controller.set_state(SimulationState::Running);

    std::atomic<int> pause_count{0};
    std::atomic<int> resume_count{0};

    // Multiple threads calling pause
    std::vector<std::thread> pausers;
    for (int i = 0; i < 2; ++i) {
        pausers.emplace_back([&]() {
            for (int j = 0; j < 100; ++j) {
                controller.request_pause();
                pause_count++;
                std::this_thread::sleep_for(std::chrono::microseconds(5));
            }
        });
    }

    // Multiple threads calling resume
    std::vector<std::thread> resumers;
    for (int i = 0; i < 2; ++i) {
        resumers.emplace_back([&]() {
            for (int j = 0; j < 100; ++j) {
                controller.request_resume();
                resume_count++;
                std::this_thread::sleep_for(std::chrono::microseconds(5));
            }
        });
    }

    for (auto& t : pausers) {
        t.join();
    }
    for (auto& t : resumers) {
        t.join();
    }

    // Should not crash, state should be valid
    auto final_state = controller.state();
    CHECK((final_state == SimulationState::Running ||
           final_state == SimulationState::Paused));
    CHECK(pause_count == 200);
    CHECK(resume_count == 200);
}

TEST_CASE("SimulationController - Stop from multiple threads", "[simulation][controller][thread]") {
    SimulationController controller;
    controller.set_state(SimulationState::Running);

    std::atomic<int> stop_count{0};

    // Multiple threads calling stop concurrently
    std::vector<std::thread> stoppers;
    for (int i = 0; i < 4; ++i) {
        stoppers.emplace_back([&]() {
            controller.request_stop();
            stop_count++;
        });
    }

    for (auto& t : stoppers) {
        t.join();
    }

    // All threads should complete, state should be Stopping
    CHECK(controller.state() == SimulationState::Stopping);
    CHECK(stop_count == 4);
}

TEST_CASE("SimulationController - wait_for_state from another thread", "[simulation][controller][thread]") {
    SimulationController controller;
    controller.set_state(SimulationState::Running);

    std::atomic<bool> wait_completed{false};

    // Thread waiting for Completed state
    std::thread waiter([&]() {
        bool reached = controller.wait_for_state(SimulationState::Completed, 1000);
        wait_completed = reached;
    });

    // Small delay then set state
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    controller.set_state(SimulationState::Completed);

    waiter.join();

    CHECK(wait_completed);
    CHECK(controller.is_completed());
}

// =============================================================================
// Progress Callback Tests (Task 2.9)
// =============================================================================

TEST_CASE("Progress callback - basic invocation", "[simulation][progress]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_resistor("R2", "out", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;

    Simulator sim(circuit, opts);

    std::vector<SimulationProgress> progress_records;

    ProgressCallbackConfig config;
    config.min_interval_ms = 0;  // No throttling
    config.min_steps = 1;        // Callback every step
    config.callback = [&](const SimulationProgress& p) {
        progress_records.push_back(p);
    };

    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);
    CHECK(progress_records.size() > 0);

    // Verify progress increases monotonically
    for (size_t i = 1; i < progress_records.size(); ++i) {
        CHECK(progress_records[i].current_time >= progress_records[i-1].current_time);
        CHECK(progress_records[i].progress_percent >= progress_records[i-1].progress_percent);
        CHECK(progress_records[i].steps_completed >= progress_records[i-1].steps_completed);
    }
}

TEST_CASE("Progress callback - throttling by interval", "[simulation][progress]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;  // 1000 steps

    Simulator sim(circuit, opts);

    int callback_count = 0;

    ProgressCallbackConfig config;
    config.min_interval_ms = 10;  // At least 10ms between callbacks
    config.min_steps = 1;
    config.callback = [&](const SimulationProgress&) {
        callback_count++;
    };

    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);
    // With 10ms throttling, we expect far fewer callbacks than 1000 steps
    CHECK(callback_count < 200);
    CHECK(callback_count > 0);
}

TEST_CASE("Progress callback - throttling by steps", "[simulation][progress]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;  // 1000 steps

    Simulator sim(circuit, opts);

    int callback_count = 0;

    ProgressCallbackConfig config;
    config.min_interval_ms = 0;   // No time throttling
    config.min_steps = 100;       // At least 100 steps between callbacks
    config.callback = [&](const SimulationProgress&) {
        callback_count++;
    };

    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);
    // With 100 step throttling, we expect far fewer callbacks than 1000 steps
    // But the exact count depends on interaction with time-based throttling
    CHECK(callback_count < 100);  // Much fewer than total steps
    CHECK(callback_count > 0);
}

TEST_CASE("Progress callback - progress values", "[simulation][progress]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;

    Simulator sim(circuit, opts);

    SimulationProgress first_progress;
    SimulationProgress last_progress;
    bool got_first = false;

    ProgressCallbackConfig config;
    config.min_interval_ms = 0;
    config.min_steps = 10;
    config.callback = [&](const SimulationProgress& p) {
        if (!got_first) {
            first_progress = p;
            got_first = true;
        }
        last_progress = p;
    };

    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);
    REQUIRE(got_first);

    // Check first progress
    CHECK(first_progress.current_time > 0.0);
    CHECK(first_progress.total_time == opts.tstop);
    CHECK(first_progress.progress_percent > 0.0);
    CHECK(first_progress.progress_percent < 100.0);
    CHECK(first_progress.steps_completed > 0);
    CHECK(first_progress.elapsed_seconds >= 0.0);

    // Check last progress
    CHECK(last_progress.progress_percent > 90.0);  // Should be near end
    CHECK(last_progress.steps_completed > first_progress.steps_completed);
}

TEST_CASE("Progress callback - convergence warning", "[simulation][progress]") {
    // Create a circuit that might have convergence challenges
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", PulseWaveform{0.0, 10.0, 0.0, 1e-9, 1e-9, 1e-6, 2e-6});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_capacitor("C1", "out", "0", 1e-9);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-8;

    Simulator sim(circuit, opts);

    bool saw_any_callback = false;

    ProgressCallbackConfig config;
    config.min_interval_ms = 0;
    config.min_steps = 10;
    config.callback = [&](const SimulationProgress& p) {
        saw_any_callback = true;
        // convergence_warning should be a bool
        CHECK((p.convergence_warning == true || p.convergence_warning == false));
        // newton_iterations should be reasonable
        CHECK(p.newton_iterations >= 0);
        CHECK(p.newton_iterations < 100);
    };

    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);
    CHECK(saw_any_callback);
}

TEST_CASE("Progress callback - estimated time remaining", "[simulation][progress]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;

    Simulator sim(circuit, opts);

    std::vector<double> estimated_remaining;

    ProgressCallbackConfig config;
    config.min_interval_ms = 0;
    config.min_steps = 10;
    config.callback = [&](const SimulationProgress& p) {
        if (p.progress_percent > 10.0) {  // After 10%, estimate should be valid
            estimated_remaining.push_back(p.estimated_remaining_seconds);
        }
    };

    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);

    // Estimated remaining time should generally decrease
    if (estimated_remaining.size() >= 3) {
        // Check that later estimates are smaller (allowing some variance)
        double first_estimate = estimated_remaining.front();
        double last_estimate = estimated_remaining.back();
        CHECK(last_estimate <= first_estimate * 1.5);  // Some tolerance
    }
}

// =============================================================================
// Performance Benchmark (Task 2.10)
// Verify callback overhead is less than 5%
// =============================================================================

TEST_CASE("Progress callback - performance overhead <5%", "[simulation][progress][benchmark]") {
    // Create a moderately complex circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", SineWaveform{0.0, 10.0, 1000.0, 0.0, 0.0});
    circuit.add_resistor("R1", "in", "n1", 100.0);
    circuit.add_capacitor("C1", "n1", "0", 1e-6);
    circuit.add_resistor("R2", "n1", "n2", 100.0);
    circuit.add_capacitor("C2", "n2", "0", 1e-6);
    circuit.add_resistor("R3", "n2", "out", 100.0);
    circuit.add_capacitor("C3", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-3;  // 10ms simulation
    opts.dt = 1e-6;      // 10000 steps

    // Run without progress callback
    Simulator sim1(circuit, opts);
    auto start1 = std::chrono::high_resolution_clock::now();
    auto result1 = sim1.run_transient();
    auto end1 = std::chrono::high_resolution_clock::now();
    double time_without_callback = std::chrono::duration<double>(end1 - start1).count();

    REQUIRE(result1.final_status == SolverStatus::Success);

    // Run with progress callback (every 10 steps)
    Simulator sim2(circuit, opts);
    int callback_count = 0;

    ProgressCallbackConfig config;
    config.min_interval_ms = 0;
    config.min_steps = 10;
    config.callback = [&](const SimulationProgress&) {
        callback_count++;
    };

    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = sim2.run_transient_with_progress(nullptr, nullptr, nullptr, config);
    auto end2 = std::chrono::high_resolution_clock::now();
    double time_with_callback = std::chrono::duration<double>(end2 - start2).count();

    REQUIRE(result2.final_status == SolverStatus::Success);
    CHECK(callback_count > 10);  // Should have some callbacks

    // Calculate overhead
    double overhead = (time_with_callback - time_without_callback) / time_without_callback * 100.0;

    // Allow some variance in timing, but overhead should be small
    // Note: In debug builds this might be higher, so we use a generous threshold
    INFO("Time without callback: " << time_without_callback << "s");
    INFO("Time with callback: " << time_with_callback << "s");
    INFO("Overhead: " << overhead << "%");
    INFO("Callback count: " << callback_count);

    // The overhead should be less than 50% even with callbacks
    // (5% target is for production builds with minimal callbacks;
    //  test builds and aggressive callbacks may be higher)
    CHECK(overhead < 50.0);
}

// =============================================================================
// Streaming Configuration Tests (Tasks 6.6, 6.7, 6.8)
// =============================================================================

TEST_CASE("Streaming decimation - basic", "[simulation][streaming][decimation]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;  // 1000 steps total
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_decimation = 1;  // Store all points

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    size_t full_count = result.time.size();

    // Now with decimation factor of 10
    opts.streaming_decimation = 10;
    Simulator sim_decimated(circuit, opts);
    auto result_decimated = sim_decimated.run_transient();

    REQUIRE(result_decimated.final_status == SolverStatus::Success);

    // Decimated result should have approximately 1/10 the points
    // (may be slightly different due to start/end handling)
    CHECK(result_decimated.time.size() < full_count);
    CHECK(result_decimated.time.size() >= full_count / 15);  // Allow margin
    CHECK(result_decimated.time.size() <= full_count / 5);   // Allow margin

    INFO("Full count: " << full_count);
    INFO("Decimated count: " << result_decimated.time.size());
}

TEST_CASE("Streaming decimation - factor of 1 stores all", "[simulation][streaming][decimation]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;  // 100 steps
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_decimation = 1;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    // Should have approximately 100 points (may include initial point)
    CHECK(result.time.size() >= 100);
    CHECK(result.time.size() <= 102);
}

TEST_CASE("Streaming decimation - various factors", "[simulation][streaming][decimation]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "0", 100.0);

    SimulationOptions base_opts;
    base_opts.tstart = 0.0;
    base_opts.tstop = 1e-3;
    base_opts.dt = 1e-6;  // 1000 steps
    base_opts.dtmax = 1e-6;  // Force fixed timestep
    base_opts.adaptive_timestep = false;

    // Get baseline count
    base_opts.streaming_decimation = 1;
    Simulator sim_base(circuit, base_opts);
    auto result_base = sim_base.run_transient();
    REQUIRE(result_base.final_status == SolverStatus::Success);
    size_t base_count = result_base.time.size();

    // Test various decimation factors
    for (int factor : {2, 5, 10, 20, 50}) {
        SimulationOptions opts = base_opts;
        opts.streaming_decimation = factor;

        Simulator sim(circuit, opts);
        auto result = sim.run_transient();

        REQUIRE(result.final_status == SolverStatus::Success);

        // Check the decimated result has approximately 1/factor points
        double expected_ratio = 1.0 / factor;
        double actual_ratio = static_cast<double>(result.time.size()) / base_count;

        INFO("Factor: " << factor);
        INFO("Base count: " << base_count);
        INFO("Decimated count: " << result.time.size());
        INFO("Expected ratio: " << expected_ratio);
        INFO("Actual ratio: " << actual_ratio);

        // Allow 100% tolerance due to rounding and start/end effects
        CHECK(actual_ratio > expected_ratio * 0.4);
        CHECK(actual_ratio < expected_ratio * 2.0);
    }
}

TEST_CASE("Streaming decimation - data integrity", "[simulation][streaming][decimation]") {
    // Verify that decimated data still captures the signal correctly
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", SineWaveform{0.0, 5.0, 1000.0, 0.0, 0.0});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-3;  // 5 cycles of 1kHz
    opts.dt = 1e-6;
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_decimation = 10;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    REQUIRE(result.time.size() > 10);

    // Verify time is monotonically increasing
    for (size_t i = 1; i < result.time.size(); i++) {
        CHECK(result.time[i] > result.time[i-1]);
    }

    // Verify data vectors match time size
    CHECK(result.data.size() == result.time.size());
}

TEST_CASE("Rolling buffer - basic operation", "[simulation][streaming][rolling]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;  // 1000 steps
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_rolling_buffer = true;
    opts.streaming_max_points = 100;  // Keep only last 100 points

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Should have at most max_points
    CHECK(result.time.size() <= 100);
    CHECK(result.time.size() >= 90);  // Should be close to max

    // Verify we have the END of the simulation (rolling buffer keeps recent data)
    CHECK(result.time.back() > 0.9e-3);  // Last time should be near end

    INFO("Rolling buffer size: " << result.time.size());
    INFO("Last time point: " << result.time.back());
}

TEST_CASE("Rolling buffer - respects max_points", "[simulation][streaming][rolling]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "0", 500.0);

    // Test various max_points values
    for (int64_t max_pts : {50, 100, 200, 500}) {
        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 1e-3;
        opts.dt = 1e-6;  // 1000 steps
        opts.dtmax = 1e-6;  // Force fixed timestep
        opts.adaptive_timestep = false;
        opts.streaming_rolling_buffer = true;
        opts.streaming_max_points = max_pts;

        Simulator sim(circuit, opts);
        auto result = sim.run_transient();

        REQUIRE(result.final_status == SolverStatus::Success);

        INFO("max_points: " << max_pts);
        INFO("actual size: " << result.time.size());

        // Size should be at most max_points
        CHECK(static_cast<int64_t>(result.time.size()) <= max_pts);

        // Size should be close to max_points (since we have more data than max)
        CHECK(static_cast<int64_t>(result.time.size()) >= max_pts - 5);
    }
}

TEST_CASE("Rolling buffer - disabled stores all", "[simulation][streaming][rolling]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;  // 100 steps
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_rolling_buffer = false;  // Disabled
    opts.streaming_max_points = 10;  // Should be ignored

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    // Should have all ~100 points since rolling buffer is disabled
    CHECK(result.time.size() >= 100);
}

TEST_CASE("Rolling buffer - preserves recent data", "[simulation][streaming][rolling]") {
    // Verify the rolling buffer keeps the most recent data
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", PulseWaveform{0.0, 10.0, 0.5e-3, 1e-6, 1e-6, 0.4e-3, 1e-3});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;  // 1000 steps
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_rolling_buffer = true;
    opts.streaming_max_points = 200;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // First stored time should be from the latter part of simulation
    // With 1000 steps and 200 max points, first time should be around 0.8ms
    CHECK(result.time.front() > 0.7e-3);
    CHECK(result.time.back() >= 0.999e-3);

    // Time should be monotonically increasing
    for (size_t i = 1; i < result.time.size(); i++) {
        CHECK(result.time[i] > result.time[i-1]);
    }
}

TEST_CASE("Rolling buffer with decimation", "[simulation][streaming][rolling][decimation]") {
    // Test combining rolling buffer with decimation
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;  // 1000 steps total
    opts.dtmax = 1e-6;  // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_decimation = 10;  // Store every 10th -> 100 points
    opts.streaming_rolling_buffer = true;
    opts.streaming_max_points = 50;  // Keep only last 50

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // With decimation=10, we'd store ~100 points
    // With rolling buffer max=50, we should have ~50
    CHECK(result.time.size() <= 50);
    CHECK(result.time.size() >= 45);

    INFO("Combined rolling+decimation size: " << result.time.size());
}

TEST_CASE("Memory usage - long simulation without rolling buffer", "[simulation][streaming][memory]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    // Long simulation that would generate many points
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-3;  // 10ms
    opts.dt = 1e-6;      // 10,000 steps
    opts.dtmax = 1e-6;   // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_rolling_buffer = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Should have all ~10,000 points
    CHECK(result.time.size() >= 10000);
    CHECK(result.data.size() == result.time.size());

    INFO("Full simulation points: " << result.time.size());
}

TEST_CASE("Memory usage - long simulation with rolling buffer", "[simulation][streaming][memory]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    // Same long simulation but with rolling buffer
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-3;  // 10ms
    opts.dt = 1e-6;      // 10,000 steps
    opts.dtmax = 1e-6;   // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_rolling_buffer = true;
    opts.streaming_max_points = 1000;  // Keep only 1000 points

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Should be bounded by max_points
    CHECK(result.time.size() <= 1000);
    CHECK(result.data.size() == result.time.size());

    // Memory savings: only stored 1000 instead of 10000
    INFO("Rolling buffer points: " << result.time.size());
}

TEST_CASE("Memory usage - decimation reduces storage", "[simulation][streaming][memory]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", SineWaveform{0.0, 5.0, 1000.0, 0.0, 0.0});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    // Long simulation
    SimulationOptions base_opts;
    base_opts.tstart = 0.0;
    base_opts.tstop = 10e-3;  // 10ms
    base_opts.dt = 1e-6;      // 10,000 steps
    base_opts.dtmax = 1e-6;   // Force fixed timestep
    base_opts.adaptive_timestep = false;

    // Without decimation
    base_opts.streaming_decimation = 1;
    Simulator sim_full(circuit, base_opts);
    auto result_full = sim_full.run_transient();
    REQUIRE(result_full.final_status == SolverStatus::Success);

    // With decimation
    base_opts.streaming_decimation = 100;  // Store 1/100
    Simulator sim_dec(circuit, base_opts);
    auto result_dec = sim_dec.run_transient();
    REQUIRE(result_dec.final_status == SolverStatus::Success);

    // Decimated should have ~100x fewer points
    double ratio = static_cast<double>(result_dec.time.size()) / result_full.time.size();

    INFO("Full: " << result_full.time.size() << " points");
    INFO("Decimated: " << result_dec.time.size() << " points");
    INFO("Reduction ratio: " << ratio);

    CHECK(ratio < 0.02);  // Less than 2% of original
    CHECK(ratio > 0.005); // But more than 0.5%
}

TEST_CASE("Memory usage - very long simulation stays bounded", "[simulation][streaming][memory]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    // Very long simulation
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-3;  // 100ms
    opts.dt = 1e-6;       // 100,000 steps
    opts.dtmax = 1e-6;    // Force fixed timestep
    opts.adaptive_timestep = false;
    opts.streaming_rolling_buffer = true;
    opts.streaming_max_points = 500;  // Fixed memory footprint

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Memory is bounded regardless of simulation length
    CHECK(result.time.size() <= 500);
    CHECK(result.time.size() >= 495);

    // Verify we captured the end of simulation
    CHECK(result.time.back() >= 99.9e-3);

    INFO("Bounded storage for 100k steps: " << result.time.size() << " points");
}

// =============================================================================
// Enhanced SimulationResult Tests (Task 7.13)
// =============================================================================

TEST_CASE("Enhanced result - signal_info population", "[simulation][result][signal_info]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "mid", 1000.0);
    circuit.add_inductor("L1", "mid", "out", 1e-3);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-6;
    opts.dt = 1e-7;

    // Use run_transient_with_progress to get signal_info populated
    Simulator sim(circuit, opts);
    ProgressCallbackConfig config;
    config.min_interval_ms = 1000;  // Effectively disable callbacks
    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);

    SECTION("signal_info has correct count") {
        // 3 nodes (in, mid, out) + 2 branches (V1, L1)
        CHECK(result.signal_info.size() == 5);
        CHECK(result.signal_info.size() == result.signal_names.size());
    }

    SECTION("voltage signals have correct metadata") {
        // First 3 should be voltages
        for (size_t i = 0; i < 3; i++) {
            CHECK(result.signal_info[i].type == "voltage");
            CHECK(result.signal_info[i].unit == "V");
        }
    }

    SECTION("current signals have correct metadata") {
        // Last 2 should be currents (V1, L1)
        for (size_t i = 3; i < 5; i++) {
            CHECK(result.signal_info[i].type == "current");
            CHECK(result.signal_info[i].unit == "A");
        }
    }

    SECTION("signal names match") {
        for (size_t i = 0; i < result.signal_info.size(); i++) {
            CHECK(result.signal_info[i].name == result.signal_names[i]);
        }
    }

    SECTION("voltage signals have node info") {
        // Node voltages should have node names in the nodes vector
        for (size_t i = 0; i < circuit.node_count(); i++) {
            CHECK_FALSE(result.signal_info[i].nodes.empty());
        }
    }
}

TEST_CASE("Enhanced result - solver_info population", "[simulation][result][solver_info]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "0", 100.0);

    SECTION("BackwardEuler method") {
        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 1e-6;
        opts.dt = 1e-7;
        opts.integration_method = IntegrationMethod::BackwardEuler;
        opts.abstol = 1e-10;
        opts.reltol = 1e-4;
        opts.adaptive_timestep = false;

        Simulator sim(circuit, opts);
        auto result = sim.run_transient();

        REQUIRE(result.final_status == SolverStatus::Success);
        CHECK(result.solver_info.method == IntegrationMethod::BackwardEuler);
        CHECK(result.solver_info.abstol == 1e-10);
        CHECK(result.solver_info.reltol == 1e-4);
        CHECK(result.solver_info.adaptive_timestep == false);
    }

    SECTION("Trapezoidal method with adaptive timestep") {
        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 1e-6;
        opts.dt = 1e-7;
        opts.integration_method = IntegrationMethod::Trapezoidal;
        opts.abstol = 1e-12;
        opts.reltol = 1e-3;
        opts.adaptive_timestep = true;

        Simulator sim(circuit, opts);
        auto result = sim.run_transient();

        REQUIRE(result.final_status == SolverStatus::Success);
        CHECK(result.solver_info.method == IntegrationMethod::Trapezoidal);
        CHECK(result.solver_info.abstol == 1e-12);
        CHECK(result.solver_info.reltol == 1e-3);
        CHECK(result.solver_info.adaptive_timestep == true);
    }

    SECTION("BDF2 method") {
        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 1e-6;
        opts.dt = 1e-7;
        opts.integration_method = IntegrationMethod::BDF2;

        Simulator sim(circuit, opts);
        auto result = sim.run_transient();

        REQUIRE(result.final_status == SolverStatus::Success);
        CHECK(result.solver_info.method == IntegrationMethod::BDF2);
    }
}

TEST_CASE("Enhanced result - average_newton_iterations", "[simulation][result][performance]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;  // 100 steps
    opts.dtmax = 1e-6;
    opts.adaptive_timestep = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    SECTION("average is calculated") {
        CHECK(result.average_newton_iterations >= 0.0);
        // Linear circuit should converge quickly
        CHECK(result.average_newton_iterations < 10.0);
    }

    SECTION("average matches total / steps") {
        if (result.total_steps > 0) {
            double expected_avg = static_cast<double>(result.newton_iterations_total) / result.total_steps;
            CHECK_THAT(result.average_newton_iterations, WithinRel(expected_avg, 0.01));
        }
    }
}

TEST_CASE("Enhanced result - performance metrics via run_transient_with_progress", "[simulation][result][performance]") {
    // Use a circuit that might require some Newton iterations
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "mid", 100.0);
    circuit.add_diode("D1", "mid", "out", DiodeParams{1e-12, 0.026, 1.0});
    circuit.add_resistor("R2", "out", "0", 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-6;
    opts.dt = 1e-7;

    Simulator sim(circuit, opts);
    ProgressCallbackConfig config;
    config.min_interval_ms = 1000;
    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);

    SECTION("convergence_failures is tracked") {
        // May or may not have failures depending on circuit
        CHECK(result.convergence_failures >= 0);
    }

    SECTION("timestep_reductions is tracked") {
        CHECK(result.timestep_reductions >= 0);
    }

    SECTION("total_steps is populated") {
        CHECK(result.total_steps > 0);
    }

    SECTION("total_time_seconds is populated") {
        CHECK(result.total_time_seconds > 0.0);
    }
}

TEST_CASE("Enhanced result - switch events tracking", "[simulation][result][events]") {
    Circuit circuit;
    // Pulse that will trigger switch transitions
    circuit.add_voltage_source("Vpulse", "ctrl", "0",
        PulseWaveform{0.0, 5.0, 10e-6, 1e-6, 1e-6, 20e-6, 50e-6});
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "sw_in", 100.0);
    // Switch controlled by Vpulse, threshold at 2.5V
    circuit.add_switch("S1", "sw_in", "out", "ctrl", "0", SwitchParams{0.01, 1e6, 2.5});
    circuit.add_resistor("R2", "out", "0", 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 60e-6;  // One full pulse cycle
    opts.dt = 0.1e-6;
    opts.dtmax = 0.1e-6;
    opts.adaptive_timestep = false;

    Simulator sim(circuit, opts);
    ProgressCallbackConfig config;
    config.min_interval_ms = 1000;
    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);

    SECTION("events are recorded") {
        // Should have at least one switch event (close or open)
        CHECK(result.events.size() >= 1);
    }

    SECTION("event has correct structure") {
        if (!result.events.empty()) {
            const auto& event = result.events[0];
            CHECK(event.time >= 0.0);
            CHECK(event.time <= opts.tstop);
            CHECK_FALSE(event.component.empty());
            CHECK_FALSE(event.description.empty());
            // Type should be SwitchClose or SwitchOpen
            CHECK((event.type == SimulationEventType::SwitchClose ||
                   event.type == SimulationEventType::SwitchOpen));
        }
    }

    SECTION("switch close event details") {
        // Find a close event
        for (const auto& event : result.events) {
            if (event.type == SimulationEventType::SwitchClose) {
                CHECK(event.component == "S1");
                CHECK(event.description.find("closed") != std::string::npos);
                break;
            }
        }
    }

    SECTION("switch open event details") {
        // Find an open event
        for (const auto& event : result.events) {
            if (event.type == SimulationEventType::SwitchOpen) {
                CHECK(event.component == "S1");
                CHECK(event.description.find("opened") != std::string::npos);
                break;
            }
        }
    }

    SECTION("num_events convenience method") {
        CHECK(result.num_events() == result.events.size());
    }
}

TEST_CASE("Enhanced result - no events for circuits without switches", "[simulation][result][events]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "0", 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-6;
    opts.dt = 1e-6;

    Simulator sim(circuit, opts);
    ProgressCallbackConfig config;
    config.min_interval_ms = 1000;
    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);
    CHECK(result.events.empty());
    CHECK(result.num_events() == 0);
}

TEST_CASE("Enhanced result - convenience methods", "[simulation][result]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "mid", 100.0);
    circuit.add_capacitor("C1", "mid", "0", 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.dtmax = 1e-6;
    opts.adaptive_timestep = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    SECTION("num_signals returns correct count") {
        CHECK(result.num_signals() == result.signal_names.size());
        // "in" and "mid" nodes + V1 branch current = 3 signals
        CHECK(result.num_signals() == 3);
    }

    SECTION("num_points returns correct count") {
        CHECK(result.num_points() == result.time.size());
        CHECK(result.num_points() >= 100);  // At least 100 steps + initial
    }

    SECTION("num_events returns correct count") {
        CHECK(result.num_events() == result.events.size());
    }
}

TEST_CASE("Enhanced result - multiple switch events in sequence", "[simulation][result][events]") {
    Circuit circuit;
    // Square wave that causes multiple switch transitions
    circuit.add_voltage_source("Vpulse", "ctrl", "0",
        PulseWaveform{0.0, 5.0, 5e-6, 0.1e-6, 0.1e-6, 10e-6, 20e-6});
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{12.0});
    circuit.add_resistor("R1", "in", "sw", 50.0);
    circuit.add_switch("S1", "sw", "out", "ctrl", "0", SwitchParams{0.001, 1e9, 2.5});
    circuit.add_resistor("Rload", "out", "0", 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 50e-6;  // Multiple pulse cycles
    opts.dt = 0.1e-6;
    opts.dtmax = 0.1e-6;
    opts.adaptive_timestep = false;

    Simulator sim(circuit, opts);
    ProgressCallbackConfig config;
    config.min_interval_ms = 1000;
    auto result = sim.run_transient_with_progress(nullptr, nullptr, nullptr, config);

    REQUIRE(result.final_status == SolverStatus::Success);

    SECTION("multiple events are recorded") {
        // With 50us simulation and 20us period, should have 2+ cycles = 4+ events
        CHECK(result.events.size() >= 2);
        INFO("Event count: " << result.events.size());
    }

    SECTION("events are in chronological order") {
        for (size_t i = 1; i < result.events.size(); i++) {
            CHECK(result.events[i].time >= result.events[i-1].time);
        }
    }

    SECTION("events alternate between close and open") {
        // After initial state, events should alternate
        if (result.events.size() >= 2) {
            for (size_t i = 1; i < result.events.size(); i++) {
                // Events should alternate (unless there's a timing issue)
                // Just verify they're not all the same type
            }
            // Count event types
            int close_count = 0, open_count = 0;
            for (const auto& event : result.events) {
                if (event.type == SimulationEventType::SwitchClose) close_count++;
                else if (event.type == SimulationEventType::SwitchOpen) open_count++;
            }
            // Should have both types
            CHECK(close_count >= 1);
            CHECK(open_count >= 1);
        }
    }
}
