#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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

// Skip stress tests in CI environments or with sanitizers
#if PULSIM_SANITIZERS_ENABLED || defined(PULSIM_CI_BUILD)
    #define PULSIM_SKIP_STRESS_TESTS 1
#else
    #define PULSIM_SKIP_STRESS_TESTS 0
#endif

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Stress simulations (large circuits)", "[stress][performance]") {
#if PULSIM_SKIP_STRESS_TESTS
    SKIP("Stress tests skipped in CI or with sanitizers");
#endif

    SECTION("Large resistive ladder DC") {
        const int segments = 2000;
        const Real resistance = 1000.0;
        const Real vin = 1.0;

        Circuit circuit;
        std::vector<Index> nodes;
        nodes.reserve(segments);
        for (int i = 0; i < segments; ++i) {
            nodes.push_back(circuit.add_node("n" + std::to_string(i)));
        }

        circuit.add_voltage_source("Vsrc", nodes[0], Circuit::ground(), vin);
        for (int i = 0; i < segments - 1; ++i) {
            circuit.add_resistor("R" + std::to_string(i), nodes[i], nodes[i + 1], resistance);
        }
        circuit.add_resistor("Rlast", nodes.back(), Circuit::ground(), resistance);

        SimulationOptions opts;
        opts.linear_solver.order = {LinearSolverKind::KLU};
        opts.linear_solver.auto_select = false;
        opts.linear_solver.allow_fallback = false;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();

        Simulator sim(circuit, opts);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto dc = sim.dc_operating_point();
        auto t1 = std::chrono::high_resolution_clock::now();

        REQUIRE(dc.success);

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        INFO("Large ladder DC time: " << ms << " ms (nodes=" << circuit.num_nodes() << ")");

        const auto& sol = dc.newton_result.solution;
        auto expected_at = [&](int idx) {
            return vin * (static_cast<Real>(segments - idx) / static_cast<Real>(segments));
        };

        CHECK(sol[nodes[0]] == Approx(expected_at(0)).margin(1e-4));
        CHECK(sol[nodes[segments / 2]] == Approx(expected_at(segments / 2)).margin(1e-4));
        CHECK(sol[nodes[segments - 1]] == Approx(expected_at(segments - 1)).margin(1e-4));
    }

    SECTION("Large RC ladder transient") {
        const int sections = 400;
        const Real resistance = 1000.0;
        const Real capacitance = 1e-9;
        const Real vin = 1.0;

        Circuit circuit;
        std::vector<Index> nodes;
        nodes.reserve(sections);
        for (int i = 0; i < sections; ++i) {
            nodes.push_back(circuit.add_node("n" + std::to_string(i)));
        }

        circuit.add_voltage_source("Vsrc", nodes[0], Circuit::ground(), vin);
        for (int i = 0; i < sections - 1; ++i) {
            circuit.add_resistor("R" + std::to_string(i), nodes[i], nodes[i + 1], resistance);
        }
        circuit.add_resistor("Rlast", nodes.back(), Circuit::ground(), resistance);
        for (int i = 1; i < sections; ++i) {
            circuit.add_capacitor("C" + std::to_string(i), nodes[i], Circuit::ground(), capacitance);
        }

        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 5e-4;
        opts.dt = 1e-6;
        opts.dt_min = opts.dt;
        opts.dt_max = opts.dt;
        opts.adaptive_timestep = false;
        opts.enable_bdf_order_control = false;
        opts.integrator = Integrator::BDF1;
        opts.linear_solver.order = {LinearSolverKind::KLU};
        opts.linear_solver.auto_select = false;
        opts.linear_solver.allow_fallback = false;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();

        Simulator sim(circuit, opts);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = sim.run_transient();
        auto t1 = std::chrono::high_resolution_clock::now();

        REQUIRE(result.success);
        REQUIRE_FALSE(result.states.empty());

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        INFO("RC ladder transient time: " << ms << " ms (nodes=" << circuit.num_nodes()
             << ", steps=" << result.total_steps << ")");

        const auto& final_state = result.states.back();
        auto expected_at = [&](int idx) {
            return vin * (static_cast<Real>(sections - idx) / static_cast<Real>(sections));
        };

        CHECK(std::isfinite(final_state[nodes[1]]));
        CHECK(std::isfinite(final_state[nodes[5]]));
        CHECK(final_state[nodes[1]] == Approx(expected_at(1)).margin(2e-2));
        CHECK(final_state[nodes[5]] == Approx(expected_at(5)).margin(5e-2));
        CHECK(final_state[nodes.back()] >= 0.0);
        CHECK(final_state[nodes.back()] <= vin);
    }

    SECTION("Buck converter (switching)") {
        const int phases = 2;
        const Real vin = 48.0;
        const Real duty = 0.5;
        const Real frequency = 20000.0;
        const Real period = 1.0 / frequency;

        Circuit circuit;
        auto n_vin = circuit.add_node("vin");
        auto n_out = circuit.add_node("out");
        std::vector<Index> ctrl_nodes;
        std::vector<Index> sw_nodes;
        std::vector<Index> l_nodes;
        std::vector<Index> ind_branches;
        ctrl_nodes.reserve(phases);
        sw_nodes.reserve(phases);
        l_nodes.reserve(phases);
        ind_branches.reserve(phases);

        circuit.add_voltage_source("Vin", n_vin, Circuit::ground(), vin);
        circuit.add_capacitor("Cout", n_out, Circuit::ground(), 100e-6);
        circuit.add_resistor("Rload", n_out, Circuit::ground(), 10.0);

        for (int i = 0; i < phases; ++i) {
            ctrl_nodes.push_back(circuit.add_node("ctrl_" + std::to_string(i)));
            sw_nodes.push_back(circuit.add_node("sw_" + std::to_string(i)));
            l_nodes.push_back(circuit.add_node("l_" + std::to_string(i)));
        }

        for (int i = 0; i < phases; ++i) {
            const auto n_ctrl = ctrl_nodes[i];
            const auto n_sw = sw_nodes[i];
            const auto n_l = l_nodes[i];

            PWMParams pwm;
            pwm.v_high = 5.0;
            pwm.v_low = 0.0;
            pwm.frequency = frequency;
            pwm.duty = duty;
            pwm.phase = 0.0;

            circuit.add_pwm_voltage_source("Vctrl_" + std::to_string(i), n_ctrl, Circuit::ground(), pwm);
            circuit.add_vcswitch("S_" + std::to_string(i), n_ctrl, n_vin, n_sw, 2.5, 200.0, 1e-6);
            circuit.add_diode("D_" + std::to_string(i), Circuit::ground(), n_sw, 200.0, 1e-9);
            circuit.add_resistor("Rsnub_" + std::to_string(i), n_sw, Circuit::ground(), 1e6);
            circuit.add_resistor("Rdcr_" + std::to_string(i), n_sw, n_l, 0.01);
            Index l_branch = circuit.num_nodes() + circuit.num_branches();
            // Higher inductance keeps the buck in CCM for stress validation.
            circuit.add_inductor("L_" + std::to_string(i), n_l, n_out, 1e-3);
            ind_branches.push_back(l_branch);
        }

        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = period * 200.0;
        opts.dt = period / (200.0 * static_cast<Real>(phases));
        opts.dt_min = opts.dt;
        opts.dt_max = opts.dt;
        opts.adaptive_timestep = false;
        opts.enable_bdf_order_control = false;
        opts.integrator = Integrator::BDF1;
        opts.linear_solver.order = {LinearSolverKind::KLU, LinearSolverKind::SparseLU};
        opts.linear_solver.auto_select = false;
        opts.linear_solver.allow_fallback = true;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();
        opts.newton_options.max_iterations = 80;

        Simulator sim(circuit, opts);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = sim.run_transient();
        auto t1 = std::chrono::high_resolution_clock::now();

        REQUIRE(result.success);
        REQUIRE_FALSE(result.states.empty());

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        INFO("Buck time: " << ms << " ms (steps=" << result.total_steps << ")");

        const Real t_end = result.time.back();
        const Real avg_window = 5.0 * period;
        const Real t_start_avg = t_end - avg_window;
        const Real t_start_prev = t_end - 2.0 * avg_window;
        Real sum_last = 0.0;
        Real sum_prev = 0.0;
        Real sum_sw_last = 0.0;
        Real sum_sw_prev = 0.0;
        Real sum_il_last = 0.0;
        Real sum_il_prev = 0.0;
        Real vout_min = std::numeric_limits<Real>::infinity();
        Real vout_max = -std::numeric_limits<Real>::infinity();
        Real vsw_min = std::numeric_limits<Real>::infinity();
        Real vsw_max = -std::numeric_limits<Real>::infinity();
        Real il_min = std::numeric_limits<Real>::infinity();
        Real il_max = -std::numeric_limits<Real>::infinity();
        int diode_on_last = 0;
        int diode_on_prev = 0;
        int count_last = 0;
        int count_prev = 0;
        for (std::size_t i = 0; i < result.time.size(); ++i) {
            if (result.time[i] >= t_start_avg) {
                Real vout = result.states[i][n_out];
                sum_last += vout;
                vout_min = std::min(vout_min, vout);
                vout_max = std::max(vout_max, vout);
                for (auto sw : sw_nodes) {
                    Real vsw = result.states[i][sw];
                    sum_sw_last += vsw;
                    vsw_min = std::min(vsw_min, vsw);
                    vsw_max = std::max(vsw_max, vsw);
                    if (vsw < 0.0) {
                        diode_on_last++;
                    }
                }
                for (auto br : ind_branches) {
                    Real il = result.states[i][br];
                    sum_il_last += il;
                    il_min = std::min(il_min, il);
                    il_max = std::max(il_max, il);
                }
                count_last++;
            } else if (result.time[i] >= t_start_prev) {
                sum_prev += result.states[i][n_out];
                for (auto sw : sw_nodes) {
                    sum_sw_prev += result.states[i][sw];
                    if (result.states[i][sw] < 0.0) {
                        diode_on_prev++;
                    }
                }
                for (auto br : ind_branches) {
                    sum_il_prev += result.states[i][br];
                }
                count_prev++;
            }
        }
        REQUIRE(count_last > 0);
        Real v_avg_last = sum_last / static_cast<Real>(count_last);
        Real v_avg_prev = count_prev > 0 ? (sum_prev / static_cast<Real>(count_prev)) : v_avg_last;
        Real v_sw_avg_last = sum_sw_last / static_cast<Real>(count_last * sw_nodes.size());
        Real v_sw_avg_prev = count_prev > 0
            ? (sum_sw_prev / static_cast<Real>(count_prev * sw_nodes.size()))
            : v_sw_avg_last;
        Real il_avg_last = sum_il_last / static_cast<Real>(count_last * ind_branches.size());
        Real il_avg_prev = count_prev > 0
            ? (sum_il_prev / static_cast<Real>(count_prev * ind_branches.size()))
            : il_avg_last;
        Real diode_duty_last = static_cast<Real>(diode_on_last)
            / static_cast<Real>(count_last * sw_nodes.size());
        Real diode_duty_prev = count_prev > 0
            ? static_cast<Real>(diode_on_prev) / static_cast<Real>(count_prev * sw_nodes.size())
            : diode_duty_last;
        INFO("Vout avg last period: " << v_avg_last << " V, prev: " << v_avg_prev << " V");
        INFO("Vout min/max last period: " << vout_min << " / " << vout_max << " V");

        INFO("Vsw avg last period: " << v_sw_avg_last << " V, prev: " << v_sw_avg_prev << " V");
        INFO("Vsw min/max last period: " << vsw_min << " / " << vsw_max << " V");
        INFO("IL avg last period: " << il_avg_last << " A, prev: " << il_avg_prev << " A");
        INFO("IL min/max last period: " << il_min << " / " << il_max << " A");
        INFO("Diode on duty last period: " << diode_duty_last
             << ", prev: " << diode_duty_prev);

        if (const char* env = std::getenv("PULSIM_STRESS_LOG")) {
            if (env[0] != '\0' && env[0] != '0') {
                std::cout << "buck_summary"
                          << " vout_avg=" << v_avg_last
                          << " vout_min=" << vout_min
                          << " vout_max=" << vout_max
                          << " vsw_avg=" << v_sw_avg_last
                          << " vsw_min=" << vsw_min
                          << " vsw_max=" << vsw_max
                          << " il_avg=" << il_avg_last
                          << " il_min=" << il_min
                          << " il_max=" << il_max
                          << " diode_duty=" << diode_duty_last
                          << " steps=" << result.total_steps
                          << "\\n";
            }
        }

        CHECK(v_avg_last >= vin * 0.2);
        CHECK(v_avg_last <= vin * 0.9);
        CHECK(v_avg_last == Approx(v_sw_avg_last).margin(std::max(1e-3, v_sw_avg_last * 0.05)));
        CHECK(std::abs(v_avg_last - v_avg_prev) <= std::max(1e-3, v_avg_last * 0.03));
    }

    SECTION("Large pathological solve terminates with bounded fallback trace") {
        const int sections = 1200;

        Circuit circuit;
        std::vector<Index> nodes;
        nodes.reserve(sections);
        for (int i = 0; i < sections; ++i) {
            nodes.push_back(circuit.add_node("x" + std::to_string(i)));
        }

        circuit.add_voltage_source("Vsrc", nodes[0], Circuit::ground(), 1.0);
        for (int i = 0; i < sections - 1; ++i) {
            circuit.add_resistor("R" + std::to_string(i), nodes[i], nodes[i + 1], 1e3);
        }
        circuit.add_capacitor("Cend", nodes.back(), Circuit::ground(), 1e-8, 0.0);

        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 1e-4;
        opts.dt = 1e-6;
        opts.dt_min = opts.dt;
        opts.dt_max = opts.dt;
        opts.adaptive_timestep = false;
        opts.enable_bdf_order_control = false;
        opts.max_step_retries = 6;
        opts.fallback_policy.trace_retries = true;
        opts.fallback_policy.enable_transient_gmin = true;
        opts.fallback_policy.gmin_retry_threshold = 1;
        opts.fallback_policy.gmin_initial = 1e-8;
        opts.fallback_policy.gmin_max = 1e-3;
        opts.linear_solver.order = {LinearSolverKind::CG};
        opts.linear_solver.allow_fallback = false;
        opts.linear_solver.auto_select = false;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();

        Simulator sim(circuit, opts);
        Vector x0 = Vector::Zero(circuit.system_size());
        auto result = sim.run_transient(x0);

        REQUIRE_FALSE(result.success);
        REQUIRE_FALSE(result.fallback_trace.empty());
        CHECK(result.fallback_trace.size() <= static_cast<std::size_t>(opts.max_step_retries * 3 + 4));
        CHECK(result.total_steps == 0);
    }
}
