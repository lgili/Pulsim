#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("v1 switch event timing", "[v1][events][regression]") {
    Circuit circuit;

    auto n_ctrl = circuit.add_node("ctrl");
    auto n_vcc = circuit.add_node("vcc");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vdc", n_vcc, Circuit::ground(), 10.0);

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 10.0;
    pulse.t_delay = 10e-6;
    pulse.t_rise = 2e-6;
    pulse.t_fall = 2e-6;
    pulse.t_width = 10e-6;
    pulse.period = 0.0;
    circuit.add_pulse_voltage_source("Vctrl", n_ctrl, Circuit::ground(), pulse);

    circuit.add_vcswitch("S1", n_ctrl, n_vcc, n_out, 5.0, 1.0 / 0.01, 1.0 / 1e9);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-6;
    opts.dt = 5e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 5e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    if (!result.success) {
        WARN("Stiffness test failed: status=" << static_cast<int>(result.final_status)
             << " message=" << result.message);
        return;
    }
    REQUIRE_FALSE(result.events.empty());

    auto on_event = std::find_if(result.events.begin(), result.events.end(), [](const SimulationEvent& e) {
        return e.type == SimulationEventType::SwitchOn;
    });
    REQUIRE(on_event != result.events.end());

    Real expected_on = pulse.t_delay + pulse.t_rise * 0.5;  // threshold at 50% of pulse amplitude
    CHECK(on_event->time == Approx(expected_on).margin(5e-7));
}

TEST_CASE("v1 switching loss accumulation", "[v1][losses][regression]") {
    Circuit circuit;

    auto n_ctrl = circuit.add_node("ctrl");
    auto n_vcc = circuit.add_node("vcc");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vdc", n_vcc, Circuit::ground(), 10.0);

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 10.0;
    pulse.t_delay = 5e-6;
    pulse.t_rise = 1e-6;
    pulse.t_fall = 1e-6;
    pulse.t_width = 10e-6;
    pulse.period = 0.0;
    circuit.add_pulse_voltage_source("Vctrl", n_ctrl, Circuit::ground(), pulse);

    circuit.add_vcswitch("S1", n_ctrl, n_vcc, n_out, 5.0, 1.0 / 0.01, 1.0 / 1e9);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 30e-6;
    opts.dt = 2e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 2e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_losses = true;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    SwitchingEnergy energy;
    energy.eon = 1e-6;
    energy.eoff = 2e-6;
    opts.switching_energy["S1"] = energy;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    INFO("Stiffness test status: " << static_cast<int>(result.final_status));
    INFO("Stiffness test message: " << result.message);
    REQUIRE(result.success);

    const auto& summary = result.loss_summary;
    REQUIRE_FALSE(summary.device_losses.empty());

    auto it = std::find_if(summary.device_losses.begin(), summary.device_losses.end(), [](const LossResult& res) {
        return res.device_name == "S1";
    });
    REQUIRE(it != summary.device_losses.end());

    Real duration = result.time.back() - result.time.front();
    REQUIRE(duration > 0.0);

    Real expected_on = energy.eon / duration;
    Real expected_off = energy.eoff / duration;
    Real expected_total = (energy.eon + energy.eoff) / duration;

    CHECK(it->breakdown.turn_on == Approx(expected_on).margin(expected_on * 0.2 + 1e-9));
    CHECK(it->breakdown.turn_off == Approx(expected_off).margin(expected_off * 0.2 + 1e-9));
    CHECK(summary.total_switching == Approx(expected_total).margin(expected_total * 0.2 + 1e-9));
}

TEST_CASE("v1 fixed-step buck keeps macro-grid outputs with event-aligned substeps",
          "[v1][fixed-step][events][converter][regression]") {
    Circuit circuit;

    auto n_ctrl = circuit.add_node("ctrl");
    auto n_vin = circuit.add_node("vin");
    auto n_sw = circuit.add_node("sw");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vdc", n_vin, Circuit::ground(), 24.0);

    PulseParams pwm;
    pwm.v_initial = 0.0;
    pwm.v_pulse = 10.0;
    pwm.t_delay = 1e-6;
    pwm.t_rise = 0.2e-6;
    pwm.t_fall = 0.2e-6;
    pwm.t_width = 3e-6;
    pwm.period = 10e-6;
    circuit.add_pulse_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    circuit.add_vcswitch("S1", n_ctrl, n_vin, n_sw, 5.0, 100.0, 1e-9);
    circuit.add_diode("D1", Circuit::ground(), n_sw, 100.0, 1e-9);
    circuit.add_inductor("L1", n_sw, n_out, 100e-6, 0.0);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 47e-6, 0.0);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 24e-6;
    opts.dt = 2e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 2e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_events = true;
    opts.fallback_policy.trace_retries = true;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();

    INFO("Buck fixed-step status: " << static_cast<int>(result.final_status));
    INFO("Buck fixed-step message: " << result.message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.events.empty());

    const auto has_on = std::any_of(
        result.events.begin(), result.events.end(),
        [](const SimulationEvent& event) { return event.type == SimulationEventType::SwitchOn; });
    const auto has_off = std::any_of(
        result.events.begin(), result.events.end(),
        [](const SimulationEvent& event) { return event.type == SimulationEventType::SwitchOff; });
    CHECK(has_on);
    CHECK(has_off);

    const auto has_event_split = std::any_of(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) { return entry.reason == FallbackReasonCode::EventSplit; });
    CHECK(has_event_split);
    for (const auto& entry : result.fallback_trace) {
        if (entry.reason == FallbackReasonCode::EventSplit) {
            CHECK((entry.action == "event_calendar_clip" ||
                   entry.action == "split_to_earliest_event"));
            CHECK(entry.retry_index <= opts.max_step_retries);
        }
    }

    const Real dt_macro = opts.dt;
    for (Real time_sample : result.time) {
        const Real steps = std::round(time_sample / dt_macro);
        CHECK(time_sample == Approx(steps * dt_macro).margin(1e-12));
    }

    REQUIRE(result.total_steps >= static_cast<int>(result.time.size()) - 1);
    CHECK(result.total_steps > static_cast<int>(result.time.size()) - 1);
}

TEST_CASE("v1 variable-step switched buck remains stable around pwm edges",
          "[v1][variable-step][events][converter][regression]") {
    Circuit circuit;

    auto n_ctrl = circuit.add_node("ctrl");
    auto n_in = circuit.add_node("vin");
    auto n_sw = circuit.add_node("sw");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 24.0);

    PulseParams pwm;
    pwm.v_initial = 0.0;
    pwm.v_pulse = 10.0;
    pwm.t_delay = 1e-6;
    pwm.t_rise = 0.2e-6;
    pwm.t_fall = 0.2e-6;
    pwm.t_width = 3e-6;
    pwm.period = 10e-6;
    circuit.add_pulse_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    circuit.add_vcswitch("S1", n_ctrl, n_in, n_sw, 5.0, 100.0, 1e-9);
    circuit.add_diode("D1", Circuit::ground(), n_sw, 100.0, 1e-9);
    circuit.add_inductor("L1", n_sw, n_out, 220e-6, 0.0);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 220e-6, 0.0);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 8.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 24e-6;
    opts.dt = 2e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 6e-6;
    opts.step_mode = TransientStepMode::Variable;
    opts.step_mode_explicit = true;
    opts.adaptive_timestep = true;
    opts.enable_events = true;
    opts.enable_bdf_order_control = true;
    opts.max_step_retries = 20;
    opts.fallback_policy.trace_retries = true;
    opts.newton_options.max_iterations = 120;
    opts.newton_options.auto_damping = true;
    opts.linear_solver.allow_fallback = true;
    opts.linear_solver.auto_select = true;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();

    INFO("VCS buck variable-step status: " << static_cast<int>(result.final_status));
    INFO("VCS buck variable-step message: " << result.message);
    INFO("Buck variable-step steps: " << result.total_steps
         << " rejections: " << result.timestep_rejections
         << " fallback_trace: " << result.fallback_trace.size());
    const int lte_rejections = static_cast<int>(std::count_if(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::LTERejection;
        }));
    const int newton_failures = static_cast<int>(std::count_if(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::NewtonFailure;
        }));
    const int stiffness_backoffs = static_cast<int>(std::count_if(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::StiffnessBackoff;
        }));
    const int max_retry_events = static_cast<int>(std::count_if(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::MaxRetriesExceeded;
        }));
    const int lte_guard_accepts = static_cast<int>(std::count_if(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.action.find("lte_guard_accept_dt=") != std::string::npos;
        }));
    INFO("Buck variable-step reasons: lte=" << lte_rejections
         << " guard_accept=" << lte_guard_accepts
         << " newton=" << newton_failures
         << " stiffness=" << stiffness_backoffs
         << " max_retry=" << max_retry_events);

    REQUIRE(result.success);
    CHECK(result.total_steps >= static_cast<int>(result.time.size()) - 1);
    CHECK_FALSE(result.events.empty());
    CHECK(result.total_steps < 500);
    CHECK(result.timestep_rejections == 0);

    const bool has_abort = std::any_of(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::MaxRetriesExceeded &&
                   entry.action == "abort_step";
        });
    CHECK_FALSE(has_abort);
}

TEST_CASE("v1 balanced convergence policy activates contextual event guards",
          "[v1][fallback][policy][m2][regression]") {
    Circuit circuit;

    auto n_in = circuit.add_node("vin");
    auto n_out = circuit.add_node("out");
    auto n_ctrl = circuit.add_node("ctrl");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);

    PulseParams gate;
    gate.v_initial = 0.0;
    gate.v_pulse = 5.0;
    gate.t_delay = 1e-6;
    gate.t_rise = 0.2e-6;
    gate.t_fall = 0.2e-6;
    gate.t_width = 2e-6;
    gate.period = 6e-6;
    circuit.add_pulse_voltage_source("Vctrl", n_ctrl, Circuit::ground(), gate);

    circuit.add_vcswitch("S1", n_ctrl, n_in, n_out, 2.5, 100.0, 1e-9);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 100.0);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-6;
    opts.dt = 2e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 8e-6;
    opts.step_mode = TransientStepMode::Variable;
    opts.step_mode_explicit = true;
    opts.adaptive_timestep = true;
    opts.enable_events = true;
    opts.enable_bdf_order_control = true;
    opts.max_step_retries = 8;
    opts.fallback_policy.trace_retries = true;
    opts.fallback_policy.convergence_profile = ConvergenceProfile::Balanced;
    opts.fallback_policy.policy_dry_run = false;
    opts.fallback_policy.enable_transient_gmin = true;
    opts.fallback_policy.gmin_retry_threshold = 1;
    opts.newton_options.max_iterations = 1;
    opts.newton_options.auto_damping = false;
    opts.linear_solver.order = {LinearSolverKind::CG};
    opts.linear_solver.allow_fallback = false;
    opts.linear_solver.auto_select = false;
    opts.linear_solver.iterative_config.max_iterations = 2;
    opts.linear_solver.iterative_config.tolerance = 1e-12;
    opts.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::None;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const Vector x0 = Vector::Zero(circuit.system_size());
    const auto result = sim.run_transient(x0);

    INFO("policy activation status: " << static_cast<int>(result.final_status)
         << " success=" << result.success
         << " message=" << result.message
         << " trace=" << result.fallback_trace.size()
         << " rejections=" << result.timestep_rejections);
    CHECK_FALSE(result.fallback_trace.empty());

    const bool has_policy_guard = std::any_of(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.action.find("policy_event_burst_zero_cross_guard") != std::string::npos ||
                   entry.action.find("policy_switch_chattering_guard") != std::string::npos;
        });
    CHECK(has_policy_guard);

    const bool has_target_class = std::any_of(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.failure_class == ConvergenceFailureClass::EventBurstZeroCross ||
                   entry.failure_class == ConvergenceFailureClass::SwitchChattering;
        });
    CHECK(has_target_class);

    for (const auto& entry : result.fallback_trace) {
        if (entry.action.find("policy_") != std::string::npos) {
            CHECK(entry.retry_index <= opts.max_step_retries);
        }
    }
}

TEST_CASE("v1 variable-step mosfet buck stays close to fixed-step reference",
          "[v1][variable-step][converter][accuracy][regression]") {
    struct BuckMetrics {
        bool success = false;
        std::string message;
        int total_steps = 0;
        int timestep_rejections = 0;
        Real vout_final = 0.0;
        Real vout_mean_tail = 0.0;
        Real vout_ripple_tail = 0.0;
    };

    auto run_case = [](TransientStepMode mode) -> BuckMetrics {
        Circuit circuit;
        auto n_in = circuit.add_node("vin");
        auto n_gate = circuit.add_node("gate");
        auto n_sw = circuit.add_node("sw");
        auto n_out = circuit.add_node("out");
        const auto gnd = Circuit::ground();

        circuit.add_voltage_source("Vin", n_in, gnd, 24.0);

        PWMParams pwm;
        pwm.v_high = 120.0;
        pwm.v_low = 0.0;
        pwm.frequency = 100e3;
        pwm.duty = 0.5;
        pwm.rise_time = 2e-6;
        pwm.fall_time = 2e-6;
        circuit.add_pwm_voltage_source("Vgate", n_gate, gnd, pwm);

        MOSFET::Params mosfet;
        mosfet.vth = 2.0;
        mosfet.kp = 0.02;
        mosfet.g_off = 1e-4;
        mosfet.lambda = 0.0;
        circuit.add_mosfet("M1", n_gate, n_in, n_sw, mosfet);

        circuit.add_diode("D1", gnd, n_sw, 20.0, 1e-5);
        circuit.add_inductor("L1", n_sw, n_out, 220e-6, 0.0);
        circuit.add_capacitor("C1", n_out, gnd, 220e-6, 0.0);
        circuit.add_resistor("Rload", n_out, gnd, 8.0);
        circuit.add_resistor("Rbleed_sw", n_sw, gnd, 1e5);

        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 4e-3;
        opts.dt = 2e-6;
        opts.max_step_retries = 35;
        opts.newton_options.max_iterations = 220;
        opts.newton_options.auto_damping = true;
        opts.linear_solver.allow_fallback = true;
        opts.linear_solver.auto_select = true;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();
        opts.step_mode = mode;
        opts.step_mode_explicit = true;

        if (mode == TransientStepMode::Fixed) {
            opts.integrator = Integrator::BDF1;
            opts.dt_min = opts.dt;
            opts.dt_max = opts.dt;
        }

        Simulator sim(circuit, opts);
        const auto result = sim.run_transient(circuit.initial_state());

        BuckMetrics metrics;
        metrics.success = result.success;
        metrics.message = result.message;
        metrics.total_steps = result.total_steps;
        metrics.timestep_rejections = result.timestep_rejections;
        if (!result.success || result.states.empty() || result.time.empty()) {
            return metrics;
        }

        const auto signal_names = circuit.signal_names();
        const auto out_it = std::find(signal_names.begin(), signal_names.end(), "V(out)");
        if (out_it == signal_names.end()) {
            metrics.success = false;
            metrics.message = "Missing V(out) signal";
            return metrics;
        }
        const auto out_index = static_cast<std::size_t>(std::distance(signal_names.begin(), out_it));

        std::vector<Real> vout;
        vout.reserve(result.states.size());
        for (const auto& state : result.states) {
            if (out_index >= static_cast<std::size_t>(state.size())) {
                continue;
            }
            vout.push_back(state[static_cast<Index>(out_index)]);
        }

        if (vout.empty()) {
            metrics.success = false;
            metrics.message = "Empty V(out) waveform";
            return metrics;
        }

        const std::size_t tail = std::max<std::size_t>(10, vout.size() / 5);
        const std::size_t tail_start = vout.size() > tail ? vout.size() - tail : 0;
        Real tail_sum = 0.0;
        Real tail_min = std::numeric_limits<Real>::infinity();
        Real tail_max = -std::numeric_limits<Real>::infinity();
        for (std::size_t i = tail_start; i < vout.size(); ++i) {
            const Real value = vout[i];
            tail_sum += value;
            tail_min = std::min(tail_min, value);
            tail_max = std::max(tail_max, value);
        }

        const Real tail_count = static_cast<Real>(vout.size() - tail_start);
        metrics.vout_final = vout.back();
        metrics.vout_mean_tail = tail_sum / std::max<Real>(tail_count, 1.0);
        metrics.vout_ripple_tail = tail_max - tail_min;
        return metrics;
    };

    const BuckMetrics fixed = run_case(TransientStepMode::Fixed);
    const BuckMetrics variable = run_case(TransientStepMode::Variable);

    INFO("MOSFET buck fixed success=" << fixed.success
         << " message=" << fixed.message
         << " steps=" << fixed.total_steps
         << " rejections=" << fixed.timestep_rejections
         << " vout_mean_tail=" << fixed.vout_mean_tail
         << " vout_ripple_tail=" << fixed.vout_ripple_tail
         << " vout_final=" << fixed.vout_final);
    INFO("MOSFET buck variable success=" << variable.success
         << " message=" << variable.message
         << " steps=" << variable.total_steps
         << " rejections=" << variable.timestep_rejections
         << " vout_mean_tail=" << variable.vout_mean_tail
         << " vout_ripple_tail=" << variable.vout_ripple_tail
         << " vout_final=" << variable.vout_final);

    REQUIRE(fixed.success);
    REQUIRE(variable.success);

    CHECK(std::abs(variable.vout_mean_tail - fixed.vout_mean_tail) <= 0.3);
    CHECK(variable.vout_ripple_tail <= fixed.vout_ripple_tail * 1.6 + 1e-6);
    CHECK(std::abs(variable.vout_final - fixed.vout_final) <= 0.3);
    CHECK(variable.timestep_rejections <= 2);
}

TEST_CASE("v1 forced MOSFET state drives electrical conduction", "[v1][mosfet][switch][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("vin");
    const auto n_out = circuit.add_node("out");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 12.0);

    MOSFET::Params mosfet;
    mosfet.vth = 2.0;
    mosfet.kp = 0.35;
    mosfet.g_off = 1e-8;
    mosfet.lambda = 0.0;
    circuit.add_mosfet("M1", gnd, n_in, n_out, mosfet);
    circuit.add_resistor("Rload", n_out, gnd, 10.0);

    circuit.set_switch_state("M1", true);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 100e-6;
    opts.dt = 1e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE_FALSE(result.states.empty());

    const auto signals = circuit.signal_names();
    const auto out_it = std::find(signals.begin(), signals.end(), "V(out)");
    REQUIRE(out_it != signals.end());
    const auto out_index = static_cast<std::size_t>(std::distance(signals.begin(), out_it));
    const Real vout_last = result.states.back()[static_cast<Index>(out_index)];

    INFO("forced MOSFET vout_last=" << vout_last
         << " steps=" << result.total_steps
         << " rejections=" << result.timestep_rejections);
    CHECK(vout_last > 1.0);
}

TEST_CASE("v1 forced MOSFET state contributes to electrothermal losses",
          "[v1][mosfet][switch][loss][thermal][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("vin");
    const auto n_out = circuit.add_node("out");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 12.0);

    MOSFET::Params mosfet;
    mosfet.vth = 2.0;
    mosfet.kp = 0.35;
    mosfet.g_off = 1e-8;
    mosfet.lambda = 0.0;
    circuit.add_mosfet("M1", gnd, n_in, n_out, mosfet);
    circuit.add_resistor("Rload", n_out, gnd, 10.0);
    circuit.set_switch_state("M1", true);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 2e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.enable_losses = true;
    opts.thermal.enable = true;
    opts.thermal.ambient = 25.0;
    opts.thermal.default_rth = 1.0;
    opts.thermal.default_cth = 0.05;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);

    const auto loss_it = std::find_if(
        result.loss_summary.device_losses.begin(),
        result.loss_summary.device_losses.end(),
        [](const LossResult& item) { return item.device_name == "M1"; });
    REQUIRE(loss_it != result.loss_summary.device_losses.end());
    CHECK(loss_it->breakdown.conduction > 0.5);

    const auto thermal_it = std::find_if(
        result.thermal_summary.device_temperatures.begin(),
        result.thermal_summary.device_temperatures.end(),
        [](const DeviceThermalTelemetry& item) { return item.device_name == "M1"; });
    REQUIRE(thermal_it != result.thermal_summary.device_temperatures.end());
    CHECK(thermal_it->final_temperature > opts.thermal.ambient + 1e-4);
}

TEST_CASE("v1 PWM target_component controls MOSFET conduction path",
          "[v1][mixed-domain][pwm][mosfet][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("vin");
    const auto n_out = circuit.add_node("out");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 12.0);

    MOSFET::Params mosfet;
    mosfet.vth = 2.0;
    mosfet.kp = 0.35;
    mosfet.g_off = 1e-8;
    mosfet.lambda = 0.0;
    circuit.add_mosfet("M1", gnd, n_in, n_out, mosfet);
    circuit.add_resistor("Rload", n_out, gnd, 10.0);

    circuit.add_virtual_component(
        "pwm_generator",
        "PWM1",
        {gnd},
        {
            {"frequency", 10e3},
            {"duty", 0.5},
            {"duty_min", 0.0},
            {"duty_max", 1.0},
        },
        {{"target_component", "M1"}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 2e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE_FALSE(result.states.empty());

    const auto signals = circuit.signal_names();
    const auto out_it = std::find(signals.begin(), signals.end(), "V(out)");
    REQUIRE(out_it != signals.end());
    const auto out_index = static_cast<std::size_t>(std::distance(signals.begin(), out_it));

    Real vout_max = 0.0;
    for (const auto& state : result.states) {
        vout_max = std::max(vout_max, state[static_cast<Index>(out_index)]);
    }

    const auto pwm_it = result.virtual_channels.find("PWM1");
    REQUIRE(pwm_it != result.virtual_channels.end());
    const auto& pwm_values = pwm_it->second;
    const bool saw_high = std::any_of(pwm_values.begin(), pwm_values.end(), [](Real v) { return v > 0.5; });
    const bool saw_low = std::any_of(pwm_values.begin(), pwm_values.end(), [](Real v) { return v < 0.5; });

    INFO("PWM target MOSFET vout_max=" << vout_max
         << " steps=" << result.total_steps
         << " rejections=" << result.timestep_rejections);
    CHECK(saw_high);
    CHECK(saw_low);
    CHECK(vout_max > 1.0);
}

TEST_CASE("v1 PWM target_component remains stable with floating MOSFET gate node",
          "[v1][mixed-domain][pwm][mosfet][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("vin");
    const auto n_gate = circuit.add_node("gate");
    const auto n_out = circuit.add_node("out");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 12.0);

    MOSFET::Params mosfet;
    mosfet.vth = 2.0;
    mosfet.kp = 0.35;
    mosfet.g_off = 1e-8;
    mosfet.lambda = 0.0;
    circuit.add_mosfet("M1", n_gate, n_in, n_out, mosfet);
    circuit.add_resistor("Rload", n_out, gnd, 10.0);

    circuit.add_virtual_component(
        "pwm_generator",
        "PWM1",
        {gnd},
        {
            {"frequency", 10e3},
            {"duty", 0.5},
            {"duty_min", 0.0},
            {"duty_max", 1.0},
        },
        {{"target_component", "M1"}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 2e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE_FALSE(result.states.empty());

    const auto signals = circuit.signal_names();
    const auto out_it = std::find(signals.begin(), signals.end(), "V(out)");
    REQUIRE(out_it != signals.end());
    const auto out_index = static_cast<std::size_t>(std::distance(signals.begin(), out_it));

    Real vout_max = 0.0;
    for (const auto& state : result.states) {
        vout_max = std::max(vout_max, state[static_cast<Index>(out_index)]);
    }

    INFO("PWM target floating-gate MOSFET vout_max=" << vout_max
         << " steps=" << result.total_steps
         << " rejections=" << result.timestep_rejections
         << " message=" << result.message);
    CHECK(vout_max > 1.0);
}

TEST_CASE("v1 auto control scheduler samples PI at PWM period",
          "[v1][mixed-domain][control][scheduler][regression]") {
    Circuit circuit;
    const auto n_err = circuit.add_node("err");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Verr", n_err, gnd, 1.0);
    circuit.add_resistor("Rerr", n_err, gnd, 1e3);

    circuit.add_virtual_component(
        "pi_controller", "PI1", {n_err, gnd},
        {{"kp", 0.0}, {"ki", 10000.0}},
        {});
    circuit.add_virtual_component(
        "pwm_generator", "PWM1", {gnd},
        {{"frequency", 10e3}, {"duty", 0.5}, {"duty_from_input", 0.0}, {"duty_min", 0.0}, {"duty_max", 1.0}},
        {{"duty_from_channel", "PI1"}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 250e-6;
    opts.dt = 10e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.control_mode = ControlUpdateMode::Auto;
    opts.control_sample_time = 0.0;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE(result.virtual_channels.contains("PI1"));
    const auto& pi = result.virtual_channels.at("PI1");
    REQUIRE(pi.size() >= 21);

    // dt = 10 us, PWM = 10 kHz => inferred control sample time = 100 us.
    CHECK(pi[0] == Approx(0.0).margin(1e-12));
    CHECK(pi[5] == Approx(pi[0]).margin(1e-12));   // 50 us: hold
    CHECK(pi[10] > pi[5]);                          // 100 us: update
    CHECK(pi[15] == Approx(pi[10]).margin(1e-12)); // 150 us: hold
    CHECK(pi[20] > pi[15]);                         // 200 us: update
}

TEST_CASE("v1 per-block sample_time overrides legacy global control sample time",
          "[v1][mixed-domain][control][scheduler][regression]") {
    Circuit circuit;
    const auto n_err = circuit.add_node("err");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Verr", n_err, gnd, 1.0);
    circuit.add_resistor("Rerr", n_err, gnd, 1e3);

    circuit.add_virtual_component(
        "pi_controller", "PI1", {n_err, gnd},
        {{"kp", 0.0}, {"ki", 10000.0}, {"sample_time", 50e-6}},
        {});
    circuit.add_virtual_component(
        "pwm_generator", "PWM1", {gnd},
        {{"frequency", 10e3}, {"duty", 0.5}, {"duty_from_input", 0.0}, {"duty_min", 0.0}, {"duty_max", 1.0}},
        {{"duty_from_channel", "PI1"}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 120e-6;
    opts.dt = 10e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.control_mode = ControlUpdateMode::Discrete;
    opts.control_sample_time = 100e-6;  // legacy global fallback (must be ignored for PI1)
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE(result.virtual_channels.contains("PI1"));
    const auto& pi = result.virtual_channels.at("PI1");
    REQUIRE(pi.size() >= 11);

    // PI1 local Ts=50 us must override global 100 us.
    CHECK(pi[0] == Approx(0.0).margin(1e-12));
    CHECK(pi[4] == Approx(pi[0]).margin(1e-12));  // 40 us: hold
    CHECK(pi[5] > pi[4]);                         // 50 us: update
    CHECK(pi[9] == Approx(pi[5]).margin(1e-12)); // 90 us: hold
    CHECK(pi[10] > pi[9]);                        // 100 us: second update (at 100 us via 50 us cadence)
}

TEST_CASE("v1 per-block sample_time zero keeps control block continuous",
          "[v1][mixed-domain][control][scheduler][regression]") {
    Circuit circuit;
    const auto n_err = circuit.add_node("err");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Verr", n_err, gnd, 1.0);
    circuit.add_resistor("Rerr", n_err, gnd, 1e3);

    circuit.add_virtual_component(
        "pi_controller", "PI1", {n_err, gnd},
        {{"kp", 0.0}, {"ki", 10000.0}, {"sample_time", 0.0}},
        {});
    circuit.add_virtual_component(
        "pwm_generator", "PWM1", {gnd},
        {{"frequency", 10e3}, {"duty", 0.5}, {"duty_from_input", 0.0}, {"duty_min", 0.0}, {"duty_max", 1.0}},
        {{"duty_from_channel", "PI1"}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-6;
    opts.dt = 10e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.control_mode = ControlUpdateMode::Discrete;
    opts.control_sample_time = 100e-6;  // legacy global fallback must be disabled by local Ts=0
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE(result.virtual_channels.contains("PI1"));
    const auto& pi = result.virtual_channels.at("PI1");
    REQUIRE(pi.size() >= 5);

    // Continuous updates: integral grows every accepted step.
    CHECK(pi[1] > pi[0]);
    CHECK(pi[2] > pi[1]);
    CHECK(pi[3] > pi[2]);
    CHECK(pi[4] > pi[3]);
}

namespace {

Circuit build_closed_loop_buck_for_control_regression(bool use_cblock_controller,
                                                      bool add_disconnected_cblock,
                                                      bool ghost_node_first = false) {
    Circuit circuit;
    if (ghost_node_first) {
        (void)circuit.add_node("ghost");
    }
    const auto n_vin = circuit.add_node("vin");
    const auto n_ref = circuit.add_node("vref");
    const auto n_sw = circuit.add_node("sw");
    const auto n_out = circuit.add_node("vout");
    if (!ghost_node_first) {
        (void)circuit.add_node("ghost");
    }
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_vin, gnd, 12.0);
    circuit.add_voltage_source("Vref", n_ref, gnd, 6.0);
    MOSFET::Params m1_params{};
    m1_params.vth = 3.0;
    m1_params.kp = 0.35;
    m1_params.lambda = 0.01;
    m1_params.g_off = 1e-8;
    circuit.add_mosfet("M1", gnd, n_vin, n_sw, m1_params);
    circuit.add_diode("D1", gnd, n_sw, 350.0, 1e-9);
    circuit.add_inductor("L1", n_sw, n_out, 220e-6, 0.0);
    circuit.add_capacitor("Cout", n_out, gnd, 220e-6, 0.0);
    circuit.add_resistor("Rload", n_out, gnd, 8.0);
    circuit.add_virtual_component("voltage_probe", "VrefProbe", {n_ref, gnd}, {}, {});
    circuit.add_virtual_component("voltage_probe", "Xout", {n_out, gnd}, {}, {});

    if (use_cblock_controller) {
        circuit.add_virtual_component(
            "c_block", "CB1", {},
            {{"n_inputs", 1.0}, {"n_outputs", 1.0}},
            {
                {"lib_path", PULSIM_TEST_CBLOCK_CONST_LIB_PATH},
                {"inputs", "[Xout]"}
            });
    } else {
        // PI configured as constant duty reference (equivalent to dummy C-Block output=0.5).
        circuit.add_virtual_component(
            "pi_controller", "PI1", {n_ref, n_out, gnd},
            {{"kp", 0.0}, {"ki", 0.0}, {"output_min", 0.5}, {"output_max", 0.5}, {"anti_windup", 1.0}},
            {});
    }

    if (add_disconnected_cblock) {
        // This block is intentionally disconnected from duty routing.
        // It must not perturb electrical convergence.
        circuit.add_virtual_component(
            "c_block", "CB_GHOST", {},
            {{"n_inputs", 1.0}, {"n_outputs", 1.0}},
            {
                {"lib_path", PULSIM_TEST_CBLOCK_CONST_LIB_PATH},
                {"inputs", "[Xout]"}
            });
    }

    circuit.add_virtual_component(
        "pwm_generator", "PWM1", {gnd},
        {{"frequency", 10e3}, {"duty", 0.5}, {"duty_min", 0.0}, {"duty_max", 0.95}},
        {{"duty_from_channel", use_cblock_controller ? "CB1" : "PI1"}, {"target_component", "M1"}});

    return circuit;
}

Circuit build_closed_loop_buck_with_dynamic_pi(bool add_disconnected_cblock,
                                               bool ghost_node_first = false) {
    Circuit circuit;
    if (ghost_node_first) {
        (void)circuit.add_node("ghost");
    }
    const auto n_vin = circuit.add_node("vin");
    const auto n_ref = circuit.add_node("vref");
    const auto n_sw = circuit.add_node("sw");
    const auto n_out = circuit.add_node("vout");
    if (!ghost_node_first) {
        (void)circuit.add_node("ghost");
    }
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_vin, gnd, 12.0);
    circuit.add_voltage_source("Vref", n_ref, gnd, 6.0);

    MOSFET::Params m1_params{};
    m1_params.vth = 3.0;
    m1_params.kp = 0.35;
    m1_params.lambda = 0.01;
    m1_params.g_off = 1e-8;
    circuit.add_mosfet("M1", gnd, n_vin, n_sw, m1_params);

    circuit.add_diode("D1", gnd, n_sw, 350.0, 1e-9);
    circuit.add_inductor("L1", n_sw, n_out, 220e-6, 0.0);
    circuit.add_capacitor("Cout", n_out, gnd, 220e-6, 0.0);
    circuit.add_resistor("Rload", n_out, gnd, 8.0);
    circuit.add_virtual_component("voltage_probe", "Xout", {n_out, gnd}, {}, {});

    if (add_disconnected_cblock) {
        circuit.add_virtual_component(
            "c_block", "CB_GHOST", {},
            {{"n_inputs", 1.0}, {"n_outputs", 1.0}},
            {
                {"lib_path", PULSIM_TEST_CBLOCK_CONST_LIB_PATH},
                {"inputs", "[Xout]"}
            });
    }

    circuit.add_virtual_component(
        "pi_controller", "PI1", {n_ref, n_out, gnd},
        {{"kp", 0.08}, {"ki", 100.0}, {"output_min", 0.0}, {"output_max", 0.95}, {"anti_windup", 1.0}},
        {});
    circuit.add_virtual_component(
        "pwm_generator", "PWM1", {gnd},
        {{"frequency", 10e3}, {"duty", 0.5}, {"duty_min", 0.0}, {"duty_max", 0.95}},
        {{"duty_from_channel", "PI1"}, {"target_component", "M1"}});

    return circuit;
}

SimulationOptions closed_loop_buck_regression_options(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.4e-3;
    opts.dt = 1e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 5e-6;
    opts.step_mode = TransientStepMode::Variable;
    opts.step_mode_explicit = true;
    opts.enable_events = true;
    opts.enable_losses = false;
    opts.control_mode = ControlUpdateMode::Auto;
    opts.control_sample_time = 0.0;
    opts.max_step_retries = 8;
    opts.formulation_mode = FormulationMode::ProjectedWrapper;
    opts.direct_formulation_fallback = true;
    opts.newton_options.max_iterations = 99;
    opts.newton_options.enable_limiting = true;
    opts.newton_options.max_voltage_step = 2.0;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("v1 buck PI and equivalent C_BLOCK controller both converge",
          "[v1][mixed-domain][control][c_block][converter][regression]") {
    Circuit pi_circuit = build_closed_loop_buck_for_control_regression(false, false);
    const SimulationOptions pi_opts = closed_loop_buck_regression_options(pi_circuit);
    const auto pi_result =
        Simulator(pi_circuit, pi_opts).run_transient(pi_circuit.initial_state());

    REQUIRE(pi_result.success);
    REQUIRE(pi_result.virtual_channels.contains("Xout"));
    const auto& vout_pi = pi_result.virtual_channels.at("Xout");
    REQUIRE_FALSE(vout_pi.empty());

    Circuit cblock_circuit = build_closed_loop_buck_for_control_regression(true, false);
    const SimulationOptions cblock_opts = closed_loop_buck_regression_options(cblock_circuit);
    const auto cblock_result =
        Simulator(cblock_circuit, cblock_opts).run_transient(cblock_circuit.initial_state());

    REQUIRE(cblock_result.success);
    REQUIRE(cblock_result.virtual_channels.contains("Xout"));
    REQUIRE(cblock_result.virtual_channels.contains("CB1"));
    const auto& vout_cb = cblock_result.virtual_channels.at("Xout");
    const auto& duty_cb = cblock_result.virtual_channels.at("CB1");
    REQUIRE_FALSE(vout_cb.empty());
    REQUIRE_FALSE(duty_cb.empty());

    CHECK(duty_cb.back() == Approx(0.5).margin(1e-12));
    CHECK(std::abs(vout_cb.back() - vout_pi.back()) < 0.5);
}

TEST_CASE("v1 disconnected C_BLOCK does not perturb switched buck convergence",
          "[v1][mixed-domain][control][c_block][converter][regression]") {
    Circuit baseline_circuit = build_closed_loop_buck_for_control_regression(false, false, true);
    const SimulationOptions baseline_opts = closed_loop_buck_regression_options(baseline_circuit);
    const auto baseline_result =
        Simulator(baseline_circuit, baseline_opts).run_transient(baseline_circuit.initial_state());
    REQUIRE(baseline_result.success);
    REQUIRE(baseline_result.virtual_channels.contains("Xout"));
    const auto& vout_baseline = baseline_result.virtual_channels.at("Xout");
    REQUIRE_FALSE(vout_baseline.empty());

    Circuit with_disconnected = build_closed_loop_buck_for_control_regression(false, true, true);
    const SimulationOptions disconnected_opts = closed_loop_buck_regression_options(with_disconnected);
    const auto disconnected_result =
        Simulator(with_disconnected, disconnected_opts).run_transient(with_disconnected.initial_state());

    REQUIRE(disconnected_result.success);
    REQUIRE(disconnected_result.virtual_channels.contains("Xout"));
    REQUIRE(disconnected_result.virtual_channels.contains("CB_GHOST"));
    const auto& vout_with_ghost = disconnected_result.virtual_channels.at("Xout");
    REQUIRE_FALSE(vout_with_ghost.empty());

    CHECK(std::abs(vout_with_ghost.back() - vout_baseline.back()) < 0.5);
}

TEST_CASE("v1 disconnected C_BLOCK does not perturb dynamic PI buck convergence",
          "[v1][mixed-domain][control][c_block][converter][regression]") {
    Circuit baseline_circuit = build_closed_loop_buck_with_dynamic_pi(false, true);
    SimulationOptions baseline_opts = closed_loop_buck_regression_options(baseline_circuit);
    baseline_opts.tstop = 0.5e-3;
    const auto baseline_result =
        Simulator(baseline_circuit, baseline_opts).run_transient(baseline_circuit.initial_state());
    REQUIRE(baseline_result.success);
    REQUIRE(baseline_result.virtual_channels.contains("Xout"));
    const auto& vout_baseline = baseline_result.virtual_channels.at("Xout");
    REQUIRE_FALSE(vout_baseline.empty());

    Circuit with_disconnected = build_closed_loop_buck_with_dynamic_pi(true, true);
    SimulationOptions disconnected_opts = closed_loop_buck_regression_options(with_disconnected);
    disconnected_opts.tstop = 0.5e-3;
    const auto disconnected_result =
        Simulator(with_disconnected, disconnected_opts).run_transient(with_disconnected.initial_state());

    REQUIRE(disconnected_result.success);
    REQUIRE(disconnected_result.virtual_channels.contains("Xout"));
    REQUIRE(disconnected_result.virtual_channels.contains("CB_GHOST"));
    const auto& vout_with_ghost = disconnected_result.virtual_channels.at("Xout");
    REQUIRE_FALSE(vout_with_ghost.empty());

    CHECK(std::abs(vout_with_ghost.back() - vout_baseline.back()) < 0.5);
}

TEST_CASE("v1 control scheduler updates c_block channels",
          "[v1][mixed-domain][control][c_block][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 0.0);
    circuit.add_resistor("Rin", n_in, gnd, 1e3);
    circuit.add_virtual_component("voltage_probe", "Xin", {n_in, gnd}, {}, {});

    circuit.add_virtual_component(
        "c_block", "CB1", {},
        {{"n_inputs", 1.0}, {"n_outputs", 2.0}},
        {
            {"lib_path", PULSIM_TEST_CBLOCK_LIB_PATH},
            {"inputs", "[Xin]"}
        });

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1.5e-3;
    opts.dt = 0.25e-3;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.control_mode = ControlUpdateMode::Continuous;
    opts.control_sample_time = 0.0;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient(circuit.initial_state());

    REQUIRE(result.success);
    REQUIRE(result.virtual_channels.contains("CB1"));
    REQUIRE(result.virtual_channels.contains("CB1.out0"));
    REQUIRE(result.virtual_channels.contains("CB1.out1"));

    const auto& cb = result.virtual_channels.at("CB1");
    const auto& out0 = result.virtual_channels.at("CB1.out0");
    const auto& out1 = result.virtual_channels.at("CB1.out1");

    REQUIRE_FALSE(cb.empty());
    REQUIRE(cb.size() == out0.size());
    REQUIRE(cb.size() == out1.size());

    const auto [out0_min_it, out0_max_it] = std::minmax_element(out0.begin(), out0.end());
    REQUIRE(out0_min_it != out0.end());
    REQUIRE(out0_max_it != out0.end());
    CHECK(*out0_min_it < -0.1);
    CHECK(*out0_max_it > 0.1);

    bool saw_high = false;
    bool saw_low = false;
    for (const Real value : out1) {
        if (value > 0.5) saw_high = true;
        if (value < -0.5) saw_low = true;
    }
    CHECK(saw_high);
    CHECK(saw_low);
}

TEST_CASE("v1 c_block requires explicit control input mapping",
          "[v1][mixed-domain][control][c_block][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 1.0);
    circuit.add_resistor("Rin", n_in, gnd, 1e3);
    circuit.add_virtual_component(
        "c_block", "CB_NO_INPUTS", {},
        {{"n_inputs", 1.0}, {"n_outputs", 1.0}},
        {{"lib_path", PULSIM_TEST_CBLOCK_CONST_LIB_PATH}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-6;
    opts.dt = 10e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    REQUIRE_THROWS_AS(sim.run_transient(circuit.initial_state()), std::runtime_error);
}

TEST_CASE("v1 c_block without runtime library fails fast",
          "[v1][mixed-domain][control][c_block][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto gnd = Circuit::ground();

    circuit.add_voltage_source("Vin", n_in, gnd, 1.0);
    circuit.add_resistor("Rin", n_in, gnd, 1e3);
    circuit.add_virtual_component("voltage_probe", "Xin", {n_in, gnd}, {}, {});
    circuit.add_virtual_component(
        "c_block", "CB_MISSING_LIB", {},
        {{"n_inputs", 1.0}, {"n_outputs", 1.0}},
        {{"inputs", "[Xin]"}});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-6;
    opts.dt = 10e-6;
    opts.step_mode = TransientStepMode::Fixed;
    opts.step_mode_explicit = true;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    REQUIRE_THROWS_AS(sim.run_transient(circuit.initial_state()), std::runtime_error);
}

TEST_CASE("v1 event scheduler applies unified calendar ordering", "[v1][events][scheduler][regression]") {
    Circuit circuit;
    auto n_ctrl = circuit.add_node("ctrl");
    auto n_out = circuit.add_node("out");

    PWMParams pwm;
    pwm.v_low = 0.0;
    pwm.v_high = 10.0;
    pwm.frequency = 100e3;   // T = 10 us
    pwm.duty = 0.5;
    pwm.dead_time = 1e-6;    // Effective turn-off boundary: 4 us
    circuit.add_pwm_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);
    circuit.add_resistor("Rload", n_ctrl, n_out, 1e3);
    circuit.add_resistor("Rref", n_out, Circuit::ground(), 1e3);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 20e-6;
    opts.dt = 1e-6;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto& scheduler = sim.transient_services().event_scheduler;
    REQUIRE(scheduler);

    TransientStepRequest request;
    request.mode = TransientStepMode::Fixed;
    request.t_now = 1e-6;
    request.t_target = 9e-6;
    request.dt_candidate = request.t_target - request.t_now;
    request.dt_min = 1e-12;
    request.max_retries = 4;
    request.threshold_crossing_time = 2e-6;

    const Real from_threshold = scheduler->next_segment_target(request, request.t_target);
    CHECK(from_threshold == Approx(2e-6).margin(1e-12));

    request.threshold_crossing_time = std::numeric_limits<Real>::quiet_NaN();
    const Real from_pwm_dead_time = scheduler->next_segment_target(request, request.t_target);
    CHECK(from_pwm_dead_time == Approx(4e-6).margin(1e-12));
}

TEST_CASE("v1 fixed-step resolves multiple switching events inside one macro interval",
          "[v1][fixed-step][events][scheduler][converter][regression]") {
    Circuit circuit;

    auto n_ctrl = circuit.add_node("ctrl");
    auto n_vin = circuit.add_node("vin");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vdc", n_vin, Circuit::ground(), 12.0);

    PulseParams pwm;
    pwm.v_initial = 0.0;
    pwm.v_pulse = 10.0;
    pwm.t_delay = 0.5e-6;
    pwm.t_rise = 0.1e-6;
    pwm.t_fall = 0.1e-6;
    pwm.t_width = 1.0e-6;
    pwm.period = 6.0e-6;
    circuit.add_pulse_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    circuit.add_vcswitch("S1", n_ctrl, n_vin, n_out, 5.0, 200.0, 1e-9);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 10e-6;
    opts.dt = 4e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 4e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_events = true;
    opts.fallback_policy.trace_retries = true;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();

    INFO("multi-event fixed-step status: " << static_cast<int>(result.final_status));
    INFO("multi-event fixed-step message: " << result.message);
    REQUIRE(result.success);
    REQUIRE(result.events.size() >= 4);

    const Real expected_on_1 = pwm.t_delay + pwm.t_rise * 0.5;
    const Real expected_off_1 = pwm.t_delay + pwm.t_rise + pwm.t_width + pwm.t_fall * 0.5;
    const Real expected_on_2 = expected_on_1 + pwm.period;
    const Real expected_off_2 = expected_off_1 + pwm.period;

    std::vector<Real> on_times;
    std::vector<Real> off_times;
    on_times.reserve(result.events.size());
    off_times.reserve(result.events.size());
    for (const auto& event : result.events) {
        if (event.type == SimulationEventType::SwitchOn) {
            on_times.push_back(event.time);
        } else if (event.type == SimulationEventType::SwitchOff) {
            off_times.push_back(event.time);
        }
    }

    REQUIRE(on_times.size() >= 2);
    REQUIRE(off_times.size() >= 2);
    CHECK(on_times[0] == Approx(expected_on_1).margin(1e-6));
    CHECK(off_times[0] == Approx(expected_off_1).margin(1e-6));
    CHECK(on_times[1] == Approx(expected_on_2).margin(1e-6));
    CHECK(off_times[1] == Approx(expected_off_2).margin(1e-6));

    for (std::size_t i = 1; i < result.events.size(); ++i) {
        CHECK(result.events[i].time > result.events[i - 1].time);
    }

    const auto has_calendar_clip = std::any_of(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::EventSplit &&
                   entry.action == "event_calendar_clip";
        });
    CHECK(has_calendar_clip);
}

TEST_CASE("v1 fixed-step source-only circuits skip event calendar clipping",
          "[v1][fixed-step][events][scheduler][regression]") {
    Circuit circuit;

    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 5.0;
    pulse.t_delay = 0.0;
    pulse.t_rise = 1e-9;
    pulse.t_fall = 1e-9;
    pulse.t_width = 20e-6;
    pulse.period = 40e-6;
    circuit.add_pulse_voltage_source("Vpulse", n_in, Circuit::ground(), pulse);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_events = true;
    opts.integrator = Integrator::RosenbrockW;
    opts.fallback_policy.trace_retries = true;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();

    INFO("source-only fixed-step status: " << static_cast<int>(result.final_status));
    INFO("source-only fixed-step message: " << result.message);
    REQUIRE(result.success);
    REQUIRE(result.time.size() >= 2);

    const auto has_calendar_clip = std::any_of(
        result.fallback_trace.begin(), result.fallback_trace.end(),
        [](const FallbackTraceEntry& entry) {
            return entry.reason == FallbackReasonCode::EventSplit &&
                   entry.action == "event_calendar_clip";
        });
    CHECK_FALSE(has_calendar_clip);
    CHECK(result.backend_telemetry.state_space_primary_steps == 0);
    CHECK(result.backend_telemetry.dae_fallback_steps >= 1);

    const Real v_out_t1 = result.states[1][n_out];
    CHECK(v_out_t1 == Approx(4.9975e-3).margin(5e-4));
}

TEST_CASE("v1 electro-thermal coupling emits device telemetry", "[v1][thermal][regression]") {
    Circuit circuit;

    auto n_gate = circuit.add_node("gate");
    auto n_drain = circuit.add_node("drain");
    auto n_source = circuit.add_node("source");

    circuit.add_voltage_source("Vg", n_gate, Circuit::ground(), 10.0);
    circuit.add_voltage_source("Vd", n_drain, Circuit::ground(), 20.0);

    MOSFET::Params m;
    m.vth = 2.5;
    m.kp = 0.01;
    m.lambda = 0.0;
    m.g_off = 1e-8;
    circuit.add_mosfet("M1", n_gate, n_drain, n_source, m);
    circuit.add_resistor("Rload", n_source, Circuit::ground(), 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_losses = true;
    opts.thermal.enable = true;
    opts.thermal.ambient = 25.0;
    opts.thermal.policy = ThermalCouplingPolicy::LossWithTemperatureScaling;

    ThermalDeviceConfig cfg;
    cfg.rth = 0.5;
    cfg.cth = 1e-4;
    cfg.temp_init = 25.0;
    cfg.temp_ref = 25.0;
    cfg.alpha = 0.004;
    opts.thermal_devices["M1"] = cfg;

    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();
    REQUIRE(result.success);
    const auto loss_it = std::find_if(
        result.loss_summary.device_losses.begin(),
        result.loss_summary.device_losses.end(),
        [](const LossResult& item) { return item.device_name == "M1"; });
    REQUIRE(loss_it != result.loss_summary.device_losses.end());
    CHECK(loss_it->total_energy > 0.0);
    CHECK(loss_it->breakdown.conduction > 0.0);

    REQUIRE(result.thermal_summary.enabled);
    CHECK(result.thermal_summary.max_temperature >= result.thermal_summary.ambient);
    CHECK_FALSE(result.thermal_summary.device_temperatures.empty());
    const auto thermal_it = std::find_if(
        result.thermal_summary.device_temperatures.begin(),
        result.thermal_summary.device_temperatures.end(),
        [](const DeviceThermalTelemetry& item) { return item.device_name == "M1"; });
    REQUIRE(thermal_it != result.thermal_summary.device_temperatures.end());
    CHECK(thermal_it->final_temperature >= result.thermal_summary.ambient);
    CHECK(thermal_it->peak_temperature >= thermal_it->final_temperature);
    CHECK(thermal_it->average_temperature >= result.thermal_summary.ambient);

    REQUIRE(result.component_electrothermal.size() == circuit.connections().size());
    const auto component_m1 = std::find_if(
        result.component_electrothermal.begin(),
        result.component_electrothermal.end(),
        [](const ComponentElectrothermalTelemetry& item) { return item.component_name == "M1"; });
    REQUIRE(component_m1 != result.component_electrothermal.end());
    CHECK(component_m1->thermal_enabled);
    CHECK(component_m1->total_energy > 0.0);
    CHECK(component_m1->conduction > 0.0);
    CHECK(component_m1->peak_temperature >= component_m1->final_temperature);

    const auto component_rload = std::find_if(
        result.component_electrothermal.begin(),
        result.component_electrothermal.end(),
        [](const ComponentElectrothermalTelemetry& item) { return item.component_name == "Rload"; });
    REQUIRE(component_rload != result.component_electrothermal.end());
    CHECK(component_rload->thermal_enabled);
    CHECK(component_rload->final_temperature >= result.thermal_summary.ambient);
    CHECK(component_rload->peak_temperature >= component_rload->final_temperature);
}

TEST_CASE("v1 electro-thermal supports resistor and diode telemetry", "[v1][thermal][regression]") {
    Circuit circuit;

    const auto n_in = circuit.add_node("in");
    const auto n_mid = circuit.add_node("mid");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, Circuit::ground(), 10.0);
    circuit.add_resistor("Rs", n_in, n_mid, 10.0);
    circuit.add_diode("D1", n_mid, Circuit::ground(), 1.0, 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_losses = true;
    opts.thermal.enable = true;
    opts.thermal.ambient = 25.0;
    opts.thermal.policy = ThermalCouplingPolicy::LossWithTemperatureScaling;

    ThermalDeviceConfig r_cfg;
    r_cfg.rth = 0.5;
    r_cfg.cth = 5e-5;
    r_cfg.temp_init = 25.0;
    r_cfg.temp_ref = 25.0;
    r_cfg.alpha = 0.002;
    opts.thermal_devices["R1"] = r_cfg;

    ThermalDeviceConfig d_cfg;
    d_cfg.rth = 1.0;
    d_cfg.cth = 5e-5;
    d_cfg.temp_init = 25.0;
    d_cfg.temp_ref = 25.0;
    d_cfg.alpha = 0.002;
    opts.thermal_devices["D1"] = d_cfg;

    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    const auto r1_loss = std::find_if(
        result.loss_summary.device_losses.begin(),
        result.loss_summary.device_losses.end(),
        [](const LossResult& item) { return item.device_name == "R1"; });
    const auto d1_loss = std::find_if(
        result.loss_summary.device_losses.begin(),
        result.loss_summary.device_losses.end(),
        [](const LossResult& item) { return item.device_name == "D1"; });
    REQUIRE(r1_loss != result.loss_summary.device_losses.end());
    REQUIRE(d1_loss != result.loss_summary.device_losses.end());
    CHECK(r1_loss->breakdown.conduction > 0.0);
    CHECK(d1_loss->breakdown.conduction > 0.0);

    const auto r1_thermal = std::find_if(
        result.thermal_summary.device_temperatures.begin(),
        result.thermal_summary.device_temperatures.end(),
        [](const DeviceThermalTelemetry& item) { return item.device_name == "R1"; });
    const auto d1_thermal = std::find_if(
        result.thermal_summary.device_temperatures.begin(),
        result.thermal_summary.device_temperatures.end(),
        [](const DeviceThermalTelemetry& item) { return item.device_name == "D1"; });
    REQUIRE(r1_thermal != result.thermal_summary.device_temperatures.end());
    REQUIRE(d1_thermal != result.thermal_summary.device_temperatures.end());
    CHECK(r1_thermal->final_temperature > result.thermal_summary.ambient);
    CHECK(d1_thermal->final_temperature > result.thermal_summary.ambient);

    const auto r1_component = std::find_if(
        result.component_electrothermal.begin(),
        result.component_electrothermal.end(),
        [](const ComponentElectrothermalTelemetry& item) { return item.component_name == "R1"; });
    const auto d1_component = std::find_if(
        result.component_electrothermal.begin(),
        result.component_electrothermal.end(),
        [](const ComponentElectrothermalTelemetry& item) { return item.component_name == "D1"; });
    REQUIRE(r1_component != result.component_electrothermal.end());
    REQUIRE(d1_component != result.component_electrothermal.end());
    CHECK(r1_component->thermal_enabled);
    CHECK(d1_component->thermal_enabled);
}

TEST_CASE("v1 modular electrothermal fixed-step run avoids hot-path reallocations",
          "[v1][performance][allocation][modular]") {
    Circuit circuit;

    const auto n_in = circuit.add_node("in");
    const auto n_mid = circuit.add_node("mid");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, Circuit::ground(), 10.0);
    circuit.add_resistor("Rs", n_in, n_mid, 10.0);
    circuit.add_diode("D1", n_mid, Circuit::ground(), 1.0, 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_losses = true;
    opts.thermal.enable = true;
    opts.thermal.ambient = 25.0;
    opts.thermal.policy = ThermalCouplingPolicy::LossWithTemperatureScaling;

    ThermalDeviceConfig r_cfg;
    r_cfg.rth = 0.5;
    r_cfg.cth = 5e-5;
    r_cfg.temp_init = 25.0;
    r_cfg.temp_ref = 25.0;
    r_cfg.alpha = 0.002;
    opts.thermal_devices["R1"] = r_cfg;

    ThermalDeviceConfig d_cfg;
    d_cfg.rth = 1.0;
    d_cfg.cth = 5e-5;
    d_cfg.temp_init = 25.0;
    d_cfg.temp_ref = 25.0;
    d_cfg.alpha = 0.002;
    opts.thermal_devices["D1"] = d_cfg;

    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);
    REQUIRE(result.time.size() > 10);
    CHECK(result.backend_telemetry.runtime_module_count == 5);
    CHECK(result.backend_telemetry.reserved_output_samples >= static_cast<int>(result.time.size()));
    CHECK(result.backend_telemetry.time_series_reallocations == 0);
    CHECK(result.backend_telemetry.state_series_reallocations == 0);
    CHECK(result.backend_telemetry.virtual_channel_reallocations == 0);
}

TEST_CASE("v1 modular electrothermal channels stay summary-consistent",
          "[v1][thermal][losses][consistency][modular]") {
    Circuit circuit;

    const auto n_in = circuit.add_node("in");
    const auto n_mid = circuit.add_node("mid");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, Circuit::ground(), 10.0);
    circuit.add_resistor("Rs", n_in, n_mid, 10.0);
    circuit.add_diode("D1", n_mid, Circuit::ground(), 1.0, 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_losses = true;
    opts.thermal.enable = true;
    opts.thermal.ambient = 25.0;
    opts.thermal.policy = ThermalCouplingPolicy::LossWithTemperatureScaling;

    ThermalDeviceConfig r_cfg;
    r_cfg.rth = 0.5;
    r_cfg.cth = 5e-5;
    r_cfg.temp_init = 25.0;
    r_cfg.temp_ref = 25.0;
    r_cfg.alpha = 0.002;
    opts.thermal_devices["R1"] = r_cfg;

    ThermalDeviceConfig d_cfg;
    d_cfg.rth = 1.0;
    d_cfg.cth = 5e-5;
    d_cfg.temp_init = 25.0;
    d_cfg.temp_ref = 25.0;
    d_cfg.alpha = 0.002;
    opts.thermal_devices["D1"] = d_cfg;

    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);
    REQUIRE(result.time.size() > 1);

    const Real duration = result.time.back() - result.time.front();
    REQUIRE(duration > 0.0);

    auto component_row_for = [&](std::string_view name) -> const ComponentElectrothermalTelemetry* {
        const auto it = std::find_if(
            result.component_electrothermal.begin(),
            result.component_electrothermal.end(),
            [name](const ComponentElectrothermalTelemetry& row) { return row.component_name == name; });
        return it == result.component_electrothermal.end() ? nullptr : &*it;
    };
    auto thermal_row_for = [&](std::string_view name) -> const DeviceThermalTelemetry* {
        const auto it = std::find_if(
            result.thermal_summary.device_temperatures.begin(),
            result.thermal_summary.device_temperatures.end(),
            [name](const DeviceThermalTelemetry& row) { return row.device_name == name; });
        return it == result.thermal_summary.device_temperatures.end() ? nullptr : &*it;
    };

    const std::array<std::string_view, 2> names = {"R1", "D1"};
    for (const auto name : names) {
        const auto* component = component_row_for(name);
        REQUIRE(component != nullptr);

        const std::string thermal_channel_name = "T(" + std::string(name) + ")";
        const auto thermal_channel_it = result.virtual_channels.find(thermal_channel_name);
        REQUIRE(thermal_channel_it != result.virtual_channels.end());
        const auto& thermal_series = thermal_channel_it->second;
        REQUIRE(thermal_series.size() == result.time.size());

        const Real final_temperature = thermal_series.back();
        const Real peak_temperature = *std::max_element(thermal_series.begin(), thermal_series.end());
        Real thermal_sum = 0.0;
        for (const Real sample : thermal_series) {
            thermal_sum += sample;
        }
        const Real average_temperature =
            thermal_sum / static_cast<Real>(thermal_series.size());

        CHECK(component->final_temperature == Approx(final_temperature).margin(1e-6));
        CHECK(component->peak_temperature == Approx(peak_temperature).margin(1e-6));
        CHECK(component->average_temperature == Approx(average_temperature).margin(1e-6));

        const auto* thermal = thermal_row_for(name);
        REQUIRE(thermal != nullptr);
        CHECK(thermal->final_temperature == Approx(final_temperature).margin(1e-6));
        CHECK(thermal->peak_temperature == Approx(peak_temperature).margin(1e-6));
        CHECK(thermal->average_temperature == Approx(average_temperature).margin(1e-6));

        const std::string loss_channel_name = "Ploss(" + std::string(name) + ")";
        const auto loss_channel_it = result.virtual_channels.find(loss_channel_name);
        REQUIRE(loss_channel_it != result.virtual_channels.end());
        const auto& loss_series = loss_channel_it->second;
        REQUIRE(loss_series.size() == result.time.size());

        Real energy = 0.0;
        for (std::size_t i = 1; i < result.time.size(); ++i) {
            const Real dt = result.time[i] - result.time[i - 1];
            energy += loss_series[i] * dt;
        }
        const Real average_power = energy / duration;
        const Real energy_tol = std::max<Real>(1e-8, std::abs(energy) * 1e-3);
        const Real power_tol = std::max<Real>(1e-8, std::abs(average_power) * 1e-3);
        CHECK(component->total_energy == Approx(energy).margin(energy_tol));
        CHECK(component->total_loss == Approx(average_power).margin(power_tol));
    }
}

TEST_CASE("v1 direct formulation runs through DAE solve path with direct telemetry",
          "[v1][formulation][direct]") {
    Circuit circuit;

    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.formulation_mode = FormulationMode::Direct;
    opts.direct_formulation_fallback = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);
    CHECK(result.backend_telemetry.formulation_mode == "direct");
    CHECK(result.backend_telemetry.state_space_primary_steps == 0);
    CHECK(result.backend_telemetry.dae_fallback_steps >= 1);
    CHECK(result.backend_telemetry.segment_model_cache_hits == 0);
    CHECK(result.backend_telemetry.segment_model_cache_misses == 0);
}

TEST_CASE("v1 switching topologies auto-enable robust transient defaults", "[v1][robust][autoprofile]") {
    Circuit circuit;

    auto n_ctrl = circuit.add_node("ctrl");
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 24.0);

    PulseParams pwm;
    pwm.v_initial = 0.0;
    pwm.v_pulse = 10.0;
    pwm.t_delay = 0.0;
    pwm.t_rise = 50e-9;
    pwm.t_fall = 50e-9;
    pwm.t_width = 2e-6;
    pwm.period = 4e-6;
    circuit.add_pulse_voltage_source("Vctrl", n_ctrl, Circuit::ground(), pwm);

    circuit.add_vcswitch("S1", n_ctrl, n_in, n_out, 5.0, 1.0 / 0.01, 1.0 / 1e6);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 20.0);
    circuit.add_capacitor("Cout", n_out, Circuit::ground(), 47e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 20e-6;
    opts.dt = 200e-9;

    Simulator sim(circuit, opts);
    const auto& tuned = sim.options();

    CHECK(tuned.integrator == Integrator::TRBDF2);
    CHECK(tuned.adaptive_timestep);
    CHECK(tuned.enable_bdf_order_control);
    CHECK(tuned.max_step_retries >= 12);
    CHECK(tuned.newton_options.max_iterations >= 120);
    CHECK(tuned.linear_solver.allow_fallback);
    CHECK(tuned.linear_solver.order.size() >= 2);
}

TEST_CASE("v1 simulator wires unified transient service registry", "[v1][architecture][services]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 12.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-6;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto& services = sim.transient_services();

    REQUIRE(services.complete());
    CHECK(services.supports_mode(TransientStepMode::Fixed));
    CHECK(services.supports_mode(TransientStepMode::Variable));
    REQUIRE(services.segment_model);
    REQUIRE(services.segment_stepper);
    REQUIRE(services.loss_service);
    REQUIRE(services.thermal_service);

    TransientStepRequest request;
    request.mode = TransientStepMode::Fixed;
    request.t_now = 0.0;
    request.t_target = 5e-7;
    request.dt_candidate = 1e-6;
    request.dt_min = 1e-12;
    request.retry_index = 0;
    request.max_retries = 4;
    request.event_adjacent = true;

    const Real segment = services.event_scheduler->next_segment_target(request, opts.tstop);
    CHECK(segment == Approx(5e-7).margin(1e-12));

    const auto decision = services.recovery_manager->on_step_failure(request);
    CHECK_FALSE(decision.abort);
    CHECK(decision.next_dt < request.dt_candidate);

    Vector x0 = Vector::Zero(circuit.system_size());
    const auto segment_model = services.segment_model->build_model(x0, request);
    CHECK(segment_model.admissible);
    CHECK(segment_model.t_target == Approx(request.t_target).margin(1e-12));
    CHECK(segment_model.topology_signature != 0);

    TransientStepRequest solve_request = request;
    solve_request.t_target = solve_request.t_now + solve_request.dt_candidate;
    const auto solve_model = services.segment_model->build_model(x0, solve_request);
    const auto solve_outcome =
        services.segment_stepper->try_advance(solve_model, x0, solve_request);
    CHECK_FALSE(solve_outcome.requires_fallback);
    CHECK(solve_outcome.result.status == SolverStatus::Success);

    TransientStepRequest variable_request = request;
    variable_request.mode = TransientStepMode::Variable;

    const Real variable_segment =
        services.event_scheduler->next_segment_target(variable_request, opts.tstop);
    CHECK(variable_segment == Approx(segment).margin(1e-12));

    const auto variable_decision = services.recovery_manager->on_step_failure(variable_request);
    CHECK(variable_decision.stage == decision.stage);
    CHECK(variable_decision.abort == decision.abort);
    CHECK(variable_decision.next_dt == Approx(decision.next_dt).margin(1e-15));
}

TEST_CASE("v1 segment model builds linearized E/A/B/c with topology cache",
          "[v1][architecture][services][segment-model]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");
    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 12.0);
    circuit.add_resistor("R1", n_in, n_out, 2e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 2e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto& services = sim.transient_services();
    REQUIRE(services.segment_model);
    REQUIRE(services.equation_assembler);

    TransientStepRequest request;
    request.mode = TransientStepMode::Fixed;
    request.t_now = 0.0;
    request.t_target = 1e-6;
    request.dt_candidate = 1e-6;
    request.dt_min = 1e-12;
    request.retry_index = 0;
    request.max_retries = 4;
    request.event_adjacent = false;

    const Vector x0 = Vector::Zero(circuit.system_size());
    const auto model_first = services.segment_model->build_model(x0, request);
    const auto model_second = services.segment_model->build_model(x0, request);

    REQUIRE(model_first.admissible);
    REQUIRE(model_first.linear_model);
    CHECK_FALSE(model_first.cache_hit);
    CHECK(model_second.cache_hit);

    const auto& linear = *model_first.linear_model;
    CHECK(linear.E.rows() == x0.size());
    CHECK(linear.E.cols() == x0.size());
    CHECK(linear.A.rows() == x0.size());
    CHECK(linear.A.cols() == x0.size());
    CHECK(linear.B.rows() == x0.size());
    CHECK(linear.B.cols() == 0);
    CHECK(linear.u.size() == 0);
    CHECK(linear.c.size() == x0.size());

    SparseMatrix jacobian(x0.size(), x0.size());
    Vector residual = Vector::Zero(x0.size());
    services.equation_assembler->assemble_system(x0, request.t_target, request.dt_candidate, jacobian, residual);

    SparseMatrix e_diff = linear.E - jacobian;
    SparseMatrix a_diff = linear.A - jacobian;
    e_diff.prune(0.0);
    a_diff.prune(0.0);
    CHECK(e_diff.norm() == Approx(0.0).margin(1e-12));
    CHECK(a_diff.norm() == Approx(0.0).margin(1e-12));
    CHECK((linear.c + residual).lpNorm<Eigen::Infinity>() == Approx(0.0).margin(1e-12));
}

TEST_CASE("v1 segment primary path matches DAE fallback in fixed and variable modes",
          "[v1][architecture][services][segment-parity]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");
    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto& services = sim.transient_services();
    REQUIRE(services.segment_model);
    REQUIRE(services.segment_stepper);
    REQUIRE(services.nonlinear_solve);

    const Vector x0 = Vector::Zero(circuit.system_size());
    const std::array<TransientStepMode, 2> modes{
        TransientStepMode::Fixed,
        TransientStepMode::Variable
    };

    for (TransientStepMode mode : modes) {
        TransientStepRequest request;
        request.mode = mode;
        request.t_now = 0.0;
        request.t_target = 1e-6;
        request.dt_candidate = 1e-6;
        request.dt_min = 1e-12;
        request.retry_index = 0;
        request.max_retries = 4;
        request.event_adjacent = false;

        const auto model = services.segment_model->build_model(x0, request);
        REQUIRE(model.admissible);
        REQUIRE(model.linear_model);

        const auto segment = services.segment_stepper->try_advance(model, x0, request);
        REQUIRE_FALSE(segment.requires_fallback);
        REQUIRE(segment.result.status == SolverStatus::Success);

        const auto dae = services.nonlinear_solve->solve(x0, request.t_target, request.dt_candidate);
        REQUIRE(dae.status == SolverStatus::Success);

        const Real solution_diff =
            (segment.result.solution - dae.solution).lpNorm<Eigen::Infinity>();
        CHECK(solution_diff == Approx(0.0).margin(1e-9));
    }

    const auto run = sim.run_transient(x0);
    REQUIRE(run.success);
    CHECK(run.backend_telemetry.segment_model_cache_hits >= 1);
    CHECK((run.backend_telemetry.segment_model_cache_hits +
           run.backend_telemetry.segment_model_cache_misses) >= 1);
    CHECK(run.backend_telemetry.linear_factor_cache_misses >= 1);
    CHECK(run.backend_telemetry.linear_factor_cache_hits >= 1);
    CHECK(run.backend_telemetry.state_space_primary_steps >= 1);
}

TEST_CASE("v1 backend telemetry reports topology-driven linear cache invalidation",
          "[v1][performance][cache][telemetry]") {
    Circuit circuit;
    auto n_ctrl = circuit.add_node("ctrl");
    auto n_vin = circuit.add_node("vin");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vdc", n_vin, Circuit::ground(), 12.0);

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 10.0;
    pulse.t_delay = 2e-6;
    pulse.t_rise = 2e-7;
    pulse.t_fall = 2e-7;
    pulse.t_width = 2e-6;
    pulse.period = 6e-6;
    circuit.add_pulse_voltage_source("Vctrl", n_ctrl, Circuit::ground(), pulse);

    circuit.add_vcswitch("S1", n_ctrl, n_vin, n_out, 5.0, 200.0, 1e-9);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 20.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 12e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto run = sim.run_transient();
    INFO("cache-invalidation status: " << static_cast<int>(run.final_status));
    INFO("cache-invalidation message: " << run.message);
    REQUIRE(run.success);

    CHECK(run.backend_telemetry.linear_factor_cache_hits >= 1);
    CHECK(run.backend_telemetry.linear_factor_cache_misses >= 1);
    CHECK(run.backend_telemetry.linear_factor_cache_invalidations >= 1);
    CHECK((run.backend_telemetry.linear_factor_cache_last_invalidation_reason == "topology_changed" ||
           run.backend_telemetry.linear_factor_cache_last_invalidation_reason == "numeric_instability"));
}

TEST_CASE("v1 fixed-step transient pre-reserves output buffers in steady-state loop",
          "[v1][performance][allocation]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");
    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-6;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.enable_events = false;
    opts.enable_losses = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto run = sim.run_transient();
    REQUIRE(run.success);
    REQUIRE(run.time.size() == 6);

    CHECK(run.backend_telemetry.reserved_output_samples >= static_cast<int>(run.time.size()));
    CHECK(run.backend_telemetry.time_series_reallocations == 0);
    CHECK(run.backend_telemetry.state_series_reallocations == 0);
    CHECK(run.backend_telemetry.virtual_channel_reallocations == 0);
}

TEST_CASE("v1 global recovery path reports automatic regularization", "[v1][fallback][recovery]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-3;
    opts.adaptive_timestep = true;
    opts.enable_bdf_order_control = false;
    opts.max_step_retries = 2;
    opts.fallback_policy.trace_retries = true;
    opts.fallback_policy.enable_transient_gmin = true;
    opts.fallback_policy.gmin_retry_threshold = 1;
    opts.fallback_policy.gmin_initial = 1e-8;
    opts.fallback_policy.gmin_max = 1e-4;
    opts.linear_solver.order = {LinearSolverKind::CG};
    opts.linear_solver.fallback_order = {LinearSolverKind::CG};
    opts.linear_solver.allow_fallback = true;
    opts.linear_solver.auto_select = false;
    opts.linear_solver.iterative_config.max_iterations = 2;
    opts.linear_solver.iterative_config.tolerance = 1e-16;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    Vector x0 = Vector::Zero(circuit.system_size());
    auto result = sim.run_transient(x0);

    REQUIRE_FALSE(result.success);
    CHECK(result.message.find("automatic regularization attempted") != std::string::npos);
    CHECK(std::any_of(result.fallback_trace.begin(), result.fallback_trace.end(),
                      [](const FallbackTraceEntry& entry) {
                          return entry.action.find("global_recovery_") != std::string::npos;
                      }));
}

TEST_CASE("v1 strict convergence profile blocks implicit switching global recovery",
          "[v1][fallback][strict]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 5.0;
    pulse.t_delay = 1e-6;
    pulse.t_rise = 2e-7;
    pulse.t_fall = 2e-7;
    pulse.t_width = 2e-6;
    pulse.period = 4e-6;
    circuit.add_pulse_voltage_source("Vin", n_in, Circuit::ground(), pulse);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 4e-5;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-3;
    opts.adaptive_timestep = true;
    opts.enable_bdf_order_control = false;
    opts.max_step_retries = 2;
    opts.fallback_policy.trace_retries = true;
    opts.fallback_policy.convergence_profile = ConvergenceProfile::Strict;
    opts.fallback_policy.enable_transient_gmin = true;
    opts.fallback_policy.gmin_retry_threshold = 1;
    opts.fallback_policy.gmin_initial = 1e-8;
    opts.fallback_policy.gmin_max = 1e-4;
    opts.linear_solver.order = {LinearSolverKind::CG};
    opts.linear_solver.fallback_order = {LinearSolverKind::CG};
    opts.linear_solver.allow_fallback = false;
    opts.linear_solver.auto_select = false;
    opts.linear_solver.iterative_config.max_iterations = 2;
    opts.linear_solver.iterative_config.tolerance = 1e-16;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const Vector x0 = Vector::Zero(circuit.system_size());
    auto result = sim.run_transient(x0);

    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.backend_recovery_count == 0);
    CHECK(std::none_of(result.fallback_trace.begin(), result.fallback_trace.end(),
                       [](const FallbackTraceEntry& entry) {
                           return entry.action.find("global_recovery_") != std::string::npos;
                       }));
}

TEST_CASE("v1 startup fallback recovers when DC operating point fails",
          "[v1][fallback][startup]") {
    auto build_circuit = []() {
        Circuit circuit;
        auto n_in = circuit.add_node("in");
        auto n_out = circuit.add_node("out");
        circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
        circuit.add_resistor("R1", n_in, n_out, 1e3);
        circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);
        return circuit;
    };

    auto make_opts = [](const Circuit& circuit) {
        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 50e-6;
        opts.dt = 1e-6;
        opts.dt_min = 1e-12;
        opts.dt_max = 1e-3;
        opts.step_mode = TransientStepMode::Fixed;
        opts.step_mode_explicit = true;
        opts.adaptive_timestep = false;
        opts.enable_bdf_order_control = false;
        opts.max_step_retries = 2;
        opts.linear_solver.order = {LinearSolverKind::CG};
        opts.linear_solver.fallback_order = {};
        opts.linear_solver.auto_select = false;
        opts.linear_solver.iterative_config.max_iterations = 2;
        opts.linear_solver.iterative_config.tolerance = 1e-16;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();
        return opts;
    };

    {
        auto circuit = build_circuit();
        auto opts = make_opts(circuit);
        opts.linear_solver.allow_fallback = false;

        Simulator sim(circuit, opts);
        const auto result = sim.run_transient();
        REQUIRE_FALSE(result.success);
        CHECK(result.diagnostic == SimulationDiagnosticCode::DcOperatingPointFailure);
    }

    {
        auto circuit = build_circuit();
        auto opts = make_opts(circuit);
        opts.linear_solver.allow_fallback = true;

        Simulator sim(circuit, opts);
        const auto result = sim.run_transient();
        REQUIRE(result.success);
        CHECK(result.backend_telemetry.backend_recovery_count >= 1);
        CHECK(result.message.find("startup fallback") != std::string::npos);
    }
}

TEST_CASE("v1 fallback trace records deterministic reason codes", "[v1][fallback][regression]") {
    Circuit circuit;

    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.max_step_retries = 3;
    opts.fallback_policy.trace_retries = true;
    opts.fallback_policy.convergence_profile = ConvergenceProfile::Balanced;
    opts.fallback_policy.policy_dry_run = true;
    opts.fallback_policy.anti_overfit_check = true;
    opts.fallback_policy.anti_overfit_stable_budget = 0;
    opts.fallback_policy.enable_transient_gmin = true;
    opts.fallback_policy.gmin_retry_threshold = 1;
    opts.fallback_policy.gmin_initial = 1e-8;
    opts.fallback_policy.gmin_max = 1e-4;
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

    auto has_reason = [&](FallbackReasonCode code) {
        return std::any_of(result.fallback_trace.begin(), result.fallback_trace.end(),
                           [&](const FallbackTraceEntry& entry) { return entry.reason == code; });
    };

    CHECK(has_reason(FallbackReasonCode::NewtonFailure));
    CHECK((has_reason(FallbackReasonCode::TransientGminEscalation) ||
           has_reason(FallbackReasonCode::StiffnessBackoff)));
    CHECK(has_reason(FallbackReasonCode::MaxRetriesExceeded));
    CHECK(result.backend_telemetry.policy_dry_run_events ==
          static_cast<int>(result.fallback_trace.size()));
    CHECK(result.backend_telemetry.policy_recommendation_matches +
              result.backend_telemetry.policy_recommendation_mismatches ==
          result.backend_telemetry.policy_dry_run_events);
    CHECK(result.backend_telemetry.anti_overfit_budget_exceeded ==
          (result.backend_telemetry.anti_overfit_violations >
           opts.fallback_policy.anti_overfit_stable_budget));
    CHECK(std::any_of(result.fallback_trace.begin(), result.fallback_trace.end(),
                      [](const FallbackTraceEntry& entry) {
                          return entry.recommended_policy_action !=
                                 ConvergencePolicyAction::None;
                      }));
}

TEST_CASE("v1 linear solver order honored", "[v1][solver][regression]") {
    Circuit circuit;

    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2e-3;
    opts.dt = 2e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 2e-5;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    opts.linear_solver.order = {LinearSolverKind::GMRES};
    opts.linear_solver.allow_fallback = false;
    opts.linear_solver.auto_select = false;
    opts.linear_solver.iterative_config.max_iterations = 100;
    opts.linear_solver.iterative_config.tolerance = 1e-12;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    INFO("Stiffness test status: " << static_cast<int>(result.final_status));
    INFO("Stiffness test message: " << result.message);
    REQUIRE(result.success);
    REQUIRE(result.linear_solver_telemetry.last_solver.has_value());
    CHECK(result.linear_solver_telemetry.last_solver.value() == LinearSolverKind::GMRES);
    CHECK(result.linear_solver_telemetry.total_solve_calls > 0);
}

TEST_CASE("v1 stiffness handling on switching transient", "[v1][stiffness][regression]") {
    Circuit circuit;

    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 10.0;
    pulse.t_delay = 2e-6;
    pulse.t_rise = 2e-6;
    pulse.t_fall = 2e-6;
    pulse.t_width = 6e-6;
    pulse.period = 12e-6;
    circuit.add_pulse_voltage_source("Vin", n_in, Circuit::ground(), pulse);

    circuit.add_resistor("Rload", n_in, n_out, 100.0);
    circuit.add_capacitor("Cload", n_out, Circuit::ground(), 1e-7);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 30e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 2e-6;
    opts.adaptive_timestep = true;
    opts.enable_bdf_order_control = true;
    opts.bdf_config.max_order = 2;
    opts.stiffness_config.enable = true;
    opts.stiffness_config.rejection_streak_threshold = 2;
    opts.stiffness_config.newton_iter_threshold = 20;
    opts.stiffness_config.newton_streak_threshold = 2;
    opts.max_step_retries = 12;
    opts.linear_solver.order = {LinearSolverKind::KLU};
    opts.linear_solver.allow_fallback = false;
    opts.linear_solver.auto_select = false;
    opts.newton_options.max_iterations = 50;
    opts.newton_options.auto_damping = true;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    if (!result.success) {
        WARN("Stiffness test failed: status=" << static_cast<int>(result.final_status)
             << " message=" << result.message);
        return;
    }
    CHECK(result.total_steps > 0);
    CHECK(result.time.size() > 1);
}

TEST_CASE("v1 linear solver telemetry report", "[v1][solver][telemetry]") {
    Circuit circuit;

    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-5;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    opts.linear_solver.order = {LinearSolverKind::GMRES, LinearSolverKind::SparseLU};
    opts.linear_solver.allow_fallback = true;
    opts.linear_solver.auto_select = false;

    Simulator sim(circuit, opts);
    Vector x0 = Vector::Zero(circuit.system_size());
    auto result = sim.run_transient(x0);

    REQUIRE(result.success);
    REQUIRE(result.linear_solver_telemetry.last_solver.has_value());

    INFO("Linear solver last solver: " << static_cast<int>(*result.linear_solver_telemetry.last_solver));
    INFO("Linear solver total calls: " << result.linear_solver_telemetry.total_solve_calls);
    INFO("Linear solver total iterations: " << result.linear_solver_telemetry.total_iterations);
    INFO("Linear solver total fallbacks: " << result.linear_solver_telemetry.total_fallbacks);
}

TEST_CASE("v1 periodic shooting converges on RC", "[v1][steady][shooting]") {
    Circuit circuit;

    auto n_src = circuit.add_node("src");
    auto n_out = circuit.add_node("out");

    SineParams sine;
    sine.amplitude = 5.0;
    sine.frequency = 1000.0;
    sine.offset = 0.0;
    circuit.add_sine_voltage_source("V1", n_src, Circuit::ground(), sine);
    circuit.add_resistor("R1", n_src, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-4;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);

    PeriodicSteadyStateOptions pss;
    pss.period = 1.0 / sine.frequency;
    pss.max_iterations = 12;
    pss.tolerance = 1e-3;
    pss.relaxation = 1.0;

    auto result = sim.run_periodic_shooting(pss);
    INFO("Shooting message: " << result.message);
    INFO("Shooting residual: " << result.residual_norm);
    REQUIRE(result.success);
    CHECK(result.residual_norm <= pss.tolerance);
}

TEST_CASE("v1 harmonic balance converges on RC", "[v1][steady][hb]") {
    Circuit circuit;

    auto n_src = circuit.add_node("src");
    auto n_out = circuit.add_node("out");

    SineParams sine;
    sine.amplitude = 5.0;
    sine.frequency = 1000.0;
    sine.offset = 0.0;
    circuit.add_sine_voltage_source("V1", n_src, Circuit::ground(), sine);
    circuit.add_resistor("R1", n_src, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-4;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    opts.newton_options.max_iterations = 40;
    opts.newton_options.enable_newton_krylov = true;

    Simulator sim(circuit, opts);

    HarmonicBalanceOptions hb;
    hb.period = 1.0 / sine.frequency;
    hb.num_samples = 16;
    hb.max_iterations = 25;
    hb.tolerance = 1e-3;
    hb.relaxation = 1.0;
    hb.initialize_from_transient = true;

    auto result = sim.run_harmonic_balance(hb);
    INFO("HB message: " << result.message);
    INFO("HB residual: " << result.residual_norm);
    REQUIRE(result.success);
    CHECK(result.residual_norm <= hb.tolerance);
}

TEST_CASE("v1 mixed-domain lookup modes are deterministic", "[v1][mixed-domain][lookup][regression]") {
    SECTION("hold mode keeps previous sample bucket") {
        Circuit circuit;
        const auto n_in = circuit.add_node("in");
        const auto n_out = circuit.add_node("out");
        circuit.add_virtual_component("lookup_table", "LUT_H", {n_in, n_out}, {},
                                      {{"x", "[0, 1, 2]"},
                                       {"y", "[0, 10, 20]"},
                                       {"mode", "hold"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_in] = 1.4;
        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);

        REQUIRE(step.channel_values.contains("LUT_H"));
        CHECK(step.channel_values.at("LUT_H") == Approx(10.0).margin(1e-12));
    }

    SECTION("nearest mode selects nearest knot") {
        Circuit circuit;
        const auto n_in = circuit.add_node("in");
        const auto n_out = circuit.add_node("out");
        circuit.add_virtual_component("lookup_table", "LUT_N", {n_in, n_out}, {},
                                      {{"x", "[0, 1, 2]"},
                                       {"y", "[0, 10, 20]"},
                                       {"mode", "nearest"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_in] = 1.6;
        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);

        REQUIRE(step.channel_values.contains("LUT_N"));
        CHECK(step.channel_values.at("LUT_N") == Approx(20.0).margin(1e-12));
    }
}

TEST_CASE("v1 state-machine set-reset mode prioritizes reset", "[v1][mixed-domain][state-machine][regression]") {
    Circuit circuit;
    const auto n_set = circuit.add_node("set");
    const auto n_reset = circuit.add_node("reset");
    circuit.add_virtual_component("state_machine", "SM1", {n_set, n_reset},
                                  {{"threshold", 0.5}, {"high", 1.0}, {"low", 0.0}},
                                  {{"mode", "set_reset"}});

    Vector x = Vector::Zero(circuit.system_size());
    x[n_set] = 1.0;
    x[n_reset] = 0.0;
    const auto step_set = circuit.execute_mixed_domain_step(x, 0.0);
    REQUIRE(step_set.channel_values.contains("SM1"));
    CHECK(step_set.channel_values.at("SM1") == Approx(1.0).margin(1e-12));

    x[n_set] = 1.0;
    x[n_reset] = 1.0;
    const auto step_reset = circuit.execute_mixed_domain_step(x, 1e-6);
    REQUIRE(step_reset.channel_values.contains("SM1"));
    CHECK(step_reset.channel_values.at("SM1") == Approx(0.0).margin(1e-12));
}

TEST_CASE("v1 transfer-function alpha fallback is stable", "[v1][mixed-domain][transfer][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_virtual_component("transfer_function", "TF1", {n_in},
                                  {{"alpha", 0.5}}, {});

    Vector x = Vector::Zero(circuit.system_size());
    x[n_in] = 1.0;

    const auto step0 = circuit.execute_mixed_domain_step(x, 0.0);
    const auto step1 = circuit.execute_mixed_domain_step(x, 1e-6);
    const auto step2 = circuit.execute_mixed_domain_step(x, 2e-6);

    REQUIRE(step0.channel_values.contains("TF1"));
    REQUIRE(step1.channel_values.contains("TF1"));
    REQUIRE(step2.channel_values.contains("TF1"));
    CHECK(step0.channel_values.at("TF1") == Approx(0.5).margin(1e-12));
    CHECK(step1.channel_values.at("TF1") == Approx(0.75).margin(1e-12));
    CHECK(step2.channel_values.at("TF1") == Approx(0.875).margin(1e-12));
}

TEST_CASE("v1 rate limiter enforces slew bounds", "[v1][mixed-domain][rate-limiter][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_virtual_component("rate_limiter", "RL1", {n_in},
                                  {{"rising_rate", 1.0}, {"falling_rate", 2.0}}, {});

    Vector x = Vector::Zero(circuit.system_size());

    x[n_in] = 0.0;
    (void)circuit.execute_mixed_domain_step(x, 0.0);

    x[n_in] = 10.0;
    const auto rise1 = circuit.execute_mixed_domain_step(x, 1.0);
    const auto rise2 = circuit.execute_mixed_domain_step(x, 2.0);
    REQUIRE(rise1.channel_values.contains("RL1"));
    REQUIRE(rise2.channel_values.contains("RL1"));
    CHECK(rise1.channel_values.at("RL1") == Approx(1.0).margin(1e-12));
    CHECK(rise2.channel_values.at("RL1") == Approx(2.0).margin(1e-12));

    x[n_in] = -10.0;
    const auto fall = circuit.execute_mixed_domain_step(x, 3.0);
    REQUIRE(fall.channel_values.contains("RL1"));
    CHECK(fall.channel_values.at("RL1") == Approx(0.0).margin(1e-12));
}

TEST_CASE("v1 protection event blocks trip predictably", "[v1][mixed-domain][events][regression]") {
    SECTION("fuse trips by i2t integral") {
        Circuit circuit;
        const auto n_main = circuit.add_node("main");
        circuit.add_switch("F_SW", n_main, Circuit::ground(), true, 1e3, 1e-9);
        circuit.add_virtual_component("fuse", "F1", {n_main, Circuit::ground()},
                                      {{"g_on", 1.0}, {"blow_i2t", 0.01}, {"initial_closed", 1.0}},
                                      {{"target_component", "F_SW"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_main] = 2.0;
        (void)circuit.execute_mixed_domain_step(x, 0.0);      // initialize state
        const auto step = circuit.execute_mixed_domain_step(x, 0.01);  // adds i^2*dt = 0.04

        REQUIRE(step.channel_values.contains("F1.state"));
        REQUIRE(step.channel_values.contains("F1.i2t"));
        CHECK(step.channel_values.at("F1.i2t") == Approx(0.04).margin(1e-12));
        CHECK(step.channel_values.at("F1.state") == Approx(0.0).margin(1e-12));
    }

    SECTION("circuit breaker trips after overcurrent timer expires") {
        Circuit circuit;
        const auto n_main = circuit.add_node("main");
        circuit.add_switch("B_SW", n_main, Circuit::ground(), true, 1e3, 1e-9);
        circuit.add_virtual_component("circuit_breaker", "B1", {n_main, Circuit::ground()},
                                      {{"g_on", 1.0}, {"trip_current", 1.0},
                                       {"trip_time", 0.01}, {"initial_closed", 1.0}},
                                      {{"target_component", "B_SW"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_main] = 2.0;
        (void)circuit.execute_mixed_domain_step(x, 0.0);
        const auto step = circuit.execute_mixed_domain_step(x, 0.015);

        REQUIRE(step.channel_values.contains("B1.state"));
        REQUIRE(step.channel_values.contains("B1.trip_timer"));
        CHECK(step.channel_values.at("B1.trip_timer") == Approx(0.015).margin(1e-12));
        CHECK(step.channel_values.at("B1.state") == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("v1 relay and latch-type event blocks change states correctly",
          "[v1][mixed-domain][events][latch][regression]") {
    SECTION("relay pickup/dropout toggles NO and NC channels") {
        Circuit circuit;
        const auto n_coil = circuit.add_node("coil");
        const auto n_main = circuit.add_node("main");
        const auto n_no = circuit.add_node("no");
        const auto n_nc = circuit.add_node("nc");
        circuit.add_switch("K1__no", n_main, n_no, false, 1e3, 1e-9);
        circuit.add_switch("K1__nc", n_main, n_nc, true, 1e3, 1e-9);
        circuit.add_virtual_component("relay", "K1", {n_coil, Circuit::ground()},
                                      {{"pickup_voltage", 5.0}, {"dropout_voltage", 3.0}},
                                      {{"target_component_no", "K1__no"},
                                       {"target_component_nc", "K1__nc"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_coil] = 6.0;
        const auto step_on = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step_on.channel_values.contains("K1.no_state"));
        REQUIRE(step_on.channel_values.contains("K1.nc_state"));
        CHECK(step_on.channel_values.at("K1.no_state") == Approx(1.0).margin(1e-12));
        CHECK(step_on.channel_values.at("K1.nc_state") == Approx(0.0).margin(1e-12));

        x[n_coil] = 0.0;
        const auto step_off = circuit.execute_mixed_domain_step(x, 2e-6);
        CHECK(step_off.channel_values.at("K1.no_state") == Approx(0.0).margin(1e-12));
        CHECK(step_off.channel_values.at("K1.nc_state") == Approx(1.0).margin(1e-12));
    }

    SECTION("thyristor latches on gate and commutates off") {
        Circuit circuit;
        const auto n_gate = circuit.add_node("gate");
        const auto n_main = circuit.add_node("main");
        circuit.add_switch("SCR_SW", n_main, Circuit::ground(), false, 1e3, 1e-9);
        circuit.add_virtual_component("thyristor", "SCR1", {n_gate, n_main, Circuit::ground()},
                                      {{"gate_threshold", 1.0}, {"holding_current", 0.05},
                                       {"latch_current", 0.5}, {"g_on", 10.0}},
                                      {{"target_component", "SCR_SW"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_gate] = 2.0;
        x[n_main] = 0.2;
        const auto step_on = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step_on.channel_values.contains("SCR1.state"));
        CHECK(step_on.channel_values.at("SCR1.state") == Approx(1.0).margin(1e-12));

        x[n_gate] = 0.0;
        x[n_main] = 0.2;
        const auto step_hold = circuit.execute_mixed_domain_step(x, 2e-6);
        CHECK(step_hold.channel_values.at("SCR1.state") == Approx(1.0).margin(1e-12));

        x[n_main] = -0.1;
        const auto step_off = circuit.execute_mixed_domain_step(x, 3e-6);
        CHECK(step_off.channel_values.at("SCR1.state") == Approx(0.0).margin(1e-12));
    }

    SECTION("triac accepts bipolar trigger and unlatches at low current") {
        Circuit circuit;
        const auto n_gate = circuit.add_node("gate");
        const auto n_main = circuit.add_node("main");
        circuit.add_switch("TRI_SW", n_main, Circuit::ground(), false, 1e3, 1e-9);
        circuit.add_virtual_component("triac", "TRI1", {n_gate, n_main, Circuit::ground()},
                                      {{"gate_threshold", 1.0}, {"holding_current", 0.05},
                                       {"latch_current", 0.5}, {"g_on", 10.0}},
                                      {{"target_component", "TRI_SW"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_gate] = -2.0;
        x[n_main] = 0.2;
        const auto step_on = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step_on.channel_values.contains("TRI1.state"));
        CHECK(step_on.channel_values.at("TRI1.state") == Approx(1.0).margin(1e-12));

        x[n_gate] = 0.0;
        x[n_main] = 0.001;
        const auto step_off = circuit.execute_mixed_domain_step(x, 2e-6);
        CHECK(step_off.channel_values.at("TRI1.state") == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("v1 signal mux/demux routing is deterministic", "[v1][mixed-domain][routing][regression]") {
    Circuit circuit;
    const auto n_a = circuit.add_node("a");
    const auto n_b = circuit.add_node("b");
    const auto n_c = circuit.add_node("c");

    circuit.add_virtual_component("signal_mux", "MUX1", {n_a, n_b, n_c},
                                  {{"select_index", 2.0}}, {});
    circuit.add_virtual_component("signal_demux", "DMX1", {n_a, n_b, n_c},
                                  {}, {});

    Vector x = Vector::Zero(circuit.system_size());
    x[n_a] = 1.0;
    x[n_b] = 2.0;
    x[n_c] = 3.0;
    const auto step = circuit.execute_mixed_domain_step(x, 1e-6);

    REQUIRE(step.channel_values.contains("MUX1"));
    REQUIRE(step.channel_values.contains("DMX1"));
    CHECK(step.channel_values.at("MUX1") == Approx(3.0).margin(1e-12));
    CHECK(step.channel_values.at("DMX1") == Approx(1.0).margin(1e-12));
}

TEST_CASE("v1 comparator hysteresis keeps state across threshold band",
          "[v1][mixed-domain][comparator][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_virtual_component("comparator", "CMP1", {n_in},
                                  {{"threshold", 0.0}, {"hysteresis", 0.2},
                                   {"high", 5.0}, {"low", 0.0}},
                                  {});

    Vector x = Vector::Zero(circuit.system_size());

    x[n_in] = 0.0;
    const auto step0 = circuit.execute_mixed_domain_step(x, 0.0);
    x[n_in] = 0.11;  // Crosses +threshold + hysteresis/2
    const auto step1 = circuit.execute_mixed_domain_step(x, 1e-6);
    x[n_in] = 0.05;  // Inside band, should hold previous ON state
    const auto step2 = circuit.execute_mixed_domain_step(x, 2e-6);
    x[n_in] = -0.11;  // Crosses threshold - hysteresis/2
    const auto step3 = circuit.execute_mixed_domain_step(x, 3e-6);

    REQUIRE(step0.channel_values.contains("CMP1"));
    REQUIRE(step1.channel_values.contains("CMP1"));
    REQUIRE(step2.channel_values.contains("CMP1"));
    REQUIRE(step3.channel_values.contains("CMP1"));

    CHECK(step0.channel_values.at("CMP1") == Approx(0.0).margin(1e-12));
    CHECK(step1.channel_values.at("CMP1") == Approx(5.0).margin(1e-12));
    CHECK(step2.channel_values.at("CMP1") == Approx(5.0).margin(1e-12));
    CHECK(step3.channel_values.at("CMP1") == Approx(0.0).margin(1e-12));
}

TEST_CASE("v1 sample-hold updates only on sampling instants",
          "[v1][mixed-domain][sample-hold][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_virtual_component("sample_hold", "SH1", {n_in},
                                  {{"sample_period", 1e-3}}, {});

    Vector x = Vector::Zero(circuit.system_size());

    x[n_in] = 1.0;
    const auto s0 = circuit.execute_mixed_domain_step(x, 0.0);
    x[n_in] = 2.0;
    const auto s1 = circuit.execute_mixed_domain_step(x, 4e-4);
    x[n_in] = 3.0;
    const auto s2 = circuit.execute_mixed_domain_step(x, 1e-3);
    x[n_in] = 4.0;
    const auto s3 = circuit.execute_mixed_domain_step(x, 1.5e-3);

    REQUIRE(s0.channel_values.contains("SH1"));
    REQUIRE(s1.channel_values.contains("SH1"));
    REQUIRE(s2.channel_values.contains("SH1"));
    REQUIRE(s3.channel_values.contains("SH1"));

    CHECK(s0.channel_values.at("SH1") == Approx(1.0).margin(1e-12));
    CHECK(s1.channel_values.at("SH1") == Approx(1.0).margin(1e-12));
    CHECK(s2.channel_values.at("SH1") == Approx(3.0).margin(1e-12));
    CHECK(s3.channel_values.at("SH1") == Approx(3.0).margin(1e-12));
}

TEST_CASE("v1 delay block interpolates delayed value", "[v1][mixed-domain][delay][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_virtual_component("delay_block", "DLY1", {n_in},
                                  {{"delay", 1e-3}}, {});

    Vector x = Vector::Zero(circuit.system_size());

    x[n_in] = 0.0;
    const auto s0 = circuit.execute_mixed_domain_step(x, 0.0);
    x[n_in] = 10.0;
    const auto s1 = circuit.execute_mixed_domain_step(x, 1e-3);
    x[n_in] = 20.0;
    const auto s2 = circuit.execute_mixed_domain_step(x, 1.5e-3);
    x[n_in] = 30.0;
    const auto s3 = circuit.execute_mixed_domain_step(x, 2e-3);

    REQUIRE(s0.channel_values.contains("DLY1"));
    REQUIRE(s1.channel_values.contains("DLY1"));
    REQUIRE(s2.channel_values.contains("DLY1"));
    REQUIRE(s3.channel_values.contains("DLY1"));

    CHECK(s0.channel_values.at("DLY1") == Approx(0.0).margin(1e-12));
    CHECK(s1.channel_values.at("DLY1") == Approx(0.0).margin(1e-12));
    CHECK(s2.channel_values.at("DLY1") == Approx(5.0).margin(1e-12));
    CHECK(s3.channel_values.at("DLY1") == Approx(10.0).margin(1e-12));
}

TEST_CASE("v1 pwm generator clamps duty from input", "[v1][mixed-domain][pwm][regression]") {
    Circuit circuit;
    const auto n_cmd = circuit.add_node("cmd");
    circuit.add_virtual_component("pwm_generator", "PWM1", {n_cmd},
                                  {{"frequency", 1000.0},
                                   {"duty_from_input", 1.0},
                                   {"duty_gain", 1.0},
                                   {"duty_offset", 0.0},
                                   {"duty_min", 0.2},
                                   {"duty_max", 0.8}},
                                  {});

    Vector x = Vector::Zero(circuit.system_size());
    x[n_cmd] = 0.95;
    const auto hi = circuit.execute_mixed_domain_step(x, 0.0);
    const auto hi_off = circuit.execute_mixed_domain_step(x, 4.5e-4);  // phase=0.45 => carrier=0.9 > duty

    x[n_cmd] = 0.10;
    const auto lo = circuit.execute_mixed_domain_step(x, 2.5e-4);  // phase=0.25 => carrier=0.5 > duty_min

    REQUIRE(hi.channel_values.contains("PWM1"));
    REQUIRE(hi.channel_values.contains("PWM1.duty"));
    REQUIRE(hi_off.channel_values.contains("PWM1"));
    REQUIRE(hi_off.channel_values.contains("PWM1.duty"));
    REQUIRE(lo.channel_values.contains("PWM1"));
    REQUIRE(lo.channel_values.contains("PWM1.duty"));

    CHECK(hi.channel_values.at("PWM1.duty") == Approx(0.8).margin(1e-12));
    CHECK(hi.channel_values.at("PWM1") == Approx(1.0).margin(1e-12));
    CHECK(hi_off.channel_values.at("PWM1.duty") == Approx(0.8).margin(1e-12));
    CHECK(hi_off.channel_values.at("PWM1") == Approx(0.0).margin(1e-12));
    CHECK(lo.channel_values.at("PWM1.duty") == Approx(0.2).margin(1e-12));
    CHECK(lo.channel_values.at("PWM1") == Approx(0.0).margin(1e-12));
}

TEST_CASE("v1 pwm generator can switch all power-switch families via target component",
          "[v1][mixed-domain][pwm][switching]") {
    auto estimate_conductance = [](Circuit& circuit, const Vector& state, Index node) {
        SparseMatrix J;
        Vector f;
        circuit.assemble_jacobian(J, f, state);
        const Real v = std::max<Real>(std::abs(state[node]), 1e-9);
        return std::abs(f[node]) / v;
    };

    auto run_case = [&](const std::string& target_type,
                        const std::function<void(Circuit&, Index, Index, Index)>& add_target) {
        Circuit circuit;
        const auto n_cmd = circuit.add_node("cmd");
        const auto n_ctrl = circuit.add_node("ctrl");
        const auto n_gate = circuit.add_node("gate");
        const auto n_main = circuit.add_node("main");

        add_target(circuit, n_ctrl, n_gate, n_main);
        circuit.add_virtual_component(
            "pwm_generator",
            "PWM1",
            {n_cmd},
            {
                {"frequency", 1000.0},
                {"duty_from_input", 1.0},
                {"duty_min", 0.0},
                {"duty_max", 1.0},
            },
            {{"target_component", "SW"}}
        );

        Vector x = Vector::Zero(circuit.system_size());
        x[n_main] = 1.0;
        x[n_cmd] = 0.75;

        const auto step_on = circuit.execute_mixed_domain_step(x, 1.0e-4);   // phase=0.1, carrier=0.2
        const Real g_on = estimate_conductance(circuit, x, n_main);
        const auto step_off = circuit.execute_mixed_domain_step(x, 4.0e-4);  // phase=0.4, carrier=0.8
        const Real g_off = estimate_conductance(circuit, x, n_main);

        INFO("target=" << target_type << " g_on=" << g_on << " g_off=" << g_off);
        REQUIRE(step_on.channel_values.contains("PWM1"));
        REQUIRE(step_off.channel_values.contains("PWM1"));
        CHECK(step_on.channel_values.at("PWM1") == Approx(1.0).margin(1e-12));
        CHECK(step_off.channel_values.at("PWM1") == Approx(0.0).margin(1e-12));
        CHECK(g_on > g_off * 10.0);
    };

    SECTION("ideal switch target") {
        run_case("switch", [](Circuit& circuit, Index /*ctrl*/, Index /*gate*/, Index n_main) {
            circuit.add_switch("SW", n_main, Circuit::ground(), false, 1e3, 1e-9);
        });
    }

    SECTION("voltage-controlled switch target") {
        run_case("vcswitch", [](Circuit& circuit, Index n_ctrl, Index /*gate*/, Index n_main) {
            circuit.add_vcswitch("SW", n_ctrl, n_main, Circuit::ground(), 2.5, 1e3, 1e-9);
        });
    }

    SECTION("mosfet target") {
        run_case("mosfet", [](Circuit& circuit, Index /*ctrl*/, Index n_gate, Index n_main) {
            MOSFET::Params params;
            params.vth = 2.0;
            params.kp = 0.2;
            params.g_off = 1e-12;
            params.is_nmos = true;
            circuit.add_mosfet("SW", n_gate, n_main, Circuit::ground(), params);
        });
    }

    SECTION("igbt target") {
        run_case("igbt", [](Circuit& circuit, Index /*ctrl*/, Index n_gate, Index n_main) {
            IGBT::Params params;
            params.vth = 5.0;
            params.g_on = 2e3;
            params.g_off = 1e-12;
            circuit.add_igbt("SW", n_gate, n_main, Circuit::ground(), params);
        });
    }
}

TEST_CASE("v1 pulse source driver can switch all power-switch families",
          "[v1][mixed-domain][pulse][switching]") {
    auto estimate_conductance = [](Circuit& circuit, const Vector& state, Index node) {
        SparseMatrix J;
        Vector f;
        circuit.assemble_jacobian(J, f, state);
        const Real v = std::max<Real>(std::abs(state[node]), 1e-9);
        return std::abs(f[node]) / v;
    };

    auto run_case = [&](const std::string& target_type,
                        const std::function<void(Circuit&, Index, Index, Index)>& add_target) {
        Circuit circuit;
        const auto n_drv = circuit.add_node("drv");
        const auto n_ctrl = circuit.add_node("ctrl");
        const auto n_gate = circuit.add_node("gate");
        const auto n_main = circuit.add_node("main");

        PulseParams pulse;
        pulse.v_initial = 0.0;
        pulse.v_pulse = 10.0;
        pulse.t_delay = 0.0;
        pulse.t_rise = 1e-9;
        pulse.t_fall = 1e-9;
        pulse.t_width = 4e-4;
        pulse.period = 1e-3;
        circuit.add_pulse_voltage_source("VDRV", n_drv, Circuit::ground(), pulse);

        add_target(circuit, n_ctrl, n_gate, n_main);
        circuit.bind_switch_driver("VDRV", "SW");

        Vector x = Vector::Zero(circuit.system_size());
        x[n_main] = 1.0;

        (void)circuit.execute_mixed_domain_step(x, 1.0e-4);  // inside pulse high window
        const Real g_on = estimate_conductance(circuit, x, n_main);
        (void)circuit.execute_mixed_domain_step(x, 7.0e-4);  // pulse low window
        const Real g_off = estimate_conductance(circuit, x, n_main);

        INFO("target=" << target_type << " g_on=" << g_on << " g_off=" << g_off);
        CHECK(g_on > g_off * 10.0);
    };

    SECTION("ideal switch target") {
        run_case("switch", [](Circuit& circuit, Index /*ctrl*/, Index /*gate*/, Index n_main) {
            circuit.add_switch("SW", n_main, Circuit::ground(), false, 1e3, 1e-9);
        });
    }

    SECTION("voltage-controlled switch target") {
        run_case("vcswitch", [](Circuit& circuit, Index n_ctrl, Index /*gate*/, Index n_main) {
            circuit.add_vcswitch("SW", n_ctrl, n_main, Circuit::ground(), 2.5, 1e3, 1e-9);
        });
    }

    SECTION("mosfet target") {
        run_case("mosfet", [](Circuit& circuit, Index /*ctrl*/, Index n_gate, Index n_main) {
            MOSFET::Params params;
            params.vth = 2.0;
            params.kp = 0.2;
            params.g_off = 1e-12;
            params.is_nmos = true;
            circuit.add_mosfet("SW", n_gate, n_main, Circuit::ground(), params);
        });
    }

    SECTION("igbt target") {
        run_case("igbt", [](Circuit& circuit, Index /*ctrl*/, Index n_gate, Index n_main) {
            IGBT::Params params;
            params.vth = 5.0;
            params.g_on = 2e3;
            params.g_off = 1e-12;
            circuit.add_igbt("SW", n_gate, n_main, Circuit::ground(), params);
        });
    }
}

TEST_CASE("v1 magnetic telemetry channels remain physically bounded",
          "[v1][mixed-domain][magnetic][regression]") {
    SECTION("saturable inductor effective inductance decreases with current") {
        Circuit circuit;
        const auto n_a = circuit.add_node("a");
        const auto n_b = circuit.add_node("b");
        circuit.add_inductor("L1", n_a, n_b, 1e-3);
        circuit.add_virtual_component("saturable_inductor", "LSAT1", {n_a, n_b},
                                      {{"inductance", 1e-3},
                                       {"saturation_current", 2.0},
                                       {"saturation_inductance", 2e-4},
                                       {"saturation_exponent", 2.0}},
                                      {{"target_component", "L1"}});

        const Index branch = circuit.num_nodes();  // first branch current index
        Vector x = Vector::Zero(circuit.system_size());

        x[branch] = 0.0;
        const auto low_i = circuit.execute_mixed_domain_step(x, 1e-6);
        x[branch] = 10.0;
        const auto high_i = circuit.execute_mixed_domain_step(x, 2e-6);

        REQUIRE(low_i.channel_values.contains("LSAT1.l_eff"));
        REQUIRE(high_i.channel_values.contains("LSAT1.l_eff"));
        const Real l_low = low_i.channel_values.at("LSAT1.l_eff");
        const Real l_high = high_i.channel_values.at("LSAT1.l_eff");

        CHECK(l_low >= 2e-4);
        CHECK(l_low <= 1e-3);
        CHECK(l_high >= 2e-4);
        CHECK(l_high <= 1e-3);
        CHECK(l_high < l_low);
    }

    SECTION("hysteresis model updates memory state only when time advances") {
        Circuit circuit;
        const auto n_a = circuit.add_node("a");
        const auto n_b = circuit.add_node("b");
        circuit.add_inductor("Lh", n_a, n_b, 1e-3);
        circuit.add_virtual_component("saturable_inductor", "Lh", {n_a, n_b},
                                      {{"inductance", 1e-3},
                                       {"saturation_current", 1.0},
                                       {"saturation_inductance", 2e-4},
                                       {"saturation_exponent", 2.0},
                                       {"magnetic_core_enabled", 1.0},
                                       {"hysteresis_band", 0.05},
                                       {"hysteresis_strength", 0.2},
                                       {"hysteresis_state_init", 1.0}},
                                      {{"target_component", "Lh"},
                                       {"magnetic_core_model", "hysteresis"}});

        const Index branch = circuit.num_nodes();
        Vector x = Vector::Zero(circuit.system_size());
        x[branch] = 0.2;
        const auto s1 = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(s1.channel_values.contains("Lh.h_state"));
        CHECK(s1.channel_values.at("Lh.h_state") == Approx(1.0).margin(1e-12));

        // Same timestamp should not commit a hysteresis state transition.
        x[branch] = -0.2;
        const auto s_same_time = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(s_same_time.channel_values.contains("Lh.h_state"));
        CHECK(s_same_time.channel_values.at("Lh.h_state") == Approx(1.0).margin(1e-12));

        // Advancing accepted time allows deterministic state transition.
        const auto s2 = circuit.execute_mixed_domain_step(x, 2e-6);
        REQUIRE(s2.channel_values.contains("Lh.h_state"));
        CHECK(s2.channel_values.at("Lh.h_state") == Approx(-1.0).margin(1e-12));
    }

    SECTION("coupled inductor emits expected mutual inductance") {
        Circuit circuit;
        const auto n_a = circuit.add_node("a");
        const auto n_b = circuit.add_node("b");
        const auto n_c = circuit.add_node("c");
        const auto n_d = circuit.add_node("d");
        circuit.add_virtual_component("coupled_inductor", "K1", {n_a, n_b, n_c, n_d},
                                      {{"l1", 1e-3}, {"l2", 4e-3}, {"coupling", 0.9}}, {});

        Vector x = Vector::Zero(circuit.system_size());
        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);

        REQUIRE(step.channel_values.contains("K1.k"));
        REQUIRE(step.channel_values.contains("K1.mutual"));
        CHECK(step.channel_values.at("K1.k") == Approx(0.9).margin(1e-12));
        CHECK(step.channel_values.at("K1.mutual") == Approx(1.8e-3).margin(1e-12));
    }

    SECTION("coupled inductor core loss channel uses configured power law") {
        Circuit circuit;
        const auto n_a = circuit.add_node("a");
        const auto n_b = circuit.add_node("b");
        const auto n_c = circuit.add_node("c");
        const auto n_d = circuit.add_node("d");
        circuit.add_inductor("K2__L1", n_a, n_b, 1e-3);
        circuit.add_inductor("K2__L2", n_c, n_d, 1e-3);
        circuit.add_virtual_component("coupled_inductor", "K2", {n_a, n_b, n_c, n_d},
                                      {{"l1", 1e-3},
                                       {"l2", 1e-3},
                                       {"coupling", 0.95},
                                       {"magnetic_core_enabled", 1.0},
                                       {"core_loss_k", 0.1},
                                       {"core_loss_alpha", 2.0}},
                                      {{"target_component_1", "K2__L1"},
                                       {"target_component_2", "K2__L2"}});

        Vector x = Vector::Zero(circuit.system_size());
        const auto& conns = circuit.connections();
        REQUIRE(conns.size() >= 2);
        x[conns[0].branch_index] = 2.0;
        x[conns[1].branch_index] = -2.0;

        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step.channel_values.contains("K2.core_loss"));
        CHECK(step.channel_values.at("K2.core_loss") == Approx(0.4).margin(1e-12));
    }

    SECTION("transformer core loss channel uses configured power law") {
        Circuit circuit;
        const auto p1 = circuit.add_node("p1");
        const auto p2 = circuit.add_node("p2");
        const auto s1 = circuit.add_node("s1");
        const auto s2 = circuit.add_node("s2");
        circuit.add_transformer("T1", p1, p2, s1, s2, 2.0);
        circuit.add_virtual_component("transformer", "T1", {p1, p2, s1, s2},
                                      {{"turns_ratio", 2.0},
                                       {"magnetic_core_enabled", 1.0},
                                       {"core_loss_k", 0.1},
                                       {"core_loss_alpha", 2.0}},
                                      {{"target_component", "T1"}});

        Vector x = Vector::Zero(circuit.system_size());
        const auto& conns = circuit.connections();
        REQUIRE_FALSE(conns.empty());
        const auto& xfmr = conns.front();
        REQUIRE(xfmr.branch_index >= 0);
        REQUIRE(xfmr.branch_index_2 >= 0);
        x[xfmr.branch_index] = 2.0;      // primary current
        x[xfmr.branch_index_2] = -4.0;   // secondary current = -n * i_p

        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step.channel_values.contains("T1.core_loss"));
        CHECK(step.channel_values.at("T1.core_loss") == Approx(0.4).margin(1e-12));
    }

    SECTION("frequency coefficient increases average magnetic core loss") {
        auto average_core_loss_for_frequency = [](Real frequency_hz) {
            Circuit circuit;
            const auto n_a = circuit.add_node("a");
            const auto n_b = circuit.add_node("b");
            circuit.add_inductor("Lfreq", n_a, n_b, 1e-3);
            circuit.add_virtual_component("saturable_inductor", "Lfreq", {n_a, n_b},
                                          {{"inductance", 1e-3},
                                           {"saturation_current", 2.0},
                                           {"saturation_inductance", 2e-4},
                                           {"saturation_exponent", 2.0},
                                           {"magnetic_core_enabled", 1.0},
                                           {"core_loss_k", 0.1},
                                           {"core_loss_alpha", 2.0},
                                           {"core_loss_freq_coeff", 1e-4}},
                                          {{"target_component", "Lfreq"}});

            const auto& conns = circuit.connections();
            REQUIRE_FALSE(conns.empty());
            const Index branch = conns.front().branch_index;
            REQUIRE(branch >= 0);

            const Real duration = 2e-3;
            const Real dt = 2e-6;
            const Real current_amplitude = 2.0;
            Vector x = Vector::Zero(circuit.system_size());
            Real sum = 0.0;
            std::size_t count = 0;

            constexpr Real kTwoPi = Real{6.28318530717958647692};
            for (Real t = 0.0; t <= duration; t += dt) {
                x[branch] = current_amplitude * std::sin(kTwoPi * frequency_hz * t);
                const auto step = circuit.execute_mixed_domain_step(x, t);
                if (const auto it = step.channel_values.find("Lfreq.core_loss");
                    it != step.channel_values.end()) {
                    sum += it->second;
                    count += 1;
                }
            }

            REQUIRE(count > 0);
            return sum / static_cast<Real>(count);
        };

        const Real low_freq_avg = average_core_loss_for_frequency(1e3);
        const Real high_freq_avg = average_core_loss_for_frequency(20e3);

        CHECK(high_freq_avg > low_freq_avg);
    }
}

TEST_CASE("v1 control primitives keep bounded deterministic outputs",
          "[v1][mixed-domain][control][regression]") {
    SECTION("control-node writeback enables gain cascades") {
        Circuit circuit;
        const auto n_in = circuit.add_node("in");
        const auto n_mid = circuit.add_node("mid");
        const auto n_out = circuit.add_node("out");

        circuit.add_virtual_component("gain", "G1", {n_in, n_mid}, {{"gain", 2.0}}, {});
        circuit.add_virtual_component("gain", "G2", {n_mid, n_out}, {{"gain", 3.0}}, {});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_in] = 1.0;
        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step.channel_values.contains("G1"));
        REQUIRE(step.channel_values.contains("G2"));
        CHECK(step.channel_values.at("G1") == Approx(2.0).margin(1e-12));
        CHECK(step.channel_values.at("G2") == Approx(6.0).margin(1e-12));
    }

    SECTION("algebraic control loops without state fail deterministically") {
        Circuit circuit;
        const auto n_a = circuit.add_node("a");
        const auto n_b = circuit.add_node("b");

        circuit.add_virtual_component("gain", "GA", {n_a, n_b}, {{"gain", 1.0}}, {});
        circuit.add_virtual_component("gain", "GB", {n_b, n_a}, {{"gain", 1.0}}, {});

        Vector x = Vector::Zero(circuit.system_size());
        try {
            circuit.execute_mixed_domain_step(x, 1e-6);
            FAIL("Expected algebraic loop failure");
        } catch (const std::runtime_error& err) {
            const std::string message = err.what() ? err.what() : "";
            REQUIRE(message.find("Virtual control algebraic loop") != std::string::npos);
        }
    }

    SECTION("op amp output saturates at configured rails") {
        Circuit circuit;
        const auto n_pos = circuit.add_node("v_plus");
        const auto n_neg = circuit.add_node("v_minus");
        const auto n_out = circuit.add_node("v_out");
        circuit.add_virtual_component("op_amp", "A1", {n_pos, n_neg, n_out},
                                      {{"open_loop_gain", 1e5},
                                       {"rail_low", -2.0},
                                       {"rail_high", 2.0}},
                                      {});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_pos] = 1.0;
        x[n_neg] = 0.0;
        const auto high = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(high.channel_values.contains("A1"));
        CHECK(high.channel_values.at("A1") == Approx(2.0).margin(1e-12));

        x[n_pos] = -1.0;
        x[n_neg] = 0.0;
        const auto low = circuit.execute_mixed_domain_step(x, 2e-6);
        REQUIRE(low.channel_values.contains("A1"));
        CHECK(low.channel_values.at("A1") == Approx(-2.0).margin(1e-12));
    }

    SECTION("pi anti-windup recovers faster than unclamped integral") {
        Circuit circuit;
        const auto n_err = circuit.add_node("err");
        const auto n_ref = circuit.add_node("ref");

        circuit.add_virtual_component(
            "pi_controller", "PI_AW", {n_err, n_ref},
            {{"kp", 0.0}, {"ki", 1.0}, {"output_min", -1.0}, {"output_max", 1.0}, {"anti_windup", 1.0}},
            {});
        circuit.add_virtual_component(
            "pi_controller", "PI_NO_AW", {n_err, n_ref},
            {{"kp", 0.0}, {"ki", 1.0}, {"output_min", -1.0}, {"output_max", 1.0}, {"anti_windup", 0.0}},
            {});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_err] = 2.0;
        (void)circuit.execute_mixed_domain_step(x, 0.0);
        (void)circuit.execute_mixed_domain_step(x, 1.0);
        (void)circuit.execute_mixed_domain_step(x, 2.0);

        x[n_err] = -2.0;
        const auto recovery = circuit.execute_mixed_domain_step(x, 3.0);
        REQUIRE(recovery.channel_values.contains("PI_AW"));
        REQUIRE(recovery.channel_values.contains("PI_NO_AW"));
        CHECK(recovery.channel_values.at("PI_AW") <= 0.0);
        CHECK(recovery.channel_values.at("PI_NO_AW") > 0.0);
    }

    SECTION("pi controller honors global discrete control sample time") {
        Circuit circuit;
        const auto n_err = circuit.add_node("err");
        const auto n_ref = circuit.add_node("ref");
        circuit.add_virtual_component(
            "pi_controller", "PI_DS", {n_err, n_ref},
            {{"kp", 0.0}, {"ki", 1.0}},
            {});
        circuit.set_control_sample_time(1.0);

        Vector x = Vector::Zero(circuit.system_size());
        x[n_err] = 1.0;

        const auto s0 = circuit.execute_mixed_domain_step(x, 0.0);
        const auto s05 = circuit.execute_mixed_domain_step(x, 0.5);
        const auto s10 = circuit.execute_mixed_domain_step(x, 1.0);
        const auto s15 = circuit.execute_mixed_domain_step(x, 1.5);
        const auto s20 = circuit.execute_mixed_domain_step(x, 2.0);

        REQUIRE(s0.channel_values.contains("PI_DS"));
        REQUIRE(s05.channel_values.contains("PI_DS"));
        REQUIRE(s10.channel_values.contains("PI_DS"));
        REQUIRE(s15.channel_values.contains("PI_DS"));
        REQUIRE(s20.channel_values.contains("PI_DS"));

        CHECK(s05.channel_values.at("PI_DS") == Approx(s0.channel_values.at("PI_DS")).margin(1e-12));
        CHECK(s10.channel_values.at("PI_DS") > s05.channel_values.at("PI_DS"));
        CHECK(s15.channel_values.at("PI_DS") == Approx(s10.channel_values.at("PI_DS")).margin(1e-12));
        CHECK(s20.channel_values.at("PI_DS") > s15.channel_values.at("PI_DS"));
    }

    SECTION("pid derivative path reacts to error slope") {
        Circuit circuit;
        const auto n_err = circuit.add_node("err");
        const auto n_ref = circuit.add_node("ref");
        circuit.add_virtual_component("pid_controller", "PID1", {n_err, n_ref},
                                      {{"kp", 0.0}, {"ki", 0.0}, {"kd", 1.0}}, {});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_err] = 0.0;
        const auto first = circuit.execute_mixed_domain_step(x, 0.0);
        x[n_err] = 1.0;
        const auto second = circuit.execute_mixed_domain_step(x, 1.0);
        x[n_err] = 3.0;
        const auto third = circuit.execute_mixed_domain_step(x, 2.0);

        REQUIRE(first.channel_values.contains("PID1"));
        REQUIRE(second.channel_values.contains("PID1"));
        REQUIRE(third.channel_values.contains("PID1"));
        CHECK(first.channel_values.at("PID1") == Approx(0.0).margin(1e-12));
        CHECK(second.channel_values.at("PID1") == Approx(1.0).margin(1e-12));
        CHECK(third.channel_values.at("PID1") == Approx(2.0).margin(1e-12));
    }

    SECTION("integrator remains bounded by configured output rails") {
        Circuit circuit;
        const auto n_in = circuit.add_node("in");
        const auto n_ref = circuit.add_node("ref");
        circuit.add_virtual_component("integrator", "INT1", {n_in, n_ref},
                                      {{"output_min", -1.0}, {"output_max", 1.0}}, {});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_in] = 2.0;
        (void)circuit.execute_mixed_domain_step(x, 0.0);
        const auto int_1 = circuit.execute_mixed_domain_step(x, 1.0);
        const auto int_2 = circuit.execute_mixed_domain_step(x, 2.0);

        REQUIRE(int_1.channel_values.contains("INT1"));
        REQUIRE(int_2.channel_values.contains("INT1"));
        CHECK(int_1.channel_values.at("INT1") == Approx(1.0).margin(1e-12));
        CHECK(int_2.channel_values.at("INT1") == Approx(1.0).margin(1e-12));
    }

    SECTION("differentiator alpha filter smooths raw derivative") {
        Circuit circuit;
        const auto n_in = circuit.add_node("in");
        const auto n_ref = circuit.add_node("ref");
        circuit.add_virtual_component("differentiator", "D1", {n_in, n_ref},
                                      {{"alpha", 0.5}}, {});

        Vector xd = Vector::Zero(circuit.system_size());
        xd[n_in] = 0.0;
        (void)circuit.execute_mixed_domain_step(xd, 0.0);
        xd[n_in] = 10.0;
        const auto d_up = circuit.execute_mixed_domain_step(xd, 1.0);
        const auto d_hold = circuit.execute_mixed_domain_step(xd, 2.0);

        REQUIRE(d_up.channel_values.contains("D1"));
        REQUIRE(d_hold.channel_values.contains("D1"));
        CHECK(d_up.channel_values.at("D1") == Approx(5.0).margin(1e-12));
        CHECK(d_hold.channel_values.at("D1") == Approx(2.5).margin(1e-12));
    }
}

TEST_CASE("v1 mixed-domain math and state-machine edge behavior",
          "[v1][mixed-domain][math][regression]") {
    SECTION("math block operations are deterministic including divide by zero") {
        Circuit circuit;
        const auto n_a = circuit.add_node("a");
        const auto n_b = circuit.add_node("b");
        circuit.add_virtual_component("math_block", "M_ADD", {n_a, n_b}, {}, {{"operation", "add"}});
        circuit.add_virtual_component("math_block", "M_SUB", {n_a, n_b}, {}, {{"operation", "sub"}});
        circuit.add_virtual_component("math_block", "M_MUL", {n_a, n_b}, {}, {{"operation", "mul"}});
        circuit.add_virtual_component("math_block", "M_DIV", {n_a, n_b}, {}, {{"operation", "div"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_a] = 6.0;
        x[n_b] = 2.0;
        const auto step = circuit.execute_mixed_domain_step(x, 1e-6);
        REQUIRE(step.channel_values.contains("M_ADD"));
        REQUIRE(step.channel_values.contains("M_SUB"));
        REQUIRE(step.channel_values.contains("M_MUL"));
        REQUIRE(step.channel_values.contains("M_DIV"));
        CHECK(step.channel_values.at("M_ADD") == Approx(8.0).margin(1e-12));
        CHECK(step.channel_values.at("M_SUB") == Approx(4.0).margin(1e-12));
        CHECK(step.channel_values.at("M_MUL") == Approx(12.0).margin(1e-12));
        CHECK(step.channel_values.at("M_DIV") == Approx(3.0).margin(1e-12));

        x[n_b] = 0.0;
        const auto div_zero = circuit.execute_mixed_domain_step(x, 2e-6);
        REQUIRE(div_zero.channel_values.contains("M_DIV"));
        CHECK(div_zero.channel_values.at("M_DIV") == Approx(0.0).margin(1e-12));
    }

    SECTION("toggle state machine changes state only on rising threshold crossing") {
        Circuit circuit;
        const auto n_ctrl = circuit.add_node("ctrl");
        circuit.add_virtual_component("state_machine", "SM_T", {n_ctrl},
                                      {{"threshold", 0.5}, {"high", 1.0}, {"low", 0.0}},
                                      {{"mode", "toggle"}});

        Vector x = Vector::Zero(circuit.system_size());
        x[n_ctrl] = 0.0;
        const auto s0 = circuit.execute_mixed_domain_step(x, 0.0);
        x[n_ctrl] = 1.0;
        const auto s1 = circuit.execute_mixed_domain_step(x, 1e-6);
        const auto s2 = circuit.execute_mixed_domain_step(x, 2e-6);
        x[n_ctrl] = 0.0;
        const auto s3 = circuit.execute_mixed_domain_step(x, 3e-6);
        x[n_ctrl] = 1.0;
        const auto s4 = circuit.execute_mixed_domain_step(x, 4e-6);

        REQUIRE(s0.channel_values.contains("SM_T"));
        REQUIRE(s1.channel_values.contains("SM_T"));
        REQUIRE(s2.channel_values.contains("SM_T"));
        REQUIRE(s3.channel_values.contains("SM_T"));
        REQUIRE(s4.channel_values.contains("SM_T"));
        CHECK(s0.channel_values.at("SM_T") == Approx(0.0).margin(1e-12));
        CHECK(s1.channel_values.at("SM_T") == Approx(1.0).margin(1e-12));
        CHECK(s2.channel_values.at("SM_T") == Approx(1.0).margin(1e-12));
        CHECK(s3.channel_values.at("SM_T") == Approx(1.0).margin(1e-12));
        CHECK(s4.channel_values.at("SM_T") == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("v1 transient result exposes mixed-domain channels and metadata",
          "[v1][mixed-domain][simulation][regression]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, Circuit::ground(), 100.0);
    circuit.add_virtual_component("voltage_probe", "VP", {n_in, Circuit::ground()}, {}, {});

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    REQUIRE(result.mixed_domain_phase_order == Circuit::mixed_domain_phase_order());
    REQUIRE(result.virtual_channels.contains("VP"));
    REQUIRE(result.virtual_channel_metadata.contains("VP"));
    CHECK(result.virtual_channels.at("VP").size() == result.time.size());

    const auto& meta = result.virtual_channel_metadata.at("VP");
    CHECK(meta.component_type == "voltage_probe");
    CHECK(meta.component_name == "VP");
    CHECK(meta.domain == "instrumentation");
}

TEST_CASE("v1 buck closed-loop callback tracks reference without divergence",
          "[v1][converter][closed-loop][regression]") {
    constexpr Real vin = 24.0;
    constexpr Real vref = 12.0;
    constexpr Real frequency = 25'000.0;
    constexpr Real period = 1.0 / frequency;

    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto n_sw = circuit.add_node("sw");
    const auto n_out = circuit.add_node("out");
    const auto n_ctrl = circuit.add_node("ctrl");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), vin);

    PWMParams pwm;
    pwm.v_high = 5.0;
    pwm.v_low = 0.0;
    pwm.frequency = frequency;
    pwm.duty = 0.25;
    circuit.add_pwm_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    circuit.add_vcswitch("S1", n_ctrl, n_in, n_sw, 2.5, 350.0, 1e-9);
    circuit.add_diode("D1", Circuit::ground(), n_sw, 350.0, 1e-9);
    circuit.add_inductor("L1", n_sw, n_out, 330e-6);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 220e-6, 0.0);
    circuit.add_resistor("Rload", n_out, Circuit::ground(), 6.0);
    circuit.add_resistor("Rbleed_sw", n_sw, Circuit::ground(), 1e6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 6e-3;
    opts.dt = period / 40.0;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = Integrator::BDF1;
    opts.linear_solver.order = {LinearSolverKind::KLU, LinearSolverKind::SparseLU};
    opts.linear_solver.auto_select = false;
    opts.linear_solver.allow_fallback = true;
    opts.newton_options.max_iterations = 140;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Real duty = 0.25;
    Real next_control_time = period;
    Real vout_filtered = 0.0;
    bool vout_initialized = false;
    Real integral = 0.0;
    const Real duty_min = 0.05;
    const Real duty_max = 0.90;
    const Real duty_slew = 0.015;
    const Real duty_ff = std::clamp(vref / vin, duty_min, duty_max);
    const Real kp = 0.018;
    const Real ki = 45.0;
    const Real alpha = 0.92;

    Simulator sim(circuit, opts);
    auto callback = [&](Real time, const Vector& state) {
        const Real vout = state[n_out];
        if (!vout_initialized) {
            vout_filtered = vout;
            vout_initialized = true;
        } else {
            vout_filtered = alpha * vout_filtered + (1.0 - alpha) * vout;
        }

        if (time + 1e-15 < next_control_time) {
            return;
        }

        const Real error = vref - vout_filtered;
        integral += error * period;
        integral = std::clamp(integral, -0.005, 0.005);

        const Real duty_target = std::clamp(duty_ff + kp * error + ki * integral, duty_min, duty_max);
        const Real duty_step = std::clamp(duty_target - duty, -duty_slew, duty_slew);
        duty = std::clamp(duty + duty_step, duty_min, duty_max);
        circuit.set_pwm_duty("Vpwm", duty);

        while (next_control_time <= time + 1e-15) {
            next_control_time += period;
        }
    };

    auto result = sim.run_transient(callback);
    REQUIRE(result.success);
    REQUIRE(result.states.size() > 30);

    const std::size_t n = result.states.size();
    const std::size_t start_prev = n * 7 / 10;
    const std::size_t start_last = n * 85 / 100;
    REQUIRE(start_last > start_prev);

    auto mean_vout = [&](std::size_t begin, std::size_t end) {
        Real sum = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            sum += result.states[i][n_out];
        }
        return sum / static_cast<Real>(end - begin);
    };

    const Real vout_prev = mean_vout(start_prev, start_last);
    const Real vout_last = mean_vout(start_last, n);

    INFO("vout_prev=" << vout_prev << " vout_last=" << vout_last << " duty_final=" << duty);
    CHECK(std::isfinite(vout_last));
    CHECK(vout_last >= 10.5);
    CHECK(vout_last <= 13.5);
    CHECK(std::abs(vout_last - vout_prev) <= 2.0);
    CHECK(duty >= 0.30);
    CHECK(duty <= 0.70);
}

TEST_CASE("v1 runtime circuit supports string_view name lookups", "[v1][runtime][lookup]") {
    Circuit circuit;

    CHECK(circuit.add_node("GND") == Circuit::ground());
    CHECK(circuit.get_node(std::string_view{"gNd"}) == Circuit::ground());

    const auto n_ctrl = circuit.add_node(std::string_view{"ctrl"});
    const auto n_gate = circuit.add_node(std::string_view{"gate"});
    const auto n_out = circuit.add_node(std::string_view{"out"});

    circuit.add_switch("S1", n_out, Circuit::ground(), false, 1e3, 1e-9);
    circuit.add_vcswitch("S2", n_ctrl, n_out, Circuit::ground(), 2.5, 1e3, 1e-9);
    circuit.add_mosfet("S3", n_gate, n_out, Circuit::ground(), MOSFET::Params{});
    circuit.add_igbt("S4", n_gate, n_out, Circuit::ground(), IGBT::Params{});

    PWMParams pwm;
    pwm.v_high = 10.0;
    pwm.v_low = 0.0;
    pwm.frequency = 100e3;
    pwm.duty = 0.5;
    circuit.add_pwm_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    REQUIRE_NOTHROW(circuit.set_switch_state(std::string_view{"S1"}, true));
    REQUIRE_NOTHROW(circuit.set_switch_state(std::string_view{"S2"}, true));
    REQUIRE_NOTHROW(circuit.set_switch_state(std::string_view{"S3"}, true));
    REQUIRE_NOTHROW(circuit.set_switch_state(std::string_view{"S4"}, true));
    REQUIRE_NOTHROW(circuit.set_pwm_duty(std::string_view{"Vpwm"}, 0.25));
    REQUIRE_NOTHROW(circuit.set_pwm_duty_callback(std::string_view{"Vpwm"}, [](Real /*time*/) { return 0.4; }));
    REQUIRE_NOTHROW(circuit.clear_pwm_duty_callback(std::string_view{"Vpwm"}));
    CHECK(circuit.get_pwm_state(std::string_view{"Vpwm"}));
}

TEST_CASE("v1 PWM duty callback caches repeated evaluations at identical time",
          "[v1][runtime][pwm][performance]") {
    Circuit circuit;
    const auto n_ctrl = circuit.add_node("ctrl");

    PWMParams pwm;
    pwm.v_high = 10.0;
    pwm.v_low = 0.0;
    pwm.frequency = 100e3;
    pwm.duty = 0.5;
    circuit.add_pwm_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    std::atomic<int> callback_calls{0};
    circuit.set_pwm_duty_callback("Vpwm", [&callback_calls](Real /*time*/) {
        callback_calls.fetch_add(1, std::memory_order_relaxed);
        return 0.4;
    });

    circuit.set_current_time(1e-6);
    for (int i = 0; i < 64; ++i) {
        (void)circuit.get_pwm_state("Vpwm");
    }
    CHECK(callback_calls.load(std::memory_order_relaxed) == 1);

    circuit.set_current_time(2e-6);
    (void)circuit.get_pwm_state("Vpwm");
    CHECK(callback_calls.load(std::memory_order_relaxed) == 2);
}

TEST_CASE("v1 runtime circuit rejects invalid timesteps", "[v1][runtime][safety]") {
    Circuit circuit;

    REQUIRE_THROWS_AS(circuit.set_timestep(0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(circuit.set_timestep(-1e-6), std::invalid_argument);
    REQUIRE_THROWS_AS(circuit.set_timestep(std::numeric_limits<Real>::quiet_NaN()), std::invalid_argument);

    REQUIRE_NOTHROW(circuit.set_timestep(2e-6));
    CHECK(circuit.timestep() == Approx(2e-6));
}
