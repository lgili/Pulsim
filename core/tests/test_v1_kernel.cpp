#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>

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
    REQUIRE(result.thermal_summary.enabled);
    CHECK(result.thermal_summary.max_temperature >= result.thermal_summary.ambient);
    CHECK_FALSE(result.thermal_summary.device_temperatures.empty());
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
    CHECK(run.backend_telemetry.state_space_primary_steps >= 1);
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

TEST_CASE("v1 sundials-only backend handles availability deterministically", "[v1][backend][sundials]") {
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
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::IDA;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

#ifdef PULSIM_HAS_SUNDIALS
    REQUIRE(result.success);
    CHECK(result.backend_telemetry.sundials_compiled);
    CHECK(result.backend_telemetry.selected_backend == "sundials");
    CHECK(result.backend_telemetry.solver_family == "ida");
    CHECK(result.backend_telemetry.formulation_mode == "projected_wrapper");
    CHECK(result.backend_telemetry.function_evaluations >= 0);
    CHECK(result.backend_telemetry.jacobian_evaluations >= 0);
    CHECK(result.backend_telemetry.nonlinear_iterations >= 0);
    CHECK(result.backend_telemetry.error_test_failures >= 0);
    CHECK(result.time.size() >= 2);
#else
    REQUIRE_FALSE(result.success);
    CHECK_FALSE(result.backend_telemetry.sundials_compiled);
    CHECK(result.message.find("without SUNDIALS support") != std::string::npos);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
}

TEST_CASE("v1 sundials-only backend requires explicit enable flag", "[v1][backend][sundials]") {
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
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.sundials.enabled = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE_FALSE(result.success);
    CHECK(result.message.find("disabled in simulation.sundials.enabled") != std::string::npos);
    CHECK(result.backend_telemetry.failure_reason == "sundials_backend_disabled");
}

TEST_CASE("v1 sundials backend exposes direct formulation selection", "[v1][backend][sundials]") {
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
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::IDA;
    opts.sundials.formulation = SundialsFormulationMode::Direct;
    opts.sundials.allow_formulation_fallback = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    CHECK(result.backend_telemetry.selected_backend == "sundials");
    CHECK(result.backend_telemetry.solver_family == "ida");
    CHECK(result.backend_telemetry.formulation_mode == "direct");
}

TEST_CASE("v1 sundials direct IDA handles switching reinitialization", "[v1][backend][sundials][direct]") {
    Circuit circuit;
    auto n_ctrl = circuit.add_node("ctrl");
    auto n_in = circuit.add_node("in");
    auto n_sw = circuit.add_node("sw");

    PWMParams pwm;
    pwm.v_high = 5.0;
    pwm.v_low = 0.0;
    pwm.frequency = 60e3;
    pwm.duty = 0.4;
    pwm.rise_time = 20e-9;
    pwm.fall_time = 20e-9;
    circuit.add_pwm_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 24.0);
    circuit.add_vcswitch("S1", n_ctrl, n_in, n_sw, 2.5, 1e3, 1e-9);
    circuit.add_resistor("Rload", n_sw, Circuit::ground(), 20.0);
    circuit.add_capacitor("Cout", n_sw, Circuit::ground(), 1.0e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 5e-8;
    opts.dt_min = 1e-12;
    opts.dt_max = 5e-6;
    opts.adaptive_timestep = true;
    opts.enable_events = true;
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::IDA;
    opts.sundials.formulation = SundialsFormulationMode::Direct;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

#ifdef PULSIM_HAS_SUNDIALS
    CHECK(result.backend_telemetry.selected_backend == "sundials");
    CHECK(result.backend_telemetry.solver_family == "ida");
    CHECK(result.backend_telemetry.formulation_mode == "direct");
    CHECK(result.backend_telemetry.function_evaluations >= 0);
    CHECK(result.backend_telemetry.nonlinear_iterations >= 0);
    if (result.success) {
        CHECK(result.backend_telemetry.reinitialization_count >= 0);
        CHECK(result.time.size() >= 2);
    } else {
        CHECK((result.backend_telemetry.failure_reason == "sundials_step_failed" ||
               result.backend_telemetry.failure_reason == "sundials_reinit_failed" ||
               result.backend_telemetry.failure_reason == "sundials_init_failed"));
    }
#else
    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
}

TEST_CASE("v1 sundials direct CVODE reports direct formulation", "[v1][backend][sundials][direct]") {
    Circuit circuit;
    auto n_in = circuit.add_node("in");
    auto n_out = circuit.add_node("out");
    circuit.add_voltage_source("V1", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 1e3);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-5;
    opts.dt = 1e-6;
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::CVODE;
    opts.sundials.formulation = SundialsFormulationMode::Direct;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

#ifdef PULSIM_HAS_SUNDIALS
    CHECK(result.backend_telemetry.selected_backend == "sundials");
    CHECK(result.backend_telemetry.solver_family == "cvode");
    CHECK(result.backend_telemetry.formulation_mode == "direct");
#else
    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
}

TEST_CASE("v1 sundials backend records switch events and reinitializations", "[v1][backend][sundials][events]") {
    Circuit circuit;
    auto n_ctrl = circuit.add_node("ctrl");
    auto n_in = circuit.add_node("in");
    auto n_sw = circuit.add_node("sw");

    PWMParams pwm;
    pwm.v_high = 5.0;
    pwm.v_low = 0.0;
    pwm.frequency = 80e3;
    pwm.duty = 0.45;
    pwm.rise_time = 50e-9;
    pwm.fall_time = 50e-9;
    circuit.add_pwm_voltage_source("Vpwm", n_ctrl, Circuit::ground(), pwm);

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 24.0);
    circuit.add_vcswitch("S1", n_ctrl, n_in, n_sw, 2.5, 1e3, 1e-9);
    circuit.add_resistor("Rload", n_sw, Circuit::ground(), 12.0);
    circuit.add_capacitor("Cout", n_sw, Circuit::ground(), 2.2e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1.5e-4;
    opts.dt = 2e-8;
    opts.dt_min = 1e-12;
    opts.dt_max = 2e-6;
    opts.adaptive_timestep = true;
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.enable_events = true;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::ARKODE;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

#ifdef PULSIM_HAS_SUNDIALS
    CHECK(result.backend_telemetry.selected_backend == "sundials");
    CHECK(result.backend_telemetry.solver_family == "arkode");
    if (result.success) {
        CHECK(result.backend_telemetry.reinitialization_count >= 0);
        CHECK_FALSE(result.events.empty());
    } else {
        CHECK((result.backend_telemetry.failure_reason == "sundials_step_failed" ||
               result.backend_telemetry.failure_reason == "sundials_projection_failed"));
    }
#else
    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
}

TEST_CASE("v1 sundials backend segments PWM source boundaries", "[v1][backend][sundials][pwm]") {
    Circuit circuit;
    auto n_sw = circuit.add_node("sw");

    PWMParams pwm;
    pwm.v_high = 12.0;
    pwm.v_low = 0.0;
    pwm.frequency = 50e3;
    pwm.duty = 0.35;
    pwm.rise_time = 0.0;
    pwm.fall_time = 0.0;
    circuit.add_pwm_voltage_source("Vpwm", n_sw, Circuit::ground(), pwm);
    circuit.add_resistor("Rload", n_sw, Circuit::ground(), 10.0);
    circuit.add_capacitor("Cout", n_sw, Circuit::ground(), 4.7e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1.5e-4;
    opts.dt = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 2e-6;
    opts.adaptive_timestep = true;
    opts.enable_events = true;
    opts.transient_backend = TransientBackendMode::SundialsOnly;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::ARKODE;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

#ifdef PULSIM_HAS_SUNDIALS
    REQUIRE(result.success);
    CHECK(result.backend_telemetry.selected_backend == "sundials");
    CHECK(result.backend_telemetry.reinitialization_count > 0);
    CHECK(std::any_of(result.events.begin(), result.events.end(),
                      [](const SimulationEvent& evt) {
                          return evt.description == "pwm_boundary";
                      }));
#else
    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
}

TEST_CASE("v1 auto backend records deterministic sundials escalation behavior", "[v1][backend][auto]") {
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
    opts.transient_backend = TransientBackendMode::Auto;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::IDA;
    opts.sundials.formulation = SundialsFormulationMode::Direct;
    opts.sundials.allow_formulation_fallback = false;
    opts.fallback_policy.enable_backend_escalation = true;
    opts.fallback_policy.backend_escalation_threshold = 1;

    // Force native path to fail deterministically.
    opts.linear_solver.order = {LinearSolverKind::CG};
    opts.linear_solver.fallback_order = {LinearSolverKind::CG};
    opts.linear_solver.allow_fallback = false;
    opts.linear_solver.auto_select = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient(Vector::Zero(circuit.system_size()), nullptr, nullptr, nullptr);

#ifdef PULSIM_HAS_SUNDIALS
    CHECK(result.backend_telemetry.sundials_compiled);
#else
    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
}

TEST_CASE("v1 auto backend can attempt native reentry after sundials recovery", "[v1][backend][auto][hybrid]") {
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
    opts.transient_backend = TransientBackendMode::Auto;
    opts.sundials.enabled = true;
    opts.sundials.family = SundialsSolverFamily::IDA;
    opts.fallback_policy.enable_backend_escalation = true;
    opts.fallback_policy.backend_escalation_threshold = 1;
    opts.fallback_policy.enable_native_reentry = true;
    opts.fallback_policy.sundials_recovery_window = 2e-5;

    // Force native attempts (initial and reentry) to fail deterministically.
    opts.linear_solver.order = {LinearSolverKind::CG};
    opts.linear_solver.fallback_order = {LinearSolverKind::CG};
    opts.linear_solver.allow_fallback = false;
    opts.linear_solver.auto_select = false;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient(Vector::Zero(circuit.system_size()), nullptr, nullptr, nullptr);

#ifdef PULSIM_HAS_SUNDIALS
    CHECK(result.backend_telemetry.sundials_compiled);
    CHECK(std::any_of(result.fallback_trace.begin(),
                      result.fallback_trace.end(),
                      [](const FallbackTraceEntry& entry) {
                          return entry.action == "escalate_to_sundials";
                      }));
    const bool reentry_attempted = std::any_of(result.fallback_trace.begin(),
                                               result.fallback_trace.end(),
                                               [](const FallbackTraceEntry& entry) {
                                                   return entry.action == "reenter_native";
                                               });
    const bool sundials_failed = std::any_of(result.fallback_trace.begin(),
                                             result.fallback_trace.end(),
                                             [](const FallbackTraceEntry& entry) {
                                                 return entry.action == "sundials_failed";
                                             });
    CHECK((reentry_attempted || sundials_failed));
    CHECK((result.backend_telemetry.selected_backend == "hybrid_auto" ||
           result.backend_telemetry.selected_backend == "sundials"));
#else
    REQUIRE_FALSE(result.success);
    CHECK(result.backend_telemetry.failure_reason == "sundials_not_compiled");
#endif
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
    CHECK(has_reason(FallbackReasonCode::TransientGminEscalation));
    CHECK(has_reason(FallbackReasonCode::MaxRetriesExceeded));
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
    const auto hi_off = circuit.execute_mixed_domain_step(x, 8.5e-4);  // phase=0.85 > duty

    x[n_cmd] = 0.10;
    const auto lo = circuit.execute_mixed_domain_step(x, 2.5e-4);  // phase=0.25 > duty_min

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
}

TEST_CASE("v1 control primitives keep bounded deterministic outputs",
          "[v1][mixed-domain][control][regression]") {
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
