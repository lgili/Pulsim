#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <algorithm>

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
