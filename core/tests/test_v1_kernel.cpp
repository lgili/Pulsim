#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/simulation.hpp"

#include <algorithm>

using namespace pulsim::v1;

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
    Vector x0 = Vector::Zero(circuit.system_size());
    auto result = sim.run_transient(x0);

    REQUIRE(result.success);
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
    Vector x0 = Vector::Zero(circuit.system_size());
    auto result = sim.run_transient(x0);

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
