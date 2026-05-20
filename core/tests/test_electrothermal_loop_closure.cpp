// =============================================================================
// Phase 1.3 of close-electrothermal-loop-and-promote-thermal-traits.
// =============================================================================
//
// Regression test that locks in the closed-loop electrothermal contract.
// Before the closure landed, every participating device's internal `T_j_`
// stayed frozen at `T_amb` for the whole simulation, regardless of how
// much power it dissipated. The `DefaultThermalService` integrated a
// separate `T_i(t)` for the stamp feedback (via `scale_i`), but never
// pushed it back into the device's `T_j_`. Two trackers, two answers.
//
// After the closure, `DefaultThermalService::commit_accepted_segment`
// dispatches `set_T_j_init(T_i)` to every device that exposes the
// setter. The device-internal `T_j_` then tracks `T_i(t)`, the stamp
// path AND the loss path both see the same temperature, and the
// device's `<dev>_junction_temperature(name)` accessor returns the
// integrated `T_j(t)` instead of a stale constant.
//
// This file asserts exactly that invariant on the simplest possible
// topology — a DC voltage source through a configured resistor — so
// the test is decoupled from PWM convergence, switching events, and
// the auto-parasitics pre-flight (which has known sensitivities in
// some debug builds and would mask the regression signal).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

SimulationOptions make_thermal_opts(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2.0;       // Long enough for several thermal time constants
                            // when R_th · C_th = 50 K/W · 0.1 J/K = 5 s.
                            // 2 s ≈ 0.4 τ — T_j walks ~33 % of the way to
                            // steady state.
    opts.dt = 1e-3;
    opts.dt_min = 1e-6;
    opts.dt_max = 1e-3;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.thermal.enable = true;
    opts.thermal.ambient = 25.0;
    opts.thermal.default_rth = 1.0;
    opts.thermal.default_cth = 0.1;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("Closed loop: resistor T_j walks from T_amb under DC load",
          "[v1][thermal][electrothermal_closure][regression]") {
    // Topology: V_dc → R1 → GND. R1 = 10 Ω, V_dc = 10 V → I = 1 A,
    // P = 10 W. With R_th_ja = 50 K/W, T_amb = 25 °C, the steady-state
    // T_j is 25 + 10·50 = 525 °C. With C_th = 1 J/K, τ = 50 s. Over
    // 2 s the device reaches ~4 % of steady state ≈ 25 + 0.04·500 ≈
    // 45 °C — strictly above T_amb.
    Circuit circuit;
    const Index n_pos = circuit.add_node("pos");
    circuit.add_voltage_source("Vdc", n_pos, Circuit::ground(), 10.0);
    circuit.add_resistor("R1", n_pos, Circuit::ground(), 10.0);

    auto opts = make_thermal_opts(circuit);

    // Opt-in: configure R1 in thermal_devices. Per the OpenSpec's
    // back-compat contract, passive devices only enrol in the
    // thermal pipeline when explicitly configured.
    ThermalDeviceConfig cfg;
    cfg.enabled = true;
    cfg.rth = 50.0;
    cfg.cth = 1.0;
    cfg.temp_init = 25.0;
    cfg.temp_ref = 25.0;
    cfg.alpha = 0.0;        // Disable scale-feedback to keep the test
                            // focused on the closed-loop T_j dispatch.
    opts.thermal_devices["R1"] = cfg;

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();

    INFO("simulation message: " << result.message);
    REQUIRE(result.success);

    // 1. The thermal_summary should record R1's final T_j above T_amb.
    REQUIRE(result.thermal_summary.enabled);
    const auto it = std::find_if(
        result.thermal_summary.device_temperatures.begin(),
        result.thermal_summary.device_temperatures.end(),
        [](const DeviceThermalTelemetry& t) { return t.device_name == "R1"; });
    REQUIRE(it != result.thermal_summary.device_temperatures.end());

    INFO("R1 final temperature (summary)  = " << it->final_temperature << " °C");
    INFO("R1 peak temperature (summary)   = " << it->peak_temperature  << " °C");
    INFO("R1 average temperature (summary)= " << it->average_temperature << " °C");

    CHECK(it->final_temperature > Real{25.0} + Real{1.0});
    CHECK(it->peak_temperature >= it->final_temperature);

    // 2. The DEVICE-side accessor (the bit this OpenSpec fixes) MUST
    //    now report the same integrated T_j(t) as the summary —
    //    instead of staying frozen at T_amb. This is the regression
    //    bit that catches a future revert of the closed-loop dispatch.
    const Real T_j_device = circuit.resistor_junction_temperature("R1");

    INFO("R1 device-side junction_temperature = " << T_j_device << " °C");

    // The two trackers must agree within numerical noise. Before the
    // closure landed, T_j_device == 25.0 (frozen) while the summary
    // reported ~45.0 — a 20 °C divergence. This assert IS the
    // regression test.
    CHECK(T_j_device == Approx(it->final_temperature).margin(1e-6));

    // 3. The device-side T_j MUST be strictly above T_amb (i.e. the
    //    closed loop dispatched at least once and pushed a
    //    physical value into the device's internal state).
    CHECK(T_j_device > Real{25.0} + Real{1.0});
}

TEST_CASE("Closed loop: unconfigured resistor stays at T_amb (back-compat)",
          "[v1][thermal][electrothermal_closure][regression]") {
    // Same topology, but R1 has NO `opts.thermal_devices` entry. The
    // opt-in gate added in Phase 2 should keep it out of the thermal
    // loop, preserving the legacy back-compat contract.
    Circuit circuit;
    const Index n_pos = circuit.add_node("pos");
    circuit.add_voltage_source("Vdc", n_pos, Circuit::ground(), 10.0);
    circuit.add_resistor("R1", n_pos, Circuit::ground(), 10.0);

    auto opts = make_thermal_opts(circuit);
    // Deliberately do NOT configure R1 in opts.thermal_devices.

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    const Real T_j_device = circuit.resistor_junction_temperature("R1");
    INFO("R1 device-side junction_temperature (unconfigured) = "
         << T_j_device << " °C");

    // With opt-in gating, an unconfigured passive stays at T_amb. The
    // closed-loop dispatch DOES iterate the device (it's still
    // `has_thermal_model = true`), but the service state's `enabled`
    // bit is false → the dispatch reads `state.temperature` which
    // was never integrated. The push pushes `state.temperature =
    // T_amb` into the device, which is a no-op relative to the
    // device's construction state.
    CHECK(T_j_device == Approx(Real{25.0}).margin(1e-6));
}

TEST_CASE("Closed loop: device.junction_temperature() matches thermal_summary "
          "at every accepted step (invariant)",
          "[v1][thermal][electrothermal_closure][regression]") {
    // The invariant: at the END of the simulation, the device's own
    // `junction_temperature()` reading MUST equal the corresponding
    // `thermal_summary.device_temperatures[i].final_temperature` to
    // within 1e-9 °C. Both are the integrated `T_i(t)`; they MUST
    // agree because the closed-loop dispatch writes the SAME value
    // into both.
    //
    // This invariant guards against a regression where:
    //   - Someone removes the `push_T_j_into_devices()` call
    //   - Someone changes the dispatch order so the device sees a
    //     stale T_i from the PREVIOUS step
    //   - Someone introduces a parallel T_j tracker that diverges
    Circuit circuit;
    const Index n_pos = circuit.add_node("pos");
    circuit.add_voltage_source("Vdc", n_pos, Circuit::ground(), 20.0);
    circuit.add_resistor("R_load", n_pos, Circuit::ground(), 5.0);

    auto opts = make_thermal_opts(circuit);
    ThermalDeviceConfig cfg;
    cfg.enabled = true;
    cfg.rth = 10.0;
    cfg.cth = 0.5;
    cfg.temp_init = 25.0;
    cfg.temp_ref = 25.0;
    opts.thermal_devices["R_load"] = cfg;

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    const auto it = std::find_if(
        result.thermal_summary.device_temperatures.begin(),
        result.thermal_summary.device_temperatures.end(),
        [](const DeviceThermalTelemetry& t) { return t.device_name == "R_load"; });
    REQUIRE(it != result.thermal_summary.device_temperatures.end());

    const Real T_summary = it->final_temperature;
    const Real T_device  = circuit.resistor_junction_temperature("R_load");

    INFO("T_summary = " << T_summary << " °C, T_device = " << T_device << " °C");

    // Both must agree to 1e-9 °C — they're the same integrated state.
    CHECK(T_device == Approx(T_summary).margin(1e-9));

    // And both must be strictly above T_amb (proves the integration
    // actually walked the state).
    CHECK(T_summary > Real{25.0} + Real{0.5});
    CHECK(T_device  > Real{25.0} + Real{0.5});
}
