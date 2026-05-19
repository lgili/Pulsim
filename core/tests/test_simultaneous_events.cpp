// simplify-and-harden-numerical-surface — Phase 5 tests.
//
// Verifies that the PWL event bisection coalesces multiple device
// commutations that fire within bisection tolerance into a single
// atomic Newton solve. The coalescence is implemented in
// `Circuit::bisect_pwl_event_alpha` and signaled by the
// `BackendTelemetry::simultaneous_event_groups` counter.
//
// Without coalescence, three vcswitches sharing a common control
// signal would each be processed one at a time — N Newton solves
// per gate edge. With coalescence they fire atomically — one solve.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("BackendTelemetry: simultaneous_event_groups defaults to zero",
          "[pwl][simultaneous_events]") {
    BackendTelemetry t{};
    CHECK(t.simultaneous_event_groups == 0);
}

TEST_CASE("PWL bisection: 3 synchronous vcswitches coalesce into one group",
          "[pwl][simultaneous_events][3phase]") {
    // Circuit topology: three voltage-controlled switches sharing the
    // same control node `vc`. A pulse source on `vc` flips from 0 V
    // to 12 V at t = 0.5 ms, crossing the 2.5 V threshold simultaneously
    // for all three switches.
    //
    // Expected:
    //   - Exactly one PWM rising edge between t=0.4ms and t=0.6ms
    //   - At that edge, all 3 switches commute → 3 commutations,
    //     coalesced into 1 group, 1 topology transition
    //   - simultaneous_event_groups >= 1 (we only count groups of
    //     size ≥ 2, so 3 commutations qualifies)
    Circuit ckt;
    auto vc  = ckt.add_node("vc");
    auto out_a = ckt.add_node("out_a");
    auto out_b = ckt.add_node("out_b");
    auto out_c = ckt.add_node("out_c");
    auto vdc = ckt.add_node("vdc");

    // DC supply for the load side.
    ckt.add_voltage_source("V_dc", vdc, Circuit::ground(), 12.0);

    // Pulse on the control node: rising edge at t=0.5ms.
    PulseParams pp{};
    pp.v_initial = 0.0;
    pp.v_pulse   = 12.0;
    pp.t_delay   = 0.5e-3;
    pp.t_rise    = 1e-9;
    pp.t_fall    = 1e-9;
    pp.t_width   = 1e-3;
    pp.period    = 0.0;
    ckt.add_pulse_voltage_source("V_ctrl", vc, Circuit::ground(), pp);

    // Three vcswitches, each gated by the same `vc` node, threshold 2.5 V.
    // Each switch lights a separate output node from the DC supply
    // through its own load resistor.
    ckt.add_vcswitch("SW_a", vc, vdc, out_a, /*v_threshold=*/2.5,
                     /*g_on=*/1e3, /*g_off=*/1e-9, /*hysteresis=*/0.5);
    ckt.add_vcswitch("SW_b", vc, vdc, out_b, 2.5, 1e3, 1e-9, 0.5);
    ckt.add_vcswitch("SW_c", vc, vdc, out_c, 2.5, 1e3, 1e-9, 0.5);

    // Load resistors per phase.
    ckt.add_resistor("R_a", out_a, Circuit::ground(), 10.0);
    ckt.add_resistor("R_b", out_b, Circuit::ground(), 10.0);
    ckt.add_resistor("R_c", out_c, Circuit::ground(), 10.0);

    SimulationOptions opts =
        SimulationOptions::from_preset(Preset::Fast, 1e-6, 1e-3);
    opts.switching_mode = SwitchingMode::Ideal;
    opts.enable_events  = true;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    auto result = sim.run_transient();
    REQUIRE(result.success);

    INFO("pwl_event_commutations    = "
         << result.backend_telemetry.pwl_event_commutations);
    INFO("pwl_topology_transitions  = "
         << result.backend_telemetry.pwl_topology_transitions);
    INFO("simultaneous_event_groups = "
         << result.backend_telemetry.simultaneous_event_groups);

    // All three switches must commute during the run.
    CHECK(result.backend_telemetry.pwl_event_commutations >= 3);
    // At least one accepted step grouped them — the PWM rising edge
    // at t=0.5ms forces all three across threshold simultaneously.
    CHECK(result.backend_telemetry.simultaneous_event_groups >= 1);
}

TEST_CASE("PWL bisection: single isolated event does NOT count as a group",
          "[pwl][simultaneous_events][single]") {
    // Single vcswitch — gate rises once, only one commutation. The
    // coalescence step still runs but finds nothing extra to add, so
    // simultaneous_event_groups stays at zero.
    Circuit ckt;
    auto vc   = ckt.add_node("vc");
    auto out  = ckt.add_node("out");
    auto vdc  = ckt.add_node("vdc");

    ckt.add_voltage_source("V_dc", vdc, Circuit::ground(), 12.0);

    PulseParams pp{};
    pp.v_initial = 0.0;
    pp.v_pulse   = 12.0;
    pp.t_delay   = 0.5e-3;
    pp.t_rise    = 1e-9;
    pp.t_fall    = 1e-9;
    pp.t_width   = 1e-3;
    ckt.add_pulse_voltage_source("V_ctrl", vc, Circuit::ground(), pp);

    ckt.add_vcswitch("SW", vc, vdc, out, 2.5, 1e3, 1e-9, 0.5);
    ckt.add_resistor("R_load", out, Circuit::ground(), 10.0);

    SimulationOptions opts =
        SimulationOptions::from_preset(Preset::Fast, 1e-6, 1e-3);
    opts.switching_mode = SwitchingMode::Ideal;
    opts.enable_events  = true;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    auto result = sim.run_transient();
    REQUIRE(result.success);

    INFO("pwl_event_commutations    = "
         << result.backend_telemetry.pwl_event_commutations);
    INFO("simultaneous_event_groups = "
         << result.backend_telemetry.simultaneous_event_groups);

    // Single switch: at least one commutation, zero simultaneous groups.
    CHECK(result.backend_telemetry.pwl_event_commutations >= 1);
    CHECK(result.backend_telemetry.simultaneous_event_groups == 0);
}
