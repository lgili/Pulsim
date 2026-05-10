// =============================================================================
// Test: PWL segment-primary path contract (Phase 3 of refactor-pwl-switching-engine)
// =============================================================================
//
// Phase 3 obligation: when the segment engine is admissible and the topology
// is stable, the Simulator step path advances each step via a single linear
// solve of the Tustin discretization without invoking Newton iteration.
// These tests assert the contract end-to-end against the compiled `pulsim`
// runtime (not just the assembly primitives covered by Phase 2 tests).
//
// Key telemetry signals used here:
//   * `state_space_primary_steps`         — count of segment-primary advances
//   * `dae_fallback_steps`                — count of Newton-DAE advances
//   * `newton_iterations_total`           — aggregate iterations across all
//                                           steps; each segment-primary step
//                                           contributes exactly one (the
//                                           linear solve), DAE steps usually
//                                           contribute several.
//   * `linear_factor_cache_hits / misses` — factorization reuse counters.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

// Build a passive RC low-pass with a DC voltage source. No switching devices
// → trivially admissible to the PWL segment engine; topology is invariant
// across the whole run.
[[nodiscard]] Circuit make_rc_circuit(Real V_src = 5.0,
                                      Real R = 1e3,
                                      Real C = 1e-6) {
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_resistor("R1", in, out, R);
    ckt.add_capacitor("C1", out, Circuit::ground(), C, 0.0);
    ckt.add_voltage_source("V1", in, Circuit::ground(), V_src);
    return ckt;
}

}  // namespace

// -----------------------------------------------------------------------------
// Phase 3 contract: every step uses segment-primary on a passive RC
// -----------------------------------------------------------------------------

TEST_CASE("PWL segment-primary fires for 100% of steps on passive RC",
          "[v1][pwl][phase3][contract]") {
    Circuit circuit = make_rc_circuit();

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;       // fix dt → stable matrix → cache hits
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);

    // Production path: DC OP first to obtain a state consistent with the
    // algebraic V-source constraint, then transient from that state.
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    REQUIRE(dc.newton_result.solution.size() == circuit.system_size());

    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(run.total_steps >= 1);

    // Phase 3 contract — every accepted step took the segment-primary path.
    CHECK(run.backend_telemetry.state_space_primary_steps == run.total_steps);
    CHECK(run.backend_telemetry.dae_fallback_steps == 0);

    // Each segment-primary step records `iterations = 1` (the Tustin linear
    // solve) — Newton's iterative loop is not invoked for stable-topology
    // PWL steps, so the aggregate stays bounded by the step count.
    CHECK(run.newton_iterations_total <= run.total_steps);
    CHECK(run.backend_telemetry.nonlinear_iterations <= run.total_steps);

    // Linear factor cache should reuse: first step misses (initial build),
    // every subsequent step hits the same matrix hash under fixed dt.
    CHECK(run.backend_telemetry.linear_factor_cache_hits >=
          run.total_steps - run.backend_telemetry.linear_factor_cache_misses);
}

// -----------------------------------------------------------------------------
// Phase 3 contract: PWL switch in Ideal mode keeps segment-primary path active
// -----------------------------------------------------------------------------

TEST_CASE("PWL segment-primary handles switching device in Ideal mode",
          "[v1][pwl][phase3][switch]") {
    // RC with an ideal switch in series. Without an event scheduler (Phase 4)
    // the switch state is fixed to whatever we commit before run_transient.
    // For this test we pin the switch closed so the circuit reduces to a
    // simple RC + small `Ron` series resistance.
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto mid = ckt.add_node("mid");
    auto out = ckt.add_node("out");

    constexpr Real V_src = 5.0;
    ckt.add_voltage_source("V1", in, Circuit::ground(), V_src);
    ckt.add_switch("SW1", in, mid, /*closed=*/false, /*g_on=*/1e6, /*g_off=*/1e-12);
    ckt.add_resistor("R1", mid, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    // Opt the switch into the new PWL stamping path and pin it closed.
    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
    ckt.set_pwl_state("SW1", true);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    // Pin BDF1: auto_transient_profile rewrites Trapezoidal → TRBDF2 for
    // switching topologies, and TRBDF2's solve path bypasses solve_segment_primary.
    // BDF1 stays unmodified, so solve_step routes through the segment engine.
    opts.integrator = Integrator::BDF1;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(run.total_steps >= 1);

    // Topology bitmask is non-zero (one closed switch).
    CHECK(ckt.pwl_topology_bitmask() == 0b1);

    // Phase 3 contract on a switching device: still 100% segment-primary
    // because topology is stable (no event scheduler flips state).
    CHECK(run.backend_telemetry.state_space_primary_steps == run.total_steps);
    CHECK(run.backend_telemetry.dae_fallback_steps == 0);
    CHECK(run.newton_iterations_total <= run.total_steps);
}

// -----------------------------------------------------------------------------
// Default Auto mode resolves to Behavioral → DAE fallback (regression guard)
// -----------------------------------------------------------------------------

TEST_CASE("PWL segment-primary stays inactive when devices are in Auto/Behavioral mode",
          "[v1][pwl][phase3][backward_compat]") {
    // Same circuit as the previous test, but leave SwitchingMode at default
    // (Auto → Behavioral). The segment engine must refuse admissibility and
    // the run should complete via the legacy DAE Newton path.
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto mid = ckt.add_node("mid");
    auto out = ckt.add_node("out");

    ckt.add_voltage_source("V1", in, Circuit::ground(), 5.0);
    ckt.add_switch("SW1", in, mid, /*closed=*/true, 1e6, 1e-12);
    ckt.add_resistor("R1", mid, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);
    // Note: no `set_switching_mode_for_all` — default Auto → Behavioral.

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
    opts.dt = 1e-6;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    // BDF1 keeps solve_step on the segment-primary lambda path so we can
    // observe the admissibility refusal via segment_non_admissible_steps.
    opts.integrator = Integrator::BDF1;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);

    // Backward compat: every step takes the DAE fallback path.
    CHECK(run.backend_telemetry.dae_fallback_steps == run.total_steps);
    CHECK(run.backend_telemetry.state_space_primary_steps == 0);
    CHECK(run.backend_telemetry.segment_non_admissible_steps == run.total_steps);
}

// -----------------------------------------------------------------------------
// Adaptive timestep + segment-primary: LTE control still works
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Phase 4 contract: event scheduler commits PWL transitions during transient
// -----------------------------------------------------------------------------

TEST_CASE("PWL event scheduler flips VCSwitch when control crosses threshold",
          "[v1][pwl][phase4][events]") {
    Circuit ckt;
    auto ctrl = ckt.add_node("ctrl");
    auto vin = ckt.add_node("vin");
    auto out = ckt.add_node("out");

    // 12 V DC rail through a voltage-controlled switch into an RC load.
    ckt.add_voltage_source("Vdc", vin, Circuit::ground(), 12.0);
    ckt.add_vcswitch("S1", ctrl, vin, out, /*v_threshold=*/2.5,
                     /*g_on=*/1e3, /*g_off=*/1e-9);
    ckt.add_resistor("R1", out, Circuit::ground(), 100.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    // Pulse on the control node: 0V → 5V at t = 0.5 µs.
    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse = 5.0;
    pulse.t_delay = 0.5e-6;
    pulse.t_rise = 1e-9;
    pulse.t_fall = 1e-9;
    pulse.t_width = 5e-6;
    pulse.period = 20e-6;
    ckt.add_pulse_voltage_source("Vctrl", ctrl, Circuit::ground(), pulse);

    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
    ckt.set_pwl_state("S1", false);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 4e-6;
    opts.dt = 2e-7;          // step well below pulse rise edge
    opts.dt_min = 1e-12;
    opts.dt_max = 2e-7;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = Integrator::BDF1;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);

    // The PWL event scheduler should have flipped S1 to closed during the
    // run (control voltage crossed 2.5 V around t = 0.5 µs). The Circuit's
    // committed state at end of run reflects the final commutation.
    CHECK(ckt.pwl_topology_bitmask() == 0b1);
    REQUIRE_FALSE(run.events.empty());

    // First event should be a SwitchOn for S1 near the pulse edge.
    const auto& first = run.events.front();
    CHECK(first.component == "S1");
    CHECK(first.type == SimulationEventType::SwitchOn);
    CHECK(first.time >= pulse.t_delay);
    CHECK(first.time <= pulse.t_delay + opts.dt * 2.0);

    // Segment-primary still served every step (event scheduler is a post-step
    // commit, the step itself uses the cached topology). The legacy
    // VCSwitch bisection helper may invoke solve_step recursively, so
    // segment-primary count can exceed the accepted-step count by a few.
    CHECK(run.backend_telemetry.state_space_primary_steps >= run.total_steps);
    CHECK(run.backend_telemetry.dae_fallback_steps == 0);

    // Phase 6 telemetry: pulse only crosses threshold once within tstop, so
    // exactly one transition with one device commutation should be recorded.
    CHECK(run.backend_telemetry.pwl_topology_transitions == 1);
    CHECK(run.backend_telemetry.pwl_event_commutations == 1);
}

TEST_CASE("PWL segment-primary preserves LTE adaptive timestep behavior",
          "[v1][pwl][phase3][adaptive_lte]") {
    Circuit circuit = make_rc_circuit(/*V_src=*/5.0, /*R=*/1e3, /*C=*/1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-5;
    opts.adaptive_timestep = true;        // let the LTE controller pick dt
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(run.total_steps >= 1);

    // Even under variable dt, the segment-primary path should still serve
    // every accepted step (topology is stable). Cache hit rate may drop
    // because the matrix hash changes when dt changes — that is acceptable
    // until refactor-linear-solver-cache rekeys the cache to (sparsity,
    // topology) instead of including dt-scaled values.
    CHECK(run.backend_telemetry.state_space_primary_steps == run.total_steps);
    CHECK(run.backend_telemetry.dae_fallback_steps == 0);
    CHECK(run.newton_iterations_total <= run.total_steps);
}

// -----------------------------------------------------------------------------
// Phase 5: simulation.switching_mode threads through to circuit + segment engine
// -----------------------------------------------------------------------------

TEST_CASE("Phase 5: SimulationOptions.switching_mode propagates to Circuit",
          "[v1][pwl][phase5][options]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    ckt.add_resistor("R1", a, b, 1e3);
    ckt.add_capacitor("C1", b, Circuit::ground(), 1e-6, 0.0);

    SECTION("default Auto") {
        SimulationOptions opts;
        Simulator sim(ckt, opts);
        REQUIRE(ckt.default_switching_mode() == SwitchingMode::Auto);
    }

    SECTION("explicit Ideal") {
        SimulationOptions opts;
        opts.switching_mode = SwitchingMode::Ideal;
        Simulator sim(ckt, opts);
        REQUIRE(ckt.default_switching_mode() == SwitchingMode::Ideal);
    }

    SECTION("explicit Behavioral") {
        SimulationOptions opts;
        opts.switching_mode = SwitchingMode::Behavioral;
        Simulator sim(ckt, opts);
        REQUIRE(ckt.default_switching_mode() == SwitchingMode::Behavioral);
    }
}

TEST_CASE("Phase 5: switching_mode=Ideal makes Auto-mode devices admissible",
          "[v1][pwl][phase5][admissibility]") {
    // Phase 5 contract: when the user sets `simulation.switching_mode = Ideal`,
    // any switching device left at default `Auto` resolves up to `Ideal`
    // through the circuit-level default that the Simulator pushes into
    // Circuit::set_default_switching_mode. The segment engine then accepts
    // the topology without requiring per-device opt-in.
    Circuit ckt;
    auto ctrl = ckt.add_node("ctrl");
    auto t1 = ckt.add_node("t1");
    auto t2 = ckt.add_node("t2");
    ckt.add_voltage_source("Vdc", t1, Circuit::ground(), 5.0);
    ckt.add_vcswitch("S1", ctrl, t1, t2, 2.5);
    ckt.add_resistor("Rload", t2, Circuit::ground(), 100.0);
    // Note: device left at default `Auto` switching mode.

    SECTION("default options leave admissibility under Behavioral") {
        SimulationOptions opts;  // switching_mode = Auto by default
        Simulator sim(ckt, opts);
        REQUIRE(ckt.default_switching_mode() == SwitchingMode::Auto);
        // Auto + circuit_default=Behavioral → resolves Behavioral → admissibility
        // refused for the device.
        REQUIRE_FALSE(ckt.all_switching_devices_in_ideal_mode(SwitchingMode::Behavioral));
    }

    SECTION("opting into Ideal at simulation level lifts admissibility") {
        SimulationOptions opts;
        opts.switching_mode = SwitchingMode::Ideal;
        Simulator sim(ckt, opts);
        REQUIRE(ckt.default_switching_mode() == SwitchingMode::Ideal);
        // Auto device + circuit_default=Ideal → resolves Ideal → admissible.
        REQUIRE(ckt.all_switching_devices_in_ideal_mode(SwitchingMode::Ideal));
    }
}

TEST_CASE("Phase 5: YAML parser maps simulation.switching_mode",
          "[v1][pwl][phase5][yaml]") {
    auto load_mode = [](std::string_view mode_value) -> SwitchingMode {
        const std::string content =
            std::string("schema: pulsim-v1\nversion: 1\nsimulation:\n  tstop: 1e-6\n  dt: 1e-7\n  switching_mode: ") +
            std::string(mode_value) + "\ncomponents:\n  - type: resistor\n    name: R1\n    nodes: [n1, 0]\n    value: 1.0\n";
        pulsim::v1::parser::YamlParser parser;
        const auto [circuit, options] = parser.load_string(content);
        return options.switching_mode;
    };

    REQUIRE(load_mode("auto") == SwitchingMode::Auto);
    REQUIRE(load_mode("ideal") == SwitchingMode::Ideal);
    REQUIRE(load_mode("pwl") == SwitchingMode::Ideal);  // alias
    REQUIRE(load_mode("behavioral") == SwitchingMode::Behavioral);
    REQUIRE(load_mode("smooth") == SwitchingMode::Behavioral);  // alias
}

TEST_CASE("Phase 5: YAML parser rejects unknown switching_mode in strict mode",
          "[v1][pwl][phase5][yaml][strict]") {
    const std::string content =
        "schema: pulsim-v1\n"
        "version: 1\n"
        "simulation:\n"
        "  tstop: 1e-6\n"
        "  dt: 1e-7\n"
        "  switching_mode: bogus\n"
        "components:\n"
        "  - type: resistor\n"
        "    name: R1\n"
        "    nodes: [n1, 0]\n"
        "    value: 1.0\n";
    pulsim::v1::parser::YamlParserOptions parser_opts;
    parser_opts.strict = true;
    pulsim::v1::parser::YamlParser parser(parser_opts);
    parser.load_string(content);
    const auto& errors = parser.errors();
    REQUIRE_FALSE(errors.empty());
    bool found_diagnostic = false;
    for (const auto& err : errors) {
        if (err.find("switching_mode") != std::string::npos &&
            err.find("bogus") != std::string::npos) {
            found_diagnostic = true;
            break;
        }
    }
    CHECK(found_diagnostic);
}

// -----------------------------------------------------------------------------
// Phase 8: Validation suite — physical invariants (KCL, energy)
// -----------------------------------------------------------------------------

TEST_CASE("Phase 8: KCL holds at the final state of a PWL transient",
          "[v1][pwl][phase8][invariants][kcl]") {
    // RC low-pass: KCL at the `out` node says
    //   (V_out − V_in)/R + C·dV_out/dt = 0.
    // After Tustin step with consistent IC, the assembled DAE residual at
    // every non-ground node row must be ≤ tolerance — that is the property
    // that says "the simulator did not violate Kirchhoff".
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_resistor("R1", in, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);
    ckt.add_voltage_source("V1", in, Circuit::ground(), 5.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-5;
    opts.dt = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = Integrator::BDF1;
    opts.switching_mode = SwitchingMode::Ideal;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE_FALSE(run.states.empty());

    // Reconstruct the DAE residual `f = N·x_final + (M/dt)·(x_final − x_prev) − b`
    // at the final step. Node rows of f should be near zero (KCL); branch rows
    // are algebraic and equally must be near zero (V-source constraint).
    SparseMatrix M, N;
    Vector b_now, b_next;
    const Vector& x_final = run.states.back();
    const Vector& x_prev = run.states[run.states.size() - 2];
    const Real dt_used = run.time.back() - run.time[run.time.size() - 2];

    ckt.assemble_state_space(M, N, b_now, run.time[run.time.size() - 2]);
    ckt.assemble_state_space(M, N, b_next, run.time.back());

    const Vector residual_dae =
        N * x_final + (M / dt_used) * (x_final - x_prev) -
        Real{0.5} * (b_now + b_next);

    const Real tol = std::max<Real>(1e-9, x_final.lpNorm<Eigen::Infinity>() * 1e-9);
    for (Index i = 0; i < residual_dae.size(); ++i) {
        INFO("residual_dae[" << i << "] = " << residual_dae[i]);
        CHECK(std::abs(residual_dae[i]) <= tol);
    }
}

TEST_CASE("Phase 8: energy is bounded in a near-lossless LC ringing circuit",
          "[v1][pwl][phase8][invariants][energy]") {
    // Charge a capacitor through a tiny resistor onto an LC tank, then let
    // it ring. With near-zero dissipation the total stored energy
    //     E(t) = ½·C·v_C² + ½·L·i_L²
    // must be bounded across many cycles — strictly conserved in the
    // continuous limit, slowly decaying under Tustin's small numerical
    // damping for non-zero R.
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");

    constexpr Real C_val = 1e-6;        // 1 µF
    constexpr Real L_val = 1e-6;        // 1 µH → f₀ ≈ 159 kHz
    constexpr Real V0 = 5.0;            // initial cap voltage
    // Tiny series resistance to make the assembly well-conditioned without
    // appreciably damping the tank in a few cycles.
    ckt.add_resistor("R1", a, b, 1e-3);
    ckt.add_inductor("L1", b, Circuit::ground(), L_val, 0.0);
    ckt.add_capacitor("C1", a, Circuit::ground(), C_val, V0);
    const Index l_branch = ckt.num_nodes() + 0;

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-5;       // ~1.6 cycles
    opts.dt = 5e-8;
    opts.dt_min = 1e-12;
    opts.dt_max = 5e-8;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = Integrator::BDF1;
    opts.switching_mode = SwitchingMode::Ideal;  // engages the segment engine
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    Vector x0 = Vector::Zero(ckt.system_size());
    x0[a] = V0;                          // cap voltage
    x0[l_branch] = 0.0;                  // inductor current

    const auto run = sim.run_transient(x0);
    REQUIRE(run.success);
    REQUIRE(run.states.size() >= 2);

    // Initial stored energy: ½ C V0² (no inductor current at t=0).
    const Real e0 = 0.5 * C_val * V0 * V0;

    // Energy across the run must never *grow* beyond the starting value
    // (passivity guard) and must stay above some lower bound (we don't lose
    // energy faster than the small R can dissipate over the run).
    Real e_max = 0.0;
    Real e_min = std::numeric_limits<Real>::infinity();
    for (const auto& x : run.states) {
        const Real v_c = x[a];
        const Real i_l = x[l_branch];
        const Real e = 0.5 * C_val * v_c * v_c + 0.5 * L_val * i_l * i_l;
        e_max = std::max(e_max, e);
        e_min = std::min(e_min, e);
    }

    INFO("e0    = " << e0);
    INFO("e_min = " << e_min);
    INFO("e_max = " << e_max);
    CHECK(e_max <= e0 * 1.02);          // ≤ 2 % gain (numerical noise)
    CHECK(e_min >= e0 * 0.85);          // ≥ 85 % retained over ~1.6 cycles
}
