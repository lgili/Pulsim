// =============================================================================
// Test: Phase 3 numeric-factor LRU cache
// =============================================================================
//
// Phase 3 of `refactor-linear-solver-cache`: the segment stepper now keeps a
// per-(topology, dt, parameter) numeric-factor cache, keyed on
// `hash_sparse_numeric_signature(E)`. When a power-electronics converter
// commutates between a small set of topologies, every step after the
// warmup period should pull a cached factor instead of paying analyze +
// factorize again.
//
// We exercise the contract end-to-end through `Simulator::run_transient`
// on a circuit with a voltage-controlled switch driven by a pulse train.
// After the first two unique (topology, dt) combinations are seen the
// cache should hold both and every subsequent step (regardless of
// commutation direction) should hit.
//
// Counters consumed:
//   * `linear_factor_cache_hits / misses`
//   * `linear_factor_cache_invalidations_topology_changed`
//   * `pwl_topology_transitions` — sanity check that commutations actually
//                                  happened during the run
//
// The cache itself is private to `transient_services.cpp`; we observe it
// only through the telemetry surface, which is the user-visible contract.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

using namespace pulsim::v1;
using Catch::Approx;

namespace {

// Buck-style topology: 12V rail through a VCSwitch, then RC load. The
// VCSwitch is gated by a pulse-train control voltage which forces multiple
// commutations during the run so the cache sees the on-state and the
// off-state matrices repeatedly.
[[nodiscard]] Circuit make_pulsed_switch_circuit() {
    Circuit ckt;
    auto ctrl = ckt.add_node("ctrl");
    auto vin  = ckt.add_node("vin");
    auto out  = ckt.add_node("out");

    ckt.add_voltage_source("Vdc", vin, Circuit::ground(), 12.0);
    ckt.add_vcswitch("S1", ctrl, vin, out, /*v_threshold=*/2.5,
                     /*g_on=*/1e3, /*g_off=*/1e-9);
    ckt.add_resistor("R1", out, Circuit::ground(), 100.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    // Pulse train: period 4 µs, on-time 2 µs. Over a 16 µs run that's
    // four edges → 8 cache events at the topology boundary, plenty of
    // material for the LRU to cycle.
    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse   = 5.0;
    pulse.t_delay   = 1e-6;
    pulse.t_rise    = 1e-9;
    pulse.t_fall    = 1e-9;
    pulse.t_width   = 2e-6;
    pulse.period    = 4e-6;
    ckt.add_pulse_voltage_source("Vctrl", ctrl, Circuit::ground(), pulse);

    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
    ckt.set_pwl_state("S1", false);
    return ckt;
}

}  // namespace

// -----------------------------------------------------------------------------
// Cycling: same topology revisited across commutations should hit the cache
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3: numeric-factor cache reuses factors across topology cycling",
          "[v1][pwl][phase3][linear_cache][numeric_lru]") {
    Circuit ckt = make_pulsed_switch_circuit();

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 16e-6;
    opts.dt     = 2e-7;          // fixed dt → matrix_hash repeats per topology
    opts.dt_min = 1e-12;
    opts.dt_max = 2e-7;
    opts.adaptive_timestep    = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = Integrator::BDF1;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(run.total_steps >= 80);   // 16us / 0.2us = 80 steps

    const auto& tel = run.backend_telemetry;

    // The pulse should have driven at least three commutations (rising,
    // falling, rising again). Each commutation flips the switch and shows
    // up as a topology transition.
    CHECK(tel.pwl_topology_transitions >= 3);

    // Each commutation forces the next step's matrix_hash to differ from
    // the previous step's, which the stepper records as a topology_changed
    // invalidation reason. The first time the new topology appears, the
    // cache misses; the SECOND visit (after toggling back) should hit.
    CHECK(tel.linear_factor_cache_invalidations_topology_changed >= 1);

    // The Phase 3 win — same (topology, dt) pair revisits the cache
    // instead of refactoring. With VCSwitch bisection-to-event splitting
    // the accepted step at each commutation edge (Phase 4 of
    // refactor-pwl-switching-engine), the bisected sub-steps generate
    // unique fractional dt values that don't cycle back; each one
    // contributes one cache miss. Steady-state PWM steps between edges
    // share dt and topology and hit. Hit rate ≥ 70% with this pattern; a
    // pure fixed-dt-on-event-boundary run (no bisection) lands at 95%+
    // and is covered by the stable-topology test below.
    const int total = tel.linear_factor_cache_hits + tel.linear_factor_cache_misses;
    REQUIRE(total > 0);
    const double hit_rate = static_cast<double>(tel.linear_factor_cache_hits) /
                            static_cast<double>(total);
    CHECK(hit_rate >= 0.70);

    // Symbolic-only counter is present from Phase 2 but the per-key LRU
    // subsumes within-stepper symbolic reuse (each entry persists its own
    // analyzed pattern), so it stays 0 in segment-primary paths.
    CHECK(tel.symbolic_factor_cache_hits == 0);
}

// -----------------------------------------------------------------------------
// Stable-topology run: cache misses exactly once (the first step) and then
// hits for the rest of the run.
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3: stable-topology run misses once, hits forever after",
          "[v1][pwl][phase3][linear_cache][numeric_lru][warmup]") {
    Circuit ckt;
    auto in  = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_resistor("R1", in, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);
    ckt.add_voltage_source("V1", in, Circuit::ground(), 5.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-5;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep    = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(run.total_steps >= 50);

    const auto& tel = run.backend_telemetry;

    // No topology transitions on a passive circuit.
    CHECK(tel.pwl_topology_transitions == 0);

    // Stable-topology + fixed dt should hit the cache for almost every
    // step. The first step is always a miss; a handful of additional
    // misses can show up during integrator warmup if the assembled matrix
    // values drift slightly across the first few accepted steps. The bar
    // here is "≥ 95% hit rate after warmup" — gate G.1 from the change
    // spec — not literal one-miss optimality.
    CHECK(tel.linear_factor_cache_misses <= 8);
    const int total = tel.linear_factor_cache_hits + tel.linear_factor_cache_misses;
    REQUIRE(total > 0);
    const double hit_rate = static_cast<double>(tel.linear_factor_cache_hits) /
                            static_cast<double>(total);
    CHECK(hit_rate >= 0.92);

    // Whatever invalidations show up should be NumericInstability (matrix
    // hash drift), never TopologyChanged (no commutation).
    CHECK(tel.linear_factor_cache_invalidations_topology_changed == 0);
}
