// =============================================================================
// Layer 2 V15 — VCVS / Op-Amp Ideal tests
// =============================================================================
//
// Validates:
//   * Open-loop: V_out = gain · (V_in_pos − V_in_neg).
//   * Voltage follower (negative feedback): V_out tracks V_in
//     within 1/gain (the "virtual short" idealisation).
//   * Non-inverting amplifier: gain = 1 + R_f/R_g (closed-loop
//     gain converges to the resistive ratio).
//   * DevicePool round-trip + input-node lookup.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/run_transient.hpp"

#include <cmath>

using namespace pulsim::v2;
using namespace pulsim::v2::builder;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("VCVS — open-loop transfer V_out = gain · V_in",
          "[v2][layer2_v15][vcvs][unit]") {
    // V_in = 0.5 V (forced by voltage source).
    // VCVS gain = 10 → V_out should be 5 V.
    // Load: R = 1 kΩ from out to gnd (gives VCVS a current to sink).
    CircuitBuilder b;
    b.add_voltage_source("V_in", "in_pos", "gnd", 0.5);
    b.add_vcvs("E1", "in_pos", "gnd", "out", "gnd", 10.0);
    b.add_resistor("R_L", "out", "gnd", 1000.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();

    SimulationOptions opts{
        .t_start = 0.0, .t_end = 1.0, .dt = 0.1};
    auto sw_fn = [](Real) { return SwitchStateMask(0); };
    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, sw_fn);

    const Index out_idx = b.node_id_of("out");
    const Real v_out = result.states[
        result.num_steps() - 1][out_idx];
    REQUIRE(v_out == Approx(5.0).margin(1e-6));
}

TEST_CASE("VCVS — voltage follower (op-amp + negative feedback)",
          "[v2][layer2_v15][vcvs][unit]") {
    // Voltage follower: V_in → in_pos, out → in_neg.
    // Closed-loop: V_out → V_in within 1/gain.
    // V_in = 3.7 V, gain = 1e5 → error ≈ 3.7/1e5 = 37 µV.
    CircuitBuilder b;
    b.add_voltage_source("V_in", "in_pos", "gnd", 3.7);
    b.add_op_amp_ideal("U1", "in_pos", "out", "out");
    // Need a small load on out so the VCVS has a finite path.
    b.add_resistor("R_L", "out", "gnd", 10000.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();

    SimulationOptions opts{
        .t_start = 0.0, .t_end = 1.0, .dt = 0.1};
    auto sw_fn = [](Real) { return SwitchStateMask(0); };
    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, sw_fn);

    const Index out_idx = b.node_id_of("out");
    const Real v_out = result.states[
        result.num_steps() - 1][out_idx];
    // V_out should equal V_in within ~1/gain.
    REQUIRE(v_out == Approx(3.7).margin(1e-4));
}

TEST_CASE("VCVS — non-inverting amplifier: closed-loop gain = 1 + R_f/R_g",
          "[v2][layer2_v15][vcvs][unit]") {
    // V_in → in_pos. out → R_f → in_neg → R_g → gnd.
    // Closed-loop V_out / V_in = 1 + R_f/R_g.
    // With R_f = 10 kΩ, R_g = 1 kΩ → gain = 11.
    // V_in = 0.5 → V_out = 5.5 V.
    CircuitBuilder b;
    b.add_voltage_source("V_in", "in_pos", "gnd", 0.5);
    b.add_op_amp_ideal("U1", "in_pos", "fb", "out");
    b.add_resistor("R_f", "out", "fb",  10000.0);
    b.add_resistor("R_g", "fb",  "gnd", 1000.0);
    b.add_resistor("R_L", "out", "gnd", 100000.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();

    SimulationOptions opts{
        .t_start = 0.0, .t_end = 1.0, .dt = 0.1};
    auto sw_fn = [](Real) { return SwitchStateMask(0); };
    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, sw_fn);

    const Index out_idx = b.node_id_of("out");
    const Real v_out = result.states[
        result.num_steps() - 1][out_idx];
    INFO("V_out = " << v_out << " (expected 5.5)");
    REQUIRE(v_out == Approx(5.5).margin(1e-3));
}

TEST_CASE("DevicePool stores VCVS params + input-node refs",
          "[v2][layer2_v15][vcvs][unit]") {
    CircuitBuilder b;
    b.add_vcvs("E1", "a", "b", "c", "gnd", 42.0);
    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
            DevicePool::StoredKind::VCVS);
    REQUIRE(b.pool().vcvs_params(0).gain == Approx(42.0));
    const auto [in_pos, in_neg] =
        b.pool().vcvs_input_nodes(0);
    REQUIRE(in_pos == b.node_id_of("a"));
    REQUIRE(in_neg == b.node_id_of("b"));
}
