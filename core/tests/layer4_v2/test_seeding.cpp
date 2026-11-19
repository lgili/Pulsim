// =============================================================================
// Layer 4 V2 — HistoryState seeding tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/seeding.hpp"
#include "pulsim/topology/graph.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("Seed history from DC OP — capacitor",
          "[v2][layer4_v2][seeding]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);  // C

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_capacitor(1, {.C = 1e-6});

    SwitchStateMask mask(0);
    const auto dc_x = compute_dc_op(g, pool, mask);
    auto history = make_seeded_history(g, pool, dc_x);

    REQUIRE(history.entries().size() == 1);
    REQUIRE(history.entries()[0].v_prev == Approx(5.0).margin(1e-9));
    REQUIRE(history.entries()[0].i_prev == Approx(0.0).margin(1e-15));
}

TEST_CASE("Seed history from DC OP — inductor",
          "[v2][layer4_v2][seeding]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::PassiveLinear);  // R
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // L

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_resistor(1, {.G = 1.0});
    pool.add_inductor(2, {.L = 10e-6});

    SwitchStateMask mask(0);
    const auto dc_x = compute_dc_op(g, pool, mask);
    auto history = make_seeded_history(g, pool, dc_x);

    REQUIRE(history.entries().size() == 1);
    // Inductor: v_prev = 0 (DC voltage is 0), i_prev = V/R = 12.
    REQUIRE(history.entries()[0].v_prev == Approx(0.0).margin(1e-6));
    REQUIRE(history.entries()[0].i_prev == Approx(12.0).margin(1e-6));
}

TEST_CASE("Diode seeding from DC OP turns on forward-biased diodes",
          "[v2][layer4_v2][seeding]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Switch);   // diode
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_diode(1, /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0.0);
    pool.add_resistor(2, {.G = 0.1});

    // For DC OP, the diode bit must be set ON to converge (the
    // user's first-call to compute_dc_op assumes OFF; the
    // seeding helper handles iteration in run_transient — for
    // this unit test, we manually set the mask).
    SwitchStateMask mask(1);
    mask.set(0, true);  // pretend diode is ON
    const auto dc_x = compute_dc_op(g, pool, mask);

    auto diodes = make_seeded_diodes(g, pool, dc_x);
    REQUIRE(diodes.num_diodes() == 1);
    // With V_dc=5 pulling node 0, and the diode forward biased,
    // it should be detected as ON.
    REQUIRE(diodes.current_diode_mask().get(0));
}

TEST_CASE("Seeding decides a gated device under its gate, off by default",
          "[v2][layer4_v2][seeding][gated]") {
    // add-pwl-igbt-and-thyristor D13: a forward-biased thyristor is not
    // seeded ON by its voltage alone.
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Switch);   // thyristor
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_gated_diode(1, 1e3, 1e-9, 0.0, 0.0,
                         models::SwitchedDiode::Gate::Latching);
    pool.add_resistor(2, {.G = 0.1});

    SwitchStateMask on(1);
    on.set(0, true);
    const auto dc_x = compute_dc_op(g, pool, on);   // forward-biased

    // control: the caller's gate on seeds it ON
    REQUIRE(make_seeded_diodes(g, pool, dc_x, &on).current_diode_mask().get(0));
    // no gate given: off
    REQUIRE_FALSE(make_seeded_diodes(g, pool, dc_x).current_diode_mask().get(0));
    const SwitchStateMask off(1);
    REQUIRE_FALSE(make_seeded_diodes(g, pool, dc_x, &off).current_diode_mask().get(0));
}

TEST_CASE("The mode bits carry the gate of each gated device",
          "[v2][layer4_v2][seeding][gated]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);   // plain diode
    g.add_branch(0, g.ground(), BranchKind::Switch);   // thyristor
    DevicePool pool;
    pool.add_diode(0, 1e3, 1e-9, 0.0);
    pool.add_gated_diode(1, 1e3, 1e-9, 0.0, 0.0,
                         models::SwitchedDiode::Gate::Latching);
    DiodeEventState d(g, pool);
    SwitchStateMask user(2);
    user.set(1, true);
    d.apply_gates(user);
    REQUIRE(d.gate_of(1));
    d.restore_on_bits({true, true});
    REQUIRE(d.snapshot_mode_bits() == std::vector<bool>{true, true, true});
    // a resume restores the gate it was decided under
    d.restore_on_bits({false, true, false});
    REQUIRE_FALSE(d.gate_of(1));
    REQUIRE(d.snapshot_mode_bits() == std::vector<bool>{false, true, false});
    REQUIRE_FALSE(d.gate_of(0));   // not a gated device
    REQUIRE_THROWS_AS(d.restore_on_bits({true}), std::invalid_argument);
}
