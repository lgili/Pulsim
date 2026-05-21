// =============================================================================
// Layer 5 V3 — Snapshot/restore round-trip unit tests
// =============================================================================
//
// HistoryState and DiodeEventState got snapshot/restore
// methods in V3 to support substep state correction. The
// round-trip invariant: `restore(snapshot())` is a no-op.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/diode_event_state.hpp"
#include "pulsim/v2/pwl/history_state.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <stdexcept>
#include <vector>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("HistoryState snapshot/restore round-trip on RC circuit",
          "[v2][layer5_v3][snapshot]") {
    // V_dc → R → C → GND. One capacitor entry in history.
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::PassiveLinear);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0 / 100.0});
    pool.add_capacitor(2, {.C = 1e-6});

    HistoryState history(g, pool);
    REQUIRE(history.entries().size() == 1);

    constexpr Real dt = 1e-5;
    const Vector b_initial = history.compute_b_extra(dt);

    // Snapshot the initial (zeroed) state.
    auto snap_initial = history.snapshot();

    // Mutate the history via a fake state.
    Vector fake_x = Vector::Zero(
        static_cast<Index>(pool.state_size(g)));
    fake_x[0] = 4.0;
    fake_x[1] = 2.0;
    history.update_from_state(fake_x, dt);
    const Vector b_after_update = history.compute_b_extra(dt);

    // Confirm b changed.
    REQUIRE_FALSE((b_after_update - b_initial)
                       .lpNorm<Eigen::Infinity>() < 1e-9);

    // Restore and check b returns to initial.
    history.restore(snap_initial);
    const Vector b_restored = history.compute_b_extra(dt);
    REQUIRE((b_restored - b_initial)
                 .lpNorm<Eigen::Infinity>() < 1e-9);
}

TEST_CASE("DiodeEventState snapshot/restore round-trip",
          "[v2][layer5_v3][snapshot]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, 1, BranchKind::Switch);   // diode

    DevicePool pool;
    pool.add_diode(0, 1.0, 1e-9, 0.7);

    DiodeEventState diodes(g, pool);
    diodes.reset();   // all OFF
    auto snap_off = diodes.snapshot_on_bits();
    REQUIRE(snap_off.size() == 1);
    REQUIRE(snap_off[0] == false);

    // Force the diode ON via a fake state vector.
    Vector fake_x = Vector::Zero(
        static_cast<Index>(pool.state_size(g)));
    fake_x[0] = 2.0;   // v_anode
    fake_x[1] = 0.0;   // v_cathode → v_diode = +2 V
    bool flipped = diodes.update_from_state(fake_x);
    REQUIRE(flipped);

    auto snap_on = diodes.snapshot_on_bits();
    REQUIRE(snap_on[0] == true);

    // Restore OFF and verify.
    diodes.restore_on_bits(snap_off);
    REQUIRE(diodes.snapshot_on_bits()[0] == false);

    // Restore ON and verify.
    diodes.restore_on_bits(snap_on);
    REQUIRE(diodes.snapshot_on_bits()[0] == true);
}

TEST_CASE("DiodeEventState restore_on_bits rejects size mismatch",
          "[v2][layer5_v3][snapshot]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, 1, BranchKind::Switch);
    DevicePool pool;
    pool.add_diode(0, 1.0, 1e-9, 0.7);

    DiodeEventState diodes(g, pool);
    REQUIRE_THROWS_AS(
        diodes.restore_on_bits(std::vector<bool>{false, true}),
        std::invalid_argument);
}
