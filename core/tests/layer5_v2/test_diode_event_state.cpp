// =============================================================================
// Layer 5 V2 — DiodeEventState tracking + mask combine tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;

TEST_CASE("DiodeEventState: no diodes → empty tracker",
          "[v2][layer5_v2][diode_event_state]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});

    DiodeEventState diodes(g, pool);
    REQUIRE(diodes.num_diodes() == 0);
    REQUIRE(diodes.current_diode_mask().bits() == 0);
    REQUIRE(diodes.diode_owned_bits().bits() == 0);
}

TEST_CASE("DiodeEventState: 1 diode initially OFF",
          "[v2][layer5_v2][diode_event_state]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_diode(0, /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0.0);

    DiodeEventState diodes(g, pool);
    REQUIRE(diodes.num_diodes() == 1);
    REQUIRE_FALSE(diodes.current_diode_mask().get(0));
    REQUIRE(diodes.diode_owned_bits().get(0));
}

TEST_CASE("DiodeEventState: forward bias flips OFF → ON",
          "[v2][layer5_v2][diode_event_state]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_diode(0, 1e3, 1e-9, 0.0);

    DiodeEventState diodes(g, pool);
    Vector x(1);
    x[0] = +5.0;  // anode = +5 V, cathode = GND (0)
    REQUIRE(diodes.update_from_state(x));   // returns true (flipped)
    REQUIRE(diodes.current_diode_mask().get(0));
}

TEST_CASE("DiodeEventState: reverse bias keeps OFF",
          "[v2][layer5_v2][diode_event_state]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_diode(0, 1e3, 1e-9, 0.0);

    DiodeEventState diodes(g, pool);
    Vector x(1);
    x[0] = -5.0;
    REQUIRE_FALSE(diodes.update_from_state(x));   // no change
    REQUIRE_FALSE(diodes.current_diode_mask().get(0));
}

TEST_CASE("DiodeEventState: reverse current flips ON → OFF",
          "[v2][layer5_v2][diode_event_state]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_diode(0, 1e3, 1e-9, 0.0);

    DiodeEventState diodes(g, pool);
    // First flip to ON.
    Vector x(1);
    x[0] = +5.0;
    REQUIRE(diodes.update_from_state(x));
    REQUIRE(diodes.current_diode_mask().get(0));

    // Then apply reverse bias → i_diode = g_on · (-0.5) = -500 ≤ 0 → OFF.
    x[0] = -0.5;
    REQUIRE(diodes.update_from_state(x));
    REQUIRE_FALSE(diodes.current_diode_mask().get(0));
}

TEST_CASE("DiodeEventState: reset() clears all diodes",
          "[v2][layer5_v2][diode_event_state]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_diode(0, 1e3, 1e-9, 0.0);

    DiodeEventState diodes(g, pool);
    Vector x(1);
    x[0] = +5.0;
    diodes.update_from_state(x);
    REQUIRE(diodes.current_diode_mask().get(0));

    diodes.reset();
    REQUIRE_FALSE(diodes.current_diode_mask().get(0));
}

TEST_CASE("combine_masks: overlays diode bits, keeps user bits",
          "[v2][layer5_v2][combine_masks]") {
    SwitchStateMask user(4);
    user.set(0, true);   // controlled switch ON
    user.set(1, false);
    user.set(2, true);
    user.set(3, false);

    SwitchStateMask diode(4);
    diode.set(0, false);
    diode.set(1, true);    // diode bit position
    diode.set(2, false);
    diode.set(3, false);

    SwitchStateMask diode_owned(4);
    diode_owned.set(1, true);   // bit 1 is the diode

    const auto combined = combine_masks(user, diode, diode_owned);
    REQUIRE(combined.get(0) == true);   // user bit, unchanged
    REQUIRE(combined.get(1) == true);   // diode bit, overlaid
    REQUIRE(combined.get(2) == true);   // user bit, unchanged
    REQUIRE(combined.get(3) == false);  // user bit, unchanged
}
