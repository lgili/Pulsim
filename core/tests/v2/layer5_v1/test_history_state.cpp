// =============================================================================
// Layer 5 V1 — HistoryState basic tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/history_state.hpp"
#include "pulsim/v2/topology/graph.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("HistoryState: empty graph yields no entries",
          "[v2][layer5_v1][history]") {
    Graph g;
    g.add_node("n0");
    DevicePool pool;

    HistoryState history(g, pool);
    REQUIRE(history.num_dynamic_branches() == 0);

    const Vector b = history.compute_b_extra(1e-6);
    REQUIRE(b.size() == 1);
    REQUIRE(b[0] == Approx(0).margin(1e-15));
}

TEST_CASE("HistoryState: 1 cap → 1 entry initialised to zero",
          "[v2][layer5_v1][history]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    HistoryState history(g, pool);
    REQUIRE(history.num_dynamic_branches() == 1);

    // No previous step → compute_b_extra returns zero.
    const Vector b = history.compute_b_extra(1e-6);
    REQUIRE(b.size() == 1);
    REQUIRE(b[0] == Approx(0).margin(1e-15));
}

TEST_CASE("HistoryState: capacitor history term sign on b_extra row",
          "[v2][layer5_v1][history]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    HistoryState history(g, pool);

    // Pretend the previous step ended with v_node = 10 V. We feed
    // an x where x[0] = 10 and let update_from_state derive the
    // entry. Internally i_new = g_eq · (v_new − v_prev) − i_prev =
    // 2 · (10 − 0) − 0 = 20.
    Vector x(1);
    x[0] = 10.0;
    history.update_from_state(x, 1e-6);

    // Next step's history is I_hist = g_eq · v_prev + i_prev = 2·10
    // + 20 = 40. b_extra(from=0) += −I_hist = −40.
    const Vector b = history.compute_b_extra(1e-6);
    REQUIRE(b[0] == Approx(-40.0).margin(1e-9));
}

TEST_CASE("HistoryState: reset zeros all entries",
          "[v2][layer5_v1][history]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    HistoryState history(g, pool);
    Vector x(1);
    x[0] = 5.0;
    history.update_from_state(x, 1e-6);

    // Confirm it's populated.
    REQUIRE(history.compute_b_extra(1e-6)[0] != Approx(0).margin(1e-12));

    history.reset();
    REQUIRE(history.compute_b_extra(1e-6)[0] ==
            Approx(0).margin(1e-15));
}

TEST_CASE("HistoryState: inductor entry tracks branch-current i_L",
          "[v2][layer5_v1][history]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_inductor(1, {.L = 1e-3});

    HistoryState history(g, pool);
    REQUIRE(history.num_dynamic_branches() == 1);
    // state size = 1 + 1 + 1 = 3.
    REQUIRE(history.compute_b_extra(1e-6).size() == 3);

    // Feed a synthetic state: v_n0 = 12, i_src = -0.012, i_L = 2.0.
    Vector x(3);
    x[0] = 12.0;
    x[1] = -0.012;
    x[2] = 2.0;
    history.update_from_state(x, 1e-6);

    // I_hist,L = i_n + g_eq_inv · v_n = 2 + 5e-4 · 12 = 2.006.
    // 2L/dt = 2000. The constraint row's `b` portion is
    //   b(row) = +(2L/dt) · I_hist,L = +4012
    // which goes into b_extra with a positive sign (cache.solve
    // negates b in rhs).
    const Vector b = history.compute_b_extra(1e-6);
    REQUIRE(b[2] == Approx(4012.0).margin(1e-6));
    REQUIRE(b[0] == Approx(0).margin(1e-12));
    REQUIRE(b[1] == Approx(0).margin(1e-12));
}
