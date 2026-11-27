// F-Z-8 follow-up: the restart rule's impulse warning names what collapsed.
// A capacitor switched onto a source is a capacitor loop (a voltage jump by
// charge sharing), never "the current of" the capacitor. Such a cell never
// trips the voltage bound by itself (the projection holds every node at a
// source's value), so the run's own record is paired with a right-limit row
// the bound rejects at its instant: the wording path, not the trip, is tested.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/voltage_sanity.hpp"
#include "pulsim/topology/graph.hpp"

#include <format>
#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;

TEST_CASE("A capacitor switched onto a source is reported as a capacitor loop",
          "[voltage_sanity][restart][F-Z-8]") {
    constexpr Real V = 100.0, C = 1e-6, t_e = 1e-4, dt = 1e-6;
    Graph g;
    const Index vin = g.add_node("vin");
    const Index c = g.add_node("c");
    g.add_branch(vin, g.ground(), BranchKind::Source, "V");
    g.add_branch(vin, c, BranchKind::Switch, "S");
    g.add_branch(c, g.ground(), BranchKind::PassiveLinear, "C");
    DevicePool pool;
    pool.add_voltage_source(0, {.V = V});
    pool.add_switch(1, 1e3, 1e-12);
    pool.add_capacitor(2, {.C = C});
    PwlStateSpaceCache cache(g, pool);
    cache.build(dt);
    SimulationOptions o;
    o.t_start = 0.0;
    o.dt = dt;
    o.t_end = 2 * t_e;
    o.discontinuity_restart = true;
    auto r = run_transient(cache, g, pool, o, [&](Real t) {
        SwitchStateMask m(1);
        m.set(0, t >= t_e);
        return m;
    });
    const ReinitRecord* rec = nullptr;
    for (const auto& q : r.reinit_records) {
        if (q.kind == "capacitor_loop") rec = &q;
    }
    REQUIRE(rec != nullptr);
    // closed form: the loop dissipates ½·C·V² whatever the switch's R_on
    REQUIRE(rec->dW == Catch::Approx(0.5 * C * V * V).epsilon(1e-6));
    REQUIRE(r.has_event_limits);
    const Size w = r.event_limit_width;
    bool planted = false;
    for (Size j = 0; j < r.right_limit_index.size(); ++j) {
        if (r.times[r.right_limit_index[j]] == rec->t) {
            r.right_limit_states[j * w + static_cast<Size>(c)] = 1e9;   // the bound rejects it
            planted = true;
        }
    }
    REQUIRE(planted);
    const auto f = find_implausible_voltage(g, pool, r);
    REQUIRE(f.node == c);
    REQUIRE(f.cut_state_kind == 'C');
    REQUIRE(f.cut_size == Catch::Approx(V).epsilon(1e-6));   // ΔV = 2·dW/λ
    const std::string msg = describe(g, pool, f);
    INFO(msg);
    REQUIRE(msg.find("closed a capacitor loop on 'C'") != std::string::npos);
    REQUIRE(msg.find(std::format("by {:.4g} V", f.cut_size)) != std::string::npos);
    REQUIRE(msg.find(std::format("{:.4g} J", rec->dW)) != std::string::npos);
    REQUIRE(msg.find("current of") == std::string::npos);

    // two records at one instant (review): the row is read against the one
    // whose devices touch the node, a cut set before a capacitor loop, and a
    // record of other devices listed first is not taken
    ReinitRecord other{.t = rec->t, .kind = "capacitor_loop", .devices = {"V"}};
    ReinitRecord cs{.t = rec->t, .kind = "cut_set", .devices = {"S"}, .states = {"C"},
                    .lambda = 1.0, .dW = 2.0};
    r.reinit_records.insert(r.reinit_records.begin(), other);
    r.reinit_records.push_back(cs);
    const auto f2 = find_implausible_voltage(g, pool, r);
    REQUIRE(f2.cut_kind == "cut_set");
    REQUIRE(f2.cut_state_kind == 0);   // its only state is no inductor: neither wording
    const std::string m2 = describe(g, pool, f2);
    INFO(m2);
    REQUIRE(m2.find("kind 'cut_set'") != std::string::npos);
    REQUIRE(m2.find("capacitor loop") == std::string::npos);
    REQUIRE(m2.find("current of") == std::string::npos);
}
