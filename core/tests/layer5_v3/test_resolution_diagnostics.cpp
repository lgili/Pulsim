// add-fixed-step-resolution-diagnostics: the truncation estimate and the
// ringing detector of the fixed engine, on an RC stepped from 0 to 10 V
// (tau = 1 ms). Prototype numbers (design D1): r_max 0.35 / 2.7e-2 /
// 3.3e-6 at dt = 10 tau / tau / tau/100, a run of >= 4 alternating steps
// at 10 tau and none at tau. Read-only: states equal with it on and off.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;

namespace {

SimulationResult rc_run(Real dt, bool diag) {
    Graph g;
    const Index in = g.add_node("in");
    const Index out = g.add_node("out");
    g.add_branch(in, g.ground(), BranchKind::Source);
    g.add_branch(in, out, BranchKind::PassiveLinear);
    g.add_branch(out, g.ground(), BranchKind::PassiveLinear);
    DevicePool pool;
    pool.add_voltage_source(0, {.V = 10.0});
    pool.add_resistor(1, {.G = 1e-3});
    pool.add_capacitor(2, {.C = 1e-6});
    PwlStateSpaceCache cache(g, pool);
    cache.build(dt);
    SimulationOptions o;
    o.t_start = 0.0;
    o.dt = dt;
    o.t_end = 30 * dt;
    o.resolution_diagnostics = diag;
    return run_transient(cache, g, pool, o,
                         [](Real) { return SwitchStateMask(0); });
}

}  // namespace

TEST_CASE("Resolution diagnostics: RC truncation estimate and ringing",
          "[resolution]") {
    const Real tau = 1e-3;
    struct Row { Real dt; Real r_proto; bool rings; };
    for (const Row row : {Row{10 * tau, 0.35, true}, Row{tau, 2.7e-2, false},
                          Row{tau / 100, 3.3e-6, false}}) {
        const auto on = rc_run(row.dt, true);
        const auto& rd = on.resolution;
        REQUIRE(rd.enabled);
        REQUIRE(rd.n_branches_covered == 1);
        REQUIRE(rd.lte.size() == on.times.size());
        CHECK(rd.lte_max > row.r_proto / 2);
        CHECK(rd.lte_max < row.r_proto * 2);
        CHECK(rd.lte_branch == 2);
        Size longest = 0;
        for (const auto& r : rd.ringing) longest = std::max(longest, r.length);
        CHECK((longest >= 4) == row.rings);

        const auto off = rc_run(row.dt, false);
        CHECK_FALSE(off.resolution.enabled);
        CHECK(off.resolution.lte.empty());
        REQUIRE(off.states.rows() == on.states.rows());
        const Size n = on.states.rows() * static_cast<Size>(on.states.cols());
        CHECK(std::equal(on.states.data(), on.states.data() + n,
                         off.states.data()));
    }
}
