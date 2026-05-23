// =============================================================================
// Layer 4 V7 — Multi-dt cache (solve_at)
// =============================================================================
//
// `solve_at(mask, dt, b_extra, x)` lets the caller use a dt
// DIFFERENT from the primary build dt. The auxiliary cache
// builds-and-stores segments per (mask, dt) pair on demand.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <memory>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

/// Chopper fixture: V_dc → Switch → R → GND (1 switch).
struct Chopper {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;

    Chopper() {
        g.add_node("vin");
        g.add_node("vout");
        g.add_branch(0, g.ground(), BranchKind::Source);
        g.add_branch(0, 1,          BranchKind::Switch);
        g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = 12.0});
        pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
        pool.add_resistor(2, {.G = 0.1});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
    }
};

}  // namespace

TEST_CASE("solve_at with primary dt matches solve bit-identical",
          "[v2][layer4_v7][multi_dt]") {
    Chopper f;
    constexpr Real dt = 0;   // static chopper
    f.cache->build(dt);

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector b_extra = Vector::Zero(3);
    Vector x_solve, x_solve_at;
    f.cache->solve(mask_on, b_extra, x_solve);
    f.cache->solve_at(mask_on, dt, b_extra, x_solve_at);

    REQUIRE(x_solve.size() == x_solve_at.size());
    for (Eigen::Index i = 0; i < x_solve.size(); ++i) {
        REQUIRE(x_solve[i] == Approx(x_solve_at[i]).margin(1e-12));
    }
}

TEST_CASE("solve_at with new dt populates the auxiliary cache",
          "[v2][layer4_v7][multi_dt]") {
    Chopper f;
    f.cache->build(Real{0});
    REQUIRE(f.cache->num_alt_dt_values() == 0);

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector b_extra = Vector::Zero(3);
    Vector x;

    f.cache->solve_at(mask_on, 1e-6, b_extra, x);
    REQUIRE(f.cache->num_alt_dt_values() == 1);
    REQUIRE(f.cache->num_alt_segments_at(1e-6) == 1);

    f.cache->solve_at(mask_on, 2e-6, b_extra, x);
    REQUIRE(f.cache->num_alt_dt_values() == 2);
    REQUIRE(f.cache->num_alt_segments_at(2e-6) == 1);

    // Reuse same (mask, dt) — should NOT bump count.
    f.cache->solve_at(mask_on, 1e-6, b_extra, x);
    REQUIRE(f.cache->num_alt_segments_at(1e-6) == 1);

    // New mask at known dt → bumps that dt's count.
    SwitchStateMask mask_off(1);
    f.cache->solve_at(mask_off, 1e-6, b_extra, x);
    REQUIRE(f.cache->num_alt_segments_at(1e-6) == 2);
}

TEST_CASE("Multi-dt solve_at gives correct chopper answer",
          "[v2][layer4_v7][multi_dt]") {
    Chopper f;
    f.cache->build(Real{0});

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector b_extra = Vector::Zero(3);
    Vector x;
    f.cache->solve_at(mask_on, 1e-6, b_extra, x);

    REQUIRE(x[0] == Approx(12.0).margin(1e-9));
    REQUIRE(x[1] == Approx(11.9988).epsilon(0.001));
}
