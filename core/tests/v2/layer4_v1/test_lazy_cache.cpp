// =============================================================================
// Layer 4 V6 — Lazy cache build tests
// =============================================================================
//
// build_lazy(dt) stores the dt but defers segment factorisation
// until solve(mask, ...) first asks for the mask. Subsequent
// solves with the same mask hit the cached factor.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

struct ChopperFixture {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n_in = -1, n_out = -1;

    ChopperFixture() {
        n_in  = g.add_node("vin");
        n_out = g.add_node("vout");
        g.add_branch(n_in, g.ground(), BranchKind::Source);
        g.add_branch(n_in, n_out,      BranchKind::Switch);
        g.add_branch(n_out, g.ground(),BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = 12.0});
        pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
        pool.add_resistor(2, {.G = 0.1});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
    }
};

}  // namespace

TEST_CASE("Lazy build starts with zero built segments",
          "[v2][layer4_v6][lazy_cache]") {
    ChopperFixture f;
    f.cache->build_lazy(Real{0});
    REQUIRE(f.cache->num_built_segments() == 0);
}

TEST_CASE("Lazy build populates on each new mask",
          "[v2][layer4_v6][lazy_cache]") {
    ChopperFixture f;
    f.cache->build_lazy(Real{0});

    SwitchStateMask mask_off(1);
    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector x;
    Vector zero_b = Vector::Zero(3);

    REQUIRE(f.cache->num_built_segments() == 0);

    // First solve with mask_off builds segment 0.
    f.cache->solve(mask_off, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 1);

    // First solve with mask_on builds segment 1.
    f.cache->solve(mask_on, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 2);

    // Second solve with mask_off reuses the cached factor.
    f.cache->solve(mask_off, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 2);

    // Second solve with mask_on also reuses.
    f.cache->solve(mask_on, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 2);
}

TEST_CASE("Lazy build produces same results as eager build",
          "[v2][layer4_v6][lazy_cache]") {
    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector zero_b = Vector::Zero(3);

    Vector x_eager, x_lazy;

    {
        ChopperFixture f;
        f.cache->build(Real{0});   // eager
        f.cache->solve(mask_on, zero_b, x_eager);
    }
    {
        ChopperFixture f;
        f.cache->build_lazy(Real{0});
        f.cache->solve(mask_on, zero_b, x_lazy);
    }

    REQUIRE(x_eager.size() == x_lazy.size());
    for (Eigen::Index i = 0; i < x_eager.size(); ++i) {
        REQUIRE(x_eager[i] == Approx(x_lazy[i]).margin(1e-12));
    }
}

TEST_CASE("Eager build after lazy clears + populates all segments",
          "[v2][layer4_v6][lazy_cache]") {
    ChopperFixture f;
    f.cache->build_lazy(Real{0});

    SwitchStateMask mask_off(1);
    Vector zero_b = Vector::Zero(3);
    Vector x;
    f.cache->solve(mask_off, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 1);

    // Switch to eager — clears lazy segments, builds all.
    f.cache->build(Real{0});
    REQUIRE(f.cache->num_built_segments() == 2);
}

TEST_CASE("Lazy build after eager clears + waits for solves",
          "[v2][layer4_v6][lazy_cache]") {
    ChopperFixture f;
    f.cache->build(Real{0});
    REQUIRE(f.cache->num_built_segments() == 2);

    f.cache->build_lazy(Real{0});
    REQUIRE(f.cache->num_built_segments() == 0);
}
