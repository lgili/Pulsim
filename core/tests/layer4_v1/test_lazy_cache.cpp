// =============================================================================
// Layer 4 V6 — Lazy cache build tests
// =============================================================================
//
// build_lazy(dt) stores the dt but defers segment factorisation
// until solve(mask, ...) first asks for the mask. Subsequent
// solves with the same mask hit the cached factor.

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

// =============================================================================
// Phase 1 — dynamic mask: >64-switch circuits through the lazy cache
// =============================================================================
//
// The v1.x SwitchStateMask constructor threw for more than 64
// switches, so a 70-switch circuit could not even be REPRESENTED
// (audit finding switch-mask-64-cap). With the Phase-1 dynamic mask,
// the lazy cache path handles arbitrary widths: only visited masks
// are factorised, so 2^70 states never materialise. Eager build()
// still fails loudly (Gray-code enumerator guard) — lazy is the
// documented route for wide circuits.

namespace {

/// 12 V source → 10 Ω series resistor → node `vout`, then kNsw
/// switch branches from vout to ground (g_on = 1 S, g_off = 1 nS).
/// With k switches closed: v_out = 12 * 0.1 / (0.1 + k · 1.0)  (up
/// to the negligible off-conductance of the open switches).
struct WideChopperFixture {
    static constexpr Size kNsw = 70;
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n_in = -1, n_out = -1;

    WideChopperFixture() {
        n_in  = g.add_node("vin");
        n_out = g.add_node("vout");
        g.add_branch(n_in, g.ground(), BranchKind::Source);
        g.add_branch(n_in, n_out,      BranchKind::PassiveLinear);
        for (Size s = 0; s < kNsw; ++s) {
            g.add_branch(n_out, g.ground(), BranchKind::Switch);
        }
        pool.add_voltage_source(0, {.V = 12.0});
        pool.add_resistor(1, {.G = 0.1});
        for (Size s = 0; s < kNsw; ++s) {
            pool.add_switch(static_cast<Index>(2 + s),
                            /*g_on=*/1.0, /*g_off=*/1e-9);
        }
        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
    }

    [[nodiscard]] Real expected_vout(Size k_closed) const {
        const Real g_on  = 1.0;
        const Real g_off = 1e-9;
        const Real g_sw = static_cast<Real>(k_closed) * g_on +
                          static_cast<Real>(kNsw - k_closed) * g_off;
        return Real{12} * Real{0.1} / (Real{0.1} + g_sw);
    }
};

}  // namespace

TEST_CASE("70-switch circuit solves through the lazy cache",
          "[v2][layer4_v6][lazy_cache][wide]") {
    WideChopperFixture f;
    REQUIRE(f.g.num_switches() == WideChopperFixture::kNsw);
    f.cache->build_lazy(Real{0});

    // Wide mask spanning the word boundary: bits 0, 64, 69.
    SwitchStateMask m3(WideChopperFixture::kNsw);
    m3.set(0, true);
    m3.set(64, true);
    m3.set(69, true);

    Vector x;
    const Vector zero_b = Vector::Zero(3);  // v_in, v_out, i_src
    f.cache->solve(m3, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 1);
    REQUIRE(x[f.n_in]  == Approx(Real{12}).margin(1e-9));
    REQUIRE(x[f.n_out] == Approx(f.expected_vout(3)).epsilon(1e-9));

    // A second, INDEPENDENTLY built but equal mask must hit the same
    // segment — locks hash/equality across the word boundary in the
    // real unordered_map, not just in unit assertions.
    SwitchStateMask m3_again(WideChopperFixture::kNsw);
    for (Size i : {Size{0}, Size{64}, Size{69}}) m3_again.set(i, true);
    f.cache->solve(m3_again, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 1);

    // Different upper-word state → new segment, new solution.
    SwitchStateMask m1(WideChopperFixture::kNsw);
    m1.set(69, true);
    f.cache->solve(m1, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 2);
    REQUIRE(x[f.n_out] == Approx(f.expected_vout(1)).epsilon(1e-9));

    // All-open: only leakage loads the node → v_out ≈ 12 V.
    SwitchStateMask m0(WideChopperFixture::kNsw);
    f.cache->solve(m0, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 3);
    REQUIRE(x[f.n_out] == Approx(f.expected_vout(0)).epsilon(1e-9));
}

TEST_CASE("70-switch circuit: eager build fails loudly, lazy is the route",
          "[v2][layer4_v6][lazy_cache][wide]") {
    WideChopperFixture f;
    // Eager enumeration of 2^70 masks is impossible — the Gray-code
    // enumerator throws instead of silently truncating the state
    // space. (The lazy path above is the supported wide-N route.)
    REQUIRE_THROWS_AS(f.cache->build(Real{0}), std::invalid_argument);
}
