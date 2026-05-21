// =============================================================================
// Layer 4 — Integration test: chopper circuit end-to-end
// =============================================================================
//
// Chopper topology:
//
//   V_dc ──[Source]── vin ──[Switch M1]── vout ──[R_load]── GND
//
// Two switch states:
//   * M1 OPEN  → vout pulled to ground through R_load (large
//                resistor + tiny g_off → vout ≈ 0)
//   * M1 CLOSED → vout pulled to vin through g_on. Voltage divider
//                  vout = V_dc · g_on / (g_on + G_R) ≈ V_dc when
//                  g_on >> G_R.
//
// This is the smallest circuit that exercises BOTH segments of
// the cache. Each state's lookup + solve produces the analytical
// answer — proves the PLECS-style caching works end-to-end.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <chrono>
#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

// Helper that owns the graph + pool + cache. The cache is built
// AFTER the graph and pool are fully populated, then wrapped in a
// `unique_ptr` so its const-references stay valid across moves of
// the helper itself.
struct Chopper {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n_in = -1;
    Index n_out = -1;
    Real V_dc = 12.0;
    Real g_on = 1e3;
    Real g_off = 1e-9;
    Real G_R = 0.1;

    Chopper() {
        n_in  = g.add_node("vin");
        n_out = g.add_node("vout");
        g.add_branch(n_in, g.ground(), BranchKind::Source);     // V_dc
        g.add_branch(n_in, n_out,      BranchKind::Switch);      // M1
        g.add_branch(n_out, g.ground(),BranchKind::PassiveLinear); // R

        pool.add_voltage_source(0, {V_dc});
        pool.add_switch(1, g_on, g_off);
        pool.add_resistor(2, {G_R});

        // Cache must be built AFTER g + pool are fully populated.
        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build();
    }
};

}  // namespace

TEST_CASE("Chopper: cache builds exactly 2 segments (1 switch)",
          "[v2][layer4][integration][chopper]") {
    Chopper c;
    REQUIRE(c.cache->num_segments() == 2);
}

TEST_CASE("Chopper M1 ON: vout = V_dc within tight tolerance",
          "[v2][layer4][integration][chopper]") {
    Chopper c;
    SwitchStateMask mask(1);
    mask.set(0, true);                             // M1 closed

    Vector x;
    Vector b_extra = Vector::Zero(3);
    c.cache->solve(mask, b_extra, x);

    // Voltage divider: vout = V_dc · g_on / (g_on + G_R)
    //                       = 12 · 1000 / 1000.1 ≈ 11.9988 V
    const Real expected = c.V_dc * c.g_on / (c.g_on + c.G_R);
    INFO("expected vout = " << expected
         << " V (V_dc=" << c.V_dc << ", g_on=" << c.g_on
         << ", G_R=" << c.G_R << ")");
    INFO("computed vout = " << x[c.n_out] << " V");

    REQUIRE(x[c.n_in]  == Approx(c.V_dc).margin(1e-12));
    REQUIRE(x[c.n_out] == Approx(expected).margin(1e-9));
}

TEST_CASE("Chopper M1 OFF: vout ≈ 0 through g_off + R",
          "[v2][layer4][integration][chopper]") {
    Chopper c;
    SwitchStateMask mask(1);                       // M1 open

    Vector x;
    Vector b_extra = Vector::Zero(3);
    c.cache->solve(mask, b_extra, x);

    // M1 open: vout = V_dc · g_off / (g_off + G_R)
    //                = 12 · 1e-9 / (1e-9 + 0.1) ≈ 1.2e-7 V
    const Real expected = c.V_dc * c.g_off / (c.g_off + c.G_R);
    INFO("expected vout = " << expected << " V");
    INFO("computed vout = " << x[c.n_out] << " V");

    REQUIRE(x[c.n_in]  == Approx(c.V_dc).margin(1e-12));
    REQUIRE(x[c.n_out] == Approx(expected).margin(1e-12));
    REQUIRE(std::abs(x[c.n_out]) < Real{1e-6});
}

TEST_CASE("Chopper: 10k consecutive lookups complete quickly",
          "[v2][layer4][integration][chopper][performance]") {
    // Performance smoke test — not a benchmark, but catches
    // accidental O(N²) regressions in the map lookup.
    Chopper c;
    SwitchStateMask mask_on(1);  mask_on.set(0, true);
    SwitchStateMask mask_off(1);
    Vector b_extra = Vector::Zero(3);
    Vector x;

    const auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; ++i) {
        c.cache->solve(((i & 1) != 0) ? mask_on : mask_off, b_extra, x);
    }
    const auto t_end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();

    INFO("10k cache.solve() calls took " << elapsed_ms << " ms");
    REQUIRE(elapsed_ms < 1000);                    // generous; <1s
}
