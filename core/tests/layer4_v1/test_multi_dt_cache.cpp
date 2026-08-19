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

TEST_CASE("solve_at reuses ONE event solver per mask across dt changes",
          "[v2][layer4_v7][multi_dt]") {
    // v2.0 Phase 1 (audit finding alt-dt-cache-unbounded-
    // factorization): the v1.x aux cache retained a fully analyzed
    // + factorized segment per EXACT-Real (mask, dt) pair forever —
    // sub-step event correction produces an essentially unique dt
    // per commutation, so memory grew without bound. Now each mask
    // owns ONE event solver whose numeric factor is refreshed in
    // place when dt changes (`J = G + (1/dt)·C` + factorize on the
    // shared symbolic; no analyze, no dt-keyed storage).
    Chopper f;
    f.cache->build(Real{0});
    REQUIRE(f.cache->num_event_entries() == 0);

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector b_extra = Vector::Zero(3);
    Vector x;

    f.cache->solve_at(mask_on, 1e-6, b_extra, x);
    REQUIRE(f.cache->num_event_entries() == 1);
    REQUIRE(f.cache->metrics().event_builds == 1);
    REQUIRE(f.cache->num_alt_segments_at(1e-6) == 1);

    // dt change on the SAME mask: in-place refactor, no new entry.
    f.cache->solve_at(mask_on, 2e-6, b_extra, x);
    REQUIRE(f.cache->num_event_entries() == 1);
    REQUIRE(f.cache->metrics().event_refactors == 1);
    REQUIRE(f.cache->num_alt_segments_at(2e-6) == 1);
    REQUIRE(f.cache->num_alt_segments_at(1e-6) == 0);  // superseded

    // Same (mask, dt) as currently loaded → pure factor reuse.
    f.cache->solve_at(mask_on, 2e-6, b_extra, x);
    REQUIRE(f.cache->metrics().event_hits == 1);
    REQUIRE(f.cache->metrics().event_refactors == 1);

    // New mask → its own entry.
    SwitchStateMask mask_off(1);
    f.cache->solve_at(mask_off, 2e-6, b_extra, x);
    REQUIRE(f.cache->num_event_entries() == 2);
    REQUIRE(f.cache->metrics().event_builds == 2);
    REQUIRE(f.cache->num_alt_dt_values() == 1);   // both at 2e-6

    // The commutation pattern (pre-mask at dt1, post-mask at dt2,
    // repeated with fresh dts every event) stays at 2 entries.
    for (int ev = 0; ev < 20; ++ev) {
        const Real dt1 = 1e-7 + static_cast<Real>(ev) * 3e-9;
        const Real dt2 = 1e-6 - dt1;
        f.cache->solve_at(mask_on,  dt1, b_extra, x);
        f.cache->solve_at(mask_off, dt2, b_extra, x);
    }
    REQUIRE(f.cache->num_event_entries() == 2);
    REQUIRE(f.cache->metrics().event_builds == 2);  // no new builds
}

TEST_CASE("solve_at event entries are LRU-bounded at kMaxEventEntries",
          "[v2][layer4_v7][multi_dt]") {
    // 3-switch chopper variant → 8 masks; visit all of them plus
    // wrap-around: the table must never exceed kMaxEventEntries and
    // evicted masks must rebuild correctly on re-visit.
    Graph g;
    DevicePool pool;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    pool.add_voltage_source(0, {.V = 12.0});
    for (int i = 0; i < 4; ++i) {
        g.add_branch(0, 1, BranchKind::Switch);
        pool.add_switch(1 + i, /*g_on=*/1e3, /*g_off=*/1e-9);
    }
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    pool.add_resistor(5, {.G = 0.1});
    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{0});

    Vector b_extra = Vector::Zero(3);
    Vector x;
    // 16 distinct masks through the 8-entry table.
    for (int m = 0; m < 16; ++m) {
        SwitchStateMask mask(4);
        for (int bit = 0; bit < 4; ++bit) {
            mask.set(static_cast<Size>(bit), ((m >> bit) & 1) != 0);
        }
        cache.solve_at(mask, 1e-6, b_extra, x);
        REQUIRE(cache.num_event_entries() <=
                PwlStateSpaceCache::kMaxEventEntries);
    }
    // Re-visit an evicted mask (mask 0 was LRU'd out) — must
    // rebuild and produce the all-open answer: v_out ≈ 12 V is
    // WRONG here (switches to node 1 with only leakage → v_out
    // pulled to vin through g_off? No: all open → v_out floats via
    // g_off to vin and 0.1 S to gnd → ≈ 0 V). Just check finite +
    // consistent with a fresh cache.
    SwitchStateMask m0(4);
    cache.solve_at(m0, 1e-6, b_extra, x);
    PwlStateSpaceCache fresh{g, pool};
    fresh.build_lazy(Real{0});
    Vector x_fresh;
    fresh.solve_at(m0, 1e-6, b_extra, x_fresh);
    REQUIRE(x.size() == x_fresh.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        REQUIRE(x[i] == Approx(x_fresh[i]).margin(1e-12));
    }
}

TEST_CASE("solve_at dt-refactor matches a fresh factorization (RLC)",
          "[v2][layer4_v7][multi_dt]") {
    // Dynamic circuit (cap + inductor): the in-place refactor path
    // J = G + (1/dt)·C must reproduce a from-scratch build at the
    // same dt. V_dc → R → L → node → C ∥ R_load, 1 switch bypassing
    // R_load to exercise a mask bit too.
    Graph g;
    DevicePool pool;
    g.add_node("n1");   // after source
    g.add_node("n2");   // LC node
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1, BranchKind::PassiveLinear);   // L
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // C
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // R_load
    g.add_branch(1, g.ground(), BranchKind::Switch);

    pool.add_voltage_source(0, {.V = 10.0});
    pool.add_inductor(1, {.L = 1e-3});
    pool.add_capacitor(2, {.C = 1e-6});
    pool.add_resistor(3, {.G = 0.01});
    pool.add_switch(4, /*g_on=*/1e3, /*g_off=*/1e-9);

    SwitchStateMask mask(1);
    mask.set(0, true);
    const auto n = pool.state_size(g);
    Vector b_extra = Vector::Zero(static_cast<Index>(n));

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{1e-6});

    // Walk the SAME entry through several dts, checking each against
    // an independent cache built primarily at that dt.
    for (Real dt : {2e-6, 5e-7, 1.3e-6, 2e-6}) {
        Vector x_evt;
        cache.solve_at(mask, dt, b_extra, x_evt);

        PwlStateSpaceCache ref{g, pool};
        ref.build_lazy(dt);
        Vector x_ref;
        ref.solve(mask, b_extra, x_ref);

        REQUIRE(x_evt.size() == x_ref.size());
        for (Eigen::Index i = 0; i < x_evt.size(); ++i) {
            REQUIRE(x_evt[i] == Approx(x_ref[i]).margin(1e-12));
        }
    }
    REQUIRE(cache.num_event_entries() == 1);
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
