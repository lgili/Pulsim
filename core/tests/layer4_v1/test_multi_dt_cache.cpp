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

TEST_CASE("Failed event refactorize erases the entry; old dt recovers",
          "[v2][layer4_v7][multi_dt][singular]") {
    // Adversarial-review finding EVT-STALE-FACTOR: the solver
    // destroys its previous factor before attempting the new one,
    // so a failed dt-change refactorize must ERASE the entry — a
    // stale entry claiming the old current_dt would make the next
    // solve_at at the previously WORKING dt take the reuse branch
    // and die on an unfactorized solver (logic_error) instead of
    // rebuilding.
    //
    // Singularity rig: node0—gnd carries a NEGATIVE conductance
    // G = −1 S and a 0.5 F capacitor → J(dt) = −1 + 1/dt, exactly
    // singular at dt = 1. A current source keeps b non-trivial.
    Graph g;
    DevicePool pool;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);  // R<0
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);  // C
    g.add_branch(0, g.ground(), BranchKind::Source);         // I

    pool.add_resistor(0, {.G = -1.0});
    pool.add_capacitor(1, {.C = 0.5});
    pool.add_current_source(2, {.I = 1.0});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{1e-3});

    SwitchStateMask m(0);
    Vector b_extra = Vector::Zero(1);
    Vector x;

    // dt = 0.5 → J = −1 + 2 = 1: fine.
    cache.solve_at(m, 0.5, b_extra, x);
    REQUIRE(cache.num_event_entries() == 1);
    REQUIRE(x[0] == Approx(1.0).margin(1e-12));  // x = −b/J = 1

    // dt = 1 → J = 0: numerically singular refactorize.
    REQUIRE_THROWS_AS(cache.solve_at(m, 1.0, b_extra, x),
                       std::runtime_error);
    REQUIRE(cache.num_event_entries() == 0);  // entry erased

    // Retry at the previously working dt: transparent rebuild.
    cache.solve_at(m, 0.5, b_extra, x);
    REQUIRE(x[0] == Approx(1.0).margin(1e-12));
    REQUIRE(cache.num_event_entries() == 1);
}

TEST_CASE("refactor_parametric invalidates event-solver entries",
          "[v2][layer4_v7][multi_dt][parametric]") {
    // The v1.x aux cache silently served STALE factors after a
    // parameter update. The event entries snapshot (G, C, b), so
    // refactor_parametric must drop them; the next solve_at rebuilds
    // from the UPDATED pool.
    Chopper f;
    f.cache->build_lazy(Real{0});

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    Vector b_extra = Vector::Zero(3);
    Vector x_before, x_after, x_fresh;

    f.cache->solve_at(mask_on, 1e-6, b_extra, x_before);
    REQUIRE(f.cache->num_event_entries() == 1);

    // R: 10 Ω → 5 Ω (branch 2 is the load resistor, R-form value).
    auto r = f.cache->refactor_parametric(2, 5.0);
    (void)r;
    REQUIRE(f.cache->num_event_entries() == 0);  // dropped

    f.cache->solve_at(mask_on, 1e-6, b_extra, x_after);

    // Independent cache built AFTER the update agrees.
    PwlStateSpaceCache fresh{f.g, f.pool};
    fresh.build_lazy(Real{0});
    fresh.solve_at(mask_on, 1e-6, b_extra, x_fresh);
    for (Eigen::Index i = 0; i < x_after.size(); ++i) {
        REQUIRE(x_after[i] == Approx(x_fresh[i]).margin(1e-12));
    }
    // And genuinely differs from the pre-update answer.
    REQUIRE(std::abs(x_after[1] - x_before[1]) > 1e-6);
}

TEST_CASE("solve_at dt<=0 falls back to static assembly",
          "[v2][layer4_v7][multi_dt][static]") {
    // Primary cache at dt>0; solve_at(dt=0) must reproduce a
    // static-only (V0) build: caps/inductors skipped entirely.
    Graph g;
    DevicePool pool;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);  // R
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);  // C
    pool.add_voltage_source(0, {.V = 7.0});
    pool.add_resistor(1, {.G = 0.5});
    pool.add_capacitor(2, {.C = 1e-6});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{1e-6});

    SwitchStateMask m(0);
    const auto n = static_cast<Index>(pool.state_size(g));
    Vector b_extra = Vector::Zero(n);
    Vector x_evt, x_ref;
    cache.solve_at(m, Real{0}, b_extra, x_evt);

    PwlStateSpaceCache ref{g, pool};
    ref.build_lazy(Real{0});   // static primary
    ref.solve(m, b_extra, x_ref);
    for (Eigen::Index i = 0; i < x_evt.size(); ++i) {
        REQUIRE(x_evt[i] == Approx(x_ref[i]).margin(1e-12));
    }
}

TEST_CASE("solve_at re-analyzes across a dynamic<->static dt regime change",
          "[v2][layer4_v7][multi_dt][regime]") {
    // Phase-1 audit finding F6: an event entry analyzed at dt > 0
    // holds the DYNAMIC sparsity pattern (capacitor blocks,
    // inductor rows). At dt <= 0 those stamps vanish entirely, so
    // the matrix has a DIFFERENT pattern — refactorizing onto the
    // old analysis violates the solver's "matching pattern"
    // contract. It happened to work only because the in-house LU
    // rebuilds the pattern inside factorize(); it is unsound in
    // general and would corrupt partial_refactor.
    //
    // The entry must be rebuilt, and BOTH regimes must give the
    // same answers as freshly-built caches.
    Graph g;
    DevicePool pool;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);   // R
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);   // C
    pool.add_voltage_source(0, {.V = 9.0});
    pool.add_resistor(1, {.G = 0.5});
    pool.add_capacitor(2, {.C = 2e-6});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{1e-6});
    SwitchStateMask m(0);
    const auto n = static_cast<Index>(pool.state_size(g));
    Vector b_extra = Vector::Zero(n);

    auto reference = [&](Real dt) {
        PwlStateSpaceCache ref{g, pool};
        ref.build_lazy(dt);
        Vector xr;
        ref.solve(m, b_extra, xr);
        return xr;
    };

    // Walk dynamic -> static -> dynamic on the SAME entry.
    for (Real dt : {2e-6, 0.0, 5e-7, 0.0, 2e-6}) {
        Vector x;
        cache.solve_at(m, dt, b_extra, x);
        const Vector want = reference(dt);
        REQUIRE(x.size() == want.size());
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            INFO("dt=" << dt << " row " << i);
            REQUIRE(x[i] == Approx(want[i]).margin(1e-12));
        }
        // Never more than one entry for one mask.
        REQUIRE(cache.num_event_entries() == 1);
    }
    // Each regime flip rebuilt (4 flips across the 5 dts above),
    // rather than silently refactorizing a mismatched analysis.
    REQUIRE(cache.metrics().event_builds >= 4);
}
