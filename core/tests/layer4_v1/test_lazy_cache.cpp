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
#include "pulsim/solver/run_transient.hpp"
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

// =============================================================================
// v2.0 Phase 1 — LRU byte-budget eviction on the lazy segment cache
// (audit finding no-mode-cache-eviction)
// =============================================================================

TEST_CASE("Lazy cache evicts LRU segments under a byte budget",
          "[v2][layer4_v6][lazy_cache][lru]") {
    // 4-switch fixture → 16 masks. With a budget sized for only a
    // few segments, visiting all masks must (a) keep resident bytes
    // near the budget, (b) count evictions, (c) stay CORRECT on
    // re-visit of an evicted mask.
    Graph g;
    DevicePool pool;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    pool.add_voltage_source(0, {.V = 12.0});
    for (int i = 0; i < 4; ++i) {
        g.add_branch(0, 1, BranchKind::Switch);
        pool.add_switch(1 + i, /*g_on=*/1.0, /*g_off=*/1e-9);
    }
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    pool.add_resistor(5, {.G = 0.1});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{0});
    REQUIRE(cache.segment_budget_bytes() ==
            PwlStateSpaceCache::kDefaultSegmentBudgetBytes);

    // Measure one segment's cost, then budget for ~3 of them.
    Vector b_extra = Vector::Zero(3);
    Vector x;
    SwitchStateMask probe(4);
    cache.solve(probe, b_extra, x);
    const std::size_t one = cache.segment_cache_bytes();
    REQUIRE(one > 0);
    cache.set_segment_budget_bytes(3 * one + one / 2);

    auto solve_mask = [&](int m, Vector& out) {
        SwitchStateMask mask(4);
        for (int bit = 0; bit < 4; ++bit) {
            mask.set(static_cast<Size>(bit), ((m >> bit) & 1) != 0);
        }
        cache.solve(mask, b_extra, out);
        return mask;
    };

    for (int m = 0; m < 16; ++m) {
        Vector xi;
        solve_mask(m, xi);
        REQUIRE(cache.segment_cache_bytes() <=
                cache.segment_budget_bytes());
    }
    // Deterministic for this fixture, so assert it EXACTLY rather
    // than with slack that would hide an off-by-one in the eviction
    // while-condition or the `resident + incoming` arithmetic
    // (adversarial-review finding F9). Every segment here has the
    // same footprint, the budget holds 3, mask 0 was already
    // resident from the probe, so masks 1..15 are 15 inserts and
    // 15 - 3 = 12 of them evict... plus the probe's own segment,
    // giving 13 evictions and 3 resident at the end.
    REQUIRE(cache.num_built_segments() == 3);
    REQUIRE(cache.metrics().segment_evictions == 13);

    // Re-visit mask 1 (long evicted): rebuild must give the same
    // answer as an unbounded fresh cache. k=1 switch closed →
    // v_out = 12·0.1/(0.1 + 1·1.0 + 3e-9).
    Vector x_again;
    solve_mask(1, x_again);
    PwlStateSpaceCache fresh{g, pool};
    fresh.build_lazy(Real{0});
    SwitchStateMask m1(4);
    m1.set(0, true);
    Vector x_fresh;
    fresh.solve(m1, b_extra, x_fresh);
    for (Eigen::Index i = 0; i < x_again.size(); ++i) {
        REQUIRE(x_again[i] == Approx(x_fresh[i]).margin(1e-12));
    }
}

TEST_CASE("Budget 0 disables eviction; eager build never evicts",
          "[v2][layer4_v6][lazy_cache][lru]") {
    ChopperFixture f;
    f.cache->set_segment_budget_bytes(0);
    f.cache->build_lazy(Real{0});

    Vector zero_b = Vector::Zero(3);
    Vector x;
    SwitchStateMask m_off(1), m_on(1);
    m_on.set(0, true);
    f.cache->solve(m_off, zero_b, x);
    f.cache->solve(m_on, zero_b, x);
    REQUIRE(f.cache->num_built_segments() == 2);
    REQUIRE(f.cache->metrics().segment_evictions == 0);

    // Eager build with an absurdly small budget: still holds 2^N.
    f.cache->set_segment_budget_bytes(1);
    f.cache->build(Real{0});
    REQUIRE(f.cache->num_built_segments() == 2);
    REQUIRE(f.cache->metrics().segment_evictions == 0);
}

TEST_CASE("run_transient under a tiny segment budget matches unbounded",
          "[v2][layer4_v6][lazy_cache][lru][integration]") {
    // Adversarial-review gap: eviction was only exercised via direct
    // cache.solve calls. Drive the REAL engine: a switched RC with a
    // per-step mask walk that revisits masks after eviction, once
    // with the default (unbounded-in-practice) budget and once with
    // a budget of ~2 segments. Waveforms must match exactly.
    auto build_fixture = [](std::unique_ptr<PwlStateSpaceCache>& cache,
                             Graph& g, DevicePool& pool) {
        g.add_node("vin");
        g.add_node("vout");
        g.add_branch(0, g.ground(), BranchKind::Source);
        pool.add_voltage_source(0, {.V = 12.0});
        for (int i = 0; i < 3; ++i) {
            g.add_branch(0, 1, BranchKind::Switch);
            pool.add_switch(1 + i, /*g_on=*/1.0, /*g_off=*/1e-9);
        }
        g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
        pool.add_resistor(4, {.G = 0.1});
        g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
        pool.add_capacitor(5, {.C = 1e-6});
        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
    };

    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 2e-4;
    opts.dt      = 1e-6;

    // Mask walk cycling through 6 of the 8 states, revisiting each
    // ~33 times across the run.
    solver::SwitchScheduleFn switch_fn = [](Real t) {
        topology::SwitchStateMask m(3);
        const auto k = static_cast<int>(t / 1e-6) % 6;
        for (int bit = 0; bit < 3; ++bit) {
            m.set(static_cast<Size>(bit), ((k >> bit) & 1) != 0);
        }
        return m;
    };

    auto run = [&](std::size_t budget, std::uint64_t& evictions) {
        Graph g;
        DevicePool pool;
        std::unique_ptr<PwlStateSpaceCache> cache;
        build_fixture(cache, g, pool);
        cache->build_lazy(opts.dt);
        Vector x0;
        // Prime one segment to size the budget.
        {
            Vector xp;
            Vector zb = Vector::Zero(
                static_cast<Index>(pool.state_size(g)));
            cache->solve(switch_fn(0.0), zb, xp);
        }
        if (budget != 0) {
            cache->set_segment_budget_bytes(
                2 * cache->segment_cache_bytes() +
                cache->segment_cache_bytes() / 2);
        }
        auto res = solver::run_transient(*cache, g, pool, opts,
                                           switch_fn);
        evictions = cache->metrics().segment_evictions;
        return res;
    };

    std::uint64_t ev_unbounded = 0, ev_tiny = 0;
    auto res_ref  = run(0, ev_unbounded);
    auto res_tiny = run(1, ev_tiny);

    REQUIRE(ev_unbounded == 0);
    REQUIRE(ev_tiny > 0);   // eviction genuinely exercised
    REQUIRE(res_ref.states.size() == res_tiny.states.size());
    for (std::size_t k = 0; k < res_ref.states.size(); ++k) {
        for (Eigen::Index i = 0; i < res_ref.states[k].size(); ++i) {
            REQUIRE(res_tiny.states[k][i] ==
                    Approx(res_ref.states[k][i]).margin(1e-12));
        }
    }
}

TEST_CASE("Evicted masks rebuild from the UPDATED pool after "
          "refactor_parametric",
          "[v2][layer4_v6][lazy_cache][lru][parametric]") {
    // Adversarial-review finding F4: refactor_parametric only walks
    // the masks that are still RESIDENT. Any mask the byte budget
    // already evicted is simply not refactored — correctness then
    // rests entirely on the unpinned assumption that rebuilding an
    // evicted mask re-reads the CURRENT pool rather than a stale
    // snapshot. Pin it.
    Graph g;
    DevicePool pool;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    pool.add_voltage_source(0, {.V = 12.0});
    for (int i = 0; i < 3; ++i) {
        g.add_branch(0, 1, BranchKind::Switch);
        pool.add_switch(1 + i, /*g_on=*/1.0, /*g_off=*/1e-9);
    }
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    const Index r_branch = 4;
    pool.add_resistor(r_branch, {.G = 0.1});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{0});

    auto mask_of = [](int m) {
        SwitchStateMask s(3);
        for (int b = 0; b < 3; ++b) {
            s.set(static_cast<Size>(b), ((m >> b) & 1) != 0);
        }
        return s;
    };
    Vector b_extra = Vector::Zero(3);
    Vector x;

    // Size the budget for ~2 segments, then visit A,B,C,D so A is
    // long gone.
    cache.solve(mask_of(0), b_extra, x);
    cache.set_segment_budget_bytes(2 * cache.segment_cache_bytes());
    for (int m = 0; m < 4; ++m) cache.solve(mask_of(m), b_extra, x);
    REQUIRE(cache.metrics().segment_evictions > 0);

    // Change R while mask 0 is NOT resident.
    auto r = cache.refactor_parametric(r_branch, /*new R=*/2.0);
    REQUIRE(r.masks_processed <= 2);   // only what was resident

    // Re-visiting the evicted mask must give the UPDATED answer.
    Vector x_after;
    cache.solve(mask_of(0), b_extra, x_after);

    PwlStateSpaceCache fresh{g, pool};
    fresh.build_lazy(Real{0});
    Vector x_fresh;
    fresh.solve(mask_of(0), b_extra, x_fresh);
    for (Eigen::Index i = 0; i < x_after.size(); ++i) {
        REQUIRE(x_after[i] == Approx(x_fresh[i]).margin(1e-12));
    }

    // A resident mask must agree too — both routes (refactored and
    // rebuilt) land on the same updated circuit.
    Vector x_res, x_res_fresh;
    cache.solve(mask_of(3), b_extra, x_res);
    fresh.solve(mask_of(3), b_extra, x_res_fresh);
    for (Eigen::Index i = 0; i < x_res.size(); ++i) {
        REQUIRE(x_res[i] == Approx(x_res_fresh[i]).margin(1e-10));
    }
}

TEST_CASE("A no-op refactor_parametric changes nothing",
          "[v2][layer4_v6][lazy_cache][parametric]") {
    // Adversarial-review finding F3-tests: the event-entry clear
    // used to run BEFORE the empty-updates early return, so a
    // documented no-op silently cost a full rebuild of every event
    // solver on the next commutation.
    ChopperFixture f;
    f.cache->build_lazy(Real{0});
    Vector zero_b = Vector::Zero(3);
    Vector x;
    SwitchStateMask m_on(1);
    m_on.set(0, true);
    f.cache->solve(m_on, zero_b, x);
    f.cache->solve_at(m_on, 1e-6, zero_b, x);
    REQUIRE(f.cache->num_event_entries() == 1);
    const auto built_before = f.cache->metrics().event_builds;

    auto r = f.cache->refactor_parametric(
        std::span<const ParametricUpdate>{}, 
        ParametricRefactorMode::AllActive);
    REQUIRE(r.masks_processed == 0);
    REQUIRE(f.cache->num_event_entries() == 1);   // untouched

    f.cache->solve_at(m_on, 1e-6, zero_b, x);
    REQUIRE(f.cache->metrics().event_builds == built_before);
}

TEST_CASE("A budget smaller than one segment still stores it",
          "[v2][layer4_v6][lazy_cache][lru]") {
    // Refusing to build would turn a memory knob into a
    // wrong-answer knob, so the incoming segment is always
    // inserted even when it alone exceeds the budget.
    ChopperFixture f;
    f.cache->build_lazy(Real{0});
    f.cache->set_segment_budget_bytes(1);
    Vector zero_b = Vector::Zero(3);
    Vector x_on, x_off;
    SwitchStateMask m_on(1), m_off(1);
    m_on.set(0, true);
    f.cache->solve(m_on, zero_b, x_on);
    REQUIRE(f.cache->num_built_segments() == 1);
    f.cache->solve(m_off, zero_b, x_off);
    // Second insert evicted the first, but is itself resident and
    // correct.
    REQUIRE(f.cache->num_built_segments() == 1);
    REQUIRE(f.cache->metrics().segment_evictions >= 1);
    PwlStateSpaceCache ref{f.g, f.pool};
    ref.build_lazy(Real{0});
    Vector want;
    ref.solve(m_off, zero_b, want);
    for (Eigen::Index i = 0; i < x_off.size(); ++i) {
        REQUIRE(x_off[i] == Approx(want[i]).margin(1e-12));
    }
}
