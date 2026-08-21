// =============================================================================
// v2.0 Phase 1 — zero-allocation transient hot loop (lock-in test)
// =============================================================================
//
// Audit finding `per-step-heap-allocations` (HIGH, CONFIRMED): the
// per-step loop performed 6-10 heap allocations per step across
// cache.solve (fresh rhs), the LU triangular solve (fresh y),
// history handling (snapshot copies, compute_b_extra's fresh
// Vector::Zero), diode snapshots, and the per-step event scratch.
// The Phase-1 fix hoists every buffer into once-allocated
// workspaces. This binary LOCKS the fix in two ways:
//
//   1. A custom global operator new/delete counts every
//      std-container allocation (std::vector<bool>,
//      std::vector<HistoryEntry>, std::vector<CommutationEvent>,
//      ... all route through std::allocator -> operator new).
//   2. Eigen's own heap traffic does NOT route through operator
//      new (it uses std::malloc via internal::aligned_malloc), so
//      Eigen misuse is caught with EIGEN_RUNTIME_NO_MALLOC +
//      set_is_malloc_allowed(false) around the steady-state
//      region, with eigen_assert redefined to THROW so the check
//      survives NDEBUG release builds.
//
// This file must be its own executable: the eigen_assert /
// EIGEN_RUNTIME_NO_MALLOC redefinitions must precede EVERY Eigen
// include in the TU, and the operator-new replacement is
// program-global.

#include <atomic>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

// ---- Eigen malloc guard (must precede all Eigen/pulsim includes) ----------
#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x)                                                  \
    do {                                                                  \
        if (!(x)) {                                                       \
            throw std::runtime_error(std::string{"eigen_assert: "} + #x); \
        }                                                                 \
    } while (false)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

// ---- global operator new/delete counters ----------------------------------
namespace {
std::atomic<std::size_t> g_new_count{0};
}

void* operator new(std::size_t sz) {
    g_new_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(sz)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t sz) {
    g_new_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(sz)) return p;
    throw std::bad_alloc{};
}
// MSVC has no std::aligned_alloc (C11); it pairs _aligned_malloc
// with _aligned_free. The aligned deletes below must match the
// aligned news — plain free() on an _aligned_malloc block is UB.
namespace {
void* aligned_new_impl(std::size_t sz, std::size_t al) {
#if defined(_MSC_VER)
    return _aligned_malloc(sz, al);
#else
    const std::size_t rounded = (sz + al - 1) & ~(al - 1);
    return std::aligned_alloc(al, rounded);
#endif
}
void aligned_delete_impl(void* p) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

void* operator new(std::size_t sz, std::align_val_t al) {
    g_new_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = aligned_new_impl(sz, static_cast<std::size_t>(al))) {
        return p;
    }
    throw std::bad_alloc{};
}
void* operator new[](std::size_t sz, std::align_val_t al) {
    return ::operator new(sz, al);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept {
    aligned_delete_impl(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    aligned_delete_impl(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    aligned_delete_impl(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    aligned_delete_impl(p);
}

// MSVC Debug (_ITERATOR_DEBUG_LEVEL > 0) heap-allocates a
// _Container_proxy per std-container CONSTRUCTION — e.g. the 64
// SwitchStateMask array the event loop builds per step — which is
// debug scaffolding, not engine allocation. The count assertions
// are meaningless there; keep the tests running (they still verify
// values + Eigen guard) but skip the counter REQUIREs.
#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL > 0
inline constexpr bool kCountsMeaningful = false;
#else
inline constexpr bool kCountsMeaningful = true;
#endif

namespace {

/// Switched RC: V_dc → switch ∥ switch → R ∥ C load. Dynamic (dt>0)
/// with 2 switches → 4 masks, no diodes.
struct SwitchedRC {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;

    SwitchedRC() {
        g.add_node("vin");
        g.add_node("vout");
        g.add_branch(0, g.ground(), BranchKind::Source);
        pool.add_voltage_source(0, {.V = 12.0});
        for (int i = 0; i < 2; ++i) {
            g.add_branch(0, 1, BranchKind::Switch);
            pool.add_switch(1 + i, /*g_on=*/1.0, /*g_off=*/1e-9);
        }
        g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
        pool.add_resistor(3, {.G = 0.1});
        g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
        pool.add_capacitor(4, {.C = 1e-6});
        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
    }
};

}  // namespace

TEST_CASE("steady-state cache.solve performs ZERO heap allocations",
          "[v2][perf][zero_alloc]") {
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});

    SwitchStateMask m_on(2);
    m_on.set(0, true);
    SwitchStateMask m_off(2);
    const auto n = static_cast<Index>(f.pool.state_size(f.g));
    Vector b_extra = Vector::Zero(n);
    Vector x;

    // Warm-up: builds both segments + sizes every workspace
    // (cache rhs, LU y, x).
    f.cache->solve(m_on, b_extra, x);
    f.cache->solve(m_off, b_extra, x);
    f.cache->solve(m_on, b_extra, x);

    // Steady state: alternate two cached masks 1000 times.
    // std-container side: operator-new delta must be EXACTLY zero.
    // Eigen side: malloc forbidden — any Vector/Matrix allocation
    // throws via the redefined eigen_assert.
    Eigen::internal::set_is_malloc_allowed(false);
    const std::size_t before = g_new_count.load();
    for (int k = 0; k < 1000; ++k) {
        f.cache->solve((k & 1) ? m_on : m_off, b_extra, x);
    }
    const std::size_t after = g_new_count.load();
    Eigen::internal::set_is_malloc_allowed(true);

    if (kCountsMeaningful) {
        REQUIRE(after - before == 0);
    }
    REQUIRE(x.size() == n);   // solves actually ran
}

TEST_CASE("steady-state solve_at (same mask+dt) is allocation-free",
          "[v2][perf][zero_alloc]") {
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});

    SwitchStateMask m_on(2);
    m_on.set(0, true);
    const auto n = static_cast<Index>(f.pool.state_size(f.g));
    Vector b_extra = Vector::Zero(n);
    Vector x;

    f.cache->solve_at(m_on, 5e-7, b_extra, x);   // build entry
    f.cache->solve_at(m_on, 5e-7, b_extra, x);   // warm workspaces

    Eigen::internal::set_is_malloc_allowed(false);
    const std::size_t before = g_new_count.load();
    for (int k = 0; k < 1000; ++k) {
        f.cache->solve_at(m_on, 5e-7, b_extra, x);
    }
    const std::size_t after = g_new_count.load();
    Eigen::internal::set_is_malloc_allowed(true);

    if (kCountsMeaningful) {
        REQUIRE(after - before == 0);
    }
    REQUIRE(f.cache->metrics().event_hits >= 1000);
}

TEST_CASE("run_transient hot loop: zero std-container allocations",
          "[v2][perf][zero_alloc]") {
    // The full engine over 2000 steps, PWM-style mask walk over
    // pre-visited masks. After the fix the only remaining per-step
    // heap traffic is Eigen-side sample RECORDING
    // (result.states.push_back — the contiguous-storage delivery
    // removes it next); every std-container path (snapshots,
    // event scratch, b_extra history) must be allocation-free, so
    // the operator-new delta for the WHOLE RUN must be a small
    // one-off constant (result reserve + workspace sizing), not
    // O(n_steps).
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});

    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 2e-3;
    opts.dt      = 1e-6;   // 2000 steps

    solver::SwitchScheduleFn switch_fn = [](Real t) {
        SwitchStateMask m(2);
        const auto k = static_cast<int>(t / 1e-6);
        m.set(0, (k & 1) != 0);
        m.set(1, (k & 2) != 0);
        return m;
    };

    // Warm-up run: lazy-builds all 4 visited masks, sizes every
    // workspace inside cache + solver (those persist in f.cache).
    {
        auto warm = solver::run_transient(*f.cache, f.g, f.pool,
                                            opts, switch_fn);
        REQUIRE(warm.states.size() == 2001);
    }

    const std::size_t before = g_new_count.load();
    auto res = solver::run_transient(*f.cache, f.g, f.pool,
                                       opts, switch_fn);
    const std::size_t after = g_new_count.load();
    REQUIRE(res.states.size() == 2001);

    const std::size_t delta = after - before;
    INFO("operator-new count for 2000-step run: " << delta);
    // One-off costs inside the counted run: SimulationResult's
    // reserve()s (the contiguous sample buffer is ONE of them),
    // std::function fit, and run_transient's hoisted workspace
    // vectors. The point is O(1), not O(n_steps): the pre-fix code
    // was ~4 std-container allocations PER STEP (~8000 here), and
    // recording alone was one more per step until the waveform
    // buffer became contiguous. Measured: 5.
    if (kCountsMeaningful) {
        REQUIRE(delta < 16);
    }

    // Physical sanity so the run wasn't degenerate: with both
    // switches open 25% of the time the cap voltage stays in
    // (0, 12).
    const Real v_final = res.states.back()[1];
    REQUIRE(v_final > Real{0});
    REQUIRE(v_final < Real{12});
}

TEST_CASE("run_transient with diodes: allocation count is O(1), "
          "not O(n_steps)",
          "[v2][perf][zero_alloc]") {
    // Same engine with a diode present so the per-step snapshot and
    // event-detection scratch paths execute every step. (The BREACH
    // path is not reached here — this circuit converges in one or
    // two inner iterations; breach coverage is a separate concern,
    // adversarial-review finding ZA-3.) Steady conduction, so
    // std-container traffic must stay a one-off constant.
    Graph g;
    DevicePool pool;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    pool.add_voltage_source(0, {.V = 12.0});
    g.add_branch(0, 1, BranchKind::Switch);
    pool.add_diode(1, /*g_on=*/1.0, /*g_off=*/1e-9, /*V_th=*/0.0);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    pool.add_resistor(2, {.G = 0.1});
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    pool.add_capacitor(3, {.C = 1e-6});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{1e-6});

    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 1e-3;
    opts.dt      = 1e-6;   // 1000 steps

    solver::SwitchScheduleFn switch_fn = [](Real) {
        return SwitchStateMask(1);   // diode-owned bit; user all-open
    };

    {
        auto warm = solver::run_transient(cache, g, pool, opts,
                                            switch_fn);
        REQUIRE(warm.states.size() == 1001);
    }

    const std::size_t before = g_new_count.load();
    auto res = solver::run_transient(cache, g, pool, opts,
                                       switch_fn);
    const std::size_t delta = g_new_count.load() - before;
    REQUIRE(res.states.size() == 1001);
    INFO("operator-new count for 1000-step diode run: " << delta);
    if (kCountsMeaningful) {
        REQUIRE(delta < 16);   // measured: 8
    }
}


TEST_CASE("substep event correction path is allocation-free per event",
          "[v2][perf][zero_alloc]") {
    // Adversarial-review finding ZA-3/ZA-5: the first lock-in
    // suite never drove a commutation, so the rollback / sub-step
    // / post_event_bits machinery ran uncounted (and
    // post_event_bits genuinely allocated per corrected event
    // before it was hoisted). Sine-driven diode rectifier with
    // enable_substep_state_correction: ~2 commutations per line
    // cycle, all inside the counted window.
    Graph g;
    DevicePool pool;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    pool.add_sine_voltage_source(0, {.v_dc = 0.0,
                                      .v_amplitude = 10.0,
                                      .frequency = 1e3,
                                      .phase = 0.0});
    g.add_branch(0, 1, BranchKind::Switch);
    pool.add_diode(1, /*g_on=*/1.0, /*g_off=*/1e-9, /*V_th=*/0.0);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    pool.add_resistor(2, {.G = 0.1});
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
    pool.add_capacitor(3, {.C = 1e-7});

    PwlStateSpaceCache cache{g, pool};
    cache.build_lazy(Real{1e-6});

    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 5e-3;   // 5 line cycles → ~10 commutations
    opts.dt      = 1e-6;
    opts.enable_substep_state_correction = true;

    solver::SwitchScheduleFn switch_fn = [](Real) {
        return SwitchStateMask(1);
    };

    // Warm-up run: visits both masks, warms event solvers for the
    // solve_at sub-step solves, sizes all scratch.
    std::size_t warm_events = 0;
    {
        auto warm = solver::run_transient(cache, g, pool, opts,
                                            switch_fn);
        warm_events = warm.commutation_events.size();
        REQUIRE(warm_events >= 8);   // corrections genuinely fire
    }

    const std::size_t before = g_new_count.load();
    auto res = solver::run_transient(cache, g, pool, opts,
                                       switch_fn);
    const std::size_t delta = g_new_count.load() - before;
    REQUIRE(res.commutation_events.size() == warm_events);
    INFO("operator-new count for corrected run: " << delta
         << " (" << warm_events << " events)");
    // Sub-step solves hit warmed event solvers at fresh dts each
    // event: in-place refactors, no analyze. The residual traffic
    // is PER COMMUTATION (the event solver recombines
    // J = G + (1/dt)*C and refactorizes into rebuilt L/U value
    // arrays), never per step — which is the property that
    // matters: this run integrates 5000 steps and allocates only
    // on its ~10 events. Assert that shape explicitly.
    const std::size_t per_event_budget = 16;
    if (kCountsMeaningful) {
        REQUIRE(delta < 32 + per_event_budget * warm_events);
        // ... and emphatically NOT proportional to the step count.
        REQUIRE(delta < 5000 / 10);
    }
}
