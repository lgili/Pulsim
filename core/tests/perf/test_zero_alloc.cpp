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

#include <algorithm>
#include <atomic>
#include <cmath>
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

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/record/recorder.hpp"
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


TEST_CASE("solve_at refactor of a resident entry at a new dt is allocation-free",
          "[v2][perf][zero_alloc]") {
    // update-trbdf2-factor-reuse part 2: in-place J = G + (1/dt)·C on the
    // entry's own pattern, and the LU keeps its work vectors and L/U
    // capacity between factorisations.
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});

    SwitchStateMask m_on(2);
    m_on.set(0, true);
    const auto n = static_cast<Index>(f.pool.state_size(f.g));
    Vector b_extra = Vector::Zero(n);
    Vector x;

    f.cache->solve_at(m_on, 5e-7, b_extra, x);   // build entry
    f.cache->solve_at(m_on, 4e-7, b_extra, x);   // first refactor

    const auto refactors_before = f.cache->metrics().event_refactors;
    Eigen::internal::set_is_malloc_allowed(false);
    const std::size_t before = g_new_count.load();
    Real dt = 3e-7;
    for (int k = 0; k < 100; ++k, dt *= 1.01) {
        f.cache->solve_at(m_on, dt, b_extra, x);
    }
    const std::size_t after = g_new_count.load();
    Eigen::internal::set_is_malloc_allowed(true);

    if (kCountsMeaningful) {
        REQUIRE(after - before == 0);
    }
    REQUIRE(f.cache->metrics().event_refactors - refactors_before == 100);
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
    // The resolution diagnostics' three one-off buffers are counted
    // separately below.
    opts.resolution_diagnostics = false;

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

    // add-fixed-step-resolution-diagnostics: on, it adds one-off
    // buffers (workspace, index buffer, `lte` reserve, the ringing
    // records this every-step toggling makes), never a per-step cost.
    {
        std::size_t n_rec = 0;
        auto count_on = [&](Real t_end) {
            auto on_opts = opts;
            on_opts.resolution_diagnostics = true;
            on_opts.t_end = t_end;
            const std::size_t b0 = g_new_count.load();
            auto on = solver::run_transient(*f.cache, f.g, f.pool,
                                              on_opts, switch_fn);
            const std::size_t d = g_new_count.load() - b0;
            n_rec = on.resolution.ringing.size();
            return d;
        };
        const std::size_t on_2k = count_on(2e-3);
        const std::size_t rec_2k = n_rec;
        const std::size_t on_4k = count_on(4e-3);
        INFO("with resolution diagnostics: " << on_2k << " / " << on_4k
             << " (ringing records " << rec_2k << " / " << n_rec << ")");
        if (kCountsMeaningful) {
            // Twice the steps: at most one more growth of the ringing
            // record vector (11 -> 20 records here), nothing per step.
            REQUIRE(on_4k <= on_2k + 1);
            REQUIRE(on_2k < delta + 16);
        }
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
    opts.resolution_diagnostics = false;   // counted in the test above

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
    // The same run at 4000 steps: O(1) means the count does not move.
    opts.t_end = 4e-3;
    const std::size_t before4 = g_new_count.load();
    auto res4 = solver::run_transient(cache, g, pool, opts, switch_fn);
    const std::size_t delta4 = g_new_count.load() - before4;
    REQUIRE(res4.states.size() == 4001);
    INFO("operator-new count for 4000-step diode run: " << delta4);
    if (kCountsMeaningful) {
        REQUIRE(delta4 == delta);
        // measured: 8; 18 with the pwl mode guard's per-run set-up (its
        // port model, diode names, passivity record), 18 at 4000 steps
        // too; +1 with the snapshot's dev_layout (constant, not per step)
        REQUIRE(delta < 32);
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
    opts.discontinuity_restart = true;   // the split requires the rule

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

TEST_CASE("Newton iteration on a run workspace is allocation-free",
          "[v2][perf][zero_alloc][newton]") {
    // update-newton-persistent-solver 3.6 / 4.4: after the first call a
    // Newton iteration allocates nothing — J_nl keeps its pattern, the
    // combined Jacobian is written in place, and the in-house LU
    // refactorises into its own L/U capacity.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "p", "gnd", 24.0);
    b.add_switch("S", "p", "sw", 1e3, 1e-9);
    b.add_nonlinear_diode("D", "gnd", "sw", models::IdealDiode::Params{});
    b.add_inductor("L", "sw", "o", 1e-4);
    b.add_capacitor("C", "o", "gnd", 4.7e-5);
    b.add_resistor("R", "o", "gnd", 10.0);
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-7);
    SwitchStateMask m(1);
    m.set(0, true);
    const auto& seg = cache.lookup(m);
    const auto n = static_cast<Index>(seg.state_size);
    NonlinearRefreshFn rf = [](const Vector& x, sparse::Matrix& J, Vector& f,
                               const Graph& g, const DevicePool& p) {
        return refresh_smooth_diodes(x, J, f, g, p);
    };
    NewtonWorkspace w;   // in-house LU
    w.iterations_per_solve.reserve(4096);   // the run's trace, amortised
    Vector x = Vector::Zero(n);
    Vector be = Vector::Zero(n);
    for (int k = 0; k < 3; ++k) {   // first calls size every buffer
        be.setConstant(1e-3 * k);
        solve_with_newton_in_place(seg, rf, b.graph(), b.pool(), x, be,
                                   Size{50}, Real{1e-9}, Real{1e-9}, false,
                                   false, &w);
    }
    const auto it0 = w.iterations;
    Eigen::internal::set_is_malloc_allowed(false);
    const std::size_t before = g_new_count.load();
    for (int k = 3; k < 203; ++k) {
        be.setConstant(1e-3 * k);
        solve_with_newton_in_place(seg, rf, b.graph(), b.pool(), x, be,
                                   Size{50}, Real{1e-9}, Real{1e-9}, false,
                                   false, &w);
    }
    const std::size_t after = g_new_count.load();
    Eigen::internal::set_is_malloc_allowed(true);
    INFO("Newton iterations counted: " << (w.iterations - it0)
         << ", operator-new calls: " << (after - before));
    REQUIRE(w.iterations - it0 > 200);   // more than one iteration per call
    if (kCountsMeaningful) {
        REQUIRE(after - before == 0);
    }
    REQUIRE(w.analyses == 1);
}

TEST_CASE("event queue: no allocation after start (ticks, edges, end)",
          "[v2][perf][zero_alloc][event_queue]") {
    using namespace pulsim::events;
    // A three-phase inverter's worth of sources: two tick rates, a
    // dead-time gate oracle (six edges per 100 µs carrier period), the end.
    EventQueue q;
    const auto t1 = q.add_periodic(Kind::ControlTick, Rate{1e-4});
    q.add_periodic(Kind::ControlTick, Rate{2.5e-5, 1e-6});
    const Real T = 1e-4;
    const auto gate = q.add_function(Kind::GateEdge, [T](Real t) {
        const Real k = std::floor(t / T);
        for (const Real f : {0.1, 0.12, 0.4, 0.42, 0.7, 0.72, 1.1}) {
            const Real e = (k + std::floor(f)) * T + (f - std::floor(f)) * T;
            if (e > t) return e;
        }
        return (k + 2) * T;
    });
    q.add_once(Kind::EndOfRun, 1.0);
    q.start(0.0, 1.0);
    Real t = 0.0;
    std::size_t n_events = 0;
    const std::size_t before = g_new_count.load();
    for (int step = 0; step < 10000; ++step) {   // 1 µs steps
        n_events += q.take_due(t, 1e-15, kind_bit(Kind::ControlTick)).size();
        const Real e = q.next_after(gate, t);
        Real t_next = t + 1e-6;
        t_next = std::min({t_next, e, q.next_of(Kind::ControlTick)});
        if (step % 97 == 0) {
            q.invalidate(gate);
        }
        t = t_next;
    }
    const std::size_t after = g_new_count.load();
    INFO("events taken: " << n_events);
    REQUIRE(q.taken(t1) > 0);
    if (kCountsMeaningful) {
        REQUIRE(after - before == 0);
    }
}

TEST_CASE("fixed engine: control ticks from the queue and step hooks",
          "[v2][perf][zero_alloc][event_queue]") {
    // add-unified-event-queue group 5: the kernel schedules the control
    // ticks (integer k, instants < t_end) and calls the accepted-step and
    // commutation hooks; neither allocates per step.
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});
    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.dt = 1e-6;
    opts.resolution_diagnostics = false;
    solver::SwitchScheduleFn switch_fn = [](Real t) {
        SwitchStateMask m(2);
        const auto k = static_cast<int>(t / 1e-6 + 0.5);
        m.set(0, (k / 10) % 2 != 0);   // toggles every 10 steps
        return m;
    };
    std::vector<std::uint32_t> order;   // tick sources, then 99 = observer
    order.reserve(16384);
    events::TickPlan plan;
    plan.rates = {events::Rate{5e-5}, events::Rate{3e-5}};   // 20, 33.3 kHz
    std::uint64_t k_seen[2] = {0, 0};
    std::size_t bad_info = 0;
    plan.fn = [&](std::uint32_t j, const events::TickInfo& ti, Real t,
                  const Vector&) {
        order.push_back(j);
        // A grid point up to the take tolerance (1e-9*dt) before the
        // scheduled instant counts as on it.
        bad_info += ti.k != k_seen[j]++ || t < ti.scheduled_t - 1e-15;
    };
    solver::StepObserverFn obs = [&](Real, const Vector&) {
        order.push_back(99);
    };
    events::StepHooks hooks;
    std::size_t n_acc = 0, n_com = 0, bad_com = 0;
    hooks.on_accept([&](Real t0, Real t1, const Vector&,
                        const SwitchStateMask&) {
        ++n_acc;
        bad_com += !(t1 > t0);
    });
    hooks.on_commutation([&](const events::Commutation& c) {
        ++n_com;
        bad_com += c.before == c.after;
    });
    auto run = [&](Real t_end) {
        opts.t_end = t_end;
        order.clear();
        k_seen[0] = k_seen[1] = 0;
        n_acc = n_com = bad_info = 0;
        const std::size_t b0 = g_new_count.load();
        auto r = solver::run_transient(*f.cache, f.g, f.pool, opts,
                                       switch_fn, {}, false, {}, obs, nullptr,
                                       {}, nullptr, {}, {}, &plan, &hooks);
        return std::pair{g_new_count.load() - b0, std::move(r)};
    };
    run(2e-3);   // warm-up: lazy masks
    auto [d2, r2] = run(2e-3);
    // ceil(2e-3 / T): 40 and 67 ticks, each before the step's observer.
    REQUIRE(r2.control_tick_counts.size() == 2);
    CHECK(r2.control_tick_counts[0] == 40);
    CHECK(r2.control_tick_counts[1] == 67);
    CHECK(bad_info == 0);
    // Every period is a multiple of dt: no lateness.
    CHECK(r2.control_tick_jitter_max < 1e-9 * opts.dt);
    REQUIRE(order.size() >= 3);
    CHECK(order[0] == 0);
    CHECK(order[1] == 1);
    CHECK(order[2] == 99);            // t_start: both ticks, then observer
    CHECK(n_acc == 2000);
    CHECK(n_com == 200);              // a toggle every 10 steps
    CHECK(bad_com == 0);
    auto [d4, r4] = run(4e-3);
    CHECK(r4.control_tick_counts[0] == 80);
    INFO("operator-new: 2000 steps " << d2 << ", 4000 steps " << d4);
    if (kCountsMeaningful) {
        CHECK(d4 <= d2 + 2);          // no per-step or per-tick allocation
    }
}

TEST_CASE("fixed engine: a recorder fed in-loop from the step hooks does not allocate per step",
          "[v2][perf][zero_alloc][record]") {
    // add-probes-and-recorders: statistics-only and decimating recorders
    // on the accepted-step hook; their windows equal the kernel-side pass
    // over the stored states (the same samples, the same arithmetic).
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});
    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.dt = 1e-6;
    opts.resolution_diagnostics = false;
    solver::SwitchScheduleFn switch_fn = [](Real t) {
        SwitchStateMask m(2);
        const auto k = static_cast<int>(t / 1e-6 + 0.5);
        m.set(0, (k / 10) % 2 != 0);
        return m;
    };
    auto make = [](record::RecorderSpec::Keep keep) {
        record::RecorderSpec spec;
        spec.keep = keep;
        spec.every_n = 50;
        spec.windows = {{1.05e-4, 1.8995e-3}};
        spec.period = 2e-5;
        spec.offset = 0.5e-6;
        record::Recorder r({record::Probe::compile("v(vout)"),
                            record::Probe::compile("v(vin) - v(vout)")},
                           spec);
        const std::vector<std::string> names{"v(vin)", "v(vout)"};
        r.bind(names);
        r.reserve(4100, 4e-3);
        return r;
    };
    auto run = [&](Real t_end, record::Recorder& rec) {
        opts.t_end = t_end;
        events::StepHooks hooks;
        record::attach(hooks, rec, {0, 1});
        const std::size_t b0 = g_new_count.load();
        auto r = solver::run_transient(*f.cache, f.g, f.pool, opts, switch_fn, {}, false,
                                       {}, {}, nullptr, {}, nullptr, {}, {}, nullptr, &hooks);
        return std::pair{g_new_count.load() - b0, std::move(r)};
    };
    for (const auto keep : {record::RecorderSpec::Keep::None, record::RecorderSpec::Keep::Decimate}) {
        auto warm = make(keep);
        (void)run(2e-3, warm);   // lazy masks
        auto r2 = make(keep);
        auto r4 = make(keep);
        auto [d2, res2] = run(2e-3, r2);
        auto [d4, res4] = run(4e-3, r4);
        INFO("operator-new: 2000 steps " << d2 << ", 4000 steps " << d4);
        if (kCountsMeaningful) {
            CHECK(d4 <= d2 + 2);
        }
        if (keep == record::RecorderSpec::Keep::Decimate) {
            CHECK(r4.values().capacity() == 2 * (4100 / 50 + 1));   // reserved, never regrown
            CHECK(r4.times().size() == 80);                         // steps 0, 50, ... of 4000
        } else {
            CHECK(r4.values().capacity() == 0);
        }
        // the same windows from the stored states (row 0 = t_start is not
        // an accepted step: the hook record starts at the first step's end)
        auto table = make(keep);
        const Size n = res2.times.size(), m = static_cast<Size>(res2.states.cols());
        std::vector<Real> rd;
        for (Size k = 1; k < n; ++k) {
            rd.push_back(res2.states.data()[k * m + 0]);
            rd.push_back(res2.states.data()[k * m + 1]);
        }
        record::feed_table(table, std::span<const Real>(res2.times).subspan(1), rd, 2, {}, {});
        REQUIRE(r2.n_samples() == n - 1);
        for (Size p = 0; p < 2; ++p) {
            CHECK(r2.windows()[p].complete());
            CHECK(r2.windows()[p].integral == Approx(table.windows()[p].integral).epsilon(1e-12));
            CHECK(r2.windows()[p].max == table.windows()[p].max);
        }
        CHECK(r2.periods()[10 * 2].avg() == Approx(table.periods()[10 * 2].avg()).epsilon(1e-12));
    }
}

TEST_CASE("fixed engine: a tick instant within 1e-9*dt of t_end does not fire",
          "[event_queue]") {
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});
    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end = 1e-4;
    opts.dt = 1e-6;
    solver::SwitchScheduleFn sw = [](Real) { return SwitchStateMask(2); };
    events::TickPlan plan;
    plan.rates = {events::Rate{2.5e-5}};   // 0, 25, 50, 75 us; 100 us = t_end
    std::uint64_t n = 0;
    plan.fn = [&](std::uint32_t, const events::TickInfo&, Real, const Vector&) {
        ++n;
    };
    auto r = solver::run_transient(*f.cache, f.g, f.pool, opts, sw, {}, false,
                                   {}, {}, nullptr, {}, nullptr, {}, {}, &plan);
    CHECK(n == 4);
    CHECK(r.control_tick_counts[0] == 4);
}

TEST_CASE("stored-charge kinds (Lauritzen, IGBT tail): allocation count is "
          "O(1) on both device paths",
          "[v2][perf][zero_alloc][device_path]") {
    // add-unified-device-interface: the two local-row laws (DECISIONS #28)
    // run through `stamp_fq_local` on 'interface' and through their hand
    // stamps on 'legacy'; neither may allocate per step. A gated chopper:
    // the IGBT switches every 50 µs so the tail charge builds and decays and
    // the Lauritzen diode freewheels, inside the counted window.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "vcc", "gnd", 15.0);   // one new node per call
    b.add_inductor("L", "vcc", "c", 1e-4);
    b.add_pulse_voltage_source("VG", "g", "gnd", 0.0, 15.0, 0.0, 2.5e-5, 5e-5);
    b.add_igbt_level1("Q", "c", "gnd", "g", 1.5, 0.05, 5.0, 10.0, 0.01, false,
                      1e-6, 0.3);
    b.add_lauritzen_diode("DL", "c", "vcc");
    b.add_resistor("R", "c", "vcc", 50.0);
    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(Real{1e-6});
    const solver::SwitchScheduleFn switch_fn = [](Real) { return SwitchStateMask(0); };
    for (const auto path : {device::DevicePath::Legacy, device::DevicePath::Interface}) {
        INFO("device path " << static_cast<int>(path));
        solver::SimulationOptions opts;
        opts.t_start = 0.0;
        opts.t_end = 1e-3;
        opts.dt = 1e-6;
        opts.resolution_diagnostics = false;
        opts.device_path = path;
        {
            auto warm = solver::run_transient(cache, b.graph(), b.pool(), opts, switch_fn);
            REQUIRE(warm.states.size() == 1001);
        }
        const std::size_t before = g_new_count.load();
        auto res = solver::run_transient(cache, b.graph(), b.pool(), opts, switch_fn);
        const std::size_t delta = g_new_count.load() - before;
        REQUIRE(res.states.size() == 1001);
        opts.t_end = 4e-3;
        const std::size_t before4 = g_new_count.load();
        auto res4 = solver::run_transient(cache, b.graph(), b.pool(), opts, switch_fn);
        const std::size_t delta4 = g_new_count.load() - before4;
        REQUIRE(res4.states.size() == 4001);
        INFO("operator-new count: 1000 steps " << delta << ", 4000 steps " << delta4);
        if (kCountsMeaningful) {
            // 3000 more steps and 60 more gate edges: a per-step
            // allocation would add >= 3000. Measured: 91 -> 93 on BOTH
            // paths (the +2 is not the laws'; amortised growth of the run's
            // per-event records over 4x the edges is the likely source).
            CHECK(delta4 >= delta);
            CHECK(delta4 <= delta + 6);
            CHECK(delta < 128);
        }
    }
}

namespace {
// The fixed engine's skeleton driver is the default since slice 3c; the
// legacy loop is internal, PULSIM_PWL_STEPPER_LEGACY=1
// (refactor-unified-transient-stepper DECISIONS 17).
void set_pwl_twin(bool on) {
#ifdef _WIN32
    _putenv_s("PULSIM_PWL_STEPPER_LEGACY", on ? "0" : "1");
#else
    setenv("PULSIM_PWL_STEPPER_LEGACY", on ? "0" : "1", 1);
#endif
}
}  // namespace

TEST_CASE("fixed engine twin on the step skeleton: the legacy run, its own hooks, no per-step allocation",
          "[v2][perf][zero_alloc][stepper]") {
    SwitchedRC f;
    f.cache->build_lazy(Real{1e-6});
    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.dt = 1e-6;
    opts.resolution_diagnostics = false;
    solver::SwitchScheduleFn switch_fn = [](Real t) {
        SwitchStateMask m(2);
        const auto k = static_cast<int>(t / 1e-6 + 0.5);
        m.set(0, (k / 10) % 2 != 0);   // toggles every 10 steps
        return m;
    };
    events::StepHooks hooks;
    std::size_t n_acc = 0, n_step = 0, n_dev = 0, bad = 0;
    Vector x_end_prev;
    Real t_end_prev = 0.0;
    hooks.on_accept([&](Real, Real, const Vector&, const SwitchStateMask&) { ++n_acc; });
    // probes R1: the step's start state is the previous step's end state
    hooks.on_step([&](Real t0, const Vector& x0, const SwitchStateMask&, Real t1,
                      const Vector& x1, const SwitchStateMask&) {
        bad += !(t1 > t0);
        if (n_step > 0) {
            bad += t0 != t_end_prev;
            bad += !(x0.size() == x_end_prev.size() && x0 == x_end_prev);
        }
        x_end_prev = x1;   // same size after the first step: no allocation
        t_end_prev = t1;
        ++n_step;
    });
    hooks.on_device_state([&](Real, const Vector&, const pwl::StatefulHistories&) { ++n_dev; });
    auto run = [&](bool twin, Real t_end) {
        set_pwl_twin(twin);
        opts.t_end = t_end;
        n_acc = n_step = n_dev = bad = 0;
        const std::size_t b0 = g_new_count.load();
        auto r = solver::run_transient(*f.cache, f.g, f.pool, opts,
                                       switch_fn, {}, false, {}, {}, nullptr,
                                       {}, nullptr, {}, {}, nullptr, &hooks);
        const std::size_t d = g_new_count.load() - b0;
        set_pwl_twin(true);   // back to the default
        return std::pair{d, std::move(r)};
    };
    auto [d_old, r_old] = run(false, 2e-3);
    CHECK(n_acc == 2000);
    // slice 3c: the hooks live in the shared phases, so the legacy loop
    // calls them too
    CHECK(n_step == 2000);
    CHECK(n_dev == 2000);
    run(true, 2e-3);      // warm-up
    auto [d2, r2] = run(true, 2e-3);
    CHECK(n_acc == 2000);
    CHECK(n_step == 2000);
    CHECK(n_dev == 2000);
    CHECK(bad == 0);
    // Two code paths: within 1e-12 of scale here (bit identity is the
    // local gate, benchmarks/tools/pwl_twin_identity.py).
    REQUIRE(r2.times.size() == r_old.times.size());
    REQUIRE(r2.states.size() == r_old.states.size());
    Real worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < r_old.states.size(); ++i) {
        CHECK(r2.times[i] == Catch::Approx(r_old.times[i]).margin(1e-18));
        worst = std::max(worst, (r2.states[i] - r_old.states[i]).cwiseAbs().maxCoeff());
        scale = std::max(scale, r_old.states[i].cwiseAbs().maxCoeff());
    }
    CHECK(worst <= 1e-12 * scale);
    auto [d4, r4] = run(true, 4e-3);
    CHECK(n_step == 4000);
    INFO("operator-new: twin 2000 steps " << d2 << ", 4000 steps " << d4 << ", legacy " << d_old);
    if (kCountsMeaningful) {
        CHECK(d4 <= d2 + 2);   // no per-step allocation on the twin
    }
}
