// =============================================================================
// Layer 5 — StateTrajectory (contiguous waveform storage) + store_every
// =============================================================================
//
// v2.0 Phase 1, audit finding `waveform-storage-vector-of-vectors`.
// The v1.x `std::vector<Vector>` states container allocated one heap
// block per recorded sample. StateTrajectory stores the run as ONE
// row-major buffer while keeping the read API source-compatible.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/state_trajectory.hpp"
#include "pulsim/topology/graph.hpp"

#include <memory>
#include <vector>

using namespace pulsim;
using namespace pulsim::solver;
using Catch::Approx;

namespace {

Vector make_state(Index n, Real base) {
    Vector v(n);
    for (Index i = 0; i < n; ++i) {
        v[i] = base + static_cast<Real>(i);
    }
    return v;
}

}  // namespace

TEST_CASE("StateTrajectory: contiguous layout, one allocation",
          "[v2][layer5][trajectory]") {
    StateTrajectory t;
    t.set_state_size(3);
    t.reserve(100);
    REQUIRE(t.capacity() >= 100);
    REQUIRE(t.empty());
    REQUIRE(t.size() == 0);

    const Real* base = t.data();
    for (int k = 0; k < 100; ++k) {
        t.push_back(make_state(3, static_cast<Real>(10 * k)));
    }
    REQUIRE(t.size() == 100);
    // Pre-reserved: the buffer never moved across 100 appends.
    REQUIRE(t.data() == base);
    REQUIRE(t.rows() == 100);
    REQUIRE(t.cols() == 3);
    REQUIRE(t.bytes() == 100 * 3 * sizeof(Real));

    // Row-major: sample k occupies [k*3, k*3+3).
    for (int k = 0; k < 100; ++k) {
        for (Index i = 0; i < 3; ++i) {
            REQUIRE(t.data()[k * 3 + i] ==
                    Approx(static_cast<Real>(10 * k) +
                           static_cast<Real>(i)));
        }
    }
}

TEST_CASE("StateTrajectory: read API matches vector<Vector>",
          "[v2][layer5][trajectory]") {
    StateTrajectory t;
    t.push_back(make_state(2, 0));
    t.push_back(make_state(2, 100));
    t.push_back(make_state(2, 200));

    REQUIRE(t.size() == 3);
    REQUIRE_FALSE(t.empty());
    REQUIRE(t[1][0] == Approx(100.0));
    REQUIRE(t[1][1] == Approx(101.0));
    REQUIRE(t.front()[0] == Approx(0.0));
    REQUIRE(t.back()[0] == Approx(200.0));
    REQUIRE(t.at(2)[1] == Approx(201.0));
    REQUIRE_THROWS_AS(t.at(3), std::out_of_range);

    // Range-for over `const auto&` binds the Map temporary.
    Real sum = 0;
    for (const auto& s : t) {
        sum += s[0];
    }
    REQUIRE(sum == Approx(300.0));

    // Copy-out to an owning Vector still works (Eigen conversion).
    const Vector copied = t[2];
    REQUIRE(copied.size() == 2);
    REQUIRE(copied[0] == Approx(200.0));

    // Brace assignment (the vector<Vector> idiom fixtures use).
    Vector a = make_state(4, 7);
    t = {a, a};
    REQUIRE(t.size() == 2);
    REQUIRE(t.cols() == 4);
    REQUIRE(t[1][3] == Approx(10.0));
}

TEST_CASE("StateTrajectory: ragged pushes throw instead of corrupting",
          "[v2][layer5][trajectory]") {
    // vector<Vector> silently accepted differing sizes; a
    // contiguous trajectory has ONE row stride by construction, so
    // a mismatch is a loud error rather than a reinterpreted row.
    StateTrajectory t;
    t.push_back(make_state(3, 0));
    REQUIRE_THROWS_AS(t.push_back(make_state(4, 0)),
                       std::invalid_argument);
    REQUIRE(t.size() == 1);   // rejected push left it untouched

    StateTrajectory u;
    u.set_state_size(2);
    u.push_back(make_state(2, 0));
    REQUIRE_THROWS_AS(u.set_state_size(5), std::invalid_argument);
}

TEST_CASE("StateTrajectory: empty trajectory is well-defined",
          "[v2][layer5][trajectory]") {
    StateTrajectory t;
    REQUIRE(t.empty());
    REQUIRE(t.size() == 0);
    REQUIRE(t.rows() == 0);
    REQUIRE(t.cols() == 0);
    REQUIRE(t.bytes() == 0);
    // With no row width yet, capacity() reports the PENDING
    // reservation — nothing is allocated until the width arrives
    // (adversarial-review finding F8: the old comment claimed the
    // reservation was already committed).
    t.reserve(50);
    REQUIRE(t.capacity() == 50);
    REQUIRE(t.bytes() == 0);       // genuinely nothing allocated yet
    REQUIRE(t.empty());
    // Supplying the width is what materialises the buffer.
    t.set_state_size(4);
    REQUIRE(t.capacity() >= 50);
    const Real* base = t.data();
    for (int k = 0; k < 50; ++k) t.push_back(make_state(4, 0));
    REQUIRE(t.data() == base);     // no reallocation
}

TEST_CASE("StateTrajectory: eager reservation is byte-capped",
          "[v2][layer5][trajectory]") {
    // Adversarial-review finding contig-02: reserving
    // n_samples x n_state x 8 B up front can be many GB for a long
    // high-fidelity run, turning a cancellable run into an
    // immediate bad_alloc. Beyond the cap the buffer grows on
    // demand instead (still contiguous, still amortized O(1)).
    StateTrajectory t;
    t.set_state_size(64);
    const Size absurd =
        (StateTrajectory::kEagerReserveByteCap / (64 * sizeof(Real))) * 8;
    t.reserve(absurd);
    REQUIRE(t.bytes() == 0);           // nothing recorded yet
    REQUIRE(t.capacity() < absurd);    // capped, not committed
    REQUIRE(t.capacity() * 64 * sizeof(Real) <=
            StateTrajectory::kEagerReserveByteCap);
    // Still correct — recording past the cap simply grows.
    for (int k = 0; k < 10; ++k) t.push_back(make_state(64, k));
    REQUIRE(t.size() == 10);
    REQUIRE(t[9][0] == Approx(9.0));
}

// =============================================================================
// store_every — decimated recording on a strictly uniform grid
// =============================================================================

namespace {

/// V_dc → switch → R ∥ C. One switch, dynamic (dt > 0).
struct Chopper {
    topology::Graph g;
    pwl::DevicePool pool;
    std::unique_ptr<pwl::PwlStateSpaceCache> cache;

    Chopper() {
        g.add_node("vin");
        g.add_node("vout");
        g.add_branch(0, g.ground(), topology::BranchKind::Source);
        pool.add_voltage_source(0, {.V = 12.0});
        g.add_branch(0, 1, topology::BranchKind::Switch);
        pool.add_switch(1, /*g_on=*/1.0, /*g_off=*/1e-9);
        g.add_branch(1, g.ground(), topology::BranchKind::PassiveLinear);
        pool.add_resistor(2, {.G = 0.1});
        g.add_branch(1, g.ground(), topology::BranchKind::PassiveLinear);
        pool.add_capacitor(3, {.C = 1e-6});
        cache = std::make_unique<pwl::PwlStateSpaceCache>(g, pool);
    }
};

SwitchScheduleFn always_on() {
    return [](Real) {
        topology::SwitchStateMask m(1);
        m.set(0, true);
        return m;
    };
}

}  // namespace

TEST_CASE("store_every: sample count and expected_sample_count agree",
          "[v2][layer5][store_every]") {
    SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 1e-4;
    opts.dt      = 1e-6;   // 101 steps
    REQUIRE(opts.expected_step_count() == 101);
    REQUIRE(opts.expected_sample_count() == 101);

    opts.store_every = 10;
    // steps 0,10,...,100 → 11 samples
    REQUIRE(opts.expected_sample_count() == 11);
    opts.store_every = 7;
    // ceil(101/7) = 15
    REQUIRE(opts.expected_sample_count() == 15);
    opts.store_every = 0;
    REQUIRE_FALSE(opts.valid());   // would record nothing
}

TEST_CASE("store_every: recorded grid stays strictly uniform",
          "[v2][layer5][store_every]") {
    // The uniform-grid guarantee is load-bearing: FFT / harmonic /
    // ripple analysis downstream assumes a constant spacing. We
    // decimate on a pure stride (steps 0, m, 2m, …) rather than
    // force-appending the final step, which would leave a short
    // last interval and silently skew those analyses.
    Chopper f;
    SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 1e-4;
    opts.dt      = 1e-6;
    opts.store_every = 10;
    f.cache->build_lazy(opts.dt);

    auto res = run_transient(*f.cache, f.g, f.pool, opts, always_on());

    REQUIRE(res.num_steps() == opts.expected_sample_count());
    REQUIRE(res.states.size() == res.times.size());
    REQUIRE(res.event_iteration_count.size() == res.times.size());

    const Real dt_eff = opts.dt * static_cast<Real>(opts.store_every);
    for (Size j = 0; j < res.times.size(); ++j) {
        REQUIRE(res.times[j] ==
                Approx(opts.t_start + static_cast<Real>(j) * dt_eff)
                    .margin(1e-15));
    }
}

TEST_CASE("store_every: decimated samples equal the full-rate ones",
          "[v2][layer5][store_every]") {
    // Decimation must change WHAT IS STORED, never the integration:
    // sample j of a store_every=m run must be bit-identical to
    // sample j*m of the full-rate run.
    SimulationOptions full;
    full.t_start = 0.0;
    full.t_end   = 2e-4;
    full.dt      = 1e-6;

    SimulationOptions dec = full;
    dec.store_every = 25;

    Chopper f1;
    f1.cache->build_lazy(full.dt);
    auto res_full = run_transient(*f1.cache, f1.g, f1.pool, full,
                                    always_on());

    Chopper f2;
    f2.cache->build_lazy(dec.dt);
    auto res_dec = run_transient(*f2.cache, f2.g, f2.pool, dec,
                                   always_on());

    REQUIRE(res_dec.num_steps() == dec.expected_sample_count());
    REQUIRE(res_dec.num_steps() < res_full.num_steps());
    for (Size j = 0; j < res_dec.num_steps(); ++j) {
        const Size k = j * dec.store_every;
        REQUIRE(res_dec.times[j] == res_full.times[k]);
        for (Index i = 0; i < res_dec.states.cols(); ++i) {
            REQUIRE(res_dec.states[j][i] == res_full.states[k][i]);
        }
    }
    // Memory actually drops by the stride: the buffer is exactly
    // n_samples x n_state reals, ~store_every times smaller.
    REQUIRE(res_dec.states.bytes() ==
            static_cast<std::size_t>(res_dec.num_steps()) *
                static_cast<std::size_t>(res_dec.states.cols()) *
                sizeof(Real));
    REQUIRE(res_dec.states.bytes() * 10 < res_full.states.bytes());
}

TEST_CASE("store_every = 1 is byte-identical to v1.x recording",
          "[v2][layer5][store_every]") {
    Chopper f;
    SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 5e-5;
    opts.dt      = 1e-6;
    f.cache->build_lazy(opts.dt);
    REQUIRE(opts.store_every == 1);   // default

    auto res = run_transient(*f.cache, f.g, f.pool, opts, always_on());
    REQUIRE(res.num_steps() == opts.expected_step_count());
    for (Size k = 0; k < res.num_steps(); ++k) {
        REQUIRE(res.times[k] ==
                Approx(opts.t_start + static_cast<Real>(k) * opts.dt)
                    .margin(1e-15));
    }
}
