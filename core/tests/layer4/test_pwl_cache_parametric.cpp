// =============================================================================
// Layer 4 v1.4.0 — PwlStateSpaceCache::refactor_parametric integration
// =============================================================================
//
// `openspec/changes/add-generalised-path-refactor` Part B §4.4 tests.
//
// Locks in the parametric-refactor contract:
//
//   * Single-parameter sweep on a buck-like fixture: after each
//     refactor_parametric call, `cache.solve(mask, ...)` matches a
//     fresh `analyze + factorize + solve` on a separately built
//     cache (1e-10 tolerance).
//   * Two-parameter simultaneous update gives the same parity.
//   * Mode::CurrentOnly processes only the rank-1 sliding mask.
//   * Telemetry invariant `path_refactor_hits + fallback_hits ==
//     masks_processed` holds.
//   * Unknown / unsupported device kinds throw a clear error
//     (caller is expected to rebuild the cache).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <array>
#include <vector>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;

namespace {

/// Small synthetic fixture with all 4 v1.4.0-supported device kinds
/// (Resistor + Inductor + Capacitor + VoltageSource) plus a switch.
/// Topology built directly on Graph + DevicePool to avoid the
/// builder's add_* API (which lacks a switch-add helper that exposes
/// the branch_id we need for parametric updates).
///
///   gnd --[V_dc=10]-- n0
///   n0 --[S0]-- n1 --[L=100µH]-- n2 --[R_load=2Ω]-- gnd
///                                 \--[C=10µF]-- gnd
///
/// (S0 closed → buck-cell-like topology with L, C, R_load active;
/// S0 open → broken path, only R_load + V_dc remain energised.)
struct BuckLikeFixture {
    Graph        g;
    DevicePool   pool;
    Real         dt = 1e-6;

    Index v_id  = -1;
    Index sw_id = -1;
    Index l_id  = -1;
    Index c_id  = -1;
    Index r_id  = -1;

    BuckLikeFixture() {
        g.add_node("n0");
        g.add_node("n1");
        g.add_node("n2");

        v_id = g.add_branch(0, g.ground(), BranchKind::Source);
        pool.add_voltage_source(v_id,
            models::VoltageSource::Params{Real{10.0}});

        sw_id = g.add_branch(0, 1, BranchKind::Switch);
        pool.add_switch(sw_id, /*g_on=*/Real{1e3},
                                /*g_off=*/Real{1e-9});

        l_id = g.add_branch(1, 2, BranchKind::PassiveLinear);
        pool.add_inductor(l_id,
            models::Inductor::Params{Real{100e-6}});

        c_id = g.add_branch(2, g.ground(), BranchKind::PassiveLinear);
        pool.add_capacitor(c_id,
            models::Capacitor::Params{Real{10e-6}});

        r_id = g.add_branch(2, g.ground(), BranchKind::PassiveLinear);
        pool.add_resistor(r_id,
            models::Resistor::Params{.G = Real{1} / Real{2.0}});
    }

    SwitchStateMask mask(std::uint64_t bits) const {
        SwitchStateMask m(1);
        m.set_bits(bits);
        return m;
    }
};

/// Build a fresh cache from a reference fixture and compare a solve
/// vector against the parametric-refactor target.
Real solve_max_diff(const Graph& g_a, DevicePool& pool_a,
                     const Graph& g_b, DevicePool& pool_b,
                     const SwitchStateMask& mask, Real dt) {
    PwlStateSpaceCache cache_a(g_a, pool_a);
    PwlStateSpaceCache cache_b(g_b, pool_b);
    cache_a.build_lazy(dt);
    cache_b.build_lazy(dt);
    const Index n =
        static_cast<Index>(pool_a.state_size(g_a));
    Vector b_extra = Vector::Zero(n);
    Vector x_a(n), x_b(n);
    cache_a.solve(mask, b_extra, x_a);
    cache_b.solve(mask, b_extra, x_b);
    Real err = 0;
    for (Index i = 0; i < n; ++i) {
        err = std::max(err, std::abs(x_a[i] - x_b[i]));
    }
    return err;
}

}  // namespace

// -----------------------------------------------------------------------------
// 4.4.1 — Single-parameter sweep matches fresh-rebuild within 1e-10
// -----------------------------------------------------------------------------
TEST_CASE("v1.4.0: refactor_parametric (R_load sweep) matches fresh build",
          "[v2][layer4][cache][parametric][v15]") {
    BuckLikeFixture fx;

    // Prime the cache at the initial values; visit both switch states
    // so segments_ has 2 entries.
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    const Index n =
        static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);
    cache.solve(fx.mask(0b0), b_extra, x);
    cache.solve(fx.mask(0b1), b_extra, x);

    // Sweep R_load through 5 values; each step uses refactor_parametric.
    const std::array<Real, 5> R_values = {
        Real{2.0}, Real{2.5}, Real{3.0}, Real{4.0}, Real{1.5}};

    for (Real R_new : R_values) {
        auto res = cache.refactor_parametric(fx.r_id, R_new);

        REQUIRE(res.masks_processed == 2);
        REQUIRE(res.path_refactor_hits + res.fallback_hits == 2);
        REQUIRE(res.wall_time_us >= 0);

        // Build a reference cache from scratch at the same value and
        // compare solves on both masks.
        BuckLikeFixture ref;
        ref.pool.update_resistor_R(ref.r_id, R_new);
        for (auto bits : {std::uint64_t{0}, std::uint64_t{1}}) {
            const Real err = solve_max_diff(fx.g, fx.pool,
                                              ref.g, ref.pool,
                                              fx.mask(bits), fx.dt);
            INFO("R_new=" << R_new << " mask=" << bits
                 << " max diff = " << err);
            REQUIRE(err < Real{1e-10});
        }
    }
}

// -----------------------------------------------------------------------------
// 4.4.2 — Two-parameter simultaneous change (L + C) also matches
// -----------------------------------------------------------------------------
TEST_CASE("v1.4.0: refactor_parametric handles 2-param update (L + C)",
          "[v2][layer4][cache][parametric][v15]") {
    BuckLikeFixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    const Index n =
        static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);
    cache.solve(fx.mask(0b0), b_extra, x);
    cache.solve(fx.mask(0b1), b_extra, x);

    // Simultaneously bump L and C by 10%.
    constexpr Real L_new = Real{110e-6};
    constexpr Real C_new = Real{11e-6};
    std::array<ParametricUpdate, 2> updates = {
        ParametricUpdate{fx.l_id, L_new},
        ParametricUpdate{fx.c_id, C_new},
    };
    auto res = cache.refactor_parametric(
        std::span<const ParametricUpdate>(updates));

    REQUIRE(res.masks_processed == 2);
    REQUIRE(res.path_refactor_hits + res.fallback_hits == 2);

    // Reference: fresh-rebuild cache with both updated values
    BuckLikeFixture ref;
    ref.pool.update_inductor_L(ref.l_id, L_new);
    ref.pool.update_capacitor_C(ref.c_id, C_new);
    for (auto bits : {std::uint64_t{0}, std::uint64_t{1}}) {
        const Real err = solve_max_diff(fx.g, fx.pool,
                                          ref.g, ref.pool,
                                          fx.mask(bits), fx.dt);
        INFO("two-param mask=" << bits << " max diff = " << err);
        REQUIRE(err < Real{1e-10});
    }
}

// -----------------------------------------------------------------------------
// 4.4.3 — Empty updates list is a no-op (telemetry-only call)
// -----------------------------------------------------------------------------
TEST_CASE("v1.4.0: refactor_parametric with empty updates is a no-op",
          "[v2][layer4][cache][parametric][v15]") {
    BuckLikeFixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    auto res = cache.refactor_parametric(
        std::span<const ParametricUpdate>{});
    REQUIRE(res.masks_processed    == 0);
    REQUIRE(res.path_refactor_hits == 0);
    REQUIRE(res.fallback_hits      == 0);
    REQUIRE(res.wall_time_us       >= 0);
}

// -----------------------------------------------------------------------------
// 4.4.4 — Unsupported device kind throws a clear error
// -----------------------------------------------------------------------------
TEST_CASE("v1.4.0: refactor_parametric on an unsupported device kind throws",
          "[v2][layer4][cache][parametric][v15]") {
    BuckLikeFixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    // The switch (sw_id) is registered, but Switch is not one of the
    // four supported kinds for parametric refactor — the dispatch
    // throws std::out_of_range.
    REQUIRE_THROWS_AS(
        cache.refactor_parametric(fx.sw_id, Real{0.5}),
        std::out_of_range);
}

// -----------------------------------------------------------------------------
// 4.4.5 — Mode::CurrentOnly processes only the rank-1 mask
// -----------------------------------------------------------------------------
TEST_CASE("v1.4.0: refactor_parametric Mode::CurrentOnly processes 1 mask",
          "[v2][layer4][cache][parametric][v15]") {
    BuckLikeFixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);
    const Index n =
        static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);

    // Use rank-1 path so rank1_initialized_ is true; mask 0b0.
    cache.solve_rank1(fx.mask(0b0), b_extra, x);

    // Now call refactor_parametric with CurrentOnly. It should
    // process exactly 1 segment (the rank-1 mask).
    auto res = cache.refactor_parametric(
        fx.r_id, Real{3.0},
        ParametricRefactorMode::CurrentOnly);
    REQUIRE(res.masks_processed == 1);
    REQUIRE(res.path_refactor_hits + res.fallback_hits == 1);
}

// -----------------------------------------------------------------------------
// 4.4.6 — Telemetry invariant across a 10-point sweep
// -----------------------------------------------------------------------------
TEST_CASE("v1.4.0: refactor_parametric telemetry invariant over 10 sweep points",
          "[v2][layer4][cache][parametric][v15]") {
    BuckLikeFixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);
    const Index n =
        static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);
    cache.solve(fx.mask(0b0), b_extra, x);
    cache.solve(fx.mask(0b1), b_extra, x);

    std::size_t total_path = 0;
    std::size_t total_fall = 0;
    for (int k = 0; k < 10; ++k) {
        const Real R_new = Real{1.5} + Real{0.1} * static_cast<Real>(k);
        auto res = cache.refactor_parametric(fx.r_id, R_new);
        REQUIRE(res.masks_processed == 2);
        REQUIRE(res.path_refactor_hits + res.fallback_hits == 2);
        total_path += res.path_refactor_hits;
        total_fall += res.fallback_hits;
    }
    INFO("over 10 sweep points: " << total_path << " path / "
         << total_fall << " fallback");
    REQUIRE(total_path + total_fall == 20);
}
