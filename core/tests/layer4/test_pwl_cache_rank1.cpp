// =============================================================================
// Layer 4 V8 — PwlStateSpaceCache::solve_rank1 (rank-1 fast-path) integration
// =============================================================================
//
// `openspec/changes/add-pwl-rank1-update` Section 5 tasks.
//
// Locks in the rank-1 fast-path contract:
//
//   * Output parity with `solve(mask, ...)` across a sequence of single-bit
//     Gray-code flips (1e-12 tolerance).
//   * `metrics().rank1_hits` increments on single-bit-diff calls when the
//     backend supports partial_refactor (PULSIM_HAVE_KLU build).
//   * `metrics().full_refactor_hits` increments on first-encounter +
//     multi-bit flips.
//   * `metrics().fallbacks` increments when the backend does NOT support
//     partial_refactor (Eigen-only build path; verified by forcing
//     Backend::Eigen via direct factory in 2.7).
//   * Cache state is unchanged (segments_ map untouched) — solve_rank1 is
//     orthogonal to the per-mask cache.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <vector>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

/// 4-switch synthetic test circuit:
///
///   gnd --[Vdc=10]-- n0 --[S0]-- n1 --[R=1]-- gnd
///                      \--[S1]-- n2 --[R=1]-- gnd
///                      \--[S2]-- n3 --[R=1]-- gnd
///                      \--[S3]-- n4 --[R=1]-- gnd
///
/// 4 switches → 2^4 = 16 possible masks. Each switch hooks a separate
/// resistor branch from n0 to a node anchored via R to ground. Toggling
/// switch i flips conductance on/off between n0 and n_{i+1}; the
/// resulting MNA system has a clean rank-1 structure for the partial-
/// refactor path to exercise.
struct Fixture {
    Graph        g;
    DevicePool   pool;
    Real         dt = 1e-6;

    Fixture() {
        g.add_node("n0");
        g.add_node("n1");
        g.add_node("n2");
        g.add_node("n3");
        g.add_node("n4");

        // Vdc between n0 (anode) and gnd (cathode), 10 V.
        const Index v_id = g.add_branch(0, g.ground(), BranchKind::Source);
        pool.add_voltage_source(v_id, {Real{10.0}});

        // Switch i between n0 and n_{i+1}, plus a 1Ω anchor from n_{i+1}
        // to ground. Anchor keeps the system non-singular when the
        // switch is OPEN (otherwise n_{i+1} floats).
        for (Index k = 0; k < 4; ++k) {
            const Index sw_id =
                g.add_branch(0, k + 1, BranchKind::Switch);
            pool.add_switch(sw_id, /*g_on=*/Real{1e3}, /*g_off=*/Real{1e-9});

            const Index r_id =
                g.add_branch(k + 1, g.ground(), BranchKind::PassiveLinear);
            pool.add_resistor(r_id, {Real{1.0}});
        }
    }

    SwitchStateMask mask(std::uint64_t bits) const {
        SwitchStateMask m(4);
        m.set_bits(bits);
        return m;
    }
};

}  // namespace

// -----------------------------------------------------------------------------
// 5.1 — Single-bit flip parity with full refactor
// -----------------------------------------------------------------------------
TEST_CASE("solve_rank1 produces same output as solve across a Gray-code sweep",
          "[v2][layer4][cache][rank1]") {
    Fixture fx;
    PwlStateSpaceCache cache_a(fx.g, fx.pool);
    PwlStateSpaceCache cache_b(fx.g, fx.pool);
    cache_a.build_lazy(fx.dt);
    cache_b.build_lazy(fx.dt);

    // Gray-code sequence for 4 bits: 0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15,
    // 14, 10, 11, 9, 8 — each consecutive pair differs by exactly 1 bit.
    const std::vector<std::uint64_t> gray_sequence = {
        0b0000, 0b0001, 0b0011, 0b0010, 0b0110, 0b0111, 0b0101, 0b0100,
        0b1100, 0b1101, 0b1111, 0b1110, 0b1010, 0b1011, 0b1001, 0b1000,
    };

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);

    for (std::uint64_t bits : gray_sequence) {
        const SwitchStateMask m = fx.mask(bits);

        Vector x_solve(n);
        cache_a.solve(m, b_extra, x_solve);

        Vector x_rank1(n);
        cache_b.solve_rank1(m, b_extra, x_rank1);

        for (Index i = 0; i < n; ++i) {
            REQUIRE(x_rank1[i] == Approx(x_solve[i]).margin(1e-12));
        }
    }
}

// -----------------------------------------------------------------------------
// 5.2 — Multi-bit flip falls back to full refactor
// -----------------------------------------------------------------------------
TEST_CASE("solve_rank1 multi-bit flips increment full_refactor_hits",
          "[v2][layer4][cache][rank1][metrics]") {
    Fixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);

    // First call: 0b0000 → full_refactor_hits++  (first encounter)
    cache.solve_rank1(fx.mask(0b0000), b_extra, x);

    // Second call: 0b0011 → 2-bit diff → full_refactor_hits++
    cache.solve_rank1(fx.mask(0b0011), b_extra, x);

    // Third call: 0b1100 → 4-bit diff vs 0b0011 → full_refactor_hits++
    cache.solve_rank1(fx.mask(0b1100), b_extra, x);

    // Fourth call: 0b1101 → 1-bit diff vs 0b1100 → either rank1_hits or
    // fallbacks (depending on backend). The metrics test 5.1 above
    // already verified parity; here we just verify the counter
    // partitioning.
    cache.solve_rank1(fx.mask(0b1101), b_extra, x);

    const auto m = cache.metrics();
    REQUIRE(m.full_refactor_hits == 3);
    // The 4th call is EITHER rank1_hits++ (KLU build) OR fallbacks++
    // (Eigen-only build). Either way, exactly one of them is 1.
    REQUIRE((m.rank1_hits + m.fallbacks) == 1);
    REQUIRE(m.rank1_hits + m.full_refactor_hits + m.fallbacks == 4);
}

// -----------------------------------------------------------------------------
// 5.3 — solve_rank1 does NOT touch the per-mask cache (segments_)
// -----------------------------------------------------------------------------
TEST_CASE("solve_rank1 leaves the per-mask segments_ cache empty",
          "[v2][layer4][cache][rank1][orthogonality]") {
    Fixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);   // lazy mode → segments_ starts empty

    REQUIRE(cache.num_built_segments() == 0);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);

    // 10 solve_rank1 calls — the sliding solver builds up but
    // segments_ MUST stay empty (the two paths are orthogonal).
    for (std::uint64_t bits : {0u, 1u, 3u, 2u, 6u, 7u, 5u, 4u, 12u, 13u}) {
        cache.solve_rank1(fx.mask(bits), b_extra, x);
    }

    REQUIRE(cache.num_built_segments() == 0);
}

// -----------------------------------------------------------------------------
// 5.4 — Re-solving the same mask just refreshes b_constant, counts as
//       rank1_hit (no refactor needed)
// -----------------------------------------------------------------------------
TEST_CASE("solve_rank1 same-mask repeats increment rank1_hits without refactor",
          "[v2][layer4][cache][rank1][metrics]") {
    Fixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);

    cache.solve_rank1(fx.mask(0b0001), b_extra, x);  // first encounter
    cache.solve_rank1(fx.mask(0b0001), b_extra, x);  // same mask, no refactor
    cache.solve_rank1(fx.mask(0b0001), b_extra, x);  // same mask, no refactor

    const auto m = cache.metrics();
    REQUIRE(m.full_refactor_hits == 1);   // only the first call
    REQUIRE(m.rank1_hits == 2);            // two repeats
    REQUIRE(m.fallbacks == 0);
}

// -----------------------------------------------------------------------------
// 5.5 — Initial metrics are all zero
// -----------------------------------------------------------------------------
TEST_CASE("CacheMetrics start at zero on a fresh cache",
          "[v2][layer4][cache][rank1][metrics]") {
    Fixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);

    const auto m = cache.metrics();
    REQUIRE(m.rank1_hits == 0);
    REQUIRE(m.full_refactor_hits == 0);
    REQUIRE(m.fallbacks == 0);
}
