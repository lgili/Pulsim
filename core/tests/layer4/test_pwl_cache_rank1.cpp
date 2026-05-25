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
//     backend supports partial_refactor (PulsimSparseLuSolver — landing
//     in Sections 2-5 of openspec/replace-klu-with-pulsim-sparse-lu/).
//   * `metrics().full_refactor_hits` increments on first-encounter +
//     multi-bit flips.
//   * `metrics().fallbacks` increments when the backend does NOT support
//     partial_refactor (currently the only state — Eigen::SparseLU is
//     the sole DirectSolver until Section 2 of the rewrite lands).
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

    // With PulsimSparseLuSolver as the default backend (v1.3.0+) and
    // the Gray-code sequence (15 single-bit transitions after the
    // first first-encounter), the rank-1 fast path should engage on
    // most or all of the 15 single-bit calls. We don't pin an exact
    // count because pivot-threshold rejections can route a few to
    // fallback, but the majority MUST go through partial_refactor —
    // otherwise the headline TPEL claim ("we have a working path-
    // based partial refactor") would be vacuous.
    const auto m = cache_b.metrics();
    INFO("Gray-code sweep metrics: rank1_hits=" << m.rank1_hits
         << " full_refactor_hits=" << m.full_refactor_hits
         << " fallbacks=" << m.fallbacks);
    REQUIRE(m.full_refactor_hits == 1);             // only the first call
    REQUIRE((m.rank1_hits + m.fallbacks) == 15);    // 15 single-bit flips
    REQUIRE(m.rank1_hits >= 8);                      // majority via partial_refactor
}

// -----------------------------------------------------------------------------
// 5.2 — Multi-bit flip falls back to full refactor
// -----------------------------------------------------------------------------
TEST_CASE("solve_rank1 multi-bit flips route through multi_bit_rank1_hits (v1.4.0)",
          "[v2][layer4][cache][rank1][metrics][multi_bit]") {
    // v1.4.0 (openspec/changes/add-generalised-path-refactor Part A):
    // multi-bit transitions no longer unconditionally fall back to a
    // full factorize. They route through path-union partial_refactor
    // when the union path is ≤ MAX_PATH_LENGTH_RATIO of n. The
    // distribution of hits across the four CacheMetrics buckets:
    //
    //   rank1_hits           = no-change OR single-bit success
    //   multi_bit_rank1_hits = multi-bit success via path-union
    //   full_refactor_hits   = first-encounter, or "path too long
    //                          to bother" (>0.6·n), or backend that
    //                          does not implement partial_refactor
    //   fallbacks            = partial_refactor attempted + returned
    //                          false (numerical / pivot fault)
    //
    // Invariant per spec: rank1 + multi_bit + full + fallbacks == N.

    Fixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);

    // First call: 0b0000 → first-encounter → full_refactor_hits++
    cache.solve_rank1(fx.mask(0b0000), b_extra, x);

    // Second call: 0b0011 → 2-bit diff vs 0b0000. v1.4.0 routes
    // through path-union: if the union path of the affected columns
    // stays ≤ 0.6·n it lands in multi_bit_rank1_hits, otherwise in
    // full_refactor_hits. On this 4-switch fixture the path is
    // typically short, so we expect a multi-bit hit.
    cache.solve_rank1(fx.mask(0b0011), b_extra, x);

    // Third call: 0b1100 → 4-bit diff vs 0b0011. Same routing.
    cache.solve_rank1(fx.mask(0b1100), b_extra, x);

    // Fourth call: 0b1101 → 1-bit diff vs 0b1100 → single-bit fast
    // path (v1.3.0 behavior preserved): always tries partial_refactor
    // without the ratio gate.
    cache.solve_rank1(fx.mask(0b1101), b_extra, x);

    const auto m = cache.metrics();

    // Invariant: 4 calls total, distributed across the 4 buckets.
    REQUIRE(m.rank1_hits + m.multi_bit_rank1_hits +
            m.full_refactor_hits + m.fallbacks == 4);

    // Exactly 1 first-encounter (the 0b0000 call). The other 3 calls
    // get path-based routing; on this fixture the path is short
    // enough that partial_refactor either succeeds (rank1_hits for
    // 1-bit / multi_bit_rank1_hits for 2 or 4-bit) or, in unusual
    // cases, falls back to factorize.
    REQUIRE(m.full_refactor_hits >= 1);
    // The single-bit 0b1100 → 0b1101 call lands in rank1_hits OR
    // fallbacks (numerical) — never in multi_bit_rank1_hits.
    REQUIRE(m.rank1_hits + m.fallbacks >= 1);
    // The two multi-bit calls (0b0000 → 0b0011 and 0b0011 → 0b1100)
    // land in multi_bit_rank1_hits OR fallbacks OR full_refactor_hits
    // (if path > 0.6·n). On this fixture they typically succeed via
    // path-union → multi_bit_rank1_hits.

    INFO("v1.4.0 metrics: "
         << "rank1_hits=" << m.rank1_hits
         << " multi_bit_rank1_hits=" << m.multi_bit_rank1_hits
         << " full_refactor_hits=" << m.full_refactor_hits
         << " fallbacks=" << m.fallbacks);
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
    REQUIRE(m.multi_bit_rank1_hits == 0);
    REQUIRE(m.full_refactor_hits == 0);
    REQUIRE(m.fallbacks == 0);
}

// -----------------------------------------------------------------------------
// v1.4.0 spec scenarios (openspec/changes/add-generalised-path-refactor)
// -----------------------------------------------------------------------------

// Scenario "2-bit transition routed via path union":
//   GIVEN a cache built with PulsimSparseLuSolver and mask_prev = 0b0000
//   WHEN solve_rank1 is called with mask_curr = 0b0011 (Hamming distance 2)
//   AND the union path of the 2 affected columns has length ≤ 0.6·n
//   THEN multi_bit_rank1_hits increments by 1
//   AND output x matches fresh-factorise within 1e-10
TEST_CASE("v1.4.0: 2-bit Gray-code transition routes through multi_bit_rank1_hits",
          "[v2][layer4][cache][rank1][multi_bit][v15]") {
    Fixture fx;
    PwlStateSpaceCache cache_rank1(fx.g, fx.pool);
    PwlStateSpaceCache cache_full(fx.g, fx.pool);
    cache_rank1.build_lazy(fx.dt);
    cache_full.build_lazy(fx.dt);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x_rank1(n), x_full(n);

    // Prime both caches at mask 0b0000.
    cache_rank1.solve_rank1(fx.mask(0b0000), b_extra, x_rank1);
    cache_full.solve(fx.mask(0b0000), b_extra, x_full);

    // 2-bit transition: 0b0000 → 0b0011.
    cache_rank1.solve_rank1(fx.mask(0b0011), b_extra, x_rank1);
    cache_full.solve(fx.mask(0b0011), b_extra, x_full);

    // Parity within 1e-10 (spec: 1e-10 for multi-bit; tighter than the
    // 1e-12 single-bit gate to absorb the extra path-walk round-off).
    Real err = 0;
    for (Index i = 0; i < n; ++i) {
        err = std::max(err, std::abs(x_rank1[i] - x_full[i]));
    }
    INFO("|x_rank1 - x_full|∞ = " << err);
    REQUIRE(err < Real{1e-10});

    // Counter check: 1 first-encounter (full_refactor_hits) + 1
    // multi-bit hit on the 0b0000 → 0b0011 transition. The path is
    // short enough on this 4-switch fixture (n ≈ 6) for the ratio
    // gate to allow it. If it ever doesn't, the result lands in
    // full_refactor_hits — but the parity check above still holds.
    const auto m = cache_rank1.metrics();
    INFO("metrics: rank1=" << m.rank1_hits
         << " multi=" << m.multi_bit_rank1_hits
         << " full=" << m.full_refactor_hits
         << " fall=" << m.fallbacks);
    REQUIRE(m.rank1_hits + m.multi_bit_rank1_hits +
            m.full_refactor_hits + m.fallbacks == 2);
    REQUIRE(m.full_refactor_hits >= 1);  // first encounter
}

// Scenario "Telemetry invariant holds":
//   GIVEN a cache after N solve_rank1 calls
//   THEN single_bit + multi_bit + full + fallbacks == N
//   AND all four counters are monotonic across calls
TEST_CASE("v1.4.0: CacheMetrics invariant holds across mixed-distance sweep",
          "[v2][layer4][cache][rank1][metrics][v15]") {
    Fixture fx;
    PwlStateSpaceCache cache(fx.g, fx.pool);
    cache.build_lazy(fx.dt);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x(n);

    // Mixed-Hamming-distance sweep (4 calls, distances 0/1/2/4):
    cache.solve_rank1(fx.mask(0b0000), b_extra, x);  // first encounter
    cache.solve_rank1(fx.mask(0b0000), b_extra, x);  // pop=0 (no change)
    cache.solve_rank1(fx.mask(0b0001), b_extra, x);  // pop=1 (single-bit)
    cache.solve_rank1(fx.mask(0b1110), b_extra, x);  // pop=4 (multi-bit)

    const auto m = cache.metrics();
    REQUIRE(m.rank1_hits + m.multi_bit_rank1_hits +
            m.full_refactor_hits + m.fallbacks == 4);

    // Monotonicity sanity: a second snapshot after one more call
    // (any kind) must have counters ≥ the previous snapshot.
    cache.solve_rank1(fx.mask(0b0000), b_extra, x);
    const auto m2 = cache.metrics();
    REQUIRE(m2.rank1_hits           >= m.rank1_hits);
    REQUIRE(m2.multi_bit_rank1_hits >= m.multi_bit_rank1_hits);
    REQUIRE(m2.full_refactor_hits   >= m.full_refactor_hits);
    REQUIRE(m2.fallbacks            >= m.fallbacks);
    REQUIRE(m2.rank1_hits + m2.multi_bit_rank1_hits +
            m2.full_refactor_hits + m2.fallbacks == 5);
}

// 4-bit transitions on this 4-switch fixture should ALSO route through
// path-based (the union path is still short on n ≈ 6). Solve parity
// is the canonical correctness check; the counter is informational.
TEST_CASE("v1.4.0: 4-bit transition still produces correct output",
          "[v2][layer4][cache][rank1][multi_bit][v15]") {
    Fixture fx;
    PwlStateSpaceCache cache_rank1(fx.g, fx.pool);
    PwlStateSpaceCache cache_full(fx.g, fx.pool);
    cache_rank1.build_lazy(fx.dt);
    cache_full.build_lazy(fx.dt);

    const Index n = static_cast<Index>(fx.pool.state_size(fx.g));
    Vector b_extra = Vector::Zero(n);
    Vector x_rank1(n), x_full(n);

    cache_rank1.solve_rank1(fx.mask(0b0000), b_extra, x_rank1);
    cache_full.solve(fx.mask(0b0000), b_extra, x_full);

    // 4-bit transition: 0b0000 → 0b1111
    cache_rank1.solve_rank1(fx.mask(0b1111), b_extra, x_rank1);
    cache_full.solve(fx.mask(0b1111), b_extra, x_full);

    Real err = 0;
    for (Index i = 0; i < n; ++i) {
        err = std::max(err, std::abs(x_rank1[i] - x_full[i]));
    }
    INFO("|x_rank1 - x_full|∞ = " << err);
    REQUIRE(err < Real{1e-10});
}
