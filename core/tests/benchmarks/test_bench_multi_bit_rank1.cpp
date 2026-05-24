// =============================================================================
// Benchmark — multi-bit rank-1 partial-refactor (v1.4.0)
// =============================================================================
//
// `openspec/changes/add-generalised-path-refactor` Part A — §7.1 of
// the proposal. Captures the per-Hamming-distance speedup of the new
// path-union routing in `PwlStateSpaceCache::solve_rank1`.
//
// v1.3.0 routed multi-bit transitions to an unconditional full
// `factorize()` call. v1.4.0 attempts a path-union `partial_refactor`
// when `path_length / n ≤ MAX_PATH_LENGTH_RATIO` (default 0.6),
// falling back to full factorize only when the union path is too
// long to amortise. This bench captures the win.
//
// Three backends compared on the same N-switch fixture, driven through
// a sequence of multi-bit transitions of fixed Hamming distance:
//
//   (A) baseline `solve()` — per-mask cache, every mask is a fresh
//       `analyze + factorize` miss
//   (B) `solve_rank1` v1.3.0 emulation — Backend::Eigen sliding
//       solver. Eigen doesn't implement partial_refactor, so every
//       transition triggers a full numeric factorize. Isolates the
//       "amortised symbolic" win from the path-based win.
//   (C) `solve_rank1` v1.4.0 — Backend::Pulsim path-union
//       partial_refactor. The headline.
//
// Sweep: Hamming distance δ ∈ {1, 2, 3, 4} × N ∈ {8, 12, 16, 20, 24}.
//
// Output:
//   * Pretty-printed table to stdout
//   * Machine-readable CSV at
//     artigos/02_tpel_methods/benchmarks/results/multi_bit_microbench.csv
//
// Excluded from `ctest` (lives in the opt-in `pulsim_benchmarks`
// binary). Run:
//   cmake --build build --target pulsim_benchmarks -j
//   ./build/core/pulsim_benchmarks "[multi_bit][microbench]"

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using namespace pulsim::bench;

namespace {

// Reuses the N-switch fixture pattern from test_bench_pwl_rank1.cpp:
// each switch hooks node 0 to a node anchored to ground via 1 Ω.
struct NSwitchFixture {
    Graph        g;
    DevicePool   pool;
    Size         N;

    explicit NSwitchFixture(Size n_switches) : N(n_switches) {
        g.add_node("n0");
        for (Size k = 0; k < N; ++k) {
            g.add_node("nbr_" + std::to_string(k));
        }
        const Index v_id = g.add_branch(0, g.ground(), BranchKind::Source);
        pool.add_voltage_source(v_id, {Real{10.0}});
        for (Size k = 0; k < N; ++k) {
            const Index sw_id =
                g.add_branch(0, static_cast<Index>(k + 1),
                              BranchKind::Switch);
            pool.add_switch(sw_id, Real{1e3}, Real{1e-9});
            const Index r_id =
                g.add_branch(static_cast<Index>(k + 1), g.ground(),
                              BranchKind::PassiveLinear);
            pool.add_resistor(r_id, {Real{1.0}});
        }
    }
};

/// Build a sequence of `n_calls` masks, each differing from its
/// predecessor by exactly `delta_bits` random bit positions. Uses a
/// fixed seed (`seed`) so runs are reproducible.
std::vector<SwitchStateMask> make_random_distance_sequence(
    Size N, int delta_bits, Size n_calls, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<SwitchStateMask> seq;
    seq.reserve(n_calls);

    std::uint64_t current = 0;
    {
        SwitchStateMask m0(N);
        m0.set_bits(0);
        seq.push_back(m0);
    }

    // For each call, pick `delta_bits` distinct bit positions in
    // [0, N) and XOR them into `current`. Re-sampling rejection
    // guarantees no duplicate positions per step.
    std::vector<Size> bit_indices(N);
    for (Size i = 0; i < N; ++i) bit_indices[i] = i;

    for (Size c = 1; c < n_calls; ++c) {
        // Shuffle first delta_bits positions.
        for (int i = 0; i < delta_bits; ++i) {
            std::uniform_int_distribution<Size> dist(i, N - 1);
            std::swap(bit_indices[i], bit_indices[dist(rng)]);
        }
        std::uint64_t flip = 0;
        for (int i = 0; i < delta_bits; ++i) {
            flip |= (std::uint64_t{1} << bit_indices[i]);
        }
        current ^= flip;
        SwitchStateMask m(N);
        m.set_bits(current);
        seq.push_back(m);
    }
    return seq;
}

struct Row {
    Size   N;
    Size   n_state;
    int    delta_bits;
    Size   n_calls;
    // (A) baseline
    double wall_solve_ms;
    double per_call_solve_us;
    // (B) Eigen sliding solver — every flip = full factorize
    double wall_eigen_ms;
    double per_call_eigen_us;
    // (C) Pulsim path-union (v1.4.0)
    double wall_pulsim_ms;
    double per_call_pulsim_us;
    double speedup_pulsim_vs_solve;
    double speedup_pulsim_vs_eigen;
    std::uint64_t pulsim_single_bit_hits;
    std::uint64_t pulsim_multi_bit_hits;
    std::uint64_t pulsim_full_refactor_hits;
    std::uint64_t pulsim_fallbacks;
};

Row run_one(Size N, int delta_bits) {
    NSwitchFixture fx(N);
    const Size n_state = fx.pool.state_size(fx.g);

    // 1000 calls per (N, δ) cell. Enough to amortise the per-call
    // measurement noise while keeping the total runtime bounded.
    constexpr Size N_CALLS = 1000;
    const auto seq = make_random_distance_sequence(
        N, delta_bits, N_CALLS, /*seed=*/0xC0FFEE);

    constexpr Real dt = 1e-6;
    Vector b_extra = Vector::Zero(static_cast<Index>(n_state));

    auto run_backend = [&](sparse::Backend backend)
        -> std::tuple<double, CacheMetrics> {
        PwlStateSpaceCache cache(fx.g, fx.pool);
        cache.build_lazy(dt);
        cache.set_rank1_backend(backend);
        Vector x(static_cast<Index>(n_state));
        Stopwatch sw;
        for (const auto& m : seq) {
            cache.solve_rank1(m, b_extra, x);
        }
        return {sw.elapsed_ms(), cache.metrics()};
    };

    // (A) baseline solve()
    double wall_solve_ms = 0.0;
    {
        PwlStateSpaceCache cache(fx.g, fx.pool);
        cache.build_lazy(dt);
        Vector x(static_cast<Index>(n_state));
        Stopwatch sw;
        for (const auto& m : seq) {
            cache.solve(m, b_extra, x);
        }
        wall_solve_ms = sw.elapsed_ms();
    }

    // (B) v1.3.0 emulation — Eigen backend (no partial_refactor)
    auto [wall_eigen_ms, eigen_metrics] =
        run_backend(sparse::Backend::Eigen);
    (void)eigen_metrics;

    // (C) v1.4.0 — Pulsim path-union
    auto [wall_pulsim_ms, pulsim_metrics] =
        run_backend(sparse::Backend::Pulsim);

    return Row{
        .N                         = N,
        .n_state                   = n_state,
        .delta_bits                = delta_bits,
        .n_calls                   = N_CALLS,
        .wall_solve_ms             = wall_solve_ms,
        .per_call_solve_us         =
            (wall_solve_ms * 1000.0) / static_cast<double>(N_CALLS),
        .wall_eigen_ms             = wall_eigen_ms,
        .per_call_eigen_us         =
            (wall_eigen_ms * 1000.0) / static_cast<double>(N_CALLS),
        .wall_pulsim_ms            = wall_pulsim_ms,
        .per_call_pulsim_us        =
            (wall_pulsim_ms * 1000.0) / static_cast<double>(N_CALLS),
        .speedup_pulsim_vs_solve   = wall_solve_ms / wall_pulsim_ms,
        .speedup_pulsim_vs_eigen   = wall_eigen_ms / wall_pulsim_ms,
        .pulsim_single_bit_hits    = pulsim_metrics.rank1_hits,
        .pulsim_multi_bit_hits     = pulsim_metrics.multi_bit_rank1_hits,
        .pulsim_full_refactor_hits = pulsim_metrics.full_refactor_hits,
        .pulsim_fallbacks          = pulsim_metrics.fallbacks,
    };
}

void print_header() {
    std::printf("\n%-3s %-7s %-3s %-7s %-10s %-10s %-10s %-8s %-8s "
                 "%-7s %-7s %-7s %-7s\n",
                 "N", "n_state", "δ", "n_calls",
                 "us/solve", "us/eigen", "us/pulsim",
                 "P/solve", "P/eigen",
                 "1bit", "Nbit", "full", "fall");
    std::printf("--- ------- --- ------- ---------- ---------- "
                 "---------- -------- -------- ------- ------- "
                 "------- -------\n");
}

void print_row(const Row& r) {
    std::printf("%-3zu %-7zu %-3d %-7zu %-7.3f us %-7.3f us %-7.3f us "
                 "%-6.2fx %-6.2fx %-7llu %-7llu %-7llu %-7llu\n",
                 r.N, r.n_state, r.delta_bits, r.n_calls,
                 r.per_call_solve_us, r.per_call_eigen_us,
                 r.per_call_pulsim_us,
                 r.speedup_pulsim_vs_solve, r.speedup_pulsim_vs_eigen,
                 static_cast<unsigned long long>(r.pulsim_single_bit_hits),
                 static_cast<unsigned long long>(r.pulsim_multi_bit_hits),
                 static_cast<unsigned long long>(r.pulsim_full_refactor_hits),
                 static_cast<unsigned long long>(r.pulsim_fallbacks));
}

std::filesystem::path resolve_csv_path() {
    const char* dir = std::getenv("PULSIM_BENCH_RESULTS_DIR");
    std::filesystem::path base =
        (dir != nullptr) ? std::filesystem::path(dir)
                          : std::filesystem::path("bench-results");
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base / "multi_bit_microbench.csv";
}

void write_csv(const std::vector<Row>& rows) {
    const auto p = resolve_csv_path();
    std::ofstream ofs(p);
    if (!ofs) {
        std::printf("  (warning) could not open %s for writing\n",
                     p.string().c_str());
        return;
    }
    ofs << "N,n_state,delta_bits,n_calls,"
           "wall_solve_ms,us_per_solve,"
           "wall_eigen_ms,us_per_eigen,"
           "wall_pulsim_ms,us_per_pulsim,"
           "speedup_pulsim_vs_solve,speedup_pulsim_vs_eigen,"
           "single_bit_hits,multi_bit_hits,full_refactor_hits,fallbacks\n";
    for (const auto& r : rows) {
        ofs << r.N << ',' << r.n_state << ',' << r.delta_bits << ','
            << r.n_calls << ','
            << r.wall_solve_ms  << ',' << r.per_call_solve_us  << ','
            << r.wall_eigen_ms  << ',' << r.per_call_eigen_us  << ','
            << r.wall_pulsim_ms << ',' << r.per_call_pulsim_us << ','
            << r.speedup_pulsim_vs_solve << ','
            << r.speedup_pulsim_vs_eigen << ','
            << r.pulsim_single_bit_hits    << ','
            << r.pulsim_multi_bit_hits     << ','
            << r.pulsim_full_refactor_hits << ','
            << r.pulsim_fallbacks          << '\n';
    }
    std::printf("  wrote %s\n", p.string().c_str());
}

}  // namespace

TEST_CASE("Bench: multi-bit rank-1 microbench — v1.4.0 path-union routing",
          "[bench][rank1][multi_bit][microbench]") {
    std::printf("\n=== multi-bit rank-1 microbench (v1.4.0) ===\n");
    std::printf("Hamming-distance sweep: δ ∈ {1, 2, 3, 4} × N ∈ {8, 12, 16, 20, 24}\n");
    std::printf("Backends:\n");
    std::printf("  (A) `solve` baseline — per-mask cache, fresh analyze+factorize per miss\n");
    std::printf("  (B) `solve_rank1` Eigen — sliding solver, full factorize per flip\n");
    std::printf("  (C) `solve_rank1` Pulsim — v1.4.0 path-union partial_refactor\n");
    std::printf("\nColumn legend:\n");
    std::printf("  P/solve = Pulsim path-union vs baseline\n");
    std::printf("  P/eigen = Pulsim path-union vs sliding-solver baseline\n");
    std::printf("  1bit / Nbit / full / fall = Pulsim metric buckets\n");
    print_header();

    std::vector<Row> rows;
    for (Size N : {Size{8}, Size{12}, Size{16}, Size{20}, Size{24}}) {
        for (int delta : {1, 2, 3, 4}) {
            const Row r = run_one(N, delta);
            print_row(r);
            rows.push_back(r);

            // Sanity gates
            REQUIRE(r.n_calls > 0);
            REQUIRE(r.wall_solve_ms  > 0.0);
            REQUIRE(r.wall_eigen_ms  > 0.0);
            REQUIRE(r.wall_pulsim_ms > 0.0);
            // Telemetry invariant
            const std::uint64_t total =
                r.pulsim_single_bit_hits + r.pulsim_multi_bit_hits +
                r.pulsim_full_refactor_hits + r.pulsim_fallbacks;
            REQUIRE(total == r.n_calls);
        }
    }

    write_csv(rows);
}
