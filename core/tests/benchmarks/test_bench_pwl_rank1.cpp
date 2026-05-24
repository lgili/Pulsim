// =============================================================================
// Benchmark — PWL cache rank-1 vs full-refactor (Layer 4 V8)
// =============================================================================
//
// `openspec/changes/add-pwl-rank1-update` Section 6 — TPEL §VI data.
//
// Sweeps the number of switches N over {4, 8, 16, 24} and, for each N,
// drives a long Gray-code mask sequence through both paths:
//
//   * `cache.solve(mask, ...)`        — per-mask cache path; on a fresh
//                                       lazy cache every distinct mask
//                                       triggers `make_segment` →
//                                       `analyze + factorize`. The
//                                       baseline.
//   * `cache.solve_rank1(mask, ...)`  — sliding-solver fast path;
//                                       `analyze` once, then `factorize`
//                                       on multi-bit flips OR
//                                       `partial_refactor` (klu_refactor
//                                       MVP) on single-bit Gray-code
//                                       flips.
//
// Output:
//   * Pretty-printed table to stdout
//   * Machine-readable CSV at
//     artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv
//     (path overridable via PULSIM_BENCH_RESULTS_DIR env var; falls
//     back to the build dir on absence)
//
// Excluded from `ctest` (lives in the `pulsim_benchmarks` opt-in target).
// Run:
//   cmake --build build --target pulsim_benchmarks
//   ./build/core/pulsim_benchmarks "[rank1][microbench]"
//
// The headline result this benchmark feeds is "speedup ratio vs N" — the
// curve that goes into TPEL §VI Fig. 1 (the regime-transition figure).

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using namespace pulsim::bench;

namespace {

/// Synthetic N-switch fixture matching test_pwl_cache_rank1.cpp's pattern,
/// scaled to arbitrary N. Each switch hooks node 0 to its own "branch
/// node" anchored to ground via 1 Ω. State size grows linearly in N
/// (≈ 2·N + a few constants for the source / ground anchors). Sparsity
/// pattern is regular and bandwidth-1, which keeps the comparison clean
/// (no pathological fill-in differences between backends).
struct NSwitchFixture {
    Graph        g;
    DevicePool   pool;
    Size         N;

    explicit NSwitchFixture(Size n_switches) : N(n_switches) {
        g.add_node("n0");
        for (Size k = 0; k < N; ++k) {
            g.add_node("nbr_" + std::to_string(k));
        }

        // Vdc between n0 and gnd.
        const Index v_id = g.add_branch(0, g.ground(), BranchKind::Source);
        pool.add_voltage_source(v_id, {Real{10.0}});

        // For each switch: branch from n0 to a node anchored to gnd.
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

/// Build the standard Gray-code sequence over `2^N` masks for one full
/// pass. Pulsim's enumerator returns the same sequence; we duplicate
/// the logic here to avoid pulling the enumerator's iterator into the
/// timing loop (one fewer abstraction in the way of the measurement).
std::vector<SwitchStateMask> make_gray_sequence(Size N, Size repeat) {
    const std::uint64_t total =
        (N >= 64) ? (1ULL << 63) : (1ULL << N);
    std::vector<SwitchStateMask> seq;
    seq.reserve(static_cast<std::size_t>(total * repeat));
    for (Size r = 0; r < repeat; ++r) {
        for (std::uint64_t i = 0; i < total; ++i) {
            SwitchStateMask m(N);
            m.set_bits(i ^ (i >> 1));  // binary → Gray
            seq.push_back(m);
        }
    }
    return seq;
}

/// One row of the rank-1 microbench output — 3-backend comparison.
struct Row {
    Size   N;
    Size   n_state;
    Size   n_calls;
    // (A) Baseline: solve() with per-mask cache (lazy — each mask is a miss)
    double wall_solve_ms;
    double per_call_solve_us;
    // (B) solve_rank1 with Eigen::SparseLU forced — Eigen doesn't support
    //     partial_refactor, so every single-bit flip falls back to a full
    //     re-factorize (NOT a fresh analyze+factorize — the sliding solver
    //     amortises the symbolic phase). Isolates the "amortised analyze"
    //     win from the path-based win below.
    double wall_eigen_ms;
    double per_call_eigen_us;
    double speedup_eigen_vs_solve;     // wall_solve / wall_eigen
    std::uint64_t eigen_rank1_hits;
    std::uint64_t eigen_fallbacks;
    // (C) solve_rank1 with PulsimSparseLuSolver — the V8.1 path-based
    //     partial_refactor path. The headline win for the TPEL paper.
    double wall_pulsim_ms;
    double per_call_pulsim_us;
    double speedup_pulsim_vs_solve;    // wall_solve / wall_pulsim
    double speedup_pulsim_vs_eigen;    // wall_eigen / wall_pulsim (isolates path-based)
    std::uint64_t pulsim_rank1_hits;
    std::uint64_t pulsim_fallbacks;
};

Row run_one(Size N, Size repeat) {
    NSwitchFixture fx(N);
    const Size n_state = fx.pool.state_size(fx.g);

    // For large N (≥ 20) the full 2^N Gray-code pass would be too long;
    // truncate to a fixed-size sequence so wall-time stays bounded.
    constexpr Size MAX_CALLS = 2000;
    auto seq = make_gray_sequence(N, repeat);
    if (seq.size() > MAX_CALLS) {
        seq.resize(MAX_CALLS);
    }
    const Size n_calls = seq.size();

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

    // (A) Baseline: per-mask cache via `solve`
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

    // (B) solve_rank1 with Backend::Eigen — Eigen doesn't support
    //     partial_refactor, so every single-bit flip incurs a full
    //     factorize (but reuses the symbolic analyze across calls).
    auto [wall_eigen_ms, eigen_metrics] = run_backend(sparse::Backend::Eigen);

    // (C) solve_rank1 with Backend::Pulsim — the path-based partial_refactor.
    auto [wall_pulsim_ms, pulsim_metrics] = run_backend(sparse::Backend::Pulsim);

    return Row{
        .N                       = N,
        .n_state                 = n_state,
        .n_calls                 = n_calls,
        .wall_solve_ms           = wall_solve_ms,
        .per_call_solve_us       = (wall_solve_ms * 1000.0) /
                                   static_cast<double>(n_calls),
        .wall_eigen_ms           = wall_eigen_ms,
        .per_call_eigen_us       = (wall_eigen_ms * 1000.0) /
                                   static_cast<double>(n_calls),
        .speedup_eigen_vs_solve  = wall_solve_ms / wall_eigen_ms,
        .eigen_rank1_hits        = eigen_metrics.rank1_hits,
        .eigen_fallbacks         = eigen_metrics.fallbacks,
        .wall_pulsim_ms          = wall_pulsim_ms,
        .per_call_pulsim_us      = (wall_pulsim_ms * 1000.0) /
                                   static_cast<double>(n_calls),
        .speedup_pulsim_vs_solve = wall_solve_ms / wall_pulsim_ms,
        .speedup_pulsim_vs_eigen = wall_eigen_ms / wall_pulsim_ms,
        .pulsim_rank1_hits       = pulsim_metrics.rank1_hits,
        .pulsim_fallbacks        = pulsim_metrics.fallbacks,
    };
}

void print_header() {
    std::printf("\n%-3s %-7s %-7s %-10s %-10s %-10s "
                 "%-8s %-8s %-12s %-12s\n",
                 "N", "n_state", "n_calls",
                 "us/solve", "us/eigen", "us/pulsim",
                 "E/solve", "P/solve",
                 "rank1(P)", "fallbacks(P)");
    std::printf("--- ------- ------- ---------- ---------- ---------- "
                 "-------- -------- ------------ ------------\n");
}

void print_row(const Row& r) {
    std::printf("%-3zu %-7zu %-7zu %-7.3f us %-7.3f us %-7.3f us "
                 "%-6.2fx %-6.2fx %-12llu %-12llu\n",
                 r.N, r.n_state, r.n_calls,
                 r.per_call_solve_us, r.per_call_eigen_us, r.per_call_pulsim_us,
                 r.speedup_eigen_vs_solve, r.speedup_pulsim_vs_solve,
                 static_cast<unsigned long long>(r.pulsim_rank1_hits),
                 static_cast<unsigned long long>(r.pulsim_fallbacks));
}

/// Resolve the output CSV path. Env var `PULSIM_BENCH_RESULTS_DIR` wins;
/// otherwise fall back to a `bench-results/` directory next to the
/// current binary (cmake/ninja default `build/`).
std::filesystem::path resolve_csv_path() {
    const char* dir = std::getenv("PULSIM_BENCH_RESULTS_DIR");
    std::filesystem::path base =
        (dir != nullptr) ? std::filesystem::path(dir)
                          : std::filesystem::path("bench-results");
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base / "rank1_microbench.csv";
}

void write_csv(const std::vector<Row>& rows) {
    const auto p = resolve_csv_path();
    std::ofstream ofs(p);
    if (!ofs) {
        std::printf("  (warning) could not open %s for writing\n",
                     p.string().c_str());
        return;
    }
    ofs << "N,n_state,n_calls,"
           "wall_solve_ms,us_per_solve,"
           "wall_eigen_ms,us_per_eigen,speedup_eigen_vs_solve,"
           "eigen_rank1_hits,eigen_fallbacks,"
           "wall_pulsim_ms,us_per_pulsim,speedup_pulsim_vs_solve,"
           "speedup_pulsim_vs_eigen,"
           "pulsim_rank1_hits,pulsim_fallbacks\n";
    for (const auto& r : rows) {
        ofs << r.N << ',' << r.n_state << ',' << r.n_calls << ','
            << r.wall_solve_ms << ',' << r.per_call_solve_us << ','
            << r.wall_eigen_ms << ',' << r.per_call_eigen_us << ','
            << r.speedup_eigen_vs_solve << ','
            << r.eigen_rank1_hits << ',' << r.eigen_fallbacks << ','
            << r.wall_pulsim_ms << ',' << r.per_call_pulsim_us << ','
            << r.speedup_pulsim_vs_solve << ','
            << r.speedup_pulsim_vs_eigen << ','
            << r.pulsim_rank1_hits << ',' << r.pulsim_fallbacks << '\n';
    }
    std::printf("  wrote %s\n", p.string().c_str());
}

}  // namespace

TEST_CASE("Bench: rank-1 microbench — 3-backend comparison across N",
          "[bench][rank1][microbench][pwl]") {
    std::printf("\n=== PWL cache rank-1 microbench (3-backend, v1.3.0) ===\n");
    std::printf("Backends:\n");
    std::printf("  (A) baseline `solve`   — per-mask cache, lazy mode "
                "(each mask is a fresh analyze+factorize miss)\n");
    std::printf("  (B) `solve_rank1` Eigen — sliding-solver fallback "
                "(amortises symbolic analyze; full numeric factorize "
                "every flip)\n");
    std::printf("  (C) `solve_rank1` Pulsim — path-based partial "
                "refactor on the in-house sparse LU (v1.3.0 default)\n");
    std::printf("\nColumn legend:\n");
    std::printf("  E/solve = speedup of Eigen   sliding solver vs baseline\n");
    std::printf("  P/solve = speedup of Pulsim path-based vs baseline\n");
    std::printf("  rank1(P)/fallbacks(P) = path-based hit/fall counts\n");
    print_header();

    std::vector<Row> rows;
    for (Size N : {Size{4}, Size{6}, Size{8}, Size{10}, Size{12},
                    Size{16}, Size{20}, Size{24}}) {
        constexpr Size REPEAT = 4;
        const Row r = run_one(N, REPEAT);
        print_row(r);
        rows.push_back(r);

        // Smoke check: all 3 backends produced output
        REQUIRE(r.n_calls > 0);
        REQUIRE(r.wall_solve_ms > 0.0);
        REQUIRE(r.wall_eigen_ms > 0.0);
        REQUIRE(r.wall_pulsim_ms > 0.0);
    }

    write_csv(rows);
}
