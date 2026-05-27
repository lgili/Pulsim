// =============================================================================
// Benchmark — AC sweep complex sparse LU (v1.4.0 in-house vs Eigen baseline)
// =============================================================================
//
// `openspec/changes/add-pulsim-complex-sparse-lu` Section 6 — TPEL §VI.B
// AC-sweep numbers.
//
// Per-frequency cost is the headline number for the AC-sweep path: every
// frequency in an AC sweep triggers one fresh `analyze + factorize +
// solve` cycle on the complex sparse matrix M_k = jω_k·E + J.
//
// This bench drives that pattern on synthetic SPD-like complex MNA
// fixtures of varying state size n (n = 8, 16, 32, 64, 128) and
// compares two backends:
//
//   (A) Backend::Eigen  — Eigen::SparseLU<complex<Real>, COLAMD>.
//       The v1.3.0 production path AND the IEEE TPEL §VI.B paper-
//       comparison baseline.
//
//   (B) Backend::Pulsim — PulsimComplexSparseLuSolver
//       (= PulsimSparseLuSolverT<std::complex<Real>>).
//       The v1.4.0 production path. Same algorithms as the real-scalar
//       v1.3.0 solver (RCM + Liu-Davis etree + Gilbert-Peierls + path-
//       based partial_refactor) on top of complex arithmetic.
//
// Expected result: ROUGH PARITY (within ~1.5× either way). v1.4.0's
// headline isn't a per-frequency speedup — it's "zero third-party LU
// in production" (software supply-chain argument). The bench captures
// the parity number for the paper rather than chasing a perf delta.
//
// Output:
//   * Pretty-printed table to stdout
//   * Machine-readable CSV at
//     artigos/02_tpel_methods/benchmarks/results/ac_sweep_microbench.csv
//     (path overridable via PULSIM_BENCH_RESULTS_DIR env var; falls back
//     to a `bench-results/` directory next to the binary)
//
// Excluded from `ctest` (lives in the `pulsim_benchmarks` opt-in target).
// Run:
//   cmake --build build --target pulsim_benchmarks
//   ./build/core/pulsim_benchmarks "[ac_sweep][microbench]"

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/pulsim_lu_solver.hpp"
#include "pulsim/sparse/solver.hpp"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <vector>

using namespace pulsim;
using namespace pulsim::bench;

namespace {

using Complex            = std::complex<Real>;
using ComplexSparseMatrix = sparse::MatrixT<Complex>;
using ComplexVector       = pulsim::VectorT<Complex>;

constexpr Real PI = std::numbers::pi_v<Real>;

// -----------------------------------------------------------------------------
// Synthetic MNA-shaped fixture
// -----------------------------------------------------------------------------
//
// We need a complex sparse matrix family parametrised on (n, ω) that
// shares the structural properties of a real circuit MNA matrix at
// frequency ω:
//
//   * Symmetric pattern (= |J| + |J^T|), tri-diagonal-dominant
//     (bandwidth 1 → roughly the post-RCM fill of buck / NPC / MMC).
//   * Real-valued conductance on the diagonal (R + numerical anchor).
//   * Frequency-dependent imaginary entry on the diagonal (j·ω·C →
//     +j·ω·C on the cap's pair of nodes; we lump it into the diagonal
//     here because we're benchmarking the solver, not the stamp).
//   * Off-diagonal real-only entries (−1·resistor conductance).
//
// This matches the structure of `(jω·E + J)` for a chain of R-C
// stages plus a few off-diagonal couplings — typical SMPS MNA.
//
// The matrix is strictly diagonally dominant, so partial pivoting at
// the diagonal is the right choice (no fallback to PIVOT_THRESH).

struct AcFixture {
    Index n;
    // The static (frequency-independent) real part J.
    sparse::Matrix J_real;
    // The capacitance diagonal — entry i contains C_i [F].
    std::vector<Real> C_diag;

    explicit AcFixture(Index n_in) : n(n_in), C_diag(n_in) {
        std::vector<sparse::Triplet> trips;
        trips.reserve(static_cast<std::size_t>(3 * n));
        // Diagonal: 1 + i·0.001 conductance per node (R = 1 + small Z).
        for (Index i = 0; i < n; ++i) {
            trips.emplace_back(i, i, Real{2.0});  // 2 Ω equivalent
        }
        // Off-diagonal couplings: -1 on (i, i+1) and (i+1, i), mimicking
        // a daisy-chain resistor network.
        for (Index i = 0; i + 1 < n; ++i) {
            trips.emplace_back(i,     i + 1, Real{-1.0});
            trips.emplace_back(i + 1, i,     Real{-1.0});
        }
        // Add a few "voltage-source" asymmetries (couples to ground via
        // a large anchor) to exercise the partial-pivot path.
        // For n ≥ 8, anchor node 0 with 1e6 to avoid singularity.
        trips.emplace_back(0, 0, Real{1.0e6});
        J_real = sparse::Matrix(n, n);
        J_real.setFromTriplets(trips.begin(), trips.end());
        J_real.makeCompressed();

        // Per-node capacitance: 1 nF on every node (small enough that the
        // imaginary contribution is small at low freq, large enough that
        // it dominates at high freq → exercises the full magnitude range).
        for (Index i = 0; i < n; ++i) {
            C_diag[static_cast<std::size_t>(i)] = Real{1e-9};
        }
    }

    /// Build M(ω) = J + jω·diag(C). Returns a ColMajor complex sparse
    /// matrix ready to feed any DirectSolverT<Complex> backend.
    [[nodiscard]] ComplexSparseMatrix build_M(Real omega) const {
        ComplexSparseMatrix M(n, n);
        std::vector<sparse::TripletT<Complex>> trips;
        trips.reserve(static_cast<std::size_t>(J_real.nonZeros() + n));

        // Copy J_real into M as complex (im = 0).
        for (Index k = 0; k < J_real.outerSize(); ++k) {
            for (sparse::Matrix::InnerIterator it(J_real, k); it; ++it) {
                trips.emplace_back(it.row(), it.col(),
                                    Complex{it.value(), Real{0}});
            }
        }
        // Add jω·C on the diagonal.
        for (Index i = 0; i < n; ++i) {
            trips.emplace_back(i, i,
                Complex{Real{0}, omega * C_diag[static_cast<std::size_t>(i)]});
        }
        M.setFromTriplets(trips.begin(), trips.end());
        M.makeCompressed();
        return M;
    }
};

// -----------------------------------------------------------------------------
// One bench row
// -----------------------------------------------------------------------------

struct Row {
    Index  n;
    Index  n_freqs;
    Index  matrix_nnz;
    // (A) Eigen baseline
    double wall_eigen_ms;
    double per_freq_eigen_us;
    // (B) Pulsim in-house
    double wall_pulsim_ms;
    double per_freq_pulsim_us;
    double speedup_pulsim_vs_eigen;   // wall_eigen / wall_pulsim
    // Correctness cross-check at one mid frequency
    double max_solve_diff;
};

Row run_one(Index n) {
    AcFixture fx(n);
    constexpr Index N_FREQS = 100;
    const Real f_lo = Real{1.0};
    const Real f_hi = Real{1.0e6};
    std::vector<Real> freqs(static_cast<std::size_t>(N_FREQS));
    const Real lo = std::log10(f_lo);
    const Real hi = std::log10(f_hi);
    for (Index k = 0; k < N_FREQS; ++k) {
        const Real e = lo + (hi - lo) * static_cast<Real>(k) /
                              static_cast<Real>(N_FREQS - 1);
        freqs[static_cast<std::size_t>(k)] = std::pow(Real{10}, e);
    }

    // Right-hand side: unit input at node 0 (mimics the unit-impulse
    // excitation in `run_mna_sweep`).
    ComplexVector rhs = ComplexVector::Zero(n);
    rhs[0] = Complex{Real{1}, Real{0}};

    auto sweep_with_backend = [&](sparse::Backend backend)
        -> std::tuple<double, ComplexVector> {
        Stopwatch sw;
        ComplexVector x_last;
        for (Index k = 0; k < N_FREQS; ++k) {
            const Real omega = Real{2} * PI *
                                  freqs[static_cast<std::size_t>(k)];
            ComplexSparseMatrix M = fx.build_M(omega);
            auto solver = sparse::make_default_solver_t<Complex>(
                static_cast<Size>(n), backend);
            const bool ok_a = solver->analyze(M);
            REQUIRE(ok_a);
            const bool ok_f = solver->factorize(M);
            REQUIRE(ok_f);
            ComplexVector x;
            solver->solve(rhs, x);
            if (k == N_FREQS / 2) {
                x_last = x;
            }
        }
        return {sw.elapsed_ms(), x_last};
    };

    auto [wall_eigen_ms,  x_eigen]  =
        sweep_with_backend(sparse::Backend::Eigen);
    auto [wall_pulsim_ms, x_pulsim] =
        sweep_with_backend(sparse::Backend::Pulsim);

    // Cross-check: both backends should produce the same answer at the
    // mid frequency within solver tolerance (1e-10).
    Real max_diff = 0;
    for (Index i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(x_eigen[i] - x_pulsim[i]));
    }

    const Index matrix_nnz = fx.build_M(Real{1}).nonZeros();

    return Row{
        .n                       = n,
        .n_freqs                 = N_FREQS,
        .matrix_nnz              = matrix_nnz,
        .wall_eigen_ms           = wall_eigen_ms,
        .per_freq_eigen_us       = (wall_eigen_ms * 1000.0) /
                                    static_cast<double>(N_FREQS),
        .wall_pulsim_ms          = wall_pulsim_ms,
        .per_freq_pulsim_us      = (wall_pulsim_ms * 1000.0) /
                                    static_cast<double>(N_FREQS),
        .speedup_pulsim_vs_eigen = wall_eigen_ms / wall_pulsim_ms,
        .max_solve_diff          = max_diff,
    };
}

void print_header() {
    std::printf("\n%-5s %-7s %-7s %-12s %-12s %-12s %-12s %-12s\n",
                 "n", "nnz", "n_freq",
                 "ms (Eigen)", "us/freq (E)",
                 "ms (Pulsim)", "us/freq (P)",
                 "speedup E/P");
    std::printf("----- ------- ------- ------------ ------------ "
                 "------------ ------------ ------------\n");
}

void print_row(const Row& r) {
    std::printf("%-5lld %-7lld %-7lld %-9.3f ms %-9.3f us %-9.3f ms "
                 "%-9.3f us %-6.3fx   (parity check Δ = %.2e)\n",
                 static_cast<long long>(r.n),
                 static_cast<long long>(r.matrix_nnz),
                 static_cast<long long>(r.n_freqs),
                 r.wall_eigen_ms,  r.per_freq_eigen_us,
                 r.wall_pulsim_ms, r.per_freq_pulsim_us,
                 r.speedup_pulsim_vs_eigen,
                 r.max_solve_diff);
}

std::filesystem::path resolve_csv_path() {
    const char* dir = std::getenv("PULSIM_BENCH_RESULTS_DIR");
    std::filesystem::path base =
        (dir != nullptr) ? std::filesystem::path(dir)
                          : std::filesystem::path("bench-results");
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base / "ac_sweep_microbench.csv";
}

void write_csv(const std::vector<Row>& rows) {
    const auto p = resolve_csv_path();
    std::ofstream ofs(p);
    if (!ofs) {
        std::printf("  (warning) could not open %s for writing\n",
                     p.string().c_str());
        return;
    }
    ofs << "n,matrix_nnz,n_freqs,"
           "wall_eigen_ms,us_per_freq_eigen,"
           "wall_pulsim_ms,us_per_freq_pulsim,speedup_pulsim_vs_eigen,"
           "max_solve_diff\n";
    for (const auto& r : rows) {
        ofs << r.n << ',' << r.matrix_nnz << ',' << r.n_freqs << ','
            << r.wall_eigen_ms  << ',' << r.per_freq_eigen_us  << ','
            << r.wall_pulsim_ms << ',' << r.per_freq_pulsim_us << ','
            << r.speedup_pulsim_vs_eigen << ','
            << r.max_solve_diff << '\n';
    }
    std::printf("  wrote %s\n", p.string().c_str());
}

}  // namespace

TEST_CASE("Bench: AC sweep complex sparse LU — Pulsim vs Eigen (v1.4.0)",
          "[bench][ac_sweep][microbench][complex]") {
    std::printf("\n=== AC sweep complex sparse LU microbench (v1.4.0) ===\n");
    std::printf("Pattern per frequency: build M(ω) + analyze + factorize + solve\n");
    std::printf("Backends:\n");
    std::printf("  (A) Eigen  — Eigen::SparseLU<complex<Real>, COLAMD>\n");
    std::printf("              (v1.3.0 production path + IEEE TPEL paper-comparison baseline)\n");
    std::printf("  (B) Pulsim — PulsimComplexSparseLuSolver (in-house, v1.4.0 production)\n");
    std::printf("Sweep: 100 log-spaced frequencies from 1 Hz to 1 MHz on each fixture\n");
    std::printf("Parity tolerance: 1e-10 (both backends must agree at f = mid-sweep)\n");
    print_header();

    std::vector<Row> rows;
    for (Index n : {Index{8}, Index{16}, Index{32}, Index{64}, Index{128}}) {
        const Row r = run_one(n);
        print_row(r);
        rows.push_back(r);

        // Parity: backends must agree within 1e-10 at the mid frequency.
        // Looser than the 1e-12 LU identity because both Eigen + Pulsim
        // accumulate fp round-off across the analyze / factorize chain
        // independently — 1e-10 catches genuine bugs while leaving room
        // for the fact that COLAMD vs RCM pick different column orderings
        // (different round-off pattern, same answer up to tolerance).
        REQUIRE(r.max_solve_diff < 1e-10);
        REQUIRE(r.wall_eigen_ms  > 0.0);
        REQUIRE(r.wall_pulsim_ms > 0.0);
    }

    write_csv(rows);
}
