// =============================================================================
// Layer 0 — Phase-1 LU core: scaling + ordering regression tests
// =============================================================================
//
// The v1.3–v1.7 factorize() carried dense inner loops (the L-update
// scanned ALL prior columns; pivot swaps relabelled every stored L
// column), giving an O(n²) floor even for tridiagonal matrices — the
// audit's hardest scalability blocker (finding lu-effectively-dense /
// lu-quadratic-inner-loops, CONFIRMED). The Phase-1 rewrite is a true
// Gilbert–Peierls left-looking factorization (DFS reach, cs_lu-style
// O(1) pivoting via pinv, O(nnz) final relabel) with COLAMD ordering.
//
// These tests lock the rewrite in:
//   1. correctness at MNA-like structures across sizes (residual);
//   2. the complexity LAW — banded factor time must scale roughly
//      linearly in n (generous 3× headroom on the ratio to absorb CI
//      noise; the OLD dense implementation shows ~n² ratios and fails
//      this by an order of magnitude);
//   3. COLAMD < RCM fill on a 2-D grid (the classic separation);
//   4. Phase-1 gate: a 5 000-unknown circuit-like factor stays under
//      a very generous wall-clock bound on any CI machine.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/sparse/pulsim_lu_solver.hpp"

#include <chrono>
#include <random>
#include <vector>

using namespace pulsim;
using namespace pulsim::sparse;

namespace {

/// Ladder network (tridiagonal conductances) + a light sprinkling of
/// long-range couplings — the shape of real MNA (quasi-banded, a few
/// global rails), far closer to circuits than uniform random sparsity.
Matrix make_mna_like(Index n, unsigned seed, double coupling_frac) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> g(0.1, 10.0);
    std::uniform_int_distribution<Index> pick(0, n - 1);
    std::vector<TripletT<Real>> t;
    for (Index i = 0; i < n; ++i) {
        t.emplace_back(i, i, Real{3} + g(rng));
        if (i > 0) {
            t.emplace_back(i, i - 1, -g(rng));
            t.emplace_back(i - 1, i, -g(rng));
        }
    }
    const auto extras = static_cast<Index>(
        static_cast<double>(n) * coupling_frac);
    for (Index e = 0; e < extras; ++e) {
        const Index a = pick(rng);
        const Index b = pick(rng);
        if (a != b) {
            t.emplace_back(a, b, -g(rng));
            t.emplace_back(b, a, -g(rng));
        }
    }
    Matrix M(n, n);
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    return M;
}

double factor_us(const Matrix& M, PulsimSparseLuSolver& s) {
    REQUIRE(s.analyze(M));
    const auto t0 = std::chrono::high_resolution_clock::now();
    REQUIRE(s.factorize(M));
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

Real relative_residual(const Matrix& M, PulsimSparseLuSolver& s,
                        unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    Vector b(M.rows());
    for (Index i = 0; i < M.rows(); ++i) b[i] = u(rng);
    Vector x;
    s.solve(b, x);
    return (M * x - b).norm() / b.norm();
}

}  // namespace

TEST_CASE("Phase-1 LU: correctness across MNA-like sizes",
          "[pulsim_lu][scaling]") {
    for (Index n : {200, 1000, 3000}) {
        Matrix M = make_mna_like(n, 7u + static_cast<unsigned>(n), 0.01);
        PulsimSparseLuSolver s;
        REQUIRE(s.analyze(M));
        REQUIRE(s.factorize(M));
        REQUIRE(relative_residual(M, s, 99u) < Real{1e-9});
    }
}

TEST_CASE("Phase-1 LU: banded factor time scales ~linearly (kills the "
          "O(n^2) floor)",
          "[pulsim_lu][scaling]") {
    // Pure tridiagonal: zero fill under any sane ordering, so a true
    // GP factorization is O(n). The old dense j<k loop was O(n²) here
    // — ratio t(4n)/t(n) ≈ 16. We assert the ratio stays below 12
    // (linear predicts ≈ 4; wide margin for CI-machine noise).
    const Index n1 = 2000;
    const Index n2 = 8000;
    Matrix M1 = make_mna_like(n1, 11u, 0.0);
    Matrix M2 = make_mna_like(n2, 13u, 0.0);
    PulsimSparseLuSolver s1, s2;
    // Warm-up + median-of-3 to tame scheduler noise.
    auto med3 = [](const Matrix& M, PulsimSparseLuSolver& s) {
        double a = factor_us(M, s);
        double b = factor_us(M, s);
        double c = factor_us(M, s);
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return b;
    };
    const double t1 = med3(M1, s1);
    const double t2 = med3(M2, s2);
    INFO("t(n=2000)=" << t1 << "us  t(n=8000)=" << t2
         << "us  ratio=" << t2 / t1);
    REQUIRE(t2 / t1 < 12.0);
}

TEST_CASE("Phase-1 LU: COLAMD beats RCM on a 2-D grid",
          "[pulsim_lu][ordering]") {
    // The classic ordering separation: on a k x k grid Laplacian, a
    // bandwidth ordering (RCM) fills ~ n * k = n^1.5 while a
    // fill-minimising ordering lands near n log n. (The arrow matrix
    // is NOT a witness — reverse-CM also pushes its hub to the end.)
    const Index k = 30;
    const Index n = k * k;
    std::vector<TripletT<Real>> t;
    auto id = [k](Index r, Index c) { return r * k + c; };
    for (Index r = 0; r < k; ++r) {
        for (Index c = 0; c < k; ++c) {
            t.emplace_back(id(r, c), id(r, c), Real{4});
            if (r + 1 < k) {
                t.emplace_back(id(r, c), id(r + 1, c), Real{-1});
                t.emplace_back(id(r + 1, c), id(r, c), Real{-1});
            }
            if (c + 1 < k) {
                t.emplace_back(id(r, c), id(r, c + 1), Real{-1});
                t.emplace_back(id(r, c + 1), id(r, c), Real{-1});
            }
        }
    }
    Matrix M(n, n);
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();

    PulsimSparseLuSolver colamd;
    colamd.set_ordering(LuOrdering::Colamd);
    REQUIRE(colamd.analyze(M));
    REQUIRE(colamd.factorize(M));

    PulsimSparseLuSolver rcm;
    rcm.set_ordering(LuOrdering::Rcm);
    REQUIRE(rcm.analyze(M));
    REQUIRE(rcm.factorize(M));

    INFO("lnnz colamd=" << colamd.l_nnz() << "  rcm=" << rcm.l_nnz());
    // COLAMD is consistently better on grids, but the margin is
    // structure-dependent (measured ~1.3x here; the big Phase-1 win
    // is the O(flops) factorization itself). Assert strictly-better.
    REQUIRE(colamd.l_nnz() < rcm.l_nnz());
    // Both must still be CORRECT.
    REQUIRE(relative_residual(M, colamd, 5u) < Real{1e-9});
    REQUIRE(relative_residual(M, rcm, 5u) < Real{1e-9});
}

TEST_CASE("Phase-1 gate: n=5000 circuit-like factorization under budget",
          "[pulsim_lu][scaling]") {
    // Roadmap gate: "factorar MNA n=5000 < 10 ms". Local Apple-Silicon
    // measurements land at 0.9–7 ms depending on coupling density; the
    // CI assertion uses 60 ms so a loaded shared runner can't flake
    // the suite while an O(n²) regression (~10× above the bound)
    // still fails loudly.
    Matrix M = make_mna_like(5000, 42u, 0.01);
    PulsimSparseLuSolver s;
    const double t_us = factor_us(M, s);
    INFO("factor(n=5000) = " << t_us / 1000.0 << " ms");
    REQUIRE(t_us < 60'000.0);
    REQUIRE(relative_residual(M, s, 1u) < Real{1e-9});
}
