// update-linear-time-assembly task 8.1 — tier 3 evidence: does a stored
// reciprocal of U's diagonal speed up the triangular solve enough
// (>= 1.15x at n >= 1500) to justify changing the rounding?
//
// Factorises the RLC ladder's J (the golden probe's circuit) with the
// in-house LU, extracts L, U and both permutations, and times two copies
// of the solver's substitute_() loop that differ only in `y[k] /= ukk`
// versus `y[k] *= rinv[k]`. Same flags as the project (-ffp-contract=off).
//
//   W=/Users/lgili/pulsim_wt/p1-linalg
//   c++ -std=c++2b -O3 -DNDEBUG -ffp-contract=off -I$W/core/include \
//       -I$W/.scratch/build-ctest/_deps/yaml-cpp-src/include \
//       -isystem /opt/homebrew/include/eigen3 solve_reciprocal_bench.cpp -o srb
//   ./srb
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/sparse/pulsim_lu_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <vector>

using namespace pulsim;

namespace {

struct Csc {
    std::vector<Index> p, i;
    std::vector<Real> x;
};

Csc to_csc(const sparse::Matrix& M) {
    Csc c;
    c.p.assign(M.outerIndexPtr(), M.outerIndexPtr() + M.cols() + 1);
    c.i.assign(M.innerIndexPtr(), M.innerIndexPtr() + M.nonZeros());
    c.x.assign(M.valuePtr(), M.valuePtr() + M.nonZeros());
    return c;
}

// Same loops as PulsimSparseLuSolverT::substitute_ (U diagonal last).
template <bool kRecip>
void substitute(const Csc& L, const Csc& U, const std::vector<Real>& rinv,
                const std::vector<Index>& prow, const std::vector<Index>& pcol,
                const Vector& b, Vector& y, Vector& x) {
    const auto n = static_cast<Index>(prow.size());
    for (Index i = 0; i < n; ++i) y[i] = b[prow[static_cast<std::size_t>(i)]];
    for (Index k = 0; k < n; ++k) {
        const Real yk = y[k];
        for (Index q = L.p[k]; q < L.p[k + 1]; ++q) y[L.i[q]] -= L.x[q] * yk;
    }
    for (Index k = n - 1; k >= 0; --k) {
        const Index d = U.p[k + 1] - 1;
        if constexpr (kRecip) {
            y[k] *= rinv[static_cast<std::size_t>(k)];
        } else {
            y[k] /= U.x[d];
        }
        const Real yk = y[k];
        for (Index q = U.p[k]; q < d; ++q) y[U.i[q]] -= U.x[q] * yk;
    }
    for (Index k = 0; k < n; ++k) x[pcol[static_cast<std::size_t>(k)]] = y[k];
}

}  // namespace

int main() {
    for (int N : {50, 500, 5000}) {
        builder::CircuitBuilder b;
        b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3);
        for (int k = 0; k < N; ++k) {
            b.add_inductor(std::format("L{}", k), std::format("n{}", k), std::format("m{}", k), 1e-5);
            b.add_resistor(std::format("R{}", k), std::format("m{}", k), std::format("n{}", k + 1), 0.1);
            b.add_capacitor(std::format("C{}", k), std::format("n{}", k + 1), "gnd", 1e-7);
        }
        b.add_resistor("Rload", std::format("n{}", N), "gnd", 50.0);
        pwl::PwlStateSpaceCache cache(b.graph(), b.pool());
        cache.build_lazy(1e-7);
        const auto& seg = cache.lookup(topology::SwitchStateMask(0));
        sparse::PulsimSparseLuSolver lu;
        if (!lu.analyze(seg.J) || !lu.factorize(seg.J)) return 1;
        const Csc L = to_csc(lu.extract_L_matrix());
        const Csc U = to_csc(lu.extract_U_matrix());
        const auto n = static_cast<Index>(seg.state_size);
        std::vector<Index> prow(lu.row_permutation().begin(), lu.row_permutation().end());
        std::vector<Index> pcol(lu.column_permutation().begin(), lu.column_permutation().end());
        std::vector<Real> rinv(static_cast<std::size_t>(n));
        for (Index k = 0; k < n; ++k) rinv[static_cast<std::size_t>(k)] = Real{1} / U.x[U.p[k + 1] - 1];
        Vector rhs = Vector::LinSpaced(n, 1.0, 2.0), y(n), xd(n), xr(n), xs(n);
        lu.solve(rhs, xs);
        substitute<false>(L, U, rinv, prow, pcol, rhs, y, xd);
        const bool same_as_solver = xd == xs;
        const int reps = std::max(200, 2000000 / static_cast<int>(n));
        double best_d = 1e99, best_r = 1e99;
        for (int round = 0; round < 7; ++round) {   // interleaved, best of 7
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) substitute<false>(L, U, rinv, prow, pcol, rhs, y, xd);
            auto t1 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) substitute<true>(L, U, rinv, prow, pcol, rhs, y, xr);
            auto t2 = std::chrono::steady_clock::now();
            best_d = std::min(best_d, std::chrono::duration<double, std::nano>(t1 - t0).count() / reps);
            best_r = std::min(best_r, std::chrono::duration<double, std::nano>(t2 - t1).count() / reps);
        }
        Real maxrel = 0;
        for (Index i = 0; i < n; ++i)
            maxrel = std::max(maxrel, std::abs(xr[i] - xd[i]) / std::max(Real{1e-300}, std::abs(xd[i])));
        std::printf("n=%6d nnz(L)=%7zu nnz(U)=%7zu  divide %10.0f ns  reciprocal %10.0f ns  "
                    "speed-up %.3fx  max rel diff %.2e  divide-copy == solver: %s\n",
                    static_cast<int>(n), L.x.size(), U.x.size(), best_d, best_r, best_d / best_r,
                    maxrel, same_as_solver ? "yes" : "NO");
    }
    return 0;
}
