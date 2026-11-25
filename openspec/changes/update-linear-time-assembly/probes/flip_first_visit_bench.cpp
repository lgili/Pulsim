// update-linear-time-assembly tier 2 evidence: cost of a lazy first visit
// with `set_flip_refactor(true)` (partial_refactor on the nearest resident
// mask's factor) versus false (tier 1: full factor on the shared analysis).
// Switched RLC ladder (the golden probe's sw_ladder, K switches spread
// along it); masks visited in Gray
// order so every new mask has a resident neighbour one switch away.
// Needs tier2_flip_refactor_prototype.patch applied (not landed).
//
//   W=/Users/lgili/pulsim_wt/p1-linalg
//   c++ -std=c++2b -O3 -DNDEBUG -ffp-contract=off -I$W/core/include \
//       -I$W/.scratch/build-ctest/_deps/yaml-cpp-src/include \
//       -isystem /opt/homebrew/include/eigen3 flip_first_visit_bench.cpp \
//       $W/.scratch/build-ctest/_deps/yaml-cpp-build/libyaml-cpp.a -o ffv
//   ./ffv
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

using namespace pulsim;

namespace {

builder::CircuitBuilder sw_ladder(int N, int K) {
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3);
    for (int k = 0; k < N; ++k) {
        const auto s = std::to_string(k);
        b.add_inductor("L" + s, "n" + s, "m" + s, 1e-5);
        b.add_resistor("R" + s, "m" + s, "n" + std::to_string(k + 1), 0.1);
        b.add_capacitor("C" + s, "n" + std::to_string(k + 1), "gnd", 1e-7);
    }
    for (int j = 0; j < K; ++j) {
        const auto node = "n" + std::to_string(1 + (j * (N - 1)) / std::max(K - 1, 1));
        b.add_switch("S" + std::to_string(j), node, "s" + std::to_string(j), 1e3, 1e-9);
        b.add_resistor("RS" + std::to_string(j), "s" + std::to_string(j), "gnd", 5.0);
    }
    b.add_resistor("Rload", "n" + std::to_string(N), "gnd", 50.0);
    return b;
}

topology::SwitchStateMask mask_of(Size nbits, unsigned v) {
    topology::SwitchStateMask m(nbits);
    for (Size i = 0; i < nbits; ++i) m.set(i, ((v >> i) & 1U) != 0U);
    return m;
}

// Best of `reps` full Gray walks; returns µs per first visit.
double walk(builder::CircuitBuilder& b, int K, bool flip, int reps,
            pwl::CacheMetrics* out) {
    const auto& g = b.graph();
    const Index n = static_cast<Index>(b.pool().state_size(g));
    const Vector be = Vector::LinSpaced(n, -1.0, 1.0);
    double best = 1e99;
    for (int r = 0; r < reps; ++r) {
        pwl::PwlStateSpaceCache c(g, b.pool());
        c.set_flip_refactor(flip);
        c.build_lazy(1e-7);
        Vector x;
        c.solve(mask_of(static_cast<Size>(K), 0U), be, x);   // first: no neighbour
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned v = 1; v < (1U << K); ++v) {
            c.solve(mask_of(static_cast<Size>(K), v ^ (v >> 1)), be, x);
        }
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() /
                          static_cast<double>((1U << K) - 1);
        best = std::min(best, us);
        if (out) *out = c.metrics();
    }
    return best;
}

}  // namespace

int main() {
    for (auto [N, K] : {std::pair{100, 8}, std::pair{1000, 6}}) {
        auto b = sw_ladder(N, K);
        const auto n = b.pool().state_size(b.graph());
        for (int round = 0; round < 3; ++round) {
            pwl::CacheMetrics m{};
            const double off = walk(b, K, false, 3, nullptr);
            const double on = walk(b, K, true, 3, &m);
            std::printf("sw_ladder n=%5zu K=%d  off %9.1f us/visit  on %9.1f us/visit"
                        "  on/off %.3f  partial %llu fallbacks %llu\n",
                        static_cast<std::size_t>(n), K, off, on, on / off,
                        static_cast<unsigned long long>(m.flip_partial_refactors),
                        static_cast<unsigned long long>(m.flip_fallbacks));
        }
    }
    return 0;
}
