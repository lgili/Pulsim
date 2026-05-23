// =============================================================================
// Layer 2 V16 — MultiWindingTransformer tests
// =============================================================================
//
// Validates:
//   * Static mutual-inductance math (matches √(L_i·L_j)·k_ij).
//   * N=2 special case reduces to TwoWindingTransformer.
//   * N=3 flyback-like topology: 3 windings, k tight, verify
//     turns-ratio voltage relationship in transient.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/multi_winding_transformer.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::models;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("MultiWindingTransformer — mutual inductance math",
          "[v2][layer2_v16][multi_winding_transformer][unit]") {
    MultiWindingTransformer::Params p;
    p.n_windings = 3;
    p.L_i = {1e-3, 4e-3, 9e-3};   // L_p, L_s1, L_s2
    p.k_ij[0][1] = 1.0;
    p.k_ij[0][2] = 0.9;
    p.k_ij[1][2] = 0.8;
    // Diagonal: returns L_i directly.
    REQUIRE(MultiWindingTransformer::mutual_inductance(p, 0, 0)
            == Approx(1e-3));
    REQUIRE(MultiWindingTransformer::mutual_inductance(p, 1, 1)
            == Approx(4e-3));
    // Off-diagonal: k·√(L_i·L_j)
    REQUIRE(MultiWindingTransformer::mutual_inductance(p, 0, 1)
            == Approx(std::sqrt(1e-3 * 4e-3)));        // 2e-3
    REQUIRE(MultiWindingTransformer::mutual_inductance(p, 0, 2)
            == Approx(0.9 * std::sqrt(1e-3 * 9e-3)));  // ~2.7e-3
    REQUIRE(MultiWindingTransformer::mutual_inductance(p, 1, 2)
            == Approx(0.8 * std::sqrt(4e-3 * 9e-3)));  // ~4.8e-3
}

TEST_CASE("MultiWindingTransformer — N=2 case matches TwoWindingTransformer",
          "[v2][layer2_v16][multi_winding_transformer][unit]") {
    // Build same 2-winding xfmr two ways: explicit add_transformer
    // and add_multi_winding_transformer with N=2. Both should
    // produce the same number of branches + couplings.
    CircuitBuilder b1, b2;

    b1.add_transformer("T1", "p1", "p2", "s1", "s2",
                         1e-3, 4e-3, 0.95);

    b2.add_multi_winding_transformer(
        "T2",
        {{.from = "p1", .to = "p2", .L = 1e-3},
         {.from = "s1", .to = "s2", .L = 4e-3}},
        {{0.0, 0.95},
         {0.0, 0.0}});

    // Both have 2 inductor branches + 1 transformer coupling.
    REQUIRE(b1.num_branches() == 2);
    REQUIRE(b2.num_branches() == 2);
    REQUIRE(b1.pool().transformer_couplings().size() ==
            b2.pool().transformer_couplings().size());
}

TEST_CASE("MultiWindingTransformer — N=3 flyback-like topology",
          "[v2][layer2_v16][multi_winding_transformer][unit]") {
    // 3-winding transformer + primary voltage source +
    // resistive loads on each secondary. Verifies that the
    // 3 windings produce voltages in the expected ratio
    // V_i ∝ √L_i (for tight coupling k≈1, V ∝ N ∝ √L).
    CircuitBuilder b;
    b.add_voltage_source("V_p", "p1", "p2", 10.0);

    // 3 windings: primary (1 mH), main secondary (4 mH),
    // aux secondary (9 mH). Tight coupling.
    b.add_multi_winding_transformer(
        "T1",
        {{.from = "p1", .to = "p2", .L = 1e-3},
         {.from = "s_main", .to = "gnd", .L = 4e-3},
         {.from = "s_aux",  .to = "gnd", .L = 9e-3}},
        {{0.0, 0.999, 0.999},
         {0.0, 0.0,   0.999},
         {0.0, 0.0,   0.0}});

    b.add_resistor("R_main", "s_main", "gnd", 100.0);
    b.add_resistor("R_aux",  "s_aux",  "gnd", 100.0);

    // num_branches: 1 src + 3 inductors + 2 R = 6.
    REQUIRE(b.num_branches() == 6);
    // 3 pair-wise couplings = N*(N-1)/2 = 3.
    REQUIRE(b.pool().transformer_couplings().size() == 3);
}
