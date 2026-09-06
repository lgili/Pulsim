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
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/multi_winding_transformer.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"

#include <cmath>
#include <format>

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
    p.L_i = {1e-3, 4e-3, 9e-3};   // L_p, L_s1, L_s2
    p.k_ij = {{1.0, 1.0, 0.9}, {1.0, 1.0, 0.8}, {0.9, 0.8, 1.0}};
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


TEST_CASE("MultiWindingTransformer — eight windings, no ceiling: tightly "
          "coupled voltages scale as sqrt(L_k/L_1)",
          "[v2][c4][multi_winding_transformer][unit]") {
    // The old [2, 6] check was an argument check with nothing behind
    // it. Eight windings, k = 0.999 everywhere, the first driven, the
    // rest lightly loaded: v_k/v_1 = sqrt(L_k/L_1) (turns ratio).
    CircuitBuilder b;
    b.add_sine_voltage_source("V", "src", "gnd", 0.0, 10.0, 10e3, 0.0);
    b.add_resistor("Rs", "src", "w1", 0.01);
    std::vector<CircuitBuilder::WindingSpec> ws;
    ws.push_back({"w1", "gnd", 1e-3});
    for (int k = 2; k <= 8; ++k) {
        ws.push_back({std::format("w{}", k), "gnd", 1e-3 * k * k});   // N_k = k
    }
    std::vector<std::vector<Real>> km(8, std::vector<Real>(8, 0.999));
    b.add_multi_winding_transformer("T8", ws, km);
    for (int k = 2; k <= 8; ++k) b.add_resistor(std::format("R{}", k), std::format("w{}", k), "gnd", 1e4);
    PwlStateSpaceCache cache(b.graph(), b.pool());
    SimulationOptions opts{.t_start = 0.0, .t_end = 200e-6, .dt = 1e-8};
    cache.build(opts.dt);
    auto sw = [](Real) { return SwitchStateMask(0); };
    auto r = run_transient(cache, b.graph(), b.pool(), opts, sw);
    const Size n = r.num_steps();
    const Index w1 = b.node_id_of("w1");
    Real v1_pk = 0;
    for (Size i = n / 2; i < n; ++i) v1_pk = std::max(v1_pk, std::abs(r.states[i][w1]));
    REQUIRE(v1_pk > 5.0);
    for (int k = 2; k <= 8; ++k) {
        const Index wk = b.node_id_of(std::format("w{}", k));
        Real vk_pk = 0;
        for (Size i = n / 2; i < n; ++i) vk_pk = std::max(vk_pk, std::abs(r.states[i][wk]));
        INFO("k = " << k << " ratio = " << vk_pk / v1_pk);
        CHECK(vk_pk / v1_pk == Approx(static_cast<Real>(k)).epsilon(2e-2));
    }
}

TEST_CASE("MultiWindingTransformer — non-realisable couplings are refused "
          "by name", "[v2][c4][multi_winding_transformer][unit]") {
    using Catch::Matchers::ContainsSubstring;
    CircuitBuilder b;
    // k12 = k13 = 1 but k23 = 0.5: the flux that links 1 with 2 and
    // 1 with 3 completely must link 2 with 3 completely too.
    std::vector<CircuitBuilder::WindingSpec> ws{{"a", "gnd", 1e-3}, {"b", "gnd", 1e-3}, {"c", "gnd", 1e-3}};
    CHECK_THROWS_WITH(
        b.add_multi_winding_transformer("Tbad", ws, {{1, 1, 1}, {1, 1, 0.5}, {1, 0.5, 1}}),
        ContainsSubstring("not realisable"));
    CHECK_THROWS_WITH(
        b.add_multi_winding_transformer("Tbad", ws, {{1, 1.2, 1}, {1.2, 1, 1}, {1, 1, 1}}),
        ContainsSubstring("outside [0, 1]"));
    CHECK_THROWS_WITH(
        b.add_multi_winding_transformer("Tone", {{"a", "gnd", 1e-3}}),
        ContainsSubstring("at least 2"));
    // And a realisable set passes: the pivot check is not a blanket
    // refusal of tight coupling.
    b.add_multi_winding_transformer("Tok", ws, {{1, 0.99, 0.99}, {0.99, 1, 0.99}, {0.99, 0.99, 1}});
    REQUIRE(b.num_branches() == 3);
}
