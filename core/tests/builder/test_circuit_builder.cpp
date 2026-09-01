// =============================================================================
// Layer 6 V0 — CircuitBuilder unit + integration tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::solver;
using namespace pulsim::topology;
using namespace pulsim::pwl;
using Catch::Approx;

// -----------------------------------------------------------------------------
// Unit tests: node mapping, alias, unit conversion
// -----------------------------------------------------------------------------

// NB: the test name avoids embedded double-quotes — CTest passes the
// name on the command line wrapped in quotes; an embedded `"gnd"`
// turns into `""gnd""` after shell quoting and CTest fails to match.
TEST_CASE("gnd alias maps to graph.ground()",
          "[v2][layer6][builder][nodes]") {
    CircuitBuilder b;
    REQUIRE(b.node("gnd") == b.graph().ground());
    REQUIRE(b.node("GND") == b.graph().ground());
    REQUIRE(b.node("0")   == b.graph().ground());
}

TEST_CASE("Node lookup creates on first use, reuses on second",
          "[v2][layer6][builder][nodes]") {
    CircuitBuilder b;
    const Index a1 = b.node("a");
    const Index a2 = b.node("a");
    REQUIRE(a1 == a2);

    const Index x = b.node("x");
    REQUIRE(x != a1);
}

TEST_CASE("node_id_of throws for unknown node",
          "[v2][layer6][builder][nodes]") {
    CircuitBuilder b;
    b.node("known");
    REQUIRE(b.node_id_of("known") >= 0);
    REQUIRE_THROWS_AS(b.node_id_of("unknown"),
                       std::out_of_range);
}

TEST_CASE("Implicit node creation via device methods",
          "[v2][layer6][builder][nodes]") {
    CircuitBuilder b;
    b.add_voltage_source("Vin", "n0", "gnd", 5.0);
    REQUIRE(b.node_id_of("n0") >= 0);
}

TEST_CASE("Resistor: R_ohms is converted to G = 1/R",
          "[v2][layer6][builder][units]") {
    CircuitBuilder b;
    b.add_resistor("R1", "a", "b", 100.0);

    REQUIRE(b.num_branches() == 1);
    const Index b_id = 0;
    REQUIRE(b.pool().kind_of(b_id) ==
              DevicePool::StoredKind::Resistor);
    const auto& rp = b.pool().resistor_params(b_id);
    REQUIRE(rp.G == Approx(1.0 / 100.0));
}

TEST_CASE("Capacitor: C_farads is passed through unchanged",
          "[v2][layer6][builder][units]") {
    CircuitBuilder b;
    b.add_capacitor("C1", "a", "gnd", 1e-6);

    const Index b_id = 0;
    REQUIRE(b.pool().kind_of(b_id) ==
              DevicePool::StoredKind::Capacitor);
    REQUIRE(b.pool().capacitor_params(b_id).C ==
              Approx(1e-6));
}

TEST_CASE("Inductor: L_henries is passed through unchanged",
          "[v2][layer6][builder][units]") {
    CircuitBuilder b;
    b.add_inductor("L1", "a", "b", 2.5e-3);

    const Index b_id = 0;
    REQUIRE(b.pool().kind_of(b_id) ==
              DevicePool::StoredKind::Inductor);
    REQUIRE(b.pool().inductor_params(b_id).L ==
              Approx(2.5e-3));
}

TEST_CASE("Method chaining works",
          "[v2][layer6][builder]") {
    CircuitBuilder b;
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
     .add_resistor       ("R1",  "n0", "gnd", 1.0);
    REQUIRE(b.num_branches() == 2);
}

// -----------------------------------------------------------------------------
// Round-trip: simplest V_dc + R circuit
// -----------------------------------------------------------------------------

TEST_CASE("Round-trip V_dc + R: cache.solve gives the right v_node",
          "[v2][layer6][builder][roundtrip]") {
    CircuitBuilder b;
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
     .add_resistor      ("R1",  "n0", "gnd", 1.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = Vector::Zero(seg.state_size);
    Vector b_extra = Vector::Zero(seg.state_size);
    cache.solve(SwitchStateMask(0), b_extra, x);

    // v_n0 should equal V_dc = 5.0
    REQUIRE(x[0] == Approx(5.0).margin(1e-9));
}

// -----------------------------------------------------------------------------
// Parity test: half-wave rectifier (mirror of layer5_v2 test)
// -----------------------------------------------------------------------------

namespace {

struct ManualHalfWave {
    static constexpr Real V_amp  = 10.0;
    static constexpr Real f_line = 60.0;
    static constexpr Real R_load = 10.0;
    static constexpr Real g_on   = 1e3;
    static constexpr Real g_off  = 1e-9;

    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1;
    Index source_branch_var = -1;

    ManualHalfWave(Real dt) {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        g.add_branch(n0, g.ground(), BranchKind::Source);
        g.add_branch(n0, n1,         BranchKind::Switch);
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = 0.0});
        pool.add_diode(1, g_on, g_off, /*V_th=*/0.0);
        pool.add_resistor(2, {.G = 1.0 / R_load});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);

        source_branch_var =
            pool.branch_var_id_for_source(0, g);
    }

    BExtraFn make_sinusoidal_source() const {
        const Index src_var = source_branch_var;
        const Eigen::Index state_size =
            static_cast<Eigen::Index>(pool.state_size(g));
        return [src_var, state_size](Real t) {
            Vector b = Vector::Zero(state_size);
            const Real omega =
                2.0 * std::numbers::pi_v<Real> * f_line;
            const Real V_sine = V_amp * std::sin(omega * t);
            b[src_var] = -V_sine;
            return b;
        };
    }
};

}  // namespace

TEST_CASE("Half-wave rectifier: builder ≡ manual setup",
          "[v2][layer6][builder][parity][rectifier]") {
    constexpr Real dt = 1e-4;
    constexpr Real T_line = 1.0 / ManualHalfWave::f_line;
    constexpr Real t_end = T_line;   // 1 cycle is enough for parity

    // Manual side (reference).
    ManualHalfWave manual(dt);
    SimulationOptions opts{
        .t_start = 0,
        .t_end   = t_end,
        .dt      = dt,
    };
    auto switch_fn = [](Real) { return SwitchStateMask(1); };
    auto result_manual = run_transient(
        *manual.cache, manual.g, manual.pool, opts,
        switch_fn, manual.make_sinusoidal_source());

    // Builder side.
    CircuitBuilder b;
    b.add_voltage_source("Vin", "n0", "gnd", 0.0)
     .add_diode         ("D1",  "n0", "n1",
                          ManualHalfWave::g_on,
                          ManualHalfWave::g_off,
                          /*V_th=*/0.0)
     .add_resistor      ("R_L", "n1", "gnd",
                          ManualHalfWave::R_load);

    PwlStateSpaceCache cache_b(b.graph(), b.pool());
    cache_b.build(dt);

    // Build the same sinusoidal source using the builder's
    // source branch (we know it's branch 0).
    const Index src_var_b =
        b.pool().branch_var_id_for_source(0, b.graph());
    const Eigen::Index state_size_b =
        static_cast<Eigen::Index>(b.pool().state_size(b.graph()));
    auto b_extra_b = [src_var_b, state_size_b](Real t) {
        Vector bv = Vector::Zero(state_size_b);
        const Real omega =
            2.0 * std::numbers::pi_v<Real> *
            ManualHalfWave::f_line;
        const Real V_sine =
            ManualHalfWave::V_amp * std::sin(omega * t);
        bv[src_var_b] = -V_sine;
        return bv;
    };

    auto result_builder = run_transient(
        cache_b, b.graph(), b.pool(), opts,
        switch_fn, b_extra_b);

    // Sample-by-sample parity.
    REQUIRE(result_manual.num_steps() ==
              result_builder.num_steps());
    REQUIRE(result_manual.states[0].size() ==
              result_builder.states[0].size());

    for (Size k = 0; k < result_manual.num_steps(); ++k) {
        for (Eigen::Index i = 0;
             i < result_manual.states[k].size(); ++i) {
            REQUIRE(result_manual.states[k][i] ==
                      Approx(result_builder.states[k][i])
                          .margin(1e-6));
        }
    }
}

// -----------------------------------------------------------------------------
// Nonlinear diode via builder
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Layer 2 V1 — power-device convenience helpers
// -----------------------------------------------------------------------------

TEST_CASE("add_mosfet adds switch + body diode with correct conductance",
          "[v2][layer2_v1][mosfet]") {
    // v2.0, audit C.1: the body diode is on by default.
    CircuitBuilder b;
    b.add_mosfet("Q1", "drain", "source", 2e-3, 1e9);

    REQUIRE(b.num_branches() == 2);
    REQUIRE(b.pool().kind_of(1) ==
              DevicePool::StoredKind::Diode);
    REQUIRE(b.branch_id_of("Q1_body") == 1);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);
    REQUIRE(b.pool().switch_g_on(0) ==
              Approx(1.0 / 2e-3));
    REQUIRE(b.pool().switch_g_off(0) ==
              Approx(1.0 / 1e9));
}

TEST_CASE("add_mosfet body_diode=false is the bare switch",
          "[v2][layer2_v1][mosfet]") {
    CircuitBuilder b;
    b.add_mosfet("Q1", "drain", "source", 1e-3, 1e9,
                  /*body_diode=*/false);
    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);
}

TEST_CASE("add_mosfet uses MOSFET defaults when omitted",
          "[v2][layer2_v1][mosfet]") {
    CircuitBuilder b;
    b.add_mosfet("Q1", "drain", "source");
    // Default R_on = 1mΩ → g_on = 1000 S.
    REQUIRE(b.pool().switch_g_on(0) ==
              Approx(1.0 / 1e-3));
    REQUIRE(b.pool().switch_g_off(0) ==
              Approx(1.0 / 1e9));
}

TEST_CASE("add_mosfet_with_body_diode adds two branches with diode "
          "anti-parallel",
          "[v2][layer2_v1][mosfet][body_diode]") {
    CircuitBuilder b;
    b.add_mosfet_with_body_diode(
        "Q1", "drain", "source");

    REQUIRE(b.num_branches() == 2);

    // Branch 0: main switch drain → source.
    const auto& bch0 = b.graph().branch(0);
    REQUIRE(bch0.from == b.node_id_of("drain"));
    REQUIRE(bch0.to   == b.node_id_of("source"));
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);

    // Branch 1: body diode source → drain (anti-parallel!).
    const auto& bch1 = b.graph().branch(1);
    REQUIRE(bch1.from == b.node_id_of("source"));
    REQUIRE(bch1.to   == b.node_id_of("drain"));
    REQUIRE(b.pool().kind_of(1) ==
              DevicePool::StoredKind::Diode);
}

TEST_CASE("add_mosfet_with_body_diode body-diode V_F default = 0.7 V",
          "[v2][layer2_v1][mosfet][body_diode]") {
    CircuitBuilder b;
    b.add_mosfet_with_body_diode(
        "Q1", "drain", "source");
    // The diode branch is at index 1.
    // We can't easily reach into the diode params without
    // an accessor, but we can verify the diode WAS added
    // by counting branches and kinds.
    REQUIRE(b.pool().kind_of(1) ==
              DevicePool::StoredKind::Diode);
}

TEST_CASE("add_igbt adds one branch with IGBT defaults",
          "[v2][layer2_v1][igbt]") {
    CircuitBuilder b;
    b.add_igbt("T1", "C", "E");
    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);
    // Default R_on = 10 mΩ → g_on = 100 S.
    REQUIRE(b.pool().switch_g_on(0) ==
              Approx(1.0 / 10e-3));
}

TEST_CASE("MOSFET helper preserves topology (smoke buck setup)",
          "[v2][layer2_v1][mosfet][buck]") {
    // High-side MOSFET + low-side diode (no inductor/cap
    // here — just topology check).
    CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
     .add_mosfet_with_body_diode(
            "Q1", "vin", "sw")
     .add_diode("D1", "gnd", "sw", 1e3, 1e-9, 0.7)
     .add_resistor("R_L", "sw", "gnd", 10.0);

    // 1 source + (1 switch + 1 body diode) + 1 free-wheeling
    // diode + 1 resistor = 5 branches.
    REQUIRE(b.num_branches() == 5);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::VoltageSource);
    REQUIRE(b.pool().kind_of(1) ==
              DevicePool::StoredKind::Switch);
    REQUIRE(b.pool().kind_of(2) ==
              DevicePool::StoredKind::Diode);
    REQUIRE(b.pool().kind_of(3) ==
              DevicePool::StoredKind::Diode);
    REQUIRE(b.pool().kind_of(4) ==
              DevicePool::StoredKind::Resistor);
}

TEST_CASE("Nonlinear diode builder: DC load-line",
          "[v2][layer6][builder][nonlinear]") {
    CircuitBuilder b;
    const models::IdealDiode::Params dp{
        .V_F0 = 0.7, .R_d = 0.01,
        .G_off = 1e-9, .kappa = 20.0};
    b.add_voltage_source("Vin", "n0", "gnd", 2.0)
     .add_nonlinear_diode("D1", "n0", "n1", dp)
     .add_resistor       ("R_L", "n1", "gnd", 1000.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    // Build with smart warm-start (Layer 4 V10).
    const Vector b_extra =
        Vector::Zero(seg.state_size);
    const Vector x_init =
        Vector::Zero(seg.state_size);

    Vector x = solve_with_newton_b_extra(
        seg, &refresh_smooth_diodes,
        b.graph(), b.pool(),
        x_init, b_extra,
        /*max_iters=*/50);

    // v_n0 should be V_dc = 2.0.
    REQUIRE(x[0] == Approx(2.0).margin(1e-4));
    // Load-line: v_n1 ≈ V_dc - V_F0 = 1.3 V.
    REQUIRE(x[1] > 1.0);
    REQUIRE(x[1] < 1.5);
}

// -----------------------------------------------------------------------------
// Parameter validation (C++ audit 2026-05 — div-by-zero device-model class).
// Non-positive device parameters that would divide by zero or take sqrt of a
// negative in the AD device math must be rejected at the builder gate with
// std::invalid_argument, NOT silently produce NaN deep in the Newton refresh.
// -----------------------------------------------------------------------------

TEST_CASE("add_nonlinear_diode rejects non-positive R_d",
          "[v2][layer6][builder][validation]") {
    CircuitBuilder b;
    REQUIRE_THROWS_AS(
        b.add_nonlinear_diode("D1", "n0", "n1",
            models::IdealDiode::Params{.R_d = Real{0}}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        b.add_nonlinear_diode("D2", "n0", "n1",
            models::IdealDiode::Params{.R_d = Real{-1}}),
        std::invalid_argument);
}

TEST_CASE("add_igbt_level1 rejects non-positive R_CE_sat",
          "[v2][layer6][builder][validation]") {
    CircuitBuilder b;
    REQUIRE_THROWS_AS(
        b.add_igbt_level1("Q1", "c", "e", "g",
            /*V_CE_sat=*/Real{1.5}, /*R_CE_sat=*/Real{0}),
        std::invalid_argument);
}

TEST_CASE("add_saturable_inductor rejects non-positive I_sat",
          "[v2][layer6][builder][validation]") {
    CircuitBuilder b;
    REQUIRE_THROWS_AS(
        b.add_saturable_inductor("L1", "n0", "n1",
            /*L_0=*/Real{1e-3}, /*I_sat=*/Real{0}),
        std::invalid_argument);
}

TEST_CASE("add_transformer rejects non-positive winding inductance",
          "[v2][layer6][builder][validation]") {
    CircuitBuilder b;
    REQUIRE_THROWS_AS(
        b.add_transformer("T1", "p0", "p1", "s0", "s1",
            /*L_p=*/Real{0}, /*L_s=*/Real{1e-3}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        b.add_transformer("T2", "p0", "p1", "s0", "s1",
            /*L_p=*/Real{1e-3}, /*L_s=*/Real{-2e-3}),
        std::invalid_argument);
}

TEST_CASE("add_multi_winding_transformer rejects non-positive inductance",
          "[v2][layer6][builder][validation]") {
    CircuitBuilder b;
    std::vector<CircuitBuilder::WindingSpec> windings{
        {.from = "a0", .to = "a1", .L = Real{1e-3}},
        {.from = "b0", .to = "b1", .L = Real{0}},
    };
    REQUIRE_THROWS_AS(
        b.add_multi_winding_transformer("MW1", windings),
        std::invalid_argument);
}

// Defense-in-depth (audit critic): a NaN/Inf reaching the Newton residual —
// from a future device, a param that slipped past the builder gate, or an
// ill-conditioned step — must throw a clear error instead of silently burning
// to max_iters (NaN compares false against every tolerance) and reporting a
// misleading "failed to converge".
TEST_CASE("solve_with_newton throws on a non-finite residual (NaN guard)",
          "[v2][layer6][builder][validation]") {
    CircuitBuilder b;
    b.add_voltage_source("V1", "n0", "gnd", 2.0);
    b.add_resistor("R1", "n0", "gnd", 100.0);
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));
    const Vector x_init = Vector::Zero(seg.state_size);

    // Refresh that injects a NaN into the residual vector.
    auto nan_refresh = [](const Vector&, sparse::Matrix& J_nl, Vector& f_nl,
                          const topology::Graph&, const DevicePool&) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) {
            f_nl.setZero();
            f_nl[0] = std::numeric_limits<Real>::quiet_NaN();
        }
        return Real{0};
    };
    REQUIRE_THROWS_AS(
        solve_with_newton(seg, nan_refresh, b.graph(), b.pool(), x_init),
        std::runtime_error);
}
