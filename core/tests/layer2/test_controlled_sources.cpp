// =============================================================================
// add-controlled-source-set — VCCS, CCCS, CCVS, gyrator stamps; table
// waveform; analytic DC two-ports through the real assembly.
// =============================================================================
//
// Every expected number is analytic. The transient, trbdf2, dsed and AC
// cases live in python/tests/test_controlled_sources.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/controlled_sources.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/stamp_controlled.hpp"

#include <cmath>
#include <limits>

using namespace pulsim;
using namespace pulsim::stamping;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

// Nodes 0..3 plus one branch row (4) and one sensed current column (5).
sparse::Matrix six() { return sparse::Matrix(6, 6); }

Vector dc(const builder::CircuitBuilder& b) {
    return pwl::compute_dc_op(
        b.graph(), b.pool(),
        topology::SwitchStateMask(b.graph().num_switches()), Real{0},
        Real{0});
}

Real rel(Real got, Real want) { return std::abs(got - want) / std::abs(want); }

}  // namespace

TEST_CASE("stamp_vccs: four entries, current leaves at `from`",
          "[v2][controlled][stamp]") {
    auto J = six();
    Vector f = Vector::Zero(6);
    stamp_vccs(J, f, Vector::Zero(6), BranchCoord{0, 1, 0}, 2, 3, 0.5);
    CHECK(J.coeff(0, 2) == -0.5);
    CHECK(J.coeff(0, 3) == 0.5);
    CHECK(J.coeff(1, 2) == 0.5);
    CHECK(J.coeff(1, 3) == -0.5);
    CHECK(J.nonZeros() == 4);
}

TEST_CASE("stamp_cccs / stamp_ccvs: the design table's signs",
          "[v2][controlled][stamp]") {
    auto J = six();
    Vector f = Vector::Zero(6);
    stamp_cccs(J, f, Vector::Zero(6), BranchCoord{0, 1, 0}, 5, 50.0);
    CHECK(J.coeff(0, 5) == -50.0);
    CHECK(J.coeff(1, 5) == 50.0);
    CHECK(J.nonZeros() == 2);

    auto K = six();
    stamp_ccvs(K, f, Vector::Zero(6), BranchCoord{2, 3, 0}, 4, 5, 100.0);
    CHECK(K.coeff(2, 4) == 1.0);
    CHECK(K.coeff(3, 4) == -1.0);
    CHECK(K.coeff(4, 2) == 1.0);
    CHECK(K.coeff(4, 3) == -1.0);
    CHECK(K.coeff(4, 5) == -100.0);
    CHECK(K.nonZeros() == 5);
}

TEST_CASE("stamp_gyrator: eight entries, antisymmetric", "[v2][controlled][stamp]") {
    // Port 1 = (0, 1), port 2 = (2, 3).
    auto J = six();
    Vector f = Vector::Zero(6);
    stamp_gyrator(J, f, Vector::Zero(6), BranchCoord{2, 3, 0}, 0, 1, 0.01);
    CHECK(J.coeff(0, 2) == 0.01);
    CHECK(J.coeff(0, 3) == -0.01);
    CHECK(J.coeff(1, 2) == -0.01);
    CHECK(J.coeff(1, 3) == 0.01);
    CHECK(J.coeff(2, 0) == -0.01);
    CHECK(J.coeff(2, 1) == 0.01);
    CHECK(J.coeff(3, 0) == 0.01);
    CHECK(J.coeff(3, 1) == -0.01);
    CHECK(J.nonZeros() == 8);
    // Lossless: J + Jᵀ = 0 on the port block.
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            CHECK(J.coeff(r, c) + J.coeff(c, r) == 0.0);
        }
    }
}

TEST_CASE("VCCS DC: 2 mS x 5 V into 1 kOhm reads +10 V", "[v2][controlled][dc]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vc", "c", "gnd", 5.0);
    b.add_vccs("G1", "c", "gnd", "o", "gnd", 2e-3);
    b.add_resistor("RL", "o", "gnd", 1e3);
    CHECK(rel(dc(b)[b.node_id_of("o")], 10.0) < 1e-12);
}

TEST_CASE("CCVS DC: 10 mA sensed, 100 Ohm reads 1 V", "[v2][controlled][dc]") {
    builder::CircuitBuilder b;
    b.add_current_source("I1", "a", "gnd", 10e-3);
    b.add_voltage_source("Vs", "a", "m", 0.0);  // sense: 10 mA from a to m
    b.add_resistor("R", "m", "gnd", 50.0);
    b.add_ccvs("H1", "Vs", "o", "gnd", 100.0);
    b.add_resistor("RL", "o", "gnd", 1e3);
    CHECK(rel(dc(b)[b.node_id_of("o")], 1.0) < 1e-12);
}

TEST_CASE("CCCS DC: 1 mA sensed, gain 50, 100 Ohm reads 5 V",
          "[v2][controlled][dc]") {
    builder::CircuitBuilder b;
    b.add_current_source("I1", "a", "gnd", 1e-3);
    b.add_voltage_source("Vs", "a", "m", 0.0);
    b.add_resistor("R", "m", "gnd", 50.0);
    b.add_cccs("F1", "Vs", "o", "gnd", 50.0);
    b.add_resistor("RL", "o", "gnd", 100.0);
    CHECK(rel(dc(b)[b.node_id_of("o")], 5.0) < 1e-12);
}

TEST_CASE("Gyrator DC: g = 0.01 S with 50 Ohm on port 2 is 200 Ohm on port 1",
          "[v2][controlled][dc]") {
    builder::CircuitBuilder b;
    b.add_current_source("I1", "p1", "gnd", 1.0);  // 1 A into port 1
    b.add_gyrator("Y1", "p1", "gnd", "p2", "gnd", 0.01);
    b.add_resistor("R2", "p2", "gnd", 50.0);
    const Vector x = dc(b);
    CHECK(rel(x[b.node_id_of("p1")], 200.0) < 1e-12);  // 1/(g²R)
    // Lossless, from the terminal currents: i1 = 1 A (all of I1), i2 =
    // −v2/50 (port 2 feeds R2).
    const Real v1 = x[b.node_id_of("p1")], v2 = x[b.node_id_of("p2")];
    CHECK(std::abs(v1 * 1.0 + v2 * (-v2 / 50.0)) <= 1e-12 * std::abs(v1));
}

TEST_CASE("Controlled sources refuse by name", "[v2][controlled][refusal]") {
    builder::CircuitBuilder b;
    b.add_resistor("R1", "a", "gnd", 1.0);
    CHECK_THROWS_WITH(b.add_cccs("F1", "R1", "o", "gnd", 2.0),
                      ContainsSubstring("F1") && ContainsSubstring("R1") &&
                          ContainsSubstring("inductor"));
    CHECK_THROWS_WITH(b.add_ccvs("H1", "Rx", "o", "gnd", 2.0),
                      ContainsSubstring("H1") && ContainsSubstring("Rx"));
    CHECK_THROWS_WITH(b.add_vccs("G1", "a", "gnd", "o", "gnd",
                                 std::numeric_limits<Real>::quiet_NaN()),
                      ContainsSubstring("G1") && ContainsSubstring("gm"));
    CHECK_THROWS_WITH(b.add_gyrator("Y1", "a", "gnd", "o", "gnd",
                                    std::numeric_limits<Real>::infinity()),
                      ContainsSubstring("Y1"));
}

TEST_CASE("PwlWaveform: value, hold, wrap, breakpoints", "[v2][controlled][pwl]") {
    const models::PwlWaveform w{{1.0, 2.0, 4.0}, {0.0, 1.0, 0.0}, false};
    CHECK(w.value_at(0.0) == 0.0);   // before t_0: i_0
    CHECK(w.value_at(1.5) == 0.5);
    CHECK(w.value_at(3.0) == 0.5);
    CHECK(w.value_at(9.0) == 0.0);   // after t_last: i_last
    CHECK(w.next_breakpoint_after(0.0) == 1.0);
    CHECK(w.next_breakpoint_after(2.0) == 4.0);
    CHECK(std::isinf(w.next_breakpoint_after(4.0)));

    const models::PwlWaveform p{{1.0, 2.0, 4.0}, {0.0, 1.0, 0.0}, true};
    CHECK(p.value_at(4.5) == 0.5);   // period 3: 4.5 ↦ 1.5
    CHECK(p.next_breakpoint_after(4.0) == 5.0);
    CHECK(p.next_breakpoint_after(5.5) == 7.0);
}

TEST_CASE("PwlWaveform refuses bad tables naming the source",
          "[v2][controlled][pwl][refusal]") {
    using W = models::PwlWaveform;
    const auto bad = [](W w) { W::validate(w, "I_tab"); };
    CHECK_THROWS_WITH(bad(W{{0.0}, {1.0}, false}),
                      ContainsSubstring("I_tab") && ContainsSubstring("two"));
    CHECK_THROWS_WITH(bad(W{{0.0, 2.0, 1.0}, {0.0, 1.0, 0.0}, false}),
                      ContainsSubstring("I_tab") && ContainsSubstring("increase"));
    CHECK_THROWS_WITH(bad(W{{0.0, 1.0}, {0.0, std::nan("")}, false}),
                      ContainsSubstring("I_tab") && ContainsSubstring("finite"));
    CHECK_THROWS_WITH(bad(W{{0.0, 1.0}, {0.0, 1.0}, true}),
                      ContainsSubstring("I_tab") && ContainsSubstring("periodic"));
    CHECK_THROWS_WITH(bad(W{{0.0, 1.0, 2.0}, {0.0, 1.0}, false}),
                      ContainsSubstring("I_tab") && ContainsSubstring("lengths"));
}

TEST_CASE("Pulse current source corners", "[v2][controlled][pulse]") {
    models::PulseCurrentSource::Params p{
        {.v_initial = 0.0, .v_pulsed = 1.0, .t_start = 1.0,
         .pulse_width = 2.0, .period = 10.0, .rise_time = 0.5,
         .fall_time = 0.5}};
    using P = models::PulseCurrentSource;
    CHECK(P::next_breakpoint_after(p, 0.0) == 1.0);
    CHECK(P::next_breakpoint_after(p, 1.0) == 1.5);
    CHECK(P::next_breakpoint_after(p, 1.5) == 3.5);
    CHECK(P::next_breakpoint_after(p, 3.5) == 4.0);
    CHECK(P::next_breakpoint_after(p, 4.0) == 11.0);
    p.w.period = 0.0;
    CHECK(std::isinf(P::next_breakpoint_after(p, 4.0)));
}

TEST_CASE("DC operating point uses the waveform value at t = 0",
          "[v2][controlled][dc]") {
    builder::CircuitBuilder b;
    // 2 + 3·sin(π/2) = 5 A into 2 Ω.
    b.add_sine_current_source("I1", "o", "gnd", 2.0, 3.0, 50.0,
                              std::acos(Real{-1}) / 2);
    b.add_resistor("R", "o", "gnd", 2.0);
    CHECK(rel(dc(b)[b.node_id_of("o")], 10.0) < 1e-12);
}
