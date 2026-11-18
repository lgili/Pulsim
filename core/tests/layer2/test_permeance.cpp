// =============================================================================
// add-permeance-magnetic-domain — the tanh permeance law, the magnetic
// builder methods, the magnetic preflight rule and the DC flux rows.
// =============================================================================
//
// Expected numbers are closed forms of the reluctance network. Transient
// cases (inductances, transformer equivalence, saturation, loss, engines)
// live in python/tests/test_permeance_magnetic.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/gapped_core.hpp"
#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/preflight.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace pulsim;
using Catch::Matchers::ContainsSubstring;
using NC = models::NonlinearCapacitor;

namespace {

// ETD29-class segment of the design (synthetic illustration).
constexpr Real kA = 76e-6, kL = 72e-3, kGap = 0.5e-3, kMuR = 2000, kBsat = 0.35;

NC::Params segment_law() {
    const Real p_air = models::MU_0 * kA / kL;
    const Real h0 = (kBsat / models::MU_0) / (kMuR - 1);
    return NC::tanh_law(kMuR * p_air, p_air, kL * h0);
}

Vector dc(const builder::CircuitBuilder& b) {
    return pwl::compute_dc_op(b.graph(), b.pool(),
                              topology::SwitchStateMask(b.graph().num_switches()));
}

Real node_v(const builder::CircuitBuilder& b, const Vector& x, const char* n) {
    return x[b.node_id_of(n)];
}

Real rel(Real got, Real want) { return std::abs(got - want) / std::abs(want); }

}  // namespace

TEST_CASE("tanh permeance law: the GappedCore material law per segment",
          "[permeance][law]") {
    const auto p = segment_law();
    NC::validate(p);
    const Real p_air = models::MU_0 * kA / kL;
    CHECK(rel(NC::capacitance(p, 0.0), kMuR * p_air) < 1e-15);
    // Φ(F) = A·B(F/l) with the GappedCore law, at a few fields.
    models::GappedCore::Params g{.N = 1, .Ae = kA, .le = kL, .lg = 0,
                                 .mu_r0 = kMuR, .B_sat = kBsat};
    for (const Real H : {-3000.0, -139.0, 1.0, 139.0, 500.0, 20000.0}) {
        CHECK(rel(NC::charge(p, H * kL), kA * models::GappedCore::B_of_H(g, H)) < 1e-13);
    }
    // Odd, and P(F) is the derivative of the evaluated Φ(F).
    for (const Real F : {1e-6, 0.3, 10.0, 25.0, 60.0, 400.0}) {
        CHECK(std::abs(NC::charge(p, -F) + NC::charge(p, F)) <=
              4 * std::numeric_limits<Real>::epsilon() * std::abs(NC::charge(p, F)));
        const Real h = 1e-6 * std::max(Real{1}, F);
        const Real fd = (NC::charge(p, F + h) - NC::charge(p, F - h)) / (2 * h);
        CHECK(rel(fd, NC::capacitance(p, F)) < 1e-7);
        CHECK(NC::capacitance(p, F) >= p_air);  // sech² underflows to 0 far out
    }
    // Saturated: P → μ₀A/l from above.
    CHECK(rel(NC::capacitance(p, 1e5), p_air) < 1e-15);
}

TEST_CASE("tanh permeance law is monotone at fine spacing", "[permeance][law]") {
    const auto p = segment_law();
    const Real knee = 2 * p.V0;  // H = 2·H₀
    // Strictly increasing at 1 pA-turn spacing around 0 and the knee.
    for (const Real c : {0.0, knee}) {
        Real prev = NC::charge(p, c - 1000e-12);
        for (int k = -999; k <= 1000; ++k) {
            const Real q = NC::charge(p, c + k * 1e-12);
            REQUIRE(q > prev);
            prev = q;
        }
    }
    // Ten times the knee: P = μ₀A/l, so a 1 pA-turn step moves Φ by
    // 1.3e-21 Wb against Φ ≈ 2.7e-5 Wb (5e-17 relative, below half an
    // ulp): strict increase is not representable there. The law never
    // decreases at that spacing, and increases strictly at 1 nA-turn.
    for (const Real step : {1e-12, 1e-9}) {
        Real prev = NC::charge(p, 10 * knee - 1000 * step);
        for (int k = -999; k <= 1000; ++k) {
            const Real q = NC::charge(p, 10 * knee + k * step);
            if (step == 1e-9) REQUIRE(q > prev);
            else REQUIRE(q >= prev);
            prev = q;
        }
    }
}

TEST_CASE("magnetic builder methods refuse bad parameters by name",
          "[permeance][builder]") {
    builder::CircuitBuilder b;
    CHECK_THROWS_WITH(b.add_winding("W1", "e", "gnd", "m", "gnd", 0.0),
                      ContainsSubstring("add_winding(\"W1\"): turns"));
    CHECK_THROWS_WITH(b.add_winding("W1", "e", "gnd", "m", "gnd", -3.0),
                      ContainsSubstring("turns"));
    CHECK_THROWS_WITH(b.add_permeance("P1", "m", "gnd", 0.0),
                      ContainsSubstring("add_permeance(\"P1\")"));
    CHECK_THROWS_WITH(b.add_air_gap("G1", "m", "gnd", 0.0, 1e-3),
                      ContainsSubstring("add_air_gap(\"G1\"): area"));
    CHECK_THROWS_WITH(b.add_air_gap("G1", "m", "gnd", 1e-4, -1.0),
                      ContainsSubstring("length"));
    CHECK_THROWS_WITH(b.add_magnetic_resistance("R1", "m", "gnd", 0.0),
                      ContainsSubstring("add_magnetic_resistance(\"R1\")"));
    CHECK_THROWS_WITH(b.add_saturable_permeance("S1", "m", "gnd", kA, kL, 1.0, kBsat),
                      ContainsSubstring("mu_r"));
    CHECK_THROWS_WITH(b.add_saturable_permeance("S1", "m", "gnd", kA, kL, kMuR, 0.0),
                      ContainsSubstring("B_sat"));
    CHECK_THROWS_WITH(b.add_permeance_law("S1", "m", "gnd", NC::junction(1e-9, 1.0)),
                      ContainsSubstring("odd law"));
}

TEST_CASE("a magnetic island without a reference is refused, not tied",
          "[permeance][preflight]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "e", "gnd", 1.0);
    b.add_winding("W", "e", "gnd", "m1", "m0", 10);
    b.add_permeance("P", "m1", "m0", 1e-6);
    CHECK_THROWS_WITH(b.run_preflight(), ContainsSubstring("magnetic network containing"));
    CHECK_THROWS_WITH(b.run_preflight(), ContainsSubstring("'m1'"));
    CHECK_THROWS_WITH(b.build(), ContainsSubstring("no node tied to the reference"));
    // Tied to the reference: nothing to report, and no tie inserted on
    // the internal node that touches only permeances.
    builder::CircuitBuilder ok;
    ok.add_voltage_source("V", "e", "gnd", 1.0);
    ok.add_winding("W", "e", "gnd", "m1", "gnd", 10);
    ok.add_permeance("Pc", "m1", "m2", 1e-6);
    ok.add_air_gap("G", "m2", "gnd", kA, kGap);
    const auto branches = ok.graph().num_branches();
    const auto report = ok.run_preflight();
    CHECK(report.empty());
    CHECK(ok.graph().num_branches() == branches);
}

TEST_CASE("DC flux rows: a gapped core under DC bias divides the MMF as the "
          "reluctances", "[permeance][dc]") {
    // 2 V through 1 Ω: 2 A in the winding at DC.
    const Real N = 25;
    const Real Pc = models::MU_0 * kMuR * kA / kL;
    const Real Pg = models::MU_0 * kA / kGap;
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 2.0);
    b.add_resistor("R", "in", "e", 1.0);
    b.add_winding("W", "e", "gnd", "m1", "gnd", N);
    b.add_permeance("Pc", "m1", "m2", Pc);
    b.add_air_gap("G", "m2", "gnd", kA, kGap);
    const Vector x = dc(b);
    const Real F = N * 2.0;
    const Real F1 = node_v(b, x, "m1");
    const Real F2 = node_v(b, x, "m2");  // across the gap
    CHECK(rel(F1, F) < 1e-9);
    // R_gap/(R_core + R_gap) = (1/Pg)/(1/Pc + 1/Pg)
    CHECK(rel(F2, F * (1 / Pg) / (1 / Pc + 1 / Pg)) < 1e-9);
    CHECK(rel(Pc * (F1 - F2), Pg * F2) < 1e-12);  // flux continuity
    // Winding voltage 0, up to the conductance floor: gmin (1e-12 S) on
    // m1 at F = 50 A-turn draws a 5e-11 Wb/s flux rate, v_e = N·5e-11.
    CHECK(std::abs(node_v(b, x, "e")) < 1e-8);
}

TEST_CASE("DC flux rows: three-leg flux division with unequal reluctances",
          "[permeance][dc]") {
    // Winding (N = 40, 0.5 A) on leg a; legs b and c in parallel from the
    // top yoke T to the bottom yoke (gnd).
    const Real N = 40, I = 0.5, Pa = 3e-6, Pb = 1e-6, Pc = 0.25e-6;
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", I * 2.0);
    b.add_resistor("R", "in", "e", 2.0);
    b.add_winding("W", "e", "gnd", "x", "gnd", N);
    b.add_permeance("Pa", "x", "T", Pa);
    b.add_permeance("Pb", "T", "gnd", Pb);
    b.add_permeance("Pc", "T", "gnd", Pc);
    const Vector x = dc(b);
    const Real F = N * I;
    // Magnetic Kirchhoff: Φ_a = F / (1/Pa + 1/(Pb + Pc)), split Pb : Pc.
    const Real phi_a = F / (1 / Pa + 1 / (Pb + Pc));
    const Real FT = node_v(b, x, "T");
    CHECK(rel(Pb * FT, phi_a * Pb / (Pb + Pc)) < 1e-9);
    CHECK(rel(Pc * FT, phi_a * Pc / (Pb + Pc)) < 1e-9);
    CHECK(rel(Pa * (node_v(b, x, "x") - FT), phi_a) < 1e-9);
}

TEST_CASE("DC flux rows: nodes joined only by a magnetic resistance share "
          "one flux row", "[permeance][dc]") {
    // W on m1, P1 m1 -> m2, R_m m2 -> m3, P2 m3 -> gnd. At DC R_m carries
    // no flux rate, so F2 = F3, and the flux of P1 equals that of P2.
    const Real N = 25, P1 = 2e-6, P2 = 0.2e-6;
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 2.0);
    b.add_resistor("R", "in", "e", 1.0);
    b.add_winding("W", "e", "gnd", "m1", "gnd", N);
    b.add_permeance("P1", "m1", "m2", P1);
    b.add_magnetic_resistance("Rm", "m2", "m3", 0.5);
    b.add_permeance("P2", "m3", "gnd", P2);
    const auto rows = pwl::charge_rows(b.graph(), b.pool());
    const auto m2 = b.node_id_of("m2"), m3 = b.node_id_of("m3");
    CHECK(rows[static_cast<Size>(m2)] == std::min(m2, m3));
    CHECK(rows[static_cast<Size>(m3)] == std::min(m2, m3));
    CHECK(rows[static_cast<Size>(b.node_id_of("m1"))] == -1);
    const Vector x = dc(b);
    const Real F3 = N * 2.0 * P1 / (P1 + P2);
    CHECK(rel(node_v(b, x, "m3"), F3) < 1e-9);
    CHECK(rel(node_v(b, x, "m2"), F3) < 1e-9);
    // F1 − F2 = 4.5 of F1 = 50: the floor's 1e-10-relative error on m1
    // (anchored by the winding) is amplified 11x in the difference.
    CHECK(rel(P1 * (node_v(b, x, "m1") - node_v(b, x, "m2")), P2 * node_v(b, x, "m3")) < 1e-8);
    // A magnetic resistance to the reference anchors the node instead:
    // at DC it shorts the MMF across it (no flux row needed).
    builder::CircuitBuilder a;
    a.add_voltage_source("V", "in", "gnd", 2.0);
    a.add_resistor("R", "in", "e", 1.0);
    a.add_winding("W", "e", "gnd", "m1", "gnd", N);
    a.add_permeance("P1", "m1", "m2", P1);
    a.add_magnetic_resistance("Rm", "m2", "gnd", 0.5);
    CHECK(pwl::charge_rows(a.graph(), a.pool())[static_cast<Size>(a.node_id_of("m2"))] == -1);
    CHECK(std::abs(node_v(a, dc(a), "m2")) < 1e-9);
}

TEST_CASE("DC flux rows: a winding whose magnetic port floats joins its "
          "nodes' flux row", "[permeance][dc]") {
    // W between a and c (neither on gnd), Pa a -> gnd, Pc c -> gnd: the
    // winding fixes F_a − F_c = N·i and carries no flux rate at DC, so the
    // flux through Pa returns through Pc: Pa·F_a + Pc·F_c = 0.
    const Real N = 10, Pa = 1e-6, Pc = 3e-6;
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 2.0);
    b.add_resistor("R", "in", "e", 1.0);
    b.add_winding("W", "e", "gnd", "a", "c", N);
    b.add_permeance("Pa", "a", "gnd", Pa);
    b.add_permeance("Pc", "c", "gnd", Pc);
    const Vector x = dc(b);
    const Real F = N * 2.0;
    CHECK(rel(node_v(b, x, "a"), F * Pc / (Pa + Pc)) < 1e-9);   // 15
    CHECK(rel(node_v(b, x, "c"), -F * Pa / (Pa + Pc)) < 1e-9);  // −5
    // A saturable permeance in that loop is refused by name, not floored.
    builder::CircuitBuilder s;
    s.add_voltage_source("V", "in", "gnd", 2.0);
    s.add_resistor("R", "in", "e", 1.0);
    s.add_winding("W", "e", "gnd", "a", "c", N);
    s.add_saturable_permeance("Core", "a", "gnd", kA, kL, kMuR, kBsat);
    s.add_permeance("Pc", "c", "gnd", Pc);
    CHECK_THROWS_WITH(dc(s), ContainsSubstring("'Core'"));
}

TEST_CASE("DC flux rows leave a circuit without magnetic nodes unchanged",
          "[permeance][dc]") {
    // Two series capacitors: the middle node is singular at DC, as before.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 1.0);
    b.add_capacitor("C1", "in", "mid", 1e-6);
    b.add_capacitor("C2", "mid", "gnd", 2e-6);
    CHECK(pwl::charge_rows(b.graph(), b.pool()).empty());
    CHECK_THROWS(dc(b));
}

TEST_CASE("DC with a saturable permeance on a node without winding is refused "
          "by name", "[permeance][dc]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 2.0);
    b.add_resistor("R", "in", "e", 1.0);
    b.add_winding("W", "e", "gnd", "m1", "gnd", 25);
    b.add_saturable_permeance("Core", "m1", "m2", kA, kL, kMuR, kBsat);
    b.add_air_gap("G", "m2", "gnd", kA, kGap);
    CHECK_THROWS_WITH(dc(b), ContainsSubstring("'Core'"));
    // Directly across the winding (no internal node) it is open at DC,
    // like any charge device, and the operating point exists.
    builder::CircuitBuilder ok;
    ok.add_voltage_source("V", "in", "gnd", 2.0);
    ok.add_resistor("R", "in", "e", 1.0);
    ok.add_winding("W", "e", "gnd", "m1", "gnd", 25);
    ok.add_saturable_permeance("Core", "m1", "gnd", kA, kL, kMuR, kBsat);
    const Vector x = dc(ok);
    CHECK(rel(node_v(ok, x, "m1"), 50.0) < 1e-9);
}
