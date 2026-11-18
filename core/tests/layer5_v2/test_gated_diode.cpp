// =============================================================================
// add-pwl-igbt-and-thyristor — the gated PWL diode law (IGBT / GTO / SCR)
// =============================================================================
//
// One law, three modes (design.md, "One law"): every (mode, c, g) row at
// V_f ± ε and I_H ± ε; the ungated diode is untouched; reverse_blocking =
// False with V_ce0 = 0 is the plain switch; the stamp's J does not depend
// on the mode and b gains ∓g_on·V_f only while conducting.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/dc_operating_point.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/yaml/emitter.hpp"

#include <stdexcept>

using namespace pulsim;
using SD = models::SwitchedDiode;
using Catch::Approx;

namespace {

SD::Params gp(SD::Gate gate, bool g, Real I_H = 0.0) {
    SD::Params p{.g_on = 100.0, .g_off = 1e-9, .V_th = 0.0, .V_f = 0.9};
    p.gate = gate;
    p.g = g;
    p.I_H = I_H;
    return p;
}

constexpr Real eps = 1e-6;

}  // namespace

TEST_CASE("Gated law: gate-held (IGBT / GTO) rows", "[gated][law]") {
    const auto on_g = gp(SD::Gate::Held, true);
    const auto off_g = gp(SD::Gate::Held, false);
    // OFF, gate on: turns on at v >= V_on = V_f.
    CHECK(SD::decide(false, 0.9 + eps, on_g));
    CHECK_FALSE(SD::decide(false, 0.9 - eps, on_g));
    // ON, gate on: stays while i = g_on (v - V_f) > 0.
    CHECK(SD::decide(true, 0.9 + eps, on_g));
    CHECK_FALSE(SD::decide(true, 0.9 - eps, on_g));
    // Gate off: OFF whatever the bias or the state.
    CHECK_FALSE(SD::decide(false, 100.0, off_g));
    CHECK_FALSE(SD::decide(true, 100.0, off_g));
    // Crossing signal: diode's while the gate is on; none (−1) with it off.
    CHECK(SD::crossing_signal(true, 1.0, on_g) == Approx(0.1));
    CHECK(SD::crossing_signal(false, 1.0, on_g) == Approx(0.1));
    CHECK(SD::crossing_signal(false, 100.0, off_g) == -1.0);
}

TEST_CASE("Gated law: latching (thyristor) rows", "[gated][law]") {
    const Real IH = 0.05;   // = g_on * 5e-4 V above V_f
    const auto on_g = gp(SD::Gate::Latching, true, IH);
    const auto off_g = gp(SD::Gate::Latching, false, IH);
    // OFF: fires only with the gate and forward bias.
    CHECK(SD::decide(false, 0.9 + eps, on_g));
    CHECK_FALSE(SD::decide(false, 0.9 - eps, on_g));
    CHECK_FALSE(SD::decide(false, 100.0, off_g));
    // ON, gate on: the diode rule (down to i = 0, not I_H).
    CHECK(SD::decide(true, 0.9 + eps, on_g));
    CHECK_FALSE(SD::decide(true, 0.9 - eps, on_g));
    // ON, gate off: latched while i > I_H.
    CHECK(SD::decide(true, 0.9 + 5e-4 + eps, off_g));
    CHECK_FALSE(SD::decide(true, 0.9 + 5e-4 - eps, off_g));
    // Crossing at i = I_H once the gate is off: v - V_f - I_H/g_on.
    CHECK(SD::crossing_signal(true, 0.9 + 5e-4, off_g) == Approx(0.0).margin(1e-15));
    CHECK(SD::crossing_signal(false, 100.0, off_g) == -1.0);
}

TEST_CASE("Gated law: ungated diode unchanged", "[gated][law]") {
    SD::Params d{.g_on = 100.0, .g_off = 1e-9, .V_th = 0.0, .V_f = 0.9};
    d.g = false;   // ignored without a gate mode
    CHECK(SD::decide(false, 0.9, d));
    CHECK(SD::decide(true, 0.9 + eps, d));
    CHECK_FALSE(SD::decide(true, 0.9 - eps, d));
}

TEST_CASE("Gated devices: builder validation names the device", "[gated][builder]") {
    builder::CircuitBuilder b;
    CHECK_THROWS_WITH(b.add_igbt("Q1", "c", "e", 10e-3, 1e9, 0.9, false),
                      Catch::Matchers::ContainsSubstring("Q1") &&
                      Catch::Matchers::ContainsSubstring("reverse_blocking"));
    CHECK_THROWS_WITH(b.add_igbt("Q2", "c", "e", 10e-3, 1e9, -0.1, true),
                      Catch::Matchers::ContainsSubstring("Q2"));
    CHECK_THROWS_WITH(b.add_thyristor("T1", "a", "k", 1e-3, 1e9, 0.0, -1.0),
                      Catch::Matchers::ContainsSubstring("T1") &&
                      Catch::Matchers::ContainsSubstring("I_H"));
    CHECK_THROWS_WITH(b.add_gto("G1", "a", "k", 1e-3, 1e9, 0.0, -0.5),
                      Catch::Matchers::ContainsSubstring("G1") &&
                      Catch::Matchers::ContainsSubstring("V_th"));
    b.add_triac("X", "a", "k");
    const auto idx = b.switch_indices_of("X");
    REQUIRE(idx.size() == 2);
    CHECK(idx[0] == b.switch_index_of("X_fwd"));
    CHECK(idx[1] == b.switch_index_of("X_rev"));
}

TEST_CASE("Gated stamp: J independent of the mode, b carries V_f only while ON",
          "[gated][stamp]") {
    // Same circuit with a plain conducting diode of drop V_f and a gated
    // IGBT of knee V_ce0 = V_f: identical J and b for each mask.
    auto build = [](bool gated) {
        builder::CircuitBuilder b;
        b.add_voltage_source("V", "c", "gnd", 10.0);
        if (gated) {
            b.add_igbt("Q", "c", "e", 1e-2, 1e9, 0.9, true);
        } else {
            b.add_diode("Q", "c", "e", 100.0, 1e-9, 0.0, 0.9);
        }
        b.add_resistor("R", "e", "gnd", 1.0);
        return b;
    };
    auto bg = build(true), bd = build(false);
    for (bool c : {false, true}) {
        topology::SwitchStateMask m(1);
        m.set(0, c);
        sparse::Matrix Gg, Cg, Gd, Cd;
        Vector bvg, bvd;
        pwl::assemble_segment_split(bg.graph(), bg.pool(), m, Gg, Cg, bvg);
        pwl::assemble_segment_split(bd.graph(), bd.pool(), m, Gd, Cd, bvd);
        CHECK(Eigen::MatrixXd(Gg) == Eigen::MatrixXd(Gd));
        CHECK(Eigen::MatrixXd(Cg) == Eigen::MatrixXd(Cd));
        CHECK(bvg == bvd);
    }
}

TEST_CASE("Gated IGBT: DC operating point shows the knee", "[gated][dc]") {
    // 10 A forced collector -> emitter, V_ce0 = 0.9 V, R_on = 10 mOhm.
    builder::CircuitBuilder b;
    b.add_current_source("I", "c", "gnd", 10.0);  // 10 A out of c, through Q
    b.add_igbt("Q", "c", "gnd", 10e-3, 1e9, 0.9, true);
    pwl::DiodeEventState diodes(b.graph(), b.pool());
    topology::SwitchStateMask gate(1);
    gate.set(0, true);
    const auto dc = pwl::compute_dc_operating_point(
        b.graph(), b.pool(), gate, {}, {}, &diodes);
    const Index c = b.node_id_of("c");
    CHECK(dc.x[c] == Approx(1.0).epsilon(1e-9));
    CHECK(dc.mask.get(0));
    // Gate off: blocks (10 A into 1 GOhm would be 1e10 V; the control is
    // that the stamped bit is OFF).
    topology::SwitchStateMask off(1);
    builder::CircuitBuilder b2;
    b2.add_voltage_source("V", "c", "gnd", 100.0);
    b2.add_igbt("Q", "c", "gnd", 10e-3, 1e9, 0.9, true);
    pwl::DiodeEventState d3(b2.graph(), b2.pool());
    const auto dc2 = pwl::compute_dc_operating_point(
        b2.graph(), b2.pool(), off, {}, {}, &d3);
    CHECK_FALSE(dc2.mask.get(0));
    // Control: the same circuit with the gate on conducts.
    pwl::DiodeEventState d4(b2.graph(), b2.pool());
    const auto dc3 = pwl::compute_dc_operating_point(
        b2.graph(), b2.pool(), gate, {}, {}, &d4);
    CHECK(dc3.mask.get(0));
}

TEST_CASE("Gated devices: dsed and bdf1 refuse by name", "[gated][caps]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "a", "gnd", 10.0);
    b.add_thyristor("T1", "a", "k");
    b.add_resistor("R", "k", "gnd", 10.0);
    try {
        require_supported(b.graph(), b.pool(), pwl::Engine::Dsed);
        FAIL("dsed accepted a thyristor");
    } catch (const UnsupportedError& e) {
        CHECK(e.engine() == "dsed");
        CHECK(e.feature() == "gated_diode");
        CHECK(e.device() == "T1");
    }
    CHECK_NOTHROW(require_supported(b.graph(), b.pool(), pwl::Engine::FixedTrap));
    CHECK_NOTHROW(require_supported(b.graph(), b.pool(), pwl::Engine::TrBdf2));
}

TEST_CASE("Gated devices: the circuit YAML (and pickle) keeps the gate", "[gated][yaml]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "a", "gnd", 10.0);
    b.add_thyristor("T", "a", "k", 1e-3, 1e9, 0.7, 0.05);
    b.add_gto("G", "a", "k2");
    b.add_diode("D", "a", "k3", 1e3, 1e-9, 0.0, 0.7);
    b.add_resistor("R", "k", "gnd", 10.0);
    b.add_resistor("R2", "k2", "gnd", 10.0);
    b.add_resistor("R3", "k3", "gnd", 10.0);
    const auto c = b.build();
    const auto y = ir::to_yaml(c);
    const auto c2 = ir::from_yaml(y);
    CHECK(c2.structure_hash() == c.structure_hash());
    CHECK(c2.value_hash() == c.value_hash());
    CHECK(ir::to_yaml(c2) == y);
    CHECK(c2.pool().num_gated_diodes() == 2);
    CHECK(c2.pool().diode_params(1).gate == SD::Gate::Latching);
    CHECK(c2.pool().diode_params(1).I_H == 0.05);
    CHECK(c2.pool().diode_params(2).gate == SD::Gate::Held);
    CHECK(c2.pool().diode_params(3).gate == SD::Gate::None);
}
