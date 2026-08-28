// =============================================================================
// Layer 4 V3 — the algebraic recovery map of the LTI extraction
// =============================================================================
//
// v2.0 Phase 3. The reduction eliminates every non-state unknown —
// node voltages, source currents — and until now they were simply
// gone: `result.v('sw_node')` was unrecoverable from a DSED run, and
// a diode event predicate had nothing to read. `ContinuousLTI` now
// exports the map the elimination was already solving for:
//
//   x_full = recover_from_state·x_s + recover_const
//          + recover_from_b·b_extra
//
// THE INVARIANT THESE TESTS PIN. At a steady state of the reduced
// system (A·x_s + b = 0), ẋ = 0 kills every C·dv/dt and L·di/dt
// term, so the reconstructed full vector must satisfy the ORIGINAL
// full-MNA equations exactly: G·x_full + b + b_extra = 0, every row.
// That single check exercises the scatter, the Schur pieces, and the
// floating-cap congruence transform T in original coordinates — a
// sign or coordinate slip anywhere shows up as a KCL violation.
//
// A second, independent cross-check: with no extra sources, the
// steady state of the LTI (caps open, inductors short) is the DC
// operating point, so x_full* must equal compute_dc_op's answer.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/dc_assemble.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using Catch::Approx;

namespace {

/// ‖G·x_full + b + b_extra‖_inf on the ORIGINAL system.
Real full_mna_residual(const topology::Graph& g, const DevicePool& p,
                        const topology::SwitchStateMask& m,
                        const Vector& x_full, const Vector& b_extra) {
    sparse::Matrix G, C;
    Vector b;
    assemble_segment_split(g, p, m, G, C, b);
    Vector r = G * x_full + b + b_extra;
    return r.size() == 0 ? Real{0} : r.cwiseAbs().maxCoeff();
}

/// Steady state of the reduced system, x_s* = -A⁻¹(b + B·b_extra).
Vector lti_steady_state(const PwlStateSpaceCache::ContinuousLTI& lti,
                          const Vector& b_extra_mna) {
    Vector rhs = lti.b_constant;
    if (b_extra_mna.size() > 0) {
        rhs += lti.b_projection * b_extra_mna;
    }
    return lti.A.fullPivLu().solve(-rhs);
}

Vector reconstruct(const PwlStateSpaceCache::ContinuousLTI& lti,
                    const Vector& x_s, const Vector& b_extra_mna) {
    Vector x = lti.recover_from_state * x_s + lti.recover_const;
    if (b_extra_mna.size() > 0) {
        x += lti.recover_from_b * b_extra_mna;
    }
    return x;
}

}  // namespace

TEST_CASE("Recovery: grounded RC satisfies the full MNA at rest",
          "[v2][layer4][lti][recovery]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "n1", 1e3);
    b.add_capacitor("C1", "n1", "gnd", 1e-6);
    b.add_resistor("R2", "n1", "gnd", 2e3);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    topology::SwitchStateMask m(0);
    const auto lti = cache.compute_lti_state_space(m);
    REQUIRE(lti.A.rows() == 1);                    // one cap state
    const Index n_mna =
        static_cast<Index>(lti.recover_from_state.rows());
    REQUIRE(n_mna == static_cast<Index>(
        b.pool().state_size(b.graph())));

    const Vector none;
    const Vector x_s = lti_steady_state(lti, none);
    const Vector x_full = reconstruct(lti, x_s, none);

    // The reconstruction satisfies the ORIGINAL equations…
    REQUIRE(full_mna_residual(b.graph(), b.pool(), m, x_full,
                                Vector::Zero(n_mna)) < 1e-9);
    // …and matches the DC operating point, computed independently.
    const Vector x_dc = compute_dc_op(b.graph(), b.pool(), m,
                                        Real{0}, /*gmin=*/Real{0});
    for (Index i = 0; i < n_mna; ++i) {
        INFO("row " << i);
        REQUIRE(x_full[i] == Approx(x_dc[i]).margin(1e-9));
    }
    // Divider sanity: v_n1 = 12·2k/3k = 8 V. Node order: vin=0, n1=1.
    REQUIRE(x_full[1] == Approx(8.0).margin(1e-9));
}

TEST_CASE("Recovery: a FLOATING cap exercises the congruence T",
          "[v2][layer4][lti][recovery]") {
    // Cf sits between two live nodes, so the extractor's coordinate
    // change T is not the identity — the case where a sign slip in
    // folding T into the recovery map would go unnoticed by every
    // grounded-cap test.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 10.0);
    b.add_resistor("R1", "vin", "a", 1e3);
    b.add_resistor("R3", "a", "gnd", 1e3);
    b.add_capacitor("Cf", "a", "bnode", 1e-6);     // floating
    b.add_resistor("R2", "bnode", "gnd", 2e3);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    topology::SwitchStateMask m(0);
    const auto lti = cache.compute_lti_state_space(m);
    const Index n_mna =
        static_cast<Index>(lti.recover_from_state.rows());

    const Vector none;
    const Vector x_s = lti_steady_state(lti, none);
    const Vector x_full = reconstruct(lti, x_s, none);

    REQUIRE(full_mna_residual(b.graph(), b.pool(), m, x_full,
                                Vector::Zero(n_mna)) < 1e-9);
    const Vector x_dc = compute_dc_op(b.graph(), b.pool(), m,
                                        Real{0}, /*gmin=*/Real{0});
    for (Index i = 0; i < n_mna; ++i) {
        INFO("row " << i);
        REQUIRE(x_full[i] == Approx(x_dc[i]).margin(1e-9));
    }
    // At rest the cap is open: v_a = 5 V (R1/R3 divider), v_b = 0.
    REQUIRE(x_full[1] == Approx(5.0).margin(1e-9));   // node 'a'
    REQUIRE(x_full[2] == Approx(0.0).margin(1e-9));   // node 'bnode'
}

TEST_CASE("Recovery: inductor branch current is reconstructed too",
          "[v2][layer4][lti][recovery]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 9.0);
    b.add_resistor("R1", "vin", "n1", 100.0);
    b.add_inductor("L1", "n1", "n2", 1e-3);
    b.add_resistor("R2", "n2", "gnd", 200.0);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    topology::SwitchStateMask m(0);
    const auto lti = cache.compute_lti_state_space(m);
    const Index n_mna =
        static_cast<Index>(lti.recover_from_state.rows());

    const Vector none;
    const Vector x_full =
        reconstruct(lti, lti_steady_state(lti, none), none);
    REQUIRE(full_mna_residual(b.graph(), b.pool(), m, x_full,
                                Vector::Zero(n_mna)) < 1e-9);
    // i_L = 9/(100+200) = 30 mA; its MNA row is the state's own row.
    REQUIRE(x_full[lti.state_row_indices[0]]
             == Approx(0.03).margin(1e-12));
    // And the interior node the reduction had eliminated:
    // v_n2 = i·R2 = 6 V (node order vin=0, n1=1, n2=2).
    REQUIRE(x_full[2] == Approx(6.0).margin(1e-9));
}

TEST_CASE("Recovery: the b_extra path carries an injected source",
          "[v2][layer4][lti][recovery]") {
    // recover_from_b is what lets time-varying sources (sine/PWM
    // overlays, and later event predicates under them) reconstruct
    // correctly. Inject a constant extra current at an ALGEBRAIC
    // node and require the shifted rest state to satisfy the shifted
    // equations exactly.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "n1", 1e3);
    b.add_resistor("Rmid", "n1", "n2", 1e3);       // n1 is algebraic
    b.add_capacitor("C1", "n2", "gnd", 1e-6);
    b.add_resistor("R2", "n2", "gnd", 2e3);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    topology::SwitchStateMask m(0);
    const auto lti = cache.compute_lti_state_space(m);
    const Index n_mna =
        static_cast<Index>(lti.recover_from_state.rows());

    Vector b_extra = Vector::Zero(n_mna);
    b_extra[1] = Real{5e-3};                        // 5 mA into n1

    const Vector x_s = lti_steady_state(lti, b_extra);
    const Vector x_full = reconstruct(lti, x_s, b_extra);
    REQUIRE(full_mna_residual(b.graph(), b.pool(), m, x_full,
                                b_extra) < 1e-9);

    // And without the injection the answer is different — the term
    // is load-bearing, not vacuously zero.
    const Vector none;
    const Vector x0 =
        reconstruct(lti, lti_steady_state(lti, none), none);
    REQUIRE(std::abs(x_full[1] - x0[1]) > 1e-3);
}

TEST_CASE("Recovery: with no algebraic block the map is the scatter",
          "[v2][layer4][lti][recovery]") {
    // Current source + cap + resistor all on one node: n_alg = 0,
    // the branch nothing was eliminated from. The map must be the
    // plain scatter, not garbage from an empty Schur.
    builder::CircuitBuilder b;
    // NOTE the terminal order: (n1, gnd) drives +I INTO n1 —
    // the (gnd, n1) order pulls it OUT (v would be −1 V).
    b.add_current_source("I1", "n1", "gnd", 1e-3);
    b.add_capacitor("C1", "n1", "gnd", 1e-6);
    b.add_resistor("R1", "n1", "gnd", 1e3);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    topology::SwitchStateMask m(0);
    const auto lti = cache.compute_lti_state_space(m);
    REQUIRE(lti.recover_from_state.rows() == 1);
    REQUIRE(lti.recover_from_state(0, 0) == Approx(1.0));
    REQUIRE(lti.recover_const.cwiseAbs().maxCoeff() == Approx(0.0));

    const Vector none;
    const Vector x_full =
        reconstruct(lti, lti_steady_state(lti, none), none);
    // v = I·R = 1 V.
    REQUIRE(x_full[0] == Approx(1.0).margin(1e-9));
}
