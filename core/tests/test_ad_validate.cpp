// =============================================================================
// Test: AD validation layer (Phase 4 of add-automatic-differentiation)
// =============================================================================
//
// Exercises `validate_nonlinear_jacobians` against a circuit containing all
// four nonlinear device types simultaneously. The build-selected stamp
// (manual under default; AD under PULSIM_USE_AD_STAMP) must agree with
// centered finite differences on the canonical "current-out" J row of every
// device, at every operating point we hand it.
//
// Fires across a small spread of operating points so the validator
// exercises distinct regions per device (cutoff / on / triode / saturation,
// plus mixed reference voltages).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/ad/validate.hpp"

#include <vector>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

[[nodiscard]] Circuit build_mixed_nonlinear_circuit() {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    auto g = ckt.add_node("g");
    auto d = ckt.add_node("d");
    auto s = ckt.add_node("s");
    auto gi = ckt.add_node("gi");
    auto c = ckt.add_node("c");
    auto e = ckt.add_node("e");
    auto ctrl = ckt.add_node("ctrl");
    auto t1 = ckt.add_node("t1");
    auto t2 = ckt.add_node("t2");

    ckt.add_diode("D1", a, b);

    MOSFET::Params m_params{};
    m_params.vth = 2.0;
    m_params.kp = 0.1;
    m_params.lambda = 0.01;
    m_params.is_nmos = true;
    ckt.add_mosfet("M1", g, d, s, m_params);

    IGBT::Params igbt_params{};
    igbt_params.vth = 5.0;
    igbt_params.g_on = 1e4;
    igbt_params.g_off = 1e-12;
    igbt_params.v_ce_sat = 1.5;
    ckt.add_igbt("Q1", gi, c, e, igbt_params);

    ckt.add_vcswitch("S1", ctrl, t1, t2, /*v_th=*/2.5);
    return ckt;
}

[[nodiscard]] Vector make_op_point(const Circuit& ckt,
                                   Real v_a, Real v_b,
                                   Real v_g, Real v_d, Real v_s,
                                   Real v_gi, Real v_c, Real v_e,
                                   Real v_ctrl, Real v_t1, Real v_t2) {
    Vector x = Vector::Zero(ckt.system_size());
    x[ckt.get_node("a")] = v_a;
    x[ckt.get_node("b")] = v_b;
    x[ckt.get_node("g")] = v_g;
    x[ckt.get_node("d")] = v_d;
    x[ckt.get_node("s")] = v_s;
    x[ckt.get_node("gi")] = v_gi;
    x[ckt.get_node("c")] = v_c;
    x[ckt.get_node("e")] = v_e;
    x[ckt.get_node("ctrl")] = v_ctrl;
    x[ckt.get_node("t1")] = v_t1;
    x[ckt.get_node("t2")] = v_t2;
    return x;
}

}  // namespace

TEST_CASE("validate_nonlinear_jacobians: healthy circuit reports zero mismatches",
          "[ad][validate][phase4][healthy]") {
    Circuit ckt = build_mixed_nonlinear_circuit();

    // Three operating points spreading the diode (forward / reverse), MOSFET
    // (cutoff / saturation / triode), IGBT (cutoff / on), VCSwitch (off / on)
    // through their distinct regions.
    std::vector<Vector> ops;
    ops.push_back(make_op_point(ckt,
        /*v_a=*/3.0, /*v_b=*/0.0,        // diode forward
        /*v_g=*/5.0, /*v_d=*/10.0, /*v_s=*/0.0,  // MOSFET saturation
        /*v_gi=*/15.0, /*v_c=*/10.0, /*v_e=*/0.0, // IGBT on
        /*v_ctrl=*/5.0, /*v_t1=*/2.0, /*v_t2=*/0.0)); // VCSwitch closed
    ops.push_back(make_op_point(ckt,
        0.0, 3.0,
        1.0, 5.0, 0.0,
        0.0, 100.0, 0.0,
        0.0, 1.0, 0.0));
    ops.push_back(make_op_point(ckt,
        0.5, 0.0,                          // diode mid-transition
        5.0, 1.0, 0.0,                     // MOSFET triode
        15.0, 0.5, 0.0,                    // IGBT on, low Vce
        2.5, 1.0, 0.0));                   // VCSwitch at threshold

    const auto mismatches =
        ad::validate_nonlinear_jacobians(ckt, ops, /*abs_tol=*/Real{1e-5});

    if (!mismatches.empty()) {
        for (const auto& m : mismatches) {
            INFO("Mismatch: " << m.device_type << " '" << m.device_name
                 << "' op=" << m.op_point_index
                 << " J(" << m.local_row << "," << m.local_col << ")"
                 << " stamp=" << m.stamp_value
                 << " fd=" << m.fd_value
                 << " |delta|=" << m.abs_delta);
        }
    }
    CHECK(mismatches.empty());
}

TEST_CASE("validate_nonlinear_jacobians: overly-tight tolerance surfaces FD truncation",
          "[ad][validate][phase4][tolerance]") {
    // Sanity: with an unrealistic tolerance (smaller than FD truncation
    // error of ~1e-10), even a healthy stamp will get flagged. This pins
    // the contract that the validator is sensitive to its tolerance knob.
    Circuit ckt = build_mixed_nonlinear_circuit();
    std::vector<Vector> ops;
    ops.push_back(make_op_point(ckt,
        3.0, 0.0,
        5.0, 10.0, 0.0,
        15.0, 10.0, 0.0,
        5.0, 2.0, 0.0));

    const auto mismatches =
        ad::validate_nonlinear_jacobians(ckt, ops, /*abs_tol=*/Real{1e-15});

    // With a 1e-15 tolerance (well below FD truncation), at least one
    // mismatch fires somewhere. We only assert non-emptiness — the exact
    // device / row that hits FD noise first is implementation-dependent.
    CHECK_FALSE(mismatches.empty());
}
