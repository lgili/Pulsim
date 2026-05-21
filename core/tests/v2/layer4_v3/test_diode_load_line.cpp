// =============================================================================
// Layer 4 V3 — DC diode load-line test (the Newton "killer test")
// =============================================================================
//
// V_dc(2V) ──[smooth-blend IdealDiode]── n1 ──[R=1kΩ]── GND
//
// The DC equation at node 1 is:
//   I = G_R · v_n1  (R takes current to GND)
// AND
//   I = i_diode(v_n0 − v_n1) = sigmoid model with V_F0=0.7, R_d=0.01
//
// Solve: v_diode = v_n0 − v_n1 = V_dc − v_n1. At DC:
//   I = G_R · v_n1
//   I = ((v_diode − V_F0) / R_d) · sigmoid(v_diode)
//
// For V_dc = 2 V the diode is firmly ON: v_diode ≈ 0.7 V (small
// drop), v_n1 ≈ 1.3 V, I ≈ 1.3 mA.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("DC diode load-line converges to expected operating point",
          "[v2][layer4_v3][newton][diode]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);     // V_dc
    g.add_branch(0, 1,          BranchKind::Nonlinear);  // diode
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear); // R

    const Real V_dc = 2.0;
    const Real R    = 1000.0;
    const Real G    = 1.0 / R;
    const models::IdealDiode::Params dp{
        .V_F0  = 0.7, .R_d   = 0.01,
        .G_off = 1e-9, .kappa = 20.0};

    DevicePool pool;
    pool.add_voltage_source(0, {.V = V_dc});
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = G});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    // Newton from x_init = 0.
    Vector x_init = Vector::Zero(seg.state_size);
    Vector x = solve_with_newton(
        seg, &refresh_smooth_diodes, g, pool, x_init,
        /*max_iters=*/50, /*tol_dx=*/1e-9, /*tol_res=*/1e-8);

    const Real v_n0 = x[0];
    const Real v_n1 = x[1];
    const Real v_diode = v_n0 - v_n1;
    const Real I = G * v_n1;   // current through R

    INFO("Diode load-line:");
    INFO("  v_n0 = " << v_n0);
    INFO("  v_n1 = " << v_n1);
    INFO("  v_diode = " << v_diode);
    INFO("  I = " << I << " A");

    // Source enforces v_n0 = V_dc.
    REQUIRE(v_n0 == Approx(V_dc).margin(1e-6));

    // Diode in firm conduction → v_diode ≈ V_F0 + I·R_d.
    // With V_F0=0.7, R_d=0.01, R=1000: solving iteratively,
    // I ≈ (V_dc − V_F0) / (R + R_d) = 1.3 / 1000.01 ≈ 1.3 mA.
    const Real I_expected = (V_dc - dp.V_F0) / (R + dp.R_d);
    REQUIRE(I == Approx(I_expected).epsilon(0.05));

    // Sanity: v_diode is in the "linear past V_F0" region.
    REQUIRE(v_diode > dp.V_F0);
    REQUIRE(v_diode < dp.V_F0 + 0.1);  // not crazy high
}

TEST_CASE("Reverse-biased diode: source constraint holds + bounded v_n1",
          "[v2][layer4_v3][newton][diode]") {
    // Reverse bias: V_dc = -2V (n0 below GND). The diode is
    // firmly OFF. The exact v_n1 depends on the sigmoid-blend's
    // tail behaviour at large negative v_diode — we just check
    // the source constraint holds and v_n1 is bounded (no
    // numerical blow-up).
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Nonlinear);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = -2.0});
    pool.add_nonlinear_diode(1, {
        .V_F0 = 0.7, .R_d = 0.01,
        .G_off = 1e-9, .kappa = 20.0});
    pool.add_resistor(2, {.G = 1e-3});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = solve_with_newton(
        seg, &refresh_smooth_diodes, g, pool,
        Vector::Zero(seg.state_size),
        /*max_iters=*/50);

    REQUIRE(x[0] == Approx(-2.0).margin(1e-6));
    INFO("Reverse-bias: v_n0=" << x[0] << "  v_n1=" << x[1]);
    // v_n1 must be bounded by |V_dc| (no anomalous amplification).
    REQUIRE(std::abs(x[1]) <= std::abs(x[0]) * 2);
}
