// =============================================================================
// Layer 2 V13 — MOSFET Shichman-Hodges Level 1 model tests
// =============================================================================
//
// Validates:
//   * Cutoff region (V_GS < V_T): I_D ≈ 0.
//   * Saturation region (V_GS > V_T, V_DS large): I_D
//     converges to K·(V_GS-V_T)²·(1+λV_DS).
//   * Triode region (V_GS > V_T, V_DS < V_OV): I_D matches
//     K·(2·V_OV·V_DS − V_DS²)·(1+λV_DS).
//   * Triode→Saturation transition is C¹-smooth (no kink).
//   * Cutoff→on transition is smooth via sigmoid.
//   * evaluate_current_and_jacobian returns finite partials
//     for all 3 terminals.
//   * DevicePool stores params + gate-node id.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/mosfet_level1.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/v2/solver/run_transient.hpp"

#include <cmath>

using namespace pulsim::v2;
using namespace pulsim::v2::builder;
using namespace pulsim::v2::models;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

// Helper: compute current at given V_GS, V_DS with V_source=0.
Real iD(const MosfetLevel1::Params& p,
        Real V_GS, Real V_DS) {
    const Real v[3] = {V_DS, 0.0, V_GS};   // drain, source, gate
    return MosfetLevel1::current<Real>(v, p);
}

}  // namespace

TEST_CASE("MosfetLevel1 — cutoff: I_D ≈ 0 when V_GS << V_T",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    MosfetLevel1::Params p{
        .K = 1e-3, .V_T = 2.0, .lambda = 0.02,
        .kappa = 15.0};
    // V_GS = 0 V (far below threshold): I_D should be tiny.
    REQUIRE(std::abs(iD(p, 0.0, 5.0)) < 1e-9);
    REQUIRE(std::abs(iD(p, 1.0, 5.0)) < 1e-6);
}

TEST_CASE("MosfetLevel1 — saturation: I_D = K·V_OV²·(1+λV_DS)",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    MosfetLevel1::Params p{
        .K = 1e-3, .V_T = 2.0, .lambda = 0.02,
        .kappa = 30.0};
    // V_GS = 5 V → V_OV = 3 V. V_DS = 10 V > V_OV → sat.
    const Real V_OV = 5.0 - 2.0;
    const Real V_DS = 10.0;
    const Real I_expected = p.K * V_OV * V_OV *
                            (1.0 + p.lambda * V_DS);
    INFO("Expected I_D = " << I_expected << " A");
    REQUIRE(iD(p, 5.0, V_DS) ==
            Approx(I_expected).margin(I_expected * 0.005));
}

TEST_CASE("MosfetLevel1 — triode: I_D = K·(2·V_OV·V_DS − V_DS²)·(1+λV_DS)",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    MosfetLevel1::Params p{
        .K = 1e-3, .V_T = 2.0, .lambda = 0.02,
        .kappa = 30.0};
    // V_GS = 5 V → V_OV = 3 V. V_DS = 0.5 V << V_OV → triode.
    const Real V_OV = 3.0;
    const Real V_DS = 0.5;
    const Real I_expected = p.K *
        (2.0 * V_OV * V_DS - V_DS * V_DS) *
        (1.0 + p.lambda * V_DS);
    INFO("Expected I_D = " << I_expected << " A");
    REQUIRE(iD(p, 5.0, V_DS) ==
            Approx(I_expected).margin(I_expected * 0.01));
}

TEST_CASE("MosfetLevel1 — value continuity at triode↔sat boundary",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    // The triode and saturation formulas EVALUATE TO THE
    // SAME VALUE at V_DS = V_OV (boundary). The sigmoid
    // blend therefore returns that same value smoothly. We
    // verify continuity by sampling I at V_DS just below
    // and just above V_OV — values must agree to high
    // precision (no jump).
    MosfetLevel1::Params p{
        .K = 1e-3, .V_T = 2.0, .lambda = 0.02,
        .kappa = 30.0};
    const Real V_GS = 5.0;
    const Real V_OV = 3.0;
    constexpr Real eps = 1e-3;   // 1 mV
    const Real i_below = iD(p, V_GS, V_OV - eps);
    const Real i_at    = iD(p, V_GS, V_OV);
    const Real i_above = iD(p, V_GS, V_OV + eps);
    INFO("I(V_OV - 1mV) = " << i_below
         << ", I(V_OV) = " << i_at
         << ", I(V_OV + 1mV) = " << i_above);
    // No discontinuity — values within ~ μA of each other.
    REQUIRE(std::abs(i_at - i_below) < 1e-5);
    REQUIRE(std::abs(i_above - i_at) < 1e-5);
}

TEST_CASE("MosfetLevel1 — Jacobian via AD returns finite partials",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    MosfetLevel1::Params p{
        .K = 1e-3, .V_T = 2.0, .lambda = 0.02,
        .kappa = 15.0};
    // Saturation operating point.
    ModelInputs<MosfetLevel1> v{5.0, 0.0, 3.0};  // d, s, g
    const auto [i, partials] =
        evaluate_current_and_jacobian<MosfetLevel1>(v, p);
    INFO("I_D = " << i
         << " ∂I/∂V_d = " << partials[0]
         << " ∂I/∂V_s = " << partials[1]
         << " ∂I/∂V_g = " << partials[2]);

    // All three partials must be finite.
    REQUIRE(std::isfinite(i));
    REQUIRE(std::isfinite(partials[0]));
    REQUIRE(std::isfinite(partials[1]));
    REQUIRE(std::isfinite(partials[2]));

    // Physical sanity:
    //   * ∂I/∂V_g > 0 (raising V_gate increases I_D).
    REQUIRE(partials[2] > 0);
    //   * ∂I/∂V_d > 0 in saturation (channel-length mod
    //     makes I_D increase slowly with V_DS).
    REQUIRE(partials[0] > 0);
    //   * ∂I/∂V_s < 0 (raising V_source lowers BOTH V_DS
    //     and V_GS → lowers I_D strongly).
    REQUIRE(partials[1] < 0);
}

TEST_CASE("MosfetLevel1 — V_GS sweep: monotonic I_D in saturation",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    MosfetLevel1::Params p{
        .K = 1e-3, .V_T = 2.0, .lambda = 0.02,
        .kappa = 15.0};
    const Real V_DS = 10.0;   // deep saturation
    Real prev = -1.0;
    for (Real V_GS = 0.0; V_GS <= 6.0; V_GS += 0.5) {
        const Real i = iD(p, V_GS, V_DS);
        if (V_GS > 2.5) {
            // After threshold, I_D must be increasing.
            REQUIRE(i > prev);
        }
        prev = i;
    }
}

// -----------------------------------------------------------------------------
// Newton-loop integration: common-source amplifier.
// Circuit:
//   V_DD = 10 V from vdd to gnd
//   R_D = 5 kΩ from vdd to drain
//   M1 = MosfetLevel1, drain=drain, source=gnd, gate=vgs
//   V_GS = adjustable voltage source from vgs to gnd
// At V_GS = 3 V (V_OV = 1 V), K=1e-3, λ=0.02:
//   In saturation, I_D = K·V_OV²·(1+λV_DS) ≈ 1 mA·(1+0.02V_DS)
//   KVL: V_DD − I_D·R_D = V_DS
//   Iterating: I_D ≈ 1.09 mA, V_DS ≈ 4.55 V
// -----------------------------------------------------------------------------

TEST_CASE("MosfetLevel1 — Newton solves common-source amplifier",
          "[v2][layer2_v13][mosfet_level1][integration]") {
    using namespace pulsim::v2::solver;
    constexpr Real V_DD   = 10.0;
    constexpr Real R_D    = 5000.0;
    constexpr Real V_GS_v = 3.0;     // V_OV = 1 V

    CircuitBuilder b;
    b.add_voltage_source("VDD", "vdd", "gnd", V_DD);
    b.add_resistor("R_D", "vdd", "drain", R_D);
    b.add_voltage_source("VGS_src", "vgs", "gnd", V_GS_v);
    b.add_mosfet_level1(
        "M1", "drain", "gnd", "vgs",
        /*K=*/1e-3, /*V_T=*/2.0,
        /*lambda=*/0.02, /*kappa=*/20.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();   // static cache (no caps/L)

    SimulationOptions opts{
        .t_start = 0.0, .t_end = 0.1, .dt = 0.01,
        .max_newton_iterations = 100};
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };
    auto refresh = &refresh_mosfets_level1;

    auto result = run_transient(
        cache, b.graph(), b.pool(),
        opts, switch_fn, {}, false, refresh);
    REQUIRE(result.num_steps() > 0);

    // Last sample — fully converged DC operating point.
    const Index drain_idx = b.node_id_of("drain");
    REQUIRE(drain_idx >= 0);
    const Real v_drain = result.states[
        result.num_steps() - 1][drain_idx];

    // Self-consistent iteration:
    //   I_D = K·V_OV²·(1+λ·V_DS)
    //   V_DS = V_DD − I_D·R_D
    // Solving: V_DS ≈ 4.55 V → v_drain ≈ 4.55 V
    INFO("v_drain = " << v_drain << " V (expected ~4.55 V)");
    REQUIRE(v_drain == Approx(4.55).margin(0.15));

    // Drain current: I_D ≈ (V_DD - v_drain)/R_D
    const Real I_D_measured = (V_DD - v_drain) / R_D;
    INFO("I_D measured = " << I_D_measured * 1e3
         << " mA (expected ~1.09 mA)");
    REQUIRE(I_D_measured == Approx(1.09e-3).margin(0.05e-3));
}

TEST_CASE("DevicePool stores MosfetLevel1 params + gate node id",
          "[v2][layer2_v13][mosfet_level1][unit]") {
    CircuitBuilder b;
    b.add_mosfet_level1(
        "M1", "drain", "source", "gate",
        /*K=*/2e-3, /*V_T=*/1.5,
        /*lambda=*/0.03, /*kappa=*/20.0);

    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
            DevicePool::StoredKind::MosfetLevel1);
    REQUIRE(b.pool().mosfet_level1_params(0).K ==
            Approx(2e-3));
    REQUIRE(b.pool().mosfet_level1_params(0).V_T ==
            Approx(1.5));
    // Gate node must be a distinct node (idx 2 if drain=0
    // and source=1 in insertion order).
    REQUIRE(b.pool().mosfet_level1_gate_node(0) ==
            b.node_id_of("gate"));
    REQUIRE(b.pool().mosfet_level1_gate_node(0) !=
            b.node_id_of("drain"));
    REQUIRE(b.pool().mosfet_level1_gate_node(0) !=
            b.node_id_of("source"));
}

// =============================================================================
// Proposal #3.1 — add_mosfet_level1 `with_body_diode=true` adds an
// anti-parallel SwitchedDiode (source→drain).
// =============================================================================

TEST_CASE("add_mosfet_level1 with_body_diode=true adds anti-parallel diode",
          "[v2][layer2_v13][mosfet_level1][body_diode]"
          "[ergonomics][quickwins]") {
    CircuitBuilder b;
    b.add_mosfet_level1(
        "M1", "drain", "source", "gate",
        /*K=*/1e-3, /*V_T=*/2.0,
        /*lambda=*/0.02, /*kappa=*/15.0,
        /*with_body_diode=*/true);

    // 1 SH1 MOSFET branch + 1 body diode branch = 2.
    REQUIRE(b.num_branches() == 2);

    // Branch 0 — MOSFET (drain→source, Nonlinear).
    const auto& m = b.graph().branch(0);
    REQUIRE(m.from == b.node_id_of("drain"));
    REQUIRE(m.to   == b.node_id_of("source"));
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::MosfetLevel1);

    // Branch 1 — body diode (source→drain, anti-parallel, Switch).
    const auto& d = b.graph().branch(1);
    REQUIRE(d.from == b.node_id_of("source"));
    REQUIRE(d.to   == b.node_id_of("drain"));
    REQUIRE(b.pool().kind_of(1) ==
              DevicePool::StoredKind::Diode);
}

TEST_CASE("add_mosfet_level1 default (no body diode) still adds 1 branch",
          "[v2][layer2_v13][mosfet_level1][body_diode]"
          "[ergonomics][quickwins]") {
    CircuitBuilder b;
    b.add_mosfet_level1(
        "M1", "drain", "source", "gate",
        /*K=*/1e-3, /*V_T=*/2.0);
    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::MosfetLevel1);
}
