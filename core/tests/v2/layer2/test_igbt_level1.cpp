// =============================================================================
// Layer 2 V14 — IGBT Level 1 model tests
// =============================================================================
//
// Validates:
//   * Cutoff (V_GE < V_T): I_C ≈ 0.
//   * On-state linear conduction: I_C = (V_CE − V_CE_sat) / R_CE_sat.
//   * Gate sigmoid transition smooth + monotonic in V_GE.
//   * DevicePool round-trip + gate-node lookup.
//   * AD partials finite (3-terminal Jacobian).
//   * Newton integration: simple resistive load — IGBT
//     converges to the expected DC operating point.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/igbt_level1.hpp"
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

Real iC(const IgbtLevel1::Params& p, Real V_GE, Real V_CE) {
    const Real v[3] = {V_CE, 0.0, V_GE};   // c, e, g
    return IgbtLevel1::current<Real>(v, p);
}

}  // namespace

TEST_CASE("IgbtLevel1 — cutoff: I_C ≈ 0 when V_GE << V_T",
          "[v2][layer2_v14][igbt_level1][unit]") {
    IgbtLevel1::Params p{
        .V_CE_sat = 1.5, .R_CE_sat = 0.05,
        .V_T = 5.0, .kappa = 10.0};
    REQUIRE(std::abs(iC(p, 0.0, 10.0)) < 1e-3);
    REQUIRE(std::abs(iC(p, 3.0, 10.0)) < 1e-1);
}

TEST_CASE("IgbtLevel1 — on-state linear conduction matches analytical",
          "[v2][layer2_v14][igbt_level1][unit]") {
    IgbtLevel1::Params p{
        .V_CE_sat = 1.5, .R_CE_sat = 0.05,
        .V_T = 5.0, .kappa = 20.0};
    // V_GE=10, well above V_T=5 → α≈1.
    // V_CE=5 → I_C = (5 - 1.5)/0.05 = 70 A
    REQUIRE(iC(p, 10.0, 5.0) == Approx(70.0).margin(0.5));
    // V_CE=3 → I_C = (3 - 1.5)/0.05 = 30 A
    REQUIRE(iC(p, 10.0, 3.0) == Approx(30.0).margin(0.5));
}

TEST_CASE("IgbtLevel1 — Jacobian finite + physically correct",
          "[v2][layer2_v14][igbt_level1][unit]") {
    IgbtLevel1::Params p{
        .V_CE_sat = 1.5, .R_CE_sat = 0.05,
        .V_T = 5.0, .kappa = 10.0};
    ModelInputs<IgbtLevel1> v{5.0, 0.0, 10.0};  // c, e, g
    const auto [i, partials] =
        evaluate_current_and_jacobian<IgbtLevel1>(v, p);
    INFO("I_C = " << i
         << " ∂I/∂V_c = " << partials[0]
         << " ∂I/∂V_e = " << partials[1]
         << " ∂I/∂V_g = " << partials[2]);
    REQUIRE(std::isfinite(i));
    REQUIRE(std::isfinite(partials[0]));
    REQUIRE(std::isfinite(partials[1]));
    REQUIRE(std::isfinite(partials[2]));
    // Physical sanity:
    REQUIRE(partials[0] > 0);   // raising V_c → more I_C
    REQUIRE(partials[1] < 0);   // raising V_e → less I_C (V_CE and V_GE both drop)
    REQUIRE(partials[2] >= 0);  // raising V_g → α saturates above V_T (small dα/dV_g)
}

TEST_CASE("IgbtLevel1 — V_GE sweep: monotonic I_C",
          "[v2][layer2_v14][igbt_level1][unit]") {
    IgbtLevel1::Params p{
        .V_CE_sat = 1.5, .R_CE_sat = 0.05,
        .V_T = 5.0, .kappa = 10.0};
    Real prev = -1.0;
    for (Real V_GE = 0.0; V_GE <= 12.0; V_GE += 0.5) {
        const Real i = iC(p, V_GE, 5.0);
        if (V_GE > 5.5) REQUIRE(i >= prev);
        prev = i;
    }
}

TEST_CASE("DevicePool stores IgbtLevel1 params + gate node id",
          "[v2][layer2_v14][igbt_level1][unit]") {
    CircuitBuilder b;
    b.add_igbt_level1(
        "T1", "collector", "emitter", "gate",
        1.8, 0.04, 5.5, 12.0);
    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
            DevicePool::StoredKind::IgbtLevel1);
    REQUIRE(b.pool().igbt_level1_params(0).V_CE_sat ==
            Approx(1.8));
    REQUIRE(b.pool().igbt_level1_params(0).V_T == Approx(5.5));
    REQUIRE(b.pool().igbt_level1_gate_node(0) ==
            b.node_id_of("gate"));
}

// -----------------------------------------------------------------------------
// Newton integration: IGBT chopper-style circuit. V_in → R_in
// → collector → IGBT → emitter=gnd, with V_GE driven by a
// constant voltage source.
//
// At V_GE = 10 (V_OV = 5), expect IGBT in active region:
//   V_CE = V_CE_sat + R_CE_sat · I_C   (KVL via IGBT)
//   I_C = (V_in - V_CE) / R_in         (KVL via R_in)
//
// Self-consistent: V_in = V_CE + I_C · R_in
//                       = V_CE_sat + R_CE_sat·I_C + R_in·I_C
//                  I_C = (V_in - V_CE_sat) / (R_in + R_CE_sat)
//
// V_in=24, R_in=10, V_CE_sat=1.5, R_CE_sat=0.05:
//   I_C = (24 - 1.5) / 10.05 = 2.239 A
//   V_CE = 1.5 + 0.05 · 2.239 = 1.612 V
// -----------------------------------------------------------------------------

TEST_CASE("IgbtLevel1 — Newton solves chopper-style DC operating point",
          "[v2][layer2_v14][igbt_level1][integration]") {
    using namespace pulsim::v2::solver;
    CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 24.0);
    b.add_resistor("R_in", "vin", "collector", 10.0);
    b.add_voltage_source("VGE_src", "gate", "gnd", 10.0);
    b.add_igbt_level1(
        "T1", "collector", "gnd", "gate",
        1.5, 0.05, 5.0, 20.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();

    SimulationOptions opts{
        .t_start = 0.0, .t_end = 0.1, .dt = 0.01,
        .max_newton_iterations = 100};
    auto sw_fn = [](Real) {
        return SwitchStateMask(0);
    };
    auto refresh = make_combined_diode_mosfet_refresh();

    auto result = run_transient(
        cache, b.graph(), b.pool(),
        opts, sw_fn, {}, false, refresh);

    const Index c_idx = b.node_id_of("collector");
    REQUIRE(c_idx >= 0);
    const Real v_ce = result.states[
        result.num_steps() - 1][c_idx];
    INFO("V_CE = " << v_ce << " V (expected ~1.612)");
    REQUIRE(v_ce == Approx(1.612).margin(0.05));

    const Real i_c = (24.0 - v_ce) / 10.0;
    INFO("I_C = " << i_c << " A (expected ~2.239)");
    REQUIRE(i_c == Approx(2.239).margin(0.05));
}
