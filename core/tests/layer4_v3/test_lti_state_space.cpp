// =============================================================================
// Bridge.3 — Validate PwlStateSpaceCache::compute_lti_state_space
//            against the analytical buck CCM state-space
// =============================================================================
//
// The MNA finite-difference recovery is documented in
// `notes/DSED_BRIDGE_DESIGN.md` §2. This test verifies that for a
// canonical buck CCM topology the extracted (A, b) per switch mask
// matches the textbook state-space (Erickson & Maksimovic eq. 7.45):
//
//   For HS_on  (high-side closed, low-side open):
//     L · di_L/dt = V_in - v_C
//     C · dv_C/dt = i_L  - v_C/R
//   For HS_off (high-side open, low-side closed):
//     L · di_L/dt = -v_C
//     C · dv_C/dt = i_L - v_C/R
//
// Same A matrix in both modes; only b differs.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/topology/switch_state.hpp"

using namespace pulsim;
using pulsim::pwl::PwlStateSpaceCache;
using pulsim::topology::SwitchStateMask;

namespace {

// Build a 2-switch buck (HS + LS, complementary): V_in → SW_HS → L → C → R → gnd
//                                                  SW_LS shorts (L-end) → gnd
//   Mask bit 0 = HS state (1=closed)
//   Mask bit 1 = LS state (1=closed)
// HS_on mask = (bit0=1, bit1=0); HS_off mask = (bit0=0, bit1=1).
builder::CircuitBuilder make_buck(Real V_in, Real L, Real C, Real R) {
    builder::CircuitBuilder b;
    // V_in source between "in" and gnd
    b.add_voltage_source("Vin", "in", "gnd", V_in);
    // High-side switch from "in" to "sw_node"
    constexpr Real G_ON  = Real{1e6};
    constexpr Real G_OFF = Real{1e-9};
    b.add_switch("SW_HS", "in", "sw_node", G_ON, G_OFF);
    // Low-side switch from "sw_node" to gnd
    b.add_switch("SW_LS", "sw_node", "gnd", G_ON, G_OFF);
    // Inductor from "sw_node" to "out"
    b.add_inductor("L",  "sw_node", "out", L);
    // Output cap + load
    b.add_capacitor("C", "out", "gnd", C);
    b.add_resistor("R",  "out", "gnd", R);
    return b;
}

}  // namespace

TEST_CASE("compute_lti_state_space recovers buck CCM analytical (A, b)",
           "[bridge][lti][buck-ccm]") {
    const Real V_in = Real{24};
    const Real L    = Real{100e-6};
    const Real C    = Real{100e-6};
    const Real R    = Real{2.4};

    auto b = make_buck(V_in, L, C, R);
    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-7});   // dt sentinel — the extractor uses its own h_a/h_b

    // 2 switches → 4 masks. The two interesting ones:
    //   mask = 0b01: HS_on  (bit 0 = HS closed; LS open)
    //   mask = 0b10: HS_off (HS open; LS closed)
    SwitchStateMask m_hs_on(2);
    m_hs_on.set(0, true);   // HS closed
    m_hs_on.set(1, false);  // LS open

    SwitchStateMask m_hs_off(2);
    m_hs_off.set(0, false); // HS open
    m_hs_off.set(1, true);  // LS closed

    // Build a known steady state baseline so we can compare A·x against
    // the expected dx/dt.

    SECTION("HS_on mask") {
        auto lti = cache.compute_lti_state_space(m_hs_on);

        REQUIRE(lti.A.rows() == 2);
        REQUIRE(lti.A.cols() == 2);
        REQUIRE(lti.b_constant.size() == 2);
        REQUIRE(lti.state_row_indices.size() == 2);
        REQUIRE(lti.state_is_cap.size() == 2);

        // Convention: caps first (v_C), inductors second (i_L)
        // → state[0] = v_C, state[1] = i_L
        REQUIRE(lti.state_is_cap[0] == true);    // cap
        REQUIRE(lti.state_is_cap[1] == false);   // inductor

        // Compare against analytical (state = [v_C, i_L]):
        //   dv_C/dt =  i_L/C    - v_C/(R·C)
        //   di_L/dt = -v_C/L    + V_in/L  (HS_on)
        //
        // A = [[ -1/(R·C),   1/C  ],
        //      [ -1/L,       0    ]]
        // b = [   0,        V_in/L]
        const Real tol_pct = Real{1e-6};
        const Real expected_A_00 = -Real{1} / (R * C);
        const Real expected_A_01 =  Real{1} / C;
        const Real expected_A_10 = -Real{1} / L;
        const Real expected_A_11 =  Real{0};
        const Real expected_b_0  =  Real{0};
        const Real expected_b_1  =  V_in / L;

        INFO("A =\n" << lti.A);
        INFO("b = " << lti.b_constant.transpose());

        // Noise floor on the zero-entries comes from the switch
        // G_OFF=1e-9 finite conductance + the finite-difference
        // recovery of M_dyn. Empirically A(1,1) ≈ 0.01, which is
        // about (G_OFF · ω_switch_scale / L) ≈ 1e-9·1e4/1e-4 = 0.001
        // amplified by ~10× through the Schur complement.
        const Real zero_noise_floor = Real{0.1};
        REQUIRE(lti.A(0, 0) == Catch::Approx(expected_A_00).epsilon(tol_pct));
        REQUIRE(lti.A(0, 1) == Catch::Approx(expected_A_01).epsilon(tol_pct));
        REQUIRE(lti.A(1, 0) == Catch::Approx(expected_A_10).epsilon(tol_pct));
        REQUIRE(std::abs(lti.A(1, 1) - expected_A_11) < zero_noise_floor);
        REQUIRE(std::abs(lti.b_constant(0) - expected_b_0) < zero_noise_floor);
        REQUIRE(lti.b_constant(1) == Catch::Approx(expected_b_1).epsilon(tol_pct));
    }

    SECTION("HS_off mask") {
        auto lti = cache.compute_lti_state_space(m_hs_off);

        // Same A; only b differs:
        //   b = [0, 0]
        const Real tol_pct = Real{1e-6};
        const Real expected_A_00 = -Real{1} / (R * C);
        const Real expected_A_01 =  Real{1} / C;
        const Real expected_A_10 = -Real{1} / L;
        const Real zero_noise_floor = Real{0.1};

        REQUIRE(lti.A(0, 0) == Catch::Approx(expected_A_00).epsilon(tol_pct));
        REQUIRE(lti.A(0, 1) == Catch::Approx(expected_A_01).epsilon(tol_pct));
        REQUIRE(lti.A(1, 0) == Catch::Approx(expected_A_10).epsilon(tol_pct));
        REQUIRE(std::abs(lti.A(1, 1)) < zero_noise_floor);

        // HS_off: both b entries are ~0 (no V_in connection)
        REQUIRE(std::abs(lti.b_constant(0)) < zero_noise_floor);
        REQUIRE(std::abs(lti.b_constant(1)) < zero_noise_floor);
    }
}

TEST_CASE("compute_lti_state_space rejects a circuit without state vars",
           "[bridge][lti]") {
    // Pure resistive divider: no caps, no inductors → no state to evolve
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "n1", "gnd", Real{5});
    b.add_resistor("R1", "n1", "n2", Real{100});
    b.add_resistor("R2", "n2", "gnd", Real{100});

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);   // no switches
    REQUIRE_THROWS_AS(
        cache.compute_lti_state_space(m),
        std::runtime_error);
}

TEST_CASE("compute_lti_state_space handles a single floating capacitor "
           "(Phase 5.1b)",
           "[bridge][lti][floating-cap]") {
    // Series R1 - C(floating) - R2 charging from V_source.
    //
    //   V → R1 → n1 → C1(floating) → n2 → R2 → gnd
    //
    // Analytical:
    //   dv_C/dt = -v_C / ((R1+R2)·C) + V / ((R1+R2)·C)
    //
    // → A = -1/(R_total · C); b = V/(R_total · C); R_total = R1+R2.
    //
    // With the BFS-orientation convention (anchor = lowest-index MNA
    // node), the state is v[pos] - v[neg]. For this topology, BFS
    // from the lowest-index node may flip the cap's sign relative to
    // the device's branch.from/branch.to order — the DYNAMICS still
    // match (|A|, |b| are correct; sign of b just flips if state =
    // -v_C). We verify the magnitudes.

    const Real V = Real{5};
    const Real R1 = Real{10};
    const Real R2 = Real{10};
    const Real C = Real{1e-6};
    const Real R_total = R1 + R2;

    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "vin", "gnd", V);
    b.add_resistor("R1",      "vin", "n1",  R1);
    b.add_capacitor("C1",     "n1",  "n2",  C);   // FLOATING
    b.add_resistor("R2",      "n2",  "gnd", R2);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);
    auto lti = cache.compute_lti_state_space(m);

    REQUIRE(lti.A.rows() == 1);
    REQUIRE(lti.A.cols() == 1);
    REQUIRE(lti.b_constant.size() == 1);
    REQUIRE(lti.state_is_cap.size() == 1);
    REQUIRE(lti.state_is_cap[0] == true);

    const Real expected_A_00 = -Real{1} / (R_total * C);
    const Real expected_b_mag = V / (R_total * C);
    const Real tol_pct = Real{1e-4};

    INFO("A = " << lti.A);
    INFO("b = " << lti.b_constant.transpose());
    INFO("expected A_00 = " << expected_A_00);
    INFO("expected |b_0| = " << expected_b_mag);

    REQUIRE(lti.A(0, 0)
            == Catch::Approx(expected_A_00).epsilon(tol_pct));
    // Sign of b depends on BFS orientation (state = ±v_C); the
    // magnitude must match.
    REQUIRE(std::abs(lti.b_constant(0))
            == Catch::Approx(expected_b_mag).epsilon(tol_pct));
}

TEST_CASE("compute_lti_state_space handles a 2-cap split-bus stack "
           "(NPC-style)",
           "[bridge][lti][floating-cap][npc]") {
    // Two floating caps in series between V_dc_pos and V_dc_neg with
    // a midpoint node ("mid"). Mimics the NPC capacitor split.
    //
    //   V_dc → R_src → n_pos → C_top → n_mid → C_bot → n_neg → R_load → gnd
    //
    // Cap chain {n_pos, n_mid, n_neg} doesn't touch ground; the
    // resistor R_load → gnd provides a ground reference outside the
    // cap component. After T^T·M·T, both caps must produce VALID
    // dynamics — we verify finite, well-conditioned A and that A is
    // stable (eigenvalues with negative real part).

    const Real V_dc = Real{100};
    const Real R_src = Real{1};
    const Real C = Real{10e-6};
    const Real R_load = Real{100};

    builder::CircuitBuilder b;
    b.add_voltage_source("Vdc", "vin", "gnd", V_dc);
    b.add_resistor("R_src",      "vin",  "n_pos", R_src);
    b.add_capacitor("C_top",     "n_pos","n_mid", C);    // FLOATING
    b.add_capacitor("C_bot",     "n_mid","n_neg", C);    // FLOATING
    b.add_resistor("R_load",     "n_neg","gnd",   R_load);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);
    auto lti = cache.compute_lti_state_space(m);

    // 2 caps, 0 inductors → 2 states
    REQUIRE(lti.A.rows() == 2);
    REQUIRE(lti.A.cols() == 2);
    REQUIRE(lti.b_constant.size() == 2);
    REQUIRE(lti.state_is_cap[0] == true);
    REQUIRE(lti.state_is_cap[1] == true);

    INFO("A =\n" << lti.A);
    INFO("b = " << lti.b_constant.transpose());

    // All A entries must be finite.
    REQUIRE(std::isfinite(lti.A(0, 0)));
    REQUIRE(std::isfinite(lti.A(0, 1)));
    REQUIRE(std::isfinite(lti.A(1, 0)));
    REQUIRE(std::isfinite(lti.A(1, 1)));
    REQUIRE(std::isfinite(lti.b_constant(0)));
    REQUIRE(std::isfinite(lti.b_constant(1)));

    // The system must be stable (eigenvalues have negative real parts).
    Eigen::EigenSolver<DenseMatrix> es(lti.A);
    INFO("Eigenvalues: " << es.eigenvalues().transpose());
    REQUIRE(es.eigenvalues()(0).real() < Real{0});
    REQUIRE(es.eigenvalues()(1).real() < Real{0});

    // Sanity: at steady state (dx/dt = 0), x_ss = -A^-1 · b.
    //   For this topology the steady-state cap voltage drop across
    //   each cap should sum (in absolute terms) to V_dc · R_load /
    //   (R_load + R_src + ε) ≈ V_dc (since R_load >> R_src).
    //   We don't predict each cap's split exactly (it depends on the
    //   BFS-chosen anchor and the algebraic Schur), but their sum
    //   magnitude must be in (0, V_dc].
    Vector x_ss = -lti.A.inverse() * lti.b_constant;
    INFO("x_ss = " << x_ss.transpose());
    const Real total_cap_voltage = std::abs(x_ss(0)) + std::abs(x_ss(1));
    REQUIRE(total_cap_voltage > Real{0});
    REQUIRE(total_cap_voltage <= V_dc * Real{1.01});  // allow 1% slack
}

TEST_CASE("compute_lti_state_space handles a 3-cap chain (MMC-style stack)",
           "[bridge][lti][floating-cap][mmc]") {
    // Three caps stacked in a chain — analog of an MMC submodule arm
    // with N=3. All caps are floating; the chain terminates at gnd
    // through R_arm.
    //
    //   V_in → R_src → n_top → C1 → n_a → C2 → n_b → C3 → n_bot → R_arm → gnd
    //
    // 3 caps → 3 states. We verify shape + stability.

    const Real V_in = Real{300};
    const Real R_src = Real{1};
    const Real C = Real{5e-6};
    const Real R_arm = Real{50};

    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", V_in);
    b.add_resistor("R_src",      "vin",  "n_top", R_src);
    b.add_capacitor("C1",        "n_top","n_a",   C);
    b.add_capacitor("C2",        "n_a",  "n_b",   C);
    b.add_capacitor("C3",        "n_b",  "n_bot", C);
    b.add_resistor("R_arm",      "n_bot","gnd",   R_arm);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);
    auto lti = cache.compute_lti_state_space(m);

    REQUIRE(lti.A.rows() == 3);
    REQUIRE(lti.A.cols() == 3);
    REQUIRE(lti.b_constant.size() == 3);
    for (int i = 0; i < 3; ++i) REQUIRE(lti.state_is_cap[i] == true);

    INFO("A =\n" << lti.A);
    INFO("b = " << lti.b_constant.transpose());

    // All entries finite
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(std::isfinite(lti.A(i, j)));
        }
        REQUIRE(std::isfinite(lti.b_constant(i)));
    }

    // Stable: all eigenvalues' real parts negative
    Eigen::EigenSolver<DenseMatrix> es(lti.A);
    INFO("Eigenvalues: " << es.eigenvalues().transpose());
    for (int i = 0; i < 3; ++i) {
        REQUIRE(es.eigenvalues()(i).real() < Real{0});
    }

    // Steady-state cap voltages must sum (in magnitude) to ≤ V_in.
    Vector x_ss = -lti.A.inverse() * lti.b_constant;
    INFO("x_ss = " << x_ss.transpose());
    const Real total_v = std::abs(x_ss(0)) + std::abs(x_ss(1))
                       + std::abs(x_ss(2));
    REQUIRE(total_v > Real{0});
    REQUIRE(total_v <= V_in * Real{1.01});
}

TEST_CASE("compute_lti_state_space rejects parallel inductors "
           "(Bridge.9)",
           "[bridge][lti][inductor-cycle]") {
    // Two inductors sharing both terminals = inductor loop.
    // KVL: v_a - v_b = L1·di_L1/dt = L2·di_L2/dt creates a linear
    // dependency on di_L/dt that makes the LTI A matrix singular.
    // Reject with a clear "merge into single L_eq" pointer.

    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "vin", "gnd", Real{5});
    b.add_resistor("R1",       "vin", "n1",  Real{10});
    b.add_inductor("L1",       "n1",  "n2",  Real{1e-3});
    b.add_inductor("L2",       "n1",  "n2",  Real{2e-3});   // PARALLEL
    b.add_resistor("R2",       "n2",  "gnd", Real{10});

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);
    REQUIRE_THROWS_AS(
        cache.compute_lti_state_space(m),
        std::runtime_error);
}

TEST_CASE("compute_lti_state_space rejects 3-inductor triangle loop "
           "(Bridge.9)",
           "[bridge][lti][inductor-cycle]") {
    // Three inductors forming a triangle = 3-cycle in the inductor
    // graph. Same KVL constraint as parallel inductors but for a
    // longer loop. Same rejection.

    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "vin", "gnd", Real{5});
    b.add_resistor("R1",       "vin", "n1",  Real{10});
    b.add_inductor("L1",       "n1",  "n2",  Real{1e-3});
    b.add_inductor("L2",       "n2",  "n3",  Real{1e-3});
    b.add_inductor("L3",       "n3",  "n1",  Real{1e-3});   // closes the loop
    b.add_resistor("R2",       "n2",  "gnd", Real{10});

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);
    REQUIRE_THROWS_AS(
        cache.compute_lti_state_space(m),
        std::runtime_error);
}

TEST_CASE("compute_lti_state_space rejects parallel capacitors",
           "[bridge][lti][floating-cap]") {
    // Two caps sharing both terminals = a cycle in the cap-edge graph.
    // BFS can only include one as a tree edge; the other is detected
    // and we throw a clear error pointing to the workaround
    // (merge them into a single equivalent cap upstream).

    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "vin", "gnd", Real{5});
    b.add_resistor("R1",       "vin", "n1",  Real{10});
    b.add_capacitor("C1",      "n1",  "n2",  Real{1e-6});
    b.add_capacitor("C2",      "n1",  "n2",  Real{2e-6});   // PARALLEL
    b.add_resistor("R2",       "n2",  "gnd", Real{10});

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(Real{1e-6});

    SwitchStateMask m(0);
    REQUIRE_THROWS_AS(
        cache.compute_lti_state_space(m),
        std::runtime_error);
}
