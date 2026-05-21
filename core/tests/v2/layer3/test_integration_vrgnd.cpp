// =============================================================================
// Layer 3 — Integration test: V-R-GND circuit end-to-end
// =============================================================================
//
// The classic single-loop circuit: voltage source in series with a
// resistor, all referenced to ground.
//
//     V_dc ──+── R ──+── GND
//            ^       ^
//            |       |
//          node 0  (ground)
//
// At convergence:
//   * v_node_0 = V_dc
//   * i_branch = V_dc · G
//
// State vector: [v_node_0, i_branch]. Size 2.
//
// This test stamps the voltage source AND the resistor into the
// SAME (J, f), solves J · Δx = -f using Layer 0's SparseLuSolver,
// updates x ← x + Δx, and verifies the result.
//
// Since the system is LINEAR, ONE Newton step from x = [0, 0]
// must land at the exact analytical solution (to within
// floating-point precision).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/models/resistor.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/sparse/solver.hpp"
#include "pulsim/v2/stamping/stamp_device.hpp"
#include "pulsim/v2/stamping/stamp_voltage_source.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::stamping;
using namespace pulsim::v2::models;
using Catch::Approx;

TEST_CASE("V-R-GND: one Newton step solves the linear system exactly",
          "[v2][layer3][integration]") {
    constexpr Real V_dc = Real{12};
    constexpr Real G = Real{0.1};         // 10 Ω resistor
    constexpr Index N = 2;                // [v_node_0, i_branch]

    // State vector, start at zero.
    Vector x = Vector::Zero(N);

    // Build the system: V_dc(0, gnd) and R(0, gnd).
    sparse::Matrix J(N, N);
    Vector f = Vector::Zero(N);

    // (1) Voltage source between node 0 and ground.
    //     Branch-current unknown lives at index 1.
    BranchCoord vsrc_coord{Index{0}, kGround, Index{0}};
    stamp_voltage_source(J, f, x, vsrc_coord,
                          /*branch_var_id=*/Index{1}, V_dc);

    // (2) Resistor between node 0 and ground.
    Resistor::Params r_params{G};
    BranchCoord r_coord{Index{0}, kGround, Index{1}};
    stamp_device<Resistor>(J, f, x, r_coord, r_params);

    sparse::compress_in_place(J);

    // Solve J · Δx = -f.
    sparse::SparseLuSolver solver;
    REQUIRE(solver.analyze(J));
    REQUIRE(solver.factorize(J));
    Vector delta_x;
    Vector neg_f = -f;
    solver.solve(neg_f, delta_x);
    x += delta_x;

    INFO("v_node_0 = " << x[0] << " (expected " << V_dc << ")");
    INFO("i_branch = " << x[1]
         << " (expected " << (V_dc * G) << ")");

    // Verify analytical solution.
    REQUIRE(x[0] == Approx(V_dc).margin(1e-12));
    REQUIRE(x[1] == Approx(-V_dc * G).margin(1e-12));
    // The branch current is NEGATIVE here because the source's
    // KCL contribution convention is `+i_branch leaves node 0`
    // while the source actually injects current INTO node 0. The
    // sign is consistent: at convergence f[0] = 0 = i_branch +
    // G·v_node_0, so i_branch = -G·V_dc.
}

TEST_CASE("V-R-GND: residual is zero at the analytical solution",
          "[v2][layer3][integration]") {
    constexpr Real V_dc = Real{5};
    constexpr Real G = Real{2.0};
    constexpr Index N = 2;

    Vector x(N);
    x << V_dc, -V_dc * G;                 // Plant the analytical solution

    sparse::Matrix J(N, N);
    Vector f = Vector::Zero(N);

    BranchCoord vsrc_coord{Index{0}, kGround, Index{0}};
    stamp_voltage_source(J, f, x, vsrc_coord, Index{1}, V_dc);

    Resistor::Params r_params{G};
    BranchCoord r_coord{Index{0}, kGround, Index{1}};
    stamp_device<Resistor>(J, f, x, r_coord, r_params);

    // Both residual entries must be zero (KCL + constraint
    // simultaneously satisfied).
    REQUIRE(f[0] == Approx(Real{0}).margin(1e-12));
    REQUIRE(f[1] == Approx(Real{0}).margin(1e-12));
}
