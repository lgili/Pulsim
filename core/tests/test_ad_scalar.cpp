// =============================================================================
// Test: AD scalar bridge (Phase 1 of add-automatic-differentiation)
// =============================================================================
//
// Validates forward-mode AD primitives that downstream phases use to derive
// device Jacobians automatically. The contracts asserted here are:
//
//   1. Scalar `f(x) = sin(x) · exp(x)` round-trip: AD-derived f' matches the
//      analytical derivative.
//   2. Vector AD on a linear residual `i = G · (v_a − v_b)`: gradient
//      entries equal `+G` and `−G` exactly.
//   3. Shichman-Hodges MOSFET saturation `id = 0.5·kp·(vgs−vth)²·(1+λ·vds)`:
//      AD partial derivatives match a centered-difference reference within
//      1e-4 absolute (ample slack for FD truncation noise).
//   4. `stack_jacobian` builds an (n_residuals × n_inputs) matrix correctly.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/ad/ad_scalar.hpp"

#include <array>
#include <cmath>

using namespace pulsim::v1;
using ad::ADReal;
using Catch::Approx;

namespace {

[[nodiscard]] Approx approx(double v, double margin = 1e-12) {
    return Approx(v).margin(margin);
}

}  // namespace

TEST_CASE("ad: scalar f(x) = sin(x) * exp(x) — AD matches analytical f'(x)",
          "[ad][scalar]") {
    constexpr Real x_val = 0.5;
    auto seeded = ad::seed_from_values({x_val});
    const ADReal& x = seeded[0];

    // Eigen AutoDiff supplies overloads of sin/exp via ADL; use unqualified
    // names so the AD overloads are found alongside std for the Real path.
    using std::sin;
    using std::exp;
    const ADReal y = sin(x) * exp(x);

    // f(x)  = sin(x) * exp(x)
    // f'(x) = cos(x) * exp(x) + sin(x) * exp(x) = exp(x) * (cos(x) + sin(x))
    const Real expected_value = std::sin(x_val) * std::exp(x_val);
    const Real expected_derivative =
        std::exp(x_val) * (std::cos(x_val) + std::sin(x_val));

    CHECK(y.value() == approx(expected_value));
    REQUIRE(y.derivatives().size() == 1);
    CHECK(y.derivatives()[0] == approx(expected_derivative));
}

TEST_CASE("ad: vector AD on linear residual — gradient is exact",
          "[ad][linear]") {
    // A resistor between nodes a and b: residual at node a is i = G * (v_a - v_b).
    constexpr Real G = 0.001;        // 1 kΩ resistor
    constexpr Real v_a = 5.0;
    constexpr Real v_b = 0.0;

    auto seeded = ad::seed_from_values({v_a, v_b});
    const ADReal i_residual = G * (seeded[0] - seeded[1]);

    CHECK(i_residual.value() == approx(G * (v_a - v_b)));
    REQUIRE(i_residual.derivatives().size() == 2);
    CHECK(i_residual.derivatives()[0] == approx(+G));
    CHECK(i_residual.derivatives()[1] == approx(-G));
}

TEST_CASE("ad: Shichman-Hodges saturation drain current — AD matches FD",
          "[ad][nonlinear][mosfet]") {
    // id = 0.5 · kp · (vgs − vth)² · (1 + λ · vds)
    // Partials:
    //   ∂id/∂vgs = kp · (vgs − vth) · (1 + λ · vds)
    //   ∂id/∂vds = 0.5 · kp · (vgs − vth)² · λ
    constexpr Real kp = 0.1;
    constexpr Real vth = 2.0;
    constexpr Real lambda = 0.01;
    constexpr Real vgs_val = 5.0;
    constexpr Real vds_val = 3.0;

    // Reference: Real-only finite difference around the operating point.
    auto id_real = [&](Real vgs, Real vds) {
        const Real vov = vgs - vth;
        return 0.5 * kp * vov * vov * (1.0 + lambda * vds);
    };
    constexpr Real eps = 1e-6;
    const Real id_center = id_real(vgs_val, vds_val);
    const Real id_dvgs =
        (id_real(vgs_val + eps, vds_val) - id_real(vgs_val - eps, vds_val)) /
        (2.0 * eps);
    const Real id_dvds =
        (id_real(vgs_val, vds_val + eps) - id_real(vgs_val, vds_val - eps)) /
        (2.0 * eps);

    // AD evaluation: build the residual scalar in-line so the AD chain is
    // unmistakable to the compiler. Eigen's AutoDiffScalar propagates
    // derivatives through Real ↔ ADReal mixed arithmetic when Real literals
    // are on the *left* of `*` (so `Real * ADReal` overload is selected).
    auto seeded = ad::seed_from_values({vgs_val, vds_val});
    const ADReal vov_ad = seeded[0] - vth;
    const ADReal mod_ad = 1.0 + lambda * seeded[1];
    const ADReal id_ad = 0.5 * kp * vov_ad * vov_ad * mod_ad;

    // Analytical reference.
    const Real id_dvgs_analytical = kp * (vgs_val - vth) * (1.0 + lambda * vds_val);
    const Real id_dvds_analytical = 0.5 * kp * std::pow(vgs_val - vth, 2) * lambda;

    CHECK(id_ad.value() == approx(id_center));
    REQUIRE(id_ad.derivatives().size() == 2);
    CHECK(id_ad.derivatives()[0] == Approx(id_dvgs).margin(1e-4));
    CHECK(id_ad.derivatives()[1] == Approx(id_dvds).margin(1e-4));
    // Tighter check vs analytical: AD should be exact up to floating-point.
    CHECK(id_ad.derivatives()[0] == approx(id_dvgs_analytical));
    CHECK(id_ad.derivatives()[1] == approx(id_dvds_analytical));
}

TEST_CASE("ad: stack_jacobian builds a correct (rows × cols) Jacobian matrix",
          "[ad][jacobian]") {
    // Two-residual system on three independent inputs:
    //   r0 = 2*x0 + x1 - x2
    //   r1 = x0 * x1 + x2
    // Expected Jacobian:
    //   [[2, 1, -1],
    //    [x1, x0, 1]]
    constexpr Real x0_val = 1.5;
    constexpr Real x1_val = 2.0;
    constexpr Real x2_val = 0.5;

    auto seeded = ad::seed_from_values({x0_val, x1_val, x2_val});
    std::array<ADReal, 2> residuals{
        2.0 * seeded[0] + seeded[1] - seeded[2],
        seeded[0] * seeded[1] + seeded[2],
    };

    const Eigen::MatrixXd J =
        ad::stack_jacobian(std::span<const ADReal>(residuals.data(), residuals.size()));

    REQUIRE(J.rows() == 2);
    REQUIRE(J.cols() == 3);
    CHECK(J(0, 0) == approx(2.0));
    CHECK(J(0, 1) == approx(1.0));
    CHECK(J(0, 2) == approx(-1.0));
    CHECK(J(1, 0) == approx(x1_val));
    CHECK(J(1, 1) == approx(x0_val));
    CHECK(J(1, 2) == approx(1.0));
}
