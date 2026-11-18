// =============================================================================
// add-mna-native-machines task 2.2 — induction machine stamp and model
// =============================================================================
//
//   * the Jacobian `refresh_pmsm_mna` stamps for an induction machine
//     equals central differences of its own residual at 100 random
//     states, under each stage (trapezoidal, BDF2, backward Euler);
//   * the model conserves power: with the stator voltages the device
//     equations imply, Σ (v_k − v_n)·i_k = copper loss + dW/dt + T_em·ω
//     at random states (dW/dt along the same trajectory direction).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/induction_machine_mna.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"

#include <cmath>
#include <random>

using namespace pulsim;
using IM = models::InductionMachineMna;

namespace {

constexpr IM::Params kTest{.R_s = 0.5, .L_ls = 2e-3, .R_r = 0.4,
                           .L_lr = 2e-3, .L_m = 60e-3, .pole_pairs = 2,
                           .T_load = 1.5};

struct Rig {
    builder::CircuitBuilder b;
    Size n = 0;
    pwl::PmsmMnaHistory h;
    Rig() {
        b.add_induction_machine_mna("M", "a", "b", "c", "n", "w",
                                    kTest.R_s, kTest.L_ls, kTest.R_r,
                                    kTest.L_lr, kTest.L_m, kTest.pole_pairs,
                                    0.02, 0.0, kTest.T_load);
        n = b.pool().state_size(b.graph());
        h.init(b.graph(), b.pool());
    }
};

Vector random_state(Size n, std::mt19937& rng) {
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    Vector x(static_cast<Index>(n));
    for (Index i = 0; i < x.size(); ++i) x[i] = 50.0 * u(rng);
    return x;
}

}  // namespace

TEST_CASE("InductionMachineMna: stamped Jacobian equals central "
          "differences of the residual", "[machines][induction]") {
    Rig r;
    REQUIRE(r.h.entries().size() == 1);
    REQUIRE(r.h.entries()[0].induction);
    std::mt19937 rng(7);
    constexpr Real h = 1e-5;
    const TrBdf2Stage stages[] = {TrBdf2Stage::Trapezoidal,
                                  TrBdf2Stage::Bdf2Stage2,
                                  TrBdf2Stage::BackwardEuler};
    Real worst = 0;
    for (int trial = 0; trial < 100; ++trial) {
        r.h.update_from_state(random_state(r.n, rng), h);
        r.h.capture_gamma(random_state(r.n, rng));
        const Vector x = random_state(r.n, rng);
        const auto stage = stages[trial % 3];
        const auto N = static_cast<Index>(r.n);
        auto residual = [&](const Vector& xx) {
            sparse::Matrix J(N, N);
            Vector f = Vector::Zero(N);
            pwl::refresh_pmsm_mna(xx, J, f, r.h, h, stage);
            return f;
        };
        sparse::Matrix J(N, N);
        Vector f = Vector::Zero(N);
        pwl::refresh_pmsm_mna(x, J, f, r.h, h, stage);
        const Eigen::MatrixXd Jd = Eigen::MatrixXd(J);
        for (Index c = 0; c < N; ++c) {
            const Real d = 1e-4 * std::max(Real{1}, std::abs(x[c]));
            Vector xp = x, xm = x;
            xp[c] += d;
            xm[c] -= d;
            const Vector col = (residual(xp) - residual(xm)) / (2 * d);
            const Real scale = std::max(Real{1}, col.cwiseAbs().maxCoeff());
            worst = std::max(worst,
                             (col - Jd.col(c)).cwiseAbs().maxCoeff() / scale);
        }
    }
    INFO("worst relative column error " << worst);
    CHECK(worst < 1e-7);
}

TEST_CASE("InductionMachineMna: power balance p_in = copper + dW/dt + "
          "T_em·omega", "[machines][induction]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    const auto sm = IM::stator(kTest);
    Real worst = 0;
    for (int trial = 0; trial < 100; ++trial) {
        IM::Vec3 i{30 * u(rng), 30 * u(rng), 30 * u(rng)};
        IM::Vec2 lr{0.5 * u(rng), 0.5 * u(rng)};
        IM::Vec3 di{1e4 * u(rng), 1e4 * u(rng), 1e4 * u(rng)};
        const Real omega = 200 * u(rng);
        const auto dlr = IM::rotor_rhs(kTest, i, lr, omega);
        // v_k − v_n = R_s·i_k + dλ_k/dt, dλ = L·di + M·dλ_r.
        Real p_in = 0;
        for (Size k = 0; k < 3; ++k) {
            Real dl = sm.M[k][0] * dlr[0] + sm.M[k][1] * dlr[1];
            for (Size j = 0; j < 3; ++j) dl += sm.L[k][j] * di[j];
            p_in += (kTest.R_s * i[k] + dl) * i[k];
        }
        // W is quadratic: the central difference along (di, dλ_r) is exact
        // up to round-off.
        const Real e = 1e-6;
        IM::Vec3 ip, im;
        IM::Vec2 lp, lm;
        for (Size k = 0; k < 3; ++k) {
            ip[k] = i[k] + e * di[k];
            im[k] = i[k] - e * di[k];
        }
        for (Size k = 0; k < 2; ++k) {
            lp[k] = lr[k] + e * dlr[k];
            lm[k] = lr[k] - e * dlr[k];
        }
        const Real dW = (IM::energy(kTest, ip, lp) -
                         IM::energy(kTest, im, lm)) / (2 * e);
        const Real rhs = IM::copper_loss(kTest, i, lr) + dW +
                         IM::torque(kTest, i, lr) * omega;
        const Real scale = std::abs(p_in) + IM::copper_loss(kTest, i, lr) +
                           std::abs(dW);
        worst = std::max(worst, std::abs(p_in - rhs) / scale);
    }
    INFO("worst relative power residual " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("InductionMachineMna: invalid parameters are refused by name",
          "[machines][induction]") {
    builder::CircuitBuilder b;
    CHECK_THROWS_WITH(
        b.add_induction_machine_mna("Mx", "a", "b", "c", "n", "w", 0.5,
                                    2e-3, 0.0, 2e-3, 60e-3, 2, 0.02),
        Catch::Matchers::ContainsSubstring("Mx") &&
            Catch::Matchers::ContainsSubstring("R_r"));
    CHECK_THROWS_WITH(
        b.add_induction_machine_mna("My", "a", "b", "c", "n", "w", 0.5,
                                    2e-3, 0.4, 2e-3, 60e-3, 1.5, 0.02),
        Catch::Matchers::ContainsSubstring("pole_pairs"));
}
