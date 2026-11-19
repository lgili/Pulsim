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
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <format>
#include <limits>
#include <string>
#include <memory>
#include <random>
#include <vector>

using namespace pulsim;
using IM = models::InductionMachineMna;
using pwl::TrBdf2Stage;

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
    std::string where;
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
            // The residual is at most bilinear in x, so a central
            // difference is exact but for roundoff (eps·|f|/d, |f| ~ 1e7
            // here): a large step, not a small one.
            const Real d = 1e-2 * std::max(Real{1}, std::abs(x[c]));
            Vector xp = x, xm = x;
            xp[c] += d;
            xm[c] -= d;
            const Vector col = (residual(xp) - residual(xm)) / (2 * d);
            const Real scale = std::max(Real{1}, col.cwiseAbs().maxCoeff());
            Index row = 0;
            const Real e =
                (col - Jd.col(c)).cwiseAbs().maxCoeff(&row) / scale;
            if (e > worst) {
                worst = e;
                where = std::format("row {} col {} stage {} fd {} J {}", row,
                                    c, trial % 3, col[row], Jd(row, c));
            }
        }
    }
    INFO("worst relative column error " << worst << " at " << where);
    CHECK(worst < 1e-6);   // measured 1.6e-7 (roundoff)
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

// -----------------------------------------------------------------------------
// Increment form of the flux rows (add-mna-native-machines, small-step fix).
// The row multiplies λ(x) − λ_old by 2/h; written as a difference of two
// absolute fluxes its round-off is eps·(2/h)·|λ| ≈ 2e-9 V at h = 10 ns,
// ψ = 0.05 — above Newton's absolute 1e-9 V, so a PMSM at 10 ns aborted.
// `lambda_increment` takes Δθ and Δi and must (a) equal the difference for
// large increments and (b) keep full relative accuracy for tiny ones, where
// the plain difference has none.
// -----------------------------------------------------------------------------

namespace {
using PM = models::PmsmMna;

PM::Vec3 lambda_abs(const PM::Params& p, Real th, const PM::Vec3& i) {
    const auto ind = PM::inductance(p, th);
    auto lam = PM::lambda_pm(p, th, 0);
    for (Size k = 0; k < 3; ++k)
        for (Size j = 0; j < 3; ++j) lam[k] += ind.L[k][j] * i[j];
    return lam;
}
}  // namespace

TEST_CASE("PmsmMna: lambda_increment is the flux difference without "
          "cancellation", "[machines][pmsm]") {
    PM::Params plain{.R_s = 0.5, .L_d = 1e-3, .L_q = 3e-3, .psi_pm = 0.05,
                     .pole_pairs = 4};
    PM::Params harm = plain;
    harm.harmonics = std::make_shared<const std::vector<PM::PmHarmonic>>(
        std::vector<PM::PmHarmonic>{{1, 0.05, 0.1}, {5, 0.004, -0.7},
                                    {7, 0.002, 0.3}});
    std::mt19937 rng(5);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    Real worst_big = 0, worst_tiny = 0;
    for (const PM::Params* p : {&plain, &harm}) {
        for (int trial = 0; trial < 100; ++trial) {
            const Real th0 = 50 * u(rng);
            const PM::Vec3 i0{20 * u(rng), 20 * u(rng), 20 * u(rng)};
            // (a) large increments: equal to the plain difference.
            const Real dth = 2 * u(rng);
            const PM::Vec3 di{5 * u(rng), 5 * u(rng), 5 * u(rng)};
            const PM::Vec3 i1{i0[0] + di[0], i0[1] + di[1], i0[2] + di[2]};
            const Real th1 = th0 + dth;
            const auto inc = PM::lambda_increment(*p, PM::inductance(*p, th1).L,
                                                  th1, th0, th1 - th0, i0,
                                                  PM::Vec3{i1[0] - i0[0],
                                                           i1[1] - i0[1],
                                                           i1[2] - i0[2]});
            const auto l1 = lambda_abs(*p, th1, i1), l0 = lambda_abs(*p, th0, i0);
            for (Size k = 0; k < 3; ++k) {
                worst_big = std::max(worst_big,
                                     std::abs(inc[k] - (l1[k] - l0[k])));
            }
            // (b) tiny increments: the first-order expansion
            //     dλ = L·di + (dL·i0 + dλ_pm)·dθ, relative to |dλ|.
            const Real e = 1e-12;
            const PM::Vec3 dt{e * u(rng), e * u(rng), e * u(rng)};
            const Real dthe = e * u(rng);
            const auto tiny = PM::lambda_increment(
                *p, PM::inductance(*p, th0 + dthe).L, th0 + dthe, th0, dthe,
                i0, dt);
            const auto ind = PM::inductance(*p, th0);
            const auto lp1 = PM::lambda_pm(*p, th0, 1);
            for (Size k = 0; k < 3; ++k) {
                Real lin = lp1[k] * dthe, mag = std::abs(lp1[k] * dthe);
                for (Size j = 0; j < 3; ++j) {
                    lin += ind.L[k][j] * dt[j] + ind.dL[k][j] * i0[j] * dthe;
                    mag += std::abs(ind.L[k][j] * dt[j])
                           + std::abs(ind.dL[k][j] * i0[j] * dthe);
                }
                worst_tiny = std::max(worst_tiny, std::abs(tiny[k] - lin) / mag);
            }
        }
    }
    INFO("large: worst |inc − diff| " << worst_big
         << " Wb; tiny: worst relative error " << worst_tiny);
    CHECK(worst_big < 1e-12);    // expected ~1e-15 (eps·|θ|·ψ, |θ| ≈ 52): 1000× margin
    CHECK(worst_tiny < 1e-6);    // the plain difference: ~1e-17/1e-14 ≈ 1e-3..1
}

TEST_CASE("PMSM / induction flux rows: the residual at the committed state "
          "carries no eps·(2/h)·|λ| term", "[machines][pmsm][induction]") {
    // At x = the committed point every flux increment is exactly 0, so the
    // rows reduce to (v − R i) [+ dλ_n/dt]: trapezoidal 2·dλ_n, backward
    // Euler and BDF2 (after capture_gamma at the same x) dλ_n. A
    // non-incremental row leaves (2/h)·(λ(x) − λ_old) = (2/h)·O(eps·|λ|)
    // here (~1e-4 V at h = 1e-10 with these states). 4 eps: FMA contraction
    // may differ between the two sites that form v − R i.
    builder::CircuitBuilder b;
    b.add_induction_machine_mna("M", "a", "b", "c", "n", "w", kTest.R_s,
                                kTest.L_ls, kTest.R_r, kTest.L_lr, kTest.L_m,
                                kTest.pole_pairs, 0.02, 0.0, kTest.T_load);
    b.add_pmsm_mna("P", "pa", "pb", "pc", "pn", "pw", "pth", 0.5, 1e-3, 3e-3,
                   0.5, 4, 1e-3, 1e-4);
    const auto N = static_cast<Index>(b.pool().state_size(b.graph()));
    pwl::PmsmMnaHistory hist;
    hist.init(b.graph(), b.pool());
    std::mt19937 rng(3);
    const Vector x = random_state(static_cast<Size>(N), rng);
    constexpr Real eps = std::numeric_limits<Real>::epsilon();
    auto rows = [&](Real h, TrBdf2Stage st) {
        sparse::Matrix J(N, N);
        Vector f = Vector::Zero(N);
        pwl::refresh_pmsm_mna(x, J, f, hist, h, st);
        return f;
    };
    hist.update_from_state(x, 1e-8);
    hist.capture_gamma(x);
    for (const Real h : {1e-6, 1e-8, 1e-10}) {
        const Vector ft = rows(h, TrBdf2Stage::Trapezoidal);
        const Vector fb = rows(h, TrBdf2Stage::BackwardEuler);
        const Vector f2 = rows(h, TrBdf2Stage::Bdf2Stage2);
        for (const auto& e : hist.entries()) {
            for (Size k = 0; k < 3; ++k) {
                const Real d = e.dlam_old[k], tol = 4 * eps * std::abs(d);
                CHECK(std::abs(ft[e.cur_row[k]] - 2 * d) <= 2 * tol);
                CHECK(std::abs(fb[e.cur_row[k]] - d) <= tol);
                CHECK(std::abs(f2[e.cur_row[k]] - d) <= tol);
            }
        }
    }
    // SDIRK pseudo-history: after extrapolate_previous(x1, c) the row at
    // the reference reads dλ_n + lam_off/h = dλ_n + c·(λ(x1) − λ(x))/h, the old form's
    // value, up to the increment's own rounding.
    const Vector x1 = random_state(static_cast<Size>(N), rng);
    const auto lam0 = [&] {
        std::vector<std::array<Real, 3>> v;
        for (const auto& e : hist.entries()) v.push_back(pwl::PmsmMnaHistory::lambda_at(e, x));
        return v;
    }();
    constexpr Real c = 0.3, h = 1e-6;
    hist.extrapolate_previous(x1, c);
    const Vector fb = rows(h, TrBdf2Stage::BackwardEuler);
    Size m = 0;
    for (const auto& e : hist.entries()) {
        const auto l1 = pwl::PmsmMnaHistory::lambda_at(e, x1);
        for (Size k = 0; k < 3; ++k) {
            const Real expect = e.dlam_old[k] + c * (l1[k] - lam0[m][k]) / h;
            CHECK(std::abs(fb[e.cur_row[k]] - expect)
                  <= 1e-9 * (std::abs(expect) + std::abs(l1[k]) / h));
        }
        ++m;
    }
    // Resume: rebase on the same x keeps the stored flux (offset 0).
    hist.update_from_state(x, 1e-8);
    hist.rebase_reference(x);
    for (const auto& e : hist.entries()) {
        for (Size k = 0; k < 3; ++k) CHECK(e.lam_off[k] == 0.0);
    }
}

TEST_CASE("PMSM DC operating point: held mechanics, back-EMF of the held "
          "speed (add-mna-native-machines D19)",
          "[layer4_v3][machines][dc]") {
    // Round rotor at ω0 = 50 rad/s, θ0 = 0.2 rad, DC phase sources:
    // i_k = (v_k − v_n + ω_e·ψ·sin(θ_e − 2πk/3))/R_s, Σ i = 0. Also runs
    // the row replacement (Eigen prune) of the held speed / angle nodes.
    const Real R = 0.5, psi = 0.05, pp = 4, w0 = 50, th0 = 0.2;
    const std::array<Real, 3> v{10.0, -4.0, -6.0};
    builder::CircuitBuilder b;
    b.add_voltage_source("Va", "ua", "gnd", v[0]);
    b.add_voltage_source("Vb", "ub", "gnd", v[1]);
    b.add_voltage_source("Vc", "uc", "gnd", v[2]);
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R, 2e-3, 2e-3,
                   psi, pp, /*J=*/1.0, /*B=*/1e-3, 0, 0, w0, th0);
    const topology::SwitchStateMask mask(0);
    const Vector x = pwl::compute_dc_op(b.graph(), b.pool(), mask);
    std::array<Real, 3> u{};
    Real mean = 0;
    for (Size k = 0; k < 3; ++k) {
        u[k] = v[k] + pp * w0 * psi
                          * std::sin(pp * th0 - 2.0 * std::numbers::pi / 3.0 * k);
        mean += u[k] / 3.0;
    }
    const auto& g = b.graph();
    for (Size k = 0; k < 3; ++k) {
        const Index bid = b.branch_id_of(std::string{"M1_"} + "abc"[k]);
        const Real i = x[b.pool().branch_var_id_for_inductor(bid, g)];
        CHECK(std::abs(i - (u[k] - mean) / R) < 1e-9);
    }
    CHECK(std::abs(x[b.node_id_of("w")] - w0) < 1e-12);
    CHECK(std::abs(x[b.node_id_of("th")] - th0) < 1e-12);
}

TEST_CASE("A machine speed node driven by a source is refused at DC by name",
          "[layer4_v3][machines][dc]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Va", "ua", "gnd", 1.0);
    b.add_voltage_source("Vb", "ub", "gnd", 0.0);
    b.add_voltage_source("Vc", "uc", "gnd", 0.0);
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", 0.5, 2e-3, 2e-3,
                   0.05, 4, 1.0);
    b.add_voltage_source("Vw", "w", "gnd", 10.0);
    const topology::SwitchStateMask mask(0);
    CHECK_THROWS_WITH(pwl::compute_dc_op(b.graph(), b.pool(), mask),
                      Catch::Matchers::ContainsSubstring("Vw"));
}
