// add-mna-native-machines group 4: the flux-map PMSM law.
//
// * a map built from L_d, L_q, psi_pm reproduces the constant-L law
//   (λ, ∂λ/∂i, ∂λ/∂θ, torque and its partials) to round-off;
// * on a saturating, cross-coupled map the analytic partials of the
//   stamp (∂λ/∂i, ∂λ/∂θ_e, ∂T/∂i, ∂T/∂θ_m) equal central differences of
//   the law at random states, and the increment form equals the plain
//   difference.
// The maps are built here in power form (no fit), SYNTHETIC test values.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/models/pmsm_mna.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pulsim;
using PM = models::PmsmMna;

namespace {

// One cell over [-40, 40]² holding W' = ψ i_d + ½L_d i_d² + ½L_q i_q²
// (+ cubic cross terms when `sat`): in power form about (-40, -40).
std::shared_ptr<models::FluxMapDQ> one_cell(Real psi, Real Ld, Real Lq, bool sat) {
    auto f = std::make_shared<models::FluxMapDQ>();
    f->kd = {-40, 40};
    f->kq = {-40, 40};
    f->source = "synthetic: test";
    // Taylor coefficients of a polynomial W'(i_d, i_q) about (x0, y0) by
    // exact re-expansion: W' = Σ a_mn i_d^m i_q^n (m, n ≤ 3).
    Real a[4][4] = {};
    a[1][0] = psi; a[2][0] = Ld / 2; a[0][2] = Lq / 2;
    if (sat) {   // softening cubic + cross coupling, PD on the cell
        a[3][0] = -0.5e-6; a[2][1] = 0.75e-6; a[1][2] = -0.25e-6; a[0][3] = -1e-6;
        a[1][1] = 2e-4;
    }
    const Real x0 = -40, y0 = -40;
    auto binom = [](int n, int k) { const int t[4][4] = {{1,0,0,0},{1,1,0,0},{1,2,1,0},{1,3,3,1}}; return Real(t[n][k]); };
    f->coef.assign(16, 0);
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            for (int p = m; p < 4; ++p)
                for (int q = n; q < 4; ++q)
                    f->coef[4 * m + n] += a[p][q] * binom(p, m) * binom(q, n)
                                          * std::pow(x0, p - m) * std::pow(y0, q - n);
    f->validate();
    return f;
}

}  // namespace

TEST_CASE("flux map: a linear map is the constant-L law", "[machines][flux_map]") {
    PM::Params lin{.R_s = 0.5, .L_d = 1e-3, .L_q = 3e-3, .psi_pm = 0.05, .pole_pairs = 4};
    PM::Params fm = lin;
    fm.flux_map = one_cell(0.05, 1e-3, 3e-3, false);
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<Real> ui(-30, 30), ut(-7, 7);
    for (int k = 0; k < 100; ++k) {
        const PM::Vec3 i{ui(rng), ui(rng), ui(rng)};
        const Real th = ut(rng);
        const auto s = PM::flux_map_state(fm, th, i);
        const auto ind = PM::inductance(lin, th);
        const auto lp = PM::lambda_pm(lin, th, 0), lp1 = PM::lambda_pm(lin, th, 1);
        const auto tq = PM::torque(lin, th, i);
        for (Size r = 0; r < 3; ++r) {
            Real lam = lp[r], dth = lp1[r];
            for (Size j = 0; j < 3; ++j) {
                lam += ind.L[r][j] * i[j];
                dth += ind.dL[r][j] * i[j];
                REQUIRE(std::abs(s.dlam_di[r][j] - ind.L[r][j]) < 1e-15);
            }
            REQUIRE(std::abs(s.lam[r] - lam) < 1e-14);
            REQUIRE(std::abs(s.dlam_dth[r] - dth) < 1e-14);
            REQUIRE(std::abs(s.dT_di[r] - tq.dT_di[r]) < 1e-13);
        }
        REQUIRE(std::abs(s.T - tq.T) < 1e-12);
        REQUIRE(std::abs(s.dT_dtheta_m - tq.dT_dtheta_m) < 1e-11);
    }
}

TEST_CASE("flux map: analytic partials equal central differences", "[machines][flux_map]") {
    PM::Params p{.R_s = 0.5, .L_d = 1e-3, .L_q = 3e-3, .psi_pm = 0.05, .pole_pairs = 4};
    p.flux_map = one_cell(0.05, 1e-3, 3e-3, true);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<Real> ui(-20, 20), ut(-7, 7);
    const Real d = 1e-5;
    for (int k = 0; k < 100; ++k) {
        PM::Vec3 i{ui(rng), ui(rng), ui(rng)};
        const Real th = ut(rng);
        const auto s = PM::flux_map_state(p, th, i);
        for (Size j = 0; j < 3; ++j) {
            auto ip = i, im = i;
            ip[j] += d; im[j] -= d;
            const auto sp = PM::flux_map_state(p, th, ip), sm = PM::flux_map_state(p, th, im);
            for (Size r = 0; r < 3; ++r) {
                REQUIRE(std::abs((sp.lam[r] - sm.lam[r]) / (2 * d) - s.dlam_di[r][j]) < 1e-9);
            }
            REQUIRE(std::abs((sp.T - sm.T) / (2 * d) - s.dT_di[j]) < 1e-8);
        }
        const auto tp = PM::flux_map_state(p, th + d, i), tm = PM::flux_map_state(p, th - d, i);
        for (Size r = 0; r < 3; ++r) {
            REQUIRE(std::abs((tp.lam[r] - tm.lam[r]) / (2 * d) - s.dlam_dth[r]) < 1e-9);
        }
        // ∂T/∂θ_m = pp·∂T/∂θ_e
        REQUIRE(std::abs(p.pole_pairs * (tp.T - tm.T) / (2 * d) - s.dT_dtheta_m) < 1e-7);
        // increment form = plain difference
        const PM::Vec3 di{1e-3 * ui(rng), 1e-3 * ui(rng), 1e-3 * ui(rng)};
        const Real dth = 1e-4 * ut(rng);
        PM::Vec3 i1{i[0] + di[0], i[1] + di[1], i[2] + di[2]};
        const auto inc = PM::flux_map_increment(p, th + dth, th, dth, i, di);
        const auto s1 = PM::flux_map_state(p, th + dth, i1);
        for (Size r = 0; r < 3; ++r) {
            REQUIRE(std::abs(inc[r] - (s1.lam[r] - s.lam[r])) < 1e-15);
        }
    }
}

TEST_CASE("flux map: refusals by name", "[machines][flux_map]") {
    bool threw = false;   // L_d < 0: refused when the map is built
    try { (void)one_cell(0.05, -1e-3, 3e-3, false); } catch (const std::invalid_argument& e) {
        threw = std::string{e.what()}.find("positive definite") != std::string::npos;
    }
    REQUIRE(threw);
    PM::Params p{.L_d = 1e-3, .L_q = 3e-3};
    p.flux_map = one_cell(0.05, 1e-3, 3e-3, false);
    p.harmonics = std::make_shared<const std::vector<PM::PmHarmonic>>(
        std::vector<PM::PmHarmonic>{{1, 0.05, 0}});
    threw = false;
    try { PM::validate(p); } catch (const std::invalid_argument& e) {
        threw = std::string{e.what()}.find("harmonics") != std::string::npos;
    }
    REQUIRE(threw);
}
