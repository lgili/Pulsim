// add-parameter-sensitivities group 5: the stage combinations ψ of the
// sensitivity recurrence (step/sensitivity.hpp) reproduce each scheme's own
// stage derivative, q̇ = a·(q − ψ), from the formulas the engines stamp.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>

#include "pulsim/solver/step/integrators.hpp"
#include "pulsim/solver/step/sensitivity.hpp"

using Catch::Matchers::WithinRel;
using pulsim::Real;
namespace step = pulsim::step;
namespace pwl = pulsim::pwl;

namespace {
Real qdot(const step::StagePsi& p, Real q, Real qn, Real qdn, Real q0) {
    const Real psi = p.c_qn * qn + p.c_dn * qdn + (p.n_prev > 0 ? p.c_q[0] * q0 : 0);
    return p.a * (q - psi);
}
}  // namespace

TEST_CASE("sensitivity ψ: trapezoid, backward Euler, TR-BDF2 stage 2, SDIRK",
          "[step][sensitivity]") {
    const Real h = 3e-6, qn = 1.7, qdn = -0.4, q = 1.69, qg = 1.695;
    // trapezoid over h: q − q_n = (h/2)(q̇_n + q̇)
    const auto tr = step::psi_trapezoidal(h);
    REQUIRE_THAT(qdot(tr, q, qn, qdn, 0), WithinRel(2 * (q - qn) / h - qdn, 1e-12));
    // backward Euler on the factor 2h: q̇ = (q − q_n)/h, no derivative slot
    const auto be = step::psi_backward_euler_on(2 * h);
    REQUIRE_THAT(qdot(be, q, qn, qdn, 0), WithinRel((q - qn) / h, 1e-12));
    // TR-BDF2 stage 2 on the factor γh: q̇_1 = (c1 q_1 + c2 q_γ + c3 q_n)/h
    const auto s0 = step::TrBdf2::stage(h, 0);
    const auto b2 = step::psi_bdf2_stage2(s0.factor_dt);
    const Real c1 = pwl::HistoryState::trbdf2_c1(), c2 = pwl::HistoryState::trbdf2_c2(),
               c3 = pwl::HistoryState::trbdf2_c3();
    REQUIRE_THAT(qdot(b2, q, qn, qdn, qg), WithinRel((c1 * q + c2 * qg + c3 * qn) / h, 1e-12));
    // SDIRK(2,2) stage 2: BE of γ_s h from q_n + r (q_1 − q_n)
    const Real gs = step::Sdirk22::gamma(), r = (1 - gs) / gs;
    const auto sd = step::psi_sdirk_stage2(2 * gs * h, r);
    REQUIRE_THAT(qdot(sd, q, qn, qdn, qg),
                 WithinRel((q - (qn + r * (qg - qn))) / (gs * h), 1e-12));
}

TEST_CASE("sensitivity ψ: ESDIRK4(3)6L implicit stages reproduce the Butcher rows",
          "[step][sensitivity]") {
    // y' = λy: stage s solves Y_s = y0 + h(a_{s+1,0} ẏ0 + Σ_j a_{s+1,j+1} Ẏ_j
    // + γ Ẏ_s); the recurrence's ψ_s with Ẏ_j = a_j (Y_j − ψ_j) must give
    // ẏ = a (Y − ψ) = λ Y at every stage.
    using E = step::Esdirk43;
    const Real h = 0.1, lam = -2.3, y0 = 1.1, yd0 = lam * y0;
    std::array<Real, E::n_stages> Y{}, Yd{};
    for (int s = 0; s < E::n_stages; ++s) {
        const auto p = step::psi_esdirk43(s, h);
        Real psi = p.c_qn * y0 + p.c_dn * yd0;
        for (int j = 0; j < p.n_prev; ++j) {
            psi += p.c_q[static_cast<std::size_t>(j)] * Y[static_cast<std::size_t>(j)]
                 + p.c_d[static_cast<std::size_t>(j)] * Yd[static_cast<std::size_t>(j)];
        }
        // a (Y − ψ) = λ Y  →  Y = a ψ / (a − λ)
        const auto si = static_cast<std::size_t>(s);
        Y[si] = p.a * psi / (p.a - lam);
        Yd[si] = p.a * (Y[si] - psi);
        REQUIRE_THAT(Yd[si], WithinRel(lam * Y[si], 1e-12));
    }
    // stiffly accurate, 4th order: the last stage ≈ y0·e^{λh}
    REQUIRE_THAT(Y[E::n_stages - 1], WithinRel(y0 * std::exp(lam * h), 1e-4));
}
