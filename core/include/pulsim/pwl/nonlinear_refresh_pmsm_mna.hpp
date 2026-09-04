#pragma once

// =============================================================================
// Pulsim — Newton stamp for the MNA-native PMSM
// =============================================================================
//
// Per phase k (branch phase_k → neutral, current unknown i_k), the
// trapezoidal rule on v_k = R i_k + dλ_k/dt is the constraint row
//
//     R_k = (v_k − R i_k) + (v_k,old − R i_k,old)
//           − (2/h)·(λ_k(θ, i) − λ_k,old) = 0
//
// with λ_k(θ, i) = Σ_j L_kj(θ_e) i_j + λ_pm,k(θ_e) evaluated at the
// CURRENT Newton iterate — the θ transformation is stamped per
// iteration, so the coupling has no lag. Its partials:
//
//     ∂R_k/∂v_phase_k = +1        ∂R_k/∂v_neutral = −1
//     ∂R_k/∂i_j       = −R δ_kj − (2/h) L_kj(θ_e)
//     ∂R_k/∂θ_m       = −(2/h)·pp·[ Σ_j dL_kj/dθ_e i_j + dλ_pm,k/dθ_e ]
//
// KCL at the terminals is the same ±1 pattern as any inductor.
//
// UNDER THE TR-BDF2 SECOND STAGE the residual changes SHAPE, not
// just a coefficient: the derivative is one-sided, so the previous
// step's dλ/dt is absent altogether,
//
//     R_k = (v_k − R i_k) − (c1·λ_k + c2·λ_γ,k + c3·λ_n,k)/h
//
// with ∂R_k/∂i_j = −R δ_kj − (c1/h)·L_kj and
// ∂R_k/∂θ_m = −(c1/h)·pp·[Σ_j dL_kj/dθ_e i_j + dλ_pm,k/dθ_e].
// Since c1/h == 2/(γh), the CONDUCTANCE block is identical to the
// trapezoidal one — which is exactly what lets both stages reuse a
// single matrix factor, and exactly why stamping the wrong history
// term converges and lies.
//
// The mechanical couplings are algebraic (a torque injected into
// the ω node, ω injected into the θ node), so they carry no
// derivative and are stage-INDEPENDENT. The ω and θ capacitors are
// ordinary linear ones and the engine already integrates those in
// both stages.
//
// Mechanics: the ω and θ nodes carry linear capacitors (J and 1 F,
// added by the builder), so only the two couplings are stamped here:
//
//     ω node:  injected current  T_em(i, θ) − T_load
//              ∂/∂i_j = ∂T/∂i_j,   ∂/∂θ_m = ∂T/∂θ_m
//     θ node:  injected current  ω          (∂/∂ω = 1)
//
// Sign: the residual convention here is f = (sum of currents
// LEAVING the node); an injection INTO a node therefore subtracts.
// Torque comes from the co-energy of the same L(θ) that sits in the
// electrical rows, which is what makes the reluctance torque appear
// without being added on.

#include "pulsim/models/pmsm_mna.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

#include <array>
#include <cmath>

namespace pulsim::pwl {

/// Stamp every MNA-native PMSM. Returns the largest |i_phase|
/// stamped, for the Newton residual indicator.
inline Real refresh_pmsm_mna(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const PmsmMnaHistory& history,
    Real h,
    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) {
    Real max_i = Real{0};
    if (history.empty() || !(h > Real{0})) {
        return max_i;
    }
    const auto kc = trbdf2_coeffs();
    const bool bdf2 = (stage == TrBdf2Stage::Bdf2Stage2);
    // c1/h in the BDF2 stage, 2/h in the trapezoidal one. The two
    // coincide when h is the stage's own step, which is what makes
    // one factor serve both.
    const Real d_coef = bdf2 ? kc.c1 / h : Real{2} / h;
    for (const auto& e : history.entries()) {
        const auto& p = e.params;
        const Real theta_e = PmsmMnaHistory::theta_e_of(e, x);
        const auto ind = models::PmsmMna::inductance(p, theta_e);
        const auto lpm0 = models::PmsmMna::lambda_pm(p, theta_e, 0);
        const auto lpm1 = models::PmsmMna::lambda_pm(p, theta_e, 1);

        std::array<Real, 3> i{}, v{};
        const Real v_n = stamping::read_node_voltage(x, e.neutral);
        for (Size k = 0; k < 3; ++k) {
            i[k] = x[e.cur_row[k]];
            v[k] = stamping::read_node_voltage(x, e.phase_node[k]) - v_n;
        }
        const bool n_act = stamping::node_is_active(e.neutral);
        const bool th_act = stamping::node_is_active(e.theta_node);
        const bool om_act = stamping::node_is_active(e.omega_node);

        // ---- electrical rows ----------------------------------
        for (Size k = 0; k < 3; ++k) {
            Real lam = lpm0[k], dlam_dth = lpm1[k];
            for (Size j = 0; j < 3; ++j) {
                lam += ind.L[k][j] * i[j];
                dlam_dth += ind.dL[k][j] * i[j];
            }
            // Trapezoidal carries the previous derivative; BDF2 is
            // one-sided and does not.
            const Real hist =
                bdf2 ? (kc.c2 * e.lambda_gamma[k]
                        + kc.c3 * e.lambda_old[k]) / h
                     : -e.dlam_old[k] - (Real{2} / h) * e.lambda_old[k];
            const Real R_row =
                (v[k] - p.R_s * i[k]) - d_coef * lam - hist;
            const Index row = e.cur_row[k];
            const bool ph_act = stamping::node_is_active(e.phase_node[k]);

            f_nl[row] += R_row;
            if (ph_act) J_nl.coeffRef(row, e.phase_node[k]) += Real{1};
            if (n_act)  J_nl.coeffRef(row, e.neutral) -= Real{1};
            for (Size j = 0; j < 3; ++j) {
                Real d = -d_coef * ind.L[k][j];
                if (j == k) d -= p.R_s;
                J_nl.coeffRef(row, e.cur_row[j]) += d;
            }
            if (th_act) {
                J_nl.coeffRef(row, e.theta_node)
                    += -d_coef * p.pole_pairs * dlam_dth;
            }

            // KCL at the terminals: i_k leaves the phase node and
            // enters the neutral.
            if (ph_act) {
                f_nl[e.phase_node[k]] += i[k];
                J_nl.coeffRef(e.phase_node[k], row) += Real{1};
            }
            if (n_act) {
                f_nl[e.neutral] -= i[k];
                J_nl.coeffRef(e.neutral, row) -= Real{1};
            }
            max_i = std::max(max_i, std::abs(i[k]));
        }

        // ---- mechanical couplings ------------------------------
        const auto tq = models::PmsmMna::torque(p, theta_e, i);
        if (om_act) {
            // Injection INTO the ω node: T_em − T_load.
            f_nl[e.omega_node] -= (tq.T - p.T_load);
            for (Size j = 0; j < 3; ++j) {
                J_nl.coeffRef(e.omega_node, e.cur_row[j]) -= tq.dT_di[j];
            }
            if (th_act) {
                J_nl.coeffRef(e.omega_node, e.theta_node) -= tq.dT_dtheta_m;
            }
        }
        if (th_act && om_act) {
            // dθ/dt = ω: inject ω into the θ node's 1 F capacitor.
            const Real omega = stamping::read_node_voltage(x, e.omega_node);
            f_nl[e.theta_node] -= omega;
            J_nl.coeffRef(e.theta_node, e.omega_node) -= Real{1};
        }
    }
    return max_i;
}

}  // namespace pulsim::pwl
