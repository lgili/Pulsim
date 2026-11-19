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

namespace detail {

/// The flux row from λ − λ_n = dλ_inc − lam_off (increment form, see
/// `PmsmMnaHistory::Entry`): the same residual as
///   trapezoidal  (v − R i) + (dλ/dt)_n − (2/h)(λ − λ_n)
///   BDF2         (v − R i) − (c1 λ + c2 λ_γ + c3 λ_n)/h
///                = (v − R i) − (c1 (λ − λ_n) + c2 (λ_γ − λ_n))/h
///                  (c1 + c2 + c3 = 0)
///   backward E.  (v − R i) − (λ − λ_n)/h
/// with every flux term an increment, so its round-off is eps·|Δλ|·d.
inline Real flux_row_(Real v_minus_ri, Real dlam_inc,
                      const PmsmMnaHistory::Entry& e, Size k, Real d_coef,
                      bool bdf2, bool be, Real h, const TrBdf2Coeffs& kc) {
    const Real dl = dlam_inc - e.lam_off[k];
    if (bdf2) return v_minus_ri - (kc.c1 * dl + kc.c2 * e.dlam_gamma[k]) / h;
    if (be) return v_minus_ri - d_coef * dl;
    return v_minus_ri + e.dlam_old[k] - d_coef * dl;
}

/// Induction-machine entry (add-mna-native-machines group 2).
///
/// Stator row k: the PMSM's flux row with λ_k = Σ_j L_kj i_j + Σ_r M_kr λ_r
/// (L, M constant), so ∂R_k/∂i_j = −R_s δ_kj − d·L_kj and
/// ∂R_k/∂λ_r = −d·M_kr; the history term is the PMSM's.
///
/// Rotor-flux node n (1 F capacitor added by the builder; the engine
/// integrates C·dλ/dt): injected current g = dλ_r/dt's right-hand side,
///   g_α = −λ_α/τ_r + (L_m/τ_r)·i_sα − ω_e·λ_β
///   g_β = −λ_β/τ_r + (L_m/τ_r)·i_sβ + ω_e·λ_α,   ω_e = pp·ω
/// residual f −= g. Speed node: f −= T_em − T_load. Stage-independent
/// (algebraic) like the PMSM's mechanical couplings.
inline Real refresh_induction_entry_(
    const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
    const PmsmMnaHistory::Entry& e, Real d_coef, bool bdf2, bool be,
    Real h) {
    using IM = models::InductionMachineMna;
    const auto& p = e.im;
    const auto kc = trbdf2_coeffs();
    const auto& sm = e.im_sm;   // built once per run (init)
    const auto dv = IM::derived(p);
    const auto C = IM::clarke();
    std::array<Real, 3> i{}, v{};
    const Real v_n = stamping::read_node_voltage(x, e.neutral);
    for (Size k = 0; k < 3; ++k) {
        i[k] = x[e.cur_row[k]];
        v[k] = stamping::read_node_voltage(x, e.phase_node[k]) - v_n;
    }
    const std::array<Real, 2> lr{
        stamping::read_node_voltage(x, e.rotor_node[0]),
        stamping::read_node_voltage(x, e.rotor_node[1])};
    const bool n_act = stamping::node_is_active(e.neutral);
    const bool om_act = stamping::node_is_active(e.omega_node);
    const std::array<bool, 2> r_act{
        stamping::node_is_active(e.rotor_node[0]),
        stamping::node_is_active(e.rotor_node[1])};
    const auto dlam = PmsmMnaHistory::delta_lambda(e, x);
    Real max_i = Real{0};

    // ---- stator rows (conductance form, increment form as the PMSM's)
    for (Size k = 0; k < 3; ++k) {
        const Real R_row = flux_row_(v[k] - p.R_s * i[k], dlam[k], e, k,
                                     d_coef, bdf2, be, h, kc);
        const Index row = e.cur_row[k];
        const bool ph_act = stamping::node_is_active(e.phase_node[k]);
        f_nl[row] += R_row;
        if (ph_act) J_nl.coeffRef(row, e.phase_node[k]) += Real{1};
        if (n_act)  J_nl.coeffRef(row, e.neutral) -= Real{1};
        for (Size j = 0; j < 3; ++j) {
            Real d = -d_coef * sm.L[k][j];
            if (j == k) d -= p.R_s;
            J_nl.coeffRef(row, e.cur_row[j]) += d;
        }
        for (Size r = 0; r < 2; ++r) {
            if (r_act[r]) {
                J_nl.coeffRef(row, e.rotor_node[r]) += -d_coef * sm.M[k][r];
            }
        }
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

    // ---- rotor-flux nodes ---------------------------------------
    const Real omega =
        om_act ? stamping::read_node_voltage(x, e.omega_node) : Real{0};
    const auto g = IM::rotor_rhs(p, i, lr, omega);
    const Real pp = p.pole_pairs, we = pp * omega;
    // ∂g_n/∂λ_m, ∂g_n/∂ω
    const Real dg_dl[2][2] = {{-dv.inv_tau_r, -we}, {we, -dv.inv_tau_r}};
    const Real dg_dw[2] = {-pp * lr[1], pp * lr[0]};
    for (Size n = 0; n < 2; ++n) {
        if (!r_act[n]) continue;
        const Index rn = e.rotor_node[n];
        const Real cn = e.c_rotor[n];   // C·dλ/dt = C·g on the node
        f_nl[rn] -= cn * g[n];
        for (Size m = 0; m < 2; ++m) {
            if (r_act[m]) J_nl.coeffRef(rn, e.rotor_node[m]) -= cn * dg_dl[n][m];
        }
        for (Size j = 0; j < 3; ++j) {
            J_nl.coeffRef(rn, e.cur_row[j]) -= cn * (dv.Lm_over_tau_r * C[n][j]);
        }
        if (om_act) J_nl.coeffRef(rn, e.omega_node) -= cn * dg_dw[n];
    }

    // ---- torque into the speed node -----------------------------
    if (om_act) {
        const Real kt = Real{1.5} * pp * dv.k_r;
        const auto is = IM::i_ab(i);
        const Real T = kt * (lr[0] * is[1] - lr[1] * is[0]);
        f_nl[e.omega_node] -= (T - p.T_load);
        for (Size j = 0; j < 3; ++j) {
            J_nl.coeffRef(e.omega_node, e.cur_row[j])
                -= kt * (lr[0] * C[1][j] - lr[1] * C[0][j]);
        }
        if (r_act[0]) J_nl.coeffRef(e.omega_node, e.rotor_node[0]) -= kt * is[1];
        if (r_act[1]) J_nl.coeffRef(e.omega_node, e.rotor_node[1]) += kt * is[0];
    }
    return max_i;
}

/// Flux-map PMSM entry (group 4): the constant-L stamp's rows with
/// λ = T⁻¹·∇W' (`PmsmMna::flux_map_state`): ∂R_k/∂i_j = −R δ_kj − d·∂λ_k/∂i_j,
/// ∂R_k/∂θ_m = −d·pp·∂λ_k/∂θ_e; torque and θ couplings as the constant-L ones.
inline Real refresh_flux_map_entry_(
    const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
    const PmsmMnaHistory::Entry& e, Real d_coef, bool bdf2, bool be,
    Real h) {
    const auto& p = e.params;
    const auto kc = trbdf2_coeffs();
    const Real theta_e = PmsmMnaHistory::theta_e_of(e, x);
    std::array<Real, 3> i{}, v{};
    const Real v_n = stamping::read_node_voltage(x, e.neutral);
    for (Size k = 0; k < 3; ++k) {
        i[k] = x[e.cur_row[k]];
        v[k] = stamping::read_node_voltage(x, e.phase_node[k]) - v_n;
    }
    const auto st = models::PmsmMna::flux_map_state(p, theta_e, i);
    const auto dlam = PmsmMnaHistory::delta_lambda(e, x);
    const bool n_act = stamping::node_is_active(e.neutral);
    const bool th_act = stamping::node_is_active(e.theta_node);
    const bool om_act = stamping::node_is_active(e.omega_node);
    Real max_i = Real{0};
    for (Size k = 0; k < 3; ++k) {
        const Real R_row = flux_row_(v[k] - p.R_s * i[k], dlam[k], e, k,
                                     d_coef, bdf2, be, h, kc);
        const Index row = e.cur_row[k];
        const bool ph_act = stamping::node_is_active(e.phase_node[k]);
        f_nl[row] += R_row;
        if (ph_act) J_nl.coeffRef(row, e.phase_node[k]) += Real{1};
        if (n_act)  J_nl.coeffRef(row, e.neutral) -= Real{1};
        for (Size j = 0; j < 3; ++j) {
            Real d = -d_coef * st.dlam_di[k][j];
            if (j == k) d -= p.R_s;
            J_nl.coeffRef(row, e.cur_row[j]) += d;
        }
        if (th_act) {
            J_nl.coeffRef(row, e.theta_node) += -d_coef * p.pole_pairs * st.dlam_dth[k];
        }
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
    if (om_act) {
        f_nl[e.omega_node] -= (st.T - p.T_load);
        for (Size j = 0; j < 3; ++j) {
            J_nl.coeffRef(e.omega_node, e.cur_row[j]) -= st.dT_di[j];
        }
        if (th_act) J_nl.coeffRef(e.omega_node, e.theta_node) -= st.dT_dtheta_m;
    }
    if (th_act && om_act) {
        const Real omega = stamping::read_node_voltage(x, e.omega_node);
        f_nl[e.theta_node] -= e.c_theta * omega;
        J_nl.coeffRef(e.theta_node, e.omega_node) -= e.c_theta;
    }
    return max_i;
}

}  // namespace detail

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
    // Backward Euler over h (restart stage): 1/h, no dλ_n.
    const bool be = (stage == TrBdf2Stage::BackwardEuler);
    const Real d_coef = bdf2 ? kc.c1 / h : be ? Real{1} / h : Real{2} / h;
    for (const auto& e : history.entries()) {
        if (e.induction) {
            max_i = std::max(max_i, detail::refresh_induction_entry_(
                                        x, J_nl, f_nl, e, d_coef, bdf2, be, h));
            continue;
        }
        if (e.params.flux_map) {
            max_i = std::max(max_i, detail::refresh_flux_map_entry_(
                                        x, J_nl, f_nl, e, d_coef, bdf2, be, h));
            continue;
        }
        const auto& p = e.params;
        const Real theta_e = PmsmMnaHistory::theta_e_of(e, x);
        const auto ind = models::PmsmMna::inductance(p, theta_e);
        const auto dlam = PmsmMnaHistory::delta_lambda(e, x, &ind.L);
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
            Real dlam_dth = lpm1[k];
            for (Size j = 0; j < 3; ++j) dlam_dth += ind.dL[k][j] * i[j];
            const Real R_row = detail::flux_row_(
                v[k] - p.R_s * i[k], dlam[k], e, k, d_coef, bdf2, be, h, kc);
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
            f_nl[e.theta_node] -= e.c_theta * omega;   // C·dθ/dt = C·ω
            J_nl.coeffRef(e.theta_node, e.omega_node) -= e.c_theta;
        }
    }
    return max_i;
}

}  // namespace pulsim::pwl
