#pragma once

// =============================================================================
// Pulsim — MNA-native squirrel-cage induction machine
// =============================================================================
//
// add-mna-native-machines, group 2. Stationary (αβ0) frame,
// amplitude-invariant Clarke transform C, rotor quantities referred to
// the stator, rotor currents eliminated. With L_s = L_ls + L_m,
// L_r = L_lr + L_m, σ = 1 − L_m²/(L_s·L_r), τ_r = L_r/R_r, k_r = L_m/L_r,
// ω_e = pp·ω_m and J₂ = [[0, −1], [1, 0]]:
//
//   λ_s,αβ = σ·L_s·i_s,αβ + k_r·λ_r          λ_s,0 = L_ls·i_s,0
//   v_k − v_n = R_s·i_k + dλ_k/dt,  λ_abc = C⁻¹·[λ_sα, λ_sβ, λ_s0]   (3 rows)
//   dλ_r/dt = −λ_r/τ_r + (L_m/τ_r)·i_s,αβ + ω_e·J₂·λ_r               (2 rows)
//   T_em = (3/2)·pp·k_r·(λ_rα·i_sβ − λ_rβ·i_sα)        into the speed node
//
// The stator inductance matrix in abc is CONSTANT (the stationary
// frame is the reason for the state choice), and the only
// nonlinearities are the two bilinear products ω·λ_r and λ_r × i_s:
// polynomial, C^∞, exact closed-form Jacobian.
//
// Power: with p_in = Σ_k (v_k − v_n)·i_k = (3/2)·v_s·i_s + 3·v_0·i_0,
//
//   p_in = (3/2)·(R_s|i_s|² + R_r|i_r|²) + 3·R_s·i_0² + dW/dt + T_em·ω_m
//   W    = (3/4)·(i_s·λ_s + i_r·λ_r) + (3/2)·L_ls·i_0²,
//   i_r  = (λ_r − L_m·i_s)/L_r
//
// (derived from the two voltage equations dotted with the currents;
// `power_balance` below evaluates each term and the C++ test checks
// the identity on random states).
//
// The rotor flux λ_r is the voltage of two internal nodes, each with a
// 1 F capacitor the builder adds, exactly as the PMSM's angle is a
// node: the engine integrates it with the same companion as every other
// capacitor (trapezoidal, BDF2 in the second TR-BDF2 stage), and the
// stamp injects dλ_r/dt's right-hand side as a current. The stator rows
// are the PMSM's flux rows (conductance form, `pmsm_mna_history.hpp`).

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace pulsim::models {

struct InductionMachineMna {
    struct Params {
        Real R_s        = Real{0.5};    //!< stator resistance [Ω]
        Real L_ls       = Real{2e-3};   //!< stator leakage [H]
        Real R_r        = Real{0.4};    //!< rotor resistance, referred [Ω]
        Real L_lr       = Real{2e-3};   //!< rotor leakage, referred [H]
        Real L_m        = Real{60e-3};  //!< magnetising inductance [H]
        Real pole_pairs = Real{2};      //!< pp
        Real T_load     = Real{0};      //!< constant load torque [N·m]
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = false;

    static void validate(const Params& p) {
        if (!(p.R_s >= Real{0}) || !std::isfinite(p.R_s)) {
            throw std::invalid_argument(
                "InductionMachineMna: R_s must be >= 0");
        }
        if (!(p.R_r > Real{0}) || !std::isfinite(p.R_r)) {
            throw std::invalid_argument(
                "InductionMachineMna: R_r must be > 0 (a zero rotor "
                "resistance has no rotor time constant)");
        }
        if (!(p.L_ls > Real{0}) || !(p.L_lr > Real{0}) ||
            !(p.L_m > Real{0}) || !std::isfinite(p.L_ls) ||
            !std::isfinite(p.L_lr) || !std::isfinite(p.L_m)) {
            throw std::invalid_argument(
                "InductionMachineMna: L_ls, L_lr and L_m must be > 0 "
                "(a zero leakage makes the stator and rotor fluxes "
                "dependent: sigma = 0)");
        }
        if (!(p.pole_pairs >= Real{1}) ||
            p.pole_pairs != std::floor(p.pole_pairs)) {
            throw std::invalid_argument(
                "InductionMachineMna: pole_pairs must be a positive "
                "integer");
        }
        if (!std::isfinite(p.T_load)) {
            throw std::invalid_argument(
                "InductionMachineMna: T_load must be finite");
        }
    }

    using Mat3 = std::array<std::array<Real, 3>, 3>;
    using Vec3 = std::array<Real, 3>;
    using Vec2 = std::array<Real, 2>;

    struct Derived {
        Real L_s, L_r, sigma_Ls, k_r, inv_tau_r, Lm_over_tau_r;
    };
    [[nodiscard]] static Derived derived(const Params& p) noexcept {
        const Real L_s = p.L_ls + p.L_m, L_r = p.L_lr + p.L_m;
        // σ·L_s = L_s − L_m²/L_r, written without the cancellation.
        const Real sigma_Ls = p.L_ls + p.L_m * p.L_lr / L_r;
        return {L_s, L_r, sigma_Ls, p.L_m / L_r, p.R_r / L_r,
                p.L_m * p.R_r / L_r};
    }

    /// Amplitude-invariant Clarke (rows α, β, 0) and its inverse.
    [[nodiscard]] static Mat3 clarke() noexcept {
        const Real s = Real{0.8660254037844386};   // √3/2
        const Real c = Real{2} / Real{3};
        return {{{c, -c / 2, -c / 2},
                 {Real{0}, c * s, -c * s},
                 {c / 2, c / 2, c / 2}}};
    }
    [[nodiscard]] static Mat3 clarke_inv() noexcept {
        const Real s = Real{0.8660254037844386};
        return {{{Real{1}, Real{0}, Real{1}},
                 {-Real{0.5}, s, Real{1}},
                 {-Real{0.5}, -s, Real{1}}}};
    }

    /// Constant stator matrices: λ_abc = L·i_abc + M·λ_r.
    struct StatorMatrices {
        Mat3 L;
        std::array<Vec2, 3> M;
    };
    [[nodiscard]] static StatorMatrices stator(const Params& p) noexcept {
        const auto d = derived(p);
        const Mat3 C = clarke(), Ci = clarke_inv();
        const Vec3 diag{d.sigma_Ls, d.sigma_Ls, p.L_ls};
        StatorMatrices out{};
        for (Size i = 0; i < 3; ++i) {
            for (Size j = 0; j < 3; ++j) {
                Real acc = 0;
                for (Size k = 0; k < 3; ++k) acc += Ci[i][k] * diag[k] * C[k][j];
                out.L[i][j] = acc;
            }
            out.M[i] = {Ci[i][0] * d.k_r, Ci[i][1] * d.k_r};
        }
        return out;
    }

    /// Stator flux linkage per phase at currents i and rotor flux lr.
    [[nodiscard]] static Vec3 lambda_s(const Params& p, const Vec3& i,
                                       const Vec2& lr) noexcept {
        const auto sm = stator(p);
        Vec3 lam{};
        for (Size k = 0; k < 3; ++k) {
            Real acc = sm.M[k][0] * lr[0] + sm.M[k][1] * lr[1];
            for (Size j = 0; j < 3; ++j) acc += sm.L[k][j] * i[j];
            lam[k] = acc;
        }
        return lam;
    }

    /// i_s,αβ from the phase currents.
    [[nodiscard]] static Vec2 i_ab(const Vec3& i) noexcept {
        const Mat3 C = clarke();
        return {C[0][0] * i[0] + C[0][1] * i[1] + C[0][2] * i[2],
                C[1][0] * i[0] + C[1][1] * i[1] + C[1][2] * i[2]};
    }

    /// Electromagnetic torque [N·m].
    [[nodiscard]] static Real torque(const Params& p, const Vec3& i,
                                     const Vec2& lr) noexcept {
        const auto d = derived(p);
        const Vec2 is = i_ab(i);
        return Real{1.5} * p.pole_pairs * d.k_r
               * (lr[0] * is[1] - lr[1] * is[0]);
    }

    /// dλ_r/dt at (i, λ_r, ω_m).
    [[nodiscard]] static Vec2 rotor_rhs(const Params& p, const Vec3& i,
                                        const Vec2& lr,
                                        Real omega_m) noexcept {
        const auto d = derived(p);
        const Vec2 is = i_ab(i);
        const Real we = p.pole_pairs * omega_m;
        return {-d.inv_tau_r * lr[0] + d.Lm_over_tau_r * is[0] - we * lr[1],
                -d.inv_tau_r * lr[1] + d.Lm_over_tau_r * is[1] + we * lr[0]};
    }

    /// Stored magnetic energy [J] (see the header comment).
    [[nodiscard]] static Real energy(const Params& p, const Vec3& i,
                                     const Vec2& lr) noexcept {
        const auto d = derived(p);
        const Vec2 is = i_ab(i);
        const Real i0 = (i[0] + i[1] + i[2]) / Real{3};
        const Vec2 ls{d.sigma_Ls * is[0] + d.k_r * lr[0],
                      d.sigma_Ls * is[1] + d.k_r * lr[1]};
        const Vec2 ir{(lr[0] - p.L_m * is[0]) / d.L_r,
                      (lr[1] - p.L_m * is[1]) / d.L_r};
        return Real{0.75} * (is[0] * ls[0] + is[1] * ls[1]
                             + ir[0] * lr[0] + ir[1] * lr[1])
               + Real{1.5} * p.L_ls * i0 * i0;
    }

    /// Copper loss [W] at (i, λ_r): stator + rotor (+ zero sequence).
    [[nodiscard]] static Real copper_loss(const Params& p, const Vec3& i,
                                          const Vec2& lr) noexcept {
        const auto d = derived(p);
        const Vec2 is = i_ab(i);
        const Real i0 = (i[0] + i[1] + i[2]) / Real{3};
        const Vec2 ir{(lr[0] - p.L_m * is[0]) / d.L_r,
                      (lr[1] - p.L_m * is[1]) / d.L_r};
        return Real{1.5} * (p.R_s * (is[0] * is[0] + is[1] * is[1])
                            + p.R_r * (ir[0] * ir[0] + ir[1] * ir[1]))
               + Real{3} * p.R_s * i0 * i0;
    }
};

}  // namespace pulsim::models
