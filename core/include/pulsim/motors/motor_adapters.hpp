// SPDX-License-Identifier: MIT
//
// Pulsim — Phase D / C++ port: motor → BlockChain adapters.
//
// Each `add_<motor>_to_chain(...)` factory:
//   1. Stores the motor parameters (R, L, K, Mechanical) in a
//      shared_ptr the chain holds.
//   2. Registers a step closure that:
//        - Reads phase currents from the state vector
//        - Computes T_em from the back-EMF · current product
//        - Calls mech->integrate(t, T_em, dt)
//        - Writes ω/θ to the chain's channels
//        - Writes the per-phase back-EMF voltage into the chain's
//          b_extra map (which the chain's b_extra_fn emits each
//          step into the kernel's residual)
//   3. Registers a reset closure that zeroes the mechanical state.
//
// Because everything lives inside the chain (the same `ChainContext`
// the chain's blocks already use), motor co-simulation is a pure-C++
// per-step operation — no Python interpreter cost.

#pragma once

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/motors/mechanical.hpp"
#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace pulsim::motors {

using blockchain::BlockChain;
using blockchain::ChainContext;

// =============================================================================
// DC motor
// =============================================================================

/// Add a DC motor to the chain. The user must have already added the
/// armature R + L + back-EMF "dummy" voltage source to the builder;
/// pass the resolved `armature_branch_var_idx` (state index of the
/// inductor current) and `bemf_source_idx` (state index of the
/// back-EMF source's branch-current row).
///
/// The closure writes:
///   * channels[omega_channel] = ω_rad_s
///   * channels[theta_channel] = θ_rad
///   * b_extra[bemf_source_idx] = −V_bemf (sign convention matches
///                                          the kernel's source row)
inline void add_dc_motor_to_chain(
    BlockChain& chain,
    const Mechanical& mech_init,
    Real R_a_ohm, Real L_a_H, Real Ke_V_s_per_rad, Real Kt_Nm_per_A,
    Index armature_branch_var_idx,
    Index bemf_source_idx,
    std::string omega_channel, std::string theta_channel) {

    auto mech = std::make_shared<Mechanical>(mech_init);

    auto step = [mech, Ke_V_s_per_rad, Kt_Nm_per_A,
                    R_a_ohm, L_a_H,
                    armature_branch_var_idx, bemf_source_idx,
                    omega_channel = std::move(omega_channel),
                    theta_channel = std::move(theta_channel)]
                   (ChainContext& ctx) {
        const Real i_a = (ctx.x && armature_branch_var_idx >= 0 &&
                            armature_branch_var_idx <
                              static_cast<Index>(ctx.x->size()))
                          ? (*ctx.x)[armature_branch_var_idx]
                          : Real{0};
        const Real T_em = Kt_Nm_per_A * i_a;
        mech->integrate(ctx.t, T_em, ctx.dt);
        ctx.channels[omega_channel] = mech->omega_rad_s;
        ctx.channels[theta_channel] = mech->theta_rad;
        const Real V_bemf = Ke_V_s_per_rad * mech->omega_rad_s;
        if (ctx.b_extra) {
            (*ctx.b_extra)[bemf_source_idx] = -V_bemf;
        }
        (void)R_a_ohm;  // captured for diagnostics; not used in math
        (void)L_a_H;
    };
    auto reset = [mech]() { mech->reset(); };
    chain.add(std::move(step), std::move(reset));
}


// =============================================================================
// 3-phase motors (PMSM + BLDC share the same skeleton)
// =============================================================================

enum class ThreePhaseBemfShape : std::uint8_t {
    Sinusoidal = 0,
    Trapezoidal = 1,
};

namespace detail {

[[nodiscard]] inline Real bemf_shape(ThreePhaseBemfShape kind,
                                          Real theta_e,
                                          int phase_index) noexcept {
    const Real offset =
        -Real{2.0943951023931953} * static_cast<Real>(phase_index);
    // Sign convention (matches the FOC chain's Park transform):
    // positive i_q must produce positive torque → BEMF shape is
    // −sin(θ_e + offset) instead of +sin.
    const Real s = -std::sin(theta_e + offset);
    if (kind == ThreePhaseBemfShape::Sinusoidal) {
        return s;
    }
    // Trapezoidal: 2·s clipped to ±1 → roughly 120°-flat-top.
    return std::clamp(Real{2} * s, Real{-1}, Real{1});
}

}  // namespace detail


// =============================================================================
// 3-phase squirrel-cage Induction Motor (IM)
// =============================================================================

/// Add a 3-phase squirrel-cage induction motor to the chain.
///
/// 5-state Krause stationary-αβ model (Krause "Analysis of Electric
/// Machinery and Drive Systems" § 6.6):
///
///   Rotor flux dynamics:
///     dψ_rα/dt = −ψ_rα / Tr + (Lm / Tr) · i_sα − ω_e · ψ_rβ
///     dψ_rβ/dt = −ψ_rβ / Tr + (Lm / Tr) · i_sβ + ω_e · ψ_rα
///   where Tr = Lr / Rr and ω_e = pole_pairs · ω_m.
///
///   Stator back-EMF (αβ frame):
///     e_α = (Lm / Lr) · dψ_rα/dt
///     e_β = (Lm / Lr) · dψ_rβ/dt
///   then inverse-Clarke to abc.
///
///   Electromagnetic torque (Krause Eq. 6.6-17, amplitude-invariant
///   Clarke convention):
///     T_em = (3/2) · pp · (Lm / Lr) · (i_sβ · ψ_rα − i_sα · ψ_rβ)
///
/// The user must have already added the per-phase Rs + σ·Ls + back-EMF
/// source to the builder; pass the resolved branch-var indices for the
/// per-phase inductor currents and the per-phase dummy-source rows.
///
/// Parameters mirror the existing 3-phase motor adapter, plus rotor
/// parameters (Rr, Lr, Lm) and four optional diagnostic channels.
inline void add_induction_motor_to_chain(
    BlockChain& chain,
    const Mechanical& mech_init,
    Real R_s_ohm, Real L_s_H,
    Real R_r_ohm, Real L_r_H, Real L_m_H,
    int pole_pairs,
    const std::array<Index, 3>& phase_inductor_idx,
    const std::array<Index, 3>& bemf_source_idx,
    std::string omega_channel,
    std::string theta_channel,
    std::string psi_alpha_channel = std::string{},
    std::string psi_beta_channel  = std::string{},
    std::string torque_channel    = std::string{},
    std::string slip_channel      = std::string{}) {

    auto mech = std::make_shared<Mechanical>(mech_init);
    // Rotor flux state — initial value 0 (demagnetised core).
    auto psi  = std::make_shared<std::array<Real, 2>>(
        std::array<Real, 2>{Real{0}, Real{0}});

    // Derived constants (computed once, captured into the closure).
    const Real Rr_safe = (R_r_ohm > Real{1e-30}) ? R_r_ohm : Real{1e-30};
    const Real Lr_safe = (L_r_H  > Real{1e-30}) ? L_r_H  : Real{1e-30};
    const Real inv_Tr  = Rr_safe / Lr_safe;        // 1 / Tr = Rr / Lr
    const Real Lm_inv_Tr = L_m_H * inv_Tr;         // (Lm / Tr) = Lm · Rr / Lr
    const Real Lm_over_Lr = L_m_H / Lr_safe;
    const Real sqrt3_over_2 = std::sqrt(Real{3}) / Real{2};

    auto step = [mech, psi,
                    inv_Tr, Lm_inv_Tr, Lm_over_Lr, sqrt3_over_2,
                    pole_pairs,
                    R_s_ohm, L_s_H,
                    phase_inductor_idx, bemf_source_idx,
                    omega_channel   = std::move(omega_channel),
                    theta_channel   = std::move(theta_channel),
                    psi_alpha_channel = std::move(psi_alpha_channel),
                    psi_beta_channel  = std::move(psi_beta_channel),
                    torque_channel    = std::move(torque_channel),
                    slip_channel      = std::move(slip_channel)]
                   (ChainContext& ctx) {
        if (!ctx.x) return;
        const Index n_state = static_cast<Index>(ctx.x->size());
        auto safe_read = [&](Index idx) {
            return (idx >= 0 && idx < n_state) ? (*ctx.x)[idx]
                                                  : Real{0};
        };

        // Measured stator currents (αβ frame via amplitude-invariant
        // Clarke transform; drops the zero-sequence which is OK for a
        // 3-wire Y-connected machine).
        const Real i_a = safe_read(phase_inductor_idx[0]);
        const Real i_b = safe_read(phase_inductor_idx[1]);
        const Real i_c = safe_read(phase_inductor_idx[2]);
        const Real i_alpha = (Real{2} / Real{3}) *
                                (i_a - Real{0.5} * i_b - Real{0.5} * i_c);
        const Real i_beta  = (Real{2} / Real{3}) * sqrt3_over_2 *
                                (i_b - i_c);

        const Real omega_m = mech->omega_rad_s;
        const Real omega_e = static_cast<Real>(pole_pairs) * omega_m;

        // Rotor flux dynamics (Krause Eq. 6.6-12 / 6.6-13).
        const Real psi_a = (*psi)[0];
        const Real psi_b = (*psi)[1];
        const Real dpsi_a = -inv_Tr * psi_a + Lm_inv_Tr * i_alpha
                                - omega_e * psi_b;
        const Real dpsi_b = -inv_Tr * psi_b + Lm_inv_Tr * i_beta
                                + omega_e * psi_a;

        // Induced stator back-EMF reflected from the rotor (αβ).
        const Real e_alpha = Lm_over_Lr * dpsi_a;
        const Real e_beta  = Lm_over_Lr * dpsi_b;

        // Inverse Clarke αβ → abc (amplitude-invariant).
        const Real e_a = e_alpha;
        const Real e_b = -Real{0.5} * e_alpha + sqrt3_over_2 * e_beta;
        const Real e_c = -Real{0.5} * e_alpha - sqrt3_over_2 * e_beta;

        // Electromagnetic torque (Krause Eq. 6.6-17).
        const Real T_em = (Real{1.5}) * static_cast<Real>(pole_pairs) *
                          Lm_over_Lr *
                          (i_beta * psi_a - i_alpha * psi_b);

        // Forward-Euler integration of the rotor flux state.
        (*psi)[0] = psi_a + dpsi_a * ctx.dt;
        (*psi)[1] = psi_b + dpsi_b * ctx.dt;

        // Mechanical integration (matches the other motor adapters).
        mech->integrate(ctx.t, T_em, ctx.dt);

        // Diagnostic channels (omitted when the user passes "").
        ctx.channels[omega_channel] = mech->omega_rad_s;
        ctx.channels[theta_channel] = mech->theta_rad;
        if (!psi_alpha_channel.empty()) {
            ctx.channels[psi_alpha_channel] = (*psi)[0];
        }
        if (!psi_beta_channel.empty()) {
            ctx.channels[psi_beta_channel] = (*psi)[1];
        }
        if (!torque_channel.empty()) {
            ctx.channels[torque_channel] = T_em;
        }
        if (!slip_channel.empty()) {
            // Slip fraction = (ω_psi − ω_m_elec) / ω_psi, clamped
            // to [-1, 1] for usability when ω_psi ≈ 0.
            const Real psi_mag = std::sqrt(
                (*psi)[0] * (*psi)[0] + (*psi)[1] * (*psi)[1]);
            Real slip = Real{1};
            if (psi_mag > Real{1e-6} && std::abs(omega_e) > Real{1e-6}) {
                // Instantaneous flux angular rate from the αβ derivatives.
                const Real omega_psi = (psi_a * dpsi_b - psi_b * dpsi_a)
                                       / (psi_mag * psi_mag);
                slip = (omega_psi - omega_e) /
                       std::max(std::abs(omega_psi), Real{1e-9});
            }
            ctx.channels[slip_channel] =
                std::clamp(slip, Real{-1}, Real{1});
        }

        // Back-EMF injection — same sign convention as the PMSM
        // adapter: source row carries +V_source; we inject −e so the
        // EMF appears in the loop direction.
        if (ctx.b_extra) {
            (*ctx.b_extra)[bemf_source_idx[0]] = -e_a;
            (*ctx.b_extra)[bemf_source_idx[1]] = -e_b;
            (*ctx.b_extra)[bemf_source_idx[2]] = -e_c;
        }
        (void)R_s_ohm; (void)L_s_H;  // captured for diagnostics
    };
    auto reset = [mech, psi]() {
        mech->reset();
        (*psi)[0] = Real{0};
        (*psi)[1] = Real{0};
    };
    chain.add(std::move(step), std::move(reset));
}


/// Add a 3-phase PMSM / BLDC motor to the chain. The user must have
/// already added the per-phase R + L + back-EMF source to the builder
/// and resolved the state indices.
///
/// Parameters:
///   `phase_inductor_idx` — array of 3 state indices for the
///       per-phase inductor branch currents.
///   `bemf_source_idx`    — array of 3 state indices for the
///       per-phase back-EMF source's branch-current row.
inline void add_three_phase_motor_to_chain(
    BlockChain& chain,
    const Mechanical& mech_init,
    Real R_s_ohm, Real L_s_H, Real psi_pm_Wb, int pole_pairs,
    ThreePhaseBemfShape bemf_kind,
    const std::array<Index, 3>& phase_inductor_idx,
    const std::array<Index, 3>& bemf_source_idx,
    std::string omega_channel, std::string theta_channel) {

    auto mech = std::make_shared<Mechanical>(mech_init);

    auto step = [mech, psi_pm_Wb, pole_pairs, bemf_kind,
                    R_s_ohm, L_s_H,
                    phase_inductor_idx, bemf_source_idx,
                    omega_channel = std::move(omega_channel),
                    theta_channel = std::move(theta_channel)]
                   (ChainContext& ctx) {
        if (!ctx.x) return;
        const Index n_state = static_cast<Index>(ctx.x->size());
        auto safe_read = [&](Index idx) {
            return (idx >= 0 && idx < n_state) ? (*ctx.x)[idx]
                                                  : Real{0};
        };
        const Real i_a = safe_read(phase_inductor_idx[0]);
        const Real i_b = safe_read(phase_inductor_idx[1]);
        const Real i_c = safe_read(phase_inductor_idx[2]);

        const Real omega_m = mech->omega_rad_s;
        const Real theta_e =
            mech->theta_rad * static_cast<Real>(pole_pairs);
        const Real E_peak =
            static_cast<Real>(pole_pairs) * psi_pm_Wb * omega_m;

        const Real sh_a = detail::bemf_shape(bemf_kind, theta_e, 0);
        const Real sh_b = detail::bemf_shape(bemf_kind, theta_e, 1);
        const Real sh_c = detail::bemf_shape(bemf_kind, theta_e, 2);

        const Real e_a = E_peak * sh_a;
        const Real e_b = E_peak * sh_b;
        const Real e_c = E_peak * sh_c;

        // Torque via instantaneous power balance.
        Real T_em;
        if (std::abs(omega_m) > Real{1e-6}) {
            T_em = (e_a * i_a + e_b * i_b + e_c * i_c) / omega_m;
        } else {
            // ω → 0 limit: T_em = pp · ψ · (Σ shape · i).
            T_em = static_cast<Real>(pole_pairs) * psi_pm_Wb *
                     (sh_a * i_a + sh_b * i_b + sh_c * i_c);
        }
        mech->integrate(ctx.t, T_em, ctx.dt);

        ctx.channels[omega_channel] = mech->omega_rad_s;
        ctx.channels[theta_channel] = mech->theta_rad;

        if (ctx.b_extra) {
            (*ctx.b_extra)[bemf_source_idx[0]] = -e_a;
            (*ctx.b_extra)[bemf_source_idx[1]] = -e_b;
            (*ctx.b_extra)[bemf_source_idx[2]] = -e_c;
        }
        (void)R_s_ohm; (void)L_s_H;  // captured for diagnostics
    };
    auto reset = [mech]() { mech->reset(); };
    chain.add(std::move(step), std::move(reset));
}

}  // namespace pulsim::motors
