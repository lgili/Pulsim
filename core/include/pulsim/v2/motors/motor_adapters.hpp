// SPDX-License-Identifier: MIT
//
// Pulsim v2 — Phase D / C++ port: motor → BlockChain adapters.
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

#include "pulsim/v2/blockchain/chain.hpp"
#include "pulsim/v2/motors/mechanical.hpp"
#include "pulsim/v2/numeric/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace pulsim::v2::motors {

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

}  // namespace pulsim::v2::motors
