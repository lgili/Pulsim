// SPDX-License-Identifier: MIT
//
// Pulsim — Phase 2.3 / C++ port: sensorless rotor observers.
//
// Mirrors python/pulsim/observers.py:
//   * SlidingModeObserver — PMSM, Utkin sliding surface + LPF + PLL
//   * FluxMRASObserver    — IM, Schauder voltage + current models with
//                           normalised cross-product (bootstrap fix)
//
// Each adapter consumes v_α, v_β, i_α, i_β from chain channels (or
// const values) and writes its outputs (theta_hat, omega_hat, etc.)
// to user-named channels. Inputs are kernel state-vector slots or
// channels — matched by the standard `InputRef` mechanism.

#pragma once

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <utility>

namespace pulsim::observers {

using blockchain::BlockChain;
using blockchain::ChainContext;

namespace {
constexpr Real kTwoPi = Real{2} * std::numbers::pi_v<Real>;
}  // namespace

// =============================================================================
// SlidingModeObserver — PMSM sensorless angle/speed
// =============================================================================

struct SlidingModeObserverParams {
    Real Rs = Real{0.5};                  // stator resistance per phase [Ω]
    Real Ls = Real{200e-6};               // stator inductance per phase [H]
    Real K_sl = Real{50.0};               // sliding-mode gain [V]
    Real omega_lpf = kTwoPi
                      * Real{500.0};      // LPF cutoff [rad/s]
    Real Kp_pll = Real{200.0};
    Real Ki_pll = Real{1e4};
    Real f_init_hz = Real{50.0};
    Real epsilon = Real{0.5};             // boundary-layer width [A]
    Real low_speed_threshold = Real{0.05};
};

namespace detail {

struct SmoState {
    Real i_hat_alpha = Real{0};
    Real i_hat_beta  = Real{0};
    Real e_hat_alpha = Real{0};
    Real e_hat_beta  = Real{0};
    Real theta_hat   = Real{0};
    Real omega_hat   = Real{0};
    Real pll_integ   = Real{0};
};

}  // namespace detail


/// Add a SlidingModeObserver block to the chain.
///
/// Reads stator-frame (v_α, v_β, i_α, i_β) — typically pre-computed
/// by the chain's Clarke transform and the controller's inverse Clarke
/// — and writes (theta_hat, omega_hat, e_alpha_hat, e_beta_hat,
/// low_speed_flag) to the user-named channels.
inline void add_sliding_mode_observer_to_chain(
    BlockChain& chain,
    const SlidingModeObserverParams& params,
    std::string v_alpha_channel,
    std::string v_beta_channel,
    std::string i_alpha_channel,
    std::string i_beta_channel,
    std::string theta_hat_channel,
    std::string omega_hat_channel,
    std::string e_alpha_hat_channel = std::string{},
    std::string e_beta_hat_channel  = std::string{},
    std::string low_speed_flag_channel = std::string{}) {

    auto state = std::make_shared<detail::SmoState>();
    state->omega_hat =
        kTwoPi * params.f_init_hz;

    auto step = [state, params,
                    v_alpha_channel = std::move(v_alpha_channel),
                    v_beta_channel  = std::move(v_beta_channel),
                    i_alpha_channel = std::move(i_alpha_channel),
                    i_beta_channel  = std::move(i_beta_channel),
                    theta_hat_channel = std::move(theta_hat_channel),
                    omega_hat_channel = std::move(omega_hat_channel),
                    e_alpha_hat_channel = std::move(e_alpha_hat_channel),
                    e_beta_hat_channel  = std::move(e_beta_hat_channel),
                    low_speed_flag_channel = std::move(low_speed_flag_channel)]
                   (ChainContext& ctx) {
        auto read = [&](const std::string& key) -> Real {
            auto it = ctx.channels.find(key);
            return (it == ctx.channels.end()) ? Real{0} : it->second;
        };
        const Real v_alpha = read(v_alpha_channel);
        const Real v_beta  = read(v_beta_channel);
        const Real i_alpha = read(i_alpha_channel);
        const Real i_beta  = read(i_beta_channel);

        // 1. Sliding-mode current observer (tanh boundary layer).
        const Real err_a = state->i_hat_alpha - i_alpha;
        const Real err_b = state->i_hat_beta  - i_beta;
        const Real z_a = params.K_sl * std::tanh(err_a / params.epsilon);
        const Real z_b = params.K_sl * std::tanh(err_b / params.epsilon);
        const Real di_hat_a =
            (v_alpha - params.Rs * state->i_hat_alpha - z_a) / params.Ls;
        const Real di_hat_b =
            (v_beta  - params.Rs * state->i_hat_beta  - z_b) / params.Ls;
        state->i_hat_alpha += di_hat_a * ctx.dt;
        state->i_hat_beta  += di_hat_b * ctx.dt;

        // 2. Equivalent-control LPF.
        const Real alpha_lpf = std::clamp(
            params.omega_lpf * ctx.dt, Real{0}, Real{1});
        state->e_hat_alpha += alpha_lpf * (z_a - state->e_hat_alpha);
        state->e_hat_beta  += alpha_lpf * (z_b - state->e_hat_beta);

        // 3. PLL on the d-axis projection of the back-EMF.
        const Real c = std::cos(state->theta_hat);
        const Real s = std::sin(state->theta_hat);
        const Real e_phase = -(state->e_hat_alpha * c
                                  + state->e_hat_beta * s);
        state->pll_integ += params.Ki_pll * ctx.dt * e_phase;
        const Real delta_omega = params.Kp_pll * e_phase
                                    + state->pll_integ;
        state->omega_hat = kTwoPi
                              * params.f_init_hz + delta_omega;
        state->theta_hat += state->omega_hat * ctx.dt;
        // Wrap to [0, 2π).
        state->theta_hat = std::fmod(state->theta_hat, kTwoPi);
        if (state->theta_hat < Real{0}) {
            state->theta_hat += kTwoPi;
        }

        // 4. Outputs.
        ctx.channels[theta_hat_channel] = state->theta_hat;
        ctx.channels[omega_hat_channel] = state->omega_hat;
        if (!e_alpha_hat_channel.empty()) {
            ctx.channels[e_alpha_hat_channel] = state->e_hat_alpha;
        }
        if (!e_beta_hat_channel.empty()) {
            ctx.channels[e_beta_hat_channel] = state->e_hat_beta;
        }
        if (!low_speed_flag_channel.empty()) {
            const Real e_mag = std::hypot(
                state->e_hat_alpha, state->e_hat_beta);
            ctx.channels[low_speed_flag_channel] =
                (e_mag < params.low_speed_threshold) ? Real{1} : Real{0};
        }
    };
    auto reset = [state, params]() {
        state->i_hat_alpha = Real{0};
        state->i_hat_beta  = Real{0};
        state->e_hat_alpha = Real{0};
        state->e_hat_beta  = Real{0};
        state->theta_hat   = Real{0};
        state->omega_hat   =
            kTwoPi * params.f_init_hz;
        state->pll_integ   = Real{0};
    };
    chain.add(std::move(step), std::move(reset));
}


// =============================================================================
// FluxMRASObserver — IM sensorless speed (normalised cross-product)
// =============================================================================

struct FluxMRASObserverParams {
    Real Rs = Real{2.46};
    Real Ls = Real{0.0948};
    Real Lr = Real{0.0948};
    Real Lm = Real{0.0874};
    Real Rr = Real{1.23};                 // 0 → fallback Rr ≈ 0.5·Rs
    Real voltage_model_hpf_omega = Real{5.0};
    Real Kp_mras = Real{50.0};
    Real Ki_mras = Real{1e3};
    bool normalise_eps = true;
    Real eps_norm_floor = Real{1e-6};
};

namespace detail {

struct MrasState {
    Real psi_r_alpha_ref = Real{0};
    Real psi_r_beta_ref  = Real{0};
    Real psi_r_alpha_adj = Real{0};
    Real psi_r_beta_adj  = Real{0};
    Real omega_hat       = Real{0};
    Real mras_integ      = Real{0};
    Real i_alpha_prev    = Real{0};
    Real i_beta_prev     = Real{0};
};

}  // namespace detail


/// Add a FluxMRASObserver block to the chain.
///
/// Reads (v_α, v_β, i_α, i_β) from the chain channels and writes
/// (omega_hat, psi_r_adj_alpha, psi_r_adj_beta) to user-named channels.
inline void add_flux_mras_observer_to_chain(
    BlockChain& chain,
    const FluxMRASObserverParams& params,
    std::string v_alpha_channel,
    std::string v_beta_channel,
    std::string i_alpha_channel,
    std::string i_beta_channel,
    std::string omega_hat_channel,
    std::string psi_adj_alpha_channel = std::string{},
    std::string psi_adj_beta_channel  = std::string{},
    std::string psi_ref_alpha_channel = std::string{},
    std::string psi_ref_beta_channel  = std::string{}) {

    auto state = std::make_shared<detail::MrasState>();

    // Precompute constants.
    const Real sigma = Real{1} - (params.Lm * params.Lm)
                          / (params.Ls * params.Lr);
    const Real L_sigma_s = sigma * params.Ls;
    const Real Rr_eff = (params.Rr > Real{0})
        ? params.Rr : Real{0.5} * params.Rs;
    const Real Tr = params.Lr / std::max(Rr_eff, Real{1e-9});
    const Real inv_Tr = Real{1} / Tr;
    const Real scale = params.Lr / params.Lm;
    const Real leak = params.voltage_model_hpf_omega;

    auto step = [state, params, L_sigma_s, inv_Tr, scale, leak,
                    v_alpha_channel = std::move(v_alpha_channel),
                    v_beta_channel  = std::move(v_beta_channel),
                    i_alpha_channel = std::move(i_alpha_channel),
                    i_beta_channel  = std::move(i_beta_channel),
                    omega_hat_channel    = std::move(omega_hat_channel),
                    psi_adj_alpha_channel = std::move(psi_adj_alpha_channel),
                    psi_adj_beta_channel  = std::move(psi_adj_beta_channel),
                    psi_ref_alpha_channel = std::move(psi_ref_alpha_channel),
                    psi_ref_beta_channel  = std::move(psi_ref_beta_channel)]
                   (ChainContext& ctx) {
        auto read = [&](const std::string& key) -> Real {
            auto it = ctx.channels.find(key);
            return (it == ctx.channels.end()) ? Real{0} : it->second;
        };
        const Real v_alpha = read(v_alpha_channel);
        const Real v_beta  = read(v_beta_channel);
        const Real i_alpha = read(i_alpha_channel);
        const Real i_beta  = read(i_beta_channel);

        // Reference model (voltage-driven, Tajima-Hori leaky integrator).
        const Real di_a = (ctx.dt > Real{0})
            ? (i_alpha - state->i_alpha_prev) / ctx.dt : Real{0};
        const Real di_b = (ctx.dt > Real{0})
            ? (i_beta  - state->i_beta_prev)  / ctx.dt : Real{0};
        state->i_alpha_prev = i_alpha;
        state->i_beta_prev  = i_beta;
        const Real e_alpha = v_alpha - params.Rs * i_alpha - L_sigma_s * di_a;
        const Real e_beta  = v_beta  - params.Rs * i_beta  - L_sigma_s * di_b;
        state->psi_r_alpha_ref +=
            (scale * e_alpha - leak * state->psi_r_alpha_ref) * ctx.dt;
        state->psi_r_beta_ref  +=
            (scale * e_beta  - leak * state->psi_r_beta_ref)  * ctx.dt;

        // Adaptive model (current-driven, uses ω̂).
        const Real dpsi_a = params.Lm * inv_Tr * i_alpha
                              - inv_Tr * state->psi_r_alpha_adj
                              - state->omega_hat * state->psi_r_beta_adj;
        const Real dpsi_b = params.Lm * inv_Tr * i_beta
                              - inv_Tr * state->psi_r_beta_adj
                              + state->omega_hat * state->psi_r_alpha_adj;
        state->psi_r_alpha_adj += dpsi_a * ctx.dt;
        state->psi_r_beta_adj  += dpsi_b * ctx.dt;

        // Adaptation — normalised cross product (bootstrap fix).
        const Real eps_raw =
            state->psi_r_beta_ref  * state->psi_r_alpha_adj
            - state->psi_r_alpha_ref * state->psi_r_beta_adj;
        Real eps;
        if (params.normalise_eps) {
            const Real psi_ref_mag = std::hypot(
                state->psi_r_alpha_ref, state->psi_r_beta_ref);
            const Real psi_adj_mag = std::hypot(
                state->psi_r_alpha_adj, state->psi_r_beta_adj);
            if (psi_ref_mag > params.eps_norm_floor &&
                psi_adj_mag > params.eps_norm_floor) {
                eps = eps_raw / (psi_ref_mag * psi_adj_mag);
            } else {
                eps = Real{0};
            }
        } else {
            eps = eps_raw;
        }
        state->mras_integ += params.Ki_mras * ctx.dt * eps;
        state->omega_hat = params.Kp_mras * eps + state->mras_integ;

        // Outputs.
        ctx.channels[omega_hat_channel] = state->omega_hat;
        if (!psi_adj_alpha_channel.empty()) {
            ctx.channels[psi_adj_alpha_channel] = state->psi_r_alpha_adj;
        }
        if (!psi_adj_beta_channel.empty()) {
            ctx.channels[psi_adj_beta_channel] = state->psi_r_beta_adj;
        }
        if (!psi_ref_alpha_channel.empty()) {
            ctx.channels[psi_ref_alpha_channel] = state->psi_r_alpha_ref;
        }
        if (!psi_ref_beta_channel.empty()) {
            ctx.channels[psi_ref_beta_channel] = state->psi_r_beta_ref;
        }
    };
    auto reset = [state]() {
        *state = detail::MrasState{};
    };
    chain.add(std::move(step), std::move(reset));
}

}  // namespace pulsim::observers
