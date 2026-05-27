// SPDX-License-Identifier: MIT
//
// Pulsim — Phase 2.2 / C++ port: Jiles-Atherton hysteretic inductor.
//
// Mirrors `python/pulsim/hysteresis.py` (JilesAthertonModel +
// HystereticInductor + make_hysteretic_inductor_observer).
//
// The model:
//   * Anhysteretic magnetisation M_an = Ms · Langevin(He / a),
//     where He = H + α · M (mean-field coupling).
//   * Irreversible component dM_irr/dH = (M_an − M) /
//                                       (k · sign(dH/dt) − α · (M_an − M))
//     with a denominator floor to handle the dH→0 / k→0 limit.
//   * Reversible component dM_rev/dH = c · dM_an/dHe /
//                                      (1 − c · α · dM_an/dHe).
//   * Total dM/dH = (1−c) · dM_irr/dH + dM_rev/dH.
//   * Forward-Euler sub-stepping on |dH|.
//
// The BlockChain adapter computes the hysteresis voltage v_M =
// N·A·μ₀·dM/dt and injects it at the dummy voltage source's residual
// row. Sign: +v_M (matches the fix landed alongside the Python sign
// correction — the JA contribution adds to L_0·di/dt in the same loop
// direction).

#pragma once

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

namespace pulsim::magnetics {

using blockchain::BlockChain;
using blockchain::ChainContext;

// Five-parameter Jiles-Atherton parameter set. POD; defaults are not
// physical — always set them from a material catalog or fit.
struct JilesAthertonParams {
    Real Ms = Real{4.0e5};   // saturation magnetisation [A/m]
    Real a  = Real{50.0};    // anhysteretic shape parameter [A/m]
    Real alpha = Real{5e-5}; // mean-field coupling (dimensionless)
    Real c  = Real{0.20};    // reversibility (0..1)
    Real k  = Real{30.0};    // pinning coefficient [A/m]
};


namespace detail {

// μ₀ = 4π·10⁻⁷ H/m, hard-coded.
inline constexpr Real MU_0 = Real{4.0} * Real{3.14159265358979323846}
                              * Real{1e-7};

// Langevin function L(x) = coth(x) − 1/x. Stable for all x.
[[nodiscard]] inline Real langevin(Real x) noexcept {
    const Real ax = std::abs(x);
    if (ax < Real{1e-4}) {
        // Taylor: L(x) ≈ x/3 − x³/45 + …
        return x * (Real{1.0 / 3.0} - x * x * Real{1.0 / 45.0});
    }
    if (ax > Real{30.0}) {
        // coth(x) ≈ 1 + 2·e^(-2|x|), so L(x) ≈ sign(x) − 1/x.
        return ((x > Real{0}) ? Real{1} : Real{-1}) - Real{1} / x;
    }
    return std::cosh(x) / std::sinh(x) - Real{1} / x;
}

// dL/dx, stable.
[[nodiscard]] inline Real langevin_deriv(Real x) noexcept {
    const Real ax = std::abs(x);
    if (ax < Real{1e-4}) {
        // L'(x) ≈ 1/3 − x²/15 + …
        return Real{1.0 / 3.0} - x * x * Real{1.0 / 15.0};
    }
    if (ax > Real{30.0}) {
        return Real{1} / (x * x);
    }
    const Real sinh_x = std::sinh(x);
    return Real{1} / (x * x) - Real{1} / (sinh_x * sinh_x);
}

}  // namespace detail


// Per-step JA integrator. Equivalent to the Python `JilesAthertonModel`.
class JilesAthertonModel {
public:
    JilesAthertonParams params;
    Real M = Real{0};   // current magnetisation [A/m]
    Real H = Real{0};   // last applied field [A/m]
    Real B = Real{0};   // current flux density [T]

    explicit JilesAthertonModel(
        const JilesAthertonParams& p, Real M0 = Real{0}) noexcept
        : params(p), M(M0) {
        B = detail::MU_0 * (H + M);
    }

    void reset(Real M0 = Real{0}) noexcept {
        M = M0;
        H = Real{0};
        B = detail::MU_0 * (H + M);
        H_prev_ = Real{0};
        initialised_ = false;
    }

    // Advance one step. Returns the new B (T).
    Real update(Real H_new, Real dt) noexcept {
        const JilesAthertonParams& p = params;
        if (!initialised_) {
            H_prev_ = H_new;
            initialised_ = true;
            H = H_new;
            B = detail::MU_0 * (H + M);
            (void)dt;
            return B;
        }

        const Real dH = H_new - H_prev_;
        const Real sign_dH = (dH >= Real{0}) ? Real{1} : Real{-1};
        const Real denom_floor =
            std::max(Real{1e-9}, Real{1e-6} * p.k);

        // Sub-step on |dH| if the per-step jump is too large
        // (forward Euler diverges when |dH| ≳ a).
        const Real max_dH = std::max(Real{0.5} * p.a, Real{1});
        const int n_sub = std::max(
            1, static_cast<int>(std::ceil(std::abs(dH) / max_dH)));
        const Real dH_sub = dH / static_cast<Real>(n_sub);
        Real H_iter = H_prev_;
        for (int i = 0; i < n_sub; ++i) {
            const Real H_sub = H_iter + dH_sub;
            const Real He_s = H_sub + p.alpha * M;
            const Real x_s = (p.a > Real{0}) ? (He_s / p.a) : Real{0};
            const Real M_an_s = p.Ms * detail::langevin(x_s);
            const Real dM_an_dHe_s = (p.a > Real{0})
                ? (p.Ms / p.a) * detail::langevin_deriv(x_s)
                : Real{0};

            const Real denom_s = p.k * sign_dH
                                   - p.alpha * (M_an_s - M);
            Real dM_irr_s = Real{0};
            if (std::abs(denom_s) >= denom_floor) {
                dM_irr_s = (M_an_s - M) / denom_s;
            }

            const Real rev_d_s = Real{1} - p.c * p.alpha * dM_an_dHe_s;
            const Real dM_rev_s = (std::abs(rev_d_s) > Real{1e-12})
                ? (p.c * dM_an_dHe_s / rev_d_s)
                : (p.c * dM_an_dHe_s);

            const Real dM_dH_s = (Real{1} - p.c) * dM_irr_s + dM_rev_s;
            M += dM_dH_s * dH_sub;
            // Saturation clamp.
            if (M > p.Ms) {
                M = p.Ms;
            } else if (M < -p.Ms) {
                M = -p.Ms;
            }
            H_iter = H_sub;
        }

        H_prev_ = H_new;
        H = H_new;
        B = detail::MU_0 * (H + M);
        (void)dt;  // present in API for future variable-dt subStepping.
        return B;
    }

private:
    Real H_prev_ = Real{0};
    bool initialised_ = false;
};


/// Add a hysteretic-inductor block to the chain.
///
/// The user must have already added the linear air-core inductor L₀
/// and the dummy V source to the builder via the Python helper
/// `pulsim.add_hysteretic_inductor`, and resolved the resulting
/// branch-var indices:
///
///   * `inductor_branch_var_idx` — state index of the linear L₀
///       inductor's branch current (read by the adapter to compute H).
///   * `bemf_source_idx`         — state index of the dummy V source's
///       residual row (the adapter writes v_M = +N·A·μ₀·dM/dt here).
///
/// `N_turns` × `A_core` × `l_m` define the magnetic-circuit geometry:
///   H = N · i_L / l_m       (Ampère's law for a long coil)
///   v_M = N · A · μ₀ · dM/dt (Faraday on the magnetisation flux).
///
/// Optional diagnostic channels:
///   `M_channel`, `B_channel`, `H_channel`, `vm_channel`.
inline void add_hysteretic_inductor_to_chain(
    BlockChain& chain,
    const JilesAthertonParams& params,
    int N_turns,
    Real l_m,
    Real A_core,
    Index inductor_branch_var_idx,
    Index bemf_source_idx,
    std::string M_channel  = std::string{},
    std::string B_channel  = std::string{},
    std::string H_channel  = std::string{},
    std::string vm_channel = std::string{}) {

    auto model = std::make_shared<JilesAthertonModel>(params);
    // ψ_M = N·A·μ₀·M ; dψ_M/dt = N·A·μ₀·dM/dt.
    const Real psi_coupling =
        static_cast<Real>(N_turns) * A_core * detail::MU_0;
    const Real H_per_amp = static_cast<Real>(N_turns) /
                             std::max(l_m, Real{1e-30});

    // Snapshot of M from the previous step (for dM/dt finite-diff).
    auto m_prev = std::make_shared<Real>(Real{0});

    auto step = [model, m_prev, psi_coupling, H_per_amp,
                    inductor_branch_var_idx, bemf_source_idx,
                    M_channel  = std::move(M_channel),
                    B_channel  = std::move(B_channel),
                    H_channel  = std::move(H_channel),
                    vm_channel = std::move(vm_channel)]
                   (ChainContext& ctx) {
        if (!ctx.x) return;
        const Index n_state = static_cast<Index>(ctx.x->size());
        const Real i_L =
            (inductor_branch_var_idx >= 0 &&
             inductor_branch_var_idx < n_state)
                ? (*ctx.x)[inductor_branch_var_idx]
                : Real{0};
        const Real H_new = H_per_amp * i_L;
        const Real M_prev = *m_prev;
        model->update(H_new, ctx.dt);
        const Real dM = model->M - M_prev;
        *m_prev = model->M;
        const Real v_M = psi_coupling
                          * (dM / std::max(ctx.dt, Real{1e-18}));

        // Diagnostic channels.
        if (!M_channel.empty()) ctx.channels[M_channel] = model->M;
        if (!B_channel.empty()) ctx.channels[B_channel] = model->B;
        if (!H_channel.empty()) ctx.channels[H_channel] = model->H;
        if (!vm_channel.empty()) ctx.channels[vm_channel] = v_M;

        // +v_M injection — matches the Python observer's sign after
        // the v1.5 fix (see PR #51 in pulsim history). Both the L₀
        // di/dt drop and the JA back-EMF live in the same passive
        // branch and add in the same loop direction.
        if (ctx.b_extra) {
            (*ctx.b_extra)[bemf_source_idx] = v_M;
        }
    };
    auto reset = [model, m_prev]() {
        model->reset();
        *m_prev = Real{0};
    };
    chain.add(std::move(step), std::move(reset));
}

}  // namespace pulsim::magnetics
