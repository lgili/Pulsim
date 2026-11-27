#pragma once

// refactor-unified-transient-stepper, task group 4 (design D2): the
// integrators as stage tables. A stage says which companion formula the
// devices stamp, which cached factor it solves on and what step the
// devices are told; it says nothing about the loop that sequences it.
//
// Every value here is computed with the expression the drivers used
// before the table existed, so a driver that takes its stage from the
// table is bit-identical to one that did not.

#include <cmath>

#include "pulsim/pwl/device_refresh.hpp"   // StepCtx
#include "pulsim/pwl/history_state.hpp"    // trbdf2_gamma()
#include "pulsim/pwl/trbdf2_stage.hpp"     // TrBdf2Stage
#include "pulsim/numeric/types.hpp"

namespace pulsim::step {

/// One solve, described. The matrix is `G + (2/factor_dt)·C`, i.e. the
/// trapezoidal factor of `factor_dt` (the key of `cache.solve_at`);
/// `ctx_h` and `formula` are what the stateful devices are told
/// (`pwl::StepCtx`); `t_off` is where the stage ends, from the step start.
struct StageSpec {
    pwl::TrBdf2Stage formula = pwl::TrBdf2Stage::Trapezoidal;
    Real factor_dt = Real{0};
    Real ctx_h = Real{0};
    Real t_off = Real{0};

    [[nodiscard]] pwl::StepCtx ctx() const noexcept { return {ctx_h, formula}; }
};

/// The trapezoidal rule over `h`: one stage on the factor of `h`.
struct Trapezoidal {
    static constexpr const char* name = "trapezoidal";
    static constexpr int order = 2;
    static constexpr int n_stages = 1;
    static constexpr Real err_exponent = Real{-1.0 / 3.0};
    [[nodiscard]] static constexpr Real weight(int, int) noexcept { return Real{0}; }
    [[nodiscard]] static StageSpec stage(Real h, int /*s*/ = 0) noexcept {
        return {pwl::TrBdf2Stage::Trapezoidal, h, h, h};
    }
};

/// Backward Euler over `h`. Its matrix G + C/h is the trapezoidal factor
/// of 2h, so its factor key is 2h; the devices are told `h`.
struct BackwardEuler {
    static constexpr const char* name = "backward_euler";
    static constexpr int order = 1;
    static constexpr int n_stages = 1;
    static constexpr Real err_exponent = Real{-0.5};
    [[nodiscard]] static constexpr Real weight(int, int) noexcept { return Real{0}; }
    [[nodiscard]] static StageSpec stage(Real h, int /*s*/ = 0) noexcept {
        return {pwl::TrBdf2Stage::BackwardEuler, Real{2} * h, h, h};
    }
    /// The backward-Euler stage that solves on the factor `factor_dt`
    /// (the fixed engine's restart stage: interval factor_dt/2).
    [[nodiscard]] static StageSpec on_factor(Real factor_dt) noexcept {
        const Real h = factor_dt * Real{0.5};
        return {pwl::TrBdf2Stage::BackwardEuler, factor_dt, h, h};
    }
};

/// TR-BDF2 (Bank et al.), γ = 2 − √2: stage 0 is the trapezoid over γh,
/// stage 1 the BDF2 step to t + h. Both solve on the factor of γh (the
/// single-factor property: c1/h = 2/(γh)); stage 1 tells the devices the
/// full step h, as `HistoryState::compute_b_extra_trbdf2_stage2` does.
struct TrBdf2 {
    static constexpr const char* name = "trbdf2";
    static constexpr int order = 2;
    static constexpr int n_stages = 2;
    /// Exponent of the step-size law: the estimate is O(h^3).
    static constexpr Real err_exponent = Real{-1.0 / 3.0};
    /// Stage 1 couples to stage 0 through its BDF2 formula (the history
    /// computes it), not through a pseudo previous state.
    [[nodiscard]] static constexpr Real weight(int, int) noexcept { return Real{0}; }
    [[nodiscard]] static Real gamma() noexcept {
        return pwl::HistoryState::trbdf2_gamma();
    }
    [[nodiscard]] static StageSpec stage(Real h, int s) noexcept {
        const Real hg = gamma() * h;
        if (s == 0) return {pwl::TrBdf2Stage::Trapezoidal, hg, hg, hg};
        return {pwl::TrBdf2Stage::Bdf2Stage2, hg, h, h};
    }
    /// The local truncation error of one component from its derivative at
    /// the step start (f_n), at the stage point (f_g) and at the step end
    /// (f_1) (Bank et al.; the factor 2 makes it the true error, p10_lte.py).
    /// Moved from run_transient_trbdf2; the same expression, so the same bits.
    [[nodiscard]] static Real lte(Real h, Real f_n, Real f_g, Real f_1) noexcept {
        const Real gamma = pwl::HistoryState::trbdf2_gamma();
        const Real clte = pwl::HistoryState::trbdf2_lte_const();
        return Real{2} * clte * h
               * (f_n / gamma
                  - f_g / (gamma * (Real{1} - gamma))
                  + f_1 / (Real{1} - gamma));
    }
};

/// Alexander's SDIRK(2,2), γ = 1 − 1/√2 (L-stable, stiffly accurate),
/// written as two backward-Euler solves of γh on the factor 2γh. Stage s
/// starts from the pseudo previous state x*_s = x0 + Σ_{j<s} w_sj·(Y_j −
/// x*_j), w_sj = a_sj / a_jj (h·f(Y_j) = (Y_j − x*_j)/a_jj), so no stage
/// derivative is ever formed. No driver runs it: it is the third method
/// the stepper carries (task group 4; the fixed engine's restart stages
/// are this table with its own γ expression, DECISIONS 4).
struct Sdirk22 {
    static constexpr const char* name = "sdirk22";
    static constexpr int order = 2;
    static constexpr int n_stages = 2;
    /// Its embedded estimate (Y_2 against x0 + h·f(Y_1)) is O(h^2).
    static constexpr Real err_exponent = Real{-0.5};
    [[nodiscard]] static Real gamma() noexcept {
        return Real{1} - Real{0.70710678118654752440};
    }
    [[nodiscard]] static StageSpec stage(Real h, int s) noexcept {
        const Real gh = gamma() * h;
        return {pwl::TrBdf2Stage::BackwardEuler, Real{2} * gh, gh,
                s == 0 ? gh : h};
    }
    [[nodiscard]] static Real weight(int s, int j) noexcept {
        return (s == 1 && j == 0) ? (Real{1} - gamma()) / gamma() : Real{0};
    }
    /// The γ of the TR-BDF2 driver's SDIRK(2,2) restart step: γ_TR-BDF2 / 2,
    /// written 1 − 1/√2 as the driver always wrote it (the literal of
    /// gamma() may round differently, DECISIONS 4, so the driver keeps this one).
    [[nodiscard]] static Real restart_gamma() noexcept {
        return Real{1} - Real{1} / std::sqrt(Real{2});
    }
    /// The restart step's embedded first-order estimate of one state entry,
    /// h(1 − γs)(k1 − k2) with k_s the stage derivatives (design D3).
    /// Moved from run_transient_trbdf2; the same expression, so the same bits.
    [[nodiscard]] static Real lte(Real h, Real gs, Real k1, Real k2) noexcept {
        return h * (Real{1} - gs) * (k1 - k2);
    }
    /// The same estimate on a charge / flux row, in increments: the stage-2
    /// increment of ψ5 minus ((1 − γs)/γs) times the step-end increment.
    [[nodiscard]] static Real lte_q(Real d_psi5, Real d_1, Real gs) noexcept {
        return d_psi5 - (Real{1} - gs) / gs * d_1;
    }
};

/// Dense output between two accepted states: the cubic Hermite through
/// (x0, f0) at θ = 0 and (x1, f1) at θ = 1, θ ∈ [0, 1] (exact for cubics;
/// third order, so it never limits a second-order method).
[[nodiscard]] inline Real hermite(Real x0, Real f0, Real x1, Real f1, Real h,
                                  Real theta) noexcept {
    const Real t2 = theta * theta;
    const Real t3 = t2 * theta;
    return (Real{2} * t3 - Real{3} * t2 + Real{1}) * x0
           + (t3 - Real{2} * t2 + theta) * h * f0
           + (Real{3} * t2 - Real{2} * t3) * x1
           + (t3 - t2) * h * f1;
}

}  // namespace pulsim::step
