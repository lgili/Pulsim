#pragma once

// refactor-unified-transient-stepper, task group 4 (design D2): the
// integrators as stage tables. A stage says which companion formula the
// devices stamp, which cached factor it solves on and what step the
// devices are told; it says nothing about the loop that sequences it.
//
// Every value here is computed with the expression the drivers used
// before the table existed, so a driver that takes its stage from the
// table is bit-identical to one that did not.

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
    [[nodiscard]] static Real gamma() noexcept {
        return pwl::HistoryState::trbdf2_gamma();
    }
    [[nodiscard]] static StageSpec stage(Real h, int s) noexcept {
        const Real hg = gamma() * h;
        if (s == 0) return {pwl::TrBdf2Stage::Trapezoidal, hg, hg, hg};
        return {pwl::TrBdf2Stage::Bdf2Stage2, hg, h, h};
    }
};

}  // namespace pulsim::step
