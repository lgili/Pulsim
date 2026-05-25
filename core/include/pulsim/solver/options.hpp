#pragma once

// =============================================================================
// Pulsim — Layer 5: SimulationOptions (fixed-dt time-stepping inputs)
// =============================================================================
//
// `pulsim-v2-solver-and-events` Phase 1.
//
// Value-type aggregate that holds the inputs to `run_transient`:
// the simulation window [t_start, t_end] and the fixed dt.
//
// `valid()` performs a self-check (finite values, dt > 0,
// t_end > t_start) — `run_transient` calls it and throws
// `std::invalid_argument` if invalid, so the user gets a clean
// error rather than a silent infinite loop.
//
// `expected_step_count()` is the size hint for output
// pre-allocation. It uses `floor((t_end - t_start) / dt) + 1` so
// the result includes both endpoint samples (at t_start and at
// t_end, or the last sample <= t_end if dt doesn't divide the
// span evenly).

#include "pulsim/numeric/types.hpp"

#include <cmath>

namespace pulsim::solver {

struct SimulationOptions {
    Real t_start = Real{0};
    Real t_end   = Real{0};
    Real dt      = Real{0};

    /// Per-step event-iteration cap. After each cache.solve,
    /// Layer 5 V2.1 re-solves with the updated diode state until
    /// it stabilises OR this many iterations have run.
    ///
    /// Default 16 — comfortable for typical PE workloads.
    /// Set to 0 to disable iteration (V2 behaviour).
    Size max_event_iterations = Size{16};

    // ---- Newton (Layer 5 V4) ----------------------------------------
    //
    // When `run_transient` is called with a non-empty
    // `NonlinearRefreshFn`, each step's inner solve becomes
    // Newton-iterated. These fields control the inner Newton loop.

    /// Max Newton iterations per step. Default 50 — comfortable
    /// for the smooth-blend IdealDiode and other PE-typical
    /// nonlinearities.
    Size max_newton_iterations = Size{50};

    /// Newton convergence tolerance on ||dx||_inf. Default 1e-9.
    Real tol_newton_dx = Real{1e-9};

    /// Newton convergence tolerance on ||residual||_inf.
    /// Default 1e-9.
    Real tol_newton_res = Real{1e-9};

    /// Enable backtracking line search inside Newton
    /// (Layer 4 V4 / globalization). When `false` (default),
    /// plain Newton is used. When `true`, each Newton iteration
    /// halves the step size up to 8 times if the full step
    /// would increase the residual norm.
    bool enable_newton_line_search = false;

    /// Enable Levenberg-Marquardt damping (Layer 4 V5). Solves
    /// `(J + λ·I) · dx = -f` with adaptive λ — grows on
    /// rejected steps, shrinks on accepted ones. Handles
    /// stiffer problems than plain line search (the κ=20
    /// sinusoidal rectifier, etc.). Takes precedence over
    /// line search when both are enabled.
    bool enable_newton_lm = false;

    /// Layer 5 V3 — sub-step state correction.
    ///
    /// When `true` AND `cache.dt() > 0` (dynamic path), each
    /// time step that detects a commutation event mid-step
    /// is RETROACTIVELY split into two sub-steps at the
    /// estimated event time `t_est`. Sub-step 1 uses the
    /// pre-event mask and `cache.solve_at(mask_pre, dt₁, …)`
    /// (Layer 4 V7's auxiliary-dt solve). Sub-step 2 uses
    /// the post-event mask and `cache.solve_at(mask_post,
    /// dt₂, …)`. The result eliminates the single-shot
    /// commutation wobble visible at coarse dt.
    ///
    /// V0 corrects ONLY the first detected event per step.
    /// Default `false` to preserve V2.2 behaviour.
    bool enable_substep_state_correction = false;

    /// Floating-inductor freeze (post-solve guard).
    ///
    /// When `> 0`, after each `cache.solve` the solver checks every
    /// tracked inductor's branch-current change against this bound
    /// (`|i_new − i_prev| > inductor_freeze_di_max`) and, if exceeded,
    /// overrides ``x[branch_var_id] = i_prev`` (freezes the state
    /// for that step). Catches the rare singular configurations
    /// where the inductor's loop has no closed conduction path —
    /// e.g. a rectifier in deep DCM or a series-blocking-diode that
    /// briefly disconnects the inductor — and prevents the solver
    /// from emitting unphysical i_L values (we observed ~1 kA spikes
    /// in a 1 kW drive).
    ///
    /// Set to 0 (default) to disable the guard. A typical bound for
    /// power-electronics circuits is `200 A` — that's well above any
    /// realistic step-to-step change at PWM rates (V_max·dt/L_min
    /// rarely exceeds 10 A per step) but cleanly catches the
    /// kiloamp-scale solver failures.
    Real inductor_freeze_di_max = Real{0};

    /// Absolute inductor-current clamp (post-solve guard).
    ///
    /// When `> 0`, after each solve every tracked inductor's branch
    /// current is hard-clamped to ``[-inductor_abs_clamp, +clamp]``.
    /// Catches the slow drift form of the floating-inductor failure
    /// where the solver gradually walks i_L past physical bounds at
    /// less than ``inductor_freeze_di_max`` per step (we observed
    /// i_L001 drifting to −1 kA at ~0.1 A/step in a rectifier in DCM).
    /// Typical PE-circuit bound: 100 A for a 1 kW drive, scaled by
    /// the design's rated current.
    Real inductor_abs_clamp = Real{0};

    [[nodiscard]] bool valid() const noexcept {
        // All three values must be finite (no NaN, no infinity).
        if (!std::isfinite(t_start) || !std::isfinite(t_end) ||
            !std::isfinite(dt)) {
            return false;
        }
        // Forward-progress invariants.
        if (dt <= Real{0}) {
            return false;
        }
        if (t_end <= t_start) {
            return false;
        }
        return true;
    }

    /// Number of output samples that `run_transient` will record
    /// for valid options. Includes both endpoint samples — the
    /// loop visits k = 0, 1, …, N - 1 with t = t_start + k · dt
    /// and the last sample is the largest k such that
    /// `t_start + k · dt <= t_end`.
    ///
    /// FP robustness: when `(t_end - t_start)` should be an exact
    /// multiple of dt (the common case for user-friendly sims),
    /// floating-point can produce values like 499.99999999999983
    /// instead of 500.0. We add a tiny tolerance before floor so
    /// borderline cases land on the intended integer.
    [[nodiscard]] Size expected_step_count() const noexcept {
        if (!valid()) {
            return 0;
        }
        const Real span = t_end - t_start;
        // Number of FULL dt steps that fit in [t_start, t_end]. We
        // pad against tiny FP rounding (≈ 1 ULP) so that
        // mathematically-exact integer ratios don't lose a sample.
        const Real ratio = span / dt + Real{1e-9};
        return static_cast<Size>(std::floor(ratio)) + 1;
    }
};

}  // namespace pulsim::solver
