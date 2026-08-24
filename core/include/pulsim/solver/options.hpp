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
#include <string>

namespace pulsim::solver {

struct SimulationOptions {
    Real t_start = Real{0};
    Real t_end   = Real{0};
    Real dt      = Real{0};

    /// Output decimation (v2.0 Phase 1, audit finding
    /// `waveform-storage-vector-of-vectors`). Record every m-th
    /// step: sample j is step `j · store_every`, so the recorded
    /// grid stays STRICTLY UNIFORM at an effective step of
    /// `store_every · dt` — FFT / harmonic / ripple analysis on
    /// the result remains valid, it just runs at the coarser
    /// spacing. The solver still INTEGRATES at `dt`; only what is
    /// stored changes.
    ///
    /// 1 (default) records every step, exactly as v1.x did.
    ///
    /// Because the grid is kept uniform, the FINAL step is
    /// recorded only when `(expected_step_count() - 1)` is a
    /// multiple of `store_every`; choose a divisor of the step
    /// count when the exact end state matters (or leave it at 1).
    Size store_every = 1;

    /// Per-step event-iteration cap. After each cache.solve,
    /// Layer 5 V2.1 re-solves with the updated diode state until
    /// it stabilises OR this many iterations have run.
    ///
    /// Default 16 — comfortable for typical PE workloads.
    /// Set to 0 to disable iteration (V2 behaviour).
    Size max_event_iterations = Size{16};

    /// Phase-0 fix #9 — behaviour when the event iteration hits a
    /// mask CYCLE (diode pair flip-flopping A→B→A, which no budget
    /// resolves) or exhausts `max_event_iterations`.
    ///
    /// false (default): accept the last consistent solve, record an
    /// `EventIterationBreach` on the result, and CONTINUE — a
    /// multilevel run no longer dies at t = 37 ms after minutes of
    /// compute. true: restore the pre-v1.8 hard throw (paper-grade
    /// strictness).
    bool strict_event_iterations = false;

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

    /// v2.0 Phase 2 (B.4) — local time-step reduction on a failed
    /// step.
    ///
    /// When the inner solve of a step throws — Newton will not
    /// converge, or the mask's matrix will not factorize — the run
    /// used to end there, discarding everything computed so far. A
    /// smaller step is the standard answer and a genuinely different
    /// problem: the trapezoidal companion's `2C/dt` grows as dt
    /// shrinks, which both improves the Jacobian's diagonal
    /// dominance and puts the previous state closer to the answer.
    ///
    /// So the step is rolled back and re-taken as 2 sub-steps of
    /// dt/2, then 4 of dt/4, up to `2^max_dt_halvings`. The run then
    /// continues at the nominal dt.
    ///
    /// THE OUTPUT GRID IS UNCHANGED. Sub-steps are internal; samples
    /// are still recorded only at nominal grid points, so
    /// `times[k] = t_start + k·dt` continues to hold exactly and an
    /// FFT of the result stays valid. Every reduction is recorded in
    /// `result.dt_retries`, because integrating an interval more
    /// finely than the user asked for is a change in accuracy they
    /// are entitled to know about.
    ///
    /// 0 disables the retry and restores the pre-v2.0 hard failure.
    Size max_dt_halvings = Size{6};   // dt/64 floor

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
    /// THIS DOES NOT SOLVE ANYTHING. It replaces a current the
    /// solver computed with a limit you configured, so on any step
    /// where it fires the trace shows YOUR NUMBER, not the circuit's.
    /// A guard that fires is evidence the model is missing something
    /// — usually a snubber across a path that opens when a bridge
    /// enters DCM — and every firing is now recorded in
    /// `result.inductor_guard_actions`, which the Python layer
    /// surfaces as a warning naming the device and the step count.
    /// The audit calls these confessions rather than features; they
    /// are kept because the 1 kW drive in `projects/inverters/
    /// pfc_vsi_drive` genuinely still needs them (without the clamp
    /// its line current peaks at ~544 A and the boost stage
    /// starves), but a run that depends on one is a run whose model
    /// is incomplete.
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
    /// Same caveat as `inductor_freeze_di_max` above: this
    /// substitutes a limit for a computed current, and every firing
    /// lands in `result.inductor_guard_actions`.
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

    // ---- Phase 2.4 adaptive RK selector (v1.5 schema, v1.6 wiring) --
    //
    // YAML-facing: lets users declare an adaptive integrator
    // (`dopri5`, `radau`) plus its tolerances inside `simulation:`.
    // Today only `"kernel"` is honoured by `run_transient` — Python
    // `simulate()` reads `integrator` and raises a clear
    // NotImplementedError for non-kernel choices, pointing at the
    // v1.6 cache refactor that will expose continuous-time (G, M, b)
    // for true DAE-aware RK integration.
    //
    // Recording the intent today keeps user YAML files
    // forward-compatible.

    /// Integrator name. `"kernel"` (default) uses the existing
    /// fixed-dt trap-companion run_transient. `"dopri5"` and
    /// `"radau"` are reserved for the v1.6 RK path. Anything
    /// else raises at the Python boundary.
    std::string integrator = "kernel";

    /// Relative tolerance for adaptive RK integrators. Ignored
    /// by `"kernel"`.
    Real rtol = Real{1e-5};

    /// Absolute tolerance for adaptive RK integrators. Ignored
    /// by `"kernel"`.
    Real atol = Real{1e-8};

    /// Initial step-size for adaptive RK integrators (0 ⇒ auto
    /// via Hairer/Wanner heuristic). Ignored by `"kernel"`.
    Real dt_init = Real{0};

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
        if (store_every == 0) {
            return false;   // would record nothing
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

    /// Number of samples actually RECORDED, i.e.
    /// `expected_step_count()` decimated by `store_every`
    /// (steps 0, m, 2m, … — ceil division). Equals
    /// `expected_step_count()` at the default `store_every = 1`.
    [[nodiscard]] Size expected_sample_count() const noexcept {
        const Size n = expected_step_count();
        if (n == 0 || store_every <= 1) {
            return n;
        }
        return (n + store_every - 1) / store_every;
    }
};

}  // namespace pulsim::solver
