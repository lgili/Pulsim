#pragma once

// =============================================================================
// Pulsim — Layer 5: run_transient (fixed-dt time-stepping loop)
// =============================================================================
//
// `pulsim-v2-solver-and-events` Phase 2.
//
// The V0 entry point. Wraps Layer 4's `cache.solve(mask, b_extra,
// x)` in a fixed-dt time-stepping loop with:
//   * User-supplied `switch_fn(t) → SwitchStateMask` for the
//     switch schedule (decouples event detection from this layer
//     — V1 will supply auto-event-driven schedules).
//   * Optional `b_extra_fn(t) → Vector` for time-varying
//     source RHS (e.g., sinusoidal sources).
//   * `Vector::Zero(state_size)` initial state (V0 has no
//     caps/inductors so no carryover state between steps).
//   * Output recording every step (no down-sampling in V0).
//
// THE per-step hot path is:
//
//     mask  = switch_fn(t)
//     b_ex  = b_extra_fn ? b_extra_fn(t) : zero_buffer
//     cache.solve(mask, b_ex, x)        ← Layer 4's PLECS-style call
//     result.push((t, x))
//
// That `cache.solve` line is ~µs (one map probe + one O(nnz)
// triangular solve on a cached factor). 1000 steps = ~ms total.
// THE architecture that beats PSIM/PLECS.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <format>
#include <functional>
#include <stdexcept>

namespace pulsim::solver {

// -----------------------------------------------------------------------------
// combine_masks — overlay diode bits on top of the user mask.
//
// Returns a new mask where every bit i = (diode_owned.get(i) ?
// diode.get(i) : user.get(i)). Used by Layer 5 V2 to merge the
// user's switch_fn output with the diode auto-state.
// -----------------------------------------------------------------------------
[[nodiscard]] inline topology::SwitchStateMask combine_masks(
    const topology::SwitchStateMask& user,
    const topology::SwitchStateMask& diode,
    const topology::SwitchStateMask& diode_owned) noexcept {
    // user, diode, diode_owned must all be the same width.
    const std::uint64_t owned = diode_owned.bits();
    const std::uint64_t merged =
        (user.bits()  & ~owned) |     // keep user bits where diode doesn't own
        (diode.bits() &  owned);      // overlay diode bits where it does
    topology::SwitchStateMask out(user.size());
    out.set_bits(merged);
    return out;
}

// -----------------------------------------------------------------------------
// interp_commutation_time — linear-interpolated zero crossing
// of the watched signal between t_prev and t_curr.
//
// Watched signal:
//   OFF → ON: v_diode − V_th    (was negative at prev, ≥ 0 at curr)
//   ON → OFF: i_diode            (was positive at prev, ≤ 0 at curr)
//
// If the two endpoints have the same sign (no real zero
// crossing), returns t_curr (the dt-grid time — the V0
// fallback).
// -----------------------------------------------------------------------------
[[nodiscard]] constexpr Real interp_commutation_time(
    Real t_prev, Real t_curr,
    Real s_prev, Real s_curr) noexcept {
    // Same sign → no crossing in the interval; clamp.
    if (s_prev * s_curr > Real{0}) {
        return t_curr;
    }
    const Real denom = s_prev - s_curr;
    // Hand-rolled absolute value — std::abs(double) is not
    // constexpr until C++26, so use a branchless ternary that
    // works in constant evaluation.
    const Real abs_denom = denom < Real{0} ? -denom : denom;
    if (abs_denom < std::numeric_limits<Real>::min()) {
        return t_curr;
    }
    const Real frac = s_prev / denom;   // ∈ [0, 1] when signs differ
    const Real t_star = t_prev + (t_curr - t_prev) * frac;
    if (t_star < t_prev) return t_prev;
    if (t_star > t_curr) return t_curr;
    return t_star;
}

// -----------------------------------------------------------------------------
// Callback type aliases.
//
// `std::function` is the V0 dispatch mechanism. A V1 follow-up
// may templatise on the callback types to remove the indirection
// (small-buffer optimisation handles most simple lambdas, but
// not all). For now, the per-step overhead is negligible vs the
// triangular solve cost.
// -----------------------------------------------------------------------------
using SwitchScheduleFn =
    std::function<topology::SwitchStateMask(Real)>;

using BExtraFn = std::function<Vector(Real)>;

/// State-aware observer invoked at the START of every step,
/// BEFORE `switch_fn(t)` and `b_extra_fn(t)` are evaluated.
/// Receives `(t_k, x_{k-1})` — the time of the upcoming solve
/// and the most recently computed state vector. Side-effect
/// only: the observer typically mutates Python-side controller
/// state (e.g. a PIController) that the user's
/// `switch_fn` / `b_extra_fn` then reads to close the loop.
///
/// For sample 0 (the initial condition), the observer is
/// called with `x_prev = x` (the IC itself — either zero or
/// the DC operating point depending on `start_from_dc_op`),
/// so a discrete-time PI controller can prime its filter at
/// the correct initial output.
using StepObserverFn =
    std::function<void(Real, const Vector&)>;

/// Cancellation callback — invoked at the start of every step.
/// Returning `false` causes `run_transient` to break the loop and
/// return whatever it has accumulated so far (so the partial trace
/// is still useful — e.g. for a live scope that the user stopped).
/// Returning `true` (or having no callback at all) continues the
/// simulation as normal.
using ShouldContinueFn = std::function<bool()>;

// -----------------------------------------------------------------------------
// run_transient — the V0 transient simulation entry point.
//
// Preconditions (each throws std::invalid_argument on violation):
//   * opts.valid()                       — finite, dt > 0, t_end > t_start
//   * state_size > 0                     — state vector must have entries
//   * static_cast<bool>(switch_fn)       — schedule callback required
//
// Postcondition: result.num_steps() == opts.expected_step_count()
//                result.times[k]   == opts.t_start + k * opts.dt
//                result.states[k]  is the solution at result.times[k]
//
// Lifetime: borrows `cache`, `opts`, `switch_fn`, `b_extra_fn`.
// Returns by value (NRVO + move).
// -----------------------------------------------------------------------------
inline SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    Size state_size,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    const StepObserverFn& step_observer = {}) {

    // ---- Input validation -------------------------------------------------
    if (!opts.valid()) {
        throw std::invalid_argument(
            "run_transient: SimulationOptions are not valid "
            "(check t_start < t_end, dt > 0, all finite)");
    }
    if (state_size == 0) {
        throw std::invalid_argument(
            "run_transient: state_size must be > 0");
    }
    if (!switch_fn) {
        throw std::invalid_argument(
            "run_transient: switch_fn callback is required "
            "(default-constructed std::function is not callable)");
    }

    // ---- Pre-allocate output + working buffers ----------------------------
    const Size n_steps = opts.expected_step_count();
    SimulationResult result;
    result.reserve(n_steps);

    Vector x = Vector::Zero(state_size);

    // Zero buffer reused when no b_extra_fn is supplied. Avoids
    // per-step allocation of a fresh zero vector.
    const Vector zero_b_extra = Vector::Zero(state_size);

    // ---- The hot loop ----------------------------------------------------
    //
    // Use an integer step counter and recompute t = t_start + k·dt
    // each iteration. Accumulating `t += dt` over thousands of
    // steps drifts by several ULP — fine for most uses, but bad
    // for PWM where a period-boundary drift causes spurious
    // switching.
    for (Size k = 0; k < n_steps; ++k) {
        const Real t = opts.t_start + static_cast<Real>(k) * opts.dt;

        // State-aware observer fires first so user-side
        // controllers can update before `switch_fn(t)` reads
        // the new duty / mask.
        if (step_observer) {
            step_observer(t, x);
        }

        const auto mask = switch_fn(t);

        if (b_extra_fn) {
            const Vector b_extra = b_extra_fn(t);
            cache.solve(mask, b_extra, x);
        } else {
            cache.solve(mask, zero_b_extra, x);
        }

        // Record the sample. `x` is copied into the result so the
        // next iteration can mutate the working `x` without
        // disturbing the recorded sample.
        result.times.push_back(t);
        result.states.push_back(x);
    }

    return result;
}

// -----------------------------------------------------------------------------
// V1 overload — history-aware transient with Capacitor / Inductor
// support.
//
// Differences vs the V0 overload above:
//   * Takes `const Graph&` + `const DevicePool&` instead of an
//     explicit `state_size` — state size is derived from
//     `pool.state_size(graph)`.
//   * Constructs an internal `HistoryState` and updates it after
//     each cache.solve.
//   * Validates `cache.dt() == opts.dt`. A mismatched cache would
//     silently produce wrong numbers — the throw catches it
//     up-front.
//   * For circuits with no Capacitor/Inductor, the behaviour is
//     bit-identical to the V0 overload (HistoryState has zero
//     entries, compute_b_extra returns the zero vector).
// -----------------------------------------------------------------------------
inline SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false,
    const pwl::NonlinearRefreshFn& nl_refresh = {},
    const StepObserverFn& step_observer = {},
    const Vector* initial_state = nullptr,
    const ShouldContinueFn& should_continue = {}) {

    // ---- Input validation ---------------------------------------------
    if (!opts.valid()) {
        throw std::invalid_argument(
            "run_transient: SimulationOptions are not valid "
            "(check t_start < t_end, dt > 0, all finite)");
    }
    if (!switch_fn) {
        throw std::invalid_argument(
            "run_transient: switch_fn callback is required");
    }
    // The cache's dt must match opts.dt. If the user built the
    // cache for a different dt, the trap companion's g_eq is wrong
    // and the solution silently drifts. Catch up-front.
    //
    // Tolerance: 0 (exact match required). The user controls both
    // values, so they can synchronise them precisely.
    if (cache.dt() > Real{0} && cache.dt() != opts.dt) {
        throw std::invalid_argument(
            "run_transient: cache.dt() does not match opts.dt; "
            "rebuild the cache with the same dt the simulation "
            "will use");
    }
    const Size state_size = pool.state_size(graph);
    if (state_size == 0) {
        throw std::invalid_argument(
            "run_transient: pool.state_size(graph) is 0");
    }

    // ---- Pre-allocate ---------------------------------------------------
    const Size n_steps = opts.expected_step_count();
    SimulationResult result;
    result.reserve(n_steps);

    Vector x = Vector::Zero(state_size);
    pwl::HistoryState history{graph, pool};
    history.reset();   // explicit (constructor already zeroes)
    // Initial-state injection (Phase B.1): if the caller supplies a
    // non-null `initial_state` vector, copy it into `x` and seed the
    // trapezoidal history from it. This lets the adaptive driver
    // chain consecutive `simulate()` segments without restarting at
    // zero. The seeded values must be self-consistent — typically
    // the last state from the previous segment.
    if (initial_state != nullptr) {
        if (static_cast<Size>(initial_state->size()) != state_size) {
            throw std::invalid_argument(
                "run_transient: initial_state size mismatch");
        }
        x = *initial_state;
        history.seed_from_dc_op(x);
    }

    // V17: saturable inductors carry their own (i_L, V_L)_old
    // history, since they need it in the Newton refresh (not
    // just as a pre-computed b_extra). Initialise to zero.
    pwl::SaturableInductorHistory sat_history;
    sat_history.init(graph, pool);
    const bool has_saturable = !sat_history.empty();

    // V17: shared dt that the saturable-inductor refresh
    // reads each iteration. Updated when sub-step correction
    // splits dt into dt1 + dt2.
    Real refresh_dt = opts.dt;

    // V17: if any saturable inductors are present, wrap the
    // user's nl_refresh with our additive saturable
    // stamping. The user-supplied refresh runs first (it
    // zero-clears + stamps diodes/MOSFETs/IGBTs); we then
    // ADD the saturable inductor contributions on top.
    pwl::NonlinearRefreshFn nl_refresh_effective = nl_refresh;
    if (has_saturable) {
        nl_refresh_effective =
            [user_refresh = nl_refresh, &sat_history,
             &refresh_dt](const Vector& x,
                            sparse::Matrix& J_nl,
                            Vector& f_nl,
                            const topology::Graph& g,
                            const pwl::DevicePool& p)
                -> Real {
                Real max_i = Real{0};
                if (user_refresh) {
                    max_i = user_refresh(x, J_nl, f_nl, g, p);
                } else {
                    if (J_nl.rows() > 0) J_nl.setZero();
                    if (f_nl.size() > 0) f_nl.setZero();
                }
                // Additive saturable-inductor stamping.
                for (const auto& e : sat_history.entries()) {
                    const Real v_from =
                        stamping::read_node_voltage(x, e.from);
                    const Real v_to =
                        stamping::read_node_voltage(x, e.to);
                    const Real i_L_new = x[e.branch_var_id];
                    const models::ModelInputs<
                            models::SaturableInductor> iv{
                        i_L_new};
                    const auto [L_eff, partials] =
                        models::evaluate_current_and_jacobian<
                            models::SaturableInductor>(
                                iv, e.params);
                    const Real dL_di = partials[0];
                    const Real two_over_dt =
                        Real{2} / refresh_dt;
                    const Real two_L_over_dt =
                        two_over_dt * L_eff;
                    const Real di = i_L_new - e.i_L_old;
                    const Real R_row =
                        (v_from - v_to) + e.V_L_old -
                        two_L_over_dt * di;
                    const bool from_active =
                        stamping::node_is_active(e.from);
                    const bool to_active =
                        stamping::node_is_active(e.to);
                    if (from_active)
                        f_nl[e.from] += i_L_new;
                    if (to_active)
                        f_nl[e.to]   -= i_L_new;
                    f_nl[e.branch_var_id] += R_row;
                    if (from_active) {
                        J_nl.coeffRef(
                            e.branch_var_id, e.from)
                            += Real{1};
                    }
                    if (to_active) {
                        J_nl.coeffRef(
                            e.branch_var_id, e.to)
                            -= Real{1};
                    }
                    const Real dR_di_L =
                        -two_L_over_dt -
                        two_over_dt * di * dL_di;
                    J_nl.coeffRef(
                        e.branch_var_id, e.branch_var_id)
                        += dR_di_L;
                    if (from_active) {
                        J_nl.coeffRef(
                            e.from, e.branch_var_id)
                            += Real{1};
                    }
                    if (to_active) {
                        J_nl.coeffRef(
                            e.to, e.branch_var_id)
                            -= Real{1};
                    }
                    max_i = std::max(
                        max_i, std::abs(i_L_new));
                }
                return max_i;
            };
    }

    pwl::DiodeEventState diodes{graph, pool};
    diodes.reset();
    const auto diode_owned = diodes.diode_owned_bits();
    const bool has_diodes = diodes.num_diodes() > 0;

    // ---- DC operating-point pre-charge (Layer 5 V3) ------------------
    //
    // When the user requests `start_from_dc_op = true`, compute
    // the DC steady-state at t_start's switch configuration,
    // iterate diode consistency at the DC system, and seed
    // HistoryState + DiodeEventState from the DC solution.
    // Sample 0 becomes the DC state vector instead of zero.
    if (start_from_dc_op) {
        const Size dc_max_iters =
            opts.max_event_iterations > 0
                ? opts.max_event_iterations
                : Size{16};
        Size iters = 0;
        bool flipped = false;
        do {
            auto mask = switch_fn(opts.t_start);
            if (has_diodes) {
                mask = combine_masks(mask,
                                      diodes.current_diode_mask(),
                                      diode_owned);
            }
            x = pwl::compute_dc_op(graph, pool, mask,
                                    opts.t_start);
            flipped = has_diodes && diodes.update_from_state(x);
            ++iters;
        } while (flipped && iters < dc_max_iters);
        if (flipped) {
            throw std::runtime_error(
                "run_transient: DC operating-point event "
                "iteration did not converge at t_start");
        }
        history.seed_from_dc_op(x);
    }

    Vector b_extra(static_cast<Index>(state_size));

    // Event-iteration cap. 0 disables iteration entirely
    // (matching Layer 5 V2 behaviour for regression testing).
    const Size max_iters = opts.max_event_iterations;

    if (cache.dt() > Real{0}) {
        // ----------- DYNAMIC PATH ---------------------------------------
        //
        // Trap rule semantics: at the start of iteration k, `x` is the
        // state at time t_k. cache.solve advances it to x at t_{k+1}.
        // To make recorded `(t, x)` pairs match the physical state at
        // that time, we record the IC at t = t_start as sample 0, then
        // each subsequent solve produces the sample at t = t_start +
        // k·dt for k = 1..N-1.
        // Sample 0: zero IC (V0/V1/V2.1 default) OR DC OP (V3
        // when start_from_dc_op=true; x was overwritten above).
        result.times.push_back(opts.t_start);
        result.states.push_back(x);
        result.event_iteration_count.push_back(0);

        // Prime the observer with the initial condition so a
        // discrete-time PI / sampler / lookup-table sees the
        // true starting state at t_start.
        if (step_observer) {
            step_observer(opts.t_start, x);
        }

        for (Size k = 1; k < n_steps; ++k) {
            // User-cancellation check (Phase: live scope). Lets a
            // GUI / external watchdog stop the simulation early
            // while preserving the partial trace.
            if (should_continue && !should_continue()) {
                break;
            }
            const Real t = opts.t_start +
                            static_cast<Real>(k) * opts.dt;
            const Real t_prev = opts.t_start +
                                 static_cast<Real>(k - 1) * opts.dt;

            // Snapshot the state at t_prev for sub-step
            // commutation timing (Layer 5 V2.2) and state
            // correction (Layer 5 V3).
            const Vector x_prev = x;

            // State-aware observer fires BEFORE `switch_fn(t)`
            // so the user can update a Python-side PI / sampler /
            // comparator with `x_prev` and have `switch_fn(t)`
            // read the new duty / mask.
            if (step_observer) {
                step_observer(t, x_prev);
            }
            // V3 snapshots — taken regardless of whether
            // correction fires (cheap; vectors are small per
            // dynamic-branch). Restored only if a commutation
            // event is detected AND
            // enable_substep_state_correction is true.
            const auto history_snap = history.snapshot();
            const std::vector<bool> diodes_snap =
                has_diodes ? diodes.snapshot_on_bits()
                            : std::vector<bool>{};
            // Pre-event mask: what mask we'd use BEFORE any
            // diode flip during this step. Captured before
            // event iteration runs.
            topology::SwitchStateMask mask_pre = switch_fn(t);
            if (has_diodes) {
                mask_pre = combine_masks(
                    mask_pre, diodes.current_diode_mask(),
                    diode_owned);
            }

            // 1. History from previous step.
            const Vector b_extra_history =
                history.compute_b_extra(opts.dt);

            // 2. Optional user-supplied b_extra(t).
            const Vector b_extra_user = b_extra_fn
                ? b_extra_fn(t)
                : Vector::Zero(state_size);

            // 3. Built-in PWM (V4) + sine (V11) + pulse
            //    (V12) sources.
            const Vector b_extra_pwm =
                sources::compute_pwm_b_extra(pool, graph, t);
            const Vector b_extra_sine =
                sources::compute_sine_b_extra(pool, graph, t);
            const Vector b_extra_pulse =
                sources::compute_pulse_b_extra(pool, graph, t);

            b_extra = b_extra_history + b_extra_user +
                      b_extra_pwm + b_extra_sine +
                      b_extra_pulse;

            // 3. Event-iteration loop. Solve, update diode state,
            //    re-solve if any diode flipped. Stop when stable
            //    or max_iters hit.
            //
            // If `nl_refresh` was supplied, the inner solve uses
            // Newton (with the trap-companion `b_extra`); else
            // it's the cached linear solve.
            Size iters = 0;
            bool flipped = false;
            do {
                auto mask = switch_fn(t);
                if (has_diodes) {
                    const auto diode_mask =
                        diodes.current_diode_mask();
                    mask = combine_masks(mask, diode_mask,
                                          diode_owned);
                }
                if (nl_refresh_effective) {
                    const auto& seg = cache.lookup(mask);
                    x = pwl::solve_with_newton_b_extra(
                        seg, nl_refresh_effective, graph, pool,
                        /*x_init=*/x, b_extra,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search,
                        opts.enable_newton_lm);
                } else {
                    cache.solve(mask, b_extra, x);
                }
                flipped = has_diodes &&
                          diodes.update_from_state(x);
                ++iters;
            } while (flipped && iters < max_iters);

            if (flipped) {
                throw std::runtime_error(std::format(
                    "run_transient: event-iteration limit "
                    "reached without convergence at t = {}; "
                    "raise max_event_iterations or reduce dt",
                    t));
            }

            // 4. Sub-step commutation timing (Layer 5 V2.2) +
            //    state correction (Layer 5 V3).
            //
            // Detect zero-crossing events for diode signals
            // and (if enabled) retroactively split the step
            // into two sub-steps at the first detected
            // event's t_est.
            std::vector<CommutationEvent> step_events;
            if (has_diodes) {
                for (const auto& e : diodes.entries()) {
                    const Real v_a_prev =
                        stamping::read_node_voltage(x_prev, e.from);
                    const Real v_k_prev =
                        stamping::read_node_voltage(x_prev, e.to);
                    const Real v_d_prev = v_a_prev - v_k_prev;
                    const Real v_a_curr =
                        stamping::read_node_voltage(x, e.from);
                    const Real v_k_curr =
                        stamping::read_node_voltage(x, e.to);
                    const Real v_d_curr = v_a_curr - v_k_curr;

                    const Real s_prev = v_d_prev - e.V_th;
                    const Real s_curr = v_d_curr - e.V_th;
                    if (s_prev * s_curr >= Real{0}) continue;

                    const Real t_est = interp_commutation_time(
                        t_prev, t, s_prev, s_curr);
                    step_events.push_back(CommutationEvent{
                        .t_estimated = t_est,
                        .branch_id   = e.branch_id,
                        .new_state   = e.is_on,
                    });
                }
            }

            // Apply V3 sub-step correction if enabled AND an
            // event was detected. V0 corrects only the FIRST
            // event (sort to pick earliest if multiple).
            // Minimum sub-step duration as a fraction of the
            // main dt. Events landing within this fraction of
            // either step boundary are NOT corrected — the
            // trap-companion's `g_eq = 2C/dt` becomes
            // ill-conditioned at tiny dt, and the boundary
            // events offer no accuracy benefit anyway (the
            // shorter sub-step is essentially a single point).
            //
            // 1% of dt is conservative: for dt = 200 µs that
            // means events within the first/last 2 µs are
            // skipped.
            const Real substep_min_dt = opts.dt * Real{0.01};
            bool corrected = false;
            if (opts.enable_substep_state_correction &&
                !step_events.empty()) {
                // Find earliest event in the step.
                Real t_est = step_events.front().t_estimated;
                for (const auto& ev : step_events) {
                    if (ev.t_estimated < t_est) {
                        t_est = ev.t_estimated;
                    }
                }
                const Real dt1 = t_est - t_prev;
                const Real dt2 = opts.dt - dt1;
                if (dt1 > substep_min_dt &&
                    dt2 > substep_min_dt) {
                    // Roll back to pre-step state.
                    x = x_prev;
                    history.restore(history_snap);
                    if (has_diodes) {
                        diodes.restore_on_bits(diodes_snap);
                    }

                    // Sub-step 1: pre-event mask, dt1.
                    {
                        const Vector be_user_1 = b_extra_fn
                            ? b_extra_fn(t_est)
                            : Vector::Zero(state_size);
                        const Vector be_hist_1 =
                            history.compute_b_extra(dt1);
                        const Vector be_pwm_1 =
                            sources::compute_pwm_b_extra(
                                pool, graph, t_est);
                        const Vector be_sine_1 =
                            sources::compute_sine_b_extra(
                                pool, graph, t_est);
                        const Vector be_pulse_1 =
                            sources::compute_pulse_b_extra(
                                pool, graph, t_est);
                        const Vector be_total_1 =
                            be_hist_1 + be_user_1 +
                            be_pwm_1 + be_sine_1 +
                            be_pulse_1;
                        cache.solve_at(
                            mask_pre, dt1, be_total_1, x);
                        history.update_from_state(x, dt1);
                        if (has_saturable) {
                            sat_history.update_from_state(x);
                        }
                    }

                    // Apply commutation events directly from
                    // V2.2's detection (don't rely on
                    // `update_from_state` to re-decide — at the
                    // exact zero-crossing both v_diode and
                    // i_diode are near the threshold, and the
                    // SwitchedDiode decision may keep the
                    // pre-event state).
                    if (has_diodes) {
                        std::vector<bool> post_event_bits =
                            diodes_snap;
                        const auto entries = diodes.entries();
                        for (const auto& ev : step_events) {
                            for (Size i = 0;
                                 i < entries.size(); ++i) {
                                if (entries[i].branch_id ==
                                    ev.branch_id) {
                                    post_event_bits[i] =
                                        ev.new_state;
                                    break;
                                }
                            }
                        }
                        diodes.restore_on_bits(
                            post_event_bits);
                    }

                    // Sub-step 2: post-event mask, dt2.
                    {
                        topology::SwitchStateMask mask_post =
                            switch_fn(t);
                        if (has_diodes) {
                            mask_post = combine_masks(
                                mask_post,
                                diodes.current_diode_mask(),
                                diode_owned);
                        }
                        const Vector be_user_2 = b_extra_fn
                            ? b_extra_fn(t)
                            : Vector::Zero(state_size);
                        const Vector be_hist_2 =
                            history.compute_b_extra(dt2);
                        const Vector be_pwm_2 =
                            sources::compute_pwm_b_extra(
                                pool, graph, t);
                        const Vector be_sine_2 =
                            sources::compute_sine_b_extra(
                                pool, graph, t);
                        const Vector be_pulse_2 =
                            sources::compute_pulse_b_extra(
                                pool, graph, t);
                        const Vector be_total_2 =
                            be_hist_2 + be_user_2 +
                            be_pwm_2 + be_sine_2 +
                            be_pulse_2;
                        cache.solve_at(
                            mask_post, dt2, be_total_2, x);
                        history.update_from_state(x, dt2);
                        if (has_saturable) {
                            sat_history.update_from_state(x);
                        }
                    }
                    corrected = true;
                }
            }

            // Record diagnostic events from this step.
            for (const auto& ev : step_events) {
                result.commutation_events.push_back(ev);
            }

            // 4b. Floating-inductor freeze guard (Layer 5 V5).
            //
            // When ``opts.inductor_freeze_di_max > 0``, walk every
            // tracked inductor and snap its branch current back to
            // the previous step's value whenever the per-step change
            // exceeds the configured bound. Catches the rare
            // near-singular MNA configurations where an inductor's
            // loop has no closed conduction path (rectifier in deep
            // DCM, series-blocking diode briefly open, etc.) and
            // the LU solve emits a kiloamp-scale unphysical jump.
            // See ``inductor_freeze_di_max`` in options.hpp.
            if (!corrected &&
                (opts.inductor_freeze_di_max > Real{0} ||
                 opts.inductor_abs_clamp > Real{0})) {
                for (const auto& e : history.entries()) {
                    if (e.kind != pwl::DevicePool::StoredKind::Inductor) {
                        continue;
                    }
                    Real i_new = x[e.inductor_branch_var_id];
                    // Step-to-step jump guard: snap back to i_prev
                    // when the solver emits an unphysical kilo-amp
                    // delta (rectifier in DCM, etc.).
                    if (opts.inductor_freeze_di_max > Real{0}) {
                        const Real di  = i_new - e.i_prev;
                        const Real adi = di < Real{0} ? -di : di;
                        if (adi > opts.inductor_freeze_di_max) {
                            i_new = e.i_prev;
                        }
                    }
                    // Absolute clamp: catches the *slow drift* form
                    // of the same failure where the per-step delta
                    // stays below the freeze threshold but i_L walks
                    // monotonically past physical bounds over many
                    // line cycles.
                    if (opts.inductor_abs_clamp > Real{0}) {
                        const Real lim = opts.inductor_abs_clamp;
                        if (i_new >  lim) i_new =  lim;
                        if (i_new < -lim) i_new = -lim;
                    }
                    x[e.inductor_branch_var_id] = i_new;
                }
            }

            // 5. Commit history for the next step (skipped if
            //    the substep correction already advanced it).
            if (!corrected) {
                history.update_from_state(x, opts.dt);
                if (has_saturable) {
                    sat_history.update_from_state(x);
                }
            }

            // 6. Record. event_iteration_count = iters - 1 (the
            //    first iteration always runs; the count is the
            //    number of EXTRA solves caused by diode flips).
            result.times.push_back(t);
            result.states.push_back(x);
            result.event_iteration_count.push_back(iters - 1);
        }
    } else {
        // ----------- STATIC PATH ----------------------------------------
        //
        // No dynamic devices → cache.solve gives the DC operating
        // point for the requested switch state. Diode iteration
        // still applies.
        const Vector zero_b_extra = Vector::Zero(state_size);
        for (Size k = 0; k < n_steps; ++k) {
            const Real t = opts.t_start +
                            static_cast<Real>(k) * opts.dt;
            const Real t_prev = k > 0
                ? (opts.t_start +
                    static_cast<Real>(k - 1) * opts.dt)
                : opts.t_start;
            // Sub-step bisection snapshot (Layer 5 V2.2). At
            // k=0 we don't have a prev step; just use the
            // current x as a degenerate snapshot — the
            // sign-change check will exclude it from events.
            const Vector x_prev = x;

            // State-aware observer fires before switch_fn(t) so
            // discrete-time controllers can update.
            if (step_observer) {
                step_observer(t, x_prev);
            }

            const Vector b_extra_user_only = b_extra_fn
                ? b_extra_fn(t)
                : zero_b_extra;
            // Built-in PWM (V4) + sine (V11) + pulse (V12)
            // sources — apply on the static path too, since
            // all are time-varying even without caps/Ls.
            const Vector b_extra_pwm =
                sources::compute_pwm_b_extra(pool, graph, t);
            const Vector b_extra_sine =
                sources::compute_sine_b_extra(pool, graph, t);
            const Vector b_extra_pulse =
                sources::compute_pulse_b_extra(pool, graph, t);
            const Vector b_extra_user = b_extra_user_only +
                                          b_extra_pwm +
                                          b_extra_sine +
                                          b_extra_pulse;

            Size iters = 0;
            bool flipped = false;
            do {
                auto mask = switch_fn(t);
                if (has_diodes) {
                    const auto diode_mask =
                        diodes.current_diode_mask();
                    mask = combine_masks(mask, diode_mask,
                                          diode_owned);
                }
                if (nl_refresh_effective) {
                    const auto& seg = cache.lookup(mask);
                    x = pwl::solve_with_newton_b_extra(
                        seg, nl_refresh_effective, graph, pool,
                        /*x_init=*/x, b_extra_user,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search,
                        opts.enable_newton_lm);
                } else {
                    cache.solve(mask, b_extra_user, x);
                }
                flipped = has_diodes &&
                          diodes.update_from_state(x);
                ++iters;
            } while (flipped && iters < max_iters);

            if (flipped) {
                throw std::runtime_error(std::format(
                    "run_transient: event-iteration limit "
                    "reached without convergence at t = {}",
                    t));
            }

            // Sub-step bisection on the static path too.
            if (has_diodes && k > 0) {
                for (const auto& e : diodes.entries()) {
                    const Real v_a_prev =
                        stamping::read_node_voltage(x_prev, e.from);
                    const Real v_k_prev =
                        stamping::read_node_voltage(x_prev, e.to);
                    const Real v_d_prev = v_a_prev - v_k_prev;
                    const Real v_a_curr =
                        stamping::read_node_voltage(x, e.from);
                    const Real v_k_curr =
                        stamping::read_node_voltage(x, e.to);
                    const Real v_d_curr = v_a_curr - v_k_curr;

                    const Real s_prev = v_d_prev - e.V_th;
                    const Real s_curr = v_d_curr - e.V_th;
                    if (s_prev * s_curr >= Real{0}) continue;

                    const Real t_est = interp_commutation_time(
                        t_prev, t, s_prev, s_curr);
                    result.commutation_events.push_back(
                        CommutationEvent{
                            .t_estimated = t_est,
                            .branch_id   = e.branch_id,
                            .new_state   = e.is_on,
                        });
                }
            }

            result.times.push_back(t);
            result.states.push_back(x);
            result.event_iteration_count.push_back(iters - 1);
        }
    }

    return result;
}

}  // namespace pulsim::solver
