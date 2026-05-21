#pragma once

// =============================================================================
// Pulsim v2 — Layer 5: run_transient (fixed-dt time-stepping loop)
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

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/dc_assemble.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/diode_event_state.hpp"
#include "pulsim/v2/pwl/history_state.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/result.hpp"
#include "pulsim/v2/topology/graph.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>

namespace pulsim::v2::solver {

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
[[nodiscard]] inline Real interp_commutation_time(
    Real t_prev, Real t_curr,
    Real s_prev, Real s_curr) noexcept {
    // Same sign → no crossing in the interval; clamp.
    if (s_prev * s_curr > Real{0}) {
        return t_curr;
    }
    const Real denom = s_prev - s_curr;
    if (std::abs(denom) < std::numeric_limits<Real>::min()) {
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
    const BExtraFn& b_extra_fn = {}) {

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
    const pwl::NonlinearRefreshFn& nl_refresh = {}) {

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
            x = pwl::compute_dc_op(graph, pool, mask);
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

        for (Size k = 1; k < n_steps; ++k) {
            const Real t = opts.t_start +
                            static_cast<Real>(k) * opts.dt;
            const Real t_prev = opts.t_start +
                                 static_cast<Real>(k - 1) * opts.dt;

            // Snapshot the state at t_prev for sub-step
            // commutation timing (Layer 5 V2.2).
            const Vector x_prev = x;

            // 1. History from previous step.
            const Vector b_extra_history =
                history.compute_b_extra(opts.dt);

            // 2. Optional user-supplied b_extra(t).
            const Vector b_extra_user = b_extra_fn
                ? b_extra_fn(t)
                : Vector::Zero(state_size);
            b_extra = b_extra_history + b_extra_user;

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
                if (nl_refresh) {
                    const auto& seg = cache.lookup(mask);
                    x = pwl::solve_with_newton_b_extra(
                        seg, nl_refresh, graph, pool,
                        /*x_init=*/x, b_extra,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search);
                } else {
                    cache.solve(mask, b_extra, x);
                }
                flipped = has_diodes &&
                          diodes.update_from_state(x);
                ++iters;
            } while (flipped && iters < max_iters);

            if (flipped) {
                throw std::runtime_error(
                    "run_transient: event-iteration limit "
                    "reached without convergence at t = " +
                    std::to_string(t) + "; raise "
                    "max_event_iterations or reduce dt");
            }

            // 4. Sub-step commutation timing (Layer 5 V2.2).
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
                    result.commutation_events.push_back(
                        CommutationEvent{
                            .t_estimated = t_est,
                            .branch_id   = e.branch_id,
                            .new_state   = e.is_on,
                        });
                }
            }

            // 5. Commit history for the next step.
            history.update_from_state(x, opts.dt);

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

            const Vector b_extra_user = b_extra_fn
                ? b_extra_fn(t)
                : zero_b_extra;

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
                if (nl_refresh) {
                    const auto& seg = cache.lookup(mask);
                    x = pwl::solve_with_newton_b_extra(
                        seg, nl_refresh, graph, pool,
                        /*x_init=*/x, b_extra_user,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search);
                } else {
                    cache.solve(mask, b_extra_user, x);
                }
                flipped = has_diodes &&
                          diodes.update_from_state(x);
                ++iters;
            } while (flipped && iters < max_iters);

            if (flipped) {
                throw std::runtime_error(
                    "run_transient: event-iteration limit "
                    "reached without convergence at t = " +
                    std::to_string(t));
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

}  // namespace pulsim::v2::solver
