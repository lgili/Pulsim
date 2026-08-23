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
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/dc_operating_point.hpp"
#include "pulsim/pwl/dc_strategy.hpp"
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

#include <array>
#include <cstdint>
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
    const topology::SwitchStateMask& diode_owned) {
    // user, diode, diode_owned must all be the same width. Word-wise
    // since Phase 1 (dynamic mask width — no 64-switch assumption).
    return user.overlay(diode, diode_owned);
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
// Postcondition: result.num_steps() == opts.expected_sample_count()
//                result.times[j]   == opts.t_start
//                                      + j * opts.store_every * opts.dt
//                result.states[j]  is the solution at result.times[j]
//                (with the default store_every == 1 these reduce to
//                 the v1.x form: one sample per step at k·dt. An
//                 early `should_continue` stop yields fewer samples.)
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
            "(check t_start < t_end, dt > 0, store_every >= 1, "
            "all finite)");
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
    // v2.0 Phase 1: state size up front → the contiguous sample
    // buffer is allocated ONCE for the whole run, sized by the
    // DECIMATED sample count (opts.store_every).
    result.reserve(opts.expected_sample_count(),
                    static_cast<Index>(state_size));

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
        // disturbing the recorded sample. `store_every` decimates
        // on a STRICTLY UNIFORM grid (steps 0, m, 2m, …) so the
        // recorded trace keeps a constant spacing of m·dt.
        if (k % opts.store_every == 0) {
            result.times.push_back(t);
            result.states.push_back(x);
        }
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
            "(check t_start < t_end, dt > 0, store_every >= 1, "
            "all finite)");
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
    // v2.0 Phase 1: state size up front → the contiguous sample
    // buffer is allocated ONCE for the whole run, sized by the
    // DECIMATED sample count (opts.store_every).
    result.reserve(opts.expected_sample_count(),
                    static_cast<Index>(state_size));

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
    // Warm-start: seed the saturable-inductor (i_L_old, V_L_old) from
    // `initial_state` too. Otherwise the first Newton refresh sees zeros
    // and stamps a wrong Jacobian → convergence failure or wrong solution
    // on step 1. No-op when there are no saturable inductors. Must run
    // AFTER init() and after `x = *initial_state` above.
    if (initial_state != nullptr) {
        sat_history.seed_from_dc_op(x);
    }
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
        // v2.0 Phase 2 (B.2): one shared implementation of "the DC
        // operating point" (pwl/dc_operating_point.hpp) — nonlinear
        // devices stamped, PWL diode states iterated to consistency,
        // and the DC cascade walked when the direct solve fails.
        //
        // Adversarial-review finding P0-R2 still governs which
        // refresh goes in: the RAW static-device chain (diodes /
        // MOSFET / IGBT), NOT nl_refresh_effective. The saturable-
        // inductor wrapper stamps trap-companion physics (a
        // 2·L_eff/dt series resistance), which is time-step dependent
        // and meaningless at DC.
        pwl::DCOperatingPointOptions dc_opts;
        dc_opts.t_eval = opts.t_start;
        dc_opts.max_event_iterations =
            opts.max_event_iterations > 0 ? opts.max_event_iterations
                                           : Size{16};
        dc_opts.max_newton_iters   = opts.max_newton_iterations;
        dc_opts.tol_dx             = opts.tol_newton_dx;
        dc_opts.tol_res            = opts.tol_newton_res;
        dc_opts.enable_line_search = opts.enable_newton_line_search;
        dc_opts.enable_lm          = opts.enable_newton_lm;

        auto dc = pwl::compute_dc_operating_point(
            graph, pool, switch_fn(opts.t_start), nl_refresh,
            dc_opts, has_diodes ? &diodes : nullptr,
            "run_transient(start_from_dc_op)");
        x = std::move(dc.x);
        history.seed_from_dc_op(x);
    }

    Vector b_extra(static_cast<Index>(state_size));

    // ---- v2.0 Phase 1: per-step workspace, allocated ONCE ----------------
    // (audit finding `per-step-heap-allocations`). Every buffer below was
    // previously constructed fresh inside the step loop — 6-10 allocator
    // round-trips per step. Hoisted here, per-step use is assignment /
    // clear() into retained capacity: on the LINEAR trapezoidal path the
    // steady-state loop performs zero heap allocations outside sample
    // recording. (The Newton nonlinear-refresh solve and the deliberately
    // uncached BDF1 comparison path still allocate per call — out of the
    // Phase-1 zero-alloc scope.)
    Vector x_prev;                                  // pre-step state snapshot
    pwl::PwlSegment retry_seg;                      // off-nominal-dt companion
    std::vector<pwl::HistoryEntry> history_snap;    // V3 rollback snapshot
    std::vector<bool> diodes_snap;                  // V3 diode-bit snapshot
    std::vector<bool> last_solved_bits;             // breach re-sync bits
    std::vector<CommutationEvent> step_events;      // per-step event scratch
    std::vector<bool> post_event_bits;              // substep-correction scratch

    // Event-iteration cap. 0 disables iteration entirely
    // (matching Layer 5 V2 behaviour for regression testing).
    const Size max_iters = opts.max_event_iterations;

    // Built-in time-varying source detection + reusable b_extra buffers
    // (audit #6/#16), shared by BOTH the dynamic and static paths below. The
    // PWM/sine/pulse helpers otherwise allocate + zero a state-size vector AND
    // scan every branch on every step; detect ONCE which kinds the circuit
    // contains, then reuse buffers and skip absent kinds in the hot loops.
    bool has_pwm = false, has_sine = false, has_pulse = false;
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (graph.branch(b_id).kind != topology::BranchKind::Source) {
            continue;
        }
        switch (pool.kind_of(b_id)) {
        case pwl::DevicePool::StoredKind::PWMVoltageSource:
            has_pwm = true; break;
        case pwl::DevicePool::StoredKind::SineVoltageSource:
            has_sine = true; break;
        case pwl::DevicePool::StoredKind::PulseVoltageSource:
            has_pulse = true; break;
        default:
            break;
        }
    }
    Vector be_pwm, be_sine, be_pulse;  // reused per step (filled in place)

    // v2.0 Phase 2: one accumulator for the step's right-hand side,
    // used by the nominal step, by both halves of the sub-step
    // commutation correction, and by the dt-retry path. It was
    // duplicated three times before; the order of the terms is what
    // makes results bit-identical (trap history, then the user's
    // b_extra_fn, then only the built-in source kinds this circuit
    // actually contains), so having one copy of it is also the only
    // way to keep that promise honest.
    auto accumulate_b_extra = [&](Real at_t, Real for_dt) {
        history.compute_b_extra(for_dt, b_extra);
        if (b_extra_fn) {
            b_extra += b_extra_fn(at_t);
        }
        if (has_pwm) {
            sources::compute_pwm_b_extra(pool, graph, at_t, be_pwm);
            b_extra += be_pwm;
        }
        if (has_sine) {
            sources::compute_sine_b_extra(pool, graph, at_t, be_sine);
            b_extra += be_sine;
        }
        if (has_pulse) {
            sources::compute_pulse_b_extra(pool, graph, at_t, be_pulse);
            b_extra += be_pulse;
        }
    };

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
            // correction (Layer 5 V3). Assignment into the hoisted
            // buffer — no per-step allocation.
            x_prev = x;

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
            history.snapshot_into(history_snap);
            if (has_diodes) {
                diodes.snapshot_on_bits_into(diodes_snap);
            } else {
                diodes_snap.clear();
            }
            // Pre-event mask: what mask we'd use BEFORE any
            // diode flip during this step. Captured before
            // event iteration runs.
            topology::SwitchStateMask mask_pre = switch_fn(t);
            if (has_diodes) {
                mask_pre = combine_masks(
                    mask_pre, diodes.current_diode_mask(),
                    diode_owned);
            }

            // Accumulate b_extra into the reused buffer (audit #6), in the
            // same left-to-right order as before so the result is
            // bit-identical: history, then the optional user fn, then only the
            // built-in source kinds the circuit contains. A null user fn and
            // absent source kinds contribute (and cost) nothing instead of
            // allocating + adding a zero vector every step.
            accumulate_b_extra(t, opts.dt);

            // 3. Event-iteration loop. Solve, update diode state,
            //    re-solve if any diode flipped. Stop when stable
            //    or max_iters hit.
            //
            // If `nl_refresh` was supplied, the inner solve uses
            // Newton (with the trap-companion `b_extra`); else
            // it's the cached linear solve.
            Size iters = 0;
            bool flipped = false;
            bool mask_cycle = false;
            // Full mask copies (Phase 1: masks can exceed 64 bits; a
            // copy is 2 inline words for <=128 switches — still no
            // heap in the loop).
            std::array<topology::SwitchStateMask, 64> masks_seen{};
            Size n_masks_seen = 0;

            // One step of `for_dt` ending at `at_t`: solve, let the
            // PWL diodes re-decide from the result, re-solve until
            // they agree. Factored out so the dt-retry path below can
            // run exactly the same iteration at a smaller step
            // (v2.0 Phase 2, B.4) — the nominal `for_dt == opts.dt`
            // call still takes the cached factorization and is
            // bit-identical to before.
            auto run_event_iteration = [&](Real at_t, Real for_dt) {
            iters = 0;
            flipped = false;
            mask_cycle = false;
            n_masks_seen = 0;
            // Diode on-bits consistent with the most recent solve
            // (captured BEFORE update_from_state advances them) —
            // restored on breach so x and the diode state agree
            // (adversarial-review finding P0-R1).
            last_solved_bits.clear();
            do {
                auto mask = switch_fn(at_t);
                if (has_diodes) {
                    const auto diode_mask =
                        diodes.current_diode_mask();
                    mask = combine_masks(mask, diode_mask,
                                          diode_owned);
                }
                // Phase-0 fix #9: a repeated mask within one step's
                // event iteration is a CYCLE (mask A -> B -> A ...,
                // the un-hysteresed diode pair around a resonant
                // node) — but ONLY on the linear path, where
                // cache.solve is a deterministic, memoryless
                // function of the mask, making the repeat a proof
                // of non-convergence. The Newton path warm-starts
                // from the current x, so revisiting a mask from a
                // different iterate can legitimately converge
                // (adversarial-review finding P0-R3); there the
                // iteration budget alone bounds the cost.
                if (!nl_refresh_effective) {
                    for (Size ms = 0; ms < n_masks_seen; ++ms) {
                        if (masks_seen[ms] == mask) {
                            mask_cycle = true;
                            break;
                        }
                    }
                    if (mask_cycle) break;
                    if (n_masks_seen < masks_seen.size()) {
                        masks_seen[n_masks_seen++] = mask;
                    }
                }
                if (nl_refresh_effective) {
                    if (for_dt == opts.dt) {
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
                        // Off-nominal dt: assemble the companion
                        // system for this step size. No factorization
                        // is wasted — Newton refactorizes
                        // J_lin + J_nl every iteration anyway, so the
                        // segment is only a carrier for J and b.
                        pwl::assemble_segment(
                            graph, pool, mask, for_dt,
                            retry_seg.J, retry_seg.b_constant);
                        sparse::compress_in_place(retry_seg.J);
                        retry_seg.state_size = state_size;
                        x = pwl::solve_with_newton_b_extra(
                            retry_seg, nl_refresh_effective, graph,
                            pool, /*x_init=*/x, b_extra,
                            opts.max_newton_iterations,
                            opts.tol_newton_dx,
                            opts.tol_newton_res,
                            opts.enable_newton_line_search,
                            opts.enable_newton_lm);
                    }
                } else {
                    // `solve_at` delegates to `solve` bit-identically
                    // when the dt matches, so the nominal path costs
                    // nothing extra.
                    cache.solve_at(mask, for_dt, b_extra, x);
                }
                if (has_diodes) {
                    diodes.snapshot_on_bits_into(last_solved_bits);
                }
                flipped = has_diodes &&
                          diodes.update_from_state(x);
                ++iters;
            } while (flipped && iters < max_iters);
            };  // run_event_iteration

            // --- The nominal attempt, then local step reduction ---
            //
            // A failed step used to end the run and discard
            // everything computed before it. A smaller step is the
            // standard answer and a genuinely DIFFERENT problem: the
            // trapezoidal companion's 2C/dt grows as dt shrinks,
            // which both improves the Jacobian's diagonal dominance
            // and puts the previous state closer to the answer. (A
            // retry that re-ran the identical computation would be
            // the dead-rung defect Phase 2 B.2 removed from the DC
            // cascade — this one is not that.)
            // dt of the FINAL solve of this outer step, fed to the
            // deferred trap-history commit below. opts.dt on the
            // normal path; dt2 when sub-step correction splits the
            // step; the retry's sub_dt when a step had to be
            // re-taken. Also distinguishes the corrected path — no
            // separate flag is needed.
            Real committed_dt = opts.dt;
            bool step_retried = false;
            try {
                run_event_iteration(t, opts.dt);
            } catch (const analysis::Cancelled&) {
                throw;
            } catch (const std::exception& nominal_failed) {
                if (opts.max_dt_halvings == 0) {
                    throw;
                }
                std::string why = nominal_failed.what();
                for (Size h = 1; h <= opts.max_dt_halvings; ++h) {
                    // Back to the state at t_prev. Note what is NOT
                    // restored: `sat_history`, which this step has
                    // not touched yet (it commits at the very end),
                    // and the cache, whose contents are a function of
                    // the circuit rather than of the run.
                    x = x_prev;
                    history.restore(history_snap);
                    if (has_diodes) {
                        diodes.restore_on_bits(diodes_snap);
                    }

                    const Size n_sub = Size{1} << h;
                    const Real sub_dt =
                        opts.dt / static_cast<Real>(n_sub);
                    bool all_ok = true;
                    for (Size j = 0; j < n_sub; ++j) {
                        const Real t_sub =
                            t_prev + static_cast<Real>(j + 1) * sub_dt;
                        accumulate_b_extra(t_sub, sub_dt);
                        try {
                            run_event_iteration(t_sub, sub_dt);
                        } catch (const analysis::Cancelled&) {
                            throw;
                        } catch (const std::exception& sub_failed) {
                            why = sub_failed.what();
                            all_ok = false;
                            break;
                        }
                        // Commit each sub-step before the next one
                        // reads the history — `compute_b_extra(dt)`,
                        // the solve at `dt` and `update_from_state(
                        // x, dt)` must all share the same dt or the
                        // capacitor companion current is silently
                        // wrong.
                        //
                        // EXCEPT the last one, whose commit is
                        // deferred past the inductor freeze guard
                        // exactly as the nominal path defers its own,
                        // so a snapped-back i_L is what gets baked
                        // into the history rather than the raw solve.
                        //
                        // And `sat_history` is never advanced here.
                        // It has no snapshot/restore, so a sub-step
                        // that advanced it and a later halving that
                        // rolled back would leave the flux history at
                        // a mid-step value with nothing able to undo
                        // it. Leaving it alone reproduces the nominal
                        // step's semantics exactly; it is committed
                        // once, at the end, from the final x.
                        if (j + 1 < n_sub) {
                            history.update_from_state(x, sub_dt);
                        } else {
                            committed_dt = sub_dt;
                        }
                    }
                    if (all_ok) {
                        step_retried = true;
                        result.dt_retries.push_back(
                            solver::DtRetry{t, h,
                                             nominal_failed.what()});
                        break;
                    }
                }
                if (!step_retried) {
                    throw std::runtime_error(std::format(
                        "run_transient: the step ending at t = {} "
                        "could not be taken, even split into {} "
                        "sub-steps of {:.3e} s (dt/{}). Last failure: "
                        "{}\nThis is no longer a step-size problem — "
                        "raise max_dt_halvings only if you believe "
                        "otherwise; more likely the circuit needs a "
                        "snubber, a softer device model, or a look at "
                        "the device the message above names.",
                        t, Size{1} << opts.max_dt_halvings,
                        opts.dt / static_cast<Real>(
                            Size{1} << opts.max_dt_halvings),
                        Size{1} << opts.max_dt_halvings, why));
                }
            }

            if (flipped || mask_cycle) {
                if (opts.strict_event_iterations) {
                    // v2.0 Phase 1: name the diodes that were still
                    // flipping. "iteration hit a mask cycle at
                    // t=3.7e-3" tells a user nothing about WHICH pair
                    // is chattering; on a 200-diode rectifier bank
                    // that is the whole question.
                    std::string culprits;
                    if (has_diodes && !last_solved_bits.empty()) {
                        const auto& entries = diodes.entries();
                        for (Size i = 0; i < entries.size() &&
                                          i < last_solved_bits.size(); ++i) {
                            if (entries[i].is_on == last_solved_bits[i]) {
                                continue;   // stable across the last solve
                            }
                            if (!culprits.empty()) culprits += ", ";
                            culprits += pwl::branch_label(
                                graph, entries[i].branch_id);
                        }
                    }
                    throw std::runtime_error(std::format(
                        "run_transient: diode event iteration {} at "
                        "t = {} after {} solves "
                        "(strict_event_iterations=true){}; raise "
                        "max_event_iterations, reduce dt, or give the "
                        "offending device(s) a small hysteresis band",
                        mask_cycle ? "hit a mask cycle"
                                    : "exhausted its budget",
                        t, iters,
                        culprits.empty()
                            ? std::string{}
                            : std::format(" — still flipping: {}",
                                           culprits)));
                }
                // Accept the last consistent solve, flag the step,
                // keep going — losing the whole run was the greater
                // wrong (audit: event-iteration-throws-away-
                // simulation, CONFIRMED). Re-sync the diode bits to
                // the state x was actually solved under, so the
                // commutation records, trap history and the next
                // step all start from a CONSISTENT (x, diode) pair
                // (adversarial-review finding P0-R1).
                if (has_diodes && !last_solved_bits.empty()) {
                    diodes.restore_on_bits(last_solved_bits);
                }
                result.event_iteration_breaches.push_back(
                    {.t = t, .iterations = iters,
                     .cycle_detected = mask_cycle});
            }

            // 4. Sub-step commutation timing (Layer 5 V2.2) +
            //    state correction (Layer 5 V3).
            //
            // Detect zero-crossing events for diode signals
            // and (if enabled) retroactively split the step
            // into two sub-steps at the first detected
            // event's t_est.
            step_events.clear();
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
            // dt of the FINAL solve in this outer step, fed to the deferred
            // trap-history commit below. Stays opts.dt on the normal path;
            // becomes dt2 when sub-step correction splits the step (sub-step 1
            // commits the dt1 half inline; the dt2 commit is deferred past the
            // freeze guard). This also distinguishes the corrected path — no
            // separate `corrected` flag is needed.
            if (opts.enable_substep_state_correction &&
                !step_retried && !step_events.empty()) {
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
                        // Accumulate the sub-step b_extra into the reused
                        // outer buffer (audit #16), in the same order as
                        // before — history, user fn, then present source kinds
                        // — so the result is bit-identical. The outer
                        // `b_extra` is free here: the main event-iteration
                        // solve already consumed it this step.
                        accumulate_b_extra(t_est, dt1);
                        cache.solve_at(
                            mask_pre, dt1, b_extra, x);
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
                        post_event_bits = diodes_snap;
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
                        // Same reused-buffer accumulation as sub-step 1
                        // (audit #16), at time t with dt2.
                        accumulate_b_extra(t, dt2);
                        cache.solve_at(
                            mask_post, dt2, b_extra, x);
                        // History commit DEFERRED to after the freeze guard
                        // (below): the guard must be able to snap an
                        // unphysical i_L back BEFORE it is baked into the
                        // trap history. Sub-step 1 already committed the dt1
                        // half above; this records dt2 as the final commit dt.
                        committed_dt = dt2;
                    }
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
            if (opts.inductor_freeze_di_max > Real{0} ||
                opts.inductor_abs_clamp > Real{0}) {
                for (const auto& e : history.entries()) {
                    if (e.kind != pwl::DevicePool::StoredKind::Inductor) {
                        continue;
                    }
                    // Baseline = this step's PRE-step current, read from
                    // x_prev rather than history.i_prev. Under sub-step
                    // correction, sub-step 1 already advanced history.i_prev
                    // to the mid-step value, so comparing against it would
                    // miss the jump (di ≈ 0). x_prev is the immutable
                    // start-of-step state across both sub-steps.
                    const Real i_prev_step =
                        x_prev[e.inductor_branch_var_id];
                    Real i_new = x[e.inductor_branch_var_id];
                    // Step-to-step jump guard: snap back to the pre-step
                    // current when the solver emits an unphysical kilo-amp
                    // delta (rectifier in DCM, etc.).
                    if (opts.inductor_freeze_di_max > Real{0}) {
                        const Real di  = i_new - i_prev_step;
                        const Real adi = di < Real{0} ? -di : di;
                        if (adi > opts.inductor_freeze_di_max) {
                            i_new = i_prev_step;
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

            // 5. Commit history for the next step. Deferred until AFTER the
            //    freeze guard so a snapped-back i_L is the value baked into
            //    the trap history. `committed_dt` is opts.dt on the normal
            //    path and dt2 when sub-step correction split the step (whose
            //    sub-step 1 already committed the dt1 half above).
            history.update_from_state(x, committed_dt);
            if (has_saturable) {
                sat_history.update_from_state(x);
            }

            // 6. Record. event_iteration_count = iters - 1 (the
            //    first iteration always runs; the count is the
            //    number of EXTRA solves caused by diode flips).
            // Decimated recording (v2.0 Phase 1): uniform grid at
            // m·dt. Sample 0 (the IC, k = 0) is always recorded
            // above, keeping the parallel arrays aligned.
            if (k % opts.store_every == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(iters - 1);
            }
        }
    } else {
        // ----------- STATIC PATH ----------------------------------------
        //
        // No dynamic devices → cache.solve gives the DC operating
        // point for the requested switch state. Diode iteration
        // still applies.
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
            x_prev = x;

            // State-aware observer fires before switch_fn(t) so
            // discrete-time controllers can update.
            if (step_observer) {
                step_observer(t, x_prev);
            }

            // Accumulate into the reused buffer (audit #6/#16): optional user
            // fn, then present built-in source kinds. No trap history on the
            // static path. Same summation order → bit-identical; absent kinds
            // and a null user fn cost nothing. `b_extra_user` aliases the
            // buffer so the solves below are unchanged.
            b_extra.setZero();
            if (b_extra_fn) {
                b_extra += b_extra_fn(t);
            }
            if (has_pwm) {
                sources::compute_pwm_b_extra(pool, graph, t, be_pwm);
                b_extra += be_pwm;
            }
            if (has_sine) {
                sources::compute_sine_b_extra(pool, graph, t, be_sine);
                b_extra += be_sine;
            }
            if (has_pulse) {
                sources::compute_pulse_b_extra(pool, graph, t, be_pulse);
                b_extra += be_pulse;
            }
            const Vector& b_extra_user = b_extra;

            Size iters = 0;
            bool flipped = false;
            bool mask_cycle = false;
            // Full mask copies (Phase 1: masks can exceed 64 bits; a
            // copy is 2 inline words for <=128 switches — still no
            // heap in the loop).
            std::array<topology::SwitchStateMask, 64> masks_seen{};
            Size n_masks_seen = 0;

            // One step of `for_dt` ending at `at_t`: solve, let the
            // PWL diodes re-decide from the result, re-solve until
            // they agree. Factored out so the dt-retry path below can
            // run exactly the same iteration at a smaller step
            // (v2.0 Phase 2, B.4) — the nominal `for_dt == opts.dt`
            // call still takes the cached factorization and is
            // bit-identical to before.
            // No dt-retry on this path, deliberately. With no
            // capacitors or inductors there is no companion term, so
            // dt does not enter the matrix at all and a smaller step
            // would re-run the byte-identical computation — the
            // dead-rung defect Phase 2 B.2 removed from the DC
            // cascade.
            // Diode on-bits consistent with the most recent solve
            // (captured BEFORE update_from_state advances them) —
            // restored on breach so x and the diode state agree
            // (adversarial-review finding P0-R1).
            last_solved_bits.clear();
            do {
                auto mask = switch_fn(t);
                if (has_diodes) {
                    const auto diode_mask =
                        diodes.current_diode_mask();
                    mask = combine_masks(mask, diode_mask,
                                          diode_owned);
                }
                // Phase-0 fix #9: a repeated mask within one step's
                // event iteration is a CYCLE (mask A -> B -> A ...,
                // the un-hysteresed diode pair around a resonant
                // node) — but ONLY on the linear path, where
                // cache.solve is a deterministic, memoryless
                // function of the mask, making the repeat a proof
                // of non-convergence. The Newton path warm-starts
                // from the current x, so revisiting a mask from a
                // different iterate can legitimately converge
                // (adversarial-review finding P0-R3); there the
                // iteration budget alone bounds the cost.
                if (!nl_refresh_effective) {
                    for (Size ms = 0; ms < n_masks_seen; ++ms) {
                        if (masks_seen[ms] == mask) {
                            mask_cycle = true;
                            break;
                        }
                    }
                    if (mask_cycle) break;
                    if (n_masks_seen < masks_seen.size()) {
                        masks_seen[n_masks_seen++] = mask;
                    }
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
                if (has_diodes) {
                    diodes.snapshot_on_bits_into(last_solved_bits);
                }
                flipped = has_diodes &&
                          diodes.update_from_state(x);
                ++iters;
            } while (flipped && iters < max_iters);

            if (flipped || mask_cycle) {
                if (opts.strict_event_iterations) {
                    // Same naming as the dynamic path — the static
                    // path runs whenever the cache was built at
                    // dt = 0, and a user there deserves the same
                    // answer to "which device is chattering?"
                    // (adversarial-review finding F2: only the
                    // dynamic throw had been updated).
                    std::string culprits;
                    if (has_diodes && !last_solved_bits.empty()) {
                        const auto& entries = diodes.entries();
                        for (Size i = 0; i < entries.size() &&
                                          i < last_solved_bits.size(); ++i) {
                            if (entries[i].is_on == last_solved_bits[i]) {
                                continue;
                            }
                            if (!culprits.empty()) culprits += ", ";
                            culprits += pwl::branch_label(
                                graph, entries[i].branch_id);
                        }
                    }
                    throw std::runtime_error(std::format(
                        "run_transient: diode event iteration {} at "
                        "t = {} after {} solves "
                        "(strict_event_iterations=true){}; raise "
                        "max_event_iterations, reduce dt, or give the "
                        "offending device(s) a small hysteresis band",
                        mask_cycle ? "hit a mask cycle"
                                    : "exhausted its budget",
                        t, iters,
                        culprits.empty()
                            ? std::string{}
                            : std::format(" — still flipping: {}",
                                           culprits)));
                }
                // P0-R1: re-sync diode bits to the solved state.
                if (has_diodes && !last_solved_bits.empty()) {
                    diodes.restore_on_bits(last_solved_bits);
                }
                result.event_iteration_breaches.push_back(
                    {.t = t, .iterations = iters,
                     .cycle_detected = mask_cycle});
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

            if (k % opts.store_every == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(iters - 1);
            }
        }
    }

    return result;
}

}  // namespace pulsim::solver
