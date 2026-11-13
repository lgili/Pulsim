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
#include "pulsim/pwl/preflight.hpp"
#include "pulsim/pwl/dc_strategy.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/nonlinear_refresh_igbt_tail.hpp"
#include "pulsim/pwl/nonlinear_refresh_lauritzen_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/pwl/nonlinear_refresh_nonlinear_capacitor.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/source_jumps.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <functional>
#include <optional>
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
//   v_diode − V_on for BOTH directions, V_on = SwitchedDiode::
//   turn_on_voltage (= V_th at V_f = 0). Exact both ways when
//   V_th ≤ V_f (both events at v = V_f); with V_th > V_f the ON → OFF
//   time is biased by the band (pre-existing, see trbdf2's
//   crossing_signal for the per-direction form).
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
            if (!result.has_switch_states) {
                // No graph here; the mask itself carries the width.
                result.begin_switch_state_record(
                    mask.size(), opts.expected_sample_count());
            }
            result.times.push_back(t);
            result.states.push_back(x);
            // No diodes on this overload: the stamped mask is the
            // one switch_fn returned for this solve.
            result.record_switch_state(mask);
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
    const ShouldContinueFn& should_continue = {},
    /// v2.0 Phase 4 — resume EXACTLY from a previous run's
    /// `final_snapshot`. Unlike `initial_state`, which restores x
    /// and INVENTS the companion history from it, this restores
    /// the history and the solver-owned diode bits too.
    const SolverSnapshot* resume_from = nullptr,
    /// add-discontinuity-restart-rule — the instants at which
    /// `b_extra_fn` jumps, declared by the caller: the first jump
    /// strictly after t, +inf when none. Only read with
    /// `discontinuity_restart`; an undeclared callback jump is not
    /// restarted (and is not detected).
    const std::function<Real(Real)>& b_extra_next_jump = {},
    /// add-fixed-step-event-splitting — the schedule's analytic gate
    /// edges (`SwitchSchedule::next_edge_after`): the first edge
    /// strictly after t, +inf when none. Empty: the gate edges of an
    /// opaque switch_fn are not located (start-of-step sampling).
    const std::function<Real(Real)>& next_edge_fn = {}) {

    // ---- Input validation ---------------------------------------------
    if (!opts.valid()) {
        throw std::invalid_argument(
            "run_transient: SimulationOptions are not valid "
            "(check t_start < t_end, dt > 0, store_every >= 1, "
            "all finite)");
    }
    if (opts.enable_substep_state_correction && !opts.discontinuity_restart) {
        // The event split ends every step that contains an event with
        // the restart rule's SDIRK window; without the rule there is no
        // window to end it with.
        throw UnsupportedError(
            "pwl", "enable_substep_state_correction", "", "",
            "run_transient: enable_substep_state_correction=True requires "
            "discontinuity_restart=True (the event split ends every step "
            "that contains an event with the restart rule's SDIRK "
            "window); set both, or neither");
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
    result.begin_switch_state_record(
        static_cast<Size>(graph.num_switches()),
        opts.expected_sample_count());

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
    if (resume_from != nullptr && !resume_from->valid) {
        // Refuse by name: a nullptr here used to start the run
        // from zero with no signal.
        throw std::invalid_argument(
            "run_transient(resume_from=...): the snapshot is not "
            "valid — it came from a run that produced none (an "
            "aborted run, or a SolverSnapshot() built by hand and "
            "never populated). Resuming from it would start from "
            "zero without saying so.");
    }
    const SolverSnapshot* resume_snap = resume_from;
    if (resume_snap != nullptr) {
        if (resume_from->x.size() != static_cast<Index>(state_size)) {
            throw std::invalid_argument(
                "run_transient(resume_from=...): the snapshot's "
                "state vector has " +
                std::to_string(resume_from->x.size()) +
                " entries but this circuit has " +
                std::to_string(state_size)
                + " — it belongs to a different circuit.");
        }
        x = resume_from->x;
        history.from_flat(resume_from->history);
    }

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
    // Resume WINS over a builder-IC seed: the snapshot is the state.
    if (resume_snap != nullptr) sat_history.from_flat(resume_snap->saturable_history);
    const bool has_saturable = !sat_history.empty();

    // V17: shared dt that the saturable-inductor refresh
    // reads each iteration. Updated when sub-step correction
    // splits dt into dt1 + dt2.
    // Fixed for the whole run. An earlier comment here claimed it
    // was "updated when sub-step correction splits dt into dt1+dt2";
    // it never was, and it is safe only because that path solves
    // linearly through cache.solve_at and never consults this. The
    // dt-retry refuses saturable-inductor circuits for the same
    // reason — see `dt_retry_can_help`.
    Real refresh_dt = opts.dt;
    // The rule of the solve in flight, set beside refresh_dt at the one
    // point every solve passes through (run_event_iteration): the
    // SDIRK stages of discontinuity_restart are BackwardEuler, with
    // refresh_dt = the γ·h each stage advances.
    pwl::TrBdf2Stage refresh_stage = pwl::TrBdf2Stage::Trapezoidal;

    // V17: if any saturable inductors are present, wrap the
    // user's nl_refresh with our additive saturable
    // stamping. The user-supplied refresh runs first (it
    // zero-clears + stamps diodes/MOSFETs/IGBTs); we then
    // ADD the saturable inductor contributions on top.
    // Phase 4 C.1 — charge-based Coss. Unlike the saturable
    // inductor below, this one is handed the ACTUAL step size each
    // solve (`refresh_dt` is assigned before every solve on every
    // path, including the sub-step and retry ones) and its history
    // has snapshot/restore, so a step that is thrown away rolls
    // back. Those two omissions are exactly why the saturable
    // inductor has to refuse a variable step.
    pwl::NonlinearCapacitorHistory coss_history;
    coss_history.init(graph, pool);
    if (initial_state != nullptr) {
        coss_history.seed_from_dc_op(x);
    }
    // Resume WINS over a builder-IC seed: the snapshot is the state.
    if (resume_snap != nullptr) coss_history.from_flat(resume_snap->coss_history);
    const bool has_coss = !coss_history.empty();

    // The Lauritzen diode's stored charge, same shape as the Coss
    // history above: state that the Newton stamp reads and the
    // step commits, snapshot/restored on rejection.
    pwl::LauritzenDiodeHistory laur_history;
    laur_history.init(graph, pool);
    if (initial_state != nullptr) {
        laur_history.seed_from_dc_op(x);
    }
    // Resume WINS over a builder-IC seed: the snapshot is the state.
    if (resume_snap != nullptr) laur_history.from_flat(resume_snap->lauritzen_history);
    const bool has_lauritzen = !laur_history.empty();

    // Phase 4 C.3 — the MNA-native PMSM's flux-linkage state. Same
    // shape as the Lauritzen diode: stamped per Newton iteration
    // from committed history, snapshot/restored on rejection.
    pwl::PmsmMnaHistory pmsm_history;
    pmsm_history.init(graph, pool);
    if (initial_state != nullptr) {
        pmsm_history.seed_from_dc_op(x);
    }
    // Resume WINS over a builder-IC seed: the snapshot is the state.
    if (resume_snap != nullptr) pmsm_history.from_flat(resume_snap->pmsm_history);
    const bool has_pmsm = !pmsm_history.empty();

    // The IGBT turn-off tail's stored charge. Same shape again,
    // except this pass stamps a DELTA on top of the ordinary IGBT
    // stamp rather than replacing it — see the refresh header.
    pwl::IgbtTailHistory tail_history;
    tail_history.init(graph, pool);
    if (initial_state != nullptr) {
        tail_history.seed_from_dc_op(x);
    }
    // Resume WINS over a builder-IC seed: the snapshot is the state.
    if (resume_snap != nullptr) tail_history.from_flat(resume_snap->igbt_tail_history);
    const bool has_igbt_tail = !tail_history.empty();

    // The static Newton chain is the kernel's: with no callback from
    // the caller, Newton devices get the standard one instead of being
    // simulated as open circuits (the raw binding used to default to
    // that: 10.000 V on a clamp that reads 0.594 V).
    const pwl::NonlinearRefreshFn nl_static =
        (nl_refresh || !pool.requires_static_refresh())
            ? nl_refresh
            : pwl::make_combined_diode_mosfet_refresh();
    pwl::NonlinearRefreshFn nl_refresh_effective = nl_static;
    if (has_coss) {
        nl_refresh_effective =
            [user_refresh = nl_refresh_effective, &coss_history,
             &refresh_dt, &refresh_stage](const Vector& xx,
                            sparse::Matrix& J_nl,
                            Vector& f_nl,
                            const topology::Graph& g,
                            const pwl::DevicePool& pp) -> Real {
                Real max_i = Real{0};
                if (user_refresh) {
                    max_i = user_refresh(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(
                    max_i,
                    pwl::refresh_nonlinear_capacitors(
                        xx, J_nl, f_nl, coss_history,
                        refresh_dt,
                        refresh_stage == pwl::TrBdf2Stage::BackwardEuler
                            ? pwl::CossStage::BackwardEuler
                            : pwl::CossStage::Trapezoidal));
            };
    }
    if (has_lauritzen) {
        nl_refresh_effective =
            [user_refresh = nl_refresh_effective, &laur_history,
             &refresh_dt, &refresh_stage](const Vector& xx,
                            sparse::Matrix& J_nl,
                            Vector& f_nl,
                            const topology::Graph& g,
                            const pwl::DevicePool& pp) -> Real {
                Real max_i = Real{0};
                if (user_refresh) {
                    max_i = user_refresh(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(
                    max_i,
                    pwl::refresh_lauritzen_diodes(
                        xx, J_nl, f_nl, laur_history,
                        refresh_dt, refresh_stage));
            };
    }
    if (has_pmsm) {
        nl_refresh_effective =
            [user_refresh = nl_refresh_effective, &pmsm_history,
             &refresh_dt, &refresh_stage](const Vector& xx,
                            sparse::Matrix& J_nl,
                            Vector& f_nl,
                            const topology::Graph& g,
                            const pwl::DevicePool& pp) -> Real {
                Real max_i = Real{0};
                if (user_refresh) {
                    max_i = user_refresh(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(
                    max_i,
                    pwl::refresh_pmsm_mna(
                        xx, J_nl, f_nl, pmsm_history, refresh_dt,
                        refresh_stage));
            };
    }
    if (has_igbt_tail) {
        nl_refresh_effective =
            [user_refresh = nl_refresh_effective, &tail_history,
             &refresh_dt, &refresh_stage](const Vector& xx,
                            sparse::Matrix& J_nl,
                            Vector& f_nl,
                            const topology::Graph& g,
                            const pwl::DevicePool& pp) -> Real {
                Real max_i = Real{0};
                if (user_refresh) {
                    max_i = user_refresh(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(
                    max_i,
                    pwl::refresh_igbt_tails(
                        xx, J_nl, f_nl, tail_history,
                        refresh_dt, refresh_stage));
            };
    }
    if (has_saturable) {
        nl_refresh_effective =
            // `nl_refresh_effective`, NOT the raw `nl_refresh`.
            // This wrapper is installed LAST, so capturing the raw
            // user callback discarded every wrapper before it: a
            // circuit with a saturable inductor and any of {Coss,
            // Lauritzen diode, MNA-native PMSM, IGBT tail} lost all
            // four of those Newton stamps in silence. Measured: an
            // electrically DISCONNECTED saturable inductor on its
            // own node dropped an IGBT's turn-off energy from
            // 16.87 mJ to 0.045 mJ — exactly the no-tail value.
            // The other four wrappers (lines above) always captured
            // the accumulated one; this was the odd one out.
            [user_refresh = nl_refresh_effective, &sat_history,
             &refresh_dt, &refresh_stage](const Vector& x,
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
                // Additive saturable-inductor stamping, through
                // the one shared helper — this file used to carry a
                // third byte-identical copy of the body, alongside
                // the two in nonlinear_refresh_saturable_inductor.hpp.
                for (const auto& e : sat_history.entries()) {
                    max_i = std::max(
                        max_i,
                        pwl::stamp_saturable_inductor(
                            e, x, J_nl, f_nl, refresh_dt,
                            refresh_stage));
                }
                return max_i;
            };
    }

    pwl::DiodeEventState diodes{graph, pool};
    diodes.reset();
    if (resume_snap != nullptr && !resume_snap->diode_on.empty()) {
        // The solver-owned conduction bits the snapshot promised to
        // carry (size-checked by name inside).
        diodes.restore_on_bits(resume_snap->diode_on);
    }
    const auto diode_owned = diodes.diode_owned_bits();
    const bool has_diodes = diodes.num_diodes() > 0;

    // ---- DC operating-point pre-charge (Layer 5 V3) ------------------
    //
    // When the user requests `start_from_dc_op = true`, compute
    // the DC steady-state at t_start's switch configuration,
    // iterate diode consistency at the DC system, and seed
    // HistoryState + DiodeEventState from the DC solution.
    // Sample 0 becomes the DC state vector instead of zero.
    // The mask stamped for sample 0 (`SimulationResult::
    // switch_state_bits`). A DC operating point was SOLVED under
    // `dc.mask`; a zero IC is consistent with every mask (v = 0, so
    // i = g·v = 0 either way) and records the mask the run starts
    // with; a `resume_from` state records the snapshot's
    // `stamped_mask`; a state seeded from `initial_state` (or a
    // snapshot without that mask) was solved by no mask this run
    // knows and is flagged unsolved.
    topology::SwitchStateMask initial_mask(
        static_cast<std::size_t>(graph.num_switches()));
    bool initial_mask_known = false;
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
            graph, pool, switch_fn(opts.t_start), nl_static,
            dc_opts, has_diodes ? &diodes : nullptr,
            "run_transient(start_from_dc_op)");
        x = std::move(dc.x);
        history.seed_from_dc_op(x);
        initial_mask = dc.mask;
        initial_mask_known = true;
    }
    // A resume's x IS the previous run's last solve; the snapshot
    // carries the mask it was stamped under (not diode_on +
    // switch_fn(t_start), which at a gate edge is the post-edge
    // circuit). Without one (hand-built / older snapshot) the row is
    // flagged unsolved below.
    if (!initial_mask_known && resume_snap != nullptr &&
        snapshot_stamped_mask(*resume_snap,
                              static_cast<Size>(graph.num_switches()),
                              initial_mask)) {
        initial_mask_known = true;
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
    std::vector<Real> history_snap;                 // rollback: (v_prev, i_prev) pairs
    // Phase 4 C.1 — the Coss rolls back with everything else. The
    // saturable inductor cannot, which is the second reason it
    // refuses a variable step; this device was given the
    // capability up front rather than as a follow-up.
    std::vector<pwl::NonlinearCapacitorHistory::Entry> coss_snap;
    std::vector<pwl::LauritzenDiodeHistory::Entry> laur_snap;
    std::vector<pwl::PmsmMnaHistory::Entry> pmsm_snap;
    std::vector<pwl::IgbtTailHistory::Entry> tail_snap;
    std::vector<pwl::SaturableInductorHistory::Entry> sat_snap;
    std::vector<bool> diodes_snap;                  // V3 diode-bit snapshot
    std::vector<bool> last_solved_bits;             // breach re-sync bits
    std::vector<CommutationEvent> step_events;      // per-step event scratch
    std::vector<bool> post_event_bits;              // event-split scratch
    // add-fixed-step-event-splitting workspace: the step's first solve
    // (pre-event bits) and its mask, the located gate/source events,
    // the diode events located in the step.
    Vector x_first;
    topology::SwitchStateMask mask_first(
        static_cast<std::size_t>(graph.num_switches()));
    constexpr Size kMaxStepEvents = 8;
    std::array<Real, kMaxStepEvents> ev_t{};
    std::vector<CommutationEvent> split_events;
    split_events.reserve(8);
    std::vector<Real> cross_scratch;
    Real edge_from = std::numeric_limits<Real>::infinity();   // oracle memo:
    Real edge_next = -std::numeric_limits<Real>::infinity();  // first edge after edge_from
    // The mask of the LAST solve that produced x — recorded with
    // each stored sample (SimulationResult::switch_state_bits).
    topology::SwitchStateMask stamped_mask(
        static_cast<std::size_t>(graph.num_switches()));
    // Whether `stamped_mask` is the mask x was solved under (false
    // only while x is still a seeded, unsolved sample 0) — what
    // final_snapshot.stamped_mask carries.
    bool stamped_known = false;

    // Event-iteration cap. 0 disables iteration entirely
    // (matching Layer 5 V2 behaviour for regression testing).
    const Size max_iters = opts.max_event_iterations;

    // Built-in time-varying source detection + reusable b_extra buffers
    // (audit #6/#16), shared by BOTH the dynamic and static paths below. The
    // PWM/sine/pulse helpers otherwise allocate + zero a state-size vector AND
    // scan every branch on every step; detect ONCE which kinds the circuit
    // contains, then reuse buffers and skip absent kinds in the hot loops.
    // update-fixed-step-loop-bookkeeping: the (branch, row) lists are
    // built once, in the branch order of the dense passes, and each step
    // writes its rows directly. `b[row] += (0 + (−v))` is the dense pass's
    // `out[row] += −v` on a zeroed buffer followed by `b += out`, bit for
    // bit; the rows a source does not own skip `b + (+0.0)`, which only
    // differs from `b` for a −0.0 entry (checked by byte comparison).
    using SrcRows = std::vector<std::pair<Index, Index>>;
    SrcRows pwm_rows, sine_rows, pulse_rows;
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& br = graph.branch(b_id);
        if (br.kind != topology::BranchKind::Source) {
            continue;
        }
        switch (pool.kind_of(br.id)) {
        case pwl::DevicePool::StoredKind::PWMVoltageSource:
            pwm_rows.emplace_back(br.id,
                pool.branch_var_id_for_source(br.id, graph)); break;
        case pwl::DevicePool::StoredKind::SineVoltageSource:
            sine_rows.emplace_back(br.id,
                pool.branch_var_id_for_source(br.id, graph)); break;
        case pwl::DevicePool::StoredKind::PulseVoltageSource:
            pulse_rows.emplace_back(br.id,
                pool.branch_var_id_for_source(br.id, graph)); break;
        default:
            break;
        }
    }
    const bool has_pwm = !pwm_rows.empty();
    const bool has_pulse = !pulse_rows.empty();
    // `src_t`: PWM/pulse sample time; `sine_t`: the sine's; `lb`: a
    // jumping source at `src_t` reads value_at(src_t − lb) (as the dense
    // passes did).
    auto add_source_rows = [&](Vector& b, Real src_t, Real sine_t,
                               Real lb) {
        const auto read_t = [&](Index id) {
            return lb > Real{0} && sources::source_jumps_at(pool, id, src_t)
                ? src_t - lb : src_t;
        };
        for (const auto& [id, row] : pwm_rows) {
            b[row] += Real{0} + -models::PWMVoltageSource::value_at(
                pool.pwm_voltage_source_params(id), read_t(id));
        }
        for (const auto& [id, row] : sine_rows) {
            b[row] += Real{0} + -models::SineVoltageSource::value_at(
                pool.sine_voltage_source_params(id), sine_t);
        }
        for (const auto& [id, row] : pulse_rows) {
            b[row] += Real{0} + -models::PulseVoltageSource::value_at(
                pool.pulse_voltage_source_params(id), read_t(id));
        }
    };

    // Is a smaller step even capable of helping? Reachability to
    // ground is a property of the GRAPH, not of dt, so a circuit
    // with a galvanically isolated subnet is singular at every step
    // size. Retrying one would burn the whole ladder and then bury
    // the named diagnostic Phase 1 built for exactly that failure
    // under a "could not be taken, even split into 64 sub-steps"
    // wrapper — which is the dead-rung defect Phase 2 B.2 removed
    // from the DC cascade, reintroduced here. Decide once, before
    // the loop, and let those circuits fail fast and legibly.
    //
    // A SATURABLE INDUCTOR used to disqualify the retry outright,
    // for two reasons that are both gone now.
    //
    // The first was that its stamp divided by `refresh_dt` while
    // that was pinned at opts.dt for the whole run, so a sub-step
    // would have assembled every capacitor and plain inductor at
    // 2C/sub_dt while the saturable one still used 2L/opts.dt — 64x
    // out at the bottom of the ladder — and committed the mixed-dt
    // answer as good. The Coss work (Phase 4 C.1) made `refresh_dt`
    // the ACTUAL step of every solve on every path, which fixed this
    // device too as a side effect; the refusal simply outlived it.
    //
    // The second was that `SaturableInductorHistory` had no
    // snapshot/restore, so a sub-step that advanced the flux and a
    // deeper halving that rolled back would leave the history at a
    // mid-step value with nothing able to undo it. It has both now,
    // and `sat_snap` is taken and restored alongside the other four.
    const bool dt_retry_can_help =
        opts.max_dt_halvings > 0 &&
        pwl::detail::components_without_ground(
            graph, [](const topology::Branch&) { return true; })
            .empty();

    // ---- update-fixed-step-loop-bookkeeping ----
    // The step's rollback snapshots are read only by the dt-retry ladder,
    // the restart rule's re-take and the split; without any of them they
    // are not taken (`rollback_snapshots` counts the ones that are).
    const bool may_roll_back = dt_retry_can_help ||
        opts.discontinuity_restart || opts.enable_substep_state_correction;
    // Event-iteration cycle scratch, hoisted (reset per iteration).
    std::array<topology::SwitchStateMask, 64> masks_seen{};
    Size n_masks_seen = 0;
    // `switch_fn` once per distinct solve time within an outer step:
    // the observer runs before the first read, so a schedule that is a
    // function of t and of observer-written state sees the same masks.
    constexpr Size kGateMemo = 16;
    std::array<Real, kGateMemo> gate_memo_t{};
    std::array<topology::SwitchStateMask, kGateMemo> gate_memo_m{};
    Size gate_memo_n = 0;
    auto gate_at = [&](Real tq) -> const topology::SwitchStateMask& {
        for (Size q = 0; q < gate_memo_n; ++q) {
            if (gate_memo_t[q] == tq) return gate_memo_m[q];
        }
        ++result.switch_fn_calls;
        const Size slot = gate_memo_n < kGateMemo ? gate_memo_n++
                                                  : kGateMemo - 1;
        gate_memo_t[slot] = tq;
        gate_memo_m[slot] = switch_fn(tq);
        return gate_memo_m[slot];
    };
    // Last nominal-dt segment, valid while the cache generation holds
    // (a lazy insert/eviction or a parametric refactor bumps it). A hit
    // still writes the LRU tick, so eviction order is unchanged.
    topology::SwitchStateMask seg_memo_mask;
    const pwl::PwlSegment* seg_memo = nullptr;
    std::uint64_t seg_memo_gen = 0;
    auto nominal_segment =
        [&](const topology::SwitchStateMask& m) -> const pwl::PwlSegment& {
        if (seg_memo != nullptr && seg_memo_gen == cache.generation() &&
            seg_memo_mask == m) {
            cache.touch_segment(*seg_memo);
            return *seg_memo;
        }
        ++result.segment_searches;
        const auto& seg = cache.lookup(m);
        seg_memo = &seg;
        seg_memo_mask = m;
        seg_memo_gen = cache.generation();
        return seg;
    };

    // v2.0 Phase 2: one accumulator for the step's right-hand side,
    // used by the nominal step, by both halves of the sub-step
    // commutation correction, and by the dt-retry path. It was
    // duplicated three times before; the order of the terms is what
    // makes results bit-identical (trap history, then the user's
    // b_extra_fn, then only the built-in source kinds this circuit
    // actually contains), so having one copy of it is also the only
    // way to keep that promise honest.
    // add-discontinuity-restart-rule. With the rule a step that ENDS
    // on a source jump reads the source's left limit (the jump applies
    // from the next step, which restarts): value_at(t - left_by), far
    // above the pwm_phase snap and far below a plateau. 0 = off.
    const bool restart_rule = opts.discontinuity_restart;
    const Real left_by = restart_rule ? opts.dt * Real{1e-3} : Real{0};
    // b_extra_fn's declared jumps (restart rule only), memoised: the
    // answer for `bj_from` stays valid until t passes `bj_next`.
    const bool has_b_jumps = restart_rule && static_cast<bool>(b_extra_next_jump);
    Real bj_from = std::numeric_limits<Real>::infinity();
    Real bj_next = -std::numeric_limits<Real>::infinity();
    auto b_jump_after = [&](Real t_q) {
        if (!(bj_from <= t_q && t_q < bj_next)) {
            bj_from = t_q;
            bj_next = b_extra_next_jump(t_q);
        }
        return bj_next;
    };
    // The time b_extra_fn is read at for a solve at `at_t`: its left
    // limit when a declared jump falls in (at_t - left_by, at_t].
    auto b_read_t = [&](Real at_t) {
        return has_b_jumps && b_jump_after(at_t - left_by) <= at_t
            ? at_t - left_by : at_t;
    };
    // add-fixed-step-event-splitting (validated above to imply the rule).
    const bool split = opts.enable_substep_state_correction;
    // Events this close to a step boundary are taken at it (design D4).
    const Real snap = opts.dt * Real{0.01};
    // A source jump snapped to the step end (t - snap <= tau < t): the
    // solve AT `src_end_at` reads the built-in sources at `src_end_read`
    // (= tau, so value_at(tau - left_by) is the pre-jump level). NaN
    // compares unequal to every time: off.
    Real src_end_at = std::numeric_limits<Real>::quiet_NaN();
    Real src_end_read = Real{0};
    // `backward_euler`: a restart stage (no stored derivative), the
    // BE-like solve of size for_dt/2 on the factor of for_dt.
    auto accumulate_b_extra = [&](Real at_t, Real for_dt,
                                  bool backward_euler = false) {
        const Real src_t = at_t == src_end_at ? src_end_read : at_t;
        if (backward_euler) {
            history.compute_b_extra_backward_euler(for_dt * Real{0.5},
                                                   b_extra);
        } else {
            history.compute_b_extra(for_dt, b_extra);
        }
        if (b_extra_fn) {
            b_extra += b_extra_fn(b_read_t(at_t));
        }
        add_source_rows(b_extra, src_t, at_t, left_by);
    };

    // v2.0 Phase 2: a run that dies part-way keeps what it computed.
    // The step loops below can throw from a dozen places; whichever
    // one it is, the samples already recorded are worth more than the
    // stack they would otherwise unwind with. `t_reached` tracks the
    // step being attempted so the exception can say where it stopped.
    Real t_reached = opts.t_start;
    try {

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
        if (initial_mask_known) {
            result.record_switch_state(initial_mask);
            stamped_mask = initial_mask;
            stamped_known = true;
        } else {
            // Unsolved = a seeded state whose stamp this run does not
            // know: `initial_state` (it carries no bits; the state may
            // have been solved with a diode ON that this row records
            // OFF), or a snapshot without `stamped_mask`. A zero IC
            // is consistent with every mask.
            const bool seeded =
                initial_state != nullptr || resume_snap != nullptr;
            stamped_mask = has_diodes
                    ? combine_masks(switch_fn(opts.t_start),
                                    diodes.current_diode_mask(),
                                    diode_owned)
                    : switch_fn(opts.t_start);
            stamped_known = !seeded;
            result.record_switch_state(stamped_mask,
                                       /*solved=*/!seeded);
        }

        // Prime the observer with the initial condition so a
        // discrete-time PI / sampler / lookup-table sees the
        // true starting state at t_start.
        if (step_observer) {
            step_observer(opts.t_start, x);
        }

        // ---- discontinuity restart (add-discontinuity-restart-rule) --
        // The next step restarts: the first step of a run that does
        // not start consistent (zero state, builder ICs,
        // initial_state=); a resume carries its own flag; a DC
        // operating point is consistent (i_C = 0, v_L = 0 exactly), and
        // its first step restarts only on a mask change or a jump.
        // Steps of the SDIRK restart still owed: a discontinuity's
        // step and the next two (kRestartSteps). A run that does not
        // start consistent owes them from its first step; a resume
        // carries its own count.
        constexpr Size kRestartSteps = 3;
        constexpr Real kGamma = Real{1} - Real{0.70710678118654752440};
        Size restart_left =
            !restart_rule ? Size{0}
            : resume_snap != nullptr ? resume_snap->restart_steps_left
            : start_from_dc_op ? Size{0} : kRestartSteps;
        const bool has_jump_sources = restart_rule && (has_pwm || has_pulse);
        // Both SDIRK stages solve on the trapezoidal factor of 2γ·dt: a
        // second FIXED step, so a second lazy cache keeps one factor per
        // mask under the same byte budget. (On the 8-entry event LRU a
        // VSI refactored on every restart step: 3172 builds in 3157.)
        // const_cast: the cache only reads the pool unless a parameter
        // update is applied, and this one never receives any.
        std::optional<pwl::PwlStateSpaceCache> restart_cache;
        const Real restart_dt = Real{2} * (kGamma * opts.dt);
        if (restart_rule) {
            restart_cache.emplace(graph, const_cast<pwl::DevicePool&>(pool));
            restart_cache->set_segment_budget_bytes(cache.segment_budget_bytes());
            restart_cache->build_lazy(restart_dt);
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
            t_reached = t;
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
            // read the new duty / mask. Under the restart rule the
            // gate is sampled at the step START (design D4); the
            // observer keeps its (t_k, x_{k-1}) label, whose move to
            // t_{k-1} belongs to fix-control-tick-semantics.
            Real gate_t = restart_rule ? t_prev : t;
            if (step_observer) {
                step_observer(t, x_prev);
            }
            gate_memo_n = 0;
            // ---- add-fixed-step-event-splitting: the step's events ----
            // Gate edges from the schedule's oracle and built-in source
            // jumps inside [t_prev + snap, t - snap), in time order. One
            // before that is taken at t_prev (the gate is sampled at it;
            // a source jump triggers the restart below), one after it at
            // t (the next step's; a source jump is read at its left
            // limit by this step's last solve).
            Size n_ev = 0;
            bool ev_overflow = false;
            bool capture_first = false;   // keep the first solve (diode split)
            if (split) {
                src_end_at = std::numeric_limits<Real>::quiet_NaN();
                auto push_ev = [&](Real e) {
                    if (n_ev < kMaxStepEvents) {
                        ev_t[n_ev++] = e;
                    } else {
                        ev_overflow = true;
                    }
                };
                if (next_edge_fn) {
                    // The oracle's answer stays valid until the step
                    // passes it (an SPWM oracle bisects on every call).
                    if (!(edge_from <= t_prev && t_prev < edge_next)) {
                        edge_from = t_prev;
                        edge_next = next_edge_fn(t_prev);
                    }
                    Real e = edge_next;
                    if (e < t) {
                        topology::SwitchStateMask gate_mask = gate_at(t_prev);
                        for (Size q = 0; q < 4 * kMaxStepEvents && e < t; ++q) {
                            if (e < t - snap) {
                                // A no-op edge (same mask both sides) is
                                // not an event.
                                const auto& m = gate_at(e);
                                if (!(m == gate_mask)) {
                                    if (e < t_prev + snap) {
                                        gate_t = e;
                                    } else {
                                        push_ev(e);
                                    }
                                    gate_mask = m;
                                }
                            }
                            edge_from = e;
                            e = next_edge_fn(e);
                            edge_next = e;
                        }
                    }
                }
                if (has_jump_sources) {
                    Real e = sources::next_source_jump_after(pool, graph, t_prev);
                    for (Size q = 0; q < 4 * kMaxStepEvents && e < t; ++q) {
                        if (e >= t - snap) {
                            src_end_at = t;
                            src_end_read = e;
                        } else if (e >= t_prev + snap) {
                            push_ev(e);
                        }
                        e = sources::next_source_jump_after(pool, graph, e);
                    }
                }
                std::sort(ev_t.begin(), ev_t.begin() + n_ev);
                // Events closer than `snap` merge into the earlier one.
                Size m = 0;
                for (Size q = 0; q < n_ev; ++q) {
                    if (m == 0 || ev_t[q] - ev_t[m - 1] >= snap) {
                        ev_t[m++] = ev_t[q];
                    }
                }
                n_ev = m;
            }
            const Real gate_t0 = gate_t;
            // V3 snapshots — taken regardless of whether
            // correction fires (cheap; vectors are small per
            // dynamic-branch). Restored only if a commutation
            // event is detected AND
            // enable_substep_state_correction is true.
            if (may_roll_back) {
                ++result.rollback_snapshots;
                history.snapshot_into(history_snap);
                coss_snap = coss_history.snapshot();
                laur_snap = laur_history.snapshot();
                pmsm_snap = pmsm_history.snapshot();
                tail_snap = tail_history.snapshot();
                sat_snap = sat_history.snapshot();
                if (has_diodes) {
                    diodes.snapshot_on_bits_into(diodes_snap);
                } else {
                    diodes_snap.clear();
                }
            }
            // Pre-event mask: what mask we'd use BEFORE any
            // diode flip during this step. Captured before
            // event iteration runs.
            topology::SwitchStateMask mask_pre = gate_at(gate_t);
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
            // allocating + adding a zero vector every step. (Below, once
            // the step is known not to restart: a restart step evaluates
            // its sources at its own stage times only.)

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
            n_masks_seen = 0;

            // One step of `for_dt` ending at `at_t`: solve, let the
            // PWL diodes re-decide from the result, re-solve until
            // they agree. Factored out so the dt-retry path below can
            // run exactly the same iteration at a smaller step
            // (v2.0 Phase 2, B.4) — the nominal `for_dt == opts.dt`
            // call still takes the cached factorization and is
            // bit-identical to before.
            // `single`: one solve under the current bits, no re-decision
            // (sub-step 1 of a diode split, whose bits are the pre-event
            // ones by construction).
            auto run_event_iteration = [&](Real at_t, Real for_dt,
                                           bool backward_euler = false,
                                           bool single = false) {
            // Phase 4 C.1 — EVERY solve tells the nonlinear
            // refresh which step it is taking. This is the single
            // point all of them pass through, including the
            // dt-retry sub-steps and the commutation split, so a
            // Coss always stamps 2C(v)/h with the h actually being
            // used. (The saturable inductor reads the same
            // variable but was written when it never moved; it
            // still refuses a variable step for the OTHER reason,
            // no rollback of its flux.)
            // A restart stage advances for_dt/2 (= γh) on the factor of
            // for_dt (= 2γh); the devices are told the interval and the rule.
            refresh_dt = backward_euler ? for_dt * Real{0.5} : for_dt;
            refresh_stage = backward_euler
                ? pwl::TrBdf2Stage::BackwardEuler
                : pwl::TrBdf2Stage::Trapezoidal;
            iters = 0;
            flipped = false;
            // NOT reset: a mask cycle in ANY sub-step of a retry is a
            // breach of the whole outer step. Clearing it here would
            // let every sub-step but the last vanish from
            // `event_iteration_breaches` — and would let
            // `strict_event_iterations` be violated in silence, which
            // is the opposite of what that flag is for.
            n_masks_seen = 0;
            // Diode on-bits consistent with the most recent solve
            // (captured BEFORE update_from_state advances them) —
            // restored on breach so x and the diode state agree
            // (adversarial-review finding P0-R1).
            last_solved_bits.clear();
            do {
                auto mask = gate_at(restart_rule ? gate_t : at_t);
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
                        const auto& seg = nominal_segment(mask);
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
                    if (restart_cache && for_dt == restart_dt) {
                        restart_cache->solve(mask, b_extra, x);
                    } else if (for_dt == opts.dt && for_dt == cache.dt()) {
                        // == solve_at's delegation to solve(mask, …)
                        cache.solve_segment(nominal_segment(mask), b_extra, x);
                    } else {
                        cache.solve_at(mask, for_dt, b_extra, x);
                    }
                }
                // The mask x was just solved under — what the
                // stored sample records (a mask cycle breaks out
                // BEFORE re-solving, so this stays the last solve).
                stamped_mask = mask;
                stamped_known = true;
                if (capture_first && iters == 0) {
                    x_first = x;
                    mask_first = mask;
                }
                if (has_diodes) {
                    diodes.snapshot_on_bits_into(last_solved_bits);
                }
                flipped = !single && has_diodes &&
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
            mask_cycle = false;    // per OUTER step, not per sub-step

            // ---- discontinuity restart (add-discontinuity-restart-rule)
            // Triggers: an unknown previous stamp, a mask that differs
            // from the last committed one, a source jump in [t_prev, t),
            // a declared b_extra_fn jump. (The first step and diode
            // flips arm the window where they happen.)
            // Each trigger (re-)arms the window; a restart step consumes
            // one of it.
            // With the split a built-in source jump inside the step is an
            // event of its own; only one taken at t_prev (within snap
            // either side) restarts the step.
            if (restart_rule &&
                (!stamped_known || !(mask_pre == stamped_mask) ||
                 (has_jump_sources &&
                  (split ? sources::source_jump_in(pool, graph, t_prev - snap,
                                                   t_prev + snap)
                         : sources::source_jump_in(pool, graph, t_prev, t))) ||
                 (has_b_jumps && b_jump_after(t_prev - left_by) < t - left_by))) {
                restart_left = kRestartSteps;
            }
            bool restart = restart_left > 0;
            if (restart) --restart_left;
            // The final solve of this step was a restart stage, so the
            // deferred commit below is a backward-Euler one (at γh).
            bool committed_be = false;
            auto commit_backward_euler = [&](Real h_e) {
                history.update_from_state_backward_euler(x, h_e);
                coss_history.update_from_state(x, h_e, true);
                laur_history.update_from_state(
                    x, h_e, pwl::TrBdf2Stage::BackwardEuler);
                pmsm_history.update_from_state(
                    x, h_e, pwl::TrBdf2Stage::BackwardEuler);
                tail_history.update_from_state(
                    x, h_e, pwl::TrBdf2Stage::BackwardEuler);
                if (has_saturable) {
                    sat_history.update_from_state(
                        x, pwl::TrBdf2Stage::BackwardEuler);
                }
            };
            // One Alexander SDIRK(2,2) step of h from t_from (γ = 1 −
            // 1/√2, L-stable, order 2). Both stages are backward-Euler-
            // like solves of size γh — conductance C/(γh), i.e. the
            // trapezoidal factor at 2γh — the second from the pseudo
            // previous state x0* = x0 + ((1 − γ)/γ)(x1 − x0). EACH stage
            // has its own diode event iteration (not atomic: a gate that
            // opens an inductor's path is damped in stage 1 before any
            // diode is asked to conduct). Stage times t_from + γh and
            // t_from + h, the sources read once at each. The commit after
            // stage 2 is the caller's (deferred past the freeze guard):
            // commit_backward_euler(γh) from x0* refreshes the derivative
            // slots with stage 2's own (x2 − x0*)/(γh).
            auto run_restart = [&](Real t_from, Real h) {
                const Real h_s = kGamma * h;
                const Real dt_eq = Real{2} * h_s;
                accumulate_b_extra(t_from + h_s, dt_eq, true);
                run_event_iteration(t_from + h_s, dt_eq, true);
                const Size iters_1 = iters;
                const bool breach_1 = flipped;
                if (breach_1 && has_diodes && !last_solved_bits.empty()) {
                    diodes.restore_on_bits(last_solved_bits);
                }
                constexpr Real c = (Real{1} - kGamma) / kGamma;
                history.extrapolate_previous(x, c);
                coss_history.extrapolate_previous(x, c);
                laur_history.extrapolate_previous(x, c);
                pmsm_history.extrapolate_previous(x, c);
                tail_history.extrapolate_previous(x, c);
                if (has_saturable) sat_history.extrapolate_previous(x, c);
                accumulate_b_extra(t_from + h, dt_eq, true);
                run_event_iteration(t_from + h, dt_eq, true);
                // A diode flip inside the step is a discontinuity of its
                // own: the window covers this step and the next two.
                if (iters_1 > 1 || iters > 1) {
                    restart_left = std::max(restart_left, kRestartSteps - 1);
                }
                iters += iters_1 - 1;
                flipped = flipped || breach_1;
            };
            // ---- add-fixed-step-event-splitting (design D1, D3, D4) ----
            // The step as sub-intervals between the located events: the
            // first one trapezoidal (or an SDIRK step when the step
            // restarts), every later one an SDIRK step from its event. A
            // diode crossing inside the first, trapezoidal interval is
            // located from its first solve (pre-event bits) and split at
            // too. Intermediate commits are inline; the last is the
            // caller's (deferred past the freeze guard). Any split arms
            // the window for the next two grid steps explicitly: their
            // masks equal the post-event one, so no trigger would.
            auto commit_trap = [&](Real h) {
                history.update_from_state(x, h);
                coss_history.update_from_state(x, h);
                laur_history.update_from_state(x, h);
                pmsm_history.update_from_state(x, h);
                tail_history.update_from_state(x, h);
                if (has_saturable) {
                    sat_history.update_from_state(x);
                }
            };
            auto split_step = [&]() {
                split_events.clear();
                bool any_split = false;
                bool breach = false;
                Real a = t_prev;
                for (Size i = 0; i <= n_ev; ++i) {
                    const bool last = i == n_ev;
                    const Real b = last ? t : ev_t[i];
                    // The whole step keeps opts.dt exactly (t - t_prev
                    // is not, in floating point): the nominal factor.
                    Real h = (i == 0 && last) ? opts.dt : b - a;
                    bool be = i == 0 ? restart : true;
                    if (i > 0) {
                        gate_t = a;   // the edge's post side (grid-exact schedules)
                        any_split = true;
                    }
                    if (be) {
                        run_restart(a, h);
                    } else {
                        if (i > 0 || h != opts.dt) {
                            accumulate_b_extra(b, h);
                        }   // else: accumulated at (t, dt) above
                        capture_first = has_diodes;
                        run_event_iteration(b, h);
                        capture_first = false;
                        if (iters > 1) {
                            // D1: the crossing from x_prev and the first
                            // solve, both under the step's start bits
                            // (only interval 0 is trapezoidal: a == t_prev).
                            Real t_d = b;
                            const auto ents = diodes.entries();
                            cross_scratch.assign(ents.size(), b + h);
                            for (Size j = 0; j < ents.size(); ++j) {
                                const auto& e = ents[j];
                                const bool on = diodes_snap[j];
                                const Real v1 =
                                    stamping::read_node_voltage(x_first, e.from) -
                                    stamping::read_node_voltage(x_first, e.to);
                                if (models::SwitchedDiode::decide(
                                        on, v1, e.params()) == on) {
                                    continue;
                                }
                                const Real v0 =
                                    stamping::read_node_voltage(x_prev, e.from) -
                                    stamping::read_node_voltage(x_prev, e.to);
                                // Already on the far side at the start:
                                // the crossing is at a (the rule alone).
                                cross_scratch[j] =
                                    models::SwitchedDiode::decide(
                                        on, v0, e.params()) != on
                                    ? a
                                    : interp_commutation_time(
                                          a, b,
                                          models::SwitchedDiode::crossing_signal(
                                              on, v0, e.params()),
                                          models::SwitchedDiode::crossing_signal(
                                              on, v1, e.params()));
                                t_d = std::min(t_d, cross_scratch[j]);
                            }
                            // Back to the step start, then: the crossing
                            // at a (an SDIRK step, as the rule alone), at
                            // b (keep the pre-event solve, flip there) or
                            // inside (split).
                            x = x_prev;
                            history.restore(history_snap);
                            coss_history.restore(coss_snap);
                            laur_history.restore(laur_snap);
                            pmsm_history.restore(pmsm_snap);
                            tail_history.restore(tail_snap);
                            sat_history.restore(sat_snap);
                            diodes.restore_on_bits(diodes_snap);
                            mask_cycle = false;
                            // Post-event bits: the diodes that cross
                            // within snap of the first one flip.
                            post_event_bits = diodes_snap;
                            for (Size j = 0; j < ents.size(); ++j) {
                                if (cross_scratch[j] <= b &&
                                    cross_scratch[j] - t_d < snap) {
                                    post_event_bits[j] = !diodes_snap[j];
                                    split_events.push_back(CommutationEvent{
                                        .t_estimated = cross_scratch[j],
                                        .branch_id = ents[j].branch_id,
                                        .new_state = post_event_bits[j]});
                                }
                            }
                            if (t_d - a < snap) {
                                split_events.clear();
                                restart = true;
                                be = true;
                                restart_left = std::max(restart_left,
                                                        kRestartSteps - 1);
                                run_restart(a, h);
                            } else if (b - t_d < snap) {
                                x = x_first;
                                stamped_mask = mask_first;
                                stamped_known = true;
                                diodes.restore_on_bits(post_event_bits);
                                iters = 1;
                                flipped = false;
                                // The flip is at b: the window starts at
                                // the next interval (this step's, or the
                                // next grid step when b = t).
                                restart_left = std::max(
                                    restart_left,
                                    last ? kRestartSteps : kRestartSteps - 1);
                            } else {
                                const Real h1 = t_d - a;
                                accumulate_b_extra(t_d, h1);
                                run_event_iteration(t_d, h1, false, true);
                                commit_trap(h1);
                                diodes.restore_on_bits(post_event_bits);
                                h = b - t_d;
                                run_restart(t_d, h);
                                be = true;
                                any_split = true;
                            }
                        }
                    }
                    breach = breach || flipped;
                    if (!last) {
                        if (be) {
                            commit_backward_euler(kGamma * h);
                        } else {
                            commit_trap(h);
                        }
                    } else {
                        committed_dt = h;
                        committed_be = be;
                    }
                    a = b;
                }
                if (any_split) {
                    ++result.n_split_steps;
                    // The event's step and the next two (the rule's window).
                    restart_left = std::max(restart_left, kRestartSteps - 1);
                }
                flipped = breach || ev_overflow;
            };
            // (A split step accumulates per sub-interval: no discarded
            // b_extra_fn call at (t, dt).)
            if (!restart && n_ev == 0) {
                accumulate_b_extra(t, opts.dt);
            }
            try {
                if (split) {
                    split_step();
                } else {
                if (!restart) {
                    run_event_iteration(t, opts.dt);
                    if (restart_rule && iters > 1) {
                        // A diode flipped inside a trapezoidal step:
                        // that is a discontinuity too. Back to the step
                        // start, and re-take it as a restart step.
                        x = x_prev;
                        history.restore(history_snap);
                        coss_history.restore(coss_snap);
                        laur_history.restore(laur_snap);
                        pmsm_history.restore(pmsm_snap);
                        tail_history.restore(tail_snap);
                        sat_history.restore(sat_snap);
                        if (has_diodes) {
                            diodes.restore_on_bits(diodes_snap);
                        }
                        mask_cycle = false;
                        restart = true;
                        restart_left = std::max(restart_left,
                                                kRestartSteps - 1);
                    }
                }
                if (restart) {
                    run_restart(t_prev, opts.dt);
                    committed_be = true;
                }
                }   // !split
            } catch (const std::runtime_error& nominal_failed) {
                // NOTE the narrow catch. Everything the solver throws
                // is a runtime_error; a foreign exception is not.
                // `pybind11::error_already_set` derives from
                // std::exception but NOT from runtime_error, and its
                // constructor calls PyErr_Fetch, which CLEARS the
                // Python error indicator — so catching it here would
                // treat a KeyboardInterrupt raised inside the user's
                // switch_fn as a step-size problem, re-take the step
                // (this time with no pending signal, so it
                // succeeds), record a DtRetry that never happened,
                // and run to completion with Ctrl-C swallowed. A
                // deterministic callback error would instead be
                // re-invoked up to 126 times before surfacing
                // stripped of its type. Neither is a numerical
                // failure and neither is ours to retry.
                if (!dt_retry_can_help) {
                    throw;
                }
                std::string why = nominal_failed.what();
                // `Size{1} << h` is undefined for h >= 64 and
                // pointless long before that: 2^25 sub-steps of one
                // outer step is not a recovery, it is a hang. The
                // option is read/write from Python and unchecked by
                // SimulationOptions::valid(), so clamp at the point
                // of use.
                constexpr Size kMaxHalvings = Size{20};
                const Size halving_cap =
                    std::min(opts.max_dt_halvings, kMaxHalvings);
                for (Size h = 1; h <= halving_cap; ++h) {
                    // Back to the state at t_prev. Note what is NOT
                    // restored: `sat_history`, which this step has
                    // not touched yet (it commits at the very end),
                    // and the cache, whose contents are a function of
                    // the circuit rather than of the run.
                    x = x_prev;
                    history.restore(history_snap);
                    coss_history.restore(coss_snap);
                    laur_history.restore(laur_snap);
                    pmsm_history.restore(pmsm_snap);
                    tail_history.restore(tail_snap);
                    sat_history.restore(sat_snap);
                    if (has_diodes) {
                        diodes.restore_on_bits(diodes_snap);
                    }
                    // shortcut: the retry ignores events inside the step
                    // (the gate keeps the step's start sample, as the
                    // restart rule alone; a built-in source jump inside it
                    // is read where the sub-steps fall and not restarted);
                    // retries are rare.
                    gate_t = gate_t0;
                    capture_first = false;
                    split_events.clear();

                    const Size n_sub = Size{1} << h;
                    const Real sub_dt =
                        opts.dt / static_cast<Real>(n_sub);
                    bool all_ok = true;
                    for (Size j = 0; j < n_sub; ++j) {
                        const Real t_sub =
                            t_prev + static_cast<Real>(j + 1) * sub_dt;
                        // Every sub-step of a restart step is an SDIRK
                        // step of sub_dt (the factor of 2γ·sub_dt).
                        const bool sub_restart = restart;
                        if (!sub_restart) {
                            accumulate_b_extra(t_sub, sub_dt);
                        }
                        try {
                            if (sub_restart) {
                                run_restart(t_sub - sub_dt, sub_dt);
                            } else {
                                run_event_iteration(t_sub, sub_dt);
                            }
                        } catch (const std::runtime_error& sub_failed) {
                            why = sub_failed.what();
                            all_ok = false;
                            break;
                        }
                        // shortcut: a flip inside a trapezoidal sub-step is
                        // not re-taken; the window restarts from the next
                        // outer step (retries are rare).
                        if (restart_rule && !sub_restart && iters > 1) {
                            restart_left = kRestartSteps;
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
                        // `sat_history` advances with the rest.
                        // Each sub-step's flux must be committed
                        // before the next one reads it, exactly like
                        // the trap history above — a saturable
                        // inductor integrated across n sub-steps from
                        // ONE stale flux would be integrating the
                        // whole outer step at sub_dt. A deeper
                        // halving rolls all of them back together
                        // from `sat_snap`.
                        if (j + 1 < n_sub && sub_restart) {
                            commit_backward_euler(kGamma * sub_dt);
                        } else if (j + 1 < n_sub) {
                            history.update_from_state(x, sub_dt);
                            coss_history.update_from_state(x,
                                                            sub_dt);
                            laur_history.update_from_state(x,
                                                            sub_dt);
                            pmsm_history.update_from_state(x,
                                                            sub_dt);
                            tail_history.update_from_state(x,
                                                            sub_dt);
                            sat_history.update_from_state(x);
                        } else {
                            committed_dt = sub_dt;
                            committed_be = sub_restart;
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
                    // A Newton failure that survived every halving on
                    // this engine is, in the one case characterised
                    // (CHANGELOG 2.0.0, test_newton_stall_and_mask_
                    // chatter.py), PWL diode-mask chatter inside the
                    // Newton loop, and it gets WORSE with a smaller
                    // dt: the PSFB reproducer aborts at 10, 5 and
                    // 2 ns and runs at 50 ns. Say so, rather than let
                    // the user reach for a smaller dt.
                    const bool newton_failed =
                        why.find("solve_with_newton")
                        != std::string::npos;
                    throw std::runtime_error(std::format(
                        "run_transient: the step ending at t = {} "
                        "could not be taken, even split into {} "
                        "sub-steps of {:.3e} s (dt/{}). Last failure: "
                        "{}\nThis is no longer a step-size problem — "
                        "raise max_dt_halvings only if you believe "
                        "otherwise; more likely the circuit needs a "
                        "snubber, a softer device model, or a look at "
                        "the device the message above names.{}",
                        t, Size{1} << halving_cap,
                        opts.dt / static_cast<Real>(
                            Size{1} << halving_cap),
                        Size{1} << halving_cap, why,
                        newton_failed
                            ? "\nIf PWL diodes (add_diode) share the "
                              "circuit with a Newton device (saturable "
                              "or hysteretic magnetics, a smooth "
                              "junction), this is likely the known "
                              "diode-mask chatter inside Newton "
                              "(CHANGELOG 2.0.0; KNOWN_LIMITATIONS.md), "
                              "and a smaller dt makes it worse, not "
                              "better. Measured workarounds: a coarser "
                              "dt (the PSFB reproducer aborts at 10 ns "
                              "and runs at 50 ns), or smooth Shockley "
                              "junctions (add_shockley_diode) for "
                              "those diodes."
                            : ""));
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

            // 4. Commutation records (Layer 5 V2.2). With the event
            //    split the diode crossings it located come first, at the
            //    located time; every other diode whose watched signal
            //    changed sign over the step gets the interpolation
            //    between x_prev and x. (The V3 sub-step correction that
            //    lived here solved Newton devices linearly and read the
            //    crossing from the post-flip state; the split replaced
            //    it: design D5.)
            step_events.clear();
            if (split) {
                step_events.assign(split_events.begin(), split_events.end());
            }
            if (has_diodes) {
                for (const auto& e : diodes.entries()) {
                    if (split && std::any_of(
                            split_events.begin(), split_events.end(),
                            [&](const CommutationEvent& ev) {
                                return ev.branch_id == e.branch_id;
                            })) {
                        continue;
                    }
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

                    const Real v_on =
                        models::SwitchedDiode::turn_on_voltage(e.params());
                    const Real s_prev = v_d_prev - v_on;
                    const Real s_curr = v_d_curr - v_on;
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
                    const Real i_solved = i_new;
                    bool froze = false;
                    bool clamped = false;
                    // Step-to-step jump guard: snap back to the pre-step
                    // current when the solver emits an unphysical kilo-amp
                    // delta (rectifier in DCM, etc.).
                    if (opts.inductor_freeze_di_max > Real{0}) {
                        const Real di  = i_new - i_prev_step;
                        const Real adi = di < Real{0} ? -di : di;
                        if (adi > opts.inductor_freeze_di_max) {
                            i_new = i_prev_step;
                            froze = true;
                        }
                    }
                    // Absolute clamp: catches the *slow drift* form
                    // of the same failure where the per-step delta
                    // stays below the freeze threshold but i_L walks
                    // monotonically past physical bounds over many
                    // line cycles.
                    if (opts.inductor_abs_clamp > Real{0}) {
                        const Real lim = opts.inductor_abs_clamp;
                        if (i_new >  lim) { i_new =  lim; clamped = true; }
                        if (i_new < -lim) { i_new = -lim; clamped = true; }
                    }
                    // v2.0 Phase 2: a guard that fires REPLACES the
                    // solver's answer with a number the user
                    // configured. Say so. Silently reporting the
                    // limit as a current is how a 1 kW drive comes to
                    // show a line current peaking at exactly 100.000
                    // A — the clamp value — and nobody notices.
                    if (froze || clamped) {
                        InductorGuardAction* rec = nullptr;
                        for (auto& g : result.inductor_guard_actions) {
                            if (g.branch_id == e.branch_id) {
                                rec = &g;
                                break;
                            }
                        }
                        if (rec == nullptr) {
                            result.inductor_guard_actions.push_back(
                                InductorGuardAction{
                                    .branch_id = e.branch_id,
                                    .t_first = t});
                            rec = &result.inductor_guard_actions
                                       .back();
                        }
                        rec->freeze_count += froze ? Size{1} : Size{0};
                        rec->clamp_count  += clamped ? Size{1} : Size{0};
                        const Real a_solved =
                            i_solved < Real{0} ? -i_solved : i_solved;
                        rec->worst_solved =
                            std::max(rec->worst_solved, a_solved);
                        // The largest magnitude the user actually
                        // SEES for this device, not whatever the last
                        // firing happened to leave. Last-write was
                        // meaningless for the freeze guard, whose
                        // substituted value is the previous step's
                        // current and can be arbitrarily small.
                        const Real a_new =
                            i_new < Real{0} ? -i_new : i_new;
                        rec->reported_limit =
                            std::max(rec->reported_limit, a_new);
                    }
                    x[e.inductor_branch_var_id] = i_new;
                }
            }

            // 5. Commit history for the next step. Deferred until AFTER the
            //    freeze guard so a snapped-back i_L is the value baked into
            //    the trap history. `committed_dt` is opts.dt on the normal
            //    path and dt2 when sub-step correction split the step (whose
            //    sub-step 1 already committed the dt1 half above).
            if (committed_be) {
                commit_backward_euler(kGamma * committed_dt);
            } else {
            history.update_from_state(x, committed_dt);
            coss_history.update_from_state(x, committed_dt);
            laur_history.update_from_state(x, committed_dt);
            pmsm_history.update_from_state(x, committed_dt);
            tail_history.update_from_state(x, committed_dt);
            if (has_saturable) {
                sat_history.update_from_state(x);
            }
            }
            if (restart) {
                ++result.n_restart_steps;
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
                result.record_switch_state(stamped_mask);
            }
        }
        result.final_snapshot.restart_steps_left = restart_left;
        if (restart_cache) {
            result.n_restart_factorizations =
                restart_cache->num_built_segments() +
                restart_cache->metrics().segment_evictions;
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
            t_reached = t;
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
            gate_memo_n = 0;

            // Accumulate into the reused buffer (audit #6/#16): optional user
            // fn, then present built-in source kinds. No trap history on the
            // static path. Same summation order → bit-identical; absent kinds
            // and a null user fn cost nothing. `b_extra_user` aliases the
            // buffer so the solves below are unchanged.
            // discontinuity_restart: the same sampling as the dynamic
            // path — sample k reads the gate at t_{k-1} and the jumping
            // sources at their left limit — so adding a capacitor does
            // not shift the recorded mask by one sample. (k = 0: t.)
            const Real gate_t = restart_rule && k > 0 ? t_prev : t;
            const Real src_left = k > 0 ? left_by : Real{0};
            b_extra.setZero();
            if (b_extra_fn) {
                b_extra += b_extra_fn(k > 0 ? b_read_t(t) : t);
            }
            add_source_rows(b_extra, t, t, src_left);
            const Vector& b_extra_user = b_extra;

            Size iters = 0;
            bool flipped = false;
            bool mask_cycle = false;
            // Full mask copies (Phase 1: masks can exceed 64 bits; a
            // copy is 2 inline words for <=128 switches — still no
            // heap in the loop).
            n_masks_seen = 0;

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
                auto mask = gate_at(gate_t);
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
                    const auto& seg = nominal_segment(mask);
                    x = pwl::solve_with_newton_b_extra(
                        seg, nl_refresh_effective, graph, pool,
                        /*x_init=*/x, b_extra_user,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search,
                        opts.enable_newton_lm);
                } else {
                    cache.solve_segment(nominal_segment(mask), b_extra_user, x);
                }
                stamped_mask = mask;
                stamped_known = true;
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

                    const Real v_on =
                        models::SwitchedDiode::turn_on_voltage(e.params());
                    const Real s_prev = v_d_prev - v_on;
                    const Real s_curr = v_d_curr - v_on;
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
                result.record_switch_state(stamped_mask);
            }
        }
    }

    }  // try
    catch (const analysis::Cancelled&) {
        // A deliberate stop, not a failure. It already leaves the
        // partial trace in `result` by unwinding no further than the
        // caller, so let it through unchanged.
        throw;
    }
    catch (const SimulationAborted&) {
        throw;                       // already carries its payload
    }
    catch (const std::runtime_error& e) {
        // Narrow on purpose — see the note at the dt-retry above.
        // Everything the solver throws is a runtime_error; a Python
        // callback's exception is not, and re-labelling one as
        // SimulationAborted would destroy its type and traceback for
        // the sake of attaching a partial trace it never asked for.
        throw SimulationAborted(e.what(), std::move(result),
                                 t_reached);
    }

    // v2.0 Phase 4 — the complete state at the end, so a follow-up
    // run can resume EXACTLY rather than re-deriving the companion
    // history from x (which is not the state and does not
    // reproduce the run).
    result.final_snapshot.t = t_reached;
    result.final_snapshot.x = x;
    result.final_snapshot.history = history.to_flat();
    result.final_snapshot.saturable_history = sat_history.to_flat();
    result.final_snapshot.coss_history = coss_history.to_flat();
    result.final_snapshot.lauritzen_history = laur_history.to_flat();
    result.final_snapshot.igbt_tail_history = tail_history.to_flat();
    result.final_snapshot.pmsm_history = pmsm_history.to_flat();
    result.final_snapshot.diode_on =
        has_diodes ? diodes.snapshot_on_bits() : std::vector<bool>{};
    // The mask x was solved under (the record's last row when the
    // last step was stored), for a resume's row 0.
    result.final_snapshot.stamped_mask =
        stamped_known ? mask_bits(stamped_mask) : std::vector<bool>{};
    result.final_snapshot.valid = true;

    return result;
}

}  // namespace pulsim::solver
