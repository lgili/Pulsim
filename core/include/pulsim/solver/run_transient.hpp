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
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
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
#include "pulsim/solver/jump_series.hpp"
#include "pulsim/solver/loss_tracker.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/events/locate.hpp"
#include "pulsim/events/mode_set.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/step/controllers.hpp"
#include "pulsim/solver/step/integrators.hpp"
#include "pulsim/solver/step/policies.hpp"
#include "pulsim/solver/step/stepper.hpp"
#include "pulsim/solver/step/stepper_core.hpp"
#include "pulsim/solver/step/sensitivity.hpp"
#include "pulsim/solver/step/source_accumulator.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/current_waveform_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/source_jumps.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/pwl/branch_structure.hpp"
#include "pulsim/pwl/reinit.hpp"
#include "pulsim/pwl/mode_ports.hpp"
#include "pulsim/pwl/transmission_line_history.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <format>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>

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
// seed_held_machine_capacitors — the DC point holds each machine's speed
// and angle node (add-mna-native-machines D19), where the capacitor is
// NOT at rest: C·dθ/dt = C·ω0, J·dω/dt = net torque. Its companion
// current is the KCL residual the hold replaced: the linear DC rows
// (no hold) plus the machine's mechanical injection from the refresh,
// i_C(leaving n) = −r[n]. A node with other than one capacitor to
// ground keeps i_C = 0.
// -----------------------------------------------------------------------------
inline void seed_held_machine_capacitors(
    const topology::Graph& graph, const pwl::DevicePool& pool,
    const topology::SwitchStateMask& mask, Real t, const Vector& x,
    const pwl::PmsmMnaHistory& pmsm, pwl::HistoryState& history) {
    std::vector<Index> nodes;
    for (const auto& m : pool.pmsm_mna_machines()) {
        nodes.push_back(m.omega_node);
        nodes.push_back(m.theta_node);
    }
    for (const auto& m : pool.induction_machines()) nodes.push_back(m.omega_node);
    if (nodes.empty()) return;
    sparse::Matrix J;
    Vector r;
    pwl::dc_assemble(graph, pool, mask, J, r, t, Real{0}, Real{1},
                     /*hold_mechanics=*/false);
    r += J * x;
    sparse::Matrix J_nl(J.rows(), J.cols());
    Vector f_nl = Vector::Zero(J.rows());
    (void)pwl::refresh_pmsm_mna(x, J_nl, f_nl, pmsm, Real{1});
    r += f_nl;   // only the mechanical rows are read
    for (const Index n : nodes) {
        if (!stamping::node_is_active(n)) continue;
        Index cap = -1;
        int n_caps = 0;
        for (Index br = 0; br < graph.num_branches(); ++br) {
            const auto& e = graph.branch(br);
            if ((e.from != n && e.to != n) || !pool.is_registered(br) ||
                pool.kind_of(br) != pwl::DevicePool::StoredKind::Capacitor) {
                continue;
            }
            ++n_caps;
            if (!stamping::node_is_active(e.from == n ? e.to : e.from)) cap = br;
        }
        if (n_caps != 1 || cap < 0) continue;
        const Real i_leaving = -r[n];
        history.set_capacitor_current(
            cap, graph.branch(cap).from == n ? i_leaving : -i_leaving);
    }
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
// branch_form_event_time — where a branch-form diode's flip over a step that
// starts at no gate edge is booked (F-Z-14): the conductance form's chord of
// s = v_d − V_on (node voltages, valid in both states) when it changes sign, so
// both forms book one instant; NaN (no event) for an OFF → ON flip from a step
// start already at or above V_on — the start's assumed bits, conducting from
// t_prev⁺, where the conductance form books none either; else the step end.
// shortcut: an ideal ON diode (R_on = 0) reads s ≡ V_f − V_on, no chord: its
// natural turn-off stays at the step end (at most one step late).
// -----------------------------------------------------------------------------
[[nodiscard]] constexpr Real branch_form_event_time(
    bool on_prev, Real t_prev, Real t_curr, Real s_prev, Real s_curr) noexcept {
    if (s_prev * s_curr < Real{0}) {
        return interp_commutation_time(t_prev, t_curr, s_prev, s_curr);
    }
    if (!on_prev && s_prev >= Real{0}) return std::numeric_limits<Real>::quiet_NaN();
    return t_curr;
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
            cache.solve(mask, b_extra, x);   // solve-exempt: V0 overload (task 11.1)
        } else {
            cache.solve(mask, zero_b_extra, x);   // solve-exempt: V0 overload (task 11.1)
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
    const SwitchScheduleFn& switch_fn_user,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false,
    const pwl::NonlinearRefreshFn& nl_refresh = {},
    const StepObserverFn& step_observer_in = {},
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
    const std::function<Real(Real)>& next_edge_fn = {},
    /// add-unified-event-queue group 5 — control ticks scheduled by the
    /// kernel's event queue: tick k of source j is delivered at the first
    /// grid point t_n >= t_start + offset_j + k*T_j (to 1e-9*dt), for the
    /// instants < t_end - 1e-9*dt, after `ticks->before` and before
    /// `step_observer` at that step.
    const events::TickPlan* ticks = nullptr,
    /// Accepted-step / commutation listeners (events/step_hooks.hpp).
    events::StepHooks* hooks = nullptr) {

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
    if (!(opts.split_snap >= Real{1e-12} && opts.split_snap <= Real{0.25})) {
        throw std::invalid_argument(
            "run_transient: split_snap must be in [1e-12, 0.25] (a "
            "fraction of dt)");
    }
    if (!switch_fn_user) {
        throw std::invalid_argument(
            "run_transient: switch_fn callback is required");
    }
    // Every mask the caller hands in has the circuit's width (a short
    // one silently dropped bits, e.g. a branch-form diode's).
    const SwitchScheduleFn switch_fn =
        [&switch_fn_user, n_sw = static_cast<Size>(graph.num_switches())](
            Real t_q) {
            auto m = switch_fn_user(t_q);
            topology::require_mask_width(m, n_sw);
            return m;
        };
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

    // ---- add-unified-event-queue group 5: control ticks on the queue --
    // The ticks run at the start of a step, before the caller's observer,
    // wherever the observer runs (the priming call at t_start included).
    // The rule of the scheduler this replaces (_control_plan.py
    // FixedStepTicks): integer k, due when t >= t_k - 1e-9*dt, instants
    // t_k < t_end - 1e-9*dt, one tick per source per step.
    events::EventQueue tick_queue;
    const Real tick_tol = Real{1e-9} * opts.dt;
    const bool has_ticks = ticks != nullptr && !ticks->rates.empty();
    if (has_ticks) {
        if (!ticks->fn) {
            throw std::invalid_argument(
                "run_transient: a TickPlan with rates needs fn");
        }
        for (const auto& r : ticks->rates) {
            tick_queue.add_periodic(events::Kind::ControlTick, r);
        }
        // t0_tol = the take tolerance: a tick a previous run left out at
        // its t_end (within tick_tol of it) is a resumed run's first.
        // A closed_end rate's instant at t_end fires at the last observer
        // call (labelled t_end, as every tick: the state there is x_{n-1}).
        tick_queue.start(opts.t_start, opts.t_end - tick_tol, tick_tol,
                         opts.t_end + tick_tol);
        result.control_tick_counts.assign(ticks->rates.size(), 0);
    }
    const StepObserverFn step_observer = !has_ticks ? StepObserverFn{} :
        StepObserverFn([&](Real t, const Vector& xs) {
            if (ticks->before) ticks->before(t, xs);
            for (const auto& ev : tick_queue.take_due(
                     t, tick_tol,
                     events::kind_bit(events::Kind::ControlTick))) {
                ++result.control_tick_counts[ev.source];
                result.control_tick_jitter_max =
                    std::max(result.control_tick_jitter_max, t - ev.t);
                ticks->fn(ev.source, events::TickInfo{ev.t, ev.ordinal}, t,
                          xs);
            }
            if (step_observer_in) step_observer_in(t, xs);
        });
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
        if (resume_from->post_edge_derivatives) {
            throw std::invalid_argument(std::format(
                "run_transient(resume_from=...): the snapshot was taken "
                "by a variable-step run that ended exactly on a gate edge "
                "(t = {}): its x is the pre-edge solve but its device "
                "derivatives are already the post-edge ones, and a "
                "fixed-step run started from that pair rings and fails at "
                "the next edge. Resume it with engine='trbdf2', or end "
                "the first run off the edge.", resume_from->t));
        }
        x = resume_from->x;   // device state: hs.load below
    }
    // The time grid: t_k = t_origin + (k_off + k)·dt. A resume from a
    // fixed-step snapshot taken with this dt continues ITS grid, so the
    // resumed sample times (and the gate edges they meet) are the
    // continuous run's to the bit; restarting the grid at t_split moved
    // them by ~1e-20 s and an edge then landed a rounding apart.
    Real t_origin = opts.t_start;
    Size k_off = 0;
    if (resume_snap != nullptr && std::isfinite(resume_snap->grid_origin)
        && resume_snap->grid_dt == opts.dt && resume_snap->t == opts.t_start
        // consistency only, to a few ulps: the check and the loop's
        // t_k may round differently (FMA contraction); the loop's
        // expression is the same in both runs, so t_k is exact anyway
        && std::abs(resume_snap->grid_origin
                    + static_cast<Real>(resume_snap->grid_step) * opts.dt
                    - opts.t_start)
               <= Real{8} * std::numeric_limits<Real>::epsilon()
                      * (std::abs(resume_snap->grid_origin)
                         + static_cast<Real>(resume_snap->grid_step)
                               * opts.dt)) {
        t_origin = resume_snap->grid_origin;
        k_off = resume_snap->grid_step;
    }
    // The grid point k of this run: t_origin + (k_off + k)·dt.
    const step::FixedClock fixed_grid{t_origin, opts.dt};

    // The five stateful-device histories (Coss, Lauritzen, PMSM, IGBT
    // tail, saturable inductor), one object: every operation below visits
    // all five (add-unified-device-interface step 1). Seeded from
    // `initial_state` so the first Newton refresh does not see zeros;
    // resume WINS over a builder-IC seed: the snapshot is the state.
    pwl::StatefulHistories hs;
    hs.get<pwl::ExternalDeviceHistory>().set_start_time(opts.t_start);   // ctx->t of init
    hs.init(graph, pool);
    hs.set_path(opts.device_path, graph, pool, "run_transient");
    if (initial_state != nullptr) hs.seed_from_dc_op(x);
    if (resume_snap != nullptr) hs.load(*resume_snap, history, "run_transient");

    // The step every stateful stamp of the next solve integrates over:
    // set at the one point every solve passes through
    // (run_event_iteration), including the dt-retry sub-steps, the
    // commutation split and the SDIRK stages of discontinuity_restart
    // (BackwardEuler over the γ·h each stage advances).
    pwl::StepCtx step_ctx{opts.dt, pwl::TrBdf2Stage::Trapezoidal, opts.t_start};

    // The Newton refresh is built by the kernel (add-unified-device-
    // interface step 2): the static stamps (with no callback from the
    // caller, Newton devices get the standard ones instead of being
    // simulated as open circuits: 10.000 V on a clamp that reads 0.594 V),
    // then every stateful history, all reading `step_ctx`, which
    // run_event_iteration sets before every solve. `nl_refresh` is a
    // deprecated override of the static stamps only.
    const pwl::NonlinearRefreshFn nl_static =
        pwl::static_device_refresh(pool, nl_refresh, opts.device_path);
    const pwl::NonlinearRefreshFn nl_refresh_effective =
        pwl::build_device_refresh(nl_static, hs, step_ctx,
                                  /*sat_increment_form=*/false);

    pwl::DiodeEventState diodes{graph, pool};
    diodes.reset();
    if (resume_snap != nullptr && !resume_snap->mask.empty()) {
        // The solver-owned conduction bits the snapshot promised to
        // carry (size-checked by name inside).
        diodes.restore_on_bits(resume_snap->mask);
    }
    const auto diode_owned = diodes.diode_owned_bits();
    // the diode-owned bits a gate does not drive (H3: a gated device's bit
    // flipping at its gate edge is a gate event, not a diode flip)
    auto diode_owned_ungated = diode_owned;
    for (const auto& d : diodes.entries()) {
        if (d.gate != models::SwitchedDiode::Gate::None) diode_owned_ungated.set(d.switch_idx, false);
    }
    const bool has_diodes = diodes.num_diodes() > 0;
    // the commanded gate per sample (2.2); cleared before the result leaves
    if (diodes.num_gated() > 0) {
        result.gate_mask_fn = [&diodes] { return diodes.gate_mask(); };
    }
    // Ideal (R_on = 0) branch-form diodes sit exactly at V_on while ON, so
    // the voltage-crossing record below never sees them turn off; their
    // commutations are read from the stamped state, step start to step end.
    std::vector<char> commut_on_prev;
    bool has_ideal_diode = false;
    for (const auto& e : diodes.entries())
        has_ideal_diode = has_ideal_diode || pool.is_branch_form(e.branch_id);

    // add-branch-form-ideal-switch D3: the per-mode structural pass, run
    // once per mode before its first solve (inactive, and free, without
    // an ideal branch-form element).
    pwl::BranchStructureChecker structure{graph, pool, "pwl"};
    // Ideal diodes turned OFF this step because they closed a stiff loop.
    std::vector<Index> loop_released;
    std::string loop_last;   // the loops of the last release, in loop order
    std::vector<bool> loop_bits;
    // Returns true when it turned loop-closing ideal diodes OFF: the
    // caller re-forms the mask and checks again.
    auto check_mode = [&](const topology::SwitchStateMask& m, bool dynamic,
                          Real at_t) -> bool {
        const auto& ms = structure.check(m, dynamic);
        result.mode_structure_checks = structure.checks_run();
        if (!ms.loop_diodes.empty()) {
            // A diode across a loop of sources and closed ideal elements
            // conducts an undefined current; OFF is its only consistent
            // state when the loop reverse-biases it (the turn-on edge of
            // a CCM buck: switch ON while the freewheeling diode still
            // conducts). If the diode law turns it back ON within the
            // same step the loop forward-biases it: refuse.
            Index b = ms.loop_diodes.front();
            std::string loop = ms.loop_texts.front();
            bool again = false;
            for (std::size_t m = 0; m < ms.loop_diodes.size(); ++m) {
                const Index d = ms.loop_diodes[m];
                if (std::find(loop_released.begin(), loop_released.end(), d) !=
                    loop_released.end()) {
                    b = d;
                    loop = ms.loop_texts[m];
                    again = true;
                    break;
                }
            }
            if (!dynamic || again) {
                throw UnsupportedError(
                    "pwl", "ideal_voltage_loop",
                    std::string(graph.branch_name(b)), "Diode",
                    std::format(
                        "engine 'pwl': at t = {} the ideal diode '{}' (R_on = "
                        "0) closes a loop of voltage sources and closed ideal "
                        "elements in mode {} (loop: {}){}: its current is undefined. "
                        "Give it R_on > 0.",
                        at_t, graph.branch_name(b), m.to_string(), loop,
                        again ? ", and the loop forward-biases it (turned OFF,"
                                " it turned back ON)" : ""));
            }
            diodes.snapshot_on_bits_into(loop_bits);
            const auto ents = diodes.entries();
            loop_last.clear();
            for (std::size_t m = 0; m < ms.loop_texts.size(); ++m)
                loop_last += (m ? "; " : "") + ms.loop_texts[m];
            for (const Index d : ms.loop_diodes) {
                loop_released.push_back(d);
                for (Size q = 0; q < ents.size(); ++q) {
                    if (ents[q].branch_id == d) loop_bits[q] = false;
                }
            }
            diodes.restore_on_bits(loop_bits);
            ++result.ideal_loop_releases;
            return true;
        }
        return false;
    };
    // D4: the SDIRK x3 restart realises the projection onto a dependent
    // mode; without the rule the trapezoid rings (+-397 V). Decided on the
    // COMMITTED mode: an event-iteration iterate may pass through one
    // (CCM buck: switch off, diode not yet on; review M6).
    // `in_dependent_mode` and the record live in pwl/reinit.hpp (shared
    // with engine='trbdf2').
    auto in_dependent_mode = [&](const pwl::DependentStructure& d,
                                 const Vector& xs) {
        return pwl::in_dependent_mode(graph, pool, d, xs);
    };
    auto refuse_dependent = [&](const pwl::ModeStructure& ms,
                                const topology::SwitchStateMask& m, Real at_t) {
        if (opts.discontinuity_restart) return;
        const auto it = std::find_if(
            ms.dependent.begin(), ms.dependent.end(),
            [&](const pwl::DependentStructure& d) { return in_dependent_mode(d, x); });
        if (it == ms.dependent.end()) return;
        const auto& d = *it;
        const Index el = d.elements.front();
        std::string states;
        for (Index b : d.states) {
            if (!states.empty()) states += ", ";
            states += graph.branch_name(b);
        }
        throw UnsupportedError(
            "pwl", "dependent_states", std::string(graph.branch_name(el)),
            std::string(pwl::stored_kind_name(pool.kind_of(el))),
            std::format(
                "engine 'pwl': at t = {} the mode {} makes {} dependent "
                "(a {} through the ideal element '{}'): its re-initialisation "
                "is the SDIRK x3 restart, so pass discontinuity_restart=True, "
                "or give '{}' {} > 0.",
                at_t, m.to_string(),
                states,
                d.kind == pwl::DependentStructure::Kind::CutSet
                    ? "cut-set of inductors" : "loop of capacitors",
                graph.branch_name(el), graph.branch_name(el),
                d.kind == pwl::DependentStructure::Kind::CutSet ? "g_off"
                                                                : "R_on"));
    };
    // D4 records: the last committed mask and each branch's mask bit.
    pwl::ReinitTracker reinit{graph, pool, structure.active()};

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

        // The DC mode is the one the DC diode iteration converges to; the
        // start mask (diodes OFF) can be a false floating island (review
        // L2). On a failed solve the start mask's check names the cause.
        const auto dc_start_mask = has_diodes
            ? diodes.combine(switch_fn(opts.t_start))
            : switch_fn(opts.t_start);
        // The DC refresh (device DECISIONS #56): the static chain has no
        // DC stamp for a Lauritzen diode (it solved it open); the caller's
        // override keeps winning.
        const pwl::NonlinearRefreshFn& dc_refresh =
            (nl_refresh || pool.lauritzen_diode_branches().empty())
                ? nl_static
                : pwl::dc_device_refresh(pool, true, opts.device_path);
        auto dc = [&] {
            try {
                return pwl::compute_dc_operating_point(
                    graph, pool, switch_fn(opts.t_start), dc_refresh,
                    dc_opts, has_diodes ? &diodes : nullptr,
                    "run_transient(start_from_dc_op)", "pwl");
            } catch (const std::runtime_error&) {
                if (structure.active()) {
                    (void)check_mode(dc_start_mask, false, opts.t_start);
                }
                throw;
            }
        }();
        if (structure.active()) (void)check_mode(dc.mask, false, opts.t_start);
        x = std::move(dc.x);
        // the linear companions, then every stateful kind on its DC curve:
        // PMSM λ, i, θ at the held point (D19), external charges, Lauritzen
        // q_M, IGBT tail Q, nonlinear-capacitor q, saturable flux (empty
        // kinds: no-op)
        step::seed(history, hs, x);
        seed_held_machine_capacitors(graph, pool, dc.mask, opts.t_start, x,
                                     hs.get<pwl::PmsmMnaHistory>(), history);
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
    pwl::NewtonWorkspace newton_ws;   // one Newton analysis per pattern per run
    const sparse::Matrix* sens_J_lin = nullptr;   // the last Newton solve's J_lin
    // add-parameter-sensitivities group 5: the stage recurrence (step/
    // sensitivity.hpp); off (no params) it is one test per step.
    step::StageSensitivity sens{graph, pool, opts.sensitivities, "pwl",
                                opts.tol_newton_dx};
    if (sens.on() && (resume_from != nullptr || start_from_dc_op)) {
        throw UnsupportedError("pwl", "sensitivities", "", "",
            resume_from != nullptr
                ? "sensitivities with resume_from: the sensitivity state is not "
                  "part of a snapshot; run from the start."
                : "sensitivities with start_from_dc_op: the DC point depends on "
                  "the parameters and its derivative is not carried; start "
                  "from a parameter-free state.");
    }
    std::vector<Real> history_snap;                 // rollback: (v_prev, i_prev) pairs
    // Phase 4 C.1 — the Coss rolls back with everything else. The
    // saturable inductor cannot, which is the second reason it
    // refuses a variable step; this device was given the
    // capability up front rather than as a follow-up.
    pwl::StatefulHistories::Snapshot hs_snap;   // V3 rollback point
    std::vector<bool> diodes_snap;                  // V3 diode-bit snapshot
    std::vector<bool> last_solved_bits;             // breach re-sync bits
    // Restart stage-1 state, kept for the stage-2 fallback (see
    // run_restart): x1, its diode bits and the histories at x0.
    Vector rs_x1, rs_x2;
    // F-Z-11: the uncut restart (sub)step, kept while its other flips are located
    Vector rk_x, rk_b;
    std::vector<Real> rk_h;
    pwl::StatefulHistories::Snapshot rk_hs;
    std::vector<bool> rk_bits, rk_lsb;
    Vector edge_xl;   // the edge's mode decision: a linearized solve (decide_edge_mode)
    std::vector<bool> edge_bits;   // the diode bits after its stage
    std::vector<bool> rs_bits, rs_bits2, rs_bits_be, rs_solved2;
    std::vector<bool> rs_bits0;   // the diode bits at stage 1's start
    // The diode bits after stage 1 of a restart that starts at the step
    // start: a branch-form flip already there was forced at t_prev.
    std::vector<bool> rs_bits_first;
    bool rs_first_valid = false;
    std::vector<Real> rs_hist;
    pwl::StatefulHistories::Snapshot rs_hs;
    // Consistent initialisation at a restart (release group A, F-5-1/F-3-3):
    // stage 1 (x1), the histories at x0, the stage's b_extra and the history
    // parts of it; the impulse records owed by the attempt (written in the
    // record phase); and the event's right limit x0+ at event_rl_t (NaN: none
    // this step), for the event-sample storage to read.
    Vector ip_x1, ip_b, ip_h0, ip_h1, ip_x2, ip_x3f, ip_x2m, ip_s2, ip_s3;
    std::vector<Real> ip_hist, ip_d, ip_r, ip_nc, ip_sl;
    // an ideal element's jump J (consistent_start), for the record of its fast part
    // J − J_i at the step's commit (the tracker's mode)
    struct IdealJump {
        Real t = 0;
        std::vector<Real> j;
        std::vector<char> move;
    };
    std::vector<IdealJump> ideal_pending;
    // the jump's vector series (consistent_start): the reading's x3, the later
    // stages and their sources; the start mode's probe (ip_y1, ip_bits1, ip_m1)
    Vector ip_x3m, ip_xk, ip_sk, ip_y1, ip_y2, ip_w, ip_c, ip_g;
    std::vector<Vector> ip_xs;   // the series' stages 4, 5, …
    DenseMatrix ip_Dw;
    std::vector<Real> ip_m1, ip_f0, ip_s0, ip_s0v;
    std::vector<Size> ip_js;
    std::vector<bool> ip_bits1;
    // An ideal element's jump (the tracker records it): stage 1 from x0+ for
    // the step's lines only (ln_pre), the run keeps its own stage 1.
    Vector ln_x1;
    std::vector<Real> ln_v0;
    std::vector<bool> ln_bits;
    bool ln_pre = false;
    pwl::StatefulHistories::Snapshot ip_hs;
    std::vector<solver::ReinitRecord> impulse_pending;
    Vector event_rl_x;
    // F-SB-1 (event_right_limit): the probe's verdict for the right limit
    // (a jump; an ideal element's), the moved histories (ip_hist stays at
    // x0), and
    // the right-limit solve's scratch.
    bool ip_jump = false, ip_struct = false, ip_moved = false;
    // the nonlinear capacitors consistent_start carried: (k, v0, q0) at x0
    std::vector<std::tuple<Size, Real, Real>> rl_nc0, rl_sl0;
    std::vector<Real> ip_hist_p, rl_hist, rl_hw, rl_hcur;
    Vector rl_x1, rl_b0, rl_b, rl_bs, rl_h, rl_xa, rl_xb, rl_rr, rl_dx;
    std::vector<Real> rl_ncq, rl_slq;
    std::vector<Size> rl_pin;
    std::vector<std::pair<Size, Real>> rl_dep_x;
    std::vector<Index> rl_dep;
    std::vector<bool> rl_bits, rl_lsb;
    // a LOCATED diode crossing (split_step): its time, the post-event bits,
    // whether only V_f = 0 diodes flip (vf0) and nothing else coincides
    // (pure), and whether it was landed on the step end b (at_b). One
    // object: the per-step closures capture one reference.
    struct RlLoc {
        Real t = std::numeric_limits<Real>::quiet_NaN();
        bool vf0 = false, pure = false, at_b = false, off = false;
        std::vector<bool> bits;
    } rl_loc;
    // A diode turn-off located inside a restart step's stage 2 (run_restart:
    // the step is cut there, restart_lined): armed by restart_lined only;
    // t (NaN: none), the post-event bits, and restart_lined's entry state.
    struct RsCut {
        bool arm = false, vf0 = false;   // vf0: only V_f = 0, ungated diodes flip
        bool off = false;                // a turn-off among the cut's flips
        bool piece1 = false;             // run_restart is the cut's first piece
        Real t = std::numeric_limits<Real>::quiet_NaN();
        std::vector<bool> bits, b0, lsb0, pre;   // pre: the bits before the cut's flips
        Vector x0, x_end;
        std::vector<Real> h0;
        pwl::StatefulHistories::Snapshot hs0;
    } rs_cut;
    // event_right_limit's pin (Anderson): buffers sized once per run
    Vector rlp_u, rlp_r, rlp_g, rlp_r0, rlp_g0, rlp_sc, rlp_rs, rlp_gam, rlp_ua;
    DenseMatrix rlp_dr, rlp_dg;
    pwl::StatefulHistories::Snapshot rl_hs;
    // The OFF state's leakage conductance in total (conductance-form
    // switches and diodes): the restart rule's current resolution
    // (`consistent_start`).
    const Real g_leak = [&] {
        Real g = 0;
        for (Index bi = 0; bi < graph.num_branches(); ++bi) {
            if (!pool.is_registered(bi)) continue;
            if (pool.kind_of(bi) == pwl::DevicePool::StoredKind::Switch) {
                g += pool.switch_g_off(bi);
            }
        }
        for (const auto& d : diodes.entries()) g += d.g_off;
        return g;
    }();
    Real event_rl_t = std::numeric_limits<Real>::quiet_NaN();
    std::vector<CommutationEvent> step_events;      // per-step event scratch
    std::vector<bool> post_event_bits;              // event-split scratch
    std::vector<bool> decided_bits;                 // the split's decided mode
    // add-fixed-step-event-splitting workspace: the step's first solve
    // (pre-event bits) and its mask, the located gate/source events,
    // the diode events located in the step.
    Vector x_first;
    Vector loc_xt;   // the split locator's trial at a refined instant
    topology::SwitchStateMask mask_first(
        static_cast<std::size_t>(graph.num_switches()));
    constexpr Size kMaxStepEvents = 8;
    std::array<Real, kMaxStepEvents> ev_t{};
    // The gate sample of the interval that starts at ev_t[i]: the LAST
    // event of a cluster merged within `snap` (review M1).
    std::array<Real, kMaxStepEvents> ev_s{};
    std::vector<CommutationEvent> split_events;
    split_events.reserve(8);
    std::vector<Real> cross_scratch;
    std::vector<Index> island;          // the split's floating islands
    std::vector<char> off_branch;
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
    // add-pwl-igbt-and-thyristor: the stamped mask at the step start,
    // so a gated device's commutation record follows its stamped state.
    topology::SwitchStateMask gated_prev_stamp = stamped_mask;
    // F-Z-14: the same for a branch-form diode's flip — whether the step
    // starts at a gate edge (its gate bits changed; the diode bits ignored)
    topology::SwitchStateMask ideal_prev_stamp = stamped_mask;
    auto gate_edge_since = [&](const topology::SwitchStateMask& prev) {
        const auto d = diodes.current_diode_mask();
        return !(combine_masks(stamped_mask, d, diode_owned) ==
                 combine_masks(prev, d, diode_owned));
    };
    // the conductance form's watched signal v_d − V_on
    auto branch_form_s = [](const pwl::DiodeEventState::DiodeEntry& e, const Vector& xs) {
        return e.law_voltage(xs, false) - models::SwitchedDiode::turn_on_voltage(e.params());
    };

    // Event-iteration cap. 0 disables iteration entirely
    // (matching Layer 5 V2 behaviour for regression testing).
    const Size max_iters = opts.max_event_iterations;

    // ---- add-complementarity-mode-selection on pwl (tasks 4.2, 4b.3, 6.3)
    // Every decision site (a step, a restart stage, a split sub-step, a
    // static step) can pose the LCP of its own diode k-port
    // (pwl/mode_ports.hpp): beside the settle ('shadow'), instead of it
    // ('lcp'), or after a settle that ended in a mode violating a diode's
    // law (the default guard). Not posed: Newton runs and branch-form
    // diodes ('shadow' / 'lcp' are refused there), max_event_iterations = 0.
    if (opts.mode_selection < 0 || opts.mode_selection > 2) {
        throw std::invalid_argument(
            "run_transient: mode_selection must be 0 (settle), 1 (shadow) or 2 (lcp)");
    }
    if (opts.mode_selection != 0 && !has_diodes) {
        throw UnsupportedError(
            "pwl", "mode_selection", "", "",
            "engine 'pwl': mode_selection decides PWL diodes and this circuit has "
            "none: nothing to select. Drop mode_selection.");
    }
    std::optional<pwl::PwlModePorts> mode_ports;
    events::ModeSelector mode_selector;
    events::PassivityClass mode_passivity;
    std::vector<std::string> mode_diode_names, mode_device_names;
    Vector mode_x0, mode_xj, mode_u, mode_xs;
    DenseMatrix mode_Z;
    std::vector<bool> mode_bits0, mode_bits;
    std::vector<char> mode_prev_on, mode_coords, mode_settled;
    // add-pwl-igbt-and-thyristor: the mode LCP and judge_mode know no gate
    // (they would turn a gate-off IGBT on), so gated circuits keep the settle.
    const bool has_gated = diodes.num_gated() > 0;
    if (opts.mode_selection != 0 && has_gated) {
        std::string gated_name;
        for (const auto& e : diodes.entries()) {
            if (e.gate != models::SwitchedDiode::Gate::None) {
                gated_name = std::string(graph.branch_name(e.branch_id));
                break;
            }
        }
        throw UnsupportedError(
            "pwl", "mode_selection", gated_name, "Diode",
            std::format("engine 'pwl': mode_selection with the gated device '{}' "
                        "(IGBT with reverse blocking, GTO, thyristor or TRIAC): the "
                        "complementarity port knows no gate. Drop mode_selection.",
                        gated_name));
    }
    if (has_diodes && (opts.mode_selection != 0 || (opts.mode_guard && !has_gated))) {
        mode_ports.emplace(cache, graph, pool, diodes);
        // A branch-form diode is a port of its own R_on (an ideal one is
        // posed OFF: task 6.1). The default guard stays off on such circuits:
        // it would change the numbers of a run that asked for nothing.
        const bool bf = !mode_ports->branch_form_diode().empty() && opts.mode_selection == 0;
        const char* mname = opts.mode_selection == 1 ? "shadow" : "lcp";
        if (opts.mode_selection != 0 && nl_refresh_effective) {
            throw UnsupportedError(
                "pwl", "mode_selection", "", "",
                std::format("engine 'pwl': mode_selection='{}' with a Newton device "
                            "in the circuit: the diode port model is linear, a Newton "
                            "device's stamp moves with its operating point. Drop "
                            "mode_selection.", mname));
        }
        if (opts.mode_selection != 0 && max_iters == 0) {
            throw UnsupportedError(
                "pwl", "mode_selection", "", "",
                std::format("engine 'pwl': mode_selection='{}' with "
                            "max_event_iterations=0: the diode modes are frozen, "
                            "nothing is decided. Drop mode_selection or allow event "
                            "iterations.", mname));
        }
        if (bf || nl_refresh_effective || max_iters == 0) {
            mode_ports.reset();
        } else {
            mode_passivity = pwl::passivity_of(graph, pool);
            mode_selector.passive = mode_passivity.passive();
            for (const auto& e : diodes.entries()) {
                mode_diode_names.push_back(pwl::branch_label(graph, e.branch_id));
                mode_device_names.emplace_back(graph.branch_name(e.branch_id));
            }
        }
    }
    result.mode_report.mode = mode_ports ? opts.mode_selection : 0;
    result.mode_report.passivity = mode_passivity.name();
    result.mode_report.passivity_device = mode_passivity.device;
    result.mode_report.passivity_kind = mode_passivity.kind;
    result.mode_guard.enabled =
        mode_ports.has_value() && opts.mode_guard && opts.mode_selection != 2;

    auto mode_mask_of = [&](const std::vector<char>& on) {
        topology::SwitchStateMask m(static_cast<std::size_t>(graph.num_switches()));
        const auto& es = diodes.entries();
        for (std::size_t j = 0; j < es.size(); ++j)
            if (on[j]) m.set(es[j].switch_idx, true);
        return m;
    };
    auto mode_str = [](const auto& on) {
        std::string s;
        for (const auto c : on) s += c ? '1' : '0';
        return s;
    };
    auto mode_kind = [&](Real for_dt, bool be) -> const char* {
        return for_dt <= Real{0} ? "static"
             : be ? "restart" : for_dt == opts.dt ? "step" : "substep";
    };
    auto mode_count_site = [](ModeReport& r, std::string_view kind) {
        if (kind == "step") ++r.sites_step;
        else if (kind == "restart") ++r.sites_restart;
        else if (kind == "substep") ++r.sites_substep;
        else ++r.sites_static;
    };
    auto mode_note = [](ModeReport& r, const events::ModeSelection& s) {
        ++r.lcp_events;
        r.lcps += static_cast<Size>(s.lcps);
        r.pivots_total += static_cast<Size>(s.pivots_total);
        r.max_pivots_per_event = std::max(r.max_pivots_per_event, s.pivots_total);
        r.ties += static_cast<Size>(s.ties);
        r.widenings += static_cast<Size>(s.widenings);
        r.capped += static_cast<Size>(s.capped);
        r.leak_pins += static_cast<Size>(s.leak_pins);
        r.pin_judgements += static_cast<Size>(s.pin_judgements);
        r.unfixed += static_cast<Size>(s.unfixed);
        r.enumerations += static_cast<Size>(s.enumerations);
        r.nonunique_modes += static_cast<Size>(s.nonunique_modes);
        r.released += static_cast<Size>(s.released);
        r.shorted_held += static_cast<Size>(s.shorted_held);
        r.shorted_released += static_cast<Size>(s.shorted_released);
    };
    auto mode_failed = [](const events::ModeSelection& s) {
        return s.status != events::LcpStatus::Solution &&
               s.status != events::LcpStatus::Trivial;
    };
    // The site's start: x0 and the diode bits its first solve uses.
    auto mode_begin = [&]() {
        mode_x0 = x;
        diodes.snapshot_on_bits_into(mode_bits0);
    };
    // The LCP of the site J(gates + start bits, for_dt)·s = −(b_c + bx);
    // be: a restart stage (γ·h on the factor of 2γh); for_dt = 0: static.
    auto mode_select = [&](const topology::SwitchStateMask& gates, Real for_dt,
                           bool be, const Vector& bx, const Vector* x_site = nullptr) {
        const auto& es = diodes.entries();
        mode_prev_on.assign(es.size(), 0);
        for (std::size_t j = 0; j < es.size(); ++j) mode_prev_on[j] = mode_bits0[j] ? 1 : 0;
        const Real h_site = for_dt <= Real{0} ? Real{1} : be ? for_dt * Real{0.5} : for_dt;
        // Every pass, the selector's tie widenings included, is posed on the
        // site's own step (DECISIONS F2): the next site is one step away, and
        // a 100·h / 1e4·h backward-Euler look-ahead with the sources frozen
        // decided a floating bridge's ties on a trajectory that never happens.
        // pwl::select_site is the engine-neutral driver (trbdf2 hook: not yet).
        return pwl::select_site(
            mode_selector, *mode_ports, cache, diodes,
            [&](const std::vector<char>& on) {
                return combine_masks(gates, mode_mask_of(on), diode_owned);
            },
            mode_prev_on, mode_x0, for_dt, h_site, bx, mode_Z, mode_u, mode_xj, mode_coords,
            x_site);
    };
    // The quiet-site test of 'lcp' (DECISIONS #40): every diode's crossing
    // signal on the start solve `xs` clear of zero by kQuietBand of the
    // circuit scale (10× the selector's tie tolerance, with its σ·C_max/h
    // term): the LCP reads the same signals from the same solve.
    topology::SwitchStateMask mode_last_gates;
    bool mode_gates_known = false;
    Vector mode_xq, mode_xc;
    std::unordered_map<topology::SwitchStateMask, std::vector<bool>> mode_answers;
    // a hysteresis port (V_th > V_f) turns ON only past V_th on the START solve,
    // which a cached answer's own solve does not show: no answer cache then
    const bool mode_hysteresis = mode_ports.has_value() &&
        std::any_of(mode_ports->ports().begin(), mode_ports->ports().end(),
                    [](const events::DiodePort& p) { return p.V_th > p.V_f; });
    auto mode_quiet = [&](const std::vector<bool>& bits, const topology::SwitchStateMask& m,
                          Real for_dt, Real h, const Vector& bx, const Vector& xs) {
        constexpr Real kQuietBand = Real{1e-8};
        const Vector* xq = &xs;
        if (mode_ports->has_islands(m)) {
            mode_xq = xs;
            mode_ports->correct(m, for_dt, bx, mode_xq);
            xq = &mode_xq;
        }
        const auto sc = mode_ports->scale(mode_x0);
        const Real band_v = kQuietBand * sc.vscale;
        const Real band_i = kQuietBand * sc.iscale + band_v * sc.c_max / h;
        const auto& es = diodes.entries();
        for (std::size_t j = 0; j < es.size(); ++j) {
            const auto p = es[j].params();
            const bool on = bits[j];
            const Real v = es[j].law_voltage(*xq, on);
            if (on ? !(models::SwitchedDiode::branch_current(true, v, p) > band_i)
                   : !(v - models::SwitchedDiode::turn_on_voltage(p) < -band_v)) {
                return false;
            }
        }
        return true;
    };
    // 'lcp': the LCP decides the site; one solve under its mode. Returns
    // whether the mode changed from the start bits (DECISIONS #40). The start
    // mode is solved first (`x_start`: given by the caller, or solved here).
    //   * quiet site — any site whose start solve gives every diode's
    //     crossing signal (ON: the current; OFF: the gap to V_on) clear of
    //     zero by kQuietBand of the circuit scale, after the floating
    //     islands' KCL: the LCP's q is that start mode, nonnegative, so Lemke
    //     returns it with no pivot and the start solve is the step. With an
    //     ideal diode ON (posed open: Lemke pivots) only on a passive circuit,
    //     whose LCP has one solution.
    //   * cached site — the LCP's last answer from the same start mask, clear
    //     of the band on its own solve: kept on a passive circuit without
    //     ideal or hysteresis ports (one solution, and nothing held).
    //   * the LCP reads the port voltages from that solve (no ideal diode ON
    //     at the start: the LCP poses those cut out, another system) and,
    //     when it keeps the start mode, the start solve is the step.
    // Not on a static site whose nominal segment is not the cache's static
    // factor (other bits than solve_at's).
    auto mode_lcp_site = [&](Real at_t, const char* kind,
                             const topology::SwitchStateMask& gates, Real for_dt,
                             bool be, const Vector& bx, auto&& solve,
                             const Vector* x_start = nullptr) {
        auto& r = result.mode_report;
        mode_count_site(r, kind);
        const auto start_mask = combine_masks(gates, diodes.current_diode_mask(), diode_owned);
        const bool same_gates = mode_gates_known && gates == mode_last_gates;
        mode_last_gates = gates;
        mode_gates_known = true;
        bool ideal_on = false;
        for (std::size_t j = 0; j < mode_bits0.size(); ++j)
            ideal_on = ideal_on || (mode_bits0[j] && mode_ports->ideal(j));
        // An ideal diode ON at a gate edge may close a loop with the new
        // gates (the CCM turn-on): its start mask is singular, and the LCP
        // poses it cut out anyway. With the previous site's gates the start
        // mask is the mask that site solved.
        if ((for_dt <= Real{0} && cache.dt() != Real{0}) || (ideal_on && !same_gates)) {
            x_start = nullptr;
        } else if (!x_start) {
            solve(start_mask, mode_xs);
            x_start = &mode_xs;
        }
        const Real h_site = for_dt <= Real{0} ? Real{1} : be ? for_dt * Real{0.5} : for_dt;
        // (a) the start mode, clear of the band: the LCP's answer
        if (x_start && (!ideal_on || mode_selector.passive) &&
            mode_quiet(mode_bits0, start_mask, for_dt, h_site, bx, *x_start)) {
            x = *x_start;
            // the step stores what the LCP decided on: the islands' KCL imposed
            // (F-2-3: a rounded common mode read a decided-OFF diode forward-biased)
            mode_ports->correct(start_mask, for_dt, bx, x);
            stamped_mask = start_mask;
            stamped_known = true;
            last_solved_bits = mode_bits0;
            ++r.quiet_sites;
            return false;
        }
        // (b) the LCP's last answer from this start mask (the active-set cache),
        // clear of the band on its own solve: the LCP's answer
        if (x_start && mode_selector.passive && !structure.active() && !mode_hysteresis) {
            const auto it = mode_answers.find(start_mask);
            if (it != mode_answers.end() && it->second != mode_bits0) {
                diodes.restore_on_bits(it->second);
                const auto cmask = combine_masks(gates, diodes.current_diode_mask(), diode_owned);
                solve(cmask, mode_xc);
                if (mode_quiet(it->second, cmask, for_dt, h_site, bx, mode_xc)) {
                    x = mode_xc;
                    mode_ports->correct(cmask, for_dt, bx, x);
                    mode_bits = it->second;
                    stamped_mask = cmask;
                    stamped_known = true;
                    last_solved_bits = mode_bits;
                    ++r.cached_sites;
                    return true;
                }
                diodes.restore_on_bits(mode_bits0);
            }
        }
        if (ideal_on) x_start = nullptr;
        const auto sel = mode_select(gates, for_dt, be, bx, x_start);
        mode_note(r, sel);
        r.port_solves = mode_ports->solves();
        r.port_matrices = mode_ports->port_builds();
        if (mode_failed(sel) && mode_Z.rows() == static_cast<Index>(mode_diode_names.size())) {
            // An ideal port shorted by a loop (Z_jj = 0) that the loop
            // forward-biases: refused by the name the settle's release uses.
            // (the selector's own short test: |Z_jj|·iscale <= tol_w)
            const auto sc = mode_ports->scale(mode_x0);
            const Real tol_w = mode_selector.tol * sc.vscale;
            for (std::size_t j = 0; j < mode_diode_names.size(); ++j) {
                const auto jj = static_cast<Index>(j);
                if (!mode_ports->ideal(j) || std::abs(mode_Z(jj, jj)) * sc.iscale > tol_w ||
                    !(mode_u[jj] - mode_ports->ports()[j].V_f > tol_w)) {
                    continue;
                }
                // name the loop: the posed mask with this diode closed
                std::string loop = mode_diode_names[j];
                if (structure.active()) {
                    std::vector<char> on_j = mode_coords;
                    on_j[j] = 1;
                    const auto& ms = structure.check(
                        combine_masks(gates, mode_mask_of(on_j), diode_owned), for_dt > Real{0});
                    const Index bj = diodes.entries()[j].branch_id;
                    for (std::size_t m = 0; m < ms.loop_diodes.size(); ++m)
                        if (ms.loop_diodes[m] == bj) loop = ms.loop_texts[m];
                }
                throw UnsupportedError(
                    "pwl", "ideal_voltage_loop", mode_device_names[j], "Diode",
                    std::format("engine 'pwl': at t = {} the ideal diode {} (R_on = 0) "
                                "closes a loop of voltage sources and closed ideal "
                                "elements (loop: {}) that forward-biases it by {} V "
                                "(mode_selection='lcp': no consistent mode): its "
                                "current is undefined. Give it R_on > 0.",
                                at_t, mode_diode_names[j], loop,
                                mode_u[jj] - mode_ports->ports()[j].V_f));
            }
            // No single port shorted: a loop through several ideal ports (each
            // posed open, none with Z_jj = 0) leaves the LCP a secondary ray.
            // Close every ideal port the posed network does not reverse-bias;
            // a loop among them is the refusal, named as the settle names it.
            if (structure.active() && sel.status == events::LcpStatus::SecondaryRay) {
                const auto sc = mode_ports->scale(mode_x0);
                const Real tol_w = mode_selector.tol * sc.vscale;
                const std::vector<char> on_f =
                    mode_selector.closure_candidate(mode_ports->ports(), mode_coords, tol_w);
                const auto& ms = structure.check(
                    combine_masks(gates, mode_mask_of(on_f), diode_owned), for_dt > Real{0});
                if (!ms.loop_diodes.empty()) {
                    const auto name = std::string(graph.branch_name(ms.loop_diodes.front()));
                    throw UnsupportedError(
                        "pwl", "ideal_voltage_loop", name, "Diode",
                        std::format("engine 'pwl': at t = {} the ideal diode {} (R_on = 0) "
                                    "closes a loop of voltage sources and closed ideal "
                                    "elements (loop: {}) that no mode of its diodes can "
                                    "carry (mode_selection='lcp': no consistent mode): its "
                                    "current is undefined. Give it R_on > 0.",
                                    at_t, name, ms.loop_texts.front()));
                }
            }
        }
        events::raise_if_refused(sel, "pwl", at_t, mode_diode_names, mode_passivity,
                                 &mode_device_names);
        mode_bits.assign(sel.on.begin(), sel.on.end());
        diodes.restore_on_bits(mode_bits);
        const auto mask = combine_masks(gates, diodes.current_diode_mask(), diode_owned);
        if (structure.active()) {
            // The LCP releases an ideal diode a loop reverse-biases (Z_jj = 0,
            // q_j > 0) and finds no mode when the loop forward-biases it, so a
            // loop diode left ON here is a defect, not a mode to release.
            const auto& ms = structure.check(mask, for_dt > Real{0});
            result.mode_structure_checks = structure.checks_run();
            if (!ms.loop_diodes.empty()) {
                throw UnsupportedError(
                    "pwl", "ideal_voltage_loop",
                    std::string(graph.branch_name(ms.loop_diodes.front())), "Diode",
                    std::format("engine 'pwl': at t = {} mode_selection='lcp' left the "
                                "ideal diode '{}' ON in a loop of voltage sources and "
                                "closed ideal elements (mode {}, loop: {}): its current "
                                "is undefined. Give it R_on > 0.",
                                at_t, graph.branch_name(ms.loop_diodes.front()),
                                mask.to_string(), ms.loop_texts.front()));
            }
        }
        if (x_start && mask == start_mask) {
            x = *x_start;                  // the start mode kept: its solve is the step
        } else {
            solve(mask, x);
        }
        mode_ports->correct(mask, for_dt, bx, x);
        stamped_mask = mask;
        stamped_known = true;
        last_solved_bits = mode_bits;
        if (x_start) {
            if (mode_answers.size() >= 256) mode_answers.clear();
            mode_answers[start_mask] = mode_bits;
        }
        if (pwl::judge_mode(diodes, sel.on, x, mode_ports->scale(mode_x0),
                            mode_selector.tol) != 1) {
            ++r.inconsistent;
        }
        return mode_bits != mode_bits0;
    };
    // 'shadow': the LCP beside a finished settle; a different mode is
    // recorded with both candidates judged on the site's own solve.
    auto mode_shadow_site = [&](Real at_t, const char* kind,
                                const topology::SwitchStateMask& gates, Real for_dt,
                                bool be, const Vector& bx, Size rounds, bool cycle) {
        auto& r = result.mode_report;
        // No solve at this site: no settled mode to compare.
        if (last_solved_bits.size() != diodes.entries().size()) return;
        mode_count_site(r, kind);
        r.settle_rounds_total += rounds;
        if (cycle) ++r.mask_cycles;
        try {
            const auto sel = mode_select(gates, for_dt, be, bx);
            mode_note(r, sel);
            if (mode_failed(sel)) {
                ++r.errors;
                r.last_error = std::string(events::to_string(sel.status));
            } else {
                mode_settled.assign(last_solved_bits.begin(), last_solved_bits.end());
                if (sel.on != mode_settled) {
                    ModeDisagreement d;
                    d.t = at_t;
                    d.site = kind;
                    d.settle_mode = mode_str(mode_settled);
                    d.lcp_mode = mode_str(sel.on);
                    d.previous_mode = mode_str(mode_prev_on);
                    d.settle_rounds = rounds;
                    d.pivots = sel.pivots_total;
                    d.status = events::to_string(sel.status);
                    const auto sc = mode_ports->scale(mode_x0);
                    d.settle_consistent = pwl::judge_mode(
                        diodes, mode_settled, x, sc, mode_selector.tol, &d.settle_violation);
                    cache.solve_at(combine_masks(gates, mode_mask_of(sel.on), diode_owned),   // factor-reuse
                                   for_dt, bx, mode_xj);
                    d.lcp_consistent = pwl::judge_mode(
                        diodes, sel.on, mode_xj, sc, mode_selector.tol, &d.lcp_violation);
                    r.disagreements.push_back(std::move(d));
                }
            }
        } catch (const std::exception& e) {
            ++r.errors;
            r.last_error = e.what();
        }
        r.port_solves = mode_ports->solves();
        r.port_matrices = mode_ports->port_builds();
    };
    // The default guard: the settled mode judged on its own solve; one that
    // violates a diode's law is replaced by the LCP's when that one is
    // consistent. Returns true when it repaired.
    auto mode_guard_site = [&](Real at_t, const topology::SwitchStateMask& gates,
                               Real for_dt, bool be, const Vector& bx, auto&& solve) {
        auto& g = result.mode_guard;
        if (last_solved_bits.size() != diodes.entries().size()) return false;   // no solve
        mode_settled.assign(last_solved_bits.begin(), last_solved_bits.end());
        const auto sc = mode_ports->scale(mode_x0);
        Real viol = 0;
        int vj = -1;
        if (pwl::judge_mode(diodes, mode_settled, x, sc, mode_selector.tol, &viol, &vj) == 1)
            return false;
        events::ModeSelection sel;
        try {
            sel = mode_select(gates, for_dt, be, bx);
        } catch (const std::exception& e) {
            ++g.failed;
            g.lcp.last_error = e.what();
            return false;
        }
        mode_note(g.lcp, sel);
        g.lcp.port_solves = mode_ports->solves();
        g.lcp.port_matrices = mode_ports->port_builds();
        if (mode_failed(sel)) { ++g.failed; return false; }
        if (sel.on == mode_settled) { ++g.confirmed; return false; }
        const auto mask = combine_masks(gates, mode_mask_of(sel.on), diode_owned);
        solve(mask, mode_xj);
        if (pwl::judge_mode(diodes, sel.on, mode_xj, sc, mode_selector.tol) != 1) {
            ++g.failed;
            return false;
        }
        x = mode_xj;
        mode_bits.assign(sel.on.begin(), sel.on.end());
        diodes.restore_on_bits(mode_bits);
        last_solved_bits = mode_bits;
        stamped_mask = mask;
        stamped_known = true;
        ++g.repairs;
        g.repair_times.push_back(at_t);
        g.repair_diodes.push_back(vj >= 0 ? mode_diode_names[static_cast<std::size_t>(vj)]
                                          : std::string{});
        g.repair_violations.push_back(viol);
        return true;
    };

    // Built-in time-varying source detection + reusable b_extra buffers
    // (audit #6/#16), shared by BOTH the dynamic and static paths below. The
    // PWM/sine/pulse helpers otherwise allocate + zero a state-size vector AND
    // scan every branch on every step; detect ONCE which kinds the circuit
    // contains, then reuse buffers and skip absent kinds in the hot loops.
    // The (branch, row) lists and the per-row adds live in
    // step::SourceAccumulator (refactor-unified-transient-stepper group 1),
    // shared with the TR-BDF2 engine and the DSED adapter; this engine keeps
    // its own association with the history and b_extra_fn (DECISIONS 7, 21).
    step::SourceAccumulator src_acc{pool, graph};
    // add-controlled-source-set: sine / pulse / table current sources
    // (node rows, two per source: kept on the dense helper).
    const bool has_iwave = src_acc.has_current_waveforms();
    Vector be_iwave;
    const bool has_pwm = src_acc.has_pwm();
    const bool has_pulse = src_acc.has_pulse();
    // `src_t`: PWM/pulse sample time; `sine_t`: the sine's; `lb`: a
    // jumping source at `src_t` reads value_at(src_t − lb) (as the dense
    // passes did).
    auto add_source_rows = [&](Vector& b, Real src_t, Real sine_t,
                               Real lb) {
        src_acc.add_voltage_rows(pool, b, src_t, sine_t, lb);
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
    events::MaskCycle<64> masks_seen;
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
    // The MNA-native machines' flux rows exist only in the Newton
    // refresh: their linear part is singular, so the cache cannot
    // factorise it and every nominal step failed into a dt/2 retry.
    // They take the assembled (unfactorised) carrier at every dt.
    const bool newton_only_rows = !pool.pmsm_mna_machines().empty() ||
                                  !pool.induction_machines().empty();
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
    // Release group B: a rule run keeps event limits (a stored sample at a
    // jump is its left limit; the dynamic path adds its restart lines)
    // (every step stored: a decimated run keeps none rather than some)
    if (restart_rule && opts.store_every == 1) {
        result.has_event_limits = true;
        result.event_limit_width = static_cast<Size>(x.size());
        result.event_limit_switches = graph.num_switches();
    }
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
    const Real snap = opts.dt * opts.split_snap;
    // the restart rule's gate sample without the split (F-SB-2): just past
    // mid-step back from its end, so an edge inside the step takes effect at
    // the nearer boundary (a tie at mid-step: the end). 0.5 + 1e-4, not 0.51:
    // 0.51 sent an edge at 0.49-0.5 h to the farther point (RL error -6.13e-3
    // at 0.495 h vs +5.95e-3 at 0.49 h, dt 2 us); 1e-4 h stays far above a
    // closure's 1e-9-of-a-period band. The step end (t - dt/100) dropped
    // every pulse narrower than a step that starts on the grid (the
    // closed-loop boost's 5 % duty floor: 0 % applied, 0.216 against trbdf2)
    constexpr Real gate_back = Real{0.5001};
    // A source jump snapped to the step end (t - snap <= tau < t): the
    // solve AT `src_end_at` reads the built-in sources at `src_end_read`
    // (= tau, so value_at(tau - left_by) is the pre-jump level). NaN
    // compares unequal to every time: off.
    Real src_end_at = std::numeric_limits<Real>::quiet_NaN();
    Real src_end_read = Real{0};
    // add-in-kernel-event-losses: the loss ports (none: no tracker, and
    // nothing below changes a bit of the run).
    std::optional<LossTracker> loss_tracker;
    if (!pool.device_losses().empty()) loss_tracker.emplace(graph, pool, "pwl");
    LossTracker* const lt = loss_tracker ? &*loss_tracker : nullptr;
    // add-realistic-passives D18: the Bergeron lines' committed past (none:
    // no object, nothing below reads or writes it).
    std::optional<pwl::LineHistory> line_history;
    if (!pool.line_pairs().empty()) {
        line_history.emplace(graph, pool);
        const auto& lname = line_history->tau_min_name();
        if (!(cache.dt() > Real{0})) {
            throw UnsupportedError("pwl", "static_path", lname, "TransmissionLine",
                "run_transient: the transmission line '" + lname + "' needs the "
                "dynamic path (a cache built with dt > 0, as simulate() builds it).");
        }
        if (opts.dt > line_history->tau_min()) {
            throw UnsupportedError("pwl", "dt", lname, "TransmissionLine", std::format(
                "run_transient: dt = {} s is longer than the delay {} s of the "
                "transmission line '{}': its far-end wave would be read from the "
                "future. Use dt <= delay (a divisor of it is exact).",
                opts.dt, line_history->tau_min(), lname));
        }
        if (resume_snap != nullptr) {
            line_history->restore(resume_snap->line_state, "run_transient",
                                  opts.t_start - resume_snap->t);
        } else {
            // a DC point or a given state (seeded as one, as the companions
            // are): its waves; the zero start: the line at rest
            line_history->start(opts.t_start, x,
                                start_from_dc_op || initial_state != nullptr);
        }
    }
    pwl::LineHistory* const lines = line_history ? &*line_history : nullptr;
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
        if (has_iwave) {
            // src_t: a pulse current jump snapped to the step end reads
            // its left limit, as the voltage sources do. shortcut: a sine
            // current is then read at tau (< 0.01 dt early), upgrade if
            // the helper ever takes two times.
            sources::compute_current_waveform_b_extra(pool, graph, src_t,
                                                      be_iwave, left_by);
            b_extra += be_iwave;
        }
        if (lt) lt->inject(b_extra);   // heat owed for this step
        if (lines) lines->inject(b_extra, at_t);   // the waves tau ago (D18)
    };

    // v2.0 Phase 2: a run that dies part-way keeps what it computed.
    // The step loops below can throw from a dozen places; whichever
    // one it is, the samples already recorded are worth more than the
    // stack they would otherwise unwind with. `t_reached` tracks the
    // step being attempted so the exception can say where it stopped.
    Real t_reached = opts.t_start;
    Size k_reached = 0;   // grid index of t_reached (snapshot)
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
                    ? diodes.combine(switch_fn(opts.t_start))
                    : switch_fn(opts.t_start);
            stamped_known = !seeded;
            result.record_switch_state(stamped_mask,
                                       /*solved=*/!seeded);
        }

        // Prime the observer with the initial condition so a
        // discrete-time PI / sampler / lookup-table sees the
        // true starting state at t_start.
        if (step_observer || step_observer_in) {
            (has_ticks ? step_observer : step_observer_in)(opts.t_start, x);
        }
        if (lt) {
            if (resume_from != nullptr && !resume_from->loss_state.empty()) {
                lt->restore(resume_from->loss_state);   // the heat in flight too
            } else {
                lt->begin(x, stamped_mask);
            }
        }

        if (hooks) hooks->start(x, stamped_mask, {&history, &hs, nullptr});
        if (sens.on()) {
            sens.refuse_setup(lt != nullptr, static_cast<bool>(nl_refresh),
                              opts.enable_newton_lm);
            sens.start(opts.t_start, x);
            if (stamped_known) sens.start_mode(stamped_mask);   // seeded: decided at step 1
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
        constexpr Size kRestartSteps = step::RestartPolicy::kSteps;
        constexpr Real kGamma = step::RestartPolicy::kGamma;
        Size restart_left =
            !restart_rule ? Size{0}
            : resume_snap != nullptr ? resume_snap->restart_steps_left
            : start_from_dc_op ? Size{0} : kRestartSteps;
        const bool has_jump_sources = restart_rule && (has_pwm || has_pulse || has_iwave);
        // Both SDIRK stages solve on the trapezoidal factor of 2γ·dt: a
        // second FIXED step, kept by the cache's event table (one entry
        // per mask, the segment byte budget, the regime's shared
        // symbolic analysis, in-place refactor; a parameter refactor
        // drops it). The 8-entry LRU this table replaced refactored on
        // every restart step of a VSI (3172 builds in 3157).
        const Real restart_dt = Real{2} * (kGamma * opts.dt);
        Size restart_factorizations = 0;

        pwl::NewtonRoundoffAcceptance newton_roundoff_run;
        newton_roundoff_run.enabled = true;
        newton_roundoff_run.max_consecutive = opts.newton_roundoff_budget;

        // ---- add-fixed-step-resolution-diagnostics (read-only) ----
        // Truncation: e = (h/12)|f_{k+1} − 2f_k + f_{k−1}| per linear
        // C/L (f = i/C, v/L from the committed companion), over the
        // variable's running peak; valid only on three plain steps under
        // one stamped mask. Ringing: per state entry, runs of
        // sign-alternating differences above max(1 % of the local
        // magnitude, 1e-9 of the peak); a run of >= 4 is recorded once.
        const bool diag = opts.resolution_diagnostics;
        auto& rd = result.resolution;
        rd.enabled = diag;
        // Two buffers for the whole workspace (the zero-alloc budget):
        // per C/L f_k, f_{k−1}, peak |y|, 1/C or 1/L; per state entry
        // Δ_{k−1}, peak |x|, run start, run max |Δ|; run length, record.
        Size rd_run = 0;
        topology::SwitchStateMask rd_prev_mask;
        Real rd_since_store = 0;
        std::vector<Real> rd_buf;
        std::vector<Size> rd_ibuf;
        Real *rd_f1 = nullptr, *rd_f2 = nullptr, *rd_peak = nullptr,
             *rd_inv = nullptr, *rd_dprev = nullptr, *rd_xpeak = nullptr,
             *rd_t0 = nullptr, *rd_dmax = nullptr;
        Size *rd_cnt = nullptr, *rd_open = nullptr;
        Size rd_steps = 0, rd_unest = 0;
        Real rd_kind_peak[2] = {Real{0}, Real{0}};   // capacitors, inductors
        constexpr Size kRdNone = std::numeric_limits<Size>::max();
        if (diag) {
            const auto& ents = history.entries();
            const Size nh = ents.size();
            const auto n = static_cast<Size>(x.size());
            rd_buf.assign(4 * nh + 4 * n, Real{0});
            rd_ibuf.assign(2 * n, Size{0});
            rd_f1 = rd_buf.data();
            rd_f2 = rd_f1 + nh;
            rd_peak = rd_f2 + nh;
            rd_inv = rd_peak + nh;
            rd_dprev = rd_inv + nh;
            rd_xpeak = rd_dprev + n;
            rd_t0 = rd_xpeak + n;
            rd_dmax = rd_t0 + n;
            rd_cnt = rd_ibuf.data();
            rd_open = rd_cnt + n;
            std::fill(rd_open, rd_open + n, kRdNone);
            for (Size j = 0; j < nh; ++j) {
                rd_inv[j] = Real{1} / ents[j].C_or_L;
            }
            rd.n_branches_covered = nh;
            rd.lte.reserve(result.times.capacity());
        }

        // ---- refactor-unified-transient-stepper slice 3a (DECISIONS 13) --
        // The step is five phases (begin, take, after_solve, commit,
        // record), each ONE lambda that the step skeleton calls
        // (`step::run<Trapezoidal>(FixedController, IndexPhaseBody)`; the
        // only driver since the legacy loop's deletion, DECISIONS 19).
        // The step's locals that cross phases live here, (re)set at the
        // program point where the loop body declared them. The step's
        // times are re-read into a const local at the top of each phase:
        // a hoisted, mutable `t` is reloaded after every opaque call (the
        // commutation-record loop runs per diode), a const local is not.
        Real t_step = Real{0};
        Real t_prev_step = Real{0};
        Real gate_t = Real{0};
        Size n_ev = 0;
        bool ev_overflow = false;
        bool capture_first = false;
        Real gate_t0 = Real{0};
        // F-Z-1: the split instant a step's last mode change was taken at and
        // the state there (its left limit), for the dependent-mode record.
        // NaN: none this step (the record reads t_prev and x_prev).
        Real mode_ev_t = std::numeric_limits<Real>::quiet_NaN();
        Vector mode_ev_x, mode_ev_x_scratch;
        topology::SwitchStateMask mask_pre;
        Size iters = 0;
        bool flipped = false;
        bool mask_cycle = false;
        Real committed_dt = opts.dt;
        bool step_retried = false;
        bool edge_restart_step = false;
        bool restart = false;
        bool committed_be = false;
        // The last restart trigger (written only when one fires) and how far
        // its event can lie from the step start (a built-in source / b_extra
        // jump inside a step without the split): δt_loc of the impulse
        // probe (consistent_start) at a TIME event.
        Real rs_trig_t = std::numeric_limits<Real>::quiet_NaN();
        Real rs_trig_loc = 0;
        // the trigger is an event (F-SB-1): not an unknown start's stamp
        bool rs_trig_ev = false;
        // F-Z-3: the split's last event piece [rs_piece_a, t] probes only the
        // modes faster than γ·(t − rs_piece_a); the grid step at anchor_t
        // probes the rest as a time event at its start (δt_loc = 0)
        Real rs_piece_a = 0;
        Real anchor_t = split && resume_snap != nullptr && resume_snap->restart_anchor_at_end
                            ? opts.t_start
                            : std::numeric_limits<Real>::quiet_NaN();
        // ... and only diode bits changed there (no gate edge, no source
        // jump): a crossing landed on that grid point stays a continuity
        // point (event_right_limit)
        bool rs_trig_dio = false;
        Real be_commit_h = kGamma * opts.dt;
        // Release group B: the lines of the step under the restart rule
        // (StepLines): stage 1 of the last restart (sub)step and how its
        // stage 2 was taken; `q_ok` false: the step is not representable
        // (the dt-retry ladder, more events than lines).
        // Allocated under the rule only: the default loop sees one null
        // pointer (instructions per step, DECISIONS 14/16/18).
        const auto lines_box = restart_rule ? std::make_unique<StepLines>() : nullptr;
        StepLines* const L = lines_box.get();
        // The step hooks, called in the shared phases (slice 3c,
        // DECISIONS 17): probes R1 (`StepHooks::step`, the step's start
        // state and mask), R2 (`on_event`, the stamps on each side of a
        // commutation) and the device interface's #31 (`device_state` after
        // the commit). The start mask is copied only with a listener, at
        // the end of the previous step's record (one test per step without
        // one); a seeded start's guessed stamp is no R2 left side.
        const bool hook_step = hooks != nullptr && hooks->has_step();
        const bool hook_event = hooks != nullptr && hooks->has_event();
        const bool hook_mask = hook_step || hook_event;
        // probes 0b.10: run-invariant flags, not the listener lists, per
        // step; each step starts from the previous one's committed end
        // (x_prev = x, mask_start = its stamp), so every step but the first
        // is contiguous for the step sink (it skips its own jump test).
        const bool hook_ds = hooks != nullptr && hooks->has_device_state();
        const bool hook_acc = hooks != nullptr && !hooks->empty();
        bool hook_contig = false;
        topology::SwitchStateMask mask_start;
        bool mask_start_known = false;
        if (hook_mask) {
            mask_start = stamped_mask;
            mask_start_known = stamped_known;
        }
        auto commit_backward_euler = [&](Real h_e) {
            step::commit(history, hs, x, h_e, pwl::TrBdf2Stage::BackwardEuler);
        };
        // add-parameter-sensitivities group 6: the calls into the recurrence
        // (solver/step/sensitivity.hpp) from the event and restart paths; the
        // jump itself lives there. `sens_inline`: this step was opened (its
        // gate change at t_prev taken) by its first call into the recurrence;
        // reset by the step's closing block. Nothing runs per step when the
        // sensitivities are off.
        bool sens_inline = false;
        topology::SwitchStateMask sens_gate;
        bool sens_gate_known = false;
        auto sens_stage = [&](int s, const step::StagePsi& psi, Real at_t, Real for_dt) {
            if (nl_refresh_effective) {
                sens.stage(s, psi, stamped_mask, x, at_t,
                           sens.newton_factor(newton_ws, at_t), sens_J_lin);
            } else {
                sens.stage(s, psi, stamped_mask, x, at_t,
                           cache.resident_factor(stamped_mask, for_dt), nullptr);
            }
        };
        auto sens_gate_event = [&](Real at, Real g_t, bool located) {
            if (!sens_gate_known) {   // an edge inside the first step counts too
                sens_gate = gate_at(opts.t_start);
                sens_gate_known = true;
            }
            const topology::SwitchStateMask g = gate_at(g_t);
            if (!(g == sens_gate)) sens.time_event(at, sens_gate, g, located);
            sens_gate = g;
        };
        // the step's gate change at its start (gate_t0 = the gate time the
        // step started with), before any stage of the step
        auto sens_open = [&]() {
            if (sens_inline) return;
            sens_inline = true;
            // located only by the schedule's oracle: an opaque switch_fn's
            // edge is sampled at a grid point (a staircase in a duty)
            sens_gate_event(t_prev_step, gate_t0,
                            split && !ev_overflow && static_cast<bool>(next_edge_fn));
        };
        auto sens_begin = [&](Real t_from, const Vector& x0) {
            sens_open();
            sens.begin_step(t_from, x0);
        };
        // the located diode crossings (split_events) at tau; the pre-event
        // mode's factor is the first solve's (mask_first at for_dt)
        auto sens_diode_events = [&](Real tau, Real for_dt) {
            const auto ents = diodes.entries();
            auto src = [&](Real ts, Vector& out) {
                add_source_rows(out, ts, ts, Real{0});
                if (has_iwave) {
                    sources::compute_current_waveform_b_extra(pool, graph, ts, be_iwave,
                                                              Real{0});
                    out += be_iwave;
                }
            };
            for (const auto& ev : split_events) {
                for (const auto& e : ents) {
                    if (e.branch_id != ev.branch_id) continue;
                    if (nl_refresh_effective) {
                        sens.refuse_newton_event(tau, graph.branch_name(e.branch_id));
                        sens.refuse_step(tau, "a diode event of a Newton solve");
                    }
                    sens.state_event(tau, graph.branch_name(e.branch_id), e.from, e.to,
                                     (!ev.new_state && e.var >= 0) ? e.var : Index{-1},
                                     mask_first, cache.resident_factor(mask_first, for_dt),
                                     Real{2} / for_dt, src);
                }
            }
        };
        // split_step is a closure built on every split step, with the
        // sensitivities on or off: each by-reference capture is a load and a
        // store per step, so the event paths reach the recurrence through
        // this ONE reference (the group-6 cost gate, DECISIONS 16).
        const struct {
            bool on;
            step::StageSensitivity& s;
            decltype(sens_begin)& begin;
            decltype(sens_stage)& stage;
            decltype(sens_gate_event)& gate_event;
            decltype(sens_diode_events)& diode_events;
        } sx{sens.on(), sens, sens_begin, sens_stage, sens_gate_event, sens_diode_events};
        // The mode of a LOCATED event (restart stage 1 at a gate edge, a
        // scheduled or located event at t_from) is decided before the
        // stage's Newton solve: the diode set consistent with the
        // pre-event state x0 in the new switch configuration, read on the
        // stage's linearization at x0 (one Newton iteration from x0, kept
        // nowhere), re-decided until no diode flips. Newton started in the
        // step-start bits must never be what discovers the commutation:
        // with an inductor cut it drives a node to kilovolts and stalls on
        // the round-off there (release finding F-3-7: mid_a −1.26e4 V,
        // abort). A linear stage IS its linearization: its event
        // iteration's solves are that decision (one memoryless solve per
        // mask, none kept from a stale mask), and 'lcp' decides at its
        // site first. Returns the flips (rule-on path only; out of line, built
        // once per run: a per-step closure cost the default path +1 %).
        auto decide_edge_mode = [&](Real at_t, Real for_dt) PULSIM_STEP_DRIVER -> Size {
            if (!nl_refresh_effective || !has_diodes) return 0;
            step_ctx = step::BackwardEuler::on_factor(for_dt).ctx();   // as the stage
            step_ctx.t = at_t;
            const auto gates_e = gate_at(gate_t);
            Size flips = 0;
            while (flips < max_iters) {
                const auto m = diodes.combine(gates_e);
                const pwl::PwlSegment* seg_e = &retry_seg;
                if (for_dt == opts.dt && !newton_only_rows) {
                    seg_e = &nominal_segment(m);
                } else {
                    pwl::assemble_segment(graph, pool, m, for_dt, retry_seg.J,
                                          retry_seg.b_constant);
                    sparse::compress_in_place(retry_seg.J);
                    retry_seg.state_size = state_size;
                }
                edge_xl = x;
                try {   // one iteration: it throws "not converged", edge_xl kept
                    pwl::solve_with_newton_in_place(*seg_e, nl_refresh_effective, graph,
                                                    pool, edge_xl, b_extra, 1,
                                                    opts.tol_newton_dx, opts.tol_newton_res);
                } catch (const std::runtime_error&) {
                }
                if (!edge_xl.allFinite() || !events::redecide(diodes, edge_xl)) break;
                ++flips;
            }
            return flips;
        };

        // Consistent initialisation (release group A, F-5-1 / F-3-3): an
        // SDIRK step may only start from a consistent state. Stage 1 (BE of
        // γh from x0) relaxes every mode faster than γh; when x0 is
        // inconsistent with the new mode (an inductor's path cut) its
        // increment x1 − x0 is O(1), not O(h), and x0* reverses it: stage 2
        // solves from −√2·x0 (100 A -> −141 A) and a device that conducts
        // there keeps it.
        //
        // The probe: x2 and x3, two more BE stages of γh from x1 (stage 1's
        // bits, same factor, discarded). Per state (capacitor voltage,
        // inductor current) the differences Δk = xk − x(k−1) give
        //   d  = Δ1 − Δ2   (x0+ = x0 + d = 2·x1 − x2: the stages' line
        //                   extrapolated back to the event),
        //   d' = Δ2 − Δ3   (the same estimate one stage later).
        // A linear mode of rate λ (a = 1/(1 + |λ|γh)) holding a jump J gives
        // d = −J(1 − a)², Δ2 = −J·a(1 − a), d' = a·d; a smooth trajectory
        // (x0 on the new mode's slow manifold, the stages O(γh·ẋ)) gives
        // d ≈ d' ≈ −(γh)²ẍ. ONE rule, in both readings of the sources (frozen
        // at t1, moved; see below): the event is impulsive when some state has
        //   |d| > max(|Δ2|/γ, (1 + 1/γ)·|d'|) + 1e-8·(1 + |x0| + |x1|)
        //                                      [+ G_off·(1 + V_max)].
        // For one mode each term is the same threshold, a < γ/(1 + γ), i.e.
        // τ < γ²h ≈ 0.09·h: a mode the grid does not resolve. On a smooth
        // trajectory |Δ2|/γ ≈ h·|ẋ| covers an inflection (d, d' = O(h³)) and
        // (1 + 1/γ)|d'| ≈ 4.4·|d| an extremum (Δ2 ≈ 0). Neither reads the rate
        // before the event: an 8 A cut of 1 µH at 100 V, dt 0.1 µs, is a jump
        // although h·V/L = 10 A (reviewer A1, third session).
        // The event-time allowance δt_loc·|ẋ(0−)|: d is measured from the
        // step start, so a kink of the state at an instant θ inside the
        // probe (a diode turning an inductor off at zero current) gives
        // |d| ≤ θ·|ẋ(0−)|, ẋ(0−) the state's rate before the event (history
        // slot i/C, v/L; L⁻¹v for coupled windings; the law's L(i), C(v)).
        // δt_loc is how far the event can be from the step start: 0 at a
        // TIME event (a gate edge, a source jump at its scheduled instant,
        // the first step) — any jump above noise is a jump there; the split's
        // resolution `snap` at a located diode crossing; the probe window γh
        // when the event is not located (a window step, a stage-1 diode flip:
        // a crossing the previous solve had not reached lies inside it).
        // ONE number per restart, chosen by the caller (`loc`); a stage-1
        // flip widens it to γh for every state of that step (shortcut: a
        // real jump below γh·|ẋ(0−)| at a gate edge whose stage 1 also
        // flips a diode is integrated, not recorded).
        // The tolerance: 1e-8 relative with a one-unit floor is the event
        // services' noise level (`events::held_tolerance`; it holds the Newton
        // tolerance 1e-9 and the round-off). For an inductor current, plus the
        // leakage scale the same tolerance gives a conductance-form OFF
        // element: G_off (every switch's and diode's g_off) times the node
        // voltages — a current the OFF elements' leakage can carry (a buck's
        // 2.4e-8 A settling at a zero start) is the model's own noise. Then
        // the jump J = κ·d is recorded (½·JᵀM·J, κ = 1/(1 − a)²: the whole
        // jump, see below), the histories move to x0 + J and stage 1 is solved again
        // from there; else nothing of the probe is kept (the restart is the
        // plain one, bit for bit).
        // Built once per run, out of line: a closure built per step (in the
        // step body) cost the default path 62-77 instructions per step.
        auto consistent_start = [&](auto& run_event_iteration, Real t_from, Real h_s,
                                    Real dt_eq, Real loc, Size& iters_1,
                                    bool& breach_1) PULSIM_STEP_DRIVER -> bool {
            const auto& ents = history.entries();
            ip_jump = false;
            ip_moved = false;
            rl_nc0.clear();
            rl_dep_x.clear();
            rl_sl0.clear();
            if (ents.empty() &&
                hs.get<pwl::SaturableInductorHistory>().entries().empty() &&
                hs.get<pwl::NonlinearCapacitorHistory>().entries().empty()) {
                return false;
            }
            // a stage-1 diode flip (iters_1 > 1: the event iteration changed
            // the bits; breach_1 alone is an unsettled iteration)
            const Real dt_loc = (iters_1 > 1 || breach_1) ? std::max(loc, h_s) : loc;
            ip_x1 = x;
            ip_b = b_extra;
            step::snapshot_into(history, hs, ip_hist, ip_hs);
            const Size iters_s = iters;
            const bool flipped_s = flipped;
            const bool cycle_s = mask_cycle;
            const auto stamped_s = stamped_mask;
            // ip_h0: the history part of b_extra at x0 (swapped for x0+'s
            // below, if impulsive).
            history.compute_b_extra_backward_euler(h_s, ip_h0);
            // Two readings of the built-in sources over the probe: frozen at
            // t1, and moved with the stages (t2 = t + 2γh, t3 = t + 3γh). A
            // jump is the state's, so it must show in both: a fast mode
            // TRACKING its forcing extrapolates to x0 + γh·ẋ with the sources
            // frozen (a rectifier's capacitor from rest: 43 V), and a device
            // whose region the moving source changes inside the probe makes
            // a kink with them moved (a MOSFET gate ramp crossing V_th: 1.9e-4
            // V on 1 µF); each is the other reading's x0. A source's own jump
            // inside the probe is read at its left limit (else the probe reads
            // it as the event's: 0.96 mJ on a rectifier from a 1 V pulse's
            // edge at 0.4·dt); a second one leaves the frozen reading only. A
            // b_extra_fn keeps its t1 value (called once per solve time;
            // shortcut: a fast mode it alone forces reads its tracking as a
            // jump in both readings, upgrade if a controller ever drives one).
            const Real t1 = t_from + h_s, t2 = t1 + h_s, t3 = t2 + h_s;
            const Real src1 = t1 == src_end_at ? src_end_read : t1;
            auto src_move = [&](Vector& out, Real at_t) {   // sources(at_t) − sources(t1)
                Real at = at_t;
                if (has_jump_sources && sources::source_jump_in(pool, graph, t1, at_t)) {
                    // (source_jump_in counts [t1, at_t))
                    at = sources::next_source_jump_after(pool, graph, t1);
                    if (!(at < at_t) ||
                        sources::next_source_jump_after(pool, graph, at) <= at_t) {
                        return false;
                    }
                }
                out.setZero(ip_b.size());
                add_source_rows(out, at, at_t, left_by);
                ip_h1.setZero(ip_b.size());
                add_source_rows(ip_h1, src1, t1, left_by);
                out -= ip_h1;
                if (has_iwave) {
                    // shortcut: past a jump in (t1, at_t] the current waveforms
                    // are read at its left limit, not at at_t; upgrade if a
                    // fast mode tracks one.
                    sources::compute_current_waveform_b_extra(pool, graph, at,
                                                              be_iwave, left_by);
                    out += be_iwave;
                    sources::compute_current_waveform_b_extra(pool, graph, src1,
                                                              be_iwave, left_by);
                    out -= be_iwave;
                }
                return true;
            };
            const bool moved = src_move(ip_s2, t2) && src_move(ip_s3, t3) &&
                               ip_s3.lpNorm<Eigen::Infinity>() > Real{0};
            bool solved = true;
            // One more BE stage of γh, from y (the histories at y), the
            // sources shifted by *s (nullptr: frozen); x: its start, its end.
            auto stage_from = [&](const Vector& y, const Vector* s, Real at) {
                step::restore(history, hs, ip_hist, ip_hs);
                history.extrapolate_previous(y, Real{1});
                hs.extrapolate_previous(y, Real{1});
                history.compute_b_extra_backward_euler(h_s, ip_h1);
                b_extra = ip_b - ip_h0 + ip_h1;
                if (s != nullptr) b_extra += *s;
                if (!solved) return;
                try {
                    run_event_iteration(at, dt_eq, true, true);
                } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                    throw;
                } catch (const std::runtime_error&) {
                    solved = false;   // no probe: the plain restart
                }
            };
            stage_from(ip_x1, nullptr, t1);
            ip_x2 = x;   // the frozen reading: x2, x3
            stage_from(ip_x2, nullptr, t1);
            ip_x3f = x;
            if (moved) {
                x = ip_x1;
                stage_from(ip_x1, &ip_s2, t2);
                ip_x2m = x;
                stage_from(ip_x2m, &ip_s3, t3);   // x: the moved reading's x3
            } else {
                ip_x2m = ip_x2;
            }
            step::restore(history, hs, ip_hist, ip_hs);
            event_rl_x = Real{2} * ip_x1 - ip_x2m;
            ip_x3m = x;   // the reading's x3 (the series' later stages reuse x)
            // Per state (C or L): the jump x0+ − x0, the rule's bound;
            // energy ½ΔᵀMΔ with the transformers' mutuals.
            auto state = [&](const pwl::HistoryEntry& e, const Vector& v) {
                return e.kind == pwl::DevicePool::StoredKind::Capacitor
                    ? stamping::read_node_voltage(v, e.from) -
                          stamping::read_node_voltage(v, e.to)
                    : v[e.inductor_branch_var_id];
            };
            auto is_cap = [](const pwl::HistoryEntry& e) {
                return e.kind == pwl::DevicePool::StoredKind::Capacitor;
            };
            const Size ne = ents.size();
            const auto& sats = hs.get<pwl::SaturableInductorHistory>().entries();
            const auto& ncs = hs.get<pwl::NonlinearCapacitorHistory>().entries();
            const Size nn = ncs.size();
            ip_d.resize(2 * ne);
            ip_nc.assign(3 * ncs.size(), Real{0});
            // ẋ(0−) per state, from the history slots at x0 (restored above)
            ip_r.resize(ne);
            for (Size k = 0; k < ne; ++k) {
                const auto& e = ents[k];
                ip_r[k] = (is_cap(e) ? e.i_prev : e.v_prev) / e.C_or_L;
            }
            for (const auto& tc : history.transformer_couplings()) {
                const Real m =
                    models::TwoWindingTransformer::mutual_inductance(tc.params);
                const auto& ep = ents[tc.p_entry_idx];
                const auto& es = ents[tc.s_entry_idx];
                const Real det = ep.C_or_L * es.C_or_L - m * m;
                if (!(det > Real{0})) continue;   // k = 1: keep v/L
                ip_r[tc.p_entry_idx] = (es.C_or_L * ep.v_prev - m * es.v_prev) / det;
                ip_r[tc.s_entry_idx] = (ep.C_or_L * es.v_prev - m * ep.v_prev) / det;
            }
            Real v_max = 0;   // node voltages after the event
            for (Index n = 0; n < graph.num_nodes(); ++n) {
                v_max = std::max({v_max, std::abs(ip_x1[n]), std::abs(x[n])});
            }
            const Real tol_i = g_leak * (Real{1} + v_max);
            // One reading of one state (s0 = x0, s1..s3 the stages): the jump
            // d = (s1 − s0) − (s2 − s1) against the rate after the event and
            // the next difference d' = (s2 − s1) − (s3 − s2) (see above).
            auto jumps = [&](Real s0, Real s1, Real s2, Real s3, Real tol) {
                return std::abs(Real{2} * s1 - s2 - s0) >
                       std::max(std::abs(s2 - s1) / kGamma,
                                (Real{1} + Real{1} / kGamma) *
                                    std::abs(Real{2} * s2 - s1 - s3)) + tol;
            };
            // A saturable core's current and a nonlinear capacitor's charge
            // are states too, under the same rule (ẋ(0−) from the law).
            auto law_jumps = [&](Real s0, Real s1, auto&& st, Real tol) {
                return solved && jumps(s0, s1, st(ip_x2m), st(ip_x3m), tol) &&
                       jumps(s0, s1, st(ip_x2), st(ip_x3f), tol);
            };
            auto ncv = [&](const pwl::NonlinearCapacitorHistory::Entry& e, const Vector& y) {
                return stamping::read_node_voltage(y, e.from) -
                       stamping::read_node_voltage(y, e.to);
            };
            auto ncq = [&](const pwl::NonlinearCapacitorHistory::Entry& e, const Vector& y) {
                return models::NonlinearCapacitor::charge(e.params, ncv(e, y));
            };
            // A saturable core (atan, table, hysteretic) in FLUX, the quantity
            // an event conserves (the twin of the charge above): in current, a
            // core leaving saturation at a constant voltage (Δi 10 A, then
            // 0.01 A) reads as a jump. ip_sl per device: (i0+, λ0+, flag).
            const Size nsat = sats.size();
            ip_sl.assign(3 * nsat, Real{0});
            auto slam = [&](const pwl::SaturableInductorHistory::Entry& e, const Vector& y) {
                return pwl::SaturableInductorHistory::flux_and_inductance(e, y[e.branch_var_id]).first;
            };
            // the stages' noise in flux: the current's floor times L(i1)
            auto snoise = [&](const pwl::SaturableInductorHistory::Entry& e, const Vector& y1) {
                const Real i1 = y1[e.branch_var_id];
                return std::abs(pwl::SaturableInductorHistory::flux_and_inductance(e, i1).second) *
                       (Real{1e-8} * (Real{1} + std::abs(e.i_L_old) + std::abs(i1)) + tol_i);
            };
            // The event-time allowance of state k: δt_loc·|ẋ(0−)|, or the
            // motion over one stage of the mode the event started in (ip_m1,
            // below) when stage 1 turned a diode: 0 until measured.
            ip_m1.assign(ne + nn + sats.size(), Real{0});
            auto allow = [&](Real r0, Size k) {
                return std::max(dt_loc * std::abs(r0), ip_m1[k]);
            };
            // The rule's test per state (linear entries: ip_d[2k + 1]; nonlinear
            // capacitors in CHARGE: ip_nc[3k + 2]).
            auto detect = [&]() {
                bool any = false;
                for (Size k = 0; k < ne; ++k) {
                    const auto& e = ents[k];
                    const Real s0 = is_cap(e) ? e.v_prev : e.i_prev;
                    const Real s1 = state(e, ip_x1);
                    const Real tol = Real{1e-8} * (Real{1} + std::abs(s0) + std::abs(s1)) +
                                     (is_cap(e) ? Real{0} : tol_i) + allow(ip_r[k], k);
                    const bool jk = jumps(s0, s1, state(e, ip_x2m), state(e, ip_x3m), tol) &&
                                    jumps(s0, s1, state(e, ip_x2), state(e, ip_x3f), tol);
                    ip_d[2 * k + 1] = jk ? Real{1} : Real{0};
                    any = any || jk;
                }
                // A nonlinear capacitor (F-S-6): tested in CHARGE, the quantity
                // the event conserves (in voltage a smooth constant-current
                // charge through a law whose C changes 100x reads as a jump);
                // the noise floor is the voltage one times C(v1), the allowance
                // δt_loc·|i(0−)|.
                for (Size k = 0; k < nn; ++k) {
                    const auto& e = ncs[k];
                    const Real v1 = ncv(e, ip_x1);
                    const Real tol = models::NonlinearCapacitor::capacitance(e.params, v1) * Real{1e-8} *
                                         (Real{1} + std::abs(e.v_prev) + std::abs(v1)) +
                                     allow(e.i_prev, ne + k);
                    const bool jk = law_jumps(e.q_prev, ncq(e, ip_x1),
                                              [&](const Vector& y) { return ncq(e, y); }, tol);
                    ip_nc[3 * k + 2] = jk ? Real{1} : Real{0};
                    any = any || jk;
                }
                // a saturable core: in flux, the allowance δt_loc·|v_L(0−)|
                for (Size k = 0; k < nsat; ++k) {
                    const auto& e = sats[k];
                    const Real tol = snoise(e, ip_x1) + allow(e.V_L_old, ne + nn + k);
                    const bool jk = law_jumps(e.lambda_old, slam(e, ip_x1),
                                              [&](const Vector& y) { return slam(e, y); }, tol);
                    ip_sl[3 * k + 2] = jk ? Real{1} : Real{0};
                    any = any || jk;
                }
                return any;
            };
            bool jumped = detect();
            // A diode stage 1 turned (its bits differ from the event start's):
            // a state that moved at a finite rate after the event and stopped
            // when that diode turned, at θ inside stage 1, makes |d| ≤ θ·|ẋ(0+)|
            // with ẋ(0+) the rate in the mode the event STARTED in — the gate
            // mask after the event with the diode bits before stage 1 (a current
            // source moving a node 0.7 V in 80 ps until a diode clamps it: a
            // state event the step does not locate, not a jump; ẋ(0−) = 0 there,
            // the switch held the node). That rate is the start mode's own probe
            // after its fast modes: three single stages from x0 in it, Δ2 = γh·ẋ where its own
            // test finds no jump.
            // A real jump is a fast mode of the start mode and leaves Δ2 at the
            // slow rate (a capacitor dumped through a switch whose stage 1 also
            // turns a diode is still recorded). Measured only when the test
            // fired (impulsive candidates); a failed solve keeps the old bound.
            if (jumped && solved && has_diodes &&
                rs_bits0.size() == diodes.entries().size()) {
                diodes.snapshot_on_bits_into(ip_bits1);
                if (ip_bits1 != rs_bits0) {
                    const auto stamped_1 = stamped_mask;
                    const bool cycle_1 = mask_cycle;
                    diodes.restore_on_bits(rs_bits0);
                    step::restore(history, hs, ip_hist, ip_hs);
                    b_extra = ip_b;
                    x = ip_x1;
                    bool m1_ok = true, y3_ok = false;
                    try {
                        run_event_iteration(t1, dt_eq, true, true);
                    } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                        throw;
                    } catch (const std::runtime_error&) {
                        m1_ok = false;
                    }
                    if (m1_ok) {
                        ip_y1 = x;
                        stage_from(ip_y1, nullptr, t1);
                        m1_ok = solved;
                        ip_y2 = x;
                        if (m1_ok) stage_from(ip_y2, nullptr, t1);   // x: y3
                        y3_ok = m1_ok && solved;   // may leave the law's range: below
                        if (!y3_ok) x = ip_y2;
                        solved = true;
                    }
                    step::restore(history, hs, ip_hist, ip_hs);   // x0's histories: s0 below
                    if (m1_ok) {
                        // per state, over one stage of the start mode: a smooth move
                        // (no jump of that mode by the rule's d-test, |Δ1 − Δ2| ≤ |Δ2|/γ
                        // + noise) is bounded by |Δ2|. A jump of that mode (Δk = s +
                        // f·ρ^(k−1): one fast mode of factor ρ, tail f, slow motion s)
                        // only by its slow rate, Aitken's s = (Δ1Δ3 − Δ2²)/(Δ1 + Δ3 −
                        // 2Δ2), held to ≤ |Δ2|: |Δ2| there holds the fast tail
                        // c·ρ(1 − ρ), and at ρ ≈ 0.18..0.23 it hid a real 4.93 µJ share
                        // (review). (Not the full d'-test: the start mode's later
                        // stages leave the state's range — a 14 V stage of the clamp's
                        // ramp reaches a junction law's floor at stage 3, whose
                        // Newton solve then fails: no third stage, no Aitken.)
                        auto m1 = [&](Real s0, Real a1, Real a2, Real a3, Real noise) {
                            const Real d1 = a1 - s0, d2 = a2 - a1, d3 = a3 - a2;
                            if (!(std::abs(d1 - d2) > std::abs(d2) / kGamma + noise)) return std::abs(d2);
                            if (!y3_ok) return Real{0};   // a jump of the start mode, no s
                            const Real den = d1 + d3 - Real{2} * d2;
                            const Real sl = std::abs(den) > Real{1e-12} * (std::abs(d1) + std::abs(d2) + std::abs(d3))
                                ? (d1 * d3 - d2 * d2) / den
                                : d3;
                            return std::min(std::abs(sl), std::abs(d2));
                        };
                        for (Size k = 0; k < ne; ++k) {
                            const auto& e = ents[k];
                            const Real s0 = is_cap(e) ? e.v_prev : e.i_prev;
                            const Real a1 = state(e, ip_y1);
                            ip_m1[k] = m1(s0, a1, state(e, ip_y2), state(e, x),
                                          Real{1e-8} * (Real{1} + std::abs(s0) + std::abs(a1)) +
                                              (is_cap(e) ? Real{0} : tol_i));
                        }
                        for (Size k = 0; k < nn; ++k) {
                            const auto& e = ncs[k];
                            const Real v1 = ncv(e, ip_y1);
                            ip_m1[ne + k] = m1(e.q_prev, ncq(e, ip_y1), ncq(e, ip_y2), ncq(e, x),
                                               models::NonlinearCapacitor::capacitance(e.params, v1) *
                                                   Real{1e-8} * (Real{1} + std::abs(e.v_prev) + std::abs(v1)));
                        }
                        for (Size k = 0; k < nsat; ++k) {
                            const auto& e = sats[k];
                            ip_m1[ne + nn + k] = m1(e.lambda_old, slam(e, ip_y1), slam(e, ip_y2),
                                                    slam(e, x), snoise(e, ip_y1));
                        }
                    }
                    diodes.restore_on_bits(ip_bits1);
                    step::restore(history, hs, ip_hist, ip_hs);
                    stamped_mask = stamped_1;
                    mask_cycle = cycle_1;
                    if (m1_ok) {
                        // A state over the old bound but within the start mode's
                        // motion MOVED (flag 2): it is carried to x0 + J like a
                        // jump — else stage 2's pseudo-state carries the move
                        // (−0.7107 V for the clamp's −0.71) — but recorded by
                        // nobody (the source did the work; no loss, no right
                        // limit at t_from: the state is continuous there).
                        ip_f0.resize(ne + nn + nsat);
                        for (Size k = 0; k < ne; ++k) ip_f0[k] = ip_d[2 * k + 1];
                        for (Size k = 0; k < nn; ++k) ip_f0[ne + k] = ip_nc[3 * k + 2];
                        for (Size k = 0; k < nsat; ++k) ip_f0[ne + nn + k] = ip_sl[3 * k + 2];
                        jumped = detect();
                        for (Size k = 0; k < nsat; ++k) {
                            if (ip_f0[ne + nn + k] != Real{0} && ip_sl[3 * k + 2] == Real{0}) {
                                ip_sl[3 * k + 2] = Real{2};
                            }
                        }
                        for (Size k = 0; k < ne; ++k) {
                            if (ip_f0[k] != Real{0} && ip_d[2 * k + 1] == Real{0}) ip_d[2 * k + 1] = Real{2};
                        }
                        for (Size k = 0; k < nn; ++k) {
                            if (ip_f0[ne + k] != Real{0} && ip_nc[3 * k + 2] == Real{0}) {
                                ip_nc[3 * k + 2] = Real{2};
                            }
                        }
                    }
                }
            }
            // jumped: a recorded jump (flag 1); moving: any state carried
            bool moving = jumped;
            for (Size k = 0; k < ne; ++k) moving = moving || ip_d[2 * k + 1] != Real{0};
            for (Size k = 0; k < nn; ++k) moving = moving || ip_nc[3 * k + 2] != Real{0};
            for (Size k = 0; k < nsat; ++k) moving = moving || ip_sl[3 * k + 2] != Real{0};
            auto refuse_jump = [&](Index br, Real dp, const char* kind, const char* unit,
                                   const char* why) {
                const std::string name(graph.branch_name(br));
                throw UnsupportedError(
                    "pwl", "impulse", name, kind,
                    std::format("discontinuity_restart: at t = {} the event "
                                "moves {} '{}' by {:.6g} {} in an instant; {}; "
                                "run this circuit without discontinuity_restart.",
                                t_from, kind, name, dp, unit, why));
            };
            // A saturable core's jump is carried in FLUX (λ0+ = λ0 + J_λ, i0+ from
            // the law's inverse): the stateless laws. The hysteretic law's λ(i)
            // depends on its (H, M) path and has no inverse here: refused by name.
            for (Size k = 0; k < nsat; ++k) {
                const auto& e = sats[k];
                if (ip_sl[3 * k + 2] != Real{0} && e.params.ja) {
                    refuse_jump(e.branch_id,
                                Real{2} * ip_x1[e.branch_var_id] - ip_x2m[e.branch_var_id] - e.i_L_old,
                                "saturable_inductor", "A",
                                "the consistent restart carries a saturable core's jump in "
                                "flux on the atan, table and gapped-core laws, not on the "
                                "hysteretic (Jiles-Atherton) law");
                }
            }
            // A jumping state's whole jump, one mode at a time: J = d/(1 − a)²,
            // a = d'/d (moved reading; < γ/(1 + γ) by the rule): per mode
            // d = −J(1 − a)², d' = a·d. With d alone the a(2 − a)·J left in x0+
            // came back reversed through stage 2's pseudo-state (0.955 A of a
            // 100 A IGBT cut held by its FWD, a = 0.0034: τ = L·g_off = 1 ps at
            // dt 1 ns). Other states keep d (O(h²) on a smooth trajectory). A
            // nonlinear capacitor's in charge: q0+ = q0 + J_q (the stages are
            // charge-form solves of the law). Two or more jumping states: the
            // vector series below replaces these.
            ip_js.clear();
            for (Size k = 0; k < ne; ++k) {
                const auto& e = ents[k];
                const Real s0 = is_cap(e) ? e.v_prev : e.i_prev;
                const Real s1 = state(e, ip_x1), s2 = state(e, ip_x2m);
                const Real dp = Real{2} * s1 - s2 - s0;
                Real jd = dp;
                if (ip_d[2 * k + 1] != Real{0}) {
                    const Real a = std::clamp((Real{2} * s2 - s1 - state(e, ip_x3m)) / dp, Real{0},
                                              kGamma / (Real{1} + kGamma));
                    jd = dp / ((Real{1} - a) * (Real{1} - a));
                    ip_js.push_back(k);
                }
                ip_d[2 * k] = jd;
            }
            for (Size k = 0; k < nn; ++k) {
                if (ip_nc[3 * k + 2] == Real{0}) continue;
                const auto& e = ncs[k];
                const Real q1 = ncq(e, ip_x1), q2 = ncq(e, ip_x2m);
                const Real dq = Real{2} * q1 - q2 - e.q_prev;
                const Real a = dq != Real{0}
                    ? std::clamp((Real{2} * q2 - q1 - ncq(e, ip_x3m)) / dq, Real{0},
                                 kGamma / (Real{1} + kGamma))
                    : Real{0};
                ip_nc[3 * k + 1] = e.q_prev + dq / ((Real{1} - a) * (Real{1} - a));
                ip_js.push_back(ne + k);
            }
            for (Size k = 0; k < nsat; ++k) {
                if (ip_sl[3 * k + 2] == Real{0}) continue;
                const auto& e = sats[k];
                const Real l1 = slam(e, ip_x1), l2 = slam(e, ip_x2m);
                const Real dl = Real{2} * l1 - l2 - e.lambda_old;
                const Real a = dl != Real{0}
                    ? std::clamp((Real{2} * l2 - l1 - slam(e, ip_x3m)) / dl, Real{0},
                                 kGamma / (Real{1} + kGamma))
                    : Real{0};
                ip_sl[3 * k + 1] = e.lambda_old + dl / ((Real{1} - a) * (Real{1} - a));
                ip_js.push_back(ne + nn + k);
            }
            // A mode with structural dependent states (ideal elements) keeps
            // the SDIRK x3 projection and the tracker's record (one service
            // owns an ideal element's re-initialisation; both would record
            // the same jump twice).
            const bool structural = structure.active() &&
                                    !structure.check(stamped_mask, true).dependent.empty();
            // The whole jump as a VECTOR (derivation in solver/jump_series.hpp and
            // fixes/jump-conservation.md): the stages' second differences D_k
            // (D1 = d, D2 = d') are a Krylov sequence of the stage map, J = [D1 …
            // Dm]·(I − H)⁻²·e1. The per-state a is that series for one mode per
            // state; two modes of different a in one floating island (C1 –1 mΩ–
            // C2 –R2– C3) took a different multiple per state and left the
            // island's charge off by 3.4 %. J lies in span{D_k} with ONE
            // coefficient vector for every state, so each charge (flux) the stages
            // conserve J conserves to rounding — over ALL states: a state of the
            // island the test does not flag (a mode the grid resolves) takes the
            // same combination (ladders of five or more capacitors lost 8 % of
            // their charge when only the flagged states did — review). A Ritz value
            // outside the rule's fast range (ρ > γ/(1 + γ): a mode the grid
            // integrates, slow content) keeps factor 1, as the rule's d keeps a
            // state that does not jump. m modes take m + 2 stages: m grows while
            // the fit of D_{m+1} is above the stages' noise (the rule's floor),
            // to m = 6. Fallbacks, in order (each conserves less): the fit over the
            // flagged states only (more than six modes over all states); the
            // per-state a (a stage failed, a source jump inside the extra stages,
            // or more than six modes among the flagged states). shortcut: those
            // fallbacks do not conserve an island's charge that involves states
            // outside the fit; upgrade if a circuit shows them (more than six
            // distinct modes at one event).
            ip_xs.clear();
            bool stages_ok = true;
            // x0 per entry, read now: each extra stage moves the histories
            ip_s0.resize(ne + nn + nsat);
            for (Size k = 0; k < ne; ++k) ip_s0[k] = is_cap(ents[k]) ? ents[k].v_prev : ents[k].i_prev;
            for (Size k = 0; k < nn; ++k) ip_s0[ne + k] = ncs[k].q_prev;
            for (Size k = 0; k < nsat; ++k) ip_s0[ne + nn + k] = sats[k].lambda_old;
            ip_s0v.resize(nn);
            for (Size k = 0; k < nn; ++k) ip_s0v[k] = ncs[k].v_prev;
            // the stage values of entry k (k < ne: linear; then the nonlinear
            // capacitors' charges; then the saturable cores' fluxes) at stage j
            // (0 = x0, 1..3 the probe, 4.. ip_xs)
            auto sval = [&](Size k, Index j) -> Real {
                if (j == 0) return ip_s0[k];
                const Vector& y = j == 1 ? ip_x1 : j == 2 ? ip_x2m : j == 3 ? ip_x3m
                                                                       : ip_xs[static_cast<Size>(j - 4)];
                return k < ne ? state(ents[k], y)
                       : k < ne + nn ? ncq(ncs[k - ne], y) : slam(sats[k - ne - nn], y);
            };
            auto stage_to = [&](Index j) {   // stages up to j exist (ip_xs)
                while (stages_ok && 3 + static_cast<Index>(ip_xs.size()) < j) {
                    const Index have = 3 + static_cast<Index>(ip_xs.size());
                    const Vector& prev = have == 3 ? ip_x3m : ip_xs.back();
                    const Real tk = t1 + static_cast<Real>(have) * h_s;
                    if (moved && !src_move(ip_sk, tk)) {
                        stages_ok = false;
                        break;
                    }
                    ip_xk = prev;
                    x = ip_xk;
                    stage_from(ip_xk, moved ? &ip_sk : nullptr, moved ? tk : t1);
                    if (!solved) {
                        solved = true;
                        stages_ok = false;
                        break;
                    }
                    ip_xs.push_back(x);
                }
                return stages_ok;
            };
            // the series over the entries in ip_js; writes J into ip_d / ip_nc
            auto series = [&](bool all) -> bool {
                const Size ns = ip_js.size();
                const auto nsi = static_cast<Index>(ns);
                ip_w.resize(nsi);
                Real thr2 = 0;
                for (Size i = 0; i < ns; ++i) {
                    const Size k = ip_js[i];
                    Real w = 0, noise = 0;
                    if (k < ne) {
                        const auto& e = ents[k];
                        w = std::sqrt(e.C_or_L);
                        noise = Real{1e-8} * (Real{1} + std::abs(sval(k, 0)) + std::abs(sval(k, 1))) +
                                (is_cap(e) ? Real{0} : tol_i);
                    } else if (k < ne + nn) {
                        const auto& e = ncs[k - ne];
                        const Real v1 = ncv(e, ip_x1);
                        const Real c1 = models::NonlinearCapacitor::capacitance(e.params, v1);
                        w = Real{1} / std::sqrt(c1);
                        noise = c1 * Real{1e-8} * (Real{1} + std::abs(ip_s0v[k - ne]) + std::abs(v1));
                    } else {   // a flux: energy weight 1/√L(i1)
                        const auto& e = sats[k - ne - nn];
                        const Real l1 = std::abs(pwl::SaturableInductorHistory::flux_and_inductance(
                            e, ip_x1[e.branch_var_id]).second);
                        w = Real{1} / std::sqrt(l1);
                        noise = snoise(e, ip_x1);
                    }
                    ip_w[static_cast<Index>(i)] = w;
                    thr2 += (w * noise) * (w * noise);
                }
                auto second = [&](Index j, Vector& out) {   // D_{j+1}, unweighted
                    out.resize(nsi);
                    for (Size i = 0; i < ns; ++i) {
                        const Size k = ip_js[i];
                        out[static_cast<Index>(i)] =
                            Real{2} * sval(k, j + 1) - sval(k, j) - sval(k, j + 2);
                    }
                };
                const Index mmax = std::min<Index>(nsi, 6);
                Index m = 1;
                Real r2 = 0;
                for (;; ++m) {
                    if (!stage_to(m + 2)) return false;
                    ip_Dw.resize(nsi, m + 1);
                    for (Index j = 0; j <= m; ++j) {
                        second(j, ip_sk);
                        ip_Dw.col(j) = ip_sk.cwiseProduct(ip_w);
                    }
                    r2 = solver::jump_series_fit(ip_Dw, m, ip_c);
                    if (!(r2 > thr2) || m == mmax) break;
                }
                if (r2 > thr2 && nsi > mmax) return false;
                if (!solver::jump_series_coeffs(ip_c, kGamma / (Real{1} + kGamma), ip_g)) return false;
                Vector jv = Vector::Zero(nsi);
                for (Index j = 0; j < m; ++j) {
                    second(j, ip_sk);
                    jv += ip_g[j] * ip_sk;
                }
                if (!jv.allFinite()) return false;
                for (Size i = 0; i < ns; ++i) {
                    const Size k = ip_js[i];
                    const Real ji = jv[static_cast<Index>(i)];
                    if (k < ne) {
                        ip_d[2 * k] = ji;
                    } else if (k < ne + nn) {
                        ip_nc[3 * (k - ne) + 1] = ip_s0[k] + ji;
                        // carried by the series without the rule's flag: moved, unrecorded
                        if (all && ip_nc[3 * (k - ne) + 2] == Real{0}) ip_nc[3 * (k - ne) + 2] = Real{3};
                    } else {
                        const Size q = k - ne - nn;
                        ip_sl[3 * q + 1] = ip_s0[k] + ji;
                        if (all && ip_sl[3 * q + 2] == Real{0}) ip_sl[3 * q + 2] = Real{3};
                    }
                }
                return true;
            };
            if (solved && moving && ne + nn + nsat >= 2) {
                ip_f0.assign(ip_js.begin(), ip_js.end());   // the flagged entries
                ip_js.clear();   // every state but a hysteretic core (no inverse: not carried)
                for (Size k = 0; k < ne + nn + nsat; ++k) {
                    if (k < ne + nn || !sats[k - ne - nn].params.ja) ip_js.push_back(k);
                }
                if (!series(true) && ip_f0.size() >= 2) {
                    ip_js.resize(ip_f0.size());
                    for (Size i = 0; i < ip_f0.size(); ++i) ip_js[i] = static_cast<Size>(ip_f0[i]);
                    (void)series(false);
                }
                step::restore(history, hs, ip_hist, ip_hs);
            }
            Real e_jump = 0;
            Size dom = 0;
            Real dom_e = -1;
            for (Size k = 0; k < ne; ++k) {
                const Real jd = ip_d[2 * k];
                const Real ek = Real{0.5} * ents[k].C_or_L * jd * jd;
                if (ip_d[2 * k + 1] == Real{2}) continue;   // a move: no loss
                e_jump += ek;
                if (ip_d[2 * k + 1] == Real{1} && ek > dom_e) {
                    dom_e = ek;
                    dom = k;
                }
            }
            for (const auto& tc : history.transformer_couplings()) {
                const Real m =
                    models::TwoWindingTransformer::mutual_inductance(tc.params);
                if (ip_d[2 * tc.p_entry_idx + 1] == Real{2} || ip_d[2 * tc.s_entry_idx + 1] == Real{2}) continue;
                e_jump += m * ip_d[2 * tc.p_entry_idx] * ip_d[2 * tc.s_entry_idx];
            }
            // A nonlinear capacitor that jumps: v0+ = Q⁻¹(q0+), its energy from
            // the law (jump_energy: ∫(Q − Q(v0)) dv), never ½·C·ΔV² with a
            // frozen C. ip_nc per device: (v0+, q0+, jumped).
            bool nc_jumped = false;
            for (Size k = 0; k < nn; ++k) {
                if (ip_nc[3 * k + 2] == Real{0}) continue;
                const auto& e = ncs[k];
                const Real qp = ip_nc[3 * k + 1];
                const Real vp = models::NonlinearCapacitor::voltage_of_charge(e.params, qp, ncv(e, ip_x1));
                if (!std::isfinite(vp)) {
                    refuse_jump(e.branch_id, Real{2} * ncv(e, ip_x1) - ncv(e, ip_x2m) - e.v_prev,
                                "nonlinear_capacitor", "V",
                                "the charge after it lies outside the law's range");
                }
                ip_nc[3 * k] = vp;
                nc_jumped = true;
                if (ip_nc[3 * k + 2] == Real{2}) continue;   // a move: no loss
                // (flag 3, carried by the series: part of J like an unflagged
                // linear state — its energy is the record's too)
                const Real ek = models::NonlinearCapacitor::jump_energy(e.params, e.v_prev, vp);
                e_jump += ek;
                if (ip_nc[3 * k + 2] == Real{1} && ek > dom_e) {
                    dom_e = ek;
                    dom = ne + k;   // ≥ ne: a nonlinear capacitor
                }
            }
            // A saturable core that jumps: i0+ = λ⁻¹(λ0+), its energy from the law
            // (∫(λ − λ(i0)) di). ip_sl per device: (i0+, λ0+, flag).
            bool sl_jumped = false;
            for (Size k = 0; k < nsat; ++k) {
                if (ip_sl[3 * k + 2] == Real{0}) continue;
                const auto& e = sats[k];
                const Real ip = models::SaturableInductor::current_of_flux(
                    e.params, ip_sl[3 * k + 1], ip_x1[e.branch_var_id]);
                if (!std::isfinite(ip)) {
                    refuse_jump(e.branch_id,
                                Real{2} * ip_x1[e.branch_var_id] - ip_x2m[e.branch_var_id] - e.i_L_old,
                                "saturable_inductor", "A",
                                "the restart's estimate of the core's flux after it lies "
                                "past the law's flux limit (the atan law with L_residual = 0 "
                                "saturates at (pi/2)*L_0*I_sat; a table at its flat end), so "
                                "no current carries that flux; give the core L_residual > 0 "
                                "(or extend the table), add a snubber or damping across the "
                                "fast loop the core collapses into, or take a smaller dt");
                }
                ip_sl[3 * k] = ip;
                sl_jumped = true;
                if (ip_sl[3 * k + 2] == Real{2}) continue;   // a move: no loss
                // (flag 3: in saturation a flux move under the allowance is a
                // large current move — 49 % of a clamp's loss was missing, review)
                const Real ek = models::SaturableInductor::jump_energy(e.params, e.i_L_old, ip);
                e_jump += ek;
                if (ip_sl[3 * k + 2] == Real{1} && ek > dom_e) {
                    dom_e = ek;
                    dom = ne + nn + k;   // ≥ ne + nn: a saturable core
                }
            }
            if (structural && sl_jumped) {
                for (Size k = 0; k < nsat; ++k) {
                    if (ip_sl[3 * k + 2] != Real{0}) {
                        refuse_jump(sats[k].branch_id, ip_sl[3 * k] - sats[k].i_L_old,
                                    "saturable_inductor", "A",
                                    "an ideal element's re-initialisation carries "
                                    "linear capacitors and inductors only");
                    }
                }
            }
            if (structural && nc_jumped) {   // the tracker re-initialises linear states only
                for (Size k = 0; k < ncs.size(); ++k) {
                    if (ip_nc[3 * k + 2] != Real{0}) {
                        refuse_jump(ncs[k].branch_id, ip_nc[3 * k] - ncs[k].v_prev,
                                    "nonlinear_capacitor", "V",
                                    "an ideal element's re-initialisation carries "
                                    "linear capacitors and inductors only");
                    }
                }
            }
            // An ideal element's jump is impulsive too (item A): its J is the
            // constraint's share J_i AND the collapse J_f of any fast mode it
            // excites (C1 -S- C2 beside 1 mΩ, τ 1 ps), both modes of the stage
            // map (the share is ρ = 0). Stage 1 from x0 left J_f to the SDIRK
            // pair: R(z) ≈ (1 − c)/z of the collapse in the step's sample
            // (+0.97 % of i at dt 0.5 µs). The tracker
            // still records J_i (from the state vectors); J_f is not recorded.
            const bool impulsive = solved && moving;
            // An ideal share whose states the rule does not flag (each one's d
            // is under its slow motion: C1 -S- C2 beside an LC at a split's
            // half step) is still an ideal jump: its constraint sum K·x jumps
            // (the same test on K·x). Else the right limit pinned every state
            // to x0, which the constraint cannot meet (diverged, no row), and
            // stage 2's pseudo-state carried c·J of the share (item C).
            bool share = false;
            if (structural && solved && !jumped) {
                for (const auto& ds : structure.check(stamped_mask, true).dependent) {
                    Real k0 = 0, k1 = 0, k2 = 0, k3 = 0, kf2 = 0, kf3 = 0, sc = 0;
                    for (Size j = 0; j < ds.states.size(); ++j) {
                        for (Size k = 0; k < ne; ++k) {
                            if (ents[k].branch_id != ds.states[j]) continue;
                            const auto& e = ents[k];
                            const Real s = static_cast<Real>(ds.signs[j]);
                            const Real a0 = is_cap(e) ? e.v_prev : e.i_prev, a1 = state(e, ip_x1);
                            k0 += s * a0; k1 += s * a1;
                            k2 += s * state(e, ip_x2m); k3 += s * state(e, ip_x3m);
                            kf2 += s * state(e, ip_x2); kf3 += s * state(e, ip_x3f);
                            sc += Real{1e-8} * (Real{1} + std::abs(a0) + std::abs(a1)) +
                                  (is_cap(e) ? Real{0} : tol_i) + allow(ip_r[k], k);
                        }
                    }
                    share = share || (jumps(k0, k1, k2, k3, sc) && jumps(k0, k1, kf2, kf3, sc));
                }
            }
            ip_jump = solved && (jumped || share);   // the right limit adds J (event_right_limit)
            ip_moved = solved && (moving || share);  // the right limit starts from x0
            ip_struct = structural;
            if (!impulsive && solved && moving && L) [[unlikely]] {
                // the lines' stage 1 from x0+ (a window integral is the
                // regular part; the jump is the tracker's record); the run's
                // histories, bits and solution are put back below
                const std::vector<Real> at_x0 = ip_hist;
                for (Size k = 0; k < ne; ++k) ip_hist[2 * k + (is_cap(ents[k]) ? 0 : 1)] += ip_d[2 * k];
                history.restore(ip_hist);
                ln_v0.resize(ne);
                for (Size k = 0; k < ne; ++k) ln_v0[k] = ents[k].v_prev;
                if (has_diodes) diodes.snapshot_on_bits_into(ln_bits);
                history.compute_b_extra_backward_euler(h_s, ip_h1);
                b_extra = ip_b - ip_h0 + ip_h1;
                x = ip_x1;
                bool ln_ok = true;
                try {   // a lines-only solve never fails the run's step
                    run_event_iteration(t_from + h_s, dt_eq, true, true);
                } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                    throw;
                } catch (const std::runtime_error&) {
                    ln_ok = false;
                }
                ln_x1 = x;
                ln_pre = ln_ok && stamped_mask == stamped_s;   // same mode: else the run's own
                ip_hist = at_x0;
                history.restore(ip_hist);
                if (has_diodes) diodes.restore_on_bits(ln_bits);
            }
            if (!impulsive) {
                x = ip_x1;
                b_extra = ip_b;
                iters = iters_s;
                flipped = flipped_s;
                mask_cycle = cycle_s;
                stamped_mask = stamped_s;
                return false;
            }
            // An ideal element's jump: the tracker records the constraint's
            // share J_i at the step's commit, in the mode the step ends in; the
            // collapse J − J_i of a fast mode the jump excites is recorded there
            // too (ideal_pending), against that same mode. Moves alone (flag 2)
            // are no jump: nothing to book (review: a spurious record)
            if (structural && ip_jump) {
                IdealJump& ij = ideal_pending.emplace_back();
                ij.t = t_from;
                ij.j.resize(ne);
                ij.move.resize(ne);
                for (Size k = 0; k < ne; ++k) {
                    ij.j[k] = ip_d[2 * k];
                    ij.move[k] = ip_d[2 * k + 1] == Real{2};
                }
            }
            // The record (every non-structural device kind). dW =
            // ½·JᵀM·J > 0 by the test above (M is positive semidefinite).
            // Only a jump (flag 1) is recorded; moves alone record nothing.
            if (jumped && !structural) {   // (structural: the tracker's record)
                const bool sl = dom >= ne + nn;    // the dominant state: a saturable core
                const bool nl = dom >= ne && !sl;  // a nonlinear capacitor
                const Index d_from = sl ? sats[dom - ne - nn].from : nl ? ncs[dom - ne].from : ents[dom].from;
                const Index d_to = sl ? sats[dom - ne - nn].to : nl ? ncs[dom - ne].to : ents[dom].to;
                const bool cut = sl || (!nl && ents[dom].kind != pwl::DevicePool::StoredKind::Capacitor);
                solver::ReinitRecord rec;
                rec.t = t_from;
                rec.kind = cut ? "cut_set" : "capacitor_loop";
                for (Size k = 0; k < ne; ++k) {
                    if (ip_d[2 * k + 1] == Real{1}) {   // the rule
                        rec.states.emplace_back(graph.branch_name(ents[k].branch_id));
                    }
                }
                for (Size k = 0; k < ncs.size(); ++k) {
                    if (ip_nc[3 * k + 2] == Real{1}) {
                        rec.states.emplace_back(graph.branch_name(ncs[k].branch_id));
                    }
                }
                for (Size k = 0; k < nsat; ++k) {
                    if (ip_sl[3 * k + 2] == Real{1}) {
                        rec.states.emplace_back(graph.branch_name(sats[k].branch_id));
                    }
                }
                // the flux (V·s) or charge (C) moved on the dominant state
                rec.lambda = sl ? ip_sl[3 * (dom - ne - nn) + 1] - sats[dom - ne - nn].lambda_old
                             : nl ? ip_nc[3 * (dom - ne) + 1] - ncs[dom - ne].q_prev
                                  : ents[dom].C_or_L * ip_d[2 * dom];
                for (const auto& tc : history.transformer_couplings()) {
                    if (dom >= ne) break;
                    const Real m =
                        models::TwoWindingTransformer::mutual_inductance(tc.params);
                    if (tc.p_entry_idx == dom) {
                        rec.lambda += m * ip_d[2 * tc.s_entry_idx];
                    }
                    if (tc.s_entry_idx == dom) {
                        rec.lambda += m * ip_d[2 * tc.p_entry_idx];
                    }
                }
                rec.dW = e_jump;
                // The devices that opened: the switching / nonlinear
                // elements at the dominant state's terminals.
                Index dev0 = kInvalidIndex;
                for (Index b = 0; b < graph.num_branches(); ++b) {
                    const auto& br = graph.branch(b);
                    if (br.kind != topology::BranchKind::Switch &&
                        br.kind != topology::BranchKind::Nonlinear) {
                        continue;
                    }
                    auto touches = [&](Index n) {
                        return stamping::node_is_active(n) &&
                               (br.from == n || br.to == n);
                    };
                    if (!touches(d_from) && !touches(d_to)) continue;
                    if (dev0 == kInvalidIndex) dev0 = b;
                    rec.devices.emplace_back(graph.branch_name(b));
                }
                rec.cause = rec.devices.empty() ? std::string{} : rec.devices.front();
                if (opts.on_impulse_error) {
                    std::string sts;
                    for (const auto& s : rec.states) sts += (sts.empty() ? "" : ", ") + s;
                    throw UnsupportedError(
                        "pwl", "impulse", rec.cause,
                        dev0 != kInvalidIndex && pool.is_registered(dev0)
                            ? std::string(pwl::stored_kind_name(pool.kind_of(dev0)))
                            : std::string{},
                        std::format(
                            "on_impulse='error': at t = {} '{}' makes {} "
                            "dependent ({}); re-initialising them moves "
                            "{:.6g} {} and removes {:.6g} J.",
                            t_from, rec.cause, sts,
                            cut ? "an inductor cut-set" : "a capacitor loop",
                            rec.lambda, cut ? "V*s" : "C", rec.dW));
                }
                impulse_pending.push_back(std::move(rec));
                event_rl_t = t_from;
            }
            // The C / L histories at x0+. The stateful device histories
            // (hs: junction / tail charges, flux of a saturable core or a
            // machine) stay at x0, restored above: they are continuous
            // through the event (only the C / L states are probed), and
            // their laws read x0+'s algebraic entries, which carry the
            // extrapolated impulse (F-5-1: -6.8e8 V).
            // (event_rl_x keeps the stages' line 2·x1 − x2; the states take
            // x0 + J from the snapshot at x0)
            // (ip_hist stays at x0: the right limit's states, event_right_limit)
            ip_hist_p = ip_hist;
            for (Size k = 0; k < ne; ++k) ip_hist_p[2 * k + (is_cap(ents[k]) ? 0 : 1)] += ip_d[2 * k];
            history.restore(ip_hist_p);
            // a nonlinear capacitor that jumped starts from its (v0+, q0+)
            auto& ncm = hs.get<pwl::NonlinearCapacitorHistory>().mutable_entries();
            for (Size k = 0; k < ncm.size(); ++k) {
                if (ip_nc[3 * k + 2] != Real{0}) {
                    rl_nc0.emplace_back(k, ncm[k].v_prev, ncm[k].q_prev);   // the row's x0
                    ncm[k].v_prev = ip_nc[3 * k];
                    ncm[k].q_prev = ip_nc[3 * k + 1];
                }
            }
            // a saturable core that jumped starts from its (λ0+, i0+)
            auto& slm = hs.get<pwl::SaturableInductorHistory>().mutable_entries();
            for (Size k = 0; k < slm.size(); ++k) {
                if (ip_sl[3 * k + 2] != Real{0}) {
                    rl_sl0.emplace_back(k, slm[k].lambda_old, slm[k].i_L_old);   // the row's x0
                    slm[k].lambda_old = ip_sl[3 * k + 1];
                    slm[k].i_L_old = ip_sl[3 * k];
                }
            }
            history.compute_b_extra_backward_euler(h_s, ip_h1);
            b_extra = ip_b - ip_h0 + ip_h1;
            x = ip_x1;   // the solve's start (Newton): the relaxed stage
            // rs_bits0 stays the bits at x0: the stage-2 guard's undo of a
            // turn-off of a diode conducting at the interval start (the
            // flyback's secondary) still applies.
            // Stage 1 again, from x0+: its mode decided there. (Here, not in
            // run_restart: one call site fewer in the step body.)
            run_event_iteration(t_from + h_s, dt_eq, true);
            iters_1 += iters - 1;
            if (flipped && has_diodes && !last_solved_bits.empty()) {
                diodes.restore_on_bits(last_solved_bits);
            }
            breach_1 = breach_1 || flipped;
            return true;
        };
        // F-SB-1: an event's RIGHT LIMIT as a state value, x(t_e+), for the
        // result (right_limit_*, inner_events), NOT the lines' quadrature
        // start. The minimal construction:
        //  - STATES (C voltages, L currents): x0, continuous; x0 + J where an
        //    ideal element's constraint moves them (the probe's structural
        //    jump). No solve decides them.
        //  - every other unknown: ONE consistent solve in the mode the step
        //    used (stage 1's: the restart's decision), re-solved in the diode
        //    mode that holds at t_e+ where stage 1's does not (F1; never
        //    settled: no row), with the states pinned to those values. It is
        //    a backward-Euler solve of a fixed ε_run = 1e-3·γ·dt (one factor
        //    per mode, cached: an event in a mode seen before is solves only)
        //    whose histories are corrected until its states land on the
        //    targets (history −= state − target): each pass cuts the states'
        //    motion by ε/τ, so the unknowns are the ε → 0 limit to (ε/τ)^k.
        //    A mode with τ ≲ ε (unresolved), an ideal constraint the targets
        //    meet only to the probe's tolerance, or a mode that is not
        //    consistent at t_e+ (a diode it holds ON carries reverse current,
        //    one it holds OFF is forward-biased) is NOT certified: a pin that
        //    does not converge, a mode that never settles, or a solve that
        //    throws (ll13's ε-mode Newton at S_main's turn-offs) stores no
        //    row and is counted (n_right_limit_fallbacks): a due row is
        //    either stored or counted.
        //  - an ideal element's jump (ip_struct): the constraint current
        //    after the jump is c/ε-polluted by the probe's tolerance; three
        //    solves at ε, 2ε, 4ε (ε = 5e-4·γh) and −2x1 + 5x2 − 2x4 remove
        //    the 1/ε and ε terms (O(ε²) left). Rare: its own factors.
        //  - a conductance-form impulse the record reports (ip_jump, not
        //    structural): ε = 1e-9·γh, below its ps mode: the model's own
        //    right limit (i_L continuous, v = i_L/g_off, F-5-10).
        //  - a LOCATED crossing where only V_f = 0 diodes flip and nothing
        //    else happens: the laws meet there (i = 0, v = 0), nothing jumps:
        //    the right limit is the left (returns 2, no solve). A V_f > 0
        //    crossing (v_D jumps by V_f; the residual current goes through
        //    1/g_off) is solved and counted.
        // Returns 0: no row; 1: `out` holds the row; 2: the row is the left.
        // Called with stage 1 solved (x, its b_extra, its bits); everything
        // it touches is put back. Out of line, once per run.
        auto event_right_limit = [&](auto& rei, Real t_from, Real h_s, Real loc, bool ev,
                                     Vector& out, topology::SwitchStateMask& m_out)
                                     PULSIM_STEP_DRIVER -> int {
            const bool at_loc = t_from == rl_loc.t;
            // F-Z-10: the run's start (not a resume seam) is re-initialised:
            // row 0 is the user's state, x(0+) is published only when the
            // re-initialisation moved it beyond the noise band (below)
            const bool at_start = !(ev || ip_jump || at_loc) && t_from == opts.t_start &&
                                  resume_snap == nullptr && !result.states.empty();
            if (!(ev || ip_jump || at_loc || at_start)) return 0;
            rl_loc.t = std::numeric_limits<Real>::quiet_NaN();   // consumed (no stale time)
            m_out = stamped_mask;
            bool count = false;
            if (at_loc && has_diodes) {
                diodes.snapshot_on_bits_into(rl_bits);
                // pure: V_f = 0 flips only, stage 1 kept exactly the located
                // bits (no other diode moved), no trigger at a crossing on b
                // (at b the next step's trigger must be the flip alone)
                if (rl_loc.pure && !(rl_loc.at_b && rs_trig_t == t_from && !rs_trig_dio) &&
                    rl_bits == rl_loc.bits) {
                    return 2;
                }
                // not a continuity point, solved in the decided mode. A
                // turn-off is counted: the locator's residual current goes
                // through 1/g_off (a V_f > 0 turn-off, R2, or one beside a
                // gate edge, R1); a turn-on at V_th > 0 is read as solved
                count = rl_loc.off;
            }
            rl_x1 = x;
            rl_b0 = b_extra;
            const Size iters_s = iters;
            const bool flipped_s = flipped;
            const bool cycle_s = mask_cycle;
            const auto stamped_s = stamped_mask;
            // a linear mode: the solves are the cached factor's alone (no
            // event machinery runs, nothing else to put back)
            const bool lin = !nl_refresh_effective;
            // stage 1's bits: the caller's rs_bits (taken just before this call)
            bool bits_moved = !lin;   // a nonlinear mode's solves run the settle
            if (!lin) {
                rl_lsb = last_solved_bits;
                hs.snapshot_into(rl_hs);
                // a nonlinear capacitor the probe carried is continuous at
                // t_from: x0's (v, q), put back with rl_hs below. Not an ideal
                // element's state (the structural path refuses those), so a
                // recorded jump (flag 1) is a fast mode that collapses AFTER
                // t_e+: the record reports it, the regular part starts at
                // x0 + J (_result_views.step_start_states), the row is x0
                auto& ncm = hs.get<pwl::NonlinearCapacitorHistory>().mutable_entries();
                for (const auto& [k, v0, q0] : rl_nc0) {
                    ncm[k].v_prev = v0;
                    ncm[k].q_prev = q0;
                }
                auto& slm = hs.get<pwl::SaturableInductorHistory>().mutable_entries();
                for (const auto& [k, l0, i0] : rl_sl0) {   // a saturable core likewise
                    slm[k].lambda_old = l0;
                    slm[k].i_L_old = i0;
                }
            }
            history.snapshot_into(rl_hcur);
            // the run's Newton round-off budget and its counters are not the
            // right limit's (a tiny-ε Newton solve stagnates at round-off)
            const auto nr_s = newton_roundoff_run;
            // the right limit (opt-in rule, new on this line) judges a fixed
            // point at its own scale, as it did when it came in (ll13: 12
            // fallbacks, 19 without); the run's steps keep 2.1.0's test
            // (nr_s puts the flag back with the budget)
            newton_roundoff_run.judge_own_scale = true;
            const auto nr_n = result.newton_roundoff_accepts;
            const auto nr_w = result.newton_roundoff_omega_max;
            const auto nr_r = result.newton_roundoff_row;
            // stage 1's b_extra less its history and source parts (a
            // b_extra_fn keeps its t1 value, as in consistent_start)
            const Real t1 = t_from + h_s;
            const Real src1 = t1 == src_end_at ? src_end_read : t1;
            // a linear stage-1 b_extra with no b_extra_fn / heat / line part is
            // its history and sources alone: that remainder is zero (cost, item 1)
            if (b_extra_fn || lt || lines || !lin) {
                history.compute_b_extra_backward_euler(h_s, rl_h);
                rl_b = rl_b0 - rl_h;
                rl_h.setZero(rl_b.size());
                add_source_rows(rl_h, src1, t1, left_by);
                rl_b -= rl_h;
                if (has_iwave) {
                    sources::compute_current_waveform_b_extra(pool, graph, src1, be_iwave,
                                                              left_by);
                    rl_b -= be_iwave;
                }
            } else {
                rl_b.setZero(rl_b0.size());
            }
            const auto& ents = history.entries();
            // the probe carried states (x0 + d): the right limit starts from
            // x0 plus an ideal element's recorded jumps (flag 1); a move
            // (flag 2) is continuous at t_from, and a conductance-form jump
            // keeps the model's own right limit (states at x0)
            if (ip_moved) {
                // every state from x0: a fast mode behind a resistor is
                // continuous (H1); an ideal element's dependent states are left
                // to its constraint, whose ε solve carries the jump's charge /
                // flux exactly (from x0 + J of the probe's FLAGGED states only, a
                // dependent state it did not flag — C1 of C1 -S- C2 || R — was
                // shared twice: 9.977 V for 9.524 V)
                rl_hist = ip_hist;
                rl_dep.clear();
                if (ip_jump && ip_struct) {
                    for (const auto& ds : structure.check(stamped_mask, true).dependent) {
                        rl_dep.insert(rl_dep.end(), ds.states.begin(), ds.states.end());
                    }
                }
                history.restore(rl_hist);
            } else {
                rl_hist = rl_hcur;   // the histories are at x0
            }
            const bool elim = ip_jump && ip_struct;
            const Real eps = elim ? Real{5e-4} * h_s
                             : ip_jump ? Real{1e-9} * h_s
                                       : Real{1e-3} * kGamma * opts.dt;
            // D6: the sources on the post side of the event's last edge,
            // t_from + loc (a split cluster's ev_s); without the split a
            // source edge inside the step is not landed (loc = 0)
            const Real t_src = t_from + (ev && split && !at_loc ? loc : Real{0});
            Real e_src = -1;   // the ε rl_bs holds the sources of
            Real e_last = 0;   // the last solve's ε (refined)
            auto solve_eps = [&](Real e) {
                if (e != e_src) {   // the post side of an edge at t_src (lb = 0)
                    rl_bs = rl_b;
                    add_source_rows(rl_bs, t_src + e, t_src + e, Real{0});
                    if (has_iwave) {
                        sources::compute_current_waveform_b_extra(pool, graph, t_src + e,
                                                                  be_iwave, Real{0});
                        rl_bs += be_iwave;
                    }
                    e_src = e;
                }
                history.compute_b_extra_backward_euler(e, rl_h);
                b_extra = rl_bs + rl_h;
                if (lin) {
                    cache.solve_at(stamped_mask, Real{2} * e, b_extra, x);   // solve-exempt: restart-line re-solve (DECISIONS 22)
                } else {
                    x = rl_x1;
                    rei(t_from + e, Real{2} * e, true, true);
                }
                e_last = e;
            };
            // F-Z-4: the published row, one step of iterative refinement on the
            // last ε solve's factor. L/ε (1e7 Ω for 1 mH) beside 1e5 Ω rows: the
            // LU's backward error left 4.6e-11 A in a divider node's KCL; one
            // step makes it componentwise (Skeel). Once per row, linear mode.
            // The correction solve is skipped when r already meets that bound,
            // |r_i| <= n·eps·(|J|·|x|)_i (the KCL tests allow 4x that).
            auto refined = [&]() -> const Vector& {
                const Vector* bc = nullptr;
                const sparse::DirectSolver* f = nullptr;
                const auto* J = lin ? cache.resident_event_matrix(stamped_mask, Real{2} * e_last,
                                                                  &bc, &f)
                                    : nullptr;
                if (J && f) {
                    rl_rr = -(*bc + b_extra);
                    rl_dx.setZero(x.size());   // (|J|·|x|) until the solve overwrites it
                    for (Eigen::Index c = 0; c < J->outerSize(); ++c) {
                        for (sparse::Matrix::InnerIterator it(*J, c); it; ++it) {
                            const Real a = it.value() * x[it.col()];
                            rl_rr[it.row()] -= a;
                            rl_dx[it.row()] += std::abs(a);
                        }
                    }
                    const Real tol = static_cast<Real>(J->rows())
                                     * std::numeric_limits<Real>::epsilon();
                    if ((rl_rr.array().abs() > tol * rl_dx.array()).any()) {
                        f->solve(rl_rr, rl_dx);   // factor-reuse
                        x += rl_dx;
                    }
                }
                return x;
            };
            // a state's value in x (C: node difference; L: its branch)
            auto state_of = [&](Size k) -> Real {
                const auto& e = ents[k];
                return e.kind == pwl::DevicePool::StoredKind::Capacitor
                           ? stamping::read_node_voltage(x, e.from) -
                                 stamping::read_node_voltage(x, e.to)
                           : x[e.inductor_branch_var_id];
            };
            auto slot = [&](Size k) {
                return 2 * k + (ents[k].kind == pwl::DevicePool::StoredKind::Capacitor ? 0 : 1);
            };
            int ok = 1;
            // no row: no diode mode holds at t_e+ (F1), or the row's solve did
            // not converge (a diverged pin is no state; counted, not published)
            bool unheld = false;
            const Size rf_s = restart_factorizations;
            struct AuxEvents {   // the run's event factors stay put, on any exit
                const decltype(cache)& c;
                explicit AuxEvents(const decltype(cache)& cc) : c(cc) { c.swap_aux_events(); }
                ~AuxEvents() { c.swap_aux_events(); }
            };
            std::optional<AuxEvents> aux_events(std::in_place, cache);
            try {
                // nonlinear capacitors: pinned in charge like the C / L
                // histories (their q_prev is the BE companion's base)
                auto* ncw = lin ? nullptr
                                : &hs.get<pwl::NonlinearCapacitorHistory>().mutable_entries();
                const Size n_nc = ncw ? ncw->size() : 0;
                auto nc_q = [&](Size k) {
                    const auto& e = (*ncw)[k];
                    return models::NonlinearCapacitor::charge(
                        e.params, stamping::read_node_voltage(x, e.from) -
                                      stamping::read_node_voltage(x, e.to));
                };
                rl_ncq.resize(n_nc);
                for (Size k = 0; k < n_nc; ++k) rl_ncq[k] = (*ncw)[k].q_prev;
                // saturable cores: pinned in flux (λ_old is the BE row's base)
                auto* slw = lin ? nullptr
                                : &hs.get<pwl::SaturableInductorHistory>().mutable_entries();
                const Size n_sl = slw ? slw->size() : 0;
                auto sl_lam = [&](Size k) {
                    const auto& e = (*slw)[k];
                    return pwl::SaturableInductorHistory::flux_and_inductance(e, x[e.branch_var_id]).first;
                };
                rl_slq.resize(n_sl);
                for (Size k = 0; k < n_sl; ++k) rl_slq[k] = (*slw)[k].lambda_old;
                auto is_dep = [&](Size k) {
                    return std::find(rl_dep.begin(), rl_dep.end(), ents[k].branch_id) != rl_dep.end();
                };
                // One ε solve with every state pinned to its target (an ideal
                // element's dependent states excepted while rl_dep names them:
                // the constraint sets them). The pin is the history that makes
                // the backward-Euler state land on the target — the states act
                // as sources, the other unknowns are the consistent-
                // initialisation solve. The history correction u ← u − r(u)
                // (r = state − target) contracts by (ε/τ)/(1 + ε/τ) per mode:
                // slow for a mode faster than ε (a cut into g_off = 1e-12:
                // τ = 1e-18 s; a 1 mΩ RC), so it is Anderson-accelerated (the
                // map is affine in a linear mode: exact after one difference
                // per fast mode). Returns whether every pinned state met its
                // target.
                auto pinned = [&](Real e, bool predict = false) -> bool {
                    rl_hw = rl_hist;
                    if (!predict) history.restore(rl_hw);   // (predict restores its own)
                    for (Size k = 0; k < n_nc; ++k) (*ncw)[k].q_prev = rl_ncq[k];
                    for (Size k = 0; k < n_sl; ++k) (*slw)[k].lambda_old = rl_slq[k];
                    rl_pin.clear();
                    for (Size k = 0; k < ents.size(); ++k) {
                        if (!is_dep(k)) rl_pin.push_back(k);
                    }
                    const Size n_pin = rl_pin.size();
                    const auto m = static_cast<Index>(n_pin + n_nc + n_sl);
                    Real s_c = 0, s_l = 0, s_q = 0, s_f = 0;   // the states' scale, per kind
                    for (Size k = 0; k < ents.size(); ++k) {
                        Real& sk = ents[k].kind == pwl::DevicePool::StoredKind::Capacitor ? s_c : s_l;
                        sk = std::max(sk, std::abs(rl_hist[slot(k)]));
                    }
                    for (Size k = 0; k < n_nc; ++k) s_q = std::max(s_q, std::abs(rl_ncq[k]));
                    for (Size k = 0; k < n_sl; ++k) s_f = std::max(s_f, std::abs(rl_slq[k]));
                    constexpr Index kDepth = 6;
                    constexpr Size kPasses = 24;
                    // run-scope buffers: no allocation per event once sized
                    Vector& u = rlp_u; Vector& r = rlp_r; Vector& g = rlp_g;
                    Vector& r0 = rlp_r0; Vector& g0 = rlp_g0; Vector& sc = rlp_sc;
                    DenseMatrix& dr = rlp_dr; DenseMatrix& dg = rlp_dg;
                    u.resize(m); r.resize(m); g.resize(m); r0.resize(m); g0.resize(m); sc.resize(m);
                    dr.resize(m, kDepth); dg.resize(m, kDepth);
                    Index nh = 0;
                    auto target = [&](Index i) -> Real {
                        const auto q = static_cast<Size>(i);
                        return q < n_pin ? rl_hist[slot(rl_pin[q])]
                               : q < n_pin + n_nc ? rl_ncq[q - n_pin] : rl_slq[q - n_pin - n_nc];
                    };
                    for (Index i = 0; i < m; ++i) u[i] = target(i);
                    // the first pin predicted from stage 1's slope (states continuous
                    // at t_from): u = target − ε·(x1 − x0)/γh; a slow state then lands
                    // within the tolerance in the first solve (one solve per
                    // event, not two — the restart line's cost, item 1)
                    if (predict) {
                        for (Size q = 0; q < n_pin; ++q) {
                            const Size k = rl_pin[q];
                            const auto& en = ents[k];
                            const Real s1 = en.kind == pwl::DevicePool::StoredKind::Capacitor
                                ? stamping::read_node_voltage(rl_x1, en.from) - stamping::read_node_voltage(rl_x1, en.to)
                                : rl_x1[en.inductor_branch_var_id];
                            u[static_cast<Index>(q)] -= e * (s1 - rl_hist[slot(k)]) / h_s;
                            rl_hw[slot(k)] = u[static_cast<Index>(q)];
                        }
                        history.restore(rl_hw);
                    }
                    bool conv = m == 0;
                    for (Size pass = 0; pass < kPasses; ++pass) {
                        solve_eps(e);
                        if (m == 0) break;
                        if (pass == 0) {
                            for (Size k = 0; k < ents.size(); ++k) {
                                Real& sk = ents[k].kind == pwl::DevicePool::StoredKind::Capacitor ? s_c : s_l;
                                sk = std::max(sk, std::abs(state_of(k)));
                            }
                            for (Size k = 0; k < n_nc; ++k) s_q = std::max(s_q, std::abs(nc_q(k)));
                            for (Size k = 0; k < n_sl; ++k) s_f = std::max(s_f, std::abs(sl_lam(k)));
                            for (Size q = 0; q < n_pin; ++q) {
                                sc[static_cast<Index>(q)] =
                                    ents[rl_pin[q]].kind == pwl::DevicePool::StoredKind::Capacitor ? s_c : s_l;
                            }
                            for (Size k = 0; k < n_nc; ++k) sc[static_cast<Index>(n_pin + k)] = s_q;
                            for (Size k = 0; k < n_sl; ++k) sc[static_cast<Index>(n_pin + n_nc + k)] = s_f;
                        }
                        for (Size q = 0; q < n_pin; ++q) {
                            r[static_cast<Index>(q)] = state_of(rl_pin[q]) - rl_hist[slot(rl_pin[q])];
                        }
                        for (Size k = 0; k < n_nc; ++k) r[static_cast<Index>(n_pin + k)] = nc_q(k) - rl_ncq[k];
                        for (Size k = 0; k < n_sl; ++k) {
                            r[static_cast<Index>(n_pin + n_nc + k)] = sl_lam(k) - rl_slq[k];
                        }
                        conv = true;
                        for (Index i = 0; i < m; ++i) {
                            conv = conv && std::abs(r[i]) <= Real{1e-9} * sc[i];   // NaN fails
                        }
                        if (conv) break;
                        g = u - r;   // the plain correction
                        if (pass > 0) {
                            const Index c = nh < kDepth ? nh : static_cast<Index>((pass - 1) % static_cast<Size>(kDepth));   // ring
                            for (Index i = 0; i < m; ++i) {
                                dr(i, c) = (r[i] - r0[i]) / (sc[i] > Real{0} ? sc[i] : Real{1});
                            }
                            dg.col(c) = g - g0;
                            if (nh < kDepth) ++nh;
                        }
                        r0 = r;
                        g0 = g;
                        u = g;
                        if (nh > 0) {
                            Vector& rs = rlp_rs;
                            rs.resize(m);
                            for (Index i = 0; i < m; ++i) rs[i] = r[i] / (sc[i] > Real{0} ? sc[i] : Real{1});
                            if (nh == 1) {   // one difference: the least-squares scalar, no QR
                                const Real dd = dr.col(0).squaredNorm();
                                const Real gm = dd > Real{0} ? dr.col(0).dot(rs) / dd : Real{0};
                                rlp_ua = g - gm * dg.col(0);
                            } else {
                                rlp_gam = dr.leftCols(nh).colPivHouseholderQr().solve(rs);
                                rlp_rs.noalias() = dg.leftCols(nh) * rlp_gam;   // (rs is spent)
                                rlp_ua = g - rlp_rs;
                            }
                            if (rlp_ua.allFinite()) u = rlp_ua;
                        }
                        for (Size q = 0; q < n_pin; ++q) rl_hw[slot(rl_pin[q])] = u[static_cast<Index>(q)];
                        for (Size k = 0; k < n_nc; ++k) (*ncw)[k].q_prev = u[static_cast<Index>(n_pin + k)];
                        for (Size k = 0; k < n_sl; ++k) {
                            (*slw)[k].lambda_old = u[static_cast<Index>(n_pin + n_nc + k)];
                        }
                        history.restore(rl_hw);
                    }
                    return conv;
                };
                // F1: the row's mode must hold at t_e+. Stage 1's decision is
                // the step's, not the instant's (a freewheel diode stage 1
                // holds ON is reverse-biased at t_e+ while a node capacitor
                // still holds the voltage, or after an ideal element's jump
                // lifts its node): while a diode sits on the side its rule
                // leaves, beyond the noise band (a V_th = 0 diode riding its
                // boundary at round-off is held), re-decide on the row and
                // re-solve at ε in that mode. A mode that never settles is not
                // published (the ledger would book from it). `again` re-solves
                // the row in the current mode and returns its convergence.
                // `imp` (an ideal jump): the jump's own solve, whose 1/ε currents
                // show the impulse's direction through each diode — an ideal
                // freewheel diode ON at t_e− that the jump drives backwards
                // reads its regular forward current after the jump (review:
                // row 0 V with D ON for the 9.52 V share).
                auto hold_mode = [&](auto&& again, bool& conv, const Vector* imp) {
                    if (!has_diodes) return;   // (the bits go back to rs_bits on exit)
                    for (int round = 0;; ++round) {
                        // an unconverged pin is no state to re-decide from (it is
                        // counted already; review: a diverged Anderson row beside
                        // a current source flipped D on garbage)
                        if (!conv) return;
                        const Size nd = diodes.entries().size();
                        auto flips = [&](const Vector& y) {
                            for (Size j = 0; j < nd; ++j) {
                                const Real sg = events::crossing_signal(diodes, j, y);
                                if (events::forbidden(diodes, j, sg) &&
                                    std::abs(sg) > events::held_tolerance(diodes, j, y)) {
                                    return true;
                                }
                            }
                            return false;
                        };
                        const Vector* at = imp && flips(*imp) ? imp : flips(x) ? &x : nullptr;
                        if (!at) return;
                        if (round == 4) break;
                        (void)events::redecide(diodes, *at);
                        bits_moved = true;
                        // the stamped mask's diode bits only: combine() would
                        // re-read a gated device's gate from a stamped bit
                        if (lin) {
                            stamped_mask = stamped_s.overlay(diodes.current_diode_mask(),
                                                             diode_owned);
                            // the structural pass the settle runs on a new mode
                            // (rei's own path runs it): an ideal diode the
                            // re-decision closed into a loop has no row
                            if (structure.active() &&
                                !structure.check(stamped_mask, true).loop_diodes.empty()) {
                                break;
                            }
                        }
                        conv = again();
                        m_out = stamped_mask;
                    }
                    unheld = true;
                };
                if (elim) {
                    // An ideal element's jump, in two consistent solves (the
                    // three-ε extrapolation left O((ε/τ)²): 4.5 % of the charge
                    // beside a 1 mΩ RC): (1) the dependent states after the
                    // jump, from a solve at an ε below every leak (the
                    // constraint shares the charge / flux of x0 exactly; the
                    // other states pinned); (2) every state pinned, the
                    // dependent ones at those values, at the regular ε: the
                    // algebraic unknowns of t_e+ without the jump's 1/ε
                    // current (equal corrections keep the loop's rates equal).
                    // Both solves are the mode's (a diode ON at 1 mΩ beside the
                    // loop drains it during (1)): a re-decided mode redoes both
                    // from x0 (rare: an ideal jump; the copies are its own).
                    const auto dep0 = rl_dep;
                    const auto hist0 = rl_hist;
                    const Size dx0 = rl_dep_x.size();
                    auto elim_solve = [&]() -> bool {
                        rl_dep = dep0;
                        rl_hist = hist0;
                        rl_dep_x.resize(dx0);
                        bool c = pinned(Real{1e-9} * h_s);
                        rl_xa = x;   // the jump's solve (hold_mode reads its impulse)
                        for (Size k = 0; k < ents.size(); ++k) {
                            if (!is_dep(k)) continue;
                            rl_hist[slot(k)] = state_of(k);
                            if (c) rl_dep_x.emplace_back(slot(k), rl_hist[slot(k)]);   // for stage 2
                        }
                        rl_dep.clear();
                        c = pinned(Real{1e-3} * kGamma * opts.dt) && c;
                        out = refined();
                        return c;
                    };
                    bool conv = elim_solve();
                    hold_mode(elim_solve, conv, &rl_xa);
                    unheld = unheld || !conv;
                } else {
                    bool conv = pinned(eps, !ip_jump && !ip_moved);
                    out = refined();
                    // stage 1's slope is the old mode's: no prediction
                    hold_mode([&] { const bool c = pinned(eps); out = refined(); return c; }, conv, nullptr);
                    unheld = unheld || !conv;
                }
            } catch (const std::runtime_error&) {
                ok = 0;   // no right limit recorded; the run goes on
                unheld = true;   // counted: a due row that is not stored is a fallback
            } catch (const std::invalid_argument&) {
                ok = 0;   // a refusal (UnsupportedError) of the ε mode
                unheld = true;
            }
            aux_events.reset();
            restart_factorizations = rf_s;
            if (ok && !out.allFinite()) unheld = true;   // no state: counted
            // a start row is due only when it differs from row 0, unknown
            // when its solve failed: not counted as a fallback
            if (((ok && count) || unheld) && !at_start) ++result.n_right_limit_fallbacks;
            if (unheld) ok = 0;
            if (ok && at_start) {
                // the noise band of the jump test (consistent_start's
                // 1e-8·(1 + |a0| + |a1|)): above the pin's 1e-9 of scale
                const Vector& x0 = result.states.front();
                bool moved = false;
                for (Index i = 0; i < out.size() && !moved; ++i) {
                    moved = std::abs(out[i] - x0[i]) >
                            Real{1e-8} * (Real{1} + std::abs(out[i]) + std::abs(x0[i]));
                }
                if (!moved) ok = 0;
            }
            history.restore(rl_hcur);
            if (!lin) {
                hs.restore(rl_hs);
                last_solved_bits = rl_lsb;
            }
            x = rl_x1;
            b_extra = rl_b0;
            iters = iters_s;
            flipped = flipped_s;
            mask_cycle = cycle_s;
            stamped_mask = stamped_s;
            if (has_diodes && bits_moved) diodes.restore_on_bits(rs_bits);
            newton_roundoff_run = nr_s;
            result.newton_roundoff_accepts = nr_n;
            result.newton_roundoff_omega_max = nr_w;
            result.newton_roundoff_row = nr_r;
            return ok;
        };
        // F-3-1's location of the decided mode's commutation (split_step);
        // built once per run, out of line (see consistent_start).
        auto locate_islands = [&](auto& run_event_iteration, Real a, Real b,
                                  Real h, Real& t_d) PULSIM_STEP_DRIVER {
            const auto ents = diodes.entries();
            diodes.snapshot_on_bits_into(decided_bits);
            {
                const Index nn = graph.num_nodes();
                island.resize(static_cast<Size>(nn) + 1);
                for (Index n = 0; n <= nn; ++n) island[n] = n;
                auto find = [&](Index n) {
                    n = n < 0 ? nn : n;   // ground: slot nn
                    while (island[n] != n) n = island[n] = island[island[n]];
                    return n;
                };
                off_branch.assign(static_cast<Size>(graph.num_branches()), 0);
                for (Size j = 0; j < ents.size(); ++j) {
                    if (!diodes_snap[j]) off_branch[ents[j].branch_id] = 1;
                }
                for (Index bi = 0; bi < graph.num_branches(); ++bi) {
                    if (off_branch[bi]) continue;
                    const auto& br = graph.branch(bi);
                    island[find(br.from)] = find(br.to);
                }
                auto island_of = [&](Size j) {
                    const Index g = find(-1);
                    const Index rf = find(ents[j].from);
                    const Index rt = find(ents[j].to);
                    return rf != g ? rf : (rt != g ? rt : kInvalidIndex);
                };
                auto flips = [&](Size j) {
                    return cross_scratch[j] <= b &&
                           decided_bits[j] != diodes_snap[j];
                };
                // A flipped diode between two floating islands joins them
                // (an isolated bridge: D1 on the transformer side, D4 on the
                // output side conduct together).
                for (Size j = 0; j < ents.size(); ++j) {
                    if (!flips(j)) continue;
                    const Index g = find(-1);
                    const Index rf = find(ents[j].from);
                    const Index rt = find(ents[j].to);
                    if (rf != g && rt != g) island[rf] = rt;
                }
                bool grouped = false;
                for (Size j = 0; j < ents.size(); ++j) {
                    const Index r = flips(j) ? island_of(j) : kInvalidIndex;
                    if (r == kInvalidIndex) continue;
                    Real s0 = 0, s1 = 0;
                    Size n_mode = 0;
                    int dir = 0;
                    bool mixed = false, first = true;
                    for (Size k = 0; k < ents.size(); ++k) {
                        if (!flips(k) || island_of(k) != r) continue;
                        first = first && k >= j;
                        const int d = decided_bits[k] ? 1 : -1;
                        mixed = mixed || (dir != 0 && d != dir);
                        dir = d;
                        const auto& e = ents[k];
                        const bool on = diodes_snap[k];
                        s0 += models::SwitchedDiode::crossing_signal(
                            on, e.law_voltage(x_prev, on), e.params());
                        s1 += models::SwitchedDiode::crossing_signal(
                            on, e.law_voltage(x_first, on), e.params());
                        ++n_mode;
                    }
                    if (!first || n_mode < 2 || mixed) continue;
                    // OFF -> ON stays off while s < 0; ON -> OFF
                    // stays on while s > 0.
                    // (`far` and `near` are macros in the Windows headers)
                    const bool no_cross = dir > 0 ? s0 >= Real{0} : s0 <= Real{0};
                    const Real t_g = no_cross ? a : interp_commutation_time(a, b, s0, s1);
                    for (Size k = 0; k < ents.size(); ++k) {
                        if (flips(k) && island_of(k) == r) cross_scratch[k] = t_g;
                    }
                    grouped = true;
                }
                if (grouped) {
                    // the crossers the decided mode does not flip
                    // are not events of this step
                    t_d = b;
                    for (Size j = 0; j < ents.size(); ++j) {
                        if (cross_scratch[j] <= b && !flips(j)) {
                            cross_scratch[j] = b + h;
                        }
                        t_d = std::min(t_d, cross_scratch[j]);
                    }
                }
            }
            // F-SB-4: the estimate above is one chord of the crossing signal
            // between x_prev and the first solve. A signal curved over the
            // step (a sine source near its peak) puts that chord's root a
            // fraction of h off (diode_rectifier: +0.227 h). Refine the
            // earliest crossing with the one locator (events::illinois, as
            // TR-BDF2) on trapezoidal trial solves from the step start under
            // the start bits, to snap. The crossers that share the estimate
            // (one diode, or a floating island's group) share the signal.
            // Then every OTHER crosser is checked on a trial solve at the
            // refined instant: one whose signal changed sign there crossed
            // earlier (two independent diodes, one signal convex and one
            // concave: the chords order them wrongly) and is refined on [a,
            // t_new] in turn (at most once per diode). A give-up keeps the
            // chord and is counted (split_locator_giveups).
            Real hi = b;
            const Vector* x_hi = &x_first;
            for (Size round = 0; round <= ents.size(); ++round) {
                const Real t_g = t_d;
                if (!(t_g > a && t_g < hi)) return;
                int dir = 0;
                for (Size j = 0; j < ents.size(); ++j) {
                    if (cross_scratch[j] != t_g) continue;
                    const int d = diodes_snap[j] ? -1 : 1;
                    if (dir != 0 && d != dir) {   // mixed directions: keep the chord
                        ++result.split_locator_giveups;
                        return;
                    }
                    dir = d;
                }
                auto signal = [&](const Vector& xv) {
                    Real s = 0;
                    for (Size j = 0; j < ents.size(); ++j) {
                        if (cross_scratch[j] != t_g) continue;
                        const bool on = diodes_snap[j];
                        s += models::SwitchedDiode::crossing_signal(
                            on, ents[j].law_voltage(xv, on), ents[j].params());
                    }
                    return s;
                };
                const Real s_a = signal(x_prev), s_b = signal(*x_hi);
                if (!(dir > 0 ? (s_a < Real{0} && s_b > Real{0})
                              : (s_a > Real{0} && s_b < Real{0}))) {
                    ++result.split_locator_giveups;
                    return;
                }
                events::Bracket br{a, s_a, hi, s_b, s_a, s_b, 0};
                // converged: the bracket, or two successive trial instants,
                // within snap
                auto narrow = [&](const events::Bracket& q) { return q.tb - q.ta <= snap; };
                // F-Z-2: stop on the signal. t_cur is taken when the root of
                // the bracket's raw chord is within snap of it (|s(t_cur)| <=
                // snap * the chord's slope: the time tolerance through the
                // signal), and the bracket's far end the same way before any
                // trial (a crossing on the grid point). Not its start: a piece
                // after a cut starts at s ~ 0 by design, and a signal that dips
                // and comes back crosses inside. A cut at the last trial's
                // instant solves at its dt: the factor the trial left is the
                // piece's.
                auto chord = [](const events::Bracket& q) {
                    return q.ta - q.sa_raw * (q.tb - q.ta) / (q.sb_raw - q.sa_raw);
                };
                // one trapezoidal trial from the step start under the start bits
                auto trial = [&](Real t_loc) {
                    x = x_prev;
                    step::restore(history, hs, history_snap, hs_snap);
                    diodes.restore_on_bits(diodes_snap);
                    try {
                        accumulate_b_extra(t_loc, t_loc - a);
                        run_event_iteration(t_loc, t_loc - a, false, true);
                    } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                        throw;
                    } catch (const std::runtime_error&) {
                        return false;
                    }
                    return x.allFinite();
                };
                Real t_cur = a, t_was = hi;
                bool failed = false;
                const Real c0 = chord(br);
                const bool at_end = hi - c0 <= snap;
                if (at_end) t_cur = hi;
                const int n_trials = at_end ? 0 : events::illinois(br, 40,
                    [&](Real, const events::Bracket& q) { return narrow(q); },
                    [&](Real t_loc, Real& sm) {
                        t_was = t_cur;
                        t_cur = t_loc;
                        if (!trial(t_loc)) {
                            failed = true;
                            return false;   // a failed trial ends the search: the bracket's chord
                        }
                        sm = signal(x);
                        return true;
                    },
                    [](events::BracketEnd) {},
                    [&](const events::Bracket& q) {
                        return narrow(q) || std::abs(t_cur - t_was) <= snap ||
                               q.sa_raw == Real{0} || q.sb_raw == Real{0} ||
                               std::abs(chord(q) - t_cur) <= snap;
                    });
                result.split_locator_trials += static_cast<Size>(n_trials);
                if (failed || (n_trials >= 40 && !narrow(br))) ++result.split_locator_giveups;
                // the root of the final bracket's unweighted chord, or the
                // last trial (or bracket end) within snap of it
                const Real t_new = at_end || (!failed && n_trials > 0 &&
                                              std::abs(chord(br) - t_cur) <= snap) ? t_cur
                    : br.sa_raw == Real{0} ? br.ta
                    : br.sb_raw == Real{0} ? br.tb
                    : std::clamp(br.ta - br.sa_raw * (br.tb - br.ta) / (br.sb_raw - br.sa_raw),
                                 br.ta, br.tb);
                t_d = b;
                bool others = false;
                for (Size j = 0; j < ents.size(); ++j) {
                    if (cross_scratch[j] == t_g) cross_scratch[j] = t_new;
                    others = others || (cross_scratch[j] > t_new && cross_scratch[j] <= b);
                    t_d = std::min(t_d, cross_scratch[j]);
                }
                if (!others || failed || !(t_new > a)) return;
                if (!trial(t_new)) {
                    ++result.split_locator_giveups;
                    return;
                }
                loc_xt = x;
                bool earlier = false;
                for (Size j = 0; j < ents.size(); ++j) {
                    if (!(cross_scratch[j] > t_new && cross_scratch[j] <= b)) continue;
                    const bool on = diodes_snap[j];
                    const Real s0 = models::SwitchedDiode::crossing_signal(
                        on, ents[j].law_voltage(x_prev, on), ents[j].params());
                    const Real s1 = models::SwitchedDiode::crossing_signal(
                        on, ents[j].law_voltage(loc_xt, on), ents[j].params());
                    if (on ? s1 <= Real{0} && s0 > Real{0} : s1 >= Real{0} && s0 < Real{0}) {
                        cross_scratch[j] = interp_commutation_time(a, t_new, s0, s1);
                        earlier = true;
                    }
                }
                if (!earlier) return;
                hi = t_new;
                x_hi = &loc_xt;
                t_d = b;
                for (Size j = 0; j < ents.size(); ++j) t_d = std::min(t_d, cross_scratch[j]);
            }
        };

        // User-cancellation check (Phase: live scope). Lets a
        // GUI / external watchdog stop the simulation early
        // while preserving the partial trace.
        auto step_cancelled = [&]() PULSIM_STEP_PHASE {
            return should_continue && !should_continue();
        };

        auto step_begin = [&](Size k) PULSIM_STEP_PHASE {
            const Real t = t_step = fixed_grid.at(k_off + k);
            t_reached = t;
            k_reached = k;
            const Real t_prev = t_prev_step = fixed_grid.at(k_off + k - 1);
            // The heat the previous step queued, as this step's power.
            if (lt) lt->advance(opts.dt);

            // Snapshot the state at t_prev for sub-step
            // commutation timing (Layer 5 V2.2) and state
            // correction (Layer 5 V3). Assignment into the hoisted
            // buffer — no per-step allocation.
            x_prev = x;
            if (has_ideal_diode) {
                const auto& es0 = diodes.entries();
                commut_on_prev.resize(es0.size());
                for (std::size_t j = 0; j < es0.size(); ++j) commut_on_prev[j] = es0[j].is_on;
            }

            // State-aware observer fires BEFORE `switch_fn(t)`
            // so the user can update a Python-side PI / sampler /
            // comparator with `x_prev` and have `switch_fn(t)`
            // read the new duty / mask. Under the restart rule the
            // gate is sampled at the step START (design D4); the
            // observer keeps its (t_k, x_{k-1}) label, whose move to
            // t_{k-1} belongs to fix-control-tick-semantics.
            // Restart rule without the split: the gate 0.5001 h before the
            // step end (gate_back), so an edge inside the step takes effect
            // at the nearer step boundary (a tie at mid-step: the end) and an edge on a grid point t_prev
            // is exact (F-SB-2). Not an ulp from t: a closure's own tolerance
            // band (1e-9 of a period) must not read an edge at t as already
            // there. The split samples t_prev and places the step's edges itself.
            gate_t = !restart_rule ? t : split ? t_prev : t - gate_back * (t - t_prev);
            if (step_observer || step_observer_in) {
                (has_ticks ? step_observer : step_observer_in)(t, x_prev);
                // An observer (or a control tick) may mutate the
                // schedule: its oracle answer is stale (review M3).
                edge_from = std::numeric_limits<Real>::infinity();
                edge_next = -std::numeric_limits<Real>::infinity();
                // ... and so may its declared b_extra_fn jumps (review L6).
                bj_from = std::numeric_limits<Real>::infinity();
                bj_next = -std::numeric_limits<Real>::infinity();
                gate_memo_n = 0;
            }
            if (!split) {
                gate_memo_n = 0;
            } else {   // F-Z-2: the last step's end is this step's start (one read, not two)
                Size keep = 0;
                for (Size q = 0; q < gate_memo_n; ++q) {
                    if (gate_memo_t[q] == t_prev) {
                        if (q != 0) {
                            gate_memo_t[0] = t_prev;
                            gate_memo_m[0] = gate_memo_m[q];
                        }
                        keep = 1;
                        break;
                    }
                }
                gate_memo_n = keep;
            }
            // ---- add-fixed-step-event-splitting: the step's events ----
            // Gate edges from the schedule's oracle and built-in source
            // jumps inside [t_prev + snap, t - snap), in time order. One
            // before that is taken at t_prev (the gate is sampled at it;
            // a source jump triggers the restart below), one after it at
            // t (the next step's; a source jump is read at its left
            // limit by this step's last solve).
            n_ev = 0;
            mode_ev_t = std::numeric_limits<Real>::quiet_NaN();
            ev_overflow = false;
            capture_first = false;   // keep the first solve (diode split)
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
                                    ++result.n_gate_events;
                                    if (e < t_prev + snap) {
                                        gate_t = e;
                                    } else {
                                        push_ev(e);
                                    }
                                    gate_mask = m;
                                }
                            } else {
                                // taken by the sample at t (no gate read
                                // here: it would be a counted switch_fn call)
                                ++result.n_gate_events;
                            }
                            edge_from = e;
                            e = next_edge_fn(e);
                            edge_next = e;
                        }
                        ev_overflow = ev_overflow || e < t - snap;
                    }
                    // an edge exactly on the grid point t: the next step's
                    // oracle query (strictly after t) skips it
                    if (edge_next == t) ++result.n_gate_events;
                } else {
                    // An opaque switch_fn (F-SB-2): each change between the
                    // step's samples t_prev and t is bisected to snap and
                    // split at like an oracle edge (post side). A pulse that
                    // opens and closes between two samples is not seen.
                    Real lo = t_prev;
                    topology::SwitchStateMask g_lo = gate_at(t_prev);
                    const topology::SwitchStateMask g_end = gate_at(t);
                    for (Size q = 0; !(g_lo == g_end); ++q) {
                        if (q == kMaxStepEvents) {
                            ++result.unlocated_gate_steps;
                            ev_overflow = true;
                            break;
                        }
                        Real hi = t;
                        // capped: below an ulp of t the midpoint stops moving
                        for (int it = 0; it < 64 && hi - lo > snap; ++it) {
                            const Real mid = Real{0.5} * (lo + hi);
                            if (mid <= lo || mid >= hi) break;
                            ++result.switch_fn_calls;
                            if (switch_fn(mid) == g_lo) {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        if (hi >= t - snap) break;   // taken at t
                        ++result.n_gate_events_bisected;
                        if (hi < t_prev + snap) {
                            gate_t = hi;
                        } else {
                            push_ev(hi);
                        }
                        g_lo = gate_at(hi);
                        lo = hi;
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
                    ev_overflow = ev_overflow || e < t - snap;
                }
                if (has_b_jumps) {
                    // Declared b_extra_fn jumps (review M4): the same
                    // window; one within snap of t is taken at t and
                    // restarts the next step.
                    Real e = b_jump_after(t_prev);
                    for (Size q = 0; q < 4 * kMaxStepEvents && e < t - snap; ++q) {
                        if (e >= t_prev + snap) {
                            push_ev(e);
                        }
                        e = b_jump_after(e);
                    }
                    ev_overflow = ev_overflow || e < t - snap;
                }
                std::sort(ev_t.begin(), ev_t.begin() + n_ev);
                // Events closer than `snap` merge into the earlier one,
                // which samples the gate at the cluster's last event (a
                // dead-time pair < snap apart lands both edges). The
                // unresolved count is of every step event merged here:
                // gate edges, source and declared b_extra_fn jumps alike.
                Size m = 0;
                for (Size q = 0; q < n_ev; ++q) {
                    if (m == 0 || ev_t[q] - ev_t[m - 1] >= snap) {
                        ev_t[m] = ev_s[m] = ev_t[q];
                        ++m;
                    } else {
                        ev_s[m - 1] = ev_t[q];
                        ++result.n_gate_events_unresolved;
                    }
                }
                n_ev = m;
            }
            gate_t0 = gate_t;
            // V3 snapshots — taken regardless of whether
            // correction fires (cheap; vectors are small per
            // dynamic-branch). Restored only if a commutation
            // event is detected AND
            // enable_substep_state_correction is true.
            if (may_roll_back) {
                ++result.rollback_snapshots;
                step::snapshot_into(history, hs, history_snap, hs_snap);
                if (has_diodes) {
                    diodes.snapshot_on_bits_into(diodes_snap);
                } else {
                    diodes_snap.clear();
                }
            }
            // Pre-event mask: what mask we'd use BEFORE any
            // diode flip during this step. Captured before
            // event iteration runs.
            mask_pre = gate_at(gate_t);
            if (ev_overflow) {
                ++result.split_event_overflows;
            }
            if (has_diodes) {
                mask_pre = diodes.combine(mask_pre);
            }
        };  // step_begin

        // A turn-off that stage 2 of a restart step (t_from, h) makes and the
        // backward-Euler re-take confirms: its instant on SDIRK trial steps
        // of tau from the step start under stage 1's bits (the method the
        // cut's first piece takes), bracketed by stage 1's own state at γh
        // (it conducts there) and the trial at h. Kept, SDIRK's stage 2
        // starts the new mode from x0* = x0 + c·(x1 − x0), stage 1's slope
        // extrapolated past the event (a 1 nH boost: v_out 51.34 V for
        // 42.11 V). Returns NaN: no cut (a turn-on in stage 2, no sign
        // change, a failed trial, an instant within snap of an end).
        // Leaves x / histories / bits as the last trial: the caller puts
        // them back. Out of line, once per run.
        // F-Z-11 (the split on): any crossing, turn-on (on the voltage) or
        // turn-off (on the current), from the pre-side bits `pre` to the
        // decided `post`; from_start: a flip of stage 1 itself, bracketed by
        // the (sub)step's start state (rs_cut.x0) instead of stage 1's.
        // also: a second decided mode (stage 2's bits, from_start) whose flips
        // are bracketed with stage 1's
        auto restart_cut_at = [&](auto& rei, Real t_from, Real h, bool from_start = false,
                                  const std::vector<bool>* also = nullptr)
                                  PULSIM_STEP_DRIVER -> Real {
            constexpr Real nan = std::numeric_limits<Real>::quiet_NaN();
            constexpr Real c = (Real{1} - kGamma) / kGamma;
            const auto ents = diodes.entries();
            const Real t2 = t_from + h;
            const Real t1 = from_start ? t_from : t_from + kGamma * h;
            const std::vector<bool>& pre = from_start ? rs_bits0 : rs_bits;
            const std::vector<bool>& post = from_start ? rs_bits : rs_bits2;
            const Vector& x_a = from_start ? rs_cut.x0 : rs_x1;
            for (Size q = 0; !split && q < ents.size(); ++q) {
                if (rs_bits[q] != rs_bits2[q] && !rs_bits[q]) return nan;   // a turn-on
            }
            auto trial = [&](Real tau) {
                step::restore(history, hs, rs_hist, rs_hs);
                diodes.restore_on_bits(pre);
                x = rs_x1;
                const Real ts = kGamma * tau;
                try {
                    accumulate_b_extra(t_from + ts, Real{2} * ts, true);
                    rei(t_from + ts, Real{2} * ts, true, true);
                    history.extrapolate_previous(x, c);
                    hs.extrapolate_previous(x, c);
                    accumulate_b_extra(t_from + tau, Real{2} * ts, true);
                    rei(t_from + tau, Real{2} * ts, true, true);
                } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                    throw;
                } catch (const std::runtime_error&) {
                    return false;
                }
                return x.allFinite();
            };
            if (!trial(h)) return nan;
            rs_cut.x_end = x;
            // > 0 while diode j stays in its pre-side state (an OFF diode's
            // crossing signal is negative while it stays off)
            auto sig = [&](Size j, const Vector& xv) {
                const bool on = pre[j];
                const Real s = models::SwitchedDiode::crossing_signal(
                    on, ents[j].law_voltage(xv, on), ents[j].params());
                return on ? s : -s;
            };
            // the earliest chord, and the diodes that cross with it (a bridge's
            // pair) share one signal: ON stays on while s > 0
            cross_scratch.assign(ents.size(), t2 + h);
            Real t_c = t2;
            for (Size j = 0; j < ents.size(); ++j) {
                const bool moved = pre[j] != post[j] || (also && pre[j] != (*also)[j]);
                if (!(moved && (split || pre[j]))) continue;
                const Real s1 = sig(j, x_a), s2 = sig(j, rs_cut.x_end);
                if (!(s1 > Real{0} && s2 < Real{0})) continue;
                cross_scratch[j] = interp_commutation_time(t1, t2, s1, s2);
                t_c = std::min(t_c, cross_scratch[j]);
            }
            if (!(t_c < t2)) return nan;
            auto signal = [&](const Vector& xv) {
                Real s = 0;
                for (Size j = 0; j < ents.size(); ++j) {
                    if (cross_scratch[j] - t_c < snap) s += sig(j, xv);
                }
                return s;
            };
            const Real s_a = signal(x_a), s_b = signal(rs_cut.x_end);
            events::Bracket br{t1, s_a, t2, s_b, s_a, s_b, 0};
            auto narrow = [&](const events::Bracket& q) { return q.tb - q.ta <= snap; };
            Real t_cur = t1, t_was = t2;
            bool failed = false;
            // F-Z-2: stop on the signal, as locate_islands (no end shortcut:
            // t2 is refused below, and near t1 a curved signal may cross
            // inside). The split's only: the rule alone reaches this cut too,
            // unchanged.
            auto chord = [](const events::Bracket& q) {
                return q.ta - q.sa_raw * (q.tb - q.ta) / (q.sb_raw - q.sa_raw);
            };
            const int n_trials = events::illinois(br, 40,
                [&](Real, const events::Bracket& q) { return narrow(q); },
                [&](Real t_loc, Real& sm) {
                    t_was = t_cur;
                    t_cur = t_loc;
                    if (!trial(t_loc - t_from)) {
                        failed = true;
                        return false;
                    }
                    sm = signal(x);
                    return true;
                },
                [](events::BracketEnd) {},
                [&](const events::Bracket& q) {
                    return narrow(q) || std::abs(t_cur - t_was) <= snap ||
                           q.sa_raw == Real{0} || q.sb_raw == Real{0} ||
                           (split && std::abs(chord(q) - t_cur) <= snap);
                });
            result.split_locator_trials += static_cast<Size>(n_trials);
            if (failed) {
                ++result.split_locator_giveups;
                return nan;
            }
            const Real t_new = split && n_trials > 0 && std::abs(chord(br) - t_cur) <= snap ? t_cur
                : br.sa_raw == Real{0} ? br.ta
                : br.sb_raw == Real{0} ? br.tb
                : std::clamp(br.ta - br.sa_raw * (br.tb - br.ta) / (br.sb_raw - br.sa_raw),
                             br.ta, br.tb);
            if (!(t_new - t_from >= snap && t2 - t_new >= snap)) return nan;
            rs_cut.pre = pre;
            rs_cut.bits = pre;
            rs_cut.vf0 = true;
            rs_cut.off = false;
            for (Size j = 0; j < ents.size(); ++j) {
                if (!(cross_scratch[j] - t_c < snap)) continue;
                rs_cut.bits[j] = !pre[j];
                rs_cut.off = rs_cut.off || pre[j];
                // a turn-on at V_th > 0 jumps by (g_on − g_off)·V_th (split_step's rule)
                rs_cut.vf0 = rs_cut.vf0 && ents[j].V_f == Real{0} &&
                             ents[j].gate == models::SwitchedDiode::Gate::None &&
                             (pre[j] || ents[j].V_th == Real{0});
            }
            return t_new;
        };

        // run_restart and restart_lined are built once per run, not per step:
        // step_take passes its run_event_iteration (as consistent_start takes it).
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
        //
        // Stage-2 fallback: x0* carries stage 1's slope, which belongs
        // to stage 1's diode mode. If stage 2's event iteration changes
        // the mode, the new mode's solve starts from that slope: a
        // rectifier at dt/τ = 10 then turns its diode off on stage 2's
        // overshoot (25 V for a 20.7 V phase) and the OFF solve returns
        // x0* itself, 37.2 V on a 24 V source. Such a stage 2 is re-tried
        // as a backward-Euler step of (1 − γ)h from x1, the last
        // physical state. If backward Euler keeps stage 1's mode, the
        // flip was the SDIRK overshoot and the BE step is kept
        // (`be_commit_h` tells the commit which step that was); else
        // the mode change is real and the SDIRK stage 2 is restored bit
        // for bit (a BE remainder there cost DCM buck 8× in accuracy).
        //
        // loc: δt_loc of the event at t_from (consistent_start)
        auto run_restart = [&](auto& run_event_iteration, Real t_from, Real h, Real loc, bool ev) {
            const Real h_s = kGamma * h;
            const Real dt_eq = Real{2} * h_s;
            be_commit_h = h_s;
            ln_pre = false;
            if (sx.on) [[unlikely]] sx.begin(t_from, x);
            accumulate_b_extra(t_from + h_s, dt_eq, true);
            if (has_diodes) diodes.snapshot_on_bits_into(rs_bits0);
            const Size edge_flips = loc < h_s ? decide_edge_mode(t_from + h_s, dt_eq) : 0;
            run_event_iteration(t_from + h_s, dt_eq, true);
            // stage 1 as solved (the mode decide_edge_mode and the settle left),
            // before the probe's solves replace a Newton stage's factor
            if (sx.on) [[unlikely]] {
                sx.stage(0, step::psi_backward_euler_on(dt_eq), t_from + h_s, dt_eq);
            }
            Size iters_1 = iters;
            if (edge_flips > 0) {
                // the decision's flips are the stage's own (the impulse probe's
                // widening, the window), unless the settle ended on the entry
                // bits: a linearization's flip the converged state undid
                diodes.snapshot_on_bits_into(edge_bits);
                iters_1 = !flipped && edge_bits == rs_bits0 ? Size{1} : iters_1 + edge_flips;
            }
            bool breach_1 = flipped;
            if (breach_1 && has_diodes && !last_solved_bits.empty()) {
                diodes.restore_on_bits(last_solved_bits);
            }
            const bool impulsive =
                consistent_start(run_event_iteration, t_from, h_s, dt_eq, loc, iters_1, breach_1);
            if (impulsive && sx.on) [[unlikely]] {
                // the histories moved to x0 + J (the jump's vector series, from
                // the probe's stages) and stage 1 was solved again from there:
                // dJ/dp is not carried
                sx.s.refuse_step(t_from, "restarted from a state its event moved (an "
                                 "impulsive event: the jump of the capacitor "
                                 "voltages / inductor currents read from the "
                                 "restart's probe has no derivative here)");
            }
            if (L) {                // stage 1, for the step's lines
                // an impulsive start moved the histories to x0+: the lines
                // start there (a window integral is the regular part; the
                // jump is the reinit record's)
                if (impulsive && L->v0.size() == history.entries().size()) {
                    const auto& es = history.entries();
                    for (Size k = 0; k < es.size(); ++k) L->v0[k] = es[k].v_prev;
                }
                if (ln_pre && L->v0.size() == ln_v0.size()) L->v0 = ln_v0;
                L->x1 = ln_pre ? ln_x1 : x;
                L->m1 = stamped_mask;
                L->be_kept = false;
                if (has_diodes) diodes.snapshot_on_bits_into(rs_bits);   // (event_right_limit's too)
                // F-SB-1: the right limit, only where something happens
                // (the caller's event, a probed jump)
                const int rl = event_right_limit(run_event_iteration, t_from, h_s, loc,
                                                 ev, L->rl_x, L->rl_m);
                L->rl_ok = rl != 0;
                L->rl_left = rl == 2;
            }
            // Item C: an ideal element's jump is taken by stage 1 from x0 (the
            // constraint shares the charge / flux), but stage 2's pseudo-state
            // x0* = x0 + c·(x1 − x0) then carries c·J of it: the stored sample
            // read a jump-current residue (i(V) = +3.83 A for −1 A). The
            // dependent states start stage 2 from x0+ (the row's solve).
            if (!impulsive && ip_struct && ip_jump && !rl_dep_x.empty()) {
                history.snapshot_into(rl_hw);
                for (const auto& [sl, v] : rl_dep_x) rl_hw[sl] = v;
                history.restore(rl_hw);
                if (sx.on) [[unlikely]] {
                    sx.s.refuse_step(t_from, "an ideal element's jump moved the dependent "
                                     "states' histories to x0+ before stage 2 (the "
                                     "row's solve): that move has no derivative here");
                }
            }
            if (has_diodes) {
                rs_x1 = x;
                if (!L) diodes.snapshot_on_bits_into(rs_bits);   // (with L: taken above)
                if (!rs_first_valid) {   // the attempt's first stage: from t_prev
                    rs_bits_first = rs_bits;   // (word copy; assign() walks bits)
                    rs_first_valid = true;
                }
                step::snapshot_into(history, hs, rs_hist, rs_hs);   // keeps the capacity
            }
            // F-Z-11: stage 1 changed the mode of a (sub)step that starts at no
            // event (a window step, the step after a sliver): a crossing inside
            // [t_from, t_from + h], located from the start state (the split's)
            const bool s1_flip = split && rs_cut.arm && has_diodes && iters_1 > 1 &&
                                 edge_flips == 0 && !ev && !(rs_trig_t == t_from) &&
                                 t_from != opts.t_start && !(rs_bits == rs_bits0);
            const bool cycle_1 = mask_cycle;
            constexpr Real c = (Real{1} - kGamma) / kGamma;
            history.extrapolate_previous(x, c);
            hs.extrapolate_previous(x, c);
            accumulate_b_extra(t_from + h, dt_eq, true);
            run_event_iteration(t_from + h, dt_eq, true);
            const Size iters_2 = iters;
            // A diode stage 1 conducts and stage 2 turns off, with no
            // diode turning on: the overshoot mechanism (`on_to_off`
            // also covers the undo below). A
            // turn-on in stage 2 (e.g. round-off at a zero start,
            // buck CCM at t = 0) keeps the SDIRK stage, unless it
            // undoes stage 1's turn-off of a diode that conducted at
            // the interval start: R(z) < 0 for z < -1/(1-2γ) flips the
            // sign of a stiff residual (a transformer cut by its
            // switch, g_off) and the flip forward-biases the diode
            // again. Kept, the flyback trapped 0.04-0.12 A in its
            // secondary at every switch-off (exact error 0.11, not
            // converging with dt).
            bool on_to_off = false;
            bool kept_be_2 = false;   // F-Z-11: the re-take kept stage 1's mode
            if (has_diodes && iters_2 > 1) {
                diodes.snapshot_on_bits_into(rs_bits2);
                for (Size q = 0; q < rs_bits.size(); ++q) {
                    if (!rs_bits[q] && rs_bits2[q] && !rs_bits0[q]) {
                        on_to_off = false;
                        break;
                    }
                    on_to_off = on_to_off || (rs_bits[q] != rs_bits2[q]);
                }
            }
            if ((on_to_off || (split && iters_2 > 1 && !(rs_bits2 == rs_bits))) &&
                rs_cut.piece1) {
                // the first piece of a cut step ends AT the located turn-off:
                // its stage 2 stays in stage 1's mode (the locator's trial;
                // the flip is the next piece's start)
                x = rs_x1;
                diodes.restore_on_bits(rs_bits);
                mask_cycle = cycle_1;   // the discarded flipping pass's cycle is not this solve's
                run_event_iteration(t_from + h, dt_eq, true, true);
                on_to_off = false;
            }
            if (on_to_off) {
                // Stage 2 changed the mode. Re-take it as backward
                // Euler from x1 under stage 1's bits; if that keeps
                // stage 1's mode, the change was SDIRK's own overshoot
                // (R(z) < 0 for h ≫ τ) and the backward-Euler step is
                // kept. If backward Euler changes the mode too, the
                // event is real and the SDIRK stage 2 stands: its
                // x0* is accurate when the slopes are continuous
                // (DCM turn-off: −2.4e-4 V at T/50 with backward
                // Euler kept, +2.9e-5 V without).
                ++result.stage2_retries;
                rs_x2 = x;
                const bool flipped_2 = flipped;
                const bool cycle_2 = mask_cycle;
                const auto stamped_2 = stamped_mask;
                rs_solved2 = last_solved_bits;
                auto histories_at_x0 = [&] {
                    step::restore(history, hs, rs_hist, rs_hs);
                };
                x = rs_x1;
                diodes.restore_on_bits(rs_bits);
                histories_at_x0();
                mask_cycle = cycle_1;
                const Real h_b = h - h_s;
                // A re-try that fails (Newton, a singular factor at
                // the new dt) falls back to the SDIRK stage 2, which
                // had already succeeded.
                bool keep_be = false, be_ok = true;
                try {
                    commit_backward_euler(h_s);      // stage 1, as taken
                    accumulate_b_extra(t_from + h, Real{2} * h_b, true);
                    run_event_iteration(t_from + h, Real{2} * h_b, true);
                    diodes.snapshot_on_bits_into(rs_bits_be);
                    keep_be = rs_bits_be == rs_bits;
                } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                    throw;   // a run budget: never swallowed by a retry
                } catch (const std::runtime_error&) {
                    ++result.stage2_retry_failures;
                    be_ok = false;
                }
                // a real turn-off inside stage 2: located, the caller
                // (restart_lined) cuts the step there; the state below
                // is put back as before either way
                if (!keep_be && be_ok && rs_cut.arm && !impulsive && !ip_jump && !ln_pre &&
                    !sx.on && !s1_flip) {
                    rs_cut.t = restart_cut_at(run_event_iteration, t_from, h);
                }
                kept_be_2 = keep_be;
                if (keep_be) {
                    ++result.stage2_be_kept;
                    if (L) L->be_kept = true;
                    be_commit_h = h_b;
                } else {
                    // Back to the SDIRK stage 2, bit for bit: the
                    // histories at x0*, recomputed from x0 and x1.
                    histories_at_x0();
                    x = rs_x1;
                    history.extrapolate_previous(x, c);
                    hs.extrapolate_previous(x, c);
                    x = rs_x2;
                    diodes.restore_on_bits(rs_bits2);
                    stamped_mask = stamped_2;
                    last_solved_bits = rs_solved2;
                    iters = iters_2;
                    flipped = flipped_2;
                    mask_cycle = cycle_2;
                }
                if (keep_be) iters += iters_2 - 1;
                if (sx.on) [[unlikely]] {
                    if (!keep_be) {
                        sx.s.refuse_step(t_from + h, "an SDIRK step whose second "
                                         "stage changed the diode mode (a "
                                         "commutation not located in time)");
                    }
                    // stage 2 as taken: backward Euler of h_b from stage 1
                    step::StagePsi pb{Real{1} / h_b, Real{0}, Real{0}, 1, {}, {}};
                    pb.c_q[0] = Real{1};
                    sx.stage(1, pb, t_from + h, Real{2} * h_b);
                }
            } else if (sx.on) [[unlikely]] {
                sx.stage(1, step::psi_sdirk_stage2(dt_eq, c), t_from + h, dt_eq);
            }
            // F-Z-11 (the split): every other flip of the (sub)step is located
            // too: stage 1's (above), or a stage-2 change that is not the
            // turn-off the re-take confirms (a turn-on). No cut: the uncut
            // (sub)step is put back as solved.
            const bool s2_flip = split && rs_cut.arm && has_diodes && !s1_flip && !on_to_off &&
                                 iters_2 > 1 && t_from != opts.t_start &&
                                 !(rs_bits2 == rs_bits);
            // shortcut: not in the run's first (sub)step, where a zero start's
            // round-off flips a diode in stage 2 and back (the buck at t = 0:
            // 3e-8 V); upgrade if a real crossing there matters
            if ((s1_flip || s2_flip) && std::isnan(rs_cut.t) && !impulsive && !ip_jump &&
                !ln_pre && !sx.on) {
                rk_x = x;
                rk_b = b_extra;
                step::snapshot_into(history, hs, rk_h, rk_hs);
                diodes.snapshot_on_bits_into(rk_bits);
                rk_lsb = last_solved_bits;
                const auto stamped_k = stamped_mask;
                const Size iters_k = iters;
                const bool flipped_k = flipped, cycle_k = mask_cycle;
                const bool s2_moved = iters_2 > 1 && !(rs_bits2 == rs_bits) && !kept_be_2;
                rs_cut.t = restart_cut_at(run_event_iteration, t_from, h, s1_flip,
                                          s1_flip && s2_moved ? &rs_bits2 : nullptr);
                if (s1_flip && std::isnan(rs_cut.t) && s2_moved) {
                    // none from the start: stage 2's own change, as without stage 1's
                    rs_cut.t = restart_cut_at(run_event_iteration, t_from, h);
                }
                {   // the uncut (sub)step as solved, cut or not (restart_lined reads
                    // its settle state before it re-takes the pieces)
                    x = rk_x;
                    b_extra = rk_b;
                    step::restore(history, hs, rk_h, rk_hs);
                    diodes.restore_on_bits(rk_bits);
                    last_solved_bits = rk_lsb;
                    stamped_mask = stamped_k;
                    iters = iters_k;
                    flipped = flipped_k;
                    mask_cycle = cycle_k;
                }
            }
            // A diode flip inside the step is a discontinuity of its
            // own: the window covers this step and the next two.
            if (iters_1 > 1 || iters_2 > 1) {
                restart_left = std::max(restart_left, kRestartSteps - 1);
            }
            iters += iters_1 - 1;
            flipped = flipped || breach_1;
        };
        // run_restart, and the step's line for it (StepLines): at the
        // step start, its right limit; inside the step, the event's left
        // limit (the sub-step that ended at a) and its right limit. The
        // start of the line is 2·w1·X1 + (1 − 2·w1)·X2, w1 the stage-1
        // weight (SDIRK 1 − γ; the backward-Euler re-take γ). A stage 2
        // in another mode than stage 1 (F-4-4) is the same quadrature as
        // two held pieces, each a solved state in its own mode: X1 over
        // w1·h, then X2 (an extra row at a + w1·h, not an event).
        // ev: the (sub)step starts at an event (a trigger at the step
        // start, a scheduled or located split event): its right limit is
        // recorded (F-SB-1); a window step's only after a probed jump.
        auto restart_lined_once = [&](auto& run_event_iteration, Real a, Real h, Real loc, bool ev) {
            rs_piece_a = a;
            const bool inner = a != t_prev_step;
            StepLines& lines = *L;   // restart steps exist under the rule only
            if (lines.owner != t_step) {   // the step's first: no per-step reset
                lines.clear();
                lines.ok = true;
                lines.owner = t_step;
            }
            const Size j = lines.n;
            // F-4-4: the capacitor currents at the line points, for an
            // in-loop recorder only (StepLines::c_*)
            const auto& es = history.entries();
            const bool caps = hook_step && !es.empty();
            auto v_cap = [](const Vector& xv, const pwl::HistoryEntry& e) {
                return stamping::read_node_voltage(xv, e.from) -
                       stamping::read_node_voltage(xv, e.to);
            };
            if (inner) {
                if (j < StepLines::kMax) {
                    lines.t[j] = a;
                    lines.left[j] = x;
                    lines.m_left[j] = stamped_mask;
                    lines.right_ok[j] = false;
                    lines.n = j + 1;
                    if (caps) {   // the sub-step that ended at a, committed
                        lines.c_left[j].resize(es.size());
                        for (Size k = 0; k < es.size(); ++k) {
                            lines.c_left[j][k] = lines.end_own ? lines.c_end[k] : es[k].i_prev;
                        }
                    }
                } else {
                    lines.ok = false;
                }
            }
            if (caps) {
                lines.v0.resize(es.size());
                for (Size k = 0; k < es.size(); ++k) lines.v0[k] = es[k].v_prev;
            }
            lines.m_a = stamped_mask;   // the mode the (sub)step starts in
            lines.end_own = false;
            const auto mode_in = stamped_mask;
            const bool split_in = split && inner;   // (the split off: the step start, as before)
            if (split_in) mode_ev_x_scratch = x;
            run_restart(run_event_iteration, a, h, loc, ev);
            if (split_in && !(stamped_mask == mode_in)) {   // a split event changed the mode
                mode_ev_t = a;
                std::swap(mode_ev_x, mode_ev_x_scratch);
            }
            if (inner && j >= StepLines::kMax) return;
            if (inner) {   // the event's right limit (state value)
                lines.held[j] = false;
                lines.rl_right_jump[j] = ip_jump;
                lines.rl_right_ok[j] = lines.rl_ok;
                if (lines.rl_ok) {
                    if (lines.rl_left) {   // a continuity point: the left
                        lines.rl_right[j] = lines.left[j];
                    } else {
                        std::swap(lines.rl_right[j], lines.rl_x);
                    }
                    lines.rl_m_right[j] = lines.rl_m;
                }
            } else {
                lines.rl_start_ok = lines.rl_ok;
                lines.rl_start_jump = ip_jump;
                lines.rl_start_left = lines.rl_left;
                if (lines.rl_ok) {
                    std::swap(lines.rl_start, lines.rl_x);
                    lines.rl_m_start = lines.rl_m;
                }
            }
            // shortcut: after a kept backward-Euler stage 2 (w1 = γ) the
            // start lies between the stages: the quadrature's value, not
            // a state value (the result's right limit is rl_*, F-SB-1)
            const Real w1 = lines.be_kept ? kGamma : Real{1} - kGamma;
            const bool mixed = !(lines.m1 == stamped_mask);
            const Real t_m = a + w1 * h;
            // held pieces only when no device's stage-1 state is neither
            // side's (off -> on -> off would read as an on/off pair the
            // loss tracker never books)
            bool one_way = mixed;
            for (Size bit = 0; one_way && bit < lines.m1.size(); ++bit) {
                const bool s1 = lines.m1.get(bit);
                one_way = s1 == lines.m_a.get(bit) || s1 == stamped_mask.get(bit);
            }
            if (mixed && (!one_way || lines.n >= StepLines::kMax ||
                          !(t_m > a && t_m < a + h))) {
                // shortcut: after an ideal element's jump (ln_pre) these
                // ends read the run's i_prev, which folds the jump in
                // (needs a mixed stage 2 too); upgrade with F-W-1's
                // lines-only stage 2
                return;   // the step's ends
            }
            Vector& dst = inner ? lines.right[j] : lines.x_start;
            if (mixed) {
                dst = lines.x1;
            } else {
                dst = (Real{2} * w1) * lines.x1 + (Real{1} - Real{2} * w1) * x;
            }
            const auto& m_dst = mixed ? lines.m1 : stamped_mask;
            if (inner) {
                lines.m_right[j] = m_dst;
                lines.right_ok[j] = true;
            } else {
                lines.m_start = m_dst;
                lines.start = true;
            }
            const Size q = lines.n;   // the held pieces' row
            if (mixed) {
                lines.t[q] = t_m;
                lines.left[q] = lines.x1;
                lines.m_left[q] = lines.m1;
                lines.right[q] = x;
                lines.m_right[q] = stamped_mask;
                lines.right_ok[q] = true;
                lines.held[q] = true;   // quadrature only: no inner event
                lines.n = q + 1;
            }
            if (caps) {
                // the stage currents as the solves enforced them: stage 1
                // backward Euler of γh from v0, stage 2 the commit's
                // (x2 − x0*)/(γh), or (x2 − x1)/h_b after a kept BE re-take
                // (the histories hold x0* / x1 until the commit)
                constexpr Real kC2 = (Real{1} - kGamma) / kGamma;
                std::vector<Real>& c = inner ? lines.c_right[j] : lines.c_start;
                c.resize(es.size());
                if (ln_pre) {   // the (sub)step's end: the lines' stage 2, not i_prev
                    lines.c_end.resize(es.size());
                    for (Size k = 0; k < es.size(); ++k) lines.c_end[k] = es[k].i_prev;
                    lines.end_own = true;
                }
                if (mixed) {
                    lines.c_left[q].resize(es.size());
                    lines.c_right[q].resize(es.size());
                }
                for (Size k = 0; k < es.size(); ++k) {
                    const auto& e = es[k];
                    if (e.kind != pwl::DevicePool::StoredKind::Capacitor) continue;
                    const models::Capacitor::Params cp{e.C_or_L};
                    const Real i1 = models::Capacitor::g_eq(Real{2} * (kGamma * h), cp) *
                                    (v_cap(lines.x1, e) - lines.v0[k]);
                    // ln_pre: stage 2's start from the lines' stage 1 (x0+*)
                    const Real vb = !ln_pre ? e.v_prev
                        : lines.be_kept ? v_cap(lines.x1, e)
                        : lines.v0[k] + kC2 * (v_cap(lines.x1, e) - lines.v0[k]);
                    const Real i2 = models::Capacitor::g_eq(Real{2} * be_commit_h, cp) *
                                    (v_cap(x, e) - vb);
                    if (ln_pre) lines.c_end[k] = i2;
                    if (mixed) {
                        c[k] = lines.c_left[q][k] = i1;
                        lines.c_right[q][k] = i2;
                    } else {
                        c[k] = (Real{2} * w1) * i1 + (Real{1} - Real{2} * w1) * i2;
                    }
                }
            }
        };
        // restart_lined_once, cut at a turn-off run_restart located inside
        // its stage 2 (restart_cut_at): back to the (sub)step start, the
        // piece up to the event in the old mode (not armed again), its
        // commit, the new mode's bits, and the rest restarted from the
        // event as a located one (split_step's sequence). The last piece's
        // commit stays the caller's.
        // shortcut: one cut per call (a second turn-off in the rest is
        // SDIRK's stage 2, as before); upgrade if two independent turn-offs
        // inside one step matter
        auto restart_lined = [&](auto& run_event_iteration, Real a, Real h, Real loc, bool ev) {
            rs_cut.arm = has_diodes && !sx.on;
            rs_cut.t = std::numeric_limits<Real>::quiet_NaN();
            if (!rs_cut.arm) {
                restart_lined_once(run_event_iteration, a, h, loc, ev);
                return;
            }
            StepLines& lines = *L;
            rs_cut.x0 = x;
            step::snapshot_into(history, hs, rs_cut.h0, rs_cut.hs0);
            diodes.snapshot_on_bits_into(rs_cut.b0);
            rs_cut.lsb0 = last_solved_bits;
            const bool cycle0 = mask_cycle, known0 = stamped_known, flipped0 = flipped;
            const auto mask0 = stamped_mask;
            const Size ip0 = impulse_pending.size(), fb0 = result.n_right_limit_fallbacks;
            const Size ij0 = ideal_pending.size();
            const Real erl0 = event_rl_t;
            const Real mev0 = mode_ev_t;
            Vector mevx0;
            if (!std::isnan(mev0)) mevx0 = mode_ev_x;
            const Real rlt0 = rl_loc.t;   // the only field consumed (event_right_limit)
            const Real owner0 = lines.owner;
            const Size n0 = lines.n;
            const bool eo0 = lines.end_own, st0 = lines.start, ok0 = lines.ok;
            const bool rso0 = lines.rl_start_ok, rsj0 = lines.rl_start_jump;
            try {
                restart_lined_once(run_event_iteration, a, h, loc, ev);
            } catch (...) {
                rs_cut.arm = false;   // the dt-retry ladder calls run_restart unarmed
                throw;
            }
            rs_cut.arm = false;
            if (std::isnan(rs_cut.t)) return;
            const Real td = rs_cut.t;
            const auto post = rs_cut.bits;
            const bool vf0 = rs_cut.vf0;
            const bool unsettled_uncut = mask_cycle || flipped;
            // the counters keep the uncut pass's work (the pieces add theirs)
            const Size s2r = result.stage2_retries, s2k = result.stage2_be_kept,
                       s2f = result.stage2_retry_failures;
            auto restore0 = [&]() {
                flipped = flipped0;
                x = rs_cut.x0;
                step::restore(history, hs, rs_cut.h0, rs_cut.hs0);
                diodes.restore_on_bits(rs_cut.b0);
                last_solved_bits = rs_cut.lsb0;
                mask_cycle = cycle0;
                stamped_known = known0;
                stamped_mask = mask0;
                impulse_pending.resize(ip0);
                ideal_pending.resize(ij0);
                result.n_right_limit_fallbacks = fb0;
                event_rl_t = erl0;
                mode_ev_t = mev0;
                if (!std::isnan(mev0)) mode_ev_x = mevx0;
                rl_loc.t = rlt0;
                lines.owner = owner0;
                lines.n = n0;
                lines.end_own = eo0;
                lines.start = st0;
                lines.ok = ok0;
                lines.rl_start_ok = rso0;
                lines.rl_start_jump = rsj0;
            };
            restore0();
            rs_cut.piece1 = true;
            try {
                restart_lined_once(run_event_iteration, a, td - a, loc, ev);
            } catch (...) {
                rs_cut.piece1 = false;
                throw;
            }
            rs_cut.piece1 = false;
            const bool flipped_1 = flipped;
            commit_backward_euler(be_commit_h);
            diodes.restore_on_bits(post);
            // the event's right limit: a located turn-off (split_step's rl_loc)
            rl_loc.t = td;
            rl_loc.at_b = false;
            rl_loc.off = rs_cut.off;
            rl_loc.vf0 = vf0;
            rl_loc.pure = vf0;
            rl_loc.bits = post;
            restart_lined_once(run_event_iteration, td, (a + h) - td, snap, true);
            flipped = flipped || flipped_1;
            const bool uncut = (mask_cycle || flipped) && !unsettled_uncut;
            if (!uncut) ++result.n_restart_cuts;   // counted once the cut is kept (a throw keeps none)
            if (!uncut && split) {   // the located flips, at the cut (split_step's records)
                const auto ents = diodes.entries();
                for (Size j = 0; j < ents.size(); ++j) {
                    if (post[j] == rs_cut.pre[j]) continue;
                    // only a net flip of the step that piece 2 kept: a cut back to the
                    // step-start mode (on in stage 1, located off) or a flip piece 2
                    // reverted would be an unpaired record (cl_boost_pi: 36 offs with
                    // no on) and a phantom sensitivity jump; the step end decides those
                    // shortcut: the short pulse inside the step goes unrecorded (as before
                    // F-Z-11); record both edges if a consumer needs it
                    if (post[j] == rs_cut.b0[j] || ents[j].is_on != post[j]) continue;
                    split_events.push_back(CommutationEvent{
                        .t_estimated = td, .branch_id = ents[j].branch_id, .new_state = post[j]});
                }
            }
            if (uncut) {
                // a piece left the diode mode unsettled where the whole step
                // settled (the three-phase rectifier's mask cycle at 2.55 ms):
                // the uncut step stands
                restore0();
                restart_lined_once(run_event_iteration, a, h, loc, ev);
                result.stage2_retries = s2r;   // the same pass again: counted once
                result.stage2_be_kept = s2k;
                result.stage2_retry_failures = s2f;
            }
        };

        auto step_take = [&]() PULSIM_STEP_PHASE {
            const Real t = t_step;
            const Real t_prev = t_prev_step;
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
            iters = 0;
            flipped = false;
            mask_cycle = false;
            // Full mask copies (Phase 1: masks can exceed 64 bits; a
            // copy is 2 inline words for <=128 switches — still no
            // heap in the loop).
            masks_seen.reset();

            // add-newton-roundoff-acceptance: a dynamic-path Newton
            // solve that exhausts its budget at row roundoff is
            // accepted and counted (DC and static paths: never).
            // One object for the run: its counters are the run budget.
            pwl::NewtonRoundoffAcceptance& newton_roundoff = newton_roundoff_run;
            auto note_roundoff = [&]() {
                if (!newton_roundoff.accepted) return;
                ++result.newton_roundoff_accepts;
                if (newton_roundoff.omega
                    >= result.newton_roundoff_omega_max) {
                    result.newton_roundoff_omega_max = newton_roundoff.omega;
                    result.newton_roundoff_row = newton_roundoff.row;
                }
            };

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
            // The linear solve of a site under `mask` (the loop's own branches).
            auto mode_solve = [&](const topology::SwitchStateMask& mask,
                                  Real for_dt, bool be, Vector& out) {
                if (restart_rule && (for_dt == restart_dt || be)) {
                    const auto m0 = cache.metrics();
                    cache.solve_at(mask, for_dt, b_extra, out);   // solve-point
                    const auto m1 = cache.metrics();
                    restart_factorizations += static_cast<Size>(
                        (m1.event_builds - m0.event_builds) +
                        (m1.event_refactors - m0.event_refactors));
                } else if (for_dt == opts.dt && for_dt == cache.dt()) {
                    cache.solve_segment(nominal_segment(mask), b_extra, out);   // solve-point
                } else {
                    cache.solve_at(mask, for_dt, b_extra, out);   // solve-point
                }
            };

            auto run_event_iteration = [&](Real at_t, Real for_dt,
                                           bool backward_euler = false,
                                           bool single = false) {
            // an earlier stage's cycle of this outer step: not this site's breach
            const bool cycle_in = mask_cycle;
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
            step_ctx = backward_euler ? step::BackwardEuler::on_factor(for_dt).ctx()
                                      : step::Trapezoidal::stage(for_dt).ctx();
            step_ctx.t = at_t;   // the external devices' ctx->t
            iters = 0;
            flipped = false;
            // NOT reset: a mask cycle in ANY sub-step of a retry is a
            // breach of the whole outer step. Clearing it here would
            // let every sub-step but the last vanish from
            // `event_iteration_breaches` — and would let
            // `strict_event_iterations` be violated in silence, which
            // is the opposite of what that flag is for.
            masks_seen.reset();
            // Diode on-bits consistent with the most recent solve
            // (captured BEFORE update_from_state advances them) —
            // restored on breach so x and the diode state agree
            // (adversarial-review finding P0-R1).
            last_solved_bits.clear();
            loop_released.clear();
            // add-complementarity-mode-selection: every non-single linear
            // site is a decision site of the selector.
            const bool mode_site = mode_ports.has_value() && !single &&
                                   !nl_refresh_effective;
            if (mode_site) mode_begin();
            if (mode_site && opts.mode_selection == 2) {
                const topology::SwitchStateMask gates =
                    gate_at(restart_rule ? gate_t : at_t);
                auto solve = [&](const topology::SwitchStateMask& m, Vector& out) {
                    // the settle's structural pass before a mode's solve: a
                    // floating node / source loop is refused by its name, not
                    // as a singular factor (F-4-1: one classification)
                    if (structure.active()) {
                        (void)structure.check(m, for_dt > Real{0});
                        result.mode_structure_checks = structure.checks_run();
                    }
                    mode_solve(m, for_dt, backward_euler, out);
                };
                if (capture_first) {
                    // the diode split locates the old mode's crossing on its
                    // own solve, as the settle's first round gives it
                    mask_first = combine_masks(gates, diodes.current_diode_mask(),
                                               diode_owned);
                    solve(mask_first, x_first);
                }
                const bool changed = mode_lcp_site(
                    at_t, mode_kind(for_dt, backward_euler), gates, for_dt,
                    backward_euler, b_extra, solve, capture_first ? &x_first : nullptr);
                // A changed mode re-arms the restart exactly as a settle
                // flip does (callers read iters > 1).
                iters = changed ? 2 : 1;
                flipped = false;
                return;
            }
            // The settle loop stays a do-while here: events::settle (or a
            // counted variant) in this loop cost 1.0-1.8 % instructions per
            // step (add-unified-event-queue DECISIONS 15).
            do {
                auto mask = gate_at(restart_rule ? gate_t : at_t);
                if (has_diodes) {
                    mask = diodes.combine(mask);
                }
                bool released = false;
                while (structure.active() && check_mode(mask, true, at_t)) {
                    released = true;
                    mask = diodes.combine(gate_at(restart_rule ? gate_t : at_t));
                }
                // A release back onto a mask already solved this step is the
                // loop forward-biasing the diode: refuse rather than let the
                // cycle detector turn it into a breach (review H1).
                if (released && !nl_refresh_effective && masks_seen.seen(mask)) {
                    const Index d = loop_released.back();
                    throw UnsupportedError(
                        "pwl", "ideal_voltage_loop",
                        std::string(graph.branch_name(d)), "Diode",
                        std::format(
                            "engine 'pwl': at t = {} the ideal diode '{}' (R_on "
                            "= 0) closes a loop of voltage sources and closed "
                            "ideal elements (loop: {}) that forward-biases it "
                            "(turned OFF, it turned back ON): its current is "
                            "undefined. Give it R_on > 0.",
                            at_t, graph.branch_name(d), loop_last));
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
                // A cycle of THIS call (`masks_seen` is reset per call):
                // `mask_cycle` stays set from an earlier stage or sub-step
                // of the outer step, and breaking on it skipped this
                // stage's solve (x stayed at the earlier stage's time,
                // committed as this one).
                if (!nl_refresh_effective && masks_seen.repeat(mask)) {
                    mask_cycle = true;
                    break;
                }
                // The solve point of the event iteration (refactor-unified-
                // transient-stepper group 2, DECISIONS 22): x under `mask`.
                // Written out here, not as a dispatch lambda: the captured
                // closure cost ~1 % instructions per step (DECISIONS 22a).
                if (nl_refresh_effective) {
                    // Newton, warm-started from x, on the nominal segment or
                    // (off-nominal dt) a companion assembled for this step —
                    // no factorization is wasted: Newton refactorizes
                    // J_lin + J_nl every iteration, the segment only carries
                    // J and b.
                    const pwl::PwlSegment* nl_seg = &retry_seg;
                    if (for_dt == opts.dt && !newton_only_rows) {
                        nl_seg = &nominal_segment(mask);
                    } else {
                        pwl::assemble_segment(
                            graph, pool, mask, for_dt,
                            retry_seg.J, retry_seg.b_constant);
                        sparse::compress_in_place(retry_seg.J);
                        retry_seg.state_size = state_size;
                    }
                    sens_J_lin = &nl_seg->J;
                    x = pwl::solve_with_newton_b_extra(   // solve-point
                        *nl_seg, nl_refresh_effective, graph, pool,
                        /*x_init=*/x, b_extra,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search,
                        opts.enable_newton_lm,
                        &newton_ws, &newton_roundoff);
                    note_roundoff();
                } else {
                    // `solve_at` delegates to `solve` bit-identically
                    // when the dt matches, so the nominal path costs
                    // nothing extra.
                    // Every restart stage counts, the split's sub-interval
                    // stages (dt_eq = 2γ·h_sub) included (review L8b).
                    if (restart_rule && (for_dt == restart_dt || backward_euler)) {
                        const auto m0 = cache.metrics();
                        cache.solve_at(mask, for_dt, b_extra, x);   // solve-point
                        const auto m1 = cache.metrics();
                        restart_factorizations += static_cast<Size>(
                            (m1.event_builds - m0.event_builds) +
                            (m1.event_refactors - m0.event_refactors));
                    } else if (for_dt == opts.dt && for_dt == cache.dt()) {
                        // == solve_at's delegation to solve(mask, …)
                        cache.solve_segment(nominal_segment(mask), b_extra, x);   // solve-point
                    } else {
                        cache.solve_at(mask, for_dt, b_extra, x);   // solve-point
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
                flipped = !single && has_diodes && events::redecide(diodes, x);
                ++iters;
            } while (flipped && iters < max_iters);
            if (mode_site) {
                const topology::SwitchStateMask gates =
                    gate_at(restart_rule ? gate_t : at_t);
                if (opts.mode_selection == 1) {
                    mode_shadow_site(at_t, mode_kind(for_dt, backward_euler), gates,
                                     for_dt, backward_euler, b_extra, iters, mask_cycle);
                }
                if (result.mode_guard.enabled) {
                    ++result.mode_guard.checks;
                    const bool breach = flipped || (mask_cycle && !cycle_in);
                    if (breach) ++result.mode_guard.settle_caps;
                    if (mask_cycle && !cycle_in) ++result.mode_guard.lcp.mask_cycles;
                    result.mode_guard.lcp.settle_rounds_total += iters;
                    if (breach &&
                        mode_guard_site(at_t, gates, for_dt, backward_euler, b_extra,
                                        [&](const topology::SwitchStateMask& m, Vector& out) {
                                            mode_solve(m, for_dt, backward_euler, out);
                                        })) {
                        flipped = false;
                        mask_cycle = cycle_in;   // an earlier stage's breach stays recorded
                    }
                }
            }
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
            committed_dt = opts.dt;
            step_retried = false;
            mask_cycle = false;    // per OUTER step, not per sub-step
            // a gated device's record compares with the last OUTER step's
            // stamp: restart stages and split sub-steps re-stamp in between
            if (diodes.num_gated() > 0) gated_prev_stamp = stamped_mask;
            if (has_ideal_diode) ideal_prev_stamp = stamped_mask;

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
            // a gate edge at the step start (stamps its forced flips there):
            // the gate bits changed, the diode bits read as committed
            edge_restart_step =
                restart_rule && stamped_known && has_ideal_diode &&
                !(mask_pre == combine_masks(stamped_mask, diodes.current_diode_mask(),
                                            diode_owned));
            rs_first_valid = false;
            if (restart_rule &&
                (!stamped_known || !(mask_pre == stamped_mask) ||
                 (has_jump_sources &&
                  (split ? sources::source_jump_in(pool, graph, t_prev - snap,
                                                   t_prev + snap)
                         : sources::source_jump_in(pool, graph, t_prev, t))) ||
                 (has_b_jumps &&
                  (split ? b_jump_after(t_prev - snap) < t_prev + snap
                         : b_jump_after(t_prev - left_by) < t - left_by)))) {
                restart_left = kRestartSteps;
                rs_trig_t = t_prev;
                // t = 0: a probed jump only; a resume seam is an event (D7)
                rs_trig_ev = stamped_known && (t_prev != opts.t_start || resume_snap != nullptr);
                rs_trig_dio = stamped_known && !has_jump_sources && !has_b_jumps &&
                              combine_masks(mask_pre, stamped_mask, diode_owned_ungated) == stamped_mask;
                // without the split a built-in source / b_extra jump inside
                // the step is solved from t_prev: δt_loc is its offset
                // (a stage reads it only up to t_prev + γh)
                rs_trig_loc = 0;
                if (!split) {
                    Real tj = t;
                    if (has_jump_sources) {
                        tj = std::min(tj, sources::next_source_jump_after(
                                              pool, graph, t_prev - left_by));
                    }
                    if (has_b_jumps) tj = std::min(tj, b_jump_after(t_prev - left_by));
                    if (tj < t) {
                        rs_trig_loc = std::clamp(tj - t_prev, Real{0}, kGamma * opts.dt);
                    }
                }
            }
            restart = restart_left > 0;
            if (restart) --restart_left;
            // The final solve of this step was a restart stage, so the
            // deferred commit below is a backward-Euler one (at γh).
            committed_be = false;
            be_commit_h = kGamma * opts.dt;   // set by run_restart
            // (A split step accumulates per sub-interval: no discarded
            // b_extra_fn call at (t, dt).)
            if (!restart && n_ev == 0) {
                accumulate_b_extra(t, opts.dt);
            }
            try {
                if (split) {
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
                        step::commit(history, hs, x, h, pwl::TrBdf2Stage::Trapezoidal);
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
                                gate_t = ev_s[i - 1];   // post side of the cluster's last edge
                                any_split = true;
                                if (sx.on) [[unlikely]] sx.gate_event(a, gate_t, !ev_overflow);
                            }
                            if (be) {
                                // a scheduled event (merged within snap) or the step's
                                // trigger: time events; a window step: not located
                                restart_lined(run_event_iteration, a, h,
                                            i > 0 ? ev_s[i - 1] - ev_t[i - 1]
                                            : rs_trig_t == t_prev ? rs_trig_loc
                                            : anchor_t == t_prev  ? Real{0}
                                                                  : kGamma * h,
                                            i > 0 || (rs_trig_t == t_prev && rs_trig_ev));
                            } else {
                                if (i > 0 || h != opts.dt) {
                                    accumulate_b_extra(b, h);
                                }   // else: accumulated at (t, dt) above
                                if (sx.on) [[unlikely]] sx.begin(a, x);
                                capture_first = has_diodes;
                                run_event_iteration(b, h);
                                capture_first = false;
                                if (sx.on && iters == 1) [[unlikely]] {
                                    sx.stage(0, step::psi_trapezoidal(h), b, h);
                                }
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
                                        const Real v1 = e.law_voltage(x_first, on);
                                        if (models::SwitchedDiode::decide(
                                                on, v1, e.params()) == on) {
                                            continue;
                                        }
                                        const Real v0 = e.law_voltage(x_prev, on);
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
                                    // F-3-1: the event is the commutation of the
                                    // DECIDED post-event mode (the event iteration's
                                    // final bits). Diodes it flips that touch the same
                                    // FLOATING island — a node set joined by every
                                    // branch but the OFF diodes, with no path to
                                    // ground — share one signal, the sum of theirs:
                                    // the island's common-mode potential is set by the
                                    // OFF diodes' leakage alone, so each one's voltage
                                    // is a leakage ratio (one ulp of C moved the first
                                    // single crossing of a floating bridge by 1e-7 s)
                                    // while the sum around the conducting path is the
                                    // circuit's. Every other diode keeps its own
                                    // crossing.
                                    locate_islands(run_event_iteration, a, b, h, t_d);
                                    // Back to the step start, then: the crossing
                                    // at a (an SDIRK step, as the rule alone), at
                                    // b (keep the pre-event solve, flip there) or
                                    // inside (split).
                                    x = x_prev;
                                    step::restore(history, hs, history_snap, hs_snap);
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
                                    // F-SB-1: a located crossing, for its right limit
                                    // (not one that lands on a scheduled event: !last).
                                    // A continuity point when only V_f = 0 diodes flip
                                    // (their ON and OFF laws meet at i = 0, v = 0) and
                                    // no trigger coincides: within snap of a, only at
                                    // the step start without a trigger there; at b,
                                    // checked by the next step (event_right_limit)
                                    {
                                        auto& lc = rl_loc;
                                        lc.vf0 = true;
                                        lc.off = false;
                                        for (Size j = 0; j < ents.size(); ++j) {
                                            // a turn-on at v_D = V_th > 0 jumps by
                                            // (g_on − g_off)·V_th: not a meeting point
                                            if (post_event_bits[j] != diodes_snap[j] && !post_event_bits[j]) {
                                                lc.off = true;
                                            }
                                            // a gated latching device (H2/H3): its turn-off at
                                            // I_H and a gate-fired turn-on are no meeting point
                                            if (post_event_bits[j] != diodes_snap[j] &&
                                                (ents[j].V_f != Real{0} ||
                                                 ents[j].gate != models::SwitchedDiode::Gate::None ||
                                                 (post_event_bits[j] && ents[j].V_th != Real{0}))) {
                                                lc.vf0 = false;
                                            }
                                        }
                                        lc.bits = post_event_bits;
                                        lc.at_b = t_d - a >= snap && b - t_d < snap;
                                        lc.t = t_d - a < snap ? a
                                               : !lc.at_b ? t_d
                                               : last ? b
                                                      : std::numeric_limits<Real>::quiet_NaN();
                                        lc.pure = lc.vf0 &&
                                                  (t_d - a >= snap ||
                                                   (a == t_prev && !(rs_trig_t == t_prev && rs_trig_ev)));
                                    }
                                    if (t_d - a < snap) {
                                        if (sx.on) [[unlikely]] sx.diode_events(a, h);
                                        split_events.clear();
                                        restart = true;
                                        be = true;
                                        restart_left = std::max(restart_left,
                                                                kRestartSteps - 1);
                                        restart_lined(run_event_iteration, a, h, snap, true);   // crossing within snap of a
                                    } else if (b - t_d < snap) {
                                        x = x_first;
                                        stamped_mask = mask_first;
                                        stamped_known = true;
                                        if (sx.on) [[unlikely]] {
                                            sx.stage(0, step::psi_trapezoidal(h), b, h);
                                            sx.diode_events(b, h);
                                        }
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
                                        if (sx.on) [[unlikely]] {
                                            sx.stage(0, step::psi_trapezoidal(h1), t_d, h1);
                                            sx.s.commit(t_d, false);
                                            sx.diode_events(t_d, h1);
                                        }
                                        commit_trap(h1);
                                        diodes.restore_on_bits(post_event_bits);
                                        h = b - t_d;
                                        restart_lined(run_event_iteration, t_d, h, snap, true);   // located to snap
                                        be = true;
                                        any_split = true;
                                    }
                                }
                            }
                            breach = breach || flipped;
                            if (!last && sx.on) [[unlikely]] sx.s.commit(b, false);
                            if (!last) {
                                if (be) {
                                    commit_backward_euler(be_commit_h);
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
                        // F-Z-3: a last piece that starts at an event inside the
                        // step (s = t − rs_piece_a < dt) projected only the modes
                        // faster than γs (a 1 ps mode after a 1e-14 s piece rang
                        // ±2.25 A on). The next grid step is a restart whose probe
                        // reads what is left as a jump at its start (δt_loc = 0:
                        // the motion since the event is already in x; no jump
                        // found: the plain restart, bit for bit on a linear
                        // circuit -- with Newton devices δt_loc < h_s also runs
                        // decide_edge_mode, which a window step skips).
                        anchor_t = std::numeric_limits<Real>::quiet_NaN();
                        // Only below γ·dt (the restart's own stage 1): a longer piece's
                        // stage 1 already spans γ²·dt, and the probe would record the
                        // rest of a resolved collapse as a jump at t (a 50 ns junction
                        // capacitor after a 0.5·dt piece: a second right-limit row).
                        if (committed_be && rs_piece_a > t_prev && t - rs_piece_a < kGamma * opts.dt) {
                            anchor_t = t;
                            restart_left = std::max(restart_left, Size{1});
                        }
                        flipped = breach || ev_overflow;
                    };
                    split_step();
                } else {
                if (!restart) {
                    run_event_iteration(t, opts.dt);
                    if (restart_rule && iters > 1) {
                        // A diode flipped inside a trapezoidal step:
                        // that is a discontinuity too. Back to the step
                        // start, and re-take it as a restart step.
                        x = x_prev;
                        step::restore(history, hs, history_snap, hs_snap);
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
                    restart_lined(run_event_iteration, t_prev, opts.dt,
                                rs_trig_t == t_prev ? rs_trig_loc : kGamma * opts.dt,
                                rs_trig_t == t_prev && rs_trig_ev);
                    committed_be = true;
                }
                }   // !split
            } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                throw;   // a run budget: never retried at a smaller dt
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
                bool why_singular = singular_in_flight();   // -> SingularError below
                // `Size{1} << h` is undefined for h >= 64 and
                // pointless long before that: 2^25 sub-steps of one
                // outer step is not a recovery, it is a hang. The
                // option is read/write from Python and unchecked by
                // SimulationOptions::valid(), so clamp at the point
                // of use.
                const Size halving_cap =
                    step::HalvingLadder::cap(opts.max_dt_halvings);
                for (Size h = 1; h <= halving_cap; ++h) {
                    // Back to the state at t_prev. Note what is NOT
                    // restored: `sat_history`, which this step has
                    // not touched yet (it commits at the very end),
                    // and the cache, whose contents are a function of
                    // the circuit rather than of the run.
                    x = x_prev;
                    step::restore(history, hs, history_snap, hs_snap);
                    if (has_diodes) {
                        diodes.restore_on_bits(diodes_snap);
                    }
                    // shortcut: the retry ignores events inside the step
                    // (the gate keeps the step's own sample, 0.5001 h back, as the
                    // restart rule alone; a built-in source jump inside it
                    // is read where the sub-steps fall and not restarted);
                    // retries are rare.
                    gate_t = gate_t0;
                    capture_first = false;
                    // shortcut: a retried step keeps no lines (its ends
                    // only); retries are rare
                    if (L) {
                        L->clear();
                        L->ok = false;
                        L->owner = t_step;
                    }
                    split_events.clear();
                    mode_ev_t = std::numeric_limits<Real>::quiet_NaN();
                    rs_first_valid = false;
                    impulse_pending.clear();
                    ideal_pending.clear();
                    event_rl_t = std::numeric_limits<Real>::quiet_NaN();

                    const Size n_sub = step::HalvingLadder::n_sub(h);
                    const Real sub_dt = step::HalvingLadder::sub_dt(opts.dt, h);
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
                                run_restart(run_event_iteration, t_sub - sub_dt, sub_dt,
                                            j == 0 && rs_trig_t == t_prev
                                                ? std::min(rs_trig_loc, kGamma * sub_dt)
                                                : kGamma * sub_dt,
                                            false);
                            } else {
                                run_event_iteration(t_sub, sub_dt);
                            }
                        } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                            throw;   // a run budget: never retried at a smaller dt
                        } catch (const std::runtime_error& sub_failed) {
                            why = sub_failed.what();
                            why_singular = singular_in_flight();
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
                            commit_backward_euler(be_commit_h);
                        } else if (j + 1 < n_sub) {
                            step::commit(history, hs, x, sub_dt, pwl::TrBdf2Stage::Trapezoidal);
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
                    std::string failure = std::format(
                        "run_transient: the step ending at t = {} "
                        "could not be taken, even split into {} "
                        "sub-steps of {:.3e} s (dt/{}). Last failure: "
                        "{}\nThis is no longer a step-size problem — "
                        "raise max_dt_halvings only if you believe "
                        "otherwise; more likely the circuit needs a "
                        "snubber, a softer device model, or a look at "
                        "the device the message above names.{}",
                        t, step::HalvingLadder::n_sub(halving_cap),
                        step::HalvingLadder::sub_dt(opts.dt, halving_cap),
                        step::HalvingLadder::n_sub(halving_cap), why,
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
                            : "");
                    // a singular matrix stays typed through the retries
                    if (why_singular) throw SingularError(failure);
                    throw std::runtime_error(failure);
                }
            }
            if (sens.on()) [[unlikely]] {
                // the event and restart paths fed their stages where they
                // were solved (and opened the step); a plain step is one
                // trapezoidal stage of dt, opened here
                if (step_retried || flipped || mask_cycle || (split && ev_overflow)) {
                    sens.refuse_step(t, step_retried ? "retried at a smaller dt"
                                     : (split && ev_overflow) ? "split at more events "
                                       "than it locates"
                                     : "left with an unsettled diode mode (an "
                                       "event-iteration breach)");
                }
                if (!sens_inline) {
                    sens_begin(t_prev, x_prev);
                    sens_stage(0, step::psi_trapezoidal(opts.dt), t, opts.dt);
                }
                sens_inline = false;
            }
        };  // step_take

        auto step_after_solve = [&]() PULSIM_STEP_PHASE {
            const Real t = t_step;
            const Real t_prev = t_prev_step;
            const bool edge_restart = edge_restart_step;
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
                for (std::size_t j = 0; j < diodes.entries().size(); ++j) {
                    const auto& e = diodes.entries()[j];
                    if (split && std::any_of(
                            split_events.begin(), split_events.end(),
                            [&](const CommutationEvent& ev) {
                                return ev.branch_id == e.branch_id;
                            })) {
                        continue;
                    }
                    if (e.gate != models::SwitchedDiode::Gate::None) {
                        // A gate decides, not v - V_on: record the stamped
                        // state's change, effective from the step start.
                        const bool now = stamped_mask.get(e.switch_idx);
                        if (now != gated_prev_stamp.get(e.switch_idx)) {
                            step_events.push_back(CommutationEvent{
                                .t_estimated = t_prev,
                                .branch_id   = e.branch_id,
                                .new_state   = now,
                            });
                        }
                        continue;
                    }
                    if (has_ideal_diode && pool.is_branch_form(e.branch_id)) {
                        // A flip at a gate edge is booked at the edge (t_prev)
                        // when the restart's first stage made it, else at the
                        // step end; any other flip as the conductance form
                        // (branch_form_event_time).
                        const bool on_prev = static_cast<bool>(commut_on_prev[j]);
                        if (on_prev != e.is_on) {
                            Real t_ev = t;
                            if (gate_edge_since(ideal_prev_stamp)) {
                                if (edge_restart && rs_first_valid &&
                                    rs_bits_first[j] == e.is_on) {
                                    t_ev = t_prev;
                                }
                            } else {
                                t_ev = branch_form_event_time(
                                    on_prev, t_prev, t, branch_form_s(e, x_prev),
                                    branch_form_s(e, x));
                            }
                            if (!std::isnan(t_ev)) {
                                step_events.push_back(CommutationEvent{
                                    .t_estimated = t_ev,
                                    .branch_id = e.branch_id, .new_state = e.is_on});
                            }
                        }
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
        };  // step_after_solve

        auto step_commit = [&]() PULSIM_STEP_PHASE {
            const Real t = t_step;
            const Real t_prev = t_prev_step;
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
                    const Real i_solved = x[e.inductor_branch_var_id];
                    // Step-to-step jump guard (snap back to the pre-step
                    // current on an unphysical kilo-amp delta: rectifier in
                    // DCM), then the absolute clamp (the slow-drift form of
                    // the same failure): step::GuardPolicy.
                    const auto guard = step::GuardPolicy::apply(
                        i_prev_step, i_solved, opts.inductor_freeze_di_max,
                        opts.inductor_abs_clamp);
                    const Real i_new = guard.i;
                    const bool froze = guard.froze;
                    const bool clamped = guard.clamped;
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
                commit_backward_euler(be_commit_h);
            } else {
            step::commit(history, hs, x, committed_dt, pwl::TrBdf2Stage::Trapezoidal);
            }
            if (restart) {
                ++result.n_restart_steps;
            }
            // add-in-kernel-event-losses: the step is accepted; read its
            // commutations and conduction energy (reading rule).
            if (lt) lt->accept(t_prev, t, x, stamped_mask,
                               L && L->owner == t && L->ok ? L : nullptr);
            if (lines) lines->commit(t, x);   // the one write of the line past (D18)
            if (sens.on()) [[unlikely]] sens.commit(t);
        };  // step_commit

        // An ideal element's jump J beside a fast mode it excites (C1 −S− C2,
        // 1 mΩ to C3: the share 2.381 nJ, then the collapse 23.23 nJ of the
        // 25.61): the tracker recorded the share J_i, the minimum-energy
        // projection onto the structures of the mode the step ends in; the rest
        // J_f = J − J_i is recorded here, dW = ½·J_fᵀM·J_f (J_iᵀM·J_f = 0: J_f
        // keeps every constraint of that mode). J_i jointly over all of them
        // (structures sharing a state constrain each other: an ideal clamp
        // diode across C2 as C1 −S− C2 closes). Nothing below 1e-6 of ½·JᵀM·J
        // (the series' J_i is exact to its fit: no fast mode).
        auto record_ideal_fast = [&](const IdealJump& ij, long sample) {
            const auto& ents = history.entries();
            const Size ne = ents.size();
            if (ij.j.size() != ne) return;
            const auto& deps = structure.check(stamped_mask, true).dependent;
            const auto nd = static_cast<Index>(deps.size()), nk = static_cast<Index>(ne);
            DenseMatrix kk = DenseMatrix::Zero(nd, nk);
            Vector mi(nk), jv(nk);
            for (Size k = 0; k < ne; ++k) {
                mi[static_cast<Index>(k)] = Real{1} / ents[k].C_or_L;
                jv[static_cast<Index>(k)] = ij.j[k];
            }
            for (Index g = 0; g < nd; ++g) {
                const auto& d = deps[static_cast<Size>(g)];
                for (Size q = 0; q < d.states.size(); ++q) {
                    for (Size k = 0; k < ne; ++k) {
                        if (ents[k].branch_id == d.states[q]) {
                            kk(g, static_cast<Index>(k)) += static_cast<Real>(d.signs[q]);
                        }
                    }
                }
            }
            Vector jf = jv;
            if (nd > 0) {
                const DenseMatrix a = kk * mi.asDiagonal() * kk.transpose();
                const Vector lam = a.completeOrthogonalDecomposition().solve(kk * jv);
                jf -= mi.cwiseProduct(kk.transpose() * lam);
            }
            if (!jf.allFinite()) return;
            Real e_f = 0, e_all = 0, dom_e = -1;
            Size dom = 0;
            for (Size k = 0; k < ne; ++k) {
                if (ij.move[k]) continue;   // a move: no loss
                const auto kq = static_cast<Index>(k);
                const Real ek = Real{0.5} * ents[k].C_or_L * jf[kq] * jf[kq];
                e_f += ek;
                e_all += Real{0.5} * ents[k].C_or_L * jv[kq] * jv[kq];
                if (ek > dom_e) {
                    dom_e = ek;
                    dom = k;
                }
            }
            for (const auto& tc : history.transformer_couplings()) {
                if (ij.move[tc.p_entry_idx] || ij.move[tc.s_entry_idx]) continue;
                const Real m = models::TwoWindingTransformer::mutual_inductance(tc.params);
                e_f += m * jf[static_cast<Index>(tc.p_entry_idx)] * jf[static_cast<Index>(tc.s_entry_idx)];
                e_all += m * jv[static_cast<Index>(tc.p_entry_idx)] * jv[static_cast<Index>(tc.s_entry_idx)];
            }
            if (!(e_f > Real{1e-6} * e_all)) return;
            solver::ReinitRecord rec;
            rec.t = ij.t;
            rec.kind = ents[dom].kind == pwl::DevicePool::StoredKind::Capacitor ? "capacitor_loop"
                                                                                : "cut_set";
            for (Size k = 0; k < ne; ++k) {
                const auto kq = static_cast<Index>(k);
                if (!ij.move[k] && Real{0.5} * ents[k].C_or_L * jf[kq] * jf[kq] > Real{1e-6} * dom_e) {
                    rec.states.emplace_back(graph.branch_name(ents[k].branch_id));
                }
            }
            rec.lambda = ents[dom].C_or_L * jf[static_cast<Index>(dom)];
            rec.dW = e_f;
            for (const auto& d : deps) {   // the ideal elements whose jump excited it
                for (const Index el : d.elements) {
                    const std::string nm(graph.branch_name(el));
                    if (std::find(rec.devices.begin(), rec.devices.end(), nm) == rec.devices.end()) {
                        rec.devices.push_back(nm);
                    }
                }
            }
            // the cause: the tracker's (its record of this event, the last one at ij.t)
            for (auto it = result.reinit_records.rbegin(); it != result.reinit_records.rend(); ++it) {
                if (it->t == ij.t || it->t == t_prev_step) {
                    rec.cause = it->cause;
                    break;
                }
            }
            if (rec.cause.empty() && !rec.devices.empty()) rec.cause = rec.devices.front();
            rec.sample = sample;
            result.reinit_records.push_back(std::move(rec));
        };
        // Release group B: a step's lines into the result's event limits
        // (`before`: the stored sample the step starts at).
        auto record_lines = [&](Size before) {
            const StepLines& lines = *L;
            const Size nb = result.event_limit_switches;
            auto put_mask = [&](const topology::SwitchStateMask& m,
                                std::vector<std::uint8_t>& out) {
                for (Size b = 0; b < nb; ++b) out.push_back(m.get(b) ? 1 : 0);
            };
            auto put_x = [](const Vector& v, std::vector<Real>& out) {
                out.insert(out.end(), v.data(), v.data() + v.size());
            };
            // F-SB-1: state values at events only (rl_*); the lines' starts
            // (x_start, right[]) and the held-pieces rows are quadrature
            if (lines.rl_start_ok) {
                result.right_limit_index.push_back(before);
                if (lines.rl_start_left) {   // a continuity point: the stored left
                    const Size n0 = result.right_limit_states.size();
                    result.right_limit_states.resize(n0 + static_cast<Size>(x.size()));
                    for (Index i = 0; i < x.size(); ++i) {
                        result.right_limit_states[n0 + static_cast<Size>(i)] =
                            result.states[before][i];
                    }
                } else {
                    put_x(lines.rl_start, result.right_limit_states);
                }
                put_mask(lines.rl_m_start, result.right_limit_switch_bits);
            }
            for (Size j = 0; j < lines.n; ++j) {
                if (lines.held[j]) continue;
                result.inner_event_after.push_back(before);
                result.inner_event_times.push_back(lines.t[j]);
                put_x(lines.left[j], result.inner_event_left);
                put_mask(lines.m_left[j], result.inner_event_left_bits);
                if (lines.rl_right_ok[j]) {
                    put_x(lines.rl_right[j], result.inner_event_right);
                    put_mask(lines.rl_m_right[j], result.inner_event_right_bits);
                } else {
                    result.inner_event_right.insert(
                        result.inner_event_right.end(), lines.left[j].size(),
                        std::numeric_limits<Real>::quiet_NaN());
                    put_mask(lines.m_left[j], result.inner_event_right_bits);
                }
            }
        };
        // F-4-4: a step taken on lines to the in-loop recorder line by line
        // (StepHooks::line), each point with its capacitor currents; the
        // `on_step` functions see the whole step.
        // F-Z-12: fed twice. The integral-type readings (FeedPart::Integrals)
        // take the step's lines as the method integrates them: quadrature
        // starts and the held-pieces row, so ∫i_C = C·Δv and the ledgers hold.
        // The sample-type readings (FeedPart::Samples) take the states only:
        // the right limits record_lines stores (rl_*) and no held row, the
        // rows the stored-array pass reads.
        auto cap = [](const std::vector<Real>& c) { return c.empty() ? nullptr : c.data(); };
        auto feed_quadrature = [&](Real t_prev, Real t) {
            const StepLines& lines = *L;
            Real ta = t_prev;
            const Vector* xa = lines.start ? &lines.x_start : &x_prev;
            const topology::SwitchStateMask* ma = lines.start ? &lines.m_start : &mask_start;
            const Real* ca = lines.start ? cap(lines.c_start) : nullptr;
            for (Size j = 0; j < lines.n; ++j) {
                hooks->line(ta, *xa, *ma, lines.t[j], lines.left[j], lines.m_left[j], ca,
                            cap(lines.c_left[j]));
                ta = lines.t[j];
                const bool r = lines.right_ok[j];
                xa = r ? &lines.right[j] : &lines.left[j];
                ma = r ? &lines.m_right[j] : &lines.m_left[j];
                ca = cap(r ? lines.c_right[j] : lines.c_left[j]);
            }
            hooks->line(ta, *xa, *ma, t, x, stamped_mask, ca,
                        lines.end_own ? cap(lines.c_end) : nullptr);
        };
        auto feed_states = [&](Real t_prev, Real t) {
            const StepLines& lines = *L;
            Real ta = t_prev;
            const bool s_rl = lines.rl_start_ok && !lines.rl_start_left;
            const Vector* xa = s_rl ? &lines.rl_start : &x_prev;
            const topology::SwitchStateMask* ma =
                lines.rl_start_ok ? &lines.rl_m_start : &mask_start;
            const Real* ca = lines.start ? cap(lines.c_start) : nullptr;
            for (Size j = 0; j < lines.n; ++j) {
                if (lines.held[j]) continue;
                hooks->line(ta, *xa, *ma, lines.t[j], lines.left[j], lines.m_left[j], ca,
                            cap(lines.c_left[j]));
                ta = lines.t[j];
                const bool r = lines.rl_right_ok[j];
                xa = r ? &lines.rl_right[j] : &lines.left[j];
                ma = r ? &lines.rl_m_right[j] : &lines.m_left[j];
                ca = cap(lines.right_ok[j] ? lines.c_right[j] : lines.c_left[j]);
            }
            hooks->line(ta, *xa, *ma, t, x, stamped_mask, ca,
                        lines.end_own ? cap(lines.c_end) : nullptr);
        };
        auto feed_lines = [&](Real t_prev, Real t) {
            using P = events::FeedPart;
            hooks->step_fns(t_prev, x_prev, mask_start, t, x, stamped_mask);
            const bool parts = hooks->takes_parts();
            if (parts) hooks->part(P::Integrals);
            if (!L->empty()) {
                feed_quadrature(t_prev, t);
            } else {
                hooks->sink_step(t_prev, x_prev, mask_start, t, x, stamped_mask);
            }
            if (!parts) return;
            hooks->part(P::Samples);
            // as lines even with none stored (one line from the stored right
            // limit): the sink places the commutation on its jump row, the
            // run's first row on its start (review)
            feed_states(t_prev, t);
            hooks->part(P::All);
        };
        auto step_record = [&](Size k) PULSIM_STEP_PHASE {
            const Real t = t_step;
            const Real t_prev = t_prev_step;
            if (diag) {
                const bool src_jump = (has_pwm || has_pulse || has_iwave) &&
                    sources::next_source_jump_after(pool, graph, t_prev) <= t;
                const bool plain = !restart && !step_retried && n_ev == 0 &&
                    committed_dt == opts.dt && !src_jump;
                rd_run = !plain ? 0
                    : (rd_run > 0 && stamped_mask == rd_prev_mask ? rd_run + 1
                                                                  : 1);
                rd_prev_mask = stamped_mask;
                ++rd_steps;
                if (rd_run < 3) {
                    ++rd_unest;
                }
                const auto& ents = history.entries();
                for (Size j = 0; j < ents.size(); ++j) {
                    const auto& e = ents[j];
                    const bool cap =
                        e.kind == pwl::DevicePool::StoredKind::Capacitor;
                    const Real f = (cap ? e.i_prev : e.v_prev) * rd_inv[j];
                    const Real y = std::abs(cap ? e.v_prev : e.i_prev);
                    rd_peak[j] = std::max(rd_peak[j], y);
                    Real& kpk = rd_kind_peak[cap ? 0 : 1];
                    kpk = std::max(kpk, y);
                    if (rd_run >= 3) {
                        // Normalised by the running peak, floored at 1e-3
                        // of the peak of its kind (volts / amps; review
                        // M5: a near-zero winding gave 1.3e5).
                        const Real den = std::max(rd_peak[j],
                                                  Real{1e-3} * kpk);
                        const Real e_abs = opts.dt / Real{12} *
                            std::abs(f - Real{2} * rd_f1[j] + rd_f2[j]);
                        if (e_abs > rd_since_store * den && den > Real{0}) {
                            const Real r = e_abs / den;
                            rd_since_store = r;
                            if (r > rd.lte_max) {
                                rd.lte_max = r;
                                rd.lte_branch = e.branch_id;
                                rd.lte_time = t;
                            }
                        }
                    }
                    rd_f2[j] = rd_f1[j];
                    rd_f1[j] = f;
                }
                const Real* xs = x.data();
                const Real* xps = x_prev.data();
                Real* dps = rd_dprev;
                Real* pks = rd_xpeak;
                const Size nx = static_cast<Size>(x.size());
                for (Size u = 0; u < nx; ++u) {
                    const Real d = xs[u] - xps[u];
                    const Real ax = std::abs(xs[u]);
                    pks[u] = ax > pks[u] ? ax : pks[u];
                    const Real d_last = dps[u];
                    dps[u] = d;
                    if (d * d_last < Real{0}) [[unlikely]] {
                        // Floor: 1e-3 of the entry's own peak (round-off
                        // alternation of a state near zero is not ringing).
                        const Real thr = std::max(
                            Real{0.01} * std::max(ax, std::abs(xps[u])),
                            Real{1e-3} * pks[u]);
                        if (std::abs(d) > thr) {
                            if (rd_cnt[u]++ == 0) {
                                rd_t0[u] = t;
                                rd_dmax[u] = 0;
                            }
                            rd_dmax[u] = std::max(rd_dmax[u], std::abs(d));
                            if (rd_cnt[u] >= 4) {
                                if (rd_open[u] == kRdNone) {
                                    rd_open[u] = rd.ringing.size();
                                    rd.ringing.push_back(
                                        {.var = static_cast<Index>(u)});
                                }
                                auto& rec = rd.ringing[rd_open[u]];
                                rec.t_start = rd_t0[u];
                                rec.length = rd_cnt[u];
                                rec.max_delta = rd_dmax[u];
                            }
                            continue;
                        }
                    }
                    if (rd_cnt[u] != 0) [[unlikely]] {
                        rd_cnt[u] = 0;
                        rd_open[u] = kRdNone;
                    }
                }
            }

            // 6. Record. event_iteration_count = iters - 1 (the
            //    first iteration always runs; the count is the
            //    number of EXTRA solves caused by diode flips).
            // Decimated recording (v2.0 Phase 1): uniform grid at
            // m·dt. Sample 0 (the IC, k = 0) is always recorded
            // above, keeping the parallel arrays aligned.
            if (hook_ds) hooks->device_state(t, x, hs);   // #31: after the commit
            if (hook_acc) hooks->accepted(t_prev, t, x, stamped_mask);
            if (k % opts.store_every == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(iters - 1);
                result.record_switch_state(stamped_mask);
                // the step's lines (release group B), when its start is the
                // sample before this one
                if (L && L->owner == t && L->ok && (!L->empty() || L->rl_start_ok) &&
                    opts.store_every == 1 &&
                    result.times.size() >= 2) [[unlikely]] {
                    record_lines(result.times.size() - 2);
                }
                if (diag) {
                    rd.lte.push_back(rd_since_store);
                    rd_since_store = 0;
                }
            }
            // The restart's impulsive re-initialisations of this step
            // (consistent_start), on the committed attempt only.
            if (!impulse_pending.empty()) {
                for (auto& r : impulse_pending) {
                    r.sample = k % opts.store_every == 0
                                   ? static_cast<long>(result.times.size()) - 1 : -1L;
                    result.reinit_records.push_back(std::move(r));
                }
                impulse_pending.clear();
                event_rl_t = std::numeric_limits<Real>::quiet_NaN();   // read above
                if (lt) lt->impulses(result.reinit_records);   // booked to `cause`
            }
            // add-branch-form-ideal-switch D4 (task 5.1): a dependent mode
            // entered this step was re-initialised by the SDIRK x3 restart;
            // record the explicit projection from the state the mode was
            // entered from: with the event split, the left limit at the
            // split instant that changed the mode (F-Z-1), else the step start.
            if (structure.active()) {
                const bool at_ev = !std::isnan(mode_ev_t);
                const Real t_in = at_ev ? mode_ev_t : t_prev;
                const auto& now = structure.check(stamped_mask, true);
                refuse_dependent(now, stamped_mask, t_in);
                reinit.record(structure, stamped_mask, at_ev ? mode_ev_x : x_prev, x, t_in,
                              k % opts.store_every == 0
                                  ? static_cast<long>(result.times.size()) - 1
                                  : -1L,
                              opts.on_impulse_error, "pwl", result);
                for (const IdealJump& ij : ideal_pending) {
                    record_ideal_fast(ij, k % opts.store_every == 0
                                              ? static_cast<long>(result.times.size()) - 1 : -1L);
                }
                ideal_pending.clear();
                if (lt) lt->impulses(result.reinit_records);
            }
            if (hook_mask) {
                if (hook_event && mask_start_known && !(stamped_mask == mask_start)) {
                    hooks->event(t_prev, x_prev, mask_start, x, stamped_mask);
                }
                if (hook_step) {
                    const bool lined = L && L->owner == t && L->ok && hooks->takes_lines();
                    if (lined && (!L->empty() || L->rl_start_ok)) [[unlikely]] {
                        feed_lines(t_prev, t);
                    } else {
                        hooks->step(t_prev, x_prev, mask_start, t, x, stamped_mask, hook_contig);
                    }
                    hook_contig = true;
                }
                mask_start = stamped_mask;   // the next step's start
                mask_start_known = true;
            }
        };  // step_record

        // The driver, the step skeleton: the loop's own phases in the
        // skeleton's order and clock (the run's step k = 1 .. n_steps - 1;
        // the clock's k0 = k_off puts them on the run's grid). It is one
        // out-of-line function with the phases inlined into it
        // (DECISIONS 14). The body is built HERE, outside the driver, so
        // the driver reaches the phases through memory as the deleted loop
        // did: built inside it, its references were nine values live
        // across the loop that spilled (+0.8-1.4 % per step, DECISIONS 18).
        auto body = step::index_phase_body(step_cancelled, step_begin, step_take,
                                           step_after_solve, step_commit, step_record);
        auto run_twin = [&]() PULSIM_STEP_DRIVER {
            step::FixedController clock{.clock = {fixed_grid.t0, fixed_grid.dt, k_off},
                                        .k = 1,
                                        .n_steps = n_steps};
            (void)step::run<step::Trapezoidal>(clock, body);
        };
        run_twin();
        if (sens.on()) {
            // an edge AT the last grid point under the restart rule (gates
            // read at t_prev): no step follows to open it (sens_open)
            const topology::SwitchStateMask g_last = gate_at(gate_t);
            sens.edge_past_end(t_reached, g_last, gate_at(t_reached));
            sens.finish(t_reached, x);
        }
        if (hook_step) hooks->flush_steps();   // the step sink's batched steps
        if (diag && rd_steps > 0) {
            rd.unestimated_fraction =
                static_cast<Real>(rd_unest) / static_cast<Real>(rd_steps);
        }
        if (diag && rd.lte.size() < result.times.size()) {
            // The initial sample(s) carry no estimate.
            rd.lte.insert(rd.lte.begin(),
                          result.times.size() - rd.lte.size(), Real{0});
        }
        result.final_snapshot.restart_steps_left = restart_left;
        result.final_snapshot.restart_anchor_at_end = anchor_t == t_reached;
        result.n_restart_factorizations = restart_factorizations;
    } else {
        // ----------- STATIC PATH ----------------------------------------
        //
        // No dynamic devices → cache.solve gives the DC operating
        // point for the requested switch state. Diode iteration
        // still applies.
        if (sens.on()) {
            throw UnsupportedError("pwl", "sensitivities", "", "",
                "sensitivities: the circuit has no capacitor or inductor (the "
                "static path): there is no step to differentiate.");
        }
        if (lt) {
            throw UnsupportedError(
                "pwl", "device_losses", pool.device_losses().front().device, "",
                "device losses: the circuit has no capacitor or inductor, so "
                "it runs on the static path, which reads no commutations (a "
                "record-only port included); give it its dynamic devices or "
                "the thermal network.");
        }
        // ---- slice 3c (DECISIONS 17): the static path's step, cut in place
        // into four phases (begin, take, after_solve, record; nothing to
        // commit: no history) that the skeleton calls. The first solve
        // (k = 0) is the start sample, before the skeleton's steps, as the
        // dynamic path's IC sample; the skeleton runs k = 1 .. n_steps - 1
        // on the dynamic path's FixedController. The static path never
        // polled cancellation, and still does not (the poll is `true`).
        const Vector& b_extra_user = b_extra;
        Real st_t = Real{0};
        Real st_t_prev = Real{0};
        Real st_gate_t = Real{0};
        Size st_iters = 0;
        bool st_flipped = false;
        bool st_mask_cycle = false;
        auto st_begin = [&](Size k) PULSIM_STEP_PHASE {
            const Real t = fixed_grid.at(k_off + k);
            st_t = t;
            t_reached = t;
            k_reached = k;
            const Real t_prev = k > 0
                ? fixed_grid.at(k_off + k - 1)
                : opts.t_start;
            st_t_prev = t_prev;
            // Sub-step bisection snapshot (Layer 5 V2.2). At
            // k=0 we don't have a prev step; just use the
            // current x as a degenerate snapshot — the
            // sign-change check will exclude it from events.
            x_prev = x;
            if (has_ideal_diode) {
                const auto& es0 = diodes.entries();
                commut_on_prev.resize(es0.size());
                for (std::size_t j = 0; j < es0.size(); ++j) commut_on_prev[j] = es0[j].is_on;
            }

            // State-aware observer fires before switch_fn(t) so
            // discrete-time controllers can update.
            if (step_observer || step_observer_in) {
                (has_ticks ? step_observer : step_observer_in)(t, x_prev);
            }
            gate_memo_n = 0;

            // Accumulate into the reused buffer (audit #6/#16): optional user
            // fn, then present built-in source kinds. No trap history on the
            // static path. Same summation order → bit-identical; absent kinds
            // and a null user fn cost nothing. `b_extra_user` aliases the
            // buffer so the solves below are unchanged.
            // discontinuity_restart: the same sampling as the dynamic
            // path — sample k reads the gate 0.5001 dt before t and the jumping
            // sources at their left limit — so adding a capacitor does
            // not shift the recorded mask by one sample. (k = 0: t.)
            st_gate_t = restart_rule && k > 0 ? t - gate_back * opts.dt : t;   // as the dynamic path
            const Real src_left = k > 0 ? left_by : Real{0};
            b_extra.setZero();
            if (b_extra_fn) {
                b_extra += b_extra_fn(k > 0 ? b_read_t(t) : t);
            }
            add_source_rows(b_extra, t, t, src_left);
            if (has_iwave) {
                sources::compute_current_waveform_b_extra(pool, graph, t,
                                                          be_iwave, src_left);
                b_extra += be_iwave;
            }

        };
        auto st_take = [&]() PULSIM_STEP_PHASE {
            const Real t = st_t;
            step_ctx.t = t;   // the external devices' ctx->t (this path solves at t)
            const Real gate_t = st_gate_t;
            Size& iters = st_iters;
            bool& flipped = st_flipped;
            bool& mask_cycle = st_mask_cycle;
            iters = 0;
            flipped = false;
            mask_cycle = false;
            // Full mask copies (Phase 1: masks can exceed 64 bits; a
            // copy is 2 inline words for <=128 switches — still no
            // heap in the loop).
            masks_seen.reset();

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
            if (diodes.num_gated() > 0) gated_prev_stamp = stamped_mask;
            if (has_ideal_diode) ideal_prev_stamp = stamped_mask;
            const bool mode_site_s = mode_ports.has_value() && !nl_refresh_effective;
            auto solve_s = [&](const topology::SwitchStateMask& m, Vector& out) {
                cache.solve_segment(nominal_segment(m), b_extra_user, out);   // solve-exempt: static path (DECISIONS 22)
            };
            if (mode_site_s) mode_begin();
            if (mode_site_s && opts.mode_selection == 2) {
                const topology::SwitchStateMask gates = gate_at(gate_t);
                (void)mode_lcp_site(t, "static", gates, Real{0}, false, b_extra_user, solve_s);
                iters = 1;
            } else
            do {
                auto mask = gate_at(gate_t);
                if (has_diodes) {
                    mask = diodes.combine(mask);
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
                if (!nl_refresh_effective && masks_seen.repeat(mask)) {
                    mask_cycle = true;
                    break;
                }
                if (nl_refresh_effective) {
                    const auto& seg = nominal_segment(mask);
                    x = pwl::solve_with_newton_b_extra(   // solve-exempt: static path (DECISIONS 22)
                        seg, nl_refresh_effective, graph, pool,
                        /*x_init=*/x, b_extra_user,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search,
                        opts.enable_newton_lm, &newton_ws);
                } else {
                    cache.solve_segment(nominal_segment(mask), b_extra_user, x);   // solve-exempt: static path (DECISIONS 22)
                }
                stamped_mask = mask;
                stamped_known = true;
                if (has_diodes) {
                    diodes.snapshot_on_bits_into(last_solved_bits);
                }
                flipped = has_diodes && events::redecide(diodes, x);
                ++iters;
            } while (flipped && iters < max_iters);
            if (mode_site_s && opts.mode_selection != 2) {
                const topology::SwitchStateMask gates = gate_at(gate_t);
                if (opts.mode_selection == 1) {
                    mode_shadow_site(t, "static", gates, Real{0}, false, b_extra_user,
                                     iters, mask_cycle);
                }
                if (result.mode_guard.enabled) {
                    ++result.mode_guard.checks;
                    if (flipped || mask_cycle) ++result.mode_guard.settle_caps;
                    if ((flipped || mask_cycle) &&
                        mode_guard_site(t, gates, Real{0}, false, b_extra_user, solve_s)) {
                        flipped = false;
                        mask_cycle = false;
                    }
                }
            }

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

        };
        auto st_after = [&](Size k) PULSIM_STEP_PHASE {
            const Real t = st_t;
            const Real t_prev = st_t_prev;
            // Sub-step bisection on the static path too.
            if (has_diodes && k > 0) {
                for (std::size_t j = 0; j < diodes.entries().size(); ++j) {
                    const auto& e = diodes.entries()[j];
                    if (e.gate != models::SwitchedDiode::Gate::None) {
                        const bool now = stamped_mask.get(e.switch_idx);
                        if (now != gated_prev_stamp.get(e.switch_idx)) {
                            result.commutation_events.push_back(
                                CommutationEvent{
                                    .t_estimated = t_prev,
                                    .branch_id   = e.branch_id,
                                    .new_state   = now,
                                });
                        }
                        continue;
                    }
                    if (has_ideal_diode && pool.is_branch_form(e.branch_id)) {
                        // a flip at a gate edge at the step end, any other as
                        // the conductance form (branch_form_event_time)
                        const bool on_prev = static_cast<bool>(commut_on_prev[j]);
                        if (on_prev != e.is_on) {
                            const Real t_ev = gate_edge_since(ideal_prev_stamp)
                                ? t
                                : branch_form_event_time(on_prev, t_prev, t,
                                                         branch_form_s(e, x_prev),
                                                         branch_form_s(e, x));
                            if (!std::isnan(t_ev)) {
                                result.commutation_events.push_back(CommutationEvent{
                                    .t_estimated = t_ev, .branch_id = e.branch_id,
                                    .new_state = e.is_on});
                            }
                        }
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
                    result.commutation_events.push_back(
                        CommutationEvent{
                            .t_estimated = t_est,
                            .branch_id   = e.branch_id,
                            .new_state   = e.is_on,
                        });
                }
            }

        };
        auto st_record = [&](Size k) PULSIM_STEP_PHASE {
            const Real t = st_t;
            const Real t_prev = st_t_prev;
            const Size iters = st_iters;
            // The static path's first solve (k = 0) is the start state.
            if (hooks) {
                if (k == 0) {
                    hooks->start(x, stamped_mask, {&history, &hs, nullptr});
                } else {
                    hooks->accepted(t_prev, t, x, stamped_mask);
                }
            }
            if (k % opts.store_every == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(iters - 1);
                result.record_switch_state(stamped_mask);
            }
        };
        auto st_run_skeleton = [&]() PULSIM_STEP_PHASE {
            if (n_steps == 0) return;
            st_begin(0);
            st_take();
            st_after(0);
            st_record(0);
            auto sk_poll = []() PULSIM_STEP_PHASE { return true; };
            auto sk_begin = [&](const step::StepFrame& f) PULSIM_STEP_PHASE {
                st_begin(f.k);
            };
            auto sk_take = [&](const step::StepFrame&) PULSIM_STEP_PHASE {
                st_take();
                return true;   // a failed solve throws
            };
            auto sk_after = [&](step::StepFrame& f) PULSIM_STEP_PHASE {
                st_after(f.k);
                return true;
            };
            auto sk_commit = [](const step::StepFrame&) PULSIM_STEP_PHASE {};
            auto sk_record = [&](const step::StepFrame& f) PULSIM_STEP_PHASE {
                st_record(f.k);
            };
            auto body = step::phase_body(sk_poll, sk_begin, sk_take, sk_after,
                                         sk_commit, sk_record);
            step::FixedController clock{.clock = {fixed_grid.t0, fixed_grid.dt, k_off},
                                        .k = 1,
                                        .n_steps = n_steps};
            (void)step::run<step::Trapezoidal>(clock, body);
        };
        st_run_skeleton();
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
    catch (const pwl::NewtonRoundoffBudgetExceeded& e) {
        if (lt) lt->write(result);
        throw SimulationAborted(e.what(), std::move(result), t_reached,
                                /*roundoff_budget=*/true);
    }
    catch (const std::runtime_error& e) {
        // Narrow on purpose — see the note at the dt-retry above.
        // Everything the solver throws is a runtime_error; a Python
        // callback's exception is not, and re-labelling one as
        // SimulationAborted would destroy its type and traceback for
        // the sake of attaching a partial trace it never asked for.
        if (lt) lt->write(result);
        throw SimulationAborted(e.what(), std::move(result),
                                 t_reached);
    }

    // v2.0 Phase 4 — the complete state at the end, so a follow-up
    // run can resume EXACTLY rather than re-deriving the companion
    // history from x (which is not the state and does not
    // reproduce the run).
    result.final_snapshot.t = t_reached;
    result.final_snapshot.x = x;
    result.final_snapshot.grid_origin = t_origin;
    result.final_snapshot.grid_step = k_off + k_reached;
    result.final_snapshot.grid_dt = opts.dt;
    hs.save(history, result.final_snapshot);
    result.final_snapshot.mask =
        has_diodes ? diodes.snapshot_mode_bits() : std::vector<bool>{};
    // The mask x was solved under (the record's last row when the
    // last step was stored), for a resume's row 0.
    result.final_snapshot.stamped_mask =
        stamped_known ? mask_bits(stamped_mask) : std::vector<bool>{};
    if (has_ticks) {   // the tick sources are ids 0..n-1
        for (std::uint32_t j = 0; j < ticks->rates.size(); ++j) {
            result.final_snapshot.tick_origin.push_back(tick_queue.grid_origin(j));
            result.final_snapshot.tick_next_k.push_back(tick_queue.next_k(j));
        }
    }
    result.final_snapshot.valid = true;
    if (lt) result.final_snapshot.loss_state = lt->save();
    if (lines) result.final_snapshot.line_state = lines->save();
    result.newton = newton_ws.take_counters();
    if (sens.on()) result.sensitivity = sens.take_trace();
    if (lt) lt->write(result);
    result.gate_mask_fn = nullptr;   // it reads this frame's diode state

    return result;
}

}  // namespace pulsim::solver
