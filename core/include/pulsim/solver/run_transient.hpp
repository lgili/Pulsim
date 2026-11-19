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
#include "pulsim/solver/loss_tracker.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/events/mode_set.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/step/controllers.hpp"
#include "pulsim/solver/step/integrators.hpp"
#include "pulsim/solver/step/stepper.hpp"
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
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
        auto dc = [&] {
            try {
                return pwl::compute_dc_operating_point(
                    graph, pool, switch_fn(opts.t_start), nl_static,
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
        history.seed_from_dc_op(x);
        hs.get<pwl::PmsmMnaHistory>().seed_from_dc_op(x);   // λ, i, θ at the held DC point (D19)
        // the external devices' charges and model state at the DC point
        hs.get<pwl::ExternalDeviceHistory>().seed_from_dc_op(x);
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
    std::vector<bool> rs_bits, rs_bits2, rs_bits_be, rs_solved2;
    std::vector<bool> rs_bits0;   // the diode bits at stage 1's start
    // The diode bits after stage 1 of a restart that starts at the step
    // start: a branch-form flip already there was forced at t_prev.
    std::vector<bool> rs_bits_first;
    bool rs_first_valid = false;
    std::vector<Real> rs_hist;
    pwl::StatefulHistories::Snapshot rs_hs;
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
    // The gate sample of the interval that starts at ev_t[i]: the LAST
    // event of a cluster merged within `snap` (review M1).
    std::array<Real, kMaxStepEvents> ev_s{};
    std::vector<CommutationEvent> split_events;
    split_events.reserve(8);
    std::vector<Real> cross_scratch;
    topology::SwitchStateMask prev_gate;   // unlocated-edge count (no oracle)
    bool have_prev_gate = false;
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
    std::vector<std::string> mode_diode_names;
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
            for (const auto& e : diodes.entries())
                mode_diode_names.push_back(pwl::branch_label(graph, e.branch_id));
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
                    "pwl", "ideal_voltage_loop", mode_diode_names[j], "Diode",
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
                std::vector<char> on_f = mode_coords;
                for (std::size_t j = 0; j < mode_diode_names.size(); ++j) {
                    const auto jj = static_cast<Index>(j);
                    if (mode_ports->ideal(j) && mode_u[jj] - mode_ports->ports()[j].V_f > -tol_w)
                        on_f[j] = 1;
                }
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
        events::raise_if_refused(sel, "pwl", at_t, mode_diode_names, mode_passivity);
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
                    cache.solve_at(combine_masks(gates, mode_mask_of(sel.on), diode_owned),
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
    // add-controlled-source-set: sine / pulse / table current sources
    // (node rows, two per source: kept on the dense helper).
    const bool has_iwave =
        sources::has_current_waveform_sources(pool, graph);
    Vector be_iwave;
    const bool has_pwm = !pwm_rows.empty();
    const bool has_pulse = !pulse_rows.empty();
    // `src_t`: PWM/pulse sample time; `sine_t`: the sine's; `lb`: a
    // jumping source at `src_t` reads value_at(src_t − lb) (as the dense
    // passes did).
    auto add_source_rows = [&](Vector& b, Real src_t, Real sine_t,
                               Real lb) {
        const auto read_t = [&](Index id) {
            const bool jumps = lb > Real{0} &&
                sources::source_jumps_at(pool, id, src_t);
            // src_t != sine_t: a jump snapped to the step end; only the
            // source that jumps there is re-timed (review: a ramped
            // pulse was read up to 0.01 dt early).
            if (src_t != sine_t && !jumps) {
                return sine_t;
            }
            return jumps ? src_t - lb : src_t;
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
        topology::SwitchStateMask mask_pre;
        Size iters = 0;
        bool flipped = false;
        bool mask_cycle = false;
        Real committed_dt = opts.dt;
        bool step_retried = false;
        bool edge_restart_step = false;
        bool restart = false;
        bool committed_be = false;
        Real be_commit_h = kGamma * opts.dt;
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
            history.update_from_state_backward_euler(x, h_e);
            hs.commit(x, h_e, pwl::TrBdf2Stage::BackwardEuler);
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
            gate_t = restart_rule ? t_prev : t;
            if (step_observer || step_observer_in) {
                (has_ticks ? step_observer : step_observer_in)(t, x_prev);
                // An observer (or a control tick) may mutate the
                // schedule: its oracle answer is stale (review M3).
                edge_from = std::numeric_limits<Real>::infinity();
                edge_next = -std::numeric_limits<Real>::infinity();
                // ... and so may its declared b_extra_fn jumps (review L6).
                bj_from = std::numeric_limits<Real>::infinity();
                bj_next = -std::numeric_limits<Real>::infinity();
            }
            gate_memo_n = 0;
            // ---- add-fixed-step-event-splitting: the step's events ----
            // Gate edges from the schedule's oracle and built-in source
            // jumps inside [t_prev + snap, t - snap), in time order. One
            // before that is taken at t_prev (the gate is sampled at it;
            // a source jump triggers the restart below), one after it at
            // t (the next step's; a source jump is read at its left
            // limit by this step's last solve).
            n_ev = 0;
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
                history.snapshot_into(history_snap);
                hs.snapshot_into(hs_snap);
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
            if (split && !next_edge_fn) {
                // An opaque switch_fn: a gate change between two grid
                // samples was not located (review M3).
                // Counted only when the change happened INSIDE the
                // previous step (the gate at t_prev - snap is already the
                // new mask); an edge exactly on the grid point t_prev was
                // taken where it is (review L4).
                if (have_prev_gate && !(mask_pre == prev_gate) &&
                    gate_at(t_prev - snap) == mask_pre) {
                    ++result.unlocated_gate_steps;
                }
                prev_gate = mask_pre;
                have_prev_gate = true;
            }
            if (ev_overflow) {
                ++result.split_event_overflows;
            }
            if (has_diodes) {
                mask_pre = diodes.combine(mask_pre);
            }
        };  // step_begin

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
                    cache.solve_at(mask, for_dt, b_extra, out);
                    const auto m1 = cache.metrics();
                    restart_factorizations += static_cast<Size>(
                        (m1.event_builds - m0.event_builds) +
                        (m1.event_refactors - m0.event_refactors));
                } else if (for_dt == opts.dt && for_dt == cache.dt()) {
                    cache.solve_segment(nominal_segment(mask), b_extra, out);
                } else {
                    cache.solve_at(mask, for_dt, b_extra, out);
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
                if (nl_refresh_effective) {
                    if (for_dt == opts.dt && !newton_only_rows) {
                        const auto& seg = nominal_segment(mask);
                        x = pwl::solve_with_newton_b_extra(
                            seg, nl_refresh_effective, graph, pool,
                            /*x_init=*/x, b_extra,
                            opts.max_newton_iterations,
                            opts.tol_newton_dx,
                            opts.tol_newton_res,
                            opts.enable_newton_line_search,
                            opts.enable_newton_lm,
                            &newton_ws, &newton_roundoff);
                        note_roundoff();
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
                            opts.enable_newton_lm,
                            &newton_ws, &newton_roundoff);
                        note_roundoff();
                    }
                } else {
                    // `solve_at` delegates to `solve` bit-identically
                    // when the dt matches, so the nominal path costs
                    // nothing extra.
                    // Every restart stage counts, the split's sub-interval
                    // stages (dt_eq = 2γ·h_sub) included (review L8b).
                    if (restart_rule && (for_dt == restart_dt || backward_euler)) {
                        const auto m0 = cache.metrics();
                        cache.solve_at(mask, for_dt, b_extra, x);
                        const auto m1 = cache.metrics();
                        restart_factorizations += static_cast<Size>(
                            (m1.event_builds - m0.event_builds) +
                            (m1.event_refactors - m0.event_refactors));
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
            }
            restart = restart_left > 0;
            if (restart) --restart_left;
            // The final solve of this step was a restart stage, so the
            // deferred commit below is a backward-Euler one (at γh).
            committed_be = false;
            be_commit_h = kGamma * opts.dt;   // set by run_restart
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
            auto run_restart = [&](Real t_from, Real h) {
                const Real h_s = kGamma * h;
                const Real dt_eq = Real{2} * h_s;
                be_commit_h = h_s;
                accumulate_b_extra(t_from + h_s, dt_eq, true);
                if (has_diodes) diodes.snapshot_on_bits_into(rs_bits0);
                run_event_iteration(t_from + h_s, dt_eq, true);
                const Size iters_1 = iters;
                const bool breach_1 = flipped;
                if (breach_1 && has_diodes && !last_solved_bits.empty()) {
                    diodes.restore_on_bits(last_solved_bits);
                }
                if (has_diodes) {
                    rs_x1 = x;
                    diodes.snapshot_on_bits_into(rs_bits);
                    if (!rs_first_valid) {   // the attempt's first stage: from t_prev
                        rs_bits_first.assign(rs_bits.begin(), rs_bits.end());
                        rs_first_valid = true;
                    }
                    history.snapshot_into(rs_hist);
                    hs.snapshot_into(rs_hs);   // keeps the capacity
                }
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
                        history.restore(rs_hist);
                        hs.restore(rs_hs);
                    };
                    x = rs_x1;
                    diodes.restore_on_bits(rs_bits);
                    histories_at_x0();
                    mask_cycle = cycle_1;
                    const Real h_b = h - h_s;
                    // A re-try that fails (Newton, a singular factor at
                    // the new dt) falls back to the SDIRK stage 2, which
                    // had already succeeded.
                    bool keep_be = false;
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
                    }
                    if (keep_be) {
                        ++result.stage2_be_kept;
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
                }
                // A diode flip inside the step is a discontinuity of its
                // own: the window covers this step and the next two.
                if (iters_1 > 1 || iters_2 > 1) {
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
                hs.commit(x, h, pwl::TrBdf2Stage::Trapezoidal);
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
                            // Back to the step start, then: the crossing
                            // at a (an SDIRK step, as the rule alone), at
                            // b (keep the pre-event solve, flip there) or
                            // inside (split).
                            x = x_prev;
                            history.restore(history_snap);
                            hs.restore(hs_snap);
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
                        hs.restore(hs_snap);
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
                    hs.restore(hs_snap);
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
                    rs_first_valid = false;

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
                        } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                            throw;   // a run budget: never retried at a smaller dt
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
                            commit_backward_euler(be_commit_h);
                        } else if (j + 1 < n_sub) {
                            history.update_from_state(x, sub_dt);
                            hs.commit(x, sub_dt,
                                      pwl::TrBdf2Stage::Trapezoidal);
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
                        // shortcut: stamped at the step end, not interpolated;
                        // the event split locates it when it is on. A flip a
                        // gate edge forced in the restart's first stage is
                        // stamped at the edge (t_prev).
                        if (static_cast<bool>(commut_on_prev[j]) != e.is_on) {
                            const bool at_edge = edge_restart && rs_first_valid &&
                                                 rs_bits_first[j] == e.is_on;
                            step_events.push_back(CommutationEvent{
                                .t_estimated = at_edge ? t_prev : t,
                                .branch_id = e.branch_id, .new_state = e.is_on});
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
                commit_backward_euler(be_commit_h);
            } else {
            history.update_from_state(x, committed_dt);
            hs.commit(x, committed_dt, pwl::TrBdf2Stage::Trapezoidal);
            }
            if (restart) {
                ++result.n_restart_steps;
            }
            // add-in-kernel-event-losses: the step is accepted; read its
            // commutations and conduction energy (reading rule).
            if (lt) lt->accept(t_prev, t, x, stamped_mask);
        };  // step_commit

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
                if (diag) {
                    rd.lte.push_back(rd_since_store);
                    rd_since_store = 0;
                }
            }
            // add-branch-form-ideal-switch D4 (task 5.1): a dependent mode
            // entered this step was re-initialised by the SDIRK x3 restart;
            // record the explicit projection from the step's start state.
            // shortcut: x- is the state at the step start (with the event
            // split, not the split instant); the record's t is t_prev.
            if (structure.active()) {
                const auto& now = structure.check(stamped_mask, true);
                refuse_dependent(now, stamped_mask, t_prev);
                reinit.record(structure, stamped_mask, x_prev, x, t_prev,
                              k % opts.store_every == 0
                                  ? static_cast<long>(result.times.size()) - 1
                                  : -1L,
                              opts.on_impulse_error, "pwl", result);
                if (lt) lt->impulses(result.reinit_records);
            }
            if (hook_mask) {
                if (hook_event && mask_start_known && !(stamped_mask == mask_start)) {
                    hooks->event(t_prev, x_prev, mask_start, x, stamped_mask);
                }
                if (hook_step) {
                    hooks->step(t_prev, x_prev, mask_start, t, x, stamped_mask, hook_contig);
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
        result.n_restart_factorizations = restart_factorizations;
    } else {
        // ----------- STATIC PATH ----------------------------------------
        //
        // No dynamic devices → cache.solve gives the DC operating
        // point for the requested switch state. Diode iteration
        // still applies.
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
            // path — sample k reads the gate at t_{k-1} and the jumping
            // sources at their left limit — so adding a capacitor does
            // not shift the recorded mask by one sample. (k = 0: t.)
            st_gate_t = restart_rule && k > 0 ? t_prev : t;
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
            const bool mode_site_s = mode_ports.has_value() && !nl_refresh_effective;
            auto solve_s = [&](const topology::SwitchStateMask& m, Vector& out) {
                cache.solve_segment(nominal_segment(m), b_extra_user, out);
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
                    x = pwl::solve_with_newton_b_extra(
                        seg, nl_refresh_effective, graph, pool,
                        /*x_init=*/x, b_extra_user,
                        opts.max_newton_iterations,
                        opts.tol_newton_dx,
                        opts.tol_newton_res,
                        opts.enable_newton_line_search,
                        opts.enable_newton_lm, &newton_ws);
                } else {
                    cache.solve_segment(nominal_segment(mask), b_extra_user, x);
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
                        // at s = 0 while ON: the crossing test never fires
                        if (static_cast<bool>(commut_on_prev[j]) != e.is_on) {
                            result.commutation_events.push_back(CommutationEvent{
                                .t_estimated = t, .branch_id = e.branch_id,
                                .new_state = e.is_on});
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
    result.newton = newton_ws.take_counters();
    if (lt) lt->write(result);
    result.gate_mask_fn = nullptr;   // it reads this frame's diode state

    return result;
}

}  // namespace pulsim::solver
