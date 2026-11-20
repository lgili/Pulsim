#pragma once

// =============================================================================
// Pulsim — variable-step TR-BDF2 transient on the sparse MNA kernel
// =============================================================================
//
// v2.0 Phase 3 — the audit's "modo default": TR-BDF2 variável com
// LTE direto no MNA esparso, eventos localizados entre passos
// (findings #38 no-lte-variable-step-mna, #39
// trap-damping-and-commutation-restart).
//
// WHAT THIS IS. One composite implicit step over h:
//
//   stage 1 (TR):    trapezoidal over γh,        γ = 2 − √2
//   stage 2 (BDF2):  BDF2 over the remaining (1−γ)h, using the
//                    stage point and the step start
//
// The method is L-stable (a decayed stiff mode stays decayed — the
// snubber ring after a commutation is crossed instead of resolved
// forever), second order, ONE-STEP (post-event restart needs no
// history bootstrap), and carries an embedded LTE estimate that
// drives a step controller — `simulate()` no longer needs the user
// to guess dt.
//
// WHY IT REUSES THE WHOLE TRAP KERNEL. γ = 2 − √2 makes the BDF2
// stage's derivative coefficient c1 = 2/γ, so the stage-2 matrix
//   G + (C_phys·c1/h) == G + C_phys·(2/(γh)) == the TRAP matrix at
// dt = γh. Both stages therefore solve through the SAME
// `cache.solve_at(mask, γh)` factor — one numeric factorization
// per (mask, h), the existing symbolic analysis, the existing
// (G, C, b) split, the existing companion history (whose stored
// state is PHYSICAL (v, i), making variable h legal by
// construction — the dt-retry sub-steps already rely on that).
// The per-device stage-2 assembly lives in
// `HistoryState::compute_b_extra_trbdf2_stage2` / `commit_trbdf2`;
// its algebra was validated against the dense matrix form to 1e-14
// before this file was written.
//
// EVENTS.
//   * Gate edges (switch_fn): before taking a step, the step is
//     CLAMPED to land exactly on the next gate edge, found by
//     bisection on switch_fn itself — no solves. The step's mask is
//     sampled at the step MIDPOINT, so a step that lands on an edge
//     uses the pre-edge mask and the next step starts on the
//     post-edge mask.
//   * Sampling blindness (shared with the fixed engine): a diode
//     conduction window that fits ENTIRELY inside one stage
//     sub-interval leaves the same sign at all three sample points
//     and is invisible — exactly as a sub-dt blip is invisible to
//     the fixed-trap engine. h_max bounds the blind window the way
//     dt does there.
//   * Diode crossings: after an LTE-accepted step, each diode's
//     per-direction signal (ON→OFF watches v_D − V_f, whose zero is
//     the current zero; OFF→ON watches v_D − V_on) is checked at the
//     stage and end points. A crossing is localized by Illinois
//     (regula falsi with endpoint halving) on trial TRAP solves at
//     dt* = t* − t, the step is committed at t*, the diode flips
//     there (`SwitchedDiode::decide`), and the
//     controller restarts small — the one-step method needs
//     nothing else.
//
// SCOPE. R, L, C, transformers, switches, switched diodes, every
// source kind (DC/PWM/sine/pulse via the same b_extra overlays
// run_transient uses), a user b_extra_fn, scheduled controller
// ticks, and NONLINEAR devices (Shockley diodes, MOSFET/IGBT L1):
// each stage then becomes a Newton solve on the same assembled
// companion, using the fixed engine's own refresh callback. That
// works because those devices are memoryless resistive I-V
// elements — no dt appears in their re-stamp, so nothing about
// them cares that h varies.
//
// Devices that carry STATE need more than that — each needs its own
// derived BDF2 second stage, since the two stages approximate dX/dt
// differently and only the CONDUCTANCE coincides. Five have one:
// the charge-based Coss, the Lauritzen diode, the IGBT turn-off
// tail, the MNA-native PMSM and the saturable inductor. There are
// no refusals left.

#include "pulsim/sim/defaults.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/events/locate.hpp"
#include "pulsim/events/mode_set.hpp"
#include "pulsim/events/sampled_gate.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/mode_ports.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_refresh_igbt_tail.hpp"
#include "pulsim/pwl/nonlinear_refresh_lauritzen_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/pwl/nonlinear_refresh_nonlinear_capacitor.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/reinit.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/models/transformer.hpp"
#include "pulsim/solver/loss_tracker.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/step/controllers.hpp"
#include "pulsim/solver/step/esdirk43.hpp"
#include "pulsim/solver/step/integrators.hpp"
#include "pulsim/solver/step/stepper.hpp"
#include "pulsim/solver/run_transient.hpp"   // combine_masks, SimulationAborted
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/current_waveform_b_extra.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::solver {

struct TrBdf2Options {
    /// add-unified-device-interface DECISIONS #23 (see SimulationOptions).
    device::DevicePath device_path = device::DevicePath::Legacy;
    Real t_start = Real{0};
    Real t_end   = Real{0};
    Real rtol    = sim::Defaults::rtol_variable;  // was 1e-4
    Real atol    = sim::Defaults::atol_variable;  // was 1e-6
    /// 0 = auto (h_max / 100).
    Real h_init  = Real{0};
    /// Step ceiling — also the gate-edge SAMPLING resolution
    /// ceiling: an opaque switch_fn is sampled every h/4 inside a
    /// step, so an isolated on-off gate pulse narrower than h_max/4
    /// can be missed entirely and SILENTLY (nothing saw it to warn
    /// about); one at least h_max/4 wide is always landed. A periodic
    /// gate is safer: once two of its edges are seen the step is
    /// capped at twice the narrowest pulse. 0 = auto (span / 1000,
    /// lowered by the circuit's own periodic sources).
    Real h_max   = Real{0};
    /// Accepted-step floor. 0 = auto (span · 1e-11).
    ///
    /// The value is measured, not chosen: a buck's freewheel
    /// commutation needs landings below ~5e-13 s to stay accurate
    /// (span·1e-10 cost 1.2 mV on a 24 V output; span·1e-11 and
    /// span·1e-12 are identical at 0.48 mV), while going far below
    /// that only degrades conditioning — the trap companion stamps
    /// 2C/h, so 1e-15 s on a 47 µF capacitor is 1e11 S and the
    /// solve stops meaning anything. Note this is the floor for a
    /// COMMITTED step; diode PROBES have their own, much smaller,
    /// because nothing they compute is kept.
    Real h_min   = Real{0};
    Size max_steps = Size{10'000'000};
    Size max_event_iterations = sim::Defaults::max_event_iterations;
    /// Digital-controller sample period. When positive, the run
    /// LANDS a step boundary on every k·T_ctrl and fires the
    /// observer there — the cadence a discrete controller actually
    /// samples on, instead of "whichever step happened to cross
    /// the boundary". 0 = no controller ticks.
    Real observer_period = Real{0};
    /// Several controller rates (fix-control-tick-semantics): source
    /// j ticks at t_start + k*observer_periods[j], ceil(span/T_j)
    /// times, through `tick_fn(j, t, x)`; sources due at one instant
    /// fire in index order. Non-empty overrides `observer_period`.
    std::vector<Real> observer_periods;
    /// Per-source tick offset (add-unified-event-queue): source j ticks
    /// at t_start + offset_j + k*T_j, ceil((span - offset_j)/T_j) times.
    /// Empty = every offset 0; else as long as `observer_periods`.
    std::vector<Real> observer_offsets;
    /// Empty = none; else per source: true = a Monitor (events::Kind::
    /// Monitor, never landed, never invalidates the gate, called at the
    /// first accepted step at or after each instant).
    std::vector<char> observer_monitors;
    /// The tick grid's origin (add-unified-event-queue): ticks at
    /// (observer_origin + offset_j) + k*T_j, from the first one >= t_start.
    /// NaN (default) = t_start. A resumed run passes the original run's
    /// t_start so it ticks bit-exactly where the uninterrupted run would.
    Real observer_origin = std::numeric_limits<Real>::quiet_NaN();
    /// Empty = `observer_origin` for every source; else one origin per
    /// source (a resume passes its snapshot's `tick_origin`).
    std::vector<Real> observer_origins;
    /// With observer_origins: each source's first unfired index from the
    /// snapshot (`tick_next_k`), empty = recompute from t_start.
    std::vector<std::uint64_t> observer_next_k;
    /// Newton settings, used only when the circuit has nonlinear
    /// devices. Same knobs and defaults as the fixed engine. The
    /// tolerances are absolute; loosening them is not a fix for an
    /// h_min abort (see SimulationOptions::tol_newton_dx).
    Size max_newton_iterations = sim::Defaults::max_newton_iterations;
    Real tol_newton_dx  = sim::Defaults::tol_newton_dx;
    Real tol_newton_res = sim::Defaults::tol_newton_res;
    bool enable_newton_line_search = false;
    bool enable_newton_lm = false;
    /// Post-event restart (update-trbdf2-post-event-restart). After
    /// every gate-edge landing the edge cascade's probe (a trapezoidal
    /// solve of a few femtoseconds under the post-edge mask) is
    /// committed as a step, so the next step starts from post-edge
    /// derivatives; a gate edge settled at a step boundary replaces the
    /// derivative slots from its probe. Not at diode events or the
    /// initial settle (see the engine). The post-gate-edge step memory
    /// stores the controller's proposal instead of the accepted step.
    /// `false` reproduces the previous step sequence exactly.
    bool refresh_after_event = true;
    /// Transient Newton roundoff acceptance (add-newton-roundoff-
    /// acceptance) on every solve of this engine. C++ only: a test and
    /// bisect handle (`false` = abort or reject at an exhausted budget,
    /// as before that change).
    bool newton_roundoff_acceptance = true;
    /// Newton roundoff-acceptance run budget: the most consecutive
    /// solves that may be accepted at roundoff with no converged solve
    /// between them (add-newton-roundoff-acceptance DECISIONS 10d). One
    /// more raises NewtonRoundoffBudgetExceeded, which no retry ladder
    /// swallows. A test handle; 130 = 10x the worst honest run.
    Size newton_roundoff_budget = sim::Defaults::newton_roundoff_budget;
    /// L-stable restart (maintainer decision 1 of the TR-BDF2 follow-up):
    /// the number of Alexander SDIRK(2,2) steps taken after a diode
    /// event and after the initial diode settle, instead of TR-BDF2
    /// steps that read the stale pre-event derivative slots. Same factor
    /// as a TR-BDF2 step (gamma_s = gamma/2). 0 = previous behaviour.
    /// Circuits with a stateful device history (Coss, Lauritzen, IGBT
    /// tail, PMSM, saturable inductor) take none yet. One after a diode
    /// event and none after the settle won the baseline table, on
    /// circuits with a nonlinear refresh (a Newton device law) only
    /// (DECISIONS item 9);
    /// C++-only handles.
    Size sdirk_after_diode_event = 1;
    Size sdirk_after_settle = 0;
    /// Also on piecewise-linear circuits (no Newton law): off, it moved
    /// the bridges' event sequences and gained nothing there.
    bool sdirk_on_linear_circuits = false;
    /// Hosea-Shampine filter W^-1 est on the stage factor (linear circuits
    /// only). C++-only, off: it lets the bridge rectifier take steps that
    /// chatter a diode 4334 times in 2 ms (DECISIONS item 10).
    bool filter_error_estimate = false;
    /// One SDIRK(2,2) step, its estimate filtered, after a turn-off of a
    /// diode with g_off > 0 on a linear circuit (add-branch-form-ideal-switch
    /// DECISIONS 24). C++-only; false restores the previous run.
    bool sdirk_after_turn_off = true;
    /// add-branch-form-ideal-switch D4: an impulsive re-initialisation
    /// raises UnsupportedError(feature='impulse') instead of being recorded
    /// (simulate(on_impulse='error')).
    bool on_impulse_error = false;
    /// Diode mode selection at the zero-time cascades (add-complementarity-
    /// mode-selection 4.2): 0 the settle, 1 'shadow' (the LCP beside it,
    /// recorded only), 2 'lcp' (the LCP decides where the settle's answer
    /// can differ from it).
    int mode_selection = 0;
    /// The estimate's round-off floor (add-high-order-stiff-integrator
    /// DECISIONS 17): 0 TR-BDF2's rule as before (an all-positive noise
    /// right-hand side), 1 alternating signs. ESDIRK always takes 1.
    int roundoff_floor = 0;
    /// Diagnostic (low-level binding only): record each forced accept in
    /// `TrBdf2Stats::forced_trace`. Off: nothing is recorded.
    bool trace_forced = false;
    /// The integrator of the variable step (add-high-order-stiff-integrator):
    /// 0 TR-BDF2, 1 ESDIRK4(3)6L[2]SA (`simulate(integrator='esdirk43')`).
    /// The phases are shared; only the stages, the estimate and the history
    /// commit differ (DECISIONS 15).
    int integrator = 0;
};

struct TrBdf2Stats {
    Size n_accept = 0;
    Size n_reject = 0;
    Size n_gate_events = 0;
    Size n_diode_events = 0;
    Size n_solves = 0;
    /// Steps where a diode kept flipping at ONE time point until
    /// the chatter guard pushed through — the signature of an
    /// un-hysteresed (V_th = 0) diode RIDING its conduction
    /// boundary (a sliding-mode model). The Python layer warns.
    Size n_chatter_breaks = 0;
    /// Steps ACCEPTED with err > 1 because h had already hit
    /// h_min (typically the sliver between two events), or the
    /// span*1e-7 floor that holds for span*1e-4 after the chatter
    /// guard gave up. Zero on a
    /// healthy run; the Python layer warns when it is not.
    Size n_forced_accepts = 0;
    /// Controller ticks fired, all sources: ceil(span / T_j) per
    /// source (tick 0 at t_start, none at t_end).
    Size n_ctrl_ticks = 0;
    /// Per-source tick counts, in source order.
    std::vector<Size> ctrl_tick_counts;
    /// Steps retaken at a smaller h because Newton did not
    /// converge — the variable-step engine's answer to a hard
    /// nonlinear step, where the fixed one can only abort.
    Size n_newton_retries = 0;
    /// Narrowest gate pulse (one switch bit's time between two of its
    /// own toggles) the run saw from an opaque switch_fn, in seconds;
    /// 0 = none. It caps the step at twice that width, so the
    /// switch_fn samples a step takes (every h/4) stay half a pulse
    /// apart.
    Real gate_pulse_min = 0;
    /// Gate pulses that turned out NARROWER than the switch_fn
    /// sampling spacing (h/4) used while that pulse was open: it was
    /// landed by luck, so pulses that narrow may have been stepped
    /// over before the width was known. The Python layer warns. An
    /// isolated (one-shot) pulse narrower than h_max/4 that is never
    /// sampled is NOT counted — nothing saw it.
    Size n_gate_undersampled = 0;
    /// Largest switch_fn sampling spacing (h/4) the run used.
    Real gate_sample_spacing_max = 0;
    /// Steps that began with a diode held in a state its commutation
    /// rule forbids (on the bits the last decision used) and so were
    /// re-decided by a zero-time cascade before stepping.
    Size n_held_diode_redecided = 0;
    /// Gate edges found at a step boundary that no step had landed
    /// (an edge within an ulp of a step's end, e.g. one a controller
    /// tick ended the step on): the diodes were re-settled under the
    /// new gates before stepping, as at a landed edge.
    Size n_gate_boundary_settles = 0;
    /// The probe step a singular factorization forced (0 = the default
    /// h_probe_floor was enough). See `probe_solve`.
    Real probe_floor_used = 0;
    /// Post-gate-event restarts (see TrBdf2Options::refresh_after_event;
    /// each commits the edge probe, which adds one recorded sample and
    /// is not counted in n_accept), the probe solves taken only for a
    /// restart (no cascade ran, or it did not settle), and the smallest
    /// accepted step after a restart (0 = none).
    Size n_derivative_refreshes = 0;
    Size n_refresh_probes = 0;
    Real h_post_event_min = 0;
    /// What set the step ceiling h_max (update-trbdf2-error-control):
    /// 0 = span/1000, 1 = a source's period or pulse width,
    /// 2 = the user's h_max. The derived floors (h_min, probe floor,
    /// chatter floor, gate-pulse floor) scale with it, not with the
    /// span.
    Size h_max_basis = 0;
    /// Steps clamped to a source breakpoint (a pulse or PWM-source
    /// corner) and landed on it.
    Size n_source_breakpoints = 0;
    /// Probe solves retaken at 100x the step because Newton did not
    /// converge (see `probe_solve`).
    Size n_probe_newton_retries = 0;
    /// Newton solves before the first accepted step that failed from the
    /// start state and converged from the step's linear solve.
    Size n_newton_start_retries = 0;
    /// Accepted SDIRK(2,2) restart steps (see TrBdf2Options).
    Size n_sdirk_steps = 0;
    /// Of those, the re-initialisation steps after entering a mode with
    /// dependent states (add-branch-form-ideal-switch D4): three per entry.
    Size n_reinit_steps = 0;
    /// Steps whose error estimate was filtered (W^-1 est).
    Size n_filtered_estimates = 0;
    /// Secant (no-solve) diode turn-off landings that a probe floor moved
    /// past the secant root (their state is taken at the moved time).
    Size n_interp_floor_landings = 0;
    /// Steps accepted because their error estimate was within the
    /// estimate's own roundoff floor (eps * J^-1 (|J||x| + |b|) read
    /// through each entry), not within atol + rtol*|y|: the requested
    /// tolerance is below what the solve can resolve at that state.
    Size n_roundoff_floor_accepts = 0;
    /// With `trace_forced`: per forced accept {t, h, err, who, w}, `who` the
    /// branch id of the entry with the largest |estimate / scale| (or
    /// −1 − r for stateful q row r) and `w` that ratio.
    std::vector<std::array<Real, 5>> forced_trace;
    /// Diode flips a zero-time cascade or a landing declined because the
    /// crossing signal was within that solve's own round-off bound (a
    /// node held only by OFF conductances at a tiny probe step).
    Size n_noise_held_decisions = 0;
    /// Probes retaken 10x longer because a flip they proposed was inside
    /// their own round-off bound (see decide_probe).
    Size n_probe_noise_retakes = 0;
    /// Linear-cache factor traffic of the run (`solve_at`): fresh
    /// (mask, dt) entries (analysis + factor), numeric refactors of a
    /// known mask at a new dt, and solves on the resident factor.
    Size n_factor_builds = 0;
    Size n_factor_refactors = 0;
    Size n_factor_hits = 0;
    /// Steps retaken longer because their factorization was singular (a
    /// step below the matrix's conditioning floor), and the step floor
    /// that learned (0 = none).
    Size n_conditioning_rejects = 0;
    Real h_conditioning_floor = 0;
};

namespace detail_trbdf2 {

// The diode crossing signal, its forbidden side and its noise band are the
// mode-set service's (add-unified-event-queue 8.2).
using events::crossing_signal;
using events::forbidden;
using events::held_tolerance;

}  // namespace detail_trbdf2

/// Variable-step TR-BDF2 transient. Returns the trace on the
/// ACCEPTED (irregular) time grid; SimulationResult's containers
/// are already grid-agnostic and every name-based accessor
/// downstream consumes `times` as data.
inline SimulationResult run_transient_trbdf2(
    pwl::PwlStateSpaceCache& cache,
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const TrBdf2Options& opts,
    const std::function<topology::SwitchStateMask(Real)>& switch_fn,
    const std::function<Vector(Real)>& b_extra_fn = {},
    const std::optional<Vector>& initial_state = std::nullopt,
    TrBdf2Stats* stats_out = nullptr,
    /// Optional analytic gate-edge oracle (e.g. NativePwm2Switch::
    /// next_edge_after): returns the first switch_fn edge strictly
    /// after t. When given, edge landing costs ZERO switch_fn
    /// probes and no bisection.
    const std::function<Real(Real)>& next_edge_fn = {},
    /// Cancellation hook, polled once per step attempt. Returning
    /// false ends the run early and KEEPS the partial trace (the
    /// pwl engine's contract).
    const std::function<bool()>& should_continue = {},
    /// Fired at every k·T_ctrl with the state at that instant,
    /// BEFORE the mask for the coming step is sampled — so a
    /// controller can set a new duty and have this step see it
    /// (the fixed engine's step_observer contract, on an exact
    /// cadence).
    const std::function<void(Real, const Vector&)>& observer_fn =
        {},
    /// Per-Newton-iteration re-stamp of the nonlinear devices —
    /// the SAME callback the fixed-step engine uses. When present
    /// every stage solve becomes a Newton solve.
    const pwl::NonlinearRefreshFn& nl_refresh = {},
    /// v2.0 Phase 4 — resume from a previous run's `final_snapshot`
    /// (either engine's): x, the companion histories of every
    /// dynamic device, and the diode bits. A TR-BDF2 snapshot also
    /// carries the step controller's memory (`controller`), which the
    /// segment continues from; without it (another engine's snapshot, a
    /// pickle) the segment restarts from h_init. Exact in state, to the
    /// run's own tolerance in trajectory (DECISIONS 8 of
    /// refactor-unified-transient-stepper lists what is not carried).
    const SolverSnapshot* resume_from = nullptr,
    /// Multi-rate ticks: called as tick_fn(j, t, x) for source j of
    /// opts.observer_periods. Takes the place of observer_fn there.
    const std::function<void(Size, Real, const Vector&)>& tick_fn = {},
    /// Accepted-step / commutation listeners (events/step_hooks.hpp).
    events::StepHooks* hooks = nullptr) {
    using topology::SwitchStateMask;

    if (!(opts.t_end > opts.t_start)) {
        throw std::invalid_argument(
            "run_transient_trbdf2: t_end must exceed t_start");
    }
    if (!switch_fn) {
        throw std::invalid_argument(
            "run_transient_trbdf2: switch_fn is required (pass an "
            "all-open mask function for uncontrolled circuits)");
    }
    // add-in-kernel-event-losses 3.1: the loss ports' tracker (null: none)
    std::optional<LossTracker> loss_tracker;
    if (!pool.device_losses().empty()) loss_tracker.emplace(graph, pool, "trbdf2");
    LossTracker* const lt = loss_tracker ? &*loss_tracker : nullptr;

    // ---- scope census ----
    // Nonlinear devices are fine: each stage becomes a Newton
    // solve on the same companion matrix (they are memoryless
    // resistive I-V elements — no dt appears in their refresh, so
    // nothing about them cares that h varies). The SATURABLE
    // inductor used to be refused here: its stamp divides by the
    // step and its flux history had no snapshot/restore. Both are
    // fixed — the state is now λ(i) with a derived BDF2 stage, and
    // the history snapshots like the other four — so the refusal is
    // gone rather than relaxed.
    // The static Newton chain (smooth / Shockley diodes, MOSFET,
    // IGBT) is the kernel's: a caller that supplies none gets the
    // standard one, so the stages never solve the linear skeleton.
    // The stateful kinds bring their own wrappers below.
    const pwl::NonlinearRefreshFn nl_static =
        pwl::static_device_refresh(pool, nl_refresh, opts.device_path);

    const Real span  = opts.t_end - opts.t_start;
    // Default step ceiling. span/1000 alone is a TRAP on a long
    // run: it is the sampling resolution for every discontinuity
    // the controller cannot see between steps, and it grows with
    // t_end. A 10 ms run of a 100 kHz narrow-pulse source got
    // h_max = 10 µs — one full period — and the engine stepped
    // straight over every pulse: a peak detector read 0 V instead
    // of 9.87 V, silently. So the default also respects the
    // FASTEST periodic source in the circuit: at least 20 steps
    // per period, and at most a third of the narrowest pulse.
    Real h_default = span / Real{1000};
    // Source breakpoints: the corners of every pulse source and the
    // edges of every PWM voltage source, per period. A step is clamped
    // to land on the next one (like a gate edge), so the error estimate
    // never straddles a jump or a ramp kink: a pulse-driven RC took 32
    // rejects and steps of 1.4e-14 s at its 3.3 us jump (p13b.py).
    struct SourceCorners {
        Real t0, period;   // period 0 = single shot
        std::array<Real, 4> off;
        int n;
    };
    std::vector<SourceCorners> corners;
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (graph.branch(b_id).kind != topology::BranchKind::Source) {
            continue;
        }
        switch (pool.kind_of(b_id)) {
        case pwl::DevicePool::StoredKind::PWMVoltageSource: {
            const auto& pp = pool.pwm_voltage_source_params(b_id);
            if (pp.frequency > Real{0}) {
                h_default = std::min(
                    h_default, Real{1} / (Real{20} * pp.frequency));
                const Real Tp = Real{1} / pp.frequency;
                SourceCorners c{pp.phase, Tp, {}, 0};
                c.off[static_cast<std::size_t>(c.n++)] = Real{0};
                if (pp.duty > Real{0} && pp.duty < Real{1}) {
                    c.off[static_cast<std::size_t>(c.n++)] = pp.duty * Tp;
                }
                corners.push_back(c);
            }
            break;
        }
        case pwl::DevicePool::StoredKind::SineVoltageSource: {
            const auto& sp = pool.sine_voltage_source_params(b_id);
            if (sp.frequency > Real{0}) {
                h_default = std::min(
                    h_default, Real{1} / (Real{20} * sp.frequency));
            }
            break;
        }
        case pwl::DevicePool::StoredKind::PulseVoltageSource: {
            const auto& up = pool.pulse_voltage_source_params(b_id);
            if (up.period > Real{0}) {
                h_default = std::min(
                    h_default, up.period / Real{20});
            }
            if (up.pulse_width > Real{0}) {
                h_default = std::min(
                    h_default, up.pulse_width / Real{3});
            }
            SourceCorners c{up.t_start, up.period, {}, 0};
            Real o = Real{0};
            for (const Real d : {Real{0}, up.rise_time, up.pulse_width,
                                 up.fall_time}) {
                o += d;
                if (c.n == 0 || o > c.off[static_cast<std::size_t>(c.n - 1)]) {
                    c.off[static_cast<std::size_t>(c.n++)] = o;
                }
            }
            corners.push_back(c);
            break;
        }
        // add-controlled-source-set: the same caps for the current
        // sources (their corners are also landed on, below).
        case pwl::DevicePool::StoredKind::SineCurrentSource: {
            const Real f = pool.sine_current_source_params(b_id).w.frequency;
            if (f > Real{0}) {
                h_default = std::min(h_default, Real{1} / (Real{20} * f));
            }
            break;
        }
        case pwl::DevicePool::StoredKind::PulseCurrentSource: {
            const auto& up = pool.pulse_current_source_params(b_id).w;
            if (up.period > Real{0}) {
                h_default = std::min(h_default, up.period / Real{20});
            }
            if (up.pulse_width > Real{0}) {
                h_default = std::min(h_default, up.pulse_width / Real{3});
            }
            break;
        }
        default: break;
        }
    }
    const Real h_max = opts.h_max > Real{0} ? opts.h_max
                                             : h_default;
    // Span-free floors (update-trbdf2-error-control D3): every floor
    // scales with the resolved ceiling instead of t_end - t_start, as
    // the old span fraction of span_ref = 1000 * h_max. While the
    // ceiling IS span/1000, span_ref is the span itself, so the floors
    // are bit-identical to before (a roundoff-level change of h_chatter
    // tipped a diode bridge into a max_steps grind). The smallest step
    // that still advances the clock bounds h_min from below.
    const Real span_ref = (h_max == span / Real{1000})
        ? span : h_max * Real{1000};
    const Real t_ulp = Real{16} * std::numeric_limits<Real>::epsilon()
        * std::max(std::abs(opts.t_start), std::abs(opts.t_end));
    const Real h_min = opts.h_min > Real{0}
        ? opts.h_min : std::max(t_ulp, span_ref * Real{1e-11});
    // Probe steps are a DIFFERENT quantity from the step floor.
    // A probe is a throwaway solve whose only job is to pin (v, i)
    // and ask the diodes which way they conduct — nothing is
    // committed, so it wants to advance as little time as
    // possible. The accepted-step floor wants the opposite (a
    // committed step at 2C/h = 1e11 S is meaningless). Conflating
    // them cost 1.2 mV and 100x the chatter on a buck when the
    // step floor was raised.
    const Real h_probe_floor = std::min(h_min, span_ref * Real{1e-12});
    const Real gamma = pwl::HistoryState::trbdf2_gamma();
    const Real c1    = pwl::HistoryState::trbdf2_c1();
    const Real c2    = pwl::HistoryState::trbdf2_c2();
    const Real c3    = pwl::HistoryState::trbdf2_c3();
    const Real clte  = pwl::HistoryState::trbdf2_lte_const();

    pwl::HistoryState history(graph, pool);
    pwl::DiodeEventState diodes(graph, pool);
    // Bits the solver owns outright; a gated device's bit is its gate,
    // driven by switch_fn (add-pwl-igbt-and-thyristor).
    const SwitchStateMask diode_owned =
        diodes.diode_owned_bits().overlay(SwitchStateMask(
            static_cast<std::size_t>(graph.num_switches())),
            diodes.gated_bits());
    const bool has_diodes = diodes.entries().size() > 0;
    // add-branch-form-ideal-switch D3/D4 on this engine: the per-mode
    // structural pass (named refusals; ON ideal diodes closing a stiff loop
    // released OFF) and the re-initialisation record, shared with the fixed
    // engine. Inactive, and free, without an ideal branch-form element.
    pwl::BranchStructureChecker structure{graph, pool, "trbdf2"};
    pwl::ReinitTracker reinit{graph, pool, structure.active()};
    std::vector<Index> loop_released;
    Real loop_release_t = -std::numeric_limits<Real>::infinity();
    Size n_loop_releases = 0;
    const Real* t_now = nullptr;   // the step loop's clock, once it exists

    const Size state_size = pool.state_size(graph);
    Vector x = Vector::Zero(static_cast<Index>(state_size));
    if (initial_state.has_value()) {
        if (initial_state->size()
            != static_cast<Index>(state_size)) {
            throw std::invalid_argument(
                "run_transient_trbdf2: initial_state has size "
                + std::to_string(initial_state->size())
                + ", expected " + std::to_string(state_size));
        }
        x = *initial_state;
        history.seed_from_dc_op(x);
    }
    // The five stateful-device histories as one object
    // (add-unified-device-interface step 1); seeded from initial_state,
    // then the snapshot, which WINS: it is the state. Each is committed
    // only on ACCEPT under the rule its last solve stamped, so a
    // rejected step never touches them and no rollback is needed.
    pwl::StatefulHistories hs;
    hs.get<pwl::ExternalDeviceHistory>().set_start_time(opts.t_start);   // ctx->t of init
    // #31 device_state once per committed time: every commit site fires it,
    // and the iteration's end (tw_on_step) only if its last commit did not
    Real ds_t = std::numeric_limits<Real>::quiet_NaN();
    const bool hook_ds = hooks != nullptr && hooks->has_device_state();   // run-invariant
    auto device_state_once = [&](Real tc, const Vector& xc) {
        if (!hook_ds || tc == ds_t) return;
        ds_t = tc;
        hooks->device_state(tc, xc, hs);
    };
    hs.init(graph, pool);
    hs.set_path(opts.device_path, graph, pool, "run_transient_trbdf2");
    if (initial_state.has_value()) hs.seed_from_dc_op(x);
    if (resume_from != nullptr) {
        if (!resume_from->valid) {
            throw std::invalid_argument(
                "run_transient_trbdf2(resume_from=...): the snapshot "
                "is not valid — it came from a run that produced none "
                "(an aborted run, or a SolverSnapshot() built by hand "
                "and never populated).");
        }
        if (resume_from->x.size() != static_cast<Index>(state_size)) {
            throw std::invalid_argument(
                "run_transient_trbdf2(resume_from=...): the snapshot's "
                "state vector has "
                + std::to_string(resume_from->x.size())
                + " entries but this circuit has "
                + std::to_string(state_size)
                + " — it belongs to a different circuit.");
        }
        // Resume WINS over initial_state: the snapshot is the state.
        x = resume_from->x;
        // the device-state layout check comes before the mode bits
        hs.load(*resume_from, history, "run_transient_trbdf2");
        if (!resume_from->mask.empty()) {
            diodes.restore_on_bits(resume_from->mask);
        }
    }

    // Built-in time-varying source census + reusable buffers
    // (same pattern as run_transient).
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
        default: break;
        }
    }
    Vector be_pwm, be_sine, be_pulse, b_src, b_stage, x_g, x_1,
        x_trial;
    std::vector<std::pair<Index, Index>> sine_vars;   // (branch, source row)
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& br = graph.branch(b_id);
        if (br.kind == topology::BranchKind::Source
            && pool.kind_of(br.id)
                   == pwl::DevicePool::StoredKind::SineVoltageSource) {
            sine_vars.emplace_back(br.id,
                                   pool.branch_var_id_for_source(br.id, graph));
        }
    }
    const bool has_iwave =
        sources::has_current_waveform_sources(pool, graph);
    Vector be_iwave;

    // Source-only overlay at time `at_t` (no history terms).
    auto accumulate_sources = [&](Real at_t, Vector& out) {
        out.setZero(static_cast<Index>(state_size));
        if (b_extra_fn) {
            out += b_extra_fn(at_t);
        }
        if (has_pwm) {
            sources::compute_pwm_b_extra(pool, graph, at_t, be_pwm);
            out += be_pwm;
        }
        if (has_sine) {
            // compute_sine_b_extra's arithmetic on the sine branches found
            // once (it scanned every branch with a pool lookup per call:
            // 9 % of the bridge rectifier's run time).
            be_sine.setZero(static_cast<Index>(state_size));
            for (const auto& [b_id, var] : sine_vars) {
                be_sine[var] += -models::SineVoltageSource::value_at(
                    pool.sine_voltage_source_params(b_id), at_t);
            }
            out += be_sine;
        }
        if (has_pulse) {
            sources::compute_pulse_b_extra(pool, graph, at_t,
                                            be_pulse);
            out += be_pulse;
        }
        if (has_iwave) {
            sources::compute_current_waveform_b_extra(pool, graph, at_t,
                                                      be_iwave);
            out += be_iwave;
        }
        if (lt) lt->inject(out);   // the heat owed, on every solve
    };

    // ---- the single solve point ----
    // Linear circuits go through the cache (one numeric factor per
    // (mask, dt), symbolic analysis retained). With nonlinear
    // devices each solve becomes a Newton loop on an assembled
    // companion — no factor is wasted, because Newton refactorizes
    // J_lin + J_nl every iteration anyway, so the segment is only
    // a carrier for (J, b). Both TR-BDF2 stages of a step use the
    // SAME dt = γh, so the assembly is reused between them.
    // ---- charge-based Coss across the two stages ----
    // Stage 1 is trapezoidal at gamma*h; stage 2 needs the BDF2
    // CHARGE history term. The conductance is identical in both
    // (c1/h = 2/(gamma*h)), which is why one factor still serves
    // the pair — but the history term is not, and stamping the
    // trapezoidal one in a BDF2 stage would converge to the wrong
    // answer with every outward sign of health.
    // One step context for every stateful stamp (add-unified-device-
    // interface step 2): every solve, probe and landing sets it once, so
    // no stamp can read a step that this solve did not set.
    pwl::StepCtx step_ctx{Real{0}, pwl::TrBdf2Stage::Trapezoidal, opts.t_start};
    const pwl::NonlinearRefreshFn nl_effective = pwl::build_device_refresh(
        nl_static, hs, step_ctx, /*sat_increment_form=*/true);

    std::string newton_last_error;
    // Roundoff acceptance (add-newton-roundoff-acceptance), on for every
    // transient solve here: stages, probes, landings. Copied to the
    // result at the end.
    pwl::NewtonRoundoffAcceptance newton_roundoff;
    newton_roundoff.enabled = opts.newton_roundoff_acceptance;
    newton_roundoff.max_consecutive = opts.newton_roundoff_budget;
    Size ro_accepts = 0;
    Real ro_omega_max = 0;
    Index ro_row = kInvalidIndex;
    // Until the first accepted step a failed Newton solve is retried once
    // from the step's linear solve (solve_dispatch).
    bool newton_at_start = true;
    Vector nl_start_x;
    Size n_newton_start_retries = 0;
    pwl::PwlSegment nl_seg;
    // One stamp-slot template for every assembly the engine does itself
    // (Newton carrier, round-off bounds): values bit-equal to
    // assemble_segment + compress_in_place, without its per-call
    // allocation (24 % of the bridge rectifier's run time).
    pwl::SegmentAssembler seg_asm;
    pwl::NewtonWorkspace newton_ws;   // one Newton analysis per pattern per run
    Real nl_seg_dt = Real{-1};
    topology::SwitchStateMask nl_seg_mask(
        static_cast<std::size_t>(graph.num_switches()));
    bool nl_seg_valid = false;
    // Returns false when Newton did not converge. The caller
    // REJECTS the step and shrinks h — which is a thing only a
    // variable-step engine can do, and it is why this engine
    // finishes circuits the fixed one aborts on (measured: an
    // ideal-Shockley flyback with a kV leakage spike, which the
    // fixed engine cannot take even split into 64 sub-steps).
    auto solve_dispatch = [&](const topology::SwitchStateMask& m,
                               Real dt_use, const Vector& b_ex,
                               Vector& out_x) -> bool {
        if (!nl_effective) {
            cache.solve_at(m, dt_use, b_ex, out_x);
            return true;
        }
        if (!nl_seg_valid || nl_seg_dt != dt_use
            || !(nl_seg_mask == m)) {
            seg_asm.assemble(graph, pool, m, dt_use, nl_seg.J,
                             nl_seg.b_constant);
            nl_seg.state_size = state_size;
            nl_seg_dt = dt_use;
            nl_seg_mask = m;
            nl_seg_valid = true;
        }
        try {
            out_x = pwl::solve_with_newton_b_extra(
                nl_seg, nl_effective, graph, pool, /*x_init=*/out_x,
                b_ex, opts.max_newton_iterations,
                opts.tol_newton_dx, opts.tol_newton_res,
                opts.enable_newton_line_search,
                opts.enable_newton_lm, &newton_ws, &newton_roundoff);
            if (newton_roundoff.accepted) {
                ++ro_accepts;
                if (newton_roundoff.omega >= ro_omega_max) {
                    ro_omega_max = newton_roundoff.omega;
                    ro_row = newton_roundoff.row;
                }
            }
        } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
            throw;   // a run budget: never retried at a smaller h
        } catch (const std::runtime_error& e) {
            newton_last_error = e.what();
            if (!newton_at_start) return false;
            // From the run's start state the round-off acceptance measures
            // each row against the magnitudes it sums AT THAT STATE: a zero
            // start has none of the 2C/h·v terms whose round-off the
            // residual is (48 V behind 1 µF at h_min: 4.8e-6 A against a
            // scale of 48), so a solve converged in dx to 7e-16 V was
            // refused at every h. Retried once from the step's linear solve
            // (the nonlinear laws left out): a start whose terms are the
            // residual's. That start can still be far off (a diode law at
            // 48 V scales its row to 1e9 A), so a round-off acceptance from
            // it is re-measured by one more solve from the converged state.
            const std::string first_error = newton_last_error;
            try {
                cache.solve_at(m, dt_use, b_ex, nl_start_x);
                if (!nl_start_x.allFinite()) return false;
                out_x = pwl::solve_with_newton_b_extra(
                    nl_seg, nl_effective, graph, pool, nl_start_x, b_ex,
                    opts.max_newton_iterations, opts.tol_newton_dx,
                    opts.tol_newton_res, opts.enable_newton_line_search,
                    opts.enable_newton_lm, &newton_ws, &newton_roundoff);
                if (newton_roundoff.accepted) {
                    nl_start_x = out_x;
                    out_x = pwl::solve_with_newton_b_extra(
                        nl_seg, nl_effective, graph, pool, nl_start_x, b_ex,
                        opts.max_newton_iterations, opts.tol_newton_dx,
                        opts.tol_newton_res, opts.enable_newton_line_search,
                        opts.enable_newton_lm, &newton_ws, &newton_roundoff);
                }
            } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                throw;
            } catch (const std::runtime_error& e2) {
                newton_last_error =
                    first_error + "; retried from the linear solve: " + e2.what();
                return false;
            }
            newton_last_error = first_error;
            ++n_newton_start_retries;
            if (newton_roundoff.accepted) {
                ++ro_accepts;
                if (newton_roundoff.omega >= ro_omega_max) {
                    ro_omega_max = newton_roundoff.omega;
                    ro_row = newton_roundoff.row;
                }
            }
        }
        return true;
    };

    // D3 on mode `m` (cached per mode): a floating island or an ideal
    // voltage loop of gated elements is refused by name; an ON ideal diode
    // that closes a loop of sources and closed ideal elements is turned OFF
    // (its only consistent state there: the CCM turn-on edge of an ideal
    // buck) and `m` re-formed. The same diode released twice at one instant
    // means the loop forward-biases it: refused (DECISIONS #12, #13 H1).
    auto settle_mode = [&](SwitchStateMask& m) {
        if (!structure.active()) return;
        const Real now = t_now != nullptr ? *t_now : opts.t_start;
        if (now != loop_release_t) {
            loop_released.clear();
            loop_release_t = now;
        }
        for (;;) {
            const auto& ms = structure.check(m, true);
            if (ms.loop_diodes.empty()) return;
            for (const Index d : ms.loop_diodes) {
                if (std::find(loop_released.begin(), loop_released.end(), d)
                    == loop_released.end()) {
                    continue;
                }
                throw UnsupportedError(
                    "trbdf2", "ideal_voltage_loop",
                    std::string(graph.branch_name(d)), "Diode",
                    std::format(
                        "engine 'trbdf2': at t = {} the ideal diode '{}' (R_on "
                        "= 0) closes a loop of voltage sources and closed ideal "
                        "elements in mode {}, and the loop forward-biases it "
                        "(turned OFF, it turned back ON): its current is "
                        "undefined. Give it R_on > 0.",
                        now, graph.branch_name(d), m.to_string()));
            }
            std::vector<bool> bits = diodes.snapshot_on_bits();
            const auto ents = diodes.entries();
            for (const Index d : ms.loop_diodes) {
                loop_released.push_back(d);
                for (Size q = 0; q < ents.size(); ++q) {
                    if (ents[q].branch_id != d) continue;
                    bits[q] = false;
                    m.set(ents[q].switch_idx, false);
                }
            }
            diodes.restore_on_bits(bits);
            ++n_loop_releases;
        }
    };

    auto mask_at = [&](Real at_t) {
        SwitchStateMask m = switch_fn(at_t);
        if (has_diodes) {
            m = diodes.combine(m);
        }
        settle_mode(m);
        return m;
    };

    // One TRAP solve over `dt` from the current committed state —
    // used for event localization trials, the event-landing
    // sub-step, and the zero-time initial-bit settle.
    auto trap_solve = [&](const SwitchStateMask& m, Real t0, Real dt,
                           Vector& out_x) -> bool {
        // EVERY trapezoidal solve tells the stateful devices which
        // step it is taking. The linear part always received `dt`
        // through compute_b_extra / solve_dispatch, but the Coss,
        // Lauritzen, IGBT-tail, PMSM and saturable-inductor stamps
        // read the step context, which the step loop assigns only
        // after this point — so the zero-time diode settle above the
        // loop, and every probe / landing solve inside it, stamped
        // those devices with h = 0 (the settle) or with the PREVIOUS
        // stage's h (the probes). A saturable inductor or a PMSM in
        // any circuit with a diode divided by zero in the settle and
        // took the process down (SIGSEGV, measured); a Coss produced
        // NaN diode bits in silence, because the settle discards the
        // solve's return value. Setting it here, at the one
        // choke point all trapezoidal solves pass through, is the
        // same fix run_transient applies to its `step_ctx`.
        step_ctx = step::Trapezoidal::stage(dt).ctx();
        step_ctx.t = t0 + dt;   // the external devices' ctx->t
        history.compute_b_extra(dt, b_stage);
        accumulate_sources(t0 + dt, b_src);
        b_stage += b_src;
        const bool ok = solve_dispatch(m, dt, b_stage, out_x);
        if (stats_out) { ++stats_out->n_solves; }
        return ok;
    };

    // ---- probe solves: a conditioning floor learned on demand ----
    // A probe (the zero-time diode settle, every cascade) is a trap
    // step of h_probe_floor = span·1e-12, whose capacitor companion is
    // 2C/h: 470 µF at 6e-14 s is 1.6e10 S. A subnet tied to the rest of
    // the circuit only through OFF diodes (g_off = 1e-9 S) — a bridge
    // rectifier whose DC side floats — then has a common-mode pivot
    // 1e-19 of the diagonal, and the factorization refuses it
    // ("numerically singular ... node dn"), measured on a 325 V bridge
    // with 470 µF across a floating dc_pos/dc_neg. engine='pwl' runs the
    // same circuit because its dt (1e-6) keeps the companion at 940 S,
    // and no regularizing tie cures it here: a 1 GΩ tie adds 1e-9 S,
    // a 1 MΩ one 1e-6 S, both still below eps against 1.6e10 S. A
    // probe only pins (v, i) to ask the diodes which way they
    // conduct, so a longer probe is still a probe: on a singular
    // factorization retry at 100x the step, up to span·1e-6 (and never
    // past 1 % of h_max), and keep the step that worked as this run's
    // probe floor so the failure is paid once, not at every cascade.
    Real h_probe_learned = Real{0};
    // The learned step floor below which a step's factorization is
    // singular, per step-start mode (the island it comes from exists in
    // that mode only; see stage 1). h_cond: the current mode's (0 = none).
    std::unordered_map<SwitchStateMask, Real> h_cond_by_mode;
    constexpr Size kGrindSteps = 200'000;
    Size grind_n0 = 0;
    Real grind_t0 = opts.t_start;
    Real h_cond = Real{0};
    // Step and outcome of the last probe (the post-event refresh
    // reads the capacitor current back through this step).
    Real last_probe_dt = Real{0};
    bool last_probe_ok = false;
    Size n_probe_newton_retries = 0;
    auto is_singular_error = [](const std::runtime_error& e) {
        return std::string(e.what()).find("numerically singular")
               != std::string::npos;
    };
    // A probe whose NEWTON solve fails (a flux row's 2L/h at a 1e-16 s
    // probe puts one ulp of current far above the absolute tolerances)
    // climbs the same ladder: its out_x is the warm start, not a
    // solution, and deciding diode bits on it was a silent wrong answer
    // (a saturable buck inductor lost 10 A at a turn-off edge: the
    // cascade never turned the freewheeling diode on). A probe that
    // still fails at the cap returns false; no caller may read out_x.
    auto probe_solve = [&](const SwitchStateMask& m, Real t0, Real dt,
                            Vector& out_x) -> bool {
        Real dt_try = std::max(dt, h_probe_learned);
        const Real dt_cap = std::max(
            dt_try, span_ref * Real{1e-6});
        const Vector x_warm = out_x;
        bool learn_next = false;
        for (;;) {
            bool ok = false, singular = false;
            try {
                ok = trap_solve(m, t0, dt_try, out_x);
                // Only a singular factor is learned: it is a property
                // of the mask and the companion size; a Newton failure
                // depends on the state and is retried per probe.
                if (ok && learn_next && dt_try > h_probe_learned) {
                    h_probe_learned = dt_try;
                }
            } catch (const std::runtime_error& e) {
                if (!is_singular_error(e)) {
                    throw;
                }
                if (dt_try >= dt_cap) {
                    // Every probe step up to the cap is below the matrix's
                    // conditioning floor (2C/h against g_off; C = 1 mF and
                    // g_off = 1e-12 S need h ~ 4e-7 s): no probe can say
                    // which way a diode conducts here.
                    throw std::runtime_error(std::format(
                        "run_transient_trbdf2: the diode probe at t = {} is "
                        "numerically singular at every probe step up to {} s "
                        "(the conditioning floor of this circuit is above "
                        "it): {} A node held only by OFF conductances has "
                        "2C/h far above g_off/eps at short steps; give it a "
                        "DC path (a resistor) or raise g_off.",
                        t0, dt_cap, e.what()));
                }
                singular = true;
            }
            last_probe_dt = dt_try;
            last_probe_ok = ok;
            if (ok || dt_try >= dt_cap) {
                return ok;
            }
            if (!singular) ++n_probe_newton_retries;
            learn_next = singular;
            out_x = x_warm;
            dt_try = std::min(dt_try * Real{100}, dt_cap);
        }
    };
    // A zero-time cascade whose probe failed at every step up to the cap
    // has no state to decide the diode bits on: refuse by name.
    auto require_probe = [&](bool ok, const char* where, Real at_t) {
        if (ok) return;
        throw std::runtime_error(
            std::string("run_transient_trbdf2: the zero-time diode cascade "
                        "at ") + where + " t = " + std::to_string(at_t)
            + " has no converged probe solve (Newton failed at every "
              "probe step up to " + std::to_string(last_probe_dt)
            + " s): " + newton_last_error
            + " The diode bits cannot be decided at this event; run "
              "this circuit on engine='pwl' with an explicit dt.");
    };

    // ---- diode decisions above their own round-off ----
    // A probe or landing at a tiny step on a subnet held only by OFF
    // conductances (a bridge whose DC side floats on a 1 MΩ tie) has a
    // common mode that is round-off: 2C/h = 2.8e9 S against 1e-6 S puts
    // eps·J⁻¹(|J||x| + |b|) in the hundreds of volts. Reading a diode to
    // ground from it decided the bits by noise (a landing at 3.3e-13 s
    // put the bus 83 V up and turned the OFF return diode ON; a 1e-15
    // relative change of one resistor then ground the run to max_steps).
    // A flip counts only when the crossing signal is past its threshold
    // by more than that bound, read through the diode's two nodes (the
    // common mode cancels in a diode inside the subnet). The bound costs
    // two solves on the probe's factor and is taken only when a flip is
    // proposed; Newton circuits keep no factor and decide as before.
    sparse::Matrix pn_J;
    Vector pn_bc, pn_r, pn_rhs, pn_y, pn_y0;
    std::vector<Real> pn_bound;
    Size n_noise_held = 0;
    Size n_probe_noise_retakes = 0;
    auto proposes_flip = [&](const Vector& xs) {
        for (const auto& e : diodes.entries()) {
            const Real v = e.law_voltage(xs, e.is_on);
            if (models::SwitchedDiode::decide(e.is_on, v, e.params())
                != e.is_on) {
                return true;
            }
        }
        return false;
    };
    // pn_bound[d] for the state xs solved under (m, dt) with b_stage.
    auto predicate_noise = [&](const Vector& xs, const SwitchStateMask& m,
                               Real dt) {
        const Vector* bc = nullptr;
        const sparse::Matrix* J = cache.resident_event_matrix(m, dt, &bc);
        if (J == nullptr) {
            seg_asm.assemble(graph, pool, m, dt, pn_J, pn_bc);
            J = &pn_J;
            bc = &pn_bc;
        }
        pn_r = J->cwiseAbs() * xs.cwiseAbs();
        pn_r += bc->cwiseAbs() + b_stage.cwiseAbs();
        pn_rhs = -pn_r;
        cache.solve_at(m, dt, pn_rhs, pn_y);
        pn_rhs.setZero(static_cast<Index>(state_size));
        cache.solve_at(m, dt, pn_rhs, pn_y0);
        pn_y -= pn_y0;
        const auto ents = diodes.entries();
        pn_bound.assign(ents.size(), Real{0});
        for (Size d = 0; d < ents.size(); ++d) {
            const auto& e = ents[d];
            if (e.is_on && e.var >= 0) {
                // ON branch form: the signal is the solved current i_s.
                pn_bound[d] = std::numeric_limits<Real>::epsilon()
                    * (std::abs(pn_y[e.var]) + std::abs(xs[e.var]));
                continue;
            }
            pn_bound[d] = std::numeric_limits<Real>::epsilon()
                * (std::abs(stamping::read_node_voltage(pn_y, e.from)
                            - stamping::read_node_voltage(pn_y, e.to))
                   + std::abs(stamping::read_node_voltage(xs, e.from))
                   + std::abs(stamping::read_node_voltage(xs, e.to)));
        }
    };
    auto decide_diodes = [&](const Vector& xs, const SwitchStateMask& m,
                             Real dt) -> bool {
        if (!proposes_flip(xs)) return false;
        if (nl_effective) return diodes.update_from_state(xs);
        predicate_noise(xs, m, dt);
        const bool flipped = diodes.update_from_state(xs, pn_bound);
        if (proposes_flip(xs)) {
            ++n_noise_held;
        }
        return flipped;
    };
    // A flip proposed by xs but inside pn_bound (call predicate_noise
    // first).
    auto noise_holds = [&](const Vector& xs) {
        const auto ents = diodes.entries();
        for (Size d = 0; d < ents.size(); ++d) {
            const auto& e = ents[d];
            const Real v = e.law_voltage(xs, e.is_on);
            if (models::SwitchedDiode::decide(e.is_on, v, e.params())
                    != e.is_on
                && std::abs(models::SwitchedDiode::crossing_signal(
                       e.is_on, v, e.params())) <= pn_bound[d]) {
                return true;
            }
        }
        return false;
    };
    // decide_diodes for the probe xp just solved by probe_solve(m, t0, ·).
    // A probe whose flip is inside its own round-off is no answer, and
    // holding the bit commits a step under it: the SEPIC's turn-off at
    // 9.8 µs probed at span·1e-12 = 2e-15 s, where 2C/h = 1e10 S against
    // the 2e-6 S of two OFF devices makes the coupling-capacitor subnet's
    // common mode undecidable (eps·G/g ≈ 1); the rectifier was held OFF,
    // the committed step drained both inductors through g_off, and the
    // run lost the first cycle's energy (0.17 % on V_out at 2 ms). Such a
    // probe is retaken 10x longer, up to span·1e-6, as a singular one is.
    auto decide_probe = [&](Vector& xp, const SwitchStateMask& m,
                            Real t0) -> bool {
        if (nl_effective || !proposes_flip(xp)) {
            return decide_diodes(xp, m, last_probe_dt);
        }
        const Real dt_cap = span_ref * Real{1e-6};
        for (;;) {
            predicate_noise(xp, m, last_probe_dt);
            const Real dt_next = last_probe_dt * Real{10};
            if (!noise_holds(xp) || dt_next > dt_cap) break;
            const Vector x_prev = xp;
            const Vector b_prev = b_stage;   // mode_site reads the probe's
            const Real dt_prev = last_probe_dt;
            if (!probe_solve(m, t0, dt_next, xp)) {
                // Not reached on a linear factor; if it were, neither the
                // failed state nor its b_stage may be read: keep the bits.
                xp = x_prev;
                b_stage = b_prev;
                last_probe_dt = dt_prev;
                last_probe_ok = true;
                ++n_noise_held;
                return false;
            }
            ++n_probe_noise_retakes;
            if (!proposes_flip(xp)) return false;
        }
        // pn_bound is fresh for (xp, m, last_probe_dt).
        const bool flipped = diodes.update_from_state(xp, pn_bound);
        if (proposes_flip(xp)) ++n_noise_held;
        return flipped;
    };

    // ---- complementarity mode selection at a zero-time cascade ----
    // (add-complementarity-mode-selection 4.2 / 4b.3 on trbdf2; DECISIONS
    // #47.) trbdf2 decides diode modes only at its zero-time cascades (the
    // initial settle, a gate edge, a diode event, a landing): those are its
    // sites. The site's system is the cascade's last probe, a trap step of
    // last_probe_dt from the committed state under the gates of t_gates,
    // whose b_extra (b_stage) does not depend on the mode; the LCP is posed
    // from the mode the cascade started in (bits0).
    //   'shadow': posed at every site beside the finished cascade; nothing
    //             it computes is used (a different mode is recorded).
    //   'lcp':    a settled cascade whose mode's crossing signals, on its
    //             own probe, are clear of zero by kQuietBand of the circuit
    //             scale is a nondegenerate solution of the site's LCP, which
    //             is then unique for a passive network: kept (quiet). Else
    //             (round budget hit, a signal inside the band: a tie, or a
    //             breach of the law) the LCP decides and its mode is probed.
    std::optional<pwl::PwlModePorts> mode_ports;
    events::ModeSelector mode_selector;
    events::PassivityClass mode_passivity;
    std::vector<std::string> mode_diode_names;
    ModeReport mode_rep;
    Vector mode_bx, mode_xj, mode_u, mode_xq;
    DenseMatrix mode_Z;
    std::vector<char> mode_prev_on, mode_coords, mode_settled;
    if (opts.mode_selection < 0 || opts.mode_selection > 2) {
        throw std::invalid_argument(
            "run_transient_trbdf2: mode_selection must be 0 (settle), 1 (shadow) or 2 (lcp)");
    }
    if (opts.mode_selection != 0) {
        const char* mname = opts.mode_selection == 1 ? "shadow" : "lcp";
        auto refuse = [&](const std::string& why) {
            throw UnsupportedError(
                "trbdf2", "mode_selection", "", "",
                std::format("engine 'trbdf2': mode_selection='{}' {} Drop mode_selection.",
                            mname, why));
        };
        if (!has_diodes)
            refuse("decides PWL diodes and this circuit has none: nothing to select.");
        if (nl_effective)
            refuse("with a Newton device in the circuit: the diode port model is "
                   "linear, a Newton device's stamp moves with its operating point.");
        if (opts.max_event_iterations == 0)
            refuse("with max_event_iterations=0: the diode modes are frozen, nothing "
                   "is decided.");
        for (const auto& e : diodes.entries()) {
            if (e.gate == models::SwitchedDiode::Gate::None) continue;
            // the port and judge_mode know no gate (pwl refuses the same)
            throw UnsupportedError(
                "trbdf2", "mode_selection", std::string(graph.branch_name(e.branch_id)),
                "Diode",
                std::format("engine 'trbdf2': mode_selection='{}' with the gated device "
                            "'{}' (IGBT with reverse blocking, GTO, thyristor or TRIAC): "
                            "the complementarity port knows no gate. Drop mode_selection.",
                            mname, graph.branch_name(e.branch_id)));
        }
        mode_ports.emplace(cache, graph, pool, diodes);
        mode_passivity = pwl::passivity_of(graph, pool);
        mode_selector.passive = mode_passivity.passive();
        for (const auto& e : diodes.entries())
            mode_diode_names.push_back(pwl::branch_label(graph, e.branch_id));
        mode_rep.mode = opts.mode_selection;
        mode_rep.passivity = mode_passivity.name();
        mode_rep.passivity_device = mode_passivity.device;
        mode_rep.passivity_kind = mode_passivity.kind;
    }
    // The site's mask `base` (its gates) with diode bits `on` (no gated
    // diode here: refused above). No switch_fn call: 'shadow' moves nothing.
    SwitchStateMask mode_dm(static_cast<std::size_t>(graph.num_switches()));
    auto mode_mask_at = [&](const SwitchStateMask& base, const std::vector<char>& on) {
        const auto& es = diodes.entries();
        for (std::size_t j = 0; j < es.size(); ++j) mode_dm.set(es[j].switch_idx, on[j] != 0);
        return base.overlay(mode_dm, diodes.diode_owned_bits());
    };
    // Every diode's crossing signal on xs (solved under mask m with the
    // diodes' current bits) clear of zero by kQuietBand of the circuit
    // scale, the floating islands' KCL imposed first (pwl's quiet test).
    auto mode_quiet = [&](const Vector& xs, const SwitchStateMask& m, Real dt,
                          Real h_site, const Vector& x0) {
        constexpr Real kQuietBand = Real{1e-8};
        const Vector* xq = &xs;
        if (mode_ports->has_islands(m)) {
            mode_xq = xs;
            mode_ports->correct(m, dt, mode_bx, mode_xq);
            xq = &mode_xq;
        }
        const auto sc = mode_ports->scale(x0);
        const Real band_v = kQuietBand * sc.vscale;
        const Real band_i = kQuietBand * sc.iscale + band_v * sc.c_max / h_site;
        for (const auto& e : diodes.entries()) {
            const auto p = e.params();
            const Real v = e.law_voltage(*xq, e.is_on);
            if (e.is_on ? !(models::SwitchedDiode::branch_current(true, v, p) > band_i)
                        : !(v - models::SwitchedDiode::turn_on_voltage(p) < -band_v)) {
                return false;
            }
        }
        return true;   // x_trial: the probe of the selected mode
    };
    auto mode_note = [&](const events::ModeSelection& s) {
        auto& r = mode_rep;
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
        r.port_solves = mode_ports->solves();
        r.port_matrices = mode_ports->port_builds();
    };
    auto mode_str = [](const auto& on) {
        std::string s;
        for (const auto c : on) s += c ? '1' : '0';
        return s;
    };
    // The site after a cascade at t0 (gates of t_gates) that started from
    // bits0; `settled`: its last probe flipped nothing (x_trial is then the
    // solve of the final bits); `h_site`: the step the next site is away.
    // Returns true when 'lcp' re-probed (x_trial / m_site: its mode; the
    // caller takes x_sig from it); `settled` is then true.
    auto mode_site = [&](Real t0, Real t_gates, const std::vector<bool>& bits0,
                         bool& settled, Real h_site, SwitchStateMask& m_site) -> bool {
        // m_site: the cascade's last mask (its gates are the site's)
        if (!mode_ports || !last_probe_ok) return false;
        ++mode_rep.sites_restart;
        const Real dt_site = last_probe_dt;
        mode_bx = b_stage;      // the last probe's b_extra (mode-independent)
        const auto bits_now = diodes.snapshot_on_bits();
        mode_settled.assign(bits_now.begin(), bits_now.end());
        if (opts.mode_selection == 2 && settled &&
            mode_quiet(x_trial, mode_mask_at(m_site, mode_settled), dt_site, h_site, x)) {
            ++mode_rep.quiet_sites;
            return false;
        }
        mode_prev_on.assign(bits0.begin(), bits0.end());
        events::ModeSelection sel;
        auto pose = [&] {
            return pwl::select_site(
                mode_selector, *mode_ports, cache, diodes,
                [&](const std::vector<char>& on) { return mode_mask_at(m_site, on); },
                mode_prev_on, x, dt_site, h_site, mode_bx, mode_Z, mode_u, mode_xj,
                mode_coords);
        };
        if (opts.mode_selection == 1) {
            try {
                sel = pose();
                mode_note(sel);
                if (sel.status != events::LcpStatus::Solution &&
                    sel.status != events::LcpStatus::Trivial) {
                    ++mode_rep.errors;
                    mode_rep.last_error = std::string(events::to_string(sel.status));
                } else if (sel.on != mode_settled) {
                    {
                        ModeDisagreement d;
                        d.t = t0;
                        d.site = "event";
                        d.settle_mode = mode_str(mode_settled);
                        d.lcp_mode = mode_str(sel.on);
                        d.previous_mode = mode_str(mode_prev_on);
                        d.pivots = sel.pivots_total;
                        d.status = events::to_string(sel.status);
                        const auto sc = mode_ports->scale(x);
                        if (settled)
                            d.settle_consistent = pwl::judge_mode(
                                diodes, mode_settled, x_trial, sc, mode_selector.tol,
                                &d.settle_violation);
                        cache.solve_at(mode_mask_at(m_site, sel.on), dt_site, mode_bx,
                                       mode_xj);
                        d.lcp_consistent = pwl::judge_mode(
                            diodes, sel.on, mode_xj, sc, mode_selector.tol,
                            &d.lcp_violation);
                        mode_rep.disagreements.push_back(std::move(d));
                    }
                }
            } catch (const std::exception& e) {
                ++mode_rep.errors;
                mode_rep.last_error = e.what();
            }
            return false;
        }
        try {
            sel = pose();
        } catch (const std::runtime_error& e) {
            // A candidate mode singular at the probe's femtoseconds (a node
            // held only by OFF diodes; the probe ladder retried the
            // cascade's own mode): the settle's answer stands, counted.
            if (!is_singular_error(e)) throw;
            ++mode_rep.errors;
            mode_rep.last_error = e.what();
            return false;
        }
        mode_note(sel);
        events::raise_if_refused(sel, "trbdf2", t0, mode_diode_names, mode_passivity);
        diodes.restore_on_bits(std::vector<bool>(sel.on.begin(), sel.on.end()));
        m_site = mask_at(t_gates);
        require_probe(probe_solve(m_site, t0, dt_site, x_trial),
                      "a mode selection,", t0);
        const auto now = diodes.snapshot_on_bits();
        mode_settled.assign(now.begin(), now.end());
        // a mode its own probe violates is not committed as settled (the
        // gate-edge restart then re-probes instead of committing x_trial)
        settled = pwl::judge_mode(diodes, mode_settled, x_trial, mode_ports->scale(x),
                                  mode_selector.tol) == 1;
        if (!settled) ++mode_rep.inconsistent;
        return true;   // x_trial: the probe of the selected mode
    };

    // Mask recorded with sample 0 (SimulationResult::
    // switch_state_bits): the bits the run STARTS from, taken before
    // the settle below re-decides them. A zero IC is consistent with
    // every mask whose diodes are OFF (v = 0, so i = g·v = 0; an ON
    // diode with a forward drop would carry −g_on·V_f) — and every
    // diode starts OFF. A `resume_from` state is the
    // previous run's last solve and records the mask the snapshot
    // says it was stamped under (`stamped_mask`), NOT mask_at(t_start):
    // a run that ended exactly on a gate edge stored that sample under
    // the pre-edge mask, then re-decided the diodes under the
    // post-edge one, so diode_on + switch_fn(t_start) is the post-edge
    // circuit (a 48 kA g_on·V phantom through a buck's MOSFET at that
    // row). A state seeded from `initial_state`, or from a snapshot
    // without that mask, was solved by no mask this row knows and is
    // flagged unsolved.
    SwitchStateMask initial_mask = mask_at(opts.t_start);
    bool initial_seeded = initial_state.has_value();
    if (resume_from != nullptr) {
        initial_seeded = !snapshot_stamped_mask(
            *resume_from, static_cast<Size>(graph.num_switches()),
            initial_mask);
    }
    // The first step lands an edge at t_start (see the clamp's gate-edge
    // landing) only on a snapshot that knows its x's mask and committed
    // no edge at its end.
    bool edge_at_resume = resume_from != nullptr && !initial_seeded
        && !resume_from->post_edge_derivatives;
    // The mask the current x was solved under (final_snapshot).
    SwitchStateMask x_mask = initial_mask;
    bool x_mask_known = !initial_seeded;
    // A resumed run continues its mode: no re-initialisation at the seam
    // (review M3). shortcut: a run cut within the 3 re-init steps of an
    // entry resumes with plain steps; carry reinit_left in the snapshot
    // if that matters.
    if (resume_from != nullptr && !initial_seeded && structure.active()) {
        reinit.note(initial_mask);
    }

    // ---- consistent initial diode bits (zero-time settle) ----
    // A tiny trap step pins every capacitor to its (v, i) and asks
    // the network which diodes conduct; iterate to a fixed point,
    // exactly run_transient's per-step event iteration at t_start.
    // The gates are read where the first step reads them (sw_now in
    // the edge clamp, t + h_min / 2): an edge closer than that is not
    // landed, so settling under mask_at(t_start) left a resume 1 ulp
    // before a buck's turn-on edge with D1 ON across M1 ON (24 kA).
    // `x_sig` is the state the diode bits were last DECIDED on: the
    // committed x after an ordinary step, but the last probe of a
    // zero-time cascade after one (a solve under the bits the cascade
    // settled on). The crossing detector reads each diode's signal at
    // the step START from it. Reading it from the committed x instead
    // — a solve under the PRE-cascade bits — is how a symmetric
    // V_th = 0 bridge ran half-wave: D1 and D3 crossed together, the
    // landing's x showed D1 at +ε, the cascade decided D1 stays OFF
    // (its probe said −ε once D3 conducted), and every later step then
    // started from the stale +ε, saw no sign change while D1's forward
    // bias grew to 6.9 V, and never turned it on again.
    Vector x_sig = x;
    if (has_diodes) {
        const Real h_probe = h_probe_floor;
        const Real t_gates = opts.t_start + Real{0.5} * h_min;
        // Outside the run's try: the budget abort is typed here too.
        try {
            const std::vector<bool> bits0 = diodes.snapshot_on_bits();
            SwitchStateMask m_site = initial_mask;
            bool settled = events::settle(opts.max_event_iterations, [&](Size) {
                // A failed (Newton) probe leaves x_trial unwritten —
                // unsized on this first call: keep the bits (it read
                // null, SIGSEGV).
                const SwitchStateMask m0 = mask_at(t_gates);
                m_site = m0;
                if (!probe_solve(m0, opts.t_start, h_probe, x_trial)) {
                    return events::Round::Stop;
                }
                const bool flipped =
                    decide_probe(x_trial, m0, opts.t_start);
                x_sig = x_trial;
                return flipped ? events::Round::Flipped : events::Round::Settled;
            }).settled;
            if (mode_site(opts.t_start, t_gates, bits0, settled,
                          opts.h_init > Real{0} ? opts.h_init : h_max / Real{100},
                          m_site)) {
                x_sig = x_trial;
            }
        } catch (const pwl::NewtonRoundoffBudgetExceeded& e) {
            throw SimulationAborted(e.what(), SimulationResult{},
                                    opts.t_start,
                                    /*roundoff_budget=*/true);
        } catch (const std::runtime_error& e) {
            // A singular probe at every step up to the cap (above), typed
            // like every in-run failure instead of escaping untyped.
            if (!is_singular_error(e)) throw;
            throw SimulationAborted(e.what(), SimulationResult{},
                                    opts.t_start);
        }
    }

    SimulationResult result;
    result.times.reserve(4096);
    result.begin_switch_state_record(
        static_cast<Size>(graph.num_switches()), 4096);
    // the commanded gate per sample (2.2); cleared before the result leaves
    if (diodes.num_gated() > 0) {
        result.gate_mask_fn = [&diodes] { return diodes.gate_mask(); };
    }
    result.times.push_back(opts.t_start);
    result.states.push_back(x);
    result.event_iteration_count.push_back(0);
    result.record_switch_state(initial_mask, !initial_seeded);
    if (hooks) hooks->start(x, initial_mask, {&history, &hs, &last_probe_dt});
    if (lt) {
        if (resume_from != nullptr && !resume_from->loss_state.empty()) {
            lt->restore(resume_from->loss_state);   // the heat in flight too
        } else {
            lt->begin(x, initial_mask);
        }
    }

    // Coupling-aware inductor di/dt for the LTE (a coupled winding's
    // v = L·di/dt + M·di_other/dt, so f ≠ v/L there). Resolved once.
    const auto& entries = history.entries();
    const auto& couplings = history.transformer_couplings();
    std::vector<Size> partner(entries.size(), Size(-1));
    std::vector<Real> partner_M(entries.size(), Real{0});
    for (const auto& tc : couplings) {
        if (tc.p_entry_idx < entries.size()
            && tc.s_entry_idx < entries.size()) {
            const Real M =
                models::TwoWindingTransformer::mutual_inductance(
                    tc.params);
            partner[tc.p_entry_idx] = tc.s_entry_idx;
            partner_M[tc.p_entry_idx] = M;
            partner[tc.s_entry_idx] = tc.p_entry_idx;
            partner_M[tc.s_entry_idx] = M;
        }
    }
    auto inductor_didt = [&](Size idx, Real v_own, Real v_partner,
                              Real /*i_own*/) -> Real {
        const Real L = entries[idx].C_or_L;
        if (partner[idx] == Size(-1)) {
            return v_own / L;
        }
        const Real Lo = entries[partner[idx]].C_or_L;
        const Real M  = partner_M[idx];
        const Real det = L * Lo - M * M;
        if (std::abs(det) < Real{1e-12} * L * Lo) {
            return v_own / L;   // k→1: estimate-only fallback
        }
        return (Lo * v_own - M * v_partner) / det;
    };
    auto entry_v = [&](const pwl::HistoryEntry& e, const Vector& xs) {
        return stamping::read_node_voltage(xs, e.from)
               - stamping::read_node_voltage(xs, e.to);
    };

    Real t = opts.t_start;
    t_now = &t;
    // The step memory is one object (refactor-unified-transient-stepper
    // 5.1): what a snapshot carries for an exact variable-step resume.
    // The names below are references into it.
    step::LteController lte_ctrl;
    Real& h = lte_ctrl.h;
    h = opts.h_init > Real{0} ? opts.h_init : h_max / Real{100};
    h = std::min(h, h_max);
    // A TR-BDF2 snapshot carries its step memory: the resumed run
    // proposes the step the stopped one would have (an explicit h_init
    // still wins). Any other controller's state is ignored.
    if (resume_from != nullptr && opts.h_init <= Real{0}
        && resume_from->controller.kind == step::ControllerState::Kind::Lte) {
        lte_ctrl.load(resume_from->controller);
    }
    Size steps = 0;
    // Zero-time chatter guard: a boundary-riding diode may flip at
    // the SAME time point repeatedly (bits ping-pong, no time
    // advance). After a few flips at one t, the step proceeds with
    // the bits as they are — run_transient's "accept the last
    // consistent solve" applied to the variable-step world.
    Real chatter_t = opts.t_start - span;
    Size chatter_n = 0;
    Real chatter_short_t = opts.t_start - span;
    Real chatter_x1_t = opts.t_start - span;
    Real chatter_exhaust_t = opts.t_start - span;
    const Real h_chatter = std::max(h_min, span_ref * Real{1e-7});
    // Post-GATE-edge restart memory, one slot per edge parity (a
    // periodic PWM alternates rising/falling, and the two corners
    // want different h). Restarting at the h that survived the
    // same corner LAST cycle converted a measured 4-reject-per-
    // period shrink storm into ~24 rejects per 1000 periods on the
    // buck. Diode events deliberately have NO memory: lending a
    // smooth corner's large h to a conduction-boundary crossing
    // quintupled the flyback's event count and biased its average
    // (measured) — the controller's own h is the right restart
    // there.
    Real (&h_mem_gate)[2] = lte_ctrl.h_mem_gate;
    Real end_cut_h = Real{0};   // the last proposal t_end cut short (0 = none)
    int& pending_gate_slot = lte_ctrl.pending_gate_slot;
    // Gate-edge cache: an LTE-reject inside an edge-clamped step
    // re-enters with the SAME upcoming edge; re-bisecting it costs
    // ~30 switch_fn calls (Python round-trips in the common case).
    // Controller-tick schedule. Absolute k·T from t_start (never
    // accumulated) so the cadence cannot drift — the whole point
    // of scheduling it instead of throttling on elapsed time.
    // One entry per source; the single observer_period is source 0.
    const bool multi_rate = !opts.observer_periods.empty();
    const std::vector<Real> t_ctrl =
        multi_rate ? opts.observer_periods
        : (opts.observer_period > Real{0}
               ? std::vector<Real>{opts.observer_period}
               : std::vector<Real>{});
    const bool has_ticks =
        (multi_rate ? static_cast<bool>(tick_fn)
                    : static_cast<bool>(observer_fn))
        && !t_ctrl.empty();
    for (const Real T : t_ctrl) {
        if (!(T > Real{0}) || !std::isfinite(T)) {
            throw std::invalid_argument(
                "run_transient_trbdf2: every controller period must be "
                "a positive finite number of seconds");
        }
    }
    // The run's event queue (add-unified-event-queue): tick source j has
    // id j (registered first), then the gate-edge source (the analytic
    // oracle, or the edges the sampling below locates), then the end.
    if (!opts.observer_offsets.empty()
        && opts.observer_offsets.size() != opts.observer_periods.size()) {
        throw std::invalid_argument(
            "run_transient_trbdf2: observer_offsets must be empty or as "
            "long as observer_periods");
    }
    for (const Real off : opts.observer_offsets) {
        if (!(off >= Real{0}) || !std::isfinite(off)) {
            throw std::invalid_argument(
                "run_transient_trbdf2: every controller offset must be a "
                "finite number of seconds >= 0");
        }
    }
    if (!opts.observer_origins.empty()
        && opts.observer_origins.size() != t_ctrl.size()) {
        throw std::invalid_argument(
            "run_transient_trbdf2: observer_origins must be empty or one "
            "per controller source");
    }
    events::EventQueue queue;
    if (has_ticks) {
        for (std::size_t j = 0; j < t_ctrl.size(); ++j) {
            const Real off = multi_rate && !opts.observer_offsets.empty()
                ? opts.observer_offsets[j] : Real{0};
            const bool mon = multi_rate && j < opts.observer_monitors.size()
                && opts.observer_monitors[j] != 0;
            queue.add_periodic(mon ? events::Kind::Monitor : events::Kind::ControlTick,
                               events::Rate{t_ctrl[j], off,
                                            opts.observer_origins.empty()
                                                ? opts.observer_origin
                                                : opts.observer_origins[j],
                                            opts.observer_next_k.size() == t_ctrl.size()
                                                    && !opts.observer_origins.empty()
                                                ? opts.observer_next_k[j]
                                                : std::numeric_limits<std::uint64_t>::max()},
                               /*lands=*/!mon);
        }
    }
    // First source corner strictly after `after` (+inf when none).
    auto breakpoint_oracle = [&](Real after) {
        Real best = std::numeric_limits<Real>::infinity();
        for (const auto& c : corners) {
            Real k0 = Real{0};
            if (c.period > Real{0} && after > c.t0) {
                k0 = std::floor((after - c.t0) / c.period);
            }
            for (int kk = 0; kk < 2; ++kk) {
                const Real base = c.period > Real{0}
                    ? c.t0 + (k0 + static_cast<Real>(kk)) * c.period
                    : c.t0;
                for (int j = 0; j < c.n; ++j) {
                    const Real bp = base + c.off[static_cast<std::size_t>(j)];
                    if (bp > after && bp < best) {
                        best = bp;
                    }
                }
                if (c.period <= Real{0}) {
                    break;
                }
            }
        }
        // The waveform current sources' corners (pulse edges, table
        // knots): landed, read at their left limit and restarted like
        // the voltage sources' (add-controlled-source-set).
        if (has_iwave) {
            best = std::min(best,
                sources::next_current_waveform_breakpoint(pool, graph, after));
        }
        return best;
    };
    const std::uint32_t edge_src = next_edge_fn
        ? queue.add_function(events::Kind::GateEdge, next_edge_fn)
        : queue.add_located(events::Kind::GateEdge);
    // Source breakpoints (task 7.1): the pulse / PWM sources' corners and
    // the waveform current sources' knots.
    const std::uint32_t bp_src =
        queue.add_function(events::Kind::SourceBreakpoint, breakpoint_oracle);
    queue.add_once(events::Kind::EndOfRun, opts.t_end);
    // h_min: a tick the previous run left out at its t_end (fire_due_ticks
    // stops at t_end - h_min) is a resumed run's first
    queue.start(opts.t_start, opts.t_end, h_min);
    TrBdf2Stats local_stats;
    TrBdf2Stats& st = stats_out ? *stats_out : local_stats;
    const pwl::CacheMetrics cache_m0 = cache.metrics();
    auto record_factor_traffic = [&] {
        const pwl::CacheMetrics m1 = cache.metrics();
        st.n_factor_builds = static_cast<Size>(m1.event_builds - cache_m0.event_builds);
        st.n_factor_refactors =
            static_cast<Size>(m1.event_refactors - cache_m0.event_refactors);
        st.n_factor_hits = static_cast<Size>(m1.event_hits - cache_m0.event_hits);
    };
    st.ctrl_tick_counts.assign(t_ctrl.size(), 0);
    st.h_max_basis = opts.h_max > Real{0} ? Size{2}
        : (h_default < span / Real{1000} ? Size{1} : Size{0});
    // Earliest source corner strictly after t (+inf when none).
    // Earliest source breakpoint strictly after t + h_min/2, through the
    // queue's breakpoint source (asked again only once it is passed).
    auto next_breakpoint_after = [&](Real at_t) {
        const Real bp = queue.next_after(bp_src, at_t + Real{0.5} * h_min);
        // "none" is not cached: the old per-step query asked again.
        if (!std::isfinite(bp)) queue.invalidate(bp_src);
        return bp;
    };

    // ---- opaque switch_fn: sample densely, learn the pulse width ----
    // Without an edge oracle the engine sees the gates only where it
    // calls switch_fn. It used to call it at t and t + h only, so a
    // step longer than one PWM period whose ends fell on the same
    // phase saw no edge at all, and the bisection that followed any
    // edge it did see assumed ONE edge in [t, t + h] and could land on
    // a later one. A 100 kHz boost run for 20 ms (default ceiling
    // span/1000 = 20 µs) took 112 gate events instead of 3999 and
    // ended at 590 V against 48.04 V, with no warning about the gates.
    // Now a step samples switch_fn at h/4, h/2, 3h/4 and h (plus one
    // walking sample, below) and bisects
    // only the FIRST sub-interval whose end differs, and the step is
    // capped at kCapPulses narrowest pulses with ceil(2 h / w) samples, so
    // the samples stay half a pulse apart once a pulse is known.
    const bool learn_gates = !next_edge_fn;
    const Size n_sw_total = static_cast<Size>(graph.num_switches());
    // The opaque gate's learned state, its step caps and its sampling
    // (add-unified-event-queue 3.1): events/sampled_gate.hpp.
    events::SampledGate gate(n_sw_total, diode_owned,
                             switch_fn(opts.t_start + Real{0.5} * h_min),
                             h_min, h_max, span_ref, learn_gates,
                             st.n_gate_undersampled);
    // The gate bits the diode bits were last decided (or last solved)
    // under. A gate edge that falls within an ulp of a step's END is
    // not seen by that step's samples (t + h rounds to the pre-edge
    // side) and so is not landed, and the zero-time cascade a landed
    // edge gets never ran: the next step then solved the post-edge
    // gates with the pre-edge diode bits. Measured where a controller
    // tick (which ends a step exactly on k·T) met a 10 kHz buck's
    // turn-on: a 2.1e-11 s sample with Q1 ON and D1 ON, v_sw = +6 V
    // across two 1 mOhm stamps. The step start compares and settles.
    SwitchStateMask gates_decided = gate.last_gate();
    // `b` is a stamped mask: a gated device's bit there is its conduction
    // state, so its gate is compared with the gate it was decided under.
    const SwitchStateMask gated = diodes.gated_bits();
    auto gates_differ = [&](const SwitchStateMask& a,
                            const SwitchStateMask& b) {
        for (const Size i : gate.gate_bits()) {
            if (gated.get(i) ? a.get(i) != diodes.gate_of(i)
                             : a.get(i) != b.get(i)) {
                return true;
            }
        }
        return false;
    };
    // Diode bits found held in a forbidden state at a step's start
    // (see the crossing detector).
    Real held_redecide_t = opts.t_start - span;

    // Time of a post-diode-event sample (see record_post_event below) that
    // is not a committed state: a sample the next commit would not lie
    // after is dropped, so the record stays increasing.
    Real event_sample_t = -std::numeric_limits<Real>::infinity();
    auto drop_superseded_event_sample = [&](Real t_new) {
        if (event_sample_t >= t_new && !result.times.empty()
            && result.times.back() == event_sample_t) {
            result.times.pop_back();
            result.states.pop_back();
            result.event_iteration_count.pop_back();
            if (result.has_switch_states) {
                result.switch_state_bits.resize(
                    result.switch_state_bits.size()
                    - result.switch_state_stride);
                if (!result.gate_state_bits.empty()) {
                    result.gate_state_bits.resize(
                        result.gate_state_bits.size()
                        - result.switch_state_stride);
                }
            }
        }
        event_sample_t = -std::numeric_limits<Real>::infinity();
    };
    // Every recorded time: a sample below the previous one is the engine
    // breaking its own invariant, raised rather than returned as a result.
    auto push_time = [&](Real ts, const char* kind) {
        if (ts < result.times.back()) {
            throw InternalError(std::format(
                "run_transient_trbdf2: a {} sample at t = {:.17g} lies before "
                "the previous sample t = {:.17g} (time went backward; a "
                "pulsim bug)", kind, ts, result.times.back()));
        }
        result.times.push_back(ts);
    };

    // ---- post-gate-edge restart ----
    // A gate edge leaves the companion history holding the PRE-edge
    // derivatives (inductor v, capacitor i, each stateful device's
    // slot). The next step's trapezoidal stage and error estimate read
    // them, so its error is O(h), not O(h^3): the controller rightly
    // refused it, collapsing the buck's post-edge step to 9e-10 s (and
    // the accepted-step memory then pinned 8e-13 s for a whole run).
    // The edge cascade's last probe is a real trapezoidal solve over
    // h_p under the post-edge mask from the committed state: COMMIT it
    // (as a diode landing commits its sub-step). Its history then
    // carries the post-edge derivatives the method itself produced,
    // and the record gets a post-edge sample h_p after the edge, so
    // anything integrated over the trace (losses, charge) sees the
    // edge as a jump instead of a ramp across the first post-edge step
    // (measured: interpolating across a long first step moved a buck's
    // conduction loss by 5 % and a switched-branch charge 10x).
    //
    // GATE-driven events only. Not at a diode event: a turn-off at the
    // zero crossing of an inductor's current leaves that inductor on a
    // g_off-only path, the probe then holds the stiff L·g_off mode
    // (ps time constant) and the trapezoidal stage does not damp it. A
    // 325 V bridge with a 1 mH line went from 22 to 5814 diode events
    // and a Level-1 MOSFET half bridge's error from 4e-4 to 4e-2. Nor at
    // the initial settle: it moved the same bridge's later events
    // enough to grind it to max_steps. See
    // update-trbdf2-post-event-restart/DECISIONS.md.
    //
    // `settled`: x_trial is already the probe (the last probe of a
    // cascade that settled, solved under gates_decided); otherwise one
    // probe is taken here. With `commit` false, or no room before
    // t_end, only the derivative slots are replaced (states kept).
    bool post_event_pending = false;
    // Where the probe whose derivatives replaced the slots without a
    // commit reached (snapshot: x is then the pre-edge solve, the slots
    // post-edge; a sliver step to t_end may still follow).
    Real slots_only_t = std::numeric_limits<Real>::quiet_NaN();
    Size sdirk_left = 0;       // SDIRK restart steps still owed (set below)
    const bool sdirk_ok = opts.sdirk_on_linear_circuits || nl_effective;
    bool sdirk_gate = false;   // the pending restart came from a gate edge
    // Where and under which mode a gate edge re-solved the slots (x itself
    // is the pre-edge solve): esdirk43's explicit stage may read them.
    Real edge_slots_t = std::numeric_limits<Real>::quiet_NaN();
    SwitchStateMask edge_slots_m;
    auto restart_after_gate_event = [&](bool settled, Real t_gates,
                                        Real hp, bool commit) {
        if (!opts.refresh_after_event) {
            return;
        }
        SwitchStateMask m_probe = gates_decided;
        if (!settled) {
            m_probe = mask_at(t_gates);
            try {
                probe_solve(m_probe, t, hp, x_trial);
            } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
                throw;
            } catch (const std::runtime_error&) {
                return;   // no usable probe: keep the old slots
            }
            ++st.n_refresh_probes;
        }
        if (!last_probe_ok || !x_trial.allFinite()) {
            return;
        }
        // Entering a mode with dependent states: the probe is a trapezoidal
        // step of femtoseconds across an impulse (1e13 V on a 10 A cut-set),
        // no state to commit nor derivatives to keep; the next step is the
        // SDIRK re-initialisation from the committed state.
        if (structure.active()
            && !reinit.entering(structure, m_probe, false).empty()) {
            return;
        }
        // probes R2 (slice 3c): the edge's two sides after the settle.
        if (hooks != nullptr && hooks->has_event() && !(m_probe == x_mask)) {
            hooks->event(t, x, x_mask, x_trial, m_probe);
        }
        if (lt && !(m_probe == x_mask)) lt->event(t, x, x_mask, x_trial, m_probe);
        const Real dtp = last_probe_dt;
        ++st.n_derivative_refreshes;
        post_event_pending = true;
        // The step after an edge starts from the consistent x0+: the states
        // are continuous, the algebraic unknowns and the derivative slots
        // are those of the NEW mode at the edge. The probe used to be
        // COMMITTED as a trapezoidal step of h_p = 1e-6 x the pre-edge step
        // whose first slot is the PRE-edge derivative: across a channel /
        // C_oss loop of tau = 0.6 ps it stopped at 38.94 V where the ODE is
        // at 30.67 V, and the loss port booked it from the right limit's
        // 2.3 MW (+4.4 % conduction energy, release group D3).
        //  - A circuit whose diode events restart by the SDIRK(2,2) step (a
        //    Newton law or a stateful kind): the same restart from x; its
        //    backward-Euler stages read no slot, its commit stores the new
        //    mode's.
        //  - A linear circuit: the slots from the probe, which KCL balances
        //    under the new mode at t + h_p (the probe's companion current, not
        //    the stale slot), and the next step is TR-BDF2 from x. (The SDIRK
        //    step there cost +40 .. +90 % accepted steps on PWM circuits for no
        //    measured gain: its embedded estimate is first order.)
        const bool lin = !sdirk_ok;
        if (lin) {
            history.refresh_derivatives(x_trial, dtp);
            if (lt) lt->slots_rewritten();
            edge_slots_t = t;
            edge_slots_m = m_probe;
        } else {
            sdirk_left = std::max(sdirk_left, Size{1});
            sdirk_gate = true;
        }
        const Real t_plus = t + dtp;
        // No sample across t_end or a controller tick (the step to it
        // starts from x anyway): the slots only, for a snapshot.
        if (!commit || t_plus >= opts.t_end - h_min
            || t_plus >= queue.next_of(events::Kind::ControlTick) - h_min) {
            if (!lin) {
                history.refresh_derivatives(x_trial, dtp);
                if (lt) lt->slots_rewritten();
                hs.refresh_derivatives(x_trial, dtp);
            }
            slots_only_t = t_plus;   // where the unused probe reached
            return;
        }
        // The toggle is at the edge, not at the next step's start.
        if (learn_gates) {
            // m_probe is a stamped mask: a gated device's bit there is its
            // conduction state, not its gate.
            gate.learn_toggles(diodes.num_gated() > 0 ? switch_fn(t_gates)
                                                      : m_probe, t);
        }
        // The right limit is recorded, not committed (as a diode event's):
        // the trace shows the edge as a jump, not a ramp across the first
        // step (losses and charges integrated over the samples, DECISIONS 7).
        x_sig = x_trial;
        gates_decided = m_probe;
        drop_superseded_event_sample(t);
        push_time(t_plus, "post-edge");
        result.states.push_back(x_trial);
        result.event_iteration_count.push_back(0);
        result.record_switch_state(m_probe);
        event_sample_t = t_plus;
    };

    // ---- L-stable SDIRK(2,2) restart (Alexander, gamma_s = 1 - 1/sqrt2) ----
    // Stage 1: backward Euler over gamma_s*h from x_n; stage 2: backward
    // Euler over gamma_s*h from x_n + ((1-gs)/gs)(X_1 - x_n). C/(gs h)
    // equals 2C/(gamma h), so both stages are the trapezoidal companion
    // at dt = gamma*h with the DERIVATIVE slots zeroed and the STATE slots
    // set to the stage's base: the same factor, no new history method.
    // The stateful kinds' q rows take the same BE stages (ed_sdirk_base /
    // ed_hs_commit, shared with esdirk43): every circuit can restart.
    const Real gs = Real{1} - Real{1} / std::sqrt(Real{2});
    // ---- the SDIRK restart after a diode turn-off on a linear circuit ----
    // A diode with g_off > 0 that turns off leaves the inductor it carried
    // on a g_off path: an L*g_off mode (tau ~ 2e-13 s on the DCM buck) the
    // raw TR-BDF2 estimate reads as error at any h, so each turn-off was
    // followed by a reject cascade down to ~1e-12 s and four or five steps
    // back up (1,806 steps below 1e-11 s in 400 periods). One SDIRK(2,2)
    // step damps it (L-stable, stiffly accurate); its embedded estimate
    // h(1-gs)(k1-k2) still reads the damped jump (k1 = jump/(gs h), constant
    // in h), so on that step the estimate is filtered, W^-1 est with the
    // stage factor W (Hosea-Shampine). Measured on the DCM buck: 7,608 ->
    // 4,212 accepted steps, 4,306 -> 2,753 rejects, no step below 1e-11 s
    // away from the event records (add-branch-form-ideal-switch DECISIONS 24).
    const bool sdirk_off_lin = opts.sdirk_after_turn_off
                               && !nl_effective && !sdirk_ok;
    bool sdirk_event = false;   // the pending restart came from such an event
    auto restart_ok = [&]() { return sdirk_ok || sdirk_event || sdirk_gate; };
    // A diode with g_off > 0 turned off (an ideal branch-form diode leaves
    // no L*g_off mode; its entries re-initialise by D4).
    auto turned_off = [&](const std::vector<bool>& before) {
        const std::vector<bool> now = diodes.snapshot_on_bits();
        const auto ents = diodes.entries();
        for (std::size_t i = 0; i < before.size() && i < now.size(); ++i) {
            if (before[i] && !now[i] && ents[i].g_off > Real{0}) return true;
        }
        return false;
    };
    // D4 (add-branch-form-ideal-switch): a step that enters a mode with
    // dependent states (an inductor cut-set opened, a capacitor loop closed
    // by ideal elements) through a gated change, or at the start, and the
    // two after it are SDIRK(2,2) steps on every circuit: backward-Euler
    // stages conserve flux / charge across the jump, where the trapezoidal
    // stage rings on the algebraic unknown for ever (+-397 V, 7.0e10 V).
    // The three-step count is the fixed engine's restart rule.
    constexpr Size kReinitSteps = 3;
    Size reinit_left = 0;
    Vector x_pre;   // the committed state before a step (records only)
    const Size n_sdirk_diode = opts.sdirk_after_diode_event;
    // ---- the SDIRK restart size, remembered per diode transition ----
    // The restart step was taken at the PRE-event h, which the step
    // controller sized for a smooth interval: on the MOSFET half bridge
    // 215 of 240 rejects were that step (err 2.4-3.5 at 4-5e-7 s, then
    // accepted at about half). Like the post-gate-edge memory, the
    // controller's proposal after the last accepted restart step for
    // the same diode turning the same way starts the next one.
    std::vector<Real>& h_mem_diode = lte_ctrl.h_mem_diode;
    Size& diode_slot = lte_ctrl.diode_slot;
    auto restart_size_after_diode_event = [&](Size d) {
        if (!restart_ok() || n_sdirk_diode == 0) {
            diode_slot = Size(-1);   // no restart: remember no size for it
            return;
        }
        const Size nd = diodes.entries().size();
        if (h_mem_diode.size() != 2 * nd) {
            h_mem_diode.assign(2 * nd, Real{0});
        }
        diode_slot = 2 * d + (diodes.entries()[d].is_on ? 1 : 0);
        if (h_mem_diode[diode_slot] > Real{0}) {
            h = std::min(h, h_mem_diode[diode_slot]);
        }
    };
    // ---- the post-event sample (ll08_buck_mosfet, 2026-11-13) ----
    // An SDIRK restart step is L-stable and its estimate sees the
    // differential variables only, so it may be long (500 ns on ll08,
    // where TR-BDF2 read stale derivatives and was rejected down to
    // ~10 ns). The algebraic variables JUMP at the event (V(sw) from
    // +24 V to -0.1 V when the freewheeling diode turns on), and with no
    // sample between the event and the end of that step the record
    // ramped across it: ll08's strict error 0.00231 -> 0.0252, its edge
    // instants 0.48 us late. The cascade's last probe is a real solve
    // under the post-event bits from the committed state, last_probe_dt
    // long: record it (not committed — the step still starts from x), so
    // the trace shows the jump where it happens.
    //
    // After a turn-off on a linear circuit the probe (femtoseconds) has not
    // moved the L*g_off mode (tau ~ 2e-13 s) the jump goes through: it read
    // v_D = -0.52 V where the circuit is at -34.7 V a picosecond later, and
    // the record then ramped across the 0.87 us restart step (a DCM boost's
    // recovery energy 9.9e-4 -> 2.4e-4 J). There the sample is a backward
    // Euler solve over dt_rec = max(h_min, 1e-4 h), the localisation's
    // conditioning floor: L-stable, it lands on the post-jump state.
    // The restart is kept only when that jump is instantaneous on the
    // trace: every state (inductor current, capacitor voltage) of the BE
    // solves over dt_rec and 2 dt_rec agrees within the step tolerance.
    // A slower mode is a transient the trace must show (a half-wave RL
    // rectifier: 31 mA decaying with tau = 1e-9 s through g_off = 1e-6 S;
    // skipped, its V(out) cell went 1.58e-5 -> 1.67e-4): TR-BDF2 steps.
    std::vector<pwl::HistoryEntry> hist_rec, hist_rec0;
    Vector b_rec, b_rec_src, x_rec, x_rec2;
    auto be_solve = [&](const SwitchStateMask& m, Real dt, Vector& out) {
        history.snapshot_into(hist_rec0);
        hist_rec = hist_rec0;
        for (auto& e : hist_rec) {   // backward Euler: no derivative slot
            if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                e.i_prev = Real{0};
            } else {
                e.v_prev = Real{0};
            }
        }
        history.restore(hist_rec);
        history.compute_b_extra(Real{2} * dt, b_rec);
        history.restore(hist_rec0);
        accumulate_sources(t + dt, b_rec_src);
        b_rec += b_rec_src;
        out = x;
        try {
            return solve_dispatch(m, Real{2} * dt, b_rec, out)
                && out.allFinite();
        } catch (const std::runtime_error&) {
            return false;   // no record rather than a wrong one
        }
    };
    auto record_post_event = [&]() {
        // A record still pending is an earlier event's at this same t (every
        // commit drops or settles it): this event supersedes it. Kept, its
        // probe step could exceed this one's and the record went backward
        // (PSFB under 1e-15 fuzz: a post-event sample 8e-16 s before it).
        drop_superseded_event_sample(t);
        if (!restart_ok() || n_sdirk_diode == 0 || !last_probe_ok
            || !x_sig.allFinite()) {
            sdirk_event = false;   // the jump was not shown instant
            return;
        }
        Real tp = t + last_probe_dt;
        const Vector* xr = &x_sig;
        // The mode is decided once (mask_at settles it: an ideal diode
        // closing a loop is released); the BE solves and the row read it.
        std::optional<SwitchStateMask> m_rec;
        if (sdirk_event) {
            // Pure overlay (no apply_gates: the event already applied them).
            m_rec = has_diodes ? switch_fn(t).overlay(diodes.current_diode_mask(),
                                                      diodes.diode_owned_bits())
                               : switch_fn(t);
            settle_mode(*m_rec);
            const Real dt_rec = std::max({h_min, Real{1e-4} * h,
                                          h_probe_learned});
            bool instant = be_solve(*m_rec, dt_rec, x_rec)
                           && be_solve(*m_rec, Real{2} * dt_rec, x_rec2);
            for (Size i = 0; instant && i < entries.size(); ++i) {
                const auto& e = entries[i];
                const bool cap = e.kind == pwl::DevicePool::StoredKind::Capacitor;
                const Real ya = cap ? entry_v(e, x_rec)
                                    : x_rec[e.inductor_branch_var_id];
                const Real yb = cap ? entry_v(e, x_rec2)
                                    : x_rec2[e.inductor_branch_var_id];
                instant = std::abs(ya - yb) <= opts.atol
                    + opts.rtol * std::max(std::abs(ya), std::abs(yb));
            }
            if (!instant) {
                sdirk_event = false;   // TR-BDF2 steps, no record
                return;
            }
            tp = t + dt_rec;
            xr = &x_rec;
        }
        if (tp >= opts.t_end - h_min
            || tp >= queue.next_of(events::Kind::ControlTick) - h_min) {
            sdirk_event = false;   // no post-jump sample: no turn-off restart
            return;
        }
        push_time(tp, "post-event");
        result.states.push_back(*xr);
        result.event_iteration_count.push_back(0);
        // The bits the probe was solved under (a landing has already
        // reset gates_decided to its pre-commutation mask).
        // Pure overlay: a record must not re-apply the gates (combine
        // sets g and opens a gate-held device). After an instant jump the
        // record is the mode the BE solves were decided under (m_rec).
        result.record_switch_state(
            m_rec ? *m_rec
            : has_diodes ? switch_fn(t).overlay(diodes.current_diode_mask(),
                                                diodes.diode_owned_bits())
                         : switch_fn(t));
        event_sample_t = tp;
    };
    sdirk_left = has_diodes ? opts.sdirk_after_settle : Size{0};
    // A run start with a stateful kind is an SDIRK restart step: the kinds'
    // derivative slots are init's zeros, not the current they carry (a
    // current-driven nonlinear capacitor: esdirk43's explicit first stage
    // read q' = 0, the q-row estimate was O(h) at any h and Newton stalled at
    // h_min, F-S-1). Its backward-Euler stages read no slot; the commit
    // stores the consistent one.
    // A CONSISTENT start needs no restart (a sine from zero into a saturable
    // L: v_L(0) = 0 is the slot; the first-order restart step cost it 5 % of
    // its error): a probe at t_start under the start mode gives each q row's
    // derivative there, and when no row's slot is off by more than its
    // tolerance over the first step, that step is the integrator's own.
    auto start_consistent = [&]() {
        const Size nq = hs.stage_rows();
        if (nq == 0) return false;
        std::vector<Real> q0(nq), f0(nq), f1(nq), sc(nq);
        hs.stage_start(Real{1}, q0.data(), f0.data());
        x_trial = x;
        try {
            if (!probe_solve(mask_at(opts.t_start + Real{0.5} * h_min), t,
                             h_probe_floor, x_trial)
                || !x_trial.allFinite()) {
                return false;
            }
        } catch (const pwl::NewtonRoundoffBudgetExceeded&) {
            throw;
        } catch (const std::runtime_error&) {
            return false;
        }
        pwl::StatefulHistories::Snapshot snap;
        hs.snapshot_into(snap);
        hs.refresh_derivatives(x_trial, last_probe_dt);
        hs.stage_start(Real{1}, q0.data(), f1.data());
        hs.restore(snap);
        hs.q_scales(sc.data());
        for (Size r = 0; r < nq; ++r) {
            if (sc[r] < Real{0}) continue;   // no q entry
            if (!(std::abs(f1[r] - f0[r]) * h
                  <= opts.atol * sc[r] + opts.rtol * std::abs(q0[r]))) {
                return false;
            }
        }
        return true;
    };
    if (hs.any() && resume_from == nullptr && !start_consistent()) {
        sdirk_left = std::max(sdirk_left, Size{1});
    }
    std::vector<pwl::HistoryEntry> hist_snap, hist_s1, hist_s2, hist_f;
    std::vector<Real> est_v, est_sc;
    sparse::Matrix noise_J;
    Vector noise_bc, noise_r, noise_rhs, noise_y, noise_y0;
    Vector b_filt, x_filt, x_filt0, b_zero;
    // est (state units: capacitor V, inductor A) -> W^-1 est: the
    // estimate as a state increment of a source-free BE step over
    // gamma_s*h = gamma*h/2, i.e. the companion at gamma*h with the state
    // slots = est and the derivative slots 0; solved on the step's
    // factor. solve_at adds b_constant, so the source-free part is
    // removed by linearity (one more solve on the same factor).
    auto filter_estimate = [&](const SwitchStateMask& m, Real dt,
                               std::vector<Real>& est) {
        history.snapshot_into(hist_snap);
        hist_f = hist_snap;
        for (Size i = 0; i < hist_f.size(); ++i) {
            auto& e = hist_f[i];
            if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                e.v_prev = est[i];
                e.i_prev = Real{0};
            } else {
                e.i_prev = est[i];
                e.v_prev = Real{0};
            }
        }
        history.restore(hist_f);
        history.compute_b_extra(dt, b_filt);
        history.restore(hist_snap);
        b_zero.setZero(static_cast<Index>(state_size));
        cache.solve_at(m, dt, b_filt, x_filt);
        cache.solve_at(m, dt, b_zero, x_filt0);
        x_filt -= x_filt0;
        for (Size i = 0; i < hist_f.size(); ++i) {
            const auto& e = hist_f[i];
            est[i] = e.kind == pwl::DevicePool::StoredKind::Capacitor
                ? entry_v(e, x_filt) : x_filt[e.inductor_branch_var_id];
        }
    };
    auto zero_derivatives = [](std::vector<pwl::HistoryEntry>& es) {
        for (auto& e : es) {
            if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                e.i_prev = Real{0};
            } else {
                e.v_prev = Real{0};
            }
        }
    };

    // ---- controller tick: every due source, in source order ----
    auto fire_due_ticks = [&]() {
        // None at t_end: no step follows (the run's tick count is ceil).
        if (t < queue.next_of(events::Kind::ControlTick) - h_min
            || t >= opts.t_end - h_min) {
            return;
        }
        // Every due source, in source order (the queue's order for
        // events of one kind).
        for (const auto& ev : queue.take_due(
                 t, h_min, events::kind_bit(events::Kind::ControlTick))) {
            const Size j = ev.source;
            if (multi_rate) {
                tick_fn(j, t, x);
            } else {
                observer_fn(t, x);
            }
            gate.note_tick(t);
            ++st.n_ctrl_ticks;
            ++st.ctrl_tick_counts[j];
        }
        // A controller just moved its duty, so switch_fn has a DIFFERENT
        // shape from here on and any bisected edge time is stale.
        // Closed-loop runs missed a quarter of their PWM edges (307 of 400
        // over 200 periods) before this.
        queue.invalidate(edge_src);
    };

    // ---- monitors: read only (Kind::Monitor) ----
    auto fire_due_monitors = [&]() {
        // a step may span several instants: take them all (one per
        // source per take_due call)
        // Not at t_end (as fire_due_ticks): an instant within h_min of
        // the end belongs to a resumed run, which takes it back from its
        // start tolerance; firing it here too called the monitor twice.
        const Real t_take = std::min(t, opts.t_end - Real{2} * h_min);
        while (t_take >= queue.next_of(events::Kind::Monitor) - h_min) {
            for (const auto& ev : queue.take_due(
                     t_take, h_min, events::kind_bit(events::Kind::Monitor))) {
                tick_fn(ev.source, t, x);
                ++st.ctrl_tick_counts[ev.source];
            }
        }
    };

    try {
        // The skeleton driver is the only driver (DECISIONS 19).
        // The attempt's locals that cross phases, (re)set where the loop body
        // declared them (class types by an in-place reset or a move: DECISIONS
        // 14 (1)).
        SwitchStateMask sw_now;
        SwitchStateMask mode_start;
        SwitchStateMask mask_step;
        std::vector<const pwl::DependentStructure*> reinit_new;
        Real h_lo = Real{0};
        Real t_stop = Real{0};
        Real h_ctrl = Real{0};
        Real bp_t = Real{0};
        Real h_pre_bp = Real{0};
        Real hg = Real{0};
        Real h_pt = Real{0};
        Real err = Real{0};
        bool landed_on_bp = false;
        bool landed_on_edge = false;
        bool gates_flat = false;
        bool reinit_step = false;
        bool sdirk = false;
        bool commutated = false;
        // ---- integrator='esdirk43' (add-high-order-stiff-integrator) ----
        // Runs on the skeleton only. The linear companions (C, L, coupled
        // windings) and every stateful kind take the five stages through
        // the stage form (βh, ψ) of StatefulHistories (DECISIONS 16, 17)
        // but the hysteretic core, refused by name.
        const bool esdirk = opts.integrator == 1;
        if (opts.integrator != 0 && !esdirk) {
            throw std::invalid_argument(
                "run_transient_trbdf2: integrator must be 0 (TR-BDF2) or 1 (ESDIRK4(3))");
        }
        if (esdirk) {
            for (const auto& e : hs.get<pwl::SaturableInductorHistory>().entries()) {
                if (!e.params.ja) continue;
                const std::string name(graph.branch_name(e.branch_id));
                throw UnsupportedError(
                    "trbdf2", "integrator=esdirk43", name, "SaturableInductor",
                    std::format("engine 'trbdf2': integrator='esdirk43' does not run the "
                                "Jiles-Atherton core '{}': its in-loop law is first order "
                                "in the step on both integrators (measured; a fourth-order "
                                "integrator buys nothing there). Drop integrator='esdirk43'.",
                                name));
            }
        }
        // the stateful kinds take the stages (their q rows below): every
        // ESDIRK stage, and an SDIRK(2,2) restart step on both integrators
        const bool ed_sf = hs.any();
        // The stages in entry space (capacitor v, inductor i): with a linear
        // q the stage combinations in x are those in q (design D2).
        std::vector<Real> ed_y0, ed_hd0, ed_psi;
        std::array<std::vector<Real>, step::Esdirk43::n_stages> ed_d;
        std::array<Vector, 4> ed_y;   // stages 1–4 (stage 5 is the output)
        std::vector<pwl::HistoryEntry> ed_hist;
        bool ed_land = false;   // the pending landing was an ESDIRK step
        bool ed_forced = false;  // this SDIRK step was forced by kE (slots of another mode)
        // The capacitor slots of a localisation bracket's two ends and of the
        // last trial (ed_slots), for a landing interpolated between them.
        std::vector<Real> ed_ia, ed_ib, ed_it;
        Real ed_thp = Real{0};
        // The stateful kinds' q rows (StatefulHistories' stage form): step
        // start q0 and h·q̇0, the stage's base ψ and ψ − q0, the increments
        // q(Y_s) − ψ_s; the last stage's base (the commit's). (No bracket
        // slots: a stateful kind makes the run nonlinear, so no interpolated
        // landing.)
        std::vector<Real> ed_sq0, ed_shd0, ed_spsi, ed_sdpsi, ed_spsi5, ed_sdpsi5;
        std::array<std::vector<Real>, step::Esdirk43::n_stages> ed_sd;
        pwl::StatefulHistories::Snapshot ed_hs_snap;
        // The q rows in the error norm, both integrators (tb_error): TR-BDF2's
        // q_n, q̇_n and the increments at γ and at the step end; per row
        // atol_q / atol (set on the first estimate of the run).
        const bool q_rows = hs.any();
        std::vector<Real> q_n, q_fn, q_dg, q_d1, q_atol;
        // TR-BDF2's rows join from the first committed step: before it the
        // kinds' derivative slots are init's zeros (a resume carries real
        // ones), an O(h) error no step can shrink below Newton's reach (a
        // nonlinear capacitor from x = 0 behind a 60 V source aborted at
        // h_min). ESDIRK keeps its first-step estimate (DECISIONS 17): a
        // machine started from rest needs it.
        bool q_live = resume_from != nullptr || esdirk;
        // The five stages of one ESDIRK4(3) step on the factor of h/2
        // (G + C/(γh)): stage s is backward Euler of γh from the pseudo
        // previous state ψ_s (the entries' state slots, derivative slots 0),
        // ψ_s = y0 + a_{s+1,0}·h·ẏ0 + Σ_{j<s} (a_{s+1,j+1}/γ)·(Y_j − ψ_j).
        // ẏ0 is the committed derivative slot: the companion current / the
        // inductor's terminal voltage of the last solve, KCL-consistent at x
        // (an evaluation, no solve). x_g = stage 1 (c = 1/2), x_1 = stage 5.
        // One ESDIRK step of hh from (t, x, the committed history) under m:
        // the five stages, the last into `out` (stiffly accurate). Leaves the
        // history as committed; ed_hist keeps the last stage's ψ (the commit's
        // base) and ed_d the increments (the estimate). A singular factor
        // throws (the caller decides); false = a stage's Newton failed.
        auto ed_stages = [&](const SwitchStateMask& m, Real hh, Vector& out,
                             bool left_limit_last) -> bool {
            using E = step::Esdirk43;
            const Real fdt = Real{2} * E::gamma * hh;   // the factor of hh/2
            history.snapshot_into(hist_snap);
            const Size ne = hist_snap.size();
            ed_y0.resize(ne);
            ed_hd0.resize(ne);
            ed_psi.resize(ne);
            for (auto& d : ed_d) d.resize(ne);
            for (Size i = 0; i < ne; ++i) {
                const auto& e = hist_snap[i];
                if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                    ed_y0[i] = e.v_prev;
                    ed_hd0[i] = hh * e.i_prev / e.C_or_L;
                } else {
                    const Real vp = partner[i] == Size(-1) ? Real{0}
                                                           : entries[partner[i]].v_prev;
                    ed_y0[i] = e.i_prev;
                    ed_hd0[i] = hh * inductor_didt(i, e.v_prev, vp, e.i_prev);
                }
            }
            ed_hist = hist_snap;
            Size nq = 0;
            if (ed_sf) {   // the stateful q rows: start, and a rollback point
                hs.snapshot_into(ed_hs_snap);
                nq = hs.stage_rows();
                ed_sq0.resize(nq);
                ed_shd0.resize(nq);
                ed_spsi.resize(nq);
                ed_sdpsi.resize(nq);
                for (auto& d : ed_sd) d.resize(nq);
                hs.stage_start(hh, ed_sq0.data(), ed_shd0.data());
            }
            bool ok = true;
            try {
                for (int s = 0; s < E::n_stages && ok; ++s) {
                    const auto si = static_cast<std::size_t>(s);
                    for (Size r = 0; r < nq; ++r) {   // ψ − q0 from increments
                        Real dp = E::explicit_weight(s) * ed_shd0[r];
                        for (int j = 0; j < s; ++j) {
                            dp += E::weight(s, j) * ed_sd[static_cast<std::size_t>(j)][r];
                        }
                        ed_sdpsi[r] = dp;
                        ed_spsi[r] = ed_sq0[r] + dp;
                    }
                    if (nq > 0) hs.stage_base(ed_spsi.data(), ed_sdpsi.data());
                    for (Size i = 0; i < ne; ++i) {
                        Real psi = ed_y0[i] + E::explicit_weight(s) * ed_hd0[i];
                        for (int j = 0; j < s; ++j) {
                            psi += E::weight(s, j) * ed_d[static_cast<std::size_t>(j)][i];
                        }
                        ed_psi[i] = psi;
                        auto& e = ed_hist[i];
                        if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                            e.v_prev = psi;
                            e.i_prev = Real{0};
                        } else {
                            e.i_prev = psi;
                            e.v_prev = Real{0};
                        }
                    }
                    history.restore(ed_hist);
                    history.compute_b_extra(fdt, b_stage);
                    const step::StageSpec spec = E::stage(hh, s);
                    // the last stage on a source jump reads its LEFT limit (as
                    // the TR-BDF2 stage 2 does)
                    accumulate_sources(
                        s == E::n_stages - 1 && left_limit_last
                            ? t + hh - std::min(Real{0.5} * hh,
                                  std::max(Real{1e-9} * hh,
                                           Real{512} * std::numeric_limits<Real>::epsilon()
                                               * (std::abs(t) + hh)))
                            : t + spec.t_off,
                        b_src);
                    b_stage += b_src;
                    Vector& ys = s == E::n_stages - 1 ? out : ed_y[si];
                    ys = s == 0 ? x : ed_y[si - 1];   // warm start: the previous stage
                    step_ctx = spec.ctx();
                    step_ctx.t = t + spec.t_off;   // ctx->t (external devices)
                    ok = solve_dispatch(m, fdt, b_stage, ys);
                    ++st.n_solves;
                    if (!ok) break;
                    if (nq > 0) hs.stage_increment(ys, ed_spsi.data(), ed_sd[si].data());
                    for (Size i = 0; i < ne; ++i) {
                        const auto& e = ed_hist[i];
                        ed_d[si][i] = (e.kind == pwl::DevicePool::StoredKind::Capacitor
                                           ? entry_v(e, ys)
                                           : ys[e.inductor_branch_var_id])
                                      - ed_psi[i];
                    }
                }
            } catch (...) {
                history.restore(hist_snap);
                if (ed_sf) { hs.restore(ed_hs_snap); hs.stage_clear(); }
                throw;
            }
            history.restore(hist_snap);
            if (ed_sf) {   // no stage leaves a trace; the commit re-applies ψ5
                std::swap(ed_spsi5, ed_spsi);   // (no copy: the next step resizes)
                std::swap(ed_sdpsi5, ed_sdpsi);
                hs.restore(ed_hs_snap);
                hs.stage_clear();
            }
            return ok;
        };
        // Commit the stateful kinds at the end of a step of hh: backward
        // Euler of h_e from the last stage's base (ed_spsi5), the stage's own
        // derivative. ESDIRK: h_e = γ·hh after ed_stages; the forced SDIRK
        // step: h_e = gs·hh after ed_sdirk_base.
        auto ed_hs_commit = [&](const Vector& x1, Real h_e, Real hh) {
            hs.stage_base(ed_spsi5.data(), ed_sdpsi5.data());
            hs.stage_commit(x1, h_e, hh);
            hs.stage_clear();
        };
        // The forced SDIRK(2,2) step's stage-2 base for the q rows: the
        // linear entries' pseudo previous state, ψ = q0 + r·(q(x_g) − q0)
        // (each kind's own extrapolate_previous moves λ_old but not the
        // increment form's i_L_old). Set on hs; the caller restores.
        auto ed_sdirk_base = [&](const Vector& xg, Real r) {
            const Size nq = hs.stage_rows();
            ed_sq0.resize(nq);
            ed_shd0.resize(nq);
            ed_spsi5.resize(nq);
            ed_sdpsi5.resize(nq);
            hs.stage_start(h, ed_sq0.data(), ed_shd0.data());
            hs.stage_increment(xg, ed_sq0.data(), ed_sdpsi5.data());
            for (Size i = 0; i < nq; ++i) {
                ed_sdpsi5[i] *= r;
                ed_spsi5[i] = ed_sq0[i] + ed_sdpsi5[i];
            }
            hs.stage_base(ed_spsi5.data(), ed_sdpsi5.data());
        };
        // The capacitor current a BE commit of stage s of the step hh that
        // ed_stages just took would store: the stage's own derivative
        // C·d_s/(γ·hh), KCL-consistent at that stage's state. Inductor slots
        // are readings of the state (0 here, unused).
        auto ed_slots = [&](int s, Real hh, std::vector<Real>& out) {
            out.resize(entries.size());
            for (Size i = 0; i < entries.size(); ++i) {
                out[i] = entries[i].kind == pwl::DevicePool::StoredKind::Capacitor
                    ? entries[i].C_or_L * ed_d[static_cast<std::size_t>(s)][i]
                          / (step::Esdirk43::gamma * hh)
                    : Real{0};
            }
        };
        // take, ESDIRK: one step of h; x_g = stage 1 (c = 1/2) for the
        // crossing detector, x_1 = stage 5.
        auto ed_take = [&]() -> step::Verdict {
            if (landed_on_bp && t + h < bp_t - h_min) landed_on_bp = false;
            // the heat owed, as a power over h; its right limit in the
            // capacitors' slots (the explicit stage reads them)
            if (lt) lt->advance(h, history, true);
            hg = Real{2} * step::Esdirk43::gamma * h;
            h_pt = Real{0.5} * h;
            bool ok = true;
            try {
                ok = ed_stages(mask_step, h, x_1, landed_on_bp);
            } catch (const std::runtime_error& e) {
                if (!is_singular_error(e) || nl_effective) throw;
                if (t + h >= t_stop && h <= Real{64} * h_min) {
                    return step::Verdict::End;   // a final sliver ends the run
                }
                const bool landing = h < h_ctrl * (Real{1} - Real{1e-12});
                if (landing || Real{4} * h > std::min(h_max, t_stop - t)) {
                    throw std::runtime_error(std::format(
                        "run_transient_trbdf2: at t = {} the step matrix is "
                        "numerically singular at h = {} s (integrator "
                        "'esdirk43'), and a longer step is not possible here: {}",
                        t, h, e.what()));
                }
                ++st.n_solves;
                ++st.n_reject;
                ++st.n_conditioning_rejects;
                h_cond = Real{4} * h;
                h_cond_by_mode[mode_start] = h_cond;
                st.h_conditioning_floor = std::max(st.h_conditioning_floor, h_cond);
                h = h_cond;
                return step::Verdict::Reject;
            }
            if (!ok) {
                ++st.n_reject;
                ++st.n_newton_retries;
                if (h <= h_min * Real{1.0001}) {
                    throw std::runtime_error(
                        "run_transient_trbdf2: Newton failed at t = "
                        + std::to_string(t) + " with h already at h_min (integrator "
                          "'esdirk43') — " + newton_last_error);
                }
                h = std::max(h * Real{0.25}, h_min);
                return step::Verdict::Reject;
            }
            x_g = ed_y[0];
            return step::Verdict::Accept;
        };
        // In-kernel losses of an ESDIRK step (or landing) just solved by
        // ed_stages: the conduction power on its quadrature, b1 on the
        // start, b2..b5 on Y1..Y4, b6 on x1 = Y5 (order 4, as the states).
        auto ed_loss_accept = [&](Real t0, Real t1, const Vector& x1,
                                  const SwitchStateMask& m) {
            using E = step::Esdirk43;
            lt->accept_step<4>(t0, t1, x1, m, {&ed_y[0], &ed_y[1], &ed_y[2], &ed_y[3]},
                               {E::b(1), E::b(2), E::b(3), E::b(4)}, E::b(0), E::b(5));
        };
        // The loop body, cut IN PLACE into the step's phases (slice 3b of
        // refactor-unified-transient-stepper, DECISIONS 15): each phase is ONE
        // lambda that the step skeleton calls (the only driver, DECISIONS 20).
        // One call of the phases is one ATTEMPT: a rejected attempt (the old
        // loop's `continue`) is not rolled back, the next attempt starts at the
        // cancellation poll and re-runs begin and clamp.
        auto tb_live = [&]() PULSIM_STEP_PHASE {
            return t < queue.next_of(events::Kind::EndOfRun) - h_min
                   && steps < opts.max_steps;
        };
        auto tb_poll = [&]() PULSIM_STEP_PHASE {
            return !(should_continue && !should_continue());
        };
        // begin: the attempt count, the grind refusal, ticks and monitors, the
        // gates at the step start, the controller's caps, the step-boundary settle
        auto tb_begin = [&]() PULSIM_STEP_PHASE {
            ++steps;
            // A grind is refused early, by name: 200,000 accepted steps that
            // advance less than 1e-4 of the span (the example flyback sat
            // at 5e-14 s on its clamp node's L*g_off mode for 10 M steps).
            if (st.n_accept >= grind_n0 + kGrindSteps) {
                if (t - grind_t0 < Real{1e-4} * span) {
                    throw std::runtime_error(std::format(
                        "run_transient_trbdf2: the controller is grinding: "
                        "{} accepted steps advanced only {:.3g} s, to t = {} "
                        "(mean step {:.3g} s) — an unresolved fast mode",
                        kGrindSteps, t - grind_t0, t,
                        (t - grind_t0) / static_cast<Real>(kGrindSteps)));
                }
                grind_n0 = st.n_accept;
                grind_t0 = t;
            }

            // ---- controller tick: fire AT the scheduled instant,
            //      before this step's mask is sampled ----
            fire_due_ticks();
            fire_due_monitors();

            // Shortly after the chatter guard gave up, the LTE may not
            // walk h below h_chatter: under all-OFF bridge bits a 1 mH
            // line and g_off make a 1 ps mode whose kink the LTE reads
            // as error at any h, and h sat at h_min until max_steps.
            // The gates as the step starts (one switch_fn call), and
            // the step cap the narrowest pulse seen so far implies.
            sw_now = switch_fn(t + Real{0.5} * h_min);
            mode_start = has_diodes
                ? sw_now.overlay(diodes.current_diode_mask(),
                                 diodes.diode_owned_bits())
                : sw_now;
            if (!h_cond_by_mode.empty()) {
                const auto it = h_cond_by_mode.find(mode_start);
                h_cond = it == h_cond_by_mode.end() ? Real{0} : it->second;
            }
            h_lo = std::max(
                (t - chatter_exhaust_t <= Real{1e3} * h_chatter)
                    ? h_chatter : h_min,
                h_cond);
            t_stop = queue.next_of(events::Kind::EndOfRun);
            // The proposal the end of the run cut short is what a resume
            // continues from (the snapshot's controller block).
            end_cut_h = t_stop - t < h ? h : Real{0};
            h = std::clamp(h, std::min(h_lo, t_stop - t),
                           std::min(h_max, t_stop - t));
            gate.learn_period(sw_now, t);
            if (has_ticks && !(opts.h_max > Real{0})) {
                h = std::min(h, std::max(gate.period_cap(), h_lo));
            }
            if (learn_gates) {
                gate.learn_toggles(sw_now, t);
                const Real cap = gate.pulse_cap(t);
                if (std::isfinite(cap)) {
                    h = std::min(h, std::max({cap, h_min, h_cond}));
                }
            }
            // The step the controller and the caps chose; a landing below
            // shortens it (see the conditioning floor at stage 1).
            h_ctrl = h;
            if (has_diodes && gates_differ(sw_now, gates_decided)) {
                // An edge at this step boundary that no step landed
                // (see gates_decided): settle the diodes under the
                // gates the step will use, as a landed edge does.
                const std::vector<bool> bits0 = diodes.snapshot_on_bits();
                bool settled = events::settle(opts.max_event_iterations, [&](Size) {
                    gates_decided = mask_at(t + Real{0.5} * h_min);
                    require_probe(probe_solve(gates_decided, t,
                                              h_probe_floor, x_trial),
                                  "a gate edge at a step boundary,", t);
                    const bool flipped =
                        decide_probe(x_trial, gates_decided, t);
                    x_sig = x_trial;
                    return flipped ? events::Round::Flipped : events::Round::Settled;
                }).settled;
                if (mode_site(t, t + Real{0.5} * h_min, bits0, settled, h,
                              gates_decided)) {
                    x_sig = x_trial;
                }
                restart_after_gate_event(settled,
                                         t + Real{0.5} * h_min,
                                         h_probe_floor,
                                         /*commit=*/false);
                ++st.n_gate_boundary_settles;
            }
        };
        // clamp: the landings (tick, source breakpoint, gate edge) and the
        // no-final-sliver rule; returns the step
        auto tb_clamp = [&]() PULSIM_STEP_PHASE {
            // Land the next tick exactly, like a gate edge.
            const Real t_next_tick =
                queue.next_of(events::Kind::ControlTick);
            if (t_next_tick > t && t_next_tick - t < h) {
                h = std::max(t_next_tick - t, h_min);
            }
            // ---- source-breakpoint landing ----
            landed_on_bp = false;
            bp_t = Real{0};
            h_pre_bp = h;
            if (!corners.empty() || has_iwave) {
                bp_t = next_breakpoint_after(t);
                if (bp_t <= t + h) {
                    h = std::max(bp_t - t, h_min);
                    landed_on_bp = true;
                }
            }

            // ---- gate-edge landing ----
            landed_on_edge = false;
            // A resumed run whose first step reads other gates than its x
            // was solved under, from a snapshot that committed no edge,
            // starts AT that edge: land it with an h_min step, as an edge
            // one ulp after t_start is (the zero-time settle alone left
            // the pre-edge derivatives in the history: 3.2e-4 V off the
            // uninterrupted run on a buck, an opaque gate resumed at its
            // turn-off, and the edge uncounted).
            // On every attempt of the first step (a rejected one retries).
            if (edge_at_resume) {
                if (t == opts.t_start) {
                    landed_on_edge = gates_differ(sw_now, initial_mask);
                    if (landed_on_edge) h = h_min;
                } else {
                    edge_at_resume = false;
                }
            }
            // The oracle is asked again only when its edge is passed or
            // a tick invalidated it; a located edge is dropped when passed.
            const Real next_edge_t = queue.next_after(edge_src, t);
            if (next_edge_t > t && next_edge_t <= t + h) {
                h = std::max(next_edge_t - t, h_min);
                landed_on_edge = true;
            }
            // True when every switch_fn sample over (t, t + h] equals
            // sw_now, so the step's gate mask IS sw_now (the midpoint
            // was one of the samples).
            gates_flat = false;
            if (!landed_on_edge) {
                // Sampling and bisection: SampledGate::locate.
                Real hi = t + h;
                const bool edge_found =
                    gate.locate(switch_fn, sw_now, t, h, hi);
                gates_flat = !edge_found;
                if (edge_found) {
                    h = std::max(hi - t, h_min);
                    landed_on_edge = true;
                    queue.locate(edge_src, hi);
                }
            }

            // ---- no final sliver ----
            // An edge within 64 h_min of the run's end is left to the next
            // run (its first step reads the gates at t_start + h_min/2 and
            // settles there): landing it left a 9.3e-15 s last step whose
            // slots were round-off / h, and steady_state found no orbit. The
            // step runs to t_end under the pre-edge gates. Not with a
            // stateful device history: the next run's step-boundary settle
            // does not commit its probe, and a buck's Coss resumed that way
            // read i(D1) 474x off the continuous run (the landed edge's
            // committed probe, flagged post_edge_derivatives, resumes right).
            // Nor past a source corner or a controller tick inside the gap.
            if (landed_on_edge && !hs.any() && t + h < t_stop
                && t_stop - (t + h) <= Real{64} * h_min
                && (!landed_on_bp || bp_t >= t_stop)
                && !(t_next_tick > t && t_next_tick < t_stop - h_min)) {
                h = t_stop - t;
                landed_on_edge = false;
                gates_flat = true;
            }
            return h;
        };
        // take: the step mask, the re-initialisation decision, the two stages
        // (kE: integrator='esdirk43'; one instantiation per integrator,
        // selected once per run in run_twin, no per-step branch)
        auto tb_take = [&]<bool kE = false>() PULSIM_STEP_PHASE {
            // names read only under kE, captured by reference outside it (MSVC)
            (void)ed_take; (void)ed_forced; (void)ed_sf; (void)ed_sdirk_base; (void)ed_hs_snap;
            (void)edge_slots_t; (void)edge_slots_m;

            // Step mask: sampled at the midpoint — piecewise-
            // constant gates are constant over (t, t+h) by the
            // clamp above, so the midpoint is unambiguous. When the
            // samples above were all equal it was one of them.
            mask_step =
                gates_flat ? sw_now : switch_fn(t + Real{0.5} * h);
            if (has_diodes) {
                mask_step = diodes.combine(mask_step);
            }
            settle_mode(mask_step);
            // D4: does this step enter a dependent mode? Then it is the
            // SDIRK re-initialisation; whether it was impulsive is decided
            // on the committed state (the record, gated entries only). A
            // diode-only entry is consistent (the ideal DCM buck's turn-off
            // at i = 0) but its trapezoidal stage still read the pre-event
            // derivative slot of the now-algebraic inductor current and was
            // rejected down to h_min: 5 steps below 1e-12 s per period.
            reinit_new.clear();
            if (structure.active()) {
                reinit_new = reinit.entering(structure, mask_step, false);
            }
            reinit_step = !reinit_new.empty();
            // ---- stage 1: TR over γh (or SDIRK stage 1, BE over γ_s h) ----
            const step::StageSpec stage0 = step::TrBdf2::stage(h, 0);
            hg = stage0.factor_dt;   // gamma * h
            sdirk = (sdirk_left > 0 && restart_ok()) || reinit_step
                               || reinit_left > 0;
            // Where x_g sits in time (the crossing detector reads it).
            h_pt = sdirk ? gs * h : hg;
            if constexpr (kE) {
                // The explicit first stage reads the derivative slots, and the
                // embedded estimate is nearly blind to them (b0 − b̂0 = 0.003
                // against b0 = 0.158): slots of another mode (a diode landing
                // commits under the pre-commutation mask) would pass as an
                // O(h) error. Such a step, and every restart, is the shared
                // SDIRK(2,2) step (backward-Euler stages, no slot read): its
                // commit leaves slots consistent with this mode (item (v)).
                ed_forced = !sdirk;
                // (or the slots a gate edge re-solved under this mode at t)
                if (!sdirk && (mask_step == x_mask
                               || (edge_slots_t == t && mask_step == edge_slots_m))) {
                    return ed_take();
                }
                if (!sdirk) {
                    sdirk = true;
                    h_pt = gs * h;
                }
            }
            if (sdirk) {
                history.snapshot_into(hist_snap);
                hist_s1 = hist_snap;
                zero_derivatives(hist_s1);
                history.restore(hist_s1);
            }
            // the heat owed, as a power over h; a TR-BDF2 attempt also puts its
            // right limit in the capacitors' slots (a restart's stages read none)
            if (lt) lt->advance(h, history, !sdirk);
            history.compute_b_extra(hg, b_stage);
            accumulate_sources(t + h_pt, b_src);
            b_stage += b_src;
            x_g = x;                 // Newton warm start
            step_ctx = stage0.ctx();
            step_ctx.t = t + h_pt;
            if (sdirk && ed_sf) {   // the q rows: BE of gs·h, the linear entries' rule
                hs.snapshot_into(ed_hs_snap);
                step_ctx = {gs * h, pwl::TrBdf2Stage::BackwardEuler, t + h_pt};
            }
            // A step below the matrix's conditioning floor: 2C/h against
            // a node held only by g_off (a capacitor between two OFF
            // diodes) leaves a pivot below eps, and the factorization
            // refuses it. Smaller steps cannot resolve that node, so the
            // floor is learned (h_cond, x4 per retry) and the controller
            // stops there (accepts at it are forced, counted), instead of
            // aborting where it chased an L*g_off mode at a tight atol.
            bool stage_singular = false;
            bool stage_ok = false;
            try {
                stage_ok = solve_dispatch(mask_step, hg, b_stage, x_g);
            } catch (const std::runtime_error& e) {
                if (!is_singular_error(e) || nl_effective) throw;
                // A final sliver (an edge landed a few h_min before t_end)
                // ends the run here, as the loop already does within h_min.
                if (t + h >= t_stop && h <= Real{64} * h_min) {
                    // the SDIRK stage swapped in derivative-free history:
                    // the run's final snapshot must hold the real one
                    if (sdirk) history.restore(hist_snap);
                    return step::Verdict::End;   // a final sliver ends the run
                }
                const bool landing = h < h_ctrl * (Real{1} - Real{1e-12});
                if (landing || Real{4} * h > std::min(h_max, t_stop - t)) {
                    throw std::runtime_error(std::format(
                        "run_transient_trbdf2: at t = {} the step matrix is "
                        "numerically singular at h = {} s{}, and a longer step "
                        "is not possible here: {} A node held only by OFF "
                        "conductances has 2C/h far above g_off/eps at short "
                        "steps; give it a DC path (a resistor) or raise g_off.",
                        t, h, landing ? " (a step landing on an event)" : "",
                        e.what()));
                }
                stage_singular = true;
            }
            ++st.n_solves;

            // A gate edge before the breakpoint shortened the step.
            if (landed_on_bp && t + h < bp_t - h_min) {
                landed_on_bp = false;
            }
            // ---- stage 2: BDF2 over the remainder, SAME factor ----
            if (sdirk) {
                // BE from x_n + ((1-gs)/gs)(X_1 - x_n), derivative slots 0.
                const Real r = (Real{1} - gs) / gs;
                hist_s2 = hist_s1;
                for (auto& e : hist_s2) {
                    if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                        e.v_prev += r * (entry_v(e, x_g) - e.v_prev);
                    } else {
                        e.i_prev += r * (x_g[e.inductor_branch_var_id]
                                         - e.i_prev);
                    }
                }
                history.restore(hist_s2);
                history.compute_b_extra(hg, b_stage);
                history.restore(hist_snap);
            } else {
                history.compute_b_extra_trbdf2_stage2(h, x_g, b_stage);
            }
            // A step landed on a source jump reads its LEFT limit; the
            // jump applies from the next step.
            accumulate_sources(
                landed_on_bp
                    ? t + h - std::min(Real{0.5} * h,
                          std::max(Real{1e-9} * h,
                                   Real{512} * std::numeric_limits<Real>::epsilon()
                                       * (std::abs(t) + h)))
                    : t + h,
                b_src);
            b_stage += b_src;
            if (stage_ok) {
                // Each device's state AT THE STAGE POINT feeds the
                // BDF2 history term. `h` is the FULL step there, while
                // the matrix is still the one factored at gamma*h — the
                // identity c1/h == 2/(gamma*h) is what allows that.
                hs.capture_gamma(x_g);
                step_ctx = step::TrBdf2::stage(h, 1).ctx();
                step_ctx.t = t + h;
                if (sdirk && ed_sf) {
                    step_ctx.t = t + h_pt;   // q(x_g) at x_g's own time
                    ed_sdirk_base(x_g, (Real{1} - gs) / gs);
                    step_ctx = {gs * h, pwl::TrBdf2Stage::BackwardEuler, t + h};
                }
                x_1 = x_g;           // Newton warm start
                stage_ok =
                    solve_dispatch(mask_step, hg, b_stage, x_1);
                ++st.n_solves;
            }
            // the q rows' stage-2 increment q(x_1) − ψ for the estimate, read
            // under the stage's own last stamp; then no stage leaves a trace
            // (the commit re-applies ψ, and the rollback also drops the γ
            // capture, so trbdf2_increments cannot serve a restart step)
            if (sdirk && ed_sf && stage_ok) {
                q_d1.resize(hs.stage_rows());
                hs.stage_increment(x_1, ed_spsi5.data(), q_d1.data());
            }
            if (sdirk && ed_sf) { hs.restore(ed_hs_snap); hs.stage_clear(); }
            if (stage_singular) {
                ++st.n_reject;
                ++st.n_conditioning_rejects;
                h_cond = Real{4} * h;
                h_cond_by_mode[mode_start] = h_cond;
                st.h_conditioning_floor = std::max(st.h_conditioning_floor, h_cond);
                h = h_cond;
                return step::Verdict::Reject;
            }
            if (!stage_ok) {
                // Newton did not converge at this h. Shrink and
                // retake — the variable-step answer to a hard
                // nonlinear step.
                ++st.n_reject;
                ++st.n_newton_retries;
                if (h <= h_min * Real{1.0001}) {
                    throw std::runtime_error(
                        "run_transient_trbdf2: Newton failed at t = "
                        + std::to_string(t)
                        + " with h already at h_min — "
                        + newton_last_error
                        + " Newton did not converge even at the "
                          "smallest step; cross-check this circuit on "
                          "engine='pwl' with an explicit dt.");
                }
                h = std::max(h * Real{0.25}, h_min);
                return step::Verdict::Reject;
            }
            return step::Verdict::Accept;
        };
        // error: the LTE and the controller's verdict (reject, forced accept)
        auto tb_error = [&]<bool kE = false>() PULSIM_STEP_PHASE {
            (void)ed_hd0; (void)ed_d; (void)ed_y0;   // captured outside kE (MSVC)
            (void)ed_sf; (void)ed_sq0; (void)ed_shd0; (void)ed_sd; (void)ed_spsi5;

            // ---- LTE over the differential variables ----
            Real err_sq = Real{0};
            Real q_sq = Real{0};   // the stateful q rows' share, kept for the floor
            Size n_q = 0;
            Real q_wmax = Real{-1};   // the largest q-row ratio (trace_forced)
            Size q_who = 0;
            Size n_err = 0;
            est_v.assign(entries.size(), Real{0});
            est_sc.assign(entries.size(), Real{1});
            for (Size i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                const Real vg1 = entry_v(e, x_g);
                const Real v11 = entry_v(e, x_1);
                Real f_n, f_g, f_1, y_old, y_new;
                if (sdirk) {
                    // Embedded first-order estimate h(1-gs)(k1 - k2)
                    // (design D3), exponent -1/2 below.
                    const auto& e1 = hist_s1[i];
                    const auto& e2 = hist_s2[i];
                    Real k1, k2;
                    if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                        k1 = (Real{2} / hg) * (vg1 - e1.v_prev);
                        k2 = (Real{2} / hg) * (v11 - e2.v_prev);
                        y_old = e.v_prev;
                        y_new = v11;
                    } else {
                        const Real vp_g = partner[i] == Size(-1) ? Real{0}
                            : entry_v(entries[partner[i]], x_g);
                        const Real vp_1 = partner[i] == Size(-1) ? Real{0}
                            : entry_v(entries[partner[i]], x_1);
                        k1 = inductor_didt(i, vg1, vp_g, Real{0});
                        k2 = inductor_didt(i, v11, vp_1, Real{0});
                        y_old = e.i_prev;
                        y_new = x_1[e.inductor_branch_var_id];
                    }
                    const Real sc = opts.atol + opts.rtol
                        * std::max(std::abs(y_old), std::abs(y_new));
                    est_v[i] = h * (Real{1} - gs) * (k1 - k2);
                    est_sc[i] = sc;
                    // The re-initialisation step: a dependent state jumps
                    // (flux / charge conserved by the BE stages at any h),
                    // so its k1 - k2 is the jump over gs*h, not an error.
                    for (const auto* dp : reinit_new) {
                        if (std::find(dp->states.begin(), dp->states.end(),
                                      e.branch_id) != dp->states.end()) {
                            est_v[i] = Real{0};
                        }
                    }
                    continue;
                }
                if constexpr (kE) {
                    // ESDIRK4(3): the embedded difference h·Σ(b − b̂)·ẏ_i
                    // (O(h^4)), filtered below on the stage factor.
                    using E = step::Esdirk43;
                    Real ee = E::err_explicit_weight() * ed_hd0[i];
                    for (int j = 0; j < E::n_stages; ++j) {
                        ee += E::err_weight(j) * ed_d[static_cast<std::size_t>(j)][i];
                    }
                    est_v[i] = ee;
                    est_sc[i] = opts.atol + opts.rtol
                        * std::max(std::abs(ed_y0[i]),
                                   std::abs(e.kind == pwl::DevicePool::StoredKind::Capacitor
                                                ? v11 : x_1[e.inductor_branch_var_id]));
                    continue;
                }
                if (e.kind
                    == pwl::DevicePool::StoredKind::Capacitor) {
                    const Real C = e.C_or_L;
                    const Real i_g =
                        (Real{2} * C / hg) * (vg1 - e.v_prev)
                        - e.i_prev;
                    const Real i_1 =
                        (C / h)
                        * (c1 * v11 + c2 * vg1 + c3 * e.v_prev);
                    f_n = e.i_prev / C;
                    f_g = i_g / C;
                    f_1 = i_1 / C;
                    y_old = e.v_prev;
                    y_new = v11;
                } else {
                    const Real vp_n =
                        partner[i] == Size(-1)
                            ? Real{0}
                            : entries[partner[i]].v_prev;
                    const Real vp_g =
                        partner[i] == Size(-1)
                            ? Real{0}
                            : entry_v(entries[partner[i]], x_g);
                    const Real vp_1 =
                        partner[i] == Size(-1)
                            ? Real{0}
                            : entry_v(entries[partner[i]], x_1);
                    f_n = inductor_didt(i, e.v_prev, vp_n, e.i_prev);
                    f_g = inductor_didt(i, vg1, vp_g, Real{0});
                    f_1 = inductor_didt(i, v11, vp_1, Real{0});
                    y_old = e.i_prev;
                    y_new = x_1[e.inductor_branch_var_id];
                }
                // The integrator's estimate (step::TrBdf2::lte, Bank et al.).
                const Real lte = step::TrBdf2::lte(h, f_n, f_g, f_1);
                const Real sc =
                    opts.atol
                    + opts.rtol
                          * std::max(std::abs(y_old),
                                     std::abs(y_new));
                est_v[i] = lte;
                est_sc[i] = sc;
            }
            // Hosea-Shampine filter: W^-1 est with W the stage matrix (the
            // (mask, gamma*h) factor both stages used). A stiff mode the
            // L-stable stages damp is not error; raw, the estimate reads a
            // 1e-13 s L*g_off mode as error at any h. Linear circuits
            // only (a Newton circuit's factor is not kept).
            // Also inside the window after the chatter guard gave up: there
            // the raw estimate reads the L*g_off mode as error at any h (see
            // h_lo), and every step sat at h_chatter as a forced accept.
            // ESDIRK: always (its embedded R̂(−∞) = −3/20 reads a damped
            // stiff mode as error).
            // and the SDIRK step it forces after a mode change: raw, its
            // first-order estimate reads the damped jump as error at any h
            // (the reject cascade of the turn-off restart, above)
            if (((kE && (!sdirk || ed_forced)) || opts.filter_error_estimate
                 || (sdirk && sdirk_event && !reinit_step)
                 || t - chatter_exhaust_t <= Real{1e3} * h_chatter)
                && !nl_effective
                && !entries.empty()) {
                filter_estimate(mask_step, hg, est_v);
                ++st.n_filtered_estimates;
            }
            for (Size i = 0; i < entries.size(); ++i) {
                const Real w = est_v[i] / est_sc[i];
                err_sq += w * w;
                ++n_err;
            }
            // The stateful kinds' q rows join the norm on both integrators
            // (update-trbdf2-error-control DECISIONS 10): a circuit whose
            // dynamics live only there (a Lauritzen recovery, an IGBT tail)
            // had no estimate and sat at its ceiling at any rtol. Per row the
            // integrator's own estimate of q, scale atol·s + rtol·max(|q_n|,
            // |q_1|) with s the device's capacitance / inductance / time
            // constant (`q_scales`), so a row that stays 0 keeps a finite
            // scale. On an SDIRK restart step the rows take the entries'
            // embedded first-order estimate h(1-gs)(k1 - k2) on q, k1 = (ψ5 −
            // q_n)/(r gs h), k2 = (q(x_1) − ψ5)/(gs h), r = (1-gs)/gs (without
            // it a restart left a saturable inductor's flux uncontrolled: the
            // boost example's i_L error 3.4x). That step reads no derivative
            // slot, so its estimate also holds at the run's first step.
            if (q_rows && (q_live || sdirk)) {
                const Size nq = hs.stage_rows();
                if (q_atol.size() != nq) {
                    q_atol.resize(nq);
                    hs.q_scales(q_atol.data());
                }
                if (!kE && !sdirk) {
                    q_n.resize(nq);
                    q_fn.resize(nq);
                    q_dg.resize(nq);
                    q_d1.resize(nq);
                    hs.stage_start(Real{1}, q_n.data(), q_fn.data());
                    hs.trbdf2_increments(x_1, q_dg.data(), q_d1.data());
                }
                for (Size r = 0; r < nq; ++r) {
                    if (q_atol[r] < Real{0}) continue;   // an external row with no q entry
                    Real ee, q0, q1;
                    if (sdirk) {
                        ee = ed_sdpsi5[r] - (Real{1} - gs) / gs * q_d1[r];
                        q0 = ed_sq0[r];
                        q1 = ed_spsi5[r] + q_d1[r];
                    } else if constexpr (kE) {
                        using E = step::Esdirk43;
                        ee = E::err_explicit_weight() * ed_shd0[r];
                        for (int j = 0; j < E::n_stages; ++j) {
                            ee += E::err_weight(j) * ed_sd[static_cast<std::size_t>(j)][r];
                        }
                        q0 = ed_sq0[r];
                        q1 = ed_spsi5[r] + ed_sd[E::n_stages - 1][r];
                    } else {
                        // the derivatives the stage rules imply: trapezoidal
                        // q_g = q_n + γh/2 (f_n + f_g), BDF2 f_1 = (c1 q_1 +
                        // c2 q_g + c3 q_n)/h with c1 + c2 + c3 = 0
                        const Real f_n = q_fn[r];
                        const Real f_g = Real{2} * q_dg[r] / hg - f_n;
                        const Real f_1 = (c1 * q_d1[r] + c2 * q_dg[r]) / h;
                        ee = step::TrBdf2::lte(h, f_n, f_g, f_1);
                        q0 = q_n[r];
                        q1 = q0 + q_d1[r];
                    }
                    const Real sc = opts.atol * q_atol[r]
                                    + opts.rtol * std::max(std::abs(q0), std::abs(q1))
                                    + std::numeric_limits<Real>::min();
                    const Real w = ee / sc;
                    q_sq += w * w;
                    ++n_q;
                    if (std::abs(w) > q_wmax) {
                        q_wmax = std::abs(w);
                        q_who = r;
                    }
                }
            }
            // The q rows as their own RMS, joined by max: they can tighten the
            // step, never loosen it (one RMS over both diluted the entries'
            // error: a saturable LC took larger steps and got 6 % worse).
            // Without q rows (n_q = 0) this is the entries' RMS, bit for bit.
            {
                const Real e_ent = n_err > 0 ? err_sq / static_cast<Real>(n_err) : Real{0};
                const Real e_q = n_q > 0 ? q_sq / static_cast<Real>(n_q) : Real{0};
                err = std::sqrt(e_q > e_ent || std::isnan(e_q) ? e_q : e_ent);
            }
            // Roundoff floor of the estimate. A node tied to the rest of
            // the circuit only through OFF conductances (a SEPIC with its
            // switch and diode both off: sw and mid held by 2 x 1e-6 S)
            // has a voltage error eps*(2C/h)*V/g_off from the elimination,
            // and the estimate reads it through v_L/L as h*dv/L: constant
            // in h (1e-10 A measured), so a scale of 1.5e-11 A (atol 1e-12,
            // rtol 1e-8 at 1.4 mA) rejected every h down to h_min, where
            // the noise turned an OFF diode 5 V forward and the run ground
            // to max_steps. Only for a step about to be rejected: the
            // floor (eps * J^-1 (|J||x| + |b|) through each entry's
            // estimate) joins the scale, and the step is accepted when the
            // estimate is within it. Steps that pass are unchanged.
            // ONE floor for both integrators (add-high-order-stiff-integrator
            // DECISIONS 17): ESDIRK steps always (their estimate had none);
            // TR-BDF2 as before unless opts.roundoff_floor = 1. With 1 the
            // noise right-hand side carries alternating signs: |J||x| + |b|
            // all positive through J^-1 cancels where the elimination's
            // round-off does not (PSFB defect (2), update-trbdf2-post-event-
            // restart DECISIONS 21).
            if (err > Real{1} && !nl_effective && !sdirk
                && !entries.empty()) {
                const Vector* nbc = nullptr;
                const sparse::Matrix* nJ =
                    cache.resident_event_matrix(mask_step, hg, &nbc);
                if (nJ == nullptr) {
                    seg_asm.assemble(graph, pool, mask_step, hg, noise_J,
                                     noise_bc);
                    nJ = &noise_J;
                    nbc = &noise_bc;
                }
                noise_r = nJ->cwiseAbs() * x_1.cwiseAbs();
                noise_r += nbc->cwiseAbs() + b_stage.cwiseAbs();
                noise_rhs = -noise_r;
                if (kE || opts.roundoff_floor == 1) {
                    for (Index k = 1; k < noise_rhs.size(); k += 2) noise_rhs[k] = -noise_rhs[k];
                }
                cache.solve_at(mask_step, hg, noise_rhs, noise_y);
                noise_rhs.setZero(static_cast<Index>(state_size));
                cache.solve_at(mask_step, hg, noise_rhs, noise_y0);
                noise_y = (noise_y - noise_y0).cwiseAbs()
                          * std::numeric_limits<Real>::epsilon();
                const Real kc = (Real{4} / hg) / (gamma * (Real{1} - gamma))
                    + (std::abs(c1) + std::abs(c2) + std::abs(c3))
                          / (h * (Real{1} - gamma));
                const Real kl = Real{1} / gamma
                    + Real{1} / (gamma * (Real{1} - gamma))
                    + Real{1} / (Real{1} - gamma);
                Real e2 = Real{0};
                // ESDIRK: est = Σ e_j·(Y_j − ψ_j) + e_0·h·ẏ0, each term a
                // state difference: its noise is Σ|e_j|·δy (+|e_0|/γ·δy
                // through the committed slot)
                using E = step::Esdirk43;
                const Real ce = [] {
                    Real c = std::abs(E::err_explicit_weight()) / E::gamma;
                    for (int j = 0; j < E::n_stages; ++j) c += std::abs(E::err_weight(j));
                    return c;
                }();
                for (Size i = 0; i < entries.size(); ++i) {
                    const auto& e = entries[i];
                    const Real dv =
                        stamping::read_node_voltage(noise_y, e.from)
                        + stamping::read_node_voltage(noise_y, e.to);
                    if constexpr (kE) {
                        const Real dy = e.kind == pwl::DevicePool::StoredKind::Capacitor
                            ? dv : noise_y[e.inductor_branch_var_id];
                        const Real w = est_v[i] / (est_sc[i] + ce * dy);
                        e2 += w * w;
                        continue;
                    }
                    Real df;
                    if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                        df = kc * dv;
                    } else if (partner[i] == Size(-1)) {
                        df = kl * dv / e.C_or_L;
                    } else {
                        const auto& ep = entries[partner[i]];
                        const Real dvp =
                            stamping::read_node_voltage(noise_y, ep.from)
                            + stamping::read_node_voltage(noise_y, ep.to);
                        const Real det = std::abs(e.C_or_L * ep.C_or_L
                            - partner_M[i] * partner_M[i]);
                        df = det > Real{1e-12} * e.C_or_L * ep.C_or_L
                            ? kl * (ep.C_or_L * dv + std::abs(partner_M[i]) * dvp) / det
                            : kl * dv / e.C_or_L;
                    }
                    const Real w = est_v[i]
                        / (est_sc[i] + std::abs(Real{2} * clte * h) * df);
                    e2 += w * w;
                }
                // the q rows keep their share without a floor: the floor
                // never accepts a step on a stateful kind's error (today
                // unreachable: a stateful kind sets nl_effective; kept so a
                // future relaxation of that guard cannot drop them)
                const Real e_ent = e2 / static_cast<Real>(entries.size());
                const Real e_q = n_q > 0 ? q_sq / static_cast<Real>(n_q) : Real{0};
                const Real err_n = std::sqrt(e_q > e_ent || std::isnan(e_q) ? e_q : e_ent);
                if (err_n <= Real{1}) {
                    // Not clamped to 1 (keep h): measured, the SEPIC at
                    // rtol 1e-8 then took 662,366 steps instead of
                    // 27,529 — h could never grow off the floor again.
                    err = err_n;
                    ++st.n_roundoff_floor_accepts;
                }
            }

            if (!std::isfinite(err)
                || !x_1.allFinite()) {
                h = std::max(h * Real{0.1}, h_min);
                ++st.n_reject;
                if (sdirk && sdirk_event && !reinit_step) {
                    drop_superseded_event_sample(t);
                    sdirk_event = false;   // as an LTE reject below
                    sdirk_left = 0;
                }
                if (h <= h_min * Real{1.0001}) {
                    throw std::runtime_error(
                        "run_transient_trbdf2: non-finite step at "
                        "t = " + std::to_string(t)
                        + " with h already at h_min — the circuit "
                          "is singular under this mask");
                }
                return step::Verdict::Reject;
            }

            if (err > Real{1} && h <= h_lo * Real{1.0001}) {
                // Cannot shrink further: accept, but CONFESS —
                // silence here would be an unbounded-error hole.
                ++st.n_forced_accepts;
                if (opts.trace_forced) {   // which unknown forced it
                    Real wmax = q_wmax;
                    Real who = -Real{1} - static_cast<Real>(q_who);
                    for (Size i = 0; i < entries.size(); ++i) {
                        const Real w = std::abs(est_v[i] / est_sc[i]);
                        if (w > wmax) {
                            wmax = w;
                            who = static_cast<Real>(entries[i].branch_id);
                        }
                    }
                    st.forced_trace.push_back({t, h, err, who, wmax});
                }
            }
            if (err > Real{1} && h > h_lo * Real{1.0001}) {
                // reject: elementary controller, order-3 exponent
                const Real fac = kE && !sdirk
                    ? step::LteController::reject_factor(err, step::Esdirk43::err_exponent)
                    : step::LteController::reject_factor(err, sdirk);
                h = std::max(h * fac, h_lo);
                ++st.n_reject;
                if (sdirk && sdirk_event && !reinit_step) {
                    // Its record claimed the jump instant: TR-BDF2 resolves
                    // it now, so the record goes.
                    drop_superseded_event_sample(t);
                    // The turn-off restart is one L-stable step over a mode
                    // its filtered estimate does not see; rejected, the mode
                    // is not stiff at this h (a flyback's coupled leakage
                    // at 1e-14 s took 100 forced accepts): TR-BDF2 steps.
                    sdirk_event = false;
                    sdirk_left = 0;
                }
                return step::Verdict::Reject;
            }
            return step::Verdict::Accept;
        };
        // after_solve: diode crossings, the localisation and its landing (a
        // commutated step is committed here, at the landing)
        auto tb_after = [&]<bool kE = false>() PULSIM_STEP_PHASE {
            (void)ed_stages; (void)ed_slots; (void)ed_loss_accept; (void)ed_hist;   // captured outside kE (MSVC)
            (void)ed_land; (void)ed_ia; (void)ed_ib; (void)ed_it; (void)ed_thp;

            // ---- diode crossing detection on the accepted step ----
            commutated = false;
            // shortcut: no crossing is located inside the re-initialisation
            // step (its landing would be a trapezoidal sub-step across the
            // impulse); a diode left on its forbidden side is re-decided at
            // the start of the next step (the held-diode path). Upgrade by
            // landing with an SDIRK sub-step.
            // So a re-initialisation step must not overrun a crossing: an
            // ideal rectifier diode turned on at 24.38 us took a 1 us reinit
            // step past its 25.01 us turn-off and ended at -7 A, then turned
            // off there (V_out 0.066 V low for the rest of the run, at every
            // rtol). A diode on its forbidden side at the step's end: retake
            // it shorter, until it ends before the crossing or h is h_lo.
            // Only a side change inside the step (allowed at x_sig), past the
            // detector's own bounds: shrinking cures nothing else.
            if (has_diodes && reinit_step && h > h_lo * Real{1.0001}) {
                bool overrun = false;
                if (!nl_effective) predicate_noise(x_1, mask_step, hg);
                for (Size d = 0; d < diodes.entries().size() && !overrun; ++d) {
                    const Real s1 =
                        detail_trbdf2::crossing_signal(diodes, d, x_1);
                    overrun = detail_trbdf2::forbidden(diodes, d, s1)
                        && !detail_trbdf2::forbidden(diodes, d,
                               detail_trbdf2::crossing_signal(diodes, d, x_sig))
                        && std::abs(s1)
                               > detail_trbdf2::held_tolerance(diodes, d, x_1)
                        && (nl_effective || std::abs(s1) > pn_bound[d]);
                }
                if (overrun) {
                    h = std::max(h * Real{0.1}, h_lo);
                    ++st.n_reject;
                    return step::Verdict::Reject;
                }
            }
            if (has_diodes && !reinit_step) {
                Real t_star = opts.t_end + span;   // earliest
                Size cross_idx = Size(-1);
                bool held_found = false;
                // The step's own round-off bound on each diode's signal
                // (as decide_diodes): with a bus floating on a 1 MΩ tie
                // the common mode carries ~1e-4 V of noise even at a
                // 1 µs step (2C/h = 940 S), and V_th = 0 diodes to ground
                // riding a zero crossing were decided by it (9 of 20
                // fuzzed runs ground to max_steps without this gate).
                // A crossing counts when the far side is outside the
                // bound, or the near side is (read on the committed x):
                // the diode then ends within noise of its threshold after
                // a clear swing. A gate on the far side alone held a V_th = 0.7 turn-on
                // (swing from reverse bias to 0.0485 V past threshold,
                // bound ~0.05 V: a 1 mΩ bridge input held by g_off) and
                // re-decided it one step (60 µs) late, unlocalised, into
                // a 493 A spike. Taken once per step, only when some
                // signal changes side.
                std::vector<Real> nb1;
                bool nb_ready = false;
                auto noise_at = [&](Size d) -> Real {
                    if (!nb_ready) {
                        if (nl_effective) {
                            nb1.assign(diodes.entries().size(), Real{0});
                        } else {
                            predicate_noise(x_1, mask_step, hg);
                            nb1 = pn_bound;
                        }
                        nb_ready = true;
                    }
                    return nb1[d];
                };
                // The pre-crossing side is clear when the committed state
                // at t (a full step, not a tiny probe whose common mode
                // may be round-off) agrees with s0 in sign and is outside
                // the bound.
                auto near_side_clear = [&](Size d, Real s0) {
                    const Real sx =
                        detail_trbdf2::crossing_signal(diodes, d, x);
                    return (sx > Real{0}) == (s0 > Real{0})
                           && std::abs(sx) > noise_at(d);
                };
                for (Size d = 0; d < diodes.entries().size(); ++d) {
                    // s0 from the state the bits were DECIDED on (see
                    // x_sig above), not from the committed x.
                    const Real s0 = detail_trbdf2::crossing_signal(
                        diodes, d, x_sig);
                    const Real sg = detail_trbdf2::crossing_signal(
                        diodes, d, x_g);
                    const Real s1 = detail_trbdf2::crossing_signal(
                        diodes, d, x_1);
                    // A crossing exists if the signal changes sign
                    // anywhere among (t, t+γh, t+h) and leaves the
                    // noise floor: an ON bridge diode
                    // carrying g_off-scale leakage (-1.6e-7 A) was
                    // re-landed on every step of a floating-rail
                    // bridge, 318k steps for 40 ms.
                    const Real tol_d = detail_trbdf2::held_tolerance(
                        diodes, d, x_1);
                    const bool crossed =
                        ((s0 > Real{0}) != (s1 > Real{0})
                         || (s0 > Real{0}) != (sg > Real{0}))
                        && std::max({std::abs(s0), std::abs(sg),
                                     std::abs(s1)}) > tol_d
                        && (std::max(
                                (s0 > Real{0}) != (s1 > Real{0})
                                    ? std::abs(s1) : Real{0},
                                (s0 > Real{0}) != (sg > Real{0})
                                    ? std::abs(sg) : Real{0})
                                > noise_at(d)
                            || near_side_clear(d, s0));
                    if (!crossed) {
                        // No sign change, but the diode may sit on the
                        // FORBIDDEN side from the start of the step to
                        // its end (a cascade that did not settle, or
                        // a chatter guard that gave up): a sign test
                        // can never see that, and it stayed so for the
                        // rest of a half-cycle — measured as a bridge
                        // diode held OFF while forward-biased by 45 V.
                        // Re-decide the bits here, at t, unless the
                        // violation is numerical noise (a V_th = 0
                        // diode riding its zero-current boundary).
                        if (detail_trbdf2::forbidden(diodes, d, s0)
                            && detail_trbdf2::forbidden(diodes, d, s1)
                            && std::max(std::abs(s0), std::abs(s1))
                                   > tol_d
                            && std::abs(s1) > noise_at(d)) {
                            held_found = true;
                            if (t < t_star) {
                                t_star = t;
                                cross_idx = d;
                            }
                        }
                        continue;
                    }
                    // Linear first estimate against the earlier of
                    // the two sub-intervals that crosses.
                    Real ta = t, sa = s0, tb = t + h, sb = s1;
                    if ((s0 > Real{0}) != (sg > Real{0})) {
                        tb = t + h_pt;
                        sb = sg;
                    }
                    const Real est =
                        (std::abs(sb - sa) > Real{0})
                            ? ta - sa * (tb - ta) / (sb - sa)
                            : tb;
                    if (est < t_star) {
                        t_star = est;
                        cross_idx = d;
                    }
                }
                if (cross_idx != Size(-1)) {
                    // Conditioning floor for localization solves:
                    // a trap at dt below this has g_eq = 2C/dt in
                    // the 1e20 range and poisons whatever it
                    // touches (the existing engine's
                    // substep_min_dt exists for the same reason).
                    const Real dt_floor =
                        std::max(h_min, h * Real{1e-4});
                    if (t_star - t < dt_floor) {
                        // Chatter-safe path: the crossing is
                        // within tolerance of the point we are
                        // already AT — flip the bits HERE with a
                        // zero-time cascade (probe solves only,
                        // nothing committed) instead of committing
                        // a femtosecond-scale trap step. Found at
                        // rtol=1e-6 on the flyback: the DCM-
                        // boundary ring produced 130 extra events
                        // whose fs-scale landings corrupted the
                        // charge bookkeeping by 60 mV.
                        if (t == chatter_t) {
                            ++chatter_n;
                        } else {
                            chatter_t = t;
                            chatter_n = 1;
                        }
                        if (held_found && t != held_redecide_t) {
                            held_redecide_t = t;
                            ++st.n_held_diode_redecided;
                        }
                        if (chatter_n <= Size{3}) {
                            // The cascade decides at t; the crossing is up
                            // to dt_floor later. When the probe at t still
                            // reads the pre-crossing side (a turn-off whose
                            // current is +2e-7 A at t and zero 2.4e-13 s
                            // later), decide on a probe dt_floor long, the
                            // post-crossing side, as a landing does. If that
                            // changes no bit either, nothing commutated here:
                            // no event is recorded (an OFF diode cannot "turn
                            // off"). The SEPIC re-took such a step three
                            // times, then guessed the bits from its end.
                            const std::vector<bool> bits0 =
                                diodes.snapshot_on_bits();
                            const Vector x_sig0 = x_sig;
                            bool settled = false;
                            for (const Real h_first : {h_probe_floor, dt_floor}) {
                                settled = events::settle(opts.max_event_iterations, [&](Size it2) {
                                    // Probe, not a committed step.
                                    gates_decided = mask_at(t);
                                    require_probe(
                                        probe_solve(gates_decided, t,
                                                    it2 == 0 ? h_first
                                                             : h_probe_floor,
                                                    x_trial),
                                        "a diode event,", t);
                                    const bool flipped = decide_probe(
                                        x_trial, gates_decided, t);
                                    x_sig = x_trial;
                                    return flipped ? events::Round::Flipped : events::Round::Settled;
                                }).settled;
                                if (diodes.snapshot_on_bits() != bits0) {
                                    break;
                                }
                            }
                            if (mode_site(t, t, bits0, settled, h,
                                          gates_decided)) {
                                x_sig = x_trial;
                            }
                            if (diodes.snapshot_on_bits() != bits0) {
                                result.commutation_events.push_back(
                                    {t,
                                     diodes.entries()[cross_idx]
                                         .branch_id,
                                     diodes.entries()[cross_idx]
                                         .is_on});
                                ++st.n_diode_events;
                                sdirk_event = sdirk_off_lin
                                    && turned_off(bits0);
                                sdirk_left = n_sdirk_diode;
                                record_post_event();
                                restart_size_after_diode_event(cross_idx);
                                commutated = true;
                            } else {
                                // Nothing flipped: no event, no restart.
                                // Retake the step at the same h; the
                                // chatter guard below bounds the retries.
                                // The bits did not change, so neither does
                                // the state they were decided on: x_sig as
                                // before these probes (a probe of a
                                // g_off-held node is round-off). That is x
                                // unless an earlier cascade at this t (a
                                // gate edge) decided the bits on its probe;
                                // x itself was solved under the bits before
                                // that cascade (see x_sig above).
                                x_sig = x_sig0;
                                // (Accepting it instead moved the SEPIC's
                                // V_out by 7.5e-5 V; localising it inside
                                // the step landed a V_th = 0 bridge diode
                                // riding its boundary every 1.6e-12 s.)
                                commutated = true;
                            }
                        }
                        else {
                            // Guard exhausted. Take a SHORT step with
                            // the bits as they are, not the h the
                            // controller had: the bits are a guess
                            // here, and the step they are guessed for
                            // is how long the guess integrates. A
                            // V_th = 0 bridge diode ping-ponging at a
                            // zero-current turn-on was carried OFF over
                            // a full 100 µs step and turned on 1.93 V
                            // forward-biased (18.9 A into 0.1 ohm,
                            // against a 7.9 A peak on the fixed
                            // engine). The retry at the same t finds
                            // the guard still exhausted and accepts
                            // (once per t: shrinking again on every
                            // retry would walk h down to h_min).
                            // Decide once from the END of the step,
                            // as the fixed-step engine does. The
                            // probes pin the line inductor's current
                            // at its g_off-scale -1e-7 A, so a bridge
                            // diode forward-biased by 47 V probed
                            // OFF again at every t (all-OFF bits, a
                            // 1 ps L*g_off mode, h stuck at h_min);
                            // the step's own end state, h of real
                            // time later, turns it ON for good.
                            chatter_exhaust_t = t;
                            if (chatter_x1_t != t) {
                                chatter_x1_t = t;
                                ++st.n_chatter_breaks;   // once per t
                                (void)noise_at(0);   // fills nb1
                                if (diodes.update_from_state(x_1, nb1)) {
                                    // The next step's signals start at t: a
                                    // probe there under the bits just taken,
                                    // not x_1, h later (reviewer on #18).
                                    const SwitchStateMask m_br = mask_at(t);
                                    x_trial = x;
                                    bool p_ok = false;
                                    try {
                                        p_ok = probe_solve(m_br, t, h_probe_floor,
                                                           x_trial);
                                    } catch (const std::runtime_error& e) {
                                        if (!is_singular_error(e)) throw;
                                    }
                                    // A probe reading a diode on the side its
                                    // new bit forbids (the -1e-7 A g_off-pinned
                                    // inductor above) would re-arm the guard: x_1.
                                    bool p_fits = p_ok && x_trial.allFinite();
                                    for (Size d = 0; p_fits && d < diodes.entries().size(); ++d) {
                                        const Real sp = detail_trbdf2::crossing_signal(
                                            diodes, d, x_trial);
                                        p_fits = !detail_trbdf2::forbidden(diodes, d, sp)
                                            || std::abs(sp) <= detail_trbdf2::held_tolerance(
                                                   diodes, d, x_trial);
                                    }
                                    x_sig = p_fits ? x_trial : x_1;
                                    return step::Verdict::Reject;
                                }
                            }
                            if (chatter_short_t != t) {
                                chatter_short_t = t;
                                h = std::max(h * Real{1e-2}, h_min);
                                ++st.n_reject;
                                return step::Verdict::Reject;
                            }
                        }
                        // Guard exhausted: fall through and ACCEPT
                        // the step with the bits as they are — the
                        // boundary rider stays put for this step
                        // and time moves forward.
                    } else {
                    // ---- Illinois localization on trial TRAP
                    //      solves from the COMMITTED state at t ----
                    Real ta = t;
                    Real sa = detail_trbdf2::crossing_signal(
                        diodes, cross_idx, x_sig);
                    Real tb = t + h;
                    Real sb = detail_trbdf2::crossing_signal(
                        diodes, cross_idx, x_1);
                    {
                        const Real sg =
                            detail_trbdf2::crossing_signal(
                                diodes, cross_idx, x_g);
                        if ((sa > Real{0}) != (sg > Real{0})) {
                            tb = t + h_pt;
                            sb = sg;
                        }
                    }
                    // A turn-off landed past its current zero leaves the
                    // overshoot current in the inductor with only g_off to
                    // go through: v = i/g_off. The time bracket alone
                    // (1e-4 h) let a DCM buck land 7.8 uA past the zero,
                    // and the run then recorded the L*g_off mode it
                    // excited (sw at 3854 V for 4e-13 s, a 24 V circuit).
                    // Bracket a turn-off until that voltage is within the
                    // run's voltage tolerance, or the bracket is h_min.
                    const auto& ce = diodes.entries()[cross_idx];
                    const bool turn_off = ce.is_on && ce.g_off > Real{0};
                    const Real v_tol =
                        opts.atol + opts.rtol * x.cwiseAbs().maxCoeff();
                    // Round-off floor: a signal within the solve's own noise
                    // cannot be bracketed further (it walked to h_min).
                    // The noise acceptance is capped by the spike it lets
                    // through: never above the circuit's own range max|x|.
                    const Real v_range = x.head(graph.num_nodes()).cwiseAbs().maxCoeff();
                    auto spike_ok = [&](Real s_post) {
                        const Real spike = std::abs(s_post) * ce.g_on / ce.g_off;
                        return !turn_off || spike <= v_tol
                            || (std::abs(s_post) <= noise_at(cross_idx)
                                && spike <= v_range);
                    };
                    Real sb_raw = sb;   // the unhalved signal at tb
                    Real sa_raw = sa;
                    // The STATE at ta = t is the committed x (the trials and
                    // the history start there), not x_sig, which may be a
                    // probe or a chatter-guard step end 40 us later: an
                    // interpolated landing from it put 5 mV on a 470 uF
                    // capacitor in 0.8 ps (-6.6 MA in its slot, the bus
                    // then -20 V in one step, a floating bridge +1.76 V).
                    Vector x_ta = x;
                    Vector x_tb = tb < t + h ? x_g : x_1;
                    if constexpr (kE) {
                        if (!sdirk) {   // the slots of x (committed) and of x_tb (its stage)
                            ed_ia.resize(entries.size());
                            for (Size i = 0; i < entries.size(); ++i) ed_ia[i] = entries[i].i_prev;
                            ed_slots(tb < t + h ? 0 : step::Esdirk43::n_stages - 1, h, ed_ib);
                        }
                    }
                    events::Bracket br{ta, sa, tb, sb, sa_raw, sb_raw, 0};
                    (void)events::illinois(br, 40,
                        [&](Real t_loc, const events::Bracket& b) {
                            return t_loc - t <= dt_floor || (b.tb - b.ta) <= h_min;
                        },
                        [&](Real t_loc, Real& sm) {
                        (void)ed_stages; (void)ed_slots; (void)ed_it; (void)sdirk;   // captured outside kE (MSVC)
                        const Real dts = t_loc - t;
                        bool trial_ok = true;
                        try {
                            // A failed Newton trial leaves x_trial at
                            // its warm start: never read it.
                            // ESDIRK: the trial is a re-step of the
                            // method itself (a trapezoidal trial over an
                            // ESDIRK-sized bracket set the event 100x off)
                            if constexpr (kE) {
                                trial_ok = sdirk ? trap_solve(mask_step, t, dts, x_trial)
                                                 : ed_stages(mask_step, dts, x_trial, false);
                            } else {
                            trial_ok = trap_solve(mask_step, t, dts,
                                                  x_trial);
                            }
                        } catch (const std::runtime_error& e) {
                            // A singular trial (see probe_solve) is
                            // ill-conditioned like a non-finite one.
                            if (!is_singular_error(e)) throw;
                            trial_ok = false;
                        }
                        if (!trial_ok || !x_trial.allFinite()) {
                            // Ill-conditioned trial (tiny-dt trap
                            // on a near-singular mask): abandon
                            // refinement, land on the bracket end.
                            return false;
                        }
                        sm = detail_trbdf2::crossing_signal(
                            diodes, cross_idx, x_trial);
                        if constexpr (kE) {
                            if (!sdirk) ed_slots(step::Esdirk43::n_stages - 1, dts, ed_it);
                        }
                        return true;
                        },
                        [&](events::BracketEnd e) {
                            (e == events::BracketEnd::A ? x_ta : x_tb) = x_trial;
                            (void)ed_ia; (void)ed_ib; (void)ed_it; (void)sdirk;   // captured outside kE (MSVC)
                            // the slots of the end that moved: its trial's stage 5
                            if constexpr (kE) {
                                if (!sdirk) (e == events::BracketEnd::A ? ed_ia : ed_ib) = ed_it;
                            }
                        },
                        [&](const events::Bracket& b) {
                            return (b.tb - b.ta) < std::max(h_min,
                                                             Real{1e-4} * h)
                                && spike_ok(b.sb_raw);
                        });
                    ta = br.ta;
                    tb = br.tb;
                    sb_raw = br.sb_raw;
                    // Land the step at t* = tb (the post-crossing
                    // side, so decide_next_state sees the crossed
                    // signal), commit the trap sub-step there.
                    Real dt_land =
                        std::max(tb - t, dt_floor);
                    // A turn-off still past spike_ok when the bracket hit
                    // its conditioning floor (the zero 0.6 ps after t on
                    // a DCM buck, bracket 8 ps: -9e-7 A left, sw at 446 V)
                    // lands on the secant root of the two bracketing
                    // states: a trap solve there would carry eps*2C/dt of
                    // noise, the size of the error. Linear circuits only.
                    // shortcut: the capacitor current is the bracket's
                    // (i_C = 2C/dt (v - v_n) - i_n of an interpolated v),
                    // exact to O(bracket * di_C/dt).
                    // Both signals are read on the two states the line runs
                    // between: sa_raw starts as x_sig's, which after a
                    // cascade or a chatter break is not x_ta (reviewer on #18).
                    sa_raw = detail_trbdf2::crossing_signal(
                        diodes, cross_idx, x_ta);
                    const bool interp_land = turn_off && !nl_effective
                        && !spike_ok(sb_raw) && sa_raw != sb_raw
                        && (sa_raw > Real{0}) != (sb_raw > Real{0})
                        // only a bracket at its floor: an abandoned or
                        // exhausted refinement still lands with a solve
                        && tb - ta <= Real{2} * dt_floor;
                    bool land_ok = true;
                    if constexpr (kE) ed_land = false;
                    if (interp_land) {
                        const Real th = sa_raw / (sa_raw - sb_raw);
                        const Real t_lin = ta + th * (tb - ta);
                        // Post-crossing side by a hair, as the solve lands.
                        // Not below the probes' learned conditioning floor
                        // (a singular solve there), never past the bracket.
                        dt_land = std::min(tb - t, std::max({t_lin - t,
                            h_probe_floor, h_probe_learned}));
                        // The state where the landing is recorded: a floor
                        // that moved the time past t_lin moves the state
                        // along the bracket with it (it was stored at t_lin).
                        const Real thp = std::min(Real{1}, std::max(
                            th * (Real{1} + Real{1e-6}),
                            (t + dt_land - ta) / (tb - ta)));
                        x_trial = x_ta + thp * (x_tb - x_ta);
                        if constexpr (kE) ed_thp = thp;
                        if (dt_land > t_lin - t) ++st.n_interp_floor_landings;
                    } else if constexpr (kE) {
                        ed_land = !sdirk;
                        land_ok = ed_land ? ed_stages(mask_step, dt_land, x_trial, false)
                                          : trap_solve(mask_step, t, dt_land, x_trial);
                    } else {
                        land_ok = trap_solve(mask_step, t, dt_land, x_trial);
                    }
                    if (!land_ok) {
                        // Newton failed on the landing sub-step: x_trial
                        // is its warm start, not a state. Reject and
                        // retake shorter (it was committed before).
                        ++st.n_reject;
                        ++st.n_newton_retries;
                        if (h <= h_min * Real{1.0001}) {
                            throw std::runtime_error(
                                "run_transient_trbdf2: Newton failed on "
                                "the diode-event landing at t = "
                                + std::to_string(t + dt_land)
                                + " with h already at h_min — "
                                + newton_last_error);
                        }
                        h = std::max(Real{0.5} * dt_land, h_min);
                        return step::Verdict::Reject;
                    }
                    if (!x_trial.allFinite()) {
                        throw std::runtime_error(
                            "run_transient_trbdf2: event landing at "
                            "t = " + std::to_string(t + dt_land)
                            + " produced a non-finite state — the "
                              "pre-commutation mask is singular at "
                              "this step size");
                    }
                    if constexpr (kE) {
                        if (ed_land) {   // the ESDIRK landing: its BE commit
                            history.restore(ed_hist);
                            history.update_from_state_backward_euler(
                                x_trial, step::Esdirk43::gamma * dt_land);
                        } else {
                            history.update_from_state(x_trial, dt_land);
                            if (interp_land && !sdirk) {
                                // the interpolated state between two ESDIRK
                                // states: its capacitor current is the same
                                // interpolation of theirs (KCL is linear under
                                // one mask); the trapezoidal reconstruction
                                // over dt_land missed KCL by ~1e-5 of the node
                                history.snapshot_into(hist_snap);
                                for (Size i = 0; i < hist_snap.size(); ++i) {
                                    if (hist_snap[i].kind == pwl::DevicePool::StoredKind::Capacitor) {
                                        hist_snap[i].i_prev = ed_ia[i] + ed_thp * (ed_ib[i] - ed_ia[i]);
                                    }
                                }
                                history.restore(hist_snap);
                            }
                        }
                    } else {
                    history.update_from_state(x_trial, dt_land);
                    }
                    // The stateful devices land here too. Only the
                    // linear companion history was committed at an
                    // event landing, so the Coss, Lauritzen, IGBT
                    // tail, PMSM and flux devices integrated the
                    // NEXT step from their pre-landing state with a
                    // stale V_old — found by an adversarial read
                    // while the hysteretic core was being wired in.
                    // (the landing is a single trapezoidal step)
                    if (kE && ed_land && ed_sf) {   // the ESDIRK landing's stages
                        ed_hs_commit(x_trial, step::Esdirk43::gamma * dt_land, dt_land);
                    } else {
                    hs.commit(x_trial, dt_land,
                              pwl::TrBdf2Stage::Trapezoidal);
                    }
                    x = x_trial;
                    x_sig = x;
                    const Real t_hook0 = t;
                    t += dt_land;
                    drop_superseded_event_sample(t);
                    const std::vector<bool> bits_land =
                        diodes.snapshot_on_bits();
                    bool flipped = false;
                    try {
                        flipped = decide_diodes(x, mask_step, dt_land);
                    } catch (const std::runtime_error& e) {
                        // An interpolated landing solved nothing at dt_land:
                        // its noise bound may need a singular factor. Take
                        // it at the bracket end's step instead, a longer
                        // (better conditioned) companion.
                        const Real dt_tb = tb - (t - dt_land);   // t moved
                        if (!is_singular_error(e) || dt_tb <= dt_land) throw;
                        flipped = decide_diodes(x, mask_step, dt_tb);
                    }
                    // Zero-time cascade at t (fixed point over the
                    // remaining diodes, tiny-dt trap pins state).
                    if (flipped) {
                        const Real h_zero =
                            std::max(h_probe_floor,
                                      dt_land * Real{1e-6});
                        bool settled = events::settle(opts.max_event_iterations, [&](Size) {
                            gates_decided = mask_at(t);
                            require_probe(
                                probe_solve(gates_decided, t, h_zero,
                                            x_trial),
                                "a diode event landing,", t);
                            const bool flipped2 =
                                decide_probe(x_trial, gates_decided, t);
                            x_sig = x_trial;
                            return flipped2 ? events::Round::Flipped : events::Round::Settled;
                        }).settled;
                        if (mode_site(t, t, bits_land, settled, h,
                                      gates_decided)) {
                            x_sig = x_trial;
                        }
                    }
                    // A cascade that undid the landing's flip decided
                    // on a probe at h_zero, which for a node held only
                    // by g_off is roundoff (a floating V_th = 0 bridge:
                    // D1 read -212 V there, +0.06 V in the landed x).
                    // Starting the next step from that probe re-found
                    // the same "crossing" every 4e-11 s, 1e7 times.
                    // Nothing commutated: no event; the next step
                    // starts from the landed x, where a diode on its
                    // forbidden side is re-decided at t under the
                    // chatter guard.
                    const bool net_flip =
                        diodes.snapshot_on_bits() != bits_land;
                    if (!net_flip) {
                        x_sig = x;
                    } else {
                        result.commutation_events.push_back(
                            {t,
                             diodes.entries()[cross_idx].branch_id,
                             diodes.entries()[cross_idx].is_on});
                        // probes R2: the landing (under the step's mask)
                        // and the cascade's settled probe.
                        if (hooks != nullptr && hooks->has_event()) {
                            hooks->event(t, x, mask_step, x_trial, gates_decided);
                        }
                        if (lt) lt->event(t, x, mask_step, x_trial, gates_decided);
                    }
                    newton_at_start = false;   // a committed step
                    // a post-event record ahead of this landing goes
                    drop_superseded_event_sample(t);
                    push_time(t, "diode-landing");
                    result.states.push_back(x);
                    result.event_iteration_count.push_back(0);
                    // The landing is a REAL trap solve over dt_land
                    // under the PRE-commutation mask (mask_step); the
                    // bits flip only after it. It is not an
                    // interpolation, and the record says which mask
                    // produced it — so g·v at this sample is the
                    // current the solver carried, with no phantom
                    // g_on·(v − V_f) from the post-flip bits.
                    result.record_switch_state(mask_step);
                    device_state_once(t, x);   // #31: the landing committed them
                    if (hooks) hooks->accepted(t_hook0, t, x, mask_step);
                    // the landing is a trapezoidal sub-step
                    if (lt) {
                        if constexpr (kE) {
                            if (ed_land) ed_loss_accept(t_hook0, t, x, mask_step);
                            else lt->accept_step(t_hook0, t, x, mask_step, nullptr, 0.5, 0, 0.5);
                        } else {
                            lt->accept_step(t_hook0, t, x, mask_step, nullptr, 0.5, 0, 0.5);
                        }
                    }
                    if (structure.active()) reinit.note(mask_step);
                    x_mask = mask_step;
                    x_mask_known = true;
                    gates_decided = mask_step;
                    if (net_flip) {
                        ++st.n_diode_events;
                        sdirk_event = sdirk_off_lin
                            && turned_off(bits_land);
                        sdirk_left = n_sdirk_diode;
                        record_post_event();
                        restart_size_after_diode_event(cross_idx);
                    }
                    ++st.n_accept;
                    // One-step method: no history to rebuild.
                    // KEEP the controller's h (see the gate-memory
                    // note at the top of the loop).
                    //
                    // If this landing CONSUMED an edge-clamped step
                    // (the crossing sat within tolerance of the
                    // gate edge), the gate bookkeeping must still
                    // happen or the edge goes uncounted and the
                    // parity memory desyncs for the rest of the
                    // run.
                    if (landed_on_edge
                        && t >= queue.pending(edge_src) - h_min) {
                        ++st.n_gate_events;
                    }
                    commutated = true;
                    }
                }
            }
            return step::Verdict::Accept;
        };
        // commit: the accepted step (history, state, record, proposal, post-edge restart)
        auto tb_commit = [&]<bool kE = false>() PULSIM_STEP_PHASE {
            (void)ed_hist; (void)ed_loss_accept;   // captured outside kE (MSVC)

            if (!commutated) {
                // ---- accept ----
                if (sdirk) {
                    // Stage-2 base, derivative 0: update_from_state then
                    // stores k2 = f(t + h, x_1), the derivative the next
                    // TR-BDF2 step needs.
                    history.restore(hist_s2);
                    history.update_from_state(x_1, hg);
                    if (sdirk_left > 0) --sdirk_left;
                    if (sdirk_left == 0) sdirk_event = sdirk_gate = false;
                    if (reinit_step) {
                        reinit_left = kReinitSteps - 1;
                        ++st.n_reinit_steps;
                    } else if (reinit_left > 0) {
                        --reinit_left;
                        ++st.n_reinit_steps;
                    }
                    ++st.n_sdirk_steps;
                } else if constexpr (kE) {
                    // the last stage's base, derivative 0: the BE commit then
                    // stores the stage's own derivative (Y5 − ψ5)/(γh), KCL-
                    // consistent at x_1 (stiffly accurate: x_1 = Y5)
                    history.restore(ed_hist);
                    history.update_from_state_backward_euler(
                        x_1, step::Esdirk43::gamma * h);
                } else {
                    history.commit_trbdf2(x_1, h, x_g);
                }
                // Committed under the rule the stage actually used,
                // so the derivative each history carries into the
                // next step is the one the method produced (the
                // hysteretic law integrates from the same base the
                // converged BDF2 residual used, (H_γ, M_γ)).
                if ((kE || sdirk) && ed_sf) {   // the q rows under the rule their stages used
                    ed_hs_commit(x_1, (sdirk ? gs : step::Esdirk43::gamma) * h, h);
                } else {
                hs.commit(x_1, h, pwl::TrBdf2Stage::Bdf2Stage2);
                }
                q_live = true;
                if (structure.active()) x_pre = x;
                const Real t_pre = t;
                x = x_1;
                x_sig = x;
                t += h;
                newton_at_start = false;
                // the slots are this step's own, not a refresh-only probe's;
                // a sliver step after a landed edge keeps the flag (its
                // slots still carry the edge's derivatives, see above)
                if (h > Real{64} * h_min)
                    slots_only_t = std::numeric_limits<Real>::quiet_NaN();
                drop_superseded_event_sample(t);
                push_time(t, "step");
                result.states.push_back(x);
                result.event_iteration_count.push_back(0);
                // Both TR-BDF2 stages solved under mask_step.
                result.record_switch_state(mask_step);
                device_state_once(t, x);   // #31: before a restart in this commit moves t
                if (hooks) hooks->accepted(t_pre, t, x, mask_step);
                if (lt) {
                    // the stage weights: SDIRK(2,2) b = (1 - gs, gs) at
                    // (gs h, h); TR-BDF2 b = (w, w, d) at (0, γh, h)
                    const Real w = Real{0.5} / (Real{2} - gamma);
                    if (sdirk) {
                        lt->accept_step(t_pre, t, x, mask_step, &x_g, 0, Real{1} - gs, gs);
                    } else if constexpr (kE) {
                        ed_loss_accept(t_pre, t, x, mask_step);
                    } else {
                        lt->accept_step(t_pre, t, x, mask_step, &x_g, w, w, Real{1} - Real{2} * w);
                    }
                }
                if (structure.active()) {
                    // D4 record (pwl/reinit.hpp, the fixed engine's code):
                    // the explicit projection from the step's start state.
                    reinit.record(structure, mask_step, x_pre, x, t_pre,
                                  static_cast<long>(result.times.size()) - 1,
                                  opts.on_impulse_error, "trbdf2", result);
                    if (lt) lt->impulses(result.reinit_records);
                }
                x_mask = mask_step;
                x_mask_known = true;
                gates_decided = mask_step;
                ++st.n_accept;
                if (post_event_pending) {
                    st.h_post_event_min = st.h_post_event_min > Real{0}
                        ? std::min(st.h_post_event_min, h) : h;
                    post_event_pending = false;
                }
                if (pending_gate_slot >= 0
                    && (landed_on_edge || !opts.refresh_after_event)) {
                    // The h that survived right after this gate
                    // corner — next cycle's same-parity edge
                    // restarts here instead of shrinking through
                    // rejects. Only a CONTROLLER-chosen h is worth
                    // remembering: an edge-clamped step's h is the
                    // accidental distance to the next edge. (With the
                    // refresh, the controller's PROPOSAL is stored
                    // instead, below.)
                    if (!landed_on_edge) {
                        h_mem_gate[pending_gate_slot] = h;
                    }
                    pending_gate_slot = -1;
                }
                if (landed_on_edge) {
                    ++st.n_gate_events;
                    // A tick due at the edge fires first (with x at t, as it
                    // would at the next step's start): its duty sets the
                    // post-edge mask, and the edge probe can then be
                    // committed. A closed loop's period start is a tick, and
                    // the commit's tick guard below recorded no post-edge
                    // sample at 199 of 199 rising edges (V(sw) ramped across
                    // the first post-edge step: ll14 / ll15).
                    fire_due_ticks();
                    // The rise is at the tick, here; the next step starts
                    // h_probe later (the committed probe), off the tick.
                    if (has_ticks) {
                        gate.learn_period(
                            switch_fn(t + Real{0.5} * h_min), t);
                    }
                    // Zero-time diode cascade AT the edge, BEFORE
                    // any real step under the new mask. Without
                    // this, a gate opening an inductor's only path
                    // integrates one no-path step (GV-scale node
                    // voltages), and the event landing then COMMITS
                    // that garbage into the companion history —
                    // measured on the buck as ~9 A of flux vanishing
                    // per commutation (v_out 19.3 V instead of 24).
                    // The probes pin (v, i) via a tiny-dt trap and
                    // ask which diodes conduct under the post-edge
                    // mask; nothing is committed.
                    const Real h_probe =
                        std::max(h_probe_floor, h * Real{1e-6});
                    bool settled = false;
                    if (has_diodes) {
                        const std::vector<bool> bits0 = diodes.snapshot_on_bits();
                        settled = events::settle(opts.max_event_iterations, [&](Size) {
                            gates_decided = mask_at(t + h_probe);
                            require_probe(
                                probe_solve(gates_decided, t, h_probe,
                                            x_trial),
                                "a gate edge,", t);
                            const bool flipped =
                                decide_probe(x_trial, gates_decided, t);
                            x_sig = x_trial;
                            return flipped ? events::Round::Flipped : events::Round::Settled;
                        }).settled;
                        if (mode_site(t, t + h_probe, bits0, settled, h,
                                      gates_decided)) {
                            x_sig = x_trial;
                        }
                    }
                    restart_after_gate_event(settled, t + h_probe,
                                             h_probe, /*commit=*/true);
                    // Post-edge restart: use the h remembered for
                    // this edge PARITY (rising/falling alternate);
                    // first cycle keeps the controller's h.
                    const int slot =
                        static_cast<int>((st.n_gate_events + lte_ctrl.gate_parity) & 1U);
                    if (h_mem_gate[slot] > Real{0}) {
                        h = h_mem_gate[slot];
                    }
                    pending_gate_slot = slot;
                } else {
                    const Real fac = kE && !sdirk
                        ? step::LteController::accept_factor(err, step::Esdirk43::err_exponent)
                        : step::LteController::accept_factor(err, sdirk);
                    const Real h_done = h;
                    h = h * fac;
                    if (sdirk && diode_slot != Size(-1)) {
                        h_mem_diode[diode_slot] = std::min(h, h_max);
                    }
                    if (landed_on_bp && !landed_on_edge) {
                        // Landed on a source jump or corner: restart as
                        // after a gate edge (commit a probe under the
                        // post-jump source) and do not let the clamped
                        // remainder set the next step.
                        ++st.n_source_breakpoints;
                        h = std::max(h, h_pre_bp);
                        fire_due_ticks();   // as at a gate edge
                        if (has_ticks) {
                            gate.learn_period(
                                switch_fn(t + Real{0.5} * h_min), t);
                        }
                        const Real hp =
                            std::max(h_probe_floor, h_done * Real{1e-6});
                        restart_after_gate_event(false, t + hp, hp,
                                                 /*commit=*/true);
                    } else if (pending_gate_slot >= 0 && !reinit_step) {
                        // Not from the re-init step: its estimate skips the
                        // dependent states, so its proposal (5x when all
                        // states are dependent) would ratchet the post-edge
                        // h up every period (review M1). Step 2 sets it.
                        // Post-edge memory = the controller's proposal
                        // after the first accepted post-edge step: with
                        // a refreshed derivative it is the right
                        // restart for the same corner next period, and
                        // it can grow as well as shrink (storing the
                        // accepted h could only ratchet down).
                        h_mem_gate[pending_gate_slot] = std::min(h, h_max);
                        pending_gate_slot = -1;
                    }
                }
                if (sdirk) {
                    diode_slot = Size(-1);
                }
            }
        };
        // The driver: the step skeleton (`step::run<TrBdf2>`). It is
        // forced inline into this function, unlike the fixed engine's
        // (DECISIONS 16): out of line, every engine local a phase touches was
        // reached through the driver's closure and the phase's closure (a
        // pointer reloaded after each opaque call, +0.3-1 % per step); inline,
        // no closure escapes and the locals are this frame's again.
        // Hooks the driver calls (as slice 3a's): probes R1
        // (`StepHooks::step`: this engine's start state is the committed
        // state at the step start, a right limit: the post-event restart
        // committed it) and the device interface's #31 (`device_state` after
        // the histories' commit). on_step runs after either commit (the
        // ordinary accept or a localised landing) and after an attempt that
        // after_solve ended without one (a zero-time cascade, a retake at
        // the same h): t did not move then, and no hook fires. The start
        // state is copied only with a listener.
        auto run_twin = [&]() PULSIM_STEP_PHASE {
            const bool hook_step = hooks != nullptr && hooks->has_step();
            // probes 0b.10: between a step hook and the next attempt's begin
            // nothing moves x or the mask, so the next step is contiguous for
            // the step sink unless an attempt in between fired no hook (a
            // zero-time cascade, a retake at the same t); the start is copied
            // only for `on_step` functions or a non-contiguous step.
            const bool hook_fns = hooks != nullptr && hooks->has_step_fns();
            bool hook_contig = false;
            Vector x_start;
            SwitchStateMask mask_start;
            Real t_start = Real{0};
            auto tw_begin = [&]() PULSIM_STEP_PHASE {
                t_start = t;
                if (hook_fns || (hook_step && !hook_contig)) {
                    x_start = x;
                    mask_start = x_mask;
                }
                tb_begin();
            };
            auto tw_on_step = [&]() PULSIM_STEP_PHASE {
                if (hooks == nullptr) return;
                if (t == t_start) {
                    hook_contig = false;
                    return;
                }
                device_state_once(t, x);
                if (hook_step) {
                    hooks->step(t_start, x_start, mask_start, t, x, x_mask, hook_contig);
                    hook_contig = true;
                }
            };
            step::BodyClock clock{tb_live, lte_ctrl, t, opts.t_end};
            if (esdirk) {
                // integrator='esdirk43': the same phases; take / error /
                // commit instantiated for the method (chosen once per run)
                auto ed_take_p = [&]() PULSIM_STEP_PHASE {
                    return tb_take.template operator()<true>();
                };
                auto ed_after_p = [&]() PULSIM_STEP_PHASE {
                    return tb_after.template operator()<true>();
                };
                auto ed_error_p = [&]() PULSIM_STEP_PHASE {
                    return tb_error.template operator()<true>();
                };
                auto ed_commit_p = [&]() PULSIM_STEP_PHASE {
                    tb_commit.template operator()<true>();
                };
                auto ebody = step::phase_body(tb_poll, tw_begin, tb_clamp, ed_take_p,
                                              ed_error_p, ed_after_p, ed_commit_p, tw_on_step);
                (void)step::run<step::Esdirk43>(clock, ebody);
                return;
            }
            auto body = step::phase_body(tb_poll, tw_begin, tb_clamp, tb_take, tb_error,
                                         tb_after, tb_commit, tw_on_step);
            (void)step::run<step::TrBdf2>(clock, body);
        };
        run_twin();
        if (hooks != nullptr) hooks->flush_steps();   // the step sink's batched steps
        // A run stopped by should_continue right after an event: the
        // record ends on the committed state, as the snapshot does.
        drop_superseded_event_sample(-std::numeric_limits<Real>::infinity());
        fire_due_monitors();   // instants inside the last step
        if (steps >= opts.max_steps) {
            throw std::runtime_error(
                "run_transient_trbdf2: max_steps ("
                + std::to_string(opts.max_steps)
                + ") exhausted at t = " + std::to_string(t)
                + " of " + std::to_string(opts.t_end)
                + " — the controller is grinding; check for an "
                  "unresolved fast mode or raise max_steps");
        }
        if (std::isfinite(gate.gate_pulse_min())) {
            st.gate_pulse_min = gate.gate_pulse_min();
        }
        st.gate_sample_spacing_max = gate.gate_spacing_max();
        st.probe_floor_used = h_probe_learned;
        st.n_noise_held_decisions = n_noise_held;
        st.n_probe_noise_retakes = n_probe_noise_retakes;
        st.n_probe_newton_retries = n_probe_newton_retries;
        st.n_newton_start_retries = n_newton_start_retries;
        result.mode_structure_checks = structure.checks_run();
        result.ideal_loop_releases = n_loop_releases;
        result.mode_report = std::move(mode_rep);
        record_factor_traffic();
    } catch (const SimulationAborted&) {
        st.n_probe_newton_retries = n_probe_newton_retries;
        throw;
    } catch (const pwl::NewtonRoundoffBudgetExceeded& e) {
        st.n_probe_newton_retries = n_probe_newton_retries;
        result.newton_roundoff_accepts = ro_accepts;
        result.newton_roundoff_omega_max = ro_omega_max;
        result.newton_roundoff_row = ro_row;
        throw SimulationAborted(e.what(), std::move(result), t,
                                /*roundoff_budget=*/true);
    } catch (const std::runtime_error& e) {
        st.n_probe_newton_retries = n_probe_newton_retries;
        result.newton_roundoff_accepts = ro_accepts;
        result.newton_roundoff_omega_max = ro_omega_max;
        result.newton_roundoff_row = ro_row;
        throw SimulationAborted(e.what(), std::move(result), t);
    }
    result.newton_roundoff_accepts = ro_accepts;
    result.newton_roundoff_omega_max = ro_omega_max;
    result.newton_roundoff_row = ro_row;

    // The complete state at the end, so either engine can continue
    // from it (this used to be left EMPTY, with `valid = false`, and
    // a fixed-engine resume from it silently started at zero).
    result.final_snapshot.t = t;
    result.final_snapshot.x = x;
    result.final_snapshot.mask = diodes.num_diodes() > 0
        ? diodes.snapshot_mode_bits() : std::vector<bool>{};
    // The mask x was solved under (the record's last row): diode_on
    // may already be post-edge (see initial_mask above).
    result.final_snapshot.stamped_mask =
        x_mask_known ? mask_bits(x_mask) : std::vector<bool>{};
    if (lt) lt->retract(history);   // the snapshot holds committed slots (DECISIONS 17)
    hs.save(history, result.final_snapshot);
    result.final_snapshot.post_edge_derivatives =
        slots_only_t >= t - h_min;   // NaN: never
    if (has_ticks) {   // the tick sources are ids 0..n-1
        for (std::uint32_t j = 0; j < t_ctrl.size(); ++j) {
            result.final_snapshot.tick_origin.push_back(queue.grid_origin(j));
            result.final_snapshot.tick_next_k.push_back(queue.next_k(j));
        }
    }
    // A resume continues the edge parity and the uncut proposal.
    lte_ctrl.gate_parity = (st.n_gate_events + lte_ctrl.gate_parity) & 1U;
    if (end_cut_h > Real{0}) h = std::max(h, end_cut_h);
    result.final_snapshot.controller = lte_ctrl.save();
    result.final_snapshot.valid = true;
    if (lt) result.final_snapshot.loss_state = lt->save();
    result.newton = newton_ws.take_counters();
    result.gate_mask_fn = nullptr;   // it reads this frame's diode state
    if (lt) lt->write(result);
    return result;
}

}  // namespace pulsim::solver
