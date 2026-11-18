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

#include "pulsim/events/event_queue.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
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
#include "pulsim/solver/result.hpp"
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
#include <vector>

namespace pulsim::solver {

struct TrBdf2Options {
    /// add-unified-device-interface DECISIONS #23 (see SimulationOptions).
    device::DevicePath device_path = device::DevicePath::Legacy;
    Real t_start = Real{0};
    Real t_end   = Real{0};
    Real rtol    = Real{1e-4};
    Real atol    = Real{1e-6};
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
    Size max_event_iterations = Size{16};
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
    /// Newton settings, used only when the circuit has nonlinear
    /// devices. Same knobs and defaults as the fixed engine. The
    /// tolerances are absolute; loosening them is not a fix for an
    /// h_min abort (see SimulationOptions::tol_newton_dx).
    Size max_newton_iterations = Size{50};
    Real tol_newton_dx  = Real{1e-9};
    Real tol_newton_res = Real{1e-9};
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
    Size newton_roundoff_budget = Size{130};
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

/// Per-direction diode crossing signal at a solved state. The
/// existing post-hoc watcher in run_transient uses v_D − V_on for
/// BOTH directions, which biases turn-OFF by the V_th − V_f band (its
/// own doc comment admits i_D is the right signal); with
/// i_D = g_on·(v_D − V_f) the ON→OFF zero is v_D = V_f, so this
/// watches v_D − V_f for ON and v_D − V_on for OFF
/// (`SwitchedDiode::crossing_signal`).
inline Real crossing_signal(const pwl::DiodeEventState& diodes,
                             Size entry_idx, const Vector& x) {
    const auto& e = diodes.entries()[entry_idx];
    // A branch-form diode reads its solved current i_s while ON (D5).
    const Real v_d = e.law_voltage(x, e.is_on);
    return models::SwitchedDiode::crossing_signal(e.is_on, v_d, e.params());
}

/// True when `s` (the diode's crossing_signal) puts it on the side its
/// own commutation rule would leave — `SwitchedDiode::
/// decide`: ON stays iff g_on·(v_D − V_f) > 0, OFF stays iff
/// v_D < V_on.
inline bool forbidden(const pwl::DiodeEventState& diodes,
                      Size entry_idx, Real s) {
    return diodes.entries()[entry_idx].is_on ? (s <= Real{0})
                                             : (s >= Real{0});
}

/// What counts as numerical noise in a forbidden-side signal: 1e-8 of
/// the diode's terminal voltages (plus V_th, V_f and 1 V). A V_th = 0 diode
/// riding its zero-current boundary sits there (the chatter guard owns
/// that case); a missed commutation leaves it volts away.
inline Real held_tolerance(const pwl::DiodeEventState& diodes,
                           Size entry_idx, const Vector& x) {
    const auto& e = diodes.entries()[entry_idx];
    // An ON branch-form diode's signal is its current (review L5).
    if (e.is_on && e.var >= 0) {
        return Real{1e-8} * (Real{1} + std::abs(e.V_f) + std::abs(x[e.var]));
    }
    // An ON conductance-form diode's signal is V_f + i/g_on: 1e-8 of the
    // node voltages is i = 0.7 mA at g_on = 1e3 on a 36 V boost, so a step
    // from +0.37 to -0.71 mA was "riding the boundary", never landed, and
    // the diode turned off later carrying -0.7 mA (sw at -3.5e5 V, 9577
    // floor-accepted steps). Its noise is a leakage-scale current,
    // g_off * (1 + |v_D|) (a bridge diode riding at -1.6e-7 A against
    // 3.3e-7 A), read as voltage through 1/g_on, plus the round-off of
    // forming v_D from its node voltages; a DCM buck's 9e-7 A overshoot
    // (446 V over g_off) is then a crossing. v_D, not the node voltages:
    // a diode floating at 10 kV common mode had a floor of 2e-8 V, a
    // 2e-5 A reverse current taken for noise.
    if (e.is_on && e.g_on > Real{0}) {
        constexpr Real eps = std::numeric_limits<Real>::epsilon();
        const Real va = stamping::read_node_voltage(x, e.from);
        const Real vk = stamping::read_node_voltage(x, e.to);
        return e.g_off / e.g_on
                   * (Real{1} + std::abs(e.V_th) + std::abs(e.V_f)
                      + std::abs(va - vk))
               + Real{64} * eps * (std::abs(va) + std::abs(vk));
    }
    return Real{1e-8}
           * (Real{1} + std::abs(e.V_th) + std::abs(e.V_f)
              + std::abs(stamping::read_node_voltage(x, e.from))
              + std::abs(stamping::read_node_voltage(x, e.to)));
}

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
    /// dynamic device, and the diode bits. To the run's own
    /// tolerance, not bit-exact: the step controller's state is not
    /// in the snapshot, so the segment restarts from h_init.
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
    refuse_device_losses(pool, "trbdf2");   // add-in-kernel-event-losses 3.1

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
    pwl::StepCtx step_ctx{};
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
            return false;
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
        step_ctx = {dt, pwl::TrBdf2Stage::Trapezoidal};
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
    // singular (0 = none seen; see stage 1).
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
            const Real dt_prev = last_probe_dt;
            if (!probe_solve(m, t0, dt_next, xp)) {
                // Not reached on a linear factor; if it were, neither the
                // failed state nor its b_stage may be read: keep the bits.
                xp = x_prev;
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
            for (Size it = 0; it < opts.max_event_iterations; ++it) {
                // A failed (Newton) probe leaves x_trial unwritten —
                // unsized on this first call: keep the bits (it read
                // null, SIGSEGV).
                const SwitchStateMask m0 = mask_at(t_gates);
                if (!probe_solve(m0, opts.t_start, h_probe, x_trial)) {
                    break;
                }
                const bool flipped =
                    decide_probe(x_trial, m0, opts.t_start);
                x_sig = x_trial;
                if (!flipped) {
                    break;
                }
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
    result.times.push_back(opts.t_start);
    result.states.push_back(x);
    result.event_iteration_count.push_back(0);
    result.record_switch_state(initial_mask, !initial_seeded);
    if (hooks) hooks->start(x, initial_mask);

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
    Real h = opts.h_init > Real{0} ? opts.h_init : h_max / Real{100};
    h = std::min(h, h_max);
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
    Real h_mem_gate[2] = {Real{0}, Real{0}};
    int  pending_gate_slot = -1;
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
                                                : opts.observer_origins[j]},
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
    constexpr int kGateSamples = 4;
    constexpr Real kCapPulses = 64;   // step cap in recent narrowest pulses
    const bool learn_gates = !next_edge_fn;
    const Size n_sw_total = static_cast<Size>(graph.num_switches());
    std::vector<Real> gate_last_toggle(n_sw_total, Real{0});
    // The last closed high / low pulse width of each gate bit (0 = none).
    std::vector<Real> gate_w_last(2 * n_sw_total, Real{0});
    // The last three toggle times of each bit per direction (fall, rise).
    std::vector<Real> gate_dir_t(6 * n_sw_total,
                                 -std::numeric_limits<Real>::infinity());
    std::vector<bool> gate_toggled(n_sw_total, false);
    SwitchStateMask gate_prev = switch_fn(opts.t_start
                                          + Real{0.5} * h_min);
    Real gate_pulse_min = std::numeric_limits<Real>::infinity();
    // A "pulse" shorter than this is a floating-point glitch of the
    // user's phase comparison, not a pulse a step could resolve.
    const Real gate_pulse_floor =
        std::max(Real{16} * h_min, span_ref * Real{1e-9});
    Real gate_spacing_max = Real{0};
    // Per gate bit: the largest sampling spacing used since that bit's
    // PREVIOUS seen toggle (or the run start, before its first one),
    // i.e. while the pulse now closing was open or while pulses could
    // have passed unseen. A pulse is "undersampled" only against THAT
    // spacing: compared with the run-wide maximum, every smooth SPWM
    // whose pulses narrow gradually (each one landed, at a spacing
    // already capped by the previous pulse) was flagged — 22 times
    // over 0.2 s of a 10 kHz three-phase VSI whose 16000 edges were
    // all landed. Not reset at a bit's FIRST toggle: a PWM that starts
    // late is found by the walking sample only after some pulses went
    // by at the idle spacing, and those are what the warning confesses.
    std::vector<Real> gate_spacing_open(n_sw_total, Real{0});
    std::vector<Size> gate_bits;
    for (Size i = 0; i < n_sw_total; ++i) {
        if (!diode_owned.get(i)) {
            gate_bits.push_back(i);
        }
    }
    Real gate_jitter = Real{0};
    // The gate bits the diode bits were last decided (or last solved)
    // under. A gate edge that falls within an ulp of a step's END is
    // not seen by that step's samples (t + h rounds to the pre-edge
    // side) and so is not landed, and the zero-time cascade a landed
    // edge gets never ran: the next step then solved the post-edge
    // gates with the pre-edge diode bits. Measured where a controller
    // tick (which ends a step exactly on k·T) met a 10 kHz buck's
    // turn-on: a 2.1e-11 s sample with Q1 ON and D1 ON, v_sw = +6 V
    // across two 1 mOhm stamps. The step start compares and settles.
    SwitchStateMask gates_decided = gate_prev;
    // `b` is a stamped mask: a gated device's bit there is its conduction
    // state, so its gate is compared with the gate it was decided under.
    const SwitchStateMask gated = diodes.gated_bits();
    auto gates_differ = [&](const SwitchStateMask& a,
                            const SwitchStateMask& b) {
        for (const Size i : gate_bits) {
            if (gated.get(i) ? a.get(i) != diodes.gate_of(i)
                             : a.get(i) != b.get(i)) {
                return true;
            }
        }
        return false;
    };
    auto learn_gate_toggles = [&](const SwitchStateMask& sw,
                                   Real at_t) {
        if (sw == gate_prev) {
            return;
        }
        for (Size i = 0; i < n_sw_total; ++i) {
            if (diode_owned.get(i) || sw.get(i) == gate_prev.get(i)) {
                continue;
            }
            if (gate_toggled[i]) {
                const Real w = at_t - gate_last_toggle[i];
                if (w > gate_pulse_floor) {
                    gate_w_last[2 * i + (gate_prev.get(i) ? 1 : 0)] = w;
                }
                if (w > gate_pulse_floor && w < gate_pulse_min) {
                    gate_pulse_min = w;
                }
                // 1 % margin: a pulse as wide as the spacing cannot fit
                // between two samples, and the widths carry the
                // bisection's h_min of noise. Every pulse, not only a new
                // minimum: the cap lapses on a quiet bit (burst mode), so
                // a pulse after the pause can be undersampled again.
                if (w > gate_pulse_floor
                    && gate_spacing_open[i] > Real{1.01} * w) {
                    ++st.n_gate_undersampled;
                }
                gate_spacing_open[i] = Real{0};
            }
            gate_toggled[i] = true;
            gate_last_toggle[i] = at_t;
            Real* dt3 = &gate_dir_t[6 * i + (sw.get(i) ? 3 : 0)];
            dt3[0] = dt3[1];
            dt3[1] = dt3[2];
            dt3[2] = at_t;
        }
        gate_prev = sw;
    };
    // A periodic gate is a periodic source too: the default ceiling gives
    // every PWM / pulse / sine source 20 steps per period, and a gate from
    // switch_fn (a closed loop's, declared edges or not) got span/1000. On
    // ll15 (closed-loop boost, 5 ms) that was one 4 us step per 5 us
    // period, each at its local tolerance and all errors of one sign:
    // I(L) 0.066 A off after 23 periods. A rise counts only at a controller
    // tick (a trailing-edge closed-loop PWM rises on its tick); once a gate
    // bit has so risen three times a constant period p apart, a step is
    // capped at p/20 (default ceiling only; the floors stay those of
    // h_max). An open-loop gate keeps its fewer steps per period (the CCM
    // buck's <= 8 per period is a pinned decision) unless its rises happen
    // to fall on ticks (then it is capped at most at T_tick/20). The cap is
    // a run-wide minimum (a short periodic episode keeps it) and is
    // relearned after a resume (the rise times are not in the snapshot).
    Real last_tick_t = -std::numeric_limits<Real>::infinity();
    std::vector<Real> gate_rise_t(3 * n_sw_total,
                                  -std::numeric_limits<Real>::infinity());
    SwitchStateMask gate_rise_prev = gate_prev;
    Real gate_period_cap = std::numeric_limits<Real>::infinity();
    auto learn_gate_period = [&](const SwitchStateMask& sw, Real at_t) {
        if (sw == gate_rise_prev) return;
        for (const Size i : gate_bits) {
            if (!sw.get(i) || gate_rise_prev.get(i)
                || !(std::abs(at_t - last_tick_t) <= Real{4} * h_min)) {
                continue;
            }
            Real* r3 = &gate_rise_t[3 * i];
            r3[0] = r3[1];
            r3[1] = r3[2];
            r3[2] = at_t;
            const Real p1 = r3[2] - r3[1], p0 = r3[1] - r3[0];
            // p1 >= 20 pulse floors: rises a few h_min apart (glitches of
            // a phase comparison) are no period to cap the run by.
            if (std::isfinite(p0) && p1 >= Real{20} * gate_pulse_floor
                && std::abs(p1 - p0) <= Real{1e-6} * p1 + Real{4} * h_min) {
                gate_period_cap = std::min(gate_period_cap, p1 / Real{20});
            }
        }
        gate_rise_prev = sw;
    };
    // The step cap: kCapPulses x the narrowest RECENT pulse. Per bit, the last
    // high and low widths count while the bit keeps toggling (quiet for at
    // most twice its last period; a quiet bit keeps the run's narrowest
    // pulse as its cap); a running minimum over the run held a
    // closed loop's 0.28 ns start-up pulse as the cap for ever (519,783
    // steps in 0.3 ms of cl_boost_pi with an opaque gate). A phase whose
    // ending toggle comes at a regular period (a trailing-edge PWM's rise
    // at kT) is not capped until just before the predicted toggle: the
    // same run took 58,514 steps with the cap over the whole long phase.
    // Returns the step limit (infinite: none).
    // The narrowest recent pulse of a bit whose cap was lifted: the gate
    // samples inside a lifted step stay half of it apart (a gate that gets
    // faster, a skipped pulse resuming, puts pulses where none is
    // predicted: only the samples can see them).
    Real gate_sample_w = std::numeric_limits<Real>::infinity();
    std::vector<Real> gate_frac;
    auto gate_pulse_cap = [&](Real at_t) {
        Real lim = std::numeric_limits<Real>::infinity();
        gate_sample_w = std::numeric_limits<Real>::infinity();
        for (const Size i : gate_bits) {
            const Real wh = gate_w_last[2 * i + 1];
            const Real wl = gate_w_last[2 * i];
            if (!gate_toggled[i] || (wh <= Real{0} && wl <= Real{0})) {
                continue;
            }
            if (at_t - gate_last_toggle[i] > Real{2} * (wh + wl)) {
                // Quiet: the narrowest pulse of the run, as before. Pulses
                // resuming after a pause (burst mode, a soft-start pulse
                // then PWM) were stepped over with the cap lapsed: 4 of 99
                // 10 ns pulses, every 1 ns one.
                if (std::isfinite(gate_pulse_min)) {
                    lim = std::min(lim, Real{2} * gate_pulse_min);
                }
                continue;
            }
            Real w = std::numeric_limits<Real>::infinity();
            if (wh > Real{0}) w = std::min(w, wh);
            if (wl > Real{0}) w = std::min(w, wl);
            // A pulse a step holds must show at a sample. A pulse ends
            // with a toggle in each direction once per period; when one
            // direction is
            // regular, a step that ends before its next predicted toggle
            // cannot hold a pulse that ends there, and any other toggle it
            // holds is seen at its end and located.
            Real reg = std::numeric_limits<Real>::infinity();
            bool any_reg = false;
            for (const Size dir : {Size{0}, Size{3}}) {
                const Real* d3 = &gate_dir_t[6 * i + dir];
                const Real p1 = d3[2] - d3[1], p0 = d3[1] - d3[0];
                if (!std::isfinite(p0) || p1 <= Real{0}
                    || std::abs(p1 - p0) > Real{1e-6} * p1 + Real{4} * h_min) {
                    continue;
                }
                any_reg = true;
                const Real margin = Real{2} * w + Real{1e-3} * p1;
                reg = std::min(reg, d3[2] + p1 - margin - at_t);
            }
            // kCapPulses * w with the gate sampled every w/2 inside the
            // step (gate_sample_w): the same sampling as 2 w with 4 samples,
            // in 32x fewer solves. cl_boost_pi's start-up pulses (0.28 ns,
            // then 0.585 ns, before the rises prove periodic) cost 26,490
            // steps in two periods at 2 w.
            lim = std::min(lim, any_reg ? std::max(kCapPulses * w, reg)
                                        : kCapPulses * w);
            gate_sample_w = std::min(gate_sample_w, w);
        }
        return lim;
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
            }
        }
        event_sample_t = -std::numeric_limits<Real>::infinity();
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
        const Real dtp = last_probe_dt;
        ++st.n_derivative_refreshes;
        post_event_pending = true;
        // No commit across t_end or a controller tick (it would fire
        // dtp late): replace the slots only.
        if (!commit || t + dtp >= opts.t_end - h_min
            || t + dtp >= queue.next_of(events::Kind::ControlTick) - h_min) {
            history.refresh_derivatives(x_trial, dtp);
            hs.refresh_derivatives(x_trial, dtp);
            slots_only_t = t + dtp;   // where the unused probe reached
            return;
        }
        history.update_from_state(x_trial, dtp);
        hs.commit(x_trial, dtp, pwl::TrBdf2Stage::Trapezoidal);
        // The toggle is at the edge, not at the next step's start
        // (t + dtp): dtp differs per edge and biased gate_pulse_min.
        if (learn_gates) {
            // m_probe is a stamped mask: a gated device's bit there is its
            // conduction state, not its gate.
            learn_gate_toggles(diodes.num_gated() > 0 ? switch_fn(t_gates)
                                                      : m_probe, t);
        }
        x = x_trial;
        x_sig = x;
        const Real t_hook0 = t;
        t += dtp;
        drop_superseded_event_sample(t);
        result.times.push_back(t);
        result.states.push_back(x);
        result.event_iteration_count.push_back(0);
        result.record_switch_state(m_probe);
        if (hooks) hooks->accepted(t_hook0, t, x, m_probe);
        if (structure.active()) reinit.note(m_probe);
        x_mask = m_probe;
        x_mask_known = true;
        gates_decided = m_probe;
    };

    // ---- L-stable SDIRK(2,2) restart (Alexander, gamma_s = 1 - 1/sqrt2) ----
    // Stage 1: backward Euler over gamma_s*h from x_n; stage 2: backward
    // Euler over gamma_s*h from x_n + ((1-gs)/gs)(X_1 - x_n). C/(gs h)
    // equals 2C/(gamma h), so both stages are the trapezoidal companion
    // at dt = gamma*h with the DERIVATIVE slots zeroed and the STATE slots
    // set to the stage's base: the same factor, no new history method.
    const Real gs = Real{1} - Real{1} / std::sqrt(Real{2});
    const bool sdirk_possible = !hs.any();
    const bool sdirk_ok = sdirk_possible
                          && (opts.sdirk_on_linear_circuits || nl_effective);
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
    const bool sdirk_off_lin = opts.sdirk_after_turn_off && sdirk_possible
                               && !nl_effective && !sdirk_ok;
    bool sdirk_event = false;   // the pending restart came from such an event
    auto restart_ok = [&]() { return sdirk_ok || sdirk_event; };
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
    std::vector<Real> h_mem_diode;
    Size diode_slot = Size(-1);
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
            m_rec = mask_at(t);
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
        result.times.push_back(tp);
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
    Size sdirk_left = has_diodes ? opts.sdirk_after_settle : Size{0};
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
            last_tick_t = t;
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
        while (t >= queue.next_of(events::Kind::Monitor) - h_min) {
            for (const auto& ev : queue.take_due(
                     t, h_min, events::kind_bit(events::Kind::Monitor))) {
                tick_fn(ev.source, t, x);
                ++st.ctrl_tick_counts[ev.source];
            }
        }
    };

    try {
        while (t < queue.next_of(events::Kind::EndOfRun) - h_min
               && steps < opts.max_steps) {
            if (should_continue && !should_continue()) {
                break;   // partial trace preserved
            }
            ++steps;

            // ---- controller tick: fire AT the scheduled instant,
            //      before this step's mask is sampled ----
            fire_due_ticks();
            fire_due_monitors();

            // Shortly after the chatter guard gave up, the LTE may not
            // walk h below h_chatter: under all-OFF bridge bits a 1 mH
            // line and g_off make a 1 ps mode whose kink the LTE reads
            // as error at any h, and h sat at h_min until max_steps.
            const Real h_lo = std::max(
                (t - chatter_exhaust_t <= Real{1e3} * h_chatter)
                    ? h_chatter : h_min,
                h_cond);
            const Real t_stop = queue.next_of(events::Kind::EndOfRun);
            h = std::clamp(h, std::min(h_lo, t_stop - t),
                           std::min(h_max, t_stop - t));
            // The gates as the step starts (one switch_fn call), and
            // the step cap the narrowest pulse seen so far implies.
            SwitchStateMask sw_now = switch_fn(t + Real{0.5} * h_min);
            learn_gate_period(sw_now, t);
            if (has_ticks && !(opts.h_max > Real{0})) {
                h = std::min(h, std::max(gate_period_cap, h_lo));
            }
            if (learn_gates) {
                learn_gate_toggles(sw_now, t);
                const Real cap = gate_pulse_cap(t);
                if (std::isfinite(cap)) {
                    h = std::min(h, std::max({cap, h_min, h_cond}));
                }
            }
            // The step the controller and the caps chose; a landing below
            // shortens it (see the conditioning floor at stage 1).
            const Real h_ctrl = h;
            if (has_diodes && gates_differ(sw_now, gates_decided)) {
                // An edge at this step boundary that no step landed
                // (see gates_decided): settle the diodes under the
                // gates the step will use, as a landed edge does.
                bool settled = false;
                for (Size it = 0; it < opts.max_event_iterations;
                     ++it) {
                    gates_decided = mask_at(t + Real{0.5} * h_min);
                    require_probe(probe_solve(gates_decided, t,
                                              h_probe_floor, x_trial),
                                  "a gate edge at a step boundary,", t);
                    const bool flipped =
                        decide_probe(x_trial, gates_decided, t);
                    x_sig = x_trial;
                    if (!flipped) {
                        settled = true;
                        break;
                    }
                }
                restart_after_gate_event(settled,
                                         t + Real{0.5} * h_min,
                                         h_probe_floor,
                                         /*commit=*/false);
                ++st.n_gate_boundary_settles;
            }
            // Land the next tick exactly, like a gate edge.
            const Real t_next_tick =
                queue.next_of(events::Kind::ControlTick);
            if (t_next_tick > t && t_next_tick - t < h) {
                h = std::max(t_next_tick - t, h_min);
            }
            // ---- source-breakpoint landing ----
            bool landed_on_bp = false;
            Real bp_t = Real{0};
            const Real h_pre_bp = h;
            if (!corners.empty() || has_iwave) {
                bp_t = next_breakpoint_after(t);
                if (bp_t <= t + h) {
                    h = std::max(bp_t - t, h_min);
                    landed_on_bp = true;
                }
            }

            // ---- gate-edge landing ----
            bool landed_on_edge = false;
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
            bool gates_flat = false;
            if (!landed_on_edge) {
                // Samples per step: 4, or half the recent narrowest pulse
                // apart in a lifted step (switch_fn calls only, no solves).
                // Once a recent pulse width w is known the samples are w/2
                // apart (DECISIONS #30), and a step shorter than w/2 needs
                // only its end: the floor of 4 is for gates not yet seen.
                // It cost the CCM buck 3 Python switch_fn calls per step.
                int n_gs = kGateSamples;
                if (learn_gates && std::isfinite(gate_sample_w)
                    && gate_sample_w > Real{0}) {
                    n_gs = static_cast<int>(std::clamp(
                        std::ceil(Real{2} * h / gate_sample_w),
                        Real{1}, Real{4096}));
                }
                if (learn_gates) {
                    const Real spacing =
                        h / static_cast<Real>(n_gs);
                    gate_spacing_max =
                        std::max(gate_spacing_max, spacing);
                    for (const Size i : gate_bits) {
                        gate_spacing_open[i] =
                            std::max(gate_spacing_open[i], spacing);
                    }
                }
                // h/4, h/2, 3h/4, h, plus one sample at a fraction
                // that walks the golden-ratio sequence from step to
                // step: fixed fractions alone PHASE-LOCK to a PWM when
                // h is a multiple of its period (h_max = span/1000 =
                // 20 µs against 10 µs), and then never see a pulse
                // that starts after the run has reached h_max.
                gate_jitter += Real{0.6180339887498949};
                gate_jitter -= std::floor(gate_jitter);
                std::vector<Real>& frac = gate_frac;
                frac.resize(static_cast<std::size_t>(n_gs) + 1);
                for (int k = 1; k <= n_gs; ++k) {
                    frac[static_cast<std::size_t>(k - 1)] =
                        static_cast<Real>(k) / static_cast<Real>(n_gs);
                }
                frac[static_cast<std::size_t>(n_gs)] =
                    std::clamp(gate_jitter, Real{0.01}, Real{0.99});
                std::sort(frac.begin(), frac.end());
                Real lo = t, hi = t + h;
                bool edge_found = false;
                for (const Real f : frac) {
                    const Real tk = f >= Real{1} ? t + h : t + h * f;
                    if (!(switch_fn(tk) == sw_now)) {
                        hi = tk;
                        edge_found = true;
                        break;
                    }
                    lo = tk;
                }
                gates_flat = !edge_found;
                if (edge_found) {
                    // A periodic bit predicts its next toggle: two calls
                    // 4 h_min either side of it bound the edge, and the
                    // bisection reads every midpoint outside the bounds
                    // from them instead of calling switch_fn (~25 calls ->
                    // ~5 per edge). Same result: the bisection already
                    // assumes one change of the mask in (lo, hi], under
                    // which a point the gate was verified at answers for
                    // every midpoint on its side.
                    Real known_lo = lo, known_hi = hi;
                    if (learn_gates) {
                        Real te = std::numeric_limits<Real>::infinity();
                        for (const Size i : gate_bits) {
                            const Real* d3 =
                                &gate_dir_t[6 * i + (sw_now.get(i) ? 0 : 3)];
                            const Real p1 = d3[2] - d3[1], p0 = d3[1] - d3[0];
                            if (std::isfinite(p0) && p1 > Real{0}
                                && std::abs(p1 - p0)
                                       <= Real{1e-6} * p1 + Real{4} * h_min
                                && d3[2] + p1 > lo) {
                                te = std::min(te, d3[2] + p1);
                            }
                        }
                        if (te > lo && te < hi) {
                            for (const Real tv : {te - Real{4} * h_min,
                                                  te + Real{4} * h_min}) {
                                if (!(tv > known_lo && tv < known_hi)) continue;
                                if (switch_fn(tv) == sw_now) {
                                    known_lo = tv;
                                } else {
                                    known_hi = tv;
                                }
                            }
                        }
                    }
                    for (int it = 0; it < 60 && (hi - lo) > h_min;
                         ++it) {
                        const Real mid = Real{0.5} * (lo + hi);
                        const bool same = mid <= known_lo ? true
                            : mid >= known_hi ? false
                            : (switch_fn(mid) == sw_now);
                        if (same) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    h = std::max(hi - t, h_min);
                    landed_on_edge = true;
                    queue.locate(edge_src, hi);
                }
            }

            // Step mask: sampled at the midpoint — piecewise-
            // constant gates are constant over (t, t+h) by the
            // clamp above, so the midpoint is unambiguous. When the
            // samples above were all equal it was one of them.
            SwitchStateMask mask_step =
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
            std::vector<const pwl::DependentStructure*> reinit_new;
            if (structure.active()) {
                reinit_new = reinit.entering(structure, mask_step, false);
            }
            const bool reinit_step = !reinit_new.empty();
            if (reinit_step && !sdirk_possible) {
                const Index el = reinit_new.front()->elements.front();
                throw UnsupportedError(
                    "trbdf2", "dependent_states",
                    std::string(graph.branch_name(el)),
                    std::string(pwl::stored_kind_name(pool.kind_of(el))),
                    std::format(
                        "engine 'trbdf2': at t = {} the mode {} makes states "
                        "dependent through the ideal element '{}'; their "
                        "re-initialisation is the SDIRK x3 restart, which this "
                        "engine does not take with a Coss, Lauritzen-diode, "
                        "IGBT-tail, PMSM or saturable-inductor history. Use "
                        "engine='pwl', or give '{}' R_on > 0 and g_off > 0.",
                        t, mask_step.to_string(), graph.branch_name(el),
                        graph.branch_name(el)));
            }

            // ---- stage 1: TR over γh (or SDIRK stage 1, BE over γ_s h) ----
            const Real hg = gamma * h;
            const bool sdirk = (sdirk_left > 0 && restart_ok()) || reinit_step
                               || reinit_left > 0;
            // Where x_g sits in time (the crossing detector reads it).
            const Real h_pt = sdirk ? gs * h : hg;
            if (sdirk) {
                history.snapshot_into(hist_snap);
                hist_s1 = hist_snap;
                zero_derivatives(hist_s1);
                history.restore(hist_s1);
            }
            history.compute_b_extra(hg, b_stage);
            accumulate_sources(t + h_pt, b_src);
            b_stage += b_src;
            x_g = x;                 // Newton warm start
            step_ctx = {hg, pwl::TrBdf2Stage::Trapezoidal};
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
                if (t + h >= opts.t_end && h <= Real{64} * h_min) {
                    // the SDIRK stage swapped in derivative-free history:
                    // the run's final snapshot must hold the real one
                    if (sdirk) history.restore(hist_snap);
                    break;
                }
                const bool landing = h < h_ctrl * (Real{1} - Real{1e-12});
                if (landing || Real{4} * h > std::min(h_max, opts.t_end - t)) {
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
                step_ctx = {h, pwl::TrBdf2Stage::Bdf2Stage2};
                x_1 = x_g;           // Newton warm start
                stage_ok =
                    solve_dispatch(mask_step, hg, b_stage, x_1);
                ++st.n_solves;
            }
            if (stage_singular) {
                ++st.n_reject;
                ++st.n_conditioning_rejects;
                h_cond = Real{4} * h;
                st.h_conditioning_floor = h_cond;
                h = h_cond;
                continue;
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
                continue;
            }

            // ---- LTE over the differential variables ----
            Real err_sq = Real{0};
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
                // Factor 2 (Bank et al.): without it the estimate was
                // half the true local error (true/estimate 1.97 at
                // h*lambda = -0.05, p10_lte.py), so rtol meant twice itself.
                const Real lte =
                    Real{2} * clte * h
                    * (f_n / gamma
                       - f_g / (gamma * (Real{1} - gamma))
                       + f_1 / (Real{1} - gamma));
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
            if ((opts.filter_error_estimate || (sdirk && sdirk_event && !reinit_step))
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
            Real err =
                n_err > 0 ? std::sqrt(err_sq
                                       / static_cast<Real>(n_err))
                           : Real{0};
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
                for (Size i = 0; i < entries.size(); ++i) {
                    const auto& e = entries[i];
                    const Real dv =
                        stamping::read_node_voltage(noise_y, e.from)
                        + stamping::read_node_voltage(noise_y, e.to);
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
                const Real err_n = std::sqrt(e2 / static_cast<Real>(entries.size()));
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
                continue;
            }

            if (err > Real{1} && h <= h_lo * Real{1.0001}) {
                // Cannot shrink further: accept, but CONFESS —
                // silence here would be an unbounded-error hole.
                ++st.n_forced_accepts;
            }
            if (err > Real{1} && h > h_lo * Real{1.0001}) {
                // reject: elementary controller, order-3 exponent
                const Real fac = std::clamp(
                    Real{0.9} * std::pow(err, sdirk ? Real{-0.5}
                                                    : Real{-1.0 / 3.0}),
                    Real{0.1}, Real{0.9});
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
                continue;
            }

            // ---- diode crossing detection on the accepted step ----
            bool commutated = false;
            // shortcut: no crossing is located inside the re-initialisation
            // step (its landing would be a trapezoidal sub-step across the
            // impulse); a diode left on its forbidden side is re-decided at
            // the start of the next step (the held-diode path). Upgrade by
            // landing with an SDIRK sub-step.
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
                            for (const Real h_first : {h_probe_floor, dt_floor}) {
                                for (Size it2 = 0;
                                     it2 < opts.max_event_iterations;
                                     ++it2) {
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
                                    if (!flipped) {
                                        break;
                                    }
                                }
                                if (diodes.snapshot_on_bits() != bits0) {
                                    break;
                                }
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
                                    x_sig = x_1;
                                    continue;
                                }
                            }
                            if (chatter_short_t != t) {
                                chatter_short_t = t;
                                h = std::max(h * Real{1e-2}, h_min);
                                ++st.n_reject;
                                continue;
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
                    Real t_loc = t_star;
                    int side = 0;
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
                    Vector x_ta = x_sig;
                    Vector x_tb = tb < t + h ? x_g : x_1;
                    for (int it = 0; it < 40; ++it) {
                        t_loc = std::clamp(
                            ta - sa * (tb - ta) / (sb - sa),
                            ta + Real{0.05} * (tb - ta),
                            tb - Real{0.05} * (tb - ta));
                        const Real dts = t_loc - t;
                        if (dts <= dt_floor
                            || (tb - ta) <= h_min) {
                            break;
                        }
                        bool trial_ok = true;
                        try {
                            // A failed Newton trial leaves x_trial at
                            // its warm start: never read it.
                            trial_ok = trap_solve(mask_step, t, dts,
                                                  x_trial);
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
                            break;
                        }
                        const Real sm =
                            detail_trbdf2::crossing_signal(
                                diodes, cross_idx, x_trial);
                        if ((sm > Real{0}) == (sa > Real{0})) {
                            ta = t_loc;
                            sa = sm;
                            sa_raw = sm;
                            x_ta = x_trial;
                            if (side == -1) { sb *= Real{0.5}; }
                            side = -1;
                        } else {
                            tb = t_loc;
                            sb = sm;
                            sb_raw = sm;
                            x_tb = x_trial;
                            if (side == +1) { sa *= Real{0.5}; }
                            side = +1;
                        }
                        if ((tb - ta) < std::max(h_min,
                                                  Real{1e-4} * h)
                            && spike_ok(sb_raw)) {
                            break;
                        }
                    }
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
                    const bool interp_land = turn_off && !nl_effective
                        && !spike_ok(sb_raw) && sa_raw != sb_raw
                        && (sa_raw > Real{0}) != (sb_raw > Real{0})
                        // only a bracket at its floor: an abandoned or
                        // exhausted refinement still lands with a solve
                        && tb - ta <= Real{2} * dt_floor;
                    bool land_ok = true;
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
                        if (dt_land > t_lin - t) ++st.n_interp_floor_landings;
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
                        continue;
                    }
                    if (!x_trial.allFinite()) {
                        throw std::runtime_error(
                            "run_transient_trbdf2: event landing at "
                            "t = " + std::to_string(t + dt_land)
                            + " produced a non-finite state — the "
                              "pre-commutation mask is singular at "
                              "this step size");
                    }
                    history.update_from_state(x_trial, dt_land);
                    // The stateful devices land here too. Only the
                    // linear companion history was committed at an
                    // event landing, so the Coss, Lauritzen, IGBT
                    // tail, PMSM and flux devices integrated the
                    // NEXT step from their pre-landing state with a
                    // stale V_old — found by an adversarial read
                    // while the hysteretic core was being wired in.
                    // (the landing is a single trapezoidal step)
                    hs.commit(x_trial, dt_land,
                              pwl::TrBdf2Stage::Trapezoidal);
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
                        for (Size it2 = 0;
                             it2 < opts.max_event_iterations;
                             ++it2) {
                            gates_decided = mask_at(t);
                            require_probe(
                                probe_solve(gates_decided, t, h_zero,
                                            x_trial),
                                "a diode event landing,", t);
                            const bool flipped2 =
                                decide_probe(x_trial, gates_decided, t);
                            x_sig = x_trial;
                            if (!flipped2) {
                                break;
                            }
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
                    }
                    result.times.push_back(t);
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
                    if (hooks) hooks->accepted(t_hook0, t, x, mask_step);
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

            if (!commutated) {
                // ---- accept ----
                if (sdirk) {
                    // Stage-2 base, derivative 0: update_from_state then
                    // stores k2 = f(t + h, x_1), the derivative the next
                    // TR-BDF2 step needs.
                    history.restore(hist_s2);
                    history.update_from_state(x_1, hg);
                    if (sdirk_left > 0) --sdirk_left;
                    if (sdirk_left == 0) sdirk_event = false;
                    if (reinit_step) {
                        reinit_left = kReinitSteps - 1;
                        ++st.n_reinit_steps;
                    } else if (reinit_left > 0) {
                        --reinit_left;
                        ++st.n_reinit_steps;
                    }
                    ++st.n_sdirk_steps;
                } else {
                    history.commit_trbdf2(x_1, h, x_g);
                }
                // Committed under the rule the stage actually used,
                // so the derivative each history carries into the
                // next step is the one the method produced (the
                // hysteretic law integrates from the same base the
                // converged BDF2 residual used, (H_γ, M_γ)).
                hs.commit(x_1, h, pwl::TrBdf2Stage::Bdf2Stage2);
                if (structure.active()) x_pre = x;
                const Real t_pre = t;
                x = x_1;
                x_sig = x;
                t += h;
                drop_superseded_event_sample(t);
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(0);
                // Both TR-BDF2 stages solved under mask_step.
                result.record_switch_state(mask_step);
                if (hooks) hooks->accepted(t_pre, t, x, mask_step);
                if (structure.active()) {
                    // D4 record (pwl/reinit.hpp, the fixed engine's code):
                    // the explicit projection from the step's start state.
                    reinit.record(structure, mask_step, x_pre, x, t_pre,
                                  static_cast<long>(result.times.size()) - 1,
                                  opts.on_impulse_error, "trbdf2", result);
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
                        learn_gate_period(
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
                        for (Size it = 0;
                             it < opts.max_event_iterations;
                             ++it) {
                            gates_decided = mask_at(t + h_probe);
                            require_probe(
                                probe_solve(gates_decided, t, h_probe,
                                            x_trial),
                                "a gate edge,", t);
                            const bool flipped =
                                decide_probe(x_trial, gates_decided, t);
                            x_sig = x_trial;
                            if (!flipped) {
                                settled = true;
                                break;
                            }
                        }
                    }
                    restart_after_gate_event(settled, t + h_probe,
                                             h_probe, /*commit=*/true);
                    // Post-edge restart: use the h remembered for
                    // this edge PARITY (rising/falling alternate);
                    // first cycle keeps the controller's h.
                    const int slot =
                        static_cast<int>(st.n_gate_events & 1U);
                    if (h_mem_gate[slot] > Real{0}) {
                        h = h_mem_gate[slot];
                    }
                    pending_gate_slot = slot;
                } else {
                    const Real e_used =
                        std::max(err, Real{1e-4});
                    const Real fac = std::clamp(
                        Real{0.9}
                            * std::pow(e_used, sdirk ? Real{-0.5}
                                                     : Real{-1.0 / 3.0}),
                        Real{0.2}, Real{5.0});
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
                            learn_gate_period(
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
        }
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
        if (std::isfinite(gate_pulse_min)) {
            st.gate_pulse_min = gate_pulse_min;
        }
        st.gate_sample_spacing_max = gate_spacing_max;
        st.probe_floor_used = h_probe_learned;
        st.n_noise_held_decisions = n_noise_held;
        st.n_probe_noise_retakes = n_probe_noise_retakes;
        st.n_probe_newton_retries = n_probe_newton_retries;
        result.mode_structure_checks = structure.checks_run();
        result.ideal_loop_releases = n_loop_releases;
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
        ? diodes.snapshot_on_bits() : std::vector<bool>{};
    // The mask x was solved under (the record's last row): diode_on
    // may already be post-edge (see initial_mask above).
    result.final_snapshot.stamped_mask =
        x_mask_known ? mask_bits(x_mask) : std::vector<bool>{};
    hs.save(history, result.final_snapshot);
    result.final_snapshot.post_edge_derivatives =
        slots_only_t >= t - h_min;   // NaN: never
    if (has_ticks) {   // the tick sources are ids 0..n-1
        for (std::uint32_t j = 0; j < t_ctrl.size(); ++j) {
            result.final_snapshot.tick_origin.push_back(queue.grid_origin(j));
            result.final_snapshot.tick_next_k.push_back(queue.next_k(j));
        }
    }
    result.final_snapshot.valid = true;
    result.newton = newton_ws.take_counters();
    return result;
}

}  // namespace pulsim::solver
