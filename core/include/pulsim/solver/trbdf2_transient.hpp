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

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
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
    /// Newton settings, used only when the circuit has nonlinear
    /// devices. Same knobs and defaults as the fixed engine. The
    /// tolerances are absolute; loosening them is not a fix for the
    /// h_min stall (see SimulationOptions::tol_newton_dx).
    Size max_newton_iterations = Size{50};
    Real tol_newton_dx  = Real{1e-9};
    Real tol_newton_res = Real{1e-9};
    bool enable_newton_line_search = false;
    bool enable_newton_lm = false;
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
    const Real v_d = stamping::read_node_voltage(x, e.from)
                     - stamping::read_node_voltage(x, e.to);
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
    const std::function<void(Size, Real, const Vector&)>& tick_fn = {}) {
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
        (nl_refresh || !pool.requires_static_refresh())
            ? nl_refresh
            : pwl::make_combined_diode_mosfet_refresh();

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
    const Real h_min = opts.h_min > Real{0} ? opts.h_min
                                             : span * Real{1e-11};
    // Probe steps are a DIFFERENT quantity from the step floor.
    // A probe is a throwaway solve whose only job is to pin (v, i)
    // and ask the diodes which way they conduct — nothing is
    // committed, so it wants to advance as little time as
    // possible. The accepted-step floor wants the opposite (a
    // committed step at 2C/h = 1e11 S is meaningless). Conflating
    // them cost 1.2 mV and 100x the chatter on a buck when the
    // step floor was raised.
    const Real h_probe_floor = std::min(h_min, span * Real{1e-12});
    const Real gamma = pwl::HistoryState::trbdf2_gamma();
    const Real c1    = pwl::HistoryState::trbdf2_c1();
    const Real c2    = pwl::HistoryState::trbdf2_c2();
    const Real c3    = pwl::HistoryState::trbdf2_c3();
    const Real clte  = pwl::HistoryState::trbdf2_lte_const();

    pwl::HistoryState history(graph, pool);
    pwl::DiodeEventState diodes(graph, pool);
    const SwitchStateMask diode_owned = diodes.diode_owned_bits();
    const bool has_diodes = diodes.entries().size() > 0;

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
        history.from_flat(resume_from->history);
        if (!resume_from->diode_on.empty()) {
            diodes.restore_on_bits(resume_from->diode_on);
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
            sources::compute_sine_b_extra(pool, graph, at_t,
                                           be_sine);
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
    pwl::NonlinearCapacitorHistory coss;
    coss.init(graph, pool);
    if (initial_state.has_value()) {
        coss.seed_from_dc_op(x);
    }
    // After the seed, never before: the snapshot is the state.
    if (resume_from != nullptr) coss.from_flat(resume_from->coss_history);
    const bool has_coss = !coss.empty();
    std::vector<pwl::NonlinearCapacitorHistory::Entry> coss_snap;
    std::vector<Real> q_gamma(coss.entries().size(), Real{0});
    auto coss_stage = pwl::CossStage::Trapezoidal;
    Real coss_h = Real{0};
    // Commit a Coss step under the BDF2 rule the stage actually
    // used — never re-derived from C(v)dv/dt, which would not
    // conserve charge.
    auto commit_coss = [&](const Vector& x_end, Real h_full) {
        const Real c1 = Real{2} + std::sqrt(Real{2});
        const Real gam = Real{2} - std::sqrt(Real{2});
        const Real rho = (Real{1} - gam) / gam;
        const Real c2 = -(Real{1} + rho) / (Real{1} - gam);
        const Real c3 = Real{1} / std::sqrt(Real{2});
        Size qi = 0;
        auto snap = coss.snapshot();
        for (auto& e : snap) {
            const Real v =
                stamping::read_node_voltage(x_end, e.from)
                - stamping::read_node_voltage(x_end, e.to);
            const Real q =
                models::NonlinearCapacitor::charge(e.params, v);
            e.i_prev = (c1 * q + c2 * q_gamma[qi] + c3 * e.q_prev)
                       / h_full;
            e.v_prev = v;
            e.q_prev = q;
            ++qi;
        }
        coss.restore(snap);
    };

    // v2.0 — the three other stateful devices. Each carries a
    // first-order state whose BDF2 second stage was derived
    // alongside the Coss's; see `trbdf2_stage.hpp`. Their histories
    // are committed only on ACCEPT, exactly like the Coss, so a
    // rejected step never touches them and no rollback is needed.
    pwl::LauritzenDiodeHistory laur;
    laur.init(graph, pool);
    if (initial_state.has_value()) laur.seed_from_dc_op(x);
    if (resume_from != nullptr) laur.from_flat(resume_from->lauritzen_history);
    const bool has_laur = !laur.empty();

    pwl::IgbtTailHistory tail;
    tail.init(graph, pool);
    if (initial_state.has_value()) tail.seed_from_dc_op(x);
    if (resume_from != nullptr) tail.from_flat(resume_from->igbt_tail_history);
    const bool has_tail = !tail.empty();

    pwl::PmsmMnaHistory pmsm;
    pmsm.init(graph, pool);
    if (initial_state.has_value()) pmsm.seed_from_dc_op(x);
    if (resume_from != nullptr) pmsm.from_flat(resume_from->pmsm_history);
    const bool has_pmsm = !pmsm.empty();

    pwl::SaturableInductorHistory sat;
    sat.init(graph, pool);
    if (initial_state.has_value()) sat.seed_from_dc_op(x);
    if (resume_from != nullptr) sat.from_flat(resume_from->saturable_history);
    const bool has_sat = !sat.empty();

    // One stage/step pair drives all four: they are always stamped
    // in the same stage of the same step.
    auto dev_stage = pwl::TrBdf2Stage::Trapezoidal;
    Real dev_h = Real{0};

    pwl::NonlinearRefreshFn nl_effective = nl_static;
    if (has_coss) {
        nl_effective =
            [user = nl_static, &coss, &q_gamma, &coss_stage,
             &coss_h](const Vector& xx, sparse::Matrix& J_nl,
                        Vector& f_nl, const topology::Graph& g,
                        const pwl::DevicePool& pp) -> Real {
                Real m = Real{0};
                if (user) {
                    m = user(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(
                    m, pwl::refresh_nonlinear_capacitors(
                           xx, J_nl, f_nl, coss, coss_h,
                           coss_stage, &q_gamma));
            };
    }

    if (has_laur) {
        nl_effective =
            [user = nl_effective, &laur, &dev_stage, &dev_h](
                const Vector& xx, sparse::Matrix& J_nl, Vector& f_nl,
                const topology::Graph& g,
                const pwl::DevicePool& pp) -> Real {
                Real m = Real{0};
                if (user) {
                    m = user(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(m, pwl::refresh_lauritzen_diodes(
                                       xx, J_nl, f_nl, laur, dev_h,
                                       dev_stage));
            };
    }
    if (has_tail) {
        nl_effective =
            [user = nl_effective, &tail, &dev_stage, &dev_h](
                const Vector& xx, sparse::Matrix& J_nl, Vector& f_nl,
                const topology::Graph& g,
                const pwl::DevicePool& pp) -> Real {
                Real m = Real{0};
                if (user) {
                    m = user(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(m, pwl::refresh_igbt_tails(
                                       xx, J_nl, f_nl, tail, dev_h,
                                       dev_stage));
            };
    }
    if (has_pmsm) {
        nl_effective =
            [user = nl_effective, &pmsm, &dev_stage, &dev_h](
                const Vector& xx, sparse::Matrix& J_nl, Vector& f_nl,
                const topology::Graph& g,
                const pwl::DevicePool& pp) -> Real {
                Real m = Real{0};
                if (user) {
                    m = user(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                return std::max(m, pwl::refresh_pmsm_mna(
                                       xx, J_nl, f_nl, pmsm, dev_h,
                                       dev_stage));
            };
    }
    if (has_sat) {
        nl_effective =
            [user = nl_effective, &sat, &dev_stage, &dev_h](
                const Vector& xx, sparse::Matrix& J_nl, Vector& f_nl,
                const topology::Graph& g,
                const pwl::DevicePool& pp) -> Real {
                Real m = Real{0};
                if (user) {
                    m = user(xx, J_nl, f_nl, g, pp);
                } else {
                    J_nl.setZero();
                    f_nl.setZero();
                }
                Real mx = Real{0};
                for (const auto& e : sat.entries()) {
                    mx = std::max(mx, pwl::stamp_saturable_inductor(
                                          e, xx, J_nl, f_nl, dev_h,
                                          dev_stage));
                }
                return std::max(m, mx);
            };
    }

    std::string newton_last_error;
    pwl::PwlSegment nl_seg;
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
            pwl::assemble_segment(graph, pool, m, dt_use,
                                   nl_seg.J, nl_seg.b_constant);
            sparse::compress_in_place(nl_seg.J);
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
                opts.enable_newton_lm, &newton_ws);
        } catch (const std::runtime_error& e) {
            newton_last_error = e.what();
            return false;
        }
        return true;
    };

    auto mask_at = [&](Real at_t) {
        SwitchStateMask m = switch_fn(at_t);
        if (has_diodes) {
            m = combine_masks(m, diodes.current_diode_mask(),
                               diode_owned);
        }
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
        // Lauritzen, IGBT-tail, PMSM and saturable-inductor wrappers
        // read (dev_h, coss_h), which the step loop assigns only
        // after this point — so the zero-time diode settle above the
        // loop, and every probe / landing solve inside it, stamped
        // those devices with h = 0 (the settle) or with the PREVIOUS
        // stage's h (the probes). A saturable inductor or a PMSM in
        // any circuit with a diode divided by zero in the settle and
        // took the process down (SIGSEGV, measured); a Coss produced
        // NaN diode bits in silence, because the settle discards the
        // solve's return value. Setting the pair here, at the one
        // choke point all trapezoidal solves pass through, is the
        // same fix run_transient applies to `refresh_dt`.
        dev_stage = pwl::TrBdf2Stage::Trapezoidal;
        dev_h = dt;
        coss_stage = pwl::CossStage::Trapezoidal;
        coss_h = dt;
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
    auto is_singular_error = [](const std::runtime_error& e) {
        return std::string(e.what()).find("numerically singular")
               != std::string::npos;
    };
    auto probe_solve = [&](const SwitchStateMask& m, Real t0, Real dt,
                            Vector& out_x) -> bool {
        Real dt_try = std::max(dt, h_probe_learned);
        const Real dt_cap = std::max(
            dt_try, std::min(span * Real{1e-6}, h_max * Real{1e-2}));
        for (;;) {
            try {
                const bool ok = trap_solve(m, t0, dt_try, out_x);
                if (dt_try > h_probe_learned && dt_try > dt) {
                    h_probe_learned = dt_try;
                }
                return ok;
            } catch (const std::runtime_error& e) {
                if (!is_singular_error(e) || dt_try >= dt_cap) {
                    throw;
                }
                dt_try = std::min(dt_try * Real{100}, dt_cap);
            }
        }
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
        for (Size it = 0; it < opts.max_event_iterations; ++it) {
            probe_solve(mask_at(t_gates), opts.t_start, h_probe,
                        x_trial);
            x_sig = x_trial;
            if (!diodes.update_from_state(x_trial)) {
                break;
            }
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
    const Real h_chatter = std::max(h_min, span * Real{1e-7});
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
    Real cached_edge_t = opts.t_start - span;
    bool edge_cache_valid = false;
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
    std::vector<Size> tick_k(t_ctrl.size(), 0);
    std::vector<Real> t_next_src(t_ctrl.size(), opts.t_start);
    Real t_next_tick = opts.t_start;
    TrBdf2Stats local_stats;
    TrBdf2Stats& st = stats_out ? *stats_out : local_stats;
    st.ctrl_tick_counts.assign(t_ctrl.size(), 0);

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
    // capped at twice the narrowest pulse any gate bit has shown, so
    // those samples stay half a pulse apart once a pulse is known.
    constexpr int kGateSamples = 4;
    const bool learn_gates = !next_edge_fn;
    const Size n_sw_total = static_cast<Size>(graph.num_switches());
    std::vector<Real> gate_last_toggle(n_sw_total, Real{0});
    std::vector<bool> gate_toggled(n_sw_total, false);
    SwitchStateMask gate_prev = switch_fn(opts.t_start
                                          + Real{0.5} * h_min);
    Real gate_pulse_min = std::numeric_limits<Real>::infinity();
    // A "pulse" shorter than this is a floating-point glitch of the
    // user's phase comparison, not a pulse a step could resolve.
    const Real gate_pulse_floor =
        std::max(Real{16} * h_min, span * Real{1e-9});
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
    auto gates_differ = [&](const SwitchStateMask& a,
                            const SwitchStateMask& b) {
        for (const Size i : gate_bits) {
            if (a.get(i) != b.get(i)) {
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
                if (w > gate_pulse_floor && w < gate_pulse_min) {
                    gate_pulse_min = w;
                    // 1 % margin: a pulse as wide as the spacing
                    // cannot fit between two samples, and the widths
                    // carry the bisection's h_min of noise.
                    if (gate_spacing_open[i] > Real{1.01} * w) {
                        ++st.n_gate_undersampled;
                    }
                }
                gate_spacing_open[i] = Real{0};
            }
            gate_toggled[i] = true;
            gate_last_toggle[i] = at_t;
        }
        gate_prev = sw;
    };
    // Diode bits found held in a forbidden state at a step's start
    // (see the crossing detector).
    Real held_redecide_t = opts.t_start - span;

    try {
        while (t < opts.t_end - h_min && steps < opts.max_steps) {
            if (should_continue && !should_continue()) {
                break;   // partial trace preserved
            }
            ++steps;

            // ---- controller tick: fire AT the scheduled instant,
            //      before this step's mask is sampled ----
            if (has_ticks && t >= t_next_tick - h_min) {
                // Every due source, in source order.
                for (std::size_t j = 0; j < t_ctrl.size(); ++j) {
                    if (t < t_next_src[j] - h_min) {
                        continue;
                    }
                    if (multi_rate) {
                        tick_fn(j, t, x);
                    } else {
                        observer_fn(t, x);
                    }
                    ++st.n_ctrl_ticks;
                    ++st.ctrl_tick_counts[j];
                    ++tick_k[j];
                    t_next_src[j] = opts.t_start
                        + static_cast<Real>(tick_k[j]) * t_ctrl[j];
                }
                t_next_tick = *std::min_element(t_next_src.begin(),
                                                t_next_src.end());
                // A controller just moved its duty, so switch_fn
                // has a DIFFERENT shape from here on and any
                // bisected edge time is stale. Closed-loop runs
                // missed a quarter of their PWM edges (307 of 400
                // over 200 periods) before this.
                edge_cache_valid = false;
            }

            // Shortly after the chatter guard gave up, the LTE may not
            // walk h below h_chatter: under all-OFF bridge bits a 1 mH
            // line and g_off make a 1 ps mode whose kink the LTE reads
            // as error at any h, and h sat at h_min until max_steps.
            const Real h_lo =
                (t - chatter_exhaust_t <= Real{1e3} * h_chatter)
                    ? h_chatter : h_min;
            h = std::clamp(h, std::min(h_lo, opts.t_end - t),
                           std::min(h_max, opts.t_end - t));
            // The gates as the step starts (one switch_fn call), and
            // the step cap the narrowest pulse seen so far implies.
            SwitchStateMask sw_now = switch_fn(t + Real{0.5} * h_min);
            if (learn_gates) {
                learn_gate_toggles(sw_now, t);
                if (std::isfinite(gate_pulse_min)) {
                    h = std::min(h, std::max(Real{2} * gate_pulse_min,
                                             h_min));
                }
            }
            if (has_diodes && gates_differ(sw_now, gates_decided)) {
                // An edge at this step boundary that no step landed
                // (see gates_decided): settle the diodes under the
                // gates the step will use, as a landed edge does.
                for (Size it = 0; it < opts.max_event_iterations;
                     ++it) {
                    gates_decided = mask_at(t + Real{0.5} * h_min);
                    probe_solve(gates_decided, t, h_probe_floor,
                                x_trial);
                    x_sig = x_trial;
                    if (!diodes.update_from_state(x_trial)) {
                        break;
                    }
                }
                ++st.n_gate_boundary_settles;
            }
            // Land the next tick exactly, like a gate edge.
            if (has_ticks && t_next_tick > t
                && t_next_tick - t < h) {
                h = std::max(t_next_tick - t, h_min);
            }
            // Land the next current-source corner (pulse edge, table
            // knot) exactly. shortcut: a local clamp; remove when
            // update-trbdf2-error-control owns source breakpoints.
            if (has_iwave) {
                const Real t_bp =
                    sources::next_current_waveform_breakpoint(pool, graph, t);
                if (t_bp - t > h_min && t_bp - t < h) {
                    h = t_bp - t;
                }
            }

            // ---- gate-edge landing ----
            bool landed_on_edge = false;
            if (next_edge_fn) {
                if (!edge_cache_valid || cached_edge_t <= t) {
                    cached_edge_t = next_edge_fn(t);
                    edge_cache_valid = true;
                }
            }
            if (edge_cache_valid && cached_edge_t > t
                && cached_edge_t <= t + h) {
                h = std::max(cached_edge_t - t, h_min);
                landed_on_edge = true;
            } else if (edge_cache_valid && cached_edge_t <= t) {
                edge_cache_valid = false;
            }
            // True when every switch_fn sample over (t, t + h] equals
            // sw_now, so the step's gate mask IS sw_now (the midpoint
            // was one of the samples).
            bool gates_flat = false;
            if (!landed_on_edge) {
                if (learn_gates) {
                    const Real spacing =
                        h / static_cast<Real>(kGateSamples);
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
                std::array<Real, kGateSamples + 1> frac{};
                for (int k = 1; k <= kGateSamples; ++k) {
                    frac[static_cast<std::size_t>(k - 1)] =
                        static_cast<Real>(k)
                        / static_cast<Real>(kGateSamples);
                }
                frac[kGateSamples] =
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
                    for (int it = 0; it < 60 && (hi - lo) > h_min;
                         ++it) {
                        const Real mid = Real{0.5} * (lo + hi);
                        if (switch_fn(mid) == sw_now) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    h = std::max(hi - t, h_min);
                    landed_on_edge = true;
                    cached_edge_t = hi;
                    edge_cache_valid = true;
                }
            }

            // Step mask: sampled at the midpoint — piecewise-
            // constant gates are constant over (t, t+h) by the
            // clamp above, so the midpoint is unambiguous. When the
            // samples above were all equal it was one of them.
            SwitchStateMask mask_step =
                gates_flat ? sw_now : switch_fn(t + Real{0.5} * h);
            if (has_diodes) {
                mask_step = combine_masks(
                    mask_step, diodes.current_diode_mask(),
                    diode_owned);
            }

            // ---- stage 1: TR over γh ----
            const Real hg = gamma * h;
            history.compute_b_extra(hg, b_stage);
            accumulate_sources(t + hg, b_src);
            b_stage += b_src;
            x_g = x;                 // Newton warm start
            coss_stage = pwl::CossStage::Trapezoidal;
            coss_h = hg;
            dev_stage = pwl::TrBdf2Stage::Trapezoidal;
            dev_h = hg;
            bool stage_ok =
                solve_dispatch(mask_step, hg, b_stage, x_g);
            ++st.n_solves;

            // ---- stage 2: BDF2 over the remainder, SAME factor ----
            history.compute_b_extra_trbdf2_stage2(h, x_g, b_stage);
            accumulate_sources(t + h, b_src);
            b_stage += b_src;
            if (stage_ok) {
                if (has_coss) {
                    // Q at the stage point feeds the BDF2 history
                    // term; h is the FULL step there, while the
                    // matrix is still the one factored at gamma*h.
                    Size qi = 0;
                    for (const auto& e : coss.entries()) {
                        const Real vg =
                            stamping::read_node_voltage(x_g, e.from)
                            - stamping::read_node_voltage(x_g, e.to);
                        q_gamma[qi++] =
                            models::NonlinearCapacitor::charge(
                                e.params, vg);
                    }
                    coss_stage = pwl::CossStage::Bdf2Stage2;
                    coss_h = h;
                }
                if (has_laur || has_tail || has_pmsm
                    || has_sat) {
                    // Each device's state AT THE STAGE POINT feeds
                    // the BDF2 history term. `h` is the FULL step
                    // there, while the matrix is still the one
                    // factored at gamma*h — the identity
                    // c1/h == 2/(gamma*h) is what allows that.
                    if (has_laur) laur.capture_gamma(x_g);
                    if (has_tail) tail.capture_gamma(x_g);
                    if (has_pmsm) pmsm.capture_gamma(x_g);
                    if (has_sat) sat.capture_gamma(x_g);
                    dev_stage = pwl::TrBdf2Stage::Bdf2Stage2;
                    dev_h = h;
                }
                x_1 = x_g;           // Newton warm start
                stage_ok =
                    solve_dispatch(mask_step, hg, b_stage, x_1);
                ++st.n_solves;
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
                        + " If t is just after a hard switching edge, "
                          "this is the known TR-BDF2 stall at h_min "
                          "(CHANGELOG 2.0.0, \"TR-BDF2 stalls at "
                          "h_min\"; KNOWN_LIMITATIONS.md): a "
                          "step-controller defect, not a tolerance "
                          "one. Run this circuit on engine='pwl' with "
                          "an explicit dt.");
                }
                h = std::max(h * Real{0.25}, h_min);
                continue;
            }

            // ---- LTE over the differential variables ----
            Real err_sq = Real{0};
            Size n_err = 0;
            for (Size i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                const Real vg1 = entry_v(e, x_g);
                const Real v11 = entry_v(e, x_1);
                Real f_n, f_g, f_1, y_old, y_new;
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
                const Real lte =
                    clte * h
                    * (f_n / gamma
                       - f_g / (gamma * (Real{1} - gamma))
                       + f_1 / (Real{1} - gamma));
                const Real sc =
                    opts.atol
                    + opts.rtol
                          * std::max(std::abs(y_old),
                                     std::abs(y_new));
                const Real w = lte / sc;
                err_sq += w * w;
                ++n_err;
            }
            const Real err =
                n_err > 0 ? std::sqrt(err_sq
                                       / static_cast<Real>(n_err))
                           : Real{0};

            if (!std::isfinite(err)
                || !x_1.allFinite()) {
                h = std::max(h * Real{0.1}, h_min);
                ++st.n_reject;
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
                    Real{0.9} * std::pow(err, Real{-1.0 / 3.0}),
                    Real{0.1}, Real{0.9});
                h = std::max(h * fac, h_lo);
                ++st.n_reject;
                continue;
            }

            // ---- diode crossing detection on the accepted step ----
            bool commutated = false;
            if (has_diodes) {
                Real t_star = opts.t_end + span;   // earliest
                Size cross_idx = Size(-1);
                bool held_found = false;
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
                                     std::abs(s1)}) > tol_d;
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
                                   > tol_d) {
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
                        tb = t + hg;
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
                            for (Size it2 = 0;
                                 it2 < opts.max_event_iterations;
                                 ++it2) {
                                // Probe, not a committed step.
                                gates_decided = mask_at(t);
                                probe_solve(gates_decided, t,
                                            h_probe_floor,
                                            x_trial);
                                x_sig = x_trial;
                                if (!diodes.update_from_state(
                                        x_trial)) {
                                    break;
                                }
                            }
                            result.commutation_events.push_back(
                                {t,
                                 diodes.entries()[cross_idx]
                                     .branch_id,
                                 diodes.entries()[cross_idx]
                                     .is_on});
                            ++st.n_diode_events;
                            commutated = true;
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
                                if (diodes.update_from_state(x_1)) {
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
                            tb = t + hg;
                            sb = sg;
                        }
                    }
                    Real t_loc = t_star;
                    int side = 0;
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
                            trap_solve(mask_step, t, dts, x_trial);
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
                            if (side == -1) { sb *= Real{0.5}; }
                            side = -1;
                        } else {
                            tb = t_loc;
                            sb = sm;
                            if (side == +1) { sa *= Real{0.5}; }
                            side = +1;
                        }
                        if ((tb - ta) < std::max(h_min,
                                                  Real{1e-4} * h)) {
                            break;
                        }
                    }
                    // Land the step at t* = tb (the post-crossing
                    // side, so decide_next_state sees the crossed
                    // signal), commit the trap sub-step there.
                    const Real dt_land =
                        std::max(tb - t, dt_floor);
                    trap_solve(mask_step, t, dt_land, x_trial);
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
                    if (has_coss) {
                        // Trapezoidal commit (the landing is a single
                        // trap step); commit_coss is the BDF2 form.
                        coss.update_from_state(x_trial, dt_land);
                    }
                    if (has_laur) {
                        laur.update_from_state(
                            x_trial, dt_land, pwl::TrBdf2Stage::Trapezoidal);
                    }
                    if (has_tail) {
                        tail.update_from_state(
                            x_trial, dt_land, pwl::TrBdf2Stage::Trapezoidal);
                    }
                    if (has_pmsm) {
                        pmsm.update_from_state(
                            x_trial, dt_land, pwl::TrBdf2Stage::Trapezoidal);
                    }
                    if (has_sat) {
                        sat.update_from_state(
                            x_trial, pwl::TrBdf2Stage::Trapezoidal);
                    }
                    x = x_trial;
                    x_sig = x;
                    t += dt_land;
                    const bool flipped =
                        diodes.update_from_state(x);
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
                            probe_solve(gates_decided, t, h_zero,
                                        x_trial);
                            x_sig = x_trial;
                            if (!diodes.update_from_state(
                                    x_trial)) {
                                break;
                            }
                        }
                    }
                    result.commutation_events.push_back(
                        {t,
                         diodes.entries()[cross_idx].branch_id,
                         diodes.entries()[cross_idx].is_on});
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
                    x_mask = mask_step;
                    x_mask_known = true;
                    gates_decided = mask_step;
                    ++st.n_diode_events;
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
                        && t >= cached_edge_t - h_min) {
                        ++st.n_gate_events;
                    }
                    commutated = true;
                    }
                }
            }

            if (!commutated) {
                // ---- accept ----
                history.commit_trbdf2(x_1, h, x_g);
                if (has_coss) {
                    commit_coss(x_1, h);
                }
                // Committed under the rule the stage actually used,
                // so the derivative each history carries into the
                // next step is the one the method produced.
                if (has_laur) {
                    laur.update_from_state(
                        x_1, h, pwl::TrBdf2Stage::Bdf2Stage2);
                }
                if (has_tail) {
                    tail.update_from_state(
                        x_1, h, pwl::TrBdf2Stage::Bdf2Stage2);
                }
                if (has_pmsm) {
                    pmsm.update_from_state(
                        x_1, h, pwl::TrBdf2Stage::Bdf2Stage2);
                }
                if (has_sat) {
                    // The stage matters now: the hysteretic law's
                    // commit must integrate from the same base the
                    // converged BDF2 residual used, (H_γ, M_γ).
                    sat.update_from_state(x_1, pwl::TrBdf2Stage::Bdf2Stage2);
                }
                x = x_1;
                x_sig = x;
                t += h;
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(0);
                // Both TR-BDF2 stages solved under mask_step.
                result.record_switch_state(mask_step);
                x_mask = mask_step;
                x_mask_known = true;
                gates_decided = mask_step;
                ++st.n_accept;
                if (pending_gate_slot >= 0) {
                    // The h that survived right after this gate
                    // corner — next cycle's same-parity edge
                    // restarts here instead of shrinking through
                    // rejects. Only a CONTROLLER-chosen h is worth
                    // remembering: an edge-clamped step's h is the
                    // accidental distance to the next edge.
                    if (!landed_on_edge) {
                        h_mem_gate[pending_gate_slot] = h;
                    }
                    pending_gate_slot = -1;
                }
                if (landed_on_edge) {
                    ++st.n_gate_events;
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
                    if (has_diodes) {
                        const Real h_probe =
                            std::max(h_probe_floor, h * Real{1e-6});
                        for (Size it = 0;
                             it < opts.max_event_iterations;
                             ++it) {
                            gates_decided = mask_at(t + h_probe);
                            probe_solve(gates_decided, t,
                                        h_probe, x_trial);
                            x_sig = x_trial;
                            if (!diodes.update_from_state(
                                    x_trial)) {
                                break;
                            }
                        }
                    }
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
                            * std::pow(e_used,
                                        Real{-1.0 / 3.0}),
                        Real{0.2}, Real{5.0});
                    h = h * fac;
                }
            }
        }
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
    } catch (const SimulationAborted&) {
        throw;
    } catch (const std::runtime_error& e) {
        throw SimulationAborted(e.what(), std::move(result), t);
    }

    // The complete state at the end, so either engine can continue
    // from it (this used to be left EMPTY, with `valid = false`, and
    // a fixed-engine resume from it silently started at zero).
    result.final_snapshot.t = t;
    result.final_snapshot.x = x;
    result.final_snapshot.history = history.to_flat();
    result.final_snapshot.diode_on = diodes.num_diodes() > 0
        ? diodes.snapshot_on_bits() : std::vector<bool>{};
    // The mask x was solved under (the record's last row): diode_on
    // may already be post-edge (see initial_mask above).
    result.final_snapshot.stamped_mask =
        x_mask_known ? mask_bits(x_mask) : std::vector<bool>{};
    result.final_snapshot.saturable_history = sat.to_flat();
    result.final_snapshot.coss_history = coss.to_flat();
    result.final_snapshot.lauritzen_history = laur.to_flat();
    result.final_snapshot.igbt_tail_history = tail.to_flat();
    result.final_snapshot.pmsm_history = pmsm.to_flat();
    result.final_snapshot.valid = true;
    return result;
}

}  // namespace pulsim::solver
