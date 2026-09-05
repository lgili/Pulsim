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
//     per-direction signal (ON→OFF watches v_D, whose zero is the
//     current zero; OFF→ON watches v_D − V_th) is checked at the
//     stage and end points. A crossing is localized by Illinois
//     (regula falsi with endpoint halving) on trial TRAP solves at
//     dt* = t* − t, the step is committed at t*, the diode flips
//     there (`SwitchedDiode::decide_next_state`), and the
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
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
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
    /// ceiling: an on-off gate pulse narrower than h_max can be
    /// missed entirely, exactly like dsed's dt_max. 0 = auto
    /// (span / 1000).
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
    /// Newton settings, used only when the circuit has nonlinear
    /// devices. Same knobs and defaults as the fixed engine.
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
    /// h_min (typically the sliver between two events). Zero on a
    /// healthy run; the Python layer warns when it is not.
    Size n_forced_accepts = 0;
    /// Controller ticks fired (should be exactly
    /// floor(span / T_ctrl) + 1 — the fixed engine's throttled
    /// observer DRIFTS and loses ticks: measured 198 of 200 on a
    /// 10 kHz loop at dt = 2 µs).
    Size n_ctrl_ticks = 0;
    /// Steps retaken at a smaller h because Newton did not
    /// converge — the variable-step engine's answer to a hard
    /// nonlinear step, where the fixed one can only abort.
    Size n_newton_retries = 0;
};

namespace detail_trbdf2 {

/// Per-direction diode crossing signal at a solved state. The
/// existing post-hoc watcher in run_transient uses v_D − V_th for
/// BOTH directions, which biases turn-OFF by V_th (its own doc
/// comment admits i_D is the right signal); with i_D = g·v_D the
/// ON→OFF zero is v_D = 0, so this watches v_D for ON and
/// v_D − V_th for OFF.
inline Real crossing_signal(const pwl::DiodeEventState& diodes,
                             Size entry_idx, const Vector& x) {
    const auto& e = diodes.entries()[entry_idx];
    const Real v_d = stamping::read_node_voltage(x, e.from)
                     - stamping::read_node_voltage(x, e.to);
    return e.is_on ? v_d : (v_d - e.V_th);
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
    const pwl::NonlinearRefreshFn& nl_refresh = {}) {
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
    bool has_nonlinear = false;
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (graph.branch(b_id).kind
            == topology::BranchKind::Nonlinear) {
            has_nonlinear = true;
            break;
        }
    }
    // A charge-based Coss brings its own refresh (wired below), so
    // it does not need one from the caller; every other nonlinear
    // device does.
    bool has_only_coss = has_nonlinear;
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (graph.branch(b_id).kind
                == topology::BranchKind::Nonlinear
            && pool.kind_of(b_id)
                != pwl::DevicePool::StoredKind::NonlinearCapacitor) {
            has_only_coss = false;
            break;
        }
    }
    if (has_nonlinear && !nl_refresh && !has_only_coss) {
        throw std::invalid_argument(
            "run_transient_trbdf2: the circuit has nonlinear "
            "devices but no nonlinear-refresh callback was "
            "supplied — the stages would solve the linear "
            "skeleton and silently return the wrong answer.");
    }

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
    const bool has_laur = !laur.empty();

    pwl::IgbtTailHistory tail;
    tail.init(graph, pool);
    if (initial_state.has_value()) tail.seed_from_dc_op(x);
    const bool has_tail = !tail.empty();

    pwl::PmsmMnaHistory pmsm;
    pmsm.init(graph, pool);
    if (initial_state.has_value()) pmsm.seed_from_dc_op(x);
    const bool has_pmsm = !pmsm.empty();

    pwl::SaturableInductorHistory sat;
    sat.init(graph, pool);
    if (initial_state.has_value()) sat.seed_from_dc_op(x);
    const bool has_sat = !sat.empty();

    // One stage/step pair drives all four: they are always stamped
    // in the same stage of the same step.
    auto dev_stage = pwl::TrBdf2Stage::Trapezoidal;
    Real dev_h = Real{0};

    pwl::NonlinearRefreshFn nl_effective = nl_refresh;
    if (has_coss) {
        nl_effective =
            [user = nl_refresh, &coss, &q_gamma, &coss_stage,
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
                opts.enable_newton_lm);
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

    // ---- consistent initial diode bits (zero-time settle) ----
    // A tiny trap step pins every capacitor to its (v, i) and asks
    // the network which diodes conduct; iterate to a fixed point,
    // exactly run_transient's per-step event iteration at t_start.
    if (has_diodes) {
        const Real h_probe = h_probe_floor;
        for (Size it = 0; it < opts.max_event_iterations; ++it) {
            trap_solve(mask_at(opts.t_start), opts.t_start, h_probe,
                        x_trial);
            if (!diodes.update_from_state(x_trial)) {
                break;
            }
        }
    }

    SimulationResult result;
    result.times.reserve(4096);
    result.times.push_back(opts.t_start);
    result.states.push_back(x);
    result.event_iteration_count.push_back(0);

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
    const Real t_ctrl = opts.observer_period;
    const bool has_ticks = observer_fn && t_ctrl > Real{0};
    Size tick_k = 0;
    Real t_next_tick = opts.t_start;
    TrBdf2Stats local_stats;
    TrBdf2Stats& st = stats_out ? *stats_out : local_stats;

    try {
        while (t < opts.t_end - h_min && steps < opts.max_steps) {
            if (should_continue && !should_continue()) {
                break;   // partial trace preserved
            }
            ++steps;

            // ---- controller tick: fire AT the scheduled instant,
            //      before this step's mask is sampled ----
            if (has_ticks && t >= t_next_tick - h_min) {
                observer_fn(t, x);
                ++st.n_ctrl_ticks;
                ++tick_k;
                t_next_tick = opts.t_start
                              + static_cast<Real>(tick_k) * t_ctrl;
                // A controller just moved its duty, so switch_fn
                // has a DIFFERENT shape from here on and any
                // bisected edge time is stale. Closed-loop runs
                // missed a quarter of their PWM edges (307 of 400
                // over 200 periods) before this.
                edge_cache_valid = false;
            }

            h = std::clamp(h, h_min, std::min(h_max, opts.t_end - t));
            // Land the next tick exactly, like a gate edge.
            if (has_ticks && t_next_tick > t
                && t_next_tick - t < h) {
                h = std::max(t_next_tick - t, h_min);
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
            if (!landed_on_edge) {
                const SwitchStateMask sw_now =
                    switch_fn(t + Real{0.5} * h_min);
                if (!(switch_fn(t + h) == sw_now)) {
                    Real lo = t, hi = t + h;
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
            // clamp above, so the midpoint is unambiguous.
            SwitchStateMask mask_step = switch_fn(t + Real{0.5} * h);
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
                        + newton_last_error);
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

            if (err > Real{1} && h <= h_min * Real{1.0001}) {
                // Cannot shrink further: accept, but CONFESS —
                // silence here would be an unbounded-error hole.
                ++st.n_forced_accepts;
            }
            if (err > Real{1} && h > h_min * Real{1.0001}) {
                // reject: elementary controller, order-3 exponent
                const Real fac = std::clamp(
                    Real{0.9} * std::pow(err, Real{-1.0 / 3.0}),
                    Real{0.1}, Real{0.9});
                h = std::max(h * fac, h_min);
                ++st.n_reject;
                continue;
            }

            // ---- diode crossing detection on the accepted step ----
            bool commutated = false;
            if (has_diodes) {
                Real t_star = opts.t_end + span;   // earliest
                Size cross_idx = Size(-1);
                for (Size d = 0; d < diodes.entries().size(); ++d) {
                    const Real s0 = detail_trbdf2::crossing_signal(
                        diodes, d, x);
                    const Real sg = detail_trbdf2::crossing_signal(
                        diodes, d, x_g);
                    const Real s1 = detail_trbdf2::crossing_signal(
                        diodes, d, x_1);
                    // A crossing exists if the signal changes sign
                    // anywhere among (t, t+γh, t+h).
                    const bool crossed =
                        (s0 > Real{0}) != (s1 > Real{0})
                        || (s0 > Real{0}) != (sg > Real{0});
                    if (!crossed) {
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
                        if (chatter_n <= Size{3}) {
                            for (Size it2 = 0;
                                 it2 < opts.max_event_iterations;
                                 ++it2) {
                                // Probe, not a committed step.
                                trap_solve(mask_at(t), t,
                                            h_probe_floor,
                                            x_trial);
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
                            ++st.n_chatter_breaks;
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
                        diodes, cross_idx, x);
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
                        trap_solve(mask_step, t, dts, x_trial);
                        if (!x_trial.allFinite()) {
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
                    x = x_trial;
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
                            trap_solve(mask_at(t), t, h_zero,
                                        x_trial);
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
                    // No stage argument: the commit reads the
                    // converged x and evaluates lambda(i) from the
                    // model's own law, which is stage-independent.
                    sat.update_from_state(x_1);
                }
                x = x_1;
                t += h;
                result.times.push_back(t);
                result.states.push_back(x);
                result.event_iteration_count.push_back(0);
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
                            trap_solve(mask_at(t + h_probe), t,
                                        h_probe, x_trial);
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
    } catch (const SimulationAborted&) {
        throw;
    } catch (const std::runtime_error& e) {
        throw SimulationAborted(e.what(), std::move(result), t);
    }

    return result;
}

}  // namespace pulsim::solver
