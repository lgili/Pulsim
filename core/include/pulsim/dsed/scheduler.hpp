#pragma once

// =============================================================================
// Pulsim — PED engine: outer scheduler loop.
// =============================================================================
//
// Path-Based Event-Driven simulator main loop. Direct port of the
// Gate 1+3 Python prototype (`prototype/dsed/scheduler.py`);
// Gate 2 validated to 0.0057 % RMSE + 0.46× wall-clock vs fixed-step
// trapezoidal on buck CCM; Gate 3 extends with body-diode commutation
// (DCM) — validated to 0.0043 % V_out error vs the Erickson/Maksimovic
// analytical DCM regulator equation (`prototype/dsed/run_buck_dcm_validation.py`).
//
// Pseudocode (one iteration):
//
//   1. compute next *gate* edge t_gate (analytical, from switch_fn)
//   2. propose dt from PI controller, capped by (t_gate - t)
//   3. step integrator (DOPRI5)
//   4. if step rejected: shrink dt, retry (FSAL invalidated)
//   5. POST-STEP predicate scan over (t, t+dt) for diode/ZCD events
//      (cubic Hermite from FSAL's pre-/post-step derivatives; Illinois
//      on the resulting g(τ))
//   6. if a predicate fired strictly inside the step: backtrack to
//      t_event, project state (e.g. clamp i_L = 0 for ZCD),
//      fire event, reset FSAL+PI
//   7. elif step landed on t_gate: fire gate event, reset FSAL+PI
//   8. record state
//
// The `System` template parameter must expose:
//
//   - ``Vector rhs(Real t, const Vector& x) const`` — RHS for current mask
//   - ``void set_mask(MaskT mask)`` — swap to new mask
//   - ``MaskT current_mask() const`` — read current mask
//
// The `SwitchFn` callable must satisfy ``MaskT(Real t)``. Optionally:
//
//   - ``Real next_edge_after(Real t) const`` — analytical gate-edge time
//     (enables the fast path; required for Gate 1 gate-only operation).
//   - ``void register_zcd_transition(Real t)`` — receives notification
//     when a non-gate predicate (ZCD, diode commutation) fires, so the
//     switch_fn can latch the post-event mode for the remainder of the
//     switching cycle. Used by `BuckDCMSwitchFn` in Gate 3.
//
// The `StateProjection` callable (optional, defaults to identity) has
// signature ``Vector(Real t, const Vector& x, PredicateType ptype)`` and
// is invoked at event firings to clamp algebraic constraints — e.g.
// for ZCD, project ``i_L = 0`` so the inductor sub-system is consistently
// rank-reduced in the ZERO_CURRENT mode. This is the seed of the Gate 4
// linear-solve hook for partial_refactor projection.

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <cstdio>
#include <utility>
#include <vector>

#include "pulsim/errors/convergence.hpp"
#include "pulsim/dsed/event_predictor.hpp"
#include "pulsim/events/mode_refusal.hpp"
#include "pulsim/events/mode_selector.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/dsed/event_projection.hpp"
#include "pulsim/dsed/exact_lti.hpp"
#include "pulsim/dsed/rk45_dormand_prince.hpp"
#include "pulsim/dsed/step_controller.hpp"
#include "pulsim/dsed/time_eps.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// One row in the simulation's event log.
template <class MaskT>
struct EventRecord {
    Real t;
    std::string name;
    PredicateType type;
    MaskT old_mask;
    MaskT new_mask;
};

/// add-complementarity-mode-selection D12 stage 1 — one event where the
/// shadow LCP and today's settle chose different diode modes (diode
/// census order; 1 = conducting).
struct ModeShadowRecord {
    Real t;
    std::vector<char> settle_mode;
    std::vector<char> lcp_mode;
    std::vector<char> previous_mode;
    int settle_rounds;
    int pivots;
    int ties;
    events::LcpStatus status;
    std::vector<Real> lcp_current;   // port currents of the LCP answer (A)
    std::vector<Real> lcp_gap;       // port gaps w (V)
    // Judge (independent of both deciders): the mode's own exact
    // trajectory from the held state over δ = 100·h_e and 1e4·h_e;
    // consistent iff every ON diode keeps i >= -tol and every OFF diode
    // v <= max(V_th, V_f) + tol. -1: no exact step for that mask.
    int settle_consistent = -1;
    int lcp_consistent = -1;
    Real settle_violation = Real{0};   // worst violation, scaled (V)
    Real lcp_violation = Real{0};
};

/// Shadow counters (task 8.1 names) and the disagreements.
struct ModeShadowReport {
    bool enabled = false;
    Real h_e = Real{0};
    std::size_t lcp_events = 0;      // events the shadow posed an LCP at
    std::size_t lcps = 0;            // LCPs solved (with widenings)
    std::size_t pivots_total = 0;
    int max_pivots_per_event = 0;
    std::size_t ties = 0;
    std::size_t widenings = 0;
    std::size_t leak_pins = 0;       // D5: leak-scale currents kept ON (island pins)
    std::size_t pin_judgements = 0;  // D5: pin candidates sent to the exact-trajectory judge
    std::size_t unfixed = 0;         // D5: decided ports freed again in a widened pass
    std::size_t capped = 0;          // widenings shortened to the mode's next event (DECISIONS #24)
    std::size_t enumerations = 0;    // D3: non-passive ray, modes enumerated
    std::size_t nonunique_modes = 0; // D3: enumerations with several modes
    std::size_t released = 0;        // hysteresis ports released (D8)
    std::size_t settle_rounds_total = 0;
    std::size_t errors = 0;          // port model or LCP failures (recorded)
    std::string last_error;
    std::vector<ModeShadowRecord> disagreements;
    bool decides = false;            // mode_selection='lcp': the LCP decided every event
};

/// The default-mode guard (slice 2, task 4): after each settle, the
/// settled mode is checked on its own exact trajectory (the shadow's
/// judge); a mode that violates its own law is replaced by the LCP's.
struct ModeGuardReport {
    bool enabled = false;
    std::size_t settle_caps = 0;     // settles that hit the round cap, decided by the LCP
    std::size_t checks = 0;          // settles judged
    std::size_t unchecked = 0;       // no exact step for the settled mask
    std::size_t repairs = 0;         // settled mode inconsistent, LCP mode applied
    std::size_t confirmed = 0;       // inconsistent, but the LCP returned the same mode
    std::vector<Real> repair_times;
    std::vector<std::string> repair_diodes;   // the diode whose law the settle violated
    std::vector<Real> repair_violations;      // its violation, scaled (V)
    ModeShadowReport lcp;            // the repairs' LCP counters
};

/// PED simulation output.
template <class MaskT>
struct PEDResult {
    std::vector<Real> times;
    std::vector<Vector> states;
    // v2.0 Phase 3 item 5 — the mask each sample was recorded
    // under. What lets a post-pass reconstruct the FULL MNA vector
    // (node voltages, source and inductor currents) from the
    // reduced states through the per-mask recovery map — including
    // solver-owned diode bits no timeline replay could recover.
    std::vector<MaskT> sample_masks;
    // The STAMPED-MASK record (the convention every kernel engine's
    // SimulationResult::switch_state_bits uses): step_masks[k] is the
    // mask in force over the integration step that ENDED at sample
    // k, i.e. over (t[k-1], t[k]]. It differs from sample_masks[k]
    // only at a sample where an event fired, because sample_masks
    // holds the POST-event mask integration continues in. Row 0 is
    // the start mask. Weighting step_masks right-hold books every
    // step to the mode that was actually integrated over it.
    std::vector<MaskT> step_masks;
    // Sparse companion: at a sample where an event fired, the state
    // at the end of the step BEFORE fire_event_ projected it
    // (pre_event_states[j] belongs to sample pre_event_index[j]).
    // That is the left limit, the state step_masks[k] produced.
    std::vector<std::size_t> pre_event_index;
    std::vector<Vector> pre_event_states;
    std::vector<EventRecord<MaskT>> event_log;
    std::size_t n_accept = 0;
    std::size_t n_reject = 0;
    std::size_t n_events = 0;
    std::size_t n_exact_steps = 0;   // closed-form LTI propagator
    std::size_t n_predicate_evals = 0;  // exact path (D10)
    Real cpu_time_seconds = Real{0};
    ModeShadowReport mode_shadow;      // mode_selection='shadow' / 'lcp'
    ModeGuardReport mode_guard;        // default mode with the guard on
};

/// Append one sample to a PED result, with both mask conventions.
/// `step_mask` is the mask the step that produced the sample was
/// integrated under; `x_pre` (nullable) is that step's end state when
/// an event at this instant has since modified x.
template <class ResultT, class MaskT>
inline void ped_push_sample(ResultT& result, Real t, const Vector& x,
                            const MaskT& post_mask,
                            const MaskT& step_mask,
                            const Vector* x_pre) {
    result.times.push_back(t);
    result.states.push_back(x);
    result.sample_masks.push_back(post_mask);
    result.step_masks.push_back(step_mask);
    if (x_pre != nullptr) {
        result.pre_event_index.push_back(result.times.size() - 1);
        result.pre_event_states.push_back(*x_pre);
    }
}

/// v2.0 Phase 3 — diode commutation support.
///
/// A System that can reconstruct a node-pair voltage from the
/// reduced state (via the algebraic recovery map) enables the
/// scheduler to own PWL diode bits: predicates locate the crossing,
/// `fire_event_` flips the bit directly, and a zero-time cascade
/// settles diodes that the mask change instantaneously
/// forward-biases (a buck's freewheel diode at gate-off). Systems
/// without it — the hand-rolled demo models — compile exactly as
/// before; every diode path below is `if constexpr`-guarded.
/// Whether the System can step the current mode exactly. The
/// adapter encapsulates HOW (plain e^{At} for DC drive, an
/// augmented oscillator system for sine drive — v2.0 Phase 3
/// item 3), so the scheduler only ever asks for "the state h
/// later".
template <class S>
concept HasExactStep = requires(const S& sys, Real t,
                                  const Vector& x, Real h) {
    { sys.has_exact_step() } -> std::convertible_to<bool>;
    { sys.exact_advance_state(t, x, h) }
        -> std::convertible_to<Vector>;
};

/// Whether the System exposes the dense LTI pair (A, b) — the
/// builder adapters do; the hand-rolled demo models need not.
template <class S>
concept HasABInterface = requires(const S& sys, Real t) {
    { sys.A_matrix() } -> std::convertible_to<const DenseMatrix&>;
    { sys.b_vector(t) } -> std::convertible_to<Vector>;
};

template <class S>
concept HasNodePairVoltage = requires(const S& sys, Real t,
                                        const Vector& x) {
    { sys.node_pair_voltage(Index{0}, Index{0}, t, x) }
        -> std::convertible_to<Real>;
};

/// One PWL diode under scheduler ownership. Mirrors the pwl
/// engine's DiodeEntry census so the two engines cannot drift.
struct SchedulerDiode {
    Size switch_idx;
    Index from;               // anode
    Index to;                 // cathode
    Real g_on;
    Real g_off;
    Real V_th;
    Real V_f;                 // forward drop while ON
    std::string label;        // named, Phase-1 style

    [[nodiscard]] models::SwitchedDiode::Params params() const noexcept {
        return {.g_on = g_on, .g_off = g_off, .V_th = V_th, .V_f = V_f};
    }
};

/// Whether ``SwitchFn`` has a ``next_edge_after`` fast path.
template <class SwitchFn>
concept HasNextEdge = requires(const SwitchFn& fn, Real t) {
    { fn.next_edge_after(t) } -> std::convertible_to<Real>;
};

/// Whether ``SwitchFn`` has a ``register_zcd_transition`` callback
/// (for Gate 3 body-diode commutation bookkeeping).
template <class SwitchFn>
concept HasZcdRegister = requires(SwitchFn& fn, Real t) {
    { fn.register_zcd_transition(t) } -> std::same_as<void>;
};

/// Cubic Hermite interpolation over ``(t, t+h)`` using FSAL's
/// pre-/post-step derivatives. DOPRI5's FSAL gives ``k1 = f(t, x)``
/// before the step and ``k7 = f(t+h, x_new)`` after at no extra cost.
/// The Hermite cubic through ``(t, x0, k1)`` and ``(t+h, x1, k7)``
/// is order-4 accurate inside [t, t+h], well within DOPRI5's dense-output
/// budget for predicate root-location.
[[nodiscard]] inline Vector hermite_interp(const Vector& x0,
                                              const Vector& x1,
                                              const Vector& k1,
                                              const Vector& k7,
                                              Real h, Real theta) {
    if (theta <= Real{0}) return x0;
    if (theta >= Real{1}) return x1;
    const Real theta_sq = theta * theta;
    const Real theta_cu = theta_sq * theta;
    const Real h00 = Real{2} * theta_cu - Real{3} * theta_sq + Real{1};
    const Real h10 = theta_cu - Real{2} * theta_sq + theta;
    const Real h01 = Real{-2} * theta_cu + Real{3} * theta_sq;
    const Real h11 = theta_cu - theta_sq;
    return h00 * x0 + (h10 * h) * k1 + h01 * x1 + (h11 * h) * k7;
}

/// Path-Based Event-Driven scheduler (Gate 3 port).
///
/// Templated on `System` (the ODE source — provides rhs + set_mask)
/// and `SwitchFn` (mask schedule — provides operator()(t) and
/// optionally next_edge_after(t), register_zcd_transition(t)).
/// MaskT is deduced from `decltype(switch_fn(0.0))`.
template <class System, class SwitchFn>
class PEDSimulator {
public:
    using MaskT = std::invoke_result_t<SwitchFn, Real>;

    PEDSimulator(System& system,
                  SwitchFn switch_fn,
                  PIController controller = PIController{},
                  EventPredictor predictor = EventPredictor{},
                  Real dt_init = Real{1e-9},
                  Real dt_max = Real{1e-5},
                  std::size_t store_every = 1,
                  StateProjectionFn state_projection = nullptr)
        : system_{system},
          switch_fn_{std::move(switch_fn)},
          controller_{std::move(controller)},
          predictor_{std::move(predictor)},
          dt_init_{dt_init},
          dt_max_{dt_max},
          store_every_{store_every},
          state_projection_{std::move(state_projection)} {}

    /// Run the simulation from ``t = 0`` to ``t = t_end``.
    [[nodiscard]] PEDResult<MaskT> simulate(const Vector& x0, Real t_end) {
        using clock = std::chrono::high_resolution_clock;
        const auto t0_wall = clock::now();

        Vector x = x0;
        Real t = Real{0};
        Real h = dt_init_;
        if constexpr (requires { system_.set_max_step(dt_max_); }) {
            system_.set_max_step(dt_max_);
        }
        MaskT mask = effective_mask_(switch_fn_(t));
        system_.set_mask(mask);
        // A diode may be forward-biased already at t = 0 (a
        // rectifier powered on mid-cycle). Settle before the first
        // step rather than integrating an inconsistent mode.
        RK45State rk_state;
        PEDResult<MaskT> result;
        {
            MaskT m0 = system_.current_mask();
            const MaskT prev0 = diode_bits_;
            const MaskT old0 = m0;
            decide_modes_(t, x, m0, old0, prev0, old0, result);
        }
        ped_push_sample(result, t, x, system_.current_mask(),
                        system_.current_mask(), nullptr);
        std::size_t step_idx = 0;
        // Stamped-mask bookkeeping for the step being taken (see
        // PEDResult::step_masks). Captured after the zero-length
        // gate-edge branch, which fires without storing a sample.
        MaskT step_mask = system_.current_mask();
        bool step_fired = false;
        Vector x_pre_event;

        constexpr std::size_t kMaxSteps = 10'000'000;

        auto f = [this](Real tau, const Vector& xx) -> Vector {
            return system_.rhs(tau, xx);
        };

        while (t < t_end) {
            if (result.n_accept + result.n_reject > kMaxSteps) {
                throw ConvergenceFailure(ConvergenceKind::MaxSteps, t, "",
                    "PEDSimulator: exceeded max-step cap");
            }

            // 1. Next gate-edge time (analytical fast path)
            const Real t_gate = next_gate_edge_(t, t_end);
            const Real t_target = std::min(t_gate, t_end);
            // a source corner is landed on exactly: a single-shot pulse
            // read one ulp early would keep its old piece for a step
            const bool at_corner = t_gate < next_switch_edge_(t, t_end);

            // 2. Propose step capped at next gate
            Real h_use = std::min({h, dt_max_, t_target - t});
            if (h_use <= Real{0}) {
                // Sitting exactly on a gate edge — fire it.
                if (t_gate < t_end) {
                    fire_event_(t_gate, "gate_edge", PredicateType::GateEdge,
                                  x, rk_state, result);
                    t = t_gate;
                }
                continue;
            }
            step_mask = system_.current_mask();
            step_fired = false;

            // 2b. v2.0 Phase 3 item 2 — EXACT segment stepping.
            //
            // Between events a PWL circuit with DC sources is
            // autonomous LTI, and its trajectory has a closed form
            // valid for ANY h. This is what removes the stability
            // limit that ground the DCM buck to h ≈ 2e-10 even with
            // the state sitting exactly at the idle mode's
            // equilibrium — no error controller fixes a stability
            // bound. Steps are still capped at dt_max so the
            // recorded waveform keeps its resolution, and predicates
            // are located on the ANALYTIC trajectory (sharper than
            // the RK45 path's Hermite interpolant). nullptr —
            // time-varying sources, or a defective eigenbasis for
            // this mask — falls through to the numeric path below,
            // which remains correct everywhere.
            if constexpr (HasExactStep<System>) {
                if (system_.has_exact_step()) {
                    const Real h_ex =
                        std::min(dt_max_, t_target - t);
                    advance_(t, x, h_ex, x_new_);

                    std::optional<PredicateEvent> evt;
                    if (predictor_.size() > 0) {
                        evt = locate_event_exact_(t, x, h_ex, x_new_,
                                                  result);
                    }
                    // An event at the very end of the step fires too:
                    // the next step starts past the crossing and could
                    // never see it again.
                    if (evt.has_value()) {
                        t = evt->t_event;
                        last_fired_index_ = evt->pred_index;
                        last_fired_t_ = t;
                        x = x_evt_;
                        x_pre_event = x;
                        step_fired = true;
                        fire_event_(t, evt->name, evt->type, x,
                                     rk_state, result,
                                     evt->aux_index);
                    } else {
                        t += h_ex;
                        x.swap(x_new_);
                        if (t_gate < t_end && near_time(t, t_gate)) {
                            if (at_corner) t = t_gate;
                            x_pre_event = x;
                            step_fired = true;
                            fire_event_(t_gate, "gate_edge",
                                          PredicateType::GateEdge,
                                          x, rk_state, result);
                        }
                    }
                    rk_state.invalidate();
                    ++result.n_accept;
                    ++result.n_exact_steps;
                    ++step_idx;
                    if (step_idx % store_every_ == 0) {
                        ped_push_sample(result, t, x,
                                        system_.current_mask(), step_mask,
                                        step_fired ? &x_pre_event
                                                   : nullptr);
                    }
                    continue;
                }
            }

            // 3. Snapshot k1 BEFORE step (needed for Hermite backtrack
            // when a predicate fires inside the step).
            const bool need_pred_scan = (predictor_.size() > 0);
            Vector k1_pre;
            if (need_pred_scan) {
                if (rk_state.fsal_valid()) {
                    k1_pre = *rk_state.k1;
                } else {
                    k1_pre = system_.rhs(t, x);
                }
            }

            // 4. DOPRI5 step (FSAL)
            auto [x_new, err] = step(f, t, x, h_use, rk_state);

            // 4b. Finite-value guard (GUI integration findings T1.3).
            // If `f(t, x)` or any RK stage produced NaN/Inf — possibly
            // because the LTI A matrix for the current mask is so
            // ill-conditioned that A·x overflows, or because a Python
            // b_extra_fn divided by zero — `x_new`/`err` propagates
            // it. Pre-fix the controller would treat e=NaN as "reject"
            // (NaN comparisons always false), shrink h, retry, and
            // after `max_rejects` fire an unhelpful generic
            // `5 consecutive rejections (err=nan, h=...)` error.
            //
            // We now detect the NaN/Inf at the RK45 level, log a
            // dedicated nan_streak_ counter, and after `kNanMaxStreak`
            // throw an actionable error pointing the caller at the
            // common root causes. On non-streak NaN we still shrink
            // dt and retry — same as a regular step rejection.
            if (!is_all_finite_(x_new) || !is_all_finite_(err)) {
                rk_state.invalidate();
                ++nan_streak_;
                ++result.n_reject;
                if (nan_streak_ >= kNanMaxStreak) {
                    throw ConvergenceFailure(ConvergenceKind::NaN, t, "",
                        nan_error_message_(t, h_use, "RK45"));
                }
                // Aggressive shrink — a NaN step usually means
                // the current dt is way too coarse for the dynamics
                // at this mask, or the dynamics are uncomputable
                // (singular A). Halve h_use and retry; if the
                // shrink hits dt_min, kNanMaxStreak will bail us out.
                h = std::max(h_use * Real{0.1}, Real{1e-18});
                continue;
            }
            nan_streak_ = 0;

            // 5. PI accept/reject
            auto [accepted, h_next] = controller_.accept(err, x, x_new, h_use);
            if (!accepted) {
                rk_state.invalidate();
                h = h_next;
                ++result.n_reject;
                continue;
            }

// v2.0 Phase 3 — progress guard. An explicit RK45 whose
            // average step has collapsed a thousandfold below dt_max
            // is resolving a time constant it will never finish —
            // the canonical case is a PWL diode turning off into
            // discontinuous conduction, where the idle mode's
            // L·g_off time constant is ~1e-13 s. Burning the
            // 10M-step cap in silence tells the user nothing, so
            // check the CLOCK, not the step size: every 100k steps,
            // simulated time must have advanced by at least
            // 100·dt_max (or a tenth of the run, whichever is
            // smaller).
            if (result.n_accept + result.n_reject
                >= progress_next_check_) {
                const Real need = std::min(dt_max_ * Real{100},
                                            t_end / Real{10});
                if (t - progress_t_mark_ < need) {
                    std::string hint;
                    if constexpr (HasNodePairVoltage<System>) {
                        if (has_diodes_) {
                            hint =
                                " The usual cause with PWL diodes "
                                "is DISCONTINUOUS CONDUCTION: at "
                                "turn-off the circuit enters an "
                                "idle mode whose time constant is "
                                "L*g_off (~1e-13 s), which an "
                                "explicit integrator must resolve "
                                "step by step. The event "
                                "projection (Phase-3 item 2) "
                                "normally removes that mode at the "
                                "commutation; seeing this error "
                                "means it is disabled or declined "
                                "(defective eigenbasis). Use the "
                                "default engine (engine='pwl'), "
                                "whose implicit solver holds DCM "
                                "regardless.";
                        }
                    }
                    throw ConvergenceFailure(ConvergenceKind::NoProgress, t, "",
                        "PEDSimulator: no meaningful progress — "
                        "100000 steps advanced simulated time by "
                        "only " + std::to_string(t
                            - progress_t_mark_)
                        + " s (t = " + std::to_string(t)
                        + "). The integrator is grinding on a "
                        "time constant far below dt_max and will "
                        "not recover." + hint);
                }
                progress_t_mark_ = t;
                progress_next_check_ =
                    result.n_accept + result.n_reject + Size{100000};
            }

            // 6. Post-step predicate scan for diode/ZCD events.
            // rk_state.k1 now holds k7 of the just-taken step (FSAL).
            std::optional<Real> t_pred;
            std::string p_name;
            PredicateType p_type = PredicateType::Custom;
            Vector x_pred;
            int p_aux = -1;
            int p_index = -1;
            if (need_pred_scan) {
                const Vector& k7 = *rk_state.k1;
                auto evt = locate_event_in_step_(t, x, t + h_use, x_new,
                                                    k1_pre, k7, h_use);
                if (evt.has_value()) {
                    t_pred = evt->t_event;
                    p_name = std::move(evt->name);
                    p_type = evt->type;
                    x_pred = std::move(evt->x_at_event);
                    p_aux = evt->aux_index;
                    p_index = evt->pred_index;
                }
            }

            if (t_pred.has_value()
                && *t_pred < t + h_use - Real{1e-15}) {
                // ^ absolute on purpose: a RELATIVE margin here
                // WIDENS the window in which a root at the step
                // end is discarded (event silently lost until
                // recross — adversarial-review finding P0-R4).
                // The proper fix (fire terminal-band roots) is
                // part of the Phase-3 event-queue overhaul.
                // Predicate fired strictly inside the step — backtrack.
                t = *t_pred;
                last_fired_index_ = p_index;
                last_fired_t_ = t;
                x = std::move(x_pred);
                x_pre_event = x;
                step_fired = true;
                fire_event_(t, p_name, p_type, x, rk_state, result,
                             p_aux);
                // After event: PI is reset, FSAL invalid. Pick a small h
                // to restart the new mask's dynamics safely.
                h = std::max(dt_init_, std::min(h_next, h_use));
                ++result.n_accept;
            } else {
                // Commit full step
                t += h_use;
                x = std::move(x_new);
                h = h_next;
                ++result.n_accept;

                // 7. Did we land on a gate edge?
                if (t_gate < t_end &&
                    near_time(t, t_gate)) {
                    if (at_corner) t = t_gate;
                    x_pre_event = x;
                    step_fired = true;
                    fire_event_(t_gate, "gate_edge", PredicateType::GateEdge,
                                  x, rk_state, result);
                }
            }

            // 8. Record
            ++step_idx;
            if (step_idx % store_every_ == 0) {
                ped_push_sample(result, t, x, system_.current_mask(),
                                step_mask,
                                step_fired ? &x_pre_event : nullptr);
            }
        }

        // Final state
        if (result.times.back() != t) {
            ped_push_sample(result, t, x, system_.current_mask(),
                            step_mask,
                            step_fired ? &x_pre_event : nullptr);
        }

        const auto t1_wall = clock::now();
        result.cpu_time_seconds =
            std::chrono::duration<Real>(t1_wall - t0_wall).count();
        return result;
    }

// ----- v2.0 Phase 3: diode commutation (builder path) ----------
    //
    // Hands the scheduler ownership of the PWL diode bits. Only
    // meaningful when System satisfies HasNodePairVoltage (the
    // native builder adapter with the recovery map); the demo
    // systems never call this and compile unchanged.
    void enable_diode_commutation(MaskT diode_owned,
                                    MaskT initial_bits,
                                    std::vector<SchedulerDiode> ds)
        requires HasNodePairVoltage<System> {
        diode_owned_ = std::move(diode_owned);
        diode_bits_ = std::move(initial_bits);
        diodes_ = std::move(ds);
        has_diodes_ = !diodes_.empty();
        diode_burst_count_.assign(diodes_.size(), Size{0});
        diode_burst_t0_.assign(diodes_.size(), Real{0});
    }

    /// add-complementarity-mode-selection D12 stage 1: after every
    /// settle, pose the LCP (step h_e) on the same held state and record
    /// the disagreements. Today's settle still decides: the run's
    /// numbers do not change.
    void enable_mode_shadow(Real h_e)
        requires HasNodePairVoltage<System> {
        shadow_.enabled = true;
        shadow_.h_e = h_e;
    }

    /// D12 stage 2, mode_selection='lcp': the LCP decides the diode mode
    /// at t = 0 and at every event, replacing the settle. `cls` is the
    /// circuit's structural passivity (D3), `engine` the name refusals
    /// carry.
    void enable_mode_lcp(Real h_e, events::PassivityClass cls)
        requires HasNodePairVoltage<System> {
        lcp_decides_ = true;
        shadow_.h_e = h_e;
        passivity_ = std::move(cls);
    }

    /// Slice 2, task 4: the settle still decides; a settled mode that
    /// violates its own law on its exact trajectory is replaced by the
    /// LCP's (counted in `PEDResult::mode_guard`).
    /// `refuse`: the settle alone, judged (DECISIONS #26): a mode the
    /// guard would repair raises UnsupportedError('dsed',
    /// 'consistent_mode', <diode>, 'Diode') instead of being continued.
    void enable_mode_guard(Real h_e, events::PassivityClass cls, bool refuse = false)
        requires HasNodePairVoltage<System> {
        guard_ = true;
        guard_refuses_ = refuse;
        shadow_.h_e = h_e;
        passivity_ = std::move(cls);
    }

    /// v2.0 Phase 3 item 2 — see event_projection.hpp. Safe to leave
    /// on: with no fast stable mode in the new mode's A, the
    /// projection is exactly the identity.
    void set_event_projection(bool on) noexcept {
        enable_event_projection_ = on;
    }

private:
    struct PredicateEvent {
        Real t_event;
        std::string name;
        PredicateType type;
        Vector x_at_event;
        int aux_index = -1;   // fired diode's switch index, or -1
        int pred_index = -1;  // index in predictor_.predicates()
    };

    /// The sign of a diode predicate's signal on its admissible side
    /// (DiodeOn: v − V_on < 0 while OFF; DiodeOff: v − V_f > 0 while ON);
    /// 0 for a predicate with no side.
    [[nodiscard]] static int admissible_sign_(PredicateType t) noexcept {
        return t == PredicateType::DiodeOn ? -1
             : t == PredicateType::DiodeOff ? 1 : 0;
    }

    /// Whether a bracket that starts at `ga` can hold an event: a diode
    /// predicate fires only when its signal LEAVES the admissible side.
    /// One that starts past its threshold by more than the zero band
    /// `band` and comes back is not an event: after a commutation the
    /// projection leaves ~1e-14 A of rounding in an inductor in series
    /// with an OFF diode, which g_off turns into 1e-5 V that relaxes in
    /// L·g_off (1e-13 s). Firing on that return re-decided the same mode
    /// until the chatter guard threw. A start inside the band is read
    /// both ways, as before: an ON diode left at i = 0 (±1e-14 V) by a
    /// series-pair turn-off whose current then drifts negative at leak
    /// scale is caught by that crossing (the single-phase bridge with a
    /// floating bus held D1 ON at −190 nA for 4.9 ms without it).
    /// `band_of()` gives the zero band (zero_band_tol_ at the step
    /// start); it is evaluated only for a start past the threshold.
    template <class Band>
    [[nodiscard]] static bool starts_admissible_(PredicateType t, Real ga,
                                                 Band&& band_of) {
        const Real s = ga * static_cast<Real>(admissible_sign_(t));
        return !(s < Real{0}) || !(s < -band_of());
    }

    /// The probe step for a predicate that starts a step exactly on
    /// zero: the judge's shortest propagation (100·h_e), or 1e-6 of the
    /// step when no h_e is set (guard and LCP off).
    [[nodiscard]] Real zero_probe_dt_(Real h) const noexcept {
        return shadow_.h_e > Real{0} ? Real{100} * shadow_.h_e
                                     : Real{1e-6} * h;
    }

    /// A predicate that starts a step at exactly g = 0 is located only
    /// when it is a diode predicate (whose admissible side is known: a
    /// DiodeOn signal v − V_on is negative while OFF, a DiodeOff signal
    /// v − V_f positive while ON), is not the one that fired at this
    /// very instant, and moves INTO its admissible side just after t0
    /// (`g_plus = g(t0 + d)`, returned in `g_out`). A predicate that
    /// stays at zero (a diode at rest) or leaves on its event side (an
    /// inconsistent mode — the mode decision's case, never a
    /// zero-length event) is skipped, so nothing re-fires at t0: no
    /// Zeno loop by construction.
    template <class G>
    [[nodiscard]] bool zero_start_admissible_(const EventPredicate& p, int ip,
                                              Real t0, Real d, G&& g_at,
                                              Real& g_out) const {
        const int s = admissible_sign_(p.type);
        if (s == 0 || (ip == last_fired_index_ && t0 == last_fired_t_)) {
            return false;
        }
        const Real gp = g_at(t0 + d);
        if (!(gp * static_cast<Real>(s) > Real{0})) return false;
        g_out = gp;
        return true;
    }

    /// The next time event: a gate edge, or a corner of a PWM / pulse
    /// source (add-exact-propagator-strategy D14: no step straddles a
    /// change of a source's piece; the corner is fired like a gate edge,
    /// so the diode modes are decided again after a source jump).
    [[nodiscard]] Real next_gate_edge_(Real t_now, Real t_end) const {
        const Real t_g = next_switch_edge_(t_now, t_end);
        if constexpr (requires { system_.next_source_corner_after(t_now); }) {
            return std::min(t_g, system_.next_source_corner_after(t_now));
        }
        return t_g;
    }

    [[nodiscard]] Real next_switch_edge_(Real t_now, Real t_end) const {
        if constexpr (HasNextEdge<SwitchFn>) {
            // Fast path: switch_fn knows when its next edge is
            // (e.g. NativePwm2Switch with an analytical formula).
            const Real raw = switch_fn_.next_edge_after(t_now);
            if (std::isfinite(raw)) {
                return std::min(raw, t_end);
            }
            // Fallback: the predictor returned ∞, which on the
            // pybind11 `PySwitchFn` adapter means "the user's
            // Python switch_fn doesn't expose next_edge_after".
            // Cap at `t_now + dt_max/10` so the scheduler is
            // forced to land at that boundary and re-sample the
            // switch_fn via fire_gate_event_ (catches any
            // mask change the user didn't pre-announce). Without
            // this cap, PWM-driven simulations with plain Python
            // switch_fns silently produced trajectories with the
            // mask frozen at t=0.
            constexpr Real kPollFractionOfDtMax = Real{0.1};
            return std::min(t_now + dt_max_ * kPollFractionOfDtMax,
                              t_end);
        } else {
            return t_end;
        }
    }

    /// The exact-branch twin of `locate_event_in_step_`: the same
    /// arming and priority rules, with g evaluated on the ANALYTIC
    /// trajectory. D3: the step is scanned at the mode's scan step (an
    /// eighth of the fastest under-damped period) so a double crossing
    /// is not skipped, and the root is located on the time bracket
    /// (locate_root_bracket). The event state is left in `x_evt_`.
    [[nodiscard]] std::optional<PredicateEvent> locate_event_exact_(
        Real t0, const Vector& x0, Real h, const Vector& x1,
        PEDResult<MaskT>& result)
        requires HasExactStep<System> {
        std::optional<PredicateEvent> best;
        int best_priority = std::numeric_limits<int>::max();
        const Real t1 = t0 + h;

        Size n_scan = 1;
        if constexpr (requires { system_.exact_scan_step(); }) {
            const Real hs = system_.exact_scan_step();
            if (std::isfinite(hs) && hs < h) {
                n_scan = std::min<Size>(
                    static_cast<Size>(std::ceil(h / hs)), Size{1024});
            }
        }
        if (scan_x_.size() < n_scan + 1) {
            scan_x_.resize(n_scan + 1);
        }
        for (Size k = 1; k < n_scan; ++k) {
            advance_(t0, x0, h * static_cast<Real>(k)
                             / static_cast<Real>(n_scan), scan_x_[k]);
        }
        const auto t_at = [&](Size k) -> Real {
            return k == n_scan ? t1
                : t0 + h * static_cast<Real>(k) / static_cast<Real>(n_scan);
        };
        const auto x_at = [&](Size k) -> const Vector& {
            return k == 0 ? x0 : (k == n_scan ? x1 : scan_x_[k]);
        };
        const Real t_tol = std::max(
            Real{4} * std::numeric_limits<Real>::epsilon() * std::abs(t1),
            Real{1e-12} * h);

        const auto& preds = predictor_.predicates();
        Real band = Real{-1};
        const auto band_of = [&]() -> Real {
            if (band < Real{0}) band = zero_band_tol_(t0, x0);
            return band;
        };
        for (std::size_t ip = 0; ip < preds.size(); ++ip) {
            const auto& p = preds[ip];
            if (p.required_bit >= 0 &&
                diode_bit_(p.aux_index) != (p.required_bit != 0)) {
                continue;
            }
            Real ga = p.value(t0, x0);
            ++result.n_predicate_evals;
            Real ta0 = t0;
            if (ga == Real{0}) {
                // Starts exactly on zero: read the side it leaves to just
                // after t0, on the exact trajectory (the judge's 100·h_e).
                const Real d = std::min(zero_probe_dt_(h),
                                        Real{0.5} * h / static_cast<Real>(n_scan));
                if (!zero_start_admissible_(p, static_cast<int>(ip), t0, d,
                        [&](Real tau) {
                            advance_(t0, x0, tau - t0, x_tmp_);
                            ++result.n_predicate_evals;
                            return p.value(tau, x_tmp_);
                        }, ga)) {
                    continue;
                }
                ta0 = t0 + d;
            }
            std::optional<Real> root;
            for (Size k = 0; k < n_scan && !root; ++k) {
                const Real ta = k == 0 ? ta0 : t_at(k);
                const Real tb = t_at(k + 1);
                const Real gb = p.value(tb, x_at(k + 1));
                ++result.n_predicate_evals;
                if (!starts_admissible_(p.type, ga, band_of)) {
                    // coming back from past the threshold: no event
                } else if (gb == Real{0}) {
                    root = tb;
                } else if (ga * gb < Real{0}) {
                    auto g_at = [&](Real tau) -> Real {
                        advance_(t0, x0, tau - t0, x_tmp_);
                        ++result.n_predicate_evals;
                        return p.value(tau, x_tmp_);
                    };
                    root = locate_root_bracket(g_at, ta, tb, ga, gb, t_tol);
                }
                ga = gb;
            }
            if (!root) continue;
            if (!best.has_value()
                || *root < best->t_event - Real{1e-15}
                || (std::abs(*root - best->t_event) <= Real{1e-15}
                    && p.priority < best_priority)) {
                best = PredicateEvent{
                    .t_event = *root,
                    .name = p.name,
                    .type = p.type,
                    .x_at_event = {},
                    .aux_index = p.aux_index,
                    .pred_index = static_cast<int>(ip),
                };
                best_priority = p.priority;
            }
        }
        if (best.has_value()) {
            advance_(t0, x0, best->t_event - t0, x_evt_);
        }
        return best;
    }

    /// Exact advance into a caller buffer (no allocation when the
    /// System offers `exact_advance_state_into`).
    void advance_(Real t, const Vector& x, Real h, Vector& out) const
        requires HasExactStep<System> {
        if constexpr (requires {
                          system_.exact_advance_state_into(t, x, h, out);
                      }) {
            system_.exact_advance_state_into(t, x, h, out);
        } else {
            out = system_.exact_advance_state(t, x, h);
        }
    }

    /// Consistent reinitialisation (event_projection.hpp), through the
    /// System's cached eigenbasis when it has one.
    void project_(Real t, Vector& x) {
        if constexpr (HasABInterface<System>) {
            if (!enable_event_projection_) {
                return;
            }
            const Real thr = Real{100} / dt_max_;
            if constexpr (requires { system_.project_slow(t, x, thr); }) {
                system_.project_slow(t, x, thr);
            } else {
                x = project_onto_slow_manifold(
                    system_.A_matrix(), system_.b_vector(t), x, thr);
            }
        }
    }

    /// Settle the diodes, project onto the settled mode's slow
    /// manifold, and settle again on the projected state (repeat while
    /// the mask changes, at most four passes). The projection relaxes
    /// the g_off modes, which can leave a diode forward-biased whose
    /// predicate would then start past its crossing and never fire:
    /// a series diode feeding an LC from rest stayed OFF for the whole
    /// run (v_C = 3.3e-6 V of a 10 V step) before this.
    /// The mask with the event's gates and the given diode bits.
    [[nodiscard]] MaskT effective_gates_(const MaskT& m, const MaskT& bits) const {
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_) return m.overlay(bits, diode_owned_);
        }
        (void)bits;
        return m;
    }

    /// One event's diode decision (t = 0 and fire_event_): today's settle,
    /// or the LCP (mode_selection='lcp'); the shadow comparison and the
    /// default-mode guard after the settle. `gates_prev`: the event's
    /// gates with the diode bits from before it; `prev_bits` those bits;
    /// `rec` the mask x was integrated in.
    void decide_modes_(Real t, Vector& x, MaskT& mask, const MaskT& gates_prev,
                       const MaskT& prev_bits, const MaskT& rec,
                       PEDResult<MaskT>& result) {
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_ && lcp_decides_) {
                auto& rep = result.mode_shadow;
                rep.enabled = rep.decides = true;
                rep.h_e = shadow_.h_e;
                ++rep.lcp_events;
                diode_bits_ = prev_bits;
                mask = lcp_apply_(t, x, gates_prev, prev_bits, rec, rep);
                project_(t, x);
                return;
            }
            if (shadow_.enabled || guard_) shadow_x_ = x;
            settle_rounds_ = 0;
            try {
                settle_and_project_(t, x, mask);
            } catch (const ConvergenceFailure& e) {
                // the settle's round cap: with the guard on, the LCP decides (#25)
                if (!guard_ || guard_refuses_ || e.kind() != ConvergenceKind::Settle) throw;
                // the settle found no mode; the guard's arbiter decides from
                // the held state (an LCP refusal there names the diode)
                result.mode_guard.enabled = true;
                ++result.mode_guard.settle_caps;
                x = shadow_x_;
                diode_bits_ = prev_bits;
                mask = lcp_apply_(t, x, gates_prev, prev_bits, rec, result.mode_guard.lcp);
                project_(t, x);
                return;
            }
            shadow_compare_(t, shadow_x_, gates_prev, prev_bits, result, &rec);
            guard_check_(t, x, mask, gates_prev, prev_bits, rec, result);
        } else {
            (void)gates_prev; (void)prev_bits; (void)rec; (void)result;
            settle_and_project_(t, x, mask);
        }
    }

    /// Selector at (t, x_held), previous mode `prev_bits` over
    /// `gates_prev`, the state read under `rec`; counters into `rep`.
    /// The judge arbitrates leak pins. Leaves the mask at `rec`.
    events::ModeSelection run_selector_(Real t, const Vector& x_held,
                                        const MaskT& gates_prev,
                                        const MaskT& prev_bits, const MaskT& rec,
                                        ModeShadowReport& rep)
        requires HasNodePairVoltage<System> {
        if (shadow_ports_.empty()) {
            for (const auto& d : diodes_) {
                shadow_ports_.push_back(events::DiodePort{
                    Real{1} / d.g_on, d.g_off, d.V_f, d.V_th});
                shadow_nodes_.emplace_back(d.from, d.to);
                shadow_labels_.push_back(d.label);
            }
        }
        std::vector<char> prev(diodes_.size());
        for (std::size_t j = 0; j < diodes_.size(); ++j)
            prev[j] = prev_bits.get(diodes_[j].switch_idx) ? 1 : 0;
        system_.set_mask(rec);
        const events::CircuitScale sc = system_.port_scale(t, x_held);
        selector_.passive = passivity_.passive();
        const auto sel = selector_.select(
            shadow_ports_, prev,
            [&](Real h, DenseMatrix& Z, Vector& u) {
                system_.diode_port_problem(gates_prev, t, x_held, h, shadow_nodes_, Z, u);
            },
            shadow_.h_e, sc,
            [&](const std::vector<char>& m) {
                Real worst = Real{0};
                const int c = judge_(gates_prev, m, t, x_held, sc, worst);
                system_.set_mask(rec);
                return c;
            },
            [&](const std::vector<char>& m, const std::vector<char>& decided) {
                const Real d = first_departure_(gates_prev, m, decided, t, x_held, sc);
                system_.set_mask(rec);
                return d;
            });
        rep.lcps += static_cast<std::size_t>(sel.lcps);
        rep.pivots_total += static_cast<std::size_t>(sel.pivots_total);
        rep.max_pivots_per_event = std::max(rep.max_pivots_per_event, sel.pivots);
        rep.ties += static_cast<std::size_t>(sel.ties);
        rep.widenings += static_cast<std::size_t>(sel.widenings);
        rep.leak_pins += static_cast<std::size_t>(sel.leak_pins);
        rep.pin_judgements += static_cast<std::size_t>(sel.pin_judgements);
        rep.unfixed += static_cast<std::size_t>(sel.unfixed);
        rep.capped += static_cast<std::size_t>(sel.capped);
        rep.released += static_cast<std::size_t>(sel.released);
        rep.enumerations += static_cast<std::size_t>(sel.enumerations);
        rep.nonunique_modes += static_cast<std::size_t>(sel.nonunique_modes);
        last_scale_ = sc;
        return sel;
    }

    /// The LCP decides: diode bits and the system mask from the selected
    /// mode; a failed selection raises the named refusal (D3, task 5).
    MaskT lcp_apply_(Real t, const Vector& x_held, const MaskT& gates_prev,
                     const MaskT& prev_bits, const MaskT& rec, ModeShadowReport& rep)
        requires HasNodePairVoltage<System> {
        const auto sel = run_selector_(t, x_held, gates_prev, prev_bits, rec, rep);
        events::raise_if_refused(sel, "dsed", t, shadow_labels_, passivity_);
        for (std::size_t j = 0; j < diodes_.size(); ++j)
            diode_bits_.set(diodes_[j].switch_idx, sel.on[j] != 0);
        const MaskT m = gates_prev.overlay(diode_bits_, diode_owned_);
        system_.set_mask(m);
        return m;
    }

    /// The default-mode guard: the settled mode, judged on its own exact
    /// trajectory from the state the run continues with. A violation is
    /// not continued silently: the event is decided again by the LCP from
    /// the held state; an LCP refusal there names the diode.
    void guard_check_(Real t, Vector& x, MaskT& mask, const MaskT& gates_prev,
                      const MaskT& prev_bits, const MaskT& rec,
                      PEDResult<MaskT>& result)
        requires HasNodePairVoltage<System> {
        if (!guard_ || !has_diodes_) return;
        auto& g = result.mode_guard;
        g.enabled = true;
        std::vector<char> settled(diodes_.size());
        for (std::size_t j = 0; j < diodes_.size(); ++j)
            settled[j] = diode_bits_.get(diodes_[j].switch_idx) ? 1 : 0;
        const MaskT cur = system_.current_mask();
        const events::CircuitScale sc = system_.port_scale(t, x);
        Real worst = Real{0};
        int worst_j = -1;
        const int c = judge_(cur, settled, t, x, sc, worst, &worst_j);
        system_.set_mask(cur);
        if (c < 0) { ++g.unchecked; return; }
        ++g.checks;
        if (c != 0) return;
        const std::string who = worst_j >= 0
            ? diodes_[static_cast<std::size_t>(worst_j)].label : std::string{};
        const MaskT bits_settled = diode_bits_;
        diode_bits_ = prev_bits;
        events::ModeSelection sel;
        try {
            sel = run_selector_(t, shadow_x_, gates_prev, prev_bits, rec, g.lcp);
            events::raise_if_refused(sel, "dsed", t, shadow_labels_, passivity_);
        } catch (const UnsupportedError& e) {
            throw UnsupportedError(
                "dsed", e.feature(),
                e.feature() == "consistent_mode" && !who.empty() ? who : e.device(),
                e.feature() == "consistent_mode" ? "Diode" : e.kind(),
                "dsed at t = " + sci_(t) + " s: the settled diode mode violates the law of "
                + who + " on its own trajectory (by " + sci_(worst) + " V), and the "
                "complementarity selection refuses there: " + e.what());
        }
        if (sel.on == settled) {
            // the LCP agrees with the settle: nothing to repair here
            diode_bits_ = bits_settled;
            system_.set_mask(cur);
            ++g.confirmed;
            return;
        }
        if (guard_refuses_) {
            throw UnsupportedError(
                "dsed", "consistent_mode", who, "Diode",
                "dsed at t = " + sci_(t) + " s: the settled diode mode violates the law of "
                + who + " on its own trajectory (by " + sci_(worst) + " V) and the "
                "complementarity selection decides otherwise; with the mode guard off "
                "the settle's mode would be continued silently wrong. Leave the guard on "
                "(the default) or pass mode_selection='lcp'.");
        }
        ++g.repairs;
        g.repair_times.push_back(t);
        g.repair_diodes.push_back(who);
        g.repair_violations.push_back(worst);
        for (std::size_t j = 0; j < diodes_.size(); ++j)
            diode_bits_.set(diodes_[j].switch_idx, sel.on[j] != 0);
        mask = gates_prev.overlay(diode_bits_, diode_owned_);
        system_.set_mask(mask);
        x = shadow_x_;
        project_(t, x);
    }

    void settle_and_project_(Real t, Vector& x, MaskT& mask) {
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_) {
                mask = settle_diode_cascade_(t, x);
                project_(t, x);
                for (int pass = 0; pass < 4; ++pass) {
                    const MaskT m2 = settle_diode_cascade_(t, x);
                    if (m2 == mask) {
                        break;
                    }
                    mask = m2;
                    project_(t, x);
                }
                return;
            }
        }
        project_(t, x);
    }

    [[nodiscard]] static std::string sci_(Real v) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.9e", v);
        return buf;
    }

    /// Scan all armed predicates over (t0, t1). Return the earliest
    /// sign-change root via Illinois on the Hermite cubic, or nullopt
    /// if no predicate changed sign. Ties broken by predicate priority.
    [[nodiscard]] std::optional<PredicateEvent> locate_event_in_step_(
        Real t0, const Vector& x0,
        Real t1, const Vector& x1,
        const Vector& k1, const Vector& k7,
        Real h) {

        std::optional<PredicateEvent> best;
        int best_priority = std::numeric_limits<int>::max();

        const auto& preds = predictor_.predicates();
        Real band = Real{-1};
        const auto band_of = [&]() -> Real {
            if (band < Real{0}) band = zero_band_tol_(t0, x0);
            return band;
        };
        for (std::size_t ip = 0; ip < preds.size(); ++ip) {
            const auto& p = preds[ip];
            // v2.0 Phase 3 — state-dependent arming. A diode's
            // DiodeOn predicate is only meaningful while its bit is
            // OFF (and vice versa); evaluating the other one would
            // fire spurious events off a signal that has no physical
            // reading in the current mode. This bit-match IS the
            // hysteresis: firing flips the bit, which disarms the
            // predicate that just fired and arms its counterpart.
            if (p.required_bit >= 0 &&
                diode_bit_(p.aux_index) !=
                    (p.required_bit != 0)) {
                continue;
            }
            Real g0 = p.value(t0, x0);
            const Real g1 = p.value(t1, x1);
            const auto hermite_g = [&p, &x0, &x1, &k1, &k7, h, t0](Real tau) -> Real {
                const Real theta = (tau - t0) / h;
                return p.value(tau, hermite_interp(x0, x1, k1, k7, h, theta));
            };

            // A predicate at exactly zero is located only when it leaves
            // into its admissible side (zero_start_admissible_).
            Real ta = t0;
            if (g0 == Real{0}) {
                const Real d = std::min(zero_probe_dt_(h), Real{0.5} * h);
                if (!zero_start_admissible_(p, static_cast<int>(ip), t0, d,
                                            hermite_g, g0)) {
                    continue;
                }
                ta = t0 + d;
            }
            if (g0 * g1 >= Real{0} || !starts_admissible_(p.type, g0, band_of)) continue;

            // Locate root via Illinois on the Hermite interpolant.
            auto g_at = [&hermite_g, ta, t1, g0, g1](Real tau) -> Real {
                if (tau <= ta) return g0;
                if (tau >= t1) return g1;
                return hermite_g(tau);
            };

            Real t_root;
            try {
                t_root = illinois(g_at, ta, t1, g0, g1);
            } catch (const std::runtime_error&) {
                predictor_.note_illinois_failure();
                t_root = bisect_fallback(g_at, ta, t1);
            }

            const auto make_event = [&]() -> PredicateEvent {
                const Real theta = (t_root - t0) / h;
                Vector x_root = hermite_interp(x0, x1, k1, k7, h, theta);
                return PredicateEvent{
                    .t_event = t_root,
                    .name = p.name,
                    .type = p.type,
                    .x_at_event = std::move(x_root),
                    .aux_index = p.aux_index,
                    .pred_index = static_cast<int>(ip),
                };
            };

            if (!best.has_value()
                || t_root < best->t_event - Real{1e-15}) {
                best = make_event();
                best_priority = p.priority;
            } else if (std::abs(t_root - best->t_event) <= Real{1e-15}
                       && p.priority < best_priority) {
                best = make_event();
                best_priority = p.priority;
            }
        }

        return best;
    }

    void fire_event_(Real t_event,
                       const std::string& name,
                       PredicateType ptype,
                       Vector& x,
                       RK45State& rk_state,
                       PEDResult<MaskT>& result,
                       int aux_index = -1) {
        const MaskT old_mask = system_.current_mask();
        const MaskT prev_bits = diode_bits_;

        // Notify switch_fn of non-gate (diode/ZCD) events so it can
        // latch the post-event mode for the remainder of this cycle.
        if constexpr (HasZcdRegister<SwitchFn>) {
            if (ptype == PredicateType::CurrentZC
                || ptype == PredicateType::DiodeOff
                || ptype == PredicateType::DiodeOn) {
                switch_fn_.register_zcd_transition(t_event);
            }
        }

        // State projection — clamp algebraic constraints at the event.
        if (state_projection_) {
            x = state_projection_(t_event, x, ptype);
        }

        // v2.0 Phase 3 — a diode predicate's firing flips the bit
        // DIRECTLY. Until now the new mask was re-sampled from
        // switch_fn(t), a pure function of time that knows nothing
        // about diodes — which is exactly why the dsed engine had no
        // diode, only a resistor whose state the user pinned.
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_ && aux_index >= 0 &&
                (ptype == PredicateType::DiodeOn ||
                 ptype == PredicateType::DiodeOff)) {
                diode_bits_.set(static_cast<Size>(aux_index),
                                 ptype == PredicateType::DiodeOn);
                note_diode_fire_(aux_index, t_event);
            }
        }

        const MaskT new_mask =
            effective_mask_(switch_fn_(advance_past(t_event)));

        // Drop spurious same-mask gate edges silently.
        if (ptype == PredicateType::GateEdge && new_mask == old_mask) {
            return;
        }

        if (new_mask != old_mask) {
            system_.set_mask(new_mask);
        }

        // Zero-time cascade. The mask change reconfigures the
        // algebraic sub-circuit INSTANTLY, so another diode can be
        // past its threshold at this very instant — a buck's
        // freewheel diode at gate-off is the canonical case: with
        // the switch open and the diode still off, the inductor
        // current forced through g_off puts the switch node at
        // ±i/g_off volts, and integrating even one step of that
        // mode would be nonsense. Settle to a consistent diode
        // configuration before integration resumes — the same
        // fixed-point iteration run_transient does per step, done
        // here only at events.
        MaskT final_mask = new_mask;
        decide_modes_(t_event, x, final_mask,
                      effective_gates_(new_mask, prev_bits), prev_bits,
                      old_mask, result);

        // Consistent reinitialization (Phase-3 item 2). The event
        // deposits x off the new mode's slow manifold — a µA of
        // root-location residual in a DCM inductor, volts of
        // difference across caps a switch just paralleled — and the
        // excited decay modes sit ~1e6 below any step the
        // integrator will take. Project them to quasi-static; slow
        // components (the conserved charges/fluxes) pass through
        // exactly. Identity whenever no fast stable mode exists.
        rk_state.invalidate();      // f discontinues at event
        controller_.reset();        // avoid PI wind-up across mode change

        result.event_log.push_back(EventRecord<MaskT>{
            .t = t_event,
            .name = name,
            .type = ptype,
            .old_mask = old_mask,
            .new_mask = final_mask,
        });
        ++result.n_events;
    }

    // ----- v2.0 Phase 3: diode-commutation machinery -----------------

    /// Overlay the solver-owned diode bits onto a switch_fn mask.
    /// The pwl engine's combine_masks, done at the same seam.
    [[nodiscard]] MaskT effective_mask_(MaskT gate_mask) const {
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_) {
                return gate_mask.overlay(diode_bits_, diode_owned_);
            }
        }
        return gate_mask;
    }

    /// Chatter detection — see the member comment. A burst is fires
    /// of one diode inside a window of dt_max_/100; 32 of them means
    /// the configuration is oscillating, not commutating.
    void note_diode_fire_(int aux_index, Real t) {
        Size di = Size{0};
        for (Size i = 0; i < diodes_.size(); ++i) {
            if (static_cast<int>(diodes_[i].switch_idx)
                == aux_index) {
                di = i;
                break;
            }
        }
        const Real window = dt_max_ / Real{100};
        if (t - diode_burst_t0_[di] > window) {
            diode_burst_t0_[di] = t;
            diode_burst_count_[di] = Size{0};
        }
        if (++diode_burst_count_[di] > Size{32}) {
            throw ConvergenceFailure(ConvergenceKind::Chatter, t, diodes_[di].label,
                "PEDSimulator: " + diodes_[di].label
                + " is chattering at t = " + sci_(t)
                + " s — it commutated 32 times inside "
                + sci_(window) + " s, which is an oscillation at a "
                "zero crossing, not switching. Use engine='pwl' or "
                "engine='trbdf2' for this circuit, or give the diode "
                "a hysteresis band (V_th = V_f + 1e-3).");
        }
    }

    [[nodiscard]] bool diode_bit_(int switch_idx) const {
        if constexpr (HasNodePairVoltage<System>) {
            return diode_bits_.get(static_cast<Size>(switch_idx));
        } else {
            (void)switch_idx;
            return false;
        }
    }

    /// Iterate the diode configuration to a fixed point at one
    /// instant, Gauss–Seidel (the mask follows each flip). Returns the
    /// settled mask (already set on the system). Budget-bounded; on
    /// exhaustion, throws NAMING the devices still flipping.
    ///
    /// D4 (add-exact-propagator-strategy): a diode whose crossing
    /// signal is zero to rounding cannot be decided by its value —
    /// `V_th = 0` at rest, where v_D = 0 turns it ON and i_D = 0 turns
    /// it OFF on alternate rounds. It is decided by where the OFF mode
    /// takes the signal (zero_band_on_); away from zero the shared
    /// `SwitchedDiode::decide` law is untouched.
    MaskT settle_diode_cascade_(Real t, const Vector& x)
        requires HasNodePairVoltage<System> {
        MaskT mask = system_.current_mask();
        const Real tol = zero_band_tol_(t, x);
        constexpr int kMaxRounds = 64;
        for (int round = 0; round < kMaxRounds; ++round) {
            ++settle_rounds_;
            bool flipped = false;
            for (const auto& d : diodes_) {
                const bool on =
                    diode_bits_.get(d.switch_idx);
                const Real v_d = system_.node_pair_voltage(
                    d.from, d.to, t, x);
                const auto prm = d.params();
                const bool by_value =
                    models::SwitchedDiode::decide(on, v_d, prm);
                const bool in_band =
                    std::abs(models::SwitchedDiode::crossing_signal(
                        on, v_d, prm)) <= tol;
                // A flip is confirmed on the OFF-mode trajectory: in
                // the OFF mode g_off turns a rounding-level current
                // into volts of v_D (1e-18 A → 1e-9 V at 1 nS), so the
                // value law alone can flip a diode the ON mode then
                // flips back.
                const bool next = (in_band || by_value != on)
                    ? zero_band_on_(t, x, d, mask, tol,
                                    in_band ? false : by_value)
                    : on;
                if (next != on) {
                    diode_bits_.set(d.switch_idx, next);
                    flipped = true;
                    const MaskT candidate =
                        mask.overlay(diode_bits_, diode_owned_);
                    if (candidate != mask) {
                        mask = candidate;
                        system_.set_mask(mask);
                    }
                }
            }
            if (!flipped) {
                break;
            }
            if (round == kMaxRounds - 1) {
                std::string culprits;
                for (const auto& d : diodes_) {
                    if (!culprits.empty()) culprits += ", ";
                    culprits += d.label;
                }
                throw ConvergenceFailure(ConvergenceKind::Settle, t, "",
                    "PEDSimulator: diode configuration did not "
                    "settle after 64 rounds at t = "
                    + sci_(t) + " s (diodes: " + culprits
                    + ") — a pair is fighting over the same node; "
                    "add a small hysteresis band (e.g. V_th = V_f + "
                    "1e-3 on add_diode) or an RC snubber");
            }
        }
        return mask;
    }

    /// The zero band: rounding level of the circuit's voltages.
    [[nodiscard]] Real zero_band_tol_(Real t, const Vector& x) const {
        Real vs = Real{0};
        if constexpr (requires { system_.max_node_voltage(t, x); }) {
            vs = system_.max_node_voltage(t, x);
        }
        for (const auto& d : diodes_) {
            vs = std::max({vs, std::abs(d.V_th), std::abs(d.V_f)});
        }
        return Real{1e-12} * vs;
    }

    /// D4 for one diode in the zero band, or whose value law wants to
    /// flip it: ON iff the OFF mode drives
    /// its signal v_D − V_on above the band. The signal is read on the
    /// exact OFF-mode trajectory at δ = 1e-12·dt_max, 1e-11·dt_max, …,
    /// dt_max; the first δ where it leaves the band decides, which is
    /// the sign of its first non-zero derivative with the stiff
    /// (g_off) modes already relaxed. A signal that never leaves the
    /// band carries no current either way: OFF.
    [[nodiscard]] bool zero_band_on_(Real t, const Vector& x,
                                     const SchedulerDiode& d,
                                     const MaskT& mask, Real tol,
                                     bool fallback)
        requires HasNodePairVoltage<System> {
        const bool was = diode_bits_.get(d.switch_idx);
        diode_bits_.set(d.switch_idx, false);
        system_.set_mask(mask.overlay(diode_bits_, diode_owned_));
        diode_bits_.set(d.switch_idx, was);
        // Without an exact OFF-mode step: the value law outside the
        // band, OFF inside it.
        bool on = fallback;
        if constexpr (HasExactStep<System>) {
            if (system_.has_exact_step()) {
                on = false;
                const auto prm = d.params();
                for (int k = 0; k <= 12; ++k) {
                    const Real delta =
                        dt_max_ * std::pow(Real{10}, Real(k - 12));
                    advance_(t, x, delta, x_tmp_);
                    const Real sig =
                        models::SwitchedDiode::crossing_signal(
                            false,
                            system_.node_pair_voltage(
                                d.from, d.to, t + delta, x_tmp_),
                            prm);
                    if (sig > tol) {
                        on = true;
                        break;
                    }
                    if (sig < -tol) {
                        break;
                    }
                }
            }
        }
        system_.set_mask(mask);
        return on;
    }

    /// The shadow LCP at one settle: `gates_prev` is the mask with the
    /// event's gates and the diode bits from BEFORE the event, the
    /// previous mode the LCP is posed in; `x_held` the state the settle
    /// started from; `rec_mask` the mask the state's recovery map is
    /// read under (the one x was integrated in; default: gates_prev).
    /// Touches no state the run reads: the system's mask is restored.
    void shadow_compare_(Real t, const Vector& x_held,
                         const MaskT& gates_prev, const MaskT& prev_bits,
                         PEDResult<MaskT>& result,
                         const MaskT* rec_mask = nullptr) {
        if constexpr (requires(const System& s, const Vector& xx,
                               DenseMatrix& Z, Vector& u) {
                          s.diode_port_problem(gates_prev, t, xx, Real{1},
                              std::vector<std::pair<Index, Index>>{}, Z, u);
                          s.port_scale(t, xx);
                      }) {
            if (!shadow_.enabled || !has_diodes_) return;
            auto& rep = result.mode_shadow;
            rep.enabled = true;
            rep.h_e = shadow_.h_e;
            std::vector<char> prev(diodes_.size()), settled(diodes_.size());
            for (std::size_t j = 0; j < diodes_.size(); ++j) {
                prev[j] = prev_bits.get(diodes_[j].switch_idx) ? 1 : 0;
                settled[j] = diode_bits_.get(diodes_[j].switch_idx) ? 1 : 0;
            }
            const MaskT cur = system_.current_mask();
            const MaskT& rec = rec_mask ? *rec_mask : gates_prev;
            ++rep.lcp_events;
            rep.settle_rounds_total += static_cast<std::size_t>(settle_rounds_);
            try {
                const auto sel = run_selector_(t, x_held, gates_prev, prev_bits, rec, rep);
                const events::CircuitScale sc = last_scale_;
                const bool solved = sel.status == events::LcpStatus::Solution
                                    || sel.status == events::LcpStatus::Trivial;
                if (!solved || sel.on != settled) {
                    const Vector& ci = selector_.last_current();
                    const Vector& cw = selector_.last_gap();
                    ModeShadowRecord r{
                        t, settled, sel.on, prev, settle_rounds_, sel.pivots,
                        sel.ties, sel.status,
                        std::vector<Real>(ci.data(), ci.data() + ci.size()),
                        std::vector<Real>(cw.data(), cw.data() + cw.size())};
                    r.settle_consistent = judge_(gates_prev, settled, t, x_held, sc,
                                                 r.settle_violation);
                    if (solved) {
                        r.lcp_consistent = judge_(gates_prev, sel.on, t, x_held, sc,
                                                  r.lcp_violation);
                    }
                    rep.disagreements.push_back(std::move(r));
                }
            } catch (const std::exception& ex) {
                // recorded, not swallowed: the count and the text are in
                // the report; the settle's answer stands (shadow only)
                ++rep.errors;
                rep.last_error = ex.what();
            }
            system_.set_mask(cur);
        } else {
            (void)t; (void)x_held; (void)gates_prev; (void)prev_bits;
            (void)result; (void)rec_mask;
        }
    }

    /// The shadow's judge: is `mode` (diode bits over `gates`) consistent
    /// on its own exact trajectory from x at t? 1 yes, 0 no, -1 unknown.
    /// Leaves the system's mask to the caller to restore.
    int judge_(const MaskT& gates, const std::vector<char>& mode, Real t,
               const Vector& x, const events::CircuitScale& sc, Real& worst,
               int* worst_j = nullptr)
        requires HasNodePairVoltage<System> {
        worst = Real{0};
        if constexpr (HasExactStep<System>) {
            MaskT m = gates;
            for (std::size_t j = 0; j < diodes_.size(); ++j)
                m.set(diodes_[j].switch_idx, mode[j] != 0);
            system_.set_mask(m);
            if (!system_.has_exact_step()) return -1;
            const Real tol_v = Real{1e-9} * sc.vscale;
            const Real zs = sc.vscale / sc.iscale;
            for (const Real f : {Real{1e2}, Real{1e4}}) {
                const Real d = f * shadow_.h_e;
                advance_(t, x, d, x_tmp_);
                for (std::size_t j = 0; j < diodes_.size(); ++j) {
                    const auto& dd = diodes_[j];
                    const Real v = system_.node_pair_voltage(dd.from, dd.to, t + d, x_tmp_);
                    // violation in volts: ON  -> −i·zs, OFF -> v − V_on
                    const Real viol = mode[j]
                        ? -(dd.g_on * (v - dd.V_f)) * zs
                        : v - std::max(dd.V_th, dd.V_f);
                    // a non-finite state (an unstable mode over 1e4·h_e) is a violation
                    const Real v_eff = std::isfinite(viol) ? viol
                                                           : std::numeric_limits<Real>::infinity();
                    if (v_eff > worst) {
                        worst = v_eff;
                        if (worst_j) *worst_j = static_cast<int>(j);
                    }
                }
            }
            return worst > tol_v ? 0 : 1;
        } else {
            (void)gates; (void)mode; (void)t; (void)x; (void)sc; (void)worst_j;
            return -1;
        }
    }

    /// The selector's horizon (DECISIONS #24): the first d of
    /// 100·h_e·2^k, k = 0..7, at which a `decided` diode of `mode` breaks
    /// its own law on the exact trajectory (the judge's measure), +∞ if
    /// none up to 12 800·h_e or no exact step. The event lies in (d/2, d].
    /// Leaves the system's mask to the caller to restore.
    Real first_departure_(const MaskT& gates, const std::vector<char>& mode,
                          const std::vector<char>& decided, Real t, const Vector& x,
                          const events::CircuitScale& sc)
        requires HasNodePairVoltage<System> {
        constexpr Real inf = std::numeric_limits<Real>::infinity();
        if constexpr (HasExactStep<System>) {
            MaskT m = gates;
            for (std::size_t j = 0; j < diodes_.size(); ++j)
                m.set(diodes_[j].switch_idx, mode[j] != 0);
            system_.set_mask(m);
            if (!system_.has_exact_step()) return inf;
            const Real tol_v = Real{1e-9} * sc.vscale;
            const Real zs = sc.vscale / sc.iscale;
            Real d = Real{100} * shadow_.h_e;
            for (int k = 0; k < 8; ++k, d *= Real{2}) {
                advance_(t, x, d, x_tmp_);
                for (std::size_t j = 0; j < diodes_.size(); ++j) {
                    if (!decided[j]) continue;
                    const auto& dd = diodes_[j];
                    const Real v = system_.node_pair_voltage(dd.from, dd.to, t + d, x_tmp_);
                    const Real viol = mode[j]
                        ? -(dd.g_on * (v - dd.V_f)) * zs
                        : v - std::max(dd.V_th, dd.V_f);
                    if (!(viol <= tol_v)) return d;   // NaN counts as a departure
                }
            }
            return inf;
        } else {
            (void)gates; (void)mode; (void)decided; (void)t; (void)x; (void)sc;
            return inf;
        }
    }

    /// Whether every entry of `v` is finite (not NaN, not Inf).
    [[nodiscard]] static bool is_all_finite_(const Vector& v) noexcept {
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v(i))) return false;
        }
        return true;
    }

    /// Compose the actionable error message thrown when the RK45 step
    /// path produces NaN/Inf for `kNanMaxStreak` consecutive iterations.
    [[nodiscard]] std::string nan_error_message_(Real t, Real h,
                                                    const char* sched) const {
        std::string msg{"PEDSimulator ("};
        msg += sched;
        msg += "): step produced NaN/Inf for ";
        msg += std::to_string(kNanMaxStreak);
        msg += " consecutive iterations at t=";
        msg += std::to_string(t);
        msg += ", h=";
        msg += std::to_string(h);
        msg += ". Common root causes:\n";
        msg += "  (1) the LTI A matrix for the current switch mask "
                "is numerically singular or so ill-conditioned that "
                "A·x overflows IEEE-754 double precision;\n";
        msg += "  (2) a Python b_extra_fn (e.g. a motor / control "
                "observer) returned NaN because it divided by zero "
                "or read uninitialized state;\n";
        msg += "  (3) a switch_fn returned a mask whose extracted "
                "state-space dynamics blow up.\n";
        msg += "Workarounds while you investigate: pass "
                "`engine='pwl'` (more robust on multi-stage switched "
                "topologies — T1.2 auto-LM handles rank-deficient "
                "Jacobians transparently), or tighten `dt_max` to "
                "force more event samples, or audit your "
                "b_extra_fn / switch_fn for NaN-producing branches.";
        return msg;
    }

    static constexpr std::size_t kNanMaxStreak = 3;

    System& system_;
    SwitchFn switch_fn_;
    PIController controller_;
    EventPredictor predictor_;
    // v2.0 Phase 3 — diode commutation state. Empty unless
    // enable_diode_commutation() was called (builder path only).
    // `diode_bits_` is the solver-owned latch, overlaid onto every
    // mask the switch_fn produces — the same combine the pwl
    // engine's run_transient does with DiodeEventState.
    MaskT diode_owned_{};
    MaskT diode_bits_{};
    std::vector<SchedulerDiode> diodes_;
    bool has_diodes_ = false;
    // Chatter guard: per-diode burst counter. A diode that fires
    // repeatedly with essentially no time progress is oscillating at
    // a zero crossing — DCM's i ≈ 0 idle, where the root-location
    // residual (~µA) through g_off (~nS) reconstructs as kilovolts
    // of v_D and re-arms the diode instantly. Detect and NAME it
    // rather than burn the 10M-step cap in silence.
    std::vector<Size> diode_burst_count_;
    std::vector<Real> diode_burst_t0_;
    Real progress_t_mark_ = Real{0};
    Size progress_next_check_ = Size{100000};
    // v2.0 Phase 3 item 2 — consistent reinitialization. When on,
    // every event projects the carried-over state onto the NEW
    // mode's slow manifold (see event_projection.hpp). Off by
    // default so the demo systems and raw-LTI paths are untouched;
    // the builder runners enable it.
    bool enable_event_projection_ = false;
    Real dt_init_;
    Real dt_max_;
    std::size_t store_every_;
    StateProjectionFn state_projection_;
    std::size_t nan_streak_ = 0;
    // add-complementarity-mode-selection — the shadow LCP (stage 1).
    struct { bool enabled = false; Real h_e = Real{0}; } shadow_;
    bool lcp_decides_ = false;   // mode_selection='lcp'
    bool guard_ = false;         // default-mode guard
    bool guard_refuses_ = false; // guard off: refuse what it would repair
    events::PassivityClass passivity_;
    events::CircuitScale last_scale_;
    std::vector<std::string> shadow_labels_;
    events::ModeSelector selector_;
    std::vector<events::DiodePort> shadow_ports_;
    std::vector<std::pair<Index, Index>> shadow_nodes_;
    Vector shadow_x_;
    int settle_rounds_ = 0;
    // The predicate that fired the last predicate event, and when: only
    // it is skipped as "just fired" at a step starting at that instant.
    int last_fired_index_ = -1;
    Real last_fired_t_ = std::numeric_limits<Real>::quiet_NaN();
    // Exact-path workspaces: a step reuses them (no allocation).
    mutable Vector x_new_, x_evt_, x_tmp_;
    std::vector<Vector> scan_x_;
};

}  // namespace pulsim::dsed
