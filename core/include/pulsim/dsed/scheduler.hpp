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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <cstdio>
#include <utility>
#include <vector>

#include "pulsim/errors/convergence.hpp"
#include "pulsim/dsed/event_predictor.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/events/mode_refusal.hpp"
#include "pulsim/events/mode_selector.hpp"
#include "pulsim/events/step_hooks.hpp"
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

/// A stored sample breaks the PWL diode law: an ON diode carries reverse
/// current, or an OFF one is forward-biased past its threshold, by more
/// than `rel` of that diode's peak over the run (PEDSimulator::check_diode_law,
/// run at the end of every simulate() unless set_diode_law_check(0); Python maps it to DiodeModeInconsistentError).
class DiodeLawViolation : public std::runtime_error {
public:
    DiodeLawViolation(std::string diode, Real t, const std::string& what)
        : std::runtime_error(what), diode_(std::move(diode)), t_(t) {}
    [[nodiscard]] const std::string& diode() const noexcept { return diode_; }
    [[nodiscard]] Real time() const noexcept { return t_; }
private:
    std::string diode_;
    Real t_;
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
        probe_t_end_ = t_end;
        last_time_event_t_ = Real{0};   // the start samples the gates like a landed edge
        using clock = std::chrono::high_resolution_clock;
        const auto t0_wall = clock::now();

        Vector x = x0;
        Real t = Real{0};
        Real h = dt_init_;
        if constexpr (requires { system_.set_max_step(dt_max_); }) {
            system_.set_max_step(dt_max_);
        }
        if constexpr (requires { system_.set_horizon(t_end); }) {
            system_.set_horizon(t_end);
        }
        start_time_events_(t_end);
        MaskT mask = effective_mask_(switch_fn_(t));
        system_.set_mask(mask);
        // A diode may be forward-biased already at t = 0 (a
        // rectifier powered on mid-cycle). Settle before the first
        // step rather than integrating an inconsistent mode.
        RK45State rk_state;
        PEDResult<MaskT> result;
        law_settled_.clear();
        // a run that threw inside fire_event_ must not leak its event context
        settle_decisions_ = false;
        has_event_sample_ = false;
        loc_dx_.resize(0);
        {
            MaskT m0 = system_.current_mask();
            const MaskT prev0 = diode_bits_;
            const MaskT old0 = m0;
            decide_modes_(t, x, m0, old0, prev0, old0, result);
        }
        // The ticks at t = 0 run before the first sample; the gate is
        // sampled again with what they wrote.
        if (run_due_ticks_(t, x)) {
            fire_event_(t, "control_tick", PredicateType::GateEdge, x, rk_state, result);
        }
        run_due_monitors_(t, x);
        if constexpr (std::is_same_v<MaskT, topology::SwitchStateMask>) {
            if (hooks_ != nullptr) hooks_->start(x, system_.current_mask());
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

        auto f = [this](Real tau, const Vector& xx) -> Vector {
            return system_.rhs(tau, xx);
        };

        while (t < t_end) {
            if (result.n_accept + result.n_reject > max_steps_) {
                throw ConvergenceFailure(ConvergenceKind::MaxSteps, t, "",
                    "PEDSimulator: exceeded max-step cap");
            }

            // 0. A control tick not taken at a landing (a run that sits
            // on its instant): run it and sample the gate again.
            if (run_due_ticks_(t, x)) {
                const MaskT before = system_.current_mask();
                rk_state.invalidate();
                fire_event_(t, "control_tick", PredicateType::GateEdge, x, rk_state, result);
                plan_fast_samples_(t, !(system_.current_mask() == before));
            }
            run_due_monitors_(t, x);

            // 1. Next time event from the queue: gate edge, source corner, tick
            Real t_sw = t_end;
            Real t_gate = next_gate_edge_(t, t_end, t_sw);
            // an edge a few ulps before t_end is t_end (scheduler_auto.hpp); ulps
            // only: the 1e-12 s floor swallowed a 1 ps pulse ending a period run
            if (t_gate < t_end && near_time(t_gate, t_end, Real{0})) t_gate = t_end;
            const Real t_target = std::min({t_gate, t_end, queue_.next_of(events::Kind::ControlTick)});
            // a source corner is landed on exactly: a single-shot pulse
            // read one ulp early would keep its old piece for a step
            const bool at_corner = t_gate < t_sw;

            // 2. Propose step capped at next gate
            Real h_use = std::min({h, dt_max_, t_target - t});
            if (h_use <= Real{0}) {
                // Sitting exactly on a gate edge — fire it.
                if (t_gate < t_end) {
                    const MaskT before = system_.current_mask();
                    fire_event_(t_gate, "gate_edge", PredicateType::GateEdge,
                                  x, rk_state, result);
                    t = t_gate;
                    plan_fast_samples_(t, at_corner || !(system_.current_mask() == before));
                }
                continue;
            }
            step_mask = system_.current_mask();
            step_fired = false;
            const Real t_step0 = t;

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
                if (!system_.has_exact_step()) {
                    // 13.2: engine='exact' refuses a numeric segment unless allowed
                    if constexpr (requires { system_.refuse_implicit_step(t); }) {
                        system_.refuse_implicit_step(t);
                    }
                }
                if (system_.has_exact_step()) {
                    Real h_ex = std::min(dt_max_, t_target - t);
                    while (fast_k_ < fast_t_.size()
                           && (fast_t_[fast_k_] <= t || near_time(t, fast_t_[fast_k_]))) {
                        ++fast_k_;
                    }
                    if (fast_k_ < fast_t_.size()) {
                        h_ex = std::min(h_ex, fast_t_[fast_k_] - t);
                    }
                    // no sliver step short of the edge (scheduler_auto.hpp)
                    if (h_ex < t_target - t && near_time(t + h_ex, t_target)) {
                        h_ex = t_target - t;
                    }
                    const MaskT mask_before = system_.current_mask();
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
                        plan_fast_samples_(t, !(system_.current_mask() == mask_before));
                    } else {
                        t += h_ex;
                        x.swap(x_new_);
                        const bool on_edge = t_gate < t_end && near_time(t, t_gate);
                        if (on_edge || tick_due_(t)) {
                            if (on_edge && at_corner) t = t_gate;
                            x_pre_event = x;
                            step_fired = true;
                            run_due_ticks_(t, x);   // a tick at the edge's instant first (D2)
                            fire_event_(on_edge ? t_gate : t, on_edge ? "gate_edge" : "control_tick",
                                          PredicateType::GateEdge,
                                          x, rk_state, result);
                            plan_fast_samples_(
                                t, (on_edge && at_corner)
                                       || !(system_.current_mask() == mask_before));
                        }
                    }
                    rk_state.invalidate();
                    ++result.n_accept;
                    ++result.n_exact_steps;
                    hook_accepted_(t_step0, t, step_fired ? x_pre_event : x, step_mask);
                    ++step_idx;
                    if (step_idx % store_every_ == 0) {
                        ped_push_sample(result, t, x,
                                        system_.current_mask(), step_mask,
                                        step_fired ? &x_pre_event
                                                   : nullptr);
                        if (step_fired && has_event_sample_)
                            law_settled_.emplace_back(result.times.size() - 1, event_sample_);
                    }
                    has_event_sample_ = false;
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

                // 7. Did we land on a gate edge or a control tick?
                const bool on_edge = t_gate < t_end && near_time(t, t_gate);
                if (on_edge || tick_due_(t)) {
                    if (on_edge && at_corner) t = t_gate;
                    x_pre_event = x;
                    step_fired = true;
                    if (run_due_ticks_(t, x)) rk_state.invalidate();   // D2: tick first
                    fire_event_(on_edge ? t_gate : t, on_edge ? "gate_edge" : "control_tick",
                                  PredicateType::GateEdge,
                                  x, rk_state, result);
                }
            }

            // 8. Record
            hook_accepted_(t_step0, t, step_fired ? x_pre_event : x, step_mask);
            ++step_idx;
            if (step_idx % store_every_ == 0) {
                ped_push_sample(result, t, x, system_.current_mask(),
                                step_mask,
                                step_fired ? &x_pre_event : nullptr);
            }
        }

        run_due_monitors_(t, x);   // instants inside the last step
        // Final state
        if (result.times.back() != t) {
            ped_push_sample(result, t, x, system_.current_mask(),
                            step_mask,
                            step_fired ? &x_pre_event : nullptr);
        }
        // Every run is checked (review of slice 5: gating it on the event
        // projection, off by default, skipped every plain C++ caller). A period
        // map run (periodic_steady_state, the orbit kernel) starts from Newton
        // trial states and is judged by its own residual: those callers set
        // set_diode_law_check(0).
        if constexpr (HasNodePairVoltage<System>) check_diode_law(result);

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

    // Test hooks: the step cap (default 10M) and the settle's round cap
    // (default 64). A small cap reaches the give-up deterministically on
    // any build and any rounding; production code leaves the defaults.
    void set_max_steps(std::size_t n) noexcept { max_steps_ = n; }
    void set_settle_round_cap(int n) noexcept { settle_round_cap_ = n < 1 ? 1 : n; }

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
    /// predicate fires only when its signal LEAVES the admissible side, so
    /// a bracket that starts past the threshold (by any amount) and comes
    /// back is never an event. Past it significantly, the mode was
    /// inconsistent: the mode decision's case (as a zero start leaving on
    /// its event side, zero_start_admissible_). Past it insignificantly,
    /// the value is rounding: after a commutation the projection leaves
    /// ~1e-14 A in an inductor in series with an OFF diode, which g_off
    /// turns into 1e-5 V relaxing in L·g_off (1e-13 s), and the start-up
    /// state reads 1e-21 V on the wrong side of a tie; firing on those
    /// returns re-decided the same mode (the chatter guard threw, or a
    /// burst of no-op events at 1e-17..5e-12 s). A signal that starts
    /// inside its significance bound on the event side and drifts further
    /// out is the scan's drift case, not a crossing. (This replaces the
    /// zero band that bounded the skip: no threshold is needed.)
    [[nodiscard]] static bool starts_admissible_(PredicateType t, Real ga) {
        return !(ga * static_cast<Real>(admissible_sign_(t)) < Real{0});
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
    /// add-unified-event-queue group 3: both come from the run's queue
    /// (`start_time_events_`). `t_sw` is the switch function's part (the
    /// gate edge, its poll, or `t_end`), which tells a corner apart.
    [[nodiscard]] Real next_gate_edge_(Real t_now, Real t_end, Real& t_sw) {
        // Both sources are asked again at every step, as before the queue:
        // an SPWM oracle's answer moves by an ulp with the query point
        // (DECISIONS 9), so a cached edge would move the trajectory.
        t_sw = t_end;
        if constexpr (HasNextEdge<SwitchFn>) {
            queue_.invalidate(gate_src_);
            const Real raw = queue_.next_after(gate_src_, t_now);
            if (std::isfinite(raw)) {
                t_sw = std::min(raw, t_end);
            } else {
                // A gate that names no edge is sampled every dt_max/10 and
                // re-read at each landing; without it a plain Python gate
                // ran with its t = 0 mask frozen.
                t_sw = std::min(t_now + dt_max_ * Real{0.1}, t_end);
            }
        }
        if constexpr (requires { system_.next_source_corner_after(t_now); }) {
            queue_.invalidate(corner_src_);
            return std::min(t_sw, queue_.next_after(corner_src_, t_now));
        }
        return t_sw;
    }

    // ---- control ticks on the queue (add-unified-event-queue group 3) ----
    void start_time_events_(Real t_end) {
        queue_ = events::EventQueue{};
        n_tick_src_ = static_cast<std::uint32_t>(ticks_.rates.size());
        for (std::size_t j = 0; j < ticks_.rates.size(); ++j) {   // ids 0..n-1 = j
            const bool mon = j < ticks_.monitor.size() && ticks_.monitor[j];
            queue_.add_periodic(mon ? events::Kind::Monitor : events::Kind::ControlTick,
                                ticks_.rates[j], /*lands=*/!mon);
        }
        if constexpr (HasNextEdge<SwitchFn>) {
            gate_src_ = queue_.add_function(events::Kind::GateEdge,
                [this](Real s) { return switch_fn_.next_edge_after(s); });
        }
        if constexpr (requires { system_.next_source_corner_after(Real{0}); }) {
            corner_src_ = queue_.add_function(events::Kind::SourceBreakpoint,
                [this](Real s) { return system_.next_source_corner_after(s); });
        }
        queue_.start(Real{0}, t_end);
    }

    [[nodiscard]] static Real tick_tol_(Real t) noexcept {
        return std::max(Real{1e-12}, Real{8} * std::numeric_limits<Real>::epsilon() * std::abs(t));
    }

    [[nodiscard]] bool tick_due_(Real t) const noexcept {
        return n_tick_src_ > 0 && t >= queue_.next_of(events::Kind::ControlTick) - tick_tol_(t);
    }

    /// Runs every tick due at `t` (Kind order, then registration) with the
    /// step time and state; a tick may move the gate's edges, so the gate
    /// source asks its oracle again. True when any ran.
    bool run_due_ticks_(Real t, const Vector& x) {
        if (!tick_due_(t)) return false;
        const auto due = queue_.take_due(t, tick_tol_(t), events::kind_bit(events::Kind::ControlTick));
        for (const auto& ev : due) {
            ticks_.fn(ev.source, events::TickInfo{ev.t, ev.ordinal}, t, x);
        }
        if constexpr (HasNextEdge<SwitchFn>) queue_.invalidate(gate_src_);
        return !due.empty();
    }

    /// Monitors due at `t` (Kind::Monitor): called with the accepted state,
    /// nothing else happens (no landing, no gate sample, no FSAL reset).
    void run_due_monitors_(Real t, const Vector& x) {
        // a step may span several instants (monitors never cap it): take
        // them all; take_due gives one per source per call
        while (n_tick_src_ > 0 && t >= queue_.next_of(events::Kind::Monitor) - tick_tol_(t)) {
            for (const auto& ev : queue_.take_due(t, tick_tol_(t),
                                                  events::kind_bit(events::Kind::Monitor))) {
                ticks_.fn(ev.source, events::TickInfo{ev.t, ev.ordinal}, t, x);
            }
        }
    }

    /// An accepted step [t0, t] integrated under `mask`, ending in x (the
    /// left limit when an event ended it): the step hooks' commutation
    /// is the pre-event state, as on pwl and trbdf2.
    void hook_accepted_(Real t0, Real t, const Vector& x, const MaskT& mask) {
        if constexpr (std::is_same_v<MaskT, topology::SwitchStateMask>) {
            if (hooks_ != nullptr) hooks_->accepted(t0, t, x, mask);
        }
    }

public:
    /// Step hooks (events/step_hooks.hpp): on_accept per accepted step,
    /// on_commutation at every change of the integrated mask. Not owned.
    void set_hooks(events::StepHooks* hooks) noexcept { hooks_ = hooks; }

    /// Control ticks scheduled by the run's event queue: `fn(j, info, t, x)`
    /// with the reduced state x at each instant of `rates[j]` before t_end;
    /// the gate is sampled again after the ticks of an instant.
    void set_ticks(events::TickPlan plan) { ticks_ = std::move(plan); }

    /// Ticks taken per source in the last run.
    [[nodiscard]] std::vector<std::uint64_t> tick_counts() const {
        std::vector<std::uint64_t> n;
        for (std::uint32_t j = 0; j < n_tick_src_; ++j) n.push_back(queue_.taken(j));
        return n;
    }

private:

    /// The selector probes' horizon (DECISIONS #28): the next gate edge
    /// the switch function announces or the next source corner, else the
    /// run's end. A switch function that says it has no edge oracle
    /// (`has_edge_oracle()` false: its next_edge_after is a dt_max/10 poll
    /// boundary, not an edge, and it caches its search) adds no cap.
    [[nodiscard]] Real next_time_event_(Real t_now) const {
        // an edge a few ulps after t_now is skipped only when it is the one
        // just fired; a predicate located just before an edge that has not
        // fired yet keeps that edge as its horizon (#31)
        const auto past = [&](auto&& next) {
            Real r = next(t_now);
            if (std::isfinite(r) && near_time(r, t_now) && std::isfinite(last_time_event_t_)
                && near_time(r, last_time_event_t_))
                r = next(r);
            return r;
        };
        Real e = probe_t_end_;
        if constexpr (HasNextEdge<SwitchFn>) {
            bool oracle = true;
            if constexpr (requires { switch_fn_.has_edge_oracle(); })
                oracle = switch_fn_.has_edge_oracle();
            const Real raw = oracle ? past([&](Real s) { return switch_fn_.next_edge_after(s); })
                                    : std::numeric_limits<Real>::infinity();
            if (std::isfinite(raw)) e = std::min(e, raw);
        }
        if constexpr (requires { system_.next_source_corner_after(t_now); }) {
            e = std::min(e, past([&](Real s) { return system_.next_source_corner_after(s); }));
        }
        // a control tick may move the gate's edges: no probe looks past it
        return std::min(e, queue_.next_of(events::Kind::ControlTick));
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
        // the root's time: 4 ulp of t (what a time can resolve), floored at 1e-12 of
        // the step (the 40-bit bisection depth past which g moves inside B only)
        const Real t_tol = std::max(
            Real{4} * std::numeric_limits<Real>::epsilon() * std::abs(t1),
            Real{1e-12} * h);
        loc_t_tol_ = t_tol;

        const auto& preds = predictor_.predicates();
        for (std::size_t ip = 0; ip < preds.size(); ++ip) {
            const auto& p = preds[ip];
            if (p.required_bit >= 0 &&
                diode_bit_(p.aux_index) != (p.required_bit != 0)) {
                continue;
            }
            Real ga = p.value(t0, x0);
            ++result.n_predicate_evals;
            Real ta0 = t0;
            Size scan_from = 0;
            const SchedulerDiode* dio = admissible_sign_(p.type) != 0 ? diode_of_(p.aux_index) : nullptr;
            // A diode predicate's value is information only beyond its own
            // significance bound B (signal_bound_). An exact zero is decided by
            // the ladder (ladder_sign_): leaving on its event side it is the mode
            // decision's case (never a zero-length event); a significant
            // admissible rung starts the scan there. Otherwise the scan reads the
            // predicate as it is, and inside B: a sign change between two
            // insignificant values is an event only if the first significant
            // value after it lies past the threshold (ladder from the bracket's
            // end); a value going from insignificant to significantly past the
            // threshold with no sign change (a drift out of zero) is an event
            // where it leaves the band, g = -s*B.
            const int s = admissible_sign_(p.type);
            Real s0 = Real{-1}, v0 = Real{0};
            const auto bound_at = [&](Real tau, const Vector& xx) {
                if (s0 < Real{0}) {
                    s0 = std::max(signal_scale_(*dio, t0, x0), signal_scale_(*dio, t1, x1));
                    v0 = std::max(signal_abs_(*dio, t0, x0), signal_abs_(*dio, t1, x1));
                }
                return signal_bound_(*dio, tau, xx, s0, v0);
            };
            std::optional<Real> root;
            if (dio && ga == Real{0}) {
                if (static_cast<int>(ip) == last_fired_index_ && t0 == last_fired_t_) continue;
                Real dj = Real{0};
                const int sg = ladder_sign_(t0, x0, *dio, decision_cap_(t0),
                    [&](Real tau, const Vector& xx) {
                        ++result.n_predicate_evals;
                        return p.value(tau, xx);
                    }, dj);
                if (sg == -s) continue;
                {
                    // the deciding rung, or the cap when undecided (nothing
                    // distinguishable from zero before it)
                    const Real tstart = t0 + dj;
                    if (!(tstart < t_at(n_scan))) continue;
                    // the first scan point past the deciding rung
                    Size k0 = 0;
                    while (k0 + 1 < n_scan && !(t_at(k0 + 1) > tstart)) ++k0;
                    advance_(t0, x0, tstart - t0, x_tmp_);
                    ++result.n_predicate_evals;
                    ga = p.value(tstart, x_tmp_);
                    ta0 = tstart;
                    scan_from = k0;
                }
            } else if (ga == Real{0}) {
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
            for (Size k = scan_from; k < n_scan && !root; ++k) {
                const Real ta = k == scan_from ? ta0 : t_at(k);
                const Real tb = t_at(k + 1);
                const Real gb = p.value(tb, x_at(k + 1));
                ++result.n_predicate_evals;
                auto g_at = [&](Real tau) -> Real {
                    advance_(t0, x0, tau - t0, x_tmp_);
                    ++result.n_predicate_evals;
                    return p.value(tau, x_tmp_);
                };
                const auto xa = [&]() -> const Vector& {
                    return ta == t_at(k) ? x_at(k) : x_tmp_;
                };
                // a crossing (or an exact zero) between two values inside B is
                // confirmed by the first significant value after it
                const auto confirmed = [&] {
                    if (!dio || !(std::abs(ga) <= bound_at(ta, xa()))
                        || !(std::abs(gb) <= bound_at(tb, x_at(k + 1))))
                        return true;
                    Real dj = Real{0};
                    return ladder_sign_(tb, x_at(k + 1), *dio, decision_cap_(tb),
                        [&](Real tau, const Vector& xx) {
                            ++result.n_predicate_evals;
                            return p.value(tau, xx);
                        }, dj) == -s;
                };
                if (ga * gb < Real{0}) {
                    // coming back from past the threshold is no event
                    if (starts_admissible_(p.type, ga) && confirmed())
                        root = locate_root_bracket(g_at, ta, tb, ga, gb, t_tol);
                } else if (gb == Real{0}) {
                    // zero to zero is no crossing (a diode at rest)
                    if (ga != Real{0} && starts_admissible_(p.type, ga) && confirmed())
                        root = tb;
                } else if (dio && static_cast<Real>(s) * gb < Real{0}
                           && !(static_cast<int>(ip) == last_fired_index_ && ta == last_fired_t_)) {
                    const Real ba = bound_at(ta, xa());
                    if (std::abs(ga) <= ba && std::abs(gb) > bound_at(tb, x_at(k + 1))) {
                        const Real off = static_cast<Real>(s) * ba;
                        const Real ga_s = ga + off;
                        root = ga_s == Real{0} ? ta
                            : locate_root_bracket([&](Real tau) { return g_at(tau) + off; },
                                                  ta, tb, ga_s, gb + off, t_tol);
                    }
                }
                ga = gb;
            }
            if (!root) continue;
            // 1e-15 s: one instant for two predicates' roots (≥ 1e3·t_tol at t ≤ 1 s,
            // far below any resolved time scale): the priority decides between them
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

    /// After an event that changed the mode or a source's piece, the
    /// exact path also stops at the instants that resolve the new mode's
    /// fast transient (exact_fast_samples): a mode decaying in 14 ns read
    /// through a 200 ns output step was drawn as one ramp, and the
    /// harness's interpolated measure read it 3 % of the span off while
    /// every sample was exact. Extra samples only; no state is touched.
    void plan_fast_samples_(Real t_e, bool changed) {
        if constexpr (requires { system_.exact_fast_samples(t_e, Real{}, fast_t_); }) {
            if (!changed) return;   // a dropped same-mask edge keeps the plan
            fast_k_ = 0;
            fast_t_.clear();
            if (system_.has_exact_step()) {
                system_.exact_fast_samples(t_e, dt_max_, fast_t_);
            }
        }
    }

    /// Consistent reinitialisation (event_projection.hpp), through the
    /// System's cached eigenbasis when it has one.
    /// The dependent-state records of the settle's projections, which a
    /// reset to shadow_x_ undoes (the arbiter's mode projects again).
    [[nodiscard]] std::size_t reinit_log_size_() const {
        if constexpr (requires { system_.reinit_log_size(); }) return system_.reinit_log_size();
        return 0;
    }
    [[nodiscard]] bool reinit_tracked_() const {
        return requires { system_.truncate_reinit_log(std::size_t{0}); };
    }
    void truncate_reinit_log_(std::size_t n) {
        if constexpr (requires { system_.truncate_reinit_log(n); }) system_.truncate_reinit_log(n);
    }
    void undo_reinit_records_() {
        if constexpr (requires { system_.truncate_reinit_log(reinit_mark_); })
            system_.truncate_reinit_log(reinit_mark_);
    }

    void project_(Real t, Vector& x) {
        // dependent states first, whatever the event projection says: an
        // inconsistent state is not a state of the mode (DECISIONS,
        // dependent states)
        if constexpr (requires { system_.reinit_to_constraint(t, x); }) {
            system_.reinit_to_constraint(t, x);
        }
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
                const std::size_t mk = reinit_log_size_();
                held_x_ = x;
                project_(t, x);
                resettle_projected_(t, x, mask, held_x_, mk);
                return;
            }
            if (shadow_.enabled || guard_) shadow_x_ = x;
            reinit_mark_ = reinit_log_size_();
            settle_rounds_ = 0;
            // the settle found no mode; the guard's arbiter decides from the
            // held state (an LCP refusal there names the diode)
            const auto arbitrate = [&] {
                result.mode_guard.enabled = true;
                ++result.mode_guard.settle_caps;
                x = shadow_x_;
                undo_reinit_records_();
                diode_bits_ = prev_bits;
                mask = lcp_apply_(t, x, gates_prev, prev_bits, rec, result.mode_guard.lcp);
                const std::size_t mk = reinit_log_size_();
                project_(t, x);
                resettle_projected_(t, x, mask, shadow_x_, mk);
            };
            try {
                settle_and_project_(t, x, mask);
            } catch (const ConvergenceFailure& e) {
                // the settle's round cap: with the guard on, the LCP decides (#25)
                if (!guard_ || guard_refuses_ || e.kind() != ConvergenceKind::Settle) throw;
                arbitrate();
                return;
            } catch (const std::runtime_error& e) {
                // A settle round reached a mode the extraction cannot pose (a node
                // left with no conductance at all): counted as a settle cap. A
                // g_off = 0 freewheel behind a g_off = 0 series diode: the cut-set
                // took the current to 0, the series diode then turned OFF and its
                // node floated (G_aa singular); the LCP hands the current to D.
                if (!guard_ || guard_refuses_
                    || std::string_view{e.what()}.find("G_aa is singular")
                           == std::string_view::npos)
                    throw;
                arbitrate();
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
            [&](Real h, const std::vector<char>& anchor, DenseMatrix& Z, Vector& u) {
                auto m = gates_prev;   // posed in the anchor mode (DECISIONS #30)
                for (std::size_t j = 0; j < diodes_.size(); ++j)
                    m.set(diodes_[j].switch_idx, anchor[j] != 0);
                system_.diode_port_problem(m, t, x_held, h, shadow_nodes_, Z, u);
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
        MaskT m = gates_prev.overlay(diode_bits_, diode_owned_);
        system_.set_mask(m);
        release_band_ties_(t, x_held, m);
        return m;
    }

    /// A diode the LCP leaves OFF inside the zero band is a tie its
    /// trajectory decides (D4, as the settle's zero_band_on_): ON when its
    /// OFF-mode signal leaves the band upwards first. Gauss–Seidel, at most
    /// one pass per diode. A gated bridge from rest (sine through D1, D4)
    /// kept D1 OFF, its signal rising 3e4 V/s from exactly 0, while a pin
    /// held D4 ON: the predicate starts on its event side and never fires.
    void release_band_ties_(Real t, const Vector& x, MaskT& mask)
        requires HasNodePairVoltage<System> {
        // The band on the circuit's scale (sources included): from rest every node
        // is 0 V and zero_band_tol_ is 0, so a rounding-level look-ahead (1e-20 V)
        // turned a bridge diode ON (rounding fuzz: D2 of a tied bridge, 11 of 21).
        const Real tol = tie_band_(t, x);
        for (std::size_t pass = 0; pass < diodes_.size(); ++pass) {
            bool flipped = false;
            for (const auto& d : diodes_) {
                if (diode_bits_.get(d.switch_idx)) continue;
                const Real sig = models::SwitchedDiode::crossing_signal(
                    false, system_.node_pair_voltage(d.from, d.to, t, x), d.params());
                if (!(std::abs(sig) <= tol) || !zero_band_on_(t, x, d, mask, tol, false))
                    continue;
                diode_bits_.set(d.switch_idx, true);
                mask = mask.overlay(diode_bits_, diode_owned_);
                system_.set_mask(mask);
                flipped = true;
            }
            if (!flipped) break;
        }
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
        int c = judge_(cur, settled, t, x, sc, worst, &worst_j);
        system_.set_mask(cur);
        if (c == 1) {
            // The projection jumps over the mode's fast transient, so a law broken
            // only inside it is invisible from x: judge again from the held state
            // (constraint re-initialised, no slow projection). The forward converter
            // kept its rectifier ON at gate-off: consistent at t+ and after the
            // projection, which drained L_o (2.7 A) through 2 MΩ of g_off in 23 ps.
            held_x_ = shadow_x_;
            const std::size_t mk = reinit_log_size_();
            if constexpr (requires { system_.reinit_to_constraint(t, held_x_); })
                system_.reinit_to_constraint(t, held_x_);
            truncate_reinit_log_(mk);
            Real w2 = Real{0};
            int j2 = -1;
            const int c2 = judge_(cur, settled, t, held_x_, sc, w2, &j2);
            system_.set_mask(cur);
            if (c2 == 0) { c = 0; worst = w2; worst_j = j2; }
            // An impulsive re-initialisation (a cut-set or capacitor loop entered
            // from an inconsistent state) is where the value law cannot see a
            // diode's bias: the constrained recovery assumes the constraint. The
            // LCP's companion model sees the impulse; it confirms or repairs.
            // (A freewheel diode with g_off = 0 stayed OFF and the inductor's
            // current was cut, ΔW = ½·L·i², at every gate-off.)
            // At a located diode event (settle_decisions_) the constraint's jump is
            // the root's error, not an impulse (bridge3_overlap: a cut-set record of
            // 1.5e-18 J sent the LCP to the held state, which kept Dbb ON).
            else if (!settle_decisions_ && reinit_log_size_() > reinit_mark_) {
                c = 0;
                worst_j = -1;
            }
        }
        if (c < 0) { ++g.unchecked; return; }
        ++g.checks;
        if (c != 0) return;
        const std::string who = worst_j >= 0
            ? diodes_[static_cast<std::size_t>(worst_j)].label : std::string{};
        const std::string what = who.empty()
            ? std::string{"the settled diode mode is entered through an impulsive "
                          "re-initialisation"}
            : "the settled diode mode violates the law of " + who
                  + " on its own trajectory (by " + sci_(worst) + " V)";
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
                "dsed at t = " + sci_(t) + " s: " + what + ", and the "
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
                "dsed at t = " + sci_(t) + " s: " + what + " and the "
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
        release_band_ties_(t, shadow_x_, mask);
        x = shadow_x_;
        undo_reinit_records_();
        const std::size_t mk = reinit_log_size_();
        project_(t, x);
        resettle_projected_(t, x, mask, shadow_x_, mk);
    }

    /// After the LCP's mode is projected: the projection jumps over the mode's
    /// fast transient, and a diode whose bias the jump reverses beyond the band
    /// is flipped by its value law, then projected again (at most four rounds;
    /// a fifth flip refuses). Inside the LCP's band the LCP decides: the settle's
    /// cascade here cycled 64 rounds on a bridge at rest (t = 0, a 1e-15
    /// perturbation). The forward converter's repair (rectifier OFF at gate-off)
    /// collapsed the open transformer's current, leaving the rectifier
    /// forward-biased by the freewheel drop (0.0137 V) for the whole interval.
    /// Records: those of the final mode alone, re-initialised once from
    /// `x_held` (as settle_and_project_ does for a trial mode).
    void resettle_projected_(Real t, Vector& x, MaskT& mask, const Vector& x_held,
                             std::size_t mark)
        requires HasNodePairVoltage<System> {
        bool changed = false;
        // An ON diode flips only on a violation on the circuit's scale (1e-6 of its sources and drops; a
        // held node can sit at MV behind g_off): at a zero crossing the LCP's
        // companion over h_e decides, not the value at the instant. A voltage
        // doubler's D2, ON at −1.05e-8 V and −2.9e-8 V where the LCP put it (its tol_w
        // 1.2e-8 V), was flipped and chattered 32 times. The forward converter's
        // rectifier is off by 0.0137 V.
        // (A reverse-current bound in its place failed both ways, DECISIONS #35.)
        Real scale = Real{1};
        if constexpr (requires { system_.port_scale_full(Vector{}); })
            scale = system_.port_scale_full(Vector{}).vscale;
        const Real tol_w = Real{1e-6} * scale;
        scale_floor_x_ = &x_held;   // signal_bound_ on a projection residue (settle_and_project_)
        const FloorReset_ reset_floor{scale_floor_x_};
        for (int pass = 0; pass <= 4; ++pass) {
            const Real band = tie_band_(t, x);
            std::string flipped;
            for (const auto& d : diodes_) {
                const bool on = diode_bits_.get(d.switch_idx);
                // An OFF diode past its threshold sits on its predicate's event side:
                // no event will ever turn it ON, so only the rounding band protects it.
                // A tied bridge (V_f = 0.7) left D1 OFF 3e-7 V above V_f, rising,
                // and it stayed OFF while forward-biased by 1.1 V.
                const Real tol = on ? std::max(band, tol_w) : band;
                const Real v_d = system_.node_pair_voltage(d.from, d.to, t, x);
                const auto prm = d.params();
                const bool by_value = models::SwitchedDiode::decide(on, v_d, prm);
                if (by_value == on
                    || std::abs(models::SwitchedDiode::crossing_signal(on, v_d, prm)) <= tol)
                    continue;
                // confirmed on the OFF-mode trajectory, as the settle confirms a flip
                if (zero_band_on_(t, x, d, mask, tol, by_value) == on) continue;
                if (pass == 4) { flipped = d.label; break; }
                diode_bits_.set(d.switch_idx, !on);
                flipped = d.label;
            }
            if (flipped.empty()) break;
            if (pass == 4)
                throw UnsupportedError(
                    "dsed", "consistent_mode", flipped, "Diode",
                    "dsed at t = " + sci_(t) + " s: the selected diode mode, projected, "
                    "still breaks the law of " + flipped + " after four re-settles");
            changed = true;
            mask = mask.overlay(diode_bits_, diode_owned_);
            system_.set_mask(mask);
            project_(t, x);
        }
        if (changed && reinit_tracked_() && reinit_log_size_() > mark) {
            // an abandoned mode's impulsive jump is not physics (settle_and_project_)
            truncate_reinit_log_(mark);
            x = x_held;
            project_(t, x);
        }
    }

public:
    /// Relative threshold of the end-of-run diode-law check (0: off).
    void set_diode_law_check(Real rel) noexcept { law_rel_ = rel; }

    /// Every stored sample, and every stored left limit at an event under
    /// the mask that produced it (step_masks), against the PWL diode law: ON
    /// with i = g_on·(v − V_f) < −max(rel·i_pk, band·g_on), or OFF with
    /// v > V_on + max(rel·v_pk, band), band = zero_band_tol_ of the sample
    /// (the rule python's check_diode_modes applied, moved here so a C++
    /// caller is covered too). Throws DiodeLawViolation naming the first.
    void check_diode_law(const PEDResult<MaskT>& result)
        requires HasNodePairVoltage<System> {
        if (!has_diodes_ || !(law_rel_ > Real{0}) || diodes_.empty()) return;
        // (sample index, state, mask): the samples, then the left limits
        // a left limit reads its sources' left limit too (a PWM corner at t),
        // value_at(t − left_by) with left_by half the step that ended there,
        // as the stored outputs do; the last sample is the left limit of t_end
        struct Row { std::size_t k; const Vector* x; const MaskT* m; Real left_by; };
        const std::size_t ns = result.times.size();
        const auto half_step = [&](std::size_t k) {
            return k > 0 ? Real{0.5} * (result.times[k] - result.times[k - 1]) : Real{0};
        };
        // an event on the last step stores the post-event state there: its
        // sources are the right limit, its left limit is the pre-event row
        const bool last_has_left = ns > 0 && !result.pre_event_index.empty()
                                   && result.pre_event_index.back() + 1 == ns;
        std::vector<Row> rows;
        rows.reserve(result.times.size() + result.pre_event_states.size());
        std::vector<const Vector*> xk(ns, nullptr);
        for (std::size_t k = 0; k < ns; ++k) xk[k] = &result.states[k];
        for (const auto& [k, xs] : law_settled_)   // D1: the state the decision read
            if (k < ns) xk[k] = &xs;
        for (std::size_t k = 0; k < result.times.size(); ++k)
            rows.push_back({k, xk[k], &result.sample_masks[k],
                            k + 1 == ns && !last_has_left ? half_step(k) : Real{0}});
        for (std::size_t j = 0; j < result.pre_event_states.size()
                                && j < result.pre_event_index.size(); ++j) {
            const std::size_t k = result.pre_event_index[j];
            if (k < result.step_masks.size())
                rows.push_back({k, &result.pre_event_states[j], &result.step_masks[k], half_step(k)});
        }
        const std::size_t n = rows.size(), nd = diodes_.size();
        std::vector<Real> v(n * nd), band(n * nd);
        // floors (1e-12 A / V) only keep law_rel_·peak positive on an idle diode;
        // there the significance band band[k·nd + j] is the larger term and decides
        std::vector<Real> ipk(nd, Real{1e-12}), vpk(nd, Real{0});
        const MaskT m_end = system_.current_mask();
        for (std::size_t k = 0; k < n; ++k) {
            const MaskT& m = *rows[k].m;
            if (!(system_.current_mask() == m)) system_.set_mask(m);
            const Real t = result.times[rows[k].k];
            const Vector& x = *rows[k].x;
            const Real zb = zero_band_tol_(t, x);
            for (std::size_t j = 0; j < nd; ++j) {
                const auto& d = diodes_[j];
                Real vd;
                if constexpr (requires { system_.node_pair_voltage(d.from, d.to, t, x, Real{0}); })
                    vd = system_.node_pair_voltage(d.from, d.to, t, x, rows[k].left_by);
                else
                    vd = system_.node_pair_voltage(d.from, d.to, t, x);
                v[k * nd + j] = vd;
                // the decision's own significance bound B (review s16 #2): a law
                // broken inside it is one no decision of this engine can resolve
                band[k * nd + j] = std::max(zb, signal_bound_(d, t, x, Real{0}, Real{0}));
                const bool on = m.get(d.switch_idx);
                const Real i = on ? d.g_on * (vd - d.V_f) : d.g_off * vd;
                ipk[j] = std::max(ipk[j], std::abs(i));
                if (!on) vpk[j] = std::max(vpk[j], std::abs(vd));
            }
        }
        if (!(system_.current_mask() == m_end)) system_.set_mask(m_end);
        for (std::size_t j = 0; j < nd; ++j) {
            const auto& d = diodes_[j];
            const Real thr = std::max(d.V_th, d.V_f);
            const Real vp = std::max({vpk[j], thr, Real{1e-12}});
            std::size_t bad = 0, first = n;
            for (std::size_t k = 0; k < n; ++k) {
                const bool on = rows[k].m->get(d.switch_idx);
                const Real vd = v[k * nd + j];
                const bool b = on
                    ? d.g_on * (vd - d.V_f) < -std::max(law_rel_ * ipk[j], band[k * nd + j] * d.g_on)
                    : vd > thr + std::max(law_rel_ * vp, band[k * nd + j]);
                if (b) {   // the earliest row (samples precede left limits in rows)
                    ++bad;
                    if (first == n || result.times[rows[k].k] < result.times[rows[first].k]) first = k;
                }
            }
            if (bad == 0) continue;
            const bool on = rows[first].m->get(d.switch_idx);
            const Real vd = v[first * nd + j];
            const std::string what = on
                ? "ON while carrying " + sci_(d.g_on * (vd - d.V_f)) + " A"
                : "OFF while forward-biased by " + sci_(vd) + " V";
            throw DiodeLawViolation(
                d.label, result.times[rows[first].k],
                "simulate(engine='dsed'): diode '" + d.label + "' is " + what + " at t = "
                + sci_(result.times[rows[first].k]) + " s (" + std::to_string(bad) + " samples break "
                "the diode law): the event settle chose a wrong mode, and the run would be "
                "silently wrong. Give the diode a small threshold band (V_th = 1e-2 V), or use "
                "engine='pwl' or 'trbdf2'.");
        }
    }

private:
    Real law_rel_ = Real{1e-3};

    /// The zero band with a floor on the circuit's voltage scale (sources
    /// included): from rest every node is 0 V and zero_band_tol_ is 0, so a
    /// rounding-level signal (1e-20 V) counted as a decided one.
    [[nodiscard]] Real tie_band_(Real t, const Vector& x) const
        requires HasNodePairVoltage<System> {
        return std::max(zero_band_tol_(t, x), Real{1e-12} * system_.port_scale(t, x).vscale);
    }

    void settle_and_project_(Real t, Vector& x, MaskT& mask) {
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_) {
                const std::size_t mark = reinit_log_size_();
                const bool track = reinit_tracked_();
                settle_x0_ = x;
                mask = settle_diode_cascade_(t, x);
                project_(t, x);
                bool changed = false;
                const FloorReset_ reset_floor{scale_floor_x_};
                for (int pass = 0; pass < 4; ++pass) {
                    scale_floor_x_ = &settle_x0_;
                    const MaskT m2 = settle_diode_cascade_(t, x);
                    scale_floor_x_ = nullptr;
                    if (m2 == mask) {
                        break;
                    }
                    mask = m2;
                    changed = true;
                    project_(t, x);
                }
                if (track && changed && reinit_log_size_() > mark) {
                    // a trial mode's projection left records behind: that jump is
                    // not physics, so the settled mode starts again from the
                    // pre-event state (its own records alone). Keeping the iterated
                    // state carried a cut-set trial's i_L = 0 into the freewheel
                    // mode it settled in, with no record (the decision on the
                    // projection's 1e-18 A residue is rounding: MSVC, R = 10·(1−u))
                    truncate_reinit_log_(mark);
                    x = settle_x0_;
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
            if (g0 * g1 >= Real{0} || !starts_admissible_(p.type, g0)) continue;

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

            // 1e-15 s: the same instant as in locate_event_exact_ (priority decides)
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
        if (ptype == PredicateType::GateEdge) last_time_event_t_ = t_event;   // next_time_event_

        // Notify switch_fn of non-gate (diode/ZCD) events so it can
        // latch the post-event mode for the remainder of this cycle.
        if constexpr (HasZcdRegister<SwitchFn>) {
            if (ptype == PredicateType::CurrentZC
                || ptype == PredicateType::DiodeOff
                || ptype == PredicateType::DiodeOn) {
                switch_fn_.register_zcd_transition(t_event);
            }
        }

        // The root's error as a state error (x along the pre-event mode over
        // the locator's tolerance): the decisions' significance includes it.
        loc_dx_.resize(0);
        if constexpr (HasExactStep<System>) {
            if ((ptype == PredicateType::DiodeOn || ptype == PredicateType::DiodeOff)
                && loc_t_tol_ > Real{0} && system_.has_exact_step()) {
                advance_(t_event, x, loc_t_tol_, loc_dx_);
                loc_dx_ -= x;
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
        settle_decisions_ = ptype == PredicateType::DiodeOn
                            || ptype == PredicateType::DiodeOff;
        decide_modes_(t_event, x, final_mask,
                      effective_gates_(new_mask, prev_bits), prev_bits,
                      old_mask, result);
        // D1: the law check reads, at a located diode event's right-limit sample,
        // the state the decision read: the settled mode after its parasitic
        // transient (~L·g_off, far below anything the run resolves; read
        // un-relaxed, the check depends on rounding). The stored sample stays
        // the exact map's state at t (sensitivities and period integrals
        // propagate from it and attach the event to its own sample).
        has_event_sample_ = false;
        if (settle_decisions_) {
            Real ts = t_event;
            const Vector& xs = settled_(t_event, x, event_sample_, ts);
            // only where the stored sample breaks a law at all (D1's case: a sample
            // that holds, keeps being read as stored), and only where the break is
            // not one that existed BEFORE the event (review s16 #1): a real current
            // forced into g_off drains inside the settle too, so a diode that already
            // broke the new mode's law under the old mode, beyond the old mode's bound
            // plus the located root's own error, is checked on the stored sample.
            if (&xs == &event_sample_ && ts > t_event) {
                bool breaks = false, real = false;
                if constexpr (HasNodePairVoltage<System>) {
                    const MaskT post = system_.current_mask();
                    for (const auto& d : diodes_) {
                        const bool on = post.get(d.switch_idx);
                        if (!breaks_law_(d, on, t_event, x)) continue;
                        breaks = true;
                        system_.set_mask(old_mask);
                        const bool was_on = old_mask.get(d.switch_idx);
                        // the new mode's law read on the old mode (an ON diode's
                        // drop is V_f in both: its current in volts of v_D)
                        const Real pre = law_signal_(d, on, t_event, x, was_on);
                        Real err = signal_bound_(d, t_event, x, Real{0}, Real{0});
                        if (loc_dx_.size() == x.size()) {
                            x_tmp_ = x + loc_dx_;
                            err += std::abs(law_signal_(d, on, t_event, x_tmp_, was_on) - pre);
                        }
                        system_.set_mask(post);
                        if (pre > err) { real = true; break; }
                    }
                }
                has_event_sample_ = breaks && !real;
            }
        }
        settle_decisions_ = false;
        loc_dx_.resize(0);

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

    /// Decisions at an event are taken on the candidate mode's slow
    /// manifold: x carried exactly through the current mode's parasitic
    /// transient (exact_settle_length over resolved_scale_), inside
    /// which no predicate is read. A diode left OFF at its located current
    /// zero keeps the root's residual ε in its inductor, which g_off reads
    /// as ε/g_off volts (kV at µA) for ~L·g_off: its sign is rounding. The
    /// settle is zero-duration for the run (the carried state is not
    /// moved; the event projection removes the same modes): the returned
    /// time t + δ is the one the settled state is read at. Returns x and t
    /// when the mode has no parasitic cluster.
    /// Only at an event located as a diode predicate's zero
    /// (settle_decisions_): there the transient's amplitude is the root's
    /// error. At a time event (gate edge, t = 0) it is the physics: a
    /// buck's inductor current forced into g_off at gate-off is what turns
    /// the freewheel diode ON.
    const Vector& settled_(Real t, const Vector& x, Vector& out, Real& ts) {
        ts = t;
        if (!settle_decisions_) return x;
        if constexpr (HasExactStep<System>) {
            if constexpr (requires { system_.exact_settle_length(Real{}); }) {
                if (system_.has_exact_step()) {
                    Real d = system_.exact_settle_length(resolved_scale_(t));
                    // never read past the next scheduled event: the gate edge is
                    // the physics that comes next (a cap shorter than the decay
                    // decides on the state at the cap)
                    const Real te = next_time_event_(t);
                    // half-way, as decision_cap_: the ladder read from t + d must still
                    // see the edge ahead of it (from t + d = te it would skip it)
                    if (std::isfinite(te) && te > t) d = std::min(d, Real{0.5} * (te - t));
                    if (d > Real{0}) {
                        advance_(t, x, d, out);
                        ts = t + d;
                        return out;
                    }
                }
            }
        }
        return x;
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
        Real ts = t;
        const Vector* xs = &settled_(t, x, x_set_, ts);
        MaskT xs_mask = mask;
        const Real tol = zero_band_tol_(ts, *xs);
        const int kMaxRounds = settle_round_cap_;
        for (int round = 0; round < kMaxRounds; ++round) {
            ++settle_rounds_;
            bool flipped = false;
            for (const auto& d : diodes_) {
                const bool on =
                    diode_bits_.get(d.switch_idx);
                if (mask != xs_mask) {
                    xs = &settled_(t, x, x_set_, ts);
                    xs_mask = mask;
                }
                const Real v_d = system_.node_pair_voltage(
                    d.from, d.to, ts, *xs);
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
                    "settle after " + std::to_string(kMaxRounds) + " rounds at t = "
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
        // 1e-12 ≈ 4500 u: a recovery's rounding over ~10³ terms; the fallback of
        // signal_bound_ when the system states no bound of its own
        return Real{1e-12} * vs;
    }

    /// The significance ladder: a DEGENERATE diode decision (its signal zero
    /// to rounding at the instant: a state at rest, a current starting as t²)
    /// is decided by the sign of the signal's first non-vanishing derivative
    /// along the mode's own solution (the lexicographic condition of linear
    /// complementarity systems), read by exact propagation, not powers of A:
    /// g along the CURRENT mode from (t0, x0) at d = cap·10^(k−12), k = 0…12,
    /// and the sign of the first rung where |g(d)| exceeds its own rounding
    /// bound (signal_bound_). 0 when no rung does: g is zero on [t0, t0 + cap]
    /// as far as double precision can tell. `d_out`: the deciding rung (cap
    /// when none). One test for the locator's zero start, the start-up
    /// decision, the settle cascade, the LCP re-settle and the guard's repair.
    template <class G>
    int ladder_sign_(Real t0, const Vector& x0, const SchedulerDiode& d, Real cap,
                     G&& g_of, Real& d_out, const Vector* dx0 = nullptr)
        requires HasExactStep<System> {
        // dx0: the start's own error (a located root's): the signal moved by it
        // along the mode, |g(x0 + dx0) − g(x0)|, is not significant either
        if (dx0) x_offp_ = x0 + *dx0;
        // the trajectory's scale over the horizon: the propagated inputs'
        // rounding is relative to it, not to a state at rest
        advance_(t0, x0, cap, x_lad_);
        const Real s0 = std::max(signal_scale_(d, t0, x0), signal_scale_(d, t0 + cap, x_lad_));
        const Real v0 = std::max(signal_abs_(d, t0, x0), signal_abs_(d, t0 + cap, x_lad_));
        for (int k = 0; k <= 12; ++k) {
            const Real delta = cap * std::pow(Real{10}, Real(k - 12));
            advance_(t0, x0, delta, x_lad_);
            const Real g = g_of(t0 + delta, x_lad_);
            Real b_loc = Real{0};
            if (dx0) {
                advance_(t0, x_offp_, delta, x_lad2_);
                b_loc = std::abs(g_of(t0 + delta, x_lad2_) - g);
            }
            if (std::abs(g) > signal_bound_(d, t0 + delta, x_lad_, s0, v0) + b_loc) {
                d_out = delta;
                return g > Real{0} ? 1 : -1;
            }
        }
        d_out = cap;
        return 0;
    }

    /// |v_D| at (t, x), 0 when the system cannot state it.
    [[nodiscard]] Real signal_abs_(const SchedulerDiode& d, Real t, const Vector& x) const {
        if constexpr (HasNodePairVoltage<System>)
            return std::abs(system_.node_pair_voltage(d.from, d.to, t, x));
        else
            return Real{0};
    }

    /// |c|ᵀ(|R||x| + |r| + |R_b||b|) of a diode's voltage (rounding scale), or
    /// −1 when the system cannot state it.
    [[nodiscard]] Real signal_scale_(const SchedulerDiode& d, Real t, const Vector& x) const {
        if constexpr (requires { system_.node_pair_voltage_scale(d.from, d.to, t, x); })
            return system_.node_pair_voltage_scale(d.from, d.to, t, x);
        else
            return Real{-1};
    }

    /// The significance bound of a diode signal read at (t, x) on a trajectory
    /// whose recovery scale is s0 and whose signal scale is v0:
    ///   γ_n·(max(S(x), s0) + |V|) + e·(max(|v(x)|, v0, F) + |V|),
    /// γ_n = n·u the inner-product rounding of the recovery over its n terms
    /// (S = node_pair_voltage_scale), e the propagator's own stated relative
    /// error (exact_step_error) applied to the signal's magnitude along the
    /// trajectory and to its forced amplitude F (node_pair_voltage_forced: the
    /// sine particular solution the free response cancels at rest), V the
    /// diode's threshold. The zero band when the system cannot state it.
    [[nodiscard]] Real signal_bound_(const SchedulerDiode& d, Real t, const Vector& x,
                                     Real s0, Real v0) const {
        if constexpr (requires { system_.node_pair_voltage_terms();
                                 system_.exact_step_error();
                                 system_.node_pair_voltage_forced(d.from, d.to); }) {
            if (s0 >= Real{0}) {
                const Real u = std::numeric_limits<Real>::epsilon() / Real{2};
                const Real gn = static_cast<Real>(system_.node_pair_voltage_terms()) * u;
                const Real e = system_.exact_step_error();
                const Real V = std::max(std::abs(d.V_f), std::abs(d.V_th));
                const Real v = signal_abs_(d, t, x);
                // a projection residue (1e-18 A of a cut inductor) collapses S with
                // it: the floor is the same sum on the state before the projection
                Real S = std::max(signal_scale_(d, t, x), s0);
                if (scale_floor_x_) S = std::max(S, signal_scale_(d, t, *scale_floor_x_));
                return gn * (S + V)
                       + e * (std::max({v, v0, system_.node_pair_voltage_forced(d.from, d.to)}) + V);
            }
        }
        return zero_band_tol_(t, x);
    }

    /// A diode's law signal in volts, positive = the law is broken: ON (`on`)
    /// −(v_D − V_f) (its reverse current over g_on), OFF v_D − V_on. `ref_on`:
    /// the mode whose drop the OFF threshold reads (an ON diode read for the
    /// OFF law crosses at V_f, review s16 #1); defaults to `on`.
    [[nodiscard]] Real law_signal_(const SchedulerDiode& d, bool on, Real t, const Vector& x,
                                   std::optional<bool> ref_on = std::nullopt) const
        requires HasNodePairVoltage<System> {
        const Real v = system_.node_pair_voltage(d.from, d.to, t, x);
        if (on) return -(v - d.V_f);
        return v - (ref_on.value_or(on) ? d.V_f : std::max(d.V_th, d.V_f));
    }

    /// THE significance test of a diode's law (review s16 #2): broken iff its
    /// signal exceeds the decision's bound B (signal_bound_) or is not finite.
    /// One rule for the judge, the selector's horizon and the event sample.
    [[nodiscard]] bool breaks_law_(const SchedulerDiode& d, bool on, Real t, const Vector& x) const
        requires HasNodePairVoltage<System> {
        const Real g = law_signal_(d, on, t, x);
        return !(g <= signal_bound_(d, t, x, Real{0}, Real{0}));
    }

    /// The resolved time scale of a decision at t: the schedule's interval that
    /// holds t (the last gate edge / corner fired, or the run's start, to the next
    /// scheduled event or the run's end). Not the step ceiling dt_max: the same
    /// state at the same instant gets the same decision whatever the ceiling. At
    /// dt_max = T/50 the boost-DCM all-OFF mode's g_off rate (~5e12 /s) fell
    /// under the cut 1e6/dt_max, the turn-off was read on the unsettled state and
    /// D1 stayed ON in 18 of 25 runs (exact engine session 18).
    [[nodiscard]] Real resolved_scale_(Real t) const {
        const Real lo = std::isfinite(last_time_event_t_) ? last_time_event_t_ : Real{0};
        return next_time_event_(t) - std::min(lo, t);
    }

    /// The ladder's cap at a mode decision: half the time to the next scheduled
    /// event (the run's end at the latest), not the step ceiling (resolved_scale_).
    /// dt_max only at a decision on the run's last instant, where nothing is ahead.
    [[nodiscard]] Real decision_cap_(Real t) const {
        const Real te = next_time_event_(t);
        if (std::isfinite(te) && te > t) return Real{0.5} * (te - t);
        return dt_max_;
    }

    /// The scheduler diode of a predicate's switch index, or nullptr.
    [[nodiscard]] const SchedulerDiode* diode_of_(int switch_idx) const {
        for (const auto& d : diodes_)
            if (static_cast<int>(d.switch_idx) == switch_idx) return &d;
        return nullptr;
    }

    /// D4 for one diode in the zero band, or whose value law wants to
    /// flip it: ON iff the OFF mode drives
    /// its signal v_D − V_on above its significance bound. The signal is
    /// read on the exact OFF-mode trajectory by the ladder (ladder_sign_,
    /// rungs up to decision_cap_), from the settled state at a located
    /// diode event; the first significant rung decides, which is the sign
    /// of its first non-zero derivative. A signal that never becomes
    /// significant carries no current either way: OFF.
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
                (void)tol;
                const auto prm = d.params();
                Real t0 = t;
                const Vector& x0 = settled_(t, x, x_off_, t0);
                // at a located event the root's error, carried to the settled start
                const Vector* dx0 = nullptr;
                Vector dxs;
                if (settle_decisions_ && loc_dx_.size() == x.size()) {
                    if (t0 > t) {
                        advance_(t, x + loc_dx_, t0 - t, dxs);
                        dxs -= x0;
                    } else {
                        dxs = loc_dx_;
                    }
                    dx0 = &dxs;
                }
                Real dj = Real{0};
                const int sg = ladder_sign_(t0, x0, d, decision_cap_(t0),
                    [&](Real tau, const Vector& xx) {
                        return models::SwitchedDiode::crossing_signal(
                            false, system_.node_pair_voltage(d.from, d.to, tau, xx), prm);
                    }, dj, dx0);
                // no significant rung: zero to double precision up to the cap,
                // the diode keeps its mode (rest and exact ties stay quiet)
                on = sg > 0;
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
            // ONE JUDGE'S BOUND: the decision's signal (v_D − V_on, volts) against
            // the decision's significance bound B (signal_bound_), not a fraction of
            // the circuit's voltage scale: a repair inside B would second-guess the
            // decision with a different rule. The horizons stay past the parasitic
            // transient (a held state's law broken inside it is the guard's case).
            const Real zs = sc.vscale / sc.iscale;
            const Real cap = next_time_event_(t) - t;   // DECISIONS #28
            if (!(cap > Real{0})) return -1;            // nothing to judge before the event
            bool violated = false;
            Real worst_all = Real{0};
            for (const Real f : {Real{1e2}, Real{1e4}}) {
                const Real d = std::min(f * shadow_.h_e, cap);
                if (!(d > Real{0})) break;
                advance_(t, x, d, x_tmp_);
                for (std::size_t j = 0; j < diodes_.size(); ++j) {
                    const auto& dd = diodes_[j];
                    const Real sig = law_signal_(dd, mode[j] != 0, t + d, x_tmp_);
                    // a non-finite state (an unstable mode over 1e4·h_e) is a violation
                    const bool bad = breaks_law_(dd, mode[j] != 0, t + d, x_tmp_);
                    // reported in volts: ON -> −i·zs, OFF -> v − V_on
                    const Real viol = mode[j] ? dd.g_on * sig * zs : sig;
                    const Real v_eff = std::isfinite(viol) ? viol
                                                           : std::numeric_limits<Real>::infinity();
                    worst_all = std::max(worst_all, v_eff);
                    if (bad && (!violated || v_eff > worst)) {
                        violated = true;
                        worst = v_eff;
                        if (worst_j) *worst_j = static_cast<int>(j);
                    }
                }
            }
            if (!violated) worst = worst_all;   // the largest (insignificant) one
            return violated ? 0 : 1;
        } else {
            (void)gates; (void)mode; (void)t; (void)x; (void)sc; (void)worst_j;
            return -1;
        }
    }

    /// The selector's horizon (DECISIONS #24): the first d of
    /// 100·h_e·2^k, k = 0..7, at which a `decided` diode of `mode` breaks
    /// its own law on the exact trajectory (the judge's measure), +∞ if
    /// none up to 12 800·h_e or no exact step. The event lies in (d/2, d].
    /// The probe stops at the next time event (#28), and that event is
    /// returned when the mode holds up to it: the widening never looks
    /// past a gate edge or source corner.
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
            (void)sc;
            // the probe stops at the next time event: past it the gates or a
            // source's piece differ, so the trajectory does not happen (#28)
            const Real cap = next_time_event_(t) - t;
            Real d = Real{100} * shadow_.h_e;
            for (int k = 0; k < 8; ++k, d *= Real{2}) {
                if (d > cap) d = cap;
                if (!(d > Real{0})) return inf;
                advance_(t, x, d, x_tmp_);
                for (std::size_t j = 0; j < diodes_.size(); ++j) {
                    if (!decided[j]) continue;
                    // the judge's own test (breaks_law_; NaN counts as a departure)
                    if (breaks_law_(diodes_[j], mode[j] != 0, t + d, x_tmp_)) return d;
                }
                if (d == cap) return cap;   // the mode's trajectory ends at the time event
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
    Real probe_t_end_ = std::numeric_limits<Real>::infinity();   // next_time_event_
    std::size_t reinit_mark_ = 0;   // reinit_log_size_() before the settle
    Vector settle_x0_;   // settle_and_project_: the pre-event state (its records)
    Real last_time_event_t_ = std::numeric_limits<Real>::quiet_NaN();   // last gate edge / corner fired
    // add-unified-event-queue group 3: the run's time events (start_time_events_)
    events::EventQueue queue_;
    events::TickPlan ticks_;
    std::uint32_t n_tick_src_ = 0;
    std::uint32_t gate_src_ = 0;
    std::uint32_t corner_src_ = 0;
    events::StepHooks* hooks_ = nullptr;
    std::vector<Real> fast_t_;     // plan_fast_samples_
    std::size_t fast_k_ = 0;
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
    Vector held_x_;            // guard: the held state, constraint re-initialised
    int settle_rounds_ = 0;
    int settle_round_cap_ = 64;
    std::size_t max_steps_ = 10'000'000;
    // The predicate that fired the last predicate event, and when: only
    // it is skipped as "just fired" at a step starting at that instant.
    int last_fired_index_ = -1;
    Real last_fired_t_ = std::numeric_limits<Real>::quiet_NaN();
    // Exact-path workspaces: a step reuses them (no allocation).
    mutable Vector x_new_, x_evt_, x_tmp_;
    Vector x_lad_;             // ladder_sign_'s rung state
    const Vector* scale_floor_x_ = nullptr;   // signal_bound_'s pre-projection state
    struct FloorReset_ {   // a throw inside a settle leaves no dangling floor
        const Vector*& p;
        ~FloorReset_() { p = nullptr; }
    };
    Vector x_set_, x_off_;     // settled_: the cascade's and the OFF probe's
    bool settle_decisions_ = false;   // fire_event_: a located diode event
    Vector event_sample_;             // D1: the settled state stored at a located diode event
    bool has_event_sample_ = false;
    std::vector<std::pair<std::size_t, Vector>> law_settled_;   // D1: (sample, settled state)
    // The located root's time tolerance (locator) and the state error it
    // carries, x(t + t_tol) − x(t) in the pre-event mode (fire_event_).
    Real loc_t_tol_ = Real{0};
    Vector loc_dx_, x_lad2_, x_offp_;
    std::vector<Vector> scan_x_;
};

}  // namespace pulsim::dsed
