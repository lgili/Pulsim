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
#include <utility>
#include <vector>

#include "pulsim/dsed/event_predictor.hpp"
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

/// PED simulation output.
template <class MaskT>
struct PEDResult {
    std::vector<Real> times;
    std::vector<Vector> states;
    std::vector<EventRecord<MaskT>> event_log;
    std::size_t n_accept = 0;
    std::size_t n_reject = 0;
    std::size_t n_events = 0;
    Real cpu_time_seconds = Real{0};
};

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
    std::string label;        // named, Phase-1 style
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
        MaskT mask = effective_mask_(switch_fn_(t));
        system_.set_mask(mask);
        // A diode may be forward-biased already at t = 0 (a
        // rectifier powered on mid-cycle). Settle before the first
        // step rather than integrating an inconsistent mode.
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_) {
                settle_diode_cascade_(t, x);
            }
        }
        if constexpr (HasABInterface<System>) {
            if (enable_event_projection_) {
                x = project_onto_slow_manifold(
                    system_.A_matrix(), system_.b_vector(t), x,
                    Real{100} / dt_max_);
            }
        }

        RK45State rk_state;
        PEDResult<MaskT> result;
        result.times.push_back(t);
        result.states.push_back(x);
        std::size_t step_idx = 0;

        constexpr std::size_t kMaxSteps = 10'000'000;

        auto f = [this](Real tau, const Vector& xx) -> Vector {
            return system_.rhs(tau, xx);
        };

        while (t < t_end) {
            if (result.n_accept + result.n_reject > kMaxSteps) {
                throw std::runtime_error(
                    "PEDSimulator: exceeded max-step cap");
            }

            // 1. Next gate-edge time (analytical fast path)
            const Real t_gate = next_gate_edge_(t, t_end);
            const Real t_target = std::min(t_gate, t_end);

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
                    Vector x_new =
                        system_.exact_advance_state(t, x, h_ex);

                    std::optional<PredicateEvent> evt;
                    if (predictor_.size() > 0) {
                        evt = locate_event_exact_(t, x, h_ex);
                    }
                    if (evt.has_value()
                        && evt->t_event < t + h_ex - Real{1e-15}) {
                        t = evt->t_event;
                        x = std::move(evt->x_at_event);
                        fire_event_(t, evt->name, evt->type, x,
                                     rk_state, result,
                                     evt->aux_index);
                    } else {
                        t += h_ex;
                        x = std::move(x_new);
                        if (t_gate < t_end && near_time(t, t_gate)) {
                            fire_event_(t_gate, "gate_edge",
                                          PredicateType::GateEdge,
                                          x, rk_state, result);
                        }
                    }
                    rk_state.invalidate();
                    ++result.n_accept;
                    ++step_idx;
                    if (step_idx % store_every_ == 0) {
                        result.times.push_back(t);
                        result.states.push_back(x);
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
                    throw std::runtime_error(
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
                    throw std::runtime_error(
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
                x = std::move(x_pred);
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
                    fire_event_(t_gate, "gate_edge", PredicateType::GateEdge,
                                  x, rk_state, result);
                }
            }

            // 8. Record
            ++step_idx;
            if (step_idx % store_every_ == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
            }
        }

        // Final state
        if (result.times.back() != t) {
            result.times.push_back(t);
            result.states.push_back(x);
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
    };

    [[nodiscard]] Real next_gate_edge_(Real t_now, Real t_end) const {
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
    /// trajectory instead of the Hermite interpolant.
    [[nodiscard]] std::optional<PredicateEvent> locate_event_exact_(
        Real t0, const Vector& x0, Real h)
        requires HasExactStep<System> {
        std::optional<PredicateEvent> best;
        int best_priority = std::numeric_limits<int>::max();
        const Real t1 = t0 + h;
        const Vector x1 = system_.exact_advance_state(t0, x0, h);

        for (const auto& p : predictor_.predicates()) {
            if (p.required_bit >= 0 &&
                diode_bit_(p.aux_index) != (p.required_bit != 0)) {
                continue;
            }
            const Real g0 = p.value(t0, x0);
            const Real g1 = p.value(t1, x1);
            if (g0 == Real{0}) continue;
            if (g0 * g1 >= Real{0}) continue;

            auto g_at = [&](Real tau) -> Real {
                if (tau <= t0) return g0;
                if (tau >= t1) return g1;
                return p.value(tau,
                    system_.exact_advance_state(t0, x0, tau - t0));
            };
            Real t_root;
            try {
                t_root = illinois(g_at, t0, t1, g0, g1);
            } catch (const std::runtime_error&) {
                predictor_.note_illinois_failure();
                t_root = bisect_fallback(g_at, t0, t1);
            }
            const auto make_event = [&]() -> PredicateEvent {
                return PredicateEvent{
                    .t_event = t_root,
                    .name = p.name,
                    .type = p.type,
                    .x_at_event = system_.exact_advance_state(
                        t0, x0, t_root - t0),
                    .aux_index = p.aux_index,
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

        for (const auto& p : predictor_.predicates()) {
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
            const Real g0 = p.value(t0, x0);
            const Real g1 = p.value(t1, x1);

            // Skip predicates already at zero (just fired) or no sign change.
            if (g0 == Real{0}) continue;
            if (g0 * g1 >= Real{0}) continue;

            // Locate root via Illinois on the Hermite interpolant.
            auto g_at = [&p, &x0, &x1, &k1, &k7, h, t0, t1, g0, g1](Real tau) -> Real {
                if (tau <= t0) return g0;
                if (tau >= t1) return g1;
                const Real theta = (tau - t0) / h;
                const Vector x_tau = hermite_interp(x0, x1, k1, k7, h, theta);
                return p.value(tau, x_tau);
            };

            Real t_root;
            try {
                t_root = illinois(g_at, t0, t1, g0, g1);
            } catch (const std::runtime_error&) {
                predictor_.note_illinois_failure();
                t_root = bisect_fallback(g_at, t0, t1);
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
        if constexpr (HasNodePairVoltage<System>) {
            if (has_diodes_) {
                final_mask = settle_diode_cascade_(t_event, x);
            }
        }

        // Consistent reinitialization (Phase-3 item 2). The event
        // deposits x off the new mode's slow manifold — a µA of
        // root-location residual in a DCM inductor, volts of
        // difference across caps a switch just paralleled — and the
        // excited decay modes sit ~1e6 below any step the
        // integrator will take. Project them to quasi-static; slow
        // components (the conserved charges/fluxes) pass through
        // exactly. Identity whenever no fast stable mode exists.
        if constexpr (HasABInterface<System>) {
            if (enable_event_projection_) {
                x = project_onto_slow_manifold(
                    system_.A_matrix(), system_.b_vector(t_event), x,
                    /*fast_threshold=*/Real{100} / dt_max_);
            }
        }

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
            throw std::runtime_error(
                "PEDSimulator: " + diodes_[di].label
                + " is chattering at t = " + std::to_string(t)
                + " — it commutated 32 times inside "
                + std::to_string(window) + " s, which is a zero-"
                "current oscillation (discontinuous conduction), "
                "not switching. The dsed engine cannot yet hold a "
                "DCM idle mode: turning the diode off leaves a "
                "microamp interpolation residual that g_off "
                "reconstructs as kilovolts of forward bias, and "
                "the consistent-reinitialization projection that "
                "removes it is Phase-3 item 2. Use the default "
                "engine (engine='pwl'), whose implicit fixed-step "
                "solver holds DCM correctly.");
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
    /// instant. Returns the settled mask (already set on the
    /// system). Budget-bounded; on exhaustion, throws NAMING the
    /// devices still flipping — the Phase-1 standard.
    MaskT settle_diode_cascade_(Real t, const Vector& x)
        requires HasNodePairVoltage<System> {
        MaskT mask = system_.current_mask();
        constexpr int kMaxRounds = 64;
        for (int round = 0; round < kMaxRounds; ++round) {
            bool flipped = false;
            for (const auto& d : diodes_) {
                const bool on =
                    diode_bits_.get(d.switch_idx);
                const Real v_d = system_.node_pair_voltage(
                    d.from, d.to, t, x);
                const Real g = on ? d.g_on : d.g_off;
                const models::SwitchedDiode::Params params{
                    d.g_on, d.g_off, d.V_th};
                // The pwl engine's exact decision rule — the two
                // engines must not drift on what "conducting" means.
                const bool next =
                    models::SwitchedDiode::decide_next_state(
                        on, v_d, g * v_d, params);
                if (next != on) {
                    diode_bits_.set(d.switch_idx, next);
                    flipped = true;
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
                throw std::runtime_error(
                    "PEDSimulator: diode configuration did not "
                    "settle after 64 rounds at t = "
                    + std::to_string(t) + " (diodes: " + culprits
                    + ") — a pair is fighting over the same node; "
                    "add a small hysteresis band or an RC snubber");
            }
            const MaskT candidate =
                mask.overlay(diode_bits_, diode_owned_);
            if (candidate != mask) {
                mask = candidate;
                system_.set_mask(mask);
            }
        }
        return mask;
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
};

}  // namespace pulsim::dsed
