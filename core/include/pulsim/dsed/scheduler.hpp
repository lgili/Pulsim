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
#include "pulsim/dsed/rk45_dormand_prince.hpp"
#include "pulsim/dsed/step_controller.hpp"
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
        MaskT mask = switch_fn_(t);
        system_.set_mask(mask);

        RK45State rk_state;
        PEDResult<MaskT> result;
        result.times.push_back(t);
        result.states.push_back(x);
        std::size_t step_idx = 0;

        constexpr std::size_t kMaxSteps = 10'000'000;
        constexpr Real kBacktrackTol = Real{1e-15};

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

            // 5. PI accept/reject
            auto [accepted, h_next] = controller_.accept(err, x, x_new, h_use);
            if (!accepted) {
                rk_state.invalidate();
                h = h_next;
                ++result.n_reject;
                continue;
            }

            // 6. Post-step predicate scan for diode/ZCD events.
            // rk_state.k1 now holds k7 of the just-taken step (FSAL).
            std::optional<Real> t_pred;
            std::string p_name;
            PredicateType p_type = PredicateType::Custom;
            Vector x_pred;
            if (need_pred_scan) {
                const Vector& k7 = *rk_state.k1;
                auto evt = locate_event_in_step_(t, x, t + h_use, x_new,
                                                    k1_pre, k7, h_use);
                if (evt.has_value()) {
                    t_pred = evt->t_event;
                    p_name = std::move(evt->name);
                    p_type = evt->type;
                    x_pred = std::move(evt->x_at_event);
                }
            }

            if (t_pred.has_value()
                && *t_pred < t + h_use - kBacktrackTol) {
                // Predicate fired strictly inside the step — backtrack.
                t = *t_pred;
                x = std::move(x_pred);
                fire_event_(t, p_name, p_type, x, rk_state, result);
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
                    std::abs(t - t_gate) < Real{1e-12}) {
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

private:
    struct PredicateEvent {
        Real t_event;
        std::string name;
        PredicateType type;
        Vector x_at_event;
    };

    [[nodiscard]] Real next_gate_edge_(Real t_now, Real t_end) const {
        if constexpr (HasNextEdge<SwitchFn>) {
            return std::min(switch_fn_.next_edge_after(t_now), t_end);
        } else {
            return t_end;
        }
    }

    /// Scan all armed predicates over (t0, t1). Return the earliest
    /// sign-change root via Illinois on the Hermite cubic, or nullopt
    /// if no predicate changed sign. Ties broken by predicate priority.
    [[nodiscard]] std::optional<PredicateEvent> locate_event_in_step_(
        Real t0, const Vector& x0,
        Real t1, const Vector& x1,
        const Vector& k1, const Vector& k7,
        Real h) {

        constexpr Real kEpsTie = Real{1e-15};
        std::optional<PredicateEvent> best;
        int best_priority = std::numeric_limits<int>::max();

        for (const auto& p : predictor_.predicates()) {
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
                };
            };

            if (!best.has_value() || t_root < best->t_event - kEpsTie) {
                best = make_event();
                best_priority = p.priority;
            } else if (std::abs(t_root - best->t_event) <= kEpsTie
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
                       PEDResult<MaskT>& result) {
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

        const MaskT new_mask = switch_fn_(t_event + Real{1e-15});

        // Drop spurious same-mask gate edges silently.
        if (ptype == PredicateType::GateEdge && new_mask == old_mask) {
            return;
        }

        if (new_mask != old_mask) {
            system_.set_mask(new_mask);
        }

        rk_state.invalidate();      // f discontinues at event
        controller_.reset();        // avoid PI wind-up across mode change

        result.event_log.push_back(EventRecord<MaskT>{
            .t = t_event,
            .name = name,
            .type = ptype,
            .old_mask = old_mask,
            .new_mask = new_mask,
        });
        ++result.n_events;
    }

    System& system_;
    SwitchFn switch_fn_;
    PIController controller_;
    EventPredictor predictor_;
    Real dt_init_;
    Real dt_max_;
    std::size_t store_every_;
    StateProjectionFn state_projection_;
};

}  // namespace pulsim::dsed
