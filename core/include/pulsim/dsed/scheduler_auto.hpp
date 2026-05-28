#pragma once

// =============================================================================
// Pulsim — PED engine: auto-dispatch scheduler (Gate 4.D).
// =============================================================================
//
// Unified PED scheduler that picks DOPRI5 (explicit) or BDF2 (implicit)
// per mode-segment based on `StiffnessDetector::select()`. This is the
// algorithmic capstone of Gate 4: the scheduler INHERITS the BEST
// integrator for each operating regime, instead of forcing the user
// to commit at construction time.
//
// Outer loop (one iteration):
//
//   1. Compute t_gate = next gate edge.
//   2. Query stiffness detector for the CURRENT mode + tentative h.
//   3. Set integrator_choice for this mode-segment.
//   4. Take steps until t_gate or t_end:
//        choice == DOPRI5  →  rk45::step + PI adapt + Hermite/predicate scan
//        choice == BDF2    →  bdf2_step (fixed h_bdf2; cap at t_gate - t)
//   5. At t_gate: fire event, swap mask, INVALIDATE BOTH integrator
//      states, re-query stiffness for the new mode.
//
// On-disk vs runtime:
//
//   * The `System` template parameter must satisfy BOTH the RK45
//     contract (``rhs(t, x)``) AND the BDF2 contract (``A_matrix()``
//     + ``b_vector(t)``). This is `HasLTIPerMode` (from scheduler_bdf2.hpp)
//     plus the implicit `rhs()` requirement of PEDSimulator. For pure
//     LTI-per-mode systems, ``rhs(t, x) = A_matrix() * x + b_vector(t)``
//     is the canonical implementation.
//   * Event predicate scan + Hermite backtrack is RK45-only in this
//     iteration (BDF2 uses fixed h + gate-edge fast path). Diode/ZCD
//     events fire only when the active integrator is RK45 — the
//     auto-dispatcher therefore prefers RK45 on modes where event
//     resolution matters (typically non-stiff transition zones).
//
// Captured behaviour (from `test_scheduler_auto.cpp`):
//
//   2-mode system: mode A is stiff (R=0.1, |λ_max|≈1e7),
//                   mode B is non-stiff (R=10, |λ_max|≈1e5).
//   At dt_max = 5 µs:
//     mode A → |λ_max|·h = 50 → BDF2
//     mode B → |λ_max|·h = 0.5 → DOPRI5
//   Auto-dispatcher correctly switches integrator at each gate edge.

#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "pulsim/dsed/bdf2_integrator.hpp"
#include "pulsim/dsed/event_predictor.hpp"
#include "pulsim/dsed/rk45_dormand_prince.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/dsed/scheduler_bdf2.hpp"   // HasLTIPerMode concept
#include "pulsim/dsed/step_controller.hpp"
#include "pulsim/dsed/stiffness_detector.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// One row in the auto-dispatcher's event log — extends the base
/// EventRecord with the integrator that was active for the mode-segment
/// LEADING UP TO this event.
template <class MaskT>
struct AutoDispatchEventRecord {
    Real t;
    std::string name;
    PredicateType type;
    MaskT old_mask;
    MaskT new_mask;
    IntegratorChoice integrator_used;       // for the segment ending here
};

/// Auto-dispatch PED simulator output. Adds per-mode-segment integrator
/// log; otherwise identical to `PEDResult`.
template <class MaskT>
struct PEDResultAuto {
    std::vector<Real> times;
    std::vector<Vector> states;
    std::vector<AutoDispatchEventRecord<MaskT>> event_log;
    std::size_t n_accept = 0;
    std::size_t n_reject = 0;
    std::size_t n_events = 0;
    std::size_t n_rk45_steps = 0;
    std::size_t n_bdf2_steps = 0;
    Real cpu_time_seconds = Real{0};
};

/// Mode-id extractor — concept's default uses static_cast<int> for
/// integral/enum MaskT. For user-defined MaskT, specialise
/// `mode_id_of<MaskT>(MaskT)` in the caller's namespace.
template <class MaskT>
[[nodiscard]] inline int mode_id_of(MaskT m) noexcept
    requires std::is_integral_v<MaskT> || std::is_enum_v<MaskT>
{
    return static_cast<int>(m);
}

/// Auto-dispatch PED simulator: picks DOPRI5 or BDF2 per mode-segment.
///
/// Templated on `System` (must satisfy `HasLTIPerMode` + provide
/// ``rhs(t, x)``) and `SwitchFn` (must satisfy `HasNextEdge`).
template <class System, class SwitchFn>
    requires HasLTIPerMode<System> && HasNextEdge<SwitchFn>
class PEDSimulatorAuto {
public:
    using MaskT = std::invoke_result_t<SwitchFn, Real>;

    PEDSimulatorAuto(System& system,
                       SwitchFn switch_fn,
                       PIController rk45_controller = PIController{},
                       StiffnessDetector detector = StiffnessDetector{},
                       Real dt_init = Real{1e-9},
                       Real dt_max = Real{1e-5},
                       Real h_bdf2 = Real{1e-6},
                       std::size_t store_every = 1)
        : system_{system},
          switch_fn_{std::move(switch_fn)},
          rk45_controller_{std::move(rk45_controller)},
          detector_{std::move(detector)},
          dt_init_{dt_init},
          dt_max_{dt_max},
          h_bdf2_{h_bdf2},
          store_every_{store_every} {}

    [[nodiscard]] PEDResultAuto<MaskT> simulate(const Vector& x0, Real t_end) {
        using clock = std::chrono::high_resolution_clock;
        const auto t0_wall = clock::now();

        Vector x = x0;
        Real t = Real{0};
        Real h_rk45 = dt_init_;

        MaskT mask = switch_fn_(t);
        system_.set_mask(mask);

        // Pick integrator for the initial mode-segment.
        IntegratorChoice choice = detector_.select(
            mode_id_of(mask), system_.A_matrix(), dt_max_);

        RK45State rk_state;
        BDF2State bdf2_state;
        PEDResultAuto<MaskT> result;
        result.times.push_back(t);
        result.states.push_back(x);
        std::size_t step_idx = 0;

        constexpr std::size_t kMaxSteps = 10'000'000;
        constexpr Real kEpsEdge = Real{1e-12};

        auto f = [this](Real tau, const Vector& xx) -> Vector {
            return system_.rhs(tau, xx);
        };
        auto b_fn = [this](Real tau) -> Vector {
            return system_.b_vector(tau);
        };

        while (t < t_end) {
            if (result.n_accept + result.n_reject > kMaxSteps) {
                throw std::runtime_error(
                    "PEDSimulatorAuto: exceeded max-step cap");
            }

            // 1. Next gate-edge time (fast path)
            const Real t_gate = std::min(
                switch_fn_.next_edge_after(t), t_end);

            // 2. Take one step using the current mode's integrator
            Real h_use;
            Vector x_new;
            bool accepted = true;
            Real h_next_rk45 = h_rk45;

            if (choice == IntegratorChoice::BDF2) {
                // BDF2: fixed h_bdf2 capped at gate edge
                h_use = std::min(h_bdf2_, t_gate - t);
                if (h_use <= Real{0}) {
                    if (t_gate < t_end) {
                        const IntegratorChoice prev_choice = choice;
                        fire_gate_event_(t_gate, x, rk_state, bdf2_state,
                                            prev_choice, result);
                        t = t_gate;
                        choice = detector_.select(
                            mode_id_of(system_.current_mask()),
                            system_.A_matrix(), dt_max_);
                        // Only reset h_rk45 when SWITCHING from BDF2 to RK45
                        // (no good prior value); otherwise PI adapts h
                        // naturally as in the RK45-only PEDSimulator.
                        if (prev_choice == IntegratorChoice::BDF2
                            && choice == IntegratorChoice::DOPRI5) {
                            h_rk45 = dt_init_;
                        }
                    }
                    continue;
                }
                const DenseMatrix A_cur = system_.A_matrix();
                auto [x_new_bdf2, _err] = bdf2_step(
                    A_cur, b_fn, t, x, h_use, bdf2_state);
                x_new = std::move(x_new_bdf2);
                ++result.n_bdf2_steps;
            } else {
                // DOPRI5: adaptive PI capped at gate edge
                h_use = std::min({h_rk45, dt_max_, t_gate - t});
                if (h_use <= Real{0}) {
                    if (t_gate < t_end) {
                        const IntegratorChoice prev_choice = choice;
                        fire_gate_event_(t_gate, x, rk_state, bdf2_state,
                                            prev_choice, result);
                        t = t_gate;
                        choice = detector_.select(
                            mode_id_of(system_.current_mask()),
                            system_.A_matrix(), dt_max_);
                        if (prev_choice == IntegratorChoice::BDF2
                            && choice == IntegratorChoice::DOPRI5) {
                            h_rk45 = dt_init_;
                        }
                    }
                    continue;
                }
                auto [x_new_rk, err] = step(f, t, x, h_use, rk_state);
                auto [accepted_rk, h_next] =
                    rk45_controller_.accept(err, x, x_new_rk, h_use);
                accepted = accepted_rk;
                h_next_rk45 = h_next;
                if (!accepted) {
                    rk_state.invalidate();
                    h_rk45 = h_next;
                    ++result.n_reject;
                    continue;
                }
                x_new = std::move(x_new_rk);
                ++result.n_rk45_steps;
            }

            // 3. Commit accepted step
            t += h_use;
            x = std::move(x_new);
            if (choice == IntegratorChoice::DOPRI5) {
                h_rk45 = h_next_rk45;
            }
            ++result.n_accept;

            // 4. Did we land on a gate edge?
            if (t_gate < t_end && std::abs(t - t_gate) < kEpsEdge) {
                const IntegratorChoice prev_choice = choice;
                fire_gate_event_(t_gate, x, rk_state, bdf2_state,
                                    prev_choice, result);
                // Re-query stiffness for the new mode
                choice = detector_.select(
                    mode_id_of(system_.current_mask()),
                    system_.A_matrix(), dt_max_);
                // Only reset h_rk45 when SWITCHING from BDF2 to RK45
                // (no good prior value); otherwise let PI adapt h
                // naturally as in the RK45-only PEDSimulator.
                if (prev_choice == IntegratorChoice::BDF2
                    && choice == IntegratorChoice::DOPRI5) {
                    h_rk45 = dt_init_;
                }
            }

            // 5. Record
            ++step_idx;
            if (step_idx % store_every_ == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
            }
        }

        if (result.times.back() != t) {
            result.times.push_back(t);
            result.states.push_back(x);
        }

        const auto t1_wall = clock::now();
        result.cpu_time_seconds =
            std::chrono::duration<Real>(t1_wall - t0_wall).count();
        return result;
    }

    /// Convenience accessor: which integrator did the dispatcher pick
    /// for ``mode_id`` (assumes the detector has been queried at least
    /// once for that mode).
    [[nodiscard]] IntegratorChoice current_integrator_for(
        int mode_id, const DenseMatrix& A) {
        return detector_.select(mode_id, A, dt_max_);
    }

    [[nodiscard]] const StiffnessDetector& detector() const noexcept {
        return detector_;
    }

private:
    void fire_gate_event_(Real t_event,
                            Vector& /*x*/,
                            RK45State& rk_state,
                            BDF2State& bdf2_state,
                            IntegratorChoice segment_integrator,
                            PEDResultAuto<MaskT>& result) {
        const MaskT old_mask = system_.current_mask();
        const MaskT new_mask = switch_fn_(t_event + Real{1e-15});
        if (new_mask == old_mask) return;   // spurious

        system_.set_mask(new_mask);
        // Discontinuous f → invalidate BOTH integrators' caches.
        rk_state.invalidate();
        bdf2_state.invalidate();
        rk45_controller_.reset();

        result.event_log.push_back(AutoDispatchEventRecord<MaskT>{
            .t = t_event,
            .name = "gate_edge",
            .type = PredicateType::GateEdge,
            .old_mask = old_mask,
            .new_mask = new_mask,
            .integrator_used = segment_integrator,
        });
        ++result.n_events;
    }

    System& system_;
    SwitchFn switch_fn_;
    PIController rk45_controller_;
    StiffnessDetector detector_;
    Real dt_init_;
    Real dt_max_;
    Real h_bdf2_;
    std::size_t store_every_;
};

}  // namespace pulsim::dsed
