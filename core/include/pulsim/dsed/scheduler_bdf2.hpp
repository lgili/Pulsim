#pragma once

// =============================================================================
// Pulsim — PED engine: BDF2 scheduler (Gate 4.C).
// =============================================================================
//
// Implicit BDF2 variant of the PED scheduler — companion to
// ``PEDSimulator`` (DOPRI5). Used on **stiff** masks where the
// explicit method's stability constraint shrinks h below what
// accuracy alone requires.
//
// Design contract (vs the RK45 scheduler):
//
//   * Same gate-edge fast path: ``switch_fn.next_edge_after(t)``.
//   * No predicate scan / Hermite backtrack in this iteration — body
//     diode + ZCD events are RK45-handled (commutation prefers small
//     dt + explicit dynamics anyway). Cross-integrator predicate
//     dispatch lands at Gate 4.D / Gate 5.
//   * Fixed h (or PI-adapted h via BDF2PIController, with refactor on
//     h change). Default fixed-h dispatch — adaptive variant is a
//     trivial extension once we land an LLC test scenario at Gate 5.
//   * State management:
//       - On mask change: invalidate BDF2 history (y_prev) + LU cache.
//         The next step uses Crank-Nicolson bootstrap to restart.
//       - On h change: invalidate LU cache only (history is still
//         valid in time-domain; BDF2 just needs to re-factor J).
//
// The `System` template parameter must expose:
//
//   * ``DenseMatrix A_matrix() const``         — current mask's A
//   * ``Vector b_vector(Real t) const``         — current mask's b(t)
//   * ``void set_mask(MaskT mask)``             — swap mask (mutates
//     A_matrix and b_vector return values)
//   * ``MaskT current_mask() const``            — read current mask
//
// (Note: this is a richer System contract than ``PEDSimulator``'s,
// which only needs ``rhs(t, x)``. The BDF2 implicit step requires
// (A, b) directly to factor J = I - (2h/3) A.)
//
// Validated end-to-end on 2-mode stiff RLC switching at 100 kHz
// in `test_bdf2_scheduler.cpp` — matches a high-resolution DOPRI5
// reference and beats DOPRI5-stability-limit wall-clock.

#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "pulsim/dsed/bdf2_integrator.hpp"
#include "pulsim/dsed/time_eps.hpp"
#include "pulsim/dsed/event_predictor.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// Whether `System` exposes the BDF2 contract: A_matrix + b_vector + mask.
template <class System>
concept HasLTIPerMode = requires(System& s, const System& cs, Real t) {
    { cs.A_matrix() } -> std::convertible_to<DenseMatrix>;
    { cs.b_vector(t) } -> std::convertible_to<Vector>;
    { cs.current_mask() };
    s.set_mask(cs.current_mask());
};

/// BDF2-based PED simulator (Gate 4.C).
///
/// Templated on `System` (LTI-per-mode source — provides A_matrix +
/// b_vector + set_mask + current_mask) and `SwitchFn` (gate-edge
/// schedule). MaskT is deduced from `decltype(switch_fn(0.0))`.
///
/// Reuses the ``EventRecord`` and ``PEDResult`` structs from
/// scheduler.hpp so the two schedulers' outputs are interchangeable.
template <class System, class SwitchFn>
    requires HasLTIPerMode<System>
class PEDSimulatorBDF2 {
public:
    using MaskT = std::invoke_result_t<SwitchFn, Real>;

    PEDSimulatorBDF2(System& system,
                      SwitchFn switch_fn,
                      Real h_fixed,
                      std::size_t store_every = 1)
        : system_{system},
          switch_fn_{std::move(switch_fn)},
          h_fixed_{h_fixed},
          store_every_{store_every} {}

    /// Run the BDF2 simulation from ``t = 0`` to ``t = t_end``.
    [[nodiscard]] PEDResult<MaskT> simulate(const Vector& x0, Real t_end) {
        using clock = std::chrono::high_resolution_clock;
        const auto t0_wall = clock::now();

        Vector x = x0;
        Real t = Real{0};
        MaskT mask = switch_fn_(t);
        system_.set_mask(mask);

        BDF2State bdf2_state;
        PEDResult<MaskT> result;
        result.times.push_back(t);
        result.states.push_back(x);
        result.sample_masks.push_back(system_.current_mask());
        std::size_t step_idx = 0;

        constexpr std::size_t kMaxSteps = 10'000'000;

        auto b_fn = [this](Real tau) -> Vector {
            return system_.b_vector(tau);
        };

        while (t < t_end) {
            if (result.n_accept > kMaxSteps) {
                throw std::runtime_error(
                    "PEDSimulatorBDF2: exceeded max-step cap");
            }

            // 1. Next gate-edge time (analytical fast path)
            const Real t_gate = next_gate_edge_(t, t_end);
            const Real t_target = std::min(t_gate, t_end);

            // 2. Step size: capped at next gate edge
            Real h_use = std::min(h_fixed_, t_target - t);
            if (h_use <= Real{0}) {
                // Sitting exactly on a gate edge — fire it.
                if (t_gate < t_end) {
                    fire_gate_event_(t_gate, x, bdf2_state, result);
                    t = t_gate;
                }
                continue;
            }

            // 3. Pull current A from the system (set by current mask).
            // By REFERENCE (v2.0 Phase 3 item 4 / audit E.4): this
            // sat inside the step loop copying the full dense matrix
            // every step — 1.3 MB per step at 400 states. The
            // adapter's entry pointer is stable until set_mask, and
            // no set_mask happens within a step.
            const DenseMatrix& A_cur = system_.A_matrix();

            // 4. Take BDF2 step
            auto [x_new, _err] = bdf2_step(A_cur, b_fn, t, x, h_use, bdf2_state);

            // 4b. Finite-value guard (GUI integration findings T1.3).
            // BDF2 is fixed-step — we can't shrink h to retry.
            // If the step produced NaN/Inf, the only thing we can
            // do is fail loudly with an actionable error so the
            // user notices and switches engine / audits their
            // b_extra_fn / etc. Pre-fix the NaN propagated silently
            // into the result, where downstream plots and metrics
            // surfaced "all NaN" without context.
            {
                bool finite = true;
                for (Eigen::Index i = 0; i < x_new.size(); ++i) {
                    if (!std::isfinite(x_new(i))) { finite = false; break; }
                }
                if (!finite) {
                    throw std::runtime_error(
                        std::string{"PEDSimulator (BDF2): step "}
                        + "produced NaN/Inf at t="
                        + std::to_string(t)
                        + ", h_fixed="
                        + std::to_string(h_use)
                        + ". Common root causes:\n"
                        + "  (1) the LTI A matrix for the current "
                          "switch mask is singular or ill-conditioned;\n"
                        + "  (2) a Python b_extra_fn (e.g. motor / "
                          "control observer) returned NaN;\n"
                        + "  (3) a switch_fn returned a mask whose "
                          "dynamics blow up.\n"
                        + "Workarounds: try engine='pwl' (T1.2 auto-LM "
                          "handles rank-deficient Jacobians); "
                          "switch to integrator='rk45' or 'auto' "
                          "(variable-step adaptive shrinks dt around "
                          "stiff regions); shrink h_bdf2; or audit "
                          "b_extra_fn / switch_fn for NaN-producing "
                          "branches.");
                }
            }

            // 5. Commit
            t += h_use;
            x = std::move(x_new);
            ++result.n_accept;

            // 6. Did we land on a gate edge?
            if (t_gate < t_end && near_time(t, t_gate)) {
                fire_gate_event_(t_gate, x, bdf2_state, result);
            }

            // 7. Record
            ++step_idx;
            if (step_idx % store_every_ == 0) {
                result.times.push_back(t);
                result.states.push_back(x);
        result.sample_masks.push_back(system_.current_mask());
            }
        }

        if (result.times.back() != t) {
            result.times.push_back(t);
            result.states.push_back(x);
        result.sample_masks.push_back(system_.current_mask());
        }

        const auto t1_wall = clock::now();
        result.cpu_time_seconds =
            std::chrono::duration<Real>(t1_wall - t0_wall).count();
        return result;
    }

private:
    [[nodiscard]] Real next_gate_edge_(Real t_now, Real t_end) const {
        if constexpr (HasNextEdge<SwitchFn>) {
            // See `scheduler.hpp::next_gate_edge_` for the rationale
            // behind the `isfinite` defensive fallback. Same logic
            // applies here: ∞ from `next_edge_after` triggers a
            // `dt_max/10` poll boundary so PWM events from plain
            // Python switch_fns aren't silently missed.
            const Real raw = switch_fn_.next_edge_after(t_now);
            if (std::isfinite(raw)) {
                return std::min(raw, t_end);
            }
            constexpr Real kPollFractionOfDtMax = Real{0.1};
            // BDF2 uses a fixed step (`h_fixed_`); same logic as
            // RK45 — derive the poll cap from the only step-size
            // knob the BDF2 scheduler has.
            return std::min(t_now + h_fixed_ * kPollFractionOfDtMax,
                              t_end);
        } else {
            return t_end;
        }
    }

    void fire_gate_event_(Real t_event,
                            Vector& /*x*/,
                            BDF2State& bdf2_state,
                            PEDResult<MaskT>& result) {
        const MaskT old_mask = system_.current_mask();
        const MaskT new_mask = switch_fn_(advance_past(t_event));
        if (new_mask == old_mask) return;   // spurious gate edge

        system_.set_mask(new_mask);
        // A and b changed → both history (y_prev) and LU (J = I - (2h/3) A_old)
        // are invalidated; next step starts fresh via Crank-Nicolson bootstrap.
        bdf2_state.invalidate();

        result.event_log.push_back(EventRecord<MaskT>{
            .t = t_event,
            .name = "gate_edge",
            .type = PredicateType::GateEdge,
            .old_mask = old_mask,
            .new_mask = new_mask,
        });
        ++result.n_events;
    }

    System& system_;
    SwitchFn switch_fn_;
    Real h_fixed_;
    std::size_t store_every_;
};

}  // namespace pulsim::dsed
