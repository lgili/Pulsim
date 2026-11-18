#pragma once

// refactor-unified-transient-stepper, task 6.1 / 7.1 (design D1, D4): the
// ONE step skeleton. It owns the order of a step and nothing else:
//
//   poll → begin (instant processing: ticks, edges, breakpoints) →
//   h = clamp(propose) → [rollback copy] → take the step (stage solves
//   through the body's single solve point) → error → controller decision
//   (accept / retry / abort) → event policy (accept, or truncate and
//   re-solve) → commit → on_step (sampling, hooks) → advance the clock.
//
// The engine supplies a Body (its services: the solve point, the event and
// mode policies, commit/rollback of device state, the recorders); the
// integrator supplies the stage table and the error exponent; the
// controller supplies the clock and the accept/retry law. With
// `C::uses_lte == false` the rollback copy, the error call and the retry
// branch are not compiled (the fixed path stays today's loop).

#include <concepts>
#include <cstdint>

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/step/controllers.hpp"
#include "pulsim/solver/step/integrators.hpp"

namespace pulsim::step {

template <class I>
concept Integrator = requires(Real h, int s) {
    { I::name } -> std::convertible_to<const char*>;
    { I::order } -> std::convertible_to<int>;
    { I::n_stages } -> std::convertible_to<int>;
    { I::err_exponent } -> std::convertible_to<Real>;
    { I::stage(h, s) } -> std::same_as<StageSpec>;
    { I::weight(s, s) } -> std::convertible_to<Real>;
};

/// The step the skeleton is taking: grid index, start, nominal end and
/// size. For a fixed clock `t1` is the grid point `t0 + k·dt` (never
/// `t0 + h`) and `h` the nominal dt; for a variable clock `t1 = t0 + h`.
struct StepFrame {
    Size k = 0;
    Real t0 = Real{0};
    Real t1 = Real{0};
    Real h = Real{0};
};

enum class Verdict : std::uint8_t { Accept, Retry, Abort };
struct Decision {
    Verdict verdict = Verdict::Accept;
    Real h = Real{0};   // the retry's step (Retry) or the next proposal (Accept)
};

enum class StepStatus : std::uint8_t { Accepted, Stopped, Done, Aborted };

template <class C>
concept StepController = requires(C& c, const C& cc, bool ok, Real err, Real h) {
    { C::uses_lte } -> std::convertible_to<bool>;
    { cc.done() } -> std::convertible_to<bool>;
    { cc.frame() } -> std::same_as<StepFrame>;
    { c.on_result(ok, err, h) } -> std::same_as<Decision>;
    { c.advance(h) };
};

/// The fixed engine's clock: steps k = k_begin .. n_steps − 1, the step k
/// ending at grid point `clock.at(k)`; a failed step is the body's to
/// handle (its retry ladder), never the clock's.
struct FixedController {
    static constexpr bool uses_lte = false;
    FixedClock clock;
    Size k = 1;
    Size n_steps = 0;
    [[nodiscard]] bool done() const noexcept { return k >= n_steps; }
    [[nodiscard]] StepFrame frame() const noexcept {
        return {k, clock.at(k - 1), clock.at(k), clock.dt};
    }
    Decision on_result(bool ok, Real, Real h) const noexcept {
        return {ok ? Verdict::Accept : Verdict::Abort, h};
    }
    void advance(Real) noexcept { ++k; }
};

/// A variable clock on the LTE controller's factor laws: t += h, the next
/// proposal from the normalised error (≤ 1 accepts) and the integrator's
/// exponent. The TR-BDF2 driver's special cases (gate-edge and diode
/// memories, chatter guards, h_min refusals) stay in its body until that
/// loop moves onto this skeleton.
template <class I>
struct LteClock {
    static constexpr bool uses_lte = true;
    LteController ctrl;
    Real t = Real{0};
    Real t_end = Real{0};
    Real h_min = Real{0};
    Real h_max = Real{0};
    Size n_accept = 0;
    Size n_reject = 0;
    [[nodiscard]] bool done() const noexcept { return !(t < t_end); }
    [[nodiscard]] StepFrame frame() const noexcept {
        const Real h = ctrl.h < t_end - t ? ctrl.h : t_end - t;
        return {n_accept, t, t + h, h};
    }
    Decision on_result(bool ok, Real err, Real h) noexcept {
        if (!ok) {
            ++n_reject;
            const Real hn = h * Real{0.25};
            return {hn < h_min ? Verdict::Abort : Verdict::Retry, hn};
        }
        if (err > Real{1}) {
            ++n_reject;
            const Real hn = h * LteController::reject_factor(err, I::err_exponent);
            return {hn < h_min ? Verdict::Abort : Verdict::Retry, hn};
        }
        const Real hn = h * LteController::accept_factor(err, I::err_exponent);
        ctrl.h = hn < h_max ? hn : h_max;
        return {Verdict::Accept, ctrl.h};
    }
    void advance(Real h) noexcept {
        t = h < t_end - t ? t + h : t_end;   // the step clamped to the end lands on it
        ++n_accept;
    }
};

/// Runs every stage of `I` over `f.h` through the body's single solve
/// point, in order; the stage spec carries the step context the devices
/// are told, so no solve can see a stale one.
template <Integrator I, class Body>
bool run_stages(Body& body, const StepFrame& f) {
    for (int s = 0; s < I::n_stages; ++s) {
        if (!body.solve_stage(I::stage(f.h, s), s, f)) return false;
    }
    return true;
}

/// One step. Body requirements (all members):
///   bool poll();                          false = stop (cancellation)
///   void begin(const StepFrame&);         instant processing at f.t0
///   Real clamp(const StepFrame&, Real h); event landing (≤ h)
///   bool take(const StepFrame&);          the step's solves (default:
///                                         run_stages<I>); false = failed
///   bool after_solve(StepFrame&);         event policy; false = f.h was
///                                         truncated, re-solve
///   void commit(const StepFrame&);        device state + history
///   void on_step(const StepFrame&);       sampling and hooks
/// and, when C::uses_lte: save() / restore() (rollback copy) and
/// Real error(const StepFrame&) (normalised, ≤ 1 accepts).
template <Integrator I, StepController C, class Body>
StepStatus step_once(C& ctrl, Body& body) {
    if (ctrl.done()) return StepStatus::Done;
    if (!body.poll()) return StepStatus::Stopped;
    StepFrame f = ctrl.frame();
    body.begin(f);
    f.h = body.clamp(f, f.h);
    if constexpr (C::uses_lte) f.t1 = f.t0 + f.h;
    for (;;) {
        if constexpr (C::uses_lte) body.save();
        const bool ok = body.take(f);
        Real err = Real{0};
        if constexpr (C::uses_lte) {
            if (ok) err = body.error(f);
        }
        const Decision d = ctrl.on_result(ok, err, f.h);
        if (d.verdict == Verdict::Abort) return StepStatus::Aborted;
        if constexpr (C::uses_lte) {
            if (d.verdict == Verdict::Retry) {
                body.restore();
                f.h = body.clamp(f, d.h);
                f.t1 = f.t0 + f.h;
                continue;
            }
        }
        if (!body.after_solve(f)) {
            if constexpr (C::uses_lte) body.restore();
            continue;
        }
        break;
    }
    body.commit(f);
    body.on_step(f);
    ctrl.advance(f.h);
    return StepStatus::Accepted;
}

/// The run: steps until the clock is done, the body stops it or a step
/// aborts. Returns the last status.
template <Integrator I, StepController C, class Body>
StepStatus run(C& ctrl, Body& body) {
    StepStatus s = StepStatus::Done;
    while ((s = step_once<I>(ctrl, body)) == StepStatus::Accepted) {
    }
    return s;
}

}  // namespace pulsim::step
