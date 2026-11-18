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

#include <cmath>
#include <concepts>
#include <limits>
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

/// The step the skeleton is taking: grid index, start, end and size, and
/// the limit the step may not pass (`t_lim`: the grid point for a fixed
/// clock, `t_end` for a variable one). A step whose size reaches the
/// limit ends ON it (`t1 = t_lim`, never `t0 + h`), so a fixed grid point
/// stays `t0 + k·dt` and a run lands on `t_end` exactly.
struct StepFrame {
    Size k = 0;
    Real t0 = Real{0};
    Real t1 = Real{0};
    Real h = Real{0};
    Real t_lim = Real{0};
    /// Sets the size; a size that reaches the limit becomes `t_lim − t0`
    /// and the step ends on the limit.
    void resize(Real h_new) noexcept {
        if (h_new < t_lim - t0) {
            h = h_new;
            t1 = t0 + h_new;
        } else {
            h = t_lim - t0;
            t1 = t_lim;
        }
    }
};

enum class Verdict : std::uint8_t { Accept, Retry, Abort };
struct Decision {
    Verdict verdict = Verdict::Accept;
    Real h = Real{0};   // the retry's step (Retry) or the next proposal (Accept)
};

enum class StepStatus : std::uint8_t { Accepted, Stopped, Done, Aborted };

template <class C>
concept StepController = requires(C& c, const C& cc, bool ok, Real err, Real h,
                                  const StepFrame& f) {
    { C::uses_lte } -> std::convertible_to<bool>;
    { cc.done() } -> std::convertible_to<bool>;
    { cc.frame() } -> std::same_as<StepFrame>;
    { c.on_result(ok, err, h) } -> std::same_as<Decision>;
    { c.accepted(h) };   // the next proposal, from the step's full attempt
    { c.advance(f) };    // the committed frame
};

/// A body whose event policy can cut a step short (`static constexpr bool
/// localizes = true`) needs the rollback copy on any clock.
template <class B>
constexpr bool body_localizes() {
    if constexpr (requires { B::localizes; }) return B::localizes;
    return false;
}

/// The fixed engine's clock: steps k = 1 .. n_steps − 1, the step k ending
/// at grid point `clock.at(k)`. A step the event policy cut short is
/// committed and the rest of the same grid interval is taken next (k does
/// not advance until the grid point is reached). A failed step is the
/// body's to handle (its retry ladder), never the clock's: it aborts.
struct FixedController {
    static constexpr bool uses_lte = false;
    FixedClock clock;
    Size k = 1;   // ≥ 1: step k starts at clock.at(k − 1)
    Size n_steps = 0;
    bool partial = false;   // inside grid interval k, at t_in
    Real t_in = Real{0};
    [[nodiscard]] bool done() const noexcept { return k >= n_steps; }
    [[nodiscard]] StepFrame frame() const noexcept {
        const Real t1 = clock.at(k);
        if (partial) return {k, t_in, t1, t1 - t_in, t1};
        return {k, clock.at(k - 1), t1, clock.dt, t1};
    }
    Decision on_result(bool ok, Real, Real h) const noexcept {
        return {ok ? Verdict::Accept : Verdict::Abort, h};
    }
    void accepted(Real) const noexcept {}
    void advance(const StepFrame& f) noexcept {
        partial = f.t1 != f.t_lim;
        if (partial) {
            t_in = f.t1;
        } else {
            ++k;
        }
    }
};

/// A variable clock on the LTE controller's factor laws: the next proposal
/// from the normalised error (≤ 1 accepts) and the integrator's exponent;
/// a failed solve or a non-finite error retries at h/4; a step below
/// `h_min` aborts. The TR-BDF2 driver's special cases (gate-edge and
/// diode memories, chatter guards) stay in its body until that loop moves
/// onto this skeleton.
template <class I>
struct LteClock {
    static constexpr bool uses_lte = true;
    LteController ctrl;
    Real t = Real{0};
    Real t_end = Real{0};
    Real h_min = Real{0};
    Real h_max = std::numeric_limits<Real>::infinity();
    Size n_accept = 0;
    Size n_reject = 0;
    [[nodiscard]] bool done() const noexcept { return !(t < t_end); }
    [[nodiscard]] StepFrame frame() const noexcept {
        StepFrame f{n_accept, t, t, Real{0}, t_end};
        f.resize(ctrl.h);
        return f;
    }
    Decision on_result(bool ok, Real err, Real h) noexcept {
        if (!ok || !std::isfinite(err) || err > Real{1}) {
            ++n_reject;
            const Real hn = (ok && std::isfinite(err))
                ? h * LteController::reject_factor(err, I::err_exponent)
                : h * Real{0.25};
            return {hn < h_min ? Verdict::Abort : Verdict::Retry, hn};
        }
        const Real hn = h * LteController::accept_factor(err, I::err_exponent);
        return {Verdict::Accept, hn < h_max ? hn : h_max};
    }
    void accepted(Real h_next) noexcept { ctrl.h = h_next; }
    void advance(const StepFrame& f) noexcept {
        t = f.t1;
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

/// Re-solves of one step (retries and event cuts) before it aborts.
inline constexpr int kMaxResolves = 200;

/// One step. Body requirements (all members):
///   bool poll();                          false = stop (cancellation)
///   void begin(const StepFrame&);         instant processing at f.t0
///   Real clamp(const StepFrame&, Real h); event landing (≤ h)
///   bool take(const StepFrame&);          the step's solves (default:
///                                         run_stages<I>); false = failed
///   bool after_solve(StepFrame&);         event policy; false = it cut the
///                                         step (f.resize), re-solve
///   void commit(const StepFrame&);        device state + history
///   void on_step(const StepFrame&);       sampling and hooks
/// and, when C::uses_lte or the body localizes: save() / restore() (the
/// rollback copy: restored before every re-solve and on an abort);
/// when C::uses_lte: Real error(const StepFrame&) (normalised, ≤ 1
/// accepts). The next proposal comes from the step's full attempt, not
/// from a sliver an event cut it to.
template <Integrator I, StepController C, class Body>
StepStatus step_once(C& ctrl, Body& body) {
    constexpr bool rollback = C::uses_lte || body_localizes<Body>();
    if (ctrl.done()) return StepStatus::Done;
    if (!body.poll()) return StepStatus::Stopped;
    StepFrame f = ctrl.frame();
    body.begin(f);
    if (const Real hc = body.clamp(f, f.h); hc < f.h) f.resize(hc);
    if constexpr (rollback) body.save();
    bool cut = false;
    Real proposal = f.h;
    for (int n = 0;; ++n) {
        Decision d{Verdict::Abort, f.h};
        if (n < kMaxResolves) {
            const bool ok = body.take(f);
            Real err = Real{0};
            if constexpr (C::uses_lte) {
                if (ok) err = body.error(f);
            }
            d = ctrl.on_result(ok, err, f.h);
        }
        if (d.verdict == Verdict::Abort) {
            if constexpr (rollback) body.restore();
            return StepStatus::Aborted;
        }
        if constexpr (C::uses_lte) {
            if (d.verdict == Verdict::Retry) {
                body.restore();
                f.resize(body.clamp(f, d.h));
                continue;
            }
        }
        if (!cut) proposal = d.h;
        if (!body.after_solve(f)) {
            cut = true;
            if constexpr (rollback) body.restore();
            continue;
        }
        break;
    }
    body.commit(f);
    body.on_step(f);
    ctrl.accepted(proposal);
    ctrl.advance(f);
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
