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

// A phase of an engine's step that both its legacy loop and the skeleton
// call, and every adapter between them (`run`, `step_once`, the body's
// forwarders): forced inline into the DRIVER that calls it. Each driver
// (the legacy loop, the skeleton's `run`) is one out-of-line function
// (PULSIM_STEP_DRIVER): an adapter left out of line costs a call per phase
// and two closure hops per captured local, and a driver inlined into the
// engine's own large body runs slower than the same driver alone
// (refactor-unified-transient-stepper DECISIONS 14) on the fixed engine.
// The TR-BDF2 engine measured the opposite (DECISIONS 16): its phases touch
// so many engine locals that an out-of-line driver's closure hops cost more,
// so both its drivers are PULSIM_STEP_PHASE. Measure per engine. Empty on
// MSVC, as the other forced-inline macros of the code base.
#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_STEP_PHASE __attribute__((always_inline))
#define PULSIM_STEP_DRIVER __attribute__((noinline))
#else
#define PULSIM_STEP_PHASE
#define PULSIM_STEP_DRIVER
#endif

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

// Reject and End are a deciding body's (`decides`, the TR-BDF2 loop):
// Reject discards the attempt and the next one starts at poll (no rollback:
// the body carries what the attempt learned); End ends the run without it.
enum class Verdict : std::uint8_t { Accept, Retry, Abort, Reject, End };
struct Decision {
    Verdict verdict = Verdict::Accept;
    Real h = Real{0};   // the retry's step (Retry) or the next proposal (Accept)
};

enum class StepStatus : std::uint8_t { Accepted, Stopped, Done, Aborted, Rejected };

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

/// A body that decides its own steps (`static constexpr bool decides =
/// true`): its phases return the Verdict and its clock is its own.
template <class B>
constexpr bool body_decides() {
    if constexpr (requires { B::decides; }) return B::decides;
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

/// The clock of a deciding body (the TR-BDF2 loop, slice 3b): the time,
/// the end test and the step memory (`LteController`, whose `h` is the
/// proposal) are the body's; the clock frames the attempt and nothing else
/// (the body advances `t` in its commit and decides every rejection).
template <class Live>
struct BodyClock {
    static constexpr bool uses_lte = true;
    Live& live_;
    const LteController& mem;
    const Real& t;
    Real t_end = Real{0};
    [[nodiscard]] PULSIM_STEP_PHASE bool done() const { return !live_(); }
    [[nodiscard]] StepFrame frame() const noexcept {
        StepFrame f{0, t, t, Real{0}, t_end};
        f.resize(mem.h);
        return f;
    }
    Decision on_result(bool ok, Real, Real h) const noexcept {
        return {ok ? Verdict::Accept : Verdict::Abort, h};
    }
    void accepted(Real) const noexcept {}
    void advance(const StepFrame&) const noexcept {}
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
PULSIM_STEP_PHASE inline StepStatus step_once(C& ctrl, Body& body) {
    if constexpr (body_decides<Body>()) {
        // One ATTEMPT (slice 3b, DECISIONS 15): begin and clamp re-run on
        // every attempt; take, error and after_solve each return the
        // verdict (after_solve may cut the step and commit at its landing).
        if (ctrl.done()) return StepStatus::Done;
        if (!body.poll()) return StepStatus::Stopped;
        StepFrame f = ctrl.frame();
        body.begin(f);
        f.resize(body.clamp(f, f.h));
        Verdict v = body.take(f);
        if (v == Verdict::Accept) v = body.error(f);
        if (v == Verdict::Accept) v = body.after_solve(f);
        if (v == Verdict::Reject) return StepStatus::Rejected;
        if (v != Verdict::Accept) {
            return v == Verdict::End ? StepStatus::Done : StepStatus::Aborted;
        }
        body.commit(f);
        body.on_step(f);
        ctrl.advance(f);
        return StepStatus::Accepted;
    } else {
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
}

/// A body made of an engine's phase callables (slice 3a, DECISIONS 13):
/// the fixed engine's loop phases, the SAME callables its legacy loop
/// calls, so the two paths cannot diverge. The event landing is inside
/// the engine's `take` (pwl splits a step at its events itself), so
/// `clamp` keeps the clock's step and `after_solve` never cuts.
template <class Poll, class Begin, class Take, class After, class Commit, class OnStep>
struct PhaseBody {
    Poll& poll_;
    Begin& begin_;
    Take& take_;
    After& after_;
    Commit& commit_;
    OnStep& on_step_;
    PULSIM_STEP_PHASE bool poll() { return poll_(); }
    PULSIM_STEP_PHASE void begin(const StepFrame& f) { begin_(f); }
    Real clamp(const StepFrame&, Real h) const noexcept { return h; }
    PULSIM_STEP_PHASE bool take(const StepFrame& f) { return take_(f); }
    PULSIM_STEP_PHASE bool after_solve(StepFrame& f) { return after_(f); }
    PULSIM_STEP_PHASE void commit(const StepFrame& f) { commit_(f); }
    PULSIM_STEP_PHASE void on_step(const StepFrame& f) { on_step_(f); }
};

template <class Poll, class Begin, class Take, class After, class Commit, class OnStep>
PhaseBody<Poll, Begin, Take, After, Commit, OnStep> phase_body(
    Poll& p, Begin& b, Take& t, After& a, Commit& c, OnStep& o) {
    return {p, b, t, a, c, o};
}

/// A fixed engine's body made of its loop's OWN phase callables, called
/// with the clock's step index (slice 3c, DECISIONS 18): `cancelled()`,
/// `begin(k)`, `take()`, `after()`, `commit()`, `record(k)`. No adapter
/// between the skeleton and the phases; every phase is whole (take and
/// after_solve never fail or cut: the engine throws on a failed step).
template <class Cancelled, class Begin, class Take, class After, class Commit,
          class Record>
struct IndexPhaseBody {
    Cancelled& cancelled_;
    Begin& begin_;
    Take& take_;
    After& after_;
    Commit& commit_;
    Record& record_;
    PULSIM_STEP_PHASE bool poll() { return !cancelled_(); }
    PULSIM_STEP_PHASE void begin(const StepFrame& f) { begin_(f.k); }
    Real clamp(const StepFrame&, Real h) const noexcept { return h; }
    PULSIM_STEP_PHASE bool take(const StepFrame&) {
        take_();
        return true;
    }
    PULSIM_STEP_PHASE bool after_solve(StepFrame&) {
        after_();
        return true;
    }
    PULSIM_STEP_PHASE void commit(const StepFrame&) { commit_(); }
    PULSIM_STEP_PHASE void on_step(const StepFrame& f) { record_(f.k); }
};

template <class Cancelled, class Begin, class Take, class After, class Commit,
          class Record>
IndexPhaseBody<Cancelled, Begin, Take, After, Commit, Record> index_phase_body(
    Cancelled& x, Begin& b, Take& t, After& a, Commit& c, Record& r) {
    return {x, b, t, a, c, r};
}

/// A deciding body made of an engine's phase callables (slice 3b): the
/// TR-BDF2 loop's phases, the SAME callables its legacy loop calls. They
/// take no frame (the engine reads its own t and h); clamp returns the
/// step it landed; take, error and after_solve return the Verdict; on_step
/// runs after every accepted step, whichever phase committed it.
template <class Poll, class Begin, class Clamp, class Take, class Error, class After,
          class Commit, class OnStep>
struct DecidingPhaseBody {
    static constexpr bool decides = true;
    Poll& poll_;
    Begin& begin_;
    Clamp& clamp_;
    Take& take_;
    Error& error_;
    After& after_;
    Commit& commit_;
    OnStep& on_step_;
    PULSIM_STEP_PHASE bool poll() { return poll_(); }
    PULSIM_STEP_PHASE void begin(const StepFrame&) { begin_(); }
    PULSIM_STEP_PHASE Real clamp(const StepFrame&, Real) { return clamp_(); }
    PULSIM_STEP_PHASE Verdict take(const StepFrame&) { return take_(); }
    PULSIM_STEP_PHASE Verdict error(const StepFrame&) { return error_(); }
    PULSIM_STEP_PHASE Verdict after_solve(StepFrame&) { return after_(); }
    PULSIM_STEP_PHASE void commit(const StepFrame&) { commit_(); }
    PULSIM_STEP_PHASE void on_step(const StepFrame&) { on_step_(); }
};

template <class Poll, class Begin, class Clamp, class Take, class Error, class After,
          class Commit, class OnStep>
DecidingPhaseBody<Poll, Begin, Clamp, Take, Error, After, Commit, OnStep> phase_body(
    Poll& p, Begin& b, Clamp& cl, Take& t, Error& e, After& a, Commit& c, OnStep& o) {
    return {p, b, cl, t, e, a, c, o};
}

/// The run: steps until the clock is done, the body stops it or a step
/// aborts. Returns the last status.
template <Integrator I, StepController C, class Body>
PULSIM_STEP_PHASE inline StepStatus run(C& ctrl, Body& body) {
    StepStatus s = StepStatus::Done;
    if constexpr (body_decides<Body>()) {
        while ((s = step_once<I>(ctrl, body)) == StepStatus::Accepted
               || s == StepStatus::Rejected) {
        }
    } else {
        while ((s = step_once<I>(ctrl, body)) == StepStatus::Accepted) {
        }
    }
    return s;
}

}  // namespace pulsim::step
