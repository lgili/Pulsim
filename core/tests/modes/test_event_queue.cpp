// add-unified-event-queue task 2.3: the queue's merge, its order at one
// instant, its integer-counter times and its half-open run window.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "pulsim/events/event_queue.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"

using pulsim::Real;
using namespace pulsim::events;

namespace {
constexpr Real kInf = std::numeric_limits<Real>::infinity();

// Drains every event of `q` in [t0, t_end) by jumping to the next landing.
std::vector<TimeEvent> drain(EventQueue& q, Real tol = 0) {
    std::vector<TimeEvent> out;
    for (Real t = q.next_landing(); t < kInf; t = q.next_landing()) {
        for (const auto& e : q.take_due(t, tol)) {
            out.push_back(e);
        }
    }
    return out;
}
}  // namespace

TEST_CASE("periodic source: k-th event is t0 + offset + k*T, half-open run",
          "[event_queue]") {
    EventQueue q;
    const auto a = q.add_periodic(Kind::ControlTick, Rate{5e-4});        // 2 kHz
    const auto b = q.add_periodic(Kind::ControlTick, Rate{5e-5, 1e-5});  // 20 kHz, offset
    q.start(0.0, 20e-3);
    const auto ev = drain(q);
    // ceil((t_end - t0 - offset)/T): 40 and ceil(19.99e-3/5e-5) = 400.
    CHECK(q.taken(a) == 40);
    CHECK(q.taken(b) == 400);
    CHECK(ev.size() == 440);
    std::size_t bad = 0;
    for (const auto& e : ev) {
        const Real T = e.source == a ? 5e-4 : 5e-5;
        const Real direct = (e.source == a ? 0.0 : 1e-5)
            + static_cast<Real>(e.ordinal) * T;
        bad += (e.t != q.periodic_time(e.source, e.ordinal)
                || std::abs(e.t - direct) > 1e-12 * T || !(e.t < 20e-3));
    }
    CHECK(bad == 0);
    CHECK(ev.front().t == 0.0);
}

TEST_CASE("a tick exactly at t_end does not fire", "[event_queue]") {
    EventQueue q;
    const auto a = q.add_periodic(Kind::ControlTick, Rate{0.25});
    q.start(0.0, 1.0);
    drain(q);
    CHECK(q.taken(a) == 4);   // 0, .25, .5, .75 — not 1.0
}

TEST_CASE("no drift: event k equals the direct product after 1e6 takes",
          "[event_queue]") {
    EventQueue q;
    const Real T = 1e-5;   // 100 kHz
    const Real t0 = 0.125;
    const auto a = q.add_periodic(Kind::GateEdge, Rate{T});
    q.start(t0, kInf);
    // The reference is computed by the queue's own formula through
    // periodic_time (same expression, same compiler contraction): what is
    // tested is that taking events never accumulates. Compared to 1e-9 of
    // a period against an independent product (contraction differs by an
    // ulp at most).
    Real accumulated = t0;
    Real worst_acc = 0.0;
    std::uint64_t mismatches = 0;
    for (std::uint64_t k = 0; k < 1'000'000; ++k) {
        const Real t = q.next_landing();
        const auto ev = q.take_due(t, 0.0);
        if (ev.size() != 1 || ev[0].ordinal != k
            || ev[0].t != q.periodic_time(a, k)
            || std::abs(ev[0].t - (t0 + static_cast<Real>(k) * T)) > 1e-9 * T) {
            ++mismatches;
        }
        if (!ev.empty()) {
            worst_acc = std::max(worst_acc, std::abs(accumulated - ev[0].t));
        }
        accumulated += T;
    }
    CHECK(mismatches == 0);
    // The test can tell: summing T drifts (by far more than an ulp of t).
    CHECK(worst_acc > 1e-15);
    // 1e7 periods: the k-th event is still the product (to 4 ulp).
    const Real t7 = t0 + 1e7 * T;
    CHECK(std::abs(q.periodic_time(a, 10'000'000) - t7)
          <= 4 * std::numeric_limits<Real>::epsilon() * t7);
}

TEST_CASE("simultaneous events come out in Kind order, then registration",
          "[event_queue]") {
    EventQueue q;
    // Registered in the reverse of the D2 order, two ticks interleaved.
    const auto end = q.add_once(Kind::EndOfRun, 1.0);
    const auto bp = q.add_periodic(Kind::SourceBreakpoint, Rate{0.5});
    const auto gate = q.add_periodic(Kind::GateEdge, Rate{0.25});
    const auto tick1 = q.add_periodic(Kind::ControlTick, Rate{0.5});
    const auto sample = q.add_periodic(Kind::OutputSample, Rate{0.5});
    const auto tick0 = q.add_periodic(Kind::ControlTick, Rate{1.0 / 3.0});
    q.start(0.0, 2.0);
    const auto ev = q.take_due(0.0, 0.0);
    REQUIRE(ev.size() == 5);
    CHECK(ev[0].source == sample);
    CHECK(ev[1].source == tick1);   // ticks in registration order:
    CHECK(ev[2].source == tick0);             // tick1 was registered first
    CHECK(ev[3].source == gate);
    CHECK(ev[4].source == bp);
    // At t = 1: sample, tick1, tick0 (k=3: 1/3*3 rounds to 1.0), gate,
    // breakpoint, end — every kind at one instant.
    const Real t1 = q.periodic_time(tick0, 3);
    CHECK(std::abs(t1 - 1.0) < 1e-15);
    while (q.next_landing() < 1.0 - 1e-12) {
        q.take_due(q.next_landing(), 0.0);
    }
    const auto at1 = q.take_due(1.0, 1e-12);
    REQUIRE(at1.size() == 6);
    CHECK(at1[0].kind == Kind::OutputSample);
    CHECK(at1[1].source == tick1);
    CHECK(at1[2].source == tick0);
    CHECK(at1[3].kind == Kind::GateEdge);
    CHECK(at1[4].kind == Kind::SourceBreakpoint);
    CHECK(at1[5].source == end);
}

TEST_CASE("take_due honours the kind mask and the tolerance", "[event_queue]") {
    EventQueue q;
    const auto tick = q.add_periodic(Kind::ControlTick, Rate{1.0});
    const auto gate = q.add_periodic(Kind::GateEdge, Rate{1.0});
    q.start(0.0, 10.0);
    // Only ticks: the gate event at 0 stays pending.
    auto ev = q.take_due(0.0, 0.0, kind_bit(Kind::ControlTick));
    REQUIRE(ev.size() == 1);
    CHECK(ev[0].source == tick);
    CHECK(q.pending(gate) == 0.0);
    // 1 - 1e-9 is due at tol 1e-8, not at tol 1e-10.
    CHECK(q.take_due(1.0 - 1e-9, 1e-10, kind_bit(Kind::ControlTick)).empty());
    ev = q.take_due(1.0 - 1e-9, 1e-8, kind_bit(Kind::ControlTick));
    REQUIRE(ev.size() == 1);
    CHECK(ev[0].t == 1.0);
    CHECK(q.pending(tick) == 2.0);
}

TEST_CASE("function source: lazy refresh, invalidate, refill from event time",
          "[event_queue]") {
    int calls = 0;
    const std::vector<Real> edges{0.1, 0.3, 0.5, 0.7, 0.9};
    auto next = [&](Real t) {
        ++calls;
        for (const Real e : edges) {
            if (e > t) return e;
        }
        return kInf;
    };
    EventQueue q;
    const auto g = q.add_function(Kind::GateEdge, next);
    q.start(0.0, 1.0);
    CHECK(q.pending(g) == kInf);           // nothing fetched yet
    CHECK(q.next_after(g, 0.0) == 0.1);
    CHECK(calls == 1);
    CHECK(q.next_after(g, 0.05) == 0.1);   // cached: still after t
    CHECK(calls == 1);
    CHECK(q.next_after(g, 0.1) == 0.3);    // stale at its own time: refetch
    CHECK(calls == 2);
    q.invalidate(g);
    CHECK(q.next_after(g, 0.2) == 0.3);
    CHECK(calls == 3);
    const auto ev = q.take_due(0.3, 0.0);
    REQUIRE(ev.size() == 1);
    CHECK(ev[0].ordinal == 0);
    CHECK(q.pending(g) == 0.5);            // refilled from the event's time
}

TEST_CASE("located source: holds the located time until passed",
          "[event_queue]") {
    EventQueue q;
    const auto g = q.add_located(Kind::GateEdge);
    q.start(0.0, 1.0);
    CHECK(q.next_after(g, 0.0) == kInf);
    q.locate(g, 0.25);
    CHECK(q.next_after(g, 0.1) == 0.25);
    CHECK(q.next_landing() == 0.25);
    CHECK(q.next_after(g, 0.25) == kInf);   // reached: dropped
    q.locate(g, 0.5);
    q.invalidate(Kind::GateEdge);
    CHECK(q.next_after(g, 0.3) == kInf);
}

TEST_CASE("a source that does not land is not a landing", "[event_queue]") {
    EventQueue q;
    q.add_periodic(Kind::OutputSample, Rate{0.1}, /*lands=*/false);
    const auto e = q.add_once(Kind::EndOfRun, 1.0);
    q.start(0.0, 1.0);
    CHECK(q.next_landing() == 1.0);
    CHECK(q.next_of(Kind::OutputSample) == 0.0);
    CHECK(q.next_of(Kind::EndOfRun) == 1.0);
    CHECK(q.take_due(1.0, 0.0, kind_bit(Kind::EndOfRun))[0].source == e);
    CHECK(q.next_of(Kind::EndOfRun) == kInf);
}

TEST_CASE("bad rates are refused", "[event_queue]") {
    EventQueue q;
    CHECK_THROWS_AS(q.add_periodic(Kind::ControlTick, Rate{0.0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(q.add_periodic(Kind::ControlTick, Rate{-1.0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(q.add_periodic(Kind::ControlTick, Rate{kInf}),
                    std::invalid_argument);
    CHECK_THROWS_AS(q.add_periodic(Kind::ControlTick, Rate{1.0, -0.5}),
                    std::invalid_argument);
}

TEST_CASE("restart re-arms every source and resets the counters",
          "[event_queue]") {
    EventQueue q;
    const auto a = q.add_periodic(Kind::ControlTick, Rate{0.5});
    q.start(0.0, 2.0);
    drain(q);
    CHECK(q.taken(a) == 4);
    q.start(10.0, 11.0);
    CHECK(q.taken(a) == 0);
    CHECK(q.next_landing() == 10.0);
    drain(q);
    CHECK(q.taken(a) == 2);
}

TEST_CASE("invalidate leaves periodic and one-shot sources alone; an oracle "
          "stuck at its own event is not re-taken", "[event_queue]") {
    EventQueue q;
    const auto tick = q.add_periodic(Kind::ControlTick, Rate{0.5});
    const auto end = q.add_once(Kind::EndOfRun, 1.0);
    const auto stuck = q.add_function(Kind::GateEdge, [](Real) { return 0.25; });
    q.start(0.0, 1.0);
    q.invalidate(tick);
    q.invalidate(end);
    CHECK(q.next_of(Kind::ControlTick) == 0.0);
    CHECK(q.next_of(Kind::EndOfRun) == 1.0);
    CHECK(q.next_after(stuck, 0.0) == 0.25);
    CHECK(q.take_due(0.25, 0.0, kind_bit(Kind::GateEdge)).size() == 1);
    CHECK(q.pending(stuck) == kInf);   // 0.25 is not after its own event
    CHECK(q.take_due(0.25, 0.0, kind_bit(Kind::GateEdge)).empty());
}

// ---- a short pulse at a period boundary (the exact engine's dropped
// edge pair: a PWM at D = 1 - 1e-7 lost both edges and gave 0 W) --------
namespace {
// Walks a gate source the way an engine does: land on the next edge,
// take what is due there (every call until empty), repeat.
std::vector<Real> walk_edges(EventQueue& q, std::uint32_t g, Real t_end,
                             Real tol) {
    std::vector<Real> out;
    Real t = 0.0;
    while (true) {
        const Real e = q.next_after(g, t);
        if (!(e < t_end)) break;
        t = e;
        for (auto ev = q.take_due(t, tol); !ev.empty();
             ev = q.take_due(t, tol)) {
            for (const auto& x : ev) out.push_back(x.t);
        }
    }
    return out;
}
}  // namespace

TEST_CASE("both edges of a pulse shorter than the tolerance come out, in "
          "order, at their own instants", "[event_queue]") {
    const Real T = 1e-5;
    const pulsim::sources::NativePwm2Switch pwm(T, 1.0 - 1e-7, 2);
    for (const Real tol : {0.0, 1e-15, 1e-11}) {   // pulse is ~1e-12 s
        EventQueue q;
        const auto g = q.add_function(
            Kind::GateEdge, [&](Real t) { return pwm.next_edge_after(t); });
        q.start(0.0, 1.0);
        // Ends mid-period: an edge within tol of the last landing is due
        // there and taken with it (3T would ride with 3T - 1e-12).
        const auto e = walk_edges(q, g, 2.5 * T, tol);
        INFO("tol = " << tol);
        // D*T, T, T + D*T, 2T: the pair at every boundary.
        REQUIRE(e.size() == 4);
        CHECK(std::is_sorted(e.begin(), e.end()));
        CHECK(std::adjacent_find(e.begin(), e.end()) == e.end());
        for (std::size_t i = 0; i < e.size(); ++i) {
            CHECK(e[i] == pwm.next_edge_after(i == 0 ? 0.0 : e[i - 1]));
        }
        // Each edge changes the mask (no edge is a no-op or merged away).
        // Compared at interval midpoints: pwm_phase snaps 64 ulps, so one
        // ulp before an edge already reads the post-edge mask.
        std::vector<Real> b{0.0};
        b.insert(b.end(), e.begin(), e.end());
        b.push_back(pwm.next_edge_after(e.back()));
        for (std::size_t i = 1; i + 1 < b.size(); ++i) {
            CHECK(!(pwm(0.5 * (b[i - 1] + b[i])) == pwm(0.5 * (b[i] + b[i + 1]))));
        }
    }
}

TEST_CASE("a one-ulp pulse at a period boundary keeps both edges in order",
          "[event_queue]") {
    // Edges at k*T - 1ulp and k*T (k = 1..1000): the shortest pulse a
    // double can hold. The queue takes events by identity, never merges
    // two events of one source by time.
    const Real T = 1e-5;
    auto next = [T](Real t) {
        const Real k = std::floor(t / T);
        for (Real kk = k; kk <= k + 2; kk += 1) {
            const Real hi = kk * T;
            const Real lo = std::nextafter(hi, 0.0);
            if (lo > t) return lo;
            if (hi > t) return hi;
        }
        return kInf;
    };
    EventQueue q;
    const auto g = q.add_function(Kind::GateEdge, next);
    q.start(0.0, 1.0);
    const auto e = walk_edges(q, g, 1000.5 * T, /*tol=*/1e-12);
    REQUIRE(e.size() == 2000);
    std::size_t bad = 0;
    for (std::size_t i = 0; i + 1 < e.size(); i += 2) {
        bad += !(e[i] < e[i + 1]) || e[i + 1] != std::nextafter(e[i], kInf);
    }
    CHECK(bad == 0);
}
