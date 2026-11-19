// add-probes-and-recorders: probes (record/probe.hpp) and recorders
// (record/recorder.hpp) against closed forms.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/record/loop_feed.hpp"
#include "pulsim/record/recorder.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using namespace pulsim;
using record::Probe;
using record::Recorder;
using record::RecorderSpec;

namespace {

Recorder make(std::vector<std::string> texts, RecorderSpec spec,
              std::vector<std::string> readings) {
    std::vector<Probe> ps;
    for (const auto& s : texts) ps.push_back(Probe::compile(s));
    Recorder r(std::move(ps), std::move(spec));
    r.bind(readings);
    return r;
}

bool close(Real a, Real b, Real rel) { return std::abs(a - b) <= rel * std::max(std::abs(b), Real{1}); }

// Triangle of period 1, 0 at integers, 1 at half-integers: piecewise linear
// with corners at k/2.
Real tri(Real t) {
    const Real f = t - std::floor(t);
    return f < 0.5 ? 2 * f : 2 - 2 * f;
}
// integral of tri from 0 to t, and of tri^2
Real tri_int(Real t) {
    const Real n = std::floor(t), f = t - n;
    // rising 2f -> f^2 ; falling 2-2f from 0.5: 0.25 + 2(f-.5) - (f^2 - .25)
    const Real in = f < 0.5 ? f * f : 0.25 + 2 * (f - 0.5) - (f * f - 0.25);
    return 0.5 * n + in;
}
Real tri_int2(Real t) {
    const Real n = std::floor(t), f = t - n;
    // rising: (2f)^2 -> 4f^3/3 ; falling: (2-2f)^2 = 4(1-f)^2 -> 1/6 + [-(4/3)(1-f)^3]_{.5}^{f}
    const Real in = f < 0.5 ? 4 * f * f * f / 3
                            : 1.0 / 6 + (4.0 / 3) * (0.125 - (1 - f) * (1 - f) * (1 - f));
    return n / 3 + in;
}

}  // namespace

TEST_CASE("probe: readings are cut out and compiled by ir::Expr", "[record]") {
    const auto p = Probe::compile("v( dp ) - v(dn) + 0.5*abs(i(D1)) - v(dp)");
    REQUIRE(p.atoms().size() == 3);
    CHECK(p.atoms()[0] == "v(dp)");
    CHECK(p.atoms()[1] == "v(dn)");
    CHECK(p.atoms()[2] == "i(D1)");
    const Real vals[] = {10.0, 4.0, -2.0};
    CHECK(p.eval(vals) == 10.0 - 4.0 + 0.5 * 2.0 - 10.0);
    CHECK(Probe::compile("v(a, b)").atoms()[0] == "v(a,b)");
    CHECK(Probe::compile("pi*p(M1)").atoms()[0] == "p(M1)");
    CHECK_THROWS_AS(Probe::compile("v(out"), std::invalid_argument);
    CHECK_THROWS_AS(Probe::compile("out + 1"), std::invalid_argument);   // bare name
    CHECK_THROWS_AS(Probe::compile("q(out)"), std::invalid_argument);    // unknown fn
    CHECK_THROWS_AS(Probe::compile("v(abs(x))"), std::invalid_argument);
    CHECK_THROWS_AS(Probe::compile("v()"), std::invalid_argument);
    // an identifier ending in v is not a reading
    CHECK_THROWS_AS(Probe::compile("dv(x)"), std::invalid_argument);
}

TEST_CASE("recorder: unbound or unknown readings are refused by name", "[record]") {
    std::vector<Probe> ps{Probe::compile("v(out)*i(L1)")};
    Recorder r(std::move(ps), {});
    const Real x[] = {1.0};
    CHECK_THROWS_AS(r.sample(0.0, x), std::logic_error);
    const std::vector<std::string> names{"v(out)"};
    try {
        r.bind(names);
        FAIL("bind accepted a missing reading");
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("i(L1)") != std::string::npos);
    }
}

TEST_CASE("recorder: windows clipped mid-step are exact for the piecewise-linear signal",
          "[record]") {
    // A clustered grid that contains every corner of the triangle, so the
    // sampled line IS the signal; windows start and end mid-step.
    std::vector<Real> t;
    for (int k = 0; k <= 8; ++k) {          // corners at k/2, 0..4
        const Real a = 0.5 * k;
        t.push_back(a);
        if (k < 8) {
            for (int j = 1; j < 7; ++j) t.push_back(a + 0.5 * std::pow(j / 7.0, 3));
        }
    }
    RecorderSpec spec;
    spec.windows = {{0.3, 3.7}, {1.0 / 3, 2.0 + 1.0 / 7}, {0.0, 4.0}};
    spec.period = 1.0;
    spec.offset = 0.25;
    auto r = make({"v(x)", "2*v(x) - 1"}, spec, {"v(x)"});
    r.reserve(t.size(), t.back());
    for (const Real tk : t) {
        const Real y = tri(tk);
        r.sample(tk, std::span<const Real>(&y, 1));
    }
    for (Size w = 0; w < spec.windows.size(); ++w) {
        const auto [a, b] = spec.windows[w];
        const auto& s = r.windows()[w * 2];
        const Real I = tri_int(b) - tri_int(a), I2 = tri_int2(b) - tri_int2(a);
        INFO("window " << a << " .. " << b);
        CHECK(s.complete());
        CHECK(close(s.integral, I, 1e-12));
        CHECK(close(s.avg(), I / (b - a), 1e-12));
        CHECK(close(s.rms(), std::sqrt(I2 / (b - a)), 1e-12));
        // the affine probe: avg(2y-1) = 2avg(y) - 1
        CHECK(close(r.windows()[w * 2 + 1].avg(), 2 * I / (b - a) - 1, 1e-12));
    }
    // [0.3, 3.7] holds peaks at 0.5..3.5 and minima at 1, 2, 3: first
    // occurrence wins; the clipped ends are not extrema here.
    const auto& s0 = r.windows()[0];
    CHECK(s0.max == 1.0);
    CHECK(s0.t_max == 0.5);
    CHECK(s0.min == 0.0);
    CHECK(s0.t_min == 1.0);
    // periods [0.25 + k, 1.25 + k]: k = 0, 1, 2 complete, avg 1/2, rms 1/sqrt(3)
    const auto& P = r.periods();
    Size n_complete = 0;
    for (Size k = 0; k < P.size() / 2; ++k) {
        if (!P[k * 2].complete()) continue;
        ++n_complete;
        CHECK(close(P[k * 2].avg(), 0.5, 1e-12));
        CHECK(close(P[k * 2].rms(), 1 / std::sqrt(3.0), 1e-12));
    }
    CHECK(n_complete == 3);
    CHECK_FALSE(P[3 * 2].complete());   // [3.25, 4.25] runs past the record
}

TEST_CASE("recorder: both limits at an event; windows clip on the right side of each",
          "[record]") {
    // A square wave 0 / 1 with jumps at t = 1 and t = 2, recorded as an
    // engine with right limits stores it: left limit at the event row,
    // the right limit through event().
    RecorderSpec spec;
    spec.windows = {{1.0, 2.0}, {0.5, 1.5}, {0.25, 2.75}};
    auto r = make({"v(s)"}, spec, {"v(s)"});
    auto feed = [&](Real t, Real left) { r.sample(t, std::span<const Real>(&left, 1)); };
    auto jump = [&](Real right) { r.event(std::span<const Real>(&right, 1)); };
    feed(0.0, 0.0);
    feed(0.6, 0.0);
    feed(1.0, 0.0);
    jump(1.0);
    feed(1.3, 1.0);
    feed(2.0, 1.0);
    jump(0.0);
    feed(3.0, 0.0);
    const auto& w = r.windows();
    // [1, 2] starts on the right limit (1) and ends on the left limit (1)
    CHECK(w[0].avg() == 1.0);
    CHECK(w[0].min == 1.0);
    CHECK(w[0].max == 1.0);
    CHECK(w[0].complete());
    // [0.5, 1.5]: half low, half high; both limits of t = 1 inside
    CHECK(w[1].avg() == 0.5);
    CHECK(w[1].min == 0.0);
    CHECK(w[1].max == 1.0);
    CHECK(w[1].t_max == 1.0);
    CHECK(w[2].avg() == Catch::Approx(1.0 / 2.5).epsilon(1e-14));
    REQUIRE(r.event_times().size() == 2);
    CHECK(r.event_times()[0] == 1.0);
    CHECK(r.event_left()[0] == 0.0);
    CHECK(r.event_right()[0] == 1.0);
    CHECK(r.event_left()[1] == 1.0);
    CHECK(r.event_right()[1] == 0.0);
    // without the right limits the jump is a ramp over the next step: the
    // error a left-limit-only record makes
    auto l = make({"v(s)"}, spec, {"v(s)"});
    for (auto [t, y] : {std::pair{0.0, 0.0}, {0.6, 0.0}, {1.0, 0.0}, {1.3, 1.0},
                        {2.0, 1.0}, {3.0, 0.0}}) {
        l.sample(t, std::span<const Real>(&y, 1));
    }
    CHECK(l.windows()[0].avg() < 0.9);
}

TEST_CASE("recorder: feed_table equals sample/event; decimation keeps every n-th row",
          "[record]") {
    const std::vector<Real> t{0, 1, 2, 3, 4, 5, 6};
    const std::vector<Real> x{0, 0, 1, 1, 2, 3, 4};   // row-major, one reading
    const std::vector<Size> ev{2};
    const std::vector<Real> right{5};
    RecorderSpec spec;
    spec.keep = RecorderSpec::Keep::Decimate;
    spec.every_n = 3;
    spec.windows = {{0.5, 5.5}};
    auto a = make({"v(n)"}, spec, {"v(n)"});
    record::feed_table(a, t, x, 1, ev, right);
    auto b = make({"v(n)"}, spec, {"v(n)"});
    for (Size k = 0; k < t.size(); ++k) {
        b.sample(t[k], std::span<const Real>(&x[k], 1));
        if (k == 2) b.event(right);
    }
    CHECK(a.windows()[0].integral == b.windows()[0].integral);
    CHECK(a.times() == std::vector<Real>{0, 3, 6});
    CHECK(a.values() == std::vector<Real>{0, 1, 4});
    CHECK(a.event_times() == std::vector<Real>{2});
    CHECK(a.event_right() == std::vector<Real>{5});
}

TEST_CASE("recorder: a long statistics-only or decimating record does not grow per sample",
          "[record][zero_alloc]") {
    for (const auto keep : {RecorderSpec::Keep::None, RecorderSpec::Keep::Decimate}) {
        RecorderSpec spec;
        spec.keep = keep;
        spec.every_n = 100;
        spec.windows = {{0.1, 0.9}};
        spec.period = 1e-3;
        auto r = make({"v(a)", "v(a)*v(b)"}, spec, {"v(a)", "v(b)"});
        constexpr Size n = 200000;
        const Real t_end = 1.0;
        r.reserve(n, t_end);
        const auto cap_t = r.times().capacity(), cap_v = r.values().capacity();
        const auto cap_p = r.periods().capacity();
        const auto* p0 = r.periods().data();
        for (Size k = 0; k < n; ++k) {
            const Real t = t_end * static_cast<Real>(k) / static_cast<Real>(n - 1);
            const Real y[] = {std::sin(2 * std::numbers::pi * 1e3 * t), 2.0};
            r.sample(t, y);
        }
        CHECK(r.times().capacity() == cap_t);
        CHECK(r.values().capacity() == cap_v);
        CHECK(r.periods().capacity() == cap_p);
        CHECK(r.periods().data() == p0);
        CHECK(r.times().size() == (keep == RecorderSpec::Keep::None ? 0 : n / 100));
        // 1000 periods of a sine, 200 samples each, edges mid-step: the
        // line's average is 0 up to O((w h)^2 h / T) ~ 1e-6
        Size n_complete = 0;
        for (Size k = 0; k < r.periods().size() / 2; ++k) {
            if (!r.periods()[2 * k].complete()) continue;
            ++n_complete;
            CHECK(std::abs(r.periods()[2 * k].avg()) < 1e-5);
        }
        CHECK(n_complete == 1000);
    }
}

TEST_CASE("recorder: an edge within 4 ulps of a sample is that sample (review findings)",
          "[record]") {
    // 3 * 0.1 = 0.30000000000000004 is one ulp above a run that ended at 0.3
    RecorderSpec spec;
    spec.period = 0.1;
    auto r = make({"v(x)"}, spec, {"v(x)"});
    for (int k = 0; k <= 30; ++k) {
        const Real t = k == 30 ? 0.3 : 0.01 * k;
        const Real y = 1.0;
        r.sample(t, std::span<const Real>(&y, 1));
    }
    Size n_complete = 0;
    for (const auto& w : r.periods()) n_complete += w.complete();
    CHECK(n_complete == 3);
    // an edge one ulp before an event starts on the right limit
    RecorderSpec s2;
    s2.windows = {{std::nextafter(1.0, 0.0), 2.0}};
    auto e = make({"v(s)"}, s2, {"v(s)"});
    const Real lo = 0.0, hi = 1.0;
    e.sample(0.0, std::span<const Real>(&lo, 1));
    e.sample(1.0, std::span<const Real>(&lo, 1));
    e.event(std::span<const Real>(&hi, 1));
    e.sample(2.0, std::span<const Real>(&hi, 1));
    CHECK(e.windows()[0].min == 1.0);
    CHECK(e.windows()[0].avg() == 1.0);
}

TEST_CASE("probe / recorder: malformed declarations are refused (review findings)",
          "[record]") {
    CHECK_THROWS_AS(Probe::compile("v(out) + __a0"), std::invalid_argument);
    CHECK_THROWS_AS(Probe::compile("v(a b)"), std::invalid_argument);
    CHECK(Probe::compile("v( a , b )").atoms()[0] == "v(a,b)");
    RecorderSpec spec;
    spec.period = 1e-3;
    spec.offset = std::numeric_limits<Real>::quiet_NaN();
    CHECK_THROWS_AS(Recorder({Probe::compile("v(x)")}, spec), std::invalid_argument);
    RecorderSpec tiny;
    tiny.period = 1e-15;
    Recorder r({Probe::compile("v(x)")}, tiny);
    CHECK_THROWS_AS(r.reserve(10, 1.0), std::invalid_argument);
}

TEST_CASE("recorder: Times rows are linear inside a step and the right limit at an event",
          "[record]") {
    // the square wave of the limits test: 0 until t = 1, jump to 1, back to 0 at t = 2
    RecorderSpec spec;
    spec.keep = RecorderSpec::Keep::Times;
    spec.sample_times = {0.0, 0.3, 1.0, 1.15, 2.0, 2.5, 3.0};
    auto r = make({"v(s)"}, spec, {"v(s)"});
    auto feed = [&](Real t, Real left) { r.sample(t, std::span<const Real>(&left, 1)); };
    auto jump = [&](Real right) { r.event(std::span<const Real>(&right, 1)); };
    feed(0.0, 0.0);
    feed(0.6, 0.6);   // a ramp 0 -> 0.6 on the first step
    feed(1.0, 0.0);
    jump(1.0);
    feed(1.3, 1.0);
    feed(2.0, 1.0);
    jump(0.0);
    feed(3.0, 0.0);
    r.flush();
    CHECK(r.times() == spec.sample_times);
    const auto& v = r.values();
    REQUIRE(v.size() == 7);
    CHECK(v[0] == 0.0);
    CHECK(v[1] == Catch::Approx(0.3).epsilon(1e-15));
    CHECK(v[2] == 1.0);   // t = 1: the right limit
    CHECK(v[3] == 1.0);
    CHECK(v[4] == 0.0);   // t = 2: the right limit
    CHECK(v[5] == 0.0);
    CHECK(v[6] == 0.0);   // the last sample, through flush()
    CHECK(r.interpolated() == std::vector<std::uint8_t>{0, 1, 0, 1, 0, 1, 0});
    CHECK(r.n_times_unreached() == 0);
}

TEST_CASE("recorder: Times outside the record are counted, not invented", "[record]") {
    RecorderSpec spec;
    spec.keep = RecorderSpec::Keep::Times;
    spec.sample_times = {-1.0, 0.5, 5.0};
    auto r = make({"v(s)"}, spec, {"v(s)"});
    const std::vector<Real> t{0, 1}, x{0, 2};
    record::feed_table(r, t, x, 1, {}, {});
    CHECK(r.times() == std::vector<Real>{0.5});
    CHECK(r.values() == std::vector<Real>{1.0});
    CHECK(r.n_times_unreached() == 2);
    RecorderSpec bad;
    bad.keep = RecorderSpec::Keep::Times;
    bad.sample_times = {0.5, 0.5};
    CHECK_THROWS_AS(make({"v(s)"}, bad, {"v(s)"}), std::invalid_argument);
}

TEST_CASE("recorder: RingSink pushes every kept row into a LiveRing", "[record]") {
    RecorderSpec spec;
    spec.keep = RecorderSpec::Keep::Decimate;
    spec.every_n = 2;
    auto r = make({"v(a)", "2*v(a)"}, spec, {"v(a)"});
    streaming::LiveRing ring(16, 2, 1);
    r.stream_to(&ring);
    for (int k = 0; k < 5; ++k) {
        const Real y = k;
        r.sample(k, std::span<const Real>(&y, 1));
    }
    CHECK(ring.head() == 3);   // rows 0, 2, 4
    streaming::LiveRing wrong(16, 3, 1);
    CHECK_THROWS_AS(r.stream_to(&wrong), std::invalid_argument);
}

TEST_CASE("recorder: attach() with Keep::Every survives an undercounting reserve",
          "[record]") {
    // the caller sizes the kept table; too small a reserve regrows (one
    // allocation per doubling) and loses nothing
    auto r = make({"v(a)", "v(b) - v(a)"}, RecorderSpec{}, {"v(a)", "v(b)"});
    r.reserve(2, 0.0);
    events::StepHooks hooks;
    record::attach(hooks, r, {1, -1});   // v(a) = x[1], v(b) = ground
    Vector x(2);
    const topology::SwitchStateMask mask(1);
    for (int k = 1; k <= 1000; ++k) {
        x << 0.0, static_cast<Real>(k);
        hooks.accepted(k - 1, k, x, mask);
    }
    REQUIRE(r.times().size() == 1000);
    CHECK(r.times().front() == 1.0);
    CHECK(r.times().back() == 1000.0);
    CHECK(r.values()[2 * 999] == 1000.0);
    CHECK(r.values()[2 * 999 + 1] == -1000.0);
}

TEST_CASE("recorder: several requested times within 4 ulps of the last sample all land",
          "[record]") {
    RecorderSpec spec;
    spec.keep = RecorderSpec::Keep::Times;
    spec.sample_times = {std::nextafter(1.0, 0.0), 1.0};
    auto r = make({"v(s)"}, spec, {"v(s)"});
    const std::vector<Real> t{0, 1}, x{0, 2};
    record::feed_table(r, t, x, 1, {}, {});
    CHECK(r.values() == std::vector<Real>{2.0, 2.0});
    CHECK(r.n_times_unreached() == 0);
}

TEST_CASE("LoopFeed settled mode: reported, unreported and jump events in the record",
          "[record][loop_feed]") {
    // the variable-step engine's conventions, driven by hand: a settled
    // on_event after the step that lands t (both limits at t, the next line
    // starts from the right one); a mode change without a report (bracketed
    // with the first step under the new mode); a restart from another state
    // than committed (a settled jump row).
    using topology::SwitchStateMask;
    record::RecorderSpec spec;
    spec.keep = record::RecorderSpec::Keep::Every;
    spec.windows = {{0.0, 5.0}};
    std::vector<record::Recorder> recs;
    recs.emplace_back(std::vector<record::Probe>{record::Probe::compile("v(a)")}, spec);
    record::StateReading va;
    va.a.cols = {0};
    va.a.coef = {1.0};
    record::LoopFeed feed({"v(a)"}, {va}, std::move(recs));
    feed.attach_own(false);
    auto& h = feed.hooks();
    SwitchStateMask m0(1), m1(1);
    m1.set(0, true);
    auto x = [](Real v) { Vector y(1); y[0] = v; return y; };
    h.step(0, x(0), m0, 1, x(1), m0);
    h.event(1, x(1), m0, x(3), m1);       // settled at t = 1: 1 -> 3
    h.step(1, x(1), m1, 2, x(3), m1);     // starts from the committed x: no jump row
    h.step(2, x(3), m1, 3, x(3), m0);     // solved under a new mode, not reported: bracket [2, 3]
    h.step(3, x(3), m0, 4, x(5), m0);
    h.step(4, x(7), m0, 5, x(7), m0);     // restarted from 7, committed 5: jump
    feed.finish();
    const auto& r = feed.recorders().front();
    using R = record::Recorder::Resolution;
    REQUIRE(r.event_times() == std::vector<Real>{1, 2, 4});
    CHECK(r.event_t_right() == std::vector<Real>{1, 3, 4});
    CHECK(r.event_resolution() == std::vector<std::uint8_t>{
        static_cast<std::uint8_t>(R::Settled), static_cast<std::uint8_t>(R::WithinDt),
        static_cast<std::uint8_t>(R::Settled)});
    CHECK(r.event_left() == std::vector<Real>{1, 3, 5});
    CHECK(r.event_right() == std::vector<Real>{3, 3, 7});
    CHECK(feed.n_jumps() == 1);
    CHECK(feed.n_events() == 2);
    // the line: [0,1] 0->1, [1,2] 3->3 (from the right limit), [2,3] 3->3,
    // [3,4] 3->5 (a bracket does not move the line), [4,5] 7->7
    CHECK(r.windows()[0].integral == Catch::Approx(0.5 + 3 + 3 + 4 + 7).epsilon(1e-15));
    CHECK(r.times() == std::vector<Real>{0, 1, 2, 3, 4, 5});
    CHECK_THROWS_AS(feed.attach_own(false), std::logic_error);
}

TEST_CASE("LoopFeed settled mode: a committed post-event probe merged into one hook step",
          "[record][loop_feed]") {
    // trbdf2 at a landed gate edge: on_event(t_ev) BEFORE the one step hook
    // that covers [t0, t_ev + dtp]; the left limit at t_ev comes from the
    // event, and the line restarts from the settled right side.
    using topology::SwitchStateMask;
    record::RecorderSpec spec;
    spec.keep = record::RecorderSpec::Keep::Every;
    std::vector<record::Recorder> recs;
    recs.emplace_back(std::vector<record::Probe>{record::Probe::compile("v(a)")}, spec);
    record::StateReading va;
    va.a.cols = {0};
    va.a.coef = {1.0};
    record::LoopFeed feed({"v(a)"}, {va}, std::move(recs));
    feed.attach_own(false);
    auto& h = feed.hooks();
    SwitchStateMask m0(1), m1(1);
    m1.set(0, true);
    auto x = [](Real v) { Vector y(1); y[0] = v; return y; };
    h.step(0, x(0), m0, 1, x(1), m0);
    h.event(2, x(2), m0, x(4), m1);
    h.step(1, x(1), m0, 2.5, x(4), m1);
    h.step(2.5, x(4), m1, 3, x(4), m1);
    const auto& r = feed.recorders().front();
    CHECK(feed.n_dropped() == 0);
    CHECK(feed.n_jumps() == 0);
    CHECK(r.times() == std::vector<Real>{0, 1, 2, 2.5, 3});
    CHECK(r.values() == std::vector<Real>{0, 1, 2, 4, 4});
    REQUIRE(r.event_times() == std::vector<Real>{2});
    CHECK(r.event_left() == std::vector<Real>{2});
    CHECK(r.event_right() == std::vector<Real>{4});
}

TEST_CASE("recorder: period windows equal the same windows declared (fast path oracle)",
          "[record]") {
    // The period path skips the windows a piece cannot touch and, inside a
    // window's interior, accumulates that window alone. Declared windows
    // visit every window on every piece: with the period windows' own
    // multiplied edges they are the oracle (bit for bit here; a tolerance
    // for other compilers' contraction) — on a jittered
    // grid with samples within a few ulps of the edges and events.
    for (const Real off : {0.0, 3.7e-7, -2.5e-5}) {
        const Real T = 1e-4;
        const Size K = 40;
        RecorderSpec ps;
        ps.keep = RecorderSpec::Keep::None;
        ps.period = T;
        ps.offset = off;
        RecorderSpec ds;
        ds.keep = RecorderSpec::Keep::None;
        for (Size k = 0; k < K; ++k) {
            ds.windows.push_back({off + static_cast<Real>(k) * T, off + static_cast<Real>(k + 1) * T});
        }
        auto a = make({"v(x)", "v(y)"}, ps, {"v(x)", "v(y)"});
        auto b = make({"v(x)", "v(y)"}, ds, {"v(x)", "v(y)"});
        const Real t_last = off + static_cast<Real>(K) * T;
        a.reserve(0, t_last);
        std::uint64_t s = 12345;
        const auto rnd = [&]() {  // splitmix64 -> [0, 1)
            s += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return static_cast<Real>((z ^ (z >> 31)) >> 11) * 0x1.0p-53;
        };
        Real t = off;
        std::vector<Real> row(2);
        Size k_edge = 1;
        while (t < t_last) {
            row[0] = std::sin(1e4 * t) + rnd();
            row[1] = rnd() - 0.5;
            a.sample(t, row);
            b.sample(t, row);
            if (rnd() < 0.02) {  // an event: a right limit at t
                row[0] += 1;
                a.event(row);
                b.event(row);
            }
            Real tn = t + 1e-6 * (0.2 + 2 * rnd());
            const Real edge = off + static_cast<Real>(k_edge) * T;
            if (tn >= edge - 1e-6) {
                // land on the edge, or a few ulps either side of it
                const int ulps = static_cast<int>(rnd() * 13) - 6;
                tn = edge;
                for (int u = 0; u < std::abs(ulps); ++u) {
                    tn = std::nextafter(tn, ulps < 0 ? -1.0 : 1.0);
                }
                if (!(tn > t)) tn = std::nextafter(t, 1.0);
                ++k_edge;
            }
            t = tn;
        }
        a.sample(t, row);
        b.sample(t, row);
        REQUIRE(a.periods().size() >= 2 * K);
        for (Size k = 0; k < K; ++k) {
            for (Size p = 0; p < 2; ++p) {
                const auto& x = a.periods()[k * 2 + p];
                const auto& y = b.windows()[k * 2 + p];
                INFO("offset " << off << " window " << k << " probe " << p);
                CHECK(x.t0 == y.t0);
                CHECK(x.t1 == y.t1);
                CHECK(x.integral == Catch::Approx(y.integral).epsilon(1e-12).margin(1e-300));
                CHECK(x.sq3 == Catch::Approx(y.sq3).epsilon(1e-12).margin(1e-300));
                CHECK(x.min == Catch::Approx(y.min).epsilon(1e-12).margin(1e-300));
                CHECK(x.max == Catch::Approx(y.max).epsilon(1e-12).margin(1e-300));
                CHECK((x.t_min == y.t_min || (std::isnan(x.t_min) && std::isnan(y.t_min))));
                CHECK((x.t_max == y.t_max || (std::isnan(x.t_max) && std::isnan(y.t_max))));
                CHECK(x.complete() == y.complete());
            }
        }
    }
}

TEST_CASE("LoopFeed: the flat reading tables read what StateReading::eval reads",
          "[record][loop_feed]") {
    // v(n) / a state column (one entry, coefficient 1: the unit table),
    // v(a,b) and a resistor's current (a linear form with a divisor: the
    // linear table) and a product (the general path), on the fast step and
    // on steps the general path takes (the first, a jump), with -0 and NaN
    // entries in the state.
    using topology::SwitchStateMask;
    std::vector<record::StateReading> rs(4);
    rs[0].a.cols = {1};
    rs[0].a.coef = {1.0};
    rs[1].a.cols = {0, 2};
    rs[1].a.coef = {1.0, -1.0};
    rs[2].a.cols = {0, 1};
    rs[2].a.coef = {1.0, -1.0};
    rs[2].a.div = 4.7;
    rs[3].a = rs[1].a;
    rs[3].product = true;
    rs[3].b.cols = {2};
    rs[3].b.coef = {1.0};
    const std::vector<std::string> names{"v(b)", "v(a,c)", "i(R)", "p(X)"};
    record::RecorderSpec spec;
    spec.keep = record::RecorderSpec::Keep::Every;
    std::vector<record::Probe> ps;
    for (const auto& n : names) ps.push_back(record::Probe::compile(n));
    std::vector<record::Recorder> recs;
    recs.emplace_back(ps, spec);
    record::LoopFeed feed(names, rs, std::move(recs));
    feed.attach_own(true);
    auto& h = feed.hooks();
    SwitchStateMask m(1);
    const auto vec = [](Real a, Real b, Real c) { Vector y(3); y << a, b, c; return y; };
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const std::vector<Vector> xs{vec(1, -0.0, 2), vec(0.5, 3, -1), vec(-0.0, nan, 4),
                                 vec(2, 1, 1), vec(7, -0.0, 0.25)};
    h.step(0, xs[0], m, 1, xs[1], m);
    h.step(1, xs[1], m, 2, xs[2], m);
    h.step(2, xs[2], m, 3, xs[3], m);
    h.step(3, xs[0], m, 4, xs[4], m);  // a jump: the general path
    feed.finish();
    const auto& r = feed.recorders().front();
    REQUIRE(r.times().size() == 5);
    CHECK(feed.n_jumps() == 1);
    const std::vector<Size> at{0, 1, 2, 3, 4};
    for (Size row = 0; row < at.size(); ++row) {
        for (Size j = 0; j < rs.size(); ++j) {
            const Real want = rs[j].eval(xs[at[row]]);
            const Real got = r.values()[row * rs.size() + j];
            INFO("row " << row << " reading " << names[j]);
            if (std::isnan(want)) {
                CHECK(std::isnan(got));
            } else {
                CHECK(got == Catch::Approx(want).epsilon(1e-15).margin(1e-300));
                CHECK(std::signbit(got) == std::signbit(want));
            }
        }
    }
}
