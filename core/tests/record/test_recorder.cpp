// add-probes-and-recorders: probes (record/probe.hpp) and recorders
// (record/recorder.hpp) against closed forms.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/record/recorder.hpp"

#include <cmath>
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
