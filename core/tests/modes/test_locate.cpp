// add-unified-event-queue tasks 8.1 / 8.2: the one locator (Illinois over a
// trial evaluator) on analytic crossings, and the mode-set service's settle
// loop and per-call cycle memory.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>
#include <vector>

#include "pulsim/events/locate.hpp"
#include "pulsim/events/mode_set.hpp"

using pulsim::Real;
using pulsim::Size;
using namespace pulsim::events;

namespace {
// Locates the zero of g in [a, b] to a bracket of width `tol`.
template <class G>
Bracket locate(G g, Real a, Real b, Real tol, int* trials = nullptr,
               std::vector<BracketEnd>* ends = nullptr) {
    Bracket br{a, g(a), b, g(b), g(a), g(b), 0};
    const int n = illinois(
        br, 60, [](Real, const Bracket&) { return false; },
        [&](Real t, Real& s) {
            s = g(t);
            return true;
        },
        [&](BracketEnd e) {
            if (ends) ends->push_back(e);
        },
        [&](const Bracket& x) { return x.tb - x.ta < tol; });
    if (trials) *trials = n;
    return br;
}
}  // namespace

TEST_CASE("illinois brackets an analytic crossing", "[event_locate]") {
    SECTION("linear: the secant root is exact, the clamp then walks in") {
        const auto br = locate([](Real t) { return t - 0.3; }, 0.0, 1.0, 1e-9);
        CHECK(br.ta <= 0.3);
        CHECK(br.tb >= 0.3);
        CHECK(br.tb - br.ta < 1e-9);
    }
    SECTION("sin(t) crosses at pi; signs at the ends are kept") {
        int n = 0;
        const auto br = locate([](Real t) { return std::sin(t); }, 3.0, 3.5, 1e-12, &n);
        CHECK(br.ta <= std::numbers::pi);
        CHECK(br.tb >= std::numbers::pi);
        CHECK(br.tb - br.ta < 1e-12);
        CHECK(br.sa_raw > 0);
        CHECK(br.sb_raw < 0);
        CHECK(n < 60);   // superlinear: far fewer than bisection's ~39
    }
    SECTION("a convex exponential: the Illinois weight halves the stuck end") {
        std::vector<BracketEnd> ends;
        int n = 0;
        // e^(5t) - 2 crosses at ln(2)/5; plain regula falsi keeps moving ta.
        const Real root = std::log(2.0) / 5.0;
        const auto br = locate([](Real t) { return std::exp(5 * t) - 2; }, 0.0, 1.0,
                               1e-10, &n, &ends);
        CHECK(br.ta <= root);
        CHECK(br.tb >= root);
        CHECK(br.tb - br.ta < 1e-10);
        // both ends moved: the weight broke the one-sided run
        bool a = false, b = false;
        for (auto e : ends) (e == BracketEnd::A ? a : b) = true;
        CHECK(a);
        CHECK(b);
        CHECK(n < 40);
    }
}

TEST_CASE("illinois stops on skip, on a failed trial and on done", "[event_locate]") {
    auto g = [](Real t) { return t - 0.5; };
    SECTION("skip before any trial leaves the bracket as given") {
        Bracket br{0.0, -0.5, 1.0, 0.5, -0.5, 0.5, 0};
        const int n = illinois(
            br, 40, [](Real, const Bracket&) { return true; },
            [&](Real t, Real& s) { s = g(t); return true; }, [](BracketEnd) {},
            [](const Bracket&) { return false; });
        CHECK(n == 0);
        CHECK(br.ta == 0.0);
        CHECK(br.tb == 1.0);
    }
    SECTION("a failed trial ends with the bracket as is, nothing moved") {
        Bracket br{0.0, -0.5, 1.0, 0.25, -0.5, 0.25, 0};
        int moved = 0;
        const int n = illinois(
            br, 40, [](Real, const Bracket&) { return false; },
            [](Real, Real&) { return false; }, [&](BracketEnd) { ++moved; },
            [](const Bracket&) { return false; });
        CHECK(n == 1);
        CHECK(moved == 0);
        CHECK(br.ta == 0.0);
        CHECK(br.tb == 1.0);
    }
    SECTION("the trial point stays in the bracket's inner 90 %") {
        // a signal that makes the secant root fall at the end: clamped
        Bracket br{0.0, -1e-12, 1.0, 1.0, -1e-12, 1.0, 0};
        Real seen = -1;
        (void)illinois(
            br, 1, [](Real, const Bracket&) { return false; },
            [&](Real t, Real& s) { seen = t; s = 1.0; return true; },
            [](BracketEnd) {}, [](const Bracket&) { return false; });
        CHECK(seen == Catch::Approx(0.05));
    }
}

TEST_CASE("settle reports rounds, the fixed point, a cycle and a stop", "[event_locate]") {
    SECTION("settles on the third solve") {
        const auto r = settle(10, [](Size i) { return i < 2 ? Round::Flipped : Round::Settled; });
        CHECK(r.rounds == 3);
        CHECK(r.settled);
        CHECK_FALSE(r.cycle);
        CHECK_FALSE(r.stopped);
    }
    SECTION("the round budget ends a flipping settle unsettled") {
        const auto r = settle(4, [](Size) { return Round::Flipped; });
        CHECK(r.rounds == 4);
        CHECK_FALSE(r.settled);
    }
    SECTION("a cycle and a stop are not counted as solved rounds") {
        const auto c = settle(10, [](Size i) { return i == 2 ? Round::Cycle : Round::Flipped; });
        CHECK(c.rounds == 2);
        CHECK(c.cycle);
        const auto s = settle(10, [](Size) { return Round::Stop; });
        CHECK(s.rounds == 0);
        CHECK(s.stopped);
    }
    SECTION("zero rounds: nothing runs") {
        int calls = 0;
        const auto r = settle(0, [&](Size) { ++calls; return Round::Settled; });
        CHECK(calls == 0);
        CHECK(r.rounds == 0);
    }
}

TEST_CASE("MaskCycle sees only the masks of its own call", "[event_locate]") {
    using pulsim::topology::SwitchStateMask;
    MaskCycle<4> mc;
    SwitchStateMask a(3), b(3);
    b.set(1, true);
    CHECK_FALSE(mc.repeat(a));
    CHECK_FALSE(mc.repeat(b));
    CHECK(mc.repeat(a));   // A -> B -> A: a cycle
    CHECK(mc.seen(b));
    // a new call (the next restart stage) does not inherit the cycle
    mc.reset();
    CHECK_FALSE(mc.seen(a));
    CHECK_FALSE(mc.repeat(a));
}
