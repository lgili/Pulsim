// =============================================================================
// Phase 4 C.4 — the in-loop Jiles-Atherton core evaluator
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/models/jiles_atherton.hpp"

#include <cmath>
#include <vector>

using namespace pulsim;
using Catch::Approx;
using JA = models::JilesAthertonCore;

namespace {
JA::Params ferrite(Real lg = 0.0) {
    JA::Params c;
    c.N = 100; c.Ae = 1e-4; c.le = 0.1; c.lg = lg;
    c.ja = models::JilesAthertonParams{.Ms = 4.0e5, .a = 50.0, .alpha = 5e-5, .c = 0.20, .k = 30.0};
    return c;
}
}  // namespace

TEST_CASE("JA core: i(H) is strictly increasing on both branches, with and "
          "without a gap", "[v2][c4][ja][unit]") {
    for (const Real lg : {0.0, 0.5e-3}) {
        const auto c = ferrite(lg);
        // Walk the state around a major loop first so (H_n, M_n) sits on
        // a real branch, then probe monotonicity from several states.
        JA::State s{0.0, 0.0, JA::substeps_for(c, 2 * 20 * c.ja.a * 2 * M_PI / 100.0)};
        std::vector<JA::State> states;
        for (int k = 1; k <= 400; ++k) {
            const Real H = 20 * c.ja.a * std::sin(2 * M_PI * k / 100.0);
            s.delta_hint = (H >= s.H) ? 1.0 : -1.0;   // the walk's own direction
            s.M = JA::integrate_M(c, s, H);
            s.H = H;
            if (k % 40 == 0) states.push_back(s);
        }
        for (auto st : states) {
            // The probe spans 30a; a step that large gets its count.
            st.n_sub = JA::substeps_for(c, 30 * c.ja.a);
            for (const Real dir : {1.0, -1.0}) {
                // The direction is a per-step decision (State::delta_hint),
                // set here to the way the probe walks.
                st.delta_hint = dir;
                Real prev_i = JA::current_of(c, st.H, st.M);
                for (int j = 1; j <= 200; ++j) {
                    const Real H = st.H + dir * 30 * c.ja.a * j / 200.0;
                    const Real M = JA::integrate_M(c, st, H);
                    const Real i = JA::current_of(c, H, M);
                    INFO("lg=" << lg << " H_n=" << st.H << " dir=" << dir << " H=" << H);
                    CHECK((dir > 0 ? i > prev_i : i < prev_i));
                    prev_i = i;
                    CHECK(std::abs(M) <= c.ja.Ms * (1 + 1e-12));
                }
            }
        }
    }
}

TEST_CASE("JA core: the inversion round-trips and L is the exact derivative",
          "[v2][c4][ja][unit]") {
    for (const Real lg : {0.0, 0.5e-3}) {
        const auto c = ferrite(lg);
        JA::State s{0.0, 0.0, JA::substeps_for(c, 2 * 10 * c.ja.a * 2 * M_PI / 100.0)};
        for (int k = 1; k <= 130; ++k) {
            const Real H = 10 * c.ja.a * std::sin(2 * M_PI * k / 100.0);
            s.delta_hint = (H >= s.H) ? 1.0 : -1.0;
            s.M = JA::integrate_M(c, s, H);
            s.H = H;
        }
        // From this state, pick targets on both branches. Δi = 3 A on
        // N = 100 / le = 0.1 is ΔH = 3000 A/m = 60a: size the count.
        s.n_sub = JA::substeps_for(c, 60 * c.ja.a);
        const Real i_n = JA::current_of(c, s.H, s.M);
        for (const Real di : {-3.0, -0.7, -0.05, 0.05, 0.7, 3.0}) {
            s.delta_hint = di >= 0 ? 1.0 : -1.0;
            const Real i = i_n + di;
            const auto e = JA::evaluate(c, s, i);
            // Round trip: the current at the found H is the target.
            CHECK(JA::current_of(c, e.H, e.M) == Approx(i).epsilon(1e-10));
            // L by central difference of λ(i) on the same branch.
            const Real d = 1e-4 * std::abs(di) + 1e-6;
            const auto ep = JA::evaluate(c, s, i + d);
            const auto em = JA::evaluate(c, s, i - d);
            const Real L_fd = (ep.lambda - em.lambda) / (2 * d);
            INFO("lg=" << lg << " di=" << di << " L=" << e.L << " L_fd=" << L_fd);
            CHECK(e.L == Approx(L_fd).epsilon(2e-3));
            CHECK(e.L > 0.0);
        }
    }
}

TEST_CASE("JA core: the anhysteretic limit is a single-valued curve",
          "[v2][c4][ja][unit]") {
    // c = 1: no irreversible part, the loop collapses to M_an(H) and the
    // path up equals the path down.
    auto c = ferrite(0.0);
    c.ja.c = 1.0;
    JA::State s{0.0, 0.0, JA::substeps_for(c, 0.1 * c.ja.a)};
    std::vector<Real> up, down;
    for (int k = 0; k <= 100; ++k) {
        const Real H = 10 * c.ja.a * k / 100.0;
        s.delta_hint = 1.0;
        s.M = JA::integrate_M(c, s, H); s.H = H; up.push_back(s.M);
    }
    for (int k = 100; k >= 0; --k) {
        const Real H = 10 * c.ja.a * k / 100.0;
        s.delta_hint = -1.0;
        s.M = JA::integrate_M(c, s, H); s.H = H; down.push_back(s.M);
    }
    for (int k = 0; k <= 100; ++k) {
        CHECK(down[100 - k] == Approx(up[k]).margin(1e-6 * c.ja.Ms));
    }
}

TEST_CASE("JA core: refuses non-physical parameters by name",
          "[v2][c4][ja][unit]") {
    using Catch::Matchers::ContainsSubstring;
    auto c = ferrite(0.0);
    c.ja.alpha = 4e-4;    // alpha·Ms = 160 >= 3a = 150
    CHECK_THROWS_WITH(JA::validate(c, "L1"), ContainsSubstring("3a"));
    c = ferrite(0.0); c.ja.c = 1.5;
    CHECK_THROWS_WITH(JA::validate(c, "L1"), ContainsSubstring("reversibility"));
    c = ferrite(0.0); c.Ae = 100.0;
    CHECK_THROWS_WITH(JA::validate(c, "L1"), ContainsSubstring("mm²"));
}
