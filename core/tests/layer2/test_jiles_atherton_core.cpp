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


TEST_CASE("JA core: the Langevin function is smooth across the series/direct "
          "switch and its derivative is consistent", "[v2][c4][ja][unit]") {
    using models::ja_detail::langevin;
    using models::ja_detail::langevin_deriv;
    // A long-double coth x − 1/x is no reference below x ~ 0.1 (it
    // cancels too); what CAN be pinned is continuity at the switch,
    // the small-x limit, and derivative consistency.
    // Straddle the switch by one ulp-ish: the function itself moves
    // L'·2e-15 ≈ 7e-16 across the interval, so any mismatch above
    // 1e-12 is the two formulas disagreeing, not the function moving.
    CHECK(langevin(0.3 - 1e-15) == Approx(langevin(0.3 + 1e-15)).epsilon(1e-12));
    CHECK(langevin_deriv(0.3 - 1e-15) == Approx(langevin_deriv(0.3 + 1e-15)).epsilon(1e-12));
    CHECK(langevin(1e-6) / 1e-6 == Approx(1.0 / 3.0).epsilon(1e-12));
    CHECK(langevin_deriv(0.0) == Approx(1.0 / 3.0).epsilon(1e-15));
    // Odd / even.
    for (const double x : {1e-3, 0.05, 0.29, 0.31, 2.0}) {
        CHECK(langevin(-x) == Approx(-langevin(x)).epsilon(1e-15));
        CHECK(langevin_deriv(-x) == Approx(langevin_deriv(x)).epsilon(1e-15));
    }
    // Derivative consistency by central difference, where the
    // difference itself is accurate (x not too small).
    for (const double x : {0.01, 0.05, 0.1, 0.2, 0.29, 0.31, 0.5, 1.0, 3.0, 10.0}) {
        const double d = 1e-5 * x;
        const double fd = (langevin(x + d) - langevin(x - d)) / (2 * d);
        INFO("x = " << x);
        CHECK(langevin_deriv(x) == Approx(fd).epsilon(1e-8));
    }
    // Against a high-precision series in long double, well inside
    // the series region (the reference's own truncation is ~1e-20).
    for (const double x : {1e-4, 1e-3, 1e-2, 0.1}) {
        const long double xl = x, x2 = xl * xl;
        const long double ref = xl * (1.0L / 3.0L + x2 * (-1.0L / 45.0L + x2 * (2.0L / 945.0L
            + x2 * (-1.0L / 4725.0L + x2 * (2.0L / 93555.0L + x2 * (-1382.0L / 638512875.0L
            + x2 * (4.0L / 18243225.0L + x2 * (-3617.0L / 162820783125.0L))))))));
        CHECK(langevin(x) == Approx(static_cast<double>(ref)).epsilon(1e-14));
    }
    // And against the direct form where IT is accurate.
    for (const double x : {0.5, 1.0, 2.0, 5.0}) {
        const long double xl = x;
        CHECK(langevin(x) == Approx(static_cast<double>(1.0L / std::tanh(xl) - 1.0L / xl)).epsilon(1e-14));
    }
}

TEST_CASE("JA core: λ(i) is monotone at picoampere spacing — what a 1e-9 V "
          "Newton tolerance at a 50 ns step actually needs",
          "[v2][c4][ja][unit]") {
    // The PSFB core: 12 turns, 3.5 cm², 100 mm, 1.5 mm gap, N87.
    JA::Params c;
    c.N = 12; c.Ae = 3.5e-4; c.le = 0.1; c.lg = 1.5e-3;
    c.ja = models::JilesAthertonParams{.Ms = 4.0e5, .a = 50.0, .alpha = 5e-5, .c = 0.20, .k = 30.0};
    JA::State s{0.0, 0.0, 8, 1.0};
    for (const Real i0 : {0.02, 0.3, 2.0}) {
        Real prev = JA::evaluate(c, s, i0).lambda;
        const Real L = JA::evaluate(c, s, i0).L;
        for (int k = 1; k <= 40; ++k) {
            const Real i = i0 + k * 1e-12;
            const Real lam = JA::evaluate(c, s, i).lambda;
            const Real slope = (lam - prev) / 1e-12;
            INFO("i0 = " << i0 << " k = " << k << " slope = " << slope << " L = " << L);
            CHECK(lam > prev);
            CHECK(slope == Approx(L).epsilon(5e-3));
            prev = lam;
        }
    }
}
