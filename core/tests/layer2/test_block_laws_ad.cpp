// =============================================================================
// The control graph's block laws on the AD scalar (the tick Jacobian):
// every law of block_laws.h against central differences of the same law
// on doubles, against closed forms (PI, integrator, low-pass), smooth at
// tiny inputs, and reporting the side of a kink — refused ON it.
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/blockchain/chain.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

using namespace pulsim;
using pulsim::blockchain::AdReal;

namespace {

// Records every comparison a law takes (the PL_LT ... hooks).
struct Notes final : pl_kink_sink<AdReal> {
    struct N { std::string law; Real a, b; bool side; };
    std::vector<N> notes;
    pl_kink_sink<AdReal>* old;
    Notes() : old(pl_kink_sink<AdReal>::current()) { pl_kink_sink<AdReal>::current() = this; }
    ~Notes() override { pl_kink_sink<AdReal>::current() = old; }
    Notes(const Notes&) = delete;
    Notes& operator=(const Notes&) = delete;
    void note(const AdReal& a, const AdReal& b, bool side, const char* law) override {
        notes.push_back({law, a.value(), b.value(), side});
    }
    // smallest |a - b| / max(|a|, |b|, 1) over the comparisons
    [[nodiscard]] Real closest() const {
        Real m = 1e300;
        for (const auto& n : notes) {
            m = std::min(m, std::abs(n.a - n.b) / std::max({std::abs(n.a), std::abs(n.b), Real{1}}));
        }
        return m;
    }
};

// f(in, out) on any scalar; checks d out / d in by AD against a central
// difference of the double law, and the AD values against the double law
// bit for bit. Returns the AD Jacobian (out x in).
template <class F>
std::vector<std::vector<Real>> check(const char* name, F f, const std::vector<Real>& x0,
                                     std::size_t nout, Real tol = 1e-9) {
    INFO(name);
    const std::size_t n = x0.size();
    std::vector<Real> yd(nout);
    {
        std::vector<Real> xin = x0;
        f(xin.data(), yd.data());
    }
    std::vector<std::vector<Real>> J(nout, std::vector<Real>(n));
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<AdReal> xa(n), ya(nout);
        for (std::size_t i = 0; i < n; ++i) xa[i] = i == j ? AdReal{x0[i], {Real{1}}} : AdReal{x0[i]};
        Notes notes;
        f(xa.data(), ya.data());
        // the points are chosen off every kink: a central difference is valid
        REQUIRE(notes.closest() > 1e-6);
        // a time step or time constant (|x| < 1e-2) needs a step relative to itself
        const Real ax = std::abs(x0[j]);
        const Real h = ax > 0 && ax < 1e-2 ? 1e-5 * ax : 1e-6 * std::max(Real{1}, ax);
        std::vector<Real> xp = x0, xm = x0, yp(nout), ym(nout);
        xp[j] += h;
        xm[j] -= h;
        f(xp.data(), yp.data());
        f(xm.data(), ym.data());
        for (std::size_t k = 0; k < nout; ++k) {
            INFO("output " << k << " input " << j);
            CHECK(ya[k].value() == yd[k]);
            const Real fd = (yp[k] - ym[k]) / (2 * h);
            const Real scale = std::max({Real{1}, std::abs(yd[k]), std::abs(fd)});
            CHECK(std::abs(ya[k].deriv(0) - fd) <= tol * scale);
            J[k][j] = ya[k].deriv(0);
        }
    }
    return J;
}

}  // namespace

TEST_CASE("Every block law's AD derivative equals a central difference of the law",
          "[blockchain][ad][tick_jacobian]") {
    check("gain", [](auto* x, auto* y) { y[0] = pl_gain(x[0], x[1]); }, {2.5, -0.7}, 1);
    check("subtract", [](auto* x, auto* y) { y[0] = pl_subtract(x[0], x[1]); }, {2.5, -0.7}, 1);
    check("sum", [](auto* x, auto* y) { y[0] = pl_sum(x, x + 2, 2); }, {0.5, -2.0, 1.25, 3.0}, 1);
    // PI: unsaturated, then pushing past hi (anti-windup holds), then clamped below
    for (const std::vector<Real>& p : {std::vector<Real>{0.05, 200.0, 0.0, 1.0, 0.3, 0.2, 5.0, 4.0, 1e-5},
                                       std::vector<Real>{0.05, 200.0, 0.0, 1.0, 1.2, 0.2, 5.0, 4.0, 1e-5},
                                       std::vector<Real>{0.05, 200.0, 0.1, 1.0, 0.02, -0.2, 3.0, 4.0, 1e-5}}) {
        check("pi", [](auto* x, auto* y) {
            auto integ = x[4];
            auto prev = x[5];
            y[0] = pl_pi(x[0], x[1], x[2], x[3], &integ, &prev, x[6], x[7], x[8]);
            y[1] = integ;
            y[2] = prev;
        }, p, 3);
    }
    check("pid", [](auto* x, auto* y) {
        auto ig = x[6];
        auto dv = x[7];
        auto pe = x[8];
        y[0] = pl_pid(x[0], x[1], x[2], x[3], x[4], x[5], &ig, &dv, &pe, x[9], x[10], x[11]);
        y[1] = ig;
        y[2] = dv;
        y[3] = pe;
    }, {0.1, 50.0, 1e-4, 0.3, -5.0, 5.0, 0.2, 0.01, 0.4, 5.0, 4.5, 1e-4}, 4);
    check("integrator", [](auto* x, auto* y) {
        auto s = x[3];
        y[0] = pl_integrator(x[0], x[1], x[2], &s, x[4], x[5]);
    }, {3.0, -1.0, 1.0, 0.2, 0.7, 1e-3}, 1);
    check("differentiator", [](auto* x, auto* y) {
        auto s = x[1];
        auto px = x[2];
        y[0] = pl_differentiator(x[0], &s, &px, x[3], x[4]);
        y[1] = px;
    }, {0.2, 0.5, 1.0, 1.3, 1e-3}, 2);
    check("ema", [](auto* x, auto* y) { auto s = x[1]; y[0] = pl_ema(x[0], &s, x[2]); },
          {0.9, 0.5, 2.0}, 1);
    check("low_pass", [](auto* x, auto* y) {
        auto s = x[1];
        y[0] = pl_low_pass(x[0], &s, x[2], x[3]);
    }, {1e-3, 0.5, 2.0, 1e-4}, 1);
    check("limiter", [](auto* x, auto* y) { y[0] = pl_limiter(x[0], x[1], x[2]); },
          {-1.0, 1.0, 0.3}, 1);
    check("limiter clamped", [](auto* x, auto* y) { y[0] = pl_limiter(x[0], x[1], x[2]); },
          {-1.0, 1.0, 1.7}, 1);
    check("comparator", [](auto* x, auto* y) {
        auto s = x[4];
        y[0] = pl_comparator(x[0], x[1], x[2], x[3], &s, x[5]);
    }, {0.5, 0.1, 3.0, -1.0, 0.0, 0.9}, 1);
    check("rate_limiter", [](auto* x, auto* y) {
        auto s = x[2];
        y[0] = pl_rate_limiter(x[0], x[1], &s, x[3], x[4]);
    }, {100.0, 50.0, 0.2, 0.25, 1e-4}, 1);
    check("rate_limiter limited", [](auto* x, auto* y) {
        auto s = x[2];
        y[0] = pl_rate_limiter(x[0], x[1], &s, x[3], x[4]);
    }, {100.0, 50.0, 0.2, 0.9, 1e-4}, 1);
    check("sample_hold", [](auto* x, auto* y) {
        auto s = x[1];
        auto lt = x[2];
        y[0] = pl_sample_hold(x[0], &s, &lt, x[3], x[4]);
    }, {1e-3, 0.4, 0.0, 0.9, 2.5e-3}, 1);
    check("transfer_function", [](auto* x, auto* y) {
        using S = std::remove_pointer_t<decltype(x)>;
        S xh[2] = {x[3], x[4]};
        S yh[1] = {x[5]};
        int nx = 2, ny = 1;
        y[0] = pl_transfer_function(x, 2, x + 2, 1, xh, &nx, yh, &ny, x[6]);
    }, {0.3, 0.2, -0.5, 0.1, 0.4, 0.25, 1.5}, 1);
    check("state_machine level", [](auto* x, auto* y) {
        auto s = x[3];
        auto tp = x[4];
        y[0] = pl_state_machine(1, x[0], x[1], x[2], &s, &tp, x[5], x[6], x[7]);
    }, {0.5, 2.0, -1.0, 0.0, 0.0, 0.8, 0.0, 0.0}, 1);
    check("op_amp", [](auto* x, auto* y) { y[0] = pl_op_amp(x[0], x[1], x[2], x[3], x[4]); },
          {1e3, -12.0, 12.0, 1.002, 1.0}, 1);
    check("lookup_1d", [](auto* x, auto* y) { y[0] = pl_lookup_1d(x, x + 3, 3, x[6]); },
          {0.0, 1.0, 2.0, 0.5, 1.5, 4.0, 1.3}, 1);
    for (int op = 0; op < 8; ++op) {
        check("math", [op](auto* x, auto* y) { y[0] = pl_math(op, x[0], x[1]); }, {1.7, -0.6}, 1);
    }
    check("svm", [](auto* x, auto* y) { pl_svm(x[0], x[1], x[2], &y[0], &y[1], &y[2]); },
          {400.0, 120.0, -45.0}, 3);
    check("clarke", [](auto* x, auto* y) { pl_clarke(x[0], x[1], x[2], &y[0], &y[1], &y[2]); },
          {1.0, -0.4, -0.6}, 3);
    check("inverse_clarke",
          [](auto* x, auto* y) { pl_inverse_clarke(x[0], x[1], x[2], &y[0], &y[1], &y[2]); },
          {1.0, -0.4, 0.1}, 3);
    check("park", [](auto* x, auto* y) { pl_park(x[0], x[1], x[2], &y[0], &y[1]); },
          {1.0, -0.4, 0.7}, 2);
    check("inverse_park", [](auto* x, auto* y) { pl_inverse_park(x[0], x[1], x[2], &y[0], &y[1]); },
          {1.0, -0.4, 0.7}, 2);
    check("pll", [](auto* x, auto* y) {
        auto th = x[3];
        auto om = x[4];
        auto ig = x[5];
        pl_pll(x[0], x[1], x[2], &th, &om, &ig, x[6], x[7], x[8]);
        y[0] = th;
        y[1] = om;
        y[2] = ig;
    // tol 1e-7: omega = 314 rad/s over a 1e-9 s step rounds to ~1.6e-5 in the difference
    }, {50.0, 100.0, 1000.0, 1.0, 314.0, 0.5, 0.3, 0.8, 1e-4}, 3, 1e-7);
}

TEST_CASE("PI tick derivatives equal their closed forms",
          "[blockchain][ad][tick_jacobian]") {
    // u = Kp e + x_i + Ki dt (e + e_prev)/2, x_i+ = x_i + Ki dt (e + e_prev)/2
    const Real Kp = 0.05, Ki = 200, dt = 1e-5, x_i = 0.3, e_prev = 0.2, sp = 5.0, m = 4.0;
    const Real e = sp - m;
    const auto J = check("pi closed form", [&](auto* x, auto* y) {
        auto integ = x[2];
        auto prev = x[3];
        y[0] = pl_pi(x[0], x[1], std::remove_reference_t<decltype(x[0])>(0.0), std::remove_reference_t<decltype(x[0])>(1.0), &integ, &prev, x[4], x[5],
                     std::remove_reference_t<decltype(x[0])>(dt));
        y[1] = integ;
    }, {Kp, Ki, x_i, e_prev, sp, m}, 2);
    const Real tr = dt * 0.5 * (e + e_prev);
    CHECK(J[0][0] == e);                                  // du/dKp = e
    CHECK(std::abs(J[0][1] - tr) <= 1e-15);               // du/dKi = dt (e + e_prev)/2
    CHECK(std::abs(J[1][1] - tr) <= 1e-15);               // dx_i+/dKi
    CHECK(J[1][2] == 1.0);                                // dx_i+/dx_i
    CHECK(std::abs(J[0][4] - (Kp + Ki * dt * 0.5)) <= 1e-15);   // du/dsetpoint
    CHECK(std::abs(J[0][5] + (Kp + Ki * dt * 0.5)) <= 1e-15);   // du/dmeasured
    CHECK(std::abs(J[0][3] - Ki * dt * 0.5) <= 1e-15);          // du/de_prev
}

TEST_CASE("A saturated PI reports its side; ON the kink it is seen",
          "[blockchain][ad][tick_jacobian]") {
    // clamped at hi with e > 0: the integrator holds, du/dKp = 0, du/dhi = 1
    {
        Notes n;
        AdReal integ{0.98}, prev{0.2};
        const AdReal kp{0.05, {1.0}};
        const AdReal u = pl_pi(kp, AdReal{200.0}, AdReal{0.0}, AdReal{1.0}, &integ, &prev,
                               AdReal{5.0}, AdReal{4.0}, AdReal{1e-5});
        CHECK(u.value() == 1.0);
        CHECK(u.deriv(0) == 0.0);
        CHECK(integ.deriv(0) == 0.0);
        bool saw = false;
        for (const auto& k : n.notes) saw = saw || (k.law == "pl_pi" && k.side && k.b == 1.0);
        CHECK(saw);   // the anti-windup branch u > hi was taken, and said so
    }
    // u exactly at hi: a comparison with a - b == 0 (the tick sits ON the kink)
    {
        Notes n;
        // e = 1, Kp e + x_i + Ki dt (e + e_prev)/2 = 0.05 + x_i + 0.0012 = 1
        AdReal integ{1.0 - 0.05 - 200.0 * 1e-5 * 0.5 * 1.2}, prev{0.2};
        (void)pl_pi(AdReal{0.05, {1.0}}, AdReal{200.0}, AdReal{0.0}, AdReal{1.0}, &integ, &prev,
                    AdReal{5.0}, AdReal{4.0}, AdReal{1e-5});
        CHECK(n.closest() <= 1e-15);
    }
}

TEST_CASE("Block laws are smooth to tiny inputs where they claim to be",
          "[blockchain][ad][tick_jacobian]") {
    // linear laws: the derivative at 1e-13 is the derivative at 1
    for (const Real s : {1e-13, 1.0}) {
        AdReal integ{0.1 * s}, prev{0.0};
        const AdReal m{4.0 * s, {1.0}};
        const AdReal u = pl_pi(AdReal{0.05}, AdReal{200.0}, AdReal{-1.0}, AdReal{1.0}, &integ, &prev,
                               AdReal{5.0 * s}, m, AdReal{1e-5});
        CHECK(std::abs(u.deriv(0) + (0.05 + 200.0 * 1e-5 * 0.5)) <= 1e-15);
        AdReal st{0.2 * s};
        const AdReal lp = pl_low_pass(AdReal{1e-3}, &st, AdReal{s, {1.0}}, AdReal{1e-4});
        CHECK(std::abs(lp.deriv(0) - 1e-4 / (1e-3 + 1e-4)) <= 1e-15);
    }
    // |a| and sqrt(a) are not smooth at 0: at a tiny a the law reports the kink
    for (int op : {4, 6}) {
        Notes n;
        (void)pl_math(op, AdReal{1e-13, {1.0}}, AdReal{0.0});
        CHECK(n.closest() <= 1e-12);
    }
}
