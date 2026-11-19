// add-high-order-stiff-integrator: ESDIRK4(3)6L[2]SA on the unified step
// skeleton, on MNA-shaped problems with closed-form or mpmath truths
// (openspec/changes/add-high-order-stiff-integrator/probes/).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <vector>

#include "mna_stage_body.hpp"

using pulsim::Real;
using pulsim::Size;
namespace step = pulsim::step;
using mna::Circuit;
using mna::Mat;
using mna::Vec;
using E = step::Esdirk43;

namespace {

constexpr Real kPi = std::numbers::pi;

// ---- circuits -------------------------------------------------------------

// Ideal LC tank, x = [v, iL]: C v' + iL = 0, L iL' − v = 0.
constexpr Real kL = 100e-6, kC = 100e-9, kV0 = 10;
Circuit lc() {
    Circuit c;
    c.C = Mat{{kC, 0}, {0, kL}};
    c.G = Mat{{0, 1}, {-1, 0}};
    c.b = [](Real) { return Vec(Vec::Zero(2)); };
    c.norm_vars = {0, 1};
    return c;
}
Real lc_w() { return 1 / std::sqrt(kL * kC); }
Real lc_T() { return 2 * kPi / lc_w(); }
// error of the complex amplitude v + j·Z0·iL, relative to V0
Real lc_err(const Vec& x, Real t) {
    const Real z0 = std::sqrt(kL / kC), w = lc_w();
    const std::complex<Real> num(x[0], z0 * x[1]), ex(kV0 * std::cos(w * t), kV0 * std::sin(w * t));
    return std::abs(num - ex) / kV0;
}

// Index-1 RC network, x = [v1, v2, v3, iV]: Vs = sin(3t) at n1, R1 n1–n2,
// R3 n2–gnd, R2 n2–n3, C n3–gnd. v3 differential; v2 (capacitor-free node)
// and iV (source current) algebraic. Closed form (probe item 2).
constexpr Real kR1 = 1, kR2 = 2, kR3 = 3, kCn = 1, kWn = 3, kV30 = 0.5;
Circuit rc_net() {
    Circuit c;
    c.C = Mat::Zero(4, 4);
    c.C(2, 2) = kCn;
    c.G = Mat{{1 / kR1, -1 / kR1, 0, 1},
              {-1 / kR1, 1 / kR1 + 1 / kR2 + 1 / kR3, -1 / kR2, 0},
              {0, -1 / kR2, 1 / kR2, 0},
              {1, 0, 0, 0}};
    c.b = [](Real t) { return Vec(Vec{{0, 0, 0, std::sin(kWn * t)}}); };
    c.norm_vars = {2};
    return c;
}
Vec rc_exact(Real t) {
    const Real S = 1 / kR1 + 1 / kR2 + 1 / kR3;
    const Real a = (1 - 1 / (kR2 * S)) / (kR2 * kCn), beta = 1 / (kR1 * S * kR2 * kCn);
    const auto vp = [&](Real u) {
        return beta * (a * std::sin(kWn * u) - kWn * std::cos(kWn * u)) / (a * a + kWn * kWn);
    };
    const Real v3 = vp(t) + (kV30 - vp(0)) * std::exp(-a * t);
    const Real v1 = std::sin(kWn * t);
    const Real v2 = (v1 / kR1 + v3 / kR2) / S;
    return Vec{{v1, v2, v3, -(v1 - v2) / kR1}};
}

// Index 2, capacitor across the source, x = [v1, iV]: C v1' + v1/R + iV = 0,
// v1 = sin(3t); iV = −3C cos(3t) − sin(3t)/R needs the source's derivative.
Circuit cap_loop() {
    Circuit c;
    c.C = Mat{{1, 0}, {0, 0}};
    c.G = Mat{{1, 1}, {1, 0}};
    c.b = [](Real t) { return Vec(Vec{{0, std::sin(3 * t)}}); };
    c.norm_vars = {0};
    return c;
}
Vec cap_loop_exact(Real t) {
    return Vec{{std::sin(3 * t), -3 * std::cos(3 * t) - std::sin(3 * t)}};
}
// Index 2, inductor in a current-source cutset, x = [v1, iL]: iL = sin(3t),
// L iL' − v1 = 0, so v1 = 3L cos(3t) (L = 1).
Circuit ind_cutset() {
    Circuit c;
    c.C = Mat{{0, 0}, {0, 1}};
    c.G = Mat{{0, 1}, {-1, 0}};
    c.b = [](Real t) { return Vec(Vec{{std::sin(3 * t), 0}}); };
    c.norm_vars = {1};
    return c;
}
Vec ind_cutset_exact(Real t) { return Vec{{3 * std::cos(3 * t), std::sin(3 * t)}}; }
// The same cutset whose index-2 voltage FEEDS a differential state, x =
// [v1, iL, vo]: a VCCS gm·v1 (gm = 1) into vo with 1 F ∥ 1 Ω:
// vo' = −vo + 3 cos(3t), vo(0) = 0.
Circuit cutset_fb() {
    Circuit c;
    c.C = Mat{{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    c.G = Mat{{0, 1, 0}, {-1, 0, 0}, {-1, 0, 1}};
    c.b = [](Real t) { return Vec(Vec{{std::sin(3 * t), 0, 0}}); };
    c.norm_vars = {1, 2};
    return c;
}
Vec cutset_fb_exact(Real t) {
    const Real vo = 0.3 * (std::cos(3 * t) + 3 * std::sin(3 * t)) - 0.3 * std::exp(-t);
    return Vec{{3 * std::cos(3 * t), std::sin(3 * t), vo}};
}

// Prothero–Robinson y' = λ(y − sin t) + cos t, exact sin t.
Circuit prothero(Real lambda) {
    Circuit c;
    c.C = Mat{{1}};
    c.G = Mat{{-lambda}};
    c.b = [lambda](Real t) { return Vec(Vec{{-lambda * std::sin(t) + std::cos(t)}}); };
    c.norm_vars = {0};
    return c;
}

// Stiff ladder (probe item 1), x = [vs, va, vb, vc, iL, iV]; time constants
// 1e-14 s (snubber), 1e-13 s (source node) against an LC of 6.3e-5 s.
constexpr Real kRs = 0.1, kCa = 1e-12, kLl = 1e-4, kCl = 1e-7, kRL = 50, kRsn = 1,
               kCsn = 1e-14;
Circuit ladder() {
    Circuit c;
    c.C = Mat::Zero(6, 6);
    c.C(1, 1) = kCa;
    c.C(2, 2) = kCl;
    c.C(3, 3) = kCsn;
    c.C(4, 4) = kLl;
    c.G = Mat::Zero(6, 6);
    c.G(0, 0) = 1 / kRs; c.G(0, 1) = -1 / kRs; c.G(0, 5) = 1;
    c.G(1, 0) = -1 / kRs; c.G(1, 1) = 1 / kRs; c.G(1, 4) = 1;
    c.G(2, 4) = -1; c.G(2, 2) = 1 / kRsn + 1 / kRL; c.G(2, 3) = -1 / kRsn;
    c.G(3, 2) = -1 / kRsn; c.G(3, 3) = 1 / kRsn;
    c.G(4, 1) = -1; c.G(4, 2) = 1;
    c.G(5, 0) = 1;
    c.b = [](Real) { return Vec(Vec{{0, 0, 0, 0, 0, 10}}); };
    c.norm_vars = {1, 2, 3, 4};
    return c;
}
struct LadderRef { Real t, va, vb, vc, iL; };
// mpmath expm, 40 digits (probes/esdirk43_truth.py)
constexpr LadderRef kLadderRef[] = {
    {1e-7, 9.9990002178171020375, 0.0049662431254719565007, 0.0049662421356214573959, 0.0099978318230136107104},
    {1e-6, 9.9901627981547849554, 0.46427498200318663644, 0.46427497309453477095, 0.098372027978038352129},
    {1e-5, 9.9714453872622045106, 13.426878585778597092, 13.426878584077742004, 0.28554612392252171192},
    {2e-5, 9.9835959608238826289, 8.8185231717135420659, 8.8185231729465490075, 0.16404039292624648788},
    {5e-5, 9.9798268660289116672, 10.016076134977213825, 10.016076134836232141, 0.20173133967463406044}};
Real ladder_err(const Vec& x, const LadderRef& r) {
    return std::max({std::abs(x[1] - r.va), std::abs(x[2] - r.vb), std::abs(x[3] - r.vc),
                     std::abs(x[4] - r.iL) * 10});   // iL in 0.1 V/A units
}
Vec ladder_start(const Circuit& c) { return c.consistent(Vec::Zero(6), 0); }

// Shockley clamp (probe item 3), x = [v1, vo, iV].
constexpr Real kIs = 1e-12, kNvt = 0.0388;
Circuit diode() {
    Circuit c;
    c.C = Mat::Zero(3, 3);
    c.C(1, 1) = 1e-5;
    c.G = Mat{{0.1, -0.1, 1}, {-0.1, 0.1 + 0.01, 0}, {1, 0, 0}};
    c.b = [](Real t) { return Vec(Vec{{0, 0, 5 * std::sin(2e3 * kPi * t)}}); };
    c.g = [](const Vec& x) { return Vec(Vec{{0, kIs * std::expm1(x[1] / kNvt), 0}}); };
    c.dg = [](const Vec& x) {
        Mat J = Mat::Zero(3, 3);
        J(1, 1) = kIs / kNvt * std::exp(x[1] / kNvt);
        return J;
    };
    c.norm_vars = {1};
    return c;
}
struct DiodeRef { Real t, vo; };
// scipy Radau rtol 1e-13 (DOP853 agrees within 1.3e-13)
constexpr DiodeRef kDiodeRef[] = {{2.5e-4, 1.0351605447068881}, {5e-4, 0.7612168542625108},
                                  {1e-3, -1.9625319548015128}, {1.5e-3, 0.7612168542625151},
                                  {2e-3, -1.9625319548015125}};

// ---- drivers --------------------------------------------------------------

template <class I>
struct Run {
    Vec x;
    Size steps = 0, rejects = 0, factors = 0, solves = 0, newton = 0, stages = 0, max_newton = 0;
    Size newton_fails = 0, refreshes = 0;
    Real err = 0;   // problem-specific, filled by the caller
};

template <class I>
Run<I> fixed(const Circuit& c, Vec x0, Real t0, Real t1, Size n,
             const std::function<void(const mna::Body<I>&, const step::StepFrame&)>& hook = {}) {
    mna::Body<I> b(c, std::move(x0));
    b.hook = hook;
    step::FixedController k{{t0, (t1 - t0) / static_cast<Real>(n)}, 1, n + 1};
    REQUIRE(step::run<I>(k, b) == step::StepStatus::Done);
    return {b.x, n, 0, b.n_factor, b.n_solve, b.n_newton, b.n_stage, b.max_newton, b.n_newton_fail, b.n_refresh, 0};
}

// Adaptive: runs to each sample time in turn (the clock lands on it) and
// calls `at(x, t)` there.
template <class I>
Run<I> adaptive(const Circuit& c, Vec x0, Real rtol, Real atol, Real h0,
                const std::vector<Real>& samples, const std::function<void(const Vec&, Real)>& at,
                bool filter = true) {
    mna::Body<I> b(c, std::move(x0));
    b.rtol = rtol;
    b.atol = atol;
    b.filter = filter;
    step::LteClock<I> k;
    k.h_min = 1e-20;
    k.ctrl.h = h0;
    for (const Real ts : samples) {
        k.t_end = ts;
        REQUIRE(step::run<I>(k, b) == step::StepStatus::Done);
        at(b.x, ts);
    }
    return {b.x, k.n_accept, k.n_reject, b.n_factor, b.n_solve, b.n_newton, b.n_stage, b.max_newton, b.n_newton_fail, b.n_refresh, 0};
}

Real order(Real e_coarse, Real e_fine) { return std::log2(e_coarse / e_fine); }

}  // namespace

// ===========================================================================
// Hidden: prints the measurements recorded in DECISIONS.md.
//   pulsim_core_step_tests "[.esdirk_measure]"
// ===========================================================================
TEST_CASE("ESDIRK4(3): measurements for DECISIONS", "[.esdirk_measure]") {
    // global errors at T on a halving sequence, per variable kind
    const auto global = [](const char* name, const Circuit& c, const Vec& x0, Real T,
                           const std::function<Vec(Real)>& ex, std::vector<int> vars) {
        for (int pass = 0; pass < 2; ++pass) {
            std::printf("%s [%s] global error at T=%g\n", name, pass ? "trbdf2" : "esdirk43", T);
            std::vector<Vec> errs;
            for (Size n : {10, 20, 40, 80, 160, 320}) {
                const Vec x = pass ? fixed<step::TrBdf2>(c, x0, 0, T, n).x : fixed<E>(c, x0, 0, T, n).x;
                errs.push_back((x - ex(T)).cwiseAbs());
            }
            for (int v : vars) {
                std::printf("  var %d:", v);
                for (std::size_t i = 0; i < errs.size(); ++i) {
                    std::printf(" %.2e", errs[i][v]);
                    if (i) std::printf("(%.2f)", order(errs[i - 1][v], errs[i][v]));
                }
                std::printf("\n");
            }
        }
    };
    global("RC network (v1 v2 v3 iV)", rc_net(), rc_exact(0), 2, rc_exact, {1, 2, 3});
    global("cap across source (v1 iV)", cap_loop(), cap_loop_exact(0), 2, cap_loop_exact, {0, 1});
    global("inductor cutset (v1 iL)", ind_cutset(), ind_cutset_exact(0), 2, ind_cutset_exact, {0, 1});
    global("cutset feeding vo (v1 iL vo)", cutset_fb(), cutset_fb_exact(0), 2, cutset_fb_exact, {0, 2});
    for (Real lam : {-1.0, -1e6}) {
        std::printf("Prothero-Robinson lambda=%g\n", lam);
        global("  PR", prothero(lam), Vec{{0.0}}, 1.5, [](Real t) { return Vec(Vec{{std::sin(t)}}); }, {0});
    }

    // local (one step from the exact state) errors: step end and dense output
    const auto local = [](const char* name, const Circuit& c, const std::function<Vec(Real)>& ex,
                          Real t0, std::vector<int> vars) {
        std::printf("%s local errors from t0=%g: step end | dense theta=0.5 | dense theta=0.3\n", name, t0);
        std::vector<std::array<Vec, 5>> errs;
        for (Real h : {0.4, 0.2, 0.1, 0.05, 0.025}) {
            mna::Body<E> b(c, ex(t0));
            step::FixedController k{{t0, h}, 1, 2};
            std::array<Vec, 5> e;
            b.hook = [&](const mna::Body<E>& bb, const step::StepFrame&) {
                e[1] = (bb.dense(0.5) - ex(t0 + 0.5 * h)).cwiseAbs();
                e[2] = (bb.dense(0.3) - ex(t0 + 0.3 * h)).cwiseAbs();
                e[3] = (bb.dense_consistent(0.5) - ex(t0 + 0.5 * h)).cwiseAbs();
                e[4] = (bb.dense_consistent(0.3) - ex(t0 + 0.3 * h)).cwiseAbs();
            };
            REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
            e[0] = (b.x - ex(t0 + h)).cwiseAbs();
            errs.push_back(e);
        }
        for (int v : vars) {
            for (int w = 0; w < 5; ++w) {
                static const char* lab[] = {"end  ", "th.5 ", "th.3 ", "cs.5 ", "cs.3 "};
                std::printf("  var %d %s:", v, lab[w]);
                for (std::size_t i = 0; i < errs.size(); ++i) {
                    std::printf(" %.2e", errs[i][w][v]);
                    if (i) std::printf("(%.2f)", order(errs[i - 1][w][v], errs[i][w][v]));
                }
                std::printf("\n");
            }
        }
    };
    local("RC network", rc_net(), rc_exact, 0.3, {1, 2, 3});
    local("cap across source", cap_loop(), cap_loop_exact, 0.3, {1});
    local("inductor cutset", ind_cutset(), ind_cutset_exact, 0.3, {0});
    local("PR lambda=-1", prothero(-1), [](Real t) { return Vec(Vec{{std::sin(t)}}); }, 0.3, {0});

    // LC fixed (task 2.3)
    for (Size n : {20, 40, 80}) {
        const Real T = lc_T();
        const Vec x0{{kV0, 0}};
        std::printf("LC T0/%zu 10 periods: esdirk %.3e  trbdf2 %.3e\n", static_cast<size_t>(n),
                    lc_err(fixed<E>(lc(), x0, 0, 10 * T, 10 * n).x, 10 * T),
                    lc_err(fixed<step::TrBdf2>(lc(), x0, 0, 10 * T, 10 * n).x, 10 * T));
    }

    // ladder after the source step: fixed h = 1e-6 (z = h/tau up to 1e8)
    {
        const Circuit c = ladder();
        std::printf("ladder fixed h=1e-7: va error after steps 1..4 (esdirk | trapezoidal)\n");
        for (int pass = 0; pass < 2; ++pass) {
            std::printf("  %s:", pass ? "trap  " : "esdirk");
            const auto hookE = [&](const mna::Body<E>& bb, const step::StepFrame& f) {
                std::printf(" va(%g)=%.6f", f.t1, bb.result()[1]);
            };
            const auto hookT = [&](const mna::Body<step::Trapezoidal>& bb, const step::StepFrame& f) {
                std::printf(" va(%g)=%.6f", f.t1, bb.result()[1]);
            };
            if (pass) fixed<step::Trapezoidal>(c, ladder_start(c), 0, 4e-7, 4, hookT);
            else fixed<E>(c, ladder_start(c), 0, 4e-7, 4, hookE);
            std::printf("\n");
        }
    }

    // work-precision, adaptive (both with the filtered estimator)
    const auto wp = [](const char* name, auto run_one) {
        std::printf("%s work-precision\n  %-9s %-8s %7s %5s %7s %7s %7s %10s\n", name, "method", "rtol",
                    "steps", "rej", "factor", "solves", "newton", "max err");
        for (Real rtol : {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9}) {
            run_one(rtol, true);
            run_one(rtol, false);
        }
    };
    const auto row = [](const char* m, Real rtol, const auto& r, Real err) {
        std::printf("  [max newton/stage %zu, stages %zu, fails %zu, refresh %zu]", static_cast<size_t>(r.max_newton), static_cast<size_t>(r.stages), static_cast<size_t>(r.newton_fails), static_cast<size_t>(r.refreshes));
        std::printf("  %-9s %-8.0e %7zu %5zu %7zu %7zu %7zu %10.3e\n", m, rtol, static_cast<size_t>(r.steps),
                    static_cast<size_t>(r.rejects), static_cast<size_t>(r.factors),
                    static_cast<size_t>(r.solves), static_cast<size_t>(r.newton), err);
    };
    wp("LC 100 periods", [&](Real rtol, bool es) {
        std::vector<Real> ts;
        for (int k = 1; k <= 10; ++k) ts.push_back(10 * k * lc_T());
        Real err = 0;
        const auto at = [&](const Vec& x, Real t) { err = std::max(err, lc_err(x, t)); };
        const Vec x0{{kV0, 0}};
        if (es) row("esdirk43", rtol, adaptive<E>(lc(), x0, rtol, rtol * 1e-3, lc_T() / 50, ts, at), err);
        else row("trbdf2", rtol, adaptive<step::TrBdf2>(lc(), x0, rtol, rtol * 1e-3, lc_T() / 50, ts, at), err);
    });
    wp("ladder 0..50us", [&](Real rtol, bool es) {
        const Circuit c = ladder();
        std::vector<Real> ts;
        for (const auto& r : kLadderRef) ts.push_back(r.t);
        Real err = 0;
        std::size_t i = 0;
        const auto at = [&](const Vec& x, Real) { err = std::max(err, ladder_err(x, kLadderRef[i++])); };
        if (es) row("esdirk43", rtol, adaptive<E>(c, ladder_start(c), rtol, rtol * 1e-3, 1e-9, ts, at), err);
        else row("trbdf2", rtol, adaptive<step::TrBdf2>(c, ladder_start(c), rtol, rtol * 1e-3, 1e-9, ts, at), err);
    });
    wp("diode clamp 0..2ms", [&](Real rtol, bool es) {
        const Circuit c = diode();
        std::vector<Real> ts;
        for (const auto& r : kDiodeRef) ts.push_back(r.t);
        Real err = 0;
        std::size_t i = 0;
        const auto at = [&](const Vec& x, Real) { err = std::max(err, std::abs(x[1] - kDiodeRef[i++].vo)); };
        if (es) row("esdirk43", rtol, adaptive<E>(c, Vec::Zero(3), rtol, rtol * 1e-3, 1e-7, ts, at), err);
        else row("trbdf2", rtol, adaptive<step::TrBdf2>(c, Vec::Zero(3), rtol, rtol * 1e-3, 1e-7, ts, at), err);
    });
    // the stiff filter on a stiff transient: PR lambda = -1e6, y0 = 1 (phi(0) = 0)
    for (bool filt : {true, false}) {
        Real err = 0;
        const auto at = [&](const Vec& x, Real t) { err = std::max(err, std::abs(x[0] - std::sin(t))); };
        row(filt ? "PR filt" : "PR raw", 1e-6,
            adaptive<E>(prothero(-1e6), Vec{{1.0}}, 1e-6, 1e-9, 0.1, {0.5, 1.0, 1.5}, at, filt), err);
    }
    // the stiff filter: the ladder with the raw embedded estimate
    for (Real rtol : {1e-5, 1e-7}) {
        const Circuit c = ladder();
        Real err = 0;
        std::size_t i = 0;
        std::vector<Real> ts;
        for (const auto& r : kLadderRef) ts.push_back(r.t);
        const auto at = [&](const Vec& x, Real) { err = std::max(err, ladder_err(x, kLadderRef[i++])); };
        row("es-raw", rtol, adaptive<E>(c, ladder_start(c), rtol, rtol * 1e-3, 1e-9, ts, at, false), err);
    }
}

namespace {

// R(z) = 1 + z·w^T (I − zA)^{-1} 1 for the weights w (b or b̂)
std::complex<Real> stab(std::complex<Real> z, bool embedded = false) {
    using CM = Eigen::Matrix<std::complex<Real>, 6, 6>;
    using CV = Eigen::Matrix<std::complex<Real>, 6, 1>;
    CM M;
    CV w;
    for (int i = 0; i < 6; ++i) {
        w[i] = embedded ? E::b_hat(i) : E::b(i);
        for (int j = 0; j < 6; ++j) M(i, j) = (i == j ? Real{1} : Real{0}) - z * E::a(i, j);
    }
    const CV k = M.partialPivLu().solve(CV::Ones());
    return Real{1} + z * w.dot(k);   // dot conjugates its first argument: w is real
}

// The eight order-4 conditions of weights w; returns max |residual| per order.
template <class W>
std::array<Real, 4> order_residuals(W w, Real th = 1) {
    std::array<Real, 6> c{}, ac{}, ac2{}, aac{};
    for (int i = 0; i < 6; ++i) c[i] = E::c(i);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            ac[i] += E::a(i, j) * c[j];
            ac2[i] += E::a(i, j) * c[j] * c[j];
        }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) aac[i] += E::a(i, j) * ac[j];
    Real s[8] = {};
    for (int i = 0; i < 6; ++i) {
        s[0] += w(i);
        s[1] += w(i) * c[i];
        s[2] += w(i) * c[i] * c[i];
        s[3] += w(i) * ac[i];
        s[4] += w(i) * c[i] * c[i] * c[i];
        s[5] += w(i) * c[i] * ac[i];
        s[6] += w(i) * ac2[i];
        s[7] += w(i) * aac[i];
    }
    const Real t2 = th * th, t3 = t2 * th, t4 = t3 * th;
    return {std::abs(s[0] - th), std::abs(s[1] - t2 / 2),
            std::max(std::abs(s[2] - t3 / 3), std::abs(s[3] - t3 / 6)),
            std::max({std::abs(s[4] - t4 / 4), std::abs(s[5] - t4 / 8), std::abs(s[6] - t4 / 12),
                      std::abs(s[7] - t4 / 24)})};
}

}  // namespace

TEST_CASE("ESDIRK4(3): tableau properties (doubles; exact in probes/)", "[step][esdirk43]") {
    STATIC_CHECK(step::Integrator<E>);
    STATIC_CHECK(E::err_exponent == Real{-0.25});
    for (int i = 0; i < 6; ++i) {
        Real row = 0, ac = 0;
        for (int j = 0; j < 6; ++j) {
            row += E::a(i, j);
            ac += E::a(i, j) * E::c(j);
        }
        CHECK(std::abs(row - E::c(i)) < 1e-15);
        CHECK(std::abs(ac - E::c(i) * E::c(i) / 2) < 1e-15);   // stage order 2
        CHECK(E::b(i) == E::a(5, i));                            // stiffly accurate
    }
    const auto rb = order_residuals([](int i) { return E::b(i); });
    for (const Real r : rb) CHECK(r < 1e-14);
    const auto rh = order_residuals([](int i) { return E::b_hat(i); });
    CHECK(rh[0] < 1e-14);
    CHECK(rh[1] < 1e-14);
    CHECK(rh[2] < 1e-14);
    CHECK(rh[3] > 1e-4);   // b̂ is third order, not fourth (1.4e-3 exact)
    // design.md reference values (task 1.3)
    CHECK(std::abs(stab(-10).real() - 0.13657) < 1e-5);
    CHECK(std::abs(stab(-100).real() - 0.075733) < 1e-6);
    CHECK(std::abs(stab(-1e4).real() - 9.3136e-4) < 1e-8);
    // R(−∞) = 0 and R̂(−∞) = −3/20 exactly (probes/esdirk43_stability.py);
    // at z = −1e6: R = 28/(3·1e6) + O(1e-12) (the reviewer's closed form)
    CHECK(std::abs(stab(-1e6)) < 1e-5);
    CHECK(std::abs(stab(-1e6, true).real() + 0.15) < 1e-4);
    Real worst = 0;
    for (int k = 0; k < 4000; ++k) {
        const Real y = std::pow(10.0, -3 + 9 * k / 3999.0);
        worst = std::max(worst, std::abs(stab({0, y})));
    }
    CHECK(worst <= 1 + 1e-12);
    // stage specs: one factor (h/2), devices told γh, the last stage ends on h
    for (int s = 0; s < E::n_stages; ++s) {
        const auto sp = E::stage(0.3, s);
        CHECK(sp.factor_dt == 0.15);
        CHECK(sp.ctx_h == 0.075);
        CHECK(sp.formula == pulsim::pwl::TrBdf2Stage::BackwardEuler);
    }
    CHECK(E::stage(0.3, 4).t_off == 0.3);
    // dense weights: w(0) = 0, w(1) = e_4, order-4 conditions for every θ
    for (int s = 0; s < E::n_stages; ++s) {
        CHECK(E::dense_weight(s, 0) == 0);
        CHECK(std::abs(E::dense_weight(s, 1) - (s == 4 ? 1 : 0)) < 1e-12);
    }
    for (const Real th : {0.1, 0.25, 0.5, 0.77, 0.9}) {
        const auto bt = [th](int j) {
            Real r = 0;
            for (int s = 0; s < E::n_stages; ++s) r += E::dense_weight(s, th) * E::a(s + 1, j);
            return r;
        };
        for (const Real r : order_residuals(bt, th)) CHECK(r < 1e-12);
    }
}

TEST_CASE("ESDIRK4(3): fixed-step orders on ODE, index-1 and index-2 MNA", "[step][esdirk43]") {
    // LC tank (task 2.3): 10 periods, amplitude error at T0/20, /40, /80
    const Real T = lc_T();
    const Vec x0{{kV0, 0}};
    const Real e20 = lc_err(fixed<E>(lc(), x0, 0, 10 * T, 200).x, 10 * T);
    const Real e40 = lc_err(fixed<E>(lc(), x0, 0, 10 * T, 400).x, 10 * T);
    const Real e80 = lc_err(fixed<E>(lc(), x0, 0, 10 * T, 800).x, 10 * T);
    CHECK(std::abs(e20 / 5.16e-4 - 1) < 0.05);
    CHECK(std::abs(e40 / 3.24e-5 - 1) < 0.05);
    CHECK(std::abs(e80 / 2.02e-6 - 1) < 0.05);
    CHECK(order(e40, e80) >= 3.9);

    // index 1: order 4 on the differential v3 AND on the algebraic v2, iV
    {
        const Vec ea = (fixed<E>(rc_net(), rc_exact(0), 0, 2, 40).x - rc_exact(2)).cwiseAbs();
        const Vec eb = (fixed<E>(rc_net(), rc_exact(0), 0, 2, 80).x - rc_exact(2)).cwiseAbs();
        for (int v : {1, 2, 3}) CHECK(order(ea[v], eb[v]) >= 3.9);   // measured 4.00
        CHECK(eb[0] < 1e-15);                                        // v1 = Vs exactly
    }
    // index 2: the algebraic unknown is order 2 (the stage order), not 4;
    // still below TR-BDF2 (2.4x at N = 320)
    const auto idx2 = [](const Circuit& c, const std::function<Vec(Real)>& ex, int v) {
        const Real a = std::abs(fixed<E>(c, ex(0), 0, 2, 160).x[v] - ex(2)[v]);
        const Real b = std::abs(fixed<E>(c, ex(0), 0, 2, 320).x[v] - ex(2)[v]);
        const Real tb = std::abs(fixed<step::TrBdf2>(c, ex(0), 0, 2, 320).x[v] - ex(2)[v]);
        CHECK(order(a, b) > 1.8);   // measured 1.99
        CHECK(order(a, b) < 2.2);
        CHECK(b < tb);
    };
    idx2(cap_loop(), cap_loop_exact, 1);
    idx2(cutset_fb(), cutset_fb_exact, 0);
    // the differential state that index-2 voltage feeds stays order 4
    // (measured 4.00; TR-BDF2 2.00): the O(h^2) error of v1 is in its
    // step-end value only, the stages see v1 through q of the inductor
    {
        const auto e = [](Size n) {
            return std::abs(fixed<E>(cutset_fb(), cutset_fb_exact(0), 0, 2, n).x[2] - cutset_fb_exact(2)[2]);
        };
        CHECK(order(e(40), e(80)) > 3.9);
    }

    // Prothero–Robinson, λ = −1e6 (stiff: hλ from −1.5e5 to −1.9e4):
    // order reduction to the stage-order regime, at least 2 (task 2.5)
    {
        const auto pr = [](Size n) {
            return std::abs(fixed<E>(prothero(-1e6), Vec{{0.0}}, 0, 1.5, n).x[0] - std::sin(1.5));
        };
        const Real e10 = pr(10), e20s = pr(20), e40s = pr(40), e80s = pr(80);
        CHECK(order(e10, e20s) >= 2.0);   // measured 2.59, 2.42, 2.27
        CHECK(order(e20s, e40s) >= 2.0);
        CHECK(order(e40s, e80s) >= 2.0);
        CHECK(order(e40s, e80s) < 3.0);
    }
}

TEST_CASE("ESDIRK4(3): dense output of order 4, consistent on algebraic unknowns",
          "[step][esdirk43]") {
    // one step from the exact state at t0 = 0.3; local errors at θ = 0.5
    const Circuit c = rc_net();
    const auto local = [&](Real h, bool consistent) {
        mna::Body<E> b(c, rc_exact(0.3));
        step::FixedController k{{0.3, h}, 1, 2};
        Vec e;
        b.hook = [&](const mna::Body<E>& bb, const step::StepFrame&) {
            e = ((consistent ? bb.dense_consistent(0.5) : bb.dense(0.5)) - rc_exact(0.3 + 0.5 * h))
                    .cwiseAbs();
        };
        REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
        return e;
    };
    const Vec p1 = local(0.1, false), p2 = local(0.05, false);
    const Vec c1 = local(0.1, true), c2 = local(0.05, true);
    // differential v3: local order 5 (a fourth-order interpolant) as is
    CHECK(order(p1[2], p2[2]) > 4.7);   // measured 5.16
    // algebraic v2, iV: the stage-value combination alone is local order 3
    // (Σw·c³ ≠ θ³: stage order 2); re-solved at t0 + θh they are order 5
    CHECK(order(p1[1], p2[1]) < 3.3);   // measured 2.82
    for (int v : {1, 2, 3}) CHECK(order(c1[v], c2[v]) > 4.7);   // measured 5.16
    CHECK(c2[1] * 100 < p2[1]);

    // event location: the instant v2 (a capacitor-free node) crosses 0.2,
    // located on the consistent dense output of fixed steps
    const auto exact_v2 = [](Real t) { return rc_exact(t)[1] - 0.2; };
    Real lo = 0, hi = 0.6;   // the first upward crossing lies in (0, 0.6)
    REQUIRE(exact_v2(lo) < 0);
    REQUIRE(exact_v2(hi) > 0);
    for (int i = 0; i < 200; ++i) ((exact_v2(0.5 * (lo + hi)) < 0) ? lo : hi) = 0.5 * (lo + hi);
    const Real t_star = 0.5 * (lo + hi);
    const auto locate = [&](Size n) {
        Real found = -1;
        const auto hook = [&](const mna::Body<E>& bb, const step::StepFrame& f) {
            if (found >= 0 || !(bb.result()[1] >= 0.2) || !(bb.x0[1] < 0.2)) return;
            Real a = 0, z = 1;
            for (int i = 0; i < 60; ++i) {
                const Real m = 0.5 * (a + z);
                (bb.dense_consistent(m)[1] < 0.2 ? a : z) = m;
            }
            found = f.t0 + 0.5 * (a + z) * f.h;
        };
        fixed<E>(c, rc_exact(0), 0, 1, n, hook);
        return std::abs(found - t_star);
    };
    const Real l10 = locate(10), l20 = locate(20);
    CHECK(order(l10, l20) > 3.7);   // a fourth-order event instant
}

TEST_CASE("ESDIRK4(3): L-stable damping, no trapezoidal ringing", "[step][esdirk43]") {
    // RC charged in steps of 100 τ (task 2.3): remaining error R(−100)^k
    {
        Circuit rc;
        rc.C = Mat{{1}};
        rc.G = Mat{{1}};
        rc.b = [](Real) { return Vec(Vec{{1.0}}); };
        rc.norm_vars = {0};
        const Real r1 = 1 - fixed<E>(rc, Vec{{0.0}}, 0, 100, 1).x[0];
        const Real r3 = 1 - fixed<E>(rc, Vec{{0.0}}, 0, 300, 3).x[0];
        CHECK(std::abs(r1 - 0.0757) < 1e-4);
        CHECK(r1 > 0);   // no overshoot
        CHECK(std::abs(r3) < 5e-4);
    }
    // the ladder after a 10 V source step, h = 1e-7 (h/τ = 1e6 and 1e7):
    // va stays on its slow manifold; the trapezoidal rule swings ±10 V
    const Circuit c = ladder();
    std::vector<Real> es, tr;
    fixed<E>(c, ladder_start(c), 0, 4e-7, 4,
             [&](const mna::Body<E>& b, const step::StepFrame&) { es.push_back(b.result()[1]); });
    fixed<step::Trapezoidal>(c, ladder_start(c), 0, 4e-7, 4,
                             [&](const mna::Body<step::Trapezoidal>& b, const step::StepFrame&) {
                                 tr.push_back(b.result()[1]);
                             });
    CHECK(std::abs(es[0] - kLadderRef[0].va) < 1e-3);   // measured 9.3e-5
    for (std::size_t i = 1; i < es.size(); ++i) CHECK(std::abs(es[i] - es[i - 1]) < 2e-3);
    for (std::size_t i = 1; i < tr.size(); ++i) CHECK(std::abs(tr[i] - tr[i - 1]) > 15);
}

namespace {

template <class I>
Run<I> lc_adaptive(Real rtol, int periods) {
    std::vector<Real> ts;
    for (int k = 1; k <= periods; ++k) ts.push_back(k * lc_T());
    Real err = 0;
    auto r = adaptive<I>(lc(), Vec{{kV0, 0}}, rtol, rtol * 1e-3, lc_T() / 50, ts,
                         [&](const Vec& x, Real t) { err = std::max(err, lc_err(x, t)); });
    r.err = err;
    return r;
}
template <class I>
Run<I> ladder_adaptive(Real rtol) {
    const Circuit c = ladder();
    std::vector<Real> ts;
    for (const auto& r : kLadderRef) ts.push_back(r.t);
    Real err = 0;
    std::size_t i = 0;
    auto r = adaptive<I>(c, ladder_start(c), rtol, rtol * 1e-3, 1e-9, ts,
                         [&](const Vec& x, Real) { err = std::max(err, ladder_err(x, kLadderRef[i++])); });
    r.err = err;
    return r;
}
template <class I>
Run<I> diode_adaptive(Real rtol) {
    std::vector<Real> ts;
    for (const auto& r : kDiodeRef) ts.push_back(r.t);
    Real err = 0;
    std::size_t i = 0;
    auto r = adaptive<I>(diode(), Vec::Zero(3), rtol, rtol * 1e-3, 1e-7, ts,
                         [&](const Vec& x, Real) { err = std::max(err, std::abs(x[1] - kDiodeRef[i++].vo)); });
    r.err = err;
    return r;
}

// The ladder with its source toggling 10 V / 0 V every 5 µs (10 events).
// At each event the engine's consistent solve re-sets the algebraic
// unknowns and the next step starts with no history (restart = true).
struct ToggleRun {
    Size steps = 0, rejects = 0;
    Vec x;
    Real alg_resid = 0;   // worst algebraic residual at the first step after an event
    std::vector<Vec> after_event;   // state after the first step past each event
    std::vector<Vec> at_event;
    Real h_at_event = 0;
};
ToggleRun toggle(bool filter, Real rtol) {
    Real level = 10;
    Circuit c = ladder();
    c.b = [&level](Real) { return Vec(Vec{{0, 0, 0, 0, 0, level}}); };
    mna::Body<E> b(c, c.consistent(Vec::Zero(6), 0));
    b.filter = filter;
    b.rtol = rtol;
    b.atol = rtol * 1e-3;
    step::LteClock<E> k;
    k.h_min = 1e-20;
    k.ctrl.h = 1e-9;
    ToggleRun out;
    bool first = false;
    b.hook = [&](const mna::Body<E>& bb, const step::StepFrame& f) {
        if (!first) return;
        first = false;
        out.after_event.push_back(bb.result());
        const Eigen::FullPivLU<Mat> lct(c.C.transpose());
        const Vec r = lct.kernel().transpose() * c.f(bb.result(), f.t1);
        out.alg_resid = std::max(out.alg_resid, r.cwiseAbs().maxCoeff() / 100);   // A vs 100 A scale
    };
    for (int e = 1; e <= 10; ++e) {
        k.t_end = 5e-6 * e;
        REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
        level = (e % 2) ? 0 : 10;
        b.x = c.consistent(b.x, k.t);   // the post-event solve
        b.restart = true;
        out.at_event.push_back(b.x);
        first = true;
    }
    out.steps = k.n_accept;
    out.rejects = k.n_reject;
    out.x = b.x;
    return out;
}

}  // namespace

TEST_CASE("ESDIRK4(3): adaptive LC, one factor per step, against TR-BDF2", "[step][esdirk43]") {
    // task 2.4 on 20 periods (1000 in DECISIONS 7): steps per period and accuracy
    const auto e6 = lc_adaptive<E>(1e-6, 20);
    CHECK(static_cast<Real>(e6.steps) / 20 <= 34);   // measured 26.1
    CHECK(e6.err < 1e-2);                            // measured 1.8e-3 (100 periods)
    // a linear circuit: one factor per attempt, 5 stage solves + 1 filter solve
    CHECK(e6.factors <= e6.steps + e6.rejects);
    CHECK(e6.solves == 6 * (e6.steps + e6.rejects));
    CHECK(e6.newton == e6.stages);
    // gate 2.6: linear solves at most a quarter of TR-BDF2's at equal rtol
    for (const Real rtol : {1e-5, 1e-6}) {
        const auto es = lc_adaptive<E>(rtol, 20);
        const auto tr = lc_adaptive<step::TrBdf2>(rtol, 20);
        CHECK(4 * es.solves <= tr.solves);   // measured 5.8x, 6.7x fewer
        CHECK(es.err < tr.err);
    }
}

TEST_CASE("ESDIRK4(3): work-precision against TR-BDF2 (ladder, Shockley clamp)",
          "[step][esdirk43]") {
    // equal or better error at a third or less of the solves: ESDIRK at
    // rtol r against TR-BDF2 at r/10 (DECISIONS 7 has the full tables)
    {
        const auto es = ladder_adaptive<E>(1e-6);
        const auto tr = ladder_adaptive<step::TrBdf2>(1e-7);
        CHECK(3 * es.solves < tr.solves);   // measured 450 vs 2919
        CHECK(es.err < 3 * tr.err);         // measured 6.6e-5 vs 5.7e-5
        CHECK(es.err < 5 * 6.7e-5);         // against the mpmath reference
    }
    {
        const auto es = diode_adaptive<E>(1e-7);
        const auto tr = diode_adaptive<step::TrBdf2>(1e-8);
        CHECK(3 * es.solves < tr.solves);   // measured 3729 vs 17216
        CHECK(es.err < tr.err);             // measured 3.3e-7 vs 1.3e-6
        CHECK(es.err < 5 * 3.3e-7);         // against scipy Radau at 1e-13
    }
}

TEST_CASE("ESDIRK4(3): Newton reuse on a Shockley clamp", "[step][esdirk43]") {
    const auto r = diode_adaptive<E>(1e-7);
    const Size attempts = r.steps + r.rejects;
    // one Jacobian / factor per attempt; a refresh only on a stalled stage
    CHECK(r.factors == attempts + r.refreshes);
    CHECK(10 * r.refreshes <= attempts);       // measured 0 (3 at rtol 1e-6)
    CHECK(20 * r.newton_fails <= attempts);    // measured 0
    CHECK(r.stages == 5 * attempts);
    // iterations per stage, from the stage predictor (bounds: +30 %)
    CHECK(r.max_newton <= 7);                                              // measured 5
    CHECK(static_cast<Real>(r.newton) <= 3.3 * static_cast<Real>(r.stages));   // measured 2.55
}

TEST_CASE("ESDIRK4(3): restart after an event needs no history; filtered estimate",
          "[step][esdirk43]") {
    const ToggleRun f = toggle(true, 1e-6);
    // stiffly accurate: the first step after each event ends consistent
    CHECK(f.after_event.size() == 9);   // the 10th event is at the end
    CHECK(f.alg_resid < 1e-12);
    // no history: the step after an event equals the first step of a
    // fresh body and clock started from the post-event state
    {
        Real level = 0;   // after event 1 (t = 5 µs) the source is 0 V
        Circuit c = ladder();
        c.b = [&level](Real) { return Vec(Vec{{0, 0, 0, 0, 0, level}}); };
        const ToggleRun probe = toggle(true, 1e-6);
        mna::Body<E> fresh(c, probe.at_event[0]);
        fresh.rtol = 1e-6;
        fresh.atol = 1e-9;
        // the clock's proposal is the one thing carried: replay it
        Vec first_after;
        {
            Real lv = 10;
            Circuit c2 = ladder();
            c2.b = [&lv](Real) { return Vec(Vec{{0, 0, 0, 0, 0, lv}}); };
            mna::Body<E> b(c2, c2.consistent(Vec::Zero(6), 0));
            b.rtol = 1e-6;
            b.atol = 1e-9;
            step::LteClock<E> k;
            k.h_min = 1e-20;
            k.ctrl.h = 1e-9;
            k.t_end = 5e-6;
            REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
            lv = 0;
            b.x = c2.consistent(b.x, k.t);
            b.restart = true;
            const Real h_next = k.ctrl.h;
            k.t_end = 1;
            REQUIRE(step::step_once<E>(k, b) == step::StepStatus::Accepted);
            first_after = b.x;
            step::LteClock<E> kf;
            kf.h_min = 1e-20;
            kf.ctrl.h = h_next;
            kf.t = 5e-6;
            kf.t_end = 1;
            REQUIRE(step::step_once<E>(kf, fresh) == step::StepStatus::Accepted);
        }
        // the same code on the same data (≤ 1e-14: no claim on the last bit)
        CHECK((fresh.x - first_after).norm() <= 1e-14 * first_after.norm());
        CHECK((fresh.x - probe.after_event[0]).norm() <= 1e-14 * first_after.norm());
    }
    // the stiff filter: each event leaves va 10 V off its 1e-13 s manifold;
    // the raw embedded estimate (R̂(−∞) = −3/20) reads that as error
    const ToggleRun raw = toggle(false, 1e-6);
    CHECK(raw.steps + raw.rejects > 2 * (f.steps + f.rejects));
}

TEST_CASE("ESDIRK4(3): long measurements for DECISIONS", "[.esdirk_measure_long]") {
    for (bool filt : {true, false}) {
        const ToggleRun t = toggle(filt, 1e-6);
        std::printf("toggle ladder %s: steps %zu rejects %zu\n", filt ? "filtered" : "raw",
                    static_cast<size_t>(t.steps), static_cast<size_t>(t.rejects));
    }
    for (const Real rtol : {1e-5, 1e-6, 1e-8}) {
        const auto es = lc_adaptive<E>(rtol, 1000);
        const auto tr = lc_adaptive<step::TrBdf2>(rtol, 1000);
        std::printf("LC 1000 periods rtol %.0e: esdirk %.1f steps/period %.1f solves/period err %.3e V | "
                    "trbdf2 %.1f steps/period %.1f solves/period err %.3e V | solves ratio %.2f\n",
                    rtol, es.steps / 1000.0, es.solves / 1000.0, es.err * kV0, tr.steps / 1000.0,
                    tr.solves / 1000.0, tr.err * kV0, static_cast<Real>(tr.solves) / es.solves);
    }
}

namespace {

// An event policy on the skeleton (body_localizes): a sign change of the
// predicate p over an accepted step is bracketed on the dense output (the
// seed), then landed by re-steps, Illinois on the re-steps' END values:
// every value it uses is a full fourth-order step, so the located instant
// does not inherit the interpolant's local order 3 on stiff modes,
// algebraic unknowns or a nonlinear q.
struct Lander : mna::Body<E> {
    static constexpr bool localizes = true;
    std::function<Real(const Vec&)> p;
    Real ptol = 1e-12;
    bool active = false;
    int side = 0;
    Real t_lo = 0, p_lo = 0, t_hi = 0, p_hi = 0;
    Size cuts = 0;
    std::vector<Real> seeds, located;
    using mna::Body<E>::Body;

    bool cut_to(step::StepFrame& f, Real tn) {
        ++cuts;
        f.h = tn - f.t0;
        f.t1 = tn;
        return false;
    }
    bool after_solve(step::StepFrame& f) {
        const Real p0 = p(x0), p1 = p(result());
        if (!active) {
            if (std::abs(p0) <= ptol || (p0 < 0) == (p1 < 0) || std::abs(p1) <= ptol) return true;
            Real a = 0, z = 1;
            for (int i = 0; i < 60; ++i) {
                const Real m = 0.5 * (a + z);
                ((p(dense(m)) < 0) == (p0 < 0) ? a : z) = m;
            }
            active = true;
            side = 0;
            t_lo = f.t0, p_lo = p0, t_hi = f.t1, p_hi = p1;
            seeds.push_back(f.t0 + 0.5 * (a + z) * f.h);
            return cut_to(f, seeds.back());
        }
        if (std::abs(p1) <= ptol || t_hi - t_lo <= 1e-15 * std::abs(t_hi)) {
            active = false;
            located.push_back(f.t1);
            return true;
        }
        if ((p1 < 0) == (p_lo < 0)) {
            t_lo = f.t1, p_lo = p1;
            if (side == -1) p_hi *= 0.5;
            side = -1;
        } else {
            t_hi = f.t1, p_hi = p1;
            if (side == +1) p_lo *= 0.5;
            side = +1;
        }
        return cut_to(f, t_lo - p_lo * (t_hi - t_lo) / (p_hi - p_lo));
    }
};

}  // namespace

TEST_CASE("ESDIRK4(3): events bracketed on the dense output, landed by re-steps",
          "[step][esdirk43]") {
    // a stiff mode (Prothero–Robinson, λ = −1e6, y = sin t exactly): the
    // instant y = 1/2 is π/6; fixed grid h = 0.1
    {
        const Circuit c = prothero(-1e6);
        Lander b(c, Vec{{0.0}});
        b.p = [](const Vec& x) { return x[0] - 0.5; };
        step::FixedController k{{0, 0.1}, 1, 11};
        REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
        REQUIRE(b.located.size() == 1);
        const Real seed_err = std::abs(b.seeds[0] - kPi / 6);
        const Real land_err = std::abs(b.located[0] - kPi / 6);
        CHECK(seed_err > 1e-6);    // the interpolant on a stiff mode: order 3
        CHECK(land_err < 1e-10);   // the landed instant: the steps' accuracy
        CHECK(b.cuts <= 8);
        CHECK(std::abs(b.x[0] - std::sin(1.0)) < 1e-9);
    }
    // an algebraic unknown (v2 of the RC network crosses 0.2)
    {
        const Circuit c = rc_net();
        Lander b(c, rc_exact(0));
        b.p = [](const Vec& x) { return x[1] - 0.2; };
        const auto ex = [](Real t) { return rc_exact(t)[1] - 0.2; };
        Real lo = 0, hi = 0.6;
        for (int i = 0; i < 200; ++i) ((ex(0.5 * (lo + hi)) < 0) ? lo : hi) = 0.5 * (lo + hi);
        step::FixedController k{{0, 0.1}, 1, 11};
        REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
        REQUIRE(!b.located.empty());
        CHECK(std::abs(b.located[0] - 0.5 * (lo + hi)) < 1e-7);
        CHECK(std::abs(b.located[0] - 0.5 * (lo + hi)) * 10 < std::abs(b.seeds[0] - 0.5 * (lo + hi)));
    }
}

TEST_CASE("ESDIRK4(3): event measurements", "[.esdirk_measure]") {
    for (const Real h : {0.2, 0.1, 0.05}) {
        const Circuit c = prothero(-1e6);
        Lander b(c, Vec{{0.0}});
        b.p = [](const Vec& x) { return x[0] - 0.5; };
        step::FixedController k{{0, h}, 1, static_cast<Size>(1.0 / h + 0.5) + 1};
        REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
        std::printf("PR event h=%g: seed err %.3e landed err %.3e cuts %zu\n", h,
                    std::abs(b.seeds[0] - kPi / 6), std::abs(b.located[0] - kPi / 6),
                    static_cast<size_t>(b.cuts));
    }
    for (const Size n : {5, 10, 20}) {
        const Circuit c = rc_net();
        Lander b(c, rc_exact(0));
        b.p = [](const Vec& x) { return x[1] - 0.2; };
        const auto ex = [](Real t) { return rc_exact(t)[1] - 0.2; };
        Real lo = 0, hi = 0.6;
        for (int i = 0; i < 200; ++i) ((ex(0.5 * (lo + hi)) < 0) ? lo : hi) = 0.5 * (lo + hi);
        step::FixedController k{{0, 1.0 / static_cast<Real>(n)}, 1, n + 1};
        REQUIRE(step::run<E>(k, b) == step::StepStatus::Done);
        std::printf("RC v2 event h=1/%zu: seed err %.3e landed err %.3e cuts %zu\n", static_cast<size_t>(n),
                    std::abs(b.seeds[0] - 0.5 * (lo + hi)), std::abs(b.located[0] - 0.5 * (lo + hi)),
                    static_cast<size_t>(b.cuts));
    }
}
