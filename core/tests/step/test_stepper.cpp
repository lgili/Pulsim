// refactor-unified-transient-stepper tasks 6.1 / 4.x (design D1, D2, D4):
// the step skeleton (core/include/pulsim/solver/step/stepper.hpp) driven by
// a toy body on x' = A·x. The body solves each stage from its StageSpec
// alone (formula, factor, step context, stage weights), so a method runs
// here only if the integrator interface carries it: Sdirk22 is a method
// neither driver hard-codes.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "pulsim/solver/step/stepper.hpp"

using pulsim::Real;
using pulsim::Size;
using pulsim::pwl::TrBdf2Stage;
namespace step = pulsim::step;

namespace {

using V2 = std::array<Real, 2>;

// x'' + 0.4 x' + 4 x = 0, x(0) = 1, x'(0) = 0.
constexpr Real kA[2][2] = {{0, 1}, {-4, -0.4}};

V2 mul(const V2& x) {
    return {kA[0][0] * x[0] + kA[0][1] * x[1], kA[1][0] * x[0] + kA[1][1] * x[1]};
}
// (alpha·I − beta·A)·y = r
V2 solve(Real alpha, Real beta, const V2& r) {
    const Real m00 = alpha - beta * kA[0][0], m01 = -beta * kA[0][1];
    const Real m10 = -beta * kA[1][0], m11 = alpha - beta * kA[1][1];
    const Real det = m00 * m11 - m01 * m10;
    return {(m11 * r[0] - m01 * r[1]) / det, (m00 * r[1] - m10 * r[0]) / det};
}
V2 exact(Real t) {
    const Real w = std::sqrt(Real{4} - Real{0.04});
    const Real e = std::exp(Real{-0.2} * t);
    const Real c = std::cos(w * t), s = std::sin(w * t);
    return {e * (c + Real{0.2} / w * s), e * (-(w + Real{0.04} / w) * s)};
}

template <class I>
struct ToyBody {
    V2 x{1, 0};
    V2 x_save{};
    std::array<V2, 4> y{};    // stage values of the step being taken
    std::array<V2, 4> xs{};   // their pseudo previous states
    std::vector<std::string> log;
    Real rtol = 1e-6, atol = 1e-9;
    Real event_t = -1;        // a state event the event policy localises
    bool fail_next_take = false;
    bool nan_once = false;
    Size n_solves = 0, n_restore = 0, n_bad_ctx = 0;

    bool poll() { log.emplace_back("poll"); return true; }
    void begin(const step::StepFrame&) { log.emplace_back("begin"); }
    Real clamp(const step::StepFrame&, Real h) { return h; }
    bool take(const step::StepFrame& f) {
        log.emplace_back("take");
        if (fail_next_take) { fail_next_take = false; return false; }
        return step::run_stages<I>(*this, f);
    }
    // The single solve point: everything it needs is in the spec.
    bool solve_stage(const step::StageSpec& s, int k, const step::StepFrame& f) {
        log.emplace_back("solve");
        ++n_solves;
        // the step context is the stage's own, never a previous solve's
        const bool ctx_ok =
            s.ctx_h > 0
            && (s.formula == TrBdf2Stage::Trapezoidal ? s.factor_dt == s.ctx_h
                : s.formula == TrBdf2Stage::BackwardEuler
                    ? s.factor_dt == Real{2} * s.ctx_h
                    : s.ctx_h == f.h);
        if (!ctx_ok) ++n_bad_ctx;
        V2 xstar = x;
        for (int j = 0; j < k; ++j) {
            const Real w = I::weight(k, j);
            for (int i = 0; i < 2; ++i) xstar[i] += w * (y[j][i] - xs[j][i]);
        }
        xs[k] = xstar;
        switch (s.formula) {
        case TrBdf2Stage::Trapezoidal: {   // from the step start
            const V2 ax = mul(x);
            const Real hh = s.ctx_h * Real{0.5};
            y[k] = solve(1, hh, {x[0] + hh * ax[0], x[1] + hh * ax[1]});
            break;
        }
        case TrBdf2Stage::BackwardEuler:
            y[k] = solve(1, s.ctx_h, xstar);
            break;
        case TrBdf2Stage::Bdf2Stage2: {   // (c1 y − h A y) = −c2 y_g − c3 x0
            using H = pulsim::pwl::HistoryState;
            y[k] = solve(H::trbdf2_c1(), f.h,
                         {-H::trbdf2_c2() * y[0][0] - H::trbdf2_c3() * x[0],
                          -H::trbdf2_c2() * y[0][1] - H::trbdf2_c3() * x[1]});
            break;
        }
        }
        return true;
    }
    V2 result() const { return y[I::n_stages - 1]; }
    Real error(const step::StepFrame& f) {
        log.emplace_back("error");
        if (nan_once) { nan_once = false; return std::nan(""); }
        const V2 y1 = result();
        V2 e{};
        if constexpr (I::n_stages == 2 && I::err_exponent == Real{-0.5}) {
            // Sdirk22: against the embedded x0 + h f(Y_1) = x0 + (Y_1 − x0)/γ
            for (int i = 0; i < 2; ++i)
                e[i] = y1[i] - (x[i] + (y[0][i] - x[i]) / step::Sdirk22::gamma());
        } else {
            const V2 fn = mul(x), fg = mul(y[0]), f1 = mul(y1);
            for (int i = 0; i < 2; ++i) e[i] = step::TrBdf2::lte(f.h, fn[i], fg[i], f1[i]);
        }
        Real s2 = 0;
        for (int i = 0; i < 2; ++i) {
            const Real sc = atol + rtol * std::max(std::abs(x[i]), std::abs(y1[i]));
            s2 += (e[i] / sc) * (e[i] / sc);
        }
        return std::sqrt(s2 / 2);
    }
    void save() { log.emplace_back("save"); x_save = x; }
    void restore() { log.emplace_back("restore"); ++n_restore; x = x_save; }
    bool after_solve(step::StepFrame& f) {
        log.emplace_back("after");
        if (event_t > f.t0 && event_t < f.t1) {   // localise: end on the event
            f.h = event_t - f.t0;
            f.t1 = event_t;
            return false;
        }
        return true;
    }
    void commit(const step::StepFrame&) { log.emplace_back("commit"); x = result(); }
    void on_step(const step::StepFrame&) { log.emplace_back("on_step"); }
};

template <class I>
Real fixed_error(Size n) {
    ToyBody<I> b;
    step::FixedController c{{0, Real{1} / static_cast<Real>(n)}, 1, n + 1};
    REQUIRE(step::run<I>(c, b) == step::StepStatus::Done);
    REQUIRE(b.n_bad_ctx == 0);
    const V2 ex = exact(c.clock.at(n));
    return std::hypot(b.x[0] - ex[0], b.x[1] - ex[1]);
}

template <class I>
Real observed_order() {
    return std::log2(fixed_error<I>(40) / fixed_error<I>(80));
}

}  // namespace

TEST_CASE("Stepper, fixed clock: each stage table converges at its order",
          "[step][stepper]") {
    CHECK(std::abs(observed_order<step::Trapezoidal>() - 2) < 0.2);
    CHECK(std::abs(observed_order<step::BackwardEuler>() - 1) < 0.2);
    CHECK(std::abs(observed_order<step::TrBdf2>() - 2) < 0.2);
    // the third method: no driver sequences it, the interface carries it
    CHECK(std::abs(observed_order<step::Sdirk22>() - 2) < 0.2);
}

TEST_CASE("Stepper, fixed clock: the phase order of one step",
          "[step][stepper]") {
    ToyBody<step::TrBdf2> b;
    step::FixedController c{{0, 0.01}, 1, 2};
    CHECK(step::step_once<step::TrBdf2>(c, b) == step::StepStatus::Accepted);
    const std::vector<std::string> want{"poll", "begin", "take", "solve", "solve",
                                        "after", "commit", "on_step"};
    CHECK(b.log == want);   // no rollback copy, no error call on a fixed clock
    CHECK(c.k == 2);
    CHECK(step::step_once<step::TrBdf2>(c, b) == step::StepStatus::Done);
}

TEST_CASE("Stepper, fixed clock: a failed step aborts before any commit",
          "[step][stepper]") {
    ToyBody<step::Trapezoidal> b;
    b.fail_next_take = true;
    step::FixedController c{{0, 0.01}, 1, 5};
    CHECK(step::run<step::Trapezoidal>(c, b) == step::StepStatus::Aborted);
    CHECK(b.x == V2{1, 0});
    CHECK(c.k == 1);
}

TEST_CASE("Stepper, fixed clock: the grid point is t0 + k*dt, never accumulated",
          "[step][stepper]") {
    ToyBody<step::Trapezoidal> b;
    // runtime operands on both sides (DECISIONS 9e)
    volatile Real t0 = 0.3, dt = 0.1;
    volatile Size k = 10;
    step::FixedController c{{t0, dt}, 1, 11};
    Real t_last = 0;
    while (!c.done()) {
        const auto f = c.frame();
        t_last = f.t1;
        CHECK(f.h == dt);
        REQUIRE(step::step_once<step::Trapezoidal>(c, b) == step::StepStatus::Accepted);
    }
    const Real grid = t0 + static_cast<Real>(k) * dt;
    CHECK(t_last == grid);
}

template <class I>
void lte_run_checks() {
    ToyBody<I> b;
    step::LteClock<I> c;
    c.t_end = 2;
    c.h_min = 1e-12;
    c.h_max = 0.5;
    c.ctrl.h = 0.5;   // far too large: the first step is rejected
    REQUIRE(step::run<I>(c, b) == step::StepStatus::Done);
    CHECK(c.t == c.t_end);
    CHECK(c.n_reject >= 1);
    CHECK(b.n_restore == c.n_reject);
    CHECK(b.n_bad_ctx == 0);
    const V2 ex = exact(c.t_end);
    CHECK(std::hypot(b.x[0] - ex[0], b.x[1] - ex[1]) < 1e-3);
}

TEST_CASE("Stepper, LTE clock: TR-BDF2 and SDIRK(2,2) reach t_end within tolerance",
          "[step][stepper]") {
    lte_run_checks<step::TrBdf2>();
    lte_run_checks<step::Sdirk22>();
}

TEST_CASE("Stepper, LTE clock: default h_max ends; a NaN error is retried, not accepted",
          "[step][stepper]") {
    ToyBody<step::TrBdf2> b;
    b.nan_once = true;
    step::LteClock<step::TrBdf2> c;
    c.t_end = 1;
    c.h_min = 1e-12;
    c.ctrl.h = 1e-3;
    REQUIRE(step::run<step::TrBdf2>(c, b) == step::StepStatus::Done);
    CHECK(c.t == c.t_end);
    CHECK(c.n_reject >= 1);
    CHECK(std::isfinite(b.x[0]));

    ToyBody<step::TrBdf2> z;   // a step below h_min aborts and rolls back
    z.fail_next_take = true;
    step::LteClock<step::TrBdf2> cz;
    cz.t_end = 1;
    cz.h_min = 0.5;
    cz.ctrl.h = 1;
    CHECK(step::step_once<step::TrBdf2>(cz, z) == step::StepStatus::Aborted);
    CHECK(z.n_restore == 1);
    CHECK(z.x == V2{1, 0});
}

TEST_CASE("Stepper, LTE clock: a rejected step is rolled back and re-solved",
          "[step][stepper]") {
    ToyBody<step::TrBdf2> b;
    step::LteClock<step::TrBdf2> c;
    c.t_end = 1;
    c.h_min = 1e-12;
    c.h_max = 1;
    c.ctrl.h = 1;
    REQUIRE(step::step_once<step::TrBdf2>(c, b) == step::StepStatus::Accepted);
    // save → take → error, rejected → restore, ... → after → commit
    REQUIRE(b.log.size() > 8);
    CHECK(b.log[2] == "save");
    CHECK(b.log[3] == "take");
    CHECK(b.log[6] == "error");
    CHECK(b.log[7] == "restore");
    CHECK(b.log.back() == "on_step");
}

namespace {
struct LocalizingBody : ToyBody<step::Trapezoidal> {
    static constexpr bool localizes = true;
};
}  // namespace

TEST_CASE("Stepper: the event policy truncates the step to a located event",
          "[step][stepper]") {
    ToyBody<step::TrBdf2> b;
    b.rtol = 1;   // accept anything: only the event cuts the step
    step::LteClock<step::TrBdf2> c;
    c.t_end = 1;
    c.h_min = 1e-12;
    c.ctrl.h = 0.0009715043441930749;   // t0 != 0 for the cut step
    REQUIRE(step::step_once<step::TrBdf2>(c, b) == step::StepStatus::Accepted);
    const Real t0 = c.t;
    c.ctrl.h = 0.02;
    b.event_t = t0 + 0.0123;
    REQUIRE(step::step_once<step::TrBdf2>(c, b) == step::StepStatus::Accepted);
    CHECK(c.t == b.event_t);   // lands on the event, not t0 + (e - t0)
    CHECK(b.n_restore == 1);
    // the next proposal is the full attempt's (5 x 0.02), not the sliver's
    CHECK(c.ctrl.h > 0.09);

    // fixed clock: the cut sub-step is committed, then the rest of the
    // grid interval; k advances only at the grid point
    LocalizingBody bf;
    bf.event_t = 0.004;
    step::FixedController cf{{0, 0.01}, 1, 3};
    REQUIRE(step::step_once<step::Trapezoidal>(cf, bf) == step::StepStatus::Accepted);
    CHECK(bf.n_restore == 1);
    CHECK(cf.k == 1);
    CHECK(cf.t_in == 0.004);
    REQUIRE(step::step_once<step::Trapezoidal>(cf, bf) == step::StepStatus::Accepted);
    CHECK(cf.k == 2);
    const V2 ex = exact(0.01);
    CHECK(std::hypot(bf.x[0] - ex[0], bf.x[1] - ex[1]) < 1e-4);
}

TEST_CASE("Dense output: the cubic Hermite is exact for a cubic", "[step]") {
    const auto p = [](Real t) { return 2 * t * t * t - t * t + 3 * t - 1; };
    const auto dp = [](Real t) { return 6 * t * t - 2 * t + 3; };
    const Real t0 = 0.5, h = 0.25;
    for (const Real th : {0.0, 0.3, 0.5, 0.9, 1.0}) {
        const Real v = step::hermite(p(t0), dp(t0), p(t0 + h), dp(t0 + h), h, th);
        CHECK(std::abs(v - p(t0 + th * h)) < 1e-14);
    }
}

TEST_CASE("Integrators: err_exponent and LteController's laws agree",
          "[step]") {
    STATIC_CHECK(step::Integrator<step::Trapezoidal>);
    STATIC_CHECK(step::Integrator<step::BackwardEuler>);
    STATIC_CHECK(step::Integrator<step::TrBdf2>);
    STATIC_CHECK(step::Integrator<step::Sdirk22>);
    for (const Real e : {1e-6, 0.3, 2.0, 40.0}) {
        CHECK(step::LteController::reject_factor(e, true)
              == step::LteController::reject_factor(e, step::Sdirk22::err_exponent));
        CHECK(step::LteController::accept_factor(e, false)
              == step::LteController::accept_factor(e, step::TrBdf2::err_exponent));
    }
    // Sdirk22: stage weight (1 − γ)/γ, stiffly accurate (last stage ends at h)
    const auto s1 = step::Sdirk22::stage(0.1, 1);
    CHECK(s1.t_off == 0.1);
    CHECK(s1.factor_dt == step::Sdirk22::stage(0.1, 0).factor_dt);
}
