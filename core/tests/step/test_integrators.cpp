// refactor-unified-transient-stepper task 4.3: the integrators as stage
// tables (core/include/pulsim/solver/step/integrators.hpp).

#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "pulsim/solver/step/integrators.hpp"

using pulsim::Real;
using pulsim::pwl::TrBdf2Stage;
namespace step = pulsim::step;

TEST_CASE("TR-BDF2: both stages solve on the factor of gamma*h", "[step]") {
    for (const Real h : {1e-9, 3.7e-6, 1e-3}) {
        const auto s0 = step::TrBdf2::stage(h, 0);
        const auto s1 = step::TrBdf2::stage(h, 1);
        CHECK(s0.factor_dt == s1.factor_dt);
        CHECK(s0.factor_dt == pulsim::pwl::HistoryState::trbdf2_gamma() * h);
        CHECK(s0.formula == TrBdf2Stage::Trapezoidal);
        CHECK(s1.formula == TrBdf2Stage::Bdf2Stage2);
        // the stage-2 devices are told the full step (c1/h == 2/(gamma h))
        CHECK(s0.ctx_h == s0.factor_dt);
        CHECK(s1.ctx_h == h);
        CHECK(s0.t_off < s1.t_off);
        CHECK(s1.t_off == h);   // the last stage ends the step
    }
}

TEST_CASE("Backward Euler over h solves on the trapezoidal factor of 2h",
          "[step]") {
    const Real h = 2.5e-7;
    const auto s = step::BackwardEuler::stage(h * Real{0.5});
    CHECK(s.factor_dt == h);
    CHECK(s.ctx_h == h * Real{0.5});
    CHECK(s.t_off == h * Real{0.5});   // the elapsed time is h/2, not h
    CHECK(s.formula == TrBdf2Stage::BackwardEuler);
    // the fixed engine's restart stage, keyed by its factor
    const auto r = step::BackwardEuler::on_factor(h);
    CHECK(r.factor_dt == s.factor_dt);
    CHECK(r.ctx_h == s.ctx_h);
}

TEST_CASE("Trapezoidal: one stage, factor = interval = h", "[step]") {
    const auto s = step::Trapezoidal::stage(1e-7);
    CHECK(s.factor_dt == 1e-7);
    CHECK(s.ctx_h == 1e-7);
    CHECK(s.t_off == 1e-7);
    const pulsim::pwl::StepCtx c = s.ctx();
    CHECK(c.h == 1e-7);
    CHECK(c.stage == TrBdf2Stage::Trapezoidal);
    STATIC_CHECK(step::Trapezoidal::n_stages == 1);
    STATIC_CHECK(step::TrBdf2::n_stages == 2);
    STATIC_CHECK(step::BackwardEuler::order == 1);
}
