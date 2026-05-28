// =============================================================================
// PED Gate 2 — Illinois root-finder + EventPredictor unit tests
// =============================================================================

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/event_predictor.hpp"

using pulsim::Real;
using pulsim::Vector;
using namespace pulsim::dsed;

TEST_CASE("illinois finds root of a linear function", "[dsed][illinois]") {
    // g(t) = t - 0.5 → root at t = 0.5
    auto g = [](Real t) -> Real { return t - Real{0.5}; };
    const Real root = illinois(g, Real{0}, Real{1});
    REQUIRE(root == Catch::Approx(Real{0.5}).epsilon(Real{1e-10}));
}

TEST_CASE("illinois finds root of a quadratic", "[dsed][illinois]") {
    // g(t) = (t - 0.3)·(t - 1.8) → roots at 0.3 and 1.8;
    // on [0, 1] only 0.3 lies inside
    auto g = [](Real t) -> Real {
        return (t - Real{0.3}) * (t - Real{1.8});
    };
    const Real root = illinois(g, Real{0}, Real{1});
    REQUIRE(root == Catch::Approx(Real{0.3}).epsilon(Real{1e-9}));
}

TEST_CASE("illinois handles a near-tangent crossing without stalling",
           "[dsed][illinois]") {
    // g(t) = (t - 0.5)^3 — has triple root at 0.5, slope 0 there
    // Plain regula-falsi stalls; Illinois weighting saves it.
    auto g = [](Real t) -> Real {
        const Real d = t - Real{0.5};
        return d * d * d;
    };
    const Real root = illinois(g, Real{0}, Real{1}, std::nullopt, std::nullopt,
                                  Real{1e-7}, 100);
    REQUIRE(root == Catch::Approx(Real{0.5}).epsilon(Real{1e-2}));
}

TEST_CASE("illinois throws if root not bracketed", "[dsed][illinois]") {
    auto g = [](Real t) -> Real { return t + Real{1}; };  // > 0 everywhere on [0,1]
    REQUIRE_THROWS_AS(illinois(g, Real{0}, Real{1}), std::invalid_argument);
}

TEST_CASE("bisect_fallback agrees with illinois on linear g",
           "[dsed][bisect]") {
    auto g = [](Real t) -> Real { return Real{2} * t - Real{1}; };
    const Real r1 = illinois(g, Real{0}, Real{1});
    const Real r2 = bisect_fallback(g, Real{0}, Real{1});
    REQUIRE(r1 == Catch::Approx(r2).epsilon(Real{1e-8}));
}

TEST_CASE("EventPredictor finds soonest event in horizon",
           "[dsed][predictor]") {
    EventPredictor pred;

    // Predicate 1: fires at t = 0.3
    pred.register_predicate(
        "early", [](Real t, const Vector&) { return t - Real{0.3}; },
        PredicateType::GateEdge);
    // Predicate 2: fires at t = 0.7
    pred.register_predicate(
        "late", [](Real t, const Vector&) { return t - Real{0.7}; },
        PredicateType::DiodeOn);

    Vector x_now(1);
    x_now << Real{0};
    auto interp = [](Real /*t*/) -> Vector {
        Vector v(1);
        v << Real{0};
        return v;
    };

    auto [t_event, idx] = pred.predict_next(
        Real{0}, x_now, interp, Real{1.0});
    REQUIRE(t_event == Catch::Approx(Real{0.3}).epsilon(Real{1e-9}));
    REQUIRE(idx == 0);  // "early" wins
}

TEST_CASE("EventPredictor priority breaks simultaneous-event ties",
           "[dsed][predictor]") {
    EventPredictor pred;

    // Both fire at t = 0.5 — exactly simultaneous
    pred.register_predicate(
        "g1", [](Real t, const Vector&) { return t - Real{0.5}; },
        PredicateType::DiodeOn);     // priority 1
    pred.register_predicate(
        "g0", [](Real t, const Vector&) { return t - Real{0.5}; },
        PredicateType::GateEdge);    // priority 0 (wins on tie)

    Vector x(1);
    x << Real{0};
    auto interp = [](Real /*t*/) -> Vector {
        Vector v(1);
        v << Real{0};
        return v;
    };
    auto [t_event, idx] = pred.predict_next(Real{0}, x, interp, Real{1});
    REQUIRE(t_event == Catch::Approx(Real{0.5}).epsilon(Real{1e-10}));
    REQUIRE(idx == 1);  // "g0" (GateEdge, lower priority) wins
}

TEST_CASE("EventPredictor returns -1 idx when no predicates fire",
           "[dsed][predictor]") {
    EventPredictor pred;
    pred.register_predicate(
        "never", [](Real /*t*/, const Vector&) { return Real{1}; },  // > 0 always
        PredicateType::Custom);

    Vector x(1);
    x << Real{0};
    auto interp = [](Real /*t*/) -> Vector {
        Vector v(1);
        v << Real{0};
        return v;
    };
    auto [t_event, idx] = pred.predict_next(Real{0}, x, interp, Real{1});
    REQUIRE(idx == -1);
    REQUIRE(t_event == Real{1});  // t_max
}
