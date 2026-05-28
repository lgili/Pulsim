// =============================================================================
// PED Gate 2 — DOPRI5 (RK45) unit tests
// =============================================================================

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/rk45_dormand_prince.hpp"

using pulsim::Real;
using pulsim::Vector;
using namespace pulsim::dsed;

TEST_CASE("DOPRI5 integrates x' = x exactly to within 5th-order error",
           "[dsed][rk45]") {
    // ODE: x' = x, x(0) = 1 → analytical x(t) = e^t
    auto f = [](Real /*t*/, const Vector& x) -> Vector { return x; };

    Vector x(1);
    x << Real{1};
    Real t = Real{0};
    const Real h = Real{0.01};
    RK45State s;

    // Take 100 steps to reach t = 1
    for (int i = 0; i < 100; ++i) {
        auto [x_new, err] = step(f, t, x, h, s);
        x = x_new;
        t += h;
    }
    // Compare to e^1 ≈ 2.718281828
    const Real expected = std::exp(Real{1});
    REQUIRE(x(0) == Catch::Approx(expected).epsilon(Real{1e-9}));
}

TEST_CASE("DOPRI5 FSAL reuses k7 → 6 RHS evals per accepted step",
           "[dsed][rk45][fsal]") {
    // Count RHS evaluations
    int n_evals = 0;
    auto f = [&n_evals](Real /*t*/, const Vector& x) -> Vector {
        ++n_evals;
        return x;
    };
    Vector x(1);
    x << Real{1};
    RK45State s;

    // First step: 7 evals (no FSAL available)
    auto [x1, err1] = step(f, Real{0}, x, Real{0.01}, s);
    REQUIRE(n_evals == 7);
    REQUIRE(s.fsal_valid());

    // Second step: 6 evals (k1 reused from k7)
    auto [x2, err2] = step(f, Real{0.01}, x1, Real{0.01}, s);
    REQUIRE(n_evals == 7 + 6);
}

TEST_CASE("DOPRI5 invalidating FSAL forces k1 recompute",
           "[dsed][rk45][fsal]") {
    int n_evals = 0;
    auto f = [&n_evals](Real /*t*/, const Vector& x) -> Vector {
        ++n_evals;
        return x;
    };
    Vector x(1);
    x << Real{1};
    RK45State s;

    auto [x1, err1] = step(f, Real{0}, x, Real{0.01}, s);
    REQUIRE(n_evals == 7);

    s.invalidate();
    REQUIRE_FALSE(s.fsal_valid());

    auto [x2, err2] = step(f, Real{0.01}, x1, Real{0.01}, s);
    REQUIRE(n_evals == 7 + 7);  // FSAL was invalidated → 7 evals again
}

TEST_CASE("DOPRI5 embedded error estimate is small for smooth ODE",
           "[dsed][rk45]") {
    // x' = -10·x (linear damping); local truncation error scales as h^5
    auto f = [](Real /*t*/, const Vector& x) -> Vector {
        return Real{-10} * x;
    };
    Vector x(1);
    x << Real{1};
    RK45State s;

    auto [x_new, err] = step(f, Real{0}, x, Real{0.001}, s);
    // |err| should be tiny (well below 1e-6 for h=1e-3 on smooth ODE)
    REQUIRE(std::abs(err(0)) < Real{1e-10});
}

TEST_CASE("DOPRI5 advances x' = A·x linearly for a 2x2 buck-like system",
           "[dsed][rk45][buck]") {
    // Mini buck: state = (i_L, v_C), constant matrix while in HS-on:
    //   di_L/dt = (V_in - v_C) / L
    //   dv_C/dt = (i_L - v_C/R) / C
    constexpr Real V_in = Real{24};
    constexpr Real L = Real{100e-6};
    constexpr Real C = Real{100e-6};
    constexpr Real R = Real{2.4};

    auto f = [](Real /*t*/, const Vector& x) -> Vector {
        Vector dx(2);
        dx(0) = (V_in - x(1)) / L;
        dx(1) = (x(0) - x(1) / R) / C;
        return dx;
    };
    Vector x(2);
    x << Real{5}, Real{12};  // CCM steady state
    RK45State s;

    // Take one 100 ns step
    auto [x_new, err] = step(f, Real{0}, x, Real{100e-9}, s);

    // i_L should rise (driving voltage > v_C/2 → net L·di/dt > 0)
    REQUIRE(x_new(0) > x(0));
    // Error should be small for a smooth interval
    REQUIRE(std::abs(err(0)) < Real{1e-6});
    REQUIRE(std::abs(err(1)) < Real{1e-6});
}
