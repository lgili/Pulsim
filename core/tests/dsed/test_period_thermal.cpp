// =============================================================================
// DSED -- the junction fixed point on a periodic orbit
// =============================================================================
//
// add-design-optimization-workflow DECISIONS 28
// (pulsim/dsed/period_thermal.hpp). Oracle: one commutation per period
// with E(T) = E0 (1 + a (T - 25)) gives P(T) = P0 (1 + a (T - 25)) and
// T* = (T_amb + R P0 (1 - 25 a)) / (1 - R P0 a); no root above T_amb
// when R P0 a >= 1 (runaway, reported with R dP/dT). Synthetic tables.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/period_thermal.hpp"

#include <cmath>
#include <stdexcept>

namespace pt = pulsim::dsed::period_thermal;
using pulsim::thermal::ConductionTable;
using pulsim::thermal::LossTable3D;

TEST_CASE("Period thermal: the junction fixed point and its runaway",
          "[v2][dsed][period_thermal]") {
    const double E0 = 1e-3, a = 0.004, T = 1e-4;   // P0 = 10 W
    const LossTable3D tab({600.0}, {10.0}, {25.0, 125.0},
                          {E0, E0 * (1 + a * 100)});
    pt::DevicePower d;
    d.on_pts = {{600.0, 10.0}};
    d.on = &tab;
    d.period = T;
    const double P0 = E0 / T, t_amb = 40.0;
    for (double R : {2.0, 20.0, 0.9 / (P0 * a)}) {
        const double want = (t_amb + R * P0 * (1 - 25 * a)) / (1 - R * P0 * a);
        const auto j = pt::solve_junction(d, R, t_amb);
        REQUIRE_FALSE(j.runaway);
        REQUIRE(std::abs(j.tj - want) < 1e-9 * std::max(1.0, want));
        REQUIRE(std::abs(j.residual) <= 1e-9);
        REQUIRE(std::abs(j.slope - R * P0 * a) < 1e-9);
    }
    const auto j = pt::solve_junction(d, 1.2 / (P0 * a), t_amb);
    REQUIRE(j.runaway);
    REQUIRE(std::abs(j.slope - 1.2) < 1e-9);
    // no table: T* = T_amb + R P exactly, P fixed
    pt::DevicePower c;
    c.p_fixed = 3.0;
    REQUIRE(pt::solve_junction(c, 2.0, 25.0).tj == 31.0);
    REQUIRE_THROWS_AS(pt::solve_junction(c, -1.0, 25.0), std::invalid_argument);
}

TEST_CASE("Period thermal: a linear conduction table in closed form",
          "[v2][dsed][period_thermal]") {
    // V_on = (1 + 0.01 i)(1 + 0.002 (T - 25)) on i = [0, 10], tj = [25, 125]
    const ConductionTable t({0.0, 10.0}, {25.0, 125.0}, {1.0, 1.2, 1.1, 1.32});
    const double q1 = 3e-4, q2 = 2.5e-3, T = 1e-4;
    for (double tj : {25.0, 75.0, 175.0}) {
        const double k = 1 + 0.002 * (tj - 25);
        const double want = k * (q1 + 0.01 * q2) / T;
        REQUIRE(std::abs(pt::conduction_power(t, q1, q2, 20.0, tj, T) - want) <
                1e-12 * want);
    }
    const ConductionTable three({0.0, 5.0, 10.0}, {25.0}, {1.0, 1.05, 1.1});
    REQUIRE_THROWS_AS(pt::conduction_power(three, q1, q2, 10.0, 25.0, T),
                      std::invalid_argument);
    // falls through 0 at i = 100 A: clamped inside [0, 200 A], refused
    const ConductionTable falls({0.0, 10.0}, {25.0}, {1.0, 0.9});
    REQUIRE_NOTHROW(pt::conduction_power(falls, q1, q2, 50.0, 25.0, T));
    REQUIRE_THROWS_AS(pt::conduction_power(falls, q1, q2, 200.0, 25.0, T),
                      std::invalid_argument);
}
