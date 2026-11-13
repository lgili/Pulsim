// add-in-kernel-event-losses 1.1/1.3 — kernel loss tables. The Python
// side (python/tests/test_loss_table_kernel.py) checks 10 000 random
// points against LossTable.__call__; here: the closed-form temperature
// table, extrapolation on every axis, the clamp, and the refusals.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/thermal/loss_table.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using pulsim::thermal::ConductionTable;
using pulsim::thermal::LossTable3D;

namespace {
bool near(double a, double b, double rel = 1e-12) {
    return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b));
}
}  // namespace

TEST_CASE("LossTable3D: E = 1 mJ (1 + 0.01 (T - 25)) inside and past the grid",
          "[thermal][loss_table]") {
    // single-point v and i axes: those dimensions collapse
    const LossTable3D t({300.0}, {10.0}, {25.0, 125.0}, {1e-3, 2e-3});
    CHECK(near(t(300.0, 10.0, 75.0), 1.5e-3));
    CHECK(near(t(300.0, 10.0, 175.0), 2.5e-3));   // linear extrapolation
    CHECK(near(t(900.0, -4.0, 25.0), 1e-3));       // collapsed axes ignore q
    CHECK(t(300.0, 10.0, -100.0) == 0.0);          // -0.25 mJ clamped at 0
}

TEST_CASE("LossTable3D: trilinear, extrapolated on every axis",
          "[thermal][loss_table]") {
    // E = 1 + 2v + 3i + 5t + v i t is multilinear, so the table reproduces
    // it exactly everywhere, extrapolation included
    const std::vector<double> v{0.0, 1.0, 3.0}, i{-1.0, 2.0}, tj{0.0, 4.0};
    auto f = [](double a, double b, double c) {
        return 1 + 2 * a + 3 * b + 5 * c + a * b * c + 100;
    };
    std::vector<double> e;
    for (double a : v)
        for (double b : i)
            for (double c : tj) e.push_back(f(a, b, c));
    const LossTable3D t(v, i, tj, e);
    for (double a : {-2.0, 0.5, 2.0, 7.0})
        for (double b : {-3.0, 0.0, 5.0})
            for (double c : {-1.0, 2.0, 9.0}) {
                const double want = std::max(f(a, b, c), 0.0);
                CHECK(std::abs(t(a, b, c) - want) <= 1e-12 * (1 + std::abs(want)));
            }
}

TEST_CASE("ConductionTable: V_on = 1 + 0.002 (T - 25) at 10 A",
          "[thermal][loss_table]") {
    const ConductionTable c({0.0, 10.0}, {25.0, 125.0},
                            {0.5, 0.7, 1.0, 1.2});
    CHECK(near(c(10.0, 75.0), 1.1));
    CHECK(near(c(5.0, 25.0), 0.75));
    CHECK(near(c(20.0, 25.0), 1.5));               // extrapolated in i
    CHECK(c(-100.0, 25.0) == 0.0);                 // clamp
}

TEST_CASE("Loss tables refuse bad input by name", "[thermal][loss_table]") {
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(LossTable3D({}, {1.0}, {1.0}, {}),
                      ContainsSubstring("v axis is empty"));
    CHECK_THROWS_WITH(LossTable3D({1.0}, {2.0, 1.0}, {1.0}, {1, 1}),
                      ContainsSubstring("i axis must be strictly ascending"));
    CHECK_THROWS_WITH(LossTable3D({1.0}, {1.0}, {1.0, NAN}, {1, 1}),
                      ContainsSubstring("tj axis has a non-finite value"));
    CHECK_THROWS_WITH(LossTable3D({1.0}, {1.0}, {1.0, 2.0}, {1.0}),
                      ContainsSubstring("energy has 1 entries"));
    CHECK_THROWS_WITH(LossTable3D({1.0}, {1.0}, {1.0}, {-1.0}),
                      ContainsSubstring("energy must be >= 0"));
    CHECK_THROWS_WITH(ConductionTable({1.0}, {1.0}, {INFINITY}),
                      ContainsSubstring("v_on has non-finite"));
    const LossTable3D t({1.0}, {1.0}, {1.0}, {1.0});
    CHECK_THROWS_WITH(t(1.0, NAN, 1.0), ContainsSubstring("i query"));
}
