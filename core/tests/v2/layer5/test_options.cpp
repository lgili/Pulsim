// =============================================================================
// Layer 5 — SimulationOptions tests
// =============================================================================
//
// Pure value-type tests. No solver, no cache. Covers the
// valid()/expected_step_count() self-checks that `run_transient`
// relies on for input validation.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/v2/solver/options.hpp"

#include <cmath>
#include <limits>

using namespace pulsim::v2;
using namespace pulsim::v2::solver;

TEST_CASE("SimulationOptions: default-constructed is invalid",
          "[v2][layer5][options]") {
    SimulationOptions opts{};
    REQUIRE_FALSE(opts.valid());
    // expected_step_count is 0 when invalid (defensive).
    REQUIRE(opts.expected_step_count() == 0);
}

TEST_CASE("SimulationOptions: t_start=0, t_end=1, dt=0.1 → 11 samples",
          "[v2][layer5][options]") {
    SimulationOptions opts{
        .t_start = Real{0},
        .t_end   = Real{1},
        .dt      = Real{0.1},
    };
    REQUIRE(opts.valid());
    // Samples at t = 0.0, 0.1, …, 1.0  ⇒ 11 samples.
    REQUIRE(opts.expected_step_count() == 11);
}

TEST_CASE("SimulationOptions: negative dt is invalid",
          "[v2][layer5][options]") {
    SimulationOptions opts{
        .t_start = Real{0},
        .t_end   = Real{1},
        .dt      = Real{-0.01},
    };
    REQUIRE_FALSE(opts.valid());
}

TEST_CASE("SimulationOptions: zero dt is invalid",
          "[v2][layer5][options]") {
    SimulationOptions opts{
        .t_start = Real{0},
        .t_end   = Real{1},
        .dt      = Real{0},
    };
    REQUIRE_FALSE(opts.valid());
}

TEST_CASE("SimulationOptions: t_end <= t_start is invalid",
          "[v2][layer5][options]") {
    SECTION("t_end == t_start") {
        SimulationOptions opts{
            .t_start = Real{1},
            .t_end   = Real{1},
            .dt      = Real{0.1},
        };
        REQUIRE_FALSE(opts.valid());
    }
    SECTION("t_end < t_start") {
        SimulationOptions opts{
            .t_start = Real{2},
            .t_end   = Real{1},
            .dt      = Real{0.1},
        };
        REQUIRE_FALSE(opts.valid());
    }
}

TEST_CASE("SimulationOptions: NaN values are invalid",
          "[v2][layer5][options]") {
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    SECTION("NaN t_start") {
        SimulationOptions opts{
            .t_start = nan, .t_end = Real{1}, .dt = Real{0.1}};
        REQUIRE_FALSE(opts.valid());
    }
    SECTION("NaN t_end") {
        SimulationOptions opts{
            .t_start = Real{0}, .t_end = nan, .dt = Real{0.1}};
        REQUIRE_FALSE(opts.valid());
    }
    SECTION("NaN dt") {
        SimulationOptions opts{
            .t_start = Real{0}, .t_end = Real{1}, .dt = nan};
        REQUIRE_FALSE(opts.valid());
    }
}

TEST_CASE("SimulationOptions: infinity is invalid",
          "[v2][layer5][options]") {
    const Real inf = std::numeric_limits<Real>::infinity();
    SimulationOptions opts{
        .t_start = Real{0}, .t_end = inf, .dt = Real{0.1}};
    REQUIRE_FALSE(opts.valid());
}

TEST_CASE("SimulationOptions: 1 ms / 1 µs PWM sim → 1001 samples",
          "[v2][layer5][options]") {
    // The chopper-PWM integration test uses these exact values.
    // Confirm the expected count up-front.
    SimulationOptions opts{
        .t_start = Real{0},
        .t_end   = Real{1e-3},
        .dt      = Real{1e-6},
    };
    REQUIRE(opts.valid());
    REQUIRE(opts.expected_step_count() == 1001);
}
