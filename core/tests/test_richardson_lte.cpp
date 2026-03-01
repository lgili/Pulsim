#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/v1/integration.hpp"
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// =============================================================================
// SolutionHistory Tests
// =============================================================================

TEST_CASE("SolutionHistory - Basic operations", "[richardson][history]") {
    SolutionHistory history(5);

    SECTION("Empty history") {
        REQUIRE(history.empty());
        REQUIRE(history.size() == 0);
        REQUIRE_FALSE(history.has_sufficient_history(2));
    }

    SECTION("Adding entries") {
        Vector x1(3);
        x1 << 1.0, 2.0, 3.0;
        history.push(x1, 0.0, 1e-6);

        REQUIRE_FALSE(history.empty());
        REQUIRE(history.size() == 1);
        REQUIRE(history[0].time == 0.0);
        REQUIRE(history[0].timestep == 1e-6);
        REQUIRE(history.most_recent_time() == 0.0);
    }

    SECTION("Ring buffer behavior") {
        for (int i = 0; i < 10; ++i) {
            Vector x(2);
            x << static_cast<Real>(i), static_cast<Real>(i * 2);
            history.push(x, i * 1e-6, 1e-6);
        }

        // Should only keep last 5 entries
        REQUIRE(history.size() == 5);

        // Most recent should be i=9
        REQUIRE(history[0].solution[0] == 9.0);

        // Oldest should be i=5
        REQUIRE(history[4].solution[0] == 5.0);
    }

    SECTION("Sufficient history check") {
        Vector x(2);
        x << 1.0, 2.0;

        history.push(x, 0.0, 1e-6);
        REQUIRE_FALSE(history.has_sufficient_history(2));  // Need at least 2 entries

        history.push(x, 1e-6, 1e-6);
        REQUIRE(history.has_sufficient_history(2));  // 2 entries sufficient for linear extrapolation

        history.push(x, 2e-6, 1e-6);
        REQUIRE(history.has_sufficient_history(2));  // 3 entries even better (quadratic)
    }
}

// =============================================================================
// RichardsonLTE Tests
// =============================================================================

TEST_CASE("RichardsonLTE - Linear extrapolation", "[richardson][lte]") {
    SolutionHistory history(5);

    // Create a linear sequence: x(t) = t
    // At t=0: x=0, at t=1e-6: x=1e-6, at t=2e-6: x=2e-6
    Vector x0(1), x1(1), x2(1);
    x0 << 0.0;
    x1 << 1e-6;
    x2 << 2e-6;

    history.push(x0, 0.0, 1e-6);
    history.push(x1, 1e-6, 1e-6);

    // Current solution at t=2e-6 should be perfectly predicted
    Vector current(1);
    current << 2e-6;

    Real lte = RichardsonLTE::compute(current, history, 2);

    // LTE should be very small since linear extrapolation is exact for linear data
    REQUIRE(lte < 1e-10);
}

TEST_CASE("RichardsonLTE - Quadratic extrapolation", "[richardson][lte]") {
    SolutionHistory history(5);

    // Create a quadratic sequence: x(t) = t^2
    // At t=0: x=0, at t=1: x=1, at t=2: x=4
    Vector x0(1), x1(1), x2(1);
    x0 << 0.0;
    x1 << 1.0;
    x2 << 4.0;

    // Push in reverse chronological order (most recent last to push)
    history.push(x0, 0.0, 1.0);
    history.push(x1, 1.0, 1.0);
    history.push(x2, 2.0, 1.0);

    // Current solution at t=3: x(3) = 9
    Vector current(1);
    current << 9.0;

    // With quadratic extrapolation from 3 points, prediction should be good
    Real lte = RichardsonLTE::compute(current, history, 2);

    // Quadratic extrapolation of quadratic data should be nearly exact
    REQUIRE(lte < 0.1);  // Allow some numerical tolerance
}

TEST_CASE("RichardsonLTE - Insufficient history", "[richardson][lte]") {
    SolutionHistory history(5);

    Vector x(2);
    x << 1.0, 2.0;

    // Only one entry
    history.push(x, 0.0, 1e-6);

    Vector current = x * 2.0;
    Real lte = RichardsonLTE::compute(current, history, 2);

    // Should return -1 for insufficient history
    REQUIRE(lte == -1.0);
}

TEST_CASE("RichardsonLTE - Per-variable computation", "[richardson][lte]") {
    SolutionHistory history(5);

    // Two variables with different behaviors
    Vector x0(2), x1(2);
    x0 << 1.0, 10.0;
    x1 << 2.0, 20.0;

    history.push(x0, 0.0, 1e-6);
    history.push(x1, 1e-6, 1e-6);

    // Linear trend continues
    Vector current(2);
    current << 3.0, 30.0;

    Vector lte_per_var = RichardsonLTE::compute_per_variable(current, history, 2);

    // Both should have small LTE since extrapolation is linear
    REQUIRE(lte_per_var.size() == 2);
    REQUIRE(lte_per_var[0] < 1e-6);
    REQUIRE(lte_per_var[1] < 1e-6);
}

TEST_CASE("RichardsonLTE - Weighted computation", "[richardson][lte]") {
    SolutionHistory history(5);

    Vector x0(4), x1(4);
    // 2 voltage nodes, 2 current branches
    x0 << 1.0, 2.0, 0.001, 0.002;  // Voltages ~1V, currents ~1mA
    x1 << 1.1, 2.1, 0.0011, 0.0021;

    history.push(x0, 0.0, 1e-6);
    history.push(x1, 1e-6, 1e-6);

    // Current solution with some deviation
    Vector current(4);
    current << 1.2, 2.2, 0.0012, 0.0022;

    Real lte = RichardsonLTE::compute_weighted(
        current, history,
        2,    // num_nodes (voltages)
        2,    // num_branches (currents)
        1e-3, // voltage tolerance
        1e-6, // current tolerance
        2     // order
    );

    REQUIRE(lte >= 0.0);
}

// =============================================================================
// AdaptiveLTEEstimator Tests
// =============================================================================

TEST_CASE("AdaptiveLTEEstimator - Richardson mode", "[richardson][adaptive]") {
    RichardsonLTEConfig config;
    config.method = TimestepMethod::Richardson;
    config.extrapolation_order = 2;
    config.history_depth = 5;

    AdaptiveLTEEstimator estimator(config);

    REQUIRE(estimator.is_richardson());
    REQUIRE_FALSE(estimator.has_sufficient_history());

    // Record solutions
    Vector x0(2), x1(2), x2(2);
    x0 << 0.0, 0.0;
    x1 << 1.0, 2.0;
    x2 << 2.0, 4.0;

    estimator.record_solution(x0, 0.0, 1e-6);
    estimator.record_solution(x1, 1e-6, 1e-6);
    estimator.record_solution(x2, 2e-6, 1e-6);

    REQUIRE(estimator.has_sufficient_history());
    REQUIRE(estimator.history_size() == 3);

    // Compute LTE
    Vector current(2);
    current << 3.0, 6.0;  // Linear continuation

    Real lte = estimator.compute(current);

    // Should be small for linear data
    REQUIRE(lte >= 0.0);
    REQUIRE(lte < 1e-5);
}

TEST_CASE("AdaptiveLTEEstimator - Step doubling mode", "[richardson][adaptive]") {
    RichardsonLTEConfig config = RichardsonLTEConfig::step_doubling();

    AdaptiveLTEEstimator estimator(config);

    REQUIRE_FALSE(estimator.is_richardson());

    // In step-doubling mode, compute() should return -1
    Vector current(2);
    current << 1.0, 2.0;

    Real lte = estimator.compute(current);

    REQUIRE(lte == -1.0);  // Indicates caller should use step-doubling
}

TEST_CASE("AdaptiveLTEEstimator - Reset", "[richardson][adaptive]") {
    AdaptiveLTEEstimator estimator;

    Vector x(2);
    x << 1.0, 2.0;

    estimator.record_solution(x, 0.0, 1e-6);
    estimator.record_solution(x, 1e-6, 1e-6);
    estimator.record_solution(x, 2e-6, 1e-6);

    REQUIRE(estimator.history_size() == 3);

    estimator.reset();

    REQUIRE(estimator.history_size() == 0);
    REQUIRE_FALSE(estimator.has_sufficient_history());
}

// =============================================================================
// TimestepMethod Tests
// =============================================================================

TEST_CASE("TimestepMethod enum", "[richardson][enum]") {
    REQUIRE(static_cast<int>(TimestepMethod::StepDoubling) == 0);
    REQUIRE(static_cast<int>(TimestepMethod::Richardson) == 1);
}

TEST_CASE("RichardsonLTEConfig defaults", "[richardson][config]") {
    auto config = RichardsonLTEConfig::defaults();

    REQUIRE(config.method == TimestepMethod::Richardson);
    REQUIRE(config.extrapolation_order == 2);
    REQUIRE(config.voltage_tolerance == 1e-3);
    REQUIRE(config.current_tolerance == 1e-6);
    REQUIRE(config.use_weighted_norm == true);
    REQUIRE(config.history_depth == 5);
}

TEST_CASE("RichardsonLTEConfig step_doubling factory", "[richardson][config]") {
    auto config = RichardsonLTEConfig::step_doubling();

    REQUIRE(config.method == TimestepMethod::StepDoubling);
}

// =============================================================================
// Performance comparison sketch (not actual benchmark)
// =============================================================================

TEST_CASE("Richardson vs StepDoubling - concept verification", "[richardson][concept]") {
    // This test verifies the concept that Richardson uses history
    // while step-doubling would need additional solves

    SolutionHistory history(5);

    // Simulate 10 timesteps, recording each
    for (int i = 0; i < 10; ++i) {
        Vector x(3);
        x << static_cast<Real>(i), std::sin(i * 0.1), std::cos(i * 0.1);
        history.push(x, i * 1e-6, 1e-6);
    }

    // At step 11, we can compute LTE without any additional solves
    Vector x_current(3);
    x_current << 10.0, std::sin(10 * 0.1), std::cos(10 * 0.1);

    Real lte = RichardsonLTE::compute(x_current, history, 2);

    // LTE should be computable
    REQUIRE(lte >= 0.0);

    // In contrast, step-doubling would require:
    // 1. One full step solve
    // 2. Two half-step solves
    // Total: 3x computational work per timestep
    // Richardson: 0x additional work (just uses history)
}

// =============================================================================
// AdvancedTimestepController Tests (Task 7)
// =============================================================================

TEST_CASE("AdvancedTimestepConfig - defaults and presets", "[timestep][config]") {
    SECTION("Default configuration") {
        auto config = AdvancedTimestepConfig::defaults();

        REQUIRE(config.target_newton_iterations == 5);
        REQUIRE(config.min_newton_iterations == 2);
        REQUIRE(config.max_newton_iterations == 15);
        REQUIRE(config.newton_feedback_gain == Catch::Approx(0.3));
        REQUIRE(config.max_growth_rate == Catch::Approx(2.0));
        REQUIRE(config.max_shrink_rate == Catch::Approx(0.25));
        REQUIRE(config.enable_smoothing == true);
        REQUIRE(config.lte_weight == Catch::Approx(0.7));
        REQUIRE(config.newton_weight == Catch::Approx(0.3));
    }

    SECTION("Switching preset") {
        auto config = AdvancedTimestepConfig::for_switching();

        REQUIRE(config.dt_min == Catch::Approx(1e-12));
        REQUIRE(config.dt_max == Catch::Approx(1e-5));
        REQUIRE(config.target_newton_iterations == 4);
        REQUIRE(config.max_growth_rate == Catch::Approx(1.5));
    }

    SECTION("Power electronics preset") {
        auto config = AdvancedTimestepConfig::for_power_electronics();

        REQUIRE(config.dt_min == Catch::Approx(1e-10));
        REQUIRE(config.newton_feedback_gain == Catch::Approx(0.4));
        REQUIRE(config.error_tolerance == Catch::Approx(1e-3));
    }
}

TEST_CASE("AdvancedTimestepController - LTE-based adjustment", "[timestep][lte]") {
    AdvancedTimestepConfig config;
    config.dt_initial = 1e-6;
    config.error_tolerance = 1e-4;

    AdvancedTimestepController controller(config);

    SECTION("Low error - timestep should increase") {
        Real lte = 1e-6;  // Much smaller than tolerance
        auto decision = controller.compute(lte);

        REQUIRE(decision.accepted);
        REQUIRE(decision.dt_new > config.dt_initial);
        REQUIRE(decision.error_ratio < 1.0);
    }

    SECTION("High error - step should be rejected") {
        Real lte = 1e-2;  // Much larger than tolerance
        auto decision = controller.compute(lte);

        REQUIRE_FALSE(decision.accepted);
        REQUIRE(decision.dt_new < config.dt_initial);
        REQUIRE(decision.error_ratio > 1.0);
    }

    SECTION("Error at tolerance - timestep stays similar") {
        Real lte = 1e-4;  // Equal to tolerance
        auto decision = controller.compute(lte);

        REQUIRE(decision.accepted);
        // Timestep should stay roughly the same (with safety factor)
        REQUIRE(decision.dt_new >= config.dt_initial * 0.5);
        REQUIRE(decision.dt_new <= config.dt_initial * 1.5);
    }
}

TEST_CASE("AdvancedTimestepController - Newton iteration feedback", "[timestep][newton]") {
    AdvancedTimestepConfig config;
    config.dt_initial = 1e-6;
    config.error_tolerance = 1e-4;
    config.target_newton_iterations = 5;
    config.newton_feedback_gain = 0.3;

    AdvancedTimestepController controller(config);

    Real lte = 5e-5;  // Acceptable error

    SECTION("Fast convergence - Newton factor > 1") {
        auto decision = controller.compute_combined(lte, 2);  // Only 2 iterations

        REQUIRE(decision.accepted);
        REQUIRE(decision.newton_factor > 1.0);  // Should want to increase
    }

    SECTION("At target iterations - Newton factor ~ 1") {
        auto decision = controller.compute_combined(lte, 5);  // Exactly at target

        REQUIRE(decision.accepted);
        REQUIRE(decision.newton_factor == Catch::Approx(1.0).margin(0.1));
    }

    SECTION("Slow convergence - Newton factor < 1") {
        auto decision = controller.compute_combined(lte, 10);  // More than target

        REQUIRE(decision.accepted);
        REQUIRE(decision.newton_factor < 1.0);  // Should want to decrease
    }

    SECTION("Too many iterations - step rejected") {
        auto decision = controller.compute_combined(lte, 20);  // Way over max

        REQUIRE_FALSE(decision.accepted);
    }
}

TEST_CASE("AdvancedTimestepController - Rate-of-change limiting", "[timestep][smoothing]") {
    AdvancedTimestepConfig config;
    config.dt_initial = 1e-6;
    config.error_tolerance = 1e-4;
    config.max_growth_rate = 2.0;
    config.max_shrink_rate = 0.25;
    config.enable_smoothing = true;

    AdvancedTimestepController controller(config);

    SECTION("Growth rate limiting") {
        // Very small error should want huge increase
        Real lte = 1e-10;
        auto decision = controller.compute(lte);

        REQUIRE(decision.accepted);
        // Should be limited to max_growth_rate
        REQUIRE(decision.dt_new <= config.dt_initial * config.max_growth_rate * 1.01);
    }

    SECTION("Shrink rate limiting on rejection") {
        // Large error causes rejection
        Real lte = 1.0;  // Very large
        auto decision = controller.compute(lte);

        REQUIRE_FALSE(decision.accepted);
        // Should be limited to max_shrink_rate (or slightly more due to multiple shrinks)
        REQUIRE(decision.dt_new >= config.dt_initial * config.max_shrink_rate * 0.9);
    }

    SECTION("Smoothing disabled") {
        config.enable_smoothing = false;
        AdvancedTimestepController controller_no_smooth(config);

        Real lte = 1e-10;  // Very small
        auto decision = controller_no_smooth.compute(lte);

        // Without smoothing, growth can exceed max_growth_rate
        // (limited only by config.growth_factor)
        REQUIRE(decision.accepted);
    }
}

TEST_CASE("AdvancedTimestepController - suggest_next_dt", "[timestep][suggest]") {
    AdvancedTimestepController controller;

    Real lte = 5e-5;
    int newton_iters = 4;

    Real dt_suggested = controller.suggest_next_dt(lte, newton_iters);

    REQUIRE(dt_suggested > 0.0);
    REQUIRE(dt_suggested >= controller.config().dt_min);
    REQUIRE(dt_suggested <= controller.config().dt_max);
}

TEST_CASE("AdvancedTimestepController - Event adjustment", "[timestep][event]") {
    AdvancedTimestepController controller;

    Real current_dt = 1e-6;

    SECTION("Event far away - no adjustment") {
        Real time_to_event = 10e-6;  // 10x current timestep
        Real adjusted = controller.adjust_for_event(time_to_event, current_dt);

        REQUIRE(adjusted == current_dt);
    }

    SECTION("Event within 1.5x - hit exactly") {
        Real time_to_event = 1.2e-6;  // Within 1.5x
        Real adjusted = controller.adjust_for_event(time_to_event, current_dt);

        REQUIRE(adjusted == time_to_event);
    }

    SECTION("Event within 2x - split step") {
        Real time_to_event = 1.8e-6;  // Between 1.5x and 2x
        Real adjusted = controller.adjust_for_event(time_to_event, current_dt);

        REQUIRE(adjusted == Catch::Approx(time_to_event / 2.0));
    }
}

TEST_CASE("AdvancedTimestepController - Reset and state", "[timestep][state]") {
    AdvancedTimestepConfig config;
    config.dt_initial = 1e-6;

    AdvancedTimestepController controller(config);

    // Make some decisions to change state
    controller.compute(1e-5);
    controller.accept(2e-6);

    REQUIRE(controller.current_dt() == 2e-6);

    // Reset should restore initial state
    controller.reset();

    REQUIRE(controller.current_dt() == config.dt_initial);
    REQUIRE(controller.rejections() == 0);
}

TEST_CASE("AdvancedTimestepController - Consecutive rejections", "[timestep][rejection]") {
    AdvancedTimestepConfig config;
    config.dt_initial = 1e-6;
    config.max_rejections = 5;
    config.error_tolerance = 1e-6;

    AdvancedTimestepController controller(config);

    // Force many rejections with large error
    for (int i = 0; i < 10; ++i) {
        controller.compute(1.0);  // Very large error
    }

    // Should have hit max rejections
    REQUIRE(controller.failed());
}
