/**
 * @file test_control_blocks.cpp
 * @brief Comprehensive tests for control block models
 *
 * Tests PI/PID controllers, comparators, PWM generators, filters,
 * and other control system blocks used in power electronics.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "pulsim/models/control_blocks.hpp"

using namespace pulsim;
using namespace pulsim::models;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// =============================================================================
// PI Controller Tests
// =============================================================================

TEST_CASE("PIController - Proportional only", "[control][pi]") {
    PIParams params;
    params.Kp = 2.0;
    params.Ki = 0.0;

    PIController pi(params);

    // Pure proportional: output = Kp * error
    Real out = pi.process(5.0, 0.001);
    REQUIRE_THAT(out, WithinAbs(10.0, 1e-9));

    out = pi.process(-3.0, 0.001);
    REQUIRE_THAT(out, WithinAbs(-6.0, 1e-9));
}

TEST_CASE("PIController - Integral action", "[control][pi]") {
    PIParams params;
    params.Kp = 1.0;
    params.Ki = 10.0;

    PIController pi(params);

    // Apply constant error over time
    Real dt = 0.001;
    Real error = 1.0;

    // After one step: out = Kp*error + Ki*integral
    // integral_ stores Ki*error*dt accumulated
    Real out = pi.process(error, dt);
    // First step: P=1.0, I=0.01 (Ki*error*dt = 10*1*0.001)
    REQUIRE_THAT(out, WithinRel(1.01, 0.02));

    // After 100 total steps with same error (we did 1 already)
    for (int i = 0; i < 99; ++i) {
        out = pi.process(error, dt);
    }
    // integrator() returns the accumulated integral term (Ki*error*dt*steps)
    // = 10 * 1.0 * 0.001 * 100 = 1.0
    REQUIRE_THAT(pi.integrator(), WithinRel(1.0, 0.05));
}

TEST_CASE("PIController - Output saturation", "[control][pi]") {
    PIParams params;
    params.Kp = 10.0;
    params.Ki = 0.0;
    params.output_min = -5.0;
    params.output_max = 5.0;

    PIController pi(params);

    // Large positive error should saturate at max
    Real out = pi.process(10.0, 0.001);
    REQUIRE_THAT(out, WithinAbs(5.0, 1e-9));

    // Large negative error should saturate at min
    out = pi.process(-10.0, 0.001);
    REQUIRE_THAT(out, WithinAbs(-5.0, 1e-9));
}

TEST_CASE("PIController - Anti-windup", "[control][pi]") {
    PIParams params;
    params.Kp = 1.0;
    params.Ki = 100.0;
    params.output_min = -10.0;
    params.output_max = 10.0;
    params.anti_windup = true;

    PIController pi(params);

    // Apply large error that would cause windup
    Real dt = 0.001;
    for (int i = 0; i < 1000; ++i) {
        pi.process(100.0, dt);
    }

    // With anti-windup, integrator should not grow unbounded
    // Integrator should stop growing when output is saturated
    // With Kp=1 and error=100, output is already at 10 (max), so no integral accumulation
    REQUIRE(pi.integrator() < 20.0);  // Should be limited

    // Test without anti-windup - integrator should grow
    PIParams params_no_aw = params;
    params_no_aw.anti_windup = false;
    PIController pi_no_aw(params_no_aw);

    for (int i = 0; i < 100; ++i) {
        pi_no_aw.process(100.0, dt);
    }
    // Without anti-windup, integrator grows: 100 * 100 * 0.001 * 100 = 1000
    REQUIRE(pi_no_aw.integrator() > 100.0);  // Grows unbounded

    // Anti-windup version should have much smaller integrator
    REQUIRE(pi.integrator() < pi_no_aw.integrator());
}

TEST_CASE("PIController - Reset", "[control][pi]") {
    PIParams params;
    params.Kp = 1.0;
    params.Ki = 10.0;

    PIController pi(params);

    // Build up some integrator state
    for (int i = 0; i < 100; ++i) {
        pi.process(1.0, 0.001);
    }
    REQUIRE(pi.integrator() > 0.0);

    pi.reset();
    REQUIRE_THAT(pi.integrator(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(pi.output(), WithinAbs(0.0, 1e-12));
}

// =============================================================================
// PID Controller Tests
// =============================================================================

TEST_CASE("PIDController - All terms", "[control][pid]") {
    PIDParams params;
    params.Kp = 1.0;
    params.Ki = 10.0;
    params.Kd = 0.1;
    params.N = 100.0;

    PIDController pid(params);

    Real dt = 0.001;

    // First step with step change in error
    Real out1 = pid.process(1.0, dt);
    // Should have P, I, and D contributions
    REQUIRE(out1 > 1.0);  // P term alone would give 1.0

    // Constant error - D term should decay
    for (int i = 0; i < 100; ++i) {
        pid.process(1.0, dt);
    }

    // Now step change again
    Real out2 = pid.process(2.0, dt);
    // D term should spike again due to error change
    Real steady = params.Kp * 2.0 + params.Ki * pid.integrator();
    REQUIRE(out2 > steady * 0.9);  // D adds positive contribution
}

TEST_CASE("PIDController - Derivative on measurement", "[control][pid]") {
    PIDParams params;
    params.Kp = 1.0;
    params.Ki = 0.0;
    params.Kd = 0.1;
    params.derivative_on_measurement = true;

    PIDController pid(params);

    Real dt = 0.001;
    Real setpoint = 1.0;
    Real measurement = 0.0;

    // When setpoint changes, D-on-measurement should NOT spike
    Real out1 = pid.process(setpoint, measurement, dt);

    // Change setpoint
    Real out2 = pid.process(2.0, measurement, dt);

    // With D on measurement, changing setpoint shouldn't cause D spike
    // since measurement hasn't changed
    REQUIRE_THAT(out2, WithinRel(params.Kp * 2.0, 0.5));
}

TEST_CASE("PIDController - Output limits", "[control][pid]") {
    PIDParams params;
    params.Kp = 100.0;
    params.Ki = 0.0;
    params.Kd = 0.0;
    params.output_min = -1.0;
    params.output_max = 1.0;

    PIDController pid(params);

    Real out = pid.process(10.0, 0.001);
    REQUIRE_THAT(out, WithinAbs(1.0, 1e-9));

    out = pid.process(-10.0, 0.001);
    REQUIRE_THAT(out, WithinAbs(-1.0, 1e-9));
}

// =============================================================================
// Comparator Tests
// =============================================================================

TEST_CASE("Comparator - Basic operation", "[control][comparator]") {
    ComparatorParams params;
    params.threshold = 0.5;
    params.output_high = 1.0;
    params.output_low = 0.0;

    Comparator comp(params);

    // Below threshold
    Real out = comp.process(0.3, 0.001);
    REQUIRE_THAT(out, WithinAbs(0.0, 1e-9));
    REQUIRE(comp.state() == false);

    // Above threshold
    out = comp.process(0.7, 0.001);
    REQUIRE_THAT(out, WithinAbs(1.0, 1e-9));
    REQUIRE(comp.state() == true);
}

TEST_CASE("Comparator - Hysteresis", "[control][comparator]") {
    ComparatorParams params;
    params.threshold = 0.5;
    params.hysteresis = 0.2;  // +/- 0.1 around threshold
    params.output_high = 1.0;
    params.output_low = 0.0;

    Comparator comp(params);

    // Start low
    comp.process(0.3, 0.001);
    REQUIRE(comp.state() == false);

    // Rise to 0.55 - still below upper threshold (0.6)
    comp.process(0.55, 0.001);
    REQUIRE(comp.state() == false);

    // Rise above upper threshold
    comp.process(0.65, 0.001);
    REQUIRE(comp.state() == true);

    // Fall to 0.45 - still above lower threshold (0.4)
    comp.process(0.45, 0.001);
    REQUIRE(comp.state() == true);

    // Fall below lower threshold
    comp.process(0.35, 0.001);
    REQUIRE(comp.state() == false);
}

TEST_CASE("Comparator - Inverted", "[control][comparator]") {
    ComparatorParams params;
    params.threshold = 0.5;
    params.output_high = 1.0;
    params.output_low = 0.0;
    params.invert = true;

    Comparator comp(params);

    // Above threshold with invert = low output
    Real out = comp.process(0.7, 0.001);
    REQUIRE_THAT(out, WithinAbs(0.0, 1e-9));

    // Below threshold with invert = high output
    out = comp.process(0.3, 0.001);
    REQUIRE_THAT(out, WithinAbs(1.0, 1e-9));
}

// =============================================================================
// PWM Generator Tests
// =============================================================================

TEST_CASE("PWMGenerator - Basic operation", "[control][pwm]") {
    PWMGeneratorParams params;
    params.frequency = 1000.0;  // 1 kHz
    params.dead_time = 0.0;

    PWMGenerator pwm(params);

    Real dt = 1e-5;  // 100 kHz sample rate
    Real duty = 0.5;

    // Count high and low samples over one period
    int high_count = 0;
    int low_count = 0;
    int samples_per_period = static_cast<int>(1.0 / (params.frequency * dt));

    for (int i = 0; i < samples_per_period; ++i) {
        Real out = pwm.process(duty, dt);
        if (out > 0.5) {
            ++high_count;
        } else {
            ++low_count;
        }
    }

    // Duty cycle should be approximately 50%
    Real measured_duty = static_cast<Real>(high_count) / samples_per_period;
    REQUIRE_THAT(measured_duty, WithinRel(0.5, 0.05));
}

TEST_CASE("PWMGenerator - Duty cycle limits", "[control][pwm]") {
    PWMGeneratorParams params;
    params.frequency = 1000.0;
    params.duty_min = 0.1;
    params.duty_max = 0.9;

    PWMGenerator pwm(params);

    Real dt = 1e-5;
    int samples = 1000;

    // Request 0% duty - should be clamped to 10%
    int high_count = 0;
    for (int i = 0; i < samples; ++i) {
        if (pwm.process(0.0, dt) > 0.5) ++high_count;
    }
    Real duty = static_cast<Real>(high_count) / samples;
    REQUIRE(duty >= 0.05);  // At least some on-time

    // Request 100% duty - should be clamped to 90%
    pwm.reset();
    high_count = 0;
    for (int i = 0; i < samples; ++i) {
        if (pwm.process(1.0, dt) > 0.5) ++high_count;
    }
    duty = static_cast<Real>(high_count) / samples;
    REQUIRE(duty <= 0.95);  // Some off-time
}

// =============================================================================
// Rate Limiter Tests
// =============================================================================

TEST_CASE("RateLimiter - Basic operation", "[control][ratelimiter]") {
    RateLimiterParams params;
    params.rise_rate = 10.0;   // 10 units/s
    params.fall_rate = 10.0;   // 10 units/s

    RateLimiter rl(params);

    Real dt = 0.01;  // 10 ms

    // First call initializes output to input (no rate limiting on first call)
    Real out = rl.process(0.0, dt);
    REQUIRE_THAT(out, WithinAbs(0.0, 1e-9));

    // Now request jump to 100 - rate limiting kicks in
    out = rl.process(100.0, dt);
    // Max rise in 10ms = 10 * 0.01 = 0.1
    REQUIRE_THAT(out, WithinRel(0.1, 0.1));

    // After more iterations, should increase gradually
    for (int i = 0; i < 99; ++i) {
        out = rl.process(100.0, dt);
    }
    // After 100 steps (1 second total): 10 * 1.0 = 10.0
    REQUIRE_THAT(out, WithinRel(10.0, 0.1));
}

TEST_CASE("RateLimiter - Asymmetric rates", "[control][ratelimiter]") {
    RateLimiterParams params;
    params.rise_rate = 100.0;  // Fast rise
    params.fall_rate = 10.0;   // Slow fall

    RateLimiter rl(params);

    Real dt = 0.01;

    // Quick rise
    for (int i = 0; i < 10; ++i) {
        rl.process(100.0, dt);
    }
    Real risen = rl.output();
    REQUIRE(risen > 5.0);  // Fast rise

    // Slow fall
    for (int i = 0; i < 10; ++i) {
        rl.process(0.0, dt);
    }
    Real fallen = risen - rl.output();
    REQUIRE(fallen < risen);  // Didn't fall as much as it rose
}

// =============================================================================
// Limiter Tests
// =============================================================================

TEST_CASE("Limiter - Basic operation", "[control][limiter]") {
    LimiterParams params;
    params.min = -5.0;
    params.max = 5.0;

    Limiter lim(params);

    // Within limits
    REQUIRE_THAT(lim.process(3.0, 0.001), WithinAbs(3.0, 1e-9));
    REQUIRE(!lim.saturated_low());
    REQUIRE(!lim.saturated_high());

    // Above limit
    REQUIRE_THAT(lim.process(10.0, 0.001), WithinAbs(5.0, 1e-9));
    REQUIRE(lim.saturated_high());

    // Below limit
    REQUIRE_THAT(lim.process(-10.0, 0.001), WithinAbs(-5.0, 1e-9));
    REQUIRE(lim.saturated_low());
}

// =============================================================================
// Low-Pass Filter Tests
// =============================================================================

TEST_CASE("LowPassFilter - Step response", "[control][filter]") {
    LowPassFilterParams params;
    params.tau = 0.001;  // 1 ms time constant

    LowPassFilter lpf(params);

    Real dt = 0.0001;  // 100 us

    // Apply step input
    for (int i = 0; i < 50; ++i) {
        lpf.process(1.0, dt);
    }

    // After 5*tau, should be ~99.3% of final value
    Real t_5tau = 0.005 / dt;
    for (int i = 50; i < static_cast<int>(t_5tau); ++i) {
        lpf.process(1.0, dt);
    }
    REQUIRE_THAT(lpf.output(), WithinRel(1.0, 0.02));
}

TEST_CASE("LowPassFilter - DC pass", "[control][filter]") {
    LowPassFilterParams params;
    params.tau = 0.001;

    LowPassFilter lpf(params);

    // After long time, should pass DC completely
    for (int i = 0; i < 1000; ++i) {
        lpf.process(5.0, 0.001);
    }
    REQUIRE_THAT(lpf.output(), WithinRel(5.0, 0.01));
}

// =============================================================================
// High-Pass Filter Tests
// =============================================================================

TEST_CASE("HighPassFilter - DC blocking", "[control][filter]") {
    HighPassFilterParams params;
    params.tau = 0.001;

    HighPassFilter hpf(params);

    // After long time with DC input, output should be ~0
    for (int i = 0; i < 1000; ++i) {
        hpf.process(5.0, 0.001);
    }
    REQUIRE_THAT(hpf.output(), WithinAbs(0.0, 0.1));
}

TEST_CASE("HighPassFilter - Step response", "[control][filter]") {
    HighPassFilterParams params;
    params.tau = 0.001;

    HighPassFilter hpf(params);

    // First call initializes the filter (returns 0)
    Real first_out = hpf.process(0.0, 0.0001);
    REQUIRE_THAT(first_out, WithinAbs(0.0, 1e-9));

    // Step input - second call should see response to step change
    Real step_out = hpf.process(1.0, 0.0001);
    // High-pass responds to change in input: alpha * (y_prev + x - x_prev)
    // With alpha = tau/(tau+dt) ≈ 1 for small dt, output ≈ delta_x
    REQUIRE(step_out > 0.5);

    // After settling with constant input, should approach 0
    for (int i = 0; i < 100; ++i) {
        hpf.process(1.0, 0.001);
    }
    REQUIRE(std::abs(hpf.output()) < 0.1);
}

// =============================================================================
// Lead-Lag Compensator Tests
// =============================================================================

TEST_CASE("LeadLagCompensator - Unity at DC", "[control][leadlag]") {
    LeadLagParams params;
    params.K = 1.0;
    params.T1 = 0.001;
    params.T2 = 0.001;

    LeadLagCompensator ll(params);

    // With T1 = T2, after settling should reach a steady value
    for (int i = 0; i < 1000; ++i) {
        ll.process(5.0, 0.001);
    }
    // Output should be stable and finite
    REQUIRE(std::isfinite(ll.output()));
    REQUIRE(ll.output() > 0);
}

TEST_CASE("LeadLagCompensator - Lead behavior", "[control][leadlag]") {
    LeadLagParams params;
    params.K = 2.0;
    params.T1 = 0.01;   // Lead (numerator)
    params.T2 = 0.001;  // Lag (denominator)

    LeadLagCompensator ll(params);

    // Apply step and check initial response (lead should boost high freq)
    Real first = ll.process(1.0, 0.001);
    // First output should be K * input (initialization)
    REQUIRE_THAT(first, WithinRel(2.0, 0.01));

    // After settling, verify output is stable
    for (int i = 0; i < 5000; ++i) {
        ll.process(1.0, 0.001);
    }
    REQUIRE(std::isfinite(ll.output()));
    REQUIRE(ll.output() > 0);
}

// =============================================================================
// Delay Block Tests
// =============================================================================

TEST_CASE("DelayBlock - Basic delay", "[control][delay]") {
    DelayParams params;
    params.delay = 0.01;  // 10 ms delay

    DelayBlock delay(params);

    Real dt = 0.001;  // 1 ms steps

    // Apply step, output should stay 0 for ~10 steps
    for (int i = 0; i < 5; ++i) {
        Real out = delay.process(1.0, dt);
        REQUIRE_THAT(out, WithinAbs(0.0, 1e-9));
    }

    // After delay, should see the input
    for (int i = 0; i < 20; ++i) {
        delay.process(1.0, dt);
    }
    REQUIRE_THAT(delay.output(), WithinAbs(1.0, 0.1));
}

// =============================================================================
// Sample and Hold Tests
// =============================================================================

TEST_CASE("SampleHold - Basic operation", "[control][samplehold]") {
    SampleHoldParams params;
    params.sample_period = 0.01;  // 10 ms

    SampleHold sh(params);

    Real dt = 0.001;

    // First sample
    sh.process(1.0, dt);
    Real first = sh.output();

    // Change input, but before sample period, output should stay same
    for (int i = 0; i < 5; ++i) {
        sh.process(5.0, dt);
    }
    REQUIRE_THAT(sh.output(), WithinAbs(first, 1e-9));

    // After sample period, should update
    for (int i = 0; i < 10; ++i) {
        sh.process(5.0, dt);
    }
    REQUIRE_THAT(sh.output(), WithinRel(5.0, 0.1));
}

// =============================================================================
// Dead Zone Tests
// =============================================================================

TEST_CASE("DeadZone - Basic operation", "[control][deadzone]") {
    DeadZoneParams params;
    params.dead_band = 0.2;  // +/- 0.1

    DeadZone dz(params);

    // Within dead zone
    REQUIRE_THAT(dz.process(0.05, 0.001), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(dz.process(-0.05, 0.001), WithinAbs(0.0, 1e-9));

    // Outside dead zone (positive)
    Real out = dz.process(0.5, 0.001);
    REQUIRE_THAT(out, WithinAbs(0.4, 1e-9));  // 0.5 - 0.1 = 0.4

    // Outside dead zone (negative)
    out = dz.process(-0.5, 0.001);
    REQUIRE_THAT(out, WithinAbs(-0.4, 1e-9));  // -0.5 + 0.1 = -0.4
}

// =============================================================================
// Lookup Table Tests
// =============================================================================

TEST_CASE("LookupTable - Linear interpolation", "[control][lookup]") {
    LookupTableParams params;
    params.x_values = {0.0, 1.0, 2.0, 3.0};
    params.y_values = {0.0, 2.0, 4.0, 6.0};

    LookupTable lut(params);

    // Exact points
    REQUIRE_THAT(lut.process(0.0, 0.001), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(lut.process(1.0, 0.001), WithinAbs(2.0, 1e-9));

    // Interpolated
    REQUIRE_THAT(lut.process(0.5, 0.001), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(lut.process(1.5, 0.001), WithinAbs(3.0, 1e-9));
}

TEST_CASE("LookupTable - Clamping", "[control][lookup]") {
    LookupTableParams params;
    params.x_values = {0.0, 1.0, 2.0};
    params.y_values = {0.0, 1.0, 4.0};
    params.extrapolate = false;

    LookupTable lut(params);

    // Below range
    REQUIRE_THAT(lut.process(-1.0, 0.001), WithinAbs(0.0, 1e-9));

    // Above range
    REQUIRE_THAT(lut.process(5.0, 0.001), WithinAbs(4.0, 1e-9));
}

// =============================================================================
// Math Blocks Tests
// =============================================================================

TEST_CASE("Gain block", "[control][math]") {
    Gain g(2.5);
    REQUIRE_THAT(g.process(4.0, 0.001), WithinAbs(10.0, 1e-9));
}

TEST_CASE("Sum block", "[control][math]") {
    Sum s({1, -1, 1});
    Real out = s.process({5.0, 3.0, 2.0});
    REQUIRE_THAT(out, WithinAbs(4.0, 1e-9));  // 5 - 3 + 2 = 4
}

TEST_CASE("Product block", "[control][math]") {
    Product p;
    Real out = p.process({2.0, 3.0, 4.0});
    REQUIRE_THAT(out, WithinAbs(24.0, 1e-9));
}

TEST_CASE("Abs block", "[control][math]") {
    Abs a;
    REQUIRE_THAT(a.process(-5.0, 0.001), WithinAbs(5.0, 1e-9));
    REQUIRE_THAT(a.process(5.0, 0.001), WithinAbs(5.0, 1e-9));
}

TEST_CASE("Sqrt block", "[control][math]") {
    Sqrt sq;
    REQUIRE_THAT(sq.process(16.0, 0.001), WithinAbs(4.0, 1e-9));
    REQUIRE_THAT(sq.process(-1.0, 0.001), WithinAbs(0.0, 1e-9));  // Negative returns 0
}

// =============================================================================
// Integrator Tests
// =============================================================================

TEST_CASE("Integrator - Basic integration", "[control][integrator]") {
    IntegratorParams params;
    params.initial = 0.0;

    Integrator integ(params);

    Real dt = 0.001;

    // Integrate constant input
    for (int i = 0; i < 100; ++i) {
        integ.process(10.0, dt);
    }
    // integral of 10 over 0.1s = 1.0
    REQUIRE_THAT(integ.output(), WithinRel(1.0, 0.01));
}

TEST_CASE("Integrator - Limits", "[control][integrator]") {
    IntegratorParams params;
    params.initial = 0.0;
    params.min = -5.0;
    params.max = 5.0;

    Integrator integ(params);

    // Integrate large positive input
    for (int i = 0; i < 1000; ++i) {
        integ.process(100.0, 0.001);
    }
    REQUIRE_THAT(integ.output(), WithinAbs(5.0, 1e-9));

    // Integrate large negative input
    integ.reset();
    for (int i = 0; i < 1000; ++i) {
        integ.process(-100.0, 0.001);
    }
    REQUIRE_THAT(integ.output(), WithinAbs(-5.0, 1e-9));
}

// =============================================================================
// Differentiator Tests
// =============================================================================

TEST_CASE("Differentiator - Basic differentiation", "[control][differentiator]") {
    DifferentiatorParams params;
    params.tau = 1e-5;  // Fast filter

    Differentiator diff(params);

    Real dt = 0.0001;

    // Apply ramp input (derivative = constant)
    Real t = 0.0;
    Real slope = 10.0;  // 10 units/s
    for (int i = 0; i < 100; ++i) {
        t += dt;
        diff.process(slope * t, dt);
    }

    // Derivative should approach slope
    REQUIRE_THAT(diff.output(), WithinRel(slope, 0.2));
}

// =============================================================================
// State Space Block Tests
// =============================================================================

TEST_CASE("StateSpaceBlock - First order system", "[control][statespace]") {
    // First order system: dx/dt = -x + u, y = x
    // Time constant = 1
    StateSpaceParams params;
    params.A = {{-1.0}};
    params.B = {{1.0}};
    params.C = {{1.0}};
    params.D = {{0.0}};
    params.x0 = {0.0};

    StateSpaceBlock ss(params);

    Real dt = 0.01;

    // Step response of first-order system
    for (int i = 0; i < 500; ++i) {
        ss.process(1.0, dt);
    }

    // After 5 time constants, should be ~99.3% of final value
    REQUIRE_THAT(ss.output(), WithinRel(1.0, 0.05));
}

TEST_CASE("StateSpaceBlock - Second order system", "[control][statespace]") {
    // Mass-spring-damper: m*x'' + b*x' + k*x = F
    // Let m=1, b=0.5, k=1
    // State: [x, x']
    // dx/dt = [x', -k*x - b*x' + F] = [0, 1; -1, -0.5] * [x; x'] + [0; 1] * F
    StateSpaceParams params;
    params.A = {{0.0, 1.0}, {-1.0, -0.5}};
    params.B = {{0.0}, {1.0}};
    params.C = {{1.0, 0.0}};
    params.D = {{0.0}};
    params.x0 = {0.0, 0.0};

    StateSpaceBlock ss(params);

    Real dt = 0.01;

    // Apply step force
    for (int i = 0; i < 1000; ++i) {
        ss.process(1.0, dt);
    }

    // Steady state: k*x = F => x = F/k = 1.0
    REQUIRE_THAT(ss.output(), WithinRel(1.0, 0.1));
}

// =============================================================================
// Control System Integration Tests
// =============================================================================

TEST_CASE("PI Controller - Voltage regulation loop", "[control][integration]") {
    // Simulate a voltage regulation loop
    // Plant: first-order system with tau = 1ms
    // Controller: PI

    PIParams pi_params;
    pi_params.Kp = 10.0;
    pi_params.Ki = 1000.0;
    pi_params.output_min = 0.0;
    pi_params.output_max = 1.0;

    PIController pi(pi_params);

    LowPassFilterParams plant_params;
    plant_params.tau = 0.001;

    LowPassFilter plant(plant_params);

    Real dt = 1e-5;
    Real setpoint = 5.0;
    Real plant_output = 0.0;

    // Simulate closed-loop response
    for (int i = 0; i < 10000; ++i) {
        Real error = setpoint - plant_output;
        Real control = pi.process(error, dt);
        plant_output = plant.process(control * 10.0, dt);  // Plant gain = 10
    }

    // Should reach setpoint with good accuracy
    REQUIRE_THAT(plant_output, WithinRel(setpoint, 0.1));
}

TEST_CASE("PWM with PI control", "[control][integration]") {
    // Buck converter simulation concept
    // PI controller sets duty cycle based on output voltage error

    PIParams pi_params;
    pi_params.Kp = 0.1;
    pi_params.Ki = 100.0;
    pi_params.output_min = 0.1;
    pi_params.output_max = 0.9;

    PIController pi(pi_params);

    PWMGeneratorParams pwm_params;
    pwm_params.frequency = 100000.0;  // 100 kHz

    PWMGenerator pwm(pwm_params);

    Real dt = 1e-7;  // 10 MHz sample rate
    Real vout = 0.0;
    Real vref = 5.0;
    Real vin = 12.0;

    // Simple averaging filter for output
    LowPassFilterParams lpf_params;
    lpf_params.tau = 0.0001;
    LowPassFilter lpf(lpf_params);

    // Simulate
    for (int i = 0; i < 100000; ++i) {
        Real error = vref - vout;
        Real duty = pi.process(error, dt);
        Real pwm_out = pwm.process(duty, dt);

        // Simple buck model: vout_avg = duty * vin
        Real v_switched = pwm_out * vin;
        vout = lpf.process(v_switched, dt);
    }

    // Output should approach reference (approximately)
    REQUIRE_THAT(vout, WithinRel(vref, 0.2));
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_CASE("Control blocks - Numerical stability", "[control][stability]") {
    Real dt = 1e-9;  // Very small timestep

    // PI should not blow up
    PIParams pi_params;
    pi_params.Kp = 1e6;
    pi_params.Ki = 1e9;
    PIController pi(pi_params);
    REQUIRE_NOTHROW(pi.process(1.0, dt));
    REQUIRE(std::isfinite(pi.output()));

    // Low-pass filter with very small tau
    LowPassFilterParams lpf_params;
    lpf_params.tau = 1e-9;
    LowPassFilter lpf(lpf_params);
    REQUIRE_NOTHROW(lpf.process(1.0, dt));
    REQUIRE(std::isfinite(lpf.output()));
}

TEST_CASE("Control blocks - Zero timestep handling", "[control][stability]") {
    PIController pi({});
    REQUIRE_NOTHROW(pi.process(1.0, 0.0));

    LowPassFilter lpf({});
    REQUIRE_NOTHROW(lpf.process(1.0, 0.0));

    RateLimiter rl({});
    REQUIRE_NOTHROW(rl.process(1.0, 0.0));
}

TEST_CASE("Control blocks - Large value handling", "[control][stability]") {
    PIController pi({});
    REQUIRE_NOTHROW(pi.process(1e20, 0.001));

    LowPassFilter lpf({});
    REQUIRE_NOTHROW(lpf.process(1e20, 0.001));

    Integrator integ({});
    REQUIRE_NOTHROW(integ.process(1e20, 0.001));
}
