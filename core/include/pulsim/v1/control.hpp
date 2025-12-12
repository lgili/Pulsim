#pragma once

// =============================================================================
// PulsimCore - Control Blocks (PI, PID, Comparator, etc.)
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>

namespace pulsim::v1 {

// =============================================================================
// PI Controller with Anti-Windup
// =============================================================================

/// Proportional-Integral controller with output limiting and anti-windup
class PIController {
public:
    /// Construct with gains and output limits
    PIController(Real Kp, Real Ki, Real output_min = 0.0, Real output_max = 1.0)
        : Kp_(Kp)
        , Ki_(Ki)
        , output_min_(output_min)
        , output_max_(output_max)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real Kp() const { return Kp_; }
    [[nodiscard]] Real Ki() const { return Ki_; }
    [[nodiscard]] Real output_min() const { return output_min_; }
    [[nodiscard]] Real output_max() const { return output_max_; }

    void set_Kp(Real kp) { Kp_ = kp; }
    void set_Ki(Real ki) { Ki_ = ki; }
    void set_output_limits(Real min, Real max) {
        output_min_ = min;
        output_max_ = max;
    }

    // =========================================================================
    // State Access
    // =========================================================================

    [[nodiscard]] Real integral() const { return integral_; }
    [[nodiscard]] Real last_output() const { return last_output_; }
    [[nodiscard]] Real last_time() const { return t_prev_; }

    // =========================================================================
    // Control Update
    // =========================================================================

    /// Update controller with error and current time
    /// Returns control output (clamped to limits)
    Real update(Real error, Real t) {
        // Calculate dt
        Real dt = (t_prev_ < 0) ? 0.0 : (t - t_prev_);
        t_prev_ = t;

        // Proportional term
        Real P = Kp_ * error;

        // Integral term with trapezoidal integration
        if (dt > 0 && Ki_ != 0) {
            // Anti-windup: only integrate if not saturated or error reduces saturation
            Real tentative_output = P + Ki_ * integral_;
            bool would_saturate_high = tentative_output > output_max_ && error > 0;
            bool would_saturate_low = tentative_output < output_min_ && error < 0;

            if (!would_saturate_high && !would_saturate_low) {
                integral_ += error * dt;
            }
        }

        // Calculate output
        Real output = P + Ki_ * integral_;

        // Clamp to limits
        output = std::clamp(output, output_min_, output_max_);
        last_output_ = output;

        return output;
    }

    /// Update with reference and feedback (calculates error internally)
    Real update(Real reference, Real feedback, Real t) {
        return update(reference - feedback, t);
    }

    /// Reset controller state
    void reset() {
        integral_ = 0.0;
        t_prev_ = -1.0;
        last_output_ = 0.0;
    }

    /// Set integral to specific value (for bumpless transfer)
    void set_integral(Real value) {
        integral_ = value;
    }

private:
    Real Kp_;
    Real Ki_;
    Real output_min_;
    Real output_max_;
    Real integral_ = 0.0;
    Real t_prev_ = -1.0;  // Negative indicates first call
    Real last_output_ = 0.0;
};

// =============================================================================
// PID Controller with Anti-Windup and Derivative Filter
// =============================================================================

/// Full PID controller with derivative filtering and anti-windup
class PIDController {
public:
    /// Construct with gains
    PIDController(Real Kp, Real Ki, Real Kd,
                  Real output_min = 0.0, Real output_max = 1.0,
                  Real derivative_filter = 0.1)  // Filter coefficient (0-1)
        : Kp_(Kp)
        , Ki_(Ki)
        , Kd_(Kd)
        , output_min_(output_min)
        , output_max_(output_max)
        , derivative_filter_(derivative_filter)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real Kp() const { return Kp_; }
    [[nodiscard]] Real Ki() const { return Ki_; }
    [[nodiscard]] Real Kd() const { return Kd_; }

    void set_gains(Real kp, Real ki, Real kd) {
        Kp_ = kp;
        Ki_ = ki;
        Kd_ = kd;
    }

    void set_output_limits(Real min, Real max) {
        output_min_ = min;
        output_max_ = max;
    }

    void set_derivative_filter(Real alpha) {
        derivative_filter_ = std::clamp(alpha, 0.0, 1.0);
    }

    // =========================================================================
    // State Access
    // =========================================================================

    [[nodiscard]] Real integral() const { return integral_; }
    [[nodiscard]] Real last_output() const { return last_output_; }

    // =========================================================================
    // Control Update
    // =========================================================================

    /// Update controller with error and current time
    Real update(Real error, Real t) {
        Real dt = (t_prev_ < 0) ? 0.0 : (t - t_prev_);
        t_prev_ = t;

        // Proportional
        Real P = Kp_ * error;

        // Derivative with filtering (derivative on error)
        Real D = 0.0;
        if (dt > 0 && Kd_ != 0) {
            Real raw_derivative = (error - error_prev_) / dt;
            // Low-pass filter: alpha=1 means no filtering
            derivative_filtered_ = derivative_filter_ * raw_derivative +
                                   (1.0 - derivative_filter_) * derivative_filtered_;
            D = Kd_ * derivative_filtered_;
        }
        error_prev_ = error;

        // Integral with anti-windup
        if (dt > 0 && Ki_ != 0) {
            Real tentative = P + Ki_ * integral_ + D;
            bool saturated_high = tentative > output_max_ && error > 0;
            bool saturated_low = tentative < output_min_ && error < 0;

            if (!saturated_high && !saturated_low) {
                integral_ += error * dt;
            }
        }

        // Output
        Real output = P + Ki_ * integral_ + D;
        output = std::clamp(output, output_min_, output_max_);
        last_output_ = output;

        return output;
    }

    /// Update with reference and feedback
    Real update(Real reference, Real feedback, Real t) {
        return update(reference - feedback, t);
    }

    /// Reset controller state
    void reset() {
        integral_ = 0.0;
        error_prev_ = 0.0;
        derivative_filtered_ = 0.0;
        t_prev_ = -1.0;
        last_output_ = 0.0;
    }

private:
    Real Kp_, Ki_, Kd_;
    Real output_min_, output_max_;
    Real derivative_filter_;
    Real integral_ = 0.0;
    Real error_prev_ = 0.0;
    Real derivative_filtered_ = 0.0;
    Real t_prev_ = -1.0;
    Real last_output_ = 0.0;
};

// =============================================================================
// Comparator (for PWM generation)
// =============================================================================

/// Simple comparator with optional hysteresis
class Comparator {
public:
    explicit Comparator(Real hysteresis = 0.0)
        : hysteresis_(hysteresis)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real hysteresis() const { return hysteresis_; }
    void set_hysteresis(Real h) { hysteresis_ = std::abs(h); }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// Compare input against reference
    /// Returns true if input > reference (with hysteresis)
    [[nodiscard]] bool compare(Real input, Real reference) {
        if (hysteresis_ == 0) {
            state_ = input > reference;
        } else {
            // Schmitt trigger behavior
            if (state_) {
                // Currently high, go low if input < reference - hysteresis/2
                if (input < reference - hysteresis_ / 2.0) {
                    state_ = false;
                }
            } else {
                // Currently low, go high if input > reference + hysteresis/2
                if (input > reference + hysteresis_ / 2.0) {
                    state_ = true;
                }
            }
        }
        return state_;
    }

    /// Get output voltage based on comparison
    [[nodiscard]] Real output(Real input, Real reference,
                              Real v_high = 1.0, Real v_low = 0.0) {
        return compare(input, reference) ? v_high : v_low;
    }

    /// Get current state without updating
    [[nodiscard]] bool state() const { return state_; }

    /// Reset comparator state
    void reset() { state_ = false; }

private:
    Real hysteresis_;
    bool state_ = false;
};

// =============================================================================
// Sample and Hold
// =============================================================================

/// Sample-and-hold block for discrete control systems
class SampleHold {
public:
    explicit SampleHold(Real sample_period)
        : period_(sample_period)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real period() const { return period_; }
    [[nodiscard]] Real frequency() const { return 1.0 / period_; }
    void set_period(Real T) { period_ = T; }

    // =========================================================================
    // State Access
    // =========================================================================

    [[nodiscard]] Real value() const { return value_; }
    [[nodiscard]] Real last_sample_time() const { return t_sample_; }

    // =========================================================================
    // Update
    // =========================================================================

    /// Update sample-and-hold, returns held value
    Real update(Real input, Real t) {
        // Check if it's time to sample
        if (t_sample_ < 0 || (t - t_sample_) >= period_) {
            value_ = input;
            t_sample_ = t;
        }
        return value_;
    }

    /// Force a sample now
    void sample_now(Real input, Real t) {
        value_ = input;
        t_sample_ = t;
    }

    /// Reset to initial state
    void reset() {
        value_ = 0.0;
        t_sample_ = -1.0;
    }

private:
    Real period_;
    Real value_ = 0.0;
    Real t_sample_ = -1.0;
};

// =============================================================================
// Rate Limiter
// =============================================================================

/// Limits the rate of change of a signal
class RateLimiter {
public:
    /// Construct with rising and falling rate limits (units/second)
    RateLimiter(Real rising_rate, Real falling_rate)
        : rising_rate_(std::abs(rising_rate))
        , falling_rate_(std::abs(falling_rate))
    {}

    /// Construct with symmetric rate limit
    explicit RateLimiter(Real rate)
        : RateLimiter(rate, rate)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real rising_rate() const { return rising_rate_; }
    [[nodiscard]] Real falling_rate() const { return falling_rate_; }
    void set_rates(Real rising, Real falling) {
        rising_rate_ = std::abs(rising);
        falling_rate_ = std::abs(falling);
    }

    // =========================================================================
    // State Access
    // =========================================================================

    [[nodiscard]] Real value() const { return value_; }

    // =========================================================================
    // Update
    // =========================================================================

    /// Update rate limiter, returns limited output
    Real update(Real input, Real t) {
        Real dt = (t_prev_ < 0) ? 0.0 : (t - t_prev_);
        t_prev_ = t;

        if (dt <= 0) {
            value_ = input;
            return value_;
        }

        Real delta = input - value_;
        Real max_rise = rising_rate_ * dt;
        Real max_fall = falling_rate_ * dt;

        if (delta > max_rise) {
            value_ += max_rise;
        } else if (delta < -max_fall) {
            value_ -= max_fall;
        } else {
            value_ = input;
        }

        return value_;
    }

    /// Reset to specific value
    void reset(Real initial = 0.0) {
        value_ = initial;
        t_prev_ = -1.0;
    }

private:
    Real rising_rate_;
    Real falling_rate_;
    Real value_ = 0.0;
    Real t_prev_ = -1.0;
};

// =============================================================================
// Moving Average Filter
// =============================================================================

/// Simple exponential moving average filter
class MovingAverageFilter {
public:
    /// Construct with time constant
    explicit MovingAverageFilter(Real time_constant)
        : tau_(time_constant)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real time_constant() const { return tau_; }
    void set_time_constant(Real tau) { tau_ = tau; }

    // =========================================================================
    // State Access
    // =========================================================================

    [[nodiscard]] Real value() const { return value_; }

    // =========================================================================
    // Update
    // =========================================================================

    /// Update filter, returns filtered output
    Real update(Real input, Real t) {
        Real dt = (t_prev_ < 0) ? 0.0 : (t - t_prev_);
        t_prev_ = t;

        if (dt <= 0 || tau_ <= 0) {
            value_ = input;
            return value_;
        }

        // First-order exponential filter: y[n] = y[n-1] + (dt/tau)*(x - y[n-1])
        Real alpha = dt / (tau_ + dt);
        value_ = value_ + alpha * (input - value_);

        return value_;
    }

    /// Reset to specific value
    void reset(Real initial = 0.0) {
        value_ = initial;
        t_prev_ = -1.0;
    }

private:
    Real tau_;
    Real value_ = 0.0;
    Real t_prev_ = -1.0;
};

// =============================================================================
// Hysteresis Controller (Bang-Bang with deadband)
// =============================================================================

/// Two-level controller with hysteresis band
class HysteresisController {
public:
    /// Construct with setpoint and hysteresis band
    HysteresisController(Real setpoint, Real band,
                         Real output_high = 1.0, Real output_low = 0.0)
        : setpoint_(setpoint)
        , band_(std::abs(band))
        , output_high_(output_high)
        , output_low_(output_low)
    {}

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] Real setpoint() const { return setpoint_; }
    [[nodiscard]] Real band() const { return band_; }
    void set_setpoint(Real sp) { setpoint_ = sp; }
    void set_band(Real b) { band_ = std::abs(b); }

    // =========================================================================
    // State Access
    // =========================================================================

    [[nodiscard]] bool state() const { return state_; }
    [[nodiscard]] Real output() const { return state_ ? output_high_ : output_low_; }

    // =========================================================================
    // Update
    // =========================================================================

    /// Update controller with feedback, returns output
    Real update(Real feedback) {
        Real upper = setpoint_ + band_ / 2.0;
        Real lower = setpoint_ - band_ / 2.0;

        if (state_) {
            // Currently ON, turn OFF if feedback > upper limit
            if (feedback > upper) {
                state_ = false;
            }
        } else {
            // Currently OFF, turn ON if feedback < lower limit
            if (feedback < lower) {
                state_ = true;
            }
        }

        return output();
    }

    /// Reset to initial state
    void reset() { state_ = false; }

private:
    Real setpoint_;
    Real band_;
    Real output_high_;
    Real output_low_;
    bool state_ = false;
};

// =============================================================================
// Lookup Table (1D interpolation)
// =============================================================================

/// 1D lookup table with linear interpolation
class LookupTable1D {
public:
    LookupTable1D() = default;

    /// Construct with x and y vectors
    LookupTable1D(std::vector<Real> x, std::vector<Real> y)
        : x_(std::move(x)), y_(std::move(y))
    {
        // Ensure x is sorted
        // (assume user provides sorted data for performance)
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    [[nodiscard]] const std::vector<Real>& x_data() const { return x_; }
    [[nodiscard]] const std::vector<Real>& y_data() const { return y_; }
    [[nodiscard]] std::size_t size() const { return x_.size(); }
    [[nodiscard]] bool empty() const { return x_.empty(); }

    // =========================================================================
    // Interpolation
    // =========================================================================

    /// Get interpolated value at x
    [[nodiscard]] Real operator()(Real x) const {
        if (x_.empty()) return 0.0;
        if (x_.size() == 1) return y_[0];

        // Find interval
        if (x <= x_.front()) return y_.front();
        if (x >= x_.back()) return y_.back();

        // Binary search for interval
        auto it = std::lower_bound(x_.begin(), x_.end(), x);
        std::size_t i = std::distance(x_.begin(), it);
        if (i == 0) i = 1;

        // Linear interpolation
        Real x0 = x_[i-1], x1 = x_[i];
        Real y0 = y_[i-1], y1 = y_[i];
        Real t = (x - x0) / (x1 - x0);

        return y0 + t * (y1 - y0);
    }

    /// Alias for operator()
    [[nodiscard]] Real interpolate(Real x) const { return (*this)(x); }

private:
    std::vector<Real> x_;
    std::vector<Real> y_;
};

} // namespace pulsim::v1
