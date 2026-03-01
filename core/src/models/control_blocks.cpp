#include "pulsim/models/control_blocks.hpp"
#include <algorithm>
#include <cmath>

namespace pulsim::models {

// =============================================================================
// PI Controller Implementation
// =============================================================================

PIController::PIController(const PIParams& params)
    : params_(params), integral_(0.0), output_(0.0) {}

Real PIController::process(Real error, Real dt) {
    // Proportional term
    Real P = params_.Kp * error;

    // Integral term with anti-windup
    Real I_new = integral_ + params_.Ki * error * dt;

    // Compute unclamped output
    Real output_unclamped = P + I_new;

    // Apply output limits
    output_ = std::clamp(output_unclamped, params_.output_min, params_.output_max);

    // Anti-windup: only integrate if not saturated, or integrating in opposite direction
    if (params_.anti_windup) {
        if ((output_unclamped > params_.output_max && error > 0) ||
            (output_unclamped < params_.output_min && error < 0)) {
            // Don't accumulate integral (already saturated)
        } else {
            integral_ = I_new;
        }
    } else {
        integral_ = I_new;
    }

    return output_;
}

void PIController::reset() {
    integral_ = 0.0;
    output_ = 0.0;
}

// =============================================================================
// PID Controller Implementation
// =============================================================================

PIDController::PIDController(const PIDParams& params)
    : params_(params) {}

Real PIDController::process(Real error, Real dt) {
    return process(error, 0.0, dt);  // No separate measurement
}

Real PIDController::process(Real setpoint, Real measurement, Real dt) {
    Real error = setpoint - measurement;

    // Proportional term
    Real P = params_.Kp * error;

    // Integral term with anti-windup
    Real I_new = integral_ + params_.Ki * error * dt;

    // Derivative term with filter
    Real D;
    if (first_call_) {
        D = 0.0;
        prev_error_ = error;
        prev_measurement_ = measurement;
        first_call_ = false;
    } else {
        Real derivative_input;
        if (params_.derivative_on_measurement) {
            // Derivative on measurement (avoids setpoint kick)
            derivative_input = -(measurement - prev_measurement_) / dt;
        } else {
            // Derivative on error
            derivative_input = (error - prev_error_) / dt;
        }

        // Low-pass filter on derivative
        Real alpha = dt / (dt + 1.0 / params_.N);
        derivative_ = derivative_ + alpha * (derivative_input - derivative_);
        D = params_.Kd * derivative_;

        prev_error_ = error;
        prev_measurement_ = measurement;
    }

    // Compute unclamped output
    Real output_unclamped = P + I_new + D;

    // Apply output limits
    output_ = std::clamp(output_unclamped, params_.output_min, params_.output_max);

    // Anti-windup
    if (params_.anti_windup) {
        if ((output_unclamped > params_.output_max && error > 0) ||
            (output_unclamped < params_.output_min && error < 0)) {
            // Don't accumulate integral
        } else {
            integral_ = I_new;
        }
    } else {
        integral_ = I_new;
    }

    return output_;
}

void PIDController::reset() {
    integral_ = 0.0;
    derivative_ = 0.0;
    prev_error_ = 0.0;
    prev_measurement_ = 0.0;
    output_ = 0.0;
    first_call_ = true;
}

// =============================================================================
// Comparator Implementation
// =============================================================================

Comparator::Comparator(const ComparatorParams& params)
    : params_(params), state_high_(false) {
    output_ = params_.invert ? params_.output_high : params_.output_low;
}

Real Comparator::process(Real input, Real dt) {
    (void)dt;

    Real threshold_high = params_.threshold + params_.hysteresis / 2.0;
    Real threshold_low = params_.threshold - params_.hysteresis / 2.0;

    if (state_high_) {
        // Currently high, switch low if input drops below lower threshold
        if (input < threshold_low) {
            state_high_ = false;
        }
    } else {
        // Currently low, switch high if input exceeds upper threshold
        if (input > threshold_high) {
            state_high_ = true;
        }
    }

    bool output_state = params_.invert ? !state_high_ : state_high_;
    output_ = output_state ? params_.output_high : params_.output_low;

    return output_;
}

void Comparator::reset() {
    state_high_ = false;
    output_ = params_.invert ? params_.output_high : params_.output_low;
}

// =============================================================================
// PWM Generator Implementation
// =============================================================================

PWMGenerator::PWMGenerator(const PWMGeneratorParams& params)
    : params_(params) {}

Real PWMGenerator::process(Real duty, Real dt) {
    // Clamp duty cycle
    duty = std::clamp(duty, params_.duty_min, params_.duty_max);

    // Update phase
    Real period = 1.0 / params_.frequency;
    phase_ += dt / period;
    if (phase_ >= 1.0) {
        phase_ -= 1.0;
    }

    // Generate PWM based on mode
    if (params_.center_aligned) {
        // Center-aligned: triangle carrier
        Real carrier;
        if (phase_ < 0.5) {
            carrier = phase_ * 2.0;  // Rising 0 to 1
        } else {
            carrier = 2.0 - phase_ * 2.0;  // Falling 1 to 0
        }
        output_ = (duty > carrier) ? 1.0 : 0.0;
    } else {
        // Edge-aligned: sawtooth carrier
        output_ = (duty > phase_) ? 1.0 : 0.0;
    }

    // Complementary output with dead time
    if (params_.dead_time > 0) {
        // Track transitions for dead time
        if (output_ > 0.5) {
            // Main is high
            if (dead_time_counter_ > 0) {
                dead_time_counter_ -= dt;
                output_comp_ = 0.0;  // Both low during dead time
            } else {
                output_comp_ = 0.0;  // Complementary is low
            }
        } else {
            // Main is low
            if (rising_edge_) {
                dead_time_counter_ = params_.dead_time;
                output_comp_ = 0.0;
                rising_edge_ = false;
            } else if (dead_time_counter_ > 0) {
                dead_time_counter_ -= dt;
                output_comp_ = 0.0;
            } else {
                output_comp_ = 1.0;  // Complementary is high
            }
        }

        // Detect falling edge of main
        static Real prev_output = 0.0;
        if (output_ < 0.5 && prev_output > 0.5) {
            rising_edge_ = true;
        }
        prev_output = output_;
    } else {
        output_comp_ = 1.0 - output_;
    }

    return output_;
}

void PWMGenerator::reset() {
    phase_ = 0.0;
    output_ = 0.0;
    output_comp_ = 1.0;
    dead_time_counter_ = 0.0;
}

// =============================================================================
// Rate Limiter Implementation
// =============================================================================

RateLimiter::RateLimiter(const RateLimiterParams& params)
    : params_(params) {}

Real RateLimiter::process(Real input, Real dt) {
    if (first_call_) {
        output_ = input;
        first_call_ = false;
        return output_;
    }

    Real delta = input - output_;
    Real max_rise = params_.rise_rate * dt;
    Real max_fall = params_.fall_rate * dt;

    if (delta > max_rise) {
        output_ += max_rise;
    } else if (delta < -max_fall) {
        output_ -= max_fall;
    } else {
        output_ = input;
    }

    return output_;
}

void RateLimiter::reset() {
    output_ = 0.0;
    first_call_ = true;
}

// =============================================================================
// Limiter Implementation
// =============================================================================

Limiter::Limiter(const LimiterParams& params)
    : params_(params) {}

Real Limiter::process(Real input, Real dt) {
    (void)dt;
    output_ = std::clamp(input, params_.min, params_.max);
    return output_;
}

// =============================================================================
// Low-Pass Filter Implementation
// =============================================================================

LowPassFilter::LowPassFilter(const LowPassFilterParams& params)
    : params_(params) {
    // Compute tau from cutoff frequency if specified
    if (params_.fc > 0) {
        params_.tau = 1.0 / (2.0 * M_PI * params_.fc);
    }
}

Real LowPassFilter::process(Real input, Real dt) {
    if (first_call_) {
        output_ = input;
        first_call_ = false;
        return output_;
    }

    // First-order IIR: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    alpha_ = dt / (params_.tau + dt);
    output_ = alpha_ * input + (1.0 - alpha_) * output_;

    return output_;
}

void LowPassFilter::reset() {
    output_ = 0.0;
    first_call_ = true;
}

// =============================================================================
// High-Pass Filter Implementation
// =============================================================================

HighPassFilter::HighPassFilter(const HighPassFilterParams& params)
    : params_(params) {
    if (params_.fc > 0) {
        params_.tau = 1.0 / (2.0 * M_PI * params_.fc);
    }
}

Real HighPassFilter::process(Real input, Real dt) {
    if (first_call_) {
        prev_input_ = input;
        output_ = 0.0;
        first_call_ = false;
        return output_;
    }

    // High-pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
    alpha_ = params_.tau / (params_.tau + dt);
    output_ = alpha_ * (output_ + input - prev_input_);
    prev_input_ = input;

    return output_;
}

void HighPassFilter::reset() {
    output_ = 0.0;
    prev_input_ = 0.0;
    first_call_ = true;
}

// =============================================================================
// Lead-Lag Compensator Implementation
// =============================================================================

LeadLagCompensator::LeadLagCompensator(const LeadLagParams& params)
    : params_(params) {}

Real LeadLagCompensator::process(Real input, Real dt) {
    if (first_call_) {
        prev_output_ = params_.K * input;  // Initialize to DC gain
        prev_input_ = input;
        output_ = prev_output_;
        first_call_ = false;
        return output_;
    }

    // Transfer function: H(s) = K * (1 + s*T1) / (1 + s*T2)
    // DC gain: H(0) = K
    //
    // Tustin discretization: s = (2/dt) * (z-1)/(z+1)
    // H(z) = K * (1 + 2*T1/dt + (1 - 2*T1/dt)*z^-1) / (1 + 2*T2/dt + (1 - 2*T2/dt)*z^-1)
    //
    // Difference equation:
    // y[n] = (a0*x[n] + a1*x[n-1] - b1*y[n-1]) / b0

    Real a0 = params_.K * (dt + 2.0 * params_.T1);
    Real a1 = params_.K * (dt - 2.0 * params_.T1);
    Real b0 = dt + 2.0 * params_.T2;
    Real b1 = dt - 2.0 * params_.T2;

    output_ = (a0 * input + a1 * prev_input_ - b1 * prev_output_) / b0;

    prev_output_ = output_;
    prev_input_ = input;

    return output_;
}

void LeadLagCompensator::reset() {
    output_ = 0.0;
    prev_output_ = 0.0;
    prev_input_ = 0.0;
    first_call_ = true;
}

// =============================================================================
// Delay Block Implementation
// =============================================================================

DelayBlock::DelayBlock(const DelayParams& params)
    : params_(params) {}

Real DelayBlock::process(Real input, Real dt) {
    // Estimate buffer size needed
    size_t buffer_size = static_cast<size_t>(params_.delay / dt) + 2;

    if (buffer_.size() != buffer_size) {
        buffer_.resize(buffer_size, 0.0);
        write_index_ = 0;
    }

    // Write input to buffer
    buffer_[write_index_] = input;

    // Calculate read index (delayed by delay time)
    size_t samples_delay = static_cast<size_t>(params_.delay / dt);
    size_t read_index = (write_index_ + buffer_size - samples_delay) % buffer_size;

    output_ = buffer_[read_index];

    // Advance write index
    write_index_ = (write_index_ + 1) % buffer_size;

    return output_;
}

void DelayBlock::reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0);
    write_index_ = 0;
    output_ = 0.0;
}

// =============================================================================
// Sample and Hold Implementation
// =============================================================================

SampleHold::SampleHold(const SampleHoldParams& params)
    : params_(params) {}

Real SampleHold::process(Real input, Real dt) {
    time_acc_ += dt;

    if (time_acc_ >= params_.sample_period) {
        output_ = input;
        time_acc_ = 0.0;
    }

    return output_;
}

void SampleHold::reset() {
    output_ = 0.0;
    time_acc_ = 0.0;
}

// =============================================================================
// Dead Zone Implementation
// =============================================================================

DeadZone::DeadZone(const DeadZoneParams& params)
    : params_(params) {}

Real DeadZone::process(Real input, Real dt) {
    (void)dt;

    Real half_band = params_.dead_band / 2.0;

    if (input > half_band) {
        output_ = input - half_band;
    } else if (input < -half_band) {
        output_ = input + half_band;
    } else {
        output_ = 0.0;
    }

    return output_;
}

// =============================================================================
// Lookup Table Implementation
// =============================================================================

LookupTable::LookupTable(const LookupTableParams& params)
    : params_(params) {}

Real LookupTable::process(Real input, Real dt) {
    (void)dt;

    if (params_.x_values.empty() || params_.y_values.empty()) {
        output_ = 0.0;
        return output_;
    }

    if (params_.x_values.size() != params_.y_values.size()) {
        output_ = 0.0;
        return output_;
    }

    // Find interval
    size_t n = params_.x_values.size();

    if (input <= params_.x_values[0]) {
        if (params_.extrapolate && n > 1) {
            Real slope = (params_.y_values[1] - params_.y_values[0]) /
                        (params_.x_values[1] - params_.x_values[0]);
            output_ = params_.y_values[0] + slope * (input - params_.x_values[0]);
        } else {
            output_ = params_.y_values[0];
        }
        return output_;
    }

    if (input >= params_.x_values[n - 1]) {
        if (params_.extrapolate && n > 1) {
            Real slope = (params_.y_values[n - 1] - params_.y_values[n - 2]) /
                        (params_.x_values[n - 1] - params_.x_values[n - 2]);
            output_ = params_.y_values[n - 1] + slope * (input - params_.x_values[n - 1]);
        } else {
            output_ = params_.y_values[n - 1];
        }
        return output_;
    }

    // Linear interpolation
    for (size_t i = 0; i < n - 1; ++i) {
        if (input >= params_.x_values[i] && input <= params_.x_values[i + 1]) {
            Real t = (input - params_.x_values[i]) /
                    (params_.x_values[i + 1] - params_.x_values[i]);
            output_ = params_.y_values[i] + t * (params_.y_values[i + 1] - params_.y_values[i]);
            return output_;
        }
    }

    output_ = 0.0;
    return output_;
}

// =============================================================================
// State Space Block Implementation
// =============================================================================

StateSpaceBlock::StateSpaceBlock(const StateSpaceParams& params)
    : params_(params) {
    if (!params_.x0.empty()) {
        state_ = params_.x0;
    } else if (!params_.A.empty()) {
        state_.resize(params_.A.size(), 0.0);
    }

    if (!params_.C.empty()) {
        outputs_.resize(params_.C.size(), 0.0);
    } else {
        outputs_.resize(1, 0.0);
    }
}

Real StateSpaceBlock::process(Real input, Real dt) {
    std::vector<Real> inputs = {input};
    auto result = process(inputs, dt);
    return result.empty() ? 0.0 : result[0];
}

std::vector<Real> StateSpaceBlock::process(const std::vector<Real>& inputs, Real dt) {
    size_t n = state_.size();
    size_t m = inputs.size();
    size_t p = outputs_.size();

    if (n == 0) {
        // No state - pure feedthrough
        if (!params_.D.empty() && params_.D.size() == p) {
            for (size_t i = 0; i < p; ++i) {
                outputs_[i] = 0.0;
                for (size_t j = 0; j < m && j < params_.D[i].size(); ++j) {
                    outputs_[i] += params_.D[i][j] * inputs[j];
                }
            }
        }
        return outputs_;
    }

    // Compute dx/dt = A*x + B*u
    std::vector<Real> dx(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        // A*x term
        if (i < params_.A.size()) {
            for (size_t j = 0; j < n && j < params_.A[i].size(); ++j) {
                dx[i] += params_.A[i][j] * state_[j];
            }
        }
        // B*u term
        if (i < params_.B.size()) {
            for (size_t j = 0; j < m && j < params_.B[i].size(); ++j) {
                dx[i] += params_.B[i][j] * inputs[j];
            }
        }
    }

    // Integrate state using forward Euler
    for (size_t i = 0; i < n; ++i) {
        state_[i] += dx[i] * dt;
    }

    // Compute output y = C*x + D*u
    for (size_t i = 0; i < p; ++i) {
        outputs_[i] = 0.0;
        // C*x term
        if (i < params_.C.size()) {
            for (size_t j = 0; j < n && j < params_.C[i].size(); ++j) {
                outputs_[i] += params_.C[i][j] * state_[j];
            }
        }
        // D*u term
        if (i < params_.D.size()) {
            for (size_t j = 0; j < m && j < params_.D[i].size(); ++j) {
                outputs_[i] += params_.D[i][j] * inputs[j];
            }
        }
    }

    return outputs_;
}

void StateSpaceBlock::reset() {
    if (!params_.x0.empty()) {
        state_ = params_.x0;
    } else {
        std::fill(state_.begin(), state_.end(), 0.0);
    }
    std::fill(outputs_.begin(), outputs_.end(), 0.0);
}

// =============================================================================
// Integrator Implementation
// =============================================================================

Integrator::Integrator(const IntegratorParams& params)
    : params_(params), output_(params.initial) {}

Real Integrator::process(Real input, Real dt) {
    output_ += input * dt;

    // Apply limits
    if (output_ > params_.max) {
        output_ = params_.max;
        if (params_.reset_on_saturation) {
            output_ = params_.initial;
        }
    } else if (output_ < params_.min) {
        output_ = params_.min;
        if (params_.reset_on_saturation) {
            output_ = params_.initial;
        }
    }

    return output_;
}

void Integrator::reset() {
    output_ = params_.initial;
}

// =============================================================================
// Differentiator Implementation
// =============================================================================

Differentiator::Differentiator(const DifferentiatorParams& params)
    : params_(params) {}

Real Differentiator::process(Real input, Real dt) {
    if (first_call_) {
        prev_input_ = input;
        filtered_derivative_ = 0.0;
        output_ = 0.0;
        first_call_ = false;
        return output_;
    }

    // Raw derivative
    Real derivative = (input - prev_input_) / dt;
    prev_input_ = input;

    // Low-pass filter for noise rejection
    Real alpha = dt / (params_.tau + dt);
    filtered_derivative_ = alpha * derivative + (1.0 - alpha) * filtered_derivative_;

    output_ = filtered_derivative_;
    return output_;
}

void Differentiator::reset() {
    output_ = 0.0;
    prev_input_ = 0.0;
    filtered_derivative_ = 0.0;
    first_call_ = true;
}

}  // namespace pulsim::models
