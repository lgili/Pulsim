#pragma once

#include "pulsim/types.hpp"
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace pulsim::models {

// =============================================================================
// Control Blocks for Power Electronics Simulation
// Implements PI, PID controllers, comparators, limiters, and more
// =============================================================================

// =============================================================================
// Base Control Block Interface
// =============================================================================

class ControlBlock {
public:
    virtual ~ControlBlock() = default;

    // Process input and return output
    virtual Real process(Real input, Real dt) = 0;

    // Reset to initial state
    virtual void reset() = 0;

    // Get current output
    virtual Real output() const = 0;
};

// =============================================================================
// Proportional-Integral (PI) Controller
// =============================================================================

struct PIParams {
    Real Kp = 1.0;           // Proportional gain
    Real Ki = 0.0;           // Integral gain
    Real output_min = -std::numeric_limits<Real>::infinity();  // Output lower limit
    Real output_max = std::numeric_limits<Real>::infinity();   // Output upper limit
    bool anti_windup = true; // Enable anti-windup
};

class PIController : public ControlBlock {
public:
    explicit PIController(const PIParams& params = {});

    Real process(Real error, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    // Get/set integrator state
    Real integrator() const { return integral_; }
    void set_integrator(Real value) { integral_ = value; }

    const PIParams& params() const { return params_; }
    void set_params(const PIParams& params) { params_ = params; }

private:
    PIParams params_;
    Real integral_ = 0.0;
    Real output_ = 0.0;
};

// =============================================================================
// Proportional-Integral-Derivative (PID) Controller
// =============================================================================

struct PIDParams {
    Real Kp = 1.0;           // Proportional gain
    Real Ki = 0.0;           // Integral gain
    Real Kd = 0.0;           // Derivative gain
    Real N = 100.0;          // Derivative filter coefficient (Td = Kd/Kp, Tf = Td/N)
    Real output_min = -std::numeric_limits<Real>::infinity();
    Real output_max = std::numeric_limits<Real>::infinity();
    bool anti_windup = true;

    // Derivative action
    bool derivative_on_measurement = false;  // D on measurement vs D on error
};

class PIDController : public ControlBlock {
public:
    explicit PIDController(const PIDParams& params = {});

    Real process(Real error, Real dt) override;

    // Process with separate setpoint and measurement (for D on measurement)
    Real process(Real setpoint, Real measurement, Real dt);

    void reset() override;
    Real output() const override { return output_; }

    // Access states
    Real integrator() const { return integral_; }
    Real derivative_state() const { return derivative_; }
    void set_integrator(Real value) { integral_ = value; }

    const PIDParams& params() const { return params_; }
    void set_params(const PIDParams& params) { params_ = params; }

private:
    PIDParams params_;
    Real integral_ = 0.0;
    Real derivative_ = 0.0;
    Real prev_error_ = 0.0;
    Real prev_measurement_ = 0.0;
    Real output_ = 0.0;
    bool first_call_ = true;
};

// =============================================================================
// Comparator
// =============================================================================

struct ComparatorParams {
    Real threshold = 0.0;    // Comparison threshold
    Real hysteresis = 0.0;   // Hysteresis band (total, centered on threshold)
    Real output_high = 1.0;  // Output when input > threshold
    Real output_low = 0.0;   // Output when input < threshold
    bool invert = false;     // Invert output
};

class Comparator : public ControlBlock {
public:
    explicit Comparator(const ComparatorParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    // Get current state (true = high, false = low)
    bool state() const { return state_high_; }

    const ComparatorParams& params() const { return params_; }

private:
    ComparatorParams params_;
    Real output_ = 0.0;
    bool state_high_ = false;
};

// =============================================================================
// PWM Generator
// =============================================================================

struct PWMGeneratorParams {
    Real frequency = 10000.0;  // Switching frequency (Hz)
    Real dead_time = 0.0;      // Dead time (s)
    Real duty_min = 0.0;       // Minimum duty cycle
    Real duty_max = 1.0;       // Maximum duty cycle
    bool center_aligned = false;  // Center-aligned vs edge-aligned
};

class PWMGenerator : public ControlBlock {
public:
    explicit PWMGenerator(const PWMGeneratorParams& params = {});

    // Input is duty cycle (0 to 1), output is PWM signal (0 or 1)
    Real process(Real duty, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    // Get complementary output (with dead time)
    Real output_complementary() const { return output_comp_; }

    // Get current phase
    Real phase() const { return phase_; }

    const PWMGeneratorParams& params() const { return params_; }

private:
    PWMGeneratorParams params_;
    Real phase_ = 0.0;
    Real output_ = 0.0;
    Real output_comp_ = 0.0;
    Real dead_time_counter_ = 0.0;
    bool rising_edge_ = true;
};

// =============================================================================
// Rate Limiter
// =============================================================================

struct RateLimiterParams {
    Real rise_rate = std::numeric_limits<Real>::infinity();   // Max rising rate (units/s)
    Real fall_rate = std::numeric_limits<Real>::infinity();   // Max falling rate (units/s)
};

class RateLimiter : public ControlBlock {
public:
    explicit RateLimiter(const RateLimiterParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    const RateLimiterParams& params() const { return params_; }

private:
    RateLimiterParams params_;
    Real output_ = 0.0;
    bool first_call_ = true;
};

// =============================================================================
// Saturation / Limiter
// =============================================================================

struct LimiterParams {
    Real min = -std::numeric_limits<Real>::infinity();
    Real max = std::numeric_limits<Real>::infinity();
};

class Limiter : public ControlBlock {
public:
    explicit Limiter(const LimiterParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

    // Check if output is saturated
    bool saturated_low() const { return output_ <= params_.min; }
    bool saturated_high() const { return output_ >= params_.max; }

private:
    LimiterParams params_;
    Real output_ = 0.0;
};

// =============================================================================
// First-Order Low-Pass Filter
// =============================================================================

struct LowPassFilterParams {
    Real tau = 1e-3;  // Time constant (s)
    // Alternatively specify cutoff frequency
    Real fc = 0.0;    // Cutoff frequency (Hz), computed from tau if 0
};

class LowPassFilter : public ControlBlock {
public:
    explicit LowPassFilter(const LowPassFilterParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    void set_initial(Real value) { output_ = value; first_call_ = false; }

    const LowPassFilterParams& params() const { return params_; }

private:
    LowPassFilterParams params_;
    Real output_ = 0.0;
    Real alpha_ = 0.0;
    bool first_call_ = true;
};

// =============================================================================
// First-Order High-Pass Filter
// =============================================================================

struct HighPassFilterParams {
    Real tau = 1e-3;  // Time constant (s)
    Real fc = 0.0;    // Cutoff frequency (Hz)
};

class HighPassFilter : public ControlBlock {
public:
    explicit HighPassFilter(const HighPassFilterParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    const HighPassFilterParams& params() const { return params_; }

private:
    HighPassFilterParams params_;
    Real output_ = 0.0;
    Real prev_input_ = 0.0;
    Real alpha_ = 0.0;
    bool first_call_ = true;
};

// =============================================================================
// Lead-Lag Compensator
// =============================================================================

struct LeadLagParams {
    Real K = 1.0;     // Gain
    Real T1 = 1e-3;   // Lead time constant (numerator)
    Real T2 = 1e-3;   // Lag time constant (denominator)
};

class LeadLagCompensator : public ControlBlock {
public:
    explicit LeadLagCompensator(const LeadLagParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    const LeadLagParams& params() const { return params_; }

private:
    LeadLagParams params_;
    Real output_ = 0.0;
    Real prev_output_ = 0.0;
    Real prev_input_ = 0.0;
    bool first_call_ = true;
};

// =============================================================================
// Delay Block
// =============================================================================

struct DelayParams {
    Real delay = 1e-3;  // Delay time (s)
};

class DelayBlock : public ControlBlock {
public:
    explicit DelayBlock(const DelayParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    const DelayParams& params() const { return params_; }

private:
    DelayParams params_;
    Real output_ = 0.0;
    std::vector<Real> buffer_;
    size_t write_index_ = 0;
};

// =============================================================================
// Sample and Hold
// =============================================================================

struct SampleHoldParams {
    Real sample_period = 1e-3;  // Sampling period (s)
};

class SampleHold : public ControlBlock {
public:
    explicit SampleHold(const SampleHoldParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    const SampleHoldParams& params() const { return params_; }

private:
    SampleHoldParams params_;
    Real output_ = 0.0;
    Real time_acc_ = 0.0;
};

// =============================================================================
// Dead Zone
// =============================================================================

struct DeadZoneParams {
    Real dead_band = 0.1;  // Dead band width (total, symmetric around 0)
};

class DeadZone : public ControlBlock {
public:
    explicit DeadZone(const DeadZoneParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

private:
    DeadZoneParams params_;
    Real output_ = 0.0;
};

// =============================================================================
// Lookup Table
// =============================================================================

struct LookupTableParams {
    std::vector<Real> x_values;
    std::vector<Real> y_values;
    bool extrapolate = false;  // Extrapolate beyond table bounds
};

class LookupTable : public ControlBlock {
public:
    explicit LookupTable(const LookupTableParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

    const LookupTableParams& params() const { return params_; }

private:
    LookupTableParams params_;
    Real output_ = 0.0;
};

// =============================================================================
// State Space Block
// dx/dt = A*x + B*u
// y = C*x + D*u
// =============================================================================

struct StateSpaceParams {
    std::vector<std::vector<Real>> A;  // State matrix
    std::vector<std::vector<Real>> B;  // Input matrix
    std::vector<std::vector<Real>> C;  // Output matrix
    std::vector<std::vector<Real>> D;  // Feedthrough matrix
    std::vector<Real> x0;              // Initial state
};

class StateSpaceBlock : public ControlBlock {
public:
    explicit StateSpaceBlock(const StateSpaceParams& params = {});

    Real process(Real input, Real dt) override;

    // Multi-input/output version
    std::vector<Real> process(const std::vector<Real>& inputs, Real dt);

    void reset() override;
    Real output() const override { return outputs_.empty() ? 0.0 : outputs_[0]; }

    // Access state
    const std::vector<Real>& state() const { return state_; }
    void set_state(const std::vector<Real>& x) { state_ = x; }

private:
    StateSpaceParams params_;
    std::vector<Real> state_;
    std::vector<Real> outputs_;
};

// =============================================================================
// Math Operations
// =============================================================================

class Gain : public ControlBlock {
public:
    explicit Gain(Real k = 1.0) : k_(k) {}

    Real process(Real input, Real dt) override {
        (void)dt;
        output_ = k_ * input;
        return output_;
    }
    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

private:
    Real k_;
    Real output_ = 0.0;
};

class Sum : public ControlBlock {
public:
    // Signs: +1 or -1 for each input
    explicit Sum(const std::vector<int>& signs = {1, 1}) : signs_(signs) {}

    Real process(Real input, Real dt) override {
        (void)dt;
        output_ = input;
        return output_;
    }

    Real process(const std::vector<Real>& inputs) {
        output_ = 0.0;
        for (size_t i = 0; i < inputs.size() && i < signs_.size(); ++i) {
            output_ += signs_[i] * inputs[i];
        }
        return output_;
    }

    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

private:
    std::vector<int> signs_;
    Real output_ = 0.0;
};

class Product : public ControlBlock {
public:
    Product() = default;

    Real process(Real input, Real dt) override {
        (void)dt;
        output_ = input;
        return output_;
    }

    Real process(const std::vector<Real>& inputs) {
        output_ = 1.0;
        for (auto x : inputs) {
            output_ *= x;
        }
        return output_;
    }

    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

private:
    Real output_ = 0.0;
};

class Abs : public ControlBlock {
public:
    Real process(Real input, Real dt) override {
        (void)dt;
        output_ = std::abs(input);
        return output_;
    }
    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

private:
    Real output_ = 0.0;
};

class Sqrt : public ControlBlock {
public:
    Real process(Real input, Real dt) override {
        (void)dt;
        output_ = (input >= 0) ? std::sqrt(input) : 0.0;
        return output_;
    }
    void reset() override { output_ = 0.0; }
    Real output() const override { return output_; }

private:
    Real output_ = 0.0;
};

// =============================================================================
// Integrator
// =============================================================================

struct IntegratorParams {
    Real initial = 0.0;
    Real min = -std::numeric_limits<Real>::infinity();
    Real max = std::numeric_limits<Real>::infinity();
    bool reset_on_saturation = false;
};

class Integrator : public ControlBlock {
public:
    explicit Integrator(const IntegratorParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

    const IntegratorParams& params() const { return params_; }

private:
    IntegratorParams params_;
    Real output_ = 0.0;
};

// =============================================================================
// Differentiator (with filter)
// =============================================================================

struct DifferentiatorParams {
    Real tau = 1e-4;  // Filter time constant (for noise rejection)
};

class Differentiator : public ControlBlock {
public:
    explicit Differentiator(const DifferentiatorParams& params = {});

    Real process(Real input, Real dt) override;
    void reset() override;
    Real output() const override { return output_; }

private:
    DifferentiatorParams params_;
    Real output_ = 0.0;
    Real prev_input_ = 0.0;
    Real filtered_derivative_ = 0.0;
    bool first_call_ = true;
};

}  // namespace pulsim::models
