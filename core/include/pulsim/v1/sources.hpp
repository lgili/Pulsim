#pragma once

// =============================================================================
// PulsimCore - Time-Varying Sources (PWM, Sine, Ramp, etc.)
// =============================================================================

#include "pulsim/v1/device_base.hpp"
#include <functional>
#include <cmath>
#include <optional>

namespace pulsim::v1 {

// =============================================================================
// Waveform Types
// =============================================================================

enum class WaveformType {
    DC,         // Constant value
    PWM,        // Square wave with duty cycle
    Sine,       // Sinusoidal
    Ramp,       // Sawtooth (ramp up, instant reset)
    Triangle,   // Triangle wave
    Pulse       // Single pulse
};

// =============================================================================
// PWM Parameters
// =============================================================================

struct PWMParams {
    Real v_high = 1.0;       // High voltage level [V]
    Real v_low = 0.0;        // Low voltage level [V]
    Real frequency = 10e3;   // Switching frequency [Hz]
    Real duty = 0.5;         // Duty cycle [0-1]
    Real phase = 0.0;        // Initial phase [rad]
    Real dead_time = 0.0;    // Dead time [s]
    Real rise_time = 0.0;    // Rise time [s] (0 = ideal)
    Real fall_time = 0.0;    // Fall time [s] (0 = ideal)
};

// =============================================================================
// PWM Voltage Source
// =============================================================================

/// Time-varying voltage source with PWM output
/// Supports fixed duty, variable duty (callback), and closed-loop control
class PWMVoltageSource : public LinearDeviceBase<PWMVoltageSource> {
public:
    using Base = LinearDeviceBase<PWMVoltageSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::VoltageSource);

    using DutyCallback = std::function<Real(Real)>;  // duty = f(time)

    /// Construct with parameters
    explicit PWMVoltageSource(const PWMParams& params, std::string name = "")
        : Base(std::move(name))
        , params_(params)
        , branch_index_(-1)
    {}

    /// Construct with basic parameters
    PWMVoltageSource(Real v_high, Real v_low, Real frequency, Real duty,
                     std::string name = "")
        : Base(std::move(name))
        , branch_index_(-1)
    {
        params_.v_high = v_high;
        params_.v_low = v_low;
        params_.frequency = frequency;
        params_.duty = duty;
    }

    // =========================================================================
    // Branch Index (required for voltage sources)
    // =========================================================================

    void set_branch_index(NodeIndex idx) { branch_index_ = idx; }
    [[nodiscard]] NodeIndex branch_index() const { return branch_index_; }

    // =========================================================================
    // Parameter Access
    // =========================================================================

    [[nodiscard]] const PWMParams& params() const { return params_; }
    [[nodiscard]] PWMParams& params() { return params_; }

    [[nodiscard]] Real frequency() const { return params_.frequency; }
    [[nodiscard]] Real period() const { return 1.0 / params_.frequency; }

    // =========================================================================
    // Duty Cycle Control
    // =========================================================================

    /// Set fixed duty cycle [0-1]
    void set_duty(Real d) {
        params_.duty = std::clamp(d, 0.0, 1.0);
        duty_callback_ = std::nullopt;
    }

    /// Set duty cycle callback (called at each timestep)
    void set_duty_callback(DutyCallback cb) {
        duty_callback_ = std::move(cb);
    }

    /// Clear duty callback (use fixed duty)
    void clear_duty_callback() {
        duty_callback_ = std::nullopt;
    }

    /// Get current duty (may depend on time if callback is set)
    [[nodiscard]] Real duty_at(Real t) const {
        if (duty_callback_) {
            return std::clamp((*duty_callback_)(t), 0.0, 1.0);
        }
        return params_.duty;
    }

    // =========================================================================
    // Voltage Calculation
    // =========================================================================

    /// Calculate output voltage at time t
    [[nodiscard]] Real voltage_at(Real t) const {
        Real T = period();
        Real d = duty_at(t);

        // Handle dead time
        Real t_on = d * T - params_.dead_time;
        if (t_on < 0) t_on = 0;

        // Phase-shifted time within period
        Real t_phase = t + params_.phase / (2.0 * M_PI) * T;
        Real t_mod = std::fmod(t_phase, T);
        if (t_mod < 0) t_mod += T;

        // Determine state
        bool is_high = (t_mod < t_on);

        // Handle rise/fall times (linear ramp)
        if (params_.rise_time > 0 || params_.fall_time > 0) {
            if (t_mod < params_.rise_time) {
                // Rising edge
                Real ratio = t_mod / params_.rise_time;
                return params_.v_low + ratio * (params_.v_high - params_.v_low);
            } else if (t_mod < t_on) {
                // High state
                return params_.v_high;
            } else if (t_mod < t_on + params_.fall_time) {
                // Falling edge
                Real ratio = (t_mod - t_on) / params_.fall_time;
                return params_.v_high - ratio * (params_.v_high - params_.v_low);
            } else {
                // Low state
                return params_.v_low;
            }
        }

        return is_high ? params_.v_high : params_.v_low;
    }

    /// Get PWM state (ON/OFF) at time t
    [[nodiscard]] bool state_at(Real t) const {
        return voltage_at(t) > (params_.v_high + params_.v_low) / 2.0;
    }

    // =========================================================================
    // MNA Stamping (Time-Dependent)
    // =========================================================================

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes, Real t) const {
        if (nodes.size() < 2 || branch_index_ < 0) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const NodeIndex br = branch_index_;

        // Stamp MNA extension (same as regular voltage source)
        if (n_plus >= 0) {
            G.coeffRef(n_plus, br) += 1.0;
            G.coeffRef(br, n_plus) += 1.0;
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, br) -= 1.0;
            G.coeffRef(br, n_minus) -= 1.0;
        }

        // RHS: voltage at current time
        b[br] = voltage_at(t);
    }

    // Overload for DC (use t=0)
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) const {
        stamp_impl(G, b, nodes, 0.0);
    }

private:
    PWMParams params_;
    NodeIndex branch_index_;
    std::optional<DutyCallback> duty_callback_;
};

// =============================================================================
// Sine Voltage Source
// =============================================================================

struct SineParams {
    Real amplitude = 1.0;    // Peak amplitude [V]
    Real offset = 0.0;       // DC offset [V]
    Real frequency = 50.0;   // Frequency [Hz]
    Real phase = 0.0;        // Initial phase [rad]
};

class SineVoltageSource : public LinearDeviceBase<SineVoltageSource> {
public:
    using Base = LinearDeviceBase<SineVoltageSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::VoltageSource);

    explicit SineVoltageSource(const SineParams& params, std::string name = "")
        : Base(std::move(name)), params_(params), branch_index_(-1) {}

    SineVoltageSource(Real amplitude, Real frequency, Real offset = 0.0,
                      std::string name = "")
        : Base(std::move(name)), branch_index_(-1)
    {
        params_.amplitude = amplitude;
        params_.frequency = frequency;
        params_.offset = offset;
    }

    void set_branch_index(NodeIndex idx) { branch_index_ = idx; }
    [[nodiscard]] NodeIndex branch_index() const { return branch_index_; }

    [[nodiscard]] const SineParams& params() const { return params_; }

    [[nodiscard]] Real voltage_at(Real t) const {
        return params_.offset +
               params_.amplitude * std::sin(2.0 * M_PI * params_.frequency * t + params_.phase);
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes, Real t) const {
        if (nodes.size() < 2 || branch_index_ < 0) return;
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const NodeIndex br = branch_index_;

        if (n_plus >= 0) {
            G.coeffRef(n_plus, br) += 1.0;
            G.coeffRef(br, n_plus) += 1.0;
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, br) -= 1.0;
            G.coeffRef(br, n_minus) -= 1.0;
        }
        b[br] = voltage_at(t);
    }

private:
    SineParams params_;
    NodeIndex branch_index_;
};

// =============================================================================
// Ramp/Triangle Generator (for PWM carrier)
// =============================================================================

struct RampParams {
    Real v_min = 0.0;        // Minimum voltage
    Real v_max = 1.0;        // Maximum voltage
    Real frequency = 10e3;   // Frequency [Hz]
    Real phase = 0.0;        // Initial phase [rad]
    bool triangle = false;   // true = triangle, false = sawtooth
};

class RampGenerator {
public:
    explicit RampGenerator(const RampParams& params = RampParams{})
        : params_(params) {}

    RampGenerator(Real frequency, Real v_min = 0.0, Real v_max = 1.0, bool triangle = false)
        : params_{v_min, v_max, frequency, 0.0, triangle} {}

    [[nodiscard]] const RampParams& params() const { return params_; }
    [[nodiscard]] RampParams& params() { return params_; }

    [[nodiscard]] Real frequency() const { return params_.frequency; }
    [[nodiscard]] Real period() const { return 1.0 / params_.frequency; }

    /// Get value at time t
    [[nodiscard]] Real value_at(Real t) const {
        Real T = period();
        Real t_phase = t + params_.phase / (2.0 * M_PI) * T;
        Real t_mod = std::fmod(t_phase, T);
        if (t_mod < 0) t_mod += T;

        Real range = params_.v_max - params_.v_min;

        if (params_.triangle) {
            // Triangle wave: ramp up then down
            Real half_T = T / 2.0;
            if (t_mod < half_T) {
                return params_.v_min + (t_mod / half_T) * range;
            } else {
                return params_.v_max - ((t_mod - half_T) / half_T) * range;
            }
        } else {
            // Sawtooth: ramp up, instant reset
            return params_.v_min + (t_mod / T) * range;
        }
    }

private:
    RampParams params_;
};

// =============================================================================
// Pulse Voltage Source (single or periodic pulse)
// =============================================================================

struct PulseParams {
    Real v_initial = 0.0;    // Initial voltage
    Real v_pulse = 1.0;      // Pulse voltage
    Real t_delay = 0.0;      // Delay before pulse [s]
    Real t_rise = 1e-9;      // Rise time [s]
    Real t_fall = 1e-9;      // Fall time [s]
    Real t_width = 1e-6;     // Pulse width [s]
    Real period = 0.0;       // Period (0 = single pulse) [s]
};

class PulseVoltageSource : public LinearDeviceBase<PulseVoltageSource> {
public:
    using Base = LinearDeviceBase<PulseVoltageSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::VoltageSource);

    explicit PulseVoltageSource(const PulseParams& params, std::string name = "")
        : Base(std::move(name)), params_(params), branch_index_(-1) {}

    void set_branch_index(NodeIndex idx) { branch_index_ = idx; }
    [[nodiscard]] NodeIndex branch_index() const { return branch_index_; }

    [[nodiscard]] const PulseParams& params() const { return params_; }

    [[nodiscard]] Real voltage_at(Real t) const {
        Real t_eff = t;

        // Handle periodic pulses
        if (params_.period > 0 && t > params_.t_delay) {
            t_eff = params_.t_delay + std::fmod(t - params_.t_delay, params_.period);
        }

        // Before delay
        if (t_eff < params_.t_delay) {
            return params_.v_initial;
        }

        Real t_rel = t_eff - params_.t_delay;
        Real t1 = params_.t_rise;
        Real t2 = t1 + params_.t_width;
        Real t3 = t2 + params_.t_fall;

        if (t_rel < t1) {
            // Rising edge
            return params_.v_initial + (t_rel / t1) * (params_.v_pulse - params_.v_initial);
        } else if (t_rel < t2) {
            // Pulse high
            return params_.v_pulse;
        } else if (t_rel < t3) {
            // Falling edge
            Real ratio = (t_rel - t2) / params_.t_fall;
            return params_.v_pulse - ratio * (params_.v_pulse - params_.v_initial);
        } else {
            // Back to initial
            return params_.v_initial;
        }
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes, Real t) const {
        if (nodes.size() < 2 || branch_index_ < 0) return;
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const NodeIndex br = branch_index_;

        if (n_plus >= 0) {
            G.coeffRef(n_plus, br) += 1.0;
            G.coeffRef(br, n_plus) += 1.0;
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, br) -= 1.0;
            G.coeffRef(br, n_minus) -= 1.0;
        }
        b[br] = voltage_at(t);
    }

private:
    PulseParams params_;
    NodeIndex branch_index_;
};

// =============================================================================
// Type Traits for New Sources
// =============================================================================

template<>
struct device_traits<PWMVoltageSource> {
    static constexpr DeviceType type = DeviceType::VoltageSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 1;  // Branch current
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool is_time_varying = true;  // NEW: time-dependent
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 5;
};

template<>
struct device_traits<SineVoltageSource> {
    static constexpr DeviceType type = DeviceType::VoltageSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 1;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool is_time_varying = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 5;
};

template<>
struct device_traits<PulseVoltageSource> {
    static constexpr DeviceType type = DeviceType::VoltageSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 1;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool is_time_varying = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 5;
};

} // namespace pulsim::v1
