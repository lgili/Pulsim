#pragma once

// =============================================================================
// PulsimCore v2 - Integration Methods for Transient Simulation
// =============================================================================
// This header provides:
// - 3.2: Trapezoidal integration with correct companion model coefficients
// - 3.3: BDF methods (BDF1-BDF5) for stiff systems
// - 3.4: Local Truncation Error (LTE) estimation
// - State history management for reactive elements
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <algorithm>
#include <limits>
#include <span>
#include <functional>
#include <vector>

// Type aliases used in this header
namespace pulsim::v1 {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
} // namespace pulsim::v1

namespace pulsim::v1 {

// =============================================================================
// Integration Method Types
// =============================================================================

enum class Integrator {
    Trapezoidal,   // Second-order, A-stable
    BDF1,          // Backward Euler, first-order, L-stable
    BDF2,          // Second-order, A-stable
    BDF3,          // Third-order
    BDF4,          // Fourth-order
    BDF5,          // Fifth-order
    Gear,          // Alias for BDF2 (compatibility)
    TRBDF2,        // Trapezoidal + BDF2 (stiff-stable)
    RosenbrockW,   // Linearly implicit Rosenbrock-W (stiff-stable)
    SDIRK2         // 2nd-order SDIRK (stiff-stable)
};

/// Get integration method order
[[nodiscard]] constexpr int method_order(Integrator m) noexcept {
    switch (m) {
        case Integrator::Trapezoidal: return 2;
        case Integrator::BDF1: return 1;
        case Integrator::BDF2: return 2;
        case Integrator::BDF3: return 3;
        case Integrator::BDF4: return 4;
        case Integrator::BDF5: return 5;
        case Integrator::Gear: return 2;
        case Integrator::TRBDF2: return 2;
        case Integrator::RosenbrockW: return 2;
        case Integrator::SDIRK2: return 2;
        default: return 1;
    }
}

/// Check if method requires startup sequence
[[nodiscard]] constexpr bool requires_startup(Integrator m) noexcept {
    return m != Integrator::BDF1 && m != Integrator::Trapezoidal;
}

[[nodiscard]] constexpr bool is_stiff_stable(Integrator m) noexcept {
    switch (m) {
        case Integrator::BDF1:
        case Integrator::TRBDF2:
        case Integrator::RosenbrockW:
        case Integrator::SDIRK2:
            return true;
        default:
            return false;
    }
}

// =============================================================================
// Stage Coefficients for Multi-Stage Stiff Integrators
// =============================================================================

/// TR-BDF2 coefficients (fixed gamma)
struct TRBDF2Coeffs {
    // gamma = 2 - sqrt(2)
    static constexpr Real gamma = Real{2.0} - Real{1.4142135623730950488};
    static constexpr Real c1 = gamma;
    static constexpr Real c2 = Real{1.0};

    struct BDF2Variable {
        Real a0 = 0.0;
        Real a1 = 0.0;
        Real a2 = 0.0;
    };

    /// Variable-step BDF2 derivative coefficients at t_{n+1}
    /// dv/dt ≈ a0*v_n + a1*v_{n+γ} + a2*v_{n+1}
    [[nodiscard]] static constexpr BDF2Variable bdf2_variable(Real h1, Real h2) noexcept {
        const Real denom = h1 + h2;
        const Real h1_safe = std::max(h1, Real{1e-15});
        const Real h2_safe = std::max(h2, Real{1e-15});
        const Real denom_safe = std::max(denom, Real{1e-15});
        BDF2Variable coeffs;
        coeffs.a0 = h2_safe / (h1_safe * denom_safe);
        coeffs.a1 = -(denom_safe) / (h1_safe * h2_safe);
        coeffs.a2 = (h1_safe + 2.0 * h2_safe) / (h2_safe * denom_safe);
        return coeffs;
    }
};

/// 2-stage SDIRK2 coefficients (stiffly accurate, L-stable)
struct SDIRK2Coeffs {
    // gamma = 1 - 1/sqrt(2)
    static constexpr Real gamma = Real{1.0} - Real{0.7071067811865475244};
    static constexpr Real a11 = gamma;
    static constexpr Real a21 = Real{1.0} - gamma;
    static constexpr Real a22 = gamma;
    static constexpr Real c1 = gamma;
    static constexpr Real c2 = Real{1.0};
};

// =============================================================================
// 3.2.3: Trapezoidal Integration Coefficients
// =============================================================================

/// Trapezoidal rule companion model coefficients
/// For capacitor: I = G_eq * V + I_eq
/// where G_eq = 2C/dt, I_eq = 2C/dt * V_{n-1} + I_{n-1}
///
/// For inductor: V = R_eq * I + V_eq
/// where R_eq = 2L/dt, V_eq = 2L/dt * I_{n-1} + V_{n-1}
/// Or equivalently as conductance: G_eq = dt/(2L), I_eq = G_eq * V_eq
struct TrapezoidalCoeffs {
    /// Calculate capacitor companion model coefficients
    /// @param C capacitance in Farads
    /// @param dt timestep in seconds
    /// @param v_prev voltage at previous timestep
    /// @param i_prev current at previous timestep
    /// @return pair of (G_eq, I_eq)
    [[nodiscard]] static constexpr std::pair<Real, Real> capacitor(
        Real C, Real dt, Real v_prev, Real i_prev) noexcept {

        // Clamp dt to prevent division by zero or overflow
        const Real dt_safe = std::max(dt, Real{1e-15});
        const Real g_eq = 2.0 * C / dt_safe;
        const Real i_eq = g_eq * v_prev + i_prev;

        return {g_eq, i_eq};
    }

    /// Calculate inductor companion model coefficients
    /// Returns conductance form for MNA stamping
    /// @param L inductance in Henrys
    /// @param dt timestep in seconds
    /// @param i_prev current at previous timestep
    /// @param v_prev voltage at previous timestep
    /// @return pair of (G_eq, I_eq) where I_eq is equivalent current source
    [[nodiscard]] static constexpr std::pair<Real, Real> inductor(
        Real L, Real dt, Real i_prev, Real v_prev) noexcept {

        const Real dt_safe = std::max(dt, Real{1e-15});
        const Real L_safe = std::max(L, Real{1e-15});

        // Conductance form: G_eq = dt / (2L)
        const Real g_eq = dt_safe / (2.0 * L_safe);

        // Equivalent voltage source: V_eq = (2L/dt) * I_{n-1} + V_{n-1}
        const Real v_eq = (2.0 * L_safe / dt_safe) * i_prev + v_prev;

        // Convert to current source: I_eq = G_eq * V_eq
        const Real i_eq = g_eq * v_eq;

        return {g_eq, i_eq};
    }

    /// Calculate capacitor current from voltage change
    /// i_n = (2C/dt)(v_n - v_{n-1}) - i_{n-1}
    [[nodiscard]] static constexpr Real capacitor_current(
        Real C, Real dt, Real v_n, Real v_prev, Real i_prev) noexcept {

        const Real dt_safe = std::max(dt, Real{1e-15});
        return (2.0 * C / dt_safe) * (v_n - v_prev) - i_prev;
    }

    /// Calculate inductor voltage from current change
    /// v_n = (2L/dt)(i_n - i_{n-1}) - v_{n-1}
    [[nodiscard]] static constexpr Real inductor_voltage(
        Real L, Real dt, Real i_n, Real i_prev, Real v_prev) noexcept {

        const Real dt_safe = std::max(dt, Real{1e-15});
        return (2.0 * L / dt_safe) * (i_n - i_prev) - v_prev;
    }
};

// =============================================================================
// 3.3.1-3.3.3: BDF Method Coefficients
// =============================================================================

/// BDF (Backward Differentiation Formula) coefficients
/// dy/dt = (sum of alpha[i] * y[n-i]) / (beta * dt)
struct BDFCoeffs {
    std::array<Real, 6> alpha{};  // Coefficients for y values (current first)
    Real beta = 1.0;              // Divisor for dt
    int order = 1;

    /// BDF1 (Backward Euler): dy/dt = (y_n - y_{n-1}) / dt
    [[nodiscard]] static constexpr BDFCoeffs bdf1() noexcept {
        BDFCoeffs c;
        c.alpha = {1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
        c.beta = 1.0;
        c.order = 1;
        return c;
    }

    /// BDF2: dy/dt = (3*y_n - 4*y_{n-1} + y_{n-2}) / (2*dt)
    [[nodiscard]] static constexpr BDFCoeffs bdf2() noexcept {
        BDFCoeffs c;
        c.alpha = {3.0/2.0, -2.0, 1.0/2.0, 0.0, 0.0, 0.0};
        c.beta = 1.0;
        c.order = 2;
        return c;
    }

    /// BDF3: dy/dt = (11*y_n - 18*y_{n-1} + 9*y_{n-2} - 2*y_{n-3}) / (6*dt)
    [[nodiscard]] static constexpr BDFCoeffs bdf3() noexcept {
        BDFCoeffs c;
        c.alpha = {11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0, 0.0, 0.0};
        c.beta = 1.0;
        c.order = 3;
        return c;
    }

    /// BDF4
    [[nodiscard]] static constexpr BDFCoeffs bdf4() noexcept {
        BDFCoeffs c;
        c.alpha = {25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0, 0.0};
        c.beta = 1.0;
        c.order = 4;
        return c;
    }

    /// BDF5
    [[nodiscard]] static constexpr BDFCoeffs bdf5() noexcept {
        BDFCoeffs c;
        c.alpha = {137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0};
        c.beta = 1.0;
        c.order = 5;
        return c;
    }

    /// Get coefficients for a given order
    [[nodiscard]] static constexpr BDFCoeffs for_order(int order) noexcept {
        switch (order) {
            case 1: return bdf1();
            case 2: return bdf2();
            case 3: return bdf3();
            case 4: return bdf4();
            case 5: return bdf5();
            default: return bdf2();
        }
    }

    /// Calculate capacitor companion model for BDF
    /// @param C capacitance
    /// @param dt timestep
    /// @param v_history array of previous voltages [v_{n-1}, v_{n-2}, ...]
    /// @param i_history array of previous currents [i_{n-1}, i_{n-2}, ...]
    [[nodiscard]] std::pair<Real, Real> capacitor_companion(
        Real C, Real dt, std::span<const Real> v_history,
        [[maybe_unused]] std::span<const Real> i_history) const noexcept {

        const Real dt_safe = std::max(dt, Real{1e-15});

        // G_eq = alpha[0] * C / dt
        const Real g_eq = alpha[0] * C / dt_safe;

        // I_eq from history terms
        Real i_eq = 0.0;
        for (int i = 1; i <= order && static_cast<std::size_t>(i - 1) < v_history.size(); ++i) {
            i_eq -= alpha[i] * C / dt_safe * v_history[i - 1];
        }

        return {g_eq, i_eq};
    }

    /// Calculate inductor companion model for BDF
    [[nodiscard]] std::pair<Real, Real> inductor_companion(
        Real L, Real dt, std::span<const Real> i_history,
        [[maybe_unused]] std::span<const Real> v_history) const noexcept {

        const Real dt_safe = std::max(dt, Real{1e-15});
        const Real L_safe = std::max(L, Real{1e-15});

        // G_eq = dt / (alpha[0] * L)
        const Real g_eq = dt_safe / (alpha[0] * L_safe);

        // V_eq from history terms
        Real v_eq = 0.0;
        for (int i = 1; i <= order && static_cast<std::size_t>(i - 1) < i_history.size(); ++i) {
            v_eq -= alpha[i] * L_safe / dt_safe * i_history[i - 1];
        }

        // Convert to current source
        const Real i_eq = g_eq * v_eq;

        return {g_eq, i_eq};
    }
};

// =============================================================================
// 3.2.4: State History Storage for Reactive Elements
// =============================================================================

/// State history for dynamic elements (capacitors, inductors)
/// Stores past values for multi-step methods
template<std::size_t MaxHistory = 6>
class StateHistory {
public:
    StateHistory() = default;

    /// Initialize with a value
    explicit StateHistory(Real initial) {
        for (auto& v : values_) v = initial;
        count_ = 1;
    }

    /// Push new value (shifts history)
    void push(Real value) {
        // Shift history back
        for (std::size_t i = MaxHistory - 1; i > 0; --i) {
            values_[i] = values_[i - 1];
        }
        values_[0] = value;
        if (count_ < MaxHistory) ++count_;
    }

    /// Get value at index (0 = most recent previous)
    [[nodiscard]] Real operator[](std::size_t i) const {
        return (i < MaxHistory) ? values_[i] : values_[MaxHistory - 1];
    }

    /// Get number of valid history entries
    [[nodiscard]] std::size_t count() const { return count_; }

    /// Clear history
    void clear() {
        for (auto& v : values_) v = 0.0;
        count_ = 0;
    }

    /// Get span of valid history values
    [[nodiscard]] std::span<const Real> span() const {
        return std::span<const Real>(values_.data(), count_);
    }

    /// Set value at index without shifting
    void set(std::size_t i, Real value) {
        if (i < MaxHistory) {
            values_[i] = value;
            count_ = std::max(count_, i + 1);
        }
    }

private:
    std::array<Real, MaxHistory> values_{};
    std::size_t count_ = 0;
};

// =============================================================================
// 3.4.1-3.4.2: Local Truncation Error Estimation
// =============================================================================

/// LTE estimator for integration methods
class LTEEstimator {
public:
    /// LTE for Trapezoidal method using "Two-step" formula
    /// Compares Trapezoidal result with Backward Euler predictor
    /// LTE ~ (1/12) * h^3 * y'''
    /// Practical estimate: LTE ~ |y_trap - y_be| / 3
    [[nodiscard]] static constexpr Real trapezoidal_lte(
        Real y_trap,      // Trapezoidal result
        Real y_be,        // Backward Euler result (or predictor)
        [[maybe_unused]] Real dt) noexcept {
        // Factor 1/3 comes from error analysis
        return std::abs(y_trap - y_be) / 3.0;
    }

    /// LTE for BDF2 using order comparison
    /// Compare BDF2 with BDF1 (lower order)
    [[nodiscard]] static constexpr Real bdf2_lte(
        Real y_bdf2,
        Real y_bdf1,
        [[maybe_unused]] Real dt) noexcept {
        // BDF2 - BDF1 difference scaled by error constant ratio
        return std::abs(y_bdf2 - y_bdf1) / 3.0;
    }

    /// General LTE estimate from higher/lower order methods
    /// @param y_high result from higher-order method
    /// @param y_low result from lower-order method
    /// @param order_diff difference in orders (usually 1)
    [[nodiscard]] static constexpr Real general_lte(
        Real y_high, Real y_low, int order_diff = 1) noexcept {
        Real scale = 1.0 / (std::pow(2.0, order_diff) - 1.0);
        return std::abs(y_high - y_low) * scale;
    }

    /// LTE for capacitor voltage
    [[nodiscard]] static Real capacitor_lte(
        Real v_current,
        Real v_predicted,
        Real C,
        Real dt,
        Integrator method) noexcept {

        Real lte = std::abs(v_current - v_predicted);

        // Scale by method-specific constant
        if (method == Integrator::Trapezoidal) {
            lte /= 3.0;
        } else {
            lte /= (method_order(method) + 1.0);
        }

        // Weight by capacitor charge
        (void)C;
        (void)dt;

        return lte;
    }

    /// LTE for inductor current
    [[nodiscard]] static Real inductor_lte(
        Real i_current,
        Real i_predicted,
        Real L,
        Real dt,
        Integrator method) noexcept {

        Real lte = std::abs(i_current - i_predicted);

        if (method == Integrator::Trapezoidal) {
            lte /= 3.0;
        } else {
            lte /= (method_order(method) + 1.0);
        }

        (void)L;
        (void)dt;

        return lte;
    }
};

// =============================================================================
// 3.2.7: Overflow/Underflow Protection
// =============================================================================

/// Clamp values to prevent numerical issues
class NumericGuard {
public:
    static constexpr Real max_voltage = 1e9;      // 1 GV max
    static constexpr Real max_current = 1e9;      // 1 GA max
    static constexpr Real min_conductance = 1e-15; // Minimum conductance
    static constexpr Real max_conductance = 1e15;  // Maximum conductance

    /// Clamp voltage to safe range
    [[nodiscard]] static constexpr Real clamp_voltage(Real v) noexcept {
        return std::clamp(v, -max_voltage, max_voltage);
    }

    /// Clamp current to safe range
    [[nodiscard]] static constexpr Real clamp_current(Real i) noexcept {
        return std::clamp(i, -max_current, max_current);
    }

    /// Clamp conductance to safe range
    [[nodiscard]] static constexpr Real clamp_conductance(Real g) noexcept {
        if (g < 0) g = min_conductance;  // Conductance should be positive
        return std::clamp(g, min_conductance, max_conductance);
    }

    /// Check if value is numerically valid
    [[nodiscard]] static constexpr bool is_valid(Real v) noexcept {
        return std::isfinite(v) && std::abs(v) < 1e100;
    }

    /// Safe division with minimum denominator
    [[nodiscard]] static constexpr Real safe_divide(Real num, Real denom) noexcept {
        if (std::abs(denom) < 1e-30) {
            return (num >= 0) ? 1e30 : -1e30;
        }
        return num / denom;
    }
};

// =============================================================================
// Integration Method Factory
// =============================================================================

/// Get companion model coefficients for any integration method
struct IntegrationCoeffs {
    Real g_eq = 0.0;   // Equivalent conductance
    Real i_eq = 0.0;   // Equivalent current source

    /// Calculate capacitor companion model for specified method
    [[nodiscard]] static IntegrationCoeffs capacitor(
        Integrator method,
        Real C, Real dt,
        std::span<const Real> v_history,
        std::span<const Real> i_history) {

        IntegrationCoeffs result;

        switch (method) {
            case Integrator::Trapezoidal: {
                Real v_prev = v_history.empty() ? 0.0 : v_history[0];
                Real i_prev = i_history.empty() ? 0.0 : i_history[0];
                auto [g, i] = TrapezoidalCoeffs::capacitor(C, dt, v_prev, i_prev);
                result.g_eq = g;
                result.i_eq = i;
                break;
            }
            case Integrator::BDF1:
            case Integrator::RosenbrockW:
            case Integrator::SDIRK2:
            case Integrator::BDF2:
            case Integrator::BDF3:
            case Integrator::BDF4:
            case Integrator::BDF5:
            case Integrator::Gear:
            case Integrator::TRBDF2: {
                int order = method_order(method);
                auto bdf = BDFCoeffs::for_order(order);
                auto [g, i] = bdf.capacitor_companion(C, dt, v_history, i_history);
                result.g_eq = g;
                result.i_eq = i;
                break;
            }
        }

        // Apply numeric guards
        result.g_eq = NumericGuard::clamp_conductance(result.g_eq);
        result.i_eq = NumericGuard::clamp_current(result.i_eq);

        return result;
    }

    /// Calculate inductor companion model for specified method
    [[nodiscard]] static IntegrationCoeffs inductor(
        Integrator method,
        Real L, Real dt,
        std::span<const Real> i_history,
        std::span<const Real> v_history) {

        IntegrationCoeffs result;

        switch (method) {
            case Integrator::Trapezoidal: {
                Real i_prev = i_history.empty() ? 0.0 : i_history[0];
                Real v_prev = v_history.empty() ? 0.0 : v_history[0];
                auto [g, i] = TrapezoidalCoeffs::inductor(L, dt, i_prev, v_prev);
                result.g_eq = g;
                result.i_eq = i;
                break;
            }
            case Integrator::BDF1:
            case Integrator::RosenbrockW:
            case Integrator::SDIRK2:
            case Integrator::BDF2:
            case Integrator::BDF3:
            case Integrator::BDF4:
            case Integrator::BDF5:
            case Integrator::Gear:
            case Integrator::TRBDF2: {
                int order = method_order(method);
                auto bdf = BDFCoeffs::for_order(order);
                auto [g, i] = bdf.inductor_companion(L, dt, i_history, v_history);
                result.g_eq = g;
                result.i_eq = i;
                break;
            }
        }

        result.g_eq = NumericGuard::clamp_conductance(result.g_eq);
        result.i_eq = NumericGuard::clamp_current(result.i_eq);

        return result;
    }
};

// =============================================================================
// Analytical Solutions for Validation (3.2.5)
// =============================================================================

namespace analytical {

/// RC circuit step response: v(t) = V_final * (1 - exp(-t/tau))
/// @param t time in seconds
/// @param R resistance in Ohms
/// @param C capacitance in Farads
/// @param V_source source voltage
/// @param v0 initial capacitor voltage
[[nodiscard]] inline Real rc_step_response(
    Real t, Real R, Real C, Real V_source, Real v0 = 0.0) {
    Real tau = R * C;
    return V_source + (v0 - V_source) * std::exp(-t / tau);
}

/// RL circuit step response: i(t) = I_final * (1 - exp(-t/tau))
/// @param t time in seconds
/// @param R resistance in Ohms
/// @param L inductance in Henrys
/// @param V_source source voltage
/// @param i0 initial inductor current
[[nodiscard]] inline Real rl_step_response(
    Real t, Real R, Real L, Real V_source, Real i0 = 0.0) {
    Real tau = L / R;
    Real I_final = V_source / R;
    return I_final + (i0 - I_final) * std::exp(-t / tau);
}

/// RLC underdamped oscillation: v(t) = A * exp(-alpha*t) * cos(omega_d*t + phi)
/// @param t time
/// @param R resistance
/// @param L inductance
/// @param C capacitance
/// @param v0 initial voltage
/// @param i0 initial current
[[nodiscard]] inline Real rlc_voltage(
    Real t, Real R, Real L, Real C, Real V_source, Real v0 = 0.0, Real i0 = 0.0) {

    Real alpha = R / (2.0 * L);
    Real omega_0_sq = 1.0 / (L * C);

    Real discriminant = alpha * alpha - omega_0_sq;

    if (discriminant < 0) {
        // Underdamped
        Real omega_d = std::sqrt(-discriminant);
        Real A = v0 - V_source;
        Real B = (alpha * A + i0 / C) / omega_d;
        return V_source + std::exp(-alpha * t) * (A * std::cos(omega_d * t) + B * std::sin(omega_d * t));
    } else if (discriminant > 1e-10) {
        // Overdamped
        Real s1 = -alpha + std::sqrt(discriminant);
        Real s2 = -alpha - std::sqrt(discriminant);
        Real A1 = ((v0 - V_source) * s2 - i0 / C) / (s2 - s1);
        Real A2 = (v0 - V_source) - A1;
        return V_source + A1 * std::exp(s1 * t) + A2 * std::exp(s2 * t);
    } else {
        // Critically damped
        Real A = v0 - V_source;
        Real B = alpha * A + i0 / C;
        return V_source + (A + B * t) * std::exp(-alpha * t);
    }
}

} // namespace analytical

// =============================================================================
// 3.5: Adaptive Timestep Controller
// =============================================================================

/// Configuration for adaptive timestep controller
struct TimestepConfig {
    Real dt_min = 1e-15;         // Minimum timestep (s)
    Real dt_max = 1e-3;          // Maximum timestep (s)
    Real dt_initial = 1e-9;      // Initial timestep (s)
    Real safety_factor = 0.9;    // Safety factor for timestep prediction
    Real error_tolerance = 1e-4; // Target LTE tolerance
    Real growth_factor = 2.0;    // Maximum growth per step
    Real shrink_factor = 0.5;    // Shrink factor on rejection
    int max_rejections = 10;     // Max consecutive rejections before failure

    // PI controller gains (3.5.7)
    Real k_p = 0.075;            // Proportional gain
    Real k_i = 0.175;            // Integral gain (for PI controller)

    [[nodiscard]] static constexpr TimestepConfig defaults() {
        return TimestepConfig{};
    }

    [[nodiscard]] static constexpr TimestepConfig conservative() {
        TimestepConfig cfg;
        cfg.safety_factor = 0.8;
        cfg.growth_factor = 1.5;
        cfg.k_p = 0.05;
        cfg.k_i = 0.1;
        return cfg;
    }

    [[nodiscard]] static constexpr TimestepConfig aggressive() {
        TimestepConfig cfg;
        cfg.safety_factor = 0.95;
        cfg.growth_factor = 3.0;
        cfg.k_p = 0.1;
        cfg.k_i = 0.2;
        return cfg;
    }
};

/// Timestep history for stability (3.5.5)
class TimestepHistory {
public:
    static constexpr std::size_t max_history = 10;

    TimestepHistory() = default;

    void push(Real dt) {
        for (std::size_t i = max_history - 1; i > 0; --i) {
            history_[i] = history_[i - 1];
        }
        history_[0] = dt;
        if (count_ < max_history) ++count_;
    }

    [[nodiscard]] Real operator[](std::size_t i) const {
        return (i < count_) ? history_[i] : history_[0];
    }

    [[nodiscard]] std::size_t count() const { return count_; }

    /// Get average timestep over history
    [[nodiscard]] Real average() const {
        if (count_ == 0) return 0.0;
        Real sum = 0.0;
        for (std::size_t i = 0; i < count_; ++i) {
            sum += history_[i];
        }
        return sum / static_cast<Real>(count_);
    }

    /// Check for oscillation (timestep changing direction frequently)
    [[nodiscard]] bool is_oscillating() const {
        if (count_ < 3) return false;

        int sign_changes = 0;
        for (std::size_t i = 1; i < count_ - 1; ++i) {
            Real diff1 = history_[i] - history_[i + 1];
            Real diff2 = history_[i - 1] - history_[i];
            if (diff1 * diff2 < 0) ++sign_changes;
        }
        return sign_changes >= 2;
    }

    void clear() {
        for (auto& h : history_) h = 0.0;
        count_ = 0;
    }

private:
    std::array<Real, max_history> history_{};
    std::size_t count_ = 0;
};

/// Result of timestep decision
struct TimestepDecision {
    Real dt_new = 0.0;       // New timestep to use
    bool accepted = true;    // Whether current step was accepted
    bool at_minimum = false; // Timestep limited by dt_min
    bool at_maximum = false; // Timestep limited by dt_max
    int rejections = 0;      // Consecutive rejections
    Real error_ratio = 0.0;  // LTE / tolerance ratio
};

/// PI Controller for adaptive timestep (3.5.1)
class PITimestepController {
public:
    explicit PITimestepController(const TimestepConfig& config = {})
        : config_(config)
        , dt_current_(config.dt_initial)
        , error_prev_(0.0)
        , rejections_(0) {}

    /// Compute new timestep based on error estimate
    /// @param lte Local truncation error estimate
    /// @param order Order of integration method
    /// @return Decision about next timestep
    [[nodiscard]] TimestepDecision compute(Real lte, int order = 2) {
        TimestepDecision result;

        // Normalize error by tolerance
        Real error_ratio = lte / config_.error_tolerance;
        result.error_ratio = error_ratio;

        // Check if step should be accepted
        result.accepted = (error_ratio <= 1.0);

        if (result.accepted) {
            rejections_ = 0;

            // PI controller formula for accepted step (3.5.1)
            // dt_new = dt * safety * (1/err)^(k_i/(order+1)) * (err_prev/err)^(k_p/(order+1))
            Real exp_i = config_.k_i / (order + 1.0);
            Real exp_p = config_.k_p / (order + 1.0);

            Real factor = config_.safety_factor;

            if (error_ratio > 1e-10) {
                factor *= std::pow(1.0 / error_ratio, exp_i);

                if (error_prev_ > 1e-10) {
                    factor *= std::pow(error_prev_ / error_ratio, exp_p);
                }
            } else {
                // Error is negligible, allow growth
                factor *= config_.growth_factor;
            }

            // Limit growth
            factor = std::min(factor, config_.growth_factor);

            // Compute new timestep
            result.dt_new = dt_current_ * factor;

            // Update history
            error_prev_ = error_ratio;
            history_.push(dt_current_);

            // Anti-oscillation: if oscillating, be more conservative
            if (history_.is_oscillating()) {
                result.dt_new = std::min(result.dt_new, history_.average());
            }
        } else {
            // Step rejected (3.5.3)
            ++rejections_;
            result.rejections = rejections_;

            // Halve timestep on rejection
            result.dt_new = dt_current_ * config_.shrink_factor;

            // More aggressive shrink if multiple rejections
            if (rejections_ > 3) {
                result.dt_new *= config_.shrink_factor;
            }
        }

        // Enforce dt limits (3.5.2)
        if (result.dt_new < config_.dt_min) {
            result.dt_new = config_.dt_min;
            result.at_minimum = true;
        }
        if (result.dt_new > config_.dt_max) {
            result.dt_new = config_.dt_max;
            result.at_maximum = true;
        }

        // On rejection, track the shrinking timestep for subsequent iterations
        if (!result.accepted) {
            dt_current_ = result.dt_new;
        }

        return result;
    }

    /// Update current timestep (call after accepting step)
    void accept(Real dt) {
        dt_current_ = dt;
    }

    /// Force timestep to specific value (for events)
    void force_timestep(Real dt) {
        dt_current_ = std::clamp(dt, config_.dt_min, config_.dt_max);
    }

    /// Handle event-aware timestep adjustment (3.5.4)
    /// @param time_to_event Time until next event
    /// @param current_dt Current proposed timestep
    /// @return Adjusted timestep to hit event
    [[nodiscard]] Real adjust_for_event(Real time_to_event, Real current_dt) const {
        if (time_to_event <= 0) return current_dt;

        // If event is within 1.5x current step, adjust to hit it exactly
        if (time_to_event < 1.5 * current_dt) {
            return time_to_event;
        }

        // If event is within 2x current step, split into two steps
        if (time_to_event < 2.0 * current_dt) {
            return time_to_event / 2.0;
        }

        return current_dt;
    }

    /// Reset controller state
    void reset() {
        dt_current_ = config_.dt_initial;
        error_prev_ = 0.0;
        rejections_ = 0;
        history_.clear();
    }

    [[nodiscard]] Real current_dt() const { return dt_current_; }
    [[nodiscard]] int rejections() const { return rejections_; }
    [[nodiscard]] bool failed() const { return rejections_ > config_.max_rejections; }
    [[nodiscard]] const TimestepConfig& config() const { return config_; }
    void set_config(const TimestepConfig& cfg) { config_ = cfg; }

private:
    TimestepConfig config_;
    Real dt_current_;
    Real error_prev_;
    int rejections_;
    TimestepHistory history_;
};

/// Simple error-based timestep controller (simpler than PI)
class BasicTimestepController {
public:
    explicit BasicTimestepController(const TimestepConfig& config = {})
        : config_(config), dt_current_(config.dt_initial), rejections_(0) {}

    /// Compute new timestep based on error
    [[nodiscard]] TimestepDecision compute(Real lte, int order = 2) {
        TimestepDecision result;

        Real error_ratio = lte / config_.error_tolerance;
        result.error_ratio = error_ratio;
        result.accepted = (error_ratio <= 1.0);

        if (result.accepted) {
            rejections_ = 0;

            // Simple formula: dt_new = dt * safety * (tol/err)^(1/(order+1))
            Real exponent = 1.0 / (order + 1.0);
            Real factor = config_.safety_factor;

            if (error_ratio > 1e-10) {
                factor *= std::pow(1.0 / error_ratio, exponent);
            } else {
                factor = config_.growth_factor;
            }

            factor = std::min(factor, config_.growth_factor);
            result.dt_new = dt_current_ * factor;
        } else {
            ++rejections_;
            result.rejections = rejections_;
            result.dt_new = dt_current_ * config_.shrink_factor;
        }

        // Enforce limits
        result.dt_new = std::clamp(result.dt_new, config_.dt_min, config_.dt_max);
        result.at_minimum = (result.dt_new == config_.dt_min);
        result.at_maximum = (result.dt_new == config_.dt_max);

        return result;
    }

    void accept(Real dt) { dt_current_ = dt; }
    void reset() { dt_current_ = config_.dt_initial; rejections_ = 0; }
    [[nodiscard]] Real current_dt() const { return dt_current_; }
    [[nodiscard]] bool failed() const { return rejections_ > config_.max_rejections; }

private:
    TimestepConfig config_;
    Real dt_current_;
    int rejections_;
};

// =============================================================================
// Task 7: Advanced Timestep Controller with Newton Feedback and Smoothing
// =============================================================================

/// Extended configuration for advanced timestep controller
struct AdvancedTimestepConfig : TimestepConfig {
    // Newton iteration feedback (7.1)
    int target_newton_iterations = 5;    // Target Newton iterations per step
    int min_newton_iterations = 2;       // Minimum acceptable iterations
    int max_newton_iterations = 15;      // Maximum before timestep reduction
    Real newton_feedback_gain = 0.3;     // Gain for Newton-based adjustment

    // Rate-of-change limiting (7.3)
    Real max_growth_rate = 2.0;          // Max ratio dt_new/dt_old
    Real max_shrink_rate = 0.25;         // Min ratio dt_new/dt_old (1/4)
    bool enable_smoothing = true;        // Enable rate-of-change limiting

    // Combined control weights
    Real lte_weight = 0.7;               // Weight for LTE-based control
    Real newton_weight = 0.3;            // Weight for Newton-based control

    [[nodiscard]] static constexpr AdvancedTimestepConfig defaults() {
        return AdvancedTimestepConfig{};
    }

    [[nodiscard]] static constexpr AdvancedTimestepConfig for_switching() {
        AdvancedTimestepConfig cfg;
        cfg.dt_min = 1e-12;              // Very small for switching transients
        cfg.dt_max = 1e-5;               // Cap for PWM periods
        cfg.target_newton_iterations = 4;
        cfg.max_growth_rate = 1.5;       // More conservative growth
        cfg.safety_factor = 0.85;
        return cfg;
    }

    [[nodiscard]] static constexpr AdvancedTimestepConfig for_power_electronics() {
        AdvancedTimestepConfig cfg;
        cfg.dt_min = 1e-10;
        cfg.dt_max = 1e-4;
        cfg.target_newton_iterations = 5;
        cfg.newton_feedback_gain = 0.4;
        cfg.error_tolerance = 1e-3;
        return cfg;
    }
};

/// Extended timestep decision with Newton info
struct AdvancedTimestepDecision : TimestepDecision {
    int newton_iterations = 0;           // Actual Newton iterations used
    Real lte_factor = 1.0;               // Timestep factor from LTE
    Real newton_factor = 1.0;            // Timestep factor from Newton iterations
    Real smoothing_factor = 1.0;         // Factor applied for rate limiting
    bool smoothing_active = false;       // Whether smoothing limited the change
};

/// Advanced Timestep Controller combining LTE and Newton iteration feedback (Task 7)
/// Features:
/// - PI control for LTE-based adjustment
/// - Newton iteration feedback for convergence-aware control
/// - Rate-of-change limiting for smooth transitions
/// - Anti-oscillation detection and damping
class AdvancedTimestepController {
public:
    explicit AdvancedTimestepController(const AdvancedTimestepConfig& config = {})
        : config_(config)
        , dt_current_(config.dt_initial)
        , dt_previous_(config.dt_initial)
        , error_prev_(0.0)
        , rejections_(0) {}

    /// Compute new timestep based on LTE only (7.2 partial)
    [[nodiscard]] AdvancedTimestepDecision compute(Real lte, int order = 2) {
        return compute_combined(lte, config_.target_newton_iterations, order);
    }

    /// Compute new timestep combining LTE and Newton iterations (7.2)
    /// @param lte Local truncation error estimate
    /// @param newton_iters Actual Newton iterations used in last step
    /// @param order Integration method order
    [[nodiscard]] AdvancedTimestepDecision compute_combined(
        Real lte, int newton_iters, int order = 2) {

        AdvancedTimestepDecision result;
        result.newton_iterations = newton_iters;

        // Normalize error by tolerance
        Real error_ratio = lte / config_.error_tolerance;
        result.error_ratio = error_ratio;

        // Check if step should be accepted
        result.accepted = (error_ratio <= 1.0) &&
                          (newton_iters <= config_.max_newton_iterations);

        if (result.accepted) {
            rejections_ = 0;

            // === LTE-based factor (PI controller) ===
            Real exp_i = config_.k_i / (order + 1.0);
            Real exp_p = config_.k_p / (order + 1.0);

            Real lte_factor = config_.safety_factor;
            if (error_ratio > 1e-10) {
                lte_factor *= std::pow(1.0 / error_ratio, exp_i);
                if (error_prev_ > 1e-10) {
                    lte_factor *= std::pow(error_prev_ / error_ratio, exp_p);
                }
            } else {
                lte_factor = config_.growth_factor;
            }
            lte_factor = std::clamp(lte_factor, config_.shrink_factor, config_.growth_factor);
            result.lte_factor = lte_factor;

            // === Newton iteration-based factor ===
            Real newton_factor = compute_newton_factor(newton_iters);
            result.newton_factor = newton_factor;

            // === Combined factor with configurable weights ===
            Real combined_factor = config_.lte_weight * lte_factor +
                                   config_.newton_weight * newton_factor;

            // === Apply rate-of-change limiting (7.3) ===
            Real dt_proposed = dt_current_ * combined_factor;
            Real dt_smoothed = apply_smoothing(dt_proposed, result);

            result.dt_new = dt_smoothed;

            // Update history
            error_prev_ = error_ratio;
            history_.push(dt_current_);

            // Anti-oscillation: if oscillating, be more conservative
            if (history_.is_oscillating()) {
                result.dt_new = std::min(result.dt_new, history_.average());
                result.smoothing_active = true;
            }
        } else {
            // Step rejected
            ++rejections_;
            result.rejections = rejections_;

            // Halve timestep on rejection
            result.dt_new = dt_current_ * config_.shrink_factor;

            // More aggressive shrink if multiple rejections or Newton failed
            if (rejections_ > 3 || newton_iters >= config_.max_newton_iterations) {
                result.dt_new *= config_.shrink_factor;
            }

            result.lte_factor = config_.shrink_factor;
            result.newton_factor = config_.shrink_factor;
        }

        // Enforce dt limits
        result.dt_new = std::clamp(result.dt_new, config_.dt_min, config_.dt_max);
        result.at_minimum = (result.dt_new <= config_.dt_min * 1.001);
        result.at_maximum = (result.dt_new >= config_.dt_max * 0.999);

        // Track for smoothing
        if (!result.accepted) {
            dt_current_ = result.dt_new;
        }

        return result;
    }

    /// Suggest next timestep (7.2) - convenience wrapper
    [[nodiscard]] Real suggest_next_dt(Real lte, int newton_iters, int order = 2) {
        auto decision = compute_combined(lte, newton_iters, order);
        return decision.dt_new;
    }

    /// Update current timestep (call after accepting step)
    void accept(Real dt) {
        dt_previous_ = dt_current_;
        dt_current_ = dt;
    }

    /// Handle event-aware timestep adjustment
    [[nodiscard]] Real adjust_for_event(Real time_to_event, Real current_dt) const {
        if (time_to_event <= 0) return current_dt;

        // If event is within 1.5x current step, adjust to hit it exactly
        if (time_to_event < 1.5 * current_dt) {
            return time_to_event;
        }

        // If event is within 2x current step, split into two steps
        if (time_to_event < 2.0 * current_dt) {
            return time_to_event / 2.0;
        }

        return current_dt;
    }

    /// Force timestep to specific value (for events or restarts)
    void force_timestep(Real dt) {
        dt_current_ = std::clamp(dt, config_.dt_min, config_.dt_max);
    }

    /// Reset controller state
    void reset() {
        dt_current_ = config_.dt_initial;
        dt_previous_ = config_.dt_initial;
        error_prev_ = 0.0;
        rejections_ = 0;
        history_.clear();
    }

    [[nodiscard]] Real current_dt() const { return dt_current_; }
    [[nodiscard]] Real previous_dt() const { return dt_previous_; }
    [[nodiscard]] int rejections() const { return rejections_; }
    [[nodiscard]] bool failed() const { return rejections_ > config_.max_rejections; }
    [[nodiscard]] const AdvancedTimestepConfig& config() const { return config_; }
    void set_config(const AdvancedTimestepConfig& cfg) { config_ = cfg; }

private:
    AdvancedTimestepConfig config_;
    Real dt_current_;
    Real dt_previous_;
    Real error_prev_;
    int rejections_;
    TimestepHistory history_;

    /// Compute timestep factor based on Newton iterations (7.1)
    [[nodiscard]] Real compute_newton_factor(int newton_iters) const {
        int target = config_.target_newton_iterations;

        if (newton_iters <= config_.min_newton_iterations) {
            // Very fast convergence - can increase timestep
            return 1.0 + config_.newton_feedback_gain;
        } else if (newton_iters <= target) {
            // Normal convergence - slight increase
            Real ratio = static_cast<Real>(target - newton_iters) / target;
            return 1.0 + config_.newton_feedback_gain * ratio;
        } else if (newton_iters < config_.max_newton_iterations) {
            // Slow convergence - decrease timestep
            Real ratio = static_cast<Real>(newton_iters - target) /
                        (config_.max_newton_iterations - target);
            return 1.0 - config_.newton_feedback_gain * ratio;
        } else {
            // Very slow or failed - significant decrease
            return config_.shrink_factor;
        }
    }

    /// Apply rate-of-change limiting for smooth transitions (7.3)
    [[nodiscard]] Real apply_smoothing(Real dt_proposed,
                                        AdvancedTimestepDecision& decision) const {
        if (!config_.enable_smoothing) {
            return dt_proposed;
        }

        Real ratio = dt_proposed / dt_current_;
        Real smoothed = dt_proposed;

        // Limit growth rate
        if (ratio > config_.max_growth_rate) {
            smoothed = dt_current_ * config_.max_growth_rate;
            decision.smoothing_active = true;
            decision.smoothing_factor = config_.max_growth_rate / ratio;
        }
        // Limit shrink rate
        else if (ratio < config_.max_shrink_rate) {
            smoothed = dt_current_ * config_.max_shrink_rate;
            decision.smoothing_active = true;
            decision.smoothing_factor = config_.max_shrink_rate / ratio;
        }

        return smoothed;
    }
};

// =============================================================================
// 3.3.4 & 3.3.6: Automatic BDF Order Selection and Reduction
// =============================================================================

/// Configuration for BDF order control
struct BDFOrderConfig {
    int min_order = 1;              // Minimum BDF order
    int max_order = 5;              // Maximum BDF order
    int initial_order = 1;          // Starting order (for startup)
    Real order_increase_threshold = 0.5;  // Increase order if error < threshold * tol
    Real order_decrease_threshold = 2.0;  // Decrease order if error > threshold * tol
    int steps_before_increase = 3;  // Steps at current order before considering increase
    bool enable_auto_order = true;  // Enable automatic order selection

    [[nodiscard]] static constexpr BDFOrderConfig defaults() {
        return BDFOrderConfig{};
    }
};

/// Result of BDF order decision
struct BDFOrderDecision {
    int new_order = 2;
    int prev_order = 2;
    bool order_changed = false;
    bool order_increased = false;
    bool order_decreased = false;
    Real error_ratio = 0.0;
    std::string reason;
};

/// BDF Order Controller for automatic order selection (3.3.4) and reduction (3.3.6)
class BDFOrderController {
public:
    explicit BDFOrderController(const BDFOrderConfig& config = {})
        : config_(config)
        , current_order_(config.initial_order)
        , steps_at_order_(0)
        , convergence_failures_(0) {}

    /// Select optimal BDF order based on error estimate (3.3.4)
    /// @param lte Local truncation error
    /// @param tolerance Error tolerance
    /// @return Decision about BDF order
    [[nodiscard]] BDFOrderDecision select_order(Real lte, Real tolerance) {
        BDFOrderDecision result;
        result.prev_order = current_order_;
        result.new_order = current_order_;
        result.error_ratio = lte / tolerance;

        if (!config_.enable_auto_order) {
            return result;
        }

        ++steps_at_order_;
        convergence_failures_ = 0;  // Reset on successful step

        // Check for order increase
        if (current_order_ < config_.max_order &&
            steps_at_order_ >= config_.steps_before_increase &&
            result.error_ratio < config_.order_increase_threshold) {

            result.new_order = current_order_ + 1;
            result.order_changed = true;
            result.order_increased = true;
            result.reason = "Error low, increasing order for efficiency";
            current_order_ = result.new_order;
            steps_at_order_ = 0;
        }
        // Check for order decrease
        else if (current_order_ > config_.min_order &&
                 result.error_ratio > config_.order_decrease_threshold) {

            result.new_order = current_order_ - 1;
            result.order_changed = true;
            result.order_decreased = true;
            result.reason = "Error high, decreasing order for stability";
            current_order_ = result.new_order;
            steps_at_order_ = 0;
        }

        return result;
    }

    /// Reduce order on convergence failure (3.3.6)
    /// Call this when Newton iteration fails to converge
    /// @return New order after reduction, or -1 if cannot reduce further
    [[nodiscard]] BDFOrderDecision reduce_on_failure() {
        BDFOrderDecision result;
        result.prev_order = current_order_;
        result.error_ratio = -1.0;  // Indicates failure, not error-based

        ++convergence_failures_;

        if (current_order_ > config_.min_order) {
            result.new_order = current_order_ - 1;
            result.order_changed = true;
            result.order_decreased = true;
            result.reason = "Convergence failure, reducing order";
            current_order_ = result.new_order;
            steps_at_order_ = 0;
        } else {
            result.new_order = current_order_;
            result.reason = "At minimum order, cannot reduce further";
        }

        return result;
    }

    /// Check if too many consecutive failures
    [[nodiscard]] bool should_abort() const {
        return convergence_failures_ > 3;  // Abort after 3 consecutive failures at min order
    }

    /// Reset to initial state (for new simulation or restart)
    void reset() {
        current_order_ = config_.initial_order;
        steps_at_order_ = 0;
        convergence_failures_ = 0;
    }

    /// Force specific order (for startup sequence)
    void set_order(int order) {
        current_order_ = std::clamp(order, config_.min_order, config_.max_order);
        steps_at_order_ = 0;
    }

    /// Get BDF coefficients for current order
    [[nodiscard]] BDFCoeffs current_coeffs() const {
        return BDFCoeffs::for_order(current_order_);
    }

    [[nodiscard]] int current_order() const { return current_order_; }
    [[nodiscard]] int steps_at_current_order() const { return steps_at_order_; }
    [[nodiscard]] int convergence_failures() const { return convergence_failures_; }
    [[nodiscard]] const BDFOrderConfig& config() const { return config_; }
    void set_config(const BDFOrderConfig& cfg) { config_ = cfg; }

    /// Check if enough history is available for current order
    [[nodiscard]] bool has_sufficient_history(std::size_t history_count) const {
        return static_cast<int>(history_count) >= current_order_;
    }

    /// Get required history depth for current order
    [[nodiscard]] int required_history() const {
        return current_order_;
    }

private:
    BDFOrderConfig config_;
    int current_order_;
    int steps_at_order_;
    int convergence_failures_;
};

// =============================================================================
// 3.4.6: LTE Logging Hook for Debugging
// =============================================================================

/// LTE log entry
struct LTELogEntry {
    Real time = 0.0;              // Simulation time
    Real dt = 0.0;                // Timestep used
    Real lte = 0.0;               // Local truncation error
    Real lte_ratio = 0.0;         // LTE / tolerance
    int bdf_order = 2;            // BDF order used
    bool step_accepted = true;    // Whether step was accepted
    int element_id = -1;          // Element ID (if per-element)
    std::string element_type;     // "capacitor", "inductor", etc.

    [[nodiscard]] std::string to_csv() const {
        return std::to_string(time) + "," +
               std::to_string(dt) + "," +
               std::to_string(lte) + "," +
               std::to_string(lte_ratio) + "," +
               std::to_string(bdf_order) + "," +
               (step_accepted ? "1" : "0") + "," +
               std::to_string(element_id) + "," +
               element_type;
    }

    [[nodiscard]] static std::string csv_header() {
        return "time,dt,lte,lte_ratio,bdf_order,accepted,element_id,element_type";
    }
};

/// LTE Logger callback type
using LTELogCallback = std::function<void(const LTELogEntry&)>;

/// LTE Logger for debugging integration methods (3.4.6)
/// Disabled by default, enable with set_enabled(true)
class LTELogger {
public:
    LTELogger() = default;

    /// Enable/disable logging
    void set_enabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool is_enabled() const { return enabled_; }

    /// Set callback for log entries
    void set_callback(LTELogCallback callback) {
        callback_ = std::move(callback);
    }

    /// Log an LTE entry (no-op if disabled)
    void log(const LTELogEntry& entry) {
        if (!enabled_) return;

        // Store in buffer
        if (buffer_.size() < max_buffer_size_) {
            buffer_.push_back(entry);
        }

        // Call callback if set
        if (callback_) {
            callback_(entry);
        }

        // Update statistics
        ++total_entries_;
        if (!entry.step_accepted) ++rejected_steps_;
        max_lte_ = std::max(max_lte_, entry.lte);
        sum_lte_ += entry.lte;
    }

    /// Convenience method to log from simulation state
    void log(Real time, Real dt, Real lte, Real tolerance, int order, bool accepted,
             int element_id = -1, const std::string& element_type = "") {
        if (!enabled_) return;

        LTELogEntry entry;
        entry.time = time;
        entry.dt = dt;
        entry.lte = lte;
        entry.lte_ratio = lte / tolerance;
        entry.bdf_order = order;
        entry.step_accepted = accepted;
        entry.element_id = element_id;
        entry.element_type = element_type;

        log(entry);
    }

    /// Get buffered entries
    [[nodiscard]] const std::vector<LTELogEntry>& buffer() const { return buffer_; }

    /// Clear buffer
    void clear_buffer() { buffer_.clear(); }

    /// Export buffer to CSV string
    [[nodiscard]] std::string to_csv() const {
        std::string result = LTELogEntry::csv_header() + "\n";
        for (const auto& entry : buffer_) {
            result += entry.to_csv() + "\n";
        }
        return result;
    }

    /// Get statistics
    [[nodiscard]] std::size_t total_entries() const { return total_entries_; }
    [[nodiscard]] std::size_t rejected_steps() const { return rejected_steps_; }
    [[nodiscard]] Real max_lte() const { return max_lte_; }
    [[nodiscard]] Real average_lte() const {
        return total_entries_ > 0 ? sum_lte_ / static_cast<Real>(total_entries_) : 0.0;
    }
    [[nodiscard]] Real rejection_rate() const {
        return total_entries_ > 0 ? static_cast<Real>(rejected_steps_) / total_entries_ : 0.0;
    }

    /// Reset all statistics and buffer
    void reset() {
        buffer_.clear();
        total_entries_ = 0;
        rejected_steps_ = 0;
        max_lte_ = 0.0;
        sum_lte_ = 0.0;
    }

    /// Set maximum buffer size
    void set_max_buffer_size(std::size_t size) { max_buffer_size_ = size; }

private:
    bool enabled_ = false;
    LTELogCallback callback_;
    std::vector<LTELogEntry> buffer_;
    std::size_t max_buffer_size_ = 10000;

    // Statistics
    std::size_t total_entries_ = 0;
    std::size_t rejected_steps_ = 0;
    Real max_lte_ = 0.0;
    Real sum_lte_ = 0.0;
};

/// Global LTE logger instance (opt-in, disabled by default)
inline LTELogger& global_lte_logger() {
    static LTELogger instance;
    return instance;
}

// =============================================================================
// Richardson LTE Estimation (Task 6)
// Efficient LTE estimation using solution history without step-doubling
// =============================================================================

/// Method for LTE estimation
enum class TimestepMethod {
    StepDoubling,   // Traditional: requires 3x work (full step + 2 half steps)
    Richardson      // Efficient: uses polynomial extrapolation from solution history
};

/// Single entry in solution history for Richardson extrapolation
struct SolutionHistoryEntry {
    Vector solution;      // Full solution vector
    Real time = 0.0;      // Time point
    Real timestep = 0.0;  // Timestep used to reach this solution

    SolutionHistoryEntry() = default;
    SolutionHistoryEntry(const Vector& x, Real t, Real dt)
        : solution(x), time(t), timestep(dt) {}
};

/// Ring buffer storing solution history for Richardson extrapolation
/// Requires minimal 3 entries for 2nd order extrapolation (BDF2)
class SolutionHistory {
public:
    static constexpr std::size_t default_capacity = 5;

    explicit SolutionHistory(std::size_t capacity = default_capacity)
        : capacity_(capacity) {
        entries_.reserve(capacity);
    }

    /// Push new solution to history (oldest is removed if at capacity)
    void push(const Vector& solution, Real time, Real timestep) {
        if (entries_.size() >= capacity_) {
            // Shift entries, remove oldest
            for (std::size_t i = entries_.size() - 1; i > 0; --i) {
                entries_[i] = std::move(entries_[i - 1]);
            }
            entries_[0] = SolutionHistoryEntry(solution, time, timestep);
        } else {
            // Insert at front, shift others
            entries_.insert(entries_.begin(), SolutionHistoryEntry(solution, time, timestep));
        }
    }

    /// Get entry at index (0 = most recent)
    [[nodiscard]] const SolutionHistoryEntry& operator[](std::size_t i) const {
        return entries_.at(i);
    }

    /// Get number of entries
    [[nodiscard]] std::size_t size() const { return entries_.size(); }

    /// Check if we have enough history for Richardson extrapolation
    [[nodiscard]] bool has_sufficient_history(int order = 2) const {
        // Need at least 2 points for linear extrapolation (minimum for any LTE estimate)
        // Quadratic extrapolation (order >= 2) ideally needs 3 points, but can fall back to linear
        return entries_.size() >= 2;
    }

    /// Clear all history
    void clear() { entries_.clear(); }

    /// Get most recent solution
    [[nodiscard]] const Vector& most_recent() const {
        return entries_.front().solution;
    }

    /// Get most recent time
    [[nodiscard]] Real most_recent_time() const {
        return entries_.empty() ? 0.0 : entries_.front().time;
    }

    /// Get most recent timestep
    [[nodiscard]] Real most_recent_timestep() const {
        return entries_.empty() ? 0.0 : entries_.front().timestep;
    }

    /// Check if empty
    [[nodiscard]] bool empty() const { return entries_.empty(); }

private:
    std::vector<SolutionHistoryEntry> entries_;
    std::size_t capacity_;
};

/// Richardson extrapolation-based LTE estimator
/// Uses polynomial extrapolation from solution history to estimate error
/// without requiring additional solves (3x speedup vs step-doubling)
class RichardsonLTE {
public:
    /// Compute LTE estimate using Richardson extrapolation
    /// @param current Current solution at time t_n
    /// @param history Solution history (must have >= 2 previous solutions)
    /// @param order Integration method order (default: 2 for BDF2/Trapezoidal)
    /// @return Estimated LTE (infinity norm), or -1 if insufficient history
    [[nodiscard]] static Real compute(
        const Vector& current,
        const SolutionHistory& history,
        int order = 2) {

        if (!history.has_sufficient_history(order)) {
            return -1.0;  // Insufficient history
        }

        // Prefer quadratic extrapolation when we have enough history
        // Quadratic is more accurate and should be used when 3+ points available
        if (order >= 2 && history.size() >= 3) {
            // Quadratic extrapolation for higher accuracy
            return compute_quadratic(current, history);
        }

        // Fall back to linear extrapolation with 2 points
        if (history.size() >= 2) {
            // Linear prediction: x_pred = x_{n-1} + (x_{n-1} - x_{n-2}) * (t_n - t_{n-1}) / (t_{n-1} - t_{n-2})
            const auto& h0 = history[0];  // Most recent (x_{n-1})
            const auto& h1 = history[1];  // Second most recent (x_{n-2})

            Real dt_ratio = h0.timestep / std::max(h1.timestep, Real{1e-15});

            // Linear extrapolation
            Vector predicted = h0.solution + (h0.solution - h1.solution) * dt_ratio;

            // LTE estimate: |x_n - x_predicted| / (2^p - 1) for Richardson
            // For BDF2, error constant is 1/3
            Real lte = (current - predicted).lpNorm<Eigen::Infinity>();
            return lte / 3.0;  // BDF2 error constant
        }

        // Fallback: linear extrapolation
        return compute_linear(current, history);
    }

    /// Compute per-variable LTE (for weighted norm convergence)
    [[nodiscard]] static Vector compute_per_variable(
        const Vector& current,
        const SolutionHistory& history,
        int order = 2) {

        if (!history.has_sufficient_history(order)) {
            return Vector::Zero(current.size());
        }

        const auto& h0 = history[0];
        const auto& h1 = history[1];

        Real dt_ratio = h0.timestep / std::max(h1.timestep, Real{1e-15});
        Vector predicted = h0.solution + (h0.solution - h1.solution) * dt_ratio;

        return (current - predicted).cwiseAbs() / 3.0;
    }

    /// Compute weighted LTE (separate tolerances for voltages and currents)
    [[nodiscard]] static Real compute_weighted(
        const Vector& current,
        const SolutionHistory& history,
        Index num_nodes,
        Index num_branches,
        Real voltage_tol = 1e-3,
        Real current_tol = 1e-6,
        int order = 2) {

        if (!history.has_sufficient_history(order)) {
            return -1.0;
        }

        Vector lte_per_var = compute_per_variable(current, history, order);

        Real max_weighted_lte = 0.0;

        // Check voltage nodes
        for (Index i = 0; i < num_nodes; ++i) {
            Real weighted = lte_per_var[i] / voltage_tol;
            max_weighted_lte = std::max(max_weighted_lte, weighted);
        }

        // Check current branches
        for (Index i = num_nodes; i < num_nodes + num_branches; ++i) {
            Real weighted = lte_per_var[i] / current_tol;
            max_weighted_lte = std::max(max_weighted_lte, weighted);
        }

        return max_weighted_lte;
    }

private:
    /// Linear extrapolation (1st order polynomial)
    [[nodiscard]] static Real compute_linear(
        const Vector& current,
        const SolutionHistory& history) {

        if (history.size() < 2) return 0.0;

        const auto& h0 = history[0];
        const auto& h1 = history[1];

        Real dt_ratio = h0.timestep / std::max(h1.timestep, Real{1e-15});
        Vector predicted = h0.solution + (h0.solution - h1.solution) * dt_ratio;

        return (current - predicted).lpNorm<Eigen::Infinity>() / 2.0;
    }

    /// Quadratic extrapolation (2nd order polynomial) using Lagrange interpolation
    [[nodiscard]] static Real compute_quadratic(
        const Vector& current,
        const SolutionHistory& history) {

        if (history.size() < 3) return compute_linear(current, history);

        const auto& h0 = history[0];  // t_{n-1}
        const auto& h1 = history[1];  // t_{n-2}
        const auto& h2 = history[2];  // t_{n-3}

        // Times relative to current time
        Real t_n = h0.time + h0.timestep;  // Current time
        Real t0 = h0.time;                  // t_{n-1}
        Real t1 = h1.time;                  // t_{n-2}
        Real t2 = h2.time;                  // t_{n-3}

        // Lagrange interpolation coefficients at t_n
        Real L0 = ((t_n - t1) * (t_n - t2)) / ((t0 - t1) * (t0 - t2));
        Real L1 = ((t_n - t0) * (t_n - t2)) / ((t1 - t0) * (t1 - t2));
        Real L2 = ((t_n - t0) * (t_n - t1)) / ((t2 - t0) * (t2 - t1));

        // Predicted solution using quadratic interpolation
        Vector predicted = L0 * h0.solution + L1 * h1.solution + L2 * h2.solution;

        // LTE estimate with BDF2 error constant
        return (current - predicted).lpNorm<Eigen::Infinity>() / 3.0;
    }
};

/// Configuration for Richardson LTE estimator
struct RichardsonLTEConfig {
    TimestepMethod method = TimestepMethod::Richardson;  // Default to efficient method
    int extrapolation_order = 2;       // Polynomial order for extrapolation
    Real voltage_tolerance = 1e-3;     // Voltage tolerance for weighted LTE [V]
    Real current_tolerance = 1e-6;     // Current tolerance for weighted LTE [A]
    bool use_weighted_norm = true;     // Use weighted norm (separate V/I tolerances)
    std::size_t history_depth = 5;     // Number of solutions to keep in history

    [[nodiscard]] static constexpr RichardsonLTEConfig defaults() {
        return RichardsonLTEConfig{};
    }

    [[nodiscard]] static constexpr RichardsonLTEConfig step_doubling() {
        RichardsonLTEConfig cfg;
        cfg.method = TimestepMethod::StepDoubling;
        return cfg;
    }
};

/// Integrated LTE estimator that supports both Richardson and step-doubling
class AdaptiveLTEEstimator {
public:
    explicit AdaptiveLTEEstimator(const RichardsonLTEConfig& config = {})
        : config_(config)
        , history_(config.history_depth) {}

    /// Record solution for Richardson extrapolation
    /// Call this after each successful timestep
    void record_solution(const Vector& solution, Real time, Real timestep) {
        if (config_.method == TimestepMethod::Richardson) {
            history_.push(solution, time, timestep);
        }
    }

    /// Compute LTE using configured method
    /// For Richardson: uses history (efficient)
    /// For StepDoubling: caller must provide both solutions
    [[nodiscard]] Real compute(
        const Vector& current,
        Index num_nodes = 0,
        Index num_branches = 0) {

        if (config_.method == TimestepMethod::Richardson) {
            if (config_.use_weighted_norm && num_nodes > 0) {
                return RichardsonLTE::compute_weighted(
                    current, history_, num_nodes, num_branches,
                    config_.voltage_tolerance, config_.current_tolerance,
                    config_.extrapolation_order);
            } else {
                return RichardsonLTE::compute(current, history_, config_.extrapolation_order);
            }
        }

        // StepDoubling: return -1 to indicate caller should use traditional method
        return -1.0;
    }

    /// Check if we have sufficient history for Richardson
    [[nodiscard]] bool has_sufficient_history() const {
        return history_.has_sufficient_history(config_.extrapolation_order);
    }

    /// Get the number of solutions in history
    [[nodiscard]] std::size_t history_size() const {
        return history_.size();
    }

    /// Clear history (call at simulation start or restart)
    void reset() {
        history_.clear();
    }

    /// Check if using Richardson method
    [[nodiscard]] bool is_richardson() const {
        return config_.method == TimestepMethod::Richardson;
    }

    [[nodiscard]] const RichardsonLTEConfig& config() const { return config_; }
    void set_config(const RichardsonLTEConfig& cfg) {
        config_ = cfg;
        history_ = SolutionHistory(cfg.history_depth);
    }

    [[nodiscard]] const SolutionHistory& history() const { return history_; }

private:
    RichardsonLTEConfig config_;
    SolutionHistory history_;
};

} // namespace pulsim::v1
