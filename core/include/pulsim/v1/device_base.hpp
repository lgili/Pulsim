#pragma once

// =============================================================================
// PulsimCore v2 - CRTP Device Base Classes
// =============================================================================
// This header provides the CRTP (Curiously Recurring Template Pattern) base
// classes for device models. CRTP enables:
// - Static polymorphism (no virtual function overhead)
// - Compile-time Jacobian sparsity analysis
// - Inlining of stamp() calls for maximum performance
// =============================================================================

#include "pulsim/v1/concepts.hpp"
#include "pulsim/v1/type_traits.hpp"
#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/cpp23_features.hpp"
#include <Eigen/Sparse>
#include <span>
#include <array>

namespace pulsim::v1 {

// =============================================================================
// Type Aliases for Device Implementations
// =============================================================================
// Use the configurable types from numeric_types.hpp with default settings

// Scalar is alias to Real (double by default) for device implementations
using Scalar = Real;
using NodeIndex = Index;

// Matrix types for MNA system
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

// =============================================================================
// CRTP Device Base - Core Interface
// =============================================================================

/// CRTP base class for all devices
/// Derived classes must implement:
///   - stamp_impl(Matrix& G, Vector& b, std::span<const NodeIndex> nodes)
///   - static constexpr auto jacobian_pattern_impl()
template<typename Derived>
class DeviceBase {
public:
    using ScalarType = Scalar;
    using Scalar = ScalarType;  // Alias for CRTP inheritance

    /// Stamp the device into the MNA matrix (static dispatch)
    template<SparseMatrixLike Matrix, VectorLike Vec>
    void stamp(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        derived().stamp_impl(G, b, nodes);
    }

    /// Get the Jacobian sparsity pattern (compile-time)
    static constexpr auto jacobian_pattern() {
        return Derived::jacobian_pattern_impl();
    }

    /// Get the device name
    [[nodiscard]] const std::string& name() const { return name_; }
    void set_name(std::string n) { name_ = std::move(n); }

protected:
    DeviceBase() = default;
    explicit DeviceBase(std::string name) : name_(std::move(name)) {}

    // CRTP helper - cast this to derived type
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

private:
    std::string name_;
};

// =============================================================================
// CRTP Base for Linear Devices (R, L, C with companion model)
// =============================================================================

template<typename Derived>
class LinearDeviceBase : public DeviceBase<Derived> {
public:
    using Base = DeviceBase<Derived>;
    using typename Base::Scalar;

protected:
    using Base::Base;
};

// =============================================================================
// CRTP Base for Nonlinear Devices (Diode, MOSFET, etc.)
// =============================================================================

template<typename Derived>
class NonlinearDeviceBase : public DeviceBase<Derived> {
public:
    using Base = DeviceBase<Derived>;
    using typename Base::Scalar;

    /// Stamp the Jacobian contribution for Newton iteration
    template<SparseMatrixLike Matrix, VectorLike Vec>
    void stamp_jacobian(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        this->derived().stamp_jacobian_impl(J, f, x, nodes);
    }

protected:
    using Base::Base;
};

// =============================================================================
// CRTP Base for Dynamic Devices (Capacitors, Inductors)
// =============================================================================

template<typename Derived>
class DynamicDeviceBase : public LinearDeviceBase<Derived> {
public:
    using Base = LinearDeviceBase<Derived>;
    using typename Base::Scalar;

    /// Update history for next timestep
    void update_history() {
        this->derived().update_history_impl();
    }

    /// Set integration timestep
    void set_timestep(Scalar dt) {
        dt_ = dt;
    }

    /// Get current timestep
    [[nodiscard]] Scalar timestep() const { return dt_; }

protected:
    using Base::Base;
    Scalar dt_ = 1e-6;  // Default timestep
};

// =============================================================================
// Example: Resistor Device (CRTP)
// =============================================================================

class Resistor : public LinearDeviceBase<Resistor> {
public:
    using Base = LinearDeviceBase<Resistor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Resistor);

    /// Parameter structure for Resistor
    struct Params {
        Scalar resistance = 1000.0;
    };

    explicit Resistor(Scalar resistance, std::string name = "")
        : Base(std::move(name)), resistance_(resistance) {}

    /// Stamp implementation (called via CRTP)
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const Scalar g = 1.0 / resistance_;

        // Stamp conductance matrix
        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g;
            }
        }
    }

    /// Jacobian pattern (compile-time)
    static constexpr auto jacobian_pattern_impl() {
        // Resistor contributes to 4 positions: (n+,n+), (n+,n-), (n-,n+), (n-,n-)
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar resistance() const { return resistance_; }
    void set_resistance(Scalar r) { resistance_ = r; }

private:
    Scalar resistance_;
};

// Specialization of device_traits for Resistor
template<>
struct device_traits<Resistor> {
    static constexpr DeviceType type = DeviceType::Resistor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;  // Conduction loss = I²R
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;  // 2x2 contribution
};

// =============================================================================
// Example: Capacitor Device (CRTP with dynamics)
// =============================================================================

class Capacitor : public DynamicDeviceBase<Capacitor> {
public:
    using Base = DynamicDeviceBase<Capacitor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Capacitor);

    /// Parameter structure for Capacitor
    struct Params {
        Scalar capacitance = 1e-6;
        Scalar initial_voltage = 0.0;
    };

    explicit Capacitor(Scalar capacitance, Scalar initial_voltage = 0.0, std::string name = "")
        : Base(std::move(name))
        , capacitance_(capacitance)
        , v_prev_(initial_voltage)
        , i_prev_(0.0) {}

    /// Stamp implementation using Trapezoidal companion model
    /// Correct formula: I_eq = (2C/dt) * V_n - (2C/dt) * V_{n-1} - I_{n-1}
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Trapezoidal: G_eq = 2C/dt
        const Scalar g_eq = 2.0 * capacitance_ / dt_;

        // Equivalent current source: I_eq = G_eq * V_{n-1} + I_{n-1}
        const Scalar i_eq = g_eq * v_prev_ + i_prev_;

        // Stamp conductance
        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g_eq;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g_eq;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g_eq;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g_eq;
            }
        }

        // Stamp equivalent current source
        if (n_plus >= 0) {
            b[n_plus] += i_eq;
        }
        if (n_minus >= 0) {
            b[n_minus] -= i_eq;
        }
    }

    /// Update history after accepted timestep
    void update_history_impl() {
        // Store current values for next step
        // Note: v_current and i_current must be set by the solver after each step
        v_prev_ = v_current_;
        i_prev_ = i_current_;
    }

    /// Set current state (called by solver after each Newton iteration)
    void set_current_state(Scalar v, Scalar i) {
        v_current_ = v;
        i_current_ = i;
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar capacitance() const { return capacitance_; }
    [[nodiscard]] Scalar voltage_prev() const { return v_prev_; }
    [[nodiscard]] Scalar current_prev() const { return i_prev_; }

private:
    Scalar capacitance_;
    Scalar v_prev_;      // Voltage at previous timestep
    Scalar i_prev_;      // Current at previous timestep (CRITICAL for Trapezoidal!)
    Scalar v_current_ = 0.0;  // Current voltage (set by solver)
    Scalar i_current_ = 0.0;  // Current current (set by solver)
};

template<>
struct device_traits<Capacitor> {
    static constexpr DeviceType type = DeviceType::Capacitor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

// =============================================================================
// Example: Inductor Device (CRTP with dynamics)
// =============================================================================

class Inductor : public DynamicDeviceBase<Inductor> {
public:
    using Base = DynamicDeviceBase<Inductor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    struct Params {
        Scalar inductance = 1e-3;
        Scalar initial_current = 0.0;
    };

    explicit Inductor(Scalar inductance, Scalar initial_current = 0.0, std::string name = "")
        : Base(std::move(name))
        , inductance_(inductance)
        , i_prev_(initial_current)
        , v_prev_(0.0) {}

    /// Stamp implementation using Trapezoidal companion model
    /// For inductor: V = L * dI/dt
    /// Trapezoidal: V_n = (2L/dt) * I_n - (2L/dt) * I_{n-1} - V_{n-1}
    /// Companion model is a resistor R_eq = 2L/dt in series with voltage V_eq
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Trapezoidal: R_eq = 2L/dt (conductance g_eq = dt/(2L))
        const Scalar g_eq = dt_ / (2.0 * inductance_);

        // Equivalent voltage: V_eq = (2L/dt) * I_{n-1} + V_{n-1}
        const Scalar v_eq = (2.0 * inductance_ / dt_) * i_prev_ + v_prev_;

        // Stamp conductance
        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g_eq;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g_eq;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g_eq;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g_eq;
            }
        }

        // Stamp equivalent current source (from V_eq through g_eq)
        const Scalar i_eq = g_eq * v_eq;
        if (n_plus >= 0) {
            b[n_plus] += i_eq;
        }
        if (n_minus >= 0) {
            b[n_minus] -= i_eq;
        }
    }

    void update_history_impl() {
        i_prev_ = i_current_;
        v_prev_ = v_current_;
    }

    void set_current_state(Scalar v, Scalar i) {
        v_current_ = v;
        i_current_ = i;
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar inductance() const { return inductance_; }
    [[nodiscard]] Scalar current_prev() const { return i_prev_; }
    [[nodiscard]] Scalar voltage_prev() const { return v_prev_; }

private:
    Scalar inductance_;
    Scalar i_prev_;
    Scalar v_prev_;
    Scalar i_current_ = 0.0;
    Scalar v_current_ = 0.0;
};

template<>
struct device_traits<Inductor> {
    static constexpr DeviceType type = DeviceType::Inductor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

// =============================================================================
// Example: Voltage Source Device (CRTP)
// =============================================================================

class VoltageSource : public LinearDeviceBase<VoltageSource> {
public:
    using Base = LinearDeviceBase<VoltageSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::VoltageSource);

    struct Params {
        Scalar voltage = 0.0;
    };

    explicit VoltageSource(Scalar voltage, std::string name = "")
        : Base(std::move(name)), voltage_(voltage), branch_index_(-1) {}

    /// Set the branch index (MNA requires extra row/col for voltage sources)
    void set_branch_index(NodeIndex idx) { branch_index_ = idx; }
    [[nodiscard]] NodeIndex branch_index() const { return branch_index_; }

    /// Stamp implementation
    /// MNA formulation: adds equation V+ - V- = V_source
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2 || branch_index_ < 0) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const NodeIndex br = branch_index_;

        // Stamp the MNA extension for voltage source
        // Row br: V+ - V- = V_source
        // Also affects KCL: current flows from n+ to n-
        if (n_plus >= 0) {
            G.coeffRef(n_plus, br) += 1.0;
            G.coeffRef(br, n_plus) += 1.0;
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, br) -= 1.0;
            G.coeffRef(br, n_minus) -= 1.0;
        }

        // RHS: voltage value
        b[br] = voltage_;
    }

    static constexpr auto jacobian_pattern_impl() {
        // Voltage source adds 4 entries plus diagonal
        return StaticSparsityPattern<5>{{
            JacobianEntry{0, 2},  // n+ to branch
            JacobianEntry{1, 2},  // n- to branch
            JacobianEntry{2, 0},  // branch to n+
            JacobianEntry{2, 1},  // branch to n-
            JacobianEntry{2, 2}   // branch diagonal (zero but pattern exists)
        }};
    }

    [[nodiscard]] Scalar voltage() const { return voltage_; }
    void set_voltage(Scalar v) { voltage_ = v; }

private:
    Scalar voltage_;
    NodeIndex branch_index_;
};

template<>
struct device_traits<VoltageSource> {
    static constexpr DeviceType type = DeviceType::VoltageSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 1;  // Branch current
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 5;
};

// =============================================================================
// Example: Current Source Device (CRTP)
// =============================================================================

class CurrentSource : public LinearDeviceBase<CurrentSource> {
public:
    using Base = LinearDeviceBase<CurrentSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::CurrentSource);

    struct Params {
        Scalar current = 0.0;
    };

    explicit CurrentSource(Scalar current, std::string name = "")
        : Base(std::move(name)), current_(current) {}

    /// Stamp implementation - current source only affects RHS
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& /*G*/, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Current flows from n+ to n- (conventional)
        // KCL: current entering n+ is positive
        if (n_plus >= 0) {
            b[n_plus] -= current_;  // Current leaves n+
        }
        if (n_minus >= 0) {
            b[n_minus] += current_;  // Current enters n-
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        // Current source doesn't affect G matrix, only RHS
        return StaticSparsityPattern<0>{{}};
    }

    [[nodiscard]] Scalar current() const { return current_; }
    void set_current(Scalar i) { current_ = i; }

private:
    Scalar current_;
};

template<>
struct device_traits<CurrentSource> {
    static constexpr DeviceType type = DeviceType::CurrentSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 0;
};

// =============================================================================
// Example: Ideal Diode Device (CRTP - Nonlinear)
// =============================================================================

class IdealDiode : public NonlinearDeviceBase<IdealDiode> {
public:
    using Base = NonlinearDeviceBase<IdealDiode>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Diode);

    struct Params {
        Scalar g_on = 1e3;     // On-state conductance (1/R_on)
        Scalar g_off = 1e-9;   // Off-state conductance (leakage)
        Scalar v_threshold = 0.0;  // Threshold voltage (0 for ideal)
        Scalar v_smooth = 0.1;     // Smoothing voltage for transition (0 = sharp)
    };

    explicit IdealDiode(Scalar g_on = 1e3, Scalar g_off = 1e-9, std::string name = "")
        : Base(std::move(name)), g_on_(g_on), g_off_(g_off), v_smooth_(0.1), is_on_(false) {}

    /// Set smoothing voltage (higher = smoother transition, 0 = sharp)
    void set_smoothing(Scalar v_smooth) { v_smooth_ = v_smooth; }
    [[nodiscard]] Scalar smoothing() const { return v_smooth_; }

    /// Stamp Jacobian for Newton iteration with smooth transition
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];   // Anode
        const NodeIndex n_minus = nodes[1];  // Cathode

        // Get voltage across diode
        Scalar v_anode = (n_plus >= 0) ? x[n_plus] : 0.0;
        Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : 0.0;
        Scalar v_diode = v_anode - v_cathode;

        Scalar g, i_diode, dg_dv;

        if (v_smooth_ > 0) {
            // Smooth transition using tanh
            // g varies smoothly from g_off to g_on as v_diode goes from negative to positive
            Scalar alpha = std::tanh(v_diode / v_smooth_);  // ranges from -1 to 1
            Scalar g_avg = (g_on_ + g_off_) / 2.0;
            Scalar g_diff = (g_on_ - g_off_) / 2.0;
            g = g_avg + g_diff * alpha;

            // Derivative of conductance w.r.t. voltage (for improved Jacobian)
            Scalar dalpha_dv = (1.0 - alpha * alpha) / v_smooth_;  // derivative of tanh
            dg_dv = g_diff * dalpha_dv;

            // Current: i = g * v
            i_diode = g * v_diode;

            // Update state indicator
            is_on_ = (alpha > 0);
        } else {
            // Sharp transition (original behavior)
            is_on_ = (v_diode > 0.0);
            g = is_on_ ? g_on_ : g_off_;
            dg_dv = 0.0;
            i_diode = g * v_diode;
        }

        // Stamp conductance (with additional term from dg/dv for smooth model)
        // The Jacobian of i = g(v)*v is: di/dv = g + v*dg/dv
        Scalar J_diag = g + v_diode * dg_dv;

        if (n_plus >= 0) {
            J.coeffRef(n_plus, n_plus) += J_diag;
            if (n_minus >= 0) {
                J.coeffRef(n_plus, n_minus) -= J_diag;
            }
        }
        if (n_minus >= 0) {
            J.coeffRef(n_minus, n_minus) += J_diag;
            if (n_plus >= 0) {
                J.coeffRef(n_minus, n_plus) -= J_diag;
            }
        }

        // Update residual (KCL contribution)
        if (n_plus >= 0) {
            f[n_plus] += i_diode;
        }
        if (n_minus >= 0) {
            f[n_minus] -= i_diode;
        }
    }

    /// Regular stamp for linear-like behavior
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        Scalar g = is_on_ ? g_on_ : g_off_;

        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g;
            }
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] bool is_conducting() const { return is_on_; }

private:
    Scalar g_on_;
    Scalar g_off_;
    Scalar v_smooth_;  // Smoothing voltage (0 = sharp transition)
    mutable bool is_on_;
};

template<>
struct device_traits<IdealDiode> {
    static constexpr DeviceType type = DeviceType::Diode;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear!
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

// =============================================================================
// Example: Ideal Switch Device (CRTP)
// =============================================================================

class IdealSwitch : public LinearDeviceBase<IdealSwitch> {
public:
    using Base = LinearDeviceBase<IdealSwitch>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Switch);

    struct Params {
        Scalar g_on = 1e6;     // On-state conductance
        Scalar g_off = 1e-12;  // Off-state conductance
        bool initial_state = false;
    };

    explicit IdealSwitch(Scalar g_on = 1e6, Scalar g_off = 1e-12, bool closed = false, std::string name = "")
        : Base(std::move(name)), g_on_(g_on), g_off_(g_off), is_closed_(closed) {}

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        Scalar g = is_closed_ ? g_on_ : g_off_;

        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g;
            }
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    void close() { is_closed_ = true; }
    void open() { is_closed_ = false; }
    void set_state(bool closed) { is_closed_ = closed; }
    [[nodiscard]] bool is_closed() const { return is_closed_; }

private:
    Scalar g_on_;
    Scalar g_off_;
    bool is_closed_;
};

template<>
struct device_traits<IdealSwitch> {
    static constexpr DeviceType type = DeviceType::Switch;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;  // Piecewise linear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

// =============================================================================
// Voltage Controlled Switch (3-terminal: control, t1, t2)
// =============================================================================

/// Voltage-controlled switch - ON when V(control) > threshold
/// Ideal for PWM-driven switching applications (Buck, etc.)
/// Terminals: Control (0), Terminal1 (1), Terminal2 (2)
class VoltageControlledSwitch : public NonlinearDeviceBase<VoltageControlledSwitch> {
public:
    using Base = NonlinearDeviceBase<VoltageControlledSwitch>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::Switch);

    struct Params {
        Scalar v_threshold = 2.5;  // Threshold voltage
        Scalar g_on = 1e3;         // On-state conductance (1mΩ)
        Scalar g_off = 1e-9;       // Off-state conductance
        Scalar hysteresis = 0.1;   // Hysteresis for smooth transition
    };

    explicit VoltageControlledSwitch(Scalar v_th = 2.5, Scalar g_on = 1e3, Scalar g_off = 1e-9,
                                     std::string name = "")
        : Base(std::move(name)), v_threshold_(v_th), g_on_(g_on), g_off_(g_off) {}

    explicit VoltageControlledSwitch(Params params, std::string name = "")
        : Base(std::move(name)), v_threshold_(params.v_threshold),
          g_on_(params.g_on), g_off_(params.g_off), hysteresis_(params.hysteresis) {}

    /// Stamp Jacobian for Newton iteration
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;

        const NodeIndex n_ctrl = nodes[0];
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        // Get voltages
        Scalar v_ctrl = (n_ctrl >= 0) ? x[n_ctrl] : 0.0;
        Scalar v_t1 = (n_t1 >= 0) ? x[n_t1] : 0.0;
        Scalar v_t2 = (n_t2 >= 0) ? x[n_t2] : 0.0;

        // Smooth switching function using tanh
        // g = g_off + (g_on - g_off) * sigmoid((v_ctrl - v_th) / hysteresis)
        Scalar v_norm = (v_ctrl - v_threshold_) / hysteresis_;
        Scalar sigmoid = 0.5 * (1.0 + std::tanh(v_norm));
        Scalar g = g_off_ + (g_on_ - g_off_) * sigmoid;

        // Derivative of g with respect to v_ctrl
        Scalar dsigmoid = 0.5 / hysteresis_ * (1.0 - std::tanh(v_norm) * std::tanh(v_norm));
        Scalar dg_dvctrl = (g_on_ - g_off_) * dsigmoid;

        // Current through switch: i = g * (v_t1 - v_t2)
        Scalar v_sw = v_t1 - v_t2;
        Scalar i_sw = g * v_sw;

        // Stamp conductance between t1 and t2
        if (n_t1 >= 0) {
            J.coeffRef(n_t1, n_t1) += g;
            if (n_t2 >= 0) J.coeffRef(n_t1, n_t2) -= g;
            // Derivative w.r.t. control voltage
            if (n_ctrl >= 0) J.coeffRef(n_t1, n_ctrl) += dg_dvctrl * v_sw;
        }
        if (n_t2 >= 0) {
            J.coeffRef(n_t2, n_t2) += g;
            if (n_t1 >= 0) J.coeffRef(n_t2, n_t1) -= g;
            // Derivative w.r.t. control voltage
            if (n_ctrl >= 0) J.coeffRef(n_t2, n_ctrl) -= dg_dvctrl * v_sw;
        }

        // Current residuals
        Scalar i_eq = i_sw - g * v_sw - dg_dvctrl * v_sw * v_ctrl;  // Linearized at operating point
        // Actually for Newton: f = i_calculated, and we subtract Jacobian*x
        // Residual contribution: f[n] += i_into_node
        if (n_t1 >= 0) f[n_t1] += i_sw;
        if (n_t2 >= 0) f[n_t2] -= i_sw;
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        // For DC/initial: assume switch is OFF (safe starting point)
        if (nodes.size() < 3) return;
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        Scalar g = g_off_;
        if (n_t1 >= 0) {
            G.coeffRef(n_t1, n_t1) += g;
            if (n_t2 >= 0) G.coeffRef(n_t1, n_t2) -= g;
        }
        if (n_t2 >= 0) {
            G.coeffRef(n_t2, n_t2) += g;
            if (n_t1 >= 0) G.coeffRef(n_t2, n_t1) -= g;
        }
    }

    [[nodiscard]] Scalar v_threshold() const { return v_threshold_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }

private:
    Scalar v_threshold_ = 2.5;
    Scalar g_on_ = 1e3;
    Scalar g_off_ = 1e-9;
    Scalar hysteresis_ = 0.5;  // Wider hysteresis for smoother convergence
};

template<>
struct device_traits<VoltageControlledSwitch> {
    static constexpr DeviceType type = DeviceType::Switch;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear (depends on control voltage)
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 9;  // 3x3
};

// =============================================================================
// MOSFET Device (CRTP - Nonlinear, 3-terminal)
// =============================================================================

/// MOSFET Level 1 model (Shichman-Hodges)
/// Terminals: Gate (0), Drain (1), Source (2)
class MOSFET : public NonlinearDeviceBase<MOSFET> {
public:
    using Base = NonlinearDeviceBase<MOSFET>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::MOSFET);

    struct Params {
        Scalar vth = 2.0;           // Threshold voltage (V)
        Scalar kp = 0.1;            // Transconductance parameter (A/V^2)
        Scalar lambda = 0.01;       // Channel-length modulation (1/V)
        Scalar g_off = 1e-12;       // Off-state conductance
        bool is_nmos = true;      // NMOS if true, PMOS if false
    };

    explicit MOSFET(std::string name = "")
        : Base(std::move(name)), params_() {}

    explicit MOSFET(Params params, std::string name)
        : Base(std::move(name)), params_(params) {}

    explicit MOSFET(Scalar vth, Scalar kp, bool is_nmos = true, std::string name = "")
        : Base(std::move(name))
        , params_{vth, kp, 0.01, 1e-12, is_nmos} {}

    /// Stamp Jacobian for Newton iteration
    /// Implements Level 1 MOSFET equations
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;

        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        // Get terminal voltages
        Scalar vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Scalar vd = (n_drain >= 0) ? x[n_drain] : 0.0;
        Scalar vs = (n_source >= 0) ? x[n_source] : 0.0;

        // For PMOS, negate voltages
        Scalar sign = params_.is_nmos ? 1.0 : -1.0;
        Scalar vgs = sign * (vg - vs);
        Scalar vds = sign * (vd - vs);

        Scalar id = 0.0;      // Drain current
        Scalar gm = 0.0;      // Transconductance dId/dVgs
        Scalar gds = 0.0;     // Output conductance dId/dVds

        Scalar vth = params_.vth;
        Scalar kp = params_.kp;
        Scalar lambda = params_.lambda;

        if (vgs <= vth) {
            // Cutoff region
            id = params_.g_off * vds;
            gds = params_.g_off;
        } else if (vds < vgs - vth) {
            // Linear (triode) region
            Scalar vov = vgs - vth;
            id = kp * (vov * vds - 0.5 * vds * vds) * (1.0 + lambda * vds);
            gm = kp * vds * (1.0 + lambda * vds);
            gds = kp * (vov - vds) * (1.0 + lambda * vds) + kp * (vov * vds - 0.5 * vds * vds) * lambda;
        } else {
            // Saturation region
            Scalar vov = vgs - vth;
            id = 0.5 * kp * vov * vov * (1.0 + lambda * vds);
            gm = kp * vov * (1.0 + lambda * vds);
            gds = 0.5 * kp * vov * vov * lambda;
        }

        // Apply sign for PMOS
        id *= sign;

        // Stamp Jacobian (Norton equivalent)
        // I_eq = id - gm * vgs - gds * vds
        Scalar i_eq = id - gm * vgs - gds * vds;

        // Conductance stamps: drain-source path
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += gds;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= gds;
            if (n_gate >= 0) J.coeffRef(n_drain, n_gate) += gm;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= gm;  // gm contribution
        }
        if (n_source >= 0) {
            J.coeffRef(n_source, n_source) += gds;
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= gds;
            if (n_gate >= 0) J.coeffRef(n_source, n_gate) -= gm;
            J.coeffRef(n_source, n_source) += gm;  // gm contribution
        }

        // Current source stamps
        if (n_drain >= 0) f[n_drain] -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        // For initial guess, stamp small conductance
        if (nodes.size() < 3) return;
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        if (n_drain >= 0) {
            G.coeffRef(n_drain, n_drain) += params_.g_off;
            if (n_source >= 0) G.coeffRef(n_drain, n_source) -= params_.g_off;
        }
        if (n_source >= 0) {
            G.coeffRef(n_source, n_source) += params_.g_off;
            if (n_drain >= 0) G.coeffRef(n_source, n_drain) -= params_.g_off;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        // 3x3 = 9 entries max, but we mainly use D-S path
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] const Params& params() const { return params_; }

private:
    Params params_;
};

template<>
struct device_traits<MOSFET> {
    static constexpr DeviceType type = DeviceType::MOSFET;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr std::size_t jacobian_size = 9;
};

// =============================================================================
// IGBT Device (CRTP - Nonlinear, 3-terminal)
// =============================================================================

/// Simplified IGBT model for power electronics
/// Terminals: Gate (0), Collector (1), Emitter (2)
class IGBT : public NonlinearDeviceBase<IGBT> {
public:
    using Base = NonlinearDeviceBase<IGBT>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::IGBT);

    struct Params {
        Scalar vth = 5.0;           // Gate threshold voltage (V)
        Scalar g_on = 1e4;          // On-state conductance (S)
        Scalar g_off = 1e-12;       // Off-state conductance (S)
        Scalar v_ce_sat = 1.5;      // Collector-emitter saturation voltage (V)
    };

    explicit IGBT(std::string name = "")
        : Base(std::move(name)), params_(), is_on_(false) {}

    explicit IGBT(Params params, std::string name)
        : Base(std::move(name)), params_(params), is_on_(false) {}

    explicit IGBT(Scalar vth, Scalar g_on = 1e4, std::string name = "")
        : Base(std::move(name))
        , params_{vth, g_on, 1e-12, 1.5}
        , is_on_(false) {}

    /// Stamp Jacobian for Newton iteration
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;

        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        Scalar vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Scalar vc = (n_collector >= 0) ? x[n_collector] : 0.0;
        Scalar ve = (n_emitter >= 0) ? x[n_emitter] : 0.0;

        Scalar vge = vg - ve;
        Scalar vce = vc - ve;

        // Determine state
        is_on_ = (vge > params_.vth) && (vce > 0);
        Scalar g = is_on_ ? params_.g_on : params_.g_off;

        // Model as voltage-controlled conductance with saturation
        Scalar ic = g * vce;
        if (is_on_ && vce > params_.v_ce_sat) {
            // Add forward voltage drop
            ic = g * (vce - params_.v_ce_sat) + params_.g_on * params_.v_ce_sat;
        }

        // Stamp collector-emitter conductance
        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            J.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= g;
        }

        // Residual
        if (n_collector >= 0) f[n_collector] += ic - g * vce;
        if (n_emitter >= 0) f[n_emitter] -= ic - g * vce;
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        Scalar g = is_on_ ? params_.g_on : params_.g_off;

        if (n_collector >= 0) {
            G.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) G.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            G.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) G.coeffRef(n_emitter, n_collector) -= g;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] bool is_conducting() const { return is_on_; }
    [[nodiscard]] const Params& params() const { return params_; }

private:
    Params params_;
    mutable bool is_on_;
};

template<>
struct device_traits<IGBT> {
    static constexpr DeviceType type = DeviceType::IGBT;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr std::size_t jacobian_size = 9;
};

// =============================================================================
// Transformer Device (CRTP - Linear, 4-terminal ideal transformer)
// =============================================================================

/// Ideal transformer with turns ratio N:1
/// Terminals: Primary+ (0), Primary- (1), Secondary+ (2), Secondary- (3)
class Transformer : public LinearDeviceBase<Transformer> {
public:
    using Base = LinearDeviceBase<Transformer>;
    static constexpr std::size_t num_pins = 4;
    static constexpr int device_type = static_cast<int>(DeviceType::Transformer);

    struct Params {
        Scalar turns_ratio = 1.0;   // N:1 (primary:secondary)
        Scalar magnetizing_inductance = 1e-3;  // Lm (H), large for ideal
    };

    explicit Transformer(Scalar turns_ratio, std::string name = "")
        : Base(std::move(name))
        , turns_ratio_(turns_ratio)
        , branch_index_p_(-1)
        , branch_index_s_(-1) {}

    /// Set branch indices for MNA (primary and secondary currents)
    void set_branch_indices(Index primary, Index secondary) {
        branch_index_p_ = primary;
        branch_index_s_ = secondary;
    }

    /// Stamp implementation using coupled inductors formulation
    /// V1 = N * V2, I2 = -N * I1
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 4 || branch_index_p_ < 0 || branch_index_s_ < 0) return;

        const NodeIndex np_plus = nodes[0];   // Primary +
        const NodeIndex np_minus = nodes[1];  // Primary -
        const NodeIndex ns_plus = nodes[2];   // Secondary +
        const NodeIndex ns_minus = nodes[3];  // Secondary -
        const NodeIndex br_p = branch_index_p_;
        const NodeIndex br_s = branch_index_s_;

        Scalar n = turns_ratio_;

        // Ideal transformer equations:
        // V1 - N*V2 = 0  (voltage relation)
        // I1 + N*I2 = 0  (current relation, power conservation)

        // Primary winding: branch equation V_p+ - V_p- = L*di_p/dt
        // For ideal: V_p+ - V_p- = N * (V_s+ - V_s-)
        if (np_plus >= 0) {
            G.coeffRef(np_plus, br_p) += 1.0;
            G.coeffRef(br_p, np_plus) += 1.0;
        }
        if (np_minus >= 0) {
            G.coeffRef(np_minus, br_p) -= 1.0;
            G.coeffRef(br_p, np_minus) -= 1.0;
        }

        // Secondary winding coupling
        if (ns_plus >= 0) {
            G.coeffRef(ns_plus, br_s) += 1.0;
            G.coeffRef(br_p, ns_plus) -= n;  // Coupling: V_p = n * V_s
        }
        if (ns_minus >= 0) {
            G.coeffRef(ns_minus, br_s) -= 1.0;
            G.coeffRef(br_p, ns_minus) += n;
        }

        // Secondary branch equation
        if (ns_plus >= 0) G.coeffRef(br_s, ns_plus) += 1.0;
        if (ns_minus >= 0) G.coeffRef(br_s, ns_minus) -= 1.0;

        // Current relationship: I_p + n * I_s = 0
        G.coeffRef(br_s, br_p) += n;
        G.coeffRef(br_s, br_s) += 0.0;  // Placeholder

        // No RHS contribution for ideal transformer
        (void)b;
    }

    static constexpr auto jacobian_pattern_impl() {
        // 6x6 contributions for 4 nodes + 2 branches
        return StaticSparsityPattern<16>{{
            JacobianEntry{0, 4}, JacobianEntry{4, 0},  // np+ <-> br_p
            JacobianEntry{1, 4}, JacobianEntry{4, 1},  // np- <-> br_p
            JacobianEntry{2, 5}, JacobianEntry{4, 2},  // ns+ <-> br_s, coupling
            JacobianEntry{3, 5}, JacobianEntry{4, 3},  // ns- <-> br_s, coupling
            JacobianEntry{5, 2}, JacobianEntry{5, 3},  // br_s <-> secondary nodes
            JacobianEntry{5, 4}, JacobianEntry{5, 5},  // br_s <-> currents
            JacobianEntry{0, 0}, JacobianEntry{1, 1},  // Diagonal placeholders
            JacobianEntry{2, 2}, JacobianEntry{3, 3}
        }};
    }

    [[nodiscard]] Scalar turns_ratio() const { return turns_ratio_; }

private:
    Scalar turns_ratio_;
    Index branch_index_p_;
    Index branch_index_s_;
};

template<>
struct device_traits<Transformer> {
    static constexpr DeviceType type = DeviceType::Transformer;
    static constexpr std::size_t num_pins = 4;
    static constexpr std::size_t num_internal_nodes = 2;  // Two branch currents
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;  // Ideal transformer (no Lm dynamics here)
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 16;
};

// =============================================================================
// Static Assertions to Verify Concepts
// =============================================================================

static_assert(StampableDevice<Resistor>, "Resistor must satisfy StampableDevice concept");
static_assert(is_linear_device_v<Resistor>, "Resistor must be linear");
static_assert(!is_dynamic_device_v<Resistor>, "Resistor must not be dynamic");

static_assert(StampableDevice<Capacitor>, "Capacitor must satisfy StampableDevice concept");
static_assert(is_linear_device_v<Capacitor>, "Capacitor must be linear");
static_assert(is_dynamic_device_v<Capacitor>, "Capacitor must be dynamic");

static_assert(StampableDevice<MOSFET>, "MOSFET must satisfy StampableDevice concept");
static_assert(!is_linear_device_v<MOSFET>, "MOSFET must be nonlinear");

static_assert(StampableDevice<IGBT>, "IGBT must satisfy StampableDevice concept");
static_assert(!is_linear_device_v<IGBT>, "IGBT must be nonlinear");

static_assert(StampableDevice<Transformer>, "Transformer must satisfy StampableDevice concept");
static_assert(is_linear_device_v<Transformer>, "Transformer must be linear");

// =============================================================================
// Device Registration for Runtime Introspection (C++26 Reflection Prep)
// =============================================================================
// These macros register device metadata for runtime introspection.
// In C++26, this will be replaced by static reflection.

PULSIM_REGISTER_DEVICE(Resistor, "Resistor", "passive", 2, true, false, false);
PULSIM_REGISTER_DEVICE(Capacitor, "Capacitor", "passive", 2, true, true, false);
PULSIM_REGISTER_DEVICE(Inductor, "Inductor", "passive", 2, true, true, false);
PULSIM_REGISTER_DEVICE(VoltageSource, "VoltageSource", "source", 2, true, false, false);
PULSIM_REGISTER_DEVICE(CurrentSource, "CurrentSource", "source", 2, true, false, false);
PULSIM_REGISTER_DEVICE(IdealDiode, "IdealDiode", "active", 2, false, false, false);
PULSIM_REGISTER_DEVICE(IdealSwitch, "IdealSwitch", "switch", 2, true, false, false);
PULSIM_REGISTER_DEVICE(MOSFET, "MOSFET", "active", 3, false, false, true);
PULSIM_REGISTER_DEVICE(IGBT, "IGBT", "active", 3, false, false, true);
PULSIM_REGISTER_DEVICE(Transformer, "Transformer", "passive", 4, true, false, false);

// Register device parameters for introspection
PULSIM_REGISTER_PARAMS(Resistor,
    PULSIM_PARAM("resistance", "Ohm", 1000.0, 0.0, 1e12)
);

PULSIM_REGISTER_PARAMS(Capacitor,
    PULSIM_PARAM("capacitance", "F", 1e-6, 0.0, 1e3),
    PULSIM_PARAM("initial_voltage", "V", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(Inductor,
    PULSIM_PARAM("inductance", "H", 1e-3, 0.0, 1e3),
    PULSIM_PARAM("initial_current", "A", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(VoltageSource,
    PULSIM_PARAM("voltage", "V", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(CurrentSource,
    PULSIM_PARAM("current", "A", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(MOSFET,
    PULSIM_PARAM("vth", "V", 2.0, -10.0, 10.0),
    PULSIM_PARAM("kp", "A/V^2", 0.1, 0.0, 100.0),
    PULSIM_PARAM("lambda", "1/V", 0.01, 0.0, 1.0)
);

PULSIM_REGISTER_PARAMS(IGBT,
    PULSIM_PARAM("vth", "V", 5.0, 0.0, 20.0),
    PULSIM_PARAM("g_on", "S", 1e4, 0.0, 1e6),
    PULSIM_PARAM("v_ce_sat", "V", 1.5, 0.0, 10.0)
);

PULSIM_REGISTER_PARAMS(Transformer,
    PULSIM_PARAM("turns_ratio", "", 1.0, 0.001, 1000.0)
);

}  // namespace pulsim::v1
