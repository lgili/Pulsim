#pragma once

// =============================================================================
// PulsimCore v2 - CRTP Device Base Classes
// =============================================================================

#include "pulsim/v1/concepts.hpp"
#include "pulsim/v1/type_traits.hpp"
#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/cpp23_features.hpp"

#include <Eigen/Sparse>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace pulsim::v1 {

// Scalar is alias to Real (double by default) for device implementations
using Scalar = Real;
using NodeIndex = Index;

// Matrix types for MNA system
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

// =============================================================================
// Switching Mode for piecewise-linear vs smooth nonlinear stamping
// =============================================================================
//
// Pulsim supports two stamping strategies for switching devices (diodes,
// switches, MOSFETs, IGBTs):
//
//   - Ideal       : sharp piecewise-linear (Ron / Roff) with no smoothing.
//                   Eligible for the state-space PWL segment engine that
//                   skips Newton iteration in stable topology windows.
//                   Event detection commutes the device on threshold/sign
//                   crossings (refer to refactor-pwl-switching-engine spec).
//
//   - Behavioral  : smooth nonlinear (tanh / Shichman-Hodges / saturating
//                   collector-emitter). Newton-friendly; preserves the
//                   pre-PWL behavior of the kernel.
//
//   - Auto        : kernel resolves the mode at simulation time. In this
//                   release Auto resolves to Behavioral for backward
//                   compatibility; future runtime work flips the default
//                   to Ideal whenever every switching device declares
//                   supports_pwl.
//
// Devices keep their own SwitchingMode field; users override globally via
// SimulationOptions::switching_mode (separate proposal task) or per-device
// via the device's set_switching_mode() setter.
enum class SwitchingMode : std::uint8_t {
    Auto = 0,
    Ideal = 1,
    Behavioral = 2,
};

[[nodiscard]] constexpr std::string_view to_string(SwitchingMode mode) noexcept {
    switch (mode) {
        case SwitchingMode::Auto:       return "Auto";
        case SwitchingMode::Ideal:      return "Ideal";
        case SwitchingMode::Behavioral: return "Behavioral";
    }
    return "Unknown";
}

/// Resolve a (possibly Auto) device-level mode against a circuit-level default.
/// Auto inherits from the circuit; explicit modes win over the circuit default.
/// The fallback chain is: device explicit -> circuit explicit -> Behavioral.
[[nodiscard]] constexpr SwitchingMode resolve_switching_mode(
    SwitchingMode device_mode,
    SwitchingMode circuit_default = SwitchingMode::Behavioral) noexcept {
    if (device_mode != SwitchingMode::Auto) {
        return device_mode;
    }
    if (circuit_default != SwitchingMode::Auto) {
        return circuit_default;
    }
    return SwitchingMode::Behavioral;
}

/// Context handed to a PWL device when the kernel is deciding whether the
/// device should commute (transition between on/off state) at the current
/// step boundary. Carries the device-relevant electrical observables.
struct PwlEventContext {
    Scalar voltage = 0.0;          ///< terminal voltage relevant to commutation
                                   ///< (e.g. anode-cathode for a diode).
    Scalar current = 0.0;          ///< branch current (e.g. forward current).
    Scalar control_voltage = 0.0;  ///< control / gate node voltage when applicable
                                   ///< (used by VoltageControlledSwitch / MOSFET / IGBT).
    Scalar event_hysteresis = 0.0; ///< hysteresis band that suppresses chatter
                                   ///< near commutation thresholds (volts or amps).
};

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

template<typename Derived>
class LinearDeviceBase : public DeviceBase<Derived> {
public:
    using Base = DeviceBase<Derived>;
    using typename Base::Scalar;

protected:
    using Base::Base;
};

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

}  // namespace pulsim::v1
