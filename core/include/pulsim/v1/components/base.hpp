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
#include <span>
#include <string>
#include <utility>

namespace pulsim::v1 {

// Scalar is alias to Real (double by default) for device implementations
using Scalar = Real;
using NodeIndex = Index;

// Matrix types for MNA system
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

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
