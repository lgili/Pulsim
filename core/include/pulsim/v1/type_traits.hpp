#pragma once

// =============================================================================
// PulsimCore v2 - Type Traits for Compile-Time Introspection
// =============================================================================
// This header provides type traits for device models and solvers that enable:
// - Compile-time device property queries
// - Static dispatch optimizations
// - Jacobian sparsity pattern analysis
// =============================================================================

#include <type_traits>
#include <cstddef>
#include <array>

namespace pulsim::v1 {

// =============================================================================
// Device Type Enumeration
// =============================================================================

enum class DeviceType : int {
    Resistor = 0,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    Switch,
    MOSFET,
    IGBT,
    Transformer,
    VCVS,  // Voltage-Controlled Voltage Source
    VCCS,  // Voltage-Controlled Current Source
    CCVS,  // Current-Controlled Voltage Source
    CCCS,  // Current-Controlled Current Source
    // Add more as needed
    Unknown = -1
};

// =============================================================================
// Device Traits - Primary Template
// =============================================================================

/// Primary template - specializations provide device-specific information
template<typename Device>
struct device_traits {
    static constexpr DeviceType type = DeviceType::Unknown;
    static constexpr std::size_t num_pins = 0;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;

    // Jacobian contribution size (nodes + internal + branches)
    static constexpr std::size_t jacobian_size = 0;
};

// =============================================================================
// Compile-Time Device Property Queries
// =============================================================================

/// Check if a device type is linear
template<typename D>
inline constexpr bool is_linear_device_v = device_traits<D>::is_linear;

/// Check if a device type has dynamics (capacitors, inductors)
template<typename D>
inline constexpr bool is_dynamic_device_v = device_traits<D>::is_dynamic;

/// Check if a device has a loss model
template<typename D>
inline constexpr bool has_loss_model_v = device_traits<D>::has_loss_model;

/// Check if a device has thermal coupling
template<typename D>
inline constexpr bool has_thermal_model_v = device_traits<D>::has_thermal_model;

/// Get the number of pins for a device type
template<typename D>
inline constexpr std::size_t num_pins_v = device_traits<D>::num_pins;

/// Get the Jacobian contribution size for a device type
template<typename D>
inline constexpr std::size_t jacobian_size_v = device_traits<D>::jacobian_size;

// =============================================================================
// Sparsity Pattern Types
// =============================================================================

/// Represents a single non-zero entry position in the Jacobian
struct JacobianEntry {
    int row;
    int col;

    constexpr JacobianEntry(int r, int c) : row(r), col(c) {}

    constexpr bool operator==(const JacobianEntry& other) const {
        return row == other.row && col == other.col;
    }
};

/// Compile-time fixed-size sparsity pattern
template<std::size_t N>
struct StaticSparsityPattern {
    std::array<JacobianEntry, N> entries;
    std::size_t count = N;

    constexpr StaticSparsityPattern(std::array<JacobianEntry, N> e) : entries(e) {}

    constexpr auto begin() const { return entries.begin(); }
    constexpr auto end() const { return entries.end(); }
    constexpr std::size_t size() const { return count; }
};

// =============================================================================
// Integration Method Traits
// =============================================================================

enum class MethodType {
    BackwardEuler,
    Trapezoidal,
    BDF2,
    BDF3,
    BDF4,
    BDF5,
    GEAR2  // Alias for Trapezoidal
};

template<MethodType Method>
struct method_traits {
    static constexpr int order = 1;
    static constexpr bool is_A_stable = true;
    static constexpr bool is_L_stable = true;
};

template<>
struct method_traits<MethodType::BackwardEuler> {
    static constexpr int order = 1;
    static constexpr bool is_A_stable = true;
    static constexpr bool is_L_stable = true;
};

template<>
struct method_traits<MethodType::Trapezoidal> {
    static constexpr int order = 2;
    static constexpr bool is_A_stable = true;
    static constexpr bool is_L_stable = false;  // Not L-stable (oscillates on stiff problems)
};

template<>
struct method_traits<MethodType::BDF2> {
    static constexpr int order = 2;
    static constexpr bool is_A_stable = true;
    static constexpr bool is_L_stable = true;
};

template<>
struct method_traits<MethodType::GEAR2> : method_traits<MethodType::Trapezoidal> {};

// =============================================================================
// Solver Policy Traits
// =============================================================================

template<typename LinearSolver>
struct linear_solver_traits {
    static constexpr bool supports_factorization_reuse = false;
    static constexpr bool supports_symbolic_analysis = false;
    static constexpr bool is_direct = true;
};

template<typename NewtonSolver>
struct newton_solver_traits {
    static constexpr bool supports_line_search = false;
    static constexpr bool supports_continuation = false;
    static constexpr bool supports_trust_region = false;
};

// =============================================================================
// Type List Utilities for Device Collections
// =============================================================================

/// Empty type list
struct TypeListEnd {};

/// Type list for compile-time device collections
template<typename Head, typename Tail = TypeListEnd>
struct TypeList {
    using head = Head;
    using tail = Tail;
};

/// Count types in a type list
template<typename TL>
struct type_list_size;

template<>
struct type_list_size<TypeListEnd> {
    static constexpr std::size_t value = 0;
};

template<typename Head, typename Tail>
struct type_list_size<TypeList<Head, Tail>> {
    static constexpr std::size_t value = 1 + type_list_size<Tail>::value;
};

template<typename TL>
inline constexpr std::size_t type_list_size_v = type_list_size<TL>::value;

/// Sum Jacobian sizes across a type list
template<typename TL>
struct total_jacobian_size;

template<>
struct total_jacobian_size<TypeListEnd> {
    static constexpr std::size_t value = 0;
};

template<typename Head, typename Tail>
struct total_jacobian_size<TypeList<Head, Tail>> {
    static constexpr std::size_t value =
        device_traits<Head>::jacobian_size + total_jacobian_size<Tail>::value;
};

template<typename TL>
inline constexpr std::size_t total_jacobian_size_v = total_jacobian_size<TL>::value;

// =============================================================================
// Compile-Time Checks
// =============================================================================

/// Check if all devices in a list are linear
template<typename TL>
struct all_linear;

template<>
struct all_linear<TypeListEnd> : std::true_type {};

template<typename Head, typename Tail>
struct all_linear<TypeList<Head, Tail>> {
    static constexpr bool value = is_linear_device_v<Head> && all_linear<Tail>::value;
};

template<typename TL>
inline constexpr bool all_linear_v = all_linear<TL>::value;

/// Check if any device in a list has dynamics
template<typename TL>
struct any_dynamic;

template<>
struct any_dynamic<TypeListEnd> : std::false_type {};

template<typename Head, typename Tail>
struct any_dynamic<TypeList<Head, Tail>> {
    static constexpr bool value = is_dynamic_device_v<Head> || any_dynamic<Tail>::value;
};

template<typename TL>
inline constexpr bool any_dynamic_v = any_dynamic<TL>::value;

}  // namespace pulsim::v1
