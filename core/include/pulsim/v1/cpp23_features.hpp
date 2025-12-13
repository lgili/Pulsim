#pragma once

// =============================================================================
// PulsimCore v2 - Advanced C++23 Features
// =============================================================================
// This header provides advanced C++23 features for high-performance simulation:
// - std::mdspan for multidimensional state arrays (1.2.3)
// - Deducing this for simplified CRTP (1.2.4)
// - Static reflection preparation for C++26 (1.2.7)
// =============================================================================

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <string_view>
#include <vector>

// Check for C++23 mdspan support
#if __has_include(<mdspan>)
    #include <mdspan>
    #define PULSIM_HAS_MDSPAN 1
#else
    #define PULSIM_HAS_MDSPAN 0
#endif

namespace pulsim::v1 {

// =============================================================================
// 1.2.3: std::mdspan for Multidimensional State Arrays
// =============================================================================

#if PULSIM_HAS_MDSPAN

/// Type alias for 2D state matrix view (time x state_variables)
template<typename T>
using StateMatrixView = std::mdspan<T, std::dextents<std::size_t, 2>>;

/// Type alias for 1D state vector view
template<typename T>
using StateVectorView = std::mdspan<T, std::dextents<std::size_t, 1>>;

/// Type alias for Jacobian matrix view (sparse, but can use dense for small circuits)
template<typename T>
using JacobianView = std::mdspan<T, std::dextents<std::size_t, 2>>;

/// Fixed-size state buffer for small circuits (compile-time size)
template<typename T, std::size_t NumStates, std::size_t HistoryDepth = 3>
class StateBuffer {
public:
    using value_type = T;
    using view_type = std::mdspan<T, std::extents<std::size_t, NumStates, HistoryDepth>>;
    using const_view_type = std::mdspan<const T, std::extents<std::size_t, NumStates, HistoryDepth>>;

    static constexpr std::size_t num_states = NumStates;
    static constexpr std::size_t history_depth = HistoryDepth;

    constexpr StateBuffer() = default;

    /// Get a view of all state history
    [[nodiscard]] constexpr view_type view() noexcept {
        return view_type(data_.data());
    }

    [[nodiscard]] constexpr const_view_type view() const noexcept {
        return const_view_type(data_.data());
    }

    /// Get current state (most recent)
    [[nodiscard]] constexpr std::span<T, NumStates> current() noexcept {
        return std::span<T, NumStates>(data_.data(), NumStates);
    }

    [[nodiscard]] constexpr std::span<const T, NumStates> current() const noexcept {
        return std::span<const T, NumStates>(data_.data(), NumStates);
    }

    /// Get state at specific history index (0 = current, 1 = previous, etc.)
    [[nodiscard]] constexpr std::span<T, NumStates> at_history(std::size_t idx) noexcept {
        return std::span<T, NumStates>(data_.data() + idx * NumStates, NumStates);
    }

    /// Shift history: move all states back by one, current becomes available for new data
    constexpr void shift_history() noexcept {
        for (std::size_t h = HistoryDepth - 1; h > 0; --h) {
            for (std::size_t s = 0; s < NumStates; ++s) {
                data_[h * NumStates + s] = data_[(h - 1) * NumStates + s];
            }
        }
    }

    /// Access element (state_index, history_index)
    [[nodiscard]] constexpr T& operator()(std::size_t state, std::size_t history = 0) noexcept {
        return data_[history * NumStates + state];
    }

    [[nodiscard]] constexpr const T& operator()(std::size_t state, std::size_t history = 0) const noexcept {
        return data_[history * NumStates + state];
    }

private:
    std::array<T, NumStates * HistoryDepth> data_{};
};

/// Dynamic-size state buffer for large circuits
template<typename T>
class DynamicStateBuffer {
public:
    using value_type = T;

    DynamicStateBuffer(std::size_t num_states, std::size_t history_depth = 3)
        : num_states_(num_states)
        , history_depth_(history_depth)
        , data_(num_states * history_depth, T{}) {}

    /// Get mdspan view of state history
    [[nodiscard]] StateMatrixView<T> view() noexcept {
        return StateMatrixView<T>(data_.data(), history_depth_, num_states_);
    }

    [[nodiscard]] StateMatrixView<const T> view() const noexcept {
        return StateMatrixView<const T>(data_.data(), history_depth_, num_states_);
    }

    /// Get current state vector
    [[nodiscard]] std::span<T> current() noexcept {
        return std::span<T>(data_.data(), num_states_);
    }

    /// Shift history
    void shift_history() {
        for (std::size_t h = history_depth_ - 1; h > 0; --h) {
            for (std::size_t s = 0; s < num_states_; ++s) {
                data_[h * num_states_ + s] = data_[(h - 1) * num_states_ + s];
            }
        }
    }

    [[nodiscard]] std::size_t num_states() const noexcept { return num_states_; }
    [[nodiscard]] std::size_t history_depth() const noexcept { return history_depth_; }

private:
    std::size_t num_states_;
    std::size_t history_depth_;
    std::vector<T> data_;
};

#else // !PULSIM_HAS_MDSPAN

// Fallback for compilers without mdspan
template<typename T, std::size_t NumStates, std::size_t HistoryDepth = 3>
class StateBuffer {
public:
    static constexpr std::size_t num_states = NumStates;
    static constexpr std::size_t history_depth = HistoryDepth;

    constexpr StateBuffer() = default;

    [[nodiscard]] constexpr std::span<T, NumStates> current() noexcept {
        return std::span<T, NumStates>(data_.data(), NumStates);
    }

    constexpr void shift_history() noexcept {
        for (std::size_t h = HistoryDepth - 1; h > 0; --h) {
            for (std::size_t s = 0; s < NumStates; ++s) {
                data_[h * NumStates + s] = data_[(h - 1) * NumStates + s];
            }
        }
    }

    [[nodiscard]] constexpr T& operator()(std::size_t state, std::size_t history = 0) noexcept {
        return data_[history * NumStates + state];
    }

private:
    std::array<T, NumStates * HistoryDepth> data_{};
};

#endif // PULSIM_HAS_MDSPAN

// =============================================================================
// 1.2.4: Deducing This for CRTP Base Classes
// =============================================================================
// C++23 "deducing this" allows us to simplify CRTP patterns by avoiding
// the static_cast<Derived*>(this) dance.

/// CRTP base using deducing this (C++23)
/// Example:
///   class MyDevice : public DeducingCRTPBase<MyDevice> {
///       void do_work_impl() { ... }
///   };
///
///   MyDevice d;
///   d.do_work(); // Automatically calls do_work_impl() on MyDevice
///
template<typename Derived>
class DeducingCRTPBase {
public:
    // Using deducing this (C++23 feature)
    // The 'this auto&&' syntax allows the compiler to deduce the exact type

    /// Stamp device into MNA matrix - uses deducing this
    template<typename Matrix, typename Vec>
    void stamp(this auto&& self, Matrix& G, Vec& b, std::span<const int> nodes) {
        // Forward to derived implementation
        self.stamp_impl(G, b, nodes);
    }

    /// Get device name - uses deducing this for const-correctness
    [[nodiscard]] auto name(this auto&& self) -> decltype(auto) {
        return self.name_;
    }

    /// Clone pattern using deducing this
    [[nodiscard]] auto clone(this auto&& self) {
        using SelfType = std::remove_cvref_t<decltype(self)>;
        return SelfType(self);
    }

protected:
    DeducingCRTPBase() = default;
    ~DeducingCRTPBase() = default;

    std::string name_;
};

/// Mixin for adding chainable setters using deducing this
template<typename Derived>
class ChainableSetters {
public:
    /// Set name with chaining - returns reference to derived type
    auto& set_name(this auto&& self, std::string name) {
        self.name_ = std::move(name);
        return self;
    }

    /// Set value with chaining
    template<typename T>
    auto& set_value(this auto&& self, T&& value) {
        self.value_ = std::forward<T>(value);
        return self;
    }
};

// =============================================================================
// 1.2.7: Static Reflection Preparation (C++26)
// =============================================================================
// Prepare infrastructure for C++26 static reflection. For now, we use
// manual registration macros that can be replaced with reflection later.

/// Device metadata for reflection-like introspection
struct DeviceMetadata {
    std::string_view name;
    std::string_view category;      // "passive", "active", "source", "switch"
    std::size_t pin_count;
    bool is_linear;
    bool is_dynamic;
    bool has_thermal_model;
};

/// Compile-time device registry entry
template<typename Device>
struct DeviceRegistry {
    static constexpr DeviceMetadata metadata = {
        .name = "Unknown",
        .category = "unknown",
        .pin_count = 0,
        .is_linear = false,
        .is_dynamic = false,
        .has_thermal_model = false
    };
};

/// Macro to register device metadata (will be replaced by reflection in C++26)
#define PULSIM_REGISTER_DEVICE(DeviceType, DeviceName, Category, Pins, Linear, Dynamic, Thermal) \
    template<> \
    struct DeviceRegistry<DeviceType> { \
        static constexpr DeviceMetadata metadata = { \
            .name = DeviceName, \
            .category = Category, \
            .pin_count = Pins, \
            .is_linear = Linear, \
            .is_dynamic = Dynamic, \
            .has_thermal_model = Thermal \
        }; \
    }

/// Get device metadata at compile time
template<typename Device>
constexpr DeviceMetadata get_device_metadata() {
    return DeviceRegistry<Device>::metadata;
}

/// Check if a type has device metadata registered
template<typename T>
concept HasDeviceMetadata = requires {
    { DeviceRegistry<T>::metadata } -> std::convertible_to<DeviceMetadata>;
    requires DeviceRegistry<T>::metadata.name != std::string_view("Unknown");
};

// =============================================================================
// Reflection-Ready Parameter Introspection
// =============================================================================

/// Parameter descriptor for future reflection
struct ParamDescriptor {
    std::string_view name;
    std::string_view unit;
    double default_value;
    double min_value;
    double max_value;
};

/// Parameter registry (will be auto-generated by reflection in C++26)
template<typename Device>
struct ParamRegistry {
    static constexpr std::array<ParamDescriptor, 0> params = {};
};

/// Macro to register device parameters
#define PULSIM_REGISTER_PARAMS(DeviceType, ...) \
    template<> \
    struct ParamRegistry<DeviceType> { \
        static constexpr auto params = std::array{ __VA_ARGS__ }; \
    }

/// Helper to define a parameter descriptor
#define PULSIM_PARAM(Name, Unit, Default, Min, Max) \
    ParamDescriptor{ .name = Name, .unit = Unit, .default_value = Default, .min_value = Min, .max_value = Max }

// =============================================================================
// Compile-Time String Utilities (for reflection preparation)
// =============================================================================

/// Fixed-size compile-time string
template<std::size_t N>
struct FixedString {
    char data[N]{};

    constexpr FixedString(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }

    constexpr operator std::string_view() const {
        return std::string_view(data, N - 1);
    }

    constexpr std::size_t size() const { return N - 1; }
};

template<std::size_t N>
FixedString(const char (&)[N]) -> FixedString<N>;

/// Compile-time type name (placeholder for reflection)
template<typename T>
constexpr std::string_view type_name() {
    // In C++26 with reflection, this would use std::meta::name_of
    // For now, return a placeholder
    return "UnknownType";
}

// Specializations for known types
template<> constexpr std::string_view type_name<double>() { return "double"; }
template<> constexpr std::string_view type_name<float>() { return "float"; }
template<> constexpr std::string_view type_name<int>() { return "int"; }
template<> constexpr std::string_view type_name<bool>() { return "bool"; }

}  // namespace pulsim::v1
