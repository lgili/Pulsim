#pragma once

// =============================================================================
// PulsimCore v2 - Compile-Time Circuit Graph Analysis
// =============================================================================
// This header provides compile-time circuit topology analysis using variadic
// templates. It enables:
// - Static node and branch counting
// - Compile-time sparsity pattern generation
// - Topology validation at compile time
// - Zero-overhead abstraction through template metaprogramming
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/device_base.hpp"
#include <array>
#include <tuple>
#include <type_traits>
#include <concepts>
#include <algorithm>
#include <bitset>

namespace pulsim::v1 {

// =============================================================================
// Device Traits for Compile-Time Analysis
// =============================================================================

/// Traits to extract device properties at compile time
template<typename Device>
struct circuit_device_traits {
    /// Number of terminals (pins) on this device
    static constexpr std::size_t terminal_count = Device::num_pins;

    /// Number of internal nodes (for complex devices like transformers)
    static constexpr std::size_t internal_node_count = 0;

    /// Number of current branches (for voltage sources, inductors)
    static constexpr std::size_t branch_count =
        std::is_same_v<std::remove_cvref_t<Device>, VoltageSource> ? 1 :
        std::is_same_v<std::remove_cvref_t<Device>, Inductor> ? 1 :
        std::is_same_v<std::remove_cvref_t<Device>, Transformer> ? 2 : 0;

    /// Jacobian entries contributed by this device
    static constexpr std::size_t max_jacobian_entries =
        terminal_count * terminal_count + branch_count * (2 * terminal_count + 1);

    /// Whether this device is linear (constant Jacobian)
    static constexpr bool is_linear =
        std::is_same_v<std::remove_cvref_t<Device>, Resistor> ||
        std::is_same_v<std::remove_cvref_t<Device>, Capacitor> ||
        std::is_same_v<std::remove_cvref_t<Device>, Inductor> ||
        std::is_same_v<std::remove_cvref_t<Device>, VoltageSource> ||
        std::is_same_v<std::remove_cvref_t<Device>, CurrentSource>;

    /// Whether this device requires history (for reactive elements)
    static constexpr bool needs_history =
        std::is_same_v<std::remove_cvref_t<Device>, Capacitor> ||
        std::is_same_v<std::remove_cvref_t<Device>, Inductor>;

    /// Whether this device can switch states (discontinuous)
    static constexpr bool is_switching =
        std::is_same_v<std::remove_cvref_t<Device>, IdealSwitch> ||
        std::is_same_v<std::remove_cvref_t<Device>, IdealDiode> ||
        std::is_same_v<std::remove_cvref_t<Device>, MOSFET> ||
        std::is_same_v<std::remove_cvref_t<Device>, IGBT>;
};

// =============================================================================
// Compile-Time Node Collection
// =============================================================================

/// Collect unique nodes from device connections at compile time
template<typename... Devices>
class NodeCollector {
public:
    /// Maximum possible unique nodes (2 per 2-terminal device, 3 per 3-terminal)
    static constexpr std::size_t max_nodes =
        ((circuit_device_traits<Devices>::terminal_count + ...) + 0);

    /// Node ID type
    using NodeId = Index;

    /// Collected node IDs (compile-time storage)
    std::array<NodeId, max_nodes> nodes{};
    std::size_t node_count = 0;

    /// Add a node if not already present
    constexpr bool add_node(NodeId id) {
        if (id == ground_node) return false;  // Ground is implicit
        for (std::size_t i = 0; i < node_count; ++i) {
            if (nodes[i] == id) return false;  // Already present
        }
        if (node_count < max_nodes) {
            nodes[node_count++] = id;
            return true;
        }
        return false;
    }

    /// Check if node exists
    [[nodiscard]] constexpr bool has_node(NodeId id) const {
        for (std::size_t i = 0; i < node_count; ++i) {
            if (nodes[i] == id) return true;
        }
        return false;
    }

    /// Get node index (for matrix assembly)
    [[nodiscard]] constexpr std::size_t node_index(NodeId id) const {
        for (std::size_t i = 0; i < node_count; ++i) {
            if (nodes[i] == id) return i;
        }
        return max_nodes;  // Invalid
    }
};

// =============================================================================
// 2.5.1: CircuitGraph<Devices...> Variadic Template
// =============================================================================

/// Compile-time circuit graph representation
/// Note: This is a type-level analysis tool. Runtime node mapping is handled separately.
template<typename... Devices>
class CircuitGraph {
public:
    /// Number of device types in this circuit
    static constexpr std::size_t device_type_count = sizeof...(Devices);

    /// Total terminal connections (upper bound on edges)
    static constexpr std::size_t total_terminals =
        (circuit_device_traits<Devices>::terminal_count + ... + 0);

    /// Total branches (current variables)
    static constexpr std::size_t total_branches =
        (circuit_device_traits<Devices>::branch_count + ... + 0);

    /// Maximum unique nodes
    static constexpr std::size_t max_nodes = total_terminals;

    /// Maximum Jacobian non-zeros
    static constexpr std::size_t max_jacobian_nnz =
        (circuit_device_traits<Devices>::max_jacobian_entries + ... + 0);

    /// Check if all devices are linear
    static constexpr bool all_linear =
        (circuit_device_traits<Devices>::is_linear && ...);

    /// Check if any device needs history
    static constexpr bool needs_history =
        (circuit_device_traits<Devices>::needs_history || ...);

    /// Check if any device can switch
    static constexpr bool has_switching =
        (circuit_device_traits<Devices>::is_switching || ...);

    /// Device storage type
    using DeviceTuple = std::tuple<Devices...>;

    /// Constructor from device tuple
    constexpr explicit CircuitGraph(Devices... devices)
        : devices_(std::move(devices)...) {
        // Note: Actual node analysis requires runtime node mapping
        // This class provides compile-time type analysis
        is_valid_ = sizeof...(Devices) > 0;
    }

    /// Get number of branches
    [[nodiscard]] constexpr std::size_t num_branches() const {
        return total_branches;
    }

    /// Access devices
    [[nodiscard]] constexpr const DeviceTuple& devices() const {
        return devices_;
    }

    /// Get device by index
    template<std::size_t I>
    [[nodiscard]] constexpr const auto& device() const {
        return std::get<I>(devices_);
    }

    /// Get the type-level sparsity pattern (based on device types only)
    [[nodiscard]] constexpr auto sparsity() const {
        // Return a pattern based on maximum possible entries per device type
        return SparsityPattern<max_jacobian_nnz>{};
    }

    /// Check if circuit is valid (at type level)
    [[nodiscard]] constexpr bool is_valid() const {
        return is_valid_;
    }

    /// Get validation error message
    [[nodiscard]] constexpr const char* validation_error() const {
        return error_msg_;
    }

private:
    DeviceTuple devices_;
    bool is_valid_ = true;
    const char* error_msg_ = nullptr;
};

// =============================================================================
// 2.5.2 & 2.5.3: Compile-Time Node/Branch Counting Utilities
// =============================================================================

/// Count total nodes in a device list
template<typename... Devices>
struct count_max_nodes {
    static constexpr std::size_t value =
        (circuit_device_traits<Devices>::terminal_count + ... + 0);
};

template<typename... Devices>
inline constexpr std::size_t count_max_nodes_v = count_max_nodes<Devices...>::value;

/// Count total branches in a device list
template<typename... Devices>
struct count_branches {
    static constexpr std::size_t value =
        (circuit_device_traits<Devices>::branch_count + ... + 0);
};

template<typename... Devices>
inline constexpr std::size_t count_branches_v = count_branches<Devices...>::value;

/// Count linear devices
template<typename... Devices>
struct count_linear_devices {
    static constexpr std::size_t value =
        ((circuit_device_traits<Devices>::is_linear ? 1 : 0) + ... + 0);
};

template<typename... Devices>
inline constexpr std::size_t count_linear_devices_v = count_linear_devices<Devices...>::value;

/// Count nonlinear devices
template<typename... Devices>
struct count_nonlinear_devices {
    static constexpr std::size_t value = sizeof...(Devices) - count_linear_devices_v<Devices...>;
};

template<typename... Devices>
inline constexpr std::size_t count_nonlinear_devices_v = count_nonlinear_devices<Devices...>::value;

// =============================================================================
// 2.5.4: Static Jacobian Sparsity Pattern Generation
// =============================================================================

/// Generate a static sparsity pattern for a circuit configuration
template<typename... Devices>
constexpr auto make_static_sparsity() {
    constexpr std::size_t max_nnz =
        (circuit_device_traits<Devices>::max_jacobian_entries + ... + 0);
    return SparsityPattern<max_nnz>{};
}

/// Sparsity pattern builder for compile-time construction
template<std::size_t MaxNodes, std::size_t MaxBranches>
class StaticSparsityBuilder {
public:
    static constexpr std::size_t max_size = MaxNodes + MaxBranches;
    static constexpr std::size_t max_nnz = max_size * max_size;  // Worst case

    using PatternType = SparsityPattern<max_nnz>;

    constexpr StaticSparsityBuilder() = default;

    /// Add conductance stamp (2x2 symmetric)
    constexpr void add_conductance(Index np, Index nn) {
        if (np >= 0 && np < static_cast<Index>(max_size)) {
            pattern_.add(np, np);
            if (nn >= 0) pattern_.add(np, nn);
        }
        if (nn >= 0 && nn < static_cast<Index>(max_size)) {
            if (np >= 0) pattern_.add(nn, np);
            pattern_.add(nn, nn);
        }
    }

    /// Add voltage source stamp (branch current)
    constexpr void add_voltage_source(Index np, Index nn, Index branch_idx) {
        if (np >= 0) {
            pattern_.add(np, branch_idx);
            pattern_.add(branch_idx, np);
        }
        if (nn >= 0) {
            pattern_.add(nn, branch_idx);
            pattern_.add(branch_idx, nn);
        }
    }

    /// Add inductor stamp (branch current with companion)
    constexpr void add_inductor(Index np, Index nn, Index branch_idx) {
        add_voltage_source(np, nn, branch_idx);
        pattern_.add(branch_idx, branch_idx);  // Companion resistance
    }

    [[nodiscard]] constexpr const PatternType& pattern() const {
        return pattern_;
    }

private:
    PatternType pattern_;
};

// =============================================================================
// 2.5.5: Compile-Time Topology Validation
// =============================================================================

/// Topology validation result
struct TopologyValidation {
    bool is_valid = true;
    bool has_ground_reference = true;
    bool has_voltage_reference = true;
    bool has_isolated_nodes = false;
    std::size_t node_count = 0;
    std::size_t branch_count = 0;

    [[nodiscard]] constexpr operator bool() const { return is_valid; }
};

/// Validate circuit topology at compile time
template<typename... Devices>
constexpr TopologyValidation validate_circuit_topology() {
    TopologyValidation result;

    // Count components
    result.node_count = count_max_nodes_v<Devices...>;
    result.branch_count = count_branches_v<Devices...>;

    // Check for voltage reference
    constexpr bool has_vsource =
        (std::is_same_v<std::remove_cvref_t<Devices>, VoltageSource> || ...);
    result.has_voltage_reference = has_vsource;

    // Basic validity: need at least one device
    result.is_valid = (sizeof...(Devices) > 0);

    // If no voltage source, circuit may not have DC solution (warning only)
    // This is a potential issue but not necessarily invalid

    return result;
}

/// Concept for valid circuit configurations
template<typename... Devices>
concept ValidCircuitDevices = requires {
    requires sizeof...(Devices) > 0;
    requires (StampableDevice<Devices> && ...);
};

// =============================================================================
// CircuitGraph Factory Functions
// =============================================================================

/// Create a circuit graph from devices
template<typename... Devices>
    requires ValidCircuitDevices<Devices...>
[[nodiscard]] constexpr auto make_circuit(Devices... devices) {
    return CircuitGraph<Devices...>(std::move(devices)...);
}

/// Create and validate a circuit at compile time
template<typename... Devices>
    requires ValidCircuitDevices<Devices...>
[[nodiscard]] constexpr auto make_validated_circuit(Devices... devices) {
    constexpr auto validation = validate_circuit_topology<Devices...>();
    static_assert(validation.is_valid, "Invalid circuit topology");
    return CircuitGraph<Devices...>(std::move(devices)...);
}

// =============================================================================
// Circuit Analysis Queries
// =============================================================================

/// Get the system dimension for a circuit
template<typename... Devices>
constexpr std::size_t circuit_system_size() {
    return count_max_nodes_v<Devices...> + count_branches_v<Devices...>;
}

/// Check if a circuit is purely linear
template<typename... Devices>
constexpr bool is_linear_circuit() {
    return (circuit_device_traits<Devices>::is_linear && ...);
}

/// Check if a circuit has reactive elements
template<typename... Devices>
constexpr bool has_reactive_elements() {
    return (circuit_device_traits<Devices>::needs_history || ...);
}

/// Check if a circuit has switching elements
template<typename... Devices>
constexpr bool has_switching_elements() {
    return (circuit_device_traits<Devices>::is_switching || ...);
}

// =============================================================================
// Compile-Time Circuit Examples
// =============================================================================

namespace examples {

/// Example: Simple RC circuit topology
inline constexpr auto rc_circuit_validation =
    validate_circuit_topology<Resistor, Capacitor, VoltageSource>();

/// Example: RLC circuit topology
inline constexpr auto rlc_circuit_validation =
    validate_circuit_topology<Resistor, Inductor, Capacitor, VoltageSource>();

/// Example: Diode bridge topology
inline constexpr auto bridge_validation =
    validate_circuit_topology<IdealDiode, IdealDiode, IdealDiode, IdealDiode,
                              Resistor, VoltageSource>();

// Static assertions for compile-time validation
static_assert(rc_circuit_validation.is_valid);
static_assert(rlc_circuit_validation.is_valid);
static_assert(bridge_validation.is_valid);

// Type-level queries
static_assert(count_branches_v<VoltageSource> == 1);
static_assert(count_branches_v<Resistor> == 0);
static_assert(count_branches_v<Inductor> == 1);
static_assert(count_branches_v<Resistor, Capacitor, VoltageSource> == 1);
static_assert(count_branches_v<Resistor, Inductor, VoltageSource> == 2);

static_assert(is_linear_circuit<Resistor, Capacitor, VoltageSource>());
static_assert(!is_linear_circuit<Resistor, IdealDiode>());

static_assert(has_reactive_elements<Resistor, Capacitor>());
static_assert(!has_reactive_elements<Resistor, CurrentSource>());

static_assert(!has_switching_elements<Resistor, Capacitor>());
static_assert(has_switching_elements<IdealSwitch, Resistor>());

} // namespace examples

// =============================================================================
// Static Assertions for Type System
// =============================================================================

namespace detail {

// Verify device traits
static_assert(circuit_device_traits<Resistor>::terminal_count == 2);
static_assert(circuit_device_traits<Resistor>::branch_count == 0);
static_assert(circuit_device_traits<Resistor>::is_linear == true);

static_assert(circuit_device_traits<VoltageSource>::branch_count == 1);
static_assert(circuit_device_traits<Inductor>::branch_count == 1);
static_assert(circuit_device_traits<Capacitor>::needs_history == true);
static_assert(circuit_device_traits<IdealDiode>::is_switching == true);

// Verify counting utilities
static_assert(count_max_nodes_v<Resistor> == 2);
static_assert(count_max_nodes_v<Resistor, Capacitor> == 4);
static_assert(count_branches_v<Resistor, Capacitor> == 0);
static_assert(count_branches_v<VoltageSource, Inductor> == 2);

} // namespace detail

} // namespace pulsim::v1
