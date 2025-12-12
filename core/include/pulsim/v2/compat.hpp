#pragma once

// =============================================================================
// PulsimCore v2 - Backward Compatibility Shim
// =============================================================================
// This header provides backward compatibility with the v1 API while using
// the v2 implementation under the hood. It allows gradual migration and
// enables v1/v2 side-by-side operation.
//
// Usage patterns:
//   1. Full v2: #include <pulsim/v2/core.hpp>
//   2. v1 compat: #include <pulsim/v2/compat.hpp>
//   3. Feature flag: #define PULSIM_USE_V2 1 before including pulsim.hpp
// =============================================================================

#include "pulsim/v2/core.hpp"

// =============================================================================
// Version Detection and Feature Flags
// =============================================================================

/// Feature macro to enable v2 API globally
#ifndef PULSIM_USE_V2
#define PULSIM_USE_V2 0  // Default: use v1 API
#endif

/// Feature macro for experimental features
#ifndef PULSIM_EXPERIMENTAL
#define PULSIM_EXPERIMENTAL 0
#endif

/// Version macros
#define PULSIM_VERSION_MAJOR 2
#define PULSIM_VERSION_MINOR 0
#define PULSIM_VERSION_PATCH 0
#define PULSIM_VERSION_STRING "2.0.0"

/// API version detection
#if PULSIM_USE_V2
#define PULSIM_API_VERSION 2
#else
#define PULSIM_API_VERSION 1
#endif

// =============================================================================
// Namespace Aliasing for v1/v2 Selection
// =============================================================================

namespace pulsim {

/// Conditional namespace alias - use pulsim::api to get either v1 or v2
#if PULSIM_USE_V2
namespace api = v2;
#else
// When v1 is selected, 'api' refers to the root pulsim namespace
// (v1 types are defined directly in pulsim::)
#endif

// Forward declarations for v2 types with v1-compatible names
namespace compat {

// =============================================================================
// Type Compatibility Layer
// =============================================================================

/// Real type - maps to v2::Real (double by default)
using Real = v2::Real;

/// Index type - maps to v2::Index (int32_t by default)
using Index = v2::Index;

/// Ground node constant
inline constexpr Index ground = v2::ground_node;

// =============================================================================
// Device Type Mapping
// =============================================================================

/// DeviceType enum compatible with v1
using DeviceType = v2::DeviceType;

// Device type aliases for v1 compatibility
using Resistor = v2::Resistor;
using Capacitor = v2::Capacitor;
using Inductor = v2::Inductor;
using VoltageSource = v2::VoltageSource;
using CurrentSource = v2::CurrentSource;
using Diode = v2::IdealDiode;      // v2 uses IdealDiode
using Switch = v2::IdealSwitch;    // v2 uses IdealSwitch
using MOSFET = v2::MOSFET;
using IGBT = v2::IGBT;
using Transformer = v2::Transformer;

// =============================================================================
// Matrix/Vector Type Mapping
// =============================================================================

using SparseMatrix = v2::SparseMatrix;
using Vector = v2::Vector;

// =============================================================================
// Utility Functions for Migration
// =============================================================================

/// Check if v2 API is enabled at runtime
constexpr bool is_v2_enabled() {
    return PULSIM_USE_V2 != 0;
}

/// Get API version string
constexpr const char* api_version_string() {
    return is_v2_enabled() ? "v2" : "v1";
}

/// Get full version string
constexpr const char* version_string() {
    return PULSIM_VERSION_STRING;
}

} // namespace compat

// =============================================================================
// Conditional Export to Root Namespace
// =============================================================================

#if PULSIM_USE_V2
// When v2 is enabled, export compat types to root pulsim namespace
using compat::Real;
using compat::Index;
using compat::ground;
using compat::DeviceType;
using compat::Resistor;
using compat::Capacitor;
using compat::Inductor;
using compat::VoltageSource;
using compat::CurrentSource;
using compat::Diode;
using compat::Switch;
using compat::MOSFET;
using compat::IGBT;
using compat::Transformer;
#endif

} // namespace pulsim

// =============================================================================
// Convenience Macros for Gradual Migration
// =============================================================================

/// Mark code as v1-only (will be removed when v2 becomes default)
#define PULSIM_V1_ONLY(code) \
    do { \
        if constexpr (!pulsim::compat::is_v2_enabled()) { code; } \
    } while(0)

/// Mark code as v2-only (new functionality)
#define PULSIM_V2_ONLY(code) \
    do { \
        if constexpr (pulsim::compat::is_v2_enabled()) { code; } \
    } while(0)

/// Conditional compilation based on API version
#define PULSIM_IF_V2(v2_code, v1_code) \
    do { \
        if constexpr (pulsim::compat::is_v2_enabled()) { v2_code; } \
        else { v1_code; } \
    } while(0)

/// Deprecation warning for v1-specific code
#if PULSIM_USE_V2
#define PULSIM_DEPRECATED_V1 [[deprecated("This API is deprecated. Use v2 API instead.")]]
#else
#define PULSIM_DEPRECATED_V1
#endif

// =============================================================================
// Migration Helpers
// =============================================================================

namespace pulsim::migration {

/// Helper to convert v1 component type to v2 device type
constexpr v2::DeviceType component_to_device_type(int v1_type) {
    // Mapping from v1 ComponentType enum values to v2 DeviceType
    switch (v1_type) {
        case 0: return v2::DeviceType::Resistor;
        case 1: return v2::DeviceType::Capacitor;
        case 2: return v2::DeviceType::Inductor;
        case 3: return v2::DeviceType::VoltageSource;
        case 4: return v2::DeviceType::CurrentSource;
        // VCVS, VCCS, CCVS, CCCS not yet in v2
        case 9: return v2::DeviceType::Diode;
        case 10: return v2::DeviceType::Switch;
        case 11: return v2::DeviceType::MOSFET;
        case 12: return v2::DeviceType::IGBT;
        case 13: return v2::DeviceType::Transformer;
        default: return v2::DeviceType::Unknown;
    }
}

/// Helper to check if a device type is supported in v2
constexpr bool is_device_supported_v2(v2::DeviceType type) {
    return type != v2::DeviceType::Unknown;
}

} // namespace pulsim::migration
