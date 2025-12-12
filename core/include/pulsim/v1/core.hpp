#pragma once

// =============================================================================
// PulsimCore v2 - High-Performance Circuit Simulation Core
// =============================================================================
// This is the main header for the v2 API. It provides:
// - C++23 concepts and type traits for compile-time optimization
// - CRTP-based device models for zero-overhead abstraction
// - Policy-based solver configuration
// - Expression templates for efficient matrix operations
// - Advanced C++23 features (mdspan, deducing this, reflection prep)
// =============================================================================

#include "pulsim/v1/concepts.hpp"
#include "pulsim/v1/type_traits.hpp"
#include "pulsim/v1/constexpr_utils.hpp"
#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/device_base.hpp"
#include "pulsim/v1/cpp23_features.hpp"
#include "pulsim/v1/expression_templates.hpp"
#include "pulsim/v1/circuit_graph.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/integration.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/convergence_aids.hpp"
#include "pulsim/v1/validation.hpp"
#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/thermal.hpp"
#include "pulsim/v1/losses.hpp"
#include "pulsim/v1/sources.hpp"
#include "pulsim/v1/control.hpp"

// Convenience namespace alias
namespace pulsim2 = pulsim::v1;
