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

#include "pulsim/v2/concepts.hpp"
#include "pulsim/v2/type_traits.hpp"
#include "pulsim/v2/constexpr_utils.hpp"
#include "pulsim/v2/numeric_types.hpp"
#include "pulsim/v2/device_base.hpp"
#include "pulsim/v2/cpp23_features.hpp"
#include "pulsim/v2/expression_templates.hpp"
#include "pulsim/v2/circuit_graph.hpp"
#include "pulsim/v2/solver.hpp"
#include "pulsim/v2/integration.hpp"
#include "pulsim/v2/high_performance.hpp"
#include "pulsim/v2/convergence_aids.hpp"

// Convenience namespace alias
namespace pulsim2 = pulsim::v2;
