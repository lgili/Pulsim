#pragma once

// =============================================================================
// Pulsim v2 — Layer 0: numeric concepts
// =============================================================================
//
// `bootstrap-pulsim-v2-kernel` Phase 1. Two C++20 concepts that higher
// layers use to constrain templates.
//
// The big design win these enable: in Layer 2, every device model is
// written ONCE as a templated function that accepts any floating-point
// type. The same template instantiates for:
//   * `double` — direct calls in tests + the Layer 5 solver
//   * `pulsim::v2::Real` — the runtime path
//   * (future) AD scalar from `pulsim::v2::ad::ADReal` — Jacobian
//     derivation via automatic differentiation, NO duplicated manual
//     stamping path. This kills the v1 quadruple-duplication problem.

#include "pulsim/v2/numeric/types.hpp"

#include <concepts>
#include <type_traits>

namespace pulsim::v2::numeric {

// -----------------------------------------------------------------------------
// FloatingPoint — satisfied by Real and by any standard floating-point.
//
// Layer 2 device-current functions are constrained on this concept:
//
//     template <FloatingPoint S>
//     S drain_current(S vg, S vd, S vs, const MosfetParams& p) noexcept;
//
// The SAME function compiles for double (call-site, tests) AND for the
// AD scalar (Jacobian extraction). No manual derivative path needed.
// -----------------------------------------------------------------------------
template <typename T>
concept FloatingPoint = std::floating_point<T> ||
                       std::is_same_v<std::remove_cv_t<T>, Real>;

// -----------------------------------------------------------------------------
// IndexLike — signed integer at least 32 bits wide.
//
// Layer 3 (stamping) and Layer 4 (state-space cache) use this to
// constrain templates that take node / row / column indices. Eigen's
// `StorageIndex` template parameter and our own Index typedef both
// satisfy it.
// -----------------------------------------------------------------------------
template <typename T>
concept IndexLike = std::signed_integral<T> && (sizeof(T) >= 4);

}  // namespace pulsim::v2::numeric
