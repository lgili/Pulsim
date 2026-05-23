#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: DeviceModel concept + Jacobian helper
// =============================================================================
//
// `pulsim-v2-device-models-ad-driven` Phase 2.
//
// The DeviceModel concept enforces the v2 "one stamp per device"
// contract at compile time. Any type T satisfying DeviceModel<T>:
//
//   * Has a nested `Params` struct (the user-facing parameters)
//   * Exposes `T::kind` (BranchKind), `T::num_terminals` (Size),
//     and `T::is_linear` (bool) as static constants
//   * Exposes ONE templated function:
//
//       template <numeric::FloatingPoint S>
//       static S current(const S* v, const Params& p) noexcept;
//
// Layer 3's generic stamper consumes the concept without knowing
// the concrete device type. The compiler verifies the contract;
// missing members fail the concept and the build breaks at the
// stamping call site with a clear diagnostic.
//
// `evaluate_current_and_jacobian` is the bridge Layer 3 uses: ONE
// call into the device model yields both the current value (for
// the residual) and the partial derivatives (for the Jacobian).
// No duplicated math.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"     // for BranchKind

#include <array>
#include <concepts>
#include <tuple>

namespace pulsim::models {

// -----------------------------------------------------------------------------
// DeviceModel concept — the v2 device contract.
// -----------------------------------------------------------------------------
template <typename T>
concept DeviceModel = requires(const typename T::Params& p,
                                const Real* v) {
    typename T::Params;
    { T::kind } -> std::convertible_to<topology::BranchKind>;
    { T::num_terminals } -> std::convertible_to<Size>;
    { T::is_linear } -> std::convertible_to<bool>;
    { T::template current<Real>(v, p) } -> std::convertible_to<Real>;
};

// -----------------------------------------------------------------------------
// ModelInputs<T> — type alias for the terminal-voltage array of a
// device. Lets test/consumer code declare `ModelInputs<Resistor> v
// = {3, 1};` without writing the array size explicitly.
// -----------------------------------------------------------------------------
template <DeviceModel T>
using ModelInputs = std::array<Real, T::num_terminals>;

// -----------------------------------------------------------------------------
// evaluate_current_and_jacobian
//
// The bridge for Layer 3. Given a device model T and its terminal
// voltages, return:
//   * the current value (for the residual / `f` vector)
//   * the partial derivatives ∂i/∂v[k] for k in [0, num_terminals)
//     (for the Jacobian `J`)
//
// Implementation: seed the inputs via ad::seed*, evaluate
// `T::current<ADRealN<N>>(...)`, and read off the value + derivs.
// ONE call. No manual derivatives. No 4-place duplication.
// -----------------------------------------------------------------------------
template <DeviceModel T>
[[nodiscard]] auto evaluate_current_and_jacobian(
    const ModelInputs<T>& v, const typename T::Params& p) noexcept {
    constexpr Size N = T::num_terminals;
    // Seed each terminal voltage as an independent AD input.
    std::array<ad::ADRealN<N>, N> seeded{};
    for (Size i = 0; i < N; ++i) {
        std::array<Real, N> d{};
        d[i] = Real{1};
        seeded[i] = ad::ADRealN<N>{v[i], d};
    }
    // Evaluate the device current via AD.
    const auto result = T::template current<ad::ADRealN<N>>(
        seeded.data(), p);
    return std::make_tuple(result.value(), result.derivatives());
}

// -----------------------------------------------------------------------------
// evaluate_current — convenience wrapper for the value-only path
// (forward evaluation, no Jacobian). Used by Layer 5's hot loop
// and by tests.
// -----------------------------------------------------------------------------
template <DeviceModel T>
[[nodiscard]] Real evaluate_current(
    const ModelInputs<T>& v, const typename T::Params& p) noexcept {
    return T::template current<Real>(v.data(), p);
}

}  // namespace pulsim::models
