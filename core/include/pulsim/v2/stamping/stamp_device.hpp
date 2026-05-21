#pragma once

// =============================================================================
// Pulsim v2 — Layer 3: Generic 2-terminal device stamper
// =============================================================================
//
// `pulsim-v2-generic-stamping-pipeline` Phase 2.
//
// THE v1-stamping-duplication killer.
//
// v1 has, per device class, FOUR stamping implementations:
//   * mosfet.hpp::stamp_jacobian_behavioral (~80 LOC)
//   * mosfet.hpp::stamp_jacobian_via_ad     (~60 LOC)
//   * mosfet.hpp::stamp_jacobian_ideal      (~70 LOC)
//   * runtime_circuit.hpp::stamp_mosfet_jacobian (~250 LOC)
//
// Same math, four places, with subtle sign-convention drift that
// caused the May 2026 diode-fails-after-reverse-bias bug.
//
// v2's `stamp_device<T>` is ONE templated function. Works for
// every `DeviceModel T` with `T::num_terminals == 2`. Reads
// terminal voltages from `x`, calls Layer 2's
// `evaluate_current_and_jacobian<T>`, stamps the standard
// 4-entry conductance block.
//
// Adding a new 2-terminal device requires ZERO changes here —
// the device's `current<S>` template is the only file touched.

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"

namespace pulsim::v2::stamping {

// -----------------------------------------------------------------------------
// stamp_device — generic 2-terminal device stamper.
//
// Stamps the device's contribution to the MNA Jacobian J and
// residual f at the current state x. Idempotent across calls:
// the function ADDS to existing entries (Layer 4 / 5 zeroes J + f
// at the start of every Newton iteration).
//
// Compile-time gate: `T::num_terminals == 2`. 3-terminal+ devices
// use a separate (future) multi-pin stamper — this one is the
// 2-terminal reference.
// -----------------------------------------------------------------------------
template <models::DeviceModel T>
void stamp_device(sparse::Matrix& J, Vector& f, const Vector& x,
                  const BranchCoord& coord,
                  const typename T::Params& p) noexcept {
    static_assert(T::num_terminals == 2,
                  "stamp_device (Layer 3 V0) handles 2-terminal devices "
                  "only. Use the multi-pin stamper for 3-terminal+ "
                  "devices (deferred to a follow-up OpenSpec).");

    // Read terminal voltages, treating ground (kGround) as 0 V.
    const std::array<Real, 2> v = {
        read_node_voltage(x, coord.from),
        read_node_voltage(x, coord.to),
    };

    // One call into Layer 2: yields current AND AD-derived partials.
    // No manual derivative path. No duplication. No drift.
    const auto [i, partials] =
        models::evaluate_current_and_jacobian<T>(v, p);

    const bool from_active = node_is_active(coord.from);
    const bool to_active   = node_is_active(coord.to);

    // Residual (KCL convention: +i leaves from-terminal, -i leaves
    // to-terminal).
    if (from_active) f[coord.from] += i;
    if (to_active)   f[coord.to]   -= i;

    // Jacobian (the standard 2x2 conductance-block pattern).
    // 4 entries, each gated on its row + column being active.
    if (from_active) {
        J.coeffRef(coord.from, coord.from) += partials[0];
        if (to_active) {
            J.coeffRef(coord.from, coord.to) += partials[1];
        }
    }
    if (to_active) {
        if (from_active) {
            J.coeffRef(coord.to, coord.from) -= partials[0];
        }
        J.coeffRef(coord.to, coord.to) -= partials[1];
    }
}

}  // namespace pulsim::v2::stamping
