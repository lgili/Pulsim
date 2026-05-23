#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: VoltageSource device model
// =============================================================================
//
// `pulsim-v2-device-models-ad-driven` Phase 4.
//
// A voltage source does NOT fit the "stamp a current" pattern. It
// imposes a constraint:  v[0] - v[1] = V (the source voltage).
//
// The `current<S>` function returns ZERO from the DeviceModel
// contract. Layer 3's stamper recognises `kind == Source` and adds
// the constraint row to the MNA matrix instead of stamping a current.
// The configured voltage is exposed via `static_voltage(p)` for that
// stamping path.
//
// Time-varying sources (PWM, sine, pulse) are a Layer 6 / frontend
// concern — they wrap a VoltageSource model whose `V` is updated at
// the start of every accepted timestep.

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct VoltageSource {
    struct Params {
        Real V;     // DC voltage [V]
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = true;

    /// Current is NOT a current here — the source's contribution is
    /// a constraint, handled by Layer 3 via `kind == Source`. Return
    /// zero to satisfy the DeviceModel concept.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static constexpr S current(const S* /*v*/,
                                              const Params& /*p*/) noexcept {
        return S{0};
    }

    /// Layer 3 reads this to build the constraint row.
    [[nodiscard]] static constexpr Real static_voltage(const Params& p) noexcept {
        return p.V;
    }
};

}  // namespace pulsim::models
