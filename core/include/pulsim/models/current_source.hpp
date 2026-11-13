#pragma once

// =============================================================================
// Pulsim — Layer 2 V3: CurrentSource device model
// =============================================================================
//
// `pulsim-v2-current-source` Phase 1.
//
// Constant DC current source. A positive `I` leaves the
// source at `from` into the external circuit and returns at
// `to` (inside the source it runs from `to` to `from`):
// 10 mA with 1 kΩ from `from` to ground reads v(from) = +10 V.
//
// In MNA terms:
//   * Contributes to KCL at both terminals.
//   * Does NOT introduce a branch-current unknown (the
//     current is GIVEN, not solved for — that's the key
//     difference from VoltageSource).
//   * Stamping (stamp_current_source.hpp): b[from] -= I,
//     b[to] += I, so J · x = -b puts +I into the `from`
//     row: the external network must carry I away from it.
//
// Useful for:
//   * Bias-current injection (e.g. testing op-amp inputs).
//   * Photovoltaic / solar cell models (light-induced
//     current source in parallel with a diode).
//   * Modeling motor stator currents in dq-frame.
//   * Norton equivalents of external networks.
//
// Time-varying current sources are NOT supported by this
// V0 model — the user achieves time variation via
// `b_extra_fn(t)` (the same trick used for sinusoidal
// VoltageSource modulation).

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct CurrentSource {
    struct Params {
        Real I;   // amperes — positive I leaves the source at
                   // `from` and returns at `to` (flows through
                   // the external circuit from `from` to `to`).
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = false;
    /// CurrentSource does NOT introduce a branch-current
    /// unknown — the current is fixed. This is the key
    /// structural difference from VoltageSource.
    static constexpr bool needs_branch_unknown = false;

    /// Static contract: current is a constant function of
    /// (effectively no) terminal voltages. Returns the
    /// fixed `I` regardless of `v`. Kept for symmetry
    /// with other device models (Layer 3 stamping).
    template <numeric::FloatingPoint S>
    [[nodiscard]] static constexpr S current(
        const S* /*v*/, const Params& p) noexcept {
        return S{p.I};
    }
};

}  // namespace pulsim::models
