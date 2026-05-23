#pragma once

// =============================================================================
// Pulsim — Layer 2 V3: CurrentSource device model
// =============================================================================
//
// `pulsim-v2-current-source` Phase 1.
//
// Constant DC current source. Pumps a fixed current `I`
// from the `from` terminal to the `to` terminal.
//
// In MNA terms:
//   * Contributes to KCL at both terminals.
//   * Does NOT introduce a branch-current unknown (the
//     current is GIVEN, not solved for — that's the key
//     difference from VoltageSource).
//   * Stamping: b[from] += +I, b[to] -= I.
//     Then J · x = -(b_constant + b_extra) recovers the
//     correct KCL: at from-node, sum of leaving = -I,
//     meaning external currents must net to +I entering
//     (= the source delivering I via the branch).
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
        Real I;   // amperes — current flowing FROM `from`
                   // TO `to` (positive = conventional current
                   // direction).
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
