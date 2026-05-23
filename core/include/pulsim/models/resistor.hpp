#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: Resistor device model
// =============================================================================
//
// `pulsim-v2-device-models-ad-driven` Phase 3.
//
// Linear two-terminal passive: i = G · (v[0] - v[1]).
// Terminal 0 is the positive terminal; current flows from terminal 0
// to terminal 1 when v[0] > v[1].
//
// Conductance G (in siemens) is the parameter, NOT resistance R. This
// keeps the math division-free in the hot path: stamping is just a
// multiply, not a divide. The user-facing builder API can accept
// `R` in ohms and convert at construction time.

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct Resistor {
    struct Params {
        Real G;     // siemens
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = true;

    /// Current from terminal 0 to terminal 1.
    /// Templated on any FloatingPoint type so the SAME function
    /// instantiates for `Real` (forward eval) and `ADRealN<2>`
    /// (Jacobian extraction).
    template <numeric::FloatingPoint S>
    [[nodiscard]] static constexpr S current(const S* v, const Params& p) noexcept {
        return p.G * (v[0] - v[1]);
    }
};

}  // namespace pulsim::models
