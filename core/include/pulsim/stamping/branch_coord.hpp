#pragma once

// =============================================================================
// Pulsim — Layer 3: BranchCoord + ground-aware helpers
// =============================================================================
//
// `pulsim-v2-generic-stamping-pipeline` Phase 1.
//
// Minimal coordinate every 2-terminal stamper takes. The Layer 1
// graph stores Branch endpoints; Layer 3 packages them with the
// branch id for stamping. The branch id is for diagnostic +
// Layer-4 cache-invalidation use (the cache keys on it).
//
// Ground handling lives here because every stamper touches it:
// `kGround = -1` is the sentinel for "this terminal is ground",
// and ground is NOT in the state vector. Helpers `read_node_voltage`
// and `node_is_active` centralise the check so individual stampers
// never repeat the `(node < 0)` ternary.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::stamping {

// -----------------------------------------------------------------------------
// BranchCoord — what every 2-terminal stamper needs to know.
// -----------------------------------------------------------------------------
struct BranchCoord {
    Index from;        // node index (>= 0) or kGround
    Index to;          // node index (>= 0) or kGround
    Index branch_id;   // for Layer 4 / diagnostic use
};

// -----------------------------------------------------------------------------
// read_node_voltage — uniformly returns the voltage at a node,
// treating ground as 0V.
//
// Layer 4's PWL cache will eventually merge node-equivalence
// classes too, but for Layer 3 the ground sentinel is the only
// special case.
// -----------------------------------------------------------------------------
[[nodiscard]] inline Real read_node_voltage(const Vector& x,
                                              Index node) noexcept {
    if (node < 0) return Real{0};
    return x[node];
}

// -----------------------------------------------------------------------------
// node_is_active — true iff the node corresponds to a real row in
// the state vector. Ground returns false; stamping rows / cols for
// ground are skipped.
// -----------------------------------------------------------------------------
[[nodiscard]] inline constexpr bool node_is_active(Index node) noexcept {
    return node >= 0;
}

}  // namespace pulsim::stamping
