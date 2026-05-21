#pragma once

// =============================================================================
// Pulsim v2 — Layer 3: MNA convention (documentation header)
// =============================================================================
//
// This header has NO executable code beyond a single static_assert.
// It exists so that the v2 stamping convention is searchable from
// any stamping translation unit (`#include "pulsim/v2/stamping/
// mna_convention.hpp"` to refresh on the contract).
//
// -----------------------------------------------------------------------------
// State-vector layout
// -----------------------------------------------------------------------------
//
//   x = [ v_0 v_1 ... v_{N-1}  i_b0 i_b1 ... i_b{M-1} ]
//        |--- node voltages ---|--- branch currents ---|
//
//   * N = graph.num_nodes() — the non-ground nodes. Ground sits at
//     the implicit index `kGround = -1` and is NOT in the state.
//   * M = number of branches with kind == Source (voltage sources
//     need a branch-current unknown to close the MNA system).
//   * Total state size: N + M.
//
// -----------------------------------------------------------------------------
// Sign convention
// -----------------------------------------------------------------------------
//
//   * Every branch has a "from" terminal (terminal 0) and a "to"
//     terminal (terminal 1).
//   * The branch current is the current flowing from terminal 0 to
//     terminal 1 (positive when v_from > v_to for a resistor).
//   * KCL at node N: sum of (currents leaving N) = 0.
//       - A branch with from == N contributes +current to f[N].
//       - A branch with to   == N contributes -current to f[N].
//   * At convergence, f = 0 means KCL holds at every active node.
//
// -----------------------------------------------------------------------------
// Newton form
// -----------------------------------------------------------------------------
//
//   Layer 5's Newton solver calls Layer 3's stamping functions
//   every iteration to produce J and f, then solves J · Δx = -f
//   and updates x ← x + Δx.
//
// -----------------------------------------------------------------------------
// Ground handling
// -----------------------------------------------------------------------------
//
//   `kGround = -1`. NOT a row in J. NOT an index in f. Stamping
//   helpers (`node_is_active`, `read_node_voltage`) gate access —
//   when a stamp touches the ground row/column the operation is
//   silently skipped.

#include "pulsim/v2/numeric/types.hpp"

namespace pulsim::v2::stamping {

// Lock in the assumption every stamping function relies on.
static_assert(kGround == Index{-1},
              "v2 MNA convention requires kGround == -1; if Layer 0 "
              "ever changes the sentinel, every stamper needs to "
              "follow.");

}  // namespace pulsim::v2::stamping
