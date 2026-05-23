#pragma once

// =============================================================================
// Pulsim — Layer 4 V2: Seed HistoryState + DiodeEventState from DC OP
// =============================================================================
//
// `pulsim-v2-dc-operating-point` Phase 2.
//
// Utility helpers used by Layer 5 V3's `run_transient(...,
// start_from_dc_op=true)` to populate HistoryState and
// DiodeEventState from a DC operating-point solution.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::pwl {

/// Build a HistoryState and seed it from a DC operating-point
/// state vector. Returns the seeded state ready for Layer 5 V3's
/// time-stepping loop.
[[nodiscard]] inline HistoryState make_seeded_history(
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& dc_x) {
    HistoryState h(graph, pool);
    h.seed_from_dc_op(dc_x);
    return h;
}

/// Build a DiodeEventState and seed each diode's initial state
/// from the DC operating-point. Diodes that are forward-biased
/// in the DC solve start ON; others start OFF.
[[nodiscard]] inline DiodeEventState make_seeded_diodes(
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& dc_x) {
    DiodeEventState d(graph, pool);
    d.reset();
    d.update_from_state(dc_x);
    return d;
}

}  // namespace pulsim::pwl
