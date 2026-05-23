#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V12: Pulse b_extra computation helper
// =============================================================================
//
// Same overlay pattern as PWM (V4) and Sine (V11): walks the
// DevicePool for `PulseVoltageSource` entries and computes the
// b_extra contribution that overlays the time-varying pulse
// value on top of the (V=0 baseline) source's constraint row.

#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::sources {

/// Overlay each PulseVoltageSource's instantaneous value
/// into b_extra at simulation time `t`. Returns a vector
/// of size `pool.state_size(graph)` with all zeros except
/// at pulse-source branch-var rows.
[[nodiscard]] inline Vector compute_pulse_b_extra(
    const pwl::DevicePool& pool,
    const topology::Graph& graph,
    Real t) {
    const Size state_size = pool.state_size(graph);
    Vector b_extra =
        Vector::Zero(static_cast<Index>(state_size));

    for (Index b_id = 0;
         b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Source) {
            continue;
        }
        const auto k = pool.kind_of(branch.id);
        if (k != pwl::DevicePool::StoredKind::PulseVoltageSource) {
            continue;
        }
        const auto& p =
            pool.pulse_voltage_source_params(branch.id);
        const Index src_var =
            pool.branch_var_id_for_source(branch.id, graph);
        const Real v_pulse =
            models::PulseVoltageSource::value_at(p, t);
        b_extra[src_var] += -v_pulse;
    }
    return b_extra;
}

}  // namespace pulsim::sources
