#pragma once

// =============================================================================
// Pulsim v2 — Layer 4: assemble_segment (build one per-state matrix)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 3.
//
// For one `SwitchStateMask`, build the MNA matrix `J` + constant
// RHS `b` by stamping every branch in the graph using Layer 3's
// stampers + Layer 2's device-model parameters from DevicePool.
//
// V0 scope: handles PassiveLinear (Resistor only), Source, and
// Switch branches. Nonlinear branches are SKIPPED.

#include "pulsim/v2/models/resistor.hpp"
#include "pulsim/v2/models/voltage_source.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/stamping/stamp_device.hpp"
#include "pulsim/v2/stamping/stamp_switch.hpp"
#include "pulsim/v2/stamping/stamp_voltage_source.hpp"
#include "pulsim/v2/topology/graph.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

namespace pulsim::v2::pwl {

// -----------------------------------------------------------------------------
// assemble_segment — build (J, b) for one switch state.
//
// Sizes J + b to `pool.state_size(graph)`, zeroes them, then
// iterates every branch in branch_id order. For each branch the
// dispatch by BranchKind picks the right Layer 3 stamper.
//
// V0 uses `x = Vector::Zero(state_size)` during stamping — every
// supported device class is linear in V0, so the stamp doesn't
// depend on the operating point. The voltage source's `-V`
// contribution to `b` is captured because `stamp_voltage_source`
// reads V from its parameter, not from `x`.
// -----------------------------------------------------------------------------
inline void assemble_segment(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              sparse::Matrix& J,
                              Vector& b) {
    const Size state_size = pool.state_size(graph);

    // Reset / size J and b.
    J = sparse::Matrix(static_cast<Index>(state_size),
                        static_cast<Index>(state_size));
    b = Vector::Zero(static_cast<Index>(state_size));

    if (state_size == 0) {
        return;
    }

    // Zero state vector: V0 stamping is linear, so the result
    // doesn't depend on x.
    const Vector x = Vector::Zero(static_cast<Index>(state_size));

    // Counter advances only on Switch-kind branches; matches the
    // SwitchStateMask bit-i = i-th Switch convention.
    Size switch_idx = 0;

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};

        switch (branch.kind) {
        case topology::BranchKind::PassiveLinear: {
            // V0 only supports Resistor under PassiveLinear.
            const auto& p = pool.resistor_params(branch.id);
            stamping::stamp_device<models::Resistor>(J, b, x, coord, p);
            break;
        }
        case topology::BranchKind::Source: {
            const auto& p = pool.voltage_source_params(branch.id);
            const Index branch_var_id =
                pool.branch_var_id_for_source(branch.id, graph);
            stamping::stamp_voltage_source(J, b, x, coord,
                                            branch_var_id, p.V);
            break;
        }
        case topology::BranchKind::Switch: {
            const bool closed = mask.get(switch_idx);
            const Real g_on  = pool.switch_g_on(branch.id);
            const Real g_off = pool.switch_g_off(branch.id);
            stamping::stamp_switch_fixed(J, b, x, coord, closed,
                                          g_on, g_off);
            ++switch_idx;
            break;
        }
        case topology::BranchKind::Nonlinear:
            // V0 deliberately skips Nonlinear branches. Layer 5
            // will handle them via per-segment Newton iteration on
            // top of the cached factor.
            break;
        }
    }
}

}  // namespace pulsim::v2::pwl
