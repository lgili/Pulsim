#pragma once

// =============================================================================
// Pulsim — the terminal voltage and current of a PWL switch / diode under
// its stamped mode: ONE law, read by the in-kernel loss tracker
// (solver/loss_tracker.hpp) and by the in-loop recorders
// (record/loop_feed.hpp).
// =============================================================================
//
//   v = v(from) − v(to)
//   i = x[var]                         branch form (a current column)
//     = g_on·(v − V_f)  /  g_off·v     conductance-form diode, ON / OFF
//     = g_on·v          /  g_off·v     conductance-form switch
// `bit` is the device's bit in the switch-state mask (switch branches in
// graph order, diodes included).

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

namespace pulsim::pwl {

struct PwlBranchLaw {
    Index from = -1, to = -1, var = -1;
    long bit = -1;
    bool diode = false;
    Real g_on = 1, g_off = 0, v_f = 0;

    /// `branch` must be a PWL switch or diode (StoredKind Switch / Diode).
    [[nodiscard]] static PwlBranchLaw of(const topology::Graph& graph, const DevicePool& pool,
                                         Index branch) {
        PwlBranchLaw d;
        const auto& br = graph.branch(branch);
        d.from = br.from;
        d.to = br.to;
        d.var = pool.branch_var_id_for_switch(branch, graph);
        d.diode = pool.kind_of(branch) == DevicePool::StoredKind::Diode;
        if (d.diode) {
            const auto& p = pool.diode_params(branch);
            d.g_on = p.g_on;
            d.g_off = p.g_off;
            d.v_f = p.V_f;
        } else {
            d.g_on = pool.switch_g_on(branch);
            d.g_off = pool.switch_g_off(branch);
        }
        long bit = 0;
        for (Index b = 0; b < branch; ++b) {
            if (graph.branch(b).kind == topology::BranchKind::Switch) ++bit;
        }
        d.bit = bit;
        return d;
    }

    [[nodiscard]] bool on(const topology::SwitchStateMask& m) const {
        return m.get(static_cast<Size>(bit));
    }
    [[nodiscard]] Real v(const Vector& x) const {
        return stamping::read_node_voltage(x, from) - stamping::read_node_voltage(x, to);
    }
    [[nodiscard]] Real i(const Vector& x, bool is_on) const {
        if (var >= 0) return x[var];
        const Real vd = v(x);
        if (diode) return is_on ? g_on * (vd - v_f) : g_off * vd;
        return vd * (is_on ? g_on : g_off);
    }
};

}  // namespace pulsim::pwl
