#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V17: Saturable-inductor history tracker
// =============================================================================
//
// Parallel to `HistoryState` (which tracks linear inductors + capacitors),
// this stand-alone tracker holds the previous-step state of each
// saturable inductor: `(i_L_old, V_L_old)`. The Newton refresh function
// reads from here at each iteration; `update_from_state` is called by
// `run_transient` after each successful Newton solve.
//
// Architecturally separate from `HistoryState` to avoid disturbing the
// existing trap-companion machinery in Layer 4. Saturable inductors
// live as `BranchKind::Nonlinear` (assemble skips them); their
// constraint row + KCL are stamped 100 %% via the refresh.

#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <vector>

namespace pulsim::pwl {

class SaturableInductorHistory {
public:
    struct Entry {
        Index branch_id;
        Index from;
        Index to;
        Index branch_var_id;   // absolute row in state vector
        Real  i_L_old = Real{0};
        Real  V_L_old = Real{0};
        models::SaturableInductor::Params params;
    };

    /// Build the entry list from the device pool + graph.
    /// Called once at the start of `run_transient`.
    void init(const topology::Graph& graph,
               const DevicePool& pool) {
        entries_.clear();
        for (const Index b_id :
                 pool.saturable_inductor_branches()) {
            const auto& branch = graph.branch(b_id);
            Entry e;
            e.branch_id = b_id;
            e.from = branch.from;
            e.to   = branch.to;
            e.branch_var_id =
                pool.branch_var_id_for_inductor(b_id, graph);
            e.params =
                pool.saturable_inductor_params(b_id);
            entries_.push_back(e);
        }
    }

    /// Read the freshly-converged `x` and stash the new
    /// (i_L, V_L) as the next step's (i_L_old, V_L_old).
    /// Called by `run_transient` after each Newton solve.
    void update_from_state(const Vector& x) {
        for (auto& e : entries_) {
            const Real v_from = stamping::read_node_voltage(x, e.from);
            const Real v_to   = stamping::read_node_voltage(x, e.to);
            e.i_L_old = x[e.branch_var_id];
            e.V_L_old = v_from - v_to;
        }
    }

    [[nodiscard]] const std::vector<Entry>&
    entries() const noexcept { return entries_; }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
