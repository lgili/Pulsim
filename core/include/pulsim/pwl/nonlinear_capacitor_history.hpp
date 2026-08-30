#pragma once

// =============================================================================
// Pulsim — per-step state of every charge-based nonlinear capacitor
// =============================================================================
//
// Phase 4, audit C.1. A Coss carries (v, Q, i) from the previous
// accepted step, exactly as a linear capacitor carries (v, i) in
// `HistoryState`.
//
// TWO THINGS THIS DOES THAT `SaturableInductorHistory` DOES NOT,
// and they are the reasons that device has to refuse a variable
// step:
//
//   * the step size is passed IN per solve rather than captured
//     once from `opts.dt`, so a sub-step or a rejected-and-retaken
//     step stamps its own h;
//   * `snapshot()` / `restore()` exist, so a step that is thrown
//     away can be rolled back instead of leaving committed state
//     behind.
//
// Together they are what lets a Coss run on the variable-step
// engine, where a resonant transition is exactly the thing worth
// resolving adaptively.

#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

class NonlinearCapacitorHistory {
public:
    struct Entry {
        Index branch_id;
        Index from;
        Index to;
        models::NonlinearCapacitor::Params params;
        Real v_prev = Real{0};
        Real q_prev = Real{0};
        Real i_prev = Real{0};
    };

    void init(const topology::Graph& graph,
              const DevicePool& pool) {
        entries_.clear();
        for (Index b_id : pool.nonlinear_capacitor_branches()) {
            const auto& br = graph.branch(b_id);
            Entry e;
            e.branch_id = b_id;
            e.from = br.from;
            e.to = br.to;
            e.params = pool.nonlinear_capacitor_params(b_id);
            e.v_prev = Real{0};
            e.q_prev = models::NonlinearCapacitor::charge(
                e.params, Real{0});
            e.i_prev = Real{0};
            entries_.push_back(e);
        }
    }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }
    [[nodiscard]] const std::vector<Entry>& entries() const
        noexcept {
        return entries_;
    }

    /// Commit the just-solved step. `h` must be the step this
    /// solve actually used.
    void update_from_state(const Vector& x, Real h) {
        if (entries_.empty() || !(h > Real{0})) {
            return;
        }
        for (auto& e : entries_) {
            const Real v =
                stamping::read_node_voltage(x, e.from)
                - stamping::read_node_voltage(x, e.to);
            const Real q =
                models::NonlinearCapacitor::charge(e.params, v);
            // The companion's own current, from the same rule the
            // stamp used — never re-derived from C(v)·dv/dt, which
            // would not conserve charge.
            e.i_prev = (Real{2} / h) * (q - e.q_prev) - e.i_prev;
            e.v_prev = v;
            e.q_prev = q;
        }
    }

    /// Seed from a DC operating point: the device holds its bias
    /// charge and no current.
    void seed_from_dc_op(const Vector& x) {
        for (auto& e : entries_) {
            const Real v =
                stamping::read_node_voltage(x, e.from)
                - stamping::read_node_voltage(x, e.to);
            e.v_prev = v;
            e.q_prev =
                models::NonlinearCapacitor::charge(e.params, v);
            e.i_prev = Real{0};
        }
    }

    [[nodiscard]] std::vector<Entry> snapshot() const {
        return entries_;
    }
    void restore(const std::vector<Entry>& snap) {
        entries_ = snap;
    }

    /// Flat (v, q, i) per device, for a run snapshot.
    [[nodiscard]] std::vector<Real> to_flat() const {
        std::vector<Real> out;
        out.reserve(entries_.size() * 3);
        for (const auto& e : entries_) {
            out.push_back(e.v_prev);
            out.push_back(e.q_prev);
            out.push_back(e.i_prev);
        }
        return out;
    }

    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 3) {
            throw std::invalid_argument(
                "NonlinearCapacitorHistory::from_flat: expected "
                + std::to_string(entries_.size() * 3)
                + " values (3 per device) but got "
                + std::to_string(flat.size()));
        }
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            entries_[i].v_prev = flat[3 * i];
            entries_[i].q_prev = flat[3 * i + 1];
            entries_[i].i_prev = flat[3 * i + 2];
        }
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
