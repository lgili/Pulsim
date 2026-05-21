#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V1: HistoryState (per-step trap-companion history)
// =============================================================================
//
// `pulsim-v2-trapezoidal-companion` Phase 5.
//
// Owns the previous-step (v, i) of every dynamic device
// (Capacitor or Inductor) in the circuit. Layer 5's
// run_transient instantiates one HistoryState per simulation and
// updates it once per accepted step.
//
// Workflow inside the time-stepping loop:
//
//   1. b_extra = history.compute_b_extra(dt)
//      ^^^ contributes the trap history-term to the right rows
//          of the RHS (cap → node rows, inductor → constraint row).
//   2. cache.solve(mask, b_extra, x)
//   3. history.update_from_state(x, dt)
//      ^^^ reads the just-computed (v_{n+1}, i_{n+1}) and stores
//          them for next step's history.
//
// At simulation start the entries are zero (all-zero IC for V0).
// DC operating-point pre-charge is a follow-up OpenSpec.

#include "pulsim/v2/models/capacitor.hpp"
#include "pulsim/v2/models/inductor.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <vector>

namespace pulsim::v2::pwl {

// -----------------------------------------------------------------------------
// HistoryEntry — one record per dynamic device.
//
// `kind` discriminates whether to use Capacitor:: or Inductor::
// helpers when computing history / updating state.
//
// For an Inductor the entry also caches the absolute state-vector
// index of the branch-current unknown, so update_from_state can
// read i_L directly from `x`.
// -----------------------------------------------------------------------------
struct HistoryEntry {
    Index branch_id;
    Index from;
    Index to;
    DevicePool::StoredKind kind;
    Index inductor_branch_var_id;   // valid only when kind == Inductor
    Real v_prev = Real{0};
    Real i_prev = Real{0};
    // Cached params (avoid pool lookup on the hot path).
    Real C_or_L;   // farads (Capacitor) or henries (Inductor)
};

class HistoryState {
public:
    HistoryState(const topology::Graph& graph, const DevicePool& pool)
        : state_size_{pool.state_size(graph)} {
        entries_.reserve(pool.num_dynamic_branches());
        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind != topology::BranchKind::PassiveLinear) {
                continue;
            }
            const auto k = pool.kind_of(branch.id);
            if (k == DevicePool::StoredKind::Capacitor) {
                HistoryEntry e{};
                e.branch_id = branch.id;
                e.from      = branch.from;
                e.to        = branch.to;
                e.kind      = k;
                e.inductor_branch_var_id = -1;
                e.C_or_L    = pool.capacitor_params(branch.id).C;
                entries_.push_back(e);
            } else if (k == DevicePool::StoredKind::Inductor) {
                HistoryEntry e{};
                e.branch_id = branch.id;
                e.from      = branch.from;
                e.to        = branch.to;
                e.kind      = k;
                e.inductor_branch_var_id =
                    pool.branch_var_id_for_inductor(branch.id, graph);
                e.C_or_L    = pool.inductor_params(branch.id).L;
                entries_.push_back(e);
            }
        }
    }

    /// Number of dynamic-device entries tracked.
    [[nodiscard]] Size num_dynamic_branches() const noexcept {
        return entries_.size();
    }

    /// Returns a fresh state-size vector populated with the
    /// trap-companion history contributions on the right rows.
    ///
    /// Capacitor (from, to):
    ///   I_hist = G_eq · v_prev + i_prev   (positive into from-node)
    ///   b_extra(from) += −I_hist
    ///   b_extra(to)   += +I_hist
    ///
    /// Inductor (constraint row at branch_var_id):
    ///   I_hist,L = i_prev + (dt/2L) · v_prev
    ///   b_extra(branch_var_id) += −(2L/dt) · I_hist,L
    ///
    /// `dt` must equal the dt that the cache was built with. Layer
    /// 5's run_transient asserts that before calling.
    [[nodiscard]] Vector compute_b_extra(Real dt) const {
        Vector b_extra =
            Vector::Zero(static_cast<Index>(state_size_));
        if (entries_.empty() || dt <= Real{0}) {
            return b_extra;
        }
        for (const auto& e : entries_) {
            if (e.kind == DevicePool::StoredKind::Capacitor) {
                models::Capacitor::Params p{e.C_or_L};
                const Real i_hist = models::Capacitor::history_term(
                    e.v_prev, e.i_prev, dt, p);
                if (stamping::node_is_active(e.from)) {
                    b_extra[e.from] -= i_hist;
                }
                if (stamping::node_is_active(e.to)) {
                    b_extra[e.to]   += i_hist;
                }
            } else {
                // Inductor
                models::Inductor::Params p{e.C_or_L};
                const Real i_hist_L = models::Inductor::history_term(
                    e.v_prev, e.i_prev, dt, p);
                // Constraint row residual (in J·x + b = 0 form):
                //   v_from − v_to − (2L/dt)·i_L + (2L/dt)·I_hist,L = 0
                // So the assembled `b` part of that row equals
                //   b(row) = +(2L/dt) · I_hist,L
                // and since `cache.solve` does `rhs = -(b_constant
                // + b_extra)`, this contribution lives in b_extra
                // with a POSITIVE sign.
                const Real two_L_over_dt =
                    Real{1} / models::Inductor::g_eq_inv(dt, p);
                b_extra[e.inductor_branch_var_id] +=
                    two_L_over_dt * i_hist_L;
            }
        }
        return b_extra;
    }

    /// Reads (v_{n+1}, i_{n+1}) per dynamic device from x and
    /// stores them as the new (v_prev, i_prev). Called by Layer 5
    /// AFTER each cache.solve.
    ///
    /// Capacitor:
    ///   v_new = v_from − v_to       (read from x)
    ///   i_new = G_eq · (v_new − v_prev) − i_prev     (companion)
    ///
    /// Inductor:
    ///   v_new = v_from − v_to
    ///   i_new = x[branch_var_id]    (direct unknown)
    void update_from_state(const Vector& x, Real dt) {
        if (entries_.empty() || dt <= Real{0}) {
            return;
        }
        for (auto& e : entries_) {
            const Real v_from = stamping::read_node_voltage(x, e.from);
            const Real v_to   = stamping::read_node_voltage(x, e.to);
            const Real v_new  = v_from - v_to;
            if (e.kind == DevicePool::StoredKind::Capacitor) {
                models::Capacitor::Params p{e.C_or_L};
                const Real g_eq = models::Capacitor::g_eq(dt, p);
                const Real i_new = g_eq * (v_new - e.v_prev) - e.i_prev;
                e.v_prev = v_new;
                e.i_prev = i_new;
            } else {
                // Inductor: read i_L directly from x.
                const Real i_new = x[e.inductor_branch_var_id];
                e.v_prev = v_new;
                e.i_prev = i_new;
            }
        }
    }

    /// Zero all entries — used at start of simulation (all-zero IC).
    void reset() noexcept {
        for (auto& e : entries_) {
            e.v_prev = Real{0};
            e.i_prev = Real{0};
        }
    }

    /// Seed each entry from a DC operating-point state vector
    /// `dc_x`. Sets:
    ///   Capacitor: v_prev = v_C from dc_x, i_prev = 0
    ///   Inductor:  v_prev = 0,             i_prev = i_L from dc_x
    /// Used by Layer 5 V3's `start_from_dc_op = true` path.
    void seed_from_dc_op(const Vector& dc_x) {
        for (auto& e : entries_) {
            const Real v_from = stamping::read_node_voltage(dc_x, e.from);
            const Real v_to   = stamping::read_node_voltage(dc_x, e.to);
            const Real v_new  = v_from - v_to;
            if (e.kind == DevicePool::StoredKind::Capacitor) {
                e.v_prev = v_new;
                e.i_prev = Real{0};
            } else {
                // Inductor
                e.v_prev = Real{0};
                e.i_prev = dc_x[e.inductor_branch_var_id];
            }
        }
    }

    /// Diagnostic accessor.
    [[nodiscard]] const std::vector<HistoryEntry>& entries() const noexcept {
        return entries_;
    }

private:
    Size state_size_;
    std::vector<HistoryEntry> entries_;
};

}  // namespace pulsim::v2::pwl
