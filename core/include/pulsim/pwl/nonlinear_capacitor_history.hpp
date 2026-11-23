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
#include "pulsim/device/device_state.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
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
        q_gamma_.assign(entries_.size(), Real{0});
    }

    /// TR-BDF2: the charge at the stage point, one per device, read by
    /// the BDF2 stage-2 stamp and by `commit_bdf2_stage2`.
    void capture_gamma(const Vector& x_gamma) {
        Size qi = 0;
        for (const auto& e : entries_) {
            const Real vg = stamping::read_node_voltage(x_gamma, e.from)
                            - stamping::read_node_voltage(x_gamma, e.to);
            q_gamma_[qi++] = models::NonlinearCapacitor::charge(e.params, vg);
        }
    }
    [[nodiscard]] const std::vector<Real>& q_gamma() const noexcept {
        return q_gamma_;
    }
    /// The interface path's stage capture writes it through the law.
    [[nodiscard]] std::vector<Real>& mutable_q_gamma() noexcept {
        return q_gamma_;
    }

    /// Commit a TR-BDF2 step under the BDF2 rule its second stage used
    /// (`h_full` = the whole step) — never re-derived from C(v)dv/dt,
    /// which would not conserve charge.
    void commit_bdf2_stage2(const Vector& x_end, Real h_full) {
        const Real c1 = Real{2} + std::sqrt(Real{2});
        const Real gam = Real{2} - std::sqrt(Real{2});
        const Real rho = (Real{1} - gam) / gam;
        const Real c2 = -(Real{1} + rho) / (Real{1} - gam);
        const Real c3 = Real{1} / std::sqrt(Real{2});
        Size qi = 0;
        for (auto& e : entries_) {
            const Real v = stamping::read_node_voltage(x_end, e.from)
                           - stamping::read_node_voltage(x_end, e.to);
            const Real q = models::NonlinearCapacitor::charge(e.params, v);
            e.i_prev = (c1 * q + c2 * q_gamma_[qi] + c3 * e.q_prev) / h_full;
            e.v_prev = v;
            e.q_prev = q;
            ++qi;
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
    /// solve actually used; `backward_euler` a restart stage
    /// rule (i = (Q - Q_n)/h, no i_n), as the stamp used it.
    void update_from_state(const Vector& x, Real h,
                           bool backward_euler = false) {
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
            e.i_prev = backward_euler
                ? (q - e.q_prev) / h
                : (Real{2} / h) * (q - e.q_prev) - e.i_prev;
            e.v_prev = v;
            e.q_prev = q;
        }
    }

    /// SDIRK restart: pseudo previous charge q0 + c·(q1 − q0) (see
    /// HistoryState::extrapolate_previous).
    void extrapolate_previous(const Vector& x1, Real c) {
        for (auto& e : entries_) {
            const Real v = stamping::read_node_voltage(x1, e.from)
                         - stamping::read_node_voltage(x1, e.to);
            e.q_prev += c * (models::NonlinearCapacitor::charge(e.params, v)
                             - e.q_prev);
        }
    }

    /// TR-BDF2 post-event refresh of the current slot from a
    /// trapezoidal probe of step `h` (the probe's own companion
    /// current); (v, q) are not touched.
    void refresh_derivatives(const Vector& x, Real h) {
        if (entries_.empty() || !(h > Real{0})) {
            return;
        }
        for (auto& e : entries_) {
            const Real v =
                stamping::read_node_voltage(x, e.from)
                - stamping::read_node_voltage(x, e.to);
            e.i_prev = (Real{2} / h)
                           * (models::NonlinearCapacitor::charge(e.params, v)
                              - e.q_prev)
                       - e.i_prev;
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

    // ---- device::StatefulKind (device/kind.hpp) ----
    static constexpr device::StateBlock kBlock = device::StateBlock::NonlinearCapacitor;
    static constexpr DevicePool::StoredKind kStoredKind =
        DevicePool::StoredKind::NonlinearCapacitor;
    static constexpr Size kFlatPerDevice = 3;
    /// Every real an entry carries from step to step, the snapshot's
    /// (`to_flat` order) first, then the stage scratch: the in-run
    /// rollback copies exactly these.
    template <class E, class F> static void for_each_state(E& e, F&& f) {
        f(e.v_prev);
        f(e.q_prev);
        f(e.i_prev);
    }
    [[nodiscard]] std::vector<Entry>& mutable_entries() noexcept {
        return entries_;
    }

    // ---- the stage form (βh, ψ) of integrator='esdirk43' ----
    // One q row per device: q = Q(v), q̇ = the committed current. A stage is
    // backward Euler of βh from the base q_prev = ψ (the stamps' BE rule).
    [[nodiscard]] Size stage_rows() const noexcept { return entries_.size(); }
    void stage_start(Real hh, Real* q0, Real* hqd0) const {
        for (const auto& e : entries_) {
            *q0++ = e.q_prev;
            *hqd0++ = hh * e.i_prev;
        }
    }
    void stage_base(const Real* psi, const Real* /*dpsi*/) {
        for (auto& e : entries_) e.q_prev = *psi++;
    }
    void stage_increment(const Vector& x, const Real* psi, Real* d) const {
        for (const auto& e : entries_) {
            const Real v = stamping::read_node_voltage(x, e.from)
                         - stamping::read_node_voltage(x, e.to);
            *d++ = models::NonlinearCapacitor::charge(e.params, v) - *psi++;
        }
    }
    void stage_clear() noexcept {}
    // ---- TR-BDF2's q rows (the error estimate) ----
    /// Q − Q_n at γ (the stage-1 capture) and at the step end (Q(v), as
    /// `commit_bdf2_stage2` computes it).
    void trbdf2_increments(const Vector& x1, Real* dqg, Real* dq1) const {
        Size qi = 0;
        for (const auto& e : entries_) {
            const Real v = stamping::read_node_voltage(x1, e.from)
                         - stamping::read_node_voltage(x1, e.to);
            *dq1++ = models::NonlinearCapacitor::charge(e.params, v) - e.q_prev;
            *dqg++ = q_gamma_[qi++] - e.q_prev;
        }
    }
    /// atol_q / atol per row: the zero-bias capacitance dQ/dv(0).
    void q_scales(Real* s) const {
        for (const auto& e : entries_) {
            *s++ = std::abs(models::NonlinearCapacitor::capacitance(e.params, Real{0}));
        }
    }

    /// Commit under the rule the solve stamped (BDF2 stage: the stage
    /// commit; backward Euler: the restart rule).
    void commit(const Vector& x, Real h, TrBdf2Stage stage) {
        if (stage == TrBdf2Stage::Bdf2Stage2) {
            commit_bdf2_stage2(x, h);
        } else {
            update_from_state(x, h, stage == TrBdf2Stage::BackwardEuler);
        }
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
    std::vector<Real> q_gamma_;   // TR-BDF2 stage scratch, not state
};

}  // namespace pulsim::pwl
