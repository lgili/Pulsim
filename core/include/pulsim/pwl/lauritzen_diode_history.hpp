#pragma once

// =============================================================================
// Pulsim — per-step state for the Lauritzen–Mattsson diode
// =============================================================================
//
// The model's state is the STORED CHARGE q_M, which is what makes
// reverse recovery possible at all: a static I-V law has nothing
// to sweep out. This class holds q_M across steps and turns the
// trapezoidal rule into the two coefficients the Newton stamp
// needs.
//
// With A = 1/T_M + 1/tau and f = q_E/T_M − A·q_M, trapezoidal on
//
//     dq_M/dt = q_E/T_M − A·q_M
//
// gives q_M^{n+1}·(1 + hA/2) = q_M^n + (h/2)·f^n
//                              + (h/(2·T_M))·q_E^{n+1}
//
// so q_M^{n+1} = K0 + K1·q_E(v) with
//
//     K1 = (h / (2·T_M)) / (1 + hA/2)          [dimensionless]
//     K0 = (q_M^n + (h/2)·f^n) / (1 + hA/2)    [C]
//
// Both depend only on committed history and the step, so the
// branch stamp stays an ordinary two-terminal nonlinearity — no
// inner iteration, no internal node, no extra unknown.

#include "pulsim/models/lauritzen_diode.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <vector>

namespace pulsim::pwl {

class LauritzenDiodeHistory {
public:
    struct Entry {
        Index branch_id;
        Index from;
        Index to;
        models::LauritzenDiode::Params params;
        Real q_m_prev = Real{0};   //!< stored charge [C]
        Real f_prev   = Real{0};   //!< dq_M/dt at the last point
        //! Coefficients of q_M = K0 + K1·q_E for the step being
        //! solved. Refreshed by `begin_step`.
        Real k0 = Real{0};
        Real k1 = Real{0};
    };

    void init(const topology::Graph& graph,
              const DevicePool& pool) {
        entries_.clear();
        for (Index b_id : pool.lauritzen_diode_branches()) {
            const auto& br = graph.branch(b_id);
            Entry e;
            e.branch_id = b_id;
            e.from = br.from;
            e.to = br.to;
            e.params = pool.lauritzen_diode_params(b_id);
            e.q_m_prev = Real{0};
            e.f_prev = Real{0};
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

    /// Recompute K0/K1 for a step of size `h`.
    ///
    /// Called by `refresh_lauritzen_diodes` on every Newton
    /// iteration rather than once per step, on purpose. The
    /// coefficients embed `h`, and a variable-step engine retries
    /// a rejected step at a different one; a "compute once per
    /// step" API would then stamp the wrong interval while Newton
    /// converged perfectly happily. Recomputing is a handful of
    /// flops per device and removes the failure mode entirely.
    /// It depends only on committed history, so it is idempotent.
    void begin_step(Real h) {
        if (!(h > Real{0})) {
            return;
        }
        for (auto& e : entries_) {
            const Real A = Real{1} / e.params.T_M
                           + Real{1} / e.params.tau;
            const Real den = Real{1} + h * A / Real{2};
            e.k1 = (h / (Real{2} * e.params.T_M)) / den;
            e.k0 = (e.q_m_prev + (h / Real{2}) * e.f_prev) / den;
        }
    }

    /// Commit the just-solved step. `h` must be the step the solve
    /// actually used — the same one the stamp saw.
    void update_from_state(const Vector& x, Real h) {
        if (entries_.empty() || !(h > Real{0})) {
            return;
        }
        for (auto& e : entries_) {
            const Real v =
                stamping::read_node_voltage(x, e.from)
                - stamping::read_node_voltage(x, e.to);
            const Real q_e =
                models::LauritzenDiode::junction_charge<Real>(
                    v, e.params);
            // The SAME affine relation the stamp used — never
            // re-integrated from the voltage, which would let the
            // committed charge disagree with the current the
            // circuit was actually solved with.
            const Real q_m = e.k0 + e.k1 * q_e;
            const Real A = Real{1} / e.params.T_M
                           + Real{1} / e.params.tau;
            e.f_prev = q_e / e.params.T_M - A * q_m;
            e.q_m_prev = q_m;
        }
    }

    /// Seed from a DC operating point: the device sits on its
    /// steady-state curve, where dq_M/dt = 0 and q_M = i·tau.
    void seed_from_dc_op(const Vector& x) {
        for (auto& e : entries_) {
            const Real v =
                stamping::read_node_voltage(x, e.from)
                - stamping::read_node_voltage(x, e.to);
            const Real q_e =
                models::LauritzenDiode::junction_charge<Real>(
                    v, e.params);
            e.q_m_prev = q_e * e.params.tau
                         / (e.params.tau + e.params.T_M);
            e.f_prev = Real{0};
        }
    }

    [[nodiscard]] std::vector<Entry> snapshot() const {
        return entries_;
    }
    void restore(const std::vector<Entry>& snap) {
        entries_ = snap;
    }

    /// Flat (q_M, f) pairs, for `SolverSnapshot`.
    [[nodiscard]] std::vector<Real> to_flat() const {
        std::vector<Real> out;
        out.reserve(entries_.size() * 2);
        for (const auto& e : entries_) {
            out.push_back(e.q_m_prev);
            out.push_back(e.f_prev);
        }
        return out;
    }

    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 2) {
            return;
        }
        for (Size k = 0; k < entries_.size(); ++k) {
            entries_[k].q_m_prev = flat[2 * k];
            entries_[k].f_prev   = flat[2 * k + 1];
        }
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
