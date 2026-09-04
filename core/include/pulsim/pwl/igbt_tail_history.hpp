#pragma once

// =============================================================================
// Pulsim — per-step state for the IGBT turn-off tail
// =============================================================================
//
// An IGBT is a PNP transistor driven by a MOSFET. When the gate
// falls below threshold the MOS channel cuts off in nanoseconds,
// but the minority carriers stored in the n- drift region can only
// RECOMBINE — a tail current that keeps flowing with the full rail
// already across the device, and that is 40-70 % of the turn-off
// energy.
//
// THE SPLIT. Let i_ss(v) be the device's steady-state law. The
// collector current is split into the part the channel carries
// directly and the part the stored charge carries:
//
//     i_C  = (1 − k)·i_ss(v) + Q/tau
//     dQ   = k·i_ss(v) − Q/tau
//     dt
//
// In equilibrium Q = k·i_ss·tau, so Q/tau = k·i_ss and the two
// terms add back to i_ss EXACTLY. The DC curve is therefore
// untouched by construction — a property worth having, because it
// means enabling a tail cannot silently move a conduction-loss
// number that was already validated.
//
// At turn-off i_ss collapses, and i_C = Q/tau decays from k·I_C
// with time constant tau. Both are datasheet quantities read
// straight off a turn-off waveform.
//
// Trapezoidal on Q closes in explicit form, exactly as for the
// Lauritzen diode:
//
//     Q^{n+1}·(1 + h/(2·tau)) = Q^n + (h/2)·f^n
//                               + (h·k/2)·i_ss^{n+1}
//
// so Q = K0 + K1·i_ss(v) is affine in i_ss and the branch stays an
// ordinary nonlinearity — no internal node, no extra unknown.
//
// TR-BDF2 SECOND STAGE. Same shape, different coefficients. With
// dQ/dt = (c1 Q + c2 Q_gamma + c3 Q_n)/h,
//
//     Q (c1/h + 1/tau) = k·i_ss - (c2 Q_gamma + c3 Q_n)/h
//
// so K1 = k/(c1/h + 1/tau) and
// K0 = -(c2 Q_gamma + c3 Q_n)/h / (c1/h + 1/tau). The conductance
// is unchanged; only the history term moves.

#include "pulsim/models/igbt_level1.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <vector>

namespace pulsim::pwl {

class IgbtTailHistory {
public:
    struct Entry {
        Index branch_id;
        Index collector;
        Index emitter;
        Index gate;
        models::IgbtLevel1::Params params;
        Real q_prev = Real{0};   //!< stored charge [C]
        Real f_prev = Real{0};   //!< dQ/dt at the last point
        //! Q at the gamma stage point, for the BDF2 stage.
        Real q_gamma = Real{0};
        //! Coefficients of Q = K0 + K1·i_ss for the step in hand.
        Real k0 = Real{0};
        Real k1 = Real{0};
    };

    void init(const topology::Graph& graph,
              const DevicePool& pool) {
        entries_.clear();
        for (Index b_id : pool.igbt_level1_branches()) {
            const auto& p = pool.igbt_level1_params(b_id);
            if (!models::IgbtLevel1::has_tail(p)) {
                continue;   // no tail: the static law is exact
            }
            const auto& br = graph.branch(b_id);
            Entry e;
            e.branch_id = b_id;
            e.collector = br.from;
            e.emitter = br.to;
            e.gate = pool.igbt_level1_gate_node(b_id);
            e.params = p;
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
    /// Called by the stamp on every Newton iteration rather than
    /// once per step, for the same reason as the Lauritzen diode:
    /// the coefficients embed `h`, a variable-step engine retries
    /// rejected steps at a different one, and a stale pair would
    /// integrate the wrong interval while Newton converged
    /// perfectly happily. It reads only committed history, so it
    /// is idempotent.
    void begin_step(Real h,
                    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) {
        if (!(h > Real{0})) {
            return;
        }
        const auto k = trbdf2_coeffs();
        for (auto& e : entries_) {
            if (stage == TrBdf2Stage::Bdf2Stage2) {
                const Real den = k.c1 / h + Real{1} / e.params.tau_tail;
                e.k1 = e.params.k_tail / den;
                e.k0 = -(k.c2 * e.q_gamma + k.c3 * e.q_prev) / h / den;
            } else {
                const Real den =
                    Real{1} + h / (Real{2} * e.params.tau_tail);
                e.k1 = (h * e.params.k_tail / Real{2}) / den;
                e.k0 = (e.q_prev + (h / Real{2}) * e.f_prev) / den;
            }
        }
    }

    /// Snapshot Q at the gamma stage point from the stage-1
    /// solution; the BDF2 history term reads it.
    void capture_gamma(const Vector& x_gamma) {
        for (auto& e : entries_) {
            const Real i_ss = steady_state_at(e, x_gamma);
            e.q_gamma = e.k0 + e.k1 * i_ss;
        }
    }

    /// Commit the just-solved step at the step and stage it
    /// actually used — `f_prev` feeds the next step's trapezoidal
    /// history term, so it has to be the derivative the method
    /// really produced.
    void update_from_state(
        const Vector& x, Real h,
        TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) {
        if (entries_.empty() || !(h > Real{0})) {
            return;
        }
        const auto k = trbdf2_coeffs();
        for (auto& e : entries_) {
            const Real i_ss = steady_state_at(e, x);
            // The SAME affine relation the stamp used, so the
            // committed charge cannot disagree with the current
            // the circuit was solved with.
            const Real q = e.k0 + e.k1 * i_ss;
            if (stage == TrBdf2Stage::Bdf2Stage2) {
                e.f_prev = (k.c1 * q + k.c2 * e.q_gamma
                            + k.c3 * e.q_prev) / h;
            } else {
                e.f_prev = e.params.k_tail * i_ss
                           - q / e.params.tau_tail;
            }
            e.q_prev = q;
        }
    }

    /// Seed from a DC operating point: charge in equilibrium,
    /// Q = k·i_ss·tau, where the split adds back to i_ss exactly.
    void seed_from_dc_op(const Vector& x) {
        for (auto& e : entries_) {
            const Real i_ss = steady_state_at(e, x);
            e.q_prev = e.params.k_tail * i_ss * e.params.tau_tail;
            e.f_prev = Real{0};
        }
    }

    /// The steady-state law at this entry's terminal voltages.
    [[nodiscard]] static Real steady_state_at(const Entry& e,
                                               const Vector& x) {
        const Real v[3] = {
            stamping::read_node_voltage(x, e.collector),
            stamping::read_node_voltage(x, e.emitter),
            stamping::read_node_voltage(x, e.gate),
        };
        return models::IgbtLevel1::steady_state_current<Real>(
            v, e.params);
    }

    [[nodiscard]] std::vector<Entry> snapshot() const {
        return entries_;
    }
    void restore(const std::vector<Entry>& snap) {
        entries_ = snap;
    }

    /// Flat (Q, f) pairs, for `SolverSnapshot`.
    [[nodiscard]] std::vector<Real> to_flat() const {
        std::vector<Real> out;
        out.reserve(entries_.size() * 2);
        for (const auto& e : entries_) {
            out.push_back(e.q_prev);
            out.push_back(e.f_prev);
        }
        return out;
    }

    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 2) {
            return;
        }
        for (Size k = 0; k < entries_.size(); ++k) {
            entries_[k].q_prev = flat[2 * k];
            entries_[k].f_prev = flat[2 * k + 1];
        }
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
