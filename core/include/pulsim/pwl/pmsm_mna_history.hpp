#pragma once

// =============================================================================
// Pulsim — per-step state for the MNA-native PMSM
// =============================================================================
//
// The machine's electrical state is the flux linkage λ_abc, and the
// trapezoidal rule on v_k = R i_k + dλ_k/dt needs, per phase, the
// previous step's (λ_k, i_k, v_k):
//
//     (λ_k,new − λ_k,old) = (h/2)·[(v_k,new − R i_k,new)
//                                  + (v_k,old − R i_k,old)]
//
// λ_old is committed from the SAME law the stamp used (L(θ)·i +
// λ_pm(θ) at the converged point), never re-integrated, so the
// committed flux cannot disagree with the currents the circuit was
// actually solved with. Snapshot/restore exists so a rejected or
// halved step can roll the flux back — the saturable inductor's
// history lacks that and therefore has to refuse dt-halving; this
// one does not have to.
//
// TR-BDF2 SECOND STAGE. Unlike the two charge-state devices, where
// only a coefficient moves, here the two stages have DIFFERENT
// SHAPES. Trapezoidal carries the previous derivative:
//
//     (v − R i) + (dλ/dt)_n − (2/h)·(λ − λ_n) = 0
//
// BDF2 is one-sided and carries none of it:
//
//     (v − R i) − (c1·λ + c2·λ_γ + c3·λ_n)/h = 0
//
// so the previous-derivative term is not merely rescaled, it is
// ABSENT. That is why the derivative is now stored explicitly as
// `dlam_old` rather than reconstructed from (v_old, i_old) at the
// stamp: the two stages read one committed quantity instead of each
// deriving its own, and cannot disagree about what the last step
// did.

#include "pulsim/models/pmsm_mna.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <vector>

namespace pulsim::pwl {

class PmsmMnaHistory {
public:
    struct Entry {
        std::array<Index, 3> branch_id{};
        std::array<Index, 3> phase_node{};   //!< a, b, c terminals
        Index neutral = -1;
        std::array<Index, 3> cur_row{};       //!< branch-current unknowns
        Index omega_node = -1;
        Index theta_node = -1;
        models::PmsmMna::Params params;
        std::array<Real, 3> lambda_old{};
        std::array<Real, 3> i_old{};
        std::array<Real, 3> v_old{};          //!< v_phase − v_neutral
        //! dλ/dt at the last committed point (= v_old − R·i_old).
        //! The trapezoidal residual reads it; the BDF2 one does not.
        std::array<Real, 3> dlam_old{};
        //! λ at the γ stage point, for the BDF2 stage.
        std::array<Real, 3> lambda_gamma{};
    };

    void init(const topology::Graph& graph, const DevicePool& pool) {
        entries_.clear();
        for (const auto& m : pool.pmsm_mna_machines()) {
            Entry e;
            e.branch_id = m.branch_id;
            for (Size k = 0; k < 3; ++k) {
                const auto& br = graph.branch(m.branch_id[k]);
                e.phase_node[k] = br.from;
                e.neutral = br.to;
                e.cur_row[k] =
                    pool.branch_var_id_for_inductor(m.branch_id[k],
                                                    graph);
            }
            e.omega_node = m.omega_node;
            e.theta_node = m.theta_node;
            e.params = m.params;
            // At rest with θ from the node's initial value the PM
            // flux is already linked; start λ there, not at zero,
            // or the first step sees a phantom dλ/dt.
            e.lambda_old = models::PmsmMna::lambda_pm(
                e.params, Real{0}, 0);
            entries_.push_back(e);
        }
    }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }
    [[nodiscard]] const std::vector<Entry>& entries() const noexcept {
        return entries_;
    }

    [[nodiscard]] static Real theta_e_of(const Entry& e,
                                         const Vector& x) noexcept {
        return e.params.pole_pairs
               * stamping::read_node_voltage(x, e.theta_node);
    }

    /// λ at the terminal voltages in `x`, from the model's own law.
    [[nodiscard]] static std::array<Real, 3> lambda_at(
        const Entry& e, const Vector& x) {
        const Real th = theta_e_of(e, x);
        const auto ind = models::PmsmMna::inductance(e.params, th);
        const auto lpm = models::PmsmMna::lambda_pm(e.params, th, 0);
        std::array<Real, 3> lam{};
        for (Size k = 0; k < 3; ++k) {
            lam[k] = lpm[k];
            for (Size j = 0; j < 3; ++j) {
                lam[k] += ind.L[k][j] * x[e.cur_row[j]];
            }
        }
        return lam;
    }

    /// Snapshot λ at the γ stage point; the BDF2 history term reads
    /// it. Must run between the two stages.
    void capture_gamma(const Vector& x_gamma) {
        for (auto& e : entries_) {
            e.lambda_gamma = lambda_at(e, x_gamma);
        }
    }

    /// Commit the converged step: (λ, i, v) at x, with λ from the
    /// model's own law at this θ.
    void update_from_state(
        const Vector& x, Real /*h*/,
        TrBdf2Stage /*stage*/ = TrBdf2Stage::Trapezoidal) {
        for (auto& e : entries_) {
            const Real th = theta_e_of(e, x);
            const auto ind = models::PmsmMna::inductance(e.params, th);
            const auto lpm = models::PmsmMna::lambda_pm(e.params, th, 0);
            const Real v_n = stamping::read_node_voltage(x, e.neutral);
            std::array<Real, 3> i{};
            for (Size k = 0; k < 3; ++k) i[k] = x[e.cur_row[k]];
            for (Size k = 0; k < 3; ++k) {
                Real lam = lpm[k];
                for (Size j = 0; j < 3; ++j) lam += ind.L[k][j] * i[j];
                e.lambda_old[k] = lam;
                e.i_old[k] = i[k];
                e.v_old[k] =
                    stamping::read_node_voltage(x, e.phase_node[k]) - v_n;
                // dλ/dt from the device equation itself, exact at
                // the converged point under either stage.
                e.dlam_old[k] = e.v_old[k] - e.params.R_s * i[k];
            }
        }
    }

    /// Seed from a warm-start / DC operating point — same fields as
    /// a commit, so the first stamp sees a consistent previous step.
    void seed_from_dc_op(const Vector& x) { update_from_state(x, Real{0}); }

    [[nodiscard]] std::vector<Entry> snapshot() const { return entries_; }
    void restore(const std::vector<Entry>& snap) { entries_ = snap; }

    /// Flat (λ, i, v) triples per machine. `dlam_old` is not
    /// carried — it is a function of the (v, i) that are — and
    /// `lambda_gamma` lives only between the two stages of one step,
    /// while a snapshot is always taken at a step boundary.
    [[nodiscard]] std::vector<Real> to_flat() const {
        std::vector<Real> out;
        out.reserve(entries_.size() * 9);
        for (const auto& e : entries_) {
            for (Size k = 0; k < 3; ++k) out.push_back(e.lambda_old[k]);
            for (Size k = 0; k < 3; ++k) out.push_back(e.i_old[k]);
            for (Size k = 0; k < 3; ++k) out.push_back(e.v_old[k]);
        }
        return out;
    }
    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 9) return;
        Size p = 0;
        for (auto& e : entries_) {
            for (Size k = 0; k < 3; ++k) e.lambda_old[k] = flat[p++];
            for (Size k = 0; k < 3; ++k) e.i_old[k] = flat[p++];
            for (Size k = 0; k < 3; ++k) e.v_old[k] = flat[p++];
            for (Size k = 0; k < 3; ++k) {
                e.dlam_old[k] = e.v_old[k] - e.params.R_s * e.i_old[k];
            }
        }
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
