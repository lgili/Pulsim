#pragma once

// =============================================================================
// Pulsim — Layer 5 V2: DiodeEventState (per-step diode tracker)
// =============================================================================
//
// `pulsim-v2-ideal-diode-auto-commutation` Phase 3.
//
// Owns the current ON/OFF state of every SwitchedDiode in the
// circuit. Layer 5 V2's run_transient instantiates one tracker
// per simulation and updates it once per accepted step:
//
//   1. before cache.solve: build the diode bit-mask, combine it
//      with the user's switch_fn output, and pass the merged
//      mask to cache.solve.
//   2. after cache.solve: call update_from_state(x) to re-decide
//      each diode's state for the NEXT step based on the just-
//      computed (v_diode, i_diode).
//
// All diodes start OFF on construction. The user can override
// the initial state via the (deferred) DC operating-point
// OpenSpec.

#include "pulsim/models/switched_diode.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::pwl {

class DiodeEventState {
public:
    DiodeEventState(const topology::Graph& graph,
                     const DevicePool& pool)
        : num_switches_{graph.num_switches()} {
        // Walk graph branches in branch order. For each Switch-
        // kind branch, increment a global switch index; if the
        // pool reports it as a Diode, record a DiodeEntry with
        // that switch index.
        Index switch_idx = 0;
        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind != topology::BranchKind::Switch) {
                continue;
            }
            if (pool.kind_of(branch.id) ==
                DevicePool::StoredKind::Diode) {
                const auto& p = pool.diode_params(branch.id);
                // Branch form (add-branch-form-ideal-switch D5): the ON
                // decision reads the solved current i_s, mapped onto the
                // voltage law as v = V_f + i_s with g_on = 1 (decide:
                // i_s > 0; crossing signal: i_s in amperes).
                const Index var =
                    pool.branch_var_id_for_switch(branch.id, graph);
                diodes_.push_back(DiodeEntry{
                    .branch_id  = branch.id,
                    .switch_idx = static_cast<Size>(switch_idx),
                    .from       = branch.from,
                    .to         = branch.to,
                    .g_on       = var >= 0 ? Real{1} : p.g_on,
                    .g_off      = p.g_off,
                    .V_th       = p.V_th,
                    .V_f        = p.V_f,
                    .is_on      = false,
                    .var        = var,
                    .gate       = p.gate,
                    .I_H        = p.I_H,
                    .ideal      = var >= 0 && pool.branch_R_on(branch.id) == Real{0},
                });
                if (p.gate != models::SwitchedDiode::Gate::None) {
                    ++num_gated_;   // switch indices rise: the index is sorted
                    gated_index_.emplace_back(static_cast<Size>(switch_idx),
                                              diodes_.size() - 1);
                }
                any_ideal_ = any_ideal_ || diodes_.back().ideal;
            }
            ++switch_idx;
        }
        owned_ = topology::SwitchStateMask(num_switches_);
        for (const auto& d : diodes_) owned_.set(d.switch_idx, true);
    }

    [[nodiscard]] Size num_diodes() const noexcept {
        return diodes_.size();
    }

    /// Returns a SwitchStateMask of `num_switches` bits with
    /// diode bits set per current state, non-diode bits cleared.
    [[nodiscard]] topology::SwitchStateMask
    current_diode_mask() const {
        topology::SwitchStateMask mask(num_switches_);
        for (const auto& d : diodes_) {
            mask.set(d.switch_idx, d.is_on);
        }
        return mask;
    }

    /// Returns a SwitchStateMask of `num_switches` bits with 1
    /// at every diode-owned position, 0 elsewhere. Used by Layer
    /// 5 V2 to overlay diode bits on top of the user's switch_fn
    /// mask.
    [[nodiscard]] topology::SwitchStateMask
    diode_owned_bits() const {
        topology::SwitchStateMask mask(num_switches_);
        for (const auto& d : diodes_) {
            mask.set(d.switch_idx, true);
        }
        return mask;
    }

    /// add-pwl-igbt-and-thyristor: read each gated device's gate bit from
    /// the USER mask (its bit before the diode overlay) and open every
    /// gate-held device whose gate is off (a hard turn-off at the edge).
    /// Every decision after this call reads that gate.
    void apply_gates(const topology::SwitchStateMask& user) {
        if (num_gated_ == 0) return;
        for (auto& d : diodes_) {
            if (d.gate == models::SwitchedDiode::Gate::None) continue;
            d.g = user.get(d.switch_idx);
            if (d.gate == models::SwitchedDiode::Gate::Held && !d.g) {
                d.is_on = false;
            }
        }
    }

    /// The mask the solver stamps: the user's mask with every diode bit
    /// replaced by its conduction state (gates applied first).
    [[nodiscard]] topology::SwitchStateMask
    combine(const topology::SwitchStateMask& user) {
        apply_gates(user);
        return user.overlay(current_diode_mask(), owned_);
    }

    /// Bits a switch_fn drives: controlled switches and the gates of the
    /// gated devices.
    [[nodiscard]] topology::SwitchStateMask gated_bits() const {
        topology::SwitchStateMask mask(num_switches_);
        for (const auto& d : diodes_) {
            if (d.gate != models::SwitchedDiode::Gate::None) {
                mask.set(d.switch_idx, true);
            }
        }
        return mask;
    }
    [[nodiscard]] Size num_gated() const noexcept { return num_gated_; }
    /// The gate bits `g` (at each gated device's switch index; 0 elsewhere).
    [[nodiscard]] topology::SwitchStateMask gate_mask() const {
        topology::SwitchStateMask mask(num_switches_);
        for (const auto& [idx, entry] : gated_index_) {
            mask.set(idx, diodes_[entry].g);
        }
        return mask;
    }
    /// The gate a gated device at switch index `idx` was last decided under.
    [[nodiscard]] bool gate_of(Size idx) const noexcept {
        const auto it = std::ranges::lower_bound(
            gated_index_, idx, {}, &std::pair<Size, Size>::first);
        return it != gated_index_.end() && it->first == idx
                   ? diodes_[it->second].g
                   : false;
    }

    /// Re-decide each diode's state based on the just-computed
    /// state vector `x`. Returns true if any diode flipped.
    bool update_from_state(const Vector& x) {
        bool changed = false;
        for (auto& d : diodes_) {
            const bool next = models::SwitchedDiode::decide(
                d.is_on, d.law_voltage(x, d.is_on), d.params());
            if (next != d.is_on) {
                d.is_on = next;
                changed = true;
            }
        }
        return changed;
    }

    /// As above, but diode i flips only when its crossing signal is
    /// further than `noise[i]` past its threshold: a decision read from
    /// round-off is not a decision (engine='trbdf2'; see
    /// run_transient_trbdf2's `decide_diodes`).
    bool update_from_state(const Vector& x, const std::vector<Real>& noise) {
        bool changed = false;
        for (Size i = 0; i < diodes_.size(); ++i) {
            auto& d = diodes_[i];
            const Real v = d.law_voltage(x, d.is_on);
            const bool next = models::SwitchedDiode::decide(d.is_on, v, d.params());
            if (next != d.is_on
                && std::abs(models::SwitchedDiode::crossing_signal(
                       d.is_on, v, d.params())) > noise[i]) {
                d.is_on = next;
                changed = true;
            }
        }
        return changed;
    }

    /// As update_from_state(x), except that an OFF ideal (R_on = 0)
    /// branch-form diode turns ON only past a round-off floor,
    /// 1e-12·max(1, ‖x‖∞): with V_f = 0 across a closed ideal switch (an
    /// inverter leg) its voltage is the loop's residual, and a positive ulp
    /// turned it ON into an undefined loop current (ideal_voltage_loop).
    /// Without such a diode it is update_from_state(x), bit for bit.
    bool update_from_state_floored(const Vector& x) {
        if (!any_ideal_) return update_from_state(x);
        const Real floor = Real{1e-12} * std::max(Real{1}, x.lpNorm<Eigen::Infinity>());
        bool changed = false;
        for (auto& d : diodes_) {
            const Real v = d.law_voltage(x, d.is_on);
            const bool next = models::SwitchedDiode::decide(d.is_on, v, d.params());
            if (next == d.is_on) continue;
            if (d.ideal && !d.is_on &&
                !(models::SwitchedDiode::crossing_signal(false, v, d.params()) > floor))
                continue;
            d.is_on = next;
            changed = true;
        }
        return changed;
    }

    /// Public DiodeEntry layout for Layer 5 V2.2's sub-step
    /// commutation timing. Layer 5 reads the entries to compute
    /// the watched signal at the pre/post-step states.
    struct DiodeEntry {
        Index branch_id;
        Size  switch_idx;
        Index from;
        Index to;
        Real  g_on;
        Real  g_off;
        Real  V_th;
        Real  V_f;
        bool  is_on;
        Index var = -1;   // branch-form current unknown, or -1
        models::SwitchedDiode::Gate gate = models::SwitchedDiode::Gate::None;
        Real  I_H = Real{0};
        bool  g = true;   // gate bit of the latest apply_gates
        bool  ideal = false;   // branch form with R_on = 0

        /// The voltage the diode law reads for state `on`: v_a - v_k; for
        /// an ON branch-form diode V_f + i_s (see the constructor).
        [[nodiscard]] Real law_voltage(const Vector& x, bool on) const noexcept {
            if (on && var >= 0) return V_f + x[var];
            return stamping::read_node_voltage(x, from) -
                   stamping::read_node_voltage(x, to);
        }

        [[nodiscard]] models::SwitchedDiode::Params params() const noexcept {
            return {.g_on = g_on, .g_off = g_off, .V_th = V_th, .V_f = V_f,
                    .gate = gate, .I_H = I_H, .g = g};
        }
    };

    [[nodiscard]] std::span<const DiodeEntry>
    entries() const noexcept {
        return std::span<const DiodeEntry>{
            diodes_.data(), diodes_.size()};
    }

    /// Reset all diodes to OFF.
    void reset() noexcept {
        for (auto& d : diodes_) {
            d.is_on = false;
        }
    }

    /// Update a diode's on-state parameters in place after an online
    /// electrothermal feedback refresh so commutation decisions stay in
    /// sync with the refactored MNA matrix.
    void update_on_state(Index branch_id, Real new_g_on, Real new_v_f,
                         Real new_v_th) noexcept {
        for (auto& d : diodes_) {
            if (d.branch_id == branch_id) {
                if (d.var < 0) d.g_on = new_g_on;
                d.V_f = new_v_f;
                d.V_th = new_v_th;
                if (d.var >= 0) d.ideal = false;
                break;
            }
        }
        any_ideal_ = false;
        for (const auto& d : diodes_) any_ideal_ = any_ideal_ || d.ideal;
    }

    // ----- Layer 5 V3: snapshot/restore for substep correction -----
    //
    // `snapshot_on_bits()` returns the current `is_on` flag for
    // every registered diode, in registration order.
    // `restore_on_bits(bits)` overwrites the flags from such a
    // snapshot. The bit-vector size MUST match the number of
    // diodes; mismatch throws `std::invalid_argument`.
    //
    // Used by run_transient's substep correction path: before a
    // step, snapshot the diode bits; if a commutation event is
    // detected mid-step, restore and redo as two sub-steps with
    // the original pre-event mask, then update normally.
    [[nodiscard]] std::vector<bool> snapshot_on_bits() const {
        std::vector<bool> bits;
        snapshot_on_bits_into(bits);
        return bits;
    }

    /// Fill-in-place variant (v2.0 Phase 1, audit finding
    /// `per-step-heap-allocations`): reuses `out`'s capacity — no
    /// heap traffic in the per-step / per-event-iteration loop.
    void snapshot_on_bits_into(std::vector<bool>& out) const {
        out.clear();
        out.reserve(diodes_.size());
        for (const auto& d : diodes_) {
            out.push_back(d.is_on);
        }
    }

    /// Accepts the conduction bits alone, or followed by one gate bit
    /// per gated device (`snapshot_mode_bits`).
    void restore_on_bits(const std::vector<bool>& bits) {
        if (bits.size() != diodes_.size() &&
            bits.size() != diodes_.size() + num_gated_) {
            throw std::invalid_argument(
                "DiodeEventState::restore_on_bits: size "
                "mismatch — expected " +
                std::to_string(diodes_.size()) + " (or " +
                std::to_string(diodes_.size() + num_gated_) +
                " with the gate bits), got " +
                std::to_string(bits.size()));
        }
        Size g = diodes_.size();
        for (Size i = 0; i < diodes_.size(); ++i) {
            diodes_[i].is_on = bits[i];
            if (bits.size() > diodes_.size() &&
                diodes_[i].gate != models::SwitchedDiode::Gate::None) {
                diodes_[i].g = bits[g++];
            }
        }
    }

    /// The solver-owned mode a run resumes from (`SolverSnapshot::mask`):
    /// the conduction bits, then the gate bit `g` each gated device was
    /// last decided under, in registration order (add-pwl-igbt-and-
    /// thyristor D13). `g` is not `switch_fn(t)`: at a gate edge on the
    /// seam it is the pre-edge gate the edge is compared against.
    [[nodiscard]] std::vector<bool> snapshot_mode_bits() const {
        auto bits = snapshot_on_bits();
        for (const auto& d : diodes_) {
            if (d.gate != models::SwitchedDiode::Gate::None) bits.push_back(d.g);
        }
        return bits;
    }

private:
    Size num_switches_;
    Size num_gated_ = 0;
    std::vector<std::pair<Size, Size>> gated_index_;   //!< (switch idx, entry)
    bool any_ideal_ = false;
    std::vector<DiodeEntry> diodes_;
    topology::SwitchStateMask owned_{0};
};

}  // namespace pulsim::pwl
