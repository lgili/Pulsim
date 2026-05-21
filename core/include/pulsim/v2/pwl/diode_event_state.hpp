#pragma once

// =============================================================================
// Pulsim v2 — Layer 5 V2: DiodeEventState (per-step diode tracker)
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

#include "pulsim/v2/models/switched_diode.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/topology/graph.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::v2::pwl {

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
                diodes_.push_back(DiodeEntry{
                    .branch_id  = branch.id,
                    .switch_idx = static_cast<Size>(switch_idx),
                    .from       = branch.from,
                    .to         = branch.to,
                    .g_on       = p.g_on,
                    .g_off      = p.g_off,
                    .V_th       = p.V_th,
                    .is_on      = false,
                });
            }
            ++switch_idx;
        }
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

    /// Re-decide each diode's state based on the just-computed
    /// state vector `x`. Returns true if any diode flipped.
    bool update_from_state(const Vector& x) {
        bool changed = false;
        for (auto& d : diodes_) {
            const Real v_a = stamping::read_node_voltage(x, d.from);
            const Real v_k = stamping::read_node_voltage(x, d.to);
            const Real v_diode = v_a - v_k;
            const Real g = d.is_on ? d.g_on : d.g_off;
            const Real i_diode = g * v_diode;

            const models::SwitchedDiode::Params p{
                d.g_on, d.g_off, d.V_th};
            const bool next = models::SwitchedDiode::decide_next_state(
                d.is_on, v_diode, i_diode, p);
            if (next != d.is_on) {
                d.is_on = next;
                changed = true;
            }
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
        bool  is_on;
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
        bits.reserve(diodes_.size());
        for (const auto& d : diodes_) {
            bits.push_back(d.is_on);
        }
        return bits;
    }

    void restore_on_bits(const std::vector<bool>& bits) {
        if (bits.size() != diodes_.size()) {
            throw std::invalid_argument(
                "DiodeEventState::restore_on_bits: size "
                "mismatch — expected " +
                std::to_string(diodes_.size()) + ", got " +
                std::to_string(bits.size()));
        }
        for (Size i = 0; i < diodes_.size(); ++i) {
            diodes_[i].is_on = bits[i];
        }
    }

private:
    Size num_switches_;
    std::vector<DiodeEntry> diodes_;
};

}  // namespace pulsim::v2::pwl
