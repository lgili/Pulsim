#pragma once

// =============================================================================
// Pulsim v2 — Layer 4: DevicePool (branch_id → params registry)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 1.
//
// Layer 1's `Graph` knows each branch's topological `kind` but NOT
// its parameters. This pool bridges that gap: it maps a branch_id
// to the (kind, parameters) tuple the stampers need.
//
// V0 supports three kinds: Resistor, VoltageSource, Switch. Adding
// new device types (Capacitor, Inductor, MOSFET, …) is one new
// `add_*` method + one new `StoredKind` variant + one Layer 4
// dispatch arm in `assemble_segment`.

#include "pulsim/v2/models/resistor.hpp"
#include "pulsim/v2/models/voltage_source.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace pulsim::v2::pwl {

class DevicePool {
public:
    enum class StoredKind { Resistor, VoltageSource, Switch };

    struct SwitchParams {
        Real g_on;
        Real g_off;
    };

    // -------- Add methods ---------------------------------------------------

    void add_resistor(Index branch_id, models::Resistor::Params p) {
        entries_[branch_id] = Entry{p};
    }

    void add_voltage_source(Index branch_id,
                             models::VoltageSource::Params p) {
        entries_[branch_id] = Entry{p};
        // Assign the branch-current row id now: it lives at
        // num_nodes + (count of sources so far).
        source_branch_var_id_[branch_id] =
            static_cast<Index>(num_sources_++);
    }

    void add_switch(Index branch_id, Real g_on, Real g_off) {
        entries_[branch_id] = Entry{SwitchParams{g_on, g_off}};
    }

    // -------- Lookups -------------------------------------------------------

    [[nodiscard]] StoredKind kind_of(Index branch_id) const {
        const auto it = entries_.find(branch_id);
        if (it == entries_.end()) {
            throw std::out_of_range(
                "DevicePool::kind_of: branch_id " +
                std::to_string(branch_id) + " not registered");
        }
        return static_cast<StoredKind>(it->second.index());
    }

    [[nodiscard]] const models::Resistor::Params&
    resistor_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::Resistor::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::resistor_params: branch " +
                std::to_string(branch_id) + " is not a Resistor");
        }
        return std::get<models::Resistor::Params>(entry);
    }

    [[nodiscard]] const models::VoltageSource::Params&
    voltage_source_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::VoltageSource::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::voltage_source_params: branch " +
                std::to_string(branch_id) + " is not a VoltageSource");
        }
        return std::get<models::VoltageSource::Params>(entry);
    }

    [[nodiscard]] Real switch_g_on(Index branch_id) const {
        return switch_params_at(branch_id).g_on;
    }
    [[nodiscard]] Real switch_g_off(Index branch_id) const {
        return switch_params_at(branch_id).g_off;
    }

    // -------- State-vector layout helpers -----------------------------------

    /// Returns the absolute state-vector index of the branch-current
    /// unknown for a Source-kind branch. Throws if the branch_id is
    /// not registered as a voltage source.
    ///
    /// The "absolute" index requires knowing the graph's node count
    /// (sources sit at [N, N+M)). The pool stores only the relative
    /// offset; callers add `graph.num_nodes()` to translate.
    [[nodiscard]] Index branch_var_id_for_source(
        Index branch_id, const topology::Graph& graph) const {
        const auto it = source_branch_var_id_.find(branch_id);
        if (it == source_branch_var_id_.end()) {
            throw std::out_of_range(
                "DevicePool::branch_var_id_for_source: branch " +
                std::to_string(branch_id) + " is not a VoltageSource");
        }
        return static_cast<Index>(graph.num_nodes()) + it->second;
    }

    [[nodiscard]] Size num_voltage_sources() const noexcept {
        return num_sources_;
    }

    [[nodiscard]] Size state_size(const topology::Graph& graph) const noexcept {
        return static_cast<Size>(graph.num_nodes()) + num_sources_;
    }

private:
    using Entry = std::variant<models::Resistor::Params,
                                models::VoltageSource::Params,
                                SwitchParams>;

    [[nodiscard]] const Entry& entry_at(Index branch_id) const {
        const auto it = entries_.find(branch_id);
        if (it == entries_.end()) {
            throw std::out_of_range(
                "DevicePool: branch_id " + std::to_string(branch_id) +
                " not registered");
        }
        return it->second;
    }

    [[nodiscard]] const SwitchParams& switch_params_at(
        Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<SwitchParams>(entry)) {
            throw std::out_of_range(
                "DevicePool::switch_params: branch " +
                std::to_string(branch_id) + " is not a Switch");
        }
        return std::get<SwitchParams>(entry);
    }

    std::unordered_map<Index, Entry> entries_;
    // Sources also get a relative branch-current-row offset assigned
    // at insertion time. The absolute index is offset by
    // graph.num_nodes() at lookup time.
    std::unordered_map<Index, Index> source_branch_var_id_;
    Size num_sources_ = 0;
};

}  // namespace pulsim::v2::pwl
