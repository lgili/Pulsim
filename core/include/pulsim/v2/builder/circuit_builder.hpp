#pragma once

// =============================================================================
// Pulsim v2 — Layer 6 V0: CircuitBuilder (high-level circuit construction)
// =============================================================================
//
// `pulsim-v2-builder-api` Phase 1.
//
// User-facing wrapper that hides the two-object `Graph +
// DevicePool` split. The builder accepts string node names,
// SI-unit parameter values, and named devices; internally it
// auto-tracks indices, dispatches to the kernel's `add_branch`
// + `pool.add_*` methods, and exposes `graph()` / `pool()`
// const-refs for `PwlStateSpaceCache` and `run_transient`.
//
// Design constraints:
//   * Pure additive wrapper. NO numerical work.
//   * Header-only (consistent with the rest of v2).
//   * No exceptions on duplicate device names (V0 — V1 may
//     add validation).
//   * "gnd" / "GND" / "0" all map to `Graph::ground()`.
//   * Case-sensitive for non-ground names.

#include "pulsim/v2/models/capacitor.hpp"
#include "pulsim/v2/models/ideal_diode.hpp"
#include "pulsim/v2/models/inductor.hpp"
#include "pulsim/v2/models/resistor.hpp"
#include "pulsim/v2/models/voltage_source.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pulsim::v2::builder {

class CircuitBuilder {
public:
    CircuitBuilder() = default;

    /// Return the node index for `name`, creating it if not
    /// yet registered. `"gnd"` / `"GND"` / `"0"` all map to
    /// `graph().ground()` without consuming a node slot.
    Index node(std::string name) {
        return resolve_node_(name);
    }

    // -------- Add methods --------------------------------------------------

    CircuitBuilder& add_voltage_source(
        std::string /*name*/, std::string from,
        std::string to, Real V) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_voltage_source(
            b_id, models::VoltageSource::Params{V});
        return *this;
    }

    CircuitBuilder& add_resistor(
        std::string /*name*/, std::string from,
        std::string to, Real R_ohms) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_resistor(
            b_id, models::Resistor::Params{
                .G = Real{1} / R_ohms});
        return *this;
    }

    CircuitBuilder& add_capacitor(
        std::string /*name*/, std::string from,
        std::string to, Real C_farads) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_capacitor(
            b_id, models::Capacitor::Params{C_farads});
        return *this;
    }

    CircuitBuilder& add_inductor(
        std::string /*name*/, std::string from,
        std::string to, Real L_henries) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_inductor(
            b_id, models::Inductor::Params{L_henries});
        return *this;
    }

    /// Add a binary switched diode (Layer 5 V2's
    /// `SwitchedDiode`). The branch uses
    /// `BranchKind::Switch` (the diode behaves as a switch
    /// from the topology's perspective).
    CircuitBuilder& add_diode(
        std::string /*name*/, std::string anode,
        std::string cathode, Real g_on, Real g_off,
        Real V_th = Real{0}) {
        const Index a_idx = resolve_node_(anode);
        const Index k_idx = resolve_node_(cathode);
        const Index b_id = graph_.add_branch(
            a_idx, k_idx,
            topology::BranchKind::Switch);
        pool_.add_diode(b_id, g_on, g_off, V_th);
        return *this;
    }

    /// Add a smooth-blend `IdealDiode` (Layer 4 V3's
    /// AD-driven nonlinear model). The branch uses
    /// `BranchKind::Nonlinear`.
    CircuitBuilder& add_nonlinear_diode(
        std::string /*name*/, std::string anode,
        std::string cathode,
        models::IdealDiode::Params params) {
        const Index a_idx = resolve_node_(anode);
        const Index k_idx = resolve_node_(cathode);
        const Index b_id = graph_.add_branch(
            a_idx, k_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_nonlinear_diode(b_id, params);
        return *this;
    }

    /// Add a controlled switch (drives by `switch_fn`
    /// at simulation time).
    CircuitBuilder& add_switch(
        std::string /*name*/, std::string from,
        std::string to, Real g_on, Real g_off) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Switch);
        pool_.add_switch(b_id, g_on, g_off);
        return *this;
    }

    // -------- Accessors ----------------------------------------------------

    [[nodiscard]] const topology::Graph& graph() const noexcept {
        return graph_;
    }

    [[nodiscard]] const pwl::DevicePool& pool() const noexcept {
        return pool_;
    }

    [[nodiscard]] Size num_branches() const noexcept {
        return graph_.num_branches();
    }

    /// Look up a previously-registered node by name. Throws
    /// `std::out_of_range` if not found. The "gnd" alias is
    /// handled here too.
    [[nodiscard]] Index node_id_of(
        const std::string& name) const {
        if (is_ground_alias_(name)) {
            return graph_.ground();
        }
        const auto it = node_map_.find(name);
        if (it == node_map_.end()) {
            throw std::out_of_range(
                "CircuitBuilder::node_id_of: node \"" +
                name + "\" was never registered");
        }
        return it->second;
    }

private:
    [[nodiscard]] static bool is_ground_alias_(
        const std::string& name) noexcept {
        return name == "gnd" || name == "GND" ||
               name == "0";
    }

    Index resolve_node_(const std::string& name) {
        if (is_ground_alias_(name)) {
            return graph_.ground();
        }
        const auto it = node_map_.find(name);
        if (it != node_map_.end()) {
            return it->second;
        }
        const Index idx = graph_.add_node(name);
        node_map_.emplace(name, idx);
        return idx;
    }

    topology::Graph                         graph_;
    pwl::DevicePool                         pool_;
    std::unordered_map<std::string, Index>  node_map_;
};

}  // namespace pulsim::v2::builder
