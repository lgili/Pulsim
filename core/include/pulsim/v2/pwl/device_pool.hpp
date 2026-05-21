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

#include "pulsim/v2/models/capacitor.hpp"
#include "pulsim/v2/models/ideal_diode.hpp"
#include "pulsim/v2/models/igbt_level1.hpp"
#include "pulsim/v2/models/inductor.hpp"
#include "pulsim/v2/models/mosfet_level1.hpp"
#include "pulsim/v2/models/multi_winding_transformer.hpp"
#include "pulsim/v2/models/saturable_inductor.hpp"
#include "pulsim/v2/models/resistor.hpp"
#include "pulsim/v2/models/current_source.hpp"
#include "pulsim/v2/models/pulse_voltage_source.hpp"
#include "pulsim/v2/models/pwm_voltage_source.hpp"
#include "pulsim/v2/models/sine_voltage_source.hpp"
#include "pulsim/v2/models/switched_diode.hpp"
#include "pulsim/v2/models/transformer.hpp"
#include "pulsim/v2/models/vcvs.hpp"
#include "pulsim/v2/models/voltage_source.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pulsim::v2::pwl {

class DevicePool {
public:
    // The enum values MUST match the order of alternatives in
    // `Entry` so that `Entry::index()` casts directly to
    // `StoredKind`.
    enum class StoredKind {
        Resistor          = 0,
        VoltageSource     = 1,
        Switch            = 2,
        Capacitor         = 3,  // trap companion (Layer 4 V1)
        Inductor          = 4,  // trap companion (Layer 4 V1)
        Diode             = 5,  // SwitchedDiode (Layer 5 V2)
        NonlinearDiode    = 6,  // models::IdealDiode (Newton, Layer 4 V3)
        CurrentSource     = 7,  // Layer 2 V3 (no branch-current unknown)
        PWMVoltageSource  = 8,  // Layer 2 V4 (time-varying voltage source)
        SineVoltageSource = 9,  // Layer 2 V11 (AC voltage source)
        PulseVoltageSource = 10, // Layer 2 V12 (pulse / step source)
        MosfetLevel1      = 11, // Layer 2 V13 (SH1 nonlinear MOSFET)
        IgbtLevel1        = 12, // Layer 2 V14 (linear-conduction IGBT)
        VCVS              = 13, // Layer 2 V15 (voltage-controlled voltage source)
        SaturableInductor = 14, // Layer 2 V17 (nonlinear L(i) inductor)
    };

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

    // Dynamic devices — see models/{capacitor,inductor}.hpp for the
    // trap-companion math. Both register with Layer 4 V1 as
    // `kind == PassiveLinear` branches; assemble dispatches on the
    // pool's `StoredKind`.
    void add_capacitor(Index branch_id,
                        models::Capacitor::Params p) {
        entries_[branch_id] = Entry{p};
    }

    void add_inductor(Index branch_id,
                       models::Inductor::Params p) {
        entries_[branch_id] = Entry{p};
        // Inductors add a branch-current unknown (analogous to
        // voltage sources). The relative offset is "this inductor's
        // position among inductors". The absolute state-vector
        // index = num_nodes + num_sources + relative_offset.
        inductor_branch_var_id_[branch_id] =
            static_cast<Index>(num_inductors_++);
    }

    /// Register a SwitchedDiode. The branch MUST have been added
    /// to the graph with `BranchKind::Switch` (the diode IS a
    /// switch from the topology's perspective). Default V_th = 0
    /// for a perfectly ideal diode; pass 0.7 for a Si-behavioral
    /// model.
    void add_diode(Index branch_id, Real g_on, Real g_off,
                    Real V_th = Real{0}) {
        entries_[branch_id] = Entry{
            models::SwitchedDiode::Params{g_on, g_off, V_th}};
        diode_branches_.push_back(branch_id);
    }

    /// Register a smooth-blend nonlinear diode (the Shockley-
    /// flavoured `models::IdealDiode`). The branch MUST have been
    /// added with `BranchKind::Nonlinear`. Layer 4 V3's Newton
    /// loop stamps it per iteration via `refresh_smooth_diodes`.
    void add_nonlinear_diode(Index branch_id,
                              models::IdealDiode::Params p) {
        entries_[branch_id] = Entry{p};
    }

    /// Register a constant DC current source (Layer 2 V3).
    /// Unlike VoltageSource, CurrentSource does NOT add a
    /// branch-current unknown — the current is fixed at I,
    /// so it only contributes to the b_constant KCL terms.
    /// The branch MUST be added with `BranchKind::Source`
    /// in the graph (same as VoltageSource); assemble.hpp
    /// dispatches on the pool's StoredKind.
    void add_current_source(Index branch_id,
                              models::CurrentSource::Params p) {
        entries_[branch_id] = Entry{p};
    }

    /// Register a PWM voltage source (Layer 2 V4).
    /// Structurally identical to a VoltageSource (uses the
    /// same branch-var bookkeeping); the time-varying value
    /// is overlaid on `b_extra` at runtime by
    /// `run_transient`'s built-in PWM pass.
    /// The branch MUST be added with `BranchKind::Source`
    /// in the graph; assemble.hpp dispatches on the pool's
    /// StoredKind and stamps it like a 0V baseline source.
    void add_pwm_voltage_source(
        Index branch_id,
        models::PWMVoltageSource::Params p) {
        entries_[branch_id] = Entry{p};
        // PWM sources DO add a branch-current unknown
        // (just like VoltageSource), and they live in the
        // same `i_src` row layout.
        source_branch_var_id_[branch_id] =
            static_cast<Index>(num_sources_++);
    }

    /// Register a sinusoidal AC voltage source (Layer 2 V11).
    /// Structurally identical to a VoltageSource — the time-
    /// varying sine value is overlaid via `b_extra` at
    /// runtime by `run_transient`'s built-in AC pass. The
    /// branch MUST be added with `BranchKind::Source` in
    /// the graph; assemble.hpp dispatches on StoredKind and
    /// stamps it as a 0 V baseline source.
    void add_sine_voltage_source(
        Index branch_id,
        models::SineVoltageSource::Params p) {
        entries_[branch_id] = Entry{p};
        source_branch_var_id_[branch_id] =
            static_cast<Index>(num_sources_++);
    }

    /// Register a pulse / step voltage source (Layer 2 V12).
    /// Same architectural pattern as PWM/Sine: V=0 baseline
    /// + b_extra overlay at runtime.
    void add_pulse_voltage_source(
        Index branch_id,
        models::PulseVoltageSource::Params p) {
        entries_[branch_id] = Entry{p};
        source_branch_var_id_[branch_id] =
            static_cast<Index>(num_sources_++);
    }

    /// Register a 3-terminal SH1 MOSFET (Layer 2 V13). The
    /// `branch_id` is the drain→source `BranchKind::Nonlinear`
    /// branch; `gate_node_id` is the gate's node index in the
    /// graph (looked up by `read_node_voltage` during the
    /// Newton refresh). Layer 4 V13's
    /// `refresh_mosfets_level1` stamps the 6 Jacobian entries
    /// per device per Newton iteration.
    void add_mosfet_level1(
        Index branch_id, Index gate_node_id,
        models::MosfetLevel1::Params p) {
        entries_[branch_id] = Entry{p};
        mosfet_level1_gate_node_id_[branch_id] = gate_node_id;
    }

    [[nodiscard]] Index
    mosfet_level1_gate_node(Index branch_id) const {
        const auto it =
            mosfet_level1_gate_node_id_.find(branch_id);
        if (it == mosfet_level1_gate_node_id_.end()) {
            throw std::out_of_range(
                "DevicePool::mosfet_level1_gate_node: "
                "branch " + std::to_string(branch_id) +
                " is not a MosfetLevel1");
        }
        return it->second;
    }

    /// Register a 4-terminal VCVS (Layer 2 V15).
    /// `branch_id` is the output branch (out_pos→out_neg);
    /// `in_pos_node` and `in_neg_node` are the sense-input
    /// node indices. Adds a branch-current unknown like a
    /// regular VoltageSource. Stamped LINEARLY by assemble —
    /// no Newton refresh needed.
    void add_vcvs(
        Index branch_id, Index in_pos_node,
        Index in_neg_node,
        models::VCVS::Params p) {
        entries_[branch_id] = Entry{p};
        source_branch_var_id_[branch_id] =
            static_cast<Index>(num_sources_++);
        vcvs_input_nodes_[branch_id] =
            std::pair<Index, Index>{in_pos_node, in_neg_node};
    }

    [[nodiscard]] std::pair<Index, Index>
    vcvs_input_nodes(Index branch_id) const {
        const auto it = vcvs_input_nodes_.find(branch_id);
        if (it == vcvs_input_nodes_.end()) {
            throw std::out_of_range(
                "DevicePool::vcvs_input_nodes: branch " +
                std::to_string(branch_id) +
                " is not a VCVS");
        }
        return it->second;
    }

    /// Register a SaturableInductor (V17). Like a regular
    /// inductor, it adds a branch-current unknown to the
    /// state vector (numbered in the same `num_inductors_`
    /// sequence). The branch MUST be added with
    /// `BranchKind::Nonlinear` in the graph — assemble skips
    /// it, and `refresh_saturable_inductors` stamps the
    /// nonlinear trap-rule constraint per Newton iteration.
    void add_saturable_inductor(
        Index branch_id,
        models::SaturableInductor::Params p) {
        entries_[branch_id] = Entry{p};
        // Piggy-back on the inductor numbering — saturable
        // and linear inductors share the same state-vector
        // segment.
        inductor_branch_var_id_[branch_id] =
            static_cast<Index>(num_inductors_++);
        saturable_inductor_branches_.push_back(branch_id);
    }

    [[nodiscard]] const std::vector<Index>&
    saturable_inductor_branches() const noexcept {
        return saturable_inductor_branches_;
    }

    /// Register a 3-terminal IGBT Level 1 (Layer 2 V14).
    /// `branch_id` is the collector→emitter Nonlinear branch;
    /// `gate_node_id` is the gate's node index. Layer 4 V14's
    /// `refresh_igbts_level1` stamps the 6 Jacobian entries
    /// per device per Newton iteration.
    void add_igbt_level1(
        Index branch_id, Index gate_node_id,
        models::IgbtLevel1::Params p) {
        entries_[branch_id] = Entry{p};
        igbt_level1_gate_node_id_[branch_id] = gate_node_id;
    }

    [[nodiscard]] Index
    igbt_level1_gate_node(Index branch_id) const {
        const auto it =
            igbt_level1_gate_node_id_.find(branch_id);
        if (it == igbt_level1_gate_node_id_.end()) {
            throw std::out_of_range(
                "DevicePool::igbt_level1_gate_node: "
                "branch " + std::to_string(branch_id) +
                " is not an IgbtLevel1");
        }
        return it->second;
    }

    // -------- Layer 2 V2: transformer (coupled inductors) ------------------
    //
    // Couplings are PAIRS of already-added inductor branches.
    // The cross-term stamping happens in `assemble.hpp` AFTER
    // the per-branch loop completes (so the self-inductance
    // diagonals are already in place); same for the history
    // contribution in `HistoryState::compute_b_extra`.

    struct TransformerCoupling {
        Index primary_branch_id;
        Index secondary_branch_id;
        models::TwoWindingTransformer::Params params;
    };

    void add_transformer_coupling(
        Index primary_branch_id,
        Index secondary_branch_id,
        const models::TwoWindingTransformer::Params& p) {
        transformer_couplings_.push_back(
            TransformerCoupling{primary_branch_id,
                                  secondary_branch_id, p});
    }

    [[nodiscard]] const std::vector<TransformerCoupling>&
    transformer_couplings() const noexcept {
        return transformer_couplings_;
    }

    // -------- Lookups -------------------------------------------------------

    /// Proposal #3.4 ergonomics: scan the pool to see if any
    /// device requires Newton (smooth-blend diode, SH1 MOSFET,
    /// Level 1 IGBT, or saturable inductor). Used by the
    /// Python `simulate()` wrapper to auto-enable the
    /// nonlinear-refresh pass.
    [[nodiscard]] bool has_nonlinear_devices() const noexcept {
        for (const auto& [branch_id, entry] : entries_) {
            const auto k = static_cast<StoredKind>(entry.index());
            if (k == StoredKind::NonlinearDiode ||
                k == StoredKind::MosfetLevel1   ||
                k == StoredKind::IgbtLevel1     ||
                k == StoredKind::SaturableInductor) {
                return true;
            }
        }
        return false;
    }

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

    [[nodiscard]] const models::CurrentSource::Params&
    current_source_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::CurrentSource::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::current_source_params: branch " +
                std::to_string(branch_id) + " is not a CurrentSource");
        }
        return std::get<models::CurrentSource::Params>(entry);
    }

    [[nodiscard]] const models::PWMVoltageSource::Params&
    pwm_voltage_source_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::PWMVoltageSource::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::pwm_voltage_source_params: "
                "branch " + std::to_string(branch_id) +
                " is not a PWMVoltageSource");
        }
        return std::get<models::PWMVoltageSource::Params>(entry);
    }

    [[nodiscard]] const models::SineVoltageSource::Params&
    sine_voltage_source_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::SineVoltageSource::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::sine_voltage_source_params: "
                "branch " + std::to_string(branch_id) +
                " is not a SineVoltageSource");
        }
        return std::get<models::SineVoltageSource::Params>(entry);
    }

    [[nodiscard]] const models::PulseVoltageSource::Params&
    pulse_voltage_source_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::PulseVoltageSource::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::pulse_voltage_source_params: "
                "branch " + std::to_string(branch_id) +
                " is not a PulseVoltageSource");
        }
        return std::get<models::PulseVoltageSource::Params>(entry);
    }

    [[nodiscard]] const models::MosfetLevel1::Params&
    mosfet_level1_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::MosfetLevel1::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::mosfet_level1_params: branch " +
                std::to_string(branch_id) +
                " is not a MosfetLevel1");
        }
        return std::get<models::MosfetLevel1::Params>(entry);
    }

    [[nodiscard]] const models::SaturableInductor::Params&
    saturable_inductor_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::SaturableInductor::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::saturable_inductor_params: "
                "branch " + std::to_string(branch_id) +
                " is not a SaturableInductor");
        }
        return std::get<
            models::SaturableInductor::Params>(entry);
    }

    [[nodiscard]] const models::VCVS::Params&
    vcvs_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::VCVS::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::vcvs_params: branch " +
                std::to_string(branch_id) +
                " is not a VCVS");
        }
        return std::get<models::VCVS::Params>(entry);
    }

    [[nodiscard]] const models::IgbtLevel1::Params&
    igbt_level1_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<
                models::IgbtLevel1::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::igbt_level1_params: branch " +
                std::to_string(branch_id) +
                " is not an IgbtLevel1");
        }
        return std::get<models::IgbtLevel1::Params>(entry);
    }

    [[nodiscard]] Real switch_g_on(Index branch_id) const {
        return switch_params_at(branch_id).g_on;
    }
    [[nodiscard]] Real switch_g_off(Index branch_id) const {
        return switch_params_at(branch_id).g_off;
    }

    [[nodiscard]] const models::Capacitor::Params&
    capacitor_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::Capacitor::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::capacitor_params: branch " +
                std::to_string(branch_id) + " is not a Capacitor");
        }
        return std::get<models::Capacitor::Params>(entry);
    }

    [[nodiscard]] const models::Inductor::Params&
    inductor_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::Inductor::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::inductor_params: branch " +
                std::to_string(branch_id) + " is not an Inductor");
        }
        return std::get<models::Inductor::Params>(entry);
    }

    [[nodiscard]] const models::SwitchedDiode::Params&
    diode_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::SwitchedDiode::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::diode_params: branch " +
                std::to_string(branch_id) + " is not a Diode");
        }
        return std::get<models::SwitchedDiode::Params>(entry);
    }

    [[nodiscard]] const models::IdealDiode::Params&
    nonlinear_diode_params(Index branch_id) const {
        const auto& entry = entry_at(branch_id);
        if (!std::holds_alternative<models::IdealDiode::Params>(entry)) {
            throw std::out_of_range(
                "DevicePool::nonlinear_diode_params: branch " +
                std::to_string(branch_id) +
                " is not a NonlinearDiode");
        }
        return std::get<models::IdealDiode::Params>(entry);
    }

    /// Iterate diode branch ids in insertion (= branch) order.
    [[nodiscard]] std::span<const Index> diode_branches() const noexcept {
        return std::span<const Index>{diode_branches_.data(),
                                       diode_branches_.size()};
    }

    [[nodiscard]] Size num_diodes() const noexcept {
        return diode_branches_.size();
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

    /// Absolute state-vector index for an inductor's branch-current
    /// unknown. Throws if the branch is not registered as an
    /// Inductor.
    ///
    /// Layout (left to right):
    ///   [v_0 .. v_{N-1}]  [i_src_0 .. i_src_{M-1}]  [i_L_0 .. i_L_{K-1}]
    ///                                                ^
    ///                                                this region
    ///
    /// Absolute index = num_nodes + num_voltage_sources + relative_offset.
    [[nodiscard]] Index branch_var_id_for_inductor(
        Index branch_id, const topology::Graph& graph) const {
        const auto it = inductor_branch_var_id_.find(branch_id);
        if (it == inductor_branch_var_id_.end()) {
            throw std::out_of_range(
                "DevicePool::branch_var_id_for_inductor: branch " +
                std::to_string(branch_id) + " is not an Inductor");
        }
        return static_cast<Index>(graph.num_nodes()) +
               static_cast<Index>(num_sources_) + it->second;
    }

    [[nodiscard]] Size num_inductors() const noexcept {
        return num_inductors_;
    }

    /// Total count of dynamic devices (Capacitor + Inductor). Used
    /// by Layer 5's HistoryState to size itself.
    [[nodiscard]] Size num_dynamic_branches() const noexcept {
        Size n = 0;
        for (const auto& [_, entry] : entries_) {
            if (std::holds_alternative<models::Capacitor::Params>(entry) ||
                std::holds_alternative<models::Inductor::Params>(entry)) {
                ++n;
            }
        }
        return n;
    }

    [[nodiscard]] Size state_size(const topology::Graph& graph) const noexcept {
        return static_cast<Size>(graph.num_nodes()) + num_sources_ +
               num_inductors_;
    }

private:
    // The alternative order is locked to `StoredKind`: index 0 is
    // Resistor, 1 is VoltageSource, … — `kind_of` casts the variant
    // index directly to StoredKind, so DO NOT reorder.
    using Entry = std::variant<models::Resistor::Params,
                                models::VoltageSource::Params,
                                SwitchParams,
                                models::Capacitor::Params,
                                models::Inductor::Params,
                                models::SwitchedDiode::Params,
                                models::IdealDiode::Params,
                                models::CurrentSource::Params,
                                models::PWMVoltageSource::Params,
                                models::SineVoltageSource::Params,
                                models::PulseVoltageSource::Params,
                                models::MosfetLevel1::Params,
                                models::IgbtLevel1::Params,
                                models::VCVS::Params,
                                models::SaturableInductor::Params>;

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

    // Inductors live AFTER sources in the state vector:
    //   [v_0 .. v_{N-1}]  [i_src_0 .. i_src_{M-1}]  [i_L_0 .. i_L_{K-1}]
    // The absolute index = num_nodes + num_sources + relative_offset.
    std::unordered_map<Index, Index> inductor_branch_var_id_;
    Size num_inductors_ = 0;

    // Diode branches in insertion order. Layer 5 V2's
    // DiodeEventState iterates this list to build per-diode
    // tracking entries.
    std::vector<Index> diode_branches_;

    // Layer 2 V13: MOSFET Level 1 — extra storage for the
    // gate node id per drain→source branch. The branch
    // itself lives in `entries_` as a `MosfetLevel1::Params`
    // entry; the gate node is a separate associative lookup.
    std::unordered_map<Index, Index> mosfet_level1_gate_node_id_;

    // Layer 2 V14: IGBT Level 1 — same gate-node lookup
    // pattern as MOSFET. The branch is collector→emitter.
    std::unordered_map<Index, Index> igbt_level1_gate_node_id_;

    // Layer 2 V15: VCVS — per-branch (in_pos, in_neg) node
    // references for the sense inputs.
    std::unordered_map<Index, std::pair<Index, Index>>
        vcvs_input_nodes_;

    // Layer 2 V17: SaturableInductor branches, in insertion
    // order. Used by the Newton refresh + history tracker to
    // iterate just the saturable inductors without scanning
    // every Nonlinear branch.
    std::vector<Index> saturable_inductor_branches_;

    // Layer 2 V2: transformer coupling registry. Each entry
    // pairs two already-added inductor branches with the
    // coupling parameters (L_p, L_s, k → M).
    std::vector<TransformerCoupling> transformer_couplings_;
};

}  // namespace pulsim::v2::pwl
