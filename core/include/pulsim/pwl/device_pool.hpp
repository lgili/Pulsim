#pragma once

// =============================================================================
// Pulsim — Layer 4: DevicePool (branch_id → params registry)
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

#include "pulsim/models/capacitor.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/models/igbt_level1.hpp"
#include "pulsim/models/inductor.hpp"
#include "pulsim/models/mosfet_level1.hpp"
#include "pulsim/models/multi_winding_transformer.hpp"
#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/models/resistor.hpp"
#include "pulsim/models/current_source.hpp"
#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/models/transformer.hpp"
#include "pulsim/models/vcvs.hpp"
#include "pulsim/models/voltage_source.hpp"
#include "pulsim/numeric/dictionary.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace pulsim::pwl {

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
            throw std::out_of_range(std::format(
                "DevicePool::mosfet_level1_gate_node: "
                "branch {} is not a MosfetLevel1",
                branch_id));
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
            throw std::out_of_range(std::format(
                "DevicePool::vcvs_input_nodes: branch {} is not a VCVS",
                branch_id));
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
            throw std::out_of_range(std::format(
                "DevicePool::igbt_level1_gate_node: "
                "branch {} is not an IgbtLevel1",
                branch_id));
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

    /// Non-throwing companion to `kind_of`: is there a device
    /// registered on this branch at all? Diagnostic passes (the
    /// preflight sweep, error paths) need to ask without risking a
    /// second exception on top of the one they are explaining.
    [[nodiscard]] bool is_registered(Index branch_id) const noexcept {
        return entries_.find(branch_id) != entries_.end();
    }

    [[nodiscard]] StoredKind kind_of(Index branch_id) const {
        const auto it = entries_.find(branch_id);
        if (it == entries_.end()) {
            throw std::out_of_range(std::format(
                "DevicePool::kind_of: branch_id {} not registered",
                branch_id));
        }
        return static_cast<StoredKind>(it->second.index());
    }

    [[nodiscard]] const models::Resistor::Params&
    resistor_params(Index branch_id) const {
        return require_params_<models::Resistor::Params>(
            branch_id, "resistor_params", "a Resistor");
    }

    [[nodiscard]] const models::VoltageSource::Params&
    voltage_source_params(Index branch_id) const {
        return require_params_<models::VoltageSource::Params>(
            branch_id, "voltage_source_params", "a VoltageSource");
    }

    [[nodiscard]] const models::CurrentSource::Params&
    current_source_params(Index branch_id) const {
        return require_params_<models::CurrentSource::Params>(
            branch_id, "current_source_params", "a CurrentSource");
    }

    [[nodiscard]] const models::PWMVoltageSource::Params&
    pwm_voltage_source_params(Index branch_id) const {
        return require_params_<models::PWMVoltageSource::Params>(
            branch_id, "pwm_voltage_source_params",
            "a PWMVoltageSource");
    }

    [[nodiscard]] const models::SineVoltageSource::Params&
    sine_voltage_source_params(Index branch_id) const {
        return require_params_<models::SineVoltageSource::Params>(
            branch_id, "sine_voltage_source_params",
            "a SineVoltageSource");
    }

    [[nodiscard]] const models::PulseVoltageSource::Params&
    pulse_voltage_source_params(Index branch_id) const {
        return require_params_<models::PulseVoltageSource::Params>(
            branch_id, "pulse_voltage_source_params",
            "a PulseVoltageSource");
    }

    [[nodiscard]] const models::MosfetLevel1::Params&
    mosfet_level1_params(Index branch_id) const {
        return require_params_<models::MosfetLevel1::Params>(
            branch_id, "mosfet_level1_params", "a MosfetLevel1");
    }

    [[nodiscard]] const models::SaturableInductor::Params&
    saturable_inductor_params(Index branch_id) const {
        return require_params_<models::SaturableInductor::Params>(
            branch_id, "saturable_inductor_params",
            "a SaturableInductor");
    }

    [[nodiscard]] const models::VCVS::Params&
    vcvs_params(Index branch_id) const {
        return require_params_<models::VCVS::Params>(
            branch_id, "vcvs_params", "a VCVS");
    }

    [[nodiscard]] const models::IgbtLevel1::Params&
    igbt_level1_params(Index branch_id) const {
        return require_params_<models::IgbtLevel1::Params>(
            branch_id, "igbt_level1_params", "an IgbtLevel1");
    }

    [[nodiscard]] Real switch_g_on(Index branch_id) const {
        return switch_params_at(branch_id).g_on;
    }
    [[nodiscard]] Real switch_g_off(Index branch_id) const {
        return switch_params_at(branch_id).g_off;
    }

    [[nodiscard]] const models::Capacitor::Params&
    capacitor_params(Index branch_id) const {
        return require_params_<models::Capacitor::Params>(
            branch_id, "capacitor_params", "a Capacitor");
    }

    [[nodiscard]] const models::Inductor::Params&
    inductor_params(Index branch_id) const {
        return require_params_<models::Inductor::Params>(
            branch_id, "inductor_params", "an Inductor");
    }

    [[nodiscard]] const models::SwitchedDiode::Params&
    diode_params(Index branch_id) const {
        return require_params_<models::SwitchedDiode::Params>(
            branch_id, "diode_params", "a Diode");
    }

    [[nodiscard]] const models::IdealDiode::Params&
    nonlinear_diode_params(Index branch_id) const {
        return require_params_<models::IdealDiode::Params>(
            branch_id, "nonlinear_diode_params", "a NonlinearDiode");
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
            throw std::out_of_range(std::format(
                "DevicePool::branch_var_id_for_source: "
                "branch {} is not a VoltageSource",
                branch_id));
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
            throw std::out_of_range(std::format(
                "DevicePool::branch_var_id_for_inductor: "
                "branch {} is not an Inductor",
                branch_id));
        }
        return static_cast<Index>(graph.num_nodes()) +
               static_cast<Index>(num_sources_) + it->second;
    }

    [[nodiscard]] Size num_inductors() const noexcept {
        return num_inductors_;
    }

    /// Cheap predicate set used by builder-level helpers
    /// (`CircuitBuilder::initial_state`) to dispatch IC placement
    /// without throwing. Each is O(1) hash lookup.
    [[nodiscard]] bool is_inductor(Index branch_id) const noexcept {
        return inductor_branch_var_id_.contains(branch_id);
    }
    [[nodiscard]] bool is_voltage_source(Index branch_id) const noexcept {
        return source_branch_var_id_.contains(branch_id);
    }
    [[nodiscard]] bool is_capacitor(Index branch_id) const noexcept {
        const auto it = entries_.find(branch_id);
        if (it == entries_.end()) return false;
        return std::visit(
            []<typename T>(const T&) noexcept -> bool {
                return std::is_same_v<std::decay_t<T>,
                                       models::Capacitor::Params>;
            },
            it->second);
    }

    /// Total count of dynamic devices (Capacitor + Inductor). Used
    /// by Layer 5's HistoryState to size itself.
    ///
    /// Implemented via `std::visit` so adding a new dynamic-device
    /// alternative to `Entry` causes a compile error here unless
    /// it's explicitly classified (compiler reminds you to update
    /// the visitor — exhaustiveness check on a closed variant).
    [[nodiscard]] Size num_dynamic_branches() const noexcept {
        Size n = 0;
        for (const auto& [_, entry] : entries_) {
            n += std::visit(
                []<typename T>(const T&) noexcept -> Size {
                    using Pure = std::decay_t<T>;
                    return std::is_same_v<Pure,
                               models::Capacitor::Params> ||
                           std::is_same_v<Pure,
                               models::Inductor::Params>
                        ? Size{1} : Size{0};
                },
                entry);
        }
        return n;
    }

    [[nodiscard]] Size state_size(const topology::Graph& graph) const noexcept {
        return static_cast<Size>(graph.num_nodes()) + num_sources_ +
               num_inductors_;
    }

    // -------- Switch / parameter affected-column helpers (v1.4.0) -----------
    //
    // `openspec/changes/add-generalised-path-refactor` Part A — the
    // multi-bit path-union routing in `PwlStateSpaceCache::solve_rank1`
    // queries these helpers to assemble the union of MNA columns
    // affected by a set of toggled switches.
    //
    // A switch between nodes (from, to) stamps its conductance at the
    // diagonal entries (from, from) + (to, to) and the off-diagonal
    // entries (from, to) + (to, from). In MNA column space, this means
    // toggling that switch changes the values of columns `from` AND `to`
    // (the off-diagonal entries live IN those columns). Ground-anchored
    // endpoints (node < 0) are skipped.

    /// Columns in J that change when the `sw_idx`-th switch toggles.
    /// `sw_idx` is the switch's index (0..num_switches-1), not its
    /// branch_id; this matches the bit ordering of `SwitchStateMask`.
    /// Throws `std::out_of_range` if `sw_idx >= graph.num_switches()`.
    [[nodiscard]] std::vector<Index> columns_affected_by_switch(
        Size sw_idx, const topology::Graph& graph) const {
        Size cur = 0;
        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind != topology::BranchKind::Switch) continue;
            if (cur == sw_idx) {
                std::vector<Index> cols;
                cols.reserve(2);
                if (branch.from >= 0) cols.push_back(branch.from);
                if (branch.to   >= 0) cols.push_back(branch.to);
                return cols;
            }
            ++cur;
        }
        throw std::out_of_range(std::format(
            "DevicePool::columns_affected_by_switch: sw_idx {} "
            ">= num_switches ({})",
            sw_idx, graph.num_switches()));
    }

    /// Columns in J that change when `branch_id`'s parameter values
    /// change (with topology fixed). Used by
    /// `PwlStateSpaceCache::refactor_parametric` to identify the
    /// minimum set of columns to re-stamp + path-refactor after a
    /// user-driven parameter sweep / Monte Carlo update.
    ///
    /// The mapping follows the MNA stamping rules:
    ///   * Resistor/Switch/Capacitor (admittance-stamped between two
    ///     nodes): both endpoint columns (skipping ground).
    ///   * Inductor (companion-stamped with branch-current row): the
    ///     branch-current column only — the inductor's L value enters
    ///     the constraint row as `-L·(2/dt)·i_L` at the inductor's
    ///     own branch-current column. The endpoint v rows are
    ///     unaffected when L changes (only the branch-current column
    ///     entry shifts).
    ///   * VoltageSource: the source-current column (one column). The
    ///     value V affects ONLY the RHS `b_constant`, not J — so the
    ///     returned column set is empty for sources. Callers updating
    ///     a source value should re-stamp `b_constant` separately.
    ///   * Other (CurrentSource, Diode, MOSFET, etc.): currently not
    ///     supported in the parametric-refactor pipeline → empty set
    ///     (caller falls back to full re-build of the cache).
    ///
    /// Returns an empty vector for any branch whose parameter change
    /// is NOT expressible as a path-based update over the existing
    /// sparsity pattern. The caller (refactor_parametric) treats an
    /// empty return as "force fallback to full re-build".
    [[nodiscard]] std::vector<Index> columns_affected_by_branch(
        Index branch_id, const topology::Graph& graph) const {
        const auto it = entries_.find(branch_id);
        if (it == entries_.end()) {
            return {};
        }
        const auto& branch = graph.branch(branch_id);
        const StoredKind sk = static_cast<StoredKind>(it->second.index());
        std::vector<Index> cols;
        switch (sk) {
            case StoredKind::Resistor:
            case StoredKind::Switch:
            case StoredKind::Capacitor:
                if (branch.from >= 0) cols.push_back(branch.from);
                if (branch.to   >= 0) cols.push_back(branch.to);
                break;
            case StoredKind::Inductor: {
                // Inductor's L value enters the branch-current
                // constraint row only. The col we touch is the
                // inductor's branch-current state slot.
                cols.push_back(branch_var_id_for_inductor(branch_id,
                                                            graph));
                break;
            }
            case StoredKind::VoltageSource:
                // Pure-RHS update — no J entries depend on V. Caller
                // gets an empty set, signalling "stamp b_constant
                // only, no refactor needed". A future revision may
                // expose this distinction via a separate method; for
                // now `empty cols ⇒ no path-refactor work to do on J`.
                break;
            default:
                // Other devices (CurrentSource, Diode, MOSFET, etc.)
                // are out of scope for the v1.4.0 parametric pipeline.
                // Returning empty triggers full-rebuild fallback.
                break;
        }
        return cols;
    }

    // -------- Parameter mutators (v1.4.0 — parametric refactor) -------------
    //
    // Modify a stored parameter in place. Pre-conditions:
    //   * `branch_id` is registered with the expected device kind
    //     (throws std::out_of_range otherwise).
    //   * The topology is unchanged (these mutators do NOT add or
    //     remove devices). Use with `PwlStateSpaceCache::refactor_parametric`.
    //
    // Storage is std::variant inside Entry; we write the new value
    // back via std::get_if + reassignment.

    /// Update a resistor's resistance value (ohms). Matches the
    /// builder convention `add_resistor(name, from, to, R_ohms)` —
    /// the pool stores conductance G = 1/R, so we convert here.
    void update_resistor_R(Index branch_id, Real new_R_ohms) {
        auto& entry = require_entry_(branch_id, "update_resistor_R");
        if (auto* p = std::get_if<models::Resistor::Params>(&entry)) {
            p->G = Real{1} / new_R_ohms;
        } else {
            throw std::out_of_range(std::format(
                "DevicePool::update_resistor_R: branch {} "
                "is not a Resistor", branch_id));
        }
    }

    void update_inductor_L(Index branch_id, Real new_L) {
        auto& entry = require_entry_(branch_id, "update_inductor_L");
        if (auto* p = std::get_if<models::Inductor::Params>(&entry)) {
            p->L = new_L;
        } else {
            throw std::out_of_range(std::format(
                "DevicePool::update_inductor_L: branch {} "
                "is not an Inductor", branch_id));
        }
    }

    void update_capacitor_C(Index branch_id, Real new_C) {
        auto& entry = require_entry_(branch_id, "update_capacitor_C");
        if (auto* p = std::get_if<models::Capacitor::Params>(&entry)) {
            p->C = new_C;
        } else {
            throw std::out_of_range(std::format(
                "DevicePool::update_capacitor_C: branch {} "
                "is not a Capacitor", branch_id));
        }
    }

    void update_voltage_source_V(Index branch_id, Real new_V) {
        auto& entry = require_entry_(branch_id, "update_voltage_source_V");
        if (auto* p = std::get_if<models::VoltageSource::Params>(&entry)) {
            p->V = new_V;
        } else {
            throw std::out_of_range(std::format(
                "DevicePool::update_voltage_source_V: branch {} "
                "is not a VoltageSource", branch_id));
        }
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

    // ---- Compile-time source-of-truth guard (audit 2026-05, #17) ----------
    // `kind_of()` / `has_nonlinear_devices()` cast the variant index straight
    // to `StoredKind`, so the enum and the `Entry` alternative list MUST stay
    // in lock-step. These asserts turn a silent desync — which would mislabel
    // every device and corrupt dispatch — into a build error: adding or
    // reordering a device fails to compile until BOTH lists agree.
    template <StoredKind K, typename P>
    static constexpr bool kind_maps_to_ = std::is_same_v<
        std::variant_alternative_t<static_cast<std::size_t>(K), Entry>, P>;

    static_assert(std::variant_size_v<Entry> == 15,
        "StoredKind has 15 values; Entry must list 15 alternatives in the "
        "same order — update both together.");
    static_assert(kind_maps_to_<StoredKind::Resistor,           models::Resistor::Params>);
    static_assert(kind_maps_to_<StoredKind::VoltageSource,      models::VoltageSource::Params>);
    static_assert(kind_maps_to_<StoredKind::Switch,             SwitchParams>);
    static_assert(kind_maps_to_<StoredKind::Capacitor,          models::Capacitor::Params>);
    static_assert(kind_maps_to_<StoredKind::Inductor,           models::Inductor::Params>);
    static_assert(kind_maps_to_<StoredKind::Diode,              models::SwitchedDiode::Params>);
    static_assert(kind_maps_to_<StoredKind::NonlinearDiode,     models::IdealDiode::Params>);
    static_assert(kind_maps_to_<StoredKind::CurrentSource,      models::CurrentSource::Params>);
    static_assert(kind_maps_to_<StoredKind::PWMVoltageSource,   models::PWMVoltageSource::Params>);
    static_assert(kind_maps_to_<StoredKind::SineVoltageSource,  models::SineVoltageSource::Params>);
    static_assert(kind_maps_to_<StoredKind::PulseVoltageSource, models::PulseVoltageSource::Params>);
    static_assert(kind_maps_to_<StoredKind::MosfetLevel1,       models::MosfetLevel1::Params>);
    static_assert(kind_maps_to_<StoredKind::IgbtLevel1,         models::IgbtLevel1::Params>);
    static_assert(kind_maps_to_<StoredKind::VCVS,               models::VCVS::Params>);
    static_assert(kind_maps_to_<StoredKind::SaturableInductor,  models::SaturableInductor::Params>);

    [[nodiscard]] const Entry& entry_at(Index branch_id) const {
        const auto it = entries_.find(branch_id);
        if (it == entries_.end()) {
            throw std::out_of_range(std::format(
                "DevicePool: branch_id {} not registered",
                branch_id));
        }
        return it->second;
    }

    /// Non-const counterpart used by the `update_*` mutators.
    /// `method_name` surfaces in the std::out_of_range message so
    /// debug output names the offending API path.
    [[nodiscard]] Entry& require_entry_(
        Index branch_id, std::string_view method_name) {
        auto it = entries_.find(branch_id);
        if (it == entries_.end()) {
            throw std::out_of_range(std::format(
                "DevicePool::{}: branch_id {} not registered",
                method_name, branch_id));
        }
        return it->second;
    }

    /// Single-shot `get_if`-based accessor. Avoids the double
    /// type-id dispatch of `holds_alternative + std::get` (which
    /// the compiler can usually fold but isn't required to), and
    /// collapses the per-device accessor body to a one-liner.
    /// `kind_label` is the human-readable device kind ("a Resistor",
    /// "an Inductor", …) — the leading article is part of the label
    /// so we get grammar right in error messages.
    template <typename T>
    [[nodiscard]] const T& require_params_(
        Index branch_id,
        std::string_view method_name,
        std::string_view kind_label) const {
        const auto& entry = entry_at(branch_id);
        if (const auto* p = std::get_if<T>(&entry)) {
            return *p;
        }
        throw std::out_of_range(std::format(
            "DevicePool::{}: branch {} is not {}",
            method_name, branch_id, kind_label));
    }

    [[nodiscard]] const SwitchParams& switch_params_at(
        Index branch_id) const {
        return require_params_<SwitchParams>(
            branch_id, "switch_params", "a Switch");
    }

    numeric::Dictionary<Index, Entry> entries_;
    // Sources also get a relative branch-current-row offset assigned
    // at insertion time. The absolute index is offset by
    // graph.num_nodes() at lookup time.
    numeric::Dictionary<Index, Index> source_branch_var_id_;
    Size num_sources_ = 0;

    // Inductors live AFTER sources in the state vector:
    //   [v_0 .. v_{N-1}]  [i_src_0 .. i_src_{M-1}]  [i_L_0 .. i_L_{K-1}]
    // The absolute index = num_nodes + num_sources + relative_offset.
    numeric::Dictionary<Index, Index> inductor_branch_var_id_;
    Size num_inductors_ = 0;

    // Diode branches in insertion order. Layer 5 V2's
    // DiodeEventState iterates this list to build per-diode
    // tracking entries.
    std::vector<Index> diode_branches_;

    // Layer 2 V13: MOSFET Level 1 — extra storage for the
    // gate node id per drain→source branch. The branch
    // itself lives in `entries_` as a `MosfetLevel1::Params`
    // entry; the gate node is a separate associative lookup.
    numeric::Dictionary<Index, Index> mosfet_level1_gate_node_id_;

    // Layer 2 V14: IGBT Level 1 — same gate-node lookup
    // pattern as MOSFET. The branch is collector→emitter.
    numeric::Dictionary<Index, Index> igbt_level1_gate_node_id_;

    // Layer 2 V15: VCVS — per-branch (in_pos, in_neg) node
    // references for the sense inputs.
    numeric::Dictionary<Index, std::pair<Index, Index>>
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

}  // namespace pulsim::pwl
