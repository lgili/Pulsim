#pragma once

// =============================================================================
// PulsimCore - Runtime Circuit Builder for Python Bindings
// =============================================================================
// This provides a runtime (non-template) circuit builder that can be used
// from Python. It uses std::variant to store devices and provides dynamic
// matrix assembly for simulation.
// =============================================================================

#include "pulsim/v1/device_base.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/sources.hpp"
#include "pulsim/v1/integration.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <variant>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <stdexcept>

namespace pulsim::v1 {

struct TransparentStringHash {
    using is_transparent = void;

    [[nodiscard]] std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }
};

struct TransparentStringEqual {
    using is_transparent = void;

    [[nodiscard]] bool operator()(std::string_view lhs, std::string_view rhs) const noexcept {
        return lhs == rhs;
    }
};

// =============================================================================
// Device Variant Type
// =============================================================================

/// Variant holding all supported device types
using DeviceVariant = std::variant<
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    IdealDiode,
    IdealSwitch,
    VoltageControlledSwitch,
    MOSFET,
    IGBT,
    PWMVoltageSource,
    SineVoltageSource,
    PulseVoltageSource,
    Transformer
>;

// =============================================================================
// Device Connection Info
// =============================================================================

struct DeviceConnection {
    std::string name;
    std::vector<Index> nodes;  // Node indices for this device
    Index branch_index = -1;   // For devices with branch currents (VS, L)
    Index branch_index_2 = -1; // For devices with two branch currents (Transformer)
};

/// Non-electrical components tracked by the mixed-domain runtime graph.
/// These nodes do not stamp MNA matrices directly.
struct VirtualComponent {
    std::string type;  // Canonical type id (e.g. "voltage_probe")
    std::string name;
    std::vector<Index> nodes;
    std::unordered_map<std::string, Real> numeric_params;
    std::unordered_map<std::string, std::string> metadata;
};

/// Deterministic mixed-domain scheduler output for one accepted timestep.
struct MixedDomainStepResult {
    std::vector<std::string> phase_order;
    std::unordered_map<std::string, Real> channel_values;
};

/// Metadata describing a mixed-domain output channel.
struct VirtualChannelMetadata {
    std::string component_type;
    std::string component_name;
    std::string domain;
    std::vector<Index> nodes;
};

// =============================================================================
// Runtime Circuit Class
// =============================================================================

class Circuit {
public:
    Circuit() = default;

    // =========================================================================
    // Node Management
    // =========================================================================

    /// Add a named node and return its index.
    Index add_node(std::string_view name) {
        if (is_ground_name(name)) {
            return ground_node;
        }
        if (const auto it = node_map_.find(name); it != node_map_.end()) {
            return it->second;
        }

        if (num_branches_ > 0) {
            for (auto& conn : connections_) {
                if (conn.branch_index >= 0) {
                    conn.branch_index += 1;
                }
                if (conn.branch_index_2 >= 0) {
                    conn.branch_index_2 += 1;
                }
            }

            for (std::size_t i = 0; i < devices_.size(); ++i) {
                auto& conn = connections_[i];
                std::visit([&](auto& dev) {
                    using T = std::decay_t<decltype(dev)>;
                    if constexpr (std::is_same_v<T, VoltageSource> ||
                                  std::is_same_v<T, PWMVoltageSource> ||
                                  std::is_same_v<T, SineVoltageSource> ||
                                  std::is_same_v<T, PulseVoltageSource>) {
                        dev.set_branch_index(conn.branch_index);
                    } else if constexpr (std::is_same_v<T, Transformer>) {
                        dev.set_branch_indices(conn.branch_index, conn.branch_index_2);
                    }
                }, devices_[i]);
            }
        }

        const Index idx = static_cast<Index>(node_names_.size());
        node_names_.emplace_back(name);
        node_map_.emplace(node_names_.back(), idx);
        return idx;
    }

    /// Get node index by name (-1 for ground)
    [[nodiscard]] Index get_node(std::string_view name) const {
        if (is_ground_name(name)) {
            return ground_node;
        }
        if (const auto it = node_map_.find(name); it != node_map_.end()) {
            return it->second;
        }
        throw std::runtime_error("Node not found: " + std::string{name});
    }

    /// Get ground node index
    [[nodiscard]] static constexpr Index ground() { return ground_node; }

    /// Number of non-ground nodes
    [[nodiscard]] Index num_nodes() const {
        return static_cast<Index>(node_names_.size());
    }

    /// Number of branch currents (for VS, inductors)
    [[nodiscard]] Index num_branches() const { return num_branches_; }

    /// Total system size (nodes + branches)
    [[nodiscard]] Index system_size() const { return num_nodes() + num_branches(); }

    /// Get node name by index
    [[nodiscard]] const std::string& node_name(Index idx) const {
        if (idx < 0) return ground_name_;
        return node_names_.at(static_cast<std::size_t>(idx));
    }

    /// Get all node names
    [[nodiscard]] const std::vector<std::string>& node_names() const {
        return node_names_;
    }

    /// Get signal names in state-vector order (V(node...), I(device...))
    [[nodiscard]] std::vector<std::string> signal_names() const {
        std::vector<std::string> names;
        names.reserve(num_nodes() + num_branches_);

        for (const auto& name : node_names_) {
            names.push_back("V(" + name + ")");
        }

        std::vector<std::string> branch_names(static_cast<std::size_t>(num_branches_));
        auto add_branch_name = [&](Index branch_index, const std::string& dev_name, const char* suffix) {
            if (branch_index < 0) return;
            Index local = branch_index - num_nodes();
            if (local < 0 || local >= num_branches_) return;
            std::string label = std::string("I(") + dev_name + suffix + ")";
            branch_names[static_cast<std::size_t>(local)] = std::move(label);
        };

        for (const auto& conn : connections_) {
            add_branch_name(conn.branch_index, conn.name, "");
            add_branch_name(conn.branch_index_2, conn.name, "_2");
        }

        for (Index i = 0; i < num_branches_; ++i) {
            auto& name = branch_names[static_cast<std::size_t>(i)];
            if (name.empty()) {
                name = "I(branch" + std::to_string(i) + ")";
            }
            names.push_back(name);
        }

        return names;
    }

    /// Build a simple initial state vector from device initial conditions.
    [[nodiscard]] Vector initial_state() const {
        Vector x = Vector::Zero(system_size());
        std::vector<bool> node_set(static_cast<std::size_t>(num_nodes()), false);

        auto set_node = [&](Index node, Real value) {
            if (node >= 0 && node < num_nodes()) {
                x[node] = value;
                node_set[static_cast<std::size_t>(node)] = true;
            }
        };

        // Initialize node voltages from sources (when referenced to ground).
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;

                if constexpr (std::is_same_v<T, VoltageSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    if (npos >= 0 && nneg < 0) {
                        set_node(npos, dev.voltage());
                    } else if (nneg >= 0 && npos < 0) {
                        set_node(nneg, -dev.voltage());
                    }
                } else if constexpr (std::is_same_v<T, PWMVoltageSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    Real v = dev.voltage_at(0.0);
                    if (npos >= 0 && nneg < 0) {
                        set_node(npos, v);
                    } else if (nneg >= 0 && npos < 0) {
                        set_node(nneg, -v);
                    }
                } else if constexpr (std::is_same_v<T, SineVoltageSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    Real v = dev.params().offset;
                    if (npos >= 0 && nneg < 0) {
                        set_node(npos, v);
                    } else if (nneg >= 0 && npos < 0) {
                        set_node(nneg, -v);
                    }
                } else if constexpr (std::is_same_v<T, PulseVoltageSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    Real v = dev.params().v_initial;
                    if (npos >= 0 && nneg < 0) {
                        set_node(npos, v);
                    } else if (nneg >= 0 && npos < 0) {
                        set_node(nneg, -v);
                    }
                }
            }, devices_[i]);
        }

        // Apply capacitor initial voltages. When a node is already
        // pinned by a voltage source, do NOT overwrite it with the
        // capacitor IC — voltage sources dominate (the cap is in
        // parallel with a stiff voltage rail; its IC is dependent,
        // not independent). This avoids initial_state() returning a
        // state that contradicts the source-driven nodes when the
        // cap shares a node with a voltage source — bug surfaced
        // by `level1_components/test_basic_components.py::TestCapacitor`,
        // where `V1 || C1(ic=0)` previously seeded V(in)=0 instead
        // of V(in)=V_source, producing a 2× source-voltage transient.
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Capacitor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Real v = dev.voltage_prev();

                    if (n1 >= 0 && n2 < 0) {
                        if (!node_set[static_cast<std::size_t>(n1)]) {
                            set_node(n1, v);
                        }
                    } else if (n2 >= 0 && n1 < 0) {
                        if (!node_set[static_cast<std::size_t>(n2)]) {
                            set_node(n2, -v);
                        }
                    } else if (n1 >= 0 && n2 >= 0) {
                        bool n1_set = node_set[static_cast<std::size_t>(n1)];
                        bool n2_set = node_set[static_cast<std::size_t>(n2)];
                        if (n1_set && !n2_set) {
                            set_node(n2, x[n1] - v);
                        } else if (n2_set && !n1_set) {
                            set_node(n1, x[n2] + v);
                        } else if (!n1_set && !n2_set) {
                            set_node(n1, v);
                            set_node(n2, 0.0);
                        }
                    }
                }
            }, devices_[i]);
        }

        // Propagate voltages across closed ideal switches.
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, IdealSwitch>) {
                    if (!dev.is_closed()) {
                        return;
                    }
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    if (n1 >= 0 && n2 >= 0) {
                        bool n1_set = node_set[static_cast<std::size_t>(n1)];
                        bool n2_set = node_set[static_cast<std::size_t>(n2)];
                        if (n1_set && !n2_set) {
                            set_node(n2, x[n1]);
                        } else if (n2_set && !n1_set) {
                            set_node(n1, x[n2]);
                        }
                    } else if (n1 >= 0 && n2 < 0) {
                        set_node(n1, 0.0);
                    } else if (n2 >= 0 && n1 < 0) {
                        set_node(n2, 0.0);
                    }
                }
            }, devices_[i]);
        }

        // Apply inductor initial currents.
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Inductor>) {
                    if (conn.branch_index >= 0 && conn.branch_index < system_size()) {
                        x[conn.branch_index] = dev.current_prev();
                    }
                }
            }, devices_[i]);
        }

        // Compute KCL-consistent V-source branch currents from the
        // node voltages and the deterministic device currents (resistor,
        // inductor IC). Without this pass `initial_state()` returns
        // x[branch_V] = 0, which makes the t=0+ residual at the source
        // node non-zero by `i_R = (V_src − V_neighbor)/R`. Trapezoidal /
        // BDF1 then "settle" the inconsistency by oscillating
        // `x[branch_V]` between 0 and 2·I_true on the first few
        // sampled steps — surfaced by
        // `level1_components/test_basic_components.py::TestCapacitor::test_capacitor_charging_current`.
        //
        // For each V-source between (npos, nneg):
        //   I_branch[V] = −(Σ currents leaving npos via other devices)
        //
        // We sum over the deterministic non-V-source neighbors:
        //   - Resistors:        i = (v_npos − v_other) / R
        //   - Inductors:        i = ±x[branch_inductor] depending on
        //                          which terminal of the inductor lands
        //                          on `npos`
        //   - Current sources:  i = ±dev.current()
        //
        // Reactive devices in their IC state (cap at v_prev, inductor at
        // i_prev) are treated as instantaneous V/I sources for this
        // single-node KCL — that is, the cap appears as a voltage rail
        // (no current contribution at t=0+ unless other devices push
        // through it) and the inductor as a forced current.
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& vs_conn = connections_[i];
            if (!std::holds_alternative<VoltageSource>(devices_[i])
                && !std::holds_alternative<PWMVoltageSource>(devices_[i])
                && !std::holds_alternative<SineVoltageSource>(devices_[i])
                && !std::holds_alternative<PulseVoltageSource>(devices_[i])) {
                continue;
            }
            if (vs_conn.branch_index < 0 || vs_conn.branch_index >= system_size()) {
                continue;
            }
            const Index npos = vs_conn.nodes[0];
            if (npos < 0) {
                // Source-pos at ground: the branch current is determined
                // by the negative-side KCL instead. Skip for now (rare).
                continue;
            }

            Real sum_leaving = 0.0;
            for (std::size_t j = 0; j < devices_.size(); ++j) {
                if (j == i) continue;  // Skip the V-source itself.
                const auto& other_conn = connections_[j];
                std::visit([&](const auto& other_dev) {
                    using OT = std::decay_t<decltype(other_dev)>;
                    if constexpr (std::is_same_v<OT, Resistor>) {
                        const Index a = other_conn.nodes[0];
                        const Index b = other_conn.nodes[1];
                        const Real R = other_dev.resistance();
                        if (R <= 0.0) return;
                        const Real va = (a >= 0) ? x[a] : 0.0;
                        const Real vb = (b >= 0) ? x[b] : 0.0;
                        // Current leaves npos only if npos is a or b.
                        if (a == npos) {
                            sum_leaving += (va - vb) / R;
                        } else if (b == npos) {
                            sum_leaving += (vb - va) / R;
                        }
                    } else if constexpr (std::is_same_v<OT, Inductor>) {
                        // Inductor branch current is x[branch] (set above
                        // from IC). Convention: branch_current is the
                        // current flowing from nodes[0] to nodes[1].
                        const Index a = other_conn.nodes[0];
                        const Index b = other_conn.nodes[1];
                        const Index br = other_conn.branch_index;
                        if (br < 0) return;
                        const Real i_ind = x[br];
                        if (a == npos) {
                            sum_leaving += i_ind;
                        } else if (b == npos) {
                            sum_leaving -= i_ind;
                        }
                    } else if constexpr (std::is_same_v<OT, CurrentSource>) {
                        // Current source pushes `dev.current()` from
                        // nodes[0] to nodes[1] (Pulsim convention).
                        const Index a = other_conn.nodes[0];
                        const Index b = other_conn.nodes[1];
                        const Real i_cs = other_dev.current();
                        if (a == npos) {
                            sum_leaving += i_cs;
                        } else if (b == npos) {
                            sum_leaving -= i_cs;
                        }
                    } else if constexpr (std::is_same_v<OT, VoltageSource>
                                      || std::is_same_v<OT, PWMVoltageSource>
                                      || std::is_same_v<OT, SineVoltageSource>
                                      || std::is_same_v<OT, PulseVoltageSource>) {
                        // Other V-sources contribute their (already-set)
                        // branch current. Convention: branch leaves
                        // nodes[0] (npos of the source).
                        const Index a = other_conn.nodes[0];
                        const Index b = other_conn.nodes[1];
                        const Index br = other_conn.branch_index;
                        if (br < 0) return;
                        const Real i_other = x[br];
                        if (a == npos) {
                            sum_leaving += i_other;
                        } else if (b == npos) {
                            sum_leaving -= i_other;
                        }
                    }
                    // Capacitors, diodes, switches, MOSFETs, IGBTs:
                    // contribute either 0 at IC (cap is a v_prev rail
                    // with no current at t=0+ from the integrator's
                    // companion model) or are nonlinear (handled by
                    // Newton at the first transient step). Skipping
                    // them here is conservative.
                }, devices_[j]);
            }
            x[vs_conn.branch_index] = -sum_leaving;
        }

        return x;
    }

    // =========================================================================
    // Device Addition
    // =========================================================================

    void add_resistor(const std::string& name, Index n1, Index n2, Real R) {
        devices_.emplace_back(Resistor(R, name));
        connections_.push_back({name, {n1, n2}, -1});
        register_connection_name(connections_.size() - 1);
        resistor_cache_.push_back({n1, n2, R == 0.0 ? 0.0 : 1.0 / R});
    }

    void add_capacitor(const std::string& name, Index n1, Index n2, Real C, Real ic = 0.0) {
        devices_.emplace_back(Capacitor(C, ic, name));
        connections_.push_back({name, {n1, n2}, -1});
        register_connection_name(connections_.size() - 1);
    }

    /// Add RC snubber branch (R and C in parallel across the same two nodes).
    void add_snubber_rc(const std::string& name, Index n1, Index n2, Real R, Real C, Real ic = 0.0) {
        add_resistor(name + "__R", n1, n2, R);
        add_capacitor(name + "__C", n1, n2, C, ic);
    }

    void add_inductor(const std::string& name, Index n1, Index n2, Real L, Real ic = 0.0) {
        Index br = num_nodes() + num_branches_;
        devices_.emplace_back(Inductor(L, ic, name));
        connections_.push_back({name, {n1, n2}, br});
        register_connection_name(connections_.size() - 1);
        num_branches_++;
    }

    void add_voltage_source(const std::string& name, Index npos, Index nneg, Real V) {
        Index br = num_nodes() + num_branches_;
        auto vs = VoltageSource(V, name);
        vs.set_branch_index(br);
        devices_.emplace_back(std::move(vs));
        connections_.push_back({name, {npos, nneg}, br});
        register_connection_name(connections_.size() - 1);
        num_branches_++;
    }

    void add_current_source(const std::string& name, Index npos, Index nneg, Real I) {
        devices_.emplace_back(CurrentSource(I, name));
        connections_.push_back({name, {npos, nneg}, -1});
        register_connection_name(connections_.size() - 1);
    }

    void add_diode(const std::string& name, Index anode, Index cathode,
                   Real g_on = 1e3, Real g_off = 1e-9) {
        devices_.emplace_back(IdealDiode(g_on, g_off, name));
        connections_.push_back({name, {anode, cathode}, -1});
        register_connection_name(connections_.size() - 1);
    }

    void add_switch(const std::string& name, Index n1, Index n2,
                    bool closed = false, Real g_on = 1e6, Real g_off = 1e-12) {
        devices_.emplace_back(IdealSwitch(g_on, g_off, closed, name));
        connections_.push_back({name, {n1, n2}, -1});
        register_connection_name(connections_.size() - 1);
    }

    /// Add voltage-controlled switch (controlled by a PWM source)
    /// ctrl: control node (typically driven by PWM), t1/t2: switch terminals
    /// hysteresis: tanh-smoothing width (V) for the behavioral conductance.
    ///   Default 0.5 preserves the legacy event-detection cadence used by
    ///   pre-existing converter tests; SwitchParams-driven Python paths
    ///   pass a narrower value (e.g. 0.05 V) for sharper threshold tests.
    void add_vcswitch(const std::string& name, Index ctrl, Index t1, Index t2,
                      Real v_threshold = 2.5, Real g_on = 1e3, Real g_off = 1e-9,
                      Real hysteresis = 0.5) {
        VoltageControlledSwitch::Params params;
        params.v_threshold = v_threshold;
        params.g_on = g_on;
        params.g_off = g_off;
        params.hysteresis = hysteresis;
        devices_.emplace_back(VoltageControlledSwitch(params, name));
        connections_.push_back({name, {ctrl, t1, t2}, -1});
        register_connection_name(connections_.size() - 1);
    }

    void add_mosfet(const std::string& name, Index gate, Index drain, Index source,
                    const MOSFET::Params& params = MOSFET::Params{}) {
        devices_.emplace_back(MOSFET(params, name));
        connections_.push_back({name, {gate, drain, source}, -1});
        register_connection_name(connections_.size() - 1);
    }

    void add_igbt(const std::string& name, Index gate, Index collector, Index emitter,
                  const IGBT::Params& params = IGBT::Params{}) {
        devices_.emplace_back(IGBT(params, name));
        connections_.push_back({name, {gate, collector, emitter}, -1});
        register_connection_name(connections_.size() - 1);
    }

    /// Add transformer with turns ratio N:1 (primary:secondary)
    /// @param name Transformer name
    /// @param p1 Primary positive terminal
    /// @param p2 Primary negative terminal
    /// @param s1 Secondary positive terminal
    /// @param s2 Secondary negative terminal
    /// @param turns_ratio N:1 ratio (e.g., 2.0 means 2:1 step-down)
    void add_transformer(const std::string& name, Index p1, Index p2,
                         Index s1, Index s2, Real turns_ratio = 1.0) {
        Index br_p = num_nodes() + num_branches_;
        Index br_s = num_nodes() + num_branches_ + 1;

        auto xfmr = Transformer(turns_ratio, name);
        xfmr.set_branch_indices(br_p, br_s);
        devices_.emplace_back(std::move(xfmr));
        connections_.push_back({name, {p1, p2, s1, s2}, br_p, br_s});
        register_connection_name(connections_.size() - 1);
        num_branches_ += 2;  // Transformer uses two branch currents
    }

    /// Number of devices
    [[nodiscard]] std::size_t num_devices() const { return devices_.size(); }

    /// Register a virtual (non-stamping) component in the runtime graph.
    void add_virtual_component(
        const std::string& type,
        const std::string& name,
        std::vector<Index> nodes,
        std::unordered_map<std::string, Real> numeric_params = {},
        std::unordered_map<std::string, std::string> metadata = {}) {
        lookup_table_cache_.erase(name);
        transfer_coeff_cache_.erase(name + ":num");
        transfer_coeff_cache_.erase(name + ":den");
        transfer_input_history_.erase(name);
        transfer_output_history_.erase(name);
        virtual_time_history_.erase(name);
        virtual_components_.push_back(
            VirtualComponent{type, name, std::move(nodes), std::move(numeric_params), std::move(metadata)});
    }

    /// Number of virtual components
    [[nodiscard]] std::size_t num_virtual_components() const { return virtual_components_.size(); }

    /// Access all virtual components
    [[nodiscard]] const std::vector<VirtualComponent>& virtual_components() const {
        return virtual_components_;
    }

    /// Convenience list of virtual component names.
    [[nodiscard]] std::vector<std::string> virtual_component_names() const {
        std::vector<std::string> names;
        names.reserve(virtual_components_.size());
        for (const auto& component : virtual_components_) {
            names.push_back(component.name);
        }
        return names;
    }

    /// Canonical mixed-domain phase order used by the runtime scheduler.
    [[nodiscard]] static std::vector<std::string> mixed_domain_phase_order() {
        return {"electrical", "control", "events", "instrumentation"};
    }

    /// Return metadata for channels produced by virtual components.
    [[nodiscard]] std::unordered_map<std::string, VirtualChannelMetadata> virtual_channel_metadata() const {
        std::unordered_map<std::string, VirtualChannelMetadata> metadata;
        metadata.reserve(virtual_components_.size() * 2);

        auto channel_domain = [](const std::string& type) -> std::string {
            if (type == "thermal_scope") return "thermal";
            if (type == "relay" || type == "fuse" || type == "circuit_breaker" ||
                type == "thyristor" || type == "triac") {
                return "events";
            }
            if (type == "saturable_inductor" || type == "coupled_inductor") {
                return "electrical";
            }
            if (type == "voltage_probe" || type == "current_probe" || type == "power_probe" ||
                type == "electrical_scope") {
                return "instrumentation";
            }
            return "control";
        };

        for (const auto& component : virtual_components_) {
            VirtualChannelMetadata base{
                component.type,
                component.name,
                channel_domain(component.type),
                component.nodes
            };
            metadata.emplace(component.name, base);

            if (component.type == "relay") {
                auto event = base;
                event.domain = "events";
                metadata.emplace(component.name + ".state", event);
                metadata.emplace(component.name + ".no_state", event);
                metadata.emplace(component.name + ".nc_state", std::move(event));
            } else if (component.type == "fuse" || component.type == "circuit_breaker" ||
                       component.type == "thyristor" || component.type == "triac") {
                auto event = base;
                event.domain = "events";
                metadata.emplace(component.name + ".state", std::move(event));
            } else if (component.type == "pwm_generator") {
                auto control = base;
                control.domain = "control";
                metadata.emplace(component.name + ".duty", std::move(control));
                auto carrier = base;
                carrier.domain = "control";
                metadata.emplace(component.name + ".carrier", std::move(carrier));
            } else if (component.type == "saturable_inductor") {
                auto electrical = base;
                electrical.domain = "electrical";
                metadata.emplace(component.name + ".l_eff", electrical);
                metadata.emplace(component.name + ".i_est", std::move(electrical));
            } else if (component.type == "coupled_inductor") {
                auto electrical = base;
                electrical.domain = "electrical";
                metadata.emplace(component.name + ".mutual", electrical);
                metadata.emplace(component.name + ".k", std::move(electrical));
            }
        }

        return metadata;
    }

    /// Execute one deterministic mixed-domain scheduler pass for a timestep.
    /// Phase order: electrical -> control -> events -> instrumentation.
    [[nodiscard]] MixedDomainStepResult execute_mixed_domain_step(const Vector& x, Real time) {
        set_current_time(time);
        MixedDomainStepResult result;
        result.phase_order = mixed_domain_phase_order();

        auto node_voltage = [&](Index node) -> Real {
            if (node < 0 || node >= system_size()) {
                return 0.0;
            }
            return x[node];
        };

        auto get_numeric = [&](const VirtualComponent& component, const std::string& key, Real fallback) {
            const auto it = component.numeric_params.find(key);
            return (it != component.numeric_params.end()) ? it->second : fallback;
        };
        auto has_numeric = [&](const VirtualComponent& component, const std::string& key) {
            return component.numeric_params.find(key) != component.numeric_params.end();
        };

        auto has_type = [](const std::unordered_set<std::string>& set, const std::string& value) {
            return set.find(value) != set.end();
        };

        const std::unordered_set<std::string> control_types = {
            "op_amp", "comparator", "pi_controller", "pid_controller",
            "math_block", "gain", "sum", "subtraction",
            "pwm_generator", "integrator", "differentiator",
            "limiter", "rate_limiter", "hysteresis", "lookup_table",
            "transfer_function", "delay_block", "sample_hold",
            "state_machine", "signal_mux", "signal_demux"
        };

        // Phase 2: control update
        for (const auto& component : virtual_components_) {
            if (!has_type(control_types, component.type)) {
                continue;
            }

            const Real in0 = (component.nodes.size() > 0) ? node_voltage(component.nodes[0]) : 0.0;
            const Real in1 = (component.nodes.size() > 1) ? node_voltage(component.nodes[1]) : 0.0;
            const Real signal = in0 - in1;
            Real output = signal * get_numeric(component, "gain", 1.0);

            const Real dt = [&]() {
                const auto it = virtual_last_time_.find(component.name);
                if (it == virtual_last_time_.end()) return Real{0.0};
                return std::max<Real>(0.0, time - it->second);
            }();
            const auto maybe_limit_output = [&](Real value, bool force_rails = false) {
                bool has_limits = force_rails ||
                    has_numeric(component, "output_min") || has_numeric(component, "output_max") ||
                    has_numeric(component, "min") || has_numeric(component, "max") ||
                    has_numeric(component, "rail_low") || has_numeric(component, "rail_high");
                if (!has_limits) {
                    return value;
                }

                Real lo = force_rails
                    ? get_numeric(component, "rail_low", -15.0)
                    : get_numeric(component, "output_min",
                                  get_numeric(component, "min", get_numeric(component, "rail_low", -1e12)));
                Real hi = force_rails
                    ? get_numeric(component, "rail_high", 15.0)
                    : get_numeric(component, "output_max",
                                  get_numeric(component, "max", get_numeric(component, "rail_high", 1e12)));
                if (lo > hi) std::swap(lo, hi);
                return std::clamp(value, lo, hi);
            };

            if (component.type == "op_amp") {
                const Real gain = get_numeric(component, "open_loop_gain",
                    get_numeric(component, "gain", 1e5));
                const Real offset = get_numeric(component, "offset", 0.0);
                output = gain * signal + offset;
                output = maybe_limit_output(output, true);
            } else if (component.type == "gain") {
                const Real gain = get_numeric(component, "gain", 1.0);
                const Real offset = get_numeric(component, "offset", 0.0);
                output = signal * gain + offset;
                output = maybe_limit_output(output);
            } else if (component.type == "sum") {
                output = in0 + in1;
                output = get_numeric(component, "gain", 1.0) * output + get_numeric(component, "offset", 0.0);
                output = maybe_limit_output(output);
            } else if (component.type == "subtraction") {
                output = in0 - in1;
                output = get_numeric(component, "gain", 1.0) * output + get_numeric(component, "offset", 0.0);
                output = maybe_limit_output(output);
            } else if (component.type == "pi_controller") {
                const Real kp = get_numeric(component, "kp", get_numeric(component, "gain", 1.0));
                const Real ki = get_numeric(component, "ki", 0.0);
                const std::string integral_key = component.name + ".integral";
                Real integral = virtual_signal_state_[integral_key];
                integral += signal * dt;
                output = kp * signal + ki * integral;
                const Real limited = maybe_limit_output(output);
                const bool anti_windup = get_numeric(component, "anti_windup", 1.0) > 0.5;
                if (anti_windup && ki != 0.0 && dt > 0.0 && limited != output) {
                    integral = (limited - kp * signal) / ki;
                    output = limited;
                } else {
                    output = limited;
                }
                virtual_signal_state_[integral_key] = integral;
            } else if (component.type == "pid_controller") {
                const Real kp = get_numeric(component, "kp", get_numeric(component, "gain", 1.0));
                const Real ki = get_numeric(component, "ki", 0.0);
                const Real kd = get_numeric(component, "kd", 0.0);
                const std::string integral_key = component.name + ".integral";
                const std::string prev_error_key = component.name + ".prev_error";
                Real integral = virtual_signal_state_[integral_key];
                integral += signal * dt;
                const Real prev_error = virtual_last_input_[prev_error_key];
                const Real derivative = (dt > 0.0) ? (signal - prev_error) / dt : 0.0;
                output = kp * signal + ki * integral + kd * derivative;
                const Real limited = maybe_limit_output(output);
                const bool anti_windup = get_numeric(component, "anti_windup", 1.0) > 0.5;
                if (anti_windup && ki != 0.0 && dt > 0.0 && limited != output) {
                    integral = (limited - kp * signal - kd * derivative) / ki;
                    output = limited;
                } else {
                    output = limited;
                }
                virtual_signal_state_[integral_key] = integral;
                virtual_last_input_[prev_error_key] = signal;
            } else if (component.type == "math_block") {
                const std::string op = [&]() -> std::string {
                    const auto it = component.metadata.find("operation");
                    return it == component.metadata.end() ? "add" : it->second;
                }();
                if (op == "sub") output = in0 - in1;
                else if (op == "mul") output = in0 * in1;
                else if (op == "div") output = std::abs(in1) > 1e-12 ? (in0 / in1) : 0.0;
                else output = in0 + in1;
            } else if (component.type == "integrator") {
                Real integral = virtual_signal_state_[component.name];
                integral += signal * dt;
                output = maybe_limit_output(integral);
            } else if (component.type == "differentiator") {
                const Real previous = virtual_last_input_[component.name];
                const Real raw = (dt > 0.0) ? (signal - previous) / dt : 0.0;
                const Real alpha = std::clamp(get_numeric(component, "alpha", 0.0), 0.0, 1.0);
                const Real previous_output = virtual_signal_state_[component.name];
                output = alpha * previous_output + (1.0 - alpha) * raw;
                output = maybe_limit_output(output);
            } else if (component.type == "limiter") {
                output = maybe_limit_output(output);
            } else if (component.type == "rate_limiter") {
                const Real rise = std::abs(get_numeric(component, "rising_rate", 1e6));
                const Real fall = std::abs(get_numeric(component, "falling_rate", rise));
                const Real previous = virtual_signal_state_[component.name];
                if (dt > 0.0) {
                    const Real delta = output - previous;
                    const Real up = rise * dt;
                    const Real down = fall * dt;
                    if (delta > up) output = previous + up;
                    else if (delta < -down) output = previous - down;
                }
                output = maybe_limit_output(output);
            } else if (component.type == "hysteresis" || component.type == "comparator") {
                const Real threshold = get_numeric(component, "threshold", 0.0);
                const Real band = std::abs(get_numeric(component, "hysteresis", 0.0));
                bool state = virtual_binary_state_[component.name];
                if (state) {
                    if (signal < threshold - 0.5 * band) state = false;
                } else {
                    if (signal > threshold + 0.5 * band) state = true;
                }
                virtual_binary_state_[component.name] = state;
                output = state ? get_numeric(component, "high", 1.0) : get_numeric(component, "low", 0.0);
            } else if (component.type == "lookup_table") {
                const auto& samples = lookup_table_samples(component);
                output = samples.empty()
                    ? signal
                    : interpolate_lookup(component, samples, signal);
                output = maybe_limit_output(output);
            } else if (component.type == "sample_hold") {
                const Real period = std::max<Real>(get_numeric(component, "sample_period", 0.0), 0.0);
                const std::string hold_value_key = component.name + ".held_value";
                const std::string hold_time_key = component.name + ".held_time";
                const auto hold_time_it = virtual_signal_state_.find(hold_time_key);
                const bool first = hold_time_it == virtual_signal_state_.end();
                if (first || period <= 0.0 || (time - hold_time_it->second) >= period) {
                    virtual_signal_state_[hold_value_key] = signal;
                    virtual_signal_state_[hold_time_key] = time;
                }
                output = virtual_signal_state_[hold_value_key];
            } else if (component.type == "delay_block") {
                const Real delay = std::max<Real>(get_numeric(component, "delay", 0.0), 0.0);
                auto& history = virtual_time_history_[component.name];
                if (history.empty() || time >= history.back().first) {
                    history.emplace_back(time, signal);
                } else {
                    history.back() = {time, signal};
                }

                if (delay <= 0.0) {
                    output = signal;
                } else {
                    const Real target_time = time - delay;
                    if (target_time <= history.front().first) {
                        output = history.front().second;
                    } else if (target_time >= history.back().first) {
                        output = history.back().second;
                    } else {
                        output = history.back().second;
                        for (std::size_t i = 1; i < history.size(); ++i) {
                            const auto& p0 = history[i - 1];
                            const auto& p1 = history[i];
                            if (target_time <= p1.first) {
                                const Real span = std::max<Real>(p1.first - p0.first, 1e-15);
                                const Real alpha = std::clamp((target_time - p0.first) / span, 0.0, 1.0);
                                output = p0.second + alpha * (p1.second - p0.second);
                                break;
                            }
                        }
                    }

                    const Real keep_after = target_time - std::max<Real>(delay, timestep_);
                    while (history.size() > 2 && history[1].first < keep_after) {
                        history.pop_front();
                    }
                }
            } else if (component.type == "transfer_function") {
                const auto& num = transfer_coefficients(component, "num");
                const auto& den = transfer_coefficients(component, "den");
                if (!num.empty() && !den.empty() && std::abs(den.front()) > 1e-15) {
                    auto& x_hist = transfer_input_history_[component.name];
                    auto& y_hist = transfer_output_history_[component.name];
                    x_hist.push_front(signal);
                    while (x_hist.size() < num.size()) x_hist.push_back(0.0);
                    while (x_hist.size() > num.size()) x_hist.pop_back();

                    Real y = 0.0;
                    for (std::size_t i = 0; i < num.size(); ++i) {
                        y += num[i] * x_hist[i];
                    }
                    for (std::size_t i = 1; i < den.size(); ++i) {
                        const Real y_prev = (i - 1 < y_hist.size()) ? y_hist[i - 1] : 0.0;
                        y -= den[i] * y_prev;
                    }
                    output = maybe_limit_output(y / den.front());
                    y_hist.push_front(output);
                    const std::size_t y_limit = den.size() > 1 ? den.size() - 1 : 0;
                    while (y_hist.size() > y_limit) y_hist.pop_back();
                } else {
                    const Real alpha = std::clamp(get_numeric(component, "alpha", 0.2), 0.0, 1.0);
                    const Real previous = virtual_signal_state_[component.name];
                    output = previous + alpha * (signal - previous);
                }
            } else if (component.type == "pwm_generator") {
                const Real frequency = std::max<Real>(get_numeric(component, "frequency", 1e3), 1.0);
                Real duty = get_numeric(component, "duty", 0.5);
                // duty_from_channel: read duty from a previously-evaluated virtual signal
                if (const auto it = component.metadata.find("duty_from_channel");
                    it != component.metadata.end() && !it->second.empty()) {
                    const auto sig_it = virtual_signal_state_.find(it->second);
                    if (sig_it != virtual_signal_state_.end()) {
                        duty = get_numeric(component, "duty_offset", 0.0) +
                               get_numeric(component, "duty_gain", 1.0) * sig_it->second;
                    }
                } else if (get_numeric(component, "duty_from_input", 0.0) > 0.5) {
                    duty = get_numeric(component, "duty_offset", 0.0) +
                           get_numeric(component, "duty_gain", 1.0) * in0;
                }
                const Real duty_min = std::clamp(get_numeric(component, "duty_min", 0.0), 0.0, 1.0);
                const Real duty_max = std::clamp(get_numeric(component, "duty_max", 1.0), 0.0, 1.0);
                if (duty_min <= duty_max) {
                    duty = std::clamp(duty, duty_min, duty_max);
                } else {
                    duty = std::clamp(duty, duty_max, duty_min);
                }
                const Real phase = std::fmod(std::max<Real>(0.0, time * frequency), 1.0);
                const Real carrier = (phase <= 0.5) ? (2.0 * phase) : (2.0 * (1.0 - phase));
                output = (duty > carrier) ? 1.0 : 0.0;
                if (const auto it = component.metadata.find("target_component");
                    it != component.metadata.end()) {
                    set_switch_state(it->second, output > 0.5);
                }
                result.channel_values[component.name + ".duty"] = duty;
                result.channel_values[component.name + ".carrier"] = carrier;
            } else if (component.type == "state_machine") {
                std::string mode = "toggle";
                if (const auto it = component.metadata.find("mode"); it != component.metadata.end()) {
                    mode = normalize_mode_token(it->second);
                }
                bool state = virtual_binary_state_[component.name];
                const Real trigger = get_numeric(component, "threshold", 0.5);
                if (mode == "set_reset" || mode == "sr") {
                    const Real set_signal = in0;
                    const Real reset_signal = (component.nodes.size() > 1)
                        ? node_voltage(component.nodes[1])
                        : 0.0;
                    const bool set_active = set_signal > trigger;
                    const bool reset_active = reset_signal > trigger;
                    if (set_active && !reset_active) {
                        state = true;
                    } else if (reset_active) {
                        state = false;
                    }
                } else if (mode == "level") {
                    state = signal > trigger;
                } else {
                    const Real prev = virtual_last_input_[component.name];
                    if (prev <= trigger && signal > trigger) {
                        state = !state;
                    }
                }
                virtual_binary_state_[component.name] = state;
                output = state ? get_numeric(component, "high", 1.0)
                               : get_numeric(component, "low", 0.0);
            } else if (component.type == "signal_mux") {
                const int select = static_cast<int>(std::llround(get_numeric(component, "select_index", 0.0)));
                const std::size_t selected = static_cast<std::size_t>(std::max(select, 0));
                output = selected < component.nodes.size() ? node_voltage(component.nodes[selected]) : in0;
            } else if (component.type == "signal_demux") {
                output = in0;
            }

            virtual_signal_state_[component.name] = output;
            virtual_last_input_[component.name] = signal;
            virtual_last_time_[component.name] = time;
            result.channel_values[component.name] = output;
        }

        for (const auto& binding : switch_driver_bindings_) {
            const auto& driver_name = binding.first;
            const auto& target_name = binding.second;
            const auto source_index = find_connection_index(driver_name);
            if (!source_index.has_value()) {
                continue;
            }

            bool apply = false;
            bool closed = false;
            std::visit([&](const auto& source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, PulseVoltageSource>) {
                    const Real level = source.voltage_at(time);
                    const auto& p = source.params();
                    const Real threshold = 0.5 * (p.v_initial + p.v_pulse);
                    closed = (p.v_pulse >= p.v_initial) ? (level > threshold)
                                                        : (level < threshold);
                    apply = true;
                } else if constexpr (std::is_same_v<T, PWMVoltageSource>) {
                    const Real level = source.voltage_at(time);
                    const auto& p = source.params();
                    const Real threshold = 0.5 * (p.v_low + p.v_high);
                    closed = (p.v_high >= p.v_low) ? (level > threshold)
                                                   : (level < threshold);
                    apply = true;
                }
            }, devices_[*source_index]);

            if (apply) {
                set_switch_state(target_name, closed);
            }
        }

        // Phase 3: event-driven transitions
        for (const auto& component : virtual_components_) {
            if (component.type != "relay" && component.type != "fuse" &&
                component.type != "circuit_breaker" && component.type != "thyristor" &&
                component.type != "triac") {
                continue;
            }

            const Real dt = [&]() {
                const auto it = virtual_last_time_.find(component.name);
                if (it == virtual_last_time_.end()) return Real{0.0};
                return std::max<Real>(0.0, time - it->second);
            }();

            const auto initial_closed = [&]() -> bool {
                Real fallback = 1.0;
                if (component.type == "relay" || component.type == "thyristor" ||
                    component.type == "triac") {
                    fallback = 0.0;
                }
                return get_numeric(component, "initial_closed", fallback) > 0.5;
            };

            bool closed = virtual_binary_state_.contains(component.name)
                ? virtual_binary_state_[component.name]
                : initial_closed();

            if (component.type == "relay") {
                const Real coil = (component.nodes.size() > 1)
                    ? (node_voltage(component.nodes[0]) - node_voltage(component.nodes[1]))
                    : 0.0;
                const Real pickup = std::abs(get_numeric(component, "pickup_current",
                                        get_numeric(component, "pickup_voltage", 1.0)));
                const Real dropout = std::abs(get_numeric(component, "dropout_current", pickup * 0.8));
                if (closed) {
                    if (std::abs(coil) < dropout) closed = false;
                } else {
                    if (std::abs(coil) >= pickup) closed = true;
                }

                const bool no_closed = closed;
                const bool nc_closed = !closed;

                if (const auto it = component.metadata.find("target_component_no");
                    it != component.metadata.end()) {
                    set_switch_state(it->second, no_closed);
                }
                if (const auto it = component.metadata.find("target_component_nc");
                    it != component.metadata.end()) {
                    set_switch_state(it->second, nc_closed);
                }
                if (const auto it = component.metadata.find("target_component");
                    it != component.metadata.end()) {
                    set_switch_state(it->second, no_closed);
                }

                result.channel_values[component.name + ".state"] = closed ? 1.0 : 0.0;
                result.channel_values[component.name + ".no_state"] = no_closed ? 1.0 : 0.0;
                result.channel_values[component.name + ".nc_state"] = nc_closed ? 1.0 : 0.0;
            } else if (component.type == "fuse" || component.type == "circuit_breaker") {
                const Real v_term = (component.nodes.size() > 1)
                    ? (node_voltage(component.nodes[0]) - node_voltage(component.nodes[1]))
                    : 0.0;
                const Real g_on = std::max<Real>(get_numeric(component, "g_on", 1e4), 1e-12);
                const Real i_abs = closed ? std::abs(v_term) * g_on : 0.0;

                if (component.type == "fuse") {
                    const Real rating = std::max<Real>(get_numeric(component, "rating", 1.0), 1e-6);
                    const Real default_i2t = rating * rating * 1e-3;
                    const Real blow_i2t = std::max<Real>(get_numeric(component, "blow_i2t", default_i2t), 1e-12);
                    const std::string stress_key = component.name + ".i2t";
                    Real i2t = virtual_signal_state_[stress_key];
                    i2t += i_abs * i_abs * dt;
                    virtual_signal_state_[stress_key] = i2t;
                    result.channel_values[stress_key] = i2t;
                    if (i2t >= blow_i2t) {
                        closed = false;
                    }
                } else {
                    const Real trip_current = std::abs(get_numeric(component, "trip_current",
                        get_numeric(component, "rating", 1.0)));
                    const Real trip_time = std::max<Real>(get_numeric(component, "trip_time", 0.0), 0.0);
                    const std::string timer_key = component.name + ".trip_timer";
                    Real timer = virtual_signal_state_[timer_key];
                    if (i_abs >= trip_current) {
                        timer += dt;
                    } else {
                        timer = 0.0;
                    }
                    virtual_signal_state_[timer_key] = timer;
                    result.channel_values[timer_key] = timer;
                    if (trip_time <= 0.0) {
                        if (i_abs >= trip_current) closed = false;
                    } else if (timer >= trip_time) {
                        closed = false;
                    }
                }
            } else {
                const Real v_main = (component.nodes.size() > 2)
                    ? (node_voltage(component.nodes[1]) - node_voltage(component.nodes[2]))
                    : 0.0;
                const Real v_gate_ref = (component.nodes.size() > 2)
                    ? node_voltage(component.nodes[2])
                    : 0.0;
                const Real v_gate = (component.nodes.size() > 0)
                    ? (node_voltage(component.nodes[0]) - v_gate_ref)
                    : 0.0;

                const Real gate_threshold = std::max<Real>(
                    std::abs(get_numeric(component, "gate_threshold", 1.0)), 1e-3);
                const Real holding_current = std::max<Real>(
                    std::abs(get_numeric(component, "holding_current", 0.05)), 1e-6);
                const Real latch_current = std::max<Real>(
                    std::abs(get_numeric(component, "latch_current", holding_current * 1.2)),
                    holding_current);
                const Real g_on = std::max<Real>(get_numeric(component, "g_on", 1e4), 1.0);

                const bool gate_active = (component.type == "triac")
                    ? (std::abs(v_gate) >= gate_threshold)
                    : (v_gate >= gate_threshold);
                const Real i_abs = std::abs(v_main) * g_on;
                const Real i_forward = std::max<Real>(v_main, 0.0) * g_on;

                if (closed) {
                    if (component.type == "thyristor") {
                        if (v_main <= 0.0 || i_forward < holding_current) {
                            closed = false;
                        }
                    } else {
                        if (i_abs < holding_current) {
                            closed = false;
                        }
                    }
                } else if (gate_active) {
                    if (component.type == "thyristor") {
                        if (v_main > 0.0 && i_forward >= latch_current) {
                            closed = true;
                        }
                    } else if (i_abs >= latch_current) {
                        closed = true;
                    }
                }

                result.channel_values[component.name + ".trigger"] = gate_active ? 1.0 : 0.0;
                result.channel_values[component.name + ".i_est"] =
                    (component.type == "thyristor") ? i_forward : i_abs;
            }

            virtual_binary_state_[component.name] = closed;
            virtual_last_time_[component.name] = time;

            if (component.type != "relay") {
                if (const auto it = component.metadata.find("target_component");
                    it != component.metadata.end()) {
                    set_switch_state(it->second, closed);
                }
                result.channel_values[component.name + ".state"] = closed ? 1.0 : 0.0;
            }
        }

        // Phase 4: instrumentation extraction
        const auto probes = evaluate_virtual_signals(x);
        for (const auto& probe : probes) {
            result.channel_values.emplace(probe.first, probe.second);
        }
        for (const auto& component : virtual_components_) {
            if (component.type == "saturable_inductor") {
                const auto target_it = component.metadata.find("target_component");
                const std::string target = (target_it == component.metadata.end()) ? component.name : target_it->second;
                const DeviceConnection* conn = find_connection(target);
                if (!conn || conn->branch_index < 0 || conn->branch_index >= system_size()) {
                    continue;
                }
                const Real i_est = x[conn->branch_index];
                const Real l_nom = get_numeric(component, "inductance", 1e-3);
                const Real l_eff = saturable_effective_inductance(component, i_est, l_nom);
                result.channel_values[component.name + ".l_eff"] = l_eff;
                result.channel_values[component.name + ".i_est"] = i_est;
                continue;
            }
            if (component.type == "coupled_inductor") {
                Real l1 = std::max<Real>(get_numeric(component, "l1", 0.0), 0.0);
                Real l2 = std::max<Real>(get_numeric(component, "l2", 0.0), 0.0);
                if (l1 <= 0.0 || l2 <= 0.0) {
                    const auto l1_it = component.metadata.find("target_component_1");
                    const auto l2_it = component.metadata.find("target_component_2");
                    if (l1_it != component.metadata.end()) {
                        if (const auto index = find_connection_index(l1_it->second); index.has_value()) {
                            if (const auto* inductor = std::get_if<Inductor>(&devices_[*index])) {
                                l1 = inductor->inductance();
                            }
                        }
                    }
                    if (l2_it != component.metadata.end()) {
                        if (const auto index = find_connection_index(l2_it->second); index.has_value()) {
                            if (const auto* inductor = std::get_if<Inductor>(&devices_[*index])) {
                                l2 = inductor->inductance();
                            }
                        }
                    }
                }
                const Real k = std::clamp(get_numeric(component, "coupling", get_numeric(component, "k", 0.0)),
                                          -0.999, 0.999);
                const Real mutual = k * std::sqrt(std::max<Real>(l1 * l2, 0.0));
                result.channel_values[component.name + ".k"] = k;
                result.channel_values[component.name + ".mutual"] = mutual;
                continue;
            }
            if (component.type == "electrical_scope" || component.type == "thermal_scope") {
                if (component.nodes.empty()) continue;
                Real acc = 0.0;
                for (Index node : component.nodes) {
                    acc += node_voltage(node);
                }
                result.channel_values[component.name] = acc / static_cast<Real>(component.nodes.size());
            }
        }

        // ------------------------------------------------------------------
        // Post-control diode-commutation re-scan.
        //
        // Phase 2 (pwm_generator, switch_driver_bindings, state_machine) can
        // forcibly toggle switch states via `set_switch_state`. In topologies
        // where a freewheel diode provides the inductor's discharge path
        // during the switch-OFF half-cycle (e.g. async boost / buck), the
        // diode needs to flip ON *together with* the switch turning OFF —
        // otherwise the next Newton step solves with switch=OFF / diode=OFF
        // and finds an artificial high-V "consistent" solution where the
        // inductor's stored current sits across the device's g_off.
        //
        // Two passes here:
        //   1. The standard scan_pwl_commutations(x) catches diodes whose
        //      anode-cathode voltage in the current solved x already crosses
        //      the conduction threshold.
        //   2. An inductor-current-aware pass forces any OFF diode whose
        //      anode (or cathode) sits on a node that an inductor is
        //      currently pumping current INTO — even when the *voltage* in
        //      x still looks reverse-biased. This is the case for the
        //      async-buck / async-boost freewheel right after the chopper
        //      switch turns OFF: V(sw) hasn't risen yet (it's stale from
        //      the previous step's solve), but the inductor branch current
        //      tells us the diode must commute.
        const auto post_control_events = scan_pwl_commutations(
            x, default_switching_mode_);
        if (!post_control_events.empty()) {
            commit_pwl_commutations(post_control_events);
        }

        force_inductor_driven_diode_commutations(x, result);

        return result;
    }

    /// Pre-empt diode commutations that the voltage-based admissibility
    /// scan can't see. For each IdealDiode in the OFF state, check whether:
    ///   (a) an inductor adjacent to the diode is pumping current that
    ///       must commutate somewhere; AND
    ///   (b) no ON conductor (closed switch / vcswitch / mosfet / igbt /
    ///       another already-ON diode) is currently providing an
    ///       alternative path at the diode's anode or cathode.
    /// When both are true, force the diode ON.
    ///
    /// This is the "inductor stored energy needs somewhere to go" rule
    /// that the voltage-only admissibility scan misses when Newton has
    /// just converged to an artificial high-V solution (because all
    /// switches around the inductor flipped OFF simultaneously). The
    /// adjacency check prevents false positives during the normal ON-
    /// chopping half-cycle, when a switch already handles the current.
    void force_inductor_driven_diode_commutations(
        const Vector& x,
        MixedDomainStepResult& result) {
        if (devices_.empty()) return;
        constexpr Real I_THRESHOLD = 1e-3;  // 1 mA
        constexpr Real R_ADJACENCY_MAX = 10.0;  // Ω — anything below this we
                                                //     treat as "near-DCR";
                                                //     two nodes joined by
                                                //     such a resistor are
                                                //     electrically adjacent.

        // Lookup #1: node → list of (inductor branch, sign, history current).
        // sign = +1: node is the "from" end (current leaves here);
        // sign = -1: node is the "to" end (current arrives here).
        // i_history is the inductor's previous-step current — using the
        // committed history rather than x[branch] avoids the "bad Newton
        // solution" trap where the current solution has inverted-sign IL.
        //
        // We also flatten "DCR-like" topology: any resistor < R_ADJACENCY_MAX
        // joins two nodes into the same electrical neighborhood, so e.g.
        // [sw → R_dcr → n_l → L1 → n_esr] sees L1 as adjacent to both sw and
        // n_esr. This is critical for real-world converter wiring where the
        // inductor's DCR is modeled as a separate series resistor.
        struct InductorRef {
            Index branch;
            int sign_at_node;
            Real i_history;
        };
        std::unordered_map<Index, std::vector<InductorRef>> node_to_inductors;

        // First, collect raw inductor → node mapping
        struct InductorInfo {
            Index branch;
            Index node_a;  // "from" end (sign +1)
            Index node_b;  // "to" end (sign -1)
            Real i_history;
        };
        std::vector<InductorInfo> inductors;

        // Lookup #2: node → does any currently-ON conductive device
        // touch this node? If yes, the diode at this node doesn't
        // need to commute — the other device handles the current.
        std::unordered_set<Index> nodes_with_on_conductor;

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Inductor>) {
                    if (conn.nodes.size() < 2 || conn.branch_index < 0) return;
                    const Index br = conn.branch_index;
                    const Real i_hist = dev.history_initialized()
                                            ? dev.current_prev()
                                            : x[br];
                    inductors.push_back({br, conn.nodes[0], conn.nodes[1], i_hist});
                } else if constexpr (std::is_same_v<T, IdealSwitch>) {
                    if (!dev.pwl_state() || conn.nodes.size() < 2) return;
                    for (Index n : conn.nodes)
                        if (n >= 0) nodes_with_on_conductor.insert(n);
                } else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                    if (!dev.pwl_state() || conn.nodes.size() < 3) return;
                    // For a vcswitch, only the conducting pair (nodes[1], nodes[2])
                    // is a current path; nodes[0] is the control input.
                    if (conn.nodes[1] >= 0) nodes_with_on_conductor.insert(conn.nodes[1]);
                    if (conn.nodes[2] >= 0) nodes_with_on_conductor.insert(conn.nodes[2]);
                } else if constexpr (std::is_same_v<T, MOSFET> ||
                                     std::is_same_v<T, IGBT>) {
                    if (!dev.pwl_state() || conn.nodes.size() < 3) return;
                    if (conn.nodes[1] >= 0) nodes_with_on_conductor.insert(conn.nodes[1]);
                    if (conn.nodes[2] >= 0) nodes_with_on_conductor.insert(conn.nodes[2]);
                } else if constexpr (std::is_same_v<T, IdealDiode>) {
                    if (!dev.pwl_state() || conn.nodes.size() < 2) return;
                    if (conn.nodes[0] >= 0) nodes_with_on_conductor.insert(conn.nodes[0]);
                    if (conn.nodes[1] >= 0) nodes_with_on_conductor.insert(conn.nodes[1]);
                }
            }, devices_[i]);
        }
        if (inductors.empty()) return;

        // Build union-find groups of nodes joined by small-R resistors.
        // Each group is one "electrical neighborhood".
        std::unordered_map<Index, Index> parent;
        std::function<Index(Index)> find = [&](Index n) -> Index {
            auto it = parent.find(n);
            if (it == parent.end()) {
                parent[n] = n;
                return n;
            }
            if (it->second == n) return n;
            Index root = find(it->second);
            parent[n] = root;
            return root;
        };
        auto unite = [&](Index a, Index b) {
            const Index ra = find(a), rb = find(b);
            if (ra != rb) parent[ra] = rb;
        };
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Resistor>) {
                    if (conn.nodes.size() < 2) return;
                    if (dev.resistance() > R_ADJACENCY_MAX) return;
                    if (conn.nodes[0] >= 0 && conn.nodes[1] >= 0)
                        unite(conn.nodes[0], conn.nodes[1]);
                }
            }, devices_[i]);
        }

        // Now populate node_to_inductors: each inductor terminal entry
        // gets registered against the GROUP REPRESENTATIVE of that node.
        for (const auto& ind : inductors) {
            if (ind.node_a >= 0) {
                const Index ra = find(ind.node_a);
                node_to_inductors[ra].push_back({ind.branch, +1, ind.i_history});
            }
            if (ind.node_b >= 0) {
                const Index rb = find(ind.node_b);
                node_to_inductors[rb].push_back({ind.branch, -1, ind.i_history});
            }
        }
        // Also fold the "ON conductor" set through the same neighborhoods
        std::unordered_set<Index> on_conductor_groups;
        for (Index n : nodes_with_on_conductor) {
            on_conductor_groups.insert(find(n));
        }
        if (node_to_inductors.empty()) return;

        std::vector<PwlCommutation> forced;
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, IdealDiode>) {
                    if (dev.pwl_state()) return;  // already ON
                    if (resolve_switching_mode(dev.switching_mode(),
                                               default_switching_mode_) !=
                        SwitchingMode::Ideal) {
                        return;
                    }
                    if (conn.nodes.size() < 2) return;
                    const Index anode = conn.nodes[0];
                    const Index cathode = conn.nodes[1];
                    const Index anode_group = find(anode);
                    const Index cathode_group = find(cathode);

                    // Compute the net inductor current arriving at the
                    // anode (positive = current flows toward anode, i.e.
                    // diode wants to forward-conduct) and the net current
                    // leaving the cathode (same direction).
                    auto sum_inductor_current = [&](Index group, bool incoming) -> Real {
                        Real sum = 0.0;
                        auto it = node_to_inductors.find(group);
                        if (it == node_to_inductors.end()) return 0.0;
                        for (const auto& ind : it->second) {
                            // Use HISTORY current (last accepted step), not
                            // x[branch] — Newton may have just converged to
                            // an artifact-sign solution that the heuristic
                            // would otherwise miss.
                            const Real i_br = ind.i_history;
                            sum += (incoming ? -ind.sign_at_node : ind.sign_at_node) * i_br;
                        }
                        return sum;
                    };

                    const Real i_into_anode = sum_inductor_current(anode_group, true);
                    const Real i_out_of_cathode = sum_inductor_current(cathode_group, false);

                    // Skip if an alternative ON conductor (closed switch /
                    // diode) is already handling current at *either*
                    // electrical neighborhood — that conductor provides a
                    // much lower-impedance path than the diode's g_off, so
                    // we don't need to force-commute.
                    if (on_conductor_groups.find(anode_group) != on_conductor_groups.end() ||
                        on_conductor_groups.find(cathode_group) != on_conductor_groups.end()) {
                        return;
                    }

                    // Forward-conduction trigger
                    if (i_into_anode > I_THRESHOLD ||
                        i_out_of_cathode > I_THRESHOLD) {
                        PwlCommutation evt;
                        evt.device_index = i;
                        evt.device_name = conn.name;
                        evt.new_state = true;
                        forced.push_back(std::move(evt));
                    }
                }
            }, devices_[i]);
        }

        if (!forced.empty()) {
            commit_pwl_commutations(forced);
            for (const auto& evt : forced) {
                result.channel_values[evt.device_name + ".forced_commute"] = 1.0;
            }
        }
    }

    /// Evaluate available probe-style virtual signals for a given state vector.
    /// Unsupported virtual types are ignored by design.
    [[nodiscard]] std::unordered_map<std::string, Real> evaluate_virtual_signals(const Vector& x) const {
        std::unordered_map<std::string, Real> values;
        values.reserve(virtual_components_.size());

        auto node_voltage = [&](Index node) -> Real {
            if (node < 0 || node >= system_size()) {
                return 0.0;
            }
            return x[node];
        };

        auto branch_current = [&](const DeviceConnection* conn) -> Real {
            if (!conn || conn->branch_index < 0 || conn->branch_index >= system_size()) {
                return 0.0;
            }
            return x[conn->branch_index];
        };

        for (const auto& component : virtual_components_) {
            if (component.type == "voltage_probe") {
                if (component.nodes.size() < 2) continue;
                values[component.name] = node_voltage(component.nodes[0]) - node_voltage(component.nodes[1]);
                continue;
            }

            if (component.type == "current_probe") {
                const auto target_it = component.metadata.find("target_component");
                if (target_it == component.metadata.end()) continue;
                values[component.name] = branch_current(find_connection(target_it->second));
                continue;
            }

            if (component.type == "power_probe") {
                if (component.nodes.size() < 2) continue;
                const Real voltage = node_voltage(component.nodes[0]) - node_voltage(component.nodes[1]);
                Real current = 0.0;
                const auto target_it = component.metadata.find("target_component");
                if (target_it != component.metadata.end()) {
                    current = branch_current(find_connection(target_it->second));
                }
                values[component.name] = voltage * current;
                continue;
            }
        }

        return values;
    }

    /// Clamp overly-ideal nonlinear parameters for better numerical robustness.
    /// This is intended for fallback use when a transient repeatedly fails.
    [[nodiscard]] int apply_numerical_regularization(
        Real mosfet_kp_max = 8.0,
        Real mosfet_g_off_min = 1e-7,
        Real diode_g_on_max = 300.0,
        Real diode_g_off_min = 1e-9,
        Real igbt_g_on_max = 5e3,
        Real igbt_g_off_min = 1e-9,
        Real switch_g_on_max = 5e5,
        Real switch_g_off_min = 1e-9,
        Real vcswitch_g_on_max = 5e5,
        Real vcswitch_g_off_min = 1e-9) {
        int changed = 0;
        constexpr Real latch_gate_threshold_min = 1e-3;
        constexpr Real latch_holding_current_min = 1e-6;
        constexpr Real latch_g_on_max = 5e5;
        constexpr Real latch_g_off_min = 1e-9;

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            auto& device = devices_[i];
            const auto& conn = connections_[i];

            if (auto* mosfet = std::get_if<MOSFET>(&device)) {
                auto params = mosfet->params();
                if (params.kp > mosfet_kp_max) {
                    params.kp = mosfet_kp_max;
                }
                if (params.g_off < mosfet_g_off_min) {
                    params.g_off = mosfet_g_off_min;
                }
                // Rebuild device even when unchanged to reset internal ON/OFF history.
                device = MOSFET(params, conn.name);
                ++changed;
                continue;
            }

            if (auto* diode = std::get_if<IdealDiode>(&device)) {
                const Real g_on = std::min(diode->g_on(), diode_g_on_max);
                const Real g_off = std::max(diode->g_off(), diode_g_off_min);
                // Rebuild also resets internal conduction state.
                device = IdealDiode(g_on, g_off, conn.name);
                ++changed;
                continue;
            }

            if (auto* igbt = std::get_if<IGBT>(&device)) {
                auto params = igbt->params();
                if (params.g_on > igbt_g_on_max) {
                    params.g_on = igbt_g_on_max;
                }
                if (params.g_off < igbt_g_off_min) {
                    params.g_off = igbt_g_off_min;
                }
                device = IGBT(params, conn.name);
                ++changed;
                continue;
            }

            if (auto* sw = std::get_if<IdealSwitch>(&device)) {
                const Real g_on = std::clamp(sw->g_on(), 1.0, switch_g_on_max);
                const Real g_off = std::max(sw->g_off(), switch_g_off_min);
                if (g_on != sw->g_on() || g_off != sw->g_off()) {
                    device = IdealSwitch(g_on, g_off, sw->is_closed(), conn.name);
                    ++changed;
                }
                continue;
            }

            if (auto* vc_switch = std::get_if<VoltageControlledSwitch>(&device)) {
                VoltageControlledSwitch::Params params;
                params.v_threshold = vc_switch->v_threshold();
                params.g_on = std::clamp(vc_switch->g_on(), 1.0, vcswitch_g_on_max);
                params.g_off = std::max(vc_switch->g_off(), vcswitch_g_off_min);
                params.hysteresis = vc_switch->hysteresis();
                const bool regularized =
                    params.g_on != vc_switch->g_on() || params.g_off != vc_switch->g_off();
                if (regularized) {
                    device = VoltageControlledSwitch(params, conn.name);
                    ++changed;
                }
                continue;
            }
        }

        for (auto& component : virtual_components_) {
            if (component.type != "thyristor" && component.type != "triac" &&
                component.type != "saturable_inductor" &&
                component.type != "coupled_inductor") {
                continue;
            }

            auto read_param = [&](const std::string& key, Real fallback) -> Real {
                const auto it = component.numeric_params.find(key);
                return (it == component.numeric_params.end()) ? fallback : it->second;
            };
            auto write_param = [&](const std::string& key, Real value) {
                const auto it = component.numeric_params.find(key);
                if (it == component.numeric_params.end() || it->second != value) {
                    component.numeric_params[key] = value;
                    ++changed;
                }
            };

            if (component.type == "thyristor" || component.type == "triac") {
                const Real g_on = std::clamp(std::abs(read_param("g_on", 1e4)), 1.0, latch_g_on_max);
                const Real g_off = std::max(std::abs(read_param("g_off", 1e-9)), latch_g_off_min);
                const Real gate_threshold = std::max(
                    std::abs(read_param("gate_threshold", 1.0)), latch_gate_threshold_min);
                const Real holding_current = std::max(
                    std::abs(read_param("holding_current", 0.05)), latch_holding_current_min);
                const Real latch_current = std::max(
                    std::abs(read_param("latch_current", holding_current * 1.2)), holding_current);

                write_param("g_on", g_on);
                write_param("g_off", g_off);
                write_param("gate_threshold", gate_threshold);
                write_param("holding_current", holding_current);
                write_param("latch_current", latch_current);
                continue;
            }

            if (component.type == "saturable_inductor") {
                const Real l_unsat = std::max(std::abs(read_param("inductance", 1e-3)), 1e-9);
                const Real i_sat = std::max(std::abs(read_param("saturation_current", 1.0)), 1e-6);
                const Real l_sat = std::clamp(
                    std::abs(read_param("saturation_inductance", l_unsat * 0.2)),
                    1e-9, l_unsat);
                const Real exponent = std::clamp(read_param("saturation_exponent", 2.0), 1.0, 6.0);
                write_param("inductance", l_unsat);
                write_param("saturation_current", i_sat);
                write_param("saturation_inductance", l_sat);
                write_param("saturation_exponent", exponent);
                continue;
            }

            const Real l1 = std::max(std::abs(read_param("l1", 1e-3)), 1e-9);
            const Real l2 = std::max(std::abs(read_param("l2", 1e-3)), 1e-9);
            const Real k = std::clamp(read_param("coupling", read_param("k", 0.98)), -0.999, 0.999);
            write_param("l1", l1);
            write_param("l2", l2);
            write_param("coupling", k);
            write_param("k", k);
        }

        return changed;
    }

    // =========================================================================
    // Time-Varying Sources
    // =========================================================================

    void add_pwm_voltage_source(const std::string& name, Index npos, Index nneg,
                                 const PWMParams& params) {
        Index br = num_nodes() + num_branches_;
        auto pwm = PWMVoltageSource(params, name);
        pwm.set_branch_index(br);
        devices_.emplace_back(std::move(pwm));
        connections_.push_back({name, {npos, nneg}, br});
        register_connection_name(connections_.size() - 1);
        num_branches_++;
    }

    void add_pwm_voltage_source(const std::string& name, Index npos, Index nneg,
                                 Real v_high, Real v_low, Real frequency, Real duty) {
        PWMParams params;
        params.v_high = v_high;
        params.v_low = v_low;
        params.frequency = frequency;
        params.duty = duty;
        add_pwm_voltage_source(name, npos, nneg, params);
    }

    void add_sine_voltage_source(const std::string& name, Index npos, Index nneg,
                                  const SineParams& params) {
        Index br = num_nodes() + num_branches_;
        auto sine = SineVoltageSource(params, name);
        sine.set_branch_index(br);
        devices_.emplace_back(std::move(sine));
        connections_.push_back({name, {npos, nneg}, br});
        register_connection_name(connections_.size() - 1);
        num_branches_++;
    }

    void add_sine_voltage_source(const std::string& name, Index npos, Index nneg,
                                  Real amplitude, Real frequency, Real offset = 0.0) {
        SineParams params;
        params.amplitude = amplitude;
        params.frequency = frequency;
        params.offset = offset;
        add_sine_voltage_source(name, npos, nneg, params);
    }

    void add_pulse_voltage_source(const std::string& name, Index npos, Index nneg,
                                   const PulseParams& params) {
        Index br = num_nodes() + num_branches_;
        auto pulse = PulseVoltageSource(params, name);
        pulse.set_branch_index(br);
        devices_.emplace_back(std::move(pulse));
        connections_.push_back({name, {npos, nneg}, br});
        register_connection_name(connections_.size() - 1);
        num_branches_++;
    }

    // =========================================================================
    // PWM Duty Control
    // =========================================================================

    /// Set fixed duty cycle for a PWM source
    void set_pwm_duty(std::string_view name, Real duty) {
        if (auto* pwm = find_device<PWMVoltageSource>(name)) {
            pwm->set_duty(duty);
            return;
        }
        throw std::runtime_error("PWM source not found: " + std::string{name});
    }

    /// Set duty callback for a PWM source
    void set_pwm_duty_callback(std::string_view name,
                               std::function<Real(Real)> callback) {
        if (auto* pwm = find_device<PWMVoltageSource>(name)) {
            pwm->set_duty_callback(std::move(callback));
            return;
        }
        throw std::runtime_error("PWM source not found: " + std::string{name});
    }

    /// Clear duty callback (use fixed duty)
    void clear_pwm_duty_callback(std::string_view name) {
        if (auto* pwm = find_device<PWMVoltageSource>(name)) {
            pwm->clear_duty_callback();
            return;
        }
        throw std::runtime_error("PWM source not found: " + std::string{name});
    }

    /// Get PWM state (ON/OFF) at current time
    [[nodiscard]] bool get_pwm_state(std::string_view name) const {
        if (const auto* pwm = find_device<PWMVoltageSource>(name)) {
            return pwm->state_at(current_time_);
        }
        throw std::runtime_error("PWM source not found: " + std::string{name});
    }

    // =========================================================================
    // Time Management
    // =========================================================================

    /// Set current simulation time (called by transient solver)
    void set_current_time(Real t) { current_time_ = t; }

    /// Get current simulation time
    [[nodiscard]] Real current_time() const { return current_time_; }

    /// Check if circuit has time-varying sources
    [[nodiscard]] bool has_time_varying() const {
        for (const auto& dev : devices_) {
            if (std::holds_alternative<PWMVoltageSource>(dev) ||
                std::holds_alternative<SineVoltageSource>(dev) ||
                std::holds_alternative<PulseVoltageSource>(dev)) {
                return true;
            }
        }
        return false;
    }

    /// Bind a time-varying source output to a target switch-like component.
    /// Supported drivers: pulse and pwm voltage sources.
    void bind_switch_driver(std::string_view source_name, std::string_view target_switch_name) {
        if (source_name.empty()) {
            throw std::invalid_argument("Driver source name must not be empty");
        }
        if (target_switch_name.empty()) {
            throw std::invalid_argument("Target switch name must not be empty");
        }
        switch_driver_bindings_[std::string(source_name)] = std::string(target_switch_name);
    }

    // =========================================================================
    // Set Switch States
    // =========================================================================

    void set_switch_state(std::string_view name, bool closed) {
        const auto index = find_connection_index(name);
        if (!index.has_value()) {
            throw std::runtime_error("Switch not found: " + std::string{name});
        }

        auto& device = devices_[*index];
        if (auto* sw = std::get_if<IdealSwitch>(&device)) {
            sw->set_state(closed);
            if (*index < forced_switch_state_.size()) {
                forced_switch_state_[*index] = std::nullopt;
            }
            return;
        }

        if (std::holds_alternative<VoltageControlledSwitch>(device) ||
            std::holds_alternative<MOSFET>(device) ||
            std::holds_alternative<IGBT>(device)) {
            ensure_forced_switch_state_storage();
            forced_switch_state_[*index] = closed;
            return;
        }

        throw std::runtime_error("Switch not found: " + std::string{name});
    }

    // ---------------------------------------------------------------------
    // Circuit-level default switching mode (refactor-pwl-switching-engine,
    // Phase 5). The Simulator pushes `SimulationOptions.switching_mode` into
    // this slot at construction; segment-model and event-scan paths consume
    // it as the resolution circuit_default for `Auto`-mode devices.
    // ---------------------------------------------------------------------
    [[nodiscard]] SwitchingMode default_switching_mode() const noexcept {
        return default_switching_mode_;
    }
    void set_default_switching_mode(SwitchingMode mode) noexcept {
        default_switching_mode_ = mode;
    }

    /// Convenience: set `SwitchingMode` on every switching device in the
    /// circuit. Devices without a `set_switching_mode` method (passives,
    /// transformers) are silently skipped.
    void set_switching_mode_for_all(SwitchingMode mode) {
        for (auto& dev : devices_) {
            std::visit([&](auto& d) {
                using T = std::decay_t<decltype(d)>;
                if constexpr (std::is_same_v<T, IdealDiode> ||
                              std::is_same_v<T, IdealSwitch> ||
                              std::is_same_v<T, VoltageControlledSwitch> ||
                              std::is_same_v<T, MOSFET> ||
                              std::is_same_v<T, IGBT>) {
                    d.set_switching_mode(mode);
                }
            }, dev);
        }
    }

    /// Commit the PWL on/off state of any switching device (diode, switch,
    /// vcswitch, mosfet, igbt). Differs from `set_switch_state` in that it
    /// always writes to the device's own `pwl_state_` rather than the
    /// `forced_switch_state_` legacy override layer; this is the path the
    /// PWL segment engine uses to commit event transitions.
    void set_pwl_state(std::string_view name, bool on) {
        const auto index = find_connection_index(name);
        if (!index.has_value()) {
            throw std::runtime_error("PWL device not found: " + std::string{name});
        }
        auto& device = devices_[*index];
        bool committed = false;
        std::visit([&](auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            if constexpr (std::is_same_v<T, IdealDiode> ||
                          std::is_same_v<T, IdealSwitch> ||
                          std::is_same_v<T, VoltageControlledSwitch> ||
                          std::is_same_v<T, MOSFET> ||
                          std::is_same_v<T, IGBT>) {
                dev.commit_pwl_state(on);
                committed = true;
            }
        }, device);
        if (!committed) {
            throw std::runtime_error(
                "Device does not support PWL state: " + std::string{name});
        }
    }

    // =========================================================================
    // Set Timestep for Dynamic Elements
    // =========================================================================

    void set_timestep(Real dt) {
        if (!(std::isfinite(dt) && dt > 0.0)) {
            throw std::invalid_argument("Timestep must be finite and > 0");
        }

        for (auto& dev : devices_) {
            std::visit([dt](auto& d) {
                using T = std::decay_t<decltype(d)>;
                if constexpr (std::is_same_v<T, Capacitor> || std::is_same_v<T, Inductor>) {
                    d.set_timestep(dt);
                }
            }, dev);
        }
        timestep_ = dt;
    }

    [[nodiscard]] Real timestep() const { return timestep_; }

    // =========================================================================
    // Integration Method Control
    // =========================================================================

    void set_integration_method(Integrator method) {
        integration_method_ = method;
        integration_order_ = companion_order(method);
    }

    void set_integration_order(int order) {
        integration_order_ = std::clamp(order, 1, 2);
        integration_method_ = (integration_order_ == 1) ? Integrator::BDF1 : Integrator::Trapezoidal;
    }

    [[nodiscard]] int integration_order() const { return integration_order_; }
    [[nodiscard]] Integrator integration_method() const { return integration_method_; }

    // =========================================================================
    // Multi-Stage Integration Support (TR-BDF2 / SDIRK2 / RosenbrockW)
    // =========================================================================

    void clear_stage_context() {
        stage_context_ = StageContext{};
    }

    void capture_trbdf2_stage1(const Vector& x_stage) {
        ensure_stage_storage();
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Capacitor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Real v1 = (n1 >= 0) ? x_stage[n1] : 0.0;
                    Real v2 = (n2 >= 0) ? x_stage[n2] : 0.0;
                    stage_cap_v_[i] = v1 - v2;
                } else if constexpr (std::is_same_v<T, Inductor>) {
                    Index br = conn.branch_index;
                    stage_ind_i_[i] = (br >= 0) ? x_stage[br] : 0.0;
                }
            }, devices_[i]);
        }

        if (stage_context_.active) {
            clear_stage_context();
        }
    }

    void capture_sdirk_stage1(const Vector& x_stage, Real dt_total, Real a11) {
        ensure_stage_storage();
        const Real dt_stage = std::max(dt_total * a11, Real{1e-15});
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Capacitor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Real v1 = (n1 >= 0) ? x_stage[n1] : 0.0;
                    Real v2 = (n2 >= 0) ? x_stage[n2] : 0.0;
                    Real v = v1 - v2;
                    stage_cap_vdot_[i] = (v - dev.voltage_prev()) / dt_stage;
                } else if constexpr (std::is_same_v<T, Inductor>) {
                    Index br = conn.branch_index;
                    Real i_val = (br >= 0) ? x_stage[br] : 0.0;
                    stage_ind_idot_[i] = (i_val - dev.current_prev()) / dt_stage;
                }
            }, devices_[i]);
        }
    }

    void begin_trbdf2_stage2(Real h1, Real h2) {
        stage_context_.active = true;
        stage_context_.scheme = StageScheme::TRBDF2;
        stage_context_.method = Integrator::TRBDF2;
        stage_context_.stage = 2;
        stage_context_.h1 = h1;
        stage_context_.h2 = h2;
    }

    void begin_sdirk_stage2(Integrator method, Real dt_total, Real a11, Real a21, Real a22) {
        stage_context_.active = true;
        stage_context_.scheme = StageScheme::SDIRK2;
        stage_context_.method = method;
        stage_context_.stage = 2;
        stage_context_.dt = dt_total;
        stage_context_.a11 = a11;
        stage_context_.a21 = a21;
        stage_context_.a22 = a22;
    }

    // =========================================================================
    // Update Dynamic Element History (for transient)
    // =========================================================================

    /// Update capacitor/inductor history after a successful timestep
    /// @param x Current state vector
    /// @param initialize If true, this is initialization (t=0) - set i_prev=0 for caps
    void update_history(const Vector& x, bool initialize = false) {
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];

            std::visit([&](auto& dev) {
                using T = std::decay_t<decltype(dev)>;

                if constexpr (std::is_same_v<T, Capacitor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Real v1 = (n1 >= 0) ? x[n1] : 0.0;
                    Real v2 = (n2 >= 0) ? x[n2] : 0.0;
                    Real v = v1 - v2;

                    Real i = 0.0;
                    // Mirror the stamp's `!history_initialized()` first-step
                    // BDF1 path so the recorded `i_prev_` matches the value
                    // used to assemble the Jacobian. If we stamp BDF1 but
                    // record the trapezoidal-doubled current, the next step
                    // sees a stale `i_prev_` and propagates the oscillation.
                    const bool use_bdf1_first_step =
                        !dev.history_initialized() && !stage_context_.active;
                    if (initialize) {
                        // At t=0: set v_prev = v, i_prev = 0 (DC steady state)
                        i = 0.0;
                    } else if (stage_context_.active && stage_context_.scheme == StageScheme::TRBDF2) {
                        auto coeffs = TRBDF2Coeffs::bdf2_variable(stage_context_.h1, stage_context_.h2);
                        Real v_prev = dev.voltage_prev();
                        Real v_stage = stage_cap_v_[i];
                        i = dev.capacitance() * (coeffs.a2 * v + coeffs.a1 * v_stage + coeffs.a0 * v_prev);
                    } else if (stage_context_.active && stage_context_.scheme == StageScheme::SDIRK2) {
                        Real dt_total = std::max(stage_context_.dt, Real{1e-15});
                        Real a22 = std::max(stage_context_.a22, Real{1e-15});
                        Real v_prev = dev.voltage_prev();
                        Real vdot1 = stage_cap_vdot_[i];
                        Real vdot2 = (v - v_prev - dt_total * stage_context_.a21 * vdot1) / (a22 * dt_total);
                        i = dev.capacitance() * vdot2;
                    } else if (integration_order_ == 1 || use_bdf1_first_step) {
                        // Backward Euler: i = C * (v_n - v_{n-1}) / dt
                        Real g_eq = dev.capacitance() / timestep_;
                        i = g_eq * (v - dev.voltage_prev());
                    } else {
                        // Trapezoidal: i_n = g_eq*(v_n - v_{n-1}) - i_{n-1}
                        // I_eq = -g_eq*v_{n-1} - i_{n-1}
                        Real g_eq = 2.0 * dev.capacitance() / timestep_;
                        Real i_hist = -dev.current_prev() - g_eq * dev.voltage_prev();
                        i = g_eq * v + i_hist;
                    }

                    // Set current state and update history
                    dev.set_current_state(v, i);
                    dev.update_history();
                }
                else if constexpr (std::is_same_v<T, Inductor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Index br = conn.branch_index;
                    Real v1 = (n1 >= 0) ? x[n1] : 0.0;
                    Real v2 = (n2 >= 0) ? x[n2] : 0.0;
                    Real v = v1 - v2;
                    Real i = x[br];

                    if (initialize) {
                        // At t=0: use i from solution (branch current), set v_prev = 0
                        // This ensures first step uses a consistent history
                        v = 0.0;  // v_prev = 0 for inductor at t=0 (steady state: v_L = 0)
                    }

                    // Set current state and update history
                    dev.set_current_state(v, i);
                    dev.update_history();
                }
            }, devices_[i]);
        }
    }

    // =========================================================================
    // Matrix Assembly for DC Analysis (Linear)
    // =========================================================================

    /// Assemble G matrix and b vector for DC analysis
    void assemble_dc(SparseMatrix& G, Vector& b) const {
        const Index n = system_size();
        G.resize(n, n);
        G.setZero();
        b.resize(n);
        b.setZero();

        // Reserve space for triplets
        auto& triplets = dc_triplets_;
        triplets.clear();
        reserve_triplets(triplets, devices_.size() * 9);

        // Fast-path stamping for resistors (SoA)
        for (const auto& r : resistor_cache_) {
            stamp_conductance(r.g, r.n1, r.n2, triplets);
        }

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Resistor>) {
                    return;
                } else {
                    stamp_device_dc(dev, conn, i, triplets, b);
                }
            }, devices_[i]);
        }

        G.setFromTriplets(triplets.begin(), triplets.end());
    }

    // =========================================================================
    // Jacobian Assembly for Newton Iteration
    // =========================================================================

    /// Assemble Jacobian J and residual f for Newton iteration
    void assemble_jacobian(SparseMatrix& J, Vector& f, const Vector& x) const {
        const Index n = system_size();
        J.resize(n, n);
        J.setZero();
        f.resize(n);
        f.setZero();

        auto& triplets = jacobian_triplets_;
        triplets.clear();
        reserve_triplets(triplets, devices_.size() * 9);

        // Fast-path stamping for resistors (SoA)
        for (const auto& r : resistor_cache_) {
            Real v1 = (r.n1 >= 0) ? x[r.n1] : 0.0;
            Real v2 = (r.n2 >= 0) ? x[r.n2] : 0.0;
            Real i = r.g * (v1 - v2);
            stamp_conductance(r.g, r.n1, r.n2, triplets);
            if (r.n1 >= 0) f[r.n1] += i;
            if (r.n2 >= 0) f[r.n2] -= i;
        }

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Resistor>) {
                    return;
                } else {
                    stamp_device_jacobian(dev, conn, i, triplets, f, x);
                }
            }, devices_[i]);
        }

        stamp_coupled_inductor_terms(triplets, f, x);
        stamp_virtual_block_output_anchors(triplets);

        J.setFromTriplets(triplets.begin(), triplets.end());
    }

    /// Assemble residual f only (Jacobian-free)
    void assemble_residual(Vector& f, const Vector& x) const {
        const Index n = system_size();
        f.resize(n);
        f.setZero();

        // Fast-path residual for resistors (SoA)
        for (const auto& r : resistor_cache_) {
            Real v1 = (r.n1 >= 0) ? x[r.n1] : 0.0;
            Real v2 = (r.n2 >= 0) ? x[r.n2] : 0.0;
            Real i = r.g * (v1 - v2);
            if (r.n1 >= 0) f[r.n1] += i;
            if (r.n2 >= 0) f[r.n2] -= i;
        }

        struct NullTriplets {
            void emplace_back(Index, Index, Real) {}
        } null_triplets;

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Resistor>) {
                    return;
                } else {
                    stamp_device_jacobian(dev, conn, i, null_triplets, f, x);
                }
            }, devices_[i]);
        }

        stamp_coupled_inductor_terms(null_triplets, f, x);
        // (anchors don't contribute to the residual — they're pure
        // Jacobian diagonals — but the symmetric helper keeps the
        // assembly paths in lock-step so the M/N stamp can reuse it.)
    }

    /// Assemble residual for harmonic balance using direct time-derivatives
    void assemble_residual_hb(Vector& f, const Eigen::Ref<const Vector>& x, Real time,
                              std::span<const Real> dv_dt_nodes,
                              std::span<const Real> di_dt_branches) const {
        const Index n = system_size();
        f.resize(n);
        f.setZero();

        // Resistors (SoA)
        for (const auto& r : resistor_cache_) {
            Real v1 = (r.n1 >= 0) ? x[r.n1] : 0.0;
            Real v2 = (r.n2 >= 0) ? x[r.n2] : 0.0;
            Real i = r.g * (v1 - v2);
            if (r.n1 >= 0) f[r.n1] += i;
            if (r.n2 >= 0) f[r.n2] -= i;
        }

        struct NullTriplets {
            void emplace_back(Index, Index, Real) {}
        } null_triplets;

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Resistor>) {
                    return;
                } else if constexpr (std::is_same_v<T, Capacitor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Real dv1 = (n1 >= 0 && static_cast<std::size_t>(n1) < dv_dt_nodes.size())
                        ? dv_dt_nodes[n1] : 0.0;
                    Real dv2 = (n2 >= 0 && static_cast<std::size_t>(n2) < dv_dt_nodes.size())
                        ? dv_dt_nodes[n2] : 0.0;
                    Real i_c = dev.capacitance() * (dv1 - dv2);
                    if (n1 >= 0) f[n1] += i_c;
                    if (n2 >= 0) f[n2] -= i_c;
                } else if constexpr (std::is_same_v<T, Inductor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Index br = conn.branch_index;
                    Real v1 = (n1 >= 0) ? x[n1] : 0.0;
                    Real v2 = (n2 >= 0) ? x[n2] : 0.0;
                    Real i_br = (br >= 0) ? x[br] : 0.0;
                    Real di_dt = (br >= 0 && static_cast<std::size_t>(br) < di_dt_branches.size())
                        ? di_dt_branches[br] : 0.0;
                    Real L_eff = effective_inductance_for(conn.name, i_br, dev.inductance());
                    if (n1 >= 0) f[n1] += i_br;
                    if (n2 >= 0) f[n2] -= i_br;
                    if (br >= 0) f[br] += (v1 - v2 - L_eff * di_dt);
                } else if constexpr (std::is_same_v<T, VoltageSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    Index br = conn.branch_index;
                    Real v_src = dev.voltage();

                    Real vpos = (npos >= 0) ? x[npos] : 0.0;
                    Real vneg = (nneg >= 0) ? x[nneg] : 0.0;
                    if (br >= 0) f[br] += (vpos - vneg - v_src);

                    Real i_br = (br >= 0) ? x[br] : 0.0;
                    if (npos >= 0) f[npos] += i_br;
                    if (nneg >= 0) f[nneg] -= i_br;
                } else if constexpr (std::is_same_v<T, CurrentSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    Real i_src = dev.current();
                    if (npos >= 0) f[npos] -= i_src;
                    if (nneg >= 0) f[nneg] += i_src;
                } else if constexpr (std::is_same_v<T, PWMVoltageSource> ||
                                     std::is_same_v<T, SineVoltageSource> ||
                                     std::is_same_v<T, PulseVoltageSource>) {
                    Index npos = conn.nodes[0];
                    Index nneg = conn.nodes[1];
                    Index br = conn.branch_index;
                    Real v_src = dev.voltage_at(time);

                    Real vpos = (npos >= 0) ? x[npos] : 0.0;
                    Real vneg = (nneg >= 0) ? x[nneg] : 0.0;
                    if (br >= 0) f[br] += (vpos - vneg - v_src);

                    Real i_br = (br >= 0) ? x[br] : 0.0;
                    if (npos >= 0) f[npos] += i_br;
                    if (nneg >= 0) f[nneg] -= i_br;
                } else {
                    stamp_device_jacobian(dev, conn, i, null_triplets, f, x);
                }
            }, devices_[i]);
        }

        stamp_coupled_inductor_hb_terms(f, x, di_dt_branches);
    }

    /// Check if circuit has nonlinear devices
    [[nodiscard]] bool has_nonlinear() const {
        for (const auto& dev : devices_) {
            if (std::holds_alternative<IdealDiode>(dev) ||
                std::holds_alternative<MOSFET>(dev) ||
                std::holds_alternative<IGBT>(dev)) {
                return true;
            }
        }
        for (const auto& component : virtual_components_) {
            if (component.type == "saturable_inductor" ||
                component.type == "thyristor" ||
                component.type == "triac") {
                return true;
            }
        }
        return false;
    }

    // =========================================================================
    // PWL state-space assembly (refactor-pwl-switching-engine, Phase 2)
    // =========================================================================

    /// Compute the topology bitmask over all switching devices currently
    /// committed to a PWL state. Switching devices are visited in registration
    /// order; each contributes one bit (0 = off, 1 = on). Returns 0 if there
    /// are no switching devices. For circuits with more than 64 switching
    /// devices the upper bits are dropped — telemetry should flag this case.
    [[nodiscard]] std::uint64_t pwl_topology_bitmask() const {
        std::uint64_t bits = 0;
        std::size_t bit_idx = 0;
        for (const auto& dev : devices_) {
            std::visit([&](const auto& d) {
                using T = std::decay_t<decltype(d)>;
                if constexpr (std::is_same_v<T, IdealDiode> ||
                              std::is_same_v<T, IdealSwitch> ||
                              std::is_same_v<T, VoltageControlledSwitch> ||
                              std::is_same_v<T, MOSFET> ||
                              std::is_same_v<T, IGBT>) {
                    if (bit_idx < 64 && d.pwl_state()) {
                        bits |= (std::uint64_t{1} << bit_idx);
                    }
                    ++bit_idx;
                }
            }, dev);
        }
        return bits;
    }

    /// Count switching devices for diagnostics / cache sizing.
    [[nodiscard]] std::size_t pwl_switching_device_count() const {
        std::size_t count = 0;
        for (const auto& dev : devices_) {
            std::visit([&](const auto& d) {
                using T = std::decay_t<decltype(d)>;
                if constexpr (std::is_same_v<T, IdealDiode> ||
                              std::is_same_v<T, IdealSwitch> ||
                              std::is_same_v<T, VoltageControlledSwitch> ||
                              std::is_same_v<T, MOSFET> ||
                              std::is_same_v<T, IGBT>) {
                    ++count;
                }
            }, dev);
        }
        return count;
    }

    /// True iff every switching device in the circuit has SwitchingMode
    /// resolved to Ideal (either explicitly or via the supplied default).
    /// Devices that don't have a switching_mode() accessor (passives) are
    /// ignored. When the circuit contains zero switching devices the
    /// circuit is trivially eligible for the PWL path.
    [[nodiscard]] bool all_switching_devices_in_ideal_mode(
        SwitchingMode circuit_default = SwitchingMode::Behavioral) const {
        bool all_ideal = true;
        for (const auto& dev : devices_) {
            std::visit([&](const auto& d) {
                using T = std::decay_t<decltype(d)>;
                if constexpr (std::is_same_v<T, IdealDiode> ||
                              std::is_same_v<T, IdealSwitch> ||
                              std::is_same_v<T, VoltageControlledSwitch> ||
                              std::is_same_v<T, MOSFET> ||
                              std::is_same_v<T, IGBT>) {
                    const SwitchingMode resolved =
                        resolve_switching_mode(d.switching_mode(), circuit_default);
                    if (resolved != SwitchingMode::Ideal) {
                        all_ideal = false;
                    }
                }
            }, dev);
        }
        return all_ideal;
    }

    /// Assemble the continuous-time PWL state-space `M·ẋ + N·x = b(t)`.
    ///
    /// Sign / KCL conventions match `assemble_residual_hb()`:
    ///   * KCL residual at node n is `Σ (currents leaving n) = 0`.
    ///   * Voltage-source branch row: `vpos − vneg − v_src = 0`.
    ///   * Inductor branch row:       `v1   − v2   − L·di/dt = 0`.
    ///
    /// Reactive contributions (capacitors, inductors) populate `M`; resistive
    /// contributions (resistors, PWL switches in their committed state, and
    /// branch-coupling identity terms for V-sources / inductors) populate `N`;
    /// independent sources contribute to `b(t)`.
    ///
    /// Switching devices stamp **their committed PWL state**. The caller is
    /// responsible for committing states before invoking this method (e.g.
    /// after event scheduling). Behavioral-mode devices are stamped with
    /// their committed state too (g_on or g_off); their tanh / Shichman
    /// nonlinearity is *not* honored on this path — it is the segment
    /// model's responsibility to bail out via `all_switching_devices_in_ideal_mode()`
    /// when it encounters a Behavioral-mode device.
    ///
    /// Time-varying sources (PWM, sine, pulse) are evaluated at `time`.
    void assemble_state_space(SparseMatrix& M,
                              SparseMatrix& N,
                              Vector& b,
                              Real time) const {
        const Index n = system_size();
        M.resize(n, n);
        M.setZero();
        N.resize(n, n);
        N.setZero();
        b.resize(n);
        b.setZero();

        std::vector<Eigen::Triplet<Real>> m_triplets;
        std::vector<Eigen::Triplet<Real>> n_triplets;
        m_triplets.reserve(devices_.size() * 4);
        n_triplets.reserve(devices_.size() * 8);

        auto stamp_g = [&](Real g, Index a, Index c) {
            // Stamp +g symmetrically across (a, c) into N (KCL contribution).
            if (a >= 0) {
                n_triplets.emplace_back(a, a, g);
                if (c >= 0) n_triplets.emplace_back(a, c, -g);
            }
            if (c >= 0) {
                n_triplets.emplace_back(c, c, g);
                if (a >= 0) n_triplets.emplace_back(c, a, -g);
            }
        };

        auto stamp_capacitance = [&](Real C, Index a, Index c) {
            // KCL: i_C = C·d(va − vc)/dt. M entries:
            //   M[a, a]=+C, M[a, c]=−C, M[c, a]=−C, M[c, c]=+C
            if (a >= 0) {
                m_triplets.emplace_back(a, a, C);
                if (c >= 0) m_triplets.emplace_back(a, c, -C);
            }
            if (c >= 0) {
                m_triplets.emplace_back(c, c, C);
                if (a >= 0) m_triplets.emplace_back(c, a, -C);
            }
        };

        auto stamp_inductor_branch = [&](Real L, Index a, Index c, Index br) {
            // Branch eq:  v_a − v_c − L·di/dt = 0
            // KCL:        f[a] += i_br;  f[c] −= i_br
            if (br >= 0) {
                m_triplets.emplace_back(br, br, -L);
                if (a >= 0) {
                    n_triplets.emplace_back(br, a, 1.0);
                    n_triplets.emplace_back(a, br, 1.0);
                }
                if (c >= 0) {
                    n_triplets.emplace_back(br, c, -1.0);
                    n_triplets.emplace_back(c, br, -1.0);
                }
            }
        };

        auto stamp_voltage_source_eq = [&](Real V, Index npos, Index nneg, Index br) {
            // Branch eq:  v_pos − v_neg − V_src = 0  →  N[br,pos]=+1, N[br,neg]=−1,  b[br]=+V_src
            // KCL:        f[npos] += i_br, f[nneg] −= i_br
            if (br < 0) return;
            if (npos >= 0) {
                n_triplets.emplace_back(br, npos, 1.0);
                n_triplets.emplace_back(npos, br, 1.0);
            }
            if (nneg >= 0) {
                n_triplets.emplace_back(br, nneg, -1.0);
                n_triplets.emplace_back(nneg, br, -1.0);
            }
            b[br] = V;
        };

        auto stamp_current_source_eq = [&](Real I, Index npos, Index nneg) {
            // KCL: f[npos] −= I_src;  f[nneg] += I_src
            //  ⇒  b[npos] = +I, b[nneg] = −I (so N·x = b means residual = 0).
            if (npos >= 0) b[npos] += I;
            if (nneg >= 0) b[nneg] -= I;
        };

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                const auto& nodes = conn.nodes;

                if constexpr (std::is_same_v<T, Resistor>) {
                    if (nodes.size() >= 2) {
                        stamp_g(1.0 / dev.resistance(), nodes[0], nodes[1]);
                    }
                } else if constexpr (std::is_same_v<T, Capacitor>) {
                    if (nodes.size() >= 2) {
                        stamp_capacitance(dev.capacitance(), nodes[0], nodes[1]);
                    }
                } else if constexpr (std::is_same_v<T, Inductor>) {
                    if (nodes.size() >= 2 && conn.branch_index >= 0) {
                        const Real L_eff = effective_inductance_for(
                            conn.name, /*current=*/0.0, dev.inductance());
                        stamp_inductor_branch(L_eff, nodes[0], nodes[1], conn.branch_index);
                    }
                } else if constexpr (std::is_same_v<T, VoltageSource>) {
                    if (nodes.size() >= 2) {
                        stamp_voltage_source_eq(dev.voltage(), nodes[0], nodes[1],
                                                conn.branch_index);
                    }
                } else if constexpr (std::is_same_v<T, PWMVoltageSource> ||
                                     std::is_same_v<T, SineVoltageSource> ||
                                     std::is_same_v<T, PulseVoltageSource>) {
                    if (nodes.size() >= 2) {
                        stamp_voltage_source_eq(dev.voltage_at(time), nodes[0], nodes[1],
                                                conn.branch_index);
                    }
                } else if constexpr (std::is_same_v<T, CurrentSource>) {
                    if (nodes.size() >= 2) {
                        stamp_current_source_eq(dev.current(), nodes[0], nodes[1]);
                    }
                } else if constexpr (std::is_same_v<T, IdealDiode>) {
                    if (nodes.size() >= 2) {
                        const Real g = dev.pwl_state() ? dev.g_on() : dev.g_off();
                        stamp_g(g, nodes[0], nodes[1]);
                    }
                } else if constexpr (std::is_same_v<T, IdealSwitch>) {
                    if (nodes.size() >= 2) {
                        const Real g = dev.pwl_state() ? dev.g_on() : dev.g_off();
                        stamp_g(g, nodes[0], nodes[1]);
                    }
                } else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                    if (nodes.size() >= 3) {
                        const Real g = dev.pwl_state() ? dev.g_on() : dev.g_off();
                        // Conducting path is between t1 and t2 (control node is observe-only).
                        stamp_g(g, nodes[1], nodes[2]);
                    }
                } else if constexpr (std::is_same_v<T, MOSFET>) {
                    if (nodes.size() >= 3) {
                        const Real g = dev.pwl_state() ? dev.params().g_on
                                                       : dev.params().g_off;
                        // Drain–source path only (gate stamps deferred to the
                        // catalog-tier change once Coss/Ciss are modeled).
                        stamp_g(g, nodes[1], nodes[2]);
                    }
                } else if constexpr (std::is_same_v<T, IGBT>) {
                    if (nodes.size() >= 3) {
                        const Real g = dev.pwl_state() ? dev.params().g_on
                                                       : dev.params().g_off;
                        stamp_g(g, nodes[1], nodes[2]);
                    }
                } else if constexpr (std::is_same_v<T, Transformer>) {
                    // Coupled inductor support uses a separate stamp pass
                    // (stamp_coupled_inductor_terms) which currently produces a
                    // Newton-Jacobian rather than separated M/N matrices. For
                    // Phase 2 we mark transformers as ineligible for PWL by
                    // surfacing this through admissibility on the segment side
                    // (caller checks the device list).
                    (void)dev;
                }
            }, devices_[i]);
        }

        // Phase 3 of `add-frequency-domain-analysis`: AC perturbation hook.
        // When `set_ac_perturbation` has been called, overlay
        // `ε·cos(2π·f·t + φ)` on the named source's RHS contribution. The
        // cosine convention makes the input's phasor representation
        // `ε·e^{jφ}` — for φ = 0 the input phasor is purely real, which
        // matches the AC sweep B-column convention (real-valued B). FRA
        // can then divide its Goertzel output `Y(f)` by ε directly to
        // recover `H(jω)`, with no extra rotation needed to match AC
        // sweep's magnitude / phase reference. The perturbation is
        // additive on top of the source's existing DC / time-varying
        // value, so the Simulator can keep the original source unchanged
        // and inject a small-signal probe for FRA.
        if (ac_perturbation_active_) {
            const auto* conn = find_connection(ac_perturbation_source_);
            if (conn != nullptr) {
                const Real omega = 2.0 * std::numbers::pi_v<Real> *
                                   ac_perturbation_frequency_;
                const Real value = ac_perturbation_amplitude_ *
                                   std::cos(omega * time + ac_perturbation_phase_);
                if (conn->branch_index >= 0 && conn->branch_index < n) {
                    // Voltage source contribution at branch row.
                    b[conn->branch_index] += value;
                } else if (conn->nodes.size() >= 2) {
                    // Current source contribution at node rows
                    // (matches stamp_current_source_eq sign convention).
                    const Index npos = conn->nodes[0];
                    const Index nneg = conn->nodes[1];
                    if (npos >= 0 && npos < n) b[npos] += value;
                    if (nneg >= 0 && nneg < n) b[nneg] -= value;
                }
            }
        }

        // Anchor floating output nodes of analog-output control blocks
        // before finalizing N. See stamp_virtual_block_output_anchors
        // for rationale.
        stamp_virtual_block_output_anchors(n_triplets);

        M.setFromTriplets(m_triplets.begin(), m_triplets.end());
        N.setFromTriplets(n_triplets.begin(), n_triplets.end());
        M.makeCompressed();
        N.makeCompressed();
    }

    /// Phase 3 of `add-frequency-domain-analysis`: configure a small-signal
    /// AC perturbation overlaid on a named source's RHS during transient
    /// runs. Used by `Simulator::run_fra` to inject `ε·sin(2π·f·t + φ)` on
    /// top of a DC source for frequency-response analysis. The perturbation
    /// stays active until `clear_ac_perturbation` is called.
    void set_ac_perturbation(std::string source_name,
                              Real amplitude,
                              Real frequency,
                              Real phase = Real{0}) {
        ac_perturbation_source_    = std::move(source_name);
        ac_perturbation_amplitude_ = amplitude;
        ac_perturbation_frequency_ = frequency;
        ac_perturbation_phase_     = phase;
        ac_perturbation_active_    = true;
    }

    void clear_ac_perturbation() {
        ac_perturbation_active_ = false;
        ac_perturbation_source_.clear();
        ac_perturbation_amplitude_ = Real{0};
        ac_perturbation_frequency_ = Real{0};
        ac_perturbation_phase_     = Real{0};
    }

    [[nodiscard]] bool ac_perturbation_active() const { return ac_perturbation_active_; }

    /// True iff every device in the circuit has a PWL state-space stamp
    /// implemented by `assemble_state_space()`. Today this excludes
    /// transformers (algebraic ideal) and coupled inductors (mutual
    /// stamp lives in `stamp_coupled_inductor_terms`, which assemble_-
    /// state_space does not invoke). Circuits with either drop through
    /// to the DAE fallback path which assembles via `assemble_jacobian`
    /// — that path DOES call `stamp_coupled_inductor_terms` and so
    /// faithfully models `V_i = L_i·dI_i/dt + Σ_j M_ij·dI_j/dt`.
    [[nodiscard]] bool pwl_state_space_supports_all_devices() const {
        for (const auto& dev : devices_) {
            bool ok = true;
            std::visit([&](const auto& d) {
                using T = std::decay_t<decltype(d)>;
                (void)d;
                if constexpr (std::is_same_v<T, Transformer>) {
                    ok = false;
                }
            }, dev);
            if (!ok) return false;
        }
        // Virtual coupled-inductor wrapper: the segment-stepper's
        // assemble_state_space does not stamp the mutual term, so any
        // coupled-inductor circuit forces the DAE fallback path.
        for (const auto& component : virtual_components_) {
            if (component.type == "coupled_inductor") {
                return false;
            }
        }
        return true;
    }

    // =========================================================================
    // PWL event detection (refactor-pwl-switching-engine, Phase 4)
    // =========================================================================
    //
    // The PWL segment engine integrates a piecewise-linear DAE assuming the
    // committed switching states stay valid throughout the step. After each
    // accepted step the simulator must verify that assumption: was any
    // device's `should_commute()` predicate triggered by the step's
    // resulting state vector? If yes, the device's `pwl_state_` is flipped
    // and the next step rebuilds the segment model under the new topology.
    //
    // Phase 4 lands a *first-order* event scheduler: events are recognized
    // at step boundaries and committed for the next step. Bisection-to-event
    // (full second-order accuracy at the event time) is a follow-up work
    // captured in tasks.md (4.4).

    /// One pending PWL state transition discovered by `scan_pwl_commutations`.
    /// `device_index` is the variant slot in `devices_`; `device_name` mirrors
    /// the connection registry; `new_state` is the post-commutation pwl_state.
    struct PwlCommutation {
        std::size_t device_index = 0;
        std::string device_name;
        bool new_state = false;
    };

    /// Walk every PWL-eligible device, build its `PwlEventContext` from the
    /// supplied state vector, and collect the devices that report
    /// `should_commute() == true`. Devices not resolved to
    /// `SwitchingMode::Ideal` (under `circuit_default`) are skipped — only
    /// the PWL segment path drives commutations through this channel; the
    /// legacy `forced_switch_state_` mechanism still owns Behavioral-mode
    /// switches.
    [[nodiscard]] std::vector<PwlCommutation> scan_pwl_commutations(
        const Vector& x,
        SwitchingMode circuit_default = SwitchingMode::Behavioral) const {
        std::vector<PwlCommutation> events;
        events.reserve(devices_.size());

        auto safe_voltage = [&](Index node) -> Real {
            if (node < 0 || node >= x.size()) return Real{0.0};
            return x[node];
        };

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, IdealDiode> ||
                              std::is_same_v<T, VoltageControlledSwitch> ||
                              std::is_same_v<T, MOSFET> ||
                              std::is_same_v<T, IGBT>) {
                    if (resolve_switching_mode(dev.switching_mode(),
                                               circuit_default) !=
                        SwitchingMode::Ideal) {
                        return;
                    }

                    PwlEventContext ctx;
                    ctx.event_hysteresis = dev.event_hysteresis();

                    if constexpr (std::is_same_v<T, IdealDiode>) {
                        if (conn.nodes.size() < 2) return;
                        const Real v_an = safe_voltage(conn.nodes[0]);
                        const Real v_ca = safe_voltage(conn.nodes[1]);
                        ctx.voltage = v_an - v_ca;
                        const Real g = dev.pwl_state() ? dev.g_on() : dev.g_off();
                        ctx.current = g * ctx.voltage;
                    } else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                        if (conn.nodes.size() < 3) return;
                        ctx.control_voltage = safe_voltage(conn.nodes[0]);
                        ctx.voltage = safe_voltage(conn.nodes[1]) -
                                      safe_voltage(conn.nodes[2]);
                        const Real g = dev.pwl_state() ? dev.g_on() : dev.g_off();
                        ctx.current = g * ctx.voltage;
                    } else if constexpr (std::is_same_v<T, MOSFET>) {
                        if (conn.nodes.size() < 3) return;
                        const Real v_g = safe_voltage(conn.nodes[0]);
                        const Real v_d = safe_voltage(conn.nodes[1]);
                        const Real v_s = safe_voltage(conn.nodes[2]);
                        ctx.control_voltage = v_g - v_s;
                        ctx.voltage = v_d - v_s;
                        const Real g = dev.pwl_state() ? dev.params().g_on
                                                       : dev.params().g_off;
                        ctx.current = g * ctx.voltage;
                    } else if constexpr (std::is_same_v<T, IGBT>) {
                        if (conn.nodes.size() < 3) return;
                        const Real v_g = safe_voltage(conn.nodes[0]);
                        const Real v_c = safe_voltage(conn.nodes[1]);
                        const Real v_e = safe_voltage(conn.nodes[2]);
                        ctx.control_voltage = v_g - v_e;
                        ctx.voltage = v_c - v_e;
                        const Real g = dev.pwl_state() ? dev.params().g_on
                                                       : dev.params().g_off;
                        ctx.current = g * ctx.voltage;
                    }

                    if (dev.should_commute(ctx)) {
                        PwlCommutation evt;
                        evt.device_index = i;
                        evt.device_name = conn.name;
                        evt.new_state = !dev.pwl_state();
                        events.push_back(std::move(evt));
                    }
                }
            }, devices_[i]);
        }
        return events;
    }

    /// Apply a list of pending commutations (typically from
    /// `scan_pwl_commutations`) by writing `new_state` into each device's
    /// `pwl_state_`. The Simulator's event scheduler is responsible for
    /// bisection / bookkeeping; this method is a pure mutator.
    void commit_pwl_commutations(const std::vector<PwlCommutation>& events) {
        for (const auto& e : events) {
            if (e.device_index >= devices_.size()) continue;
            std::visit([&](auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, IdealDiode> ||
                              std::is_same_v<T, IdealSwitch> ||
                              std::is_same_v<T, VoltageControlledSwitch> ||
                              std::is_same_v<T, MOSFET> ||
                              std::is_same_v<T, IGBT>) {
                    dev.commit_pwl_state(e.new_state);
                }
            }, devices_[e.device_index]);
        }
    }

    // =========================================================================
    // Accessors for Circuit Conversion (AC Analysis)
    // =========================================================================

    /// Get all devices (for conversion to IR circuit)
    [[nodiscard]] const std::vector<DeviceVariant>& devices() const { return devices_; }
    /// Mutable accessor used by the AD validation layer (Phase 4 of
    /// `add-automatic-differentiation`), where stamps may mutate `pwl_state_`.
    [[nodiscard]] std::vector<DeviceVariant>& devices_mutable() { return devices_; }

    /// Get all connections (for conversion to IR circuit)
    [[nodiscard]] const std::vector<DeviceConnection>& connections() const { return connections_; }

    /// Update per-device electrothermal electrical scaling for the next accepted segment.
    void set_device_temperature_scale(std::size_t device_index, Real scale) {
        ensure_device_temperature_scale_storage();
        if (device_index >= device_temperature_scale_.size()) {
            return;
        }
        device_temperature_scale_[device_index] = std::clamp(scale, Real{0.05}, Real{4.0});
    }

    /// Update all per-device electrothermal electrical scales in one commit.
    void set_device_temperature_scales(const std::vector<Real>& scales) {
        ensure_device_temperature_scale_storage();
        const std::size_t n = device_temperature_scale_.size();
        for (std::size_t i = 0; i < n; ++i) {
            const Real value = i < scales.size() ? scales[i] : 1.0;
            device_temperature_scale_[i] = std::clamp(value, Real{0.05}, Real{4.0});
        }
    }

    /// Reset all electrothermal electrical scaling to unity.
    void reset_device_temperature_scales() {
        ensure_device_temperature_scale_storage();
        std::fill(device_temperature_scale_.begin(), device_temperature_scale_.end(), 1.0);
    }

    /// Get current electrothermal electrical scale for a device index.
    [[nodiscard]] Real device_temperature_scale(std::size_t device_index) const {
        if (device_index >= device_temperature_scale_.size()) {
            return 1.0;
        }
        return std::clamp(device_temperature_scale_[device_index], Real{0.05}, Real{4.0});
    }

private:
    enum class StageScheme {
        None,
        TRBDF2,
        SDIRK2
    };

    struct StageContext {
        bool active = false;
        StageScheme scheme = StageScheme::None;
        Integrator method = Integrator::Trapezoidal;
        int stage = 0;
        Real dt = 0.0;
        Real h1 = 0.0;
        Real h2 = 0.0;
        Real a11 = 0.0;
        Real a21 = 0.0;
        Real a22 = 0.0;
    };

    struct ResistorStamp {
        Index n1 = -1;
        Index n2 = -1;
        Real g = 0.0;
    };

    [[nodiscard]] static constexpr char ascii_lower(char value) noexcept {
        return (value >= 'A' && value <= 'Z')
            ? static_cast<char>(value - 'A' + 'a')
            : value;
    }

    [[nodiscard]] static constexpr bool ascii_iequals(
        std::string_view lhs,
        std::string_view rhs) noexcept {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            if (ascii_lower(lhs[i]) != ascii_lower(rhs[i])) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static constexpr bool is_ground_name(std::string_view name) noexcept {
        return name == "0" || ascii_iequals(name, "gnd");
    }

    void ensure_stage_storage() {
        if (stage_cap_v_.size() != devices_.size()) {
            stage_cap_v_.assign(devices_.size(), 0.0);
            stage_ind_i_.assign(devices_.size(), 0.0);
            stage_cap_vdot_.assign(devices_.size(), 0.0);
            stage_ind_idot_.assign(devices_.size(), 0.0);
        }
    }

    void ensure_device_temperature_scale_storage() {
        if (device_temperature_scale_.size() != devices_.size()) {
            device_temperature_scale_.assign(devices_.size(), 1.0);
        }
    }

    static void reserve_triplets(std::vector<Eigen::Triplet<Real>>& triplets, std::size_t estimate) {
        if (triplets.capacity() < estimate) {
            triplets.reserve(estimate);
        }
    }

    void register_connection_name(std::size_t index) {
        if (index >= connections_.size()) {
            return;
        }
        connection_name_to_index_.try_emplace(connections_[index].name, index);
        if (forced_switch_state_.size() < connections_.size()) {
            forced_switch_state_.resize(connections_.size());
        }
    }

    void ensure_forced_switch_state_storage() {
        if (forced_switch_state_.size() != devices_.size()) {
            forced_switch_state_.resize(devices_.size());
        }
    }

    [[nodiscard]] std::optional<bool> forced_switch_state(std::size_t device_index) const {
        if (device_index >= forced_switch_state_.size()) {
            return std::nullopt;
        }
        return forced_switch_state_[device_index];
    }

public:
    /// Public lookups by device name. These are const-correct utilities used
    /// across the simulator (DC OP, AC sweep, post-processing). They're
    /// exposed publicly because external services (e.g. AC sweep's
    /// perturbation-source resolution) need to dispatch on device type
    /// without re-walking the device list. Mutation paths stay private.
    [[nodiscard]] const DeviceConnection* find_connection(std::string_view name) const {
        if (const auto index = find_connection_index(name); index.has_value()) {
            return &connections_[*index];
        }
        return nullptr;
    }

    [[nodiscard]] std::optional<std::size_t> find_connection_index(std::string_view name) const {
        if (const auto it = connection_name_to_index_.find(name);
            it != connection_name_to_index_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    template<typename Device>
    [[nodiscard]] const Device* find_device(std::string_view name) const {
        if (const auto index = find_connection_index(name); index.has_value()) {
            return std::get_if<Device>(&devices_[*index]);
        }
        return nullptr;
    }

private:
    template<typename Device>
    [[nodiscard]] Device* find_device(std::string_view name) {
        if (const auto index = find_connection_index(name); index.has_value()) {
            return std::get_if<Device>(&devices_[*index]);
        }
        return nullptr;
    }

    [[nodiscard]] static Real get_param_value(
        const std::unordered_map<std::string, Real>& params,
        const std::string& key,
        Real fallback) {
        const auto it = params.find(key);
        return (it == params.end()) ? fallback : it->second;
    }

    [[nodiscard]] static std::string normalize_mode_token(std::string token) {
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
            if (c == '-' || c == ' ') return '_';
            return static_cast<char>(std::tolower(c));
        });
        return token;
    }

    [[nodiscard]] static std::optional<Real> parse_yaml_real(const YAML::Node& node) {
        if (!node || node.IsNull()) return std::nullopt;
        try {
            return node.as<Real>();
        } catch (...) {
            try {
                return static_cast<Real>(std::stod(node.as<std::string>()));
            } catch (...) {
                return std::nullopt;
            }
        }
    }

    [[nodiscard]] static std::vector<Real> parse_yaml_real_sequence(const YAML::Node& node) {
        std::vector<Real> values;
        if (!node || node.IsNull()) return values;
        if (node.IsSequence()) {
            values.reserve(node.size());
            for (const auto& item : node) {
                if (const auto value = parse_yaml_real(item); value.has_value()) {
                    values.push_back(*value);
                }
            }
        } else if (const auto value = parse_yaml_real(node); value.has_value()) {
            values.push_back(*value);
        }
        return values;
    }

    [[nodiscard]] static std::vector<std::pair<Real, Real>> normalize_lookup_samples(
        std::vector<std::pair<Real, Real>> samples) {
        if (samples.empty()) return samples;
        std::stable_sort(samples.begin(), samples.end(),
                         [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        std::vector<std::pair<Real, Real>> merged;
        merged.reserve(samples.size());
        for (const auto& sample : samples) {
            if (!merged.empty() && std::abs(merged.back().first - sample.first) < 1e-15) {
                merged.back().second = sample.second;
            } else {
                merged.push_back(sample);
            }
        }
        return merged;
    }

    [[nodiscard]] std::vector<std::pair<Real, Real>> parse_lookup_samples(
        const VirtualComponent& component) const {
        std::vector<std::pair<Real, Real>> samples;

        auto parse_metadata_node = [&](const std::string& key) -> YAML::Node {
            const auto it = component.metadata.find(key);
            if (it == component.metadata.end()) return YAML::Node();
            try {
                return YAML::Load(it->second);
            } catch (...) {
                return YAML::Node();
            }
        };

        const YAML::Node x_node = parse_metadata_node("x");
        const YAML::Node y_node = parse_metadata_node("y");
        if (x_node && y_node) {
            const auto x_vals = parse_yaml_real_sequence(x_node);
            const auto y_vals = parse_yaml_real_sequence(y_node);
            const std::size_t n = std::min(x_vals.size(), y_vals.size());
            samples.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                samples.emplace_back(x_vals[i], y_vals[i]);
            }
        }

        const YAML::Node table_node = parse_metadata_node("table");
        if (table_node && table_node.IsSequence()) {
            for (const auto& entry : table_node) {
                if (entry.IsSequence() && entry.size() >= 2) {
                    const auto x = parse_yaml_real(entry[0]);
                    const auto y = parse_yaml_real(entry[1]);
                    if (x.has_value() && y.has_value()) {
                        samples.emplace_back(*x, *y);
                    }
                } else if (entry.IsMap()) {
                    const auto x = parse_yaml_real(entry["x"]);
                    const auto y = parse_yaml_real(entry["y"]);
                    if (x.has_value() && y.has_value()) {
                        samples.emplace_back(*x, *y);
                    }
                }
            }
        }

        const YAML::Node mapping_node = parse_metadata_node("mapping");
        if (mapping_node && mapping_node.IsMap()) {
            for (const auto& entry : mapping_node) {
                const auto x = parse_yaml_real(entry.first);
                const auto y = parse_yaml_real(entry.second);
                if (x.has_value() && y.has_value()) {
                    samples.emplace_back(*x, *y);
                }
            }
        }

        return normalize_lookup_samples(std::move(samples));
    }

    [[nodiscard]] const std::vector<std::pair<Real, Real>>& lookup_table_samples(
        const VirtualComponent& component) const {
        if (const auto it = lookup_table_cache_.find(component.name);
            it != lookup_table_cache_.end()) {
            return it->second;
        }
        return lookup_table_cache_
            .emplace(component.name, parse_lookup_samples(component))
            .first->second;
    }

    [[nodiscard]] Real interpolate_lookup(
        const VirtualComponent& component,
        const std::vector<std::pair<Real, Real>>& samples,
        Real input) const {
        if (samples.empty()) return input;
        if (samples.size() == 1) return samples.front().second;
        if (input <= samples.front().first) return samples.front().second;
        if (input >= samples.back().first) return samples.back().second;

        const auto upper = std::lower_bound(
            samples.begin(), samples.end(), input,
            [](const std::pair<Real, Real>& sample, Real value) {
                return sample.first < value;
            });
        if (upper == samples.begin()) return upper->second;
        if (upper == samples.end()) return samples.back().second;

        const auto& p1 = *upper;
        const auto& p0 = *(upper - 1);

        std::string mode = "linear";
        if (const auto it = component.metadata.find("mode"); it != component.metadata.end()) {
            mode = normalize_mode_token(it->second);
        }
        if (mode == "hold" || mode == "step") {
            return p0.second;
        }
        if (mode == "nearest") {
            const Real d0 = std::abs(input - p0.first);
            const Real d1 = std::abs(input - p1.first);
            return (d0 <= d1) ? p0.second : p1.second;
        }

        const Real span = std::max<Real>(p1.first - p0.first, 1e-15);
        const Real alpha = std::clamp((input - p0.first) / span, 0.0, 1.0);
        return p0.second + alpha * (p1.second - p0.second);
    }

    [[nodiscard]] const std::vector<Real>& transfer_coefficients(
        const VirtualComponent& component,
        const std::string& key) const {
        const std::string cache_key = component.name + ":" + key;
        if (const auto it = transfer_coeff_cache_.find(cache_key);
            it != transfer_coeff_cache_.end()) {
            return it->second;
        }

        std::vector<Real> coeffs;
        if (const auto it = component.metadata.find(key); it != component.metadata.end()) {
            try {
                YAML::Node node = YAML::Load(it->second);
                coeffs = parse_yaml_real_sequence(node);
            } catch (...) {
                coeffs.clear();
            }
        }

        return transfer_coeff_cache_.emplace(cache_key, std::move(coeffs)).first->second;
    }

    [[nodiscard]] Real saturable_effective_inductance(
        const VirtualComponent& component,
        Real i_est,
        Real fallback_l) const {
        const Real l_unsat = std::max<Real>(
            std::abs(get_param_value(component.numeric_params, "inductance", fallback_l)), 1e-12);
        const Real i_sat = std::max<Real>(
            std::abs(get_param_value(component.numeric_params, "saturation_current", 1.0)), 1e-12);
        const Real l_sat_raw = std::abs(get_param_value(component.numeric_params, "saturation_inductance",
                                                        l_unsat * 0.2));
        const Real l_sat = std::clamp(l_sat_raw, 1e-12, l_unsat);
        const Real exponent = std::clamp(
            get_param_value(component.numeric_params, "saturation_exponent", 2.0), 1.0, 8.0);
        const Real ratio = std::pow(std::abs(i_est) / i_sat, exponent);
        const Real l_eff = l_sat + (l_unsat - l_sat) / (1.0 + ratio);
        return std::max<Real>(l_eff, 1e-12);
    }

    [[nodiscard]] Real effective_inductance_for(
        const std::string& connection_name,
        Real i_est,
        Real fallback_l) const {
        for (const auto& component : virtual_components_) {
            if (component.type != "saturable_inductor") {
                continue;
            }
            const auto target_it = component.metadata.find("target_component");
            const std::string target = (target_it == component.metadata.end())
                ? component.name : target_it->second;
            if (target != connection_name) {
                continue;
            }
            return saturable_effective_inductance(component, i_est, fallback_l);
        }
        return fallback_l;
    }

    /// Anchor virtual control block output nodes to ground via a
    /// tiny conductance (g_anchor = 1 nS, i.e. ~1 GΩ pull-down).
    ///
    /// Control blocks declare nodes for ergonomic wiring (e.g.
    /// pi_controller takes [in_pos, in_neg, output];
    /// lookup_table / gain / integrator / etc. take [in, out]).
    /// But virtual blocks only emit channel values — they never
    /// stamp anything electrically into the output node. With
    /// nothing else touching that node, the MNA matrix has an
    /// unconstrained row/column and the DC OP / Newton solve
    /// fails with "Max iterations reached" or "DC operating
    /// point failed", with no indication that the fix was to
    /// hand-add a high-impedance pull-down.
    ///
    /// We anchor:
    ///   - node[2] of pi_controller / pid_controller / op_amp
    ///     (the "output" of a 3-node block)
    ///   - node[1] of all "single-input" control blocks where
    ///     in1 is conventionally a reference and the second node
    ///     is read but rarely driven (lookup_table, gain,
    ///     integrator, differentiator, limiter, rate_limiter,
    ///     sample_hold, transfer_function, delay_block,
    ///     hysteresis, comparator)
    ///
    /// Truly differential blocks (sum, subtraction, math_block,
    /// signal_mux, signal_demux) are NOT anchored — both their
    /// nodes are real inputs that the user is expected to drive.
    ///
    /// pwm_generator is also excluded: its "output" goes to a
    /// switch's PWL state via `target_component`, never to a
    /// node, so anchoring would be meaningless.
    ///
    /// At 1 nS the bias current through the anchor (e.g. ~10 nA
    /// at 10 V) is well below the gmin/iref scales the solver
    /// already tolerates, so no observable circuit behavior
    /// changes — verified bit-for-bit against the existing
    /// closed-loop baselines.
    template<typename Triplets>
    void stamp_virtual_block_output_anchors(Triplets& triplets) const {
        if (virtual_components_.empty()) return;
        constexpr Real g_anchor = 1.0e-9;  // 1 GΩ to ground
        // Single-input blocks whose 2nd node is read but not
        // user-driven by typical usage.
        static const std::unordered_set<std::string> kAnchorNode1 = {
            "lookup_table", "gain", "integrator", "differentiator",
            "limiter", "rate_limiter", "sample_hold",
            "transfer_function", "delay_block",
            "hysteresis", "comparator",
        };
        // 3-node analog-output blocks; node[2] is the synthesized
        // output and never electrically driven.
        static const std::unordered_set<std::string> kAnchorNode2 = {
            "pi_controller", "pid_controller", "op_amp",
        };

        auto anchor = [&](Index node) {
            if (node < 0 || node >= system_size()) return;
            triplets.emplace_back(node, node, g_anchor);
        };

        for (const auto& component : virtual_components_) {
            if (kAnchorNode2.find(component.type) != kAnchorNode2.end()) {
                if (component.nodes.size() >= 3) anchor(component.nodes[2]);
            } else if (kAnchorNode1.find(component.type) != kAnchorNode1.end()) {
                if (component.nodes.size() >= 2) anchor(component.nodes[1]);
            }
        }
    }

    template<typename Triplets>
    void stamp_coupled_inductor_terms(Triplets& triplets, Vector& f, const Vector& x) const {
        if (virtual_components_.empty()) return;

        for (const auto& component : virtual_components_) {
            if (component.type != "coupled_inductor") {
                continue;
            }

            const auto t1_it = component.metadata.find("target_component_1");
            const auto t2_it = component.metadata.find("target_component_2");
            if (t1_it == component.metadata.end() || t2_it == component.metadata.end()) {
                continue;
            }

            const auto idx1 = find_connection_index(t1_it->second);
            const auto idx2 = find_connection_index(t2_it->second);
            if (!idx1.has_value() || !idx2.has_value()) continue;

            const auto& conn1 = connections_[*idx1];
            const auto& conn2 = connections_[*idx2];
            const Index br1 = conn1.branch_index;
            const Index br2 = conn2.branch_index;
            if (br1 < 0 || br2 < 0 || br1 >= system_size() || br2 >= system_size()) {
                continue;
            }

            const auto* ind1 = std::get_if<Inductor>(&devices_[*idx1]);
            const auto* ind2 = std::get_if<Inductor>(&devices_[*idx2]);
            if (!ind1 || !ind2) continue;

            const Real i1 = x[br1];
            const Real i2 = x[br2];
            const Real l1 = effective_inductance_for(conn1.name, i1, ind1->inductance());
            const Real l2 = effective_inductance_for(conn2.name, i2, ind2->inductance());
            const Real k = std::clamp(
                get_param_value(component.numeric_params, "coupling",
                                get_param_value(component.numeric_params, "k", 0.98)),
                -0.999, 0.999);
            const Real mutual = k * std::sqrt(std::max<Real>(l1 * l2, 0.0));
            if (std::abs(mutual) <= 1e-18) {
                continue;
            }

            Real coeff_m = 0.0;
            Real history_1 = 0.0;
            Real history_2 = 0.0;

            if (stage_context_.active && stage_context_.scheme == StageScheme::TRBDF2 &&
                *idx1 < stage_ind_i_.size() && *idx2 < stage_ind_i_.size()) {
                auto coeffs = TRBDF2Coeffs::bdf2_variable(stage_context_.h1, stage_context_.h2);
                coeff_m = mutual * coeffs.a2;
                history_1 = mutual * (coeffs.a1 * stage_ind_i_[*idx2] + coeffs.a0 * ind2->current_prev());
                history_2 = mutual * (coeffs.a1 * stage_ind_i_[*idx1] + coeffs.a0 * ind1->current_prev());
            } else if (stage_context_.active && stage_context_.scheme == StageScheme::SDIRK2 &&
                       *idx1 < stage_ind_idot_.size() && *idx2 < stage_ind_idot_.size()) {
                Real dt_total = std::max(stage_context_.dt, Real{1e-15});
                Real a22 = std::max(stage_context_.a22, Real{1e-15});
                coeff_m = mutual / (a22 * dt_total);
                history_1 = -coeff_m * (ind2->current_prev() + dt_total * stage_context_.a21 * stage_ind_idot_[*idx2]);
                history_2 = -coeff_m * (ind1->current_prev() + dt_total * stage_context_.a21 * stage_ind_idot_[*idx1]);
            } else if (integration_order_ == 1) {
                coeff_m = mutual / timestep_;
                history_1 = coeff_m * ind2->current_prev();
                history_2 = coeff_m * ind1->current_prev();
            } else {
                coeff_m = 2.0 * mutual / timestep_;
                history_1 = coeff_m * ind2->current_prev();
                history_2 = coeff_m * ind1->current_prev();
            }

            triplets.emplace_back(br1, br2, -coeff_m);
            triplets.emplace_back(br2, br1, -coeff_m);
            f[br1] += (-coeff_m * i2 + history_1);
            f[br2] += (-coeff_m * i1 + history_2);
        }
    }

    void stamp_coupled_inductor_hb_terms(
        Vector& f,
        const Eigen::Ref<const Vector>& x,
        std::span<const Real> di_dt_branches) const {
        if (virtual_components_.empty()) return;

        for (const auto& component : virtual_components_) {
            if (component.type != "coupled_inductor") {
                continue;
            }

            const auto t1_it = component.metadata.find("target_component_1");
            const auto t2_it = component.metadata.find("target_component_2");
            if (t1_it == component.metadata.end() || t2_it == component.metadata.end()) {
                continue;
            }
            const auto idx1 = find_connection_index(t1_it->second);
            const auto idx2 = find_connection_index(t2_it->second);
            if (!idx1.has_value() || !idx2.has_value()) continue;

            const auto& conn1 = connections_[*idx1];
            const auto& conn2 = connections_[*idx2];
            const Index br1 = conn1.branch_index;
            const Index br2 = conn2.branch_index;
            if (br1 < 0 || br2 < 0) continue;
            if (static_cast<std::size_t>(br1) >= di_dt_branches.size() ||
                static_cast<std::size_t>(br2) >= di_dt_branches.size()) {
                continue;
            }

            const auto* ind1 = std::get_if<Inductor>(&devices_[*idx1]);
            const auto* ind2 = std::get_if<Inductor>(&devices_[*idx2]);
            if (!ind1 || !ind2) continue;

            const Real l1 = effective_inductance_for(conn1.name, x[br1], ind1->inductance());
            const Real l2 = effective_inductance_for(conn2.name, x[br2], ind2->inductance());
            const Real k = std::clamp(
                get_param_value(component.numeric_params, "coupling",
                                get_param_value(component.numeric_params, "k", 0.98)),
                -0.999, 0.999);
            const Real mutual = k * std::sqrt(std::max<Real>(l1 * l2, 0.0));
            if (std::abs(mutual) <= 1e-18) continue;

            f[br1] -= mutual * di_dt_branches[br2];
            f[br2] -= mutual * di_dt_branches[br1];
        }
    }

    static constexpr int companion_order(Integrator method) {
        switch (method) {
            case Integrator::BDF1:
            case Integrator::RosenbrockW:
            case Integrator::SDIRK2:
                return 1;
            case Integrator::Trapezoidal:
            case Integrator::BDF2:
            case Integrator::TRBDF2:
            case Integrator::Gear:
            case Integrator::BDF3:
            case Integrator::BDF4:
            case Integrator::BDF5:
            default:
                return 2;
        }
    }

    std::vector<DeviceVariant> devices_;
    std::vector<VirtualComponent> virtual_components_;
    std::unordered_map<std::string, Real> virtual_signal_state_;
    std::unordered_map<std::string, Real> virtual_last_input_;
    std::unordered_map<std::string, Real> virtual_last_time_;
    std::unordered_map<std::string, bool> virtual_binary_state_;
    std::unordered_map<std::string, std::deque<std::pair<Real, Real>>> virtual_time_history_;
    mutable std::unordered_map<std::string, std::vector<std::pair<Real, Real>>> lookup_table_cache_;
    mutable std::unordered_map<std::string, std::vector<Real>> transfer_coeff_cache_;
    std::unordered_map<std::string, std::deque<Real>> transfer_input_history_;
    std::unordered_map<std::string, std::deque<Real>> transfer_output_history_;
    std::vector<DeviceConnection> connections_;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, TransparentStringEqual>
        connection_name_to_index_;
    std::vector<Real> device_temperature_scale_;
    std::vector<std::optional<bool>> forced_switch_state_;
    std::unordered_map<std::string, std::string, TransparentStringHash, TransparentStringEqual>
        switch_driver_bindings_;
    std::vector<ResistorStamp> resistor_cache_;
    std::unordered_map<std::string, Index, TransparentStringHash, TransparentStringEqual> node_map_;
    std::vector<std::string> node_names_;
    Index num_branches_ = 0;
    Real timestep_ = 1e-6;
    Integrator integration_method_ = Integrator::Trapezoidal;
    int integration_order_ = 2;  // companion-model order (1 = BE, 2 = TR)
    Real current_time_ = 0.0;
    SwitchingMode default_switching_mode_ = SwitchingMode::Auto;
    inline static const std::string ground_name_ = "0";

    // Phase 3 of `add-frequency-domain-analysis`: AC perturbation state.
    // When `set_ac_perturbation` is called, `assemble_state_space` overlays
    // `ε·sin(2π·f·t + φ)` on the named source's RHS — used by FRA to
    // inject a small-signal probe on top of an otherwise-DC source.
    bool ac_perturbation_active_ = false;
    std::string ac_perturbation_source_;
    Real ac_perturbation_amplitude_ = 0.0;
    Real ac_perturbation_frequency_ = 0.0;
    Real ac_perturbation_phase_     = 0.0;

    StageContext stage_context_{};
    std::vector<Real> stage_cap_v_;
    std::vector<Real> stage_ind_i_;
    std::vector<Real> stage_cap_vdot_;
    std::vector<Real> stage_ind_idot_;

    mutable std::vector<Eigen::Triplet<Real>> dc_triplets_;
    mutable std::vector<Eigen::Triplet<Real>> jacobian_triplets_;

    // =========================================================================
    // DC Stamping Helpers
    // =========================================================================

    template<typename Device>
    void stamp_device_dc(const Device& dev, const DeviceConnection& conn,
                         std::size_t device_index,
                         std::vector<Eigen::Triplet<Real>>& triplets, Vector& b) const {
        using T = std::decay_t<Device>;

        if constexpr (std::is_same_v<T, Resistor>) {
            stamp_resistor(dev.resistance(), conn.nodes, triplets);
        }
        else if constexpr (std::is_same_v<T, Capacitor>) {
            // DC: capacitor is open (infinite impedance), but add small conductance
            // for numerical stability (R = 1e12 ohms -> g = 1e-12 S)
            stamp_resistor(1e12, conn.nodes, triplets);
        }
        else if constexpr (std::is_same_v<T, Inductor>) {
            // DC: inductor is short, stamp as voltage source with V=0
            stamp_voltage_source(0.0, conn.nodes, conn.branch_index, triplets, b);
        }
        else if constexpr (std::is_same_v<T, VoltageSource>) {
            stamp_voltage_source(dev.voltage(), conn.nodes, conn.branch_index, triplets, b);
        }
        else if constexpr (std::is_same_v<T, CurrentSource>) {
            stamp_current_source(dev.current(), conn.nodes, b);
        }
        else if constexpr (std::is_same_v<T, IdealDiode>) {
            // Initial guess: use off-state conductance for DC stamping
            Real g = dev.is_conducting() ? 1e3 : 1e-9;
            stamp_resistor(1.0 / g, conn.nodes, triplets);
        }
        else if constexpr (std::is_same_v<T, IdealSwitch>) {
            Real g = dev.is_closed() ? 1e6 : 1e-12;
            stamp_resistor(1.0 / g, conn.nodes, triplets);
        }
        else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
            const auto forced = forced_switch_state(device_index);
            const Real g = forced.has_value()
                ? (*forced ? dev.g_on() : dev.g_off())
                : dev.g_off();
            stamp_resistor(1.0 / std::max<Real>(g, 1e-18), {conn.nodes[1], conn.nodes[2]}, triplets);
        }
        else if constexpr (std::is_same_v<T, MOSFET> || std::is_same_v<T, IGBT>) {
            auto params = dev.params();
            const Real scale = device_temperature_scale(device_index);
            const Real inv_scale = 1.0 / std::max<Real>(scale, Real{0.05});
            const auto forced = forced_switch_state(device_index);
            Real g = std::max<Real>(params.g_off * inv_scale, Real{1e-18});
            if (forced.has_value() && *forced) {
                if constexpr (std::is_same_v<T, MOSFET>) {
                    g = std::max<Real>(params.kp * inv_scale, Real{1e-6});
                } else {
                    g = std::max<Real>(params.g_on * inv_scale, Real{1e-6});
                }
            }
            stamp_resistor(1.0 / g, {conn.nodes[1], conn.nodes[2]}, triplets);
        }
        else if constexpr (std::is_same_v<T, PWMVoltageSource>) {
            // DC: use voltage at t=0
            stamp_voltage_source(dev.voltage_at(0.0), conn.nodes, conn.branch_index, triplets, b);
        }
        else if constexpr (std::is_same_v<T, SineVoltageSource>) {
            // DC: use offset (average value)
            stamp_voltage_source(dev.params().offset, conn.nodes, conn.branch_index, triplets, b);
        }
        else if constexpr (std::is_same_v<T, PulseVoltageSource>) {
            // DC: use initial voltage
            stamp_voltage_source(dev.params().v_initial, conn.nodes, conn.branch_index, triplets, b);
        }
        else if constexpr (std::is_same_v<T, Transformer>) {
            // DC: ideal transformer (V1 = n*V2, I1 = -I2/n)
            stamp_transformer(dev.turns_ratio(), conn.nodes, conn.branch_index,
                              conn.branch_index_2, triplets, b);
        }
    }

    template<typename Device, typename Triplets>
    void stamp_device_jacobian(const Device& dev, const DeviceConnection& conn,
                               std::size_t device_index,
                               Triplets& triplets,
                               Vector& f, const Vector& x) const {
        using T = std::decay_t<Device>;

        if constexpr (std::is_same_v<T, Resistor>) {
            Real g = 1.0 / dev.resistance();
            Index n1 = conn.nodes[0];
            Index n2 = conn.nodes[1];
            Real v1 = (n1 >= 0) ? x[n1] : 0.0;
            Real v2 = (n2 >= 0) ? x[n2] : 0.0;
            Real i = g * (v1 - v2);

            // Stamp conductance
            stamp_conductance(g, n1, n2, triplets);

            // Residual (KCL)
            if (n1 >= 0) f[n1] += i;
            if (n2 >= 0) f[n2] -= i;
        }
        else if constexpr (std::is_same_v<T, VoltageSource>) {
            Index npos = conn.nodes[0];
            Index nneg = conn.nodes[1];
            Index br = conn.branch_index;
            Real v_src = dev.voltage();

            // Stamp MNA extension
            if (npos >= 0) {
                triplets.emplace_back(npos, br, 1.0);
                triplets.emplace_back(br, npos, 1.0);
            }
            if (nneg >= 0) {
                triplets.emplace_back(nneg, br, -1.0);
                triplets.emplace_back(br, nneg, -1.0);
            }

            // Residual
            Real vpos = (npos >= 0) ? x[npos] : 0.0;
            Real vneg = (nneg >= 0) ? x[nneg] : 0.0;
            f[br] += (vpos - vneg - v_src);

            // KCL contribution from branch current
            Real i_br = x[br];
            if (npos >= 0) f[npos] += i_br;
            if (nneg >= 0) f[nneg] -= i_br;
        }
        else if constexpr (std::is_same_v<T, CurrentSource>) {
            Index npos = conn.nodes[0];
            Index nneg = conn.nodes[1];
            Real i = dev.current();
            if (npos >= 0) f[npos] -= i;
            if (nneg >= 0) f[nneg] += i;
        }
        else if constexpr (std::is_same_v<T, IdealDiode>) {
            stamp_diode_jacobian(dev, conn.nodes, triplets, f, x);
        }
        else if constexpr (std::is_same_v<T, IdealSwitch>) {
            Real g = dev.is_closed() ? 1e6 : 1e-12;
            Index n1 = conn.nodes[0];
            Index n2 = conn.nodes[1];
            Real v1 = (n1 >= 0) ? x[n1] : 0.0;
            Real v2 = (n2 >= 0) ? x[n2] : 0.0;
            Real i = g * (v1 - v2);

            stamp_conductance(g, n1, n2, triplets);
            if (n1 >= 0) f[n1] += i;
            if (n2 >= 0) f[n2] -= i;
        }
        else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
            stamp_vcswitch_jacobian(dev, conn.nodes, device_index, triplets, f, x);
        }
        else if constexpr (std::is_same_v<T, Capacitor>) {
            Real C = dev.capacitance();
            Index n1 = conn.nodes[0];
            Index n2 = conn.nodes[1];
            Real v1 = (n1 >= 0) ? x[n1] : 0.0;
            Real v2 = (n2 >= 0) ? x[n2] : 0.0;
            Real v = v1 - v2;
            Real v_prev = dev.voltage_prev();
            Real g_eq = 0.0;
            Real i_hist = 0.0;

            if (stage_context_.active && stage_context_.scheme == StageScheme::TRBDF2) {
                auto coeffs = TRBDF2Coeffs::bdf2_variable(stage_context_.h1, stage_context_.h2);
                Real v_stage = stage_cap_v_[device_index];
                g_eq = C * coeffs.a2;
                i_hist = C * (coeffs.a1 * v_stage + coeffs.a0 * v_prev);
            } else if (stage_context_.active && stage_context_.scheme == StageScheme::SDIRK2) {
                Real dt_total = std::max(stage_context_.dt, Real{1e-15});
                Real a22 = std::max(stage_context_.a22, Real{1e-15});
                Real vdot1 = stage_cap_vdot_[device_index];
                g_eq = C / (a22 * dt_total);
                i_hist = -g_eq * (v_prev + dt_total * stage_context_.a21 * vdot1);
            } else if (integration_order_ == 1 || !dev.history_initialized()) {
                // Backward Euler: i = (C/dt) * (v - v_prev)
                //
                // Also used as the first-step warm-up when the cap's
                // history is not yet populated (i.e. `i_prev_` still
                // sits at its constructor default of 0). Trapezoidal
                // applied at this boundary doubles the cap current
                // (`i_n = g_eq·v_n - 0` instead of the analytical
                // `g_eq·(v_n - v_{n-1}) - i_{analytical_t0+}`), which
                // shows up as I(V1) alternating between 0 and 2·I_true
                // on the first few samples — surfaced by
                // `level1_components/test_basic_components.py::TestCapacitor::test_capacitor_charging_current`.
                g_eq = C / timestep_;
                i_hist = -g_eq * v_prev;
            } else {
                // Trapezoidal: i_n = g_eq*(v_n - v_{n-1}) - i_{n-1}
                // i = g_eq*v + (-g_eq*v_prev - i_prev)
                g_eq = 2.0 * C / timestep_;
                Real i_prev = dev.current_prev();
                i_hist = -i_prev - g_eq * v_prev;
            }

            Real i = g_eq * v + i_hist;

            stamp_conductance(g_eq, n1, n2, triplets);
            if (n1 >= 0) f[n1] += i;
            if (n2 >= 0) f[n2] -= i;
        }
        else if constexpr (std::is_same_v<T, Inductor>) {
            Index n1 = conn.nodes[0];
            Index n2 = conn.nodes[1];
            Index br = conn.branch_index;
            const Real i_est = (br >= 0 && br < system_size()) ? x[br] : dev.current_prev();
            Real L = effective_inductance_for(conn.name, i_est, dev.inductance());
            Real v_prev = dev.voltage_prev();
            Real i_prev = dev.current_prev();

            Real coeff = 0.0;
            Real v_eq = 0.0;

            if (stage_context_.active && stage_context_.scheme == StageScheme::TRBDF2) {
                auto coeffs = TRBDF2Coeffs::bdf2_variable(stage_context_.h1, stage_context_.h2);
                Real i_stage = stage_ind_i_[device_index];
                coeff = L * coeffs.a2;
                v_eq = L * (coeffs.a1 * i_stage + coeffs.a0 * i_prev);
            } else if (stage_context_.active && stage_context_.scheme == StageScheme::SDIRK2) {
                Real dt_total = std::max(stage_context_.dt, Real{1e-15});
                Real a22 = std::max(stage_context_.a22, Real{1e-15});
                Real i_dot1 = stage_ind_idot_[device_index];
                coeff = L / (a22 * dt_total);
                v_eq = -coeff * (i_prev + dt_total * stage_context_.a21 * i_dot1);
            } else if (integration_order_ == 1 || !dev.history_initialized()) {
                // Backward Euler: v_n - (L/dt) * i_n = - (L/dt) * i_{n-1}
                // First-step warm-up when history is not yet populated
                // (mirrors the capacitor logic above). Trapezoidal at
                // step 1 with `v_prev_=0` would alternate the inductor
                // voltage between 0 and 2·V_true.
                coeff = L / timestep_;
                v_eq = -coeff * i_prev;
            } else {
                // Trapezoidal: v_n - (2L/dt) * i_n = - (2L/dt) * i_{n-1} - v_{n-1}
                coeff = 2.0 * L / timestep_;
                v_eq = -coeff * i_prev - v_prev;
            }

            // MNA extension
            if (n1 >= 0) {
                triplets.emplace_back(n1, br, 1.0);
                triplets.emplace_back(br, n1, 1.0);
            }
            if (n2 >= 0) {
                triplets.emplace_back(n2, br, -1.0);
                triplets.emplace_back(br, n2, -1.0);
            }
            triplets.emplace_back(br, br, -coeff);

            Real v1 = (n1 >= 0) ? x[n1] : 0.0;
            Real v2 = (n2 >= 0) ? x[n2] : 0.0;
            Real i_br = x[br];
            f[br] += (v1 - v2 - coeff * i_br - v_eq);
            if (n1 >= 0) f[n1] += i_br;
            if (n2 >= 0) f[n2] -= i_br;
        }
        else if constexpr (std::is_same_v<T, MOSFET>) {
            stamp_mosfet_jacobian(dev, conn.nodes, device_index, triplets, f, x);
        }
        else if constexpr (std::is_same_v<T, IGBT>) {
            stamp_igbt_jacobian(dev, conn.nodes, device_index, triplets, f, x);
        }
        else if constexpr (std::is_same_v<T, PWMVoltageSource> ||
                           std::is_same_v<T, SineVoltageSource> ||
                           std::is_same_v<T, PulseVoltageSource>) {
            // Time-varying voltage source
            Index npos = conn.nodes[0];
            Index nneg = conn.nodes[1];
            Index br = conn.branch_index;
            Real v_src = dev.voltage_at(current_time_);

            // Stamp MNA extension (same as regular voltage source)
            if (npos >= 0) {
                triplets.emplace_back(npos, br, 1.0);
                triplets.emplace_back(br, npos, 1.0);
            }
            if (nneg >= 0) {
                triplets.emplace_back(nneg, br, -1.0);
                triplets.emplace_back(br, nneg, -1.0);
            }

            // Residual
            Real vpos = (npos >= 0) ? x[npos] : 0.0;
            Real vneg = (nneg >= 0) ? x[nneg] : 0.0;
            f[br] += (vpos - vneg - v_src);

            // KCL contribution from branch current
            Real i_br = x[br];
            if (npos >= 0) f[npos] += i_br;
            if (nneg >= 0) f[nneg] -= i_br;
        }
        else if constexpr (std::is_same_v<T, Transformer>) {
            // Ideal transformer Jacobian
            stamp_transformer_jacobian(dev.turns_ratio(), conn.nodes, conn.branch_index,
                                       conn.branch_index_2, triplets, f, x);
        }
    }

    // =========================================================================
    // Primitive Stamping Functions
    // =========================================================================

    template<typename Triplets>
    void stamp_resistor(Real R, const std::vector<Index>& nodes,
                        Triplets& triplets) const {
        Real g = 1.0 / R;
        stamp_conductance(g, nodes[0], nodes[1], triplets);
    }

    template<typename Triplets>
    void stamp_conductance(Real g, Index n1, Index n2,
                           Triplets& triplets) const {
        if (n1 >= 0) {
            triplets.emplace_back(n1, n1, g);
            if (n2 >= 0) triplets.emplace_back(n1, n2, -g);
        }
        if (n2 >= 0) {
            triplets.emplace_back(n2, n2, g);
            if (n1 >= 0) triplets.emplace_back(n2, n1, -g);
        }
    }

    template<typename Triplets>
    void stamp_voltage_source(Real V, const std::vector<Index>& nodes, Index br,
                              Triplets& triplets, Vector& b) const {
        Index npos = nodes[0];
        Index nneg = nodes[1];

        if (npos >= 0) {
            triplets.emplace_back(npos, br, 1.0);
            triplets.emplace_back(br, npos, 1.0);
        }
        if (nneg >= 0) {
            triplets.emplace_back(nneg, br, -1.0);
            triplets.emplace_back(br, nneg, -1.0);
        }
        b[br] = V;
    }

    void stamp_current_source(Real I, const std::vector<Index>& nodes, Vector& b) const {
        Index npos = nodes[0];
        Index nneg = nodes[1];
        if (npos >= 0) b[npos] -= I;
        if (nneg >= 0) b[nneg] += I;
    }

    template<typename Triplets>
    void stamp_diode_jacobian(const IdealDiode& /*dev*/, const std::vector<Index>& nodes,
                              Triplets& triplets,
                              Vector& f, const Vector& x) const {
        Index n_anode = nodes[0];
        Index n_cathode = nodes[1];

        Real v_anode = (n_anode >= 0) ? x[n_anode] : 0.0;
        Real v_cathode = (n_cathode >= 0) ? x[n_cathode] : 0.0;
        Real v_diode = v_anode - v_cathode;

        // Determine state (simple ideal model)
        Real g = (v_diode > 0.0) ? 1e3 : 1e-9;
        Real i = g * v_diode;

        stamp_conductance(g, n_anode, n_cathode, triplets);
        if (n_anode >= 0) f[n_anode] += i;
        if (n_cathode >= 0) f[n_cathode] -= i;
    }

    template<typename Triplets>
    void stamp_vcswitch_jacobian(const VoltageControlledSwitch& dev, const std::vector<Index>& nodes,
                                 std::size_t device_index,
                                 Triplets& triplets,
                                 Vector& f, const Vector& x) const {
        Index n_ctrl = nodes[0];
        Index n_t1 = nodes[1];
        Index n_t2 = nodes[2];

        Real v_ctrl = (n_ctrl >= 0) ? x[n_ctrl] : 0.0;
        Real v_t1 = (n_t1 >= 0) ? x[n_t1] : 0.0;
        Real v_t2 = (n_t2 >= 0) ? x[n_t2] : 0.0;

        // Smooth transition using tanh for better convergence.
        // Honor the device's configured hysteresis instead of the
        // previously hardcoded 0.5 V — that was too wide for the
        // typical 100 mV-margin threshold tests and meant a v_ctrl
        // 2.5 V below threshold still produced a non-trivial residual
        // conductance (sigmoid ≈ 5e-5 → g ≈ 5e-5 · g_on, R_sw ≈ 200 Ω
        // for g_on=100 → V_out ≈ 8.2 V instead of ~0 V on a 1 kΩ
        // load). Surfaced by `level3_nonlinear/test_switch_circuits.py`.
        Real v_th = dev.v_threshold();
        Real g_on = dev.g_on();
        Real g_off = dev.g_off();
        Real hysteresis = dev.hysteresis();

        Real g = g_off;
        Real dg_dvctrl = 0.0;
        const auto forced = forced_switch_state(device_index);
        if (forced.has_value()) {
            g = *forced ? g_on : g_off;
        } else {
            const Real v_norm = (v_ctrl - v_th) / hysteresis;
            const Real tanh_val = std::tanh(v_norm);
            const Real sigmoid = 0.5 * (1.0 + tanh_val);
            g = g_off + (g_on - g_off) * sigmoid;
            const Real dsigmoid = 0.5 / hysteresis * (1.0 - tanh_val * tanh_val);
            dg_dvctrl = (g_on - g_off) * dsigmoid;
        }

        // Current through switch
        Real v_sw = v_t1 - v_t2;
        Real i_sw = g * v_sw;

        // Stamp conductance
        stamp_conductance(g, n_t1, n_t2, triplets);

        // Additional Jacobian terms for control voltage dependency
        if (n_ctrl >= 0 && n_t1 >= 0) {
            triplets.emplace_back(n_t1, n_ctrl, dg_dvctrl * v_sw);
        }
        if (n_ctrl >= 0 && n_t2 >= 0) {
            triplets.emplace_back(n_t2, n_ctrl, -dg_dvctrl * v_sw);
        }

        // Residuals
        if (n_t1 >= 0) f[n_t1] += i_sw;
        if (n_t2 >= 0) f[n_t2] -= i_sw;
    }

    template<typename Triplets>
    void stamp_mosfet_jacobian(const MOSFET& dev, const std::vector<Index>& nodes,
                               std::size_t device_index,
                               Triplets& triplets,
                               Vector& f, const Vector& x) const {
        // Phase 12 fix: replace the legacy hard-branch (cutoff /
        // triode / saturation) stamp with the same smooth blend used
        // by `MOSFET::stamp_jacobian_behavioral` in `mosfet.hpp`. The
        // hard branches gave Newton no continuous gradient across
        // region transitions, so DC OP for fixed-V_GATE NMOS tests
        // (`level3_nonlinear/test_mosfet.py::test_threshold_voltage_effect`
        // with V_GATE=4 V, vth=2/3 V, kp=0.5) failed across all
        // strategies (Direct, GminStepping, PseudoTransient,
        // SourceStepping). The smooth model reduces to the legacy
        // hard branches at saturated tails, so SPICE-parity benches
        // are bit-identical.
        const Index n_gate = nodes[0];
        const Index n_drain = nodes[1];
        const Index n_source = nodes[2];

        const Real vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        const Real vd = (n_drain >= 0) ? x[n_drain] : 0.0;
        const Real vs = (n_source >= 0) ? x[n_source] : 0.0;

        const auto& p = dev.params();
        const Real scale = device_temperature_scale(device_index);
        const Real inv_scale = 1.0 / std::max<Real>(scale, Real{0.05});
        const Real kp_eff = p.kp * inv_scale;
        const Real g_off_eff = p.g_off * inv_scale;
        const Real sign = p.is_nmos ? 1.0 : -1.0;
        Real vgs = sign * (vg - vs);
        Real vds = sign * (vd - vs);

        const auto forced = forced_switch_state(device_index);
        if (forced.has_value()) {
            // Forced PWL state: stamp as a pure conductance (g_on or
            // g_off) without the smooth-model overhead. This matches
            // the legacy behavior when the kernel pins the device.
            const Real g = *forced ? p.g_on : g_off_eff;
            const Real id = g * vds;
            if (n_drain >= 0) {
                triplets.emplace_back(n_drain, n_drain, g);
                if (n_source >= 0) triplets.emplace_back(n_drain, n_source, -g);
            }
            if (n_source >= 0) {
                triplets.emplace_back(n_source, n_source, g);
                if (n_drain >= 0) triplets.emplace_back(n_source, n_drain, -g);
            }
            const Real i_eq_forced = id * sign - g * vds;  // = 0
            if (n_drain >= 0) f[n_drain] -= sign * id - i_eq_forced;
            if (n_source >= 0) f[n_source] += sign * id - i_eq_forced;
            return;
        }

        // Smooth Shichman-Hodges (Phase-8 model, replicated here for
        // the runtime triplet-based stamping pipeline). Symbols match
        // `MOSFET::stamp_jacobian_behavioral` in mosfet.hpp.
        constexpr Real kappa = 50.0;
        const Real vth = p.vth;
        const Real lambda = p.lambda;
        const Real g_off = g_off_eff;

        // --- Smooth Vov_eff ---
        const Real sigma_g = 1.0 / (1.0 + std::exp(-kappa * (vgs - vth)));
        const Real dsigma_g_d_vgs = kappa * sigma_g * (1.0 - sigma_g);
        const Real vov_eff = (vgs - vth) * sigma_g;
        const Real dvov_dvgs = sigma_g + (vgs - vth) * dsigma_g_d_vgs;

        // --- Smooth Vds_eff = soft_min(vds, vov_eff) ---
        const Real sigma_sat = 1.0 / (1.0 + std::exp(-kappa * (vov_eff - vds)));
        const Real dsigma_sat_d_arg = kappa * sigma_sat * (1.0 - sigma_sat);
        const Real dsigma_sat_dvgs = dsigma_sat_d_arg * dvov_dvgs;
        const Real dsigma_sat_dvds = -dsigma_sat_d_arg;

        const Real vds_eff = sigma_sat * vds + (1.0 - sigma_sat) * vov_eff;
        const Real dvds_eff_dvgs =
            dsigma_sat_dvgs * vds
            - dsigma_sat_dvgs * vov_eff
            + (1.0 - sigma_sat) * dvov_dvgs;
        const Real dvds_eff_dvds =
            sigma_sat
            + dsigma_sat_dvds * vds
            - dsigma_sat_dvds * vov_eff;

        // --- Channel current id_ch = kp · (Vov_eff·Vds_eff − ½·Vds_eff²) · (1+λvds)
        const Real core = vov_eff * vds_eff - 0.5 * vds_eff * vds_eff;
        const Real lambda_factor = 1.0 + lambda * vds;
        const Real id_ch = kp_eff * core * lambda_factor;

        const Real dcore_dvgs = vds_eff * dvov_dvgs
                                + (vov_eff - vds_eff) * dvds_eff_dvgs;
        const Real dcore_dvds = (vov_eff - vds_eff) * dvds_eff_dvds;

        const Real di_ch_dvgs = kp_eff * dcore_dvgs * lambda_factor;
        const Real di_ch_dvds = kp_eff * (dcore_dvds * lambda_factor + core * lambda);

        // --- Total id (with g_off leakage) ---
        const Real id_internal = id_ch + g_off * vds;
        const Real di_internal_dvgs = di_ch_dvgs;
        const Real di_internal_dvds = di_ch_dvds + g_off;

        // PMOS sign-fold of the OUTPUT current (i_actual = sign · i_internal).
        // ∂id_actual/∂vg = di_internal_dvgs   (sign factors cancel)
        // ∂id_actual/∂vd = di_internal_dvds   (sign factors cancel)
        // ∂id_actual/∂vs = − di_internal_dvgs − di_internal_dvds
        const Real id = sign * id_internal;
        const Real di_dvg = di_internal_dvgs;
        const Real di_dvd = di_internal_dvds;
        const Real di_dvs = -di_internal_dvgs - di_internal_dvds;

        // Newton-Raphson Jacobian + physical residual stamp. The
        // legacy Norton-companion form (`J += di_dvN`, `f -= i_eq`
        // with `i_eq = id - Σ di_dvN·vN`) is meant for MNA-style
        // direct assembly `G·x = b`, NOT for Pulsim's Newton iteration
        // `J·Δx = −f` — in the latter the convergence equation
        // becomes `Σ di_dvN·vN = i_R + id`, which has the vth
        // dependency algebraically cancel out. Newton then settles on
        // a non-physical fixed point where Pulsim's residual is 0
        // even though the actual KCL `id = i_R` is violated by ~50 %.
        // Surfaced by `level3_nonlinear/test_mosfet.py::
        // TestMOSFETParameters::test_threshold_voltage_effect` (same
        // V_drain = 0.025 V for vth ∈ {1, 2}, when physically they
        // should differ).
        //
        // Replace with the standard Newton-Raphson stamp: `J = ∂id/∂x`
        // (same as before), `f = +id(x_old)` (the actual current
        // physically leaving drain via the channel). This is the same
        // sign convention as the R and IGBT stamps (`f[node] +=
        // current leaving node`).

        // Drain row: + ∂id/∂x_i.
        if (n_drain >= 0) {
            triplets.emplace_back(n_drain, n_drain, di_dvd);
            if (n_gate >= 0)   triplets.emplace_back(n_drain, n_gate, di_dvg);
            if (n_source >= 0) triplets.emplace_back(n_drain, n_source, di_dvs);
        }
        // Source row: − ∂id/∂x_i  (current arriving at source = −id).
        if (n_source >= 0) {
            triplets.emplace_back(n_source, n_source, -di_dvs);
            if (n_drain >= 0) triplets.emplace_back(n_source, n_drain, -di_dvd);
            if (n_gate >= 0)  triplets.emplace_back(n_source, n_gate, -di_dvg);
        }

        // Physical residual: +id leaves drain, −id arrives at source.
        if (n_drain >= 0)  f[n_drain]  += id;
        if (n_source >= 0) f[n_source] -= id;
    }

    template<typename Triplets>
    void stamp_igbt_jacobian(const IGBT& dev, const std::vector<Index>& nodes,
                             std::size_t device_index,
                             Triplets& triplets,
                             Vector& f, const Vector& x) const {
        Index n_gate = nodes[0];
        Index n_collector = nodes[1];
        Index n_emitter = nodes[2];

        Real vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Real vc = (n_collector >= 0) ? x[n_collector] : 0.0;
        Real ve = (n_emitter >= 0) ? x[n_emitter] : 0.0;

        const auto& p = dev.params();
        const Real scale = device_temperature_scale(device_index);
        const Real inv_scale = 1.0 / std::max<Real>(scale, Real{0.05});
        const Real g_on_eff = p.g_on * inv_scale;
        const Real g_off_eff = p.g_off * inv_scale;
        Real vge = vg - ve;
        Real vce = vc - ve;

        bool is_on = (vge > p.vth) && (vce > 0);
        if (const auto forced = forced_switch_state(device_index); forced.has_value()) {
            is_on = *forced;
        }
        Real g = is_on ? g_on_eff : g_off_eff;
        Real ic = g * vce;

        // Simple model without saturation for now
        stamp_conductance(g, n_collector, n_emitter, triplets);
        if (n_collector >= 0) f[n_collector] += ic;
        if (n_emitter >= 0) f[n_emitter] -= ic;
    }

    // =========================================================================
    // Transformer Stamping
    // =========================================================================

    /// Stamp ideal transformer for DC analysis
    /// Transformer: V_p = n * V_s, I_p + n * I_s = 0 (power conservation)
    /// nodes: {p1, p2, s1, s2} (primary+, primary-, secondary+, secondary-)
    template<typename Triplets>
    void stamp_transformer(Real n, const std::vector<Index>& nodes,
                           Index br_p, Index br_s,
                           Triplets& triplets, Vector& b) const {
        Index np1 = nodes[0];  // Primary +
        Index np2 = nodes[1];  // Primary -
        Index ns1 = nodes[2];  // Secondary +
        Index ns2 = nodes[3];  // Secondary -

        // KCL: Primary current I_p flows from np1 to np2
        if (np1 >= 0) triplets.emplace_back(np1, br_p, 1.0);
        if (np2 >= 0) triplets.emplace_back(np2, br_p, -1.0);

        // KCL: Secondary current I_s flows from ns1 to ns2
        if (ns1 >= 0) triplets.emplace_back(ns1, br_s, 1.0);
        if (ns2 >= 0) triplets.emplace_back(ns2, br_s, -1.0);

        // Branch equation 1 (row br_p): Vp1 - Vp2 - n*(Vs1 - Vs2) = 0
        // This enforces V_primary = n * V_secondary
        if (np1 >= 0) triplets.emplace_back(br_p, np1, 1.0);
        if (np2 >= 0) triplets.emplace_back(br_p, np2, -1.0);
        if (ns1 >= 0) triplets.emplace_back(br_p, ns1, -n);
        if (ns2 >= 0) triplets.emplace_back(br_p, ns2, n);

        // Branch equation 2 (row br_s): n * I_p + I_s = 0
        // This enforces power conservation: Vp*Ip = Vs*Is
        triplets.emplace_back(br_s, br_p, n);
        triplets.emplace_back(br_s, br_s, 1.0);

        // No DC offset
        b[br_p] = 0.0;
        b[br_s] = 0.0;
    }

    /// Stamp ideal transformer Jacobian for Newton iteration
    template<typename Triplets>
    void stamp_transformer_jacobian(Real n, const std::vector<Index>& nodes,
                                    Index br_p, Index br_s,
                                    Triplets& triplets,
                                    Vector& f, const Vector& x) const {
        Index np1 = nodes[0];
        Index np2 = nodes[1];
        Index ns1 = nodes[2];
        Index ns2 = nodes[3];

        Real vp1 = (np1 >= 0) ? x[np1] : 0.0;
        Real vp2 = (np2 >= 0) ? x[np2] : 0.0;
        Real vs1 = (ns1 >= 0) ? x[ns1] : 0.0;
        Real vs2 = (ns2 >= 0) ? x[ns2] : 0.0;
        Real i_p = x[br_p];
        Real i_s = x[br_s];

        Real v_p = vp1 - vp2;  // Primary voltage
        Real v_s = vs1 - vs2;  // Secondary voltage

        // KCL: Primary current
        if (np1 >= 0) triplets.emplace_back(np1, br_p, 1.0);
        if (np2 >= 0) triplets.emplace_back(np2, br_p, -1.0);

        // KCL: Secondary current
        if (ns1 >= 0) triplets.emplace_back(ns1, br_s, 1.0);
        if (ns2 >= 0) triplets.emplace_back(ns2, br_s, -1.0);

        // Branch equation 1: Vp - n*Vs = 0
        if (np1 >= 0) triplets.emplace_back(br_p, np1, 1.0);
        if (np2 >= 0) triplets.emplace_back(br_p, np2, -1.0);
        if (ns1 >= 0) triplets.emplace_back(br_p, ns1, -n);
        if (ns2 >= 0) triplets.emplace_back(br_p, ns2, n);

        // Branch equation 2: n*I_p + I_s = 0
        triplets.emplace_back(br_s, br_p, n);
        triplets.emplace_back(br_s, br_s, 1.0);

        // Residuals
        // f_br_p = Vp - n*Vs = 0
        f[br_p] += (v_p - n * v_s);

        // f_br_s = n*I_p + I_s = 0
        f[br_s] += (n * i_p + i_s);

        // KCL residual contributions
        if (np1 >= 0) f[np1] += i_p;
        if (np2 >= 0) f[np2] -= i_p;
        if (ns1 >= 0) f[ns1] += i_s;
        if (ns2 >= 0) f[ns2] -= i_s;
    }
};

} // namespace pulsim::v1
