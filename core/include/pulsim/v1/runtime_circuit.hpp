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

        // Apply capacitor initial voltages.
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Capacitor>) {
                    Index n1 = conn.nodes[0];
                    Index n2 = conn.nodes[1];
                    Real v = dev.voltage_prev();

                    if (n1 >= 0 && n2 < 0) {
                        set_node(n1, v);
                    } else if (n2 >= 0 && n1 < 0) {
                        set_node(n2, -v);
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
    void add_vcswitch(const std::string& name, Index ctrl, Index t1, Index t2,
                      Real v_threshold = 2.5, Real g_on = 1e3, Real g_off = 1e-9) {
        devices_.emplace_back(VoltageControlledSwitch(v_threshold, g_on, g_off, name));
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
                if (get_numeric(component, "duty_from_input", 0.0) > 0.5) {
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
                output = (phase < duty) ? 1.0 : 0.0;
                result.channel_values[component.name + ".duty"] = duty;
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
        for (const auto& [probe_name, probe_value] : probes) {
            result.channel_values.emplace(probe_name, probe_value);
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

        return result;
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

    // =========================================================================
    // Set Switch States
    // =========================================================================

    void set_switch_state(std::string_view name, bool closed) {
        if (auto* sw = find_device<IdealSwitch>(name)) {
            sw->set_state(closed);
            return;
        }
        throw std::runtime_error("Switch not found: " + std::string{name});
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
                    } else if (integration_order_ == 1) {
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
    // Accessors for Circuit Conversion (AC Analysis)
    // =========================================================================

    /// Get all devices (for conversion to IR circuit)
    [[nodiscard]] const std::vector<DeviceVariant>& devices() const { return devices_; }

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
    }

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
    [[nodiscard]] Device* find_device(std::string_view name) {
        if (const auto index = find_connection_index(name); index.has_value()) {
            return std::get_if<Device>(&devices_[*index]);
        }
        return nullptr;
    }

    template<typename Device>
    [[nodiscard]] const Device* find_device(std::string_view name) const {
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
    std::vector<ResistorStamp> resistor_cache_;
    std::unordered_map<std::string, Index, TransparentStringHash, TransparentStringEqual> node_map_;
    std::vector<std::string> node_names_;
    Index num_branches_ = 0;
    Real timestep_ = 1e-6;
    Integrator integration_method_ = Integrator::Trapezoidal;
    int integration_order_ = 2;  // companion-model order (1 = BE, 2 = TR)
    Real current_time_ = 0.0;
    inline static const std::string ground_name_ = "0";

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
            // Initial: off-state conductance between t1 and t2
            stamp_resistor(1.0 / dev.g_off(), {conn.nodes[1], conn.nodes[2]}, triplets);
        }
        else if constexpr (std::is_same_v<T, MOSFET> || std::is_same_v<T, IGBT>) {
            // Initial: off-state conductance
            auto params = dev.params();
            const Real scale = device_temperature_scale(device_index);
            const Real g_off = std::max<Real>(params.g_off / std::max<Real>(scale, Real{0.05}),
                                              Real{1e-18});
            stamp_resistor(1.0 / g_off, {conn.nodes[1], conn.nodes[2]}, triplets);
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
            stamp_vcswitch_jacobian(dev, conn.nodes, triplets, f, x);
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
            } else if (integration_order_ == 1) {
                // Backward Euler: i = (C/dt) * (v - v_prev)
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
            } else if (integration_order_ == 1) {
                // Backward Euler: v_n - (L/dt) * i_n = - (L/dt) * i_{n-1}
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
                                 Triplets& triplets,
                                 Vector& f, const Vector& x) const {
        Index n_ctrl = nodes[0];
        Index n_t1 = nodes[1];
        Index n_t2 = nodes[2];

        Real v_ctrl = (n_ctrl >= 0) ? x[n_ctrl] : 0.0;
        Real v_t1 = (n_t1 >= 0) ? x[n_t1] : 0.0;
        Real v_t2 = (n_t2 >= 0) ? x[n_t2] : 0.0;

        // Smooth transition using tanh for better convergence
        Real v_th = dev.v_threshold();
        Real g_on = dev.g_on();
        Real g_off = dev.g_off();
        Real hysteresis = 0.5;  // Smooth transition width

        Real v_norm = (v_ctrl - v_th) / hysteresis;
        const Real tanh_val = std::tanh(v_norm);
        Real sigmoid = 0.5 * (1.0 + tanh_val);
        Real g = g_off + (g_on - g_off) * sigmoid;

        // Derivative of g w.r.t. v_ctrl
        Real dsigmoid = 0.5 / hysteresis * (1.0 - tanh_val * tanh_val);
        Real dg_dvctrl = (g_on - g_off) * dsigmoid;

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
        Index n_gate = nodes[0];
        Index n_drain = nodes[1];
        Index n_source = nodes[2];

        Real vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Real vd = (n_drain >= 0) ? x[n_drain] : 0.0;
        Real vs = (n_source >= 0) ? x[n_source] : 0.0;

        const auto& p = dev.params();
        const Real scale = device_temperature_scale(device_index);
        const Real inv_scale = 1.0 / std::max<Real>(scale, Real{0.05});
        const Real kp_eff = p.kp * inv_scale;
        const Real g_off_eff = p.g_off * inv_scale;
        Real sign = p.is_nmos ? 1.0 : -1.0;
        Real vgs = sign * (vg - vs);
        Real vds = sign * (vd - vs);

        Real id = 0.0, gm = 0.0, gds = 0.0;

        if (vgs <= p.vth) {
            // Cutoff
            id = g_off_eff * vds;
            gds = g_off_eff;
        } else if (vds < vgs - p.vth) {
            // Linear
            Real vov = vgs - p.vth;
            id = kp_eff * (vov * vds - 0.5 * vds * vds) * (1.0 + p.lambda * vds);
            gm = kp_eff * vds * (1.0 + p.lambda * vds);
            gds = kp_eff * (vov - vds) * (1.0 + p.lambda * vds) +
                  kp_eff * (vov * vds - 0.5 * vds * vds) * p.lambda;
        } else {
            // Saturation
            Real vov = vgs - p.vth;
            id = 0.5 * kp_eff * vov * vov * (1.0 + p.lambda * vds);
            gm = kp_eff * vov * (1.0 + p.lambda * vds);
            gds = 0.5 * kp_eff * vov * vov * p.lambda;
        }

        id *= sign;
        Real i_eq = id - gm * vgs - gds * vds;

        // Stamp Jacobian
        if (n_drain >= 0) {
            triplets.emplace_back(n_drain, n_drain, gds);
            if (n_source >= 0) triplets.emplace_back(n_drain, n_source, -(gds + gm));
            if (n_gate >= 0) triplets.emplace_back(n_drain, n_gate, gm);
        }
        if (n_source >= 0) {
            triplets.emplace_back(n_source, n_source, gds + gm);
            if (n_drain >= 0) triplets.emplace_back(n_source, n_drain, -gds);
            if (n_gate >= 0) triplets.emplace_back(n_source, n_gate, -gm);
        }

        // Residual
        if (n_drain >= 0) f[n_drain] -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
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
