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
#include <algorithm>
#include <variant>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace pulsim::v1 {

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

// =============================================================================
// Runtime Circuit Class
// =============================================================================

class Circuit {
public:
    Circuit() = default;

    // =========================================================================
    // Node Management
    // =========================================================================

    /// Add a named node and return its index
    Index add_node(const std::string& name) {
        if (name == "0" || name == "gnd" || name == "GND") {
            return ground_node;
        }
        auto it = node_map_.find(name);
        if (it != node_map_.end()) {
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
        Index idx = static_cast<Index>(node_names_.size());
        node_map_[name] = idx;
        node_names_.push_back(name);
        return idx;
    }

    /// Get node index by name (-1 for ground)
    [[nodiscard]] Index get_node(const std::string& name) const {
        if (name == "0" || name == "gnd" || name == "GND") {
            return ground_node;
        }
        auto it = node_map_.find(name);
        if (it == node_map_.end()) {
            throw std::runtime_error("Node not found: " + name);
        }
        return it->second;
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
        resistor_cache_.push_back({n1, n2, R == 0.0 ? 0.0 : 1.0 / R});
    }

    void add_capacitor(const std::string& name, Index n1, Index n2, Real C, Real ic = 0.0) {
        devices_.emplace_back(Capacitor(C, ic, name));
        connections_.push_back({name, {n1, n2}, -1});
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
        num_branches_++;
    }

    void add_voltage_source(const std::string& name, Index npos, Index nneg, Real V) {
        Index br = num_nodes() + num_branches_;
        auto vs = VoltageSource(V, name);
        vs.set_branch_index(br);
        devices_.emplace_back(std::move(vs));
        connections_.push_back({name, {npos, nneg}, br});
        num_branches_++;
    }

    void add_current_source(const std::string& name, Index npos, Index nneg, Real I) {
        devices_.emplace_back(CurrentSource(I, name));
        connections_.push_back({name, {npos, nneg}, -1});
    }

    void add_diode(const std::string& name, Index anode, Index cathode,
                   Real g_on = 1e3, Real g_off = 1e-9) {
        devices_.emplace_back(IdealDiode(g_on, g_off, name));
        connections_.push_back({name, {anode, cathode}, -1});
    }

    void add_switch(const std::string& name, Index n1, Index n2,
                    bool closed = false, Real g_on = 1e6, Real g_off = 1e-12) {
        devices_.emplace_back(IdealSwitch(g_on, g_off, closed, name));
        connections_.push_back({name, {n1, n2}, -1});
    }

    /// Add voltage-controlled switch (controlled by a PWM source)
    /// ctrl: control node (typically driven by PWM), t1/t2: switch terminals
    void add_vcswitch(const std::string& name, Index ctrl, Index t1, Index t2,
                      Real v_threshold = 2.5, Real g_on = 1e3, Real g_off = 1e-9) {
        devices_.emplace_back(VoltageControlledSwitch(v_threshold, g_on, g_off, name));
        connections_.push_back({name, {ctrl, t1, t2}, -1});
    }

    void add_mosfet(const std::string& name, Index gate, Index drain, Index source,
                    const MOSFET::Params& params = MOSFET::Params{}) {
        devices_.emplace_back(MOSFET(params, name));
        connections_.push_back({name, {gate, drain, source}, -1});
    }

    void add_igbt(const std::string& name, Index gate, Index collector, Index emitter,
                  const IGBT::Params& params = IGBT::Params{}) {
        devices_.emplace_back(IGBT(params, name));
        connections_.push_back({name, {gate, collector, emitter}, -1});
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

        auto find_connection = [&](const std::string& name) -> const DeviceConnection* {
            for (const auto& conn : connections_) {
                if (conn.name == name) {
                    return &conn;
                }
            }
            return nullptr;
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
        Real igbt_g_off_min = 1e-9) {
        int changed = 0;

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
        num_branches_++;
    }

    // =========================================================================
    // PWM Duty Control
    // =========================================================================

    /// Set fixed duty cycle for a PWM source
    void set_pwm_duty(const std::string& name, Real duty) {
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (connections_[i].name == name) {
                if (auto* pwm = std::get_if<PWMVoltageSource>(&devices_[i])) {
                    pwm->set_duty(duty);
                    return;
                }
            }
        }
        throw std::runtime_error("PWM source not found: " + name);
    }

    /// Set duty callback for a PWM source
    void set_pwm_duty_callback(const std::string& name,
                                std::function<Real(Real)> callback) {
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (connections_[i].name == name) {
                if (auto* pwm = std::get_if<PWMVoltageSource>(&devices_[i])) {
                    pwm->set_duty_callback(std::move(callback));
                    return;
                }
            }
        }
        throw std::runtime_error("PWM source not found: " + name);
    }

    /// Clear duty callback (use fixed duty)
    void clear_pwm_duty_callback(const std::string& name) {
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (connections_[i].name == name) {
                if (auto* pwm = std::get_if<PWMVoltageSource>(&devices_[i])) {
                    pwm->clear_duty_callback();
                    return;
                }
            }
        }
        throw std::runtime_error("PWM source not found: " + name);
    }

    /// Get PWM state (ON/OFF) at current time
    [[nodiscard]] bool get_pwm_state(const std::string& name) const {
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (connections_[i].name == name) {
                if (const auto* pwm = std::get_if<PWMVoltageSource>(&devices_[i])) {
                    return pwm->state_at(current_time_);
                }
            }
        }
        throw std::runtime_error("PWM source not found: " + name);
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

    void set_switch_state(const std::string& name, bool closed) {
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (connections_[i].name == name) {
                if (auto* sw = std::get_if<IdealSwitch>(&devices_[i])) {
                    sw->set_state(closed);
                    return;
                }
            }
        }
        throw std::runtime_error("Switch not found: " + name);
    }

    // =========================================================================
    // Set Timestep for Dynamic Elements
    // =========================================================================

    void set_timestep(Real dt) {
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
                    stamp_device_dc(dev, conn, triplets, b);
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
                    if (n1 >= 0) f[n1] += i_br;
                    if (n2 >= 0) f[n2] -= i_br;
                    if (br >= 0) f[br] += (v1 - v2 - dev.inductance() * di_dt);
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
        return false;
    }

    // =========================================================================
    // Accessors for Circuit Conversion (AC Analysis)
    // =========================================================================

    /// Get all devices (for conversion to IR circuit)
    [[nodiscard]] const std::vector<DeviceVariant>& devices() const { return devices_; }

    /// Get all connections (for conversion to IR circuit)
    [[nodiscard]] const std::vector<DeviceConnection>& connections() const { return connections_; }

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

    void ensure_stage_storage() {
        if (stage_cap_v_.size() != devices_.size()) {
            stage_cap_v_.assign(devices_.size(), 0.0);
            stage_ind_i_.assign(devices_.size(), 0.0);
            stage_cap_vdot_.assign(devices_.size(), 0.0);
            stage_ind_idot_.assign(devices_.size(), 0.0);
        }
    }

    static void reserve_triplets(std::vector<Eigen::Triplet<Real>>& triplets, std::size_t estimate) {
        if (triplets.capacity() < estimate) {
            triplets.reserve(estimate);
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
    std::vector<DeviceConnection> connections_;
    std::vector<ResistorStamp> resistor_cache_;
    std::unordered_map<std::string, Index> node_map_;
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
            stamp_resistor(1.0 / params.g_off, {conn.nodes[1], conn.nodes[2]}, triplets);
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
            Real L = dev.inductance();
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
            stamp_mosfet_jacobian(dev, conn.nodes, triplets, f, x);
        }
        else if constexpr (std::is_same_v<T, IGBT>) {
            stamp_igbt_jacobian(dev, conn.nodes, triplets, f, x);
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
        Real sigmoid = 0.5 * (1.0 + std::tanh(v_norm));
        Real g = g_off + (g_on - g_off) * sigmoid;

        // Derivative of g w.r.t. v_ctrl
        Real tanh_val = std::tanh(v_norm);
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
                               Triplets& triplets,
                               Vector& f, const Vector& x) const {
        Index n_gate = nodes[0];
        Index n_drain = nodes[1];
        Index n_source = nodes[2];

        Real vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Real vd = (n_drain >= 0) ? x[n_drain] : 0.0;
        Real vs = (n_source >= 0) ? x[n_source] : 0.0;

        const auto& p = dev.params();
        Real sign = p.is_nmos ? 1.0 : -1.0;
        Real vgs = sign * (vg - vs);
        Real vds = sign * (vd - vs);

        Real id = 0.0, gm = 0.0, gds = 0.0;

        if (vgs <= p.vth) {
            // Cutoff
            id = p.g_off * vds;
            gds = p.g_off;
        } else if (vds < vgs - p.vth) {
            // Linear
            Real vov = vgs - p.vth;
            id = p.kp * (vov * vds - 0.5 * vds * vds) * (1.0 + p.lambda * vds);
            gm = p.kp * vds * (1.0 + p.lambda * vds);
            gds = p.kp * (vov - vds) * (1.0 + p.lambda * vds) +
                  p.kp * (vov * vds - 0.5 * vds * vds) * p.lambda;
        } else {
            // Saturation
            Real vov = vgs - p.vth;
            id = 0.5 * p.kp * vov * vov * (1.0 + p.lambda * vds);
            gm = p.kp * vov * (1.0 + p.lambda * vds);
            gds = 0.5 * p.kp * vov * vov * p.lambda;
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
                             Triplets& triplets,
                             Vector& f, const Vector& x) const {
        Index n_gate = nodes[0];
        Index n_collector = nodes[1];
        Index n_emitter = nodes[2];

        Real vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Real vc = (n_collector >= 0) ? x[n_collector] : 0.0;
        Real ve = (n_emitter >= 0) ? x[n_emitter] : 0.0;

        const auto& p = dev.params();
        Real vge = vg - ve;
        Real vce = vc - ve;

        bool is_on = (vge > p.vth) && (vce > 0);
        Real g = is_on ? p.g_on : p.g_off;
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
