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
    MOSFET,
    IGBT
>;

// =============================================================================
// Device Connection Info
// =============================================================================

struct DeviceConnection {
    std::string name;
    std::vector<Index> nodes;  // Node indices for this device
    Index branch_index = -1;   // For devices with branch currents (VS, L)
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

    // =========================================================================
    // Device Addition
    // =========================================================================

    void add_resistor(const std::string& name, Index n1, Index n2, Real R) {
        devices_.emplace_back(Resistor(R, name));
        connections_.push_back({name, {n1, n2}, -1});
    }

    void add_capacitor(const std::string& name, Index n1, Index n2, Real C, Real ic = 0.0) {
        devices_.emplace_back(Capacitor(C, ic, name));
        connections_.push_back({name, {n1, n2}, -1});
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

    /// Number of devices
    [[nodiscard]] std::size_t num_devices() const { return devices_.size(); }

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

                    Real i;
                    if (initialize) {
                        // At t=0: set v_prev = v, i_prev = 0 (DC steady state)
                        i = 0.0;
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
                        // This ensures first step uses: v_eq = -(2L/dt)*i_prev - v_prev
                        // With v_prev = 0, i_prev = i (from DC), the history is consistent
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
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(devices_.size() * 9);  // Estimate

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                stamp_device_dc(dev, conn, triplets, b);
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

        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(devices_.size() * 9);

        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const auto& conn = connections_[i];
            std::visit([&](const auto& dev) {
                stamp_device_jacobian(dev, conn, triplets, f, x);
            }, devices_[i]);
        }

        J.setFromTriplets(triplets.begin(), triplets.end());
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

private:
    std::vector<DeviceVariant> devices_;
    std::vector<DeviceConnection> connections_;
    std::unordered_map<std::string, Index> node_map_;
    std::vector<std::string> node_names_;
    Index num_branches_ = 0;
    Real timestep_ = 1e-6;
    inline static const std::string ground_name_ = "0";

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
        else if constexpr (std::is_same_v<T, MOSFET> || std::is_same_v<T, IGBT>) {
            // Initial: off-state conductance
            auto params = dev.params();
            stamp_resistor(1.0 / params.g_off, {conn.nodes[1], conn.nodes[2]}, triplets);
        }
    }

    template<typename Device>
    void stamp_device_jacobian(const Device& dev, const DeviceConnection& conn,
                               std::vector<Eigen::Triplet<Real>>& triplets,
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
        else if constexpr (std::is_same_v<T, Capacitor>) {
            // Trapezoidal companion model: i_n = g_eq*(v_n - v_{n-1}) - i_{n-1}
            // Rewritten as: i_n = g_eq*v_n + I_eq, where I_eq = -g_eq*v_{n-1} - i_{n-1}
            Real C = dev.capacitance();
            Real g_eq = 2.0 * C / timestep_;
            Index n1 = conn.nodes[0];
            Index n2 = conn.nodes[1];
            Real v1 = (n1 >= 0) ? x[n1] : 0.0;
            Real v2 = (n2 >= 0) ? x[n2] : 0.0;
            Real v = v1 - v2;
            Real v_prev = dev.voltage_prev();
            Real i_prev = dev.current_prev();
            Real i_hist = -i_prev - g_eq * v_prev;
            Real i = g_eq * v + i_hist;

            stamp_conductance(g_eq, n1, n2, triplets);
            if (n1 >= 0) f[n1] += i;
            if (n2 >= 0) f[n2] -= i;
        }
        else if constexpr (std::is_same_v<T, Inductor>) {
            // Trapezoidal: stamp as voltage source with history
            Index n1 = conn.nodes[0];
            Index n2 = conn.nodes[1];
            Index br = conn.branch_index;
            Real L = dev.inductance();
            Real v_prev = dev.voltage_prev();
            Real i_prev = dev.current_prev();

            // Trapezoidal (inductor): v_n = (2L/dt)(i_n - i_{n-1}) - v_{n-1}
            // Rearranged for MNA branch equation:
            //   v_n - (2L/dt) * i_n = - (2L/dt) * i_{n-1} - v_{n-1}
            // The right-hand side is encoded as v_eq with a negative sign.
            Real v_eq = - (2.0 * L / timestep_) * i_prev - v_prev;

            // MNA extension
            if (n1 >= 0) {
                triplets.emplace_back(n1, br, 1.0);
                triplets.emplace_back(br, n1, 1.0);
            }
            if (n2 >= 0) {
                triplets.emplace_back(n2, br, -1.0);
                triplets.emplace_back(br, n2, -1.0);
            }
            triplets.emplace_back(br, br, -2.0 * L / timestep_);

            Real v1 = (n1 >= 0) ? x[n1] : 0.0;
            Real v2 = (n2 >= 0) ? x[n2] : 0.0;
            Real i_br = x[br];
            f[br] += (v1 - v2 - (2.0 * L / timestep_) * i_br - v_eq);
            if (n1 >= 0) f[n1] += i_br;
            if (n2 >= 0) f[n2] -= i_br;
        }
        else if constexpr (std::is_same_v<T, MOSFET>) {
            stamp_mosfet_jacobian(dev, conn.nodes, triplets, f, x);
        }
        else if constexpr (std::is_same_v<T, IGBT>) {
            stamp_igbt_jacobian(dev, conn.nodes, triplets, f, x);
        }
    }

    // =========================================================================
    // Primitive Stamping Functions
    // =========================================================================

    void stamp_resistor(Real R, const std::vector<Index>& nodes,
                        std::vector<Eigen::Triplet<Real>>& triplets) const {
        Real g = 1.0 / R;
        stamp_conductance(g, nodes[0], nodes[1], triplets);
    }

    void stamp_conductance(Real g, Index n1, Index n2,
                           std::vector<Eigen::Triplet<Real>>& triplets) const {
        if (n1 >= 0) {
            triplets.emplace_back(n1, n1, g);
            if (n2 >= 0) triplets.emplace_back(n1, n2, -g);
        }
        if (n2 >= 0) {
            triplets.emplace_back(n2, n2, g);
            if (n1 >= 0) triplets.emplace_back(n2, n1, -g);
        }
    }

    void stamp_voltage_source(Real V, const std::vector<Index>& nodes, Index br,
                              std::vector<Eigen::Triplet<Real>>& triplets, Vector& b) const {
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

    void stamp_diode_jacobian(const IdealDiode& /*dev*/, const std::vector<Index>& nodes,
                              std::vector<Eigen::Triplet<Real>>& triplets,
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

    void stamp_mosfet_jacobian(const MOSFET& dev, const std::vector<Index>& nodes,
                               std::vector<Eigen::Triplet<Real>>& triplets,
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

    void stamp_igbt_jacobian(const IGBT& dev, const std::vector<Index>& nodes,
                             std::vector<Eigen::Triplet<Real>>& triplets,
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
};

} // namespace pulsim::v1
