#include "pulsim/circuit.hpp"
#include <algorithm>
#include <stdexcept>

namespace pulsim {

void Circuit::add_resistor(const std::string& name, const NodeId& n1,
                           const NodeId& n2, Real resistance) {
    if (resistance <= 0) {
        throw std::invalid_argument("Resistor " + name + ": resistance must be positive");
    }
    ensure_node(n1);
    ensure_node(n2);
    add_component(Component(name, ComponentType::Resistor, {n1, n2},
                           ResistorParams{resistance}));
}

void Circuit::add_capacitor(const std::string& name, const NodeId& n1,
                            const NodeId& n2, Real capacitance, Real ic) {
    if (capacitance <= 0) {
        throw std::invalid_argument("Capacitor " + name + ": capacitance must be positive");
    }
    ensure_node(n1);
    ensure_node(n2);
    add_component(Component(name, ComponentType::Capacitor, {n1, n2},
                           CapacitorParams{capacitance, ic}));
}

void Circuit::add_inductor(const std::string& name, const NodeId& n1,
                           const NodeId& n2, Real inductance, Real ic) {
    if (inductance <= 0) {
        throw std::invalid_argument("Inductor " + name + ": inductance must be positive");
    }
    ensure_node(n1);
    ensure_node(n2);

    // Inductors add a branch current variable
    branch_map_[name] = branch_count_;
    branch_count_++;

    add_component(Component(name, ComponentType::Inductor, {n1, n2},
                           InductorParams{inductance, ic}));
}

void Circuit::add_voltage_source(const std::string& name, const NodeId& npos,
                                 const NodeId& nneg, const Waveform& waveform) {
    ensure_node(npos);
    ensure_node(nneg);

    // Voltage sources add a branch current variable
    branch_map_[name] = branch_count_;
    branch_count_++;

    add_component(Component(name, ComponentType::VoltageSource, {npos, nneg},
                           VoltageSourceParams{waveform}));
}

void Circuit::add_current_source(const std::string& name, const NodeId& npos,
                                 const NodeId& nneg, const Waveform& waveform) {
    ensure_node(npos);
    ensure_node(nneg);
    add_component(Component(name, ComponentType::CurrentSource, {npos, nneg},
                           CurrentSourceParams{waveform}));
}

void Circuit::add_diode(const std::string& name, const NodeId& anode,
                        const NodeId& cathode, const DiodeParams& params) {
    ensure_node(anode);
    ensure_node(cathode);
    add_component(Component(name, ComponentType::Diode, {anode, cathode}, params));
}

void Circuit::add_switch(const std::string& name, const NodeId& n1,
                         const NodeId& n2, const NodeId& ctrl_pos,
                         const NodeId& ctrl_neg, const SwitchParams& params) {
    ensure_node(n1);
    ensure_node(n2);
    ensure_node(ctrl_pos);
    ensure_node(ctrl_neg);
    add_component(Component(name, ComponentType::Switch, {n1, n2, ctrl_pos, ctrl_neg},
                           params));
}

void Circuit::add_mosfet(const std::string& name, const NodeId& drain,
                         const NodeId& gate, const NodeId& source,
                         const MOSFETParams& params) {
    ensure_node(drain);
    ensure_node(gate);
    ensure_node(source);
    add_component(Component(name, ComponentType::MOSFET, {drain, gate, source}, params));
}

void Circuit::add_igbt(const std::string& name, const NodeId& collector,
                       const NodeId& gate, const NodeId& emitter,
                       const IGBTParams& params) {
    ensure_node(collector);
    ensure_node(gate);
    ensure_node(emitter);
    add_component(Component(name, ComponentType::IGBT, {collector, gate, emitter}, params));
}

void Circuit::add_transformer(const std::string& name, const NodeId& p1,
                              const NodeId& p2, const NodeId& s1,
                              const NodeId& s2, const TransformerParams& params) {
    ensure_node(p1);
    ensure_node(p2);
    ensure_node(s1);
    ensure_node(s2);

    // Transformer adds two branch currents (primary and secondary)
    branch_map_[name + "_p"] = branch_count_++;
    branch_map_[name + "_s"] = branch_count_++;

    add_component(Component(name, ComponentType::Transformer, {p1, p2, s1, s2}, params));
}

const Component* Circuit::find_component(const std::string& name) const {
    auto it = std::find_if(components_.begin(), components_.end(),
                          [&name](const Component& c) { return c.name() == name; });
    return it != components_.end() ? &(*it) : nullptr;
}

Index Circuit::node_index(const NodeId& node) const {
    if (is_ground(node)) {
        return -1;  // Ground is not in the matrix
    }
    auto it = node_map_.find(node);
    if (it == node_map_.end()) {
        throw std::runtime_error("Unknown node: " + node);
    }
    return it->second;
}

bool Circuit::is_ground(const NodeId& node) const {
    return node == GROUND_NODE || node == "gnd" || node == "GND";
}

const NodeId& Circuit::node_name(Index index) const {
    if (index < 0 || index >= static_cast<Index>(node_names_.size())) {
        throw std::out_of_range("Node index out of range");
    }
    return node_names_[index];
}

std::vector<NodeId> Circuit::node_names() const {
    return node_names_;
}

std::string Circuit::signal_name(Index index) const {
    Index n_nodes = node_count();
    if (index < n_nodes) {
        return "V(" + node_names_[index] + ")";
    }

    // Branch current
    Index branch_idx = index - n_nodes;
    for (const auto& [name, idx] : branch_map_) {
        if (idx == branch_idx) {
            return "I(" + name + ")";
        }
    }
    return "x[" + std::to_string(index) + "]";
}

bool Circuit::validate(std::string& error_message) const {
    // Check for at least one component
    if (components_.empty()) {
        error_message = "Circuit has no components";
        return false;
    }

    // Check for at least one source
    bool has_source = false;
    for (const auto& comp : components_) {
        if (comp.type() == ComponentType::VoltageSource ||
            comp.type() == ComponentType::CurrentSource) {
            has_source = true;
            break;
        }
    }
    if (!has_source) {
        error_message = "Circuit has no sources";
        return false;
    }

    // Check for ground connection
    bool has_ground = false;
    for (const auto& comp : components_) {
        for (const auto& node : comp.nodes()) {
            if (is_ground(node)) {
                has_ground = true;
                break;
            }
        }
        if (has_ground) break;
    }
    if (!has_ground) {
        error_message = "Circuit has no ground connection";
        return false;
    }

    return true;
}

void Circuit::ensure_node(const NodeId& node) {
    if (is_ground(node)) {
        return;  // Ground is implicit
    }
    if (node_map_.find(node) == node_map_.end()) {
        Index idx = static_cast<Index>(node_map_.size());
        node_map_[node] = idx;
        node_names_.push_back(node);
    }
}

void Circuit::add_component(Component component) {
    components_.push_back(std::move(component));
}

// Schematic position management
void Circuit::set_position(const std::string& component_name, const SchematicPosition& pos) {
    positions_[component_name] = pos;
}

std::optional<SchematicPosition> Circuit::get_position(const std::string& component_name) const {
    auto it = positions_.find(component_name);
    if (it != positions_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool Circuit::has_position(const std::string& component_name) const {
    return positions_.count(component_name) > 0;
}

std::unordered_map<std::string, SchematicPosition> Circuit::all_positions() const {
    return positions_;
}

void Circuit::set_all_positions(const std::unordered_map<std::string, SchematicPosition>& positions) {
    positions_ = positions;
}

void Circuit::clear_positions() {
    positions_.clear();
}

}  // namespace pulsim
