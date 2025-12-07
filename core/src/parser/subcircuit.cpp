#include "pulsim/parser/subcircuit.hpp"
#include <algorithm>
#include <cmath>
#include <regex>
#include <sstream>

namespace pulsim::parser {

// =============================================================================
// SubcircuitDefinition Implementation
// =============================================================================

SubcircuitDefinition::SubcircuitDefinition(std::string name, std::vector<std::string> ports)
    : name_(std::move(name))
    , ports_(std::move(ports))
{}

void SubcircuitDefinition::add_resistor(const std::string& name, const std::string& n1,
                                         const std::string& n2, double value) {
    ComponentDef comp;
    comp.type = "R";
    comp.name = name;
    comp.nodes = {n1, n2};
    comp.value = value;
    components_.push_back(comp);
}

void SubcircuitDefinition::add_capacitor(const std::string& name, const std::string& n1,
                                          const std::string& n2, double value, double ic) {
    ComponentDef comp;
    comp.type = "C";
    comp.name = name;
    comp.nodes = {n1, n2};
    comp.value = value;
    if (ic != 0) {
        comp.params["IC"] = std::to_string(ic);
    }
    components_.push_back(comp);
}

void SubcircuitDefinition::add_inductor(const std::string& name, const std::string& n1,
                                         const std::string& n2, double value, double ic) {
    ComponentDef comp;
    comp.type = "L";
    comp.name = name;
    comp.nodes = {n1, n2};
    comp.value = value;
    if (ic != 0) {
        comp.params["IC"] = std::to_string(ic);
    }
    components_.push_back(comp);
}

void SubcircuitDefinition::add_voltage_source(const std::string& name, const std::string& np,
                                               const std::string& nn, double value) {
    ComponentDef comp;
    comp.type = "V";
    comp.name = name;
    comp.nodes = {np, nn};
    comp.value = value;
    components_.push_back(comp);
}

void SubcircuitDefinition::add_current_source(const std::string& name, const std::string& np,
                                               const std::string& nn, double value) {
    ComponentDef comp;
    comp.type = "I";
    comp.name = name;
    comp.nodes = {np, nn};
    comp.value = value;
    components_.push_back(comp);
}

void SubcircuitDefinition::add_diode(const std::string& name, const std::string& anode,
                                      const std::string& cathode, const std::string& model) {
    ComponentDef comp;
    comp.type = "D";
    comp.name = name;
    comp.nodes = {anode, cathode};
    comp.model = model;
    components_.push_back(comp);
}

void SubcircuitDefinition::add_mosfet(const std::string& name, const std::string& d,
                                       const std::string& g, const std::string& s,
                                       const std::string& b, const std::string& model,
                                       double w, double l) {
    ComponentDef comp;
    comp.type = "M";
    comp.name = name;
    comp.nodes = {d, g, s, b};
    comp.model = model;
    comp.params["W"] = std::to_string(w);
    comp.params["L"] = std::to_string(l);
    components_.push_back(comp);
}

void SubcircuitDefinition::add_switch(const std::string& name, const std::string& n1,
                                       const std::string& n2, const std::string& control,
                                       double ron) {
    ComponentDef comp;
    comp.type = "SW";
    comp.name = name;
    comp.nodes = {n1, n2};
    comp.params["CONTROL"] = control;
    comp.params["RON"] = std::to_string(ron);
    components_.push_back(comp);
}

void SubcircuitDefinition::add_subcircuit_instance(const std::string& instance_name,
                                                    const std::string& subckt_name,
                                                    const std::vector<std::string>& connections) {
    SubcircuitInstance inst;
    inst.instance_name = instance_name;
    inst.subckt_name = subckt_name;
    inst.connections = connections;
    instances_.push_back(inst);
}

void SubcircuitDefinition::set_parameter(const std::string& name, double value) {
    parameters_[name] = std::to_string(value);
}

void SubcircuitDefinition::set_parameter(const std::string& name, const std::string& value) {
    parameters_[name] = value;
}

// =============================================================================
// SubcircuitLibrary Implementation
// =============================================================================

void SubcircuitLibrary::add(std::shared_ptr<SubcircuitDefinition> subckt) {
    subcircuits_[subckt->name()] = std::move(subckt);
}

void SubcircuitLibrary::add(const std::string& name, std::vector<std::string> ports) {
    subcircuits_[name] = std::make_shared<SubcircuitDefinition>(name, std::move(ports));
}

SubcircuitDefinition* SubcircuitLibrary::get(const std::string& name) {
    auto it = subcircuits_.find(name);
    return it != subcircuits_.end() ? it->second.get() : nullptr;
}

const SubcircuitDefinition* SubcircuitLibrary::get(const std::string& name) const {
    auto it = subcircuits_.find(name);
    return it != subcircuits_.end() ? it->second.get() : nullptr;
}

bool SubcircuitLibrary::exists(const std::string& name) const {
    return subcircuits_.count(name) > 0;
}

std::vector<std::string> SubcircuitLibrary::list() const {
    std::vector<std::string> names;
    names.reserve(subcircuits_.size());
    for (const auto& [name, _] : subcircuits_) {
        names.push_back(name);
    }
    return names;
}

void SubcircuitLibrary::clear() {
    subcircuits_.clear();
}

// =============================================================================
// SubcircuitExpander Implementation
// =============================================================================

SubcircuitExpander::SubcircuitExpander(const SubcircuitLibrary& library)
    : library_(library)
{}

std::string SubcircuitExpander::make_internal_node(
    const std::string& instance,
    const std::string& node,
    const std::vector<std::string>& ports,
    const std::vector<std::string>& connections) {

    // Check if node is a port
    for (size_t i = 0; i < ports.size(); ++i) {
        if (node == ports[i]) {
            return connections[i];  // Map to external connection
        }
    }

    // Ground is always global
    if (node == "0" || node == "GND" || node == "gnd") {
        return "0";
    }

    // Internal node - prefix with instance name
    return instance + "." + node;
}

double SubcircuitExpander::evaluate_parameter(
    const std::string& expr,
    const std::unordered_map<std::string, std::string>& params,
    const std::unordered_map<std::string, std::string>& defaults) {

    // Simple parameter substitution (no expression evaluation)
    std::string value = expr;

    // First check overrides, then defaults
    if (params.count(expr)) {
        value = params.at(expr);
    } else if (defaults.count(expr)) {
        value = defaults.at(expr);
    }

    try {
        return std::stod(value);
    } catch (...) {
        return 0.0;
    }
}

void SubcircuitExpander::expand_into(
    Circuit& circuit,
    const std::string& instance_name,
    const std::string& subckt_name,
    const std::vector<std::string>& port_connections,
    const std::unordered_map<std::string, std::string>& params) {

    // Check recursion depth
    if (expansion_depth_++ > max_depth_) {
        errors_.push_back("Maximum expansion depth exceeded for " + instance_name);
        expansion_depth_--;
        return;
    }

    // Get subcircuit definition
    const auto* subckt = library_.get(subckt_name);
    if (!subckt) {
        errors_.push_back("Subcircuit not found: " + subckt_name);
        expansion_depth_--;
        return;
    }

    // Check port count
    if (port_connections.size() != subckt->ports().size()) {
        errors_.push_back("Port count mismatch for " + instance_name +
                          ": expected " + std::to_string(subckt->ports().size()) +
                          ", got " + std::to_string(port_connections.size()));
        expansion_depth_--;
        return;
    }

    const auto& ports = subckt->ports();
    [[maybe_unused]] const auto& defaults = subckt->parameters();

    // Expand components
    for (const auto& comp : subckt->components()) {
        std::string full_name = instance_name + "." + comp.name;

        // Map nodes
        std::vector<std::string> mapped_nodes;
        for (const auto& node : comp.nodes) {
            mapped_nodes.push_back(make_internal_node(instance_name, node, ports, port_connections));
        }

        // Get parameter value
        double value = comp.value;

        // Add to circuit
        if (comp.type == "R") {
            circuit.add_resistor(full_name, mapped_nodes[0], mapped_nodes[1], value);
        } else if (comp.type == "C") {
            double ic = comp.params.count("IC") ? std::stod(comp.params.at("IC")) : 0;
            circuit.add_capacitor(full_name, mapped_nodes[0], mapped_nodes[1], value, ic);
        } else if (comp.type == "L") {
            double ic = comp.params.count("IC") ? std::stod(comp.params.at("IC")) : 0;
            circuit.add_inductor(full_name, mapped_nodes[0], mapped_nodes[1], value, ic);
        } else if (comp.type == "V") {
            DCWaveform dc{value};
            circuit.add_voltage_source(full_name, mapped_nodes[0], mapped_nodes[1], dc);
        } else if (comp.type == "I") {
            DCWaveform dc{value};
            circuit.add_current_source(full_name, mapped_nodes[0], mapped_nodes[1], dc);
        } else if (comp.type == "D") {
            DiodeParams diode_params;
            circuit.add_diode(full_name, mapped_nodes[0], mapped_nodes[1], diode_params);
        } else if (comp.type == "M") {
            MOSFETParams mos_params;
            mos_params.w = comp.params.count("W") ? std::stod(comp.params.at("W")) : 100e-6;
            mos_params.l = comp.params.count("L") ? std::stod(comp.params.at("L")) : 10e-6;
            circuit.add_mosfet(full_name, mapped_nodes[0], mapped_nodes[1],
                               mapped_nodes[2], mos_params);
        } else if (comp.type == "SW") {
            SwitchParams sw_params;
            sw_params.ron = comp.params.count("RON") ? std::stod(comp.params.at("RON")) : 0.001;
            // Switches need 4 nodes: n1, n2, ctrl_pos, ctrl_neg
            if (mapped_nodes.size() >= 4) {
                circuit.add_switch(full_name, mapped_nodes[0], mapped_nodes[1],
                                   mapped_nodes[2], mapped_nodes[3], sw_params);
            }
        }
    }

    // Recursively expand nested subcircuit instances
    for (const auto& inst : subckt->instances()) {
        std::vector<std::string> nested_connections;
        for (const auto& conn : inst.connections) {
            nested_connections.push_back(
                make_internal_node(instance_name, conn, ports, port_connections));
        }

        expand_into(circuit,
                    instance_name + "." + inst.instance_name,
                    inst.subckt_name,
                    nested_connections,
                    params);
    }

    expansion_depth_--;
}

// =============================================================================
// Subcircuit Templates
// =============================================================================

namespace templates {

std::shared_ptr<SubcircuitDefinition> half_bridge(const std::string& name,
                                                   double rds_on,
                                                   double dead_time) {
    // Ports: VDC, GND, OUT, CTRL_H, CTRL_L
    auto subckt = std::make_shared<SubcircuitDefinition>(
        name, std::vector<std::string>{"VDC", "GND", "OUT", "CTRL_H", "CTRL_L"});

    // High-side switch
    subckt->add_switch("SH", "VDC", "OUT", "CTRL_H", rds_on);

    // Low-side switch
    subckt->add_switch("SL", "OUT", "GND", "CTRL_L", rds_on);

    // Body diodes (simplified as ideal diodes)
    subckt->add_diode("DH", "OUT", "VDC");
    subckt->add_diode("DL", "GND", "OUT");

    subckt->set_parameter("RDS_ON", rds_on);
    subckt->set_parameter("DEAD_TIME", dead_time);

    return subckt;
}

std::shared_ptr<SubcircuitDefinition> full_bridge(const std::string& name, double rds_on) {
    // Ports: VDC, GND, OUTA, OUTB, CTRL_AH, CTRL_AL, CTRL_BH, CTRL_BL
    auto subckt = std::make_shared<SubcircuitDefinition>(
        name, std::vector<std::string>{"VDC", "GND", "OUTA", "OUTB",
                                       "CTRL_AH", "CTRL_AL", "CTRL_BH", "CTRL_BL"});

    // Leg A
    subckt->add_switch("SAH", "VDC", "OUTA", "CTRL_AH", rds_on);
    subckt->add_switch("SAL", "OUTA", "GND", "CTRL_AL", rds_on);
    subckt->add_diode("DAH", "OUTA", "VDC");
    subckt->add_diode("DAL", "GND", "OUTA");

    // Leg B
    subckt->add_switch("SBH", "VDC", "OUTB", "CTRL_BH", rds_on);
    subckt->add_switch("SBL", "OUTB", "GND", "CTRL_BL", rds_on);
    subckt->add_diode("DBH", "OUTB", "VDC");
    subckt->add_diode("DBL", "GND", "OUTB");

    subckt->set_parameter("RDS_ON", rds_on);

    return subckt;
}

std::shared_ptr<SubcircuitDefinition> buck_output(const std::string& name, double L, double C) {
    // Ports: SW, GND, VOUT
    auto subckt = std::make_shared<SubcircuitDefinition>(
        name, std::vector<std::string>{"SW", "GND", "VOUT"});

    subckt->add_inductor("L", "SW", "VOUT", L);
    subckt->add_capacitor("C", "VOUT", "GND", C);

    subckt->set_parameter("L", L);
    subckt->set_parameter("C", C);

    return subckt;
}

std::shared_ptr<SubcircuitDefinition> lc_filter(const std::string& name, double L, double C) {
    // Ports: IN, GND, OUT
    auto subckt = std::make_shared<SubcircuitDefinition>(
        name, std::vector<std::string>{"IN", "GND", "OUT"});

    subckt->add_inductor("L", "IN", "OUT", L);
    subckt->add_capacitor("C", "OUT", "GND", C);

    subckt->set_parameter("L", L);
    subckt->set_parameter("C", C);

    return subckt;
}

std::shared_ptr<SubcircuitDefinition> rc_snubber(const std::string& name, double R, double C) {
    // Ports: N1, N2
    auto subckt = std::make_shared<SubcircuitDefinition>(
        name, std::vector<std::string>{"N1", "N2"});

    subckt->add_resistor("R", "N1", "MID", R);
    subckt->add_capacitor("C", "MID", "N2", C);

    subckt->set_parameter("R", R);
    subckt->set_parameter("C", C);

    return subckt;
}

}  // namespace templates

}  // namespace pulsim::parser
