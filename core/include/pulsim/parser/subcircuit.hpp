#pragma once

#include "pulsim/circuit.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::parser {

// =============================================================================
// Subcircuit Support
// Allows hierarchical circuit definitions with .subckt/.ends blocks
// =============================================================================

// Subcircuit definition - template that can be instantiated
class SubcircuitDefinition {
public:
    SubcircuitDefinition(std::string name, std::vector<std::string> ports);

    const std::string& name() const { return name_; }
    const std::vector<std::string>& ports() const { return ports_; }

    // Add components to the subcircuit template
    void add_resistor(const std::string& name, const std::string& n1,
                      const std::string& n2, double value);
    void add_capacitor(const std::string& name, const std::string& n1,
                       const std::string& n2, double value, double ic = 0);
    void add_inductor(const std::string& name, const std::string& n1,
                      const std::string& n2, double value, double ic = 0);
    void add_voltage_source(const std::string& name, const std::string& np,
                            const std::string& nn, double value);
    void add_current_source(const std::string& name, const std::string& np,
                            const std::string& nn, double value);
    void add_diode(const std::string& name, const std::string& anode,
                   const std::string& cathode, const std::string& model = "");
    void add_mosfet(const std::string& name, const std::string& d, const std::string& g,
                    const std::string& s, const std::string& b,
                    const std::string& model = "", double w = 100e-6, double l = 10e-6);
    void add_switch(const std::string& name, const std::string& n1, const std::string& n2,
                    const std::string& control, double ron = 0.001);

    // Add nested subcircuit instance
    void add_subcircuit_instance(const std::string& instance_name,
                                 const std::string& subckt_name,
                                 const std::vector<std::string>& connections);

    // Set default parameter values
    void set_parameter(const std::string& name, double value);
    void set_parameter(const std::string& name, const std::string& value);

    // Internal component representation
    struct ComponentDef {
        std::string type;
        std::string name;
        std::vector<std::string> nodes;
        double value = 0;
        std::string model;
        std::unordered_map<std::string, std::string> params;
    };

    const std::vector<ComponentDef>& components() const { return components_; }

    // Nested subcircuit instances
    struct SubcircuitInstance {
        std::string instance_name;
        std::string subckt_name;
        std::vector<std::string> connections;
    };

    const std::vector<SubcircuitInstance>& instances() const { return instances_; }
    const std::unordered_map<std::string, std::string>& parameters() const { return parameters_; }

private:
    std::string name_;
    std::vector<std::string> ports_;
    std::vector<ComponentDef> components_;
    std::vector<SubcircuitInstance> instances_;
    std::unordered_map<std::string, std::string> parameters_;
};

// =============================================================================
// Subcircuit Library - Manages subcircuit definitions
// =============================================================================

class SubcircuitLibrary {
public:
    SubcircuitLibrary() = default;

    // Register a subcircuit definition
    void add(std::shared_ptr<SubcircuitDefinition> subckt);
    void add(const std::string& name, std::vector<std::string> ports);

    // Get a subcircuit definition
    SubcircuitDefinition* get(const std::string& name);
    const SubcircuitDefinition* get(const std::string& name) const;

    // Check if subcircuit exists
    bool exists(const std::string& name) const;

    // List all subcircuits
    std::vector<std::string> list() const;

    // Clear all definitions
    void clear();

    // Load subcircuits from SPICE library file
    void load_library(const std::string& path);

private:
    std::unordered_map<std::string, std::shared_ptr<SubcircuitDefinition>> subcircuits_;
};

// =============================================================================
// Subcircuit Expander - Flattens hierarchy into single circuit
// =============================================================================

class SubcircuitExpander {
public:
    explicit SubcircuitExpander(const SubcircuitLibrary& library);

    // Expand a subcircuit instance into a circuit
    // instance_name: prefix for internal nodes (e.g., "X1")
    // subckt_name: name of subcircuit to instantiate
    // port_connections: actual node names connected to ports
    // params: parameter overrides
    void expand_into(Circuit& circuit,
                     const std::string& instance_name,
                     const std::string& subckt_name,
                     const std::vector<std::string>& port_connections,
                     const std::unordered_map<std::string, std::string>& params = {});

    // Recursively expand all subcircuit instances in a circuit
    void expand_all(Circuit& circuit);

    // Get expansion errors
    const std::vector<std::string>& errors() const { return errors_; }

private:
    const SubcircuitLibrary& library_;
    std::vector<std::string> errors_;
    int expansion_depth_ = 0;
    static constexpr int max_depth_ = 100;  // Prevent infinite recursion

    std::string make_internal_node(const std::string& instance,
                                   const std::string& node,
                                   const std::vector<std::string>& ports,
                                   const std::vector<std::string>& connections);

    double evaluate_parameter(const std::string& expr,
                              const std::unordered_map<std::string, std::string>& params,
                              const std::unordered_map<std::string, std::string>& defaults);
};

// =============================================================================
// Common Subcircuit Templates
// =============================================================================

namespace templates {

// Half-bridge leg with high and low MOSFETs
std::shared_ptr<SubcircuitDefinition> half_bridge(
    const std::string& name = "HALFBRIDGE",
    double rds_on = 0.01,
    double dead_time = 100e-9);

// Full-bridge (H-bridge) with 4 MOSFETs
std::shared_ptr<SubcircuitDefinition> full_bridge(
    const std::string& name = "FULLBRIDGE",
    double rds_on = 0.01);

// Buck converter output stage
std::shared_ptr<SubcircuitDefinition> buck_output(
    const std::string& name = "BUCKOUT",
    double L = 100e-6,
    double C = 100e-6);

// LC filter
std::shared_ptr<SubcircuitDefinition> lc_filter(
    const std::string& name = "LCFILTER",
    double L = 100e-6,
    double C = 100e-6);

// RC snubber
std::shared_ptr<SubcircuitDefinition> rc_snubber(
    const std::string& name = "SNUBBER",
    double R = 10,
    double C = 100e-9);

}  // namespace templates

}  // namespace pulsim::parser
