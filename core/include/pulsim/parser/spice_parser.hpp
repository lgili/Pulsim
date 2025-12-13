#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/simulation.hpp"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pulsim::parser {

// =============================================================================
// SPICE Netlist Parser
// Supports .cir, .sp, .net, and LTspice .asc formats
// =============================================================================

// Waveform specification from SPICE
struct SpiceWaveform {
    enum class Type { DC, Pulse, Sin, Exp, Sffm, Pwl };
    Type type = Type::DC;
    std::vector<double> params;  // Type-specific parameters
};

// Parsed component from SPICE netlist
struct SpiceComponent {
    std::string type;      // R, C, L, V, I, D, M, Q, X, etc.
    std::string name;      // Component name (without prefix)
    std::vector<std::string> nodes;
    double value = 0.0;
    std::string model;     // Model name reference
    std::unordered_map<std::string, std::string> params;
    std::optional<SpiceWaveform> waveform;
};

// Parsed model statement
struct SpiceModel {
    std::string name;
    std::string type;      // D, NMF, PMF, NJF, PJF, NPN, PNP, NMOS, PMOS, SW, etc.
    std::unordered_map<std::string, double> params;
};

// Parsed subcircuit definition
struct SpiceSubcircuit {
    std::string name;
    std::vector<std::string> ports;
    std::vector<SpiceComponent> components;
    std::vector<std::string> models;  // Model names used
    std::unordered_map<std::string, std::string> params;  // Default parameters
};

// Simulation command
struct SpiceSimulation {
    enum class Type { Op, Dc, Ac, Tran, Noise, Tf };
    Type type = Type::Tran;

    // Transient parameters
    double step = 0;
    double stop = 0;
    double start = 0;
    double max_step = 0;

    // AC parameters
    std::string sweep_type;  // dec, oct, lin
    int points = 0;
    double start_freq = 0;
    double stop_freq = 0;

    // DC parameters
    std::string source;
    double dc_start = 0;
    double dc_stop = 0;
    double dc_step = 0;
};

// Parse result
struct SpiceNetlist {
    std::string title;
    std::vector<SpiceComponent> components;
    std::vector<SpiceModel> models;
    std::vector<SpiceSubcircuit> subcircuits;
    std::vector<SpiceSimulation> simulations;
    std::vector<std::string> includes;  // .include files
    std::unordered_map<std::string, std::string> options;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

// Parser options
struct SpiceParserOptions {
    bool strict = false;              // Fail on unknown directives
    bool case_sensitive = false;      // SPICE is traditionally case-insensitive
    bool auto_ground = true;          // Auto-connect node '0' as ground
    bool expand_subcircuits = true;   // Expand X instances inline
    std::vector<std::filesystem::path> include_paths;  // Search paths for .include
};

// =============================================================================
// SPICE Parser Class
// =============================================================================

class SpiceParser {
public:
    explicit SpiceParser(SpiceParserOptions options = {});

    // Parse from file
    SpiceNetlist parse_file(const std::filesystem::path& path);

    // Parse from string
    SpiceNetlist parse_string(const std::string& content,
                              const std::string& source_name = "<string>");

    // Convert parsed netlist to Pulsim circuit
    std::pair<Circuit, SimulationOptions> to_circuit(const SpiceNetlist& netlist);

    // Convenience: parse and convert in one step
    std::pair<Circuit, SimulationOptions> load(const std::filesystem::path& path);
    std::pair<Circuit, SimulationOptions> load_string(const std::string& content);

    // Get last parse errors/warnings
    const std::vector<std::string>& errors() const { return errors_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    SpiceParserOptions options_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;
    std::unordered_map<std::string, SpiceModel> model_library_;
    std::unordered_map<std::string, SpiceSubcircuit> subcircuit_library_;

    // Parsing helpers
    void parse_lines(const std::vector<std::string>& lines, SpiceNetlist& result);
    void parse_component(const std::string& line, SpiceNetlist& result);
    void parse_directive(const std::string& line, SpiceNetlist& result);
    void parse_model(const std::string& line, SpiceNetlist& result);
    void parse_subcircuit(std::vector<std::string>::const_iterator& it,
                          std::vector<std::string>::const_iterator end,
                          SpiceNetlist& result);
    void parse_simulation(const std::string& line, SpiceNetlist& result);

    // Value parsing
    double parse_value(const std::string& str);
    SpiceWaveform parse_waveform(const std::string& type,
                                 const std::vector<std::string>& params);

    // Line continuation handling
    std::vector<std::string> join_continuation_lines(const std::vector<std::string>& lines);

    // Subcircuit expansion
    void expand_subcircuit_instance(const SpiceComponent& inst,
                                    const SpiceSubcircuit& subckt,
                                    SpiceNetlist& result);

    // Conversion helpers
    void add_component_to_circuit(const SpiceComponent& comp, Circuit& circuit);
    void apply_model(const std::string& model_name,
                     const SpiceComponent& comp,
                     Circuit& circuit);
};

// =============================================================================
// LTspice ASC Parser (Schematic Format)
// =============================================================================

struct LTspiceSymbol {
    std::string name;
    std::string type;
    double x = 0, y = 0;
    int rotation = 0;
    bool mirror = false;
    std::unordered_map<std::string, std::string> attributes;
};

struct LTspiceWire {
    double x1, y1, x2, y2;
};

struct LTspiceSchematic {
    std::vector<LTspiceSymbol> symbols;
    std::vector<LTspiceWire> wires;
    std::vector<std::string> text_items;
    std::string spice_directive;  // Embedded SPICE commands
};

class LTspiceParser {
public:
    // Parse .asc schematic file
    LTspiceSchematic parse_asc(const std::filesystem::path& path);

    // Convert schematic to SPICE netlist
    SpiceNetlist to_netlist(const LTspiceSchematic& schematic);

    // Convenience: load and convert
    std::pair<Circuit, SimulationOptions> load(const std::filesystem::path& path);

    // Get warnings
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    std::vector<std::string> warnings_;

    void parse_symbol(const std::string& line, LTspiceSchematic& result);
    void parse_wire(const std::string& line, LTspiceSchematic& result);
    void extract_connections(const LTspiceSchematic& schematic,
                             std::unordered_map<std::string, std::vector<std::string>>& node_map);
};

// =============================================================================
// Utility Functions
// =============================================================================

// Detect file format from extension
enum class NetlistFormat {
    Unknown,
    SpiceCir,     // .cir, .sp, .net, .spi
    LTspiceAsc,   // .asc
    Json,         // .json
    Yaml          // .yaml, .yml
};

NetlistFormat detect_format(const std::filesystem::path& path);

// Universal load function - detects format and parses
std::pair<Circuit, SimulationOptions> load_netlist(const std::filesystem::path& path);

}  // namespace pulsim::parser
