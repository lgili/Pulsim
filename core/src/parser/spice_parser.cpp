#include "pulsim/parser/spice_parser.hpp"
#include "pulsim/parser.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>

namespace pulsim::parser {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Note: split() is defined for future use but currently unused
// std::vector<std::string> split(const std::string& s, char delim = ' ');

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::string current;
    bool in_parens = false;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"' && !in_parens) {
            in_quotes = !in_quotes;
            current += c;
        } else if (c == '(' && !in_quotes) {
            in_parens = true;
            current += c;
        } else if (c == ')' && !in_quotes) {
            in_parens = false;
            current += c;
        } else if ((c == ' ' || c == '\t') && !in_parens && !in_quotes) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

// =============================================================================
// SpiceParser Implementation
// =============================================================================

SpiceParser::SpiceParser(SpiceParserOptions options)
    : options_(std::move(options))
{}

SpiceNetlist SpiceParser::parse_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        SpiceNetlist result;
        result.errors.push_back("Cannot open file: " + path.string());
        return result;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_string(buffer.str(), path.string());
}

SpiceNetlist SpiceParser::parse_string(const std::string& content,
                                        const std::string& /*source_name*/) {
    SpiceNetlist result;
    errors_.clear();
    warnings_.clear();

    // Split into lines
    std::vector<std::string> raw_lines;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        raw_lines.push_back(line);
    }

    // Handle line continuations (lines starting with +)
    auto lines = join_continuation_lines(raw_lines);

    // First line is title
    if (!lines.empty()) {
        result.title = trim(lines[0]);
    }

    // Parse remaining lines
    parse_lines(lines, result);

    result.errors = errors_;
    result.warnings = warnings_;

    return result;
}

std::vector<std::string> SpiceParser::join_continuation_lines(
    const std::vector<std::string>& lines) {

    std::vector<std::string> result;
    std::string current;

    for (const auto& line : lines) {
        std::string trimmed = trim(line);

        // Skip empty lines and comments (except first line which is title)
        if (result.empty()) {
            result.push_back(trimmed);
            continue;
        }

        if (trimmed.empty()) continue;
        if (trimmed[0] == '*') continue;  // Comment
        if (trimmed[0] == ';') continue;  // Alternative comment

        // Continuation line
        if (trimmed[0] == '+') {
            if (!result.empty()) {
                result.back() += " " + trimmed.substr(1);
            }
        } else {
            result.push_back(trimmed);
        }
    }

    return result;
}

void SpiceParser::parse_lines(const std::vector<std::string>& lines,
                               SpiceNetlist& result) {
    for (size_t i = 1; i < lines.size(); ++i) {  // Skip title
        const std::string& line = lines[i];
        if (line.empty()) continue;

        char first = options_.case_sensitive ? line[0] : std::toupper(line[0]);

        if (first == '.') {
            parse_directive(line, result);
        } else if (first == '*' || first == ';') {
            // Comment - skip
        } else {
            parse_component(line, result);
        }
    }
}

void SpiceParser::parse_component(const std::string& line, SpiceNetlist& result) {
    auto tokens = tokenize(line);
    if (tokens.empty()) return;

    SpiceComponent comp;
    std::string name = tokens[0];
    char type_char = options_.case_sensitive ? name[0] : std::toupper(name[0]);

    comp.type = std::string(1, type_char);
    comp.name = name.substr(1);  // Remove type prefix

    switch (type_char) {
        case 'R':  // Resistor
        case 'C':  // Capacitor
        case 'L':  // Inductor
            if (tokens.size() < 4) {
                errors_.push_back("Invalid " + comp.type + " definition: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2]};
            comp.value = parse_value(tokens[3]);

            // Check for model or additional params
            for (size_t i = 4; i < tokens.size(); ++i) {
                if (tokens[i].find('=') != std::string::npos) {
                    auto eq = tokens[i].find('=');
                    comp.params[tokens[i].substr(0, eq)] = tokens[i].substr(eq + 1);
                } else if (comp.model.empty()) {
                    comp.model = tokens[i];
                }
            }
            break;

        case 'V':  // Voltage source
        case 'I':  // Current source
            if (tokens.size() < 3) {
                errors_.push_back("Invalid source definition: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2]};

            // Parse source specification
            if (tokens.size() >= 4) {
                std::string spec = to_upper(tokens[3]);

                if (spec == "DC" || spec == "AC") {
                    if (tokens.size() >= 5) {
                        comp.value = parse_value(tokens[4]);
                    }
                } else if (spec == "PULSE" || spec == "SIN" || spec == "EXP" ||
                           spec == "SFFM" || spec == "PWL") {
                    // Extract parameters from parentheses
                    std::vector<std::string> wf_params;
                    bool in_wf = false;
                    for (size_t i = 3; i < tokens.size(); ++i) {
                        std::string t = tokens[i];
                        if (t.find('(') != std::string::npos) {
                            in_wf = true;
                            t = t.substr(t.find('(') + 1);
                        }
                        if (t.find(')') != std::string::npos) {
                            t = t.substr(0, t.find(')'));
                            if (!t.empty()) wf_params.push_back(t);
                            break;
                        }
                        if (in_wf && !t.empty()) {
                            wf_params.push_back(t);
                        }
                    }
                    comp.waveform = parse_waveform(spec, wf_params);
                } else {
                    // Assume DC value
                    comp.value = parse_value(tokens[3]);
                }
            }
            break;

        case 'D':  // Diode
            if (tokens.size() < 3) {
                errors_.push_back("Invalid diode definition: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2]};
            if (tokens.size() >= 4) {
                comp.model = tokens[3];
            }
            break;

        case 'Q':  // BJT
            if (tokens.size() < 5) {
                errors_.push_back("Invalid BJT definition: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2], tokens[3]};  // C, B, E
            comp.model = tokens[4];
            break;

        case 'M':  // MOSFET
            if (tokens.size() < 6) {
                errors_.push_back("Invalid MOSFET definition: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2], tokens[3], tokens[4]};  // D, G, S, B
            comp.model = tokens[5];

            // Parse W, L parameters
            for (size_t i = 6; i < tokens.size(); ++i) {
                auto eq = tokens[i].find('=');
                if (eq != std::string::npos) {
                    std::string key = to_upper(tokens[i].substr(0, eq));
                    std::string val = tokens[i].substr(eq + 1);
                    comp.params[key] = val;
                }
            }
            break;

        case 'X':  // Subcircuit instance
            if (tokens.size() < 3) {
                errors_.push_back("Invalid subcircuit instance: " + line);
                return;
            }
            // Last token is subcircuit name, rest are nodes
            comp.model = tokens.back();
            for (size_t i = 1; i < tokens.size() - 1; ++i) {
                comp.nodes.push_back(tokens[i]);
            }
            break;

        case 'K':  // Coupled inductors
            comp.type = "K";
            if (tokens.size() < 4) {
                errors_.push_back("Invalid coupled inductor: " + line);
                return;
            }
            comp.params["L1"] = tokens[1];
            comp.params["L2"] = tokens[2];
            comp.value = parse_value(tokens[3]);  // Coupling coefficient
            break;

        case 'S':  // Voltage-controlled switch
        case 'W':  // Current-controlled switch
            if (tokens.size() < 5) {
                errors_.push_back("Invalid switch definition: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2], tokens[3], tokens[4]};
            if (tokens.size() >= 6) {
                comp.model = tokens[5];
            }
            break;

        case 'E':  // VCVS
        case 'G':  // VCCS
        case 'F':  // CCCS
        case 'H':  // CCVS
            if (tokens.size() < 6) {
                errors_.push_back("Invalid controlled source: " + line);
                return;
            }
            comp.nodes = {tokens[1], tokens[2], tokens[3], tokens[4]};
            comp.value = parse_value(tokens[5]);
            break;

        default:
            warnings_.push_back("Unknown component type: " + comp.type + " in: " + line);
            return;
    }

    result.components.push_back(comp);
}

void SpiceParser::parse_directive(const std::string& line, SpiceNetlist& result) {
    auto tokens = tokenize(line);
    if (tokens.empty()) return;

    std::string directive = to_upper(tokens[0]);

    if (directive == ".MODEL") {
        parse_model(line, result);
    } else if (directive == ".SUBCKT") {
        // This requires multi-line parsing - simplified here
        SpiceSubcircuit subckt;
        if (tokens.size() >= 2) {
            subckt.name = tokens[1];
            for (size_t i = 2; i < tokens.size(); ++i) {
                if (tokens[i].find('=') == std::string::npos) {
                    subckt.ports.push_back(tokens[i]);
                } else {
                    auto eq = tokens[i].find('=');
                    subckt.params[tokens[i].substr(0, eq)] = tokens[i].substr(eq + 1);
                }
            }
        }
        result.subcircuits.push_back(subckt);
    } else if (directive == ".ENDS") {
        // End subcircuit - handled in multi-line parsing
    } else if (directive == ".TRAN") {
        parse_simulation(line, result);
    } else if (directive == ".AC") {
        parse_simulation(line, result);
    } else if (directive == ".DC") {
        parse_simulation(line, result);
    } else if (directive == ".OP") {
        SpiceSimulation sim;
        sim.type = SpiceSimulation::Type::Op;
        result.simulations.push_back(sim);
    } else if (directive == ".INCLUDE" || directive == ".INC" || directive == ".LIB") {
        if (tokens.size() >= 2) {
            std::string path = tokens[1];
            // Remove quotes if present
            if (path.front() == '"' && path.back() == '"') {
                path = path.substr(1, path.size() - 2);
            }
            result.includes.push_back(path);
        }
    } else if (directive == ".PARAM") {
        // Parameter definition
        for (size_t i = 1; i < tokens.size(); ++i) {
            auto eq = tokens[i].find('=');
            if (eq != std::string::npos) {
                result.options[tokens[i].substr(0, eq)] = tokens[i].substr(eq + 1);
            }
        }
    } else if (directive == ".OPTION" || directive == ".OPTIONS") {
        for (size_t i = 1; i < tokens.size(); ++i) {
            auto eq = tokens[i].find('=');
            if (eq != std::string::npos) {
                result.options[to_upper(tokens[i].substr(0, eq))] = tokens[i].substr(eq + 1);
            } else {
                result.options[to_upper(tokens[i])] = "1";
            }
        }
    } else if (directive == ".END") {
        // End of netlist
    } else if (directive == ".GLOBAL") {
        // Global nodes - typically ground
    } else if (directive == ".IC") {
        // Initial conditions
        for (size_t i = 1; i < tokens.size(); ++i) {
            // V(node)=value format
            std::regex ic_regex(R"(V\((\w+)\)=([^\s]+))", std::regex::icase);
            std::smatch match;
            if (std::regex_search(tokens[i], match, ic_regex)) {
                result.options["IC_" + match[1].str()] = match[2].str();
            }
        }
    } else {
        if (options_.strict) {
            errors_.push_back("Unknown directive: " + directive);
        } else {
            warnings_.push_back("Ignoring unknown directive: " + directive);
        }
    }
}

void SpiceParser::parse_model(const std::string& line, SpiceNetlist& result) {
    // .MODEL name type (params...)
    auto tokens = tokenize(line);
    if (tokens.size() < 3) {
        errors_.push_back("Invalid .MODEL statement: " + line);
        return;
    }

    SpiceModel model;
    model.name = tokens[1];
    model.type = to_upper(tokens[2]);

    // Parse parameters in parentheses or key=value pairs
    std::string params_str;
    size_t paren_start = line.find('(');
    size_t paren_end = line.rfind(')');
    if (paren_start != std::string::npos && paren_end != std::string::npos) {
        params_str = line.substr(paren_start + 1, paren_end - paren_start - 1);
    } else {
        // Parameters after type
        for (size_t i = 3; i < tokens.size(); ++i) {
            params_str += tokens[i] + " ";
        }
    }

    // Parse individual parameters
    auto param_tokens = tokenize(params_str);
    for (const auto& pt : param_tokens) {
        auto eq = pt.find('=');
        if (eq != std::string::npos) {
            std::string key = to_upper(pt.substr(0, eq));
            double value = parse_value(pt.substr(eq + 1));
            model.params[key] = value;
        }
    }

    result.models.push_back(model);
    model_library_[model.name] = model;
}

void SpiceParser::parse_simulation(const std::string& line, SpiceNetlist& result) {
    auto tokens = tokenize(line);
    if (tokens.empty()) return;

    SpiceSimulation sim;
    std::string cmd = to_upper(tokens[0]);

    if (cmd == ".TRAN") {
        sim.type = SpiceSimulation::Type::Tran;
        if (tokens.size() >= 3) {
            sim.step = parse_value(tokens[1]);
            sim.stop = parse_value(tokens[2]);
            if (tokens.size() >= 4) {
                sim.start = parse_value(tokens[3]);
            }
            if (tokens.size() >= 5) {
                sim.max_step = parse_value(tokens[4]);
            }
        }
    } else if (cmd == ".AC") {
        sim.type = SpiceSimulation::Type::Ac;
        if (tokens.size() >= 5) {
            sim.sweep_type = to_lower(tokens[1]);  // dec, oct, lin
            sim.points = std::stoi(tokens[2]);
            sim.start_freq = parse_value(tokens[3]);
            sim.stop_freq = parse_value(tokens[4]);
        }
    } else if (cmd == ".DC") {
        sim.type = SpiceSimulation::Type::Dc;
        if (tokens.size() >= 5) {
            sim.source = tokens[1];
            sim.dc_start = parse_value(tokens[2]);
            sim.dc_stop = parse_value(tokens[3]);
            sim.dc_step = parse_value(tokens[4]);
        }
    }

    result.simulations.push_back(sim);
}

double SpiceParser::parse_value(const std::string& str) {
    if (str.empty()) return 0.0;

    std::string s = trim(str);
    double multiplier = 1.0;

    // Check for SPICE suffixes
    char last = std::toupper(s.back());
    std::string num_part = s;

    // Handle metric suffixes
    static const std::unordered_map<char, double> suffixes = {
        {'T', 1e12}, {'G', 1e9}, {'X', 1e6}, {'K', 1e3},
        {'M', 1e-3}, {'U', 1e-6}, {'N', 1e-9}, {'P', 1e-12}, {'F', 1e-15}
    };

    // Handle MEG specially (not M for milli)
    if (s.size() >= 3) {
        std::string last3 = to_upper(s.substr(s.size() - 3));
        if (last3 == "MEG") {
            multiplier = 1e6;
            num_part = s.substr(0, s.size() - 3);
        } else if (last3 == "MIL") {
            multiplier = 25.4e-6;  // mils to meters
            num_part = s.substr(0, s.size() - 3);
        }
    }

    if (multiplier == 1.0 && suffixes.count(last)) {
        multiplier = suffixes.at(last);
        num_part = s.substr(0, s.size() - 1);
    }

    try {
        return std::stod(num_part) * multiplier;
    } catch (...) {
        warnings_.push_back("Cannot parse value: " + str);
        return 0.0;
    }
}

SpiceWaveform SpiceParser::parse_waveform(const std::string& type,
                                          const std::vector<std::string>& params) {
    SpiceWaveform wf;
    std::string t = to_upper(type);

    if (t == "PULSE") {
        wf.type = SpiceWaveform::Type::Pulse;
        // PULSE(V1 V2 TD TR TF PW PER)
    } else if (t == "SIN") {
        wf.type = SpiceWaveform::Type::Sin;
        // SIN(VO VA FREQ TD THETA)
    } else if (t == "EXP") {
        wf.type = SpiceWaveform::Type::Exp;
    } else if (t == "SFFM") {
        wf.type = SpiceWaveform::Type::Sffm;
    } else if (t == "PWL") {
        wf.type = SpiceWaveform::Type::Pwl;
    } else {
        wf.type = SpiceWaveform::Type::DC;
    }

    for (const auto& p : params) {
        wf.params.push_back(parse_value(p));
    }

    return wf;
}

// =============================================================================
// Conversion to Pulsim Circuit
// =============================================================================

std::pair<Circuit, SimulationOptions> SpiceParser::to_circuit(const SpiceNetlist& netlist) {
    Circuit circuit;
    circuit.set_name(netlist.title);

    SimulationOptions options;

    // Convert components
    for (const auto& comp : netlist.components) {
        add_component_to_circuit(comp, circuit);
    }

    // Convert simulation commands
    for (const auto& sim : netlist.simulations) {
        if (sim.type == SpiceSimulation::Type::Tran) {
            options.tstop = sim.stop;
            options.dt = sim.step > 0 ? sim.step : sim.stop / 1000;
            if (sim.max_step > 0) {
                options.dtmax = sim.max_step;
            }
        }
    }

    // Apply options
    if (netlist.options.count("ABSTOL")) {
        options.abstol = std::stod(netlist.options.at("ABSTOL"));
    }
    if (netlist.options.count("RELTOL")) {
        options.reltol = std::stod(netlist.options.at("RELTOL"));
    }

    return {circuit, options};
}

void SpiceParser::add_component_to_circuit(const SpiceComponent& comp, Circuit& circuit) {
    const auto& n = comp.nodes;

    if (comp.type == "R") {
        circuit.add_resistor(comp.type + comp.name, n[0], n[1], comp.value);
    } else if (comp.type == "C") {
        double ic = 0;
        if (comp.params.count("IC")) {
            ic = parse_value(comp.params.at("IC"));
        }
        circuit.add_capacitor(comp.type + comp.name, n[0], n[1], comp.value, ic);
    } else if (comp.type == "L") {
        double ic = 0;
        if (comp.params.count("IC")) {
            ic = parse_value(comp.params.at("IC"));
        }
        circuit.add_inductor(comp.type + comp.name, n[0], n[1], comp.value, ic);
    } else if (comp.type == "V") {
        if (comp.waveform) {
            // Convert waveform
            auto& wf = *comp.waveform;
            if (wf.type == SpiceWaveform::Type::Pulse && wf.params.size() >= 7) {
                PulseWaveform pulse{
                    wf.params[0],  // v1
                    wf.params[1],  // v2
                    wf.params[2],  // td
                    wf.params[3],  // tr
                    wf.params[4],  // tf
                    wf.params[5],  // pw
                    wf.params[6]   // period
                };
                circuit.add_voltage_source(comp.type + comp.name, n[0], n[1], pulse);
            } else if (wf.type == SpiceWaveform::Type::Sin && wf.params.size() >= 3) {
                SineWaveform sine{
                    wf.params[0],  // offset
                    wf.params[1],  // amplitude
                    wf.params[2],  // frequency
                    wf.params.size() > 3 ? wf.params[3] : 0.0,  // delay
                    wf.params.size() > 4 ? wf.params[4] : 0.0   // damping
                };
                circuit.add_voltage_source(comp.type + comp.name, n[0], n[1], sine);
            } else if (wf.type == SpiceWaveform::Type::Pwl && wf.params.size() >= 2) {
                PWLWaveform pwl;
                // params contains alternating time, value pairs
                for (size_t i = 0; i + 1 < wf.params.size(); i += 2) {
                    pwl.points.emplace_back(wf.params[i], wf.params[i + 1]);
                }
                circuit.add_voltage_source(comp.type + comp.name, n[0], n[1], pwl);
            } else {
                DCWaveform dc{comp.value};
                circuit.add_voltage_source(comp.type + comp.name, n[0], n[1], dc);
            }
        } else {
            DCWaveform dc{comp.value};
            circuit.add_voltage_source(comp.type + comp.name, n[0], n[1], dc);
        }
    } else if (comp.type == "I") {
        DCWaveform dc{comp.value};
        circuit.add_current_source(comp.type + comp.name, n[0], n[1], dc);
    } else if (comp.type == "D") {
        DiodeParams params;
        // Could parse model parameters if model library is available
        circuit.add_diode(comp.type + comp.name, n[0], n[1], params);
    } else if (comp.type == "M") {
        // MOSFET: D, G, S, B (we use D, G, S - ignore bulk for now)
        MOSFETParams params;
        if (comp.params.count("W")) {
            params.w = parse_value(comp.params.at("W"));
        }
        if (comp.params.count("L")) {
            params.l = parse_value(comp.params.at("L"));
        }
        // Note: Circuit API doesn't take bulk node - only D, G, S
        circuit.add_mosfet(comp.type + comp.name, n[0], n[1], n[2], params);
    } else if (comp.type == "X") {
        // Subcircuit instance - would need expansion
        warnings_.push_back("Subcircuit instance not fully supported: X" + comp.name);
    } else {
        warnings_.push_back("Component type not converted: " + comp.type + comp.name);
    }
}

std::pair<Circuit, SimulationOptions> SpiceParser::load(const std::filesystem::path& path) {
    auto netlist = parse_file(path);
    return to_circuit(netlist);
}

std::pair<Circuit, SimulationOptions> SpiceParser::load_string(const std::string& content) {
    auto netlist = parse_string(content);
    return to_circuit(netlist);
}

// =============================================================================
// LTspice ASC Parser
// =============================================================================

LTspiceSchematic LTspiceParser::parse_asc(const std::filesystem::path& path) {
    LTspiceSchematic result;

    std::ifstream file(path);
    if (!file.is_open()) {
        return result;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (starts_with(line, "SYMBOL")) {
            parse_symbol(line, result);
        } else if (starts_with(line, "WIRE")) {
            parse_wire(line, result);
        } else if (starts_with(line, "TEXT")) {
            result.text_items.push_back(line);
        } else if (starts_with(line, "SYMATTR")) {
            // Attribute for previous symbol
            if (!result.symbols.empty()) {
                auto tokens = tokenize(line);
                if (tokens.size() >= 3) {
                    result.symbols.back().attributes[tokens[1]] = tokens[2];
                }
            }
        }
    }

    return result;
}

void LTspiceParser::parse_symbol(const std::string& line, LTspiceSchematic& result) {
    // SYMBOL res 128 64 R90
    auto tokens = tokenize(line);
    if (tokens.size() < 4) return;

    LTspiceSymbol sym;
    sym.type = tokens[1];
    sym.x = std::stod(tokens[2]);
    sym.y = std::stod(tokens[3]);

    if (tokens.size() >= 5) {
        std::string rot = tokens[4];
        if (rot[0] == 'R') {
            sym.rotation = std::stoi(rot.substr(1));
        } else if (rot[0] == 'M') {
            sym.mirror = true;
            sym.rotation = std::stoi(rot.substr(1));
        }
    }

    result.symbols.push_back(sym);
}

void LTspiceParser::parse_wire(const std::string& line, LTspiceSchematic& result) {
    // WIRE x1 y1 x2 y2
    auto tokens = tokenize(line);
    if (tokens.size() < 5) return;

    LTspiceWire wire;
    wire.x1 = std::stod(tokens[1]);
    wire.y1 = std::stod(tokens[2]);
    wire.x2 = std::stod(tokens[3]);
    wire.y2 = std::stod(tokens[4]);

    result.wires.push_back(wire);
}

SpiceNetlist LTspiceParser::to_netlist(const LTspiceSchematic& /*schematic*/) {
    // TODO: implement full schematic conversion
    SpiceNetlist result;
    result.title = "Converted from LTspice schematic";

    // This is a simplified conversion - real implementation would need
    // proper connectivity extraction from wire endpoints
    warnings_.push_back("LTspice schematic conversion is simplified");

    return result;
}

std::pair<Circuit, SimulationOptions> LTspiceParser::load(const std::filesystem::path& path) {
    auto schematic = parse_asc(path);
    auto netlist = to_netlist(schematic);

    SpiceParser parser;
    return parser.to_circuit(netlist);
}

// =============================================================================
// Utility Functions
// =============================================================================

NetlistFormat detect_format(const std::filesystem::path& path) {
    std::string ext = to_lower(path.extension().string());

    if (ext == ".cir" || ext == ".sp" || ext == ".net" || ext == ".spi" || ext == ".spice") {
        return NetlistFormat::SpiceCir;
    } else if (ext == ".asc") {
        return NetlistFormat::LTspiceAsc;
    } else if (ext == ".json") {
        return NetlistFormat::Json;
    } else if (ext == ".yaml" || ext == ".yml") {
        return NetlistFormat::Yaml;
    }

    return NetlistFormat::Unknown;
}

std::pair<Circuit, SimulationOptions> load_netlist(const std::filesystem::path& path) {
    auto format = detect_format(path);

    switch (format) {
        case NetlistFormat::SpiceCir: {
            SpiceParser parser;
            return parser.load(path);
        }
        case NetlistFormat::LTspiceAsc: {
            LTspiceParser parser;
            return parser.load(path);
        }
        case NetlistFormat::Json: {
            auto result = NetlistParser::parse_file(path);
            if (!result) {
                throw std::runtime_error("JSON parse error: " + result.error().message);
            }
            SimulationOptions options;
            auto opts_result = NetlistParser::parse_simulation_options(path);
            if (opts_result) {
                options = *opts_result;
            }
            return {*result, options};
        }
        case NetlistFormat::Yaml: {
            // Would call YAML parser
            throw std::runtime_error("YAML format not yet implemented");
        }
        default:
            throw std::runtime_error("Unknown netlist format: " + path.string());
    }
}

}  // namespace pulsim::parser
