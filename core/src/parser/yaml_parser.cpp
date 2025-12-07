#include "pulsim/parser/yaml_parser.hpp"
#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace pulsim::parser {

// =============================================================================
// Simple YAML Parser (Subset)
// Supports basic key: value, lists, and nested objects
// For full YAML support, integrate yaml-cpp library
// =============================================================================

namespace {

std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

int get_indent(const std::string& line) {
    int indent = 0;
    for (char c : line) {
        if (c == ' ') indent++;
        else if (c == '\t') indent += 2;
        else break;
    }
    return indent;
}

std::pair<std::string, std::string> parse_key_value(const std::string& line) {
    auto colon = line.find(':');
    if (colon == std::string::npos) {
        return {"", trim(line)};
    }

    std::string key = trim(line.substr(0, colon));
    std::string value = trim(line.substr(colon + 1));

    // Remove quotes
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }
    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
        value = value.substr(1, value.size() - 2);
    }

    return {key, value};
}

bool is_list_item(const std::string& line) {
    std::string trimmed = trim(line);
    return !trimmed.empty() && trimmed[0] == '-';
}

double parse_number(const std::string& s) {
    if (s.empty()) return 0.0;

    std::string val = s;
    double multiplier = 1.0;

    // Handle SI suffixes
    char last = std::toupper(val.back());
    static const std::unordered_map<char, double> suffixes = {
        {'T', 1e12}, {'G', 1e9}, {'M', 1e6}, {'K', 1e3},
        {'m', 1e-3}, {'u', 1e-6}, {'n', 1e-9}, {'p', 1e-12}, {'f', 1e-15}
    };

    // Check for "meg" suffix
    if (val.size() >= 3) {
        std::string last3 = val.substr(val.size() - 3);
        std::transform(last3.begin(), last3.end(), last3.begin(), ::tolower);
        if (last3 == "meg") {
            multiplier = 1e6;
            val = val.substr(0, val.size() - 3);
        }
    }

    if (multiplier == 1.0) {
        // Check single-char suffix
        if (suffixes.count(last) || suffixes.count(std::tolower(last))) {
            char suffix = std::tolower(last);
            if (suffix == 'm' && val.size() > 1) {
                // Distinguish M (mega) from m (milli) based on context
                // In YAML we use lowercase m for milli
                multiplier = (last == 'M') ? 1e6 : 1e-3;
            } else if (suffixes.count(suffix)) {
                multiplier = suffixes.at(suffix);
            }
            val = val.substr(0, val.size() - 1);
        }
    }

    try {
        return std::stod(val) * multiplier;
    } catch (...) {
        return 0.0;
    }
}

// Simple recursive descent YAML parser state
struct YamlNode {
    std::string key;
    std::string value;
    std::vector<YamlNode> children;
    bool is_list = false;
};

YamlNode parse_yaml_tree(const std::vector<std::string>& lines, size_t& idx, int base_indent) {
    YamlNode root;
    root.key = "root";

    while (idx < lines.size()) {
        const std::string& line = lines[idx];
        int indent = get_indent(line);
        std::string trimmed = trim(line);

        if (trimmed.empty() || trimmed[0] == '#') {
            idx++;
            continue;
        }

        if (indent < base_indent) {
            break;  // Return to parent
        }

        if (indent > base_indent) {
            // This is a child of the previous node
            if (!root.children.empty()) {
                auto& parent = root.children.back();
                auto child = parse_yaml_tree(lines, idx, indent);
                for (auto& c : child.children) {
                    parent.children.push_back(c);
                }
            }
            continue;
        }

        // Same level
        if (is_list_item(trimmed)) {
            YamlNode item;
            item.is_list = true;
            std::string content = trim(trimmed.substr(1));  // Remove -

            auto [k, v] = parse_key_value(content);
            if (k.empty()) {
                item.value = v;
            } else {
                item.key = k;
                item.value = v;
            }
            root.children.push_back(item);
        } else {
            auto [k, v] = parse_key_value(trimmed);
            YamlNode node;
            node.key = k;
            node.value = v;
            root.children.push_back(node);
        }

        idx++;
    }

    return root;
}

}  // namespace

// =============================================================================
// YamlParser Implementation
// =============================================================================

YamlParser::YamlParser(YamlParserOptions options)
    : options_(std::move(options))
{}

std::pair<Circuit, SimulationOptions> YamlParser::load(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        errors_.push_back("Cannot open file: " + path.string());
        return {Circuit(), SimulationOptions()};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return load_string(buffer.str());
}

std::pair<Circuit, SimulationOptions> YamlParser::load_string(const std::string& content) {
    Circuit circuit;
    SimulationOptions options;
    errors_.clear();
    warnings_.clear();

    parse_yaml(content, circuit, options);

    return {circuit, options};
}

void YamlParser::parse_yaml(const std::string& content, Circuit& circuit, SimulationOptions& options) {
    // Split into lines
    std::vector<std::string> lines;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }

    // Parse YAML tree
    size_t idx = 0;
    auto root = parse_yaml_tree(lines, idx, 0);

    // Process parsed structure
    for (const auto& node : root.children) {
        if (node.key == "name") {
            circuit.set_name(node.value);
        } else if (node.key == "components") {
            // Parse component list
            for (const auto& comp_node : node.children) {
                if (!comp_node.is_list) continue;

                std::string type, name;
                std::vector<std::string> nodes;
                double value = 0;
                std::string model;
                std::unordered_map<std::string, std::string> params;

                // Find type from child nodes or inline
                for (const auto& attr : comp_node.children) {
                    if (attr.key == "type") type = attr.value;
                    else if (attr.key == "name") name = attr.value;
                    else if (attr.key == "value") value = parse_number(attr.value);
                    else if (attr.key == "model") model = attr.value;
                    else if (attr.key == "nodes") {
                        // Parse node list
                        for (const auto& n : attr.children) {
                            nodes.push_back(n.value.empty() ? n.key : n.value);
                        }
                        // Also handle inline format: nodes: [a, b]
                        if (attr.value.front() == '[') {
                            std::string nlist = attr.value.substr(1, attr.value.size() - 2);
                            std::istringstream ns(nlist);
                            std::string n;
                            while (std::getline(ns, n, ',')) {
                                nodes.push_back(trim(n));
                            }
                        }
                    } else {
                        params[attr.key] = attr.value;
                    }
                }

                // Also check if type is the key itself (compact format)
                if (type.empty() && !comp_node.key.empty()) {
                    type = comp_node.key;
                }

                // Add component
                std::transform(type.begin(), type.end(), type.begin(), ::toupper);

                if (type == "R" || type == "RESISTOR") {
                    if (nodes.size() >= 2) {
                        circuit.add_resistor(name, nodes[0], nodes[1], value);
                    }
                } else if (type == "C" || type == "CAPACITOR") {
                    double ic = params.count("ic") ? parse_number(params["ic"]) : 0;
                    if (nodes.size() >= 2) {
                        circuit.add_capacitor(name, nodes[0], nodes[1], value, ic);
                    }
                } else if (type == "L" || type == "INDUCTOR") {
                    double ic = params.count("ic") ? parse_number(params["ic"]) : 0;
                    if (nodes.size() >= 2) {
                        circuit.add_inductor(name, nodes[0], nodes[1], value, ic);
                    }
                } else if (type == "V" || type == "VOLTAGE") {
                    if (nodes.size() >= 2) {
                        DCWaveform dc{value};
                        circuit.add_voltage_source(name, nodes[0], nodes[1], dc);
                    }
                } else if (type == "I" || type == "CURRENT") {
                    if (nodes.size() >= 2) {
                        DCWaveform dc{value};
                        circuit.add_current_source(name, nodes[0], nodes[1], dc);
                    }
                } else if (type == "D" || type == "DIODE") {
                    if (nodes.size() >= 2) {
                        DiodeParams diode_params;
                        circuit.add_diode(name, nodes[0], nodes[1], diode_params);
                    }
                } else if (type == "M" || type == "MOSFET") {
                    if (nodes.size() >= 3) {
                        MOSFETParams mos_params;
                        mos_params.w = params.count("w") ? parse_number(params["w"]) : 100e-6;
                        mos_params.l = params.count("l") ? parse_number(params["l"]) : 10e-6;
                        circuit.add_mosfet(name, nodes[0], nodes[1], nodes[2], mos_params);
                    }
                } else if (type == "SWITCH") {
                    if (nodes.size() >= 4) {
                        SwitchParams sw_params;
                        sw_params.ron = params.count("ron") ? parse_number(params["ron"]) : 0.001;
                        circuit.add_switch(name, nodes[0], nodes[1], nodes[2], nodes[3], sw_params);
                    }
                } else {
                    warnings_.push_back("Unknown component type: " + type);
                }
            }
        } else if (node.key == "simulation") {
            for (const auto& sim_attr : node.children) {
                if (sim_attr.key == "type") {
                    // transient, dc, ac
                } else if (sim_attr.key == "stop_time") {
                    options.tstop = parse_number(sim_attr.value);
                } else if (sim_attr.key == "timestep") {
                    options.dt = parse_number(sim_attr.value);
                } else if (sim_attr.key == "abstol") {
                    options.abstol = parse_number(sim_attr.value);
                } else if (sim_attr.key == "reltol") {
                    options.reltol = parse_number(sim_attr.value);
                } else if (sim_attr.key == "max_newton_iterations") {
                    options.max_newton_iterations = std::stoi(sim_attr.value);
                }
            }
        } else if (node.key == "options") {
            for (const auto& opt : node.children) {
                if (opt.key == "abstol") options.abstol = parse_number(opt.value);
                else if (opt.key == "reltol") options.reltol = parse_number(opt.value);
            }
        }
    }
}

}  // namespace pulsim::parser
