#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/simulation.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace pulsim::parser {

// =============================================================================
// YAML Netlist Parser
// Supports .yaml and .yml formats
// =============================================================================

struct YamlParserOptions {
    bool strict = false;              // Fail on unknown fields
    bool validate_nodes = true;       // Check for floating nodes
    std::vector<std::filesystem::path> include_paths;
};

class YamlParser {
public:
    explicit YamlParser(YamlParserOptions options = {});

    // Parse from file
    std::pair<Circuit, SimulationOptions> load(const std::filesystem::path& path);

    // Parse from string
    std::pair<Circuit, SimulationOptions> load_string(const std::string& content);

    // Get parse errors/warnings
    const std::vector<std::string>& errors() const { return errors_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    YamlParserOptions options_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;

    // Internal parsing (uses simple YAML subset parser)
    void parse_yaml(const std::string& content, Circuit& circuit, SimulationOptions& options);
};

}  // namespace pulsim::parser
