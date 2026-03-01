#pragma once

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/simulation.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace pulsim::v1::parser {

struct YamlParserOptions {
    bool strict = true;              // Fail on unknown fields
    bool validate_nodes = true;      // Check for floating nodes (future)
};

class YamlParser {
public:
    explicit YamlParser(YamlParserOptions options = {});

    // Parse from file
    std::pair<Circuit, SimulationOptions> load(const std::filesystem::path& path);

    // Parse from string
    std::pair<Circuit, SimulationOptions> load_string(const std::string& content);

    const std::vector<std::string>& errors() const { return errors_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    YamlParserOptions options_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;

    void parse_yaml(const std::string& content, Circuit& circuit, SimulationOptions& options);
};

}  // namespace pulsim::v1::parser

