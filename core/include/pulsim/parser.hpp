#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/types.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <variant>
#include <optional>

namespace pulsim {

// Parse error information
struct ParseError {
    std::string message;
    int line = -1;
    int column = -1;

    std::string to_string() const {
        if (line >= 0) {
            return "Line " + std::to_string(line) + ", Col " + std::to_string(column) + ": " + message;
        }
        return message;
    }
};

// Simple Result type (until C++23 std::expected is available)
template<typename T>
class ParseResult {
public:
    ParseResult(T value) : data_(std::move(value)) {}
    ParseResult(ParseError error) : data_(std::move(error)) {}

    bool has_value() const { return std::holds_alternative<T>(data_); }
    explicit operator bool() const { return has_value(); }

    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }

    T& operator*() { return value(); }
    const T& operator*() const { return value(); }

    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }

    ParseError& error() { return std::get<ParseError>(data_); }
    const ParseError& error() const { return std::get<ParseError>(data_); }

private:
    std::variant<T, ParseError> data_;
};

// Parser for JSON netlist format
class NetlistParser {
public:
    // Parse from file
    static ParseResult<Circuit> parse_file(const std::filesystem::path& path);

    // Parse from string
    static ParseResult<Circuit> parse_string(const std::string& content);

    // Parse simulation options from JSON string
    static ParseResult<SimulationOptions> parse_options(const std::string& content);

    // Parse simulation options from file
    static ParseResult<SimulationOptions> parse_simulation_options(const std::filesystem::path& path);

    // Export circuit to JSON string (for GUI)
    static std::string to_json(const Circuit& circuit, bool include_positions = true);

    // Utility function (public for testing)
    static Real parse_value_with_suffix(const std::string& str);

private:
    static ParseResult<Circuit> parse_json(const std::string& content);
    static ParseResult<Waveform> parse_waveform(const nlohmann::json& j);
};

}  // namespace pulsim
