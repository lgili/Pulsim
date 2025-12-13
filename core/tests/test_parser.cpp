#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/parser.hpp"

using namespace pulsim;
using Catch::Matchers::WithinRel;

TEST_CASE("Parse value with SI suffix", "[parser]") {
    SECTION("Plain numbers") {
        CHECK_THAT(NetlistParser::parse_value_with_suffix("1.5"), WithinRel(1.5, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("100"), WithinRel(100.0, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("1e-6"), WithinRel(1e-6, 1e-10));
    }

    SECTION("SI prefixes") {
        CHECK_THAT(NetlistParser::parse_value_with_suffix("1k"), WithinRel(1e3, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("4.7k"), WithinRel(4.7e3, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("10u"), WithinRel(10e-6, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("100n"), WithinRel(100e-9, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("1m"), WithinRel(1e-3, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("2.2meg"), WithinRel(2.2e6, 1e-10));
        CHECK_THAT(NetlistParser::parse_value_with_suffix("47p"), WithinRel(47e-12, 1e-10));
    }
}

TEST_CASE("Parse simple resistor circuit", "[parser]") {
    std::string netlist = R"({
        "components": [
            {"type": "resistor", "name": "R1", "n1": "in", "n2": "0", "value": "1k"},
            {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0", "waveform": 5.0}
        ]
    })";

    auto result = NetlistParser::parse_string(netlist);
    REQUIRE(result.has_value());

    const Circuit& circuit = *result;
    CHECK(circuit.node_count() == 1);  // Only "in" (ground is implicit)
    CHECK(circuit.components().size() == 2);
}

TEST_CASE("Parse RC circuit", "[parser]") {
    std::string netlist = R"({
        "components": [
            {"type": "V", "name": "V1", "n1": "in", "n2": "0", "waveform": {"type": "pulse", "v1": 0, "v2": 5, "period": 1e-3}},
            {"type": "R", "name": "R1", "n1": "in", "n2": "out", "value": "1k"},
            {"type": "C", "name": "C1", "n1": "out", "n2": "0", "value": "1u"}
        ]
    })";

    auto result = NetlistParser::parse_string(netlist);
    REQUIRE(result.has_value());

    const Circuit& circuit = *result;
    CHECK(circuit.node_count() == 2);  // "in" and "out"
    CHECK(circuit.components().size() == 3);
}

TEST_CASE("Parse RLC circuit", "[parser]") {
    std::string netlist = R"({
        "components": [
            {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0", "waveform": 10},
            {"type": "resistor", "name": "R1", "n1": "in", "n2": "n1", "value": 100},
            {"type": "inductor", "name": "L1", "n1": "n1", "n2": "out", "value": "10m"},
            {"type": "capacitor", "name": "C1", "n1": "out", "n2": "0", "value": "100u"}
        ]
    })";

    auto result = NetlistParser::parse_string(netlist);
    REQUIRE(result.has_value());

    const Circuit& circuit = *result;
    CHECK(circuit.node_count() == 3);  // in, n1, out
    CHECK(circuit.branch_count() == 2);  // V1, L1
    CHECK(circuit.total_variables() == 5);
}

TEST_CASE("Parse errors", "[parser]") {
    SECTION("Invalid JSON") {
        auto result = NetlistParser::parse_string("{invalid json}");
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().message.find("parse error") != std::string::npos);
    }

    SECTION("Missing components") {
        auto result = NetlistParser::parse_string(R"({"name": "test"})");
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("Unknown component type") {
        std::string netlist = R"({
            "components": [
                {"type": "unknown", "name": "X1", "n1": "a", "n2": "0"}
            ]
        })";
        auto result = NetlistParser::parse_string(netlist);
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("Parse waveforms", "[parser]") {
    SECTION("Sine waveform") {
        std::string netlist = R"({
            "components": [
                {"type": "V", "name": "V1", "n1": "out", "n2": "0",
                 "waveform": {"type": "sin", "offset": 2.5, "amplitude": 2.5, "frequency": 1000}}
            ]
        })";

        auto result = NetlistParser::parse_string(netlist);
        REQUIRE(result.has_value());
    }

    SECTION("PWL waveform") {
        std::string netlist = R"({
            "components": [
                {"type": "V", "name": "V1", "n1": "out", "n2": "0",
                 "waveform": {"type": "pwl", "points": [[0, 0], [1e-3, 5], [2e-3, 5], [3e-3, 0]]}}
            ]
        })";

        auto result = NetlistParser::parse_string(netlist);
        REQUIRE(result.has_value());
    }
}
