#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/circuit.hpp"
#include "pulsim/parser.hpp"

using namespace pulsim;
using Catch::Matchers::WithinRel;

TEST_CASE("Circuit construction", "[circuit]") {
    Circuit circuit;

    SECTION("Add resistor") {
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
        circuit.add_resistor("R1", "in", "0", 1000.0);

        CHECK(circuit.node_count() == 1);
        CHECK(circuit.components().size() == 2);
    }

    SECTION("Node indexing") {
        circuit.add_voltage_source("V1", "a", "0", DCWaveform{1.0});
        circuit.add_resistor("R1", "a", "b", 100.0);
        circuit.add_resistor("R2", "b", "0", 100.0);

        CHECK(circuit.node_count() == 2);
        CHECK(circuit.is_ground("0"));
        CHECK(circuit.is_ground("gnd"));
        CHECK_FALSE(circuit.is_ground("a"));

        Index idx_a = circuit.node_index("a");
        Index idx_b = circuit.node_index("b");
        CHECK(idx_a >= 0);
        CHECK(idx_b >= 0);
        CHECK(idx_a != idx_b);
    }

    SECTION("Branch currents") {
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
        circuit.add_inductor("L1", "in", "out", 1e-3);
        circuit.add_resistor("R1", "out", "0", 100.0);

        CHECK(circuit.node_count() == 2);  // in, out
        CHECK(circuit.branch_count() == 2);  // V1, L1
        CHECK(circuit.total_variables() == 4);
    }
}

TEST_CASE("Circuit validation", "[circuit]") {
    Circuit circuit;
    std::string error;

    SECTION("Empty circuit fails") {
        CHECK_FALSE(circuit.validate(error));
        CHECK(error.find("no components") != std::string::npos);
    }

    SECTION("No source fails") {
        circuit.add_resistor("R1", "a", "0", 100.0);
        CHECK_FALSE(circuit.validate(error));
        CHECK(error.find("no sources") != std::string::npos);
    }

    SECTION("No ground fails") {
        circuit.add_voltage_source("V1", "a", "b", DCWaveform{1.0});
        circuit.add_resistor("R1", "a", "b", 100.0);
        CHECK_FALSE(circuit.validate(error));
        CHECK(error.find("no ground") != std::string::npos);
    }

    SECTION("Valid circuit passes") {
        circuit.add_voltage_source("V1", "a", "0", DCWaveform{1.0});
        circuit.add_resistor("R1", "a", "0", 100.0);
        CHECK(circuit.validate(error));
    }
}

TEST_CASE("Signal names", "[circuit]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_inductor("L1", "out", "0", 1e-3);

    // Node voltages
    CHECK(circuit.signal_name(0) == "V(in)");
    CHECK(circuit.signal_name(1) == "V(out)");

    // Branch currents (after node voltages)
    CHECK(circuit.signal_name(2) == "I(V1)");
    CHECK(circuit.signal_name(3) == "I(L1)");
}

TEST_CASE("Component parameters", "[circuit]") {
    Circuit circuit;

    SECTION("Capacitor with initial condition") {
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
        circuit.add_capacitor("C1", "in", "0", 1e-6, 2.5);

        const auto* comp = circuit.find_component("C1");
        REQUIRE(comp != nullptr);

        const auto& params = std::get<CapacitorParams>(comp->params());
        CHECK(params.capacitance == 1e-6);
        CHECK(params.initial_voltage == 2.5);
    }

    SECTION("Inductor with initial condition") {
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
        circuit.add_inductor("L1", "in", "0", 10e-3, 0.5);

        const auto* comp = circuit.find_component("L1");
        REQUIRE(comp != nullptr);

        const auto& params = std::get<InductorParams>(comp->params());
        CHECK(params.inductance == 10e-3);
        CHECK(params.initial_current == 0.5);
    }
}

// =============================================================================
// Schematic Position Tests (Task 4.11)
// =============================================================================

TEST_CASE("SchematicPosition - basic operations", "[circuit][position]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SECTION("No position initially") {
        CHECK_FALSE(circuit.has_position("V1"));
        CHECK_FALSE(circuit.has_position("R1"));
        CHECK_FALSE(circuit.get_position("V1").has_value());
    }

    SECTION("Set and get position") {
        SchematicPosition pos{100.0, 200.0, 90, true};
        circuit.set_position("R1", pos);

        CHECK(circuit.has_position("R1"));
        auto retrieved = circuit.get_position("R1");
        REQUIRE(retrieved.has_value());
        CHECK(retrieved->x == 100.0);
        CHECK(retrieved->y == 200.0);
        CHECK(retrieved->orientation == 90);
        CHECK(retrieved->mirrored == true);
    }

    SECTION("Update existing position") {
        circuit.set_position("V1", {10.0, 20.0, 0, false});
        circuit.set_position("V1", {30.0, 40.0, 180, true});

        auto pos = circuit.get_position("V1");
        REQUIRE(pos.has_value());
        CHECK(pos->x == 30.0);
        CHECK(pos->y == 40.0);
        CHECK(pos->orientation == 180);
        CHECK(pos->mirrored == true);
    }

    SECTION("Get position for nonexistent component") {
        CHECK_FALSE(circuit.get_position("NonExistent").has_value());
    }
}

TEST_CASE("SchematicPosition - all_positions and set_all_positions", "[circuit][position]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_resistor("R2", "out", "0", 2000.0);

    SECTION("all_positions returns empty map initially") {
        auto positions = circuit.all_positions();
        CHECK(positions.empty());
    }

    SECTION("all_positions returns all set positions") {
        circuit.set_position("V1", {0.0, 0.0, 0, false});
        circuit.set_position("R1", {100.0, 0.0, 0, false});
        circuit.set_position("R2", {200.0, 0.0, 90, true});

        auto positions = circuit.all_positions();
        CHECK(positions.size() == 3);
        CHECK(positions.count("V1") == 1);
        CHECK(positions.count("R1") == 1);
        CHECK(positions.count("R2") == 1);
    }

    SECTION("set_all_positions replaces all positions") {
        circuit.set_position("V1", {0.0, 0.0, 0, false});

        std::unordered_map<std::string, SchematicPosition> new_positions = {
            {"R1", {50.0, 60.0, 270, false}},
            {"R2", {70.0, 80.0, 0, true}}
        };
        circuit.set_all_positions(new_positions);

        auto positions = circuit.all_positions();
        CHECK(positions.size() == 2);
        CHECK_FALSE(positions.count("V1"));  // V1 was replaced
        CHECK(positions.count("R1") == 1);
        CHECK(positions.count("R2") == 1);
        CHECK(positions.at("R1").x == 50.0);
        CHECK(positions.at("R2").mirrored == true);
    }
}

TEST_CASE("SchematicPosition - JSON round-trip", "[circuit][position][roundtrip]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{12.0});
    circuit.add_resistor("R1", "in", "mid", 1000.0);
    circuit.add_capacitor("C1", "mid", "0", 10e-6);

    // Set positions with various values
    circuit.set_position("V1", {0.0, 0.0, 0, false});
    circuit.set_position("R1", {150.0, 0.0, 90, false});
    circuit.set_position("C1", {150.0, 100.0, 180, true});

    SECTION("Export to JSON and re-import preserves positions") {
        // Export to JSON
        std::string json_str = NetlistParser::to_json(circuit, true);

        // Check JSON contains position data
        CHECK(json_str.find("\"position\"") != std::string::npos);
        CHECK(json_str.find("\"orientation\"") != std::string::npos);
        CHECK(json_str.find("\"mirrored\"") != std::string::npos);

        // Re-import
        auto result = NetlistParser::parse_string(json_str);
        REQUIRE(result.has_value());
        Circuit& reimported = result.value();

        // Check positions are preserved
        REQUIRE(reimported.has_position("V1"));
        REQUIRE(reimported.has_position("R1"));
        REQUIRE(reimported.has_position("C1"));

        auto pos_v1 = reimported.get_position("V1");
        REQUIRE(pos_v1.has_value());
        CHECK(pos_v1->x == 0.0);
        CHECK(pos_v1->y == 0.0);
        CHECK(pos_v1->orientation == 0);
        CHECK(pos_v1->mirrored == false);

        auto pos_r1 = reimported.get_position("R1");
        REQUIRE(pos_r1.has_value());
        CHECK(pos_r1->x == 150.0);
        CHECK(pos_r1->y == 0.0);
        CHECK(pos_r1->orientation == 90);
        CHECK(pos_r1->mirrored == false);

        auto pos_c1 = reimported.get_position("C1");
        REQUIRE(pos_c1.has_value());
        CHECK(pos_c1->x == 150.0);
        CHECK(pos_c1->y == 100.0);
        CHECK(pos_c1->orientation == 180);
        CHECK(pos_c1->mirrored == true);
    }

    SECTION("Export without positions when flag is false") {
        std::string json_str = NetlistParser::to_json(circuit, false);
        CHECK(json_str.find("\"position\"") == std::string::npos);
    }
}

TEST_CASE("SchematicPosition - JSON round-trip with all orientations", "[circuit][position][roundtrip]") {
    // Test all valid orientation values
    for (int orientation : {0, 90, 180, 270}) {
        Circuit circuit;
        circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
        circuit.add_resistor("R1", "a", "0", 100.0);
        circuit.set_position("R1", {100.0, 200.0, orientation, false});

        std::string json_str = NetlistParser::to_json(circuit, true);
        auto result = NetlistParser::parse_string(json_str);
        REQUIRE(result.has_value());

        auto pos = result.value().get_position("R1");
        REQUIRE(pos.has_value());
        CHECK(pos->orientation == orientation);
    }
}

TEST_CASE("SchematicPosition - JSON round-trip preserves floating-point precision", "[circuit][position][roundtrip]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "0", 100.0);

    // Use precise floating-point values
    double precise_x = 123.456789;
    double precise_y = -987.654321;
    circuit.set_position("R1", {precise_x, precise_y, 0, false});

    std::string json_str = NetlistParser::to_json(circuit, true);
    auto result = NetlistParser::parse_string(json_str);
    REQUIRE(result.has_value());

    auto pos = result.value().get_position("R1");
    REQUIRE(pos.has_value());
    CHECK_THAT(pos->x, WithinRel(precise_x, 1e-9));
    CHECK_THAT(pos->y, WithinRel(precise_y, 1e-9));
}

TEST_CASE("SchematicPosition - partial positions", "[circuit][position][roundtrip]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_resistor("R2", "out", "0", 2000.0);

    // Only set position for some components
    circuit.set_position("R1", {100.0, 50.0, 90, false});
    // V1 and R2 have no position

    std::string json_str = NetlistParser::to_json(circuit, true);
    auto result = NetlistParser::parse_string(json_str);
    REQUIRE(result.has_value());
    Circuit& reimported = result.value();

    // R1 should have position
    CHECK(reimported.has_position("R1"));
    auto pos = reimported.get_position("R1");
    REQUIRE(pos.has_value());
    CHECK(pos->x == 100.0);

    // V1 and R2 should not have positions
    CHECK_FALSE(reimported.has_position("V1"));
    CHECK_FALSE(reimported.has_position("R2"));
}
