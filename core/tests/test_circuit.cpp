#include <catch2/catch_test_macros.hpp>
#include "spicelab/circuit.hpp"

using namespace spicelab;

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
