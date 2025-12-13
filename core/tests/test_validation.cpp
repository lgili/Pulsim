#include <catch2/catch_test_macros.hpp>
#include "pulsim/validation.hpp"
#include "pulsim/circuit.hpp"

using namespace pulsim;

// =============================================================================
// ValidationResult Tests
// =============================================================================

TEST_CASE("ValidationResult - basic operations", "[validation][result]") {
    ValidationResult result;

    SECTION("Default state is valid") {
        CHECK(result.is_valid);
        CHECK_FALSE(result.has_errors());
        CHECK_FALSE(result.has_warnings());
        CHECK(result.diagnostics.empty());
    }

    SECTION("Adding error makes invalid") {
        result.add_error(DiagnosticCode::E_NO_GROUND, "Test error");
        CHECK_FALSE(result.is_valid);
        CHECK(result.has_errors());
        CHECK(result.errors().size() == 1);
    }

    SECTION("Adding warning keeps valid") {
        result.add_warning(DiagnosticCode::W_FLOATING_NODE, "Test warning");
        CHECK(result.is_valid);
        CHECK(result.has_warnings());
        CHECK(result.warnings().size() == 1);
    }

    SECTION("Adding info keeps valid") {
        result.add_info(DiagnosticCode::I_IDEAL_SWITCH, "Test info");
        CHECK(result.is_valid);
        CHECK_FALSE(result.has_errors());
        CHECK_FALSE(result.has_warnings());
        CHECK(result.infos().size() == 1);
    }

    SECTION("Multiple diagnostics are tracked separately") {
        result.add_error(DiagnosticCode::E_NO_GROUND, "Error 1");
        result.add_warning(DiagnosticCode::W_FLOATING_NODE, "Warning 1");
        result.add_info(DiagnosticCode::I_IDEAL_SWITCH, "Info 1");
        result.add_error(DiagnosticCode::E_DUPLICATE_NAME, "Error 2");

        CHECK(result.diagnostics.size() == 4);
        CHECK(result.errors().size() == 2);
        CHECK(result.warnings().size() == 1);
        CHECK(result.infos().size() == 1);
    }
}

TEST_CASE("Diagnostic - component and node info", "[validation][diagnostic]") {
    ValidationResult result;

    result.add_error(DiagnosticCode::E_INVALID_PARAMETER, "Invalid resistance",
                     "R1", "node_a");

    REQUIRE(result.diagnostics.size() == 1);
    const auto& d = result.diagnostics[0];

    CHECK(d.severity == DiagnosticSeverity::Error);
    CHECK(d.code == DiagnosticCode::E_INVALID_PARAMETER);
    CHECK(d.message == "Invalid resistance");
    CHECK(d.component_name == "R1");
    CHECK(d.node_name == "node_a");
}

// =============================================================================
// E_NO_COMPONENTS - Empty Circuit Detection
// =============================================================================

TEST_CASE("E_NO_COMPONENTS - empty circuit", "[validation][error]") {
    Circuit circuit;
    auto result = validate_circuit(circuit);

    CHECK_FALSE(result.is_valid);
    CHECK(result.has_errors());

    bool found = false;
    for (const auto& d : result.errors()) {
        if (d.code == DiagnosticCode::E_NO_COMPONENTS) {
            found = true;
            CHECK(d.message.find("no components") != std::string::npos);
        }
    }
    CHECK(found);
}

// =============================================================================
// E_NO_GROUND - Missing Ground Detection
// =============================================================================

TEST_CASE("E_NO_GROUND - circuit without ground", "[validation][error]") {
    Circuit circuit;
    // Create circuit with no ground reference
    circuit.add_voltage_source("V1", "a", "b", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 1000.0);

    auto result = validate_circuit(circuit);

    CHECK_FALSE(result.is_valid);

    bool found_no_ground = false;
    for (const auto& d : result.errors()) {
        if (d.code == DiagnosticCode::E_NO_GROUND) {
            found_no_ground = true;
            CHECK(d.message.find("ground") != std::string::npos);
        }
    }
    CHECK(found_no_ground);
}

TEST_CASE("E_NO_GROUND - circuit with ground is valid", "[validation][error]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "0", 1000.0);

    auto result = validate_circuit(circuit);

    // Should not have E_NO_GROUND error
    for (const auto& d : result.errors()) {
        CHECK(d.code != DiagnosticCode::E_NO_GROUND);
    }
}

// =============================================================================
// E_DUPLICATE_NAME - Duplicate Component Name Detection
// =============================================================================

TEST_CASE("E_DUPLICATE_NAME - duplicate component names", "[validation][error]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 1000.0);
    circuit.add_resistor("R1", "b", "0", 2000.0);  // Duplicate name

    auto result = validate_circuit(circuit);

    CHECK_FALSE(result.is_valid);

    bool found_duplicate = false;
    for (const auto& d : result.errors()) {
        if (d.code == DiagnosticCode::E_DUPLICATE_NAME) {
            found_duplicate = true;
            CHECK(d.component_name == "R1");
            CHECK(d.message.find("Duplicate") != std::string::npos);
        }
    }
    CHECK(found_duplicate);
}

// =============================================================================
// E_VOLTAGE_SOURCE_LOOP - Parallel Voltage Sources Detection
// =============================================================================

TEST_CASE("E_VOLTAGE_SOURCE_LOOP - parallel voltage sources", "[validation][error]") {
    Circuit circuit;
    // Two voltage sources connected to the same nodes
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_voltage_source("V2", "a", "0", DCWaveform{10.0});  // Parallel!
    circuit.add_resistor("R1", "a", "0", 1000.0);

    auto result = validate_circuit(circuit);

    CHECK_FALSE(result.is_valid);

    bool found_loop = false;
    for (const auto& d : result.errors()) {
        if (d.code == DiagnosticCode::E_VOLTAGE_SOURCE_LOOP) {
            found_loop = true;
            CHECK(d.message.find("V1") != std::string::npos);
            CHECK(d.message.find("V2") != std::string::npos);
            CHECK(d.related_components.size() == 2);
        }
    }
    CHECK(found_loop);
}

TEST_CASE("E_VOLTAGE_SOURCE_LOOP - series voltage sources are OK", "[validation][error]") {
    Circuit circuit;
    // Series voltage sources (different nodes)
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_voltage_source("V2", "b", "a", DCWaveform{5.0});  // Series
    circuit.add_resistor("R1", "b", "0", 1000.0);

    auto result = validate_circuit(circuit);

    // Should not have voltage source loop error
    for (const auto& d : result.errors()) {
        CHECK(d.code != DiagnosticCode::E_VOLTAGE_SOURCE_LOOP);
    }
}

// =============================================================================
// E_INDUCTOR_LOOP - Inductor/Voltage Source Loop Detection
// =============================================================================

TEST_CASE("E_INDUCTOR_LOOP - inductor parallel to voltage source", "[validation][error]") {
    Circuit circuit;
    // Inductor directly across voltage source (no series resistance)
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_inductor("L1", "a", "0", 1e-3);  // Parallel to V1!

    auto result = validate_circuit(circuit);

    CHECK_FALSE(result.is_valid);

    bool found_loop = false;
    for (const auto& d : result.errors()) {
        if (d.code == DiagnosticCode::E_INDUCTOR_LOOP) {
            found_loop = true;
            CHECK(d.component_name == "L1");
            CHECK(d.message.find("loop") != std::string::npos);
        }
    }
    CHECK(found_loop);
}

TEST_CASE("E_INDUCTOR_LOOP - inductor with series resistance is OK", "[validation][error]") {
    Circuit circuit;
    // Inductor with series resistance (valid circuit)
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 10.0);  // Series resistance
    circuit.add_inductor("L1", "b", "0", 1e-3);

    auto result = validate_circuit(circuit);

    // Should not have inductor loop error
    for (const auto& d : result.errors()) {
        CHECK(d.code != DiagnosticCode::E_INDUCTOR_LOOP);
    }
}

// =============================================================================
// E_INVALID_PARAMETER - Invalid Component Parameters
// =============================================================================

TEST_CASE("E_INVALID_PARAMETER - negative resistance throws", "[validation][error]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});

    // Adding negative resistance throws an exception at circuit construction time
    CHECK_THROWS(circuit.add_resistor("R1", "a", "0", -100.0));
}

TEST_CASE("E_INVALID_PARAMETER - zero capacitance throws", "[validation][error]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 1000.0);

    // Adding zero capacitance throws an exception at circuit construction time
    CHECK_THROWS(circuit.add_capacitor("C1", "b", "0", 0.0));
}

TEST_CASE("E_INVALID_PARAMETER - negative inductance throws", "[validation][error]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 100.0);

    // Adding negative inductance throws an exception at circuit construction time
    CHECK_THROWS(circuit.add_inductor("L1", "b", "0", -1e-3));
}

TEST_CASE("E_INVALID_PARAMETER - switch with invalid resistance", "[validation][error]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_voltage_source("Vctrl", "ctrl", "0", DCWaveform{10.0});

    SwitchParams sw;
    sw.ron = -0.01;  // Invalid negative Ron
    sw.roff = 1e6;
    sw.vth = 5.0;
    circuit.add_switch("S1", "a", "0", "ctrl", "0", sw);

    auto result = validate_circuit(circuit);

    CHECK_FALSE(result.is_valid);

    bool found_invalid = false;
    for (const auto& d : result.errors()) {
        if (d.code == DiagnosticCode::E_INVALID_PARAMETER &&
            d.component_name == "S1") {
            found_invalid = true;
        }
    }
    CHECK(found_invalid);
}

// =============================================================================
// W_FLOATING_NODE - Floating Node Detection
// =============================================================================

TEST_CASE("W_FLOATING_NODE - node with single connection", "[validation][warning]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 1000.0);
    // Node "b" only has one connection (to R1)

    auto result = validate_circuit(circuit);

    bool found_floating = false;
    for (const auto& d : result.warnings()) {
        if (d.code == DiagnosticCode::W_FLOATING_NODE) {
            found_floating = true;
            CHECK(d.node_name == "b");
            CHECK(d.message.find("floating") != std::string::npos);
        }
    }
    CHECK(found_floating);
}

TEST_CASE("W_FLOATING_NODE - well-connected nodes have no warning", "[validation][warning]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "b", 1000.0);
    circuit.add_resistor("R2", "b", "0", 2000.0);

    auto result = validate_circuit(circuit);

    // Should not have floating node warning
    for (const auto& d : result.warnings()) {
        CHECK(d.code != DiagnosticCode::W_FLOATING_NODE);
    }
}

// =============================================================================
// W_SHORT_CIRCUIT - Short Circuit Detection
// =============================================================================

TEST_CASE("W_SHORT_CIRCUIT - very low resistance", "[validation][warning]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "a", "0", 1e-9);  // Extremely low - potential short

    auto result = validate_circuit(circuit);

    bool found_short = false;
    for (const auto& d : result.warnings()) {
        if (d.code == DiagnosticCode::W_SHORT_CIRCUIT) {
            found_short = true;
            CHECK(d.message.find("low resistance") != std::string::npos);
        }
    }
    CHECK(found_short);
}

// =============================================================================
// I_IDEAL_SWITCH - Info About Ideal Switch Model
// =============================================================================

TEST_CASE("I_IDEAL_SWITCH - switch generates info", "[validation][info]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "a", "0", DCWaveform{5.0});
    circuit.add_voltage_source("Vctrl", "ctrl", "0", DCWaveform{10.0});

    SwitchParams sw;
    sw.ron = 0.01;
    sw.roff = 1e9;
    sw.vth = 5.0;
    circuit.add_switch("S1", "a", "b", "ctrl", "0", sw);
    circuit.add_resistor("R1", "b", "0", 100.0);

    auto result = validate_circuit(circuit);

    bool found_info = false;
    for (const auto& d : result.infos()) {
        if (d.code == DiagnosticCode::I_IDEAL_SWITCH) {
            found_info = true;
            CHECK(d.component_name == "S1");
            CHECK(d.message.find("ideal switch") != std::string::npos);
        }
    }
    CHECK(found_info);
}

// =============================================================================
// diagnostic_code_description Tests
// =============================================================================

TEST_CASE("diagnostic_code_description - returns descriptions", "[validation][utility]") {
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_NO_GROUND).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_VOLTAGE_SOURCE_LOOP).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_INDUCTOR_LOOP).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_NO_DC_PATH).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_INVALID_PARAMETER).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_UNKNOWN_NODE).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_DUPLICATE_NAME).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::E_NO_COMPONENTS).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::W_FLOATING_NODE).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::W_SHORT_CIRCUIT).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::W_HIGH_VOLTAGE).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::W_MISSING_IC).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::W_LARGE_TIMESTEP).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::I_IDEAL_SWITCH).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::I_NO_LOSS_MODEL).empty());
    CHECK_FALSE(diagnostic_code_description(DiagnosticCode::I_PARALLEL_SOURCES).empty());
}

// =============================================================================
// Complete Valid Circuit Test
// =============================================================================

TEST_CASE("Valid circuit passes validation", "[validation][complete]") {
    Circuit circuit;

    // Build a complete, valid circuit
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{12.0});
    circuit.add_resistor("R1", "in", "mid", 100.0);
    circuit.add_capacitor("C1", "mid", "0", 1e-6);
    circuit.add_inductor("L1", "mid", "out", 1e-3);
    circuit.add_resistor("R2", "out", "0", 50.0);

    auto result = validate_circuit(circuit);

    CHECK(result.is_valid);
    CHECK_FALSE(result.has_errors());
    // May have some info messages but no errors
}

TEST_CASE("Complex power electronics circuit validation", "[validation][complete]") {
    Circuit circuit;

    // Buck converter topology
    circuit.add_voltage_source("Vin", "vin", "0", DCWaveform{48.0});

    // PWM control
    PulseWaveform pwm;
    pwm.v1 = 0.0;
    pwm.v2 = 10.0;
    pwm.td = 0.0;
    pwm.tr = 10e-9;
    pwm.tf = 10e-9;
    pwm.pw = 5e-6;
    pwm.period = 10e-6;
    circuit.add_voltage_source("Vctrl", "ctrl", "0", pwm);

    // Switch
    SwitchParams sw;
    sw.ron = 0.01;
    sw.roff = 1e9;
    sw.vth = 5.0;
    circuit.add_switch("S1", "vin", "sw", "ctrl", "0", sw);

    // Freewheeling diode
    DiodeParams diode;
    diode.ideal = true;
    circuit.add_diode("D1", "0", "sw", diode);

    // Output filter
    circuit.add_inductor("L1", "sw", "out", 100e-6);
    circuit.add_capacitor("C1", "out", "0", 100e-6);

    // Load
    circuit.add_resistor("Rload", "out", "0", 10.0);

    auto result = validate_circuit(circuit);

    // Should be valid (may have info messages for ideal switch/diode)
    CHECK(result.is_valid);
    CHECK_FALSE(result.has_errors());

    // Should have I_IDEAL_SWITCH info
    bool found_switch_info = false;
    for (const auto& d : result.infos()) {
        if (d.code == DiagnosticCode::I_IDEAL_SWITCH) {
            found_switch_info = true;
        }
    }
    CHECK(found_switch_info);
}
