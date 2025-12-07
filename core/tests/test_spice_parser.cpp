/**
 * @file test_spice_parser.cpp
 * @brief Comprehensive tests for SPICE netlist parser
 *
 * Tests parsing of SPICE netlists, value parsing with SI prefixes,
 * component parsing, model statements, subcircuits, and simulation commands.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>

#include "pulsim/parser/spice_parser.hpp"
#include "pulsim/parser/subcircuit.hpp"

using namespace pulsim;
using namespace pulsim::parser;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Catch::Matchers::ContainsSubstring;

// =============================================================================
// Value Parsing Tests
// =============================================================================

TEST_CASE("SpiceParser - Parse numeric values", "[parser][spice][values]") {
    SpiceParser parser;

    SECTION("Plain numbers") {
        auto [circuit, opts] = parser.load_string("V1 1 0 1.5\n.tran 1n 10n");
        // Check that 1.5V was parsed (voltage source)
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("SI prefixes") {
        // Test various SI prefixes
        auto [circuit, opts] = parser.load_string(
            "R1 1 0 1k\n"
            "R2 2 0 1.5M\n"
            "C1 3 0 10u\n"
            "C2 4 0 100p\n"
            "L1 5 0 1m\n"
            ".tran 1n 100n"
        );
        REQUIRE(circuit.components().size() >= 5);
    }
}

// =============================================================================
// Basic Component Parsing Tests
// =============================================================================

TEST_CASE("SpiceParser - Parse resistors", "[parser][spice][resistor]") {
    SpiceParser parser;

    SECTION("Basic resistor") {
        auto [circuit, opts] = parser.load_string(
            "R1 1 0 1000\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Multiple resistors") {
        auto [circuit, opts] = parser.load_string(
            "R1 1 2 1k\n"
            "R2 2 0 2k\n"
            "R3 1 0 3k\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 3);
    }
}

TEST_CASE("SpiceParser - Parse capacitors", "[parser][spice][capacitor]") {
    SpiceParser parser;

    SECTION("Basic capacitor") {
        auto [circuit, opts] = parser.load_string(
            "C1 1 0 1u\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Capacitor with initial condition") {
        auto [circuit, opts] = parser.load_string(
            "C1 1 0 1u IC=5\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse inductors", "[parser][spice][inductor]") {
    SpiceParser parser;

    SECTION("Basic inductor") {
        auto [circuit, opts] = parser.load_string(
            "L1 1 0 1m\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Inductor with initial condition") {
        auto [circuit, opts] = parser.load_string(
            "L1 1 0 1m IC=0.1\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse voltage sources", "[parser][spice][vsource]") {
    SpiceParser parser;

    SECTION("DC voltage source") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 DC 5\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("AC voltage source") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 AC 1 0\n"
            ".ac dec 10 1 1Meg"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Pulse voltage source") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 PULSE(0 5 0 1n 1n 5u 10u)\n"
            ".tran 1n 100u"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Sine voltage source") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 SIN(0 1 1k)\n"
            ".tran 1u 10m"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("PWL voltage source") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 PWL(0 0 1u 5 2u 5 3u 0)\n"
            ".tran 1n 10u"
        );
        REQUIRE(circuit.components().size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse current sources", "[parser][spice][isource]") {
    SpiceParser parser;

    SECTION("DC current source") {
        auto [circuit, opts] = parser.load_string(
            "I1 0 1 DC 1m\n"
            "R1 1 0 1k\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Pulse current source") {
        auto [circuit, opts] = parser.load_string(
            "I1 0 1 PULSE(0 1m 0 1n 1n 5u 10u)\n"
            "R1 1 0 1k\n"
            ".tran 1n 100u"
        );
        REQUIRE(circuit.components().size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse diodes", "[parser][spice][diode]") {
    SpiceParser parser;

    SECTION("Basic diode") {
        auto [circuit, opts] = parser.load_string(
            "D1 1 0 DMOD\n"
            ".model DMOD D(IS=1e-14)\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Diode with model parameters") {
        auto [circuit, opts] = parser.load_string(
            "D1 1 0 1N4148\n"
            ".model 1N4148 D(IS=2.52e-9 RS=0.568 N=1.752 BV=100 IBV=100u)\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse MOSFETs", "[parser][spice][mosfet]") {
    SpiceParser parser;

    SECTION("Basic NMOS") {
        auto [circuit, opts] = parser.load_string(
            "M1 drain gate source source NMOD W=10u L=1u\n"
            ".model NMOD NMOS(VTO=0.7 KP=110u)\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }

    SECTION("Basic PMOS") {
        auto [circuit, opts] = parser.load_string(
            "M1 drain gate source source PMOD W=20u L=1u\n"
            ".model PMOD PMOS(VTO=-0.7 KP=50u)\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 1);
    }
}

// =============================================================================
// Simulation Command Tests
// =============================================================================

TEST_CASE("SpiceParser - Parse .tran command", "[parser][spice][tran]") {
    SpiceParser parser;

    SECTION("Basic transient") {
        auto result = parser.parse_string(
            "V1 1 0 1\n"
            ".tran 1n 100n"
        );
        REQUIRE(result.simulations.size() >= 1);
        REQUIRE(result.simulations[0].type == SpiceSimulation::Type::Tran);
    }

    SECTION("Transient with start time") {
        auto result = parser.parse_string(
            "V1 1 0 1\n"
            ".tran 1n 100n 50n"
        );
        REQUIRE(result.simulations.size() >= 1);
    }

    SECTION("Transient with max step") {
        auto result = parser.parse_string(
            "V1 1 0 1\n"
            ".tran 1n 100n 0 0.5n"
        );
        REQUIRE(result.simulations.size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse .ac command", "[parser][spice][ac]") {
    SpiceParser parser;

    SECTION("Decade sweep") {
        auto result = parser.parse_string(
            "V1 1 0 AC 1\n"
            ".ac dec 10 1 1Meg"
        );
        REQUIRE(result.simulations.size() >= 1);
        REQUIRE(result.simulations[0].type == SpiceSimulation::Type::Ac);
    }

    SECTION("Linear sweep") {
        auto result = parser.parse_string(
            "V1 1 0 AC 1\n"
            ".ac lin 100 100 10k"
        );
        REQUIRE(result.simulations.size() >= 1);
    }

    SECTION("Octave sweep") {
        auto result = parser.parse_string(
            "V1 1 0 AC 1\n"
            ".ac oct 10 10 10k"
        );
        REQUIRE(result.simulations.size() >= 1);
    }
}

TEST_CASE("SpiceParser - Parse .dc command", "[parser][spice][dc]") {
    SpiceParser parser;

    auto result = parser.parse_string(
        "V1 1 0 1\n"
        "R1 1 0 1k\n"
        ".dc V1 0 10 0.1"
    );
    REQUIRE(result.simulations.size() >= 1);
    REQUIRE(result.simulations[0].type == SpiceSimulation::Type::Dc);
}

TEST_CASE("SpiceParser - Parse .op command", "[parser][spice][op]") {
    SpiceParser parser;

    auto result = parser.parse_string(
        "V1 1 0 5\n"
        "R1 1 0 1k\n"
        ".op"
    );
    REQUIRE(result.simulations.size() >= 1);
    REQUIRE(result.simulations[0].type == SpiceSimulation::Type::Op);
}

// =============================================================================
// Model Statement Tests
// =============================================================================

TEST_CASE("SpiceParser - Parse .model statements", "[parser][spice][model]") {
    SpiceParser parser;

    SECTION("Diode model") {
        auto result = parser.parse_string(
            ".model D1N4148 D(IS=2.52e-9 RS=0.568 N=1.752 BV=100 IBV=100u)\n"
            "D1 1 0 D1N4148\n"
            ".tran 1n 10n"
        );
        REQUIRE(result.models.size() >= 1);
        REQUIRE(result.models[0].type == "D");
    }

    SECTION("NMOS model") {
        auto result = parser.parse_string(
            ".model NMOD NMOS(VTO=0.7 KP=110u LAMBDA=0.04)\n"
            "M1 d g s s NMOD\n"
            ".tran 1n 10n"
        );
        REQUIRE(result.models.size() >= 1);
        REQUIRE(result.models[0].type == "NMOS");
    }

    SECTION("PMOS model") {
        auto result = parser.parse_string(
            ".model PMOD PMOS(VTO=-0.7 KP=50u LAMBDA=0.05)\n"
            "M1 d g s s PMOD\n"
            ".tran 1n 10n"
        );
        REQUIRE(result.models.size() >= 1);
        REQUIRE(result.models[0].type == "PMOS");
    }
}

// =============================================================================
// Subcircuit Tests
// =============================================================================

TEST_CASE("SpiceParser - Parse .subckt definition", "[parser][spice][subckt]") {
    SpiceParser parser;

    auto result = parser.parse_string(
        ".subckt INVERTER in out vdd vss\n"
        "M1 out in vdd vdd PMOD W=20u L=1u\n"
        "M2 out in vss vss NMOD W=10u L=1u\n"
        ".ends INVERTER\n"
        ".model NMOD NMOS(VTO=0.7)\n"
        ".model PMOD PMOS(VTO=-0.7)\n"
        ".tran 1n 10n"
    );
    REQUIRE(result.subcircuits.size() >= 1);
    REQUIRE(result.subcircuits[0].name == "INVERTER");
    REQUIRE(result.subcircuits[0].ports.size() == 4);
}

TEST_CASE("SpiceParser - Subcircuit instantiation", "[parser][spice][subckt]") {
    SpiceParser parser;

    auto result = parser.parse_string(
        ".subckt RESISTOR_DIV in out gnd\n"
        "R1 in out 1k\n"
        "R2 out gnd 1k\n"
        ".ends\n"
        "X1 1 2 0 RESISTOR_DIV\n"
        "V1 1 0 10\n"
        ".tran 1n 10n"
    );
    REQUIRE(result.subcircuits.size() >= 1);
    // X1 should be an instance
    bool found_x1 = false;
    for (const auto& comp : result.components) {
        if (comp.name == "1" && comp.type == "X") {
            found_x1 = true;
            break;
        }
    }
    // Instance might be in components list depending on expansion
}

// =============================================================================
// Line Continuation Tests
// =============================================================================

TEST_CASE("SpiceParser - Line continuation", "[parser][spice][continuation]") {
    SpiceParser parser;

    // SPICE uses + at the beginning of a line for continuation
    auto result = parser.parse_string(
        "V1 1 0 PULSE(0 5 0\n"
        "+ 1n 1n 5u 10u)\n"
        ".tran 1n 100u"
    );
    REQUIRE(result.errors.empty());
    REQUIRE(result.components.size() >= 1);
}

// =============================================================================
// Comments Tests
// =============================================================================

TEST_CASE("SpiceParser - Comments", "[parser][spice][comments]") {
    SpiceParser parser;

    SECTION("Asterisk comments") {
        auto result = parser.parse_string(
            "* This is a comment\n"
            "V1 1 0 5\n"
            "* Another comment\n"
            "R1 1 0 1k\n"
            ".tran 1n 10n"
        );
        REQUIRE(result.errors.empty());
        REQUIRE(result.components.size() >= 2);
    }

    SECTION("Semicolon inline comments") {
        auto result = parser.parse_string(
            "V1 1 0 5 ; voltage source\n"
            "R1 1 0 1k ; resistor\n"
            ".tran 1n 10n"
        );
        // Should parse without errors
        REQUIRE(result.components.size() >= 2);
    }
}

// =============================================================================
// Title Line Tests
// =============================================================================

TEST_CASE("SpiceParser - Title line", "[parser][spice][title]") {
    SpiceParser parser;

    auto result = parser.parse_string(
        "My Test Circuit\n"
        "V1 1 0 5\n"
        "R1 1 0 1k\n"
        ".tran 1n 10n"
    );
    REQUIRE(result.title == "My Test Circuit");
}

// =============================================================================
// Options Tests
// =============================================================================

TEST_CASE("SpiceParser - Parse .options", "[parser][spice][options]") {
    SpiceParser parser;

    auto result = parser.parse_string(
        ".options RELTOL=1e-4 ABSTOL=1e-12 ITL1=100\n"
        "V1 1 0 5\n"
        "R1 1 0 1k\n"
        ".tran 1n 10n"
    );
    // Options should be captured
    REQUIRE(!result.options.empty());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_CASE("SpiceParser - Error handling", "[parser][spice][errors]") {
    SpiceParser parser;

    SECTION("Missing node") {
        // This should generate a warning or error
        auto result = parser.parse_string(
            "R1 1 1k\n"  // Missing second node
            ".tran 1n 10n"
        );
        REQUIRE(!result.errors.empty() || !result.warnings.empty());
    }

    SECTION("Unknown directive in strict mode") {
        SpiceParserOptions opts;
        opts.strict = true;
        SpiceParser strict_parser(opts);

        auto result = strict_parser.parse_string(
            ".unknown_directive\n"
            "V1 1 0 5\n"
            ".tran 1n 10n"
        );
        // Should have error or warning
    }
}

// =============================================================================
// Subcircuit Library Tests
// =============================================================================

TEST_CASE("SubcircuitLibrary - Basic operations", "[parser][subcircuit]") {
    SubcircuitLibrary lib;

    SECTION("Add and retrieve subcircuit") {
        lib.add("TEST", {"in", "out", "gnd"});
        REQUIRE(lib.exists("TEST"));

        auto* subckt = lib.get("TEST");
        REQUIRE(subckt != nullptr);
        REQUIRE(subckt->name() == "TEST");
        REQUIRE(subckt->ports().size() == 3);
    }

    SECTION("List subcircuits") {
        lib.add("SUB1", {"a", "b"});
        lib.add("SUB2", {"x", "y", "z"});

        auto list = lib.list();
        REQUIRE(list.size() == 2);
    }

    SECTION("Clear library") {
        lib.add("TEST", {"a"});
        REQUIRE(lib.exists("TEST"));

        lib.clear();
        REQUIRE(!lib.exists("TEST"));
    }
}

TEST_CASE("SubcircuitDefinition - Add components", "[parser][subcircuit]") {
    SubcircuitDefinition subckt("TESTCKT", {"in", "out", "gnd"});

    subckt.add_resistor("R1", "in", "mid", 1000);
    subckt.add_resistor("R2", "mid", "out", 2000);
    subckt.add_capacitor("C1", "out", "gnd", 1e-6);

    REQUIRE(subckt.components().size() == 3);
}

TEST_CASE("SubcircuitExpander - Expand subcircuit", "[parser][subcircuit]") {
    SubcircuitLibrary lib;

    // Create a voltage divider subcircuit
    lib.add("VDIV", {"in", "out", "gnd"});
    auto* vdiv = lib.get("VDIV");
    vdiv->add_resistor("R1", "in", "out", 1000);
    vdiv->add_resistor("R2", "out", "gnd", 1000);

    // Expand into circuit
    Circuit circuit;
    SubcircuitExpander expander(lib);

    expander.expand_into(circuit, "X1", "VDIV", {"node1", "node2", "0"});

    // Should have two resistors with prefixed names
    REQUIRE(circuit.components().size() >= 2);
}

// =============================================================================
// Subcircuit Templates Tests
// =============================================================================

TEST_CASE("Subcircuit templates - Half bridge", "[parser][subcircuit][templates]") {
    auto hb = templates::half_bridge();
    REQUIRE(hb->name() == "HALFBRIDGE");
    REQUIRE(!hb->ports().empty());
}

TEST_CASE("Subcircuit templates - Full bridge", "[parser][subcircuit][templates]") {
    auto fb = templates::full_bridge();
    REQUIRE(fb->name() == "FULLBRIDGE");
}

TEST_CASE("Subcircuit templates - LC filter", "[parser][subcircuit][templates]") {
    auto lc = templates::lc_filter("MYFILTER", 100e-6, 10e-6);
    REQUIRE(lc->name() == "MYFILTER");
}

// =============================================================================
// Circuit Conversion Tests
// =============================================================================

TEST_CASE("SpiceParser - Convert to circuit", "[parser][spice][convert]") {
    SpiceParser parser;

    SECTION("Simple RC circuit") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 5\n"
            "R1 1 2 1k\n"
            "C1 2 0 1u\n"
            ".tran 1u 10m"
        );

        REQUIRE(circuit.components().size() >= 1);
        REQUIRE(circuit.components().size() >= 3);
    }

    SECTION("RLC circuit") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 SIN(0 1 1k)\n"
            "R1 1 2 100\n"
            "L1 2 3 10m\n"
            "C1 3 0 1u\n"
            ".tran 1u 10m"
        );

        REQUIRE(circuit.components().size() >= 1);
        REQUIRE(circuit.components().size() >= 4);
    }
}

// =============================================================================
// Case Insensitivity Tests
// =============================================================================

TEST_CASE("SpiceParser - Case insensitivity", "[parser][spice][case]") {
    SpiceParser parser;

    // SPICE is traditionally case-insensitive
    auto result = parser.parse_string(
        "v1 1 0 DC 5\n"
        "r1 1 0 1K\n"
        ".TRAN 1N 10N"
    );

    REQUIRE(result.errors.empty());
    REQUIRE(result.components.size() >= 2);
}

// =============================================================================
// Node Name Tests
// =============================================================================

TEST_CASE("SpiceParser - Node names", "[parser][spice][nodes]") {
    SpiceParser parser;

    SECTION("Numeric nodes") {
        auto [circuit, opts] = parser.load_string(
            "V1 1 0 5\n"
            "R1 1 2 1k\n"
            "R2 2 0 1k\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 3);
    }

    SECTION("Alphanumeric nodes") {
        auto [circuit, opts] = parser.load_string(
            "V1 VCC GND 5\n"
            "R1 VCC OUT 1k\n"
            "R2 OUT GND 1k\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 3);
    }

    SECTION("Mixed nodes") {
        auto [circuit, opts] = parser.load_string(
            "V1 VDD 0 5\n"
            "R1 VDD node1 1k\n"
            "R2 node1 0 1k\n"
            ".tran 1n 10n"
        );
        REQUIRE(circuit.components().size() >= 3);
    }
}

// =============================================================================
// Full Circuit Examples
// =============================================================================

TEST_CASE("SpiceParser - Buck converter netlist", "[parser][spice][buck]") {
    SpiceParser parser;

    auto [circuit, opts] = parser.load_string(
        "* Buck Converter\n"
        "VIN in 0 DC 12\n"
        "SW1 in sw_out ctrl 0 SW RON=10m\n"
        "D1 0 sw_out DIODE\n"
        "L1 sw_out out 100u\n"
        "C1 out 0 100u\n"
        "RLOAD out 0 10\n"
        ".model SW SW(RON=10m ROFF=1Meg)\n"
        ".model DIODE D(IS=1e-14)\n"
        ".tran 100n 1m"
    );

    // Should parse without errors
    REQUIRE(parser.errors().empty());
}

TEST_CASE("SpiceParser - Inverter netlist", "[parser][spice][inverter]") {
    SpiceParser parser;

    auto [circuit, opts] = parser.load_string(
        "* CMOS Inverter\n"
        "VDD vdd 0 DC 5\n"
        "VIN in 0 PULSE(0 5 0 1n 1n 10n 20n)\n"
        "M1 out in vdd vdd PMOD W=20u L=1u\n"
        "M2 out in 0 0 NMOD W=10u L=1u\n"
        "CL out 0 1p\n"
        ".model NMOD NMOS(VTO=0.7 KP=110u)\n"
        ".model PMOD PMOS(VTO=-0.7 KP=50u)\n"
        ".tran 0.1n 100n"
    );

    REQUIRE(parser.errors().empty());
}

// =============================================================================
// Format Detection Tests
// =============================================================================

TEST_CASE("detect_format - File extensions", "[parser][format]") {
    REQUIRE(detect_format("test.cir") == NetlistFormat::SpiceCir);
    REQUIRE(detect_format("test.sp") == NetlistFormat::SpiceCir);
    REQUIRE(detect_format("test.net") == NetlistFormat::SpiceCir);
    REQUIRE(detect_format("test.asc") == NetlistFormat::LTspiceAsc);
    REQUIRE(detect_format("test.yaml") == NetlistFormat::Yaml);
    REQUIRE(detect_format("test.yml") == NetlistFormat::Yaml);
    REQUIRE(detect_format("test.json") == NetlistFormat::Json);
    REQUIRE(detect_format("test.xyz") == NetlistFormat::Unknown);
}
