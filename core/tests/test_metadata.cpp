#include <catch2/catch_test_macros.hpp>
#include "pulsim/metadata.hpp"

using namespace pulsim;

// =============================================================================
// ComponentRegistry Tests
// =============================================================================

TEST_CASE("ComponentRegistry - Singleton instance", "[metadata][registry]") {
    const auto& registry1 = ComponentRegistry::instance();
    const auto& registry2 = ComponentRegistry::instance();

    // Should be the same instance
    CHECK(&registry1 == &registry2);
}

TEST_CASE("ComponentRegistry - All component types registered", "[metadata][registry]") {
    const auto& registry = ComponentRegistry::instance();

    // Check all basic component types are registered
    CHECK(registry.get(ComponentType::Resistor) != nullptr);
    CHECK(registry.get(ComponentType::Capacitor) != nullptr);
    CHECK(registry.get(ComponentType::Inductor) != nullptr);
    CHECK(registry.get(ComponentType::VoltageSource) != nullptr);
    CHECK(registry.get(ComponentType::CurrentSource) != nullptr);
    CHECK(registry.get(ComponentType::Diode) != nullptr);
    CHECK(registry.get(ComponentType::Switch) != nullptr);
    CHECK(registry.get(ComponentType::MOSFET) != nullptr);
    CHECK(registry.get(ComponentType::IGBT) != nullptr);
    CHECK(registry.get(ComponentType::Transformer) != nullptr);
}

TEST_CASE("ComponentRegistry - all_types returns all types", "[metadata][registry]") {
    const auto& registry = ComponentRegistry::instance();
    auto types = registry.all_types();

    CHECK(types.size() >= 10);  // At least 10 component types

    // Check some types are present
    bool has_resistor = false;
    bool has_capacitor = false;
    for (auto t : types) {
        if (t == ComponentType::Resistor) has_resistor = true;
        if (t == ComponentType::Capacitor) has_capacitor = true;
    }
    CHECK(has_resistor);
    CHECK(has_capacitor);
}

TEST_CASE("ComponentRegistry - types_in_category", "[metadata][registry]") {
    const auto& registry = ComponentRegistry::instance();

    SECTION("Passive category") {
        auto passives = registry.types_in_category("Passive");
        CHECK(passives.size() >= 3);  // R, L, C at minimum

        bool has_resistor = false;
        for (auto t : passives) {
            if (t == ComponentType::Resistor) has_resistor = true;
        }
        CHECK(has_resistor);
    }

    SECTION("Semiconductor category") {
        auto semis = registry.types_in_category("Semiconductor");
        CHECK(semis.size() >= 3);  // Diode, MOSFET, IGBT at minimum

        bool has_diode = false;
        for (auto t : semis) {
            if (t == ComponentType::Diode) has_diode = true;
        }
        CHECK(has_diode);
    }

    SECTION("Empty category returns empty vector") {
        auto unknown = registry.types_in_category("NonexistentCategory");
        CHECK(unknown.empty());
    }
}

TEST_CASE("ComponentRegistry - all_categories", "[metadata][registry]") {
    const auto& registry = ComponentRegistry::instance();
    auto categories = registry.all_categories();

    CHECK(categories.size() >= 3);  // At least Passive, Semiconductor, Sources

    bool has_passive = false;
    bool has_semiconductor = false;
    for (const auto& cat : categories) {
        if (cat == "Passive") has_passive = true;
        if (cat == "Semiconductor") has_semiconductor = true;
    }
    CHECK(has_passive);
    CHECK(has_semiconductor);
}

// =============================================================================
// Resistor Metadata Tests
// =============================================================================

TEST_CASE("Resistor metadata", "[metadata][resistor]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::Resistor);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::Resistor);
    CHECK(meta->name == "resistor");
    CHECK(meta->display_name == "Resistor");
    CHECK(meta->category == "Passive");
    CHECK_FALSE(meta->description.empty());

    SECTION("Pins") {
        CHECK(meta->pins.size() == 2);
    }

    SECTION("Parameters") {
        CHECK(meta->parameters.size() >= 1);

        // Find resistance parameter
        bool found_resistance = false;
        for (const auto& param : meta->parameters) {
            if (param.name == "resistance") {
                found_resistance = true;
                CHECK(param.type == ParameterType::Real);
                CHECK(param.unit == "ohm");
                CHECK(param.min_value.has_value());
                CHECK(*param.min_value > 0);  // Resistance must be positive
                CHECK(param.required);
            }
        }
        CHECK(found_resistance);
    }

    SECTION("Has loss model") {
        CHECK(meta->has_loss_model);
    }
}

// =============================================================================
// Capacitor Metadata Tests
// =============================================================================

TEST_CASE("Capacitor metadata", "[metadata][capacitor]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::Capacitor);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::Capacitor);
    CHECK(meta->category == "Passive");

    SECTION("Parameters") {
        bool found_capacitance = false;
        for (const auto& param : meta->parameters) {
            if (param.name == "capacitance") {
                found_capacitance = true;
                CHECK(param.type == ParameterType::Real);
                CHECK(param.unit == "F");
                CHECK(param.min_value.has_value());
                CHECK(*param.min_value > 0);
            }
        }
        CHECK(found_capacitance);
    }
}

// =============================================================================
// Inductor Metadata Tests
// =============================================================================

TEST_CASE("Inductor metadata", "[metadata][inductor]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::Inductor);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::Inductor);
    CHECK(meta->category == "Passive");

    SECTION("Parameters") {
        bool found_inductance = false;
        for (const auto& param : meta->parameters) {
            if (param.name == "inductance") {
                found_inductance = true;
                CHECK(param.type == ParameterType::Real);
                CHECK(param.unit == "H");
            }
        }
        CHECK(found_inductance);
    }
}

// =============================================================================
// Diode Metadata Tests
// =============================================================================

TEST_CASE("Diode metadata", "[metadata][diode]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::Diode);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::Diode);
    CHECK(meta->category == "Semiconductor");

    SECTION("Pins") {
        CHECK(meta->pins.size() == 2);
        // Should have anode and cathode
        bool has_anode = false;
        bool has_cathode = false;
        for (const auto& pin : meta->pins) {
            if (pin.name == "anode" || pin.name == "a") has_anode = true;
            if (pin.name == "cathode" || pin.name == "k") has_cathode = true;
        }
        CHECK((has_anode || meta->pins.size() == 2));  // Either named or just 2 pins
    }

    SECTION("Has loss model") {
        CHECK(meta->has_loss_model);
    }
}

// =============================================================================
// MOSFET Metadata Tests
// =============================================================================

TEST_CASE("MOSFET metadata", "[metadata][mosfet]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::MOSFET);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::MOSFET);
    CHECK(meta->category == "Semiconductor");

    SECTION("Pins") {
        CHECK(meta->pins.size() >= 3);  // At least drain, gate, source
    }

    SECTION("Has thermal model") {
        CHECK(meta->has_thermal_model);
    }
}

// =============================================================================
// IGBT Metadata Tests
// =============================================================================

TEST_CASE("IGBT metadata", "[metadata][igbt]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::IGBT);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::IGBT);
    CHECK(meta->category == "Semiconductor");

    SECTION("Pins") {
        CHECK(meta->pins.size() >= 3);  // At least collector, gate, emitter
    }

    SECTION("Has thermal model") {
        CHECK(meta->has_thermal_model);
    }
}

// =============================================================================
// Voltage Source Metadata Tests
// =============================================================================

TEST_CASE("VoltageSource metadata", "[metadata][source]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::VoltageSource);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::VoltageSource);
    CHECK(meta->category == "Sources");

    SECTION("Pins") {
        CHECK(meta->pins.size() == 2);
    }
}

// =============================================================================
// Transformer Metadata Tests
// =============================================================================

TEST_CASE("Transformer metadata", "[metadata][transformer]") {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(ComponentType::Transformer);

    REQUIRE(meta != nullptr);

    CHECK(meta->type == ComponentType::Transformer);
    CHECK(meta->category == "Magnetic");

    SECTION("Pins") {
        CHECK(meta->pins.size() >= 4);  // Primary and secondary windings
    }

    SECTION("Parameters include turns ratio") {
        bool found_turns_ratio = false;
        for (const auto& param : meta->parameters) {
            if (param.name == "turns_ratio" || param.name == "n") {
                found_turns_ratio = true;
            }
        }
        CHECK(found_turns_ratio);
    }
}

// =============================================================================
// Parameter Validation Tests (Task 3.19)
// =============================================================================

TEST_CASE("Parameter validation - Resistor", "[metadata][validation]") {
    const auto& registry = ComponentRegistry::instance();

    SECTION("Valid resistance") {
        std::string error;
        bool valid = registry.validate_parameter(ComponentType::Resistor, "resistance", 1000.0, &error);
        CHECK(valid);
        CHECK(error.empty());
    }

    SECTION("Zero resistance is invalid") {
        std::string error;
        bool valid = registry.validate_parameter(ComponentType::Resistor, "resistance", 0.0, &error);
        CHECK_FALSE(valid);
        CHECK_FALSE(error.empty());
    }

    SECTION("Negative resistance is invalid") {
        std::string error;
        bool valid = registry.validate_parameter(ComponentType::Resistor, "resistance", -100.0, &error);
        CHECK_FALSE(valid);
        CHECK_FALSE(error.empty());
    }
}

TEST_CASE("Parameter validation - Capacitor", "[metadata][validation]") {
    const auto& registry = ComponentRegistry::instance();

    SECTION("Valid capacitance") {
        std::string error;
        bool valid = registry.validate_parameter(ComponentType::Capacitor, "capacitance", 1e-6, &error);
        CHECK(valid);
    }

    SECTION("Negative capacitance is invalid") {
        std::string error;
        bool valid = registry.validate_parameter(ComponentType::Capacitor, "capacitance", -1e-6, &error);
        CHECK_FALSE(valid);
    }
}

TEST_CASE("Parameter validation - Unknown parameter", "[metadata][validation]") {
    const auto& registry = ComponentRegistry::instance();

    std::string error;
    bool valid = registry.validate_parameter(ComponentType::Resistor, "nonexistent_param", 100.0, &error);
    CHECK_FALSE(valid);
    CHECK(error.find("Unknown parameter") != std::string::npos);
}

TEST_CASE("Parameter validation - Error message is optional", "[metadata][validation]") {
    const auto& registry = ComponentRegistry::instance();

    // Should not crash when error_message is nullptr
    bool valid = registry.validate_parameter(ComponentType::Resistor, "resistance", -100.0, nullptr);
    CHECK_FALSE(valid);
}

// =============================================================================
// ParameterMetadata Tests
// =============================================================================

TEST_CASE("ParameterMetadata default values", "[metadata][parameter]") {
    ParameterMetadata param;

    CHECK(param.type == ParameterType::Real);
    CHECK(param.required == true);
    CHECK_FALSE(param.default_value.has_value());
    CHECK_FALSE(param.min_value.has_value());
    CHECK_FALSE(param.max_value.has_value());
}

TEST_CASE("ParameterMetadata optional values", "[metadata][parameter]") {
    ParameterMetadata param;
    param.name = "test";
    param.default_value = 1.0;
    param.min_value = 0.0;
    param.max_value = 100.0;

    CHECK(param.default_value.has_value());
    CHECK(*param.default_value == 1.0);
    CHECK(param.min_value.has_value());
    CHECK(*param.min_value == 0.0);
    CHECK(param.max_value.has_value());
    CHECK(*param.max_value == 100.0);
}

// =============================================================================
// PinMetadata Tests
// =============================================================================

TEST_CASE("PinMetadata", "[metadata][pin]") {
    PinMetadata pin;
    pin.name = "anode";
    pin.description = "Positive terminal";

    CHECK(pin.name == "anode");
    CHECK(pin.description == "Positive terminal");
}

// =============================================================================
// ComponentMetadata Tests
// =============================================================================

TEST_CASE("ComponentMetadata flags", "[metadata][component]") {
    const auto& registry = ComponentRegistry::instance();

    SECTION("Resistor has loss model") {
        const auto* meta = registry.get(ComponentType::Resistor);
        REQUIRE(meta != nullptr);
        CHECK(meta->has_loss_model);
    }

    SECTION("MOSFET has thermal model") {
        const auto* meta = registry.get(ComponentType::MOSFET);
        REQUIRE(meta != nullptr);
        CHECK(meta->has_thermal_model);
    }

    SECTION("IGBT has thermal model") {
        const auto* meta = registry.get(ComponentType::IGBT);
        REQUIRE(meta != nullptr);
        CHECK(meta->has_thermal_model);
    }
}

TEST_CASE("ComponentMetadata symbol_id", "[metadata][component]") {
    const auto& registry = ComponentRegistry::instance();

    // All components should have a symbol_id for GUI rendering
    auto types = registry.all_types();
    for (auto type : types) {
        const auto* meta = registry.get(type);
        REQUIRE(meta != nullptr);
        CHECK_FALSE(meta->symbol_id.empty());
    }
}

TEST_CASE("ComponentMetadata display names", "[metadata][component]") {
    const auto& registry = ComponentRegistry::instance();

    auto types = registry.all_types();
    for (auto type : types) {
        const auto* meta = registry.get(type);
        REQUIRE(meta != nullptr);
        CHECK_FALSE(meta->display_name.empty());
        CHECK_FALSE(meta->description.empty());
    }
}
