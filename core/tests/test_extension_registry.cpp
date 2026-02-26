#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

#include "pulsim/v1/core.hpp"

using namespace pulsim::v1;

namespace {

struct CustomDeviceExtension {
    static constexpr ExtensionCategory kCategory = ExtensionCategory::Device;

    [[nodiscard]] static ExtensionMetadata metadata() {
        ExtensionMetadata metadata;
        metadata.category = ExtensionCategory::Device;
        metadata.extension_id = "custom_device_stub";
        metadata.display_name = "CustomDeviceStub";
        metadata.version = "1.0.0";
        metadata.capabilities = {"stub", "nonlinear"};
        metadata.telemetry_fields = {"custom_device_metric"};
        return metadata;
    }
};

struct CustomSolverExtension {
    static constexpr ExtensionCategory kCategory = ExtensionCategory::Solver;

    [[nodiscard]] static ExtensionMetadata metadata() {
        ExtensionMetadata metadata;
        metadata.category = ExtensionCategory::Solver;
        metadata.extension_id = "custom_solver_stub";
        metadata.display_name = "CustomSolverStub";
        metadata.version = "1.0.0";
        metadata.capabilities = {"iterative", "stub"};
        metadata.telemetry_fields = {"custom_solver_metric"};
        return metadata;
    }
};

struct CustomIntegratorExtension {
    static constexpr ExtensionCategory kCategory = ExtensionCategory::Integrator;

    [[nodiscard]] static ExtensionMetadata metadata() {
        ExtensionMetadata metadata;
        metadata.category = ExtensionCategory::Integrator;
        metadata.extension_id = "custom_integrator_stub";
        metadata.display_name = "CustomIntegratorStub";
        metadata.version = "1.0.0";
        metadata.capabilities = {"order_2", "stiff_stable"};
        metadata.telemetry_fields = {"custom_integrator_metric"};
        return metadata;
    }
};

struct WrongCategoryDeviceExtension {
    static constexpr ExtensionCategory kCategory = ExtensionCategory::Device;

    [[nodiscard]] static ExtensionMetadata metadata() {
        ExtensionMetadata metadata;
        metadata.category = ExtensionCategory::Solver;
        metadata.extension_id = "wrong_category";
        metadata.display_name = "WrongCategory";
        metadata.version = "1.0.0";
        metadata.capabilities = {"iterative"};
        metadata.telemetry_fields = {"metric"};
        return metadata;
    }
};

struct MissingCategoryContract {
    [[nodiscard]] static ExtensionMetadata metadata() {
        ExtensionMetadata metadata;
        metadata.extension_id = "missing_category";
        metadata.display_name = "MissingCategory";
        return metadata;
    }
};

static_assert(ExtensionContract<CustomDeviceExtension>);
static_assert(ExtensionContract<CustomSolverExtension>);
static_assert(ExtensionContract<CustomIntegratorExtension>);
static_assert(!ExtensionContract<MissingCategoryContract>);
static_assert(DeviceExtensionContract<CustomDeviceExtension>);
static_assert(SolverExtensionContract<CustomSolverExtension>);
static_assert(IntegratorExtensionContract<CustomIntegratorExtension>);
static_assert(!DeviceExtensionContract<CustomSolverExtension>);
static_assert(!SolverExtensionContract<CustomIntegratorExtension>);

}  // namespace

TEST_CASE("v1 extension registry exposes builtin device/solver/integrator metadata",
          "[v1][extensions][registry]") {
    const auto& registry = kernel_extension_registry();

    REQUIRE(registry.size(ExtensionCategory::Device) >= 11);
    REQUIRE(registry.size(ExtensionCategory::Solver) >= 6);
    REQUIRE(registry.size(ExtensionCategory::Integrator) >= 10);

    const auto device = registry.find(ExtensionCategory::Device, "resistor");
    REQUIRE(device.has_value());
    REQUIRE(device->display_name == "Resistor");

    const auto solver = registry.find(ExtensionCategory::Solver, "gmres");
    REQUIRE(solver.has_value());
    REQUIRE(solver->display_name == "GMRES");
}

TEST_CASE("v1 extension registry accepts typed contracts for new classes",
          "[v1][extensions][registry][contract]") {
    ExtensionRegistry registry;

    const ExtensionValidationResult device_status = registry.register_device<CustomDeviceExtension>();
    const ExtensionValidationResult solver_status = registry.register_solver<CustomSolverExtension>();
    const ExtensionValidationResult integrator_status =
        registry.register_integrator<CustomIntegratorExtension>();

    REQUIRE(device_status.ok());
    REQUIRE(solver_status.ok());
    REQUIRE(integrator_status.ok());

    REQUIRE(registry.size(ExtensionCategory::Device) == 1);
    REQUIRE(registry.size(ExtensionCategory::Solver) == 1);
    REQUIRE(registry.size(ExtensionCategory::Integrator) == 1);
}

TEST_CASE("v1 extension registry rejects incompatible contracts deterministically",
          "[v1][extensions][registry][validation]") {
    ExtensionRegistry registry;

    const ExtensionValidationResult first = registry.register_device<CustomDeviceExtension>();
    const ExtensionValidationResult duplicate = registry.register_device<CustomDeviceExtension>();
    const ExtensionValidationResult wrong_category =
        registry.register_device<WrongCategoryDeviceExtension>();

    REQUIRE(first.ok());
    REQUIRE_FALSE(duplicate.ok());
    REQUIRE(duplicate.code == ExtensionValidationCode::DuplicateId);
    REQUIRE(duplicate.field == "extension_id");
    REQUIRE(registry.size(ExtensionCategory::Device) == 1);

    REQUIRE_FALSE(wrong_category.ok());
    REQUIRE(wrong_category.code == ExtensionValidationCode::CategoryMismatch);
}

TEST_CASE("v1 extension registry validates compatibility requirements",
          "[v1][extensions][registry][compatibility]") {
    ExtensionRegistry registry;
    REQUIRE(registry.register_solver<CustomSolverExtension>().ok());

    const ExtensionRequirement ok_requirement {
        .category = ExtensionCategory::Solver,
        .extension_id = "custom_solver_stub",
        .required_capability = "iterative"
    };

    const ExtensionRequirement missing_capability {
        .category = ExtensionCategory::Solver,
        .extension_id = "custom_solver_stub",
        .required_capability = "direct"
    };

    const ExtensionRequirement missing_extension {
        .category = ExtensionCategory::Integrator,
        .extension_id = "not_registered",
        .required_capability = "order_2"
    };

    REQUIRE(registry.validate_requirement(ok_requirement).ok());

    const ExtensionValidationResult missing_capability_status =
        registry.validate_requirement(missing_capability);
    REQUIRE_FALSE(missing_capability_status.ok());
    REQUIRE(missing_capability_status.code == ExtensionValidationCode::CapabilityNotSupported);

    const ExtensionValidationResult missing_extension_status =
        registry.validate_requirement(missing_extension);
    REQUIRE_FALSE(missing_extension_status.ok());
    REQUIRE(missing_extension_status.code == ExtensionValidationCode::ExtensionNotFound);

    const std::array<ExtensionRequirement, 2> requirements = {
        ok_requirement,
        missing_capability
    };
    const ExtensionValidationResult batch_status = registry.validate_requirements(requirements);
    REQUIRE_FALSE(batch_status.ok());
    REQUIRE(batch_status.code == ExtensionValidationCode::CapabilityNotSupported);
}

TEST_CASE("v1 simulator exposes active extension registry without orchestrator customization",
          "[v1][extensions][registry][simulator]") {
    Circuit circuit;
    circuit.add_voltage_source("Vin", 0, -1, 48.0);
    circuit.add_resistor("Rin", 0, 1, 10.0);
    circuit.add_capacitor("Cout", 1, -1, 1e-6, 0.0);
    circuit.add_resistor("Rload", 1, -1, 5.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 1e-6;
    options.dt = 1e-7;
    options.adaptive_timestep = false;

    Simulator simulator(circuit, options);
    const auto& registry = simulator.extension_registry();

    REQUIRE(registry.size(ExtensionCategory::Device) >= 11);
    REQUIRE(registry.find(ExtensionCategory::Integrator, "trbdf2").has_value());
}
