#include <catch2/catch_test_macros.hpp>

#include <array>

#include "pulsim/v1/core.hpp"

using namespace pulsim::v1;

TEST_CASE("v1 runtime module resolver returns deterministic default execution order",
          "[v1][runtime-modules][resolver]") {
    const RuntimeModuleResolution resolution = resolve_default_runtime_module_plan();
    REQUIRE(resolution.ok());
    REQUIRE(resolution.plan.size() == 5);

    const std::vector<std::string> order = resolution.plan.execution_order();
    REQUIRE(order.size() == 5);
    CHECK(order[0] == "events_topology");
    CHECK(order[1] == "control_mixed_domain");
    CHECK(order[2] == "loss_accounting");
    CHECK(order[3] == "thermal_coupling");
    CHECK(order[4] == "telemetry_channels");

    CHECK(resolution.plan.contains("events_topology"));
    CHECK(resolution.plan.contains("telemetry_channels"));
}

TEST_CASE("v1 runtime module resolver rejects duplicate module ids deterministically",
          "[v1][runtime-modules][resolver][validation]") {
    std::array<RuntimeModuleDescriptor, 2> modules = {
        RuntimeModuleDescriptor{
            .module_id = "dup",
            .display_name = "Duplicate A",
            .provides_capabilities = {"cap_a"},
            .requires_capabilities = {},
            .hooks = {RuntimeModuleHook::RunInitialize}
        },
        RuntimeModuleDescriptor{
            .module_id = "dup",
            .display_name = "Duplicate B",
            .provides_capabilities = {"cap_b"},
            .requires_capabilities = {},
            .hooks = {RuntimeModuleHook::RunInitialize}
        }
    };

    const RuntimeModuleResolution resolution = resolve_runtime_module_plan(modules);
    REQUIRE_FALSE(resolution.ok());
    CHECK(resolution.validation.code == RuntimeModuleValidationCode::DuplicateModuleId);
    CHECK(resolution.validation.module_id == "dup");
}

TEST_CASE("v1 runtime module resolver rejects missing capability providers",
          "[v1][runtime-modules][resolver][validation]") {
    std::array<RuntimeModuleDescriptor, 2> modules = {
        RuntimeModuleDescriptor{
            .module_id = "producer",
            .display_name = "Producer",
            .provides_capabilities = {"cap_a"},
            .requires_capabilities = {},
            .hooks = {RuntimeModuleHook::RunInitialize}
        },
        RuntimeModuleDescriptor{
            .module_id = "consumer",
            .display_name = "Consumer",
            .provides_capabilities = {"cap_b"},
            .requires_capabilities = {"missing_capability"},
            .hooks = {RuntimeModuleHook::RunInitialize}
        }
    };

    const RuntimeModuleResolution resolution = resolve_runtime_module_plan(modules);
    REQUIRE_FALSE(resolution.ok());
    CHECK(resolution.validation.code == RuntimeModuleValidationCode::CapabilityWithoutProvider);
    CHECK(resolution.validation.module_id == "consumer");
    CHECK(resolution.validation.capability == "missing_capability");
}

TEST_CASE("v1 runtime module resolver rejects dependency cycles",
          "[v1][runtime-modules][resolver][validation]") {
    std::array<RuntimeModuleDescriptor, 2> modules = {
        RuntimeModuleDescriptor{
            .module_id = "module_a",
            .display_name = "ModuleA",
            .provides_capabilities = {"cap_a"},
            .requires_capabilities = {"cap_b"},
            .hooks = {RuntimeModuleHook::RunInitialize}
        },
        RuntimeModuleDescriptor{
            .module_id = "module_b",
            .display_name = "ModuleB",
            .provides_capabilities = {"cap_b"},
            .requires_capabilities = {"cap_a"},
            .hooks = {RuntimeModuleHook::RunInitialize}
        }
    };

    const RuntimeModuleResolution resolution = resolve_runtime_module_plan(modules);
    REQUIRE_FALSE(resolution.ok());
    CHECK(resolution.validation.code == RuntimeModuleValidationCode::CyclicDependency);
}

TEST_CASE("v1 simulator exposes resolved runtime module graph",
          "[v1][runtime-modules][simulator]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto n_out = circuit.add_node("out");
    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 24.0);
    circuit.add_resistor("Rload", n_in, n_out, 4.0);
    circuit.add_capacitor("Cout", n_out, Circuit::ground(), 10e-6, 0.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 2e-6;
    options.dt = 1e-7;
    options.adaptive_timestep = false;

    Simulator simulator(circuit, options);
    const RuntimeModuleResolution& resolution = simulator.runtime_module_resolution();
    REQUIRE(resolution.ok());
    REQUIRE(resolution.plan.size() == 5);
    CHECK(resolution.plan.contains("events_topology"));
    CHECK(resolution.plan.contains("loss_accounting"));
    CHECK(resolution.plan.contains("thermal_coupling"));

    const SimulationResult result = simulator.run_transient();
    REQUIRE(result.success);
    CHECK(result.backend_telemetry.runtime_module_count == 5);
    CHECK(result.backend_telemetry.runtime_module_order ==
          "events_topology,control_mixed_domain,loss_accounting,thermal_coupling,telemetry_channels");
}
