/**
 * @file runtime_modules.hpp
 * @brief Runtime module contracts and deterministic dependency resolution.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pulsim::v1 {

/**
 * @brief Runtime lifecycle hooks used by modular execution concerns.
 */
enum class RuntimeModuleHook : std::uint8_t {
    RunInitialize,
    StepAttempt,
    StepAccepted,
    SampleEmit,
    Finalize
};

/**
 * @brief Declares one runtime module and its capability contract.
 */
struct RuntimeModuleDescriptor {
    std::string module_id;
    std::string display_name;
    std::vector<std::string> provides_capabilities;
    std::vector<std::string> requires_capabilities;
    std::vector<RuntimeModuleHook> hooks;
};

/**
 * @brief Validation/result codes for runtime module dependency checks.
 */
enum class RuntimeModuleValidationCode : std::uint8_t {
    Ok,
    MissingModuleId,
    DuplicateModuleId,
    EmptyProvidedCapability,
    DuplicateProvidedCapability,
    EmptyRequiredCapability,
    CapabilityWithoutProvider,
    CapabilityWithMultipleProviders,
    CyclicDependency
};

/**
 * @brief Human-readable string for @ref RuntimeModuleValidationCode.
 */
[[nodiscard]] constexpr std::string_view to_string(RuntimeModuleValidationCode code) noexcept {
    switch (code) {
        case RuntimeModuleValidationCode::Ok: return "ok";
        case RuntimeModuleValidationCode::MissingModuleId: return "missing_module_id";
        case RuntimeModuleValidationCode::DuplicateModuleId: return "duplicate_module_id";
        case RuntimeModuleValidationCode::EmptyProvidedCapability: return "empty_provided_capability";
        case RuntimeModuleValidationCode::DuplicateProvidedCapability:
            return "duplicate_provided_capability";
        case RuntimeModuleValidationCode::EmptyRequiredCapability: return "empty_required_capability";
        case RuntimeModuleValidationCode::CapabilityWithoutProvider: return "capability_without_provider";
        case RuntimeModuleValidationCode::CapabilityWithMultipleProviders:
            return "capability_with_multiple_providers";
        case RuntimeModuleValidationCode::CyclicDependency: return "cyclic_dependency";
    }
    return "unknown";
}

/**
 * @brief Structured validation report for module graph resolution.
 */
struct RuntimeModuleValidationResult {
    RuntimeModuleValidationCode code = RuntimeModuleValidationCode::Ok;
    std::string module_id;
    std::string capability;
    std::string message;

    [[nodiscard]] bool ok() const noexcept { return code == RuntimeModuleValidationCode::Ok; }

    [[nodiscard]] static RuntimeModuleValidationResult success() {
        return {};
    }

    [[nodiscard]] static RuntimeModuleValidationResult failure(RuntimeModuleValidationCode failure_code,
                                                               std::string module,
                                                               std::string capability_name,
                                                               std::string detail) {
        RuntimeModuleValidationResult out;
        out.code = failure_code;
        out.module_id = std::move(module);
        out.capability = std::move(capability_name);
        out.message = std::move(detail);
        return out;
    }
};

/**
 * @brief Deterministic topologically-sorted runtime module plan.
 */
struct RuntimeModulePlan {
    std::vector<RuntimeModuleDescriptor> ordered_modules;

    [[nodiscard]] bool empty() const noexcept { return ordered_modules.empty(); }

    [[nodiscard]] std::size_t size() const noexcept { return ordered_modules.size(); }

    [[nodiscard]] bool contains(std::string_view module_id) const {
        return std::any_of(
            ordered_modules.begin(),
            ordered_modules.end(),
            [module_id](const RuntimeModuleDescriptor& module) { return module.module_id == module_id; });
    }

    [[nodiscard]] std::vector<std::string> execution_order() const {
        std::vector<std::string> out;
        out.reserve(ordered_modules.size());
        for (const auto& module : ordered_modules) {
            out.push_back(module.module_id);
        }
        return out;
    }
};

/**
 * @brief Resolution payload for dependency-checked runtime module plans.
 */
struct RuntimeModuleResolution {
    RuntimeModulePlan plan;
    RuntimeModuleValidationResult validation;

    [[nodiscard]] bool ok() const noexcept { return validation.ok(); }
};

/**
 * @brief Resolves module dependencies to a deterministic execution order.
 *
 * Requirements:
 * - module ids are non-empty and unique.
 * - provided/required capabilities are non-empty strings.
 * - each required capability has exactly one provider.
 * - dependency graph is acyclic.
 *
 * Deterministic order:
 * - Kahn topological sort with lexical tie-break on module id.
 */
[[nodiscard]] inline RuntimeModuleResolution resolve_runtime_module_plan(
    std::span<const RuntimeModuleDescriptor> modules) {

    RuntimeModuleResolution out;
    if (modules.empty()) {
        out.validation = RuntimeModuleValidationResult::success();
        return out;
    }

    std::unordered_map<std::string, std::size_t> module_index;
    module_index.reserve(modules.size());

    for (std::size_t i = 0; i < modules.size(); ++i) {
        const auto& module = modules[i];
        if (module.module_id.empty()) {
            out.validation = RuntimeModuleValidationResult::failure(
                RuntimeModuleValidationCode::MissingModuleId,
                "",
                "",
                "runtime module id must not be empty");
            return out;
        }
        const auto [it, inserted] = module_index.emplace(module.module_id, i);
        if (!inserted) {
            out.validation = RuntimeModuleValidationResult::failure(
                RuntimeModuleValidationCode::DuplicateModuleId,
                module.module_id,
                "",
                "runtime module id already registered");
            return out;
        }
        (void)it;
    }

    std::unordered_map<std::string, std::string> capability_provider;
    capability_provider.reserve(modules.size() * 4);
    for (const auto& module : modules) {
        std::unordered_set<std::string_view> local_seen;
        local_seen.reserve(module.provides_capabilities.size());
        for (const auto& capability : module.provides_capabilities) {
            if (capability.empty()) {
                out.validation = RuntimeModuleValidationResult::failure(
                    RuntimeModuleValidationCode::EmptyProvidedCapability,
                    module.module_id,
                    capability,
                    "provided capability must not be empty");
                return out;
            }
            if (!local_seen.emplace(capability).second) {
                out.validation = RuntimeModuleValidationResult::failure(
                    RuntimeModuleValidationCode::DuplicateProvidedCapability,
                    module.module_id,
                    capability,
                    "module provides the same capability more than once");
                return out;
            }
            const auto provider_it = capability_provider.find(capability);
            if (provider_it != capability_provider.end() &&
                provider_it->second != module.module_id) {
                out.validation = RuntimeModuleValidationResult::failure(
                    RuntimeModuleValidationCode::CapabilityWithMultipleProviders,
                    module.module_id,
                    capability,
                    "capability has multiple providers");
                return out;
            }
            capability_provider[capability] = module.module_id;
        }
    }

    std::vector<int> indegree(modules.size(), 0);
    std::vector<std::vector<std::size_t>> outgoing(modules.size());
    for (std::size_t i = 0; i < modules.size(); ++i) {
        const auto& module = modules[i];
        std::unordered_set<std::string_view> local_seen;
        local_seen.reserve(module.requires_capabilities.size());
        for (const auto& capability : module.requires_capabilities) {
            if (capability.empty()) {
                out.validation = RuntimeModuleValidationResult::failure(
                    RuntimeModuleValidationCode::EmptyRequiredCapability,
                    module.module_id,
                    capability,
                    "required capability must not be empty");
                return out;
            }
            if (!local_seen.emplace(capability).second) {
                continue;
            }
            const auto provider_it = capability_provider.find(capability);
            if (provider_it == capability_provider.end()) {
                out.validation = RuntimeModuleValidationResult::failure(
                    RuntimeModuleValidationCode::CapabilityWithoutProvider,
                    module.module_id,
                    capability,
                    "required capability has no provider");
                return out;
            }
            const auto owner_it = module_index.find(provider_it->second);
            if (owner_it == module_index.end()) {
                out.validation = RuntimeModuleValidationResult::failure(
                    RuntimeModuleValidationCode::CapabilityWithoutProvider,
                    module.module_id,
                    capability,
                    "required capability provider module is missing");
                return out;
            }

            const std::size_t provider_index = owner_it->second;
            if (provider_index == i) {
                continue;
            }
            outgoing[provider_index].push_back(i);
            indegree[i] += 1;
        }
    }

    auto module_id_less = [&modules](std::size_t lhs, std::size_t rhs) {
        return modules[lhs].module_id > modules[rhs].module_id;
    };
    std::priority_queue<std::size_t,
                        std::vector<std::size_t>,
                        decltype(module_id_less)> ready(module_id_less);
    for (std::size_t i = 0; i < indegree.size(); ++i) {
        if (indegree[i] == 0) {
            ready.push(i);
        }
    }

    out.plan.ordered_modules.reserve(modules.size());
    while (!ready.empty()) {
        const std::size_t i = ready.top();
        ready.pop();
        out.plan.ordered_modules.push_back(modules[i]);
        for (const std::size_t next : outgoing[i]) {
            indegree[next] -= 1;
            if (indegree[next] == 0) {
                ready.push(next);
            }
        }
    }

    if (out.plan.ordered_modules.size() != modules.size()) {
        out.plan.ordered_modules.clear();
        out.validation = RuntimeModuleValidationResult::failure(
            RuntimeModuleValidationCode::CyclicDependency,
            "",
            "",
            "runtime module dependency graph contains a cycle");
        return out;
    }

    out.validation = RuntimeModuleValidationResult::success();
    return out;
}

/**
 * @brief Canonical default runtime module descriptors for v1 transient orchestration.
 */
[[nodiscard]] inline std::vector<RuntimeModuleDescriptor> make_default_runtime_modules() {
    std::vector<RuntimeModuleDescriptor> modules;
    modules.reserve(5);

    modules.push_back(RuntimeModuleDescriptor{
        .module_id = "events_topology",
        .display_name = "Events and Topology",
        .provides_capabilities = {"events", "topology"},
        .requires_capabilities = {},
        .hooks = {
            RuntimeModuleHook::RunInitialize,
            RuntimeModuleHook::StepAttempt,
            RuntimeModuleHook::StepAccepted,
            RuntimeModuleHook::Finalize
        }
    });

    modules.push_back(RuntimeModuleDescriptor{
        .module_id = "control_mixed_domain",
        .display_name = "Control and Mixed-Domain",
        .provides_capabilities = {"control"},
        .requires_capabilities = {"events"},
        .hooks = {
            RuntimeModuleHook::RunInitialize,
            RuntimeModuleHook::StepAttempt,
            RuntimeModuleHook::StepAccepted,
            RuntimeModuleHook::SampleEmit,
            RuntimeModuleHook::Finalize
        }
    });

    modules.push_back(RuntimeModuleDescriptor{
        .module_id = "loss_accounting",
        .display_name = "Loss Accounting",
        .provides_capabilities = {"loss"},
        .requires_capabilities = {"events", "topology"},
        .hooks = {
            RuntimeModuleHook::RunInitialize,
            RuntimeModuleHook::StepAccepted,
            RuntimeModuleHook::SampleEmit,
            RuntimeModuleHook::Finalize
        }
    });

    modules.push_back(RuntimeModuleDescriptor{
        .module_id = "thermal_coupling",
        .display_name = "Thermal Coupling",
        .provides_capabilities = {"thermal"},
        .requires_capabilities = {"loss"},
        .hooks = {
            RuntimeModuleHook::RunInitialize,
            RuntimeModuleHook::StepAccepted,
            RuntimeModuleHook::SampleEmit,
            RuntimeModuleHook::Finalize
        }
    });

    modules.push_back(RuntimeModuleDescriptor{
        .module_id = "telemetry_channels",
        .display_name = "Telemetry and Channels",
        .provides_capabilities = {"telemetry"},
        .requires_capabilities = {"control", "events", "loss", "thermal"},
        .hooks = {
            RuntimeModuleHook::RunInitialize,
            RuntimeModuleHook::SampleEmit,
            RuntimeModuleHook::Finalize
        }
    });

    return modules;
}

/**
 * @brief Builds and validates the canonical default runtime module plan.
 */
[[nodiscard]] inline RuntimeModuleResolution resolve_default_runtime_module_plan() {
    const std::vector<RuntimeModuleDescriptor> modules = make_default_runtime_modules();
    return resolve_runtime_module_plan(modules);
}

}  // namespace pulsim::v1
