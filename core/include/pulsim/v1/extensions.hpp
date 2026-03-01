#pragma once

#include "pulsim/v1/cpp23_features.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/integration.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsim::v1 {

enum class ExtensionCategory : std::uint8_t {
    Device,
    Solver,
    Integrator
};

[[nodiscard]] constexpr std::string_view to_string(ExtensionCategory category) noexcept {
    switch (category) {
        case ExtensionCategory::Device: return "device";
        case ExtensionCategory::Solver: return "solver";
        case ExtensionCategory::Integrator: return "integrator";
    }
    return "unknown";
}

enum class ExtensionValidationCode : std::uint8_t {
    Ok,
    MissingId,
    MissingName,
    DuplicateId,
    CategoryMismatch,
    EmptyCapability,
    DuplicateCapability,
    EmptyTelemetryField,
    DuplicateTelemetryField,
    ExtensionNotFound,
    CapabilityNotSupported
};

[[nodiscard]] constexpr std::string_view to_string(ExtensionValidationCode code) noexcept {
    switch (code) {
        case ExtensionValidationCode::Ok: return "ok";
        case ExtensionValidationCode::MissingId: return "missing_id";
        case ExtensionValidationCode::MissingName: return "missing_name";
        case ExtensionValidationCode::DuplicateId: return "duplicate_id";
        case ExtensionValidationCode::CategoryMismatch: return "category_mismatch";
        case ExtensionValidationCode::EmptyCapability: return "empty_capability";
        case ExtensionValidationCode::DuplicateCapability: return "duplicate_capability";
        case ExtensionValidationCode::EmptyTelemetryField: return "empty_telemetry_field";
        case ExtensionValidationCode::DuplicateTelemetryField: return "duplicate_telemetry_field";
        case ExtensionValidationCode::ExtensionNotFound: return "extension_not_found";
        case ExtensionValidationCode::CapabilityNotSupported: return "capability_not_supported";
    }
    return "unknown";
}

struct ExtensionValidationResult {
    ExtensionValidationCode code = ExtensionValidationCode::Ok;
    std::string extension_id;
    std::string field;
    std::string message;

    [[nodiscard]] bool ok() const noexcept {
        return code == ExtensionValidationCode::Ok;
    }

    [[nodiscard]] static ExtensionValidationResult success() {
        return {};
    }

    [[nodiscard]] static ExtensionValidationResult failure(ExtensionValidationCode failure_code,
                                                           std::string extension_id_value,
                                                           std::string field_name,
                                                           std::string message_value) {
        ExtensionValidationResult result;
        result.code = failure_code;
        result.extension_id = std::move(extension_id_value);
        result.field = std::move(field_name);
        result.message = std::move(message_value);
        return result;
    }
};

struct ExtensionMetadata {
    ExtensionCategory category = ExtensionCategory::Device;
    std::string extension_id;
    std::string display_name;
    std::string version = "builtin";
    std::vector<std::string> capabilities;
    std::vector<std::string> telemetry_fields;
};

struct ExtensionRequirement {
    ExtensionCategory category = ExtensionCategory::Device;
    std::string extension_id;
    std::string required_capability;
};

template<typename Extension>
concept ExtensionContract = requires {
    { Extension::kCategory } -> std::convertible_to<ExtensionCategory>;
    { Extension::metadata() } -> std::same_as<ExtensionMetadata>;
};

template<typename Extension>
concept DeviceExtensionContract = ExtensionContract<Extension> &&
    (Extension::kCategory == ExtensionCategory::Device);

template<typename Extension>
concept SolverExtensionContract = ExtensionContract<Extension> &&
    (Extension::kCategory == ExtensionCategory::Solver);

template<typename Extension>
concept IntegratorExtensionContract = ExtensionContract<Extension> &&
    (Extension::kCategory == ExtensionCategory::Integrator);

class ExtensionRegistry {
public:
    using MetadataMap = std::map<std::string, ExtensionMetadata, std::less<>>;

    [[nodiscard]] ExtensionValidationResult register_metadata(ExtensionMetadata metadata) {
        const ExtensionValidationResult validation = validate_metadata(metadata);
        if (!validation.ok()) {
            return validation;
        }

        MetadataMap& target_map = map_for(metadata.category);
        if (target_map.find(metadata.extension_id) != target_map.end()) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::DuplicateId,
                metadata.extension_id,
                "extension_id",
                "extension id already registered");
        }

        const std::string stable_id = metadata.extension_id;
        target_map.emplace(stable_id, std::move(metadata));
        return ExtensionValidationResult::success();
    }

    [[nodiscard]] ExtensionValidationResult register_device(ExtensionMetadata metadata) {
        if (metadata.category != ExtensionCategory::Device) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::CategoryMismatch,
                metadata.extension_id,
                "category",
                "device registration requires device category");
        }
        return register_metadata(std::move(metadata));
    }

    [[nodiscard]] ExtensionValidationResult register_solver(ExtensionMetadata metadata) {
        if (metadata.category != ExtensionCategory::Solver) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::CategoryMismatch,
                metadata.extension_id,
                "category",
                "solver registration requires solver category");
        }
        return register_metadata(std::move(metadata));
    }

    [[nodiscard]] ExtensionValidationResult register_integrator(ExtensionMetadata metadata) {
        if (metadata.category != ExtensionCategory::Integrator) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::CategoryMismatch,
                metadata.extension_id,
                "category",
                "integrator registration requires integrator category");
        }
        return register_metadata(std::move(metadata));
    }

    template<DeviceExtensionContract Extension>
    [[nodiscard]] ExtensionValidationResult register_device() {
        ExtensionMetadata metadata = Extension::metadata();
        if (metadata.category != Extension::kCategory) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::CategoryMismatch,
                metadata.extension_id,
                "category",
                "device extension category does not match contract");
        }
        return register_device(std::move(metadata));
    }

    template<SolverExtensionContract Extension>
    [[nodiscard]] ExtensionValidationResult register_solver() {
        ExtensionMetadata metadata = Extension::metadata();
        if (metadata.category != Extension::kCategory) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::CategoryMismatch,
                metadata.extension_id,
                "category",
                "solver extension category does not match contract");
        }
        return register_solver(std::move(metadata));
    }

    template<IntegratorExtensionContract Extension>
    [[nodiscard]] ExtensionValidationResult register_integrator() {
        ExtensionMetadata metadata = Extension::metadata();
        if (metadata.category != Extension::kCategory) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::CategoryMismatch,
                metadata.extension_id,
                "category",
                "integrator extension category does not match contract");
        }
        return register_integrator(std::move(metadata));
    }

    [[nodiscard]] std::optional<ExtensionMetadata> find(ExtensionCategory category,
                                                        std::string_view extension_id) const {
        const MetadataMap& map = map_for(category);
        const auto it = map.find(extension_id);
        if (it == map.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] std::vector<ExtensionMetadata> list(ExtensionCategory category) const {
        const MetadataMap& map = map_for(category);
        std::vector<ExtensionMetadata> out;
        out.reserve(map.size());
        for (const auto& item : map) {
            out.push_back(item.second);
        }
        return out;
    }

    [[nodiscard]] std::size_t size(ExtensionCategory category) const {
        return map_for(category).size();
    }

    [[nodiscard]] ExtensionValidationResult validate_requirement(
        const ExtensionRequirement& requirement) const {
        const std::optional<ExtensionMetadata> metadata = find(requirement.category,
                                                               requirement.extension_id);
        if (!metadata.has_value()) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::ExtensionNotFound,
                requirement.extension_id,
                "extension_id",
                "required extension id is not registered");
        }

        if (!requirement.required_capability.empty()) {
            const auto& caps = metadata->capabilities;
            const auto cap_it = std::find(caps.begin(), caps.end(), requirement.required_capability);
            if (cap_it == caps.end()) {
                return ExtensionValidationResult::failure(
                    ExtensionValidationCode::CapabilityNotSupported,
                    requirement.extension_id,
                    "required_capability",
                    "required capability is not declared by extension");
            }
        }

        return ExtensionValidationResult::success();
    }

    [[nodiscard]] ExtensionValidationResult validate_requirements(
        std::span<const ExtensionRequirement> requirements) const {
        for (const auto& requirement : requirements) {
            const ExtensionValidationResult status = validate_requirement(requirement);
            if (!status.ok()) {
                return status;
            }
        }
        return ExtensionValidationResult::success();
    }

private:
    [[nodiscard]] static ExtensionValidationResult validate_metadata(const ExtensionMetadata& metadata) {
        if (metadata.extension_id.empty()) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::MissingId,
                {},
                "extension_id",
                "extension id must not be empty");
        }
        if (metadata.display_name.empty()) {
            return ExtensionValidationResult::failure(
                ExtensionValidationCode::MissingName,
                metadata.extension_id,
                "display_name",
                "display name must not be empty");
        }

        {
            std::unordered_set<std::string> seen_caps;
            seen_caps.reserve(metadata.capabilities.size());
            for (std::size_t i = 0; i < metadata.capabilities.size(); ++i) {
                const std::string& capability = metadata.capabilities[i];
                if (capability.empty()) {
                    return ExtensionValidationResult::failure(
                        ExtensionValidationCode::EmptyCapability,
                        metadata.extension_id,
                        "capabilities[" + std::to_string(i) + "]",
                        "capability entry must not be empty");
                }
                if (!seen_caps.insert(capability).second) {
                    return ExtensionValidationResult::failure(
                        ExtensionValidationCode::DuplicateCapability,
                        metadata.extension_id,
                        "capabilities[" + std::to_string(i) + "]",
                        "duplicate capability entry");
                }
            }
        }

        {
            std::unordered_set<std::string> seen_telemetry;
            seen_telemetry.reserve(metadata.telemetry_fields.size());
            for (std::size_t i = 0; i < metadata.telemetry_fields.size(); ++i) {
                const std::string& field = metadata.telemetry_fields[i];
                if (field.empty()) {
                    return ExtensionValidationResult::failure(
                        ExtensionValidationCode::EmptyTelemetryField,
                        metadata.extension_id,
                        "telemetry_fields[" + std::to_string(i) + "]",
                        "telemetry field entry must not be empty");
                }
                if (!seen_telemetry.insert(field).second) {
                    return ExtensionValidationResult::failure(
                        ExtensionValidationCode::DuplicateTelemetryField,
                        metadata.extension_id,
                        "telemetry_fields[" + std::to_string(i) + "]",
                        "duplicate telemetry field entry");
                }
            }
        }

        return ExtensionValidationResult::success();
    }

    [[nodiscard]] MetadataMap& map_for(ExtensionCategory category) {
        switch (category) {
            case ExtensionCategory::Device: return devices_;
            case ExtensionCategory::Solver: return solvers_;
            case ExtensionCategory::Integrator: return integrators_;
        }
        return devices_;
    }

    [[nodiscard]] const MetadataMap& map_for(ExtensionCategory category) const {
        switch (category) {
            case ExtensionCategory::Device: return devices_;
            case ExtensionCategory::Solver: return solvers_;
            case ExtensionCategory::Integrator: return integrators_;
        }
        return devices_;
    }

    MetadataMap devices_;
    MetadataMap solvers_;
    MetadataMap integrators_;
};

template<typename Device>
[[nodiscard]] ExtensionMetadata make_builtin_device_extension(std::string extension_id) {
    const DeviceMetadata metadata = get_device_metadata<Device>();
    ExtensionMetadata ext;
    ext.category = ExtensionCategory::Device;
    ext.extension_id = std::move(extension_id);
    ext.display_name = std::string(metadata.name);
    ext.version = "builtin";
    ext.capabilities.push_back(std::string(metadata.category));
    ext.capabilities.push_back(metadata.is_linear ? "linear" : "nonlinear");
    ext.capabilities.push_back(metadata.is_dynamic ? "dynamic" : "static");
    if (metadata.has_thermal_model) {
        ext.capabilities.push_back("thermal");
    }
    ext.telemetry_fields = {"device_power", "device_temperature"};
    return ext;
}

[[nodiscard]] inline ExtensionMetadata make_builtin_solver_extension(LinearSolverKind kind) {
    ExtensionMetadata ext;
    ext.category = ExtensionCategory::Solver;
    ext.version = "builtin";
    ext.telemetry_fields = {"last_iterations", "last_error", "total_fallbacks"};

    switch (kind) {
        case LinearSolverKind::SparseLU:
            ext.extension_id = "sparse_lu";
            ext.display_name = "SparseLU";
            ext.capabilities = {"direct", "symbolic_analysis"};
            break;
        case LinearSolverKind::EnhancedSparseLU:
            ext.extension_id = "enhanced_sparse_lu";
            ext.display_name = "EnhancedSparseLU";
            ext.capabilities = {"direct", "symbolic_analysis", "pattern_reuse"};
            break;
        case LinearSolverKind::KLU:
            ext.extension_id = "klu";
            ext.display_name = "KLU";
            ext.capabilities = {"direct", "symbolic_analysis", "circuit_sparse"};
            break;
        case LinearSolverKind::GMRES:
            ext.extension_id = "gmres";
            ext.display_name = "GMRES";
            ext.capabilities = {"iterative", "preconditioned"};
            break;
        case LinearSolverKind::BiCGSTAB:
            ext.extension_id = "bicgstab";
            ext.display_name = "BiCGSTAB";
            ext.capabilities = {"iterative", "preconditioned"};
            break;
        case LinearSolverKind::CG:
            ext.extension_id = "cg";
            ext.display_name = "CG";
            ext.capabilities = {"iterative", "spd_only", "preconditioned"};
            break;
    }
    return ext;
}

[[nodiscard]] inline ExtensionMetadata make_builtin_integrator_extension(Integrator integrator) {
    ExtensionMetadata ext;
    ext.category = ExtensionCategory::Integrator;
    ext.version = "builtin";
    ext.telemetry_fields = {"dt", "integration_order", "lte"};

    switch (integrator) {
        case Integrator::Trapezoidal:
            ext.extension_id = "trapezoidal";
            ext.display_name = "Trapezoidal";
            break;
        case Integrator::BDF1:
            ext.extension_id = "bdf1";
            ext.display_name = "BDF1";
            break;
        case Integrator::BDF2:
            ext.extension_id = "bdf2";
            ext.display_name = "BDF2";
            break;
        case Integrator::BDF3:
            ext.extension_id = "bdf3";
            ext.display_name = "BDF3";
            break;
        case Integrator::BDF4:
            ext.extension_id = "bdf4";
            ext.display_name = "BDF4";
            break;
        case Integrator::BDF5:
            ext.extension_id = "bdf5";
            ext.display_name = "BDF5";
            break;
        case Integrator::Gear:
            ext.extension_id = "gear";
            ext.display_name = "Gear";
            break;
        case Integrator::TRBDF2:
            ext.extension_id = "trbdf2";
            ext.display_name = "TRBDF2";
            break;
        case Integrator::RosenbrockW:
            ext.extension_id = "rosenbrock_w";
            ext.display_name = "RosenbrockW";
            break;
        case Integrator::SDIRK2:
            ext.extension_id = "sdirk2";
            ext.display_name = "SDIRK2";
            break;
    }

    ext.capabilities.push_back("order_" + std::to_string(method_order(integrator)));
    ext.capabilities.push_back(is_stiff_stable(integrator) ? "stiff_stable" : "general");
    ext.capabilities.push_back(requires_startup(integrator) ? "startup_required" : "single_step");
    return ext;
}

[[nodiscard]] inline ExtensionRegistry make_default_extension_registry() {
    ExtensionRegistry registry;

    auto register_checked = [&registry](ExtensionMetadata metadata) {
        const ExtensionValidationResult status = registry.register_metadata(std::move(metadata));
        return status.ok();
    };

    if (!register_checked(make_builtin_device_extension<Resistor>("resistor"))) return {};
    if (!register_checked(make_builtin_device_extension<Capacitor>("capacitor"))) return {};
    if (!register_checked(make_builtin_device_extension<Inductor>("inductor"))) return {};
    if (!register_checked(make_builtin_device_extension<VoltageSource>("voltage_source"))) return {};
    if (!register_checked(make_builtin_device_extension<CurrentSource>("current_source"))) return {};
    if (!register_checked(make_builtin_device_extension<IdealDiode>("ideal_diode"))) return {};
    if (!register_checked(make_builtin_device_extension<IdealSwitch>("ideal_switch"))) return {};
    if (!register_checked(make_builtin_device_extension<VoltageControlledSwitch>("voltage_controlled_switch"))) return {};
    if (!register_checked(make_builtin_device_extension<MOSFET>("mosfet"))) return {};
    if (!register_checked(make_builtin_device_extension<IGBT>("igbt"))) return {};
    if (!register_checked(make_builtin_device_extension<Transformer>("transformer"))) return {};

    if (!register_checked(make_builtin_solver_extension(LinearSolverKind::SparseLU))) return {};
    if (!register_checked(make_builtin_solver_extension(LinearSolverKind::EnhancedSparseLU))) return {};
    if (!register_checked(make_builtin_solver_extension(LinearSolverKind::KLU))) return {};
    if (!register_checked(make_builtin_solver_extension(LinearSolverKind::GMRES))) return {};
    if (!register_checked(make_builtin_solver_extension(LinearSolverKind::BiCGSTAB))) return {};
    if (!register_checked(make_builtin_solver_extension(LinearSolverKind::CG))) return {};

    if (!register_checked(make_builtin_integrator_extension(Integrator::Trapezoidal))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::BDF1))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::BDF2))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::BDF3))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::BDF4))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::BDF5))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::Gear))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::TRBDF2))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::RosenbrockW))) return {};
    if (!register_checked(make_builtin_integrator_extension(Integrator::SDIRK2))) return {};

    return registry;
}

[[nodiscard]] inline const ExtensionRegistry& kernel_extension_registry() {
    static const ExtensionRegistry registry = make_default_extension_registry();
    return registry;
}

}  // namespace pulsim::v1
