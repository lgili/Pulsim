#pragma once

/**
 * @file metadata.hpp
 * @brief Component metadata system for GUI component palettes and property editors.
 *
 * This header provides comprehensive metadata about circuit components, enabling
 * GUI applications to:
 * - Build component palettes organized by category
 * - Generate property editors with correct field types and validation
 * - Display helpful descriptions and units
 * - Validate user input before circuit creation
 *
 * @section metadata_usage Usage Example
 * @code
 * #include "pulsim/metadata.hpp"
 *
 * // Get the singleton registry
 * const auto& registry = ComponentRegistry::instance();
 *
 * // Build a component palette
 * for (const auto& category : registry.all_categories()) {
 *     create_palette_group(category);
 *     for (auto type : registry.types_in_category(category)) {
 *         const auto* meta = registry.get(type);
 *         add_palette_item(meta->display_name, meta->symbol_id);
 *     }
 * }
 *
 * // Build a property editor for a resistor
 * const auto* meta = registry.get(ComponentType::Resistor);
 * for (const auto& param : meta->parameters) {
 *     create_property_field(param.display_name, param.type,
 *                          param.min_value, param.max_value, param.unit);
 * }
 * @endcode
 */

#include "pulsim/types.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim {

/**
 * @brief Data types for component parameters.
 *
 * Used to determine appropriate input controls in GUI property editors.
 */
enum class ParameterType {
    Real,       ///< Floating-point number (use spin box or slider)
    Integer,    ///< Integer value (use integer spin box)
    Boolean,    ///< True/false (use checkbox)
    Enum,       ///< One of several choices (use dropdown)
    String      ///< Text input (e.g., model name)
};

/**
 * @brief Metadata for a single component parameter.
 *
 * Contains all information needed to create a property editor field
 * for this parameter, including display name, validation constraints,
 * and units.
 */
struct ParameterMetadata {
    std::string name;           ///< Internal name (e.g., "resistance")
    std::string display_name;   ///< GUI display (e.g., "Resistance")
    std::string description;    ///< Help text / tooltip
    ParameterType type = ParameterType::Real;

    // For Real/Integer types
    std::optional<double> default_value;  ///< Default value (nullopt if none)
    std::optional<double> min_value;      ///< Minimum allowed (nullopt if unbounded)
    std::optional<double> max_value;      ///< Maximum allowed (nullopt if unbounded)
    std::string unit;                     ///< Unit string (e.g., "ohm", "F", "H")

    // For Enum type
    std::vector<std::string> enum_values; ///< Valid choices for Enum type

    // Validation
    bool required = true;  ///< True if parameter must be provided
};

/**
 * @brief Metadata for a component pin/terminal.
 *
 * Describes a connection point on a component for GUI display
 * and netlist generation.
 */
struct PinMetadata {
    std::string name;        ///< Pin name (e.g., "anode", "drain", "gate")
    std::string description; ///< Description (e.g., "Positive terminal")
};

/**
 * @brief Complete metadata for a component type.
 *
 * Contains all information needed to:
 * - Display the component in a palette
 * - Render the component symbol
 * - Create property editor fields
 * - Validate component parameters
 */
struct ComponentMetadata {
    ComponentType type;          ///< The component type enum value
    std::string name;            ///< Internal name (e.g., "resistor")
    std::string display_name;    ///< GUI display name (e.g., "Resistor")
    std::string description;     ///< Help text / tooltip
    std::string category;        ///< Category (e.g., "Passive", "Semiconductor", "Sources")

    std::vector<PinMetadata> pins;           ///< Component pins/terminals
    std::vector<ParameterMetadata> parameters; ///< Editable parameters

    // GUI hints
    std::string symbol_id;           ///< Reference for symbol rendering
    bool has_loss_model = false;     ///< True if power loss calculation supported
    bool has_thermal_model = false;  ///< True if thermal simulation supported
};

/**
 * @brief Singleton registry providing metadata for all component types.
 *
 * ComponentRegistry provides a central source of truth for component
 * metadata, enabling consistent GUI behavior and parameter validation.
 *
 * @section registry_categories Component Categories
 * Components are organized into categories:
 * - **Passive**: Resistor, Capacitor, Inductor
 * - **Sources**: VoltageSource, CurrentSource
 * - **Semiconductor**: Diode, MOSFET, IGBT
 * - **Switches**: Switch
 * - **Magnetic**: Transformer
 *
 * @section registry_validation Parameter Validation
 * The registry provides parameter validation:
 * @code
 * std::string error;
 * bool valid = registry.validate_parameter(
 *     ComponentType::Resistor, "resistance", user_value, &error);
 * if (!valid) {
 *     show_error_dialog(error);
 * }
 * @endcode
 */
class ComponentRegistry {
public:
    /**
     * @brief Get the singleton registry instance.
     * @return Reference to the global ComponentRegistry
     */
    static const ComponentRegistry& instance();

    /**
     * @brief Get metadata for a specific component type.
     * @param type The component type to look up
     * @return Pointer to metadata, or nullptr if type not found
     */
    const ComponentMetadata* get(ComponentType type) const;

    /**
     * @brief Get all registered component types.
     * @return Vector of all available component types
     */
    std::vector<ComponentType> all_types() const;

    /**
     * @brief Get component types in a specific category.
     * @param category Category name (e.g., "Passive", "Semiconductor")
     * @return Vector of component types in that category
     */
    std::vector<ComponentType> types_in_category(const std::string& category) const;

    /**
     * @brief Get all available categories.
     * @return Vector of category names
     */
    std::vector<std::string> all_categories() const;

    /**
     * @brief Validate a parameter value against metadata constraints.
     *
     * Checks that the value is within min/max bounds for the parameter.
     *
     * @param type Component type
     * @param param_name Parameter name (e.g., "resistance")
     * @param value Value to validate
     * @param error_message Optional output for error description
     * @return true if valid, false if invalid
     */
    bool validate_parameter(ComponentType type, const std::string& param_name, double value,
                           std::string* error_message = nullptr) const;

private:
    ComponentRegistry();
    void register_all_components();

    std::unordered_map<ComponentType, ComponentMetadata> metadata_;
};

}  // namespace pulsim
