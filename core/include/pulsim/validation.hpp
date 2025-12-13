#pragma once

/**
 * @file validation.hpp
 * @brief Circuit validation API with detailed diagnostics for GUI error display.
 *
 * This header provides comprehensive circuit validation that goes beyond
 * simple pass/fail, returning structured diagnostics suitable for GUI
 * error displays with:
 * - Severity levels (Error, Warning, Info)
 * - Specific error codes for programmatic handling
 * - Human-readable messages
 * - Location information (component, node, parameter)
 * - Related components for visual highlighting
 *
 * @section validation_usage Usage Example
 * @code
 * #include "pulsim/validation.hpp"
 *
 * ValidationResult result = validate_circuit(circuit);
 *
 * if (!result.is_valid) {
 *     // Display errors in GUI
 *     for (const auto& diag : result.errors()) {
 *         show_error_marker(diag.component_name);
 *         show_error_message(diag.message);
 *     }
 * }
 *
 * // Show warnings even if valid
 * for (const auto& diag : result.warnings()) {
 *     show_warning_icon(diag.component_name);
 * }
 * @endcode
 */

#include "pulsim/types.hpp"
#include <string>
#include <vector>

namespace pulsim {

// Forward declaration
class Circuit;

/**
 * @brief Severity levels for diagnostic messages.
 *
 * - **Error**: Circuit cannot be simulated, must be fixed
 * - **Warning**: Simulation may have issues, but can proceed
 * - **Info**: Informational message, no action required
 */
enum class DiagnosticSeverity {
    Error,      ///< Must be fixed before simulation can run
    Warning,    ///< May cause issues but simulation can proceed
    Info        ///< Informational message only
};

/**
 * @brief Specific diagnostic codes for programmatic error handling.
 *
 * Codes are prefixed by severity:
 * - `E_xxx`: Error codes
 * - `W_xxx`: Warning codes
 * - `I_xxx`: Info codes
 *
 * Use diagnostic_code_description() to get human-readable descriptions.
 */
enum class DiagnosticCode {
    // Errors (E_xxx) - simulation cannot proceed
    E_NO_GROUND,            ///< No ground reference node (node "0" not connected)
    E_VOLTAGE_SOURCE_LOOP,  ///< Voltage sources form a loop (overdetermined)
    E_INDUCTOR_LOOP,        ///< Inductors/V-sources form a loop (singular matrix)
    E_NO_DC_PATH,           ///< Node has no DC path to ground (floating)
    E_INVALID_PARAMETER,    ///< Parameter value out of valid range
    E_UNKNOWN_NODE,         ///< Referenced node doesn't exist
    E_DUPLICATE_NAME,       ///< Component name already used in circuit
    E_NO_COMPONENTS,        ///< Circuit has no components

    // Warnings (W_xxx) - simulation may have issues
    W_FLOATING_NODE,        ///< Node has only one connection (likely error)
    W_SHORT_CIRCUIT,        ///< Very low impedance path detected
    W_HIGH_VOLTAGE,         ///< Unusually high voltage expected
    W_MISSING_IC,           ///< Capacitor/inductor has no initial condition
    W_LARGE_TIMESTEP,       ///< Timestep may be too large for circuit dynamics

    // Info (I_xxx) - informational only
    I_IDEAL_SWITCH,         ///< Using ideal switch model (infinite off resistance)
    I_NO_LOSS_MODEL,        ///< Power loss calculation not available for component
    I_PARALLEL_SOURCES,     ///< Parallel voltage sources detected (may be intentional)
};

/**
 * @brief A single diagnostic message with location information.
 *
 * Contains all information needed to display the diagnostic in a GUI,
 * including the specific location (component, node, parameter) and
 * related components that should be highlighted.
 */
struct Diagnostic {
    DiagnosticSeverity severity = DiagnosticSeverity::Error;
    DiagnosticCode code = DiagnosticCode::E_NO_GROUND;
    std::string message;           ///< Human-readable error message

    // Location info (empty string if not applicable)
    std::string component_name;    ///< Component causing the issue (empty if circuit-level)
    std::string node_name;         ///< Node involved (empty if component-level)
    std::string parameter_name;    ///< Parameter name (empty if not parameter-related)

    /// Components to highlight in GUI (e.g., both ends of a short circuit)
    std::vector<std::string> related_components;
};

/**
 * @brief Result of circuit validation containing all diagnostics.
 *
 * Use is_valid to check if simulation can proceed. Even if valid,
 * check warnings for potential issues.
 *
 * @code
 * ValidationResult result = validate_circuit(circuit);
 *
 * if (result.is_valid) {
 *     // Can simulate, but check warnings
 *     if (result.has_warnings()) {
 *         show_warning_dialog("Circuit has warnings. Continue?");
 *     }
 *     simulator.run_transient();
 * } else {
 *     // Cannot simulate, show errors
 *     for (const auto& error : result.errors()) {
 *         highlight_component(error.component_name);
 *     }
 * }
 * @endcode
 */
struct ValidationResult {
    bool is_valid = true;              ///< True if no errors (warnings OK)
    std::vector<Diagnostic> diagnostics; ///< All diagnostic messages

    /// @name Query Methods
    /// @{
    bool has_errors() const;    ///< True if any Error-severity diagnostics
    bool has_warnings() const;  ///< True if any Warning-severity diagnostics

    std::vector<Diagnostic> errors() const;   ///< Get only Error diagnostics
    std::vector<Diagnostic> warnings() const; ///< Get only Warning diagnostics
    std::vector<Diagnostic> infos() const;    ///< Get only Info diagnostics
    /// @}

    /// @name Add Diagnostics (used internally)
    /// @{
    void add_error(DiagnosticCode code, const std::string& message,
                   const std::string& component = "", const std::string& node = "");
    void add_warning(DiagnosticCode code, const std::string& message,
                     const std::string& component = "", const std::string& node = "");
    void add_info(DiagnosticCode code, const std::string& message,
                  const std::string& component = "", const std::string& node = "");
    /// @}
};

/**
 * @brief Validate a circuit and return detailed diagnostics.
 *
 * Performs comprehensive validation including:
 * - Ground reference check
 * - Voltage source loop detection
 * - Floating node detection
 * - Parameter validation
 * - Duplicate name check
 *
 * @param circuit The circuit to validate
 * @return ValidationResult containing all diagnostics
 */
ValidationResult validate_circuit(const Circuit& circuit);

/**
 * @brief Get human-readable description of a diagnostic code.
 *
 * Useful for displaying help text or tooltips in GUI.
 *
 * @param code The diagnostic code
 * @return Description string
 */
std::string diagnostic_code_description(DiagnosticCode code);

}  // namespace pulsim
