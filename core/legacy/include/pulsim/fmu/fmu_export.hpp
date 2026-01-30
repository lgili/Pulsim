#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/simulation.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::fmu {

// =============================================================================
// FMU (Functional Mock-up Unit) Export Support
// Implements FMI 2.0 and 3.0 standards for Model Exchange and Co-Simulation
// =============================================================================

// FMI Version
enum class FmiVersion {
    FMI_2_0,
    FMI_3_0
};

// FMU Type
enum class FmuType {
    MODEL_EXCHANGE,
    CO_SIMULATION,
    SCHEDULED_EXECUTION  // FMI 3.0 only
};

// Variable causality
enum class Causality {
    PARAMETER,
    CALCULATED_PARAMETER,
    INPUT,
    OUTPUT,
    LOCAL,
    INDEPENDENT
};

// Variable variability
enum class Variability {
    CONSTANT,
    FIXED,
    TUNABLE,
    DISCRETE,
    CONTINUOUS
};

// Initial value handling
enum class Initial {
    EXACT,
    APPROX,
    CALCULATED,
    NOT_SPECIFIED
};

// =============================================================================
// FMU Variable Descriptor
// =============================================================================

struct FmuVariable {
    std::string name;
    std::string description;
    uint32_t value_reference;

    Causality causality = Causality::LOCAL;
    Variability variability = Variability::CONTINUOUS;
    Initial initial = Initial::NOT_SPECIFIED;

    // Type information
    enum class Type { REAL, INTEGER, BOOLEAN, STRING } type = Type::REAL;

    // For Real type
    std::string unit;
    double start = 0.0;
    double min = -std::numeric_limits<double>::infinity();
    double max = std::numeric_limits<double>::infinity();
    double nominal = 1.0;

    // Derivative info (for state variables)
    int32_t derivative_of = -1;  // value reference of state variable

    // Dependencies (for outputs)
    std::vector<uint32_t> dependencies;
};

// =============================================================================
// FMU Model Description
// =============================================================================

struct FmuModelDescription {
    std::string model_name;
    std::string model_identifier;
    std::string description;
    std::string author;
    std::string version = "1.0";
    std::string generation_tool = "Pulsim";
    std::string generation_date;
    std::string guid;

    FmiVersion fmi_version = FmiVersion::FMI_2_0;
    FmuType fmu_type = FmuType::MODEL_EXCHANGE;

    // Capabilities
    bool can_handle_variable_communication_step_size = true;
    bool can_interpolate_inputs = true;
    bool can_get_and_set_fmu_state = true;
    bool can_serialize_fmu_state = false;
    bool provides_directional_derivative = true;
    bool needs_execution_tool = false;

    // Model structure
    size_t number_of_states = 0;
    size_t number_of_event_indicators = 0;

    // Variables
    std::vector<FmuVariable> variables;

    // Default experiment
    double start_time = 0.0;
    double stop_time = 1.0;
    double tolerance = 1e-6;
    double step_size = 1e-4;
};

// =============================================================================
// FMU Exporter - Exports Pulsim circuits as FMU
// =============================================================================

struct FmuExportOptions {
    FmiVersion version = FmiVersion::FMI_2_0;
    FmuType type = FmuType::MODEL_EXCHANGE;

    std::string author = "Pulsim";
    std::string version_string = "1.0";
    std::string description;

    // Build options
    bool include_source = false;          // Include source code in FMU
    bool include_documentation = true;    // Include doc folder
    bool compress = true;                 // Create ZIP archive

    // Target platforms
    std::vector<std::string> platforms = {"linux64", "darwin64", "win64"};

    // Solver options for Co-Simulation
    double default_step_size = 1e-6;
    double relative_tolerance = 1e-4;
    double absolute_tolerance = 1e-6;

    // Variable exposure
    bool expose_all_nodes = false;        // Expose all node voltages
    bool expose_all_currents = false;     // Expose all branch currents
    std::vector<std::string> expose_nodes;    // Specific nodes to expose
    std::vector<std::string> expose_currents; // Specific currents to expose
};

class FmuExporter {
public:
    explicit FmuExporter(FmuExportOptions options = {});

    // Export circuit as FMU
    bool export_fmu(const Circuit& circuit,
                    const std::filesystem::path& output_path,
                    const std::string& model_name = "");

    // Export with simulation options
    bool export_fmu(const Circuit& circuit,
                    const SimulationOptions& sim_options,
                    const std::filesystem::path& output_path,
                    const std::string& model_name = "");

    // Get/set options
    const FmuExportOptions& options() const { return options_; }
    void set_options(const FmuExportOptions& options) { options_ = options; }

    // Get generated model description
    const FmuModelDescription& model_description() const { return model_desc_; }

    // Error handling
    const std::vector<std::string>& errors() const { return errors_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    FmuExportOptions options_;
    FmuModelDescription model_desc_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;

    // Build model description from circuit
    void build_model_description(const Circuit& circuit,
                                 const SimulationOptions& sim_options,
                                 const std::string& model_name);

    // Generate GUID
    std::string generate_guid();

    // Generate modelDescription.xml
    std::string generate_model_description_xml();

    // Generate FMU source code
    std::string generate_fmu_source();

    // Generate FMU header
    std::string generate_fmu_header();

    // Create FMU directory structure
    bool create_fmu_structure(const std::filesystem::path& fmu_dir);

    // Write files
    bool write_model_description(const std::filesystem::path& fmu_dir);
    bool write_source_files(const std::filesystem::path& fmu_dir);
    bool write_binaries(const std::filesystem::path& fmu_dir);

    // Create ZIP archive
    bool create_fmu_archive(const std::filesystem::path& fmu_dir,
                            const std::filesystem::path& output_path);

    // Variable mapping
    uint32_t next_value_reference_ = 0;
    std::unordered_map<std::string, uint32_t> node_to_vr_;
    std::unordered_map<std::string, uint32_t> state_to_vr_;
};

// =============================================================================
// FMU Model Exchange Implementation
// Runtime component embedded in the FMU
// =============================================================================

// FMU state storage
struct FmuState {
    double time = 0.0;
    std::vector<double> reals;
    std::vector<int32_t> integers;
    std::vector<bool> booleans;
    std::vector<double> states;
    std::vector<double> derivatives;
    std::vector<double> event_indicators;
    bool initialized = false;
};

// FMU Model Exchange interface
class FmuModelExchange {
public:
    FmuModelExchange();
    ~FmuModelExchange();

    // Initialization
    bool initialize(const std::string& instance_name,
                    const std::string& guid,
                    bool logging_on);

    // Setup
    bool setup_experiment(double start_time,
                          bool stop_time_defined,
                          double stop_time,
                          bool tolerance_defined,
                          double tolerance);

    bool enter_initialization_mode();
    bool exit_initialization_mode();

    // State machine
    bool enter_continuous_time_mode();
    bool enter_event_mode();
    bool new_discrete_states(bool& discrete_states_need_update,
                             bool& terminate_simulation,
                             bool& nominals_of_continuous_states_changed,
                             bool& values_of_continuous_states_changed,
                             bool& next_event_time_defined,
                             double& next_event_time);

    // Get/Set operations
    bool get_real(const uint32_t* vr, size_t nvr, double* values);
    bool set_real(const uint32_t* vr, size_t nvr, const double* values);
    bool get_integer(const uint32_t* vr, size_t nvr, int32_t* values);
    bool set_integer(const uint32_t* vr, size_t nvr, const int32_t* values);
    bool get_boolean(const uint32_t* vr, size_t nvr, bool* values);
    bool set_boolean(const uint32_t* vr, size_t nvr, const bool* values);

    // Continuous states
    bool get_continuous_states(double* states, size_t nx);
    bool set_continuous_states(const double* states, size_t nx);
    bool get_derivatives(double* derivatives, size_t nx);
    bool get_nominals_of_continuous_states(double* nominals, size_t nx);

    // Event indicators
    bool get_event_indicators(double* event_indicators, size_t ni);

    // Time
    bool set_time(double time);

    // Completed integrator step
    bool completed_integrator_step(bool no_set_fmu_state_prior,
                                   bool& enter_event_mode,
                                   bool& terminate_simulation);

    // State serialization
    bool get_fmu_state(void** fmu_state);
    bool set_fmu_state(void* fmu_state);
    bool free_fmu_state(void** fmu_state);

    // Directional derivatives
    bool get_directional_derivative(const uint32_t* unknown_refs, size_t n_unknown,
                                    const uint32_t* known_refs, size_t n_known,
                                    const double* seed, double* sensitivity);

    // Terminate
    bool terminate();
    bool reset();

private:
    std::unique_ptr<FmuState> state_;
    std::unique_ptr<Circuit> circuit_;

    double current_time_ = 0.0;
    bool initialized_ = false;
    bool in_event_mode_ = false;
    bool in_continuous_mode_ = false;

    // Variable storage
    std::vector<double> continuous_states_;
    std::vector<double> derivatives_;
    std::vector<double> event_indicators_;
    std::vector<double> previous_event_indicators_;

    // Update circuit state from continuous states
    void update_circuit_state();

    // Compute derivatives from circuit
    void compute_derivatives();

    // Check for events
    bool check_events();
};

// =============================================================================
// FMU Co-Simulation Implementation
// =============================================================================

class FmuCoSimulation {
public:
    FmuCoSimulation();
    ~FmuCoSimulation();

    // Initialization (same as Model Exchange)
    bool initialize(const std::string& instance_name,
                    const std::string& guid,
                    bool logging_on);

    bool setup_experiment(double start_time,
                          bool stop_time_defined,
                          double stop_time,
                          bool tolerance_defined,
                          double tolerance);

    bool enter_initialization_mode();
    bool exit_initialization_mode();

    // Get/Set operations
    bool get_real(const uint32_t* vr, size_t nvr, double* values);
    bool set_real(const uint32_t* vr, size_t nvr, const double* values);
    bool get_integer(const uint32_t* vr, size_t nvr, int32_t* values);
    bool set_integer(const uint32_t* vr, size_t nvr, const int32_t* values);
    bool get_boolean(const uint32_t* vr, size_t nvr, bool* values);
    bool set_boolean(const uint32_t* vr, size_t nvr, const bool* values);

    // Co-Simulation specific
    bool do_step(double current_communication_point,
                 double communication_step_size,
                 bool no_set_fmu_state_prior,
                 bool& early_return);

    bool cancel_step();

    // Get step status
    bool get_status(int* status);
    bool get_real_status(int kind, double* value);
    bool get_integer_status(int kind, int* value);
    bool get_boolean_status(int kind, bool* value);
    bool get_string_status(int kind, std::string& value);

    // State serialization
    bool get_fmu_state(void** fmu_state);
    bool set_fmu_state(void* fmu_state);
    bool free_fmu_state(void** fmu_state);

    // Terminate
    bool terminate();
    bool reset();

private:
    std::unique_ptr<FmuState> state_;
    std::unique_ptr<Circuit> circuit_;

    double current_time_ = 0.0;
    double step_size_ = 1e-6;
    double tolerance_ = 1e-6;
    bool initialized_ = false;

    // Input/output buffers
    std::unordered_map<uint32_t, double> input_values_;
    std::unordered_map<uint32_t, double> output_values_;

    // Apply inputs to circuit
    void apply_inputs();

    // Read outputs from circuit
    void read_outputs();
};

}  // namespace pulsim::fmu
