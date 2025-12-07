#include "pulsim/fmu/fmu_export.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

namespace pulsim::fmu {

// =============================================================================
// FmuExporter Implementation
// =============================================================================

FmuExporter::FmuExporter(FmuExportOptions options)
    : options_(std::move(options))
{}

bool FmuExporter::export_fmu(const Circuit& circuit,
                              const std::filesystem::path& output_path,
                              const std::string& model_name) {
    SimulationOptions sim_options;
    sim_options.tstop = 1.0;
    sim_options.dt = 1e-6;
    return export_fmu(circuit, sim_options, output_path, model_name);
}

bool FmuExporter::export_fmu(const Circuit& circuit,
                              const SimulationOptions& sim_options,
                              const std::filesystem::path& output_path,
                              const std::string& model_name) {
    errors_.clear();
    warnings_.clear();

    // Determine model name
    std::string name = model_name.empty() ? circuit.name() : model_name;
    if (name.empty()) {
        name = "PulsimModel";
    }

    // Build model description
    build_model_description(circuit, sim_options, name);

    // Create temporary FMU directory
    std::filesystem::path fmu_dir = output_path.parent_path() / (name + "_fmu_temp");
    std::filesystem::remove_all(fmu_dir);

    if (!create_fmu_structure(fmu_dir)) {
        errors_.push_back("Failed to create FMU directory structure");
        return false;
    }

    // Write modelDescription.xml
    if (!write_model_description(fmu_dir)) {
        errors_.push_back("Failed to write modelDescription.xml");
        return false;
    }

    // Write source files
    if (!write_source_files(fmu_dir)) {
        errors_.push_back("Failed to write source files");
        return false;
    }

    // Create FMU archive
    if (options_.compress) {
        if (!create_fmu_archive(fmu_dir, output_path)) {
            errors_.push_back("Failed to create FMU archive");
            return false;
        }
        // Clean up temp directory
        std::filesystem::remove_all(fmu_dir);
    } else {
        // Just rename the directory
        std::filesystem::rename(fmu_dir, output_path);
    }

    return errors_.empty();
}

void FmuExporter::build_model_description(const Circuit& circuit,
                                           const SimulationOptions& sim_options,
                                           const std::string& model_name) {
    model_desc_ = FmuModelDescription();
    model_desc_.model_name = model_name;
    model_desc_.model_identifier = model_name;
    model_desc_.description = options_.description.empty() ?
        "Pulsim circuit simulation model" : options_.description;
    model_desc_.author = options_.author;
    model_desc_.version = options_.version_string;
    model_desc_.fmi_version = options_.version;
    model_desc_.fmu_type = options_.type;
    model_desc_.guid = generate_guid();

    // Generation timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    model_desc_.generation_date = ss.str();

    // Default experiment
    model_desc_.start_time = 0.0;
    model_desc_.stop_time = sim_options.tstop;
    model_desc_.tolerance = sim_options.reltol;
    model_desc_.step_size = sim_options.dt;

    next_value_reference_ = 0;
    node_to_vr_.clear();
    state_to_vr_.clear();

    // Add time as independent variable
    FmuVariable time_var;
    time_var.name = "time";
    time_var.description = "Simulation time";
    time_var.value_reference = next_value_reference_++;
    time_var.causality = Causality::INDEPENDENT;
    time_var.variability = Variability::CONTINUOUS;
    time_var.unit = "s";
    model_desc_.variables.push_back(time_var);

    // Expose node voltages
    std::vector<std::string> nodes_to_expose;
    if (options_.expose_all_nodes) {
        nodes_to_expose = circuit.node_names();
    } else {
        nodes_to_expose = options_.expose_nodes;
    }

    for (const auto& node : nodes_to_expose) {
        if (node == "0" || node == "GND") continue;  // Skip ground

        FmuVariable var;
        var.name = "V_" + node;
        var.description = "Voltage at node " + node;
        var.value_reference = next_value_reference_++;
        var.causality = Causality::OUTPUT;
        var.variability = Variability::CONTINUOUS;
        var.unit = "V";
        var.start = 0.0;

        node_to_vr_[node] = var.value_reference;
        model_desc_.variables.push_back(var);
    }

    // Add state variables for capacitors and inductors
    size_t state_index = 0;
    for (const auto& comp : circuit.components()) {
        if (comp.type() == ComponentType::Capacitor) {
            // Capacitor voltage is a state variable
            const auto& cap_params = std::get<CapacitorParams>(comp.params());
            FmuVariable state_var;
            state_var.name = "x_" + comp.name() + "_v";
            state_var.description = "Capacitor voltage state for " + comp.name();
            state_var.value_reference = next_value_reference_++;
            state_var.causality = Causality::LOCAL;
            state_var.variability = Variability::CONTINUOUS;
            state_var.unit = "V";
            state_var.start = cap_params.initial_voltage;

            state_to_vr_[comp.name()] = state_var.value_reference;
            model_desc_.variables.push_back(state_var);

            // Derivative of state
            FmuVariable der_var;
            der_var.name = "der_" + comp.name() + "_v";
            der_var.description = "Derivative of capacitor voltage for " + comp.name();
            der_var.value_reference = next_value_reference_++;
            der_var.causality = Causality::LOCAL;
            der_var.variability = Variability::CONTINUOUS;
            der_var.unit = "V/s";
            der_var.derivative_of = static_cast<int32_t>(state_var.value_reference);

            model_desc_.variables.push_back(der_var);
            state_index++;
        } else if (comp.type() == ComponentType::Inductor) {
            // Inductor current is a state variable
            const auto& ind_params = std::get<InductorParams>(comp.params());
            FmuVariable state_var;
            state_var.name = "x_" + comp.name() + "_i";
            state_var.description = "Inductor current state for " + comp.name();
            state_var.value_reference = next_value_reference_++;
            state_var.causality = Causality::LOCAL;
            state_var.variability = Variability::CONTINUOUS;
            state_var.unit = "A";
            state_var.start = ind_params.initial_current;

            state_to_vr_[comp.name()] = state_var.value_reference;
            model_desc_.variables.push_back(state_var);

            // Derivative of state
            FmuVariable der_var;
            der_var.name = "der_" + comp.name() + "_i";
            der_var.description = "Derivative of inductor current for " + comp.name();
            der_var.value_reference = next_value_reference_++;
            der_var.causality = Causality::LOCAL;
            der_var.variability = Variability::CONTINUOUS;
            der_var.unit = "A/s";
            der_var.derivative_of = static_cast<int32_t>(state_var.value_reference);

            model_desc_.variables.push_back(der_var);
            state_index++;
        }
    }

    model_desc_.number_of_states = state_index;

    // Add parameters for component values
    for (const auto& comp : circuit.components()) {
        FmuVariable param;
        param.name = comp.name() + "_value";
        param.description = "Value of component " + comp.name();
        param.value_reference = next_value_reference_++;
        param.causality = Causality::PARAMETER;
        param.variability = Variability::TUNABLE;

        switch (comp.type()) {
            case ComponentType::Resistor: {
                const auto& r_params = std::get<ResistorParams>(comp.params());
                param.start = r_params.resistance;
                param.unit = "Ohm";
                param.min = 1e-12;
                break;
            }
            case ComponentType::Capacitor: {
                const auto& c_params = std::get<CapacitorParams>(comp.params());
                param.start = c_params.capacitance;
                param.unit = "F";
                param.min = 1e-18;
                break;
            }
            case ComponentType::Inductor: {
                const auto& l_params = std::get<InductorParams>(comp.params());
                param.start = l_params.inductance;
                param.unit = "H";
                param.min = 1e-15;
                break;
            }
            case ComponentType::VoltageSource:
                param.start = 0.0;  // DC value from waveform
                param.unit = "V";
                break;
            case ComponentType::CurrentSource:
                param.start = 0.0;  // DC value from waveform
                param.unit = "A";
                break;
            default:
                param.start = 0.0;
                break;
        }

        model_desc_.variables.push_back(param);
    }

    // Add input variables for sources
    for (const auto& comp : circuit.components()) {
        if (comp.type() == ComponentType::VoltageSource ||
            comp.type() == ComponentType::CurrentSource) {
            FmuVariable input;
            input.name = comp.name() + "_input";
            input.description = "Input value for source " + comp.name();
            input.value_reference = next_value_reference_++;
            input.causality = Causality::INPUT;
            input.variability = Variability::CONTINUOUS;
            input.unit = (comp.type() == ComponentType::VoltageSource) ? "V" : "A";
            input.start = 0.0;

            model_desc_.variables.push_back(input);
        }
    }
}

std::string FmuExporter::generate_guid() {
    // Generate a UUID-like GUID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << std::hex;

    for (int i = 0; i < 8; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++) ss << dis(gen);
    ss << "-4";  // Version 4 UUID
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    ss << ((dis(gen) & 0x3) | 0x8);  // Variant
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << dis(gen);

    return ss.str();
}

std::string FmuExporter::generate_model_description_xml() {
    std::stringstream xml;

    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";

    if (options_.version == FmiVersion::FMI_2_0) {
        xml << "<fmiModelDescription\n";
        xml << "  fmiVersion=\"2.0\"\n";
    } else {
        xml << "<fmiModelDescription\n";
        xml << "  fmiVersion=\"3.0\"\n";
    }

    xml << "  modelName=\"" << model_desc_.model_name << "\"\n";
    xml << "  guid=\"" << model_desc_.guid << "\"\n";
    xml << "  description=\"" << model_desc_.description << "\"\n";
    xml << "  author=\"" << model_desc_.author << "\"\n";
    xml << "  version=\"" << model_desc_.version << "\"\n";
    xml << "  generationTool=\"" << model_desc_.generation_tool << "\"\n";
    xml << "  generationDateAndTime=\"" << model_desc_.generation_date << "\"\n";
    xml << "  variableNamingConvention=\"structured\"\n";
    xml << "  numberOfEventIndicators=\"" << model_desc_.number_of_event_indicators << "\">\n\n";

    // Model Exchange or Co-Simulation
    if (options_.type == FmuType::MODEL_EXCHANGE) {
        xml << "  <ModelExchange modelIdentifier=\"" << model_desc_.model_identifier << "\">\n";
        xml << "    <SourceFiles>\n";
        xml << "      <File name=\"fmu_model.cpp\"/>\n";
        xml << "    </SourceFiles>\n";
        xml << "  </ModelExchange>\n\n";
    } else {
        xml << "  <CoSimulation modelIdentifier=\"" << model_desc_.model_identifier << "\"\n";
        xml << "    canHandleVariableCommunicationStepSize=\"true\"\n";
        xml << "    canInterpolateInputs=\"true\"\n";
        xml << "    canGetAndSetFMUstate=\"true\">\n";
        xml << "    <SourceFiles>\n";
        xml << "      <File name=\"fmu_model.cpp\"/>\n";
        xml << "    </SourceFiles>\n";
        xml << "  </CoSimulation>\n\n";
    }

    // Unit definitions
    xml << "  <UnitDefinitions>\n";
    xml << "    <Unit name=\"V\"><BaseUnit kg=\"1\" m=\"2\" s=\"-3\" A=\"-1\"/></Unit>\n";
    xml << "    <Unit name=\"A\"><BaseUnit A=\"1\"/></Unit>\n";
    xml << "    <Unit name=\"Ohm\"><BaseUnit kg=\"1\" m=\"2\" s=\"-3\" A=\"-2\"/></Unit>\n";
    xml << "    <Unit name=\"F\"><BaseUnit kg=\"-1\" m=\"-2\" s=\"4\" A=\"2\"/></Unit>\n";
    xml << "    <Unit name=\"H\"><BaseUnit kg=\"1\" m=\"2\" s=\"-2\" A=\"-2\"/></Unit>\n";
    xml << "    <Unit name=\"s\"><BaseUnit s=\"1\"/></Unit>\n";
    xml << "    <Unit name=\"V/s\"><BaseUnit kg=\"1\" m=\"2\" s=\"-4\" A=\"-1\"/></Unit>\n";
    xml << "    <Unit name=\"A/s\"><BaseUnit s=\"-1\" A=\"1\"/></Unit>\n";
    xml << "  </UnitDefinitions>\n\n";

    // Default experiment
    xml << "  <DefaultExperiment\n";
    xml << "    startTime=\"" << model_desc_.start_time << "\"\n";
    xml << "    stopTime=\"" << model_desc_.stop_time << "\"\n";
    xml << "    tolerance=\"" << model_desc_.tolerance << "\"\n";
    xml << "    stepSize=\"" << model_desc_.step_size << "\"/>\n\n";

    // Model variables
    xml << "  <ModelVariables>\n";
    for (const auto& var : model_desc_.variables) {
        xml << "    <ScalarVariable name=\"" << var.name << "\"\n";
        xml << "      valueReference=\"" << var.value_reference << "\"\n";
        xml << "      description=\"" << var.description << "\"\n";

        // Causality
        switch (var.causality) {
            case Causality::PARAMETER:
                xml << "      causality=\"parameter\"\n";
                break;
            case Causality::CALCULATED_PARAMETER:
                xml << "      causality=\"calculatedParameter\"\n";
                break;
            case Causality::INPUT:
                xml << "      causality=\"input\"\n";
                break;
            case Causality::OUTPUT:
                xml << "      causality=\"output\"\n";
                break;
            case Causality::LOCAL:
                xml << "      causality=\"local\"\n";
                break;
            case Causality::INDEPENDENT:
                xml << "      causality=\"independent\"\n";
                break;
        }

        // Variability
        switch (var.variability) {
            case Variability::CONSTANT:
                xml << "      variability=\"constant\"\n";
                break;
            case Variability::FIXED:
                xml << "      variability=\"fixed\"\n";
                break;
            case Variability::TUNABLE:
                xml << "      variability=\"tunable\"\n";
                break;
            case Variability::DISCRETE:
                xml << "      variability=\"discrete\"\n";
                break;
            case Variability::CONTINUOUS:
                xml << "      variability=\"continuous\"\n";
                break;
        }

        xml << "    >\n";

        // Type element
        xml << "      <Real";
        if (!var.unit.empty()) {
            xml << " unit=\"" << var.unit << "\"";
        }
        xml << " start=\"" << var.start << "\"";
        if (var.derivative_of >= 0) {
            xml << " derivative=\"" << var.derivative_of << "\"";
        }
        xml << "/>\n";

        xml << "    </ScalarVariable>\n";
    }
    xml << "  </ModelVariables>\n\n";

    // Model structure
    xml << "  <ModelStructure>\n";

    // Outputs
    xml << "    <Outputs>\n";
    for (size_t i = 0; i < model_desc_.variables.size(); i++) {
        if (model_desc_.variables[i].causality == Causality::OUTPUT) {
            xml << "      <Unknown index=\"" << (i + 1) << "\"/>\n";
        }
    }
    xml << "    </Outputs>\n";

    // Derivatives
    xml << "    <Derivatives>\n";
    for (size_t i = 0; i < model_desc_.variables.size(); i++) {
        if (model_desc_.variables[i].derivative_of >= 0) {
            xml << "      <Unknown index=\"" << (i + 1) << "\"/>\n";
        }
    }
    xml << "    </Derivatives>\n";

    // Initial unknowns
    xml << "    <InitialUnknowns>\n";
    for (size_t i = 0; i < model_desc_.variables.size(); i++) {
        const auto& var = model_desc_.variables[i];
        if (var.causality == Causality::OUTPUT ||
            var.derivative_of >= 0) {
            xml << "      <Unknown index=\"" << (i + 1) << "\"/>\n";
        }
    }
    xml << "    </InitialUnknowns>\n";

    xml << "  </ModelStructure>\n";
    xml << "</fmiModelDescription>\n";

    return xml.str();
}

std::string FmuExporter::generate_fmu_header() {
    std::stringstream h;

    h << "// Auto-generated FMU header for " << model_desc_.model_name << "\n";
    h << "// Generated by Pulsim FMU Exporter\n\n";

    h << "#ifndef FMU_MODEL_H\n";
    h << "#define FMU_MODEL_H\n\n";

    h << "#include \"fmi2Functions.h\"\n\n";

    h << "// Model constants\n";
    h << "#define MODEL_IDENTIFIER \"" << model_desc_.model_identifier << "\"\n";
    h << "#define MODEL_GUID \"" << model_desc_.guid << "\"\n";
    h << "#define NUMBER_OF_STATES " << model_desc_.number_of_states << "\n";
    h << "#define NUMBER_OF_EVENT_INDICATORS " << model_desc_.number_of_event_indicators << "\n";
    h << "#define NUMBER_OF_REALS " << model_desc_.variables.size() << "\n\n";

    h << "// Value references\n";
    for (const auto& var : model_desc_.variables) {
        std::string upper_name = var.name;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);
        std::replace(upper_name.begin(), upper_name.end(), '.', '_');
        h << "#define VR_" << upper_name << " " << var.value_reference << "\n";
    }

    h << "\n#endif // FMU_MODEL_H\n";

    return h.str();
}

std::string FmuExporter::generate_fmu_source() {
    std::stringstream src;

    src << "// Auto-generated FMU source for " << model_desc_.model_name << "\n";
    src << "// Generated by Pulsim FMU Exporter\n\n";

    src << "#include \"fmu_model.h\"\n";
    src << "#include <cmath>\n";
    src << "#include <cstring>\n";
    src << "#include <vector>\n\n";

    src << "// FMU state structure\n";
    src << "struct FmuState {\n";
    src << "    double time;\n";
    src << "    double reals[NUMBER_OF_REALS];\n";
    src << "    double states[NUMBER_OF_STATES];\n";
    src << "    double derivatives[NUMBER_OF_STATES];\n";
    src << "    double event_indicators[NUMBER_OF_EVENT_INDICATORS > 0 ? NUMBER_OF_EVENT_INDICATORS : 1];\n";
    src << "    bool initialized;\n";
    src << "};\n\n";

    // FMI 2.0 function implementations
    src << "extern \"C\" {\n\n";

    // fmi2Instantiate
    src << "fmi2Component fmi2Instantiate(\n";
    src << "    fmi2String instanceName,\n";
    src << "    fmi2Type fmuType,\n";
    src << "    fmi2String fmuGUID,\n";
    src << "    fmi2String fmuResourceLocation,\n";
    src << "    const fmi2CallbackFunctions* functions,\n";
    src << "    fmi2Boolean visible,\n";
    src << "    fmi2Boolean loggingOn) {\n";
    src << "    \n";
    src << "    if (strcmp(fmuGUID, MODEL_GUID) != 0) return nullptr;\n";
    src << "    \n";
    src << "    auto* state = new FmuState();\n";
    src << "    memset(state, 0, sizeof(FmuState));\n";
    src << "    \n";
    src << "    // Set initial values\n";

    for (const auto& var : model_desc_.variables) {
        src << "    state->reals[" << var.value_reference << "] = "
            << var.start << ";  // " << var.name << "\n";
    }

    src << "    \n";
    src << "    return state;\n";
    src << "}\n\n";

    // fmi2FreeInstance
    src << "void fmi2FreeInstance(fmi2Component c) {\n";
    src << "    delete static_cast<FmuState*>(c);\n";
    src << "}\n\n";

    // fmi2SetupExperiment
    src << "fmi2Status fmi2SetupExperiment(\n";
    src << "    fmi2Component c,\n";
    src << "    fmi2Boolean toleranceDefined,\n";
    src << "    fmi2Real tolerance,\n";
    src << "    fmi2Real startTime,\n";
    src << "    fmi2Boolean stopTimeDefined,\n";
    src << "    fmi2Real stopTime) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    state->time = startTime;\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2EnterInitializationMode
    src << "fmi2Status fmi2EnterInitializationMode(fmi2Component c) {\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2ExitInitializationMode
    src << "fmi2Status fmi2ExitInitializationMode(fmi2Component c) {\n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    state->initialized = true;\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2GetReal
    src << "fmi2Status fmi2GetReal(\n";
    src << "    fmi2Component c,\n";
    src << "    const fmi2ValueReference vr[],\n";
    src << "    size_t nvr,\n";
    src << "    fmi2Real value[]) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    for (size_t i = 0; i < nvr; i++) {\n";
    src << "        if (vr[i] >= NUMBER_OF_REALS) return fmi2Error;\n";
    src << "        value[i] = state->reals[vr[i]];\n";
    src << "    }\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2SetReal
    src << "fmi2Status fmi2SetReal(\n";
    src << "    fmi2Component c,\n";
    src << "    const fmi2ValueReference vr[],\n";
    src << "    size_t nvr,\n";
    src << "    const fmi2Real value[]) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    for (size_t i = 0; i < nvr; i++) {\n";
    src << "        if (vr[i] >= NUMBER_OF_REALS) return fmi2Error;\n";
    src << "        state->reals[vr[i]] = value[i];\n";
    src << "    }\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2SetTime (Model Exchange)
    src << "fmi2Status fmi2SetTime(fmi2Component c, fmi2Real time) {\n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    state->time = time;\n";
    src << "    state->reals[0] = time;  // time variable\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2GetContinuousStates
    src << "fmi2Status fmi2GetContinuousStates(\n";
    src << "    fmi2Component c,\n";
    src << "    fmi2Real x[],\n";
    src << "    size_t nx) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    for (size_t i = 0; i < nx && i < NUMBER_OF_STATES; i++) {\n";
    src << "        x[i] = state->states[i];\n";
    src << "    }\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2SetContinuousStates
    src << "fmi2Status fmi2SetContinuousStates(\n";
    src << "    fmi2Component c,\n";
    src << "    const fmi2Real x[],\n";
    src << "    size_t nx) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    for (size_t i = 0; i < nx && i < NUMBER_OF_STATES; i++) {\n";
    src << "        state->states[i] = x[i];\n";
    src << "    }\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2GetDerivatives
    src << "fmi2Status fmi2GetDerivatives(\n";
    src << "    fmi2Component c,\n";
    src << "    fmi2Real derivatives[],\n";
    src << "    size_t nx) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    // TODO: Compute derivatives from circuit equations\n";
    src << "    for (size_t i = 0; i < nx && i < NUMBER_OF_STATES; i++) {\n";
    src << "        derivatives[i] = state->derivatives[i];\n";
    src << "    }\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2DoStep (Co-Simulation)
    src << "fmi2Status fmi2DoStep(\n";
    src << "    fmi2Component c,\n";
    src << "    fmi2Real currentCommunicationPoint,\n";
    src << "    fmi2Real communicationStepSize,\n";
    src << "    fmi2Boolean noSetFMUStatePriorToCurrentPoint) {\n";
    src << "    \n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    // TODO: Integrate circuit from currentCommunicationPoint to\n";
    src << "    //       currentCommunicationPoint + communicationStepSize\n";
    src << "    state->time = currentCommunicationPoint + communicationStepSize;\n";
    src << "    state->reals[0] = state->time;\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2Terminate
    src << "fmi2Status fmi2Terminate(fmi2Component c) {\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    // fmi2Reset
    src << "fmi2Status fmi2Reset(fmi2Component c) {\n";
    src << "    auto* state = static_cast<FmuState*>(c);\n";
    src << "    state->time = 0.0;\n";
    src << "    state->initialized = false;\n";
    src << "    return fmi2OK;\n";
    src << "}\n\n";

    src << "}  // extern \"C\"\n";

    return src.str();
}

bool FmuExporter::create_fmu_structure(const std::filesystem::path& fmu_dir) {
    try {
        std::filesystem::create_directories(fmu_dir);
        std::filesystem::create_directories(fmu_dir / "sources");
        std::filesystem::create_directories(fmu_dir / "binaries");
        std::filesystem::create_directories(fmu_dir / "documentation");

        for (const auto& platform : options_.platforms) {
            std::filesystem::create_directories(fmu_dir / "binaries" / platform);
        }

        return true;
    } catch (const std::exception& e) {
        errors_.push_back(std::string("Failed to create directories: ") + e.what());
        return false;
    }
}

bool FmuExporter::write_model_description(const std::filesystem::path& fmu_dir) {
    try {
        std::ofstream file(fmu_dir / "modelDescription.xml");
        if (!file.is_open()) return false;

        file << generate_model_description_xml();
        return true;
    } catch (...) {
        return false;
    }
}

bool FmuExporter::write_source_files(const std::filesystem::path& fmu_dir) {
    try {
        // Write header
        {
            std::ofstream file(fmu_dir / "sources" / "fmu_model.h");
            if (!file.is_open()) return false;
            file << generate_fmu_header();
        }

        // Write source
        {
            std::ofstream file(fmu_dir / "sources" / "fmu_model.cpp");
            if (!file.is_open()) return false;
            file << generate_fmu_source();
        }

        // Copy FMI headers (would need to be bundled with Pulsim)
        // For now, create stub headers
        {
            std::ofstream file(fmu_dir / "sources" / "fmi2Functions.h");
            if (!file.is_open()) return false;

            file << "// FMI 2.0 Function Types\n";
            file << "#ifndef FMI2_FUNCTIONS_H\n";
            file << "#define FMI2_FUNCTIONS_H\n\n";
            file << "#include <stddef.h>\n\n";
            file << "typedef void* fmi2Component;\n";
            file << "typedef unsigned int fmi2ValueReference;\n";
            file << "typedef double fmi2Real;\n";
            file << "typedef int fmi2Integer;\n";
            file << "typedef int fmi2Boolean;\n";
            file << "typedef const char* fmi2String;\n\n";
            file << "typedef enum {\n";
            file << "    fmi2ModelExchange,\n";
            file << "    fmi2CoSimulation\n";
            file << "} fmi2Type;\n\n";
            file << "typedef enum {\n";
            file << "    fmi2OK,\n";
            file << "    fmi2Warning,\n";
            file << "    fmi2Discard,\n";
            file << "    fmi2Error,\n";
            file << "    fmi2Fatal,\n";
            file << "    fmi2Pending\n";
            file << "} fmi2Status;\n\n";
            file << "typedef struct {\n";
            file << "    void* componentEnvironment;\n";
            file << "} fmi2CallbackFunctions;\n\n";
            file << "#endif\n";
        }

        return true;
    } catch (...) {
        return false;
    }
}

bool FmuExporter::write_binaries(const std::filesystem::path& /*fmu_dir*/) {
    // Note: Actual binary compilation would require invoking compiler
    // This is a placeholder - real implementation would compile the sources
    warnings_.push_back("Binary compilation not implemented - sources only");
    return true;
}

bool FmuExporter::create_fmu_archive(const std::filesystem::path& fmu_dir,
                                      const std::filesystem::path& output_path) {
    // Simple implementation using system zip command
    // For production, use a proper ZIP library like libzip or miniz

    std::string cmd = "cd \"" + fmu_dir.string() + "\" && zip -r \"" +
                      output_path.string() + "\" .";

    int result = std::system(cmd.c_str());
    return result == 0;
}

// =============================================================================
// FmuModelExchange Implementation
// =============================================================================

FmuModelExchange::FmuModelExchange() = default;
FmuModelExchange::~FmuModelExchange() = default;

bool FmuModelExchange::initialize(const std::string& /*instance_name*/,
                                   const std::string& /*guid*/,
                                   bool /*logging_on*/) {
    // Create circuit and simulation
    circuit_ = std::make_unique<Circuit>();
    return true;
}

bool FmuModelExchange::setup_experiment(double start_time,
                                         bool /*stop_time_defined*/,
                                         double /*stop_time*/,
                                         bool /*tolerance_defined*/,
                                         double /*tolerance*/) {
    current_time_ = start_time;
    return true;
}

bool FmuModelExchange::enter_initialization_mode() {
    return true;
}

bool FmuModelExchange::exit_initialization_mode() {
    initialized_ = true;
    return true;
}

bool FmuModelExchange::enter_continuous_time_mode() {
    in_continuous_mode_ = true;
    in_event_mode_ = false;
    return true;
}

bool FmuModelExchange::enter_event_mode() {
    in_event_mode_ = true;
    in_continuous_mode_ = false;
    return true;
}

bool FmuModelExchange::new_discrete_states(bool& discrete_states_need_update,
                                            bool& terminate_simulation,
                                            bool& nominals_of_continuous_states_changed,
                                            bool& values_of_continuous_states_changed,
                                            bool& next_event_time_defined,
                                            double& /*next_event_time*/) {
    discrete_states_need_update = false;
    terminate_simulation = false;
    nominals_of_continuous_states_changed = false;
    values_of_continuous_states_changed = false;
    next_event_time_defined = false;
    return true;
}

bool FmuModelExchange::get_real(const uint32_t* /*vr*/, size_t /*nvr*/, double* /*values*/) {
    // Implementation would map value references to circuit values
    return true;
}

bool FmuModelExchange::set_real(const uint32_t* /*vr*/, size_t /*nvr*/, const double* /*values*/) {
    return true;
}

bool FmuModelExchange::get_integer(const uint32_t* /*vr*/, size_t /*nvr*/, int32_t* /*values*/) {
    return true;
}

bool FmuModelExchange::set_integer(const uint32_t* /*vr*/, size_t /*nvr*/, const int32_t* /*values*/) {
    return true;
}

bool FmuModelExchange::get_boolean(const uint32_t* /*vr*/, size_t /*nvr*/, bool* /*values*/) {
    return true;
}

bool FmuModelExchange::set_boolean(const uint32_t* /*vr*/, size_t /*nvr*/, const bool* /*values*/) {
    return true;
}

bool FmuModelExchange::get_continuous_states(double* states, size_t nx) {
    for (size_t i = 0; i < nx && i < continuous_states_.size(); i++) {
        states[i] = continuous_states_[i];
    }
    return true;
}

bool FmuModelExchange::set_continuous_states(const double* states, size_t nx) {
    continuous_states_.resize(nx);
    for (size_t i = 0; i < nx; i++) {
        continuous_states_[i] = states[i];
    }
    update_circuit_state();
    return true;
}

bool FmuModelExchange::get_derivatives(double* derivatives, size_t nx) {
    compute_derivatives();
    for (size_t i = 0; i < nx && i < derivatives_.size(); i++) {
        derivatives[i] = derivatives_[i];
    }
    return true;
}

bool FmuModelExchange::get_nominals_of_continuous_states(double* nominals, size_t nx) {
    for (size_t i = 0; i < nx; i++) {
        nominals[i] = 1.0;  // Default nominal value
    }
    return true;
}

bool FmuModelExchange::get_event_indicators(double* event_indicators, size_t ni) {
    for (size_t i = 0; i < ni && i < event_indicators_.size(); i++) {
        event_indicators[i] = event_indicators_[i];
    }
    return true;
}

bool FmuModelExchange::set_time(double time) {
    current_time_ = time;
    return true;
}

bool FmuModelExchange::completed_integrator_step(bool /*no_set_fmu_state_prior*/,
                                                  bool& enter_event_mode,
                                                  bool& terminate_simulation) {
    enter_event_mode = check_events();
    terminate_simulation = false;
    return true;
}

bool FmuModelExchange::get_fmu_state(void** /*fmu_state*/) {
    return false;  // Not implemented
}

bool FmuModelExchange::set_fmu_state(void* /*fmu_state*/) {
    return false;
}

bool FmuModelExchange::free_fmu_state(void** /*fmu_state*/) {
    return false;
}

bool FmuModelExchange::get_directional_derivative(const uint32_t* /*unknown_refs*/, size_t /*n_unknown*/,
                                                   const uint32_t* /*known_refs*/, size_t /*n_known*/,
                                                   const double* /*seed*/, double* /*sensitivity*/) {
    // Would compute Jacobian entries
    return false;
}

bool FmuModelExchange::terminate() {
    initialized_ = false;
    return true;
}

bool FmuModelExchange::reset() {
    current_time_ = 0.0;
    continuous_states_.clear();
    derivatives_.clear();
    event_indicators_.clear();
    return true;
}

void FmuModelExchange::update_circuit_state() {
    // Map continuous states back to circuit state variables
}

void FmuModelExchange::compute_derivatives() {
    // Compute state derivatives from circuit equations
    derivatives_.resize(continuous_states_.size(), 0.0);
}

bool FmuModelExchange::check_events() {
    // Check if any event indicator crossed zero
    for (size_t i = 0; i < event_indicators_.size(); i++) {
        if (previous_event_indicators_.size() > i) {
            if (event_indicators_[i] * previous_event_indicators_[i] < 0) {
                return true;  // Zero crossing detected
            }
        }
    }
    previous_event_indicators_ = event_indicators_;
    return false;
}

// =============================================================================
// FmuCoSimulation Implementation
// =============================================================================

FmuCoSimulation::FmuCoSimulation() = default;
FmuCoSimulation::~FmuCoSimulation() = default;

bool FmuCoSimulation::initialize(const std::string& /*instance_name*/,
                                  const std::string& /*guid*/,
                                  bool /*logging_on*/) {
    circuit_ = std::make_unique<Circuit>();
    return true;
}

bool FmuCoSimulation::setup_experiment(double start_time,
                                        bool /*stop_time_defined*/,
                                        double /*stop_time*/,
                                        bool tolerance_defined,
                                        double tolerance) {
    current_time_ = start_time;
    if (tolerance_defined) {
        tolerance_ = tolerance;
    }
    return true;
}

bool FmuCoSimulation::enter_initialization_mode() {
    return true;
}

bool FmuCoSimulation::exit_initialization_mode() {
    initialized_ = true;
    return true;
}

bool FmuCoSimulation::get_real(const uint32_t* vr, size_t nvr, double* values) {
    for (size_t i = 0; i < nvr; i++) {
        auto it = output_values_.find(vr[i]);
        values[i] = (it != output_values_.end()) ? it->second : 0.0;
    }
    return true;
}

bool FmuCoSimulation::set_real(const uint32_t* vr, size_t nvr, const double* values) {
    for (size_t i = 0; i < nvr; i++) {
        input_values_[vr[i]] = values[i];
    }
    return true;
}

bool FmuCoSimulation::get_integer(const uint32_t* /*vr*/, size_t /*nvr*/, int32_t* /*values*/) {
    return true;
}

bool FmuCoSimulation::set_integer(const uint32_t* /*vr*/, size_t /*nvr*/, const int32_t* /*values*/) {
    return true;
}

bool FmuCoSimulation::get_boolean(const uint32_t* /*vr*/, size_t /*nvr*/, bool* /*values*/) {
    return true;
}

bool FmuCoSimulation::set_boolean(const uint32_t* /*vr*/, size_t /*nvr*/, const bool* /*values*/) {
    return true;
}

bool FmuCoSimulation::do_step(double current_communication_point,
                               double communication_step_size,
                               bool /*no_set_fmu_state_prior*/,
                               bool& early_return) {
    apply_inputs();

    // Run simulation from current_communication_point to
    // current_communication_point + communication_step_size
    // Using internal step size

    current_time_ = current_communication_point + communication_step_size;

    read_outputs();
    early_return = false;
    return true;
}

bool FmuCoSimulation::cancel_step() {
    return true;
}

bool FmuCoSimulation::get_status(int* status) {
    *status = 0;  // OK
    return true;
}

bool FmuCoSimulation::get_real_status(int /*kind*/, double* value) {
    *value = current_time_;
    return true;
}

bool FmuCoSimulation::get_integer_status(int /*kind*/, int* value) {
    *value = 0;
    return true;
}

bool FmuCoSimulation::get_boolean_status(int /*kind*/, bool* value) {
    *value = false;
    return true;
}

bool FmuCoSimulation::get_string_status(int /*kind*/, std::string& value) {
    value = "";
    return true;
}

bool FmuCoSimulation::get_fmu_state(void** /*fmu_state*/) {
    return false;
}

bool FmuCoSimulation::set_fmu_state(void* /*fmu_state*/) {
    return false;
}

bool FmuCoSimulation::free_fmu_state(void** /*fmu_state*/) {
    return false;
}

bool FmuCoSimulation::terminate() {
    initialized_ = false;
    return true;
}

bool FmuCoSimulation::reset() {
    current_time_ = 0.0;
    input_values_.clear();
    output_values_.clear();
    return true;
}

void FmuCoSimulation::apply_inputs() {
    // Apply input values to circuit sources
}

void FmuCoSimulation::read_outputs() {
    // Read output values from circuit simulation
}

}  // namespace pulsim::fmu
