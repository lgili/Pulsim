#include "pulsim/metadata.hpp"
#include <algorithm>
#include <set>

namespace pulsim {

const ComponentRegistry& ComponentRegistry::instance() {
    static ComponentRegistry registry;
    return registry;
}

ComponentRegistry::ComponentRegistry() {
    register_all_components();
}

const ComponentMetadata* ComponentRegistry::get(ComponentType type) const {
    auto it = metadata_.find(type);
    return it != metadata_.end() ? &it->second : nullptr;
}

std::vector<ComponentType> ComponentRegistry::all_types() const {
    std::vector<ComponentType> types;
    types.reserve(metadata_.size());
    for (const auto& [type, meta] : metadata_) {
        types.push_back(type);
    }
    return types;
}

std::vector<ComponentType> ComponentRegistry::types_in_category(const std::string& category) const {
    std::vector<ComponentType> types;
    for (const auto& [type, meta] : metadata_) {
        if (meta.category == category) {
            types.push_back(type);
        }
    }
    return types;
}

std::vector<std::string> ComponentRegistry::all_categories() const {
    std::set<std::string> categories;
    for (const auto& [type, meta] : metadata_) {
        categories.insert(meta.category);
    }
    return std::vector<std::string>(categories.begin(), categories.end());
}

bool ComponentRegistry::validate_parameter(ComponentType type, const std::string& param_name,
                                           double value, std::string* error_message) const {
    const auto* meta = get(type);
    if (!meta) {
        if (error_message) *error_message = "Unknown component type";
        return false;
    }

    for (const auto& param : meta->parameters) {
        if (param.name == param_name) {
            if (param.min_value && value < *param.min_value) {
                if (error_message) {
                    *error_message = param.display_name + " must be >= " +
                                    std::to_string(*param.min_value) + " " + param.unit;
                }
                return false;
            }
            if (param.max_value && value > *param.max_value) {
                if (error_message) {
                    *error_message = param.display_name + " must be <= " +
                                    std::to_string(*param.max_value) + " " + param.unit;
                }
                return false;
            }
            return true;
        }
    }

    if (error_message) *error_message = "Unknown parameter: " + param_name;
    return false;
}

void ComponentRegistry::register_all_components() {
    // Resistor
    {
        ComponentMetadata meta;
        meta.type = ComponentType::Resistor;
        meta.name = "resistor";
        meta.display_name = "Resistor";
        meta.description = "A two-terminal passive component that opposes current flow";
        meta.category = "Passive";
        meta.symbol_id = "resistor";
        meta.has_loss_model = true;
        meta.has_thermal_model = true;

        meta.pins = {
            {"p", "Positive terminal"},
            {"n", "Negative terminal"}
        };

        meta.parameters = {
            {"resistance", "Resistance", "Resistance value in ohms", ParameterType::Real,
             1000.0, 1e-9, 1e15, "ohm", {}, true}
        };

        metadata_[ComponentType::Resistor] = std::move(meta);
    }

    // Capacitor
    {
        ComponentMetadata meta;
        meta.type = ComponentType::Capacitor;
        meta.name = "capacitor";
        meta.display_name = "Capacitor";
        meta.description = "A two-terminal passive component that stores energy in an electric field";
        meta.category = "Passive";
        meta.symbol_id = "capacitor";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Positive terminal"},
            {"n", "Negative terminal"}
        };

        meta.parameters = {
            {"capacitance", "Capacitance", "Capacitance value in farads", ParameterType::Real,
             1e-6, 1e-15, 1e3, "F", {}, true},
            {"ic", "Initial Voltage", "Initial voltage across capacitor", ParameterType::Real,
             0.0, std::nullopt, std::nullopt, "V", {}, false}
        };

        metadata_[ComponentType::Capacitor] = std::move(meta);
    }

    // Inductor
    {
        ComponentMetadata meta;
        meta.type = ComponentType::Inductor;
        meta.name = "inductor";
        meta.display_name = "Inductor";
        meta.description = "A two-terminal passive component that stores energy in a magnetic field";
        meta.category = "Passive";
        meta.symbol_id = "inductor";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Positive terminal"},
            {"n", "Negative terminal"}
        };

        meta.parameters = {
            {"inductance", "Inductance", "Inductance value in henrys", ParameterType::Real,
             1e-6, 1e-15, 1e3, "H", {}, true},
            {"ic", "Initial Current", "Initial current through inductor", ParameterType::Real,
             0.0, std::nullopt, std::nullopt, "A", {}, false}
        };

        metadata_[ComponentType::Inductor] = std::move(meta);
    }

    // Voltage Source
    {
        ComponentMetadata meta;
        meta.type = ComponentType::VoltageSource;
        meta.name = "voltage_source";
        meta.display_name = "Voltage Source";
        meta.description = "An independent voltage source with configurable waveform";
        meta.category = "Sources";
        meta.symbol_id = "voltage_source";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Positive terminal"},
            {"n", "Negative terminal"}
        };

        meta.parameters = {
            {"dc", "DC Value", "DC voltage value", ParameterType::Real,
             0.0, std::nullopt, std::nullopt, "V", {}, false},
            {"amplitude", "Amplitude", "AC amplitude for sinusoidal source", ParameterType::Real,
             0.0, 0.0, std::nullopt, "V", {}, false},
            {"frequency", "Frequency", "Frequency for AC/Pulse sources", ParameterType::Real,
             0.0, 0.0, std::nullopt, "Hz", {}, false},
            {"phase", "Phase", "Phase offset in degrees", ParameterType::Real,
             0.0, -360.0, 360.0, "deg", {}, false},
            {"waveform", "Waveform", "Waveform type", ParameterType::Enum,
             std::nullopt, std::nullopt, std::nullopt, "",
             {"dc", "sin", "pulse", "pwl", "pwm"}, false}
        };

        metadata_[ComponentType::VoltageSource] = std::move(meta);
    }

    // Current Source
    {
        ComponentMetadata meta;
        meta.type = ComponentType::CurrentSource;
        meta.name = "current_source";
        meta.display_name = "Current Source";
        meta.description = "An independent current source with configurable waveform";
        meta.category = "Sources";
        meta.symbol_id = "current_source";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Positive terminal (current flows out)"},
            {"n", "Negative terminal (current flows in)"}
        };

        meta.parameters = {
            {"dc", "DC Value", "DC current value", ParameterType::Real,
             0.0, std::nullopt, std::nullopt, "A", {}, false},
            {"amplitude", "Amplitude", "AC amplitude for sinusoidal source", ParameterType::Real,
             0.0, 0.0, std::nullopt, "A", {}, false},
            {"frequency", "Frequency", "Frequency for AC/Pulse sources", ParameterType::Real,
             0.0, 0.0, std::nullopt, "Hz", {}, false}
        };

        metadata_[ComponentType::CurrentSource] = std::move(meta);
    }

    // Diode
    {
        ComponentMetadata meta;
        meta.type = ComponentType::Diode;
        meta.name = "diode";
        meta.display_name = "Diode";
        meta.description = "A semiconductor device that allows current flow in one direction";
        meta.category = "Semiconductor";
        meta.symbol_id = "diode";
        meta.has_loss_model = true;
        meta.has_thermal_model = true;

        meta.pins = {
            {"anode", "Anode (positive terminal)"},
            {"cathode", "Cathode (negative terminal)"}
        };

        meta.parameters = {
            {"Is", "Saturation Current", "Reverse saturation current", ParameterType::Real,
             1e-14, 1e-20, 1e-6, "A", {}, false},
            {"n", "Ideality Factor", "Emission coefficient", ParameterType::Real,
             1.0, 0.5, 3.0, "", {}, false},
            {"Vt", "Thermal Voltage", "Thermal voltage at operating temperature", ParameterType::Real,
             0.026, 0.01, 0.1, "V", {}, false},
            {"Vf", "Forward Voltage", "Forward voltage drop at rated current", ParameterType::Real,
             0.7, 0.2, 2.0, "V", {}, false}
        };

        metadata_[ComponentType::Diode] = std::move(meta);
    }

    // Switch
    {
        ComponentMetadata meta;
        meta.type = ComponentType::Switch;
        meta.name = "switch";
        meta.display_name = "Switch";
        meta.description = "A voltage-controlled switch";
        meta.category = "Switching";
        meta.symbol_id = "switch";
        meta.has_loss_model = true;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Positive terminal"},
            {"n", "Negative terminal"},
            {"ctrl_p", "Control positive"},
            {"ctrl_n", "Control negative"}
        };

        meta.parameters = {
            {"Ron", "On Resistance", "Resistance when switch is closed", ParameterType::Real,
             0.001, 1e-9, 1e6, "ohm", {}, true},
            {"Roff", "Off Resistance", "Resistance when switch is open", ParameterType::Real,
             1e9, 1.0, 1e15, "ohm", {}, false},
            {"Vth", "Threshold Voltage", "Control voltage threshold for switching", ParameterType::Real,
             0.5, std::nullopt, std::nullopt, "V", {}, false}
        };

        metadata_[ComponentType::Switch] = std::move(meta);
    }

    // MOSFET
    {
        ComponentMetadata meta;
        meta.type = ComponentType::MOSFET;
        meta.name = "mosfet";
        meta.display_name = "MOSFET";
        meta.description = "Metal-Oxide-Semiconductor Field-Effect Transistor";
        meta.category = "Semiconductor";
        meta.symbol_id = "mosfet_n";
        meta.has_loss_model = true;
        meta.has_thermal_model = true;

        meta.pins = {
            {"drain", "Drain terminal"},
            {"gate", "Gate terminal"},
            {"source", "Source terminal"}
        };

        meta.parameters = {
            {"Vth", "Threshold Voltage", "Gate-source threshold voltage", ParameterType::Real,
             2.0, 0.1, 10.0, "V", {}, true},
            {"Kp", "Transconductance", "Transconductance parameter", ParameterType::Real,
             100e-6, 1e-9, 1.0, "A/V^2", {}, false},
            {"W", "Width", "Channel width", ParameterType::Real,
             100e-6, 1e-9, 1.0, "m", {}, false},
            {"L", "Length", "Channel length", ParameterType::Real,
             10e-6, 1e-9, 1e-3, "m", {}, false},
            {"Rds_on", "On Resistance", "Drain-source on resistance", ParameterType::Real,
             0.01, 1e-6, 100.0, "ohm", {}, false},
            {"type", "Type", "NMOS or PMOS", ParameterType::Enum,
             std::nullopt, std::nullopt, std::nullopt, "",
             {"nmos", "pmos"}, false},
            {"model", "Model Level", "MOSFET model level", ParameterType::Enum,
             std::nullopt, std::nullopt, std::nullopt, "",
             {"level1", "level2", "level3", "bsim3", "ekv"}, false}
        };

        metadata_[ComponentType::MOSFET] = std::move(meta);
    }

    // IGBT
    {
        ComponentMetadata meta;
        meta.type = ComponentType::IGBT;
        meta.name = "igbt";
        meta.display_name = "IGBT";
        meta.description = "Insulated-Gate Bipolar Transistor";
        meta.category = "Semiconductor";
        meta.symbol_id = "igbt";
        meta.has_loss_model = true;
        meta.has_thermal_model = true;

        meta.pins = {
            {"collector", "Collector terminal"},
            {"gate", "Gate terminal"},
            {"emitter", "Emitter terminal"}
        };

        meta.parameters = {
            {"Vce_sat", "Saturation Voltage", "Collector-emitter saturation voltage", ParameterType::Real,
             1.5, 0.5, 5.0, "V", {}, false},
            {"Vge_th", "Gate Threshold", "Gate-emitter threshold voltage", ParameterType::Real,
             5.0, 2.0, 10.0, "V", {}, false},
            {"gm", "Transconductance", "Transconductance", ParameterType::Real,
             10.0, 0.1, 100.0, "A/V", {}, false}
        };

        metadata_[ComponentType::IGBT] = std::move(meta);
    }

    // Transformer
    {
        ComponentMetadata meta;
        meta.type = ComponentType::Transformer;
        meta.name = "transformer";
        meta.display_name = "Transformer";
        meta.description = "Ideal or saturable transformer";
        meta.category = "Magnetic";
        meta.symbol_id = "transformer";
        meta.has_loss_model = true;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p1", "Primary positive"},
            {"n1", "Primary negative"},
            {"p2", "Secondary positive"},
            {"n2", "Secondary negative"}
        };

        meta.parameters = {
            {"n", "Turns Ratio", "Primary to secondary turns ratio (N1/N2)", ParameterType::Real,
             1.0, 0.001, 1000.0, "", {}, true},
            {"Lp", "Primary Inductance", "Primary winding inductance", ParameterType::Real,
             1e-3, 1e-9, 1e3, "H", {}, false},
            {"Rp", "Primary Resistance", "Primary winding resistance", ParameterType::Real,
             0.0, 0.0, 1e6, "ohm", {}, false},
            {"Rs", "Secondary Resistance", "Secondary winding resistance", ParameterType::Real,
             0.0, 0.0, 1e6, "ohm", {}, false}
        };

        metadata_[ComponentType::Transformer] = std::move(meta);
    }

    // VCVS
    {
        ComponentMetadata meta;
        meta.type = ComponentType::VCVS;
        meta.name = "vcvs";
        meta.display_name = "VCVS";
        meta.description = "Voltage-Controlled Voltage Source";
        meta.category = "Controlled Sources";
        meta.symbol_id = "vcvs";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Output positive"},
            {"n", "Output negative"},
            {"ctrl_p", "Control positive"},
            {"ctrl_n", "Control negative"}
        };

        meta.parameters = {
            {"gain", "Voltage Gain", "Output voltage / control voltage", ParameterType::Real,
             1.0, std::nullopt, std::nullopt, "V/V", {}, true}
        };

        metadata_[ComponentType::VCVS] = std::move(meta);
    }

    // VCCS
    {
        ComponentMetadata meta;
        meta.type = ComponentType::VCCS;
        meta.name = "vccs";
        meta.display_name = "VCCS";
        meta.description = "Voltage-Controlled Current Source";
        meta.category = "Controlled Sources";
        meta.symbol_id = "vccs";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Output positive"},
            {"n", "Output negative"},
            {"ctrl_p", "Control positive"},
            {"ctrl_n", "Control negative"}
        };

        meta.parameters = {
            {"gm", "Transconductance", "Output current / control voltage", ParameterType::Real,
             1e-3, std::nullopt, std::nullopt, "A/V", {}, true}
        };

        metadata_[ComponentType::VCCS] = std::move(meta);
    }

    // CCVS
    {
        ComponentMetadata meta;
        meta.type = ComponentType::CCVS;
        meta.name = "ccvs";
        meta.display_name = "CCVS";
        meta.description = "Current-Controlled Voltage Source";
        meta.category = "Controlled Sources";
        meta.symbol_id = "ccvs";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Output positive"},
            {"n", "Output negative"}
        };

        meta.parameters = {
            {"rm", "Transresistance", "Output voltage / control current", ParameterType::Real,
             1.0, std::nullopt, std::nullopt, "V/A", {}, true},
            {"control_source", "Control Source", "Name of the current source to sense", ParameterType::String,
             std::nullopt, std::nullopt, std::nullopt, "", {}, true}
        };

        metadata_[ComponentType::CCVS] = std::move(meta);
    }

    // CCCS
    {
        ComponentMetadata meta;
        meta.type = ComponentType::CCCS;
        meta.name = "cccs";
        meta.display_name = "CCCS";
        meta.description = "Current-Controlled Current Source";
        meta.category = "Controlled Sources";
        meta.symbol_id = "cccs";
        meta.has_loss_model = false;
        meta.has_thermal_model = false;

        meta.pins = {
            {"p", "Output positive"},
            {"n", "Output negative"}
        };

        meta.parameters = {
            {"gain", "Current Gain", "Output current / control current", ParameterType::Real,
             1.0, std::nullopt, std::nullopt, "A/A", {}, true},
            {"control_source", "Control Source", "Name of the current source to sense", ParameterType::String,
             std::nullopt, std::nullopt, std::nullopt, "", {}, true}
        };

        metadata_[ComponentType::CCCS] = std::move(meta);
    }
}

}  // namespace pulsim
