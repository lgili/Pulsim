#include "pulsim/v1/parser/yaml_parser.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace pulsim::v1::parser {

namespace {

constexpr const char* kSchemaId = "pulsim-v1";

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool is_known_key(const std::string& key, const std::unordered_set<std::string>& allowed) {
    return allowed.find(key) != allowed.end();
}

void validate_keys(const YAML::Node& node,
                   const std::unordered_set<std::string>& allowed,
                   const std::string& context,
                   std::vector<std::string>& errors,
                   bool strict) {
    if (!strict || !node || !node.IsMap()) return;
    for (const auto& it : node) {
        const std::string key = it.first.as<std::string>();
        if (!is_known_key(key, allowed)) {
            errors.push_back("Unknown field '" + key + "' in " + context);
        }
    }
}

Real parse_real_string(const std::string& raw) {
    if (raw.empty()) return 0.0;

    std::string value = raw;
    double multiplier = 1.0;

    // Handle 'meg'
    if (value.size() >= 3) {
        std::string last3 = to_lower(value.substr(value.size() - 3));
        if (last3 == "meg") {
            multiplier = 1e6;
            value = value.substr(0, value.size() - 3);
        }
    }

    if (multiplier == 1.0 && !value.empty()) {
        char last = value.back();
        char lower = static_cast<char>(std::tolower(last));
        switch (lower) {
            case 't': multiplier = 1e12; value.pop_back(); break;
            case 'g': multiplier = 1e9; value.pop_back(); break;
            case 'm': multiplier = (last == 'M') ? 1e6 : 1e-3; value.pop_back(); break;
            case 'k': multiplier = 1e3; value.pop_back(); break;
            case 'u': multiplier = 1e-6; value.pop_back(); break;
            case 'n': multiplier = 1e-9; value.pop_back(); break;
            case 'p': multiplier = 1e-12; value.pop_back(); break;
            case 'f': multiplier = 1e-15; value.pop_back(); break;
            default: break;
        }
    }

    return std::stod(value) * multiplier;
}

Real parse_real(const YAML::Node& node,
                const std::string& context,
                std::vector<std::string>& errors) {
    if (!node) {
        errors.push_back("Missing value in " + context);
        return 0.0;
    }

    try {
        if (node.IsScalar()) {
            const std::string raw = node.as<std::string>();
            return parse_real_string(raw);
        }
    } catch (...) {
        errors.push_back("Invalid numeric value in " + context);
    }

    return 0.0;
}

std::vector<std::string> parse_nodes(const YAML::Node& node,
                                     const std::string& context,
                                     std::vector<std::string>& errors) {
    std::vector<std::string> nodes;
    if (!node || !node.IsSequence()) {
        errors.push_back("Missing or invalid nodes in " + context);
        return nodes;
    }

    for (const auto& n : node) {
        nodes.push_back(n.as<std::string>());
    }

    return nodes;
}

YAML::Node merge_nodes(const YAML::Node& base, const YAML::Node& overrides) {
    if (!base) return overrides;
    if (!overrides) return base;

    if (!base.IsMap() || !overrides.IsMap()) {
        return overrides;
    }

    YAML::Node result(YAML::NodeType::Map);
    for (const auto& it : base) {
        result[it.first.as<std::string>()] = it.second;
    }
    for (const auto& it : overrides) {
        const std::string key = it.first.as<std::string>();
        if (result[key] && result[key].IsMap() && it.second.IsMap()) {
            result[key] = merge_nodes(result[key], it.second);
        } else {
            result[key] = it.second;
        }
    }
    return result;
}

}  // namespace

YamlParser::YamlParser(YamlParserOptions options)
    : options_(options) {}

std::pair<Circuit, SimulationOptions> YamlParser::load(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        errors_.push_back("Cannot open file: " + path.string());
        return {Circuit(), SimulationOptions()};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return load_string(buffer.str());
}

std::pair<Circuit, SimulationOptions> YamlParser::load_string(const std::string& content) {
    Circuit circuit;
    SimulationOptions options;
    errors_.clear();
    warnings_.clear();

    parse_yaml(content, circuit, options);
    return {circuit, options};
}

void YamlParser::parse_yaml(const std::string& content, Circuit& circuit, SimulationOptions& options) {
    YAML::Node root;
    try {
        root = YAML::Load(content);
    } catch (const std::exception& e) {
        errors_.push_back(std::string("YAML parse error: ") + e.what());
        return;
    }

    validate_keys(root, {"schema", "version", "simulation", "models", "components"},
                  "root", errors_, options_.strict);

    if (!root["schema"]) {
        errors_.push_back("Missing required field 'schema'");
        return;
    }
    if (!root["version"]) {
        errors_.push_back("Missing required field 'version'");
        return;
    }

    std::string schema = root["schema"].as<std::string>();
    if (schema != kSchemaId) {
        errors_.push_back("Unsupported schema: " + schema);
        return;
    }

    int version = root["version"].as<int>();
    if (version != 1) {
        errors_.push_back("Unsupported schema version: " + std::to_string(version));
        return;
    }

    // Simulation options
    if (root["simulation"]) {
        YAML::Node sim = root["simulation"];
        validate_keys(sim, {"tstart", "tstop", "dt", "dt_min", "dt_max", "adaptive_timestep",
                            "enable_events", "enable_losses", "newton", "timestep", "lte", "bdf"},
                      "simulation", errors_, options_.strict);

        if (sim["tstart"]) options.tstart = parse_real(sim["tstart"], "simulation.tstart", errors_);
        if (sim["tstop"]) options.tstop = parse_real(sim["tstop"], "simulation.tstop", errors_);
        if (sim["dt"]) options.dt = parse_real(sim["dt"], "simulation.dt", errors_);
        if (sim["dt_min"]) options.dt_min = parse_real(sim["dt_min"], "simulation.dt_min", errors_);
        if (sim["dt_max"]) options.dt_max = parse_real(sim["dt_max"], "simulation.dt_max", errors_);
        if (sim["adaptive_timestep"]) options.adaptive_timestep = sim["adaptive_timestep"].as<bool>();
        if (sim["enable_events"]) options.enable_events = sim["enable_events"].as<bool>();
        if (sim["enable_losses"]) options.enable_losses = sim["enable_losses"].as<bool>();

        if (sim["newton"]) {
            YAML::Node n = sim["newton"];
            validate_keys(n, {"max_iterations", "initial_damping", "min_damping", "auto_damping"},
                          "simulation.newton", errors_, options_.strict);
            if (n["max_iterations"]) options.newton_options.max_iterations = n["max_iterations"].as<int>();
            if (n["initial_damping"]) options.newton_options.initial_damping = parse_real(n["initial_damping"], "newton.initial_damping", errors_);
            if (n["min_damping"]) options.newton_options.min_damping = parse_real(n["min_damping"], "newton.min_damping", errors_);
            if (n["auto_damping"]) options.newton_options.auto_damping = n["auto_damping"].as<bool>();
        }

        if (sim["timestep"]) {
            YAML::Node t = sim["timestep"];
            validate_keys(t, {"preset", "dt_min", "dt_max", "error_tolerance", "target_newton_iterations"},
                          "simulation.timestep", errors_, options_.strict);
            auto apply_base = [&](const TimestepConfig& base) {
                options.timestep_config.dt_min = base.dt_min;
                options.timestep_config.dt_max = base.dt_max;
                options.timestep_config.dt_initial = base.dt_initial;
                options.timestep_config.safety_factor = base.safety_factor;
                options.timestep_config.error_tolerance = base.error_tolerance;
                options.timestep_config.growth_factor = base.growth_factor;
                options.timestep_config.shrink_factor = base.shrink_factor;
                options.timestep_config.max_rejections = base.max_rejections;
                options.timestep_config.k_p = base.k_p;
                options.timestep_config.k_i = base.k_i;
            };
            if (t["preset"]) {
                std::string preset = to_lower(t["preset"].as<std::string>());
                if (preset == "conservative") {
                    apply_base(TimestepConfig::conservative());
                } else if (preset == "aggressive") {
                    apply_base(TimestepConfig::aggressive());
                } else if (preset == "power_electronics") {
                    options.timestep_config = AdvancedTimestepConfig::for_power_electronics();
                }
            }
            if (t["dt_min"]) options.timestep_config.dt_min = parse_real(t["dt_min"], "timestep.dt_min", errors_);
            if (t["dt_max"]) options.timestep_config.dt_max = parse_real(t["dt_max"], "timestep.dt_max", errors_);
            if (t["error_tolerance"]) options.timestep_config.error_tolerance = parse_real(t["error_tolerance"], "timestep.error_tolerance", errors_);
            if (t["target_newton_iterations"]) options.timestep_config.target_newton_iterations = t["target_newton_iterations"].as<int>();
        }

        if (sim["lte"]) {
            YAML::Node l = sim["lte"];
            validate_keys(l, {"method", "voltage_tolerance", "current_tolerance"},
                          "simulation.lte", errors_, options_.strict);
            if (l["method"]) {
                std::string method = to_lower(l["method"].as<std::string>());
                options.lte_config.method = (method == "step_doubling") ?
                    TimestepMethod::StepDoubling : TimestepMethod::Richardson;
            }
            if (l["voltage_tolerance"]) options.lte_config.voltage_tolerance = parse_real(l["voltage_tolerance"], "lte.voltage_tolerance", errors_);
            if (l["current_tolerance"]) options.lte_config.current_tolerance = parse_real(l["current_tolerance"], "lte.current_tolerance", errors_);
        }

        if (sim["bdf"]) {
            YAML::Node b = sim["bdf"];
            validate_keys(b, {"enable", "min_order", "max_order", "initial_order"},
                          "simulation.bdf", errors_, options_.strict);
            if (b["enable"]) options.enable_bdf_order_control = b["enable"].as<bool>();
            if (b["min_order"]) options.bdf_config.min_order = b["min_order"].as<int>();
            if (b["max_order"]) options.bdf_config.max_order = b["max_order"].as<int>();
            if (b["initial_order"]) options.bdf_config.initial_order = b["initial_order"].as<int>();
        }
    }

    // Models map (optional)
    std::unordered_map<std::string, YAML::Node> model_nodes;
    std::unordered_map<std::string, YAML::Node> resolved_models;

    if (root["models"]) {
        YAML::Node models = root["models"];
        if (!models.IsMap()) {
            errors_.push_back("models must be a map");
        } else {
            for (const auto& it : models) {
                model_nodes[it.first.as<std::string>()] = it.second;
            }
        }
    }

    std::function<YAML::Node(const std::string&, std::unordered_set<std::string>&)> resolve_model;
    resolve_model = [&](const std::string& name, std::unordered_set<std::string>& stack) -> YAML::Node {
        if (resolved_models.count(name)) return resolved_models[name];
        if (!model_nodes.count(name)) return YAML::Node();
        if (stack.count(name)) {
            errors_.push_back("Cyclic model inheritance for model: " + name);
            return YAML::Node();
        }

        stack.insert(name);
        YAML::Node node = model_nodes[name];
        if (node["extends"]) {
            std::string base = node["extends"].as<std::string>();
            YAML::Node base_node = resolve_model(base, stack);
            node = merge_nodes(base_node, node);
        }
        stack.erase(name);
        resolved_models[name] = node;
        return node;
    };

    // Components
    if (!root["components"] || !root["components"].IsSequence()) {
        errors_.push_back("Missing or invalid components list");
        return;
    }

    YAML::Node components = root["components"];

    for (const auto& comp_node_raw : components) {
        YAML::Node comp_node = comp_node_raw;

        if (comp_node["use"]) {
            std::string model_name = comp_node["use"].as<std::string>();
            std::unordered_set<std::string> stack;
            YAML::Node model = resolve_model(model_name, stack);
            if (!model) {
                errors_.push_back("Unknown model: " + model_name);
                continue;
            }
            comp_node = merge_nodes(model, comp_node);
        }

        validate_keys(comp_node,
            {"type", "name", "nodes", "value", "params", "waveform", "use", "loss",
             "resistance", "capacitance", "inductance", "ic",
             "g_on", "g_off", "ron", "roff", "v_threshold", "initial_state",
             "vth", "kp", "lambda", "is_nmos",
             "turns_ratio", "ratio"},
            "component", errors_, options_.strict);

        if (!comp_node["type"] || !comp_node["name"] || !comp_node["nodes"]) {
            errors_.push_back("Component missing type, name, or nodes");
            continue;
        }

        std::string type = to_lower(comp_node["type"].as<std::string>());
        std::string name = comp_node["name"].as<std::string>();
        std::vector<std::string> nodes = parse_nodes(comp_node["nodes"], name, errors_);
        if (nodes.empty()) continue;

        YAML::Node params = comp_node["params"];

        auto get_param = [&](const std::string& key) -> YAML::Node {
            if (comp_node[key]) return comp_node[key];
            if (params && params[key]) return params[key];
            return YAML::Node();
        };

        auto idx = [&](const std::string& node_name) {
            return circuit.add_node(node_name);
        };

        // Loss model (switching energy)
        if (comp_node["loss"]) {
            YAML::Node loss = comp_node["loss"];
            SwitchingEnergy energy;
            if (loss["eon"]) energy.eon = parse_real(loss["eon"], name + ".loss.eon", errors_);
            if (loss["eoff"]) energy.eoff = parse_real(loss["eoff"], name + ".loss.eoff", errors_);
            if (loss["err"]) energy.err = parse_real(loss["err"], name + ".loss.err", errors_);
            options.switching_energy[name] = energy;
        }

        if (type == "resistor" || type == "r") {
            YAML::Node value_node = get_param("value");
            if (!value_node) value_node = get_param("resistance");
            Real value = parse_real(value_node, name + ".value", errors_);
            circuit.add_resistor(name, idx(nodes[0]), idx(nodes[1]), value);
        }
        else if (type == "capacitor" || type == "c") {
            YAML::Node value_node = get_param("value");
            if (!value_node) value_node = get_param("capacitance");
            Real value = parse_real(value_node, name + ".value", errors_);
            Real ic = 0.0;
            if (get_param("ic")) ic = parse_real(get_param("ic"), name + ".ic", errors_);
            circuit.add_capacitor(name, idx(nodes[0]), idx(nodes[1]), value, ic);
        }
        else if (type == "inductor" || type == "l") {
            YAML::Node value_node = get_param("value");
            if (!value_node) value_node = get_param("inductance");
            Real value = parse_real(value_node, name + ".value", errors_);
            Real ic = 0.0;
            if (get_param("ic")) ic = parse_real(get_param("ic"), name + ".ic", errors_);
            circuit.add_inductor(name, idx(nodes[0]), idx(nodes[1]), value, ic);
        }
        else if (type == "voltage_source" || type == "v") {
            YAML::Node waveform = comp_node["waveform"];
            if (waveform && waveform["type"]) {
                std::string wtype = to_lower(waveform["type"].as<std::string>());
                if (wtype == "dc") {
                    Real value = parse_real(waveform["value"], name + ".waveform.value", errors_);
                    circuit.add_voltage_source(name, idx(nodes[0]), idx(nodes[1]), value);
                } else if (wtype == "pwm") {
                    PWMParams p;
                    if (waveform["v_high"]) p.v_high = parse_real(waveform["v_high"], name + ".waveform.v_high", errors_);
                    if (waveform["v_low"]) p.v_low = parse_real(waveform["v_low"], name + ".waveform.v_low", errors_);
                    if (waveform["frequency"]) p.frequency = parse_real(waveform["frequency"], name + ".waveform.frequency", errors_);
                    if (waveform["duty"]) p.duty = parse_real(waveform["duty"], name + ".waveform.duty", errors_);
                    if (waveform["dead_time"]) p.dead_time = parse_real(waveform["dead_time"], name + ".waveform.dead_time", errors_);
                    if (waveform["phase"]) p.phase = parse_real(waveform["phase"], name + ".waveform.phase", errors_);
                    circuit.add_pwm_voltage_source(name, idx(nodes[0]), idx(nodes[1]), p);
                } else if (wtype == "sine") {
                    SineParams p;
                    if (waveform["amplitude"]) p.amplitude = parse_real(waveform["amplitude"], name + ".waveform.amplitude", errors_);
                    if (waveform["frequency"]) p.frequency = parse_real(waveform["frequency"], name + ".waveform.frequency", errors_);
                    if (waveform["offset"]) p.offset = parse_real(waveform["offset"], name + ".waveform.offset", errors_);
                    if (waveform["phase"]) p.phase = parse_real(waveform["phase"], name + ".waveform.phase", errors_);
                    circuit.add_sine_voltage_source(name, idx(nodes[0]), idx(nodes[1]), p);
                } else if (wtype == "pulse") {
                    PulseParams p;
                    if (waveform["v_initial"]) p.v_initial = parse_real(waveform["v_initial"], name + ".waveform.v_initial", errors_);
                    if (waveform["v_pulse"]) p.v_pulse = parse_real(waveform["v_pulse"], name + ".waveform.v_pulse", errors_);
                    if (waveform["t_delay"]) p.t_delay = parse_real(waveform["t_delay"], name + ".waveform.t_delay", errors_);
                    if (waveform["t_rise"]) p.t_rise = parse_real(waveform["t_rise"], name + ".waveform.t_rise", errors_);
                    if (waveform["t_fall"]) p.t_fall = parse_real(waveform["t_fall"], name + ".waveform.t_fall", errors_);
                    if (waveform["t_width"]) p.t_width = parse_real(waveform["t_width"], name + ".waveform.t_width", errors_);
                    if (waveform["period"]) p.period = parse_real(waveform["period"], name + ".waveform.period", errors_);
                    circuit.add_pulse_voltage_source(name, idx(nodes[0]), idx(nodes[1]), p);
                } else {
                    errors_.push_back("Unsupported waveform type for voltage_source: " + wtype);
                }
            } else {
                Real value = parse_real(get_param("value"), name + ".value", errors_);
                circuit.add_voltage_source(name, idx(nodes[0]), idx(nodes[1]), value);
            }
        }
        else if (type == "current_source" || type == "i") {
            Real value = parse_real(get_param("value"), name + ".value", errors_);
            circuit.add_current_source(name, idx(nodes[0]), idx(nodes[1]), value);
        }
        else if (type == "diode" || type == "d") {
            YAML::Node g_on_node = get_param("g_on");
            if (!g_on_node && get_param("ron")) {
                Real ron = parse_real(get_param("ron"), name + ".ron", errors_);
                g_on_node = YAML::Node(ron > 0 ? 1.0 / ron : 0.0);
            }
            YAML::Node g_off_node = get_param("g_off");
            if (!g_off_node && get_param("roff")) {
                Real roff = parse_real(get_param("roff"), name + ".roff", errors_);
                g_off_node = YAML::Node(roff > 0 ? 1.0 / roff : 0.0);
            }
            Real g_on = g_on_node ? parse_real(g_on_node, name + ".g_on", errors_) : 1e3;
            Real g_off = g_off_node ? parse_real(g_off_node, name + ".g_off", errors_) : 1e-9;
            circuit.add_diode(name, idx(nodes[0]), idx(nodes[1]), g_on, g_off);
        }
        else if (type == "switch" || type == "s") {
            YAML::Node g_on_node = get_param("g_on");
            if (!g_on_node && get_param("ron")) {
                Real ron = parse_real(get_param("ron"), name + ".ron", errors_);
                g_on_node = YAML::Node(ron > 0 ? 1.0 / ron : 0.0);
            }
            YAML::Node g_off_node = get_param("g_off");
            if (!g_off_node && get_param("roff")) {
                Real roff = parse_real(get_param("roff"), name + ".roff", errors_);
                g_off_node = YAML::Node(roff > 0 ? 1.0 / roff : 0.0);
            }
            Real g_on = g_on_node ? parse_real(g_on_node, name + ".g_on", errors_) : 1e6;
            Real g_off = g_off_node ? parse_real(g_off_node, name + ".g_off", errors_) : 1e-12;
            bool closed = get_param("initial_state") ? get_param("initial_state").as<bool>() : false;
            circuit.add_switch(name, idx(nodes[0]), idx(nodes[1]), closed, g_on, g_off);
        }
        else if (type == "vcswitch") {
            if (nodes.size() < 3) {
                errors_.push_back("Voltage-controlled switch requires 3 nodes: " + name);
                continue;
            }
            Real v_threshold = get_param("v_threshold") ? parse_real(get_param("v_threshold"), name + ".v_threshold", errors_) : 2.5;
            Real g_on = get_param("g_on") ? parse_real(get_param("g_on"), name + ".g_on", errors_) : 1e3;
            Real g_off = get_param("g_off") ? parse_real(get_param("g_off"), name + ".g_off", errors_) : 1e-9;
            circuit.add_vcswitch(name, idx(nodes[0]), idx(nodes[1]), idx(nodes[2]), v_threshold, g_on, g_off);
        }
        else if (type == "mosfet" || type == "nmos" || type == "pmos" || type == "m") {
            MOSFET::Params p;
            if (get_param("vth")) p.vth = parse_real(get_param("vth"), name + ".vth", errors_);
            if (get_param("kp")) p.kp = parse_real(get_param("kp"), name + ".kp", errors_);
            if (get_param("lambda")) p.lambda = parse_real(get_param("lambda"), name + ".lambda", errors_);
            if (get_param("g_off")) p.g_off = parse_real(get_param("g_off"), name + ".g_off", errors_);
            if (get_param("is_nmos")) p.is_nmos = get_param("is_nmos").as<bool>();
            if (type == "pmos") p.is_nmos = false;
            circuit.add_mosfet(name, idx(nodes[0]), idx(nodes[1]), idx(nodes[2]), p);
        }
        else if (type == "igbt" || type == "q") {
            IGBT::Params p;
            if (get_param("vth")) p.vth = parse_real(get_param("vth"), name + ".vth", errors_);
            if (get_param("g_on")) p.g_on = parse_real(get_param("g_on"), name + ".g_on", errors_);
            if (get_param("g_off")) p.g_off = parse_real(get_param("g_off"), name + ".g_off", errors_);
            if (get_param("v_ce_sat")) p.v_ce_sat = parse_real(get_param("v_ce_sat"), name + ".v_ce_sat", errors_);
            circuit.add_igbt(name, idx(nodes[0]), idx(nodes[1]), idx(nodes[2]), p);
        }
        else if (type == "transformer" || type == "t") {
            if (nodes.size() < 4) {
                errors_.push_back("Transformer requires 4 nodes: " + name);
                continue;
            }
            YAML::Node ratio_node = get_param("turns_ratio");
            if (!ratio_node) ratio_node = get_param("ratio");
            Real turns_ratio = parse_real(ratio_node, name + ".turns_ratio", errors_);
            circuit.add_transformer(name, idx(nodes[0]), idx(nodes[1]), idx(nodes[2]), idx(nodes[3]), turns_ratio);
        }
        else {
            errors_.push_back("Unsupported component type: " + type);
        }
    }
}

}  // namespace pulsim::v1::parser
