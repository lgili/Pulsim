#include "pulsim/v1/parser/yaml_parser.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace pulsim::v1::parser {

namespace {

constexpr const char* kSchemaId = "pulsim-v1";
constexpr const char* kDiagUnsupportedComponent = "PULSIM_YAML_E_COMPONENT_UNSUPPORTED";
constexpr const char* kDiagInvalidPinCount = "PULSIM_YAML_E_PIN_COUNT";
constexpr const char* kDiagInvalidParameter = "PULSIM_YAML_E_PARAM_INVALID";
constexpr const char* kDiagVirtualComponent = "PULSIM_YAML_W_COMPONENT_VIRTUAL";
constexpr const char* kDiagSurrogateComponent = "PULSIM_YAML_W_COMPONENT_SURROGATE";
constexpr const char* kDiagLegacyTransientBackend = "PULSIM_YAML_E_LEGACY_TRANSIENT_BACKEND";
constexpr const char* kDiagInvalidStepMode = "PULSIM_YAML_E_STEP_MODE_INVALID";
constexpr const char* kDiagUnknownField = "PULSIM_YAML_E_UNKNOWN_FIELD";
constexpr const char* kDiagTypeMismatch = "PULSIM_YAML_E_TYPE_MISMATCH";
constexpr const char* kDiagDeprecatedField = "PULSIM_YAML_W_DEPRECATED_FIELD";

Real parse_real_string(const std::string& raw);

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string normalize_key(std::string s) {
    s = to_lower(s);
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

bool is_known_key(const std::string& key, const std::unordered_set<std::string>& allowed) {
    return allowed.find(key) != allowed.end();
}

std::string with_diag_code(const std::string& code, const std::string& message) {
    return "[" + code + "] " + message;
}

void push_error(std::vector<std::string>& errors, const std::string& code, const std::string& message) {
    errors.push_back(with_diag_code(code, message));
}

void push_warning(std::vector<std::string>& warnings, const std::string& code, const std::string& message) {
    warnings.push_back(with_diag_code(code, message));
}

std::string yaml_node_class(const YAML::Node& node) {
    if (!node || node.IsNull()) {
        return "null";
    }
    if (node.IsScalar()) {
        return "scalar";
    }
    if (node.IsSequence()) {
        return "sequence";
    }
    if (node.IsMap()) {
        return "map";
    }
    return "unknown";
}

void push_type_mismatch_error(std::vector<std::string>& errors,
                              const std::string& path,
                              const std::string& expected,
                              const YAML::Node& received) {
    push_error(
        errors,
        kDiagTypeMismatch,
        "Type mismatch at '" + path + "' (expected " + expected +
            ", got " + yaml_node_class(received) + ")");
}

std::optional<bool> parse_bool_scalar(const YAML::Node& node,
                                      const std::string& path,
                                      std::vector<std::string>& errors) {
    if (!node) {
        return std::nullopt;
    }
    if (!node.IsScalar()) {
        push_type_mismatch_error(errors, path, "boolean", node);
        return std::nullopt;
    }
    try {
        return node.as<bool>();
    } catch (...) {
        push_type_mismatch_error(errors, path, "boolean", node);
        return std::nullopt;
    }
}

std::optional<int> parse_int_scalar(const YAML::Node& node,
                                    const std::string& path,
                                    std::vector<std::string>& errors) {
    if (!node) {
        return std::nullopt;
    }
    if (!node.IsScalar()) {
        push_type_mismatch_error(errors, path, "integer", node);
        return std::nullopt;
    }
    try {
        return node.as<int>();
    } catch (...) {
        push_type_mismatch_error(errors, path, "integer", node);
        return std::nullopt;
    }
}

std::optional<std::string> parse_string_scalar(const YAML::Node& node,
                                               const std::string& path,
                                               std::vector<std::string>& errors) {
    if (!node) {
        return std::nullopt;
    }
    if (!node.IsScalar()) {
        push_type_mismatch_error(errors, path, "string", node);
        return std::nullopt;
    }
    try {
        return node.as<std::string>();
    } catch (...) {
        push_type_mismatch_error(errors, path, "string", node);
        return std::nullopt;
    }
}

void push_deprecated_field_migration_warning(std::vector<std::string>& warnings,
                                             const std::string& key_path,
                                             const std::string& replacement) {
    push_warning(
        warnings,
        kDiagDeprecatedField,
        "Deprecated field '" + key_path + "' remains accepted in schema v1 migration window; use '" +
            replacement + "' instead.");
}

void apply_mode_derived_defaults(SimulationOptions& options,
                                 TransientStepMode mode,
                                 bool dt_min_explicit,
                                 bool dt_max_explicit) {
    options.step_mode = mode;
    options.step_mode_explicit = true;
    options.adaptive_timestep = (mode == TransientStepMode::Variable);

    // Deterministic robust defaults shared by canonical modes.
    options.integrator = Integrator::TRBDF2;
    options.stiffness_config.enable = true;
    options.stiffness_config.switch_integrator = true;
    options.stiffness_config.stiff_integrator = Integrator::BDF1;
    options.max_step_retries = std::max(options.max_step_retries, 6);

    if (!dt_min_explicit) {
        options.dt_min = std::max<Real>(1e-12, options.dt * 1e-3);
    }
    if (!dt_max_explicit) {
        options.dt_max = std::max<Real>(options.dt * 20.0, options.dt);
    }

    options.timestep_config = AdvancedTimestepConfig::for_power_electronics();
    options.timestep_config.dt_initial = std::max<Real>(options.dt, 1e-18);
    options.timestep_config.dt_min = std::max<Real>(options.dt_min, 1e-12);
    options.timestep_config.dt_max = std::max<Real>(options.dt_max, options.timestep_config.dt_min);

    if (mode == TransientStepMode::Fixed) {
        options.enable_bdf_order_control = false;
    } else {
        options.enable_bdf_order_control = true;
        options.bdf_config.min_order = 1;
        options.bdf_config.max_order = 2;
        options.bdf_config.initial_order = 1;
        options.lte_config = RichardsonLTEConfig::defaults();
    }
}

void enforce_mode_semantics(SimulationOptions& options) {
    if (!options.step_mode_explicit) {
        return;
    }
    options.adaptive_timestep = (options.step_mode == TransientStepMode::Variable);
}

void push_legacy_backend_migration_error(std::vector<std::string>& errors, const std::string& key_path) {
    push_error(
        errors,
        kDiagLegacyTransientBackend,
        "Removed transient backend key '" + key_path +
            "' is unsupported in strict mode. Migrate to 'simulation.step_mode: fixed|variable' "
            "with the native core.");
}

const std::unordered_map<std::string, std::string>& component_alias_map() {
    static const std::unordered_map<std::string, std::string> aliases = [] {
        std::unordered_map<std::string, std::string> map;

        auto add_aliases = [&](const std::string& canonical, std::initializer_list<const char*> names) {
            map.emplace(normalize_key(canonical), canonical);
            for (const char* name : names) {
                map.emplace(normalize_key(name), canonical);
            }
        };

        add_aliases("resistor", {"r"});
        add_aliases("capacitor", {"c"});
        add_aliases("inductor", {"l"});
        add_aliases("voltage_source", {"v", "voltagesource", "source_v", "vsource"});
        add_aliases("current_source", {"i", "currentsource", "isource"});
        add_aliases("diode", {"d"});
        add_aliases("switch", {"s"});
        add_aliases("vcswitch", {"voltagecontrolledswitch"});
        add_aliases("mosfet", {"m", "nmos", "pmos"});
        add_aliases("igbt", {"q"});
        add_aliases("transformer", {"t"});
        add_aliases("snubber_rc", {"snubber", "snubberrc"});

        add_aliases("bjt_npn", {"bjtnpn", "bjt-npn"});
        add_aliases("bjt_pnp", {"bjtpnp", "bjt-pnp"});
        add_aliases("thyristor", {"scr"});
        add_aliases("triac", {"triac"});
        add_aliases("fuse", {"fuse"});
        add_aliases("circuit_breaker", {"breaker", "circuitbreaker", "circuit-breaker"});
        add_aliases("relay", {"relay"});
        add_aliases("op_amp", {"opamp", "op-amp"});
        add_aliases("comparator", {"comparator"});
        add_aliases("pi_controller", {"picontroller", "pi-controller"});
        add_aliases("pid_controller", {"pidcontroller", "pid-controller"});
        add_aliases("math_block", {"mathblock", "math"});
        add_aliases("pwm_generator", {"pwmgenerator", "pwm"});
        add_aliases("integrator", {"integrator"});
        add_aliases("differentiator", {"differentiator"});
        add_aliases("limiter", {"limiter"});
        add_aliases("rate_limiter", {"ratelimiter", "rate-limiter"});
        add_aliases("hysteresis", {"hysteresis"});
        add_aliases("lookup_table", {"lookuptable", "lookup-table", "lut"});
        add_aliases("transfer_function", {"transferfunction", "transfer-function"});
        add_aliases("delay_block", {"delayblock", "delay"});
        add_aliases("sample_hold", {"samplehold", "sample-and-hold"});
        add_aliases("state_machine", {"statemachine", "state-machine"});
        add_aliases("saturable_inductor", {"saturableinductor", "sat_inductor"});
        add_aliases("coupled_inductor", {"coupledinductor"});
        add_aliases("voltage_probe", {"voltageprobe", "vprobe"});
        add_aliases("current_probe", {"currentprobe", "iprobe"});
        add_aliases("power_probe", {"powerprobe", "pprobe"});
        add_aliases("electrical_scope", {"electricalscope", "scope"});
        add_aliases("thermal_scope", {"thermalscope"});
        add_aliases("signal_mux", {"signalmux", "mux"});
        add_aliases("signal_demux", {"signaldemux", "demux"});

        return map;
    }();
    return aliases;
}

const std::unordered_map<std::string, std::pair<std::size_t, std::size_t>>& component_node_arity() {
    static const std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> arity = {
        {"resistor", {2, 2}},
        {"capacitor", {2, 2}},
        {"inductor", {2, 2}},
        {"voltage_source", {2, 2}},
        {"current_source", {2, 2}},
        {"diode", {2, 2}},
        {"switch", {2, 2}},
        {"vcswitch", {3, 3}},
        {"mosfet", {3, 3}},
        {"igbt", {3, 3}},
        {"transformer", {4, 4}},
        {"snubber_rc", {2, 2}},
        {"bjt_npn", {3, 3}},
        {"bjt_pnp", {3, 3}},
        {"thyristor", {3, 3}},
        {"triac", {3, 3}},
        {"fuse", {2, 2}},
        {"circuit_breaker", {2, 2}},
        {"relay", {5, 5}},
        {"op_amp", {3, 3}},
        {"comparator", {3, 3}},
        {"pi_controller", {3, 3}},
        {"pid_controller", {3, 3}},
        {"math_block", {2, std::numeric_limits<std::size_t>::max()}},
        {"pwm_generator", {1, 3}},
        {"integrator", {2, 2}},
        {"differentiator", {2, 2}},
        {"limiter", {2, 2}},
        {"rate_limiter", {2, 2}},
        {"hysteresis", {2, 2}},
        {"lookup_table", {2, 2}},
        {"transfer_function", {2, 2}},
        {"delay_block", {2, 2}},
        {"sample_hold", {2, 2}},
        {"state_machine", {1, std::numeric_limits<std::size_t>::max()}},
        {"saturable_inductor", {2, 2}},
        {"coupled_inductor", {4, 4}},
        {"voltage_probe", {2, 2}},
        {"current_probe", {2, 2}},
        {"power_probe", {2, 2}},
        {"electrical_scope", {1, std::numeric_limits<std::size_t>::max()}},
        {"thermal_scope", {1, std::numeric_limits<std::size_t>::max()}},
        {"signal_mux", {2, std::numeric_limits<std::size_t>::max()}},
        {"signal_demux", {2, std::numeric_limits<std::size_t>::max()}}
    };
    return arity;
}

const std::unordered_set<std::string>& virtual_component_types() {
    static const std::unordered_set<std::string> types = {
        "relay",
        "op_amp",
        "comparator",
        "pi_controller",
        "pid_controller",
        "math_block",
        "pwm_generator",
        "integrator",
        "differentiator",
        "limiter",
        "rate_limiter",
        "hysteresis",
        "lookup_table",
        "transfer_function",
        "delay_block",
        "sample_hold",
        "state_machine",
        "voltage_probe",
        "current_probe",
        "power_probe",
        "electrical_scope",
        "thermal_scope",
        "signal_mux",
        "signal_demux"
    };
    return types;
}

std::string canonical_component_type(const std::string& raw_type) {
    const auto key = normalize_key(raw_type);
    const auto& aliases = component_alias_map();
    const auto it = aliases.find(key);
    if (it == aliases.end()) {
        return {};
    }
    return it->second;
}

bool validate_node_count(const std::string& type,
                         const std::vector<std::string>& nodes,
                         std::vector<std::string>& errors,
                         const std::string& name) {
    const auto& arity = component_node_arity();
    const auto it = arity.find(type);
    if (it == arity.end()) {
        return true;
    }

    const auto [min_nodes, max_nodes] = it->second;
    if (nodes.size() < min_nodes || nodes.size() > max_nodes) {
        std::ostringstream oss;
        oss << "Invalid pin count for '" << name << "' (" << type << "): got "
            << nodes.size() << ", expected ";
        if (min_nodes == max_nodes) {
            oss << min_nodes;
        } else if (max_nodes == std::numeric_limits<std::size_t>::max()) {
            oss << min_nodes << "+";
        } else {
            oss << min_nodes << "-" << max_nodes;
        }
        push_error(errors, kDiagInvalidPinCount, oss.str());
        return false;
    }

    return true;
}

void collect_virtual_component_fields(
    const YAML::Node& component,
    std::unordered_map<std::string, Real>& numeric_params,
    std::unordered_map<std::string, std::string>& metadata) {
    auto ingest_map = [&](const YAML::Node& node, bool include_reserved) {
        if (!node || !node.IsMap()) return;
        for (const auto& it : node) {
            const std::string key = it.first.as<std::string>();
            if (!include_reserved &&
                (key == "type" || key == "name" || key == "nodes" || key == "params")) {
                continue;
            }

            const YAML::Node value = it.second;
            if (!value || value.IsNull()) continue;

            if (value.IsScalar()) {
                try {
                    const std::string as_text = value.as<std::string>();
                    numeric_params[key] = parse_real_string(as_text);
                } catch (...) {
                    try {
                        metadata[key] = value.as<std::string>();
                    } catch (...) {
                        metadata[key] = YAML::Dump(value);
                    }
                }
                continue;
            }

            metadata[key] = YAML::Dump(value);
        }
    };

    ingest_map(component, false);
    ingest_map(component["params"], true);
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
            push_error(errors,
                       kDiagUnknownField,
                       "Unknown field at '" + context + "." + key + "'");
        }
    }
}

Real parse_real_string(const std::string& raw) {
    if (raw.empty()) return 0.0;

    char* end = nullptr;
    const double base = std::strtod(raw.c_str(), &end);
    if (end == raw.c_str()) {
        throw std::invalid_argument("invalid numeric value");
    }

    std::string suffix = raw.substr(static_cast<std::size_t>(end - raw.c_str()));
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!suffix.empty() && is_space(static_cast<unsigned char>(suffix.front()))) {
        suffix.erase(suffix.begin());
    }
    while (!suffix.empty() && is_space(static_cast<unsigned char>(suffix.back()))) {
        suffix.pop_back();
    }
    if (suffix.empty()) return base;

    const std::string lower = to_lower(suffix);
    auto starts_with = [&](const std::string& prefix) {
        return lower.rfind(prefix, 0) == 0;
    };

    double multiplier = 1.0;
    if (starts_with("tera") || starts_with("t")) {
        multiplier = 1e12;
    } else if (starts_with("giga") || starts_with("g")) {
        multiplier = 1e9;
    } else if (starts_with("mega") || starts_with("meg") ||
               (!suffix.empty() && suffix.front() == 'M')) {
        multiplier = 1e6;
    } else if (starts_with("kilo") || starts_with("k")) {
        multiplier = 1e3;
    } else if (starts_with("milli")) {
        multiplier = 1e-3;
    } else if (starts_with("micro") || starts_with("u")) {
        multiplier = 1e-6;
    } else if (starts_with("nano") || starts_with("n")) {
        multiplier = 1e-9;
    } else if (starts_with("pico") || starts_with("p")) {
        multiplier = 1e-12;
    } else if (starts_with("femto") || starts_with("f")) {
        multiplier = 1e-15;
    } else if (starts_with("m")) {
        multiplier = 1e-3;
    }

    return base * multiplier;
}

Real parse_real(const YAML::Node& node,
                const std::string& context,
                std::vector<std::string>& errors) {
    if (!node || node.IsNull()) {
        return 0.0;
    }

    if (!node.IsScalar()) {
        push_type_mismatch_error(errors, context, "number", node);
        return 0.0;
    }

    try {
        const std::string raw = node.as<std::string>();
        return parse_real_string(raw);
    } catch (...) {
        push_type_mismatch_error(errors, context, "number", node);
    }

    return 0.0;
}

std::vector<std::string> parse_nodes(const YAML::Node& node,
                                     const std::string& context,
                                     std::vector<std::string>& errors) {
    std::vector<std::string> nodes;
    if (!node || !node.IsSequence()) {
        if (!node) {
            push_error(errors, kDiagInvalidPinCount, "Missing nodes in component '" + context + "'");
        } else {
            push_type_mismatch_error(errors, context + ".nodes", "sequence", node);
        }
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
    const auto first_non_space = content.find_first_not_of(" \t\r\n");
    if (first_non_space != std::string::npos) {
        const char first = content[first_non_space];
        if ((first == '{' || first == '[') && content.find('"') != std::string::npos) {
            errors_.push_back(
                "JSON netlists are unsupported in the v1 YAML parser. "
                "Migrate to pulsim-v1 YAML netlists (see docs/refactor-python-only-v1-hardening/runtime-surface.md).");
            return;
        }
    }

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

    const std::optional<std::string> schema = parse_string_scalar(root["schema"], "root.schema", errors_);
    if (!schema) {
        return;
    }
    if (*schema != kSchemaId) {
        errors_.push_back("Unsupported schema: " + *schema);
        return;
    }

    const std::optional<int> version = parse_int_scalar(root["version"], "root.version", errors_);
    if (!version) {
        return;
    }
    const int schema_version = *version;
    if (schema_version != 1) {
        errors_.push_back("Unsupported schema version: " + std::to_string(schema_version));
        return;
    }

    // Simulation options
    if (root["simulation"]) {
        YAML::Node sim = root["simulation"];
        validate_keys(sim, {"tstart", "tstop", "dt", "dt_min", "dt_max", "step_mode", "adaptive_timestep",
                            "enable_events", "enable_losses", "integrator", "integration", "newton", "timestep",
                            "lte", "bdf", "solver", "shooting", "harmonic_balance", "hb", "thermal",
                            "max_step_retries", "fallback", "backend", "sundials", "advanced"},
                      "simulation", errors_, options_.strict);

        YAML::Node advanced = sim["advanced"];
        if (advanced) {
            validate_keys(advanced, {"adaptive_timestep", "integrator", "integration", "newton", "timestep",
                                     "lte", "bdf", "solver", "fallback", "backend", "sundials"},
                          "simulation.advanced", errors_, options_.strict);
        }

        if (sim["tstart"]) options.tstart = parse_real(sim["tstart"], "simulation.tstart", errors_);
        if (sim["tstop"]) options.tstop = parse_real(sim["tstop"], "simulation.tstop", errors_);
        if (sim["dt"]) options.dt = parse_real(sim["dt"], "simulation.dt", errors_);
        const bool dt_min_explicit = static_cast<bool>(sim["dt_min"]);
        const bool dt_max_explicit = static_cast<bool>(sim["dt_max"]);
        if (sim["dt_min"]) options.dt_min = parse_real(sim["dt_min"], "simulation.dt_min", errors_);
        if (sim["dt_max"]) options.dt_max = parse_real(sim["dt_max"], "simulation.dt_max", errors_);

        if (sim["step_mode"]) {
            try {
                const std::optional<std::string> step_mode_raw =
                    parse_string_scalar(sim["step_mode"], "simulation.step_mode", errors_);
                if (!step_mode_raw) {
                    throw std::runtime_error("invalid_step_mode_type");
                }
                const std::string mode = normalize_key(*step_mode_raw);
                if (mode == "fixed") {
                    apply_mode_derived_defaults(
                        options, TransientStepMode::Fixed, dt_min_explicit, dt_max_explicit);
                } else if (mode == "variable" || mode == "adaptive") {
                    apply_mode_derived_defaults(
                        options, TransientStepMode::Variable, dt_min_explicit, dt_max_explicit);
                } else {
                    push_error(errors_,
                               kDiagInvalidStepMode,
                               "Invalid simulation.step_mode: " + sim["step_mode"].as<std::string>() +
                                   " (expected 'fixed' or 'variable')");
                }
            } catch (...) {
                if (std::none_of(errors_.begin(), errors_.end(), [](const std::string& err) {
                        return err.find("simulation.step_mode") != std::string::npos &&
                               err.find(kDiagTypeMismatch) != std::string::npos;
                    })) {
                    push_error(errors_,
                               kDiagInvalidStepMode,
                               "Invalid simulation.step_mode (expected 'fixed' or 'variable')");
                }
            }
        }

        auto expert_node = [&](const char* key) -> YAML::Node {
            if (advanced && advanced[key]) {
                return advanced[key];
            }
            return sim[key];
        };

        YAML::Node adaptive_timestep = expert_node("adaptive_timestep");
        if (adaptive_timestep) {
            const std::string adaptive_path =
                (advanced && advanced["adaptive_timestep"])
                    ? "simulation.advanced.adaptive_timestep"
                    : "simulation.adaptive_timestep";
            if (!sim["step_mode"]) {
                push_deprecated_field_migration_warning(
                    warnings_,
                    adaptive_path,
                    "simulation.step_mode: fixed|variable");
            }
            if (const auto adaptive = parse_bool_scalar(adaptive_timestep, adaptive_path, errors_)) {
                options.adaptive_timestep = *adaptive;
            }
        }
        if (sim["enable_events"]) {
            if (const auto value = parse_bool_scalar(sim["enable_events"], "simulation.enable_events", errors_)) {
                options.enable_events = *value;
            }
        }
        if (sim["enable_losses"]) {
            if (const auto value = parse_bool_scalar(sim["enable_losses"], "simulation.enable_losses", errors_)) {
                options.enable_losses = *value;
            }
        }
        if (sim["max_step_retries"]) {
            if (const auto value = parse_int_scalar(sim["max_step_retries"], "simulation.max_step_retries", errors_)) {
                options.max_step_retries = *value;
            }
        }

        if (options_.strict) {
            if (sim["backend"]) {
                push_legacy_backend_migration_error(errors_, "simulation.backend");
            }
            if (sim["sundials"]) {
                push_legacy_backend_migration_error(errors_, "simulation.sundials");
            }
            if (advanced && advanced["backend"]) {
                push_legacy_backend_migration_error(errors_, "simulation.advanced.backend");
            }
            if (advanced && advanced["sundials"]) {
                push_legacy_backend_migration_error(errors_, "simulation.advanced.sundials");
            }
            if (sim["fallback"]) {
                const YAML::Node fallback = sim["fallback"];
                if (fallback["enable_backend_escalation"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.fallback.enable_backend_escalation");
                }
                if (fallback["backend_escalation_threshold"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.fallback.backend_escalation_threshold");
                }
                if (fallback["enable_native_reentry"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.fallback.enable_native_reentry");
                }
                if (fallback["sundials_recovery_window"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.fallback.sundials_recovery_window");
                }
            }
            if (advanced && advanced["fallback"]) {
                const YAML::Node fallback = advanced["fallback"];
                if (fallback["enable_backend_escalation"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.advanced.fallback.enable_backend_escalation");
                }
                if (fallback["backend_escalation_threshold"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.advanced.fallback.backend_escalation_threshold");
                }
                if (fallback["enable_native_reentry"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.advanced.fallback.enable_native_reentry");
                }
                if (fallback["sundials_recovery_window"]) {
                    push_legacy_backend_migration_error(errors_, "simulation.advanced.fallback.sundials_recovery_window");
                }
            }
        } else {
            if (sim["backend"]) {
                push_deprecated_field_migration_warning(
                    warnings_,
                    "simulation.backend",
                    "simulation.step_mode: fixed|variable");
            }
            if (sim["sundials"]) {
                push_deprecated_field_migration_warning(
                    warnings_,
                    "simulation.sundials",
                    "simulation.step_mode: fixed|variable");
            }
            if (advanced && advanced["backend"]) {
                push_deprecated_field_migration_warning(
                    warnings_,
                    "simulation.advanced.backend",
                    "simulation.step_mode: fixed|variable");
            }
            if (advanced && advanced["sundials"]) {
                push_deprecated_field_migration_warning(
                    warnings_,
                    "simulation.advanced.sundials",
                    "simulation.step_mode: fixed|variable");
            }
            if (sim["fallback"]) {
                const YAML::Node fallback = sim["fallback"];
                if (fallback["enable_backend_escalation"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.fallback.enable_backend_escalation",
                        "simulation.fallback trace_retries/gmin_*");
                }
                if (fallback["backend_escalation_threshold"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.fallback.backend_escalation_threshold",
                        "simulation.fallback trace_retries/gmin_*");
                }
                if (fallback["enable_native_reentry"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.fallback.enable_native_reentry",
                        "simulation.fallback trace_retries/gmin_*");
                }
                if (fallback["sundials_recovery_window"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.fallback.sundials_recovery_window",
                        "simulation.fallback trace_retries/gmin_*");
                }
            }
            if (advanced && advanced["fallback"]) {
                const YAML::Node fallback = advanced["fallback"];
                if (fallback["enable_backend_escalation"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.advanced.fallback.enable_backend_escalation",
                        "simulation.fallback trace_retries/gmin_*");
                }
                if (fallback["backend_escalation_threshold"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.advanced.fallback.backend_escalation_threshold",
                        "simulation.fallback trace_retries/gmin_*");
                }
                if (fallback["enable_native_reentry"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.advanced.fallback.enable_native_reentry",
                        "simulation.fallback trace_retries/gmin_*");
                }
                if (fallback["sundials_recovery_window"]) {
                    push_deprecated_field_migration_warning(
                        warnings_,
                        "simulation.advanced.fallback.sundials_recovery_window",
                        "simulation.fallback trace_retries/gmin_*");
                }
            }
        }

        if (sim["thermal"]) {
            YAML::Node thermal = sim["thermal"];
            validate_keys(thermal, {"enabled", "ambient", "policy", "default_rth", "default_cth"},
                          "simulation.thermal", errors_, options_.strict);
            options.thermal.enable = true;
            if (thermal["enabled"]) options.thermal.enable = thermal["enabled"].as<bool>();
            if (thermal["ambient"]) options.thermal.ambient = parse_real(thermal["ambient"], "simulation.thermal.ambient", errors_);
            if (thermal["default_rth"]) options.thermal.default_rth = parse_real(thermal["default_rth"], "simulation.thermal.default_rth", errors_);
            if (thermal["default_cth"]) options.thermal.default_cth = parse_real(thermal["default_cth"], "simulation.thermal.default_cth", errors_);
            if (thermal["policy"]) {
                const std::string policy = normalize_key(thermal["policy"].as<std::string>());
                if (policy == "lossonly") {
                    options.thermal.policy = ThermalCouplingPolicy::LossOnly;
                } else if (policy == "losswithtemperaturescaling" ||
                           policy == "losswithscaling") {
                    options.thermal.policy = ThermalCouplingPolicy::LossWithTemperatureScaling;
                } else {
                    errors_.push_back("Invalid thermal policy: " + thermal["policy"].as<std::string>());
                }
            }
        }

        YAML::Node fallback_root = expert_node("fallback");
        if (fallback_root) {
            YAML::Node fallback = fallback_root;
            validate_keys(fallback, {"trace_retries", "enable_transient_gmin",
                                     "gmin_retry_threshold", "gmin_initial",
                                     "gmin_max", "gmin_growth",
                                     "enable_backend_escalation", "backend_escalation_threshold",
                                     "enable_native_reentry", "sundials_recovery_window"},
                          "simulation.advanced.fallback", errors_, options_.strict);
            if (fallback["trace_retries"]) {
                options.fallback_policy.trace_retries = fallback["trace_retries"].as<bool>();
            }
            if (fallback["enable_transient_gmin"]) {
                options.fallback_policy.enable_transient_gmin = fallback["enable_transient_gmin"].as<bool>();
            }
            if (fallback["gmin_retry_threshold"]) {
                options.fallback_policy.gmin_retry_threshold = fallback["gmin_retry_threshold"].as<int>();
            }
            if (fallback["gmin_initial"]) {
                options.fallback_policy.gmin_initial = parse_real(fallback["gmin_initial"], "simulation.fallback.gmin_initial", errors_);
            }
            if (fallback["gmin_max"]) {
                options.fallback_policy.gmin_max = parse_real(fallback["gmin_max"], "simulation.fallback.gmin_max", errors_);
            }
            if (fallback["gmin_growth"]) {
                options.fallback_policy.gmin_growth = parse_real(fallback["gmin_growth"], "simulation.fallback.gmin_growth", errors_);
            }
        }

        YAML::Node integrator_node = expert_node("integrator");
        if (!integrator_node) {
            integrator_node = expert_node("integration");
        }
        if (integrator_node) {
            std::string method = normalize_key(integrator_node.as<std::string>());
            if (method == "trapezoidal" || method == "tr") {
                options.integrator = Integrator::Trapezoidal;
            } else if (method == "bdf1" || method == "be" || method == "backwardeuler") {
                options.integrator = Integrator::BDF1;
            } else if (method == "bdf2") {
                options.integrator = Integrator::BDF2;
            } else if (method == "bdf3") {
                options.integrator = Integrator::BDF3;
            } else if (method == "bdf4") {
                options.integrator = Integrator::BDF4;
            } else if (method == "bdf5") {
                options.integrator = Integrator::BDF5;
            } else if (method == "gear") {
                options.integrator = Integrator::Gear;
            } else if (method == "trbdf2") {
                options.integrator = Integrator::TRBDF2;
            } else if (method == "rosenbrock" || method == "rosenbrockw" || method == "rosenbrockw2") {
                options.integrator = Integrator::RosenbrockW;
            } else if (method == "sdirk2") {
                options.integrator = Integrator::SDIRK2;
            } else {
                errors_.push_back("Invalid integrator: " + integrator_node.as<std::string>());
            }
        }

        YAML::Node newton = expert_node("newton");
        if (newton) {
            YAML::Node n = newton;
            validate_keys(n, {"max_iterations", "initial_damping", "min_damping", "auto_damping",
                              "enable_anderson", "anderson_depth", "anderson_beta",
                              "enable_broyden", "broyden_max_size", "enable_newton_krylov",
                              "enable_trust_region", "trust_radius", "trust_shrink", "trust_expand",
                              "trust_min", "trust_max", "reuse_jacobian_pattern",
                              "krylov_residual_cache_tolerance"},
                          "simulation.newton", errors_, options_.strict);
            if (n["max_iterations"]) options.newton_options.max_iterations = n["max_iterations"].as<int>();
            if (n["initial_damping"]) options.newton_options.initial_damping = parse_real(n["initial_damping"], "newton.initial_damping", errors_);
            if (n["min_damping"]) options.newton_options.min_damping = parse_real(n["min_damping"], "newton.min_damping", errors_);
            if (n["auto_damping"]) options.newton_options.auto_damping = n["auto_damping"].as<bool>();
            if (n["enable_anderson"]) options.newton_options.enable_anderson = n["enable_anderson"].as<bool>();
            if (n["anderson_depth"]) options.newton_options.anderson_depth = n["anderson_depth"].as<int>();
            if (n["anderson_beta"]) options.newton_options.anderson_beta = parse_real(n["anderson_beta"], "newton.anderson_beta", errors_);
            if (n["enable_broyden"]) options.newton_options.enable_broyden = n["enable_broyden"].as<bool>();
            if (n["broyden_max_size"]) options.newton_options.broyden_max_size = n["broyden_max_size"].as<int>();
            if (n["enable_newton_krylov"]) options.newton_options.enable_newton_krylov = n["enable_newton_krylov"].as<bool>();
            if (n["enable_trust_region"]) options.newton_options.enable_trust_region = n["enable_trust_region"].as<bool>();
            if (n["trust_radius"]) options.newton_options.trust_radius = parse_real(n["trust_radius"], "newton.trust_radius", errors_);
            if (n["trust_shrink"]) options.newton_options.trust_shrink = parse_real(n["trust_shrink"], "newton.trust_shrink", errors_);
            if (n["trust_expand"]) options.newton_options.trust_expand = parse_real(n["trust_expand"], "newton.trust_expand", errors_);
            if (n["trust_min"]) options.newton_options.trust_min = parse_real(n["trust_min"], "newton.trust_min", errors_);
            if (n["trust_max"]) options.newton_options.trust_max = parse_real(n["trust_max"], "newton.trust_max", errors_);
            if (n["reuse_jacobian_pattern"]) options.newton_options.reuse_jacobian_pattern = n["reuse_jacobian_pattern"].as<bool>();
            if (n["krylov_residual_cache_tolerance"]) {
                options.newton_options.krylov_residual_cache_tolerance =
                    parse_real(n["krylov_residual_cache_tolerance"], "newton.krylov_residual_cache_tolerance", errors_);
            }
        }

        YAML::Node timestep = expert_node("timestep");
        if (timestep) {
            YAML::Node t = timestep;
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

        YAML::Node lte = expert_node("lte");
        if (lte) {
            YAML::Node l = lte;
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

        YAML::Node bdf = expert_node("bdf");
        if (bdf) {
            YAML::Node b = bdf;
            validate_keys(b, {"enable", "min_order", "max_order", "initial_order"},
                          "simulation.bdf", errors_, options_.strict);
            if (b["enable"]) options.enable_bdf_order_control = b["enable"].as<bool>();
            if (b["min_order"]) options.bdf_config.min_order = b["min_order"].as<int>();
            if (b["max_order"]) options.bdf_config.max_order = b["max_order"].as<int>();
            if (b["initial_order"]) options.bdf_config.initial_order = b["initial_order"].as<int>();
        }

        YAML::Node solver = expert_node("solver");
        if (solver) {
            YAML::Node s = solver;
            validate_keys(s, {"linear", "iterative", "nonlinear", "order", "fallback_order",
                              "allow_fallback", "auto_select", "size_threshold", "nnz_threshold",
                              "diag_min_threshold", "preconditioner", "ilut_drop_tolerance",
                              "ilut_fill_factor"},
                          "simulation.solver", errors_, options_.strict);

            auto parse_linear_kind = [&](const std::string& value, const std::string& path) -> std::optional<LinearSolverKind> {
                std::string v = to_lower(value);
                if (v == "klu") return LinearSolverKind::KLU;
                if (v == "sparse_lu" || v == "sparselu" || v == "sparse") return LinearSolverKind::SparseLU;
                if (v == "enhanced_sparse_lu" || v == "enhanced" || v == "enhanced_sparselu") return LinearSolverKind::EnhancedSparseLU;
                if (v == "gmres") return LinearSolverKind::GMRES;
                if (v == "bicgstab" || v == "bicg") return LinearSolverKind::BiCGSTAB;
                if (v == "cg" || v == "conjugate_gradient") return LinearSolverKind::CG;
                errors_.push_back("Invalid solver kind for " + path + ": " + value);
                return std::nullopt;
            };

            auto parse_order = [&](const YAML::Node& node, const std::string& path,
                                   std::vector<LinearSolverKind>& target) {
                if (!node.IsSequence()) {
                    errors_.push_back(path + " must be a list");
                    return;
                }
                target.clear();
                for (const auto& it : node) {
                    if (!it.IsScalar()) {
                        errors_.push_back(path + " entries must be strings");
                        continue;
                    }
                    auto kind = parse_linear_kind(it.as<std::string>(), path);
                    if (kind) target.push_back(*kind);
                }
            };

            if (s["order"]) {
                parse_order(s["order"], "simulation.solver.order", options.linear_solver.order);
            }
            if (s["fallback_order"]) {
                parse_order(s["fallback_order"], "simulation.solver.fallback_order",
                            options.linear_solver.fallback_order);
            }
            if (s["allow_fallback"]) options.linear_solver.allow_fallback = s["allow_fallback"].as<bool>();
            if (s["auto_select"]) options.linear_solver.auto_select = s["auto_select"].as<bool>();
            if (s["size_threshold"]) options.linear_solver.size_threshold = s["size_threshold"].as<int>();
            if (s["nnz_threshold"]) options.linear_solver.nnz_threshold = s["nnz_threshold"].as<int>();
            if (s["diag_min_threshold"]) options.linear_solver.diag_min_threshold = parse_real(s["diag_min_threshold"], "solver.diag_min_threshold", errors_);

            if (s["preconditioner"]) {
                std::string pre = to_lower(s["preconditioner"].as<std::string>());
                if (pre == "none") {
                    options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::None;
                } else if (pre == "jacobi") {
                    options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::Jacobi;
                } else if (pre == "ilu0" || pre == "ilu") {
                    options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::ILU0;
                } else if (pre == "ilut") {
                    options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::ILUT;
                } else if (pre == "amg") {
                    options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::AMG;
                } else {
                    errors_.push_back("Invalid solver.preconditioner: " + pre);
                }
            }
            if (s["ilut_drop_tolerance"]) {
                options.linear_solver.iterative_config.ilut_drop_tolerance =
                    parse_real(s["ilut_drop_tolerance"], "solver.ilut_drop_tolerance", errors_);
            }
            if (s["ilut_fill_factor"]) {
                options.linear_solver.iterative_config.ilut_fill_factor =
                    parse_real(s["ilut_fill_factor"], "solver.ilut_fill_factor", errors_);
            }

            if (s["linear"]) {
                YAML::Node l = s["linear"];
                validate_keys(l, {"order", "allow_fallback", "auto_select", "size_threshold",
                                  "nnz_threshold", "diag_min_threshold"},
                              "simulation.solver.linear", errors_, options_.strict);
                if (l["order"]) parse_order(l["order"], "simulation.solver.linear.order",
                                            options.linear_solver.order);
                if (l["allow_fallback"]) options.linear_solver.allow_fallback = l["allow_fallback"].as<bool>();
                if (l["auto_select"]) options.linear_solver.auto_select = l["auto_select"].as<bool>();
                if (l["size_threshold"]) options.linear_solver.size_threshold = l["size_threshold"].as<int>();
                if (l["nnz_threshold"]) options.linear_solver.nnz_threshold = l["nnz_threshold"].as<int>();
                if (l["diag_min_threshold"]) options.linear_solver.diag_min_threshold = parse_real(l["diag_min_threshold"], "solver.linear.diag_min_threshold", errors_);
            }

            if (s["iterative"]) {
                YAML::Node it = s["iterative"];
                validate_keys(it, {"max_iterations", "tolerance", "restart", "preconditioner",
                                   "enable_scaling", "scaling_floor", "ilut_drop_tolerance",
                                   "ilut_fill_factor"},
                              "simulation.solver.iterative", errors_, options_.strict);
                if (it["max_iterations"]) options.linear_solver.iterative_config.max_iterations = it["max_iterations"].as<int>();
                if (it["tolerance"]) options.linear_solver.iterative_config.tolerance = parse_real(it["tolerance"], "solver.iterative.tolerance", errors_);
                if (it["restart"]) options.linear_solver.iterative_config.restart = it["restart"].as<int>();
                if (it["enable_scaling"]) options.linear_solver.iterative_config.enable_scaling = it["enable_scaling"].as<bool>();
                if (it["scaling_floor"]) options.linear_solver.iterative_config.scaling_floor = parse_real(it["scaling_floor"], "solver.iterative.scaling_floor", errors_);
                if (it["preconditioner"]) {
                    std::string pre = to_lower(it["preconditioner"].as<std::string>());
                    if (pre == "none") {
                        options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::None;
                    } else if (pre == "jacobi") {
                        options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::Jacobi;
                    } else if (pre == "ilu0" || pre == "ilu") {
                        options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::ILU0;
                    } else if (pre == "ilut") {
                        options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::ILUT;
                    } else if (pre == "amg") {
                        options.linear_solver.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::AMG;
                    } else {
                        errors_.push_back("Invalid solver.iterative.preconditioner: " + pre);
                    }
                }
                if (it["ilut_drop_tolerance"]) {
                    options.linear_solver.iterative_config.ilut_drop_tolerance =
                        parse_real(it["ilut_drop_tolerance"], "solver.iterative.ilut_drop_tolerance", errors_);
                }
                if (it["ilut_fill_factor"]) {
                    options.linear_solver.iterative_config.ilut_fill_factor =
                        parse_real(it["ilut_fill_factor"], "solver.iterative.ilut_fill_factor", errors_);
                }
            }

            if (s["nonlinear"]) {
                YAML::Node nl = s["nonlinear"];
                validate_keys(nl, {"anderson", "broyden", "newton_krylov", "trust_region", "reuse_jacobian_pattern"},
                              "simulation.solver.nonlinear", errors_, options_.strict);

                if (nl["anderson"]) {
                    YAML::Node a = nl["anderson"];
                    validate_keys(a, {"enable", "depth", "beta"},
                                  "simulation.solver.nonlinear.anderson", errors_, options_.strict);
                    if (a["enable"]) options.newton_options.enable_anderson = a["enable"].as<bool>();
                    if (a["depth"]) options.newton_options.anderson_depth = a["depth"].as<int>();
                    if (a["beta"]) options.newton_options.anderson_beta = parse_real(a["beta"], "solver.nonlinear.anderson.beta", errors_);
                }

                if (nl["broyden"]) {
                    YAML::Node b = nl["broyden"];
                    validate_keys(b, {"enable", "max_size"},
                                  "simulation.solver.nonlinear.broyden", errors_, options_.strict);
                    if (b["enable"]) options.newton_options.enable_broyden = b["enable"].as<bool>();
                    if (b["max_size"]) options.newton_options.broyden_max_size = b["max_size"].as<int>();
                }

                if (nl["newton_krylov"]) {
                    YAML::Node nk = nl["newton_krylov"];
                    validate_keys(nk, {"enable"},
                                  "simulation.solver.nonlinear.newton_krylov", errors_, options_.strict);
                    if (nk["enable"]) options.newton_options.enable_newton_krylov = nk["enable"].as<bool>();
                }

                if (nl["trust_region"]) {
                    YAML::Node tr = nl["trust_region"];
                    validate_keys(tr, {"enable", "radius", "shrink", "expand", "min", "max"},
                                  "simulation.solver.nonlinear.trust_region", errors_, options_.strict);
                    if (tr["enable"]) options.newton_options.enable_trust_region = tr["enable"].as<bool>();
                    if (tr["radius"]) options.newton_options.trust_radius = parse_real(tr["radius"], "solver.nonlinear.trust_region.radius", errors_);
                    if (tr["shrink"]) options.newton_options.trust_shrink = parse_real(tr["shrink"], "solver.nonlinear.trust_region.shrink", errors_);
                    if (tr["expand"]) options.newton_options.trust_expand = parse_real(tr["expand"], "solver.nonlinear.trust_region.expand", errors_);
                    if (tr["min"]) options.newton_options.trust_min = parse_real(tr["min"], "solver.nonlinear.trust_region.min", errors_);
                    if (tr["max"]) options.newton_options.trust_max = parse_real(tr["max"], "solver.nonlinear.trust_region.max", errors_);
                }

                if (nl["reuse_jacobian_pattern"]) {
                    options.newton_options.reuse_jacobian_pattern = nl["reuse_jacobian_pattern"].as<bool>();
                }
            }
        }

        enforce_mode_semantics(options);

        if (sim["shooting"]) {
            YAML::Node sh = sim["shooting"];
            validate_keys(sh, {"period", "max_iterations", "tolerance", "relaxation", "store_last_transient"},
                          "simulation.shooting", errors_, options_.strict);
            options.enable_periodic_shooting = true;
            if (sh["period"]) options.periodic_options.period = parse_real(sh["period"], "shooting.period", errors_);
            if (sh["max_iterations"]) options.periodic_options.max_iterations = sh["max_iterations"].as<int>();
            if (sh["tolerance"]) options.periodic_options.tolerance = parse_real(sh["tolerance"], "shooting.tolerance", errors_);
            if (sh["relaxation"]) options.periodic_options.relaxation = parse_real(sh["relaxation"], "shooting.relaxation", errors_);
            if (sh["store_last_transient"]) options.periodic_options.store_last_transient = sh["store_last_transient"].as<bool>();
        }

        YAML::Node hb = sim["harmonic_balance"] ? sim["harmonic_balance"] : sim["hb"];
        if (hb) {
            validate_keys(hb, {"period", "num_samples", "max_iterations", "tolerance",
                               "relaxation", "initialize_from_transient"},
                          "simulation.harmonic_balance", errors_, options_.strict);
            options.enable_harmonic_balance = true;
            if (hb["period"]) options.harmonic_balance.period = parse_real(hb["period"], "hb.period", errors_);
            if (hb["num_samples"]) options.harmonic_balance.num_samples = hb["num_samples"].as<int>();
            if (hb["max_iterations"]) options.harmonic_balance.max_iterations = hb["max_iterations"].as<int>();
            if (hb["tolerance"]) options.harmonic_balance.tolerance = parse_real(hb["tolerance"], "hb.tolerance", errors_);
            if (hb["relaxation"]) options.harmonic_balance.relaxation = parse_real(hb["relaxation"], "hb.relaxation", errors_);
            if (hb["initialize_from_transient"]) {
                options.harmonic_balance.initialize_from_transient =
                    hb["initialize_from_transient"].as<bool>();
            }
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

        const YAML::Node comp = comp_node;

        validate_keys(comp,
            {"type", "name", "nodes", "value", "params", "waveform", "use", "loss",
             "thermal",
             "resistance", "capacitance", "inductance", "ic",
             "g_on", "g_off", "ron", "roff", "v_threshold", "initial_state",
             "vth", "kp", "lambda", "lambda_", "is_nmos", "v_ce_sat",
             "turns_ratio", "ratio", "magnetizing_inductance", "lm",
             "target_component", "target_device", "target", "operation", "channels", "inputs", "outputs",
             "x", "y", "num", "den", "delay", "sample_period",
             "trip_current", "trip_time", "trip", "rating", "blow_i2t", "i2t",
             "pickup_current", "dropout_current", "pickup_voltage", "dropout_voltage",
             "contact_resistance", "off_resistance", "target_component_no", "target_component_nc",
             "beta", "vbe_on", "holding_current", "latch_current", "gate_threshold",
             "saturation_current", "saturation_inductance", "saturation_exponent",
             "coupling", "k", "mutual_inductance", "l1", "l2", "ic1", "ic2", "ic_primary", "ic_secondary",
             "state", "mode", "select_index", "gain", "open_loop_gain", "offset",
             "kp", "ki", "kd", "threshold", "high", "low",
             "frequency", "duty", "duty_min", "duty_max", "duty_from_input", "duty_gain", "duty_offset",
             "alpha", "anti_windup", "min", "max", "output_min", "output_max",
             "rising_rate", "falling_rate",
             "rail_high", "rail_low", "hysteresis", "metadata", "table", "mapping"},
            "component", errors_, options_.strict);

        if (!comp["type"] || !comp["name"]) {
            errors_.push_back("Component missing type or name");
            continue;
        }

        const std::string raw_type = comp["type"].as<std::string>();
        const std::string normalized_raw_type = normalize_key(raw_type);
        std::string type = canonical_component_type(raw_type);
        if (type.empty()) {
            push_error(errors_, kDiagUnsupportedComponent,
                       "Unsupported component type: " + raw_type);
            continue;
        }

        std::string name = comp["name"].as<std::string>();
        if (!comp["nodes"]) {
            push_error(errors_, kDiagInvalidPinCount, "Component missing nodes: " + name);
            continue;
        }

        std::vector<std::string> nodes = parse_nodes(comp["nodes"], name, errors_);
        if (nodes.empty()) continue;
        if (!validate_node_count(type, nodes, errors_, name)) continue;

        const YAML::Node comp_view = comp;
        const YAML::Node params = comp_view["params"];

        auto get_param = [&](const std::string& key) -> YAML::Node {
            try {
                YAML::Node top = comp_view[key];
                if (top.IsDefined() && !top.IsNull()) return top;
                YAML::Node nested = (params && params.IsMap()) ? params[key] : YAML::Node();
                if (nested.IsDefined() && !nested.IsNull()) return nested;
                if (key == "lambda") {
                    top = comp_view["lambda_"];
                    if (top.IsDefined() && !top.IsNull()) return top;
                    nested = (params && params.IsMap()) ? params["lambda_"] : YAML::Node();
                    if (nested.IsDefined() && !nested.IsNull()) return nested;
                }
                if (key == "target_component") {
                    top = comp_view["target_device"];
                    if (top.IsDefined() && !top.IsNull()) return top;
                    nested = (params && params.IsMap()) ? params["target_device"] : YAML::Node();
                    if (nested.IsDefined() && !nested.IsNull()) return nested;
                }
            } catch (const YAML::Exception&) {
                return YAML::Node();
            }
            return YAML::Node();
        };

        auto idx = [&](const std::string& node_name) {
            return circuit.add_node(node_name);
        };
        std::vector<Index> node_indices;
        node_indices.reserve(nodes.size());
        for (const auto& node_name : nodes) {
            node_indices.push_back(idx(node_name));
        }
        auto node_at = [&](std::size_t index) -> Index {
            return node_indices.at(index);
        };
        auto is_set = [](const YAML::Node& node) {
            return node.IsDefined() && !node.IsNull();
        };

        // Loss model (switching energy)
        if (comp["loss"]) {
            YAML::Node loss = comp["loss"];
            validate_keys(loss, {"eon", "eoff", "err"}, name + ".loss", errors_, options_.strict);
            SwitchingEnergy energy;
            if (loss["eon"]) energy.eon = parse_real(loss["eon"], name + ".loss.eon", errors_);
            if (loss["eoff"]) energy.eoff = parse_real(loss["eoff"], name + ".loss.eoff", errors_);
            if (loss["err"]) energy.err = parse_real(loss["err"], name + ".loss.err", errors_);
            options.switching_energy[name] = energy;
        }

        if (comp["thermal"]) {
            YAML::Node thermal = comp["thermal"];
            validate_keys(thermal, {"enabled", "rth", "cth", "temp_init", "temp_ref", "alpha"},
                          name + ".thermal", errors_, options_.strict);
            ThermalDeviceConfig cfg;
            if (thermal["enabled"]) cfg.enabled = thermal["enabled"].as<bool>();
            if (thermal["rth"]) cfg.rth = parse_real(thermal["rth"], name + ".thermal.rth", errors_);
            if (thermal["cth"]) cfg.cth = parse_real(thermal["cth"], name + ".thermal.cth", errors_);
            if (thermal["temp_init"]) cfg.temp_init = parse_real(thermal["temp_init"], name + ".thermal.temp_init", errors_);
            if (thermal["temp_ref"]) cfg.temp_ref = parse_real(thermal["temp_ref"], name + ".thermal.temp_ref", errors_);
            if (thermal["alpha"]) cfg.alpha = parse_real(thermal["alpha"], name + ".thermal.alpha", errors_);
            options.thermal_devices[name] = cfg;
            if (cfg.enabled) {
                options.thermal.enable = true;
            }
        }

        if (type == "resistor") {
            YAML::Node value_node = get_param("value");
            if (!value_node) value_node = get_param("resistance");
            Real value = parse_real(value_node, name + ".value", errors_);
            circuit.add_resistor(name, node_at(0), node_at(1), value);
        }
        else if (type == "capacitor") {
            YAML::Node value_node = get_param("value");
            if (!value_node) value_node = get_param("capacitance");
            Real value = parse_real(value_node, name + ".value", errors_);
            Real ic = 0.0;
            if (get_param("ic")) ic = parse_real(get_param("ic"), name + ".ic", errors_);
            circuit.add_capacitor(name, node_at(0), node_at(1), value, ic);
        }
        else if (type == "inductor") {
            YAML::Node value_node = get_param("value");
            if (!value_node) value_node = get_param("inductance");
            Real value = parse_real(value_node, name + ".value", errors_);
            Real ic = 0.0;
            if (get_param("ic")) ic = parse_real(get_param("ic"), name + ".ic", errors_);
            circuit.add_inductor(name, node_at(0), node_at(1), value, ic);
        }
        else if (type == "voltage_source") {
            YAML::Node waveform = comp["waveform"];
            if (waveform && waveform["type"]) {
                std::string wtype = to_lower(waveform["type"].as<std::string>());
                if (wtype == "dc") {
                    Real value = parse_real(waveform["value"], name + ".waveform.value", errors_);
                    circuit.add_voltage_source(name, node_at(0), node_at(1), value);
                } else if (wtype == "pwm") {
                    PWMParams p;
                    if (waveform["v_high"]) p.v_high = parse_real(waveform["v_high"], name + ".waveform.v_high", errors_);
                    if (waveform["v_low"]) p.v_low = parse_real(waveform["v_low"], name + ".waveform.v_low", errors_);
                    if (waveform["frequency"]) p.frequency = parse_real(waveform["frequency"], name + ".waveform.frequency", errors_);
                    if (waveform["duty"]) p.duty = parse_real(waveform["duty"], name + ".waveform.duty", errors_);
                    if (waveform["dead_time"]) p.dead_time = parse_real(waveform["dead_time"], name + ".waveform.dead_time", errors_);
                    if (waveform["phase"]) p.phase = parse_real(waveform["phase"], name + ".waveform.phase", errors_);
                    circuit.add_pwm_voltage_source(name, node_at(0), node_at(1), p);
                } else if (wtype == "sine") {
                    SineParams p;
                    if (waveform["amplitude"]) p.amplitude = parse_real(waveform["amplitude"], name + ".waveform.amplitude", errors_);
                    if (waveform["frequency"]) p.frequency = parse_real(waveform["frequency"], name + ".waveform.frequency", errors_);
                    if (waveform["offset"]) p.offset = parse_real(waveform["offset"], name + ".waveform.offset", errors_);
                    if (waveform["phase"]) p.phase = parse_real(waveform["phase"], name + ".waveform.phase", errors_);
                    circuit.add_sine_voltage_source(name, node_at(0), node_at(1), p);
                } else if (wtype == "pulse") {
                    PulseParams p;
                    if (waveform["v_initial"]) p.v_initial = parse_real(waveform["v_initial"], name + ".waveform.v_initial", errors_);
                    if (waveform["v_pulse"]) p.v_pulse = parse_real(waveform["v_pulse"], name + ".waveform.v_pulse", errors_);
                    if (waveform["t_delay"]) p.t_delay = parse_real(waveform["t_delay"], name + ".waveform.t_delay", errors_);
                    if (waveform["t_rise"]) p.t_rise = parse_real(waveform["t_rise"], name + ".waveform.t_rise", errors_);
                    if (waveform["t_fall"]) p.t_fall = parse_real(waveform["t_fall"], name + ".waveform.t_fall", errors_);
                    if (waveform["t_width"]) p.t_width = parse_real(waveform["t_width"], name + ".waveform.t_width", errors_);
                    if (waveform["period"]) p.period = parse_real(waveform["period"], name + ".waveform.period", errors_);
                    circuit.add_pulse_voltage_source(name, node_at(0), node_at(1), p);
                } else {
                    errors_.push_back("Unsupported waveform type for voltage_source: " + wtype);
                }
            } else {
                Real value = parse_real(get_param("value"), name + ".value", errors_);
                circuit.add_voltage_source(name, node_at(0), node_at(1), value);
            }
        }
        else if (type == "current_source") {
            Real value = parse_real(get_param("value"), name + ".value", errors_);
            circuit.add_current_source(name, node_at(0), node_at(1), value);
        }
        else if (type == "diode") {
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
            circuit.add_diode(name, node_at(0), node_at(1), g_on, g_off);
        }
        else if (type == "switch") {
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
            YAML::Node initial_state_node = get_param("initial_state");
            bool closed = (initial_state_node.IsDefined() && !initial_state_node.IsNull())
                ? initial_state_node.as<bool>()
                : false;
            circuit.add_switch(name, node_at(0), node_at(1), closed, g_on, g_off);
        }
        else if (type == "vcswitch") {
            Real v_threshold = get_param("v_threshold") ? parse_real(get_param("v_threshold"), name + ".v_threshold", errors_) : 2.5;
            Real g_on = get_param("g_on") ? parse_real(get_param("g_on"), name + ".g_on", errors_) : 1e3;
            Real g_off = get_param("g_off") ? parse_real(get_param("g_off"), name + ".g_off", errors_) : 1e-9;
            circuit.add_vcswitch(name, node_at(0), node_at(1), node_at(2), v_threshold, g_on, g_off);
        }
        else if (type == "mosfet") {
            MOSFET::Params p;
            if (get_param("vth")) p.vth = parse_real(get_param("vth"), name + ".vth", errors_);
            if (get_param("kp")) p.kp = parse_real(get_param("kp"), name + ".kp", errors_);
            if (get_param("lambda")) p.lambda = parse_real(get_param("lambda"), name + ".lambda", errors_);
            if (get_param("g_off")) p.g_off = parse_real(get_param("g_off"), name + ".g_off", errors_);
            YAML::Node is_nmos_node = get_param("is_nmos");
            if (is_nmos_node.IsDefined() && !is_nmos_node.IsNull()) p.is_nmos = is_nmos_node.as<bool>();
            if (normalized_raw_type == "pmos") p.is_nmos = false;
            circuit.add_mosfet(name, node_at(0), node_at(1), node_at(2), p);
        }
        else if (type == "igbt") {
            IGBT::Params p;
            if (get_param("vth")) p.vth = parse_real(get_param("vth"), name + ".vth", errors_);
            if (get_param("g_on")) p.g_on = parse_real(get_param("g_on"), name + ".g_on", errors_);
            if (get_param("g_off")) p.g_off = parse_real(get_param("g_off"), name + ".g_off", errors_);
            if (get_param("v_ce_sat")) p.v_ce_sat = parse_real(get_param("v_ce_sat"), name + ".v_ce_sat", errors_);
            circuit.add_igbt(name, node_at(0), node_at(1), node_at(2), p);
        }
        else if (type == "transformer") {
            YAML::Node ratio_node = get_param("turns_ratio");
            if (!ratio_node) ratio_node = get_param("ratio");
            Real turns_ratio = parse_real(ratio_node, name + ".turns_ratio", errors_);
            circuit.add_transformer(name, node_at(0), node_at(1), node_at(2), node_at(3), turns_ratio);
        }
        else if (type == "snubber_rc") {
            YAML::Node r_node = get_param("resistance");
            if (!r_node) r_node = get_param("value");
            YAML::Node c_node = get_param("capacitance");
            Real r_value = parse_real(r_node, name + ".resistance", errors_);
            Real c_value = parse_real(c_node, name + ".capacitance", errors_);
            Real ic = 0.0;
            if (get_param("ic")) ic = parse_real(get_param("ic"), name + ".ic", errors_);
            circuit.add_snubber_rc(name, node_at(0), node_at(1), r_value, c_value, ic);
        }
        else if (type == "bjt_npn" || type == "bjt_pnp") {
            MOSFET::Params p;
            p.is_nmos = (type == "bjt_npn");
            if (get_param("vbe_on")) p.vth = parse_real(get_param("vbe_on"), name + ".vbe_on", errors_);
            if (get_param("g_off")) p.g_off = parse_real(get_param("g_off"), name + ".g_off", errors_);
            if (get_param("beta")) {
                Real beta = parse_real(get_param("beta"), name + ".beta", errors_);
                if (beta <= 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "beta must be > 0 for " + name);
                    continue;
                }
                // First parity slice: map beta to transconductance proxy.
                p.kp = std::max<Real>(1e-6, beta * 1e-3);
            }
            circuit.add_mosfet(name, node_at(0), node_at(1), node_at(2), p);
            push_warning(warnings_, kDiagSurrogateComponent,
                         "Component '" + name + "' (" + type + ") mapped to MOSFET surrogate model");
        }
        else if (type == "relay") {
            bool energized = false;
            YAML::Node state_node = get_param("initial_state");
            if (state_node.IsDefined() && !state_node.IsNull()) {
                if (state_node.IsScalar()) {
                    try {
                        energized = state_node.as<bool>();
                    } catch (...) {
                        const std::string state = normalize_key(state_node.as<std::string>());
                        energized = (state == "closed" || state == "energized" || state == "on" || state == "true");
                    }
                }
            }

            Real g_on = 1e4;
            Real g_off = 1e-9;
            if (get_param("g_on")) {
                g_on = parse_real(get_param("g_on"), name + ".g_on", errors_);
            } else if (get_param("contact_resistance")) {
                const Real r_on = parse_real(get_param("contact_resistance"), name + ".contact_resistance", errors_);
                g_on = (r_on > 0.0) ? (1.0 / r_on) : 1e4;
            } else if (get_param("ron")) {
                const Real r_on = parse_real(get_param("ron"), name + ".ron", errors_);
                g_on = (r_on > 0.0) ? (1.0 / r_on) : 1e4;
            }

            if (get_param("g_off")) {
                g_off = parse_real(get_param("g_off"), name + ".g_off", errors_);
            } else if (get_param("off_resistance")) {
                const Real r_off = parse_real(get_param("off_resistance"), name + ".off_resistance", errors_);
                g_off = (r_off > 0.0) ? (1.0 / r_off) : 1e-9;
            } else if (get_param("roff")) {
                const Real r_off = parse_real(get_param("roff"), name + ".roff", errors_);
                g_off = (r_off > 0.0) ? (1.0 / r_off) : 1e-9;
            }

            const std::string no_switch = name + "__no";
            const std::string nc_switch = name + "__nc";
            circuit.add_switch(no_switch, node_at(2), node_at(3), energized, g_on, g_off);
            circuit.add_switch(nc_switch, node_at(2), node_at(4), !energized, g_on, g_off);

            std::unordered_map<std::string, Real> numeric_params;
            numeric_params["initial_closed"] = energized ? 1.0 : 0.0;
            numeric_params["g_on"] = g_on;
            numeric_params["g_off"] = g_off;
            if (get_param("pickup_current")) {
                numeric_params["pickup_current"] = parse_real(get_param("pickup_current"), name + ".pickup_current", errors_);
            }
            if (get_param("dropout_current")) {
                numeric_params["dropout_current"] = parse_real(get_param("dropout_current"), name + ".dropout_current", errors_);
            }
            if (get_param("pickup_voltage")) {
                numeric_params["pickup_voltage"] = parse_real(get_param("pickup_voltage"), name + ".pickup_voltage", errors_);
            }
            if (get_param("dropout_voltage")) {
                numeric_params["dropout_voltage"] = parse_real(get_param("dropout_voltage"), name + ".dropout_voltage", errors_);
            }

            std::unordered_map<std::string, std::string> metadata;
            metadata["target_component"] = no_switch;
            metadata["target_component_no"] = no_switch;
            metadata["target_component_nc"] = nc_switch;
            circuit.add_virtual_component(type, name, node_indices, std::move(numeric_params), std::move(metadata));

            push_warning(warnings_, kDiagSurrogateComponent,
                         "Component '" + name + "' (relay) mapped to dual-switch surrogate with event controller");
        }
        else if (type == "thyristor" || type == "triac") {
            Real gate_threshold = get_param("gate_threshold")
                ? parse_real(get_param("gate_threshold"), name + ".gate_threshold", errors_)
                : 1.0;
            Real g_on = get_param("g_on") ? parse_real(get_param("g_on"), name + ".g_on", errors_) : 1e4;
            Real g_off = get_param("g_off") ? parse_real(get_param("g_off"), name + ".g_off", errors_) : 1e-9;
            Real holding_current = get_param("holding_current")
                ? parse_real(get_param("holding_current"), name + ".holding_current", errors_)
                : 0.05;
            Real latch_current = get_param("latch_current")
                ? parse_real(get_param("latch_current"), name + ".latch_current", errors_)
                : (holding_current * 1.2);

            circuit.add_switch(name, node_at(1), node_at(2), false, g_on, g_off);

            std::unordered_map<std::string, Real> numeric_params;
            numeric_params["gate_threshold"] = gate_threshold;
            numeric_params["holding_current"] = holding_current;
            numeric_params["latch_current"] = latch_current;
            numeric_params["g_on"] = g_on;
            numeric_params["g_off"] = g_off;
            numeric_params["initial_closed"] = 0.0;
            std::unordered_map<std::string, std::string> metadata;
            metadata["target_component"] = name;
            circuit.add_virtual_component(type, name, node_indices, std::move(numeric_params), std::move(metadata));

            push_warning(warnings_, kDiagSurrogateComponent,
                         "Component '" + name + "' (" + type + ") mapped to switch surrogate with latching event controller");
        }
        else if (type == "fuse" || type == "circuit_breaker") {
            Real g_on = get_param("g_on") ? parse_real(get_param("g_on"), name + ".g_on", errors_) : 1e4;
            Real g_off = get_param("g_off") ? parse_real(get_param("g_off"), name + ".g_off", errors_) : 1e-9;
            bool closed = true;
            YAML::Node state_node = get_param("initial_state");
            if (state_node.IsDefined() && !state_node.IsNull()) {
                if (state_node.IsScalar()) {
                    try {
                        closed = state_node.as<bool>();
                    } catch (...) {
                        const std::string state = normalize_key(state_node.as<std::string>());
                        closed = !(state == "open" || state == "tripped" || state == "blown" || state == "false");
                    }
                }
            }
            circuit.add_switch(name, node_at(0), node_at(1), closed, g_on, g_off);
            std::unordered_map<std::string, Real> numeric_params;
            numeric_params["g_on"] = g_on;
            numeric_params["g_off"] = g_off;
            numeric_params["initial_closed"] = closed ? 1.0 : 0.0;
            if (get_param("rating")) {
                numeric_params["rating"] = parse_real(get_param("rating"), name + ".rating", errors_);
            }
            if (type == "fuse") {
                YAML::Node blow_i2t = get_param("blow_i2t");
                if (!blow_i2t) blow_i2t = get_param("i2t");
                if (blow_i2t) {
                    numeric_params["blow_i2t"] = parse_real(blow_i2t, name + ".blow_i2t", errors_);
                }
            } else {
                YAML::Node trip_current = get_param("trip_current");
                if (!trip_current) trip_current = get_param("trip");
                if (trip_current) {
                    numeric_params["trip_current"] = parse_real(trip_current, name + ".trip_current", errors_);
                }
                if (get_param("trip_time")) {
                    numeric_params["trip_time"] = parse_real(get_param("trip_time"), name + ".trip_time", errors_);
                }
            }
            std::unordered_map<std::string, std::string> metadata;
            metadata["target_component"] = name;
            circuit.add_virtual_component(type, name, node_indices, std::move(numeric_params), std::move(metadata));
            push_warning(warnings_, kDiagSurrogateComponent,
                         "Component '" + name + "' (" + type + ") mapped to switch surrogate with event controller");
        }
        else if (type == "saturable_inductor") {
            YAML::Node l_node = get_param("inductance");
            if (!is_set(l_node)) l_node = get_param("value");
            const Real inductance = parse_real(l_node, name + ".inductance", errors_);
            if (inductance <= 0.0) {
                push_error(errors_, kDiagInvalidParameter,
                           "inductance must be > 0 for " + name);
                continue;
            }
            Real sat_current = get_param("saturation_current")
                ? parse_real(get_param("saturation_current"), name + ".saturation_current", errors_)
                : 1.0;
            if (sat_current <= 0.0) {
                push_error(errors_, kDiagInvalidParameter,
                           "saturation_current must be > 0 for " + name);
                continue;
            }
            Real sat_inductance = get_param("saturation_inductance")
                ? parse_real(get_param("saturation_inductance"), name + ".saturation_inductance", errors_)
                : (0.2 * inductance);
            if (sat_inductance <= 0.0 || sat_inductance > inductance) {
                push_error(errors_, kDiagInvalidParameter,
                           "saturation_inductance must be in (0, inductance] for " + name);
                continue;
            }
            Real ic = 0.0;
            if (get_param("ic")) ic = parse_real(get_param("ic"), name + ".ic", errors_);
            circuit.add_inductor(name, node_at(0), node_at(1), inductance, ic);
            std::unordered_map<std::string, Real> numeric_params;
            numeric_params["inductance"] = inductance;
            numeric_params["saturation_current"] = sat_current;
            numeric_params["saturation_inductance"] = sat_inductance;
            if (get_param("saturation_exponent")) {
                numeric_params["saturation_exponent"] =
                    parse_real(get_param("saturation_exponent"), name + ".saturation_exponent", errors_);
            }
            std::unordered_map<std::string, std::string> metadata;
            metadata["target_component"] = name;
            circuit.add_virtual_component(type, name, node_indices, std::move(numeric_params), std::move(metadata));
            push_warning(warnings_, kDiagVirtualComponent,
                         "Component '" + name + "' (saturable_inductor) registered with nonlinear inductance controller");
        }
        else if (type == "coupled_inductor") {
            Real l1 = get_param("l1")
                ? parse_real(get_param("l1"), name + ".l1", errors_)
                : (get_param("inductance") ? parse_real(get_param("inductance"), name + ".inductance", errors_) : 1e-3);
            Real l2 = get_param("l2")
                ? parse_real(get_param("l2"), name + ".l2", errors_)
                : l1;

            if (l1 <= 0.0 || l2 <= 0.0) {
                push_error(errors_, kDiagInvalidParameter,
                           "l1 and l2 must be > 0 for " + name);
                continue;
            }

            Real k = 0.98;
            if (get_param("coupling")) {
                k = parse_real(get_param("coupling"), name + ".coupling", errors_);
            } else if (get_param("k")) {
                k = parse_real(get_param("k"), name + ".k", errors_);
            } else if (get_param("mutual_inductance")) {
                const Real m = parse_real(get_param("mutual_inductance"), name + ".mutual_inductance", errors_);
                k = m / std::sqrt(l1 * l2);
            }
            if (std::abs(k) > 0.999) {
                push_error(errors_, kDiagInvalidParameter,
                           "coupling must satisfy |k| <= 0.999 for " + name);
                continue;
            }

            Real ic1 = 0.0;
            Real ic2 = 0.0;
            if (get_param("ic1")) ic1 = parse_real(get_param("ic1"), name + ".ic1", errors_);
            else if (get_param("ic_primary")) ic1 = parse_real(get_param("ic_primary"), name + ".ic_primary", errors_);
            if (get_param("ic2")) ic2 = parse_real(get_param("ic2"), name + ".ic2", errors_);
            else if (get_param("ic_secondary")) ic2 = parse_real(get_param("ic_secondary"), name + ".ic_secondary", errors_);

            const std::string l1_name = name + "__L1";
            const std::string l2_name = name + "__L2";
            circuit.add_inductor(l1_name, node_at(0), node_at(1), l1, ic1);
            circuit.add_inductor(l2_name, node_at(2), node_at(3), l2, ic2);

            std::unordered_map<std::string, Real> numeric_params;
            numeric_params["l1"] = l1;
            numeric_params["l2"] = l2;
            numeric_params["coupling"] = k;
            numeric_params["k"] = k;
            std::unordered_map<std::string, std::string> metadata;
            metadata["target_component_1"] = l1_name;
            metadata["target_component_2"] = l2_name;
            circuit.add_virtual_component(type, name, node_indices, std::move(numeric_params), std::move(metadata));

            push_warning(warnings_, kDiagVirtualComponent,
                         "Component '" + name + "' (coupled_inductor) expanded to coupled inductor pair");
        }
        else if (virtual_component_types().contains(type)) {
            std::unordered_map<std::string, Real> numeric_params;
            std::unordered_map<std::string, std::string> metadata;
            collect_virtual_component_fields(comp, numeric_params, metadata);
            YAML::Node target = get_param("target_component");
            if (!target) target = get_param("target");
            if (target && target.IsScalar()) {
                metadata["target_component"] = target.as<std::string>();
            }

            auto has_num = [&](const std::string& key) {
                return numeric_params.find(key) != numeric_params.end();
            };
            auto get_num = [&](const std::string& key, Real fallback) {
                const auto it = numeric_params.find(key);
                return (it == numeric_params.end()) ? fallback : it->second;
            };
            auto metadata_node = [&](const std::string& key) -> YAML::Node {
                const auto it = metadata.find(key);
                if (it == metadata.end()) return YAML::Node();
                try {
                    return YAML::Load(it->second);
                } catch (...) {
                    return YAML::Node();
                }
            };
            auto metadata_sequence_size = [&](const std::string& key) -> std::size_t {
                const YAML::Node node = metadata_node(key);
                if (!node || !node.IsSequence()) return 0;
                return node.size();
            };

            if (type == "op_amp") {
                const Real lo = get_num("rail_low", -15.0);
                const Real hi = get_num("rail_high", 15.0);
                if (lo >= hi) {
                    push_error(errors_, kDiagInvalidParameter,
                               "rail_low must be < rail_high for " + name);
                    continue;
                }
            } else if (type == "comparator" || type == "hysteresis") {
                const Real hyst = get_num("hysteresis", 0.0);
                if (hyst < 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "hysteresis must be >= 0 for " + name);
                    continue;
                }
                const Real low = get_num("low", 0.0);
                const Real high = get_num("high", 1.0);
                if (high < low) {
                    push_error(errors_, kDiagInvalidParameter,
                               "high must be >= low for " + name);
                    continue;
                }
            } else if (type == "pi_controller" || type == "pid_controller") {
                if (has_num("output_min") && has_num("output_max")) {
                    if (get_num("output_min", 0.0) > get_num("output_max", 0.0)) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "output_min must be <= output_max for " + name);
                        continue;
                    }
                }
            } else if (type == "integrator" || type == "limiter") {
                if (has_num("output_min") && has_num("output_max")) {
                    if (get_num("output_min", 0.0) > get_num("output_max", 0.0)) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "output_min must be <= output_max for " + name);
                        continue;
                    }
                }
                if (has_num("min") && has_num("max")) {
                    if (get_num("min", 0.0) > get_num("max", 0.0)) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "min must be <= max for " + name);
                        continue;
                    }
                }
            } else if (type == "differentiator") {
                if (has_num("alpha")) {
                    const Real alpha = get_num("alpha", 0.0);
                    if (alpha < 0.0 || alpha > 1.0) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "alpha must be in [0, 1] for " + name);
                        continue;
                    }
                }
            } else if (type == "rate_limiter") {
                if (has_num("rising_rate") && get_num("rising_rate", 0.0) < 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "rising_rate must be >= 0 for " + name);
                    continue;
                }
                if (has_num("falling_rate") && get_num("falling_rate", 0.0) < 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "falling_rate must be >= 0 for " + name);
                    continue;
                }
            } else if (type == "lookup_table") {
                const std::size_t x_size = metadata_sequence_size("x");
                const std::size_t y_size = metadata_sequence_size("y");
                std::size_t point_count = 0;
                if (x_size > 0 || y_size > 0) {
                    if (x_size != y_size || x_size < 2) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "lookup_table requires x and y arrays with the same size >= 2 for " + name);
                        continue;
                    }
                    point_count = x_size;
                } else {
                    const YAML::Node table = metadata_node("table");
                    if (table && table.IsSequence() && table.size() >= 2) {
                        point_count = table.size();
                    } else {
                        const YAML::Node mapping = metadata_node("mapping");
                        if (mapping && mapping.IsMap() && mapping.size() >= 2) {
                            point_count = mapping.size();
                        }
                    }
                }
                if (point_count < 2) {
                    push_error(errors_, kDiagInvalidParameter,
                               "lookup_table requires at least two points (x/y, table, or mapping) for " + name);
                    continue;
                }
            } else if (type == "transfer_function") {
                const std::size_t num_size = metadata_sequence_size("num");
                const std::size_t den_size = metadata_sequence_size("den");
                if ((num_size > 0) != (den_size > 0)) {
                    push_error(errors_, kDiagInvalidParameter,
                               "transfer_function requires both num and den arrays for " + name);
                    continue;
                }
                if (den_size > 0) {
                    const YAML::Node den = metadata_node("den");
                    if (!den || !den.IsSequence() || den.size() == 0) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "transfer_function den must be a non-empty array for " + name);
                        continue;
                    }
                    bool den0_ok = false;
                    try {
                        den0_ok = std::abs(den[0].as<Real>()) > 1e-15;
                    } catch (...) {
                        den0_ok = false;
                    }
                    if (!den0_ok) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "transfer_function den[0] must be non-zero for " + name);
                        continue;
                    }
                }
            } else if (type == "delay_block") {
                if (has_num("delay") && get_num("delay", 0.0) < 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "delay must be >= 0 for " + name);
                    continue;
                }
            } else if (type == "sample_hold") {
                if (has_num("sample_period") && get_num("sample_period", 0.0) < 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "sample_period must be >= 0 for " + name);
                    continue;
                }
            } else if (type == "state_machine") {
                const auto mode_it = metadata.find("mode");
                if (mode_it != metadata.end()) {
                    const std::string mode = normalize_key(mode_it->second);
                    if (mode != "toggle" && mode != "level" &&
                        mode != "set_reset" && mode != "sr") {
                        push_error(errors_, kDiagInvalidParameter,
                                   "state_machine mode must be toggle/level/set_reset/sr for " + name);
                        continue;
                    }
                }
            } else if (type == "pwm_generator") {
                const Real freq = get_num("frequency", 1e3);
                if (freq <= 0.0) {
                    push_error(errors_, kDiagInvalidParameter,
                               "frequency must be > 0 for " + name);
                    continue;
                }
                if (has_num("duty")) {
                    const Real duty = get_num("duty", 0.5);
                    if (duty < 0.0 || duty > 1.0) {
                        push_error(errors_, kDiagInvalidParameter,
                                   "duty must be in [0, 1] for " + name);
                        continue;
                    }
                }
                const Real duty_min = get_num("duty_min", 0.0);
                const Real duty_max = get_num("duty_max", 1.0);
                if (duty_min < 0.0 || duty_min > 1.0 ||
                    duty_max < 0.0 || duty_max > 1.0 || duty_min > duty_max) {
                    push_error(errors_, kDiagInvalidParameter,
                               "duty_min/duty_max must satisfy 0 <= duty_min <= duty_max <= 1 for " + name);
                    continue;
                }
            } else if (type == "math_block") {
                const auto op_it = metadata.find("operation");
                if (op_it != metadata.end()) {
                    const std::string op = normalize_key(op_it->second);
                    if (op != "add" && op != "sub" && op != "mul" && op != "div") {
                        push_error(errors_, kDiagInvalidParameter,
                                   "operation must be add/sub/mul/div for " + name);
                        continue;
                    }
                }
            }

            circuit.add_virtual_component(type, name, node_indices,
                                          std::move(numeric_params), std::move(metadata));
            push_warning(warnings_, kDiagVirtualComponent,
                         "Component '" + name + "' (" + type + ") registered as virtual runtime node");
        }
        else {
            push_error(errors_, kDiagUnsupportedComponent,
                       "Unsupported component type: " + type);
        }
    }
}

}  // namespace pulsim::v1::parser
