#include "pulsim/v1/parser/yaml_parser.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace pulsim::v1::parser {

namespace {

constexpr const char* kSchemaId = "pulsim-v1";

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
                        "enable_events", "enable_losses", "integrator", "integration", "newton", "timestep",
                        "lte", "bdf", "solver", "shooting", "harmonic_balance", "hb", "thermal"},
                      "simulation", errors_, options_.strict);

        if (sim["tstart"]) options.tstart = parse_real(sim["tstart"], "simulation.tstart", errors_);
        if (sim["tstop"]) options.tstop = parse_real(sim["tstop"], "simulation.tstop", errors_);
        if (sim["dt"]) options.dt = parse_real(sim["dt"], "simulation.dt", errors_);
        if (sim["dt_min"]) options.dt_min = parse_real(sim["dt_min"], "simulation.dt_min", errors_);
        if (sim["dt_max"]) options.dt_max = parse_real(sim["dt_max"], "simulation.dt_max", errors_);
        if (sim["adaptive_timestep"]) options.adaptive_timestep = sim["adaptive_timestep"].as<bool>();
        if (sim["enable_events"]) options.enable_events = sim["enable_events"].as<bool>();
        if (sim["enable_losses"]) options.enable_losses = sim["enable_losses"].as<bool>();
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

        YAML::Node integrator_node = sim["integrator"] ? sim["integrator"] : sim["integration"];
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

        if (sim["newton"]) {
            YAML::Node n = sim["newton"];
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

        if (sim["solver"]) {
            YAML::Node s = sim["solver"];
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
             "vth", "kp", "lambda", "is_nmos", "v_ce_sat",
             "turns_ratio", "ratio", "magnetizing_inductance", "lm"},
            "component", errors_, options_.strict);

        if (!comp["type"] || !comp["name"] || !comp["nodes"]) {
            errors_.push_back("Component missing type, name, or nodes");
            continue;
        }

        std::string type = to_lower(comp["type"].as<std::string>());
        std::string name = comp["name"].as<std::string>();
        std::vector<std::string> nodes = parse_nodes(comp["nodes"], name, errors_);
        if (nodes.empty()) continue;

        const YAML::Node comp_view = comp;
        const YAML::Node params = comp_view["params"];

        auto get_param = [&](const std::string& key) -> YAML::Node {
            YAML::Node top = comp_view[key];
            if (top.IsDefined() && !top.IsNull()) return top;
            YAML::Node nested = params ? params[key] : YAML::Node();
            if (nested.IsDefined() && !nested.IsNull()) return nested;
            return YAML::Node();
        };

        auto idx = [&](const std::string& node_name) {
            return circuit.add_node(node_name);
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
            YAML::Node waveform = comp["waveform"];
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
            YAML::Node initial_state_node = get_param("initial_state");
            bool closed = (initial_state_node.IsDefined() && !initial_state_node.IsNull())
                ? initial_state_node.as<bool>()
                : false;
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
            YAML::Node is_nmos_node = get_param("is_nmos");
            if (is_nmos_node.IsDefined() && !is_nmos_node.IsNull()) p.is_nmos = is_nmos_node.as<bool>();
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
