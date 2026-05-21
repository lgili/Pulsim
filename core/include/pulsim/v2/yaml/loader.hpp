#pragma once

// =============================================================================
// Pulsim v2 — Layer 8: YAML circuit loader
// =============================================================================
//
// `pulsim-v2-yaml-parser` Phase 1.
//
// Reads a YAML file (or string) describing a circuit and an
// optional `simulation:` block, returns a populated
// `CircuitBuilder` + `SimulationOptions` ready for
// `PwlStateSpaceCache` and `run_transient`.
//
// Schema (top-level):
//
//   circuit:
//     nodes: [optional, list of strings]
//     devices:
//       - type: voltage_source | resistor | capacitor |
//                inductor | diode | nonlinear_diode |
//                switch | mosfet | mosfet_with_body_diode |
//                igbt | transformer
//         name: <string, optional — used in error messages>
//         <type-specific fields, see design.md>
//
//   simulation: [optional]
//     t_start: <Real>
//     t_end: <Real>
//     dt: <Real>
//     enable_newton_line_search: <bool>
//     enable_newton_lm: <bool>
//     enable_substep_state_correction: <bool>
//     max_event_iterations: <Size>
//     max_newton_iterations: <Size>
//     tol_newton_dx: <Real>
//     tol_newton_res: <Real>
//
// See `examples/v2/` for working samples.

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/solver/options.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pulsim::v2::yaml {

/// Result type: a populated builder + simulation options.
struct LoadedCircuit {
    builder::CircuitBuilder    builder;
    solver::SimulationOptions  options;
};

namespace detail {

[[nodiscard]] inline std::string device_label(
    const YAML::Node& dev, std::size_t idx,
    const std::string& fallback_type = {}) {
    if (dev["name"] && dev["name"].IsScalar()) {
        return "'" + dev["name"].as<std::string>() + "'";
    }
    return "#" + std::to_string(idx) +
           (fallback_type.empty() ? std::string{}
            : (" (" + fallback_type + ")"));
}

[[noreturn]] inline void throw_missing_field(
    const YAML::Node& dev, std::size_t idx,
    const std::string& type, const std::string& field) {
    throw std::runtime_error(
        "yaml::load: device " + device_label(dev, idx, type) +
        " is missing required field '" + field + "'");
}

[[nodiscard]] inline Real require_real(
    const YAML::Node& dev, std::size_t idx,
    const std::string& type, const std::string& field) {
    if (!dev[field] || !dev[field].IsScalar()) {
        throw_missing_field(dev, idx, type, field);
    }
    return dev[field].as<Real>();
}

[[nodiscard]] inline std::string require_string(
    const YAML::Node& dev, std::size_t idx,
    const std::string& type, const std::string& field) {
    if (!dev[field] || !dev[field].IsScalar()) {
        throw_missing_field(dev, idx, type, field);
    }
    return dev[field].as<std::string>();
}

[[nodiscard]] inline Real real_or(
    const YAML::Node& node, const std::string& field,
    Real default_value) {
    if (node[field] && node[field].IsScalar()) {
        return node[field].as<Real>();
    }
    return default_value;
}

[[nodiscard]] inline bool bool_or(
    const YAML::Node& node, const std::string& field,
    bool default_value) {
    if (node[field] && node[field].IsScalar()) {
        return node[field].as<bool>();
    }
    return default_value;
}

[[nodiscard]] inline Size size_or(
    const YAML::Node& node, const std::string& field,
    Size default_value) {
    if (node[field] && node[field].IsScalar()) {
        return node[field].as<Size>();
    }
    return default_value;
}

/// Parse a single device entry; dispatches by `type`.
inline void load_device(
    builder::CircuitBuilder& b, const YAML::Node& dev,
    std::size_t idx) {
    if (!dev["type"] || !dev["type"].IsScalar()) {
        throw std::runtime_error(
            "yaml::load: device #" + std::to_string(idx) +
            " is missing required field 'type'");
    }
    const std::string type = dev["type"].as<std::string>();
    const std::string name = dev["name"] && dev["name"].IsScalar()
        ? dev["name"].as<std::string>()
        : std::string{"<unnamed>"};

    if (type == "voltage_source") {
        b.add_voltage_source(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "V"));
    } else if (type == "current_source") {
        b.add_current_source(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "I"));
    } else if (type == "pwm_voltage_source") {
        b.add_pwm_voltage_source(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "v_high"),
            require_real(dev, idx, type, "v_low"),
            require_real(dev, idx, type, "frequency"),
            require_real(dev, idx, type, "duty"),
            real_or(dev, "phase", Real{0}));
    } else if (type == "sine_voltage_source") {
        b.add_sine_voltage_source(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            real_or(dev, "v_dc", Real{0}),
            require_real(dev, idx, type, "v_amplitude"),
            require_real(dev, idx, type, "frequency"),
            real_or(dev, "phase", Real{0}));
    } else if (type == "resistor") {
        b.add_resistor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "R"));
    } else if (type == "capacitor") {
        b.add_capacitor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "C"));
    } else if (type == "inductor") {
        b.add_inductor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "L"));
    } else if (type == "diode") {
        b.add_diode(
            name,
            require_string(dev, idx, type, "anode"),
            require_string(dev, idx, type, "cathode"),
            require_real(dev, idx, type, "g_on"),
            require_real(dev, idx, type, "g_off"),
            real_or(dev, "V_th", Real{0}));
    } else if (type == "nonlinear_diode") {
        models::IdealDiode::Params p{
            .V_F0  = real_or(dev, "V_F0", Real{0.7}),
            .R_d   = real_or(dev, "R_d",  Real{0.01}),
            .G_off = real_or(dev, "G_off", Real{1e-9}),
            .kappa = real_or(dev, "kappa", Real{20}),
        };
        b.add_nonlinear_diode(
            name,
            require_string(dev, idx, type, "anode"),
            require_string(dev, idx, type, "cathode"),
            p);
    } else if (type == "switch") {
        b.add_switch(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "g_on"),
            require_real(dev, idx, type, "g_off"));
    } else if (type == "mosfet") {
        b.add_mosfet(
            name,
            require_string(dev, idx, type, "drain"),
            require_string(dev, idx, type, "source"),
            real_or(dev, "R_on",  Real{1e-3}),
            real_or(dev, "R_off", Real{1e9}));
    } else if (type == "mosfet_with_body_diode") {
        b.add_mosfet_with_body_diode(
            name,
            require_string(dev, idx, type, "drain"),
            require_string(dev, idx, type, "source"),
            real_or(dev, "R_on",         Real{1e-3}),
            real_or(dev, "R_off",        Real{1e9}),
            real_or(dev, "V_F",          Real{0.7}),
            real_or(dev, "g_on_diode",   Real{1e3}),
            real_or(dev, "g_off_diode",  Real{1e-9}));
    } else if (type == "igbt") {
        b.add_igbt(
            name,
            require_string(dev, idx, type, "collector"),
            require_string(dev, idx, type, "emitter"),
            real_or(dev, "R_on",  Real{10e-3}),
            real_or(dev, "R_off", Real{1e9}));
    } else if (type == "transformer") {
        b.add_transformer(
            name,
            require_string(dev, idx, type, "p_from"),
            require_string(dev, idx, type, "p_to"),
            require_string(dev, idx, type, "s_from"),
            require_string(dev, idx, type, "s_to"),
            require_real(dev, idx, type, "L_p"),
            require_real(dev, idx, type, "L_s"),
            real_or(dev, "k", Real{1}));
    } else {
        throw std::runtime_error(
            "yaml::load: device " + device_label(dev, idx) +
            " has unknown type '" + type + "'. Supported "
            "types: voltage_source, current_source, "
            "pwm_voltage_source, sine_voltage_source, "
            "resistor, capacitor, inductor, diode, "
            "nonlinear_diode, switch, mosfet, "
            "mosfet_with_body_diode, igbt, transformer");
    }
}

inline void load_simulation_options(
    solver::SimulationOptions& opts,
    const YAML::Node& sim) {
    opts.t_start = real_or(sim, "t_start", Real{0});
    opts.t_end   = real_or(sim, "t_end",   Real{0});
    opts.dt      = real_or(sim, "dt",      Real{0});
    opts.enable_newton_line_search =
        bool_or(sim, "enable_newton_line_search", false);
    opts.enable_newton_lm =
        bool_or(sim, "enable_newton_lm", false);
    opts.enable_substep_state_correction =
        bool_or(sim, "enable_substep_state_correction",
                 false);
    opts.max_event_iterations =
        size_or(sim, "max_event_iterations", Size{16});
    opts.max_newton_iterations =
        size_or(sim, "max_newton_iterations", Size{50});
    opts.tol_newton_dx =
        real_or(sim, "tol_newton_dx", Real{1e-9});
    opts.tol_newton_res =
        real_or(sim, "tol_newton_res", Real{1e-9});
}

}  // namespace detail

/// Parse a YAML circuit description from a string. Throws
/// `std::runtime_error` on validation errors or malformed
/// YAML.
[[nodiscard]] inline LoadedCircuit load_string(
    const std::string& yaml_text) {
    YAML::Node root;
    try {
        root = YAML::Load(yaml_text);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string{"yaml::load_string: malformed YAML — "} +
            e.what());
    }

    if (!root.IsMap()) {
        throw std::runtime_error(
            "yaml::load_string: top-level YAML must be a map "
            "with 'circuit:' and optional 'simulation:'");
    }

    LoadedCircuit loaded;

    const YAML::Node circuit = root["circuit"];
    if (!circuit || !circuit.IsMap()) {
        throw std::runtime_error(
            "yaml::load_string: missing required top-level "
            "key 'circuit:'");
    }

    // Optional: pre-declare nodes (auto-created on device
    // reference anyway, but exposing it gives the user
    // explicit control of insertion order).
    if (circuit["nodes"] && circuit["nodes"].IsSequence()) {
        for (const auto& node_name : circuit["nodes"]) {
            if (node_name.IsScalar()) {
                loaded.builder.node(
                    node_name.as<std::string>());
            }
        }
    }

    const YAML::Node devices = circuit["devices"];
    if (!devices || !devices.IsSequence()) {
        throw std::runtime_error(
            "yaml::load_string: missing required key "
            "'circuit.devices:' (must be a sequence)");
    }

    std::size_t idx = 0;
    for (const auto& dev : devices) {
        detail::load_device(loaded.builder, dev, idx);
        ++idx;
    }

    // Optional simulation block.
    if (root["simulation"] && root["simulation"].IsMap()) {
        detail::load_simulation_options(
            loaded.options, root["simulation"]);
    }

    return loaded;
}

/// Parse a YAML circuit description from a file on disk.
/// Wraps any I/O or parse errors with the file path.
[[nodiscard]] inline LoadedCircuit load_file(
    const std::string& path) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error(
            "yaml::load_file: could not open '" + path + "'");
    }
    std::stringstream ss;
    ss << stream.rdbuf();
    try {
        return load_string(ss.str());
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "yaml::load_file('" + path + "'): " + e.what());
    }
}

}  // namespace pulsim::v2::yaml
