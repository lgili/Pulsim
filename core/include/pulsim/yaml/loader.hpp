#pragma once

// =============================================================================
// Pulsim — Layer 8: YAML circuit loader
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

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"

#include <yaml-cpp/yaml.h>

#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pulsim::yaml {

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
        return std::format("'{}'", dev["name"].as<std::string>());
    }
    return fallback_type.empty()
        ? std::format("#{}", idx)
        : std::format("#{} ({})", idx, fallback_type);
}

[[noreturn]] inline void throw_missing_field(
    const YAML::Node& dev, std::size_t idx,
    const std::string& type, const std::string& field) {
    throw std::runtime_error(std::format(
        "yaml::load: device {} is missing required field '{}'",
        device_label(dev, idx, type), field));
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

[[nodiscard]] inline std::string string_or(
    const YAML::Node& node, const std::string& field,
    const std::string& default_value) {
    if (node[field] && node[field].IsScalar()) {
        return node[field].as<std::string>();
    }
    return default_value;
}

/// Parse a single device entry; dispatches by `type`.
inline void load_device(
    builder::CircuitBuilder& b, const YAML::Node& dev,
    std::size_t idx) {
    if (!dev["type"] || !dev["type"].IsScalar()) {
        throw std::runtime_error(std::format(
            "yaml::load: device #{} is missing required field 'type'",
            idx));
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
    } else if (type == "pulse_voltage_source") {
        b.add_pulse_voltage_source(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            real_or(dev, "v_initial", Real{0}),
            require_real(dev, idx, type, "v_pulsed"),
            real_or(dev, "t_start", Real{0}),
            require_real(dev, idx, type, "pulse_width"),
            real_or(dev, "period", Real{0}),
            real_or(dev, "rise_time", Real{0}),
            real_or(dev, "fall_time", Real{0}));
    } else if (type == "mosfet_level1") {
        b.add_mosfet_level1(
            name,
            require_string(dev, idx, type, "drain"),
            require_string(dev, idx, type, "source"),
            require_string(dev, idx, type, "gate"),
            require_real(dev, idx, type, "K"),
            require_real(dev, idx, type, "V_T"),
            real_or(dev, "lambda", Real{0.02}),
            real_or(dev, "kappa",  Real{15.0}));
    } else if (type == "vcvs") {
        b.add_vcvs(
            name,
            require_string(dev, idx, type, "in_pos"),
            require_string(dev, idx, type, "in_neg"),
            require_string(dev, idx, type, "out_pos"),
            require_string(dev, idx, type, "out_neg"),
            require_real(dev, idx, type, "gain"));
    } else if (type == "op_amp_ideal") {
        b.add_op_amp_ideal(
            name,
            require_string(dev, idx, type, "in_pos"),
            require_string(dev, idx, type, "in_neg"),
            require_string(dev, idx, type, "out"),
            real_or(dev, "gain", Real{1e5}));
    } else if (type == "saturable_inductor") {
        b.add_saturable_inductor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "L_0"),
            require_real(dev, idx, type, "I_sat"),
            real_or(dev, "L_residual", Real{0}));
    } else if (type == "igbt_level1") {
        b.add_igbt_level1(
            name,
            require_string(dev, idx, type, "collector"),
            require_string(dev, idx, type, "emitter"),
            require_string(dev, idx, type, "gate"),
            real_or(dev, "V_CE_sat", Real{1.5}),
            real_or(dev, "R_CE_sat", Real{0.05}),
            real_or(dev, "V_T",      Real{5.0}),
            real_or(dev, "kappa",    Real{10.0}));
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
    } else if (type == "ideal_transformer") {
        b.add_ideal_transformer(
            name,
            require_string(dev, idx, type, "p_from"),
            require_string(dev, idx, type, "p_to"),
            require_string(dev, idx, type, "s_from"),
            require_string(dev, idx, type, "s_to"),
            require_real(dev, idx, type, "n"));
    } else if (type == "induction_motor") {
        // Phase 2.1 composite device — the YAML defines the TOPOLOGY
        // (3× per-phase Rs + σ·Ls + dummy V source). The user wires
        // the BlockChain (`chain.add_induction_motor(...)`) in Python
        // using the branch names this loader creates:
        //   {name}_Rs_{a,b,c}    — per-phase stator resistance
        //   {name}_Lsig_{a,b,c}  — per-phase stator leakage inductor
        //   {name}_E_{a,b,c}     — per-phase back-EMF dummy V source
        // J / B / T_load are accepted for documentation but NOT
        // consumed at load time (they go to the chain helper).
        if (!dev["phase_nodes"] || !dev["phase_nodes"].IsSequence() ||
            dev["phase_nodes"].size() != 3) {
            throw std::runtime_error(std::format(
                "yaml::load: induction_motor '{}' requires "
                "phase_nodes: [a, b, c] (sequence of 3 strings)",
                name));
        }
        const Real R_s = require_real(dev, idx, type, "R_s");
        const Real L_s = require_real(dev, idx, type, "L_s");
        // R_r is required (presence-validated by require_real) but its value is
        // consumed by the Python BlockChain helper, not by the topology built
        // here — mark maybe_unused so we keep the validation without warning.
        [[maybe_unused]] const Real R_r =
            require_real(dev, idx, type, "R_r");
        const Real L_r = require_real(dev, idx, type, "L_r");
        const Real L_m = require_real(dev, idx, type, "L_m");
        if (L_m * L_m >= L_s * L_r) {
            throw std::runtime_error(std::format(
                "yaml::load: induction_motor '{}' has unphysical "
                "leakage: L_m^2={} must be < L_s*L_r={}",
                name, L_m * L_m, L_s * L_r));
        }
        const Real sigma = Real{1} - (L_m * L_m) / (L_s * L_r);
        const Real L_sigma_s = sigma * L_s;
        const std::string neutral =
            require_string(dev, idx, type, "neutral_node");
        const char phase_tag[3] = {'a', 'b', 'c'};
        for (int k = 0; k < 3; ++k) {
            // Validate each element is a scalar before `.as<std::string>()`
            // (audit #9): without this, a malformed entry (e.g. a nested
            // sequence/map) throws an uncaught yaml-cpp exception instead of a
            // controlled error, matching the require_string validation style.
            if (!dev["phase_nodes"][k] ||
                !dev["phase_nodes"][k].IsScalar()) {
                throw std::runtime_error(std::format(
                    "yaml::load: induction_motor '{}' phase_nodes[{}] must be "
                    "a scalar node name", name, k));
            }
            const std::string p_node =
                dev["phase_nodes"][k].as<std::string>();
            const std::string mid_r =
                name + "_R_mid_" + phase_tag[k];
            const std::string mid_l =
                name + "_L_mid_" + phase_tag[k];
            b.add_resistor(
                name + "_Rs_" + phase_tag[k],
                p_node, mid_r, R_s);
            b.add_inductor(
                name + "_Lsig_" + phase_tag[k],
                mid_r, mid_l, L_sigma_s);
            b.add_voltage_source(
                name + "_E_" + phase_tag[k],
                mid_l, neutral, Real{0});
        }
    } else if (type == "hysteretic_inductor") {
        // Phase 2.2 composite device — JA hysteretic inductor. YAML
        // creates the linear air-core L_0 + dummy v_M source. The user
        // wires the BlockChain (`chain.add_hysteretic_inductor(...)`)
        // in Python using the branch names this loader creates:
        //   {name}_L0   — linear air-core inductor (from → mid)
        //   {name}_V_M  — dummy V source for hysteresis EMF (mid → to)
        const int N_turns =
            static_cast<int>(require_real(dev, idx, type, "N_turns"));
        const Real l_m = require_real(dev, idx, type, "l_m");
        const Real A_core = require_real(dev, idx, type, "A_core");
        if (N_turns <= 0 || l_m <= Real{0} || A_core <= Real{0}) {
            throw std::runtime_error(std::format(
                "yaml::load: hysteretic_inductor '{}' has "
                "non-physical geometry: N_turns={}, l_m={}, A_core={}",
                name, N_turns, l_m, A_core));
        }
        // L_0 = N²·A·μ₀/l_m.
        constexpr Real MU_0 = Real{4} * Real{3.14159265358979323846}
                                * Real{1e-7};
        const Real L_0 =
            static_cast<Real>(N_turns) * static_cast<Real>(N_turns)
            * A_core * MU_0 / l_m;
        const std::string from = require_string(dev, idx, type, "from");
        const std::string to = require_string(dev, idx, type, "to");
        const std::string mid = name + "_mid";
        b.add_inductor(name + "_L0", from, mid, L_0);
        b.add_voltage_source(name + "_V_M", mid, to, Real{0});
    } else {
        throw std::runtime_error(
            "yaml::load: device " + device_label(dev, idx) +
            " has unknown type '" + type + "'. Supported "
            "types: voltage_source, current_source, "
            "pwm_voltage_source, sine_voltage_source, "
            "pulse_voltage_source, resistor, capacitor, "
            "inductor, saturable_inductor, hysteretic_inductor, "
            "diode, nonlinear_diode, switch, "
            "mosfet, mosfet_with_body_diode, mosfet_level1, "
            "igbt, igbt_level1, vcvs, op_amp_ideal, "
            "transformer, ideal_transformer, induction_motor");
    }
}

inline void load_simulation_options(
    solver::SimulationOptions& opts,
    const YAML::Node& sim) {
    opts.t_start = real_or(sim, "t_start", Real{0});
    opts.t_end   = real_or(sim, "t_end",   Real{0});
    opts.dt      = real_or(sim, "dt",      Real{0});
    // v2.0 Phase 1: output decimation. Parsed here so a YAML that
    // sets it is honoured instead of silently producing a full-rate
    // trace (the loader accepts unknown keys, so an unparsed option
    // would just vanish — the exact silent-drop pattern Phase 0
    // eliminated elsewhere).
    opts.store_every = size_or(sim, "store_every", Size{1});
    if (opts.store_every == 0) {
        throw std::invalid_argument(
            "simulation.store_every must be >= 1 "
            "(1 = record every step); 0 would record nothing.");
    }
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
    // Phase 2.4 adaptive RK selector (schema v1.5, wiring v1.6).
    opts.integrator =
        string_or(sim, "integrator", std::string{"kernel"});
    opts.rtol    = real_or(sim, "rtol",    Real{1e-5});
    opts.atol    = real_or(sim, "atol",    Real{1e-8});
    opts.dt_init = real_or(sim, "dt_init", Real{0});
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

}  // namespace pulsim::yaml
