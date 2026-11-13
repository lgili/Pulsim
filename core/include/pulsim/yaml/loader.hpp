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
//   simulation: [optional, TOP-LEVEL — a sibling of `circuit:`]
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
// Key validation (2.0.x): the only top-level keys are `circuit` and
// `simulation`, and the only `circuit:` keys are `nodes` and
// `devices`. Any other key, at those levels or inside `simulation:`,
// is not fatal: it is collected in `LoadedCircuit::warnings` (the
// Python binding re-emits each one as a `UserWarning`). REFUSED: a
// `simulation:` block indented under `circuit:` (it used to be ignored
// silently and left t_end = dt = 0) and a legacy `schema: pulsim-v1`
// netlist.
//
// See `examples/*.yaml` for working samples.

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"

#include <yaml-cpp/yaml.h>

#include <format>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pulsim::yaml {

/// Result type: a populated builder + simulation options.
struct LoadedCircuit {
    builder::CircuitBuilder    builder;
    solver::SimulationOptions  options;
    /// Where the YAML came from: the path given to `load_file`, or
    /// `"<string>"` for `load_string`. Named in error messages.
    std::string                source{"<string>"};
    /// True when the YAML has a top-level `simulation:` map. When it
    /// is false `options.t_end` / `options.dt` are 0 (not set).
    bool                       has_simulation{false};
    /// Non-fatal findings (unknown `simulation:` keys), in file order.
    std::vector<std::string>   warnings{};
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
            real_or(dev, "V_th", Real{0}),
            real_or(dev, "V_f", Real{0}));
    } else if (type == "nonlinear_diode") {
        // Retired (fix-nonlinear-diode-passivity): the law sources
        // power below V_F0. Same text as the Python refusal.
        if (!bool_or(dev, "allow_non_passive", false)) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: nonlinear_diode is retired: its "
                "smooth-blend law logistic(kappa*(v - V_F0))*(v - V_F0)/R_d "
                "is negative for 0 < v < V_F0, so the device sources power "
                "below its knee (-1.39 A out of a forward-biased anode at "
                "+0.636 V with the default parameters). Use type: diode "
                "with V_f (a PWL diode with a forward drop), or build the "
                "circuit in Python with add_shockley_diode (a smooth "
                "junction). Set allow_non_passive: true to build the "
                "legacy device anyway.",
                device_label(dev, idx, type)));
        }
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
    } else if (type == "gapped_core_inductor") {
        b.add_gapped_core_inductor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "N"),
            require_real(dev, idx, type, "Ae"),
            require_real(dev, idx, type, "le"),
            require_real(dev, idx, type, "lg"),
            real_or(dev, "mu_r0", Real{2000}),
            real_or(dev, "B_sat", Real{0.35}));
    } else if (type == "multi_winding_transformer") {
        // windings: [{from, to, L}, ...]; k: optional N×N matrix.
        if (!dev["windings"] || !dev["windings"].IsSequence()
            || dev["windings"].size() < 2) {
            throw std::runtime_error(std::format(
                "yaml::load: multi_winding_transformer '{}' needs a "
                "'windings' list of at least 2 {{from, to, L}} entries.",
                name));
        }
        std::vector<builder::CircuitBuilder::WindingSpec> ws;
        for (std::size_t k = 0; k < dev["windings"].size(); ++k) {
            const auto& w = dev["windings"][k];
            if (!w || !w.IsMap() || !w["from"] || !w["to"] || !w["L"]) {
                throw std::runtime_error(std::format(
                    "yaml::load: multi_winding_transformer '{}' winding {} "
                    "needs from, to and L.", name, k));
            }
            ws.push_back({w["from"].as<std::string>(),
                          w["to"].as<std::string>(),
                          w["L"].as<Real>()});
        }
        std::vector<std::vector<Real>> km;
        if (dev["k"] && dev["k"].IsSequence()) {
            for (std::size_t i = 0; i < dev["k"].size(); ++i) {
                std::vector<Real> row;
                if (dev["k"][i].IsSequence()) {
                    for (std::size_t j = 0; j < dev["k"][i].size(); ++j) {
                        row.push_back(dev["k"][i][j].as<Real>());
                    }
                }
                km.push_back(std::move(row));
            }
        }
        b.add_multi_winding_transformer(name, ws, km);
    } else if (type == "saturable_transformer") {
        const Real N_p = require_real(dev, idx, type, "N_p");
        const Real Ae = require_real(dev, idx, type, "Ae");
        const Real le = require_real(dev, idx, type, "le");
        const Real lg = require_real(dev, idx, type, "lg");
        const Real mu_r0 = real_or(dev, "mu_r0", Real{2000});
        const Real B_sat = real_or(dev, "B_sat", Real{0.35});
        const Real L_leak_p = real_or(dev, "L_leak_p", Real{0});
        std::vector<builder::CircuitBuilder::SecondaryWinding> sw;
        if (dev["secondaries"] && dev["secondaries"].IsSequence()) {
            // Any number of secondaries: [{s_from, s_to, N_s, L_leak_s}, …]
            for (std::size_t k = 0; k < dev["secondaries"].size(); ++k) {
                const auto& w = dev["secondaries"][k];
                if (!w || !w.IsMap() || !w["s_from"] || !w["s_to"] || !w["N_s"]) {
                    throw std::runtime_error(std::format(
                        "yaml::load: saturable_transformer '{}' secondary {} "
                        "needs s_from, s_to and N_s.", name, k));
                }
                sw.push_back({w["s_from"].as<std::string>(),
                              w["s_to"].as<std::string>(),
                              w["N_s"].as<Real>(),
                              w["L_leak_s"] ? w["L_leak_s"].as<Real>() : Real{0}});
            }
        } else {
            // The two-winding keys.
            sw.push_back({require_string(dev, idx, type, "s_from"),
                          require_string(dev, idx, type, "s_to"),
                          require_real(dev, idx, type, "N_s"),
                          real_or(dev, "L_leak_s", Real{0})});
        }
        b.add_saturable_transformer_n(
            name,
            require_string(dev, idx, type, "p_from"),
            require_string(dev, idx, type, "p_to"),
            N_p, L_leak_p, sw, Ae, le, lg, mu_r0, B_sat);
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
        // Phase 4 C.4 — Jiles-Atherton INSIDE the loop. This used to
        // build an air-core `{name}_L0` in series with a 0 V dummy
        // source `{name}_V_M` for a chain block to modulate one step
        // late — with its EMF sign inverted and unstable for every
        // shipped use. The device is now one Newton flux branch named
        // `name`; the material's five JA parameters go HERE (they
        // used to be silently ignored and re-stated in the chain
        // block), `l_gap` and `M0` are optional.
        const Real N_turns = require_real(dev, idx, type, "N_turns");
        const Real l_m     = require_real(dev, idx, type, "l_m");
        const Real A_core  = require_real(dev, idx, type, "A_core");
        models::JilesAthertonParams ja;
        ja.Ms    = require_real(dev, idx, type, "Ms");
        ja.a     = require_real(dev, idx, type, "a");
        ja.alpha = require_real(dev, idx, type, "alpha");
        ja.c     = require_real(dev, idx, type, "c");
        ja.k     = require_real(dev, idx, type, "k");
        b.add_hysteretic_core_inductor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            N_turns, A_core, l_m, real_or(dev, "l_gap", Real{0}), ja,
            /*substeps_min=*/8, real_or(dev, "M0", Real{0}));
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
            "gapped_core_inductor, transformer, multi_winding_transformer, "
            "ideal_transformer, saturable_transformer, induction_motor");
    }
}

/// Keys `load_simulation_options` reads. Anything else in the block
/// is reported (not fatal) — see `collect_unknown_simulation_keys`.
inline constexpr std::string_view kSimulationKeys[] = {
    "t_start", "t_end", "dt", "store_every",
    "enable_newton_line_search", "enable_newton_lm",
    "enable_substep_state_correction",
    "max_event_iterations", "max_newton_iterations",
    "tol_newton_dx", "tol_newton_res",
    "integrator", "rtol", "atol", "dt_init",
};

[[nodiscard]] inline bool is_known_simulation_key(std::string_view k) {
    for (const auto known : kSimulationKeys) {
        if (k == known) return true;
    }
    return false;
}

/// The v1 schema spelled some simulation keys differently; a v1 key in
/// a v2 file would otherwise read as "not set" (t_end = 0).
[[nodiscard]] inline std::string_view v2_spelling_of(std::string_view k) {
    if (k == "tstop")  return "t_end";
    if (k == "tstart") return "t_start";
    if (k == "t_stop") return "t_end";
    return {};
}

inline void collect_unknown_simulation_keys(
    const YAML::Node& sim, std::vector<std::string>& out) {
    for (const auto& kv : sim) {
        if (!kv.first.IsScalar()) continue;
        const std::string key = kv.first.as<std::string>();
        if (is_known_simulation_key(key)) continue;
        const std::string_view hint = v2_spelling_of(key);
        out.push_back(hint.empty()
            ? std::format(
                  "simulation.{} is not a v2 simulation key and was "
                  "ignored (known keys: t_start, t_end, dt, store_every, "
                  "max_newton_iterations, max_event_iterations, "
                  "tol_newton_dx, tol_newton_res, "
                  "enable_newton_line_search, enable_newton_lm, "
                  "enable_substep_state_correction, integrator, rtol, "
                  "atol, dt_init)", key)
            : std::format(
                  "simulation.{} is the v1 spelling and was ignored; "
                  "the v2 key is '{}'", key, hint));
    }
}

/// Report (in `out`, not fatal) any key of `map` that is not in
/// `allowed`. `where` names the map in the message ("top-level",
/// "circuit:"). Owner decision 2026-11-12: a warning, not an error, so a
/// user YAML carrying extra metadata (e.g. `name:`) still loads.
inline void warn_unknown_keys(
    const YAML::Node& map, std::initializer_list<std::string_view> allowed,
    std::string_view where, std::vector<std::string>& out) {
    for (const auto& kv : map) {
        if (!kv.first.IsScalar()) {
            throw std::runtime_error(std::format(
                "yaml::load: {} keys must be scalars", where));
        }
        const std::string key = kv.first.as<std::string>();
        bool ok = false;
        std::string accepted;
        for (const auto a : allowed) {
            if (key == a) ok = true;
            if (!accepted.empty()) accepted += ", ";
            accepted += a;
        }
        if (!ok) {
            out.push_back(std::format(
                "unknown {} key '{}' was ignored (accepted: {}); check "
                "for a typo or a mis-indented block", where, key,
                accepted));
        }
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

namespace detail {

/// Shared body of `load_string` / `load_file`. `source` is recorded on
/// the result and is what error messages call the file.
[[nodiscard]] inline LoadedCircuit load_text(
    const std::string& yaml_text, const std::string& source) {
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

    // A legacy v1 netlist (`schema: pulsim-v1` + `components:`) is a
    // different format, not a typo: say so before the generic
    // unknown-key refusal.
    const bool v1_schema =
        (root["schema"] && root["schema"].IsScalar() &&
         root["schema"].as<std::string>().rfind("pulsim-v1", 0) == 0) ||
        (!root["circuit"] && root["components"]);
    if (v1_schema) {
        throw std::runtime_error(
            "yaml::load_string: this is a pulsim v1 netlist "
            "('schema: pulsim-v1' / 'components:'), which the 2.0 "
            "loader does not read. A v2 netlist has 'circuit: devices:' "
            "and a top-level 'simulation:' (t_end, dt); see "
            "docs/migration-from-v1.md.");
    }

    LoadedCircuit loaded;
    loaded.source = source;
    detail::warn_unknown_keys(root, {"circuit", "simulation"},
                              "top-level", loaded.warnings);

    const YAML::Node circuit = root["circuit"];
    if (!circuit || !circuit.IsMap()) {
        throw std::runtime_error(
            "yaml::load_string: missing required top-level "
            "key 'circuit:'");
    }

    // The silent-zero trap: `simulation:` indented one level too far,
    // under `circuit:`. Refused with the fix spelled out, because the
    // run it would otherwise produce has t_end = dt = 0.
    if (circuit["simulation"]) {
        throw std::runtime_error(
            "yaml::load_string: 'simulation:' is nested under "
            "'circuit:'. It must be a TOP-LEVEL key, a sibling of "
            "'circuit:' — dedent the block. As written, t_end and dt "
            "would not be read and would load as 0.");
    }
    detail::warn_unknown_keys(circuit, {"nodes", "devices"},
                              "circuit:", loaded.warnings);

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

    // Optional simulation block. Present-but-not-a-map (an empty
    // `simulation:` reads as null) is refused rather than ignored.
    if (root["simulation"]) {
        const YAML::Node sim = root["simulation"];
        if (!sim.IsMap()) {
            throw std::runtime_error(
                "yaml::load_string: 'simulation:' must be a map "
                "(t_end: ..., dt: ...); it is empty or not a map.");
        }
        detail::collect_unknown_simulation_keys(sim, loaded.warnings);
        detail::load_simulation_options(loaded.options, sim);
        loaded.has_simulation = true;
    }

    return loaded;
}

}  // namespace detail

/// Parse a YAML circuit description from a string. Throws
/// `std::runtime_error` on validation errors or malformed
/// YAML.
[[nodiscard]] inline LoadedCircuit load_string(
    const std::string& yaml_text) {
    return detail::load_text(yaml_text, "<string>");
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
        return detail::load_text(ss.str(), path);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "yaml::load_file('" + path + "'): " + e.what());
    }
}

}  // namespace pulsim::yaml
