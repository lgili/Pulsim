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

#include "pulsim/ir/expr.hpp"
#include "pulsim/yaml/emitter.hpp"

#include <array>
#include <charconv>
#include <filesystem>
#include <cmath>
#include <numbers>
#include <format>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::yaml {

/// update-number-changing-release: the `form` of a `switch` / `diode` that
/// names none (the branch form; `form: conductance` builds the pre-release
/// element). MOSFET / IGBT types keep their own default.
inline constexpr const char* kDefaultSwitchForm = "branch";

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
    /// The known `simulation:` keys the YAML sets, in file order:
    /// `simulate(loaded)` forwards exactly these (a key set to its
    /// default value still wins over an engine's own default).
    std::vector<std::string>   simulation_keys{};
};

/// add-external-device-abi 1.7: what a netlist may load. A YAML
/// `external_device` names a shared library, i.e. native code: it is loaded
/// only when its path (symlinks followed) is one of `allow_native_code` or
/// lies inside one of its directories. Empty: every external device is
/// refused by name. A netlist from an untrusted source cannot choose the
/// library — the caller lists what may run.
struct LoadOptions {
    std::vector<std::string> allow_native_code;
    /// Where a relative `lib:` is taken from (the YAML file's folder for
    /// `load_file`; the working directory for `load_string`).
    std::string base_dir;
    /// The `form` of a `switch` / `diode` that names none: "branch" (the
    /// default since the number-changing release) or "conductance" (the
    /// pre-release netlist meaning, for the legacy period).
    std::string default_switch_form{kDefaultSwitchForm};
};

namespace detail {

/// The options of the load in progress on this thread (load_text's scope):
/// reached by `load_device` at any nesting (subsystems) without threading
/// them through every loader signature.
inline thread_local const LoadOptions* tl_load_options = nullptr;

/// The form a `switch` / `diode` without `form:` builds (LoadOptions).
[[nodiscard]] inline std::string default_switch_form() {
    return tl_load_options ? tl_load_options->default_switch_form
                           : std::string{kDefaultSwitchForm};
}

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

/// add-controlled-source-set: the new types are strict — a key outside
/// `allowed` (plus `type` / `name`) is an error naming the device.
inline void require_only_keys(const YAML::Node& dev, std::size_t idx,
                              const std::string& type,
                              std::initializer_list<std::string_view> allowed) {
    for (const auto& kv : dev) {
        const std::string key = kv.first.as<std::string>();
        bool ok = key == "type" || key == "name";
        for (const auto a : allowed) ok = ok || key == a;
        if (!ok) {
            std::string accepted;
            for (const auto a : allowed) {
                accepted += accepted.empty() ? "" : ", ";
                accepted += a;
            }
            throw std::runtime_error(std::format(
                "yaml::load: device {} has unknown key '{}' (accepted: "
                "type, name, {})", device_label(dev, idx, type), key,
                accepted));
        }
    }
}

[[nodiscard]] inline std::vector<Real> require_real_list(
    const YAML::Node& dev, std::size_t idx, const std::string& type,
    const std::string& field) {
    if (!dev[field] || !dev[field].IsSequence()) {
        throw std::runtime_error(std::format(
            "yaml::load: device {} needs '{}' as a list of numbers",
            device_label(dev, idx, type), field));
    }
    return dev[field].as<std::vector<Real>>();
}

/// A two-column table (t, i) from a text file: comma or whitespace
/// separated, blank lines and lines starting with '#' skipped. A
/// relative path is taken from the working directory.
inline void read_table_file(const std::string& path, const std::string& who,
                            std::vector<Real>& t, std::vector<Real>& v) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(std::format(
            "yaml::load: device {}: cannot open table file '{}'", who, path));
    }
    std::string line;
    for (std::size_t n = 1; std::getline(in, line); ++n) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        for (char& c : line) {
            if (c == ',' || c == ';') c = ' ';
        }
        std::istringstream row(line);
        Real a = 0, b = 0;
        std::string rest;
        if (!(row >> a >> b) || (row >> rest)) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: line {} of '{}' is not two numbers",
                who, n, path));
        }
        t.push_back(a);
        v.push_back(b);
    }
}

/// A non-negative finite parameter (default 0); refused by name
/// otherwise. Used by the parasitic keys of `capacitor` / `inductor`
/// and by `dc_machine` (add-realistic-passives, add-mna-native-machines).
[[nodiscard]] inline Real nonneg_or_zero(
    const YAML::Node& node, const std::string& type,
    const std::string& name, const std::string& field) {
    const Real v = real_or(node, field, Real{0});
    if (!std::isfinite(v) || v < Real{0}) {
        throw std::runtime_error(std::format(
            "yaml::load: {} '{}': {} must be finite and >= 0 (got {})",
            type, name, field, v));
    }
    return v;
}

/// Refuses a key of `node` that is not in `keys`, naming the device.
inline void strict_keys(const YAML::Node& dev, std::size_t idx, const YAML::Node& node,
                        const std::string& where,
                        std::initializer_list<std::string_view> keys) {
    for (const auto& kv : node) {
        const auto k = kv.first.as<std::string>();
        bool ok = false;
        for (auto a : keys) ok = ok || k == a;
        if (!ok) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: unknown key '{}' in {}",
                device_label(dev, idx), k, where));
        }
    }
}

/// A charge-law map `{law: junction | vdmos_cgd | table, ...}` (power_mosfet
/// C_gd / C_ds, nonlinear_capacitor law). `what` names the table in errors.
[[nodiscard]] inline models::NonlinearCapacitor::Params charge_law_node(
    const YAML::Node& dev, std::size_t idx, const YAML::Node& n,
    const std::string& what, const std::string& where) {
    auto reals = [&](const std::string& f) {
        if (!n[f] || !n[f].IsSequence()) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: '{}' must be a list of numbers",
                device_label(dev, idx), f));
        }
        return n[f].as<std::vector<Real>>();
    };
    const std::string law = n["law"] ? n["law"].as<std::string>() : "";
    if (law == "junction") {
        strict_keys(dev, idx, n, where, {"law", "C0", "V0", "m", "v_floor"});
        return models::NonlinearCapacitor::junction(
            require_real(n, idx, where, "C0"), require_real(n, idx, where, "V0"),
            real_or(n, "m", Real{0.5}), real_or(n, "v_floor", Real{-0.9}));
    }
    if (law == "vdmos_cgd") {
        strict_keys(dev, idx, n, where, {"law", "cgd_max", "cgd_min", "a"});
        return models::NonlinearCapacitor::vdmos_cgd(
            require_real(n, idx, where, "cgd_max"), require_real(n, idx, where, "cgd_min"),
            require_real(n, idx, where, "a"));
    }
    if (law == "table") {
        strict_keys(dev, idx, n, where, {"law", "v", "Q", "C"});
        auto t = n["Q"]
            ? models::ChargeTable(reals("v"), reals("Q"), reals("C"), what)
            : models::ChargeTable::from_capacitance(reals("v"), reals("C"), what);
        return models::NonlinearCapacitor::from_table(
            std::make_shared<const models::ChargeTable>(std::move(t)));
    }
    throw std::runtime_error(std::format(
        "yaml::load: device {}: {} must be a number (linear, F) or a "
        "map with law: junction | vdmos_cgd | table (got law '{}')",
        device_label(dev, idx), where, law));
}

/// Parse a single device entry; dispatches by `type`.
/// add-in-kernel-event-losses 3.5: a `losses:` block on a switch-class
/// device. Strict keys; tables as nested lists:
///   e_on / e_off: {v: [...], i: [...], tj: [...], energy: [[[...]]]}
///   v_on:         {i: [...], tj: [...], v: [[...]]}
inline void load_losses(builder::CircuitBuilder& b, const YAML::Node& dev,
                        std::size_t idx, const std::string& type,
                        const std::string& name) {
    const std::string who = device_label(dev, idx, type);
    if (type != "switch" && type != "diode" && type != "mosfet" &&
        type != "mosfet_with_body_diode" && type != "igbt") {
        throw std::runtime_error(std::format(
            "yaml::load: device {}: a losses: block is for switch, diode, "
            "mosfet, mosfet_with_body_diode and igbt", who));
    }
    const YAML::Node l = dev["losses"];
    if (!l.IsMap()) {
        throw std::runtime_error(std::format(
            "yaml::load: device {}: losses: must be a mapping", who));
    }
    for (const auto& kv : l) {
        const std::string key = kv.first.as<std::string>();
        if (key != "junction" && key != "Tj" && key != "heat_node" &&
            key != "e_on" && key != "e_off" && key != "v_on" &&
            key != "conduction") {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: unknown key '{}' in losses (accepted: "
                "junction, Tj, heat_node, e_on, e_off, v_on, conduction)",
                who, key));
        }
    }
    // a table's own keys are strict too: a misspelt axis is not "absent"
    auto keys = [&](const YAML::Node& t, const char* table,
                    std::initializer_list<std::string_view> ok) {
        if (!t.IsMap()) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: losses.{} must be a mapping", who,
                table));
        }
        for (const auto& kv : t) {
            const std::string k = kv.first.as<std::string>();
            if (std::find(ok.begin(), ok.end(), k) == ok.end()) {
                throw std::runtime_error(std::format(
                    "yaml::load: device {}: unknown key '{}' in losses.{}",
                    who, k, table));
            }
        }
    };
    auto list = [&](const YAML::Node& t, const char* table, const char* f) {
        if (!t[f] || !t[f].IsSequence()) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: losses.{} needs '{}' as a list", who,
                table, f));
        }
        return t[f];
    };
    auto flat = [](const YAML::Node& n, std::vector<Real>& out, auto& self) -> void {
        if (n.IsSequence()) {
            for (const auto& c : n) self(c, out, self);
        } else {
            out.push_back(n.as<Real>());
        }
    };
    auto e_table = [&](const char* key) -> std::shared_ptr<const thermal::LossTable3D> {
        if (!l[key]) return nullptr;
        const YAML::Node t = l[key];
        keys(t, key, {"v", "i", "tj", "energy"});
        std::vector<Real> e;
        flat(list(t, key, "energy"), e, flat);
        try {
            return std::make_shared<const thermal::LossTable3D>(
                list(t, key, "v").as<std::vector<Real>>(),
                list(t, key, "i").as<std::vector<Real>>(),
                list(t, key, "tj").as<std::vector<Real>>(), std::move(e));
        } catch (const std::invalid_argument& ex) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: losses.{}: {}", who, key, ex.what()));
        }
    };
    thermal::DeviceLossSpec s;
    s.device = name;
    s.junction = string_or(l, "junction", "");
    s.heat_node = string_or(l, "heat_node", "");
    s.tj = l["Tj"] ? l["Tj"].as<Real>() : std::nan("");
    s.e_on = e_table("e_on");
    s.e_off = e_table("e_off");
    if (l["v_on"]) {
        const YAML::Node t = l["v_on"];
        keys(t, "v_on", {"i", "tj", "v"});
        std::vector<Real> v;
        flat(list(t, "v_on", "v"), v, flat);
        try {
            s.v_on = std::make_shared<const thermal::ConductionTable>(
                list(t, "v_on", "i").as<std::vector<Real>>(),
                list(t, "v_on", "tj").as<std::vector<Real>>(), std::move(v));
        } catch (const std::invalid_argument& ex) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: losses.v_on: {}", who, ex.what()));
        }
    }
    const std::string cond = string_or(l, "conduction", "electrical");
    if (cond != "electrical" && cond != "table") {
        throw std::runtime_error(std::format(
            "yaml::load: device {}: losses.conduction must be electrical or "
            "table, not '{}'", who, cond));
    }
    s.conduction = cond == "table" ? thermal::ConductionHeat::Table
                                   : thermal::ConductionHeat::Electrical;
    try {
        b.set_device_losses(std::move(s));
    } catch (const std::invalid_argument& ex) {
        throw std::runtime_error(std::format("yaml::load: {}", ex.what()));
    }
}

/// `type: external_device` — {name, nodes, lib, model, params}; params by
/// the model's field names, a number or [value, unit] (strict: an unknown
/// field is refused by the model's field table).
inline void load_external_device(builder::CircuitBuilder& b, const YAML::Node& dev,
                                 std::size_t idx) {
    namespace fs = std::filesystem;
    const std::string who = device_label(dev, idx, "external_device");
    require_only_keys(dev, idx, "external_device", {"nodes", "lib", "model", "params"});
    const LoadOptions* opts = tl_load_options;
    if (!opts || opts->allow_native_code.empty()) {
        throw std::runtime_error(std::format(
            "yaml::load: device {} loads native code (a shared library runs with your "
            "privileges); refused unless the load call lists the library: pass "
            "allow_native_code=[<library path or directory>, ...]", who));
    }
    if (!dev["nodes"] || !dev["nodes"].IsSequence()) {
        throw std::runtime_error(std::format("yaml::load: device {} needs 'nodes' as a list "
                                             "of node names, in the model's pin order", who));
    }
    const auto nodes = dev["nodes"].as<std::vector<std::string>>();
    const std::string lib = require_string(dev, idx, "external_device", "lib");
    const std::string model = require_string(dev, idx, "external_device", "model");
    auto u8 = [](const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); };
    fs::path lp = u8(lib);
    if (lp.is_relative()) lp = u8(opts->base_dir.empty() ? "." : opts->base_dir) / lp;
    std::error_code ec;
    const fs::path canon = fs::canonical(lp, ec);
    if (ec) {
        throw std::runtime_error(std::format("yaml::load: device {}: library '{}' not found ({})",
                                             who, lib, ec.message()));
    }
    bool allowed = false;
    for (const auto& a : opts->allow_native_code) {
        std::error_code e2;
        const fs::path ca = fs::canonical(u8(a), e2);
        if (e2) continue;   // an entry that does not exist allows nothing
        if (ca == canon) allowed = true;
        if (fs::is_directory(ca, e2)) {
            const fs::path rel = canon.lexically_relative(ca);
            allowed = allowed || (!rel.empty() && *rel.begin() != "..");
        }
    }
    const auto u8s = canon.u8string();
    const std::string canon_s(u8s.begin(), u8s.end());
    if (!allowed) {
        throw std::runtime_error(std::format(
            "yaml::load: device {}: library '{}' (resolved: '{}') is not in "
            "allow_native_code; refused", who, lib, canon_s));
    }
    auto m = xdev::load_model(canon_s, model);
    std::vector<xdev::NamedParam> given;
    if (dev["params"]) {
        if (!dev["params"].IsMap()) {
            throw std::runtime_error(std::format("yaml::load: device {}: 'params' must be a map "
                                                 "of field name to a number or [value, unit]", who));
        }
        for (const auto& kv : dev["params"]) {
            xdev::NamedParam np;
            np.name = kv.first.as<std::string>();
            const YAML::Node v = kv.second;
            try {
                if (v.IsSequence() && v.size() == 2) {
                    np.value = v[0].as<Real>();
                    np.unit = v[1].as<std::string>();
                    np.has_unit = true;
                } else {
                    np.value = v.as<Real>();
                }
            } catch (const YAML::Exception&) {
                throw std::runtime_error(std::format("yaml::load: device {}: parameter '{}' must "
                                                     "be a number or [value, unit]", who, np.name));
            }
            given.push_back(std::move(np));
        }
    }
    const std::string name = dev["name"].as<std::string>();
    auto prm = xdev::resolve_params(*m, name, given);
    b.add_external_device(name, nodes, std::move(m), std::move(prm));
}

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
    if (type == "external_device") {
        if (!dev["name"] || !dev["name"].IsScalar()) {
            throw std::runtime_error(std::format(
                "yaml::load: device #{} (external_device) needs a 'name'", idx));
        }
        load_external_device(b, dev, idx);
        return;
    }

    // add-branch-form-ideal-switch: `form` is conductance | branch, on
    // every type that builds a PWL switch or ungated diode (a typo must
    // not fall back silently).
    const std::string form = string_or(dev, "form", "conductance");
    if (dev["form"]) {
        if ((form != "conductance" && form != "branch") ||
            (type != "switch" && type != "diode" && type != "mosfet" &&
             type != "mosfet_with_body_diode" && type != "igbt" &&
             type != "mosfet_level1" && type != "igbt_level1")) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: form: '{}' is not supported; use "
                "form: conductance or form: branch on a switch, diode, "
                "mosfet, mosfet_with_body_diode, igbt, mosfet_level1 or "
                "igbt_level1.",
                device_label(dev, idx, type), form));
        }
    }

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
            real_or(dev, "kappa",  Real{15.0}),
            bool_or(dev, "with_body_diode", true),
            real_or(dev, "V_F",    Real{0.7}), form);
    } else if (type == "vcvs") {
        b.add_vcvs(
            name,
            require_string(dev, idx, type, "in_pos"),
            require_string(dev, idx, type, "in_neg"),
            require_string(dev, idx, type, "out_pos"),
            require_string(dev, idx, type, "out_neg"),
            require_real(dev, idx, type, "gain"));
    } else if (type == "vccs") {
        require_only_keys(dev, idx, type,
                          {"in_pos", "in_neg", "from", "to", "gm"});
        b.add_vccs(name, require_string(dev, idx, type, "in_pos"),
                   require_string(dev, idx, type, "in_neg"),
                   require_string(dev, idx, type, "from"),
                   require_string(dev, idx, type, "to"),
                   require_real(dev, idx, type, "gm"));
    } else if (type == "ccvs") {
        require_only_keys(dev, idx, type,
                          {"control", "out_pos", "out_neg", "rm"});
        b.add_ccvs(name, require_string(dev, idx, type, "control"),
                   require_string(dev, idx, type, "out_pos"),
                   require_string(dev, idx, type, "out_neg"),
                   require_real(dev, idx, type, "rm"));
    } else if (type == "cccs") {
        require_only_keys(dev, idx, type, {"control", "from", "to", "gain"});
        b.add_cccs(name, require_string(dev, idx, type, "control"),
                   require_string(dev, idx, type, "from"),
                   require_string(dev, idx, type, "to"),
                   require_real(dev, idx, type, "gain"));
    } else if (type == "gyrator") {
        require_only_keys(dev, idx, type,
                          {"p1_pos", "p1_neg", "p2_pos", "p2_neg", "g"});
        b.add_gyrator(name, require_string(dev, idx, type, "p1_pos"),
                      require_string(dev, idx, type, "p1_neg"),
                      require_string(dev, idx, type, "p2_pos"),
                      require_string(dev, idx, type, "p2_neg"),
                      require_real(dev, idx, type, "g"));
    } else if (type == "winding") {  // add-permeance-magnetic-domain
        require_only_keys(dev, idx, type,
                          {"e_pos", "e_neg", "m_pos", "m_neg", "turns"});
        b.add_winding(name, require_string(dev, idx, type, "e_pos"),
                      require_string(dev, idx, type, "e_neg"),
                      require_string(dev, idx, type, "m_pos"),
                      require_string(dev, idx, type, "m_neg"),
                      require_real(dev, idx, type, "turns"));
    } else if (type == "permeance") {
        require_only_keys(dev, idx, type, {"from", "to", "P"});
        b.add_permeance(name, require_string(dev, idx, type, "from"),
                        require_string(dev, idx, type, "to"),
                        require_real(dev, idx, type, "P"));
    } else if (type == "air_gap") {
        require_only_keys(dev, idx, type, {"from", "to", "area", "length"});
        b.add_air_gap(name, require_string(dev, idx, type, "from"),
                      require_string(dev, idx, type, "to"),
                      require_real(dev, idx, type, "area"),
                      require_real(dev, idx, type, "length"));
    } else if (type == "magnetic_resistance") {
        require_only_keys(dev, idx, type, {"from", "to", "R_m"});
        b.add_magnetic_resistance(name, require_string(dev, idx, type, "from"),
                                  require_string(dev, idx, type, "to"),
                                  require_real(dev, idx, type, "R_m"));
    } else if (type == "saturable_permeance") {
        require_only_keys(dev, idx, type,
                          {"from", "to", "area", "length", "mu_r", "B_sat"});
        b.add_saturable_permeance(name, require_string(dev, idx, type, "from"),
                                  require_string(dev, idx, type, "to"),
                                  require_real(dev, idx, type, "area"),
                                  require_real(dev, idx, type, "length"),
                                  require_real(dev, idx, type, "mu_r"),
                                  require_real(dev, idx, type, "B_sat"));
    } else if (type == "hysteretic_permeance") {
        require_only_keys(dev, idx, type, {"from", "to", "area", "length", "Ms",
                                           "a", "alpha", "c", "k", "M0"});
        models::JilesAthertonParams ja;
        ja.Ms    = require_real(dev, idx, type, "Ms");
        ja.a     = require_real(dev, idx, type, "a");
        ja.alpha = require_real(dev, idx, type, "alpha");
        ja.c     = require_real(dev, idx, type, "c");
        ja.k     = require_real(dev, idx, type, "k");
        b.add_hysteretic_permeance(name, require_string(dev, idx, type, "from"),
                                   require_string(dev, idx, type, "to"),
                                   require_real(dev, idx, type, "area"),
                                   require_real(dev, idx, type, "length"), ja,
                                   /*substeps_min=*/8, real_or(dev, "M0", Real{0}));
    } else if (type == "transmission_line") {
        require_only_keys(dev, idx, type,
                          {"a_pos", "a_neg", "b_pos", "b_neg", "Z0", "delay", "R"});
        b.add_transmission_line(name, require_string(dev, idx, type, "a_pos"),
                                require_string(dev, idx, type, "a_neg"),
                                require_string(dev, idx, type, "b_pos"),
                                require_string(dev, idx, type, "b_neg"),
                                require_real(dev, idx, type, "Z0"),
                                require_real(dev, idx, type, "delay"),
                                real_or(dev, "R", Real{0}));
    } else if (type == "excess_loss") {
        require_only_keys(dev, idx, type, {"from", "to", "c", "eps"});
        b.add_excess_loss(name, require_string(dev, idx, type, "from"),
                          require_string(dev, idx, type, "to"),
                          require_real(dev, idx, type, "c"),
                          require_real(dev, idx, type, "eps"));
    } else if (type == "sine_current_source") {
        require_only_keys(dev, idx, type, {"from", "to", "i_dc", "i_amplitude",
                                           "frequency", "phase"});
        b.add_sine_current_source(
            name, require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "i_dc"),
            require_real(dev, idx, type, "i_amplitude"),
            require_real(dev, idx, type, "frequency"),
            real_or(dev, "phase", Real{0}));
    } else if (type == "pulse_current_source") {
        require_only_keys(dev, idx, type,
                          {"from", "to", "i_initial", "i_pulsed", "t_start",
                           "pulse_width", "period", "rise_time", "fall_time"});
        b.add_pulse_current_source(
            name, require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "i_initial"),
            require_real(dev, idx, type, "i_pulsed"),
            require_real(dev, idx, type, "t_start"),
            require_real(dev, idx, type, "pulse_width"),
            real_or(dev, "period", Real{0}),
            real_or(dev, "rise_time", Real{0}),
            real_or(dev, "fall_time", Real{0}));
    } else if (type == "pwl_current_source") {
        require_only_keys(dev, idx, type,
                          {"from", "to", "t", "i", "file", "periodic"});
        std::vector<Real> t, v;
        if (dev["file"]) {
            if (dev["t"] || dev["i"]) {
                throw std::runtime_error(std::format(
                    "yaml::load: device {} gives both 'file' and 't'/'i'; "
                    "use one", device_label(dev, idx, type)));
            }
            read_table_file(require_string(dev, idx, type, "file"),
                            device_label(dev, idx, type), t, v);
        } else {
            t = require_real_list(dev, idx, type, "t");
            v = require_real_list(dev, idx, type, "i");
        }
        b.add_pwl_current_source(name, require_string(dev, idx, type, "from"),
                                 require_string(dev, idx, type, "to"),
                                 std::move(t), std::move(v),
                                 bool_or(dev, "periodic", false));
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
            real_or(dev, "kappa",    Real{10.0}),
            real_or(dev, "v_knee",   Real{0.01}),
            bool_or(dev, "with_fwd", false),
            Real{0}, Real{0},
            real_or(dev, "V_F",      Real{0.7}), form);
    } else if (type == "resistor") {
        b.add_resistor(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "R"));
    } else if (type == "capacitor") {
        // esr / esl: series composite, the same expansion as the
        // Python keywords (python/pulsim/passives.py); zero adds nothing.
        const Real esr = nonneg_or_zero(dev, type, name, "esr");
        const Real esl = nonneg_or_zero(dev, type, name, "esl");
        std::string from = require_string(dev, idx, type, "from");
        if (esr > Real{0}) {
            b.add_resistor(name + ".esr", from, name + ".esr_n", esr);
            from = name + ".esr_n";
        }
        if (esl > Real{0}) {
            b.add_inductor(name + ".esl", from, name + ".esl_n", esl);
            from = name + ".esl_n";
        }
        b.add_capacitor(
            name,
            from,
            require_string(dev, idx, type, "to"),
            require_real(dev, idx, type, "C"));
    } else if (type == "inductor") {
        Real esr = nonneg_or_zero(dev, type, name, "esr");
        const Real c_par = nonneg_or_zero(dev, type, name, "c_par");
        const std::string from = require_string(dev, idx, type, "from");
        const std::string to = require_string(dev, idx, type, "to");
        Real l_main = require_real(dev, idx, type, "L");
        // rac: an R_ac ladder given by its cells (the expansion of
        // python/pulsim/passives.py add_inductor(rac=)). Fitting an
        // (f, R_ac) table needs Python's least squares: refused here.
        std::vector<Real> rk, lk;
        if (const YAML::Node rac = dev["rac"]) {
            const std::string who = std::format("yaml::load: inductor '{}': rac", name);
            if (!rac.IsMap() || rac["f"] || rac["R_ac"]) {
                throw std::runtime_error(who +
                    " must be a map {R_dc, L_hf, R: [...], L: [...]} of ladder "
                    "cells; fit an (f, R_ac) curve in Python "
                    "(pulsim.passives.fit_rac_ladder) and write its cells.");
            }
            strict_keys(dev, idx, rac, "rac", {"R_dc", "L_hf", "R", "L"});
            if (esr > Real{0}) {
                throw std::runtime_error(who + ": give esr or rac, not both "
                                         "(the ladder's R_dc is the series resistance)");
            }
            esr = nonneg_or_zero(rac, type + ".rac", name, "R_dc");
            const Real l_hf = require_real(rac, idx, type + ".rac", "L_hf");
            rk = require_real_list(rac, idx, type + ".rac", "R");
            lk = require_real_list(rac, idx, type + ".rac", "L");
            Real l_tot = l_hf;
            bool ok = std::isfinite(l_hf) && l_hf > Real{0} && rk.size() == lk.size();
            for (std::size_t k = 0; ok && k < rk.size(); ++k) {
                ok = std::isfinite(rk[k]) && rk[k] > Real{0} &&
                     std::isfinite(lk[k]) && lk[k] > Real{0};
                l_tot += ok ? lk[k] : Real{0};
            }
            if (!ok) {
                throw std::runtime_error(who + ": L_hf and every cell R, L must be "
                                         "finite and > 0, with as many R as L");
            }
            if (std::abs(l_tot - l_main) > Real{1e-9} * std::abs(l_main)) {
                throw std::runtime_error(std::format(
                    "{}: L_hf + sum(L) = {:.9g} H differs from L = {:.9g} H",
                    who, l_tot, l_main));
            }
            l_main = l_hf;
        }
        std::string a = from;
        if (esr > Real{0}) {
            b.add_resistor(name + ".esr", a, name + ".esr_n", esr);
            a = name + ".esr_n";
        }
        const std::size_t cells = rk.size();
        std::string nxt = cells ? name + ".rac_n0" : to;
        b.add_inductor(name, a, nxt, l_main);
        for (std::size_t k = 0; k < cells; ++k) {
            a = nxt;
            nxt = k + 1 < cells ? std::format("{}.rac_n{}", name, k + 1) : to;
            b.add_resistor(std::format("{}.rac_r{}", name, k), a, nxt, rk[k]);
            b.add_inductor(std::format("{}.rac_l{}", name, k), a, nxt, lk[k]);
        }
        if (c_par > Real{0}) {
            b.add_capacitor(name + ".cpar", from, to, c_par);
        }
    } else if (type == "dc_machine") {
        // add-mna-native-machines: the composite of
        // CircuitBuilder.add_dc_machine (python/pulsim/machines.py).
        const Real R_a = nonneg_or_zero(dev, type, name, "R_a");
        const Real L_a = require_real(dev, idx, type, "L_a");
        const Real K = require_real(dev, idx, type, "K");
        const Real J = require_real(dev, idx, type, "J");
        const Real B = nonneg_or_zero(dev, type, name, "B");
        const Real T_load = real_or(dev, "T_load", Real{0});
        for (const auto& [label, v] : {std::pair{"L_a", L_a},
                                       std::pair{"K", K},
                                       std::pair{"J", J}}) {
            if (!std::isfinite(v) || v <= Real{0}) {
                throw std::runtime_error(std::format(
                    "yaml::load: dc_machine '{}': {} must be finite and "
                    "> 0 (got {})", name, label, v));
            }
        }
        if (!std::isfinite(T_load)) {
            throw std::runtime_error(std::format(
                "yaml::load: dc_machine '{}': T_load must be finite",
                name));
        }
        const std::string a_pos = require_string(dev, idx, type, "a_pos");
        const std::string a_neg = require_string(dev, idx, type, "a_neg");
        const std::string omega = require_string(dev, idx, type, "omega_node");
        const std::string emf = name + ".emf";
        std::string l_from = a_pos;
        if (R_a > Real{0}) {
            l_from = name + ".ra_la";
            b.add_resistor(name + ".ra", a_pos, l_from, R_a);
        }
        b.add_inductor(name + ".la", l_from, emf, L_a);
        b.add_ideal_transformer(name + ".em", omega, "gnd", emf, a_neg, K);
        b.add_capacitor(name + ".j", omega, "gnd", J, Real{0});
        if (B > Real{0}) {
            b.add_resistor(name + ".b", omega, "gnd", Real{1} / B);
        }
        if (T_load != Real{0}) {
            b.add_current_source(name + ".tl", "gnd", omega, T_load);
        }
        std::vector<std::string> members;
        if (R_a > Real{0}) members.push_back(name + ".ra");
        for (const char* s : {".la", ".em", ".j"}) members.push_back(name + s);
        if (B > Real{0}) members.push_back(name + ".b");
        if (T_load != Real{0}) members.push_back(name + ".tl");
        b.record_composite("dc_machine", name, name + ".em", std::move(members));
    } else if (type == "diode" &&
               string_or(dev, "form", default_switch_form()) == "branch") {
        // add-branch-form-ideal-switch: `form: branch` (R_on, g_off
        // default 0 = ideal; a given g_on means R_on = 1/g_on).
        b.add_diode_branch(
            name,
            require_string(dev, idx, type, "anode"),
            require_string(dev, idx, type, "cathode"),
            dev["g_on"] && !dev["R_on"]
                ? Real{1} / require_real(dev, idx, type, "g_on")
                : real_or(dev, "R_on", Real{0}),
            real_or(dev, "g_off", Real{0}),
            real_or(dev, "V_th", Real{0}),
            real_or(dev, "V_f", Real{0}));
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
    } else if (type == "switch" &&
               string_or(dev, "form", default_switch_form()) == "branch") {
        b.add_ideal_switch(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"),
            dev["g_on"] && !dev["R_on"]
                ? Real{1} / require_real(dev, idx, type, "g_on")
                : real_or(dev, "R_on", Real{0}),
            real_or(dev, "g_off", Real{0}));
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
            real_or(dev, "R_off", Real{1e9}),
            /*body_diode=*/true,
            real_or(dev, "V_F",   Real{0.7}), form);
    } else if (type == "mosfet_with_body_diode") {
        b.add_mosfet_with_body_diode(
            name,
            require_string(dev, idx, type, "drain"),
            require_string(dev, idx, type, "source"),
            real_or(dev, "R_on",         Real{1e-3}),
            real_or(dev, "R_off",        Real{1e9}),
            real_or(dev, "V_F",          Real{0.7}),
            real_or(dev, "g_on_diode",   Real{1e3}),
            real_or(dev, "g_off_diode",  Real{1e-9}), form);
    } else if (type == "igbt") {
        // add-pwl-igbt-and-thyristor: V_ce0, reverse_blocking (default
        // False, True when V_ce0 > 0 — the builder's stage-1 default).
        require_only_keys(dev, idx, type, {"collector", "emitter", "R_on", "R_off",
                                           "V_ce0", "reverse_blocking", "V_th",
                                           "losses", "form"});   // load_losses, below
        const Real V_ce0 = real_or(dev, "V_ce0", Real{0});
        b.add_igbt(
            name,
            require_string(dev, idx, type, "collector"),
            require_string(dev, idx, type, "emitter"),
            real_or(dev, "R_on",  Real{10e-3}),
            real_or(dev, "R_off", Real{1e9}),
            V_ce0,
            bool_or(dev, "reverse_blocking", V_ce0 > Real{0}),
            real_or(dev, "V_th", Real{0}), form);
    } else if (type == "thyristor" || type == "gto" || type == "triac") {
        const bool tri = type == "triac";
        const bool gto = type == "gto";
        if (gto) {
            require_only_keys(dev, idx, type,
                              {"anode", "cathode", "R_on", "R_off", "V_f", "V_th"});
        } else if (tri) {
            require_only_keys(dev, idx, type,
                              {"t1", "t2", "R_on", "R_off", "V_f", "I_H", "V_th"});
        } else {
            require_only_keys(dev, idx, type, {"anode", "cathode", "R_on",
                                               "R_off", "V_f", "I_H", "V_th"});
        }
        const std::string a = require_string(dev, idx, type, tri ? "t1" : "anode");
        const std::string k = require_string(dev, idx, type, tri ? "t2" : "cathode");
        const Real R_on = real_or(dev, "R_on", Real{1e-3});
        const Real R_off = real_or(dev, "R_off", Real{1e9});
        const Real V_f = real_or(dev, "V_f", Real{0});
        const Real V_th = real_or(dev, "V_th", Real{0});
        if (gto) {
            b.add_gto(name, a, k, R_on, R_off, V_f, V_th);
        } else if (tri) {
            b.add_triac(name, a, k, R_on, R_off, V_f,
                        real_or(dev, "I_H", Real{0}), V_th);
        } else {
            b.add_thyristor(name, a, k, R_on, R_off, V_f,
                            real_or(dev, "I_H", Real{0}), V_th);
        }
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
    } else if (type == "pmsm" || type == "bldc") {
        // add-mna-native-machines 3.6: CircuitBuilder.add_pmsm_mna and
        // add_bldc_mna (python/pulsim/machines.py). The BLDC's PM-flux
        // series is `trapezoid_harmonics`, ported here; test_yaml_composites
        // pins the two expansions equal.
        const bool bldc = type == "bldc";
        if (bldc) {
            require_only_keys(dev, idx, type,
                              {"phase_nodes", "neutral_node", "omega_node",
                               "theta_node", "R_s", "L_s", "k_e", "pole_pairs",
                               "J", "B", "T_load", "flat_top_deg",
                               "n_harmonics", "omega0", "theta0"});
        } else {
            require_only_keys(dev, idx, type,
                              {"phase_nodes", "neutral_node", "omega_node",
                               "theta_node", "R_s", "L_d", "L_q", "L_0",
                               "psi_pm", "pole_pairs", "J", "B", "T_load",
                               "omega0", "theta0"});
        }
        if (!dev["phase_nodes"] || !dev["phase_nodes"].IsSequence() ||
            dev["phase_nodes"].size() != 3) {
            throw std::runtime_error(std::format(
                "yaml::load: {} '{}' requires phase_nodes: [a, b, c] "
                "(sequence of 3 strings)", type, name));
        }
        std::array<std::string, 3> ph;
        for (std::size_t k = 0; k < 3; ++k) {
            if (!dev["phase_nodes"][k].IsScalar()) {
                throw std::runtime_error(std::format(
                    "yaml::load: {} '{}' phase_nodes[{}] must be a scalar "
                    "node name", type, name, k));
            }
            ph[k] = dev["phase_nodes"][k].as<std::string>();
        }
        const std::string nn = require_string(dev, idx, type, "neutral_node");
        const std::string w = require_string(dev, idx, type, "omega_node");
        const std::string th = require_string(dev, idx, type, "theta_node");
        const Real R_s = require_real(dev, idx, type, "R_s");
        const Real pp = require_real(dev, idx, type, "pole_pairs");
        const Real J = require_real(dev, idx, type, "J");
        const Real B = nonneg_or_zero(dev, type, name, "B");
        const Real T_load = real_or(dev, "T_load", Real{0});
        const Real omega0 = real_or(dev, "omega0", Real{0});
        const Real theta0 = real_or(dev, "theta0", Real{0});
        if (!bldc) {
            b.add_pmsm_mna(name, ph[0], ph[1], ph[2], nn, w, th, R_s,
                           require_real(dev, idx, type, "L_d"),
                           require_real(dev, idx, type, "L_q"),
                           require_real(dev, idx, type, "psi_pm"), pp, J, B,
                           T_load, real_or(dev, "L_0", Real{0}), omega0, theta0);
        } else {
            const Real k_e = require_real(dev, idx, type, "k_e");
            const Real flat = real_or(dev, "flat_top_deg", Real{120});
            const Real nh = real_or(dev, "n_harmonics", Real{25});
            if (!(std::isfinite(k_e) && k_e > Real{0}) || !(pp >= Real{1}) ||
                pp != std::floor(pp)) {
                throw std::runtime_error(std::format(
                    "yaml::load: bldc '{}': k_e must be > 0 and pole_pairs a "
                    "positive integer", name));
            }
            if (!(flat >= Real{0} && flat < Real{180})) {
                throw std::runtime_error(std::format(
                    "yaml::load: bldc '{}': flat_top_deg must be in [0, 180)",
                    name));
            }
            if (!(nh >= Real{1}) || nh != std::floor(nh) || nh > Real{1000}) {
                throw std::runtime_error(std::format(
                    "yaml::load: bldc '{}': n_harmonics must be an integer in "
                    "[1, 1000]", name));
            }
            // psi_h = (k_e/pp)·4·sin(h·γ)/(π·γ·h³), odd h (trapezoid_harmonics)
            const Real gamma =
                (Real{180} - flat) * (std::numbers::pi / Real{180}) / Real{2};
            std::vector<std::array<Real, 3>> harm;
            for (int h = 1; h <= static_cast<int>(nh); h += 2) {
                const Real hr = h;
                const Real s = gamma > Real{0}
                    ? std::sin(hr * gamma) / (hr * gamma) : Real{1};
                harm.push_back({hr, (k_e / pp) * Real{4} * s
                                        / (std::numbers::pi * hr * hr),
                                Real{0}});
            }
            const Real L_s = require_real(dev, idx, type, "L_s");
            b.add_pmsm_mna(name, ph[0], ph[1], ph[2], nn, w, th, R_s, L_s, L_s,
                           harm[0][1], pp, J, B, T_load, Real{0}, omega0,
                           theta0, harm);
        }
    } else if (type == "induction_machine" ||
               (type == "induction_motor" &&
                string_or(dev, "model", "observer") == "mna")) {
        // add-mna-native-machines: CircuitBuilder::add_induction_machine_mna.
        // `induction_machine` takes leakages (L_ls, L_lr); the legacy
        // `induction_motor` with `model: mna` keeps its self-inductance
        // keys (L_s, L_r) and is converted here.
        if (const auto m = string_or(dev, "model", "mna"); m != "mna") {
            throw std::runtime_error(std::format(
                "yaml::load: induction_machine '{}': model must be 'mna' "
                "(got '{}'); the observer is type: induction_motor",
                name, m));
        }
        if (type == "induction_machine") {
            require_only_keys(dev, idx, type,
                              {"model", "phase_nodes", "neutral_node",
                               "omega_node", "R_s", "L_ls", "R_r", "L_lr",
                               "L_m", "pole_pairs", "J", "B", "T_load",
                               "omega0"});
        } else {
            require_only_keys(dev, idx, type,
                              {"model", "phase_nodes", "neutral_node",
                               "omega_node", "R_s", "L_s", "R_r", "L_r",
                               "L_m", "pole_pairs", "J", "B", "T_load",
                               "omega0"});
        }
        if (!dev["phase_nodes"] || !dev["phase_nodes"].IsSequence() ||
            dev["phase_nodes"].size() != 3) {
            throw std::runtime_error(std::format(
                "yaml::load: {} '{}' requires phase_nodes: [a, b, c] "
                "(sequence of 3 strings)", type, name));
        }
        std::array<std::string, 3> ph;
        for (std::size_t k = 0; k < 3; ++k) {
            if (!dev["phase_nodes"][k].IsScalar()) {
                throw std::runtime_error(std::format(
                    "yaml::load: {} '{}' phase_nodes[{}] must be a scalar "
                    "node name", type, name, k));
            }
            ph[k] = dev["phase_nodes"][k].as<std::string>();
        }
        const Real L_m = require_real(dev, idx, type, "L_m");
        const bool legacy = type == "induction_motor";
        const Real L_ls = legacy ? require_real(dev, idx, type, "L_s") - L_m
                                 : require_real(dev, idx, type, "L_ls");
        const Real L_lr = legacy ? require_real(dev, idx, type, "L_r") - L_m
                                 : require_real(dev, idx, type, "L_lr");
        b.add_induction_machine_mna(
            name, ph[0], ph[1], ph[2],
            require_string(dev, idx, type, "neutral_node"),
            require_string(dev, idx, type, "omega_node"),
            require_real(dev, idx, type, "R_s"), L_ls,
            require_real(dev, idx, type, "R_r"), L_lr, L_m,
            require_real(dev, idx, type, "pole_pairs"),
            require_real(dev, idx, type, "J"),
            nonneg_or_zero(dev, type, name, "B"),
            real_or(dev, "T_load", Real{0}),
            real_or(dev, "omega0", Real{0}));
    } else if (type == "induction_motor") {
        if (const auto m = string_or(dev, "model", "observer");
            m != "observer") {
            throw std::runtime_error(std::format(
                "yaml::load: induction_motor '{}': model must be 'observer' "
                "or 'mna' (got '{}')", name, m));
        }
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
    } else if (type == "power_mosfet") {
        // add-power-mosfet-datasheet-device. Strict keys: a misspelt
        // parameter would otherwise silently take its default.
        auto strict = [&](const YAML::Node& node, const std::string& where,
                          std::initializer_list<std::string_view> keys) {
            strict_keys(dev, idx, node, where, keys);
        };
        strict(dev, "power_mosfet",
               {"type", "name", "drain", "source", "gate", "K", "V_T",
                "lambda", "kappa", "R_d", "C_gs", "C_gd", "C_ds", "R_g",
                "L_s", "body_diode", "T_j", "rds_tc1", "rds_tc2", "vt_tc"});
        auto cap = [&](const std::string& f) -> models::PowerMosfet::Capacitance {
            const YAML::Node n = dev[f];
            if (!n) return Real{0};
            if (n.IsScalar()) return n.as<Real>();
            return charge_law_node(dev, idx, n, name + "." + f, type + "." + f);
        };
        models::PowerMosfet::Params p;
        p.K = require_real(dev, idx, type, "K");
        p.V_T = require_real(dev, idx, type, "V_T");
        p.lambda = real_or(dev, "lambda", Real{0});
        p.kappa = real_or(dev, "kappa", Real{15});
        p.R_d = real_or(dev, "R_d", Real{0});
        p.C_gs = require_real(dev, idx, type, "C_gs");
        p.C_gd = cap("C_gd");
        p.C_ds = cap("C_ds");
        p.R_g = real_or(dev, "R_g", Real{0});
        p.L_s = real_or(dev, "L_s", Real{0});
        p.T_j = real_or(dev, "T_j", Real{25});
        p.rds_tc1 = real_or(dev, "rds_tc1", Real{0});
        p.rds_tc2 = real_or(dev, "rds_tc2", Real{0});
        p.vt_tc = real_or(dev, "vt_tc", Real{0});
        if (const YAML::Node bd = dev["body_diode"]; bd && !bd.IsNull()) {
            const std::string model = bd["model"] ? bd["model"].as<std::string>() : "";
            if (model == "lauritzen") {
                strict(bd, "body_diode", {"model", "I_S", "n", "V_T", "tau", "T_M", "G_min"});
                p.body = models::PowerMosfet::BodyDiode::Lauritzen;
                auto& L = p.lauritzen;
                L.I_S = real_or(bd, "I_S", L.I_S); L.n = real_or(bd, "n", L.n);
                L.V_T = real_or(bd, "V_T", L.V_T); L.tau = real_or(bd, "tau", L.tau);
                L.T_M = real_or(bd, "T_M", L.T_M); L.G_min = real_or(bd, "G_min", L.G_min);
            } else if (model == "shockley") {
                strict(bd, "body_diode", {"model", "I_S", "n", "V_T", "G_min", "BV"});
                p.body = models::PowerMosfet::BodyDiode::Shockley;
                auto& S = p.shockley;
                S.I_S = real_or(bd, "I_S", S.I_S); S.n = real_or(bd, "n", S.n);
                S.V_T = real_or(bd, "V_T", S.V_T); S.G_min = real_or(bd, "G_min", S.G_min);
                S.BV = real_or(bd, "BV", S.BV);
            } else {
                throw std::runtime_error(std::format(
                    "yaml::load: device {}: body_diode.model must be "
                    "'lauritzen' or 'shockley' (got '{}')",
                    device_label(dev, idx), model));
            }
        }
        b.add_power_mosfet(
            name,
            require_string(dev, idx, type, "drain"),
            require_string(dev, idx, type, "source"),
            require_string(dev, idx, type, "gate"), p);
    } else if (type == "nonlinear_capacitor" && !dev["pins"]) {
        // add-realistic-passives 4.5: a charge-law capacitor (the
        // authoring form, from/to; the IR emitter's `pins:` entry takes
        // the registry branch below). `law:` is
        // the power_mosfet capacitance map; `dc_bias: {v, C}` (v >= 0,
        // the first at 0) is an MLCC's small-signal C(v_bias) curve,
        // integrated into an ODD charge table (C taken even in v).
        strict_keys(dev, idx, dev, type, {"type", "name", "from", "to", "law", "dc_bias"});
        if (static_cast<bool>(dev["law"]) == static_cast<bool>(dev["dc_bias"])) {
            throw std::runtime_error(std::format(
                "yaml::load: device {}: nonlinear_capacitor needs exactly one "
                "of 'law' (map with law: junction | vdmos_cgd | table) or "
                "'dc_bias' (map with v, C)", device_label(dev, idx)));
        }
        models::NonlinearCapacitor::Params law;
        if (const YAML::Node n = dev["law"]) {
            if (!n.IsMap()) {
                throw std::runtime_error(std::format(
                    "yaml::load: device {}: 'law' must be a map with law: "
                    "junction | vdmos_cgd | table", device_label(dev, idx)));
            }
            law = charge_law_node(dev, idx, n, name, type + ".law");
        } else {
            const YAML::Node db = dev["dc_bias"];
            if (!db.IsMap()) {
                throw std::runtime_error(std::format(
                    "yaml::load: device {}: 'dc_bias' must be a map with v, C",
                    device_label(dev, idx)));
            }
            strict_keys(dev, idx, db, "dc_bias", {"v", "C"});
            const auto v = require_real_list(db, idx, type + ".dc_bias", "v");
            const auto c = require_real_list(db, idx, type + ".dc_bias", "C");
            if (v.size() != c.size() || v.size() < 2 || v.front() != Real{0}) {
                throw std::runtime_error(std::format(
                    "yaml::load: device {}: dc_bias needs v and C of equal "
                    "length >= 2 with v[0] = 0", device_label(dev, idx)));
            }
            std::vector<Real> vv, cc;
            for (std::size_t k = v.size(); k-- > 1;) { vv.push_back(-v[k]); cc.push_back(c[k]); }
            vv.insert(vv.end(), v.begin(), v.end());
            cc.insert(cc.end(), c.begin(), c.end());
            law = models::NonlinearCapacitor::from_table(
                std::make_shared<const models::ChargeTable>(
                    models::ChargeTable::from_capacitance(vv, cc, name)));
        }
        b.add_nonlinear_capacitor_law(
            name,
            require_string(dev, idx, type, "from"),
            require_string(dev, idx, type, "to"), law);
    } else if (std::ranges::find(ir::kKindYaml, std::string_view{type}) !=
               ir::kKindYaml.end()) {
        // A registry kind without an authoring type of its own
        // (shockley_diode, lauritzen_diode, nonlinear_capacitor, pmsm_mna,
        // ...): the emitter's entry, `pins: [...]` and its stored fields
        // (one schema table for both, `ir::kKindYaml` / `kKindFields`).
        YAML::Node d = YAML::Clone(dev);
        d.remove("type");
        d["kind"] = type;
        try {
            ir::YamlIo::read_entry(b, d);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error(std::format("yaml::load: device {}: {}",
                                                 device_label(dev, idx), e.what()));
        }
    } else {
        throw std::runtime_error(
            "yaml::load: device " + device_label(dev, idx) +
            " has unknown type '" + type + "'. Supported "
            "types: voltage_source, current_source, "
            "pwm_voltage_source, sine_voltage_source, "
            "pulse_voltage_source, resistor, capacitor, "
            "inductor, saturable_inductor, hysteretic_inductor, "
            "diode, nonlinear_diode (retired; needs allow_non_passive), switch, "
            "mosfet, mosfet_with_body_diode, mosfet_level1, "
            "power_mosfet, igbt, igbt_level1, thyristor, gto, triac, vcvs, vccs, ccvs, cccs, gyrator, "
            "sine_current_source, pulse_current_source, "
            "pwl_current_source, op_amp_ideal, "
            "gapped_core_inductor, transformer, multi_winding_transformer, "
            "ideal_transformer, saturable_transformer, induction_motor, "
            "induction_machine, dc_machine, pmsm, bldc, winding, permeance, "
            "air_gap, magnetic_resistance, saturable_permeance, "
            "hysteretic_permeance, excess_loss, transmission_line");
    }
    if (dev["losses"]) load_losses(b, dev, idx, type, name);
}

/// Keys `load_simulation_options` reads. Anything else in the block
/// is reported (not fatal) — see `collect_unknown_simulation_keys`.
/// A number as the loader has always read it (yaml-cpp's own decode:
/// `.inf`, `+5`, `1e-3`), as opposed to an expression.
[[nodiscard]] inline bool is_number_node(const YAML::Node& n) {
    Real x{};
    return YAML::convert<Real>::decode(n, x);
}

/// `circuit.parameters`: [{name, value, unit?, lo?, hi?}], where a
/// string value is an expression over the other parameters (any order;
/// a cycle is refused naming it).
inline void load_parameters(builder::CircuitBuilder& b, const YAML::Node& ps) {
    if (!ps.IsSequence()) {
        throw std::runtime_error("yaml::load: 'circuit.parameters' must be a list "
                                 "of {name, value, unit}");
    }
    std::vector<std::string> names;
    std::vector<YAML::Node> nodes;
    for (const auto& p : ps) {
        if (!p.IsMap() || !p["name"] || !p["value"] || !p["value"].IsScalar()) {
            throw std::runtime_error("yaml::load: each 'circuit.parameters' entry needs "
                                     "a name and a scalar value");
        }
        for (const auto& kv : p) {
            const auto k = kv.first.as<std::string>();
            if (k != "name" && k != "value" && k != "unit" && k != "lo" && k != "hi") {
                throw std::runtime_error(std::format(
                    "yaml::load: parameter '{}': unknown key '{}' (name, value, unit, "
                    "lo, hi)", p["name"].as<std::string>(), k));
            }
        }
        names.push_back(p["name"].as<std::string>());
        nodes.push_back(p);
    }
    std::vector<std::vector<std::size_t>> deps(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto v = nodes[i]["value"].as<std::string>();
        if (is_number_node(nodes[i]["value"])) continue;
        for (const auto& n : ir::Expr::names_in(v, std::format("param(\"{}\")", names[i]))) {
            const auto it = std::find(names.begin(), names.end(), n);
            if (it != names.end()) deps[i].push_back(static_cast<std::size_t>(it - names.begin()));
        }
    }
    for (const auto i : ir::dependency_order(names, deps)) {
        const auto& p = nodes[i];
        const auto v = p["value"].as<std::string>();
        const auto unit = p["unit"] ? p["unit"].as<std::string>() : std::string{};
        std::optional<Real> lo, hi;
        if (p["lo"]) lo = p["lo"].as<Real>();
        if (p["hi"]) hi = p["hi"].as<Real>();
        if (is_number_node(p["value"])) b.param(names[i], p["value"].as<Real>(), unit, lo, hi);
        else b.param_expr(names[i], v, unit, lo, hi);
    }
}

/// `load_device`, where a field key may hold an expression string: the
/// device is built with its value, then the field is driven by a derived
/// parameter "<device>:<field>", so `with_params` re-evaluates it.
/// Enumerated keys (law, model, form, ...) are not fields and are never
/// read as expressions.
inline void load_device_expr(builder::CircuitBuilder& b, const YAML::Node& dev,
                             std::size_t idx) {
    auto is_field = [](std::string_view key) {
        for (const auto& row : ir::kKindFields) {
            for (const auto& f : row) if (f.named(key)) return true;
        }
        return false;
    };
    std::vector<std::pair<std::string, std::string>> exprs;
    YAML::Node d = dev.IsMap() ? YAML::Clone(dev) : dev;
    const std::string name = dev.IsMap() && dev["name"] && dev["name"].IsScalar()
        ? dev["name"].as<std::string>() : std::format("#{}", idx);
    if (dev.IsMap()) {
        for (const auto& kv : dev) {
            const auto key = kv.first.as<std::string>();
            if (!kv.second.IsScalar() || !is_field(key)) continue;
            if (is_number_node(kv.second)) continue;
            const auto text = kv.second.as<std::string>();
            const auto who = std::format("device \"{}\" field \"{}\"", name, key);
            const auto params = b.parameters();
            const auto e = ir::Expr::compile(text, [&](std::string_view n) {
                return b.lookup_param(n);   // the open scope, then outward
            }, who);
            std::vector<Real> vals;
            for (const auto& q : params) vals.push_back(q.value);
            d[key] = e.eval(vals);
            exprs.emplace_back(key, text);
        }
    }
    load_device(b, d, idx);
    for (const auto& [key, text] : exprs) {
        Index br = -1;
        try { br = b.branch_id_of(name); } catch (const std::exception&) {}
        if (br < 0 || ir::find_field(b.pool().kind_of(br), key) == nullptr) {
            throw std::runtime_error(std::format(
                "yaml::load: device \"{}\" field \"{}\": an expression can drive "
                "only a value field of the device's own branch; write a number here",
                name, key));
        }
        // Declared in the device's scope, under its local name there.
        const auto prefix = b.scoped_name("");
        const std::string_view local = name.starts_with(prefix) && b.current_scope() != 0
            ? std::string_view{name}.substr(prefix.size()) : std::string_view{name};
        const auto hidden = b.param_expr(std::format("{}:{}", local, key), text);
        b.bind_param(std::format("{}.{}", name, key), hidden.name);
    }
}

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

/// Keys of authoring device entries whose value is a NODE name (group 8:
/// renamed into a subsystem's scope). `control` names a branch. A key
/// added to `load_device` that names a node belongs here; the subsystem
/// test loads one device of every authoring type inside a scope and
/// compares it with its hand-prefixed twin.
inline constexpr std::string_view kNodeKeys[] = {
    "from", "to", "anode", "cathode", "drain", "source", "gate", "collector",
    "emitter", "in_pos", "in_neg", "out", "out_pos", "out_neg", "a_pos", "a_neg",
    "b_pos", "b_neg",
    "p_from", "p_to", "s_from", "s_to", "p1_pos", "p1_neg", "p2_pos", "p2_neg",
    "neutral_node", "omega_node", "theta_node",
    "omega", "theta"};   // a registry-kind entry (pmsm_mna); its `pins` lists too

/// One subsystem block's naming: its ports (local name -> the node it
/// binds in the enclosing scope) on top of the builder's open scope.
struct BlockNames {
    builder::CircuitBuilder& b;
    std::unordered_map<std::string, std::string> ports;

    [[nodiscard]] std::string node(const std::string& local) const {
        if (const auto it = ports.find(local); it != ports.end()) return it->second;
        if (builder::CircuitBuilder::is_ground_name(local)) return local;
        return b.scoped_name(local);
    }
    /// A copy of `dev` with its name, nodes and sensed branch in scope.
    [[nodiscard]] YAML::Node device(const YAML::Node& dev) const {
        if (!dev.IsMap()) return dev;
        YAML::Node d = YAML::Clone(dev);
        auto scalar = [](const YAML::Node& n) { return n && n.IsScalar(); };
        if (scalar(d["name"])) d["name"] = b.scoped_name(d["name"].as<std::string>());
        if (scalar(d["control"])) d["control"] = b.scoped_name(d["control"].as<std::string>());
        for (const auto k : kNodeKeys) {
            const std::string key{k};
            if (scalar(d[key])) d[key] = node(d[key].as<std::string>());
        }
        for (const auto* list : {"windings", "secondaries"}) {
            if (!d[list] || !d[list].IsSequence()) continue;
            for (std::size_t i = 0; i < d[list].size(); ++i) {
                YAML::Node w = d[list][i];
                if (!w.IsMap()) continue;
                for (const auto* k : {"from", "to", "s_from", "s_to"}) {
                    if (scalar(w[k])) w[k] = node(w[k].as<std::string>());
                }
            }
        }
        auto pins = [&](YAML::Node list) {
            if (!list || !list.IsSequence()) return;
            for (std::size_t i = 0; i < list.size(); ++i) {
                if (scalar(list[i])) list[i] = node(list[i].as<std::string>());
            }
        };
        pins(d["pins"]);
        pins(d["nodes"]);   // external_device
        if (d["phases"] && d["phases"].IsSequence()) {
            for (std::size_t i = 0; i < d["phases"].size(); ++i) {
                YAML::Node ph = d["phases"][i];
                if (!ph.IsMap()) continue;
                if (scalar(ph["name"])) ph["name"] = b.scoped_name(ph["name"].as<std::string>());
                pins(ph["pins"]);
            }
        }
        if (d["phase_nodes"] && d["phase_nodes"].IsSequence()) {
            for (std::size_t i = 0; i < d["phase_nodes"].size(); ++i) {
                if (scalar(d["phase_nodes"][i])) {
                    d["phase_nodes"][i] = node(d["phase_nodes"][i].as<std::string>());
                }
            }
        }
        return d;
    }
};

/// `circuit.subsystems`: [{name, kind?, ports?, parameters?, nodes?,
/// devices?, subsystems?}]. Each block is a scope: its parameters are
/// declared in it (expressions see them first, then the enclosing
/// scopes), and every device and node name it writes is prefixed with
/// its path, except its ports (bound to nodes of the enclosing block)
/// and ground.
/// `nodes:` entries: a name (electrical unless a pin implies otherwise)
/// or `{name, domain}` with domain electrical / thermal / mechanical
/// (group 9); an unknown domain is refused naming node and domain.
inline void load_nodes(builder::CircuitBuilder& b, const YAML::Node& ns,
                       const BlockNames& names) {
    for (const auto& n : ns) {
        if (n.IsScalar()) {
            b.node(names.node(n.as<std::string>()));
            continue;
        }
        if (!n.IsMap() || n.size() != 2 || !n["name"] || !n["name"].IsScalar() ||
            !n["domain"] || !n["domain"].IsScalar()) {
            throw std::runtime_error(
                "yaml::load: a 'nodes' entry is a name or {name, domain}");
        }
        const auto name = n["name"].as<std::string>();
        const auto dn = n["domain"].as<std::string>();
        const auto d = ir::domain_from_name(dn);
        if (!d || *d == ir::Domain::Magnetic) {
            throw std::runtime_error(std::format(
                "yaml::load: node '{}': unknown domain '{}' (electrical, thermal, "
                "mechanical)", name, dn));
        }
        try {
            b.node(names.node(name), *d);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error(std::format("yaml::load: {}", e.what()));
        }
    }
}

inline void load_subsystems(builder::CircuitBuilder& b, const YAML::Node& list,
                            const BlockNames& outer, std::vector<std::string>& warnings) {
    if (!list.IsSequence()) {
        throw std::runtime_error("yaml::load: 'subsystems' must be a list of blocks");
    }
    for (const auto& blk : list) {
        if (!blk.IsMap() || !blk["name"] || !blk["name"].IsScalar()) {
            throw std::runtime_error("yaml::load: each subsystem needs a 'name'");
        }
        const auto name = blk["name"].as<std::string>();
        if (name.empty() || name.find('/') != std::string::npos) {
            throw std::runtime_error(std::format(
                "yaml::load: subsystem '{}': a block name is one non-empty path "
                "segment (no '/'); nest blocks with 'subsystems'", name));
        }
        warn_unknown_keys(blk, {"name", "kind", "ports", "parameters", "nodes", "devices",
                                "subsystems"},
                          std::format("subsystem '{}':", b.scoped_name(name)), warnings);
        BlockNames inner{b, {}};
        if (const auto ps = blk["ports"]) {
            if (!ps.IsMap()) {
                throw std::runtime_error(std::format(
                    "yaml::load: subsystem '{}': 'ports' must map a local name to "
                    "a node of the enclosing block", name));
            }
            for (const auto& kv : ps) {
                inner.ports.emplace(kv.first.as<std::string>(),
                                    outer.node(kv.second.as<std::string>()));
            }
        }
        const auto kind = blk["kind"] && blk["kind"].IsScalar()
            ? blk["kind"].as<std::string>() : std::string{};
        for (const auto& sc : b.scopes()) {   // a block is an instance: never merged
            if (sc.path == b.scoped_name(name)) {
                throw std::runtime_error(std::format(
                    "yaml::load: subsystem '{}' is listed twice", sc.path));
            }
        }
        try {
            (void)b.open_scope(name, kind);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error(std::format("yaml::load: subsystem: {}", e.what()));
        }
        if (blk["parameters"]) load_parameters(b, blk["parameters"]);
        if (blk["nodes"] && blk["nodes"].IsSequence()) load_nodes(b, blk["nodes"], inner);
        if (const auto devs = blk["devices"]) {
            if (!devs.IsSequence()) {
                throw std::runtime_error(std::format(
                    "yaml::load: subsystem '{}': 'devices' must be a list", name));
            }
            std::size_t idx = 0;
            for (const auto& dev : devs) load_device_expr(b, inner.device(dev), idx++);
        }
        if (blk["subsystems"]) load_subsystems(b, blk["subsystems"], inner, warnings);
        b.close_scope();
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
    opts.store_every = size_or(sim, "store_every", pulsim::sim::Defaults::store_every);
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
        size_or(sim, "max_event_iterations",
                pulsim::sim::Defaults::max_event_iterations);
    opts.max_newton_iterations =
        size_or(sim, "max_newton_iterations",
                pulsim::sim::Defaults::max_newton_iterations);
    opts.tol_newton_dx =
        real_or(sim, "tol_newton_dx", pulsim::sim::Defaults::tol_newton_dx);
    opts.tol_newton_res =
        real_or(sim, "tol_newton_res", pulsim::sim::Defaults::tol_newton_res);
    // Phase 2.4 adaptive RK selector (schema v1.5, wiring v1.6).
    opts.integrator =
        string_or(sim, "integrator", std::string{"kernel"});
    opts.rtol    = real_or(sim, "rtol",    pulsim::sim::Defaults::rtol_variable);
    opts.atol    = real_or(sim, "atol",    pulsim::sim::Defaults::atol_variable);
    opts.dt_init = real_or(sim, "dt_init", Real{0});
}

}  // namespace detail

namespace detail {

/// Shared body of `load_string` / `load_file`. `source` is recorded on
/// the result and is what error messages call the file.
[[nodiscard]] inline LoadedCircuit load_text(
    const std::string& yaml_text, const std::string& source,
    const LoadOptions& opts = {}) {
    struct Scope {
        const LoadOptions* prev;
        explicit Scope(const LoadOptions* o) : prev(tl_load_options) { tl_load_options = o; }
        ~Scope() { tl_load_options = prev; }
    } const scope(&opts);
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
    // The emitter's document (`Circuit.to_yaml()`): read by its inverse,
    // so every circuit the emitter writes is a netlist this loader takes.
    if (root["pulsim_circuit"]) {
        try {
            loaded.builder = ir::from_yaml(yaml_text).to_builder();
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error(std::string{"yaml::load_string: "} + e.what());
        }
        return loaded;
    }
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
    detail::warn_unknown_keys(circuit, {"nodes", "devices", "parameters", "subsystems"},
                              "circuit:", loaded.warnings);
    if (circuit["parameters"]) {
        detail::load_parameters(loaded.builder, circuit["parameters"]);
    }

    // Optional: pre-declare nodes (auto-created on device
    // reference anyway, but exposing it gives the user
    // explicit control of insertion order).
    if (circuit["nodes"] && circuit["nodes"].IsSequence()) {
        detail::load_nodes(loaded.builder, circuit["nodes"],
                           detail::BlockNames{loaded.builder, {}});
    }

    const YAML::Node devices = circuit["devices"];
    if (!devices || !devices.IsSequence()) {
        throw std::runtime_error(
            "yaml::load_string: missing required key "
            "'circuit.devices:' (must be a sequence)");
    }

    std::size_t idx = 0;
    for (const auto& dev : devices) {
        detail::load_device_expr(loaded.builder, dev, idx);
        ++idx;
    }
    if (circuit["subsystems"]) {
        detail::load_subsystems(loaded.builder, circuit["subsystems"],
                                detail::BlockNames{loaded.builder, {}}, loaded.warnings);
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
        for (const auto& kv : sim) {
            if (!kv.first.IsScalar()) continue;
            std::string key = kv.first.as<std::string>();
            if (detail::is_known_simulation_key(key)) {
                loaded.simulation_keys.push_back(std::move(key));
            }
        }
        loaded.has_simulation = true;
    }

    return loaded;
}

}  // namespace detail

/// Parse a YAML circuit description from a string. Throws
/// `std::runtime_error` on validation errors or malformed
/// YAML.
[[nodiscard]] inline LoadedCircuit load_string(
    const std::string& yaml_text, const LoadOptions& opts = {}) {
    return detail::load_text(yaml_text, "<string>", opts);
}

/// Parse a YAML circuit description from a file on disk.
/// Wraps any I/O or parse errors with the file path.
[[nodiscard]] inline LoadedCircuit load_file(
    const std::string& path, LoadOptions opts = {}) {
    if (opts.base_dir.empty()) {
        const auto parent = std::filesystem::path(std::u8string(path.begin(), path.end()))
                                .parent_path().u8string();
        opts.base_dir.assign(parent.begin(), parent.end());
    }
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error(
            "yaml::load_file: could not open '" + path + "'");
    }
    std::stringstream ss;
    ss << stream.rdbuf();
    try {
        return detail::load_text(ss.str(), path, opts);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "yaml::load_file('" + path + "'): " + e.what());
    }
}

}  // namespace pulsim::yaml
