#pragma once

#include "pulsim/v1/catalog/diode_catalog.hpp"
#include "pulsim/v1/catalog/igbt_catalog.hpp"
#include "pulsim/v1/catalog/lookup_table_2d.hpp"
#include "pulsim/v1/catalog/mosfet_catalog.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace pulsim::v1::catalog {

// =============================================================================
// add-catalog-device-models — Phase 7: device-catalog YAML loader
// =============================================================================
//
// Reads `<vendor>/<part>.yaml` device-data manifests into the three
// catalog Param structs. The schema:
//
//   class: mosfet | igbt | diode
//   vendor: <string>
//   part: <string>
//
//   # Common
//   tj_celsius: 25
//
//   # Per class — see docs/catalog-devices.md for the full schema
//
// The loader returns a `std::variant` over the three Param types so
// the call site can `std::visit` and dispatch to the right device
// constructor.

using CatalogDeviceParams = std::variant<MosfetCatalogParams,
                                          IgbtCatalogParams,
                                          DiodeCatalogParams>;

namespace detail {

[[nodiscard]] inline LookupTable1D parse_table_1d(const YAML::Node& node,
                                                    const std::string& ctx) {
    if (!node || !node.IsSequence() || node.size() < 2) {
        throw std::invalid_argument(
            ctx + ": expected sequence of {x, y} entries (≥ 2 points)");
    }
    std::vector<Real> xs, ys;
    xs.reserve(node.size());
    ys.reserve(node.size());
    for (const auto& pt : node) {
        if (!pt.IsMap() || pt.size() < 2) {
            throw std::invalid_argument(
                ctx + ": every point must be a 2-key map");
        }
        // Accept any 2-key naming: try (x,y), (V,C), (Vds,C), etc.
        // by iterating the map keys.
        Real x = std::numeric_limits<Real>::quiet_NaN();
        Real y = std::numeric_limits<Real>::quiet_NaN();
        bool first = true;
        for (const auto& it : pt) {
            const Real v = it.second.as<Real>();
            if (first) { x = v; first = false; } else { y = v; }
        }
        xs.push_back(x);
        ys.push_back(y);
    }
    return LookupTable1D(std::move(xs), std::move(ys));
}

[[nodiscard]] inline LookupTable2D parse_table_2d(const YAML::Node& node,
                                                    const std::string& ctx) {
    if (!node || !node.IsMap()) {
        throw std::invalid_argument(
            ctx + ": expected a map with 'x', 'y', 'values' fields");
    }
    if (!node["x"] || !node["y"] || !node["values"]) {
        throw std::invalid_argument(
            ctx + ": missing 'x' / 'y' / 'values' axes");
    }
    std::vector<Real> xs, ys, vs;
    for (const auto& v : node["x"]) xs.push_back(v.as<Real>());
    for (const auto& v : node["y"]) ys.push_back(v.as<Real>());
    for (const auto& v : node["values"]) vs.push_back(v.as<Real>());
    return LookupTable2D(std::move(xs), std::move(ys), std::move(vs));
}

[[nodiscard]] inline MosfetCatalogParams parse_mosfet(const YAML::Node& root) {
    MosfetCatalogParams p;
    if (root["vendor"]) p.vendor = root["vendor"].as<std::string>();
    if (root["part"])   p.part_number = root["part"].as<std::string>();
    if (root["V_th_25c"])           p.V_th_25c = root["V_th_25c"].as<Real>();
    if (root["V_th_temp_coef"])     p.V_th_temp_coef =
        root["V_th_temp_coef"].as<Real>();
    if (root["R_ds_on_25c"])        p.R_ds_on_25c =
        root["R_ds_on_25c"].as<Real>();
    if (root["R_ds_on_temp_coef"])  p.R_ds_on_temp_coef =
        root["R_ds_on_temp_coef"].as<Real>();
    if (root["V_ds_max"])           p.V_ds_max = root["V_ds_max"].as<Real>();
    if (root["I_dss_25c"])          p.I_dss_25c = root["I_dss_25c"].as<Real>();
    if (root["Coss"]) p.Coss = parse_table_1d(root["Coss"], "Coss");
    if (root["Ciss"]) p.Ciss = parse_table_1d(root["Ciss"], "Ciss");
    if (root["Crss"]) p.Crss = parse_table_1d(root["Crss"], "Crss");
    if (root["Eon"])  p.Eon  = parse_table_2d(root["Eon"], "Eon");
    if (root["Eoff"]) p.Eoff = parse_table_2d(root["Eoff"], "Eoff");
    return p;
}

[[nodiscard]] inline IgbtCatalogParams parse_igbt(const YAML::Node& root) {
    IgbtCatalogParams p;
    if (root["vendor"]) p.vendor = root["vendor"].as<std::string>();
    if (root["part"])   p.part_number = root["part"].as<std::string>();
    if (root["V_ce_sat_default"]) p.V_ce_sat_default =
        root["V_ce_sat_default"].as<Real>();
    if (root["V_ge_th"])    p.V_ge_th = root["V_ge_th"].as<Real>();
    if (root["V_ces_max"])  p.V_ces_max = root["V_ces_max"].as<Real>();
    if (root["tau_tail"])   p.tau_tail = root["tau_tail"].as<Real>();
    if (root["I_tail_fraction"]) p.I_tail_fraction =
        root["I_tail_fraction"].as<Real>();
    if (root["V_ce_sat_table"])
        p.V_ce_sat_table = parse_table_2d(root["V_ce_sat_table"], "V_ce_sat_table");
    if (root["Eon"])  p.Eon  = parse_table_2d(root["Eon"], "Eon");
    if (root["Eoff"]) p.Eoff = parse_table_2d(root["Eoff"], "Eoff");
    if (root["Erec"]) p.Erec = parse_table_2d(root["Erec"], "Erec");
    return p;
}

[[nodiscard]] inline DiodeCatalogParams parse_diode(const YAML::Node& root) {
    DiodeCatalogParams p;
    if (root["vendor"]) p.vendor = root["vendor"].as<std::string>();
    if (root["part"])   p.part_number = root["part"].as<std::string>();
    if (root["V_f_default"]) p.V_f_default = root["V_f_default"].as<Real>();
    if (root["R_on"])        p.R_on = root["R_on"].as<Real>();
    if (root["V_r_max"])     p.V_r_max = root["V_r_max"].as<Real>();
    if (root["s_rec"])       p.s_rec = root["s_rec"].as<Real>();
    if (root["V_f_table"])
        p.V_f_table = parse_table_2d(root["V_f_table"], "V_f_table");
    if (root["Q_rr_table"])
        p.Q_rr_table = parse_table_2d(root["Q_rr_table"], "Q_rr_table");
    return p;
}

}  // namespace detail

[[nodiscard]] inline CatalogDeviceParams parse_device_catalog_yaml(
    const std::string& yaml_text) {
    YAML::Node root;
    try {
        root = YAML::Load(yaml_text);
    } catch (const std::exception& e) {
        throw std::invalid_argument(
            std::string("CatalogDevice: invalid YAML: ") + e.what());
    }
    if (!root || !root.IsMap()) {
        throw std::invalid_argument("CatalogDevice: root must be a map");
    }
    if (!root["class"]) {
        throw std::invalid_argument(
            "CatalogDevice: missing 'class:' (mosfet | igbt | diode)");
    }
    const std::string cls = root["class"].as<std::string>();
    if (cls == "mosfet") {
        return detail::parse_mosfet(root);
    }
    if (cls == "igbt") {
        return detail::parse_igbt(root);
    }
    if (cls == "diode") {
        return detail::parse_diode(root);
    }
    throw std::invalid_argument(
        "CatalogDevice: unknown class '" + cls +
        "' (expected mosfet | igbt | diode)");
}

[[nodiscard]] inline CatalogDeviceParams load_device_catalog_file(
    const std::filesystem::path& path) {
    std::ifstream fp(path);
    if (!fp) {
        throw std::invalid_argument(
            "CatalogDevice: cannot open " + path.string());
    }
    std::ostringstream ss;
    ss << fp.rdbuf();
    return parse_device_catalog_yaml(ss.str());
}

}  // namespace pulsim::v1::catalog
