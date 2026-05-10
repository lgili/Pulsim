#pragma once

#include "pulsim/v1/magnetic/bh_curve.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::v1::magnetic {

// =============================================================================
// add-magnetic-core-models — Phase 5: core-catalog YAML loader
// =============================================================================
//
// Reads a `<vendor>/<material>.yaml` core-data manifest into the
// magnetic primitives. The file shape:
//
// ```yaml
// vendor: TDK
// material: N87
// geometry:
//   area_m2: 1.5e-4              # effective core cross-section A_e
//   path_length_m: 4.5e-2        # mean magnetic path l_e
// bh_curve:
//   - { H: -1500, B: -0.40 }
//   - { H:  -200, B: -0.30 }
//   - { H:     0, B:  0.00 }
//   - { H:   200, B:  0.30 }
//   - { H:  1500, B:  0.40 }
// steinmetz:
//   k:     1.5
//   alpha: 1.6
//   beta:  2.7
// jiles_atherton:               # optional
//   Ms:    1.0e6
//   a:     100
//   alpha: 1.0e-4
//   k:     50
//   c:     0.1
// ```
//
// Vendor-published datasheets typically give:
//   - the B-H "first quadrant" or full hysteresis loop at one or more
//     temperatures (we capture the upper / monotone branch)
//   - Steinmetz coefficients fitted from the loss-density curves
//   - sometimes J-A parameters; if absent the loader emits defaults
//     suitable for a soft ferrite
//
// The loader is yaml-cpp-based to match the existing Pulsim parser
// dependency (no new deps).

struct CatalogCore {
    std::string vendor;
    std::string material;
    Real area_m2 = 0.0;
    Real path_length_m = 0.0;
    BHCurveTable bh_curve;
    std::optional<SteinmetzLoss> steinmetz;
    std::optional<JilesAthertonParams> jiles_atherton;
};

[[nodiscard]] inline CatalogCore parse_core_catalog_yaml(const std::string& yaml_text) {
    YAML::Node root;
    try {
        root = YAML::Load(yaml_text);
    } catch (const std::exception& e) {
        throw std::invalid_argument(
            std::string("CatalogCore: invalid YAML: ") + e.what());
    }
    if (!root || !root.IsMap()) {
        throw std::invalid_argument("CatalogCore: root must be a map");
    }

    CatalogCore core;
    if (root["vendor"])    core.vendor   = root["vendor"].as<std::string>();
    if (root["material"])  core.material = root["material"].as<std::string>();

    if (!root["geometry"] || !root["geometry"].IsMap()) {
        throw std::invalid_argument(
            "CatalogCore: missing 'geometry:' block (area_m2, path_length_m)");
    }
    const auto geom = root["geometry"];
    if (geom["area_m2"])         core.area_m2 = geom["area_m2"].as<Real>();
    if (geom["path_length_m"])   core.path_length_m =
        geom["path_length_m"].as<Real>();
    if (!(core.area_m2 > Real{0}) || !(core.path_length_m > Real{0})) {
        throw std::invalid_argument(
            "CatalogCore: geometry.area_m2 and geometry.path_length_m must be positive");
    }

    if (!root["bh_curve"] || !root["bh_curve"].IsSequence() ||
        root["bh_curve"].size() < 2) {
        throw std::invalid_argument(
            "CatalogCore: 'bh_curve:' must be a sequence of {H, B} entries (≥ 2 points)");
    }
    std::vector<Real> H_vec;
    std::vector<Real> B_vec;
    H_vec.reserve(root["bh_curve"].size());
    B_vec.reserve(root["bh_curve"].size());
    for (const auto& point : root["bh_curve"]) {
        if (!point.IsMap() || !point["H"] || !point["B"]) {
            throw std::invalid_argument(
                "CatalogCore: every bh_curve entry must be {H: ..., B: ...}");
        }
        H_vec.push_back(point["H"].as<Real>());
        B_vec.push_back(point["B"].as<Real>());
    }
    core.bh_curve = BHCurveTable(std::move(H_vec), std::move(B_vec));

    if (root["steinmetz"] && root["steinmetz"].IsMap()) {
        SteinmetzLoss s;
        const auto& sm = root["steinmetz"];
        if (sm["k"])     s.k     = sm["k"].as<Real>();
        if (sm["alpha"]) s.alpha = sm["alpha"].as<Real>();
        if (sm["beta"])  s.beta  = sm["beta"].as<Real>();
        core.steinmetz = s;
    }

    if (root["jiles_atherton"] && root["jiles_atherton"].IsMap()) {
        JilesAthertonParams ja;
        const auto& ja_node = root["jiles_atherton"];
        if (ja_node["Ms"])    ja.Ms    = ja_node["Ms"].as<Real>();
        if (ja_node["a"])     ja.a     = ja_node["a"].as<Real>();
        if (ja_node["alpha"]) ja.alpha = ja_node["alpha"].as<Real>();
        if (ja_node["k"])     ja.k     = ja_node["k"].as<Real>();
        if (ja_node["c"])     ja.c     = ja_node["c"].as<Real>();
        core.jiles_atherton = ja;
    }

    return core;
}

[[nodiscard]] inline CatalogCore load_core_catalog_file(
    const std::filesystem::path& path) {
    std::ifstream fp(path);
    if (!fp) {
        throw std::invalid_argument(
            "CatalogCore: cannot open " + path.string());
    }
    std::ostringstream ss;
    ss << fp.rdbuf();
    return parse_core_catalog_yaml(ss.str());
}

}  // namespace pulsim::v1::magnetic
