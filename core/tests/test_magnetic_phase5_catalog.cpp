// =============================================================================
// Phase 5 of `add-magnetic-core-models`: core catalog YAML loader
// =============================================================================
//
// Pins the YAML core catalog round-trip and the four reference cores
// shipped under `devices/cores/<vendor>/<material>.yaml` — Magnetics,
// TDK, Ferroxcube, EPCOS — covering gate G.4 (≥ 3 of 4 importable).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/magnetic/core_catalog.hpp"
#include "pulsim/v1/magnetic/saturable_inductor.hpp"

#include <filesystem>

using namespace pulsim::v1;
using namespace pulsim::v1::magnetic;
using Catch::Approx;

namespace {

[[nodiscard]] std::filesystem::path repo_devices_cores() {
    // Tests run from the build dir; walk up to find `devices/cores/`.
    namespace fs = std::filesystem;
    fs::path here = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        if (fs::exists(here / "devices" / "cores")) {
            return here / "devices" / "cores";
        }
        if (here.has_parent_path()) here = here.parent_path();
        else break;
    }
    return {};
}

}  // namespace

TEST_CASE("Phase 5: core-catalog YAML round-trip parses every required field",
          "[v1][magnetic][phase5][catalog][parser]") {
    const std::string yaml = R"(
vendor: TDK
material: N87
geometry:
  area_m2: 1.5e-4
  path_length_m: 4.5e-2
bh_curve:
  - { H: -1000, B: -0.40 }
  - { H:     0, B:  0.00 }
  - { H:  1000, B:  0.40 }
steinmetz:
  k:     1.5e-3
  alpha: 1.6
  beta:  2.7
jiles_atherton:
  Ms:    3.5e5
  a:     150
  alpha: 1.0e-4
  k:     50
  c:     0.1
)";
    const CatalogCore core = parse_core_catalog_yaml(yaml);
    CHECK(core.vendor == "TDK");
    CHECK(core.material == "N87");
    CHECK(core.area_m2 == Approx(1.5e-4));
    CHECK(core.path_length_m == Approx(4.5e-2));
    CHECK(core.bh_curve.size() == 3);
    REQUIRE(core.steinmetz.has_value());
    CHECK(core.steinmetz->k == Approx(1.5e-3));
    CHECK(core.steinmetz->alpha == Approx(1.6));
    CHECK(core.steinmetz->beta == Approx(2.7));
    REQUIRE(core.jiles_atherton.has_value());
    CHECK(core.jiles_atherton->Ms == Approx(3.5e5));
    CHECK(core.jiles_atherton->a == Approx(150.0));
    CHECK(core.jiles_atherton->c == Approx(0.1));
}

TEST_CASE("Phase 5: catalog YAML rejects degenerate inputs",
          "[v1][magnetic][phase5][catalog][validation]") {
    // Missing geometry block.
    CHECK_THROWS_AS(parse_core_catalog_yaml("vendor: X\n"),
                    std::invalid_argument);
    // bh_curve too short.
    CHECK_THROWS_AS(parse_core_catalog_yaml(R"(
vendor: X
geometry:
  area_m2: 1e-4
  path_length_m: 5e-2
bh_curve:
  - { H: 0, B: 0 }
)"), std::invalid_argument);
}

TEST_CASE("Phase 5: ≥ 3 of 4 reference cores load successfully (gate G.4)",
          "[v1][magnetic][phase5][catalog][reference][gate_G4]") {
    const auto root = repo_devices_cores();
    REQUIRE_FALSE(root.empty());

    int loaded = 0;
    for (const auto& [vendor, material] : std::vector<std::pair<std::string, std::string>>{
             {"TDK",        "N87.yaml"},
             {"Ferroxcube", "3C90.yaml"},
             {"Magnetics",  "MPP_60u.yaml"},
             {"EPCOS",      "N97.yaml"},
         }) {
        const auto path = root / vendor / material;
        try {
            const CatalogCore core = load_core_catalog_file(path);
            CHECK(core.vendor == vendor);
            CHECK(core.bh_curve.size() >= 3);
            REQUIRE(core.steinmetz.has_value());
            ++loaded;
        } catch (const std::exception& e) {
            INFO("Failed to load " << path << ": " << e.what());
            FAIL_CHECK("could not load " << path);
        }
    }
    CHECK(loaded >= 3);   // gate G.4 floor
}

TEST_CASE("Phase 5: catalog core drives a SaturableInductor end-to-end",
          "[v1][magnetic][phase5][catalog][end_to_end]") {
    const auto root = repo_devices_cores();
    REQUIRE_FALSE(root.empty());
    const CatalogCore core = load_core_catalog_file(root / "TDK" / "N87.yaml");

    SaturableInductor<BHCurveTable> ind(
        {.turns = 50.0, .area = core.area_m2, .path_length = core.path_length_m},
        core.bh_curve);

    // Origin: i(0) = 0, L_d(0) > 0 (linear regime).
    CHECK(ind.current_from_flux(0.0) == Approx(0.0).margin(1e-12));
    CHECK(ind.differential_inductance(0.0) > 0.0);

    // At a flux past the catalog's rated B, i should be sizeable.
    const Real lambda_high = 0.35 * 50.0 * core.area_m2;
    CHECK(ind.current_from_flux(lambda_high) > 0.0);
}
