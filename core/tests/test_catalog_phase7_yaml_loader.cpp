// =============================================================================
// Phase 7 of `add-catalog-device-models`: YAML loader + 6 reference devices
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/catalog/device_catalog_yaml.hpp"

#include <filesystem>

using namespace pulsim::v1;
using namespace pulsim::v1::catalog;
using Catch::Approx;

namespace {

[[nodiscard]] std::filesystem::path repo_devices_catalog() {
    namespace fs = std::filesystem;
    fs::path here = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        if (fs::exists(here / "devices" / "catalog")) {
            return here / "devices" / "catalog";
        }
        if (here.has_parent_path()) here = here.parent_path();
        else break;
    }
    return {};
}

}  // namespace

TEST_CASE("Phase 7: device catalog YAML loader dispatches by class:",
          "[v1][catalog][phase7][yaml][parser]") {
    const std::string yaml = R"(
class: mosfet
vendor: TestCo
part: M1
V_th_25c: 3.5
R_ds_on_25c: 0.05
)";
    const auto params = parse_device_catalog_yaml(yaml);
    REQUIRE(std::holds_alternative<MosfetCatalogParams>(params));
    const auto& mp = std::get<MosfetCatalogParams>(params);
    CHECK(mp.vendor == "TestCo");
    CHECK(mp.part_number == "M1");
    CHECK(mp.V_th_25c == Approx(3.5));
    CHECK(mp.R_ds_on_25c == Approx(0.05));
}

TEST_CASE("Phase 7: rejects missing class: and unknown class",
          "[v1][catalog][phase7][yaml][validation]") {
    CHECK_THROWS_AS(parse_device_catalog_yaml("vendor: X\n"),
                    std::invalid_argument);
    CHECK_THROWS_AS(parse_device_catalog_yaml(R"(
class: bogus
vendor: X
)"), std::invalid_argument);
}

TEST_CASE("Phase 7: 6 reference catalog YAMLs all load (gate G.4)",
          "[v1][catalog][phase7][yaml][reference][gate_G4]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());

    int n_loaded = 0;
    for (const auto& [vendor, file] : std::vector<std::pair<std::string, std::string>>{
             {"Infineon",  "IPP60R190P7.yaml"},
             {"Wolfspeed", "C3M0065090J.yaml"},
             {"GaNSystems","GS66508T.yaml"},
             {"Infineon",  "IKW40N120T2.yaml"},
             {"Wolfspeed", "C4D20120D.yaml"},
             {"Vishay",    "VS-30CTH02.yaml"},
         }) {
        const auto path = root / vendor / file;
        try {
            const auto params = load_device_catalog_file(path);
            std::visit([&](const auto& p) {
                CHECK(!p.vendor.empty());
                CHECK(!p.part_number.empty());
            }, params);
            ++n_loaded;
        } catch (const std::exception& e) {
            INFO("Failed to load " << path << ": " << e.what());
            FAIL_CHECK("could not load " << path);
        }
    }
    CHECK(n_loaded == 6);   // exceeds gate G.4 floor (≥3 of 6)
}

TEST_CASE("Phase 7: loaded MOSFET drives the catalog device end-to-end",
          "[v1][catalog][phase7][yaml][end_to_end]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());

    const auto params = load_device_catalog_file(
        root / "Wolfspeed" / "C3M0065090J.yaml");
    REQUIRE(std::holds_alternative<MosfetCatalogParams>(params));

    MosfetCatalog m{std::get<MosfetCatalogParams>(params)};
    CHECK(m.params().vendor == "Wolfspeed");
    CHECK(m.params().part_number == "C3M0065090J");

    // Drain current sanity at a typical operating point.
    const Real I_d = m.drain_current(/*V_ds*/0.1, /*V_gs*/15.0, /*Tj*/25.0);
    CHECK(I_d > 0.0);

    // Coss at 600V should be around 50 pF per the catalog.
    CHECK(m.C_oss(600.0) == Approx(50e-12).margin(1e-12));

    // Switching loss at 20A / 600V should be in the 150 µJ neighborhood
    // (per the catalog's Eon table).
    CHECK(m.switching_energy_on(20.0, 600.0) == Approx(150e-6).margin(1e-6));
}

TEST_CASE("Phase 7: IGBT and Schottky-diode reference parts load + run",
          "[v1][catalog][phase7][yaml][igbt_diode]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());

    {
        const auto params = load_device_catalog_file(
            root / "Infineon" / "IKW40N120T2.yaml");
        REQUIRE(std::holds_alternative<IgbtCatalogParams>(params));
        IgbtCatalog ig{std::get<IgbtCatalogParams>(params)};
        // V_ce_sat at 20A, 25 °C: from the catalog table = 1.85 V.
        CHECK(ig.params().V_ce_sat(20.0, 25.0) == Approx(1.85).margin(0.05));
        // Tail-current decay: drops to e⁻¹ over τ_tail.
        CHECK(ig.tail_current(20.0, 0.0) > ig.tail_current(20.0, 200e-9));
    }
    {
        const auto params = load_device_catalog_file(
            root / "Wolfspeed" / "C4D20120D.yaml");
        REQUIRE(std::holds_alternative<DiodeCatalogParams>(params));
        DiodeCatalog d{std::get<DiodeCatalogParams>(params)};
        // V_f at 5A, 25 °C: 1.30 V from the catalog.
        CHECK(d.V_f(5.0, 25.0) == Approx(1.30).margin(0.05));
        // Q_rr is small (Schottky): < 100 nC at any operating point.
        CHECK(d.reverse_recovery_charge(20.0, 1e9) < 100e-9);
    }
}
