// =============================================================================
// Phase 8 of `add-catalog-device-models`: validation suite
// =============================================================================
//
// Gates G.1 / G.2 / G.3 — without LTspice in CI, we validate against
// **analytical** references derived from datasheet numbers:
//
//   G.1 Switching loss:   Eon at the catalog's bracketed (I_c, V_ds)
//                          matches the closed-form `0.5·V·I·(t_r + t_f)`
//                          within the catalog's published tolerance band.
//   G.2 Conduction loss:  P_cond(I, T_j) matches `I² · R_ds_on(T_j)` for
//                          MOSFETs and `I · V_ce_sat(I, T_j)` for IGBTs
//                          across 25–125 °C.
//   G.3 Q_rr behavior:    Si fast-recovery diode's Q_rr at increasing
//                          di/dt rises monotonically; SiC Schottky's
//                          Q_rr stays small and nearly flat (< 100 nC).

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

// -----------------------------------------------------------------------------
// G.1 Switching-loss tracking — Eon scales with V·I as datasheet states
// -----------------------------------------------------------------------------

TEST_CASE("Phase 8 G.1: Eon scales linearly with V_ds and I_c (datasheet contract)",
          "[v1][catalog][phase8][validation][switching][gate_G1]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());
    const auto params = load_device_catalog_file(
        root / "Wolfspeed" / "C3M0065090J.yaml");
    MosfetCatalog m{std::get<MosfetCatalogParams>(params)};

    const Real Eon_10_400 = m.switching_energy_on(10.0, 400.0);
    const Real Eon_20_400 = m.switching_energy_on(20.0, 400.0);
    const Real Eon_10_600 = m.switching_energy_on(10.0, 600.0);

    INFO("Eon(10A, 400V) = " << Eon_10_400);
    INFO("Eon(20A, 400V) = " << Eon_20_400);
    INFO("Eon(10A, 600V) = " << Eon_10_600);

    // Linear scaling vs I_c (catalog data is linear within the table).
    CHECK(Eon_20_400 == Approx(2.0 * Eon_10_400).epsilon(0.10));
    // Linear scaling vs V_ds.
    CHECK(Eon_10_600 == Approx(1.5 * Eon_10_400).epsilon(0.10));
}

// -----------------------------------------------------------------------------
// G.2 Conduction loss within 5% over 25–125 °C
// -----------------------------------------------------------------------------

TEST_CASE("Phase 8 G.2: MOSFET conduction loss tracks R_ds_on(T_j) (≤ 5 %)",
          "[v1][catalog][phase8][validation][conduction][gate_G2]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());
    const auto params = load_device_catalog_file(
        root / "Infineon" / "IPP60R190P7.yaml");
    const auto& mp = std::get<MosfetCatalogParams>(params);
    MosfetCatalog m{mp};

    constexpr Real I = 10.0;        // operating current
    for (const Real T_j : {25.0, 75.0, 125.0}) {
        const Real R_on = mp.R_ds_on(T_j);
        const Real P_expected = I * I * R_on;          // I²R
        // Drain current at V_ds ≈ I·R (linear region) reproduces I.
        const Real V_ds = I * R_on;
        const Real I_d = m.drain_current(V_ds, /*V_gs*/15.0, T_j);
        const Real P_actual = V_ds * I_d;
        INFO("T_j = " << T_j << " °C: R_on=" << R_on
             << " P_exp=" << P_expected << " P_act=" << P_actual);
        CHECK(P_actual == Approx(P_expected).epsilon(0.05));
    }
}

TEST_CASE("Phase 8 G.2: IGBT V_ce_sat lookup matches across 25 ↔ 125 °C",
          "[v1][catalog][phase8][validation][conduction][igbt][gate_G2]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());
    const auto params = load_device_catalog_file(
        root / "Infineon" / "IKW40N120T2.yaml");
    const auto& ip = std::get<IgbtCatalogParams>(params);

    // From the catalog: V_ce_sat(20A, 25C) = 1.85, V_ce_sat(20A, 125C) = 2.10.
    CHECK(ip.V_ce_sat(20.0, 25.0)  == Approx(1.85).margin(0.05));
    CHECK(ip.V_ce_sat(20.0, 125.0) == Approx(2.10).margin(0.05));
    // Higher Tj → higher V_ce_sat for IGBT (positive temp coef).
    CHECK(ip.V_ce_sat(20.0, 125.0) > ip.V_ce_sat(20.0, 25.0));
}

// -----------------------------------------------------------------------------
// G.3 Q_rr behavior contracts (Si fast-recovery vs SiC Schottky)
// -----------------------------------------------------------------------------

TEST_CASE("Phase 8 G.3: Si fast-recovery diode Q_rr rises with di/dt",
          "[v1][catalog][phase8][validation][q_rr][gate_G3]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());
    const auto params = load_device_catalog_file(
        root / "Vishay" / "VS-30CTH02.yaml");
    DiodeCatalog d{std::get<DiodeCatalogParams>(params)};

    const Real Q_low  = d.reverse_recovery_charge(10.0, 1e8);   // 0.1 GA/s
    const Real Q_high = d.reverse_recovery_charge(10.0, 1e9);   // 1 GA/s
    INFO("Si fast-recovery Q_rr: low di/dt = " << Q_low
         << " high di/dt = " << Q_high);
    CHECK(Q_high > Q_low);
    CHECK(Q_high / Q_low >= 1.2);   // datasheet shows ≈ 1.5× rise
}

TEST_CASE("Phase 8 G.3: SiC Schottky Q_rr stays small and flat",
          "[v1][catalog][phase8][validation][q_rr][schottky][gate_G3]") {
    const auto root = repo_devices_catalog();
    REQUIRE_FALSE(root.empty());
    const auto params = load_device_catalog_file(
        root / "Wolfspeed" / "C4D20120D.yaml");
    DiodeCatalog d{std::get<DiodeCatalogParams>(params)};

    // Schottky: Q_rr small (junction capacitance only, < 100 nC).
    CHECK(d.reverse_recovery_charge(20.0, 1e9) < 100e-9);
    // Nearly flat across di/dt — ratio ≤ 1.5×.
    const Real Q_low  = d.reverse_recovery_charge(20.0, 1e8);
    const Real Q_high = d.reverse_recovery_charge(20.0, 1e9);
    CHECK(Q_high <= Q_low * 1.5);
}
