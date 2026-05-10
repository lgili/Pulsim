// =============================================================================
// Phases 1-4 of `add-catalog-device-models`: primitives + Mosfet/Igbt/Diode
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/catalog/lookup_table_2d.hpp"
#include "pulsim/v1/catalog/mosfet_catalog.hpp"
#include "pulsim/v1/catalog/igbt_catalog.hpp"
#include "pulsim/v1/catalog/diode_catalog.hpp"

#include <cmath>

using namespace pulsim::v1;
using namespace pulsim::v1::catalog;
using Catch::Approx;

// -----------------------------------------------------------------------------
// Phase 1: LookupTable2D
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: LookupTable2D bilinear interp + clamp",
          "[v1][catalog][phase1][lookup_table_2d]") {
    // 2x2 table:  z(0,0)=0,  z(1,0)=1,  z(0,1)=2,  z(1,1)=3
    LookupTable2D t({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0, 3.0});

    CHECK(t(0.0, 0.0) == Approx(0.0).margin(1e-12));
    CHECK(t(1.0, 0.0) == Approx(1.0).margin(1e-12));
    CHECK(t(0.0, 1.0) == Approx(2.0).margin(1e-12));
    CHECK(t(1.0, 1.0) == Approx(3.0).margin(1e-12));
    CHECK(t(0.5, 0.5) == Approx(1.5).margin(1e-12));
    CHECK(t(0.5, 0.0) == Approx(0.5).margin(1e-12));
    CHECK(t(0.0, 0.5) == Approx(1.0).margin(1e-12));

    // Clamp.
    CHECK(t(-10.0, -10.0) == Approx(0.0).margin(1e-12));
    CHECK(t(100.0,   0.0) == Approx(1.0).margin(1e-12));
    CHECK(t(100.0, 100.0) == Approx(3.0).margin(1e-12));
}

TEST_CASE("Phase 1: LookupTable2D rejects degenerate input",
          "[v1][catalog][phase1][lookup_table_2d][validation]") {
    CHECK_THROWS_AS(LookupTable2D({0.0}, {0.0}, {0.0}), std::invalid_argument);
    CHECK_THROWS_AS(LookupTable2D({0.0, 1.0}, {0.0, 1.0}, {0.0, 0.0, 0.0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        LookupTable2D({1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0, 0.0, 0.0}),
        std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Phase 2: MosfetCatalog
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2: MosfetCatalog drain current respects Vth and R_ds_on(Tj)",
          "[v1][catalog][phase2][mosfet]") {
    MosfetCatalogParams p;
    p.vendor = "Test";
    p.part_number = "M1";
    p.V_th_25c = 3.0;
    p.V_th_temp_coef = -6e-3;
    p.R_ds_on_25c = 50e-3;
    p.R_ds_on_temp_coef = 6e-3;

    MosfetCatalog m(p);

    // Off-state: Vgs below Vth → leakage floor.
    CHECK(m.drain_current(10.0, 1.0, 25.0) == Approx(p.I_dss_25c).margin(1e-15));

    // Linear region: Vgs = 10V, Vds = 0.1V → I_d = Vds / R_on(25C) = 2 A.
    CHECK(m.drain_current(0.1, 10.0, 25.0) ==
          Approx(0.1 / 50e-3).margin(1e-9));

    // Saturation: Vgs = 5V (Vov = 2V), Vds = 5V → I_d = Vov / R_on = 40 A.
    CHECK(m.drain_current(5.0, 5.0, 25.0) ==
          Approx(2.0 / 50e-3).margin(1e-9));

    // Temperature dependence: R_ds_on at 100°C = R_25 · (1 + 6e-3 · 75)
    //   = 50e-3 · 1.45 = 72.5 mΩ → I_d at Vds=0.1V drops to 0.1/0.0725.
    const Real I_at_25  = m.drain_current(0.1, 10.0, 25.0);
    const Real I_at_100 = m.drain_current(0.1, 10.0, 100.0);
    CHECK(I_at_100 < I_at_25);
    CHECK(p.R_ds_on(100.0) == Approx(p.R_ds_on_25c * 1.45).margin(1e-9));
}

TEST_CASE("Phase 2: MosfetCatalog Coss/Ciss/Crss interpolate from datasheet",
          "[v1][catalog][phase2][mosfet][caps]") {
    MosfetCatalogParams p;
    // Datasheet-shape: capacitance falls off as Vds rises (depletion).
    p.Coss = LookupTable1D({0.0, 50.0, 200.0, 600.0},
                            {2e-9, 500e-12, 100e-12, 40e-12});
    p.Ciss = LookupTable1D({0.0, 600.0}, {3e-9, 3e-9});      // ≈ flat
    p.Crss = LookupTable1D({0.0, 200.0, 600.0},
                            {500e-12, 50e-12, 20e-12});

    MosfetCatalog m(p);

    CHECK(m.C_oss(0.0)   == Approx(2e-9).margin(1e-15));
    CHECK(m.C_oss(50.0)  == Approx(500e-12).margin(1e-15));
    CHECK(m.C_oss(600.0) == Approx(40e-12).margin(1e-15));
    // Mid-range bilinear: at Vds=300V: between (200, 100pF) and (600, 40pF).
    //   t = (300-200)/(600-200) = 0.25, C = 100p + 0.25·(40-100)p = 85 pF.
    CHECK(m.C_oss(300.0) == Approx(85e-12).margin(1e-15));

    CHECK(m.C_iss(123.0) == Approx(3e-9).margin(1e-15));
}

TEST_CASE("Phase 2: MosfetCatalog switching-energy lookup",
          "[v1][catalog][phase2][mosfet][switching_energy]") {
    MosfetCatalogParams p;
    // 2×2 Eon table at (I_c × V_ds): identity-like for testing.
    p.Eon = LookupTable2D(
        /* I_c axis */ {10.0, 20.0},
        /* V_ds axis */ {200.0, 400.0},
        /* values     */ {1e-6, 2e-6,    // (10, 200)=1µJ, (20, 200)=2µJ
                          2e-6, 4e-6});   // (10, 400)=2µJ, (20, 400)=4µJ
    MosfetCatalog m(p);

    CHECK(m.switching_energy_on(10.0, 200.0) == Approx(1e-6));
    CHECK(m.switching_energy_on(20.0, 400.0) == Approx(4e-6));
    CHECK(m.switching_energy_on(15.0, 300.0) == Approx(2.25e-6).margin(1e-9));

    // Empty Eoff → 0
    CHECK(m.switching_energy_off(10.0, 200.0) == Approx(0.0));
}

// -----------------------------------------------------------------------------
// Phase 3: IgbtCatalog
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3: IgbtCatalog tail current decays with τ_tail",
          "[v1][catalog][phase3][igbt][tail]") {
    IgbtCatalogParams p;
    p.tau_tail = 200e-9;
    p.I_tail_fraction = 0.15;
    IgbtCatalog ig(p);

    const Real I_c0 = 50.0;
    // At t=0 immediately after off: tail = 15% of I_c0 = 7.5 A.
    CHECK(ig.tail_current(I_c0, 0.0) == Approx(7.5).margin(1e-9));
    // At t = τ: 1/e of the t=0 value.
    CHECK(ig.tail_current(I_c0, p.tau_tail) ==
          Approx(7.5 / std::exp(1.0)).margin(1e-9));
    // At t = 5τ: ≈ 0.
    CHECK(ig.tail_current(I_c0, 5.0 * p.tau_tail) < 0.1);
}

TEST_CASE("Phase 3: IgbtCatalog V_ce_sat falls back to default when no table",
          "[v1][catalog][phase3][igbt][vce_sat]") {
    IgbtCatalogParams p;
    p.V_ce_sat_default = 1.5;
    IgbtCatalog ig(p);

    CHECK(ig.params().V_ce_sat(20.0, 25.0) == Approx(1.5));
    CHECK(ig.collector_current(0.0, 1.0) == Approx(1e-9));     // off (Vge < Vge_th)
    CHECK(ig.collector_current(2.0, 12.0) > 0.0);               // on (above threshold)
}

// -----------------------------------------------------------------------------
// Phase 4: DiodeCatalog
// -----------------------------------------------------------------------------

TEST_CASE("Phase 4: DiodeCatalog forward voltage matches default + R_on",
          "[v1][catalog][phase4][diode]") {
    DiodeCatalogParams p;
    p.V_f_default = 0.7;
    p.R_on = 10e-3;
    DiodeCatalog d(p);

    CHECK(d.V_f(1.0, 25.0) == Approx(0.7 + 10e-3).margin(1e-12));
    CHECK(d.V_f(10.0, 25.0) == Approx(0.7 + 0.1).margin(1e-12));

    // Inversion: forward_current(V_applied) inverts the linear V_f model.
    CHECK(d.forward_current(0.7, 25.0)  == Approx(0.0).margin(1e-9));
    CHECK(d.forward_current(0.71, 25.0) == Approx(1.0).margin(1e-6));
    CHECK(d.forward_current(0.72, 25.0) == Approx(2.0).margin(1e-6));
}

TEST_CASE("Phase 4: DiodeCatalog Q_rr returns 0 when table empty",
          "[v1][catalog][phase4][diode][q_rr]") {
    DiodeCatalog d{DiodeCatalogParams{}};
    CHECK(d.reverse_recovery_charge(10.0, 1e9) == Approx(0.0));
    CHECK(d.reverse_recovery_energy(10.0, 1e9, 400.0) == Approx(0.0));
}

TEST_CASE("Phase 4: DiodeCatalog Q_rr lookup + recovery-energy estimate",
          "[v1][catalog][phase4][diode][recovery]") {
    DiodeCatalogParams p;
    // Q_rr table: at I_f=10A, di_dt=1e9 A/s → Q_rr = 50 nC.
    p.Q_rr_table = LookupTable2D(
        {1.0, 10.0},
        {1e8, 1e9},
        {5e-9, 1e-8,
         5e-8, 5e-8});
    p.s_rec = 0.5;     // symmetric triangle
    DiodeCatalog d(p);

    CHECK(d.reverse_recovery_charge(10.0, 1e9) == Approx(5e-8).margin(1e-12));
    // E_rec = Q · V · (1 - s/(1+s)) = 5e-8 · 400 · (1 - 0.5/1.5)
    //       = 5e-8 · 400 · 0.6667 ≈ 1.333e-5
    const Real E_rec = d.reverse_recovery_energy(10.0, 1e9, 400.0);
    CHECK(E_rec == Approx(5e-8 * 400.0 * (1.0 - 0.5 / 1.5)).epsilon(1e-3));
}
