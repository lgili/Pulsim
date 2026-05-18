// Refrigerant table — Catch2 unit tests.
//
// compressor-models follow-up: smoke tests for the curated refrigerant
// property table in `loads/refrigerants.hpp` and the
// `compressor_defaults_for(...)` helper.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/loads/compressor_load.hpp"
#include "pulsim/v1/loads/refrigerants.hpp"

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Refrigerant table: each entry has plausible values",
          "[loads][refrigerant]") {
    using loads::Refrigerant;

    const auto r600a = loads::refrigerant(Refrigerant::R600a);
    CHECK(r600a.name == "R600a");
    CHECK(r600a.polytropic_n == Approx(1.13));
    CHECK(r600a.typical_P_suction_Pa < r600a.typical_P_discharge_Pa);
    CHECK(r600a.critical_temperature_K > 300.0);

    const auto r134a = loads::refrigerant(Refrigerant::R134a);
    CHECK(r134a.name == "R134a");
    CHECK(r134a.polytropic_n == Approx(1.30));

    const auto r290 = loads::refrigerant(Refrigerant::R290);
    CHECK(r290.name == "R290");
    CHECK(r290.polytropic_n == Approx(1.18));

    const auto r32 = loads::refrigerant(Refrigerant::R32);
    CHECK(r32.name == "R32");
    CHECK(r32.polytropic_n == Approx(1.30));

    const auto r744 = loads::refrigerant(Refrigerant::R744);
    CHECK(r744.name == "R744");
    // CO₂ has the lowest critical temperature among these refrigerants
    // (304 K = 31 °C) — that's why heat pumps using it operate
    // transcritically.
    CHECK(r744.critical_temperature_K < 310.0);
    CHECK(r744.typical_P_discharge_Pa > 50e5);  // CO₂ runs at very high pressures
}

TEST_CASE("Refrigerant table: string round-trip",
          "[loads][refrigerant]") {
    using loads::Refrigerant;
    CHECK(loads::to_string(Refrigerant::R600a) == "R600a");
    CHECK(loads::to_string(Refrigerant::R134a) == "R134a");
    CHECK(loads::to_string(Refrigerant::R744) == "R744");

    CHECK(loads::refrigerant_from_string("R600a") == Refrigerant::R600a);
    CHECK(loads::refrigerant_from_string("R32") == Refrigerant::R32);
    // Unknown string falls back to R600a.
    CHECK(loads::refrigerant_from_string("nonsense") == Refrigerant::R600a);
}

TEST_CASE("compressor_defaults_for fills polytropic_n + cycle pressures",
          "[loads][compressor][refrigerant]") {
    using loads::Refrigerant;

    auto p = loads::compressor_defaults_for(Refrigerant::R600a);
    CHECK(p.polytropic_n == Approx(1.13));
    CHECK(p.P_suction_Pa == Approx(0.59e5));
    CHECK(p.P_discharge_Pa == Approx(5.30e5));

    // The factory leaves topology / displacement / friction at defaults.
    CHECK(p.topology == loads::CompressorTopology::Reciprocating);
    CHECK(p.num_cylinders == 1);
    CHECK(p.b_friction > 0.0);

    // The CompressorLoad built from these params should produce non-zero
    // mean torque (positive — compressing gas costs energy).
    loads::CompressorLoad load(p);
    CHECK(load.mean_torque() > 0.0);
    CHECK(load.indicated_work_per_cycle() > 0.0);
}

TEST_CASE("apply_refrigerant swaps only the refrigerant-dependent fields",
          "[loads][refrigerant]") {
    using loads::Refrigerant;

    loads::CompressorParams p{};
    p.topology = loads::CompressorTopology::Rotary;
    p.displacement_m3 = 12.0e-6;
    p.b_friction = 5e-3;

    loads::apply_refrigerant(p, Refrigerant::R290);

    // Refrigerant-driven fields updated.
    CHECK(p.polytropic_n == Approx(1.18));
    CHECK(p.P_suction_Pa == Approx(2.03e5));
    CHECK(p.P_discharge_Pa == Approx(13.69e5));
    // Non-refrigerant fields preserved.
    CHECK(p.topology == loads::CompressorTopology::Rotary);
    CHECK(p.displacement_m3 == Approx(12.0e-6));
    CHECK(p.b_friction == Approx(5e-3));
}
