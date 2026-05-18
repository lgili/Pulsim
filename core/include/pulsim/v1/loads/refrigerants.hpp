#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <cstdint>
#include <string_view>

namespace pulsim::v1::loads {

// =============================================================================
// Refrigerant table — curated polytropic exponent + typical cycle pressures
// =============================================================================
//
// Lookup table for the most common refrigerants used in domestic / commercial
// refrigeration and air-conditioning compressors. Each entry exposes:
//
//   - polytropic_n    : compression polytropic exponent (used by the
//                       refrigeration-compressor torque model in
//                       `loads/compressor_load.hpp`)
//   - typical_P_suction_Pa  : low-side absolute pressure (Pa) at design
//                             evaporator temperature
//   - typical_P_discharge_Pa: high-side absolute pressure (Pa) at design
//                             condenser temperature
//   - critical_temperature_K, critical_pressure_Pa: thermodynamic constants
//                             (useful for COP estimates and safe-operating
//                             window checks)
//
// Sources: ASHRAE Handbook (Refrigeration), Emerson / Embraco compressor
// datasheets, and EPA SNAP listings. Values are *typical defaults* — real
// systems vary with ambient conditions; users should override based on
// the actual evaporator / condenser design.
//
// Why a table? The previous `CompressorParams::polytropic_n` field
// required users to remember the refrigerant-specific exponent. This
// header lets you say:
//
//   loads::CompressorParams p{};
//   p.polytropic_n  = loads::refrigerant(loads::Refrigerant::R600a).polytropic_n;
//   p.P_suction_Pa  = loads::refrigerant(loads::Refrigerant::R600a).typical_P_suction_Pa;
//   p.P_discharge_Pa = loads::refrigerant(loads::Refrigerant::R600a).typical_P_discharge_Pa;
//
// or use the one-shot factory:
//
//   auto p = loads::compressor_defaults_for(loads::Refrigerant::R600a);
//   p.displacement_m3 = 8.0e-6;  // override only what differs
//
// The factory lives in `compressor_load.hpp` to avoid a circular include.

enum class Refrigerant : std::uint8_t {
    /// Isobutane (HC-600a). The modern domestic standard — replaced
    /// R134a in EU/Latin America fridges/freezers post-2015 due to
    /// near-zero GWP. Low pressure, mildly flammable (A3).
    R600a,

    /// 1,1,1,2-Tetrafluoroethane (HFC-134a). Was dominant in domestic
    /// refrigeration through the 2010s; phased out under HFC-phasedown
    /// regulations but still common in legacy units and automotive AC.
    R134a,

    /// Propane (HC-290). Hydrocarbon, near-zero GWP, mildly flammable
    /// (A3). Used in some EU domestic freezers and commercial chillers.
    R290,

    /// Difluoromethane (HFC-32). HFC blend constituent and increasingly
    /// used as a single-component fluid in residential split AC.
    /// Higher pressure than R134a, lower GWP.
    R32,

    /// CO₂ (R-744). Natural refrigerant, transcritical at typical
    /// condenser temperatures. Used in heat-pump water heaters and
    /// some commercial supermarket cascades. Very high operating
    /// pressures (≥ 80 bar discharge).
    R744,
};

struct RefrigerantProperties {
    /// Human-readable refrigerant designation.
    std::string_view name;

    /// Polytropic compression exponent used by `CompressorLoad`.
    Real polytropic_n;

    /// Typical low-side absolute pressure for the design evaporator
    /// temperature this refrigerant is normally operated at.
    Real typical_P_suction_Pa;

    /// Typical high-side absolute pressure for the design condenser
    /// temperature this refrigerant is normally operated at.
    Real typical_P_discharge_Pa;

    /// Critical temperature (K). Above this, no liquid phase exists
    /// regardless of pressure — important for CO₂ (transcritical).
    Real critical_temperature_K;

    /// Critical pressure (Pa).
    Real critical_pressure_Pa;
};

/// Look up the curated properties for a refrigerant. The returned
/// struct is a value (compiler-optimized constexpr) — feel free to
/// copy individual fields into `CompressorParams`.
[[nodiscard]] constexpr RefrigerantProperties refrigerant(
        Refrigerant r) noexcept {
    switch (r) {
        case Refrigerant::R600a:
            // Isobutane. Typical domestic fridge: T_evap ≈ −25 °C,
            // T_cond ≈ +40 °C. Saturation pressures: 0.59 / 5.30 bar.
            return RefrigerantProperties{
                /*name=*/"R600a",
                /*polytropic_n=*/1.13,
                /*typical_P_suction_Pa=*/0.59e5,
                /*typical_P_discharge_Pa=*/5.30e5,
                /*critical_temperature_K=*/407.85,
                /*critical_pressure_Pa=*/36.40e5,
            };
        case Refrigerant::R134a:
            // Tetrafluoroethane. Typical fridge: T_evap ≈ −15 °C,
            // T_cond ≈ +40 °C. Saturation: 1.64 / 10.17 bar.
            return RefrigerantProperties{
                /*name=*/"R134a",
                /*polytropic_n=*/1.30,
                /*typical_P_suction_Pa=*/1.64e5,
                /*typical_P_discharge_Pa=*/10.17e5,
                /*critical_temperature_K=*/374.21,
                /*critical_pressure_Pa=*/40.59e5,
            };
        case Refrigerant::R290:
            // Propane. Typical freezer: T_evap ≈ −25 °C, T_cond ≈ +40 °C.
            // Saturation: 2.03 / 13.69 bar.
            return RefrigerantProperties{
                /*name=*/"R290",
                /*polytropic_n=*/1.18,
                /*typical_P_suction_Pa=*/2.03e5,
                /*typical_P_discharge_Pa=*/13.69e5,
                /*critical_temperature_K=*/369.83,
                /*critical_pressure_Pa=*/42.48e5,
            };
        case Refrigerant::R32:
            // Difluoromethane. Typical residential AC: T_evap ≈ +5 °C,
            // T_cond ≈ +45 °C. Saturation: 9.51 / 27.78 bar.
            return RefrigerantProperties{
                /*name=*/"R32",
                /*polytropic_n=*/1.30,
                /*typical_P_suction_Pa=*/9.51e5,
                /*typical_P_discharge_Pa=*/27.78e5,
                /*critical_temperature_K=*/351.26,
                /*critical_pressure_Pa=*/57.82e5,
            };
        case Refrigerant::R744:
            // CO₂ (transcritical heat pump). Typical T_evap ≈ +5 °C,
            // gas-cooler outlet ≈ +35 °C, but operating above the
            // 31 °C critical point so "discharge" is gas cooler
            // pressure (~ 90 bar at typical operating points).
            return RefrigerantProperties{
                /*name=*/"R744",
                /*polytropic_n=*/1.30,
                /*typical_P_suction_Pa=*/40.0e5,
                /*typical_P_discharge_Pa=*/90.0e5,
                /*critical_temperature_K=*/304.13,
                /*critical_pressure_Pa=*/73.77e5,
            };
    }
    // Default fallback (R600a) for unrecognized enums; the
    // constexpr switch covers every enum value but compilers
    // require a return outside the switch.
    return RefrigerantProperties{
        "R600a", 1.13, 0.59e5, 5.30e5, 407.85, 36.40e5
    };
}

/// String form for diagnostics / YAML round-trip.
[[nodiscard]] constexpr std::string_view to_string(Refrigerant r) noexcept {
    return refrigerant(r).name;
}

/// Parse a refrigerant name (case-sensitive). Returns R600a (default)
/// on unknown strings; callers that want strict validation should
/// check by string-comparing the returned struct's name field.
[[nodiscard]] constexpr Refrigerant refrigerant_from_string(
        std::string_view name) noexcept {
    if (name == "R600a")  return Refrigerant::R600a;
    if (name == "R134a")  return Refrigerant::R134a;
    if (name == "R290")   return Refrigerant::R290;
    if (name == "R32")    return Refrigerant::R32;
    if (name == "R744")   return Refrigerant::R744;
    return Refrigerant::R600a;
}

}  // namespace pulsim::v1::loads
