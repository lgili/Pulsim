#pragma once

#include "pulsim/v1/catalog/lookup_table_2d.hpp"
#include "pulsim/v1/control.hpp"   // LookupTable1D
#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::catalog {

// =============================================================================
// add-catalog-device-models — IgbtCatalog (Phase 3)
// =============================================================================
//
// IGBT model with `V_ce_sat(I_c, T_j)` lookup, tail-current decay
// post-turn-off, and switching-energy tables. Sits alongside
// `MosfetCatalog` as one of the catalog-tier devices that match
// vendor datasheets within the spec gates (G.1 / G.2).
//
// Tail current is the IGBT's signature loss mechanism — minority
// carriers in the n-base recombine over a τ_tail time scale
// (typically 100 ns – 1 µs), producing a current decay long after the
// gate turns off. The model is `i_tail(t) = I_c · exp(-(t - t_off)/τ)`,
// integrated into the switching-event loss accumulator.

struct IgbtCatalogParams {
    std::string vendor;
    std::string part_number;

    // Static parameters
    Real V_ce_sat_default = 1.5;       // V — used when V_ce_sat table is empty
    Real V_ge_th          = 5.5;       // gate threshold (V)
    Real V_ces_max        = 1200.0;    // collector-emitter blocking (V)

    // V_ce_sat(I_c, T_j) lookup. Optional — empty defaults to flat.
    LookupTable2D V_ce_sat_table;

    // Tail-current model: I_tail(t) = I_c · exp(-(t - t_off) / τ_tail)
    Real tau_tail = 200e-9;            // s (typical Si IGBT)
    Real I_tail_fraction = 0.15;       // fraction of I_c that becomes tail

    // Switching energies (J vs I_c, V_ce).
    LookupTable2D Eon;
    LookupTable2D Eoff;
    LookupTable2D Erec;                // co-pack diode reverse-recovery loss

    /// V_ce_sat at the operating point. Falls back to `V_ce_sat_default`
    /// when no table is provided (typical for first-order designs).
    [[nodiscard]] Real V_ce_sat(Real I_c, Real T_j) const {
        if (V_ce_sat_table.size_x() == 0 || V_ce_sat_table.size_y() == 0) {
            return V_ce_sat_default;
        }
        return V_ce_sat_table(I_c, T_j);
    }
};

class IgbtCatalog {
public:
    IgbtCatalog() = default;
    explicit IgbtCatalog(IgbtCatalogParams params) : params_(std::move(params)) {}

    /// Collector current under conduction. Below the gate threshold the
    /// device is off — `I_dss` floor is small and uniform in the
    /// catalog-tier model.
    [[nodiscard]] Real collector_current(Real V_ce, Real V_ge,
                                          Real T_j = 25.0) const {
        if (V_ge <= params_.V_ge_th) {
            return Real{1e-9};
        }
        // Above threshold: device behaves as a voltage-source plus
        // R_on. Approximate the on-state by I_c that produces the
        // measured V_ce_sat at this operating point.
        // For first-cut catalog use we return I_c such that
        // V_ce ≈ V_ce_sat(I_c, T_j); a closed-form inverse isn't
        // tractable, so we use the static-load assumption: I_c proportional
        // to (V_ce - 0.5V) / R_eff with R_eff ≈ V_ce_sat_default / I_c_ref.
        // The catalog YAML carries the V_ce_sat lookup so the simulator
        // resolves I_c via Newton on the device's i_v residual at the
        // outer level. This accessor is mainly for telemetry / loss.
        const Real V_ce_sat_at_unity = params_.V_ce_sat(Real{1}, T_j);
        if (V_ce_sat_at_unity <= Real{0}) {
            return Real{0};
        }
        const Real R_eff = V_ce_sat_at_unity;   // implied at I_c = 1A
        return std::max(Real{0}, V_ce / R_eff);
    }

    /// Tail-current decay: `I_tail(t_after_off) = I_c0 · I_tail_fraction
    /// · exp(-t / τ_tail)`. Used by the switching-loss accumulator to
    /// integrate tail energy over `t_after_off ∈ [0, ~5τ]`.
    [[nodiscard]] Real tail_current(Real I_c0, Real t_after_off) const {
        if (t_after_off < Real{0}) return Real{0};
        return I_c0 * params_.I_tail_fraction *
               std::exp(-t_after_off / params_.tau_tail);
    }

    [[nodiscard]] Real switching_energy_on(Real I_c, Real V_ce) const {
        return (params_.Eon.size_x() == 0) ? Real{0}
                                            : params_.Eon(I_c, V_ce);
    }
    [[nodiscard]] Real switching_energy_off(Real I_c, Real V_ce) const {
        return (params_.Eoff.size_x() == 0) ? Real{0}
                                             : params_.Eoff(I_c, V_ce);
    }
    [[nodiscard]] Real reverse_recovery_energy(Real I_f, Real V_r) const {
        return (params_.Erec.size_x() == 0) ? Real{0}
                                             : params_.Erec(I_f, V_r);
    }

    [[nodiscard]] const IgbtCatalogParams& params() const noexcept {
        return params_;
    }

private:
    IgbtCatalogParams params_;
};

}  // namespace pulsim::v1::catalog
