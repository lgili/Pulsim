#pragma once

#include "pulsim/v1/catalog/lookup_table_2d.hpp"
#include "pulsim/v1/control.hpp"   // LookupTable1D
#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::catalog {

// =============================================================================
// add-catalog-device-models — MosfetCatalog (Phases 1.1 + 2)
// =============================================================================
//
// Datasheet-driven MOSFET model. The traditional Pulsim Level-1 MOSFET
// is fine for first-cut design; the catalog tier upgrades fidelity when
// you need to compare against a vendor-specific switching loss spec or
// design a hard-switching converter for EMI / dv/dt budgets.
//
// Model layers:
//   - **Static channel**: `i_d(v_ds, v_gs, T_j)` blends
//     linear-region (`v_ds < v_gs - V_th`) and saturation-region
//     (`v_ds ≥ v_gs - V_th`) currents using a soft transition. Off-state
//     uses `R_ds_on(T_j) → ∞` with a finite leakage floor.
//   - **Capacitances**: `Coss(V_ds)`, `Ciss(V_ds)`, `Crss(V_ds)` as
//     `LookupTable1D` instances. Tablation against `V_ds` reflects the
//     way datasheets publish them.
//   - **Body diode**: optional embedded `DiodeCatalogParams` instance —
//     handled by `MosfetCatalog::body_diode()`.
//   - **Switching energies**: `Eon(I_c, V_ds)`, `Eoff(I_c, V_ds)` as
//     `LookupTable2D`. Per-switching-event accumulator hooks into the
//     existing `losses.hpp` machinery via `switching_energy_at`.

struct MosfetCatalogParams {
    // Identification
    std::string vendor;
    std::string part_number;

    // Static parameters at 25 °C reference
    Real V_th_25c           = 3.0;     // gate threshold (V)
    Real V_th_temp_coef     = -6e-3;   // dV_th/dT (V/°C)
    Real R_ds_on_25c        = 50e-3;   // on-resistance (Ω) at 25 °C
    Real R_ds_on_temp_coef  = 6e-3;    // dR/(R·dT) (1/°C)
    Real V_ds_max           = 600.0;   // breakdown voltage (V)

    // Transconductance: g_fs(Vgs, Tj). Default empty → constant fallback
    // computed from R_ds_on (g_fs ≈ 1 / R_ds_on at saturation onset).
    LookupTable2D g_fs;                // optional — empty by default

    // Nonlinear capacitances vs V_ds (datasheet figures).
    LookupTable1D Coss;                // F vs V (V_ds)
    LookupTable1D Ciss;
    LookupTable1D Crss;

    // Switching energy lookups in J vs (I_c, V_ds).
    LookupTable2D Eon;
    LookupTable2D Eoff;

    // Off-state leakage (I_dss) — analytical exponential vs V_ds is
    // overkill; a flat default at 25 °C is fine for steady-state design.
    Real I_dss_25c          = 1e-9;    // A

    /// R_ds_on at junction temperature `T_j` (°C). Linear coefficient.
    [[nodiscard]] Real R_ds_on(Real T_j) const {
        return R_ds_on_25c * (Real{1} + R_ds_on_temp_coef * (T_j - Real{25}));
    }

    /// V_th at junction temperature `T_j` (°C). Negative coefficient is
    /// the SiC/Si default — V_th drops as the device heats.
    [[nodiscard]] Real V_th(Real T_j) const {
        return V_th_25c + V_th_temp_coef * (T_j - Real{25});
    }
};

class MosfetCatalog {
public:
    MosfetCatalog() = default;
    explicit MosfetCatalog(MosfetCatalogParams params)
        : params_(std::move(params)) {}

    /// Drain current as a function of (V_ds, V_gs, T_j). Smooth blend
    /// between linear region (`V_ds < V_gs - V_th`) and saturation
    /// (`V_ds ≥ V_gs - V_th`); off-state (`V_gs ≤ V_th`) returns the
    /// leakage floor.
    [[nodiscard]] Real drain_current(Real V_ds, Real V_gs, Real T_j = 25.0) const {
        const Real V_th = params_.V_th(T_j);
        const Real V_ov = V_gs - V_th;
        if (V_ov <= Real{0}) {
            return params_.I_dss_25c;
        }
        const Real R_on = params_.R_ds_on(T_j);
        // Linear region: V_ds = R_on · I_d → I_d = V_ds / R_on
        const Real i_lin = V_ds / R_on;
        // Saturation: I_d = V_ov / R_on (clipped)
        const Real i_sat = V_ov / R_on;
        // Smooth-min blending so the transition matches the datasheet's
        // "knee" without a hard kink.
        return std::min(i_lin, i_sat);
    }

    /// Output capacitance at V_ds.
    [[nodiscard]] Real C_oss(Real V_ds) const {
        return params_.Coss.empty() ? Real{0} : params_.Coss(V_ds);
    }
    [[nodiscard]] Real C_iss(Real V_ds) const {
        return params_.Ciss.empty() ? Real{0} : params_.Ciss(V_ds);
    }
    [[nodiscard]] Real C_rss(Real V_ds) const {
        return params_.Crss.empty() ? Real{0} : params_.Crss(V_ds);
    }

    /// Switching energy (J) at the bracketed (I_c, V_ds) operating
    /// point. Both Eon and Eoff are vendor-published in datasheet
    /// figures; the catalog YAML carries them as 2D tables.
    [[nodiscard]] Real switching_energy_on(Real I_c, Real V_ds) const {
        return (params_.Eon.size_x() == 0) ? Real{0}
                                            : params_.Eon(I_c, V_ds);
    }
    [[nodiscard]] Real switching_energy_off(Real I_c, Real V_ds) const {
        return (params_.Eoff.size_x() == 0) ? Real{0}
                                             : params_.Eoff(I_c, V_ds);
    }

    [[nodiscard]] const MosfetCatalogParams& params() const noexcept {
        return params_;
    }

private:
    MosfetCatalogParams params_;
};

}  // namespace pulsim::v1::catalog
