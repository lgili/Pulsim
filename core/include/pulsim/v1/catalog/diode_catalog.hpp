#pragma once

#include "pulsim/v1/catalog/lookup_table_2d.hpp"
#include "pulsim/v1/control.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::catalog {

// =============================================================================
// add-catalog-device-models — DiodeCatalog (Phase 4)
// =============================================================================
//
// Datasheet diode model:
//   - Forward voltage `V_f(I_f, T_j)` as a 2D lookup
//   - Reverse-recovery charge `Q_rr(I_f, di/dt)` for switching-loss
//     accumulation
//   - Optional reverse-recovery shape parameter `s_rec` controlling the
//     i(t) trajectory during recovery (ratio of fall time to t_rr)
//
// Used standalone OR embedded inside `MosfetCatalog::body_diode` /
// `IgbtCatalog::Erec`. The PWL Ideal mode (segment-primary stepper)
// bypasses the smooth I/V model entirely — it uses the catalog only
// for loss accounting via `Q_rr` lookups.

struct DiodeCatalogParams {
    std::string vendor;
    std::string part_number;

    // Static parameters
    Real V_f_default = 0.7;            // V — fallback when table empty
    Real R_on        = 1e-3;           // series on-resistance (Ω)
    Real V_r_max     = 600.0;          // reverse blocking (V)

    // V_f(I_f, T_j) lookup.
    LookupTable2D V_f_table;

    // Reverse-recovery charge `Q_rr(I_f, di/dt)`. di/dt in A/s.
    LookupTable2D Q_rr_table;

    // Reverse-recovery shape factor (default 0.5 = symmetric triangle):
    //   s_rec = t_b / t_a where t_a is the rise to peak reverse current
    //   and t_b is the fall back to zero. Soft-recovery diodes have
    //   s_rec ≈ 1.5–2.0; hard-recovery diodes ≈ 0.2–0.5.
    Real s_rec = 0.5;

    /// Forward voltage at the operating point.
    [[nodiscard]] Real V_f(Real I_f, Real T_j) const {
        if (V_f_table.size_x() == 0 || V_f_table.size_y() == 0) {
            return V_f_default + R_on * I_f;
        }
        return V_f_table(I_f, T_j);
    }

    /// Reverse-recovery charge at the switching event.
    [[nodiscard]] Real Q_rr(Real I_f, Real di_dt) const {
        if (Q_rr_table.size_x() == 0 || Q_rr_table.size_y() == 0) {
            return Real{0};
        }
        return Q_rr_table(I_f, std::abs(di_dt));
    }
};

class DiodeCatalog {
public:
    DiodeCatalog() = default;
    explicit DiodeCatalog(DiodeCatalogParams params) : params_(std::move(params)) {}

    /// Forward conduction current as a function of applied forward
    /// voltage. Solves `V = V_f(I, T_j)` with V_f modeled as
    /// `V_f_default + R_on · I` when no table is supplied — the closed
    /// form inverts trivially.
    [[nodiscard]] Real forward_current(Real V_applied, Real T_j = 25.0) const {
        if (params_.V_f_table.size_x() == 0) {
            const Real V_drop = V_applied - params_.V_f_default;
            return std::max(Real{0}, V_drop / params_.R_on);
        }
        // With a tabulated V_f(I, T_j), invert by Newton iteration on
        // the residual `V_f(I) - V_applied = 0`. For most diode tables
        // V_f is monotonically increasing in I so convergence is fast.
        Real I = std::max(Real{1e-6},
                           (V_applied - params_.V_f_default) / params_.R_on);
        for (int k = 0; k < 24; ++k) {
            const Real Vf = params_.V_f_table(I, T_j);
            const Real Vf_p = params_.V_f_table(I * Real{1.001}, T_j);
            const Real dVf_dI = (Vf_p - Vf) / (I * Real{1e-3});
            if (dVf_dI <= Real{0}) break;
            const Real dI = (Vf - V_applied) / dVf_dI;
            I = std::max(Real{1e-9}, I - dI);
            if (std::abs(dI) < I * Real{1e-6}) break;
        }
        return I;
    }

    /// Forward voltage drop at the operating point.
    [[nodiscard]] Real V_f(Real I_f, Real T_j = 25.0) const {
        return params_.V_f(I_f, T_j);
    }

    /// Reverse-recovery charge at the switching event.
    [[nodiscard]] Real reverse_recovery_charge(Real I_f, Real di_dt) const {
        return params_.Q_rr(I_f, di_dt);
    }

    /// Reverse-recovery loss energy at the switching event:
    ///   E_rec ≈ Q_rr · V_r · (1 - s_rec / (1 + s_rec))
    /// (textbook approximation; vendor datasheets typically publish a
    /// direct `E_rec` table, in which case prefer that.)
    [[nodiscard]] Real reverse_recovery_energy(Real I_f, Real di_dt,
                                                Real V_r) const {
        const Real Q = reverse_recovery_charge(I_f, di_dt);
        const Real shape = Real{1} - params_.s_rec / (Real{1} + params_.s_rec);
        return Q * V_r * shape;
    }

    [[nodiscard]] const DiodeCatalogParams& params() const noexcept {
        return params_;
    }

private:
    DiodeCatalogParams params_;
};

}  // namespace pulsim::v1::catalog
