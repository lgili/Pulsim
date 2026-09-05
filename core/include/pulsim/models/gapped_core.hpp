#pragma once

// =============================================================================
// Pulsim — Phase 4 C.4: gapped magnetic core → λ(i) flux table
// =============================================================================
//
// A winding of N turns on a core of effective area Ae and mean path
// le, with a total air gap lg, and a soft-saturating material.
// Ampère around the mean path, flux continuous through the gap (no
// fringing), H_gap = B/μ₀:
//
//     N·i = H_core(B)·le + (B/μ₀)·lg,        B = λ / (N·Ae)
//
// so the CURRENT IS EXPLICIT IN THE FLUX. That is the direction the
// stamp does not want (it needs λ(i)), and it is also the direction in
// which the physics is trivial to evaluate — so the core is swept in
// λ, i is computed at each point, and the pairs go to a FluxTable,
// which hands λ(i) back with L = dλ/di. No inversion, no root-find.
//
// Reluctances: R_core = le/(μ₀ μ_r Ae), R_gap = lg/(μ₀ Ae), and
// L_unsat = N²/(R_core + R_gap). For a ferrite (μ_r ~ 2000,
// le ~ 70 mm) a 0.5 mm gap is already 14× the core's reluctance —
// that is the whole reason gapped cores have a stable inductance, and
// why in deep saturation (μ_r → 1) L collapses to N²μ₀Ae/(le + lg),
// the AIR value, not to zero.
//
// Material law (the default; any B(H) can be tabulated instead):
//
//     μ_r(B) = 1 + (μ_r0 − 1) / (1 + (|B|/B_sat)^p)
//
// smooth, even in B, μ_r0 at zero, → 1 far past B_sat; p sets the
// sharpness of the knee (4 is a fair ferrite; powder cores are
// softer, ~2). B_sat here is where μ_r has halved.
//
// Worked numbers (ETD29-class, N = 25, Ae = 76 mm², le = 72 mm,
// lg = 0.5 mm, μ_r0 = 2000, B_sat = 0.35 T): R_gap/R_core = 13.9,
// L_unsat = 111.4 µH (A_L = 178 nH/T²), λ(B_sat) = 0.665 mWb·t,
// i(B_sat) = 6.37 A, L_air = 0.82 µH.
//
// Not modelled, stated: fringing at the gap (effective Ae grows with
// lg; a few % for lg ≪ core width), the temperature dependence of
// B_sat (N87 loses ~25 % from 25 to 100 °C — pass the hot value),
// hysteresis (the Jiles-Atherton observer is a separate path), and a
// gap split across legs (pass the TOTAL gap length).

#include "pulsim/models/flux_table.hpp"
#include "pulsim/numeric/types.hpp"

#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::models {

inline constexpr Real MU_0 = Real{4} * std::numbers::pi_v<Real> * Real{1e-7};

struct GappedCore {
    struct Params {
        Real N       = Real{1};       //!< turns of the winding this λ(i) is referred to
        Real Ae      = Real{1e-4};    //!< effective core area [m²]
        Real le      = Real{0.1};     //!< mean magnetic path length in the core [m]
        Real lg      = Real{0};       //!< TOTAL air-gap length [m]
        Real mu_r0   = Real{2000};    //!< initial relative permeability
        Real B_sat   = Real{0.35};    //!< where μ_r has halved [T]
        Real p       = Real{4};       //!< knee sharpness
        Real B_max   = Real{0};       //!< table extent [T]; 0 → 3·B_sat
        Size knots   = 96;            //!< table size
    };

    [[nodiscard]] static Real mu_r(const Params& c, Real B) noexcept {
        const Real r = std::abs(B) / c.B_sat;
        return Real{1} + (c.mu_r0 - Real{1}) / (Real{1} + std::pow(r, c.p));
    }

    [[nodiscard]] static Real reluctance_core(const Params& c, Real B = Real{0}) noexcept {
        return c.le / (MU_0 * mu_r(c, B) * c.Ae);
    }
    [[nodiscard]] static Real reluctance_gap(const Params& c) noexcept {
        return c.lg / (MU_0 * c.Ae);
    }
    /// Small-signal inductance at rest, N²/(R_core + R_gap).
    [[nodiscard]] static Real L_unsat(const Params& c) noexcept {
        return c.N * c.N / (reluctance_core(c) + reluctance_gap(c));
    }
    /// Deep-saturation floor: the air value N²μ₀Ae/(le + lg).
    [[nodiscard]] static Real L_air(const Params& c) noexcept {
        return c.N * c.N * MU_0 * c.Ae / (c.le + c.lg);
    }

    /// The explicit direction: current for a given flux linkage.
    [[nodiscard]] static Real current_at_flux(const Params& c, Real lambda) noexcept {
        const Real B = lambda / (c.N * c.Ae);
        const Real H_core = B / (MU_0 * mu_r(c, B));
        return (H_core * c.le + (B / MU_0) * c.lg) / c.N;
    }

    static void validate(const Params& c, const std::string& what) {
        auto bad = [&](const char* name, Real v, const char* why) {
            throw std::invalid_argument(std::format(
                "{}: {} = {} — {}", what, name, v, why));
        };
        if (!(c.N > 0) || c.N != std::floor(c.N)) bad("N", c.N, "turns must be a positive integer");
        if (!(c.Ae > 0)) bad("Ae", c.Ae, "core area must be > 0 m²");
        if (!(c.le > 0)) bad("le", c.le, "mean path length must be > 0 m");
        if (!(c.lg >= 0)) bad("lg", c.lg, "gap length must be >= 0 m");
        if (!(c.mu_r0 >= 1)) bad("mu_r0", c.mu_r0, "relative permeability must be >= 1");
        if (!(c.B_sat > 0)) bad("B_sat", c.B_sat, "saturation flux density must be > 0 T");
        if (!(c.p > 0)) bad("p", c.p, "knee exponent must be > 0");
        if (c.knots < 8) bad("knots", static_cast<Real>(c.knots), "at least 8 knots");
        // Plausibility, because these are the mistakes people make
        // with units: an Ae of 76 (mm² typed as m²) or an le of 72.
        if (c.Ae > Real{1e-2}) bad("Ae", c.Ae, "> 100 cm² — is this mm² typed as m²?");
        if (c.le > Real{10}) bad("le", c.le, "> 10 m — is this mm typed as m?");
        if (c.lg > c.le) bad("lg", c.lg, "gap longer than the core path — mm typed as m?");
        if (c.B_sat > Real{3}) bad("B_sat", c.B_sat, "> 3 T exceeds every known material");
    }

    /// Tabulate λ(i) by sweeping λ uniformly up to B_max. Uniform in λ
    /// means dense in i below the knee (where L is large) and sparse
    /// above — the spacing a FluxTable wants, for free.
    [[nodiscard]] static FluxTable make_table(const Params& c,
                                              const std::string& what = "GappedCore") {
        validate(c, what);
        const Real B_max = c.B_max > 0 ? c.B_max : Real{3} * c.B_sat;
        const Real lam_max = c.N * c.Ae * B_max;
        std::vector<Real> i, lam;
        i.reserve(c.knots); lam.reserve(c.knots);
        for (Size k = 0; k < c.knots; ++k) {
            const Real l = lam_max * static_cast<Real>(k) / static_cast<Real>(c.knots - 1);
            lam.push_back(l);
            i.push_back(current_at_flux(c, l));
        }
        return FluxTable(std::move(i), std::move(lam), what);
    }
};

}  // namespace pulsim::models
