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
// Material law, in the direction materials are measured — H:
//
//     M(H) = M_s · tanh(H / H₀),   H₀ = M_s / (μ_r0 − 1)
//     B(H) = μ₀ · (H + M(H))
//
// so dB/dH at the origin is μ₀ μ_r0 exactly, M is monotone and
// bounded by M_s, and dB/dH → μ₀ from above. Everything the table
// needs is then EXPLICIT in H — no inversion in any direction:
//
//     B = μ₀(H + M),  λ = N·Ae·B,  i = [H·le + (B/μ₀)·lg]/N,
//     L = dλ/di = N²·Ae / [ le·dH/dB + lg/μ₀ ],
//     dH/dB = 1 / (μ₀ (1 + dM/dH)),   dM/dH = (μ_r0−1)·sech²(H/H₀).
//
// Two conditions any core law must meet, both held here for any
// μ_r0 > 1: (C1) dH/dB > 0, so i(λ) is monotone and L > 0; (C2)
// M non-decreasing with M → M_s, so L → L_air from above and never
// below it. A μ_r(B)-style law shipped here briefly and failed C2 —
// its implied M peaked and then decayed, the knee was far too soft
// and L dipped to 0.64 L_air. A B-parameterised M(B) tried next had
// the opposite defect (knee at 0.2 T needing 64 A). Sweeping H is
// what makes the knee land where the material puts it: at H ≈ H₀,
// a few hundred A/m for a ferrite.
//
// B_sat is the datasheet saturation flux density, identified with
// μ₀ M_s (N87: 0.49 T at 25 °C, 0.39 T at 100 °C — pass the hot
// value). The knee sharpness follows from μ_r0 and M_s; there is no
// separate shape knob.
//
// Worked numbers (ETD29-class, N = 25, Ae = 76 mm², le = 72 mm,
// lg = 0.5 mm, μ_r0 = 2000, B_sat = 0.35 T): H₀ = 139 A/m,
// R_gap/R_core = 13.9, L_unsat = 111.4 µH (A_L = 178 nH/T²),
// L_air = 0.82 µH, knee current (H = 2H₀, M = 0.96 M_s) ≈ 6.2 A,
// and by 20 A the differential inductance is air.
//
// Table extent: the sweep runs to H_max where L is within a percent
// of L_air — with tanh that is (μ_r0−1)·sech²(H/H₀) ≤ 0.01, about
// 7 H₀ — so the tail, which extrapolates at the EXACT L_air, meets
// the last knot without a jump. Knots are uniform in H, which is
// uniform in λ below the knee and dense in i above it.
//
// Not modelled, stated: fringing at the gap (effective Ae grows with
// lg; ~10 % for a 0.5 mm gap in a 10 mm leg — it enters L_unsat, so
// fit lg to the measured A_L), hysteresis (the Jiles-Atherton
// observer is a separate path), and a gap split across legs (pass
// the TOTAL gap length; only the fringing differs).

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
        Real B_sat   = Real{0.35};    //!< saturation flux density, ≡ μ₀·M_s [T]
        Real H_max   = Real{0};       //!< table extent [A/m]; 0 → where L ≈ 1.01·L_air
        Size knots   = 128;           //!< table size
    };

    [[nodiscard]] static Real M_s(const Params& c) noexcept { return c.B_sat / MU_0; }
    [[nodiscard]] static Real H_0(const Params& c) noexcept {
        return M_s(c) / (c.mu_r0 - Real{1});
    }
    [[nodiscard]] static Real magnetisation(const Params& c, Real H) noexcept {
        return M_s(c) * std::tanh(H / H_0(c));
    }
    [[nodiscard]] static Real dM_dH(const Params& c, Real H) noexcept {
        const Real t = std::tanh(H / H_0(c));
        return (c.mu_r0 - Real{1}) * (Real{1} - t * t);
    }
    [[nodiscard]] static Real B_of_H(const Params& c, Real H) noexcept {
        return MU_0 * (H + magnetisation(c, H));
    }
    [[nodiscard]] static Real flux_of_H(const Params& c, Real H) noexcept {
        return c.N * c.Ae * B_of_H(c, H);
    }
    [[nodiscard]] static Real current_of_H(const Params& c, Real H) noexcept {
        return (H * c.le + (B_of_H(c, H) / MU_0) * c.lg) / c.N;
    }
    /// Differential inductance at core field H, in closed form.
    [[nodiscard]] static Real inductance_of_H(const Params& c, Real H) noexcept {
        const Real dH_dB = Real{1} / (MU_0 * (Real{1} + dM_dH(c, H)));
        return c.N * c.N * c.Ae / (c.le * dH_dB + c.lg / MU_0);
    }
    /// Secant relative permeability B/(μ₀ H), for reporting.
    [[nodiscard]] static Real mu_r(const Params& c, Real H) noexcept {
        if (H == Real{0}) return c.mu_r0;
        return B_of_H(c, H) / (MU_0 * H);
    }

    [[nodiscard]] static Real reluctance_core(const Params& c) noexcept {
        return c.le / (MU_0 * c.mu_r0 * c.Ae);
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
    /// The knee, reported as one number: the current at H = 2H₀,
    /// where M has reached 96 % of M_s.
    [[nodiscard]] static Real knee_current(const Params& c) noexcept {
        return current_of_H(c, Real{2} * H_0(c));
    }
    /// Where L has fallen to within `tol` of L_air.
    [[nodiscard]] static Real H_of_air_approach(const Params& c, Real tol = Real{1e-2}) noexcept {
        // (L − L_air)/L_air ≈ le·dM/dH /(le + lg) ≤ tol
        const Real sech2 = tol * (c.le + c.lg) / (c.le * (c.mu_r0 - Real{1}));
        const Real t = std::sqrt(std::max(Real{0}, Real{1} - sech2));
        return std::max(Real{3}, std::atanh(std::min(t, Real{1} - Real{1e-12}))) * H_0(c);
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
        if (!(c.mu_r0 > 1)) bad("mu_r0", c.mu_r0, "initial relative permeability must be > 1");
        if (!(c.B_sat > 0)) bad("B_sat", c.B_sat, "saturation flux density must be > 0 T");
        if (c.knots < 8) bad("knots", static_cast<Real>(c.knots), "at least 8 knots");
        // Plausibility, because these are the mistakes people make
        // with units: an Ae of 76 (mm² typed as m²) or an le of 72.
        if (c.Ae > Real{1e-2}) bad("Ae", c.Ae, "> 100 cm² — is this mm² typed as m²?");
        if (c.le > Real{10}) bad("le", c.le, "> 10 m — is this mm typed as m?");
        if (c.lg > c.le) bad("lg", c.lg, "gap longer than the core path — mm typed as m?");
        if (c.B_sat > Real{3}) bad("B_sat", c.B_sat, "> 3 T exceeds every known material");
    }

    /// Tabulate λ(i) by sweeping H uniformly to H_max, with the EXACT
    /// slope at every knot and the exact air inductance as the tail.
    [[nodiscard]] static FluxTable make_table(const Params& c,
                                              const std::string& what = "GappedCore") {
        validate(c, what);
        const Real H_max = c.H_max > 0 ? c.H_max : H_of_air_approach(c);
        std::vector<Real> i, lam, L;
        i.reserve(c.knots); lam.reserve(c.knots); L.reserve(c.knots);
        for (Size k = 0; k < c.knots; ++k) {
            const Real H = H_max * static_cast<Real>(k) / static_cast<Real>(c.knots - 1);
            lam.push_back(flux_of_H(c, H));
            i.push_back(current_of_H(c, H));
            L.push_back(inductance_of_H(c, H));
        }
        return FluxTable(std::move(i), std::move(lam), std::move(L), L_air(c), what);
    }
};

}  // namespace pulsim::models
