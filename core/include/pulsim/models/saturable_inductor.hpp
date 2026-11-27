#pragma once

// =============================================================================
// Pulsim — Layer 2 V16: SaturableInductor (model only — V0)
// =============================================================================
//
// Nonlinear inductor whose effective inductance smoothly drops
// as the magnetising current exceeds a saturation threshold.
// Model:
//
//   L(i) = L_residual + (L_0 − L_residual) / (1 + (|i|/I_sat)^n)
//
// with `n = 2` by default (soft saturation, C¹ smooth for
// Newton stability). Higher `n` → sharper knee.
//
// SCOPE OF THIS COMMIT (V16): MODEL FUNCTION ONLY.
//   * `L(i)` is exposed as a templated `current<S>()` function
//     usable by AD machinery (returns Real or ADRealN<1>).
//   * Unit tests validate the shape of L(i): monotonic decrease,
//     L → L_0 at i → 0, L → L_residual at i → ∞.
//
// NOT YET SHIPPED (deferred to V17):
//   * Full transient Newton integration. The trap-companion
//     stamping for a NONLINEAR L would require:
//       (a) Newton refresh function evaluating L(i_L) per iter
//       (b) HistoryState extension to track i_L_old + V_L_old
//           for each saturable branch
//       (c) Layer 4 cache invalidation / lazy re-stamp when
//           L_eff changes between timesteps
//     These are non-trivial. V17 work.
//
// Until V17 ships, users can:
//   * Query L(i) directly for sizing calculations
//   * Approximate saturation in transient by manually setting
//     a regular linear inductor's L to L(I_peak_expected)
//   * Or just use this as a documentation pattern for future
//     extension

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/models/flux_table.hpp"
#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace pulsim::models {

struct SaturableInductor {
    struct Params {
        Real L_0       = Real{1e-3};   // [H] unsaturated inductance
        Real I_sat     = Real{1.0};    // [A] saturation current
        // The smoothing exponent is FIXED at n=2 (Atan-shape),
        // the canonical SMPS-magnetics curve. This keeps the
        // model AD-friendly (no pow needed — just (i/I_sat)²).
        // Smoothing floor: minimum effective L at high |i|.
        Real L_residual = Real{0};    // [H] (0 = no floor)
        // Phase 4 C.4 — an alternative LAW. When set, λ(i) and L(i)
        // come from this monotone table (a gapped core, a datasheet
        // curve, a JA anhysteretic) and the three fields above are
        // ignored except as reported metadata. Null = the analytic
        // atan law. A pointer, not a table by value: Params lives in
        // the pool's variant and is copied into the history on every
        // snapshot, and a 96-knot table by value would be copied per
        // step. Trailing with a default so every brace-init compiles.
        std::shared_ptr<const FluxTable> table{};
        // Phase 4 C.4 — the HYSTERETIC law (Jiles-Atherton in the
        // loop). Unlike the two above it is history-dependent: λ(i)
        // needs the step-start (H, M), which lives in the history
        // entry, so `flux()` / `inductance()` cannot serve it — the
        // stamp evaluates it through `JilesAthertonCore::evaluate`
        // with that state. `table` and `ja` are mutually exclusive.
        std::shared_ptr<const JilesAthertonCore::Params> ja{};
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 1;
    static constexpr bool is_linear = false;

    /// Effective inductance L(i) = L_residual +
    /// (L_0 − L_residual) / (1 + (i/I_sat)²)
    ///
    /// Atan-shape saturation. C¹ smooth. Symmetric in i.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S current(
        const S* v, const Params& p) noexcept {
        const S i_L = v[0];
        const S ratio = i_L / p.I_sat;
        const S denom = S{1} + ratio * ratio;
        const S delta_L = p.L_0 - p.L_residual;
        return p.L_residual + delta_L / denom;
    }

    /// FLUX LINKAGE λ(i) = ∫₀ⁱ L(u) du — the device's actual state.
    ///
    ///   λ(i) = L_residual·i
    ///          + (L_0 − L_residual)·I_sat·atan(i / I_sat)
    ///
    /// The comment above calls this an "Atan-shape" saturation, and
    /// that is literally true: the flux IS an arctangent. Because
    /// L is even in i, λ is odd; λ(0) = 0; and dλ/di = L(i) exactly,
    /// since d/di [I_sat·atan(i/I_sat)] = 1/(1 + (i/I_sat)²).
    ///
    /// WHY THIS EXISTS. A trapezoidal step on v = dλ/dt is
    ///
    ///     λ(i_new) − λ(i_old) = (h/2)·(v_new + v_old)
    ///
    /// and the stamp used to write the left-hand side as
    /// L(i_new)·(i_new − i_old) — a RIGHT-ENDPOINT RECTANGLE RULE,
    /// exact only while L is constant across the step. It is not a
    /// symmetric error, because the stamp solves for the current
    /// increment given the voltage:
    ///
    ///     Δi = h·(v_new + v_old) / (2·L(i_new))
    ///
    /// Ascending, L(i_new) is the SMALLEST L on the interval, so Δi
    /// comes out too large. Descending, it is the LARGEST, so |Δi|
    /// comes out too small. Both push the current outward: the error
    /// rectifies rather than cancels. Measured on a 1 kHz zero-mean
    /// sine at five thousand steps per cycle, L_0 = 1 mH,
    /// I_sat = 5 A, the DC current climbed 63.6 → 72.3 → 104.7 →
    /// 145.5 A over 10 → 40 → 160 → 400 cycles, from a source with
    /// no DC at all. First order in h, unbounded in time. A linear
    /// inductor in the same circuit drifts 7e-15 A per cycle.
    ///
    /// Limits, both exact: as I_sat → ∞, I_sat·atan(i/I_sat) → i, so
    /// λ → L_0·i; as L_residual → L_0, λ = L_0·i directly. The
    /// non-saturating device reduces to the linear one with no
    /// residue.
    ///
    /// Plain `Real` rather than templated on the AD scalar: the only
    /// derivative anyone needs is dλ/di, and that is `current()`
    /// above, analytically. Routing λ through AD would demand an
    /// `atan` overload on the AD scalar to compute a value the model
    /// already knows in closed form.
    [[nodiscard]] static Real flux(Real i_L, const Params& p) noexcept {
        if (p.table) return p.table->flux(i_L);
        return p.L_residual * i_L
               + (p.L_0 - p.L_residual) * p.I_sat
                     * std::atan(i_L / p.I_sat);
    }

    /// λ(i1) − λ(i0) of the atan law without forming either flux:
    /// atan a − atan b = atan((a − b)/(1 + ab)) for ab > −1, so the
    /// increment is accurate RELATIVE TO ITSELF. The difference of two
    /// fluxes carries eps·|λ|; a residual that multiplies it by 2/h
    /// (1e15 at a femtosecond landing step) turned that ulp into 1e-7 V
    /// of Newton noise on a saturable transformer, above the 1e-9 V
    /// absolute step tolerance, and which side of it a run fell on was
    /// decided by the last bit (TR-BDF2 aborted at h_min on Linux CI,
    /// and in 2 of 22 rounding-fuzzed runs on macOS). Monotone in i1:
    /// d/di1 of (a − b)/(1 + ab) is (1 + b²)/(1 + ab)² > 0. The atan law
    /// only (`table` and `ja` are not closed forms).
    [[nodiscard]] static Real flux_increment(Real i1, Real i0,
                                             const Params& p) noexcept {
        const Real a = i1 / p.I_sat;
        const Real b = i0 / p.I_sat;
        const Real den = Real{1} + a * b;
        const Real datan = den > Real{0}
            ? std::atan(((i1 - i0) / p.I_sat) / den)
            : std::atan(a) - std::atan(b);   // opposite signs: no cancellation
        return p.L_residual * (i1 - i0)
               + (p.L_0 - p.L_residual) * p.I_sat * datan;
    }

    /// L(i) = dλ/di, for the law in force — the EXACT derivative of
    /// the same expression `flux()` evaluates, table or atan. The
    /// stamp reads this, not `current<Real>()`, so a table-backed
    /// device cannot end up with a Jacobian from one law and a
    /// residual from another.
    [[nodiscard]] static Real inductance(Real i_L, const Params& p) noexcept {
        if (p.table) return p.table->inductance(i_L);
        return current<Real>(&i_L, p);
    }

    /// i(λ): the inverse of the stateless laws (atan, table), the twin of
    /// NonlinearCapacitor::voltage_of_charge — bracketed Newton from
    /// i_guess. NaN when no current reaches λ (L_residual = 0 saturates the
    /// atan law at ±(π/2)·L_0·I_sat; a table's flat end).
    [[nodiscard]] static Real current_of_flux(const Params& p, Real lam,
                                              Real i_guess) noexcept {
        const Real step0 = Real{1e-3} * std::max(std::abs(i_guess), std::abs(p.I_sat));
        Real lo = i_guess, hi = i_guess, step = step0;
        for (int k = 0; flux(lo, p) > lam; ++k, step *= 2) {
            if (k == 200) return std::numeric_limits<Real>::quiet_NaN();
            lo -= step;
        }
        step = step0;
        for (int k = 0; flux(hi, p) < lam; ++k, step *= 2) {
            if (k == 200) return std::numeric_limits<Real>::quiet_NaN();
            hi += step;
        }
        Real i = std::clamp(i_guess, lo, hi);
        for (int k = 0; k < 100; ++k) {
            const Real f = flux(i, p) - lam;
            if (f == Real{0}) return i;
            (f > Real{0} ? hi : lo) = i;
            Real in = i - f / inductance(i, p);
            if (!(in > lo && in < hi)) in = Real{0.5} * (lo + hi);
            if (std::abs(in - i) <= Real{1e-15} * (Real{1} + std::abs(i))) return in;
            i = in;
        }
        return i;
    }

    /// The energy a flux-conserving jump a → b (currents) dissipates in the
    /// path that takes it: ∫ₐᵇ (λ(i) − λ(a)) di (½·L·(b − a)² for a linear
    /// law; a → 0 gives the stored ∫ i dλ). The twin of
    /// NonlinearCapacitor::jump_energy, same quadrature.
    /// shortcut: fixed Gauss–Legendre (5 points × 32 panels), no error
    /// estimate; upgrade to adaptive if a table's kinks need it.
    [[nodiscard]] static Real jump_energy(const Params& p, Real a, Real b) noexcept {
        constexpr Real xg[5] = {0.0, -0.5384693101056831, 0.5384693101056831,
                                -0.9061798459386640, 0.9061798459386640};
        constexpr Real wg[5] = {0.5688888888888889, 0.4786286704993665,
                                0.4786286704993665, 0.2369268850561891,
                                0.2369268850561891};
        constexpr int n = 32;
        const Real hp = (b - a) / n;
        const Real la = flux(a, p);
        Real s = 0;
        for (int j = 0; j < n; ++j) {
            const Real c = a + (j + Real{0.5}) * hp;
            for (int g = 0; g < 5; ++g) {
                const Real u = c + Real{0.5} * hp * xg[g];
                s += wg[g] * (p.table ? flux(u, p) - la : flux_increment(u, a, p));
            }
        }
        return Real{0.5} * hp * s;
    }
};

}  // namespace pulsim::models
