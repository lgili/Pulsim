#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V16: SaturableInductor (model only — V0)
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

#include "pulsim/v2/ad/ad_scalar.hpp"
#include "pulsim/v2/numeric/concepts.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>

namespace pulsim::v2::models {

struct SaturableInductor {
    struct Params {
        Real L_0       = Real{1e-3};   // [H] unsaturated inductance
        Real I_sat     = Real{1.0};    // [A] saturation current
        Real n_exp     = Real{2.0};    // [-] saturation curve sharpness (>= 1)
        // Smoothing floor: minimum effective L to keep Jacobian
        // well-conditioned at very high i_L. Typically a few %% of L_0.
        Real L_residual = Real{0};    // [H] (0 = no floor)
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    // The "branch" is from→to (like a regular inductor). The
    // current i_L is the controlling state — but for AD purposes
    // we model L(i) as a function of a single scalar input.
    static constexpr Size num_terminals = 1;   // input: i_L (current)
    static constexpr bool is_linear = false;

    /// Effective inductance L(i) as a smooth function of the
    /// instantaneous current. Templated for AD: returns either
    /// Real (forward) or ADRealN<1> (Jacobian) depending on `S`.
    ///
    /// Terminal convention: v[0] = i_L (the magnetising current).
    /// Despite the name `current`, this is `L(i)` — the model's
    /// "output" is the effective inductance, not a branch current.
    /// The refresh helper handles the trap-rule integration.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S current(
        const S* v, const Params& p) noexcept {
        using std::pow;
        using std::abs;
        const S i_L = v[0];
        // L(i) = L_residual + (L_0 - L_residual) / (1 + (|i|/I_sat)^n)
        // This guarantees L → L_residual >= 0 for large |i|, and
        // L → L_0 for small |i|.
        const S abs_ratio = abs(i_L) / p.I_sat;
        const S denom = S{1} + pow(abs_ratio, p.n_exp);
        const S delta_L = p.L_0 - p.L_residual;
        return p.L_residual + delta_L / denom;
    }
};

}  // namespace pulsim::v2::models
