#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

// =============================================================================
// add-magnetic-core-models — magnetic primitives (Phase 1)
// =============================================================================
//
// B-H curves, Steinmetz core loss, iGSE for non-sinusoidal flux, and a
// Jiles-Atherton hysteresis ODE step. Used by `SaturableInductor` and
// `SaturableTransformer` (Phases 2/3) to stamp current-vs-flux
// behavior with vendor-data fidelity.
//
// All curves expose three operations:
//   - `h_from_b(B)`   forward characteristic
//   - `b_from_h(H)`   inverse characteristic (saturates per material Bs)
//   - `dbdh(H)`       differential permeability dB/dH at H, used by AD
//                     and Newton Jacobians on the device-level stamp
//
// The curves are plain values (POD-ish): they're cheap to copy / pass by
// value and have no hidden runtime dispatch. The trade-off is that you
// can't keep a heterogeneous list of `BHCurve*` — you template the
// device on the curve type instead. For a runtime-polymorphic case, wrap
// the curve in a `std::variant` at the device site.

namespace pulsim::v1::magnetic {

/// Tabulated B-H characteristic from datasheet measurements. Storage is a
/// monotone-in-H sequence of (H, B) pairs; lookups use binary search +
/// linear interpolation. Beyond the table range the curve clamps to the
/// edge value (saturation), with `dbdh` returning the slope of the last
/// segment so the Newton path stays smooth at saturation.
class BHCurveTable {
public:
    BHCurveTable() = default;

    /// Construct from parallel H and B arrays. H must be strictly
    /// increasing. Both arrays must have the same length ≥ 2.
    BHCurveTable(std::vector<Real> H, std::vector<Real> B)
        : H_(std::move(H)), B_(std::move(B)) {
        if (H_.size() != B_.size() || H_.size() < 2) {
            throw std::invalid_argument(
                "BHCurveTable: H and B must be same length and ≥ 2 points");
        }
        for (std::size_t i = 1; i < H_.size(); ++i) {
            if (!(H_[i] > H_[i-1])) {
                throw std::invalid_argument(
                    "BHCurveTable: H must be strictly increasing");
            }
        }
    }

    [[nodiscard]] Real h_from_b(Real B) const {
        // Inverse via search on the B array. B may not be strictly
        // monotone if the user provided a hysteresis loop — table curves
        // assume the upper / single-valued branch.
        const std::size_t i = std::distance(
            B_.begin(),
            std::lower_bound(B_.begin(), B_.end(), B));
        if (i == 0) return H_.front();
        if (i >= B_.size()) return H_.back();
        const Real t = (B - B_[i-1]) / (B_[i] - B_[i-1]);
        return H_[i-1] + t * (H_[i] - H_[i-1]);
    }

    [[nodiscard]] Real b_from_h(Real H) const {
        const std::size_t i = std::distance(
            H_.begin(),
            std::lower_bound(H_.begin(), H_.end(), H));
        if (i == 0) return B_.front();
        if (i >= H_.size()) return B_.back();
        const Real t = (H - H_[i-1]) / (H_[i] - H_[i-1]);
        return B_[i-1] + t * (B_[i] - B_[i-1]);
    }

    [[nodiscard]] Real dbdh(Real H) const {
        const std::size_t i = std::distance(
            H_.begin(),
            std::lower_bound(H_.begin(), H_.end(), H));
        const std::size_t lo = (i == 0) ? 0 : (i >= H_.size() ? H_.size() - 2 : i - 1);
        const std::size_t hi = lo + 1;
        return (B_[hi] - B_[lo]) / (H_[hi] - H_[lo]);
    }

    [[nodiscard]] Real saturation_density() const {
        return std::max(std::abs(B_.front()), std::abs(B_.back()));
    }

    [[nodiscard]] std::size_t size() const noexcept { return H_.size(); }

private:
    std::vector<Real> H_;
    std::vector<Real> B_;
};

/// Analytical arctan fit:  B(H) = (2·Bs/π) · arctan(H / Hc)
///
/// Smooth, monotonic, easy to invert. Good first-cut for soft ferrites
/// when the user only has Bs (saturation flux density) and Hc
/// (characteristic field at half-saturation) from the datasheet.
class BHCurveArctan {
public:
    constexpr BHCurveArctan(Real Bs, Real Hc) noexcept : Bs_(Bs), Hc_(Hc) {}

    [[nodiscard]] Real b_from_h(Real H) const noexcept {
        return (Real{2} * Bs_ / std::numbers::pi_v<Real>) * std::atan(H / Hc_);
    }

    [[nodiscard]] Real h_from_b(Real B) const noexcept {
        // Inverse: H = Hc · tan(π·B / (2·Bs))
        return Hc_ * std::tan(std::numbers::pi_v<Real> * B /
                              (Real{2} * Bs_));
    }

    [[nodiscard]] Real dbdh(Real H) const noexcept {
        // d/dH [(2Bs/π) atan(H/Hc)] = (2Bs/π) · 1/(Hc·(1 + (H/Hc)²))
        const Real x = H / Hc_;
        return (Real{2} * Bs_ / std::numbers::pi_v<Real>) /
               (Hc_ * (Real{1} + x * x));
    }

    [[nodiscard]] constexpr Real saturation_density() const noexcept { return Bs_; }

private:
    Real Bs_;
    Real Hc_;
};

/// Langevin function fit:  B(H) = Bs · L(H/a)
/// where L(x) = coth(x) − 1/x.
///
/// Better for paramagnetic / superparamagnetic materials where the
/// magnetization saturates more slowly than the arctan model predicts.
/// The Langevin function is smooth and monotonic; we use a Taylor
/// expansion `L(x) ≈ x/3 − x³/45 + ...` near zero to avoid the 0/0
/// singularity in `coth(x) − 1/x`.
class BHCurveLangevin {
public:
    constexpr BHCurveLangevin(Real Bs, Real a) noexcept : Bs_(Bs), a_(a) {}

    [[nodiscard]] Real b_from_h(Real H) const noexcept {
        return Bs_ * langevin(H / a_);
    }

    /// Inverse: solved via fixed-point iteration on `L(x) = B/Bs`. For
    /// `|B|/Bs < 1` (within saturation) this converges quickly.
    [[nodiscard]] Real h_from_b(Real B) const noexcept {
        const Real target = std::clamp(B / Bs_, Real{-0.999}, Real{0.999});
        // Initial guess via Taylor inverse `x ≈ 3·B/Bs` (small-signal).
        Real x = Real{3} * target;
        for (int i = 0; i < 16; ++i) {
            const Real f  = langevin(x) - target;
            const Real fp = langevin_derivative(x);
            if (std::abs(fp) < Real{1e-12}) break;
            x -= f / fp;
            if (std::abs(f) < Real{1e-9}) break;
        }
        return x * a_;
    }

    [[nodiscard]] Real dbdh(Real H) const noexcept {
        return (Bs_ / a_) * langevin_derivative(H / a_);
    }

    [[nodiscard]] constexpr Real saturation_density() const noexcept { return Bs_; }

private:
    Real Bs_;
    Real a_;

    [[nodiscard]] static Real langevin(Real x) noexcept {
        // L(x) = coth(x) − 1/x. Taylor-expand near x = 0 to avoid 0/0.
        if (std::abs(x) < Real{1e-3}) {
            return x / Real{3} - (x * x * x) / Real{45};
        }
        return Real{1} / std::tanh(x) - Real{1} / x;
    }

    [[nodiscard]] static Real langevin_derivative(Real x) noexcept {
        // L'(x) = 1/x² − csch²(x)
        if (std::abs(x) < Real{1e-3}) {
            return Real{1} / Real{3} - x * x / Real{15};
        }
        const Real s = std::sinh(x);
        return Real{1} / (x * x) - Real{1} / (s * s);
    }
};

// =============================================================================
// Steinmetz core loss
// =============================================================================

/// Original Steinmetz equation:  P_v = k · f^α · B_pk^β   (W / m³)
///
/// Material constants `k`, `α`, `β` are vendor-published in datasheet
/// loss curves. Typical values for ferrites: α ≈ 1.3–1.6, β ≈ 2.5–2.8.
struct SteinmetzLoss {
    Real k     = 0.0;   // material constant (units depend on f, B exponents)
    Real alpha = 1.5;   // frequency exponent
    Real beta  = 2.5;   // peak-flux exponent

    /// Cycle-averaged specific loss density at sinusoidal frequency f
    /// and peak flux density B_pk.
    [[nodiscard]] Real cycle_average(Real f, Real B_pk) const noexcept {
        return k * std::pow(f, alpha) * std::pow(std::abs(B_pk), beta);
    }
};

/// Improved Generalized Steinmetz Equation (iGSE) for **non-sinusoidal**
/// flux waveforms. Computes loss from the actual `dB/dt` time series
/// rather than assuming a single peak.
///
///   P_v = (1/T) · ∫₀ᵀ k_i · |dB/dt|^α · (ΔB)^(β-α) dt
///
/// where  k_i = k / ((2π)^(α-1) · ∫₀^{2π} |cos(θ)|^α · 2^(β-α) dθ)
/// and ΔB is the peak-to-peak swing of the cycle.
///
/// Inputs:
///   `flux_density` — uniformly-sampled B(t) over one cycle, length ≥ 4.
///   `dt`          — sample interval.
///   `params`      — Steinmetz parameters fitted from the datasheet's
///                   sinusoidal loss curves.
///
/// Returns the cycle-averaged specific loss in the same units as
/// `cycle_average`.
[[nodiscard]] inline Real igse_specific_loss(
    std::span<const Real> flux_density,
    Real dt,
    const SteinmetzLoss& params) {
    if (flux_density.size() < 4 || !(dt > Real{0})) {
        return Real{0};
    }

    // Peak-to-peak ΔB.
    Real B_min = flux_density.front();
    Real B_max = B_min;
    for (const Real B : flux_density) {
        if (B < B_min) B_min = B;
        if (B > B_max) B_max = B;
    }
    const Real delta_B = B_max - B_min;
    if (delta_B <= Real{0}) {
        return Real{0};
    }

    // k_i factor — closed-form Riemann-sum approximation for the
    // ∫|cos(θ)|^α dθ integral, accurate enough for typical α ∈ [1, 2].
    constexpr int n_cos = 256;
    Real cos_integral = Real{0};
    const Real dtheta = Real{2} * std::numbers::pi_v<Real> / Real{n_cos};
    for (int i = 0; i < n_cos; ++i) {
        const Real theta = static_cast<Real>(i) * dtheta;
        cos_integral += std::pow(std::abs(std::cos(theta)), params.alpha);
    }
    cos_integral *= dtheta;
    const Real two_pi = Real{2} * std::numbers::pi_v<Real>;
    const Real k_i = params.k /
                     (std::pow(two_pi, params.alpha - Real{1}) *
                      std::pow(Real{2}, params.beta - params.alpha) *
                      cos_integral);

    // Period and per-step contribution.
    const Real T = dt * static_cast<Real>(flux_density.size());
    const Real beta_alpha = params.beta - params.alpha;

    Real loss = Real{0};
    for (std::size_t i = 0; i + 1 < flux_density.size(); ++i) {
        const Real dBdt = (flux_density[i+1] - flux_density[i]) / dt;
        loss += std::pow(std::abs(dBdt), params.alpha) * dt;
    }
    return (k_i / T) * loss * std::pow(delta_B, beta_alpha);
}

// =============================================================================
// Jiles-Atherton hysteresis model
// =============================================================================

/// Five-parameter Jiles-Atherton hysteresis model.
///
///   M_an(H_e) = Ms · L(H_e / a)         (anhysteretic, Langevin)
///   H_e       = H + α·M                  (effective field)
///   dM_irr/dH = (M_an − M_irr) / (k·δ − α·(M_an − M_irr))
///   M         = M_irr + c·(M_an − M_irr)
///
/// Parameters (typical ranges for soft ferromagnetic materials):
///   Ms     1e5 – 1e6   A/m   saturation magnetization
///   a      10  – 1000  A/m   domain density parameter
///   alpha  1e-5 – 1e-3       inter-domain coupling
///   k      10  – 1000  A/m   pinning coefficient (loop width)
///   c      0.05 – 0.5         reversibility coefficient
struct JilesAthertonParams {
    Real Ms    = 1.0e6;
    Real a     = 100.0;
    Real alpha = 1.0e-4;
    Real k     = 100.0;
    Real c     = 0.1;
};

/// Mutable state carried across J-A integration steps.
struct JilesAthertonState {
    Real M       = 0.0;   // total magnetization
    Real M_irr   = 0.0;   // irreversible component
    Real H_prev  = 0.0;   // previous H (for dH sign)
};

/// Advance the J-A state from `state.H_prev` to `H_new` in one step.
///
/// Implements the standard Bergqvist / Jiles forward-Euler discretization.
/// The output `state.M` is the new total magnetization; the helper writes
/// back into `state` so the caller passes the same struct on the next
/// step.
inline void jiles_atherton_step(JilesAthertonState& state,
                                 const JilesAthertonParams& p,
                                 Real H_new) {
    const Real dH = H_new - state.H_prev;
    if (dH == Real{0}) {
        state.H_prev = H_new;
        return;
    }
    const Real delta = (dH > Real{0}) ? Real{1} : Real{-1};

    // Effective field, anhysteretic magnetization (Langevin).
    const Real H_e = H_new + p.alpha * state.M;
    const Real x = H_e / p.a;
    const Real L =
        (std::abs(x) < Real{1e-3})
            ? (x / Real{3} - (x * x * x) / Real{45})
            : (Real{1} / std::tanh(x) - Real{1} / x);
    const Real M_an = p.Ms * L;

    // Irreversible-magnetization update.
    const Real diff_an_irr = M_an - state.M_irr;
    const Real denom = p.k * delta - p.alpha * diff_an_irr;
    Real dM_irr = Real{0};
    if (std::abs(denom) > Real{1e-12}) {
        dM_irr = diff_an_irr * dH / denom;
    }
    state.M_irr += dM_irr;
    // Defensive clamp: forward-Euler integration of the J-A ODE can
    // drift past ±Ms with large dH steps (the model's "wipe-out
    // property" doesn't enforce the bound numerically). Clamp to keep
    // the state physically meaningful.
    state.M_irr = std::clamp(state.M_irr, -p.Ms, p.Ms);

    // Total magnetization = irreversible + reversible. Clamp the
    // total too — the reversible component is bounded by `c·(M_an −
    // M_irr)` which can momentarily push M just outside ±Ms during a
    // sign reversal.
    state.M = state.M_irr + p.c * (M_an - state.M_irr);
    state.M = std::clamp(state.M, -p.Ms, p.Ms);
    state.H_prev = H_new;
}

}  // namespace pulsim::v1::magnetic
