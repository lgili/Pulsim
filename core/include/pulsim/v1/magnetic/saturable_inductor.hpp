#pragma once

#include "pulsim/v1/magnetic/bh_curve.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::magnetic {

// =============================================================================
// add-magnetic-core-models — SaturableInductor (Phase 2)
// =============================================================================
//
// A flux-linkage-state inductor whose `i(λ)` characteristic comes from a
// templated B-H curve. The model is intentionally a thin
// device-mathematics class — pure functions over geometry + curve plus
// the trapezoidal-step state advance — so it can be unit-tested in
// isolation and later embedded inside the existing `Circuit` device
// variant by a follow-up change.
//
// Mathematical reduction:
//   Faraday:  v = N · dΦ/dt = dλ/dt        (λ ≡ N·Φ, flux linkage)
//   Ampere:   N · i = H · l_e              (l_e mean magnetic path)
//   B = λ / (N · A_e)                      (A_e effective core area)
//   H = bh.h_from_b(B)
//   i(λ) = H · l_e / N
//   L_d(λ) = (di/dλ)^(-1) = (N² · A_e) / (l_e · dB/dH(H))
//
// Trapezoidal advance (companion-model-equivalent):
//   λ_{n+1} = λ_n + (dt/2) · (v_n + v_{n+1})
//
// The differential inductance L_d evaluated at λ_{n+1} is what the
// MNA companion stamp uses in place of the linear `L` constant. This
// header exposes both quantities so the integration layer (a future
// change) can stamp `g_eq = dt / (2·L_d(λ))` and feed the predictor /
// corrector pieces.

template <typename Curve>
class SaturableInductor {
public:
    /// Geometry parameters.
    struct Geometry {
        Real turns = 1.0;             ///< N (turns count, dimensionless)
        Real area = 1e-4;             ///< A_e (m²) — effective core cross section
        Real path_length = 5e-2;      ///< l_e (m) — mean magnetic path
    };

    SaturableInductor() = default;

    SaturableInductor(Geometry geom, Curve curve, std::string name = "")
        : name_(std::move(name)), geom_(geom), curve_(std::move(curve)) {}

    /// Current as a function of flux linkage λ.
    [[nodiscard]] Real current_from_flux(Real lambda) const {
        const Real B = lambda / (geom_.turns * geom_.area);
        const Real H = curve_.h_from_b(B);
        return H * geom_.path_length / geom_.turns;
    }

    /// Flux linkage λ as a function of current i — for setting initial
    /// conditions / sanity checks. Inverts `current_from_flux` via the
    /// curve's `b_from_h`.
    [[nodiscard]] Real flux_from_current(Real i) const {
        const Real H = i * geom_.turns / geom_.path_length;
        const Real B = curve_.b_from_h(H);
        return B * geom_.turns * geom_.area;
    }

    /// Differential inductance at flux λ. Needed for the MNA companion
    /// stamp (`g_eq = dt / (2·L_d)`).
    ///
    /// Derivation:  λ = N·A·B,  i = H·l_e/N
    ///   →  dλ/di = (N·A)·dB / (l_e/N · dH) = N²·A·(dB/dH)/l_e
    /// In saturation `dB/dH → 0`, so `L_d → 0` (i blows up for a small
    /// dλ change, which is the steep current-rise behavior the model
    /// exists to capture).
    [[nodiscard]] Real differential_inductance(Real lambda) const {
        const Real B = lambda / (geom_.turns * geom_.area);
        const Real H = curve_.h_from_b(B);
        const Real dbdh = curve_.dbdh(H);
        // Floor at the air-core differential inductance to keep the
        // companion stamp non-degenerate in deep saturation.
        constexpr Real mu_0 = Real{4e-7} * std::numbers::pi_v<Real>;
        const Real L_air = mu_0 * geom_.turns * geom_.turns * geom_.area /
                           geom_.path_length;
        const Real L_d = (geom_.turns * geom_.turns * geom_.area * dbdh) /
                         geom_.path_length;
        return std::max(L_d, L_air);
    }

    /// Trapezoidal advance: given the average voltage across the
    /// inductor over the step `(v_n + v_{n+1}) / 2` and the timestep dt,
    /// update the stored λ. The model is reversible — successive
    /// equal-and-opposite voltage averages cycle λ around its starting
    /// point modulo numerical drift.
    void advance_trapezoidal(Real v_average, Real dt) {
        lambda_ += dt * v_average;
    }

    /// Backward-Euler advance: simpler, slightly more dissipative.
    /// Useful for warm-up / fixed-point initialization.
    void advance_backward_euler(Real v_now, Real dt) {
        lambda_ += dt * v_now;
    }

    /// State accessors.
    [[nodiscard]] Real flux() const noexcept { return lambda_; }
    void set_flux(Real lambda) noexcept { lambda_ = lambda; }

    /// Convenience: current at current state.
    [[nodiscard]] Real current() const { return current_from_flux(lambda_); }

    /// Convenience: differential inductance at current state.
    [[nodiscard]] Real differential_inductance() const {
        return differential_inductance(lambda_);
    }

    [[nodiscard]] const Geometry& geometry() const noexcept { return geom_; }
    [[nodiscard]] const Curve& curve() const noexcept { return curve_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    std::string name_;
    Geometry geom_{};
    Curve curve_{};
    Real lambda_ = 0.0;       // flux linkage state (V·s)
};

}  // namespace pulsim::v1::magnetic
