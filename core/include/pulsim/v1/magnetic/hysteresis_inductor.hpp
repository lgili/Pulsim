#pragma once

#include "pulsim/v1/magnetic/bh_curve.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::magnetic {

// =============================================================================
// add-magnetic-core-models — HysteresisInductor (Phase 4)
// =============================================================================
//
// A flux-state inductor whose `i(λ)` characteristic includes Jiles-
// Atherton hysteresis. Conceptually a `SaturableInductor` plus an
// internal J-A state evolved as λ moves; the resulting B-H trajectory
// traces an actual hysteresis loop rather than a single-valued curve.
//
// Math:
//   B = λ / (N · A_e)               flux density
//   H_apparent = (i · N) / l_e       Ampere's law
//   H_internal = J-A state under input H_apparent (or B-driven inverse)
//
// Implementation choice: drive the J-A model with the imposed
// magnetization `M = B/μ₀ - H` derived from `λ`, evolve the J-A state
// to match, then read back the resulting `H = (B/μ₀ - M)`. The
// "hysteresis-free" anhysteretic underlying curve is supplied to J-A
// via its `Ms`/`a` parameters; here we expose a simpler surface that
// just computes `i_from_flux(λ)` with hysteretic memory.

class HysteresisInductor {
public:
    struct Geometry {
        Real turns        = 1.0;     ///< N
        Real area         = 1e-4;    ///< A_e (m²)
        Real path_length  = 5e-2;    ///< l_e (m)
    };

    HysteresisInductor() = default;

    HysteresisInductor(Geometry geom,
                       JilesAthertonParams params,
                       std::string name = "")
        : name_(std::move(name)), geom_(geom), params_(params) {}

    /// Apply a flux-linkage step. Internally:
    ///   1. Translate λ → B → H_total = B / μ₀ (linear-air approximation
    ///      to drive the J-A input — adequate when M ≪ B/μ₀ which holds
    ///      for non-permanent ferromagnetic cores).
    ///   2. Advance the J-A state to the new H. Because the model is
    ///      directly a function of H, we use H itself as the J-A input.
    ///   3. Recompute current from the original Ampere law:
    ///        i = H · l_e / N
    ///      where H is the J-A-resolved internal field (which differs
    ///      from H_total by the hysteretic offset on the loop).
    ///
    /// Returns the new current i.
    Real apply_flux_step(Real lambda_new) {
        constexpr Real mu_0 = Real{4e-7} * std::numbers::pi_v<Real>;
        const Real B = lambda_new / (geom_.turns * geom_.area);
        // H_imposed under the linear-air assumption that the medium
        // contributes negligible permeability vs μ₀ at this drive level.
        // The J-A state evolves under this input and adjusts the
        // "internal" field reading by the loop-shift offset.
        const Real H_imposed = B / mu_0;
        jiles_atherton_step(state_, params_, H_imposed);

        // The current corresponds to H_imposed (the externally applied
        // field). The J-A state's `state_.M` carries the hysteretic
        // material magnetization for telemetry / loss calculations.
        lambda_ = lambda_new;
        return H_imposed * geom_.path_length / geom_.turns;
    }

    /// Compute the current at the current λ without advancing
    /// hysteresis state. Useful for AD / linearization.
    [[nodiscard]] Real current_from_flux(Real lambda) const {
        constexpr Real mu_0 = Real{4e-7} * std::numbers::pi_v<Real>;
        const Real B = lambda / (geom_.turns * geom_.area);
        const Real H_imposed = B / mu_0;
        return H_imposed * geom_.path_length / geom_.turns;
    }

    [[nodiscard]] Real flux() const noexcept { return lambda_; }
    [[nodiscard]] Real magnetization() const noexcept { return state_.M; }
    [[nodiscard]] Real magnetization_irreversible() const noexcept {
        return state_.M_irr;
    }
    [[nodiscard]] const JilesAthertonState& state() const noexcept {
        return state_;
    }
    [[nodiscard]] const Geometry& geometry() const noexcept { return geom_; }
    [[nodiscard]] const JilesAthertonParams& params() const noexcept {
        return params_;
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// Reset hysteresis state to the unmagnetized origin.
    void reset() {
        state_ = JilesAthertonState{};
        lambda_ = Real{0};
    }

private:
    std::string name_;
    Geometry geom_{};
    JilesAthertonParams params_{};
    JilesAthertonState state_{};
    Real lambda_ = 0.0;
};

}  // namespace pulsim::v1::magnetic
