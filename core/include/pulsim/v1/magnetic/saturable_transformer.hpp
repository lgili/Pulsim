#pragma once

#include "pulsim/v1/magnetic/bh_curve.hpp"
#include "pulsim/v1/magnetic/saturable_inductor.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::v1::magnetic {

// =============================================================================
// add-magnetic-core-models — SaturableTransformer (Phase 3)
// =============================================================================
//
// N-winding transformer with a single saturable magnetic core. Each
// winding has its own turns count and leakage inductance; all windings
// share a magnetizing branch driven by the core's flux linkage λ_m.
//
// State variables:
//   - λ_m            common core flux linkage (V·s)
//   - i_leak[k]      leakage-branch current per winding k
//
// Faraday on winding k:
//   v_k = (N_k / N_ref) · dλ_m/dt + L_leak[k] · di_leak[k]/dt
//
// Ampere (no leakage):
//   sum_k (N_k · i_winding[k] - N_k · i_leak[k]) = H · l_e
//                                                = (N_ref / l_e) · h_from_b(λ_m / (N_ref · A_e))
//
// We define a reference winding (typically the primary) with N_ref turns
// and treat per-winding turns as ratios. The magnetizing branch then
// sees an effective current `Σ (N_k/N_ref)·(i_winding[k] - i_leak[k])`.
// Each leakage branch is a linear inductor with its own constant L_k.
//
// Like `SaturableInductor`, this is a math object (no MNA stamp wiring)
// — Phase 4+ wires it into Circuit's device variant. The state-advance
// API is provided so unit tests + downstream integration can exercise
// the transformer independently of the simulator.

template <typename Curve>
class SaturableTransformer {
public:
    struct WindingSpec {
        Real turns         = 1.0;     ///< winding turns count
        Real leakage       = 1e-6;    ///< L_leak (H) — series with the winding
    };

    struct CoreSpec {
        Real area          = 1e-4;    ///< A_e (m²)
        Real path_length   = 5e-2;    ///< l_e (m)
    };

    SaturableTransformer() = default;

    SaturableTransformer(CoreSpec core,
                         std::vector<WindingSpec> windings,
                         Curve curve,
                         std::string name = "")
        : name_(std::move(name))
        , core_(core)
        , windings_(std::move(windings))
        , curve_(std::move(curve))
        , i_leak_(windings_.size(), Real{0}) {
        if (windings_.empty()) {
            throw std::invalid_argument(
                "SaturableTransformer: at least one winding required");
        }
        for (const auto& w : windings_) {
            if (!(w.turns > Real{0})) {
                throw std::invalid_argument(
                    "SaturableTransformer: winding turns must be positive");
            }
            if (!(w.leakage >= Real{0})) {
                throw std::invalid_argument(
                    "SaturableTransformer: leakage inductance must be ≥ 0");
            }
        }
        // Use the first winding as the reference for turns ratios. This
        // is a convention — the choice does not affect physics, only the
        // numerical conditioning of the state-space.
        N_ref_ = windings_.front().turns;
    }

    /// Magnetizing current driven by the core's λ_m. Computed from the
    /// inverse B-H curve at the current flux density and scaled into
    /// the reference winding's turns:
    ///   B = λ_m / (N_ref · A_e)
    ///   H = bh.h_from_b(B)
    ///   i_m = H · l_e / N_ref
    [[nodiscard]] Real magnetizing_current() const {
        const Real B = lambda_m_ / (N_ref_ * core_.area);
        const Real H = curve_.h_from_b(B);
        return H * core_.path_length / N_ref_;
    }

    /// Differential magnetizing inductance dλ_m/di_m at current state.
    [[nodiscard]] Real magnetizing_inductance() const {
        const Real B = lambda_m_ / (N_ref_ * core_.area);
        const Real H = curve_.h_from_b(B);
        const Real dbdh = curve_.dbdh(H);
        constexpr Real mu_0 = Real{4e-7} * std::numbers::pi_v<Real>;
        const Real L_air = mu_0 * N_ref_ * N_ref_ * core_.area /
                           core_.path_length;
        const Real L_mag = (N_ref_ * N_ref_ * core_.area * dbdh) /
                           core_.path_length;
        return std::max(L_mag, L_air);
    }

    /// Turns ratio of winding `k` relative to the reference winding.
    [[nodiscard]] Real turns_ratio(std::size_t k) const {
        return windings_.at(k).turns / N_ref_;
    }

    /// Voltage across winding `k` for a given core dλ_m/dt and the
    /// winding's own di_leak/dt. Provides the Faraday + leakage terms
    /// the MNA stamp would assemble in v_k.
    [[nodiscard]] Real winding_voltage(std::size_t k, Real dlambda_m_dt,
                                        Real di_leak_dt) const {
        return turns_ratio(k) * dlambda_m_dt +
               windings_.at(k).leakage * di_leak_dt;
    }

    /// Trapezoidal advance of the core flux (only the magnetizing
    /// branch — leakage branches are linear and tracked per winding by
    /// the integration layer that wires this transformer into a
    /// Circuit). `v_mag_avg` is the AVERAGE of the magnetizing-branch
    /// voltage over the step (as seen at the reference winding).
    void advance_core_flux_trapezoidal(Real v_mag_avg, Real dt) {
        lambda_m_ += dt * v_mag_avg;
    }

    /// Direct setters for state — used by integration layer / tests.
    void set_core_flux(Real lambda_m) noexcept { lambda_m_ = lambda_m; }
    void set_leakage_current(std::size_t k, Real i) {
        i_leak_.at(k) = i;
    }

    [[nodiscard]] Real core_flux() const noexcept { return lambda_m_; }
    [[nodiscard]] Real leakage_current(std::size_t k) const {
        return i_leak_.at(k);
    }
    [[nodiscard]] std::size_t num_windings() const noexcept {
        return windings_.size();
    }
    [[nodiscard]] const CoreSpec& core() const noexcept { return core_; }
    [[nodiscard]] const WindingSpec& winding(std::size_t k) const {
        return windings_.at(k);
    }
    [[nodiscard]] const Curve& curve() const noexcept { return curve_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    std::string name_;
    CoreSpec core_{};
    std::vector<WindingSpec> windings_;
    Curve curve_{};
    std::vector<Real> i_leak_;
    Real lambda_m_ = 0.0;
    Real N_ref_ = 1.0;
};

}  // namespace pulsim::v1::magnetic
