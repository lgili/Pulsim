#pragma once

#include "pulsim/v1/motors/frame_transforms.hpp"     // Park / Clarke
#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/templates/pi_compensator.hpp"

#include <cmath>
#include <cstddef>
#include <deque>
#include <numbers>
#include <utility>

namespace pulsim::v1::grid {

// =============================================================================
// add-three-phase-grid-library — Phase 3: PLLs
// =============================================================================
//
// Three PLL variants:
//   - SrfPll (synchronous-reference-frame): the classical single-PI
//     loop on `V_q` (Park-projected onto the rotor frame). Fast but
//     unbalance-sensitive.
//   - DsogiPll (Dual Second-Order Generalized Integrator): pre-filters
//     αβ via two SOGI banks to extract the positive-sequence component
//     before Park, robust against unbalance and harmonics.
//   - MafPll (Moving-Average-Filter): SrfPll with a fundamental-period
//     MAF on V_q, which kills all integer-multiple harmonics at the
//     cost of a one-cycle group delay.
//
// All three expose the same surface:
//   step(va, vb, vc, dt) → returns the locked angle θ̂ (rad) and updates
//   internal state. After "lock" the angle tracks the grid's positive-
//   sequence θ to within the PLL's tuned phase error.

class SrfPll {
public:
    struct Params {
        Real kp        = 100.0;       ///< PI proportional gain (rad/s/V)
        Real ki        = 1000.0;      ///< PI integral gain (rad/s²/V)
        Real freq_init = 50.0;        ///< nominal grid frequency (Hz)
        Real omega_min = 0.0;         ///< lower clamp on ω̂ (rad/s)
        Real omega_max = 1000.0;      ///< upper clamp on ω̂ (rad/s)
    };

    SrfPll() = default;
    explicit SrfPll(Params params)
        : params_(params),
          theta_(0.0),
          omega_(Real{2} * std::numbers::pi_v<Real> * params.freq_init) {
        templates::PiCompensator::Params pi_p{};
        pi_p.kp = params.kp;
        pi_p.ki = params.ki;
        pi_p.u_min = params.omega_min - omega_;     // PI tracks Δω
        pi_p.u_max = params.omega_max - omega_;
        pi_ = templates::PiCompensator(pi_p);
    }

    /// Drive the PLL with one (a, b, c) sample and the loop dt.
    /// Returns `(θ_locked, ω_locked)`.
    std::pair<Real, Real> step(Real va, Real vb, Real vc, Real dt) {
        // Park into the current estimated frame.
        auto [vd, vq] = motors::abc_to_dq(va, vb, vc, theta_);

        // PI on Vq → frequency correction Δω. Output of the PI is the
        // delta from the nominal `freq_init` angular speed; total ω
        // is the sum.
        const Real domega = pi_.step(/*error*/-vq, dt);
        const Real omega_nominal =
            Real{2} * std::numbers::pi_v<Real> * params_.freq_init;
        omega_ = omega_nominal + domega;

        // Integrate angle.
        theta_ += omega_ * dt;
        // Wrap into [0, 2π) to keep numerical hygiene over long runs.
        const Real two_pi = Real{2} * std::numbers::pi_v<Real>;
        if (theta_ >= two_pi) theta_ -= two_pi;
        if (theta_ <  Real{0}) theta_ += two_pi;
        return {theta_, omega_};
    }

    [[nodiscard]] Real theta() const noexcept { return theta_; }
    [[nodiscard]] Real omega() const noexcept { return omega_; }
    [[nodiscard]] const Params& params() const noexcept { return params_; }

    void reset() {
        theta_ = 0.0;
        omega_ = Real{2} * std::numbers::pi_v<Real> * params_.freq_init;
        pi_.reset();
    }

private:
    Params params_{};
    Real theta_ = 0.0;
    Real omega_ = 0.0;
    templates::PiCompensator pi_{};
};

// =============================================================================
// SOGI bank — second-order generalized integrator filter
// =============================================================================
//
// One SOGI filters a single signal x(t) into in-phase y_α and
// quadrature y_β components at frequency ω. The state-space form:
//
//   ẏ_α = ω·k·(x − y_α) − ω·y_β
//   ẏ_β = ω·y_α
//
// Tuned for `k = √2` for typical grid use (gives -3 dB at ω, with
// ~ 60° phase margin).

struct SogiState {
    Real y_alpha = 0.0;
    Real y_beta  = 0.0;
};

inline void sogi_step(SogiState& s, Real x, Real omega, Real dt,
                       Real k = Real{1.4142135623730951}) {
    const Real dy_alpha = omega * k * (x - s.y_alpha) - omega * s.y_beta;
    const Real dy_beta  = omega * s.y_alpha;
    s.y_alpha += dt * dy_alpha;
    s.y_beta  += dt * dy_beta;
}

class DsogiPll {
public:
    struct Params {
        Real kp        = 100.0;
        Real ki        = 1000.0;
        Real freq_init = 50.0;
    };

    DsogiPll() = default;
    explicit DsogiPll(Params p)
        : params_(p),
          srf_(SrfPll::Params{
              .kp = p.kp, .ki = p.ki, .freq_init = p.freq_init,
              .omega_min = 0.0,
              .omega_max = Real{4} * std::numbers::pi_v<Real> * p.freq_init,
          }) {}

    /// Step the dual-SOGI + SrfPll on a fresh (a, b, c) sample.
    std::pair<Real, Real> step(Real va, Real vb, Real vc, Real dt) {
        const auto [v_alpha, v_beta] = motors::clarke(va, vb, vc);
        // Two SOGI banks, one on α and one on β, tuned at the current
        // estimated angular frequency.
        const Real omega_est = srf_.omega();
        sogi_step(sogi_a_, v_alpha, omega_est, dt);
        sogi_step(sogi_b_, v_beta,  omega_est, dt);
        // Positive-sequence αβ from the dual-SOGI outputs:
        //   v_α+ = ½ (y_α_a − qy_β_b)
        //   v_β+ = ½ (qy_β_a + y_α_b)
        // where qy_β is the SOGI's quadrature output (y_beta).
        const Real v_alpha_pos = Real{0.5} * (sogi_a_.y_alpha - sogi_b_.y_beta);
        const Real v_beta_pos  = Real{0.5} * (sogi_a_.y_beta  + sogi_b_.y_alpha);
        // Drive the inner SrfPll on the positive-sequence αβ
        // (synthesize a balanced "fake" 3φ from the αβ pair via
        // inverse-Clarke).
        const auto [a, b, c] =
            motors::inverse_clarke(v_alpha_pos, v_beta_pos);
        return srf_.step(a, b, c, dt);
    }

    [[nodiscard]] Real theta() const noexcept { return srf_.theta(); }
    [[nodiscard]] Real omega() const noexcept { return srf_.omega(); }

    void reset() {
        srf_.reset();
        sogi_a_ = sogi_b_ = SogiState{};
    }

private:
    Params params_{};
    SrfPll srf_{};
    SogiState sogi_a_{};
    SogiState sogi_b_{};
};

// =============================================================================
// MafPll — moving-average filter on V_q kills all integer harmonics
// =============================================================================

class MafPll {
public:
    struct Params {
        Real kp        = 100.0;
        Real ki        = 1000.0;
        Real freq_init = 50.0;
        Real maf_window_periods = 1.0;    ///< window length in fundamental periods
    };

    MafPll() = default;
    explicit MafPll(Params p)
        : params_(p),
          srf_(SrfPll::Params{
              .kp = p.kp, .ki = p.ki, .freq_init = p.freq_init,
              .omega_min = 0.0,
              .omega_max = Real{4} * std::numbers::pi_v<Real> * p.freq_init,
          }) {}

    std::pair<Real, Real> step(Real va, Real vb, Real vc, Real dt) {
        // Park into the current estimated frame, MAF the v_q, then
        // feed the filtered v_q back through the SrfPll's PI as if it
        // came from a clean grid.
        auto [vd, vq] = motors::abc_to_dq(va, vb, vc, srf_.theta());
        // MAF window length in samples.
        const std::size_t N = static_cast<std::size_t>(std::max<Real>(
            Real{1},
            params_.maf_window_periods / (params_.freq_init * dt)));
        vq_buf_.push_back(vq);
        vq_sum_ += vq;
        while (vq_buf_.size() > N) {
            vq_sum_ -= vq_buf_.front();
            vq_buf_.pop_front();
        }
        const Real vq_filtered = vq_sum_ / static_cast<Real>(vq_buf_.size());

        // Drive the SrfPll's inner state using the FILTERED v_q. We
        // can't use SrfPll::step directly because it does its own
        // Park; we replicate the PI + integrate step here.
        return srf_step_with_filtered_vq(vq_filtered, dt);
    }

    [[nodiscard]] Real theta() const noexcept { return srf_.theta(); }
    [[nodiscard]] Real omega() const noexcept { return srf_.omega(); }

    void reset() {
        srf_.reset();
        vq_buf_.clear();
        vq_sum_ = 0.0;
    }

private:
    Params params_{};
    SrfPll srf_{};
    std::deque<Real> vq_buf_{};
    Real vq_sum_ = 0.0;

    /// Internal: push a (pre-Park, pre-filtered) `vq_filtered` straight
    /// into the SrfPll's PI + integrator. We achieve this by faking
    /// a 3φ signal whose Park projection at the current angle gives
    /// `(v_d = 0, v_q = vq_filtered)`.
    std::pair<Real, Real> srf_step_with_filtered_vq(Real vq_filtered, Real dt) {
        // Inverse-Park on (0, vq_filtered) at the current angle gives
        // an αβ pair whose subsequent Park (which the SrfPll does
        // internally) will reproduce the same vq.
        const auto [alpha, beta] = motors::inverse_park(
            Real{0}, vq_filtered, srf_.theta());
        const auto [a, b, c] = motors::inverse_clarke(alpha, beta);
        return srf_.step(a, b, c, dt);
    }
};

}  // namespace pulsim::v1::grid
