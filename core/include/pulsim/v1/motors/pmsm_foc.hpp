#pragma once

#include "pulsim/v1/motors/pmsm.hpp"
#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/templates/pi_compensator.hpp"

#include <cmath>
#include <numbers>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// add-motor-models — PMSM-FOC current-loop helper (Phase 7)
// =============================================================================
//
// Field-oriented control current loops for PMSM:
//   - id-loop: tracks i_d_ref (typically 0 for SPM, or negative for IPM
//     under field-weakening)
//   - iq-loop: tracks i_q_ref (proportional to torque demand)
//
// Per-loop PI tuned via the standard "pole-zero cancellation" rule:
//   K_p = bandwidth_rad · L_axis
//   K_i = K_p · R_s / L_axis
//
// This places the PI's zero at the plant pole (R_s/L_axis) and yields
// a unity-gain crossover at `bandwidth_hz · 2π`. The same recipe
// applies to both d-axis and q-axis with their respective inductances.

struct PmsmFocCurrentLoopParams {
    Real bandwidth_hz = 1000.0;    ///< target current-loop crossover frequency
    Real Vd_min       = -200.0;    ///< output clamp on Vd
    Real Vd_max       =  200.0;
    Real Vq_min       = -200.0;
    Real Vq_max       =  200.0;
};

/// Cascaded id / iq PI current controllers, sized from the PMSM's
/// (R_s, L_d, L_q) parameters and a target loop bandwidth.
class PmsmFocCurrentLoop {
public:
    PmsmFocCurrentLoop() = default;

    PmsmFocCurrentLoop(const PmsmParams& motor,
                       const PmsmFocCurrentLoopParams& foc_params) {
        retune(motor, foc_params);
    }

    /// Re-derive the d / q PI gains for the given motor + bandwidth
    /// budget. Useful when motor parameters drift or the user
    /// dynamically retunes the loop.
    void retune(const PmsmParams& motor,
                const PmsmFocCurrentLoopParams& foc_params) {
        const Real omega_c =
            Real{2} * std::numbers::pi_v<Real> * foc_params.bandwidth_hz;

        templates::PiCompensator::Params pd{};
        pd.kp = omega_c * motor.Ld;
        pd.ki = pd.kp * motor.Rs / motor.Ld;
        pd.u_min = foc_params.Vd_min;
        pd.u_max = foc_params.Vd_max;

        templates::PiCompensator::Params pq{};
        pq.kp = omega_c * motor.Lq;
        pq.ki = pq.kp * motor.Rs / motor.Lq;
        pq.u_min = foc_params.Vq_min;
        pq.u_max = foc_params.Vq_max;

        pi_d_ = templates::PiCompensator(pd);
        pi_q_ = templates::PiCompensator(pq);
        params_ = foc_params;
    }

    /// One control-loop step. Returns `(Vd_ref, Vq_ref)` to feed into
    /// the inverter / PMSM. `dt` is the control-loop period (typically
    /// the PWM period or some sub-multiple of it).
    [[nodiscard]] std::pair<Real, Real> step(
        Real id_ref, Real iq_ref,
        Real id_meas, Real iq_meas,
        Real dt) {
        const Real Vd = pi_d_.step(id_ref - id_meas, dt);
        const Real Vq = pi_q_.step(iq_ref - iq_meas, dt);
        return {Vd, Vq};
    }

    [[nodiscard]] const templates::PiCompensator& pi_d() const noexcept {
        return pi_d_;
    }
    [[nodiscard]] const templates::PiCompensator& pi_q() const noexcept {
        return pi_q_;
    }
    [[nodiscard]] const PmsmFocCurrentLoopParams& params() const noexcept {
        return params_;
    }

    void reset() {
        pi_d_.reset();
        pi_q_.reset();
    }

private:
    templates::PiCompensator pi_d_{};
    templates::PiCompensator pi_q_{};
    PmsmFocCurrentLoopParams params_{};
};

}  // namespace pulsim::v1::motors
