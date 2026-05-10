#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <numbers>

namespace pulsim::v1::templates {

// =============================================================================
// add-converter-templates — PI compensator (Phase 6.1)
// =============================================================================
//
// Discrete PI compensator with anti-windup. Used by the converter
// templates' control loop to close around `vout` against `vref`.
//
// Continuous form:   u(s) = (Kp + Ki/s) · e(s)
// Discrete form (trapezoidal):
//   u_n = u_{n-1} + Kp·(e_n - e_{n-1}) + Ki·dt·(e_n + e_{n-1})/2
//
// Output is clamped to [u_min, u_max]; integrator anti-wound-up via
// back-calculation: if the unclamped `u_n` exceeds the limits the
// integrator state is rolled back so the next step starts clean.

class PiCompensator {
public:
    struct Params {
        Real kp     = 1.0;
        Real ki     = 0.0;
        Real u_min  = 0.0;        ///< output lower clamp
        Real u_max  = 1.0;        ///< output upper clamp
    };

    PiCompensator() = default;
    explicit PiCompensator(Params p) : params_(p) {}

    /// Helper: configure the PI gains directly from a target crossover
    /// frequency `f_c` and a desired phase margin `phi_m_deg`.
    /// Approximates Kp / Ki for a one-pole plant with DC gain `K_plant`
    /// — useful for the converter templates' default-tune flow.
    static PiCompensator from_crossover(Real f_c, Real K_plant,
                                          Real u_min = 0.0, Real u_max = 1.0) {
        // For Kp = 1/(K_plant), Ki = 2π·f_c · Kp gives a unity-gain
        // crossover at f_c with ~ 90° phase margin (single-pole plant).
        // The converter templates pre-compute K_plant from their static
        // gain (e.g. 1/Vin for buck duty-to-vout) and pass it here.
        Params p;
        p.kp = (K_plant > Real{0}) ? Real{1} / K_plant : Real{1};
        p.ki = Real{2} * std::numbers::pi_v<Real> * f_c * p.kp;
        p.u_min = u_min;
        p.u_max = u_max;
        return PiCompensator{p};
    }

    /// Single-step update.  e_n = ref - measurement, dt is the control
    /// loop period.  Returns the (possibly-clamped) controller output.
    Real step(Real error, Real dt) {
        // Trapezoidal integration of the integral term.
        const Real integ = integrator_state_ +
                           params_.ki * dt * Real{0.5} *
                           (error + last_error_);
        const Real u_unclamped = params_.kp * error + integ;
        const Real u = std::clamp(u_unclamped, params_.u_min, params_.u_max);

        // Anti-windup: only commit the integrator state when the
        // unclamped output is within bounds. Otherwise hold the
        // integrator at its previous value.
        if (u == u_unclamped) {
            integrator_state_ = integ;
        }
        last_error_ = error;
        return u;
    }

    void reset(Real integrator_value = 0.0) {
        integrator_state_ = integrator_value;
        last_error_ = Real{0};
    }

    [[nodiscard]] Real integrator_state() const noexcept {
        return integrator_state_;
    }
    [[nodiscard]] const Params& params() const noexcept { return params_; }

private:
    Params params_{};
    Real integrator_state_ = 0.0;
    Real last_error_ = 0.0;
};

}  // namespace pulsim::v1::templates
