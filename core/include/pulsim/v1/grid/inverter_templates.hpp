#pragma once

#include "pulsim/v1/grid/pll.hpp"
#include "pulsim/v1/motors/frame_transforms.hpp"
#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/templates/pi_compensator.hpp"

#include <cmath>
#include <numbers>
#include <tuple>
#include <utility>

namespace pulsim::v1::grid {

// =============================================================================
// add-three-phase-grid-library — Phase 5 + 6: inverter templates
// =============================================================================
//
// Two control structures:
//   - Grid-following: PLL locks to grid θ, dq current loops track
//     id*/iq* references derived from a P/Q command.
//   - Grid-forming: synthesizes its own θ via a P-f droop, magnitude
//     via a Q-V droop. Acts as a voltage source.
//
// Both are math objects; `step(...)` returns the modulator command
// `(V_d_ref, V_q_ref)` (grid-following) or `(V_d, V_q, θ_inv)`
// (grid-forming) that the simulator's PWM modulator + inverter
// stamp would consume.

// -----------------------------------------------------------------------------
// Phase 5 — Grid-following
// -----------------------------------------------------------------------------

struct GridFollowingParams {
    // PLL bandwidth in rad/s — converted into SrfPll Kp/Ki via the
    // unity-gain crossover rule below.
    Real pll_bandwidth_hz = 50.0;

    // Inner current loop tuning (similar to PMSM-FOC's pole-zero
    // cancellation, but the plant here is the LCL filter inductance L1
    // — the user supplies the inductance value).
    Real current_bandwidth_hz = 1000.0;
    Real L_filter             = 5e-3;     ///< grid-side (or converter-side) L
    Real R_filter             = 0.1;      ///< filter equivalent series R
    Real grid_freq_hz         = 50.0;
    Real V_grid_rms           = 230.0;    ///< per-phase RMS

    // Output command clamps — typically the DC-link voltage / sqrt(3).
    Real Vd_min = -400.0;
    Real Vd_max =  400.0;
    Real Vq_min = -400.0;
    Real Vq_max =  400.0;
};

class GridFollowingInverter {
public:
    GridFollowingInverter() = default;
    explicit GridFollowingInverter(GridFollowingParams p)
        : params_(p) {
        // PI tuning for the inner current loops:
        //   K_p = ω_c · L,  K_i = K_p · R / L  (pole-zero cancellation)
        const Real omega_c =
            Real{2} * std::numbers::pi_v<Real> * p.current_bandwidth_hz;
        templates::PiCompensator::Params pid{};
        pid.kp = omega_c * p.L_filter;
        pid.ki = pid.kp * p.R_filter / p.L_filter;
        pid.u_min = p.Vd_min;
        pid.u_max = p.Vd_max;
        pi_d_ = templates::PiCompensator(pid);

        templates::PiCompensator::Params piq{};
        piq.kp = pid.kp;
        piq.ki = pid.ki;
        piq.u_min = p.Vq_min;
        piq.u_max = p.Vq_max;
        pi_q_ = templates::PiCompensator(piq);

        // SrfPll tuned to the requested PLL bandwidth via
        //   Kp = 2·ζ·ω_pll,  Ki = ω_pll²
        // with ζ = 1/√2 (critically damped). The full transfer-function
        // derivation is in docs/three-phase-grid.md.
        const Real omega_pll =
            Real{2} * std::numbers::pi_v<Real> * p.pll_bandwidth_hz;
        const Real zeta = Real{1} / std::numbers::sqrt2_v<Real>;
        SrfPll::Params spll{};
        spll.kp = Real{2} * zeta * omega_pll / p.V_grid_rms;
        spll.ki = (omega_pll * omega_pll) / p.V_grid_rms;
        spll.freq_init = p.grid_freq_hz;
        spll.omega_min = Real{0};
        spll.omega_max = Real{4} * std::numbers::pi_v<Real> * p.grid_freq_hz;
        pll_ = SrfPll(spll);
    }

    /// One control-loop step. Inputs:
    ///   - `(va, vb, vc)` measured grid voltage (typically post-LCL on
    ///     the grid side)
    ///   - `(ia_meas, ib_meas, ic_meas)` inverter current
    ///   - `P_ref, Q_ref` active / reactive power references (W, var)
    ///   - `dt` control-loop period
    /// Returns `(Vd_ref, Vq_ref, theta_pll)` for the modulator.
    std::tuple<Real, Real, Real> step(
        Real va, Real vb, Real vc,
        Real ia, Real ib, Real ic,
        Real P_ref, Real Q_ref,
        Real dt) {
        // PLL lock to grid voltage.
        const auto [theta, _omega] = pll_.step(va, vb, vc, dt);
        (void)_omega;

        // Park current measurements into the rotor frame.
        const auto [id_meas, iq_meas] = motors::abc_to_dq(ia, ib, ic, theta);

        // Convert (P, Q) commands into (id*, iq*) references.
        // For dq-aligned-to-V (V_d = |V|, V_q = 0):
        //   P = (3/2) · V_d · i_d
        //   Q = -(3/2) · V_d · i_q
        // → id_ref = (2/3) · P / V_d ;  iq_ref = -(2/3) · Q / V_d
        const Real V_pk = params_.V_grid_rms * std::numbers::sqrt2_v<Real>;
        const Real id_ref =  (Real{2}/Real{3}) * P_ref / V_pk;
        const Real iq_ref = -(Real{2}/Real{3}) * Q_ref / V_pk;

        // PI on each axis.
        const Real Vd = pi_d_.step(id_ref - id_meas, dt);
        const Real Vq = pi_q_.step(iq_ref - iq_meas, dt);
        return {Vd, Vq, theta};
    }

    [[nodiscard]] const GridFollowingParams& params() const noexcept {
        return params_;
    }
    [[nodiscard]] const SrfPll& pll() const noexcept { return pll_; }

    void reset() {
        pi_d_.reset();
        pi_q_.reset();
        pll_.reset();
    }

private:
    GridFollowingParams params_{};
    SrfPll pll_{};
    templates::PiCompensator pi_d_{};
    templates::PiCompensator pi_q_{};
};

// -----------------------------------------------------------------------------
// Phase 6 — Grid-forming via P-f / Q-V droop
// -----------------------------------------------------------------------------

struct GridFormingParams {
    Real f_nominal_hz   = 50.0;        ///< nominal grid frequency
    Real V_nominal_rms  = 230.0;       ///< per-phase nominal RMS
    Real droop_p_f      = 0.02;        ///< Δf / Δ(P/P_rated)  (per-unit)
    Real droop_q_v      = 0.05;        ///< ΔV / Δ(Q/Q_rated)  (per-unit)
    Real P_rated        = 1e3;         ///< W
    Real Q_rated        = 1e3;         ///< var
};

class GridFormingInverter {
public:
    GridFormingInverter() = default;
    explicit GridFormingInverter(GridFormingParams p)
        : params_(p), theta_(0.0) {}

    /// One control-loop step. Inputs:
    ///   - `P_meas, Q_meas` measured active / reactive power output
    ///   - `dt` control-loop period
    /// Returns `(V_d, V_q, θ)` describing the inverter's synthesized
    /// voltage. The downstream PWM modulator uses these to drive the
    /// switching legs.
    std::tuple<Real, Real, Real> step(Real P_meas, Real Q_meas, Real dt) {
        // P-f droop: f = f_nom - droop_p_f · (P_meas / P_rated) · f_nom
        const Real f_actual = params_.f_nominal_hz *
            (Real{1} - params_.droop_p_f * P_meas / params_.P_rated);
        const Real omega = Real{2} * std::numbers::pi_v<Real> * f_actual;
        theta_ += omega * dt;
        const Real two_pi = Real{2} * std::numbers::pi_v<Real>;
        if (theta_ >= two_pi) theta_ -= two_pi;
        if (theta_ <  Real{0}) theta_ += two_pi;

        // Q-V droop: V = V_nom · (1 - droop_q_v · Q_meas / Q_rated)
        const Real V_rms = params_.V_nominal_rms *
            (Real{1} - params_.droop_q_v * Q_meas / params_.Q_rated);
        const Real V_pk = V_rms * std::numbers::sqrt2_v<Real>;

        // Inverter output is the voltage source itself: V_d = V_pk,
        // V_q = 0 (the dq frame rotates with θ_inverter, so the peak
        // phasor sits on the d axis).
        return {V_pk, Real{0}, theta_};
    }

    [[nodiscard]] Real theta() const noexcept { return theta_; }
    [[nodiscard]] const GridFormingParams& params() const noexcept { return params_; }

    void reset() { theta_ = 0.0; }

private:
    GridFormingParams params_{};
    Real theta_ = 0.0;
};

}  // namespace pulsim::v1::grid
