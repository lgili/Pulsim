#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>

namespace pulsim::v1::loads {

// =============================================================================
// Refrigeration compressor load (mechanical) — pure math object
// =============================================================================
//
// Models the mechanical torque a refrigeration compressor demands from its
// driving shaft as a function of rotor angle and speed. Intended for use
// with the consolidate-motors-and-three-phase motor devices (BldcMotor,
// PmsmDevice, DcMotor) — the motor's `set_load_torque` setter is fed each
// step with the value returned by `load_torque(θ_m, ω_m)`.
//
// Three topologies are supported, covering the dominant family of
// hermetic refrigeration compressors:
//
//   - Reciprocating: piston-driven (the classic Embraco / Secop /
//     Aspera "compressor convencional" and the inverter VCC variants).
//     Strong torque ripple from the discrete suction / compression /
//     discharge / expansion strokes.
//
//   - Rotary (rolling piston): an eccentric rotor in a fixed
//     chamber. Smoother torque than reciprocating; still periodic.
//     Used in some inverter freezer / air-conditioning units.
//
//   - Scroll: two interleaved spirals. Near-constant torque with
//     small high-frequency ripple. Common in commercial chillers,
//     rare in domestic appliances.
//
// Physical model (polytropic compression):
//
//   W_indicated = P_suction · V_d / (n − 1)
//                · [(P_discharge / P_suction)^((n−1)/n) − 1]
//
// for n ≠ 1 (the polytropic exponent — 1.13 for R600a / isobutane,
// 1.30 for R134a, 1.18 for R290 / propane). For n = 1 (ideal isothermal)
// the formula degenerates to W_indicated = P_suc · V_d · ln(P_disch / P_suc).
//
// Mean torque per revolution (multiplied by number of cylinders for
// multi-piston reciprocating compressors):
//
//   T_mean = W_indicated · num_cylinders / (2π)
//
// Angle-dependent torque (topology-specific shape):
//
//   - Reciprocating: T(θ) = T_mean · (1 + α · cos(2N · θ))
//                    where N = num_cylinders (each cylinder fires twice per
//                    revolution — suction + compression strokes).
//   - Rotary:        T(θ) = T_mean · (1 + 0.2·α · cos(2 · θ))
//   - Scroll:        T(θ) = T_mean · (1 + 0.05·α · cos(8 · θ))
//
// `α` (ripple_amplitude) is in [0, 1]; defaults to 0.5 for reciprocating.
//
// Plus a Newton-style friction term:
//
//   T_friction(ω) = b_friction · ω + tau_coulomb · sign(ω)
//
// Total load: T_load(θ, ω) = T_compression(θ) + T_friction(ω). The motor's
// mechanical equation J·dω/dt = T_em − T_load(θ, ω) then closes the loop.

enum class CompressorTopology : std::uint8_t {
    Reciprocating,
    Rotary,
    Scroll,
};

[[nodiscard]] constexpr const char* to_string(CompressorTopology t) noexcept {
    switch (t) {
        case CompressorTopology::Reciprocating: return "Reciprocating";
        case CompressorTopology::Rotary:        return "Rotary";
        case CompressorTopology::Scroll:        return "Scroll";
    }
    return "Unknown";
}

struct CompressorParams {
    /// Mechanical topology (see CompressorTopology enum).
    CompressorTopology topology = CompressorTopology::Reciprocating;

    /// Number of cylinders / chambers per revolution. Most domestic
    /// reciprocating compressors are single-cylinder; some inverter
    /// scrolls are multi-stage.
    int num_cylinders = 1;

    /// Cylinder swept volume per revolution (m³). Typical Embraco
    /// domestic refrigerator: ~6 cm³ = 6e-6 m³.
    Real displacement_m3 = 6.0e-6;

    /// Suction (low-side) pressure in Pa absolute. Typical domestic
    /// fridge at standard ambient: 0.7 bar ≈ 7.0e4 Pa.
    Real P_suction_Pa = 7.0e4;

    /// Discharge (high-side) pressure in Pa absolute. Typical domestic
    /// fridge: 8 bar ≈ 8.0e5 Pa.
    Real P_discharge_Pa = 8.0e5;

    /// Polytropic compression exponent. Refrigerant-dependent:
    ///   R600a (isobutane, the modern domestic standard): 1.13
    ///   R134a: 1.30
    ///   R290 (propane):  1.18
    ///   R32: 1.30
    /// Defaults to R600a since that's the dominant domestic refrigerant
    /// post-2015 (HFC phase-out replaced R134a with R600a).
    Real polytropic_n = 1.13;

    /// Viscous friction coefficient on the shaft (N·m·s).
    Real b_friction = 1e-3;

    /// Coulomb (static) friction torque on the shaft (N·m).
    Real tau_coulomb = 0.05;

    /// Ripple amplitude as a fraction of the mean torque (0..1).
    /// Reciprocating compressors typically show ripple/mean ≈ 0.3-0.7;
    /// rotary ≈ 0.05-0.20; scroll ≈ 0.01-0.05. Set to 0 for a
    /// constant-torque idealization useful for control-loop sanity tests.
    Real ripple_amplitude = 0.5;
};

class CompressorLoad {
public:
    CompressorLoad() = default;

    explicit CompressorLoad(CompressorParams p)
        : params_(std::move(p))
    {
        recompute_mean_torque_();
    }

    /// Instantaneous shaft torque demand at given rotor angle / speed.
    /// θ_m is the mechanical angle in radians (period 2π); ω_m is the
    /// mechanical angular velocity in rad/s (signed).
    [[nodiscard]] Real load_torque(Real theta_m, Real omega_m) const {
        return compression_torque(theta_m) + friction_torque(omega_m);
    }

    /// Mean compression torque per revolution (N·m). Constant after the
    /// params struct is set — recomputed only on `set_params`.
    [[nodiscard]] Real mean_torque() const noexcept { return tau_mean_; }

    /// Indicated work per cycle per cylinder (J). Useful for energy
    /// balance / COP estimates from the simulation.
    [[nodiscard]] Real indicated_work_per_cycle() const noexcept {
        return W_ind_;
    }

    /// Reset parameters (online retune support).
    void set_params(CompressorParams p) {
        params_ = std::move(p);
        recompute_mean_torque_();
    }

    [[nodiscard]] const CompressorParams& params() const noexcept {
        return params_;
    }

private:
    [[nodiscard]] Real compression_torque(Real theta_m) const {
        const Real alpha = params_.ripple_amplitude;
        switch (params_.topology) {
            case CompressorTopology::Reciprocating: {
                // Each cylinder fires twice per revolution → 2N peaks.
                const Real N = static_cast<Real>(params_.num_cylinders);
                return tau_mean_ * (Real{1} + alpha * std::cos(Real{2} * N * theta_m));
            }
            case CompressorTopology::Rotary: {
                // Single-lobe rolling-piston: smoother, ~20% of α.
                return tau_mean_ * (Real{1} + Real{0.2} * alpha * std::cos(Real{2} * theta_m));
            }
            case CompressorTopology::Scroll: {
                // Two-spiral scroll: very smooth, ~5% of α at 8x speed.
                return tau_mean_ * (Real{1} + Real{0.05} * alpha * std::cos(Real{8} * theta_m));
            }
        }
        return tau_mean_;
    }

    [[nodiscard]] Real friction_torque(Real omega_m) const {
        const Real sign = (omega_m > Real{0}) ? Real{1}
                        : (omega_m < Real{0}) ? Real{-1} : Real{0};
        return params_.b_friction * omega_m + params_.tau_coulomb * sign;
    }

    void recompute_mean_torque_() {
        // Polytropic indicated work per cycle per cylinder.
        const Real pr = params_.P_discharge_Pa / params_.P_suction_Pa;
        const Real n  = params_.polytropic_n;
        const Real Vd = params_.displacement_m3;
        const Real Ps = params_.P_suction_Pa;
        if (std::abs(n - Real{1}) < Real{1e-6}) {
            // Isothermal limit: W = P_s · V_d · ln(P_d/P_s)
            W_ind_ = Ps * Vd * std::log(pr);
        } else {
            // Polytropic: W = P_s · V_d / (n−1) · [(P_d/P_s)^((n−1)/n) − 1]
            W_ind_ = Ps * Vd / (n - Real{1}) *
                     (std::pow(pr, (n - Real{1}) / n) - Real{1});
        }
        // Mean torque: total work × cylinder count / 2π.
        const Real W_total = W_ind_ * static_cast<Real>(params_.num_cylinders);
        constexpr Real two_pi = Real{2} * std::numbers::pi_v<Real>;
        tau_mean_ = W_total / two_pi;
    }

    CompressorParams params_{};
    Real tau_mean_ = 0.0;
    Real W_ind_    = 0.0;
};

}  // namespace pulsim::v1::loads
