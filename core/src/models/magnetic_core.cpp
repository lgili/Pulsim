#include "pulsim/models/magnetic_core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulsim::models {

// Physical constants
constexpr Real MU0 = 4.0 * M_PI * 1e-7;  // Vacuum permeability (H/m)

// =============================================================================
// Jiles-Atherton Model Implementation
// =============================================================================

JilesAthertonModel::JilesAthertonModel(const JilesAthertonParams& params,
                                       const CoreGeometry& geometry)
    : params_(params), geometry_(geometry), M_(params.M0), H_prev_(0.0) {}

void JilesAthertonModel::reset() {
    M_ = params_.M0;
    H_prev_ = 0.0;
}

Real JilesAthertonModel::compute_He(Real H, Real M) const {
    // Effective field including domain interaction
    return H + params_.alpha * M;
}

Real JilesAthertonModel::compute_Man(Real He) const {
    // Langevin function for anhysteretic magnetization
    // Man = Ms * (coth(He/a) - a/He)
    if (std::abs(He) < 1e-10) {
        return 0.0;
    }

    Real x = He / params_.a;
    if (std::abs(x) > 20.0) {
        // Large argument: coth(x) ≈ sign(x)
        return params_.Ms * (x > 0 ? 1.0 : -1.0);
    }

    // Langevin function: L(x) = coth(x) - 1/x
    Real coth_x = 1.0 / std::tanh(x);
    return params_.Ms * (coth_x - 1.0 / x);
}

Real JilesAthertonModel::compute_dM_dH(Real H, Real M, Real Man, int delta) const {
    // Differential equation for magnetization
    // dM/dH = (Man - M) / (k*delta - alpha*(Man - M)) + c * dMan/dHe * (1 + alpha*dM/dH)

    Real He = compute_He(H, M);
    Real diff = Man - M;

    // Irreversible component denominator
    Real denom_irr = params_.k * delta - params_.alpha * diff;

    // Prevent division by zero
    if (std::abs(denom_irr) < 1e-10) {
        denom_irr = 1e-10 * (denom_irr >= 0 ? 1 : -1);
    }

    // Derivative of anhysteretic magnetization
    Real dMan_dHe;
    Real x = He / params_.a;
    if (std::abs(x) < 1e-10) {
        dMan_dHe = params_.Ms / (3.0 * params_.a);
    } else if (std::abs(x) > 20.0) {
        dMan_dHe = 0.0;
    } else {
        Real sinh_x = std::sinh(x);
        dMan_dHe = params_.Ms / params_.a * (1.0 / (x * x) - 1.0 / (sinh_x * sinh_x));
    }

    // Irreversible component
    Real dM_dH_irr = diff / denom_irr;

    // Reversible component
    Real dM_dH_rev = params_.c * dMan_dHe;

    // Total (solving implicit equation)
    // dM/dH = dM_dH_irr + dM_dH_rev * (1 + alpha * dM/dH)
    // dM/dH * (1 - dM_dH_rev * alpha) = dM_dH_irr + dM_dH_rev
    Real factor = 1.0 - dM_dH_rev * params_.alpha;
    if (std::abs(factor) < 1e-10) {
        factor = 1e-10;
    }

    return (dM_dH_irr + dM_dH_rev) / factor;
}

MagneticCoreOpPoint JilesAthertonModel::evaluate(Real H, Real dH_dt) {
    (void)dH_dt;  // For future extension (dynamic effects)

    MagneticCoreOpPoint op;
    op.H = H;

    // Determine direction of H change
    Real dH = H - H_prev_;
    int delta = (dH >= 0) ? 1 : -1;

    // Compute effective field and anhysteretic magnetization
    Real He = compute_He(H, M_);
    Real Man = compute_Man(He);

    // Update magnetization using simple forward Euler
    if (std::abs(dH) > 1e-12) {
        // Check energy consistency: irreversible magnetization change only when
        // moving away from anhysteretic curve
        Real diff = Man - M_;
        bool allow_irreversible = (delta * diff >= 0);

        Real dM_dH;
        if (allow_irreversible) {
            // Full Jiles-Atherton equation
            dM_dH = compute_dM_dH(H, M_, Man, delta);
        } else {
            // Only reversible component (moving toward anhysteretic)
            Real x = He / params_.a;
            Real dMan_dHe;
            if (std::abs(x) < 1e-10) {
                dMan_dHe = params_.Ms / (3.0 * params_.a);
            } else if (std::abs(x) > 20.0) {
                dMan_dHe = 0.0;
            } else {
                Real sinh_x = std::sinh(x);
                dMan_dHe = params_.Ms / params_.a * (1.0 / (x * x) - 1.0 / (sinh_x * sinh_x));
            }
            dM_dH = params_.c * dMan_dHe / (1.0 - params_.c * dMan_dHe * params_.alpha);
        }

        // Ensure dM has correct sign (magnetization should follow H direction for virgin curve)
        Real dM = dM_dH * dH;

        // Limit dM/dH to prevent numerical instability
        Real max_dM = params_.Ms * 0.1;  // Max change per step
        if (std::abs(dM) > max_dM) {
            dM = max_dM * (dM >= 0 ? 1 : -1);
        }

        M_ += dM;

        // Clamp magnetization
        M_ = std::clamp(M_, -params_.Ms, params_.Ms);
    }

    // Compute B = μ0 * (H + M)
    op.M = M_;
    op.B = MU0 * (H + M_);

    // Differential permeability
    He = compute_He(H, M_);
    Man = compute_Man(He);
    op.dB_dH = MU0 * (1.0 + compute_dM_dH(H, M_, Man, delta));
    op.mu = op.dB_dH;

    // Flux
    op.flux = op.B * geometry_.Ae;

    // Stored energy (approximate)
    op.energy = 0.5 * op.B * H * geometry_.effective_volume();

    // Update previous H
    H_prev_ = H;

    return op;
}

// =============================================================================
// PWL Core Model Implementation
// =============================================================================

PWLCoreParams PWLCoreParams::default_ferrite() {
    PWLCoreParams params;
    params.curve = {
        {0.0, 0.0},
        {10.0, 0.1},
        {50.0, 0.25},
        {100.0, 0.35},
        {200.0, 0.40},
        {500.0, 0.45},
        {1000.0, 0.48},
        {5000.0, 0.50}
    };
    params.symmetric = true;
    return params;
}

PWLCoreParams PWLCoreParams::default_silicon_steel() {
    PWLCoreParams params;
    params.curve = {
        {0.0, 0.0},
        {40.0, 0.5},
        {80.0, 1.0},
        {160.0, 1.4},
        {400.0, 1.6},
        {1200.0, 1.8},
        {4000.0, 1.95},
        {10000.0, 2.0}
    };
    params.symmetric = true;
    return params;
}

PWLCoreParams PWLCoreParams::default_amorphous() {
    PWLCoreParams params;
    params.curve = {
        {0.0, 0.0},
        {5.0, 0.3},
        {10.0, 0.6},
        {20.0, 1.0},
        {40.0, 1.3},
        {80.0, 1.45},
        {200.0, 1.55},
        {1000.0, 1.6}
    };
    params.symmetric = true;
    return params;
}

PWLCoreModel::PWLCoreModel(const PWLCoreParams& params, const CoreGeometry& geometry)
    : params_(params), geometry_(geometry) {
    sort_curve();
}

void PWLCoreModel::sort_curve() {
    std::sort(params_.curve.begin(), params_.curve.end(),
              [](const PWLCorePoint& a, const PWLCorePoint& b) {
                  return a.H < b.H;
              });
}

MagneticCoreOpPoint PWLCoreModel::evaluate(Real H) const {
    MagneticCoreOpPoint op;
    op.H = H;

    Real H_abs = std::abs(H);
    Real sign = (H >= 0) ? 1.0 : -1.0;

    if (params_.curve.empty()) {
        // Linear with relative permeability of 1000
        op.B = MU0 * 1000.0 * H;
        op.dB_dH = MU0 * 1000.0;
    } else if (params_.curve.size() == 1) {
        op.B = sign * params_.curve[0].B;
        op.dB_dH = MU0;
    } else {
        // Find segment
        size_t i = 0;
        for (; i < params_.curve.size() - 1; ++i) {
            if (H_abs <= params_.curve[i + 1].H) {
                break;
            }
        }

        if (i >= params_.curve.size() - 1) {
            i = params_.curve.size() - 2;
        }

        // Linear interpolation
        Real H1 = params_.curve[i].H;
        Real H2 = params_.curve[i + 1].H;
        Real B1 = params_.curve[i].B;
        Real B2 = params_.curve[i + 1].B;

        Real t = (H_abs - H1) / (H2 - H1 + 1e-12);
        t = std::clamp(t, 0.0, 1.0);

        op.B = sign * (B1 + t * (B2 - B1));
        op.dB_dH = (B2 - B1) / (H2 - H1 + 1e-12);

        // Beyond last point - saturated
        if (H_abs > params_.curve.back().H) {
            op.B = sign * params_.curve.back().B;
            op.dB_dH = MU0;  // Air permeability in saturation
        }
    }

    op.mu = op.dB_dH;
    op.M = op.B / MU0 - H;
    op.flux = op.B * geometry_.Ae;
    op.energy = 0.5 * op.B * H * geometry_.effective_volume();

    return op;
}

// =============================================================================
// Steinmetz Loss Model Implementation
// =============================================================================

SteinmetzParams SteinmetzParams::ferrite_3C90() {
    return {1.41, 1.36, 2.86, 0.0};
}

SteinmetzParams SteinmetzParams::ferrite_3F3() {
    return {1.08, 1.44, 2.88, 0.0};
}

SteinmetzParams SteinmetzParams::silicon_steel() {
    return {0.79, 1.51, 1.74, 0.0};
}

SteinmetzParams SteinmetzParams::amorphous_2605SA1() {
    // 2605SA1 has very low losses compared to ferrites
    // k is significantly lower, alpha and beta similar
    return {0.057, 1.34, 2.18, 0.0};  // Realistic parameters for amorphous metal
}

SteinmetzLossModel::SteinmetzLossModel(const SteinmetzParams& params)
    : params_(params) {
    compute_ki();
}

void SteinmetzLossModel::compute_ki() {
    // Compute iGSE coefficient from Steinmetz parameters
    // ki = k / (2^(beta-1) * pi^(alpha-1) * integral)
    // Approximation for the integral
    if (params_.ki > 0) {
        ki_ = params_.ki;
    } else {
        // Simplified computation
        Real integral_approx = 2.0 * std::pow(2.0, params_.beta - 1);
        ki_ = params_.k / (std::pow(M_PI, params_.alpha - 1) * integral_approx);
    }
}

Real SteinmetzLossModel::loss_density_sine(Real frequency, Real Bpk) const {
    // Steinmetz equation: Pv = k * f^alpha * B^beta
    if (frequency <= 0 || Bpk <= 0) {
        return 0.0;
    }
    return params_.k * std::pow(frequency, params_.alpha) * std::pow(Bpk, params_.beta);
}

Real SteinmetzLossModel::loss_density_igse(Real frequency, Real dB_dt, Real Bpk) const {
    // Improved Generalized Steinmetz Equation
    // Pv = ki * |dB/dt|^alpha * (ΔB)^(beta-alpha)
    if (frequency <= 0 || Bpk <= 0) {
        return 0.0;
    }

    Real deltaB = 2.0 * Bpk;  // Peak-to-peak
    Real dB_dt_abs = std::abs(dB_dt);

    if (dB_dt_abs < 1e-12) {
        // Use sinusoidal approximation
        dB_dt_abs = 2.0 * M_PI * frequency * Bpk;
    }

    return ki_ * std::pow(dB_dt_abs, params_.alpha) *
           std::pow(deltaB, params_.beta - params_.alpha);
}

Real SteinmetzLossModel::total_loss_sine(Real frequency, Real Bpk, Real volume) const {
    return loss_density_sine(frequency, Bpk) * volume;
}

// =============================================================================
// Combined Core Model Implementation
// =============================================================================

CombinedCoreModel::CombinedCoreModel(const CombinedCoreParams& params)
    : params_(params),
      hysteresis_(params.hysteresis, params.geometry),
      losses_(params.losses) {}

void CombinedCoreModel::reset() {
    hysteresis_.reset();
}

MagneticCoreOpPoint CombinedCoreModel::evaluate(Real H, Real dH_dt, Real frequency) {
    MagneticCoreOpPoint op;

    if (params_.include_hysteresis) {
        op = hysteresis_.evaluate(H, dH_dt);
    } else {
        // Simple linear model
        Real mu_r = 1000.0;  // Typical relative permeability
        op.H = H;
        op.B = MU0 * mu_r * H;
        op.M = (mu_r - 1.0) * H;
        op.dB_dH = MU0 * mu_r;
        op.mu = op.dB_dH;
        op.flux = op.B * params_.geometry.Ae;
    }

    // Add eddy current effects
    if (params_.include_eddy_current && params_.sigma > 0 && params_.d > 0) {
        // Eddy current loss density: Pe = (pi * d * Bpk * f)^2 / (6 * rho)
        // For arbitrary waveforms, use dB/dt
        Real dB_dt = op.dB_dH * dH_dt;
        Real Pe_density = (params_.d * params_.d * params_.sigma / 12.0) *
                          dB_dt * dB_dt;
        op.loss += Pe_density * params_.geometry.effective_volume();
    }

    // Add hysteresis/core losses (Steinmetz)
    if (frequency > 0) {
        Real Bpk = std::abs(op.B);  // Approximate peak
        op.loss += losses_.total_loss_sine(frequency, Bpk, params_.geometry.effective_volume());
    }

    return op;
}

// =============================================================================
// Saturable Inductor Implementation
// =============================================================================

SaturableInductor::SaturableInductor(const SaturableInductorParams& params)
    : params_(params) {
    Lsat_ = (params.Lsat > 0) ? params.Lsat : 0.1 * params.L0;

    if (params.use_core_model) {
        core_model_ = std::make_unique<CombinedCoreModel>(params.core);
    }
}

Real SaturableInductor::inductance(Real current) const {
    // Simple saturation model: L(i) = Lsat + (L0 - Lsat) / (1 + (i/Isat)^2)
    Real i_norm = current / params_.Isat;
    return Lsat_ + (params_.L0 - Lsat_) / (1.0 + i_norm * i_norm);
}

Real SaturableInductor::flux_linkage(Real current) const {
    // Integral of L(i) * di
    // For the simple model: λ = Lsat*i + (L0-Lsat)*Isat*atan(i/Isat)
    Real i_norm = current / params_.Isat;
    return Lsat_ * current +
           (params_.L0 - Lsat_) * params_.Isat * std::atan(i_norm);
}

Real SaturableInductor::voltage(Real current, Real di_dt) const {
    // v = dλ/dt = L(i) * di/dt
    return inductance(current) * di_dt;
}

Real SaturableInductor::di_dt(Real voltage, Real current) const {
    // di/dt = v / L(i)
    Real L = inductance(current);
    if (L < 1e-12) {
        L = 1e-12;  // Prevent division by zero
    }
    return voltage / L;
}

// =============================================================================
// Saturable Transformer Implementation
// =============================================================================

SaturableTransformer::SaturableTransformer(const SaturableTransformerParams& params)
    : params_(params) {
    core_model_ = std::make_unique<CombinedCoreModel>(params.core);
}

TransformerOpPoint SaturableTransformer::evaluate(Real v1, Real i1, Real i2,
                                                   Real di1_dt, Real di2_dt) {
    TransformerOpPoint op;
    op.v1 = v1;
    op.i1 = i1;
    op.i2 = i2;

    // Turns ratio
    Real n = turns_ratio();

    // Magnetizing current (referred to primary)
    // im = i1 + i2/n (assuming ideal current transformation)
    op.im = i1 + i2 / n;

    // Compute magnetizing inductance with saturation
    Real im_norm = op.im / params_.Isat;
    Real Lm_ratio = 1.0 / (1.0 + im_norm * im_norm);
    op.Lm_eff = params_.Lm * Lm_ratio;

    // Flux from magnetizing current
    op.phi = op.Lm_eff * op.im / params_.N1;

    // Secondary voltage (ideal transformer)
    // v2 = v1/n - R2*i2 - Llk2*di2/dt
    Real v2_ideal = v1 / n;
    op.v2 = v2_ideal - params_.R2 * i2 - params_.Llk2 * di2_dt;

    // Primary voltage drop
    Real v1_drop = params_.R1 * i1 + params_.Llk1 * di1_dt;

    // Core losses (simplified - proportional to flux squared and frequency)
    // Actual implementation would track dφ/dt
    op.Pcore = 0.0;  // Would need frequency information

    (void)v1_drop;  // Used for detailed model

    return op;
}

}  // namespace pulsim::models
