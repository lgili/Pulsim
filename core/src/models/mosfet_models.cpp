#include "pulsim/models/mosfet_models.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulsim::models {

// Physical constants
constexpr Real q_e = 1.602176634e-19;    // Electron charge (C)
constexpr Real k_b = 1.380649e-23;       // Boltzmann constant (J/K)
[[maybe_unused]] constexpr Real eps_si = 11.7 * 8.854e-12; // Silicon permittivity (F/m)
constexpr Real eps_ox = 3.9 * 8.854e-12;  // Oxide permittivity (F/m)
[[maybe_unused]] constexpr Real ni_300 = 1.45e16;  // Intrinsic carrier concentration at 300K (1/m³)

// Helper functions
inline Real safe_sqrt(Real x) {
    return std::sqrt(std::max(x, 0.0));
}

inline Real safe_log(Real x) {
    return std::log(std::max(x, 1e-30));
}

inline Real thermal_voltage(Real temp) {
    return k_b * temp / q_e;
}

// =============================================================================
// Level 1 MOSFET Implementation
// =============================================================================

MOSFETLevel1::MOSFETLevel1(const MOSFETModelParams& params)
    : params_(params)
{}

Real MOSFETLevel1::compute_vth(Real vbs, Real temp) const {
    // Threshold voltage with body effect and temperature dependence
    Real vth = params_.vth0;

    // Temperature dependence of Vth (typically -2mV/K for NMOS)
    // Vth(T) = Vth(Tnom) + KT1 * (T - Tnom)
    constexpr Real KT1 = -2.0e-3;  // Temperature coefficient (V/K)
    vth += KT1 * (temp - params_.tnom);

    // Body effect
    if (params_.gamma > 0) {
        Real phi_t = params_.phi * (temp / params_.tnom);
        vth += params_.gamma * (safe_sqrt(phi_t - vbs) - safe_sqrt(phi_t));
    }

    return params_.is_pmos ? -vth : vth;
}

Real MOSFETLevel1::compute_beta(Real temp) const {
    // Temperature-adjusted transconductance
    Real beta = params_.kp * inst_.w / inst_.l;

    // Temperature dependence of mobility
    Real temp_ratio = temp / params_.tnom;
    beta *= std::pow(temp_ratio, -1.5);

    return beta;
}

MOSFETOpPoint MOSFETLevel1::evaluate(Real vgs, Real vds, Real vbs, Real temp) const {
    MOSFETOpPoint op;
    op.vgs = vgs;
    op.vds = vds;
    op.vbs = vbs;

    // Handle PMOS by flipping signs
    Real sign = params_.is_pmos ? -1.0 : 1.0;
    Real vgs_eff = sign * vgs;
    Real vds_eff = sign * vds;
    Real vbs_eff = sign * vbs;

    // Compute threshold voltage
    op.vth = compute_vth(vbs_eff, temp);
    Real vth_eff = sign * op.vth;

    Real vov = vgs_eff - vth_eff;  // Overdrive voltage

    // Compute beta
    Real beta = compute_beta(temp);

    if (vov <= 0) {
        // Cutoff region
        op.region = MOSRegion::CUTOFF;
        op.ids = 0.0;
        op.gm = 0.0;
        op.gds = 0.0;
        op.gmbs = 0.0;
    } else {
        op.vdsat = vov;

        if (vds_eff < vov) {
            // Linear (triode) region
            op.region = MOSRegion::LINEAR;
            op.ids = beta * (vov * vds_eff - 0.5 * vds_eff * vds_eff) *
                     (1.0 + params_.lambda * vds_eff);

            op.gm = beta * vds_eff * (1.0 + params_.lambda * vds_eff);
            op.gds = beta * (vov - vds_eff) * (1.0 + params_.lambda * vds_eff) +
                     beta * (vov * vds_eff - 0.5 * vds_eff * vds_eff) * params_.lambda;
        } else {
            // Saturation region
            op.region = MOSRegion::SATURATION;
            op.ids = 0.5 * beta * vov * vov * (1.0 + params_.lambda * vds_eff);

            op.gm = beta * vov * (1.0 + params_.lambda * vds_eff);
            op.gds = 0.5 * beta * vov * vov * params_.lambda;
        }

        // Body transconductance
        if (params_.gamma > 0) {
            Real phi_t = params_.phi * (temp / params_.tnom);
            Real sqrt_term = safe_sqrt(phi_t - vbs_eff);
            if (sqrt_term > 0) {
                op.gmbs = op.gm * params_.gamma / (2.0 * sqrt_term);
            }
        }

        // Apply sign for PMOS
        op.ids *= sign;
    }

    // Compute capacitances (simplified Meyer model)
    Real cox = eps_ox / params_.tox;
    Real wl = inst_.w * inst_.l;

    if (op.region == MOSRegion::CUTOFF) {
        op.cgb = cox * wl;
        op.cgs = params_.cgso * inst_.w;
        op.cgd = params_.cgdo * inst_.w;
    } else if (op.region == MOSRegion::LINEAR) {
        (void)vds;  // Used in formulas below
        op.cgs = (2.0 / 3.0) * cox * wl * (1.0 - std::pow((vov - vds_eff) / (2.0 * vov - vds_eff), 2)) +
                 params_.cgso * inst_.w;
        op.cgd = (2.0 / 3.0) * cox * wl * (1.0 - std::pow(vov / (2.0 * vov - vds_eff), 2)) +
                 params_.cgdo * inst_.w;
        op.cgb = 0.0;
    } else {
        // Saturation
        op.cgs = (2.0 / 3.0) * cox * wl + params_.cgso * inst_.w;
        op.cgd = params_.cgdo * inst_.w;
        op.cgb = 0.0;
    }

    // Junction capacitances (simplified)
    if (params_.cj > 0) {
        Real vbi = params_.pb;
        op.cbs = params_.cj * inst_.as * std::pow(1.0 - vbs_eff / vbi, -params_.mj);
        op.cbd = params_.cj * inst_.ad * std::pow(1.0 - (vbs_eff - vds_eff) / vbi, -params_.mj);
    }

    return op;
}

// =============================================================================
// Level 2 MOSFET Implementation
// =============================================================================

MOSFETLevel2::MOSFETLevel2(const MOSFETModelParams& params)
    : params_(params)
{}

Real MOSFETLevel2::compute_vth(Real vbs, Real temp) const {
    Real phi_t = params_.phi * (temp / params_.tnom);
    Real vth = params_.vth0 + params_.gamma * (safe_sqrt(phi_t - vbs) - safe_sqrt(phi_t));
    return params_.is_pmos ? -vth : vth;
}

Real MOSFETLevel2::compute_mobility(Real vgs, Real /*vds*/, Real temp) const {
    Real u_eff = params_.u0;

    // Transverse field degradation
    if (params_.utra > 0) {
        Real eox = std::abs(vgs) / params_.tox;
        u_eff = params_.u0 / (1.0 + params_.utra * eox);
    }

    // Temperature dependence
    Real temp_ratio = temp / params_.tnom;
    u_eff *= std::pow(temp_ratio, -1.5);

    return u_eff;
}

Real MOSFETLevel2::compute_vdsat(Real vgs, Real vth, Real temp) const {
    Real vov = vgs - vth;

    if (params_.vmax > 0) {
        // Velocity saturation
        Real u_eff = compute_mobility(vgs, 0.0, temp);
        Real esat = 2.0 * params_.vmax / (u_eff * 1e-4);  // Convert to V/m
        Real vdsat_vs = esat * inst_.l;
        return vov * vdsat_vs / (vov + vdsat_vs);
    }

    return vov;
}

MOSFETOpPoint MOSFETLevel2::evaluate(Real vgs, Real vds, Real vbs, Real temp) const {
    MOSFETOpPoint op;
    op.vgs = vgs;
    op.vds = vds;
    op.vbs = vbs;

    Real sign = params_.is_pmos ? -1.0 : 1.0;
    Real vgs_eff = sign * vgs;
    Real vds_eff = sign * vds;
    Real vbs_eff = sign * vbs;

    op.vth = compute_vth(vbs_eff, temp);
    Real vth_eff = sign * op.vth;
    Real vov = vgs_eff - vth_eff;

    // Check for subthreshold
    Real vt = thermal_voltage(temp);
    Real n_sub = 1.0 + params_.gamma / (2.0 * safe_sqrt(params_.phi - vbs_eff));

    if (vov < 0) {
        // Subthreshold region
        if (params_.nfs > 0 || true) {  // Enable subthreshold
            op.region = MOSRegion::SUBTHRESHOLD;

            Real cox = eps_ox / params_.tox;
            Real u_eff = compute_mobility(vgs_eff, vds_eff, temp);
            Real beta = u_eff * 1e-4 * cox * inst_.w / inst_.l;

            Real ids_sub = beta * n_sub * vt * vt *
                           std::exp((vov) / (n_sub * vt)) *
                           (1.0 - std::exp(-vds_eff / vt));

            op.ids = sign * ids_sub;
            op.gm = ids_sub / (n_sub * vt);
            op.gds = ids_sub * std::exp(-vds_eff / vt) / vt;
            op.vdsat = 0.0;
        } else {
            op.region = MOSRegion::CUTOFF;
            op.ids = 0.0;
            op.gm = 0.0;
            op.gds = 0.0;
        }
    } else {
        op.vdsat = compute_vdsat(vgs_eff, vth_eff, temp);

        Real cox = eps_ox / params_.tox;
        Real u_eff = compute_mobility(vgs_eff, vds_eff, temp);
        Real beta = u_eff * 1e-4 * cox * inst_.w / inst_.l;

        if (vds_eff < op.vdsat) {
            // Linear region
            op.region = MOSRegion::LINEAR;

            if (params_.vmax > 0) {
                // With velocity saturation
                Real esat = 2.0 * params_.vmax / (u_eff * 1e-4);
                Real factor = 1.0 + vds_eff / (esat * inst_.l);
                op.ids = beta * (vov * vds_eff - 0.5 * vds_eff * vds_eff) / factor *
                         (1.0 + params_.lambda * vds_eff);
            } else {
                op.ids = beta * (vov * vds_eff - 0.5 * vds_eff * vds_eff) *
                         (1.0 + params_.lambda * vds_eff);
            }

            op.gm = beta * vds_eff * (1.0 + params_.lambda * vds_eff);
            op.gds = beta * (vov - vds_eff) * (1.0 + params_.lambda * vds_eff);
        } else {
            // Saturation region
            op.region = MOSRegion::SATURATION;

            if (params_.vmax > 0) {
                Real esat = 2.0 * params_.vmax / (u_eff * 1e-4);
                Real factor = 1.0 + op.vdsat / (esat * inst_.l);
                op.ids = 0.5 * beta * op.vdsat * op.vdsat / factor *
                         (1.0 + params_.lambda * vds_eff);
            } else {
                op.ids = 0.5 * beta * vov * vov * (1.0 + params_.lambda * vds_eff);
            }

            op.gm = beta * vov * (1.0 + params_.lambda * vds_eff);
            op.gds = op.ids * params_.lambda / (1.0 + params_.lambda * vds_eff);
        }

        op.ids *= sign;
    }

    // Body transconductance
    if (params_.gamma > 0 && vov > 0) {
        Real sqrt_term = safe_sqrt(params_.phi - vbs_eff);
        if (sqrt_term > 0) {
            op.gmbs = op.gm * params_.gamma / (2.0 * sqrt_term);
        }
    }

    return op;
}

// =============================================================================
// Level 3 MOSFET Implementation
// =============================================================================

MOSFETLevel3::MOSFETLevel3(const MOSFETModelParams& params)
    : params_(params)
{}

Real MOSFETLevel3::compute_vth(Real vbs, Real vds, Real temp) const {
    Real phi_t = params_.phi * (temp / params_.tnom);
    Real vth = params_.vth0;

    // Body effect
    vth += params_.gamma * (safe_sqrt(phi_t - vbs) - safe_sqrt(phi_t));

    // Short-channel effect (DIBL)
    if (params_.eta > 0) {
        vth -= params_.eta * vds;
    }

    return params_.is_pmos ? -vth : vth;
}

Real MOSFETLevel3::compute_mobility(Real vgs, Real /*vbs*/, Real temp) const {
    Real u_eff = params_.u0;

    // Surface scattering
    if (params_.theta > 0) {
        Real vov = vgs - params_.vth0;
        u_eff = params_.u0 / (1.0 + params_.theta * std::max(vov, 0.0));
    }

    // Temperature dependence
    Real temp_ratio = temp / params_.tnom;
    u_eff *= std::pow(temp_ratio, -1.5);

    return u_eff;
}

MOSFETOpPoint MOSFETLevel3::evaluate(Real vgs, Real vds, Real vbs, Real temp) const {
    MOSFETOpPoint op;
    op.vgs = vgs;
    op.vds = vds;
    op.vbs = vbs;

    Real sign = params_.is_pmos ? -1.0 : 1.0;
    Real vgs_eff = sign * vgs;
    Real vds_eff = sign * vds;
    Real vbs_eff = sign * vbs;

    op.vth = compute_vth(vbs_eff, vds_eff, temp);
    Real vth_eff = sign * op.vth;
    Real vov = vgs_eff - vth_eff;

    Real vt = thermal_voltage(temp);

    if (vov <= 0) {
        op.region = MOSRegion::SUBTHRESHOLD;

        // Subthreshold current
        Real n_sub = 1.0 + params_.gamma / (2.0 * safe_sqrt(params_.phi - vbs_eff));
        Real cox = eps_ox / params_.tox;
        Real u_eff = compute_mobility(vgs_eff, vbs_eff, temp);
        Real beta = u_eff * 1e-4 * cox * inst_.w / inst_.l;

        Real ids_sub = beta * n_sub * vt * vt *
                       std::exp(vov / (n_sub * vt)) *
                       (1.0 - std::exp(-vds_eff / vt));

        op.ids = sign * ids_sub;
        op.gm = ids_sub / (n_sub * vt);
        op.gds = ids_sub * std::exp(-vds_eff / vt) / vt;
        op.vdsat = 0.0;
    } else {
        Real cox = eps_ox / params_.tox;
        Real u_eff = compute_mobility(vgs_eff, vbs_eff, temp);
        Real beta = u_eff * 1e-4 * cox * inst_.w / inst_.l;

        // Compute saturation voltage
        if (params_.vmax > 0) {
            Real esat = 2.0 * params_.vmax / (u_eff * 1e-4);
            op.vdsat = vov * esat * inst_.l / (vov + esat * inst_.l);
        } else {
            op.vdsat = vov;
        }

        if (vds_eff < op.vdsat) {
            op.region = MOSRegion::LINEAR;

            Real fn = 1.0;
            if (params_.vmax > 0) {
                Real esat = 2.0 * params_.vmax / (u_eff * 1e-4);
                fn = 1.0 / (1.0 + vds_eff / (esat * inst_.l));
            }

            op.ids = beta * fn * (vov * vds_eff - 0.5 * vds_eff * vds_eff) *
                     (1.0 + params_.lambda * vds_eff);
            op.gm = beta * fn * vds_eff * (1.0 + params_.lambda * vds_eff);
            op.gds = beta * fn * (vov - vds_eff) * (1.0 + params_.lambda * vds_eff);
        } else {
            op.region = MOSRegion::SATURATION;

            Real fn = 1.0;
            if (params_.vmax > 0) {
                Real esat = 2.0 * params_.vmax / (u_eff * 1e-4);
                fn = 1.0 / (1.0 + op.vdsat / (esat * inst_.l));
            }

            op.ids = 0.5 * beta * fn * op.vdsat * op.vdsat *
                     (1.0 + params_.lambda * vds_eff);
            op.gm = beta * fn * op.vdsat * (1.0 + params_.lambda * vds_eff);
            op.gds = op.ids * params_.lambda / (1.0 + params_.lambda * vds_eff);
        }

        op.ids *= sign;
    }

    // Body transconductance
    if (params_.gamma > 0 && vov > 0) {
        Real sqrt_term = safe_sqrt(params_.phi - vbs_eff);
        if (sqrt_term > 0) {
            op.gmbs = op.gm * params_.gamma / (2.0 * sqrt_term);
        }
    }

    return op;
}

// =============================================================================
// BSIM3v3 MOSFET Implementation
// =============================================================================

MOSFETBSIM3::MOSFETBSIM3(const MOSFETModelParams& params)
    : params_(params)
{}

void MOSFETBSIM3::set_instance(const MOSFETInstance& inst) {
    inst_ = inst;
    size_params_valid_ = false;
}

void MOSFETBSIM3::compute_size_params() const {
    if (size_params_valid_) return;

    // Effective length and width (with minimum value protection)
    leff_ = std::max(inst_.l - 2.0 * params_.ld, 1e-9);
    weff_ = std::max(inst_.w - 2.0 * params_.wd, 1e-9);

    // Size-dependent threshold voltage (short-channel effect)
    Real dsub = 0.0;
    if (params_.nlx > 0 && params_.dvt0 != 0) {
        Real t0 = params_.dvt1 * leff_ / params_.nlx;
        if (std::abs(t0) > 1e-10) {
            Real t1 = std::exp(-std::min(t0, 50.0));  // Limit to prevent overflow
            dsub = params_.dvt0 * (1.0 - t1) / t0;
        }
    }

    vth0_eff_ = params_.vth0 - dsub;

    // Size-dependent body effect
    k1_eff_ = params_.k1;
    k2_eff_ = params_.k2;

    size_params_valid_ = true;
}

Real MOSFETBSIM3::compute_vth(Real vbs, Real vds, Real /*temp*/) const {
    compute_size_params();

    Real phi_s = params_.phi;
    Real sqrtphi = safe_sqrt(phi_s);

    // Body effect
    Real vth = vth0_eff_ + k1_eff_ * (safe_sqrt(phi_s - vbs) - sqrtphi);

    // DIBL effect
    Real theta_rout = params_.pdiblc1 + params_.pdiblc2 * leff_;
    vth -= theta_rout * vds;

    return params_.is_pmos ? -vth : vth;
}

Real MOSFETBSIM3::compute_vdsat(Real vgs, Real vth, Real vbs, Real /*temp*/) const {
    Real vov = vgs - vth;
    if (vov <= 0) return 0.0;

    // Abulk calculation (body effect on channel charge)
    Real abulk = 1.0 + params_.k1 / (2.0 * safe_sqrt(params_.phi - vbs));

    // Velocity saturation
    Real vsat_eff = params_.vsat;
    Real esat = 2.0 * vsat_eff / (params_.u0 * 1e-4);

    Real vdsat = vov / abulk;

    // Include velocity saturation
    if (esat * leff_ > 0) {
        Real t0 = vdsat / (esat * leff_);
        vdsat = vdsat / (1.0 + t0);
    }

    return vdsat;
}

Real MOSFETBSIM3::compute_ids_linear(Real vgs, Real vds, Real vth, Real /*vdsat*/, Real /*temp*/) const {
    Real vov = vgs - vth;
    if (vov <= 0) return 0.0;

    Real cox = eps_ox / params_.tox;
    Real u_eff = params_.u0 * 1e-4;  // Convert to m²/V·s

    // Mobility degradation
    if (params_.ua > 0 || params_.ub > 0) {
        Real eeff = (vgs + vth) / (2.0 * params_.tox);
        u_eff = u_eff / (1.0 + (params_.ua + params_.ub * eeff) * eeff);
    }

    Real beta = u_eff * cox * weff_ / leff_;

    // Abulk
    Real abulk = 1.0 + params_.k1 / (2.0 * safe_sqrt(params_.phi));

    Real ids = beta * ((vov / abulk) * vds - 0.5 * vds * vds / abulk);

    // Channel length modulation
    ids *= (1.0 + params_.pclm * vds / leff_);

    return ids;
}

Real MOSFETBSIM3::compute_ids_sat(Real vgs, Real vds, Real vth, Real vdsat, Real /*temp*/) const {
    Real vov = vgs - vth;
    if (vov <= 0) return 0.0;

    Real cox = eps_ox / params_.tox;
    Real u_eff = params_.u0 * 1e-4;

    Real beta = u_eff * cox * weff_ / leff_;

    // Abulk
    Real abulk = 1.0 + params_.k1 / (2.0 * safe_sqrt(params_.phi));

    Real ids = 0.5 * beta * vdsat * vdsat / abulk;

    // CLM and DIBL output resistance
    Real va = params_.pclm * leff_ / vdsat +
              params_.pdiblc1 + params_.pdiblc2;
    ids *= (1.0 + vds / va);

    return ids;
}

MOSFETOpPoint MOSFETBSIM3::evaluate(Real vgs, Real vds, Real vbs, Real temp) const {
    compute_size_params();

    MOSFETOpPoint op;
    op.vgs = vgs;
    op.vds = vds;
    op.vbs = vbs;

    Real sign = params_.is_pmos ? -1.0 : 1.0;
    Real vgs_eff = sign * vgs;
    Real vds_eff = sign * vds;
    Real vbs_eff = sign * vbs;

    op.vth = compute_vth(vbs_eff, vds_eff, temp);
    Real vth_eff = sign * op.vth;
    Real vov = vgs_eff - vth_eff;

    Real vt = thermal_voltage(temp);

    if (vov <= 0) {
        // Subthreshold
        op.region = MOSRegion::SUBTHRESHOLD;

        Real n_sub = 1.0 + params_.nfactor;
        Real cox = eps_ox / params_.tox;
        Real beta = params_.u0 * 1e-4 * cox * weff_ / leff_;

        Real ids_sub = beta * n_sub * vt * vt *
                       std::exp((vov - params_.voff) / (n_sub * vt)) *
                       (1.0 - std::exp(-vds_eff / vt));

        op.ids = sign * ids_sub;
        op.gm = ids_sub / (n_sub * vt);
        op.gds = ids_sub * std::exp(-vds_eff / vt) / vt;
        op.vdsat = 0.0;
    } else {
        op.vdsat = compute_vdsat(vgs_eff, vth_eff, vbs_eff, temp);

        if (vds_eff < op.vdsat) {
            op.region = MOSRegion::LINEAR;
            op.ids = sign * compute_ids_linear(vgs_eff, vds_eff, vth_eff, op.vdsat, temp);
        } else {
            op.region = MOSRegion::SATURATION;
            op.ids = sign * compute_ids_sat(vgs_eff, vds_eff, vth_eff, op.vdsat, temp);
        }

        // Numerical derivatives for gm and gds
        // Need to recompute vdsat for perturbed vgs to get correct gm
        Real delta = 1e-6;
        Real ids_base = std::abs(op.ids);

        // For gm: perturb vgs and recompute vdsat
        Real vdsat_gm = compute_vdsat(vgs_eff + delta, vth_eff, vbs_eff, temp);
        Real ids_gm = (vds_eff < vdsat_gm)
            ? compute_ids_linear(vgs_eff + delta, vds_eff, vth_eff, vdsat_gm, temp)
            : compute_ids_sat(vgs_eff + delta, vds_eff, vth_eff, vdsat_gm, temp);

        // For gds: keep original vdsat, only perturb vds
        Real ids_gds = (op.region == MOSRegion::LINEAR)
            ? compute_ids_linear(vgs_eff, vds_eff + delta, vth_eff, op.vdsat, temp)
            : compute_ids_sat(vgs_eff, vds_eff + delta, vth_eff, op.vdsat, temp);

        op.gm = (ids_gm - ids_base) / delta;
        op.gds = (ids_gds - ids_base) / delta;
    }

    // Body transconductance
    if (vov > 0) {
        Real sqrt_term = safe_sqrt(params_.phi - vbs_eff);
        if (sqrt_term > 0) {
            op.gmbs = op.gm * params_.k1 / (2.0 * sqrt_term);
        }
    }

    return op;
}

// =============================================================================
// EKV MOSFET Implementation
// =============================================================================

MOSFETEKV::MOSFETEKV(const EKVParams& params)
    : params_(params)
{}

Real MOSFETEKV::compute_vp(Real vgs, Real vbs) const {
    Real gamma = params_.gamma;
    Real phi = params_.phi;

    // Pinch-off voltage
    Real vp = vgs - params_.vto - gamma * (safe_sqrt(phi - vbs) - safe_sqrt(phi));
    return vp;
}

Real MOSFETEKV::compute_n(Real vp, Real vbs) const {
    Real phi = params_.phi;
    Real gamma = params_.gamma;

    // Slope factor
    Real n = 1.0 + gamma / (2.0 * safe_sqrt(phi - vbs + vp));
    return std::max(n, 1.0);
}

Real MOSFETEKV::compute_if(Real vp, Real vs) const {
    // Forward normalized current using smooth EKV interpolation function
    // This ensures continuous transition between weak and strong inversion
    Real x = (vp - vs);
    // Smooth interpolation: (ln(1 + exp(x/2)))^2
    // For large positive x: approaches (x/2)^2 (strong inversion)
    // For large negative x: approaches exp(x) (weak inversion)
    if (x > 40.0) {
        Real t = x / 2.0;
        return t * t;  // Avoid overflow
    } else if (x < -40.0) {
        return std::exp(x);  // Avoid underflow in ln(1+exp)
    } else {
        Real t = std::log1p(std::exp(x / 2.0));
        return t * t;
    }
}

Real MOSFETEKV::compute_ir(Real vp, Real vd) const {
    // Reverse normalized current using smooth EKV interpolation
    Real x = (vp - vd);
    if (x > 40.0) {
        Real t = x / 2.0;
        return t * t;
    } else if (x < -40.0) {
        return std::exp(x);
    } else {
        Real t = std::log1p(std::exp(x / 2.0));
        return t * t;
    }
}

MOSFETOpPoint MOSFETEKV::evaluate(Real vgs, Real vds, Real vbs, Real temp) const {
    MOSFETOpPoint op;
    op.vgs = vgs;
    op.vds = vds;
    op.vbs = vbs;

    Real sign = params_.is_pmos ? -1.0 : 1.0;
    Real vgs_eff = sign * vgs;
    Real vds_eff = sign * vds;
    Real vbs_eff = sign * vbs;

    Real vt = thermal_voltage(temp);

    // Pinch-off voltage
    Real vp = compute_vp(vgs_eff, vbs_eff);
    op.vth = params_.vto + params_.gamma * (safe_sqrt(params_.phi - vbs_eff) - safe_sqrt(params_.phi));
    op.vth = sign * op.vth;

    // Slope factor
    Real n = compute_n(vp, vbs_eff);

    // Normalize voltages
    Real vs = 0.0;  // Source at reference
    Real vd = vds_eff;
    Real vp_norm = vp / (n * vt);
    Real vs_norm = vs / (n * vt);
    Real vd_norm = vd / (n * vt);

    // Forward and reverse currents
    Real i_f = compute_if(vp_norm, vs_norm);
    Real i_r = compute_ir(vp_norm, vd_norm);

    // Specific current
    Real is = params_.kp * n * vt * vt * inst_.w / inst_.l;

    // Drain current (single equation valid in all regions)
    Real ids = is * (i_f - i_r);

    // Mobility degradation
    if (params_.theta > 0) {
        Real vov = std::max(vgs_eff - op.vth * sign, 0.0);
        ids /= (1.0 + params_.theta * vov);
    }

    // Velocity saturation
    if (params_.ucrit > 0 && vds_eff > 0) {
        Real esat = params_.ucrit;
        Real vc = esat * inst_.l;
        Real vdsat = vp - vs;
        if (vdsat > 0) {
            Real x = vds_eff / vc;
            ids *= (1.0 + x) / (1.0 + x + x * x);
        }
    }

    // Channel length modulation
    if (params_.lambda > 0) {
        ids *= (1.0 + params_.lambda * vds_eff);
    }

    op.ids = sign * ids;

    // Determine operating region
    if (vp < 0) {
        op.region = MOSRegion::SUBTHRESHOLD;
    } else if (vds_eff < vp) {
        op.region = MOSRegion::LINEAR;
    } else {
        op.region = MOSRegion::SATURATION;
    }

    op.vdsat = std::max(vp, 0.0);

    // Transconductances using central difference for better accuracy
    Real delta = 1e-6;

    // gm = dIds/dVgs
    auto calc_ids_at_vgs = [&](Real vgs_test) {
        Real vp_t = compute_vp(vgs_test, vbs_eff);
        Real n_t = compute_n(vp_t, vbs_eff);
        Real is_t = params_.kp * n_t * vt * vt * inst_.w / inst_.l;
        Real i_f_t = compute_if(vp_t / (n_t * vt), vs / (n_t * vt));
        Real i_r_t = compute_ir(vp_t / (n_t * vt), vd / (n_t * vt));
        Real ids_t = is_t * (i_f_t - i_r_t);
        if (params_.theta > 0) {
            Real vov_t = std::max(vgs_test - op.vth * sign, 0.0);
            ids_t /= (1.0 + params_.theta * vov_t);
        }
        if (params_.lambda > 0) {
            ids_t *= (1.0 + params_.lambda * vds_eff);
        }
        return ids_t;
    };

    Real ids_plus = calc_ids_at_vgs(vgs_eff + delta);
    Real ids_minus = calc_ids_at_vgs(vgs_eff - delta);
    op.gm = (ids_plus - ids_minus) / (2.0 * delta);

    // gds = dIds/dVds (simpler calculation)
    Real i_r_plus = compute_ir(vp_norm, (vd + delta) / (n * vt));
    Real i_r_minus = compute_ir(vp_norm, (vd - delta) / (n * vt));
    op.gds = is * (i_r_minus - i_r_plus) / (2.0 * delta);
    if (params_.lambda > 0) {
        op.gds += std::abs(ids) * params_.lambda / (1.0 + params_.lambda * vds_eff);
    }

    return op;
}

// =============================================================================
// Factory Implementation
// =============================================================================

class MOSFETLevel1Wrapper : public MOSFETModelBase {
public:
    explicit MOSFETLevel1Wrapper(const MOSFETModelParams& params) : model_(params) {}
    void set_instance(const MOSFETInstance& inst) override { model_.set_instance(inst); }
    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp) const override {
        return model_.evaluate(vgs, vds, vbs, temp);
    }
    MOSFETModelLevel level() const override { return MOSFETModelLevel::LEVEL1; }
private:
    MOSFETLevel1 model_;
};

class MOSFETLevel2Wrapper : public MOSFETModelBase {
public:
    explicit MOSFETLevel2Wrapper(const MOSFETModelParams& params) : model_(params) {}
    void set_instance(const MOSFETInstance& inst) override { model_.set_instance(inst); }
    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp) const override {
        return model_.evaluate(vgs, vds, vbs, temp);
    }
    MOSFETModelLevel level() const override { return MOSFETModelLevel::LEVEL2; }
private:
    MOSFETLevel2 model_;
};

class MOSFETLevel3Wrapper : public MOSFETModelBase {
public:
    explicit MOSFETLevel3Wrapper(const MOSFETModelParams& params) : model_(params) {}
    void set_instance(const MOSFETInstance& inst) override { model_.set_instance(inst); }
    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp) const override {
        return model_.evaluate(vgs, vds, vbs, temp);
    }
    MOSFETModelLevel level() const override { return MOSFETModelLevel::LEVEL3; }
private:
    MOSFETLevel3 model_;
};

class MOSFETBSIM3Wrapper : public MOSFETModelBase {
public:
    explicit MOSFETBSIM3Wrapper(const MOSFETModelParams& params) : model_(params) {}
    void set_instance(const MOSFETInstance& inst) override { model_.set_instance(inst); }
    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp) const override {
        return model_.evaluate(vgs, vds, vbs, temp);
    }
    MOSFETModelLevel level() const override { return MOSFETModelLevel::BSIM3; }
private:
    MOSFETBSIM3 model_;
};

class MOSFETEKVWrapper : public MOSFETModelBase {
public:
    explicit MOSFETEKVWrapper(const EKVParams& params) : model_(params) {}
    void set_instance(const MOSFETInstance& inst) override { model_.set_instance(inst); }
    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp) const override {
        return model_.evaluate(vgs, vds, vbs, temp);
    }
    MOSFETModelLevel level() const override { return MOSFETModelLevel::EKV; }
private:
    MOSFETEKV model_;
};

std::unique_ptr<MOSFETModelBase> create_mosfet_model(
    MOSFETModelLevel level,
    const MOSFETModelParams& params) {

    switch (level) {
        case MOSFETModelLevel::LEVEL1:
            return std::make_unique<MOSFETLevel1Wrapper>(params);
        case MOSFETModelLevel::LEVEL2:
            return std::make_unique<MOSFETLevel2Wrapper>(params);
        case MOSFETModelLevel::LEVEL3:
            return std::make_unique<MOSFETLevel3Wrapper>(params);
        case MOSFETModelLevel::BSIM3:
            return std::make_unique<MOSFETBSIM3Wrapper>(params);
        case MOSFETModelLevel::EKV:
            return std::make_unique<MOSFETEKVWrapper>(EKVParams{});
        default:
            throw std::invalid_argument("Unknown MOSFET model level");
    }
}

}  // namespace pulsim::models
