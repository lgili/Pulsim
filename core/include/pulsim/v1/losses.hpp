#pragma once

// =============================================================================
// PulsimCore - Power Loss Calculation Module
// =============================================================================
// Provides power loss calculations for power electronics devices:
// - Conduction losses (I²R, Vf*I + Rd*I²)
// - Switching losses (Eon, Eoff, Err)
// - Loss integration over time
// - Efficiency calculation
//
// Key equations:
// - Resistor: P_cond = I² * R
// - MOSFET: P_cond = I² * Rds_on(Vgs, T)
// - IGBT: P_cond = Vce_sat * I + Rce * I²
// - Diode: P_cond = Vf * I + Rd * I²
// - Switching: P_sw = (Eon + Eoff) * f_sw
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace pulsim::v1 {

// =============================================================================
// Loss Model Parameters
// =============================================================================

/// MOSFET loss model parameters
struct MOSFETLossParams {
    Real Rds_on = 0.05;          ///< On-state resistance at 25C (Ω)
    Real Rds_on_tc = 0.005;      ///< Temperature coefficient (Ω/K)
    Real Qg = 50e-9;             ///< Total gate charge (C)
    Real Eon_25C = 0.5e-3;       ///< Turn-on energy at 25C, ref conditions (J)
    Real Eoff_25C = 0.3e-3;      ///< Turn-off energy at 25C, ref conditions (J)
    Real I_ref = 10.0;           ///< Reference current for Eon/Eoff (A)
    Real V_ref = 400.0;          ///< Reference voltage for Eon/Eoff (V)
    Real T_ref = 25.0;           ///< Reference temperature (C)
    Real Esw_tc = 0.003;         ///< Switching energy temp coefficient (1/K)

    /// Calculate Rds_on at temperature T
    [[nodiscard]] Real Rds_on_at_T(Real T) const {
        return Rds_on * (1.0 + Rds_on_tc * (T - T_ref));
    }

    /// Scale switching energy for operating point
    [[nodiscard]] Real scale_Esw(Real I, Real V, Real T, Real E_ref) const {
        // Linear scaling with I and V, temperature coefficient
        Real E = E_ref * (I / I_ref) * (V / V_ref);
        E *= (1.0 + Esw_tc * (T - T_ref));
        return E;
    }
};

/// IGBT loss model parameters
struct IGBTLossParams {
    Real Vce_sat = 1.5;          ///< Collector-emitter saturation voltage (V)
    Real Rce = 0.02;             ///< Collector-emitter resistance (Ω)
    Real Vce_tc = 0.002;         ///< Vce temperature coefficient (V/K)
    Real Eon_25C = 2.0e-3;       ///< Turn-on energy at 25C (J)
    Real Eoff_25C = 1.5e-3;      ///< Turn-off energy at 25C (J)
    Real I_ref = 50.0;           ///< Reference current (A)
    Real V_ref = 600.0;          ///< Reference voltage (V)
    Real T_ref = 25.0;           ///< Reference temperature (C)
    Real Esw_tc = 0.003;         ///< Switching energy temp coefficient (1/K)

    /// Calculate Vce_sat at temperature T
    [[nodiscard]] Real Vce_sat_at_T(Real T) const {
        return Vce_sat + Vce_tc * (T - T_ref);
    }

    /// Scale switching energy
    [[nodiscard]] Real scale_Esw(Real I, Real V, Real T, Real E_ref) const {
        Real E = E_ref * (I / I_ref) * (V / V_ref);
        E *= (1.0 + Esw_tc * (T - T_ref));
        return E;
    }
};

/// Diode loss model parameters
struct DiodeLossParams {
    Real Vf = 0.7;               ///< Forward voltage at 25C (V)
    Real Rd = 0.01;              ///< Dynamic resistance (Ω)
    Real Vf_tc = -0.002;         ///< Vf temperature coefficient (V/K)
    Real Qrr = 50e-9;            ///< Reverse recovery charge (C)
    Real trr = 50e-9;            ///< Reverse recovery time (s)
    Real Irr_factor = 0.25;      ///< Irr as fraction of If
    Real Err_factor = 0.5;       ///< Err ≈ factor * Qrr * Vr
    Real T_ref = 25.0;           ///< Reference temperature (C)

    /// Calculate Vf at temperature T
    [[nodiscard]] Real Vf_at_T(Real T) const {
        return Vf + Vf_tc * (T - T_ref);
    }

    /// Calculate reverse recovery energy
    [[nodiscard]] Real Err([[maybe_unused]] Real If, Real Vr, Real /*T*/) const {
        // Simplified: Err ≈ Qrr * Vr * factor
        // More accurate: Err = 0.5 * Qrr * Vr * (Irr/If)^factor
        return Err_factor * Qrr * Vr;
    }
};

// =============================================================================
// Conduction Loss Calculator
// =============================================================================

/// Calculate conduction losses for various device types
class ConductionLoss {
public:
    /// Resistor conduction loss: P = I² * R
    [[nodiscard]] static Real resistor(Real I, Real R) {
        return I * I * R;
    }

    /// MOSFET conduction loss: P = I² * Rds_on(T)
    [[nodiscard]] static Real mosfet(Real I, const MOSFETLossParams& params, Real T) {
        Real Rds = params.Rds_on_at_T(T);
        return I * I * Rds;
    }

    /// IGBT conduction loss: P = Vce_sat * I + Rce * I²
    [[nodiscard]] static Real igbt(Real I, const IGBTLossParams& params, Real T) {
        Real Vce = params.Vce_sat_at_T(T);
        return Vce * I + params.Rce * I * I;
    }

    /// Diode conduction loss: P = Vf * I + Rd * I²
    [[nodiscard]] static Real diode(Real I, const DiodeLossParams& params, Real T) {
        Real Vf = params.Vf_at_T(T);
        return Vf * I + params.Rd * I * I;
    }
};

// =============================================================================
// Switching Loss Calculator
// =============================================================================

/// Calculate switching losses
class SwitchingLoss {
public:
    /// MOSFET turn-on energy
    [[nodiscard]] static Real mosfet_Eon(Real I, Real V, Real T,
                                          const MOSFETLossParams& params) {
        return params.scale_Esw(I, V, T, params.Eon_25C);
    }

    /// MOSFET turn-off energy
    [[nodiscard]] static Real mosfet_Eoff(Real I, Real V, Real T,
                                           const MOSFETLossParams& params) {
        return params.scale_Esw(I, V, T, params.Eoff_25C);
    }

    /// MOSFET total switching energy per cycle
    [[nodiscard]] static Real mosfet_total(Real I, Real V, Real T,
                                            const MOSFETLossParams& params) {
        return mosfet_Eon(I, V, T, params) + mosfet_Eoff(I, V, T, params);
    }

    /// MOSFET switching power at frequency f_sw
    [[nodiscard]] static Real mosfet_power(Real I, Real V, Real T, Real f_sw,
                                            const MOSFETLossParams& params) {
        return mosfet_total(I, V, T, params) * f_sw;
    }

    /// IGBT turn-on energy
    [[nodiscard]] static Real igbt_Eon(Real I, Real V, Real T,
                                        const IGBTLossParams& params) {
        return params.scale_Esw(I, V, T, params.Eon_25C);
    }

    /// IGBT turn-off energy
    [[nodiscard]] static Real igbt_Eoff(Real I, Real V, Real T,
                                         const IGBTLossParams& params) {
        return params.scale_Esw(I, V, T, params.Eoff_25C);
    }

    /// IGBT total switching energy per cycle
    [[nodiscard]] static Real igbt_total(Real I, Real V, Real T,
                                          const IGBTLossParams& params) {
        return igbt_Eon(I, V, T, params) + igbt_Eoff(I, V, T, params);
    }

    /// IGBT switching power at frequency f_sw
    [[nodiscard]] static Real igbt_power(Real I, Real V, Real T, Real f_sw,
                                          const IGBTLossParams& params) {
        return igbt_total(I, V, T, params) * f_sw;
    }

    /// Diode reverse recovery energy
    [[nodiscard]] static Real diode_Err(Real If, Real Vr, Real T,
                                         const DiodeLossParams& params) {
        return params.Err(If, Vr, T);
    }

    /// Diode reverse recovery power at frequency f_sw
    [[nodiscard]] static Real diode_power(Real If, Real Vr, Real T, Real f_sw,
                                           const DiodeLossParams& params) {
        return params.Err(If, Vr, T) * f_sw;
    }
};

// =============================================================================
// Loss Accumulator
// =============================================================================

/// Breakdown of losses by type
struct LossBreakdown {
    Real conduction = 0.0;       ///< Conduction loss (W)
    Real turn_on = 0.0;          ///< Turn-on switching loss (W)
    Real turn_off = 0.0;         ///< Turn-off switching loss (W)
    Real reverse_recovery = 0.0; ///< Diode reverse recovery loss (W)

    /// Total loss
    [[nodiscard]] Real total() const {
        return conduction + turn_on + turn_off + reverse_recovery;
    }

    /// Switching loss (on + off + recovery)
    [[nodiscard]] Real switching() const {
        return turn_on + turn_off + reverse_recovery;
    }

    /// Add another breakdown
    LossBreakdown& operator+=(const LossBreakdown& other) {
        conduction += other.conduction;
        turn_on += other.turn_on;
        turn_off += other.turn_off;
        reverse_recovery += other.reverse_recovery;
        return *this;
    }
};

/// Accumulates losses over time for a device
class LossAccumulator {
public:
    LossAccumulator() = default;

    /// Reset accumulated energy
    void reset() {
        total_energy_ = 0.0;
        conduction_energy_ = 0.0;
        switching_energy_ = 0.0;
        num_samples_ = 0;
        t_start_ = 0.0;
        t_end_ = 0.0;
    }

    /// Add instantaneous power sample
    void add_sample(Real P_cond, Real dt) {
        conduction_energy_ += P_cond * dt;
        total_energy_ += P_cond * dt;
        num_samples_++;
        t_end_ += dt;
    }

    /// Add switching event energy
    void add_switching_event(Real E_sw) {
        switching_energy_ += E_sw;
        total_energy_ += E_sw;
    }

    /// Get total accumulated energy (J)
    [[nodiscard]] Real total_energy() const { return total_energy_; }

    /// Get conduction energy (J)
    [[nodiscard]] Real conduction_energy() const { return conduction_energy_; }

    /// Get switching energy (J)
    [[nodiscard]] Real switching_energy() const { return switching_energy_; }

    /// Get average power (W)
    [[nodiscard]] Real average_power() const {
        Real duration = t_end_ - t_start_;
        if (duration <= 0) return 0.0;
        return total_energy_ / duration;
    }

    /// Get average conduction power (W)
    [[nodiscard]] Real average_conduction_power() const {
        Real duration = t_end_ - t_start_;
        if (duration <= 0) return 0.0;
        return conduction_energy_ / duration;
    }

    /// Get average switching power (W)
    [[nodiscard]] Real average_switching_power() const {
        Real duration = t_end_ - t_start_;
        if (duration <= 0) return 0.0;
        return switching_energy_ / duration;
    }

    /// Get simulation duration
    [[nodiscard]] Real duration() const { return t_end_ - t_start_; }

    /// Get number of samples
    [[nodiscard]] std::size_t num_samples() const { return num_samples_; }

private:
    Real total_energy_ = 0.0;
    Real conduction_energy_ = 0.0;
    Real switching_energy_ = 0.0;
    std::size_t num_samples_ = 0;
    Real t_start_ = 0.0;
    Real t_end_ = 0.0;
};

// =============================================================================
// Efficiency Calculator
// =============================================================================

/// Calculate converter efficiency from power measurements
class EfficiencyCalculator {
public:
    /// Calculate efficiency from input/output power
    [[nodiscard]] static Real from_power(Real P_in, Real P_out) {
        if (P_in <= 0) return 0.0;
        return P_out / P_in;
    }

    /// Calculate efficiency from output power and losses
    [[nodiscard]] static Real from_losses(Real P_out, Real P_loss) {
        Real P_in = P_out + P_loss;
        if (P_in <= 0) return 0.0;
        return P_out / P_in;
    }

    /// Calculate losses from efficiency and output power
    [[nodiscard]] static Real losses_from_efficiency(Real eta, Real P_out) {
        if (eta <= 0 || eta >= 1.0) return 0.0;
        return P_out * (1.0 / eta - 1.0);
    }

    /// Calculate input power from efficiency and output power
    [[nodiscard]] static Real input_power(Real eta, Real P_out) {
        if (eta <= 0) return 0.0;
        return P_out / eta;
    }
};

// =============================================================================
// Loss Result Structure
// =============================================================================

/// Complete loss analysis result
struct LossResult {
    std::string device_name;
    LossBreakdown breakdown;           ///< Loss breakdown by type
    Real total_energy = 0.0;           ///< Total energy dissipated (J)
    Real average_power = 0.0;          ///< Average power loss (W)
    Real peak_power = 0.0;             ///< Peak instantaneous power (W)
    Real rms_current = 0.0;            ///< RMS current through device (A)
    Real avg_current = 0.0;            ///< Average current (A)
    Real efficiency_contribution = 0.0; ///< Contribution to total losses (%)
    std::vector<Real> power_waveform;  ///< Instantaneous power vs time
    std::vector<Real> times;           ///< Time points

    /// Compute statistics from waveform
    void compute_stats() {
        if (power_waveform.empty()) return;

        peak_power = *std::max_element(power_waveform.begin(), power_waveform.end());
        average_power = std::accumulate(power_waveform.begin(), power_waveform.end(), 0.0)
                       / static_cast<Real>(power_waveform.size());
    }
};

/// System-wide loss summary
struct SystemLossSummary {
    std::vector<LossResult> device_losses;  ///< Per-device losses
    Real total_loss = 0.0;                  ///< Total system loss (W)
    Real total_conduction = 0.0;            ///< Total conduction loss (W)
    Real total_switching = 0.0;             ///< Total switching loss (W)
    Real input_power = 0.0;                 ///< System input power (W)
    Real output_power = 0.0;                ///< System output power (W)
    Real efficiency = 0.0;                  ///< System efficiency (0-1)

    /// Compute system totals from device losses
    void compute_totals() {
        total_loss = 0.0;
        total_conduction = 0.0;
        total_switching = 0.0;

        for (const auto& dev : device_losses) {
            total_loss += dev.breakdown.total();
            total_conduction += dev.breakdown.conduction;
            total_switching += dev.breakdown.switching();
        }

        if (input_power > 0) {
            output_power = input_power - total_loss;
            efficiency = output_power / input_power;
        }

        // Compute efficiency contribution for each device
        if (total_loss > 0) {
            for (auto& dev : device_losses) {
                dev.efficiency_contribution =
                    100.0 * dev.breakdown.total() / total_loss;
            }
        }
    }
};

} // namespace pulsim::v1
