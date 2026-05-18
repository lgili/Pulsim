#pragma once

// =============================================================================
// PulsimCore - System-Level Loss Aggregation
// =============================================================================
// Per-device loss parameters (Rds_on(T_j), V_F0, Eon_25, Qrr, R_th_ja, ...)
// live on each device own Params struct so the stamp uses them directly:
// MOSFETParams, IGBTParams, RealisticDiodeParams, ResistorParams,
// CapacitorParams, InductorParams. That is the single source of truth for
// electrothermal coupling.
//
// This header still owns the reusable system-level types consumed by
// DefaultLossService and exposed via SimulationResult.loss_summary:
//   - LossAccumulator    : reusable per-device energy bucket (J)
//   - LossBreakdown      : conduction / turn_on / turn_off / reverse_recovery
//   - LossResult         : one device loss report (with optional waveform)
//   - SystemLossSummary  : whole-converter aggregate
//   - EfficiencyCalculator : eta = P_out / P_in utility
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace pulsim::v1 {

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
