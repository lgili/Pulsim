/**
 * @file runtime_module_adapters.hpp
 * @brief Internal runtime module adapters for transient channel emission.
 */

#pragma once

#include "pulsim/v1/simulation.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace pulsim::v1 {

/**
 * @brief Emits canonical loss/thermal virtual channels through module-style hooks.
 *
 * This internal adapter isolates electrothermal channel logic from the transient
 * orchestrator while preserving the existing public channel and metadata contract.
 */
class ElectrothermalTelemetryModule final {
public:
    ElectrothermalTelemetryModule(const Circuit& circuit,
                                  const SimulationOptions& options,
                                  const TransientServiceRegistry& services,
                                  SimulationResult& result,
                                  std::size_t sample_reserve,
                                  Real initial_time);

    /// Accumulates per-device segment energy for an accepted electrical interval.
    void on_step_accepted(Real dt_segment);

    /// Emits one sample for canonical loss/thermal channels.
    void on_sample_emit(Real sample_time, std::size_t sample_count);

private:
    struct ThermalTraceBinding {
        std::size_t device_index = 0;
        std::string channel_name;
    };

    struct LossTraceBinding {
        std::size_t device_index = 0;
        std::string conduction_channel;
        std::string turn_on_channel;
        std::string turn_off_channel;
        std::string reverse_recovery_channel;
        std::string total_channel;
    };

    [[nodiscard]] static Real sanitize_power(Real value);
    void push_series_value(std::vector<Real>& series, Real value);
    void ensure_prefix(std::vector<Real>& series, std::size_t sample_count, Real fill_value);
    void initialize_loss_interval();
    void initialize_loss_channels();
    void initialize_thermal_channels();
    void sample_loss_channels(Real sample_time, std::size_t sample_count);
    void sample_thermal_channels(std::size_t sample_count);

    const Circuit& circuit_;
    const TransientServiceRegistry& services_;
    SimulationResult& result_;
    std::size_t sample_reserve_ = 0;

    bool has_loss_trace_ = false;
    bool has_thermal_trace_ = false;
    bool loss_interval_initialized_ = false;
    bool loss_channels_initialized_ = false;
    bool thermal_channels_initialized_ = false;
    Real loss_interval_start_time_ = 0.0;

    std::vector<ThermalTraceBinding> thermal_bindings_;
    std::vector<LossTraceBinding> loss_bindings_;
    std::vector<Real> loss_interval_cond_energy_;
    std::vector<Real> loss_interval_turn_on_energy_;
    std::vector<Real> loss_interval_turn_off_energy_;
    std::vector<Real> loss_interval_reverse_recovery_energy_;
    std::vector<Real> loss_interval_total_energy_;
};

}  // namespace pulsim::v1
