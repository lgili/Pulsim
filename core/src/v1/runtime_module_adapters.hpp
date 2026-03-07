/**
 * @file runtime_module_adapters.hpp
 * @brief Internal runtime module adapters for transient channel emission.
 */

#pragma once

#include "pulsim/v1/simulation.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace pulsim::v1 {

/**
 * @brief Runtime watcher state for forced-switch event transitions.
 */
struct ForcedSwitchEventMonitor {
    std::string name;
    std::size_t device_index = 0;
    Index t1 = -1;
    Index t2 = -1;
    std::optional<bool> was_forced_on;
};

/// Callback invoked when forced-switch state transition should emit an event.
using ForcedSwitchEventEmitter = std::function<void(
    const ForcedSwitchEventMonitor& monitor,
    const Vector& state,
    Real event_time,
    bool new_state)>;

/**
 * @brief Detects forced-switch transitions and emits deterministic switch events.
 */
class EventTopologyModule final {
public:
    EventTopologyModule(const SimulationOptions& options,
                        Circuit& circuit,
                        std::vector<ForcedSwitchEventMonitor> forced_monitors);

    /// Captures initial forced states so first sampled point does not emit false transitions.
    void on_run_initialize();

    /// Checks forced-state transitions and emits events for this output sample.
    void on_sample_emit(const Vector& state,
                        Real sample_time,
                        std::size_t sample_count,
                        bool is_terminal_sample,
                        const ForcedSwitchEventEmitter& emit_event);

private:
    const SimulationOptions& options_;
    Circuit& circuit_;
    std::vector<ForcedSwitchEventMonitor> forced_monitors_;
};

/**
 * @brief Emits mixed-domain control channels through module-style sample hooks.
 *
 * The module owns mixed-domain channel registration/append semantics while
 * preserving existing metadata and channel naming contracts.
 */
class ControlMixedDomainModule final {
public:
    ControlMixedDomainModule(Circuit& circuit,
                             SimulationResult& result,
                             std::size_t sample_reserve);

    /// Ensures base virtual-channel metadata from runtime circuit is present.
    void ensure_base_metadata();

    /// Appends mixed-domain channel samples for the current accepted output sample.
    void on_sample_emit(const Vector& state, Real sample_time, std::size_t sample_count);

private:
    void push_series_value(std::vector<Real>& series, Real value);
    void ensure_prefix(std::vector<Real>& series, std::size_t sample_count, Real fill_value);

    Circuit& circuit_;
    SimulationResult& result_;
    std::size_t sample_reserve_ = 0;
    bool metadata_initialized_ = false;
};

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

    /// Commits accepted-step loss/thermal state and accumulates trace interval energy.
    void on_step_accepted(const Vector& state, Real dt_segment);

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
