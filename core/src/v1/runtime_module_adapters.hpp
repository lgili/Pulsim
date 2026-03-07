/**
 * @file runtime_module_adapters.hpp
 * @brief Internal runtime module adapters for modular transient concerns.
 */

#pragma once

#include "pulsim/v1/simulation.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
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
 * @brief Owns loss-service lifecycle and accepted-step loss accounting hooks.
 */
class LossAccountingModule final {
public:
    LossAccountingModule(const SimulationOptions& options,
                         const TransientServiceRegistry& services);

    /// Resets loss accounting service state for a new transient run.
    void on_run_initialize() const;

    /// Commits one accepted electrical segment into the loss accounting service.
    void on_step_accepted(const Vector& state,
                          Real dt_segment,
                          std::span<const Real> thermal_scale) const;

    /// Indicates whether sampled loss channels are enabled for this run.
    [[nodiscard]] bool has_trace_enabled() const { return has_loss_trace_; }

    /// Exposes the latest per-device loss power vectors for telemetry sampling.
    [[nodiscard]] std::span<const Real> last_device_conduction_power() const;
    [[nodiscard]] std::span<const Real> last_device_turn_on_power() const;
    [[nodiscard]] std::span<const Real> last_device_turn_off_power() const;
    [[nodiscard]] std::span<const Real> last_device_reverse_recovery_power() const;
    [[nodiscard]] std::span<const Real> last_device_power() const;

    /// Finalizes system/device loss summary tables.
    void on_finalize(SimulationResult& result, Real duration) const;

private:
    const TransientServiceRegistry& services_;
    bool has_loss_trace_ = false;
};

/**
 * @brief Owns thermal-service lifecycle and accepted-step thermal coupling hooks.
 */
class ThermalCouplingModule final {
public:
    ThermalCouplingModule(const SimulationOptions& options,
                          const TransientServiceRegistry& services);

    /// Resets thermal service state for a new transient run.
    void on_run_initialize() const;

    /// Commits one accepted segment using latest loss power for electrothermal coupling.
    void on_step_accepted(Real dt_segment, std::span<const Real> device_power) const;

    /// Indicates whether sampled thermal channels are enabled for this run.
    [[nodiscard]] bool has_trace_enabled() const { return has_thermal_trace_; }

    /// Exposes thermal scale and device temperature lookups for coupled modules.
    [[nodiscard]] std::span<const Real> thermal_scale_vector() const;
    [[nodiscard]] bool is_device_enabled(std::size_t device_index) const;
    [[nodiscard]] Real device_temperature(std::size_t device_index) const;

    /// Finalizes thermal summary tables.
    void on_finalize(SimulationResult& result) const;

private:
    const TransientServiceRegistry& services_;
    bool has_thermal_trace_ = false;
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
                                  const LossAccountingModule& loss_module,
                                  const ThermalCouplingModule& thermal_module,
                                  SimulationResult& result,
                                  std::size_t sample_reserve,
                                  Real initial_time);

    /// Accumulates trace interval loss energy from latest accepted-step power vectors.
    void on_step_accepted(Real dt_segment);

    /// Emits one sample for canonical loss/thermal channels.
    void on_sample_emit(Real sample_time, std::size_t sample_count);

    /// Finalizes loss/thermal summaries and validates channel consistency contracts.
    void on_finalize();

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
    void finalize_component_electrothermal();
    void validate_electrothermal_consistency();

    const Circuit& circuit_;
    const LossAccountingModule& loss_module_;
    const ThermalCouplingModule& thermal_module_;
    SimulationResult& result_;
    std::size_t sample_reserve_ = 0;

    bool losses_enabled_ = false;
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
