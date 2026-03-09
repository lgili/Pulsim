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
#include <string_view>
#include <unordered_map>
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

/**
 * @brief Runtime watcher state for threshold-driven switch transitions.
 */
struct SwitchThresholdEventMonitor {
    std::string name;
    Index ctrl = -1;
    Index t1 = -1;
    Index t2 = -1;
    Real v_threshold = 0.0;
    bool was_on = false;
};

/// Callback invoked when forced-switch state transition should emit an event.
using ForcedSwitchEventEmitter = std::function<void(
    const ForcedSwitchEventMonitor& monitor,
    const Vector& state,
    Real event_time,
    bool new_state)>;

/// Callback that refines threshold-crossing event time inside one accepted step.
using ThresholdEventRefiner = std::function<bool(
    const SwitchThresholdEventMonitor& monitor,
    Real t_start,
    Real t_end,
    const Vector& x_start,
    Real& t_event,
    Vector& x_event)>;

/// Callback that emits threshold-driven switch transition events.
using ThresholdEventEmitter = std::function<void(
    const SwitchThresholdEventMonitor& monitor,
    Real event_time,
    const Vector& event_state,
    bool new_state)>;

/**
 * @brief Detects forced-switch transitions and emits deterministic switch events.
 */
class EventTopologyModule final {
public:
    EventTopologyModule(const SimulationOptions& options,
                        Circuit& circuit,
                        std::vector<ForcedSwitchEventMonitor> forced_monitors,
                        std::vector<SwitchThresholdEventMonitor> threshold_monitors);

    /// Captures initial forced states so first sampled point does not emit false transitions.
    void on_run_initialize();

    /// Initializes threshold-switch monitor states from the current state vector.
    void initialize_threshold_states(const Vector& state);

    /// Returns true when any threshold-switch control is close to switching boundary.
    [[nodiscard]] bool near_threshold(const Vector& state) const;

    /// Finds earliest threshold-crossing candidate time in one accepted electrical step.
    [[nodiscard]] std::optional<Real> earliest_threshold_crossing_time(
        Real t_start,
        Real dt_used,
        const Vector& x_start,
        const Vector& x_end,
        const ThresholdEventRefiner& refine_event_time) const;

    /// Emits threshold-driven transition events and updates monitor states after accepted step.
    void on_step_accepted(Real t_start,
                          Real dt_used,
                          const Vector& x_start,
                          const Vector& x_end,
                          const ThresholdEventRefiner& refine_event_time,
                          const ThresholdEventEmitter& emit_event);

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
    std::vector<SwitchThresholdEventMonitor> threshold_monitors_;
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

    /// Evaluates mixed-domain control once per accepted electrical step.
    void on_step_accepted(const Vector& state, Real step_time);

    /// Appends mixed-domain channel samples for the current accepted output sample.
    void on_sample_emit(const Vector& state, Real sample_time, std::size_t sample_count);

private:
    void push_series_value(std::vector<Real>& series, Real value);
    void ensure_prefix(std::vector<Real>& series, std::size_t sample_count, Real fill_value);
    void evaluate_step(const Vector& state, Real step_time);

    Circuit& circuit_;
    SimulationResult& result_;
    std::size_t sample_reserve_ = 0;
    bool metadata_initialized_ = false;
    bool has_cached_step_ = false;
    Real cached_step_time_ = 0.0;
    std::unordered_map<std::string, Real> cached_channel_values_;
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

    struct MagneticLossSummaryBinding {
        std::string component_name;
        std::string channel_name;
        std::string summary_name;
    };

    [[nodiscard]] static Real sanitize_power(Real value);
    [[nodiscard]] static bool loss_policy_includes_summary(std::string_view token);
    void push_series_value(std::vector<Real>& series, Real value);
    void ensure_prefix(std::vector<Real>& series, std::size_t sample_count, Real fill_value);
    void initialize_loss_interval();
    void initialize_loss_channels();
    void initialize_thermal_channels();
    void initialize_magnetic_loss_summary_bindings();
    void sample_loss_channels(Real sample_time, std::size_t sample_count);
    void sample_thermal_channels(std::size_t sample_count);
    void merge_magnetic_core_loss_into_loss_summary();
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
    bool magnetic_loss_bindings_initialized_ = false;
    Real loss_interval_start_time_ = 0.0;

    std::vector<ThermalTraceBinding> thermal_bindings_;
    std::vector<LossTraceBinding> loss_bindings_;
    std::vector<MagneticLossSummaryBinding> magnetic_loss_summary_bindings_;
    std::vector<Real> loss_interval_cond_energy_;
    std::vector<Real> loss_interval_turn_on_energy_;
    std::vector<Real> loss_interval_turn_off_energy_;
    std::vector<Real> loss_interval_reverse_recovery_energy_;
    std::vector<Real> loss_interval_total_energy_;
};

/**
 * @brief Bundles runtime modules behind unified hook entrypoints for transient orchestration.
 *
 * This adapter keeps module ownership localized while exposing policy-only operations to
 * `Simulator::run_transient_native_impl(...)`.
 */
class RuntimeModuleOrchestrator final {
public:
    RuntimeModuleOrchestrator(const SimulationOptions& options,
                              Circuit& circuit,
                              const TransientServiceRegistry& services,
                              SimulationResult& result,
                              std::size_t sample_reserve,
                              Real initial_time,
                              std::vector<ForcedSwitchEventMonitor> forced_monitors,
                              std::vector<SwitchThresholdEventMonitor> threshold_monitors);

    /// Initializes module run state and threshold monitors from the initial state vector.
    void on_run_initialize(const Vector& initial_state);

    /// Emits per-sample channel updates from all relevant runtime modules.
    void on_sample_emit(const Vector& state,
                        Real sample_time,
                        std::size_t sample_count,
                        bool has_virtual_components,
                        bool is_terminal_sample,
                        const ForcedSwitchEventEmitter& emit_forced_event);

    /// Indicates whether current state is close to threshold-driven switching boundaries.
    [[nodiscard]] bool near_threshold(const Vector& state) const;

    /// Finds earliest threshold-crossing candidate inside one accepted step.
    [[nodiscard]] std::optional<Real> earliest_threshold_crossing_time(
        Real t_start,
        Real dt_used,
        const Vector& x_start,
        const Vector& x_end,
        const ThresholdEventRefiner& refine_event_time) const;

    /// Applies accepted-step event/loss/thermal/electrothermal hooks.
    void on_step_accepted(Real t_start,
                          Real dt_used,
                          const Vector& x_start,
                          const Vector& x_end,
                          const ThresholdEventRefiner& refine_event_time,
                          const ThresholdEventEmitter& emit_threshold_event);

    /// Applies hold-step loss/thermal/electrothermal hooks without threshold-event emission.
    void on_hold_step_accepted(const Vector& held_state, Real dt_used);

    /// Finalizes module summaries/channels and post-run consistency checks.
    void on_finalize();

private:
    void push_series_value(std::vector<Real>& series, Real value);
    void ensure_channel_prefix(std::size_t sample_count, Real fill_value);
    void fill_missing_sample(std::size_t sample_count, Real fill_value);

    const SimulationOptions& options_;
    SimulationResult& result_;

    ControlMixedDomainModule control_mixed_module_;
    EventTopologyModule event_topology_module_;
    LossAccountingModule loss_module_;
    ThermalCouplingModule thermal_module_;
    ElectrothermalTelemetryModule electrothermal_module_;
};

}  // namespace pulsim::v1
