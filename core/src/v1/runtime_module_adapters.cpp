/**
 * @file runtime_module_adapters.cpp
 * @brief Internal runtime module adapters for loss/thermal channel emission.
 */

#include "runtime_module_adapters.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

namespace pulsim::v1 {

namespace {
// Loss consistency is compared between step-averaged telemetry and sampled channels.
// Pulsed channels carry quantization uncertainty on sampled integration (order P_peak * dt).
constexpr Real kLossConsistencyRelTol = 1e-2;
constexpr Real kLossConsistencyAbsTol = 1e-8;

/// Relative/absolute floating-point comparator used by post-run consistency guards.
[[nodiscard]] bool nearly_equal(Real a,
                                Real b,
                                Real rel_tol = 1e-6,
                                Real abs_tol = 1e-9) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    const Real diff = std::abs(a - b);
    if (diff <= abs_tol) {
        return true;
    }
    const Real scale = std::max<Real>({Real{1.0}, std::abs(a), std::abs(b)});
    return diff <= rel_tol * scale;
}
}  // namespace

EventTopologyModule::EventTopologyModule(const SimulationOptions& options,
                                         Circuit& circuit,
                                         std::vector<ForcedSwitchEventMonitor> forced_monitors)
    : options_(options),
      circuit_(circuit),
      forced_monitors_(std::move(forced_monitors)) {}

void EventTopologyModule::on_run_initialize() {
    for (auto& monitor : forced_monitors_) {
        monitor.was_forced_on = circuit_.forced_state_for_device(monitor.device_index);
    }
}

void EventTopologyModule::on_sample_emit(const Vector& state,
                                         Real sample_time,
                                         std::size_t sample_count,
                                         bool is_terminal_sample,
                                         const ForcedSwitchEventEmitter& emit_event) {
    if (!options_.enable_events) {
        return;
    }
    if (sample_count <= 1 || is_terminal_sample || forced_monitors_.empty()) {
        return;
    }

    for (auto& monitor : forced_monitors_) {
        const std::optional<bool> current_forced =
            circuit_.forced_state_for_device(monitor.device_index);
        if (!monitor.was_forced_on.has_value()) {
            monitor.was_forced_on = current_forced;
            continue;
        }

        if (monitor.was_forced_on.has_value() &&
            current_forced.has_value() &&
            *monitor.was_forced_on != *current_forced) {
            emit_event(monitor, state, sample_time, *current_forced);
        }

        monitor.was_forced_on = current_forced;
    }
}

ControlMixedDomainModule::ControlMixedDomainModule(Circuit& circuit,
                                                   SimulationResult& result,
                                                   std::size_t sample_reserve)
    : circuit_(circuit),
      result_(result),
      sample_reserve_(sample_reserve) {}

void ControlMixedDomainModule::ensure_base_metadata() {
    if (metadata_initialized_) {
        return;
    }
    result_.virtual_channel_metadata = circuit_.virtual_channel_metadata();
    result_.virtual_channels.reserve(result_.virtual_channel_metadata.size());
    metadata_initialized_ = true;
}

void ControlMixedDomainModule::push_series_value(std::vector<Real>& series, Real value) {
    const std::size_t capacity_before = series.capacity();
    series.push_back(value);
    if (series.capacity() != capacity_before) {
        result_.backend_telemetry.virtual_channel_reallocations += 1;
    }
}

void ControlMixedDomainModule::ensure_prefix(std::vector<Real>& series,
                                             std::size_t sample_count,
                                             Real fill_value) {
    while (series.size() + 1 < sample_count) {
        push_series_value(series, fill_value);
    }
}

void ControlMixedDomainModule::on_sample_emit(const Vector& state,
                                              Real sample_time,
                                              std::size_t sample_count) {
    if (circuit_.num_virtual_components() == 0) {
        return;
    }
    ensure_base_metadata();

    const MixedDomainStepResult mixed_step = circuit_.execute_mixed_domain_step(state, sample_time);
    if (result_.mixed_domain_phase_order.empty()) {
        result_.mixed_domain_phase_order = mixed_step.phase_order;
    }

    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (const auto& [channel, value] : mixed_step.channel_values) {
        auto [it, inserted] = result_.virtual_channels.try_emplace(channel);
        auto& series = it->second;
        if (inserted && sample_reserve_ > 0) {
            series.reserve(sample_reserve_);
        }
        ensure_prefix(series, sample_count, nan);
        push_series_value(series, value);

        if (!result_.virtual_channel_metadata.contains(channel)) {
            result_.virtual_channel_metadata[channel] = VirtualChannelMetadata{
                "virtual",
                channel,
                channel,
                "control",
                "",
                {}
            };
        }
    }
}

ElectrothermalTelemetryModule::ElectrothermalTelemetryModule(
    const Circuit& circuit,
    const SimulationOptions& options,
    const TransientServiceRegistry& services,
    SimulationResult& result,
    std::size_t sample_reserve,
    Real initial_time)
    : circuit_(circuit),
      services_(services),
      result_(result),
      sample_reserve_(sample_reserve),
      losses_enabled_(options.enable_losses),
      has_loss_trace_(options.enable_losses &&
                      static_cast<bool>(services.loss_service)),
      has_thermal_trace_(options.enable_losses &&
                         options.thermal.enable &&
                         static_cast<bool>(services.thermal_service)),
      loss_interval_start_time_(initial_time) {}

Real ElectrothermalTelemetryModule::sanitize_power(Real value) {
    if (!std::isfinite(value) || value < 0.0) {
        return 0.0;
    }
    return value;
}

void ElectrothermalTelemetryModule::push_series_value(std::vector<Real>& series, Real value) {
    const std::size_t capacity_before = series.capacity();
    series.push_back(value);
    if (series.capacity() != capacity_before) {
        result_.backend_telemetry.virtual_channel_reallocations += 1;
    }
}

void ElectrothermalTelemetryModule::ensure_prefix(std::vector<Real>& series,
                                                  std::size_t sample_count,
                                                  Real fill_value) {
    while (series.size() + 1 < sample_count) {
        push_series_value(series, fill_value);
    }
}

void ElectrothermalTelemetryModule::initialize_loss_interval() {
    if (!has_loss_trace_) {
        return;
    }
    const std::size_t device_count = circuit_.connections().size();
    auto reset_bucket = [device_count](std::vector<Real>& bucket) {
        if (bucket.size() != device_count) {
            bucket.assign(device_count, 0.0);
        } else {
            std::fill(bucket.begin(), bucket.end(), 0.0);
        }
    };
    reset_bucket(loss_interval_cond_energy_);
    reset_bucket(loss_interval_turn_on_energy_);
    reset_bucket(loss_interval_turn_off_energy_);
    reset_bucket(loss_interval_reverse_recovery_energy_);
    reset_bucket(loss_interval_total_energy_);
    loss_interval_initialized_ = true;
}

void ElectrothermalTelemetryModule::initialize_loss_channels() {
    if (!has_loss_trace_ || loss_channels_initialized_) {
        return;
    }

    const auto& conns = circuit_.connections();
    loss_bindings_.reserve(conns.size());
    if (sample_reserve_ > 0) {
        result_.virtual_channels.reserve(result_.virtual_channels.size() + conns.size() * 5);
    }

    auto register_loss_channel = [&](const std::string& channel,
                                     const std::string& source,
                                     const std::string& quantity) {
        auto [it, inserted] = result_.virtual_channels.try_emplace(channel);
        if (inserted && sample_reserve_ > 0) {
            it->second.reserve(sample_reserve_);
        }
        result_.virtual_channel_metadata.try_emplace(
            channel,
            VirtualChannelMetadata{
                quantity,
                channel,
                source,
                "loss",
                "W",
                {}
            });
    };

    for (std::size_t i = 0; i < conns.size(); ++i) {
        const std::string source = conns[i].name;
        LossTraceBinding binding;
        binding.device_index = i;
        binding.conduction_channel = "Pcond(" + source + ")";
        binding.turn_on_channel = "Psw_on(" + source + ")";
        binding.turn_off_channel = "Psw_off(" + source + ")";
        binding.reverse_recovery_channel = "Prr(" + source + ")";
        binding.total_channel = "Ploss(" + source + ")";
        register_loss_channel(binding.conduction_channel, source, "loss_trace_conduction");
        register_loss_channel(binding.turn_on_channel, source, "loss_trace_turn_on");
        register_loss_channel(binding.turn_off_channel, source, "loss_trace_turn_off");
        register_loss_channel(binding.reverse_recovery_channel, source, "loss_trace_reverse_recovery");
        register_loss_channel(binding.total_channel, source, "loss_trace_total");
        loss_bindings_.push_back(std::move(binding));
    }

    loss_channels_initialized_ = true;
}

void ElectrothermalTelemetryModule::initialize_thermal_channels() {
    if (!has_thermal_trace_ || thermal_channels_initialized_) {
        return;
    }

    const auto& conns = circuit_.connections();
    thermal_bindings_.reserve(conns.size());
    if (sample_reserve_ > 0) {
        result_.virtual_channels.reserve(result_.virtual_channels.size() + conns.size());
    }

    for (std::size_t i = 0; i < conns.size(); ++i) {
        if (!services_.thermal_service->is_device_enabled(i)) {
            continue;
        }
        const std::string channel = "T(" + conns[i].name + ")";
        auto [it, inserted] = result_.virtual_channels.try_emplace(channel);
        if (inserted && sample_reserve_ > 0) {
            it->second.reserve(sample_reserve_);
        }
        result_.virtual_channel_metadata.try_emplace(
            channel,
            VirtualChannelMetadata{
                "thermal_trace",
                channel,
                conns[i].name,
                "thermal",
                "degC",
                {}
            });
        thermal_bindings_.push_back(ThermalTraceBinding{i, channel});
    }

    thermal_channels_initialized_ = true;
}

void ElectrothermalTelemetryModule::on_step_accepted(const Vector& state, Real dt_segment) {
    if (services_.loss_service && services_.thermal_service) {
        services_.loss_service->commit_accepted_segment(
            state,
            dt_segment,
            services_.thermal_service->thermal_scale_vector());
        services_.thermal_service->commit_accepted_segment(
            dt_segment,
            services_.loss_service->last_device_power());
    }

    if (!has_loss_trace_ || dt_segment <= 0.0) {
        return;
    }
    if (!loss_interval_initialized_) {
        initialize_loss_interval();
    }
    if (!loss_interval_initialized_) {
        return;
    }

    const auto conduction = services_.loss_service->last_device_conduction_power();
    const auto turn_on = services_.loss_service->last_device_turn_on_power();
    const auto turn_off = services_.loss_service->last_device_turn_off_power();
    const auto reverse_recovery = services_.loss_service->last_device_reverse_recovery_power();
    const auto total = services_.loss_service->last_device_power();
    const std::size_t device_count = loss_interval_total_energy_.size();
    for (std::size_t i = 0; i < device_count; ++i) {
        const Real p_cond = i < conduction.size() ? sanitize_power(conduction[i]) : 0.0;
        const Real p_on = i < turn_on.size() ? sanitize_power(turn_on[i]) : 0.0;
        const Real p_off = i < turn_off.size() ? sanitize_power(turn_off[i]) : 0.0;
        const Real p_rr = i < reverse_recovery.size() ? sanitize_power(reverse_recovery[i]) : 0.0;
        const Real p_total = i < total.size() ? sanitize_power(total[i]) : 0.0;

        loss_interval_cond_energy_[i] += p_cond * dt_segment;
        loss_interval_turn_on_energy_[i] += p_on * dt_segment;
        loss_interval_turn_off_energy_[i] += p_off * dt_segment;
        loss_interval_reverse_recovery_energy_[i] += p_rr * dt_segment;
        loss_interval_total_energy_[i] += p_total * dt_segment;
    }
}

void ElectrothermalTelemetryModule::sample_loss_channels(Real sample_time, std::size_t sample_count) {
    if (!has_loss_trace_) {
        return;
    }

    if (!loss_channels_initialized_) {
        initialize_loss_channels();
    }
    if (!loss_interval_initialized_) {
        initialize_loss_interval();
    }

    const Real interval_duration = sample_time - loss_interval_start_time_;
    const bool valid_interval = std::isfinite(interval_duration) && interval_duration > 0.0;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();

    auto sample_loss_channel = [&](const std::string& channel_name, Real value) {
        auto channel_it = result_.virtual_channels.find(channel_name);
        if (channel_it == result_.virtual_channels.end()) {
            return;
        }
        auto& series = channel_it->second;
        ensure_prefix(series, sample_count, nan);
        push_series_value(series, value);
    };

    for (const auto& binding : loss_bindings_) {
        const std::size_t i = binding.device_index;
        const auto average_power = [&](const std::vector<Real>& energy_bucket) {
            if (!valid_interval || i >= energy_bucket.size()) {
                return Real{0.0};
            }
            return energy_bucket[i] / interval_duration;
        };

        const Real p_cond = average_power(loss_interval_cond_energy_);
        const Real p_on = average_power(loss_interval_turn_on_energy_);
        const Real p_off = average_power(loss_interval_turn_off_energy_);
        const Real p_rr = average_power(loss_interval_reverse_recovery_energy_);
        const Real p_total = average_power(loss_interval_total_energy_);

        sample_loss_channel(binding.conduction_channel, p_cond);
        sample_loss_channel(binding.turn_on_channel, p_on);
        sample_loss_channel(binding.turn_off_channel, p_off);
        sample_loss_channel(binding.reverse_recovery_channel, p_rr);
        sample_loss_channel(binding.total_channel, p_total);
    }

    if (loss_interval_initialized_) {
        if (valid_interval) {
            std::fill(loss_interval_cond_energy_.begin(), loss_interval_cond_energy_.end(), Real{0.0});
            std::fill(loss_interval_turn_on_energy_.begin(), loss_interval_turn_on_energy_.end(), Real{0.0});
            std::fill(loss_interval_turn_off_energy_.begin(), loss_interval_turn_off_energy_.end(), Real{0.0});
            std::fill(
                loss_interval_reverse_recovery_energy_.begin(),
                loss_interval_reverse_recovery_energy_.end(),
                Real{0.0});
            std::fill(loss_interval_total_energy_.begin(), loss_interval_total_energy_.end(), Real{0.0});
        }
        loss_interval_start_time_ = sample_time;
    }
}

void ElectrothermalTelemetryModule::sample_thermal_channels(std::size_t sample_count) {
    if (!has_thermal_trace_) {
        return;
    }
    if (!thermal_channels_initialized_) {
        initialize_thermal_channels();
    }

    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (const auto& binding : thermal_bindings_) {
        auto channel_it = result_.virtual_channels.find(binding.channel_name);
        if (channel_it == result_.virtual_channels.end()) {
            continue;
        }
        auto& series = channel_it->second;
        ensure_prefix(series, sample_count, nan);
        push_series_value(series, services_.thermal_service->device_temperature(binding.device_index));
    }
}

void ElectrothermalTelemetryModule::finalize_loss_summary() {
    if (!services_.loss_service) {
        return;
    }

    Real duration = 0.0;
    if (result_.time.size() >= 2) {
        duration = result_.time.back() - result_.time.front();
    }
    result_.loss_summary = services_.loss_service->finalize(duration);
}

void ElectrothermalTelemetryModule::finalize_thermal_summary() {
    if (!services_.thermal_service) {
        return;
    }

    const ThermalServiceSummary service_summary = services_.thermal_service->finalize();
    ThermalSummary summary;
    summary.enabled = service_summary.enabled;
    summary.ambient = service_summary.ambient;
    summary.max_temperature = service_summary.max_temperature;
    summary.device_temperatures.reserve(service_summary.device_temperatures.size());
    for (const auto& item : service_summary.device_temperatures) {
        DeviceThermalTelemetry telemetry;
        telemetry.device_name = item.device_name;
        telemetry.enabled = item.enabled;
        telemetry.final_temperature = item.final_temperature;
        telemetry.peak_temperature = item.peak_temperature;
        telemetry.average_temperature = item.average_temperature;
        summary.device_temperatures.push_back(std::move(telemetry));
    }
    result_.thermal_summary = std::move(summary);
}

void ElectrothermalTelemetryModule::finalize_component_electrothermal() {
    const auto& conns = circuit_.connections();
    result_.component_electrothermal.clear();
    result_.component_electrothermal.reserve(conns.size());

    std::unordered_map<std::string, const LossResult*> loss_by_name;
    loss_by_name.reserve(result_.loss_summary.device_losses.size());
    for (const auto& item : result_.loss_summary.device_losses) {
        loss_by_name[item.device_name] = &item;
    }

    std::unordered_map<std::string, const DeviceThermalTelemetry*> thermal_by_name;
    thermal_by_name.reserve(result_.thermal_summary.device_temperatures.size());
    for (const auto& item : result_.thermal_summary.device_temperatures) {
        thermal_by_name[item.device_name] = &item;
    }

    const Real ambient = result_.thermal_summary.ambient;
    for (const auto& conn : conns) {
        ComponentElectrothermalTelemetry entry;
        entry.component_name = conn.name;
        entry.final_temperature = ambient;
        entry.peak_temperature = ambient;
        entry.average_temperature = ambient;

        if (const auto loss_it = loss_by_name.find(conn.name); loss_it != loss_by_name.end()) {
            const LossResult& loss = *loss_it->second;
            entry.conduction = loss.breakdown.conduction;
            entry.turn_on = loss.breakdown.turn_on;
            entry.turn_off = loss.breakdown.turn_off;
            entry.reverse_recovery = loss.breakdown.reverse_recovery;
            entry.total_loss = loss.breakdown.total();
            entry.total_energy = loss.total_energy;
            entry.average_power = loss.average_power;
            entry.peak_power = loss.peak_power;
        }

        if (const auto thermal_it = thermal_by_name.find(conn.name); thermal_it != thermal_by_name.end()) {
            const DeviceThermalTelemetry& thermal = *thermal_it->second;
            entry.thermal_enabled = thermal.enabled;
            entry.final_temperature = thermal.final_temperature;
            entry.peak_temperature = thermal.peak_temperature;
            entry.average_temperature = thermal.average_temperature;
        }

        result_.component_electrothermal.push_back(std::move(entry));
    }
}

void ElectrothermalTelemetryModule::validate_electrothermal_consistency() {
    if (!result_.success) {
        return;
    }
    if (result_.time.empty()) {
        return;
    }

    auto fail = [&](const std::string& detail) {
        result_.success = false;
        result_.final_status = SolverStatus::NumericalError;
        result_.diagnostic = SimulationDiagnosticCode::TransientStepFailure;
        result_.message = "Electrothermal consistency failure: " + detail;
        result_.backend_telemetry.failure_reason = "electrothermal_consistency_failure";
    };

    auto find_channel = [&](const std::string& channel_name) -> const std::vector<Real>* {
        const auto it = result_.virtual_channels.find(channel_name);
        if (it == result_.virtual_channels.end()) {
            return nullptr;
        }
        if (it->second.size() != result_.time.size()) {
            fail("channel '" + channel_name + "' length mismatch");
            return nullptr;
        }
        return &it->second;
    };

    auto integrate_channel_energy = [&](const std::string& channel_name,
                                        const std::vector<Real>& series) -> std::optional<Real> {
        Real energy = 0.0;
        for (std::size_t i = 1; i < result_.time.size(); ++i) {
            const Real dt = result_.time[i] - result_.time[i - 1];
            if (!std::isfinite(dt) || dt < 0.0) {
                fail("non-monotonic or non-finite time base while integrating '" + channel_name + "'");
                return std::nullopt;
            }
            const Real power = series[i];
            if (!std::isfinite(power) || power < -1e-12) {
                fail("non-finite or negative power sample in '" + channel_name + "'");
                return std::nullopt;
            }
            energy += power * dt;
        }
        return energy;
    };

    std::unordered_map<std::string, const DeviceThermalTelemetry*> thermal_summary_by_name;
    thermal_summary_by_name.reserve(result_.thermal_summary.device_temperatures.size());
    for (const auto& row : result_.thermal_summary.device_temperatures) {
        thermal_summary_by_name[row.device_name] = &row;
    }

    std::unordered_map<std::string, const LossResult*> loss_summary_by_name;
    loss_summary_by_name.reserve(result_.loss_summary.device_losses.size());
    for (const auto& row : result_.loss_summary.device_losses) {
        loss_summary_by_name[row.device_name] = &row;
    }

    std::unordered_map<std::string, const ComponentElectrothermalTelemetry*> component_by_name;
    component_by_name.reserve(result_.component_electrothermal.size());
    for (const auto& row : result_.component_electrothermal) {
        component_by_name[row.component_name] = &row;
    }

    for (const auto& [name, thermal] : thermal_summary_by_name) {
        const std::string channel_name = "T(" + name + ")";
        const std::vector<Real>* series_ptr = find_channel(channel_name);
        if (series_ptr == nullptr) {
            return;
        }
        const auto& series = *series_ptr;
        if (series.empty()) {
            fail("empty thermal channel '" + channel_name + "'");
            return;
        }
        if (std::any_of(series.begin(), series.end(), [](Real value) { return !std::isfinite(value); })) {
            fail("non-finite sample in thermal channel '" + channel_name + "'");
            return;
        }

        const Real final_temperature = series.back();
        const Real peak_temperature = *std::max_element(series.begin(), series.end());
        const Real sum_temperature = std::accumulate(series.begin(), series.end(), Real{0.0});
        const Real average_temperature = sum_temperature / static_cast<Real>(series.size());

        if (!nearly_equal(final_temperature, thermal->final_temperature) ||
            !nearly_equal(peak_temperature, thermal->peak_temperature) ||
            !nearly_equal(average_temperature, thermal->average_temperature)) {
            fail("thermal summary mismatch for component '" + name + "'");
            return;
        }

        const auto component_it = component_by_name.find(name);
        if (component_it == component_by_name.end()) {
            fail("component_electrothermal missing thermal-enabled component '" + name + "'");
            return;
        }
        const auto* component = component_it->second;
        if (!nearly_equal(final_temperature, component->final_temperature) ||
            !nearly_equal(peak_temperature, component->peak_temperature) ||
            !nearly_equal(average_temperature, component->average_temperature)) {
            fail("component_electrothermal thermal mismatch for component '" + name + "'");
            return;
        }
    }

    if (!losses_enabled_) {
        return;
    }

    const Real duration =
        result_.time.size() >= 2 ? (result_.time.back() - result_.time.front()) : Real{0.0};
    Real max_dt = 0.0;
    for (std::size_t i = 1; i < result_.time.size(); ++i) {
        const Real dt = result_.time[i] - result_.time[i - 1];
        if (std::isfinite(dt) && dt > 0.0) {
            max_dt = std::max(max_dt, dt);
        }
    }
    max_dt = std::max(max_dt, Real{1e-18});

    auto channel_energy_quantization_tol = [&](const std::vector<Real>& series) -> Real {
        Real peak_power = 0.0;
        for (Real sample : series) {
            if (std::isfinite(sample)) {
                peak_power = std::max(peak_power, std::abs(sample));
            }
        }
        return std::max(kLossConsistencyAbsTol, peak_power * max_dt);
    };
    auto channel_average_power_quantization_tol = [&](const std::vector<Real>& series) -> Real {
        if (duration <= 0.0) {
            return kLossConsistencyAbsTol;
        }
        return channel_energy_quantization_tol(series) / duration;
    };

    Real aggregated_channel_energy = 0.0;
    Real aggregated_channel_energy_tol = 0.0;
    const auto& conns = circuit_.connections();
    for (const auto& conn : conns) {
        const std::string p_cond_name = "Pcond(" + conn.name + ")";
        const std::string p_on_name = "Psw_on(" + conn.name + ")";
        const std::string p_off_name = "Psw_off(" + conn.name + ")";
        const std::string p_rr_name = "Prr(" + conn.name + ")";
        const std::string p_total_name = "Ploss(" + conn.name + ")";

        const std::vector<Real>* p_cond = find_channel(p_cond_name);
        if (p_cond == nullptr) return;
        const std::vector<Real>* p_on = find_channel(p_on_name);
        if (p_on == nullptr) return;
        const std::vector<Real>* p_off = find_channel(p_off_name);
        if (p_off == nullptr) return;
        const std::vector<Real>* p_rr = find_channel(p_rr_name);
        if (p_rr == nullptr) return;
        const std::vector<Real>* p_total = find_channel(p_total_name);
        if (p_total == nullptr) return;

        const std::optional<Real> e_cond_opt = integrate_channel_energy(p_cond_name, *p_cond);
        if (!e_cond_opt.has_value()) return;
        const std::optional<Real> e_on_opt = integrate_channel_energy(p_on_name, *p_on);
        if (!e_on_opt.has_value()) return;
        const std::optional<Real> e_off_opt = integrate_channel_energy(p_off_name, *p_off);
        if (!e_off_opt.has_value()) return;
        const std::optional<Real> e_rr_opt = integrate_channel_energy(p_rr_name, *p_rr);
        if (!e_rr_opt.has_value()) return;
        const std::optional<Real> e_total_opt = integrate_channel_energy(p_total_name, *p_total);
        if (!e_total_opt.has_value()) return;

        const Real e_cond = *e_cond_opt;
        const Real e_on = *e_on_opt;
        const Real e_off = *e_off_opt;
        const Real e_rr = *e_rr_opt;
        const Real e_total = *e_total_opt;
        const Real e_breakdown = e_cond + e_on + e_off + e_rr;
        const Real e_cond_tol = channel_energy_quantization_tol(*p_cond);
        const Real e_on_tol = channel_energy_quantization_tol(*p_on);
        const Real e_off_tol = channel_energy_quantization_tol(*p_off);
        const Real e_rr_tol = channel_energy_quantization_tol(*p_rr);
        const Real e_total_tol = channel_energy_quantization_tol(*p_total);
        const Real e_breakdown_tol = e_cond_tol + e_on_tol + e_off_tol + e_rr_tol;
        if (!nearly_equal(
                e_total,
                e_breakdown,
                kLossConsistencyRelTol,
                std::max(e_total_tol, e_breakdown_tol))) {
            fail("Ploss channel mismatch against breakdown for component '" + conn.name + "'");
            return;
        }
        aggregated_channel_energy += e_total;
        aggregated_channel_energy_tol += e_total_tol;

        const Real avg_cond = duration > 0.0 ? e_cond / duration : 0.0;
        const Real avg_on = duration > 0.0 ? e_on / duration : 0.0;
        const Real avg_off = duration > 0.0 ? e_off / duration : 0.0;
        const Real avg_rr = duration > 0.0 ? e_rr / duration : 0.0;
        const Real avg_total = duration > 0.0 ? e_total / duration : 0.0;
        const Real avg_cond_tol = channel_average_power_quantization_tol(*p_cond);
        const Real avg_on_tol = channel_average_power_quantization_tol(*p_on);
        const Real avg_off_tol = channel_average_power_quantization_tol(*p_off);
        const Real avg_rr_tol = channel_average_power_quantization_tol(*p_rr);
        const Real avg_total_tol = channel_average_power_quantization_tol(*p_total);

        const auto component_it = component_by_name.find(conn.name);
        if (component_it == component_by_name.end()) {
            fail("component_electrothermal missing row for component '" + conn.name + "'");
            return;
        }
        const auto* component = component_it->second;
        if (!nearly_equal(component->conduction, avg_cond, kLossConsistencyRelTol, avg_cond_tol) ||
            !nearly_equal(component->turn_on, avg_on, kLossConsistencyRelTol, avg_on_tol) ||
            !nearly_equal(component->turn_off, avg_off, kLossConsistencyRelTol, avg_off_tol) ||
            !nearly_equal(component->reverse_recovery, avg_rr, kLossConsistencyRelTol, avg_rr_tol) ||
            !nearly_equal(component->total_loss, avg_total, kLossConsistencyRelTol, avg_total_tol) ||
            !nearly_equal(component->total_energy, e_total, kLossConsistencyRelTol, e_total_tol)) {
            fail("component_electrothermal loss mismatch for component '" + conn.name + "'");
            return;
        }

        const auto loss_it = loss_summary_by_name.find(conn.name);
        if (loss_it == loss_summary_by_name.end()) {
            if (!nearly_equal(e_total, 0.0, 1e-6, 1e-10)) {
                fail("loss_summary missing non-zero component '" + conn.name + "'");
                return;
            }
            continue;
        }
        const auto* loss = loss_it->second;
        if (!nearly_equal(loss->breakdown.conduction, avg_cond, kLossConsistencyRelTol, avg_cond_tol) ||
            !nearly_equal(loss->breakdown.turn_on, avg_on, kLossConsistencyRelTol, avg_on_tol) ||
            !nearly_equal(loss->breakdown.turn_off, avg_off, kLossConsistencyRelTol, avg_off_tol) ||
            !nearly_equal(loss->breakdown.reverse_recovery, avg_rr, kLossConsistencyRelTol, avg_rr_tol) ||
            !nearly_equal(loss->average_power, avg_total, kLossConsistencyRelTol, avg_total_tol) ||
            !nearly_equal(loss->total_energy, e_total, kLossConsistencyRelTol, e_total_tol)) {
            fail("loss_summary mismatch for component '" + conn.name + "'");
            return;
        }
    }

    if (duration > 0.0 &&
        !nearly_equal(result_.loss_summary.total_loss * duration,
                      aggregated_channel_energy,
                      kLossConsistencyRelTol,
                      std::max(kLossConsistencyAbsTol, aggregated_channel_energy_tol))) {
        fail("aggregate loss_summary total_loss mismatch against channel energy");
    }
}

void ElectrothermalTelemetryModule::on_finalize() {
    finalize_loss_summary();
    finalize_thermal_summary();
    finalize_component_electrothermal();
    validate_electrothermal_consistency();
}

void ElectrothermalTelemetryModule::on_sample_emit(Real sample_time, std::size_t sample_count) {
    if (!has_loss_trace_ && !has_thermal_trace_) {
        return;
    }
    sample_loss_channels(sample_time, sample_count);
    sample_thermal_channels(sample_count);
}

}  // namespace pulsim::v1
