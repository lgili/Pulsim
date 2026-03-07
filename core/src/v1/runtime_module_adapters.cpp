/**
 * @file runtime_module_adapters.cpp
 * @brief Internal runtime module adapters for loss/thermal channel emission.
 */

#include "runtime_module_adapters.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace pulsim::v1 {

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

void ElectrothermalTelemetryModule::on_step_accepted(Real dt_segment) {
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

void ElectrothermalTelemetryModule::on_sample_emit(Real sample_time, std::size_t sample_count) {
    if (!has_loss_trace_ && !has_thermal_trace_) {
        return;
    }
    sample_loss_channels(sample_time, sample_count);
    sample_thermal_channels(sample_count);
}

}  // namespace pulsim::v1
