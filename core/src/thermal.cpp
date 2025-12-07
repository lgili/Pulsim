#include "pulsim/thermal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulsim {

void ThermalSimulator::add_model(const ThermalModel& model) {
    models_.push_back(model);
}

void ThermalSimulator::initialize() {
    states_.clear();
    warnings_.clear();
    current_time_ = 0.0;

    for (const auto& model : models_) {
        ThermalState state;
        state.device_name = model.device_name;
        state.tj = t_ambient_;
        state.tc = t_ambient_;
        state.ts = t_ambient_;
        state.tj_peak = t_ambient_;
        state.power_in = 0.0;

        // Initialize Foster network state
        if (model.type == ThermalNetworkType::Foster) {
            state.foster_temps.resize(model.foster.stages.size(), 0.0);
        }

        states_.push_back(state);
    }
}

Real ThermalSimulator::step(Real dt, const std::unordered_map<std::string, Real>& device_powers) {
    current_time_ += dt;
    Real max_tj = t_ambient_;

    for (auto& state : states_) {
        const ThermalModel* model = find_model(state.device_name);
        if (!model) continue;

        // Get power dissipation for this device
        auto it = device_powers.find(state.device_name);
        Real power = (it != device_powers.end()) ? it->second : 0.0;
        state.power_in = power;

        if (model->type == ThermalNetworkType::Foster) {
            // Foster network: each stage is an independent first-order system
            // dT_i/dt = (P * Rth_i - T_i) / tau_i
            Real tj_rise = 0.0;
            for (size_t i = 0; i < model->foster.stages.size(); ++i) {
                const auto& stage = model->foster.stages[i];
                Real target = power * stage.rth;

                if (stage.cth > 0 && stage.tau() > 0) {
                    // First-order exponential response
                    Real alpha = dt / stage.tau();
                    if (alpha > 10.0) {
                        // Effectively instant
                        state.foster_temps[i] = target;
                    } else {
                        // Exponential approach
                        state.foster_temps[i] += (target - state.foster_temps[i]) * (1.0 - std::exp(-alpha));
                    }
                } else {
                    // No thermal capacitance - instant response
                    state.foster_temps[i] = target;
                }
                tj_rise += state.foster_temps[i];
            }

            // Add case-to-sink and sink-to-ambient (assumed quasi-static for now)
            Real tc_rise = power * model->rth_cs;
            Real ts_rise = power * model->rth_sa;

            state.ts = t_ambient_ + ts_rise;
            state.tc = state.ts + tc_rise;
            state.tj = state.tc + tj_rise;

        } else if (model->type == ThermalNetworkType::Cauer) {
            // Cauer network: ladder topology
            // More complex - each stage depends on previous
            // For simplicity, use quasi-static approximation for now
            Real tj_rise = power * model->cauer.rth_total();
            Real tc_rise = power * model->rth_cs;
            Real ts_rise = power * model->rth_sa;

            state.ts = t_ambient_ + ts_rise;
            state.tc = state.ts + tc_rise;
            state.tj = state.tc + tj_rise;

        } else {
            // Simple model: single thermal resistance
            Real tj_rise = power * model->rth_ja();
            state.tj = t_ambient_ + tj_rise;
            state.tc = t_ambient_ + power * (model->rth_cs + model->rth_sa);
            state.ts = t_ambient_ + power * model->rth_sa;
        }

        // Update peak tracking
        if (state.tj > state.tj_peak) {
            state.tj_peak = state.tj;
            state.tj_peak_time = current_time_;
        }

        // Check thermal limits
        if (state.tj > model->tj_warn && !state.exceeded_warning) {
            state.exceeded_warning = true;
            warnings_.push_back({
                state.device_name,
                state.tj,
                current_time_,
                false
            });
        }

        if (state.tj > model->tj_max && !state.exceeded_max) {
            state.exceeded_max = true;
            warnings_.push_back({
                state.device_name,
                state.tj,
                current_time_,
                true
            });
        }

        max_tj = std::max(max_tj, state.tj);
    }

    return max_tj;
}

Real ThermalSimulator::junction_temp(const std::string& device_name) const {
    for (const auto& state : states_) {
        if (state.device_name == device_name) {
            return state.tj;
        }
    }
    return t_ambient_;
}

ThermalState* ThermalSimulator::find_state(const std::string& device_name) {
    for (auto& state : states_) {
        if (state.device_name == device_name) {
            return &state;
        }
    }
    return nullptr;
}

const ThermalModel* ThermalSimulator::find_model(const std::string& device_name) const {
    for (const auto& model : models_) {
        if (model.device_name == device_name) {
            return &model;
        }
    }
    return nullptr;
}

Real ThermalSimulator::adjust_rds_on(Real rds_on_25c, Real tj, Real tc) const {
    // Rds_on(T) = Rds_on(25C) * (1 + tc * (T - 25))
    return rds_on_25c * (1.0 + tc * (tj - 25.0));
}

Real ThermalSimulator::adjust_vth(Real vth_25c, Real tj, Real tc) const {
    // Vth(T) = Vth(25C) + tc * (T - 25)
    return vth_25c + tc * (tj - 25.0);
}

FosterNetwork fit_foster_network(
    const std::vector<std::pair<Real, Real>>& zth_curve,
    int num_stages) {

    FosterNetwork network;

    if (zth_curve.empty()) {
        return network;
    }

    // Simple fitting approach:
    // Use logarithmically spaced time constants
    Real t_min = zth_curve.front().first;
    Real t_max = zth_curve.back().first;
    Real rth_total = zth_curve.back().second;

    // Generate time constants
    std::vector<Real> taus(num_stages);
    Real log_range = std::log10(t_max / t_min);
    for (int i = 0; i < num_stages; ++i) {
        Real frac = static_cast<Real>(i) / (num_stages - 1);
        taus[i] = t_min * std::pow(10.0, frac * log_range);
    }

    // Simple equal distribution of Rth
    Real rth_per_stage = rth_total / num_stages;

    for (int i = 0; i < num_stages; ++i) {
        ThermalRCStage stage;
        stage.rth = rth_per_stage;
        stage.cth = taus[i] / stage.rth;  // tau = R * C
        network.stages.push_back(stage);
    }

    return network;
}

ThermalModel create_mosfet_thermal(
    const std::string& name,
    Real rth_jc,
    Real rth_cs,
    Real rth_sa) {

    ThermalModel model;
    model.device_name = name;
    model.type = ThermalNetworkType::Foster;
    model.rth_jc = rth_jc;
    model.rth_cs = rth_cs;
    model.rth_sa = rth_sa;

    // Create typical 4-stage Foster network
    // Time constants from 1ms to 10s (typical for TO-220 package)
    Real rth_per_stage = rth_jc / 4.0;
    std::vector<Real> taus = {0.001, 0.01, 0.1, 1.0};

    for (int i = 0; i < 4; ++i) {
        ThermalRCStage stage;
        stage.rth = rth_per_stage;
        stage.cth = taus[i] / stage.rth;
        model.foster.stages.push_back(stage);
    }

    return model;
}

}  // namespace pulsim
