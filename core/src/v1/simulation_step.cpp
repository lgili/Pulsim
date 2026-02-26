#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace pulsim::v1 {

namespace {
constexpr int kMaxBisections = 12;

[[nodiscard]] bool legacy_fixed_timestep_heuristic(const SimulationOptions& options) {
    const Real span = std::abs(options.dt_max - options.dt_min);
    const Real scale = std::max<Real>({Real{1.0}, std::abs(options.dt), std::abs(options.dt_max)});
    return span <= scale * Real{1e-12};
}

[[nodiscard]] TransientStepMode resolve_step_mode(const SimulationOptions& options) {
    if (options.step_mode_explicit) {
        return options.step_mode;
    }
    if (!options.adaptive_timestep) {
        return TransientStepMode::Fixed;
    }
    return legacy_fixed_timestep_heuristic(options) ? TransientStepMode::Fixed
                                                    : TransientStepMode::Variable;
}

class ScopedStageContext final {
public:
    explicit ScopedStageContext(Circuit& circuit)
        : circuit_(&circuit) {
        circuit_->clear_stage_context();
    }

    ScopedStageContext(const ScopedStageContext&) = delete;
    ScopedStageContext& operator=(const ScopedStageContext&) = delete;

    ~ScopedStageContext() {
        if (circuit_ != nullptr) {
            circuit_->clear_stage_context();
        }
    }

private:
    Circuit* circuit_ = nullptr;
};
}  // namespace

NewtonResult Simulator::solve_step(Real t_next, Real dt, const Vector& x_prev) {
    last_step_segment_cache_hit_ = false;
    last_step_linear_factor_cache_hit_ = false;
    last_step_linear_factor_cache_miss_ = false;

    Integrator method = options_.enable_bdf_order_control
        ? (bdf_controller_.current_order() == 1 ? Integrator::BDF1 : Integrator::Trapezoidal)
        : circuit_.integration_method();

    if (!options_.enable_bdf_order_control) {
        if (method == Integrator::TRBDF2) {
            return solve_trbdf2_step(t_next, dt, x_prev);
        }
        if (method == Integrator::RosenbrockW || method == Integrator::SDIRK2) {
            return solve_sdirk2_step(t_next, dt, x_prev, method);
        }
    }

    circuit_.set_current_time(t_next);
    circuit_.set_timestep(dt);
    if (options_.enable_bdf_order_control) {
        circuit_.set_integration_order(std::clamp(bdf_controller_.current_order(), 1, 2));
    }

    transient_services_.equation_assembler->set_transient_gmin(transient_gmin_);

    auto solve_dae_fallback = [this, &x_prev]() {
        auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
            const auto start = std::chrono::steady_clock::now();
            circuit_.assemble_jacobian(J, f, x);
            if (transient_gmin_ <= 0.0) {
                const auto end = std::chrono::steady_clock::now();
                direct_assemble_system_calls_ += 1;
                direct_assemble_system_time_seconds_ +=
                    std::chrono::duration<double>(end - start).count();
                return;
            }
            for (Index i = 0; i < circuit_.num_nodes(); ++i) {
                J.coeffRef(i, i) += transient_gmin_;
                f[i] += transient_gmin_ * x[i];
            }
            const auto end = std::chrono::steady_clock::now();
            direct_assemble_system_calls_ += 1;
            direct_assemble_system_time_seconds_ +=
                std::chrono::duration<double>(end - start).count();
        };

        auto residual_func = [this](const Vector& x, Vector& f) {
            const auto start = std::chrono::steady_clock::now();
            circuit_.assemble_residual(f, x);
            if (transient_gmin_ <= 0.0) {
                const auto end = std::chrono::steady_clock::now();
                direct_assemble_residual_calls_ += 1;
                direct_assemble_residual_time_seconds_ +=
                    std::chrono::duration<double>(end - start).count();
                return;
            }
            for (Index i = 0; i < circuit_.num_nodes(); ++i) {
                f[i] += transient_gmin_ * x[i];
            }
            const auto end = std::chrono::steady_clock::now();
            direct_assemble_residual_calls_ += 1;
            direct_assemble_residual_time_seconds_ +=
                std::chrono::duration<double>(end - start).count();
        };

        return newton_solver_.solve(x_prev, system_func, residual_func);
    };

    TransientStepRequest request;
    request.mode = resolve_step_mode(options_);
    request.t_now = t_next - dt;
    request.t_target = t_next;
    request.dt_candidate = dt;
    request.dt_min = options_.dt_min;
    request.retry_index = 0;
    request.max_retries = std::max(1, options_.max_step_retries + 1);
    request.event_adjacent = false;

    if (segment_primary_disabled_for_run_) {
        last_step_solve_path_ = StepSolvePath::DaeFallback;
        last_step_solve_reason_ = "segment_disabled_cached_non_admissible";
        return solve_dae_fallback();
    }

    const auto segment_model = transient_services_.segment_model->build_model(x_prev, request);
    last_step_segment_cache_hit_ = segment_model.cache_hit;
    if (!segment_model.admissible &&
        segment_model.classification == "segment_not_admissible_nonlinear_device") {
        segment_primary_disabled_for_run_ = true;
    }
    const auto segment_outcome =
        transient_services_.segment_stepper->try_advance(segment_model, x_prev, request);
    last_step_linear_factor_cache_hit_ = segment_outcome.linear_factor_cache_hit;
    last_step_linear_factor_cache_miss_ = segment_outcome.linear_factor_cache_miss;
    if (!segment_outcome.requires_fallback) {
        last_step_solve_path_ = StepSolvePath::SegmentPrimary;
        last_step_solve_reason_ = segment_outcome.reason;
        return segment_outcome.result;
    }

    last_step_solve_path_ = StepSolvePath::DaeFallback;
    last_step_solve_reason_ = segment_outcome.reason.empty()
        ? "segment_not_admissible"
        : segment_outcome.reason;
    return solve_dae_fallback();
}

NewtonResult Simulator::solve_trbdf2_step(Real t_next, Real dt, const Vector& x_prev) {
    last_step_solve_path_ = StepSolvePath::DaeFallback;
    last_step_solve_reason_ = "trbdf2_multistage";

    const Real gamma = TRBDF2Coeffs::gamma;
    const Real h1 = gamma * dt;
    const Real h2 = dt - h1;

    if (h1 <= 0.0 || h2 <= 0.0) {
        NewtonResult result;
        result.status = SolverStatus::NumericalError;
        result.error_message = "TR-BDF2 invalid timestep split";
        return result;
    }

    ScopedStageContext stage_scope(circuit_);
    circuit_.set_integration_method(Integrator::Trapezoidal);
    circuit_.set_current_time(t_next - h2);
    circuit_.set_timestep(h1);

    transient_services_.equation_assembler->set_transient_gmin(transient_gmin_);
    NewtonResult stage1 = transient_services_.nonlinear_solve->solve(x_prev, t_next - h2, h1);
    if (stage1.status != SolverStatus::Success) {
        circuit_.set_integration_method(Integrator::TRBDF2);
        return stage1;
    }

    circuit_.capture_trbdf2_stage1(stage1.solution);
    circuit_.begin_trbdf2_stage2(h1, h2);
    circuit_.set_integration_method(Integrator::TRBDF2);
    circuit_.set_current_time(t_next);
    circuit_.set_timestep(dt);

    transient_services_.equation_assembler->set_transient_gmin(transient_gmin_);
    NewtonResult stage2 = transient_services_.nonlinear_solve->solve(stage1.solution, t_next, dt);
    return stage2;
}

NewtonResult Simulator::solve_sdirk2_step(Real t_next, Real dt, const Vector& x_prev, Integrator method) {
    last_step_solve_path_ = StepSolvePath::DaeFallback;
    last_step_solve_reason_ = "sdirk2_multistage";

    // RosenbrockW shares SDIRK2 stage coefficients (implicit solve per stage).
    const Real a11 = SDIRK2Coeffs::a11;
    const Real a21 = SDIRK2Coeffs::a21;
    const Real a22 = SDIRK2Coeffs::a22;
    const Real h = dt;
    const Real h1 = a11 * h;

    if (h1 <= 0.0) {
        NewtonResult result;
        result.status = SolverStatus::NumericalError;
        result.error_message = "SDIRK2 invalid timestep split";
        return result;
    }

    ScopedStageContext stage_scope(circuit_);
    circuit_.set_integration_method(Integrator::BDF1);
    circuit_.set_current_time(t_next - (1.0 - SDIRK2Coeffs::c1) * h);
    circuit_.set_timestep(h1);

    transient_services_.equation_assembler->set_transient_gmin(transient_gmin_);
    NewtonResult stage1 = transient_services_.nonlinear_solve->solve(
        x_prev,
        t_next - (1.0 - SDIRK2Coeffs::c1) * h,
        h1);
    if (stage1.status != SolverStatus::Success) {
        circuit_.set_integration_method(method);
        return stage1;
    }

    circuit_.capture_sdirk_stage1(stage1.solution, h, a11);
    circuit_.begin_sdirk_stage2(method, h, a11, a21, a22);
    circuit_.set_integration_method(method);
    circuit_.set_current_time(t_next);
    circuit_.set_timestep(dt);

    transient_services_.equation_assembler->set_transient_gmin(transient_gmin_);
    NewtonResult stage2 = transient_services_.nonlinear_solve->solve(stage1.solution, t_next, dt);
    return stage2;
}

bool Simulator::find_switch_event_time(const SwitchMonitor& sw,
                                       Real t_start, Real t_end,
                                       const Vector& x_start,
                                       Real& t_event, Vector& x_event) {
    if (t_end <= t_start) return false;

    Real t_lo = t_start;
    Real t_hi = t_end;
    Vector x_lo = x_start;
    Vector x_hi;

    // Initial solve at t_hi
    auto result_hi = solve_step(t_hi, t_hi - t_lo, x_lo);
    circuit_.clear_stage_context();
    if (result_hi.status != SolverStatus::Success) {
        return false;
    }
    x_hi = result_hi.solution;

    auto ctrl_value = [&](const Vector& x) -> Real {
        return (sw.ctrl >= 0) ? x[sw.ctrl] : 0.0;
    };

    Real v_lo = ctrl_value(x_lo) - sw.v_threshold;

    if (v_lo == 0.0) {
        t_event = t_lo;
        x_event = x_lo;
        return true;
    }

    for (int i = 0; i < kMaxBisections && (t_hi - t_lo) > options_.dt_min; ++i) {
        Real t_mid = 0.5 * (t_lo + t_hi);
        auto result_mid = solve_step(t_mid, t_mid - t_lo, x_lo);
        circuit_.clear_stage_context();
        if (result_mid.status != SolverStatus::Success) {
            t_hi = t_mid;
            continue;
        }

        Vector x_mid = result_mid.solution;
        Real v_mid = ctrl_value(x_mid) - sw.v_threshold;

        if ((v_lo > 0 && v_mid > 0) || (v_lo < 0 && v_mid < 0)) {
            t_lo = t_mid;
            x_lo = x_mid;
            v_lo = v_mid;
        } else {
            t_hi = t_mid;
            x_hi = x_mid;
        }
    }

    t_event = t_hi;
    x_event = x_hi;
    return true;
}

void Simulator::record_switch_event(const SwitchMonitor& sw, Real time,
                                    const Vector& x_state, bool new_state,
                                    SimulationResult& result, EventCallback event_callback) {
    // Compute switch voltage and current
    Real v_switch = 0.0;
    if (sw.t1 >= 0) v_switch += x_state[sw.t1];
    if (sw.t2 >= 0) v_switch -= x_state[sw.t2];

    // Use discrete g_on/g_off based on state for loss estimates
    Real g_on = 1e3;
    Real g_off = 1e-9;

    const auto& devices = circuit_.devices();
    auto idx_it = device_index_.find(sw.name);
    if (idx_it != device_index_.end()) {
        const auto* dev = std::get_if<VoltageControlledSwitch>(&devices[idx_it->second]);
        if (dev) {
            g_on = dev->g_on();
            g_off = dev->g_off();
        }
    }

    Real g = new_state ? g_on : g_off;
    Real i_switch = g * v_switch;

    SimulationEvent evt;
    evt.time = time;
    evt.type = new_state ? SimulationEventType::SwitchOn : SimulationEventType::SwitchOff;
    evt.component = sw.name;
    evt.description = sw.name + (new_state ? " on" : " off");
    evt.value1 = v_switch;
    evt.value2 = i_switch;
    result.events.push_back(evt);

    if (event_callback) {
        SwitchEvent se;
        se.switch_name = sw.name;
        se.time = time;
        se.new_state = new_state;
        se.voltage = v_switch;
        se.current = i_switch;
        event_callback(se);
    }

    // Accumulate switching energy if configured
    const auto it = options_.switching_energy.find(sw.name);
    if (it != options_.switching_energy.end()) {
        const Real e = new_state ? it->second.eon : it->second.eoff;
        accumulate_switching_loss(sw.name, new_state, e);
    }
}

void Simulator::accumulate_switching_loss(const std::string& name, bool turning_on, Real energy) {
    if (!transient_services_.loss_service) {
        return;
    }
    transient_services_.loss_service->commit_switching_event(name, turning_on, energy);
}

void Simulator::accumulate_reverse_recovery_loss(const std::string& name, Real energy) {
    if (!transient_services_.loss_service) {
        return;
    }
    transient_services_.loss_service->commit_reverse_recovery_event(name, energy);
}

void Simulator::accumulate_conduction_losses(const Vector& x, Real dt) {
    if (!transient_services_.loss_service || !transient_services_.thermal_service) {
        return;
    }
    transient_services_.loss_service->commit_accepted_segment(
        x,
        dt,
        transient_services_.thermal_service->thermal_scale_vector());
}

void Simulator::update_thermal_state(Real dt) {
    if (!transient_services_.thermal_service || !transient_services_.loss_service) {
        return;
    }
    transient_services_.thermal_service->commit_accepted_segment(
        dt,
        transient_services_.loss_service->last_device_power());
}

void Simulator::finalize_loss_summary(SimulationResult& result) {
    if (!transient_services_.loss_service) {
        return;
    }

    Real duration = 0.0;
    if (result.time.size() >= 2) {
        duration = result.time.back() - result.time.front();
    }
    result.loss_summary = transient_services_.loss_service->finalize(duration);
}

void Simulator::finalize_thermal_summary(SimulationResult& result) {
    if (!transient_services_.thermal_service) {
        return;
    }

    const ThermalServiceSummary service_summary = transient_services_.thermal_service->finalize();
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
    result.thermal_summary = std::move(summary);
}


}  // namespace pulsim::v1
