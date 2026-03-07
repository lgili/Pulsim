/**
 * @file simulation_step.cpp
 * @brief Step-level solve paths and per-step post-processing for transient simulation.
 */

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <unordered_map>

namespace pulsim::v1 {

namespace {
constexpr int kMaxBisections = 12;
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

/// Legacy heuristic used when only dt_min/dt_max are provided.
[[nodiscard]] bool legacy_fixed_timestep_heuristic(const SimulationOptions& options) {
    const Real span = std::abs(options.dt_max - options.dt_min);
    const Real scale = std::max<Real>({Real{1.0}, std::abs(options.dt), std::abs(options.dt_max)});
    return span <= scale * Real{1e-12};
}

/// Resolves canonical fixed/variable stepping mode from simulation options.
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

/// RAII guard that always clears stage context across multi-stage integrator paths.
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

/**
 * @brief Solves one accepted transient step using primary segment path with DAE fallback.
 * @param t_next Target time for this step.
 * @param dt Candidate timestep.
 * @param x_prev Previous accepted state.
 * @return Newton solve result for the chosen path.
 */
NewtonResult Simulator::solve_step(Real t_next, Real dt, const Vector& x_prev) {
    last_step_segment_cache_hit_ = false;
    last_step_segment_attempted_ = false;
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
    last_step_linear_factor_cache_invalidation_reason_.clear();

    auto solve_segment_primary = [this, &x_prev, &request, &solve_dae_fallback](bool direct_fallback_attempt) {
        if (segment_primary_disabled_for_run_) {
            last_step_solve_path_ = StepSolvePath::DaeFallback;
            last_step_solve_reason_ = "segment_disabled_cached_non_admissible";
            return solve_dae_fallback();
        }

        last_step_segment_attempted_ = true;
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
        last_step_linear_factor_cache_invalidation_reason_ = segment_outcome.cache_invalidation_reason;
        if (!segment_outcome.requires_fallback) {
            last_step_solve_path_ = StepSolvePath::SegmentPrimary;
            if (direct_fallback_attempt) {
                last_step_solve_reason_ = "direct_failure_projected_fallback";
            } else {
                last_step_solve_reason_ = segment_outcome.reason;
            }
            return segment_outcome.result;
        }

        last_step_solve_path_ = StepSolvePath::DaeFallback;
        last_step_solve_reason_ = segment_outcome.reason.empty()
            ? "segment_not_admissible"
            : segment_outcome.reason;
        return solve_dae_fallback();
    };

    if (options_.formulation_mode == FormulationMode::Direct) {
        last_step_solve_path_ = StepSolvePath::DaeFallback;
        last_step_solve_reason_ = "direct_dae_formulation";
        NewtonResult direct_result = solve_dae_fallback();
        if (direct_result.status == SolverStatus::Success || !options_.direct_formulation_fallback) {
            return direct_result;
        }

        return solve_segment_primary(true);
    }

    return solve_segment_primary(false);
}

/**
 * @brief Solves one TR-BDF2 step using two implicit stages.
 * @param t_next Target time for this step.
 * @param dt Candidate timestep.
 * @param x_prev Previous accepted state.
 * @return Stage-2 Newton result (or stage-1 failure).
 */
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

/**
 * @brief Solves one SDIRK2/Rosenbrock-W step using two implicit stages.
 * @param t_next Target time for this step.
 * @param dt Candidate timestep.
 * @param x_prev Previous accepted state.
 * @param method Active SDIRK2-family integrator.
 * @return Stage-2 Newton result (or stage-1 failure).
 */
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

/**
 * @brief Locates switching threshold crossing time via bisection and sub-solves.
 * @param sw Switch monitor descriptor.
 * @param t_start Step start time.
 * @param t_end Step end time.
 * @param x_start State at @p t_start.
 * @param t_event Output event time.
 * @param x_event Output state at event time.
 * @return `true` when an event time is found; `false` on solver failure.
 */
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

/**
 * @brief Records a switch transition event and optional switching-energy contribution.
 * @param sw Monitored switch descriptor.
 * @param time Event timestamp.
 * @param x_state State used to estimate event voltage/current.
 * @param new_state New logical state (`true`=on, `false`=off).
 * @param result Simulation result accumulator.
 * @param event_callback Optional streaming callback.
 */
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
        const std::size_t device_index = idx_it->second;
        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                g_on = dev.g_on();
                g_off = dev.g_off();
            } else if constexpr (std::is_same_v<T, MOSFET>) {
                const auto params = dev.params();
                g_on = std::max<Real>(params.kp, Real{1e-6});
                g_off = std::max<Real>(params.g_off, Real{1e-18});
            } else if constexpr (std::is_same_v<T, IGBT>) {
                const auto params = dev.params();
                g_on = std::max<Real>(params.g_on, Real{1e-6});
                g_off = std::max<Real>(params.g_off, Real{1e-18});
            }
        }, devices[device_index]);
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

    // Accumulate switching energy from scalar or datasheet surface model.
    const Real e = resolve_switching_event_energy(sw.name, new_state, v_switch, i_switch);
    if (e > 0.0) {
        accumulate_switching_loss(sw.name, new_state, e);
    }
}

/// Resolves event switching energy from scalar or datasheet loss models.
[[nodiscard]] Real Simulator::resolve_switching_event_energy(const std::string& name,
                                                             bool turning_on,
                                                             Real voltage,
                                                             Real current) const {
    const Real i_abs = std::abs(current);
    const Real v_abs = std::abs(voltage);

    if (const auto it = options_.switching_energy_surfaces.find(name);
        it != options_.switching_energy_surfaces.end()) {
        Real temperature = options_.thermal.ambient;
        const auto index_it = device_index_.find(name);
        if (index_it != device_index_.end() && transient_services_.thermal_service) {
            temperature = transient_services_.thermal_service->device_temperature(index_it->second);
        }

        const Real e = turning_on
            ? it->second.evaluate_eon(i_abs, v_abs, temperature)
            : it->second.evaluate_eoff(i_abs, v_abs, temperature);
        return std::max<Real>(0.0, e);
    }

    if (const auto it = options_.switching_energy.find(name);
        it != options_.switching_energy.end()) {
        const Real e = turning_on ? it->second.eon : it->second.eoff;
        return std::max<Real>(0.0, e);
    }

    return 0.0;
}

/// Accumulates discrete turn-on/off switching energy into the loss service.
void Simulator::accumulate_switching_loss(const std::string& name, bool turning_on, Real energy) {
    if (!transient_services_.loss_service) {
        return;
    }
    transient_services_.loss_service->commit_switching_event(name, turning_on, energy);
}

/// Accumulates reverse-recovery energy into the loss service.
void Simulator::accumulate_reverse_recovery_loss(const std::string& name, Real energy) {
    if (!transient_services_.loss_service) {
        return;
    }
    transient_services_.loss_service->commit_reverse_recovery_event(name, energy);
}

/// Finalizes aggregate loss summary after transient completion.
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

/// Finalizes thermal summary after transient completion.
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

/// Joins loss and thermal summaries into per-component electrothermal telemetry.
void Simulator::finalize_component_electrothermal(SimulationResult& result) {
    const auto& conns = circuit_.connections();
    result.component_electrothermal.clear();
    result.component_electrothermal.reserve(conns.size());

    std::unordered_map<std::string, const LossResult*> loss_by_name;
    loss_by_name.reserve(result.loss_summary.device_losses.size());
    for (const auto& item : result.loss_summary.device_losses) {
        loss_by_name[item.device_name] = &item;
    }

    std::unordered_map<std::string, const DeviceThermalTelemetry*> thermal_by_name;
    thermal_by_name.reserve(result.thermal_summary.device_temperatures.size());
    for (const auto& item : result.thermal_summary.device_temperatures) {
        thermal_by_name[item.device_name] = &item;
    }

    const Real ambient = result.thermal_summary.ambient;
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

        result.component_electrothermal.push_back(std::move(entry));
    }
}

/// Verifies deterministic consistency between canonical channels and summary telemetry.
void Simulator::validate_electrothermal_consistency(SimulationResult& result) {
    if (!result.success) {
        return;
    }
    if (result.time.empty()) {
        return;
    }

    auto fail = [&](const std::string& detail) {
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.diagnostic = SimulationDiagnosticCode::TransientStepFailure;
        result.message = "Electrothermal consistency failure: " + detail;
        result.backend_telemetry.failure_reason = "electrothermal_consistency_failure";
    };

    auto find_channel = [&](const std::string& channel_name) -> const std::vector<Real>* {
        const auto it = result.virtual_channels.find(channel_name);
        if (it == result.virtual_channels.end()) {
            return nullptr;
        }
        if (it->second.size() != result.time.size()) {
            fail("channel '" + channel_name + "' length mismatch");
            return nullptr;
        }
        return &it->second;
    };

    auto integrate_channel_energy = [&](const std::string& channel_name,
                                        const std::vector<Real>& series) -> std::optional<Real> {
        Real energy = 0.0;
        for (std::size_t i = 1; i < result.time.size(); ++i) {
            const Real dt = result.time[i] - result.time[i - 1];
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
    thermal_summary_by_name.reserve(result.thermal_summary.device_temperatures.size());
    for (const auto& row : result.thermal_summary.device_temperatures) {
        thermal_summary_by_name[row.device_name] = &row;
    }

    std::unordered_map<std::string, const LossResult*> loss_summary_by_name;
    loss_summary_by_name.reserve(result.loss_summary.device_losses.size());
    for (const auto& row : result.loss_summary.device_losses) {
        loss_summary_by_name[row.device_name] = &row;
    }

    std::unordered_map<std::string, const ComponentElectrothermalTelemetry*> component_by_name;
    component_by_name.reserve(result.component_electrothermal.size());
    for (const auto& row : result.component_electrothermal) {
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

    if (!options_.enable_losses) {
        return;
    }

    const Real duration =
        result.time.size() >= 2 ? (result.time.back() - result.time.front()) : Real{0.0};
    Real max_dt = 0.0;
    for (std::size_t i = 1; i < result.time.size(); ++i) {
        const Real dt = result.time[i] - result.time[i - 1];
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
        !nearly_equal(result.loss_summary.total_loss * duration,
                      aggregated_channel_energy,
                      kLossConsistencyRelTol,
                      std::max(kLossConsistencyAbsTol, aggregated_channel_energy_tol))) {
        fail("aggregate loss_summary total_loss mismatch against channel energy");
    }
}


}  // namespace pulsim::v1
