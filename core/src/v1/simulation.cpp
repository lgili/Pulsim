#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <span>

namespace pulsim::v1 {

namespace {
constexpr int kMaxBisections = 12;
constexpr int kMaxGlobalRecoveryAttempts = 2;

struct CircuitRobustnessHints {
    int switching_devices = 0;
    int nonlinear_devices = 0;
    int pwm_blocks = 0;
    int control_blocks = 0;
};

[[nodiscard]] bool is_default_linear_order(const LinearSolverStackConfig& cfg) {
    return cfg.order.empty() ||
           (cfg.order.size() == 1 && cfg.order.front() == LinearSolverKind::SparseLU);
}

void apply_robust_linear_solver_defaults(LinearSolverStackConfig& cfg, bool force = true) {
    const bool has_default_order = is_default_linear_order(cfg);
    if (!force && !has_default_order) {
        return;
    }

    if (has_default_order) {
        cfg.order = {
            LinearSolverKind::KLU,
            LinearSolverKind::EnhancedSparseLU,
            LinearSolverKind::GMRES,
            LinearSolverKind::BiCGSTAB
        };
    }
    if (cfg.fallback_order.empty()) {
        cfg.fallback_order = {
            LinearSolverKind::EnhancedSparseLU,
            LinearSolverKind::SparseLU,
            LinearSolverKind::GMRES,
            LinearSolverKind::BiCGSTAB
        };
    }
    cfg.allow_fallback = true;
    cfg.auto_select = true;
    cfg.size_threshold = std::min(cfg.size_threshold, 1200);
    cfg.nnz_threshold = std::min(cfg.nnz_threshold, 120000);
    cfg.diag_min_threshold = std::max(cfg.diag_min_threshold, Real{1e-12});

    auto& it = cfg.iterative_config;
    it.max_iterations = std::max(it.max_iterations, 300);
    it.tolerance = std::min(it.tolerance, Real{1e-8});
    it.restart = std::max(it.restart, 40);
    it.enable_scaling = true;
    it.scaling_floor = std::min(it.scaling_floor, Real{1e-12});
    if (it.preconditioner == IterativeSolverConfig::PreconditionerKind::None ||
        it.preconditioner == IterativeSolverConfig::PreconditionerKind::Jacobi) {
        it.preconditioner = IterativeSolverConfig::PreconditionerKind::ILUT;
    }
    it.ilut_drop_tolerance = std::min(it.ilut_drop_tolerance, Real{1e-3});
    it.ilut_fill_factor = std::max(it.ilut_fill_factor, Real{10.0});
}

[[nodiscard]] bool is_default_newton_profile(const NewtonOptions& opts) {
    NewtonOptions defaults;
    return opts.max_iterations <= defaults.max_iterations &&
           opts.enable_limiting == defaults.enable_limiting &&
           opts.enable_trust_region == defaults.enable_trust_region &&
           opts.max_voltage_step <= defaults.max_voltage_step &&
           opts.max_current_step <= defaults.max_current_step &&
           opts.min_damping >= defaults.min_damping;
}

void apply_robust_newton_defaults(NewtonOptions& opts, bool force = true) {
    if (!force && !is_default_newton_profile(opts)) {
        return;
    }

    opts.max_iterations = std::max(opts.max_iterations, 120);
    opts.auto_damping = true;
    opts.min_damping = std::min(opts.min_damping, Real{1e-4});
    opts.enable_limiting = true;
    opts.max_voltage_step = std::max(opts.max_voltage_step, Real{10.0});
    opts.max_current_step = std::max(opts.max_current_step, Real{20.0});
    opts.enable_trust_region = true;
    opts.trust_radius = std::max(opts.trust_radius, Real{8.0});
    opts.trust_shrink = std::min(opts.trust_shrink, Real{0.5});
    opts.trust_expand = std::max(opts.trust_expand, Real{1.5});
    opts.detect_stall = false;
}

[[nodiscard]] CircuitRobustnessHints analyze_circuit_robustness(const Circuit& circuit) {
    CircuitRobustnessHints hints;
    const auto& devices = circuit.devices();
    for (const auto& device : devices) {
        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            if constexpr (std::is_same_v<T, VoltageControlledSwitch> ||
                          std::is_same_v<T, IdealSwitch> ||
                          std::is_same_v<T, MOSFET> ||
                          std::is_same_v<T, IGBT>) {
                hints.switching_devices++;
                hints.nonlinear_devices++;
            } else if constexpr (std::is_same_v<T, IdealDiode>) {
                hints.nonlinear_devices++;
            }
        }, device);
    }

    const auto& virtual_components = circuit.virtual_components();
    for (const auto& component : virtual_components) {
        if (component.type == "pwm_generator") {
            hints.pwm_blocks++;
            hints.control_blocks++;
        } else if (component.type == "pi_controller" ||
                   component.type == "pid_controller" ||
                   component.type == "hysteresis" ||
                   component.type == "comparator" ||
                   component.type == "state_machine" ||
                   component.type == "relay") {
            hints.control_blocks++;
        }
    }
    return hints;
}

[[nodiscard]] bool is_fixed_timestep(const SimulationOptions& options) {
    const Real span = std::abs(options.dt_max - options.dt_min);
    const Real scale = std::max<Real>({Real{1.0}, std::abs(options.dt), std::abs(options.dt_max)});
    return span <= scale * Real{1e-12};
}

void apply_auto_transient_profile(SimulationOptions& options, const Circuit& circuit) {
    // Respect explicit strict mode: users can disable fallback for deterministic debugging.
    if (!options.linear_solver.allow_fallback) {
        return;
    }

    const auto hints = analyze_circuit_robustness(circuit);
    const bool switching_topology =
        hints.switching_devices > 0 || hints.pwm_blocks > 0 || hints.control_blocks > 0;
    if (!switching_topology) {
        return;
    }

    const bool fixed_step = is_fixed_timestep(options);
    if (!fixed_step) {
        options.adaptive_timestep = true;
        options.timestep_config = AdvancedTimestepConfig::for_power_electronics();
        options.timestep_config.dt_initial = options.dt;
        options.timestep_config.dt_min = std::max(options.dt_min, Real{1e-12});
        options.timestep_config.dt_max = std::max(options.timestep_config.dt_max, options.dt * 20.0);
        options.timestep_config.error_tolerance =
            std::max(options.timestep_config.error_tolerance, Real{5e-3});

        options.enable_bdf_order_control = true;
        options.bdf_config.min_order = 1;
        options.bdf_config.max_order = 2;
        options.bdf_config.initial_order = 1;
    }

    if (options.integrator == Integrator::Trapezoidal) {
        options.integrator = Integrator::TRBDF2;
    }

    options.max_step_retries = std::max(options.max_step_retries, 12);
    options.stiffness_config.enable = true;
    options.stiffness_config.switch_integrator = true;
    options.stiffness_config.stiff_integrator = Integrator::BDF1;
    options.stiffness_config.rejection_streak_threshold =
        std::min(options.stiffness_config.rejection_streak_threshold, 2);
    options.stiffness_config.newton_iter_threshold =
        std::min(options.stiffness_config.newton_iter_threshold, 30);
    options.stiffness_config.newton_streak_threshold =
        std::min(options.stiffness_config.newton_streak_threshold, 2);
    options.stiffness_config.cooldown_steps = std::max(options.stiffness_config.cooldown_steps, 3);

    if (options.fallback_policy.enable_transient_gmin) {
        options.fallback_policy.gmin_retry_threshold =
            std::min(options.fallback_policy.gmin_retry_threshold, 1);
        options.fallback_policy.gmin_initial =
            std::max(options.fallback_policy.gmin_initial, Real{1e-8});
        options.fallback_policy.gmin_max = std::max(options.fallback_policy.gmin_max, Real{1e-3});
        options.fallback_policy.gmin_growth = std::max(options.fallback_policy.gmin_growth, Real{10.0});
    }

    apply_robust_newton_defaults(options.newton_options, false);
    apply_robust_linear_solver_defaults(options.linear_solver, false);
}

Matrix spectral_diff_matrix(int samples, Real period) {
    Matrix D = Matrix::Zero(samples, samples);
    if (samples <= 1 || period <= 0.0) {
        return D;
    }
    const Real pi = Real{3.14159265358979323846};
    const Real scale = (2.0 * pi) / period;
    const bool even = (samples % 2) == 0;

    for (int j = 0; j < samples; ++j) {
        for (int k = 0; k < samples; ++k) {
            if (j == k) continue;
            const int diff = j - k;
            const Real sign = (diff % 2 == 0) ? 1.0 : -1.0;
            const Real angle = pi * static_cast<Real>(diff) / static_cast<Real>(samples);
            if (even) {
                D(j, k) = 0.5 * sign / std::tan(angle);
            } else {
                D(j, k) = 0.5 * sign / std::sin(angle);
            }
        }
    }
    return D * scale;
}
}

Simulator::Simulator(Circuit& circuit, const SimulationOptions& options)
    : circuit_(circuit)
    , options_(options)
    , newton_solver_(options_.newton_options)
    , timestep_controller_(options_.timestep_config)
    , lte_estimator_(options_.lte_config)
    , bdf_controller_(options_.bdf_config) {

    options_.newton_options.num_nodes = circuit_.num_nodes();
    options_.newton_options.num_branches = circuit_.num_branches();
    apply_auto_transient_profile(options_, circuit_);
    options_.newton_options.num_nodes = circuit_.num_nodes();
    options_.newton_options.num_branches = circuit_.num_branches();
    newton_solver_.set_options(options_.newton_options);
    newton_solver_.linear_solver().set_config(options_.linear_solver);

    // Ensure timestep controller limits align with simulation options
    auto cfg = options_.timestep_config;
    cfg.dt_min = options_.dt_min;
    cfg.dt_max = options_.dt_max;
    if (cfg.dt_initial <= 0) {
        cfg.dt_initial = options_.dt;
    }
    timestep_controller_ = AdvancedTimestepController(cfg);

    // Build device index map and switch monitors
    const auto& devices = circuit_.devices();
    const auto& conns = circuit_.connections();
    device_index_.clear();
    switch_monitors_.clear();

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& conn = conns[i];
        device_index_[conn.name] = i;

        if (const auto* sw = std::get_if<VoltageControlledSwitch>(&devices[i])) {
            if (conn.nodes.size() >= 3) {
                SwitchMonitor monitor;
                monitor.name = conn.name;
                monitor.ctrl = conn.nodes[0];
                monitor.t1 = conn.nodes[1];
                monitor.t2 = conn.nodes[2];
                monitor.v_threshold = sw->v_threshold();
                monitor.was_on = false;
                switch_monitors_.push_back(monitor);
            }
        }
    }

    initialize_loss_tracking();

    for (const auto& [name, energy] : options_.switching_energy) {
        set_switching_energy(name, energy);
    }
}

void Simulator::initialize_loss_tracking() {
    const auto& devices = circuit_.devices();
    loss_states_.assign(devices.size(), DeviceLossState{});
    switching_energy_.assign(devices.size(), std::nullopt);
    diode_conducting_.assign(devices.size(), false);
    last_device_power_.assign(devices.size(), 0.0);
    initialize_thermal_tracking();
}

void Simulator::set_switching_energy(const std::string& device_name, const SwitchingEnergy& energy) {
    auto it = device_index_.find(device_name);
    if (it == device_index_.end()) return;
    switching_energy_[it->second] = energy;
}

void Simulator::record_fallback_event(SimulationResult& result,
                                      int step_index,
                                      int retry_index,
                                      Real time,
                                      Real dt,
                                      FallbackReasonCode reason,
                                      SolverStatus solver_status,
                                      const std::string& action) {
    if (!options_.fallback_policy.trace_retries) {
        return;
    }
    FallbackTraceEntry entry;
    entry.step_index = step_index;
    entry.retry_index = retry_index;
    entry.time = time;
    entry.dt = dt;
    entry.reason = reason;
    entry.solver_status = solver_status;
    entry.action = action;
    result.fallback_trace.push_back(std::move(entry));
}

void Simulator::apply_transient_gmin(SparseMatrix& J, Vector& f, const Vector& x) const {
    if (transient_gmin_ <= 0.0) {
        return;
    }
    for (Index i = 0; i < circuit_.num_nodes(); ++i) {
        J.coeffRef(i, i) += transient_gmin_;
        f[i] += transient_gmin_ * x[i];
    }
}

void Simulator::apply_transient_gmin_residual(Vector& f, const Vector& x) const {
    if (transient_gmin_ <= 0.0) {
        return;
    }
    for (Index i = 0; i < circuit_.num_nodes(); ++i) {
        f[i] += transient_gmin_ * x[i];
    }
}

void Simulator::initialize_thermal_tracking() {
    const auto& devices = circuit_.devices();
    const auto& conns = circuit_.connections();

    thermal_states_.assign(devices.size(), DeviceThermalState{});
    if (!options_.thermal.enable) {
        return;
    }

    for (std::size_t i = 0; i < devices.size(); ++i) {
        bool supports_thermal = false;
        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            supports_thermal = device_traits<T>::has_thermal_model;
        }, devices[i]);

        if (!supports_thermal) {
            continue;
        }

        auto& state = thermal_states_[i];
        state.enabled = true;
        state.config.enabled = true;
        state.config.rth = options_.thermal.default_rth;
        state.config.cth = options_.thermal.default_cth;
        state.config.temp_init = options_.thermal.ambient;
        state.config.temp_ref = options_.thermal.ambient;

        auto cfg_it = options_.thermal_devices.find(conns[i].name);
        if (cfg_it != options_.thermal_devices.end()) {
            state.config = cfg_it->second;
        }

        state.enabled = state.config.enabled;
        state.temperature = state.config.temp_init;
        state.peak_temperature = state.temperature;
        state.sum_temperature = 0.0;
        state.samples = 0;
    }
}

Real Simulator::thermal_scale_factor(std::size_t device_index) const {
    if (!options_.thermal.enable ||
        options_.thermal.policy == ThermalCouplingPolicy::LossOnly ||
        device_index >= thermal_states_.size()) {
        return 1.0;
    }

    const auto& state = thermal_states_[device_index];
    if (!state.enabled) {
        return 1.0;
    }

    Real scale = 1.0 + state.config.alpha * (state.temperature - state.config.temp_ref);
    return std::max<Real>(0.05, scale);
}

DCAnalysisResult Simulator::dc_operating_point() {
    // Large timestep to emulate DC for dynamic elements
    circuit_.set_timestep(1e6);
    circuit_.set_integration_order(1);

    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        circuit_.assemble_jacobian(J, f, x);
    };

    Vector x0 = Vector::Zero(circuit_.system_size());

    DCConvergenceSolver<RuntimeLinearSolver> solver(options_.dc_config);
    solver.set_linear_solver_config(options_.linear_solver);
    return solver.solve(x0, circuit_.num_nodes(), circuit_.num_branches(), system_func, nullptr);
}

NewtonResult Simulator::solve_step(Real t_next, Real dt, const Vector& x_prev) {
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

    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        circuit_.assemble_jacobian(J, f, x);
        apply_transient_gmin(J, f, x);
    };

    auto residual_func = [this](const Vector& x, Vector& f) {
        circuit_.assemble_residual(f, x);
        apply_transient_gmin_residual(f, x);
    };

    return newton_solver_.solve(x_prev, system_func, residual_func);
}

NewtonResult Simulator::solve_trbdf2_step(Real t_next, Real dt, const Vector& x_prev) {
    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        circuit_.assemble_jacobian(J, f, x);
        apply_transient_gmin(J, f, x);
    };
    auto residual_func = [this](const Vector& x, Vector& f) {
        circuit_.assemble_residual(f, x);
        apply_transient_gmin_residual(f, x);
    };

    const Real gamma = TRBDF2Coeffs::gamma;
    const Real h1 = gamma * dt;
    const Real h2 = dt - h1;

    if (h1 <= 0.0 || h2 <= 0.0) {
        NewtonResult result;
        result.status = SolverStatus::NumericalError;
        result.error_message = "TR-BDF2 invalid timestep split";
        return result;
    }

    circuit_.clear_stage_context();
    circuit_.set_integration_method(Integrator::Trapezoidal);
    circuit_.set_current_time(t_next - h2);
    circuit_.set_timestep(h1);

    NewtonResult stage1 = newton_solver_.solve(x_prev, system_func, residual_func);
    if (stage1.status != SolverStatus::Success) {
        circuit_.set_integration_method(Integrator::TRBDF2);
        circuit_.clear_stage_context();
        return stage1;
    }

    circuit_.capture_trbdf2_stage1(stage1.solution);
    circuit_.begin_trbdf2_stage2(h1, h2);
    circuit_.set_integration_method(Integrator::TRBDF2);
    circuit_.set_current_time(t_next);
    circuit_.set_timestep(dt);

    NewtonResult stage2 = newton_solver_.solve(stage1.solution, system_func, residual_func);
    if (stage2.status != SolverStatus::Success) {
        circuit_.clear_stage_context();
    }
    return stage2;
}

NewtonResult Simulator::solve_sdirk2_step(Real t_next, Real dt, const Vector& x_prev, Integrator method) {
    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        circuit_.assemble_jacobian(J, f, x);
        apply_transient_gmin(J, f, x);
    };
    auto residual_func = [this](const Vector& x, Vector& f) {
        circuit_.assemble_residual(f, x);
        apply_transient_gmin_residual(f, x);
    };

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

    circuit_.clear_stage_context();
    circuit_.set_integration_method(Integrator::BDF1);
    circuit_.set_current_time(t_next - (1.0 - SDIRK2Coeffs::c1) * h);
    circuit_.set_timestep(h1);

    NewtonResult stage1 = newton_solver_.solve(x_prev, system_func, residual_func);
    if (stage1.status != SolverStatus::Success) {
        circuit_.set_integration_method(method);
        circuit_.clear_stage_context();
        return stage1;
    }

    circuit_.capture_sdirk_stage1(stage1.solution, h, a11);
    circuit_.begin_sdirk_stage2(method, h, a11, a21, a22);
    circuit_.set_integration_method(method);
    circuit_.set_current_time(t_next);
    circuit_.set_timestep(dt);

    NewtonResult stage2 = newton_solver_.solve(stage1.solution, system_func, residual_func);
    if (stage2.status != SolverStatus::Success) {
        circuit_.clear_stage_context();
    }
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
    auto it = device_index_.find(sw.name);
    if (it != device_index_.end()) {
        const auto& energy_opt = switching_energy_[it->second];
        if (energy_opt) {
            Real e = new_state ? energy_opt->eon : energy_opt->eoff;
            accumulate_switching_loss(sw.name, new_state, e);
        }
    }
}

void Simulator::accumulate_switching_loss(const std::string& name, bool turning_on, Real energy) {
    if (energy <= 0.0) return;
    auto it = device_index_.find(name);
    if (it == device_index_.end()) return;

    auto& state = loss_states_[it->second];
    state.accumulator.add_switching_event(energy);
    if (turning_on) {
        state.switching_energy.turn_on += energy;
    } else {
        state.switching_energy.turn_off += energy;
    }
}

void Simulator::accumulate_reverse_recovery_loss(const std::string& name, Real energy) {
    if (energy <= 0.0) return;
    auto it = device_index_.find(name);
    if (it == device_index_.end()) return;

    auto& state = loss_states_[it->second];
    state.accumulator.add_switching_event(energy);
    state.switching_energy.reverse_recovery += energy;
}

void Simulator::accumulate_conduction_losses(const Vector& x, Real dt) {
    if (!options_.enable_losses) return;

    const auto& devices = circuit_.devices();
    const auto& conns = circuit_.connections();
    if (last_device_power_.size() != devices.size()) {
        last_device_power_.assign(devices.size(), 0.0);
    } else {
        std::fill(last_device_power_.begin(), last_device_power_.end(), 0.0);
    }

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& conn = conns[i];
        Real p_cond = 0.0;

        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;

            auto node_voltage = [&](Index n) -> Real {
                return (n >= 0) ? x[n] : 0.0;
            };

            if constexpr (std::is_same_v<T, Resistor>) {
                Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                p_cond = (v * v) / dev.resistance();
            }
            else if constexpr (std::is_same_v<T, IdealSwitch>) {
                Real g = dev.is_closed() ? dev.g_on() : dev.g_off();
                Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                Real i = g * v;
                p_cond = std::abs(v * i);
            }
            else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                Real v_ctrl = node_voltage(conn.nodes[0]);
                bool on = v_ctrl > dev.v_threshold();
                Real g = on ? dev.g_on() : dev.g_off();
                Real v = node_voltage(conn.nodes[1]) - node_voltage(conn.nodes[2]);
                Real i = g * v;
                p_cond = std::abs(v * i);
            }
            else if constexpr (std::is_same_v<T, IdealDiode>) {
                Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                Real g = dev.is_conducting() ? dev.g_on() : dev.g_off();
                Real i = g * v;
                p_cond = std::max<Real>(0.0, v * i);

                bool conducting = dev.is_conducting();
                if (diode_conducting_[i] && !conducting) {
                    const auto& energy_opt = switching_energy_[i];
                    if (energy_opt && energy_opt->err > 0.0) {
                        accumulate_reverse_recovery_loss(conn.name, energy_opt->err);
                    }
                }
                diode_conducting_[i] = conducting;
            }
            else if constexpr (std::is_same_v<T, MOSFET>) {
                Real vg = node_voltage(conn.nodes[0]);
                Real vd = node_voltage(conn.nodes[1]);
                Real vs = node_voltage(conn.nodes[2]);
                auto params = dev.params();

                Real sign = params.is_nmos ? 1.0 : -1.0;
                Real vgs = sign * (vg - vs);
                Real vds = sign * (vd - vs);

                Real id = 0.0;
                if (vgs <= params.vth) {
                    id = params.g_off * vds;
                } else if (vds < vgs - params.vth) {
                    Real vov = vgs - params.vth;
                    id = params.kp * (vov * vds - 0.5 * vds * vds) * (1.0 + params.lambda * vds);
                } else {
                    Real vov = vgs - params.vth;
                    id = 0.5 * params.kp * vov * vov * (1.0 + params.lambda * vds);
                }

                id *= sign;
                p_cond = std::abs((vd - vs) * id);
            }
            else if constexpr (std::is_same_v<T, IGBT>) {
                Real vg = node_voltage(conn.nodes[0]);
                Real vc = node_voltage(conn.nodes[1]);
                Real ve = node_voltage(conn.nodes[2]);
                auto params = dev.params();

                Real vge = vg - ve;
                Real vce = vc - ve;
                bool on = (vge > params.vth) && (vce > 0);
                Real g = on ? params.g_on : params.g_off;
                Real i = g * vce;
                p_cond = std::abs(vce * i);
            }
        }, devices[i]);

        p_cond *= thermal_scale_factor(i);
        last_device_power_[i] = std::max<Real>(0.0, p_cond);

        if (p_cond > 0.0) {
            auto& state = loss_states_[i];
            state.accumulator.add_sample(p_cond, dt);
            state.peak_power = std::max(state.peak_power, p_cond);
        }
    }
}

void Simulator::update_thermal_state(Real dt) {
    if (!options_.thermal.enable || dt <= 0.0) {
        return;
    }

    if (thermal_states_.size() != last_device_power_.size()) {
        return;
    }

    for (std::size_t i = 0; i < thermal_states_.size(); ++i) {
        auto& state = thermal_states_[i];
        if (!state.enabled) {
            continue;
        }

        const Real power = std::max<Real>(0.0, last_device_power_[i]);
        const Real ambient = options_.thermal.ambient;
        const Real rth = std::max<Real>(state.config.rth, 1e-12);
        const Real cth = state.config.cth;

        if (cth <= 0.0) {
            state.temperature = ambient + power * rth;
        } else {
            const Real tau = std::max<Real>(rth * cth, 1e-12);
            const Real delta = state.temperature - ambient;
            const Real delta_dot = (power * rth - delta) / tau;
            state.temperature = ambient + delta + dt * delta_dot;
        }

        state.peak_temperature = std::max(state.peak_temperature, state.temperature);
        state.sum_temperature += state.temperature;
        state.samples += 1;
    }
}

void Simulator::finalize_loss_summary(SimulationResult& result) {
    if (!options_.enable_losses) return;

    SystemLossSummary summary;
    const auto& conns = circuit_.connections();

    Real duration = 0.0;
    if (result.time.size() >= 2) {
        duration = result.time.back() - result.time.front();
    }

    for (std::size_t i = 0; i < loss_states_.size(); ++i) {
        const auto& state = loss_states_[i];
        if (state.accumulator.num_samples() == 0 &&
            state.accumulator.switching_energy() == 0.0) {
            continue;
        }

        LossResult res;
        res.device_name = conns[i].name;
        res.total_energy = state.accumulator.total_energy();
        res.average_power = duration > 0 ? res.total_energy / duration : 0.0;
        res.peak_power = state.peak_power;

        Real conduction_energy = state.accumulator.conduction_energy();
        Real switching_energy = state.accumulator.switching_energy();

        res.breakdown.conduction = duration > 0 ? conduction_energy / duration : 0.0;
        res.breakdown.turn_on = duration > 0 ? state.switching_energy.turn_on / duration : 0.0;
        res.breakdown.turn_off = duration > 0 ? state.switching_energy.turn_off / duration : 0.0;
        res.breakdown.reverse_recovery = duration > 0 ? state.switching_energy.reverse_recovery / duration : 0.0;

        if (res.breakdown.turn_on == 0.0 &&
            res.breakdown.turn_off == 0.0 &&
            res.breakdown.reverse_recovery == 0.0) {
            res.breakdown.turn_on = duration > 0 ? switching_energy / duration : 0.0;
        }

        summary.device_losses.push_back(res);
    }

    summary.compute_totals();
    result.loss_summary = summary;
}

void Simulator::finalize_thermal_summary(SimulationResult& result) {
    ThermalSummary summary;
    summary.enabled = options_.thermal.enable;
    summary.ambient = options_.thermal.ambient;
    summary.max_temperature = options_.thermal.ambient;

    if (!options_.thermal.enable) {
        result.thermal_summary = summary;
        return;
    }

    const auto& conns = circuit_.connections();
    for (std::size_t i = 0; i < thermal_states_.size() && i < conns.size(); ++i) {
        const auto& state = thermal_states_[i];
        if (!state.enabled) {
            continue;
        }

        DeviceThermalTelemetry telemetry;
        telemetry.device_name = conns[i].name;
        telemetry.enabled = true;
        telemetry.final_temperature = state.temperature;
        telemetry.peak_temperature = state.peak_temperature;
        telemetry.average_temperature =
            (state.samples > 0) ? (state.sum_temperature / static_cast<Real>(state.samples))
                                : state.temperature;

        summary.max_temperature = std::max(summary.max_temperature, telemetry.peak_temperature);
        summary.device_temperatures.push_back(std::move(telemetry));
    }

    result.thermal_summary = std::move(summary);
}

SimulationResult Simulator::run_transient(SimulationCallback callback,
                                          EventCallback event_callback,
                                          SimulationControl* control) {
    auto dc = dc_operating_point();
    if (!dc.success) {
        SimulationResult result;
        result.success = false;
        result.final_status = dc.newton_result.status;
        result.message = "DC operating point failed: " + dc.message;
        result.linear_solver_telemetry = dc.linear_solver_telemetry;
        return result;
    }

    return run_transient(dc.newton_result.solution, callback, event_callback, control);
}

SimulationResult Simulator::run_transient(const Vector& x0,
                                          SimulationCallback callback,
                                          EventCallback event_callback,
                                          SimulationControl* control) {
    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    initialize_loss_tracking();
    for (const auto& [name, energy] : options_.switching_energy) {
        set_switching_energy(name, energy);
    }
    lte_estimator_.reset();
    transient_gmin_ = 0.0;
    const Integrator base_integrator = options_.integrator;
    bool using_stiff_integrator = false;

    if (options_.enable_bdf_order_control) {
        bdf_controller_.set_order(std::clamp(options_.bdf_config.initial_order, 1, 2));
    } else {
        circuit_.set_integration_method(base_integrator);
    }

    Real t = options_.tstart;
    Real dt = options_.dt;
    Vector x = x0;

    int rejection_streak = 0;
    int high_iter_streak = 0;
    int stiffness_cooldown = 0;
    int global_recovery_attempts = 0;
    bool auto_recovery_attempted = false;
    const bool can_auto_recover = options_.adaptive_timestep &&
                                  options_.linear_solver.allow_fallback &&
                                  !is_fixed_timestep(options_);

    circuit_.set_current_time(t);
    circuit_.set_timestep(dt);
    if (options_.enable_bdf_order_control) {
        circuit_.set_integration_order(std::clamp(bdf_controller_.current_order(), 1, 2));
    }

    circuit_.update_history(x, true);

    auto append_virtual_sample = [&](const Vector& state, Real sample_time) {
        if (circuit_.num_virtual_components() == 0) {
            return;
        }

        if (result.virtual_channel_metadata.empty()) {
            result.virtual_channel_metadata = circuit_.virtual_channel_metadata();
        }

        auto mixed_step = circuit_.execute_mixed_domain_step(state, sample_time);
        if (result.mixed_domain_phase_order.empty()) {
            result.mixed_domain_phase_order = mixed_step.phase_order;
        }

        const std::size_t sample_count = result.time.size();
        const Real nan = std::numeric_limits<Real>::quiet_NaN();

        for (auto& [channel, series] : result.virtual_channels) {
            while (series.size() + 1 < sample_count) {
                series.push_back(nan);
            }
        }

        for (const auto& [channel, value] : mixed_step.channel_values) {
            auto& series = result.virtual_channels[channel];
            while (series.size() + 1 < sample_count) {
                series.push_back(nan);
            }
            series.push_back(value);

            if (!result.virtual_channel_metadata.contains(channel)) {
                result.virtual_channel_metadata[channel] = VirtualChannelMetadata{
                    "virtual", channel, "control", {}
                };
            }
        }

        for (auto& [channel, series] : result.virtual_channels) {
            if (series.size() < sample_count) {
                series.push_back(nan);
            }
        }
    };

    result.time.push_back(t);
    result.states.push_back(x);
    append_virtual_sample(x, t);

    if (callback) {
        callback(t, x);
    }

    for (auto& sw : switch_monitors_) {
        Real v_ctrl = (sw.ctrl >= 0) ? x[sw.ctrl] : 0.0;
        sw.was_on = v_ctrl > sw.v_threshold;
    }

    while (t < options_.tstop) {
        if (control) {
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                break;
            }
            while (control->should_pause() && !control->should_stop()) {
                control->wait_until_resumed();
            }
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                break;
            }
        }

        if (t + dt > options_.tstop) {
            dt = options_.tstop - t;
            if (dt < options_.dt_min * 0.1) {
                break;
            }
        }

        bool accepted = false;
        int retries = 0;
        NewtonResult step_result;
        Real dt_used = dt;

        while (!accepted && retries <= options_.max_step_retries) {
            if (options_.stiffness_config.enable && stiffness_cooldown > 0) {
                dt = std::max(options_.dt_min, dt * options_.stiffness_config.dt_backoff);
                record_fallback_event(result,
                                      result.total_steps,
                                      retries,
                                      t,
                                      dt,
                                      FallbackReasonCode::StiffnessBackoff,
                                      SolverStatus::Success,
                                      "dt_backoff");
                if (options_.enable_bdf_order_control) {
                    bdf_controller_.set_order(
                        std::min(bdf_controller_.current_order(), options_.stiffness_config.max_bdf_order));
                } else if (options_.stiffness_config.switch_integrator) {
                    circuit_.set_integration_method(options_.stiffness_config.stiff_integrator);
                    using_stiff_integrator = true;
                }
            }

            Real t_next = t + dt;
            dt_used = dt;

            step_result = solve_step(t_next, dt_used, x);

            if (step_result.status != SolverStatus::Success) {
                circuit_.clear_stage_context();
                dt = std::max(options_.dt_min, dt * 0.5);
                result.timestep_rejections++;
                retries++;
                record_fallback_event(result,
                                      result.total_steps,
                                      retries,
                                      t,
                                      dt_used,
                                      FallbackReasonCode::NewtonFailure,
                                      step_result.status,
                                      "halve_dt");
                rejection_streak++;
                high_iter_streak = 0;
                if (options_.stiffness_config.enable &&
                    rejection_streak >= options_.stiffness_config.rejection_streak_threshold) {
                    stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                }
                if (options_.fallback_policy.enable_transient_gmin &&
                    retries >= options_.fallback_policy.gmin_retry_threshold) {
                    Real next_gmin = transient_gmin_ > 0.0
                        ? transient_gmin_ * options_.fallback_policy.gmin_growth
                        : options_.fallback_policy.gmin_initial;
                    next_gmin = std::min(options_.fallback_policy.gmin_max, next_gmin);
                    if (next_gmin > transient_gmin_) {
                        transient_gmin_ = next_gmin;
                        std::ostringstream action;
                        action << "gmin=" << transient_gmin_;
                        record_fallback_event(result,
                                              result.total_steps,
                                              retries,
                                              t,
                                              dt_used,
                                              FallbackReasonCode::TransientGminEscalation,
                                              step_result.status,
                                              action.str());
                    }
                }
                if (options_.enable_bdf_order_control) {
                    (void)bdf_controller_.reduce_on_failure();
                }
                continue;
            }

            Real lte = -1.0;
            if (options_.adaptive_timestep && lte_estimator_.has_sufficient_history()) {
                lte = lte_estimator_.compute(step_result.solution,
                                             circuit_.num_nodes(),
                                             circuit_.num_branches());
            }

            if (options_.adaptive_timestep && lte >= 0.0) {
                auto decision = timestep_controller_.compute_combined(
                    lte, step_result.iterations,
                    options_.enable_bdf_order_control ? bdf_controller_.current_order() : 2);

                if (!decision.accepted) {
                    circuit_.clear_stage_context();
                    dt = decision.dt_new;
                    result.timestep_rejections++;
                    retries++;
                    std::ostringstream action;
                    action << "lte_dt=" << decision.dt_new;
                    record_fallback_event(result,
                                          result.total_steps,
                                          retries,
                                          t,
                                          dt_used,
                                          FallbackReasonCode::LTERejection,
                                          step_result.status,
                                          action.str());
                    rejection_streak++;
                    high_iter_streak = 0;
                    if (options_.stiffness_config.enable &&
                        rejection_streak >= options_.stiffness_config.rejection_streak_threshold) {
                        stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                    }
                    continue;
                }

                dt = decision.dt_new;
            }

            // Event-aligned step splitting for hard switching edges
            if (options_.enable_events && dt_used > options_.dt_min * 1.01) {
                bool split_for_event = false;
                for (auto& sw : switch_monitors_) {
                    Real v_now = (sw.ctrl >= 0) ? step_result.solution[sw.ctrl] : 0.0;
                    bool now_on = v_now > sw.v_threshold;
                    if (now_on != sw.was_on) {
                        Real t_event = t + dt_used;
                        Vector x_event = step_result.solution;
                        if (find_switch_event_time(sw, t, t + dt_used, x, t_event, x_event)) {
                            Real dt_event = t_event - t;
                            if (dt_event > options_.dt_min * 1.01 && dt_event < dt_used * 0.999) {
                                dt = std::max(options_.dt_min, dt_event);
                                retries++;
                                record_fallback_event(result,
                                                      result.total_steps,
                                                      retries,
                                                      t,
                                                      dt_used,
                                                      FallbackReasonCode::EventSplit,
                                                      step_result.status,
                                                      "split_to_event");
                                split_for_event = true;
                                break;
                            }
                        }
                    }
                }
                if (split_for_event) {
                    circuit_.clear_stage_context();
                    continue;
                }
            }

            accepted = true;
        }

        if (!accepted) {
            record_fallback_event(result,
                                  result.total_steps,
                                  retries,
                                  t,
                                  dt_used,
                                  FallbackReasonCode::MaxRetriesExceeded,
                                  step_result.status,
                                  "abort_step");

            if (can_auto_recover && global_recovery_attempts < kMaxGlobalRecoveryAttempts) {
                ++global_recovery_attempts;
                auto_recovery_attempted = true;

                circuit_.clear_stage_context();
                dt = std::max(options_.dt_min, dt_used * (global_recovery_attempts == 1 ? Real{0.25} : Real{0.1}));
                rejection_streak = 0;
                high_iter_streak = 0;
                stiffness_cooldown = std::max(stiffness_cooldown, options_.stiffness_config.cooldown_steps);

                NewtonOptions tuned_newton = newton_solver_.options();
                apply_robust_newton_defaults(tuned_newton);
                tuned_newton.max_iterations = std::max(tuned_newton.max_iterations, 200);
                newton_solver_.set_options(tuned_newton);

                LinearSolverStackConfig tuned_linear = options_.linear_solver;
                apply_robust_linear_solver_defaults(tuned_linear);
                newton_solver_.linear_solver().set_config(tuned_linear);

                if (options_.fallback_policy.enable_transient_gmin) {
                    Real next_gmin = transient_gmin_ > 0.0
                        ? transient_gmin_ * options_.fallback_policy.gmin_growth
                        : options_.fallback_policy.gmin_initial;
                    transient_gmin_ = std::min(options_.fallback_policy.gmin_max,
                                               std::max(next_gmin, options_.fallback_policy.gmin_initial));
                }

                if (options_.enable_bdf_order_control) {
                    bdf_controller_.set_order(1);
                    circuit_.set_integration_order(1);
                } else {
                    circuit_.set_integration_method(Integrator::TRBDF2);
                    using_stiff_integrator = true;
                }

                std::ostringstream action;
                action << "global_recovery_" << global_recovery_attempts;
                if (transient_gmin_ > 0.0) {
                    action << "_gmin=" << transient_gmin_;
                }
                record_fallback_event(result,
                                      result.total_steps,
                                      retries + global_recovery_attempts,
                                      t,
                                      dt,
                                      FallbackReasonCode::MaxRetriesExceeded,
                                      step_result.status,
                                      action.str());
                continue;
            }

            result.success = false;
            result.final_status = step_result.status;
            result.message = "Transient failed at t=" + std::to_string(t + dt_used) +
                             ": " + step_result.error_message;
            if (auto_recovery_attempted) {
                result.message += " (automatic regularization attempted)";
            }
            break;
        }

        result.newton_iterations_total += step_result.iterations;
        rejection_streak = 0;
        global_recovery_attempts = 0;
        transient_gmin_ = 0.0;

        if (options_.stiffness_config.enable) {
            if (step_result.iterations >= options_.stiffness_config.newton_iter_threshold) {
                high_iter_streak++;
            } else {
                high_iter_streak = 0;
            }

            if (high_iter_streak >= options_.stiffness_config.newton_streak_threshold) {
                stiffness_cooldown = options_.stiffness_config.cooldown_steps;
            }

            if (options_.stiffness_config.monitor_conditioning) {
                auto telemetry = newton_solver_.linear_solver().telemetry();
                if (telemetry.last_error > options_.stiffness_config.conditioning_error_threshold) {
                    stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                }
            }

            if (stiffness_cooldown > 0) {
                stiffness_cooldown--;
            } else if (using_stiff_integrator && options_.stiffness_config.switch_integrator &&
                       !options_.enable_bdf_order_control) {
                circuit_.set_integration_method(base_integrator);
                using_stiff_integrator = false;
            }
        }

        if (options_.enable_events) {
            for (auto& sw : switch_monitors_) {
                Real v_now = (sw.ctrl >= 0) ? step_result.solution[sw.ctrl] : 0.0;
                bool now_on = v_now > sw.v_threshold;

                if (now_on != sw.was_on) {
                    Real t_event = t + dt_used;
                    Vector x_event = step_result.solution;

                    if (find_switch_event_time(sw, t, t + dt_used, x, t_event, x_event)) {
                        record_switch_event(sw, t_event, x_event, now_on, result, event_callback);
                    } else {
                        record_switch_event(sw, t + dt_used, step_result.solution, now_on, result, event_callback);
                    }

                    sw.was_on = now_on;
                }
            }
        }

        accumulate_conduction_losses(step_result.solution, dt_used);
        update_thermal_state(dt_used);

        t += dt_used;
        x = step_result.solution;
        circuit_.update_history(x);

        if (options_.adaptive_timestep) {
            lte_estimator_.record_solution(x, t, dt_used);

            if (options_.enable_bdf_order_control && lte_estimator_.has_sufficient_history()) {
                Real lte = lte_estimator_.compute(x, circuit_.num_nodes(), circuit_.num_branches());
                auto decision = bdf_controller_.select_order(lte, options_.timestep_config.error_tolerance);
                if (decision.order_changed) {
                    circuit_.set_integration_order(std::clamp(decision.new_order, 1, 2));
                }
            }
        }

        result.time.push_back(t);
        result.states.push_back(x);
        append_virtual_sample(x, t);

        if (callback) {
            callback(t, x);
        }

        result.total_steps++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (result.success) {
        result.final_status = SolverStatus::Success;
        if (result.message.empty()) {
            result.message = "Transient completed";
        }
    }

    finalize_loss_summary(result);
    finalize_thermal_summary(result);

    result.linear_solver_telemetry = newton_solver_.linear_solver().telemetry();

    return result;
}

SimulationResult Simulator::run_transient_with_progress(
    SimulationCallback callback,
    EventCallback event_callback,
    SimulationControl* control,
    const ProgressCallbackConfig& progress_config) {

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_progress_time = start_time;
    int steps_since_progress = 0;
    int64_t steps_total = 0;

    auto wrapped_callback = [&](Real time, const Vector& state) {
        steps_total++;
        steps_since_progress++;

        if (callback) {
            callback(time, state);
        }

        if (!progress_config.callback) {
            return;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_progress_time).count();

        bool due_time = elapsed_ms >= progress_config.min_interval_ms;
        bool due_steps = steps_since_progress >= progress_config.min_steps;

        if (due_time && due_steps) {
            SimulationProgress progress;
            progress.current_time = time;
            progress.total_time = options_.tstop;
            progress.progress_percent = options_.tstop > 0
                ? 100.0 * progress.current_time / options_.tstop
                : 0.0;
            progress.steps_completed = steps_total;
            progress.elapsed_seconds = std::chrono::duration<double>(now - start_time).count();

            progress_config.callback(progress);

            last_progress_time = now;
            steps_since_progress = 0;
        }
    };

    SimulationResult result = run_transient(wrapped_callback, event_callback, control);

    if (progress_config.callback) {
        SimulationProgress progress;
        if (!result.time.empty()) {
            progress.current_time = result.time.back();
            progress.total_time = options_.tstop;
            progress.progress_percent = options_.tstop > 0
                ? 100.0 * progress.current_time / options_.tstop
                : 0.0;
            progress.steps_completed = steps_total;
        }
        progress.elapsed_seconds = result.total_time_seconds;
        progress_config.callback(progress);
    }

    return result;
}

PeriodicSteadyStateResult Simulator::run_periodic_shooting(const PeriodicSteadyStateOptions& options) {
    auto dc = dc_operating_point();
    if (!dc.success) {
        PeriodicSteadyStateResult result;
        result.success = false;
        result.message = "DC operating point failed: " + dc.message;
        return result;
    }

    return run_periodic_shooting(dc.newton_result.solution, options);
}

PeriodicSteadyStateResult Simulator::run_periodic_shooting(const Vector& x0,
                                                          const PeriodicSteadyStateOptions& options) {
    PeriodicSteadyStateResult result;

    if (options.period <= 0.0) {
        result.message = "Periodic shooting requires a positive period";
        return result;
    }
    if (x0.size() == 0) {
        result.message = "Initial state is empty";
        return result;
    }

    SimulationOptions local = options_;
    local.tstart = 0.0;
    local.tstop = options.period;
    if (local.dt <= 0.0) {
        local.dt = options.period / 100.0;
    }
    if (local.dt > options.period) {
        local.dt = options.period;
    }

    Simulator shooting_sim(circuit_, local);

    Vector guess = x0;
    Vector residual;
    for (int iter = 0; iter < options.max_iterations; ++iter) {
        SimulationResult cycle = shooting_sim.run_transient(guess);
        if (!cycle.success || cycle.states.empty()) {
            result.success = false;
            result.message = "Shooting cycle failed: " + cycle.message;
            result.last_cycle = cycle;
            return result;
        }

        Vector xT = cycle.states.back();
        residual = xT - guess;
        Real rms = std::sqrt(residual.squaredNorm() / static_cast<Real>(residual.size()));

        result.iterations = iter + 1;
        result.residual_norm = rms;
        if (options.store_last_transient) {
            result.last_cycle = cycle;
        }

        if (rms <= options.tolerance) {
            result.success = true;
            result.steady_state = xT;
            result.message = "Periodic steady-state converged";
            return result;
        }

        guess = guess + options.relaxation * residual;
    }

    result.success = false;
    result.steady_state = guess;
    result.message = "Periodic steady-state did not converge within max iterations";
    return result;
}

HarmonicBalanceResult Simulator::run_harmonic_balance(const HarmonicBalanceOptions& options) {
    auto dc = dc_operating_point();
    if (!dc.success) {
        HarmonicBalanceResult result;
        result.success = false;
        result.message = "DC operating point failed: " + dc.message;
        return result;
    }

    return run_harmonic_balance(dc.newton_result.solution, options);
}

HarmonicBalanceResult Simulator::run_harmonic_balance(const Vector& x0,
                                                     const HarmonicBalanceOptions& options) {
    HarmonicBalanceResult result;
    const int n = static_cast<int>(circuit_.system_size());
    const int samples = std::max(3, options.num_samples);

    if (options.period <= 0.0) {
        result.message = "Harmonic balance requires a positive period";
        return result;
    }
    if (x0.size() != n) {
        result.message = "Initial state size mismatch";
        return result;
    }

    Matrix D = spectral_diff_matrix(samples, options.period);
    Matrix Dt = D.transpose();
    if (D.size() == 0) {
        result.message = "Failed to build spectral differentiation matrix";
        return result;
    }

    std::vector<Real> times;
    times.reserve(samples);
    const Real dt_sample = options.period / static_cast<Real>(samples);
    for (int k = 0; k < samples; ++k) {
        times.push_back(dt_sample * static_cast<Real>(k));
    }
    result.sample_times = times;

    Vector X = Vector::Zero(n * samples);
    if (options.initialize_from_transient) {
        SimulationOptions init_opts = options_;
        init_opts.tstart = 0.0;
        init_opts.tstop = options.period;
        init_opts.dt = dt_sample;
        init_opts.dt_min = dt_sample;
        init_opts.dt_max = dt_sample;
        init_opts.adaptive_timestep = false;
        init_opts.enable_bdf_order_control = false;
        init_opts.newton_options.num_nodes = circuit_.num_nodes();
        init_opts.newton_options.num_branches = circuit_.num_branches();

        Simulator init_sim(circuit_, init_opts);
        auto init_result = init_sim.run_transient(x0);
        if (init_result.success && init_result.states.size() >= static_cast<std::size_t>(samples)) {
            for (int k = 0; k < samples; ++k) {
                X.segment(k * n, n) = init_result.states[k];
            }
        } else {
            for (int k = 0; k < samples; ++k) {
                X.segment(k * n, n) = x0;
            }
        }
    } else {
        for (int k = 0; k < samples; ++k) {
            X.segment(k * n, n) = x0;
        }
    }

    Matrix dX(n, samples);

    auto residual_func = [&](const Vector& state, Vector& f_out) {
        f_out.setZero(state.size());
        const int nodes = static_cast<int>(circuit_.num_nodes());
        const int branches = static_cast<int>(circuit_.num_branches());

        Eigen::Map<const Matrix> X_mat(state.data(), n, samples);
        dX.noalias() = X_mat * Dt;

        for (int k = 0; k < samples; ++k) {
            Eigen::Map<const Vector> xk(state.data() + k * n, n);
            Vector fk(n);
            fk.setZero();
            const Real* col = dX.col(k).data();
            circuit_.assemble_residual_hb(fk, xk, times[k],
                                          std::span<const Real>(col, nodes),
                                          std::span<const Real>(col + nodes, branches));
            f_out.segment(k * n, n) = fk;
        }
    };

    NewtonOptions hb_newton = options_.newton_options;
    hb_newton.max_iterations = options.max_iterations;
    hb_newton.initial_damping = options.relaxation;
    hb_newton.min_damping = options.relaxation;
    hb_newton.auto_damping = false;
    hb_newton.tolerances.residual_tol = options.tolerance;
    hb_newton.krylov_residual_cache_tolerance = 1e-3;
    hb_newton.krylov_tolerance = std::max(options.tolerance * 0.1, Real{1e-10});
    hb_newton.krylov_residual_cache_tolerance = -1.0;
    hb_newton.enable_newton_krylov = true;
    hb_newton.reuse_jacobian_pattern = false;
    hb_newton.num_nodes = circuit_.num_nodes();
    hb_newton.num_branches = circuit_.num_branches();

    NewtonRaphsonSolver<RuntimeLinearSolver> hb_solver(hb_newton);
    LinearSolverStackConfig hb_linear = options_.linear_solver;
    hb_linear.auto_select = false;
    hb_linear.allow_fallback = true;
    hb_linear.order = {LinearSolverKind::GMRES};
    hb_linear.fallback_order = {LinearSolverKind::BiCGSTAB, LinearSolverKind::SparseLU};
    hb_linear.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::ILUT;
    hb_linear.iterative_config.tolerance = hb_newton.krylov_tolerance;
    hb_linear.iterative_config.restart = std::max(hb_linear.iterative_config.restart, 40);
    hb_linear.iterative_config.max_iterations = std::max(hb_linear.iterative_config.max_iterations, 300);
    hb_solver.linear_solver().set_config(hb_linear);

    auto system_func = [&](const Vector& state, Vector& f, SparseMatrix& J) {
        residual_func(state, f);
        const Index size = state.size();
        const Real eps_base = std::max(options_.newton_options.krylov_fd_epsilon, Real{1e-12});
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(static_cast<std::size_t>(size) * 4);

        Vector x_pert = state;
        Vector f_pert(size);

        for (Index col = 0; col < size; ++col) {
            Real step = eps_base * std::max(Real{1.0}, std::abs(state[col]));
            x_pert[col] = state[col] + step;
            residual_func(x_pert, f_pert);
            x_pert[col] = state[col];

            for (Index row = 0; row < size; ++row) {
                Real val = (f_pert[row] - f[row]) / step;
                if (std::abs(val) > 1e-12) {
                    triplets.emplace_back(row, col, val);
                }
            }
        }

        J.resize(size, size);
        if (!triplets.empty()) {
            J.setFromTriplets(triplets.begin(), triplets.end());
        } else {
            J.setZero();
        }
    };

    NewtonResult solve_result = hb_solver.solve(X, system_func, residual_func);
    result.iterations = solve_result.iterations;
    result.residual_norm = solve_result.final_residual;

    if (solve_result.status != SolverStatus::Success) {
        result.success = false;
        result.message = "HB solver failed: " + solve_result.error_message;
        result.solution = solve_result.solution;
        return result;
    }

    result.success = true;
    result.solution = solve_result.solution;
    result.message = "Harmonic balance converged";
    return result;
}

}  // namespace pulsim::v1
