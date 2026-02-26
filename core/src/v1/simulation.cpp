#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>

namespace pulsim::v1 {

namespace {
constexpr int kMaxGlobalRecoveryAttempts = 2;
constexpr int kLteEventGraceSteps = 2;
constexpr int kMaxDtMinHoldAdvances = 128;

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

void enforce_explicit_step_mode(SimulationOptions& options) {
    if (!options.step_mode_explicit) {
        return;
    }
    options.adaptive_timestep = (options.step_mode == TransientStepMode::Variable);
}

void apply_auto_transient_profile(SimulationOptions& options, const Circuit& circuit) {
    // Respect explicit strict mode: users can disable fallback for deterministic debugging.
    if (!options.linear_solver.allow_fallback) {
        return;
    }
    // Respect explicit fixed-step selection from user/scenario.
    // Auto profile can tune adaptive runs, but must not silently flip fixed -> variable.
    if (resolve_step_mode(options) == TransientStepMode::Fixed) {
        return;
    }

    const auto hints = analyze_circuit_robustness(circuit);
    const bool switching_topology =
        hints.switching_devices > 0 || hints.pwm_blocks > 0 || hints.control_blocks > 0;
    if (!switching_topology) {
        return;
    }
    const SimulationOptions default_options{};
    auto nearly_equal = [](Real lhs, Real rhs) {
        const Real scale = std::max<Real>({Real{1.0}, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <= scale * Real{1e-12};
    };

    const bool fixed_step = resolve_step_mode(options) == TransientStepMode::Fixed;
    if (!fixed_step) {
        options.adaptive_timestep = true;
        options.timestep_config = AdvancedTimestepConfig::for_power_electronics();
        options.timestep_config.dt_initial = options.dt;
        options.timestep_config.dt_min = std::max(options.dt_min, Real{1e-12});
        options.timestep_config.dt_max = std::max(options.timestep_config.dt_max, options.dt * 20.0);
        options.timestep_config.error_tolerance =
            std::min(options.timestep_config.error_tolerance, Real{1e-3});

        options.enable_bdf_order_control = true;
        options.bdf_config.min_order = 1;
        options.bdf_config.max_order = 2;
        options.bdf_config.initial_order = 1;
    }

    // In strongly switching nonlinear circuits, unconstrained default dt_min/dt_max
    // can create pathological picosecond retries. Keep a conservative adaptive window
    // unless the user explicitly configured these bounds.
    const bool nonlinear_switching = hints.switching_devices > 0 && hints.nonlinear_devices > 0;
    if (!fixed_step && nonlinear_switching) {
        if (nearly_equal(options.dt_max, default_options.dt_max)) {
            options.dt_max = std::max(options.dt, options.dt_min);
        }
        if (nearly_equal(options.dt_min, default_options.dt_min)) {
            options.dt_min = std::max(options.dt_min, options.dt * Real{5e-3});
        }
        if (options.dt_min > options.dt_max) {
            options.dt_min = options.dt_max;
        }
        options.timestep_config.dt_min = std::max(options.timestep_config.dt_min, options.dt_min);
        options.timestep_config.dt_max = std::min(options.timestep_config.dt_max, options.dt_max);
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

[[nodiscard]] bool sundials_compiled() {
#ifdef PULSIM_HAS_SUNDIALS
    return true;
#else
    return false;
#endif
}

[[nodiscard]] std::string backend_mode_to_string(TransientBackendMode mode) {
    switch (mode) {
        case TransientBackendMode::Native:
            return "native";
        case TransientBackendMode::SundialsOnly:
            return "sundials";
        case TransientBackendMode::Auto:
            return "auto";
    }
    return "native";
}

[[nodiscard]] std::string_view diagnostic_code_to_reason(SimulationDiagnosticCode code) {
    switch (code) {
        case SimulationDiagnosticCode::None:
            return "";
        case SimulationDiagnosticCode::DcOperatingPointFailure:
            return "dc_operating_point_failure";
        case SimulationDiagnosticCode::LegacyBackendUnsupported:
            return "legacy_backend_removed";
        case SimulationDiagnosticCode::InvalidInitialState:
            return "invalid_initial_state";
        case SimulationDiagnosticCode::InvalidTimeWindow:
            return "invalid_time_window";
        case SimulationDiagnosticCode::InvalidTimestep:
            return "invalid_timestep";
        case SimulationDiagnosticCode::UserStopRequested:
            return "user_stop_requested";
        case SimulationDiagnosticCode::TransientStepFailure:
            return "transient_step_failure";
        case SimulationDiagnosticCode::PeriodicInvalidPeriod:
            return "periodic_invalid_period";
        case SimulationDiagnosticCode::PeriodicInvalidInitialState:
            return "periodic_invalid_initial_state";
        case SimulationDiagnosticCode::PeriodicCycleFailure:
            return "periodic_cycle_failure";
        case SimulationDiagnosticCode::PeriodicNoConvergence:
            return "periodic_no_convergence";
        case SimulationDiagnosticCode::HarmonicInvalidPeriod:
            return "harmonic_invalid_period";
        case SimulationDiagnosticCode::HarmonicInvalidInitialState:
            return "harmonic_invalid_initial_state";
        case SimulationDiagnosticCode::HarmonicDifferentiationFailure:
            return "harmonic_diff_matrix_failure";
        case SimulationDiagnosticCode::HarmonicSolverFailure:
            return "harmonic_solver_failure";
    }
    return "";
}

struct TransientInputIssue {
    SimulationDiagnosticCode diagnostic = SimulationDiagnosticCode::None;
    std::string message;
};

[[nodiscard]] std::optional<TransientInputIssue> validate_transient_inputs(
    const Circuit& circuit,
    const SimulationOptions& options,
    const Vector& x0) {
    if (x0.size() == 0) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidInitialState,
            "Transient simulation requires a non-empty initial state"
        };
    }

    const Index expected_size = static_cast<Index>(circuit.system_size());
    if (x0.size() != expected_size) {
        std::ostringstream message;
        message << "Initial state size mismatch: expected " << expected_size
                << ", got " << x0.size();
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidInitialState,
            message.str()
        };
    }

    if (!x0.allFinite()) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidInitialState,
            "Initial state contains non-finite values"
        };
    }

    const bool finite_time_window = std::isfinite(options.tstart) && std::isfinite(options.tstop);
    if (!finite_time_window || options.tstop < options.tstart) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidTimeWindow,
            "Invalid simulation time window: tstop must be finite and >= tstart"
        };
    }

    const bool finite_timesteps = std::isfinite(options.dt) &&
                                  std::isfinite(options.dt_min) &&
                                  std::isfinite(options.dt_max);
    if (!finite_timesteps) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidTimestep,
            "Invalid timestep configuration: dt, dt_min, and dt_max must be finite"
        };
    }

    if (options.dt <= 0.0 || options.dt_min <= 0.0 || options.dt_max < options.dt_min) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidTimestep,
            "Invalid timestep bounds: require dt > 0, dt_min > 0, and dt_max >= dt_min"
        };
    }

    return std::nullopt;
}

[[nodiscard]] bool nearly_same_time(Real a, Real b) {
    const Real scale = std::max<Real>({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= scale * Real{1e-12};
}

class VariableStepPolicy final {
public:
    VariableStepPolicy() = default;

    VariableStepPolicy(AdvancedTimestepController& controller, Real dt_min, Real dt_max)
        : enabled_(true)
        , controller_(&controller)
        , dt_min_(std::max(dt_min, Real{1e-18}))
        , dt_max_(std::max(dt_max, dt_min_)) {
        const auto& cfg = controller.config();
        min_shrink_factor_ = std::clamp(cfg.shrink_factor, Real{0.05}, Real{1.0});
        max_growth_factor_ = std::max(cfg.growth_factor, Real{1.0});
    }

    [[nodiscard]] bool enabled() const {
        return enabled_ && controller_ != nullptr;
    }

    [[nodiscard]] Real clamp_dt(Real t_now, Real dt_candidate, Real t_stop) const {
        Real dt = std::clamp(dt_candidate, dt_min_, dt_max_);
        if (t_now + dt > t_stop) {
            dt = t_stop - t_now;
        }
        return std::max<Real>(0.0, dt);
    }

    [[nodiscard]] AdvancedTimestepDecision evaluate(Real lte,
                                                    int newton_iterations,
                                                    int integration_order,
                                                    Real dt_current) const {
        if (!enabled() || !std::isfinite(lte) || lte < 0.0) {
            AdvancedTimestepDecision passthrough;
            passthrough.accepted = true;
            passthrough.dt_new = std::clamp(dt_current, dt_min_, dt_max_);
            passthrough.error_ratio = 0.0;
            passthrough.newton_iterations = newton_iterations;
            return passthrough;
        }

        AdvancedTimestepDecision decision =
            controller_->compute_combined(lte, newton_iterations, integration_order);
        const Real dt_ref = std::clamp(dt_current, dt_min_, dt_max_);
        const Real dt_guard_min = std::max(dt_min_, dt_ref * min_shrink_factor_);
        const Real dt_guard_max = std::min(dt_max_, dt_ref * max_growth_factor_);
        decision.dt_new = std::clamp(decision.dt_new, dt_guard_min, dt_guard_max);
        return decision;
    }

    void on_step_accepted(Real dt_used) {
        if (!enabled()) {
            return;
        }
        controller_->accept(std::clamp(dt_used, dt_min_, dt_max_));
    }

private:
    bool enabled_ = false;
    AdvancedTimestepController* controller_ = nullptr;
    Real dt_min_ = 0.0;
    Real dt_max_ = 0.0;
    Real min_shrink_factor_ = 0.5;
    Real max_growth_factor_ = 2.0;
};

class FixedStepPolicy final {
public:
    FixedStepPolicy() = default;

    FixedStepPolicy(Real t_start,
                    Real t_stop,
                    Real macro_dt,
                    Real dt_min,
                    int max_substeps_per_macro,
                    int max_recovery_retries_per_macro)
        : enabled_(true)
        , t_stop_(t_stop)
        , dt_min_(std::max(dt_min, Real{1e-18}))
        , macro_dt_(std::max(macro_dt, dt_min_))
        , max_substeps_per_macro_(std::max(1, max_substeps_per_macro))
        , max_recovery_retries_per_macro_(std::max(1, max_recovery_retries_per_macro)) {
        current_macro_target_ = std::min(t_stop_, t_start + macro_dt_);
        if (current_macro_target_ <= t_start && t_stop_ > t_start) {
            current_macro_target_ = t_stop_;
        }
    }

    [[nodiscard]] bool enabled() const {
        return enabled_;
    }

    [[nodiscard]] Real default_dt() const {
        return macro_dt_;
    }

    [[nodiscard]] bool can_take_internal_substep() const {
        if (!enabled_) {
            return true;
        }
        return substeps_in_current_macro_ < max_substeps_per_macro_;
    }

    [[nodiscard]] bool register_recovery_retry() {
        if (!enabled_) {
            return true;
        }
        recovery_retries_in_current_macro_ += 1;
        return recovery_retries_in_current_macro_ <= max_recovery_retries_per_macro_;
    }

    [[nodiscard]] Real clamp_dt(Real t_now, Real dt_candidate) {
        if (!enabled_) {
            return dt_candidate;
        }

        advance_macro_cursor(t_now);
        Real dt = std::max(dt_min_, dt_candidate);

        const Real target = std::min(t_stop_, current_macro_target_);
        const Real remaining = target - t_now;
        if (remaining > dt_min_ * 0.1) {
            dt = std::min(dt, remaining);
        } else {
            const Real t_stop_remaining = t_stop_ - t_now;
            dt = t_stop_remaining > 0.0 ? std::min(dt, t_stop_remaining) : 0.0;
        }
        return std::max<Real>(0.0, dt);
    }

    [[nodiscard]] bool on_step_accepted(Real t_next) {
        if (!enabled_) {
            return true;
        }
        const Real tol = std::max<Real>(dt_min_ * 1e-3, 1e-15);
        bool reached_macro_target = t_next >= (current_macro_target_ - tol) ||
                                    nearly_same_time(t_next, t_stop_);
        if (!reached_macro_target) {
            substeps_in_current_macro_ += 1;
        }
        advance_macro_cursor(t_next);
        return reached_macro_target;
    }

private:
    void advance_macro_cursor(Real t_now) {
        const Real tol = std::max<Real>(dt_min_ * 1e-3, 1e-15);
        while (current_macro_target_ < t_stop_ &&
               t_now >= current_macro_target_ - tol) {
            current_macro_target_ = std::min(t_stop_, current_macro_target_ + macro_dt_);
            substeps_in_current_macro_ = 0;
            recovery_retries_in_current_macro_ = 0;
        }
    }

    bool enabled_ = false;
    Real t_stop_ = 0.0;
    Real dt_min_ = 0.0;
    Real macro_dt_ = 0.0;
    Real current_macro_target_ = 0.0;
    int max_substeps_per_macro_ = 1;
    int max_recovery_retries_per_macro_ = 1;
    int substeps_in_current_macro_ = 0;
    int recovery_retries_in_current_macro_ = 0;
};

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
    enforce_explicit_step_mode(options_);
    apply_auto_transient_profile(options_, circuit_);
    enforce_explicit_step_mode(options_);
    options_.newton_options.num_nodes = circuit_.num_nodes();
    options_.newton_options.num_branches = circuit_.num_branches();
    newton_solver_.set_options(options_.newton_options);
    newton_solver_.linear_solver().set_config(options_.linear_solver);
    transient_services_ =
        make_default_transient_service_registry(circuit_, options_, newton_solver_);

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
}

void Simulator::initialize_loss_tracking() {
    if (transient_services_.loss_service) {
        transient_services_.loss_service->reset();
    }
    initialize_thermal_tracking();
}

void Simulator::set_switching_energy(const std::string& device_name, const SwitchingEnergy& energy) {
    options_.switching_energy[device_name] = energy;
    if (transient_services_.loss_service) {
        transient_services_.loss_service->reset();
    }
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

void Simulator::initialize_thermal_tracking() {
    if (transient_services_.thermal_service) {
        transient_services_.thermal_service->reset();
    }
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

SimulationResult Simulator::run_transient(SimulationCallback callback,
                                          EventCallback event_callback,
                                          SimulationControl* control) {
    auto dc = dc_operating_point();
    if (!dc.success) {
        SimulationResult result;
        result.success = false;
        result.final_status = dc.newton_result.status;
        result.diagnostic = SimulationDiagnosticCode::DcOperatingPointFailure;
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
    const auto requested_backend = options_.transient_backend;

    if (requested_backend != TransientBackendMode::Native) {
        SimulationResult result;
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.diagnostic = SimulationDiagnosticCode::LegacyBackendUnsupported;
        result.message =
            "Legacy transient backend selection '" + backend_mode_to_string(requested_backend) +
            "' is no longer supported. Use the native core with simulation.step_mode: fixed|variable.";
        result.backend_telemetry.requested_backend = backend_mode_to_string(requested_backend);
        result.backend_telemetry.selected_backend = "native";
        result.backend_telemetry.solver_family = "native";
        result.backend_telemetry.formulation_mode = "native";
        result.backend_telemetry.sundials_compiled = sundials_compiled();
        result.backend_telemetry.failure_reason =
            std::string(diagnostic_code_to_reason(result.diagnostic));
        return result;
    }

    if (const auto input_issue = validate_transient_inputs(circuit_, options_, x0)) {
        SimulationResult result;
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.diagnostic = input_issue->diagnostic;
        result.message = input_issue->message;
        result.backend_telemetry.requested_backend = "native";
        result.backend_telemetry.selected_backend = "native";
        result.backend_telemetry.solver_family = "native";
        result.backend_telemetry.formulation_mode = "native";
        result.backend_telemetry.sundials_compiled = sundials_compiled();
        result.backend_telemetry.failure_reason =
            std::string(diagnostic_code_to_reason(result.diagnostic));
        return result;
    }

    SimulationResult native_result = run_transient_native_impl(
        x0,
        std::move(callback),
        std::move(event_callback),
        control);

    native_result.backend_telemetry.requested_backend = "native";
    native_result.backend_telemetry.sundials_compiled = sundials_compiled();
    if (native_result.backend_telemetry.selected_backend.empty()) {
        native_result.backend_telemetry.selected_backend = "native";
    }
    if (native_result.backend_telemetry.solver_family.empty()) {
        native_result.backend_telemetry.solver_family = "native";
    }
    if (native_result.backend_telemetry.formulation_mode.empty()) {
        native_result.backend_telemetry.formulation_mode = "native";
    }
    return native_result;
}

SimulationResult Simulator::run_transient_native_impl(const Vector& x0,
                                                      SimulationCallback callback,
                                                      EventCallback event_callback,
                                                      SimulationControl* control) {
    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    result.backend_telemetry.selected_backend = "native";
    result.backend_telemetry.solver_family = "native";
    result.backend_telemetry.formulation_mode = "native";
    result.backend_telemetry.sundials_compiled = sundials_compiled();

    initialize_loss_tracking();
    lte_estimator_.reset();
    transient_gmin_ = 0.0;
    direct_assemble_system_calls_ = 0;
    direct_assemble_residual_calls_ = 0;
    direct_assemble_system_time_seconds_ = 0.0;
    direct_assemble_residual_time_seconds_ = 0.0;
    transient_services_.equation_assembler->reset_telemetry();
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
    int dt_min_hold_advances = 0;
    bool auto_recovery_attempted = false;
    const TransientStepMode step_mode = resolve_step_mode(options_);
    const bool can_auto_recover = (step_mode == TransientStepMode::Variable) &&
                                  options_.linear_solver.allow_fallback;
    VariableStepPolicy variable_step_policy;
    if (step_mode == TransientStepMode::Variable) {
        variable_step_policy = VariableStepPolicy(
            timestep_controller_,
            options_.dt_min,
            options_.dt_max);
    }
    FixedStepPolicy fixed_step_policy;
    if (step_mode == TransientStepMode::Fixed) {
        const int fixed_substep_budget = std::max(4, options_.max_step_retries + 4);
        const int fixed_recovery_budget = std::max(8, options_.max_step_retries + 6);
        fixed_step_policy = FixedStepPolicy(
            options_.tstart,
            options_.tstop,
            std::max(options_.dt, options_.dt_min),
            options_.dt_min,
            fixed_substep_budget,
            fixed_recovery_budget);
        dt = fixed_step_policy.clamp_dt(t, fixed_step_policy.default_dt());
    } else if (variable_step_policy.enabled()) {
        dt = variable_step_policy.clamp_dt(t, dt, options_.tstop);
    }
    const NewtonOptions baseline_newton_options = transient_services_.nonlinear_solve->options();

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

    int lte_discontinuity_grace_steps = 0;

    while (t < options_.tstop) {
        if (control) {
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                result.diagnostic = SimulationDiagnosticCode::UserStopRequested;
                break;
            }
            while (control->should_pause() && !control->should_stop()) {
                control->wait_until_resumed();
            }
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                result.diagnostic = SimulationDiagnosticCode::UserStopRequested;
                break;
            }
        }

        auto clamp_dt_for_mode = [&](Real dt_candidate) {
            if (fixed_step_policy.enabled()) {
                return fixed_step_policy.clamp_dt(t, dt_candidate);
            }
            if (variable_step_policy.enabled()) {
                return variable_step_policy.clamp_dt(t, dt_candidate, options_.tstop);
            }
            return dt_candidate;
        };
        auto min_event_substep_dt = [&](Real dt_reference) {
            const Real dt_ref = std::max(std::abs(dt_reference), options_.dt_min);
            const Real profile_floor = std::max(std::abs(options_.dt) * Real{1e-4}, Real{1e-12});
            return std::max(options_.dt_min,
                            std::max(profile_floor, dt_ref * Real{1e-3}));
        };
        auto near_switching_threshold = [&](const Vector& state) {
            if (!options_.enable_events) {
                return false;
            }
            for (const auto& sw : switch_monitors_) {
                if (sw.ctrl < 0 || sw.ctrl >= state.size()) {
                    continue;
                }
                const Real v_ctrl = state[sw.ctrl];
                const Real threshold_scale = std::max<Real>(std::abs(sw.v_threshold), Real{1.0});
                const Real guard_band = std::max<Real>(threshold_scale * 0.05, Real{0.05});
                if (std::abs(v_ctrl - sw.v_threshold) <= guard_band) {
                    return true;
                }
            }
            return false;
        };

        if (fixed_step_policy.enabled()) {
            dt = clamp_dt_for_mode(fixed_step_policy.default_dt());
        } else {
            dt = clamp_dt_for_mode(dt);
        }
        bool discontinuity_adjacent = variable_step_policy.enabled() && near_switching_threshold(x);
        if (discontinuity_adjacent && options_.stiffness_config.enable && stiffness_cooldown <= 0) {
            stiffness_cooldown = std::max(1, options_.stiffness_config.cooldown_steps);
        }

        if (variable_step_policy.enabled() && options_.enable_events &&
            discontinuity_adjacent && dt > options_.dt_min * 1.01) {
            const Real clipped_dt = clamp_dt_for_mode(std::max(options_.dt_min, dt * 0.5));
            if (clipped_dt < dt) {
                record_fallback_event(result,
                                      result.total_steps,
                                      0,
                                      t,
                                      dt,
                                      FallbackReasonCode::EventSplit,
                                      SolverStatus::Success,
                                      "adaptive_event_clip");
                dt = clipped_dt;
                lte_discontinuity_grace_steps =
                    std::max(lte_discontinuity_grace_steps, kLteEventGraceSteps);
            }
        }
        if (dt < options_.dt_min * 0.1) {
            break;
        }

        bool accepted = false;
        int retries = 0;
        NewtonResult step_result;
        Real dt_used = dt;
        const Vector step_anchor_state = x;
        const bool post_event_lte_grace_active = lte_discontinuity_grace_steps > 0;
        bool accepted_step_event_adjacent = false;
        bool accepted_step_lte_guarded = false;
        bool split_encountered_this_step = false;

        while (!accepted && retries <= options_.max_step_retries) {
            auto consume_fixed_recovery_budget = [&](const std::string& action_tag) {
                if (!fixed_step_policy.enabled()) {
                    return true;
                }
                if (fixed_step_policy.register_recovery_retry()) {
                    return true;
                }
                record_fallback_event(result,
                                      result.total_steps,
                                      retries,
                                      t,
                                      dt_used,
                                      FallbackReasonCode::MaxRetriesExceeded,
                                      step_result.status,
                                      action_tag);
                retries = options_.max_step_retries + 1;
                return false;
            };

            if (options_.stiffness_config.enable && stiffness_cooldown > 0) {
                const bool discontinuity_profile = discontinuity_adjacent && retries <= 1;
                if (!discontinuity_profile && retries > 0) {
                    dt = clamp_dt_for_mode(std::max(options_.dt_min, dt * options_.stiffness_config.dt_backoff));
                    record_fallback_event(result,
                                          result.total_steps,
                                          retries,
                                          t,
                                          dt,
                                          FallbackReasonCode::StiffnessBackoff,
                                          SolverStatus::Success,
                                          "dt_backoff");
                }
                if (options_.enable_bdf_order_control) {
                    const int previous_order = bdf_controller_.current_order();
                    const int capped_order =
                        std::min(previous_order, options_.stiffness_config.max_bdf_order);
                    bdf_controller_.set_order(capped_order);
                    if (discontinuity_profile && capped_order != previous_order) {
                        record_fallback_event(result,
                                              result.total_steps,
                                              retries,
                                              t,
                                              dt,
                                              FallbackReasonCode::StiffnessBackoff,
                                              SolverStatus::Success,
                                              "discontinuity_bdf_profile");
                    }
                } else if (options_.stiffness_config.switch_integrator) {
                    if (!using_stiff_integrator) {
                        circuit_.set_integration_method(options_.stiffness_config.stiff_integrator);
                        using_stiff_integrator = true;
                        if (discontinuity_profile) {
                            record_fallback_event(result,
                                                  result.total_steps,
                                                  retries,
                                                  t,
                                                  dt,
                                                  FallbackReasonCode::StiffnessBackoff,
                                                  SolverStatus::Success,
                                                  "discontinuity_stiff_profile");
                        }
                    }
                }
            }

            TransientStepRequest step_request;
            step_request.mode = step_mode;
            step_request.t_now = t;
            step_request.t_target = t + dt;
            step_request.dt_candidate = dt;
            step_request.dt_min = options_.dt_min;
            step_request.pwm_boundary_time = std::numeric_limits<Real>::quiet_NaN();
            step_request.dead_time_boundary_time = std::numeric_limits<Real>::quiet_NaN();
            step_request.threshold_crossing_time = std::numeric_limits<Real>::quiet_NaN();
            step_request.retry_index = retries;
            step_request.max_retries = std::max(1, options_.max_step_retries + 1);
            step_request.event_adjacent = discontinuity_adjacent;
            bool calendar_event_clipped = false;

            bool has_calendar_boundary_in_step = false;
            Real dt_segment = 0.0;
            if (options_.enable_events && dt > options_.dt_min * 1.01) {
                const Real t_segment_target =
                    transient_services_.event_scheduler->next_segment_target(step_request, t + dt);
                dt_segment = t_segment_target - t;
                const Real min_calendar_clip_dt = min_event_substep_dt(dt);
                has_calendar_boundary_in_step =
                    dt_segment > min_calendar_clip_dt && dt_segment < dt * 0.999;
            }

            if (variable_step_policy.enabled() && has_calendar_boundary_in_step) {
                step_request.event_adjacent = true;
                discontinuity_adjacent = true;
                if (options_.stiffness_config.enable && stiffness_cooldown <= 0) {
                    stiffness_cooldown = std::max(1, options_.stiffness_config.cooldown_steps);
                }
            }

            const bool allow_event_calendar_clipping =
                options_.enable_events && fixed_step_policy.enabled();
            if (allow_event_calendar_clipping && has_calendar_boundary_in_step) {
                if (fixed_step_policy.enabled() && !fixed_step_policy.can_take_internal_substep()) {
                    record_fallback_event(result,
                                          result.total_steps,
                                          retries,
                                          t,
                                          dt,
                                          FallbackReasonCode::MaxRetriesExceeded,
                                          SolverStatus::Success,
                                          "fixed_substep_budget_reached_skip_calendar_clip");
                } else {
                    dt = clamp_dt_for_mode(std::max(options_.dt_min, dt_segment));
                    step_request.t_target = t + dt;
                    step_request.dt_candidate = dt;
                    step_request.event_adjacent = true;
                    calendar_event_clipped = true;
                    discontinuity_adjacent = true;
                    if (options_.stiffness_config.enable && stiffness_cooldown <= 0) {
                        stiffness_cooldown = std::max(1, options_.stiffness_config.cooldown_steps);
                    }
                    record_fallback_event(result,
                                          result.total_steps,
                                          retries,
                                          t,
                                          dt_segment,
                                          FallbackReasonCode::EventSplit,
                                          SolverStatus::Success,
                                          "event_calendar_clip");
                }
            }

            transient_services_.telemetry_collector->on_step_attempt(step_request);

            Real t_next = t + dt;
            dt_used = dt;

            step_result = solve_step(t_next, dt_used, x);
            if (last_step_segment_cache_hit_) {
                result.backend_telemetry.segment_model_cache_hits += 1;
            } else {
                result.backend_telemetry.segment_model_cache_misses += 1;
            }
            if (last_step_linear_factor_cache_hit_) {
                result.backend_telemetry.linear_factor_cache_hits += 1;
            }
            if (last_step_linear_factor_cache_miss_) {
                result.backend_telemetry.linear_factor_cache_misses += 1;
                if (!last_step_linear_factor_cache_invalidation_reason_.empty()) {
                    result.backend_telemetry.linear_factor_cache_invalidations += 1;
                    result.backend_telemetry.linear_factor_cache_last_invalidation_reason =
                        last_step_linear_factor_cache_invalidation_reason_;
                }
            }
            if (last_step_solve_path_ == StepSolvePath::SegmentPrimary) {
                result.backend_telemetry.state_space_primary_steps += 1;
            } else {
                result.backend_telemetry.dae_fallback_steps += 1;
                if (last_step_solve_reason_.find("segment_not_admissible") != std::string::npos) {
                    result.backend_telemetry.segment_non_admissible_steps += 1;
                }
            }

            if (step_result.status != SolverStatus::Success) {
                circuit_.clear_stage_context();
                RecoveryDecision recovery = transient_services_.recovery_manager->on_step_failure(step_request);
                transient_services_.telemetry_collector->on_step_reject(recovery);
                const Real recovered_dt =
                    recovery.next_dt > 0.0 ? recovery.next_dt : (step_request.dt_candidate * 0.5);
                dt = clamp_dt_for_mode(std::max(options_.dt_min, recovered_dt));
                result.timestep_rejections++;
                retries++;
                if (!consume_fixed_recovery_budget("fixed_recovery_budget_newton")) {
                    x = step_anchor_state;
                    continue;
                }

                FallbackReasonCode recovery_reason_code = FallbackReasonCode::NewtonFailure;
                std::string recovery_action =
                    recovery.reason.empty() ? std::string("recover_dt") : recovery.reason;

                if (recovery.stage == RecoveryStage::GlobalizationEscalation) {
                    NewtonOptions tuned_newton = transient_services_.nonlinear_solve->options();
                    tuned_newton.auto_damping = true;
                    tuned_newton.enable_trust_region = true;
                    tuned_newton.max_iterations = std::max(tuned_newton.max_iterations, 120);
                    tuned_newton.min_damping = std::min(tuned_newton.min_damping, Real{1e-5});
                    tuned_newton.initial_damping = std::min(tuned_newton.initial_damping, Real{0.7});
                    tuned_newton.trust_radius = std::max(Real{1.0}, tuned_newton.trust_radius * Real{0.5});
                    transient_services_.nonlinear_solve->set_options(tuned_newton);
                } else if (recovery.stage == RecoveryStage::StiffProfile) {
                    recovery_reason_code = FallbackReasonCode::StiffnessBackoff;
                    if (options_.enable_bdf_order_control) {
                        bdf_controller_.set_order(
                            std::min(bdf_controller_.current_order(), options_.stiffness_config.max_bdf_order));
                        circuit_.set_integration_order(std::clamp(bdf_controller_.current_order(), 1, 2));
                    } else if (options_.stiffness_config.switch_integrator) {
                        circuit_.set_integration_method(options_.stiffness_config.stiff_integrator);
                        using_stiff_integrator = true;
                    }
                    stiffness_cooldown = std::max(stiffness_cooldown, options_.stiffness_config.cooldown_steps);
                } else if (recovery.stage == RecoveryStage::Regularization) {
                    if (options_.fallback_policy.enable_transient_gmin) {
                        Real next_gmin = transient_gmin_ > 0.0
                            ? transient_gmin_ * options_.fallback_policy.gmin_growth
                            : options_.fallback_policy.gmin_initial;
                        transient_gmin_ = std::min(options_.fallback_policy.gmin_max,
                                                   std::max(next_gmin, options_.fallback_policy.gmin_initial));
                        recovery_reason_code = FallbackReasonCode::TransientGminEscalation;
                        std::ostringstream action;
                        action << "recovery_stage_regularization_gmin=" << transient_gmin_;
                        recovery_action = action.str();
                    } else {
                        recovery_action = "recovery_stage_regularization_disabled";
                    }
                } else if (recovery.stage == RecoveryStage::Abort) {
                    recovery_reason_code = FallbackReasonCode::MaxRetriesExceeded;
                }

                record_fallback_event(result,
                                      result.total_steps,
                                      retries,
                                      t,
                                      dt_used,
                                      recovery_reason_code,
                                      step_result.status,
                                      recovery_action);
                rejection_streak++;
                high_iter_streak = 0;
                if (options_.stiffness_config.enable &&
                    rejection_streak >= options_.stiffness_config.rejection_streak_threshold) {
                    stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                }
                if (options_.enable_bdf_order_control) {
                    (void)bdf_controller_.reduce_on_failure();
                }
                if (recovery.abort) {
                    retries = options_.max_step_retries + 1;
                }
                x = step_anchor_state;
                continue;
            }

            Real lte = -1.0;
            if (variable_step_policy.enabled() && lte_estimator_.has_sufficient_history()) {
                lte = lte_estimator_.compute(step_result.solution,
                                             circuit_.num_nodes(),
                                             circuit_.num_branches());
            }

            if (variable_step_policy.enabled() && lte >= 0.0) {
                const bool lte_guard_window =
                    step_request.event_adjacent || calendar_event_clipped ||
                    post_event_lte_grace_active || split_encountered_this_step;
                if (lte_guard_window) {
                    // LTE estimators are unreliable across switching discontinuities.
                    // In these windows we keep advancing and let event clipping/recovery
                    // control the step, instead of feeding repeated LTE rejections back
                    // into the adaptive controller state.
                    accepted_step_lte_guarded =
                        lte > options_.timestep_config.error_tolerance;
                    const Real guard_growth =
                        (step_request.event_adjacent || calendar_event_clipped) ? Real{1.0} : Real{2.0};
                    const Real dt_recovered = std::max(options_.dt_min, dt_used * guard_growth);
                    dt = clamp_dt_for_mode(std::min(dt_recovered, options_.dt_max));
                } else {
                    const int integration_order =
                        options_.enable_bdf_order_control ? bdf_controller_.current_order() : 2;
                    auto decision = variable_step_policy.evaluate(
                        lte,
                        step_result.iterations,
                        integration_order,
                        dt_used);

                    if (!decision.accepted) {
                        const bool lte_at_min_floor =
                            decision.at_minimum || decision.dt_new <= options_.dt_min * Real{1.001};
                        if (lte_at_min_floor) {
                            // Avoid deadlock loops when LTE remains high at dt_min.
                            // Accept the step and let event/stiffness policies move
                            // the trajectory forward instead of exhausting retries.
                            accepted_step_lte_guarded = true;
                            const Real recovered_dt =
                                std::max(options_.dt_min, dt_used * Real{1.25});
                            dt = clamp_dt_for_mode(std::min(recovered_dt, options_.dt_max));
                            record_fallback_event(result,
                                                  result.total_steps,
                                                  retries,
                                                  t,
                                                  dt_used,
                                                  FallbackReasonCode::LTERejection,
                                                  step_result.status,
                                                  "lte_min_floor_accept");
                        } else {
                        circuit_.clear_stage_context();
                        dt = clamp_dt_for_mode(decision.dt_new);
                        RecoveryDecision lte_recovery;
                        lte_recovery.stage = RecoveryStage::DtBackoff;
                        lte_recovery.next_dt = decision.dt_new;
                        lte_recovery.abort = false;
                        lte_recovery.reason = "lte_rejection";
                        transient_services_.telemetry_collector->on_step_reject(lte_recovery);
                        result.timestep_rejections++;
                        retries++;
                        if (!consume_fixed_recovery_budget("fixed_recovery_budget_lte")) {
                            continue;
                        }
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
                    }

                    dt = clamp_dt_for_mode(decision.dt_new);
                }
            }

            // Event-aligned step splitting for hard switching edges
            if (options_.enable_events && dt_used > options_.dt_min * 1.01) {
                std::optional<Real> earliest_event_time;
                for (const auto& sw : switch_monitors_) {
                    Real v_now = (sw.ctrl >= 0) ? step_result.solution[sw.ctrl] : 0.0;
                    bool now_on = v_now > sw.v_threshold;
                    if (now_on == sw.was_on) {
                        continue;
                    }

                    Real t_event = t + dt_used;
                    Vector x_event = step_result.solution;
                    if (!find_switch_event_time(sw, t, t + dt_used, x, t_event, x_event)) {
                        continue;
                    }
                    if (!earliest_event_time.has_value() || t_event < *earliest_event_time) {
                        earliest_event_time = t_event;
                    }
                }

                bool split_for_event = false;
                if (earliest_event_time.has_value()) {
                    TransientStepRequest event_request = step_request;
                    event_request.event_adjacent = true;
                    event_request.t_target = *earliest_event_time;
                    event_request.threshold_crossing_time = *earliest_event_time;
                    const Real t_segment =
                        transient_services_.event_scheduler->next_segment_target(
                            event_request,
                            t + dt_used);
                    Real dt_event = t_segment - t;
                    const Real min_event_split_dt = min_event_substep_dt(dt_used);
                    if (dt_event > min_event_split_dt && dt_event < dt_used * 0.999) {
                        if (fixed_step_policy.enabled() &&
                            !fixed_step_policy.can_take_internal_substep()) {
                            record_fallback_event(result,
                                                  result.total_steps,
                                                  retries,
                                                  t,
                                                  dt_used,
                                                  FallbackReasonCode::MaxRetriesExceeded,
                                                  step_result.status,
                                                  "fixed_substep_budget_reached_skip_split");
                        } else {
                            dt = clamp_dt_for_mode(std::max(options_.dt_min, dt_event));
                            RecoveryDecision split_recovery;
                            split_recovery.stage = RecoveryStage::DtBackoff;
                            split_recovery.next_dt = dt;
                            split_recovery.abort = false;
                            split_recovery.reason = "event_split";
                            transient_services_.telemetry_collector->on_step_reject(split_recovery);
                            discontinuity_adjacent = true;
                            if (options_.stiffness_config.enable && stiffness_cooldown <= 0) {
                                stiffness_cooldown = std::max(1, options_.stiffness_config.cooldown_steps);
                            }
                            record_fallback_event(result,
                                                  result.total_steps,
                                                  retries,
                                                  t,
                                                  dt_used,
                                                  FallbackReasonCode::EventSplit,
                                                  step_result.status,
                                                  "split_to_earliest_event");
                            lte_discontinuity_grace_steps =
                                std::max(lte_discontinuity_grace_steps, kLteEventGraceSteps);
                            split_encountered_this_step = true;
                            split_for_event = true;
                        }
                    }
                }

                if (split_for_event) {
                    circuit_.clear_stage_context();
                    continue;
                }
            }

            accepted_step_event_adjacent =
                step_request.event_adjacent || calendar_event_clipped || split_encountered_this_step;
            accepted = true;
        }

        if (!accepted) {
            const bool at_dt_min_floor = dt_used <= options_.dt_min * Real{1.001};
            if (variable_step_policy.enabled() && at_dt_min_floor &&
                dt_min_hold_advances < kMaxDtMinHoldAdvances) {
                // Last-resort progress safeguard for hard switching points where
                // Newton repeatedly fails at dt_min. Advance with state hold to
                // cross the pathological instant instead of aborting.
                dt_min_hold_advances++;
                circuit_.clear_stage_context();
                record_fallback_event(result,
                                      result.total_steps,
                                      retries,
                                      t,
                                      dt_used,
                                      FallbackReasonCode::MaxRetriesExceeded,
                                      step_result.status,
                                      "dt_min_hold_advance");

                rejection_streak = 0;
                high_iter_streak = 0;
                global_recovery_attempts = 0;
                transient_gmin_ = 0.0;
                transient_services_.nonlinear_solve->set_options(baseline_newton_options);

                accumulate_conduction_losses(step_anchor_state, dt_used);
                update_thermal_state(dt_used);

                t += dt_used;
                variable_step_policy.on_step_accepted(dt_used);
                x = step_anchor_state;
                circuit_.update_history(x);
                lte_estimator_.record_solution(x, t, dt_used);

                result.time.push_back(t);
                result.states.push_back(x);
                append_virtual_sample(x, t);
                if (callback) {
                    callback(t, x);
                }
                result.total_steps++;

                continue;
            }

            RecoveryDecision abort_recovery;
            abort_recovery.stage = RecoveryStage::Abort;
            abort_recovery.next_dt = dt_used;
            abort_recovery.abort = true;
            abort_recovery.reason = "max_retries_exceeded";
            transient_services_.telemetry_collector->on_step_reject(abort_recovery);
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
                dt = clamp_dt_for_mode(std::max(options_.dt_min, dt_used * (
                    global_recovery_attempts == 1 ? Real{0.25} : Real{0.1})));
                rejection_streak = 0;
                high_iter_streak = 0;
                stiffness_cooldown = std::max(stiffness_cooldown, options_.stiffness_config.cooldown_steps);

                NewtonOptions tuned_newton = transient_services_.nonlinear_solve->options();
                apply_robust_newton_defaults(tuned_newton);
                tuned_newton.max_iterations = std::max(tuned_newton.max_iterations, 200);
                transient_services_.nonlinear_solve->set_options(tuned_newton);

                LinearSolverStackConfig tuned_linear = options_.linear_solver;
                apply_robust_linear_solver_defaults(tuned_linear);
                transient_services_.linear_solve->solver().set_config(tuned_linear);

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
            result.diagnostic = SimulationDiagnosticCode::TransientStepFailure;
            result.message = "Transient failed at t=" + std::to_string(t + dt_used) +
                             ": " + step_result.error_message;
            if (auto_recovery_attempted) {
                result.message += " (automatic regularization attempted)";
            }
            result.backend_telemetry.failure_reason =
                std::string(diagnostic_code_to_reason(result.diagnostic));
            break;
        }

        result.newton_iterations_total += step_result.iterations;
        transient_services_.telemetry_collector->on_step_accept(t + dt_used, step_result.solution);
        rejection_streak = 0;
        dt_min_hold_advances = 0;
        global_recovery_attempts = 0;
        transient_gmin_ = 0.0;
        transient_services_.nonlinear_solve->set_options(baseline_newton_options);

        if (variable_step_policy.enabled()) {
            if (accepted_step_event_adjacent || accepted_step_lte_guarded) {
                lte_discontinuity_grace_steps =
                    std::max(lte_discontinuity_grace_steps, kLteEventGraceSteps);
            } else if (lte_discontinuity_grace_steps > 0) {
                lte_discontinuity_grace_steps--;
            }
        }

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
                auto telemetry = transient_services_.linear_solve->solver().telemetry();
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
        variable_step_policy.on_step_accepted(dt_used);
        bool emit_sample = true;
        if (fixed_step_policy.enabled()) {
            emit_sample = fixed_step_policy.on_step_accepted(t);
        }
        x = step_result.solution;
        circuit_.update_history(x);

        if (variable_step_policy.enabled()) {
            lte_estimator_.record_solution(x, t, dt_used);

            if (options_.enable_bdf_order_control && lte_estimator_.has_sufficient_history()) {
                Real lte = lte_estimator_.compute(x, circuit_.num_nodes(), circuit_.num_branches());
                auto decision = bdf_controller_.select_order(lte, options_.timestep_config.error_tolerance);
                if (decision.order_changed) {
                    circuit_.set_integration_order(std::clamp(decision.new_order, 1, 2));
                }
            }
        }

        if (emit_sample || nearly_same_time(t, options_.tstop)) {
            result.time.push_back(t);
            result.states.push_back(x);
            append_virtual_sample(x, t);

            if (callback) {
                callback(t, x);
            }
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
    } else if (result.backend_telemetry.failure_reason.empty() &&
               result.diagnostic != SimulationDiagnosticCode::None) {
        result.backend_telemetry.failure_reason =
            std::string(diagnostic_code_to_reason(result.diagnostic));
    }

    finalize_loss_summary(result);
    finalize_thermal_summary(result);

    result.linear_solver_telemetry = transient_services_.linear_solve->solver().telemetry();
    const EquationAssemblerTelemetry assembler_telemetry =
        transient_services_.equation_assembler->telemetry();
    auto saturating_int = [](std::uint64_t value) {
        constexpr std::uint64_t max_int = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        return static_cast<int>(std::min(value, max_int));
    };
    result.backend_telemetry.equation_assemble_system_calls =
        saturating_int(assembler_telemetry.system_calls + direct_assemble_system_calls_);
    result.backend_telemetry.equation_assemble_residual_calls =
        saturating_int(assembler_telemetry.residual_calls + direct_assemble_residual_calls_);
    result.backend_telemetry.equation_assemble_system_time_seconds =
        assembler_telemetry.system_time_seconds + direct_assemble_system_time_seconds_;
    result.backend_telemetry.equation_assemble_residual_time_seconds =
        assembler_telemetry.residual_time_seconds + direct_assemble_residual_time_seconds_;

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

}  // namespace pulsim::v1
