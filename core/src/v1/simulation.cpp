#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <string_view>

namespace pulsim::v1 {

namespace {
constexpr int kMaxGlobalRecoveryAttempts = 2;
constexpr int kLteEventGraceSteps = 2;
constexpr int kMaxDtMinHoldAdvances = 128;
constexpr std::size_t kMaxSampleReserve = 1'000'000;
constexpr std::size_t kMaxFallbackReserve = 2'000'000;

[[nodiscard]] int saturating_int(std::uint64_t value) {
    constexpr std::uint64_t max_int = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    return static_cast<int>(std::min(value, max_int));
}

[[nodiscard]] std::size_t estimate_output_sample_reserve(const SimulationOptions& options) {
    if (!std::isfinite(options.tstart) || !std::isfinite(options.tstop) || options.tstop <= options.tstart) {
        return 0;
    }

    Real dt_nominal = std::max(options.dt, options.dt_min);
    if (!std::isfinite(dt_nominal) || dt_nominal <= 0.0) {
        return 0;
    }

    const long double span = static_cast<long double>(options.tstop - options.tstart);
    const long double step = static_cast<long double>(dt_nominal);
    if (!(span > 0.0L && step > 0.0L)) {
        return 0;
    }

    const long double estimate = std::ceil(span / step) + 2.0L;
    if (!(estimate > 0.0L)) {
        return 0;
    }

    const long double capped = std::min(estimate, static_cast<long double>(kMaxSampleReserve));
    return static_cast<std::size_t>(capped);
}

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

[[nodiscard]] std::string_view diagnostic_code_to_reason(SimulationDiagnosticCode code) {
    switch (code) {
        case SimulationDiagnosticCode::None:
            return "";
        case SimulationDiagnosticCode::DcOperatingPointFailure:
            return "dc_operating_point_failure";
        case SimulationDiagnosticCode::InvalidInitialState:
            return "invalid_initial_state";
        case SimulationDiagnosticCode::InvalidTimeWindow:
            return "invalid_time_window";
        case SimulationDiagnosticCode::InvalidTimestep:
            return "invalid_timestep";
        case SimulationDiagnosticCode::InvalidThermalConfiguration:
            return "invalid_thermal_configuration";
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

    if (!options.thermal.enable) {
        return std::nullopt;
    }

    if (!std::isfinite(options.thermal.ambient)) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidThermalConfiguration,
            "Invalid thermal configuration: simulation.thermal.ambient must be finite"
        };
    }
    if (!std::isfinite(options.thermal.default_rth) || options.thermal.default_rth <= 0.0) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidThermalConfiguration,
            "Invalid thermal configuration: simulation.thermal.default_rth must be finite and > 0"
        };
    }
    if (!std::isfinite(options.thermal.default_cth) || options.thermal.default_cth < 0.0) {
        return TransientInputIssue{
            SimulationDiagnosticCode::InvalidThermalConfiguration,
            "Invalid thermal configuration: simulation.thermal.default_cth must be finite and >= 0"
        };
    }

    const auto& devices = circuit.devices();
    const auto& conns = circuit.connections();
    for (std::size_t i = 0; i < devices.size() && i < conns.size(); ++i) {
        bool supports_thermal = false;
        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            supports_thermal = device_traits<T>::has_thermal_model;
        }, devices[i]);

        if (!supports_thermal) {
            continue;
        }

        ThermalDeviceConfig cfg;
        cfg.enabled = true;
        cfg.rth = options.thermal.default_rth;
        cfg.cth = options.thermal.default_cth;
        cfg.temp_init = options.thermal.ambient;
        cfg.temp_ref = options.thermal.ambient;

        const auto it = options.thermal_devices.find(conns[i].name);
        if (it != options.thermal_devices.end()) {
            cfg = it->second;
        }

        if (!cfg.enabled) {
            continue;
        }

        auto fail = [&](const std::string& field, const std::string& rule) {
            return TransientInputIssue{
                SimulationDiagnosticCode::InvalidThermalConfiguration,
                "Invalid thermal configuration for component '" + conns[i].name +
                    "': " + field + " " + rule
            };
        };

        if (!std::isfinite(cfg.rth)) {
            return fail("rth", "must be finite");
        }
        if (cfg.rth <= 0.0) {
            return fail("rth", "must be > 0");
        }
        if (!std::isfinite(cfg.cth)) {
            return fail("cth", "must be finite");
        }
        if (cfg.cth < 0.0) {
            return fail("cth", "must be >= 0");
        }
        if (!std::isfinite(cfg.temp_init)) {
            return fail("temp_init", "must be finite");
        }
        if (!std::isfinite(cfg.temp_ref)) {
            return fail("temp_ref", "must be finite");
        }
        if (!std::isfinite(cfg.alpha)) {
            return fail("alpha", "must be finite");
        }
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
        recovery_retries_in_current_macro_ = 0;
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
    // Push the resolved switching-mode default into the circuit so the
    // segment-model and event-scan services (Phases 2/4) consume the user's
    // simulation-level intent (refactor-pwl-switching-engine, Phase 5).
    circuit_.set_default_switching_mode(options_.switching_mode);
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

LinearSystem Simulator::linearize_around(const Vector& x_op, Real t_op) {
    // Phase 1 of `add-frequency-domain-analysis`: lift the segment engine's
    // PWL state-space `M·dx/dt + N·x = b(t)` into a generic descriptor
    // form `E·dx/dt = A·x + B·u, y = C·x + D·u`.
    //
    // Today only PWL-admissible circuits are linearized — we read M and N
    // straight off `Circuit::assemble_state_space`, which is exactly what
    // the segment engine does each step. Behavioral devices fall out via
    // an explicit `failure_reason`; AD-driven Behavioral linearization is
    // a Phase 1.2 follow-up.
    LinearSystem result;
    result.t_linearization = t_op;
    result.x_linearization = x_op;

    const Index n = circuit_.system_size();
    if (x_op.size() != n) {
        result.failure_reason = "linearize_state_size_mismatch";
        return result;
    }

    // PWL admissibility — same gate the segment engine applies. If any
    // device is in Behavioral mode (or unsupported by the PWL state-space
    // assembler), bail out with a typed reason. Phase 1.2 will replace
    // this with an AD/finite-difference path.
    const auto default_mode = circuit_.default_switching_mode();
    if (!circuit_.all_switching_devices_in_ideal_mode(default_mode)) {
        result.method = "non_admissible";
        result.failure_reason = "linearize_non_admissible_behavioral_device";
        return result;
    }
    if (!circuit_.pwl_state_space_supports_all_devices()) {
        result.method = "non_admissible";
        result.failure_reason = "linearize_non_admissible_unsupported_device";
        return result;
    }

    // Assemble M, N, b at the operating-point time. M and N are
    // time-invariant within a PWL topology, so `t_op` only affects b.
    SparseMatrix M(n, n);
    SparseMatrix N(n, n);
    Vector b(n);
    circuit_.assemble_state_space(M, N, b, t_op);

    if (M.rows() != n || N.rows() != n || b.size() != n) {
        result.failure_reason = "linearize_state_space_dim_mismatch";
        return result;
    }
    if (!b.allFinite()) {
        result.failure_reason = "linearize_source_non_finite";
        return result;
    }

    // E = M, A = -N. Eigen sparse expressions evaluate into compressed
    // form via `.eval()` so the returned matrices own their storage.
    result.E = M;
    result.A = (-N).eval();
    result.E.makeCompressed();
    result.A.makeCompressed();

    // B is a single column = b(t_op), the lumped-source contribution at
    // the linearization time. AC sweep treats this as a unit perturbation
    // amplitude on the lumped input. Per-source B columns are Phase 4 of
    // the change (multi-input transfer-function matrix).
    SparseMatrix B(n, 1);
    {
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) {
            if (b[i] != Real{0}) {
                triplets.emplace_back(i, 0, b[i]);
            }
        }
        B.setFromTriplets(triplets.begin(), triplets.end());
        B.makeCompressed();
    }
    result.B = std::move(B);

    // C = identity (output = full state). User-selected measurement nodes
    // (a smaller C with one row per requested node) is a Phase 2 / 4
    // refinement; for now downstream consumers can index into the full x
    // by node name via `Circuit::get_node`.
    SparseMatrix C(n, n);
    {
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, Real{1});
        }
        C.setFromTriplets(triplets.begin(), triplets.end());
        C.makeCompressed();
    }
    result.C = std::move(C);

    // D = 0 (no direct feedthrough). MNA sources contribute to the
    // dynamics via b(t), not to the measurement output.
    SparseMatrix D(n, 1);
    D.makeCompressed();
    result.D = std::move(D);

    result.state_size  = n;
    result.input_size  = 1;
    result.output_size = n;
    result.method      = "piecewise_linear_segment";
    return result;
}

namespace {

// Build the B column for a named perturbation source. Returns true on
// success; populates `triplets` with the source's contribution to the
// linearized RHS. Convention matches `Circuit::assemble_state_space`:
//   Voltage sources (V/PWM/Sine/Pulse): b[branch_index] = +V_src
//     → ∂b/∂V = δ(row = branch_index)
//   Current sources: b[npos] += I, b[nneg] -= I
//     → ∂b/∂I = +1 at npos, -1 at nneg
[[nodiscard]] bool build_perturbation_b_column(
    const Circuit& circuit,
    const std::string& source_name,
    Index state_size,
    std::vector<Eigen::Triplet<Real>>& triplets,
    std::string& failure_reason) {
    const auto* conn = circuit.find_connection(source_name);
    if (conn == nullptr) {
        failure_reason = "ac_sweep_perturbation_source_not_found:" + source_name;
        return false;
    }

    if (circuit.find_device<VoltageSource>(source_name) ||
        circuit.find_device<PWMVoltageSource>(source_name) ||
        circuit.find_device<SineVoltageSource>(source_name) ||
        circuit.find_device<PulseVoltageSource>(source_name)) {
        if (conn->branch_index < 0 || conn->branch_index >= state_size) {
            failure_reason = "ac_sweep_perturbation_voltage_source_no_branch_index";
            return false;
        }
        triplets.emplace_back(conn->branch_index, 0, Real{1});
        return true;
    }

    if (circuit.find_device<CurrentSource>(source_name)) {
        if (conn->nodes.size() < 2) {
            failure_reason = "ac_sweep_perturbation_current_source_missing_nodes";
            return false;
        }
        const Index npos = conn->nodes[0];
        const Index nneg = conn->nodes[1];
        if (npos >= 0 && npos < state_size) {
            triplets.emplace_back(npos, 0, Real{ 1});
        }
        if (nneg >= 0 && nneg < state_size) {
            triplets.emplace_back(nneg, 0, Real{-1});
        }
        return true;
    }

    failure_reason = "ac_sweep_perturbation_source_unsupported_type:" + source_name;
    return false;
}

[[nodiscard]] std::vector<Real> generate_frequency_grid(
    const AcSweepOptions& opt) {
    std::vector<Real> freqs;
    if (!(opt.f_start > 0.0) || !(opt.f_stop >= opt.f_start)) {
        return freqs;
    }
    if (opt.scale == AcSweepScale::Logarithmic) {
        const Real decades = std::log10(opt.f_stop / opt.f_start);
        const int n = std::max(2, static_cast<int>(
            std::round(decades * std::max(1, opt.points_per_decade))) + 1);
        freqs.reserve(static_cast<std::size_t>(n));
        const Real log_start = std::log10(opt.f_start);
        const Real log_stop  = std::log10(opt.f_stop);
        for (int k = 0; k < n; ++k) {
            const Real t = (n == 1) ? Real{0}
                                     : static_cast<Real>(k) / static_cast<Real>(n - 1);
            freqs.push_back(std::pow(Real{10}, log_start + t * (log_stop - log_start)));
        }
    } else {
        const int n = std::max(2, opt.num_points > 0
                                      ? opt.num_points
                                      : static_cast<int>(opt.points_per_decade));
        freqs.reserve(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            const Real t = (n == 1) ? Real{0}
                                     : static_cast<Real>(k) / static_cast<Real>(n - 1);
            freqs.push_back(opt.f_start + t * (opt.f_stop - opt.f_start));
        }
    }
    return freqs;
}

}  // namespace

AcSweepResult Simulator::run_ac_sweep(const AcSweepOptions& options) {
    AcSweepResult result;
    const auto t_wall_start = std::chrono::steady_clock::now();

    auto fail = [&](std::string reason) {
        result.success = false;
        result.failure_reason = std::move(reason);
        result.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_wall_start).count();
        return result;
    };

    // 1) Resolve the operating point: either run DC OP or trust a caller-
    //    supplied `x_op`. For Phase 2 we don't perturb-then-resolve — the
    //    DC OP is taken as-is and the linearization happens at that
    //    state.
    Vector x_op;
    if (options.use_dc_op) {
        const auto dc = dc_operating_point();
        if (!dc.success) {
            return fail("ac_sweep_dc_op_failed");
        }
        x_op = dc.newton_result.solution;
    } else {
        x_op = options.x_op;
    }

    const Index n = circuit_.system_size();
    if (x_op.size() != n) {
        return fail("ac_sweep_state_size_mismatch");
    }

    // 2) PWL admissibility check, mirroring `linearize_around`. Behavioral
    //    devices fall out with a typed reason; AD-driven Behavioral AC
    //    sweep is a Phase 1.2 follow-up.
    const auto default_mode = circuit_.default_switching_mode();
    if (!circuit_.all_switching_devices_in_ideal_mode(default_mode)) {
        return fail("ac_sweep_non_admissible_behavioral_device");
    }
    if (!circuit_.pwl_state_space_supports_all_devices()) {
        return fail("ac_sweep_non_admissible_unsupported_device");
    }

    // 3) Assemble M, N at t_op. We discard b — Phase 2 builds B from the
    //    named perturbation source, not from the DC b vector.
    SparseMatrix M(n, n), N(n, n);
    Vector b_unused(n);
    circuit_.assemble_state_space(M, N, b_unused, options.t_op);
    M.makeCompressed();
    N.makeCompressed();

    // 4) Resolve perturbation source list. Phase 4 of the change supports
    //    a vector of source names; Phases 2/3 fall back to the single
    //    `perturbation_source`. Either way we end up with N_inputs ≥ 1.
    std::vector<std::string> source_names;
    if (!options.perturbation_sources.empty()) {
        source_names = options.perturbation_sources;
    } else {
        source_names = {options.perturbation_source};
    }
    const int n_inputs = static_cast<int>(source_names.size());

    // Build B as a multi-column matrix: column k carries the perturbation
    // contribution of source k. For Phase-2 single-source sweeps this
    // collapses to a 1-column matrix identical to the prior behavior.
    SparseMatrix B(n, n_inputs);
    {
        std::vector<Eigen::Triplet<Real>> b_triplets;
        for (int k = 0; k < n_inputs; ++k) {
            std::vector<Eigen::Triplet<Real>> col_triplets;
            if (!build_perturbation_b_column(circuit_, source_names[k],
                                             n, col_triplets, result.failure_reason)) {
                return fail(result.failure_reason);
            }
            for (const auto& t : col_triplets) {
                // build_perturbation_b_column emits col=0 by convention;
                // remap to the requested column k of the multi-input B.
                b_triplets.emplace_back(t.row(), k, t.value());
            }
        }
        B.setFromTriplets(b_triplets.begin(), b_triplets.end());
        B.makeCompressed();
    }

    // 5) Resolve measurement nodes to state indices. Empty list → return
    //    the full state. Each output node combines with each input source
    //    to produce one `AcMeasurement` (the H[i,j] cell of the matrix).
    struct Slot { std::string node; Index state_index; };
    std::vector<Slot> output_slots;
    if (options.measurement_nodes.empty()) {
        output_slots.reserve(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) {
            output_slots.push_back({"", i});
        }
    } else {
        output_slots.reserve(options.measurement_nodes.size());
        for (const auto& name : options.measurement_nodes) {
            const Index idx = circuit_.get_node(name);
            if (idx < 0 || idx >= n) {
                return fail("ac_sweep_measurement_node_not_found:" + name);
            }
            output_slots.push_back({name, idx});
        }
    }
    std::vector<AcMeasurement> measurements;
    measurements.reserve(output_slots.size() * static_cast<std::size_t>(n_inputs));
    for (const auto& slot : output_slots) {
        for (int k = 0; k < n_inputs; ++k) {
            AcMeasurement m;
            m.node = slot.node;
            m.state_index = slot.state_index;
            // For Phase-2 single-source sweeps `source_names[0]` is the
            // user's `perturbation_source` (could be empty if they
            // provided neither); leave it on the measurement for symmetry.
            m.perturbation_source = source_names[k];
            measurements.push_back(std::move(m));
        }
    }

    // 6) Generate frequency grid.
    result.frequencies = generate_frequency_grid(options);
    if (result.frequencies.empty()) {
        return fail("ac_sweep_invalid_frequency_range");
    }
    for (auto& m : measurements) {
        m.magnitude_db.reserve(result.frequencies.size());
        m.phase_deg.reserve(result.frequencies.size());
        m.real_part.reserve(result.frequencies.size());
        m.imag_part.reserve(result.frequencies.size());
    }

    // 7) Per-frequency complex solve. K(ω) = jω·E - A where E = M, A = -N
    //    in our convention, so K(ω) = jω·M + N. Sparsity pattern is the
    //    union of M and N's patterns and is constant across ω, so we
    //    `analyzePattern` once and `factorize` per ω.
    using ComplexScalar = std::complex<Real>;
    using ComplexSparse = Eigen::SparseMatrix<ComplexScalar>;
    using ComplexVector = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;

    // Build the union sparsity pattern by overlaying M and N's complex casts.
    ComplexSparse K(n, n);
    {
        std::vector<Eigen::Triplet<ComplexScalar>> tri;
        tri.reserve(static_cast<std::size_t>(M.nonZeros() + N.nonZeros()));
        for (Index col = 0; col < M.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(M, col); it; ++it) {
                tri.emplace_back(it.row(), it.col(), ComplexScalar{it.value(), Real{0}});
            }
        }
        for (Index col = 0; col < N.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(N, col); it; ++it) {
                tri.emplace_back(it.row(), it.col(), ComplexScalar{it.value(), Real{0}});
            }
        }
        K.setFromTriplets(tri.begin(), tri.end());
        K.makeCompressed();
    }

    // Build the multi-input RHS matrix from B (densified for the solve).
    using ComplexMatrix = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>;
    ComplexMatrix rhs = ComplexMatrix::Zero(n, n_inputs);
    for (Index col = 0; col < B.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(B, col); it; ++it) {
            rhs(it.row(), it.col()) = ComplexScalar{it.value(), Real{0}};
        }
    }

    Eigen::SparseLU<ComplexSparse> solver;
    solver.analyzePattern(K);  // once — pattern is constant across ω

    for (const Real f : result.frequencies) {
        const Real omega = Real{2} * std::numbers::pi_v<Real> * f;
        // K(ω) = jω·M + N, computed as a fresh complex matrix.
        ComplexSparse Kf(n, n);
        {
            std::vector<Eigen::Triplet<ComplexScalar>> tri;
            tri.reserve(static_cast<std::size_t>(M.nonZeros() + N.nonZeros()));
            for (Index col = 0; col < M.outerSize(); ++col) {
                for (SparseMatrix::InnerIterator it(M, col); it; ++it) {
                    tri.emplace_back(it.row(), it.col(),
                                     ComplexScalar{Real{0}, omega * it.value()});
                }
            }
            for (Index col = 0; col < N.outerSize(); ++col) {
                for (SparseMatrix::InnerIterator it(N, col); it; ++it) {
                    tri.emplace_back(it.row(), it.col(),
                                     ComplexScalar{it.value(), Real{0}});
                }
            }
            Kf.setFromTriplets(tri.begin(), tri.end());
            Kf.makeCompressed();
        }

        solver.factorize(Kf);
        result.total_factorizations += 1;
        if (solver.info() != Eigen::Success) {
            return fail("ac_sweep_factorization_failed_at_f:" + std::to_string(f));
        }
        // Multi-input solve: X has one column per input source.
        const ComplexMatrix x = solver.solve(rhs);
        result.total_solves += 1;
        if (solver.info() != Eigen::Success) {
            return fail("ac_sweep_solve_failed_at_f:" + std::to_string(f));
        }

        // Walk measurements in (output_slot, input_source) order — same
        // order they were created above.
        std::size_t mi = 0;
        for (const auto& slot : output_slots) {
            for (int k = 0; k < n_inputs; ++k) {
                auto& m = measurements[mi++];
                const ComplexScalar y = x(slot.state_index, k);
                m.real_part.push_back(y.real());
                m.imag_part.push_back(y.imag());
                const Real mag = std::abs(y);
                // Clamp tiny magnitudes so log10(0) doesn't produce -inf
                // in the returned data — a node with strictly zero
                // transfer at some frequency reports `-300 dB` rather
                // than `-inf`.
                const Real mag_db = (mag > Real{1e-300})
                                        ? Real{20} * std::log10(mag)
                                        : Real{-300};
                m.magnitude_db.push_back(mag_db);
                m.phase_deg.push_back(
                    std::arg(y) * Real{180} / std::numbers::pi_v<Real>);
            }
        }
    }

    result.measurements = std::move(measurements);
    result.success = true;
    result.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_wall_start).count();
    return result;
}

namespace {

// Goertzel single-bin DFT: for samples y[k] (k = 0 .. N-1) sampled
// uniformly at dt, return the complex Fourier coefficient at frequency
// `f_target` Hz. Output is normalized so that a pure cosine of amplitude
// A at exactly the target bin returns approximately A (real part).
//
// Used by FRA to extract the fundamental at the perturbation frequency
// from a transient-captured time series.
[[nodiscard]] std::complex<Real> goertzel_dft(const std::vector<Real>& samples,
                                               Real dt,
                                               Real f_target) {
    if (samples.empty() || dt <= Real{0} || f_target <= Real{0}) {
        return {Real{0}, Real{0}};
    }
    const Real omega = Real{2} * std::numbers::pi_v<Real> * f_target * dt;
    const Real cos_omega = std::cos(omega);
    const Real sin_omega = std::sin(omega);
    const Real coeff = Real{2} * cos_omega;

    Real q1 = Real{0};
    Real q2 = Real{0};
    for (const Real s : samples) {
        const Real q0 = s + coeff * q1 - q2;
        q2 = q1;
        q1 = q0;
    }
    // Real / imag parts of the Goertzel result, scaled to match a unit-
    // amplitude cosine input (factor 2 / N).
    const Real real_part = q1 - q2 * cos_omega;
    const Real imag_part = q2 * sin_omega;
    const Real scale = Real{2} / static_cast<Real>(samples.size());
    return {real_part * scale, imag_part * scale};
}

}  // namespace

FraResult Simulator::run_fra(const FraOptions& options) {
    FraResult result;
    const auto t_wall_start = std::chrono::steady_clock::now();

    auto fail = [&](std::string reason) {
        result.success = false;
        result.failure_reason = std::move(reason);
        result.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_wall_start).count();
        return result;
    };

    // PWL admissibility — same gate as AC sweep. Behavioral devices fall
    // out with a typed reason; FRA on Behavioral converters is the next
    // milestone (Phase 3.6 idea: relax PWL gate, accept Newton-DAE).
    const auto default_mode = circuit_.default_switching_mode();
    if (!circuit_.all_switching_devices_in_ideal_mode(default_mode)) {
        return fail("fra_non_admissible_behavioral_device");
    }

    // Validate the perturbation source by reusing the AC sweep helper —
    // throws away the resulting B but confirms the source resolves.
    {
        std::vector<Eigen::Triplet<Real>> dummy_triplets;
        std::string reason_buffer;
        if (!build_perturbation_b_column(circuit_, options.perturbation_source,
                                         circuit_.system_size(),
                                         dummy_triplets, reason_buffer)) {
            return fail(reason_buffer);
        }
    }

    // Resolve measurement node indices.
    std::vector<Index> measurement_indices;
    measurement_indices.reserve(options.measurement_nodes.size());
    for (const auto& name : options.measurement_nodes) {
        const Index idx = circuit_.get_node(name);
        if (idx < 0) {
            return fail("fra_measurement_node_not_found:" + name);
        }
        measurement_indices.push_back(idx);
        FraMeasurement m;
        m.node = name;
        m.state_index = idx;
        result.measurements.push_back(std::move(m));
    }
    if (measurement_indices.empty()) {
        return fail("fra_no_measurement_nodes");
    }

    // Frequency grid (reuse the AC sweep generator semantics).
    {
        AcSweepOptions ac_for_grid;
        ac_for_grid.f_start = options.f_start;
        ac_for_grid.f_stop  = options.f_stop;
        ac_for_grid.points_per_decade = options.points_per_decade;
        ac_for_grid.scale = options.scale;
        result.frequencies = generate_frequency_grid(ac_for_grid);
    }
    if (result.frequencies.empty()) {
        return fail("fra_invalid_frequency_range");
    }
    for (auto& m : result.measurements) {
        m.magnitude_db.reserve(result.frequencies.size());
        m.phase_deg.reserve(result.frequencies.size());
        m.real_part.reserve(result.frequencies.size());
        m.imag_part.reserve(result.frequencies.size());
    }

    // Take one DC operating-point snapshot up-front; reuse it as the
    // initial state for every per-frequency transient. The simulator
    // mutates `options_` so we save / restore around the FRA loop.
    const auto baseline_options = options_;
    SimulationOptions fra_opts = options_;
    fra_opts.adaptive_timestep = false;
    fra_opts.enable_bdf_order_control = false;
    fra_opts.dt_min = 1e-15;
    // Trapezoidal preserves phase exactly along the unit circle (bilinear
    // transform; only frequency-warping at higher ω·dt). FRA needs the
    // tightest phase agreement vs AC sweep — the spec gate is ≤ 5° — so
    // pin it here regardless of what the user set on the Simulator.
    fra_opts.integrator = Integrator::Trapezoidal;

    const auto dc = dc_operating_point();
    if (!dc.success) {
        options_ = baseline_options;
        return fail("fra_dc_op_failed");
    }
    const Vector x_dc = dc.newton_result.solution;

    // Per-frequency loop.
    for (const Real f : result.frequencies) {
        const Real period = Real{1} / f;
        const Real dt = period /
                        static_cast<Real>(std::max(4, options.samples_per_cycle));
        const int total_samples = std::max(1, options.n_cycles) *
                                  std::max(4, options.samples_per_cycle);
        const int discard_samples = std::clamp(options.discard_cycles, 0,
                                               options.n_cycles - 1) *
                                    std::max(4, options.samples_per_cycle);

        fra_opts.tstart = 0.0;
        fra_opts.tstop  = static_cast<Real>(total_samples) * dt;
        fra_opts.dt     = dt;
        fra_opts.dt_max = dt;
        options_        = fra_opts;

        // Configure perturbation, then capture the measurement-node time
        // series via the simulation callback.
        circuit_.set_ac_perturbation(options.perturbation_source,
                                     options.perturbation_amplitude,
                                     f, options.perturbation_phase);

        std::vector<std::vector<Real>> per_node_samples(
            measurement_indices.size());
        for (auto& v : per_node_samples) {
            v.reserve(static_cast<std::size_t>(total_samples + 1));
        }
        int captured_steps = 0;
        SimulationCallback cb = [&](Real /*t*/, const Vector& x) {
            for (std::size_t k = 0; k < measurement_indices.size(); ++k) {
                per_node_samples[k].push_back(x[measurement_indices[k]]);
            }
            captured_steps += 1;
            return true;
        };

        const auto run = run_transient(x_dc, cb);
        circuit_.clear_ac_perturbation();
        if (!run.success) {
            options_ = baseline_options;
            return fail("fra_transient_failed_at_f:" + std::to_string(f));
        }
        result.total_transient_steps += run.total_steps;

        // Discard the warmup window, then Goertzel each measurement node.
        for (std::size_t k = 0; k < measurement_indices.size(); ++k) {
            const auto& full = per_node_samples[k];
            const std::size_t start = std::min<std::size_t>(
                full.size(), static_cast<std::size_t>(discard_samples));
            std::vector<Real> window(full.begin() + start, full.end());
            // Subtract mean to remove the DC operating-point offset, so the
            // Goertzel result reflects only the small-signal response.
            if (!window.empty()) {
                Real mean = Real{0};
                for (const Real s : window) mean += s;
                mean /= static_cast<Real>(window.size());
                for (auto& s : window) s -= mean;
            }

            // Sample-time correction: the SimulationCallback fires after
            // each accepted step, so `samples[k]` is observed at
            // `t = (k + 1)·dt`. Goertzel assumes index 0 is at t=0 and
            // would otherwise return a phasor rotated by `+ω·dt`. The
            // simulator's trapezoidal scheme also evaluates the
            // perturbation at the trapezoidal midpoint t = (k + 0.5)·dt,
            // so the EFFECTIVE input lags the captured output by `ω·dt/2`,
            // not the full `ω·dt`. The two effects partially cancel: the
            // residual offset is `+ω·dt/2`, which we compensate by
            // multiplying the Goertzel result by `e^{-j·ω·dt/2}`.
            const std::complex<Real> coef_raw = goertzel_dft(window, dt, f);
            const Real omega = Real{2} * std::numbers::pi_v<Real> * f;
            const std::complex<Real> phase_correction =
                std::polar(Real{1}, -omega * dt / Real{2});
            const std::complex<Real> coef = coef_raw * phase_correction;

            const std::complex<Real> H =
                coef / std::complex<Real>{options.perturbation_amplitude, Real{0}};

            const Real mag = std::abs(H);
            const Real mag_db = (mag > Real{1e-300})
                                    ? Real{20} * std::log10(mag)
                                    : Real{-300};
            const Real phase_deg =
                std::arg(H) * Real{180} / std::numbers::pi_v<Real>;

            result.measurements[k].real_part.push_back(H.real());
            result.measurements[k].imag_part.push_back(H.imag());
            result.measurements[k].magnitude_db.push_back(mag_db);
            result.measurements[k].phase_deg.push_back(phase_deg);
        }
    }

    options_ = baseline_options;
    result.success = true;
    result.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_wall_start).count();
    return result;
}

[[nodiscard]] std::string_view formulation_mode_to_string(FormulationMode mode) {
    switch (mode) {
        case FormulationMode::ProjectedWrapper:
            return "projected_wrapper";
        case FormulationMode::Direct:
            return "direct";
        default:
            return "projected_wrapper";
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

// =============================================================================
// run_transient_native_impl helpers
// =============================================================================

void Simulator::collect_step_solve_telemetry(SimulationResult& result) {
    if (last_step_segment_attempted_) {
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
            if (last_step_symbolic_factor_cache_hit_) {
                result.backend_telemetry.symbolic_factor_cache_hits += 1;
            }
            if (last_step_linear_factor_cache_invalidation_reason_typed_ !=
                CacheInvalidationReason::None) {
                result.backend_telemetry.linear_factor_cache_invalidations += 1;
                result.backend_telemetry.linear_factor_cache_last_invalidation_reason =
                    last_step_linear_factor_cache_invalidation_reason_;
                result.backend_telemetry.linear_factor_cache_last_invalidation_reason_typed =
                    last_step_linear_factor_cache_invalidation_reason_typed_;
                switch (last_step_linear_factor_cache_invalidation_reason_typed_) {
                    case CacheInvalidationReason::TopologyChanged:
                        result.backend_telemetry
                            .linear_factor_cache_invalidations_topology_changed += 1;
                        break;
                    case CacheInvalidationReason::StampParamChanged:
                        result.backend_telemetry
                            .linear_factor_cache_invalidations_stamp_param_changed += 1;
                        break;
                    case CacheInvalidationReason::GminEscalated:
                        result.backend_telemetry
                            .linear_factor_cache_invalidations_gmin_escalated += 1;
                        break;
                    case CacheInvalidationReason::SourceSteppingActive:
                        result.backend_telemetry
                            .linear_factor_cache_invalidations_source_stepping_active += 1;
                        break;
                    case CacheInvalidationReason::NumericInstability:
                        result.backend_telemetry
                            .linear_factor_cache_invalidations_numeric_instability += 1;
                        break;
                    case CacheInvalidationReason::ManualInvalidate:
                        result.backend_telemetry
                            .linear_factor_cache_invalidations_manual_invalidate += 1;
                        break;
                    case CacheInvalidationReason::None:
                        break;
                }
            }
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
}

void Simulator::apply_post_accept_stiffness_update(const NewtonResult& step_result,
                                                    Integrator base_integrator,
                                                    int& high_iter_streak,
                                                    int& stiffness_cooldown,
                                                    bool& using_stiff_integrator) {
    if (!options_.stiffness_config.enable) {
        return;
    }

    if (step_result.iterations >= options_.stiffness_config.newton_iter_threshold) {
        high_iter_streak++;
    } else {
        high_iter_streak = 0;
    }

    if (high_iter_streak >= options_.stiffness_config.newton_streak_threshold) {
        stiffness_cooldown = options_.stiffness_config.cooldown_steps;
    }

    if (options_.stiffness_config.monitor_conditioning) {
        const auto telemetry = transient_services_.linear_solve->solver().telemetry();
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

void Simulator::process_accepted_step_events(Real t, Real dt_used,
                                              const Vector& x_prev,
                                              const NewtonResult& step_result,
                                              SimulationResult& result,
                                              EventCallback event_callback) {
    if (!options_.enable_events) {
        return;
    }
    for (auto& sw : switch_monitors_) {
        const Real v_now = (sw.ctrl >= 0) ? step_result.solution[sw.ctrl] : 0.0;
        const bool now_on = v_now > sw.v_threshold;
        if (now_on == sw.was_on) {
            continue;
        }
        Real t_event = t + dt_used;
        Vector x_event = step_result.solution;
        if (find_switch_event_time(sw, t, t + dt_used, x_prev, t_event, x_event)) {
            record_switch_event(sw, t_event, x_event, now_on, result, event_callback);
        } else {
            record_switch_event(sw, t + dt_used, step_result.solution, now_on, result, event_callback);
        }
        sw.was_on = now_on;
    }

    // refactor-pwl-switching-engine, Phase 4: detect PWL device commutations
    // (diode sign flip, gate threshold crossing, vcswitch threshold) using
    // each device's `should_commute()` predicate. First-order: commit at the
    // end of the accepted step; bisection-to-event is a follow-up (4.4).
    // The circuit_default reflects `SimulationOptions.switching_mode`
    // (Phase 5 plumbing).
    const auto pwl_events = circuit_.scan_pwl_commutations(
        step_result.solution, circuit_.default_switching_mode());
    if (!pwl_events.empty()) {
        circuit_.commit_pwl_commutations(pwl_events);
        // Phase 6 telemetry: a transition is "≥1 commutation in this step";
        // commutations counts individual device flips.
        result.backend_telemetry.pwl_topology_transitions += 1;
        result.backend_telemetry.pwl_event_commutations +=
            static_cast<int>(pwl_events.size());
        for (const auto& evt : pwl_events) {
            SimulationEvent sim_event;
            sim_event.time = t + dt_used;
            sim_event.type = evt.new_state ? SimulationEventType::SwitchOn
                                            : SimulationEventType::SwitchOff;
            sim_event.component = evt.device_name;
            sim_event.description = "pwl_commutation";
            result.events.push_back(std::move(sim_event));
            if (event_callback) {
                SwitchEvent cb;
                cb.switch_name = evt.device_name;
                cb.time = t + dt_used;
                cb.new_state = evt.new_state;
                event_callback(cb);
            }
        }
    }
}

void Simulator::finalize_transient_telemetry(SimulationResult& result) {
    // Phase 3 of `refactor-linear-solver-cache`: aggregate counters from
    // both the shared linear-solve service (Newton-DAE workload) and the
    // segment stepper's per-key LRU cache (segment-primary / PWL workload).
    // Pre-Phase-3 the segment stepper used the shared service so a single
    // read covered everything; with the per-key cache, the segment-primary
    // analyze/factorize/solve work lives in the cached entries and must be
    // summed in here to keep `LinearSolverTelemetry` authoritative.
    auto shared = transient_services_.linear_solve->solver().telemetry();
    const auto segment = transient_services_.segment_stepper->linear_solver_telemetry();
    shared.total_solve_calls       += segment.total_solve_calls;
    shared.total_analyze_calls     += segment.total_analyze_calls;
    shared.total_factorize_calls   += segment.total_factorize_calls;
    shared.total_iterations        += segment.total_iterations;
    shared.total_fallbacks         += segment.total_fallbacks;
    shared.total_analyze_time_seconds   += segment.total_analyze_time_seconds;
    shared.total_factorize_time_seconds += segment.total_factorize_time_seconds;
    shared.total_solve_time_seconds     += segment.total_solve_time_seconds;
    // Prefer segment's "last_*" when it has any work — the most recent
    // segment-primary step is what the user typically wants to inspect.
    if (segment.total_solve_calls > 0) {
        shared.last_iterations             = segment.last_iterations;
        shared.last_error                  = segment.last_error;
        shared.last_analyze_time_seconds   = segment.last_analyze_time_seconds;
        shared.last_factorize_time_seconds = segment.last_factorize_time_seconds;
        shared.last_solve_time_seconds     = segment.last_solve_time_seconds;
        shared.last_solver                 = segment.last_solver;
        shared.last_preconditioner         = segment.last_preconditioner;
    }
    result.linear_solver_telemetry = shared;
    const EquationAssemblerTelemetry assembler_telemetry =
        transient_services_.equation_assembler->telemetry();
    result.backend_telemetry.equation_assemble_system_calls =
        saturating_int(assembler_telemetry.system_calls + direct_assemble_system_calls_);
    result.backend_telemetry.equation_assemble_residual_calls =
        saturating_int(assembler_telemetry.residual_calls + direct_assemble_residual_calls_);
    result.backend_telemetry.equation_assemble_system_time_seconds =
        assembler_telemetry.system_time_seconds + direct_assemble_system_time_seconds_;
    result.backend_telemetry.equation_assemble_residual_time_seconds =
        assembler_telemetry.residual_time_seconds + direct_assemble_residual_time_seconds_;
    result.backend_telemetry.function_evaluations =
        result.backend_telemetry.equation_assemble_residual_calls;
    result.backend_telemetry.jacobian_evaluations =
        result.backend_telemetry.equation_assemble_system_calls;
    result.backend_telemetry.nonlinear_iterations = result.newton_iterations_total;
    result.backend_telemetry.error_test_failures = std::max(0, result.timestep_rejections);
    result.backend_telemetry.nonlinear_convergence_failures =
        (result.success || result.diagnostic != SimulationDiagnosticCode::TransientStepFailure) ? 0 : 1;
}

// =============================================================================

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
    if (dc.success) {
        return run_transient(dc.newton_result.solution, callback, event_callback, control);
    }

    if (options_.linear_solver.allow_fallback) {
        SimulationOptions startup_options = options_;
        startup_options.linear_solver.order.clear();
        startup_options.linear_solver.fallback_order.clear();
        startup_options.linear_solver.auto_select = true;
        apply_robust_newton_defaults(startup_options.newton_options);
        apply_robust_linear_solver_defaults(startup_options.linear_solver);
        startup_options.max_step_retries = std::max(startup_options.max_step_retries, 12);
        startup_options.fallback_policy.trace_retries = true;
        startup_options.fallback_policy.enable_transient_gmin = true;
        startup_options.fallback_policy.gmin_retry_threshold =
            std::max(startup_options.fallback_policy.gmin_retry_threshold, 1);

        Vector startup_state = circuit_.initial_state();
        const Index expected_size = static_cast<Index>(circuit_.system_size());
        if (startup_state.size() != expected_size || !startup_state.allFinite()) {
            startup_state = Vector::Zero(expected_size);
        }

        Simulator startup_sim(circuit_, startup_options);
        SimulationResult startup_result = startup_sim.run_transient(
            startup_state,
            std::move(callback),
            std::move(event_callback),
            control);
        startup_result.backend_telemetry.backend_recovery_count += 1;
        startup_result.backend_telemetry.escalation_count += 1;

        if (startup_result.success) {
            startup_result.message =
                "Transient completed (startup fallback after DC operating point failure)";
            return startup_result;
        }

        if (startup_result.message.empty()) {
            startup_result.message =
                "DC operating point failed: " + dc.message + "; startup fallback failed";
        } else {
            startup_result.message =
                "DC operating point failed: " + dc.message +
                "; startup fallback failed: " + startup_result.message;
        }
        return startup_result;
    }

    SimulationResult result;
    result.success = false;
    result.final_status = dc.newton_result.status;
    result.diagnostic = SimulationDiagnosticCode::DcOperatingPointFailure;
    result.message = "DC operating point failed: " + dc.message;
    result.linear_solver_telemetry = dc.linear_solver_telemetry;
    return result;
}

SimulationResult Simulator::run_transient(const Vector& x0,
                                          SimulationCallback callback,
                                          EventCallback event_callback,
                                          SimulationControl* control) {
    const std::string formulation_mode = std::string(formulation_mode_to_string(options_.formulation_mode));
    if (const auto input_issue = validate_transient_inputs(circuit_, options_, x0)) {
        SimulationResult result;
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.diagnostic = input_issue->diagnostic;
        result.message = input_issue->message;
        result.backend_telemetry.requested_backend = "native";
        result.backend_telemetry.selected_backend = "native";
        result.backend_telemetry.solver_family = "native";
        result.backend_telemetry.formulation_mode = formulation_mode;
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
    if (native_result.backend_telemetry.selected_backend.empty()) {
        native_result.backend_telemetry.selected_backend = "native";
    }
    if (native_result.backend_telemetry.solver_family.empty()) {
        native_result.backend_telemetry.solver_family = "native";
    }
    if (native_result.backend_telemetry.formulation_mode.empty()) {
        native_result.backend_telemetry.formulation_mode = formulation_mode;
    }
    return native_result;
}

SimulationResult Simulator::run_transient_native_impl(const Vector& x0,
                                                      SimulationCallback callback,
                                                      EventCallback event_callback,
                                                      SimulationControl* control) {
    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    const std::string formulation_mode = std::string(formulation_mode_to_string(options_.formulation_mode));
    result.backend_telemetry.selected_backend = "native";
    result.backend_telemetry.solver_family = "native";
    result.backend_telemetry.formulation_mode = formulation_mode;
    const std::size_t sample_reserve = estimate_output_sample_reserve(options_);
    result.backend_telemetry.reserved_output_samples =
        saturating_int(static_cast<std::uint64_t>(sample_reserve));
    if (sample_reserve > 0) {
        result.time.reserve(sample_reserve);
        result.states.reserve(sample_reserve);
    }
    if (options_.fallback_policy.trace_retries) {
        const std::size_t retries_per_step =
            static_cast<std::size_t>(std::max(1, options_.max_step_retries + 1));
        const std::size_t fallback_reserve =
            std::min<std::size_t>(sample_reserve * retries_per_step, kMaxFallbackReserve);
        if (fallback_reserve > 0) {
            result.fallback_trace.reserve(fallback_reserve);
        }
    }
    if (options_.enable_events && sample_reserve > 0 && !switch_monitors_.empty()) {
        const std::size_t event_reserve =
            std::min<std::size_t>(sample_reserve * switch_monitors_.size(), sample_reserve * 2);
        if (event_reserve > 0) {
            result.events.reserve(event_reserve);
        }
    }

    initialize_loss_tracking();
    lte_estimator_.reset();
    transient_gmin_ = 0.0;
    segment_primary_disabled_for_run_ = false;
    last_step_solve_path_ = StepSolvePath::DaeFallback;
    last_step_solve_reason_ = "init";
    last_step_segment_cache_hit_ = false;
    last_step_segment_attempted_ = false;
    last_step_linear_factor_cache_hit_ = false;
    last_step_linear_factor_cache_miss_ = false;
    last_step_symbolic_factor_cache_hit_ = false;
    last_step_linear_factor_cache_invalidation_reason_.clear();
    last_step_linear_factor_cache_invalidation_reason_typed_ = CacheInvalidationReason::None;
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
    int model_regularization_escalations = 0;
    std::optional<Real> pending_dt_override;
    const TransientStepMode step_mode = resolve_step_mode(options_);
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
    const bool switching_recovery_profile = std::any_of(
        circuit_.devices().begin(),
        circuit_.devices().end(),
        [](const auto& device_variant) {
            return std::visit(
                [](const auto& dev) {
                    using T = std::decay_t<decltype(dev)>;
                    return std::is_same_v<T, IdealSwitch> ||
                           std::is_same_v<T, VoltageControlledSwitch> ||
                           std::is_same_v<T, MOSFET> ||
                           std::is_same_v<T, IGBT> ||
                           std::is_same_v<T, PWMVoltageSource> ||
                           std::is_same_v<T, PulseVoltageSource>;
                },
                device_variant);
        });
    const bool can_auto_recover =
        options_.linear_solver.allow_fallback || switching_recovery_profile;
    const int max_global_recovery_attempts =
        switching_recovery_profile ? (kMaxGlobalRecoveryAttempts + 1) : kMaxGlobalRecoveryAttempts;

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
            result.virtual_channels.reserve(result.virtual_channel_metadata.size());
        }

        auto mixed_step = circuit_.execute_mixed_domain_step(state, sample_time);
        if (result.mixed_domain_phase_order.empty()) {
            result.mixed_domain_phase_order = mixed_step.phase_order;
        }

        const std::size_t sample_count = result.time.size();
        const Real nan = std::numeric_limits<Real>::quiet_NaN();
        auto append_series_value = [&](std::vector<Real>& series, Real value) {
            const std::size_t capacity_before = series.capacity();
            series.push_back(value);
            if (series.capacity() != capacity_before) {
                result.backend_telemetry.virtual_channel_reallocations += 1;
            }
        };

        for (auto& [channel, series] : result.virtual_channels) {
            while (series.size() + 1 < sample_count) {
                append_series_value(series, nan);
            }
        }

        for (const auto& [channel, value] : mixed_step.channel_values) {
            auto [it, inserted] = result.virtual_channels.try_emplace(channel);
            auto& series = it->second;
            if (inserted && sample_reserve > 0) {
                series.reserve(sample_reserve);
            }
            while (series.size() + 1 < sample_count) {
                append_series_value(series, nan);
            }
            append_series_value(series, value);

            if (!result.virtual_channel_metadata.contains(channel)) {
                result.virtual_channel_metadata[channel] = VirtualChannelMetadata{
                    "virtual", channel, "control", {}
                };
            }
        }

        for (auto& [channel, series] : result.virtual_channels) {
            if (series.size() < sample_count) {
                append_series_value(series, nan);
            }
        }
    };

    auto append_sample = [&](Real sample_time, const Vector& state) {
        const std::size_t time_capacity_before = result.time.capacity();
        const std::size_t state_capacity_before = result.states.capacity();
        result.time.push_back(sample_time);
        result.states.push_back(state);
        if (result.time.capacity() != time_capacity_before) {
            result.backend_telemetry.time_series_reallocations += 1;
        }
        if (result.states.capacity() != state_capacity_before) {
            result.backend_telemetry.state_series_reallocations += 1;
        }
        append_virtual_sample(state, sample_time);
    };

    append_sample(t, x);

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

        const Real dt_candidate = pending_dt_override.has_value()
            ? *pending_dt_override
            : (fixed_step_policy.enabled() ? fixed_step_policy.default_dt() : dt);
        pending_dt_override.reset();
        dt = clamp_dt_for_mode(dt_candidate);
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
            bool calendar_boundary_adjacent = false;
            Real dt_segment = 0.0;
            if (options_.enable_events && dt > options_.dt_min * 1.01) {
                const Real min_calendar_clip_dt = min_event_substep_dt(dt);
                const Real segment_probe_stop =
                    std::min(options_.tstop, t + dt + min_calendar_clip_dt);
                TransientStepRequest segment_probe_request = step_request;
                segment_probe_request.t_target = segment_probe_stop;
                const Real t_segment_target =
                    transient_services_.event_scheduler->next_segment_target(segment_probe_request,
                                                                             segment_probe_stop);
                dt_segment = t_segment_target - t;
                has_calendar_boundary_in_step =
                    dt_segment > min_calendar_clip_dt && dt_segment < dt * 0.999;
                const Real boundary_adjacent_tol =
                    std::max(min_calendar_clip_dt * Real{0.5}, Real{1e-15});
                calendar_boundary_adjacent =
                    !has_calendar_boundary_in_step &&
                    dt_segment > 0.0 &&
                    dt_segment >= dt * Real{0.999} &&
                    dt_segment <= dt + boundary_adjacent_tol;
            }

            if (variable_step_policy.enabled() &&
                (has_calendar_boundary_in_step || calendar_boundary_adjacent)) {
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
            collect_step_solve_telemetry(result);

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
                    const int retry_ordinal = std::max(1, retries);
                    const int gmin_retry_threshold =
                        std::max(1, options_.fallback_policy.gmin_retry_threshold);
                    const int model_retry_threshold =
                        std::max(1, options_.model_regularization.retry_threshold);
                    const int model_max_escalations =
                        std::max(0, options_.model_regularization.max_escalations);

                    const bool can_apply_model_regularization =
                        options_.model_regularization.enable_auto &&
                        retry_ordinal >= model_retry_threshold &&
                        model_regularization_escalations < model_max_escalations;

                    if (can_apply_model_regularization &&
                        (!options_.model_regularization.apply_only_in_recovery || retries > 0)) {
                        const Real stage_factor =
                            std::max<Real>(1.0, options_.model_regularization.escalation_factor);
                        const Real stage_scale = std::pow(
                            stage_factor,
                            static_cast<Real>(model_regularization_escalations));

                        const Real mosfet_kp_max =
                            options_.model_regularization.mosfet_kp_max / stage_scale;
                        const Real mosfet_g_off_min =
                            options_.model_regularization.mosfet_g_off_min * stage_scale;
                        const Real diode_g_on_max =
                            options_.model_regularization.diode_g_on_max / stage_scale;
                        const Real diode_g_off_min =
                            options_.model_regularization.diode_g_off_min * stage_scale;
                        const Real igbt_g_on_max =
                            options_.model_regularization.igbt_g_on_max / stage_scale;
                        const Real igbt_g_off_min =
                            options_.model_regularization.igbt_g_off_min * stage_scale;
                        const Real switch_g_on_max =
                            options_.model_regularization.switch_g_on_max / stage_scale;
                        const Real switch_g_off_min =
                            options_.model_regularization.switch_g_off_min * stage_scale;
                        const Real vcswitch_g_on_max =
                            options_.model_regularization.vcswitch_g_on_max / stage_scale;
                        const Real vcswitch_g_off_min =
                            options_.model_regularization.vcswitch_g_off_min * stage_scale;

                        const int changed_devices = circuit_.apply_numerical_regularization(
                            mosfet_kp_max,
                            mosfet_g_off_min,
                            diode_g_on_max,
                            diode_g_off_min,
                            igbt_g_on_max,
                            igbt_g_off_min,
                            switch_g_on_max,
                            switch_g_off_min,
                            vcswitch_g_on_max,
                            vcswitch_g_off_min);

                        model_regularization_escalations += 1;
                        const Real intensity = std::min<Real>(
                            1.0,
                            static_cast<Real>(model_regularization_escalations) /
                                static_cast<Real>(std::max(1, model_max_escalations)));
                        result.backend_telemetry.model_regularization_events += 1;
                        result.backend_telemetry.model_regularization_last_changed = changed_devices;
                        result.backend_telemetry.model_regularization_last_intensity = intensity;

                        std::ostringstream model_action;
                        model_action << "recovery_stage_regularization_model"
                                     << " changed=" << changed_devices
                                     << " intensity=" << intensity
                                     << " escalation=" << model_regularization_escalations
                                     << "/" << std::max(1, model_max_escalations);
                        recovery_action = model_action.str();
                    }

                    if (options_.fallback_policy.enable_transient_gmin &&
                        retry_ordinal >= gmin_retry_threshold) {
                        Real next_gmin = transient_gmin_ > 0.0
                            ? transient_gmin_ * options_.fallback_policy.gmin_growth
                            : options_.fallback_policy.gmin_initial;
                        transient_gmin_ = std::min(options_.fallback_policy.gmin_max,
                                                   std::max(next_gmin, options_.fallback_policy.gmin_initial));
                        recovery_reason_code = FallbackReasonCode::TransientGminEscalation;
                        std::ostringstream action;
                        action << recovery_action << " gmin=" << transient_gmin_;
                        recovery_action = action.str();
                    } else if (options_.fallback_policy.enable_transient_gmin) {
                        std::ostringstream action;
                        action << recovery_action << " gmin_wait_threshold=" << gmin_retry_threshold;
                        recovery_action = action.str();
                    } else if (recovery_action == "recover_dt") {
                        recovery_action = "recovery_stage_regularization_disabled";
                    }

                    if (switching_recovery_profile) {
                        const Real backoff_factor = retry_ordinal >= 3 ? Real{0.1} : Real{0.25};
                        const Real dt_before_backoff = dt;
                        const Real dt_target = std::max(options_.dt_min, dt_before_backoff * backoff_factor);
                        dt = clamp_dt_for_mode(dt_target);

                        if (options_.enable_bdf_order_control) {
                            bdf_controller_.set_order(1);
                            circuit_.set_integration_order(1);
                        } else if (options_.stiffness_config.switch_integrator) {
                            circuit_.set_integration_method(options_.stiffness_config.stiff_integrator);
                            using_stiff_integrator = true;
                        }

                        std::ostringstream action;
                        action << recovery_action
                               << " switching_backoff=" << backoff_factor
                               << " dt=" << dt_before_backoff << "->" << dt;
                        recovery_action = action.str();
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
            const Real fixed_hold_threshold =
                std::max(std::abs(options_.dt) * Real{1e-3}, Real{1e-15});
            const bool fixed_pathological_substep =
                fixed_step_policy.enabled() &&
                dt_used <= fixed_hold_threshold &&
                dt_used < fixed_step_policy.default_dt() * Real{0.5};
            const bool fixed_recovery_exhausted =
                fixed_step_policy.enabled() &&
                can_auto_recover &&
                global_recovery_attempts >= max_global_recovery_attempts;
            const bool allow_dt_hold_advance =
                ((variable_step_policy.enabled() && at_dt_min_floor) ||
                 fixed_pathological_substep ||
                 fixed_recovery_exhausted) &&
                dt_min_hold_advances < kMaxDtMinHoldAdvances;

            if (allow_dt_hold_advance) {
                // Last-resort progress safeguard for hard switching points where
                // Newton repeatedly fails at pathological dt values. Advance
                // with state hold to cross the instant instead of aborting.
                dt_min_hold_advances++;
                circuit_.clear_stage_context();
                Real hold_dt = dt_used;
                std::string hold_action = "dt_min_hold_advance";
                if (variable_step_policy.enabled() && at_dt_min_floor) {
                    const Real variable_hold_dt =
                        clamp_dt_for_mode(std::max(options_.dt, options_.dt_min));
                    if (variable_hold_dt > 0.0) {
                        hold_dt = std::max(hold_dt, variable_hold_dt);
                    }
                    hold_action = "variable_macro_hold_advance";
                }
                if (fixed_step_policy.enabled() &&
                    (fixed_pathological_substep || fixed_recovery_exhausted)) {
                    const Real macro_hold_dt = clamp_dt_for_mode(fixed_step_policy.default_dt());
                    if (macro_hold_dt > 0.0) {
                        hold_dt = macro_hold_dt;
                    }
                    hold_action = "fixed_macro_hold_advance";
                }
                record_fallback_event(result,
                                      result.total_steps,
                                      retries,
                                      t,
                                      hold_dt,
                                      FallbackReasonCode::MaxRetriesExceeded,
                                      step_result.status,
                                      hold_action);

                rejection_streak = 0;
                high_iter_streak = 0;
                global_recovery_attempts = 0;
                transient_gmin_ = 0.0;
                transient_services_.nonlinear_solve->set_options(baseline_newton_options);

                accumulate_conduction_losses(step_anchor_state, hold_dt);
                update_thermal_state(hold_dt);

                t += hold_dt;
                pending_dt_override = hold_dt;
                if (variable_step_policy.enabled()) {
                    variable_step_policy.on_step_accepted(hold_dt);
                }
                bool emit_sample = true;
                if (fixed_step_policy.enabled()) {
                    emit_sample = fixed_step_policy.on_step_accepted(t);
                }
                x = step_anchor_state;
                circuit_.update_history(x);
                if (variable_step_policy.enabled()) {
                    lte_estimator_.record_solution(x, t, hold_dt);
                }

                if (emit_sample || nearly_same_time(t, options_.tstop)) {
                    append_sample(t, x);
                    if (callback) {
                        callback(t, x);
                    }
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

            if (can_auto_recover && global_recovery_attempts < max_global_recovery_attempts) {
                ++global_recovery_attempts;
                auto_recovery_attempted = true;
                result.backend_telemetry.backend_recovery_count += 1;
                result.backend_telemetry.escalation_count += 1;

                circuit_.clear_stage_context();
                const Real recovered_dt = clamp_dt_for_mode(std::max(options_.dt_min, dt_used * (
                    global_recovery_attempts == 1 ? Real{0.25} : Real{0.1})));
                dt = recovered_dt;
                pending_dt_override = recovered_dt;
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

        apply_post_accept_stiffness_update(step_result, base_integrator,
                                           high_iter_streak, stiffness_cooldown,
                                           using_stiff_integrator);
        process_accepted_step_events(t, dt_used, x, step_result, result, event_callback);

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
            append_sample(t, x);

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
    finalize_component_electrothermal(result);
    finalize_transient_telemetry(result);

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
