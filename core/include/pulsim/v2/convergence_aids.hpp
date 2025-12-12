#pragma once

// =============================================================================
// PulsimCore v2 - Advanced Convergence Aids for DC Analysis
// =============================================================================
// This header provides convergence assistance strategies for difficult circuits:
// - 5.1: Gmin Stepping - exponential conductance to ground
// - 5.2: Source Stepping - continuation-based source scaling
// - 5.3: Pseudo-Transient Continuation - implicit dynamics for DC
// - 5.4: Robust Initialization - intelligent initial guesses
// =============================================================================

#include "pulsim/v2/numeric_types.hpp"
#include "pulsim/v2/solver.hpp"
#include <vector>
#include <functional>
#include <optional>
#include <random>
#include <cmath>
#include <string>
#include <algorithm>

namespace pulsim::v2 {

// =============================================================================
// 5.1: Gmin Stepping
// =============================================================================

/// Configuration for Gmin stepping
struct GminConfig {
    Real initial_gmin = 1e-2;       // Starting Gmin (S)
    Real final_gmin = 1e-12;        // Target Gmin (S)
    Real reduction_factor = 10.0;   // Factor to reduce Gmin each step
    int max_steps = 20;             // Maximum Gmin steps
    bool enable_logging = false;    // Log Gmin ramp parameters

    /// Get number of steps needed
    [[nodiscard]] int required_steps() const {
        if (initial_gmin <= final_gmin) return 1;
        return static_cast<int>(std::ceil(
            std::log(initial_gmin / final_gmin) / std::log(reduction_factor)));
    }
};

/// Log entry for Gmin stepping (5.1.5)
struct GminLogEntry {
    int step = 0;
    Real gmin = 0.0;
    bool converged = false;
    int newton_iterations = 0;
    Real final_residual = 0.0;
};

/// Gmin stepping strategy for DC convergence (5.1.1-5.1.5)
class GminStepping {
public:
    /// Callback to add Gmin to system (node_index, gmin_value)
    using AddGminCallback = std::function<void(Index node, Real gmin)>;

    /// DC solve function signature
    using SolveFunction = std::function<NewtonResult(const Vector& x0)>;

    explicit GminStepping(const GminConfig& config = {})
        : config_(config) {}

    /// Reset stepping state
    void reset() {
        current_gmin_ = config_.initial_gmin;
        current_step_ = 0;
        log_.clear();
    }

    /// Get current Gmin value
    [[nodiscard]] Real current_gmin() const { return current_gmin_; }

    /// Get current step
    [[nodiscard]] int current_step() const { return current_step_; }

    /// Check if stepping is complete
    [[nodiscard]] bool is_complete() const {
        return current_gmin_ <= config_.final_gmin ||
               current_step_ >= config_.max_steps;
    }

    /// Advance to next Gmin value (5.1.3)
    void advance() {
        current_gmin_ /= config_.reduction_factor;
        if (current_gmin_ < config_.final_gmin) {
            current_gmin_ = config_.final_gmin;
        }
        ++current_step_;
    }

    /// Add Gmin to all nodes (5.1.2)
    void apply_gmin(Index num_nodes, SparseMatrix& G, Vector& rhs,
                    const Vector& solution) const {
        for (Index i = 0; i < num_nodes; ++i) {
            // Add Gmin conductance to diagonal
            G.coeffRef(i, i) += current_gmin_;
            // Add current source to maintain voltage: I = Gmin * V
            rhs[i] += current_gmin_ * solution[i];
        }
    }

    /// Execute Gmin stepping (5.1.4)
    [[nodiscard]] NewtonResult execute(
        const Vector& x0,
        Index num_nodes,
        SolveFunction solve_func) {

        reset();
        Vector current_solution = x0;
        NewtonResult result;

        while (!is_complete()) {
            // Solve with current Gmin
            result = solve_func(current_solution);

            // Log if enabled (5.1.5)
            if (config_.enable_logging) {
                GminLogEntry entry;
                entry.step = current_step_;
                entry.gmin = current_gmin_;
                entry.converged = result.success();
                entry.newton_iterations = result.iterations;
                entry.final_residual = result.final_residual;
                log_.push_back(entry);
            }

            if (!result.success()) {
                // Convergence failed at this Gmin level
                result.error_message = "Gmin stepping failed at step " +
                    std::to_string(current_step_) + " with Gmin = " +
                    std::to_string(current_gmin_);
                return result;
            }

            // Update solution and advance
            current_solution = result.solution;
            advance();
        }

        // Final solve without Gmin (or with final_gmin)
        result = solve_func(current_solution);

        if (config_.enable_logging) {
            GminLogEntry entry;
            entry.step = current_step_;
            entry.gmin = config_.final_gmin;
            entry.converged = result.success();
            entry.newton_iterations = result.iterations;
            entry.final_residual = result.final_residual;
            log_.push_back(entry);
        }

        return result;
    }

    /// Get log entries
    [[nodiscard]] const std::vector<GminLogEntry>& log() const { return log_; }

    /// Export log to CSV format
    [[nodiscard]] std::string log_to_csv() const {
        std::string csv = "step,gmin,converged,newton_iters,residual\n";
        for (const auto& e : log_) {
            csv += std::to_string(e.step) + "," +
                   std::to_string(e.gmin) + "," +
                   (e.converged ? "true" : "false") + "," +
                   std::to_string(e.newton_iterations) + "," +
                   std::to_string(e.final_residual) + "\n";
        }
        return csv;
    }

    [[nodiscard]] const GminConfig& config() const { return config_; }
    void set_config(const GminConfig& config) { config_ = config; }

private:
    GminConfig config_;
    Real current_gmin_ = 1e-2;
    int current_step_ = 0;
    std::vector<GminLogEntry> log_;
};

// =============================================================================
// 5.2: Source Stepping
// =============================================================================

/// Configuration for source stepping
struct SourceSteppingConfig {
    Real initial_scale = 0.0;       // Start scale (0 = all sources off)
    Real final_scale = 1.0;         // End scale (1 = full sources)
    Real initial_step = 0.25;       // Initial step size
    Real min_step = 0.01;           // Minimum step size
    Real max_step = 0.5;            // Maximum step size
    int max_steps = 100;            // Maximum continuation steps
    Real step_increase = 1.5;       // Factor to increase step on success
    Real step_decrease = 0.5;       // Factor to decrease step on failure
    int max_failures = 5;           // Max consecutive failures before abort
    bool enable_logging = false;    // Log continuation parameters
};

/// Log entry for source stepping (5.2.5)
struct SourceStepLogEntry {
    int step = 0;
    Real scale = 0.0;
    Real step_size = 0.0;
    bool converged = false;
    int newton_iterations = 0;
    Real final_residual = 0.0;
};

/// Source stepping result
struct SourceSteppingResult {
    NewtonResult final_result;
    bool success = false;
    int total_steps = 0;
    int total_newton_iterations = 0;
    std::string error_message;
    std::vector<SourceStepLogEntry> log;
};

/// Source stepping continuation strategy (5.2.1-5.2.5)
class SourceStepping {
public:
    /// DC solve function with scaled sources
    using ScaledSolveFunction = std::function<NewtonResult(const Vector& x0, Real scale)>;

    explicit SourceStepping(const SourceSteppingConfig& config = {})
        : config_(config) {}

    /// Execute source stepping (5.2.4)
    [[nodiscard]] SourceSteppingResult execute(
        const Vector& x0,
        ScaledSolveFunction solve_func) {

        SourceSteppingResult result;
        Vector current_solution = x0;
        Real current_scale = config_.initial_scale;
        Real step_size = config_.initial_step;
        int consecutive_failures = 0;

        while (current_scale < config_.final_scale &&
               result.total_steps < config_.max_steps) {

            // Compute next scale (5.2.2)
            Real next_scale = std::min(current_scale + step_size, config_.final_scale);

            // Solve at this scale (5.2.1)
            NewtonResult newton_result = solve_func(current_solution, next_scale);
            result.total_newton_iterations += newton_result.iterations;

            // Log if enabled
            if (config_.enable_logging) {
                SourceStepLogEntry entry;
                entry.step = result.total_steps;
                entry.scale = next_scale;
                entry.step_size = step_size;
                entry.converged = newton_result.success();
                entry.newton_iterations = newton_result.iterations;
                entry.final_residual = newton_result.final_residual;
                result.log.push_back(entry);
            }

            if (newton_result.success()) {
                // Success - advance and potentially increase step (5.2.3)
                current_solution = newton_result.solution;
                current_scale = next_scale;
                step_size = std::min(step_size * config_.step_increase, config_.max_step);
                consecutive_failures = 0;
            } else {
                // Failure - reduce step size
                step_size *= config_.step_decrease;
                ++consecutive_failures;

                // Check abort criteria (5.2.5)
                if (step_size < config_.min_step ||
                    consecutive_failures >= config_.max_failures) {
                    result.success = false;
                    result.error_message = std::string("Source stepping failed: ") +
                        (step_size < config_.min_step ?
                         "step size below minimum" :
                         "max consecutive failures reached");
                    result.final_result = newton_result;
                    result.total_steps++;
                    return result;
                }
            }

            result.total_steps++;
        }

        // Final solve at full scale
        result.final_result = solve_func(current_solution, config_.final_scale);
        result.total_newton_iterations += result.final_result.iterations;
        result.success = result.final_result.success();

        if (!result.success) {
            result.error_message = "Final solve at full scale failed";
        }

        return result;
    }

    [[nodiscard]] const SourceSteppingConfig& config() const { return config_; }
    void set_config(const SourceSteppingConfig& config) { config_ = config; }

private:
    SourceSteppingConfig config_;
};

// =============================================================================
// 5.3: Pseudo-Transient Continuation
// =============================================================================

/// Configuration for pseudo-transient continuation
struct PseudoTransientConfig {
    Real initial_dt = 1e-9;         // Initial pseudo-timestep (s)
    Real max_dt = 1e3;              // Maximum pseudo-timestep (s)
    Real min_dt = 1e-15;            // Minimum pseudo-timestep (s)
    Real dt_increase = 2.0;         // Factor to increase dt on success
    Real dt_decrease = 0.5;         // Factor to decrease dt on failure
    Real convergence_threshold = 1e-6; // Residual threshold for "converged"
    int max_iterations = 1000;      // Maximum pseudo-transient iterations
    Real capacitance = 1e-9;        // Pseudo-capacitance to add (F)
    bool enable_logging = false;    // Log pseudo-dt metrics
};

/// Log entry for pseudo-transient (5.3.5)
struct PseudoTransientLogEntry {
    int iteration = 0;
    Real pseudo_dt = 0.0;
    Real residual_norm = 0.0;
    bool newton_converged = false;
    int newton_iterations = 0;
};

/// Pseudo-transient continuation result
struct PseudoTransientResult {
    Vector solution;
    bool success = false;
    int total_iterations = 0;
    int total_newton_iterations = 0;
    Real final_residual = 0.0;
    std::string error_message;
    std::vector<PseudoTransientLogEntry> log;
};

/// Pseudo-transient continuation for DC analysis (5.3.1-5.3.5)
class PseudoTransientContinuation {
public:
    /// System evaluation function
    using SystemFunction = std::function<void(const Vector& x, Vector& f, SparseMatrix& J)>;

    /// Newton solve function with modified system
    using NewtonSolveFunction = std::function<NewtonResult(const Vector& x0, Real pseudo_dt)>;

    explicit PseudoTransientContinuation(const PseudoTransientConfig& config = {})
        : config_(config) {}

    /// Add pseudo-capacitance for DC convergence (5.3.2)
    void add_pseudo_capacitance(Index num_nodes, Real pseudo_dt,
                                 SparseMatrix& J, Vector& f,
                                 const Vector& x, const Vector& x_prev) const {
        Real pseudo_G = config_.capacitance / pseudo_dt;

        for (Index i = 0; i < num_nodes; ++i) {
            // Add C/dt to diagonal (Jacobian modification)
            J.coeffRef(i, i) += pseudo_G;
            // Add C/dt * (x - x_prev) to residual
            f[i] += pseudo_G * (x[i] - x_prev[i]);
        }
    }

    /// Execute pseudo-transient continuation (5.3.4)
    [[nodiscard]] PseudoTransientResult execute(
        const Vector& x0,
        Index num_nodes,
        NewtonSolveFunction newton_solve) {

        PseudoTransientResult result;
        result.solution = x0;
        Vector x_prev = x0;
        Real pseudo_dt = config_.initial_dt;

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            // Solve Newton step with pseudo-transient term
            NewtonResult newton_result = newton_solve(result.solution, pseudo_dt);
            result.total_newton_iterations += newton_result.iterations;

            // Log if enabled (5.3.5)
            if (config_.enable_logging) {
                PseudoTransientLogEntry entry;
                entry.iteration = iter;
                entry.pseudo_dt = pseudo_dt;
                entry.residual_norm = newton_result.final_residual;
                entry.newton_converged = newton_result.success();
                entry.newton_iterations = newton_result.iterations;
                result.log.push_back(entry);
            }

            if (newton_result.success()) {
                // Check for DC convergence
                Real change_norm = (newton_result.solution - result.solution).norm();
                result.solution = newton_result.solution;
                result.final_residual = newton_result.final_residual;

                if (newton_result.final_residual < config_.convergence_threshold &&
                    change_norm < config_.convergence_threshold) {
                    // DC converged
                    result.success = true;
                    result.total_iterations = iter + 1;
                    return result;
                }

                // Increase pseudo-dt (5.3.3)
                pseudo_dt = std::min(pseudo_dt * config_.dt_increase, config_.max_dt);
                x_prev = result.solution;
            } else {
                // Newton failed - decrease pseudo-dt (5.3.5 safety clamp)
                pseudo_dt *= config_.dt_decrease;

                if (pseudo_dt < config_.min_dt) {
                    result.success = false;
                    result.error_message = "Pseudo-dt below minimum";
                    result.total_iterations = iter + 1;
                    return result;
                }
            }
        }

        result.success = false;
        result.error_message = "Max pseudo-transient iterations reached";
        result.total_iterations = config_.max_iterations;
        return result;
    }

    [[nodiscard]] const PseudoTransientConfig& config() const { return config_; }
    void set_config(const PseudoTransientConfig& config) { config_ = config; }

private:
    PseudoTransientConfig config_;
};

// =============================================================================
// 5.4: Robust Initialization
// =============================================================================

/// Configuration for robust initialization
struct InitializationConfig {
    Real default_voltage = 0.0;     // Default node voltage (V)
    Real supply_voltage = 12.0;     // Assumed supply voltage (V)
    Real diode_forward = 0.7;       // Diode forward voltage (V)
    Real mosfet_threshold = 2.0;    // MOSFET threshold voltage (V)
    bool use_zero_init = false;     // Start from zero
    bool use_warm_start = true;     // Use previous solution
    int max_random_restarts = 5;    // Max random restarts on failure
    std::uint64_t random_seed = 42; // Seed for deterministic restarts (5.4.5)
    Real random_voltage_range = 5.0;// Random voltage range (+/-)
};

/// Device type hint for initialization
enum class DeviceHint {
    None,
    DiodeAnode,
    DiodeCathode,
    MOSFETGate,
    MOSFETDrain,
    MOSFETSource,
    BJTBase,
    BJTCollector,
    BJTEmitter,
    SupplyPositive,
    SupplyNegative,
    Ground
};

/// Node initialization hint
struct NodeInitHint {
    Index node_index = 0;
    DeviceHint hint = DeviceHint::None;
    Real hint_voltage = 0.0;        // Optional explicit hint
    bool has_explicit_hint = false;
};

/// Robust initialization strategy (5.4.1-5.4.5)
class RobustInitialization {
public:
    explicit RobustInitialization(const InitializationConfig& config = {})
        : config_(config), rng_(config.random_seed) {}

    /// Set node hints for initialization
    void set_hints(const std::vector<NodeInitHint>& hints) {
        hints_ = hints;
    }

    /// Add a single hint
    void add_hint(const NodeInitHint& hint) {
        hints_.push_back(hint);
    }

    /// Clear all hints
    void clear_hints() {
        hints_.clear();
    }

    /// Generate initial guess based on heuristics (5.4.1)
    [[nodiscard]] Vector generate_initial_guess(Index num_nodes, Index num_branches) const {
        Index total = num_nodes + num_branches;
        Vector x0 = Vector::Zero(total);

        if (config_.use_zero_init) {
            return x0;
        }

        // Apply default voltage to nodes
        for (Index i = 0; i < num_nodes; ++i) {
            x0[i] = config_.default_voltage;
        }

        // Apply device-specific hints (5.4.2)
        for (const auto& hint : hints_) {
            if (hint.node_index >= num_nodes) continue;

            if (hint.has_explicit_hint) {
                x0[hint.node_index] = hint.hint_voltage;
            } else {
                x0[hint.node_index] = voltage_from_hint(hint.hint);
            }
        }

        return x0;
    }

    /// Get voltage guess from device hint (5.4.2)
    [[nodiscard]] Real voltage_from_hint(DeviceHint hint) const {
        switch (hint) {
            case DeviceHint::Ground:
                return 0.0;
            case DeviceHint::SupplyPositive:
                return config_.supply_voltage;
            case DeviceHint::SupplyNegative:
                return -config_.supply_voltage;
            case DeviceHint::DiodeAnode:
                return config_.diode_forward;
            case DeviceHint::DiodeCathode:
                return 0.0;
            case DeviceHint::MOSFETGate:
                return config_.mosfet_threshold * 1.5;
            case DeviceHint::MOSFETDrain:
                return config_.supply_voltage * 0.5;
            case DeviceHint::MOSFETSource:
                return 0.0;
            case DeviceHint::BJTBase:
                return config_.diode_forward;
            case DeviceHint::BJTCollector:
                return config_.supply_voltage * 0.5;
            case DeviceHint::BJTEmitter:
                return 0.0;
            default:
                return config_.default_voltage;
        }
    }

    /// Warm start from previous solution (5.4.3)
    [[nodiscard]] Vector warm_start(const Vector& previous_solution,
                                     Index num_nodes, Index num_branches) const {
        Index total = num_nodes + num_branches;

        if (!config_.use_warm_start || previous_solution.size() == 0) {
            return generate_initial_guess(num_nodes, num_branches);
        }

        // If sizes match, use directly
        if (previous_solution.size() == total) {
            return previous_solution;
        }

        // Otherwise, use what we can
        Vector x0 = generate_initial_guess(num_nodes, num_branches);
        Index copy_size = std::min(static_cast<Index>(previous_solution.size()), total);
        for (Index i = 0; i < copy_size; ++i) {
            x0[i] = previous_solution[i];
        }

        return x0;
    }

    /// Generate random initial guess (5.4.4, 5.4.5)
    [[nodiscard]] Vector random_initial_guess(Index num_nodes, Index num_branches) {
        Index total = num_nodes + num_branches;
        Vector x0(total);

        std::uniform_real_distribution<Real> dist(
            -config_.random_voltage_range,
            config_.random_voltage_range);

        // Random voltages for nodes
        for (Index i = 0; i < num_nodes; ++i) {
            x0[i] = dist(rng_);
        }

        // Zero for branch currents (more stable starting point)
        for (Index i = num_nodes; i < total; ++i) {
            x0[i] = 0.0;
        }

        return x0;
    }

    /// Reset random seed for deterministic behavior (5.4.5)
    void reset_random_seed() {
        rng_.seed(config_.random_seed);
    }

    /// Set seed for deterministic restarts (5.4.5)
    void set_seed(std::uint64_t seed) {
        config_.random_seed = seed;
        rng_.seed(seed);
    }

    [[nodiscard]] const InitializationConfig& config() const { return config_; }
    void set_config(const InitializationConfig& config) {
        config_ = config;
        rng_.seed(config.random_seed);
    }

private:
    InitializationConfig config_;
    std::vector<NodeInitHint> hints_;
    mutable std::mt19937_64 rng_;
};

// =============================================================================
// Integrated DC Solver with Convergence Aids
// =============================================================================

/// Strategy selection for DC analysis
enum class DCStrategy {
    Direct,             // Direct Newton solve
    GminStepping,       // Use Gmin stepping
    SourceStepping,     // Use source stepping
    PseudoTransient,    // Use pseudo-transient
    Auto               // Try strategies in order until one works
};

/// DC solver configuration
struct DCConvergenceConfig {
    DCStrategy strategy = DCStrategy::Auto;
    GminConfig gmin_config;
    SourceSteppingConfig source_config;
    PseudoTransientConfig pseudo_config;
    InitializationConfig init_config;
    bool enable_random_restart = true;
    int max_strategy_attempts = 3;
};

/// Result of DC analysis with convergence aids
struct DCAnalysisResult {
    NewtonResult newton_result;
    DCStrategy strategy_used = DCStrategy::Direct;
    int random_restarts = 0;
    int total_newton_iterations = 0;
    bool success = false;
    std::string message;
};

/// Integrated DC solver with automatic strategy selection
template<LinearSolverPolicy LinearPolicy = SparseLUPolicy>
class DCConvergenceSolver {
public:
    using SystemFunction = typename NewtonRaphsonSolver<LinearPolicy>::SystemFunction;
    using ScaledSystemFunction = std::function<void(const Vector& x, Vector& f,
                                                      SparseMatrix& J, Real scale)>;

    explicit DCConvergenceSolver(const DCConvergenceConfig& config = {})
        : config_(config),
          gmin_(config.gmin_config),
          source_(config.source_config),
          pseudo_(config.pseudo_config),
          init_(config.init_config) {}

    /// Solve DC operating point with automatic convergence aids
    [[nodiscard]] DCAnalysisResult solve(
        const Vector& x0,
        Index num_nodes,
        Index num_branches,
        SystemFunction system_func,
        ScaledSystemFunction scaled_func = nullptr) {

        DCAnalysisResult result;

        // Generate initial guess
        Vector x_init = init_.warm_start(x0, num_nodes, num_branches);

        // Configure Newton solver
        NewtonOptions newton_opts;
        newton_opts.num_nodes = num_nodes;
        newton_opts.num_branches = num_branches;
        NewtonRaphsonSolver<LinearPolicy> newton(newton_opts);

        // Strategy selection
        if (config_.strategy == DCStrategy::Auto) {
            // Try strategies in order
            result = try_direct(x_init, newton, system_func);
            if (result.success) return result;

            result = try_gmin(x_init, num_nodes, newton, system_func);
            if (result.success) return result;

            if (scaled_func) {
                result = try_source_stepping(x_init, scaled_func);
                if (result.success) return result;
            }

            result = try_pseudo_transient(x_init, num_nodes, newton, system_func);
            if (result.success) return result;

            // Random restart if all strategies fail
            if (config_.enable_random_restart) {
                result = try_random_restart(num_nodes, num_branches, newton, system_func);
            }
        } else {
            // Use specified strategy
            switch (config_.strategy) {
                case DCStrategy::Direct:
                    result = try_direct(x_init, newton, system_func);
                    break;
                case DCStrategy::GminStepping:
                    result = try_gmin(x_init, num_nodes, newton, system_func);
                    break;
                case DCStrategy::SourceStepping:
                    if (scaled_func) {
                        result = try_source_stepping(x_init, scaled_func);
                    } else {
                        result.message = "Source stepping requires scaled system function";
                    }
                    break;
                case DCStrategy::PseudoTransient:
                    result = try_pseudo_transient(x_init, num_nodes, newton, system_func);
                    break;
                default:
                    break;
            }
        }

        return result;
    }

    [[nodiscard]] const DCConvergenceConfig& config() const { return config_; }
    void set_config(const DCConvergenceConfig& config) {
        config_ = config;
        gmin_.set_config(config.gmin_config);
        source_.set_config(config.source_config);
        pseudo_.set_config(config.pseudo_config);
        init_.set_config(config.init_config);
    }

    [[nodiscard]] RobustInitialization& initialization() { return init_; }
    [[nodiscard]] const RobustInitialization& initialization() const { return init_; }

private:
    DCConvergenceConfig config_;
    GminStepping gmin_;
    SourceStepping source_;
    PseudoTransientContinuation pseudo_;
    RobustInitialization init_;

    DCAnalysisResult try_direct(
        const Vector& x0,
        NewtonRaphsonSolver<LinearPolicy>& newton,
        SystemFunction& system_func) {

        DCAnalysisResult result;
        result.strategy_used = DCStrategy::Direct;
        result.newton_result = newton.solve(x0, system_func);
        result.total_newton_iterations = result.newton_result.iterations;
        result.success = result.newton_result.success();
        result.message = result.success ? "Direct solve succeeded" :
                         "Direct solve failed: " + result.newton_result.error_message;
        return result;
    }

    DCAnalysisResult try_gmin(
        const Vector& x0,
        Index num_nodes,
        NewtonRaphsonSolver<LinearPolicy>& newton,
        SystemFunction& system_func) {

        DCAnalysisResult result;
        result.strategy_used = DCStrategy::GminStepping;

        auto gmin_solve = [&](const Vector& x_start) -> NewtonResult {
            // Modify system function to include current Gmin
            Real current_gmin = gmin_.current_gmin();
            auto modified_func = [&](const Vector& x, Vector& f, SparseMatrix& J) {
                system_func(x, f, J);
                // Add Gmin to diagonal
                for (Index i = 0; i < num_nodes; ++i) {
                    J.coeffRef(i, i) += current_gmin;
                }
            };
            return newton.solve(x_start, modified_func);
        };

        result.newton_result = gmin_.execute(x0, num_nodes, gmin_solve);
        result.total_newton_iterations = result.newton_result.iterations;
        result.success = result.newton_result.success();
        result.message = result.success ? "Gmin stepping succeeded" :
                         "Gmin stepping failed: " + result.newton_result.error_message;
        return result;
    }

    DCAnalysisResult try_source_stepping(
        const Vector& x0,
        ScaledSystemFunction& scaled_func) {

        DCAnalysisResult result;
        result.strategy_used = DCStrategy::SourceStepping;

        auto scaled_solve = [&](const Vector& x_start, Real scale) -> NewtonResult {
            NewtonOptions opts;
            NewtonRaphsonSolver<LinearPolicy> newton(opts);
            auto system = [&](const Vector& x, Vector& f, SparseMatrix& J) {
                scaled_func(x, f, J, scale);
            };
            return newton.solve(x_start, system);
        };

        auto ss_result = source_.execute(x0, scaled_solve);
        result.newton_result = ss_result.final_result;
        result.total_newton_iterations = ss_result.total_newton_iterations;
        result.success = ss_result.success;
        result.message = ss_result.success ? "Source stepping succeeded" :
                         "Source stepping failed: " + ss_result.error_message;
        return result;
    }

    DCAnalysisResult try_pseudo_transient(
        const Vector& x0,
        Index num_nodes,
        NewtonRaphsonSolver<LinearPolicy>& newton,
        SystemFunction& system_func) {

        DCAnalysisResult result;
        result.strategy_used = DCStrategy::PseudoTransient;

        Vector x_prev = x0;
        auto ptc_solve = [&](const Vector& x_start, Real pseudo_dt) -> NewtonResult {
            auto modified_func = [&](const Vector& x, Vector& f, SparseMatrix& J) {
                system_func(x, f, J);
                pseudo_.add_pseudo_capacitance(num_nodes, pseudo_dt, J, f, x, x_prev);
            };
            auto res = newton.solve(x_start, modified_func);
            if (res.success()) {
                x_prev = res.solution;
            }
            return res;
        };

        auto ptc_result = pseudo_.execute(x0, num_nodes, ptc_solve);
        result.newton_result.solution = ptc_result.solution;
        result.newton_result.status = ptc_result.success ?
            SolverStatus::Success : SolverStatus::MaxIterationsReached;
        result.newton_result.final_residual = ptc_result.final_residual;
        result.total_newton_iterations = ptc_result.total_newton_iterations;
        result.success = ptc_result.success;
        result.message = ptc_result.success ? "Pseudo-transient succeeded" :
                         "Pseudo-transient failed: " + ptc_result.error_message;
        return result;
    }

    DCAnalysisResult try_random_restart(
        Index num_nodes,
        Index num_branches,
        NewtonRaphsonSolver<LinearPolicy>& newton,
        SystemFunction& system_func) {

        DCAnalysisResult result;
        result.strategy_used = DCStrategy::Direct;

        init_.reset_random_seed();

        for (int i = 0; i < config_.init_config.max_random_restarts; ++i) {
            Vector x_rand = init_.random_initial_guess(num_nodes, num_branches);
            result.newton_result = newton.solve(x_rand, system_func);
            result.total_newton_iterations += result.newton_result.iterations;
            result.random_restarts = i + 1;

            if (result.newton_result.success()) {
                result.success = true;
                result.message = "Random restart succeeded on attempt " +
                                 std::to_string(i + 1);
                return result;
            }
        }

        result.success = false;
        result.message = "All random restarts failed";
        return result;
    }
};

} // namespace pulsim::v2
