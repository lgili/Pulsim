#pragma once

// =============================================================================
// PulsimCore v2 - High-Performance Newton Solver with Numerical Robustness
// =============================================================================
// This header provides the v2 Newton solver with:
// - Weighted norm for mixed voltage/current convergence (3.1.3)
// - Per-variable convergence checking (3.1.4)
// - Convergence history tracking (3.1.5)
// - Deterministic ordering guarantees (3.1.7)
// - Policy-based design for linear solvers
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <limits>
#include <vector>
#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>

namespace pulsim::v1 {

// =============================================================================
// Type Aliases for Eigen Types
// =============================================================================

using Scalar = double;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

// =============================================================================
// Solver Status and Result Types
// =============================================================================

enum class SolverStatus {
    Success,
    MaxIterationsReached,
    SingularMatrix,
    NumericalError,
    ConvergenceStall,
    Diverging
};

/// Convert status to string
[[nodiscard]] constexpr const char* to_string(SolverStatus status) noexcept {
    switch (status) {
        case SolverStatus::Success: return "Success";
        case SolverStatus::MaxIterationsReached: return "MaxIterationsReached";
        case SolverStatus::SingularMatrix: return "SingularMatrix";
        case SolverStatus::NumericalError: return "NumericalError";
        case SolverStatus::ConvergenceStall: return "ConvergenceStall";
        case SolverStatus::Diverging: return "Diverging";
        default: return "Unknown";
    }
}

// =============================================================================
// 3.1.5: Convergence History Tracking
// =============================================================================

/// Single iteration record for convergence analysis
struct IterationRecord {
    int iteration = 0;
    Real residual_norm = 0.0;
    Real max_voltage_error = 0.0;
    Real max_current_error = 0.0;
    Real step_norm = 0.0;
    Real damping = 1.0;
    bool converged = false;
};

/// Complete convergence history
class ConvergenceHistory {
public:
    static constexpr std::size_t max_history = 100;

    ConvergenceHistory() = default;

    void clear() {
        records_.clear();
        final_status_ = SolverStatus::Success;
    }

    void add_record(const IterationRecord& record) {
        if (records_.size() < max_history) {
            records_.push_back(record);
        }
    }

    void set_final_status(SolverStatus status) {
        final_status_ = status;
    }

    [[nodiscard]] std::size_t size() const { return records_.size(); }
    [[nodiscard]] bool empty() const { return records_.empty(); }

    [[nodiscard]] const IterationRecord& operator[](std::size_t i) const {
        return records_[i];
    }

    [[nodiscard]] const IterationRecord& last() const {
        return records_.back();
    }

    [[nodiscard]] SolverStatus final_status() const { return final_status_; }

    [[nodiscard]] auto begin() const { return records_.begin(); }
    [[nodiscard]] auto end() const { return records_.end(); }

    /// Check for convergence stall (residual not decreasing)
    [[nodiscard]] bool is_stalling(std::size_t window = 5, Real threshold = 0.9) const {
        if (records_.size() < window) return false;

        Real first_res = records_[records_.size() - window].residual_norm;
        Real last_res = records_.back().residual_norm;

        return last_res > threshold * first_res;
    }

    /// Check for divergence
    [[nodiscard]] bool is_diverging(std::size_t window = 3) const {
        if (records_.size() < window) return false;

        for (std::size_t i = records_.size() - window + 1; i < records_.size(); ++i) {
            if (records_[i].residual_norm <= records_[i-1].residual_norm) {
                return false;
            }
        }
        return true;
    }

    /// Get convergence rate (average reduction per iteration)
    [[nodiscard]] Real convergence_rate() const {
        if (records_.size() < 2) return 0.0;

        Real first = records_.front().residual_norm;
        Real last = records_.back().residual_norm;

        if (first <= 0 || last <= 0) return 0.0;

        return std::pow(last / first, 1.0 / static_cast<Real>(records_.size() - 1));
    }

private:
    std::vector<IterationRecord> records_;
    SolverStatus final_status_ = SolverStatus::Success;
};

// =============================================================================
// 3.1.4: Per-Variable Convergence Status
// =============================================================================

/// Convergence status for each variable
struct VariableConvergence {
    Index index = 0;
    Real value = 0.0;
    Real delta = 0.0;
    Real tolerance = 0.0;
    Real normalized_error = 0.0;  // |delta| / tolerance
    bool converged = false;
    bool is_voltage = true;  // true = voltage, false = current

    [[nodiscard]] static VariableConvergence voltage(Index idx, Real val, Real delta,
                                                      Real abstol, Real reltol) {
        Real tol = abstol + reltol * std::abs(val);
        Real err = std::abs(delta) / tol;
        return {idx, val, delta, tol, err, err <= 1.0, true};
    }

    [[nodiscard]] static VariableConvergence current(Index idx, Real val, Real delta,
                                                      Real abstol, Real reltol) {
        Real tol = abstol + reltol * std::abs(val);
        Real err = std::abs(delta) / tol;
        return {idx, val, delta, tol, err, err <= 1.0, false};
    }
};

/// Per-variable convergence tracker
class PerVariableConvergence {
public:
    PerVariableConvergence() = default;

    void clear() { vars_.clear(); }

    void add(const VariableConvergence& v) {
        vars_.push_back(v);
    }

    [[nodiscard]] std::size_t size() const { return vars_.size(); }
    [[nodiscard]] bool empty() const { return vars_.empty(); }

    [[nodiscard]] const VariableConvergence& operator[](std::size_t i) const {
        return vars_[i];
    }

    [[nodiscard]] auto begin() const { return vars_.begin(); }
    [[nodiscard]] auto end() const { return vars_.end(); }

    /// Check if all variables converged
    [[nodiscard]] bool all_converged() const {
        for (const auto& v : vars_) {
            if (!v.converged) return false;
        }
        return true;
    }

    /// Get the worst (highest normalized error) variable
    [[nodiscard]] const VariableConvergence* worst() const {
        if (vars_.empty()) return nullptr;

        const VariableConvergence* w = &vars_[0];
        for (const auto& v : vars_) {
            if (v.normalized_error > w->normalized_error) {
                w = &v;
            }
        }
        return w;
    }

    /// Get maximum normalized error
    [[nodiscard]] Real max_error() const {
        Real m = 0.0;
        for (const auto& v : vars_) {
            if (v.normalized_error > m) m = v.normalized_error;
        }
        return m;
    }

    /// Count of non-converged variables
    [[nodiscard]] std::size_t non_converged_count() const {
        std::size_t count = 0;
        for (const auto& v : vars_) {
            if (!v.converged) ++count;
        }
        return count;
    }

private:
    std::vector<VariableConvergence> vars_;
};

// =============================================================================
// 3.1.3: Weighted Norm Convergence Checker
// =============================================================================

/// Convergence checker with weighted norms for mixed voltage/current
class ConvergenceChecker {
public:
    /// Tolerance configuration
    struct Tolerances {
        Real voltage_abstol = 1e-9;    // Absolute tolerance for voltages (V)
        Real voltage_reltol = 1e-3;    // Relative tolerance for voltages
        Real current_abstol = 1e-12;   // Absolute tolerance for currents (A)
        Real current_reltol = 1e-3;    // Relative tolerance for currents
        Real residual_tol = 1e-9;      // Residual tolerance for F(x)

        static constexpr Tolerances defaults() {
            return Tolerances{1e-9, 1e-3, 1e-12, 1e-3, 1e-9};
        }
    };

    ConvergenceChecker() : tol_(Tolerances::defaults()) {}
    explicit ConvergenceChecker(const Tolerances& tol) : tol_(tol) {}

    /// Check convergence using weighted infinity norm
    /// Returns the maximum normalized error (converged if <= 1.0)
    [[nodiscard]] Real check_weighted_norm(
        const Vector& delta,
        const Vector& solution,
        Index num_nodes,
        Index num_branches) const {

        Real max_error = 0.0;

        // Check voltage nodes
        for (Index i = 0; i < num_nodes; ++i) {
            Real tol = tol_.voltage_abstol + tol_.voltage_reltol * std::abs(solution[i]);
            Real err = std::abs(delta[i]) / tol;
            max_error = std::max(max_error, err);
        }

        // Check current branches
        for (Index i = num_nodes; i < num_nodes + num_branches; ++i) {
            Real tol = tol_.current_abstol + tol_.current_reltol * std::abs(solution[i]);
            Real err = std::abs(delta[i]) / tol;
            max_error = std::max(max_error, err);
        }

        return max_error;
    }

    /// Check per-variable convergence
    [[nodiscard]] PerVariableConvergence check_per_variable(
        const Vector& delta,
        const Vector& solution,
        Index num_nodes,
        Index num_branches) const {

        PerVariableConvergence result;

        // Check voltage nodes
        for (Index i = 0; i < num_nodes; ++i) {
            result.add(VariableConvergence::voltage(
                i, solution[i], delta[i],
                tol_.voltage_abstol, tol_.voltage_reltol));
        }

        // Check current branches
        for (Index i = num_nodes; i < num_nodes + num_branches; ++i) {
            result.add(VariableConvergence::current(
                i, solution[i], delta[i],
                tol_.current_abstol, tol_.current_reltol));
        }

        return result;
    }

    /// Check if residual is small enough
    [[nodiscard]] bool check_residual(const Vector& f) const {
        return f.lpNorm<Eigen::Infinity>() < tol_.residual_tol;
    }

    /// Combined convergence check
    [[nodiscard]] bool has_converged(
        const Vector& delta,
        const Vector& solution,
        const Vector& residual,
        Index num_nodes,
        Index num_branches) const {

        Real weighted_error = check_weighted_norm(delta, solution, num_nodes, num_branches);
        return weighted_error <= 1.0 && check_residual(residual);
    }

    [[nodiscard]] const Tolerances& tolerances() const { return tol_; }
    void set_tolerances(const Tolerances& tol) { tol_ = tol; }

private:
    Tolerances tol_;
};

// =============================================================================
// Linear Solver Result Type (portable alternative to std::expected)
// =============================================================================

struct LinearSolveResult {
    std::optional<Vector> solution;
    std::string error;

    [[nodiscard]] bool has_value() const { return solution.has_value(); }
    [[nodiscard]] explicit operator bool() const { return has_value(); }
    [[nodiscard]] const Vector& value() const { return *solution; }
    [[nodiscard]] Vector& value() { return *solution; }
    [[nodiscard]] const Vector& operator*() const { return *solution; }
    [[nodiscard]] Vector& operator*() { return *solution; }

    static LinearSolveResult success(Vector v) {
        return {std::move(v), {}};
    }

    static LinearSolveResult failure(std::string err) {
        return {std::nullopt, std::move(err)};
    }
};

// =============================================================================
// Linear Solver Policy Concept
// =============================================================================

template<typename T>
concept LinearSolverPolicy = requires(T solver, const SparseMatrix& A, const Vector& b) {
    { solver.analyze(A) } -> std::same_as<bool>;
    { solver.factorize(A) } -> std::same_as<bool>;
    { solver.solve(b) } -> std::same_as<LinearSolveResult>;
    { solver.is_singular() } -> std::same_as<bool>;
};

// =============================================================================
// SparseLU Linear Solver Policy
// =============================================================================

class SparseLUPolicy {
public:
    SparseLUPolicy() = default;

    bool analyze(const SparseMatrix& A) {
        solver_.analyzePattern(A);
        analyzed_ = true;
        return true;
    }

    bool factorize(const SparseMatrix& A) {
        if (!analyzed_) analyze(A);
        solver_.factorize(A);
        singular_ = (solver_.info() != Eigen::Success);
        factorized_ = !singular_;
        return factorized_;
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
        if (!factorized_) {
            return LinearSolveResult::failure("Matrix not factorized");
        }

        Vector x = solver_.solve(b);

        if (solver_.info() != Eigen::Success) {
            return LinearSolveResult::failure("Linear solve failed");
        }

        return LinearSolveResult::success(std::move(x));
    }

    [[nodiscard]] bool is_singular() const { return singular_; }

private:
    Eigen::SparseLU<SparseMatrix> solver_;
    bool analyzed_ = false;
    bool factorized_ = false;
    bool singular_ = false;
};

// =============================================================================
// Newton Solver Result
// =============================================================================

struct NewtonResult {
    Vector solution;
    SolverStatus status = SolverStatus::NumericalError;
    int iterations = 0;
    Real final_residual = 0.0;
    Real final_weighted_error = 0.0;
    ConvergenceHistory history;
    PerVariableConvergence variable_convergence;
    struct NonlinearTelemetry {
        bool used_anderson = false;
        bool used_broyden = false;
        bool used_newton_krylov = false;
        int line_search_backtracks = 0;
        int trust_region_shrinks = 0;
        int trust_region_expands = 0;
        std::vector<std::string> fallback_reasons;
    } telemetry;
    std::string error_message;

    [[nodiscard]] bool success() const {
        return status == SolverStatus::Success;
    }
};

// =============================================================================
// Newton Solver Options
// =============================================================================

struct NewtonOptions {
    int max_iterations = 50;
    Real initial_damping = 1.0;
    Real min_damping = 0.01;
    bool auto_damping = true;
    bool track_history = true;
    bool check_per_variable = true;
    Index num_nodes = 0;      // For weighted norm
    Index num_branches = 0;   // For weighted norm
    ConvergenceChecker::Tolerances tolerances;

    // Anderson acceleration (optional)
    bool enable_anderson = false;
    int anderson_depth = 5;
    Real anderson_beta = 1.0;

    // Jacobian pattern reuse
    bool reuse_jacobian_pattern = true;

    // Broyden update (optional, dense for small systems)
    bool enable_broyden = false;
    int broyden_max_size = 200;

    // Newton-Krylov path (iterative linear solver)
    bool enable_newton_krylov = false;

    // Voltage/current limiting (can slow convergence in switching circuits)
    Real max_voltage_step = 5.0;      // Max voltage change per iteration [V]
    Real max_current_step = 10.0;     // Max current change per iteration [A]
    bool enable_limiting = false;     // Disabled by default - can prevent convergence at switching transitions

    // Stall detection (disable for transient simulations with stiff transitions)
    bool detect_stall = false;        // Check for convergence stall (default: disabled for robustness)
    int stall_window = 10;            // Window size for stall detection
    Real stall_threshold = 0.95;      // Threshold for stall detection (0.95 = 5% improvement required)

    // Trust-region control
    bool enable_trust_region = false;
    Real trust_radius = 10.0;
    Real trust_shrink = 0.5;
    Real trust_expand = 1.5;
    Real trust_min = 1e-6;
    Real trust_max = 1e6;
};

// =============================================================================
// Newton-Raphson Solver with Weighted Norms and History Tracking
// =============================================================================

template<LinearSolverPolicy LinearPolicy = SparseLUPolicy>
class NewtonRaphsonSolver {
public:
    using SystemFunction = std::function<void(const Vector& x, Vector& f, SparseMatrix& J)>;

    explicit NewtonRaphsonSolver(const NewtonOptions& opts = {})
        : options_(opts), convergence_checker_(opts.tolerances) {}

    /// Solve F(x) = 0 with weighted norm convergence
    [[nodiscard]] NewtonResult solve(const Vector& x0, SystemFunction system_func) {
        NewtonResult result;
        result.solution = x0;
        result.history.clear();
        anderson_f_history_.clear();

        result.telemetry.used_anderson = options_.enable_anderson;
        result.telemetry.used_broyden = options_.enable_broyden;
        result.telemetry.used_newton_krylov = options_.enable_newton_krylov;

        if constexpr (requires(LinearPolicy policy) { policy.set_force_iterative(false); }) {
            linear_solver_.set_force_iterative(options_.enable_newton_krylov);
        }

        const Index n = x0.size();
        Vector f(n);
        SparseMatrix J(n, n);
        Vector dx(n);

        Real damping = options_.initial_damping;
        Real prev_residual = std::numeric_limits<Real>::max();
        Real trust_radius = options_.trust_radius;
        bool pattern_analyzed = false;

        const bool use_broyden = options_.enable_broyden && n <= options_.broyden_max_size;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J_dense;
        Vector prev_x;
        Vector prev_f;
        bool broyden_ready = false;

        for (int iter = 0; iter < options_.max_iterations; ++iter) {
            // Evaluate system
            system_func(result.solution, f, J);
            const Vector x_current = result.solution;

            if (options_.reuse_jacobian_pattern && !pattern_analyzed) {
                if (!linear_solver_.analyze(J)) {
                    result.status = SolverStatus::NumericalError;
                    result.error_message = "Jacobian pattern analysis failed";
                    result.iterations = iter + 1;
                    result.history.set_final_status(result.status);
                    return result;
                }
                pattern_analyzed = true;
            }

            if (use_broyden) {
                if (!broyden_ready) {
                    J_dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(J);
                } else {
                    Vector s = result.solution - prev_x;
                    Vector y = f - prev_f;
                    Real denom = s.dot(s);
                    if (denom > std::numeric_limits<Real>::epsilon()) {
                        Vector Js = J_dense * s;
                        J_dense += (y - Js) * (s.transpose() / denom);
                    }
                }
            }

            // Compute residual norm
            Real f_norm = f.norm();
            result.final_residual = f_norm;

            // Record iteration
            if (options_.track_history) {
                IterationRecord record;
                record.iteration = iter;
                record.residual_norm = f_norm;
                record.damping = damping;
                result.history.add_record(record);
            }

            // Check for divergence
            if (f_norm > 1e6 * prev_residual && iter > 3) {
                result.status = SolverStatus::Diverging;
                result.error_message = "Newton iteration diverging";
                result.history.set_final_status(result.status);
                return result;
            }
            prev_residual = f_norm;

            // Solve J * dx = -f
            if (use_broyden) {
                dx = J_dense.colPivHouseholderQr().solve(-f);
                if (!dx.allFinite()) {
                    result.telemetry.fallback_reasons.push_back("BroydenFallback");
                    if (!linear_solver_.factorize(J)) {
                        result.status = SolverStatus::SingularMatrix;
                        result.error_message = "Jacobian is singular";
                        result.iterations = iter + 1;
                        result.history.set_final_status(result.status);
                        return result;
                    }
                    auto solve_result = linear_solver_.solve(-f);
                    if (!solve_result) {
                        result.status = SolverStatus::NumericalError;
                        result.error_message = solve_result.error;
                        result.iterations = iter + 1;
                        result.history.set_final_status(result.status);
                        return result;
                    }
                    dx = *solve_result;
                }
            } else {
                if (!linear_solver_.factorize(J)) {
                    result.status = SolverStatus::SingularMatrix;
                    result.error_message = "Jacobian is singular";
                    result.iterations = iter + 1;
                    result.history.set_final_status(result.status);
                    return result;
                }

                auto solve_result = linear_solver_.solve(-f);
                if (!solve_result) {
                    result.status = SolverStatus::NumericalError;
                    result.error_message = solve_result.error;
                    result.iterations = iter + 1;
                    result.history.set_final_status(result.status);
                    return result;
                }
                dx = *solve_result;
            }

            // Apply voltage/current limiting to prevent divergence
            bool limiting_active = false;
            if (options_.enable_limiting && options_.num_nodes > 0) {
                limiting_active = apply_limiting(dx, options_.num_nodes, options_.num_branches);
            }

            // Trust region scaling
            if (options_.enable_trust_region) {
                Real step_norm = dx.norm();
                if (step_norm > trust_radius && step_norm > std::numeric_limits<Real>::epsilon()) {
                    dx *= (trust_radius / step_norm);
                    result.telemetry.trust_region_shrinks += 1;
                    result.telemetry.fallback_reasons.push_back("TrustRegion");
                }
            }

            // Apply update with damping
            if (options_.auto_damping) {
                auto ls = line_search(result.solution, dx, f_norm, system_func, damping);
                damping = ls.damping;
                if (ls.backtracks > 0) {
                    result.telemetry.line_search_backtracks += ls.backtracks;
                    result.telemetry.fallback_reasons.push_back("LineSearch");
                }

                if (options_.enable_trust_region) {
                    if (ls.backtracks > 0) {
                        trust_radius = std::max(options_.trust_min, trust_radius * options_.trust_shrink);
                    } else {
                        trust_radius = std::min(options_.trust_max, trust_radius * options_.trust_expand);
                        result.telemetry.trust_region_expands += 1;
                    }
                }
            }
            Vector g = result.solution + dx * damping;
            if (options_.enable_anderson) {
                g = apply_anderson(result.solution, g);
            }
            result.solution = std::move(g);

            prev_x = x_current;
            prev_f = f;
            broyden_ready = true;

            // Check convergence with weighted norm
            if (options_.num_nodes > 0 || options_.num_branches > 0) {
                Real weighted_error = convergence_checker_.check_weighted_norm(
                    dx, result.solution, options_.num_nodes, options_.num_branches);
                result.final_weighted_error = weighted_error;

                // Per-variable convergence check
                if (options_.check_per_variable) {
                    result.variable_convergence = convergence_checker_.check_per_variable(
                        dx, result.solution, options_.num_nodes, options_.num_branches);
                }

                if (weighted_error <= 1.0 && convergence_checker_.check_residual(f)) {
                    result.status = SolverStatus::Success;
                    result.iterations = iter + 1;
                    result.history.set_final_status(result.status);
                    return result;
                }
            } else {
                // Fall back to simple norm
                if (f_norm < convergence_checker_.tolerances().residual_tol) {
                    result.status = SolverStatus::Success;
                    result.iterations = iter + 1;
                    result.history.set_final_status(result.status);
                    return result;
                }
            }

            // Check for stall (skip if limiting is actively clamping values or stall detection is disabled)
            // When limiting is active, the solver intentionally makes slow progress
            if (options_.detect_stall && options_.track_history && !limiting_active &&
                result.history.is_stalling(options_.stall_window, options_.stall_threshold)) {
                result.status = SolverStatus::ConvergenceStall;
                result.error_message = "Convergence stalled";
                result.iterations = iter + 1;
                result.history.set_final_status(result.status);
                return result;
            }
        }

        // Final check
        system_func(result.solution, f, J);
        result.final_residual = f.norm();
        result.iterations = options_.max_iterations;

        if (result.final_residual < convergence_checker_.tolerances().residual_tol) {
            result.status = SolverStatus::Success;
        } else {
            result.status = SolverStatus::MaxIterationsReached;
            result.error_message = "Max iterations reached";
        }

        result.history.set_final_status(result.status);
        return result;
    }

    [[nodiscard]] const NewtonOptions& options() const { return options_; }
    void set_options(const NewtonOptions& opts) {
        options_ = opts;
        convergence_checker_.set_tolerances(opts.tolerances);
    }

    [[nodiscard]] LinearPolicy& linear_solver() { return linear_solver_; }
    [[nodiscard]] const LinearPolicy& linear_solver() const { return linear_solver_; }

    [[nodiscard]] const ConvergenceChecker& convergence_checker() const {
        return convergence_checker_;
    }

private:
    NewtonOptions options_;
    LinearPolicy linear_solver_;
    ConvergenceChecker convergence_checker_;
    std::vector<Vector> anderson_f_history_;

    struct LineSearchResult {
        Real damping = 1.0;
        int backtracks = 0;
    };

    /// Apply voltage/current limiting to dx vector
    /// Returns true if any value was clamped (limiting is active)
    bool apply_limiting(Vector& dx, Index num_nodes, Index num_branches) {
        bool clamped = false;

        // Limit voltage changes (first num_nodes entries)
        for (Index i = 0; i < num_nodes && i < dx.size(); ++i) {
            if (dx[i] > options_.max_voltage_step) {
                dx[i] = options_.max_voltage_step;
                clamped = true;
            } else if (dx[i] < -options_.max_voltage_step) {
                dx[i] = -options_.max_voltage_step;
                clamped = true;
            }
        }

        // Limit current changes (entries after num_nodes)
        for (Index i = num_nodes; i < num_nodes + num_branches && i < dx.size(); ++i) {
            if (dx[i] > options_.max_current_step) {
                dx[i] = options_.max_current_step;
                clamped = true;
            } else if (dx[i] < -options_.max_current_step) {
                dx[i] = -options_.max_current_step;
                clamped = true;
            }
        }

        return clamped;
    }

    /// Simple backtracking line search
    LineSearchResult line_search(const Vector& x, const Vector& dx, Real f_norm,
                                 SystemFunction& system_func, Real initial_damping) {
        Real damping = initial_damping;
        const Index n = x.size();
        Vector x_new(n);
        Vector f_new(n);
        SparseMatrix J_dummy(n, n);

        x_new = x + dx * damping;
        system_func(x_new, f_new, J_dummy);
        Real f_new_norm = f_new.norm();

        // Reduce damping while residual increases
        int max_backtracks = 10;
        int bt = 0;
        while (f_new_norm > f_norm && damping > options_.min_damping && bt < max_backtracks) {
            damping *= 0.5;
            x_new = x + dx * damping;
            system_func(x_new, f_new, J_dummy);
            f_new_norm = f_new.norm();
            ++bt;
        }

        // Gradually restore damping
        LineSearchResult result;
        result.damping = std::min(damping * 1.5, options_.initial_damping);
        result.backtracks = bt;
        return result;
    }

    /// Anderson acceleration using previous update vectors
    [[nodiscard]] Vector apply_anderson(const Vector& x, const Vector& g) {
        if (options_.anderson_depth <= 0) {
            return g;
        }

        Vector f_k = g - x;
        const int history_size = static_cast<int>(anderson_f_history_.size());
        const int m = std::min(options_.anderson_depth, history_size);

        Vector accelerated = g;

        if (m > 0) {
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> F(x.size(), m);
            for (int i = 0; i < m; ++i) {
                F.col(i) = anderson_f_history_[history_size - m + i];
            }

            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> alpha = F.colPivHouseholderQr().solve(f_k);
            if (alpha.allFinite()) {
                accelerated = g - options_.anderson_beta * (F * alpha);
            }
        }

        anderson_f_history_.push_back(f_k);
        if (static_cast<int>(anderson_f_history_.size()) > options_.anderson_depth) {
            anderson_f_history_.erase(anderson_f_history_.begin());
        }

        return accelerated;
    }
};

// =============================================================================
// 3.1.7: Deterministic Ordering Utilities
// =============================================================================

/// Ensure deterministic iteration order for device/node assembly
template<typename Container, typename KeyFunc>
void sort_for_determinism(Container& items, KeyFunc key_func) {
    std::sort(items.begin(), items.end(), [&](const auto& a, const auto& b) {
        return key_func(a) < key_func(b);
    });
}

/// Device ordering key for deterministic assembly
struct DeviceOrderKey {
    std::string type_name;  // Device type (resistor, capacitor, etc.)
    std::string name;       // Instance name
    Index id = 0;           // Numeric ID

    [[nodiscard]] bool operator<(const DeviceOrderKey& other) const {
        if (type_name != other.type_name) return type_name < other.type_name;
        if (name != other.name) return name < other.name;
        return id < other.id;
    }

    [[nodiscard]] bool operator==(const DeviceOrderKey& other) const {
        return type_name == other.type_name && name == other.name && id == other.id;
    }
};

/// Node ordering for deterministic assembly
struct DeterministicNodeOrder {
    std::vector<Index> node_order;      // Maps internal index -> external index
    std::vector<Index> inverse_order;   // Maps external index -> internal index

    void set_order(Index n) {
        node_order.resize(n);
        inverse_order.resize(n);
        for (Index i = 0; i < n; ++i) {
            node_order[i] = i;
            inverse_order[i] = i;
        }
    }

    /// Apply permutation to a vector
    [[nodiscard]] Vector permute(const Vector& v) const {
        Vector result(v.size());
        for (std::size_t i = 0; i < node_order.size(); ++i) {
            result[i] = v[node_order[i]];
        }
        return result;
    }

    /// Apply inverse permutation to a vector
    [[nodiscard]] Vector unpermute(const Vector& v) const {
        Vector result(v.size());
        for (std::size_t i = 0; i < inverse_order.size(); ++i) {
            result[i] = v[inverse_order[i]];
        }
        return result;
    }

    /// Natural ordering (identity)
    [[nodiscard]] static DeterministicNodeOrder natural(Index n) {
        DeterministicNodeOrder order;
        order.set_order(n);
        return order;
    }

    /// Sorted ordering by node name/ID
    template<typename NodeContainer, typename KeyFunc>
    [[nodiscard]] static DeterministicNodeOrder sorted(const NodeContainer& nodes, KeyFunc key_func) {
        DeterministicNodeOrder order;
        Index n = static_cast<Index>(nodes.size());
        order.node_order.resize(n);
        order.inverse_order.resize(n);

        // Create index pairs
        std::vector<std::pair<decltype(key_func(nodes[0])), Index>> pairs;
        pairs.reserve(n);
        for (Index i = 0; i < n; ++i) {
            pairs.emplace_back(key_func(nodes[i]), i);
        }

        // Sort by key
        std::sort(pairs.begin(), pairs.end());

        // Build ordering
        for (Index i = 0; i < n; ++i) {
            order.node_order[i] = pairs[i].second;
            order.inverse_order[pairs[i].second] = i;
        }

        return order;
    }

    /// Reverse Cuthill-McKee ordering (for bandwidth reduction)
    [[nodiscard]] static DeterministicNodeOrder rcm(const SparseMatrix& A) {
        Index n = A.rows();
        DeterministicNodeOrder order;
        order.node_order.resize(n);
        order.inverse_order.resize(n);

        // Build adjacency information
        std::vector<std::vector<Index>> adj(n);
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
                if (it.row() != it.col()) {
                    adj[it.row()].push_back(it.col());
                }
            }
        }

        // Sort adjacencies for determinism
        for (auto& a : adj) {
            std::sort(a.begin(), a.end());
        }

        // Find starting node (minimum degree)
        Index start = 0;
        std::size_t min_degree = adj[0].size();
        for (Index i = 1; i < n; ++i) {
            if (adj[i].size() < min_degree) {
                min_degree = adj[i].size();
                start = i;
            }
        }

        // BFS for Cuthill-McKee
        std::vector<bool> visited(n, false);
        std::vector<Index> result;
        result.reserve(n);

        std::vector<Index> queue;
        queue.push_back(start);
        visited[start] = true;

        while (!queue.empty()) {
            // Sort queue by degree for determinism
            std::sort(queue.begin(), queue.end(), [&adj](Index a, Index b) {
                return adj[a].size() < adj[b].size();
            });

            Index current = queue.front();
            queue.erase(queue.begin());
            result.push_back(current);

            // Add unvisited neighbors
            for (Index neighbor : adj[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        // Handle disconnected components
        for (Index i = 0; i < n; ++i) {
            if (!visited[i]) {
                result.push_back(i);
            }
        }

        // Reverse for RCM
        std::reverse(result.begin(), result.end());

        // Build ordering
        for (Index i = 0; i < n; ++i) {
            order.node_order[i] = result[i];
            order.inverse_order[result[i]] = i;
        }

        return order;
    }
};

/// Assembly order tracker for deterministic matrix construction
class DeterministicAssemblyOrder {
public:
    DeterministicAssemblyOrder() = default;

    /// Register a device for assembly
    void register_device(const DeviceOrderKey& key) {
        devices_.push_back(key);
        sorted_ = false;
    }

    /// Sort devices for deterministic iteration
    void sort() {
        if (!sorted_) {
            std::sort(devices_.begin(), devices_.end());
            sorted_ = true;
        }
    }

    /// Get sorted device order
    [[nodiscard]] const std::vector<DeviceOrderKey>& devices() const {
        return devices_;
    }

    /// Clear all registered devices
    void clear() {
        devices_.clear();
        sorted_ = false;
    }

    /// Check if sorted
    [[nodiscard]] bool is_sorted() const { return sorted_; }

    /// Get device count
    [[nodiscard]] std::size_t size() const { return devices_.size(); }

    /// Iterator support
    [[nodiscard]] auto begin() const { return devices_.begin(); }
    [[nodiscard]] auto end() const { return devices_.end(); }

private:
    std::vector<DeviceOrderKey> devices_;
    bool sorted_ = false;
};

/// Triplet ordering for deterministic sparse matrix construction
struct DeterministicTriplet {
    Index row;
    Index col;
    Real value;

    [[nodiscard]] bool operator<(const DeterministicTriplet& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

/// Build sparse matrix from triplets in deterministic order
[[nodiscard]] inline SparseMatrix build_matrix_deterministic(
    Index rows, Index cols,
    std::vector<DeterministicTriplet>& triplets) {

    // Sort triplets by (row, col) for deterministic assembly
    std::sort(triplets.begin(), triplets.end());

    // Combine duplicates (same row, col)
    std::vector<Eigen::Triplet<Real>> eigen_triplets;
    eigen_triplets.reserve(triplets.size());

    for (std::size_t i = 0; i < triplets.size(); ) {
        Index r = triplets[i].row;
        Index c = triplets[i].col;
        Real sum = 0.0;

        // Sum all values at same (row, col)
        while (i < triplets.size() && triplets[i].row == r && triplets[i].col == c) {
            sum += triplets[i].value;
            ++i;
        }

        eigen_triplets.emplace_back(r, c, sum);
    }

    SparseMatrix result(rows, cols);
    result.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
    return result;
}

// =============================================================================
// Static Assertions
// =============================================================================

static_assert(LinearSolverPolicy<SparseLUPolicy>);

} // namespace pulsim::v1
