#pragma once

#include "pulsim/types.hpp"
#include "pulsim/solver.hpp"
#include <Eigen/SparseLU>
#include <memory>
#include <functional>
#include <vector>

// Optional KLU support
#ifdef PULSIM_HAS_KLU
#include <klu.h>
#endif

namespace pulsim {

// IntegrationMethod is already defined in types.hpp

// =============================================================================
// Timestep Controller for Adaptive Stepping
// =============================================================================

struct TimestepControllerOptions {
    Real rtol;
    Real atol;
    Real safety_factor;
    Real min_factor;
    Real max_factor;
    int max_rejects;

    TimestepControllerOptions()
        : rtol(1e-3)
        , atol(1e-9)
        , safety_factor(0.9)
        , min_factor(0.1)
        , max_factor(5.0)
        , max_rejects(10) {}
};

struct TimestepResult {
    Real dt_new;               // Suggested next timestep
    Real error_estimate;       // Local truncation error estimate
    bool accepted;             // Whether the step was accepted
    int rejection_count;       // Number of consecutive rejections
};

class TimestepController {
public:
    explicit TimestepController(const TimestepControllerOptions& opts = TimestepControllerOptions())
        : options_(opts), rejection_count_(0), prev_error_(1.0) {}

    // Estimate local truncation error using two solutions
    // x_high: higher-order solution, x_low: lower-order solution
    Real estimate_error(const Vector& x_high, const Vector& x_low) const;

    // Compute new timestep based on error estimate
    // Returns suggested dt and whether current step should be accepted
    TimestepResult compute_new_timestep(Real dt_current, Real error,
                                        Real dt_min, Real dt_max);

    // Reset controller state (e.g., after rejected step)
    void reset() { rejection_count_ = 0; prev_error_ = 1.0; }

    const TimestepControllerOptions& options() const { return options_; }
    void set_options(const TimestepControllerOptions& opts) { options_ = opts; }

private:
    TimestepControllerOptions options_;
    int rejection_count_;
    Real prev_error_;
};

// =============================================================================
// Advanced Linear Solver with KLU Support and Factorization Reuse
// =============================================================================

class AdvancedLinearSolver {
public:
    enum class Backend {
        EigenSparseLU,   // Default Eigen SparseLU
        KLU,             // SuiteSparse KLU (circuit-optimized)
    };

    struct Options {
        Backend backend;
        bool reuse_factorization;
        Real refactor_threshold;
        int max_reuses;

        Options()
            : backend(Backend::EigenSparseLU)
            , reuse_factorization(true)
            , refactor_threshold(0.1)
            , max_reuses(100) {}
    };

    explicit AdvancedLinearSolver(const Options& opts = Options());
    ~AdvancedLinearSolver();

    // Analyze sparsity pattern (call once when structure is known)
    void analyze_pattern(const SparseMatrix& A);

    // Factorize matrix (call when values change)
    // Returns true if factorization succeeded
    bool factorize(const SparseMatrix& A);

    // Solve Ax = b using previously factorized A
    LinearSolveResult solve(const Vector& b);

    // Solve Ax = b (factorize and solve in one step)
    LinearSolveResult solve(const SparseMatrix& A, const Vector& b);

    // Check if refactorization is needed based on matrix change
    bool needs_refactorization(const SparseMatrix& A_new) const;

    // Force refactorization on next solve
    void invalidate_factorization() { factorized_ = false; reuse_count_ = 0; }

    // Statistics
    int factorization_count() const { return factorization_count_; }
    int reuse_count() const { return reuse_count_; }
    bool is_singular() const { return is_singular_; }

    const Options& options() const { return options_; }
    void set_options(const Options& opts) { options_ = opts; }

private:
    Options options_;
    bool pattern_analyzed_ = false;
    bool factorized_ = false;
    bool is_singular_ = false;
    int factorization_count_ = 0;
    int reuse_count_ = 0;

    // Previous matrix for change detection
    SparseMatrix A_prev_;

    // Eigen backend
    Eigen::SparseLU<SparseMatrix> eigen_solver_;

#ifdef PULSIM_HAS_KLU
    // KLU backend
    klu_symbolic* klu_symbolic_ = nullptr;
    klu_numeric* klu_numeric_ = nullptr;
    klu_common klu_common_;
    std::vector<int> Ap_, Ai_;
    std::vector<double> Ax_;
    Index n_ = 0;

    void cleanup_klu();
    bool factorize_klu(const SparseMatrix& A);
    LinearSolveResult solve_klu(const Vector& b);
#endif
};

// =============================================================================
// Trapezoidal (GEAR-2) Integration Companion Models
// =============================================================================

// Companion model coefficients for different integration methods
struct CompanionCoefficients {
    Real alpha;    // Coefficient for current state contribution
    Real beta;     // Coefficient for previous state contribution
    Real gamma;    // Coefficient for second previous state (for BDF2)

    // For capacitor: I = alpha * C/dt * V_n - beta * C/dt * V_{n-1} + ...
    // For inductor:  V = alpha * L/dt * I_n - beta * L/dt * I_{n-1} + ...
};

// Get companion coefficients for a given integration method
CompanionCoefficients get_companion_coefficients(IntegrationMethod method, Real dt, Real dt_prev = 0);

// =============================================================================
// Advanced Newton Solver with Better Convergence
// =============================================================================

class AdvancedNewtonSolver {
public:
    struct Options {
        int max_iterations;
        Real abstol;
        Real reltol;
        Real damping;
        bool auto_damping;

        // Advanced options
        bool use_line_search;
        Real line_search_alpha;
        Real line_search_beta;
        int max_line_search_iters;

        bool continuation;
        Real continuation_start;
        int continuation_steps;

        Options()
            : max_iterations(50)
            , abstol(1e-12)
            , reltol(1e-3)
            , damping(1.0)
            , auto_damping(true)
            , use_line_search(true)
            , line_search_alpha(1e-4)
            , line_search_beta(0.5)
            , max_line_search_iters(10)
            , continuation(false)
            , continuation_start(0.1)
            , continuation_steps(5) {}
    };

    explicit AdvancedNewtonSolver(const Options& opts = Options())
        : options_(opts), linear_solver_() {}

    using SystemFunction = std::function<void(const Vector& x, Vector& f, SparseMatrix& J)>;

    NewtonResult solve(const Vector& x0, SystemFunction system_func);

    // Solve with continuation (gradually increase source strengths)
    NewtonResult solve_with_continuation(const Vector& x0,
                                         std::function<void(Real param, const Vector& x, Vector& f, SparseMatrix& J)> system_func);

    const Options& options() const { return options_; }
    void set_options(const Options& opts) { options_ = opts; }

    // Access the linear solver for configuration
    AdvancedLinearSolver& linear_solver() { return linear_solver_; }
    const AdvancedLinearSolver& linear_solver() const { return linear_solver_; }

private:
    Options options_;
    AdvancedLinearSolver linear_solver_;

    // Backtracking line search
    Real line_search(const Vector& x, const Vector& dx, const Vector& f,
                     SystemFunction system_func);
};

// =============================================================================
// SUNDIALS IDA Wrapper (if available)
// =============================================================================

#ifdef PULSIM_HAS_SUNDIALS

class SUNDIALSSolver {
public:
    struct Options {
        Real rtol = 1e-6;              // Relative tolerance
        Real atol = 1e-9;              // Absolute tolerance
        int max_steps = 10000;         // Maximum number of steps
        Real max_step_size = 0.0;      // Maximum step size (0 = no limit)
        Real init_step_size = 0.0;     // Initial step size (0 = auto)
        int max_order = 5;             // Maximum BDF order
        bool suppress_alg = true;      // Suppress algebraic variables in error test
    };

    SUNDIALSSolver();
    ~SUNDIALSSolver();

    // Initialize solver with system size and initial conditions
    bool initialize(Index n, const Vector& y0, const Vector& yp0);

    // Residual function type: F(t, y, y') = 0
    using ResidualFunction = std::function<void(Real t, const Vector& y, const Vector& yp, Vector& F)>;

    // Jacobian function type: J = dF/dy + cj * dF/dy'
    using JacobianFunction = std::function<void(Real t, Real cj, const Vector& y, const Vector& yp, SparseMatrix& J)>;

    // Set system functions
    void set_residual(ResidualFunction res_func) { residual_func_ = std::move(res_func); }
    void set_jacobian(JacobianFunction jac_func) { jacobian_func_ = std::move(jac_func); }

    // Solve to time t_out, returns solution at t_out
    struct StepResult {
        Real t_actual;       // Actual time reached
        Vector y;            // Solution at t_actual
        Vector yp;           // Derivative at t_actual
        SolverStatus status;
        std::string error_message;
    };

    StepResult solve_to(Real t_out);

    // Reset solver (keep configuration)
    void reset(const Vector& y0, const Vector& yp0);

    const Options& options() const { return options_; }
    void set_options(const Options& opts) { options_ = opts; }

private:
    Options options_;
    ResidualFunction residual_func_;
    JacobianFunction jacobian_func_;

    // SUNDIALS internal data (opaque pointer)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // PULSIM_HAS_SUNDIALS

}  // namespace pulsim
