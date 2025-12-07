#include "pulsim/advanced_solver.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#ifdef PULSIM_HAS_SUNDIALS
#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sundials/sundials_types.h>
#endif

namespace pulsim {

// =============================================================================
// TimestepController Implementation
// =============================================================================

Real TimestepController::estimate_error(const Vector& x_high, const Vector& x_low) const {
    // Compute weighted RMS norm of the difference
    // error = sqrt(sum((x_high - x_low)^2 / (atol + rtol * |x_high|)^2) / n)

    Real sum_sq = 0.0;
    Index n = x_high.size();

    for (Index i = 0; i < n; ++i) {
        Real scale = options_.atol + options_.rtol * std::abs(x_high(i));
        Real diff = (x_high(i) - x_low(i)) / scale;
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq / static_cast<Real>(n));
}

TimestepResult TimestepController::compute_new_timestep(Real dt_current, Real error,
                                                         Real dt_min, Real dt_max) {
    TimestepResult result;
    result.error_estimate = error;

    // Accept if error <= 1 (within tolerance)
    result.accepted = (error <= 1.0);

    if (result.accepted) {
        rejection_count_ = 0;

        // Compute optimal timestep using PI controller
        // dt_new = safety * dt_current * (1/error)^(1/p) * (prev_error/error)^(1/q)
        // For second-order methods: p = 3, q = 3

        Real error_ratio = std::max(error, 1e-10);
        Real factor = options_.safety_factor * std::pow(1.0 / error_ratio, 1.0 / 3.0);

        // Include integral term for smoother control
        if (prev_error_ > 1e-10) {
            factor *= std::pow(prev_error_ / error_ratio, 1.0 / 3.0);
        }

        // Clamp factor
        factor = std::max(options_.min_factor, std::min(factor, options_.max_factor));

        result.dt_new = dt_current * factor;
        result.dt_new = std::max(dt_min, std::min(result.dt_new, dt_max));

        prev_error_ = error;
    } else {
        rejection_count_++;
        result.rejection_count = rejection_count_;

        // Reduce timestep more aggressively
        Real factor = options_.safety_factor * std::pow(1.0 / error, 1.0 / 3.0);
        factor = std::max(options_.min_factor, std::min(factor, 0.5));

        result.dt_new = dt_current * factor;
        result.dt_new = std::max(dt_min, result.dt_new);

        // Check for too many rejections
        if (rejection_count_ > options_.max_rejects) {
            result.dt_new = dt_min;  // Fall back to minimum
        }
    }

    return result;
}

// =============================================================================
// Companion Coefficients for Integration Methods
// =============================================================================

CompanionCoefficients get_companion_coefficients(IntegrationMethod method, Real dt, Real dt_prev) {
    CompanionCoefficients coef;

    switch (method) {
        case IntegrationMethod::BackwardEuler:
            // Backward Euler: y_n = y_{n-1} + dt * f(y_n)
            // For capacitor: I_n = C/dt * (V_n - V_{n-1})
            // Companion: G = C/dt, I_eq = C/dt * V_{n-1}
            coef.alpha = 1.0;
            coef.beta = 1.0;
            coef.gamma = 0.0;
            break;

        case IntegrationMethod::Trapezoidal:
        case IntegrationMethod::GEAR2:
            // Trapezoidal: y_n = y_{n-1} + dt/2 * (f(y_n) + f(y_{n-1}))
            // For capacitor: I_n = 2C/dt * V_n - 2C/dt * V_{n-1} - I_{n-1}
            // Companion: G = 2C/dt, I_eq = 2C/dt * V_{n-1} + I_{n-1}
            coef.alpha = 2.0;
            coef.beta = 2.0;
            coef.gamma = 0.0;
            break;

        case IntegrationMethod::BDF2:
            // BDF2: y_n = 4/3 * y_{n-1} - 1/3 * y_{n-2} + 2/3 * dt * f(y_n)
            // Variable step: coefficients depend on step ratio r = dt / dt_prev
            if (dt_prev > 0) {
                Real r = dt / dt_prev;
                Real denom = 1.0 + 2.0 * r;
                coef.alpha = (1.0 + r) / denom;                    // multiplies f(y_n)
                coef.beta = (1.0 + r) * (1.0 + r) / denom;         // multiplies y_{n-1}
                coef.gamma = r * r / denom;                         // multiplies y_{n-2}
            } else {
                // Fall back to backward Euler for first step
                coef.alpha = 1.0;
                coef.beta = 1.0;
                coef.gamma = 0.0;
            }
            break;

        default:
            // Default to backward Euler
            coef.alpha = 1.0;
            coef.beta = 1.0;
            coef.gamma = 0.0;
            break;
    }

    return coef;
}

// =============================================================================
// AdvancedLinearSolver Implementation
// =============================================================================

AdvancedLinearSolver::AdvancedLinearSolver(const Options& opts)
    : options_(opts)
{
#ifdef PULSIM_HAS_KLU
    klu_defaults(&klu_common_);
#endif
}

AdvancedLinearSolver::~AdvancedLinearSolver() {
#ifdef PULSIM_HAS_KLU
    cleanup_klu();
#endif
}

#ifdef PULSIM_HAS_KLU
void AdvancedLinearSolver::cleanup_klu() {
    if (klu_numeric_) {
        klu_free_numeric(&klu_numeric_, &klu_common_);
        klu_numeric_ = nullptr;
    }
    if (klu_symbolic_) {
        klu_free_symbolic(&klu_symbolic_, &klu_common_);
        klu_symbolic_ = nullptr;
    }
}

bool AdvancedLinearSolver::factorize_klu(const SparseMatrix& A) {
    n_ = A.rows();

    // Convert Eigen CSC to KLU format
    Ap_.resize(n_ + 1);
    Ai_.resize(A.nonZeros());
    Ax_.resize(A.nonZeros());

    // Copy column pointers
    for (Index j = 0; j <= n_; ++j) {
        Ap_[j] = A.outerIndexPtr()[j];
    }

    // Copy row indices and values
    for (Index k = 0; k < A.nonZeros(); ++k) {
        Ai_[k] = A.innerIndexPtr()[k];
        Ax_[k] = A.valuePtr()[k];
    }

    // Symbolic analysis (only if pattern changed or first time)
    if (!klu_symbolic_ || !pattern_analyzed_) {
        cleanup_klu();
        klu_symbolic_ = klu_analyze(n_, Ap_.data(), Ai_.data(), &klu_common_);
        if (!klu_symbolic_) {
            is_singular_ = true;
            return false;
        }
    }

    // Numeric factorization
    if (klu_numeric_) {
        // Try to refactor (reuse symbolic)
        int ok = klu_refactor(Ap_.data(), Ai_.data(), Ax_.data(),
                              klu_symbolic_, klu_numeric_, &klu_common_);
        if (!ok) {
            // Refactor failed, try full factorization
            klu_free_numeric(&klu_numeric_, &klu_common_);
            klu_numeric_ = nullptr;
        }
    }

    if (!klu_numeric_) {
        klu_numeric_ = klu_factor(Ap_.data(), Ai_.data(), Ax_.data(),
                                   klu_symbolic_, &klu_common_);
        factorization_count_++;
    } else {
        reuse_count_++;
    }

    if (!klu_numeric_) {
        is_singular_ = true;
        return false;
    }

    is_singular_ = false;
    return true;
}

LinearSolveResult AdvancedLinearSolver::solve_klu(const Vector& b) {
    LinearSolveResult result;

    if (!klu_numeric_) {
        result.status = SolverStatus::SingularMatrix;
        result.error_message = "KLU: Matrix not factorized";
        return result;
    }

    // Copy b to result and solve in-place
    result.x = b;

    int ok = klu_solve(klu_symbolic_, klu_numeric_, n_, 1,
                       result.x.data(), &klu_common_);

    if (!ok) {
        result.status = SolverStatus::NumericalError;
        result.error_message = "KLU solve failed";
        return result;
    }

    result.status = SolverStatus::Success;
    return result;
}
#endif // PULSIM_HAS_KLU

void AdvancedLinearSolver::analyze_pattern(const SparseMatrix& A) {
#ifdef PULSIM_HAS_KLU
    if (options_.backend == Backend::KLU) {
        // Pattern analysis is done in factorize_klu
        pattern_analyzed_ = false;  // Will be set true after factorization
        return;
    }
#endif

    eigen_solver_.analyzePattern(A);
    pattern_analyzed_ = true;
    factorized_ = false;
    reuse_count_ = 0;
}

bool AdvancedLinearSolver::needs_refactorization(const SparseMatrix& A_new) const {
    if (!factorized_ || !options_.reuse_factorization) {
        return true;
    }

    if (reuse_count_ >= options_.max_reuses) {
        return true;
    }

    // Check if matrix changed significantly
    if (A_prev_.rows() != A_new.rows() || A_prev_.cols() != A_new.cols()) {
        return true;
    }

    if (A_prev_.nonZeros() != A_new.nonZeros()) {
        return true;
    }

    // Compute relative change in matrix values
    Real norm_diff = 0.0;
    Real norm_old = 0.0;

    for (Index k = 0; k < A_new.nonZeros(); ++k) {
        Real diff = A_new.valuePtr()[k] - A_prev_.valuePtr()[k];
        norm_diff += diff * diff;
        norm_old += A_prev_.valuePtr()[k] * A_prev_.valuePtr()[k];
    }

    if (norm_old > 0) {
        Real rel_change = std::sqrt(norm_diff / norm_old);
        return rel_change > options_.refactor_threshold;
    }

    return true;
}

bool AdvancedLinearSolver::factorize(const SparseMatrix& A) {
#ifdef PULSIM_HAS_KLU
    if (options_.backend == Backend::KLU) {
        bool success = factorize_klu(A);
        if (success) {
            pattern_analyzed_ = true;
            factorized_ = true;
            A_prev_ = A;
        }
        return success;
    }
#endif

    if (!pattern_analyzed_) {
        analyze_pattern(A);
    }

    // Check if we can reuse factorization
    if (options_.reuse_factorization && factorized_ && !needs_refactorization(A)) {
        reuse_count_++;
        // For Eigen SparseLU, we can't truly reuse - we'd need to refactorize
        // This is a limitation; KLU is better at refactorization
    }

    eigen_solver_.factorize(A);

    if (eigen_solver_.info() != Eigen::Success) {
        is_singular_ = true;
        return false;
    }

    is_singular_ = false;
    factorized_ = true;
    factorization_count_++;
    A_prev_ = A;

    return true;
}

LinearSolveResult AdvancedLinearSolver::solve(const Vector& b) {
#ifdef PULSIM_HAS_KLU
    if (options_.backend == Backend::KLU) {
        return solve_klu(b);
    }
#endif

    LinearSolveResult result;

    if (!factorized_) {
        result.status = SolverStatus::SingularMatrix;
        result.error_message = "Matrix not factorized";
        return result;
    }

    result.x = eigen_solver_.solve(b);

    if (eigen_solver_.info() != Eigen::Success) {
        result.status = SolverStatus::NumericalError;
        result.error_message = "Linear solve failed";
        return result;
    }

    result.status = SolverStatus::Success;
    return result;
}

LinearSolveResult AdvancedLinearSolver::solve(const SparseMatrix& A, const Vector& b) {
    if (!factorize(A)) {
        LinearSolveResult result;
        result.status = SolverStatus::SingularMatrix;
        result.error_message = "Matrix is singular or near-singular";
        return result;
    }

    return solve(b);
}

// =============================================================================
// AdvancedNewtonSolver Implementation
// =============================================================================

Real AdvancedNewtonSolver::line_search(const Vector& x, const Vector& dx, const Vector& f,
                                        SystemFunction system_func) {
    Real alpha = 1.0;
    Real f_norm = f.squaredNorm();

    Vector x_new(x.size());
    Vector f_new(x.size());
    SparseMatrix J_dummy(x.size(), x.size());

    for (int i = 0; i < options_.max_line_search_iters; ++i) {
        x_new = x + alpha * dx;
        system_func(x_new, f_new, J_dummy);

        Real f_new_norm = f_new.squaredNorm();

        // Armijo condition: f(x + alpha*dx)^2 <= f(x)^2 + 2*alpha*c*f(x)^T*J*dx
        // Simplified: f_new^2 <= f^2 * (1 - 2*alpha*c)
        if (f_new_norm <= f_norm * (1.0 - 2.0 * options_.line_search_alpha * alpha)) {
            return alpha;
        }

        alpha *= options_.line_search_beta;
    }

    return alpha;  // Return best found even if condition not satisfied
}

NewtonResult AdvancedNewtonSolver::solve(const Vector& x0, SystemFunction system_func) {
    NewtonResult result;
    result.x = x0;

    Index n = x0.size();
    Vector f(n);
    SparseMatrix J(n, n);
    Vector dx(n);

    Real damping = options_.damping;

    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        // Evaluate system
        system_func(result.x, f, J);

        // Check convergence
        Real f_norm = f.norm();
        result.final_residual = f_norm;
        result.iterations = iter + 1;

        if (f_norm < options_.abstol) {
            result.status = SolverStatus::Success;
            return result;
        }

        // Solve J * dx = -f
        auto linear_result = linear_solver_.solve(J, -f);

        if (linear_result.status != SolverStatus::Success) {
            result.status = linear_result.status;
            result.error_message = linear_result.error_message;
            return result;
        }

        dx = linear_result.x;

        // Apply update with optional line search
        Real step_size = damping;

        if (options_.use_line_search) {
            step_size = damping * line_search(result.x, dx, f, system_func);
        } else if (options_.auto_damping) {
            // Simple auto-damping
            Vector x_new = result.x + damping * dx;
            Vector f_new(n);
            SparseMatrix J_dummy(n, n);
            system_func(x_new, f_new, J_dummy);

            Real f_new_norm = f_new.norm();

            while (f_new_norm > f_norm && damping > 0.01) {
                damping *= 0.5;
                x_new = result.x + damping * dx;
                system_func(x_new, f_new, J_dummy);
                f_new_norm = f_new.norm();
            }

            step_size = damping;
            damping = std::min(damping * 1.5, options_.damping);
        }

        result.x += step_size * dx;
    }

    result.status = SolverStatus::MaxIterationsReached;
    result.error_message = "Newton iteration did not converge after " +
                          std::to_string(options_.max_iterations) + " iterations. " +
                          "Final residual: " + std::to_string(result.final_residual);
    return result;
}

NewtonResult AdvancedNewtonSolver::solve_with_continuation(
    const Vector& x0,
    std::function<void(Real param, const Vector& x, Vector& f, SparseMatrix& J)> system_func) {

    if (!options_.continuation) {
        // Wrap and call regular solve
        auto wrapped_func = [&](const Vector& x, Vector& f, SparseMatrix& J) {
            system_func(1.0, x, f, J);
        };
        return solve(x0, wrapped_func);
    }

    NewtonResult result;
    result.x = x0;

    Real param_start = options_.continuation_start;
    Real param_step = (1.0 - param_start) / options_.continuation_steps;

    for (int step = 0; step <= options_.continuation_steps; ++step) {
        Real param = param_start + step * param_step;

        auto wrapped_func = [&](const Vector& x, Vector& f, SparseMatrix& J) {
            system_func(param, x, f, J);
        };

        result = solve(result.x, wrapped_func);

        if (result.status != SolverStatus::Success) {
            result.error_message = "Continuation failed at param=" + std::to_string(param) +
                                  ": " + result.error_message;
            return result;
        }
    }

    return result;
}

// =============================================================================
// SUNDIALS IDA Solver Implementation
// =============================================================================

#ifdef PULSIM_HAS_SUNDIALS

struct SUNDIALSSolver::Impl {
    void* ida_mem = nullptr;
    N_Vector y = nullptr;
    N_Vector yp = nullptr;
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;
    SUNContext ctx = nullptr;
    Index n = 0;

    // User functions (stored here to be accessible from callbacks)
    ResidualFunction* res_func = nullptr;
    JacobianFunction* jac_func = nullptr;

    ~Impl() {
        if (LS) SUNLinSolFree(LS);
        if (J) SUNMatDestroy(J);
        if (yp) N_VDestroy(yp);
        if (y) N_VDestroy(y);
        if (ida_mem) IDAFree(&ida_mem);
        if (ctx) SUNContext_Free(&ctx);
    }
};

// Static callback functions for SUNDIALS
static int ida_residual_callback(sunrealtype t, N_Vector y, N_Vector yp, N_Vector r, void* user_data) {
    auto* impl = static_cast<SUNDIALSSolver::Impl*>(user_data);
    if (!impl || !impl->res_func) return -1;

    // Wrap N_Vectors as Eigen vectors
    Index n = impl->n;
    Eigen::Map<const Vector> y_vec(N_VGetArrayPointer(y), n);
    Eigen::Map<const Vector> yp_vec(N_VGetArrayPointer(yp), n);
    Eigen::Map<Vector> r_vec(N_VGetArrayPointer(r), n);

    Vector r_tmp(n);
    (*impl->res_func)(static_cast<Real>(t), y_vec, yp_vec, r_tmp);
    r_vec = r_tmp;

    return 0;
}

SUNDIALSSolver::SUNDIALSSolver() : impl_(std::make_unique<Impl>()) {}

SUNDIALSSolver::~SUNDIALSSolver() = default;

bool SUNDIALSSolver::initialize(Index n, const Vector& y0, const Vector& yp0) {
    impl_ = std::make_unique<Impl>();
    impl_->n = n;

    // Create SUNDIALS context
    int ret = SUNContext_Create(SUN_COMM_NULL, &impl_->ctx);
    if (ret != 0) return false;

    // Create vectors
    impl_->y = N_VNew_Serial(n, impl_->ctx);
    impl_->yp = N_VNew_Serial(n, impl_->ctx);

    if (!impl_->y || !impl_->yp) return false;

    // Copy initial conditions
    Real* y_data = N_VGetArrayPointer(impl_->y);
    Real* yp_data = N_VGetArrayPointer(impl_->yp);
    for (Index i = 0; i < n; ++i) {
        y_data[i] = y0(i);
        yp_data[i] = yp0(i);
    }

    // Create IDA solver
    impl_->ida_mem = IDACreate(impl_->ctx);
    if (!impl_->ida_mem) return false;

    // Initialize IDA
    ret = IDAInit(impl_->ida_mem, ida_residual_callback, 0.0, impl_->y, impl_->yp);
    if (ret != IDA_SUCCESS) return false;

    // Set tolerances
    ret = IDASStolerances(impl_->ida_mem, options_.rtol, options_.atol);
    if (ret != IDA_SUCCESS) return false;

    // Set user data
    impl_->res_func = &residual_func_;
    impl_->jac_func = &jacobian_func_;
    ret = IDASetUserData(impl_->ida_mem, impl_.get());
    if (ret != IDA_SUCCESS) return false;

    // Create sparse matrix and KLU solver
    // Estimate nnz as 10 * n (typical for circuit matrices)
    Index nnz_estimate = 10 * n;
    impl_->J = SUNSparseMatrix(n, n, nnz_estimate, CSC_MAT, impl_->ctx);
    if (!impl_->J) return false;

    impl_->LS = SUNLinSol_KLU(impl_->y, impl_->J, impl_->ctx);
    if (!impl_->LS) return false;

    ret = IDASetLinearSolver(impl_->ida_mem, impl_->LS, impl_->J);
    if (ret != IDA_SUCCESS) return false;

    // Set optional parameters
    if (options_.max_step_size > 0) {
        IDASetMaxStep(impl_->ida_mem, options_.max_step_size);
    }
    if (options_.init_step_size > 0) {
        IDASetInitStep(impl_->ida_mem, options_.init_step_size);
    }
    IDASetMaxNumSteps(impl_->ida_mem, options_.max_steps);
    IDASetMaxOrd(impl_->ida_mem, options_.max_order);

    return true;
}

SUNDIALSSolver::StepResult SUNDIALSSolver::solve_to(Real t_out) {
    StepResult result;

    if (!impl_ || !impl_->ida_mem) {
        result.status = SolverStatus::NumericalError;
        result.error_message = "Solver not initialized";
        return result;
    }

    sunrealtype t_actual;
    int ret = IDASolve(impl_->ida_mem, t_out, &t_actual, impl_->y, impl_->yp, IDA_NORMAL);

    result.t_actual = static_cast<Real>(t_actual);

    // Copy solution
    result.y.resize(impl_->n);
    result.yp.resize(impl_->n);
    Real* y_data = N_VGetArrayPointer(impl_->y);
    Real* yp_data = N_VGetArrayPointer(impl_->yp);
    for (Index i = 0; i < impl_->n; ++i) {
        result.y(i) = y_data[i];
        result.yp(i) = yp_data[i];
    }

    switch (ret) {
        case IDA_SUCCESS:
            result.status = SolverStatus::Success;
            break;
        case IDA_TSTOP_RETURN:
            result.status = SolverStatus::Success;
            break;
        case IDA_TOO_MUCH_WORK:
            result.status = SolverStatus::MaxIterationsReached;
            result.error_message = "IDA: Too much work";
            break;
        default:
            result.status = SolverStatus::NumericalError;
            result.error_message = "IDA solve failed with code " + std::to_string(ret);
            break;
    }

    return result;
}

void SUNDIALSSolver::reset(const Vector& y0, const Vector& yp0) {
    if (!impl_ || !impl_->ida_mem) return;

    Real* y_data = N_VGetArrayPointer(impl_->y);
    Real* yp_data = N_VGetArrayPointer(impl_->yp);
    for (Index i = 0; i < impl_->n; ++i) {
        y_data[i] = y0(i);
        yp_data[i] = yp0(i);
    }

    IDAReInit(impl_->ida_mem, 0.0, impl_->y, impl_->yp);
}

#endif // PULSIM_HAS_SUNDIALS

}  // namespace pulsim
