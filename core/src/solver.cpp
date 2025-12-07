#include "pulsim/solver.hpp"
#include <cmath>
#include <iostream>

namespace pulsim {

void LinearSolver::analyze_pattern(const SparseMatrix& A) {
    solver_.analyzePattern(A);
    pattern_analyzed_ = true;
    factorized_ = false;
}

bool LinearSolver::factorize(const SparseMatrix& A) {
    if (!pattern_analyzed_) {
        analyze_pattern(A);
    }

    solver_.factorize(A);

    if (solver_.info() != Eigen::Success) {
        is_singular_ = true;
        return false;
    }

    is_singular_ = false;
    factorized_ = true;
    return true;
}

LinearSolveResult LinearSolver::solve(const Vector& b) {
    LinearSolveResult result;

    if (!factorized_) {
        result.status = SolverStatus::SingularMatrix;
        result.error_message = "Matrix not factorized";
        return result;
    }

    result.x = solver_.solve(b);

    if (solver_.info() != Eigen::Success) {
        result.status = SolverStatus::NumericalError;
        result.error_message = "Linear solve failed";
        return result;
    }

    result.status = SolverStatus::Success;
    return result;
}

LinearSolveResult LinearSolver::solve(const SparseMatrix& A, const Vector& b) {
    LinearSolveResult result;

    if (!factorize(A)) {
        result.status = SolverStatus::SingularMatrix;
        result.error_message = "Matrix is singular or near-singular";
        return result;
    }

    return solve(b);
}

NewtonResult NewtonSolver::solve(const Vector& x0, SystemFunction system_func) {
    NewtonResult result;
    result.x = x0;

    Index n = x0.size();
    Vector f(n);
    SparseMatrix J(n, n);
    Vector dx(n);

    Real damping = options_.damping;

    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        // Evaluate system: compute f(x) and Jacobian J(x)
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

        // Apply update first, then check convergence
        // (checking before update caused premature convergence)

        // Apply damping if enabled
        if (options_.auto_damping) {
            // Simple line search: reduce damping if residual increases
            Vector x_new = result.x + damping * dx;
            Vector f_new(n);
            SparseMatrix J_dummy(n, n);
            system_func(x_new, f_new, J_dummy);

            Real f_new_norm = f_new.norm();

            // If residual increased, reduce damping
            while (f_new_norm > f_norm && damping > 0.01) {
                damping *= 0.5;
                x_new = result.x + damping * dx;
                system_func(x_new, f_new, J_dummy);
                f_new_norm = f_new.norm();
            }

            result.x = x_new;

            // Gradually restore damping
            damping = std::min(damping * 1.5, options_.damping);
        } else {
            result.x += damping * dx;
        }
    }

    result.status = SolverStatus::MaxIterationsReached;
    result.error_message = "Newton iteration did not converge after " +
                          std::to_string(options_.max_iterations) + " iterations. " +
                          "Final residual: " + std::to_string(result.final_residual);
    return result;
}

}  // namespace pulsim
