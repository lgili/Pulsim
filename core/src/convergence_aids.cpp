#include "pulsim/convergence_aids.hpp"
#include <cmath>
#include <iostream>

namespace pulsim {

// =============================================================================
// Gmin Stepping Implementation
// =============================================================================

void GminStepping::add_gmin(SparseMatrix& G, Index num_nodes, Real gmin) {
    // Add gmin to diagonal entries for node equations only
    // (not for branch equations like voltage sources or inductors)
    for (Index i = 0; i < num_nodes; ++i) {
        G.coeffRef(i, i) += gmin;
    }
}

NewtonResult GminStepping::solve(Index num_nodes, BuildSystemFunc /*build_func*/,
                                  JacobianFunc jacobian_func, const Vector& x0) {
    NewtonResult result;
    result.x = x0;
    gmin_steps_used_ = 0;
    final_gmin_ = 0.0;

    Real gmin = options_.gmin_initial;
    NewtonSolver newton_solver;

    // Configure Newton solver
    NewtonSolver::Options newton_opts;
    newton_opts.max_iterations = 50;
    newton_opts.abstol = 1e-9;
    newton_opts.reltol = 1e-3;
    newton_opts.auto_damping = true;
    newton_solver.set_options(newton_opts);

    // Start with large Gmin and gradually reduce
    while (gmin >= options_.gmin_final && gmin_steps_used_ < options_.max_steps) {
        final_gmin_ = gmin;

        // Create system function with Gmin added
        auto system_with_gmin = [&](const Vector& x, Vector& f, SparseMatrix& J) {
            jacobian_func(x, f, J);
            // Add Gmin to diagonal of Jacobian for node equations
            for (Index i = 0; i < num_nodes && i < J.rows(); ++i) {
                J.coeffRef(i, i) += gmin;
            }
            // Also add Gmin contribution to residual: f += gmin * x for node voltages
            for (Index i = 0; i < num_nodes && i < f.size(); ++i) {
                f(i) += gmin * x(i);
            }
        };

        // Try to solve with current Gmin
        auto step_result = newton_solver.solve(result.x, system_with_gmin);

        if (step_result.status == SolverStatus::Success) {
            result.x = step_result.x;
            result.iterations += step_result.iterations;

            // Check if we can proceed to smaller Gmin
            if (gmin <= options_.gmin_final) {
                // We've reached target Gmin, done!
                result.status = SolverStatus::Success;
                return result;
            }

            // Reduce Gmin for next iteration
            gmin /= options_.reduction_factor;
            gmin_steps_used_++;
        } else {
            // Failed at this Gmin level - try smaller reduction
            gmin /= std::sqrt(options_.reduction_factor);
            if (gmin < options_.gmin_final) {
                gmin = options_.gmin_final;
            }
        }
    }

    // Final solve without Gmin (or minimal Gmin)
    auto final_result = newton_solver.solve(result.x, jacobian_func);

    if (final_result.status == SolverStatus::Success) {
        result.x = final_result.x;
        result.iterations += final_result.iterations;
        result.status = SolverStatus::Success;
        result.final_residual = final_result.final_residual;
    } else {
        // Even with Gmin stepping, couldn't converge
        result.status = final_result.status;
        result.error_message = "Gmin stepping failed to achieve convergence. " +
                               final_result.error_message;
        result.final_residual = final_result.final_residual;
    }

    return result;
}

// =============================================================================
// Source Stepping Implementation
// =============================================================================

NewtonResult SourceStepping::solve(ScaledSystemFunc /*build_func*/,
                                    ScaledJacobianFunc jacobian_func,
                                    const Vector& x0) {
    NewtonResult result;
    result.x = x0;
    source_steps_used_ = 0;
    final_factor_ = 0.0;

    NewtonSolver newton_solver;
    NewtonSolver::Options newton_opts;
    newton_opts.max_iterations = 50;
    newton_opts.abstol = 1e-9;
    newton_opts.reltol = 1e-3;
    newton_opts.auto_damping = true;
    newton_solver.set_options(newton_opts);

    Real factor = options_.factor_initial;

    // Gradually increase source scaling from initial to 1.0
    while (factor <= 1.0 && source_steps_used_ < options_.max_steps) {
        final_factor_ = factor;

        // Create system function with scaled sources
        auto system_with_scale = [&](const Vector& x, Vector& f, SparseMatrix& J) {
            jacobian_func(factor, x, f, J);
        };

        auto step_result = newton_solver.solve(result.x, system_with_scale);

        if (step_result.status == SolverStatus::Success) {
            result.x = step_result.x;
            result.iterations += step_result.iterations;

            if (factor >= 1.0) {
                // Reached full source values
                result.status = SolverStatus::Success;
                result.final_residual = step_result.final_residual;
                return result;
            }

            // Increase factor for next step
            factor += options_.factor_increment;
            if (factor > 1.0) factor = 1.0;
            source_steps_used_++;
        } else {
            // Failed at this level - try smaller increment
            factor -= options_.factor_increment * 0.5;
            factor += options_.factor_increment * 0.25;
            source_steps_used_++;

            if (source_steps_used_ >= options_.max_steps) {
                result.status = step_result.status;
                result.error_message = "Source stepping failed at factor " +
                                       std::to_string(final_factor_) + ". " +
                                       step_result.error_message;
                result.final_residual = step_result.final_residual;
                return result;
            }
        }
    }

    result.status = SolverStatus::Success;
    return result;
}

// =============================================================================
// Pseudo-Transient Continuation Implementation
// =============================================================================

NewtonResult PseudoTransientContinuation::solve(SystemFunc system_func,
                                                 CapacitanceFunc cap_func,
                                                 const Vector& x0) {
    NewtonResult result;
    result.x = x0;
    total_iterations_ = 0;
    tau_reductions_ = 0;
    final_tau_ = options_.tau_initial;

    Index n = x0.size();
    Vector C = cap_func(n);  // Get pseudo-capacitance vector

    Real tau = options_.tau_initial;
    Vector x_prev = x0;

    LinearSolver linear_solver;
    Vector f(n);
    SparseMatrix J(n, n);

    int iter_at_tau = 0;

    while (total_iterations_ < options_.max_iterations) {
        // Evaluate system
        system_func(result.x, f, J);

        // Add pseudo-transient term: C/tau * (x - x_prev)
        // Modified residual: f_ptc = f + C/tau * (x - x_prev)
        // Modified Jacobian: J_ptc = J + diag(C/tau)
        Vector f_ptc = f;
        SparseMatrix J_ptc = J;

        for (Index i = 0; i < n; ++i) {
            Real c_tau = C(i) / tau;
            f_ptc(i) += c_tau * (result.x(i) - x_prev(i));
            J_ptc.coeffRef(i, i) += c_tau;
        }

        // Check convergence
        Real f_norm = f.norm();  // Use original residual for convergence check
        result.final_residual = f_norm;

        if (f_norm < 1e-9) {  // Converged
            result.status = SolverStatus::Success;
            result.iterations = total_iterations_;
            return result;
        }

        // Solve linear system: J_ptc * dx = -f_ptc
        auto linear_result = linear_solver.solve(J_ptc, -f_ptc);

        if (linear_result.status != SolverStatus::Success) {
            result.status = linear_result.status;
            result.error_message = "Linear solve failed in pseudo-transient: " +
                                   linear_result.error_message;
            result.iterations = total_iterations_;
            return result;
        }

        // Update solution (this is now the "pseudo-time" step)
        x_prev = result.x;
        result.x += linear_result.x;

        total_iterations_++;
        iter_at_tau++;

        // Check if we should reduce tau
        if (iter_at_tau >= options_.iterations_per_tau && tau > options_.tau_final) {
            tau /= options_.tau_factor;
            if (tau < options_.tau_final) tau = options_.tau_final;
            final_tau_ = tau;
            tau_reductions_++;
            iter_at_tau = 0;
        }
    }

    result.status = SolverStatus::MaxIterationsReached;
    result.error_message = "Pseudo-transient continuation did not converge after " +
                           std::to_string(total_iterations_) + " iterations";
    result.iterations = total_iterations_;
    return result;
}

NewtonResult PseudoTransientContinuation::solve(SystemFunc system_func,
                                                 const Vector& x0,
                                                 Real uniform_cap) {
    auto cap_func = [uniform_cap](Index n) {
        return Vector::Constant(n, uniform_cap);
    };
    return solve(system_func, cap_func, x0);
}

}  // namespace pulsim
