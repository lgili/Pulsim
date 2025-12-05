#include "spicelab/simulation.hpp"
#include <chrono>
#include <iostream>

namespace spicelab {

Simulator::Simulator(const Circuit& circuit, const SimulationOptions& options)
    : circuit_(circuit)
    , options_(options)
    , assembler_(circuit)
    , newton_solver_()
{
    NewtonSolver::Options newton_opts;
    newton_opts.max_iterations = options.max_newton_iterations;
    newton_opts.abstol = options.abstol;
    newton_opts.reltol = options.reltol;
    newton_opts.damping = options.damping_factor;
    newton_opts.auto_damping = true;
    newton_solver_.set_options(newton_opts);
}

NewtonResult Simulator::dc_operating_point() {
    Index n = circuit_.total_variables();

    // Initial guess: all zeros
    Vector x0 = Vector::Zero(n);

    // For DC analysis: capacitors open, inductors shorted
    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        // Assemble DC system
        SparseMatrix G;
        Vector b;
        assembler_.assemble_dc(G, b);

        // Add nonlinear contributions if any
        if (assembler_.has_nonlinear()) {
            SparseMatrix J_nl;
            Vector f_nl;
            assembler_.assemble_nonlinear(J_nl, f_nl, x);
            G += J_nl;
            b += f_nl;
        }

        // f(x) = G*x - b
        f = G * x - b;
        J = G;
    };

    return newton_solver_.solve(x0, system_func);
}

void Simulator::build_system(const Vector& x, Vector& f, SparseMatrix& J,
                             Real time, Real dt, const Vector& x_prev) {
    // Assemble transient system with companion models
    Vector b;
    assembler_.assemble_transient(J, b, x_prev, dt);

    // Update source values for current time
    assembler_.evaluate_sources(b, time);

    // Add nonlinear contributions if any
    if (assembler_.has_nonlinear()) {
        SparseMatrix J_nl;
        Vector f_nl;
        assembler_.assemble_nonlinear(J_nl, f_nl, x);
        J += J_nl;
        b += f_nl;
    }

    // f(x) = J*x - b
    f = J * x - b;
}

NewtonResult Simulator::step(Real time, Real dt, const Vector& x_prev) {
    auto system_func = [this, time, dt, &x_prev](const Vector& x, Vector& f, SparseMatrix& J) {
        build_system(x, f, J, time, dt, x_prev);
    };

    // Use previous solution as initial guess
    return newton_solver_.solve(x_prev, system_func);
}

SimulationResult Simulator::run_transient() {
    return run_transient(nullptr);
}

SimulationResult Simulator::run_transient(SimulationCallback callback) {
    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    Index n = circuit_.total_variables();

    // Build signal names
    for (Index i = 0; i < n; ++i) {
        result.signal_names.push_back(circuit_.signal_name(i));
    }

    // Get initial state
    Vector x;
    if (options_.use_ic) {
        // Use specified initial conditions (zeros for now)
        x = Vector::Zero(n);
    } else {
        // Compute DC operating point
        auto dc_result = dc_operating_point();
        if (dc_result.status != SolverStatus::Success) {
            result.final_status = dc_result.status;
            result.error_message = "DC operating point failed: " + dc_result.error_message;
            return result;
        }
        x = dc_result.x;
        result.newton_iterations_total += dc_result.iterations;
    }

    // Store initial state
    Real time = options_.tstart;
    result.time.push_back(time);
    result.data.push_back(x);

    if (callback) {
        callback(time, x);
    }

    // Time stepping loop
    Real dt = options_.dt;
    int step_count = 0;

    while (time < options_.tstop) {
        // Don't overshoot tstop
        if (time + dt > options_.tstop) {
            dt = options_.tstop - time;
        }

        Real next_time = time + dt;

        // Take a step
        auto step_result = step(next_time, dt, x);

        if (step_result.status != SolverStatus::Success) {
            // Try with smaller timestep
            bool converged = false;
            Real dt_try = dt * 0.5;

            while (dt_try >= options_.dtmin && !converged) {
                next_time = time + dt_try;
                step_result = step(next_time, dt_try, x);

                if (step_result.status == SolverStatus::Success) {
                    converged = true;
                    dt = dt_try;  // Use this timestep going forward
                } else {
                    dt_try *= 0.5;
                }
            }

            if (!converged) {
                result.final_status = step_result.status;
                result.error_message = "Simulation failed at t=" + std::to_string(time) +
                                      ": " + step_result.error_message;
                break;
            }
        }

        result.newton_iterations_total += step_result.iterations;

        // Update state
        x = step_result.x;
        time = next_time;
        step_count++;

        // Store result
        result.time.push_back(time);
        result.data.push_back(x);

        if (callback) {
            callback(time, x);
        }

        // Adaptive timestep: increase if converged quickly
        if (step_result.iterations < 5 && dt < options_.dtmax) {
            dt = std::min(dt * 1.2, options_.dtmax);
        }
    }

    result.total_steps = step_count;
    result.final_status = SolverStatus::Success;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

SimulationResult simulate(const Circuit& circuit, const SimulationOptions& options) {
    Simulator sim(circuit, options);
    return sim.run_transient();
}

}  // namespace spicelab
