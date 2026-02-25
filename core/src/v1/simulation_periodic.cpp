#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pulsim::v1 {

namespace {

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

}  // namespace

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
