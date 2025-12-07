#pragma once

#include "pulsim/types.hpp"
#include <Eigen/SparseLU>
#include <optional>

namespace pulsim {

// Result of a linear solve
struct LinearSolveResult {
    Vector x;
    SolverStatus status;
    std::string error_message;
};

// Result of Newton iteration
struct NewtonResult {
    Vector x;
    SolverStatus status = SolverStatus::NumericalError;
    int iterations = 0;
    Real final_residual = 0.0;
    std::string error_message;
};

// Linear solver wrapper using Eigen SparseLU
class LinearSolver {
public:
    LinearSolver() = default;

    // Analyze sparsity pattern (call once when structure is known)
    void analyze_pattern(const SparseMatrix& A);

    // Factorize matrix (call when values change)
    bool factorize(const SparseMatrix& A);

    // Solve Ax = b using previously factorized A
    LinearSolveResult solve(const Vector& b);

    // Solve Ax = b (factorize and solve in one step)
    LinearSolveResult solve(const SparseMatrix& A, const Vector& b);

    // Check if matrix is singular
    bool is_singular() const { return is_singular_; }

private:
    Eigen::SparseLU<SparseMatrix> solver_;
    bool pattern_analyzed_ = false;
    bool factorized_ = false;
    bool is_singular_ = false;
};

// Newton-Raphson solver for nonlinear systems
class NewtonSolver {
public:
    struct Options {
        int max_iterations;
        Real abstol;
        Real reltol;
        Real damping;
        bool auto_damping;

        Options()
            : max_iterations(50)
            , abstol(1e-12)
            , reltol(1e-3)
            , damping(1.0)
            , auto_damping(true) {}
    };

    explicit NewtonSolver(const Options& opts = Options()) : options_(opts) {}

    // Solve F(x) = 0 given:
    // - Initial guess x0
    // - Function to compute F(x) and Jacobian J(x)
    // Returns the solution
    using SystemFunction = std::function<void(const Vector& x, Vector& f, SparseMatrix& J)>;
    NewtonResult solve(const Vector& x0, SystemFunction system_func);

    const Options& options() const { return options_; }
    void set_options(const Options& opts) { options_ = opts; }

private:
    Options options_;
    LinearSolver linear_solver_;
};

}  // namespace pulsim
