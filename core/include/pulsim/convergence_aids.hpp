#pragma once

#include "pulsim/types.hpp"
#include "pulsim/solver.hpp"
#include <functional>
#include <vector>

namespace pulsim {

// =============================================================================
// Gmin Stepping - Add conductance from each node to ground for convergence
// =============================================================================

struct GminSteppingOptions {
    Real gmin_initial;      // Initial Gmin value (large, e.g., 1e-2)
    Real gmin_final;        // Target Gmin value (small, e.g., 1e-12)
    Real reduction_factor;  // Factor to reduce Gmin each step (e.g., 10.0)
    int max_steps;          // Maximum number of Gmin steps

    GminSteppingOptions()
        : gmin_initial(1e-2)
        , gmin_final(1e-12)
        , reduction_factor(10.0)
        , max_steps(20) {}
};

// Gmin stepping convergence aid
// Adds a small conductance from each node to ground and gradually reduces it
class GminStepping {
public:
    explicit GminStepping(const GminSteppingOptions& opts = GminSteppingOptions())
        : options_(opts) {}

    // Add Gmin to system matrix
    // G becomes G + gmin * I (where I is identity for node equations)
    void add_gmin(SparseMatrix& G, Index num_nodes, Real gmin);

    // Solve with Gmin stepping
    // system_func: Function that builds (G, b) for the system G*x = b
    // jacobian_func: Function that returns Jacobian for Newton iteration
    using BuildSystemFunc = std::function<void(SparseMatrix& G, Vector& b)>;
    using JacobianFunc = std::function<void(const Vector& x, Vector& f, SparseMatrix& J)>;

    NewtonResult solve(Index num_nodes, BuildSystemFunc build_func,
                       JacobianFunc jacobian_func, const Vector& x0);

    const GminSteppingOptions& options() const { return options_; }
    void set_options(const GminSteppingOptions& opts) { options_ = opts; }

    // Statistics
    int gmin_steps_used() const { return gmin_steps_used_; }
    Real final_gmin() const { return final_gmin_; }

private:
    GminSteppingOptions options_;
    int gmin_steps_used_ = 0;
    Real final_gmin_ = 0.0;
};

// =============================================================================
// Source Stepping - Gradually ramp sources from 0 to nominal value
// =============================================================================

struct SourceSteppingOptions {
    Real factor_initial;    // Initial source scaling (e.g., 0.1)
    Real factor_increment;  // Increment per step (e.g., 0.1)
    int max_steps;          // Maximum number of source steps

    SourceSteppingOptions()
        : factor_initial(0.1)
        , factor_increment(0.1)
        , max_steps(20) {}
};

// Source stepping convergence aid
// Gradually increases source values from a fraction to full value
class SourceStepping {
public:
    explicit SourceStepping(const SourceSteppingOptions& opts = SourceSteppingOptions())
        : options_(opts) {}

    // Solve with source stepping
    // system_func: Function that builds (G, b, factor) where b is scaled by factor
    using ScaledSystemFunc = std::function<void(SparseMatrix& G, Vector& b, Real factor)>;
    using ScaledJacobianFunc = std::function<void(Real factor, const Vector& x, Vector& f, SparseMatrix& J)>;

    NewtonResult solve(ScaledSystemFunc build_func, ScaledJacobianFunc jacobian_func,
                       const Vector& x0);

    const SourceSteppingOptions& options() const { return options_; }
    void set_options(const SourceSteppingOptions& opts) { options_ = opts; }

    // Statistics
    int source_steps_used() const { return source_steps_used_; }
    Real final_factor() const { return final_factor_; }

private:
    SourceSteppingOptions options_;
    int source_steps_used_ = 0;
    Real final_factor_ = 0.0;
};

// =============================================================================
// Pseudo-Transient Continuation - Add C*(dx/dt) term for convergence
// =============================================================================

struct PseudoTransientOptions {
    Real tau_initial;       // Initial pseudo time constant (large, e.g., 1e-3)
    Real tau_final;         // Final pseudo time constant (small, e.g., 1e-15)
    Real tau_factor;        // Factor to reduce tau each iteration (e.g., 10.0)
    int max_iterations;     // Maximum total iterations
    int iterations_per_tau; // Newton iterations before reducing tau

    PseudoTransientOptions()
        : tau_initial(1e-3)
        , tau_final(1e-15)
        , tau_factor(10.0)
        , max_iterations(500)
        , iterations_per_tau(5) {}
};

// Pseudo-transient continuation
// Adds a pseudo time derivative term: C/tau * (x - x_prev) = f(x)
// This adds damping that helps convergence, then tau -> 0 recovers DC solution
class PseudoTransientContinuation {
public:
    explicit PseudoTransientContinuation(const PseudoTransientOptions& opts = PseudoTransientOptions())
        : options_(opts) {}

    // Solve using pseudo-transient continuation
    // system_func: Function that computes f(x) and Jacobian J
    using SystemFunc = std::function<void(const Vector& x, Vector& f, SparseMatrix& J)>;

    // capacitance_func: Function that returns pseudo-capacitance for each variable
    // (typically 1.0 for voltage nodes, inductance for current branches)
    using CapacitanceFunc = std::function<Vector(Index n)>;

    NewtonResult solve(SystemFunc system_func, CapacitanceFunc cap_func, const Vector& x0);

    // Simplified solve with uniform capacitance
    NewtonResult solve(SystemFunc system_func, const Vector& x0, Real uniform_cap = 1e-9);

    const PseudoTransientOptions& options() const { return options_; }
    void set_options(const PseudoTransientOptions& opts) { options_ = opts; }

    // Statistics
    int total_iterations() const { return total_iterations_; }
    int tau_reductions() const { return tau_reductions_; }
    Real final_tau() const { return final_tau_; }

private:
    PseudoTransientOptions options_;
    int total_iterations_ = 0;
    int tau_reductions_ = 0;
    Real final_tau_ = 0.0;
};

// =============================================================================
// Combined Convergence Strategy
// =============================================================================

enum class ConvergenceStrategy {
    None,               // Standard Newton only
    GminStepping,       // Gmin stepping
    SourceStepping,     // Source stepping
    PseudoTransient,    // Pseudo-transient continuation
    Auto,               // Try strategies in order until one works
};

struct ConvergenceAidOptions {
    ConvergenceStrategy strategy;
    GminSteppingOptions gmin_options;
    SourceSteppingOptions source_options;
    PseudoTransientOptions ptc_options;

    ConvergenceAidOptions()
        : strategy(ConvergenceStrategy::Auto)
        , gmin_options()
        , source_options()
        , ptc_options() {}
};

}  // namespace pulsim
