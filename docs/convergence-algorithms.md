# Convergence Algorithms API Reference

This document describes the advanced convergence algorithms and options available in PulsimCore v1 API. These algorithms improve DC operating point convergence and transient simulation reliability, particularly for power electronics circuits with nonlinear devices.

## Table of Contents

1. [Overview](#overview)
2. [Newton Solver Options](#newton-solver-options)
3. [DC Convergence Strategies](#dc-convergence-strategies)
4. [GMIN Stepping](#gmin-stepping)
5. [Source Stepping](#source-stepping)
6. [Pseudo-Transient Continuation](#pseudo-transient-continuation)
7. [Timestep Control](#timestep-control)
8. [Richardson LTE Estimation](#richardson-lte-estimation)
9. [Linear Solver Configuration](#linear-solver-configuration)
10. [Profiling Utilities](#profiling-utilities)

---

## Overview

PulsimCore provides a multi-strategy approach to circuit convergence:

1. **Newton-Raphson** with voltage limiting - fast when circuits are well-behaved
2. **GMIN Stepping** - adds decreasing conductance to stabilize floating nodes
3. **Source Stepping** - gradually ramps up source values from zero
4. **Pseudo-Transient** - uses transient simulation to find DC solution

The solver automatically tries strategies in sequence until convergence is achieved.

---

## Newton Solver Options

### `NewtonOptions` Structure

```cpp
#include "pulsim/v1/solver.hpp"

namespace pulsim::v1 {

struct NewtonOptions {
    // Convergence tolerances
    Real abs_tol = 1e-12;         // Absolute tolerance for residual
    Real rel_tol = 1e-6;          // Relative tolerance for solution change
    Real v_tol = 1e-6;            // Voltage tolerance (V)
    Real i_tol = 1e-12;           // Current tolerance (A)

    // Iteration limits
    int max_iterations = 50;       // Maximum Newton iterations
    int min_iterations = 1;        // Minimum iterations before checking

    // Voltage limiting
    bool enable_limiting = true;   // Enable voltage/current limiting
    Real max_voltage_step = 0.5;   // Max voltage change per iteration (V)
    Real max_current_step = 1e-3;  // Max current change per iteration (A)

    // Damping
    bool enable_damping = false;   // Enable Newton damping
    Real damping_factor = 0.7;     // Damping factor (0 < d <= 1)
    Real min_damping = 0.1;        // Minimum damping factor

    // Logging
    bool verbose = false;          // Print iteration progress
};

}
```

### Usage Example

```cpp
#include "pulsim/v1/solver.hpp"
using namespace pulsim::v1;

NewtonOptions opts;
opts.max_iterations = 100;
opts.enable_limiting = true;
opts.max_voltage_step = 0.3;  // More conservative for stiff circuits
opts.abs_tol = 1e-10;

NewtonSolver<EnhancedSparseLUPolicy> solver(opts);
auto result = solver.solve(A, b, x0);

if (result.converged) {
    std::cout << "Converged in " << result.iterations << " iterations\n";
}
```

---

## DC Convergence Strategies

### `DCStrategy` Enum

```cpp
enum class DCStrategy {
    Newton,              // Direct Newton-Raphson
    GminStepping,        // GMIN stepping algorithm
    SourceStepping,      // Source ramping algorithm
    PseudoTransient,     // Pseudo-transient continuation
    Auto                 // Try strategies in sequence (recommended)
};
```

### `DCConvergenceConfig` Structure

```cpp
struct DCConvergenceConfig {
    DCStrategy strategy = DCStrategy::Auto;

    // Newton options
    NewtonOptions newton;

    // GMIN stepping
    GminConfig gmin;

    // Source stepping
    SourceSteppingConfig source;

    // Pseudo-transient
    PseudoTransientConfig pseudo_transient;

    // Strategy sequence for Auto mode
    std::vector<DCStrategy> strategy_order = {
        DCStrategy::Newton,
        DCStrategy::GminStepping,
        DCStrategy::SourceStepping,
        DCStrategy::PseudoTransient
    };
};
```

### `DCConvergenceSolver` Class

```cpp
class DCConvergenceSolver {
public:
    explicit DCConvergenceSolver(const DCConvergenceConfig& config);

    // Solve for DC operating point
    DCResult solve(
        SolveFunction fn,           // Returns (A, b) for given x
        const Vector& x0,           // Initial guess
        Index num_nodes             // Number of nodes
    );

    // Get last successful strategy
    DCStrategy last_strategy() const;

    // Get iteration count for each strategy
    const std::map<DCStrategy, int>& strategy_iterations() const;
};
```

---

## GMIN Stepping

GMIN stepping adds a small conductance to all nodes, then gradually reduces it. This stabilizes floating nodes and ill-conditioned matrices.

### `GminConfig` Structure

```cpp
struct GminConfig {
    Real initial_gmin = 1e-3;   // Starting GMIN value (S)
    Real final_gmin = 1e-12;    // Final GMIN floor (S)
    Real reduction_factor = 10.0; // GMIN reduction per step
    int max_steps = 10;          // Maximum GMIN steps
    bool apply_to_all = true;    // Apply to all nodes vs just floating
};
```

### `GminStepping` Class

```cpp
class GminStepping {
public:
    explicit GminStepping(const GminConfig& config = {});

    // Apply current GMIN to matrix diagonal
    void apply_gmin(SparseMatrix& A, Index num_nodes) const;

    // Advance to next smaller GMIN
    bool advance();

    // Check if at minimum GMIN
    bool is_complete() const;

    // Get current GMIN value
    Real current_gmin() const;

    // Reset to initial GMIN
    void reset();

    // Execute full GMIN stepping solve
    std::optional<Vector> execute(
        SolveFunction fn,
        const Vector& x0,
        Index num_nodes
    );
};
```

### How GMIN Stepping Works

1. Start with large GMIN (e.g., 1e-3 S)
2. Add GMIN to diagonal: `A[i,i] += gmin` for all voltage nodes
3. Solve Newton until convergence
4. Reduce GMIN by factor of 10
5. Use previous solution as initial guess
6. Repeat until GMIN reaches minimum (1e-12 S)
7. Final solve without GMIN to verify

---

## Source Stepping

Source stepping gradually ramps independent sources from zero to full value. This provides a smooth path from trivial (zero) solution to actual DC point.

### `SourceSteppingConfig` Structure

```cpp
struct SourceSteppingConfig {
    Real initial_scale = 0.0;     // Starting source scale
    Real final_scale = 1.0;       // Final source scale (100%)
    Real step_size = 0.1;         // Scale increment per step
    Real min_step = 0.01;         // Minimum step size
    int max_steps = 20;           // Maximum stepping iterations
    bool adaptive = true;         // Adjust step size based on convergence
};
```

### `SourceStepping` Class

```cpp
class SourceStepping {
public:
    explicit SourceStepping(const SourceSteppingConfig& config = {});

    // Get current source scale factor
    Real current_scale() const;

    // Advance to next scale level
    bool advance();

    // Reduce step size (on convergence failure)
    void reduce_step();

    // Check if at full scale
    bool is_complete() const;

    // Reset to initial scale
    void reset();

    // Execute full source stepping solve
    using ScaledSolveFunction = std::function<
        std::optional<Vector>(const Vector& x, Real scale)
    >;

    std::optional<Vector> execute(
        ScaledSolveFunction fn,
        const Vector& x0,
        Index num_nodes
    );
};
```

---

## Pseudo-Transient Continuation

When other methods fail, pseudo-transient runs a short transient simulation to find the DC operating point. This is the most robust method but also slowest.

### `PseudoTransientConfig` Structure

```cpp
struct PseudoTransientConfig {
    Real duration = 10e-3;        // Transient duration (s)
    Real initial_dt = 1e-9;       // Initial timestep
    Real max_dt = 1e-6;           // Maximum timestep
    Real dc_tolerance = 1e-6;     // Steady-state detection tolerance
    int steady_state_count = 10;  // Steps below tolerance for SS
    bool use_gear2 = true;        // Use Gear-2 integration
};
```

### `PseudoTransientContinuation` Class

```cpp
class PseudoTransientContinuation {
public:
    explicit PseudoTransientContinuation(
        const PseudoTransientConfig& config = {}
    );

    // Execute pseudo-transient solve
    std::optional<Vector> execute(
        TransientSolveFunction fn,
        const Vector& x0,
        Index num_nodes
    );

    // Get final simulation time
    Real final_time() const;

    // Get number of timesteps taken
    int timestep_count() const;
};
```

---

## Timestep Control

### `AdvancedTimestepConfig` Structure

```cpp
struct AdvancedTimestepConfig {
    // LTE-based control
    Real lte_target = 1e-5;       // Target LTE
    Real lte_safety = 0.9;        // Safety factor

    // Newton-based control
    int target_newton_iters = 5;  // Target iterations per step
    Real newton_weight = 0.3;     // Weight vs LTE control

    // Timestep limits
    Real dt_min = 1e-15;          // Minimum timestep
    Real dt_max = 1e-3;           // Maximum timestep

    // Smoothing
    Real max_growth_rate = 2.0;   // Max dt increase ratio
    Real max_shrink_rate = 0.5;   // Max dt decrease ratio

    // Presets
    static AdvancedTimestepConfig switching_preset();
    static AdvancedTimestepConfig power_electronics_preset();
};
```

### `AdvancedTimestepController` Class

```cpp
class AdvancedTimestepController {
public:
    explicit AdvancedTimestepController(
        const AdvancedTimestepConfig& config = {}
    );

    // Suggest next timestep
    Real suggest_next_dt(
        Real lte,              // Local truncation error estimate
        int newton_iters,      // Newton iterations used
        Real current_dt        // Current timestep
    );

    // Record step result for adaptive smoothing
    void record_step(bool accepted, Real dt, Real lte, int iters);

    // Get statistics
    const TimestepStatistics& statistics() const;

    // Reset controller state
    void reset();
};
```

---

## Richardson LTE Estimation

Richardson extrapolation estimates local truncation error without step doubling.

### `SolutionHistory` Class

```cpp
class SolutionHistory {
public:
    explicit SolutionHistory(std::size_t capacity = 5);

    // Add solution to history
    void push(const Vector& state, Real time, Real dt);

    // Get entry by index (0 = most recent)
    const SolutionHistoryEntry& operator[](std::size_t i) const;

    // Check if sufficient history exists
    bool has_sufficient_history(int order = 2) const;

    // Get current size
    std::size_t size() const;

    // Clear history
    void clear();
};
```

### `RichardsonLTE` Class

```cpp
class RichardsonLTE {
public:
    // Compute scalar LTE estimate (max norm)
    static Real compute(
        const Vector& current,
        const SolutionHistory& history,
        int order = 2
    );

    // Compute per-variable LTE estimates
    static Vector compute_per_variable(
        const Vector& current,
        const SolutionHistory& history,
        int order = 2
    );

    // Compute weighted LTE (voltage/current scaled)
    static Real compute_weighted(
        const Vector& current,
        const SolutionHistory& history,
        const Vector& weights,
        int order = 2
    );
};
```

---

## Linear Solver Configuration

### `LinearSolverConfig` Structure

```cpp
struct LinearSolverConfig {
    enum class Backend {
        Auto,       // Auto-detect best solver
        Eigen,      // Eigen SparseLU
        KLU         // SuiteSparse KLU
    };

    Backend backend = Backend::Auto;

    // Symbolic reuse
    bool reuse_symbolic = true;
    int reanalyze_threshold = 100;  // Reanalyze after N solves

    // Condition monitoring
    bool monitor_condition = false;
    Real condition_warning = 1e10;
    Real condition_error = 1e14;

    // Ordering
    bool use_amd = true;  // AMD ordering for KLU
};
```

### `EnhancedSparseLUPolicy` Class

```cpp
class EnhancedSparseLUPolicy {
public:
    void set_config(const LinearSolverConfig& config);

    // Symbolic analysis
    bool analyze(const SparseMatrix& A);

    // Numeric factorization
    bool factorize(const SparseMatrix& A);

    // Solve Ax = b
    std::optional<Vector> solve(const Vector& b);

    // Check if structure changed
    bool structure_changed(const SparseMatrix& A) const;

    // Get condition estimate
    Real estimated_condition() const;

    // Get factorization statistics
    int symbolic_count() const;
    int numeric_count() const;
};
```

---

## Profiling Utilities

### Enabling Profiling

Define `PULSIM_ENABLE_PROFILING` before including headers:

```cpp
#define PULSIM_ENABLE_PROFILING
#include "pulsim/v1/profiling.hpp"
```

Or add to CMake:

```cmake
target_compile_definitions(your_target PRIVATE PULSIM_ENABLE_PROFILING)
```

### `Timer` Class

```cpp
class Timer {
public:
    void start();
    void stop();

    double elapsed_us() const;  // Microseconds
    double elapsed_ms() const;  // Milliseconds
    double elapsed_s() const;   // Seconds

    bool is_running() const;
};
```

### `Profiler` Singleton

```cpp
class Profiler {
public:
    static Profiler& instance();

    void start(const std::string& name);
    void stop(const std::string& name);

    ProfileStats get_stats(const std::string& name) const;
    std::vector<ProfileStats> all_stats() const;

    std::string report() const;
    void reset();

    static constexpr bool is_enabled();
};
```

### Profiling Macros

```cpp
// Profile a scope
{
    PULSIM_PROFILE_SCOPE("dc_solve");
    // ... code to profile ...
}

// Profile a function
void my_function() {
    PULSIM_PROFILE_FUNCTION();
    // ... function body ...
}

// Manual start/stop
PULSIM_PROFILE_START("operation");
// ... code ...
PULSIM_PROFILE_STOP("operation");

// Operation counting
PULSIM_COUNT_OP("linear_solves");
PULSIM_COUNT_OP_N("newton_iterations", 5);
```

### `SimulationMetrics` Structure

```cpp
struct SimulationMetrics {
    // Timing
    double total_time_ms;
    double dc_time_ms;
    double transient_time_ms;

    // Iteration counts
    std::size_t dc_iterations;
    std::size_t total_newton_iterations;
    std::size_t total_timesteps;
    std::size_t accepted_timesteps;
    std::size_t rejected_timesteps;

    // Linear solver
    std::size_t linear_solves;
    std::size_t symbolic_factorizations;
    std::size_t numeric_factorizations;

    // Convergence
    std::size_t convergence_failures;
    std::size_t gmin_steps_used;
    std::size_t source_steps_used;

    // Derived metrics
    double avg_newton_per_step() const;
    double timestep_acceptance_rate() const;
    double steps_per_second() const;

    std::string report() const;
};
```

---

## Complete Example

```cpp
#include "pulsim/v1/core.hpp"
#include "pulsim/v1/convergence_aids.hpp"
#include "pulsim/v1/integration.hpp"

using namespace pulsim::v1;

int main() {
    // Configure DC solver
    DCConvergenceConfig dc_config;
    dc_config.strategy = DCStrategy::Auto;
    dc_config.newton.max_iterations = 100;
    dc_config.newton.enable_limiting = true;
    dc_config.gmin.initial_gmin = 1e-4;

    DCConvergenceSolver dc_solver(dc_config);

    // Define circuit solve function
    auto solve_fn = [&](const Vector& x) -> std::pair<SparseMatrix, Vector> {
        // Build and return MNA system for given state x
        // ...
    };

    // Solve for DC operating point
    Vector x0 = Vector::Zero(num_nodes);
    auto result = dc_solver.solve(solve_fn, x0, num_nodes);

    if (result.converged) {
        std::cout << "DC converged using "
                  << strategy_name(dc_solver.last_strategy()) << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
    }

    // Configure timestep controller for transient
    auto ts_config = AdvancedTimestepConfig::power_electronics_preset();
    AdvancedTimestepController controller(ts_config);

    // Solution history for LTE estimation
    SolutionHistory history(5);

    // Transient simulation loop
    Real t = 0, dt = 1e-9;
    while (t < t_end) {
        // ... solve timestep ...

        history.push(x, t, dt);

        Real lte = RichardsonLTE::compute(x, history, 2);
        dt = controller.suggest_next_dt(lte, newton_iters, dt);

        t += dt;
    }

    return 0;
}
```

---

## See Also

- [Convergence Tuning Guide](convergence-tuning-guide.md) - Practical tuning advice
- [Device Models](device-models.md) - Device model parameters
- [User Guide](user-guide.md) - General usage instructions
