#pragma once

#include "spicelab/circuit.hpp"
#include "spicelab/mna.hpp"
#include "spicelab/solver.hpp"
#include "spicelab/types.hpp"
#include <functional>

namespace spicelab {

// Callback for streaming results during simulation
using SimulationCallback = std::function<void(Real time, const Vector& state)>;

// Main simulation engine
class Simulator {
public:
    explicit Simulator(const Circuit& circuit, const SimulationOptions& options = {});

    // Run DC operating point analysis
    NewtonResult dc_operating_point();

    // Run transient simulation
    SimulationResult run_transient();

    // Run transient with streaming callback
    SimulationResult run_transient(SimulationCallback callback);

    // Access the circuit
    const Circuit& circuit() const { return circuit_; }

    // Access options
    const SimulationOptions& options() const { return options_; }
    void set_options(const SimulationOptions& options) { options_ = options; }

private:
    // Single timestep of transient simulation
    NewtonResult step(Real time, Real dt, const Vector& x_prev);

    // Build system function for Newton solver
    void build_system(const Vector& x, Vector& f, SparseMatrix& J,
                     Real time, Real dt, const Vector& x_prev);

    const Circuit& circuit_;
    SimulationOptions options_;
    MNAAssembler assembler_;
    NewtonSolver newton_solver_;

    // Cached matrices for reuse
    SparseMatrix G_;  // Conductance matrix
    Vector b_;        // RHS vector
};

// Convenience function for quick simulation
SimulationResult simulate(const Circuit& circuit, const SimulationOptions& options = {});

}  // namespace spicelab
